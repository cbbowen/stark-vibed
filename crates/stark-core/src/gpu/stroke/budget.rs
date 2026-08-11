//! What a stroke is allowed to cost: the cadences the swept-exchange loop runs at,
//! the ceilings on one region-sized piece of it, and the flattening budget those two
//! together buy (§6.2).
//!
//! These are the numbers a person actually tunes, and they are only meaningful
//! against one another — so they live together, with the measurements and the dead
//! ends that fixed each one recorded on the constant itself. [`flatten_tolerance`] is
//! where they are spent: it is the single place a brush's settings become a segment
//! length, which is what makes a live tail and the commit that replaces it cut the
//! same path (§1.3).
//!
//! Nothing here touches the GPU. It is float arithmetic over a [`BrushParams`], which
//! is what lets the segment-budget tests pin it exactly (`segments::tests`).

use crate::document::{BrushDynamics, BrushParams, BrushShape};
/// The numbers `dynamics.wesl` computes with, generated from its own declarations
/// (§6.10) — the bleed stencil's shares. Relations this file has to honour rather
/// than values it may choose, so it reads them from the shader instead of keeping a
/// second copy to drift.
use stark_shaders::mirror::dynamics as shader;

/// The optical depth one full pass of an opaque tip lays over a point — the τ
/// ceiling `assets::build_prefix_tau` clamps to.
///
/// Every exchange in the stamp loop is a rate *per unit optical depth*, because
/// that is the currency the swept integral is denominated in and the only one both
/// sides can agree on (§6.2). But τ ≈ 7 for a single pass, so read
/// literally a `lift` of 0.5 would strip 99% of the canvas in one pass. Dividing
/// the rates through by this makes an axis mean a fraction **per pass of the tip**
/// — hardness-independent, and what a 0..1 knob is expected to mean.
pub(super) const TAU_PER_PASS: f32 = 6.9;
/// Largest region edge (canvas px) the stamp loop composites at once. A stroke that
/// wants more is drawn in as many region-sized *pieces* as it takes
/// ([`chunk_segments`](super::segments::chunk_segments)), so this bounds the loop's
/// transient GPU memory rather than deciding which strokes the loop can draw at all
/// (§6.2).
///
/// At 2048² that is ~67 MB for a piece: colour and aux are both `Rgba16Float`, so
/// each is 2048² × 8 B = 32 MiB. And it really is *per piece* rather than per stroke,
/// because `DynamicsRun::flush` destroys a piece's region as soon as it submits it.
pub(super) const MAX_REGION_DIM: u32 = 2048;
/// Cap on the **segments** one piece dispatches. Reached only by a stroke fine enough
/// to fill a whole region with them, and it cuts a new piece rather than coarsening
/// anything.
///
/// It bounds the stamp uniform buffer, but not one slot per segment: `dynamics_plan`
/// also emits a bleed slot per crossing of the bleed cadence — up to
/// [`MAX_BLEED_FIRES_PER_SEGMENT`] of them — and the pen-up settle, so a piece plans
/// at most `(1 + MAX_BLEED_FIRES_PER_SEGMENT) · MAX_STAMPS + 1` slots. At
/// [`UNIFORM_STRIDE`](super::UNIFORM_STRIDE) apiece that is ~17.8 MB, which is why the
/// factor is worth stating and not worth chunking around: making the cut count planned
/// slots would couple `chunk_segments` to the bleed cadence to save a few megabytes it
/// does not need. Only a bleeding brush pays any of it, and only one whose segments
/// outrun its own cadence pays more than double.
///
/// The figure moved with the cadence: it was ~9.4 MB when a firing carried half a
/// radius, and halving [`BLEED_TRAVEL_QUANTUM`] doubled the fires a segment may
/// contribute and so this bound with it.
pub(super) const MAX_STAMPS: usize = 4096;
/// How far the tool may travel per exchange, as a fraction of the brush radius
/// (§6.2) — which, since the tool exchanges once per *segment*, is simply a cap on the
/// flattened segment length for a dynamics brush (see [`flatten_tolerance`]).
///
/// **Quoted at one transfer rate.** This is the travel for `lift = deposit = 0.95`;
/// [`exchange_travel`] scales it by how fast the brush being drawn actually trades,
/// because that — not the travel — is what the error is first order in. A gentler brush
/// is not being given a tolerance, it is being charged its own price.
///
/// A property of the exchange loop rather than of the tip, so nothing about a shape's
/// coverage mask should change it. What it bounds is the pair of mean-field
/// approximations either side of the transfer — `bake` gives the canvas a reservoir
/// frozen at the segment's entry, `exchange` gives the tool a canvas frozen at the same
/// instant — and halving it halves that error, cleanly, with no knee to sit on.
///
/// Why the error is a visible bug rather than a tolerance, why no reformulation of the
/// pair kernel avoids it, and why the gain from the sliding kernel was banked as
/// accuracy instead of spent here: **§6.2**.
/// `golden_drained_brush_length_independent` is what pins it.
const RESERVOIR_EXCHANGE_STEP: f32 = 0.125;
/// How much travel (in radii) the `bleed` stencil carries per firing (§6.2):
/// **the cadence carries the step, so the segmentation cannot.**
///
/// Firing per segment was measured non-conservative on real input, and the failure is
/// numeric rather than conceptual. A hand that draws slowly is fitted at a control
/// point per pointer sample — the repro's stroke carries 177 knots over 68 px, mean
/// span 0.39 px — and at that cut a texel's per-segment flux is
/// `share · w · Δ ≈ 1e-4` of a height whose f16 ULP is ~4e-3: deep in the regime
/// where every store either snaps the flux away or ratchets a whole ULP, and the
/// nonlinear rebuild of `(premult, op, height)` between segments turns that into a
/// *directional* drift. Measured on the repro: 20 levels of ghost at 176 spans, 2 at
/// 44, bit-exact zero on a uniform coat at any cut — so the arithmetic is right and
/// the quantization regime is the whole defect.
///
/// Keyed on **absolute arc length**, so the
/// firings — and the windows they sweep — are a pure function of the record,
/// independent of how the path was cut (§6.2, live == committed). Each firing is a
/// dedicated **bleed slot** in the dispatch plan (`dynamics::bleed_fires`): a quad
/// whose sweep *is* the window — bent along the path it stands for, since a quarter
/// radius of travel is many tip-widths of it once the pen modulates the radius
/// down — so its exposure is an ordinary,
/// well-conditioned prefix difference over the window's travel, and one firing
/// moves the paint 176 micro-segments would have tried to move — in one step that
/// sits far above the f16 noise floor. The painting segments themselves carry
/// λ_bleed = 0 and take the no-bleed path bit-for-bit.
///
/// **A quarter rather than the half it was**, and the reason is the ladder the stencil
/// became (`dynamics.wesl`, [`BLEED_SHARE_LADDER`](shader::BLEED_SHARE_LADDER)):
/// spreading a firing's shed evenly over the reach instead of loading it onto the
/// longest tap costs second moment — `(T+1)(2T+1)/(6T²)` of what the same share
/// carries out at the reach, 0.40 at eight rungs. Variance adds linearly in the travel,
/// so buying it back is exactly a cadence this much finer, which is what
/// [`BLEED_DIFFUSIVITY`] being *derived* through both then re-establishes: the knob's
/// top is where it was, spent in twice as many, half as long steps. The firings are
/// cheap next to what they are cut into (measured at radius 500, the whole bleed is
/// ~2 ms of a ~25 ms replay) and [`MAX_BLEED_FIRES_PER_SEGMENT`] doubles with it, so
/// no segment is cut shorter than before either.
pub(super) const BLEED_TRAVEL_QUANTUM: f32 = 0.25;
/// How many firings one segment may contribute, so a plan's slot count stays bounded
/// ([`MAX_STAMPS`]).
///
/// A segment crosses the cadence `travel / (BLEED_TRAVEL_QUANTUM · radius)` times, and
/// those two numbers are priced apart on purpose: [`flatten_tolerance`] buys segment
/// length off the brush's **nominal** radius while the cadence is the **modulated**
/// one, so a pen thinning the tip runs the count up without shortening a thing. Sixteen
/// covers a tip down to a quarter of its brush — every ordinary stroke, where a segment
/// at the travel cap crosses four times — and is sixteen rather than eight only because
/// the cadence is twice as fine. What it stands for is unchanged.
///
/// Below that the axis quietly under-delivers, on a tip carrying almost no paint to
/// spread. The alternative is not "diffuse correctly", it is a plan whose size a
/// degenerate stroke chooses, and unbounded memory is the worse failure of the two.
pub(super) const MAX_BLEED_FIRES_PER_SEGMENT: usize = 16;

/// The blend one firing aims to move — the fraction of a texel's difference from a
/// neighbour that crosses at the window's nominal exposure.
///
/// **Not "as much as possible", and that is the point.** The stencil's worst-case
/// eigenvalue is `1 − 8·Σshare·w = 1 − w` (`dynamics.wesl`, `BLEED_SHARE_NEAR`), so a
/// firing driven to `w → 1` annihilates its worst mode instead of damping it: the
/// operator stops being a Laplacian and becomes a hard local average, and consecutive
/// firings stop composing. Half leaves the margin the diffusion model is written
/// against, and costs nothing — the variance a firing is asked for is bought with the
/// *reach* instead, which is quadratic in it and bounded only by geometry.
///
/// It is an aim rather than a law: the reach is an integer texel count and has a
/// ceiling of its own, so [`bleed_stencil`] lands `w` here when it can and spends the
/// difference on the rate when it cannot.
const BLEED_BLEND: f32 = 0.5;
/// The longest tap a firing may take, as a fraction of the tip's radius.
///
/// The bound is the footprint, not stability. A tap landing outside the sweep has
/// `w_n = 0` and carries nothing (`dynamics.wesl`), so a reach approaching the tip's
/// own size is truncated for most of the tip: the delivered diffusivity falls below
/// the asked-for one, and does so *position-dependently*, which is worse than falling
/// short evenly. Half the radius keeps the long tap live over the inner half of the
/// footprint. Past this the honest way to diffuse further is a finer cadence
/// ([`BLEED_TRAVEL_QUANTUM`]) — more steps, not longer ones, exactly as it would be
/// in any explicit diffusion solver.
const BLEED_REACH_MAX: f32 = 0.5;
/// The diffusivity `bleed = 1` asks for, in **radius² per pass of the tip** — the
/// unit that makes the axis mean the same look at every brush size, as
/// [`TAU_PER_PASS`] does for the vertical rates.
///
/// **Derived, not chosen.** It is whatever puts a full-crank firing at both of its
/// ceilings at once: the reach at [`BLEED_REACH_MAX`] and the blend at
/// [`BLEED_BLEND`]. Solving `σ² = 2·D·radius²` per pass against the stencil's own
/// second moment at that reach gives the expression below, so the three constants
/// cannot drift apart — moving either ceiling moves the top of the knob to match, and
/// the knob stays linear in `D` all the way to it.
///
/// A pass of the tip at full crank buys `σ = sqrt(2·D) · radius`, about a fifth of the
/// radius — where the old saturating knob topped out. The ladder's lower moment and
/// the finer cadence it bought move the two factors in opposite directions and leave
/// this within a tenth of where it was, which is the point of deriving it: the look at
/// the top of the knob is a consequence of the ceilings, not a number to be kept.
const BLEED_DIFFUSIVITY: f32 =
    2.0 * BLEED_BLEND * STENCIL_MOMENT_PER_REACH2 * BLEED_REACH_MAX * BLEED_REACH_MAX
        / BLEED_TRAVEL_QUANTUM;
/// `Σ(share·d²)` per unit of reach², in the continuous limit where the ladder's rungs
/// sit exactly at `j·reach/TAPS` — what [`bleed_stencil`] inverts to get a first guess
/// at the reach, and what [`BLEED_DIFFUSIVITY`] is calibrated through. `Σj²/T³` in
/// closed form; it tends to a third as the ladder fills in, against the 1 a single tap
/// out at the reach would carry. The near tap is a constant and drops out of both.
const STENCIL_MOMENT_PER_REACH2: f32 = shader::BLEED_SHARE_LADDER
    * ((shader::BLEED_LADDER_TAPS + 1) * (2 * shader::BLEED_LADDER_TAPS + 1)) as f32
    / (6 * shader::BLEED_LADDER_TAPS * shader::BLEED_LADDER_TAPS) as f32;

/// `Σ(share·d²)` in texels², for the stencil the shader will actually build at this
/// integer reach.
///
/// Exact rather than the continuous form above, down to flooring each rung the way the
/// shader's integer division does — including the rungs that collapse onto one another
/// (and onto the near tap's 1) once a small tip's reach is only a few texels. This is
/// the number the delivered diffusivity is computed against, so an approximation here
/// is a quiet calibration error rather than a rounding one.
fn stencil_moment(reach: i32) -> f32 {
    let per_rung = shader::BLEED_SHARE_LADDER / shader::BLEED_LADDER_TAPS as f32;
    let mut moment = shader::BLEED_SHARE_NEAR;
    for k in 1..=shader::BLEED_LADDER_TAPS {
        let d =
            ((reach * k + shader::BLEED_LADDER_TAPS / 2) / shader::BLEED_LADDER_TAPS).max(1) as f32;
        moment += per_rung * d * d;
    }
    moment
}

/// One bleed firing's stencil: the longest tap in texels, and `λ_bleed` — solved from
/// the diffusivity the axis asks for, over a window of `span` px at this `radius`.
///
/// **The axis is a diffusivity, and this is where it is spent** (§6.2). Variance adds
/// linearly in travel, so a window carries `σ² = D · radius · span` texels² per axis;
/// one firing of the stencil injects `2·w·Σ(share·d²)`. Two unknowns, one equation —
/// [`BLEED_BLEND`] closes it:
///
/// 1. the reach that delivers the variance at the aimed-for blend, rounded to a texel
///    and clamped to [`BLEED_REACH_MAX`];
/// 2. the blend that *this* reach actually needs, which absorbs both the rounding and
///    the clamp exactly, so the delivered `D` tracks the knob smoothly even though the
///    reach steps;
/// 3. the rate that lands the window's nominal exposure on that blend. `w` and `e` are
///    both per firing, so `k = −ln(1 − w) / e_nom` — and the per-texel
///    `w = 1 − exp(−k·e)` the shader computes still fades to zero at the sweep's rim,
///    which is what carries the no-flux wall.
///
/// The blend is capped at 1 rather than at [`BLEED_BLEND`]: once the reach is at its
/// ceiling the only headroom left *is* the rate, and spending it — at the cost of the
/// eigenvalue margin — beats silently under-delivering. Small brushes live here, since
/// a 4 px tip has a 2-texel reach to work with.
pub(super) fn bleed_stencil(bleed: f32, radius: f32, span: f32) -> (f32, f32) {
    // Per pass the tip travels its own diameter and diffuses `σ² = 2·D·radius²`; this
    // window travels `span` of that.
    let sigma2 = BLEED_DIFFUSIVITY * bleed.clamp(0.0, 1.0) * radius * span;
    if sigma2 <= 0.0 {
        // λ = 0 is the identity, and the slot still dispatches: keeping the plan a
        // pure function of the segmentation is worth more than the dispatch it saves.
        return (1.0, 0.0);
    }
    let reach_max = (BLEED_REACH_MAX * radius).round().max(1.0);
    let reach = (sigma2 / (2.0 * BLEED_BLEND * STENCIL_MOMENT_PER_REACH2))
        .sqrt()
        .round()
        .clamp(1.0, reach_max);
    // What the stencil that reach builds can carry, and therefore what share of it
    // this window has to move. Clamped short of 1: at `w = 1` the rate is infinite and
    // the worst mode is annihilated rather than damped.
    let blend = (sigma2 / (2.0 * stencil_moment(reach as i32))).min(BLEED_BLEND_CEILING);
    // The window's exposure at the centreline, in τ: `span / (2 · radius)` of a pass.
    let e_nom = TAU_PER_PASS * span / (2.0 * radius);
    (reach, (1.0 - blend).ln() / e_nom)
}

/// The domain guard on [`bleed_stencil`]'s solve, not a tuning knob: `ln(1 − w)` is
/// `−∞` at 1 and NaN past it, and a NaN λ does not render a wrong picture, it poisons
/// every texel the firing touches.
///
/// As calibrated the solve never reaches it — [`BLEED_DIFFUSIVITY`] is *derived* from
/// the two ceilings, so a full-crank firing lands at [`BLEED_BLEND`], and rounding the
/// reach to a texel only ever moves the blend down (the near and middle taps make the
/// discrete moment exceed the continuous one that sized it). `the_calibration_never_
/// needs_the_blend_ceiling` is what states that, so retuning either ceiling into a
/// regime where this bites fails a test rather than shipping.
const BLEED_BLEND_CEILING: f32 = 0.9;

/// Cap on `radius · |curvature|`: how fat the tip may be relative to the turn it is
/// swept through before the segment goes back to being straight (§6.2).
///
/// Both shaders sweep a curved segment by **unrolling** the annulus about its centre
/// of curvature into the straight travel frame, which treats a canvas point as sliding
/// through the tip frame along a line of constant lateral offset. It does not: the
/// true track is an arc of radius `ρ`, so a point out at the footprint's shoulder is
/// off that line by `≈ r²/2R`, i.e. **`radius · |curvature| / 2` as a fraction of the
/// tip radius**. That is the constant's real job. The annular sector the swept path
/// rasterizes also folds over itself once `radius ≥ |R|`, but that bound (1.0) is five
/// times looser and never the one that bites.
///
/// 0.1 holds the lateral error to 5% of the tip. It was 0.5 — 25% — which the plain
/// swept deposit absorbed (its segments overlap heavily and the error is smooth) but
/// the dynamics loop did not: there the same offset picks the wrong reservoir texel to
/// serve a canvas texel, and because the loop is sequential the error compounds down
/// the stroke into crescent seams at the reservoir cadence, worst where the tool is
/// dragging paint with nothing left to `add` over them.
pub(super) const MAX_TIP_TURN: f32 = 0.1;

/// The flattening budget for a brush (§6.2). The error bounds are
/// brush-independent — sub-pixel position, a small tangent turn, a small attribute
/// step — but a segment is swept with *constant* attributes, so any brush quantity
/// that varies with distance travelled and is applied per segment (rather than
/// recovered per fragment, as the colour-dynamics arc is) needs a length cap too.
pub(crate) fn flatten_tolerance(b: &BrushParams) -> crate::path::FlattenTolerance {
    let mut tol = crate::path::FLATTEN_TOLERANCE;
    // Use a more relaxed tolerance for larger brushes.
    tol.position = tol.position.max(0.01 * b.radius);
    // The `attribute` bound is a step in the **pen's** own units, and every quantity
    // it was sized for used to be linear in one: radius followed pressure directly,
    // so 2% of pressure was 2% of radius. A modulation puts a curve between the two
    // (§6.2), and a steep one turns that 2% into as much as 18% of the parameter —
    // which draws a ramp as a staircase, since a segment sweeps at one value of
    // everything. So the budget is charged the curve's own slope, which is bounded by
    // construction (`document::MIN_BIAS`) precisely so this bill is.
    //
    // Exactly 1 for the unmodulated brush and for every plain linear mapping,
    // including the default pressure → size: those brushes flatten on the budget they
    // always did, to the bit.
    tol.attribute /= b.modulation.max_slope();
    // The tightest arc this tip may be swept along (§6.2). Both the
    // flattener and the segment generator get it from here, so an edge too tight to
    // sweep as an arc is priced as a chord as well as drawn as one.
    tol.max_arc_curvature = MAX_TIP_TURN / b.radius.max(0.5);
    // `drain` used to be bought here, at `0.02 / drain` px per segment — a cap that
    // could dominate everything else (at `drain = 0.02`, one segment per pixel). It is
    // gone because the falloff is no longer a per-segment constant: both paths
    // evaluate it from the fragment's own arc length, so the amount laid is exactly
    // independent of how the path was cut and there is nothing left for a length cap
    // to bound (`generate_segments_in`).
    // The stamp loop exchanges once per segment, so the segment length *is* the step
    // at which the tool reloads and drains — and unlike the canvas side, which the
    // prefix-τ integral makes exact at any length, that step is a plain first-order
    // discretization of a coupled ODE. [`RESERVOIR_EXCHANGE_STEP`] is what keeps it
    // fine enough. The cap also bounds the snapshot scratch, which is sized by the
    // longest segment.
    let d = b.dynamics;
    if d.lift > 0.0 || d.deposit > 0.0 || d.charge > 0.0 || d.bleed > 0.0 {
        tol.max_len = tol.max_len.min((exchange_travel(d) * b.radius).max(0.5));
    }
    tol
}

/// `λ = ln(1 − axis) / TAU_PER_PASS ≤ 0` — the transfer rate an axis becomes in the
/// shader's terms (§6.2), clamped away from −∞ (axis = 1 ⇒ e^{−20} ≈ scraped
/// clean). Dividing by [`TAU_PER_PASS`] is what makes an axis read as a fraction
/// **per pass of the tip** rather than per unit optical depth. Zero is "no
/// transfer".
///
/// The one definition, on purpose: the plan fills every slot's λ lanes from it,
/// and [`exchange_travel`] prices the flattening budget off the same clamp
/// ([`ln_keep`]). The flattener charging exactly the rates the shader will run is
/// what the exchange-step bound rests on — and it used to rest on two closures,
/// here and in the plan, agreeing by comment.
pub(super) fn lambda(axis: f32) -> f32 {
    ln_keep(axis) / TAU_PER_PASS
}

/// `ln(1 − axis) ≤ 0`, clamped away from −∞ — the shared core of [`lambda`] and of
/// the transfer magnitude [`exchange_travel`] prices.
fn ln_keep(axis: f32) -> f32 {
    (1.0 - axis.clamp(0.0, 1.0)).max(1e-9).ln().max(-20.0)
}

/// How far the tool may travel per segment, in radii — [`RESERVOIR_EXCHANGE_STEP`]
/// scaled by how fast *this* brush actually trades.
///
/// The error the step bounds is a first-order splitting error, and what it is first
/// order in is not the travel but the **transfer the segment completes**: the pair
/// relaxes at `k_lift + k_deposit` per unit optical depth, so a segment's progress
/// through it is `(k_lift + k_deposit) · τ · lr`. Holding *that* fixed rather than `lr`
/// is what makes one constant mean the same thing to every brush.
///
/// The rate falls out in closed form. Each axis enters the shader as
/// `λ = ln(1 − axis) / TAU_PER_PASS` ([`lambda`]), so `(k_lift + k_deposit) · τ` is
/// just `−ln((1 − lift)(1 − deposit))` — the `τ` cancels, and there is no calibration
/// hiding in it.
///
/// Two things this fixes, beyond the arithmetic:
///
/// * **`charge` is not a rate.** It sets the load the tool *starts* with, and a brush
///   that charges but neither lifts nor deposits has `k = 0`: `exchange_at` takes its
///   no-trading branch and the only thing reaching the canvas is `add`, which is linear
///   in exposure and therefore exact at any segment length. Such a brush was paying the
///   full cap for a transfer that never happens.
/// * **The old test was a boolean.** Any brush with a non-zero axis was priced as the
///   most extreme one, so a tip that lifts a tenth of a pass cost the same per pixel as
///   a full smear.
///
/// The budget is calibrated so that `lift = deposit = 0.95` — the repro's brush, and
/// about as hard as the transfer gets — comes out at exactly
/// [`RESERVOIR_EXCHANGE_STEP`], leaving every golden that uses it untouched. A gentler
/// brush earns its relaxation and nothing else changes.
///
/// Priced off the brush's own rates, not the modulated ones. A modulation only ever
/// scales an axis down (`document::Modulation`), which lowers the transfer a segment
/// completes and so the error the step bounds — the brush is charged its worst case
/// and every segment of every stroke it draws comes in under it.
fn exchange_travel(d: BrushDynamics) -> f32 {
    // [`ln_keep`] — the very clamp [`lambda`] hands the shader, so the flattener
    // prices the rates it will actually run (an axis at 1.0 is `−∞` otherwise).
    let rate_of = |axis: f32| -ln_keep(axis);
    // `bleed` is deliberately *not* in this sum: it fires on its own travel cadence
    // with the window's exposure ([`BLEED_TRAVEL_QUANTUM`]), so segment length does
    // not set its step and shortening segments buys it nothing.
    let rate = rate_of(d.lift) + rate_of(d.deposit);
    if rate <= 0.0 {
        return MAX_EXCHANGE_TRAVEL;
    }
    // **Only ever a relaxation.** Rates above the reference are left at the reference
    // step rather than priced below it. Partly because the scaling has only been
    // measured across the band where a brush is usable — `lift = 1.0` is clamped to
    // `λ = −20` in the shader anyway (`dynamics.rs`), so past a point the axis stops
    // meaning what the rule reads it as — and partly because this is a *cost* change:
    // clamping here is what makes it incapable of charging any brush more than it
    // already pays, so no setting can regress on either axis.
    (RESERVOIR_EXCHANGE_STEP * EXCHANGE_REFERENCE_RATE / rate)
        .clamp(RESERVOIR_EXCHANGE_STEP, MAX_EXCHANGE_TRAVEL)
}

/// The transfer rate [`RESERVOIR_EXCHANGE_STEP`] is quoted at: `lift = deposit = 0.95`,
/// i.e. `−ln(0.05 · 0.05)`.
const EXCHANGE_REFERENCE_RATE: f32 = 5.991_465;

/// Ceiling on the travel per segment however slowly the brush trades, in radii.
///
/// Not an accuracy bound — a structural one. A segment carries **one** tip orientation
/// and one curvature (§6.6), the snapshot scratch is sized by the longest of them, and
/// the sweep's own arc approximation is only good while the segment is short next to
/// the tip. None of those care how fast paint changes hands.
const MAX_EXCHANGE_TRAVEL: f32 = 1.0;

/// Ceiling on the footprint cell, in texels. [`footprint_cell`]'s own law reaches 10
/// at the 500 px radius cap, so this is headroom against a future cap rather than a
/// number any brush hits today — it exists so a degenerate input cannot ask the cell
/// scratch for a stencil coarser than the shoulder argument was ever measured at.
const FOOTPRINT_CELL_MAX: f32 = 16.0;

/// The **footprint cell** (§6.2): the edge, in canvas texels, of the square over which
/// the coarse deposit may evaluate the exchange laws *once* and apply the result to
/// every texel inside — 1 meaning the exact per-texel kernel and nothing else.
///
/// The bound is the tip's **shoulder** — the width of a round tip's coverage falloff,
/// `3·(1−hardness)·radius` for the `1 − |y|^h` profile family — because that is the
/// finest feature the footprint-domain fields can carry: the prefix-τ differences, the
/// baked reservoir means and the exchange solves the cell hoists are all smooth at the
/// scale the coverage itself varies. A quarter of the shoulder puts at least four
/// cells across the falloff; the `0.02·radius` term keeps the cell a fixed small
/// fraction of the tip where the shoulder is generous. Both constants are the
/// stroke-space march round's, kept because they were *measured* there: the ripple a
/// coarse cell prints stayed at the no-coarsening floor under the shoulder bound
/// (0.62 vs 0.58 levels rms column-mean) and broke it under a radius-only bound
/// (1.04), and a radius-scaled cell over a shoulderless tip was exactly the
/// stroke-end spike regression of 2026-08-07.
///
/// Two properties are load-bearing rather than tuning:
///
/// * **A hard tip earns no coarsening.** `hardness = 1` has no shoulder, so the min
///   is 0 and the cell is 1 — the exact kernel, bit-for-bit, by construction. A
///   `Stamp` mask can be arbitrarily hard, so it is treated as the sharpest case and
///   never coarsened at all.
/// * **A pure function of the brush shape and the segment's radius**, like every
///   other number in this file — a live tail and its commit pick the same cell for
///   the same segment, which is what `preview == committed` (§1.3) needs from it.
///
/// The threshold is 2: a cell must *beat* two texels before the coarse path engages,
/// because below that the hoist pass costs more than the ~4× it saves — which also
/// means the whole bench sweep at radius ≤ 100 (where `0.02·r ≤ 2`) stays on the
/// exact kernel, dispatch for dispatch.
pub(super) fn footprint_cell(shape: &BrushShape, radius: f32) -> u32 {
    let shoulder = match shape {
        BrushShape::Round { hardness } => 3.0 * (1.0 - hardness.clamp(0.0, 1.0)) * radius,
        BrushShape::Stamp(_) => 0.0,
    };
    let cell = (0.02 * radius).min(0.25 * shoulder);
    if cell <= 2.0 {
        1
    } else {
        cell.min(FOOTPRINT_CELL_MAX) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The diffusivity a firing actually delivers, recovered from what
    /// [`bleed_stencil`] hands the shader — the reach it builds its stencil at and the
    /// rate it relaxes with. In radius² per pass, the unit the axis is quoted in.
    ///
    /// This is the shader's own arithmetic read backwards: it blends
    /// `w = 1 − exp(−k·e)` of each neighbour's difference across a stencil of second
    /// moment `Σ(share·d²)`, injecting `σ² = 2·w·Σ` per axis over a window of
    /// `radius · span` — so anything the solve gets wrong shows up here rather than
    /// only on a GPU.
    fn delivered(bleed: f32, radius: f32, span: f32) -> f32 {
        let (reach, lambda) = bleed_stencil(bleed, radius, span);
        let e_nom = TAU_PER_PASS * span / (2.0 * radius);
        let w = 1.0 - (lambda * e_nom).exp();
        2.0 * w * stencil_moment(reach as i32) / (radius * span)
    }

    /// The windows the solve is actually handed: one cadence quantum, at tips from
    /// "the reach has a single texel to work with" up to a full-canvas blender.
    ///
    /// One quantum is not a simplification here — `bleed_fires` emits a firing *per*
    /// crossing precisely so that it is the only span this is ever asked for, since
    /// what a firing can carry is flat in the travel while what a window asks for grows
    /// with it.
    fn cases() -> impl Iterator<Item = (f32, f32)> {
        [1.0f32, 3.0, 8.0, 20.0, 40.0, 100.0]
            .into_iter()
            .map(|r| (r, BLEED_TRAVEL_QUANTUM * r))
    }

    /// **The property the axis exists to have**: `bleed` is a diffusivity, so what it
    /// buys is linear in the knob — at every brush size, and through the reach's
    /// rounding to whole texels.
    ///
    /// Linearity is the whole claim. The axis used to drive only the rate, and the
    /// rate enters through a blend that clips at 1, so the knob's top end delivered
    /// nothing at all: measured on a 40 px tip, all of 0.95 → 1.0 bought ×1.9 in
    /// diffusivity and then stopped. Here the reach carries what the rate cannot, and
    /// the solve re-derives the blend from the *rounded* reach — which is why this can
    /// assert a tight relative error rather than a trend.
    #[test]
    fn the_bleed_axis_delivers_a_diffusivity_linear_in_the_knob() {
        for (radius, span) in cases() {
            let full = delivered(1.0, radius, span);
            for &knob in &[0.05f32, 0.2, 0.4, 0.6, 0.8, 1.0] {
                let got = delivered(knob, radius, span);
                let want = knob * full;
                assert!(
                    (got - want).abs() <= 1e-3 * full,
                    "radius {radius}, span {span}, bleed {knob}: delivered {got}, \
                     linear would be {want}",
                );
            }
        }
    }

    /// …and the constant it is linear *in* is one number, not one per brush. That is
    /// what quoting the reach in radii buys: the same knob is the same look at every
    /// size, the way the tapers are.
    #[test]
    fn the_diffusivity_is_the_same_at_every_brush_size() {
        for (radius, span) in cases() {
            let got = delivered(1.0, radius, span);
            assert!(
                (got - BLEED_DIFFUSIVITY).abs() <= 0.02 * BLEED_DIFFUSIVITY,
                "radius {radius}, span {span}: full crank delivered {got}, not \
                 {BLEED_DIFFUSIVITY}",
            );
        }
    }

    /// [`BLEED_DIFFUSIVITY`] is derived rather than chosen, and this is the claim that
    /// derivation makes: at full crank a firing sits at *both* ceilings at once — the
    /// reach at [`BLEED_REACH_MAX`], the blend at [`BLEED_BLEND`]. Move either and the
    /// top of the knob moves with it; that is what keeps the three from drifting apart.
    ///
    /// Checked on a tip large enough that the reach is not dominated by its rounding
    /// to a whole texel.
    ///
    /// The reach lands on its cap *exactly* — that half of the derivation is algebra,
    /// and cancels. The blend only lands near its aim, and the slack is the ladder:
    /// every rung is an integer texel, so the stencil the shader builds has a slightly
    /// different second moment from the continuous one
    /// ([`STENCIL_MOMENT_PER_REACH2`]) that sized the reach, and `bleed_stencil`
    /// re-derives the blend against the *discrete* moment — which is the whole reason
    /// it re-derives it. A few percent either way is that quantization; a drift of the
    /// kind this test exists to catch would move the top of the knob, not nudge it.
    #[test]
    fn full_crank_lands_on_both_ceilings_at_once() {
        let (radius, span) = (40.0, BLEED_TRAVEL_QUANTUM * 40.0);
        let (reach, lambda) = bleed_stencil(1.0, radius, span);
        assert_eq!(
            reach,
            BLEED_REACH_MAX * radius,
            "the reach is not at its cap"
        );
        let e_nom = TAU_PER_PASS * span / (2.0 * radius);
        let w = 1.0 - (lambda * e_nom).exp();
        assert!(
            (w - BLEED_BLEND).abs() < 0.05,
            "full crank blends {w}, not the {BLEED_BLEND} it aims for",
        );
    }

    /// The guard on the solve's domain stays slack at every setting the axis can be
    /// asked for, so it is a guard and not a second calibration. If retuning a ceiling
    /// makes this fire, the knob has quietly stopped being linear at its top end —
    /// which is the one property the whole reformulation is for.
    #[test]
    fn the_calibration_never_needs_the_blend_ceiling() {
        for (radius, span) in cases() {
            let (reach, lambda) = bleed_stencil(1.0, radius, span);
            let e_nom = TAU_PER_PASS * span / (2.0 * radius);
            let w = 1.0 - (lambda * e_nom).exp();
            assert!(
                w < BLEED_BLEND_CEILING - 1e-3,
                "radius {radius}, span {span}: full crank needs a blend of {w} at \
                 reach {reach}, which the ceiling is clamping",
            );
        }
    }

    /// A brush that does not bleed relaxes nothing: λ = 0 is exactly the identity, and
    /// the reach is left at the 1 the shader's own `max` would impose anyway — so the
    /// slot still dispatches and the plan stays a pure function of the segmentation.
    #[test]
    fn a_zero_axis_solves_to_the_identity() {
        for (radius, span) in cases() {
            assert_eq!(bleed_stencil(0.0, radius, span), (1.0, 0.0));
        }
    }

    /// The reach never outgrows the footprint it diffuses inside. Past this the long
    /// tap leaves the sweep for most of the tip, where it carries nothing and the
    /// delivered diffusivity falls short of the asked-for one unevenly across the
    /// footprint (`BLEED_REACH_MAX`).
    #[test]
    fn the_reach_stays_inside_the_tip() {
        for (radius, span) in cases() {
            let (reach, _) = bleed_stencil(1.0, radius, span);
            assert!(
                reach <= (BLEED_REACH_MAX * radius).round().max(1.0),
                "radius {radius}, span {span}: reach {reach} overruns the footprint",
            );
        }
    }

    /// **Why `bleed_fires` fires per crossing rather than per segment**, stated as the
    /// arithmetic that forced it: what one firing can carry is a property of the
    /// stencil and flat in the travel, while what a window asks for grows with it. So
    /// a window merged across N quanta is clamped back towards `1/N` of the axis.
    ///
    /// Two quanta is not an exotic case — it is half a segment at the travel cap
    /// against a quarter-radius cadence, i.e. an ordinary fast stroke — which is what
    /// makes the
    /// merged form a shortfall on real input rather than a corner. The clamp itself is
    /// the right behaviour for a solve that cannot be satisfied; this pins that it is
    /// still a clamp and not a NaN, and that the loss is the shape the ceiling implies.
    #[test]
    fn a_window_merged_across_quanta_cannot_be_satisfied() {
        let radius = 40.0;
        let quantum = BLEED_TRAVEL_QUANTUM * radius;
        let one = delivered(1.0, radius, quantum);
        for n in [2.0f32, 5.0] {
            let merged = delivered(1.0, radius, quantum * n);
            assert!(
                merged.is_finite() && merged > 0.0,
                "a {n}-quantum window solved to {merged}",
            );
            assert!(
                merged < 0.95 * one,
                "a {n}-quantum window delivered {merged} against the {one} a firing \
                 per crossing gets, so the shortfall this guards has gone away",
            );
        }
    }

    // --- the footprint cell ------------------------------------------------

    /// The property the whole coarse deposit rests on: **a tip with no shoulder earns
    /// no coarsening**, at any size. `hardness = 1` and every `Stamp` mask take the
    /// exact per-texel kernel bit-for-bit — not approximately, structurally: the cell
    /// is 1, so the host never even dispatches the coarse pipelines. This is the
    /// 2026-08-07 stroke-end spike regression, pinned as arithmetic.
    #[test]
    fn a_shoulderless_tip_is_never_coarsened() {
        let stamp = BrushShape::Stamp(crate::assets::AssetId([7u8; 32]));
        for radius in [8.0f32, 100.0, 250.0, 500.0, 4000.0] {
            assert_eq!(
                footprint_cell(&BrushShape::Round { hardness: 1.0 }, radius),
                1
            );
            assert_eq!(footprint_cell(&stamp, radius), 1);
        }
    }

    /// A softer tip is never resolved *finer* than a harder one of the same size —
    /// the law is monotone in the shoulder, so there is no hardness at which
    /// softening a brush makes it more expensive.
    #[test]
    fn a_softer_tip_never_gets_a_finer_cell() {
        for radius in [50.0f32, 250.0, 500.0] {
            let mut last = u32::MAX;
            for h in [0.0f32, 0.25, 0.5, 0.8, 0.95, 0.99, 1.0] {
                let cell = footprint_cell(&BrushShape::Round { hardness: h }, radius);
                assert!(
                    cell <= last,
                    "radius {radius}: hardness {h} got cell {cell}, harder was {last}",
                );
                last = cell;
            }
        }
    }

    /// Where the bench sweep actually lands, pinned so a retune is a deliberate act:
    /// every radius up to 100 floors to the exact kernel (the `0.02·r` term is ≤ 2
    /// there), the 250/500 lines coarsen under it, and a nearly-hard wide tip is
    /// bounded by its shoulder instead.
    #[test]
    fn the_cell_law_lands_where_the_bench_reads_it() {
        let soft = BrushShape::Round { hardness: 0.5 };
        for radius in [8.0f32, 30.0, 100.0] {
            assert_eq!(
                footprint_cell(&soft, radius),
                1,
                "radius {radius} must stay exact"
            );
        }
        assert_eq!(footprint_cell(&soft, 250.0), 5);
        assert_eq!(footprint_cell(&soft, 500.0), 10);
        // Shoulder-bound: at hardness 0.99 a 500 px tip's shoulder is 15 px, so the
        // quarter-shoulder term (3.75) undercuts the 10 the radius term would give.
        assert_eq!(
            footprint_cell(&BrushShape::Round { hardness: 0.99 }, 500.0),
            3
        );
        // At least four cells across the shoulder wherever the shoulder binds.
        for h in [0.9f32, 0.95, 0.99] {
            let shoulder = 3.0 * (1.0 - h) * 500.0;
            let cell = footprint_cell(&BrushShape::Round { hardness: h }, 500.0);
            assert!(
                cell as f32 * 4.0 <= shoulder || cell == 1,
                "hardness {h}: cell {cell} puts fewer than 4 cells across the \
                 {shoulder} px shoulder",
            );
        }
    }
}
