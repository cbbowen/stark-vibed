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

use crate::document::{BrushDynamics, BrushParams};

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
/// ([`chunk_segments`](super::segments::chunk_segments)), so this bounds the loop's transient GPU memory —
/// at 2048² the colour and aux regions are ~34 MB together — rather than deciding
/// which strokes the loop can draw at all (§6.2).
pub(super) const MAX_REGION_DIM: u32 = 2048;
/// Cap on the segments one piece dispatches, which bounds its stamp uniform buffer.
/// Reached only by a stroke fine enough to fill a whole region with segments, and it
/// cuts a new piece rather than coarsening anything.
pub(super) const MAX_STAMPS: usize = 4096;
/// How far the tool may travel per exchange, as a fraction of the brush radius
/// (§6.2) — which, since the tool now exchanges once per *segment*, is
/// simply a cap on the flattened segment length for a dynamics brush
/// (see [`flatten_tolerance`]).
///
/// **Quoted at one transfer rate.** This is the travel for `lift = deposit = 0.95`;
/// [`exchange_travel`] scales it by how fast the brush being drawn actually trades,
/// because that — not the travel — is what the error is first order in. A gentler
/// brush is not being given a tolerance, it is being charged its own price.
///
/// A property of the exchange loop, not of the tip: it sets how finely the reservoir
/// tracks the evolving canvas, and nothing about a shape's coverage mask should change
/// it. It was once a cadence of its own — the tool reloaded every `spacing·radius`
/// while the canvas was stripped every segment — and the lag between the two is what
/// left a stroke's last footprint short of paint (`dynamics.wesl`).
///
/// **Nothing here is free.** Measured on `golden_drained_brush_length_independent` — a
/// tip that runs dry and then *carries* paint 400px into view, so every visible pixel
/// arrived through the reservoir and the transport error has nowhere to hide — against
/// a reference at 0.03125, and on the green channel (the one red paint moves furthest).
/// The `sliding` column is what ships; `pair` is the closed-pair kernel it replaced, and
/// is kept because the argument below is about the difference between them:
///
/// ```text
///           error vs reference          length-dependence
///   step     pair          sliding      pair         sliding
///   1.0      50 / 25.77    31 / 8.50    15 / 6.57    16 / 4.48
///   0.5      24 / 10.69    12 / 2.57     8 / 3.22     7 / 1.58
///   0.25     12 /  4.31     7 / 1.26     4 / 1.12     4 / 0.85
///   0.125     7 /  2.38     6 / 1.21     2 / 0.53     2 / 0.57
/// ```
///
/// (max / rms levels. The two kernels converge to the *same* answer — their references
/// differ by 2 max / 0.49 rms, which is the 8-bit floor — so the sliding form is a
/// refinement of the pair, not a different brush.)
///
/// The second column pair is what makes it a bug and
/// not a tolerance: the flattener bisects, so a span's segment length depends on the
/// *whole path's* length, and the same visible stretch of stroke therefore renders
/// differently depending on where the pen went afterwards. The error prints as one
/// tip-shaped arc per segment — the tool lifts at a point and lays back down swept, so
/// the smear translates the canvas by exactly one segment length per segment, which is
/// a delay line ringing at the segment cadence. 0.125 is where that falls into the
/// 8-bit quantization noise.
///
/// It is worth knowing why 0.5 looked fine for a while. The goldens could not see it:
/// nearly every one of them paints with the shared `brush()` helper, which sets
/// `drain = 0.0015`, and `drain` used to impose its own `0.02 / drain` = 13.3px cap on
/// segment length. For any tip wider than 13.3px that cap was the tighter of the two,
/// so the goldens rendered at 13.3px segments *whatever this constant said* — a change
/// here moved nothing, and looked free. Only once the drain cap was retired (it is
/// evaluated per fragment now, see [`flatten_tolerance`]) did this become the binding
/// constraint and start deciding pixels. A benchmark or a golden that does not move is
/// evidence about the test, not about the change. For a radius-80 tip that old cap
/// worked out to 0.166, so this value is very close to what actually shipped; the step
/// was never really at 0.5.
///
/// **Four cheaper things were tried and none of them work**, which is worth recording
/// because each looks obvious:
///
/// * *Averaging the canvas along the reservoir texel's track* instead of the single
///   midpoint tap `dynamics.wesl::exchange` takes. Changes the result by less than the
///   8-bit noise floor on both this test and the pointer-sample-density spread it was
///   meant for. The midpoint tap is not the error.
/// * *Sub-stepping the tool's own kernel* over `e/N`. It looks like refinement and is a
///   different model: the tool lifts its share of a canvas held fixed, N times over,
///   while the deposit gives up a single share of `e`, so the halves stop being
///   complements. At a step four times finer than this one it lands 12 levels rms from
///   where the single step converges.
/// * *Baking the post-exchange reservoir* rather than the entering one. Tempting — it
///   scores 5.0 rms at a step of 0.5, better than the honest scheme manages at 0.125 —
///   and it is a leak: the canvas receives a share of a reservoir the tool never gave
///   up. It converges to a *different* answer, 3.2 rms from the true one, and stalls
///   there however fine the step. The good score at 0.5 is discretization error
///   cancelling the bias.
/// * *Matching `BAKE_RES` to the prefix-τ volume's 256.* No effect; the two grids
///   meeting in `deposit`'s ratio are not the problem.
///
/// **There is no fix for this inside the closed-pair model, and that is a theorem
/// rather than a failure to find one.** Write the pair kernel as the transfer matrix
///
/// ```text
///   M(e) = [ keep(e)      dep(e)  ]
///          [ 1−keep(e)  1−dep(e)  ]
/// ```
///
/// whose columns sum to 1 — that column-stochasticity *is* the complementarity, and it
/// is why the transfer conserves. Its eigenvalues are 1 and `exp(−s·e)` with
/// `s = k_lift + k_deposit`, and the stationary split `k_deposit : k_lift` is
/// independent of `e`, so
///
/// ```text
///   M(e/K)^K = M(e)     exactly, for every K, every exposure, every rate pair.
/// ```
///
/// The kernel already composes perfectly under subdivision. So no re-derivation of it,
/// no sub-stepping of it, no K-fold refinement of it can change a single pixel while
/// remaining a closed pair — a product of column-stochastic matrices is the matrix it
/// started from. Subdividing only does something if the *partner* is held fixed across
/// the sub-steps, and that is not a refinement: it is a one-parameter deformation away
/// from the pair model, whose `K → ∞` limit is exactly the sliding kernel below
/// (`keep` runs 0.503 → 0.076 for the smear brush as K goes 1 → ∞).
///
/// So the error is not in the kernel at all, *while the kernel is a closed pair*. It is
/// in the two mean-field approximations either side of it — `bake` gives the canvas a
/// reservoir frozen at the segment's entry, and `exchange` gives the tool a canvas
/// frozen at the same instant — and those are bounded by the segment length and nothing
/// else. Which is what this constant is.
///
/// **What the theorem really argues is for leaving the closed pair**, and that is what
/// `dynamics.wesl::exchange_at` now does: a **sliding** kernel, `keep = exp(−k_lift·e)`
/// and `dep = 1 − exp(−k_dep·e)`, on the grounds that a canvas point does not stay under
/// one reservoir cell for a segment but slides through a stream of them, each pairing
/// lasting an instant — so the pair's saturation at `k_lift/s` is modelling a coupling
/// that is not there. Worth 2–4× at every step in the table above.
///
/// It was recorded here as blocked: the sliding form "gives up the column-stochasticity
/// … so it needs a conserving formulation (bake the *flux*, not the load) before it can
/// land". **That reading was too pessimistic, in two separate ways.** The shares still
/// sum to one identically — `keep + (1−keep)` is a tautology whatever `keep` is, and the
/// tool retains exactly the `1 − dep` the canvas does not receive — so the
/// 39%-of-height failure in `dynamics.wesl`'s header, which came of the two sides
/// solving *different equations*, cannot recur. And each direction balances in the
/// aggregate on its own, with no flux bookkeeping at all: integrate the canvas's gain
/// `∫k_dep·τ(x−p)·R₀(x−p)·exp(−k_dep·τ(x−p)·p)dp` over the canvas, substitute
/// `u = x−p`, and what comes back is exactly `∫R₀(u)(1−exp(−k_dep·τ(u)·lr))du` — the
/// tool's loss, to the last term. The lift direction telescopes the same way under
/// `q = P(1)−P(u)`. The derivation sits on `exchange_at`.
///
/// What *is* left of the concern: the two sides evaluate their exponentials at different
/// **quadratures** of the same exposure (the canvas differences the prefix-τ, the tool
/// takes `τ(u)·lr`), and an exponential that saturates harder turns the same small
/// quadrature disagreement into a larger height disagreement. Real, and it measured as
/// nothing — on the two conservation tests in `tests/dynamics.rs`, the worst lightening
/// of a smeared field goes 50 → 51 levels against a bound of 60, and a 240-sample
/// zig-zag smear's ink growth goes 0.97940 → 0.97938 against a bound of 1.0.
///
/// **The gain is banked as accuracy rather than spent as step size**, deliberately.
/// Sliding at 0.25 would halve the segment count and still beat the shipped pair kernel
/// on absolute error (1.26 vs 2.38 rms) — but its length-dependence, 4 max / 0.85 rms,
/// is the row that was already weighed and rejected once, when the pair kernel sat at
/// 0.25. That is the column a user actually sees, so the answer has not changed.
/// Spending it later is one constant, and the table above is the price list.
const RESERVOIR_EXCHANGE_STEP: f32 = 0.125;
/// How far the tip travels between `wick` passes, in radii (§6.2).
///
/// **The wick keeps a cadence of its own, decoupled from the segment cadence**, and
/// this is it. It used to run once per segment and absorb whatever travel that was by
/// widening the flux's reach — which is well founded about variance and badly
/// conditioned in practice, because a four-point stencil at an integer distance `d`
/// couples only cells of the same parity in `d`. At the reach of exactly 2 that
/// `MAX_EXCHANGE_TRAVEL` produces, the grid's two sublattices decouple entirely: the
/// checkerboard mode has eigenvalue 1 and never decays, and the cell keeps none of its
/// own value. Every brush relaxed to the travel cap ran there. See `dynamics.wesl::wick`
/// for the measured eigenvalues across the range.
///
/// The value is not a tolerance, it is the travel one pass of the stencil can carry.
/// The shader's 1-D binomial `C(2m, m+k)/4^m` has variance `m/2` against a target of
/// `0.5 · WICK_RATE · lr`, so `lr = m / WICK_RATE`. **It must track `dynamics.wesl`'s
/// `WICK_HALF / WICK_RATE`**, which is 2/4.
///
/// Because variance adds under composition, a stroke gets the same total smoothing
/// whatever the segmentation and whatever the quantum — widening the kernel and firing
/// it less often is an exact trade, not an approximation, which is what makes this a
/// free parameter to spend on cost.
///
/// `m = 1` is the tensor `[¼, ½, ¼]²` this started at, at a quantum of 0.25 radii and 8
/// taps per firing. `m = 2` doubles the quantum and, run *separably* as two 1-D passes,
/// costs 8 taps — so per unit of travel the wick does **half the work** for the same
/// dispatch count. (A non-separable 2-D kernel of the same variance would want
/// `(2m+1)² − 1 = 24` taps. Separability is what makes widening pay at all.)
///
/// **It stops at 2 on purpose, and the reason is a cost the variance argument cannot
/// see.** A firing lands at the start of whichever segment its boundary fell in, so its
/// position jitters by up to one segment length, and a kernel carrying more variance per
/// firing amplifies that jitter proportionally. Measured on
/// `a_carried_stroke_is_independent_of_how_the_path_was_cut`, an `m` of 1/2/4/8 gives
/// 2/3/4/5 levels of cut-dependence. `m = 4` would halve the dispatches as well — but 4
/// levels is the row [`RESERVOIR_EXCHANGE_STEP`] was tightened *away* from, and buying
/// it back to save a wick dispatch is the wrong trade.
///
/// **The separability is what the stroke-space march needs**, and is the reason this is
/// shaped as two passes rather than one wider gather — as well as the reason to hold the
/// width here. A march that owns a lateral row runs the along-travel pass inside its own
/// workgroup with no barrier, so only the across-row pass is a dispatch and only *it*
/// wants a wide kernel. Here both axes are dispatches and must share one cadence, so
/// widening now would bank the jitter on both to save on one.
pub(super) const WICK_TRAVEL_QUANTUM: f32 = 0.5;
/// How much travel (in radii) the `bleed` stencil carries per firing (§6.2) — the
/// same cadence pattern as [`WICK_TRAVEL_QUANTUM`], and adopted for the same reason
/// the wick has one: **the cadence carries the step, so the segmentation cannot.**
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
/// Keyed on **absolute arc length**, exactly like the wick's crossings, so the
/// firings — and the windows they sweep — are a pure function of the record,
/// independent of how the path was cut (§6.2, live == committed). Each firing is a
/// dedicated **bleed slot** in the dispatch plan (`dynamics::bleed_fires`): a
/// straight quad whose sweep *is* the window, so its exposure is an ordinary,
/// well-conditioned prefix difference over half a radius of travel, and one firing
/// moves the paint 176 micro-segments would have tried to move — in one step that
/// sits far above the f16 noise floor. The painting segments themselves carry
/// λ_bleed = 0 and take the no-bleed path bit-for-bit.
pub(super) const BLEED_TRAVEL_QUANTUM: f32 = 0.5;
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
/// `λ = ln(1 − axis) / TAU_PER_PASS` (`dynamics.rs`), so `(k_lift + k_deposit) · τ` is
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
    // Mirrors `dynamics.rs`'s own clamp, so the flattener prices the rates the shader
    // will actually run — an axis at 1.0 is `−∞` otherwise.
    let rate_of = |axis: f32| -(1.0 - axis.clamp(0.0, 1.0)).max(1e-9).ln().max(-20.0);
    // `bleed` is deliberately *not* in this sum: it fires on its own travel cadence
    // with the window's exposure ([`BLEED_TRAVEL_QUANTUM`]), so segment length does
    // not set its step and shortening segments buys it nothing — the same reasoning
    // that keeps the wick's quantum out of here.
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
