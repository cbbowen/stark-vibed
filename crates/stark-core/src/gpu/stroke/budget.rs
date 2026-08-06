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
use stark_shaders::mirror::dynamics as wick;

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
/// also emits a bleed slot per crossing of the bleed cadence — at most one per segment
/// — and the pen-up settle, so a piece plans at most `2 · MAX_STAMPS + 1` slots. At
/// [`UNIFORM_STRIDE`](super::UNIFORM_STRIDE) apiece that is ~2.1 MB, which is why the
/// factor is worth stating and not worth chunking around: making the cut count planned
/// slots would couple `chunk_segments` to the bleed cadence to save a megabyte it does
/// not need.
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
/// How far the tip travels between `wick` passes, in radii (§6.2).
///
/// **The wick keeps a cadence of its own, decoupled from the segment cadence** — why
/// the reach it replaced was badly conditioned, and why the stencil is separable, are
/// in §6.2. The value is not a tolerance: it is the travel one pass of that stencil
/// carries, so **it must track `dynamics.wesl`'s `WICK_HALF / WICK_RATE`** — which the
/// assertion below now states to the compiler rather than to the reader.
///
/// Because variance adds under composition, a stroke gets the same total smoothing
/// whatever the segmentation and whatever the quantum — widening the kernel and firing
/// it less often is an exact trade, not an approximation, which is what makes this a
/// free parameter to spend on cost.
///
/// **It stops at 2 on purpose, and the reason is a cost the variance argument cannot
/// see.** A firing lands at the start of whichever segment its boundary fell in, so its
/// position jitters by up to one segment length, and a kernel carrying more variance per
/// firing amplifies that jitter proportionally. Measured on
/// `a_carried_stroke_is_independent_of_how_the_path_was_cut`, a stencil half-width of
/// 1/2/4/8 gives 2/3/4/5 levels of cut-dependence — and 4 levels is what
/// [`RESERVOIR_EXCHANGE_STEP`] was tightened *away* from, so buying it back to save a
/// wick dispatch is the wrong trade.
pub(super) const WICK_TRAVEL_QUANTUM: f32 = 0.5;

/// The wick's cadence is the shader's stencil, divided by its rate.
///
/// A stencil widened on one side without moving the cadence on the other smooths by
/// the wrong amount per unit travel: it does not crash, it renders subtly wrong. This
/// used to be a runtime test that could only reach `WICK_HALF`, because `WICK_RATE`
/// survives in the shader only as prose — it computes with the baked `WICK_KERNEL` —
/// and the linker strips what no entry point reaches. Both are generated from the
/// *unlinked* source now (§6.10), so all three numbers are real and the relation is
/// checked where it is declared.
const _: () = assert!(
    WICK_TRAVEL_QUANTUM * wick::WICK_RATE == wick::WICK_HALF as f32,
    "the wick's travel quantum and `dynamics.wesl`'s stencil have diverged — the \n     smoothing per unit travel moved with one of them",
);
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
