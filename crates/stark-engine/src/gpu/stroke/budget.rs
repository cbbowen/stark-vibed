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

use stark_model::document::{BrushDynamics, BrushParams, BrushShape};

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
/// At 2048² that is ~67 MB for a piece: color and aux are both `Rgba16Float`, so
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

/// Cap on `radius · |curvature|`: how fat the tip may be relative to the turn it is
/// swept through before the segment goes back to being straight (§6.2).
///
/// Both shaders sweep a curved segment by **unrolling** the annulus about its centre
/// of curvature into the straight travel frame, which treats a canvas point as sliding
/// through the tip frame along a line of constant lateral offset. It does not: the
/// true track is an arc of radius `ρ`, so a point out at the extent's shoulder is
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
/// recovered per fragment, as the color-dynamics arc is) needs a length cap too.
pub(crate) fn flatten_tolerance(b: &BrushParams) -> crate::path::FlattenTolerance {
    let mut tol = crate::path::FLATTEN_TOLERANCE;
    // Use a more relaxed tolerance for larger brushes.
    tol.position = tol.position.max(0.01 * b.radius);
    // The `attribute` bound is a step in the **pen's** own units, so it prices a brush
    // quantity correctly only while the two are proportional — 2% of pressure being 2%
    // of radius. A modulation puts a curve between them (§6.2), and a steep one turns
    // that 2% into as much as 18% of the parameter, which draws a ramp as a staircase
    // since a segment sweeps at one value of everything. So the budget is charged the
    // curve's own slope, bounded by construction (`document::MIN_BIAS`) precisely so
    // this bill is.
    //
    // Exactly 1 for the unmodulated brush and for every plain linear mapping,
    // including the default pressure → size, so those brushes are unaffected to the
    // bit.
    tol.attribute /= b.modulation.max_slope();
    // The tightest arc this tip may be swept along (§6.2). Both the
    // flattener and the segment generator get it from here, so an edge too tight to
    // sweep as an arc is priced as a chord as well as drawn as one.
    //
    // Against the tip's **stretched** reach, not its radius (§6.6). Every reason the
    // cap exists is about the extent rather than about the number that names it: the
    // swept sector stays a simple polygon only while the inner rim clears the centre of
    // curvature, and the reservoir's crescent seams are a misplacement measured across
    // the tip. An extent drawn out `s` times reaches `s` times as far, so it may bend
    // `s` times less. The brush's own elongation and not a segment's, for the reason
    // every bound here is stated against `b`: a modulation only ever scales the knob
    // down, so this one bounds them all.
    tol.max_arc_curvature = MAX_TIP_TURN / (b.radius * BrushParams::elongation(b.stretch)).max(0.5);
    // **`drain` is deliberately not bought here.** A `0.02 / drain` px cap per segment
    // dominates everything else (at `drain = 0.02`, one segment per pixel), and it buys
    // nothing: the falloff is not a per-segment constant, since both paths evaluate it
    // from the fragment's own arc length. The amount laid is exactly independent of how
    // the path was cut, so there is nothing for a length cap to bound
    // (`generate_segments_in`).
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
/// what the exchange-step bound rests on, so it cannot rest on two closures — one
/// here, one in the plan — agreeing by comment.
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
/// Two things the pricing gets right that a cruder one would not:
///
/// * **`charge` is not a rate.** It sets the load the tool *starts* with, and a brush
///   that charges but neither lifts nor deposits has `k = 0`: `exchange_at` takes its
///   no-trading branch and the only thing reaching the canvas is `add`, which is linear
///   in exposure and therefore exact at any segment length. Such a brush must not pay
///   the full cap for a transfer that never happens.
/// * **It is continuous in the rates, not a boolean.** Priced on "has a non-zero axis"
///   alone, every brush is charged as the most extreme one, and a tip that lifts a
///   tenth of a pass costs the same per pixel as a full smear.
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

/// Ceiling on the extent cell, in texels. [`extent_cell`]'s own law reaches 10
/// at the 500 px radius cap, so this is headroom against a future cap rather than a
/// number any brush hits today — it exists so a degenerate input cannot ask the cell
/// scratch for a stencil coarser than the shoulder argument was ever measured at.
const EXTENT_CELL_MAX: f32 = 16.0;

/// The **extent cell** (§6.2): the edge, in canvas texels, of the square over which
/// the coarse deposit may evaluate the exchange laws *once* and apply the result to
/// every texel inside — 1 meaning the exact per-texel kernel and nothing else.
///
/// The bound is the tip's **shoulder** — the width of a round tip's coverage falloff,
/// `3·(1−hardness)·radius` for the `1 − |y|^h` profile family — because that is the
/// finest feature the extent-domain fields can carry: the prefix-τ differences, the
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
///
/// The **shoulder** — the width of the tip's coverage falloff per unit radius — is
/// [`shoulder_per_radius`], shared with the taper's subdivision, which leans on the
/// same fact from the other side: a feature narrower than a quarter of the shoulder
/// is one the coverage cannot show.
pub(super) fn extent_cell(shape: &BrushShape, radius: f32) -> u32 {
    let shoulder = shoulder_per_radius(shape) * radius;
    let cell = (0.02 * radius).min(0.25 * shoulder);
    if cell <= 2.0 {
        1
    } else {
        cell.min(EXTENT_CELL_MAX) as u32
    }
}

/// The width of the tip's coverage falloff — its **shoulder** — per unit radius:
/// `3·(1−hardness)` for the round tip's `1 − |y|^h` profile family, and 0 for a
/// `Stamp`, which may be arbitrarily hard and is treated as the sharpest case.
///
/// The one definition, used from both sides of the same fact: features narrower than
/// a fraction of the shoulder are ones the coverage cannot carry. [`extent_cell`]
/// spends that as *coarsening* (the cell the coarse deposit may evaluate at), the
/// taper's subdivision as *smoothness* (the radius step a segment boundary may take
/// without printing, `segments::Taper`).
pub(super) fn shoulder_per_radius(shape: &BrushShape) -> f32 {
    match shape {
        BrushShape::Round { hardness } => 3.0 * (1.0 - hardness.clamp(0.0, 1.0)),
        BrushShape::Stamp(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // --- the extent cell ------------------------------------------------

    /// The property the whole coarse deposit rests on: **a tip with no shoulder earns
    /// no coarsening**, at any size. `hardness = 1` and every `Stamp` mask take the
    /// exact per-texel kernel bit-for-bit — not approximately, structurally: the cell
    /// is 1, so the host never even dispatches the coarse pipelines. This is the
    /// 2026-08-07 stroke-end spike regression, pinned as arithmetic.
    #[test]
    fn a_shoulderless_tip_is_never_coarsened() {
        let stamp = BrushShape::Stamp(stark_model::AssetId([7u8; 32]));
        for radius in [8.0f32, 100.0, 250.0, 500.0, 4000.0] {
            assert_eq!(extent_cell(&BrushShape::Round { hardness: 1.0 }, radius), 1);
            assert_eq!(extent_cell(&stamp, radius), 1);
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
                let cell = extent_cell(&BrushShape::Round { hardness: h }, radius);
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
                extent_cell(&soft, radius),
                1,
                "radius {radius} must stay exact"
            );
        }
        assert_eq!(extent_cell(&soft, 250.0), 5);
        assert_eq!(extent_cell(&soft, 500.0), 10);
        // Shoulder-bound: at hardness 0.99 a 500 px tip's shoulder is 15 px, so the
        // quarter-shoulder term (3.75) undercuts the 10 the radius term would give.
        assert_eq!(extent_cell(&BrushShape::Round { hardness: 0.99 }, 500.0), 3);
        // At least four cells across the shoulder wherever the shoulder binds.
        for h in [0.9f32, 0.95, 0.99] {
            let shoulder = 3.0 * (1.0 - h) * 500.0;
            let cell = extent_cell(&BrushShape::Round { hardness: h }, 500.0);
            assert!(
                cell as f32 * 4.0 <= shoulder || cell == 1,
                "hardness {h}: cell {cell} puts fewer than 4 cells across the \
                 {shoulder} px shoulder",
            );
        }
    }
}
