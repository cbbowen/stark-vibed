//! The **bleed** axis: how far a firing reaches, how hard it relaxes, and how often it
//! fires (§6.2).
//!
//! One model, so one file. The cadence lived in `budget.rs` beside the flattening
//! caps, the stencil solve beside it, and [`bleed_fires`](super::plan::bleed_fires) —
//! which is the other half of the same thing — a directory away in the plan. They are
//! together now: the axis is a **diffusivity**, and everything that decides what that
//! buys is here.
//!
//! Nothing here touches the GPU. It is float arithmetic over a radius and a span, which
//! is what lets the whole calibration be pinned without an adapter (`tests`).

use super::super::budget::TAU_PER_PASS;
/// The numbers `dynamics.wesl` computes with, generated from its own declarations
/// (§6.10) — the bleed stencil's shares. Relations this file has to honour rather
/// than values it may choose, so it reads them from the shader instead of keeping a
/// second copy to drift.
use stark_shaders::mirror::dynamics as shader;

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
pub(in crate::gpu::stroke) const BLEED_TRAVEL_QUANTUM: f32 = 0.25;
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
}
