//! The brush-dynamics path (§6.2): a serial swept-exchange loop that lets
//! a stroke pick paint up off the canvas and put it back down.
//!
//! Where the swept path composes by summing optical depth — and so can draw its
//! segments in any order — this one is *sequential* by nature: what the tip carries
//! into a segment is what the previous segment left on it. The loop runs on the GPU
//! (no CPU readback, so it works on WebGPU) with a per-segment x per-lateral-band
//! reservoir texture standing in for the tip's load.
//!
//! The path is three modules, split by what a maintainer is holding in their head:
//! [`plan`] works out what to dispatch and touches no GPU at all, [`kit`] builds the
//! objects it is dispatched with, and [`run`] records it — checking its working
//! textures out of the stroke-level [`scratch`](super::scratch) pool through the
//! submit scope that owns their release. What is left here is the one question
//! asked before any of them — which path a stroke takes at all.

use stark_model::document::BrushParams;

use super::budget::{MIN_SEGMENT_LEN, dynamics_len, fit_len, flatten_tolerance};

mod bleed;
mod kit;
mod plan;
mod run;
mod slots;

pub(in crate::gpu::stroke) use bleed::BLEED_TRAVEL_QUANTUM;
pub(super) use kit::{DynamicsKit, build_dynamics_kit};

/// fp32, for the same reason the prefix-τ volume is: every fragment reads the baked
/// swept prefix as a *difference* of two prefix sums (§6.2), so f16 would band exactly
/// where the difference is smallest. Shared by the layout that declares the storage
/// texture ([`kit`]) and the run that allocates it ([`run`]).
const BAKE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Which path a stroke takes, as [`dynamics_setup`] decides it.
///
/// The two swept answers are kept apart because they are not the same event: one is
/// the fast path doing its job, the other is the renderer failing to draw the brush
/// it was given. Only the caller knows how loudly to say so, so the distinction is
/// carried out rather than resolved here.
pub(super) enum StrokePath {
    /// Run the sequential stamp loop.
    Loop,
    /// The brush manipulates no paint already on the canvas, so the swept deposit
    /// *is* the whole stroke — one pass, no region, nothing given up.
    Swept,
    /// The brush manipulates paint, but its tip alone — before any travel is priced
    /// in — wants more than one region, and the region is the one thing pieces
    /// cannot subdivide. The swept deposit draws what it can, which is the brush's
    /// own `add` paint and none of the manipulation.
    TipTooLarge,
}

/// Which path a stroke takes and the budget it flattens at — both decided together,
/// because both are answers about the brush alone and every path needs the second.
pub(super) struct StrokePlan {
    pub(super) path: StrokePath,
    pub(super) tol: crate::path::FlattenTolerance,
    /// How far the region floor shortened the segments the brush's own budget
    /// wanted — `None` when the fit cost nothing, which is every brush whose
    /// full-length segment already fits. Carried out so the renderer can say so
    /// once per stroke: the cap is silent geometry, but its price is real — the
    /// loop exchanges once per segment, so the stroke's stamp count multiplies by
    /// `wanted / got`.
    pub(super) shortened: Option<Shortened>,
}

/// The two sides of a binding fit cap, in canvas px — what the brush budgeted per
/// segment ([`dynamics_len`]) and what the region left of it ([`fit_len`]).
pub(super) struct Shortened {
    pub(super) wanted: f32,
    pub(super) got: f32,
}

/// Which path a brush's strokes take, and the flattening budget if it is the stamp
/// loop.
///
/// **A pure function of the brush**, structurally: this answer has to agree across
/// every render of every piece of the stroke and with the commit that eventually
/// replaces them — a live tail that took the stamp loop while the commit degraded
/// to the swept deposit would redraw the stroke the moment the pointer came up.
/// Taking only the brush is the strongest form of that guarantee — there is nothing
/// about the piece in hand, or the stroke's length, for it to disagree over — and
/// it is what lets `render_range` re-ask on every pointer move for free.
///
/// It can read that way because the stroke's *size* decides nothing: an oversized
/// stroke is drawn one region-sized piece at a time (`chunk_segments`) rather than
/// degraded, and an oversized *segment* is shortened until it fits
/// ([`fit_len`]). All that is left is the floor no shortening gets under — the
/// tip's own extent plus a minimal segment — and only a brush past that degrades.
pub(super) fn dynamics_setup(b: &BrushParams) -> StrokePlan {
    let d = b.dynamics;
    // The brush's **own** rates, not the modulated ones — and that is sound rather
    // than an oversight the pen could catch out. A modulation is a factor in [0, 1]
    // (`document::Modulation`), so an axis the brush leaves at zero is zero at every
    // point of every stroke it could ever draw, and one it leaves positive is
    // positive *somewhere*. There is no segment this test could be asked about that
    // would answer differently — which is exactly the property the function's
    // contract above needs, and the reason a modulation was built as a multiplier.
    // The same flattened segments whichever path runs, at the same budget: a long
    // stroke costs more pieces, not coarser geometry — and the swept fallback below
    // draws the very segments the loop would have.
    let tol = flatten_tolerance(b);
    if d.lift <= 0.0 && d.deposit <= 0.0 && d.charge <= 0.0 && d.bleed <= 0.0 {
        return StrokePlan {
            path: StrokePath::Swept,
            tol,
            shortened: None,
        };
    }
    let fit = fit_len(b);
    if fit < MIN_SEGMENT_LEN {
        return StrokePlan {
            path: StrokePath::TipTooLarge,
            tol,
            shortened: None,
        };
    }
    // `flatten_tolerance` has already taken the min; this only names the price so
    // the renderer can quote it.
    let wanted = dynamics_len(b);
    StrokePlan {
        path: StrokePath::Loop,
        tol,
        shortened: (fit < wanted).then_some(Shortened { wanted, got: fit }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brush(size: f32, lift: f32) -> BrushParams {
        let mut b = BrushParams {
            size,
            ..BrushParams::default()
        };
        b.dynamics.lift = lift;
        b
    }

    /// The floor the shortening cannot get under: the tip's own extent plus one
    /// minimal segment. A brush whose tip fits is drawn by the loop however long
    /// the stroke gets and however long its budget wanted each segment; only one
    /// whose tip alone overflows degrades to the swept deposit.
    #[test]
    fn only_a_tip_that_alone_overflows_the_region_degrades() {
        assert!(matches!(
            dynamics_setup(&brush(1.0, 0.5)).path,
            StrokePath::Loop
        ));
        // The largest brush the UI offers (`panels::brush::MAX_RADIUS`), at rates
        // gentle enough to earn the fully relaxed segment length.
        assert!(matches!(
            dynamics_setup(&brush(500.0, 0.05)).path,
            StrokePath::Loop
        ));
        // A tip wider than the whole region cannot fit at any segment length.
        let b = brush(super::super::budget::MAX_REGION_DIM as f32, 0.5);
        assert!(matches!(dynamics_setup(&b).path, StrokePath::TipTooLarge));
    }

    /// The 2026-08-23 repro: a gentle full-size stamp earned the relaxed
    /// full-radius segment, overflowed the region by one tile row, and lost its
    /// dynamics entirely. Now the region floor shortens its segments instead —
    /// still comfortably above the reference exchange step — and the loop runs.
    #[test]
    fn a_full_size_stamp_buys_its_fit_with_shorter_segments() {
        let mut b = brush(500.0, 0.05);
        b.shape = stark_model::document::BrushShape::Stamp(stark_model::AssetId([7u8; 32]));
        let plan = dynamics_setup(&b);
        assert!(matches!(plan.path, StrokePath::Loop), "the loop must run");
        let s = plan.shortened.expect("…and the cap must actually bind");
        assert!(
            s.got < s.wanted && s.got > 0.25 * b.size,
            "shortened to {} px against {} wanted — the fit should cost segments, \
             not approach the exchange-step floor",
            s.got,
            s.wanted,
        );
        assert_eq!(
            plan.tol.max_len, s.got,
            "the budget spends what the fit left"
        );
    }

    /// The cap is exactly a no-op for every brush that already fits: `shortened`
    /// stays `None` and the flattening budget is the brush's own, to the bit — so
    /// no existing stroke re-flattens.
    #[test]
    fn the_fit_cap_costs_a_fitting_brush_nothing() {
        for size in [1.0, 8.0, 100.0, 250.0, 500.0] {
            for lift in [0.05, 0.5, 0.95] {
                let plan = dynamics_setup(&brush(size, lift));
                assert!(
                    plan.shortened.is_none(),
                    "a round tip at {size} px (lift {lift}) fits uncapped",
                );
                assert_eq!(plan.tol.max_len, dynamics_len(&brush(size, lift)));
            }
        }
    }

    /// Stretch multiplies the tip's extent past what any segment shortening can
    /// buy back: past `size · elongation ≈ 880` canvas px the tip alone overflows
    /// the region, and the stroke degrades — the one frontier left (§6.6).
    #[test]
    fn an_extreme_stretch_still_degrades() {
        let mut b = brush(500.0, 0.5);
        b.stretch = 0.875; // elongation 8
        assert!(matches!(dynamics_setup(&b).path, StrokePath::TipTooLarge));
        // …while a moderate stretch pays segments instead of losing its dynamics.
        b.stretch = 0.3; // elongation ≈ 1.43
        let plan = dynamics_setup(&b);
        assert!(matches!(plan.path, StrokePath::Loop));
        assert!(plan.shortened.is_some());
    }
}
