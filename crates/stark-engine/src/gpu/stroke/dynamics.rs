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

use super::budget::{MIN_SEGMENT_LEN, dynamics_len, fit_len, flatten_tolerance, manipulates_paint};

mod bleed;
mod kit;
mod plan;
mod run;
mod slots;

pub(in crate::gpu::stroke) use bleed::BLEED_TRAVEL_QUANTUM;
pub(super) use kit::{DynamicsKit, build_dynamics_kit};
pub(in crate::gpu::stroke) use run::LoopBrush;

/// Which path a stroke takes, as [`dynamics_setup`] decides it.
///
/// The two swept answers are kept apart because they are not the same event: one is
/// the fast path doing its job, the other is the renderer failing to draw the brush
/// it was given. Only the caller knows how loudly to say so, so the distinction is
/// carried out rather than resolved here.
pub(super) enum StrokePath {
    /// Run the sequential stamp loop, with the axes that sent it here.
    ///
    /// The variant carries what it proved. `dynamics_setup` reads the axes off the
    /// `Paint` effect to decide this arm at all, so a stroke on this path is a paint
    /// stroke by construction — which the run then re-derived twice through a helper
    /// whose `expect` said "the stamp loop draws paint brushes". A value that is in
    /// hand at the decision does not need an assertion at the use.
    Loop {
        dynamics: stark_model::document::BrushDynamics,
    },
    /// The brush manipulates no paint already on the canvas, so the swept deposit
    /// *is* the whole stroke — one pass, no region, nothing given up.
    Swept,
    /// The brush erases (§6.12): the same swept extent, accumulated across the
    /// whole stroke and turned on the base's *visible* opacity instead of laid as
    /// paint. Its own arm because the effect is its own variant
    /// (`BrushEffect::Erase`) — an eraser has no dynamics axes to gate on, and no
    /// region is ever needed, so no tip is too large for it.
    Erase,
    /// The brush manipulates paint, but its tip alone — before any travel is priced
    /// in — wants more than one region, and the region is the one thing pieces
    /// cannot subdivide. The swept deposit draws what it can, which is the brush's
    /// own `add` paint and none of the manipulation.
    ///
    /// Unreachable from a brush this app built: the frontier is published as
    /// [`max_tip_reach`](super::budget::max_tip_reach) and the frontend clamps to
    /// it. What is left is a record from a peer or another build.
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
    // The same flattened segments whichever path runs, at the same budget: a long
    // stroke costs more pieces, not coarser geometry — and the swept fallback below
    // draws the very segments the loop would have.
    let tol = flatten_tolerance(b);
    // The brush's **own** rates, not the modulated ones — and that is sound rather
    // than an oversight the pen could catch out. A modulation is a factor in [0, 1]
    // (`document::Modulation`), so an axis the brush leaves at zero is zero at every
    // point of every stroke it could ever draw, and one it leaves positive is
    // positive *somewhere*. There is no segment this test could be asked about that
    // would answer differently — which is exactly the property the function's
    // contract above needs, and the reason a modulation was built as a multiplier.
    let d = match &b.effect {
        stark_model::document::BrushEffect::Erase(_) => {
            return StrokePlan {
                path: StrokePath::Erase,
                tol,
                shortened: None,
            };
        }
        stark_model::document::BrushEffect::Paint(p) => p.dynamics,
    };
    if !manipulates_paint(&d) {
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
        path: StrokePath::Loop { dynamics: d },
        tol,
        shortened: (fit < wanted).then_some(Shortened { wanted, got: fit }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::budget::{max_stretch, max_tip_reach};
    use super::*;

    fn brush(size: f32, lift: f32) -> BrushParams {
        let mut b = BrushParams {
            size,
            ..BrushParams::default()
        };
        b.paint_mut()
            .expect("the default brush paints")
            .dynamics
            .lift = lift;
        b
    }

    /// An `Erase` brush takes the erase path at any size: the pass needs no
    /// region, so it has no tip too large for it, and there are no dynamics axes
    /// on it for the loop to gate on (§6.12).
    #[test]
    fn an_erase_brush_takes_the_erase_path_whatever_its_size() {
        let mut b = brush(40.0, 0.0);
        b.effect = stark_model::document::BrushEffect::Erase(Default::default());
        assert!(matches!(dynamics_setup(&b).path, StrokePath::Erase));
        // The tip that would be too large for the loop.
        b.size = super::super::budget::MAX_REGION_DIM as f32;
        assert!(matches!(dynamics_setup(&b).path, StrokePath::Erase));
    }

    /// The floor the shortening cannot get under: the tip's own extent plus one
    /// minimal segment. A brush whose tip fits is drawn by the loop however long
    /// the stroke gets and however long its budget wanted each segment; only one
    /// whose tip alone overflows degrades to the swept deposit.
    #[test]
    fn only_a_tip_that_alone_overflows_the_region_degrades() {
        assert!(matches!(
            dynamics_setup(&brush(1.0, 0.5)).path,
            StrokePath::Loop { .. }
        ));
        // The largest brush the UI offers (`panels::brush::MAX_RADIUS`), at rates
        // gentle enough to earn the fully relaxed segment length.
        assert!(matches!(
            dynamics_setup(&brush(500.0, 0.05)).path,
            StrokePath::Loop { .. }
        ));
        // A tip wider than the whole region cannot fit at any segment length.
        let b = brush(super::super::budget::MAX_REGION_DIM as f32, 0.5);
        assert!(matches!(dynamics_setup(&b).path, StrokePath::TipTooLarge));
    }

    /// The 2026-08-23 repro: a gentle full-size stamp earned the relaxed
    /// full-radius segment, overflowed the region by one tile row through the
    /// `√2` corner bound, and lost its dynamics entirely. A canonical mask's
    /// content lies inside its inscribed disc (`Sweep::reach`), so a stamp now
    /// prices exactly as the round tip the 500/2048 calibration was built for:
    /// the loop runs at the brush's own full budget, nothing shortened at all.
    #[test]
    fn a_full_size_stamp_prices_as_the_round_tip_does() {
        let mut b = brush(500.0, 0.05);
        b.shape = stark_model::document::BrushShape::Stamp(stark_model::AssetId([7u8; 32]));
        let plan = dynamics_setup(&b);
        assert!(
            matches!(plan.path, StrokePath::Loop { .. }),
            "the loop must run"
        );
        assert!(
            plan.shortened.is_none(),
            "a stamp's tip costs its radius, not √2 of it — nothing to shorten",
        );
        assert_eq!(plan.tol.max_len, dynamics_len(&b));
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

    /// **[`max_tip_reach`] is exactly the frontier this gate refuses at**, which is
    /// the whole of what makes it a limit an editor can clamp against: a brush built
    /// at the cap draws, and the arithmetic that says so is the same arithmetic
    /// inverted rather than a second copy of it. Drift either way and the editor
    /// either offers a brush that silently loses its dynamics (the 2026-08-23 bug,
    /// one knob over) or withholds one that would have drawn.
    ///
    /// Swept over both knobs and over the bleed, because the reach is a *product*
    /// and the cap has to say the same thing wherever the two put it — including
    /// for a bleeding brush, whose firings buy it a smaller cap of its own.
    #[test]
    fn the_published_reach_limit_is_the_gates_frontier() {
        let mut saw_both = (false, false);
        for size in [1.0f32, 30.0, 111.0, 250.0, 500.0, 900.0] {
            for knob in [0.0f32, 0.25, 0.5, 0.75, BrushParams::MAX_STRETCH] {
                for bleed in [0.0f32, 0.4] {
                    let mut b = brush(size, 0.5);
                    b.paint_mut().expect("a paint brush").dynamics.bleed = bleed;
                    b.stretch = knob;
                    let reach = size * BrushParams::elongation(knob);
                    let fits = reach <= max_tip_reach(&b);
                    let drawn = matches!(dynamics_setup(&b).path, StrokePath::Loop { .. });
                    assert_eq!(
                        drawn,
                        fits,
                        "size {size}, stretch {knob}, bleed {bleed}: a reach of \
                         {reach} against a cap of {} took the wrong path",
                        max_tip_reach(&b),
                    );
                    if fits {
                        saw_both.0 = true
                    } else {
                        saw_both.1 = true
                    }
                }
            }
        }
        assert_eq!(
            saw_both,
            (true, true),
            "the sweep has to straddle the cap, or it pins nothing",
        );
    }

    /// **What the editor offers is always drawable**, at every size and for every
    /// other setting that moves the cap.
    ///
    /// This is the engine's half of the bargain `stark-ui` keeps by clamping to
    /// [`max_stretch`] (`state::update_brush`, and the stretch slider's own top):
    /// take the largest knob this says is available and the loop runs. So the
    /// degradation is unreachable from the UI *structurally* rather than by a
    /// number kept in step by hand, and what is left for `TipTooLarge` is a record
    /// from somewhere else — a peer, or a file built by another build.
    ///
    /// It also pins that the cap is not vacuous: a non-bleeding tip up to 400 px
    /// keeps the *whole* slider, because the knob tops out at an elongation of 8 and
    /// such a tip cannot spend its way past the region however far it is drawn out.
    /// A cap that quietly became "no stretch for anybody" would pass the frontier
    /// test above and fail here.
    ///
    /// 400 is a checked bound rather than the true one, which sits near 492 — the
    /// region holds a reach of ~3936 and the knob can ask for eight times the size.
    /// Stated loosely on purpose: the exact figure moves with `MAX_TEXTURE_DIM_2D`
    /// and the tile arithmetic, and a test that pinned it would fail on every
    /// retune while claiming to be about the slider.
    #[test]
    fn the_offered_stretch_is_always_drawable() {
        for size in [1.0f32, 30.0, 110.0, 250.0, 400.0, 492.0, 500.0] {
            for bleed in [0.0f32, 0.6] {
                let mut b = brush(size, 0.5);
                b.paint_mut().expect("a paint brush").dynamics.bleed = bleed;
                b.stretch = max_stretch(&b);
                assert!(
                    matches!(dynamics_setup(&b).path, StrokePath::Loop { .. }),
                    "size {size}, bleed {bleed}: the editor's top stretch of {} \
                     degrades",
                    b.stretch,
                );
                if size <= 400.0 && bleed == 0.0 {
                    assert_eq!(
                        b.stretch,
                        BrushParams::MAX_STRETCH,
                        "size {size} should keep the whole slider",
                    );
                }
            }
        }
        // …and a large brush really does give something up, so the clamp is doing
        // work rather than being a no-op the UI could have skipped.
        let mut big = brush(500.0, 0.5);
        big.stretch = max_stretch(&big);
        assert!(
            big.stretch < BrushParams::MAX_STRETCH,
            "a 500 px tip cannot keep the whole stretch range",
        );
    }
}
