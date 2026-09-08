//! The brush-dynamics path (§6.2): a serial swept-exchange loop that lets a stroke
//! pick paint up off the canvas and put it back down.
//!
//! Sequential, unlike the swept path: what the tip carries into a segment is what the
//! previous segment left on it. It runs entirely on the GPU (no CPU readback, so it
//! works on WebGPU), a per-segment × per-lateral-band reservoir texture standing in
//! for the tip's load.
//!
//! [`plan`] decides what to dispatch and touches no GPU, [`kit`] builds the objects
//! it is dispatched with, and [`run`] records it — checking its working textures out
//! of the stroke-level [`scratch`](crate::gpu::scratch) pool. What is left here is
//! the question asked before any of them: which path a stroke takes at all.

use stark_model::document::{BrushEffect, BrushParams};

use super::budget::{MIN_SEGMENT_LEN, Shortened, fit_len, flatten_budget};

mod bleed;
mod kit;
mod liquify;
mod plan;
mod run;
mod slots;

pub(in crate::gpu::stroke) use bleed::BLEED_TRAVEL_QUANTUM;
pub(super) use kit::{DynamicsKit, build_dynamics_kit};

/// Which path a stroke takes, as [`dynamics_setup`] decides it.
///
/// `Swept` and `TipTooLarge` are kept apart because they are not the same event —
/// the fast path doing its job versus the renderer failing to draw the brush it was
/// given — and only the caller knows how loudly to say so.
pub(super) enum StrokePath {
    /// Run the sequential stamp loop, carrying the dynamics axes that sent it here —
    /// so a stroke on this path is a wet stroke by construction, with nothing left
    /// for the run to re-derive.
    Loop {
        dynamics: stark_model::document::BrushDynamics,
    },
    /// Run the same region machinery with the liquify field's kernels (§6.13): a
    /// displacement field composed over the canvas under the stroke, the base
    /// resampled through it. It needs a region too, so — like the loop — a tip whose
    /// extent alone overflows one cannot run. No payload; the follow rides the
    /// segments.
    Liquify,
    /// The brush manipulates no paint already on the canvas, so the swept deposit
    /// *is* the whole stroke — one pass, no region, nothing given up.
    Swept,
    /// The brush erases (§6.12): the same swept extent, accumulated across the whole
    /// stroke and turned on the base's *visible* opacity instead of laid as paint.
    /// Needs no region, so no tip is too large for it.
    Erase,
    /// The brush manipulates paint, but its tip alone — before any travel is priced
    /// in — wants more than one region, and a region is the one thing pieces cannot
    /// subdivide. The swept deposit draws what it can: the brush's own `add` paint
    /// and none of the manipulation.
    ///
    /// Unreachable from a brush this app built — the frontier is published as
    /// [`max_tip_reach`](super::budget::max_tip_reach) and the frontend clamps to it
    /// — so what arrives here is a record from a peer or another build.
    TipTooLarge,
}

/// Which path a stroke takes and the budget it flattens at — both decided together,
/// because both are answers about the brush alone and every path needs the second.
pub(super) struct StrokePlan {
    pub(super) path: StrokePath,
    pub(super) tol: crate::path::FlattenTolerance,
    /// How far the region floor shortened the segments the brush's own budget
    /// wanted; `None` when the brush's full-length segment already fits. Carried out
    /// so the renderer can report it once per stroke: the loop exchanges once per
    /// segment, so the stamp count multiplies by `wanted / got`.
    pub(super) shortened: Option<Shortened>,
}

/// Which path a brush's strokes take, and the flattening budget if it is the stamp
/// loop.
///
/// **A pure function of the brush**, and must stay one: the answer has to agree
/// across every render of every piece of a stroke and with the commit that replaces
/// them, or a live tail takes the stamp loop while its commit degrades to the swept
/// deposit and the stroke redraws the moment the pointer comes up. Nothing about the
/// piece in hand or the stroke's length may enter. `rise` is the tip's
/// (`tips::ResolvedTip::rise`), content-addressed and so as fixed as the brush is;
/// only a liquify brush's budget reads it (§6.13).
///
/// Size decides nothing: an oversized stroke is drawn one region-sized piece at a
/// time (`chunk_segments`) and an oversized *segment* is shortened until it fits
/// ([`fit_len`]). Only a brush past the floor those cannot get under — the tip's own
/// extent plus a minimal segment — degrades.
pub(super) fn dynamics_setup(b: &BrushParams, rise: f32) -> StrokePlan {
    // The same flattened segments whichever path runs, at the same budget: a long
    // stroke costs more pieces, not coarser geometry, and the swept fallback below
    // draws the very segments the loop would have.
    let (tol, shortened) = flatten_budget(b, rise);
    // The path is the effect's variant (§6.2), never a rate predicate: there is then
    // no number for a piece and its commit to read differently.
    let path = match &b.effect {
        BrushEffect::Erase(_) => StrokePath::Erase,
        BrushEffect::Paint(_) => StrokePath::Swept,
        // The two region effects price their tip alike: one whose extent alone
        // overflows a region runs neither.
        BrushEffect::Wet(_) | BrushEffect::Liquify(_) if fit_len(b) < MIN_SEGMENT_LEN => {
            StrokePath::TipTooLarge
        }
        BrushEffect::Wet(w) => StrokePath::Loop {
            dynamics: w.dynamics,
        },
        BrushEffect::Liquify(_) => StrokePath::Liquify,
    };
    StrokePlan {
        path,
        tol,
        shortened,
    }
}

#[cfg(test)]
mod tests {
    use super::super::budget::{dynamics_len, max_stretch, max_tip_reach};
    use super::*;

    fn brush(size: f32, lift: f32) -> BrushParams {
        BrushParams {
            size,
            effect: stark_model::document::BrushEffect::wet_with(
                [0.0; 3],
                stark_model::document::BrushDynamics {
                    lift,
                    ..Default::default()
                },
            ),
            ..BrushParams::default()
        }
    }

    /// An `Erase` brush takes the erase path at any size: the pass needs no
    /// region, so it has no tip too large for it, and there are no dynamics axes
    /// on it for the loop to gate on (§6.12).
    #[test]
    fn an_erase_brush_takes_the_erase_path_whatever_its_size() {
        let mut b = brush(40.0, 0.0);
        b.effect = stark_model::document::BrushEffect::Erase(Default::default());
        assert!(matches!(dynamics_setup(&b, 0.0).path, StrokePath::Erase));
        // The tip that would be too large for the loop.
        b.size = super::super::budget::MAX_REGION_DIM as f32;
        assert!(matches!(dynamics_setup(&b, 0.0).path, StrokePath::Erase));
    }

    /// A `Liquify` brush takes the warp path (§6.13), and prices its tip against
    /// the region exactly as the loop does: it needs one too, so a tip that alone
    /// overflows degrades the same way — to a swept deposit that, with every rate
    /// at zero, draws nothing at all.
    #[test]
    fn a_liquify_brush_takes_the_warp_path_and_prices_its_tip_like_the_loop() {
        let liquified = |size: f32, strength: f32| {
            let mut b = brush(size, 0.0);
            b.effect =
                stark_model::document::BrushEffect::Liquify(stark_model::document::LiquifyEffect {
                    strength,
                    ..Default::default()
                });
            b
        };
        let rise = super::super::tips::round_rise_of(&liquified(40.0, 1.0));
        let plan = dynamics_setup(&liquified(40.0, 1.0), rise);
        assert!(matches!(plan.path, StrokePath::Liquify));
        // The warp's own step cap (§6.13): the contraction budget, so a
        // full-strength drag flattens at the tip's own rise…
        assert_eq!(
            plan.tol.max_len,
            super::super::budget::liquify_len(&liquified(40.0, 1.0), rise),
            "the flattener must spend the warp's own budget",
        );
        // …and a drag that moves nothing has no step error for a cap to bound:
        // only the region floor remains, and nothing warns of a shortening that
        // never happened.
        let idle = dynamics_setup(&liquified(40.0, 0.0), rise);
        assert!(matches!(idle.path, StrokePath::Liquify));
        assert_eq!(idle.tol.max_len, fit_len(&liquified(40.0, 0.0)));
        assert!(idle.shortened.is_none());
        // The region floor is the loop's: a tip wider than the whole region
        // cannot warp at any segment length.
        let b = liquified(super::super::budget::MAX_REGION_DIM as f32, 1.0);
        assert!(matches!(
            dynamics_setup(&b, rise).path,
            StrokePath::TipTooLarge
        ));
    }

    /// The floor the shortening cannot get under: the tip's own extent plus one
    /// minimal segment. A brush whose tip fits is drawn by the loop however long
    /// the stroke gets and however long its budget wanted each segment; only one
    /// whose tip alone overflows degrades to the swept deposit.
    #[test]
    fn only_a_tip_that_alone_overflows_the_region_degrades() {
        assert!(matches!(
            dynamics_setup(&brush(1.0, 0.5), 0.0).path,
            StrokePath::Loop { .. }
        ));
        // The largest brush the UI offers (`panels::brush::MAX_RADIUS`), at rates
        // gentle enough to earn the fully relaxed segment length.
        assert!(matches!(
            dynamics_setup(&brush(500.0, 0.05), 0.0).path,
            StrokePath::Loop { .. }
        ));
        // A tip wider than the whole region cannot fit at any segment length.
        let b = brush(super::super::budget::MAX_REGION_DIM as f32, 0.5);
        assert!(matches!(
            dynamics_setup(&b, 0.0).path,
            StrokePath::TipTooLarge
        ));
    }

    /// A canonical mask's content lies inside its inscribed disc (`Sweep::reach`),
    /// so a stamp prices exactly as the round tip the 500/2048 calibration was built
    /// for: a gentle full-size stamp runs the loop at the brush's own full budget
    /// rather than overflowing the region through a `√2` corner bound.
    #[test]
    fn a_full_size_stamp_prices_as_the_round_tip_does() {
        let mut b = brush(500.0, 0.05);
        b.shape = stark_model::document::BrushShape::Stamp(stark_model::AssetId([7u8; 32]));
        let plan = dynamics_setup(&b, 0.0);
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
                let plan = dynamics_setup(&brush(size, lift), 0.0);
                assert!(
                    plan.shortened.is_none(),
                    "a round tip at {size} px (lift {lift}) fits uncapped",
                );
                assert_eq!(plan.tol.max_len, dynamics_len(&brush(size, lift)));
            }
        }
    }

    /// The price the plan quotes is the price the budget paid: `shortened` is `Some`
    /// exactly when the region floor bound the segment length, `wanted` is the
    /// brush's own length, `got` is the floor, and the budget spent is whichever
    /// bound.
    #[test]
    fn the_shortening_quoted_is_the_shortening_paid() {
        let mut big = brush(500.0, 0.5);
        big.stretch = max_stretch(&big);
        let brushes = [brush(8.0, 0.5), brush(250.0, 0.95), brush(500.0, 0.05), big];
        let mut saw_both = (false, false);
        for b in &brushes {
            let plan = dynamics_setup(b, 0.0);
            assert!(matches!(plan.path, StrokePath::Loop { .. }));
            let (wanted, fit) = (dynamics_len(b), fit_len(b));
            match &plan.shortened {
                Some(s) => {
                    assert!(
                        fit < wanted,
                        "size {}: shortened with no binding floor",
                        b.size
                    );
                    assert_eq!((s.wanted, s.got), (wanted, fit));
                    assert_eq!(plan.tol.max_len, fit);
                    saw_both.0 = true;
                }
                None => {
                    assert!(
                        fit >= wanted,
                        "size {}: a binding floor went unquoted",
                        b.size
                    );
                    assert_eq!(plan.tol.max_len, wanted);
                    saw_both.1 = true;
                }
            }
        }
        assert_eq!(
            saw_both,
            (true, true),
            "the brushes have to straddle the floor, or this pins nothing",
        );
    }

    /// **[`max_tip_reach`] is exactly the frontier this gate refuses at** — what
    /// makes it a limit an editor can clamp against. Drift either way and the editor
    /// offers a brush that silently loses its dynamics, or withholds one that would
    /// have drawn.
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
                    b.wet_mut().expect("a wet brush").dynamics.bleed = bleed;
                    b.stretch = knob;
                    let reach = size * BrushParams::elongation(knob);
                    let fits = reach <= max_tip_reach(&b);
                    let drawn = matches!(dynamics_setup(&b, 0.0).path, StrokePath::Loop { .. });
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

    /// **What the editor offers is always drawable**: the engine's half of the
    /// bargain `stark-dioxus-frontend` keeps by clamping to [`max_stretch`]
    /// (`state::update_brush`). Take the largest knob this says is available and the
    /// loop runs, which is what makes `TipTooLarge` unreachable from the UI
    /// structurally rather than by a number kept in step by hand.
    ///
    /// Also that the cap is not vacuous: a non-bleeding tip up to 400 px keeps the
    /// *whole* slider. A cap that quietly became "no stretch for anybody" would pass
    /// the frontier test above and fail here. 400 is a loose checked bound — the true
    /// one sits near 492 and moves with `MAX_TEXTURE_DIM_2D` and the tile arithmetic,
    /// so pinning it would fail on every retune.
    #[test]
    #[expect(
        clippy::float_cmp_const,
        reason = "the clamp lands on MAX_STRETCH itself, so the assertion is identity rather than proximity"
    )]
    fn the_offered_stretch_is_always_drawable() {
        for size in [1.0f32, 30.0, 110.0, 250.0, 400.0, 492.0, 500.0] {
            for bleed in [0.0f32, 0.6] {
                let mut b = brush(size, 0.5);
                b.wet_mut().expect("a wet brush").dynamics.bleed = bleed;
                b.stretch = max_stretch(&b);
                assert!(
                    matches!(dynamics_setup(&b, 0.0).path, StrokePath::Loop { .. }),
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
