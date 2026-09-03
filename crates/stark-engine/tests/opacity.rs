//! The paint effect's opacity (§6.2) — [`EraseEffect::opacity`]'s law run in
//! the laying direction.
//!
//! The claims worth pinning are the ones the knob exists for: the dial is quoted
//! in what the eye sees (a saturated stroke at 0.5 covers half — held here as
//! *equality with the half-covered fill*, the erase headline mirrored), the
//! opacity is a ceiling a stroke cannot scrub past, separate strokes compound as
//! glazes do — and the fast path is an optimization, never a semantics: the
//! stamp loop mints the prefix differences of the same capped law, so a whisper
//! of `deposit` routes the stroke without changing the paint, and the `charge`
//! glob is capped by being finite. The piecewise-vs-whole and round-trip
//! obligations ride the corpus (`corpus.rs`, cases `opacity` and
//! `wet_opacity`), which is also what exercises the carried parcel and the
//! carried mint budget a live stroke accumulates below full opacity.

mod common;

use common::*;
use stark_engine::command::DocCommand;
use stark_model::Srgb;
use stark_model::document::{BrushParams, FillOp, ModSource, Modulation, SelectionShape};
use stark_model::geom::Vec2;

const RED: [f32; 3] = [1.0, 0.0, 0.0];

/// A swept-path wash at `opacity`: flow high enough that one pass saturates the
/// parcel over the stroke's core, so a test sampling the core is reading the
/// dial and not a half-built fringe — the erase suite's construction, laying
/// instead of removing.
fn washed(opacity: f32, radius: f32) -> BrushParams {
    let mut b = brush(RED, radius);
    b.paint_mut().expect("a paint brush").flow = 2.5;
    b.effect.set_opacity(opacity);
    b.drain = 0.0;
    b
}

/// Fill the whole canvas with `color` at `opacity` — the stated-coverage
/// reference the headline compares against (`FillOp::opacity`).
fn fill_canvas(engine: &mut stark_engine::Engine, color: [f32; 3], opacity: f32) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::new(
            SelectionShape::rect_from_corners(Vec2::new(-128.0, -128.0), Vec2::new(128.0, 128.0)),
            0.0,
            Srgb::new(color),
            opacity,
        ),
    });
}

/// **The dial is quoted in visible coverage** — the claim the knob exists for
/// (§6.2), stated with no media arithmetic in it: a saturated stroke at 0.5
/// over bare canvas *is* the canvas filled at 0.5. Same latent, same per-unit
/// opacity, and the same height — the scaled parcel is inverted through the
/// slab law into the amount that shows exactly the scaled coverage, which is
/// the arithmetic a fill's stated opacity runs the other way.
///
/// The old per-unit alpha fails this by construction: it thinned the material
/// and let scrubbing build back to full coverage.
#[test]
fn a_saturated_half_opacity_stroke_is_the_half_covered_fill() {
    let Some(mut stroked) = engine_or_skip() else {
        return;
    };
    stroke_with(
        &mut stroked,
        washed(0.5, 24.0),
        &[Vec2::new(-90.0, 0.0), Vec2::new(90.0, 0.0)],
    );
    let stroked = stroked.render_to_image();

    let mut asked = engine_or_skip().expect("the adapter answered once already");
    fill_canvas(&mut asked, RED, 0.5);
    let asked = asked.render_to_image();

    for p in [Vec2::ZERO, Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)] {
        let (s, a) = (texel(&stroked, p), texel(&asked, p));
        assert!(
            apart(s, a) <= 2,
            "at {p:?} the half-opacity stroke {s:?} must be the half-covered \
             fill {a:?} — the dial is not landing in the coverage domain"
        );
    }
}

/// `opacity` is a **ceiling, not a rate**: a stroke worrying the same spot five
/// times leaves what one clean pass leaves. The parcel accumulates across the
/// whole stroke and its coverage saturates at 1, so what lands saturates at the
/// dial (§6.2) — where a per-piece or per-pass scale would compound toward
/// full coverage. This is also the self-crossing claim: the scrub crosses its
/// own trail and must not darken past the cap.
#[test]
fn scrubbing_cannot_pass_the_dial() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // One clean pass across y = −40…
    stroke_with(
        &mut engine,
        washed(0.5, 20.0),
        &[Vec2::new(-80.0, -40.0), Vec2::new(80.0, -40.0)],
    );
    // …and one stroke worrying the same band at y = 40, five crossings.
    let scrub: Vec<Vec2> = (0..5)
        .map(|i| {
            let x = if i % 2 == 0 { -80.0 } else { 80.0 };
            Vec2::new(x, 40.0 + i as f32 * 0.25)
        })
        .collect();
    stroke_with(&mut engine, washed(0.5, 20.0), &scrub);
    let after = engine.render_to_image();

    let once = texel(&after, Vec2::new(0.0, -40.0));
    let scrubbed = texel(&after, Vec2::new(0.0, 40.0));
    assert!(
        apart(once, scrubbed) <= 2,
        "a scrubbed half-opacity stroke must land where one pass lands: one \
         pass {once:?}, scrubbed {scrubbed:?}"
    );
    // The dial did something, or the equality above is two saturated strokes
    // agreeing at full coverage.
    let mut full = engine_or_skip().expect("the adapter answered once already");
    stroke_with(
        &mut full,
        washed(1.0, 20.0),
        &[Vec2::new(-80.0, -40.0), Vec2::new(80.0, -40.0)],
    );
    assert!(
        apart(once, texel(&full.render_to_image(), Vec2::new(0.0, -40.0))) > 20,
        "the half-opacity stroke reads as the full one, so this test measured nothing"
    );
}

/// Separate strokes compound as glazes: two saturated passes at 0.5 cover
/// `1 − (1−½)² = ¾` — held as equality with the fill at 0.75, since each
/// stroke's scaled parcel stacks its mass on the resident paint through the one
/// shared law (§6.1).
#[test]
fn two_half_opacity_strokes_are_the_three_quarter_fill() {
    let Some(mut stroked) = engine_or_skip() else {
        return;
    };
    for _ in 0..2 {
        stroke_with(
            &mut stroked,
            washed(0.5, 20.0),
            &[Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)],
        );
    }
    let stroked = stroked.render_to_image();

    let mut asked = engine_or_skip().expect("the adapter answered once already");
    fill_canvas(&mut asked, RED, 0.75);
    let asked = asked.render_to_image();

    let (s, a) = (texel(&stroked, Vec2::ZERO), texel(&asked, Vec2::ZERO));
    assert!(
        apart(s, a) <= 2,
        "two half-opacity strokes {s:?} must equal the three-quarter fill {a:?}"
    );
}

/// **The fast path is an optimization, not a semantics** (§6.2): nudging
/// `deposit` off zero routes the same brush through the stamp loop, and the
/// paint must not change. Below full opacity that is the capped mint's whole
/// job — the loop lays the prefix differences of the very law the swept
/// integrate applies to its accumulated parcel, so the two renderers agree
/// texel for texel, saturation included.
#[test]
fn a_whisper_of_deposit_does_not_change_the_paint() {
    let whisper = |opacity: f32| {
        let mut b = washed(opacity, 20.0);
        // Off zero so the stroke takes the loop; with nothing lifted and nothing
        // charged, the axis itself moves no paint (`dynamics_setup`).
        b.make_wet().dynamics.deposit = 0.01;
        b
    };
    let path = [Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)];

    let Some(mut looped) = engine_or_skip() else {
        return;
    };
    stroke_with(&mut looped, whisper(0.5), &path);
    let looped = looped.render_to_image();

    let mut swept = engine_or_skip().expect("the adapter answered once already");
    stroke_with(&mut swept, washed(0.5, 20.0), &path);
    let swept = swept.render_to_image();

    for p in [Vec2::ZERO, Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)] {
        let (l, s) = (texel(&looped, p), texel(&swept, p));
        assert!(
            apart(l, s) <= 2,
            "at {p:?} the loop's stroke {l:?} must be the fast path's {s:?} — \
             the whisper of deposit changed the paint"
        );
    }
    // …and the dial is doing something at all, or the agreement above is two
    // full-strength strokes agreeing.
    let mut full = engine_or_skip().expect("the adapter answered once already");
    stroke_with(&mut full, whisper(1.0), &path);
    assert!(
        apart(
            texel(&looped, Vec2::ZERO),
            texel(&full.render_to_image(), Vec2::ZERO)
        ) > 20,
        "the half-opacity stroke reads as the full one, so this test measured nothing"
    );
}

/// The ceiling holds on the loop as it holds on the fast path: a loop-routed
/// stroke worrying one spot five times lands where the fast path's single
/// clean pass lands — the budget lanes saturate the mint at the dial, across
/// segments and across the stroke's own crossings (§6.2).
#[test]
fn scrubbing_the_loop_cannot_pass_the_dial() {
    let mut whisper = washed(0.5, 20.0);
    whisper.make_wet().dynamics.deposit = 0.01;

    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let scrub: Vec<Vec2> = (0..5)
        .map(|i| {
            let x = if i % 2 == 0 { -80.0 } else { 80.0 };
            Vec2::new(x, i as f32 * 0.25)
        })
        .collect();
    stroke_with(&mut engine, whisper, &scrub);
    let scrubbed = engine.render_to_image();

    let mut once = engine_or_skip().expect("the adapter answered once already");
    stroke_with(
        &mut once,
        washed(0.5, 20.0),
        &[Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)],
    );
    let once = once.render_to_image();

    let (s, o) = (texel(&scrubbed, Vec2::ZERO), texel(&once, Vec2::ZERO));
    assert!(
        apart(s, o) <= 2,
        "the loop's scrubbed stroke {s:?} must land where the fast path's one \
         pass lands {o:?}"
    );
}

/// The `charge` glob is minted paint too (§6.2), and a finite source scaled at
/// the mint is its own ceiling: a pre-loaded tool at half opacity delivers what
/// half the glob delivers, everything it can ever deliver being the scaled glob.
#[test]
fn a_charged_glob_at_half_opacity_is_the_half_charge_glob() {
    let glob = |charge: f32, opacity: f32| {
        let mut b = brush(RED, 24.0);
        b.drain = 0.0;
        let d = &mut b.make_wet().dynamics;
        d.add = 0.0;
        d.deposit = 0.6;
        d.charge = charge;
        b.effect.set_opacity(opacity);
        b
    };
    let path = [Vec2::new(-80.0, 0.0), Vec2::new(80.0, 0.0)];

    let Some(mut dialed) = engine_or_skip() else {
        return;
    };
    stroke_with(&mut dialed, glob(0.25, 0.5), &path);
    let dialed = dialed.render_to_image();

    let mut halved = engine_or_skip().expect("the adapter answered once already");
    stroke_with(&mut halved, glob(0.125, 1.0), &path);
    let halved = halved.render_to_image();

    // Sampled where the glob lands, near the start — it runs dry along the way.
    let at = Vec2::new(-70.0, 0.0);
    let (d, h) = (texel(&dialed, at), texel(&halved, at));
    assert!(
        apart(d, h) <= 2,
        "the dialed glob {d:?} must deliver the half glob {h:?}"
    );
}

// ---- the ceiling under the pen (§6.2) ---------------------------------------------

/// [`washed`] with its ceiling under the pen: opacity mapped to pressure,
/// linearly, so the pen's share of the dial *is* its pressure.
fn under_the_pen(opacity: f32, radius: f32) -> BrushParams {
    let mut b = washed(opacity, radius);
    b.paint_mut().expect("a paint brush").modulation.opacity =
        Some(Modulation::linear(ModSource::Pressure));
    b
}

/// **A pen that never moves is the dial.** Mapped to pressure and pressed home
/// throughout, the ceiling lane holds the plain coverage and the stroke is the
/// unmapped one — through the other sweep pipeline, a fourth carried lane and
/// the lane's own inversion, so the agreement is to the lane's f16 rather than
/// to the bit. The mouse's case, since a mouse reports 1: a pressure-mapped
/// preset under one is the preset.
#[test]
fn a_pen_driven_ceiling_pressed_home_is_the_dial() {
    let path = [Vec2::new(-90.0, 0.0), Vec2::new(90.0, 0.0)];
    let Some(mut mapped) = engine_or_skip() else {
        return;
    };
    stroke_with(&mut mapped, under_the_pen(0.5, 24.0), &path);
    let mapped = mapped.render_to_image();

    let mut plain = engine_or_skip().expect("the adapter answered once already");
    stroke_with(&mut plain, washed(0.5, 24.0), &path);
    let plain = plain.render_to_image();

    // The core, and the shoulder — where the coverage is below 1 and the lane's
    // claim is a fraction rather than the saturated 1.
    for p in [
        Vec2::ZERO,
        Vec2::new(-40.0, 0.0),
        Vec2::new(40.0, 0.0),
        Vec2::new(0.0, 20.0),
    ] {
        let (m, u) = (texel(&mapped, p), texel(&plain, p));
        assert!(
            apart(m, u) <= 2,
            "at {p:?} the pressed-home mapped stroke {m:?} must be the dial's {u:?}"
        );
    }
}

/// A straight leg along `y` from `x0` to `x1`, reported every 10 px at one
/// `pressure` — dense enough that the fit holds the pressure along the leg
/// rather than smoothing a neighbouring leg's into it.
fn leg(x0: f32, x1: f32, y: f32, pressure: f32) -> Vec<(Vec2, f32)> {
    let n = 16;
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            (Vec2::new(x0 + (x1 - x0) * t, y), pressure)
        })
        .collect()
}

/// **The larger ceiling wins** (§6.2), one way round: a light pass back over a
/// heavy mark, within one stroke, leaves the mark as it was — the property a
/// pen-driven ceiling has to have in the stroke's own time, because a live
/// stroke that took paint away as it went on would read as an eraser.
///
/// At a dial below 1, where the claim's inversion is exact. At exactly 1 a
/// saturated claim rounds to f16's 1 and the integrate lands the whole parcel —
/// the light pass's mass included, as impasto — which is what the unmodulated
/// brush at dial 1 does too, and is a claim about height rather than the one
/// made here.
#[test]
fn a_light_pass_back_over_a_heavy_mark_leaves_it() {
    // Both legs reported every 10 px, so the fit pins each leg's pressure to
    // its own — and the drop from one to the other happens on an excursion
    // away from the band, many knots from either leg. The fitted pressure is a
    // smoothed spline, and a step in it ripples back along the curve: with the
    // drop at the hairpin itself the heavy leg's factor sat a few percent under
    // 1 for its whole length, which the ceiling lane read faithfully and this
    // test is not about.
    let heavy = leg(-80.0, 80.0, 0.0, 1.0);
    let light = leg(80.0, -80.0, 0.25, 0.15);
    let excursion: Vec<(Vec2, f32)> = (1..=6)
        .map(|i| (Vec2::new(80.0, 10.0 * i as f32), 1.0))
        .chain((0..=6).map(|i| {
            let t = i as f32 / 6.0;
            (Vec2::new(80.0, 60.0 - 60.0 * t), 1.0 - 0.85 * t)
        }))
        .collect();

    let Some(mut doubled) = engine_or_skip() else {
        return;
    };
    // Out heavy, and back light over the same band — one stroke.
    let both: Vec<(Vec2, f32)> = heavy
        .iter()
        .chain(&excursion)
        .chain(&light)
        .copied()
        .collect();
    stroke_pressed(&mut doubled, under_the_pen(0.8, 20.0), &both);
    let doubled = doubled.render_to_image();

    let mut once = engine_or_skip().expect("the adapter answered once already");
    stroke_pressed(&mut once, under_the_pen(0.8, 20.0), &heavy);
    let once = once.render_to_image();

    for p in [Vec2::ZERO, Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)] {
        let (d, o) = (texel(&doubled, p), texel(&once, p));
        assert!(
            apart(d, o) <= 2,
            "at {p:?} the heavy mark crossed back lightly {d:?} must be the heavy \
             mark alone {o:?} — the light pass took a claim it had no right to"
        );
    }

    // The control: the light pass on its own is a faint mark, so the equality
    // above is not two full strokes agreeing.
    let mut faint = engine_or_skip().expect("the adapter answered once already");
    stroke_pressed(&mut faint, under_the_pen(0.8, 20.0), &light);
    assert!(
        apart(
            texel(&once, Vec2::ZERO),
            texel(&faint.render_to_image(), Vec2::ZERO)
        ) > 20,
        "the light pass reads as the heavy one, so this test measured nothing"
    );
}

/// …and the other way round: a heavy pass over a light mark, within one stroke,
/// fills it in to the heavy pass's ceiling — the crossing shows the *max*, and
/// exactly, the two ceilings falling in different thirds of the dial
/// (`paint_common::claimed_coverage`). What the first-claim law this replaced
/// could not do: at a saturating flow the light pass had claimed the spot.
#[test]
fn a_heavy_pass_over_a_light_mark_fills_it_in() {
    let light = leg(-80.0, 80.0, 0.0, 0.15);
    let heavy = leg(80.0, -80.0, 0.25, 1.0);
    let excursion: Vec<(Vec2, f32)> = (1..=6)
        .map(|i| (Vec2::new(80.0, 10.0 * i as f32), 0.15))
        .chain((0..=6).map(|i| {
            let t = i as f32 / 6.0;
            (Vec2::new(80.0, 60.0 - 60.0 * t), 0.15 + 0.85 * t)
        }))
        .collect();

    let Some(mut doubled) = engine_or_skip() else {
        return;
    };
    let both: Vec<(Vec2, f32)> = light
        .iter()
        .chain(&excursion)
        .chain(&heavy)
        .copied()
        .collect();
    stroke_pressed(&mut doubled, under_the_pen(0.8, 20.0), &both);
    let doubled = doubled.render_to_image();

    let mut once = engine_or_skip().expect("the adapter answered once already");
    stroke_pressed(
        &mut once,
        under_the_pen(0.8, 20.0),
        &leg(-80.0, 80.0, 0.25, 1.0),
    );
    let once = once.render_to_image();

    for p in [Vec2::ZERO, Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)] {
        let (d, o) = (texel(&doubled, p), texel(&once, p));
        assert!(
            apart(d, o) <= 2,
            "at {p:?} the light mark crossed back heavily {d:?} must be the heavy              mark alone {o:?} — the light pass held a ceiling the heavy one should own"
        );
    }
}

/// **The fast path is an optimization, not a semantics**, under the pen too
/// (§6.2): a whisper of `deposit` routes the identical pressed stroke through
/// the stamp loop, whose region aux keeps the claim in its own lane and mints
/// the prefix differences of the claimed law — and the two renderers agree
/// texel for texel along a pressure ramp *and* across a crossing that the
/// stroke covers at two pressures.
#[test]
fn the_loop_lays_the_pen_driven_ceiling_the_fast_path_lays() {
    // Out along y = 0 easing off, and back along y = 0.25 bearing down — so
    // every spot of the band is covered twice, at two pressures, in the order
    // the first-claim rule cares about.
    //
    // Both legs reported every 10 px, so the fitted pressure is the ramp asked
    // for along each leg rather than a spline pulled toward the hairpin's jump
    // over the whole of it.
    let n = 16;
    let path: Vec<(Vec2, f32)> = (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            (Vec2::new(-80.0 + 160.0 * t, 0.0), 1.0 - 0.8 * t)
        })
        .chain((0..=n).map(|i| {
            let t = i as f32 / n as f32;
            (Vec2::new(80.0 - 160.0 * t, 0.25), 0.5 + 0.4 * t)
        }))
        .collect();
    let Some(mut swept) = engine_or_skip() else {
        return;
    };
    stroke_pressed(&mut swept, under_the_pen(0.8, 20.0), &path);
    let swept = swept.render_to_image();

    let mut looped = engine_or_skip().expect("the adapter answered once already");
    let mut whisper = under_the_pen(0.8, 20.0);
    whisper.make_wet().dynamics.deposit = 0.01;
    stroke_pressed(&mut looped, whisper, &path);
    let looped = looped.render_to_image();

    for x in [-60.0, -30.0, 0.0, 30.0, 60.0] {
        let p = Vec2::new(x, 0.0);
        let (l, s) = (texel(&looped, p), texel(&swept, p));
        assert!(
            apart(l, s) <= 2,
            "at {p:?} the loop's stroke {l:?} must be the fast path's {s:?} — \
             the whisper of deposit changed the pen's ceiling"
        );
    }
    // …and the ramp is real: the band fades from the end the pen bore down on
    // first to the end it eased off at, so the agreement above is over a
    // varying ceiling and not a flat one.
    let (bore, eased) = (
        texel(&swept, Vec2::new(-60.0, 0.0)),
        texel(&swept, Vec2::new(60.0, 0.0)),
    );
    assert!(
        apart(bore, eased) > 20,
        "the pressure ramp did not reach the ceiling ({bore:?} against {eased:?}), so this          test measured nothing"
    );
}
