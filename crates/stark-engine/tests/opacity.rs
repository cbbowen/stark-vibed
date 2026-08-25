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
use stark_engine::RgbaImage;
use stark_engine::command::DocCommand;
use stark_model::Srgb;
use stark_model::document::{BrushParams, FillOp, SelectionShape};
use stark_model::geom::Vec2;

const RED: [f32; 3] = [1.0, 0.0, 0.0];

/// A swept-path wash at `opacity`: flow high enough that one pass saturates the
/// parcel over the stroke's core, so a test sampling the core is reading the
/// dial and not a half-built fringe — the erase suite's construction, laying
/// instead of removing.
fn washed(opacity: f32, radius: f32) -> BrushParams {
    let mut b = brush(RED, radius);
    b.paint_mut().expect("a paint brush").dynamics.flow = 2.5;
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

/// A pixel's screen position for a canvas point, under the tests' identity view.
fn screen_of(canvas: Vec2) -> (u32, u32) {
    let half = Vec2::new(SIZE.width as f32, SIZE.height as f32) * 0.5;
    let p = canvas + half;
    (p.x as u32, p.y as u32)
}

fn texel(img: &RgbaImage, canvas: Vec2) -> [i32; 3] {
    let (x, y) = screen_of(canvas);
    let c = img.pixel(x, y);
    [c[0] as i32, c[1] as i32, c[2] as i32]
}

/// The worst per-channel distance between two texels.
fn apart(a: [i32; 3], b: [i32; 3]) -> i32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).max().unwrap()
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
        b.paint_mut().expect("a paint brush").dynamics.deposit = 0.01;
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
    whisper.paint_mut().expect("a paint brush").dynamics.deposit = 0.01;

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
        let p = b.paint_mut().expect("a paint brush");
        p.dynamics.flow = 0.0;
        p.dynamics.deposit = 0.6;
        p.dynamics.charge = charge;
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
