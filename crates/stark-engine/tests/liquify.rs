//! Liquify-brush tests (§6.13): the stroke drags the picture itself — a
//! displacement field composed segment by segment, the picture resampled through
//! it once — rather than trading paint through a reservoir. Covers the effect's
//! identities (a zero-strength drag and a drag along an invariant field both leave
//! the canvas byte-alone; bare canvas takes nothing), the one thing the tool exists
//! for (an edge moves downstream), the reason the field exists (a run of short
//! drags keeps an edge as sharp as one), and the run's rule for what a paint
//! stroke does to it.
//!
//! The whole-battery invariants — golden, live-vs-commit seam, refinement,
//! translation — ride the corpus's `liquify` case (`tests/corpus.rs`), which is
//! where a regression in the loop's plumbing would show; this file asks the
//! questions only this effect poses.

mod common;

use common::palette::RED;
use common::*;
use stark_engine::command::DocCommand;
use stark_engine::{Engine, RgbaImage};
use stark_model::Srgb;
use stark_model::document::{
    BrushEffect, BrushParams, BrushShape, FillOp, LiquifyEffect, SelectionShape,
};
use stark_model::geom::Vec2;

/// A liquify brush of the given strength — no pigment (the effect carries none),
/// no drain, no jitter, so a test's claim is about the warp alone.
fn liquify_brush(radius: f32, strength: f32) -> BrushParams {
    let mut b = brush(RED, radius);
    b.drain = 0.0;
    b.jitter = 0.0;
    b.effect = BrushEffect::Liquify(LiquifyEffect {
        strength,
        ..LiquifyEffect::default()
    });
    b
}

/// A red band laid without jitter — what the guard test below drags across.
fn flat_band(engine: &mut Engine, radius: f32, from: Vec2, to: Vec2) {
    let mut b = brush(RED, radius);
    b.drain = 0.0;
    b.jitter = 0.0;
    replay_with(engine, b, &[from, to]);
}

/// A rect of **bitwise-uniform** paint: a feather-0 fill lays one parcel value on
/// every interior texel, which no stroke can — a swept deposit's per-segment
/// sums leave ±1-ulp f16 speckle along its own length, real texture a warp
/// honestly relocates (the first run of the identity test below measured exactly
/// that, worst 2 levels). The identity claim needs a field with nothing to
/// relocate.
fn flat_fill(engine: &mut Engine, min: Vec2, max: Vec2) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Fill {
        layer,
        op: FillOp::new(
            SelectionShape::rect_from_corners(min, max),
            0.0,
            Srgb::new(RED),
            1.0,
        ),
    });
}

/// The rewrite guard, end to end: a stroke whose strength is zero moves nothing
/// and must store nothing — not "nearly the same picture", the same bytes. This
/// is `dynamics.wesl::warp`'s early return doing its job (§6.13), and it is the
/// property that keeps a near-inert drag from walking texels down the f16
/// lattice the way §6.2's wiggle-path repro did.
#[test]
fn a_liquify_stroke_at_zero_strength_leaves_the_canvas_byte_identical() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    flat_band(
        &mut engine,
        40.0,
        Vec2::new(-110.0, 0.0),
        Vec2::new(110.0, 0.0),
    );
    let before = engine.render_to_image();
    stroke_with(
        &mut engine,
        liquify_brush(24.0, 0.0),
        &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 10.0)],
    );
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 0),
        "a zero-strength drag re-stored the canvas",
    );
}

/// Dragging through a field with **nothing to relocate** is the identity, to the
/// byte: every gather tap holds the one value the fill laid, the bilinear mix of
/// equal values lands back on it through `lib::store::f16_nearest`, and no texel
/// outside the extent is ever written. This is the warp's "conserves where there
/// is nothing to move" — the smudge's
/// `conservative_smear_preserves_uniform_field`, in the geometry this effect
/// composes in (§6.13). The field is a feather-0 *fill*, not a stroke, and that
/// is load-bearing: a stroke's interior carries ±1-ulp f16 speckle along its own
/// length (per-segment sums), real texture the warp honestly relocates.
#[test]
fn a_drag_through_a_uniform_fill_is_the_identity() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    flat_fill(
        &mut engine,
        Vec2::new(-110.0, -60.0),
        Vec2::new(110.0, 60.0),
    );
    let before = engine.render_to_image();
    // A bent drag, so the identity is asked of the arc arithmetic and the
    // cross-row taps too — the field is uniform in every direction, so any
    // path must leave it alone. Extents and upstream reads stay well inside
    // the fill's hard edge.
    stroke_with(
        &mut engine,
        liquify_brush(18.0, 1.0),
        &[
            Vec2::new(-50.0, -12.0),
            Vec2::new(0.0, 14.0),
            Vec2::new(50.0, -12.0),
        ],
    );
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 0),
        "a drag through a uniform fill moved paint it could not distinguish",
    );
}

/// The tool's whole point: a drag across a band's **end** carries the edge
/// downstream — paper that had no paint is covered by the paint pulled over it,
/// and the band upstream of the tip stays put.
#[test]
fn a_liquify_drag_carries_an_edge_downstream() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A hard-shouldered band ending near x = 20 (the endpoint plus the tip's
    // reach), so the probe just past it starts as bare paper.
    let mut lay = brush(RED, 40.0);
    lay.drain = 0.0;
    lay.jitter = 0.0;
    lay.shape = BrushShape::Round { hardness: 0.9 };
    replay_with(
        &mut engine,
        lay,
        &[Vec2::new(-150.0, 0.0), Vec2::new(-20.0, 0.0)],
    );
    let probe = Vec2::new(35.0, 0.0);
    let upstream = Vec2::new(-60.0, 0.0);
    let before = engine.render_to_image();
    assert!(
        !painted(&before, probe),
        "the probe must start on bare paper for the drag to be visible",
    );
    assert!(painted(&before, upstream), "the band must reach upstream");

    // A hard tip, so most of its extent sits on the profile's plateau and the
    // follow approaches the travel itself (§6.13).
    let mut b = liquify_brush(30.0, 1.0);
    b.shape = BrushShape::Round { hardness: 0.9 };
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-10.0, 0.0), Vec2::new(100.0, 0.0)],
    );
    let after = engine.render_to_image();
    assert!(
        painted(&after, probe),
        "the band's edge did not follow the drag",
    );
    assert!(
        painted(&after, upstream),
        "paint upstream of the drag must stay put",
    );
}

/// A liquify stroke over bare canvas commits nothing visible: there is no paint
/// for the warp to carry, so the picture is the picture it found — the eraser's
/// "lays nothing" from the other side (§6.13).
#[test]
fn a_liquify_stroke_on_bare_canvas_shows_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let before = engine.render_to_image();
    stroke_with(
        &mut engine,
        liquify_brush(30.0, 1.0),
        &[Vec2::new(-80.0, -20.0), Vec2::new(80.0, 20.0)],
    );
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 0),
        "a drag over bare canvas left a mark",
    );
}

/// How many px wide the band's leading edge is along `y`, between `x0` and `x1`:
/// the count of texels that are neither the band's own red nor bare paper — the
/// haze a resample leaves on a hard edge.
fn edge_width(img: &RgbaImage, y: f32, x0: i32, x1: i32) -> usize {
    let on = texel(img, Vec2::new(x0 as f32, y));
    let off = texel(img, Vec2::new(x1 as f32, y));
    assert!(
        apart(on, off) > 60,
        "the row must run from the band to bare paper for an edge to be measured",
    );
    (x0..=x1)
        .filter(|&x| {
            let c = texel(img, Vec2::new(x as f32, y));
            // Strictly between the two plateaus, by more than a rounding level.
            apart(c, on) > 8 && apart(c, off) > 8
        })
        .count()
}

/// **The reason the field exists** (§6.13): an edge worked by a run of short
/// drags is as sharp as one worked by a single drag of the same length. Ten
/// strokes compose into one field and the picture is resampled once through it,
/// where a resample per stroke — the first design — hazed the edge a little more
/// each time until the tool that exists to correct a drawing was softening it.
#[test]
fn a_run_of_short_drags_keeps_an_edge_as_sharp_as_one_drag() {
    let Some(mut once) = engine_or_skip() else {
        return;
    };
    let Some(mut many) = engine_or_skip() else {
        return;
    };
    // A hard-edged band ending at x = −20, so the leading edge sits under the
    // tip's plateau for the whole of the drag below.
    for e in [&mut once, &mut many] {
        flat_fill(e, Vec2::new(-110.0, -60.0), Vec2::new(-20.0, 60.0));
    }
    let mut b = liquify_brush(24.0, 1.0);
    b.shape = BrushShape::Round { hardness: 0.8 };
    let before = edge_width(&once.render_to_image(), 0.0, -60, 60);
    // One drag of 40 px…
    stroke_with(&mut once, b, &[Vec2::new(-30.0, 0.0), Vec2::new(10.0, 0.0)]);
    // …against eight drags of 5 px, each a stroke of its own, each starting
    // where the last stopped.
    for k in 0..8 {
        let x = -30.0 + 5.0 * k as f32;
        stroke_with(&mut many, b, &[Vec2::new(x, 0.0), Vec2::new(x + 5.0, 0.0)]);
    }
    let one = once.render_to_image();
    let eight = many.render_to_image();
    // Both carried the edge downstream: the paper past the old edge is covered.
    assert!(
        painted(&one, Vec2::new(0.0, 0.0)) && painted(&eight, Vec2::new(0.0, 0.0)),
        "the edge did not follow the drag",
    );
    let w_one = edge_width(&one, 0.0, -60, 60);
    let w_eight = edge_width(&eight, 0.0, -60, 60);
    assert!(
        w_one <= before + 2,
        "one drag hazed the edge from {before} to {w_one} px",
    );
    assert!(
        w_eight <= w_one + 1,
        "eight composed drags hazed the edge to {w_eight} px against one drag's {w_one}",
    );
}

/// A liquify stroke composes into the layer's run, and a paint stroke decides
/// whether it may — by where it lands (§6.13, §12.6). Inside the next stroke's
/// declared reach a paint changes a tile the run recorded, so the stroke starts a
/// run afresh; beyond the reach the run carries on, and the two strokes' tiles
/// are one run's.
#[test]
fn a_paint_inside_the_reach_resets_the_run_and_one_beyond_it_does_not() {
    let run_size = |e: &Engine| {
        let layer = e.observe().active_layer;
        e.document()
            .layer(layer)
            .and_then(|l| l.liquify_run())
            .map_or(0, |r| r.written())
    };
    let first = [Vec2::new(-100.0, -40.0), Vec2::new(-60.0, -40.0)];
    let second = [Vec2::new(40.0, 40.0), Vec2::new(80.0, 40.0)];
    for (paint_at, composes) in [
        (Vec2::new(3000.0, 3000.0), true),
        (Vec2::new(10.0, 0.0), false),
    ] {
        let Some(mut engine) = engine_or_skip() else {
            return;
        };
        flat_fill(
            &mut engine,
            Vec2::new(-120.0, -120.0),
            Vec2::new(120.0, 120.0),
        );
        stroke_with(&mut engine, liquify_brush(10.0, 1.0), &first);
        let after_first = run_size(&engine);
        assert_eq!(after_first, 1, "the first drag writes its one tile");
        paint(
            &mut engine,
            RED,
            5.0,
            &[paint_at, paint_at + Vec2::new(20.0, 0.0)],
        );
        assert_eq!(
            run_size(&engine),
            after_first,
            "a paint stroke never touches the run",
        );
        stroke_with(&mut engine, liquify_brush(10.0, 1.0), &second);
        let want = if composes { 2 } else { 1 };
        assert_eq!(
            run_size(&engine),
            want,
            "after a paint at {paint_at:?} the second drag should {} the run",
            if composes { "compose into" } else { "restart" },
        );
    }
}
