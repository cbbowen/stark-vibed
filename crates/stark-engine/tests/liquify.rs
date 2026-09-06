//! Liquify-brush tests (§6.13): the stroke drags the picture itself — a
//! backward-mapped resample along the travel — rather than trading paint through
//! a reservoir. Covers the effect's identities (a zero-strength drag and a drag
//! along an invariant field both leave the canvas byte-alone; bare canvas takes
//! nothing) and the one thing the tool exists for: an edge moves downstream.
//!
//! The whole-battery invariants — golden, live-vs-commit seam, refinement,
//! translation — ride the corpus's `liquify` case (`tests/corpus.rs`), which is
//! where a regression in the loop's plumbing would show; this file asks the
//! questions only this effect poses.

mod common;

use common::palette::RED;
use common::*;
use stark_engine::Engine;
use stark_engine::command::DocCommand;
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

    // A hard tip, so most of its extent sits at the mask's peak τ and the
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
