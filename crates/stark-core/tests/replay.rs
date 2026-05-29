//! Step-3 history & replay-equivalence tests (DESIGN.md §9).
//!
//! These guard the invariant that pixels are a deterministic function of the
//! action log (DESIGN.md §1): identical scripts produce identical images, undo
//! then redo is lossless, and undoing the last action reproduces the exact
//! state of a history that never had it — which exercises the `history` crate's
//! checkpoint-and-replay path through the GPU. (Serialize round-trip joins this
//! set in step 5.)

mod common;

use common::*;
use stark_core::geom::Vec2;
use stark_core::InputCommand;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

const STROKE_A: &[Vec2] = &[Vec2::new(-40.0, -20.0), Vec2::new(40.0, 20.0)];
const STROKE_B: &[Vec2] = &[Vec2::new(-40.0, 40.0), Vec2::new(40.0, -40.0)];

#[test]
fn rendering_is_deterministic() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    paint(&mut a, RED, 30.0, STROKE_A);
    paint(&mut b, RED, 30.0, STROKE_A);

    let ia = a.render_to_image(BG);
    let ib = b.render_to_image(BG);
    assert!(
        images_match(&ia, &ib, 0),
        "two identical scripts must render bit-equal images"
    );
}

#[test]
fn undo_redo_roundtrip_is_lossless() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 30.0, STROKE_A);
    let before = engine.render_to_image(BG);

    engine.process(InputCommand::Undo);
    engine.process(InputCommand::Redo);
    let after = engine.render_to_image(BG);

    assert!(
        images_match(&before, &after, 0),
        "undo then redo must reproduce identical pixels"
    );
}

#[test]
fn undo_matches_shorter_history() {
    // Two committed strokes, then undo the second. Popping the second action
    // forces `history` to replay stroke A from a cached checkpoint, re-rendering
    // it on the GPU — the result must equal a timeline that only ever had A.
    let Some(mut two) = engine_or_skip() else {
        return;
    };
    paint(&mut two, RED, 28.0, STROKE_A);
    paint(&mut two, GREEN, 28.0, STROKE_B);
    two.process(InputCommand::Undo);
    let undone = two.render_to_image(BG);

    let Some(mut one) = engine_or_skip() else {
        return;
    };
    paint(&mut one, RED, 28.0, STROKE_A);
    let only_a = one.render_to_image(BG);

    assert!(
        images_match(&undone, &only_a, 0),
        "undoing the last stroke must equal never having committed it"
    );
}
