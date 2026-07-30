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
use stark_core::command::DocCommand;
use stark_core::geom::Vec2;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

const STROKE_A: &[Vec2] = &[Vec2::new(-40.0, -20.0), Vec2::new(40.0, 20.0)];
const STROKE_B: &[Vec2] = &[Vec2::new(-40.0, 40.0), Vec2::new(40.0, -40.0)];

#[test]
fn rendering_is_deterministic() {
    let (Some(mut a), Some(mut b)) = (engine_or_skip_blue(), engine_or_skip_blue()) else {
        return;
    };
    paint(&mut a, RED, 30.0, STROKE_A);
    paint(&mut b, RED, 30.0, STROKE_A);

    let ia = a.render_to_image();
    let ib = b.render_to_image();
    assert!(
        images_match(&ia, &ib, 0),
        "two identical scripts must render bit-equal images"
    );
}

#[test]
fn undo_redo_roundtrip_is_lossless() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut engine, RED, 30.0, STROKE_A);
    let before = engine.render_to_image();

    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Redo);
    let after = engine.render_to_image();

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
    let Some(mut two) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut two, RED, 28.0, STROKE_A);
    paint(&mut two, GREEN, 28.0, STROKE_B);
    two.process(DocCommand::Undo);
    let undone = two.render_to_image();

    let Some(mut one) = engine_or_skip_blue() else {
        return;
    };
    paint(&mut one, RED, 28.0, STROKE_A);
    let only_a = one.render_to_image();

    assert!(
        images_match(&undone, &only_a, 0),
        "undoing the last stroke must equal never having committed it"
    );
}

// --- Timeline scrubbing (MISSING_FEATURES §2.4) -----------------------------
//
// The scrubber moves the same applied/withheld split undo and redo move, so what
// these guard is that moving it in bulk lands exactly where moving it one step at
// a time would — the property that lets the mode reuse the undo stack instead of
// keeping a second playhead beside it.
//
// These build on `engine_or_skip` rather than the blue-ground helper the tests
// above use: setting the substrate is a *logged* action (FRAME_DESIGN.md §5), so a
// blue engine starts one step into its own history and every position here would
// have to be written as an offset from that. What is being asserted is arithmetic
// on positions, and it should read as the arithmetic it is.

const STROKE_C: &[Vec2] = &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)];

#[test]
fn the_scrub_range_tracks_the_playhead() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    assert_eq!(
        engine.scrub_range(),
        Some((0, 0)),
        "a fresh document is empty"
    );

    paint(&mut engine, RED, 24.0, STROKE_A);
    paint(&mut engine, GREEN, 24.0, STROKE_B);
    assert_eq!(engine.scrub_range(), Some((2, 2)));

    engine.process(DocCommand::Seek(1));
    assert_eq!(
        engine.scrub_range(),
        Some((1, 2)),
        "scrubbing back withholds without discarding"
    );

    // Out of range clamps rather than panicking or wrapping.
    engine.process(DocCommand::Seek(99));
    assert_eq!(engine.scrub_range(), Some((2, 2)));
}

#[test]
fn scrubbing_back_and_forward_is_lossless() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 26.0, STROKE_A);
    paint(&mut engine, GREEN, 26.0, STROKE_B);
    paint(&mut engine, RED, 26.0, STROKE_C);
    let before = engine.render_to_image();

    engine.process(DocCommand::Seek(0));
    engine.process(DocCommand::Seek(3));
    let after = engine.render_to_image();

    assert!(
        images_match(&before, &after, 0),
        "a scrub to the start and back must reproduce identical pixels"
    );
}

#[test]
fn a_bulk_scrub_lands_where_repeated_undo_lands() {
    // The whole reason `seek` may pop in bulk: the cheaper path has to compute the
    // same state the expensive one does, pixel for pixel.
    let (Some(mut bulk), Some(mut stepwise)) = (engine_or_skip(), engine_or_skip()) else {
        return;
    };
    for engine in [&mut bulk, &mut stepwise] {
        paint(engine, RED, 22.0, STROKE_A);
        paint(engine, GREEN, 22.0, STROKE_B);
        paint(engine, RED, 22.0, STROKE_C);
    }
    bulk.process(DocCommand::Seek(1));
    for _ in 0..2 {
        stepwise.process(DocCommand::Undo);
    }

    assert_eq!(bulk.scrub_range(), stepwise.scrub_range());
    assert!(
        images_match(&bulk.render_to_image(), &stepwise.render_to_image(), 0),
        "one seek of two steps must equal two undos"
    );
}

#[test]
fn scrubbing_forward_matches_a_history_that_only_ever_got_that_far() {
    let Some(mut scrubbed) = engine_or_skip() else {
        return;
    };
    paint(&mut scrubbed, RED, 26.0, STROKE_A);
    paint(&mut scrubbed, GREEN, 26.0, STROKE_B);
    scrubbed.process(DocCommand::Seek(1));

    let Some(mut only_a) = engine_or_skip() else {
        return;
    };
    paint(&mut only_a, RED, 26.0, STROKE_A);

    assert!(
        images_match(&scrubbed.render_to_image(), &only_a.render_to_image(), 0),
        "standing at step 1 must equal a document that only ever had step 1"
    );
}

#[test]
fn committing_at_a_scrubbed_position_truncates_the_future() {
    // Scrubbing is navigation, so painting from where it left the playhead has to
    // behave exactly as painting after an undo does — the withheld steps go.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 26.0, STROKE_A);
    paint(&mut engine, GREEN, 26.0, STROKE_B);
    engine.process(DocCommand::Seek(0));
    paint(&mut engine, RED, 26.0, STROKE_C);

    assert_eq!(engine.scrub_range(), Some((1, 1)));
    assert!(!engine.observe().can_redo);
}

#[test]
fn scrub_labels_span_the_whole_range() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 20.0, STROKE_A);
    engine.process(DocCommand::AddLayer { above: None });
    paint(&mut engine, GREEN, 20.0, STROKE_B);
    engine.process(DocCommand::Seek(1));

    assert_eq!(
        engine.scrub_labels(),
        vec!["Stroke", "Add layer", "Stroke"],
        "the withheld steps are still steps the scrubber can travel to"
    );
}

#[test]
fn a_scrub_past_a_layers_creation_leaves_the_brush_somewhere_to_paint() {
    // Dragging to the start withdraws every `AddLayer`, so the selected layer
    // routinely stops existing — a playhead that left it dangling would swallow
    // the next stroke in silence.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(DocCommand::AddLayer { above: None });
    paint(&mut engine, RED, 20.0, STROKE_A);
    engine.process(DocCommand::Seek(0));

    let obs = engine.observe();
    assert!(
        obs.layers.iter().any(|l| l.id == obs.active_layer),
        "the selected layer must exist wherever the playhead lands"
    );
}
