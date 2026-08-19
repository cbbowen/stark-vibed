//! The hover mark (§18.1.10): the stroke the last two hover reports would have
//! committed, folded under the resting pointer — a hypothesis, never work.
//!
//! What is checked is the word "would": the mark must be the commit's own pixels
//! (`hover == committed`), and it must behave as a hypothesis everywhere else —
//! commit nothing, publish nothing, stand down for a real gesture, and never
//! reach a file.
//!
//! Presence and absence are probed by **exact image equality against a
//! baseline** wherever possible, so the checks hold under `debug-unfrozen` too
//! (which repaints live tails magenta by design); only the assertions about the
//! mark's *color* are gated off there, as `stroke.rs` gates its own.

mod common;

use common::*;
use stark_engine::command::{GestureCommand, InputSample, PeerCommand, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::{Background, Engine, ExportScale, Offscreen, Rendered};
use stark_model::document::{MattePaint, MatteRegion, Place, Tool};
use stark_model::geom::Vec2;

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];

/// The two reports of the standard hover: a short travel straddling the canvas
/// origin, where [`center`] reads.
const A: Vec2 = Vec2::new(-20.0, 0.0);
const B: Vec2 = Vec2::new(20.0, 0.0);

/// Feed the mark the two standard reports, the way the frontend does — one
/// sample per move, the engine holding the pair.
fn hover_ab(engine: &mut Engine) {
    for p in [A, B] {
        engine.process(ViewCommand::PreviewHover(Some((
            InputSample::at(p),
            DEFAULT_TOLERANCE,
        ))));
    }
}

/// The mark appears under the hover and vanishes when it is dropped — and none
/// of it is work: nothing to undo, no stroke in flight, no committed change.
#[test]
fn the_mark_appears_and_commits_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    let before = engine.render_to_image();
    let rev = engine.observe().doc_revision;

    hover_ab(&mut engine);
    let during = engine.render_to_image();
    assert!(
        !images_match(&before, &during, 0),
        "the hover folded no mark at all"
    );
    let obs = engine.observe();
    assert!(!obs.is_stroking, "a hover is not a stroke in flight");
    assert!(!obs.can_undo, "a hover left something to undo");
    assert_eq!(
        obs.doc_revision, rev,
        "a hover moved the committed document"
    );

    engine.process(ViewCommand::PreviewHover(None));
    let after = engine.render_to_image();
    assert!(
        images_match(&before, &after, 0),
        "dropping the mark did not restore the canvas exactly"
    );
}

/// The mark is red because the brush is: the claim that what is folded is the
/// brush's own paint, not a chrome-colored stand-in. Color-reading, so gated
/// off under the tail tint (see the module doc).
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn the_mark_is_the_brushs_own_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    hover_ab(&mut engine);
    assert!(
        red_dominant(center(&engine.render_to_image())),
        "the mark under the hover is not the brush's paint"
    );
}

/// `hover == committed`: the mark is bit-for-bit what a gesture of exactly the
/// same two samples commits. The strongest claim here, and the cheapest to
/// hold — the mark is built by the same fitter and rendered by the same
/// renderer with the same seed, so this is inherited, not maintained.
///
/// Gated for its comparison's sake: under `debug-unfrozen` the hover renders
/// as live tail (magenta) while the commit lands in the stroke's color.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn the_mark_is_what_the_same_samples_would_commit() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    hover_ab(&mut engine);
    let hovered = engine.render_to_image();

    // The same two samples as a real gesture, on the same engine: `Start` takes
    // the same Lamport seed the hover's fold used, since the hover banked
    // nothing that could have advanced it.
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(A),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(B),
    });
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image();
    assert!(
        images_match(&hovered, &committed, 0),
        "the mark is not the commit's own pixels"
    );
}

/// A real gesture outranks the hypothesis, and ends it: the mark leaves with
/// the press and does not reappear at pen-up — a stale pair surviving the
/// stroke would resurrect a pre-press mark the moment the hand lifted.
#[test]
fn a_real_gesture_takes_the_mark_down() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    let paper = center(&engine.render_to_image());
    hover_ab(&mut engine);

    // Press elsewhere: the mark at the origin must vanish while the gesture at
    // y = 80 previews.
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-30.0, 80.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(30.0, 80.0)),
    });
    assert_eq!(
        center(&engine.render_to_image()),
        paper,
        "the hover mark survived a real gesture starting"
    );
    engine.process(GestureCommand::End);
    assert_eq!(
        center(&engine.render_to_image()),
        paper,
        "the pre-press mark reappeared at pen-up"
    );
}

/// The mark never reaches a file: a `Rendered::Live` export takes it down and
/// renders without it — and what the screen shows afterwards agrees, because
/// the drop is a drop rather than a per-render exclusion.
#[test]
fn the_mark_never_reaches_a_live_export() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    let screen_before = engine.render_to_image();
    let export_before = export_live(&mut engine);

    hover_ab(&mut engine);
    assert!(
        !images_match(&screen_before, &engine.render_to_image(), 0),
        "the hover folded no mark, so the export check below is vacuous"
    );
    let exported = export_live(&mut engine);
    assert!(
        images_match(&export_before, &exported, 0),
        "the hover mark reached an export"
    );
    assert!(
        images_match(&screen_before, &engine.render_to_image(), 0),
        "the export left the screen still showing the mark it excluded"
    );
}

/// A selection tool drags a shape, not the brush: hover reports under one fold
/// nothing, and switching back to the brush does not resurrect the pair fed
/// while the marquee was armed.
#[test]
fn a_selection_tool_folds_no_mark() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    engine.process(ViewCommand::SetTool(Tool::SelectRect));
    let before = engine.render_to_image();
    hover_ab(&mut engine);
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a marquee tool folded a brush mark"
    );
    engine.process(ViewCommand::SetTool(Tool::Brush));
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a pair fed under the marquee resurfaced on the switch back"
    );
}

/// An unpaintable active layer refuses the mark exactly as it refuses a stroke
/// (§15.7): the same renderer answers both, so there is no gate to forget.
#[test]
fn an_unpaintable_layer_refuses_the_mark() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 40.0)));
    engine.process(stark_engine::command::DocCommand::AddMatte {
        carrier: None,
        at: Place::Top,
        region: MatteRegion::OutsideRect {
            min: Vec2::new(-120.0, -120.0),
            max: Vec2::new(120.0, 120.0),
        },
        paint: MattePaint::Solid([0.0, 0.0, 0.0]),
    });
    let matte = engine.observe().layers.last().expect("the matte").id;
    engine.process(PeerCommand::SetActiveLayer(matte));
    let before = engine.render_to_image();
    hover_ab(&mut engine);
    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a matte took the hover mark though it takes no stroke"
    );
}

/// One `Rendered::Live` export at the viewport, blocking — the shape every
/// export test in `export.rs` uses, without a frame so the fallback rect is
/// the same viewport `render_to_image` shows.
fn export_live(engine: &mut Engine) -> stark_engine::RgbaImage {
    pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                None,
                ExportScale::Factor(1.0),
                Background::Substrate,
                Rendered::Live,
            )
            .expect("export"),
    )
    .expect("the readback completes")
}
