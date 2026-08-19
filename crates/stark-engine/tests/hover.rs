//! The hover mark (§18.1.10): the stroke a drag begun this instant would open,
//! folded ahead of the resting pointer — a prediction, never work.
//!
//! What is checked is the word "will": the mark must be the commit's own pixels
//! for the gesture it predicts (`hover == committed`), it must reach *ahead* of
//! the cursor rather than re-draw the trail behind it, and it must behave as a
//! prediction everywhere else — commit nothing, publish nothing, stand down for
//! a real gesture, and never reach a file. The estimator's own claim — the
//! probe holds a heading the raw pixel-stepped reports cannot — closes the
//! file.
//!
//! Presence and absence are probed by **exact image equality against a
//! baseline** wherever possible, so the checks hold under `debug-unfrozen` too
//! (which repaints live tails magenta by design); only the assertions about the
//! mark's *color* are gated off there, as `stroke.rs` gates its own.

mod common;

use common::*;
use stark_engine::command::{GestureCommand, HoverReport, InputSample, PeerCommand, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::{Background, Engine, ExportScale, Offscreen, Rendered};
use stark_model::document::{MattePaint, MatteRegion, Place, Tool};
use stark_model::geom::Vec2;

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];

/// The two reports of the standard hover, ending with the cursor at the canvas
/// origin — where [`center`] reads, and where the probe's touch-down lands.
const A: Vec2 = Vec2::new(-40.0, 0.0);
const B: Vec2 = Vec2::new(0.0, 0.0);

/// How far the probe reaches in these tests, canvas px. The standard hover
/// runs along +x, so the mark spans the origin to `(REACH, 0)`.
const REACH: f32 = 40.0;

/// One hover report at `p`, asking for the standard reach.
fn report(p: Vec2) -> ViewCommand {
    ViewCommand::PreviewHover(Some(HoverReport {
        sample: InputSample::at(p),
        tolerance: DEFAULT_TOLERANCE,
        reach: REACH,
    }))
}

/// Feed the mark the two standard reports, the way the frontend does — one
/// sample per move, the engine holding the window.
fn hover_ab(engine: &mut Engine) {
    for p in [A, B] {
        engine.process(report(p));
    }
}

/// The mark appears under the hover and vanishes when it is dropped — and none
/// of it is work: nothing to undo, no stroke in flight, no committed change.
#[test]
fn the_mark_appears_and_commits_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 10.0)));
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
    engine.process(ViewCommand::SetBrush(brush(RED, 10.0)));
    hover_ab(&mut engine);
    assert!(
        red_dominant(center(&engine.render_to_image())),
        "the touch-down under the cursor is not the brush's paint"
    );
}

/// The mark is a prediction, not a recording: it reaches **ahead** of the
/// cursor along the hover's heading, and the trail behind — where the pointer
/// actually was — stays bare. The screen already shows where you were; what is
/// worth painting is where a press would go.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn the_mark_reaches_ahead_of_the_cursor_not_behind() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 10.0)));
    hover_ab(&mut engine);
    let img = engine.render_to_image();
    assert!(
        red_dominant(img.pixel(img.width / 2 + 20, img.height / 2)),
        "the probe should paint ahead of the cursor"
    );
    assert!(
        red_dominant(center(&img)),
        "the touch-down should land under the cursor"
    );
    assert!(
        !red_dominant(img.pixel(img.width / 2 - 25, img.height / 2)),
        "the trail behind the cursor is not the mark"
    );
}

/// `hover == committed`, prediction edition: the probe is bit-for-bit what
/// committing the gesture it predicts — press at the cursor, drag `REACH`
/// along the hover's heading — would land. The prediction is synthesized; the
/// rendering of it is not, and this is the test that pins the difference.
///
/// Gated for its comparison's sake: under `debug-unfrozen` the hover renders
/// as live tail (magenta) while the commit lands in the stroke's color.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn the_mark_is_what_committing_the_prediction_would_land() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 10.0)));
    hover_ab(&mut engine);
    let hovered = engine.render_to_image();

    // The predicted gesture, made real — with the same Lamport seed, since the
    // hover banked nothing that could have advanced it.
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(B),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(REACH, 0.0)),
    });
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image();
    assert!(
        images_match(&hovered, &committed, 0),
        "the probe is not the commit's own pixels"
    );
}

/// A real gesture outranks the prediction, and ends it: the mark leaves with
/// the press and does not reappear at pen-up — a stale window surviving the
/// stroke would resurrect a pre-press mark the moment the hand lifted.
#[test]
fn a_real_gesture_takes_the_mark_down() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 10.0)));
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
    engine.process(ViewCommand::SetBrush(brush(RED, 10.0)));
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
/// nothing, and switching back to the brush does not resurrect the window fed
/// while the marquee was armed.
#[test]
fn a_selection_tool_folds_no_mark() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 10.0)));
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
        "a window fed under the marquee resurfaced on the switch back"
    );
}

/// An unpaintable active layer refuses the mark exactly as it refuses a stroke
/// (§15.7): the same renderer answers both, so there is no gate to forget.
#[test]
fn an_unpaintable_layer_refuses_the_mark() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::SetBrush(brush(RED, 10.0)));
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

/// The window is the estimator: a shallow line quantized to whole-pixel steps —
/// the staircase a mouse reports — yields a probe whose heading is the *line's*,
/// carried forward from the cursor at exactly the asked reach. The raw pair can
/// only answer 0° or 45°, wrong by the whole heading; the fit over the window
/// prices that jitter away (§6.2).
///
/// Session-level and CPU-only: the estimator is the machinery under test, and
/// it needs no GPU, so this runs even where the render suites skip.
#[test]
fn a_pixel_staircase_holds_its_heading() {
    use stark_engine::peer::GestureView;
    use stark_engine::session::Session;
    use stark_engine::{LiveGesture, ViewTransform};
    use stark_model::document::{ActorId, LayerId};
    use stark_model::geom::Extent2;

    let mut session = Session::new(ViewTransform::identity(Extent2::new(512, 512)), LayerId(0));
    // A 1-in-4 staircase: the pixel grid's rendering of a line at atan(1/4).
    let true_heading = 0.25_f32.atan();
    let mut samples = Vec::new();
    let mut y = 0;
    for i in 0..24 {
        if i % 4 == 3 {
            y += 1;
        }
        samples.push(InputSample {
            pos: Vec2::new(i as f32, y as f32),
            time: i as f64 * 0.016,
            ..Default::default()
        });
    }
    for s in &samples {
        // A mouse's grain at 1:1 — the tolerance the frontend would state.
        assert!(session.hover_to(*s, 1.0, 50.0));
    }

    // The jitter being claimed away: the newest raw pair's heading misses the
    // line by double digits, whichever compass point it lands on.
    let step = samples[23].pos - samples[22].pos;
    let naive = step.y.atan2(step.x);
    assert!(
        (naive - true_heading).abs() > 10_f32.to_radians(),
        "the staircase no longer exhibits the jitter this test is about"
    );

    let Some(GestureView {
        gesture: LiveGesture::Stroke(rec),
        ..
    }) = session.hover_view(ActorId::SOLO, 0)
    else {
        panic!("a fed hover offers no view");
    };
    // The probe starts under the cursor and spans the asked reach.
    let (start, end) = (
        rec.path.first().expect("a probe has knots").pos,
        rec.path.last().expect("a probe has knots").pos,
    );
    assert_eq!(
        start, samples[23].pos,
        "the probe must start under the cursor"
    );
    let d = end - start;
    assert!(
        (d.length() - 50.0).abs() < 0.5,
        "the probe spans {} canvas px where 50 were asked",
        d.length()
    );
    let heading = d.y.atan2(d.x);
    assert!(
        (heading - true_heading).abs() < 6_f32.to_radians(),
        "the probe's heading is {:.1}° where the line is {:.1}°",
        heading.to_degrees(),
        true_heading.to_degrees(),
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
