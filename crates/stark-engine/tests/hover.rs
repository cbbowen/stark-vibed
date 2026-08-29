//! The hover mark (§18.1.10): the stroke a drag begun this instant would open,
//! folded ahead of the resting pointer — a prediction, never work.
//!
//! What is checked is the word "will": the mark must be the commit's own pixels
//! for the gesture it predicts (`hover == committed`), it must reach *ahead* of
//! the cursor rather than re-draw the trail behind it, and it must behave as a
//! prediction everywhere else — commit nothing, publish nothing, stand down for
//! a real gesture, and never reach a file. The estimator's own claim — the
//! probe holds a heading the raw pixel-stepped reports cannot — and its
//! afterlife as a stroke's run-up, extending the committed curve behind a
//! marker that is never painted (§6.2), close the file.
//!
//! Presence and absence are probed by **exact image equality against a
//! baseline** wherever possible, so the checks hold under `debug-unfrozen` too
//! (which repaints live tails magenta by design); only the assertions about the
//! mark's *color* are gated off there, as `stroke.rs` gates its own.

mod common;

use common::*;
use stark_engine::command::Tool;
use stark_engine::command::{GestureCommand, HoverReport, InputSample, PeerCommand, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::{Background, Engine, ExportScale, Offscreen, Rendered};
use stark_model::Srgb;
use stark_model::document::{MatteRegion, Parcel, Place};
use stark_model::geom::Vec2;

const RED: [f32; 3] = [0.85, 0.1, 0.1];

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
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
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
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
    hover_ab(&mut engine);
    assert!(
        red_dominant(center(&engine.render_to_image())),
        "the mark under the cursor is not the brush's paint"
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
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
    hover_ab(&mut engine);
    let img = engine.render_to_image();
    assert!(
        red_dominant(img.pixel(img.width / 2 + 20, img.height / 2)),
        "the probe should paint ahead of the cursor"
    );
    assert!(
        red_dominant(center(&img)),
        "the mark should begin under the cursor"
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
/// The window is surrendered before the press, deliberately: a press made
/// *while hovering* takes the window as its run-up and smooths its entry
/// through the watched motion (§6.2) — the right stroke, and a different
/// record from the two-sample gesture the probe stands for. What is pinned
/// here is the probe's own claim; what a run-up press lays is
/// [`the_run_up_is_evidence_never_paint`]'s.
///
/// Gated for its comparison's sake: under `debug-unfrozen` the hover renders
/// as live tail (magenta) while the commit lands in the stroke's color.
#[cfg(not(feature = "debug-unfrozen"))]
#[test]
fn the_mark_is_what_committing_the_prediction_would_land() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
    hover_ab(&mut engine);
    let hovered = engine.render_to_image();

    // The predicted gesture, made real — with the same Lamport seed, since the
    // hover banked nothing that could have advanced it, and without the window,
    // which would otherwise become the gesture's run-up (see above).
    engine.process(ViewCommand::PreviewHover(None));
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
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
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
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
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
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
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
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
    engine.process(stark_engine::command::DocCommand::AddMatte {
        carrier: None,
        at: Place::Top,
        region: MatteRegion::OutsideRect {
            min: Vec2::new(-120.0, -120.0),
            max: Vec2::new(120.0, 120.0),
        },
        paint: Parcel::Solid(Srgb::new([0.0, 0.0, 0.0])),
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
    use stark_engine::GestureView;
    use stark_engine::Session;
    use stark_engine::{LiveGesture, ViewTransform};
    use stark_model::document::{ActorId, LayerId};
    use stark_model::geom::Extent2;

    let mut session = Session::new(
        ViewTransform::identity(Extent2::new(512, 512)),
        LayerId::ROOT,
    );
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
        // A mouse's resolution at 1:1 — the tolerance the frontend would state.
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

/// The estimator outlives the hover: the window a press interrupts becomes the
/// stroke's **run-up** — real leading samples the fitted curve extends back
/// through, with the record's marker (`StrokeRecord::start`) saying where on
/// that curve the stroke itself begins (§6.2, `PathFitter::seed_runup`).
///
/// The scenario is the jitter pathology at full strength: a 14° line reported
/// as a 1-in-4 pixel staircase. Cold, the entry's heading is decided by where
/// one control point lands against tolerance-quantized samples; after a watched
/// approach, the press is an *interior* sample of a longer curve and the entry
/// is smoothed through it. Measured when this was written: the conditioned
/// entry reads 14.4° at the marker on the 14.04° line, 0.19 px off the press,
/// where the cold entry reads 12.4° — every bound below is set wide of its
/// measurement, not tight to it. A press nowhere near the trail takes no
/// run-up at all, to the bit.
///
/// Session-level and CPU-only, like the staircase test and for its reason.
#[test]
fn the_run_up_conditions_the_strokes_entry() {
    use stark_engine::Session;
    use stark_engine::ViewTransform;
    use stark_engine::path::{point_at, span_count, span_end};
    use stark_model::document::LayerId;
    use stark_model::geom::Extent2;

    let session = || {
        Session::new(
            ViewTransform::identity(Extent2::new(512, 512)),
            LayerId::ROOT,
        )
    };
    let true_heading = 0.25_f32.atan();

    // The 1-in-4 staircase — the grid's rendering of the 14° line — from
    // `from`, with the rise pattern offset by `phase` so the approach and the
    // drag quantize differently, as two stretches of one line would.
    let stair = |n: usize, from: Vec2, phase: usize, t0: f64| {
        let mut y = 0;
        (0..n)
            .map(|i| {
                if (i + phase) % 4 == 3 {
                    y += 1;
                }
                InputSample {
                    pos: from + Vec2::new(i as f32 + 1.0, y as f32),
                    time: t0 + i as f64 * 0.016,
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>()
    };
    // The watched approach ends where the press begins.
    let hover = stair(32, Vec2::new(-33.0, -8.0), 0, 0.0);
    let after_hover = 32.0 * 0.016;
    let stroke = |s: &mut Session, press: Vec2| {
        s.start_stroke(
            Tool::Brush,
            InputSample {
                pos: press,
                time: after_hover,
                ..Default::default()
            },
            0,
            DEFAULT_TOLERANCE,
            0.0,
        );
        for m in stair(20, press, 1, after_hover + 0.016) {
            s.stroke_to(m);
        }
        s.end_stroke().expect("the drag commits a stroke")
    };
    // The entry's heading: where the deposit begins to where the curve is one
    // span later — the stretch the stroke actually opens with.
    let entry = |rec: &stark_model::document::StrokeRecord| {
        let d = point_at(&rec.path, rec.start + 1.0) - point_at(&rec.path, rec.start);
        d.y.atan2(d.x)
    };

    let press = Vec2::ZERO;
    let mut warm = session();
    for s in &hover {
        warm.hover_to(*s, DEFAULT_TOLERANCE, 50.0);
    }
    let conditioned = stroke(&mut warm, press);

    // The curve extends; the stroke does not. The record's path reaches back
    // to the approach's first report — pinned there, as the curve's head
    // always is — while the marker holds the deposit to the press.
    assert!(
        conditioned.start > 0.0,
        "a run-up seeded stroke must carry a marker past the curve's head"
    );
    assert_eq!(
        conditioned.path[0].pos,
        Vec2::new(-32.0, -8.0),
        "the curve's head is the approach's first report, pinned"
    );
    let entry_point = point_at(&conditioned.path, conditioned.start);
    assert!(
        entry_point.distance(press) < 1.0,
        "the marker sits {} px off the press — the entry is smoothed through \
         the press, not moved away from it",
        entry_point.distance(press),
    );
    // The other pin is untouched: the mark still ends where the hand did.
    let tip = span_end(
        &conditioned.path,
        span_count(conditioned.path.len()).saturating_sub(1),
    );
    assert!(
        tip.distance(Vec2::new(20.0, 5.0)) < 1e-3,
        "the curve's tip must stay pinned to the drag's last report"
    );

    let cold = stroke(&mut session(), press);
    assert_eq!(cold.start, 0.0, "a cold stroke has no run-up to mark off");

    let err_conditioned = (entry(&conditioned) - true_heading).abs();
    let err_cold = (entry(&cold) - true_heading).abs();
    assert!(
        err_conditioned < err_cold,
        "the run-up made the entry worse: {:.1}° vs {:.1}° cold",
        entry(&conditioned).to_degrees(),
        entry(&cold).to_degrees(),
    );
    assert!(
        err_conditioned < 4_f32.to_radians(),
        "the conditioned entry is {:.1}° where the line is {:.1}°",
        entry(&conditioned).to_degrees(),
        true_heading.to_degrees(),
    );

    // A press nowhere near the trail: the window is history from somewhere
    // else, and the stroke fits exactly as if it had begun cold.
    let far = press + Vec2::new(500.0, 0.0);
    let mut teleported = session();
    for s in &hover {
        teleported.hover_to(*s, DEFAULT_TOLERANCE, 50.0);
    }
    let gated = stroke(&mut teleported, far);
    let cold_far = stroke(&mut session(), far);
    assert_eq!(
        (gated.path, gated.start),
        (cold_far.path, cold_far.start),
        "a teleported press still took the stale trail as its run-up"
    );
}

/// The run-up is evidence, never paint: a stroke begun off a watched approach
/// lays its paint from the press on, the approach's trail stays bare, and —
/// because the marker is trimmed in the one flattening funnel both render
/// paths share — the mid-gesture preview already agrees with the commit about
/// all of it, to the bit (§1.3, §6.2).
#[test]
fn the_run_up_is_evidence_never_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
    let bare = engine.render_to_image();

    // A watched approach along −x, ending under the press...
    for i in 0..21 {
        engine.process(report(Vec2::new(-80.0 + 4.0 * i as f32, 0.0)));
    }
    // ...then the press at the origin, dragging on to +40.
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(B),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(40.0, 0.0)),
    });
    // Captured only where it can equal the commit: the tail tint repaints a
    // live stroke magenta by design under `debug-unfrozen`.
    #[cfg(not(feature = "debug-unfrozen"))]
    let live = engine.render_to_image();
    engine.process(GestureCommand::End);
    let committed = engine.render_to_image();

    #[cfg(not(feature = "debug-unfrozen"))]
    {
        assert!(
            images_match(&live, &committed, 0),
            "a run-up stroke's preview is not the commit's own pixels"
        );
        assert!(
            red_dominant(committed.pixel(committed.width / 2 + 20, committed.height / 2)),
            "the drag from the press must lay its paint"
        );
    }
    // Mid-approach, several radii from the press: bare before, bare after.
    for dx in [40, 60] {
        assert_eq!(
            committed.pixel(committed.width / 2 - dx, committed.height / 2),
            bare.pixel(bare.width / 2 - dx, bare.height / 2),
            "the approach's trail was painted, {dx} px behind the press"
        );
    }
}

/// A click at the end of a watched approach is still a click: its curve holds
/// the whole run-up, its marker sits at the very end of it, so it can deposit
/// nothing — and a release that cannot deposit commits nothing (§6.2,
/// `Session::end_stroke`).
#[test]
fn a_click_after_a_watched_approach_commits_nothing() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    engine.process(ViewCommand::set_brush(brush(RED, 10.0)));
    let before = engine.render_to_image();
    let rev = engine.observe().doc_revision;

    hover_ab(&mut engine);
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(B),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::End);

    assert!(
        images_match(&before, &engine.render_to_image(), 0),
        "a hovered click left paint"
    );
    let obs = engine.observe();
    assert!(!obs.can_undo, "a hovered click left something to undo");
    assert_eq!(obs.doc_revision, rev, "a hovered click moved the document");
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
