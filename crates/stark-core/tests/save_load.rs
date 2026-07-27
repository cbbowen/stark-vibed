//! Step-5 save/load + timelapse tests (DESIGN.md §8).
//!
//! Completes the replay-equivalence set from step 3: save → load reproduces the
//! exact pixels, undo works after loading, the saved file is compact, and a
//! timelapse yields one frame per action ending at the final image.

mod common;

use common::*;
use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushShape, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;
use stark_core::{Engine, SurfaceId};

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];
const GREEN: [f32; 4] = [0.1, 0.8, 0.2, 1.0];

const STROKE_A: &[Vec2] = &[Vec2::new(-40.0, -20.0), Vec2::new(40.0, 20.0)];
const STROKE_B: &[Vec2] = &[Vec2::new(-40.0, 40.0), Vec2::new(40.0, -40.0)];

fn paint_two(engine: &mut Engine) {
    paint(engine, RED, 30.0, STROKE_A);
    paint(engine, GREEN, 30.0, STROKE_B);
}

#[test]
fn save_load_roundtrip_is_lossless() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    paint_two(&mut original);
    let before = original.render_to_image();
    let bytes = original.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip_blue().expect("adapter available (original built)");
    loaded.load_bytes(&bytes).expect("deserialize + replay");
    let after = loaded.render_to_image();

    assert!(
        images_match(&before, &after, 0),
        "save → load must reproduce identical pixels"
    );
    // The full undo timeline is available after loading (undo-after-load).
    assert!(loaded.observe().can_undo);
}

#[test]
fn undo_after_load_drops_last_stroke() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    paint_two(&mut original);
    let bytes = original.save_bytes().expect("serialize");

    // Reference: a document that only ever had stroke A.
    let mut just_a = engine_or_skip_blue().expect("adapter");
    paint(&mut just_a, RED, 30.0, STROKE_A);
    let only_a = just_a.render_to_image();

    // Load both strokes, then undo the second.
    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    loaded.process(DocCommand::Undo);
    let undone = loaded.render_to_image();

    assert!(
        images_match(&undone, &only_a, 0),
        "undo after load must drop the last stroke exactly"
    );
}

#[test]
fn timelapse_yields_one_frame_per_action() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    paint_two(&mut engine);
    let file = engine.document_file();
    let final_image = engine.render_to_image();

    let mut frames = Vec::new();
    engine.replay_timelapse(&file, |frame| frames.push(frame));

    assert_eq!(frames.len(), file.actions.len(), "one frame per action");
    assert!(
        images_match(frames.last().unwrap(), &final_image, 0),
        "last timelapse frame must equal the fully-replayed image"
    );
}

#[test]
fn brush_assets_survive_save_load() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    // Paint with an image brush shape (the asset lives only in this engine).
    let id = original
        .import_brush(&stark_testdata::assets::bristles())
        .expect("import");
    let mut brush = brush(RED, 60.0);
    brush.shape = BrushShape::Stamp(id);
    original.process(ViewCommand::SetBrush(brush));
    original.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-70.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    original.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(70.0, 0.0)),
    });
    original.process(GestureCommand::End);
    let before = original.render_to_image();

    let bytes = original.save_bytes().expect("serialize");

    // A fresh engine that never imported the brush must still reproduce it,
    // because the asset is bundled in the file.
    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    let after = loaded.render_to_image();

    assert!(
        images_match(&before, &after, 0),
        "image-brush stroke must round-trip through save/load via bundled assets"
    );
}

#[test]
fn saved_file_is_compact() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    // A long, many-sample stroke — the kind that could bloat a file.
    let path: Vec<Vec2> = (0..400)
        .map(|i| Vec2::new(-100.0 + i as f32 * 0.5, (i as f32 * 0.2).sin() * 30.0))
        .collect();
    paint(&mut engine, RED, 12.0, &path);

    let bytes = engine.save_bytes().expect("serialize");
    let rendered_bytes = (SIZE.width * SIZE.height * 4) as usize;
    assert!(
        bytes.len() < rendered_bytes,
        "action log ({}) should be far smaller than a raster ({rendered_bytes})",
        bytes.len()
    );
    eprintln!("400-sample stroke document: {} bytes", bytes.len());
}

/// The canvas surface is document state (DESIGN.md §6.4), so a mid-document switch
/// is a logged action: it survives save/load, and undo takes it back.
///
/// `CanvasMeta::surface` records the surface the log *starts* from; the switch
/// itself rides in the log, which is why loading has to replay to learn the
/// current one rather than reading it off the header.
#[test]
fn a_surface_switch_is_historized() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let start = engine.surface();
    assert_ne!(start, SurfaceId::Linen, "test needs a surface to switch to");

    paint(
        &mut engine,
        RED,
        20.0,
        &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
    );
    engine.process(DocCommand::SetSurface(SurfaceId::Linen));
    assert_eq!(engine.surface(), SurfaceId::Linen);

    // Undo reaches it, because it is an action like any other.
    engine.process(DocCommand::Undo);
    assert_eq!(engine.surface(), start, "undo must revert the surface");
    engine.process(DocCommand::Redo);
    assert_eq!(engine.surface(), SurfaceId::Linen);

    // And it round-trips: the header still says the document *started* on `start`.
    let file = engine.document_file();
    assert_eq!(
        file.canvas.surface, start,
        "CanvasMeta records the initial surface, not the current one"
    );
    let bytes = engine.save_bytes().expect("serialize");

    let Some(mut loaded) = engine_or_skip_blue() else {
        return;
    };
    loaded.load_bytes(&bytes).expect("load");
    assert_eq!(
        loaded.surface(),
        SurfaceId::Linen,
        "replaying the log must land on the switched-to surface"
    );
}
