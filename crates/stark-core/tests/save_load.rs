//! Step-5 save/load + timelapse tests (DESIGN.md §8).
//!
//! Completes the replay-equivalence set from step 3: save → load reproduces the
//! exact pixels, undo works after loading, the saved file is compact, and a
//! timelapse yields one frame per action ending at the final image.

mod common;

use common::*;
use stark_core::command::{InputCommand as Cmd, InputSample};
use stark_core::document::{BrushShape, Tool};
use stark_core::geom::Vec2;
use stark_core::{Engine, InputCommand};

const BRISTLES: &[u8] = include_bytes!("../../../resources/shapes/WornBristles.png");

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
    let Some(mut original) = engine_or_skip() else {
        return;
    };
    paint_two(&mut original);
    let before = original.render_to_image(BG);
    let bytes = original.save_bytes().expect("serialize");

    let mut loaded = engine_or_skip().expect("adapter available (original built)");
    loaded.load_bytes(&bytes).expect("deserialize + replay");
    let after = loaded.render_to_image(BG);

    assert!(
        images_match(&before, &after, 0),
        "save → load must reproduce identical pixels"
    );
    // The full undo timeline is available after loading (undo-after-load).
    assert!(loaded.observe().can_undo);
}

#[test]
fn undo_after_load_drops_last_stroke() {
    let Some(mut original) = engine_or_skip() else {
        return;
    };
    paint_two(&mut original);
    let bytes = original.save_bytes().expect("serialize");

    // Reference: a document that only ever had stroke A.
    let mut just_a = engine_or_skip().expect("adapter");
    paint(&mut just_a, RED, 30.0, STROKE_A);
    let only_a = just_a.render_to_image(BG);

    // Load both strokes, then undo the second.
    let mut loaded = engine_or_skip().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    loaded.process(InputCommand::Undo);
    let undone = loaded.render_to_image(BG);

    assert!(
        images_match(&undone, &only_a, 0),
        "undo after load must drop the last stroke exactly"
    );
}

#[test]
fn timelapse_yields_one_frame_per_action() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint_two(&mut engine);
    let file = engine.document_file();
    let final_image = engine.render_to_image(BG);

    let mut frames = Vec::new();
    engine.replay_timelapse(&file, BG, |frame| frames.push(frame));

    assert_eq!(frames.len(), file.actions.len(), "one frame per action");
    assert!(
        images_match(frames.last().unwrap(), &final_image, 0),
        "last timelapse frame must equal the fully-replayed image"
    );
}

#[test]
fn brush_assets_survive_save_load() {
    let Some(mut original) = engine_or_skip() else {
        return;
    };
    // Paint with an image brush shape (the asset lives only in this engine).
    let id = original.import_brush(BRISTLES).expect("import");
    let mut brush = brush(RED, 60.0);
    brush.shape = BrushShape::Stamp(id);
    brush.spacing = 0.08;
    original.process(Cmd::SetBrush(brush));
    original.process(Cmd::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-70.0, 0.0)),
    });
    original.process(Cmd::StrokeTo {
        sample: InputSample::at(Vec2::new(70.0, 0.0)),
    });
    original.process(Cmd::EndStroke);
    let before = original.render_to_image(BG);

    let bytes = original.save_bytes().expect("serialize");

    // A fresh engine that never imported the brush must still reproduce it,
    // because the asset is bundled in the file.
    let mut loaded = engine_or_skip().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    let after = loaded.render_to_image(BG);

    assert!(
        images_match(&before, &after, 0),
        "image-brush stroke must round-trip through save/load via bundled assets"
    );
}

#[test]
fn saved_file_is_compact() {
    let Some(mut engine) = engine_or_skip() else {
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
