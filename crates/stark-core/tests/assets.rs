//! User-supplied brush-shape import robustness (DESIGN.md §6.6).
//!
//! Arbitrary user images must import safely: oversized sources are capped to
//! [`stark_core::assets::MAX_SHAPE_DIM`] (a raw 4096² upload would otherwise
//! exceed the device texture limit), the canonical stored PNG is a fixed point
//! of import (re-importing it yields the same id), and a stroke drawn with an
//! oversized-source shape survives save/load in a fresh engine.

mod common;

use common::*;
use stark_core::assets::MAX_SHAPE_DIM;
use stark_core::command::{GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushShape, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;

const RED: [f32; 4] = [0.85, 0.1, 0.1, 1.0];

/// Encode a grayscale PNG of the given size with a soft radial blob — a stand-in
/// for a user's scanned/painted brush-shape image.
fn blob_png(width: u32, height: u32) -> Vec<u8> {
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let r = cx.min(cy);
    let mut pixels = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / r;
            pixels[(y * width + x) as usize] = (255.0 * (1.0 - d).clamp(0.0, 1.0)) as u8;
        }
    }
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(&pixels).expect("png data");
    }
    out
}

fn decode_dims(png_bytes: &[u8]) -> (u32, u32) {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let reader = decoder.read_info().expect("canonical png decodes");
    let info = reader.info();
    (info.width, info.height)
}

#[test]
fn oversized_imports_are_capped_and_canonical() {
    let Some(engine) = engine_or_skip_blue() else {
        return;
    };
    // Non-square and larger than the device limit on one edge.
    let id = engine
        .import_brush(&blob_png(2600, 1300))
        .expect("oversized import succeeds");

    let bytes = engine.asset_bytes(id).expect("canonical bytes retained");
    let (w, h) = decode_dims(&bytes);
    assert!(
        w <= MAX_SHAPE_DIM && h <= MAX_SHAPE_DIM,
        "stored shape {w}×{h} must fit within {MAX_SHAPE_DIM}"
    );
    assert_eq!(
        (w, h),
        (2600 / 3, 1300 / 3),
        "integer box downsample by the smallest sufficient factor"
    );

    // The canonical PNG is a fixed point: importing it again maps to the same
    // id (this is what makes save→load→save stable and peers converge).
    let again = engine.import_brush(&bytes).expect("re-import");
    assert_eq!(id, again, "canonical bytes must re-import to the same id");
}

#[test]
fn a_capped_shape_paints_and_survives_save_load() {
    let Some(mut original) = engine_or_skip_blue() else {
        return;
    };
    let id = original
        .import_brush(&blob_png(2600, 1300))
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
    let mut loaded = engine_or_skip_blue().expect("adapter");
    loaded.load_bytes(&bytes).expect("load");
    let after = loaded.render_to_image();

    assert!(
        images_match(&before, &after, 0),
        "capped-shape stroke must round-trip via the bundled canonical asset"
    );
}
