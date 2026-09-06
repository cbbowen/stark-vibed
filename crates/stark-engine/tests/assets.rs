//! User-supplied brush-shape import robustness (§6.6).
//!
//! Arbitrary user images must import safely: oversized sources are capped to
//! [`stark_model::MAX_SHAPE_DIM`] (a raw 4096² upload would otherwise
//! exceed the device texture limit), the canonical stored PNG is a fixed point
//! of import (re-importing it yields the same id), and a stroke drawn with an
//! oversized-source shape survives save/load in a fresh engine.

mod common;

use common::palette::RED_SOFT;
use common::*;
use stark_engine::command::Tool;
use stark_engine::command::{GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_model::MAX_SHAPE_DIM;
use stark_model::document::{BrushShape, OrientationSource};
use stark_model::geom::Vec2;

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
    let mut brush = brush(RED_SOFT, 60.0);
    brush.shape = BrushShape::Stamp(id);
    original.process(ViewCommand::set_brush(brush));
    original.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-70.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
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

/// **The two orientation bakes must agree where they describe the same footprint**
/// (§6.6) — which is the whole of what makes the pen volume a change of
/// representation rather than a change of brush.
///
/// The two are separate bakes: follow-stroke is the mask as it stands, one layer,
/// while pen is the same mask rotated per relative angle on its own grid
/// (`assets::rotate_layers` — sound unpadded because a canonical mask's content lies
/// inside its inscribed disc, which a rotation maps to itself). At a **relative angle
/// of zero** they stand for the identical tip, so the marks have to land on top of
/// one another — the one comparison that pins the rotation arithmetic, the τ
/// integral's column width and the frame agreement at once. Any of them wrong shows
/// here as a mark that is not the other mark, and nowhere else as anything but "the
/// tip looks off".
///
/// A mouse stroke along `+x` is the zero-angle case without contrivance: the travel
/// direction and the (absent) tilt azimuth are both 0, so `orientation_turns` returns 0
/// and the pen volume is read at its own slice 0.
///
/// The tolerance is for resampling and nothing else. The pen bake carries the mask
/// through slice 0's rotation arithmetic and the shader reads it back with its own
/// bilinear, so the two differ by what two filter passes do to an edge; they cannot
/// be compared to the bit. In the padded era the same comparison measured 2 levels of
/// resampling noise against misses of 27 and 53 for the two ways the padding could be
/// dropped on one side only, so the bound has half a decade of room on the side that
/// matters.
#[test]
fn the_pen_bake_paints_what_the_follow_stroke_bake_does_at_zero_angle() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let id = engine.import_brush(&blob_png(160, 160)).expect("import");

    let draw = |engine: &mut stark_engine::Engine, orientation| {
        let mut b = brush(RED_SOFT, 60.0);
        b.shape = BrushShape::Stamp(id);
        b.orientation = orientation;
        // Lift and deposit, so the **dynamics loop** draws this rather than the swept
        // fast path. Both paths read the frame and the volume, and they read them from
        // different sides — one uniform each, filled at different call sites — so a
        // bake that only one of them agreed with would show here and in no golden
        // the corpus draws.
        b.make_wet().dynamics.lift = 0.6;
        b.make_wet().dynamics.deposit = 0.6;
        engine.process(ViewCommand::set_brush(b));
        engine.process(GestureCommand::Start {
            tool: Tool::Brush,
            sample: InputSample::at(Vec2::new(-70.0, 0.0)),
            tolerance: DEFAULT_TOLERANCE,
            rope: 0.0,
        });
        engine.process(GestureCommand::To {
            sample: InputSample::at(Vec2::new(70.0, 0.0)),
        });
        engine.process(GestureCommand::End);
        let img = engine.render_to_image();
        engine.process(stark_engine::command::DocCommand::Undo);
        img
    };

    let follow = draw(&mut engine, OrientationSource::FollowStroke);
    let pen = draw(&mut engine, OrientationSource::Pen);

    let worst = (0..follow.height)
        .flat_map(|y| (0..follow.width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let (a, b) = (follow.pixel(x, y), pen.pixel(x, y));
            (0..3)
                .map(|c| (a[c] as i32 - b[c] as i32).abs())
                .max()
                .expect("three channels")
        })
        .max()
        .expect("a viewport of pixels");
    assert!(
        worst <= 12,
        "the pen bake's zero-angle mark is {worst} levels off the follow-stroke bake's \
         — the padded volume is not standing for the same tip"
    );
}
