//! User-supplied brush-shape import robustness (§6.6).
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
use stark_core::document::{BrushShape, OrientationSource, Tool};
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

/// **The two orientation bakes must agree where they describe the same footprint**
/// (§6.6) — which is the whole of what makes the pen volume's padding a change of
/// representation rather than a change of brush.
///
/// The two are separate bakes now: follow-stroke is the mask as it stands, one layer,
/// while pen is the same mask shrunk into a `PEN_PAD`-wider square, stacked per angle,
/// and read back at a frame scaled to match. At a **relative angle of zero** they stand
/// for the identical tip, so the marks have to land on top of one another — and that is
/// the one comparison that pins every number the padding moves at once. Get the frame
/// wrong and the pen mark is `√2` too wide or too narrow; get `build_prefix_tau`'s `dx`
/// wrong and it is the right size at the wrong darkness. Either shows here as a mark
/// that is not the other mark, and nowhere else as anything but "the nib looks off".
///
/// A mouse stroke along `+x` is the zero-angle case without contrivance: the travel
/// direction and the (absent) tilt azimuth are both 0, so `orientation_turns` returns 0
/// and the pen volume is read at its own slice 0.
///
/// The tolerance is for resampling and nothing else. The pen bake carries the mask
/// through a rotation into a finer grid and the shader reads it back with its own
/// bilinear, so the two differ by what two filter passes do to an edge; they cannot be
/// compared to the bit. Measured, they agree to **2 levels** — while dropping the
/// `PEN_PAD` out of the τ integral misses by 27 and reading the padded volume at an
/// unpadded frame misses by 53, so the bound has half a decade of room on the side that
/// matters and six times the resampling noise on the other.
#[test]
fn the_pen_bake_paints_what_the_follow_stroke_bake_does_at_zero_angle() {
    let Some(mut engine) = engine_or_skip_blue() else {
        return;
    };
    let id = engine.import_brush(&blob_png(160, 160)).expect("import");

    let draw = |engine: &mut stark_core::Engine, orientation| {
        let mut b = brush(RED, 60.0);
        b.shape = BrushShape::Stamp(id);
        b.orientation = orientation;
        // Lift and deposit, so the **dynamics loop** draws this rather than the swept
        // fast path. Both paths read the frame and the volume, and they read them from
        // different sides — one uniform each, filled at different call sites — so a
        // padding that only one of them agreed with would show here and in no golden
        // the corpus draws.
        //
        // It does *not* pin the tool's own `frame_scale` (`dynamics.wesl::exchange`):
        // what that corrects is stranded in the padding, where the deposit's τ-weighted
        // read finds nothing, so the mark comes out within 2 levels either way. Said
        // plainly because the alternative is a test that looks like coverage it has not
        // got.
        b.dynamics.lift = 0.6;
        b.dynamics.deposit = 0.6;
        engine.process(ViewCommand::SetBrush(b));
        engine.process(GestureCommand::Start {
            tool: Tool::Brush,
            sample: InputSample::at(Vec2::new(-70.0, 0.0)),
            tolerance: DEFAULT_TOLERANCE,
        });
        engine.process(GestureCommand::To {
            sample: InputSample::at(Vec2::new(70.0, 0.0)),
        });
        engine.process(GestureCommand::End);
        let img = engine.render_to_image();
        engine.process(stark_core::command::DocCommand::Undo);
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
