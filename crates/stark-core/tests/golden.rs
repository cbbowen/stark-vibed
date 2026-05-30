//! Step-3 golden-image tests (DESIGN.md §9). Render known scripts and compare
//! against committed reference PNGs. Regenerate with `STARK_BLESS=1`.

mod common;

use common::*;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushShape, Tool};
use stark_core::geom::Vec2;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

/// The example brush shape, embedded so the test is self-contained.
const BRISTLES: &[u8] = include_bytes!("../../../resources/shapes/WornBristles.png");

#[test]
fn golden_single_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        RED,
        40.0,
        &[
            Vec2::new(-30.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 0.0),
        ],
    );
    let img = engine.render_to_image(BG);
    assert_golden("single_stroke", &img, 6);
}

#[test]
fn golden_two_strokes_cross() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(
        &mut engine,
        RED,
        24.0,
        &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)],
    );
    paint(
        &mut engine,
        GREEN,
        24.0,
        &[Vec2::new(0.0, -60.0), Vec2::new(0.0, 60.0)],
    );
    let img = engine.render_to_image(BG);
    assert_golden("two_strokes_cross", &img, 6);
}

#[test]
fn golden_curved_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // A coarse zigzag of control points; cubic interpolation should render a
    // smooth curve through them rather than sharp polyline corners.
    paint(
        &mut engine,
        RED,
        18.0,
        &[
            Vec2::new(-90.0, 40.0),
            Vec2::new(-45.0, -50.0),
            Vec2::new(0.0, 40.0),
            Vec2::new(45.0, -50.0),
            Vec2::new(90.0, 40.0),
        ],
    );
    let img = engine.render_to_image(BG);
    assert_golden("curved_stroke", &img, 6);
}

#[test]
fn golden_bristle_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let id = engine.import_brush(BRISTLES).expect("import brush shape");

    let mut brush = brush(RED, 70.0);
    brush.shape = BrushShape::Stamp(id);
    brush.spacing = 0.06; // dense so the bristle texture reads as a continuous stroke
    brush.drain = 0.0;
    engine.process(InputCommand::SetBrush(brush));

    // A horizontal stroke; the worn-bristle mask should break up its coverage.
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-90.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(90.0, 0.0)),
    });
    engine.process(InputCommand::EndStroke);

    let img = engine.render_to_image(BG);
    assert_golden("bristle_stroke", &img, 6);
}
