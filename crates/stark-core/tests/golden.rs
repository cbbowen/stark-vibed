//! Step-3 golden-image tests (§9). Render known scripts and compare
//! against committed reference PNGs. Regenerate by deleting the existing golden image.

mod common;

use common::*;
use stark_core::SurfaceId;
use stark_core::colorspace::ColorSpaceId;
use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushDynamics, BrushParams, BrushShape, OrientationSource, Tool};
use stark_core::geom::Vec2;
use stark_core::path::DEFAULT_TOLERANCE;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

/// The example brush shape, embedded so the test is self-contained.

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
    let img = engine.render_to_image();
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
    let img = engine.render_to_image();
    assert_golden("two_strokes_cross", &img, 6);
}

#[test]
fn golden_mixbox_mix() {
    let Some(mut engine) = engine_or_skip_with(ColorSpaceId::Mixbox) else {
        return;
    };
    // Blue and yellow crossing → Mixbox mixes them to green at the overlap (where
    // an RGB blend would give a muddy gray); a white bar over the lower half tints.
    let blue = [0.10, 0.30, 0.80, 1.0];
    let yellow = [0.92, 0.82, 0.10, 1.0];
    let white = [0.96, 0.96, 0.96, 1.0];

    paint(
        &mut engine,
        blue,
        34.0,
        &[Vec2::new(-90.0, -70.0), Vec2::new(90.0, 70.0)],
    );
    paint(
        &mut engine,
        yellow,
        34.0,
        &[Vec2::new(-90.0, 70.0), Vec2::new(90.0, -70.0)],
    );
    paint(
        &mut engine,
        white,
        30.0,
        &[Vec2::new(-100.0, 55.0), Vec2::new(100.0, 55.0)],
    );

    let img = engine.render_to_image();
    assert_golden("mixbox_mix", &img, 6);
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
    let img = engine.render_to_image();
    assert_golden("curved_stroke", &img, 6);
}

#[test]
fn golden_bristle_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let id = engine
        .import_brush(&stark_testdata::assets::bristles())
        .expect("import brush shape");

    let mut brush = brush(RED, 70.0);
    brush.shape = BrushShape::Stamp(id);
    brush.drain = 0.0;
    engine.process(ViewCommand::SetBrush(brush));

    // A horizontal stroke; the worn-bristle mask should break up its coverage.
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-90.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(90.0, 0.0)),
    });
    engine.process(GestureCommand::End);

    let img = engine.render_to_image();
    assert_golden("bristle_stroke", &img, 6);
}

#[test]
fn golden_pen_orientation_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let id = engine
        .import_brush(&stark_testdata::assets::bristles())
        .expect("import brush shape");

    // The anisotropic bristle mask in `Pen` orientation: the footprint is pinned to the
    // pen's tilt azimuth (here a constant 45° in canvas space) instead of tracking the
    // stroke tangent. So as this stroke changes direction the bristle streaks keep the
    // *same* world angle — the whole point of arbitrary orientation (§6.6).
    let mut brush = brush(RED, 60.0);
    brush.shape = BrushShape::Stamp(id);
    brush.orientation = OrientationSource::Pen;
    brush.drain = 0.0;
    engine.process(ViewCommand::SetBrush(brush));

    // A fixed tilt azimuth (atan2(1, 1) = 45°) on every sample, while the stroke itself
    // bends from rightward to downward — so travel direction and shape orientation diverge.
    let tilt = Vec2::new(1.0, 1.0);
    let sample = |x: f32, y: f32| InputSample {
        pos: Vec2::new(x, y),
        pressure: 1.0,
        tilt,
        time: 0.0,
    };
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: sample(-80.0, -50.0),
        tolerance: DEFAULT_TOLERANCE,
    });
    for &(x, y) in &[(0.0, -50.0), (60.0, -10.0), (60.0, 70.0)] {
        engine.process(GestureCommand::To {
            sample: sample(x, y),
        });
    }
    engine.process(GestureCommand::End);

    let img = engine.render_to_image();
    assert_golden("pen_orientation_stroke", &img, 6);
}

#[test]
fn golden_canvas_surface() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Paint on the linen canvas surface (§6.4): a light, partial-coverage
    // stroke sits on the weave, and the bare paper
    // shows the woven relief under raking light. The other goldens never register the
    // linen bytes, so they fall back to the flat builtin and test orthogonally. The
    // surface bytes are read from disk and registered (the engine embeds none — the
    // frontend provides them at runtime).
    let linen_png = stark_testdata::assets::linen();
    engine.register_surface(SurfaceId::Linen, linen_png);
    engine.process(DocCommand::SetSurface(SurfaceId::Linen));
    // Turn the weave up: `surface_strength` defaults to 0, which leaves the relief
    // there for paint to sit in but keeps the light from embossing it. This golden is
    // about the embossing, so it has to ask for it.
    engine.process(ViewCommand::SetMediaParams(stark_core::MediaParams {
        surface_strength: 0.6,
        ..Default::default()
    }));
    let mut brush = brush(RED, 60.0);
    brush.drain = 0.005;
    engine.process(ViewCommand::SetBrush(brush));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-95.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(95.0, 0.0)),
    });
    engine.process(GestureCommand::End);

    let img = engine.render_to_image();
    assert_golden("linen_surface", &img, 6);
}

#[test]
fn golden_lift_end_regression() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let brush = BrushParams {
        radius: 80.0,
        shape: BrushShape::Round { hardness: 0.95 },
        drain: 0.005,
        dynamics: BrushDynamics {
            add: 1.0,
            lift: 0.95,
            deposit: 0.95,
            ..BrushDynamics::default()
        },
        ..BrushParams::default()
    };
    for (i, x) in [-200.0, -300.0, -400.0].into_iter().enumerate() {
        stroke_with(
            &mut engine,
            brush,
            &[Vec2::new(x, 0.0), Vec2::new(0.0, 0.0), Vec2::new(30.0, 0.0)],
        );
        let img = engine.render_to_image();
        assert_golden(&format!("lift_end_regression_{i}"), &img, 6);
        engine.process(DocCommand::Undo);
    }
}

#[test]
fn golden_heavy_smear_regression() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let points = [
        Vec2::new(-100.0, 0.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(100.0, 0.0),
    ];
    let shape = BrushShape::Round { hardness: 0.95 };
    stroke_with(
        &mut engine,
        BrushParams {
            radius: 200.0,
            shape,
            dynamics: BrushDynamics {
                add: 1.5,
                ..BrushDynamics::default()
            },
            ..BrushParams::default()
        },
        &points,
    );
    stroke_with(
        &mut engine,
        BrushParams {
            radius: 50.0,
            shape,
            dynamics: BrushDynamics {
                add: 0.0,
                lift: 0.95,
                deposit: 0.95,
                ..BrushDynamics::default()
            },
            ..BrushParams::default()
        },
        &points,
    );
    let img = engine.render_to_image();
    assert_golden("heavy_smear_regression", &img, 6);
    engine.process(DocCommand::Undo);
}

#[test]
fn golden_drained_brush_length_independent() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let brush = BrushParams {
        radius: 80.0,
        shape: BrushShape::Round { hardness: 0.95 },
        drain: 0.005,
        dynamics: BrushDynamics {
            add: 1.0,
            lift: 0.95,
            deposit: 0.95,
            ..BrushDynamics::default()
        },
        ..BrushParams::default()
    };
    // These should all be visually indistiguishable because the brush runs out of paint before the end of the stroke (which is also off the edge of the image).
    for (i, x) in [200.0, 300.0, 400.0, 500.0, 600.0].into_iter().enumerate() {
        stroke_with(
            &mut engine,
            brush,
            &[Vec2::new(-400.0, 0.0), Vec2::new(0.0, 0.0), Vec2::new(x, 0.0)],
        );
        let img = engine.render_to_image();
        assert_golden(&format!("drained_brush_length_independent_{i}"), &img, 6);
        engine.process(DocCommand::Undo);
    }
}
