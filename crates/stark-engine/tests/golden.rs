//! Step-3 golden-image tests (§9). Render known scripts and compare
//! against committed reference PNGs. Regenerate by deleting the existing golden image.
//!
//! **Scripts, not single strokes.** A golden of one stroke with one brush belongs in
//! the corpus ([`corpus.rs`](corpus.rs)), where it is also held to every invariant the
//! battery knows about instead of only to its own pixels; the ones that lived here
//! moved there. What is left is what the corpus's shape cannot express — a *sequence*
//! of strokes and undos, a color space, a lighting environment, a stroke drawn five
//! times at five lengths — plus the marks that are about the media pass rather than
//! about the stroke that made them.

mod common;

use common::*;
use stark_engine::command::Tool;

use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
#[cfg(feature = "mixbox")]
use stark_model::ColorSpaceId;
use stark_model::document::{BrushDynamics, BrushEffect, BrushParams, BrushShape, PaintEffect};
use stark_model::geom::Vec2;

const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.0, 1.0, 0.0];

/// The **image-based** lighting path, on the one golden that runs it (§6.3).
///
/// Everything else in the suite paints under the procedural `Neutral` reference light,
/// so that a rendered byte means what the pipeline produced rather than what a
/// particular sky did to it. But `Ferndale` is a switch away in the Lighting panel and
/// it is the *more* demanding path — an HDR decode, an irradiance convolution and a
/// prefiltered specular chain, none of which `Neutral` runs at all — so it needs
/// somewhere to show. This is it, and it is deliberately the one place: a decoded
/// environment map standing behind every other golden is how a change to the *stroke*
/// path arrives pre-multiplied by a change in the light.
///
/// Ridged paint at a glancing hardness, which is what puts the specular chain to work:
/// a flat patch would come back as little more than the map's average.
#[test]
fn golden_studio_environment() {
    let Some(mut engine) = engine_or_skip_studio() else {
        return;
    };
    engine.process(ViewCommand::SetMediaParams(stark_engine::MediaParams {
        specular: 0.8,
        height_strength: 1.0,
        ..Default::default()
    }));
    let mut b = brush(RED, 34.0);
    b.shape = BrushShape::Round { hardness: 0.55 };
    b.drain = 0.0;
    b.paint_mut().expect("a paint brush").flow = 1.2;
    for (color, y) in [
        ([1.0, 0.35, 0.2], -60.0),
        ([0.2, 0.4, 0.9], 0.0),
        ([0.9, 0.88, 0.8], 60.0),
    ] {
        b.paint_mut().expect("a paint brush").color = color;
        stroke_with(
            &mut engine,
            b,
            &[Vec2::new(-95.0, y - 18.0), Vec2::new(95.0, y + 18.0)],
        );
    }
    let img = engine.render_to_image();
    assert_golden("studio_environment", &img, 6);
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

/// Mixbox-only, so it exists only in a build carrying the `mixbox` feature.
/// `ColorSpaceId::Mixbox` still *names* a space there — the save format's enum
/// indices cannot depend on a feature (§8) — but nothing can open one.
#[cfg(feature = "mixbox")]
#[test]
fn golden_mixbox_mix() {
    let Some(mut engine) = engine_or_skip_with(ColorSpaceId::Mixbox) else {
        return;
    };
    // Blue and yellow crossing → Mixbox mixes them to green at the overlap (where
    // an RGB blend would give a muddy gray); a white bar over the lower half tints.
    let blue = [0.10, 0.30, 0.80];
    let yellow = [0.92, 0.82, 0.10];
    let white = [0.96, 0.96, 0.96];

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
    engine.process(ViewCommand::set_brush(brush));

    // A horizontal stroke; the worn-bristle mask should break up its coverage.
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-90.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(90.0, 0.0)),
    });
    engine.process(GestureCommand::End);

    let img = engine.render_to_image();
    assert_golden("bristle_stroke", &img, 6);
}

#[test]
fn golden_canvas_substrate() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Paint on the linen canvas substrate (§6.4): a light, partial-coverage
    // stroke sits on the substrate, and the bare paper
    // shows the woven relief under raking light. The other goldens never register the
    // linen bytes, so they fall back to the flat builtin and test orthogonally. The
    // substrate bytes are read from disk and registered (the engine embeds none — the
    // frontend provides them at runtime).
    let linen = engine
        .import_substrate(&stark_testdata::assets::linen())
        .expect("the linen height map imports");
    engine.process(DocCommand::SetSubstrate(linen));
    // Turn the substrate up: `substrate_strength` defaults to 0, which leaves the relief
    // there for paint to sit in but keeps the light from embossing it. This golden is
    // about the embossing, so it has to ask for it.
    engine.process(ViewCommand::SetMediaParams(stark_engine::MediaParams {
        substrate_strength: 0.6,
        ..Default::default()
    }));
    let mut brush = brush(RED, 60.0);
    // Per radius (§6.2): 0.3 over a 60px tip is the 0.005 per canvas px this was.
    brush.drain = 0.3;
    engine.process(ViewCommand::set_brush(brush));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-95.0, 0.0)),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    engine.process(GestureCommand::To {
        sample: InputSample::at(Vec2::new(95.0, 0.0)),
    });
    engine.process(GestureCommand::End);

    let img = engine.render_to_image();
    assert_golden("linen_substrate", &img, 6);
}

#[test]
fn golden_lift_end_regression() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let brush = BrushParams {
        size: 80.0,
        shape: BrushShape::Round { hardness: 0.95 },
        // 0.4 per radius = 0.005 per canvas px at this tip: 200px to bone dry.
        drain: 0.4,
        // Black, as `BrushParams::default()` always painted here — the blessed
        // pixels are of a black smear.
        effect: BrushEffect::wet_with(
            [0.0; 3],
            BrushDynamics {
                flow: 1.0,
                lift: 0.95,
                deposit: 0.95,
                ..BrushDynamics::default()
            },
        ),
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
            size: 200.0,
            shape,
            effect: BrushEffect::Paint(PaintEffect {
                flow: 1.5,
                ..Default::default()
            }),
            ..BrushParams::default()
        },
        &points,
    );
    stroke_with(
        &mut engine,
        BrushParams {
            size: 50.0,
            shape,
            effect: BrushEffect::wet_with(
                [0.0; 3],
                BrushDynamics {
                    flow: 0.0,
                    lift: 0.95,
                    deposit: 0.95,
                    ..BrushDynamics::default()
                },
            ),
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
        size: 80.0,
        shape: BrushShape::Round { hardness: 0.95 },
        // 0.4 per radius = 0.005 per canvas px at this tip: 200px to bone dry.
        drain: 0.4,
        // Black, as `BrushParams::default()` always painted here — the blessed
        // pixels are of a black smear.
        effect: BrushEffect::wet_with(
            [0.0; 3],
            BrushDynamics {
                flow: 1.0,
                lift: 0.95,
                deposit: 0.95,
                ..BrushDynamics::default()
            },
        ),
        ..BrushParams::default()
    };
    // These should all be visually indistiguishable because the brush runs out of paint before the end of the stroke (which is also off the edge of the image).
    for (i, x) in [200.0, 300.0, 400.0, 500.0, 600.0].into_iter().enumerate() {
        stroke_with(
            &mut engine,
            brush,
            &[
                Vec2::new(-400.0, 0.0),
                Vec2::new(0.0, 0.0),
                Vec2::new(x, 0.0),
            ],
        );
        let img = engine.render_to_image();
        assert_golden(&format!("drained_brush_length_independent_{i}"), &img, 6);
        engine.process(DocCommand::Undo);
    }
}

#[test]
fn golden_straight_smear_into_paint() {
    let Some(mut engine) = engine_or_skip_studio() else {
        return;
    };
    engine.process(ViewCommand::SetMediaParams(stark_engine::MediaParams {
        specular: 0.5,
        height_strength: 1.0,
        ..Default::default()
    }));

    let color = [0.5, 0.0, 0.0];
    let shape = BrushShape::Round { hardness: 0.95 };

    stroke_with(
        &mut engine,
        BrushParams {
            size: 256.0,
            shape,
            effect: BrushEffect::Paint(PaintEffect {
                color,
                flow: 2.0,
                ..Default::default()
            }),
            ..Default::default()
        },
        &[Vec2::new(-256.0, 0.0), Vec2::new(256.0, 0.0)],
    );

    stroke_with(
        &mut engine,
        BrushParams {
            size: 64.0,
            shape,
            effect: BrushEffect::wet_with(
                color,
                BrushDynamics {
                    flow: 0.0,
                    lift: 0.5,
                    deposit: 0.95,
                    ..Default::default()
                },
            ),
            ..Default::default()
        },
        &[Vec2::new(-256.0, 0.0), Vec2::new(0.0, 0.0)],
    );

    let img = engine.render_to_image();
    assert_golden("straight_smear_into_paint", &img, 6);
}

#[test]
fn golden_wiggly_smear_into_paint() {
    let Some(mut engine) = engine_or_skip_studio() else {
        return;
    };
    engine.process(ViewCommand::SetMediaParams(stark_engine::MediaParams {
        specular: 0.5,
        height_strength: 1.0,
        ..Default::default()
    }));

    let color = [0.5, 0.0, 0.0];
    let shape = BrushShape::Round { hardness: 0.95 };

    stroke_with(
        &mut engine,
        BrushParams {
            size: 256.0,
            shape,
            effect: BrushEffect::Paint(PaintEffect {
                color,
                flow: 2.0,
                ..Default::default()
            }),
            ..Default::default()
        },
        &[Vec2::new(-256.0, 0.0), Vec2::new(256.0, 0.0)],
    );

    let points: Vec<_> = (0..128)
        .map(|i| {
            let t = (i as f32) / 128.0;
            Vec2::new(-256.0, 0.0) + Vec2::new(t * 256.0, 5.0 * (128.0 * t).sin())
        })
        .collect();

    stroke_with(
        &mut engine,
        BrushParams {
            size: 64.0,
            shape,
            effect: BrushEffect::wet_with(
                color,
                BrushDynamics {
                    flow: 0.0,
                    lift: 0.5,
                    deposit: 0.95,
                    ..Default::default()
                },
            ),
            ..Default::default()
        },
        &points,
    );

    let img = engine.render_to_image();
    assert_golden("wiggly_smear_into_paint", &img, 6);
}
