//! Step-3 golden-image tests (DESIGN.md §9). Render known scripts and compare
//! against committed reference PNGs. Regenerate with `STARK_BLESS=1`.

mod common;

use common::*;
use stark_core::colorspace::ColorSpaceId;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushDynamics, BrushShape, MixerParams, Tool};
use stark_core::geom::Vec2;
use stark_core::SurfaceId;

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

/// The example brush shape, embedded so the test is self-contained.
const BRISTLES: &[u8] = include_bytes!("../../stark-ui/assets/shape/WornBristles.png");

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
    let img = engine.render_to_image(PAPER);
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
    let img = engine.render_to_image(PAPER);
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

    paint(&mut engine, blue, 34.0, &[Vec2::new(-90.0, -70.0), Vec2::new(90.0, 70.0)]);
    paint(&mut engine, yellow, 34.0, &[Vec2::new(-90.0, 70.0), Vec2::new(90.0, -70.0)]);
    paint(&mut engine, white, 30.0, &[Vec2::new(-100.0, 55.0), Vec2::new(100.0, 55.0)]);

    let img = engine.render_to_image(PAPER);
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
    let img = engine.render_to_image(PAPER);
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

    let img = engine.render_to_image(PAPER);
    assert_golden("bristle_stroke", &img, 6);
}

#[test]
fn golden_smear_mixer() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // A committed green bar, then a red Mixer stroke dragged left→right across it:
    // the brush should pick up green where it crosses and carry a fading green
    // tint past the bar (DESIGN.md §6.2). A Dry red stroke would just lay flat red.
    paint(
        &mut engine,
        GREEN,
        38.0,
        &[Vec2::new(0.0, -90.0), Vec2::new(0.0, 90.0)],
    );

    let mut brush = brush(RED, 16.0);
    brush.dynamics = BrushDynamics::Mixer(MixerParams::default());
    engine.process(InputCommand::SetBrush(brush));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-110.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(110.0, 0.0)),
    });
    engine.process(InputCommand::EndStroke);

    let img = engine.render_to_image(PAPER);
    assert_golden("smear_mixer", &img, 6);
}

#[test]
fn golden_lateral_pickup() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // Lateral reservoir bands (DESIGN.md §6.2): a thin horizontal green bar that
    // sits entirely *inside* the upper lateral band of where a wide, hard-edged red
    // Mixer stroke will travel. The brush's upper edge rolls along the green the
    // whole way (picking it up and carrying it) while its center/lower edge never
    // touch it. Because the hard red stroke fully buries the green bar, any green in
    // the result is paint the *upper band* picked up and re-deposited — a single
    // reservoir (one color across the whole tip) could not produce it. The boundary
    // runs *along* the stroke, unlike `smear_mixer` (which crosses it).
    paint(
        &mut engine,
        GREEN,
        9.0,
        &[Vec2::new(-110.0, 17.0), Vec2::new(110.0, 17.0)],
    );

    let mut brush = brush(RED, 30.0);
    brush.hardness = 0.9; // near-solid, so it opaquely covers the green bar
    brush.dynamics = BrushDynamics::Mixer(MixerParams::default());
    engine.process(InputCommand::SetBrush(brush));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-110.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(110.0, 0.0)),
    });
    engine.process(InputCommand::EndStroke);

    let img = engine.render_to_image(PAPER);
    assert_golden("lateral_pickup", &img, 6);
}

#[test]
fn golden_canvas_surface() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // Paint on the linen canvas surface (DESIGN.md §6.4): a light, partial-coverage
    // stroke catches on the weave's peaks (dry-brush tooth), while the bare paper
    // shows the woven relief under raking light. The other goldens stay on the flat
    // default so they test orthogonally. The surface bytes are read from disk and
    // registered (the engine embeds none — the frontend provides them at runtime).
    let linen_png = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../stark-ui/assets/surface/Linen.png"
    ))
    .expect("read surface PNG");
    engine.register_surface(SurfaceId::Linen, linen_png);
    engine.set_surface(SurfaceId::Linen);
    let mut brush = brush(RED, 60.0);
    brush.flow = 0.22; // light, so the tooth reads
    engine.process(InputCommand::SetBrush(brush));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-95.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(95.0, 0.0)),
    });
    engine.process(InputCommand::EndStroke);

    let img = engine.render_to_image(PAPER);
    assert_golden("linen_surface", &img, 6);
}
