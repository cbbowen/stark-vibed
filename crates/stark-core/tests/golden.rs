//! Step-3 golden-image tests (DESIGN.md §9). Render known scripts and compare
//! against committed reference PNGs. Regenerate by deleting the existing golden image.

mod common;

use common::*;
use stark_core::colorspace::ColorSpaceId;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{
    BrushDynamics, BrushShape, OrientationSource, StrokeRecord, Tool,
};
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
fn golden_pen_orientation_stroke() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let id = engine.import_brush(BRISTLES).expect("import brush shape");

    // The anisotropic bristle mask in `Pen` orientation: the footprint is pinned to the
    // pen's tilt azimuth (here a constant 45° in canvas space) instead of tracking the
    // stroke tangent. So as this stroke changes direction the bristle streaks keep the
    // *same* world angle — the whole point of arbitrary orientation (DESIGN.md §6.6).
    let mut brush = brush(RED, 60.0);
    brush.shape = BrushShape::Stamp(id);
    brush.orientation = OrientationSource::Pen;
    brush.spacing = 0.06;
    brush.drain = 0.0;
    engine.process(InputCommand::SetBrush(brush));

    // A fixed tilt azimuth (atan2(1, 1) = 45°) on every sample, while the stroke itself
    // bends from rightward to downward — so travel direction and shape orientation diverge.
    let tilt = Vec2::new(1.0, 1.0);
    let sample = |x: f32, y: f32| InputSample {
        pos: Vec2::new(x, y),
        pressure: 1.0,
        tilt,
        time: 0.0,
    };
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: sample(-80.0, -50.0),
    });
    for &(x, y) in &[(0.0, -50.0), (60.0, -10.0), (60.0, 70.0)] {
        engine.process(InputCommand::StrokeTo { sample: sample(x, y) });
    }
    engine.process(InputCommand::EndStroke);

    let img = engine.render_to_image(PAPER);
    assert_golden("pen_orientation_stroke", &img, 6);
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

    let b = dry_brush(RED, 16.0, BrushDynamics { add: 0.1, load: 0.5, deposit: 0.5, ridge: 0.0, ..Default::default() });
    stroke_with(&mut engine, b, &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("smear_mixer", &img, 6);
}

#[test]
fn golden_knife_scrape() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // A committed solid red field, then a clean palette-knife scrape dragged
    // *across* it: the knife subtractively removes paint along its path (the
    // mutable-medium write-back, DESIGN.md §6.2), thinning the red toward the bare
    // paper where it passes — something the additive deposit fundamentally cannot do.
    paint(
        &mut engine,
        RED,
        60.0,
        &[Vec2::new(-95.0, 0.0), Vec2::new(95.0, 0.0)],
    );

    // Explicit (not `::default()`) so this "clean scrape" golden is stable regardless
    // of the default knife's feel: hard bite, no film, no carry, no ridge.
    let knife = dry_brush(RED, 22.0, BrushDynamics {
        add: 0.0,
        load: 0.8,
        deposit: 0.0, // lift onto the tool, lay nothing back → a clean scrape/erase
        ridge: 0.0, ..Default::default() });
    stroke_with(&mut engine, knife, &[Vec2::new(0.0, -80.0), Vec2::new(0.0, 80.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("knife_scrape", &img, 6);
}

#[test]
fn golden_knife_tooth() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // The knife's tooth interaction on the linen weave (DESIGN.md §6.2, Phase 1):
    // a solid red field, then a high-tooth knife scraped across it. Unlike the
    // clean scrape on flat (`knife_scrape`, uniform thinning), the tooth gate
    // clears paint off the weave's peaks but leaves it in the valleys, so the
    // scraped band reads the linen texture — the weave is *revealed* by scraping.
    let linen_png = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../stark-ui/assets/surface/Linen.png"
    ))
    .expect("read surface PNG");
    engine.register_surface(SurfaceId::Linen, linen_png);
    engine.set_surface(SurfaceId::Linen);

    paint(
        &mut engine,
        RED,
        60.0,
        &[Vec2::new(-95.0, 0.0), Vec2::new(95.0, 0.0)],
    );

    // Explicit params so the tooth-reveal golden is stable against the default knife.
    let mut knife = dry_brush(RED, 26.0, BrushDynamics {
        add: 0.0,
        load: 0.8,
        deposit: 0.0, // lift onto the tool, lay nothing back → a clean scrape/erase
        ridge: 0.0, ..Default::default() });
    knife.tooth = 1.0; // full tooth: the gate bites only the weave's peaks
    stroke_with(&mut engine, knife, &[Vec2::new(0.0, -80.0), Vec2::new(0.0, 80.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("knife_tooth", &img, 6);
}

#[test]
fn golden_knife_carry() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // Conservative reservoir carry (DESIGN.md §6.2, Phase 1): a green vertical bar
    // on the left, then a knife with `carry` dragged left→right across it onto a long
    // bare runway. The knife scrapes green off the bar (lightening the source) into
    // its reservoir and re-lays it downstream, so green is *dragged* across the bare
    // paper to the right and fades out as the load depletes — paint a clean scrape
    // (`golden_knife_scrape`) would simply have destroyed.
    paint(
        &mut engine,
        GREEN,
        30.0,
        &[Vec2::new(0.0, -80.0), Vec2::new(0.0, 80.0)],
    );

    let knife = dry_brush(RED, 40.0, BrushDynamics {
        add: 0.0,
        load: 0.9,     // lift green onto the tool (lightening the source)
        deposit: 0.12, // slowly re-lay it onto the bare runway, dragging a long fading tail
        ridge: 0.0, ..Default::default() });
    stroke_with(&mut engine, knife, &[Vec2::new(-90.0, 0.0), Vec2::new(90.0, 0.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("knife_carry", &img, 6);
}

#[test]
fn golden_knife_ridge() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // Lateral pile-up (DESIGN.md §6.2, Phase 1): a solid red field, then a knife
    // with high `ridge` scraped down through it. Paint shoved aside heaps into the
    // footprint's edge band, raising lips along both sides of the scraped channel —
    // height-only relief, so it reads as raking-light highlight/shadow ridges in the
    // impasto media pass (a clean scrape, `golden_knife_scrape`, leaves flat paint).
    paint(
        &mut engine,
        RED,
        70.0,
        &[Vec2::new(-95.0, 0.0), Vec2::new(95.0, 0.0)],
    );

    let knife = dry_brush(RED, 30.0, BrushDynamics {
        add: 0.0,
        load: 0.15, // gentle load: keep the colour ~uniform so the lip reads as relief
        deposit: 0.0,
        ridge: 1.0, ..Default::default() });
    stroke_with(&mut engine, knife, &[Vec2::new(0.0, -80.0), Vec2::new(0.0, 80.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("knife_ridge", &img, 6);
}

#[test]
fn golden_smudge_paint() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // The unified Dry brush running all three axes at once (DESIGN.md §6.2): a green
    // bar, then a red stroke that simultaneously *lifts* (picking green up and lightening
    // the bar), *deposits* (dragging green along the path), and *adds* (lays its own red) —
    // a single brush doing what previously needed the knife and mixer together. The result
    // past the bar is a fading red↔green blend over partly-lifted paper, not reachable by
    // any one axis.
    paint(
        &mut engine,
        GREEN,
        38.0,
        &[Vec2::new(0.0, -90.0), Vec2::new(0.0, 90.0)],
    );

    let b = dry_brush(RED, 18.0, BrushDynamics { add: 0.4, load: 0.7, deposit: 0.4, ridge: 0.0, ..Default::default() });
    stroke_with(&mut engine, b, &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("smudge_paint", &img, 6);
}

/// Visualize the brush-dynamics reservoir for diagnosis (DESIGN.md §6.2): each reservoir
/// column (along the stroke) × band (across the tip), scaled up into blocks. Two strips:
/// the **net height transfer** per column (the conserved lift/deposit — diverging colour,
/// red = deposits paint, blue = lifts it, grey = neutral) and the **deposit opacity**
/// (the tool's per-unit alpha, greyscale). Discontinuities here show as column-aligned
/// seams, making per-segment artifacts legible.
fn reservoir_image(color: &[f32], aux: &[f32], w: u32, bands: u32) -> stark_core::RgbaImage {
    let scale = 4u32;
    let net_max = (0..(w * bands) as usize)
        .map(|i| aux[i * 4].abs())
        .fold(1e-4f32, f32::max);
    let strips = 2u32;
    let (ow, oh) = (w * scale, bands * strips * scale);
    let mut px = vec![0u8; (ow * oh * 4) as usize];
    for oy in 0..oh {
        for ox in 0..ow {
            let col = ox / scale;
            let strip = (oy / scale) / bands;
            let band = (oy / scale) % bands;
            let idx = (band * w + col) as usize;
            let rgba = if strip == 0 {
                let n = (aux[idx * 4] / net_max).clamp(-1.0, 1.0);
                if n >= 0.0 {
                    [40 + (n * 215.0) as u8, 40, 40, 255]
                } else {
                    [40, 40, 40 + (-n * 215.0) as u8, 255]
                }
            } else {
                let a = (color[idx * 4 + 3].clamp(0.0, 1.0) * 255.0) as u8;
                [a, a, a, 255]
            };
            let o = ((oy * ow + ox) * 4) as usize;
            px[o..o + 4].copy_from_slice(&rgba);
        }
    }
    stark_core::RgbaImage::new(ow, oh, px)
}

#[test]
fn golden_reservoir_viz() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    // The same setup as `golden_dry_smear_artifact`, but visualizing the reservoir the
    // deposit samples rather than the rendered canvas — so a per-column discontinuity in
    // lift/deposit is directly legible.
    paint(
        &mut engine,
        RED,
        50.0,
        &[Vec2::new(-90.0, 50.0), Vec2::new(90.0, 50.0)],
    );
    let layer = engine.observe().active_layer;
    let rec = StrokeRecord {
        layer,
        tool: Tool::Brush,
        brush: dry_brush(RED, 50.0, BrushDynamics { add: 0.0, load: 1.0, deposit: 0.05, ridge: 0.0, ..Default::default() }),
        path: vec![
            InputSample::at(Vec2::new(0.0, 50.0)),
            InputSample::at(Vec2::new(0.0, -25.0)),
        ],
        seed: 0,
    };
    let (color, aux, w, bands) = engine.debug_reservoir(&rec).expect("mixer should run");
    let img = reservoir_image(&color, &aux, w, bands);
    assert_golden("reservoir_viz", &img, 6);
}

#[test]
fn golden_dry_smear_artifact() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    paint(
        &mut engine,
        RED,
        50.0,
        &[Vec2::new(-90.0, 50.0), Vec2::new(90.0, 50.0)],
    );

    let b = dry_brush(RED, 50.0, BrushDynamics { add: 0.0, load: 1.0, deposit: 0.05, ridge: 0.0, ..Default::default() });
    stroke_with(&mut engine, b, &[Vec2::new(0.0, 50.0), Vec2::new(0.0, -25.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("dry_smear_artifact", &img, 6);
}

#[test]
fn golden_dry_erase_artifact() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    paint(
        &mut engine,
        RED,
        50.0,
        &[Vec2::new(-90.0, 0.0), Vec2::new(90.0, 0.0)],
    );

    let b = dry_brush(RED, 50.0, BrushDynamics { add: 0.0, load: 1.0, deposit: 0.0, ridge: 0.0, ..Default::default() });
    stroke_with(
        &mut engine,
        b,
        &[Vec2::new(-90.0, 0.0), Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)],
    );

    let img = engine.render_to_image(PAPER);
    assert_golden("dry_erase_artifact", &img, 6);
}

#[test]
fn empty_smear_adds_no_height() {
    // Regression (DESIGN §6.1/§6.2): lift+deposit (add=0) over BARE canvas must deposit
    // nothing — there is no paint to lift, so the tool stays empty and lays no colour *and
    // no height*. The earlier bug laid the brush's height ungated by the reservoir presence,
    // raising relief out of nothing. Now everything rides the tool (zero when empty), so
    // the lifted-empty canvas must render bit-close to the untouched paper.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    let blank = engine.render_to_image(PAPER);

    let b = dry_brush(RED, 24.0, BrushDynamics { add: 0.0, load: 1.0, deposit: 1.0, ridge: 0.0, ..Default::default() });
    stroke_with(&mut engine, b, &[Vec2::new(-100.0, 0.0), Vec2::new(100.0, 0.0)]);

    let after = engine.render_to_image(PAPER);
    assert!(
        images_match(&blank, &after, 2),
        "a pure smear over empty canvas must add no paint/height"
    );
}

#[test]
fn golden_wet_blend() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // Wet-on-wet diffusion (DESIGN.md §6.2, Phase 2): a committed solid red field,
    // then a Wet blue stroke dragged across it. The wet paint bleeds and levels into
    // the wet red around it, so the blue↔red boundary softens into a purple halo —
    // where a Dry blue stroke would leave a hard edge. Verifies the region-composite
    // → ping-pong diffusion → write-back path.
    paint(
        &mut engine,
        RED,
        70.0,
        &[Vec2::new(-95.0, 0.0), Vec2::new(95.0, 0.0)],
    );

    let blue = [0.10, 0.20, 0.85, 1.0];
    let mut wet = brush(blue, 26.0);
    wet.dynamics = BrushDynamics { add: 1.0, bleed: 0.9, drag: 0.0, ..Default::default() };
    engine.process(InputCommand::SetBrush(wet));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(0.0, -80.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(0.0, 80.0)),
    });
    engine.process(InputCommand::EndStroke);

    let img = engine.render_to_image(PAPER);
    assert_golden("wet_blend", &img, 6);
}

#[test]
fn golden_wet_blend_no_add() {
    // A Wet brush with `add = 0` is a pure blender (DESIGN.md §6.2): it lays none of
    // its own paint and only works what's already on the canvas. A red and a green
    // field meet near x = 0; dragging the add-less blender down that boundary must
    // soften it into a red↔green gradient. Regression guard: the bleed's footprint
    // mask used to be the *deposited opacity* — identically zero when `add = 0` — so
    // the blender silently did nothing; the mask is the brush's footprint coverage,
    // which is deposit-independent.
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    paint(&mut engine, RED, 50.0, &[Vec2::new(-95.0, 0.0), Vec2::new(-2.0, 0.0)]);
    paint(&mut engine, GREEN, 50.0, &[Vec2::new(2.0, 0.0), Vec2::new(95.0, 0.0)]);

    let mut blender = brush(RED, 26.0); // its own colour is irrelevant: add = 0 lays nothing
    blender.dynamics = BrushDynamics { add: 0.0, bleed: 0.9, drag: 0.0, ..Default::default() };
    stroke_with(&mut engine, blender, &[Vec2::new(0.0, -80.0), Vec2::new(0.0, 80.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("wet_blend_no_add", &img, 6);
}

#[test]
fn golden_wet_drag() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // The Wet brush's *drag* axis (advection, DESIGN.md §6.2): a red field with a
    // horizontal green bar, then a Wet stroke (drag only, no bleed) dragged *vertically*
    // through it. The velocity injected along the stroke advects the wet paint downward,
    // pulling the green into a vertical streak along the stroke path — directional drag
    // that neither Dry nor pure-bleed Wet (isotropic) would produce.
    paint(
        &mut engine,
        RED,
        90.0,
        &[Vec2::new(-95.0, 0.0), Vec2::new(95.0, 0.0)],
    );
    paint(
        &mut engine,
        GREEN,
        14.0,
        &[Vec2::new(-95.0, -30.0), Vec2::new(95.0, -30.0)],
    );

    let mut wet = brush(RED, 24.0);
    wet.dynamics = BrushDynamics { add: 1.0, bleed: 0.0, drag: 0.9, ..Default::default() };
    engine.process(InputCommand::SetBrush(wet));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(0.0, -60.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(0.0, 70.0)),
    });
    engine.process(InputCommand::EndStroke);

    let img = engine.render_to_image(PAPER);
    assert_golden("wet_drag", &img, 6);
}

#[test]
fn golden_wet_flow() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // The unified Wet brush running *both* axes at once: a red field with a horizontal
    // green bar, then a Wet stroke with bleed AND drag dragged vertically through it.
    // The green is both pulled downward (drag) and softened/bled into the red (bleed) —
    // a single brush doing alla-prima diffusion and fluid raking simultaneously.
    paint(
        &mut engine,
        RED,
        90.0,
        &[Vec2::new(-95.0, 0.0), Vec2::new(95.0, 0.0)],
    );
    paint(
        &mut engine,
        GREEN,
        14.0,
        &[Vec2::new(-95.0, -30.0), Vec2::new(95.0, -30.0)],
    );

    let mut wet = brush(RED, 24.0);
    wet.dynamics = BrushDynamics { add: 1.0, bleed: 0.8, drag: 0.8, ..Default::default() };
    engine.process(InputCommand::SetBrush(wet));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(0.0, -60.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(0.0, 70.0)),
    });
    engine.process(InputCommand::EndStroke);

    let img = engine.render_to_image(PAPER);
    assert_golden("wet_flow", &img, 6);
}

#[test]
fn golden_smear_mixer_cap() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };

    // A committed green bar, then a red Mixer stroke dragged left→right across it:
    // the brush should pick up green where it crosses and carry a fading green
    // tint past the bar (DESIGN.md §6.2). A Dry red stroke would just lay flat red.
    paint(
        &mut engine,
        GREEN,
        50.0,
        &[Vec2::new(-90.0, -90.0), Vec2::new(-90.0, 90.0)],
    );

    let b = dry_brush(RED, 25.0, BrushDynamics { add: 0.1, load: 0.5, deposit: 0.5, ridge: 0.0, ..Default::default() });
    stroke_with(&mut engine, b, &[Vec2::new(-90.0, 0.0), Vec2::new(90.0, 0.0)]);

    let img = engine.render_to_image(PAPER);
    assert_golden("smear_mixer_cap", &img, 6);
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

    let mut b = dry_brush(RED, 30.0, BrushDynamics { add: 0.1, load: 0.5, deposit: 0.5, ridge: 0.0, ..Default::default() });
    b.hardness = 0.9; // near-solid, so it opaquely covers the green bar
    stroke_with(&mut engine, b, &[Vec2::new(-110.0, 0.0), Vec2::new(110.0, 0.0)]);

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
