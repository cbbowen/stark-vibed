//! Step-3 golden-image tests (DESIGN.md §9). Render known scripts and compare
//! against committed reference PNGs. Regenerate by deleting the existing golden image.

mod common;

use common::*;
use stark_core::colorspace::ColorSpaceId;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushDynamics, BrushShape, DryParams, Tool, WetParams};
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
    brush.dynamics = BrushDynamics::Dry(DryParams { add: 0.1, lift: 0.5, deposit: 0.5, ridge: 0.0 });
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

    let mut knife = brush(RED, 22.0);
    // Explicit (not `::default()`) so this "clean scrape" golden is stable regardless
    // of the default knife's feel: hard bite, no film, no carry, no ridge.
    knife.dynamics = BrushDynamics::Dry(DryParams {
        add: 0.0,
        lift: 0.8,
        deposit: 0.0, // lift onto the tool, lay nothing back → a clean scrape/erase
        ridge: 0.0,
    });
    engine.process(InputCommand::SetBrush(knife));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(0.0, -80.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(0.0, 80.0)),
    });
    engine.process(InputCommand::EndStroke);

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

    let mut knife = brush(RED, 26.0);
    knife.tooth = 1.0; // full tooth: the gate bites only the weave's peaks
    // Explicit params so the tooth-reveal golden is stable against the default knife.
    knife.dynamics = BrushDynamics::Dry(DryParams {
        add: 0.0,
        lift: 0.8,
        deposit: 0.0, // lift onto the tool, lay nothing back → a clean scrape/erase
        ridge: 0.0,
    });
    engine.process(InputCommand::SetBrush(knife));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(0.0, -80.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(0.0, 80.0)),
    });
    engine.process(InputCommand::EndStroke);

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

    let mut knife = brush(RED, 40.0);
    knife.dynamics = BrushDynamics::Dry(DryParams {
        add: 0.0,
        lift: 0.9,     // lift green onto the tool (lightening the source)
        deposit: 0.12, // slowly re-lay it onto the bare runway, dragging a long fading tail
        ridge: 0.0,
    });
    engine.process(InputCommand::SetBrush(knife));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-90.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(90.0, 0.0)),
    });
    engine.process(InputCommand::EndStroke);

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

    let mut knife = brush(RED, 30.0);
    knife.dynamics = BrushDynamics::Dry(DryParams {
        add: 0.0,
        lift: 0.15, // gentle lift: keep the colour ~uniform so the lip reads as relief
        deposit: 0.0,
        ridge: 1.0,
    });
    engine.process(InputCommand::SetBrush(knife));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(0.0, -80.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(0.0, 80.0)),
    });
    engine.process(InputCommand::EndStroke);

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

    let mut brush = brush(RED, 18.0);
    brush.dynamics = BrushDynamics::Dry(DryParams { add: 0.4, lift: 0.7, deposit: 0.4, ridge: 0.0 });
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
    assert_golden("smudge_paint", &img, 6);
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

    let mut brush = brush(RED, 24.0);
    brush.dynamics = BrushDynamics::Dry(DryParams { add: 0.0, lift: 1.0, deposit: 1.0, ridge: 0.0 });
    engine.process(InputCommand::SetBrush(brush));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-100.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(100.0, 0.0)),
    });
    engine.process(InputCommand::EndStroke);

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
    wet.dynamics = BrushDynamics::Wet(WetParams { bleed: 0.9, drag: 0.0 });
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
    wet.dynamics = BrushDynamics::Wet(WetParams { bleed: 0.0, drag: 0.9 });
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
    wet.dynamics = BrushDynamics::Wet(WetParams { bleed: 0.8, drag: 0.8 });
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

    let mut brush = brush(RED, 25.0);
    brush.dynamics = BrushDynamics::Dry(DryParams { add: 0.1, lift: 0.5, deposit: 0.5, ridge: 0.0 });
    engine.process(InputCommand::SetBrush(brush));
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(Vec2::new(-90.0, 0.0)),
    });
    engine.process(InputCommand::StrokeTo {
        sample: InputSample::at(Vec2::new(90.0, 0.0)),
    });
    engine.process(InputCommand::EndStroke);

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

    let mut brush = brush(RED, 30.0);
    brush.hardness = 0.9; // near-solid, so it opaquely covers the green bar
    brush.dynamics = BrushDynamics::Dry(DryParams { add: 0.1, lift: 0.5, deposit: 0.5, ridge: 0.0 });
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
