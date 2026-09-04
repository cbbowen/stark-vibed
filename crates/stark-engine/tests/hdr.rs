//! The HDR output (§6.5): what a half-float surface is handed, held against the
//! SDR picture. Each test is a thing the display setting must *not* be able to
//! change — the SDR preview, everything under white, the paint, and the export.

mod common;

use common::*;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_engine::{
    Background, Engine, EnvironmentId, ExportScale, MediaParams, Offscreen, Output, Rendered,
    Transfer,
};
use stark_model::Srgb;
use stark_model::color::{linear_to_srgb, srgb_to_linear};
use stark_model::document::{BrushEffect, BrushParams, BrushShape};
use stark_model::geom::Vec2;

const F16: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Near-mirror gloss over steep relief, undithered: the most a scene can gain from
/// headroom, with nothing stochastic to compare through.
const GLOSSY: MediaParams = MediaParams {
    height_strength: 0.6,
    specular: 1.0,
    substrate_strength: 0.0,
    dither: false,
};

/// The same scene on any engine: the studio light over a thick opaque stroke, so the
/// lamps have something to glint off (§6.3).
fn scene(engine: &mut Engine) {
    engine
        .register_environment(
            EnvironmentId::Ferndale,
            stark_testdata::assets::studio_hdr(),
        )
        .expect("the bundled studio HDR decodes");
    engine.process(ViewCommand::SetEnvironment(EnvironmentId::Ferndale));
    engine.process(ViewCommand::SetMediaParams(GLOSSY));
    let brush = BrushParams {
        effect: BrushEffect::painted([0.85, 0.2, 0.15]),
        size: 40.0,
        shape: BrushShape::Round { hardness: 0.9 },
        drain: 0.0,
        ..BrushParams::default()
    };
    for _ in 0..2 {
        stroke_with(
            engine,
            brush,
            &[Vec2::new(-90.0, -30.0), Vec2::new(90.0, 30.0)],
        );
    }
}

/// An 8-bit engine and a half-float one, both showing [`scene`].
fn pair() -> Option<(Engine, Engine)> {
    let mut sdr = engine_or_skip_in_format(TARGET)?;
    let mut hdr = engine_or_skip_in_format(F16)?;
    scene(&mut sdr);
    scene(&mut hdr);
    Some((sdr, hdr))
}

/// Zoom the view about the centre by `factor`, so a comparison can run through the
/// presentation resolve (§6.4) as well as at 1:1.
fn zoom(engine: &mut Engine, factor: f32) {
    let anchor = Vec2::new(SIZE.width as f32 / 2.0, SIZE.height as f32 / 2.0);
    engine.process(ViewCommand::Zoom { anchor, factor });
}

/// A half-float texel back to the 8-bit code the SDR path would have stored — the
/// decode is the transfer's own, the encode is sRGB's, and the clamp is the 8-bit
/// store's. Per channel, so the P3 transfers are out of its reach: `wide.rs` is where
/// a change of primaries is measured.
fn code(transfer: Transfer, v: f32) -> i32 {
    let lin = match transfer {
        Transfer::Linear => v,
        Transfer::ExtendedSrgb => srgb_to_linear(v),
        Transfer::Srgb => srgb_to_linear(v.clamp(0.0, 1.0)),
        p3 => unreachable!("{p3:?} changes primaries, which one channel cannot undo"),
    };
    (linear_to_srgb(lin.clamp(0.0, 1.0)) * 255.0).round() as i32
}

/// At headroom 1, whatever transfer the surface speaks, the half-float picture is
/// the 8-bit one to within its own rounding — at 1:1 and through the resolve.
#[test]
fn an_sdr_preview_on_an_hdr_surface_is_the_sdr_picture() {
    let Some((mut sdr, mut hdr)) = pair() else {
        return;
    };
    for factor in [1.0, 0.25] {
        zoom(&mut sdr, factor);
        zoom(&mut hdr, factor);
        let want = sdr.render_to_image();
        for transfer in [Transfer::Linear, Transfer::ExtendedSrgb] {
            hdr.process(ViewCommand::SetOutput(Output::new(transfer, 1.0)));
            let got = hdr.render_to_floats();
            assert_eq!(got.len(), want.pixels.len(), "one texel per texel");
            let mut worst = 0;
            let mut off_by_two = 0usize;
            for (g, w) in got
                .as_chunks::<4>()
                .0
                .iter()
                .zip(want.pixels.as_chunks::<4>().0)
            {
                for c in 0..3 {
                    let d = (code(transfer, g[c]) - i32::from(w[c])).abs();
                    worst = worst.max(d);
                    if d > 1 {
                        off_by_two += 1;
                    }
                }
            }
            // One code is the 8-bit path's own rounding; the resolve averages
            // quantized texels on one side and half floats on the other, which
            // can land a second code apart at a few edges and nowhere else.
            let texels = got.len() / 4;
            assert!(
                worst <= 2 && off_by_two < texels / 200,
                "{transfer:?} at zoom {factor}: worst {worst} codes, {off_by_two} channels off \
                 by two — an SDR preview on an HDR surface is not the SDR picture"
            );
        }
        // Back to 1:1 for the next round.
        zoom(&mut sdr, 1.0 / factor);
        zoom(&mut hdr, 1.0 / factor);
    }
}

/// Raising the headroom lets the glints through and moves nothing under white.
#[test]
fn headroom_lets_the_glints_through_and_nothing_else() {
    let Some((_, mut hdr)) = pair() else {
        return;
    };
    hdr.process(ViewCommand::SetOutput(Output::new(Transfer::Linear, 1.0)));
    let sdr = hdr.render_to_floats();
    hdr.process(ViewCommand::SetOutput(Output::new(Transfer::Linear, 4.0)));
    let wide = hdr.render_to_floats();

    let peak = wide
        .as_chunks::<4>()
        .0
        .iter()
        .map(|t| t[0].max(t[1]).max(t[2]))
        .fold(0.0f32, f32::max);
    assert!(
        peak > 1.05,
        "nothing glinted above white ({peak}) — the scene has no highlight for the headroom \
         to let through, so this test measures nothing"
    );
    assert!(
        peak <= 4.0 + 1e-3,
        "the tonemap let {peak} past a headroom of 4"
    );

    let mut moved_under_white = 0usize;
    let mut lowered = 0usize;
    for (a, b) in sdr.as_chunks::<4>().0.iter().zip(wide.as_chunks::<4>().0) {
        let peak = |t: &[f32; 4]| t[0].max(t[1]).max(t[2]);
        // The *peak* channel, not each channel: the SDR curve desaturates what it
        // compresses, lifting a glint's dark channels, and the wide render leaves
        // them where the light put them.
        if peak(b) < peak(a) - 2e-3 {
            lowered += 1;
        }
        // Well under the knee, the two curves are the same function — to half-float
        // precision, which is what the slack is.
        if peak(a) < 0.9 && (0..3).any(|c| (a[c] - b[c]).abs() > 2e-3) {
            moved_under_white += 1;
        }
    }
    assert_eq!(lowered, 0, "headroom darkened the peak of {lowered} texels");
    assert_eq!(
        moved_under_white, 0,
        "headroom moved {moved_under_white} texels that were already well under white"
    );
}

/// Paint is reflectance: an opaque white patch stays at white however much headroom
/// the surface has (§6.5).
#[test]
fn paint_never_leaves_white_behind_it() {
    let Some(mut hdr) = engine_or_skip_in_format(F16) else {
        return;
    };
    hdr.process(ViewCommand::SetMediaParams(MediaParams {
        height_strength: 0.0,
        specular: 0.0,
        substrate_strength: 0.0,
        dither: false,
    }));
    hdr.process(ViewCommand::SetOutput(Output::new(Transfer::Linear, 4.0)));
    hdr.process(DocCommand::SetSubstrateColor(Srgb::new([0.0, 0.0, 0.0])));
    let brush = BrushParams {
        effect: BrushEffect::painted([1.0, 1.0, 1.0]),
        size: 48.0,
        shape: BrushShape::Round { hardness: 0.95 },
        drain: 0.0,
        ..BrushParams::default()
    };
    for _ in 0..2 {
        stroke_with(
            &mut hdr,
            brush,
            &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
        );
    }
    let px = hdr.render_to_floats();
    let i = ((SIZE.height / 2) * SIZE.width + SIZE.width / 2) as usize * 4;
    let centre = [px[i], px[i + 1], px[i + 2]];
    assert!(
        centre.iter().all(|&c| (0.97..=1.01).contains(&c)),
        "white paint rendered {centre:?} on a surface with headroom 4 — headroom must not \
         lift the paint"
    );
}

/// An export from an HDR session is the SDR export, byte for byte (§15.6).
#[test]
fn an_export_from_an_hdr_session_is_the_sdr_export() {
    let Some((mut sdr, mut hdr)) = pair() else {
        return;
    };
    hdr.process(ViewCommand::SetOutput(Output::new(Transfer::Linear, 4.0)));
    let shot = |engine: &mut Engine| {
        pollster::block_on(
            engine
                .export(
                    &mut Offscreen::default(),
                    None,
                    ExportScale::Factor(1.0),
                    Background::Substrate,
                    Rendered::Committed,
                )
                .expect("export"),
        )
        .expect("the readback completes")
    };
    let a = shot(&mut sdr);
    let b = shot(&mut hdr);
    assert!(
        images_match(&a, &b, 0),
        "an export from the HDR engine differs from the SDR engine's"
    );
}
