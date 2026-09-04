//! Wide-gamut paint (§6.5): a color outside the sRGB cube survives the pipeline onto
//! a surface that can show it, keeps its chroma through a filter, and is held to the
//! cube on the way into a file.

mod common;

use common::*;
use stark_engine::command::{DocCommand, ViewCommand};
use stark_engine::{
    Background, Engine, ExportScale, MediaParams, Offscreen, Output, Rendered, Transfer,
};
use stark_model::Srgb;
use stark_model::color::{linear_p3_to_linear_srgb, linear_to_srgb, srgb_to_linear};
use stark_model::document::{BrushEffect, BrushParams, BrushShape, ColorAdjust, Filter};
use stark_model::geom::Vec2;

const F16: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Display P3's green, as the extended sRGB the document carries: a negative red
/// channel, which is the whole of what makes it wide.
fn p3_green() -> [f32; 3] {
    linear_p3_to_linear_srgb([0.0, 1.0, 0.0]).map(linear_to_srgb)
}

/// The reference light over an opaque patch of [`p3_green`] on black: nothing in the
/// tail but the paint's own color.
fn painted(format: wgpu::TextureFormat) -> Option<Engine> {
    let mut engine = engine_or_skip_in_format(format)?;
    engine.process(ViewCommand::SetMediaParams(MediaParams {
        height_strength: 0.0,
        specular: 0.0,
        substrate_strength: 0.0,
        dither: false,
    }));
    engine.process(DocCommand::SetSubstrateColor(Srgb::new([0.0, 0.0, 0.0])));
    let brush = BrushParams {
        effect: BrushEffect::painted(p3_green()),
        size: 48.0,
        shape: BrushShape::Round { hardness: 0.95 },
        drain: 0.0,
        ..BrushParams::default()
    };
    for _ in 0..2 {
        stroke_with(
            &mut engine,
            brush,
            &[Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
        );
    }
    Some(engine)
}

/// The centre texel of a half-float render, in the engine's transfer.
fn centre(engine: &mut Engine) -> [f32; 3] {
    let px = engine.render_to_floats();
    let i = ((SIZE.height / 2) * SIZE.width + SIZE.width / 2) as usize * 4;
    [px[i], px[i + 1], px[i + 2]]
}

/// The paint reaches a linear surface as the color it is: the red channel below
/// zero, which the sRGB cube cannot say.
#[test]
fn wide_paint_reaches_a_linear_surface() {
    let Some(mut hdr) = painted(F16) else {
        return;
    };
    hdr.process(ViewCommand::SetOutput(Output::new(Transfer::Linear, 1.0)));
    let c = centre(&mut hdr);
    let want = linear_p3_to_linear_srgb([0.0, 1.0, 0.0]);
    assert!(
        c[0] < -0.15 && c[1] > 0.9,
        "P3 green rendered {c:?} on a linear surface; its linear sRGB is {want:?}"
    );
}

/// A P3 surface is handed the same color in *its* coordinates — the primary itself,
/// inside `[0, 1]`, where an sRGB surface needs a channel below zero to say it.
///
/// Measured in **linear** P3: the off-primaries carry the achromatic sheen the
/// reference light adds to any paint (`tests/reference.rs`'s ~1.6% dielectric floor),
/// and the sRGB curve is steep enough near zero to make that 2% of light read as 16%
/// of an encoded channel. The claim is about the color, so it is asked of the light.
#[test]
fn wide_paint_reaches_a_p3_surface_as_its_primary() {
    let Some(mut hdr) = painted(F16) else {
        return;
    };
    hdr.process(ViewCommand::SetOutput(Output::new(
        Transfer::ExtendedDisplayP3,
        1.0,
    )));
    let enc = centre(&mut hdr);
    assert!(
        enc.iter().all(|c| (-1e-3..=1.0 + 1e-3).contains(c)),
        "P3 green rendered {enc:?}, which a Display P3 surface cannot hold"
    );
    let lin = enc.map(srgb_to_linear);
    assert!(
        lin[1] > 0.9 && lin[0] < 0.04 && lin[2] < 0.04,
        "P3 green is {lin:?} in linear Display P3, and should be near (0, 1, 0)"
    );
}

/// A filter over the paint keeps its chroma: the floors on the way through are in
/// light, not per sRGB channel (§6.5).
#[test]
fn a_filter_keeps_the_chroma() {
    let Some(mut hdr) = painted(F16) else {
        return;
    };
    hdr.process(ViewCommand::SetOutput(Output::new(Transfer::Linear, 1.0)));
    let before = centre(&mut hdr);
    hdr.process(DocCommand::AddFilter {
        carrier: None,
        above: None,
        filter: Filter::Color(ColorAdjust::NEUTRAL),
    });
    let after = centre(&mut hdr);
    assert!(
        after[0] < -0.15,
        "a neutral filter clipped P3 green from {before:?} to {after:?}"
    );
}

/// An export is sRGB (§15.6): the channel the cube cannot hold is clipped at zero,
/// and the rest of the color comes through.
#[test]
fn an_export_holds_the_paint_to_the_cube() {
    let Some(mut hdr) = painted(F16) else {
        return;
    };
    hdr.process(ViewCommand::SetOutput(Output::new(Transfer::Linear, 1.0)));
    let img = pollster::block_on(
        hdr.export(
            &mut Offscreen::default(),
            None,
            ExportScale::Factor(1.0),
            Background::Substrate,
            Rendered::Committed,
        )
        .expect("export"),
    )
    .expect("the readback completes");
    let c = center(&img);
    assert!(
        c[0] == 0 && c[1] >= 250,
        "P3 green exported as {c:?}; an sRGB file clips the red channel at zero"
    );
}
