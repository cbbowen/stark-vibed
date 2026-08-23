//! What a zoomed-out view shows (§6.4).
//!
//! **The claim.** Minifying the canvas is *averaging* it, not sampling one texel in
//! every `1/z²` the output pixel covers. Sampling is what one bilinear tap per
//! fragment gives — into a tile texture with no mip chain, a canvas substrate read at
//! LOD 0, impasto normals taken as a finite difference across screen pixels — and a
//! thumbnail built that way is a sparse, phase-dependent sample of the painting, which
//! crawls under a pan and buzzes on every edge. The navigator's miniature (a whole
//! piece in ~250 px) is the worst case.
//!
//! **How it is answered.** Everything from pass A to the guides runs at `ss` samples
//! per axis and a box filter resolves it (`resolve.wesl`), so a 1:4 render is a 1:1
//! render boxed down by four. That is exactly what this asserts: at `ss × zoom = 1`
//! the supersampled render *is* the full-size one, so the GPU's minified image has to
//! agree with the full-size image downsampled on the CPU — and must not agree with the
//! one texel in sixteen a single tap would pick.
//!
//! Nothing here can be answered by prefiltering the tiles alone, which is the reason
//! the fix is where it is: the relief shading is a nonlinear function of the height
//! field, so the right pixel is the average of the shading and not the shading of the
//! average.

mod common;

use common::*;
use stark_engine::command::DocCommand;
use stark_engine::{Background, ExportScale, Offscreen, Rendered, RgbaImage};
use stark_model::Srgb;
use stark_model::document::{BrushParams, BrushShape, MattePaint, MatteRegion, Place};
use stark_model::geom::Vec2;

/// The frame the exports are taken against: a rect whose size divides by [`SS`], so
/// the two renders are the same picture at two sizes rather than two croppings.
const HALF: f32 = 96.0;
/// The minification under test. 1:4 is where `supersample` saturates, so this also
/// pins the cap's behaviour.
const SS: u32 = 4;

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// `img` box-averaged by `n` in light — the CPU statement of what `resolve.wesl`
/// does, written out rather than shared so the test does not agree with the shader
/// by construction.
fn box_down(img: &RgbaImage, n: u32) -> RgbaImage {
    let (w, h) = (img.width / n, img.height / n);
    let at = |x: u32, y: u32| {
        let i = ((y * img.width + x) * 4) as usize;
        let p = &img.pixels[i..i + 4];
        std::array::from_fn::<f32, 4, _>(|c| p[c] as f32 / 255.0)
    };
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (mut rgb, mut a) = ([0.0f32; 3], 0.0f32);
            for j in 0..n {
                for i in 0..n {
                    let t = at(x * n + i, y * n + j);
                    for c in 0..3 {
                        rgb[c] += srgb_to_linear(t[c]) * t[3];
                    }
                    a += t[3];
                }
            }
            for c in rgb {
                pixels.push((linear_to_srgb(c / a.max(1e-5)) * 255.0).round() as u8);
            }
            pixels.push((a / (n * n) as f32 * 255.0).round() as u8);
        }
    }
    RgbaImage {
        width: w,
        height: h,
        pixels,
    }
}

/// `img` decimated: the top-left texel of each `n × n` block, which is what a single
/// tap per output pixel amounts to. The control — see [`minified_agrees_with_the_box`].
fn decimate(img: &RgbaImage, n: u32) -> RgbaImage {
    let (w, h) = (img.width / n, img.height / n);
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let i = (((y * n) * img.width + x * n) * 4) as usize;
            pixels.extend_from_slice(&img.pixels[i..i + 4]);
        }
    }
    RgbaImage {
        width: w,
        height: h,
        pixels,
    }
}

/// A hard, thin tip: the content whose edges a single tap per pixel cannot represent.
fn fine(color: [f32; 4]) -> BrushParams {
    BrushParams {
        size: 2.0,
        shape: BrushShape::Round { hardness: 1.0 },
        ..brush(color, 2.0)
    }
}

/// Paint a fan of thin strokes at angles no pixel grid is kind to, and frame it.
fn hatched(engine: &mut stark_engine::Engine) {
    // Spaced by a period that is not a multiple of the 4-px output block, so which
    // stroke a decimating sample lands on walks across the picture — the phase
    // dependence that reads as banding.
    let start = -HALF + 5.0;
    // Indexed rather than accumulated — the same fan, minus the question of whether
    // the last stroke falls inside the frame by arithmetic or by luck.
    let count = ((HALF - start) / 9.0).ceil() as u32;
    for i in 0..count {
        let x = start + i as f32 * 9.0;
        stroke_with(
            engine,
            fine([0.85, 0.10, 0.10, 1.0]),
            &[Vec2::new(x, -HALF), Vec2::new(x + 34.0, HALF)],
        );
    }
    engine.process(DocCommand::AddMatte {
        carrier: None,
        at: Place::Top,
        region: MatteRegion::OutsideRect {
            min: Vec2::new(-HALF, -HALF),
            max: Vec2::new(HALF, HALF),
        },
        paint: MattePaint::Solid(Srgb::new([0.0, 0.0, 0.0])),
    });
}

fn shot(engine: &mut stark_engine::Engine, scale: f32) -> RgbaImage {
    let frame = engine.observe().layers.last().expect("matte").id;
    pollster::block_on(
        engine
            .export(
                &mut Offscreen::default(),
                Some(frame),
                ExportScale::Factor(scale),
                Background::Substrate,
                Rendered::Committed,
            )
            .expect("export"),
    )
    .expect("the readback completes")
}

/// A 1:4 render is the 1:1 render averaged down — and is *not* one texel in sixteen
/// of it.
///
/// The two halves are one assertion. Agreeing with the box says the resolve is doing
/// the filtering; disagreeing with the decimation says there was something to filter,
/// so the first half cannot pass by the content being smooth enough that every
/// downsample looks alike.
#[test]
fn minified_agrees_with_the_box() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    hatched(&mut engine);

    let full = shot(&mut engine, 1.0);
    let small = shot(&mut engine, 1.0 / SS as f32);
    assert_eq!(
        (small.width, small.height),
        (full.width / SS, full.height / SS),
        "the two exports are not the same picture at two sizes"
    );

    // Both sides quantize to 8 bits at the same point — the supersampled target is
    // the target format — so the only slack is the rounding of the two averages.
    let (frac, worst) = diff_fraction(&small, &box_down(&full, SS));
    assert!(
        worst <= 4 && frac_exceeding(&small, &box_down(&full, SS), 2) < 0.02,
        "a 1:{SS} render is not the box-average of the 1:1 one \
         (worst {worst} levels, {:.1}% of pixels differ at all) — the presentation \
         resolve is not filtering what it minifies",
        frac * 100.0
    );

    let (_, worst_point) = diff_fraction(&small, &decimate(&full, SS));
    assert!(
        worst_point > 40,
        "the hatching is not high-frequency enough for this test to mean anything: \
         decimating the 1:1 render differs from the 1:{SS} one by only {worst_point} \
         levels"
    );
}

/// Magnification pays nothing, which is what keeps every golden — all of them
/// blessed at 1:1 — valid, and painting at 100% exactly as expensive as it was.
///
/// Stated against the *pixels* rather than against `supersample`'s return value: what
/// matters is that a 1:1 render is untouched by any of this, and a unit test of the
/// policy function could not say that.
#[test]
fn a_view_that_is_not_minifying_is_bit_identical() {
    let Some(mut engine) = engine_or_skip() else {
        return;
    };
    hatched(&mut engine);

    let (frac, worst) = diff_fraction(&shot(&mut engine, 1.0), &shot(&mut engine, 1.0));
    assert_eq!((frac, worst), (0.0, 0), "a 1:1 render is not reproducible");
    // 2× is the other side of the threshold: still no resolve, so the same render
    // twice is still exact.
    let (frac, worst) = diff_fraction(&shot(&mut engine, 2.0), &shot(&mut engine, 2.0));
    assert_eq!((frac, worst), (0.0, 0), "a 2:1 render is not reproducible");
}
