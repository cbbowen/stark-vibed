//! Shared test harness: headless engine setup and golden-image comparison
//! (§9). Included by integration tests via `mod common;`.

#![allow(dead_code)] // not every test binary uses every helper

pub mod corpus;

use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use stark_core::colorspace::ColorSpaceId;
use stark_core::command::{GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushParams, BrushShape, Tool};
use stark_core::engine::{headless_engine, headless_engine_with};
use stark_core::geom::{Extent2, Vec2};
use stark_core::path::DEFAULT_TOLERANCE;
use stark_core::{Engine, RgbaImage};

pub const SIZE: Extent2 = Extent2 {
    width: 256,
    height: 256,
};
pub const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const BG: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};
/// A light neutral grey substrate, for color spaces that composite over a light
/// ground. Matches the engine's `DEFAULT_BACKGROUND`. Neutral on purpose: a warm
/// paper rendered so red-dominant it defeated channel-dominance checks like `is_red`
/// (tests asserting "is paint here?" were vacuously true on bare paper). Now that
/// [`engine_or_skip`] leaves the light on `Neutral` as well, nothing anywhere in the
/// pipeline pushes these toward a hue.
pub const PAPER: wgpu::Color = wgpu::Color {
    r: 0.85,
    g: 0.85,
    b: 0.85,
    a: 1.0,
};

/// Set to `1` to allow GPU tests to skip when no adapter is available. **Unset by
/// default, and deliberately so**: a skipped GPU test still reports `ok`, so a
/// machine (or a CI runner) with no adapter would take the entire golden / seam /
/// dynamics / selection suite green having rendered nothing at all. Skipping has
/// to be asked for.
const ALLOW_NO_GPU: &str = "STARK_ALLOW_NO_GPU";

/// Set to `1` to run the golden tests but not *compare* their output — see
/// [`assert_golden`]. For adapters other than the one the goldens were blessed on.
const SKIP_GOLDEN: &str = "STARK_SKIP_GOLDEN";

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// `None` if this machine has no usable adapter *and* [`ALLOW_NO_GPU`] permits the
/// skip; panics otherwise, so a missing adapter is a failure by default. The engine
/// comes back on whatever environment it booted with — the procedural `Neutral`.
fn or_skip(built: stark_core::Result<Engine>) -> Option<Engine> {
    match built {
        Ok(e) => Some(e),
        Err(e) if env_flag(ALLOW_NO_GPU) => {
            eprintln!("skipping GPU test ({ALLOW_NO_GPU}=1): {e}");
            None
        }
        Err(e) => panic!("no usable GPU adapter: {e}\nset {ALLOW_NO_GPU}=1 to skip GPU tests"),
    }
}

/// Build a headless engine, or `None` if this machine has no usable adapter and
/// skipping is permitted (see [`or_skip`]).
///
/// **On the procedural `Neutral` environment** — the reference light (§6.3), whose
/// exposure is 1.0 and whose whole purpose is to be the identity a color can be
/// judged against. The suite used to install the studio HDR here instead, which meant
/// every claim it made about a rendered byte was a claim about a *particular decoded
/// environment map*: a warm tint stood between every test and the paint it was
/// checking, near-white paper read red-dominant by ~33 levels, and a difference in the
/// pipeline arrived at the goldens multiplied by whatever the light happened to be
/// doing there. Under the reference light what a test reads back is what the pipeline
/// produced. [`engine_or_skip_studio`] keeps the image-based path covered.
pub fn engine_or_skip() -> Option<Engine> {
    or_skip(pollster::block_on(headless_engine(TARGET, SIZE)))
}

/// A headless engine with a chosen viewport, for tests whose stroke has to be far
/// wider than the default [`SIZE`] window shows.
pub fn engine_or_skip_sized(size: Extent2) -> Option<Engine> {
    or_skip(pollster::block_on(headless_engine(TARGET, size)))
}

/// A headless engine on the **blue** ground ([`BG`]) rather than the default
/// paper.
///
/// The substrate is document state (§15.5), so choosing it is a
/// logged edit like any other — which is exactly how a user would do it. Blue is
/// deliberate for these tests: they ask "is there paint here?" by channel dominance,
/// and a saturated ground answers in the *other* direction, so `is_blue` reads "bare
/// canvas" as positively as `is_red` reads "paint". Grey paper can only ever fail
/// `is_red`, which a black frame would too.
pub fn engine_or_skip_blue() -> Option<Engine> {
    engine_or_skip().map(on_blue)
}

/// [`engine_or_skip_sized`] on the blue ground.
pub fn engine_or_skip_sized_blue(size: Extent2) -> Option<Engine> {
    engine_or_skip_sized(size).map(on_blue)
}

pub fn on_blue(mut engine: Engine) -> Engine {
    engine.process(stark_core::command::DocCommand::SetBackground([
        BG.r as f32,
        BG.g as f32,
        BG.b as f32,
    ]));
    engine
}

/// A headless engine rendering to a chosen target format.
///
/// Exists because everything else here uses [`TARGET`] (`Rgba8Unorm`) while a real
/// browser surface is usually `Bgra8Unorm` — a difference no single-format test
/// can see, and one that silently swapped red and blue in exported PNGs.
pub fn engine_or_skip_in_format(format: wgpu::TextureFormat) -> Option<Engine> {
    or_skip(pollster::block_on(headless_engine(format, SIZE)))
}

/// A headless engine in a chosen color space (§6.7).
pub fn engine_or_skip_with(id: ColorSpaceId) -> Option<Engine> {
    or_skip(pollster::block_on(headless_engine_with(TARGET, SIZE, id)))
}

/// A headless engine lit by the real studio HDR: image-based lighting from an
/// actually-decoded environment map rather than the procedural one (§6.3).
///
/// The more demanding of the two paths — a decode, an irradiance convolution and a
/// prefiltered specular chain, none of which `Neutral` runs — and one switch away in
/// the Lighting panel, so it needs *some* pixel coverage. It gets exactly as much as
/// it is worth: one golden ([`golden_studio_environment`](../golden.rs)), rather than
/// standing behind the whole suite the way it used to. A test about compositing or
/// about a stroke should not also be a test of a particular sky.
pub fn engine_or_skip_studio() -> Option<Engine> {
    engine_or_skip().map(|mut engine| {
        let hdr = stark_testdata::assets::studio_hdr();
        engine.register_environment(stark_core::EnvironmentId::Ferndale, hdr);
        engine.process(ViewCommand::SetEnvironment(
            stark_core::EnvironmentId::Ferndale,
        ));
        engine
    })
}

pub fn brush(color: [f32; 4], radius: f32) -> BrushParams {
    BrushParams {
        color,
        radius,
        drain: 0.0015,
        shape: BrushShape::Round { hardness: 0.8 },
        ..BrushParams::default()
    }
}

/// Paint and commit a stroke through `points` with an explicit brush.
pub fn stroke_with(engine: &mut Engine, b: BrushParams, points: &[Vec2]) {
    engine.process(ViewCommand::SetBrush(b));
    let mut it = points.iter();
    let first = *it.next().expect("at least one point");
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: InputSample::at(first),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    for &p in it {
        engine.process(GestureCommand::To {
            sample: InputSample::at(p),
        });
    }
    engine.process(GestureCommand::End);
}

/// Paint and commit a stroke through the given canvas points with `color`.
pub fn paint(engine: &mut Engine, color: [f32; 4], radius: f32, points: &[Vec2]) {
    stroke_with(engine, brush(color, radius), points);
}

/// The centre pixel — where the suites' standard stroke crosses.
pub fn center(img: &RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2)
}

/// Whether a pixel reads as (red) paint rather than bare canvas: the red channel
/// clear of both others by a margin no substrate or light in the suite produces
/// (see [`PAPER`] and the `Neutral` reference light in [`engine_or_skip`]).
///
/// One copy with one margin, shared by every suite that asks "is there paint
/// here?" — so a render change that shifts channel separation moves every test the
/// same distance instead of being patched file by file.
pub fn red_dominant(c: [u8; 4]) -> bool {
    c[0] as i32 > c[1] as i32 + 30 && c[0] as i32 > c[2] as i32 + 30
}

/// Fraction of pixels whose maximum per-channel difference exceeds `tol`.
pub fn diff_fraction(a: &RgbaImage, b: &RgbaImage) -> (f64, u8) {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "image size mismatch"
    );
    let mut bad = 0u64;
    let mut worst = 0u8;
    for (pa, pb) in a
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.pixels.as_chunks::<4>().0)
    {
        let d = pa
            .iter()
            .zip(pb)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        worst = worst.max(d);
        if d > 0 {
            bad += 1;
        }
    }
    let total = (a.width * a.height) as f64;
    (bad as f64 / total, worst)
}

/// Fraction of pixels whose maximum per-channel difference *exceeds* `tol`. Unlike the
/// per-pixel worst, this distinguishes a contiguous seam (a band of many significantly-
/// different pixels) from a handful of isolated precision specks.
pub fn frac_exceeding(a: &RgbaImage, b: &RgbaImage, tol: u8) -> f64 {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "image size mismatch"
    );
    let mut bad = 0u64;
    for (pa, pb) in a
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.pixels.as_chunks::<4>().0)
    {
        let d = pa
            .iter()
            .zip(pb)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        if d > tol as u32 {
            bad += 1;
        }
    }
    bad as f64 / (a.width * a.height) as f64
}

/// True if `a` and `b` match within `tol` per channel for every pixel.
pub fn images_match(a: &RgbaImage, b: &RgbaImage, tol: u8) -> bool {
    let (_, worst) = diff_fraction(a, b);
    worst <= tol
}

/// Compare `img` against the committed golden `tests/golden/<name>.png`.
///
/// If the golden file is absent it is created and the check passes — so
/// **deleting a golden re-blesses it** on the next run. Goldens are
/// GPU-dependent; cross-adapter runs may need re-blessing (§9).
///
/// Which is why [`SKIP_GOLDEN`] exists: a committed golden can only match the one
/// adapter it was blessed on, so a CI runner on a different adapter would fail on
/// pixels rather than on behaviour. With it set the stroke is still *rendered* —
/// so shader compilation, validation errors and panics are still caught — only
/// the pixel comparison is dropped.
pub fn assert_golden(name: &str, img: &RgbaImage, tol: u8) {
    if env_flag(SKIP_GOLDEN) {
        eprintln!("golden {name}: rendered, comparison skipped ({SKIP_GOLDEN}=1)");
        return;
    }
    let dir = golden_dir();
    fs::create_dir_all(&dir).expect("create golden dir");
    let path = dir.join(format!("{name}.png"));

    if !path.exists() {
        write_png(&path, img);
        eprintln!("blessed new golden: {}", path.display());
        return;
    }

    let golden = read_png(&path);
    assert_eq!(
        (img.width, img.height),
        (golden.width, golden.height),
        "golden {name}: size mismatch"
    );

    let mut bad = 0u64;
    for (a, b) in img
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(golden.pixels.as_chunks::<4>().0)
    {
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        if d as u8 > tol {
            bad += 1;
        }
    }
    let total = (img.width * img.height) as u64;
    let frac = bad as f64 / total as f64;
    if frac > 0.01 {
        let actual = path.with_extension("actual.png");
        write_png(&actual, img);
        panic!(
            "golden {name} mismatch: {bad}/{total} px exceed tol {tol} ({:.2}%); wrote {}",
            frac * 100.0,
            actual.display()
        );
    }
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

pub fn write_png(path: &Path, img: &RgbaImage) {
    let file = fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(BufWriter::new(file), img.width, img.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(&img.pixels).expect("png data");
}

fn read_png(path: &Path) -> RgbaImage {
    let decoder = png::Decoder::new(BufReader::new(fs::File::open(path).expect("open png")));
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("buffer size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(
        info.color_type,
        png::ColorType::Rgba,
        "golden must be RGBA8"
    );
    buf.truncate(info.buffer_size());
    RgbaImage::new(info.width, info.height, buf)
}
