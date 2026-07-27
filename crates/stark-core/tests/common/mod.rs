//! Shared test harness: headless engine setup and golden-image comparison
//! (DESIGN.md §9). Included by integration tests via `mod common;`.

#![allow(dead_code)] // not every test binary uses every helper

use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use stark_core::colorspace::ColorSpaceId;
use stark_core::command::{GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushParams, Tool};
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
/// A neutral near-white paper substrate, for color spaces that composite over a light
/// ground. Matches the app's default (`stark-ui` `render::BG`). Neutral on purpose: the
/// studio HDR lights the scene warm, and a warm paper on top of that rendered so
/// red-dominant it defeated channel-dominance checks like `is_red` (tests asserting "is
/// paint here?" were vacuously true on bare paper).
pub const PAPER: wgpu::Color = wgpu::Color {
    r: 0.97,
    g: 0.97,
    b: 0.97,
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
/// skip; panics otherwise, so a missing adapter is a failure by default.
fn or_skip(built: stark_core::Result<Engine>) -> Option<Engine> {
    match built {
        Ok(e) => Some(with_studio_env(e)),
        Err(e) if env_flag(ALLOW_NO_GPU) => {
            eprintln!("skipping GPU test ({ALLOW_NO_GPU}=1): {e}");
            None
        }
        Err(e) => panic!("no usable GPU adapter: {e}\nset {ALLOW_NO_GPU}=1 to skip GPU tests"),
    }
}

/// Build a headless engine, or `None` if this machine has no usable adapter and
/// skipping is permitted (see [`or_skip`]).
pub fn engine_or_skip() -> Option<Engine> {
    or_skip(pollster::block_on(headless_engine(TARGET, SIZE)))
}

/// A headless engine with a chosen viewport, for tests whose stroke has to be far
/// wider than the default [`SIZE`] window shows.
pub fn engine_or_skip_sized(size: Extent2) -> Option<Engine> {
    or_skip(pollster::block_on(headless_engine(TARGET, size)))
}

/// A headless engine in a chosen color space (DESIGN.md §6.7).
pub fn engine_or_skip_with(id: ColorSpaceId) -> Option<Engine> {
    or_skip(pollster::block_on(headless_engine_with(TARGET, SIZE, id)))
}

/// Light the test engine with the real studio HDR (the frontend's default), so
/// goldens exercise the same image-based lighting the app uses rather than the
/// procedural fallback (DESIGN.md §6.3).
fn with_studio_env(mut engine: Engine) -> Engine {
    let hdr = stark_testdata::assets::studio_hdr();
    engine.register_environment(stark_core::EnvironmentId::Ferndale, hdr);
    engine.process(ViewCommand::SetEnvironment(
        stark_core::EnvironmentId::Ferndale,
    ));
    engine
}

pub fn brush(color: [f32; 4], radius: f32) -> BrushParams {
    BrushParams {
        color,
        radius,
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
/// GPU-dependent; cross-adapter runs may need re-blessing (DESIGN.md §9).
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

fn write_png(path: &Path, img: &RgbaImage) {
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
