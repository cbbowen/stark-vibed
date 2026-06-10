//! Shared test harness: headless engine setup and golden-image comparison
//! (DESIGN.md §9). Included by integration tests via `mod common;`.

#![allow(dead_code)] // not every test binary uses every helper

use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use stark_core::colorspace::ColorSpaceId;
use stark_core::command::{InputCommand, InputSample};
use stark_core::document::{BrushDynamics, BrushParams, DryParams, Tool};
use stark_core::engine::{headless_engine, headless_engine_with};
use stark_core::geom::{Extent2, Vec2};
use stark_core::{Engine, RgbaImage};

pub const SIZE: Extent2 = Extent2 { width: 256, height: 256 };
pub const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const BG: wgpu::Color = wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
/// A neutral near-white paper substrate, for color spaces that composite over a light
/// ground. Matches the app's default (`stark-ui` `render::BG`). Neutral on purpose: the
/// studio HDR lights the scene warm, and a warm paper on top of that rendered so
/// red-dominant it defeated channel-dominance checks like `is_red` (tests asserting "is
/// paint here?" were vacuously true on bare paper).
pub const PAPER: wgpu::Color = wgpu::Color { r: 0.97, g: 0.97, b: 0.97, a: 1.0 };

/// Build a headless engine, or `None` if this machine has no usable adapter
/// (the test should then skip rather than fail).
pub fn engine_or_skip() -> Option<Engine> {
    match pollster::block_on(headless_engine(TARGET, SIZE)) {
        Ok(e) => Some(with_studio_env(e)),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

/// A headless engine in a chosen color space (DESIGN.md §6.7).
pub fn engine_or_skip_with(id: ColorSpaceId) -> Option<Engine> {
    match pollster::block_on(headless_engine_with(TARGET, SIZE, id)) {
        Ok(e) => Some(with_studio_env(e)),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

/// Light the test engine with the real studio HDR (the frontend's default), so
/// goldens exercise the same image-based lighting the app uses rather than the
/// procedural fallback (DESIGN.md §6.3).
fn with_studio_env(mut engine: Engine) -> Engine {
    let hdr = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../stark-ui/assets/environment/ferndale_studio_11_1k.hdr"
    ))
    .expect("read studio HDR");
    engine.register_environment(stark_core::EnvironmentId::Ferndale, hdr);
    engine.set_environment(stark_core::EnvironmentId::Ferndale);
    engine
}

pub fn brush(color: [f32; 4], radius: f32) -> BrushParams {
    BrushParams {
        color,
        radius,
        ..BrushParams::default()
    }
}

/// A brush with the given [`DryParams`] dynamics (the add/lift/deposit brush, §6.2).
pub fn dry_brush(color: [f32; 4], radius: f32, p: DryParams) -> BrushParams {
    BrushParams {
        dynamics: BrushDynamics::Dry(p),
        ..brush(color, radius)
    }
}

/// Paint and commit a stroke through `points` with an explicit brush.
pub fn stroke_with(engine: &mut Engine, b: BrushParams, points: &[Vec2]) {
    engine.process(InputCommand::SetBrush(b));
    let mut it = points.iter();
    let first = *it.next().expect("at least one point");
    engine.process(InputCommand::StartStroke {
        tool: Tool::Brush,
        sample: InputSample::at(first),
    });
    for &p in it {
        engine.process(InputCommand::StrokeTo {
            sample: InputSample::at(p),
        });
    }
    engine.process(InputCommand::EndStroke);
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
    for (pa, pb) in a.pixels.chunks_exact(4).zip(b.pixels.chunks_exact(4)) {
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
    assert_eq!((a.width, a.height), (b.width, b.height), "image size mismatch");
    let mut bad = 0u64;
    for (pa, pb) in a.pixels.chunks_exact(4).zip(b.pixels.chunks_exact(4)) {
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
pub fn assert_golden(name: &str, img: &RgbaImage, tol: u8) {
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
    for (a, b) in img.pixels.chunks_exact(4).zip(golden.pixels.chunks_exact(4)) {
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
    assert_eq!(info.color_type, png::ColorType::Rgba, "golden must be RGBA8");
    buf.truncate(info.buffer_size());
    RgbaImage::new(info.width, info.height, buf)
}
