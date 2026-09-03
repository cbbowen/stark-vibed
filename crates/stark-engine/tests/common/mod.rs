//! Shared test harness: headless engine setup and golden-image comparison
//! (§9). Included by integration tests via `mod common;`.
//!
//! One of the two files in the workspace still named `mod.rs`: Cargo would take a
//! `tests/common.rs` for an integration test of its own, so the harness has to hide
//! in a directory. `examples/common` is the same case.

#![expect(dead_code, reason = "not every test binary uses every helper")]

pub mod corpus;

use stark_engine::command::Tool;
use stark_model::Srgb;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::{Engine, RgbaImage};
use stark_model::ColorSpaceId;
use stark_model::document::LayerId;
use stark_model::document::{BrushParams, BrushShape};
use stark_model::geom::{Extent2, TILE_APRON, TILE_SIZE, TILE_TEX, Vec2};

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
/// substrate. Matches the engine's `DEFAULT_SUBSTRATE_COLOR`. Neutral on purpose: a warm
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

/// Set to `1` to run the golden tests but not *compare* their output — see
/// [`assert_golden`]. For adapters other than the one the goldens were blessed on.
const SKIP_GOLDEN: &str = "STARK_SKIP_GOLDEN";

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// The device every engine in this test binary is built on.
///
/// **One device per process, not one per test**, which is where this suite's time
/// went. Measured on the machine that wrote this: a fresh `headless_engine` — request
/// an adapter, request a device, then compile ~19 shaders and ~30 pipelines — is
/// ~338 ms, while building an engine on a device that already exists is ~22 ms,
/// because the driver has the compiled shaders. At 386 construction sites the
/// difference is minutes of wall clock, and "the test suite is slow — run it once"
/// was a rule contributors had to remember in place of fixing it.
///
/// A `OnceLock` rather than a `thread_local`: libtest runs each test on a thread of
/// its own, so a per-thread donor would be a per-test donor again. `GpuContext` is
/// `Send + Sync` on native — wgpu's handles are, and the health cell is an
/// `Arc<Mutex<..>>` — and tests are native by construction, so the lock is all the
/// sharing needs.
///
/// **What is *not* shared is as important.** Each engine still builds its own
/// `TilePool`, its own renderers and its own document, because `Engine::new` does; so
/// a test that counts pooled textures (`tests/tile_pool.rs`) or paints on a document
/// sees nothing of any other test. What is shared is the device and the driver's
/// shader cache, neither of which any test asserts on.
///
/// Two things to know if a test ever behaves oddly because of this. Tests in one
/// binary now run concurrently against a *single* device rather than one each, which
/// is less driver stress rather than more, but is a different shape of it. And the
/// device's health cell (§5) is shared, so a test that deliberately provoked a wgpu
/// validation error would leave every later test in that binary observing a failed
/// GPU — no test does, and one that wants to should build its own context.
fn shared_context() -> Option<&'static stark_engine::GpuContext> {
    static CTX: std::sync::OnceLock<Option<stark_engine::GpuContext>> = std::sync::OnceLock::new();
    CTX.get_or_init(|| {
        // The decision — skip or fail — is `stark_engine::testing`'s, so this harness,
        // `tests/tile_pool.rs` and `benches/stroke.rs` cannot come to disagree about
        // what a missing adapter means. The blocking and the caching stay here.
        stark_engine::testing::or_skip(
            pollster::block_on(stark_engine::GpuContext::headless()),
            "GPU tests",
        )
    })
    .as_ref()
}

/// An engine of `size` in `space`, on this binary's shared device — or `None` where
/// there is no adapter and `stark_engine::testing::ALLOW_NO_GPU` permits the skip.
///
/// The one constructor every helper below funnels through, so what "a test engine is"
/// is stated once. The engine comes back on whatever environment it booted with — the
/// procedural `Neutral`.
fn build(size: Extent2, space: ColorSpaceId) -> Option<Engine> {
    let ctx = shared_context()?;
    match Engine::new_with_color_space(ctx.clone(), TARGET, size, space) {
        Ok(e) => Some(e),
        // Not an adapter problem — the device is already up. A space this build does
        // not carry (Mixbox without the feature) is the only way here, and it is a
        // legitimate skip rather than a failure.
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

/// Build a headless engine, or `None` if this machine has no usable adapter and
/// skipping is permitted (see [`shared_context`]).
///
/// **On the procedural `Neutral` environment** — the reference light (§6.3), whose
/// exposure is 1.0 and whose whole purpose is to be the identity a color can be
/// judged against. Install an image-based HDR here instead and every claim the suite
/// makes about a rendered byte becomes a claim about a *particular decoded environment
/// map*: a warm tint stands between each test and the paint it is checking, near-white
/// paper reads red-dominant by ~33 levels, and a difference in the pipeline reaches
/// the goldens multiplied by whatever the light happens to be doing there. Under the
/// reference light what a test reads back is what the pipeline produced.
/// [`engine_or_skip_studio`] keeps the image-based path covered.
pub fn engine_or_skip() -> Option<Engine> {
    build(SIZE, ColorSpaceId::Oklab)
}

/// A headless engine with a chosen viewport, for tests whose stroke has to be far
/// wider than the default [`SIZE`] window shows.
pub fn engine_or_skip_sized(size: Extent2) -> Option<Engine> {
    build(size, ColorSpaceId::Oklab)
}

/// A headless engine on the **blue** substrate ([`BG`]) rather than the default
/// paper.
///
/// The substrate is document state (§15.5), so choosing it is a
/// logged edit like any other — which is exactly how a user would do it. Blue is
/// deliberate for these tests: they ask "is there paint here?" by channel dominance,
/// and a saturated substrate answers in the *other* direction, so `is_blue` reads "bare
/// canvas" as positively as `is_red` reads "paint". Grey paper can only ever fail
/// `is_red`, which a black frame would too.
pub fn engine_or_skip_blue() -> Option<Engine> {
    engine_or_skip().map(on_blue)
}

/// [`engine_or_skip_sized`] on the blue substrate.
pub fn engine_or_skip_sized_blue(size: Extent2) -> Option<Engine> {
    engine_or_skip_sized(size).map(on_blue)
}

pub fn on_blue(mut engine: Engine) -> Engine {
    engine.process(stark_engine::command::DocCommand::SetSubstrateColor(
        Srgb::new([BG.r as f32, BG.g as f32, BG.b as f32]),
    ));
    engine
}

/// A headless engine rendering to a chosen target format.
///
/// Exists because everything else here uses [`TARGET`] (`Rgba8Unorm`) while a real
/// browser substrate is usually `Bgra8Unorm` — a difference no single-format test
/// can see, and one that silently swapped red and blue in exported PNGs.
pub fn engine_or_skip_in_format(format: wgpu::TextureFormat) -> Option<Engine> {
    // Not through `build`, which pins [`TARGET`] — the whole point here is a
    // different one. Same shared device.
    Some(Engine::new(shared_context()?.clone(), format, SIZE))
}

/// A headless engine in a chosen color space (§6.7).
pub fn engine_or_skip_with(id: ColorSpaceId) -> Option<Engine> {
    build(SIZE, id)
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
        engine
            .register_environment(stark_engine::EnvironmentId::Ferndale, hdr)
            .expect("the bundled studio HDR decodes");
        engine.process(ViewCommand::SetEnvironment(
            stark_engine::EnvironmentId::Ferndale,
        ));
        engine
    })
}

pub fn brush(color: [f32; 3], radius: f32) -> BrushParams {
    BrushParams {
        effect: stark_model::document::BrushEffect::painted(color),
        size: radius,
        // `drain` is quoted per *radius* (§6.2), and this helper's radius is its
        // caller's — so the old flat `0.0015` per canvas px is written as the product
        // that restores it at whatever tip is asked for. Every golden that comes
        // through here therefore renders the exact stroke it was blessed on, which is
        // the point: the unit change is a change of interpretation, and no test in
        // the suite was measuring the interpretation.
        drain: 0.0015 * radius,
        shape: BrushShape::Round { hardness: 0.8 },
        ..BrushParams::default()
    }
}

/// Paint and commit a stroke through `points` with an explicit brush.
pub fn stroke_with(engine: &mut Engine, b: BrushParams, points: &[Vec2]) {
    engine.process(ViewCommand::set_brush(b));
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

/// [`stroke_with`], each point pressed at its own pressure — for a claim about
/// what the pen does *along* one stroke, where every other stroke here is
/// pressed home throughout.
pub fn stroke_pressed(engine: &mut Engine, b: BrushParams, points: &[(Vec2, f32)]) {
    let sample = |&(pos, pressure): &(Vec2, f32)| InputSample {
        pos,
        pressure,
        ..InputSample::default()
    };
    engine.process(ViewCommand::set_brush(b));
    let mut it = points.iter();
    let first = sample(it.next().expect("at least one point"));
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: first,
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    for p in it {
        engine.process(GestureCommand::To { sample: sample(p) });
    }
    engine.process(GestureCommand::End);
}

/// Paint and commit a stroke through the given canvas points with `color`.
pub fn paint(engine: &mut Engine, color: [f32; 3], radius: f32, points: &[Vec2]) {
    stroke_with(engine, brush(color, radius), points);
}

/// Commit a stroke through `points` as a **replay** rather than a gesture: one
/// commit, rendered whole at the fold.
///
/// That is the canonical render — what a file, a redo and a peer each produce —
/// where [`stroke_with`] lands the live preview's own tiles (`PreparedStroke`,
/// §6.2), which sit within [`SEAM_LEVELS`] of it. A test whose claim is about
/// replay fidelity draws the reference side with this, so the claim stays exact.
pub fn replay_with(engine: &mut Engine, b: BrushParams, points: &[Vec2]) {
    engine.process(ViewCommand::set_brush(b));
    let samples: Vec<InputSample> = points.iter().copied().map(InputSample::at).collect();
    engine.replay_stroke(Tool::Brush, &samples);
}

/// The committed document rendered again **whole**: undo and redo re-fold the last
/// action through `apply`, which has no preview to take, so a stroke committed from
/// its live preview (`PreparedStroke`, §6.2) comes back as the one-pass render a
/// replay, a file or a peer would make of it.
///
/// What every `preview == committed` claim is held against. A live commit *is* its
/// preview now, so "the cut between head and tail does not matter" is a claim
/// about this render, not about the frame after pen-up. Solo engines only: a
/// shared session logs its undo rather than navigating.
pub fn whole_render(engine: &mut Engine) -> RgbaImage {
    engine.process(DocCommand::Undo);
    engine.process(DocCommand::Redo);
    engine.render_to_image()
}

/// Per-channel levels a stroke committed from its live preview may sit from the
/// same stroke rendered whole (§6.2). The preview is a frozen head with the tail
/// drawn over it, and the head is stored in f16 at the cut where one pass carries
/// f32 through — a level or two on a plain deposit. The corpus measures the bound
/// per stroke (`corpus::Tol::seam`: 4 on the swept path, up to 12 through a wide
/// reservoir); this is the swept figure, for the tests outside it that hold a live
/// commit against a replay.
pub const SEAM_LEVELS: u8 = 4;

/// The total **height** on `layer` — the channel §6.1 says is conserved, summed off
/// the tiles themselves rather than inferred from how dark the render came out.
///
/// **Interiors only.** Every tile's apron is a bit-identical copy of its neighbour's
/// edge (§6.4), so summing whole blocks counts the seams twice — which is invisible on
/// a one-tile stroke and grows with the tile count, exactly the way a conservation
/// drift would.
///
/// The committed document, so a caller mid-gesture is asking about the state before
/// the live tail.
pub fn total_height(engine: &Engine, layer: LayerId) -> f64 {
    let Some(coords) = engine
        .document()
        .layer(layer)
        .and_then(|l| l.tiles())
        .map(|t| t.keys().copied().collect::<Vec<_>>())
    else {
        return 0.0;
    };
    let lo = TILE_APRON as usize;
    let hi = lo + TILE_SIZE as usize;
    coords
        .iter()
        .filter_map(|c| engine.tile_channels(layer, *c))
        .map(|ch| {
            (lo..hi)
                .flat_map(|y| (lo..hi).map(move |x| y * TILE_TEX as usize + x))
                .map(|i| ch.height[i] as f64)
                .sum::<f64>()
        })
        .sum()
}

/// The **height** and per-unit **opacity** at one canvas point on `layer` — §6.1's two
/// channels, read off the tile rather than inferred from the pixel they produced.
///
/// `None` where the layer holds no tile there, which *is* the answer for bare canvas:
/// a tile that was never minted is not a tile of zeroes.
///
/// The interior, never the apron: a point maps to exactly one tile's interior texel,
/// and the copy of it in a neighbour's apron is the same value by §6.4 — so taking the
/// interior is a choice about which of two identical readings to name, and naming the
/// owner is the one that stays right if the seam rule ever breaks.
pub fn paint_at(engine: &Engine, layer: LayerId, canvas: Vec2) -> Option<(f32, f32)> {
    let tile = |v: f32| (v / TILE_SIZE as f32).floor() as i32;
    let coord = stark_model::geom::TileCoord::new(tile(canvas.x), tile(canvas.y));
    let local = canvas - coord.origin();
    let (x, y) = (
        local.x as usize + TILE_APRON as usize,
        local.y as usize + TILE_APRON as usize,
    );
    let ch = engine.tile_channels(layer, coord)?;
    let i = y * TILE_TEX as usize + x;
    Some((ch.height[i], ch.color[i * 4 + 3]))
}

/// The centre pixel — where the suites' standard stroke crosses.
pub fn center(img: &RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2)
}

/// Where canvas point `canvas` lands on a [`SIZE`] screen, at the default 1:1 view
/// centred on the origin.
pub fn screen_of(canvas: Vec2) -> (u32, u32) {
    let half = Vec2::new(SIZE.width as f32, SIZE.height as f32) * 0.5;
    let p = canvas + half;
    (p.x as u32, p.y as u32)
}

/// The RGB at canvas point `canvas`, as signed channels a difference can be taken in.
pub fn texel(img: &RgbaImage, canvas: Vec2) -> [i32; 3] {
    let (x, y) = screen_of(canvas);
    rgb(img.pixel(x, y))
}

/// A pixel's three color channels, signed — what [`leads`] and [`apart`] both compare.
pub fn rgb(c: [u8; 4]) -> [i32; 3] {
    [c[0] as i32, c[1] as i32, c[2] as i32]
}

/// The worst per-channel distance between two texels.
pub fn apart(a: [i32; 3], b: [i32; 3]) -> i32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).max().unwrap()
}

/// Which channel a probe is asking about.
///
/// An enum rather than an index because half the files that probe also declare a
/// brush color named `RED`, and the two would collide on import.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lead {
    Red,
    Green,
    Blue,
}

/// Whether channel `lead` of `c` exceeds **both** the others by at least `margin`.
///
/// **The one predicate under every "is there paint here?" in the suite.** It was
/// written out twelve times across nine test files, under four names — `is_red`,
/// `is_painted`, `red_dominant`, and the same three for the other channels — varying
/// along two axes the copies did not separate: which channel leads, and by how much.
/// The margins that resulted — 30, 40 and 60 — were not a considered range but the
/// order the files were written in, and no copy said which number was a threshold and
/// which was a leftover.
///
/// The margin stays a parameter, because it is not a formatting detail: about half
/// the call sites read `!`, so raising it weakens an assertion in one file and
/// strengthens it in the next. [`MARGIN`], [`MARGIN_FLAT`] and [`MARGIN_LIT`] are the
/// three the suite distinguishes; a file wanting a fourth should say why in a `const`
/// of its own rather than in a literal.
///
/// A *margin* rather than a level, because the canvas is not black: what separates
/// paint from paper is the hue, and a level would need re-tuning for every substrate
/// the suite paints on.
pub fn leads(c: [i32; 3], lead: Lead, margin: i32) -> bool {
    let i = lead as usize;
    (0..3).all(|k| k == i || c[i] > c[k] + margin)
}

/// The separation the compositor guarantees between the suite's paint and its paper.
pub const MARGIN: i32 = 30;

/// The separation asked for by the files that probe **flat** paint at a point.
///
/// Above [`MARGIN`] not because those renders are noisier but because their negative
/// assertions are the load-bearing ones: a fill test's job is largely to show paint
/// did *not* reach somewhere, and a gradient's far stop is a blend of two of these
/// colors that has to fail the test for the other one.
pub const MARGIN_FLAT: i32 = 40;

/// The separation needed where **lit bare paper reads warm on its own**.
///
/// The media/lighting pass (§6.5) tints the substrate towards the light, which is
/// enough to carry unpainted paper past [`MARGIN`] on red — `dynamics.rs` asserts
/// exactly that as its own precondition. It is a property of the render, not of the
/// tests, so the numbers are both right and the suite needs both.
pub const MARGIN_LIT: i32 = 60;

/// Whether a pixel reads as the suite's red paint rather than as canvas, at [`MARGIN`].
///
/// One copy with one margin, shared by every suite that asks "is there paint here?" —
/// so a render change that shifts channel separation moves every test the same
/// distance instead of being patched file by file.
pub fn red_dominant(c: [u8; 4]) -> bool {
    leads(rgb(c), Lead::Red, MARGIN)
}

/// [`red_dominant`] for the other color the suite paints in.
pub fn green_dominant(c: [u8; 4]) -> bool {
    leads(rgb(c), Lead::Green, MARGIN)
}

/// [`red_dominant`] asked of a **canvas point** on a [`SIZE`] render, at [`MARGIN_FLAT`].
pub fn painted(img: &RgbaImage, canvas: Vec2) -> bool {
    leads(texel(img, canvas), Lead::Red, MARGIN_FLAT)
}

/// The fraction of pixels that differ **at all**, and the worst per-channel
/// difference anywhere — in that order.
///
/// The two answer different questions and most callers want both: the fraction
/// separates a contiguous seam from a speck, the worst says how far the seam went.
/// [`frac_exceeding`] is the same fraction taken above a threshold.
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

    // **Worst-texel**, which is the same statistic the corpus battery holds every
    // other one of its checks to (`corpus::Report::check`) and is here for that
    // comparator's stated reason: "steps and seams are loud in the maximum and quiet
    // in the average".
    //
    // Counting a *fraction* instead passes while up to 1% of the frame differs by any
    // amount whatever. 1% of a 256² golden is 655 texels — a disc of radius 14, the
    // size of the lift-end ring, the settle crease and the stranded glob, each of
    // which has a bespoke test in `tests/dynamics.rs` because a golden did not catch
    // it. A fraction also hides ordinary drift: `corpus_curve` and `corpus_bleed` sat
    // 456 and 559 texels out (max 10 and 8) against un-re-blessed goldens, silently,
    // for as long as it stayed under 1%.
    //
    // What the fraction was really absorbing is visible in that measurement's own
    // shape — 2100 texels at 1 level, 3400 at 2, decaying to 2 at 10, scattered at
    // ~3% density over the stroke. Legitimate numeric drift is a *decaying tail over
    // a region*; an artifact is a *bump at high magnitude*. Only the maximum tells
    // them apart, so only the maximum is read, and `tol` is the whole bound.
    let mut worst = (0u32, 0usize);
    for (i, (a, b)) in img
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(golden.pixels.as_chunks::<4>().0)
        .enumerate()
    {
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        if d > worst.0 {
            worst = (d, i);
        }
    }
    let (d, i) = worst;
    if d as u8 > tol {
        let actual = path.with_extension("actual.png");
        write_png(&actual, img);
        panic!(
            "golden {name} mismatch: worst {d} > tol {tol}, at ({}, {}); wrote {}",
            i as u32 % img.width,
            i as u32 / img.width,
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
