//! The device every test engine is built on, the engines themselves, and the image
//! bytes an import wants.

// No `use super::*` here, and that is the layering: an engine is built before there is
// a mark on it, a reading of it or a golden to hold it to, so this module names none of
// its siblings and all four of them name it.
use stark_engine::Extent2;
use stark_engine::command::{DocCommand, Edit, ViewCommand};
use stark_engine::{Engine, EngineError};
use stark_model::Srgb;
use stark_model::document::LayerId;
use stark_model::{ColorSpaceId, DocError};

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

/// The device every engine in this test binary is built on.
///
/// **One device per process** — which under `cargo nextest`, the runner this suite is
/// meant to be run with, is one device per *test*: nextest gives every test its own
/// process, so this cell is initialized once and donates to nobody. It pays only where
/// one process runs many tests (`cargo test`), and there it is ~338 ms for a fresh
/// context — adapter, device, ~19 shaders, ~30 pipelines — against ~22 ms for an
/// engine on a device that already exists, because the driver has the compiled shaders.
///
/// So it is **not** what pays for the suite's wall clock as the suite is actually run;
/// the `gpu` test group in `.config/nextest.toml` is, by bounding how many tests ask
/// the driver for a device at once. Nothing here may assume a device it did not build.
///
/// A `OnceLock` rather than a `thread_local`: libtest runs each test on a thread of
/// its own, so a per-thread donor would be a per-test donor again. `GpuContext` is
/// `Send + Sync` on native — wgpu's handles are, and the health cell is an
/// `Arc<Mutex<..>>` — and tests are native by construction, so the lock is all the
/// sharing needs.
///
/// **What is *not* shared is as important.** Each engine still builds its own
/// `TilePool`, its own renderers and its own document, because `Engine::new` does; so
/// a test that counts pooled textures (`tests/engine/tile_pool.rs`) or paints on a document
/// sees nothing of any other test. What is shared is the device and the driver's
/// shader cache, neither of which any test asserts on.
///
/// Two things to know if a test ever behaves oddly because of this — both under
/// `cargo test`, since under nextest there is nothing to share. Tests in one binary
/// run concurrently against a *single* device rather than one each, which is less
/// driver stress rather than more, but is a different shape of it. And the
/// device's health cell (§5) is shared, so a test that deliberately provoked a wgpu
/// validation error would leave every later test in that binary observing a failed
/// GPU — no test does, and one that wants to should build its own context.
fn shared_context() -> Option<&'static stark_engine::GpuContext> {
    // The decision — skip or fail — is `stark_engine::testing`'s, so this harness,
    // `tests/engine/tile_pool.rs` and `benches/stroke.rs` cannot come to disagree about
    // what a missing adapter means. **The cache is that module's too**, and for a reason
    // that is nothing to do with sharing: the `OnceLock`'s `Sync` proof outruns the
    // default recursion limit, which is a crate property, and every test crate that
    // shares the cache would have had to raise it. The blocking stays here, so
    // `pollster` stays a dev-dependency.
    stark_engine::testing::shared_context(|| {
        stark_engine::testing::or_skip(
            pollster::block_on(stark_engine::GpuContext::headless()),
            "GPU tests",
        )
    })
}

/// An engine of `size` in `space`, on this binary's shared device — or `None` where
/// there is no adapter and `stark_engine::testing::ALLOW_NO_GPU` permits the skip, or
/// where this build does not carry `space`. Any other construction error panics.
///
/// The one constructor every helper below funnels through, so what "a test engine is"
/// is stated once. The engine comes back on whatever environment it booted with — the
/// procedural `Neutral`.
fn build(size: Extent2, space: ColorSpaceId) -> Option<Engine> {
    let ctx = shared_context()?;
    match Engine::new_with_color_space(ctx.clone(), TARGET, size, space) {
        Ok(e) => Some(e),
        // The device is already up, so the one legitimate skip is a space this build
        // does not carry (Mixbox without the feature). Anything else fails: a skip
        // reports `ok` having rendered nothing.
        Err(EngineError::Document(DocError::UnsupportedColorSpace(id))) => {
            assert!(
                !stark_engine::available(id),
                "{id:?} is in this build, yet the engine refused it"
            );
            eprintln!("skipping: {id:?} is not in this build");
            None
        }
        Err(e) => panic!("building a test engine failed: {e}"),
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

pub fn on_blue(mut engine: Engine) -> Engine {
    engine.process(DocCommand::Set(Edit::SubstrateColor(Srgb::new([
        BG.r as f32,
        BG.g as f32,
        BG.b as f32,
    ]))));
    engine
}

/// Add a layer at the top of the document and return its id.
pub fn add_layer(engine: &mut Engine) -> LayerId {
    engine.process(DocCommand::AddLayer {
        carrier: None,
        above: None,
    });
    top(engine)
}

/// The layer the most recent `AddLayer` created.
///
/// **Read back rather than predicted.** A `LayerId` is the id of the action that
/// minted it (§17.9), so what it is depends on how many actions the test has already
/// committed — exactly what a test should not be counting. A fresh layer is armed for
/// painting (§14.8), so the active one *is* the new one; every caller asks right after
/// the add, or of a document that has not moved the brush since.
pub fn top(engine: &Engine) -> LayerId {
    engine.observe().active_layer
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

/// A `size` render target in the format [`Engine::target_format`] reports, for
/// [`Engine::render`] — the path a frontend draws on, whose attachments the engine
/// keeps (§11).
///
/// **For anything that times a frame.** `Probe::render_to_image` allocates a
/// viewport-sized export target per call, and that allocation costs whatever state the
/// work before it left the GPU allocator in — a per-*picture* term that lands inside a
/// per-frame or per-move number and reads as growth (`composite.rs`, `stroke.rs`). A
/// target the caller holds across frames allocates nothing after the first.
///
/// The returned view keeps its texture alive, so callers need bind only the view.
pub fn kept_target(engine: &Engine, size: Extent2) -> wgpu::TextureView {
    engine
        .gpu()
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("kept frame target"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: engine.target_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// Block until the device has executed everything submitted so far.
///
/// [`Engine::render`] and [`Engine::process`] only *queue*, so a wall clock around
/// either measures command encoding until this has run.
pub fn drain(engine: &Engine) {
    engine
        .gpu()
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device drains");
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

/// A substrate of hard vertical stripes, 2048 texels wide — two map texels per canvas px
/// at natural scale, so which of a canvas px's two texels a path reads decides whether a
/// stripe's shoulder gates it. An odd period puts both parities on a shoulder (§6.4).
pub fn stripe_substrate() -> Vec<u8> {
    const W: u32 = stark_assetid::MAX_SUBSTRATE_DIM;
    const H: u32 = 8;
    const PERIOD: u32 = 25;
    let row: Vec<u8> = (0..W)
        .map(|x| if x % PERIOD < PERIOD / 2 { 200 } else { 0 })
        .collect();
    gray_png(W, H, &row.repeat(H as usize))
}

/// An 8-bit grayscale PNG of `w` × `h` row-major `pixels` — a height map or a brush
/// stamp, without reaching for a bundled asset.
pub fn gray_png(w: u32, h: u32, pixels: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, w, h);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer.write_image_data(pixels).expect("png body");
    writer.finish().expect("png finish");
    out
}

/// [`gray_png`] of a field sampled per texel: `f(x, y)` over `w` × `h`.
pub fn gray_png_from(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Vec<u8> {
    let pixels: Vec<u8> = (0..w * h).map(|i| f(i % w, i / w)).collect();
    gray_png(w, h, &pixels)
}

/// A canonical RGBA picture of `w` × `h` row-major `pixels`, encoded the way an
/// import expects (§23) — the placed-image counterpart of [`gray_png`].
pub fn picture_png(w: u32, h: u32, pixels: Vec<u8>) -> Vec<u8> {
    stark_assetid::Picture {
        width: w,
        height: h,
        pixels,
    }
    .encode()
    .expect("a well-formed picture")
}
