//! What the test suite, the benchmarks and the examples need of this crate and a
//! shipping frontend does not (§9).
//!
//! An integration test is a separate crate, so anything it reaches has to be `pub`;
//! a name only they spell is offered here rather than at the root, and the diagnostic
//! reads on [`Engine`] are the [`Probe`](crate::testing::Probe) trait rather than
//! inherent methods.
//!
//! Not behind a cargo feature: every command in CLAUDE.md's list, and a
//! self-referential dev-dependency, would have to enable it.

use stark_model::SubstrateId;
use stark_model::geom::Vec2;
#[cfg(not(target_arch = "wasm32"))]
use stark_model::{document::LayerId, geom::TileCoord};

#[cfg(not(target_arch = "wasm32"))]
use crate::RgbaImage;
use crate::view::Extent2;
use crate::{Engine, Result};

pub use crate::document::audit::undeclared;
pub use crate::document::layer::{CompositeParams, Layer, LayerContent, PropValue};
pub use crate::document::state::{DocState, Guide};
pub use crate::document::timeline::TimelineStats;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::gpu::tile::TileChannels;
/// The recycling tile pool (§6.1), for the one test that drives it directly
/// (`tests/engine/tile_pool.rs`), and the tag `acquire_tex` demands of a caller.
pub use crate::gpu::tile::{AllocSource, TilePool};
pub use crate::peer::{GestureView, LiveGesture};
pub use crate::session::Session;

/// An engine on a headless device, in the default color space.
pub async fn headless_engine(
    target_format: wgpu::TextureFormat,
    viewport: Extent2,
) -> Result<Engine> {
    let gpu = crate::GpuContext::headless().await?;
    Engine::new_with_color_space(
        gpu,
        target_format,
        viewport,
        stark_model::ColorSpaceId::Oklab,
    )
}

/// What the suite reads off an [`Engine`] that pixels cannot show: caches, counters
/// and the raw channels under the lit composite. `use stark_engine::testing::Probe`.
pub trait Probe {
    /// The committed document.
    fn document(&self) -> &DocState;

    /// One committed tile's channels, straight off the GPU — height and alpha without
    /// the lit composite, which cannot separate "height was not conserved" from "the
    /// light changed" (§6.1). `None` where that layer has no paint tile. Reads the
    /// committed document, so a caller mid-gesture sees the state before the live tail.
    #[cfg(not(target_arch = "wasm32"))]
    fn tile_channels(&self, layer: LayerId, coord: TileCoord) -> Option<TileChannels>;

    /// The canvas at the viewport size, read back (§9) — the golden tests' backbone.
    /// Blocking, so native only; a frontend awaits [`Engine::export`] instead.
    #[cfg(not(target_arch = "wasm32"))]
    fn render_to_image(&mut self) -> RgbaImage;

    /// The canvas through the screen's passes into a half-float target, read back as
    /// four `f32` per texel, rows top-down, in the engine's [`Output`](crate::Output)
    /// transfer (§6.5). Only for an engine presenting to `Rgba16Float`.
    #[cfg(not(target_arch = "wasm32"))]
    fn render_to_floats(&mut self) -> Vec<f32>;

    /// Service a pending preview fold. A frontend gets this from
    /// [`Engine::render`]; a benchmark pays it explicitly to time it.
    fn flush_live(&mut self);

    /// How the timeline serviced materializations: the commutation fast paths versus
    /// rewind-and-replay (§12.6). Zeros when solo.
    fn timeline_stats(&self) -> TimelineStats;

    /// How many in-flight strokes the preview fold caches a settled head for (§17.6),
    /// after a flush. A head held for an ended gesture is not a wrong picture but a
    /// `DocState` of tile handles the pool cannot reclaim.
    fn live_head_count(&mut self) -> usize;

    /// How many of this client's stroke commits took the preview's tiles rather than
    /// rendering again (§6.2). Both paths are the same pixels by design.
    fn strokes_reused(&self) -> u64;

    /// The preview's invalidation epoch (§17.6): it advances whenever the document the
    /// in-flight gestures composite onto is replaced, including by an unlogged drag.
    fn preview_epoch(&self) -> u64;

    /// How many times the preview fold has moved what is shown (§17.6), after a flush —
    /// the term the draw list's cache key adds for a gesture in flight.
    fn preview_fold(&mut self) -> u64;

    /// The share of substrate `id`, at the document's scale, that a tip with this
    /// `give` and contact `softness` stands on travelling along `dir` (§6.4). Builds
    /// the substrate on the first ask.
    fn substrate_bearing(&self, id: SubstrateId, give: f32, softness: f32, dir: Vec2) -> f32;

    /// The tile memory history retention measures against its budget, idle textures
    /// included (§5).
    fn resident_tile_bytes(&self) -> u64;

    /// The most working-scratch bytes held out of the pool at once since the last
    /// ask, which restarts the count — a render's transient working set (§6.2).
    fn take_scratch_peak(&mut self) -> u64;
}

/// The environment variable that turns a missing GPU from a failure into a skip.
///
/// **A missing adapter is a failure unless this says otherwise**, and that is the
/// whole of why it exists. A skipped test still reports `ok`, so a suite that quietly
/// stopped finding a device would take the golden, seam and dynamics rounds green
/// having rendered nothing (CLAUDE.md). CI sets it because CI has no adapter; a
/// developer's machine must not.
const ALLOW_NO_GPU: &str = "STARK_ALLOW_NO_GPU";

/// Whether the environment permits skipping GPU work. Exactly `"1"`: an empty or
/// misspelt value must not read as permission.
fn allowed_to_skip() -> bool {
    std::env::var(ALLOW_NO_GPU).is_ok_and(|v| v == "1")
}

/// **The one place a missing GPU is decided about**: `Some` for something that built,
/// `None` for a permitted skip, and a panic otherwise. `what` names what is being
/// skipped, for the message.
///
/// The *blocking* and the caching stay with each caller — `pollster` is a
/// dev-dependency and has no business in the shipped crate — so what is shared here
/// is the judgement and not the plumbing. Every harness must route through it, or one
/// of them grows a silent skip.
pub fn or_skip<T, E: std::fmt::Display>(built: std::result::Result<T, E>, what: &str) -> Option<T> {
    match built {
        Ok(t) => Some(t),
        Err(e) if allowed_to_skip() => {
            eprintln!("skipping {what} ({ALLOW_NO_GPU}=1): {e}");
            None
        }
        Err(e) => panic!("no usable GPU adapter: {e}\nset {ALLOW_NO_GPU}=1 to skip {what}"),
    }
}

/// The one GPU context a test binary builds, kept for that binary's life.
///
/// **The caching is here; the blocking is still the caller's.** `build` runs at most
/// once, and `pollster` stays a dev-dependency of the suite rather than a dependency
/// of the shipped crate — which is the split the module note above draws, with the
/// cache moved across it and the plumbing left where it was.
///
/// What moving it buys is the crate the static is *proved* in.
/// `OnceLock<Option<GpuContext>>: Sync` unfolds through wgpu's dispatch enum and
/// wgpu-core's `Global` deeply enough to outrun the default recursion limit, and the
/// limit is a **crate** property: this crate raises it once (`lib.rs`), where every
/// test crate that shares the cache would have had to raise its own, and the next one
/// would have to remember.
///
/// Still one device per test *binary*, which is what it was when `tests/engine/common` held
/// the lock: a static in a linked rlib is per-binary, and each test target is its own
/// binary. What that means for a test — a shared device and driver shader cache, but
/// its own pool, renderers and document — `tests/engine/common` explains at the call site.
///
/// Native only, and not by taste: a `GpuContext` is `Send + Sync` on native — wgpu's
/// handles are, and the health cell is an `Arc<Mutex<..>>` — while on wasm they are
/// `Rc<Cell<..>>` and a `static` could not hold one at all. The suite is native by
/// construction, which is what let `tests/engine/common` keep this without a gate; the gate
/// is what the move across the crate boundary costs.
#[cfg(not(target_arch = "wasm32"))]
pub fn shared_context(
    build: impl FnOnce() -> Option<crate::GpuContext>,
) -> Option<&'static crate::GpuContext> {
    static CTX: std::sync::OnceLock<Option<crate::GpuContext>> = std::sync::OnceLock::new();
    CTX.get_or_init(build).as_ref()
}
