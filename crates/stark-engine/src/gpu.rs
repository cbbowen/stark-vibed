//! GPU subsystem: device context, the recycling tile pool, stroke rasterization,
//! selection masks, and compositing/media (§6, §7).
//!
//! **The `pub use` list below is this subsystem's whole public surface.** The
//! modules are `pub(crate)`, so a `pub` item inside one is visible to the rest of
//! the engine and to nobody else until it is named here — which is the point: what
//! leaves this module should be a decision, not a consequence of how a file happened
//! to be split.
//!
//! So a module is `pub(crate)` **or** re-exported, never both: two public names for
//! one type is two ways for a call site to spell the same import, and no way to tell
//! which one the module meant to offer.

pub(crate) mod channels;
pub(crate) mod composite;
pub(crate) mod context;
pub(crate) mod desc;
pub(crate) mod environment;
pub(crate) mod fill;
pub(crate) mod half;
pub(crate) mod merge;
pub(crate) mod pigment;
pub(crate) mod place;
pub(crate) mod readback;
pub(crate) mod registry;
pub(crate) mod scratch;
pub(crate) mod selection;
pub(crate) mod stroke;
pub(crate) mod submit;
pub(crate) mod substrate;
pub(crate) mod tile;
pub(crate) mod transform;
pub(crate) mod uniforms;

// `gpu` is `pub(crate)` (see `lib.rs`), so this list is what the *crate* reaches
// across module boundaries, not a public surface. Five names have left it for one
// reason — `GroupContent`, `MergeSide`, `Resource`, `MaskHandle` and `f32_to_f16`
// were re-exported for a public path nobody outside took, while the crate itself
// reached them through their own module, so the line had no reader at either end.
pub(crate) use composite::{BlendPass, FilterPass, export_format};
pub use composite::{
    CompositeGroup, CompositeItem, CompositeScene, Compositor, CompositorPipeline, FilterDraw,
    MatteDraw, MediaParams, Offscreen, Output, SelectionOutline, Transfer,
};
pub use context::{DeviceFailure, FailureKind, GpuContext, GpuHealth};
pub use environment::{Environment, EnvironmentId};
pub use fill::FillRenderer;
pub use merge::MergeRenderer;
pub use place::PlaceRenderer;
pub use registry::Registry;
pub use selection::SelectionRenderer;
pub(crate) use stroke::StrokeSpans;
pub use stroke::{StrokeRenderer, max_stretch, max_tip_reach};
pub use substrate::{Substrate, SubstrateMap};
/// Gated to match what it is *for*. `TileChannels` is what a tile reads back as, and
/// both ends of that — [`TilePairHandle::read_channels`] which mints one and
/// `Engine::tile_channels` which hands it to the suite — are
/// `#[cfg(not(target_arch = "wasm32"))]`, because a readback blocks and the browser
/// has no thread to block. Re-exported unconditionally, this was a name the wasm
/// build imported and could not use: the shape CLAUDE.md warns about, a `#[cfg]` that
/// drifted off the `use` it guarded.
#[cfg(not(target_arch = "wasm32"))]
pub use tile::TileChannels;
pub use tile::{
    AllocSource, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, MASK_TEX, TilePool, mask_tex_origin,
};
pub use transform::TransformRenderer;
