//! GPU subsystem: device context, the recycling tile pool, stroke rasterization,
//! selection masks, and compositing/media (§6, §7).
//!
//! `gpu` is itself `pub(crate)` (see [`crate`]), and its modules are too, so a `pub`
//! item inside one reaches the rest of the engine only once the re-export list below
//! names it. A module is `pub(crate)` **or** re-exported, never both: two public names
//! for one type is two ways to spell one import.

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

// A name earns a line here by being reached from *another* module.
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
/// Gated to match what it is *for*: both ends of a channel readback —
/// [`TilePairHandle::read_channels`](tile::TilePairHandle::read_channels) and
/// `Engine::tile_channels` — are native-only, because a readback blocks and the
/// browser has no thread to block.
#[cfg(not(target_arch = "wasm32"))]
pub use tile::TileChannels;
pub use tile::{
    AllocSource, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, MASK_TEX, TilePool, mask_tex_origin,
};
pub use transform::TransformRenderer;
