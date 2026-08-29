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

pub(crate) use composite::{BlendPass, FilterPass};
pub use composite::{
    CompositeGroup, CompositeItem, CompositeScene, Compositor, CompositorPipeline, FilterDraw,
    GroupContent, MatteDraw, MediaParams, Offscreen, SelectionOutline,
};
pub use context::{DeviceFailure, FailureKind, GpuContext, GpuHealth};
pub use environment::{Environment, EnvironmentId};
pub use fill::FillRenderer;
pub use merge::{MergeRenderer, MergeSide};
pub use place::PlaceRenderer;
pub use registry::{Registry, Resource};
pub use selection::SelectionRenderer;
pub(crate) use stroke::StrokeSpans;
pub use stroke::{StrokeRenderer, max_stretch, max_tip_reach};
pub use substrate::{Substrate, SubstrateMap};
pub use tile::{
    AllocSource, INTERIOR_UV_BIAS, INTERIOR_UV_SCALE, MASK_TEX, MaskHandle, TilePairHandle,
    TilePool, mask_tex_origin,
};
pub use transform::TransformRenderer;
