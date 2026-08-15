//! GPU subsystem: device context, the recycling tile pool, stroke rasterization,
//! selection masks, and compositing/media (§6, §7).
//!
//! **The `pub use` list below is this subsystem's whole public surface.** The
//! modules are `pub(crate)`, so a `pub` item inside one is visible to the rest of
//! the engine and to nobody else until it is named here — which is the point: what
//! leaves this module should be a decision, not a consequence of how a file happened
//! to be split.
//!
//! It used to be both at once. Every module was `pub` *and* re-exported, so
//! `gpu::composite::Compositor` and `gpu::Compositor` were both public names for one
//! type and `engine.rs` used each in different places — with `readback` the lone
//! module that had no re-export, for no reason anyone had written down.

pub(crate) mod channels;
pub(crate) mod composite;
pub(crate) mod context;
pub(crate) mod desc;
pub(crate) mod environment;
pub(crate) mod fill;
pub(crate) mod half;
pub(crate) mod merge;
pub(crate) mod pigment;
pub(crate) mod readback;
pub(crate) mod registry;
pub(crate) mod selection;
pub(crate) mod stroke;
pub(crate) mod submit;
pub(crate) mod surface;
pub(crate) mod tile;
pub(crate) mod transform;
pub(crate) mod uniforms;

pub(crate) use composite::{BlendPass, FilterPass};
pub use composite::{
    CompositeGroup, CompositeItem, CompositeScene, Compositor, CompositorPipeline, FilterDraw,
    GroupContent, MatteDraw, MediaParams, Offscreen, SelectionOutline,
};
pub use context::GpuContext;
pub use environment::{Environment, EnvironmentId};
pub use fill::FillRenderer;
pub use merge::{MergeRenderer, MergeSide};
pub use registry::{Registry, Resource};
pub use selection::SelectionRenderer;
pub use stroke::{StrokeRenderer, StrokeSpans, ToolState};
pub use surface::{Surface, SurfaceId};
pub use tile::{AllocSource, MaskHandle, TilePairHandle, TilePool};
pub use transform::TransformRenderer;
