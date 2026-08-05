//! GPU subsystem: device context, the recycling tile pool, stroke rasterization,
//! selection masks, and compositing/media (§6, §7).

pub mod composite;
pub mod context;
pub(crate) mod desc;
pub mod environment;
pub mod fill;
pub mod pigment;
pub mod readback;
pub mod registry;
pub mod selection;
pub mod stroke;
pub mod surface;
pub mod tile;
pub mod transform;
mod wesl;

pub use composite::{
    CompositeGroup, CompositeItem, CompositeScene, Compositor, CompositorPipeline, GroupContent,
    MatteDraw, MediaParams, Offscreen, SelectionOutline,
};
pub use context::GpuContext;
pub use environment::{Environment, EnvironmentId};
pub use fill::FillRenderer;
pub use registry::{Registry, Resource};
pub use selection::SelectionRenderer;
pub use stroke::{StrokeRenderer, StrokeSpans};
pub use surface::{Surface, SurfaceId};
pub use tile::{MaskHandle, TilePairHandle, TilePool};
pub use transform::TransformRenderer;
