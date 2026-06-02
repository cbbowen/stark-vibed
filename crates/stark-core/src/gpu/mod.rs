//! GPU subsystem: device context, the recycling tile pool, stroke rasterization,
//! and compositing/media (DESIGN.md §6, §7).

pub mod composite;
pub mod context;
pub mod environment;
pub mod readback;
pub mod stroke;
pub mod surface;
pub mod tile;

pub use composite::{Compositor, MediaParams};
pub use context::GpuContext;
pub use environment::{Environment, EnvironmentId};
pub use stroke::StrokeRenderer;
pub use surface::{Surface, SurfaceId};
pub use tile::{TileHandle, TilePool};
