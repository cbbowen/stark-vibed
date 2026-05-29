//! GPU subsystem: device context, the recycling tile pool, and presentation
//! (DESIGN.md §6, §7). Stroke rendering and compositing land in later steps.

pub mod context;
pub mod present;
pub mod tile;

pub use context::GpuContext;
pub use present::Presenter;
pub use tile::{TileHandle, TilePool, COLOR_FORMAT};
