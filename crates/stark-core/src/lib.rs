//! Stark engine core — the frontend-agnostic GPU painting backend (DESIGN.md).
//!
//! Build progress (DESIGN.md §12 build order):
//! - [x] Step 1: GPU + tiles skeleton — [`GpuContext`], [`TilePool`],
//!   [`Presenter`] rendering tiles under a [`ViewTransform`].
//! - [ ] Step 2+: stroke engine, history, compositing, save/load, collaboration.

pub mod error;
pub mod geom;
pub mod gpu;

pub use error::{EngineError, Result};
pub use geom::{Extent2, TileCoord, ViewTransform, Vec2, TILE_SIZE};
pub use gpu::{GpuContext, Presenter, TileHandle, TilePool, COLOR_FORMAT};
