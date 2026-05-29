//! Stark engine core — the frontend-agnostic GPU painting backend (DESIGN.md).
//!
//! Build progress (DESIGN.md §13 build order):
//! - [x] Step 1: GPU + tiles skeleton — [`GpuContext`], [`TilePool`],
//!   [`Presenter`] rendering tiles under a [`ViewTransform`].
//! - [x] Step 2: stroke MVP — the command/action split ([`InputCommand`] vs
//!   [`document::Action`]), color stamping with copy-on-write tiles, and the
//!   `history`-backed [`document::LinearTimeline`] driving undo/redo via
//!   [`Engine`].
//! - [ ] Step 3+: golden harness, multi-channel media, save/load, collaboration.

pub mod command;
pub mod document;
pub mod engine;
pub mod error;
pub mod geom;
pub mod gpu;
pub mod session;

pub use command::{InputCommand, InputSample};
pub use engine::{Engine, ObservableState};
pub use error::{EngineError, Result};
pub use geom::{Extent2, TileCoord, Vec2, ViewTransform, TILE_SIZE};
pub use gpu::{GpuContext, Presenter, StrokeRenderer, TileHandle, TilePool, COLOR_FORMAT};
