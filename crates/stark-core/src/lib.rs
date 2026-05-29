//! Stark engine core — the frontend-agnostic GPU painting backend (DESIGN.md).
//!
//! Build progress (DESIGN.md §13 build order):
//! - [x] Step 1: GPU + tiles skeleton — [`GpuContext`], [`TilePool`],
//!   [`Presenter`] rendering tiles under a [`ViewTransform`].
//! - [x] Step 2: stroke MVP — the command/action split ([`InputCommand`] vs
//!   [`document::Action`]), color stamping with copy-on-write tiles, and the
//!   `history`-backed [`document::LinearTimeline`] driving undo/redo via
//!   [`Engine`].
//! - [x] Step 3: history + golden harness — [`Engine::render_to_image`] for
//!   readback/export, golden-image tests, and determinism / undo-redo /
//!   replay-equivalence tests guarding the action-log invariant.
//! - [ ] Step 4+: multi-channel media, save/load, layers + UI, collaboration.

pub mod command;
pub mod document;
pub mod engine;
pub mod error;
pub mod geom;
pub mod gpu;
pub mod image;
pub mod session;

pub use command::{InputCommand, InputSample};
pub use engine::{Engine, ObservableState};
pub use error::{EngineError, Result};
pub use geom::{Extent2, TileCoord, Vec2, ViewTransform, TILE_SIZE};
pub use gpu::{GpuContext, Presenter, StrokeRenderer, TileHandle, TilePool, COLOR_FORMAT};
pub use image::RgbaImage;
