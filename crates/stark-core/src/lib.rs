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
//! - [x] Step 4: multi-channel + media pass — Oklab color ([`color`]), tiles
//!   carry color + `(height, wet)` aux, the brush deposits all channels with a
//!   load reservoir, and a [`gpu::Compositor`] composites then lights the
//!   impasto (normal-from-height + wet gloss) into display sRGB.
//! - [x] Step 5: save/load + timelapse — the [`io::DocumentFile`] action-log
//!   format (postcard + deflate), [`Engine::save_bytes`]/[`Engine::load_bytes`]
//!   with undo-after-load, and [`Engine::replay_timelapse`].
//! - [x] Step 6a: layers — active-layer selection (session), per-layer
//!   opacity/visibility/blend + reorder (historized actions), and opacity-aware
//!   compositing. [`ObservableState`] exposes the layer stack.
//! - [x] Step 6b: Dioxus UI — the `stark-ui` Dioxus **web** app drives the
//!   engine via [`InputCommand`]/[`ObservableState`] and paints through a
//!   **WebGPU surface** bound to the canvas (no readback). Backend runs in WASM.
//! - [x] Step 6c: navigation — pan (middle-drag) and cursor-anchored zoom
//!   (wheel) via [`ViewTransform::zoom_about`]; window-fit canvas + resize.
//!   (Tile LOD descoped to a future nice-to-have — DESIGN §13.)
//! - [ ] Step 7: collaboration.

pub mod color;
pub mod command;
pub mod document;
pub mod engine;
pub mod error;
pub mod geom;
pub mod gpu;
pub mod image;
pub mod io;
pub mod session;

pub use command::{InputCommand, InputSample};
pub use engine::{Engine, LayerInfo, ObservableState};
pub use error::{EngineError, Result};
pub use geom::{Extent2, TileCoord, Vec2, ViewTransform, TILE_SIZE};
pub use gpu::{
    Compositor, GpuContext, MediaParams, Presenter, StrokeRenderer, TileHandle, TilePool,
    COLOR_FORMAT,
};
pub use image::RgbaImage;
pub use io::{BuildId, CanvasMeta, ColorSpace, DocumentFile};
