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
//! - [x] Step 7: brush shapes & assets — content-addressed [`assets::AssetStore`]
//!   coverage masks, [`document::BrushShape`] (`Round`/`Stamp`), path-following
//!   rotated stamps, [`Engine::import_brush`], and referenced assets bundled
//!   into the save file as compact grayscale PNGs (DESIGN §6.6, §8).
//! - [x] Step 8: cubic stroke interpolation (DESIGN §6.2) — [`path`] fits raw
//!   samples to spline control points (RDP) and flattens a centripetal
//!   Catmull–Rom curve for stamping. Kills stair-stepping, shrinks the log.
//! - [x] Step 8b: continuous swept-segment stamping (DESIGN §6.2) — each segment
//!   is one quad whose coverage is the brush swept along it via a precomputed
//!   prefix-τ texture (`τ=−ln(1−α)`); over-blend sums depth exactly. Removes the
//!   discrete-dab artifact with hard tips.
//! - [x] Step 8c: tile aprons (DESIGN §6.4) — tiles carry a `TILE_APRON` halo
//!   (`TILE_TEX` textures) rendered, not copied, so the compositor's bilinear
//!   filter reads across tile boundaries instead of clamping. Kills the lighting
//!   seams the media pass amplified under zoom/sub-pixel pan (`tests/seam.rs`).
//! - [x] Step 9: pluggable color spaces (DESIGN §6.7) — [`colorspace::ColorSpace`]
//!   trait, [`colorspace::OkLabColorSpace`] (migrated, no behavior change), and
//!   [`colorspace::PigmentColorSpace`]: four-pigment Kubelka–Munk ([`pigment`])
//!   with additive deposition + NNLS RGB→pigment picker. Engine selects via
//!   [`Engine::new_with_color_space`]/[`Engine::set_color_space`]. (UI toggle TBD.)
//! - [ ] Step 10: brush file upload · Step 11: collaboration.

pub mod assets;
pub mod color;
pub mod colorspace;
pub mod command;
pub mod pigment;
pub mod document;
pub mod engine;
pub mod error;
pub mod geom;
pub mod gpu;
pub mod image;
pub mod io;
pub mod path;
pub mod session;

pub use assets::{AssetId, AssetStore};
pub use colorspace::{ColorSpace, ColorSpaceId};
pub use command::{InputCommand, InputSample};
pub use engine::{Engine, LayerInfo, ObservableState};
pub use error::{EngineError, Result};
pub use geom::{Extent2, TileCoord, Vec2, ViewTransform, TILE_SIZE};
pub use gpu::{
    Compositor, GpuContext, MediaParams, Presenter, StrokeRenderer, TileHandle, TilePool,
};
pub use image::RgbaImage;
pub use io::{BuildId, CanvasMeta, DocumentFile};
