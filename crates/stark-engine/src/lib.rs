//! Stark engine core — the frontend-agnostic GPU painting backend (CLAUDE.md).
//!
//! # Where to start
//!
//! [`Engine`] owns everything and is the only entry point. Two things go in and
//! one comes out:
//!
//! - [`InputCommand`] — user intent, one-way (§4). It splits by which
//!   class of state it touches: [`command::DocCommand`] mutates the document
//!   (historized, replicated, replayed), [`command::ViewCommand`] mutates view
//!   state (per-client, transient), and [`command::GestureCommand`] is the
//!   press-drag-release lifecycle that builds in view state and commits a
//!   document action at the end.
//! - **Requests** — the operations that must answer, and so cannot be commands:
//!   [`Engine::import_brush`], [`Engine::save_bytes`], [`Engine::merge_remote`],
//!   [`Engine::pick_color`], and friends. They stay direct methods (§4).
//! - [`ObservableState`] — the cheap UI-facing projection, read back after each
//!   command.
//!
//! # The layers underneath
//!
//! - [`document`] — the versioned document: [`Action`](stark_model::document::Action)s, the
//!   [`document::Timeline`] that orders them (linear solo, replicated when
//!   shared), and [`document::DocState`], a persistent map of copy-on-write
//!   tiles (§5).
//! - [`session`] — view state: tool, brush, view transform, the in-flight
//!   gesture (§3), plus the half of it that is published to collaborators.
//! - [`peer`] — presence: per-client state every client reads and only its owner
//!   writes, held outside the timeline because replay does not need it
//!   (§17.4). The one piece of per-client state that *is* needed by
//!   replay — the selection — lives in [`document::DocState`] keyed by
//!   [`ActorId`](stark_model::document::ActorId) instead (§17.3).
//! - [`gpu`] — the tile pool, the stroke renderer, compositing and the media
//!   pass (§6).
//! - [`path`] — pointer samples fitted to a cubic B-spline (`spline`), then
//!   flattened adaptively into the segments the brush sweeps along (§6.2).
//! - `stark-model`'s `io` — the save format, which *is* the action log (§8).
//! - [`timing`] — where a frame's time goes: one histogram per named phase of the
//!   pipeline, collected in the shipped app as well as under a benchmark (§7.1).
//!
//! # What is public, and what is only re-exported
//!
//! A module here is `pub` **or** its types are re-exported below, never both —
//! `document` and `gpu` state the rule for their own submodules and it is the same
//! one: two public paths to a type are two ways to spell an import and no way to
//! tell which one the crate meant to offer.
//!
//! So the modules a caller genuinely navigates (`document`, `command`, `geom`,
//! `path`, …) are `pub` and curate their own surface, while the ones that exist to
//! *serve* them are `pub(crate)` with their public types re-exported at the root:
//! `error`, `image` and `content` were reached only that way already, and `assist`,
//! `noise`, `spline`, `guides` and `tow` were reached from outside the crate for a
//! grand total of two names — both of which the re-export list already carried.
//!
//! The practical reading: `stark_engine::RgbaImage`, not `stark_engine::image::RgbaImage`.
//!
//! Build status lives in §13, not here — one checklist, so there is nothing to drift.

pub mod assets;
pub(crate) mod assist;
pub mod colorspace;
pub mod command;
pub mod document;
pub mod engine;
pub(crate) mod error;
pub mod filters;
pub mod gpu;
pub(crate) mod guides;
pub(crate) mod image;
pub(crate) mod noise;
pub mod path;
pub mod peer;
mod presence;
pub mod session;
pub(crate) mod spline;
pub mod timing;
pub(crate) mod tow;

/// Take a lock whose only contents are a **cache, a free list or a tally**, poisoned
/// or not.
///
/// Every `Mutex` this crate holds guards derived state — a baked tip texture
/// (`gpu::stroke::tips`), a pooled scratch list (`gpu::stroke::scratch`), the seed of
/// the last stroke complained about, the timing histograms ([`timing`]) — whose
/// values are moved in whole after the work producing them has finished, so a panic
/// while the lock is held cannot leave a torn value behind. All poisoning tells us is
/// that some *other* thread panicked while it happened to be looking something up;
/// propagating that as a panic of our own turns one thread's failure into a dead
/// renderer, which is a worse answer than a cold cache.
///
/// At the crate root rather than beside any one of them, because the argument is
/// about what this crate puts *in* a `Mutex` and not about which module did it — and
/// the second module to want it is what made that worth saying once.
/// `std::result::Result` spelled out, because this crate re-exports an
/// [`EngineError`]-shaped `Result` below and a bare one here would resolve to that.
pub(crate) fn unpoisoned<'a, T>(
    lock: std::result::Result<
        std::sync::MutexGuard<'a, T>,
        std::sync::PoisonError<std::sync::MutexGuard<'a, T>>,
    >,
) -> std::sync::MutexGuard<'a, T> {
    lock.unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub use assets::AssetStore;
pub use colorspace::ColorSpace;
pub use command::{InputCommand, InputSample};
pub use document::Selection;
pub use engine::{
    Background, DEFAULT_HISTORY_BUDGET, Engine, EngineShared, ExportPlan, ExportScale, LayerInfo,
    Layers, MatteInfo, ObservableState, PickOptions, PickSource, PresenceTick, Rendered,
};
pub use error::{EngineError, Result};
pub use gpu::{
    Compositor, CompositorPipeline, DeviceFailure, EnvironmentId, FailureKind, GpuContext,
    GpuHealth, MediaParams, Offscreen, StrokeRenderer, TilePairHandle, TilePool,
};
pub use guides::{AxisPencil, AxisPlane, GuideScene, Lens, PairTrace, PerspectiveGuide, Scaffold};
pub use image::RgbaImage;
pub use peer::{LiveGesture, Peer, Peers};
pub use tow::TowString;
