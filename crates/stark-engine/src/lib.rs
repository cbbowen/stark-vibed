//! Stark engine core: the derived view of the document (CLAUDE.md). No UI, no
//! windowing, and it compiles to wasm.
//!
//! # Where to start
//!
//! [`Engine`] owns everything and is the only entry point. Two things go in and
//! one comes out:
//!
//! - [`InputCommand`](command::InputCommand) — user intent, one-way (§4), split by
//!   which class of state it touches: [`command::DocCommand`] mutates the document
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
//! - [`document`] — [`Action`](stark_model::document::Action)s, the
//!   [`document::Timeline`] that orders them (linear solo, replicated when
//!   shared), and [`document::DocState`], a persistent map of copy-on-write
//!   tiles (§5).
//! - [`session`] — view state: tool, brush, view transform, the in-flight
//!   gesture (§3), plus the half of it that is published to collaborators.
//! - [`peer`] — presence: per-client state every client reads and only its owner
//!   writes, held outside the timeline (§17.4). The selection is the one piece of
//!   per-client state replay does need, so it lives in [`document::DocState`]
//!   keyed by [`ActorId`](stark_model::document::ActorId) instead (§17.3).
//! - [`gpu`] — the tile pool, the stroke renderer, compositing and the media
//!   pass (§6).
//! - [`path`] — pointer samples fitted to a cubic B-spline (`spline`), then
//!   flattened adaptively into the segments the brush sweeps along (§6.2).
//! - `stark-model`'s `io` — the save format, which *is* the action log (§8).
//! - [`timing`] — where a frame's time goes: one histogram per named phase of the
//!   pipeline (§7.1).
//!
//! # What is public
//!
//! A module here is `pub` **or** its types are re-exported below, never both: two
//! public paths to a type are two ways to spell an import and no way to tell which
//! one the crate meant to offer. So `stark_engine::RgbaImage`, not
//! `stark_engine::image::RgbaImage`. A name nothing outside `src/` spells is not
//! offered, and one only the suite spells is offered by [`testing`] instead.
//!
//! Build status lives in §13, not here.

pub(crate) mod assets;
pub(crate) mod assist;
pub(crate) mod colorspace;
pub mod command;
pub mod document;
pub(crate) mod engine;
pub(crate) mod error;
pub mod filters;
pub(crate) mod gpu;
pub(crate) mod image;
pub(crate) mod noise;
pub mod path;
pub(crate) mod peer;
pub(crate) mod pictures;
mod presence;
mod projection;
pub(crate) mod session;
pub(crate) mod spline;
pub mod timing;
pub(crate) mod tow;
/// The canvas view: pan, zoom, rotation and the mirror (§18.1.2). Session
/// state, which is why it is here and not in the document crate.
pub(crate) mod view;

/// Take a lock whose only contents are a **cache, a free list or a tally**, poisoned
/// or not.
///
/// Every `Mutex` this crate holds guards derived state whose value is moved in whole
/// after the work producing it has finished, so a panic while the lock is held cannot
/// leave a torn value behind. Poisoning then says only that some *other* thread
/// panicked while looking something up, and propagating it would turn one thread's
/// failure into a dead renderer — a worse answer than a cold cache.
///
/// **Every lock in the crate goes through here**, which is what makes that a property
/// of the crate rather than of the paths somebody remembered; a new `Mutex` belongs
/// on it too.
///
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

/// The suite's and the benchmarks' own harness, and the diagnostic methods only they
/// call — see the module (§9).
#[doc(hidden)]
pub mod testing;

pub use assist::Assisted;
// A frontend asks these to decide which color spaces to offer.
pub use colorspace::{all_available, available};
pub use engine::{
    Background, DEFAULT_FAST_COMMIT, DEFAULT_HISTORY_BUDGET, Engine, EngineShared, ExportPlan,
    ExportScale, GuideInfo, Guides, LayerInfo, Layers, MatteInfo, ObservableState, PickOptions,
    PickSource, PresenceTick, Projected, Rendered, headless_engine,
};
pub use error::{EngineError, ExportError, Produces, Result};
pub use gpu::{
    DeviceFailure, EnvironmentId, FailureKind, GpuContext, GpuHealth, MediaParams, Offscreen,
    Output, Transfer, max_stretch, max_tip_reach,
};
pub use image::RgbaImage;
pub use peer::{GestureView, Identity, LiveGesture, Peer};
pub use session::Session;
pub use tow::TowString;
pub use view::{Extent2, ViewTransform};
