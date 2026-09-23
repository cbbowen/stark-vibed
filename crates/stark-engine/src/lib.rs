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
//! - [`session`] — view state: the tool, the brush, what they are aimed at and the
//!   in-flight gesture (§3). The pan/zoom ([`view`]) is the engine's own field, for
//!   the reason `session`'s own doc gives.
//! - [`peer`] — presence: per-client state every client reads and only its owner
//!   writes, held outside the timeline (§17.4). The selection is the one piece of
//!   per-client state replay does need, so it lives in [`document::DocState`]
//!   keyed by [`ActorId`](stark_model::document::ActorId) instead (§17.3). What
//!   either side has in *flight* is named once, in `gesture`.
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
//! What a frontend needs, each name by one path: a module is `pub` **or** its types
//! are re-exported below, never both, so `stark_engine::RgbaImage` and not
//! `stark_engine::image::RgbaImage`. That includes every type a frontend-facing
//! signature carries, so it can be named. What only the suite, the benchmarks and
//! the examples need is offered by [`testing`], except [`path`]'s, which they reach
//! at its own path. Everything else is `pub(crate)` or narrower, and
//! `unreachable_pub` holds every `pub` to that.
//!
//! Build status lives in §13, not here.

// **The auto-trait proofs here are deep**, and the default limit of 128 no longer
// reaches the end of them: `Mutex<HashMap<AssetId, assets::Mask>>: Send` unfolds
// through the shape cache, a `wgpu::TextureView`, wgpu's dispatch enum and wgpu-core's `Global`, and
// runs out before it settles. Overrunning the limit is a future-incompatibility
// warning today (`recursion_depth_exceeding_limit`, rust#159228) and a hard error
// later; it is attached to the whole crate, so it cannot be quietened where it
// arises, and the compiler names this as the fix. Nothing here earned it — wgpu's
// own nesting grew.
#![recursion_limit = "256"]
// Here rather than in the workspace `[lints]`, which every crate shares.
#![warn(unreachable_pub)]

pub(crate) mod assets;
pub(crate) mod colorspace;
pub mod command;
pub mod document;
pub(crate) mod engine;
pub(crate) mod error;
pub mod filters;
pub(crate) mod gesture;
pub(crate) mod gpu;
pub mod guides;
pub(crate) mod image;
pub(crate) mod noise;
pub(crate) mod peer;
pub(crate) mod pictures;
mod presence;
mod projection;
pub(crate) mod session;
pub mod timing;
/// The canvas view: pan, zoom, rotation and the mirror (§18.1.2). Session
/// state, which is why it is here and not in the document crate.
pub(crate) mod view;

// The CPU geometry is `stark-stroke`'s (CLAUDE.md): it names no wgpu type, so its own
// tests do not wait on this crate. Aliased at the paths the rest of the engine spells
// it by, which is the whole of what the move cost the call sites.
pub(crate) use stark_stroke::{assist, detmath, tow};

/// Stroke path fitting and adaptive flattening — [`stark_stroke::path`], at the path
/// frontends and the suite name it by.
///
/// Listed name by name rather than re-exported whole, because the engine's public
/// surface is what a frontend needs and the geometry crate's is wider: the arc
/// primitives and the input tolerance below are the stroke renderer's, not anyone
/// else's.
pub mod path {
    pub use stark_stroke::path::{
        DEFAULT_TOLERANCE, FLATTEN_TOLERANCE, FlattenTolerance, IntermediateSample, MAX_TOLERANCE,
        MIN_TOLERANCE, PathFitter, cumulative_lengths, fit, fit_with_tolerance, flatten,
        flatten_spans_from, locate, point_at, span_count, span_end,
    };

    pub(crate) use stark_stroke::path::{
        Arc, InputTolerance, arc_at, arc_sagitta, fit_arc, fit_at, flatten_spans, frozen_spans_for,
    };
    // The arc series' domain and the half-turn it caps, for the one guard that ties
    // them to the bleed walk-back that widened it (`gpu::stroke::dynamics::bleed`).
    #[cfg(test)]
    pub(crate) use stark_stroke::path::{MAX_ARC_TURN, MAX_HALF_TURN_SIN};
}

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

/// The suite's and the benchmarks' own harness, and the diagnostic reads only they
/// make — see the module (§9).
#[doc(hidden)]
pub mod testing;

pub use assist::Assisted;
pub use colorspace::{all_available, available};
pub use engine::{
    Background, DEFAULT_FAST_COMMIT, DEFAULT_HISTORY_BUDGET, Engine, EngineShared, ExportPlan,
    ExportScale, GuideInfo, Guides, LayerInfo, Layers, MatteInfo, ObservableState, PickOptions,
    PickSource, PresenceTick, Projected, Rendered,
};
pub use error::{EngineError, ExportError, Produces, Result};
pub use gpu::{
    DeviceFailure, EnvironmentId, FailureKind, GpuContext, GpuHealth, MediaParams, Offscreen,
    Output, Transfer, max_stretch, max_tip_reach,
};
pub use image::RgbaImage;
pub use peer::{Identity, Peer};
pub use tow::TowString;
pub use view::{Extent2, ViewTransform};
