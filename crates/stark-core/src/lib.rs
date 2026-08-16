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
//! - [`document`] — the versioned document: [`document::Action`]s, the
//!   [`document::Timeline`] that orders them (linear solo, replicated when
//!   shared), and [`document::DocState`], a persistent map of copy-on-write
//!   tiles (§5).
//! - [`session`] — view state: tool, brush, view transform, the in-flight
//!   gesture (§3), plus the half of it that is published to collaborators.
//! - [`peer`] — presence: per-client state every client reads and only its owner
//!   writes, held outside the timeline because replay does not need it
//!   (§17.4). The one piece of per-client state that *is* needed by
//!   replay — the selection — lives in [`document::DocState`] keyed by
//!   [`document::ActorId`] instead (§17.3).
//! - [`gpu`] — the tile pool, the stroke renderer, compositing and the media
//!   pass (§6).
//! - [`path`] / [`spline`] — pointer samples fitted to a cubic B-spline, then
//!   flattened adaptively into the segments the brush sweeps along (§6.2).
//! - [`io`] — the save format, which *is* the action log (§8).
//!
//! Build status lives in §13, not here — one checklist, so there is nothing to drift.

pub mod assets;
pub mod assist;
pub mod color;
pub mod colorspace;
pub mod command;
pub mod content;
pub mod document;
pub mod engine;
pub mod error;
pub mod geom;
pub mod gpu;
pub mod gradient;
pub mod guides;
pub mod image;
pub mod io;
pub mod noise;
pub mod path;
pub mod peer;
mod presence;
pub mod session;
pub mod spline;
pub mod tow;

pub use assets::{AssetId, AssetStore};
pub use colorspace::{ColorSpace, ColorSpaceId};
pub use command::{InputCommand, InputSample};
pub use content::{AssetNeed, action_content};
pub use document::{LayerId, Selection, SelectionMode, SelectionOp, SelectionShape};
pub use engine::{
    Background, Engine, ExportPlan, ExportScale, LayerInfo, MatteInfo, ObservableState,
    PickOptions, PickSource, PresenceTick, Rendered,
};
pub use error::{EngineError, Result};
pub use geom::{Extent2, TILE_SIZE, TileCoord, Vec2, ViewTransform};
pub use gpu::{
    Compositor, CompositorPipeline, DeviceFailure, EnvironmentId, FailureKind, GpuContext,
    GpuHealth, MediaParams, Offscreen, StrokeRenderer, SurfaceId, TilePairHandle, TilePool,
};
pub use gradient::{Gradient, GradientStop};
pub use guides::{AxisPencil, AxisPlane, GuideScene, Lens, PairTrace, PerspectiveGuide, Scaffold};
pub use image::RgbaImage;
pub use io::{BuildId, CanvasMeta, DocumentFile};
pub use peer::{LiveGesture, Peer, PeerFrame, Peers};
