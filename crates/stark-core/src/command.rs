//! `InputCommand`: raw, high-frequency user intent (DESIGN.md §4).
//!
//! Commands are deliberately distinct from [`Action`](crate::document::Action)s.
//! Many commands are ephemeral (pointer moves mid-stroke, pan/zoom, tool changes)
//! and never enter history; only committed mutations become actions. The `Session`
//! (DESIGN.md §3) interprets commands and decides what, if anything, to commit.
//!
//! # The three kinds
//!
//! Which of the engine's two state classes a command touches decides almost
//! everything about it — whether it is logged, whether peers see it, whether undo
//! reaches it — so it lives in the type rather than in a comment:
//!
//! - [`DocCommand`] mutates **document state**: historized, replicated to peers,
//!   and reproduced by replay. Every one of these becomes an `Action`.
//! - [`ViewCommand`] mutates **view state**: per-client, transient, never logged
//!   and never sent. Two people sharing a drawing pan independently.
//! - [`GestureCommand`] is the press-drag-release lifecycle, which is neither: it
//!   *builds* in view state (`Session::in_flight`) and commits a document action
//!   at the end — or nothing at all, if cancelled.
//!
//! # What is deliberately *not* a command
//!
//! Commands are one-way: they carry intent in and nothing back, which is what lets
//! them become messages over a channel when the engine moves off the UI thread
//! (DESIGN.md §7). Anything that must answer — importing a brush and getting its
//! id, saving bytes, merging a remote action and learning whether it applied — is
//! a **request**, and requests stay direct methods on [`Engine`](crate::Engine)
//! until there is an actor to give them a reply channel. See DESIGN.md §4.

use serde::{Deserialize, Serialize};

use crate::document::{BlendMode, BrushParams, LayerId, SelectionMode, SelectionOp, Tool};
use crate::geom::{Extent2, Vec2};
use crate::gpu::{EnvironmentId, MediaParams, SurfaceId};

/// One pen/mouse sample in canvas space.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputSample {
    pub pos: Vec2,
    pub pressure: f32,
    pub tilt: Vec2,
    /// Timestamp in seconds, for velocity and timelapse (DESIGN.md §8).
    pub time: f64,
}

impl InputSample {
    /// A simple full-pressure sample (e.g. mouse input or tests).
    pub fn at(pos: Vec2) -> Self {
        Self {
            pos,
            ..Default::default()
        }
    }
}

impl Default for InputSample {
    fn default() -> Self {
        Self {
            pos: Vec2::ZERO,
            pressure: 1.0,
            tilt: Vec2::ZERO,
            time: 0.0,
        }
    }
}

/// Every stateful interaction the backend accepts (GOALS §Inputs, DESIGN.md §4).
///
/// Construct the inner enums directly and rely on `Into` — `engine.process(
/// ViewCommand::Pan { delta }.into())` — rather than spelling both levels out.
#[derive(Clone, Debug)]
pub enum InputCommand {
    Gesture(GestureCommand),
    Doc(DocCommand),
    View(ViewCommand),
}

/// The press-drag-release lifecycle, shared by painting and by the selection tools
/// (DESIGN.md §6.8): from the frontend's side both are one gesture, and the `tool`
/// decides which the session builds — a `StrokeRecord` or a `SelectionOp`.
///
/// In flight this is view state. [`GestureCommand::End`] is the only edge that
/// produces document state, and [`GestureCommand::Cancel`] produces none.
#[derive(Clone, Debug)]
pub enum GestureCommand {
    Start {
        tool: Tool,
        sample: InputSample,
        /// How finely this gesture's input actually resolves position, in canvas px
        /// — see [`PathFitter::with_tolerance`](crate::path::PathFitter::with_tolerance).
        ///
        /// The frontend states it because the frontend is the only thing that knows
        /// it: canvas px are 64× coarser zoomed out than zoomed in, and a pen
        /// digitizer, a touchscreen and a mouse each report at a different grain
        /// through the same API. Passing [`DEFAULT_TOLERANCE`](crate::path::DEFAULT_TOLERANCE)
        /// asks for the fit the engine has always done: one canvas px, i.e. a mouse
        /// at 1:1.
        ///
        /// It tunes the **fit** and nothing else — the selection tools, which share
        /// this gesture but fit no curve, ignore it, and so does the flattening that
        /// turns a fitted path into segments.
        tolerance: f32,
    },
    To {
        sample: InputSample,
    },
    End,
    Cancel,
}

/// Mutations of **document state**: each becomes an [`Action`](crate::document::Action),
/// enters the undo history, is replicated to peers, and is reproduced by replay.
#[derive(Clone, Debug)]
pub enum DocCommand {
    Undo,
    Redo,

    AddLayer {
        above: Option<LayerId>,
    },
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    MoveLayer {
        id: LayerId,
        above: Option<LayerId>,
    },

    /// Apply a selection op directly — the menu path (Select All / Deselect), and
    /// how a frontend with its own geometry can drive the selection without a
    /// gesture (DESIGN.md §6.8).
    Select(SelectionOp),
    /// Swap selected for unselected everywhere.
    InvertSelection,

    /// Switch the canvas surface (DESIGN.md §6.4).
    ///
    /// Document state, not view state: which canvas a piece was painted on is part
    /// of what the document *is* — it is saved, and reopening on a different weave
    /// would be a different painting. It will also gate deposition again if the
    /// tooth idea returns (§6.4); logging it now means that would be a rendering
    /// change rather than a history one.
    SetSurface(SurfaceId),
}

/// Mutations of **view state**: per-client, transient, never logged and never sent
/// to peers. Undo does not reach these, and two people sharing a drawing each have
/// their own.
#[derive(Clone, Debug)]
pub enum ViewCommand {
    SetTool(Tool),
    SetBrush(BrushParams),
    /// Pan the view by a screen-pixel drag delta.
    Pan {
        delta: Vec2,
    },
    /// Zoom by `factor`, keeping the canvas point under `anchor` (a screen-pixel
    /// position, e.g. the cursor) fixed on screen.
    Zoom {
        anchor: Vec2,
        factor: f32,
    },
    /// The viewport changed size (window/canvas resize).
    Resize(Extent2),

    /// How the next selection gesture combines with the current selection. Shapes
    /// the *next* op; the op itself is what gets logged (DESIGN.md §6.8).
    SetSelectionMode(SelectionMode),
    /// Edge softness (canvas px) for the next selection gesture.
    SetSelectionFeather(f32),

    /// Which layer the next stroke goes on. Per-client: collaborators paint on
    /// whichever layer each has selected.
    SetActiveLayer(LayerId),

    /// Tune the media/lighting pass (DESIGN.md §6.3). Changes how the canvas
    /// looks, not what it is.
    SetMediaParams(MediaParams),
    /// Switch the HDR lighting environment (DESIGN.md §6.3).
    SetEnvironment(EnvironmentId),
}

impl From<GestureCommand> for InputCommand {
    fn from(c: GestureCommand) -> Self {
        InputCommand::Gesture(c)
    }
}

impl From<DocCommand> for InputCommand {
    fn from(c: DocCommand) -> Self {
        InputCommand::Doc(c)
    }
}

impl From<ViewCommand> for InputCommand {
    fn from(c: ViewCommand) -> Self {
        InputCommand::View(c)
    }
}
