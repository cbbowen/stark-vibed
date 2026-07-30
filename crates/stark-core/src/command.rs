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
//!   *and never sent*. Two people sharing a drawing pan independently.
//! - [`PeerCommand`] mutates **presence**: per-client and never logged, like view
//!   state, but *published* — every collaborator reads it and only its owner writes
//!   it (PEER_DESIGN.md §7). The private/published line is in the type for the same
//!   reason the logged/unlogged one is: it decides who sees the change.
//! - [`GestureCommand`] is the press-drag-release lifecycle, which is neither: it
//!   *builds* in view state (`Session::in_flight`) and commits a document action
//!   at the end — or nothing at all, if cancelled. In a shared session the building
//!   is published too, so peers watch the stroke as it is drawn.
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

use crate::document::{
    BlendMode, BrushParams, FillOp, LayerId, MatteRegion, SelectionOp, ShapeAction, Tool,
};
use crate::geom::{Affine2, Extent2, Vec2};
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
    Peer(PeerCommand),
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
    /// Name a layer, or with `None` clear the name so it goes back to being
    /// described by its place in the stack. The text is trimmed and length-capped
    /// on the way in, and one that comes out blank clears the name rather than
    /// setting an empty one — so "a name is either absent or something you can
    /// read" holds however the frontend collects it.
    SetLayerName(LayerId, Option<String>),
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

    /// Fill a region of `layer` with paint (MISSING_FEATURES §0.4) — the direct
    /// path, next to `Select`: the selection bar's Fill button and any frontend
    /// that has its own geometry. The gesture path goes through
    /// [`ShapeAction::Fill`](crate::document::ShapeAction) instead, and commits the
    /// identical action.
    ///
    /// A fill whose region is [`SelectionShape::All`](crate::document::SelectionShape)
    /// means "the selection", and is refused when there is none — the canvas is
    /// unbounded, and inventing a boundary would be a different fill on every client.
    Fill {
        layer: LayerId,
        op: FillOp,
    },

    /// Add a **matte** layer — a region filled with a flat colour
    /// (FRAME_DESIGN.md §2). A frame is one of these on top of the stack. The
    /// engine mints the id, as it does for `AddLayer`; unlike `AddLayer` it does
    /// *not* become the active layer, because a matte cannot be painted on.
    AddMatte {
        above: Option<LayerId>,
        region: MatteRegion,
        /// Straight sRGB.
        color: [f32; 3],
    },
    /// Move a matte's rect — one action per frame drag, committed on release.
    SetMatteRect(LayerId, Vec2, Vec2),
    /// Recolour a matte (straight sRGB).
    SetMatteColor(LayerId, [f32; 3]),
    /// Set the canvas substrate colour — the ground under everything, straight
    /// sRGB (FRAME_DESIGN.md §5). A document property, not a view setting: it is
    /// what the piece was painted on, and it is saved.
    SetBackground([f32; 3]),

    /// Affine-transform this client's selected paint on `layer`, carrying the
    /// selection along with it (TRANSFORM_DESIGN.md). A universal selection moves
    /// the whole layer. One action per gesture — the interactive drag builds in
    /// view state and commits once on release, like the frame drag.
    Transform {
        layer: LayerId,
        affine: Affine2,
    },

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

    /// What the next shape gesture does with the region it encloses: combine it
    /// into the selection one of four ways, or fill it (DESIGN.md §6.8,
    /// MISSING_FEATURES §0.4). Shapes the *next* op; the op itself is what gets
    /// logged.
    SetShapeAction(ShapeAction),
    /// Edge softness (canvas px) for the next shape gesture — the same ramp whether
    /// it selects or fills.
    SetSelectionFeather(f32),

    /// Whether collaborators' selection outlines are drawn over the canvas
    /// (PEER_DESIGN.md §3). View state, so each client decides for itself — this is
    /// a preference about what you look at, not a fact about the drawing.
    SetShowPeerSelections(bool),

    /// Show a matte at `min..max` **without logging it** — the in-flight half of a
    /// frame-handle drag (FRAME_DESIGN.md §7). `None` drops the preview.
    ///
    /// A view command rather than a `GestureCommand` because a frame drag is
    /// handle-relative, not sample-driven: there is no `InputSample` to feed
    /// `Start`/`To`/`End`, and which handle is held is the frontend's business.
    /// What it shares with a gesture is the shape that matters — it builds in view
    /// state and the frontend commits one [`DocCommand::SetMatteRect`] on release,
    /// so a drag costs one undo step rather than one per pointer move.
    PreviewMatteRect(Option<(LayerId, Vec2, Vec2)>),

    /// Show the document as a [`DocCommand::Transform`] would leave it, **without
    /// logging it** — the in-flight half of the transform gesture
    /// (TRANSFORM_DESIGN.md §6). `None` drops the preview.
    ///
    /// The same bargain as [`PreviewMatteRect`](Self::PreviewMatteRect): the drag
    /// builds in view state and the frontend commits one `Transform` on "Done".
    /// The preview runs the *same renderer* as the commit over the committed
    /// tiles, so what is shown is what committing would produce — and because
    /// each preview resamples from the committed tiles under the accumulated
    /// affine, a long drag never compounds resampling loss.
    PreviewTransform(Option<(LayerId, Affine2)>),

    /// Show a substrate colour **without logging it** — the in-flight half of a
    /// canvas-colour drag (FRAME_DESIGN.md §5). `None` drops the preview.
    ///
    /// The same bargain as [`PreviewMatteRect`](Self::PreviewMatteRect), for the
    /// same reason: a colour picker reports a value per pointer *move*, so
    /// committing each one would spend an undo step — and, in a shared session, a
    /// replicated log entry — on every sample of a single drag. The frontend knows
    /// where the drag ends and commits one [`DocCommand::SetBackground`] there.
    PreviewBackground(Option<[f32; 3]>),

    /// Tune the media/lighting pass (DESIGN.md §6.3). Changes how the canvas
    /// looks, not what it is.
    SetMediaParams(MediaParams),
    /// Switch the HDR lighting environment (DESIGN.md §6.3).
    SetEnvironment(EnvironmentId),
}

/// Mutations of **presence**: per-client and never logged — undo does not reach
/// these and they are not in the save file — but *published*, so every collaborator
/// sees them (PEER_DESIGN.md §4, §7).
///
/// What separates these from [`ViewCommand`] is only who reads the result. What
/// separates them from [`DocCommand`] is that replay does not need them to reproduce
/// a pixel: the selected layer is already closed over by
/// [`StrokeRecord::layer`](crate::document::StrokeRecord), a cursor paints nothing,
/// and a name is not part of the artwork.
#[derive(Clone, Debug)]
pub enum PeerCommand {
    /// The selected layer — where the next stroke goes, if that layer can take
    /// one. Per-client: collaborators paint on whichever layer each has selected,
    /// and each can see where the others are working.
    ///
    /// A **matte** may be selected like any other layer (FRAME_DESIGN.md §7). It
    /// has no tile map, so a stroke aimed at one draws nothing — refused
    /// identically by `apply` and by the preview path, so the frontend needs no
    /// rule of its own. Selection is therefore one concept rather than "the paint
    /// target" plus a separate frame-focus the engine cannot see.
    SetActiveLayer(LayerId),

    /// Where this client's pointer is, in canvas space; `None` when it leaves the
    /// canvas. Cheap at pointer rate: it writes a field, and the publish latch
    /// coalesces to one frame per tick (PEER_DESIGN.md §5.1).
    SetCursor(Option<Vec2>),

    /// This client's display name. Empty falls back to a short id-derived one, so
    /// two unnamed peers are still distinguishable.
    SetName(String),
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

impl From<PeerCommand> for InputCommand {
    fn from(c: PeerCommand) -> Self {
        InputCommand::Peer(c)
    }
}
