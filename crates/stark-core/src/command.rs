//! `InputCommand`: raw, high-frequency user intent (§4).
//!
//! Commands are deliberately distinct from [`Action`](crate::document::Action)s.
//! Many commands are ephemeral (pointer moves mid-stroke, pan/zoom, tool changes)
//! and never enter history; only committed mutations become actions. The `Session`
//! (§3) interprets commands and decides what, if anything, to commit.
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
//!   it (§17.7). The private/published line is in the type for the same
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
//! (§7). Anything that must answer — importing a brush and getting its
//! id, saving bytes, merging a remote action and learning whether it applied — is
//! a **request**, and requests stay direct methods on [`Engine`](crate::Engine)
//! until there is an actor to give them a reply channel. See §4.

use serde::{Deserialize, Serialize};

use crate::document::{
    BlendMode, BrushParams, FillOp, LayerId, MatteRegion, Place, SelectionOp, ShapeAction, Tool,
    TransformMap,
};
use crate::geom::{Extent2, Vec2};
use crate::gpu::{EnvironmentId, MediaParams, SurfaceId};

/// One pen/mouse sample in canvas space.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputSample {
    pub pos: Vec2,
    pub pressure: f32,
    pub tilt: Vec2,
    /// Timestamp in seconds, for velocity and timelapse (§8).
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

/// Every stateful interaction the backend accepts (§4).
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
/// (§6.8): from the frontend's side both are one gesture, and the `tool`
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
    /// The pointer has been **held still** mid-gesture: snap the stroke in flight to
    /// the line or ellipse it resembles, and hand the rest of the drag to that shape
    /// (§6.9).
    ///
    /// The dwell itself is the frontend's to detect and is not described here, for the
    /// same reason [`ViewCommand::SetRotation`] is absolute: how long a pause has to be
    /// and how still the hand has to hold is *gesture feel*, a property of the device
    /// and the hand, and the engine has no clock to measure it with anyway (§7). What
    /// the engine owns is what a hold **means**, which is this command.
    ///
    /// Idempotent, and a no-op for a gesture that has already snapped, for a selection
    /// drag, or for a stroke that resembles nothing — so the frontend may send it
    /// whenever it thinks the pointer has stopped without first asking what state the
    /// gesture is in.
    Hold,
    End,
    Cancel,
}

/// Mutations of **document state**: each becomes an [`Action`](crate::document::Action),
/// enters the undo history, is replicated to peers, and is reproduced by replay.
#[derive(Clone, Debug)]
pub enum DocCommand {
    Undo,
    Redo,
    /// Move the history playhead to an absolute position, in actions from the
    /// start of the log — the scrubber's command (§18.2.4).
    ///
    /// Navigation, exactly like [`Undo`](Self::Undo) and [`Redo`](Self::Redo),
    /// and it lives beside them for that reason: it moves the same applied /
    /// withheld split those two move one step at a time, so nothing is logged and
    /// nothing is sent. Absolute rather than a delta because a scrubber knows
    /// exactly where it wants the playhead and nothing about where it was — the
    /// same argument [`ViewCommand::CenterOn`] makes against expressing a jump as
    /// a drag.
    ///
    /// Clamped to the range the timeline reports, and a no-op on a timeline that
    /// has no single playhead to move (a shared session — see
    /// [`Timeline::scrub_range`](crate::document::Timeline::scrub_range)).
    Seek(usize),

    /// Add an empty paint layer to the stack carried by `carrier` (the
    /// document's own when `None`), directly above `above` (§14.8).
    AddLayer {
        carrier: Option<LayerId>,
        above: Option<LayerId>,
    },
    /// Copy a layer **and everything it carries** into its own stack, directly
    /// above it (§14.8) — the subtree travels as one, for the reason
    /// [`RemoveLayer`](Self::RemoveLayer)'s does.
    ///
    /// The engine mints the copies' ids, as it does for
    /// [`AddLayer`](Self::AddLayer) — one per layer of the subtree — and the copy
    /// becomes the active layer when it can be painted on.
    DuplicateLayer(LayerId),
    /// Remove a layer **and everything it carries** — the subtree is the group
    /// (§14.2). To keep what it carried, release those layers with
    /// [`MoveLayer`](Self::MoveLayer) first.
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    /// Clip a layer to the paint beneath it in its own stack, or stop
    /// (§14.4). On the base of a group this clips the whole group
    /// to what lies under the group.
    SetLayerClip(LayerId, bool),
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    /// Name a layer, or with `None` clear the name so it goes back to being
    /// described by its place in the stack. The text is trimmed and length-capped
    /// on the way in, and one that comes out blank clears the name rather than
    /// setting an empty one — so "a name is either absent or something you can
    /// read" holds however the frontend collects it.
    SetLayerName(LayerId, Option<String>),
    /// Move a layer — with everything it carries — into the stack carried by
    /// `carrier` (the document's own when `None`), at the place `at` names in it.
    ///
    /// One command for all three gestures (§14.8): **reorder**
    /// leaves `carrier` as it was, **carry** sets it to the layer being dropped
    /// onto, **release** clears it. Asking a layer to carry its own ancestor is
    /// declined.
    ///
    /// That is also why the layers panel can spell a drag-and-drop reorder with
    /// this one command: a drop lands in *some* stack at *some* place in it, which
    /// is exactly the pair below (§14.6).
    MoveLayer {
        id: LayerId,
        carrier: Option<LayerId>,
        at: Place,
    },

    /// Apply a selection op directly — the menu path (Select All / Deselect), and
    /// how a frontend with its own geometry can drive the selection without a
    /// gesture (§6.8).
    Select(SelectionOp),
    /// Swap selected for unselected everywhere.
    InvertSelection,

    /// Fill a region of `layer` with paint (§18.0.4) — the direct
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
    /// (§15.2). A frame is one of these on top of the stack. The
    /// engine mints the id, as it does for `AddLayer`; unlike `AddLayer` it does
    /// *not* become the active layer, because a matte cannot be painted on.
    AddMatte {
        carrier: Option<LayerId>,
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
    /// sRGB (§15.5). A document property, not a view setting: it is
    /// what the piece was painted on, and it is saved.
    SetBackground([f32; 3]),

    /// Transform this client's selected paint on `layer` — affine, perspective
    /// or warp (§16, §16.8, §16.9) — carrying the selection along with it. A
    /// universal selection moves the whole layer. One action per gesture — the
    /// interactive drag builds in view state and commits once on "Done", like
    /// the frame drag. The engine routes each map family to its own action
    /// kind, so the log stays wire-stable.
    Transform {
        layer: LayerId,
        map: TransformMap,
    },

    /// Switch the canvas surface (§6.4).
    ///
    /// Document state, not view state: which canvas a piece was painted on is part
    /// of what the document *is* — it is saved, and reopening on a different weave
    /// would be a different painting. It also **gates deposition** (§6.4): the tooth
    /// reads the surface in force at each action's point in the log, so a switch
    /// part-way through changes the strokes after it and not the ones before. That
    /// this was already a logged action is what made the tooth a rendering change
    /// rather than a history one, exactly as this note anticipated.
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
    /// Move, scale and turn the view together — the two-finger gesture
    /// (§18.1.7). The canvas point under `anchor` (screen px) ends up under `to`,
    /// scaled by `scale` and turned by `turn` radians clockwise about it.
    ///
    /// One command rather than a [`Pan`](Self::Pan), a [`Zoom`](Self::Zoom) and a
    /// turn, because a pinch is one motion of one pair of fingers and the three are
    /// not independent: each anchors against the view it is applied to, so sent in
    /// sequence the last two would anchor against a view the hand never saw and the
    /// canvas would slide out from under it. What the fingers hold, they hold —
    /// see [`ViewTransform::pinch`](crate::geom::ViewTransform::pinch).
    ///
    /// **Incremental**, unlike [`SetRotation`](Self::SetRotation), and for the
    /// opposite reason: a pinch does not know what angle it wants, only how far the
    /// hand has turned since the last report. What the frontend keeps on its side is
    /// gesture feel — the twist a gesture must earn before it turns the canvas at
    /// all, and the pull onto a quarter turn — which it spends by choosing `turn`.
    Pinch {
        anchor: Vec2,
        to: Vec2,
        scale: f32,
        turn: f32,
    },
    /// Turn the canvas to this angle (radians, clockwise) — the navigator's
    /// right-drag (§18.1.2).
    ///
    /// Absolute, like [`CenterOn`](Self::CenterOn) and for the same reason: the
    /// gesture knows exactly where it wants the canvas, and an incremental command
    /// would have the frontend keep a copy of the angle to add to. What the *drag*
    /// gives is a direction, which
    /// [`ViewTransform::rotation_for_up`](crate::geom::ViewTransform::rotation_for_up)
    /// turns into an angle; the easing and the snap-to-square between the two are the
    /// frontend's, because they are properties of dragging with a hand.
    SetRotation(f32),
    /// Mirror what is on screen, left↔right — the oldest way of catching a drawing
    /// error, since the eye stops recognising what it expected and starts seeing what
    /// is there.
    ///
    /// A toggle rather than a setting, and **screen-relative**: it swaps the left of
    /// the screen with the right at any angle, so the check means the same thing
    /// however the easel is turned (see
    /// [`ViewTransform::mirror_screen_h`](crate::geom::ViewTransform::mirror_screen_h)).
    /// View state, so it changes nothing about the painting and nobody else sees it.
    MirrorH,

    /// Show this canvas-space point at the centre of the viewport, leaving the zoom
    /// alone — a jump rather than a drag.
    ///
    /// Absolute where [`Pan`](Self::Pan) is incremental, because the callers are
    /// different in kind: a drag knows only how far the pointer moved, while a
    /// navigator click knows exactly where it wants to be and nothing about where
    /// the view was. Expressing the second as the first means reading the zoom back
    /// out, dividing by it, and having the engine multiply it in again — a copy of
    /// view state in the frontend and a round trip through it, to say something the
    /// engine can do in one assignment.
    CenterOn(Vec2),
    /// The viewport changed size (window/canvas resize).
    Resize(Extent2),

    /// What the next shape gesture does with the region it encloses: combine it
    /// into the selection one of four ways, or fill it (§6.8,
    /// §18.0.4). Shapes the *next* op; the op itself is what gets
    /// logged.
    SetShapeAction(ShapeAction),
    /// Edge softness (canvas px) for the next shape gesture — the same ramp whether
    /// it selects or fills.
    SetSelectionFeather(f32),

    /// Whether collaborators' selection outlines are drawn over the canvas
    /// (§17.3). View state, so each client decides for itself — this is
    /// a preference about what you look at, not a fact about the drawing.
    SetShowPeerSelections(bool),

    /// Replace the drawing-guide list (§20.5) — the whole list at once, the
    /// same read-modify-commit shape as [`SetMediaParams`](Self::SetMediaParams):
    /// the frontend reads the current list off the projection, adjusts one
    /// guide or one field, and sends it back, so the engine never needs one
    /// command per slider, per drag sample, or per row.
    ///
    /// View state *for now*: a guide is an aid for the hand holding the pen,
    /// per-client like the pan and the zoom. If guides later become part of
    /// what a document carries, that is a new `DocCommand` and an action —
    /// this command would remain as the in-flight preview half, the bargain
    /// [`PreviewMatteRect`](Self::PreviewMatteRect) strikes.
    SetGuides(Vec<crate::guides::PerspectiveGuide>),

    /// Show a matte at `min..max` **without logging it** — the in-flight half of a
    /// frame-handle drag (§15.7). `None` drops the preview.
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
    /// (§16.6). `None` drops the preview.
    ///
    /// The same bargain as [`PreviewMatteRect`](Self::PreviewMatteRect): the drag
    /// builds in view state and the frontend commits one `Transform` on "Done".
    /// The preview runs the *same renderer* as the commit over the committed
    /// tiles, so what is shown is what committing would produce — and because
    /// each preview resamples from the committed tiles under the accumulated
    /// map, a long drag never compounds resampling loss.
    PreviewTransform(Option<(LayerId, TransformMap)>),

    /// Show a substrate colour **without logging it** — the in-flight half of a
    /// canvas-colour drag (§15.5). `None` drops the preview.
    ///
    /// The same bargain as [`PreviewMatteRect`](Self::PreviewMatteRect), for the
    /// same reason: a colour picker reports a value per pointer *move*, so
    /// committing each one would spend an undo step — and, in a shared session, a
    /// replicated log entry — on every sample of a single drag. The frontend knows
    /// where the drag ends and commits one [`DocCommand::SetBackground`] there.
    PreviewBackground(Option<[f32; 3]>),

    /// Tune the media/lighting pass (§6.3). Changes how the canvas
    /// looks, not what it is.
    SetMediaParams(MediaParams),
    /// Switch the HDR lighting environment (§6.3).
    SetEnvironment(EnvironmentId),
}

/// Mutations of **presence**: per-client and never logged — undo does not reach
/// these and they are not in the save file — but *published*, so every collaborator
/// sees them (§17.4, §7).
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
    /// A **matte** may be selected like any other layer (§15.7). It
    /// has no tile map, so a stroke aimed at one draws nothing — refused
    /// identically by `apply` and by the preview path, so the frontend needs no
    /// rule of its own. Selection is therefore one concept rather than "the paint
    /// target" plus a separate frame-focus the engine cannot see.
    SetActiveLayer(LayerId),

    /// Where this client's pointer is, in canvas space; `None` when it leaves the
    /// canvas. Cheap at pointer rate: it writes a field, and the publish latch
    /// coalesces to one frame per tick (§17.5).
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
