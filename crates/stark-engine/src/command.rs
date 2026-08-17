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
    BlendMode, BrushParams, FillOp, Filter, LayerId, MattePaint, MatteRegion, Place, SelectionOp,
    ShapeAction, Tool, TransformMap,
};
use crate::gpu::{EnvironmentId, MediaParams};
use stark_model::SurfaceId;
use stark_model::geom::{Extent2, Vec2};

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

    /// Whether every number here is finite — the gate a report has to pass before
    /// anything stateful is allowed to remember it.
    ///
    /// **A sample arrives from outside and one NaN in it is a panic**, which is why
    /// this is a property of the sample rather than a check at each consumer. A
    /// pointer position is the frontend's `screen_to_canvas` of a pointer event, so
    /// it inherits whatever the view transform is — and a non-finite one used to be
    /// reachable ([`ViewTransform`](stark_model::geom::ViewTransform), whose mutators now
    /// refuse to store one). Downstream, `PathFitter` parameterizes its samples by
    /// arc length, a NaN spreads from `arc` into every curve parameter, and the
    /// normal equations it builds are then unsolvable at *any* ridge — which
    /// `spline`'s solve reports by panicking, because for finite input it genuinely
    /// cannot happen.
    ///
    /// Every channel, not just the position. `time` seeds the fitter's epoch, so a
    /// NaN there makes every later report's relative time NaN; `pressure` and `tilt`
    /// ride the same least-squares solve the geometry does and reach it through the
    /// same matrix.
    pub fn is_finite(&self) -> bool {
        self.pos.is_finite()
            && self.pressure.is_finite()
            && self.tilt.is_finite()
            && self.time.is_finite()
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
        /// The stroke-smoothing string length in canvas px (§6.11): the mark is
        /// drawn by a tip towed this far behind the pointer. `0` is no tow at
        /// all — the raw samples reach the fitter bit-identically to before.
        ///
        /// The frontend's to state, like `tolerance` and for the same reason:
        /// the brush's smoothing amount is denominated in **screen** px,
        /// because wobble is a fact about the hand, and only the frontend
        /// holds the view that converts it. Fixed for the gesture. Ignored by
        /// the selection tools, which fit no curve.
        rope: f32,
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
    /// Merge a layer **down** onto the one beneath it (§14.11): the lower layer keeps
    /// its name, its place and its properties, and takes the paint of both; the upper
    /// one is gone.
    ///
    /// **Declined when it would change what the document looks like**, which is the
    /// whole of what a merge promises and the reason this is not always available. The
    /// engine answers the same question the panel asks before offering the control
    /// ([`LayerInfo::merge_down`](crate::LayerInfo::merge_down)), so a command sent
    /// anyway is a silent no-op rather than a different picture.
    MergeLayerDown(LayerId),
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

    /// Add a **matte** layer — a region filled with a [`MattePaint`]
    /// (§15.2). A frame is one of these on top of the stack; a ground
    /// ([`MatteRegion::Everything`]) is one at the bottom, hence the full
    /// [`Place`] anchor (§15.5). The engine mints the id, as it does
    /// for `AddLayer`; unlike `AddLayer` it does *not* become the active layer,
    /// because a matte cannot be painted on.
    AddMatte {
        carrier: Option<LayerId>,
        at: Place,
        region: MatteRegion,
        paint: MattePaint,
    },
    /// Add a **filter** layer — a function of everything composited beneath it in
    /// the stack it lands in (§21.2). The engine mints the id, as it does for
    /// `AddLayer`; and as for `AddMatte` it does *not* become the active layer,
    /// because a filter cannot be painted on.
    ///
    /// It arrives at its neutral setting, so adding one changes nothing until it is
    /// dialled with [`SetFilter`](Self::SetFilter). Where it lands is its scope —
    /// the root stack filters the whole painting, a carried stack filters that group
    /// — so there is no third parameter saying how far it reaches.
    AddFilter {
        carrier: Option<LayerId>,
        above: Option<LayerId>,
        filter: Filter,
    },
    /// Retune a filter layer — one action per adjustment, committed when the slider
    /// settles (§21.6). The whole filter travels, so a new knob or a new kind of
    /// filter needs no command of its own.
    SetFilter(LayerId, Filter),
    /// Move a matte's rect — one action per frame drag, committed on release.
    SetMatteRect(LayerId, Vec2, Vec2),
    /// Repaint a matte — a flat color or a gradient ramp (§15.4, §22.4). One
    /// action per pick or per composed axis, committed when the gesture settles.
    SetMattePaint(LayerId, MattePaint),
    /// Set the canvas substrate color — the ground under everything, straight
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
    /// see [`ViewTransform::pinch`](stark_model::geom::ViewTransform::pinch).
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
    /// [`ViewTransform::rotation_for_up`](stark_model::geom::ViewTransform::rotation_for_up)
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
    /// [`ViewTransform::mirror_screen_h`](stark_model::geom::ViewTransform::mirror_screen_h)).
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
    /// Put the whole piece on screen: the canvas-space rect an export of `frame`
    /// would write (§15.6), centred and fitted to the viewport with the easel
    /// straightened ([`ViewTransform::show_rect`](stark_model::geom::ViewTransform::show_rect)).
    ///
    /// `frame` names a matte layer whose rect is the piece, exactly as
    /// [`Engine::export_plan`](crate::Engine::export_plan) takes it, and the rect is
    /// worked out by the same rule — so what you are looking at and what a file would
    /// hold cannot come to disagree about where the piece ends. With no frame it fits
    /// the painted bounds; with neither there is nothing to show and the view stays
    /// where it is (an export, which must write *something*, falls back to the
    /// viewport instead).
    ///
    /// Absolute, like [`CenterOn`](Self::CenterOn) and for its reason: the caller
    /// knows exactly what it wants shown and nothing about where the view was.
    /// **Opening a document is that caller** (§8): a painting arrives framed, rather
    /// than at the pan and zoom the last one happened to be left at — which on an
    /// unbounded canvas can be nowhere near the paint that just loaded.
    ShowPiece(Option<LayerId>),
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
    /// How strongly the next shape gesture's coverage lands, `0..=1` — the mask's
    /// own strength when the gesture selects, the fill's when it fills (§6.8).
    SetShapeOpacity(f32),

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

    /// Show a matte wearing `paint` **without logging it** — the in-flight half
    /// of a frame-color pick or a gradient-axis drag (§15.7, §22.4). `None`
    /// drops the preview.
    ///
    /// The rect's argument, made by the other halves of the same control: a
    /// color picker reports a value per pointer move while it is open, and an
    /// axis drag reports one per sample, so the frontend previews those and
    /// commits one [`DocCommand::SetMattePaint`] when the gesture settles. What
    /// is being chosen here is how a matte reads against the piece, which is a
    /// judgement made *by looking* — so the picking has to show on the canvas,
    /// and only the answer belongs in the log.
    PreviewMattePaint(Option<(LayerId, MattePaint)>),

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

    /// Show the document as a [`DocCommand::Fill`] would leave it, **without
    /// logging it** — the in-flight half of the gradient-fill gesture, where the
    /// drag composes the ramp's axis and "Done" commits one `Fill` (§22.4).
    /// `None` drops the preview.
    ///
    /// The same bargain as [`PreviewTransform`](Self::PreviewTransform), through
    /// the same renderer as its commit (`FillRenderer::apply`), so what is shown
    /// is what committing would produce. Each preview lays the parcel on the
    /// *committed* tiles, so redrawing the axis a hundred times previews one
    /// fill, never a hundred stacked glazes.
    PreviewFill(Option<(LayerId, FillOp)>),

    /// Show a substrate color **without logging it** — the in-flight half of a
    /// canvas-color drag (§15.5). `None` drops the preview.
    ///
    /// The same bargain as [`PreviewMatteRect`](Self::PreviewMatteRect), for the
    /// same reason: a color picker reports a value per pointer *move*, so
    /// committing each one would spend an undo step — and, in a shared session, a
    /// replicated log entry — on every sample of a single drag. The frontend knows
    /// where the drag ends and commits one [`DocCommand::SetBackground`] there.
    PreviewBackground(Option<[f32; 3]>),

    /// Show a layer at `opacity` **without logging it** — the in-flight half of an
    /// opacity-slider drag (§14.6). `None` drops the preview.
    ///
    /// The same bargain as [`PreviewBackground`](Self::PreviewBackground), for the
    /// same reason: a slider reports a value per pointer *move*, so committing each
    /// one spends an undo step — and, in a shared session, a replicated action — on
    /// every sample of what the hand did once. The frontend knows where the drag
    /// ends and commits one [`DocCommand::SetLayerOpacity`] there.
    ///
    /// A view command rather than a `GestureCommand` for the reason
    /// [`PreviewMatteRect`](Self::PreviewMatteRect) is one: a slider is not
    /// sample-driven, so there is no [`InputSample`] to feed `Start`/`To`/`End`
    /// with, and it reports the value it wants rather than the motion that reached
    /// it. What it borrows from a gesture is the shape that matters — build in view
    /// state, commit once on release.
    ///
    /// Opacity is a presentation property folded in at composite time (§14.7), so a
    /// preview costs a refold and moves no pixels: unlike
    /// [`PreviewTransform`](Self::PreviewTransform) there is nothing to resample,
    /// which is what makes it affordable at pointer rate.
    PreviewLayerOpacity(Option<(LayerId, f32)>),

    /// Show a filter layer set to `filter` **without logging it** — the in-flight
    /// half of a filter-slider drag (§21.6). `None` drops the preview.
    ///
    /// The same bargain as [`PreviewLayerOpacity`](Self::PreviewLayerOpacity), and
    /// the one this feature could least do without: a color adjustment is judged
    /// *by looking* — how much saturation is too much is a question about the
    /// painting, not about the number — so every value the pointer crosses has to
    /// reach the canvas, and only the answer belongs in the log.
    ///
    /// A filter is presentation, folded in at composite time (§21.3), so a preview
    /// costs one fullscreen pass and moves no pixels of stored paint. That is what
    /// makes it affordable at pointer rate where
    /// [`PreviewTransform`](Self::PreviewTransform) has to resample.
    PreviewFilter(Option<(LayerId, Filter)>),

    /// Show a layer blending as `blend` **without logging it** — the in-flight half
    /// of a drag on one of a mode's own parameters (§18.0.4). `None` drops the
    /// preview.
    ///
    /// The same bargain as [`PreviewLayerOpacity`](Self::PreviewLayerOpacity), for
    /// [`PreviewFilter`](Self::PreviewFilter)'s reason: a mode's parameter is judged
    /// *by looking* — how hot Radiance should run is a question about the painting —
    /// so every value the pointer crosses has to reach the canvas and only the answer
    /// belongs in the log.
    ///
    /// It carries the whole mode rather than a parameter, which is what keeps this
    /// one command rather than one per knob a mode grows: what a frontend has in hand
    /// while dragging is the layer's mode with one field moved, and that is exactly
    /// what [`DocCommand::SetLayerBlend`] takes when the drag settles.
    ///
    /// A blend is folded in at composite time, so a preview costs the isolation and
    /// the merge pass the layer was already paying for and moves no pixels of stored
    /// paint — affordable at pointer rate for the reason the two above are.
    PreviewLayerBlend(Option<(LayerId, BlendMode)>),

    /// Tune the media/lighting pass (§6.3). Changes how the canvas
    /// looks, not what it is.
    SetMediaParams(MediaParams),
    /// Switch the HDR lighting environment (§6.3).
    SetEnvironment(EnvironmentId),

    /// How much resident tile memory history retention may hold before the engine
    /// starts giving up undo depth, in bytes (§5).
    ///
    /// **View state, and it belongs here rather than in the document**: how deep an
    /// undo stack this machine can afford is a fact about the machine, not about the
    /// painting. Two people sharing a drawing have different amounts of memory and
    /// should keep different amounts of history, and neither answer belongs in the
    /// file. It is also why this is a command at all rather than a setter — §4's
    /// rule is that anything mutating state and returning nothing is one.
    ///
    /// The engine's own default is a bound on a cost that has not been measured (see
    /// `HISTORY_TILE_BUDGET`), and it is one number for a range of devices that runs
    /// from a phone to a workstation. A frontend that knows which it is on should say
    /// so.
    ///
    /// Setting it does not itself trim; the next commit does, which is the only
    /// moment the stack grows. `0` means "keep the minimum" — retention is still
    /// floored at `MIN_UNDO_DEPTH` steps, because trimming below that frees nothing
    /// when the memory is held by the current document rather than by history.
    SetHistoryBudget(u64),
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
