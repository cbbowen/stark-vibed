//! `InputCommand`: raw, high-frequency user intent (§4).
//!
//! Commands are deliberately distinct from [`Action`](stark_model::document::Action)s.
//! Many are ephemeral (pointer moves mid-stroke, pan/zoom, tool changes) and never
//! enter history; only committed mutations become actions. The `Session` (§3)
//! interprets commands and decides what, if anything, to commit.
//!
//! # The four kinds
//!
//! Which class of state a command touches decides whether it is logged, whether
//! peers see it and whether undo reaches it, so it lives in the type rather than in
//! a comment:
//!
//! - [`DocCommand`] mutates **document state**: historized, replicated to peers,
//!   and reproduced by replay. Every one of these becomes an `Action`.
//! - [`ViewCommand`] mutates **view state**: per-client, transient, never logged
//!   *and never sent*. Two people sharing a drawing pan independently.
//! - [`PeerCommand`] mutates **presence**: per-client and never logged, like view
//!   state, but *published* — every collaborator reads it and only its owner writes
//!   it (§17.7).
//! - [`GestureCommand`] is the press-drag-release lifecycle, which is neither: it
//!   *builds* in view state (`Session::in_flight`) and commits a document action
//!   at the end — or nothing at all, if cancelled. In a shared session the building
//!   is published too, so peers watch the stroke as it is drawn.
//!
//! # What is deliberately *not* a command
//!
//! Commands are one-way: they carry intent in and nothing back, which is what lets
//! them become messages over a channel when the engine moves off the UI thread (§7).
//! Anything that must answer — importing a brush and getting its id, saving bytes,
//! merging a remote action and learning whether it applied — is a **request**, and
//! requests stay direct methods on [`Engine`](crate::Engine). See §4.
//!
//! # Where the line falls
//!
//! A command states what the **engine** owns and asks for what only a frontend can
//! know. Three things are always the frontend's, and the variants below name their own
//! side of the line rather than re-arguing it:
//!
//! - **the view** — canvas px are 64× coarser zoomed out than in, so anything felt in
//!   screen px (an input tolerance, a smoothing length, where an image lands) has to
//!   be converted before it arrives;
//! - **the platform** — which digitizer is reporting, which image formats decode;
//! - **gesture feel** — how long a dwell lasts, how much twist a pinch must earn,
//!   what a drag eases onto. The engine has no clock and no window (§7), so it owns
//!   what a gesture *means* and not when one happened.

/// The tool a gesture drives. Tools become an open registry later (§10).
///
/// **Session state, not document state**, which is why it is here and not in
/// `stark-model` (§2): which tool was in hand is not part of what a document *is* —
/// the stroke or the op it produced is, and that is what the log carries. One enum,
/// and so one pointer-gesture path, because press-drag-release is the same
/// interaction either way; only [`Brush`](Self::Brush) reaches a `StrokeRecord`, the
/// selection tools produce a `SelectionOp` (§6.8).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Brush,
    /// Rectangular marquee.
    SelectRect,
    /// Elliptical marquee.
    SelectEllipse,
    /// Freehand lasso.
    SelectLasso,
}

impl Tool {
    /// Whether this tool edits the selection rather than the paint.
    pub fn is_selection(self) -> bool {
        matches!(
            self,
            Tool::SelectRect | Tool::SelectEllipse | Tool::SelectLasso
        )
    }
}

use crate::gpu::composite::{MediaParams, Output};
use crate::gpu::environment::EnvironmentId;
use crate::view::Extent2;
use stark_model::AssetId;
use stark_model::Srgb;
use stark_model::document::{
    BlendMode, BrushParams, FillOp, Filter, GuideId, LayerId, MatteRegion, Parcel,
    PerspectiveGuide, Place, SelectionOp, ShapeAction, TransformMap,
};
use stark_model::geom::{IVec2, Vec2};
use stark_model::{SubstrateId, SubstrateScale};

/// One pen/mouse sample in canvas space — the geometry's own input type, so it is
/// declared where the geometry is (`stark_stroke`) and named here, where it arrives.
pub use stark_stroke::InputSample;

/// One hover report for [`ViewCommand::PreviewHover`] (§18.1.10).
///
/// `tolerance` is the input tolerance, exactly [`GestureCommand::Start`]'s field,
/// restated per report because the zoom it derives from can change mid-hover.
/// `reach` is how far ahead of the cursor the predicted mark extends — canvas px
/// **by nature rather than by conversion**: paint is denominated on the canvas, and
/// a screen-fixed length would grow in canvas terms as the view zoomed out,
/// promising more painting the less closely you looked. The window the heading is
/// estimated over rides neither field — it is a fact about the tolerance alone, so
/// the engine derives it (`session`'s `WINDOW_ARC_TOLERANCES`).
#[derive(Copy, Clone, Debug)]
pub struct HoverReport {
    pub sample: InputSample,
    /// How finely this hover's input resolves position, canvas px.
    pub tolerance: f32,
    /// How far ahead of the cursor the predicted mark extends, canvas px.
    pub reach: f32,
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

/// The press-drag-release lifecycle (§6.8); [`Tool`] decides what the session builds
/// from it.
///
/// In flight this is view state. [`End`](Self::End) is the only edge that produces
/// document state, and [`Cancel`](Self::Cancel) produces none.
#[derive(Clone, Debug)]
pub enum GestureCommand {
    Start {
        tool: Tool,
        sample: InputSample,
        /// How finely this gesture's input actually resolves position, in canvas px
        /// — see [`PathFitter::with_tolerance`](crate::path::PathFitter::with_tolerance).
        /// The frontend's, per the view and the platform both: a pen digitizer, a
        /// touchscreen and a mouse each report at a different tolerance through the
        /// same API. [`DEFAULT_TOLERANCE`](crate::path::DEFAULT_TOLERANCE) is one
        /// canvas px, i.e. a mouse at 1:1.
        ///
        /// It tunes the **fit** and nothing else — the selection tools, which fit no
        /// curve, ignore it, and so does the flattening into segments.
        tolerance: f32,
        /// The stroke-smoothing string length in canvas px (§6.11): the mark is
        /// drawn by a tip towed this far behind the pointer. `0` is no tow at all.
        ///
        /// The brush's smoothing amount is denominated in **screen** px, because
        /// wobble is a fact about the hand — so this is the frontend's to convert,
        /// like `tolerance`. Fixed for the gesture, and ignored by the selection tools.
        rope: f32,
    },
    To {
        sample: InputSample,
    },
    /// The pointer has been **held still** mid-gesture: snap the stroke in flight to
    /// the line or ellipse it resembles, and hand the rest of the drag to that shape
    /// (§6.9).
    ///
    /// Detecting the dwell is the frontend's; what a hold **means** is the engine's.
    ///
    /// Idempotent, and a no-op for a gesture that has already snapped, for a selection
    /// drag, or for a stroke that resembles nothing — so the frontend may send it
    /// whenever it thinks the pointer has stopped.
    Hold,
    End,
    Cancel,
}

/// Mutations of **document state**: each becomes an [`Action`](stark_model::document::Action),
/// enters the undo history, is replicated to peers, and is reproduced by replay.
///
/// **Ids are minted by the engine, not named by the caller** — a layer's id *is* the
/// id of the action that minted it (§17.9), so it cannot exist before the commit. A
/// minted layer becomes the active one exactly when it is paint: a matte or a filter
/// has no tile map, so arming one would swallow the next stroke (§15.7, §21.4).
/// [`AddGuide`](Self::AddGuide) is the exception, and says why.
#[derive(Clone, Debug)]
pub enum DocCommand {
    Undo,
    Redo,
    /// Move the history playhead to an absolute position, in actions from the
    /// start of the log — the scrubber's command (§18.2.4).
    ///
    /// Navigation, like [`Undo`](Self::Undo) and [`Redo`](Self::Redo): it moves the
    /// same applied / withheld split those two move one step at a time, so nothing is
    /// logged and nothing is sent.
    ///
    /// Clamped to the range the timeline reports, and a no-op on a timeline that
    /// has no single playhead to move (a shared session — see
    /// [`Engine::scrub_range`](crate::Engine::scrub_range)).
    Seek(usize),

    /// Commit an [`Edit`] — one action, one undo step, superseding any preview in
    /// force.
    Set(Edit),

    /// Add an empty paint layer to the stack carried by `carrier` (the
    /// document's own when `None`), directly above `above` (§14.8).
    AddLayer {
        carrier: Option<LayerId>,
        above: Option<LayerId>,
    },
    /// Copy a layer **and everything it carries** into its own stack, directly
    /// above it (§14.8) — the subtree travels as one, for the reason
    /// [`RemoveLayer`](Self::RemoveLayer)'s does, and so one id is minted per layer
    /// of it.
    DuplicateLayer(LayerId),
    /// Remove a layer **and everything it carries** — the subtree is the group
    /// (§14.2). To keep what it carried, release those layers with
    /// [`MoveLayer`](Self::MoveLayer) first.
    RemoveLayer(LayerId),
    /// Merge a layer **down** onto the one beneath it (§14.11): the lower layer keeps
    /// its name, its place and its properties, and takes the paint of both; the upper
    /// one is gone.
    ///
    /// **Declined when it would change what the document looks like** — which is the
    /// whole of what a merge promises, and the same question the panel asks before
    /// offering the control ([`LayerInfo::merge_down`](crate::LayerInfo::merge_down)),
    /// so a command sent anyway is a silent no-op rather than a different picture.
    MergeLayerDown(LayerId),
    /// Clip a layer to the paint beneath it in its own stack, or stop
    /// (§14.4). On the base of a group this clips the whole group
    /// to what lies under the group.
    SetLayerClip(LayerId, bool),
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
    /// One command for all three gestures (§14.8, §14.6), since a drop lands in *some*
    /// stack at *some* place in it: **reorder** leaves `carrier` as it was, **carry**
    /// sets it to the layer being dropped onto, **release** clears it. Asking a layer
    /// to carry its own ancestor is declined.
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

    /// Add a **matte** layer — a region filled with a [`Parcel`] (§15.2). A frame is
    /// one of these on top of the stack; a substrate ([`MatteRegion::Everything`]) is
    /// one at the bottom, hence the full [`Place`] anchor (§15.5).
    AddMatte {
        carrier: Option<LayerId>,
        at: Place,
        region: MatteRegion,
        paint: Parcel,
    },
    /// Bring an image in from **outside** the document — an image file, or the system
    /// clipboard — as a new layer holding it as paint (§23).
    ///
    /// `at` is the canvas position of the image's top-left texel in **whole canvas
    /// pixels**, so the placement resamples nothing — see
    /// [`ActionKind::PlaceImage`](stark_model::document::ActionKind::PlaceImage).
    /// Where it lands is the frontend's, per the view (§18.1.2); scaling and turning it
    /// afterwards is [`Edit::Transform`], which with nothing selected moves the whole
    /// layer. Decoding is the frontend's too, per the platform.
    ///
    /// The picture is named by **content id**, imported first through
    /// [`Engine::import_picture`](crate::Engine::import_picture) — the same two-step a
    /// stamp brush takes (§6.6), and for the same reasons: the id has to exist before
    /// the action that references it can be built, and in a shared session the bytes
    /// have to be registered before the commit that names them goes out.
    PlaceImage {
        carrier: Option<LayerId>,
        above: Option<LayerId>,
        at: IVec2,
        /// What to call the layer — the file it came from. `None` for an image with no
        /// name, which is what the clipboard hands over.
        name: Option<String>,
        /// The picture's content id, from `Engine::import_picture`. An id this engine
        /// does not hold places an empty layer and warns — see `document::apply`.
        image: AssetId,
    },

    /// Add a **filter** layer — a function of everything composited beneath it in
    /// the stack it lands in (§21.2).
    ///
    /// It arrives at its neutral setting, so adding one changes nothing until it is
    /// dialled with [`Edit::Filter`]. Where it lands is its scope —
    /// the root stack filters the whole painting, a carried stack filters that group
    /// — so there is no parameter saying how far it reaches.
    AddFilter {
        carrier: Option<LayerId>,
        above: Option<LayerId>,
        filter: Filter,
    },

    /// Cut what the author's selection holds on `layer` into a fresh child
    /// layer at the foot of its stack — the float (§16.12) — and make the child
    /// the paint target, since the float is what the hand is about to move.
    /// Refused (nothing logged) when the layer is not paint, nothing is selected, or
    /// the cut would be empty or oversized.
    FloatSelection {
        layer: LayerId,
    },

    /// Switch the canvas substrate (§6.4).
    ///
    /// Document state, not view state: which canvas a piece was painted on is saved,
    /// and reopening on a different substrate would be a different painting. It also
    /// **gates deposition** (§6.4) — the tooth reads the substrate in force at each
    /// action's point in the log, so a switch part-way through changes the strokes
    /// after it and not the ones before.
    SetSubstrate(SubstrateId),

    /// Add a **drawing guide** — a perspective grid to construct through
    /// (§20.5), directly after `after` in the roster or at its head.
    ///
    /// Unlike every other `Add…`, the engine mints no id: a guide's identity is the
    /// id of the action that adds it ([`GuideId`]), so the frontend learns it by
    /// reading the roster back off the projection, and there is no counter here to
    /// resync (§17.9).
    AddGuide {
        guide: PerspectiveGuide,
        after: Option<GuideId>,
        /// What to call it. Trimmed and length-capped on the way in, and one
        /// that comes out blank leaves the guide unnamed rather than named
        /// something blank — [`SetLayerName`](Self::SetLayerName)'s rule, shared
        /// with it (`normalize_name`).
        name: Option<String>,
    },
    /// Remove a drawing guide.
    RemoveGuide(GuideId),
    /// Name a guide, or with `None` clear the name so it goes back to being
    /// described by its place in the roster. Trimmed and capped like
    /// [`SetLayerName`](Self::SetLayerName), through the same funnel.
    SetGuideName(GuideId, Option<String>),
    /// Move a guide within the roster — the panel's drag-to-reorder. `after` is
    /// [`AddGuide`](Self::AddGuide)'s anchor: the guide it lands directly after,
    /// or the head of the roster when `None`.
    MoveGuide {
        id: GuideId,
        after: Option<GuideId>,
    },
}

/// A document value a control sets: what its drag **previews** per pointer sample
/// ([`ViewCommand::Preview`]) and its release **commits** once ([`DocCommand::Set`]).
///
/// A slider, a color picker, a frame handle and the transform widget are judged by
/// looking, so every value the hand crosses has to reach the canvas — and only the
/// answer belongs in the log: one undo step and one replicated action per adjustment
/// rather than per pointer move. Not a [`GestureCommand`], because a control reports
/// the value it wants rather than samples of the motion that reached it.
///
/// Both halves mint their action through one function off the committed document, so
/// a preview is what its commit would lay (§1.3) — a [`Fill`](Self::Fill) or a
/// [`Transform`](Self::Transform) through the commit's own renderer over the committed
/// tiles, so a long drag never compounds. Canvas-space payloads are carried into the
/// layer's frame there (§14.12).
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// A layer's opacity (§14.6).
    LayerOpacity(LayerId, f32),
    /// A layer's blend mode — the whole mode rather than one of its parameters, so a
    /// knob a mode grows needs no variant of its own (§18.0.4).
    LayerBlend(LayerId, BlendMode),
    /// How strongly the **whole** selection mask gates, `0..=1` (§6.8). Document state
    /// where the Select panel's other settings ([`ViewCommand::SetSelectionFeather`],
    /// [`ViewCommand::SetShapeOpacity`]) are view state: those shape the *next*
    /// gesture, and this reaches the mask already drawn.
    SelectionOpacity(f32),
    /// A filter layer's settings (§21.6). The whole filter travels, so a new knob or a
    /// new kind of filter needs no variant of its own.
    Filter(LayerId, Filter),
    /// A matte's rect, `min..max` on the canvas (§15.7).
    MatteRect(LayerId, Vec2, Vec2),
    /// A matte's paint — a flat color or a gradient ramp (§15.4, §22.4).
    MattePaint(LayerId, Parcel),
    /// The canvas substrate's color, straight sRGB (§15.5).
    SubstrateColor(Srgb),
    /// How large the canvas substrate is laid (§6.4) — a gate on deposition, like
    /// [`DocCommand::SetSubstrate`], since the tooth reads its rise over a reach in
    /// canvas px. A preview bakes nothing, so it shows in the light the media pass
    /// re-reads every frame and not in the tooth, a stored bake: paint already down
    /// looks right at once, and what the next stroke bites is right from the commit.
    SubstrateScale(SubstrateScale),
    /// Put `layer`'s frame at `to` on the canvas — a property write that moves no tile
    /// (§14.12, §16.11). The layer's whole subtree moves by the same delta, since a
    /// group moves as one and translation does not inherit.
    Translate { layer: LayerId, to: IVec2 },
    /// A drawing guide — the whole camera at once (§20.5).
    Guide(GuideId, PerspectiveGuide),
    /// Fill a region of `layer` with paint (§18.0.4, §22.4) — the identical action a
    /// fill gesture commits ([`ShapeAction::Fill`]). A region of
    /// [`SelectionShape::All`](stark_model::document::SelectionShape) means the
    /// selection, and is refused when there is none: the canvas is unbounded, and
    /// inventing a boundary would be a different fill on every client.
    Fill { layer: LayerId, op: FillOp },
    /// Transform this client's selected paint on `layer` — affine, perspective or warp
    /// (§16, §16.8, §16.9) — carrying the selection with it; a universal selection
    /// moves the whole layer. A map that is not [usable](TransformMap::usable) is
    /// refused rather than logged (§16.1).
    Transform { layer: LayerId, map: TransformMap },
}

/// Mutations of **view state**: per-client, transient, never logged and never sent
/// to peers. Undo does not reach these, and two people sharing a drawing each have
/// their own.
#[derive(Clone, Debug)]
pub enum ViewCommand {
    SetTool(Tool),
    /// The brush in hand, and beside it the hand's **color** — which is not always
    /// the brush's: an erasing brush carries no pigment, while a fill still lays the
    /// color you have in hand ([`Session::start_selection`]). One command rather than
    /// two, so the two can never arrive out of step.
    ///
    /// [`Session::start_selection`]: crate::session::Session::start_selection
    SetBrush {
        brush: BrushParams,
        color: [f32; 3],
    },
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
    /// turn, because the three are not independent: each anchors against the view it
    /// is applied to, so sent in sequence the last two would anchor against a view the
    /// hand never saw and the canvas would slide out from under it — see
    /// [`ViewTransform::pinch`](crate::view::ViewTransform::pinch).
    ///
    /// **Incremental**, unlike [`SetRotation`](Self::SetRotation): a pinch does not
    /// know what angle it wants, only how far the hand has turned since the last
    /// report. The frontend spends its feel — the twist a gesture must earn, the pull
    /// onto a quarter turn — by choosing `turn`.
    Pinch {
        anchor: Vec2,
        to: Vec2,
        scale: f32,
        turn: f32,
    },
    /// Turn the canvas to this angle (radians, clockwise) — the navigator's
    /// right-drag (§18.1.2). Absolute, like [`CenterOn`](Self::CenterOn).
    ///
    /// The *drag* gives a direction, which
    /// [`ViewTransform::rotation_for_up`](crate::view::ViewTransform::rotation_for_up)
    /// turns into an angle; the easing and the snap-to-square are the frontend's feel.
    SetRotation(f32),
    /// Mirror what is on screen, left↔right — the oldest way of catching a drawing
    /// error, since the eye stops recognising what it expected and starts seeing what
    /// is there.
    ///
    /// A toggle rather than a setting, and **screen-relative**: it swaps the left of
    /// the screen with the right at any angle, so the check means the same thing
    /// however the easel is turned (see
    /// [`ViewTransform::mirror_screen_h`](crate::view::ViewTransform::mirror_screen_h)).
    MirrorH,

    /// Show this canvas-space point at the centre of the viewport, leaving the zoom
    /// alone — a jump rather than a drag.
    ///
    /// Absolute where [`Pan`](Self::Pan) is incremental, because the callers differ
    /// in kind: a drag knows only how far the pointer moved, while a navigator click
    /// knows exactly where it wants to be and nothing about where the view was.
    CenterOn(Vec2),
    /// Put the whole piece on screen: the canvas-space rect an export of `frame`
    /// would write (§15.6), centred and fitted to the viewport with the easel
    /// straightened ([`ViewTransform::show_rect`](crate::view::ViewTransform::show_rect)).
    ///
    /// `frame` names a matte layer whose rect is the piece, exactly as
    /// [`Engine::export_plan`](crate::Engine::export_plan) takes it, and the rect is
    /// worked out by the same rule — so what you are looking at and what a file would
    /// hold cannot disagree about where the piece ends. With no frame it fits the
    /// painted bounds; with neither there is nothing to show and the view stays where
    /// it is (an export, which must write *something*, falls back to the viewport).
    ///
    /// Absolute, like [`CenterOn`](Self::CenterOn). **Opening a document is that
    /// caller** (§8): a painting arrives framed rather than at the pan and zoom the
    /// last one was left at, which on an unbounded canvas can be nowhere near the
    /// paint that just loaded.
    ShowPiece(Option<LayerId>),
    /// The viewport changed size (window/canvas resize).
    Resize(Extent2),

    /// What the next shape gesture does with the region it encloses: combine it into
    /// the selection one of four ways, or fill it (§6.8, §18.0.4). Shapes the *next*
    /// op; the op itself is what gets logged.
    SetShapeAction(ShapeAction),
    /// Edge softness (canvas px) for the next shape gesture — the same ramp whether
    /// it selects or fills.
    SetSelectionFeather(f32),
    /// How strongly a **fill** gesture's parcel lands, `0..=1`
    /// ([`FillOp::opacity`](stark_model::document::FillOp), §18.0.4) — the Select
    /// panel's Opacity slider, mounted under the Fill action.
    ///
    /// Not what a *selecting* gesture lands at: a selection's strength is the whole
    /// mask's and is set after the fact ([`Edit::SelectionOpacity`], §6.8),
    /// which is why that one is the selection bar's slider rather than the panel's.
    SetShapeOpacity(f32),

    /// Whether collaborators' selection outlines are drawn over the canvas (§17.3).
    SetShowPeerSelections(bool),

    /// Open or shut a guide's **eye** (§20.5) — whether this client draws it.
    ///
    /// The one thing about a guide that is *not* document state: shutting a guide to
    /// see the picture underneath is a statement about what you are looking at rather
    /// than about the drawing. Everything else a guide has is logged ([`Edit::Guide`]
    /// and its siblings).
    ///
    /// **A guide is hidden until this client asks for it**, so one arriving from a
    /// peer or out of a file draws nothing until an eye is opened on it — the
    /// alternative lays every perspective the document has ever carried over the
    /// canvas the moment it opens.
    ///
    /// The frontend need not send this on the common path: picking a guide up to
    /// shape it opens its eye (`panels::guides`' `begin_guide_edit`).
    SetGuideVisible(GuideId, bool),

    /// Show the document as committing an [`Edit`] would leave it, **without logging
    /// it** — the in-flight half of a control's drag. `None` takes the preview down.
    Preview(Option<Edit>),

    /// Show the mark the brush **will** make if a drag begins this instant,
    /// **without logging it** — the brush cursor's painted half (§18.1.10).
    /// `None` drops the mark.
    ///
    /// Each report carries the newest hover sample ([`HoverReport`]); the engine
    /// appends it to a trailing window of recent reports, fits the window, and folds
    /// a **probe**: a straight stroke from the cursor, `reach` canvas px along the
    /// fit's extrapolated heading — rendered by the same renderer, through the
    /// selection mask and onto the paint already there, wet-mixing included, so it is
    /// exactly what committing that gesture would land. A window rather than the
    /// newest pair because reports are quantized to the device tolerance and a lone
    /// pair's heading snaps between the eight compass points (`Session::hover_to`).
    ///
    /// Unlike [`Preview`](Self::Preview) this is a prediction rather than the
    /// in-flight half of any commit, and three things follow. It is **outranked** by
    /// a real gesture (the fold holds one gesture per actor); it is **dropped** by
    /// anything that ends its premise — a gesture starting, a tool switch, a load;
    /// and it may never reach a file — a `Rendered::Live` export takes it down first,
    /// because an export is not a moment the hand is painting in.
    PreviewHover(Option<HoverReport>),

    /// Tune the media/lighting pass (§6.3). Changes how the canvas
    /// looks, not what it is.
    SetMediaParams(MediaParams),
    /// Switch the HDR lighting environment (§6.3).
    SetEnvironment(EnvironmentId),
    /// State the display the screen is presented on (§6.5). A view setting, like
    /// the two above, and one an export never reads.
    SetOutput(Output),

    /// How much resident tile memory history retention may hold before the engine
    /// starts giving up undo depth, in bytes (§5).
    ///
    /// **View state**: how deep an undo stack this machine can afford is a fact about
    /// the machine, not about the painting — two people sharing a drawing have
    /// different amounts of memory, and neither answer belongs in the file. The
    /// default ([`DEFAULT_HISTORY_BUDGET`](crate::DEFAULT_HISTORY_BUDGET)) is one
    /// number for devices from a phone to a workstation; a frontend that knows which
    /// it is on should say so.
    ///
    /// Setting it does not itself trim; the next commit does, which is the only moment
    /// the stack grows. `0` means "keep the minimum" — retention is still floored at
    /// `MIN_UNDO_DEPTH` steps, because trimming below that frees nothing when the
    /// memory is held by the current document rather than by history.
    SetHistoryBudget(u64),

    /// Whether a stroke's commit takes the tiles its live preview already drew,
    /// rather than rendering the stroke again when the pointer comes up (§6.2).
    ///
    /// **View state, and per-client for
    /// [`SetHistoryBudget`](Self::SetHistoryBudget)'s reason**: what it changes is how
    /// this machine spends the moment of a release, not what the drawing is. A peer
    /// receives the stroke as an action and renders it whole whatever this says, and
    /// neither setting changes what a *file* means.
    ///
    /// Off buys a bit-for-bit reproduction, on (the default) gives back the hitch at
    /// the end of a long stroke — see
    /// [`DEFAULT_FAST_COMMIT`](crate::DEFAULT_FAST_COMMIT) for the trade.
    SetFastCommit(bool),
}

impl ViewCommand {
    /// [`SetBrush`](Self::SetBrush) with the hand's color taken from the brush's
    /// own pigment — right for every caller whose hand holds nothing beside the
    /// brush: tests, benches, the thumbnail rig. The frontend does **not** come
    /// through here: its hand keeps a color an erasing brush does not carry
    /// (`stark-dioxus-frontend`'s `BrushConfig`), and sends it alongside explicitly.
    pub fn set_brush(brush: BrushParams) -> Self {
        Self::SetBrush {
            color: brush.pigment().unwrap_or([0.0; 3]),
            brush,
        }
    }
}

/// Mutations of **presence**: per-client and never logged — undo does not reach
/// these and they are not in the save file — but *published*, so every collaborator
/// sees them (§17.4, §7).
///
/// What separates these from [`ViewCommand`] is only who reads the result. What
/// separates them from [`DocCommand`] is that replay does not need them to reproduce
/// a pixel: the selected layer is already closed over by
/// [`StrokeRecord::layer`](stark_model::document::StrokeRecord), a cursor paints nothing,
/// and a name is not part of the artwork.
#[derive(Clone, Debug)]
pub enum PeerCommand {
    /// The selected layer — where the next stroke goes, if that layer can take one.
    /// Per-client: collaborators paint on whichever layer each has selected, and each
    /// can see where the others are working.
    ///
    /// A **matte** may be selected like any other layer (§15.7), so selection stays one
    /// concept rather than a paint target plus a separate frame-focus. A stroke aimed
    /// at one draws nothing, refused identically by `apply` and by the preview path, so
    /// the frontend needs no rule of its own.
    SetActiveLayer(LayerId),

    /// Where this client's pointer is, in canvas space; `None` when it leaves the
    /// canvas. Cheap at pointer rate: it writes a field, and the publish latch
    /// coalesces to one frame per tick (§17.5).
    ///
    /// Classified by what it *is* rather than by who reads it: this client's own
    /// guides draw their rays through it (§20.9), the way this client's own next
    /// stroke goes to [`SetActiveLayer`](Self::SetActiveLayer)'s layer. The cost to
    /// the frontend is a repaint, and so a decision about *when* to send this
    /// (`input::point_at`).
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
