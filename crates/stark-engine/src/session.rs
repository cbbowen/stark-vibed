//! Session: ephemeral, non-historized state (§3).
//!
//! The session holds the current tool/brush, the pan/zoom view, and the
//! in-flight stroke being dragged out. None of this is undoable — switching
//! tools or panning never creates a history step. The session interprets
//! pointer commands and, on `EndStroke`, hands the [`Engine`](crate::Engine) a
//! finished [`StrokeRecord`] to commit.

use crate::assist::{AssistShape, PenProfile};
use crate::command::InputSample;
use crate::command::Tool;
use crate::path::PathFitter;
use crate::peer::{GestureView, Identity, LiveGesture, default_name};
use crate::presence::{GestureSource, GestureTx};
use crate::tow::{Tow, TowString};
use crate::view::ViewTransform;
use stark_model::Srgb;
use stark_model::document::Scaffold;
use stark_model::document::{
    ActorId, BrushParams, FillOp, GuideId, LayerId, SelectionMode, SelectionOp, SelectionShape,
    ShapeAction, StrokeRecord,
};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;
use stark_model::peer::{HEARTBEAT, PeerFrame, StrokeHead};

/// Minimum spacing (canvas px) between lasso vertices. The mask shader costs one
/// segment test per texel per vertex, and pointer samples arrive far denser than a
/// mask boundary can resolve, so the polyline is thinned as it is collected —
/// bounding both the rasterization cost and the size of the logged op (§6.8).
const LASSO_MIN_STEP: f32 = 2.0;

/// Accumulates the stroke currently being drawn.
///
/// Pointer samples are fitted to control points *as they arrive* rather than
/// buffered and re-fitted on every move (§6.2): the builder holds only
/// the fitter's short working window, and each new sample costs work proportional
/// to that window instead of to the stroke so far.
struct StrokeBuilder {
    brush: BrushParams,
    layer: LayerId,
    seed: u64,
    fitter: PathFitter,
    /// What the frontend said this gesture's input resolves to (canvas px). Kept
    /// because the drawing assist prices its recognition in the same unit the fit
    /// does — see [`crate::assist`].
    tolerance: f32,
    /// Set once the gesture has been **held** and snapped to a shape (§6.9). From
    /// there the fitter stops being fed and this decides the path.
    assist: Option<Assist>,
    /// The towed tip (§6.11) — `Some` when the gesture started with a rope.
    /// Feeds **the fitter only**: it sits between the raw samples and
    /// [`PathFitter::push`], and a snapped gesture's steering reads the raw
    /// pointer (through [`Assist::shift`]) rather than a towed one.
    tow: Option<Tow>,
}

/// A stroke that has snapped to an ideal shape, and is now being steered rather than
/// extended (§6.9).
///
/// `base` and `grip` are kept beside the current `shape` on purpose: every adjustment
/// is `base.adjust(grip, pointer)`, never `shape.adjust(...)`, so a minute of dragging
/// is the same as the same displacement made at once. `path` is the realization of
/// `shape`, cached because a preview asks for it on every frame and a pointer move is
/// the only thing that can change it.
struct Assist {
    base: AssistShape,
    grip: Vec2,
    shape: AssistShape,
    /// The pen channels of the stroke as it was drawn, carried onto the ideal shape.
    pen: PenProfile,
    /// How many control points the drawn stroke had — the floor on the snapped one's,
    /// so the profile above has somewhere to live.
    knots: usize,
    path: Vec<ControlPoint>,
    /// The string's standing offset at the snap, tip − pointer (§6.11) — zero
    /// for an untowed gesture. A towed stroke snaps to a shape recognized from
    /// the *towed* trace while the raw pointer sits up to a rope beyond it, so
    /// steering with the raw position would apply that gap as a jump on the
    /// first move. Added to every steer instead: the hand's deltas land 1:1 and
    /// nothing jumps, the same bargain the grip strikes for the fit residual.
    shift: Vec2,
}

impl Assist {
    /// Re-derive the shape and its path for a pointer at `pos`.
    fn steer(&mut self, pos: Vec2) {
        let shape = self.base.adjust(self.grip, pos + self.shift);
        if shape == self.shape {
            return;
        }
        self.shape = shape;
        self.path = shape.to_path(&self.pen, self.knots);
    }
}

/// What a finished shape gesture resolves to (§6.8,
/// §18.0.4): an edit to the selection, or a fill.
///
/// The two travel together because they are one gesture with one preview — the
/// [`ShapeAction`] chosen when the drag started decides which of them the release
/// commits, and nothing downstream has to ask what tool was in hand.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeResult {
    Select(SelectionOp),
    Fill(FillOp),
}

/// What a shape gesture's action *means* against the selection it is drawn over:
/// **Add, with nothing selected, is New** (§6.8).
///
/// The algebra says otherwise and goes on saying it — `max(1, s) = 1`, so a union
/// with the unrestricted selection *is* the unrestricted selection, and that is the
/// answer every peer would rasterize. What it is not is what anyone means by the
/// gesture: "add this region to the selection" with no selection in hand asks for a
/// selection of that region, and a mask that comes back covering everything reads as
/// the tool having done nothing at all. The Select panel's action row is one question
/// with five answers — *what does this shape do?* — and one of them being inert on a
/// fresh document is the row failing to answer it.
///
/// Resolved **here, where a gesture becomes an op**, and deliberately not in the
/// mask algebra. `SelectionMode::combine` stays the honest soft-set operation it
/// documents, `Selection::plan` keeps its four identities, and what reaches the log
/// is `Replace` — which is what the user got. So replay, undo, a save file and a peer
/// receiving the op all reproduce the picture without knowing this rule exists, and
/// no reordering of the log can make one op mean two things (§12.6).
///
/// Only `Union` has anything to answer for. Subtracting from everything is the
/// complement and intersecting with it is the shape, both of which are already what
/// the gesture reads as.
fn against_selection(action: ShapeAction, has_selection: bool) -> ShapeAction {
    match action {
        ShapeAction::Select(SelectionMode::Union) if !has_selection => {
            ShapeAction::Select(SelectionMode::Replace)
        }
        action => action,
    }
}

/// The shape gesture currently being dragged out (§6.8). Like a stroke it
/// is ephemeral: only the [`ShapeResult`] it resolves to on release is committed,
/// and the shape is derived from the drag on demand so a live preview and the
/// committed op come from exactly the same code.
struct ShapeDrag {
    tool: Tool,
    /// What the release will do with the region — captured at the *start* of the
    /// drag, like the feather, so re-picking a chip mid-gesture cannot change what
    /// the gesture already looks like it is doing.
    ///
    /// The action as [`against_selection`] resolved it, not as the panel is set:
    /// what a gesture means depends on what it is drawn over, and the only moment
    /// that is settled is the press.
    action: ShapeAction,
    feather: f32,
    /// The color a fill will lay, taken off the brush when the drag began. Unused
    /// by a selecting gesture, which has no paint.
    color: Srgb,
    /// How strongly a **fill** lands — captured with the rest, so moving the slider
    /// mid-drag cannot change what the drag already looks like it is doing.
    ///
    /// Unused by a selecting gesture, which mints its op at full strength: how
    /// strongly a selection gates is the *whole* mask's opacity now, set after the
    /// region is drawn rather than baked into it (§6.8). The per-shape
    /// [`SelectionOp::opacity`] still stands underneath — nothing in the UI reaches
    /// it.
    opacity: f32,
    /// Where the drag started; for the marquees this is one corner of the box.
    start: Vec2,
    /// The lasso's decimated outline (empty for the marquees).
    points: Vec<Vec2>,
    /// The newest sample, so the marquees can span `start`..`current`.
    current: Vec2,
}

impl ShapeDrag {
    fn push(&mut self, pos: Vec2) {
        self.current = pos;
        if self.tool == Tool::SelectLasso
            && self
                .points
                .last()
                .is_none_or(|q| q.distance(pos) >= LASSO_MIN_STEP)
        {
            self.points.push(pos);
        }
    }

    /// The region this drag currently encloses — `None` for a gesture that encloses
    /// nothing (a click with a marquee, a lasso too short to have an interior).
    fn to_shape(&self) -> Option<SelectionShape> {
        let shape = match self.tool {
            Tool::SelectRect => {
                let (min, max) = (self.start.min(self.current), self.start.max(self.current));
                if (max - min).min_element() <= 0.0 {
                    return None;
                }
                SelectionShape::rect_from_corners(self.start, self.current)
            }
            Tool::SelectEllipse => {
                let (min, max) = (self.start.min(self.current), self.start.max(self.current));
                if (max - min).min_element() <= 0.0 {
                    return None;
                }
                SelectionShape::ellipse_from_corners(self.start, self.current)
            }
            Tool::SelectLasso => {
                // Close the loop with the newest sample: the shape has to reach the
                // cursor mid-gesture, exactly as a stroke preview does.
                //
                // No second decimation pass: `push` already enforces `LASSO_MIN_STEP`
                // between consecutive kept points as they arrive, and the one point
                // that could fail that test — the trailing `current` — is one
                // `decimate` would put straight back under its keep-the-last-sample
                // rule. Running it here was an O(n) scan and an allocation, per
                // preview frame *and* per publish tick, that could not change a thing.
                let mut points = self.points.clone();
                if points.last().is_none_or(|q| *q != self.current) {
                    points.push(self.current);
                }
                if points.len() < 3 {
                    return None;
                }
                SelectionShape::Lasso(points)
            }
            Tool::Brush => return None,
        };
        Some(shape)
    }

    /// The action this drag currently stands for — what a release right now would
    /// commit, and equally what the preview fold draws.
    fn to_result(&self) -> Option<ShapeResult> {
        let shape = self.to_shape()?;
        Some(match self.action {
            // At full strength: the mask's own opacity is a separate, retroactive
            // number (`ActionKind::SetSelectionOpacity`, §6.8), so a shape says
            // *where* and never how strongly.
            ShapeAction::Select(mode) => {
                ShapeResult::Select(SelectionOp::new(mode, shape, self.feather))
            }
            ShapeAction::Fill => {
                ShapeResult::Fill(FillOp::new(shape, self.feather, self.color, self.opacity))
            }
        })
    }
}

/// What this session last put on the wire, for change detection (see
/// [`Session::publish`]). Compared rather than dirty-flagged: a comparison cannot be
/// forgotten at a call site, and these are three cheap fields.
///
/// The gesture is *not* here: what the wire has been told about it is
/// [`GestureTx`]'s business, and asking it ([`GestureTx::in_flight`]) beats keeping a
/// second copy in step.
#[derive(Clone, PartialEq)]
struct Published {
    active_layer: LayerId,
    cursor: Option<Vec2>,
    name: String,
}

pub struct Session {
    pub view: ViewTransform,
    pub tool: Tool,
    pub brush: BrushParams,
    /// The **hand's** color — straight sRGB, the Color panel's number. Its own
    /// field rather than a reading of the brush because the brush does not
    /// always have one: an erasing brush carries no pigment, and a fill drawn
    /// while one is held still lays the color you have in hand
    /// ([`start_selection`](Self::start_selection)). While the brush paints, the
    /// frontend keeps this equal to the paint effect's own color
    /// (`ViewCommand::SetBrush` carries both together).
    pub color: [f32; 3],
    pub active_layer: LayerId,
    /// What the next shape gesture does with the region it encloses (§6.8,
    /// §18.0.4) — one of the four ways to combine it into the
    /// selection, or fill it.
    pub shape_action: ShapeAction,
    /// Edge softness (canvas px) applied by the next shape gesture, whichever
    /// action it takes: a feathered fill and a feathered selection are the same
    /// ramp, rasterized by the same shader.
    pub selection_feather: f32,
    /// How strongly the next **fill** gesture's parcel lands, `0..=1` — the Select
    /// panel's Opacity slider under the Fill action, and [`FillOp::opacity`].
    ///
    /// The counterpart of [`selection_feather`](Self::selection_feather) only for
    /// that one action: one says how soft the coverage's edge is, this says how
    /// strong the paint it lays is. Under the four *selecting* actions the same
    /// question is asked of the mask instead, by the selection bar's slider — and
    /// that answer is document state, because it reaches the region already drawn
    /// (`ActionKind::SetSelectionOpacity`, §6.8).
    pub shape_opacity: f32,
    /// Whether collaborators' selection outlines are drawn (§17.3).
    ///
    /// View state, so each client decides for itself and nothing about it is logged
    /// or sent. Off by default: knowing which region someone else is working inside
    /// is occasionally useful, but a second contour over the artwork is a cost paid
    /// on every frame you look at it, and most of the time the answer to "what is
    /// that line?" should be "the one I drew".
    pub show_peer_selections: bool,
    /// The drawing guides whose **eye this client has opened** (§20.5).
    ///
    /// The one per-client thing about a guide. The guides themselves are document
    /// state now — logged, saved, replicated, undoable (`DocState::guides`) —
    /// because a perspective set up over a drawing is part of its construction;
    /// but whether *you* are looking at one is not, so it is here beside the pan
    /// and the zoom, never logged and never sent.
    ///
    /// **Opened rather than shut, so an absent entry means not drawn**, and a
    /// guide is therefore hidden until somebody on this client asks for it. That
    /// is the direction that reads right once a guide can arrive from somewhere
    /// else: opening a painting from last year, or joining a session, would
    /// otherwise lay every perspective anyone ever built over the canvas at once,
    /// and a scaffold you did not ask for is exactly what a scaffold must not be.
    /// The document remembers the construction; looking at it is a thing you do.
    ///
    /// It is also the cheaper end. A client typically has one guide up while it
    /// works, out of however many the document carries, so the set holds what is
    /// *in use* rather than what is being ignored — the same argument
    /// `DocState::selections` makes for storing only the actors that have selected
    /// something, pointed the other way because the common case is the other way.
    ///
    /// Nothing puts an id in here but the artist: `panels::guides`'
    /// `begin_guide_edit` opens the eye of a guide picked up to be shaped — which
    /// covers adding and duplicating one too, since both end by picking up what
    /// they made — and the row's eye toggles it directly.
    ///
    /// A guide removed from the document leaves its id here. Deliberately: the
    /// removal may be undone, and an eye that shut itself because the guide went
    /// away and came back would be the tool changing what you are looking at.
    /// Bounded by the number of guides an artist has ever opened, which is not a
    /// number that grows on its own.
    visible_guides: std::collections::HashSet<GuideId>,

    // --- the published half (§17.4) -------------------------
    //
    // Everything above this line is *private* view state — pan/zoom, brush, the
    // selection mode. Everything below it, plus `active_layer` and the in-flight
    // gesture, is what [`Session::publish`] projects for other clients. The split is
    // recorded by what that method returns rather than described in a comment, which
    // is the same discipline §4 applies to the command classes.
    /// This client's display name; empty until it is set, in which case peers fall
    /// back to [`default_name`]. Private because it has an invariant `name_chosen`
    /// carries — see [`Session::set_name`].
    name: String,
    /// Whether [`name`](Self::name) is one the *user* chose rather than the
    /// id-derived default. Sharing or joining a document mints this client a new
    /// actor id and wants to refresh that default, but must not overwrite a name
    /// somebody typed.
    name_chosen: bool,
    /// Hover position in canvas space; `None` when the pointer is off the canvas.
    pub cursor: Option<Vec2>,

    in_flight: Option<StrokeBuilder>,
    selecting: Option<ShapeDrag>,
    /// The hover mark (§18.1.10): the stroke a drag begun this instant would
    /// open — laid from the cursor along the hover's extrapolated heading —
    /// previewed under the resting pointer. `None` when the pointer is
    /// elsewhere or the hand is doing something other than hovering.
    ///
    /// Beside [`in_flight`](Self::in_flight) because it is the same kind of
    /// thing — an uncommitted stroke the fold draws — and deliberately **not**
    /// it: this one never commits, never publishes (the wire half,
    /// [`gesture_source`](Self::gesture_source), reads only the two slots
    /// above), and never sets [`is_stroking`](Self::is_stroking). A real
    /// gesture always outranks it ([`hover_view`](Self::hover_view)).
    hover: Option<HoverStroke>,
    /// Bumped on every gesture start, so a peer can tell a restart from a
    /// continuation without a clock — and on every hover update, so the
    /// renderer's per-actor head cache can never mistake one hover's render
    /// for the next's, or for a real gesture's. **One counter for both on
    /// purpose**: two counters could collide, and a collision revives a cached
    /// head — including one still holding a cancelled stroke's paint.
    gesture_ordinal: u64,

    // Publish bookkeeping — the latch, not a queue (§17.5).
    published: Option<Published>,
    /// Which run of this client is publishing; see [`Identity::boot`]. Rides every
    /// frame so a peer can order this run's frames against the previous one's.
    boot: u64,
    pub_seq: u64,
    pub_at: f64,
    /// The sending half of the gesture protocol — the watermarks recording what the
    /// wire has been told. Its counterpart, [`GestureRx`](crate::presence::GestureRx),
    /// lives beside it in [`crate::presence`] rather than in the receiver's module.
    tx: GestureTx,
}

impl Session {
    pub fn new(view: ViewTransform, active_layer: LayerId) -> Self {
        Self {
            view,
            tool: Tool::Brush,
            brush: BrushParams::default(),
            color: [0.0, 0.0, 0.0],
            active_layer,
            shape_action: ShapeAction::default(),
            selection_feather: 0.0,
            shape_opacity: 1.0,
            show_peer_selections: false,
            visible_guides: std::collections::HashSet::new(),
            name: String::new(),
            name_chosen: false,
            cursor: None,
            in_flight: None,
            selecting: None,
            hover: None,
            gesture_ordinal: 0,
            published: None,
            boot: 0,
            pub_seq: 0,
            pub_at: f64::NEG_INFINITY,
            tx: GestureTx::new(),
        }
    }

    /// This client's display name, as peers see it. Empty when none has been set, in
    /// which case peers show [`default_name`] instead.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the display name the user chose. **Sticky**: hosting or joining a session
    /// mints this client a new actor id, and
    /// [`adopt_default_name`](Self::adopt_default_name) will not overwrite a name set
    /// here. Setting it empty gives the choice back, and peers resume showing the
    /// id-derived default.
    pub fn set_name(&mut self, name: String) {
        let name = name.trim();
        self.name_chosen = !name.is_empty();
        self.name.clear();
        self.name.push_str(name);
    }

    // --- the guide roster's per-client half (§20.5) --------------------------

    /// Whether this client draws the guide `id`. A guide nobody here has opened is
    /// **not** drawn, which is what an absent entry in [`visible_guides`] means (§20.5).
    ///
    /// [`visible_guides`]: Self::visible_guides
    pub fn guide_visible(&self, id: GuideId) -> bool {
        self.visible_guides.contains(&id)
    }

    /// Open or shut a guide's eye. Answers whether that changed anything, so a
    /// command that says what is already true costs no repaint and no
    /// observation.
    pub fn set_guide_visible(&mut self, id: GuideId, visible: bool) -> bool {
        if visible {
            self.visible_guides.insert(id)
        } else {
            self.visible_guides.remove(&id)
        }
    }

    /// The guides this client actually draws, in roster order: the document's own
    /// list, kept down to the ones whose eye this client has opened.
    ///
    /// **The one place the two halves of a guide meet** (§20.5). Everything that
    /// asks "what perspective is on the screen" — the guide overlay pass, the
    /// stroke assist's [`Scaffold`], the panel's rows — asks it through here, so
    /// the filter is written once rather than repeated wherever a roster is read.
    ///
    /// Takes the document rather than reaching for one, because a `Session` has
    /// none: it is the *other* half, and which document it is combined with is
    /// the caller's business — the committed one for a commit, the previewed one
    /// for what is on screen.
    pub fn shown_guides<'a>(
        &'a self,
        doc: &'a crate::document::DocState,
    ) -> impl Iterator<Item = &'a crate::document::Guide> {
        doc.guides()
            .iter()
            .filter(|g| self.visible_guides.contains(&g.id))
    }

    /// Adopt the identity a session has given this client.
    ///
    /// Records the run counter every published frame carries, and takes the
    /// id-derived name as a default — unless the user has chosen one. That used to
    /// assign unconditionally, which meant pressing *Share* replaced a name someone
    /// had typed with a hex id.
    pub fn adopt_identity(&mut self, identity: Identity) {
        self.boot = identity.boot;
        if !self.name_chosen {
            self.name = default_name(identity.actor);
        }
    }

    /// The publishable half of this session, if anything a peer would care about has
    /// changed since the last call — otherwise `None` (§17.5).
    ///
    /// This is a **latch, not a queue**: it reports the *current* state, and the
    /// path delta is computed here, at drain time, against what has actually been
    /// sent. A pen reporting at 240 Hz against a 30 Hz publish tick therefore
    /// coalesces losslessly — eight moves produce one frame carrying all eight
    /// control points — which is exactly why presence is allowed to be lossy where
    /// the action log is not.
    ///
    /// A frame goes out when something changed, when a gesture is in flight (its
    /// path just grew), or every [`HEARTBEAT`] regardless, so a silent peer still
    /// proves it is here.
    pub fn publish(&mut self, now: f64) -> Option<PeerFrame> {
        // Every `GESTURE_RESYNC`, re-send the gesture's invariant head and its whole
        // path. That repairs any receiver that missed a delta and primes any client
        // that arrived mid-stroke — without either of them having to ask, which is
        // what keeps the wire one-way and the sender stateless about its audience.
        let resync = self.tx.resync_due(now);
        let source = self.gesture_source();

        let state = Published {
            active_layer: self.active_layer,
            cursor: self.cursor,
            name: self.name.clone(),
        };
        let changed = self.published.as_ref() != Some(&state)
            // A live gesture's path grows every move, so its frame always differs.
            || source.is_some()
            // ...and one that has just ended still owes the frame that clears it.
            || self.tx.in_flight();
        if !changed && now - self.pub_at < HEARTBEAT {
            return None;
        }

        // The name only rides a frame when it changed, or on a resync — a peer that
        // already knows it does not need it thirty times a second.
        let name = (self.published.as_ref().is_none_or(|p| p.name != self.name) || resync)
            .then(|| self.name.clone())
            .filter(|n| !n.is_empty());

        // Everything below this line commits to sending, which is the only point at
        // which the watermarks may move: they record what the receiver has been told.
        // Encoding *before* the early return above worked only because a gesture
        // always forced `changed` — a coupling nothing stated and nothing checked.
        let gesture = self.tx.encode(self.gesture_ordinal, source, resync);
        if resync {
            self.tx.stamp_resync(now);
        }
        self.pub_at = now;
        self.pub_seq += 1;
        self.published = Some(state);
        Some(PeerFrame {
            boot: self.boot,
            seq: self.pub_seq,
            name,
            active_layer: self.active_layer,
            cursor: self.cursor,
            gesture,
            leaving: false,
        })
    }

    /// Whether [`publish`](Self::publish) could produce a frame.
    ///
    /// Deliberately **conservative**: it may say yes where `publish` then returns
    /// `None`, but it must never say no where `publish` would have produced a frame,
    /// because a pump that trusts it would then drop that frame on the floor. So it
    /// tests the cheap fields directly and treats "a gesture exists" as "something
    /// changed" without building the gesture frame to find out.
    ///
    /// It exists so an idle session costs nothing: without it the pump has to take a
    /// mutable borrow of the engine thirty times a second to discover there was
    /// nothing to send.
    pub fn publish_due(&self, now: f64) -> bool {
        now - self.pub_at >= HEARTBEAT
            || self.in_flight.is_some()
            || self.selecting.is_some()
            // A gesture that just ended: the frame clearing it is the one that stops
            // peers drawing a stroke nobody is making any more.
            || self.tx.in_flight()
            || self.published.as_ref().is_none_or(|p| {
                p.active_layer != self.active_layer
                    || p.cursor != self.cursor
                    || p.name != self.name
            })
    }

    /// The farewell frame: one publish that removes this client from every peer's
    /// roster at once, rather than making them wait out [`PEER_TIMEOUT`](stark_model::peer::PEER_TIMEOUT).
    pub fn publish_leaving(&mut self) -> PeerFrame {
        self.pub_seq += 1;
        PeerFrame {
            boot: self.boot,
            seq: self.pub_seq,
            name: None,
            active_layer: self.active_layer,
            cursor: None,
            gesture: None,
            leaving: true,
        }
    }

    /// The gesture in flight, in the form the encoder reads it — a window onto the
    /// live fitter, not a copy of it. `None` when nothing is being dragged out (which
    /// includes a selection gesture that so far encloses nothing).
    ///
    /// Only frozen control points are reported as settled: the provisional tail is
    /// resent every frame because it can still move (§6.2). That is the same
    /// partition the renderer's `FrozenHead` uses, spent on the wire instead of on
    /// the GPU.
    fn gesture_source(&self) -> Option<GestureSource> {
        match self.in_flight.as_ref() {
            Some(b) => {
                let (path, start) = b.fitted();
                Some(GestureSource::Stroke {
                    head: StrokeHead {
                        layer: b.layer,
                        brush: b.brush,
                        seed: b.seed,
                    },
                    path,
                    // A snapped stroke has **no settled prefix**: steering it moves every
                    // control point at once, so nothing may be retired. The whole path
                    // therefore rides every frame — which it can afford to, being a shape.
                    // The ordinal bump in `assist_stroke` is what lets the watermark drop
                    // back to zero without walking backwards within one gesture (§17.5).
                    frozen: b
                        .assist
                        .as_ref()
                        .map_or_else(|| b.fitter.frozen_points(), |_| 0),
                    start,
                })
            }
            None => self.preview_shape().map(|r| match r {
                ShapeResult::Select(op) => GestureSource::Selection(op),
                ShapeResult::Fill(op) => GestureSource::Fill(op),
            }),
        }
    }

    pub fn is_stroking(&self) -> bool {
        self.in_flight.is_some()
    }

    /// The ordinal of the gesture in flight — bumped on every start, so a cached
    /// render of one stroke is never mistaken for a render of the next.
    pub fn gesture_ordinal(&self) -> u64 {
        self.gesture_ordinal
    }

    /// Whether a shape gesture is being dragged out (§6.8).
    pub fn is_selecting(&self) -> bool {
        self.selecting.is_some()
    }

    /// Begin a shape gesture with the session's current action, feather and brush.
    /// Any in-flight stroke or earlier gesture is abandoned.
    ///
    /// `has_selection` is the engine's to supply — whether the author already has a
    /// mask in force (`DocState::has_selection`) — because it decides what an Add
    /// gesture means; see [`against_selection`]. Off the *committed* document, which
    /// is the only selection this gesture can be adding to: the one thing that could
    /// change it mid-drag is this drag.
    pub fn start_selection(&mut self, tool: Tool, pos: Vec2, has_selection: bool) {
        self.tool = tool;
        self.in_flight = None;
        // The press supersedes the hover it interrupted — and the window must
        // not survive the gesture, or the mark would reappear at the pre-press
        // position the moment the hand lifted (§18.1.10).
        self.hover = None;
        self.gesture_ordinal += 1;
        let [r, g, b] = self.color;
        self.selecting = Some(ShapeDrag {
            tool,
            action: against_selection(self.shape_action, has_selection),
            feather: self.selection_feather,
            // The color is the hand's ([`color`](Self::color)) — a fill lays the
            // paint you have in hand, even while the brush held is an eraser.
            // Its *alpha* is not: that is the brush's pigment talking, and how
            // strongly a fill lands is the panel's own question (see
            // [`shape_opacity`](Self::shape_opacity)).
            color: Srgb::new([r, g, b]),
            opacity: self.shape_opacity,
            start: pos,
            points: vec![pos],
            current: pos,
        });
    }

    /// Extend the in-flight shape gesture.
    pub fn selection_to(&mut self, pos: Vec2) {
        if let Some(drag) = self.selecting.as_mut() {
            drag.push(pos);
        }
    }

    /// What the in-flight gesture currently stands for, for live preview — the very
    /// same call [`Self::end_shape`] commits, so preview == committed.
    pub fn preview_shape(&self) -> Option<ShapeResult> {
        self.selecting.as_ref().and_then(ShapeDrag::to_result)
    }

    /// Finish the gesture, returning what to commit (`None` if it encloses nothing).
    ///
    /// The **selecting** actions are momentary: having drawn a selection, the session
    /// hands the canvas straight back to the brush. Selecting is a step *towards*
    /// painting, essentially never something you do twice in a row, so making it
    /// modal costs a deliberate switch-back on the overwhelmingly common path — and
    /// leaves a brush gesture silently redefining the selection when the user forgets.
    ///
    /// **Fill** stays armed, because that argument is about a gesture that is a step
    /// towards painting, and a fill *is* painting. Blocking in is done many times in
    /// a row, and being handed the brush back after each one would be the same cost
    /// the momentary rule exists to avoid, paid in the other direction. So the rule
    /// is one sentence rather than a special case: the tool disarms when the gesture
    /// was a step towards painting, and stays armed when the gesture was painting.
    ///
    /// A gesture that enclosed nothing (a stray click) leaves the tool armed either
    /// way, so a mis-click doesn't disarm it.
    pub fn end_shape(&mut self) -> Option<ShapeResult> {
        let result = self.selecting.take().and_then(|d| d.to_result());
        if matches!(result, Some(ShapeResult::Select(_))) {
            self.tool = Tool::Brush;
        }
        result
    }

    /// Begin a stroke. `seed` is supplied by the engine so it can be derived
    /// deterministically (§6.2). `tolerance` is what the frontend says its
    /// input resolves to, in canvas px ([`PathFitter::with_tolerance`]). Replaces any
    /// abandoned in-flight one.
    /// `rope` is the stroke-smoothing string length in canvas px (§6.11), also
    /// the frontend's to state — it derives it from the brush's own smoothing
    /// amount, in screen px, because wobble is a fact about the hand. `0` (or
    /// anything unusable) constructs no tow at all: the fitter is fed the raw
    /// samples, bit-identically to the pre-§6.11 path.
    pub fn start_stroke(
        &mut self,
        tool: Tool,
        sample: InputSample,
        seed: u64,
        tolerance: f32,
        rope: f32,
    ) {
        self.tool = tool;
        self.selecting = None;
        // The press supersedes the hover — but not before its window is put to
        // work: the estimator becomes the stroke's **run-up**, real leading
        // samples the fitted curve extends back through, with the record's
        // marker saying where on that curve the stroke itself begins — so the
        // entry's direction and curvature are measured from watched motion
        // rather than guessed from the first tolerance-quantized steps (§6.2,
        // `PathFitter::seed_runup`). Taking the window here also keeps
        // `start_selection`'s promise: a stale one outliving the gesture would
        // resurrect a pre-press mark at pen-up (§18.1.10).
        let runup = self.take_hover_context(sample.pos, tolerance);
        self.gesture_ordinal += 1;
        let mut fitter = PathFitter::with_tolerance(tolerance);
        fitter.seed_runup(&runup);
        fitter.push(sample);
        self.in_flight = Some(StrokeBuilder {
            brush: self.brush,
            layer: self.active_layer,
            seed,
            fitter,
            tolerance,
            assist: None,
            // A non-finite first report anchors nothing: the tip would park on it
            // and no later report could pull the string off it, so the gesture runs
            // untowed rather than towed from nowhere. The fitter drops the same
            // report on its own account, so the stroke is the one its finite reports
            // describe either way.
            tow: (rope.is_finite() && rope > 0.0 && sample.is_admissible())
                .then(|| Tow::new(rope, tolerance, sample)),
        });
    }

    /// Extend the in-flight stroke with another sample — or, once it has snapped,
    /// **steer** the shape it snapped to (§6.9).
    ///
    /// One entry point for both halves of the gesture, so the frontend goes on sending
    /// the same `To` for every pointer move and nothing about the dwell has to be
    /// mirrored on that side. What the drag means changed; how it arrives did not.
    ///
    /// A report carrying a non-finite number is dropped here, before it reaches any
    /// of the three things that would *remember* it. The fitter refuses one on its
    /// own account too ([`PathFitter::push`]), but it is not the only state a report
    /// enters: [`Tow`] keeps the tip and the target across reports, so one NaN parks
    /// the string somewhere no later report can pull it back from, and
    /// [`Assist::steer`] folds the position into a snapped shape that outlives it.
    /// Refusing at the door is what makes the drop cost one report rather than the
    /// rest of the gesture.
    ///
    /// [`PathFitter::push`]: crate::path::PathFitter::push
    /// [`Assist::steer`]: crate::assist::Assist::steer
    pub fn stroke_to(&mut self, sample: InputSample) {
        if !sample.is_admissible() {
            return;
        }
        if let Some(b) = self.in_flight.as_mut() {
            let StrokeBuilder {
                assist,
                tow,
                fitter,
                ..
            } = b;
            match assist.as_mut() {
                Some(assist) => assist.steer(sample.pos),
                // The towed tip (§6.11) sits between the raw stream and the
                // fitter — and only there, which is why the steer above reads
                // the raw position.
                None => match tow.as_mut() {
                    Some(tow) => tow.to(sample, &mut |s| fitter.push(s)),
                    None => fitter.push(sample),
                },
            }
        }
    }

    /// Snap the stroke in flight to the shape it resembles — what a **hold** means
    /// (§6.9). Returns whether anything snapped.
    ///
    /// Declining is a normal outcome, not a failure: a stroke that is neither a line
    /// nor an ellipse is left exactly as it was drawn, and the gesture carries on
    /// through the fitter as though nothing had happened. The frontend needs no answer
    /// for that reason — it asks once per dwell and the canvas either changes or does
    /// not.
    ///
    /// A snap **bumps the gesture ordinal**, because it is a discontinuity in a stream
    /// that is otherwise append-only: the path is replaced rather than extended. That
    /// one increment is what invalidates the renderer's cached head (§6.2) and makes
    /// peers restart their assembly (§17.5) instead of splicing a delta onto a path
    /// that no longer exists.
    ///
    /// A recognized shape is offered whatever perspective guides are on the screen: a
    /// line takes an axis it is aimed near (§20.6), a loop becomes a circle on a plane
    /// it nearly lies on (§20.7). That is the only coupling between the two features,
    /// and it runs one way — the guides are read, never touched, and a document with
    /// none up puts up an empty scaffold.
    ///
    /// The scaffold arrives from the caller rather than being built here, and that
    /// is the guides having become document state (§20.5): they live in `DocState`
    /// now and the eye that hides one lives in this session, so the only place
    /// holding both halves is `Engine` — see [`Session::shown_guides`], which is
    /// the filter, and `Engine::scaffold`, which applies it.
    pub fn assist_stroke(&mut self, guides: &Scaffold) -> bool {
        let Some(b) = self.in_flight.as_mut() else {
            return false;
        };
        if b.assist.is_some() {
            return false;
        }
        let Some(base) = crate::assist::recognize(&b.fitter.trace(), b.tolerance, guides) else {
            return false;
        };
        let drawn = b.fitter.path();
        let pen = PenProfile::of(&drawn);
        let knots = drawn.len();
        b.assist = Some(Assist {
            base,
            grip: base.grip(),
            shape: base,
            path: base.to_path(&pen, knots),
            pen,
            knots,
            // A towed gesture snapped from its towed trace; the raw pointer is
            // up to a rope beyond it, and this is what keeps the first steer
            // from applying that gap as a jump (§6.11).
            shift: b.tow.as_ref().map_or(Vec2::ZERO, |t| {
                let s = t.string();
                s.tip - s.target
            }),
        });
        self.gesture_ordinal += 1;
        true
    }

    /// Which shape the stroke in flight has snapped to (§6.9), or `None` where
    /// there is no stroke or the hold found nothing to snap it to.
    ///
    /// The family rather than the shape (`assist::Assisted`): the geometry is the
    /// assist's own business, and what is asked from outside the crate is only
    /// whether a hold landed and whether it landed on a line.
    pub fn assisted(&self) -> Option<crate::assist::Assisted> {
        self.in_flight
            .as_ref()
            .and_then(|b| b.assist.as_ref())
            .map(|a| a.shape.family())
    }

    /// This client's gesture in flight as the preview fold wants it, authored by
    /// `actor` — the same shape [`Peer::gesture_view`](crate::peer::Peer::gesture_view)
    /// produces for everyone else, so the fold treats them alike without either being
    /// stored in the other's form.
    pub fn gesture_view(&self, actor: ActorId) -> Option<GestureView> {
        let (gesture, frozen_spans) = match self.preview_record() {
            Some(rec) => (LiveGesture::Stroke(rec), self.frozen_spans()),
            // A marquee has no settled prefix to retire: its closing edge follows the
            // cursor, so every part of it can still move.
            None => match self.preview_shape()? {
                ShapeResult::Select(op) => (LiveGesture::Selection(op), 0),
                ShapeResult::Fill(op) => (
                    LiveGesture::Fill {
                        layer: self.active_layer,
                        op,
                    },
                    0,
                ),
            },
        };
        Some(GestureView {
            actor,
            gesture,
            ordinal: self.gesture_ordinal,
            frozen_spans,
        })
    }

    /// Snapshot the in-flight stroke as a record without ending it, for live
    /// preview (§6.2). `None` if no stroke is active.
    pub fn preview_record(&self) -> Option<StrokeRecord> {
        self.in_flight.as_ref().map(StrokeBuilder::to_record)
    }

    /// How many spans of the in-flight stroke are settled — the prefix a live
    /// preview could render once instead of repainting per pointer move
    /// (see [`PathFitter::frozen_spans`]). 0 when no stroke is active.
    ///
    /// Also 0 once the stroke has snapped (§6.9): steering a shape moves every control
    /// point, so there is no settled prefix to retire — the same answer a marquee gives
    /// for the same reason.
    pub fn frozen_spans(&self) -> usize {
        self.in_flight.as_ref().map_or(0, |b| match b.assist {
            Some(_) => 0,
            None => b.fitter.frozen_spans(),
        })
    }

    /// Finish the stroke, returning the record to commit — `None` if empty, and
    /// `None` for a record that cannot deposit: a swept deposit is a definite
    /// integral over travel, and a step in the log that changes no pixel would
    /// spend an undo step invisibly. The travel that counts starts at the
    /// record's marker, so a click at the end of a watched approach is still a
    /// click — its curve holds the whole run-up, and its marker sits at the
    /// very end of it ([`PathFitter::painted`]). The same answer a marquee
    /// click has always been given ([`Self::end_shape`]).
    pub fn end_stroke(&mut self) -> Option<StrokeRecord> {
        self.in_flight
            .take()
            .and_then(|mut b| {
                // A snapped stroke's path is the shape's, not the fit's — there is
                // nothing left for a last solve to settle.
                if b.assist.is_none() {
                    // The towed tip needs nothing at pen-up (§6.11): lifting stops
                    // pulling the string rather than reeling the tip in, so the fit
                    // closes on the last towed sample and the mark ends where the
                    // rope had towed it to — the trace the preview was showing.
                    b.fitter.finish();
                    if !b.fitter.painted() {
                        return None;
                    }
                }
                Some(b.to_record())
            })
            .filter(|rec| rec.path.len() >= 2)
    }

    /// The in-flight tow, for the frontend's string overlay (§6.11): `None`
    /// when no stroke is active, the gesture carries no rope, or it has
    /// snapped to a shape (a steered shape is driven by the raw pointer, so
    /// there is no string to show).
    pub fn tow_string(&self) -> Option<TowString> {
        self.in_flight
            .as_ref()
            .filter(|b| b.assist.is_none())
            .and_then(|b| b.tow.as_ref())
            .map(Tow::string)
    }

    /// Discard the in-flight stroke without committing. Takes the hover mark
    /// with it: every caller — a pinch's cancel, a tool switch, the reset before
    /// a load — is a moment the hypothesis stopped describing anything, and the
    /// next hover report re-seeds it.
    pub fn cancel_stroke(&mut self) {
        self.in_flight = None;
        self.selecting = None;
        self.hover = None;
    }

    // --- the hover mark (§18.1.10) --------------------------------------------

    /// Feed the hover mark one report: append it to the trailing window of
    /// recent reports, refit the window, and lay the **probe** — the stroke a
    /// drag begun this instant would open, `reach` canvas px from the cursor
    /// along the hover's extrapolated heading. The engine keeps the window, so
    /// the frontend sends one sample per move exactly as it does for a stroke.
    ///
    /// **The window is the estimator, not the mark** (§6.2). Reports are
    /// quantized to the device tolerance, so the heading of two adjacent reports
    /// snaps between the eight compass points — jitter in the *input*, which
    /// the fitter is precisely the machinery to price against detail, given
    /// the redundancy a window carries and a bare pair never did. But what is
    /// rendered is not the window's own trace — that is where the pointer
    /// *was*, which the screen already shows — it is the trace's heading
    /// carried forward from the cursor: a press is being predicted, and a
    /// press starts where the pointer is.
    ///
    /// The probe is **straight**, deliberately: continuing the trace's
    /// curvature would double down on the very quantity the tolerance makes
    /// noisiest, and "from here, this way" is the whole of what a press this
    /// instant can honestly be said to do. It is built through the fitter from
    /// two synthesized samples wearing the newest report's own channels, so it
    /// is bit-for-bit the record a real gesture of those two samples would
    /// commit — the prediction is synthesized; the rendering of it is not.
    ///
    /// `tolerance` is the frontend's statement of its input tolerance, as
    /// [`start_stroke`](Self::start_stroke) takes it — the window's own extent
    /// derives from it ([`WINDOW_ARC_TOLERANCES`]), since how much history the
    /// estimator needs is a fact about the tolerance. `reach` is how far the probe
    /// extends, in **canvas px by nature rather than by conversion**: the mark
    /// is a hypothesis about paint, paint is denominated on the canvas, and a
    /// screen-fixed length grew in canvas terms as the view zoomed out —
    /// promising more painting the less closely you looked.
    ///
    /// A non-finite report is refused at the door for
    /// [`stroke_to`](Self::stroke_to)'s reason — the window *remembers* — and
    /// a report within a tolerance of the last is dropped: it carries nothing the
    /// fit could use, and a resting pen's sub-tolerance drift would otherwise buy
    /// a whole-window refit per report. Hence the answer: whether anything
    /// changed, so a caller can skip the refold for a report that changed
    /// nothing.
    pub fn hover_to(&mut self, sample: InputSample, tolerance: f32, reach: f32) -> bool {
        if !sample.is_admissible() || !reach.is_finite() {
            return false;
        }
        if let Some(h) = self.hover.as_ref()
            && let Some(last) = h.window.last()
            && (last.pos == sample.pos || last.pos.distance(sample.pos) < tolerance)
        {
            return false;
        }
        let mut window = self.hover.take().map(|h| h.window).unwrap_or_default();
        window.push(sample);
        prune_window(&mut window, WINDOW_ARC_TOLERANCES * tolerance);
        // The estimator: the window's smoothed trace, wanted only for where it
        // is going. Finished, so the heading is the one a pen-up would have
        // settled on rather than the mid-solve one.
        let trace = fit_finished(tolerance, window.iter().copied());
        let path = match probe_heading(&trace) {
            Some(dir) => fit_finished(
                tolerance,
                [
                    sample,
                    InputSample {
                        pos: sample.pos + dir * reach,
                        ..sample
                    },
                ],
            ),
            // One report, or a trace too tangled to carry a heading: the honest
            // prediction of a press with no motion behind it is a click's — a
            // lone knot, which deposits nothing, exactly as the click would.
            None => fit_finished(tolerance, [sample]),
        };
        self.gesture_ordinal += 1;
        self.hover = Some(HoverStroke {
            path,
            window,
            ordinal: self.gesture_ordinal,
        });
        true
    }

    /// Drop the hover mark. Answers whether there was one to drop, so a caller
    /// can skip the refold — and the repaint — when there was not.
    pub fn clear_hover(&mut self) -> bool {
        self.hover.take().is_some()
    }

    /// The hover window, surrendered as a stroke's run-up (§6.2) — empty when
    /// there is none, or when it does not lead up to `press`: a window whose
    /// newest report sits farther from the press than the window's own scale
    /// (the same tolerance-derived arc that bounds it, [`WINDOW_ARC_TOLERANCES`]) is
    /// history from somewhere else — a teleported pointer, a trail gone stale —
    /// and evidence about nothing this stroke does.
    fn take_hover_context(&mut self, press: Vec2, tolerance: f32) -> Vec<InputSample> {
        let Some(h) = self.hover.take() else {
            return Vec::new();
        };
        let near = h
            .window
            .last()
            .is_some_and(|last| last.pos.distance(press) <= WINDOW_ARC_TOLERANCES * tolerance);
        if near { h.window } else { Vec::new() }
    }

    /// Whether a hover mark is held — what the frontend peeks before spending a
    /// command (and the frame behind it) taking one down.
    pub fn hover_held(&self) -> bool {
        self.hover.is_some()
    }

    /// The hover mark as the fold wants it, authored by `actor` with `seed` —
    /// the same [`GestureView`] shape a real gesture folds as, so the renderer
    /// cannot tell them apart, and the probe renders as exactly the pixels its
    /// gesture would commit: the prediction is synthesized, its rendering is
    /// inherited.
    ///
    /// `None` while a real gesture is in flight — a fact always outranks a
    /// hypothesis, and the fold may hold at most one gesture per actor — and
    /// under a selection tool, which drags a shape rather than the brush. An
    /// unpaintable active layer needs no test here: the stroke renderer refuses
    /// it exactly as a commit would.
    pub fn hover_view(&self, actor: ActorId, seed: u64) -> Option<GestureView> {
        if self.in_flight.is_some() || self.selecting.is_some() || self.tool.is_selection() {
            return None;
        }
        let h = self.hover.as_ref()?;
        Some(GestureView {
            actor,
            gesture: LiveGesture::Stroke(StrokeRecord {
                layer: self.active_layer,
                brush: self.brush,
                path: h.path.clone(),
                seed,
                // The probe is the whole prediction: two synthesized samples,
                // no run-up ahead of them, so the mark starts where its curve
                // does — exactly as the gesture it predicts would record it.
                start: 0.0,
            }),
            ordinal: h.ordinal,
            // Nothing settles: both samples move with every report, so the whole
            // mark is live tail — the same answer a marquee gives.
            frozen_spans: 0,
        })
    }
}

/// The hover mark's state (§18.1.10): the trailing window of recent reports —
/// the heading's estimator — the probe laid from it, and the bookkeeping that
/// keeps its renders distinct.
struct HoverStroke {
    /// The estimator, oldest first: every entry a report the hand really made.
    /// Never rendered — the mark is where a press would *go*, not where the
    /// pointer was — but it is what the prediction is honest about.
    window: Vec<InputSample>,
    /// The fitted probe — bit-for-bit the record a gesture of its two samples
    /// would commit, from the cursor along the extrapolated heading. Rebuilt
    /// per accepted report, kept so a fold costs a clone.
    path: Vec<ControlPoint>,
    /// This update's slot in the session's gesture-ordinal space (see
    /// [`Session::gesture_ordinal`]).
    ordinal: u64,
}

/// How much motion the estimator's window may span, in multiples of the input
/// **tolerance** — the report's own positional resolution ([`Session::hover_to`]).
///
/// Denominated on the tolerance because the tolerance is what the window exists to
/// average away: this many tolerances of recent motion hold the heading to roughly
/// `atan(1/this)` of noise, whatever the device and whatever the zoom — which
/// is also why it is the engine's to derive rather than the frontend's to
/// state, unlike the reach: nothing about it is a fact about the screen.
const WINDOW_ARC_TOLERANCES: f32 = 40.0;

/// Most reports the hover window keeps ([`Session::hover_to`]).
///
/// The tolerance-derived arc ([`WINDOW_ARC_TOLERANCES`]) is the extent; this is the
/// cost ceiling behind it. Every accepted report refits the whole window, so
/// this bounds what a report can cost — and it binds only where motion is
/// dense at the tolerance (a pen crawling pitch by pitch), where the window is
/// this many *tolerances* long and the heading has long since settled.
const HOVER_WINDOW: usize = 32;

/// Prune `window` from its old end until its arc fits `arc` and its count
/// fits [`HOVER_WINDOW`] — but never below the newest **pair**, which is the
/// least history a heading can be read from at all.
fn prune_window(window: &mut Vec<InputSample>, arc: f32) {
    let mut kept: f32 = window.windows(2).map(|w| w[0].pos.distance(w[1].pos)).sum();
    let mut from = 0;
    while window.len() - from > 2 && (window.len() - from > HOVER_WINDOW || kept > arc) {
        kept -= window[from].pos.distance(window[from + 1].pos);
        from += 1;
    }
    window.drain(..from);
}

/// Fit `samples` as a finished gesture — the path `End` would adopt, which is
/// what makes anything built from it the commit's own pixels.
fn fit_finished(
    tolerance: f32,
    samples: impl IntoIterator<Item = InputSample>,
) -> Vec<ControlPoint> {
    let mut fitter = PathFitter::with_tolerance(tolerance);
    for s in samples {
        fitter.push(s);
    }
    fitter.finish();
    fitter.path()
}

/// The heading at the cursor end of the fitted trace — the chord of its final
/// span, unit length — or `None` where the trace is too short to carry one (a
/// lone report) or the chord degenerates (a trace that folds back onto its own
/// end).
fn probe_heading(trace: &[ControlPoint]) -> Option<Vec2> {
    let spans = crate::path::span_count(trace.len());
    if spans < 2 {
        return None;
    }
    let d = crate::path::span_end(trace, spans - 1) - crate::path::span_end(trace, spans - 2);
    let len = d.length();
    (len.is_finite() && len > 1e-6).then(|| d / len)
}

impl StrokeBuilder {
    /// The stroke's path and start marker as they now stand: the fit's, or —
    /// once the gesture has snapped — the shape's (§6.9), whose marker is 0
    /// because a snapped path *is* the gesture, with no run-up in it.
    ///
    /// The one place the two are chosen between, which is what keeps the assist a
    /// *path transform between the fitter and the renderer* (§18.1.3) rather than a
    /// second kind of stroke. Everything above this reads the pair and cannot
    /// tell which it got.
    fn fitted(&self) -> (Vec<ControlPoint>, f32) {
        match self.assist.as_ref() {
            Some(a) => (a.path.clone(), 0.0),
            // The fitted control points **as `finish` would leave them** (§6.2): the
            // free window re-solves at pen-up, so rendering the mid-stroke window
            // would put the preview on a curve the commit is not on — sub-pixel, but
            // a step wherever a lookup is discontinuous in position (the tooth,
            // §6.4). As-finished, `End` adopts this very solve and live == committed
            // holds bit for bit — the marker included, since it comes from the
            // same solve's own arc profile. The last control point is pinned to
            // the newest sample either way, so the preview still reaches the
            // cursor.
            None => self.fitter.as_finished(),
        }
    }

    fn to_record(&self) -> StrokeRecord {
        let (path, start) = self.fitted();
        StrokeRecord {
            layer: self.layer,
            brush: self.brush,
            path,
            seed: self.seed,
            start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::ViewTransform;
    use stark_model::geom::Extent2;

    fn session(action: ShapeAction) -> Session {
        let mut s = Session::new(ViewTransform::identity(Extent2::new(256, 256)), LayerId(0));
        s.shape_action = action;
        s
    }

    /// The mode a marquee dragged across `(0,0)..(10,10)` resolves to.
    fn dragged(action: ShapeAction, has_selection: bool) -> SelectionMode {
        let mut s = session(action);
        s.start_selection(Tool::SelectRect, Vec2::ZERO, has_selection);
        s.selection_to(Vec2::splat(10.0));
        match s.preview_shape().expect("the marquee encloses something") {
            ShapeResult::Select(op) => op.mode(),
            ShapeResult::Fill(_) => panic!("a selecting action does not fill"),
        }
    }

    /// **Add, with nothing selected, is New** ([`against_selection`]) — and the
    /// gesture is logged as the New it behaved as, so nothing downstream carries the
    /// rule.
    ///
    /// Read off the *preview*, which is the same call the release commits
    /// ([`Session::end_shape`]): the resolution happens once, at the press, so the
    /// two cannot disagree about what the drag is doing.
    #[test]
    fn adding_to_nothing_selects_the_region() {
        let add = ShapeAction::Select(SelectionMode::Union);
        assert_eq!(dragged(add, false), SelectionMode::Replace);
        assert_eq!(
            dragged(add, true),
            SelectionMode::Union,
            "with a mask in force there is something to add to",
        );
    }

    /// The other three are already what the gesture reads as against an empty
    /// selection — the shape, its complement, and the shape again — so only Add is
    /// touched, and Fill is not a combining question at all.
    #[test]
    fn the_other_actions_are_unchanged_by_an_empty_selection() {
        for mode in [
            SelectionMode::Replace,
            SelectionMode::Subtract,
            SelectionMode::Intersect,
        ] {
            assert_eq!(dragged(ShapeAction::Select(mode), false), mode);
        }
        let mut s = session(ShapeAction::Fill);
        s.start_selection(Tool::SelectRect, Vec2::ZERO, false);
        s.selection_to(Vec2::splat(10.0));
        assert!(matches!(s.preview_shape(), Some(ShapeResult::Fill(_))));
    }
}
