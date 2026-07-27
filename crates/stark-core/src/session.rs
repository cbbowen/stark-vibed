//! Session: ephemeral, non-historized state (DESIGN.md §3).
//!
//! The session holds the current tool/brush, the pan/zoom view, and the
//! in-flight stroke being dragged out. None of this is undoable — switching
//! tools or panning never creates a history step. The session interprets
//! pointer commands and, on `EndStroke`, hands the [`Engine`](crate::Engine) a
//! finished [`StrokeRecord`] to commit.

use crate::command::InputSample;
use crate::document::{
    ActorId, BrushDynamics, BrushParams, ColorDynamics, LayerId, NoiseKind, SelectionMode,
    SelectionOp, SelectionShape, StrokeRecord, Tool,
};
use crate::geom::{Vec2, ViewTransform};
use crate::path::PathFitter;
use crate::peer::{
    GestureView, HEARTBEAT, Identity, LiveGesture, PeerFrame, StrokeHead, default_name,
};
use crate::presence::{GestureSource, GestureTx};

/// Minimum spacing (canvas px) between lasso vertices. The mask shader costs one
/// segment test per texel per vertex, and pointer samples arrive far denser than a
/// mask boundary can resolve, so the polyline is thinned as it is collected —
/// bounding both the rasterization cost and the size of the logged op (DESIGN §6.8).
const LASSO_MIN_STEP: f32 = 2.0;

/// Accumulates the stroke currently being drawn.
///
/// Pointer samples are fitted to control points *as they arrive* rather than
/// buffered and re-fitted on every move (DESIGN.md §6.2): the builder holds only
/// the fitter's short working window, and each new sample costs work proportional
/// to that window instead of to the stroke so far.
struct StrokeBuilder {
    tool: Tool,
    brush: BrushParams,
    layer: LayerId,
    seed: u64,
    fitter: PathFitter,
}

/// The selection gesture currently being dragged out (DESIGN.md §6.8). Like a
/// stroke it is ephemeral: only the [`SelectionOp`] it resolves to on release is
/// committed, and the shape is derived from the drag on demand so a live preview and
/// the committed op come from exactly the same code.
struct SelectionDrag {
    tool: Tool,
    mode: SelectionMode,
    feather: f32,
    /// Where the drag started; for the marquees this is one corner of the box.
    start: Vec2,
    /// The lasso's decimated outline (empty for the marquees).
    points: Vec<Vec2>,
    /// The newest sample, so the marquees can span `start`..`current`.
    current: Vec2,
}

impl SelectionDrag {
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

    /// The op this drag currently stands for — `None` for a gesture that encloses
    /// nothing (a click with a marquee, a lasso too short to have an interior).
    fn to_op(&self) -> Option<SelectionOp> {
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
        Some(SelectionOp::new(self.mode, shape, self.feather))
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
    pub active_layer: LayerId,
    /// How the next selection gesture combines with the selection in force (§6.8).
    pub selection_mode: SelectionMode,
    /// Edge softness (canvas px) applied by the next selection gesture.
    pub selection_feather: f32,
    /// Whether collaborators' selection outlines are drawn (PEER_DESIGN.md §3).
    ///
    /// View state, so each client decides for itself and nothing about it is logged
    /// or sent. Off by default: knowing which region someone else is working inside
    /// is occasionally useful, but a second contour over the artwork is a cost paid
    /// on every frame you look at it, and most of the time the answer to "what is
    /// that line?" should be "the one I drew".
    pub show_peer_selections: bool,

    // --- the published half (PEER_DESIGN.md §4.1) -------------------------
    //
    // Everything above this line is *private* view state — pan/zoom, brush, the
    // selection mode. Everything below it, plus `active_layer` and the in-flight
    // gesture, is what [`Session::publish`] projects for other clients. The split is
    // recorded by what that method returns rather than described in a comment, which
    // is the same discipline DESIGN §4 applies to the command classes.
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
    selecting: Option<SelectionDrag>,
    /// Bumped on every gesture start, so a peer can tell a restart from a
    /// continuation without a clock.
    gesture_ordinal: u64,

    // Publish bookkeeping — the latch, not a queue (PEER_DESIGN.md §5.1).
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

fn hard_round_brush_params() -> BrushParams {
    BrushParams {
        radius: 100.0,
        hardness: 0.95,
        dynamics: BrushDynamics {
            add: 0.4,
            lift: 0.6,
            deposit: 0.95,
            ..BrushDynamics::default()
        },
        color_dynamics: ColorDynamics {
            noise: NoiseKind::Simplex,
            frequency: [1.0, 4.0],
            amplitude: [0.0, 0.05, 0.1],
        },
        ..BrushParams::default()
    }
}

impl Session {
    pub fn new(view: ViewTransform, active_layer: LayerId) -> Self {
        Self {
            view,
            tool: Tool::Brush,
            brush: hard_round_brush_params(),
            active_layer,
            selection_mode: SelectionMode::default(),
            selection_feather: 0.0,
            show_peer_selections: false,
            name: String::new(),
            name_chosen: false,
            cursor: None,
            in_flight: None,
            selecting: None,
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
    /// changed since the last call — otherwise `None` (PEER_DESIGN.md §5.1).
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
    /// roster at once, rather than making them wait out [`PEER_TIMEOUT`](crate::peer::PEER_TIMEOUT).
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
    /// resent every frame because it can still move (DESIGN §6.2). That is the same
    /// partition the renderer's `FrozenHead` uses, spent on the wire instead of on
    /// the GPU.
    fn gesture_source(&self) -> Option<GestureSource> {
        match self.in_flight.as_ref() {
            Some(b) => Some(GestureSource::Stroke {
                head: StrokeHead {
                    layer: b.layer,
                    tool: b.tool,
                    brush: b.brush,
                    seed: b.seed,
                },
                path: b.fitter.path(),
                frozen: b.fitter.frozen_points(),
            }),
            None => self.preview_selection().map(GestureSource::Selection),
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

    /// Whether a selection gesture is being dragged out (DESIGN.md §6.8).
    pub fn is_selecting(&self) -> bool {
        self.selecting.is_some()
    }

    /// Begin a selection gesture with the session's current mode and feather. Any
    /// in-flight stroke or earlier gesture is abandoned.
    pub fn start_selection(&mut self, tool: Tool, pos: Vec2) {
        self.tool = tool;
        self.in_flight = None;
        self.gesture_ordinal += 1;
        self.selecting = Some(SelectionDrag {
            tool,
            mode: self.selection_mode,
            feather: self.selection_feather,
            start: pos,
            points: vec![pos],
            current: pos,
        });
    }

    /// Extend the in-flight selection gesture.
    pub fn selection_to(&mut self, pos: Vec2) {
        if let Some(drag) = self.selecting.as_mut() {
            drag.push(pos);
        }
    }

    /// The op the in-flight gesture currently stands for, for live preview — the very
    /// same call [`Self::end_selection`] commits, so preview == committed.
    pub fn preview_selection(&self) -> Option<SelectionOp> {
        self.selecting.as_ref().and_then(SelectionDrag::to_op)
    }

    /// Finish the gesture, returning the op to commit (`None` if it encloses nothing).
    ///
    /// The selection tools are **momentary**: having drawn a selection, the session
    /// hands the canvas straight back to the brush. Selecting is a step *towards*
    /// painting, essentially never something you do twice in a row, so making it
    /// modal costs a deliberate switch-back on the overwhelmingly common path — and
    /// leaves a brush gesture silently redefining the selection when the user forgets.
    /// A gesture that enclosed nothing (a stray click) is not a selection and leaves
    /// the tool armed, so a mis-click doesn't disarm it.
    pub fn end_selection(&mut self) -> Option<SelectionOp> {
        let op = self.selecting.take().and_then(|d| d.to_op());
        if op.is_some() {
            self.tool = Tool::Brush;
        }
        op
    }

    /// Discard the in-flight selection gesture without committing.
    pub fn cancel_selection(&mut self) {
        self.selecting = None;
    }

    /// Begin a stroke. `seed` is supplied by the engine so it can be derived
    /// deterministically (DESIGN.md §6.2). `tolerance` is what the frontend says its
    /// input resolves to, in canvas px ([`PathFitter::with_tolerance`]). Replaces any
    /// abandoned in-flight one.
    pub fn start_stroke(&mut self, tool: Tool, sample: InputSample, seed: u64, tolerance: f32) {
        self.tool = tool;
        self.selecting = None;
        self.gesture_ordinal += 1;
        let mut fitter = PathFitter::with_tolerance(tolerance);
        fitter.push(sample);
        self.in_flight = Some(StrokeBuilder {
            tool,
            brush: self.brush,
            layer: self.active_layer,
            seed,
            fitter,
        });
    }

    /// Extend the in-flight stroke with another sample.
    pub fn stroke_to(&mut self, sample: InputSample) {
        if let Some(b) = self.in_flight.as_mut() {
            b.fitter.push(sample);
        }
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
            None => (LiveGesture::Selection(self.preview_selection()?), 0),
        };
        Some(GestureView {
            actor,
            gesture,
            ordinal: self.gesture_ordinal,
            frozen_spans,
        })
    }

    /// Snapshot the in-flight stroke as a record without ending it, for live
    /// preview (DESIGN.md §6.2). `None` if no stroke is active.
    pub fn preview_record(&self) -> Option<StrokeRecord> {
        self.in_flight.as_ref().map(StrokeBuilder::to_record)
    }

    /// How many spans of the in-flight stroke are settled — the prefix a live
    /// preview could render once instead of repainting per pointer move
    /// (see [`PathFitter::frozen_spans`]). 0 when no stroke is active.
    pub fn frozen_spans(&self) -> usize {
        self.in_flight
            .as_ref()
            .map_or(0, |b| b.fitter.frozen_spans())
    }

    /// Finish the stroke, returning the record to commit (`None` if empty).
    pub fn end_stroke(&mut self) -> Option<StrokeRecord> {
        self.in_flight.take().map(|mut b| {
            b.fitter.finish();
            b.to_record()
        })
    }

    /// Discard the in-flight stroke without committing.
    pub fn cancel_stroke(&mut self) {
        self.in_flight = None;
        self.selecting = None;
    }
}

impl StrokeBuilder {
    fn to_record(&self) -> StrokeRecord {
        StrokeRecord {
            layer: self.layer,
            tool: self.tool,
            brush: self.brush,
            // The fitted control points (DESIGN.md §6.2). Mid-stroke this ends in
            // a provisional knot at the newest sample, so the preview reaches the
            // cursor; the same fitter produces the committed path, so live ==
            // committed.
            path: self.fitter.path(),
            seed: self.seed,
        }
    }
}
