//! Session: ephemeral, non-historized state (§3).
//!
//! The current tool and brush, what they are aimed at, and the gesture being dragged
//! out. None of it is undoable — switching tools never creates a history step. The
//! session interprets pointer commands and, on [`end_gesture`](Session::end_gesture),
//! hands the [`Engine`](crate::Engine) what the release commits.
//!
//! **The hand, not the eye**: the pan/zoom view is per-client state too, but nothing
//! here reads it, so it is the engine's field and [`crate::view`]'s type.
//!
//! Two of its subjects have a file of their own: the shape gesture ([`shape`]) and
//! the hover mark ([`hover`]). **Nothing here is about the wire**: what a gesture
//! puts on it is `presence`'s, and all the session owes that side is
//! [`live`](Session::live).

#[cfg(test)]
mod fixtures;
mod hover;
mod shape;

pub use shape::Outline;

use crate::assist::{AssistShape, PenProfile};
use crate::command::InputSample;
use crate::command::Tool;
use crate::gesture::{GestureView, LiveGesture};
use crate::path::{InputTolerance, PathFitter, frozen_spans_for};
use crate::tow::{Tow, TowString};
use stark_model::document::Scaffold;
use stark_model::document::{ActorId, BrushParams, GuideId, LayerId, ShapeAction, StrokeRecord};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;
use stark_model::sanitize::{at_least_zero, clamp01};

use hover::HoverStroke;
use shape::ShapeDrag;

/// What the hand is doing — **at most one thing at a time**, which is the whole of why
/// this is an enum (§3).
///
/// The four states were three `Option` fields, kept mutually exclusive by hand at every
/// door: a press nulled the other two, the hover door refused unless both were empty,
/// and a cancel nulled all three. Those nullings are now the assignment.
///
/// **There is one of these**, the field of the one [`Session`] the engine owns, so the
/// enum's width is a fixed cost in that struct rather than a per-value one — and the
/// boxing clippy asks for would buy it back with an allocation at every press and an
/// indirection on the fitter's per-report push. [`Trace`] answers the lint the same
/// way, for the same reason.
#[expect(
    clippy::large_enum_variant,
    reason = "one value exists, and its widest arm is the hot one"
)]
#[derive(Default)]
enum Hand {
    #[default]
    Idle,
    /// The hover mark (§18.1.10): the stroke a drag begun this instant would open.
    /// It never commits, never publishes and never sets
    /// [`is_stroking`](Session::is_stroking) — a real gesture outranks it, which this
    /// enum states rather than each reader re-deriving.
    ///
    /// **Only ever held with a tool that paints.** [`Session::hover_to`] refuses the
    /// report otherwise, and every door that arms a selection tool
    /// ([`Session::set_tool`], [`Session::start_selection`]) empties the hand first.
    Hovering(HoverStroke),
    Stroking(StrokeBuilder),
    Dragging(ShapeDrag),
}

impl Hand {
    /// Take the hover mark, leaving any other hand exactly where it was — `Option::take`
    /// on the slot this enum replaced.
    fn take_hover(&mut self) -> Option<HoverStroke> {
        match std::mem::take(self) {
            Hand::Hovering(h) => Some(h),
            other => {
                *self = other;
                None
            }
        }
    }
}

/// Accumulates the stroke currently being drawn.
///
/// Pointer samples are fitted to control points *as they arrive* (§6.2), so a new
/// sample costs work proportional to the fitter's short working window rather than to
/// the stroke so far.
struct StrokeBuilder {
    brush: BrushParams,
    layer: LayerId,
    seed: u64,
    /// The target layer's translation at the press (§14.12). Everything in the
    /// builder — fitter, tow, assist — works in canvas space, where the hand is;
    /// [`to_record`](Self::to_record) is the one door out and the one conversion, so
    /// the commit, the live fold and the wire cannot disagree about the frame.
    translation: stark_model::geom::IVec2,
    /// What the frontend said this gesture's input resolves to. Kept because the
    /// drawing assist prices its recognition in the same unit the fit does — see
    /// [`crate::assist`].
    tolerance: InputTolerance,
    trace: Trace,
}

/// Where the stroke's path comes from — fitted from the reports, or, once the gesture
/// has been **held**, the ideal shape it snapped to (§6.9).
///
/// A snap **drops** the fitter and the tow rather than leaving them running beside a
/// path nothing reads them for: from there the shape decides the path, and its steering
/// reads the raw pointer.
#[expect(clippy::large_enum_variant, reason = "see `Hand`")]
enum Trace {
    Fitting {
        fitter: PathFitter,
        /// The towed tip (§6.11) — `Some` when the gesture started with a rope.
        /// Feeds **the fitter only**: it sits between the raw samples and
        /// [`PathFitter::push`].
        tow: Option<Tow>,
    },
    Snapped(Assist),
}

/// A stroke that has snapped to an ideal shape, and is now being steered rather than
/// extended (§6.9).
///
/// **Every adjustment is `base.adjust(grip, pointer)`, never `shape.adjust(..)`**, so
/// a minute of dragging lands the same as the same displacement made at once. `path`
/// caches the realization of `shape`, which only a pointer move can change.
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
    /// The string's standing offset at the snap, tip − pointer (§6.11); zero for an
    /// untowed gesture. Added to every steer, so the gap between the towed trace the
    /// shape was recognized from and the raw pointer — up to a rope beyond it — does
    /// not land as a jump on the first move, and the hand's deltas land 1:1.
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

pub struct Session {
    tool: Tool,
    brush: BrushParams,
    /// The **hand's** color, straight sRGB. Its own field rather than a reading of
    /// the brush, which does not always have one: an erasing brush carries no pigment,
    /// and a fill drawn while one is held still lays the color in hand
    /// ([`start_selection`](Self::start_selection)). While the brush paints, the
    /// frontend keeps this equal to the paint effect's own color.
    color: [f32; 3],
    /// Published to peers (§17.4) beside the [`PresenceTx`] half, and read on this
    /// side as well, so it is the session's own rather than the publisher's: the
    /// wire is *told* it at publish time.
    ///
    /// [`PresenceTx`]: crate::presence::PresenceTx
    active_layer: LayerId,
    /// What the next shape gesture does with the region it encloses (§6.8,
    /// §18.0.4) — one of the four ways to combine it into the
    /// selection, or fill it.
    pub shape_action: ShapeAction,
    /// Edge softness (canvas px) applied by the next shape gesture, whichever
    /// action it takes: a feathered fill and a feathered selection are the same
    /// ramp, rasterized by the same shader.
    selection_feather: f32,
    /// How strongly the next **fill** gesture's parcel lands, `0..=1` — the Select
    /// panel's Opacity slider under the Fill action, and [`FillOp::opacity`].
    ///
    /// The four *selecting* actions ask the same question of the mask instead, and
    /// that answer is document state rather than session state, because it reaches a
    /// region already drawn (`ActionKind::SetSelectionOpacity`, §6.8).
    ///
    /// [`FillOp::opacity`]: stark_model::document::FillOp::opacity
    shape_opacity: f32,
    /// The drawing guides whose **eye this client has opened** (§20.5).
    ///
    /// The one per-client half of a guide: the guides themselves are document state
    /// (`DocState::guides`), but whether *you* are looking at one is not, so this sits
    /// beside the pan and the zoom, never logged and never sent.
    ///
    /// **Opened rather than shut, so an absent entry means not drawn.** A guide
    /// arriving from a file or from a peer is hidden until somebody on this client
    /// asks for it, rather than laying every perspective anyone ever built over the
    /// canvas at once. It is also the cheaper end: a client typically has one guide up
    /// out of however many the document carries.
    ///
    /// A guide removed from the document leaves its id here, because the removal may
    /// be undone and an eye that shut itself would be the tool changing what you are
    /// looking at.
    visible_guides: std::collections::HashSet<GuideId>,

    hand: Hand,
    /// Bumped on every gesture start, so a peer can tell a restart from a continuation
    /// without a clock, and on every hover update, so the renderer's per-actor head
    /// cache cannot mistake one hover's render for the next's. **One counter for both
    /// on purpose**: two could collide, and a collision revives a cached head —
    /// including one still holding a cancelled stroke's paint.
    gesture_ordinal: u64,
}

impl Session {
    pub fn new(active_layer: LayerId) -> Self {
        Self {
            tool: Tool::Brush,
            brush: BrushParams::default(),
            color: [0.0, 0.0, 0.0],
            active_layer,
            shape_action: ShapeAction::default(),
            selection_feather: 0.0,
            shape_opacity: 1.0,
            visible_guides: std::collections::HashSet::new(),
            hand: Hand::Idle,
            gesture_ordinal: 0,
        }
    }

    // --- the hand: tool, brush, color, target -----------------------------

    /// Point the hand at `tool`. Switching away mid-gesture **abandons** it rather
    /// than committing a half-dragged marquee — the gate `ViewCommand::SetTool`'s
    /// dispatch arm applies, carried onto the thing it is about, for the reason
    /// [`set_selection_feather`](Self::set_selection_feather) gives.
    pub fn tool(&self) -> Tool {
        self.tool
    }

    pub fn brush(&self) -> BrushParams {
        self.brush
    }

    pub fn color(&self) -> [f32; 3] {
        self.color
    }

    pub fn active_layer(&self) -> LayerId {
        self.active_layer
    }

    pub(crate) fn set_tool(&mut self, tool: Tool) {
        self.cancel_stroke();
        self.tool = tool;
    }

    /// Take a brush, **sanitized**, for the reason `PeerFrame::sanitized` holds a
    /// peer's: a committed stroke's brush is held by `ActionKind::sanitized`, and a
    /// live one is drawn by the same renderer without ever becoming an action, so
    /// nothing else would. `preview == committed` needs both doors (§6.2).
    pub fn set_brush(&mut self, brush: BrushParams) {
        self.brush = brush.sanitized();
    }

    /// Take the hand's color. No gate: `ViewCommand::SetBrush`'s arm applies none,
    /// and the value is straight sRGB that only ever becomes paint through an op's
    /// own constructor ([`start_selection`](Self::start_selection)).
    pub fn set_color(&mut self, color: [f32; 3]) {
        self.color = color;
    }

    /// Arm `id` as the active layer if the document has it — **any** layer, a matte
    /// included: the active layer is *the selected layer*, not a paint target, which
    /// is what lets the frontend have one selection concept instead of two (§15.7). A
    /// stroke aimed at a matte draws nothing, refused identically by `apply` and by
    /// the preview path. Answers whether it landed.
    ///
    /// Takes the document because a session holds none.
    pub(crate) fn set_active_layer(
        &mut self,
        id: LayerId,
        doc: &crate::document::DocState,
    ) -> bool {
        let known = doc.contains_layer(id);
        if known {
            self.active_layer = id;
        }
        known
    }

    /// Edge softness (canvas px) for the next shape gesture, floored at zero.
    ///
    /// A setter rather than a `pub` field because the value arrives in a command from
    /// outside and this is where the gate holds — the model's own, so what the panel
    /// shows is what [`SelectionOp::at`] and [`FillOp::with_paint`] will use.
    ///
    /// [`SelectionOp::at`]: stark_model::document::SelectionOp::at
    /// [`FillOp::with_paint`]: stark_model::document::FillOp::with_paint
    pub(crate) fn set_selection_feather(&mut self, feather: f32) {
        self.selection_feather = at_least_zero(feather, 0.0);
    }

    /// Edge softness the next shape gesture will use.
    pub fn selection_feather(&self) -> f32 {
        self.selection_feather
    }

    /// How strongly the next fill gesture's parcel lands, clamped to `0..=1`.
    pub(crate) fn set_shape_opacity(&mut self, opacity: f32) {
        self.shape_opacity = clamp01(opacity);
    }

    /// How strongly the next fill gesture's parcel lands.
    pub fn shape_opacity(&self) -> f32 {
        self.shape_opacity
    }

    // --- the guide roster's per-client half (§20.5) --------------------------

    /// Whether this client draws the guide `id`. A guide nobody here has opened is
    /// **not** drawn, which is what an absent entry in [`visible_guides`] means (§20.5).
    ///
    /// [`visible_guides`]: Self::visible_guides
    pub(crate) fn guide_visible(&self, id: GuideId) -> bool {
        self.visible_guides.contains(&id)
    }

    /// Open or shut a guide's eye. Answers whether that changed anything, so a
    /// command that says what is already true costs no repaint and no
    /// observation.
    pub(crate) fn set_guide_visible(&mut self, id: GuideId, visible: bool) -> bool {
        if visible {
            self.visible_guides.insert(id)
        } else {
            self.visible_guides.remove(&id)
        }
    }

    /// The guides this client actually draws, in roster order: the document's own
    /// list, kept down to the ones whose eye this client has opened.
    ///
    /// **The one place the two halves of a guide meet** (§20.5) — the overlay pass,
    /// the stroke assist's [`Scaffold`] and the panel's rows all ask through here.
    ///
    /// Takes the document because a `Session` holds none, and which one it is combined
    /// with is the caller's: the committed document for a commit, the previewed one
    /// for what is on screen.
    pub(crate) fn shown_guides<'a>(
        &'a self,
        doc: &'a crate::document::DocState,
    ) -> impl Iterator<Item = &'a crate::document::Guide> {
        doc.guides()
            .iter()
            .filter(|g| self.visible_guides.contains(&g.id))
    }

    // --- the gesture in flight (§6.2, §6.8) -------------------------------------

    /// Whether a **gesture** is being dragged out — a stroke or a shape drag, and not
    /// a hover, which commits nothing and publishes nothing.
    ///
    /// The whole of what the publish latch (§17.5) asks of the hand: a live gesture's
    /// path grows every move, so its frame always differs from the last one sent.
    pub(crate) fn is_gesturing(&self) -> bool {
        matches!(self.hand, Hand::Stroking(_) | Hand::Dragging(_))
    }

    /// **What this client is dragging out, and how much of it is settled** — the one
    /// thing the session says about a gesture (§17.4). The wire reads it
    /// (`PresenceTx::publish`), the preview fold reads it
    /// ([`live_view`](Self::live_view)), and the release commits it
    /// ([`end_gesture`](Self::end_gesture)), so none of the three can disagree about
    /// what is being drawn.
    ///
    /// The count is of **frozen control points** (§6.2): leading points the fitter
    /// has stopped revising, which is what lets the wire send a delta and the fold
    /// keep a rendered head. Zero for everything else — a region's closing edge
    /// follows the cursor, so no part of it has settled.
    ///
    /// `None` when nothing is being dragged out: an idle hand, a shape gesture that
    /// so far encloses nothing, and a hand that is only hovering, since a hypothesis
    /// commits nothing and publishes nothing (§18.1.10).
    pub(crate) fn live(&self) -> Option<(LiveGesture, usize)> {
        match &self.hand {
            Hand::Stroking(b) => Some((LiveGesture::Stroke(b.to_record()), b.frozen_points())),
            Hand::Dragging(d) => Some((d.to_gesture()?, 0)),
            Hand::Idle | Hand::Hovering(_) => None,
        }
    }

    pub fn is_stroking(&self) -> bool {
        self.stroke().is_some()
    }

    /// The stroke in flight, if that is what the hand is making.
    fn stroke(&self) -> Option<&StrokeBuilder> {
        match &self.hand {
            Hand::Stroking(b) => Some(b),
            Hand::Idle | Hand::Hovering(_) | Hand::Dragging(_) => None,
        }
    }

    /// The ordinal of the gesture in flight — bumped on every start, so a cached
    /// render of one stroke is never mistaken for a render of the next.
    pub fn gesture_ordinal(&self) -> u64 {
        self.gesture_ordinal
    }

    /// Begin a brush stroke, replacing whatever the hand was doing.
    ///
    /// `seed` is the engine's, so it can be derived deterministically (§6.2).
    /// `tolerance` is what the frontend says its input resolves to, in canvas px
    /// ([`PathFitter::with_tolerance`]), held to its range here (`InputTolerance`) and
    /// nowhere else. `rope` is the stroke-smoothing string length,
    /// also canvas px (§6.11); `0` or anything unusable constructs no tow and feeds
    /// the fitter the raw samples. `translation` is the active layer's translation at
    /// the press (§14.12), pinned here so a translate landing mid-stroke cannot shear
    /// the record against its own gating.
    pub fn start_stroke(
        &mut self,
        sample: InputSample,
        seed: u64,
        tolerance: f32,
        rope: f32,
        translation: stark_model::geom::IVec2,
    ) {
        let tolerance = InputTolerance::new(tolerance);
        self.tool = Tool::Brush;
        // The hover window becomes the stroke's **run-up** (§6.2,
        // `PathFitter::seed_runup`): real leading samples the fitted curve extends
        // back through, with the record's marker saying where the stroke itself
        // begins, so the entry's direction and curvature come from watched motion
        // rather than from the first tolerance-quantized steps. Taking the window
        // here also retires the hover, which would otherwise resurrect a pre-press
        // mark at pen-up (§18.1.10).
        let runup = self.take_hover_context(sample.pos, tolerance);
        self.gesture_ordinal += 1;
        let mut fitter = PathFitter::with_tolerance(tolerance);
        fitter.seed_runup(&runup);
        fitter.push(sample);
        self.hand = Hand::Stroking(StrokeBuilder {
            brush: self.brush,
            layer: self.active_layer,
            seed,
            translation,
            tolerance,
            trace: Trace::Fitting {
                fitter,
                // A non-finite first report anchors nothing — the tip would park on it
                // and no later report could pull the string off — so the gesture runs
                // untowed rather than towed from nowhere.
                tow: (rope.is_finite() && rope > 0.0 && sample.is_admissible())
                    .then(|| Tow::new(rope, tolerance, sample)),
            },
        });
    }

    /// Extend the gesture in flight with another pointer report — one entry point for
    /// every kind, so the frontend sends the same `To` for a stroke, for a stroke that
    /// has snapped and is now being **steered** (§6.9), and for a shape drag.
    ///
    /// A hand that is idle or merely hovering takes nothing from it: a hover is fed
    /// through its own door ([`hover_to`](Self::hover_to)), which is not free.
    ///
    /// **The two live halves gate the report differently, because they remember
    /// different amounts of it.** A stroke keeps the report's channels across reports
    /// — [`Tow`] holds the tip and the target, and [`AssistShape::adjust`] folds the
    /// position into a shape that outlives it — so one unusable number would park the
    /// gesture where no later report could pull it back from, and the whole sample has
    /// to be admissible. A shape drag reads only the position, and the marquees would
    /// carry a bad corner only until the next move; the lasso is what makes even that
    /// matter, since it keeps every point it is given and one would sit in the logged
    /// polyline for the rest of the gesture.
    pub fn gesture_to(&mut self, sample: InputSample) {
        match &mut self.hand {
            Hand::Stroking(b) => {
                if sample.is_admissible() {
                    b.extend(sample);
                }
            }
            Hand::Dragging(d) => {
                if sample.pos.is_finite() {
                    d.push(sample.pos);
                }
            }
            Hand::Idle | Hand::Hovering(_) => {}
        }
    }

    /// Snap the stroke in flight to the shape it resembles — what a **hold** means
    /// (§6.9). Returns whether anything snapped.
    ///
    /// Declining is a normal outcome, not a failure: a stroke that is neither a line
    /// nor an ellipse is left exactly as it was drawn, and the gesture carries on
    /// through the fitter.
    ///
    /// A snap **bumps the gesture ordinal**, because the path is replaced rather than
    /// extended. That one increment is what invalidates the renderer's cached head
    /// (§6.2) and makes peers restart their assembly (§17.5) instead of splicing a
    /// delta onto a path that no longer exists.
    ///
    /// A recognized shape is offered whatever perspective guides are on the screen: a
    /// line takes an axis it is aimed near (§20.6), a loop becomes a circle on a plane
    /// it nearly lies on (§20.7). The guides are read, never touched, and a document
    /// with none up puts up an empty scaffold.
    ///
    /// `guides` arrives from the caller because only `Engine` holds both halves of a
    /// guide (§20.5): the roster is `DocState`'s, the open eye is this session's —
    /// [`Session::shown_guides`] is the filter, `Engine::scaffold` applies it.
    pub fn assist_stroke(&mut self, guides: &Scaffold) -> bool {
        let Hand::Stroking(b) = &mut self.hand else {
            return false;
        };
        if !b.snap(guides) {
            return false;
        }
        self.gesture_ordinal += 1;
        true
    }

    /// Which shape the stroke in flight has snapped to (§6.9), or `None` where
    /// there is no stroke or the hold found nothing to snap it to.
    ///
    /// The family rather than the shape (`assist::Assisted`): all that is asked from
    /// outside the crate is whether a hold landed, and whether it landed on a line.
    pub fn assisted(&self) -> Option<crate::assist::Assisted> {
        self.stroke().and_then(StrokeBuilder::assisted)
    }

    /// [`Session::live`](Self::live) as the preview fold wants it, authored by
    /// `actor` — the same [`GestureView`] `Peer::gesture_view` produces for everyone
    /// else, so the fold treats them alike without either being stored in the
    /// other's form.
    ///
    /// A hand that is only *hovering* offers the §18.1.10 mark instead: the stroke its
    /// recent reports would have committed, in the same shape, so the fold cannot tell
    /// it from a real gesture and `hover == committed` is inherited rather than
    /// maintained. One value out, because an actor folds at most one gesture and a
    /// fact outranks a hypothesis — which `Hand` now says rather than a `match` at
    /// the fold.
    ///
    /// `seed` and `translation` are the hover's alone: a probe has no press to pin
    /// them at.
    pub fn live_view(
        &self,
        actor: ActorId,
        seed: u64,
        translation: stark_model::geom::IVec2,
    ) -> Option<GestureView> {
        let (gesture, frozen) = match self.live() {
            Some(live) => live,
            None => (LiveGesture::Stroke(self.hover_probe(seed, translation)?), 0),
        };
        Some(GestureView {
            actor,
            // The same count the wire carries, converted to spans the one way a
            // receiver converts it (`GestureRx::frozen_spans`) — so what this client
            // retires from a repaint and what a peer retires are one number (§17.6).
            frozen_spans: match &gesture {
                LiveGesture::Stroke(rec) => frozen_spans_for(frozen, rec.path.len()),
                LiveGesture::Selection(_) | LiveGesture::Fill { .. } => 0,
            },
            gesture,
            ordinal: self.gesture_ordinal,
        })
    }

    /// Snapshot the in-flight stroke as a record without ending it, for live
    /// preview (§6.2). `None` if no stroke is active.
    pub fn preview_record(&self) -> Option<StrokeRecord> {
        self.stroke().map(StrokeBuilder::to_record)
    }

    /// Finish the gesture in flight, returning what to commit — and nothing at all if
    /// the hand was idle or only hovering, which a release leaves exactly as it was.
    ///
    /// A stroke answers `None` if it is empty, and `None` for a record that cannot
    /// deposit, since a step in the log that changes no pixel would spend an undo step
    /// invisibly (a marquee click is answered the same way). The travel that counts
    /// starts at the record's marker, so a click at the end of a watched approach is
    /// still a click: its curve holds the whole run-up, and its marker sits at the end
    /// of it ([`PathFitter::painted`]).
    ///
    /// A record reaching too far to be drawn at all is one that cannot deposit: every
    /// render refuses it whole, so logging it would also hand every peer a record to
    /// refuse again. Asked of the record as the log will hold it, which is also the
    /// record returned (`gpu::stroke::loggable`).
    ///
    /// **The tool disarms when the gesture was a step towards painting, and stays
    /// armed when the gesture was painting.** So a selecting action is momentary —
    /// the canvas goes straight back to the brush, since selecting is essentially
    /// never done twice in a row and a modal selection tool the user forgot about
    /// means a brush gesture silently redefining the selection. A fill stays armed,
    /// because blocking in *is* painting and is done many times in a row. A gesture
    /// that enclosed nothing (a stray click) leaves the tool armed either way, so a
    /// mis-click doesn't disarm it.
    pub fn end_gesture(&mut self) -> Option<LiveGesture> {
        match std::mem::take(&mut self.hand) {
            Hand::Stroking(b) => b.finish().map(LiveGesture::Stroke),
            Hand::Dragging(d) => {
                // Through the very call the preview and the wire read (`Session::live`),
                // so preview == committed (§1.3).
                let committed = d.to_gesture();
                if matches!(committed, Some(LiveGesture::Selection(_))) {
                    self.tool = Tool::Brush;
                }
                committed
            }
            // A hover is not a gesture: it commits nothing, so a release with nothing
            // in flight must leave the mark where it was.
            hand @ (Hand::Idle | Hand::Hovering(_)) => {
                self.hand = hand;
                None
            }
        }
    }

    /// The in-flight tow, for the frontend's string overlay (§6.11): `None`
    /// when no stroke is active, the gesture carries no rope, or it has
    /// snapped to a shape.
    pub fn tow_string(&self) -> Option<TowString> {
        match &self.stroke()?.trace {
            Trace::Fitting { tow, .. } => tow.as_ref().map(Tow::string),
            // A steered shape is driven by the raw pointer, so there is no string to
            // show — and, since the snap dropped it, none left to show one with.
            Trace::Snapped(_) => None,
        }
    }

    /// Empty the hand without committing. Takes the hover mark with it: every caller
    /// — a pinch's cancel, a tool switch, the reset before a load — is a moment the
    /// hypothesis stopped describing anything, and the next hover report re-seeds it.
    pub(crate) fn cancel_stroke(&mut self) {
        self.hand = Hand::Idle;
    }

    /// Forget everything this session held *about the document that just left*.
    ///
    /// **The one door per-document session state goes out through.** A session
    /// outlives the document it was looking at, so every field keyed on something the
    /// document mints belongs here rather than in a list the engine keeps.
    ///
    /// [`visible_guides`] is the sharp case: a [`GuideId`] is an `ActionId` and a
    /// reset puts the client back to `Authoring::solo()`, so the *next* document's
    /// first guide is minted at the same `{ lamport: 0, actor }` and an eye left open
    /// would reopen itself on a guide nobody here has ever seen.
    ///
    /// [`visible_guides`]: Self::visible_guides
    pub(crate) fn forget_document(&mut self) {
        self.cancel_stroke();
        self.visible_guides.clear();
    }
}

impl StrokeBuilder {
    /// Take one pointer report: fitted into the curve, or — once the gesture has
    /// snapped — used to **steer** the shape (§6.9).
    fn extend(&mut self, sample: InputSample) {
        match &mut self.trace {
            // The towed tip (§6.11) sits between the raw stream and the fitter — and
            // only there, which is why the steer below reads the raw position.
            Trace::Fitting { fitter, tow } => match tow {
                Some(tow) => tow.to(sample, &mut |s| fitter.push(s)),
                None => fitter.push(sample),
            },
            Trace::Snapped(assist) => assist.steer(sample.pos),
        }
    }

    /// Recognize the drawn trace and replace it with the ideal shape (§6.9). Answers
    /// whether anything snapped; a stroke already snapped never snaps again.
    fn snap(&mut self, guides: &Scaffold) -> bool {
        let Trace::Fitting { fitter, tow } = &self.trace else {
            return false;
        };
        let Some(base) = crate::assist::recognize(&fitter.trace(), self.tolerance, guides) else {
            return false;
        };
        let drawn = fitter.path();
        // From the marker on, as `trace` is: the run-up is not part of the shape.
        let pen = PenProfile::of(&drawn, fitter.start());
        let knots = drawn.len();
        // A towed gesture snapped from its towed trace; the raw pointer is up to a
        // rope beyond it, and this is what keeps the first steer from applying that
        // gap as a jump (§6.11). Read before the tow goes with the fitter.
        let shift = tow.as_ref().map_or(Vec2::ZERO, |t| {
            let s = t.string();
            s.tip - s.target
        });
        self.trace = Trace::Snapped(Assist {
            base,
            grip: base.grip(),
            shape: base,
            path: base.to_path(&pen, knots),
            pen,
            knots,
            shift,
        });
        true
    }

    /// Which family the hold snapped this stroke to, if it has been held (§6.9).
    fn assisted(&self) -> Option<crate::assist::Assisted> {
        match &self.trace {
            Trace::Snapped(a) => Some(a.shape.family()),
            Trace::Fitting { .. } => None,
        }
    }

    /// The stroke's path and start marker as they now stand: the fit's, or —
    /// once the gesture has snapped — the shape's (§6.9), whose marker is 0
    /// because a snapped path *is* the gesture, with no run-up in it.
    ///
    /// **The one place the two are chosen between**, which is what keeps the assist a
    /// path transform between the fitter and the renderer (§18.1.3) rather than a
    /// second kind of stroke: every caller reads the pair and cannot tell which it got.
    fn fitted(&self) -> (Vec<ControlPoint>, f32) {
        match &self.trace {
            Trace::Snapped(a) => (a.path.clone(), 0.0),
            // The fitted control points **as `finish` would leave them** (§6.2): the
            // free window re-solves at pen-up, so rendering the mid-stroke window
            // would put the preview on a curve the commit is not on — sub-pixel, but
            // a step wherever a lookup is discontinuous in position (the tooth,
            // §6.4). As-finished, the release adopts this very solve and live ==
            // committed holds bit for bit, the marker included. The last control point
            // is pinned to the newest sample either way, so the preview reaches the
            // cursor.
            Trace::Fitting { fitter, .. } => fitter.as_finished(),
        }
    }

    /// How many leading control points of [`fitted`](Self::fitted) are final (§6.2).
    ///
    /// A **snapped** stroke has none: steering moves every control point at once, so
    /// the whole path rides every frame. The ordinal bump in
    /// [`Session::assist_stroke`] is what lets the wire's watermark drop back to zero
    /// without walking backwards inside a gesture (§17.5).
    fn frozen_points(&self) -> usize {
        match &self.trace {
            Trace::Snapped(_) => 0,
            Trace::Fitting { fitter, .. } => fitter.frozen_points(),
        }
    }

    fn to_record(&self) -> StrokeRecord {
        let (path, start) = self.fitted();
        StrokeRecord {
            layer: self.layer,
            brush: self.brush,
            path: path_in_frame(path, self.translation),
            seed: self.seed,
            start,
            translation: self.translation,
        }
    }

    /// The record to commit, or `None` for a gesture that deposits nothing — see
    /// [`Session::end_gesture`], which is the only caller.
    fn finish(mut self) -> Option<StrokeRecord> {
        // A snapped stroke's path is the shape's, not the fit's — there is nothing
        // left for a last solve to settle.
        if let Trace::Fitting { fitter, .. } = &mut self.trace {
            // The towed tip needs nothing at pen-up (§6.11): lifting stops pulling the
            // string rather than reeling the tip in, so the mark ends where the rope
            // had towed it — the trace the preview showed.
            fitter.finish();
            if !fitter.painted() {
                return None;
            }
        }
        Some(self.to_record())
            .filter(|rec| rec.path.len() >= 2)
            .and_then(crate::gpu::stroke::loggable)
    }
}

/// A canvas-space path brought into a layer's frame at `translation` (§14.12) — the
/// conversion [`StrokeRecord::translation`] pairs with. A zero translation is the path
/// untouched, bit for bit; otherwise every position drops the whole-pixel
/// offset, exact in `f32` to the 2²⁴ the canvas already lives within.
fn path_in_frame(
    mut path: Vec<ControlPoint>,
    translation: stark_model::geom::IVec2,
) -> Vec<ControlPoint> {
    if translation != stark_model::geom::IVec2::ZERO {
        let f = translation.as_vec2();
        for p in &mut path {
            p.pos -= f;
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assist::Assisted;
    use fixtures::{UNUSABLE_TOLERANCES, session};
    use stark_model::document::ShapeAction;

    /// The family a hold snaps a drag through `f`, sampled `n` times, to.
    fn snapped(tolerance: f32, n: usize, f: impl Fn(f32) -> Vec2) -> Option<Assisted> {
        let mut s = session(ShapeAction::default());
        let at = |i: usize| InputSample::at(f(i as f32 / (n - 1) as f32));
        s.start_stroke(at(0), 1, tolerance, 0.0, stark_model::geom::IVec2::ZERO);
        for i in 1..n {
            s.gesture_to(at(i));
        }
        s.assist_stroke(&Scaffold::default());
        s.assisted()
    }

    /// **What the panel shows is what the gesture will use.** Both numbers pass the
    /// model's own gate here, so a value the op would refuse cannot sit in the session
    /// claiming otherwise — an infinite feather is the one that used to.
    #[test]
    fn the_shape_settings_land_where_the_op_that_takes_them_will() {
        use stark_model::document::{FillOp, SelectionMode, SelectionOp, SelectionShape};
        let shape = || SelectionShape::rect_from_corners(Vec2::ZERO, Vec2::splat(8.0));
        let mut s = session(ShapeAction::default());
        for raw in UNUSABLE_TOLERANCES {
            s.set_selection_feather(raw);
            s.set_shape_opacity(raw);
            assert_eq!(
                s.selection_feather(),
                SelectionOp::new(SelectionMode::Replace, shape(), raw).feather(),
                "feather {raw}",
            );
            assert_eq!(
                (s.selection_feather(), s.shape_opacity()),
                {
                    let op = FillOp::new(shape(), raw, stark_model::Srgb::BLACK, raw);
                    (op.feather(), op.opacity())
                },
                "fill parcel at {raw}",
            );
        }
    }

    /// Recognition is priced in the tolerance the fit uses, the clamped one: a raw 0
    /// snapped a sub-pixel stroke, and a raw NaN put every loop out of reach.
    #[test]
    fn the_assist_reads_the_tolerance_the_fit_does() {
        let speck = |t: f32| Vec2::new(0.4 * t, 0.0);
        let hoop = |t: f32| {
            let u = t * std::f32::consts::TAU;
            Vec2::new(150.0 * u.cos(), 90.0 * u.sin())
        };
        for raw in UNUSABLE_TOLERANCES {
            let held = InputTolerance::new(raw).get();
            assert_eq!(
                snapped(raw, 12, speck),
                snapped(held, 12, speck),
                "a 0.4 px line at tolerance {raw}"
            );
            assert_eq!(
                snapped(raw, 90, hoop),
                snapped(held, 90, hoop),
                "a loop at tolerance {raw}"
            );
        }
    }
}
