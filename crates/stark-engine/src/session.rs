//! Session: ephemeral, non-historized state (§3).
//!
//! The current tool/brush, the pan/zoom view, and the in-flight stroke being
//! dragged out. None of it is undoable — switching tools or panning never creates a
//! history step. The session interprets pointer commands and, on `EndStroke`, hands
//! the [`Engine`](crate::Engine) a finished [`StrokeRecord`] to commit.
//!
//! Three of its subjects have a file of their own: the shape gesture ([`shape`]),
//! the hover mark ([`hover`]), and the half published to collaborators
//! ([`crate::presence`]).

#[cfg(test)]
mod fixtures;
mod hover;
mod shape;

pub use shape::ShapeResult;

use crate::assist::{AssistShape, PenProfile};
use crate::command::InputSample;
use crate::command::Tool;
use crate::path::PathFitter;
use crate::peer::{GestureView, Identity, LiveGesture};
use crate::presence::{GestureSource, PresenceTx};
use crate::tow::{Tow, TowString};
use crate::view::ViewTransform;
use stark_model::document::Scaffold;
use stark_model::document::{ActorId, BrushParams, GuideId, LayerId, ShapeAction, StrokeRecord};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;
use stark_model::peer::{PeerFrame, StrokeHead};

use hover::HoverStroke;
use shape::ShapeDrag;

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
    pub view: ViewTransform,
    tool: Tool,
    brush: BrushParams,
    /// The **hand's** color, straight sRGB. Its own field rather than a reading of
    /// the brush, which does not always have one: an erasing brush carries no pigment,
    /// and a fill drawn while one is held still lays the color in hand
    /// ([`start_selection`](Self::start_selection)). While the brush paints, the
    /// frontend keeps this equal to the paint effect's own color.
    color: [f32; 3],
    /// Published to peers (§17.4) beside the `presence` half, and read on this
    /// side as well, so it is the session's own rather than the publisher's: the
    /// wire is *told* it at publish time.
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
    /// Whether collaborators' selection outlines are drawn (§17.3).
    ///
    /// View state, so each client decides for itself and nothing about it is logged or
    /// sent. Off by default: a second contour over the artwork is a cost paid on every
    /// frame you look at it, for a fact that is occasionally useful.
    pub show_peer_selections: bool,
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

    in_flight: Option<StrokeBuilder>,
    selecting: Option<ShapeDrag>,
    /// The hover mark (§18.1.10): the stroke a drag begun this instant would open,
    /// laid from the cursor along the hover's extrapolated heading. `None` when the
    /// pointer is elsewhere or the hand is doing something other than hovering.
    ///
    /// Deliberately not [`in_flight`](Self::in_flight): this one never commits, never
    /// publishes ([`gesture_source`](Self::gesture_source) reads only the two slots
    /// above) and never sets [`is_stroking`](Self::is_stroking), and a real gesture
    /// always outranks it ([`hover_view`](Self::hover_view)).
    hover: Option<HoverStroke>,
    /// Bumped on every gesture start, so a peer can tell a restart from a continuation
    /// without a clock, and on every hover update, so the renderer's per-actor head
    /// cache cannot mistake one hover's render for the next's. **One counter for both
    /// on purpose**: two could collide, and a collision revives a cached head —
    /// including one still holding a cancelled stroke's paint.
    gesture_ordinal: u64,

    /// The half of this session published to collaborators (§17.4): the name, the
    /// cursor, and the latch that decides when a frame goes out.
    /// [`active_layer`](Self::active_layer) and the in-flight gesture ride the frame
    /// too, and are handed over at publish time; everything else here is private.
    presence: PresenceTx,
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
            in_flight: None,
            selecting: None,
            hover: None,
            gesture_ordinal: 0,
            presence: PresenceTx::new(),
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

    pub fn set_tool(&mut self, tool: Tool) {
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
    pub fn set_active_layer(&mut self, id: LayerId, doc: &crate::document::DocState) -> bool {
        let known = doc.contains_layer(id);
        if known {
            self.active_layer = id;
        }
        known
    }

    /// Edge softness (canvas px) for the next shape gesture, floored at zero.
    ///
    /// A setter rather than a `pub` field because the value arrives in a command from
    /// outside and this is where the floor holds. NaN lands on 0, the neutral setting,
    /// since both of its comparisons are false.
    pub fn set_selection_feather(&mut self, feather: f32) {
        self.selection_feather = if feather > 0.0 { feather } else { 0.0 };
    }

    /// Edge softness the next shape gesture will use.
    pub fn selection_feather(&self) -> f32 {
        self.selection_feather
    }

    /// How strongly the next fill gesture's parcel lands, clamped to `0..=1`.
    pub fn set_shape_opacity(&mut self, opacity: f32) {
        // `clamp01`'s rule, spelled here because the model keeps it private: NaN lands
        // on 0 rather than passing through, since both of its comparisons are false.
        self.shape_opacity = if opacity > 0.0 { opacity.min(1.0) } else { 0.0 };
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
    /// **The one place the two halves of a guide meet** (§20.5) — the overlay pass,
    /// the stroke assist's [`Scaffold`] and the panel's rows all ask through here.
    ///
    /// Takes the document because a `Session` holds none, and which one it is combined
    /// with is the caller's: the committed document for a commit, the previewed one
    /// for what is on screen.
    pub fn shown_guides<'a>(
        &'a self,
        doc: &'a crate::document::DocState,
    ) -> impl Iterator<Item = &'a crate::document::Guide> {
        doc.guides()
            .iter()
            .filter(|g| self.visible_guides.contains(&g.id))
    }

    // --- the published half (§17.4) -------------------------------------------
    //
    // Delegated to `PresenceTx`, which holds the name, the cursor and the latch.
    // What only the session knows — the active layer, the gesture in flight and its
    // ordinal — is handed over per call.

    /// This client's display name, as peers see it. Empty when none has been set, in
    /// which case peers show [`default_name`](crate::peer::default_name) instead.
    pub fn name(&self) -> &str {
        self.presence.name()
    }

    /// Set the display name the user chose. **Sticky**: hosting or joining a session
    /// mints this client a new actor id, and
    /// [`adopt_identity`](Self::adopt_identity) will not overwrite a name set
    /// here. Setting it empty gives the choice back, and peers resume showing the
    /// id-derived default.
    pub fn set_name(&mut self, name: String) {
        self.presence.set_name(&name);
    }

    /// Where this client's pointer is, or `None` when it is off the canvas.
    /// **Filtered**, because this one goes on the wire and the guide rays are
    /// drawn through it (§20.9): a non-finite position is refused into `None`.
    pub fn set_cursor(&mut self, at: Option<Vec2>) {
        self.presence.set_cursor(at);
    }

    /// Where this client's pointer is.
    pub fn cursor(&self) -> Option<Vec2> {
        self.presence.cursor()
    }

    /// Adopt the identity a session has given this client: the run counter every
    /// published frame carries, and the id-derived name as a default — unless the
    /// user has chosen one ([`set_name`](Self::set_name)).
    pub fn adopt_identity(&mut self, identity: Identity) {
        self.presence.adopt_identity(identity);
    }

    /// The publishable half of this session, if anything a peer would care about has
    /// changed since the last call — otherwise `None` (§17.5). A **latch, not a
    /// queue**: the path delta is computed here, at drain time, against what has
    /// actually been sent, which is what lets a 240 Hz pen coalesce losslessly
    /// into a 30 Hz tick.
    pub fn publish(&mut self, now: f64) -> Option<PeerFrame> {
        let source = self.gesture_source();
        self.presence
            .publish(now, self.active_layer, self.gesture_ordinal, source)
    }

    /// Whether [`publish`](Self::publish) could produce a frame — conservative, so
    /// an idle session costs the pump nothing: it may say yes where `publish` then
    /// returns `None`, but never no where `publish` would have produced a frame.
    pub fn publish_due(&self, now: f64) -> bool {
        let in_gesture = self.in_flight.is_some() || self.selecting.is_some();
        self.presence
            .publish_due(now, self.active_layer, in_gesture)
    }

    /// The farewell frame: one publish that removes this client from every peer's
    /// roster at once, rather than making them wait out [`PEER_TIMEOUT`](stark_model::peer::PEER_TIMEOUT).
    pub fn publish_leaving(&mut self) -> PeerFrame {
        self.presence.publish_leaving(self.active_layer)
    }

    /// The gesture in flight, in the form the encoder reads it — a window onto the
    /// live fitter, not a copy of it. `None` when nothing is being dragged out (which
    /// includes a selection gesture that so far encloses nothing).
    ///
    /// Only frozen control points are reported as settled; the provisional tail is
    /// resent every frame because it can still move (§6.2) — the renderer's
    /// `FrozenHead` partition, spent on the wire instead of on the GPU.
    fn gesture_source(&self) -> Option<GestureSource> {
        match self.in_flight.as_ref() {
            Some(b) => {
                let (path, start) = b.fitted();
                Some(GestureSource::Stroke {
                    head: Box::new(StrokeHead {
                        layer: b.layer,
                        brush: b.brush,
                        seed: b.seed,
                        translation: b.translation,
                    }),
                    // In the layer's frame, as [`to_record`] converts it — the wire
                    // carries what a receiver's fold renders, so the two convert in
                    // the same place or not at all.
                    path: path_in_frame(path, b.translation),
                    // A snapped stroke has **no settled prefix**: steering moves every
                    // control point at once, so the whole path rides every frame. The
                    // ordinal bump in `assist_stroke` is what lets the watermark drop
                    // back to zero without walking backwards inside a gesture (§17.5).
                    frozen: b
                        .assist
                        .as_ref()
                        .map_or_else(|| b.fitter.frozen_points(), |_| 0),
                    start,
                })
            }
            None => self.preview_shape().map(|r| match r {
                ShapeResult::Select(op) => GestureSource::Selection(op),
                // The wire's fill rides `PeerFrame::active_layer` for its target
                // (§17.5); the pinned layer travels only in-process, so a peer's
                // preview can drift from the commit across the retarget race the
                // pin closes locally — a preview, not the document.
                ShapeResult::Fill {
                    op,
                    translation: frame,
                    ..
                } => GestureSource::Fill {
                    op,
                    translation: frame,
                },
            }),
        }
    }

    // --- the in-flight stroke (§6.2) --------------------------------------------

    pub fn is_stroking(&self) -> bool {
        self.in_flight.is_some()
    }

    /// The ordinal of the gesture in flight — bumped on every start, so a cached
    /// render of one stroke is never mistaken for a render of the next.
    pub fn gesture_ordinal(&self) -> u64 {
        self.gesture_ordinal
    }

    /// Begin a stroke, replacing any abandoned in-flight one.
    ///
    /// `seed` is the engine's, so it can be derived deterministically (§6.2).
    /// `tolerance` is what the frontend says its input resolves to, in canvas px
    /// ([`PathFitter::with_tolerance`]). `rope` is the stroke-smoothing string length,
    /// also canvas px (§6.11); `0` or anything unusable constructs no tow and feeds
    /// the fitter the raw samples. `translation` is the active layer's translation at
    /// the press (§14.12), pinned here so a translate landing mid-stroke cannot shear
    /// the record against its own gating.
    pub fn start_stroke(
        &mut self,
        tool: Tool,
        sample: InputSample,
        seed: u64,
        tolerance: f32,
        rope: f32,
        translation: stark_model::geom::IVec2,
    ) {
        self.tool = tool;
        self.selecting = None;
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
        self.in_flight = Some(StrokeBuilder {
            brush: self.brush,
            layer: self.active_layer,
            seed,
            translation,
            fitter,
            tolerance,
            assist: None,
            // A non-finite first report anchors nothing — the tip would park on it
            // and no later report could pull the string off — so the gesture runs
            // untowed rather than towed from nowhere.
            tow: (rope.is_finite() && rope > 0.0 && sample.is_admissible())
                .then(|| Tow::new(rope, tolerance, sample)),
        });
    }

    /// Extend the in-flight stroke with another sample — or, once it has snapped,
    /// **steer** the shape it snapped to (§6.9). One entry point for both halves of
    /// the gesture, so the frontend sends the same `To` for every pointer move either
    /// way.
    ///
    /// A report carrying a non-finite number is dropped here, before it reaches
    /// anything that would *remember* it: [`Tow`] keeps the tip and the target across
    /// reports, so one NaN parks the string where no later report can pull it back
    /// from, and [`Assist::steer`] folds the position into a shape that outlives it.
    /// Refusing at the door costs one report rather than the rest of the gesture.
    ///
    /// [`Assist::steer`]: crate::assist::AssistShape::adjust
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
    /// The family rather than the shape (`assist::Assisted`): all that is asked from
    /// outside the crate is whether a hold landed, and whether it landed on a line.
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
                ShapeResult::Fill {
                    layer,
                    op,
                    translation: frame,
                } => (
                    LiveGesture::Fill {
                        layer,
                        op,
                        translation: frame,
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
    /// point, so there is no settled prefix to retire.
    pub fn frozen_spans(&self) -> usize {
        self.in_flight.as_ref().map_or(0, |b| match b.assist {
            Some(_) => 0,
            None => b.fitter.frozen_spans(),
        })
    }

    /// Finish the stroke, returning the record to commit — `None` if it is empty, and
    /// `None` for a record that cannot deposit, since a step in the log that changes
    /// no pixel would spend an undo step invisibly ([`Self::end_shape`] answers a
    /// marquee click the same way). The travel that counts starts at the record's
    /// marker, so a click at the end of a watched approach is still a click: its curve
    /// holds the whole run-up, and its marker sits at the end of it
    /// ([`PathFitter::painted`]).
    pub fn end_stroke(&mut self) -> Option<StrokeRecord> {
        self.in_flight
            .take()
            .and_then(|mut b| {
                // A snapped stroke's path is the shape's, not the fit's — there is
                // nothing left for a last solve to settle.
                if b.assist.is_none() {
                    // The towed tip needs nothing at pen-up (§6.11): lifting stops
                    // pulling the string rather than reeling the tip in, so the mark
                    // ends where the rope had towed it — the trace the preview showed.
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
    pub fn forget_document(&mut self) {
        self.cancel_stroke();
        self.visible_guides.clear();
    }
}

impl StrokeBuilder {
    /// The stroke's path and start marker as they now stand: the fit's, or —
    /// once the gesture has snapped — the shape's (§6.9), whose marker is 0
    /// because a snapped path *is* the gesture, with no run-up in it.
    ///
    /// **The one place the two are chosen between**, which is what keeps the assist a
    /// path transform between the fitter and the renderer (§18.1.3) rather than a
    /// second kind of stroke: every caller reads the pair and cannot tell which it got.
    fn fitted(&self) -> (Vec<ControlPoint>, f32) {
        match self.assist.as_ref() {
            Some(a) => (a.path.clone(), 0.0),
            // The fitted control points **as `finish` would leave them** (§6.2): the
            // free window re-solves at pen-up, so rendering the mid-stroke window
            // would put the preview on a curve the commit is not on — sub-pixel, but
            // a step wherever a lookup is discontinuous in position (the tooth,
            // §6.4). As-finished, `End` adopts this very solve and live == committed
            // holds bit for bit, the marker included. The last control point is
            // pinned to the newest sample either way, so the preview reaches the
            // cursor.
            None => self.fitter.as_finished(),
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
