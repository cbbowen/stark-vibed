//! Accelerator-and-drag tunes the brush instead of painting with it (§18.1.9):
//! Size sideways, Flow up and down.
//!
//! The canvas's own gesture rather than a shared one, unlike [`Nav`]:
//! it moves the *brush*, and the overlays that navigate have no brush. Its
//! readout is what `overlays::TuneReadoutOverlay` draws — a ring for Size, a bar
//! for Flow — and the Brush panel's own sliders, which is why this is one of the
//! two gestures that deliberately does **not** fade the chrome: the answer is on
//! a panel as well as under the hand.
//!
//! **What the drag means is `stark_ui::tune`'s** (§11.2) — the rates, the axis lock
//! and the clamps, shared with the native frontend. What is left here is the three
//! things only this frontend can do: capture the pointer, write the brush, and draw
//! the readout.

use super::*;

use stark_ui::tune::Knob;

/// The brush-tuning drag — sideways for **Size** and up-and-down for **Flow**,
/// the Brush panel's two knobs under the hand that is already on the painting
/// (§18.1.9). Which chord opens it is the drag table's row (`crate::drags`,
/// accelerator+left-drag by default).
///
/// A hook shaped like [`Nav`] and driven the same way — [`begin`](Self::begin) on
/// press, [`advance`](Self::advance) on move, [`stop`](Self::stop) on release or
/// cancel — and deliberately *not* part of it: this moves the brush rather than the
/// view, so it belongs to the surfaces that paint and not to the transform overlay or
/// the guide editor, which have no brush and no use for one.
///
/// It writes through [`update_brush`] like the sliders do, which is what earns it the
/// quick-brush rack for free: while a number is held the live brush *is* that slot's,
/// so the drag tunes the slot, and the tail of the pen tunes the eraser (§18.1.8).
///
/// Either knob draws itself, in [`TuneReadout`] — which is not decoration but the
/// readout. A size drag's ring is a ratio on the one the press found
/// (`stark_ui::tune::SIZE_DRAG_DOUBLE`), so the pair of circles at the press point *is*
/// what the gesture means: the brush it started on, and the brush it is asking for. A
/// flow drag's bar is a level, which is the only honest picture of a knob with no
/// length on the canvas. And for as long as one of them is up the canvas takes the
/// crosshair down (`canvas`): what the gesture is about is a number, so the pointer has
/// stopped promising paint anywhere.
#[derive(Clone, Copy)]
pub struct Tune {
    state: AppState,
    /// The tuning drag in flight, or `None`.
    drag: Signal<Option<TuneDrag>>,
}

/// A tuning drag in flight: the shared gesture, and the one number that is this
/// frontend's.
#[derive(Copy, Clone)]
struct TuneDrag {
    /// Where the press landed, what it landed on, and the knob it has committed to —
    /// all of it measured from the press, which is what makes the drag a function of
    /// where the pointer *is* rather than an accumulation of steps
    /// (`stark_ui::tune::Tune`).
    gesture: stark_ui::tune::Tune,
    /// The view's zoom when the drag began — what turns a canvas radius into the ring's
    /// radius on screen.
    ///
    /// The size does not pass through it: a ratio on the radius the drag began with is
    /// the same ratio at every zoom, and that is one thing the exponential mapping
    /// bought. What is left is the drawing, and it is latched so that a wheel notch
    /// mid-drag (the pointer is captured, but the wheel is not) cannot rescale the ring
    /// under a hand that is holding still — the readout would read as the size moving
    /// when it has not.
    zoom: f32,
}

impl Tune {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_tune(state: AppState) -> Self {
        Self {
            state,
            drag: use_signal(|| None),
        }
    }

    /// Whether a tuning drag holds the pointer.
    pub fn holds_pointer(self) -> bool {
        self.drag.peek().is_some()
    }

    /// Begin the tuning drag at `e`: capture the pointer and raise the ring.
    /// `true` means "this press tunes the brush, it does not paint".
    ///
    /// *Which* press opens this is no longer asked here: the drag table names it
    /// (`crate::drags` — the accelerator chord by default), and the canvas calls
    /// this only for the press the table gave it, after [`Nav::begin`] — which is
    /// what leaves space+accelerator a zoom rather than a size drag.
    ///
    /// Declines before the engine exists, where there is neither a brush to tune nor a
    /// zoom to measure the drag against. The press then falls through to the paint
    /// path, which does nothing with it for the same reason.
    pub fn begin(self, e: &Event<PointerData>) -> bool {
        let Some(view) = view_of(self.state) else {
            return false;
        };
        let was = *self.state.transient.peek();
        e.prevent_default();
        e.stop_propagation();
        capture_pointer(e);
        let at = page_xy(e);
        let in_flight = TuneDrag {
            gesture: stark_ui::tune::Tune::press(at, was),
            zoom: view.zoom,
        };
        let mut drag = self.drag;
        drag.set(Some(in_flight));
        // This gesture is what one of the tour's lessons is *about*, so the brush
        // writes it is going to make are not evidence that anybody needs telling
        // about it (§24.2). Closed by `stop`, which every release runs.
        crate::tutor::not_reaching(self.state, true);
        // Up from the press, before the drag has said what it is about, showing the brush
        // at the size it already is — which is the size every ratio this gesture asks for
        // is a ratio *of*, so the circle is the reference and not merely the first frame.
        // It is also the one thing that makes this binding discoverable: press with the
        // accelerator held and the brush draws itself.
        self.show_ring(&in_flight, was.size);
        true
    }

    /// Advance the tuning drag in flight, if any. `true` means the move was tuning and
    /// the caller's own gesture logic should not see it — including the moves before
    /// the knob is chosen, which are this gesture's even though they change nothing.
    pub fn advance(self, e: &Event<PointerData>) -> bool {
        let mut drag = self.drag;
        let Some(mut in_flight) = *drag.peek() else {
            return false;
        };
        // The in-force effect's own ceiling (`BrushConfig::max_flow`) — read live
        // rather than latched at the press, since the effect chips are on a panel the
        // captured pointer does not cover. Its own statement, so no read guard is alive
        // when the write below rewrites the brush signal.
        let max = self.state.brush.peek().max_flow();
        let turn = in_flight.gesture.moved(page_xy(e), max);
        drag.set(Some(in_flight));
        if let Some(turn) = turn {
            update_brush(self.state, |_, t| turn.write(t));
            match turn.knob {
                // The ring follows the *clamp* rather than the pointer, so a drag that
                // has run past the largest brush stops growing where the brush did.
                Knob::Size => self.show_ring(&in_flight, turn.value),
                // The bar does not have to wait for the ring to come down. It was the
                // *size* drag's readout and this is the flow drag's, and a gesture has
                // one — which `TuneReadout` says by being one value, so putting the bar
                // up *is* taking the ring down.
                Knob::Flow => self.show_bar(&in_flight, turn.fill(max)),
            }
        }
        true
    }

    /// End the tuning drag in flight, taking the readout down with it — and, with the
    /// readout, giving the canvas its crosshair back. Harmless when there is none.
    pub fn stop(self) {
        let mut drag = self.drag;
        if drag.peek().is_some() {
            drag.set(None);
            // Inside the guard, not beside it: this runs on every release the canvas
            // sees, and the tour's bracket is a depth count — a close for a drag that
            // never opened one would cancel somebody else's (§24.2).
            crate::tutor::not_reaching(self.state, false);
        }
        self.hide_readout();
    }

    /// Draw the size ring for `drag`, asking for `radius` (canvas px). Converted to
    /// screen px here, which is the one place that knows both numbers — see
    /// [`BrushRing`].
    fn show_ring(self, drag: &TuneDrag, radius: f32) {
        let mut readout = self.state.tune_readout;
        readout.set(Some(TuneReadout::Size(BrushRing {
            at: drag.gesture.from(),
            was: drag.gesture.was().size * drag.zoom,
            now: radius * drag.zoom,
        })));
    }

    /// Draw the flow bar for `drag`, already reduced to how `fill` of its own
    /// range the knob is — divided where the brush was in hand, since the range
    /// is the effect's (`BrushConfig::max_flow`); the overlay is told how full,
    /// and decides for itself how long a bar that is ([`FlowBar`]).
    fn show_bar(self, drag: &TuneDrag, fill: f32) {
        let mut readout = self.state.tune_readout;
        readout.set(Some(TuneReadout::Flow(FlowBar {
            at: drag.gesture.from(),
            fill,
        })));
    }

    /// Take the readout down. Harmless when it is already down, and written only on a
    /// change, since every write re-renders the overlay — and, through the memo the
    /// canvas reads this with, hands the crosshair back.
    fn hide_readout(self) {
        let mut readout = self.state.tune_readout;
        if readout.peek().is_some() {
            readout.set(None);
        }
    }
}

/// What a brush-tuning drag is showing (§18.1.9): the ring while it is about Size,
/// the bar once it is about Flow.
///
/// **One value rather than two `Option`s**, which is what makes "the gesture shows one
/// thing" a shape the state cannot break instead of a rule every write has to keep.
/// The drag commits to a single knob and its readout has to say *which*; with a signal
/// apiece, both being up at once would be expressible, and taking the ring down when
/// the drag turns out to be about flow would be a step the flow branch remembers. Here
/// it is not a step at all — it is what assigning the other variant already means.
///
/// It also answers the canvas's own question by being `Some`
/// ([`Signals::tune_readout`](crate::state::Signals::tune_readout)): a tuning drag
/// hides the crosshair, and it does that from the press, before either knob has been
/// chosen.
#[derive(Copy, Clone, PartialEq)]
pub enum TuneReadout {
    /// Sideways: the ring. Also what the *press* raises, before the drag has said which
    /// knob it is about — the brush at the size it already is, which is the size every
    /// ratio the gesture goes on to ask for is a ratio of.
    Size(BrushRing),
    /// Up and down: the bar.
    Flow(FlowBar),
}

/// What a brush-tuning drag's size indicator draws (§18.1.9): the size being asked
/// for, and the size the drag started from, about the point it pressed on.
///
/// **Screen px, not canvas px** — deliberately a drawing instruction rather than a
/// statement about the brush. The gesture holds the one zoom it measures against
/// ([`TuneDrag::zoom`]), so converting there means the ring and the radius it
/// reports cannot be scaled by two different numbers; and it leaves the overlay pure
/// layout, with no view to read and nothing to re-render it when the engine writes.
///
/// A circle in canvas space is still a circle on screen at any angle or handedness, so
/// a radius through the zoom is the whole of the transform: this needs no matrix, which
/// is the one thing that makes a `<div>` a fair way to draw it.
#[derive(Copy, Clone, PartialEq)]
pub struct BrushRing {
    /// The press position in page px — where the ring is centred, and the one point a
    /// gesture agrees on however far it has wandered since.
    pub at: Vec2,
    /// The radius the brush had when the drag began, screen px. The reference: without
    /// it the ring says how big the brush is about to be and nothing about whether that
    /// is bigger or smaller than what the last stroke was made with.
    pub was: f32,
    /// The radius being asked for now, screen px.
    pub now: f32,
}

/// What a brush-tuning drag's flow indicator draws (§18.1.9): how full the brush is,
/// beside the point the drag pressed on.
///
/// **A share of the range, not the value** — the bar stands for the whole of
/// `0..MAX_FLOW` and the fill says where in it the brush sits, so the overlay needs
/// neither the maximum nor the panel's units to draw one. A drawing instruction on
/// [`BrushRing`]'s argument, arrived at from the other end: the ring converts here
/// because the gesture holds the one zoom that could scale it, and this carries no
/// length at all because flow has none on screen — how long a bar is, is the
/// stylesheet's to say.
///
/// No reference mark behind it, where the ring carries the size it started from. The
/// ring needs one because "it will be this big" is not an answer without "bigger than
/// what"; a bar that is a share of the whole range has already said how much, and a
/// second mark on it would be a picture of where the gesture began rather than of what
/// the brush is now carrying.
#[derive(Copy, Clone, PartialEq)]
pub struct FlowBar {
    /// The press position in page px — where the bar is centred. [`BrushRing::at`] and
    /// for its reason, and centred on it for one more: the ring is, and a readout that
    /// moved sideways at the moment the drag worked out which knob it was about would
    /// look like a fault rather than an answer.
    pub at: Vec2,
    /// How full, 0..=1 — the flow as a share of the range the sliders allow.
    pub fill: f32,
}
