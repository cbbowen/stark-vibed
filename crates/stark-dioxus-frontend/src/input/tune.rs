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
        let Some(mut in_flight) = drag() else {
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
