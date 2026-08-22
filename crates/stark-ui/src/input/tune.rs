//! Accelerator-and-drag tunes the brush instead of painting with it (§18.1.9):
//! Size sideways, Flow up and down.
//!
//! The canvas's own gesture rather than a shared one, unlike [`Nav`](super::Nav):
//! it moves the *brush*, and the overlays that navigate have no brush. Its
//! readout is the ring `overlays::BrushSizeRing` draws and the Brush panel's own
//! sliders, which is why this is one of the two gestures that deliberately does
//! **not** fade the chrome — the answer is on a panel.

use super::*;

/// How far a tuning drag must travel before it commits to a knob, in page px
/// (§18.1.9).
///
/// [`MIN_SPAN`]'s reasoning applied to one pointer instead of two: below this the
/// drag's *direction* is noise, and the direction is the whole of what picks the
/// parameter. A press meant for Size that happens to leave the glass two pixels high
/// must not arrive as Flow.
const AXIS_DEADZONE: f32 = 8.0;

/// The radius a tuning drag asks for, as a fraction of how far it has been dragged
/// sideways **from the press** — in canvas px, so the answer is a size for the brush
/// and not for the screen it is being set on.
///
/// Absolute rather than a rate, which is what makes the gesture legible: the drag does
/// not *nudge* the size, it states it, and the ring drawn at the press point
/// ([`BrushRing`]) is the picture of that statement. Left and right are the same
/// gesture (the travel is taken as a magnitude) because the hand is describing a
/// circle's size, and a circle has no side.
///
/// A quarter, so the **diameter is half the drag**: the ring always fits inside the
/// gesture that made it, and the cursor stays outside the circle it is describing
/// instead of sitting in the middle of it. Since the canvas radius is the screen
/// travel divided by the zoom, the ring's radius on screen is a quarter of the drag at
/// *any* zoom — the gesture measures the same in the hand, and what changes with the
/// zoom is the size in canvas px, which is the thing being set.
const RADIUS_PER_DRAG: f32 = 0.25;

/// How far a tuning drag has to travel vertically to sweep the **whole** flow range,
/// in page px.
///
/// A *rate* where the radius is stated outright ([`RADIUS_PER_DRAG`]), because there
/// is nothing for flow to be a picture of: a size drag can be shown as the circle it
/// asks for, while flow has no length on screen to be measured against, so the honest
/// mapping is the one every slider has — move the hand, move the number. Wider than a
/// screen is tall on purpose: the everyday range is the narrow band around 1, and this
/// is what makes a tenth of it a visible movement of the hand.
const FLOW_DRAG_SPAN: f32 = 800.0;

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
/// A size drag also draws itself, in [`BrushRing`] — which is not decoration but the
/// readout: the size is stated as a distance from the press, so the circle at the press
/// point *is* what the gesture means, with the size it started from behind it.
#[derive(Clone, Copy)]
pub struct Tune {
    state: AppState,
    /// The tuning drag in flight, or `None`.
    drag: Signal<Option<TuneDrag>>,
}

/// A tuning drag in flight.
#[derive(Copy, Clone)]
struct TuneDrag {
    /// The pointer's last position, page px — what a *step* is measured from, which is
    /// what Flow moves by.
    last: Vec2,
    /// Where the press was, page px. Both the axis and the **size** are measured from
    /// here rather than from the last move: which knob this gesture is about is a fact
    /// about the whole gesture, and so is the radius it is asking for
    /// ([`RADIUS_PER_DRAG`]).
    from: Vec2,
    /// The radius the brush had at the press, canvas px — the ring's reference.
    ///
    /// Kept here rather than read back off the ring so that every write to
    /// [`AppState::brush_ring`] is a write and never a read-modify-write: the drag holds
    /// everything the indicator shows, which is what keeps the picture from drifting
    /// out of step with the gesture (and what keeps a `peek` out of an `if`).
    was: f32,
    /// The view's zoom when the drag began — what turns its travel into canvas px.
    ///
    /// Latched rather than read each move, so the gesture measures against the view it
    /// started in. A wheel notch mid-drag (the pointer is captured, but the wheel is
    /// not) would otherwise move the scale under a hand that is holding still, and the
    /// size would jump without the pointer going anywhere.
    zoom: f32,
    /// The knob this drag has committed to, once it has travelled far enough to say
    /// ([`AXIS_DEADZONE`]); `None` until then.
    ///
    /// One knob per gesture, and that is the point of locking it. Both at once would
    /// read better on paper and be worse in the hand: flow's useful range is narrow
    /// enough that the incidental drift of a long sideways drag would empty or bury
    /// the brush, and the user would have no way to ask for size *alone*. The travel
    /// spent earning the lock is spent — a deadband, not a jump.
    knob: Option<Knob>,
}

/// The two parameters a tuning drag can reach, and the axis each is on.
#[derive(Copy, Clone)]
enum Knob {
    /// Sideways: the brush radius.
    Size,
    /// Up and down: how much paint the brush lays (`BrushDynamics::add`).
    Flow,
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
    /// No test on the tool, deliberately: the eraser end tunes the eraser for the
    /// reason it erases (§18.1.8), and Size and Flow are the live brush's whatever
    /// the canvas is set to do with it — a marquee tool's Fill spends `add` as its
    /// opacity.
    ///
    /// Declines before the engine exists, where there is neither a brush to tune nor a
    /// zoom to measure the drag against. The press then falls through to the paint
    /// path, which does nothing with it for the same reason.
    pub fn begin(self, e: &Event<PointerData>) -> bool {
        let Some(view) = view_of(self.state) else {
            return false;
        };
        let Some(radius) = self.state.obs.peek().as_ref().map(|o| o.brush.radius) else {
            return false;
        };
        e.prevent_default();
        e.stop_propagation();
        capture_pointer(e);
        let at = page_xy(e);
        let in_flight = TuneDrag {
            last: at,
            from: at,
            was: radius,
            zoom: view.zoom,
            knob: None,
        };
        let mut drag = self.drag;
        drag.set(Some(in_flight));
        // This gesture is what one of the tour's lessons is *about*, so the brush
        // writes it is going to make are not evidence that anybody needs telling
        // about it (§24.2). Closed by `stop`, which every release runs.
        crate::tutor::not_reaching(self.state, true);
        // Up from the press, before the drag has said what it is about, showing the brush
        // at the size it already is. That is the reference the new size will be judged
        // against, and it is also the one thing that makes this binding discoverable:
        // press with the accelerator held and the brush draws itself.
        self.show_ring(&in_flight, radius);
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
        let p = page_xy(e);
        let step = p - in_flight.last;
        in_flight.last = p;
        if in_flight.knob.is_none() {
            let travel = p - in_flight.from;
            if travel.length() >= AXIS_DEADZONE {
                in_flight.knob = Some(if travel.x.abs() >= travel.y.abs() {
                    Knob::Size
                } else {
                    Knob::Flow
                });
            }
        }
        let knob = in_flight.knob;
        drag.set(Some(in_flight));
        // Clamped to the sliders' own bounds (`panels::brush`), so the drag cannot put
        // the brush somewhere the panel is unable to show or take back.
        match knob {
            Some(Knob::Size) => {
                // Set, not nudged: the radius is a function of where the pointer *is*,
                // so it cannot drift over a long gesture, and dragging back to the press
                // asks for the finest brush rather than for the one it started on.
                let reach = (p.x - in_flight.from.x).abs();
                let radius =
                    (RADIUS_PER_DRAG * reach / in_flight.zoom).clamp(MIN_RADIUS, MAX_RADIUS);
                update_brush(self.state, |b| b.radius = radius);
                // The ring follows the *clamp* rather than the pointer, so a drag that
                // has run past the largest brush stops growing where the brush did.
                self.show_ring(&in_flight, radius);
            }
            // Up is more, because up is more on every slider in the app — and page y
            // grows downward, which is the whole of why this reads as a subtraction.
            Some(Knob::Flow) => {
                update_brush(self.state, |b| {
                    b.dynamics.add =
                        (b.dynamics.add - step.y * MAX_FLOW / FLOW_DRAG_SPAN).clamp(0.0, MAX_FLOW);
                });
                // The ring is the *size* drag's readout. Once this gesture has turned out
                // to be about flow, leaving it up would advertise a number that is not
                // moving, so it goes.
                self.hide_ring();
            }
            None => {}
        }
        true
    }

    /// End the tuning drag in flight. Harmless when there is none.
    pub fn stop(self) {
        let mut drag = self.drag;
        if drag.peek().is_some() {
            drag.set(None);
            // Inside the guard, not beside it: this runs on every release the canvas
            // sees, and the tour's bracket is a depth count — a close for a drag that
            // never opened one would cancel somebody else's (§24.2).
            crate::tutor::not_reaching(self.state, false);
        }
        self.hide_ring();
    }

    /// Draw the indicator for `drag`, asking for `radius` (canvas px). Converted to
    /// screen px here, which is the one place that knows both numbers — see
    /// [`BrushRing`].
    fn show_ring(self, drag: &TuneDrag, radius: f32) {
        let mut ring = self.state.brush_ring;
        ring.set(Some(BrushRing {
            at: drag.from,
            was: drag.was * drag.zoom,
            now: radius * drag.zoom,
        }));
    }

    /// Take the size ring down. Harmless when it is already down, and written only on a
    /// change, since every write re-renders the overlay.
    fn hide_ring(self) {
        let mut ring = self.state.brush_ring;
        if ring.peek().is_some() {
            ring.set(None);
        }
    }
}
