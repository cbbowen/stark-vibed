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

/// How far a tuning drag has to travel sideways to **double** the brush radius, in
/// page px (§18.1.9).
///
/// A ratio on the size the drag began with, not a size stated outright: the hand keeps
/// whatever brush it had chosen and asks for *more* or *less* of it, which is the
/// gesture every other editor binds here and the one a hand already reaches for. Right
/// is bigger and left is smaller, so the two directions are no longer the same gesture
/// — the drag has a sign now, because a change does and a size does not.
///
/// **Exponential** for the scrubby zoom's reason (`nav::ZOOM_DRAG_DOUBLE`): radius is felt
/// proportionally, so a fixed step per pixel would crawl on a wash and leap on a liner.
/// Equal distances are equal ratios, which also makes the gesture exactly reversible —
/// dragging back to the press restores the brush it started on.
///
/// Faster than the zoom's rate rather than matched to it, and set from the range it has
/// to cover: `MIN_RADIUS..MAX_RADIUS` is about nine doublings, and a size drag spends
/// its travel on *one* side of the press where a zoom drag may run either way from it,
/// so the budget is half a screen and not a whole one. At this rate that half-screen
/// carries the finest brush to the widest.
const SIZE_DRAG_DOUBLE: f32 = 100.0;

/// How far a tuning drag has to travel vertically to sweep the **whole** flow range,
/// in page px.
///
/// Linear where the radius is exponential ([`SIZE_DRAG_DOUBLE`]), because flow's zero
/// is a value it has to be able to reach and no number of halvings gets there. There is
/// also nothing for flow to be a picture of: a size drag can be shown as the circle it
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
/// readout: the size is a ratio on the one the press found ([`SIZE_DRAG_DOUBLE`]), so
/// the pair of circles at the press point *is* what the gesture means — the brush it
/// started on, and the brush it is asking for.
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
    /// about the whole gesture, and so is the ratio it is asking for
    /// ([`SIZE_DRAG_DOUBLE`]).
    from: Vec2,
    /// The radius the brush had at the press, canvas px — what the size drag is a ratio
    /// **on**, and the ring's reference. One number doing both, which is why the ring
    /// reads as before-and-after: the circle behind is the size the gesture is measured
    /// against, not merely the size it happened to start at.
    ///
    /// Latching it is what makes the drag a function of where the pointer *is* rather
    /// than an accumulation of steps — so a long gesture cannot drift, and a drag run
    /// past `MAX_RADIUS` and back comes back down the way it went up, since the clamp
    /// is never folded into the base. It is also why every write to
    /// [`AppState::brush_ring`] is a write and never a read-modify-write: the drag holds
    /// everything the indicator shows, which keeps the picture from drifting out of step
    /// with the gesture (and keeps a `peek` out of an `if`).
    was: f32,
    /// The view's zoom when the drag began — what turns a canvas radius into the ring's
    /// radius on screen.
    ///
    /// The size no longer passes through it: a ratio on the radius the drag began with
    /// is the same ratio at every zoom, and that is one thing the exponential mapping
    /// bought. What is left is the drawing, and it is latched so that a wheel notch
    /// mid-drag (the pointer is captured, but the wheel is not) cannot rescale the ring
    /// under a hand that is holding still — the readout would read as the size moving
    /// when it has not.
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
        // at the size it already is — which is the size every ratio this gesture asks for
        // is a ratio *of*, so the circle is the reference and not merely the first frame.
        // It is also the one thing that makes this binding discoverable: press with the
        // accelerator held and the brush draws itself.
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
                // Right is bigger, left is smaller — a ratio on the size at the press
                // rather than a size stated outright, so the hand asks for more or less
                // of the brush it already chose. Still a function of where the pointer
                // *is* and not of how it got there (`TuneDrag::was`): a long gesture
                // cannot drift, and dragging back to the press restores the brush it
                // started on exactly.
                let travel = p.x - in_flight.from.x;
                let radius = (in_flight.was * (travel / SIZE_DRAG_DOUBLE).exp2())
                    .clamp(MIN_RADIUS, MAX_RADIUS);
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
