//! Pointer and keyboard input: turning DOM events into
//! [`InputCommand`](stark_core::InputCommand)s
//! (§4).

use std::str::FromStr;

use dioxus::dioxus_core::spawn_forever;
use dioxus::html::geometry::ElementPoint;
use dioxus::html::input_data::MouseButton;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use crate::collab::now_seconds;
use crate::panels::brush::{MAX_FLOW, MAX_RADIUS, MIN_RADIUS};
use crate::panels::select::{current_action, modifier_mode};
use crate::platform::{
    capture_pointer, on_window_blur, on_window_event, on_window_key, on_window_pointer, sleep_ms,
};
use crate::slots::{self, Grip};
use crate::state::{AppState, BrushRing, Dwell, PickScope, TowUi, dispatch, update_brush};
use stark_core::InputSample;
use stark_core::command::{DocCommand, GestureCommand, ViewCommand};
use stark_core::document::SelectionOp;
use stark_core::document::{ShapeAction, Tool};
use stark_core::geom::{Vec2, ViewTransform};
use stark_core::{PickOptions, PickSource};

/// How close to a quarter turn a turn has to land to be pulled onto it, radians
/// (about 5°).
///
/// The frontend's to decide, like the fitting tolerance: it is a property of turning
/// something with a hand rather than of the view. Without it a canvas that has been
/// turned could only be *approximately* straightened, and a piece left a degree off
/// square reads as an accident rather than as a choice.
pub const TURN_SNAP: f32 = 0.09;

/// How far a two-finger gesture has to twist before it turns the canvas at all,
/// radians (about 6°).
///
/// Not a tolerance on the input — a touchscreen resolves an order finer than this.
/// It is the width of the band in which a pinch is *only* a pinch: two fingers
/// closing on a target do not travel along the line between them, they roll about
/// the hand, and without a band to spend that in every zoom would leave the canvas a
/// couple of degrees off true. Measured from the start of the gesture and subtracted
/// once it is crossed, so the turn picks up from where the hand is instead of
/// jumping by the width of the band.
const TWIST_DEADZONE: f32 = 0.10;

/// How far apart two fingers have to be for the pair to mean anything, in page px.
///
/// Below this the pair's *direction* is noise and its length is a divisor, so a
/// pinch reported through it would be an arbitrary rotation and an unbounded zoom.
/// Pinches that close this far are simply not reported; the fingers slip, which is
/// what two fingers on the same spot deserve.
const MIN_SPAN: f32 = 8.0;

/// How far the accelerator+space drag has to travel to **double** the zoom, in page
/// px (§18.1.9).
///
/// Set from the range it has to cover rather than by taste: the view's whole zoom
/// range is about ten doublings (`ViewTransform::MIN_ZOOM`..`MAX_ZOOM`), so at this
/// rate a sweep of roughly one screen width takes the canvas from as far out as it
/// goes to as far in — reachable in one gesture, without a short drag overshooting
/// the picture.
const ZOOM_DRAG_DOUBLE: f32 = 180.0;

/// The view-navigation bindings — two-finger pan/zoom/turn, middle-drag and
/// space-drag pan, space+accelerator scrubby zoom, cursor-anchored wheel zoom —
/// shared by every surface that sits over the canvas: the canvas itself and the
/// transform mode's catcher, box and handles. One implementation, so what "the pan
/// bindings" and "the zoom rate" mean cannot drift between surfaces.
///
/// Each surface makes its own with [`Nav::use_nav`]; the pointer capture on the
/// pressed element keeps two instances from ever navigating at once. Policy stays
/// at the call site — the canvas fades the chrome while it pans and cancels the
/// stroke a second finger interrupted, the transform overlay deliberately does
/// neither — only the mechanics live here.
///
/// The three entry points are a lifecycle and are meant to be called as one:
/// [`begin`](Self::begin) on press, [`advance`](Self::advance) on move,
/// [`release`](Self::release) on release or cancel. Each answers the same question —
/// *was this event mine?* — so a surface routes its pointers by asking three times
/// and never by inspecting buttons or pointer types itself.
#[derive(Clone, Copy)]
pub struct Nav {
    state: AppState,
    /// The one-pointer drag in flight, or `None`.
    drag: Signal<Option<Drag>>,
    /// The fingers on this surface (§18.1.7). Separate from `drag` because a
    /// finger is identified by its id rather than by being *the* pointer — that is
    /// the whole difference touch makes.
    touch: Signal<Touch>,
}

/// A one-pointer view drag — a middle-drag or a space-drag — and what it does with
/// the motion.
#[derive(Copy, Clone)]
struct Drag {
    /// The pointer's last position in **page px** (the one frame every surface
    /// reports in, whatever its own origin).
    last: Vec2,
    /// What the motion means. Decided at the press and kept for the whole gesture: a
    /// drag is what it was begun as, so letting go of the accelerator halfway through
    /// a zoom does not hand the canvas to the pan mid-motion, under a hand that is
    /// still making one gesture.
    mode: Mode,
}

/// What a one-pointer view drag does.
#[derive(Copy, Clone)]
enum Mode {
    /// Move the canvas with the pointer.
    Pan,
    /// Scale the canvas about the **press** position (page px) as the pointer is
    /// dragged right or up — the scrubby zoom of every raster editor, for the hand
    /// that is already holding space (§18.1.9). Rebelle's two directions, taken
    /// together rather than one or the other, so the hand does not have to know which
    /// axis this app chose.
    ///
    /// The anchor is fixed for the gesture rather than following the pointer, because
    /// a zoom is a scale *about a point*: re-anchoring each move would slide the
    /// canvas out from under the hand while it scaled, and the point the drag started
    /// on is the one the user aimed at.
    Zoom { anchor: Vec2 },
}

/// The fingers on one surface, and the two-finger gesture they are making
/// (§18.1.7).
#[derive(Clone, Default)]
struct Touch {
    /// Every finger down, in the order it landed: pointer id and last position in
    /// page px. The order is load-bearing — the gesture is made by the **first two**,
    /// so a third finger joining changes nothing and a lift re-forms the pair from
    /// whoever is left, both without a jump.
    down: Vec<(i32, Vec2)>,
    /// The gesture in flight. Born when a second finger lands and buried when the
    /// last one lifts — deliberately outliving the *second* finger, so a pinch that
    /// ends with one finger still on the glass keeps panning rather than going dead
    /// under a hand that never left.
    pinch: Option<Pinch>,
}

/// What a two-finger gesture accumulates: the part of it that a pair of finger
/// positions cannot say, because it is a fact about the gesture's whole history.
#[derive(Copy, Clone)]
struct Pinch {
    /// The view's angle when the gesture began — what the twist is measured from.
    from: f32,
    /// Raw twist since then, radians, **before** the deadzone. Raw so that twisting
    /// back out of the band un-turns by exactly as much as twisting into it did:
    /// accumulating the deadzoned angle instead would ratchet, and a gesture that
    /// wandered a degree either way would walk the canvas around.
    twist: f32,
    /// The angle last asked for. Each report is the step from here, so the snap's
    /// pull onto a quarter turn is spent once instead of re-applied every move.
    asked: f32,
}

impl Nav {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_nav(state: AppState) -> Self {
        Self {
            state,
            drag: use_signal(|| None),
            touch: use_signal(Touch::default),
        }
    }

    /// Whether `e` is a press this takes as navigation — a second finger on the
    /// glass, the middle button anywhere, or space with a contact — and if so,
    /// begin: capture the pointer and swallow the event. `true` means "this press
    /// is navigation, not yours"; callers check it before starting their own
    /// gesture, and abandon any gesture already in flight.
    ///
    /// Space with the accelerator held is the same press asking to *zoom* rather than
    /// to pan ([`Mode::Zoom`], §18.1.9), so it answers `true` for exactly the presses
    /// it did before: the modifier chooses between two navigations rather than
    /// deciding whether this is one.
    ///
    /// A *contact* rather than the primary button ([`is_contact`]), so the pen's
    /// eraser end pans under space exactly as its tip does (§18.1.8). Space held
    /// means "this press moves the canvas" whichever end of the stylus is against
    /// it — the alternative is a pan that works one way up and paints the other.
    pub fn begin(self, e: &Event<PointerData>) -> bool {
        if is_finger(e) {
            return self.finger_down(e);
        }
        let mode = match e.trigger_button() {
            // The middle button is the pan whatever is held down with it: it is the
            // binding for a hand already on the mouse, and there is no second gesture
            // there for a modifier to pick out.
            Some(MouseButton::Auxiliary) => Some(Mode::Pan),
            _ if is_contact(e) && *self.state.space_down.peek() => Some(if accel(e.modifiers()) {
                Mode::Zoom { anchor: page_xy(e) }
            } else {
                Mode::Pan
            }),
            _ => None,
        };
        let Some(mode) = mode else { return false };
        e.prevent_default(); // suppress middle-click autoscroll
        e.stop_propagation();
        capture_pointer(e);
        let mut drag = self.drag;
        drag.set(Some(Drag {
            last: page_xy(e),
            mode,
        }));
        true
    }

    /// Advance the navigation in flight, if any. `true` means the move was
    /// navigation and the caller's own gesture logic should not see it.
    ///
    /// Not a no-op when it answers `false`: a lone finger's moves are recorded even
    /// while it paints, because a second finger landing has to pair with where the
    /// first one has got to rather than with where it pressed. So call it on **every**
    /// move, ahead of whatever the surface does with the ones it keeps — not only on
    /// the ones a gesture has left over.
    pub fn advance(self, e: &Event<PointerData>) -> bool {
        if is_finger(e) {
            return self.finger_move(e);
        }
        let mut drag = self.drag;
        let Some(in_flight) = drag() else {
            return false;
        };
        let p = page_xy(e);
        let command = match in_flight.mode {
            // `Pan` is incremental, so the anchor is re-set each move.
            Mode::Pan => Some(ViewCommand::Pan {
                delta: p - in_flight.last,
            }),
            Mode::Zoom { anchor } => {
                // Right and up both zoom in — page y grows downward, which is the whole
                // of why the second term is subtracted. **Summed** rather than
                // projected onto the diagonal, so a drag along either axis alone runs
                // at exactly the documented rate and one that asks for both gets both;
                // either way this is linear in the pointer's position, so the zoom is a
                // function of where the pointer *is* and a drag that wanders out and
                // back leaves the canvas where it found it.
                //
                // Exponential in that distance, which is what makes the gesture feel
                // the same at every zoom level: adding a fixed step to a multiplicative
                // quantity instead would crawl when zoomed out and leap when zoomed in.
                let step = p - in_flight.last;
                let travel = step.x - step.y;
                // A pointer that has not moved along the gesture's axis is not asking
                // for a zoom of 1.0, it is not asking for a zoom at all — dispatching
                // one would repaint the canvas to leave it exactly as it was.
                (travel != 0.0).then(|| ViewCommand::Zoom {
                    anchor,
                    factor: (travel / ZOOM_DRAG_DOUBLE).exp2(),
                })
            }
        };
        drag.set(Some(Drag {
            last: p,
            ..in_flight
        }));
        if let Some(command) = command {
            dispatch(self.state, command);
        }
        true
    }

    /// Report a release or a cancel. `true` means fingers are **still down** and the
    /// interaction is not over, so the caller should hold its own teardown: lifting
    /// one finger of a pinch ends nothing, and a surface that tore down there would
    /// end the gesture on whichever finger the hand happened to raise first.
    ///
    /// Always `false` for a mouse or a pen, which have nothing to be the rest of.
    pub fn release(self, e: &Event<PointerData>) -> bool {
        if !is_finger(e) {
            return false;
        }
        let mut touch = self.touch;
        let mut t = touch.write();
        t.down.retain(|(id, _)| *id != e.pointer_id());
        if t.down.is_empty() {
            t.pinch = None;
            return false;
        }
        true
    }

    /// End the navigation in flight, whatever it was. Harmless when there is none.
    pub fn stop(self) {
        let mut drag = self.drag;
        if drag.peek().is_some() {
            drag.set(None);
        }
        let mut touch = self.touch;
        if !touch.peek().down.is_empty() {
            touch.set(Touch::default());
        }
    }

    /// Cursor-anchored wheel zoom. Anchored by page position: it equals the
    /// canvas's own coordinates for full-viewport surfaces, and it is the only
    /// frame an element like the transform box (whose local coordinates move
    /// with it) can meaningfully report.
    pub fn wheel(self, e: Event<WheelData>) {
        e.prevent_default();
        e.stop_propagation();
        let dy = e.delta().strip_units().y;
        if dy != 0.0 {
            let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
            let p = e.page_coordinates();
            dispatch(
                self.state,
                ViewCommand::Zoom {
                    anchor: Vec2::new(p.x as f32, p.y as f32),
                    factor,
                },
            );
        }
    }

    /// A finger landing. `true` once there are two of them, which is where the
    /// gesture becomes navigation.
    fn finger_down(self, e: &Event<PointerData>) -> bool {
        // Read before the fingers are locked, so nothing holds two signals at once.
        let angle = view_of(self.state).map_or(0.0, |v| v.rotation);
        let mut touch = self.touch;
        let mut t = touch.write();
        // A *primary* touch is the first contact of its type, so anything still
        // listed when one arrives is a finger whose release never came — a cancel the
        // browser swallowed, a tab switched away from mid-gesture. Cleared on the
        // fact that says so rather than by trying to catch every way a release can go
        // missing, since a single stale entry would make the next lone finger a pinch
        // and painting by touch would stop working for the rest of the session.
        if e.is_primary() {
            *t = Touch::default();
        }
        let id = e.pointer_id();
        if !t.down.iter().any(|(down, _)| *down == id) {
            t.down.push((id, page_xy(e)));
        }
        if t.down.len() < 2 {
            return false; // one finger is the caller's — it paints
        }
        e.prevent_default();
        e.stop_propagation();
        capture_pointer(e);
        if t.pinch.is_none() {
            t.pinch = Some(Pinch {
                from: angle,
                twist: 0.0,
                asked: angle,
            });
        }
        true
    }

    /// A finger moving. Drives the view once a gesture is in flight; before then it
    /// only records where the finger is, which is what lets a second finger landing
    /// mid-stroke pair with where the first one *is* rather than where it pressed.
    fn finger_move(self, e: &Event<PointerData>) -> bool {
        let mut touch = self.touch;
        let mut t = touch.write();
        let id = e.pointer_id();
        let Some(i) = t.down.iter().position(|(down, _)| *down == id) else {
            return false;
        };
        let now = page_xy(e);
        let was = std::mem::replace(&mut t.down[i].1, now);
        let Some(mut pinch) = t.pinch else {
            return false; // a lone finger with no gesture behind it: painting
        };

        let command = if t.down.len() < 2 {
            // Down to the gesture's last finger: a plain pan, so the canvas stays in
            // hand until the hand itself leaves.
            ViewCommand::Pan { delta: now - was }
        } else {
            // A third finger is a bystander — the gesture is the first two — and its
            // moves are swallowed rather than acted on, so a hand resting on the glass
            // mid-pinch does not fight the two fingers doing the work.
            if i > 1 {
                return true;
            }
            // The pair as it was and as it now is. One finger moved, so the other side
            // of the pair is the same in both.
            let (a, b) = (t.down[0].1, t.down[1].1);
            let (before, after) = if i == 0 {
                ((was, b), (a, b))
            } else {
                ((a, was), (a, b))
            };
            let (u, v) = (before.1 - before.0, after.1 - after.0);
            let (span, spans) = (u.length(), v.length());
            if span < MIN_SPAN || spans < MIN_SPAN {
                return true;
            }
            // The twist the hand has actually made, then what the canvas is asked to
            // do about it: nothing until the deadzone has been crossed, and pulled
            // onto a quarter turn whenever the result lands near one.
            pinch.twist += u.angle_to(v);
            let earned = (pinch.twist.abs() - TWIST_DEADZONE).max(0.0) * pinch.twist.signum();
            let asked = snap_quarter(pinch.from + earned);
            let command = ViewCommand::Pinch {
                anchor: 0.5 * (before.0 + before.1),
                to: 0.5 * (after.0 + after.1),
                scale: spans / span,
                turn: asked - pinch.asked,
            };
            pinch.asked = asked;
            t.pinch = Some(pinch);
            command
        };
        // Released before dispatching: the command re-enters the engine and rewrites
        // the frontend's observable, and nothing that runs there should be able to
        // find this surface's fingers half-updated.
        drop(t);
        dispatch(self.state, command);
        true
    }
}

/// Whether `e` came from a finger — the one pointer type that arrives in pairs.
///
/// A pen is deliberately not one, on the same screen and through the same API: it
/// reports a single contact, and the whole point of the two-finger gesture is to be
/// able to move the canvas *without* putting the pen down.
fn is_finger(e: &Event<PointerData>) -> bool {
    e.pointer_type() == "touch"
}

/// Whether `e` puts the tool **on** the canvas: the primary button, or the pen's
/// other end against the glass.
///
/// One definition, because "a press that draws" is asked in three places — the
/// canvas's own press, the space-drag pan, and the eraser hold — and a press that
/// counted as a contact in one of them and not another would either paint without
/// its brush or arm a brush without painting.
pub fn is_contact(e: &Event<PointerData>) -> bool {
    e.trigger_button() == Some(MouseButton::Primary) || is_eraser(e)
}

/// Whether `m` holds the platform's **accelerator** — Ctrl, or Command on a Mac.
///
/// Either, everywhere, rather than asking which platform this is: the keyboard
/// shortcuts have always accepted both, and a binding that insisted on Ctrl would be
/// unreachable on the one platform where Ctrl+drag is how the browser reports a
/// secondary click in the first place.
fn accel(m: Modifiers) -> bool {
    m.contains(Modifiers::CONTROL) || m.contains(Modifiers::META)
}

/// Whether `e` is the pen's **eraser end** — the tail of the stylus, reported as
/// a pen contact carrying the eraser button (§18.1.8).
///
/// Read off the raw event rather than through [`MouseButton`], which stops at the
/// fifth button and folds every code past it into `Unknown` — so a pen's eraser
/// (`button` 5, `buttons` bit 32, per Pointer Events) and a mouse's seventh
/// thumb button would arrive here as the same value. The web event says which,
/// and it is already in the tree; off-wasm the downcast simply finds nothing,
/// which is the right answer on a platform with no pens.
///
pub fn is_eraser(e: &Event<PointerData>) -> bool {
    e.downcast::<web_sys::PointerEvent>()
        .is_some_and(is_eraser_event)
}

/// [`is_eraser`] against a raw web event — what the window-level binding sees
/// ([`bind_pen`]), where there is no dioxus event to unwrap.
///
/// Both button fields, because the two halves of a press report differently: the
/// press and the release name the button that *changed* (`button`), while every
/// move between them names only what is still down (`buttons`, with `button` at
/// −1). A test on either alone would arm on the press and then, one move later,
/// disagree with itself.
fn is_eraser_event(raw: &web_sys::PointerEvent) -> bool {
    /// `button` for the eraser end, per Pointer Events.
    const ERASER_BUTTON: i16 = 5;
    /// The same, as its bit in `buttons`.
    const ERASER_BUTTONS: u16 = 32;

    raw.pointer_type() == "pen"
        && (raw.button() == ERASER_BUTTON || raw.buttons() & ERASER_BUTTONS != 0)
}

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

/// The brush-tuning drag: accelerator+drag over the canvas, sideways for **Size** and
/// up-and-down for **Flow** — the Brush panel's two knobs, under the hand that is
/// already on the painting (§18.1.9).
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

    /// Whether `e` is a press this takes as brush tuning — a contact with the
    /// accelerator held — and if so, begin: capture the pointer and swallow the
    /// event. `true` means "this press tunes the brush, it does not paint".
    ///
    /// Asked *after* [`Nav::begin`], which is what leaves space+accelerator a zoom
    /// rather than a size drag.
    ///
    /// A [`is_contact`] rather than the primary button, and no test on the tool: the
    /// eraser end tunes the eraser for the reason it erases (§18.1.8), and Size and
    /// Flow are the live brush's whatever the canvas is set to do with it — a marquee
    /// tool's Fill spends `add` as its opacity.
    ///
    /// Declines before the engine exists, where there is neither a brush to tune nor a
    /// zoom to measure the drag against. The press then falls through to the paint
    /// path, which does nothing with it for the same reason.
    pub fn begin(self, e: &Event<PointerData>) -> bool {
        if !(is_contact(e) && accel(e.modifiers())) {
            return false;
        }
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

/// The canvas's **paint** gesture: a stroke or a marquee, from the press that
/// starts one to the release that commits it (§6.8, §6.9, §6.11).
///
/// A hook shaped like [`Nav`] and [`Tune`] and driven the same way, and it exists
/// for the reason those two do — *one thing owns one gesture*. This one was
/// spread over three places: the `drawing` flag and the stashed shape action were
/// the `Canvas` component's, the tow string and the assist watcher were
/// `AppState`'s, and the two teardown paths were free functions that took the
/// component's halves by `&mut`. [`end_interaction`] therefore had five
/// parameters and `abandon_gesture` had three, and between them they had to agree,
/// by hand, about what "in flight" means — across four call sites.
///
/// What is *not* here is deliberate. Whether this press is navigation, brush
/// tuning, an eyedropper sample or a stroke is **routing**, and routing stays at
/// the canvas: it is the one place that can see all four bindings at once and put
/// them in the order that makes space+Alt a pan and Ctrl+space a zoom. This owns
/// only what happens once the press has turned out to be paint.
///
/// The signals that stay in [`AppState`] stay there for stated reasons and are
/// unaffected: `pick.dragging` because the eyedropper's options bar reads it,
/// `brush_ring` and `tow` because sibling overlays draw them, `canvas_active`
/// because the whole chrome fades on it.
#[derive(Clone, Copy)]
pub struct Paint {
    state: AppState,
    /// Whether a gesture is in flight — the thing the three entry points below
    /// keep in step with the engine, and the whole of what used to be passed
    /// around as `&mut Signal<bool>`.
    drawing: Signal<bool>,
    /// The panel's shape action, stashed while a gesture's modifier keys override
    /// it (§6.8) and put back when the gesture ends, however it ends.
    restore: Signal<Option<ShapeAction>>,
}

impl Paint {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_paint(state: AppState) -> Self {
        Self {
            state,
            drawing: use_signal(|| false),
            restore: use_signal(|| None),
        }
    }

    /// Open a gesture at `e` with `tool`. `false` when there is nothing to open it
    /// against — a press that arrives before WebGPU init has finished has no canvas
    /// space to land in, and leaving `drawing` clear is what keeps the moves after
    /// it inert too.
    pub fn begin(self, e: &Event<PointerData>, tool: Tool) -> bool {
        let state = self.state;
        let (Some(sample), Some(tolerance)) = (sample(state, e), input_tolerance(state, e)) else {
            return false;
        };
        // The eraser end's brush is already in force by now — it holds its slot
        // from a window-level binding that runs ahead of every handler in the tree
        // (`bind_pen`, §18.1.8). This press only has to *draw*, with whatever brush
        // the engine is holding.

        // The marquee modifiers override the *combine mode*, so they apply only
        // while the panel's action is a selecting one: under Fill there is nothing
        // to combine, and letting shift quietly turn a fill into a union-select
        // would be the worst kind of surprise (§18.0.4).
        let action = current_action(state);
        if tool.is_selection()
            && action.is_select()
            && let Some(m) = modifier_mode(e.modifiers())
        {
            let mut restore = self.restore;
            restore.set(Some(action));
            dispatch(state, ViewCommand::SetShapeAction(ShapeAction::Select(m)));
        }
        dispatch(
            state,
            GestureCommand::Start {
                tool,
                sample,
                // What this device and this zoom level actually resolve to, which
                // is what the fit prices against.
                tolerance,
                // The brush's smoothing as a canvas-px string (§6.11); zero for the
                // selection tools, which fit no curve.
                rope: if tool.is_selection() {
                    0.0
                } else {
                    input_rope(state)
                },
            },
        );
        let mut drawing = self.drawing;
        drawing.set(true);
        // Seed the string overlay; a ropeless gesture leaves it `None` and the
        // per-move refresh stays gated off.
        refresh_tow(state);
        // Watch for the pen being held still, which snaps the stroke to the shape
        // it resembles (§6.9). Painting only: a marquee is already an exact shape,
        // so there is nothing for a hold to improve.
        if !tool.is_selection() {
            watch_for_hold(state, elem_xy(e));
        }
        true
    }

    /// Feed a move to the gesture in flight. `false` when there is none, which is
    /// what leaves the caller's cursor reporting to run.
    pub fn advance(self, e: &Event<PointerData>) -> bool {
        if !(self.drawing)() {
            return false;
        }
        let state = self.state;
        // In screen px, before the sample is mapped: whether the hand is holding
        // still is a fact about the hand (§6.9). Once the stroke has snapped this
        // stops watching and the same `To` steers the shape instead.
        pointer_moved(state, elem_xy(e));
        // Every report the browser folded into this event reaches the fitter
        // (`samples`), not just the one it chose to deliver. `dispatch_sample`, not
        // `dispatch`: a sample changes pixels, not chrome, and the full dispatch's
        // observable refresh re-diffs the chrome per pointer move. The preview fold
        // is rebuilt once per painted frame either way, so extra samples cost a fit
        // push each, not a render.
        for s in samples(state, e).unwrap_or_default() {
            crate::state::dispatch_sample(state, GestureCommand::To { sample: s });
        }
        // The string overlay tracks the tow (§6.11). Gated on its own signal so a
        // plain brush pays nothing here: only a gesture that started with a rope
        // ever reads the engine or dirties the overlay's scope per move.
        if state.tow.peek().is_some() {
            refresh_tow(state);
        }
        true
    }

    /// End the gesture, **committing** what it drew. Harmless when there is none.
    ///
    /// Unless a composing mode opened under the hand mid-gesture, in which case
    /// this release belongs to a canvas that stopped taking paint the moment the
    /// mode took it (`crate::modes`) — the canvas's own move handler abandons as
    /// soon as the pointer stirs, and this is the hand that opened a transform and
    /// then simply lifted, whose release would otherwise commit the press as a dot.
    pub fn end(self) {
        let composing = crate::modes::is_composing(self.state);
        self.close(if composing {
            GestureCommand::Cancel
        } else {
            GestureCommand::End
        });
    }

    /// End the gesture **committing nothing** — what a press that turns out to be
    /// navigation does to the stroke it interrupted (§18.1.7). Harmless when there
    /// is none.
    ///
    /// [`GestureCommand::Cancel`] rather than `End`, and that is the whole point: a
    /// second finger landing means the first was never drawing, it was opening a
    /// pinch, so navigating must leave no mark. The same applies to a middle-drag
    /// begun mid-stroke, which is the one other way [`Nav::begin`] can answer
    /// `true` with a gesture already running.
    pub fn abandon(self) {
        self.close(GestureCommand::Cancel);
    }

    /// The one teardown both ends share, so "what a finished gesture leaves behind"
    /// is stated once: the command, the shape action put back, the hold watcher
    /// stopped, the string taken down. It was two copies that had to agree.
    fn close(self, command: GestureCommand) {
        let state = self.state;
        let mut drawing = self.drawing;
        if drawing() {
            dispatch(state, command);
            drawing.set(false);
        }
        let mut restore = self.restore;
        if let Some(base) = restore.take() {
            dispatch(state, ViewCommand::SetShapeAction(base));
        }
        stop_watching(state);
        // The stroke the string belonged to is over, however it ended (§6.11).
        refresh_tow(state);
    }
}

/// `to` pulled onto the nearest quarter turn if it is within [`TURN_SNAP`] of one.
pub fn snap_quarter(to: f32) -> f32 {
    let quarter = (to / std::f32::consts::FRAC_PI_2).round() * std::f32::consts::FRAC_PI_2;
    if (to - quarter).abs() <= TURN_SNAP {
        quarter
    } else {
        to
    }
}

/// The signed turn from `from` to `to`, the short way round — so easing between two
/// angles never takes the long way about, and 1° short of a full circle is 1°.
pub fn shortest_turn(from: f32, to: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (to - from + PI).rem_euclid(TAU) - PI
}

/// Pointer position in page coordinates — the frame that stays still while
/// absolutely-positioned chrome (frame handles, the transform box) moves under
/// the pointer mid-drag.
pub fn page_xy(e: &Event<PointerData>) -> Vec2 {
    let p = e.page_coordinates();
    Vec2::new(p.x as f32, p.y as f32)
}

/// Bind the app's keyboard shortcuts, once, for the life of the page.
///
/// On the window rather than on the app's root element — see
/// [`platform::on_window_key`] for why an element cannot hold them.
///
/// Only the **keydown** side is withheld from a field being typed into. Keyup is
/// what disarms `space_down` and corrects `alt_down`, and focus can move between a
/// press and its release — a click into the rename field with space held — so a
/// guarded keyup would leave the pan armed with nothing to release it. Nothing is
/// given up by letting it through: on keyup there is no default action left to
/// cancel, since a character is inserted on the press.
pub fn bind_shortcuts(state: AppState) {
    on_window_key("keydown", move |e| {
        if !typing_into_a_field(&e) {
            handle_keydown(state, &e);
        }
    });
    on_window_key("keyup", move |e| handle_keyup(state, &e));
    // The one event that takes a key away without ever sending its keyup: focus
    // leaving the window. A number held across an Alt+Tab would otherwise hold
    // its brush for the rest of the session, with the key that ends it now
    // belonging to another window (`slots::release_all`).
    on_window_blur(move || slots::release_all(state));
}

/// Bind the pen's eraser end to its brush slot, once, for the life of the page
/// (§18.1.8).
///
/// The pointer half of [`bind_shortcuts`], and deliberately shaped like it: the
/// tail of the stylus is a **hold**, exactly as a number key is, so it is bound
/// once at the window rather than being armed by whichever surface happens to be
/// pressed. That is what lets it reach past the canvas — dragging Size or Flow
/// with the eraser tunes *the eraser*, and eraser-clicking a preset assigns it to
/// the eraser — for the same reason holding `3` while dragging Size tunes slot 3.
/// Armed by each surface instead, it would work on the surfaces somebody
/// remembered, and the list of the ones they did not is the kind nobody keeps
/// complete.
///
/// The two tests are deliberately **not** the same one:
///
/// - The **press** has to really be the eraser ([`is_eraser_event`]), or the tip
///   would arm the eraser's slot and every ordinary stroke would erase.
/// - The **release** is any pen leaving the glass. A stylus has one contact, so a
///   tip release cannot coexist with the tail being down, and a driver that
///   reports the release without the eraser bit still ends the hold — where the
///   stricter test would leave the brush swapped with nothing left to swap it
///   back. [`slots::release`] is a no-op unless an eraser hold is in flight, so
///   asking too often costs nothing where asking too rarely costs the session.
///
/// A *finger's* release is left alone on purpose: a palm settling on the glass
/// mid-erase would otherwise hand the brush back under a pen that never moved.
pub fn bind_pen(state: AppState) {
    on_window_pointer("pointerdown", move |e| {
        if is_eraser_event(&e) {
            slots::hold(state, slots::ERASER, Grip::Eraser);
        }
    });
    // Both edges, because a cancel is a release the browser made on your behalf —
    // a gesture the system took over, a tab switched away from mid-stroke.
    for kind in ["pointerup", "pointercancel"] {
        on_window_pointer(kind, move |e| {
            if e.pointer_type() == "pen" {
                slots::release(state, slots::ERASER, Grip::Eraser);
            }
        });
    }
}

/// Refuse the browser's context menu, once, for the life of the page.
///
/// A pen held still is a **gesture** here, not a request for a menu: the drawing
/// assist snaps a stroke to the shape it resembles after 0.45s of dwell (§6.9,
/// [`DWELL`]), which is inside the half-second Windows spends deciding that a
/// held stylus meant a right-click. So the menu arrives on top of the assist,
/// over the canvas, mid-stroke. The same hold ends the same way on a slider
/// being dragged, a preset row, a layer being reordered, a transform handle —
/// every drag long enough to be deliberate.
///
/// Bound at the window rather than per surface, [`bind_pen`]'s argument exactly:
/// the surfaces where this is unwanted are all of them, and a handler per surface
/// would work on the ones somebody remembered. The right button is a tool in the
/// navigator's miniature and means nothing anywhere else, so there is no reading
/// of a press this takes away.
///
/// The one exception is a text field, where the browser's menu is the only cut,
/// copy and paste the app offers — the same carve-out the shortcuts make for the
/// same reason ([`typing_into_a_field`]).
pub fn bind_context_menu() {
    on_window_event("contextmenu", |e| {
        if !e.target().is_some_and(|t| is_text_entry(&t)) {
            e.prevent_default();
        }
    });
}

/// Whether `e` was typed into a control that owns its own keystrokes — a text
/// field, a `<select>`, a contenteditable region.
///
/// Read off the event's target, which for a key event *is* what has focus, rather
/// than off a flag the fields set on focus and clear on blur. A field that unmounts
/// while focused — commit-and-close on a rename — never fires its blur, and a flag
/// left stuck on would kill every shortcut for the rest of the session. The DOM is
/// asked at the moment of the keystroke, so it cannot fall out of step.
///
/// Declining here is also what hands the field the browser's own editing bindings:
/// Ctrl+Z undoes the *text* rather than the document, and Ctrl+A selects the text
/// rather than the canvas, purely because nothing calls `prevent_default` on them.
///
/// This is the *only* place a widget can opt out. `e.stop_propagation()` in an
/// element's own `onkeydown` will not do it: dioxus-web reads `prevent_default`
/// off a handled event but never calls `stopPropagation` on the underlying DOM
/// event, so propagation is halted inside the virtual tree only and the real event
/// reaches the window regardless.
fn typing_into_a_field(e: &web_sys::KeyboardEvent) -> bool {
    e.target().is_some_and(|t| is_text_entry(&t))
}

/// Whether `target` is a control that owns its own keystrokes — a text field, a
/// `<select>`, a contenteditable region.
///
/// Split out from [`typing_into_a_field`] because the *menu* asks the same
/// question of a different event ([`bind_context_menu`]): the one place the
/// browser's own editing bindings are wanted is the one place its own menu is,
/// and two lists of which elements those are would be one list too many.
fn is_text_entry(target: &web_sys::EventTarget) -> bool {
    use wasm_bindgen::JsCast;

    let Some(el) = target.dyn_ref::<web_sys::HtmlElement>() else {
        return false;
    };
    el.is_content_editable()
        || match el.tag_name().as_str() {
            "TEXTAREA" | "SELECT" => true,
            // Sliders, checkboxes and color wells are not text entry. They want
            // arrows and space from the browser, but Ctrl+Z over one still means
            // the document — there is no text there for it to mean anything else.
            "INPUT" => !matches!(
                el.unchecked_ref::<web_sys::HtmlInputElement>()
                    .type_()
                    .as_str(),
                "button" | "checkbox" | "color" | "file" | "radio" | "range" | "reset" | "submit"
            ),
            _ => false,
        }
}

/// The pressed key, in the same typed vocabulary the rsx! handlers read.
fn key_of(e: &web_sys::KeyboardEvent) -> Key {
    Key::from_str(&e.key()).unwrap_or(Key::Unidentified)
}

/// The modifier set held during `e`.
fn modifiers_of(e: &web_sys::KeyboardEvent) -> Modifiers {
    let mut m = Modifiers::empty();
    if e.alt_key() {
        m.insert(Modifiers::ALT);
    }
    if e.ctrl_key() {
        m.insert(Modifiers::CONTROL);
    }
    if e.meta_key() {
        m.insert(Modifiers::META);
    }
    if e.shift_key() {
        m.insert(Modifiers::SHIFT);
    }
    m
}

fn handle_keydown(mut state: AppState, e: &web_sys::KeyboardEvent) {
    match key_of(e) {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(true);
            e.prevent_default();
        }
        // Alt on its own focuses the browser's menu bar on Windows and Linux, which
        // would take the keyboard away the moment the eyedropper is reached for.
        Key::Alt => e.prevent_default(),
        _ => {}
    }

    let m = modifiers_of(e);
    track_alt(state, m);
    if !accel(m) {
        // Unmodified keys: the view bindings and the quick-brush rack. Checked
        // here, after the modifier set is known, so `Ctrl+H` stays the browser's
        // and only a bare press is ours.
        if m.contains(Modifiers::ALT) {
            return;
        }
        // A digit holds its brush for as long as it is down (§18.1.8). Shift is
        // not excluded: on most layouts it is what the digit row types under, and
        // a hand resting on it should not silently disarm the rack. `slots::hold`
        // ignores a press while a hold is in flight, which is what makes the key's
        // own auto-repeat harmless.
        if let Some(slot) = slots::of_code(&e.code()) {
            slots::hold(state, slot, Grip::Key);
            e.prevent_default();
        } else if let Key::Character(c) = key_of(e)
            && c.eq_ignore_ascii_case("h")
        {
            // Screen-relative, so it swaps the left of the screen with the right
            // whatever angle the canvas is at (`ViewTransform::mirror_screen_h`).
            dispatch(state, ViewCommand::MirrorH);
            e.prevent_default();
        }
        return;
    }
    match key_of(e) {
        Key::Character(c) if c.eq_ignore_ascii_case("z") => {
            let cmd = if m.contains(Modifiers::SHIFT) {
                DocCommand::Redo
            } else {
                DocCommand::Undo
            };
            edit_history(state, cmd);
            e.prevent_default();
        }
        Key::Character(c) if c.eq_ignore_ascii_case("y") => edit_history(state, DocCommand::Redo),
        // Selection commands (§6.8). "Select all" and "Deselect" are the
        // same edit here — a selection covering the whole canvas *is* no selection —
        // so both shortcuts land on the same op.
        //
        // `prevent_default` whether or not the edit is accepted: the browser's own
        // Ctrl+A would select the page's text, and a refusal that let that through
        // would answer a declined command with a highlighted user interface.
        Key::Character(c) if c.eq_ignore_ascii_case("a") || c.eq_ignore_ascii_case("d") => {
            if may_edit(state) {
                dispatch(state, DocCommand::Select(SelectionOp::select_all()));
            }
            e.prevent_default();
        }
        Key::Character(c) if c.eq_ignore_ascii_case("i") && m.contains(Modifiers::SHIFT) => {
            if may_edit(state) {
                dispatch(state, DocCommand::InvertSelection);
            }
            e.prevent_default();
        }
        _ => {}
    }
}

/// Whether a **document edit** may be accepted from the keyboard right now.
///
/// The two questions the canvas already asks of a press, asked of the shortcuts —
/// which were the one door into the document that asked neither:
///
/// - **The playhead is moving.** A commit clears the withheld half of the
///   timeline, so an edit laid under a running playback deletes the rest of the
///   piece (`crate::panels::timeline`). The canvas refuses a press for this;
///   Ctrl+A went through and truncated the history from the keyboard.
/// - **A mode is composing.** Its preview is computed against the committed
///   document (`crate::modes`), and the bar that carries these very commands has
///   stood down for the mode's own — deselecting mid-transform would move the
///   wrong region on "Done" (`crate::panels::select::SelectionBar`). The keyboard
///   says what the screen says.
fn may_edit(state: AppState) -> bool {
    !crate::panels::timeline::is_playing(state) && !crate::modes::is_composing(state)
}

/// Undo or redo, having first put down whatever was in hand.
///
/// Not [`may_edit`]'s flat refusal, because these two are not refusable in the
/// same sense. Nothing on screen says undo is unavailable — no bar stood down to
/// carry the message — so a shortcut that silently did nothing would read as a
/// broken keyboard rather than as a rule. Editing the history is instead an
/// unambiguous statement that the composition in flight is over, so it ends the
/// way scrubbing ends one: the preview dropped, nothing committed. Playback
/// stops for the same reason it stops when the transport is touched — the hand
/// has taken the playhead back off the loop that was moving it.
fn edit_history(state: AppState, command: DocCommand) {
    crate::panels::timeline::stop(state);
    crate::modes::leave(state);
    dispatch(state, command);
}

fn handle_keyup(mut state: AppState, e: &web_sys::KeyboardEvent) {
    match key_of(e) {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(false);
            e.prevent_default();
        }
        _ => {}
    }
    // The rack's release, named by the slot it lets go of — so a hand rolling
    // from 3 to 4 and off 4 first does not end the hold 3 still has (§18.1.8).
    // Unguarded by `typing_into_a_field` like the two above, and for the same
    // reason: focus can move between a press and its release, and a release that
    // never arrived would leave the brush swapped.
    if let Some(slot) = slots::of_code(&e.code()) {
        slots::release(state, slot, Grip::Key);
    }
    track_alt(state, modifiers_of(e));
}

/// Record whether Alt is held, so the canvas can wear the eyedropper cursor while it
/// is (§18.0.2).
///
/// Read off the event's **modifier set** rather than off the Alt key itself: a
/// keystroke that arrives after Alt was pressed or released while the window was not
/// focused then corrects the flag, instead of leaving it stuck on a press whose
/// release never came. Written only on a change, since every write re-renders the
/// canvas component.
fn track_alt(state: AppState, m: Modifiers) {
    let alt = m.contains(Modifiers::ALT);
    let mut held = state.pick.alt_down;
    if *held.peek() != alt {
        held.set(alt);
    }
}

/// Sample the canvas color under `pos` and load the brush with it — the eyedropper
/// (§18.0.2).
///
/// One sample at a time. A pick is a render plus an asynchronous readback, and
/// Alt+drag asks for one per pointer move, so a move arriving while one is still in
/// flight is **dropped rather than queued**: queueing would spend a GPU submit per
/// pointer move and let an older sample land after a newer one, and for a sampler
/// being dragged only the latest answer matters anyway.
pub fn pick_color(state: AppState, pos: Vec2) {
    let mut busy = state.pick.busy;
    if *busy.peek() {
        return;
    }
    // The *choice* is what the panel holds; which layer it means is resolved here,
    // against whichever layer is selected at the moment of the sample — and a
    // document with no layer selected falls back to the composite rather than
    // sampling nothing.
    let scope = *state.pick.scope.peek();
    let active = state.obs.peek().as_ref().map(|o| o.active_layer);
    let options = PickOptions {
        source: match (scope, active) {
            (PickScope::ThisLayer, Some(id)) => PickSource::Layer(id),
            (PickScope::AllLayersAndCanvas, _) => PickSource::CompositeOverSubstrate,
            _ => PickSource::Composite,
        },
        radius: *state.pick.radius.peek(),
    };

    // Render now and **drop the guard before awaiting** — the readback future owns
    // everything it needs, so nothing holds the renderer while the browser's event
    // loop runs the copy, which it must be free to do since the UI re-renders during
    // it (the same bargain `files::export_png` makes).
    let Some(readback) = crate::state::with_engine_quiet(state, |r| r.pick_color(pos, options))
    else {
        return;
    };
    busy.set(true);
    // Detached: the sample outlives the pointer gesture that asked for it (a release
    // must not cancel the answer to the press), and every signal it writes is
    // root-owned — see `state::root_signal`.
    spawn_forever(async move {
        let picked = readback.await;
        busy.set(false);
        // Nothing under the sampler: leave the brush as it was. Bare canvas is the
        // ground, not paint to pick up.
        let Some(rgb) = picked else { return };
        update_brush(state, |br| br.color = [rgb[0], rgb[1], rgb[2], br.color[3]]);
        // Tell the Color panel the color moved from outside its own picker, so its
        // markers follow (see `AppState::color_epoch`).
        let mut epoch = state.color_epoch;
        let next = *epoch.peek() + 1;
        epoch.set(next);
    });
}

// ---------------------------------------------------------------------------
// The hold, for the drawing assist (§6.9)
// ---------------------------------------------------------------------------

/// How long the pointer has to hold still before the stroke in flight snaps.
///
/// The cost of the two mistakes is asymmetric, and that is what sets it. Too short and
/// the natural pause before lifting the pen turns considered strokes into shapes; too
/// long and the gesture just feels unresponsive and gets tried again. So it sits at the
/// long end of what still reads as immediate — the same figure Procreate settled on.
const DWELL: f64 = 0.45;

/// How far the pointer may drift and still count as held, in **CSS px**: a hand's own
/// tremor on a pen resting against the glass, plus the digitizer's noise.
const DWELL_SLOP: f32 = 4.0;

/// How often the watcher looks. Well under a tenth of [`DWELL`], so the snap lands
/// within a frame or two of the hold being earned, and far too rare to cost anything.
const DWELL_POLL_MS: i32 = 60;

/// Begin watching a stroke gesture for a hold (§6.9). A no-op when the assist is off.
///
/// `at` is the press position in element (CSS) px — the frame the dwell is measured in;
/// see [`Dwell::at`].
pub fn watch_for_hold(state: AppState, at: Vec2) {
    if !*state.assist.enabled.peek() {
        return;
    }
    let mut dwell = state.assist.dwell;
    dwell.set(Some(Dwell {
        at,
        since: now_seconds(),
        fired: false,
    }));
    // `spawn_forever` for the reason `request_paint` uses it: this is started from a
    // component's event handler and must not be tied to that scope's lifetime. Every
    // signal it touches is root-owned (see `state::root_signal`).
    let task = spawn_forever(async move {
        let mut dwell = state.assist.dwell;
        loop {
            sleep_ms(DWELL_POLL_MS).await;
            // Cleared means the gesture is over, and that is the only way out: the
            // watcher runs for as long as the pointer is down, because a hold that was
            // declined (or that snapped a shape now being steered) may be followed by
            // another worth reporting.
            let Some(held) = *dwell.peek() else { return };
            if held.fired || now_seconds() - held.since < DWELL {
                continue;
            }
            // Latch *before* dispatching, so a pointer that simply stays put does not
            // report the same hold thirty times a second.
            dwell.set(Some(Dwell {
                fired: true,
                ..held
            }));
            dispatch(state, GestureCommand::Hold);
        }
    });
    let mut watcher = state.assist.task;
    if let Some(old) = watcher.write().replace(task) {
        old.cancel();
    }
}

/// Report a pointer move against the hold being watched, in element (CSS) px.
///
/// Restarts the clock only when the pointer has actually gone somewhere: a pen resting
/// on glass reports continuously, so treating every report as movement would mean the
/// dwell never completed on the one device the feature exists for.
pub fn pointer_moved(state: AppState, at: Vec2) {
    let mut dwell = state.assist.dwell;
    let Some(held) = *dwell.peek() else { return };
    if held.at.distance(at) > DWELL_SLOP {
        dwell.set(Some(Dwell {
            at,
            since: now_seconds(),
            fired: false,
        }));
    }
}

/// Stop watching for a hold. Harmless when nothing is being watched.
pub fn stop_watching(state: AppState) {
    let mut dwell = state.assist.dwell;
    if dwell.peek().is_some() {
        dwell.set(None);
    }
    let mut task = state.assist.task;
    if let Some(task) = task.write().take() {
        task.cancel();
    }
}

/// The engine's current view, or `None` before WebGPU init has finished.
///
/// Fallible rather than `expect`ing, because the canvas element is in the DOM and
/// taking pointer events from the first frame while [`render::init`](crate::render)
/// is still awaiting its adapter — so "a pointer event with no engine behind it" is
/// an ordinary early state, not a bug. Everything that needs canvas coordinates
/// therefore returns `None` too, and its callers do nothing until there is an engine
/// to do it to.
fn view_of(state: AppState) -> Option<ViewTransform> {
    state.renderer.read().as_ref().map(|r| r.view())
}

/// Pointer position within an element, in CSS pixels.
pub fn elem_xy(e: &Event<PointerData>) -> Vec2 {
    let ElementPoint { x, y, .. } = e.element_coordinates();
    Vec2::new(x as f32, y as f32)
}

/// How finely a pointer report resolves position, in **CSS px**.
///
/// Not a preference — an estimate of the device's grain, which is what the fitter
/// needs in order to tell jitter from detail (see
/// [`PathFitter::with_tolerance`](stark_core::path::PathFitter::with_tolerance)). A
/// mouse walks the screen in whole *physical* pixels, so `1 / devicePixelRatio` CSS
/// px is its floor. A pen or a finger comes off a digitizer that resolves well below
/// the screen it sits under, so what limits those is the hand rather than the API;
/// half a physical pixel is a deliberate under-estimate — too fine only costs a few
/// extra control points, while too coarse rounds off detail that was really there.
fn input_resolution(e: &Event<PointerData>) -> f32 {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio() as f32)
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or(1.0);
    let physical = match e.pointer_type().as_str() {
        "pen" | "touch" => 0.5,
        _ => 1.0,
    };
    physical / dpr
}

/// The fitting tolerance to declare for a gesture starting with `e`, in canvas px:
/// the device's own grain (above) carried through `view`, since canvas space is
/// where the fit measures error.
pub fn input_tolerance_in(view: ViewTransform, e: &Event<PointerData>) -> f32 {
    input_resolution(e) / view.zoom
}

/// [`input_tolerance_in`] against the main canvas's view; `None` before the engine
/// exists.
pub fn input_tolerance(state: AppState, e: &Event<PointerData>) -> Option<f32> {
    Some(input_tolerance_in(view_of(state)?, e))
}

/// The longest smoothing string a brush can ask for, in **screen px** — what
/// `smoothing = 1` means. Screen px because wobble is a fact about the hand:
/// the same tremor spans 64× more canvas zoomed out than in.
const ROPE_MAX_SCREEN_PX: f32 = 160.0;

/// The §6.11 rope a smoothing amount means against `view`, in canvas px: the
/// 0..=1 knob mapped **quadratically** to a screen-px string — so the low end
/// is fine-grained while the top is a real lettering tow — then carried
/// through the view like the tolerance above. Zooming in therefore shrinks the
/// dead zone in canvas terms: the escape hatch from heavy smoothing is the one
/// artists already reach for to do fine work.
///
/// Stated once against an explicit view because two canvases ask: the main
/// canvas below, and the brush editor's preview against its own.
pub fn rope_in(view: ViewTransform, amount: f32) -> f32 {
    let a = amount.clamp(0.0, 1.0);
    a * a * ROPE_MAX_SCREEN_PX / view.zoom
}

/// [`rope_in`] for the live brush against the main canvas's view. Zero (no tow
/// at all) when the amount is zero or there is no view yet.
pub fn input_rope(state: AppState) -> f32 {
    match view_of(state) {
        Some(view) => rope_in(view, *state.smoothing.peek()),
        None => 0.0,
    }
}

/// Refresh the on-screen tow string from the engine (§6.11), converting to the
/// canvas element's own px against the view the stroke holds — a pinch cancels
/// the stroke it interrupts, so a live string never straddles two views. Sets
/// `None` when there is nothing to show, and leaves the signal untouched when
/// nothing changed, so an idle call dirties no scope.
pub fn refresh_tow(state: AppState) {
    let mut tow = state.tow;
    let ui = state.renderer.peek().as_ref().and_then(|r| {
        let t = r.tow_string()?;
        let view = r.view();
        Some(TowUi {
            tip: view.canvas_to_screen(t.tip),
            target: view.canvas_to_screen(t.target),
            rope: t.rope * view.zoom,
        })
    });
    if ui != *tow.peek() {
        tow.set(ui);
    }
}

/// Map an element-relative pointer position to a canvas-space input sample; `None`
/// before the engine exists, since there is no view to map through yet.
pub fn sample(state: AppState, e: &Event<PointerData>) -> Option<InputSample> {
    let view = view_of(state)?;
    Some(InputSample {
        pos: view.screen_to_canvas(elem_xy(e)),
        pressure: e.pressure(),
        // Pen tilt (degrees from vertical, ±90 per axis) → a canvas-space lean vector. The
        // palette knife's deposit reads its component along the stroke direction
        // (§6.2); a mouse reports (0, 0), so the deposit falls back to its constant rate.
        tilt: Vec2::new(e.tilt_x() as f32, e.tilt_y() as f32) / 90.0,
        time: event_time(e),
    })
}

/// The event's own timestamp in seconds — `performance.now()`'s clock, monotonic
/// and shared by every event on the page, which is what [`InputSample::time`]
/// ("for velocity and timelapse") needs. The fitter keys its time channel to the
/// *first* sample it sees, so only differences matter and the origin is free.
/// Zero when the raw event is out of reach (off-wasm), matching the field's
/// default.
fn event_time(e: &Event<PointerData>) -> f64 {
    e.downcast::<web_sys::PointerEvent>()
        .map(|raw| raw.time_stamp() / 1000.0)
        .unwrap_or(0.0)
}

/// Every canvas-space sample a `pointermove` carries, oldest first.
///
/// The browser delivers roughly one `pointermove` per animation frame and folds
/// the reports it withheld — most of what a 120–240 Hz pen produces — into the
/// delivered event's *coalesced* list. Reading that list is what gets the full
/// input rate to the fitter; reading only the event caps every stroke at display
/// rate, whatever the device resolved. Each entry carries its own position,
/// pressure, tilt and timestamp, so the samples land as the hand made them
/// rather than as delivery batched them.
///
/// Coalesced entries are mapped through the target's bounding rect (measured
/// once per delivered event) because their client coordinates are the ones the
/// spec guarantees; the delivered event's own data equals the list's last entry,
/// so nothing is dispatched twice. Falls back to the event itself when there is
/// no list (off-wasm, a synthetic event); `None` before the engine exists, like
/// [`sample`].
pub fn samples(state: AppState, e: &Event<PointerData>) -> Option<Vec<InputSample>> {
    let view = view_of(state)?;
    let coalesced = e.downcast::<web_sys::PointerEvent>().and_then(|raw| {
        use wasm_bindgen::JsCast;
        let rect = raw
            .target()
            .and_then(|t| t.dyn_into::<web_sys::Element>().ok())?
            .get_bounding_client_rect();
        let list = raw.get_coalesced_events();
        (list.length() > 0).then(|| {
            list.iter()
                .filter_map(|v| v.dyn_into::<web_sys::PointerEvent>().ok())
                .map(|c| InputSample {
                    pos: view.screen_to_canvas(Vec2::new(
                        (c.client_x() as f64 - rect.left()) as f32,
                        (c.client_y() as f64 - rect.top()) as f32,
                    )),
                    pressure: c.pressure(),
                    tilt: Vec2::new(c.tilt_x() as f32, c.tilt_y() as f32) / 90.0,
                    time: c.time_stamp() / 1000.0,
                })
                .collect::<Vec<_>>()
        })
    });
    match coalesced {
        Some(list) if !list.is_empty() => Some(list),
        _ => sample(state, e).map(|s| vec![s]),
    }
}

/// End every gesture the canvas can have in hand at once — the paint gesture, the
/// navigation, the brush-tuning drag and the eyedropper — and hand the canvas back,
/// so the floating chrome fades in.
///
/// Four `Copy` values and no `&mut`: each of them owns its own state now, so this
/// is the *order* they are put down in and nothing else. It used to take the paint
/// gesture apart into two signals it borrowed from the component, which is what
/// made "what counts as in flight" a thing two functions had to agree about
/// (`Paint`).
pub fn end_interaction(mut state: AppState, paint: Paint, nav: Nav, tune: Tune) {
    paint.end();
    nav.stop();
    tune.stop();
    // Not a parameter like the three above because the eyedropper's drag flag is
    // shared state, not a gesture object — the options bar reads it (see
    // `PickState`). Nothing to undo, either: a sample already in flight is left to
    // land, since it is the answer to a press the user made.
    state.pick.dragging.set(false);
    state.canvas_active.set(false);
}
