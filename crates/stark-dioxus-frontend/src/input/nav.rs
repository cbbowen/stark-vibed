//! The shared pan/zoom/turn bindings (§18.1.7): the one definition of what a
//! second finger, a middle-drag, space-and-drag or a wheel notch does to the view.
//!
//! One gesture object, made by the canvas and by every mode catcher that covers
//! it, so navigation means the same thing wherever the pointer happens to land —
//! composing a transform must not cost the artist the ability to look around.
//!
//! What the presses and the fingers *mean* is `stark_ui::nav`'s: which press pans
//! and which zooms, and the pinch and the tap two fingers make ([`Touch`], §18.1.11).
//! What is here is reading a DOM event for them, taking the pointer, and the signals
//! the state lives in.

use super::*;
use stark_ui::nav::{self, Lift, Mode, Moved, Touch};

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
///
/// A fourth answer is on offer and nobody has to take it: fingers that came and went
/// without moving the view made a **tap**, which [`take_tap`](Self::take_tap) reports
/// and the canvas alone spends (§18.1.11).
#[derive(Clone, Copy)]
pub struct Nav {
    state: AppState,
    /// The one-pointer drag in flight, or `None`.
    drag: Signal<Option<Drag>>,
    /// The fingers on this surface (§18.1.7). Separate from `drag` because a
    /// finger is identified by its id rather than by being *the* pointer — that is
    /// the whole difference touch makes.
    fingers: Signal<Touch>,
    /// The tap the last release turned out to be, waiting to be spent
    /// ([`Nav::take_tap`], §18.1.11). Written on every episode that ends, so it
    /// can never be older than the last hand off the glass.
    tap: Signal<Option<usize>>,
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

impl Nav {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_nav(state: AppState) -> Self {
        Self {
            state,
            drag: use_signal(|| None),
            fingers: use_signal(Touch::default),
            tap: use_signal(|| None),
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
        // Which press means what is `stark_ui::nav`'s, so the two frontends
        // cannot come to disagree about what a middle-drag is. What stays here is
        // reading a DOM event for the three facts it takes.
        let button = match e.trigger_button() {
            Some(MouseButton::Auxiliary) => Some(nav::Button::Middle),
            _ if is_contact(e) => Some(nav::Button::Left),
            _ => None,
        };
        let mode = button.and_then(|button| {
            nav::press(
                button,
                page_xy(e),
                *self.state.space_down.peek(),
                accel(e.modifiers()),
            )
        });
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
        let command = in_flight.mode.moved(in_flight.last, p);
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
    /// one finger of a pinch ends nothing.
    ///
    /// Always `false` for a mouse or a pen, which have nothing to be the rest of.
    ///
    /// The release that empties the surface is also where the episode is *judged*:
    /// a hand that came and went without ever meaning anything by it made a tap,
    /// which [`take_tap`](Self::take_tap) hands to whoever asked (§18.1.11).
    pub fn release(self, e: &Event<PointerData>) -> bool {
        if !is_finger(e) {
            return false;
        }
        let now = now_seconds();
        let mut fingers = self.fingers;
        // Its own statement, so the fingers are released before the write below:
        // nothing that reads them should find them half-judged.
        let lift = fingers.write().finger_up(e.pointer_id(), now);
        let Lift::Ended { tap: tapped } = lift else {
            return true;
        };
        let mut tap = self.tap;
        tap.set(tapped);
        false
    }

    /// The tap the last episode turned out to be — the number of fingers at its
    /// widest — and spend it, so one tap is acted on once (§18.1.11).
    ///
    /// Asked by the canvas alone, which is the *policy* half of this file's split:
    /// [`Nav`] can say that a pair of fingers came and went without meaning
    /// anything, and only the surface they came and went on can say what that is
    /// worth. Over the transform box or the gradient trace it is worth nothing, and
    /// those surfaces simply never ask.
    pub fn take_tap(self) -> Option<usize> {
        let taken = *self.tap.peek();
        let mut tap = self.tap;
        if taken.is_some() {
            tap.set(None);
        }
        taken
    }

    /// End the navigation in flight, whatever it was. Harmless when there is none.
    pub fn stop(self) {
        let mut drag = self.drag;
        if drag.peek().is_some() {
            drag.set(None);
        }
        let mut fingers = self.fingers;
        if !fingers.peek().is_idle() {
            fingers.set(Touch::default());
        }
        // A tap nobody spent is dropped here rather than kept: this is the canvas
        // being put down, and an undo that fired on the *next* hand off the glass
        // would be an act with no gesture behind it.
        let mut tap = self.tap;
        if tap.peek().is_some() {
            tap.set(None);
        }
    }

    /// Cursor-anchored wheel zoom. Anchored by page position: it equals the
    /// canvas's own coordinates for full-viewport surfaces, and it is the only
    /// frame an element like the transform box (whose local coordinates move
    /// with it) can meaningfully report.
    pub fn wheel(self, e: Event<WheelData>) {
        e.prevent_default();
        e.stop_propagation();
        // A browser reports the wheel downward-positive, like a document being
        // scrolled; `nav::wheel` takes notches the way a hand turns them, so the sign
        // is flipped at this edge rather than in the shared rule.
        let dy = e.delta().strip_units().y;
        let p = e.page_coordinates();
        let anchor = Vec2::new(p.x as f32, p.y as f32);
        if let Some(command) = nav::wheel(anchor, -dy.signum() as f32) {
            dispatch(self.state, command);
        }
    }

    /// A finger landing. `true` once there are two of them, which is where the
    /// gesture becomes navigation and this surface takes the pointer.
    fn finger_down(self, e: &Event<PointerData>) -> bool {
        // Read before the fingers are locked, so nothing holds two signals at once.
        let angle = view_of(self.state).map_or(0.0, |v| v.rotation);
        let (at, now) = (page_xy(e), now_seconds());
        let mut fingers = self.fingers;
        let pinching = fingers
            .write()
            .finger_down(e.pointer_id(), at, e.is_primary(), angle, now);
        if pinching {
            e.prevent_default();
            e.stop_propagation();
            capture_pointer(e);
        }
        pinching
    }

    /// A finger moving: drives the view once a gesture is in flight, and before then
    /// only records where the finger is.
    fn finger_move(self, e: &Event<PointerData>) -> bool {
        let mut fingers = self.fingers;
        // Its own statement: the command re-enters the engine and rewrites the
        // frontend's observable, and nothing that runs there should be able to find
        // this surface's fingers half-updated.
        let moved = fingers.write().finger_move(e.pointer_id(), page_xy(e));
        match moved {
            Moved::Surface => false,
            Moved::Navigation(command) => {
                if let Some(command) = command {
                    dispatch(self.state, *command);
                }
                true
            }
        }
    }
}
