//! The shared pan/zoom/turn bindings (§18.1.7): the one definition of what a
//! second finger, a middle-drag, space-and-drag or a wheel notch does to the view.
//!
//! One gesture object, made by the canvas and by every mode catcher that covers
//! it, so navigation means the same thing wherever the pointer happens to land —
//! composing a transform must not cost the artist the ability to look around.
//!
//! It also owns the **tap**, which is the one thing here that is not navigation:
//! a two- or three-finger episode that never travelled and never lingered is undo
//! or redo (§18.1.11). It lives here because "never travelled" is measured against
//! the same slop a pinch is, and the two must not drift apart —
//! `a_tap_can_never_have_painted` is what says so.

use super::*;
use stark_ui::nav::{self, Mode};

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

/// How long a touch episode — the first finger landing to the last one lifting —
/// may last and still read as a **tap** (§18.1.11), seconds.
///
/// The other half of [`TOUCH_SLOP`]: a tap is short *and* still, and neither test
/// alone is one. Long enough that a deliberate two-finger tap is never missed,
/// short enough that a pair of fingers resting on the glass while the hand thinks
/// is not an undo.
const TAP_TIME: f64 = 0.3;

/// A held pick can never also fire a tap, and this is why: the wait it earns its
/// sample with ([`DWELL`]) is longer than the longest thing a tap may be. Without
/// it, a hold-to-sample that the hand lifted off promptly would undo the stroke
/// before it, having just picked a color — the worst kind of surprise, since the
/// two acts have nothing to do with each other.
///
/// Stated here because it is a relation *between* two constants that were each set
/// for their own unrelated reasons, and neither definition would notice them
/// crossing (§18.1.11). At compile time because it can be: this costs the built
/// binary nothing and cannot be left un-run.
const _: () = assert!(
    DWELL > TAP_TIME,
    "a hold-to-sample the hand lifted promptly would undo the stroke before it"
);

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
    fingers: Signal<Fingers>,
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

/// The fingers on one surface, and the two-finger gesture they are making
/// (§18.1.7).
///
/// The last three fields are about the **episode** — everything between the first
/// finger landing on an empty surface and the last one leaving it — rather than
/// about any one finger, because that is the span a tap is a fact about
/// (§18.1.11). They are reset with the set itself and never mid-gesture: a hand
/// that put a third finger down and took it away again has still made one episode.
#[derive(Clone, Default)]
struct Fingers {
    /// Every finger down, in the order it landed. The order is load-bearing — the
    /// gesture is made by the **first two**, so a third finger joining changes
    /// nothing and a lift re-forms the pair from whoever is left, both without a
    /// jump.
    down: Vec<Contact>,
    /// The gesture in flight. Born when a second finger lands and buried when the
    /// last one lifts — deliberately outliving the *second* finger, so a pinch that
    /// ends with one finger still on the glass keeps panning rather than going dead
    /// under a hand that never left.
    pinch: Option<Pinch>,
    /// When the first finger of this episode landed, on the monotonic clock.
    since: f64,
    /// The furthest any finger of this episode has been from where it landed, page
    /// px. Monotone, and it outlives the finger that earned it — a pair that has
    /// moved has moved, whichever half of it did the moving and whether or not that
    /// half is still down.
    strayed: f32,
    /// The most fingers this episode has had down at once. What makes a two-finger
    /// tap and a three-finger tap different acts rather than the same one counted
    /// at different moments: the count is taken at the episode's widest, not at its
    /// end, where every episode has zero.
    most: usize,
}

/// One finger on the glass: which it is, where it landed, and where it has got to.
///
/// The landing point is kept because a stray is measured from it, and a stray is
/// what tells a tap from a drag ([`TOUCH_SLOP`]).
#[derive(Copy, Clone)]
struct Contact {
    /// The pointer id — how a finger is named, since none of them is *the* pointer.
    id: i32,
    /// Where it landed, page px.
    from: Vec2,
    /// Where it is now, page px.
    at: Vec2,
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
            fingers: use_signal(Fingers::default),
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
    /// one finger of a pinch ends nothing, and a surface that tore down there would
    /// end the gesture on whichever finger the hand happened to raise first.
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
        let mut fingers = self.fingers;
        let mut t = fingers.write();
        t.down.retain(|c| c.id != e.pointer_id());
        if !t.down.is_empty() {
            return true;
        }
        let tapped = tap_of(t.strayed, now_seconds() - t.since, t.most);
        // The whole record and not just the pinch, because the last three fields are
        // the *episode's* and the episode is what just ended. Clearing them here
        // rather than leaving it to the next primary touch is what keeps a stray
        // earned by a stroke from arriving as the next gesture's deadzone already
        // spent — `finger_down`'s clearing stays as the backstop it was written to
        // be, for the release that never comes at all.
        *t = Fingers::default();
        // Released before the write below: nothing that reads this surface's
        // fingers should be able to find them half-judged.
        drop(t);
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
        if !fingers.peek().down.is_empty() {
            fingers.set(Fingers::default());
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
    /// gesture becomes navigation.
    fn finger_down(self, e: &Event<PointerData>) -> bool {
        // Read before the fingers are locked, so nothing holds two signals at once.
        let angle = view_of(self.state).map_or(0.0, |v| v.rotation);
        let now = now_seconds();
        let mut fingers = self.fingers;
        let mut t = fingers.write();
        // A *primary* touch is the first contact of its type, so anything still
        // listed when one arrives is a finger whose release never came — a cancel the
        // browser swallowed, a tab switched away from mid-gesture. Cleared on the
        // fact that says so rather than by trying to catch every way a release can go
        // missing, since a single stale entry would make the next lone finger a pinch
        // and painting by touch would stop working for the rest of the session.
        if e.is_primary() {
            *t = Fingers::default();
        }
        // The episode's clock starts on the finger that finds the surface empty, not
        // on the pair: what a tap has to be short is the *whole* touch, or a finger
        // that had been painting for a minute could be turned into an undo by a
        // second one arriving and both lifting quickly.
        if t.down.is_empty() {
            t.since = now;
        }
        let id = e.pointer_id();
        if !t.down.iter().any(|c| c.id == id) {
            let at = page_xy(e);
            t.down.push(Contact { id, from: at, at });
            t.most = t.most.max(t.down.len());
        }
        if t.down.len() < 2 {
            return false; // one finger is the caller's — it paints, or it waits
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
        let mut fingers = self.fingers;
        let mut t = fingers.write();
        let id = e.pointer_id();
        let Some(i) = t.down.iter().position(|c| c.id == id) else {
            return false;
        };
        let now = page_xy(e);
        let was = std::mem::replace(&mut t.down[i].at, now);
        // Recorded whether or not there is a gesture behind it, for the reason the
        // positions themselves are: a lone finger's travel is what a second finger
        // landing has to be judged against.
        let strayed = now.distance(t.down[i].from);
        t.strayed = t.strayed.max(strayed);
        let Some(mut pinch) = t.pinch else {
            return false; // a lone finger with no gesture behind it: painting
        };
        // The pair's own deadzone, and the same one the lone finger has (§18.1.11).
        // Two fingers land milliseconds apart and roll as they settle, so a pair
        // believed immediately nudges the canvas every time it is put down — and a
        // canvas that shifts a pixel under a tap is a canvas that cannot be tapped.
        // Spent once and never re-earned, like [`TWIST_DEADZONE`]: what the band
        // costs is a couple of mm at the start of the first pan, not a jump.
        if t.strayed <= TOUCH_SLOP {
            return true;
        }

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
            let (a, b) = (t.down[0].at, t.down[1].at);
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

/// The tap a finished touch episode made, as the number of fingers at its widest —
/// or `None` if it was not one (§18.1.11).
///
/// A tap is defined by what it *failed* to do. It never travelled far enough to move
/// the view or to open a stroke — [`TOUCH_SLOP`] is the same threshold in all three
/// places, so "this episode painted nothing" and "this episode was a tap" are one
/// fact rather than two that have to agree — and it did not linger, which is what
/// separates a tap from a hand parked on the glass while its owner thinks.
///
/// The count is the episode's widest and not its last, since every episode ends with
/// no fingers down. Free rather than a method for the reason the file's other
/// arithmetic is: it is a decision about three numbers, and a decision about three
/// numbers can be *read*.
fn tap_of(strayed: f32, held: f64, most: usize) -> Option<usize> {
    (strayed <= TOUCH_SLOP && held <= TAP_TIME).then_some(most)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four ways an episode can end, and only one of them is a tap.
    #[test]
    fn a_tap_is_short_and_still() {
        assert_eq!(tap_of(0.0, 0.05, 2), Some(2));
        assert_eq!(tap_of(TOUCH_SLOP, TAP_TIME, 2), Some(2)); // both ends inclusive
        assert_eq!(tap_of(TOUCH_SLOP + 0.1, 0.05, 2), None); // travelled: a pinch
        assert_eq!(tap_of(0.0, TAP_TIME + 0.01, 2), None); // lingered: a rest
    }

    /// The count is the episode's widest, so a hand that put a third finger down
    /// and took it off again asked for redo rather than for undo — the fingers a
    /// gesture *had* are what it meant, not the ones it ended holding.
    #[test]
    fn a_tap_is_counted_at_its_widest() {
        assert_eq!(tap_of(0.0, 0.05, 3), Some(3));
        assert_eq!(tap_of(0.0, 0.05, 1), Some(1));
    }

    /// And the other half of the same guarantee, which is what makes spending a
    /// two-finger tap on undo safe: an episode that stayed inside the slop never
    /// opened a stroke, because opening one is what crossing the slop *is*
    /// ([`Landing::advance`]). One constant, so the two cannot drift apart.
    #[test]
    fn a_tap_can_never_have_painted() {
        let painted = |strayed: f32| strayed > TOUCH_SLOP;
        for strayed in [0.0, 1.0, TOUCH_SLOP, TOUCH_SLOP + 0.1, 100.0] {
            assert_ne!(tap_of(strayed, 0.05, 2).is_some(), painted(strayed));
        }
    }
}
