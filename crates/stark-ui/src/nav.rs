//! What a press, a drag and a wheel notch do to the **view** (§18.1.7).
//!
//! Navigation is the one gesture family that has to mean the same thing wherever the
//! pointer lands — composing a transform must not cost the artist the ability to look
//! around — and now that there are two frontends, "wherever" includes "whichever
//! app". So the decisions and the rates are here: which press is a pan and which is a
//! zoom, how far a scrubby drag travels per doubling, and how much a notch of the
//! wheel is worth.
//!
//! The two-finger gesture is here too, as [`Touch`]: a finger set keyed by pointer id,
//! the pinch it makes and the tap it may turn out to have been (§18.1.11). Only the web
//! has fingers today, but what a pinch *is* is not a fact about a toolkit. What stays in
//! each app is reading its own event and where it keeps the state — a `Signal` on the
//! web, one `Held` natively.

use stark_engine::ViewTransform;
use stark_engine::command::ViewCommand;
use stark_model::geom::Vec2;

use crate::input::TOUCH_SLOP;

/// How far the accelerator+space drag travels to **double** the zoom, in screen px
/// (§18.1.9).
///
/// Set from the range it has to cover rather than by taste: the view's whole zoom
/// range is about ten doublings (`ViewTransform::MIN_ZOOM`..`MAX_ZOOM`), so at this
/// rate a sweep of roughly one screen width takes the canvas from as far out as it
/// goes to as far in — reachable in one gesture, without a short drag overshooting
/// the picture.
pub const ZOOM_DRAG_DOUBLE: f32 = 180.0;

/// What one notch of the wheel multiplies the zoom by.
///
/// A ratio rather than a step, for [`ZOOM_DRAG_DOUBLE`]'s reason: zoom is
/// multiplicative, so a fixed addition would crawl when zoomed out and leap when
/// zoomed in. About five notches to the doubling, which is fine enough to land on a
/// size deliberately and coarse enough to cross the range without the hand tiring.
pub const WHEEL_STEP: f32 = 1.15;

/// A wheel report that names lines rather than pixels — a mouse notch — is worth
/// this many of whatever the platform counts.
///
/// Only the *sign and count* of the notches matter to a zoom, so this exists to give
/// a trackpad's pixel deltas a comparable scale rather than to be a measurement:
/// a pixel-denominated surface reports tens of units per notch, and dividing by this
/// puts the two within reach of one another.
pub const WHEEL_PIXELS_PER_NOTCH: f32 = 40.0;

/// Which mouse button a press came from — the two navigation cares about.
///
/// A vocabulary of its own rather than either toolkit's, which is what lets this
/// crate name neither of them — a claim `tests::no_toolkit_types` checks by reading
/// the source, and which this sentence had to be reworded to keep. Both apps map
/// their own button onto it at the edge, the bargain `keys::Mods` makes for a
/// modifier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Button {
    /// The one that paints.
    Left,
    /// The wheel pressed in.
    Middle,
}

/// What a navigation drag in flight is doing.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    /// Move the view under the hand.
    Pan,
    /// Scale it about the point the press landed on.
    ///
    /// The anchor is the *press*, not the current pointer, which is what makes the
    /// gesture reversible: the zoom is a function of how far the hand has travelled
    /// from where it started, so a drag that wanders out and back leaves the canvas
    /// where it found it.
    Zoom { anchor: Vec2 },
}

/// What this press means for the view, or `None` if it means nothing and the surface
/// should keep it.
///
/// - **Middle button**: pan, whatever else is held. It is the binding for a hand
///   already on the mouse, and there is no second gesture there for a modifier to
///   pick out.
/// - **Space**: pan, which is the binding for a hand already on the keyboard.
/// - **Space + accelerator**: the scrubby zoom (§18.1.9).
///
/// Decided at the press and held for the whole gesture by the caller: a drag is what
/// it was begun as, so letting go of the accelerator halfway through a zoom does not
/// hand the canvas to the pan mid-motion, under a hand that is still moving.
pub fn press(button: Button, at: Vec2, space: bool, accel: bool) -> Option<Mode> {
    match button {
        Button::Middle => Some(Mode::Pan),
        Button::Left if space && accel => Some(Mode::Zoom { anchor: at }),
        Button::Left if space => Some(Mode::Pan),
        Button::Left => None,
    }
}

impl Mode {
    /// The command a move from `from` to `to` asks for, or `None` when it asks for
    /// nothing — which a zoom that has not travelled along its own axis does, and
    /// dispatching one would repaint the canvas to leave it exactly as it was.
    pub fn moved(self, from: Vec2, to: Vec2) -> Option<ViewCommand> {
        match self {
            // Incremental, so the caller re-sets its anchor on every move.
            Mode::Pan => Some(ViewCommand::Pan { delta: to - from }),
            Mode::Zoom { anchor } => {
                // Right and up both zoom in — screen y grows downward, which is the
                // whole of why the second term is subtracted. **Summed** rather than
                // projected onto the diagonal, so a drag along either axis alone runs
                // at exactly the documented rate and one that asks for both gets both.
                //
                // Exponential in that distance, which is what makes the gesture feel
                // the same at every zoom level.
                let step = to - from;
                let travel = step.x - step.y;
                (travel != 0.0).then(|| ViewCommand::Zoom {
                    anchor,
                    factor: (travel / ZOOM_DRAG_DOUBLE).exp2(),
                })
            }
        }
    }
}

/// The cursor-anchored zoom a wheel report asks for, or `None` for one that scrolled
/// nowhere vertically.
///
/// `notches` is positive for a wheel turned *away* from the hand, which every
/// platform reports as scrolling up and which zooms **in** — the convention every
/// canvas application shares, and the opposite of what a document would do.
pub fn wheel(anchor: Vec2, notches: f32) -> Option<ViewCommand> {
    (notches != 0.0).then(|| ViewCommand::Zoom {
        anchor,
        factor: WHEEL_STEP.powf(notches),
    })
}

/// How close to a quarter turn a turn has to land to be pulled onto it, radians (about
/// 5°).
///
/// Without it a turned canvas could only be *approximately* straightened, and a piece
/// left a degree off square reads as an accident rather than as a choice.
pub const TURN_SNAP: f32 = 0.09;

/// `to` pulled onto the nearest quarter turn if it is within [`TURN_SNAP`] of one.
pub fn snap_quarter(to: f32) -> f32 {
    use std::f32::consts::FRAC_PI_2;
    let quarter = (to / FRAC_PI_2).round() * FRAC_PI_2;
    if (to - quarter).abs() <= TURN_SNAP {
        quarter
    } else {
        to
    }
}

/// The signed turn from `from` to `to` the short way round, so easing between two angles
/// never goes the long way about.
pub fn shortest_turn(from: f32, to: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (to - from + PI).rem_euclid(TAU) - PI
}

/// How far a navigator turn-drag has to be pulled before the canvas follows it exactly, in
/// miniature px.
///
/// Near the press the direction of a two-pixel vector is noise, and following it exactly
/// snaps the canvas to a wild angle as the button goes down; short of this the turn eases
/// in with the pull.
pub const TURN_FOLLOW_PX: f32 = 64.0;

/// The angle a navigator turn-drag of `pull` — miniature px from the press — asks the
/// canvas to be at, having started at `was`; `None` for a drag that has gone nowhere.
///
/// The snap is applied to the *target*, so a long pull lands exactly square while a short
/// one still eases toward it. The miniature is an upright, uniformly scaled picture of
/// canvas space, so a direction in its px is a direction on the canvas.
pub fn turn_to(view: ViewTransform, pull: Vec2, was: f32) -> Option<f32> {
    let target = snap_quarter(view.rotation_for_up(pull)?);
    let ease = (pull.length() / TURN_FOLLOW_PX).clamp(0.0, 1.0);
    Some(was + ease * shortest_turn(was, target))
}

/// How far a two-finger gesture has to twist before it turns the canvas at all, radians
/// (about 6°).
///
/// Two fingers closing on a target roll about the hand, and without a band to spend that
/// in every zoom would leave the canvas a couple of degrees off true. Subtracted once
/// crossed, so the turn picks up from where the hand is instead of jumping by the band.
pub const TWIST_DEADZONE: f32 = 0.10;

/// How far apart two fingers have to be for the pair to mean anything, in screen px.
///
/// Closer than this the pair's direction is noise and its length a divisor — an arbitrary
/// rotation and an unbounded zoom — so the fingers simply slip.
pub const MIN_SPAN: f32 = 8.0;

/// How long a touch episode — the first finger landing to the last one lifting — may last
/// and still be a **tap** (§18.1.11), seconds.
///
/// The other half of [`TOUCH_SLOP`]: a tap is short *and* still. Short enough that a pair
/// of fingers resting on the glass while the hand thinks is not an undo.
pub const TAP_TIME: f64 = 0.3;

// A held pick can never also fire a tap: the wait it earns its sample with is longer than
// the longest a tap may be. Two constants set for unrelated reasons, so the relation is
// asserted where it cannot be left un-run (§18.1.11).
const _: () = assert!(
    crate::input::DWELL > TAP_TIME,
    "a hold-to-sample the hand lifted promptly would undo the stroke before it"
);

/// The fingers on one surface, and the two-finger gesture they are making (§18.1.7).
///
/// Keyed by pointer id, since no finger is *the* pointer. The last three fields are the
/// **episode**'s — the first finger landing on an empty surface to the last one leaving —
/// which is the span a tap is a fact about (§18.1.11), so they reset with the set and never
/// mid-gesture.
#[derive(Clone, Debug, Default)]
pub struct Touch {
    /// Every finger down, in the order it landed. The gesture is made by the **first
    /// two**, so a third joining changes nothing and a lift re-forms the pair from whoever
    /// is left, both without a jump.
    down: Vec<Contact>,
    /// Born when a second finger lands and buried when the last one lifts — outliving the
    /// second finger, so a pinch that ends with one finger still down keeps panning.
    pinch: Option<Pinch>,
    /// When the episode's first finger landed, seconds on a monotonic clock.
    since: f64,
    /// The furthest any finger of the episode has been from where it landed. Monotone, and
    /// it outlives the finger that earned it.
    strayed: f32,
    /// The most fingers down at once — what makes a two- and a three-finger tap different
    /// acts, since every episode ends with none.
    most: usize,
}

/// One finger on the glass. Where it landed is kept because a stray is measured from it.
#[derive(Clone, Copy, Debug)]
struct Contact {
    id: i32,
    from: Vec2,
    at: Vec2,
}

/// What a two-finger gesture accumulates that a pair of positions cannot say.
#[derive(Clone, Copy, Debug)]
struct Pinch {
    /// The view's angle when the gesture began.
    from: f32,
    /// Raw twist since then, **before** the deadzone — so twisting back out of the band
    /// un-turns by exactly what twisting in did, where accumulating the deadzoned angle
    /// would ratchet.
    twist: f32,
    /// The angle last asked for. Each report is the step from here, so the snap's pull
    /// onto a quarter turn is spent once instead of re-applied every move.
    asked: f32,
}

/// What one finger's move was, to the surface it moved on.
#[derive(Clone, Debug)]
pub enum Moved {
    /// The surface's own: a finger this set does not hold, or a lone finger with no
    /// gesture behind it.
    Surface,
    /// Navigation, asking the view for this — or for nothing yet: inside the slop, a
    /// bystanding third finger, a pair too close to measure. Boxed, because a view
    /// command is hundreds of bytes and most moves are a lone finger's `Surface`.
    Navigation(Option<Box<ViewCommand>>),
}

/// What a finger leaving did to the episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lift {
    /// Fingers are still down, so nothing ends: a surface that tore down here would end a
    /// pinch on whichever finger the hand happened to raise first.
    Continuing,
    /// The last finger left. `tap` is the fingers at the episode's widest, if it was a tap
    /// (§18.1.11).
    Ended { tap: Option<usize> },
}

impl Touch {
    /// Finger `id` landing at `at`. `true` once there are two — where the gesture becomes
    /// navigation, and the surface should take the pointer.
    ///
    /// `primary` is the platform saying this is the first contact of its type, so anything
    /// still listed is a finger whose release never came; one stale entry would make every
    /// lone finger after it a pinch. `angle` is the view's rotation, which a twist is
    /// measured from, and `now` is seconds on a monotonic clock.
    pub fn finger_down(&mut self, id: i32, at: Vec2, primary: bool, angle: f32, now: f64) -> bool {
        if primary {
            *self = Self::default();
        }
        // The clock starts on the finger that finds the surface empty: what a tap has to
        // be short is the whole touch, or a finger that had painted for a minute could be
        // turned into an undo by a second one landing and both lifting quickly.
        if self.down.is_empty() {
            self.since = now;
        }
        if !self.down.iter().any(|c| c.id == id) {
            self.down.push(Contact { id, from: at, at });
            self.most = self.most.max(self.down.len());
        }
        if self.down.len() < 2 {
            return false;
        }
        if self.pinch.is_none() {
            self.pinch = Some(Pinch {
                from: angle,
                twist: 0.0,
                asked: angle,
            });
        }
        true
    }

    /// Finger `id` moving to `at`.
    ///
    /// Recorded whether or not a gesture is in flight, because a second finger landing
    /// pairs with where the first one *is* rather than where it pressed — so a surface
    /// asks this of every finger's move, not only of those a gesture has left over.
    pub fn finger_move(&mut self, id: i32, at: Vec2) -> Moved {
        let Some(i) = self.down.iter().position(|c| c.id == id) else {
            return Moved::Surface;
        };
        let was = std::mem::replace(&mut self.down[i].at, at);
        self.strayed = self.strayed.max(at.distance(self.down[i].from));
        let Some(mut pinch) = self.pinch else {
            return Moved::Surface;
        };
        // The pair's deadzone is the lone finger's slop (§18.1.11): two fingers land
        // milliseconds apart and roll as they settle, and a canvas that shifts under a tap
        // is a canvas that cannot be tapped. Spent once and never re-earned.
        if self.strayed <= TOUCH_SLOP {
            return Moved::Navigation(None);
        }
        if self.down.len() < 2 {
            // Down to the gesture's last finger: a plain pan, until the hand itself leaves.
            return Moved::Navigation(Some(Box::new(ViewCommand::Pan { delta: at - was })));
        }
        // A third finger is a bystander, swallowed so a hand resting on the glass does not
        // fight the two doing the work.
        if i > 1 {
            return Moved::Navigation(None);
        }
        // The pair as it was and as it is. One finger moved, so the other side is the same
        // in both.
        let (a, b) = (self.down[0].at, self.down[1].at);
        let (before, after) = if i == 0 {
            ((was, b), (a, b))
        } else {
            ((a, was), (a, b))
        };
        let (u, v) = (before.1 - before.0, after.1 - after.0);
        let (span, spans) = (u.length(), v.length());
        if span < MIN_SPAN || spans < MIN_SPAN {
            return Moved::Navigation(None);
        }
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
        self.pinch = Some(pinch);
        Moved::Navigation(Some(Box::new(command)))
    }

    /// Finger `id` leaving, at `now`.
    ///
    /// The release that empties the surface is where the episode is judged, and the whole
    /// record goes with it — a stray earned by a stroke must not arrive as the next
    /// gesture's deadzone already spent.
    pub fn finger_up(&mut self, id: i32, now: f64) -> Lift {
        self.down.retain(|c| c.id != id);
        if !self.down.is_empty() {
            return Lift::Continuing;
        }
        let tap = tap_of(self.strayed, now - self.since, self.most);
        *self = Self::default();
        Lift::Ended { tap }
    }

    /// Whether no finger is down.
    pub fn is_idle(&self) -> bool {
        self.down.is_empty()
    }
}

/// The tap a finished episode made, as the fingers at its widest, or `None` (§18.1.11).
///
/// Defined by what it failed to do: it never strayed past [`TOUCH_SLOP`] — the threshold
/// that moves the view and opens a stroke, so "painted nothing" and "was a tap" are one
/// fact — and it did not linger past [`TAP_TIME`].
pub(crate) fn tap_of(strayed: f32, held: f64, most: usize) -> Option<usize> {
    (strayed <= TOUCH_SLOP && held <= TAP_TIME).then_some(most)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_engine::Extent2;

    /// Two fingers `span` apart along x, landing at time zero on an unturned view.
    fn pair(span: f32) -> Touch {
        let mut touch = Touch::default();
        assert!(!touch.finger_down(1, Vec2::ZERO, true, 0.0, 0.0));
        assert!(touch.finger_down(2, Vec2::new(span, 0.0), false, 0.0, 0.0));
        touch
    }

    /// The scale and turn a move asked for, which must have been a pinch.
    fn pinch(moved: Moved) -> (f32, f32) {
        let Moved::Navigation(Some(command)) = moved else {
            panic!("expected a pinch, got {moved:?}");
        };
        match *command {
            ViewCommand::Pinch { scale, turn, .. } => (scale, turn),
            other => panic!("expected a pinch, got {other:?}"),
        }
    }

    /// Each report scales by the pair's own step, so over a gesture the zoom is the span
    /// the pair has now over the span it started from.
    #[test]
    fn a_pinch_scales_by_the_span() {
        let mut touch = pair(100.0);
        let mut scale = 1.0;
        for x in [200.0, 250.0, 400.0] {
            scale *= pinch(touch.finger_move(2, Vec2::new(x, 0.0))).0;
        }
        assert!((scale - 4.0).abs() < 1e-5, "{scale}");
    }

    /// A twist inside the band turns nothing; past it the band is spent once, off the whole
    /// twist rather than off each step; and out and back returns the canvas to where it was.
    ///
    /// Landed off a quarter turn: landed on one, the snap pulled every small answer back to
    /// the starting angle, and this passed with no band at all.
    #[test]
    fn a_twist_spends_its_band_once_and_out_and_back_returns() {
        let angle: f32 = 0.5;
        let reach = angle + TWIST_DEADZONE * 3.0;
        assert_eq!(snap_quarter(angle), angle, "clear of the snap");
        assert_eq!(snap_quarter(reach), reach, "clear of the snap");
        let mut touch = Touch::default();
        assert!(!touch.finger_down(1, Vec2::ZERO, true, angle, 0.0));
        assert!(touch.finger_down(2, Vec2::new(100.0, 0.0), false, angle, 0.0));
        // Out past the slop along the pair first, so what follows is twist alone.
        let (_, along) = pinch(touch.finger_move(2, Vec2::new(150.0, 0.0)));
        assert_eq!(along, 0.0);
        let at = |angle: f32| Vec2::from_angle(angle) * 150.0;
        let (_, inside) = pinch(touch.finger_move(2, at(TWIST_DEADZONE * 0.5)));
        assert_eq!(inside, 0.0, "inside the band");
        // Three bands of twist ask for two; a band spent per step would leave one and a
        // half. The tolerance is for `angle_to`'s `acos`, which is coarse near small angles.
        let (_, out) = pinch(touch.finger_move(2, at(TWIST_DEADZONE * 3.0)));
        assert!((out - TWIST_DEADZONE * 2.0).abs() < 1e-4, "{out}");
        let (_, back) = pinch(touch.finger_move(2, at(0.0)));
        assert!(
            (out + back).abs() < 1e-4,
            "back where it started: {out} + {back}"
        );
    }

    /// A third finger is a bystander: its moves are the gesture's and ask for nothing,
    /// while the first two go on pinching.
    #[test]
    fn a_third_finger_asks_for_nothing() {
        let mut touch = pair(100.0);
        assert!(touch.finger_down(3, Vec2::new(0.0, 100.0), false, 0.0, 0.0));
        assert!(matches!(
            touch.finger_move(2, Vec2::new(200.0, 0.0)),
            Moved::Navigation(Some(_))
        ));
        assert!(matches!(
            touch.finger_move(3, Vec2::new(0.0, 300.0)),
            Moved::Navigation(None)
        ));
    }

    /// A pair closer than it can be measured asks for nothing, rather than an arbitrary
    /// turn and an unbounded zoom.
    #[test]
    fn a_pair_too_close_to_measure_asks_for_nothing() {
        let mut touch = pair(MIN_SPAN * 0.5);
        let away = Vec2::new(MIN_SPAN * 0.5 + TOUCH_SLOP * 2.0, 0.0);
        assert!(matches!(
            touch.finger_move(2, away),
            Moved::Navigation(None)
        ));
    }

    /// Lifting one finger of a pinch ends nothing, and the one left pans the view under it.
    #[test]
    fn lifting_one_of_two_goes_on_panning() {
        let mut touch = pair(100.0);
        pinch(touch.finger_move(2, Vec2::new(200.0, 0.0)));
        assert_eq!(touch.finger_up(1, 0.1), Lift::Continuing);
        let Moved::Navigation(Some(command)) = touch.finger_move(2, Vec2::new(210.0, 5.0)) else {
            panic!("the last finger of a pinch still navigates");
        };
        match *command {
            ViewCommand::Pan { delta } => assert_eq!(delta, Vec2::new(10.0, 5.0)),
            other => panic!("expected a pan, got {other:?}"),
        }
    }

    /// A primary touch is the first of its type, so fingers still listed when one lands
    /// never lifted — and are forgotten, rather than making the lone finger a pinch.
    #[test]
    fn a_primary_touch_clears_stale_fingers() {
        let mut touch = pair(100.0);
        assert!(!touch.finger_down(9, Vec2::new(50.0, 50.0), true, 0.0, 5.0));
        assert!(matches!(
            touch.finger_move(1, Vec2::new(90.0, 0.0)),
            Moved::Surface
        ));
        assert!(matches!(
            touch.finger_move(9, Vec2::new(90.0, 90.0)),
            Moved::Surface
        ));
    }

    /// A quick, still pair is judged on the lift that empties the surface.
    #[test]
    fn a_quick_still_pair_lifts_as_a_tap() {
        let mut touch = pair(100.0);
        assert_eq!(touch.finger_up(2, 0.1), Lift::Continuing);
        assert_eq!(touch.finger_up(1, 0.1), Lift::Ended { tap: Some(2) });
        assert!(touch.is_idle());
    }

    /// The clock starts on the finger that finds the surface empty and no later one, so a
    /// long press joined by a quick second finger is not a two-finger tap.
    #[test]
    fn a_late_second_finger_does_not_restart_the_clock() {
        let mut touch = Touch::default();
        assert!(!touch.finger_down(1, Vec2::ZERO, true, 0.0, 0.0));
        assert!(touch.finger_down(2, Vec2::new(100.0, 0.0), false, 0.0, 5.0));
        assert_eq!(touch.finger_up(2, 5.1), Lift::Continuing);
        assert_eq!(touch.finger_up(1, 5.1), Lift::Ended { tap: None });
    }

    /// Lifting one of the pair re-forms it from the next finger down, from where that
    /// finger *is*: a bystander's moves are recorded, so the new pair's first step scales
    /// by that step rather than jumping back to where the finger landed.
    #[test]
    fn a_lift_re_forms_the_pair_without_a_jump() {
        let mut touch = pair(100.0);
        assert!(touch.finger_down(3, Vec2::new(0.0, 100.0), false, 0.0, 0.0));
        pinch(touch.finger_move(2, Vec2::new(200.0, 0.0)));
        assert!(matches!(
            touch.finger_move(3, Vec2::new(0.0, 300.0)),
            Moved::Navigation(None)
        ));
        assert_eq!(touch.finger_up(1, 0.1), Lift::Continuing);
        let (scale, turn) = pinch(touch.finger_move(3, Vec2::new(0.0, 301.0)));
        let step = Vec2::new(-200.0, 301.0).length() / Vec2::new(-200.0, 300.0).length();
        assert!((scale - step).abs() < 1e-5, "{scale} for a step of {step}");
        assert_eq!(turn, 0.0);
    }

    /// The four ways an episode can end, and only one of them is a tap.
    #[test]
    fn a_tap_is_short_and_still() {
        assert_eq!(tap_of(0.0, 0.05, 2), Some(2));
        assert_eq!(tap_of(TOUCH_SLOP, TAP_TIME, 2), Some(2)); // both ends inclusive
        assert_eq!(tap_of(TOUCH_SLOP + 0.1, 0.05, 2), None); // travelled: a pinch
        assert_eq!(tap_of(0.0, TAP_TIME + 0.01, 2), None); // lingered: a rest
    }

    /// The count is the episode's widest, so a hand that put a third finger down and took
    /// it off again asked for redo — the fingers a gesture *had* are what it meant.
    #[test]
    fn a_tap_is_counted_at_its_widest() {
        assert_eq!(tap_of(0.0, 0.05, 3), Some(3));
        assert_eq!(tap_of(0.0, 0.05, 1), Some(1));
    }

    /// A short pull eases toward where it points in proportion to the pull, so the first
    /// few px barely turn the canvas.
    #[test]
    fn a_short_pull_eases_the_turn_in() {
        let view = ViewTransform::identity(Extent2::new(200, 200));
        // Straight up asks for no turn; a quarter of the follow distance goes a quarter of
        // the way there from a canvas turned one radian.
        let to = turn_to(view, Vec2::new(0.0, -TURN_FOLLOW_PX * 0.25), 1.0).expect("a pull");
        assert!((to - 0.75).abs() < 1e-6, "{to}");
        assert_eq!(turn_to(view, Vec2::ZERO, 1.0), None);
    }

    /// A long pull a little off square lands square: the snap is on the target, and a full
    /// ease arrives at it.
    #[test]
    fn a_long_pull_near_square_lands_square() {
        let view = ViewTransform::identity(Extent2::new(200, 200));
        let off = TURN_SNAP * 0.5;
        let up = Vec2::from_angle(-std::f32::consts::FRAC_PI_2 - off);
        let to = turn_to(view, up * TURN_FOLLOW_PX * 2.0, 1.0).expect("a pull");
        assert!(to.abs() < 1e-6, "{to}");
    }

    /// The short way round is never more than half a turn, and it always arrives.
    #[test]
    fn the_shortest_turn_never_exceeds_half_a_turn() {
        use std::f32::consts::{PI, TAU};
        let angles = [-7.0, -PI, -1.0, 0.0, 0.5, PI, 4.0, TAU, 12.0];
        for from in angles {
            for to in angles {
                let turn = shortest_turn(from, to);
                assert!(turn.abs() <= PI, "{from} -> {to}: {turn}");
                let miss = (from + turn - to).rem_euclid(TAU);
                assert!(
                    miss < 1e-4 || TAU - miss < 1e-4,
                    "{from} -> {to} misses by {miss}"
                );
            }
        }
    }

    fn factor(command: Option<ViewCommand>) -> f32 {
        match command {
            Some(ViewCommand::Zoom { factor, .. }) => factor,
            other => panic!("expected a zoom, got {other:?}"),
        }
    }

    /// The middle button pans whatever is held with it, and a bare left press is the
    /// surface's own — which is what keeps painting the resting gesture.
    #[test]
    fn the_bindings_are_the_two_hands_already_have() {
        let at = Vec2::ZERO;
        assert_eq!(press(Button::Middle, at, false, false), Some(Mode::Pan));
        assert_eq!(press(Button::Middle, at, true, true), Some(Mode::Pan));
        assert_eq!(press(Button::Left, at, false, false), None);
        assert_eq!(press(Button::Left, at, true, false), Some(Mode::Pan));
        assert_eq!(
            press(Button::Left, at, true, true),
            Some(Mode::Zoom { anchor: at })
        );
    }

    /// A scrubby drag doubles the zoom over its stated distance, and does it the same
    /// way along either axis — the sum, not the projection.
    #[test]
    fn a_scrub_doubles_over_the_distance_it_says() {
        let mode = Mode::Zoom { anchor: Vec2::ZERO };
        let right = mode.moved(Vec2::ZERO, Vec2::new(ZOOM_DRAG_DOUBLE, 0.0));
        assert!((factor(right) - 2.0).abs() < 1e-5);
        // Up is the same, screen y running the other way.
        let up = mode.moved(Vec2::ZERO, Vec2::new(0.0, -ZOOM_DRAG_DOUBLE));
        assert!((factor(up) - 2.0).abs() < 1e-5);
        // And back out again: the gesture is a function of where the pointer *is*.
        let out = mode.moved(Vec2::ZERO, Vec2::new(-ZOOM_DRAG_DOUBLE, 0.0));
        assert!((factor(out) - 0.5).abs() < 1e-5);
    }

    /// A move along the axis the gesture does not spend asks for nothing at all,
    /// rather than for a zoom of one.
    #[test]
    fn a_zoom_that_travelled_nowhere_asks_for_nothing() {
        let mode = Mode::Zoom { anchor: Vec2::ZERO };
        // x and y cancel: `travel` is zero even though the pointer moved.
        assert!(mode.moved(Vec2::ZERO, Vec2::splat(10.0)).is_none());
    }

    /// A pan is the raw delta, so the point under the hand stays under it.
    #[test]
    fn a_pan_is_the_hands_own_travel() {
        let moved = Mode::Pan.moved(Vec2::new(10.0, 10.0), Vec2::new(15.0, 4.0));
        match moved {
            Some(ViewCommand::Pan { delta }) => assert_eq!(delta, Vec2::new(5.0, -6.0)),
            other => panic!("expected a pan, got {other:?}"),
        }
    }

    /// Wheel notches compound, and away from the hand zooms in.
    #[test]
    fn the_wheel_compounds_and_zooms_in_away_from_the_hand() {
        assert!(factor(wheel(Vec2::ZERO, 1.0)) > 1.0);
        assert!(factor(wheel(Vec2::ZERO, -1.0)) < 1.0);
        // Two notches is one notch twice, which is what makes a fast scroll and a
        // slow one land in the same place.
        let once = factor(wheel(Vec2::ZERO, 1.0));
        assert!((factor(wheel(Vec2::ZERO, 2.0)) - once * once).abs() < 1e-5);
        assert!(wheel(Vec2::ZERO, 0.0).is_none());
    }
}
