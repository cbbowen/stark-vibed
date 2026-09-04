//! What a press, a drag and a wheel notch do to the **view** (§18.1.7).
//!
//! Navigation is the one gesture family that has to mean the same thing wherever the
//! pointer lands — composing a transform must not cost the artist the ability to look
//! around — and now that there are two frontends, "wherever" includes "whichever
//! app". So the decisions and the rates are here: which press is a pan and which is a
//! zoom, how far a scrubby drag travels per doubling, and how much a notch of the
//! wheel is worth.
//!
//! What is *not* here is the bookkeeping, because it is a different thing in each
//! app: the web tracks pointer ids so it can pair two fingers into a pinch, and holds
//! its in-flight drag in a `Signal`; the native frontend has a mouse, a wheel and one
//! `Held`. Touch belongs to the web alone until wgpui reports fingers.

use stark_engine::command::ViewCommand;
use stark_model::Vec2;

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

#[cfg(test)]
mod tests {
    use super::*;

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
