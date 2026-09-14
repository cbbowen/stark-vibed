//! Which gesture a canvas's pointer event belongs to once a second pointer is down
//! (§25.4).
//!
//! A canvas holds at most one gesture. A frontend builds a [`Holder`] from the gesture's
//! own record of its pointer on every event, so this module keeps no copy that could
//! disagree with it. One pointer alone is never refused.

use crate::input::PointerKind;
use crate::nav::Lift;

/// The pointer that pressed a gesture. The kind rides with the id because a touch
/// contact, or a pen back in range, can return with a new id after a lost release.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pointer {
    pub id: i32,
    pub kind: PointerKind,
}

/// The canvas gestures a single pointer holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gesture {
    /// A middle-drag or a space-drag moving the view (§18.1.7).
    ViewDrag,
    /// The brush-tuning drag (§18.1.9).
    Tune,
    /// The layer carry (§16.11), until its release. One awaiting its readback holds
    /// nothing.
    Carry,
    /// The eyedropper, from the chord's press or a finger's hold (§18.0.2, §18.1.11).
    Pick,
    /// A stroke or a marquee, or a finger's press held in front of one.
    Paint,
}

/// What holds the canvas.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Holder {
    /// Fingers navigating, from the second landing to the last lift. Every finger down
    /// is the pinch's.
    Pinch,
    Gesture(Gesture, Pointer),
}

/// What a press asks to open.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Opens {
    /// A finger making a pair with the fingers already down.
    Pinch,
    Gesture(Gesture),
}

/// Where a move goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Moves {
    /// The holder's own pointer: its gesture advances.
    Holder,
    /// Another pointer under a gesture that does not yield: not even a hover or the
    /// cursor peers see.
    Ignored,
    /// Nothing shuts it out, so paint or the hover decides.
    Free,
}

impl Holder {
    pub fn holds(self, pointer: Pointer) -> bool {
        match self {
            Holder::Pinch => pointer.kind == PointerKind::Touch,
            Holder::Gesture(_, by) => by.id == pointer.id,
        }
    }

    fn kind(self) -> PointerKind {
        match self {
            Holder::Pinch => PointerKind::Touch,
            Holder::Gesture(_, by) => by.kind,
        }
    }

    /// Whether any press may take the canvas from it. Only paint: a finger's stroke is
    /// the opening half of a pinch, and a chord pressed over a stroke means the chord.
    fn yields(self) -> bool {
        matches!(self, Holder::Gesture(Gesture::Paint, _))
    }
}

/// Whether a press of `pointer` proves the holder's release never arrived, so the
/// holder must be put down before the press is read.
///
/// The holder's own pointer cannot press while down, and a **primary** pointer means
/// no other of its kind is down.
pub fn release_lost(holder: Option<Holder>, pointer: Pointer, primary: bool) -> bool {
    match holder {
        None => false,
        Some(Holder::Gesture(_, by)) if by.id == pointer.id => true,
        Some(held) => primary && pointer.kind == held.kind(),
    }
}

/// Whether a press may open `opens` while `holder` holds the canvas.
///
/// Paint yields to anything, and a pinch admits only a further finger. Every other
/// gesture refuses a second press, even of the same kind, since a takeover would strand
/// the first pointer's record.
pub fn admits(holder: Option<Holder>, opens: Opens) -> bool {
    match holder {
        None => true,
        Some(held) if held.yields() => true,
        Some(Holder::Pinch) => opens == Opens::Pinch,
        Some(Holder::Gesture(..)) => false,
    }
}

/// Whose a move of `pointer` is while `holder` holds the canvas.
pub fn moved(holder: Option<Holder>, pointer: Pointer) -> Moves {
    match holder {
        Some(held) if held.holds(pointer) => Moves::Holder,
        Some(held) if !held.yields() => Moves::Ignored,
        _ => Moves::Free,
    }
}

/// What a release or cancel of `pointer` does, `holder` being what held the canvas
/// **before** it, and `fingers` what the finger set said about it (`None` for a pointer
/// that is not a finger).
///
/// A one-pointer gesture ends on its own pointer's release alone. Only a pinch is ended
/// by the finger count and carries a tap: a refused pair means nothing by lifting.
pub fn released(holder: Option<Holder>, pointer: Pointer, fingers: Option<Lift>) -> Lift {
    const ENDED: Lift = Lift::Ended { tap: None };
    match holder {
        Some(Holder::Pinch) if pointer.kind == PointerKind::Touch => fingers.unwrap_or(ENDED),
        Some(held) if held.holds(pointer) => ENDED,
        Some(_) => Lift::Continuing,
        // Nothing to end, and no tail to run while a hand is still on the glass.
        None if fingers == Some(Lift::Continuing) => Lift::Continuing,
        None => ENDED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav::{TAP_TIME, Touch};
    use stark_model::geom::Vec2;

    const MOUSE: Pointer = Pointer {
        id: 1,
        kind: PointerKind::Mouse,
    };
    const PEN: Pointer = Pointer {
        id: 2,
        kind: PointerKind::Pen,
    };

    const fn finger(id: i32) -> Pointer {
        Pointer {
            id,
            kind: PointerKind::Touch,
        }
    }

    const ENDED: Lift = Lift::Ended { tap: None };

    /// What a press's button and modifiers ask for, before anything is asked of the
    /// router: the drag table's rows, the view drag's holds, or nothing (paint).
    #[derive(Clone, Copy, Debug)]
    enum Chord {
        Plain,
        ViewDrag,
        Tune,
        Carry,
        Pick,
    }

    const CHORDS: [Chord; 5] = [
        Chord::Plain,
        Chord::ViewDrag,
        Chord::Tune,
        Chord::Carry,
        Chord::Pick,
    ];

    impl Chord {
        fn gesture(self) -> Gesture {
            match self {
                Chord::Plain => Gesture::Paint,
                Chord::ViewDrag => Gesture::ViewDrag,
                Chord::Tune => Gesture::Tune,
                Chord::Carry => Gesture::Carry,
                Chord::Pick => Gesture::Pick,
            }
        }
    }

    /// A canvas in miniature, asking the router in the web `input::Gestures`' order: a
    /// lost release, a finger recorded and its pair offered as a pinch, the chord, then
    /// paint. It uses the real `Touch` finger set.
    #[derive(Default)]
    struct Canvas {
        one: Option<(Gesture, Pointer)>,
        touch: Touch,
        now: f64,
    }

    impl Canvas {
        fn holder(&self) -> Option<Holder> {
            match self.one {
                Some((gesture, by)) => Some(Holder::Gesture(gesture, by)),
                None => self.touch.is_pinching().then_some(Holder::Pinch),
            }
        }

        /// Every gesture put down. The fingers stay on the glass.
        fn put_down(&mut self) {
            self.one = None;
            self.touch.stop();
        }

        /// What the press opened, if anything.
        fn press(&mut self, pointer: Pointer, primary: bool, chord: Chord) -> Option<Opens> {
            if release_lost(self.holder(), pointer, primary) {
                self.put_down();
            }
            if pointer.kind == PointerKind::Touch
                && self
                    .touch
                    .finger_down(pointer.id, Vec2::ZERO, primary, self.now)
                && admits(self.holder(), Opens::Pinch)
            {
                // What the first finger was drawing was the pinch's opening half.
                self.one = None;
                self.touch.pinch(0.0);
                return Some(Opens::Pinch);
            }
            // A refused chord falls through to paint, as the ladder's does.
            let opened = [chord.gesture(), Gesture::Paint]
                .into_iter()
                .find(|&gesture| admits(self.holder(), Opens::Gesture(gesture)))?;
            self.one = Some((opened, pointer));
            Some(Opens::Gesture(opened))
        }

        fn moves(&self, pointer: Pointer) -> Moves {
            moved(self.holder(), pointer)
        }

        fn release(&mut self, pointer: Pointer) -> Lift {
            let holder = self.holder();
            let fingers = (pointer.kind == PointerKind::Touch)
                .then(|| self.touch.finger_up(pointer.id, self.now));
            let lift = released(holder, pointer, fingers);
            if lift != Lift::Continuing {
                self.put_down();
            }
            lift
        }
    }

    /// Opened by `pointer` alone on an empty canvas.
    fn holding(gesture: Gesture, pointer: Pointer) -> Canvas {
        let mut canvas = Canvas::default();
        let chord = CHORDS
            .into_iter()
            .find(|c| c.gesture() == gesture)
            .expect("every gesture has a chord");
        assert_eq!(
            canvas.press(pointer, true, chord),
            Some(Opens::Gesture(gesture))
        );
        canvas
    }

    #[test]
    fn one_pointer_alone_presses_moves_and_releases_as_it_always_did() {
        for pointer in [MOUSE, PEN, finger(5)] {
            for chord in CHORDS {
                let mut canvas = Canvas::default();
                let opened = Opens::Gesture(chord.gesture());
                assert_eq!(canvas.press(pointer, true, chord), Some(opened));
                assert_eq!(
                    canvas.moves(pointer),
                    Moves::Holder,
                    "{pointer:?} {chord:?}"
                );
                assert_eq!(canvas.release(pointer), ENDED, "{pointer:?} {chord:?}");
                assert_eq!(canvas.holder(), None);
                assert_eq!(canvas.moves(pointer), Moves::Free, "a hover again");
            }
        }
    }

    #[test]
    fn a_quick_still_pair_is_a_pinch_that_lifts_as_a_tap() {
        let mut canvas = Canvas::default();
        let paint = Some(Opens::Gesture(Gesture::Paint));
        assert_eq!(canvas.press(finger(5), true, Chord::Plain), paint);
        assert_eq!(
            canvas.press(finger(6), false, Chord::Plain),
            Some(Opens::Pinch)
        );
        assert_eq!(
            canvas.press(finger(7), false, Chord::Plain),
            Some(Opens::Pinch)
        );
        canvas.now = TAP_TIME * 0.5;
        assert_eq!(canvas.release(finger(6)), Lift::Continuing);
        assert_eq!(canvas.release(finger(5)), Lift::Continuing);
        assert_eq!(canvas.release(finger(7)), Lift::Ended { tap: Some(3) });
    }

    #[test]
    fn a_one_pointer_gesture_refuses_every_second_press_and_its_release_still_ends_it() {
        for held in [
            Gesture::ViewDrag,
            Gesture::Tune,
            Gesture::Carry,
            Gesture::Pick,
        ] {
            for (by, second) in [
                (MOUSE, PEN),
                (MOUSE, finger(5)),
                (PEN, MOUSE),
                (PEN, finger(5)),
            ] {
                for chord in CHORDS {
                    let mut canvas = holding(held, by);
                    let before = canvas.holder();
                    assert_eq!(
                        canvas.press(second, true, chord),
                        None,
                        "{held:?} {chord:?}"
                    );
                    assert_eq!(canvas.holder(), before, "{held:?} kept under {chord:?}");
                    assert_eq!(canvas.moves(second), Moves::Ignored);
                    assert_eq!(canvas.release(second), Lift::Continuing);
                    assert_eq!(canvas.release(by), ENDED);
                }
            }
        }
    }

    #[test]
    fn paint_yields_to_every_press() {
        for chord in [Chord::ViewDrag, Chord::Tune, Chord::Carry, Chord::Pick] {
            let mut canvas = holding(Gesture::Paint, PEN);
            let opened = Opens::Gesture(chord.gesture());
            assert_eq!(canvas.press(MOUSE, true, chord), Some(opened));
            assert_eq!(canvas.release(PEN), Lift::Continuing, "the stroke went");
            assert_eq!(canvas.release(MOUSE), ENDED);
        }
    }

    #[test]
    fn a_pinch_admits_a_finger_joining_it_and_nothing_else() {
        let mut canvas = holding(Gesture::Paint, finger(5));
        assert_eq!(
            canvas.press(finger(6), false, Chord::Plain),
            Some(Opens::Pinch)
        );
        for pointer in [MOUSE, PEN] {
            for chord in CHORDS {
                assert_eq!(canvas.press(pointer, true, chord), None, "{chord:?}");
                assert_eq!(canvas.moves(pointer), Moves::Ignored);
                assert_eq!(canvas.release(pointer), Lift::Continuing);
            }
        }
        assert_eq!(
            canvas.press(finger(7), false, Chord::Tune),
            Some(Opens::Pinch)
        );
        assert_eq!(canvas.moves(finger(7)), Moves::Holder);
    }

    /// Fingers are recorded before admission, so a refused palm's count must not veto
    /// the holder's own release.
    #[test]
    fn the_holders_release_ends_its_gesture_past_a_resting_palm() {
        for held in [Gesture::Tune, Gesture::Carry] {
            let mut canvas = holding(held, finger(5));
            // The chord is let go, and a palm lands.
            assert_eq!(canvas.press(finger(6), false, Chord::Plain), None);
            assert_eq!(canvas.moves(finger(6)), Moves::Ignored);
            assert_eq!(canvas.release(finger(5)), ENDED, "{held:?}");
            assert_eq!(canvas.release(finger(6)), ENDED, "no tap");
            assert_eq!(
                canvas.press(finger(7), true, Chord::Plain),
                Some(Opens::Gesture(Gesture::Paint)),
                "{held:?} left the canvas free"
            );
        }
    }

    #[test]
    fn a_pair_of_fingers_cannot_take_over_a_view_drag() {
        let mut canvas = holding(Gesture::ViewDrag, MOUSE);
        assert_eq!(canvas.press(finger(5), true, Chord::Plain), None);
        assert_eq!(canvas.press(finger(6), false, Chord::Plain), None);
        assert_eq!(canvas.moves(MOUSE), Moves::Holder);
        assert_eq!(canvas.moves(finger(6)), Moves::Ignored);
        assert_eq!(canvas.release(MOUSE), ENDED);
        assert_eq!(canvas.moves(MOUSE), Moves::Free, "a hover, not a pan");
    }

    #[test]
    fn a_palm_under_a_held_chord_opens_nothing() {
        for chord in [Chord::Tune, Chord::Carry, Chord::Pick] {
            let mut canvas = holding(chord.gesture(), MOUSE);
            assert_eq!(canvas.press(finger(5), true, chord), None);
            assert_eq!(
                canvas.holder(),
                Some(Holder::Gesture(chord.gesture(), MOUSE))
            );
        }
    }

    /// A touch, and a pen re-identified after leaving range, come back with a new id.
    #[test]
    fn a_release_lost_by_each_kind_of_pointer_is_recovered_by_its_next_press() {
        let paint = Some(Opens::Gesture(Gesture::Paint));
        let returning = [
            (MOUSE, MOUSE),
            (PEN, Pointer { id: 3, ..PEN }),
            (finger(5), finger(8)),
        ];
        for (by, next) in returning {
            for held in [
                Gesture::ViewDrag,
                Gesture::Tune,
                Gesture::Carry,
                Gesture::Pick,
                Gesture::Paint,
            ] {
                // Pressed, and its release never arrives.
                let mut canvas = holding(held, by);
                assert_eq!(
                    canvas.press(next, true, Chord::Plain),
                    paint,
                    "{by:?} {held:?}"
                );
                assert_eq!(canvas.holder(), Some(Holder::Gesture(Gesture::Paint, next)));
            }
        }

        let mut pinch = holding(Gesture::Paint, finger(5));
        assert_eq!(
            pinch.press(finger(6), false, Chord::Plain),
            Some(Opens::Pinch)
        );
        assert_eq!(
            pinch.press(finger(9), true, Chord::Plain),
            paint,
            "a lone finger"
        );
    }

    #[test]
    fn a_second_finger_under_a_finger_is_refused_rather_than_taken_for_a_lost_release() {
        let mut canvas = holding(Gesture::Tune, finger(5));
        assert_eq!(canvas.press(finger(6), false, Chord::Tune), None);
        assert_eq!(
            canvas.holder(),
            Some(Holder::Gesture(Gesture::Tune, finger(5)))
        );
    }

    #[test]
    fn a_pair_refused_under_a_carry_is_no_tap_when_the_carry_ends() {
        let mut canvas = holding(Gesture::Carry, MOUSE);
        assert_eq!(canvas.press(finger(5), true, Chord::Plain), None);
        assert_eq!(canvas.press(finger(6), false, Chord::Plain), None);
        assert_eq!(canvas.release(MOUSE), ENDED);
        canvas.now = TAP_TIME * 0.5;
        assert_eq!(canvas.release(finger(5)), Lift::Continuing);
        assert_eq!(canvas.release(finger(6)), ENDED);
    }

    /// One finger rests under the carry and the second lands after it: their pinch began
    /// under the carry.
    #[test]
    fn a_pinch_begun_by_a_finger_that_rested_under_a_gesture_is_no_tap() {
        let mut canvas = holding(Gesture::Carry, MOUSE);
        assert_eq!(canvas.press(finger(5), true, Chord::Plain), None);
        assert_eq!(canvas.release(MOUSE), ENDED);
        assert_eq!(
            canvas.press(finger(6), false, Chord::Plain),
            Some(Opens::Pinch)
        );
        canvas.now = TAP_TIME * 0.5;
        assert_eq!(canvas.release(finger(5)), Lift::Continuing);
        assert_eq!(canvas.release(finger(6)), ENDED);
    }

    #[test]
    fn a_refused_pair_then_the_holders_release_leaves_the_canvas_free() {
        let mut canvas = holding(Gesture::Tune, PEN);
        assert_eq!(canvas.press(finger(5), true, Chord::Plain), None);
        assert_eq!(canvas.press(finger(6), false, Chord::Plain), None);
        assert_eq!(canvas.release(PEN), ENDED);
        assert_eq!(
            canvas.release(finger(6)),
            Lift::Continuing,
            "a hand still down"
        );
        assert_eq!(canvas.release(finger(5)), ENDED);
        assert_eq!(
            canvas.press(PEN, true, Chord::Plain),
            Some(Opens::Gesture(Gesture::Paint))
        );
    }

    #[test]
    fn a_pick_keeps_the_canvas_past_a_palm() {
        let mut canvas = holding(Gesture::Pick, PEN);
        assert_eq!(canvas.press(finger(5), true, Chord::Pick), None);
        assert_eq!(canvas.press(finger(6), false, Chord::Plain), None);
        assert_eq!(canvas.moves(finger(5)), Moves::Ignored);
        assert_eq!(canvas.moves(PEN), Moves::Holder);
        assert_eq!(canvas.release(finger(5)), Lift::Continuing);
        assert_eq!(canvas.holder(), Some(Holder::Gesture(Gesture::Pick, PEN)));
        assert_eq!(canvas.release(PEN), ENDED);
    }

    #[test]
    fn under_a_stroke_another_pointer_moves_freely_for_paint_to_judge() {
        let canvas = holding(Gesture::Paint, PEN);
        assert_eq!(canvas.moves(PEN), Moves::Holder);
        assert_eq!(canvas.moves(finger(5)), Moves::Free);
    }
}
