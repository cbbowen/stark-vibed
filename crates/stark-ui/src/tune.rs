//! The brush-tuning drag (§18.1.9): the size sideways, the flow up and down, from
//! where the press landed.
//!
//! [`nav`](crate::nav)'s shape for the gesture that moves the *brush* instead of the
//! view — decided at the press, and from then on a pure function of the press point
//! and where the pointer is now. What is not here is the bookkeeping and the drawing:
//! which press opens this is the drag table's row (`crate::drags`), pointer capture is
//! each toolkit's, and the readout a size drag's ring or a flow drag's bar makes of
//! [`Turn`] is each frontend's own.
//!
//! It was written twice before it was written here, and the two disagreed about all
//! three of its numbers — and about the axis lock, which the native frontend did not
//! have at all: both knobs moved on every event there, so a long sideways drag emptied
//! or buried the brush on the way, and asking for size *alone* was not something the
//! gesture could express (§11.2).

use stark_model::geom::Vec2;

use crate::brush_config::{MAX_RADIUS, MIN_RADIUS, Transient};

/// How far a tuning drag must travel before it commits to a knob, in screen px.
///
/// `AXIS_DEADZONE` is the same reasoning a pinch's minimum span uses, applied to one
/// pointer: below this the drag's *direction* is noise, and the direction is the whole
/// of what picks the parameter. A press meant for Size that happens to leave the glass
/// two pixels high must not arrive as Flow.
pub const AXIS_DEADZONE: f32 = 8.0;

/// How far a tuning drag has to travel sideways to **double** the brush radius, in
/// screen px.
///
/// A ratio on the size the drag began with, not a size stated outright: the hand keeps
/// whatever brush it had chosen and asks for *more* or *less* of it, which is the
/// gesture every other editor binds here and the one a hand already reaches for. Right
/// is bigger and left is smaller, so the two directions are no longer the same gesture
/// — the drag has a sign, because a change does and a size does not.
///
/// **Exponential** for the scrubby zoom's reason ([`nav::ZOOM_DRAG_DOUBLE`]): radius
/// is felt proportionally, so a fixed step per pixel would crawl on a wash and leap on
/// a liner. Equal distances are equal ratios, which is also what makes the gesture
/// exactly reversible — dragging back to the press restores the brush it started on.
///
/// Faster than the zoom's rate rather than matched to it, and set from the range it
/// has to cover: [`MIN_RADIUS`]..[`MAX_RADIUS`] is about nine doublings, and a size
/// drag spends its travel on *one* side of the press where a zoom drag may run either
/// way from it, so the budget is half a screen and not a whole one. At this rate that
/// half-screen carries the finest brush to the widest.
///
/// [`nav::ZOOM_DRAG_DOUBLE`]: crate::nav::ZOOM_DRAG_DOUBLE
pub const SIZE_DRAG_DOUBLE: f32 = 100.0;

/// How far a tuning drag has to travel vertically to sweep the **whole** flow range,
/// in screen px.
///
/// Linear where the radius is exponential ([`SIZE_DRAG_DOUBLE`]), because flow's zero
/// is a value it has to be able to reach and no number of halvings gets there. There
/// is also nothing for flow to be a picture of: a size drag can be shown as the circle
/// it asks for, while flow has no length on screen to be measured against, so the
/// honest mapping is the one every slider has — move the hand, move the number. Wider
/// than a screen is tall on purpose: the everyday range is the narrow band around 1,
/// and this is what makes a tenth of it a visible movement of the hand.
pub const FLOW_DRAG_SPAN: f32 = 200.0;

/// The two parameters a tuning drag can reach, and the axis each is on.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Knob {
    /// Sideways: the brush radius (`Transient::size`).
    Size,
    /// Up and down: the rate of whichever effect is in force (`Transient::flow`,
    /// §6.2).
    Flow,
}

/// What a move asks the brush for: the knob the gesture has committed to, and where
/// it now stands.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Turn {
    /// The one knob this gesture spends.
    pub knob: Knob,
    /// What it stands at, in the knob's own units — canvas px for the size, the
    /// in-force effect's own rate for the flow.
    pub value: f32,
}

impl Turn {
    /// Move the knob this gesture spends, and nothing else.
    ///
    /// Written knob by knob rather than as a whole tune, so the **colour** the
    /// transient also carries cannot ride along: it is the Colour panel's, not the
    /// drag's (§18.1.8).
    pub fn write(self, t: &mut Transient) {
        match self.knob {
            Knob::Size => t.size = self.value,
            Knob::Flow => t.flow = self.value,
        }
    }

    /// How full the flow bar is drawn — the value as a share of the range it was
    /// measured against, since that range is the in-force effect's
    /// (`BrushConfig::max_flow`) and the overlay is told how full rather than how
    /// much.
    ///
    /// A size drag has no such picture: its readout is the ring, which is the two
    /// radii themselves ([`Tune::was`]).
    pub fn fill(self, max_flow: f32) -> f32 {
        if max_flow > 0.0 {
            self.value / max_flow
        } else {
            0.0
        }
    }
}

/// A tuning drag in flight: where the press landed, the tune it landed on, and the
/// knob it has committed to.
///
/// **Everything is measured from the press**, never from the last move. Which knob the
/// gesture is about is a fact about the whole gesture; so is the ratio — and so is the
/// level, now that both halves are stated the same way. That is what keeps the drag a
/// function of where the pointer *is* rather than an accumulation of steps: a long
/// gesture cannot drift, and a drag run past a clamp and back comes down the way it
/// went up, because the clamp is never folded into the base.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Tune {
    from: Vec2,
    was: Transient,
    knob: Option<Knob>,
}

impl Tune {
    /// Open a tuning drag at `at` on the brush's current tune.
    ///
    /// No test on the tool, deliberately: the eraser end tunes the eraser for the
    /// reason it erases (§18.1.8), and the size and the rate are the live brush's
    /// whatever the canvas is set to do with it.
    pub fn press(at: Vec2, was: Transient) -> Self {
        Self {
            from: at,
            was,
            knob: None,
        }
    }

    /// Advance to `at`. `None` until the drag has travelled far enough to say which
    /// knob it is about ([`AXIS_DEADZONE`]) — those moves are the gesture's too, even
    /// though they change nothing, and the travel spent earning the lock is spent: a
    /// deadband, not a jump.
    ///
    /// `max_flow` is the in-force effect's own ceiling (`BrushConfig::max_flow`),
    /// passed in because the range belongs to the brush and this value holds only the
    /// gesture — the liquify strength's is its quoted 1 where every other rate's is
    /// the slider's own top (§6.13).
    ///
    /// **One knob per gesture**, and locking it is the point. Both at once reads
    /// better on paper and is worse in the hand: flow's useful range is narrow enough
    /// that the incidental drift of a long sideways drag would empty or bury the
    /// brush, and the user would have no way to ask for size alone.
    pub fn moved(&mut self, at: Vec2, max_flow: f32) -> Option<Turn> {
        let travel = at - self.from;
        if self.knob.is_none() && travel.length() >= AXIS_DEADZONE {
            self.knob = Some(if travel.x.abs() >= travel.y.abs() {
                Knob::Size
            } else {
                Knob::Flow
            });
        }
        let knob = self.knob?;
        let value = match knob {
            // Right is bigger, left is smaller — a ratio on the size at the press
            // rather than a size stated outright, so the hand asks for more or less of
            // the brush it already chose.
            Knob::Size => {
                (self.was.size * (travel.x / SIZE_DRAG_DOUBLE).exp2()).clamp(MIN_RADIUS, MAX_RADIUS)
            }
            // Up is more, because up is more on every slider in the app — and screen y
            // grows downward, which is the whole of why this reads as a subtraction.
            // Scaled by the range so a full drag is a full knob whichever effect is in
            // force.
            Knob::Flow => {
                (self.was.flow - travel.y * max_flow / FLOW_DRAG_SPAN).clamp(0.0, max_flow)
            }
        };
        Some(Turn { knob, value })
    }

    /// Where the press landed — what a readout is anchored to.
    pub fn from(self) -> Vec2 {
        self.from
    }

    /// The tune the drag began on. The size half is the ring's reference: the circle
    /// behind is the brush every ratio this gesture asks for is a ratio *of*, not
    /// merely the one it happened to start at.
    pub fn was(self) -> Transient {
        self.was
    }

    /// The knob the drag has committed to, `None` while it is still inside the
    /// deadzone.
    pub fn knob(self) -> Option<Knob> {
        self.knob
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tune(size: f32, flow: f32) -> Transient {
        Transient {
            size,
            flow,
            ..Transient::default()
        }
    }

    fn at(x: f32, y: f32) -> Vec2 {
        Vec2::new(x, y)
    }

    /// A press asks for nothing, and neither does a twitch: the direction is what
    /// picks the parameter, and inside the deadzone there is no direction to read.
    #[test]
    fn a_drag_inside_the_deadzone_asks_for_nothing() {
        let mut drag = Tune::press(at(0.0, 0.0), tune(40.0, 1.0));
        assert_eq!(drag.moved(at(0.0, 0.0), 3.0), None);
        assert_eq!(drag.moved(at(4.0, 4.0), 3.0), None, "under 8 px of travel");
        assert_eq!(drag.knob(), None);
        assert!(drag.moved(at(AXIS_DEADZONE, 0.0), 3.0).is_some());
    }

    /// The lock is for the whole gesture: a drag that earned Size keeps it however far
    /// the hand then wanders up the glass, which is what makes asking for size alone
    /// possible at all.
    #[test]
    fn one_gesture_spends_one_knob() {
        let mut drag = Tune::press(at(0.0, 0.0), tune(40.0, 1.0));
        assert_eq!(
            drag.moved(at(20.0, 0.0), 3.0).map(|t| t.knob),
            Some(Knob::Size)
        );
        let wandered = drag.moved(at(20.0, 400.0), 3.0).expect("still tuning");
        assert_eq!(wandered.knob, Knob::Size, "the lock holds");
        assert_eq!(
            wandered.value,
            drag.moved(at(20.0, 0.0), 3.0).unwrap().value
        );

        let mut drag = Tune::press(at(0.0, 0.0), tune(40.0, 1.0));
        assert_eq!(
            drag.moved(at(0.0, -20.0), 3.0).map(|t| t.knob),
            Some(Knob::Flow)
        );
        assert_eq!(
            drag.moved(at(400.0, -20.0), 3.0).map(|t| t.knob),
            Some(Knob::Flow),
        );
    }

    /// The size doubles over the distance it says, and halves over the same distance
    /// the other way.
    #[test]
    fn a_size_drag_doubles_over_the_distance_it_says() {
        let mut drag = Tune::press(at(0.0, 0.0), tune(40.0, 1.0));
        let out = drag.moved(at(SIZE_DRAG_DOUBLE, 0.0), 3.0).expect("tuning");
        assert!((out.value - 80.0).abs() < 1e-4);
        let back = drag.moved(at(-SIZE_DRAG_DOUBLE, 0.0), 3.0).expect("tuning");
        assert!((back.value - 20.0).abs() < 1e-4);
    }

    /// A full vertical drag is a full knob **whichever effect is in force**: the range
    /// is the brush's (`BrushConfig::max_flow`), so a liquify brush's 1 sweeps in the
    /// same travel a paint brush's 3 does.
    #[test]
    fn a_flow_drag_sweeps_whatever_range_it_is_given() {
        for max in [1.0, 3.0] {
            let mut drag = Tune::press(at(0.0, 0.0), tune(40.0, 0.0));
            let up = drag.moved(at(0.0, -FLOW_DRAG_SPAN), max).expect("tuning");
            assert!((up.value - max).abs() < 1e-4, "max {max}");
            assert!((up.fill(max) - 1.0).abs() < 1e-4);
            // ...and it stops there rather than going somewhere the slider cannot show.
            let past = drag
                .moved(at(0.0, -FLOW_DRAG_SPAN * 3.0), max)
                .expect("tuning");
            assert!((past.value - max).abs() < 1e-4);
        }
    }

    /// **Dragging out and back to the press restores the brush it started on**, which
    /// is what the rates are chosen to make true and what neither app had a test for.
    ///
    /// Exactly, not nearly: both knobs are a function of where the pointer *is*, so
    /// the return trip is the same arithmetic on a zero travel rather than a sum of
    /// steps that has to cancel. A drag run past a clamp and back is the case that
    /// makes the difference — an accumulating one would come back short by however far
    /// it had been held against the end.
    #[test]
    fn a_drag_out_and_back_restores_the_brush_it_found() {
        let was = tune(40.0, 1.0);
        for wander in [
            at(320.0, 0.0),
            at(-320.0, 0.0),
            at(0.0, 900.0),
            at(0.0, -900.0),
        ] {
            let mut drag = Tune::press(at(100.0, 100.0), was);
            let mut live = was;
            drag.moved(at(100.0, 100.0) + wander, 3.0)
                .expect("the wander is past the deadzone")
                .write(&mut live);
            drag.moved(at(100.0, 100.0), 3.0)
                .expect("the knob is locked, so the return still answers")
                .write(&mut live);
            assert_eq!(live, was, "a drag out to {wander:?} and back");
        }
    }

    /// The clamps are the app's own bounds, so the drag cannot put the brush anywhere
    /// a slider is unable to show or take back.
    #[test]
    #[expect(
        clippy::float_cmp_const,
        reason = "a clamp lands *on* its bound exactly, which is the whole claim"
    )]
    fn the_drag_stops_where_the_sliders_do() {
        let mut drag = Tune::press(at(0.0, 0.0), tune(40.0, 1.0));
        let huge = drag.moved(at(4000.0, 0.0), 3.0).expect("tuning");
        assert_eq!(huge.value, MAX_RADIUS);
        let mut drag = Tune::press(at(0.0, 0.0), tune(40.0, 1.0));
        let tiny = drag.moved(at(-4000.0, 0.0), 3.0).expect("tuning");
        assert_eq!(tiny.value, MIN_RADIUS);
        let mut drag = Tune::press(at(0.0, 0.0), tune(40.0, 1.0));
        let empty = drag.moved(at(0.0, 4000.0), 3.0).expect("tuning");
        assert_eq!(empty.value, 0.0);
    }

    /// A turn moves its own knob and leaves the rest of the tune alone — the colour
    /// above all, which is the Colour panel's (§18.1.8).
    #[test]
    fn a_turn_moves_one_knob() {
        let mut live = Transient {
            color: [0.9, 0.1, 0.2],
            ..tune(40.0, 1.0)
        };
        Turn {
            knob: Knob::Size,
            value: 88.0,
        }
        .write(&mut live);
        assert_eq!((live.size, live.flow), (88.0, 1.0));
        Turn {
            knob: Knob::Flow,
            value: 0.25,
        }
        .write(&mut live);
        assert_eq!((live.size, live.flow), (88.0, 0.25));
        assert_eq!(live.color, [0.9, 0.1, 0.2]);
    }
}
