//! **Adjustment**: what the rest of the drag means once a shape has been recognized
//! (§6.9).
//!
//! Both arms are derived from the shape *as recognized* plus the pointer's travel
//! since, never from the previous frame's shape, so a long adjustment cannot
//! accumulate drift — the same bargain the transform gesture makes (§16.6).

use super::AssistShape;
use stark_model::geom::{Ellipse, Vec2};

/// Shortest grip an ellipse adjustment will turn or scale about, in canvas px. Nearer
/// the centre than this the pointer's *direction* is noise, exactly as it is at the
/// start of the navigator's rotate drag (§18.1.2).
const MIN_GRIP: f32 = 1.0;

impl AssistShape {
    /// The shape as the pointer's travel from `grip` to `pointer` leaves it.
    ///
    /// Always applied to the shape **as recognized**, with the grip where the pointer
    /// was when it snapped — so what a caller holds is one shape plus one displacement,
    /// and a minute of adjustment is no different from the same drag made at once.
    ///
    /// - A **line** moves the end the pointer is holding. By the pointer's delta rather
    ///   than to the pointer itself: snapping moved that end off the hand by up to the
    ///   fit residual, and driving it absolutely would jump it back on the first move.
    ///   A line that snapped to a **guide axis** keeps that direction and takes only
    ///   the component of the travel along it, so the end runs out and back along the
    ///   grid line for the rest of the drag. Adjustment preserves what recognition
    ///   established — the same bargain that keeps a drawn loop's eccentricity — and
    ///   an alignment that a single sideways nudge could break would not be one.
    /// - An **ellipse** turns and scales about its centre, so that the point being held
    ///   follows the pointer. Turning is what the feature is for; the scale rides along
    ///   because a one-pointer drag has two degrees of freedom and the radius is the
    ///   only other thing a hand at that position could mean. The eccentricity the
    ///   drawn loop established is preserved.
    /// - A **perspective circle** (§20.7) is sized, and only sized, in the plane it is
    ///   a circle on: turning a circle does nothing, so the turn the free arm spends a
    ///   degree of freedom on is not there to spend. Its eccentricity and tilt on the
    ///   canvas then follow from where on the plane it sits — which is the point of the
    ///   thing, and is why it cannot be done by scaling the drawn ellipse.
    pub fn adjust(self, grip: Vec2, pointer: Vec2) -> Self {
        match self {
            Self::Line { a, b, on_axis } => {
                let end = b + (pointer - grip);
                // The base direction, so the constraint is a fixed line and the result
                // stays a pure function of the total travel.
                let held = (b - a).try_normalize().filter(|_| on_axis);
                Self::Line {
                    a,
                    b: match held {
                        Some(u) => a + u * (end - a).dot(u),
                        None => end,
                    },
                    on_axis,
                }
            }
            Self::Ellipse {
                center,
                radii,
                angle,
                phase,
                winding,
                plane: Some(plane),
            } => {
                // A circle has no orientation, so there is nothing here for the drag to
                // turn: what is left of the ellipse's two degrees of freedom is the
                // size, taken in the plane's own coordinates about the centre it has
                // *there*. The canvas ellipse's centre is not the image of the circle's
                // centre, so scaling the drawn shape about it would leave the plane at
                // once.
                let sized = plane
                    .circle_behind(Ellipse::new(center, radii, angle))
                    .zip(plane.to_plane(grip).zip(plane.to_plane(pointer)))
                    .and_then(|((flat, radius), (from, to))| {
                        let l0 = (from - flat).length();
                        (l0 > 0.0)
                            .then(|| plane.circle_seen(flat, radius * (to - flat).length() / l0))
                            .flatten()
                    });
                match sized {
                    Some(seen) => Self::Ellipse {
                        center: seen.center,
                        radii: seen.radii,
                        angle: seen.angle,
                        phase,
                        winding,
                        plane: Some(plane),
                    },
                    // Nothing to draw at the far end of that drag — the circle has been
                    // pulled through its own vanishing line, or down to a point. Holding
                    // still is the honest answer, and dragging back recovers.
                    None => self,
                }
            }
            Self::Ellipse {
                center,
                radii,
                angle,
                phase,
                winding,
                plane: None,
            } => {
                let (from, to) = (grip - center, pointer - center);
                let (l0, l1) = (from.length(), to.length());
                if l0 < MIN_GRIP || l1 < MIN_GRIP {
                    return self;
                }
                Self::Ellipse {
                    center,
                    radii: radii * (l1 / l0),
                    angle: angle + from.angle_to(to),
                    phase,
                    winding,
                    plane: None,
                }
            }
        }
    }

    /// The point a drag is deemed to be holding, for a gesture that has just snapped.
    ///
    /// Not the pointer's own position: it is the shape that moves from here on, so the
    /// grip has to be a point *of the shape*, or the first adjustment would apply the
    /// fit residual as a jump.
    pub fn grip(&self) -> Vec2 {
        match *self {
            Self::Line { b, .. } => b,
            // The seam: where the loop starts, which is where the hand is holding it.
            Self::Ellipse { .. } => self.at(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line's parts, or a panic naming what came back instead. The root's test module
    /// has a copy: seven of its assertions want it and one of this file's does, and one
    /// four-line `match` in each place is cheaper than making it visible across a
    /// module boundary for a single caller.
    fn as_line(shape: AssistShape) -> (Vec2, Vec2, bool) {
        match shape {
            AssistShape::Line { a, b, on_axis } => (a, b, on_axis),
            other => panic!("adjusted into {other:?}, not a line"),
        }
    }

    /// Adjustment is always measured from the shape as recognized, so the same total
    /// travel means the same thing however it is broken up.
    #[test]
    fn adjustment_does_not_accumulate() {
        let shape = AssistShape::Ellipse {
            center: Vec2::ZERO,
            radii: Vec2::new(100.0, 50.0),
            angle: 0.0,
            phase: 0.0,
            winding: 1.0,
            plane: None,
        };
        let grip = shape.grip();
        let target = Vec2::new(0.0, 140.0);
        let direct = shape.adjust(grip, target);
        let stepped = (1..=8).fold(shape, |_, i| {
            shape.adjust(grip, grip.lerp(target, i as f32 / 8.0))
        });
        assert_eq!(direct, stepped);
    }

    #[test]
    fn a_line_adjustment_moves_the_held_end() {
        let shape = AssistShape::Line {
            a: Vec2::ZERO,
            b: Vec2::new(100.0, 0.0),
            on_axis: false,
        };
        let moved = shape.adjust(shape.grip(), Vec2::new(100.0, 60.0));
        assert_eq!(
            moved,
            AssistShape::Line {
                a: Vec2::ZERO,
                b: Vec2::new(100.0, 60.0),
                on_axis: false,
            }
        );
    }

    /// Steering a line that took an axis runs it out along that axis: the pointer's
    /// travel across the line is dropped, and what it means is where along the line the
    /// end lands. An alignment a sideways nudge could break would not be one.
    #[test]
    fn an_axis_line_is_steered_along_its_axis() {
        let (a, b) = (Vec2::new(10.0, 10.0), Vec2::new(210.0, 110.0));
        let shape = AssistShape::Line {
            a,
            b,
            on_axis: true,
        };
        let u = (b - a).normalize();
        // A pointer that has wandered a long way off the line, and some way along it.
        let target = b + u * 90.0 + u.perp() * 140.0;
        let (a2, b2, on) = as_line(shape.adjust(shape.grip(), target));
        assert_eq!((a2, on), (a, true));
        assert!(
            (b2 - a).perp_dot(u).abs() < 1e-3,
            "the end left the axis, at {b2}"
        );
        assert!(
            ((b2 - a).dot(u) - ((b - a).length() + 90.0)).abs() < 1e-3,
            "the end did not run out along the axis, at {b2}"
        );

        // And, like every adjustment, it is a function of the total travel (§16.6).
        let stepped = (1..=8).fold(shape, |_, i| {
            shape.adjust(shape.grip(), shape.grip().lerp(target, i as f32 / 8.0))
        });
        assert_eq!(shape.adjust(shape.grip(), target), stepped);
    }
}
