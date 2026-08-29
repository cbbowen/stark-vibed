//! Drag-and-hold drawing assist (§6.9).
//!
//! Drag out a rough line or a rough ellipse and then **hold the pen down without
//! moving it**: the stroke in flight snaps to the ideal shape it resembles, and the
//! rest of the same drag steers that shape rather than extending the path. Lift to
//! commit, exactly as for any other stroke.
//!
//! The feature attaches at the seam §18.1.3 named for it — *a path transform applied
//! between the fitter and the renderer*. Nothing downstream of
//! [`StrokeRecord::path`](stark_model::document::StrokeRecord) learns that assist exists: a
//! snapped stroke is still a list of [`ControlPoint`]s, so the renderer, the wire
//! format, the save file, replay and the goldens are untouched, and the assisted
//! stroke is undoable, replayable and collaborative for free. That is why this module
//! answers in control points instead of carrying a shape into the action log.
//!
//! Three separable pieces:
//!
//! - **Recognition** ([`recognize`]) — which shape, if any, the raw pointer trace is.
//!   It answers `None` freely: dwelling at the end of a stroke that is *not* a line or
//!   an ellipse has to leave that stroke exactly as it was drawn, because a false
//!   positive (a considered stroke silently replaced) costs far more than a miss
//!   (hold it a little straighter and try again). A line that lands near an axis of a
//!   perspective guide the artist has up is turned onto it (§20.6) — the only place
//!   this module knows a guide exists.
//! - **Adjustment** ([`AssistShape::adjust`]) — what the rest of the drag means. Both
//!   arms are derived from the shape *as recognized* plus the pointer's travel since,
//!   never from the previous frame's shape, so a long adjustment cannot accumulate
//!   drift — the same bargain the transform gesture makes (§16.6).
//! - **Realization** ([`AssistShape::to_path`]) — the shape as a fitted path, carrying
//!   the pen channels the stroke was actually drawn with. A snapped line keeps the
//!   pressure swell that was put into it rather than becoming a plotted vector, which
//!   is the whole difference between this and a shape tool.
//!
//! Everything the recognizer measures against is denominated in the gesture's **input
//! tolerance** — the device's own tolerance in canvas px, which the frontend already
//! states for the fit (see [`PathFitter::with_tolerance`](crate::path::PathFitter::with_tolerance)).
//! A "close enough to a line" bar fixed in canvas px would mean two different things
//! at two zoom levels, for the same reason the fit's own prices would.
//!
//! **Three files, named for the three pieces above.** They were one 1,891-line module
//! for as long as the pieces shared a page; what separates them is not size but that
//! each answers a different question and shares almost nothing with the others — the
//! constants partition without a remainder (twelve to recognition, six to realization,
//! one to adjustment), and no helper is called from two of them. The tests stay *here*
//! rather than moving with the code they exercise, because they are tests of the
//! feature and not of the pieces: almost every one draws a trace, recognizes it,
//! sometimes steers it, and reads the path that comes out. Splitting them three ways
//! would put one copy of the trace builders in each file, which is the duplication a
//! split is supposed to be worth avoiding.

mod adjust;
mod realize;
mod recognize;

pub use realize::PenProfile;
pub use recognize::recognize;

use stark_model::document::AxisPlane;
use stark_model::geom::Vec2;

/// The ideal shape a held gesture snapped to (§6.9).
///
/// Held in the session for the rest of the gesture and thrown away at the end: what is
/// committed is the path it produces, not the shape itself. Post-hoc stroke editing
/// (§18.2.1) is where keeping it would start to pay, and that is the point at which to
/// decide it costs a wire-format change.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AssistShape {
    /// A straight segment from `a` to `b`. `b` is the end the stroke finished at —
    /// the end under the pointer, and so the end a drag moves.
    Line {
        a: Vec2,
        b: Vec2,
        /// Whether the direction is a perspective guide's rather than the hand's
        /// (§20.6). One bit, because that is all the difference amounts to: the
        /// pencil's line through `a` *is* a straight canvas line, so a guided line
        /// is the same segment held to a direction it may not leave, and the rest
        /// of the drag runs **along** it instead of steering it off.
        on_axis: bool,
    },
    /// An ellipse, drawn as one closed loop.
    Ellipse {
        center: Vec2,
        /// Semi-axes along the ellipse's own frame, major first.
        radii: Vec2,
        /// Rotation of that frame, radians clockwise in canvas space.
        angle: f32,
        /// The perspective plane this is a **circle** on, if it is one (§20.7).
        ///
        /// The ellipse above stays the truth — it is what is drawn, and every
        /// consumer below reads it without knowing a guide exists. This says only
        /// which family the shape came out of and may not leave, which is all
        /// [`adjust`](Self::adjust) needs to go on steering it a circle. The same
        /// bit of bookkeeping [`Line::on_axis`](Self::Line::on_axis) is, for the
        /// same reason.
        plane: Option<AxisPlane>,
        /// Where on the ellipse the loop starts and ends, in the ellipse's **own**
        /// frame — so turning the shape carries the seam with it, and the join stays
        /// where the hand put it rather than sliding round under a rotation.
        phase: f32,
        /// Which way round the loop travels: `+1` or `-1` on [`phase`](Self::Ellipse::phase).
        /// Taken from the drawn trace so the pen channels ride the shape the way they
        /// were laid down.
        winding: f32,
    },
}

/// Which family a held gesture snapped to (§6.9) — as much of [`AssistShape`] as
/// anything outside the engine has needed to know.
///
/// A second, smaller enum rather than exporting the shape itself, because what a
/// caller out there wants is *which kind* and what the shape carries is geometry —
/// two points, a frame, a winding, the plane it is a circle on. Publishing those
/// would fix the assist's internals as an interface for the sake of a question
/// answered by one bit. It gains a field the day something outside needs one.
///
/// The caller outside the crate is the guided tour (§24.2), which counts an assisted
/// stroke and tells a line from a loop.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Assisted {
    /// A straight segment.
    Line,
    /// A closed loop.
    Ellipse,
}

impl AssistShape {
    /// Which family this is.
    pub fn family(&self) -> Assisted {
        match self {
            AssistShape::Line { .. } => Assisted::Line,
            AssistShape::Ellipse { .. } => Assisted::Ellipse,
        }
    }
}

#[cfg(test)]
mod tests {
    // The pieces' own items the feature tests reach for: a recognition internal each
    // (`Frame` builds a perspective circle's plane, `wrap_pi` reads an angle back),
    // and realization's ellipse tolerance, which is the bar a "is it round?" assertion
    // is held to and would be a second number if it were restated here.
    use super::realize::ELLIPSE_ERROR;
    use super::recognize::{Frame, wrap_pi};
    use super::*;
    use crate::path::{FLATTEN_TOLERANCE, flatten};
    use stark_model::document::Scaffold;
    use stark_model::geom::Ellipse;
    use stark_model::path::ControlPoint;
    use std::f32::consts::TAU;

    const TOL: f32 = 1.0;

    /// The recognizer with no guides on the screen — what every test but the §20.6
    /// and §20.7 ones is asking about.
    fn free(trace: &[Vec2]) -> Option<AssistShape> {
        recognize(trace, TOL, &Scaffold::default())
    }

    /// A trace along an ideal shape, with `wobble` px of deterministic zig-zag across
    /// it — a stand-in for a hand.
    fn trace_of(n: usize, wobble: f32, f: impl Fn(f32) -> Vec2) -> Vec<Vec2> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                f(t) + Vec2::new(0.0, wobble * (i as f32 * 1.7).sin())
            })
            .collect()
    }

    fn line_trace(a: Vec2, b: Vec2, wobble: f32) -> Vec<Vec2> {
        trace_of(40, wobble, |t| a.lerp(b, t))
    }

    fn ellipse_trace(center: Vec2, radii: Vec2, angle: f32, wobble: f32) -> Vec<Vec2> {
        loop_trace(center, radii, angle, wobble, 0.0)
    }

    /// A hand-drawn loop. `turns` is how far past (or short of) one full turn it runs —
    /// nobody closes a loop exactly on the point they started from.
    fn loop_trace(center: Vec2, radii: Vec2, angle: f32, wobble: f32, turns: f32) -> Vec<Vec2> {
        trace_of(120, wobble, |t| {
            let u = t * (1.0 + turns) * TAU;
            center + Vec2::from_angle(angle).rotate(Vec2::new(radii.x * u.cos(), radii.y * u.sin()))
        })
    }

    #[test]
    fn a_rough_drag_is_a_line() {
        let (a, b) = (Vec2::new(10.0, 20.0), Vec2::new(310.0, 90.0));
        let shape = free(&line_trace(a, b, 3.0)).expect("a line");
        let AssistShape::Line {
            a: fa,
            b: fb,
            on_axis,
        } = shape
        else {
            panic!("recognized {shape:?}, not a line");
        };
        assert!(fa.distance(a) < 6.0, "start moved to {fa}");
        assert!(fb.distance(b) < 6.0, "end moved to {fb}");
        assert!(!on_axis, "there were no guides to be on the axis of");
    }

    #[test]
    fn a_rough_loop_is_an_ellipse() {
        let (center, radii) = (Vec2::new(100.0, -40.0), Vec2::new(120.0, 60.0));
        let shape = free(&ellipse_trace(center, radii, 0.4, 3.0)).expect("an ellipse");
        let AssistShape::Ellipse {
            center: c,
            radii: r,
            angle,
            ..
        } = shape
        else {
            panic!("recognized {shape:?}, not an ellipse");
        };
        assert!(c.distance(center) < 8.0, "centre moved to {c}");
        // The eccentricity is the thing the naive moment fit gets wrong — a 2:1
        // ellipse read through an arc-length measure comes back nearer 1.7:1.
        let ratio = r.x / r.y;
        assert!(
            (ratio - 2.0).abs() < 0.2,
            "axis ratio {ratio} (radii {r}), expected 2:1"
        );
        assert!((wrap_pi(angle - 0.4)).abs() < 0.15, "angle {angle}");
    }

    /// Eccentricity is not something the recognizer may be shy about: a long thin loop
    /// is a *more* deliberate thing to draw than a circle, not a less confident one.
    ///
    /// This is the case a naive moment fit cannot reach at all — read through an
    /// arc-length measure a 2:1 ellipse comes back nearer 1.7:1, and the error grows
    /// with the axis ratio until the shape misses the bar for being what it is.
    #[test]
    fn a_long_thin_loop_is_still_an_ellipse() {
        for ratio in [2.0f32, 4.0, 8.0] {
            let radii = Vec2::new(400.0, 400.0 / ratio);
            let shape = free(&ellipse_trace(Vec2::ZERO, radii, -1.2, 4.0))
                .unwrap_or_else(|| panic!("{ratio}:1 declined"));
            let AssistShape::Ellipse { radii: r, .. } = shape else {
                panic!("{ratio}:1 recognized as {shape:?}");
            };
            assert!(
                (r.x / r.y - ratio).abs() < 0.15 * ratio,
                "{ratio}:1 read as {:.2}:1 (radii {r})",
                r.x / r.y
            );
        }
    }

    /// **Where a loop closes is where a hand is least accurate**, so neither running
    /// past the start nor stopping short of it may cost the shape.
    ///
    /// Both are easy to get wrong: unweighted, an overshoot double-counts the
    /// re-traversed wedge and walks the centre 78px off a 400px ellipse, while an
    /// undershoot leaves the closed-form inversion describing an arc while assuming a
    /// whole turn. See [`weigh`].
    /// A loop is a longer gesture than a drag of the same size and the wrist reverses
    /// twice on the way round it, so the hand that draws one is not the hand that draws
    /// a line — see [`ELLIPSE_RESIDUAL`]. A tenth of the radius of wobble is a shaky
    /// hand, not a considered shape, and it has to come back as the ellipse it was aimed
    /// at.
    #[test]
    fn a_shaky_loop_is_still_an_ellipse() {
        let radii = Vec2::new(300.0, 150.0);
        let wobble = 0.10 * 0.5 * (radii.x + radii.y);
        let shape = free(&ellipse_trace(Vec2::ZERO, radii, 0.4, wobble)).expect("an ellipse");
        let AssistShape::Ellipse {
            center, radii: r, ..
        } = shape
        else {
            panic!("recognized {shape:?}, not an ellipse");
        };
        assert!(center.length() < 0.1 * radii.x, "centre moved to {center}");
        assert!(
            (r.x / r.y - 2.0).abs() < 0.3,
            "axis ratio {} (radii {r})",
            r.x / r.y
        );
    }

    #[test]
    fn a_loop_that_misses_its_own_start_still_snaps() {
        // A fifth of a turn short is what [`CLOSE_GAP`] allows and what the fill in
        // [`weigh`] can carry; three quarters of a turn is an arc and is refused.
        for turns in [-0.18f32, -0.10, -0.04, 0.0, 0.04, 0.10, 0.18] {
            for ratio in [1.0f32, 4.0] {
                let radii = Vec2::new(400.0, 400.0 / ratio);
                let trace = loop_trace(Vec2::ZERO, radii, 0.6, 4.0, turns);
                let shape =
                    free(&trace).unwrap_or_else(|| panic!("{ratio}:1 at {turns:+} turns declined"));
                let AssistShape::Ellipse {
                    center, radii: r, ..
                } = shape
                else {
                    panic!("recognized as {shape:?}");
                };
                assert!(
                    center.length() < 0.05 * radii.x,
                    "{ratio}:1 at {turns:+} turns put the centre at {center}"
                );
                assert!(
                    (r.x - radii.x).abs() < 0.08 * radii.x,
                    "{ratio}:1 at {turns:+} turns read the major axis as {}",
                    r.x
                );
            }
        }
    }

    /// The other side of that: an **arc** is a shape somebody can mean, and closing it
    /// for them would be the expensive kind of mistake. A quarter of a turn missing is
    /// not a hand that fumbled the join.
    #[test]
    fn a_three_quarter_arc_is_not_a_loop() {
        for ratio in [1.0f32, 2.0] {
            let radii = Vec2::new(400.0, 400.0 / ratio);
            let trace = loop_trace(Vec2::ZERO, radii, 0.6, 2.0, -0.25);
            assert!(
                !matches!(free(&trace), Some(AssistShape::Ellipse { .. })),
                "{ratio}:1 three-quarter arc was closed into an ellipse"
            );
        }
    }

    /// The start of a drag is placed deliberately; the end is wherever the hand had got
    /// to. So a snapped line keeps the first sample exactly and fits only its direction.
    #[test]
    fn a_snapped_line_starts_where_the_stroke_started() {
        let (a, b) = (Vec2::new(10.0, 20.0), Vec2::new(310.0, 90.0));
        let trace = line_trace(a, b, 3.0);
        let AssistShape::Line { a: start, .. } = free(&trace).expect("a line") else {
            panic!("not a line");
        };
        assert_eq!(start, trace[0], "the drawn start was moved to {start}");
    }

    /// The bar has to *decline*. A considered curve that happens to end near where it
    /// started must survive being held.
    #[test]
    fn a_deliberate_curve_does_not_snap() {
        // A cardioid-ish loop: closed, but no ellipse.
        let trace = trace_of(80, 0.0, |t| {
            let u = t * TAU;
            let r = 100.0 * (1.0 + 0.45 * (2.0 * u).cos());
            Vec2::new(r * u.cos(), r * u.sin())
        });
        assert_eq!(free(&trace), None);
    }

    #[test]
    fn a_bowed_stroke_is_not_a_line() {
        // 300px across with a 40px sagitta — a curve somebody meant.
        let trace = trace_of(40, 0.0, |t| Vec2::new(300.0 * t, 160.0 * t * (1.0 - t)));
        assert_eq!(free(&trace), None);
    }

    #[test]
    fn a_twitch_is_not_a_shape() {
        assert_eq!(free(&[Vec2::ZERO; 4]), None);
        let tiny = line_trace(Vec2::ZERO, Vec2::new(6.0, 0.0), 0.0);
        assert_eq!(free(&tiny), None);
    }

    /// The realized path has to *be* the shape, not merely resemble it — including at
    /// the clamped ends, which is what the ellipse's least-squares placement buys.
    #[test]
    fn a_snapped_line_is_straight() {
        let (a, b) = (Vec2::new(-50.0, 30.0), Vec2::new(250.0, 130.0));
        let pen = PenProfile::of(&[ControlPoint::at(a), ControlPoint::at(b)]);
        let path = AssistShape::Line {
            a,
            b,
            on_axis: false,
        }
        .to_path(&pen, 12);
        let poly = flatten(&path, FLATTEN_TOLERANCE);
        let dir = (b - a).normalize();
        let worst = poly
            .iter()
            .map(|s| (s.pos - a).perp_dot(dir).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02, "drifted {worst}px off the line");
    }

    #[test]
    fn a_snapped_ellipse_is_round() {
        for radius in [40.0f32, 200.0, 1200.0] {
            let shape = AssistShape::Ellipse {
                center: Vec2::new(7.0, -13.0),
                radii: Vec2::splat(radius),
                angle: 0.0,
                phase: 0.3,
                winding: 1.0,
                plane: None,
            };
            let pen = PenProfile::of(&[ControlPoint::at(Vec2::ZERO), ControlPoint::at(Vec2::X)]);
            let path = shape.to_path(&pen, 8);
            let poly = flatten(&path, FLATTEN_TOLERANCE);
            let worst = poly
                .iter()
                .map(|s| (s.pos.distance(Vec2::new(7.0, -13.0)) - radius).abs())
                .fold(0.0f32, f32::max);
            assert!(worst <= ELLIPSE_ERROR, "radius {radius}: off by {worst}px");
        }
    }

    /// The pen channels ride the ideal shape rather than being flattened out of it.
    #[test]
    fn a_snapped_stroke_keeps_its_pressure() {
        let drawn: Vec<ControlPoint> = (0..9)
            .map(|i| {
                let t = i as f32 / 8.0;
                ControlPoint {
                    pos: Vec2::new(t * 300.0, 0.0),
                    // A swell: light at both ends, full in the middle.
                    pressure: 0.15 + 0.85 * (t * std::f32::consts::PI).sin(),
                    tilt: Vec2::ZERO,
                    time: t,
                }
            })
            .collect();
        let pen = PenProfile::of(&drawn);
        let path = AssistShape::Line {
            a: Vec2::ZERO,
            b: Vec2::new(300.0, 0.0),
            on_axis: false,
        }
        .to_path(&pen, drawn.len());
        let mid = path[path.len() / 2].pressure;
        let ends = path[0].pressure.max(path[path.len() - 1].pressure);
        assert!(mid > 0.8, "the swell was flattened out (mid {mid})");
        assert!(ends < 0.4, "the light ends were filled in (ends {ends})");
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

    // --- lines on a guide axis (§20.6) -------------------------------------

    use stark_model::document::PerspectiveGuide;

    /// A guide in general position: three finite vanishing points, none of its axes
    /// level, so nothing here can pass by accidentally agreeing with the canvas.
    fn guide() -> PerspectiveGuide {
        PerspectiveGuide {
            center: Vec2::new(120.0, -60.0),
            focal: 700.0,
            rotation: glam::Quat::from_rotation_z(0.2)
                * glam::Quat::from_rotation_x(0.3)
                * glam::Quat::from_rotation_y(0.55),
            ..Default::default()
        }
    }

    /// A trace drawn from `a`, `len` px long, `off` radians away from the guide line of
    /// `axis` through `a` — the shape of every question here: how far off may a hand be?
    fn aimed(
        g: &PerspectiveGuide,
        axis: usize,
        a: Vec2,
        len: f32,
        off: f32,
        wobble: f32,
    ) -> Vec<Vec2> {
        let u = g.pencils()[axis]
            .expect("axis shown")
            .through(a)
            .expect("a line through a");
        let dir = Vec2::from_angle(off).rotate(u);
        line_trace(a, a + dir * len, wobble)
    }

    fn as_line(shape: AssistShape) -> (Vec2, Vec2, bool) {
        match shape {
            AssistShape::Line { a, b, on_axis } => (a, b, on_axis),
            other => panic!("recognized {other:?}, not a line"),
        }
    }

    /// The feature itself: a line drawn roughly toward a vanishing point comes back
    /// aimed exactly at it, *from where the hand started it* — turned onto the grid,
    /// not moved onto it.
    #[test]
    fn a_line_aimed_near_an_axis_snaps_onto_it() {
        let g = guide();
        let vps = g.scene().vps;
        let up = Scaffold::of(std::slice::from_ref(&g));
        let start = Vec2::new(-260.0, 210.0);
        for (axis, vp) in vps.iter().enumerate() {
            let vp = vp.expect("3-point: all finite");
            let trace = aimed(&g, axis, start, 420.0, 0.05, 2.5);
            let (a, b, on_axis) = as_line(recognize(&trace, TOL, &up).expect("a line"));
            assert!(on_axis, "axis {axis} was not taken");
            assert_eq!(a, start, "the drawn start moved to {a}");
            let (u, to_vp) = ((b - a).normalize(), (vp - a).normalize());
            assert!(
                u.perp_dot(to_vp).abs() < 1e-3,
                "axis {axis}: the snapped line points {u} and the VP is at {to_vp}"
            );
        }
    }

    /// ...and it has to decline. A line drawn deliberately across the grid is a line
    /// the artist meant, and the grid does not get to claim it.
    #[test]
    fn a_line_across_the_grid_keeps_its_own_direction() {
        let g = guide();
        let up = Scaffold::of(std::slice::from_ref(&g));
        let start = Vec2::new(-260.0, 210.0);
        // A fifth of a turn off the X axis — and, in this pose, well away from the
        // other two as well.
        let trace = aimed(&g, 0, start, 420.0, 0.6, 2.5);
        let (a, b, on_axis) = as_line(recognize(&trace, TOL, &up).expect("a line"));
        assert!(!on_axis, "a deliberate diagonal was pulled onto an axis");
        let drawn = (trace[trace.len() - 1] - trace[0]).normalize();
        assert!((b - a).normalize().perp_dot(drawn).abs() < 0.02);
    }

    /// The snap is one question about direction, so it cannot depend on how the trace
    /// was *scaled*: the bar is a fraction of the line's own length, which is the same
    /// cone for a short stroke and a long one.
    #[test]
    fn the_snap_window_is_an_angle() {
        let g = guide();
        let up = Scaffold::of(std::slice::from_ref(&g));
        let start = Vec2::new(40.0, 320.0);
        for len in [120.0f32, 400.0, 1600.0] {
            // Inside the cone at every length…
            let near = aimed(&g, 2, start, len, 0.04, 0.0);
            let (_, _, on) = as_line(recognize(&near, TOL, &up).expect("a line"));
            assert!(on, "{len}px missed an axis 2.3° away");
            // …out to the edge of it, which is the width the *aim* needs rather than
            // the width the wrist needs: a vanishing point off the page is not
            // somewhere the eye can look while drawing toward it.
            let wide = aimed(&g, 2, start, len, 0.12, 0.0);
            let (_, _, on) = as_line(recognize(&wide, TOL, &up).expect("a line"));
            assert!(on, "{len}px missed an axis 6.9° away");
            // …and outside it at every length.
            let far = aimed(&g, 2, start, len, 0.25, 0.0);
            let (_, _, on) = as_line(recognize(&far, TOL, &up).expect("a line"));
            assert!(!on, "{len}px was claimed by an axis 14° away");
        }
    }

    /// What the artist cannot see cannot bend a line: the same trace, with the
    /// guide's eye shut, is the hand's own.
    ///
    /// The eye is per-client, so it is not a field of the guide any more (§20.5) —
    /// shutting one is the engine leaving it out of the scaffold, and what this
    /// pins is that the assist reads the scaffold and nothing else. A guide it was
    /// not handed cannot reach a stroke however visible it is to somebody.
    #[test]
    fn a_hidden_guide_does_not_snap() {
        let g = guide();
        let start = Vec2::new(-260.0, 210.0);
        let trace = aimed(&g, 1, start, 420.0, 0.05, 2.5);
        assert!(
            as_line(
                recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("a line")
            )
            .2
        );

        let shut = Scaffold::of(std::iter::empty());
        assert!(!as_line(recognize(&trace, TOL, &shut).expect("a line")).2);
    }

    /// A curve is still a curve. The grid may only choose *which* line a line is, and
    /// never promote something that was not one — the two bars are not interchangeable.
    #[test]
    fn a_bowed_stroke_near_an_axis_is_still_not_a_line() {
        let g = guide();
        let start = Vec2::new(-260.0, 210.0);
        let u = g.pencils()[0].unwrap().through(start).unwrap();
        // 300px along the axis with a 40px sagitta across it — the §6.9 curve, laid on
        // the grid.
        let trace = trace_of(40, 0.0, |t| {
            start + u * (300.0 * t) + u.perp() * (160.0 * t * (1.0 - t))
        });
        assert_eq!(
            recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))),
            None
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

    // --- circles on a plane (§20.7) ----------------------------------------

    /// Where the §20.7 tests put their circles: near the foot of each plane's normal,
    /// which is the part of a plane that is squarely in front of the eye. Out toward a
    /// plane's own vanishing line a circle straddles it and has no bounded image at
    /// all — a real answer (`circle_seen` declines), and not the case under test.
    const ON_PLANE: Vec2 = Vec2::new(0.05, -0.04);

    /// Mean radius, in canvas px, of the *image* every §20.7 test draws around.
    ///
    /// Sized on the canvas rather than in the plane, because a plane radius is not a
    /// comparable quantity across three planes lying at three depths: the same 0.12 in
    /// plane units images to 130px on one of this guide's planes and to 1500px on
    /// another, and a residual bar quoted as a fraction of the shape means something
    /// quite different at those two sizes. What a hand draws is a few hundred px
    /// across, so that is what these ask about.
    const DRAWN_SIZE: f32 = 180.0;

    fn as_ellipse(shape: AssistShape) -> (Vec2, Vec2, f32, Option<AxisPlane>) {
        match shape {
            AssistShape::Ellipse {
                center,
                radii,
                angle,
                plane,
                ..
            } => (center, radii, angle, plane),
            other => panic!("recognized {other:?}, not an ellipse"),
        }
    }

    /// The circle at [`ON_PLANE`] on plane `k` whose image is [`DRAWN_SIZE`] across the
    /// mean: the plane, its radius there, and the ellipse it is seen as.
    fn perspective_circle(g: &PerspectiveGuide, k: usize) -> (AxisPlane, f32, Ellipse) {
        let plane = g.planes()[k].expect("plane shown");
        let probe = plane.circle_seen(ON_PLANE, 0.02).expect("a bounded image");
        let radius = 0.02 * DRAWN_SIZE / probe.scale();
        let seen = plane
            .circle_seen(ON_PLANE, radius)
            .expect("a bounded image");
        (plane, radius, seen)
    }

    /// A hand-drawn loop around that circle's image, with its eccentricity stretched by
    /// `wrong` and its tilt turned by `tilt` — exactly the pair of things a hand gets
    /// wrong about a circle in perspective, and the two the snap exists to fix.
    fn perspective_loop(g: &PerspectiveGuide, k: usize, wrong: f32, tilt: f32) -> Vec<Vec2> {
        let (
            _,
            _,
            Ellipse {
                center,
                radii,
                angle,
            },
        ) = perspective_circle(g, k);
        loop_trace(
            center,
            Vec2::new(radii.x, radii.y * wrong),
            angle + tilt,
            2.0,
            0.0,
        )
    }

    /// The feature: a loop drawn roughly where a circle on the grid would be comes back
    /// as *that circle*, not as the ellipse the hand actually managed.
    #[test]
    fn a_loop_drawn_in_perspective_snaps_to_a_circle() {
        let g = guide();
        let up = Scaffold::of(std::slice::from_ref(&g));
        for k in 0..3 {
            let (plane, radius, _) = perspective_circle(&g, k);
            // A tenth too round and leaning 3° off — a hand that saw it about right.
            let trace = perspective_loop(&g, k, 1.1, 0.05);
            let (center, radii, angle, on) =
                as_ellipse(recognize(&trace, TOL, &up).expect("an ellipse"));
            let on = on.unwrap_or_else(|| panic!("plane {k} was not taken"));
            assert_eq!(Some(plane), Some(on), "plane {k}: another plane claimed it");

            // ...and it is a circle *there*: at the place and the size it was drawn at.
            let (flat, r) = on
                .circle_behind(Ellipse::new(center, radii, angle))
                .expect("a circle behind it");
            assert!(
                flat.distance(ON_PLANE) < 0.15 * radius,
                "plane {k}: the circle sits at {flat}, not {ON_PLANE}"
            );
            assert!(
                (r - radius).abs() < 0.2 * radius,
                "plane {k}: radius {r}, drawn around {radius}"
            );
        }
    }

    /// The snapped shape has to *be* the perspective circle, not merely resemble one:
    /// its whole outline pulls back to one constant radius. That is the statement the
    /// canvas ellipse on its own cannot make.
    #[test]
    fn the_snapped_loop_is_round_in_its_plane() {
        let g = guide();
        let trace = perspective_loop(&g, 2, 1.1, -0.05);
        let (center, radii, angle, plane) = as_ellipse(
            recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("an ellipse"),
        );
        let plane = plane.expect("a perspective circle");
        let (flat, r) = plane
            .circle_behind(Ellipse::new(center, radii, angle))
            .expect("a circle");

        let frame = Frame::new(center, radii, angle);
        let worst = (0..64)
            .map(|i| {
                let p = frame.point(i as f32 / 64.0 * TAU);
                let q = plane.to_plane(p).expect("on the plane");
                (q.distance(flat) - r).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3 * r, "the outline strays {worst} off round");
    }

    /// And it has to decline. A loop no plane would put there is the ellipse it was
    /// drawn as — an ellipse is a shape somebody can mean.
    #[test]
    fn a_loop_no_plane_would_put_there_stays_an_ellipse() {
        let g = guide();
        // Twice as eccentric as the plane calls for, and turned 40° out of its tilt.
        let trace = perspective_loop(&g, 2, 1.6, 0.5);
        let (_, _, _, plane) = as_ellipse(
            recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("an ellipse"),
        );
        assert!(plane.is_none(), "a deliberate ellipse was made a circle");
    }

    /// What the artist cannot see cannot reshape a loop — and a plane is switched off
    /// directly, being the thing the bar's chips name.
    #[test]
    fn an_unshown_plane_does_not_snap() {
        let mut g = guide();
        let trace = perspective_loop(&g, 2, 1.1, -0.05);
        let claimed = |g: &PerspectiveGuide| {
            as_ellipse(
                recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(g))).expect("an ellipse"),
            )
            .3
            .is_some()
        };
        assert!(claimed(&g));

        // The loop is a circle on pair 2, and only that plane's own flag governs
        // it: switching off the other two leaves it claimed, switching off its
        // own takes it.
        g.pairs = [false, false, true];
        assert!(claimed(&g), "a plane went with its neighbours");
        g.pairs = [true, true, false];
        assert!(!claimed(&g), "a plane survived being switched off");
        // …and a guide this client has hidden reaches nothing at all. The eye is
        // per-client now (§20.5), so shutting one is the engine leaving the guide
        // out of the scaffold rather than a flag on the guide itself.
        g.pairs = [true; 3];
        assert!(claimed(&g));
        assert!(
            as_ellipse(
                recognize(&trace, TOL, &Scaffold::of(std::iter::empty())).expect("an ellipse")
            )
            .3
            .is_none(),
            "a guide that was never handed over reshaped a loop"
        );
    }

    /// The width of the window is the feature, not an implementation detail: a hand
    /// that has to draw the circle accurately to be given the circle has been given
    /// nothing. A loop a fifth too round and leaning 6° out of its plane's tilt — well
    /// past what an eye can judge — is still that plane's circle, and
    /// [`GUIDE_CIRCLE_RESIDUAL`] is set wide enough to say so.
    #[test]
    fn a_loop_only_roughly_in_perspective_still_snaps() {
        let g = guide();
        let up = Scaffold::of(std::slice::from_ref(&g));
        for k in 0..3 {
            let (plane, _, _) = perspective_circle(&g, k);
            for (wrong, tilt) in [(1.2f32, 0.0f32), (1.0, 0.105), (1.12, 0.06)] {
                let trace = perspective_loop(&g, k, wrong, tilt);
                let (.., on) = as_ellipse(recognize(&trace, TOL, &up).expect("an ellipse"));
                assert_eq!(
                    on,
                    Some(plane),
                    "plane {k}: {wrong:.2} round, {:.1}° off was not claimed",
                    tilt.to_degrees()
                );
            }
        }
    }

    /// Steering a perspective circle sizes it *in its plane* and leaves it there, so
    /// the canvas ellipse's eccentricity and tilt go on following the grid rather than
    /// the hand — which is the whole reason the plane is carried at all.
    ///
    /// The pointer is put on the circle half again as wide, at a place of its own
    /// choosing: the drag is therefore mostly *across* the shape, and the answer has to
    /// be exactly the circle half again as wide, because sideways travel means nothing
    /// to a shape with no orientation to turn.
    #[test]
    fn a_perspective_circle_is_steered_in_its_plane() {
        let g = guide();
        let trace = perspective_loop(&g, 2, 1.08, 0.04);
        let shape =
            recognize(&trace, TOL, &Scaffold::of(std::slice::from_ref(&g))).expect("an ellipse");
        let (center, radii, angle, plane) = as_ellipse(shape);
        let plane = plane.expect("a perspective circle");
        let (flat, r) = plane
            .circle_behind(Ellipse::new(center, radii, angle))
            .expect("a circle behind it");

        let wider = plane.circle_seen(flat, r * 1.5).expect("a wider circle");
        let target = Frame::of(wider).point(1.0);
        let (center, radii, angle, kept) = as_ellipse(shape.adjust(shape.grip(), target));
        assert_eq!(kept, Some(plane), "the circle left its plane");

        let (moved, wider) = plane
            .circle_behind(Ellipse::new(center, radii, angle))
            .expect("still a circle");
        assert!(
            moved.distance(flat) < 1e-3 * r,
            "the centre wandered from {flat} to {moved}"
        );
        assert!(
            (wider - 1.5 * r).abs() < 1e-3 * r,
            "radius {wider}, expected {}",
            1.5 * r
        );

        // ...and it is a function of the total travel, like every adjustment (§16.6).
        let stepped = (1..=8).fold(shape, |_, i| {
            shape.adjust(shape.grip(), shape.grip().lerp(target, i as f32 / 8.0))
        });
        assert_eq!(shape.adjust(shape.grip(), target), stepped);
    }
}
