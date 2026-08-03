//! Drawing guides — the perspective grid (§20).
//!
//! One projective camera, three familiar cases. A "1-point", "2-point" or
//! "3-point" perspective is not three tools: it is one camera whose view
//! direction happens to be parallel to none, one, or two of the world axes,
//! and the guide keeps that structure visible instead of hiding it (§20.1).
//! The state is exactly a camera — a **center of view** (principal point), a
//! **focal length**, and an orientation — and everything an artist is shown
//! is derived from it:
//!
//! - each world axis's **vanishing point**, the projection of a direction —
//!   a projective point that is allowed to be at infinity (§20.2);
//! - each axis pair's **vanishing line** (the horizon, for the ground pair);
//! - the **45° circle**, radius = focal length: the cone at 45° off the view
//!   axis, the classical "keep the drawing inside this" bound (§20.1);
//! - the three **station points**, the eye rotated into the picture plane
//!   about each vanishing line — each sits on the Thales circle over its two
//!   vanishing points, which is why it sees them at a right angle (§20.2).
//!
//! [`PerspectiveGuide`] is one guide; the session carries a list of them
//! (§20.5) and the panel edits it whole through
//! [`ViewCommand::SetGuides`](crate::command::ViewCommand::SetGuides).
//! [`PerspectiveGuide::scene`] derives the [`GuideScene`] the compositor's
//! guide pass draws (§20.4), and [`PerspectiveGuide::dragged`] is the direct
//! manipulation: the grabbed direction follows the pointer, snapping to and
//! lockable about the world axes (§20.5). Everything here is plain CPU math,
//! computed once per render and unit-tested against the classical theorems.

use glam::{Mat3, Quat, Vec2, Vec3};

/// A world-axis direction whose camera-space `z` is smaller than this is taken
/// to vanish *at infinity* — its lines are drawn parallel. At any plausible
/// focal length the finite vanishing point would already be tens of millions
/// of canvas pixels away, far beyond where the distinction could move a line
/// by a texel.
const VP_FINITE_EPS: f32 = 1e-4;

/// A pair plane whose normal has less than this much component in the picture
/// plane has its vanishing line at infinity (the plane faces the camera
/// square-on), so there is no line — and no station point — to draw.
const LINE_EPS: f32 = 1e-4;

/// How close a free drag's rotation axis must lie to a world axis before the
/// drag snaps to turning purely about it (§20.5): cos 15°. Wide enough to
/// fall into deliberately, narrow enough that a diagonal drag stays free.
const SNAP_COS: f32 = 0.966;

/// The perspective-grid guide: a camera, plus how densely to dress it
/// (§20.1). View state — per-client, never logged, never sent — one entry of
/// the list the `Session` carries (§20.5).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveGuide {
    /// Whether this guide is drawn — the list row's eye (§20.5).
    pub visible: bool,
    /// The **center of view** (principal point): where the view axis meets the
    /// picture plane, in canvas px. The one point of the drawing seen without
    /// obliquity, and the center of the 45° circle.
    pub center: Vec2,
    /// Focal length in canvas px: the eye's distance from the picture plane,
    /// and therefore also the radius of the 45° circle (`tan 45° = 1`).
    pub focal: f32,
    /// The camera's orientation: rotates the world's orthogonal axis frame
    /// into camera space (x right, y down to match the canvas, z forward).
    /// Identity looks straight down the world Z axis — 1-point perspective —
    /// and every other case is a turn of this one quaternion; the *count of
    /// finite vanishing points* is the only thing "1/2/3-point" ever names.
    pub rotation: Quat,
    /// Fan lines per half-turn of each axis's plane pencil (§20.3). The step
    /// between guide lines is `π / density` of *visual angle* — equal steps as
    /// the eye turns, not equal spacing on the canvas, which is what makes the
    /// same slider mean the same thing in every perspective case.
    pub density: u32,
    /// Master opacity of the whole overlay.
    pub opacity: f32,
    /// Which world axes' fans are drawn (X, Y, Z). The markers derived from an
    /// axis (its vanishing point, its pair lines) follow its fan.
    pub axes: [bool; 3],
}

impl Default for PerspectiveGuide {
    /// Visible, centred on the canvas origin, at a moderate lens, turned to
    /// the most-reached-for case (2-point), with 15° fans. The caller placing
    /// a new guide moves `center` to where the artist is looking.
    fn default() -> Self {
        Self {
            visible: true,
            center: Vec2::ZERO,
            focal: 900.0,
            rotation: Quat::from_rotation_y(30f32.to_radians()),
            density: 12,
            opacity: 0.65,
            axes: [true; 3],
        }
    }
}

impl PerspectiveGuide {
    /// The world axes as directions in **camera space** (§20.1) — the columns
    /// of [`rotation`](Self::rotation) as a matrix.
    ///
    /// Directions, not points: everything downstream treats them projectively,
    /// so an axis and its negation describe the same pencil of lines and no
    /// consumer depends on a sign.
    pub fn axis_dirs(&self) -> [Vec3; 3] {
        let m = Mat3::from_quat(self.rotation);
        [m.x_axis, m.y_axis, m.z_axis]
    }

    /// The eye's ray through canvas point `p`, unit, in camera space. The
    /// shared first step of projection and of every canvas gesture: what the
    /// hand touches on the picture plane *is* a direction in the world.
    pub fn ray(&self, p: Vec2) -> Vec3 {
        Vec3::new(
            (p.x - self.center.x) / self.focal,
            (p.y - self.center.y) / self.focal,
            1.0,
        )
        .normalize()
    }

    /// The orbit drag (§20.5): the world direction grabbed at `from` follows
    /// the pointer to `to`, by rotating the whole frame — always computed from
    /// the drag's *start* state, so a long drag cannot drift and the snap
    /// decision cannot flicker.
    ///
    /// `locked` axes are held fixed. Rotations fixing one axis are exactly the
    /// turns about it, so one lock constrains the drag to that axis's orbit —
    /// lock the vertical and a 2-point setup stays 2-point under any drag —
    /// and two locks pin the frame entirely (the identity is the only rotation
    /// fixing two axes). Unlocked, the drag is a free grab, *snapping* to a
    /// pure axis turn whenever the rotation it implies lies within
    /// [`SNAP_COS`] of a world axis.
    #[must_use]
    pub fn dragged(&self, from: Vec2, to: Vec2, locked: [bool; 3]) -> Self {
        if locked.iter().filter(|l| **l).count() >= 2 {
            return *self;
        }
        let (r0, r1) = (self.ray(from), self.ray(to));
        let dirs = self.axis_dirs();
        let delta = if let Some(i) = locked.iter().position(|l| *l) {
            axis_turn(dirs[i], r0, r1)
        } else {
            let w = r0.cross(r1);
            if w.length_squared() < 1e-12 {
                Quat::IDENTITY
            } else {
                // Snap to the closest world axis the free rotation nearly is.
                let w = w.normalize();
                let close = (0..3)
                    .map(|i| (w.dot(dirs[i]).abs(), i))
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .filter(|(d, _)| *d > SNAP_COS);
                match close {
                    Some((_, i)) => axis_turn(dirs[i], r0, r1),
                    None => Quat::from_rotation_arc(r0, r1),
                }
            }
        };
        Self {
            rotation: (delta * self.rotation).normalize(),
            ..*self
        }
    }

    /// Everything the guide pass draws, derived from the camera (§20.2). Cheap
    /// — a rotation and a handful of products — so it is recomputed per render
    /// rather than cached beside the state it would shadow.
    pub fn scene(&self) -> GuideScene {
        let dirs = self.axis_dirs();
        let c = self.center;
        let f = self.focal;

        // Vanishing point of axis `d`: the picture-plane point its lines
        // converge to, `c + f·(d.x, d.y)/d.z` — projective, so an axis lying
        // in the picture plane (d.z ≈ 0) vanishes at infinity instead.
        let vps =
            dirs.map(|d| (d.z.abs() > VP_FINITE_EPS).then(|| c + Vec2::new(d.x, d.y) * (f / d.z)));

        // Pair k spans axes (k, k+1): its vanishing line and station point.
        let mut lines = [None; 3];
        let mut stations = [None; 3];
        for k in 0..3 {
            let m = dirs[k].cross(dirs[(k + 1) % 3]);
            // The vanishing line of the plane with (unit) normal `m` is the
            // trace of the parallel plane through the eye: with the eye at
            // distance f over c, that is `m.x·x + m.y·y + (f·m.z − m·c) = 0`,
            // normalized so its first two coefficients are a unit normal and
            // evaluating it *is* signed canvas-px distance. A plane facing the
            // camera square-on (m.xy ≈ 0) has its line at infinity: nothing to
            // draw, and no station point either.
            let planar = Vec2::new(m.x, m.y);
            let len = planar.length();
            if len < LINE_EPS {
                continue;
            }
            let n = planar / len;
            let offset = (f * m.z - planar.dot(c)) / len;
            lines[k] = Some(Vec3::new(n.x, n.y, offset));

            // The station point: the eye, rotated into the picture plane about
            // the vanishing line (§20.2). The eye sits at height f over c, at
            // distance √(a² + f²) from the line (a = the line's distance from
            // c); rotating preserves that distance, landing on the ray from
            // the foot of c's perpendicular through c. With the view axis in
            // the pair plane (a ≈ 0 — exact 2-point) either side is the same
            // rotation; the canvas-down side is the drawing-board convention.
            let s = n.dot(c) + offset;
            let a = s.abs();
            let foot = c - n * s;
            let u = if a > f * 1e-3 {
                n * s.signum()
            } else if n.y.abs() > 0.5 {
                n * n.y.signum()
            } else {
                n * n.x.signum()
            };
            stations[k] = Some(foot + u * (a * a + f * f).sqrt());
        }

        GuideScene {
            center: c,
            focal: f,
            step: std::f32::consts::PI / (self.density.clamp(2, 90) as f32),
            opacity: self.opacity.clamp(0.0, 1.0),
            dirs,
            axis_alpha: self.axes.map(|on| if on { 1.0 } else { 0.0 }),
            lines,
            vps,
            stations,
        }
    }
}

/// The turn about `axis` that best carries ray `r0` toward ray `r1`: the
/// signed angle between their projections onto the plane the axis pierces.
/// A ray lying along the axis has no projection and asks for nothing.
fn axis_turn(axis: Vec3, r0: Vec3, r1: Vec3) -> Quat {
    let u = r0 - axis * axis.dot(r0);
    let v = r1 - axis * axis.dot(r1);
    if u.length_squared() < 1e-8 || v.length_squared() < 1e-8 {
        return Quat::IDENTITY;
    }
    let (u, v) = (u.normalize(), v.normalize());
    Quat::from_axis_angle(axis, u.cross(v).dot(axis).atan2(u.dot(v)))
}

/// The derived, draw-ready guide: what the compositor's guide pass uniform
/// carries (§20.4). All canvas-space; the pass adds the view mapping.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GuideScene {
    /// Center of view, canvas px.
    pub center: Vec2,
    /// Focal length, canvas px — also the 45° circle's radius.
    pub focal: f32,
    /// Fan step, radians of visual angle (§20.3).
    pub step: f32,
    /// Master opacity, 0..=1.
    pub opacity: f32,
    /// World axes in camera space. The fan for axis `i` measures its pencil
    /// angle against the other two axes, so the pair planes — the vanishing
    /// lines — are themselves fan lines (§20.3).
    pub dirs: [Vec3; 3],
    /// Per-axis fan opacity (0 = axis hidden).
    pub axis_alpha: [f32; 3],
    /// Vanishing line of pair `(k, k+1)` as `(n.x, n.y, offset)` with `n`
    /// unit: evaluating against a canvas point gives signed distance in px.
    /// `None` when the pair plane faces the camera square-on.
    pub lines: [Option<Vec3>; 3],
    /// Finite vanishing points, canvas px; `None` at infinity.
    pub vps: [Option<Vec2>; 3],
    /// Station point of pair `(k, k+1)`, canvas px; `None` with its line.
    pub stations: [Option<Vec2>; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A guide at the classical Euler pose — the tests state poses this way
    /// because the theorems are stated this way; the *state* is one
    /// quaternion.
    fn guide(yaw: f32, pitch: f32, roll: f32) -> PerspectiveGuide {
        PerspectiveGuide {
            center: Vec2::new(320.0, -140.0),
            focal: 800.0,
            rotation: Quat::from_rotation_z(roll)
                * Quat::from_rotation_x(pitch)
                * Quat::from_rotation_y(yaw),
            ..Default::default()
        }
    }

    /// 1-point: the view axis meets the picture plane at the center of view,
    /// so that is where the Z axis vanishes — and the transverse axes, lying
    /// in the picture plane, vanish at infinity.
    #[test]
    fn one_point_vanishes_at_the_center_of_view() {
        let g = guide(0.0, 0.0, 0.0);
        let s = g.scene();
        assert!(s.vps[0].is_none(), "X should vanish at infinity");
        assert!(s.vps[1].is_none(), "Y should vanish at infinity");
        let vz = s.vps[2].expect("Z vanishes on the canvas");
        assert!((vz - g.center).length() < 1e-3, "got {vz:?}");
        // The X/Y pair plane faces the camera square-on: no line, no station.
        assert!(s.lines[0].is_none());
        assert!(s.stations[0].is_none());
    }

    /// 2-point: verticals stay parallel, the two ground axes vanish on a
    /// horizon through the center of view, and the ground pair's station point
    /// hangs exactly one focal length below it — the drawing-board layout.
    #[test]
    fn two_point_horizon_runs_through_the_center_of_view() {
        let g = guide(0.6, 0.0, 0.0);
        let s = g.scene();
        let vx = s.vps[0].expect("X finite");
        let vz = s.vps[2].expect("Z finite");
        assert!(s.vps[1].is_none(), "verticals stay parallel");
        assert!((vx.y - g.center.y).abs() < 1e-3);
        assert!((vz.y - g.center.y).abs() < 1e-3);
        // Pair 2 spans (Z, X): the ground. Its line passes through c…
        let l = s.lines[2].expect("horizon");
        assert!((l.x * g.center.x + l.y * g.center.y + l.z).abs() < 1e-3);
        // …and its station point sits one focal length below the horizon.
        let sp = s.stations[2].expect("station point");
        assert!(
            (sp - (g.center + Vec2::new(0.0, g.focal))).length() < 1e-2,
            "got {sp:?}"
        );
    }

    /// The classical theorem the whole construction hangs on: for a camera
    /// with orthogonal world axes, the center of view is the **orthocenter**
    /// of the vanishing-point triangle.
    #[test]
    fn the_center_of_view_is_the_orthocenter_of_the_vp_triangle() {
        let g = guide(0.5, 0.35, 0.2);
        let s = g.scene();
        let vp: Vec<Vec2> = s
            .vps
            .iter()
            .map(|v| v.expect("3-point: all finite"))
            .collect();
        for i in 0..3 {
            let (a, b, o) = (vp[(i + 1) % 3], vp[(i + 2) % 3], vp[i]);
            // The altitude from each vertex through c is perpendicular to the
            // opposite side.
            let side = b - a;
            let alt = o - g.center;
            assert!(
                side.dot(alt).abs() / (side.length() * alt.length()) < 1e-3,
                "altitude {i} not perpendicular"
            );
        }
    }

    /// Each station point is the eye rotated into the picture plane, so it
    /// still sees its pair's two vanishing points at the right angle they
    /// subtend in the world — it lies on the Thales circle over them.
    #[test]
    fn station_points_see_their_vanishing_points_at_right_angles() {
        for (yaw, pitch, roll) in [(0.5, 0.35, 0.2), (0.9, -0.4, 0.0), (0.3, 0.0, 1.0)] {
            let s = guide(yaw, pitch, roll).scene();
            for k in 0..3 {
                let (Some(sp), Some(vi), Some(vj)) = (s.stations[k], s.vps[k], s.vps[(k + 1) % 3])
                else {
                    continue;
                };
                let (u, v) = (vi - sp, vj - sp);
                assert!(
                    u.dot(v).abs() / (u.length() * v.length()) < 1e-3,
                    "pair {k} at yaw {yaw}, pitch {pitch}, roll {roll}: {:?}",
                    u.dot(v)
                );
            }
        }
    }

    /// The fan phase convention (§20.3): axis `i`'s pencil angle is measured
    /// against the other two axes, so the plane spanned with either partner —
    /// whose trace is a vanishing line — lands on a multiple of a quarter
    /// turn. With an even density, every vanishing line is a fan line.
    #[test]
    fn pair_planes_land_on_quarter_turns_of_the_fan() {
        let s = guide(0.5, 0.35, 0.2).scene();
        for i in 0..3 {
            for step in [1, 2] {
                let j = (i + step) % 3;
                // The pencil coordinate of the plane span(a_i, a_j): its
                // normal, resolved against the measuring basis.
                let n = s.dirs[i].cross(s.dirs[j]);
                let u = n.dot(s.dirs[(i + 2) % 3]);
                let v = n.dot(s.dirs[(i + 1) % 3]);
                let theta = v.atan2(u);
                let quarter = theta / std::f32::consts::FRAC_PI_2;
                assert!(
                    (quarter - quarter.round()).abs() < 1e-3,
                    "axis {i} with partner {j}: θ = {theta}"
                );
            }
        }
    }

    /// A direction 45° off the view axis projects onto the circle of radius
    /// `focal` about the center of view — the 45° circle is exactly that
    /// statement, so it is checked rather than described.
    #[test]
    fn a_45_degree_direction_projects_onto_the_circle() {
        let g = guide(0.0, 0.0, 0.0);
        let d = Vec3::new(0.0, 1.0, 1.0).normalize(); // 45° from +z
        let p = g.center + Vec2::new(d.x, d.y) * (g.focal / d.z);
        assert!(((p - g.center).length() - g.focal).abs() < 1e-3);
    }

    // --- the orbit drag (§20.5) --------------------------------------------

    /// The drag's contract: the world direction under the pointer at the start
    /// is under it at the end. (A diagonal drag, chosen away from every axis
    /// so the snap stays out of the way.)
    #[test]
    fn a_drag_keeps_the_grabbed_direction_under_the_pointer() {
        let g = guide(0.5, 0.35, 0.2);
        let (from, to) = (
            g.center + Vec2::new(120.0, -60.0),
            g.center + Vec2::new(-152.0, 217.0),
        );
        let g2 = g.dragged(from, to, [false; 3]);
        // The grabbed direction, in *world* coordinates…
        let world = g.rotation.inverse() * g.ray(from);
        // …projected through the dragged camera, lands at `to`.
        let now = g2.rotation * world;
        assert!((now.normalize() - g2.ray(to)).length() < 1e-4);
    }

    /// One locked axis constrains the drag to turning about it: the axis's
    /// own direction — and with it its vanishing point — cannot move. Lock
    /// the vertical and 2-point stays exactly 2-point.
    #[test]
    fn a_locked_axis_survives_any_drag() {
        let g = guide(0.6, 0.0, 0.0);
        let before = g.axis_dirs()[1];
        let g2 = g.dragged(
            g.center + Vec2::new(40.0, 30.0),
            g.center + Vec2::new(-180.0, 95.0),
            [false, true, false],
        );
        assert!((g2.axis_dirs()[1] - before).length() < 1e-4);
        // Still 2-point: the verticals still vanish at infinity.
        assert!(g2.scene().vps[1].is_none());
    }

    /// Two locks pin the frame: the identity is the only rotation fixing two
    /// distinct axes, so the drag must change nothing.
    #[test]
    fn two_locks_pin_the_frame() {
        let g = guide(0.5, 0.35, 0.2);
        let g2 = g.dragged(
            g.center + Vec2::new(40.0, 30.0),
            g.center + Vec2::new(300.0, -200.0),
            [true, true, false],
        );
        assert_eq!(g.rotation, g2.rotation);
    }

    /// The snap: a horizontal drag through the center of view implies a
    /// rotation about the (near-vertical) Y axis, well within the snap cone —
    /// so it turns purely about Y and the verticals stay parallel, without
    /// any lock being held.
    #[test]
    fn a_near_axis_drag_snaps_to_a_pure_axis_turn() {
        let g = guide(0.6, 0.0, 0.0);
        let g2 = g.dragged(
            g.center + Vec2::new(-100.0, 0.0),
            g.center + Vec2::new(140.0, 0.0),
            [false; 3],
        );
        let before = g.axis_dirs()[1];
        assert!((g2.axis_dirs()[1] - before).length() < 1e-4, "snapped turn");
        assert!(g2.scene().vps[1].is_none(), "still 2-point");
        // And it did actually turn.
        assert!((g2.rotation.dot(g.rotation).abs() - 1.0).abs() > 1e-4);
    }
}
