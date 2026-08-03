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
//! [`PerspectiveGuide`] is the view-state the session holds and the panel
//! edits; [`PerspectiveGuide::scene`] derives the [`GuideScene`] the
//! compositor's guide pass draws (§20.4). Everything here is plain CPU math,
//! computed once per render and unit-tested against the classical theorems.

use glam::{Mat3, Vec2, Vec3};

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

/// The perspective-grid guide: a camera, plus how densely to dress it
/// (§20.1). View state — per-client, never logged, never sent — carried by
/// the `Session` and edited through
/// [`ViewCommand::SetGuide`](crate::command::ViewCommand::SetGuide).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveGuide {
    /// Whether the guide is drawn at all.
    pub enabled: bool,
    /// The **center of view** (principal point): where the view axis meets the
    /// picture plane, in canvas px. The one point of the drawing seen without
    /// obliquity, and the center of the 45° circle.
    pub center: Vec2,
    /// Focal length in canvas px: the eye's distance from the picture plane,
    /// and therefore also the radius of the 45° circle (`tan 45° = 1`).
    pub focal: f32,
    /// Turn of the world about its vertical axis, radians. Zero looks straight
    /// down the world Z axis: 1-point perspective.
    pub yaw: f32,
    /// Tilt of the view up/down, radians. Zero keeps the verticals parallel:
    /// with any yaw, 2-point perspective.
    pub pitch: f32,
    /// Roll of the camera about its own view axis, radians. Turns the whole
    /// figure on the canvas without changing which case it is — the count of
    /// finite vanishing points is roll-invariant, and the guide shows that.
    pub roll: f32,
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
    /// Off, centred on the canvas origin, at a moderate lens, turned to the
    /// most-reached-for case (2-point), with 15° fans.
    fn default() -> Self {
        Self {
            enabled: false,
            center: Vec2::ZERO,
            focal: 900.0,
            yaw: 30f32.to_radians(),
            pitch: 0.0,
            roll: 0.0,
            density: 12,
            opacity: 0.65,
            axes: [true; 3],
        }
    }
}

impl PerspectiveGuide {
    /// The world axes as directions in **camera space** (§20.1) — x right,
    /// y down (canvas-aligned), z forward into the scene. Columns of
    /// `Rz(roll)·Rx(pitch)·Ry(yaw)`; at rest the world axes coincide with the
    /// camera's and the guide reads as 1-point.
    ///
    /// Directions, not points: everything downstream treats them projectively,
    /// so an axis and its negation describe the same pencil of lines and no
    /// consumer depends on a sign.
    pub fn axis_dirs(&self) -> [Vec3; 3] {
        let r = Mat3::from_rotation_z(self.roll)
            * Mat3::from_rotation_x(self.pitch)
            * Mat3::from_rotation_y(self.yaw);
        [r.x_axis, r.y_axis, r.z_axis]
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

    fn guide(yaw: f32, pitch: f32, roll: f32) -> PerspectiveGuide {
        PerspectiveGuide {
            enabled: true,
            center: Vec2::new(320.0, -140.0),
            focal: 800.0,
            yaw,
            pitch,
            roll,
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
}
