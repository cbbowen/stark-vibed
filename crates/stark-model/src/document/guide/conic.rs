//! Conics on a guide's plane (§20.7): the ellipse a circle drawn *on* a plane is
//! seen as, and the circle behind an ellipse that was drawn on screen.
//!
//! Projective geometry and nothing else — a 3x3 conic, its congruence under the
//! plane's homography, and the ellipse read off its quadratic part. It knows what a
//! plane is and nothing about documents, which is why it reads as its own thing even
//! though §20.5 rightly keeps it beside the camera it is derived from.

use glam::{Mat3, Vec3};

use super::PLANE_REACH;
use crate::geom::{Ellipse, Vec2, principal_axis};

/// One plane of one guide — the plane spanned by a *pair* of axes — as a
/// **chart**: the map between the canvas and the plane's own flat, metric
/// coordinates, both ways (§20.7).
///
/// A pair plane has no depth of its own to fix, because scaling the depth
/// scales every circle on it by the same factor and leaves the *images*
/// unchanged. So the chart is taken at unit distance along the plane's normal,
/// and then it is one 3×3:
///
/// ```text
/// canvas_from_plane = K · [ a_i | a_j | a_i × a_j ]
/// ```
///
/// with `K` the lens (`focal`, `center`). The three planes of a guide are that
/// product with the axis frame's columns *cyclically shifted* — one matrix read
/// three ways — and it is invertible for every pose, since `K` and a rotation
/// both are. There is no degenerate plane to guard against, which is the whole
/// reason the chart is the representation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AxisPlane {
    /// `pub(super)` rather than private:
    /// [`PerspectiveGuide::planes`](super::PerspectiveGuide::planes) builds these and
    /// sits in `camera`, one file over. Nothing outside `guide` can reach them.
    pub(super) canvas_from_plane: Mat3,
    pub(super) plane_from_canvas: Mat3,
}

impl AxisPlane {
    /// `p` in the plane's own coordinates, or `None` when it does not lie on the
    /// plane in any useful sense — at or beyond `PLANE_REACH`.
    pub fn to_plane(&self, p: Vec2) -> Option<Vec2> {
        self.charted(p).map(|(q, _)| q)
    }

    /// `points` in the plane's own coordinates, or `None` if they are not all on
    /// one piece of it.
    ///
    /// The gate is that they share a **side of the pair's vanishing line**, which
    /// is where the chart's homogeneous coordinate changes sign. No circle in
    /// front of the eye is ever *seen* across its plane's vanishing line — the
    /// line is the image of that plane's infinity — so a trace that crosses one
    /// cannot be a circle on that plane, and pulling it back would answer with
    /// the two branches of a hyperbola instead.
    pub fn chart(&self, points: &[Vec2]) -> Option<Vec<Vec2>> {
        let mut side = 0.0f32;
        points
            .iter()
            .map(|p| {
                let (q, w) = self.charted(*p)?;
                if side * w < 0.0 {
                    return None;
                }
                side = w;
                Some(q)
            })
            .collect()
    }

    /// The ellipse a circle of `radius` about `center` (plane coordinates) is
    /// seen as: centre, semi-axes major first, and the frame's rotation, all in
    /// canvas px.
    ///
    /// The image of a conic under a homography is a conic — `Hᵀ C H` on the
    /// matrix — so this is exact and closed-form rather than a fit of sampled
    /// points, and it is the *same* operation in both directions. `None` when
    /// that conic is not a bounded curve: a circle crossing its plane's
    /// vanishing line images to a hyperbola, which is not something a stroke can
    /// be, and falls out of the classification instead of being a case.
    pub fn circle_seen(&self, center: Vec2, radius: f32) -> Option<Ellipse> {
        let circle = conic_of(Ellipse::new(center, Vec2::splat(radius), 0.0))?;
        ellipse_of(congruent(circle, self.plane_from_canvas))
    }

    /// The circle on this plane that is seen as the given ellipse — the exact
    /// inverse of [`circle_seen`](Self::circle_seen), by the same congruence the
    /// other way round.
    ///
    /// The pulled-back conic *is* a circle whenever the ellipse came from one, so
    /// the radius is read as the one of equal area; nothing here has to trust
    /// that, since a shape which is not a perspective circle never carries a
    /// plane in the first place (§20.7).
    pub fn circle_behind(&self, seen: Ellipse) -> Option<(Vec2, f32)> {
        let seen = conic_of(seen)?;
        let back = ellipse_of(congruent(seen, self.canvas_from_plane))?;
        Some((back.center, (back.radii.x * back.radii.y).sqrt()))
    }

    /// `p` in plane coordinates, with the signed homogeneous weight that says
    /// which side of the vanishing line it came from.
    fn charted(&self, p: Vec2) -> Option<(Vec2, f32)> {
        let q = self.plane_from_canvas * Vec3::new(p.x, p.y, 1.0);
        let far = Vec2::new(q.x, q.y);
        (q.z.abs() * PLANE_REACH > far.length()).then(|| (far / q.z, q.z))
    }
}

/// The conic `[[axx, axy, bx], [axy, ayy, by], [bx, by, c]]` of an ellipse. `None`
/// for radii that are not positive, which describe no curve.
fn conic_of(e: Ellipse) -> Option<Mat3> {
    let Ellipse {
        center,
        radii,
        angle,
    } = e;
    if !(radii.x > 0.0 && radii.y > 0.0 && center.is_finite()) {
        return None;
    }
    let u = Vec2::from_angle(angle);
    let v = u.perp();
    let (ix, iy) = (radii.x.powi(-2), radii.y.powi(-2));
    let (axx, axy, ayy) = (
        ix * u.x * u.x + iy * v.x * v.x,
        ix * u.x * u.y + iy * v.x * v.y,
        ix * u.y * u.y + iy * v.y * v.y,
    );
    let b = -Vec2::new(
        axx * center.x + axy * center.y,
        axy * center.x + ayy * center.y,
    );
    Some(Mat3::from_cols(
        Vec3::new(axx, axy, b.x),
        Vec3::new(axy, ayy, b.y),
        Vec3::new(b.x, b.y, -b.dot(center) - 1.0),
    ))
}

/// The ellipse a conic describes — centre, semi-axes major first, frame rotation
/// — or `None` if it is not one.
///
/// A conic and its negation are the same curve, so the sign is normalized first
/// and every test after it is a plain inequality: a positive-definite quadratic
/// part is exactly "an ellipse rather than a hyperbola", and a positive constant
/// at the centre is exactly "a real one rather than an imaginary one".
fn ellipse_of(c: Mat3) -> Option<Ellipse> {
    let flip = if c.x_axis.x + c.y_axis.y < 0.0 {
        -1.0
    } else {
        1.0
    };
    let (axx, axy, ayy) = (flip * c.x_axis.x, flip * c.x_axis.y, flip * c.y_axis.y);
    let b = flip * Vec2::new(c.x_axis.z, c.y_axis.z);
    // With the sign normalized so the trace is positive, a positive determinant
    // is exactly "positive definite", which is exactly "an ellipse rather than a
    // hyperbola" — and it carries `minor > 0` with it, so the semi-axes below
    // need no separate guard. Written to reject a NaN rather than admit one.
    let det = axx * ayy - axy * axy;
    if !(det.is_finite() && det > 0.0) {
        return None;
    }
    // The centre solves `A m = −b`; the constant there is what the semi-axes
    // divide into, and its sign is "a real ellipse rather than an imaginary one".
    let center = Vec2::new(axy * b.y - ayy * b.x, axy * b.x - axx * b.y) / det;
    let s = -(flip * c.z_axis.z + b.dot(center));
    if !(s.is_finite() && s > 0.0) {
        return None;
    }
    let (major, minor, dir) = principal_axis(axx, axy, ayy);
    // `dir` belongs to the *larger* eigenvalue, which is the *shorter* semi-axis:
    // the quadratic form is steepest across the ellipse's waist.
    let along = dir.perp();
    let radii = Vec2::new((s / minor).sqrt(), (s / major).sqrt());
    radii
        .is_finite()
        .then(|| Ellipse::new(center, radii, along.y.atan2(along.x)))
}

/// `Mᵀ C M` — a conic carried through the map `M` takes points by.
fn congruent(conic: Mat3, m: Mat3) -> Mat3 {
    m.transpose() * conic * m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::guide::{PerspectiveGuide, fixtures::guide};

    /// A point of plane `k`, projected the long way round: straight from the axis
    /// directions and the lens, with nothing from [`AxisPlane`] involved. What the
    /// chart is checked *against*, so that its matrix is under test and not merely
    /// consistent with itself.
    fn projected(g: &PerspectiveGuide, k: usize, q: Vec2) -> Vec2 {
        let d = g.axis_dirs();
        let (i, j) = (k, (k + 1) % 3);
        let x = d[i].cross(d[j]) + d[i] * q.x + d[j] * q.y;
        g.center + Vec2::new(x.x, x.y) * (g.focal / x.z)
    }

    /// The chart is the projection: what it says a canvas point is on the plane, the
    /// camera puts back where it came from.
    #[test]
    fn the_chart_is_the_projection_both_ways() {
        let g = guide(0.5, 0.35, 0.2);
        for k in 0..3 {
            let plane = g.planes()[k].expect("plane shown");
            for q in [Vec2::new(0.1, -0.08), Vec2::new(-0.3, 0.22)] {
                let p = projected(&g, k, q);
                let back = plane.to_plane(p).expect("on the plane");
                assert!(back.distance(q) < 1e-4, "plane {k}: {q} -> {p} -> {back}");
            }
        }
    }

    /// The theorem the whole snap rests on: the image of a circle on a plane is an
    /// ellipse, and it is the one [`AxisPlane::circle_seen`] answers with — checked by
    /// projecting the circle's own points and finding them on it.
    #[test]
    fn a_circle_on_a_plane_is_seen_as_the_ellipse_it_answers_with() {
        let g = guide(0.5, 0.35, 0.2);
        let (at, radius) = (Vec2::new(0.06, -0.05), 0.09);
        for k in 0..3 {
            let plane = g.planes()[k].expect("plane shown");
            let Ellipse {
                center,
                radii,
                angle,
            } = plane.circle_seen(at, radius).expect("a bounded image");
            let (u, v) = (Vec2::from_angle(angle), Vec2::from_angle(angle).perp());
            let worst = (0..48)
                .map(|i| {
                    let t = i as f32 / 48.0 * std::f32::consts::TAU;
                    let d = projected(&g, k, at + Vec2::new(t.cos(), t.sin()) * radius) - center;
                    // The ellipse's own equation: 1 exactly, on it.
                    ((d.dot(u) / radii.x).powi(2) + (d.dot(v) / radii.y).powi(2) - 1.0).abs()
                })
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-3, "plane {k}: off its ellipse by {worst}");
        }
    }

    /// A plane facing the camera square-on foreshortens nothing, so its circles are
    /// seen as circles — the case that says the construction is not merely *some*
    /// conic map. In 1-point that is the X/Y pair.
    #[test]
    fn a_plane_facing_the_camera_sees_circles_as_circles() {
        let g = guide(0.0, 0.0, 0.0);
        let plane = g.planes()[0].expect("plane shown");
        let Ellipse { center, radii, .. } = plane
            .circle_seen(Vec2::new(0.2, -0.1), 0.05)
            .expect("an image");
        assert!((radii.x / radii.y - 1.0).abs() < 1e-4, "radii {radii}");
        // ...and at the projection of its centre, which is *not* true in general.
        assert!(center.distance(projected(&g, 0, Vec2::new(0.2, -0.1))) < 1e-3);
    }

    /// The image of a circle is not centred on the image of its centre — the classical
    /// fact, and the reason a perspective circle cannot be steered by scaling the drawn
    /// ellipse about what one sees (§20.7).
    #[test]
    fn the_image_of_a_circle_is_not_centred_on_its_centre() {
        let g = guide(0.5, 0.35, 0.2);
        let (at, radius) = (Vec2::new(0.06, -0.05), 0.12);
        let plane = g.planes()[2].expect("plane shown");
        let Ellipse { center, radii, .. } = plane.circle_seen(at, radius).expect("an image");
        let drift = center.distance(projected(&g, 2, at));
        assert!(
            drift > 0.02 * radii.x,
            "the centres coincided to {drift}px on a {radii} ellipse"
        );
    }

    /// Seen and unseen are exact inverses, which is what lets an adjustment work in the
    /// plane and come back (§20.7).
    #[test]
    fn a_circle_survives_the_round_trip() {
        let g = guide(0.5, 0.35, 0.2);
        let plane = g.planes()[2].expect("plane shown");
        let (at, radius) = (Vec2::new(0.06, -0.05), 0.09);
        let seen = plane.circle_seen(at, radius).expect("an image");
        let (back, r) = plane.circle_behind(seen).expect("a circle");
        assert!(back.distance(at) < 1e-4 * radius, "centre {back}, was {at}");
        assert!(
            (r - radius).abs() < 1e-4 * radius,
            "radius {r}, was {radius}"
        );
    }

    /// A circle that runs past its plane's own horizon is seen as a hyperbola, not a
    /// loop — so there is nothing for a stroke to be, and the classification says so
    /// instead of answering with a shape.
    #[test]
    fn a_circle_across_the_horizon_has_no_ellipse() {
        let g = guide(0.5, 0.35, 0.2);
        let plane = g.planes()[2].expect("plane shown");
        let d = g.axis_dirs();
        // Where the plane crosses the eye's own depth: `x.z = 0`, in plane coordinates.
        let n = d[2].cross(d[0]);
        let gradient = Vec2::new(d[2].z, d[0].z);
        let reach = n.z.abs() / gradient.length();
        assert!(plane.circle_seen(Vec2::ZERO, reach * 0.5).is_some());
        assert!(plane.circle_seen(Vec2::ZERO, reach * 1.5).is_none());
    }
}
