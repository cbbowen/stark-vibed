//! The camera's **derivations** (§20.5): what a perspective guide is looked at
//! *through*, as opposed to what it is.
//!
//! Everything here is a pure function of [`PerspectiveGuide`]'s own fields — a ray
//! through a screen point, an axis's vanishing trace, the fan of lines a pencil
//! rules, the packed [`GuideScene`] the overlay pass reads. That is exactly §20.5's
//! argument for keeping guides whole on this side of the crate boundary: nothing
//! derived from a camera needs a pixel, so the derivations belong beside the fact
//! rather than in the engine.
//!
//! It is a *file* of its own for the ordinary reason. The fact and its derivations
//! are two things to read, and together they were the largest module in the crate by
//! half again. Nothing crosses a crate boundary and nothing became public that was
//! not; `document`'s re-export list is untouched.

use glam::{Mat3, Quat, Vec3};

use super::conic::AxisPlane;
use super::{
    FISHEYE_LINE_EPS, GuideScene, LATTICE_EPS, LINE_EPS, Lens, PerspectiveGuide, RAY_EPS,
    VP_FINITE_EPS,
};
use crate::geom::Vec2;

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

    /// The [`lattice`](Self::lattice)'s corner in **camera space**, still in
    /// cells (§20.3) — the guide's one world-metric datum, turned by the camera
    /// exactly as an axis is, which is why the drag never mentions it.
    ///
    /// `None` for a corner sitting *on* the eye (`LATTICE_EPS`): there a cell
    /// has no angular size, all three planes pass through the eye at once, and
    /// what the fans would draw is not an inaccurate grid but the whole canvas.
    /// No grid rather than a bad one.
    pub fn corner(&self) -> Option<Vec3> {
        let d = self.axis_dirs();
        let l = self.lattice;
        (l.is_finite() && l.length_squared() > LATTICE_EPS * LATTICE_EPS)
            .then(|| d[0] * l.x + d[1] * l.y + d[2] * l.z)
    }

    /// The eye's ray through canvas point `p`, unit, in camera space — through
    /// this guide's [`lens`](Self::lens). The shared first step of projection
    /// and of every canvas gesture: what the hand touches on the picture plane
    /// *is* a direction in the world, and because the orbit drag and the snap
    /// are stated in directions, they work under the fisheye without a line of
    /// code knowing it exists (§20.8).
    pub fn ray(&self, p: Vec2) -> Vec3 {
        let q = (p - self.center) / self.focal;
        match self.lens {
            Lens::Rectilinear => Vec3::new(q.x, q.y, 1.0).normalize(),
            // Inverse stereographic. `(2u, 1−s)` has length exactly `1+s`, so
            // the division lands on the unit sphere with no normalize — and
            // `s > 1` reaches past the 90° ring onto the *back* hemisphere,
            // which is the fisheye's whole point.
            Lens::Fisheye => {
                let u = q * 0.5;
                let s = u.length_squared();
                Vec3::new(2.0 * u.x, 2.0 * u.y, 1.0 - s) / (1.0 + s)
            }
        }
    }

    /// Where the **unit** direction `d` (camera space) images on the canvas —
    /// the forward half of [`ray`](Self::ray), `None` where the lens cannot
    /// see it. Rectilinear treats `d` projectively (an axis and its negation
    /// image together, or at infinity when it lies in the picture plane); the
    /// fisheye sees every direction except the exact backward pole, its
    /// projection point.
    pub fn project(&self, d: Vec3) -> Option<Vec2> {
        match self.lens {
            Lens::Rectilinear => (d.z.abs() > VP_FINITE_EPS)
                .then(|| self.center + Vec2::new(d.x, d.y) * (self.focal / d.z)),
            Lens::Fisheye => {
                let w = 1.0 + d.z;
                (w > VP_FINITE_EPS)
                    .then(|| self.center + Vec2::new(d.x, d.y) * (2.0 * self.focal / w))
            }
        }
    }

    /// Both **poles** of the axis direction `d`: where it vanishes going forward,
    /// and where its negation does.
    ///
    /// The second is `None` under the rectilinear lens, where it is not a second
    /// place at all — directions there are projective, so an axis and its
    /// negation image together (§20.1) — and only the fisheye separates them,
    /// which is the whole reason a 1-point pose reads as 5-point under it
    /// (§20.8). Written down once here because two derivations want it and the
    /// rule is easy to state twice slightly differently: the scene's marker
    /// slots, and the cut that makes a cursor ray a ray ([`axis_ray`](Self::axis_ray)).
    pub fn poles(&self, d: Vec3) -> [Option<Vec2>; 2] {
        [
            self.project(d),
            match self.lens {
                Lens::Rectilinear => None,
                Lens::Fisheye => self.project(-d),
            },
        ]
    }

    /// The dressed view-cone rings, canvas px: the 45° ring, and the 90° ring
    /// where the lens has one ([`Lens::ring_factors`]). What the overlay's
    /// lens-drag grabs — dragging either ring *is* setting the focal length.
    pub fn rings(&self) -> (f32, Option<f32>) {
        let (r45, r90) = self.lens.ring_factors();
        (r45 * self.focal, r90.map(|r| r * self.focal))
    }

    /// Where the plane through the eye with **unit** normal `m` images (§20.2,
    /// §20.8): the straight line of the rectilinear lens, or the circle the
    /// fisheye bows it into.
    ///
    /// Every curve the overlay draws that is not a marker is one of these, and
    /// there are two kinds of plane that produce one — a pair plane, whose image
    /// is the vanishing trace ([`pair_trace`](Self::pair_trace)), and the plane
    /// an axis spans with the ray under the pointer, whose image is a cursor ray
    /// ([`axis_ray`](Self::axis_ray)). They are the same construction and so they
    /// are one function, which is the whole of why §20.9 works under the fisheye
    /// without knowing it exists.
    ///
    /// **Unit**, and the caller owes that. Both results are *projective* in `m`
    /// — the line is normalized by `|m.xy|`, the circle's center is a ratio — so
    /// a scaled normal names the same curve, with the one exception that decides
    /// everything: the fisheye radius `2f|m|/|m.z|` reads the length. The two
    /// epsilons below are stated for a unit normal too, being cosines of the
    /// angle the plane makes with the picture plane.
    ///
    /// `None` when the trace is at infinity, which is the plane facing the
    /// camera square-on: there is no curve on the canvas, and so nothing to
    /// draw, to measure a station point against, or to grab.
    pub fn plane_trace(&self, m: Vec3) -> Option<PlaneTrace> {
        let (c, f) = (self.center, self.focal);
        let planar = Vec2::new(m.x, m.y);
        match self.lens {
            // The vanishing line of the plane with (unit) normal `m` is the
            // trace of the parallel plane through the eye: with the eye at
            // distance f over c, that is `m.x·x + m.y·y + (f·m.z − m·c) = 0`,
            // normalized so its first two coefficients are a unit normal and
            // evaluating it *is* signed canvas-px distance.
            Lens::Rectilinear => {
                let len = planar.length();
                (len >= LINE_EPS).then(|| PlaneTrace::Line {
                    normal: planar / len,
                    offset: (f * m.z - planar.dot(c)) / len,
                })
            }
            // The stereographic image of the great circle of directions in the
            // plane: an exact circle — conformality's gift (§20.8) — with
            // center `c + 2f·m.xy/m.z` and radius `2f/|m.z|`, from substituting
            // the inverse projection into `m·d = 0`. A plane containing the
            // view axis (m.z ≈ 0) images straight, through the center of view.
            Lens::Fisheye if m.z.abs() > FISHEYE_LINE_EPS => Some(PlaneTrace::Circle {
                center: c + planar * (2.0 * f / m.z),
                radius: 2.0 * f / m.z.abs(),
            }),
            Lens::Fisheye => planar.try_normalize().map(|n| PlaneTrace::Line {
                normal: n,
                offset: -n.dot(c),
            }),
        }
    }

    /// The **vanishing trace** of pair plane `k` — the plane axes `k` and
    /// `k + 1` span — exactly as the guide pass draws it (§20.2, §20.8).
    ///
    /// The image of the parallel plane *through the eye*, whose normal is the
    /// two axes' cross product — unit, the frame being orthonormal, which is
    /// what [`plane_trace`](Self::plane_trace) asks of a caller.
    pub fn pair_trace(&self, k: usize) -> Option<PlaneTrace> {
        let dirs = self.axis_dirs();
        self.plane_trace(dirs[k % 3].cross(dirs[(k + 1) % 3]))
    }

    /// The **cursor ray** for axis `i`: where the world line through the point
    /// under the pointer, parallel to that axis, images (§20.9).
    ///
    /// The classical draughtsman's line from the vanishing point through the
    /// hand — the direction the grid would have a stroke take *here*, shown
    /// before the stroke is made. Its curve is derived as the image of a plane
    /// rather than by joining two points, and that buys both of the cases a join
    /// cannot state:
    ///
    /// - an axis lying in the picture plane has no vanishing point to join to,
    ///   and its ray is the parallel line through the cursor — which falls out
    ///   here with no branch, exactly as §20.3's fans do;
    /// - under the **fisheye** the curve is the arc through *both* poles, because
    ///   the plane's image is a circle and nothing else about the derivation
    ///   changes (§20.8).
    ///
    /// The plane is the one the axis spans with the eye's ray through `at`, so
    /// its normal is their cross product — which needs normalizing, unlike a
    /// pair's, since an axis and a ray are not orthogonal. Both being unit, the
    /// length of that product is the sine of the angle between them, and
    /// [`RAY_EPS`] is stated in it: `None` where the hand has come to rest on
    /// the axis's own vanishing point, and for the approach to it, where the
    /// normal's *direction* is already noise.
    ///
    /// # Why it is a ray and not a line
    ///
    /// The trace is the whole *projective* line, and only half of it is a place
    /// the artist can draw. A world line's points behind the eye image too —
    /// at their opposite direction, so on the far side of the vanishing point —
    /// and that half is a reflection of the drawing, not part of it. Cutting
    /// there is therefore geometry rather than taste: the ray is exactly the
    /// image of the half of the world line **in front of the eye**, and the
    /// vanishing point is where the two halves meet because that is what the
    /// point *is*.
    ///
    /// [`CursorRay::cut`] carries the half-plane that says so, oriented to keep
    /// the cursor's side. It is `None` for a ray nothing bounds — an axis
    /// vanishing at infinity keeps its whole line in front of the eye, and its
    /// ray is honestly the whole parallel.
    pub fn axis_ray(&self, i: usize, at: Vec2) -> Option<CursorRay> {
        let axis = self.axis_dirs()[i % 3];
        let n = axis.cross(self.ray(at));
        if n.length() <= RAY_EPS {
            return None;
        }
        let trace = self.plane_trace(n.normalize())?;
        let [fwd, back] = self.poles(axis);
        // The cut is the line separating the trace's two halves, and which line
        // that is depends on how the trace closes up.
        let cut = match (trace, fwd, back) {
            // Bowed: the trace is a canvas circle, closed already, and the two
            // poles are two points on it — so the two arcs are the two sides of
            // the **chord** joining them. Exact for either arc, however the poles
            // fall, which is what recommends it over the obvious alternative of
            // comparing arc angles: a stereographic image does not preserve arc
            // length, so the arc the hand is on can be the long way round, and
            // the wrap that then has to be got right has no natural place to put
            // its seam.
            (PlaneTrace::Circle { .. }, Some(v), Some(w)) => cut_at(perp(w - v), v, at),
            // Straight: the trace closes through infinity, so its two halves meet
            // at the pole *and* out there, and the line separating them is the
            // **perpendicular** at the pole. Whichever pole is on the canvas — an
            // axis is unsigned (§20.1), so which of its two directions is the
            // forward one is not a fact about the drawing.
            //
            // Both, under the fisheye on a plane that contains the view axis, and
            // then one cut is one short: the honest figure is the segment between
            // the poles and this draws it running past the far one. That pose is
            // the *knife edge* where a fisheye trace straightens — the cursor
            // crosses it in well under a pixel of travel — so it is a flash on the
            // way past rather than a picture anybody reads.
            (PlaneTrace::Line { normal, .. }, fwd, back) => {
                fwd.or(back).and_then(|v| cut_at(perp(normal), v, at))
            }
            // A bowed trace with one pole is a circle of astronomical radius,
            // both of whose ends are off any canvas (§20.8's `FISHEYE_LINE_EPS`
            // band). Nothing to cut at, and nothing that would show if there were.
            (PlaneTrace::Circle { .. }, _, _) => None,
        };
        Some(CursorRay { trace, cut })
    }

    /// The **horizons**, indexed by the world axis each one turns the camera
    /// about (§20.5): entry `n` is the vanishing trace of the plane *normal* to
    /// axis `n` — the pair the other two axes span, drawn through their two
    /// vanishing points. Grabbing it is how a constrained turn is asked for, so
    /// this is the same list the overlay hit-tests a press against.
    ///
    /// That indexing is the whole point of the method: `pair_trace(k)` is
    /// stated in terms of the two axes spanning the plane, and the drag is
    /// stated in terms of the one axis it holds fixed. The two are related by
    /// the cross product — pair `(n+1, n+2)` has normal `n` — and it is written
    /// down once here rather than at each call site, where "the line between
    /// the X and Z vanishing points turns about Y" is a step it is easy to take
    /// off by one.
    ///
    /// `None` for a horizon that is not on the screen to be grabbed: a guide
    /// turned down to nothing, a plane switched off, or a trace at infinity —
    /// the same rule [`pencils`](Self::pencils) is gated by, over the same
    /// controls, because it exists for the same reason. A horizon
    /// belongs to a plane rather than to the axis it turns about, and follows
    /// that plane's flag: it is the plane's own infinity that is being drawn.
    ///
    /// Unlike a pencil, a horizon is offered under **both** lenses. What the
    /// hand grabs here is the curve itself and what it asks for is a turn about
    /// an axis, and a turn is a statement in direction space that the lens
    /// never enters (§20.8) — where a pencil would have had to promise a
    /// straight line the fisheye does not draw.
    pub fn horizons(&self) -> [Option<PlaneTrace>; 3] {
        let shown = self.opacity > 0.0;
        std::array::from_fn(|n| {
            // Pair `(n+1, n+2)` is the plane axis `n` is normal to.
            let k = (n + 1) % 3;
            (shown && self.pairs[k])
                .then(|| self.pair_trace(k))
                .flatten()
        })
    }

    /// The pencils a stroke may align to (§20.6): one per world axis, and
    /// `None` for an axis that is not on the screen.
    ///
    /// Gated on what is *shown* rather than on what exists, because a snap the
    /// artist cannot see coming reads as the tool bending a considered line. An
    /// overlay turned down to nothing and an axis with no plane left to rule
    /// both offer nothing — the same rule stated once, over the controls the
    /// panel puts on the bar.
    ///
    /// That last one is [`is_drawn`](Self::is_drawn), and it is the same rule
    /// rather than a second one. A guide line is a line *in a pair plane*
    /// (§20.3), so an axis draws only on the two planes it is a side of; switch
    /// both of those off and nothing of the axis appears, so nothing of it may
    /// bend a stroke — even though its vanishing point is as computable as ever.
    ///
    /// The guide's own **eye** is the third way to be unshown, and it is
    /// deliberately not asked here: it is per-client, so it is not a fact this
    /// type carries (§20.5). It is applied one level up instead, by
    /// [`Scaffold::of`](super::Scaffold::of) being handed only the guides this client draws — one
    /// filter at one place rather than a term repeated in three derivations.
    ///
    /// A **fisheye** guide offers nothing either (§20.8): its guide lines are
    /// circles, and the pencil this returns describes straight lines — a snap
    /// through it would align a stroke to a line the guide does not draw,
    /// which is exactly the bent-considered-line surprise the visibility gate
    /// exists to prevent. Snapping strokes to the fisheye's arcs is its own
    /// future piece of work.
    pub fn pencils(&self) -> [Option<AxisPencil>; 3] {
        let shown = self.opacity > 0.0 && self.lens == Lens::Rectilinear;
        let dirs = self.axis_dirs();
        std::array::from_fn(|i| {
            (shown && self.is_drawn(i)).then_some(AxisPencil {
                center: self.center,
                focal: self.focal,
                dir: dirs[i],
            })
        })
    }

    /// Whether axis `i` appears on the canvas at all: whether either of the two
    /// pair planes it is a side of is drawn (§20.3).
    ///
    /// An axis is not a thing that is shown or hidden — a plane is. What an
    /// axis *has* is lines, and every one of them lies in one of its two
    /// planes, so switching both off leaves the axis with nothing on the screen
    /// however well defined its direction still is. That is the question its
    /// vanishing point and its stroke pencil both turn on, and it is written
    /// down once here rather than as the same disjunction in three places.
    pub fn is_drawn(&self, i: usize) -> bool {
        self.pairs[i % 3] || self.pairs[(i + 2) % 3]
    }

    /// The planes a circle can be drawn on (§20.7): pair `k` spans axes
    /// `(k, k+1)`, one chart each.
    ///
    /// Gated like [`pencils`](Self::pencils) — including the fisheye
    /// exclusion, since the chart is a homography of the *flat* picture plane
    /// — but on the plane's own flag, which is the whole of the question here:
    /// a circle is drawn *on a plane*, so the plane the artist switched off is
    /// exactly the plane no loop may be read as a circle on. The chart's third
    /// column is `a_i × a_j`, which for a right-handed frame is the remaining
    /// axis, so the three planes are this guide's one axis matrix with its
    /// columns cyclically shifted.
    pub fn planes(&self) -> [Option<AxisPlane>; 3] {
        let shown = self.opacity > 0.0 && self.focal > 0.0 && self.lens == Lens::Rectilinear;
        let dirs = self.axis_dirs();
        let lens = Mat3::from_cols(
            Vec3::new(self.focal, 0.0, 0.0),
            Vec3::new(0.0, self.focal, 0.0),
            Vec3::new(self.center.x, self.center.y, 1.0),
        );
        std::array::from_fn(|k| {
            let (i, j) = (k, (k + 1) % 3);
            (shown && self.pairs[k]).then(|| {
                let canvas_from_plane = lens * Mat3::from_cols(dirs[i], dirs[j], dirs[(k + 2) % 3]);
                AxisPlane {
                    plane_from_canvas: canvas_from_plane.inverse(),
                    canvas_from_plane,
                }
            })
        })
    }

    /// Everything the guide pass draws, derived from the camera (§20.2). Cheap
    /// — a rotation and a handful of products — so it is recomputed per render
    /// rather than cached beside the state it would shadow.
    ///
    /// `cursor` is where this client's pointer is on the canvas, and it is an
    /// **argument** rather than a field for the reason the guide's eye is not one
    /// either (§20.5): a camera is document state and a pointer is not, so the
    /// one thing on the overlay that follows the hand is handed in by the side
    /// holding both — which is `Session`, exactly as for the eye. `None` — off
    /// the canvas, or a render that is not a screen — draws no rays (§20.9).
    pub fn scene(&self, cursor: Option<Vec2>) -> GuideScene {
        let dirs = self.axis_dirs();
        let c = self.center;
        let f = self.focal;

        // Where each axis's direction images, both ways ([`poles`]).
        //
        // [`poles`]: Self::poles
        let poles = dirs.map(|d| self.poles(d));
        let vps = poles.map(|p| p[0]);
        let anti_vps = poles.map(|p| p[1]);

        // Pair k spans axes (k, k+1): its vanishing trace ([`pair_trace`], the
        // same curve the drag grabs to turn about the remaining axis) and its
        // station point.
        let mut lines = [None; 3];
        let mut stations = [None; 3];
        for k in 0..3 {
            lines[k] = self.pair_trace(k);
            // The station point: the eye, rotated into the picture plane about
            // the vanishing line (§20.2). The eye sits at height f over c, at
            // distance √(a² + f²) from the line (a = the line's distance from
            // c); rotating preserves that distance, landing on the ray from the
            // foot of c's perpendicular through c. With the view axis in the
            // pair plane (a ≈ 0 — exact 2-point) either side is the same
            // rotation; the canvas-down side is the drawing-board convention.
            //
            // Rectilinear only, and never from the fisheye's straight traces:
            // rotating the eye into the picture plane is a *flat-plane*
            // measuring construction, and under a curved lens the distances it
            // would transfer do not exist on the canvas to be measured. A pair
            // with no line at all — the plane facing the camera square-on — has
            // no station point either, and falls out of the same `let`.
            if self.lens == Lens::Rectilinear
                && let Some(PlaneTrace::Line { normal: n, offset }) = lines[k]
            {
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
        }

        GuideScene {
            center: c,
            focal: f,
            lens: self.lens,
            rings: self.rings(),
            lattice: self.corner(),
            opacity: self.opacity.clamp(0.0, 1.0),
            dirs,
            // Both derived here rather than in the shader, which is the pass's
            // standing bargain (§20.4): the CPU hands it numbers, not rules.
            // An axis is drawn when either plane it is a side of is
            // ([`is_drawn`](Self::is_drawn)).
            axis_alpha: std::array::from_fn(|i| if self.is_drawn(i) { 1.0 } else { 0.0 }),
            pair_alpha: self.pairs.map(|on| if on { 1.0 } else { 0.0 }),
            lines,
            vps,
            anti_vps,
            stations,
            // The rays through the hand (§20.9). Computed for every axis and
            // left to the pass's own `axis_alpha` gate, like the vanishing
            // points a ray runs *from* — the two are the same marker twice, one
            // a point and one a curve, and they must appear and go together.
            rays: match cursor {
                Some(at) => std::array::from_fn(|i| self.axis_ray(i, at)),
                None => [None; 3],
            },
        }
    }

    /// The orbit drag (§20.5): the world direction grabbed at `from` follows
    /// the pointer to `to`, by rotating the whole frame — always computed from
    /// the drag's *start* state, so a long drag cannot drift.
    ///
    /// `locked` axes are held fixed. Rotations fixing one axis are exactly the
    /// turns about it, so one lock constrains the drag to that axis's orbit —
    /// lock the vertical and a 2-point setup stays 2-point under any drag —
    /// and two locks pin the frame entirely (the identity is the only rotation
    /// fixing two axes). Unlocked, the drag is the free arc, and it carries the
    /// grabbed direction to the pointer *exactly*.
    ///
    /// There is no snap. A free drag that happens to pass near an axis turn
    /// stays free: the constrained turn is something the hand asks for by
    /// grabbing the axis's [`horizon`](Self::horizons) — which is a lock held
    /// for the drag's duration and arrives here as one — rather than something
    /// the drag falls into partway through. Deciding it from the geometry meant
    /// the same gesture could be free at the press and constrained a moment
    /// later, and a rotation that changes what it is mid-drag reads as the tool
    /// grabbing the guide out of the hand.
    #[must_use]
    pub fn dragged(&self, from: Vec2, to: Vec2, locked: [bool; 3]) -> Self {
        if locked.iter().filter(|l| **l).count() >= 2 {
            return *self;
        }
        let (r0, r1) = (self.ray(from), self.ray(to));
        let delta = match locked.iter().position(|l| *l) {
            Some(i) => axis_turn(self.axis_dirs()[i], r0, r1),
            None if r0.cross(r1).length_squared() < 1e-12 => Quat::IDENTITY,
            None => Quat::from_rotation_arc(r0, r1),
        };
        Self {
            rotation: (delta * self.rotation).normalize(),
            ..*self
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

/// One axis of one guide, as the thing a stroke can be aligned to: the
/// **pencil** of images of every world line along that axis (§20.6).
///
/// This is the whole of a perspective guide that the drawing assist sees, and
/// it is deliberately not a direction — a pencil converges, so what it can
/// answer is a direction *at a point*. That is also what makes the snap an
/// alignment rather than a move: the line the assist takes is the pencil's
/// line through the point the hand started from, so a stroke is turned onto
/// the grid without being slid along it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AxisPencil {
    center: Vec2,
    focal: f32,
    dir: Vec3,
}

impl AxisPencil {
    /// The pencil's line through canvas point `p`, as a unit direction —
    /// unsigned, like every direction here, since an axis and its negation
    /// name the same pencil.
    ///
    /// It is `V(a) − p` cleared of its denominator: multiplying through by the
    /// `d.z` that the vanishing point divides by leaves
    /// `f·(d.x, d.y) + d.z·(c − p)`, which stays finite as `d.z → 0` and
    /// becomes the parallel direction of an axis lying in the picture plane.
    /// So no vanishing point is ever computed on this path and there is no
    /// case to branch on — the same reason §20.3's fans work in direction
    /// space. `None` only *at* a vanishing point, where the pencil determines
    /// no line.
    pub fn through(&self, p: Vec2) -> Option<Vec2> {
        (Vec2::new(self.dir.x, self.dir.y) * self.focal + (self.center - p) * self.dir.z)
            .try_normalize()
    }
}

/// The image of the directions in one plane through the eye: the straight line
/// of the rectilinear lens, or the circle the fisheye bows it into (§20.8). One
/// curve either way, so the shader carries a kind beside four numbers rather
/// than two pipelines.
///
/// Named for the plane rather than for either of the two things the overlay
/// draws with it — a pair plane's **vanishing trace** (§20.2) and an axis's
/// **cursor ray** (§20.9) — because it is one construction serving both, and a
/// name taken from the first of them would have made the second read as a reuse
/// of somebody else's type ([`PerspectiveGuide::plane_trace`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PlaneTrace {
    /// `normal · p + offset = 0`, with `normal` unit — evaluating is signed
    /// canvas-px distance.
    Line { normal: Vec2, offset: f32 },
    /// `|p − center| = radius`, canvas px.
    Circle { center: Vec2, radius: f32 },
}

impl PlaneTrace {
    /// How far canvas point `p` lies from the trace, canvas px — the one
    /// number both kinds answer, which is why the shader draws them with one
    /// `stroke_cov` and the overlay hit-tests them with one comparison
    /// (§20.4, §20.5).
    pub fn distance(self, p: Vec2) -> f32 {
        match self {
            PlaneTrace::Line { normal, offset } => (normal.dot(p) + offset).abs(),
            PlaneTrace::Circle { center, radius } => (p.distance(center) - radius).abs(),
        }
    }
}

/// Half of the canvas: the points where `normal · p + offset ≥ 0`, `normal`
/// unit — so [`signed`](Self::signed) is distance in canvas px, as evaluating a
/// [`PlaneTrace::Line`] is.
///
/// One of these is what turns a cursor ray's whole trace into a *ray*
/// ([`CursorRay::cut`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Halfplane {
    pub normal: Vec2,
    pub offset: f32,
}

impl Halfplane {
    /// Signed distance from the boundary, canvas px — positive on the half that
    /// is kept.
    pub fn signed(self, p: Vec2) -> f32 {
        self.normal.dot(p) + self.offset
    }
}

/// One axis's **cursor ray** (§20.9): the line from that axis's vanishing point
/// through the point under the pointer, which is what the grid would have a
/// stroke there do.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CursorRay {
    /// The whole curve the axis's world lines through that point image into —
    /// straight, or bowed by the fisheye
    /// ([`plane_trace`](PerspectiveGuide::plane_trace)).
    pub trace: PlaneTrace,
    /// Which half of it is drawn: the half in front of the eye, which is the
    /// cursor's own. `None` for a ray with no vanishing point to run from, whose
    /// world line is in front of the eye along its whole length
    /// ([`axis_ray`](PerspectiveGuide::axis_ray)).
    pub cut: Option<Halfplane>,
}

/// The canvas normal of a line **along** `v` — a quarter turn, and the one place
/// this file needs the operation by name.
fn perp(v: Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}

/// The half-plane whose boundary runs through `through` with normal along `n`,
/// oriented to **keep** `keep`. `None` where `n` names no direction, or where
/// `keep` lies exactly on the boundary and so picks no side.
fn cut_at(n: Vec2, through: Vec2, keep: Vec2) -> Option<Halfplane> {
    let normal = n.try_normalize()?;
    let cut = Halfplane {
        normal,
        offset: -normal.dot(through),
    };
    let s = cut.signed(keep);
    // The cursor is on the boundary only when it is at the vanishing point
    // itself, which `RAY_EPS` refused before this was reached — so this is the
    // representation declining to express a side rather than a case to handle.
    if s == 0.0 {
        return None;
    }
    Some(if s > 0.0 {
        cut
    } else {
        Halfplane {
            normal: -cut.normal,
            offset: -cut.offset,
        }
    })
}
