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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::guide::fixtures::{fisheye, guide};

    /// 1-point: the view axis meets the picture plane at the center of view,
    /// so that is where the Z axis vanishes — and the transverse axes, lying
    /// in the picture plane, vanish at infinity.
    #[test]
    fn one_point_vanishes_at_the_center_of_view() {
        let g = guide(0.0, 0.0, 0.0);
        let s = g.scene(None);
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
        let s = g.scene(None);
        let vx = s.vps[0].expect("X finite");
        let vz = s.vps[2].expect("Z finite");
        assert!(s.vps[1].is_none(), "verticals stay parallel");
        assert!((vx.y - g.center.y).abs() < 1e-3);
        assert!((vz.y - g.center.y).abs() < 1e-3);
        // Pair 2 spans (Z, X): the ground. Its line passes through c…
        let Some(PlaneTrace::Line { normal, offset }) = s.lines[2] else {
            panic!("horizon should be a straight line");
        };
        assert!((normal.dot(g.center) + offset).abs() < 1e-3);
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
        let s = g.scene(None);
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
            let s = guide(yaw, pitch, roll).scene(None);
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

    // --- the lattice the fans measure out (§20.3) --------------------------

    /// The two continuous **cell indices** canvas point `p` carries for axis
    /// `i` — the expression `guides.wesl` evaluates at every texel (§20.3),
    /// restated here so the lattice's claim can be held to the lines that will
    /// actually be drawn rather than to a description of them.
    ///
    /// A guide line is where one of the two is an integer. `.x` counts cells
    /// across the pair plane axis `i` spans with its **successor**, `.y` the
    /// one it spans with its predecessor; an infinity is that plane's vanishing
    /// line, where the whole plane images onto one line and there is no cell to
    /// be in.
    fn cell(g: &PerspectiveGuide, i: usize, p: Vec2) -> Vec2 {
        let d = g.axis_dirs();
        let corner = g.corner().expect("a lattice");
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let r = g.ray(p);
        let (u, v) = (r.dot(d[j]), -r.dot(d[k]));
        // Each family's plane contributes only its own distance from the eye,
        // since the phase is the eye rather than the corner (§20.3).
        let (pj, pk) = (corner.dot(d[j]), corner.dot(d[k]));
        Vec2::new(-pk * u / v, -pj * v / u)
    }

    /// The point of the plane normal to axis `n` that is nearest the eye — the
    /// foot of the eye's own perpendicular, `lattice[n]` cells away. Where the
    /// grid's phase is anchored, so both families crossing there read cell zero
    /// and everything else is counted from it in whole cells.
    fn foot(g: &PerspectiveGuide, n: usize) -> Vec3 {
        let d = g.axis_dirs();
        let corner = g.corner().expect("a lattice");
        d[n] * corner.dot(d[n])
    }

    /// The property that makes a grid something to measure on rather than
    /// something to look at (§20.3): the cells the fans cut on a pair plane are
    /// **squares** — one lattice cell each, the same size in both directions,
    /// the same size everywhere on the plane, and the same size on all three
    /// planes because there is one cube behind them.
    ///
    /// Checked where the two halves meet. Cell corner `(a, b)` of the plane —
    /// counted in whole cells from the eye's own foot on it — is projected; the
    /// *drawn* fans are asked which cell that texel falls in and have to answer
    /// `(a, b)` exactly; and then the chart §20.7 already tests independently is
    /// asked how far apart those corners lie **on the plane itself**, where a
    /// square is a statement with no perspective left in it.
    #[test]
    fn the_fans_cut_squares_on_every_pair_plane() {
        let g = guide(0.5, 0.35, 0.2);
        let d = g.axis_dirs();
        let corner = g.corner().expect("a lattice");
        for k in 0..3 {
            let (i, j) = (k, (k + 1) % 3);
            let plane = g.planes()[k].expect("plane shown");
            // One cell on this plane, in the chart's units: the chart is taken
            // at unit distance along the plane's normal and the plane is
            // `corner · m` cells away, so the two are reciprocal (§20.7).
            let side = 1.0 / corner.dot(d[(k + 2) % 3]).abs();
            let origin = foot(&g, (k + 2) % 3);
            let corner_at = |a: f32, b: f32| {
                // Projectively: a plane's nearest point can lie behind the eye,
                // and both the fans and the chart read the pencil rather than the
                // ray's sign, so the image of `-x` answers for `x`.
                let x = origin + d[i] * b + d[j] * a;
                let p = g.project(x.normalize()).expect("the cell corner images");
                // Axis `i` steps across this plane in `a` (its successor's
                // direction), axis `j` in `b` (its predecessor's) — the two
                // families the plane is served by, and no others.
                assert!(
                    (cell(&g, i, p).x - a).abs() < 1e-3,
                    "plane {k}: axis {i}'s fan puts ({a}, {b}) in cell {}",
                    cell(&g, i, p).x
                );
                assert!(
                    (cell(&g, j, p).y - b).abs() < 1e-3,
                    "plane {k}: axis {j}'s fan puts ({a}, {b}) in cell {}",
                    cell(&g, j, p).y
                );
                plane.to_plane(p).expect("on the plane")
            };
            for (a, b) in [(-2.0, 1.0), (0.0, 0.0), (2.0, -1.0)] {
                let o = corner_at(a, b);
                let (ea, eb) = (corner_at(a + 1.0, b) - o, corner_at(a, b + 1.0) - o);
                assert!(
                    (ea.length() - side).abs() < 1e-3 * side
                        && (eb.length() - side).abs() < 1e-3 * side,
                    "plane {k} at ({a}, {b}): sides {} and {}, not {side}",
                    ea.length(),
                    eb.length()
                );
                assert!(
                    ea.dot(eb).abs() < 1e-3 * side * side,
                    "plane {k} at ({a}, {b}): {ea:?} and {eb:?} are not square"
                );
            }
        }
    }

    /// Where two planes meet, their grids meet: along the edge a pair shares,
    /// both planes put their cell corners on the same points, one cell apart.
    ///
    /// Two planes crossing at a shared edge is the only thing that stops the
    /// three grids from being three grids: it is why a count carried along an
    /// edge means the same on either side of it, and why a box drawn in one
    /// plane lands on the grid of the next. It holds for any lattice at all,
    /// because both planes count the same whole cells from the same eye — the
    /// edge is measured from the point of it nearest the viewer.
    #[test]
    fn the_planes_agree_along_the_edges_they_share() {
        let g = guide(0.5, 0.35, 0.2);
        let d = g.axis_dirs();
        let corner = g.corner().expect("a lattice");
        for (i, along) in d.iter().enumerate() {
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            // The edge is where the two planes meet; its nearest point to the
            // eye is the corner with this axis's own offset taken out.
            let near = corner - *along * corner.dot(*along);
            for n in [-2.0, 0.0, 1.0, 3.0] {
                // A point `n` cells along the edge that the planes spanned with
                // the successor and with the predecessor both contain.
                let x = near + *along * n;
                assert!(x.z > 0.0, "the sample must be in front of the eye");
                let p = g.center + Vec2::new(x.x, x.y) * (g.focal / x.z);
                // The cross-family of each plane — the one whose lines run
                // across the edge — has to call this the *same* cell corner.
                assert!(
                    (cell(&g, j, p).y - n).abs() < 1e-3 && (cell(&g, k, p).x - n).abs() < 1e-3,
                    "edge {i} at {n}: the two planes cut it at {} and {}",
                    cell(&g, j, p).y,
                    cell(&g, k, p).x
                );
            }
        }
    }

    /// **The viewer stands on the grid** (§20.3): the foot of the eye's own
    /// perpendicular to a plane is one of that plane's cell corners, so looking
    /// straight down an axis puts the crossing beneath the center of view on the
    /// grid's own. That is what anchoring the phase at the eye *means*, and it
    /// is where the two families lying in each plane both read cell zero.
    ///
    /// Asserted with a deliberately **fractional** lattice, because that is the
    /// case the anchoring exists to answer. With the phase at the corner this
    /// held only when the corner's components happened to be whole — and a scale
    /// control that halves turns `(-4, 3, 6)` into `(-1, ¾, 1½)`, which put the
    /// crossing a quarter of a cell off on one axis and a half on the next.
    #[test]
    fn the_viewer_stands_on_the_grid_of_every_plane() {
        let g = PerspectiveGuide {
            lattice: Vec3::new(-3.7, 2.15, 6.4),
            ..guide(0.5, 0.35, 0.2)
        };
        for n in 0..3 {
            // Plane `n` is the one normal to axis `n`, spanned by the other two;
            // axis `n+1`'s first family and axis `n+2`'s second are the pair that
            // lie in it.
            let (i, j) = ((n + 1) % 3, (n + 2) % 3);
            let x = foot(&g, n);
            let p = g.project(x.normalize()).expect("the foot images");
            let (across, along) = (cell(&g, i, p).x, cell(&g, j, p).y);
            assert!(
                across.abs() < 1e-3 && along.abs() < 1e-3,
                "plane {n}: the foot of the eye's perpendicular sits at cell \
                 ({across}, {along}), not the origin"
            );
        }
    }

    /// The classical chequerboard, where the answer can be written down. Look
    /// straight down Z with the floor `h` cells below the eye, and its transverse
    /// guide lines fall at canvas heights `f·h/n` — the tile `n` cells out is
    /// seen at the depth `n`, the harmonic run of a receding chequered floor and
    /// what "squares in perspective" means when there is only one vanishing point
    /// to say it with. An equal-angle fan would draw `f·cot(n·θ)` there, and the
    /// two agree at exactly one line.
    ///
    /// The depths are counted from the **eye**, not from where the walls happen
    /// to stand, which is why `h` is the only part of the lattice in the answer:
    /// a fractional floor height moves every line together and still lands them
    /// on whole cells out from the viewer.
    #[test]
    fn a_one_point_floor_recedes_harmonically() {
        let h = 4.25;
        let g = PerspectiveGuide {
            lattice: Vec3::new(-3.0, h, 6.0),
            ..guide(0.0, 0.0, 0.0)
        };
        for n in [1.0, 2.0, 3.0, 7.0] {
            // The floor is the X/Z plane, so its transverse lines belong to
            // axis X and step along Z — X's *second* family (`.y`).
            let at = g.center + Vec2::new(0.0, g.focal * h / n);
            assert!(
                (cell(&g, 0, at).y - n).abs() < 1e-3,
                "the tile {n} cells out is drawn at cell {}",
                cell(&g, 0, at).y
            );
        }
    }

    /// Halving the cell **subdivides** the grid rather than sliding it: every
    /// line of the coarser grid is still a line of the finer one, with one new
    /// line between each pair, so nothing an artist has already counted against
    /// moves (§20.3). It is why the bar's scale steps in halvings and offers
    /// nothing between them.
    ///
    /// Stated over arbitrary texels rather than over the lines themselves,
    /// because the strong form is what makes it true of *every* line at once:
    /// doubling the lattice doubles the cell index everywhere, so an integer
    /// index goes to an even one and no texel's place in the grid is renamed by
    /// anything but a factor of two.
    #[test]
    fn doubling_the_cells_keeps_every_line_it_had() {
        let coarse = guide(0.5, 0.35, 0.2);
        let fine = PerspectiveGuide {
            lattice: coarse.lattice * 2.0,
            ..coarse
        };
        for p in [
            coarse.center + Vec2::new(-380.0, 210.0),
            coarse.center + Vec2::new(140.0, -95.0),
            coarse.center + Vec2::new(620.0, 480.0),
        ] {
            for i in 0..3 {
                let (was, now) = (cell(&coarse, i, p), cell(&fine, i, p));
                assert!(
                    (now - was * 2.0).abs().max_element() < 1e-3 * was.abs().max_element().max(1.0),
                    "axis {i} at {p:?}: cell {was:?} became {now:?}, not twice it"
                );
            }
        }
    }

    /// A lattice collapsed onto the eye names no grid, and the scene says so
    /// rather than handing the shader a corner every family would fall onto.
    #[test]
    fn a_lattice_at_the_eye_is_no_lattice() {
        let mut g = guide(0.5, 0.35, 0.2);
        assert!(g.scene(None).lattice.is_some());
        g.lattice = Vec3::ZERO;
        assert!(g.corner().is_none());
        assert!(g.scene(None).lattice.is_none());
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

    // --- the pencils strokes align to (§20.6) ------------------------------

    /// A pencil's line through a point aims at that axis's vanishing point.
    /// That is the §20.1 identity again, checked through the expression the
    /// assist actually uses — which reaches it without computing a VP at all.
    #[test]
    fn a_pencil_aims_at_its_vanishing_point() {
        let g = guide(0.5, 0.35, 0.2);
        let s = g.scene(None);
        for (i, pencil) in g.pencils().into_iter().enumerate() {
            let pencil = pencil.expect("every axis shown");
            let vp = s.vps[i].expect("3-point: all finite");
            for p in [Vec2::new(-300.0, 200.0), Vec2::new(640.0, -80.0)] {
                let u = pencil.through(p).expect("not at the vanishing point");
                let to_vp = (vp - p).normalize();
                assert!(
                    u.perp_dot(to_vp).abs() < 1e-3,
                    "axis {i} at {p:?}: {u:?} vs {to_vp:?}"
                );
            }
        }
    }

    /// An axis lying in the picture plane has no vanishing point to aim at, and
    /// its pencil is the parallel one — from the same expression, with nothing
    /// special-cased and nothing dividing by a vanishing `d.z`.
    #[test]
    fn a_pencil_at_infinity_is_parallel() {
        // 1-point: X and Y lie in the picture plane, so both vanish at infinity.
        let g = guide(0.0, 0.0, 0.0);
        assert!(g.scene(None).vps[0].is_none());
        let x = g.pencils()[0].expect("X shown");
        let far = Vec2::new(-9000.0, 4000.0);
        let (u, v) = (
            x.through(g.center).expect("a line at the centre"),
            x.through(far).expect("a line far away"),
        );
        assert!(
            u.perp_dot(v).abs() < 1e-4,
            "{u:?} and {v:?} are not parallel"
        );
        assert!(u.y.abs() < 1e-4, "X's lines should be level: {u:?}");
    }

    /// What is not on the screen offers nothing to snap to (§20.6) — one rule
    /// over the guide's eye, its planes and its opacity.
    ///
    /// An axis survives while *either* of its two planes does, because its
    /// lines are still being ruled on that one: switching a plane off is not a
    /// statement about the two axes bordering it. Only when both of an axis's
    /// planes are gone does it have nothing on the canvas, and only then does
    /// it stop offering a pencil.
    #[test]
    fn an_axis_with_no_plane_left_offers_no_pencil() {
        let mut g = guide(0.5, 0.35, 0.2);
        // Only the Y/Z plane. Y and Z still rule it; X borders neither it nor
        // anything else that is left.
        g.pairs = [false, true, false];
        assert!(g.pencils()[0].is_none(), "X has no plane left");
        assert!(g.pencils()[1].is_some() && g.pencils()[2].is_some());

        // One plane off leaves every axis with one, so every pencil stands.
        g.pairs = [true, true, false];
        assert!(g.pencils().iter().all(Option::is_some), "still all ruled");

        g.pairs = [false; 3];
        assert!(g.pencils().iter().all(Option::is_none), "nothing drawn");

        g.pairs = [true; 3];
        g.opacity = 0.0;
        assert!(g.pencils().iter().all(Option::is_none), "an invisible one");
    }

    /// The eight states the bar can reach, and the axes each leaves standing —
    /// the reason the control names planes and not axes (§20.3).
    ///
    /// Written out exhaustively because the claim *is* the completeness: three
    /// axis toggles could only ever produce none, one plane, or all three (a
    /// plane needed both its axes, so the second axis always brought a second
    /// plane free), and "the ground and one wall" — three of these rows — was
    /// not sayable at all. The axis column is the derived half, and it is here
    /// so that the derivation is checked against every input rather than the
    /// two or three anyone would think to try.
    #[test]
    fn every_combination_of_planes_is_reachable() {
        let mut g = guide(0.5, 0.35, 0.2);
        for bits in 0..8u8 {
            let pairs = std::array::from_fn(|k| bits >> k & 1 == 1);
            g.pairs = pairs;
            for i in 0..3 {
                // Axis `i` borders pairs `i` and `i+2`.
                let expected = pairs[i] || pairs[(i + 2) % 3];
                assert_eq!(
                    g.is_drawn(i),
                    expected,
                    "planes {pairs:?}: axis {i} should{} be drawn",
                    if expected { "" } else { " not" }
                );
                assert_eq!(g.pencils()[i].is_some(), expected, "planes {pairs:?}");
                assert_eq!(
                    g.scene(None).axis_alpha[i] > 0.0,
                    expected,
                    "planes {pairs:?}"
                );
                // A plane answers for itself, and for nothing else.
                assert_eq!(g.planes()[i].is_some(), pairs[i], "planes {pairs:?}");
                assert_eq!(
                    g.scene(None).pair_alpha[i] > 0.0,
                    pairs[i],
                    "planes {pairs:?}"
                );
            }
        }
    }

    /// A plane's chart follows that plane's own flag and nothing else — and the
    /// guide's eye and opacity govern it exactly as they govern a pencil.
    #[test]
    fn a_plane_follows_its_own_flag() {
        let mut g = guide(0.5, 0.35, 0.2);
        g.pairs = [false, true, false];
        assert!(g.planes()[1].is_some(), "the plane that is on");
        assert!(g.planes()[0].is_none() && g.planes()[2].is_none());

        g.pairs = [true; 3];
        g.opacity = 0.0;
        assert!(g.planes().iter().all(Option::is_none));
    }

    // --- the orbit drag (§20.5) --------------------------------------------

    /// The drag's contract: the world direction under the pointer at the start
    /// is under it at the end — for *any* drag, since the free arc is what an
    /// unconstrained grab always is now (§20.5).
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
        assert!(g2.scene(None).vps[1].is_none());
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

    /// A free drag does **not** snap (§20.5). The case most likely to be swept into
    /// one: a horizontal drag through the center of view implies a rotation
    /// within a few degrees of the near-vertical Y axis, and it is still the
    /// free arc — the grabbed direction lands exactly under the pointer, and Y
    /// moves, because nothing asked for it to be held.
    ///
    /// Asserted as a pair, since "did not snap" on its own would also be
    /// satisfied by a drag that did nothing at all: the exact carry is what
    /// says the free arc ran, and Y's motion is what says the constrained turn
    /// did not.
    #[test]
    fn a_drag_near_an_axis_turn_stays_free() {
        let g = guide(0.6, 0.0, 0.0);
        let (from, to) = (
            g.center + Vec2::new(-100.0, 0.0),
            g.center + Vec2::new(140.0, 12.0),
        );
        // The rotation this implies really is a near-Y one, so the drag sits exactly
        // where an axis cone would fire.
        let w = g.ray(from).cross(g.ray(to)).normalize();
        assert!(w.dot(g.axis_dirs()[1]).abs() > 0.99, "not a near-Y turn");

        let g2 = g.dragged(from, to, [false; 3]);
        let world = g.rotation.inverse() * g.ray(from);
        assert!(
            ((g2.rotation * world).normalize() - g2.ray(to)).length() < 1e-4,
            "the free arc carries the grabbed direction exactly"
        );
        assert!(
            (g2.axis_dirs()[1] - g.axis_dirs()[1]).length() > 1e-3,
            "Y was held without being asked for"
        );
    }

    /// The horizon a press grabs to turn about axis `n` is the line **between
    /// the other two axes' vanishing points** (§20.5) — the claim the whole
    /// gesture is described by, checked by finding those two points on it.
    ///
    /// A vanishing point lying on a vanishing line is not a coincidence to be
    /// spot-checked but the definition of both: the line is the image of the
    /// pair plane's infinity, and each of the two axes spanning that plane
    /// vanishes there. Checking it through [`PlaneTrace::distance`] — the same
    /// expression the overlay hit-tests with — is what ties the index the drag
    /// uses to the curve the hand can actually reach for.
    #[test]
    fn the_horizon_of_an_axis_runs_between_the_other_two_vanishing_points() {
        let g = guide(0.5, 0.35, 0.2);
        let s = g.scene(None);
        for n in 0..3 {
            let horizon = g.horizons()[n].expect("every axis shown");
            for i in [(n + 1) % 3, (n + 2) % 3] {
                let vp = s.vps[i].expect("3-point: all finite");
                assert!(
                    horizon.distance(vp) < 1e-2,
                    "axis {n}'s horizon misses axis {i}'s vanishing point by {}px",
                    horizon.distance(vp)
                );
            }
            // …and it is the curve the pass draws, not a second derivation of
            // it: the drag grabs exactly what is on the screen.
            assert_eq!(Some(horizon), s.lines[(n + 1) % 3]);
        }
    }

    /// Grabbing axis `n`'s horizon is a lock on `n` for one drag, so the turn it
    /// asks for is the one the lock already gives: the axis holds still, and a
    /// 2-point setup dragged by its horizon stays exactly 2-point.
    ///
    /// The horizon *is* the lock rather than a second path to the same place —
    /// which is why this asserts the drag through the grabbed axis's index
    /// agrees with the lock's own arm, and why there is no third rotation mode
    /// to keep in step with the other two.
    #[test]
    fn a_horizon_grab_turns_about_its_axis() {
        let g = guide(0.6, 0.0, 0.0);
        let (from, to) = (
            g.center + Vec2::new(-100.0, 40.0),
            g.center + Vec2::new(180.0, -70.0),
        );
        for n in 0..3 {
            let mut held = [false; 3];
            held[n] = true;
            let g2 = g.dragged(from, to, held);
            assert!(
                (g2.axis_dirs()[n] - g.axis_dirs()[n]).length() < 1e-4,
                "axis {n}'s horizon moved axis {n}"
            );
            assert!(
                (g2.rotation.dot(g.rotation).abs() - 1.0).abs() > 1e-4,
                "axis {n}'s horizon turned nothing at all"
            );
        }
        // The vertical one, in the terms an artist would put it in.
        let mut held = [false; 3];
        held[1] = true;
        assert!(
            g.dragged(from, to, held).scene(None).vps[1].is_none(),
            "2-point"
        );
    }

    /// A horizon is a handle only where its curve is drawn (§20.5). It belongs
    /// to a *plane* — the plane's own infinity is what is being drawn — so
    /// switching a plane off takes exactly the one horizon that turns about its
    /// normal, and leaves the other two standing. The guide's eye and its
    /// opacity govern all three at once, as they govern a pencil.
    #[test]
    fn a_horizon_is_offered_only_where_it_is_drawn() {
        let mut g = guide(0.5, 0.35, 0.2);
        assert!(g.horizons().iter().all(Option::is_some));

        // Pair 2 is the Z/X plane, normal to Y: switching it off takes Y's
        // horizon and nothing else.
        g.pairs = [true, true, false];
        assert!(g.horizons()[1].is_none(), "Y's horizon went with its plane");
        assert!(g.horizons()[0].is_some() && g.horizons()[2].is_some());

        g.pairs = [true; 3];
        g.opacity = 0.0;
        assert!(g.horizons().iter().all(Option::is_none), "an invisible one");
    }

    /// A pair plane facing the camera square-on images its infinity nowhere on
    /// the canvas, so there is no horizon to grab — the 1-point pose's X/Y pair,
    /// which is also the one with no station point (§20.2). Turning about the
    /// view axis is then asked for with the lock chip, the control that does not
    /// need a curve to exist.
    #[test]
    fn a_trace_at_infinity_offers_no_horizon() {
        let g = guide(0.0, 0.0, 0.0);
        assert!(g.horizons()[2].is_none(), "Z's horizon is at infinity");
        assert!(g.horizons()[0].is_some() && g.horizons()[1].is_some());
    }

    // --- the fisheye lens (§20.8) ------------------------------------------

    /// The lens pair really is a pair: projecting a direction and casting a
    /// ray back through the image land on the same direction, for points well
    /// past the 90° ring where the fisheye sees behind the camera.
    #[test]
    fn the_fisheye_ray_inverts_its_projection() {
        let g = fisheye(0.0, 0.0, 0.0);
        for p in [
            g.center + Vec2::new(37.0, -12.5),
            g.center + Vec2::new(-700.0, 450.0),
            g.center + Vec2::new(2600.0, 1900.0), // beyond the 90° ring
        ] {
            let back = g.project(g.ray(p)).expect("visible");
            assert!((back - p).length() < 1e-2, "{p:?} -> {back:?}");
        }
    }

    /// The classical claim behind the toggle: a 1-point pose seen through the
    /// fisheye *is* the 5-point curvilinear grid — the view axis vanishes at
    /// the center, and the four transverse poles land on the 90° ring at its
    /// compass points.
    #[test]
    fn one_point_through_the_fisheye_is_five_point() {
        let g = fisheye(0.0, 0.0, 0.0);
        let s = g.scene(None);
        let r90 = s.rings.1.expect("the fisheye has a 90° ring");
        assert!((r90 - 2.0 * g.focal).abs() < 1e-3);
        let vz = s.vps[2].expect("the view axis vanishes at the center");
        assert!((vz - g.center).length() < 1e-3);
        assert!(
            s.anti_vps[2].is_none(),
            "the back pole is the one blind spot"
        );
        // The transverse axes' poles: ±2f east/west, ±2f north/south.
        for (vp, at) in [
            (s.vps[0], Vec2::new(r90, 0.0)),
            (s.anti_vps[0], Vec2::new(-r90, 0.0)),
            (s.vps[1], Vec2::new(0.0, r90)),
            (s.anti_vps[1], Vec2::new(0.0, -r90)),
        ] {
            let vp = vp.expect("a transverse pole is seen");
            assert!((vp - (g.center + at)).length() < 1e-2, "got {vp:?}");
        }
    }

    /// Pair traces under the fisheye: a plane square to the view bows into the
    /// 90° ring itself, while a plane containing the view axis stays straight
    /// — through the center of view, as every great circle through the eye's
    /// axis must.
    #[test]
    fn fisheye_traces_are_circles_except_through_the_axis() {
        let s = fisheye(0.0, 0.0, 0.0).scene(None);
        // Pair 0 spans (X, Y): the picture plane. Its trace is the equator —
        // the 90° ring.
        let Some(PlaneTrace::Circle { center, radius }) = s.lines[0] else {
            panic!("the transverse pair should trace a circle");
        };
        assert!((center - s.center).length() < 1e-3);
        assert!((radius - 2.0 * s.focal).abs() < 1e-2);
        // Pairs containing the view axis trace straight lines through c.
        for k in [1, 2] {
            let Some(PlaneTrace::Line { normal, offset }) = s.lines[k] else {
                panic!("pair {k} should trace a line");
            };
            assert!((normal.dot(s.center) + offset).abs() < 1e-3);
        }
        // And no station points: a flat-plane measuring device has nothing to
        // measure on a curved image.
        assert!(s.stations.iter().all(Option::is_none));
    }

    /// The 45° ring is still the image of the 45° cone — the identity that
    /// makes it the focal length's handle — just at the stereographic radius.
    #[test]
    fn the_45_degree_cone_lands_on_the_fisheye_45_ring() {
        let g = fisheye(0.0, 0.0, 0.0);
        let d = Vec3::new(0.0, 1.0, 1.0).normalize();
        let p = g.project(d).expect("well inside the lens");
        let (r45, _) = g.rings();
        assert!(((p - g.center).length() - r45).abs() < 1e-2);
        assert!((r45 - 2.0 * (std::f32::consts::SQRT_2 - 1.0) * g.focal).abs() < 1e-2);
    }

    /// The drag's contract survives the lens swap: grab a direction through
    /// the fisheye and it is still under the pointer when the drag ends —
    /// nothing about the orbit ever mentioned the projection. Well past the
    /// 45° ring, where the two lenses have long since parted company.
    #[test]
    fn a_fisheye_drag_keeps_the_grabbed_direction_under_the_pointer() {
        let g = fisheye(0.5, 0.35, 0.2);
        let from = g.center + Vec2::new(320.0, -260.0);
        let to = g.center + Vec2::new(-500.0, 1400.0);
        let g2 = g.dragged(from, to, [false; 3]);
        let world = g.rotation.inverse() * g.ray(from);
        let now = g2.rotation * world;
        assert!((now.normalize() - g2.ray(to)).length() < 1e-4);
    }

    // --- the rays through the cursor (§20.9) --------------------------------

    /// The ray's defining property, and the only one worth stating twice: it
    /// passes through the hand. Every axis, every pose, both lenses — because
    /// the plane it traces is the one the eye's ray through that point lies in,
    /// so the point is on the curve by construction rather than by arithmetic
    /// that could drift.
    #[test]
    fn a_ray_passes_through_the_cursor() {
        for g in [guide(0.6, -0.35, 0.2), fisheye(0.6, -0.35, 0.2)] {
            for at in [
                g.center + Vec2::new(210.0, -160.0),
                g.center + Vec2::new(-940.0, 620.0),
            ] {
                for i in 0..3 {
                    let r = g.axis_ray(i, at).expect("a ray away from the poles");
                    // Scaled by the focal length: a circle of radius 10⁴ px is
                    // held to a part in 10⁶ of itself, not to a part in 10⁶ px.
                    assert!(
                        r.trace.distance(at) < 1e-3 * g.focal,
                        "axis {i} at {at:?}: {}",
                        r.trace.distance(at)
                    );
                    // …and on the half of it that is drawn, which is what makes
                    // the cut's orientation a fact rather than a coin toss.
                    assert!(
                        r.cut.is_none_or(|c| c.signed(at) > 0.0),
                        "axis {i} at {at:?}: the ray was cut away from the hand",
                    );
                }
            }
        }
    }

    /// And the property it is *for*: the ray runs to the vanishing point, so
    /// the line under the hand is the line the grid would have a stroke take
    /// there. Stated at a 3-point pose, where all three poles are finite.
    #[test]
    fn a_ray_runs_to_its_vanishing_point() {
        let g = guide(0.7, 0.4, 0.0);
        let s = g.scene(Some(g.center + Vec2::new(-260.0, 180.0)));
        for i in 0..3 {
            let vp = s.vps[i].expect("3-point: every axis vanishes on the canvas");
            let r = s.rays[i].expect("and so every axis has a ray");
            assert!(r.trace.distance(vp) < 1e-3 * g.focal, "axis {i}: {vp:?}");
        }
    }

    /// **And stops there.** The vanishing point is where the world line's two
    /// halves meet, and the far one is the half *behind* the eye — a reflection
    /// of the drawing rather than part of it. So the cut lands on the vanishing
    /// point, and the trace's far side is not drawn.
    #[test]
    fn a_ray_is_cut_at_its_vanishing_point() {
        let g = guide(0.7, 0.4, 0.0);
        let at = g.center + Vec2::new(-260.0, 180.0);
        let s = g.scene(Some(at));
        for i in 0..3 {
            let vp = s.vps[i].expect("3-point: every axis vanishes on the canvas");
            let cut = s.rays[i]
                .expect("a ray")
                .cut
                .expect("with a finite pole to cut at");
            // The boundary passes *through* the vanishing point…
            assert!(cut.signed(vp).abs() < 1e-2, "axis {i}: cut misses the pole");
            // …with the hand on the kept side, and the far side — the same
            // distance along the ray, the other way — off it.
            assert!(cut.signed(at) > 0.0, "axis {i}: the hand was cut away");
            assert!(
                cut.signed(vp + (vp - at)) < 0.0,
                "axis {i}: the ray runs past the pole"
            );
        }
    }

    /// An axis lying in the picture plane has no vanishing point to run to, and
    /// its ray is the parallel line through the hand — which needs no case of
    /// its own, the trace being derived from a plane rather than from a join.
    /// In 1-point that is a horizontal for X and a vertical for Y, whatever the
    /// cursor.
    #[test]
    fn a_ray_of_an_axis_at_infinity_is_the_parallel_through_the_cursor() {
        let g = guide(0.0, 0.0, 0.0);
        let at = g.center + Vec2::new(137.0, -412.0);
        let s = g.scene(Some(at));
        assert!(s.vps[0].is_none() && s.vps[1].is_none());
        // X runs across the picture plane and Y down it, so their rays are the
        // horizontal and the vertical through the hand. A line's normal is
        // across it, hence the swap.
        for (i, across) in [(0, Vec2::X), (1, Vec2::Y)] {
            let Some(CursorRay {
                trace: t @ PlaneTrace::Line { normal, .. },
                cut,
            }) = s.rays[i]
            else {
                panic!("axis {i}'s ray should be a straight parallel");
            };
            assert!(
                normal.dot(across).abs() < 1e-4,
                "axis {i} runs the wrong way"
            );
            assert!(t.distance(at) < 1e-2, "axis {i} misses the cursor");
            // And nothing cuts it: the whole world line is in front of the eye,
            // there being no point on it where the eye's ray turns around.
            assert!(cut.is_none(), "axis {i} was cut at a pole it does not have");
        }
    }

    /// Under the fisheye a ray bows into the arc through **both** poles of its
    /// axis — the same circle the lens draws every world line as (§20.8), and
    /// it comes out of the identical derivation.
    #[test]
    fn a_fisheye_ray_is_the_arc_through_both_poles() {
        let g = fisheye(0.55, -0.3, 0.15);
        let at = g.center + Vec2::new(430.0, 260.0);
        let s = g.scene(Some(at));
        for i in 0..3 {
            let Some(CursorRay {
                trace: t @ PlaneTrace::Circle { .. },
                cut,
            }) = s.rays[i]
            else {
                panic!("axis {i}'s ray should bow into a circle");
            };
            let cut = cut.expect("two poles to cut between");
            for pole in [s.vps[i], s.anti_vps[i]] {
                let pole = pole.expect("the fisheye sees both poles of every axis");
                assert!(t.distance(pole) < 1e-3 * g.focal, "axis {i} pole {pole:?}");
                // Both poles bound the arc, so the cut — their chord — runs
                // through both, and the hand is on the arc between them.
                assert!(
                    cut.signed(pole).abs() < 1e-2 * g.focal,
                    "axis {i}: the chord misses pole {pole:?}",
                );
            }
            assert!(cut.signed(at) > 0.0, "axis {i}: the hand was cut away");
        }
    }

    /// The one pose with no ray to name: the hand resting on a vanishing point,
    /// where every line of that axis's pencil runs through it and none of them
    /// is *the* one. Its two neighbours are unaffected — the answer is about
    /// one axis, not about the cursor.
    #[test]
    fn no_ray_at_the_vanishing_point_itself() {
        let g = guide(0.7, 0.4, 0.0);
        let vp = g.scene(None).vps[2].expect("Z vanishes on the canvas");
        let s = g.scene(Some(vp));
        assert!(s.rays[2].is_none(), "no ray at Z's own vanishing point");
        assert!(s.rays[0].is_some() && s.rays[1].is_some());
    }

    /// No pointer, no rays. A render nobody's hand is over — an export, the
    /// navigator's miniature — asks for the scene the same way and gets the
    /// overlay without them.
    #[test]
    fn no_cursor_draws_no_rays() {
        let s = guide(0.7, 0.4, 0.0).scene(None);
        assert!(s.rays.iter().all(Option::is_none));
    }
}
