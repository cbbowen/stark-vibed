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
//! - the **lattice** the fans measure out: a cube of cells, whose three
//!   coordinate planes through one corner each carry a grid of *squares*, which
//!   is what makes a grid something to measure on rather than to look at
//!   (§20.3);
//! - the **45° circle**, radius = focal length: the cone at 45° off the view
//!   axis, the classical "keep the drawing inside this" bound (§20.1);
//! - the three **station points**, the eye rotated into the picture plane
//!   about each vanishing line — each sits on the Thales circle over its two
//!   vanishing points, which is why it sees them at a right angle (§20.2).
//!
//! All of it is projection through a [`Lens`], and swapping the rectilinear
//! lens for the stereographic **fisheye** (§20.8) is one field: the fans, the
//! orbit drag, the snap and the locks are stated in direction space and carry
//! over untouched, while the straight guide lines bow into exact circles and
//! both poles of every axis come into view.
//!
//! # Why the whole of it is here, in the document
//!
//! A guide used to be view state — per-client, unlogged, unsaved, outside the
//! undo history — on the argument that it is an aid for the hand holding the
//! pen, like the pan and the zoom. That argument was wrong about what a guide
//! *is*. A perspective set up over a drawing is part of the drawing's
//! construction: it is what the artist reasons in, it is worth as much care as
//! a layer, and losing it on reload — or leaving a collaborator unable to see
//! the scaffold the work is being built on — is losing work. So a guide is a
//! document entity now, with an id, saved in the log, replicated to peers and
//! undoable (§20.5).
//!
//! The one thing that stayed per-client is **whether a guide is drawn**. That
//! genuinely is about the hand rather than about the drawing: shutting a
//! guide's eye to see the picture underneath must not reach across the session,
//! must not be saved, and must not cost an undo step. It lives beside the pan
//! and the zoom in `stark-engine`'s `Session`, which is also where the two are
//! combined into one per-client reading of the roster (`GuideInfo`).
//!
//! And it defaults to **not drawn**, which is the same sentence read forwards: a
//! document now carries every perspective anyone ever built over it, so laying
//! them all on the canvas the moment it opens would make the scaffolding into
//! something you have to clear away. The construction is kept; looking at it is
//! a thing you ask for. Nothing here knows that — a camera has no eye — but it
//! is why the derivations below gate on the guide's own controls and never on
//! whether it is on screen.
//!
//! # What is derived
//!
//! [`PerspectiveGuide`] is one guide's camera; the document carries a roster of
//! them keyed by [`GuideId`], and every edit is an action
//! ([`AddGuide`](super::ActionKind::AddGuide),
//! [`SetGuide`](super::ActionKind::SetGuide),
//! [`SetGuideName`](super::ActionKind::SetGuideName),
//! [`MoveGuide`](super::ActionKind::MoveGuide),
//! [`RemoveGuide`](super::ActionKind::RemoveGuide)).
//! [`PerspectiveGuide::scene`] derives the [`GuideScene`] the compositor's
//! guide pass draws (§20.4), and [`PerspectiveGuide::dragged`] is the direct
//! manipulation: the grabbed direction follows the pointer, held about the
//! world axes by the locks — one of which is what grabbing a
//! [`horizon`](PerspectiveGuide::horizons) puts on for a single drag (§20.5).
//! [`PerspectiveGuide::pencils`] and
//! [`PerspectiveGuide::planes`] are the other direction — what the drawing
//! assist holds a snapped stroke to: the axes a line is aimed along (§20.6) and
//! the planes a loop is a circle on (§20.7), gathered for it as a [`Scaffold`].
//!
//! **None of it needs a pixel**, which is what puts the derivations here beside
//! the fact rather than leaving them in the engine: every one is a pure
//! function of the camera, exactly as [`fill_bounds`](super::fill_bounds) and
//! the homography solve are pure functions of their own action payloads (§2).
//! What the engine keeps is the part that genuinely needs the GPU — packing a
//! [`GuideScene`] into the guide pass's uniform — and the part that needs the
//! *session*, which is the visibility filter deciding which guides are handed
//! to [`Scaffold::of`] at all. Everything here is plain CPU math, computed once
//! per render and unit-tested against the classical theorems.

use serde::{Deserialize, Serialize};

use glam::{Mat3, Quat, Vec2, Vec3};

use super::action::ActionId;
use crate::geom::{Ellipse, principal_axis};

/// The shortest focal length a guide may hold, canvas px (`sanitized`).
///
/// Not a taste limit — the panel's own range starts two orders of magnitude
/// above it (§20.5) — but the floor under which the camera stops being one:
/// `ray` and `project` both divide by the focal length, so at zero every
/// direction images at infinity and the whole overlay is one uniform of
/// `inf`. One pixel is already far past any pose an artist can reach.
const MIN_FOCAL: f32 = 1.0;

/// The shortest a quaternion may be and still name a direction to normalize
/// towards (`sanitized`). Below it the division amplifies whatever rounding
/// produced the value into an arbitrary pose, so the identity — looking straight
/// down the world Z axis, the 1-point case — is the honest answer instead.
const MIN_QUAT_SQ: f32 = 1e-12;

/// A lattice whose corner sits closer than this many cells to the eye names no
/// grid (§20.3). Not a tolerance: there all three planes pass through the eye,
/// every family's index is constant, and what the fans would draw is not an
/// inaccurate grid but the whole canvas at once.
const LATTICE_EPS: f32 = 1e-3;

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

/// How far into a plane its chart reaches, in units of the plane's own distance
/// from the eye (§20.7). Past it a point is treated as not being on the plane at
/// all.
///
/// Not a tolerance but a horizon: plane distance and the image's distance from
/// the vanishing line are reciprocal, so at a thousand eye-heights the whole
/// remaining plane is imaged inside about a pixel of the line, and a circle out
/// there is a sliver with no shape to recognize. It is also what keeps the chart
/// from answering with astronomical coordinates as the divisor goes to zero.
const PLANE_REACH: f32 = 1e3;

/// Below this much normal component along the view axis, a fisheye pair trace
/// is drawn as its limiting straight line rather than as a circle (§20.8).
/// Coarser than [`LINE_EPS`] on purpose: the circle's radius is `2f/|m.z|`,
/// and at radii near ten million pixels the f32 distance-to-ring subtraction
/// cancels catastrophically and the "line" wobbles. The line it is replaced
/// with deviates from the true circle by `f·|m.z|` at most — under a pixel at
/// this threshold — so the swap is invisible where it happens.
const FISHEYE_LINE_EPS: f32 = 1e-3;

/// The lens a guide projects through (§20.8): how a world direction becomes a
/// canvas point, and the *only* thing that differs between a classical and a
/// curvilinear perspective — the camera, the fans, the drag and the locks are
/// all stated in direction space and never notice.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
pub enum Lens {
    /// The gnomonic picture plane of classical perspective: straight world
    /// lines image straight. Directions are projective (a direction and its
    /// negation land together), so each axis has one vanishing point, possibly
    /// at infinity.
    #[default]
    Rectilinear,
    /// The curvilinear lens: **stereographic** projection of the view sphere,
    /// scaled to agree with the rectilinear lens at the center of view.
    /// Chosen over the other fisheye mappings because it is conformal and
    /// takes circles to circles — every guide line is an exact circle on the
    /// canvas, closed-form. Directions are *not* projective here: an axis and
    /// its negation land at two points (both poles are seen), which is why a
    /// 1-point pose shows the classical **5-point** grid (§20.8).
    Fisheye,
}

impl Lens {
    /// The radii of the dressed view-cone rings, per unit focal length:
    /// the 45° ring, and the 90° ring where the lens has one.
    ///
    /// Rectilinear: `tan 45° = 1`, so the 45° ring *is* the focal length, and
    /// the 90° cone is at infinity. Fisheye: `2·tan(θ/2)` puts 45° at
    /// `2(√2 − 1) ≈ 0.83` and the 90° ring — the image of the whole forward
    /// hemisphere's rim, the circle the classical 5-point grid is drawn in —
    /// at exactly `2`.
    pub fn ring_factors(self) -> (f32, Option<f32>) {
        match self {
            Lens::Rectilinear => (1.0, None),
            Lens::Fisheye => (2.0 * (std::f32::consts::SQRT_2 - 1.0), Some(2.0)),
        }
    }
}

/// Stable identifier for a drawing guide within a document (§20.5).
///
/// **The id of the action that added it**, and that is the whole of how guide
/// identity is kept unique across peers — there is no counter here, and nothing
/// to resync when a log is picked back up (compare
/// [`LayerId::mint`](super::LayerId::mint), which needs both).
///
/// An [`ActionId`] is already the log's total-order key `(lamport, actor)`, so
/// it is already globally unique: two actions cannot share one, therefore two
/// guides cannot share a `GuideId`. That makes "no two guides carry the same
/// id" a property of the representation rather than a rule a call site could
/// forget, which is what §1 asks for wherever a guarantee can be made
/// structural.
///
/// It works here and not for layers because of a difference between the two
/// vocabularies rather than a preference: one `AddGuide` mints exactly one
/// guide, where `DuplicateLayer` mints one id per layer of a subtree and so has
/// to carry a map of them. An action id can name one thing.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    carbonite::Schema,
)]
pub struct GuideId(pub ActionId);

/// The perspective-grid guide: a camera, plus how densely to dress it (§20.1).
///
/// **Document state** (§20.5): saved with the log, replicated to peers, and
/// undoable like any other edit. What is *not* here is the guide's name, which
/// the roster carries beside it because the state holds it as an `Arc<str>` and
/// the wire as a `String` — the split
/// [`SetLayerName`](super::ActionKind::SetLayerName) already draws — and
/// whether it is drawn, which is per-client and lives in the session.
///
/// `Copy`, which it became by losing those two fields and is worth keeping so:
/// a drag rebuilds one of these per pointer sample from the pose the press
/// started in, and every derivation below takes `&self` and answers with
/// something new.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct PerspectiveGuide {
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
    /// The lens the guide projects through (§20.8): classical straight-line
    /// perspective, or the curvilinear fisheye. Everything else about the
    /// camera — orientation, center, focal — means the same thing under both,
    /// which is what makes this one field a *toggle* rather than a mode.
    pub lens: Lens,
    /// The **lattice** the fans measure out (§20.3): how far each of the three
    /// squared planes lies from the eye, in *cells*, one component per world
    /// axis. Equivalently the corner they meet at, as a displacement from the
    /// eye in world-axis coordinates.
    ///
    /// One vector is the whole of the grid's metric, and it has to be a world
    /// quantity rather than a canvas one: a camera carries no world scale, so
    /// the only thing about a cell that can be seen is how many of them lie
    /// between the eye and a plane. Scaling this vector is therefore the density
    /// control — twice as many cells to the same planes is a grid of half the
    /// size — and turning it is where the grid sits.
    ///
    /// **The grid's phase is the eye, not the corner.** Every guide line lies a
    /// whole number of cells from the *viewer*, so the foot of the perpendicular
    /// from the eye to each plane is one of that plane's cell corners: look
    /// straight down any axis and the crossing under the center of view is the
    /// grid's own, at every scale and for any lattice whatever. Anchoring the
    /// phase at the corner instead made that true only when the corner's
    /// components happened to be whole — and a scale control that halves is
    /// exactly what turns `(-4, 3, 6)` into `(-1, ¾, 1½)`, one axis on a line,
    /// one a quarter off and one a half, which reads as three grids laid out
    /// separately. Nothing about the corner needs to be round now, and the
    /// scale expands and contracts *about the viewer* rather than pivoting on a
    /// line some cells away.
    ///
    /// A zero component puts the eye *inside* the plane normal to that axis,
    /// whose grid then images onto its own vanishing line and has nothing to
    /// show; the pass declines it by the test that says what it is, that a level
    /// set with no gradient is not a curve. A zero *vector* is that for all
    /// three at once, and [`corner`](Self::corner) declines it at the source.
    pub lattice: Vec3,
    /// Master opacity of the whole overlay.
    pub opacity: f32,
    /// Which of the three **pair planes** are drawn — pair `k` being the plane
    /// axes `k` and `k + 1` span, so the three are XY, YZ and ZX (§20.3), the
    /// names the bar's chips wear.
    ///
    /// The plane rather than the axis, because the plane is what is actually
    /// drawn. A guide line is a line *in* a plane: an axis on its own rules
    /// nothing, and the two families an axis contributes live on two different
    /// planes and are only ever wanted together by coincidence. Naming the axes
    /// instead could reach three of the eight states — none, one plane, all
    /// three — because a plane needed both of its axes and so a second plane
    /// came free with the second axis. "Draw the ground and the near wall, not
    /// the far one" was not sayable at all, and it is the most common thing to
    /// want.
    ///
    /// Everything a plane carries follows its flag: the two fans that rule it,
    /// its vanishing trace (and with it the [`horizon`](Self::horizons) that
    /// turns about its normal), and its station point. An **axis** is drawn
    /// when either plane it is a side of is — that is what its vanishing point
    /// and its stroke [`pencil`](Self::pencils) follow, since an axis with no
    /// plane left draws no line anywhere.
    ///
    /// A statement about what the guide *is*, and so document state, where the
    /// guide's own eye is a statement about what one client is looking at and
    /// is not.
    pub pairs: [bool; 3],
}

impl Default for PerspectiveGuide {
    /// Centred on the canvas origin, at a moderate lens, turned to the
    /// most-reached-for case (2-point). The caller placing a new guide moves
    /// `center` to where the artist is looking.
    ///
    /// The lattice stands the artist four cells above its floor and a short way
    /// in from its two walls, which under this turn puts the corner just below
    /// the center of view — squares the eye can count rather than a haze it
    /// reads as tone. All three components are nonzero, as they must be for all
    /// three planes to have a grid to show at all; whole numbers are not required
    /// of them, but they cost nothing and put the corner's own three edges on
    /// guide lines.
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            focal: 900.0,
            rotation: Quat::from_rotation_y(30f32.to_radians()),
            lens: Lens::Rectilinear,
            lattice: Vec3::new(-4.0, 4.0, 8.0),
            opacity: 0.65,
            pairs: [true; 3],
        }
    }
}

impl PerspectiveGuide {
    /// This camera with every number finite and in range — the guide's half of
    /// [`ActionKind::sanitized`](super::ActionKind::sanitized), the one funnel an
    /// action passes through on its way into the document (§21.5).
    ///
    /// Every field here reaches a shader: the center and the focal length are
    /// the guide pass's own uniform lanes (§20.4), the rotation becomes the axis
    /// frame every fan is ruled from, and the opacity multiplies the overlay. A
    /// `NaN` in any of them is not a wrong picture but no picture — and it would
    /// be *saved*, now that a guide is document state.
    ///
    /// **Idempotent**, which is what lets it run both at mint and on the way into
    /// state without a load becoming a small edit: every step is a clamp or a
    /// replacement, and the rotation is left exactly alone once it is near enough
    /// to unit for [`Quat::is_normalized`] — so a second pass changes nothing a
    /// first pass produced.
    ///
    /// The focal floor is one canvas pixel. Not a taste limit — the panel's own
    /// range starts far above it (§20.5) — but the point where [`ray`](Self::ray)
    /// and [`project`](Self::project) stop being a projection at all, since both
    /// divide by it.
    #[must_use]
    pub fn sanitized(self) -> Self {
        /// A finite replacement, or the fallback — the same shape three times
        /// below, and worth naming so the fallback reads beside what it stands in
        /// for rather than after two lines of `is_finite`.
        fn or(v: f32, fallback: f32) -> f32 {
            if v.is_finite() { v } else { fallback }
        }
        let default = Self::default();
        Self {
            center: if self.center.is_finite() {
                self.center
            } else {
                default.center
            },
            focal: or(self.focal, default.focal).max(MIN_FOCAL),
            // Left untouched once it is a rotation, so this is idempotent. A
            // quaternion that is not one — un-normalized by a drag's accumulated
            // error past the tolerance, or `NaN` — is renormalized, and one with
            // no direction left to normalize towards looks down the world Z axis,
            // which is the identity pose and the honest answer to "no rotation
            // was stated".
            rotation: match self.rotation {
                q if q.is_finite() && q.is_normalized() => q,
                q if q.is_finite() && q.length_squared() > MIN_QUAT_SQ => q.normalize(),
                _ => Quat::IDENTITY,
            },
            lens: self.lens,
            lattice: if self.lattice.is_finite() {
                self.lattice
            } else {
                default.lattice
            },
            opacity: or(self.opacity, 1.0).clamp(0.0, 1.0),
            pairs: self.pairs,
        }
    }

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
    /// `None` for a corner sitting *on* the eye ([`LATTICE_EPS`]): there a cell
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

    /// The dressed view-cone rings, canvas px: the 45° ring, and the 90° ring
    /// where the lens has one ([`Lens::ring_factors`]). What the overlay's
    /// lens-drag grabs — dragging either ring *is* setting the focal length.
    pub fn rings(&self) -> (f32, Option<f32>) {
        let (r45, r90) = self.lens.ring_factors();
        (r45 * self.focal, r90.map(|r| r * self.focal))
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

    /// The **vanishing trace** of pair plane `k` — the plane axes `k` and
    /// `k + 1` span — exactly as the guide pass draws it (§20.2, §20.8): the
    /// straight vanishing line of the rectilinear lens, or the circle the
    /// fisheye bows it into.
    ///
    /// `None` when the trace is at infinity, which is the plane facing the
    /// camera square-on: there is no curve on the canvas, and so nothing to
    /// draw, to measure a station point against, or to grab.
    pub fn pair_trace(&self, k: usize) -> Option<PairTrace> {
        let dirs = self.axis_dirs();
        let m = dirs[k % 3].cross(dirs[(k + 1) % 3]);
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
                (len >= LINE_EPS).then(|| PairTrace::Line {
                    normal: planar / len,
                    offset: (f * m.z - planar.dot(c)) / len,
                })
            }
            // The stereographic image of the great circle of directions in the
            // pair plane: an exact circle — conformality's gift (§20.8) — with
            // center `c + 2f·m.xy/m.z` and radius `2f/|m.z|`, from substituting
            // the inverse projection into `m·d = 0`. A pair plane containing the
            // view axis (m.z ≈ 0) images straight, through the center of view.
            Lens::Fisheye if m.z.abs() > FISHEYE_LINE_EPS => Some(PairTrace::Circle {
                center: c + planar * (2.0 * f / m.z),
                radius: 2.0 * f / m.z.abs(),
            }),
            Lens::Fisheye => planar.try_normalize().map(|n| PairTrace::Line {
                normal: n,
                offset: -n.dot(c),
            }),
        }
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
    pub fn horizons(&self) -> [Option<PairTrace>; 3] {
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
    /// [`Scaffold::of`] being handed only the guides this client draws — one
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
    pub fn scene(&self) -> GuideScene {
        let dirs = self.axis_dirs();
        let c = self.center;
        let f = self.focal;

        // Vanishing points: where each axis's direction images ([`project`]).
        // Rectilinear directions are projective, so the antipodes add nothing
        // and stay empty; the fisheye sees both poles of every axis — which is
        // the whole reason a 1-point pose reads as 5-point under it (§20.8).
        let vps = dirs.map(|d| self.project(d));
        let anti_vps = match self.lens {
            Lens::Rectilinear => [None; 3],
            Lens::Fisheye => dirs.map(|d| self.project(-d)),
        };

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
                && let Some(PairTrace::Line { normal: n, offset }) = lines[k]
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

/// One plane of one guide — the surface spanned by a *pair* of axes — as a
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
    canvas_from_plane: Mat3,
    plane_from_canvas: Mat3,
}

impl AxisPlane {
    /// `p` in the plane's own coordinates, or `None` when it does not lie on the
    /// plane in any useful sense — at or beyond [`PLANE_REACH`].
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
    let along = dir?.perp();
    let radii = Vec2::new((s / minor).sqrt(), (s / major).sqrt());
    radii
        .is_finite()
        .then(|| Ellipse::new(center, radii, along.y.atan2(along.x)))
}

/// `Mᵀ C M` — a conic carried through the map `M` takes points by.
fn congruent(conic: Mat3, m: Mat3) -> Mat3 {
    m.transpose() * conic * m
}

/// What every guide on the screen offers a stroke that is snapping: the axes a
/// line may be aimed along (§20.6), and the planes a loop may be a circle on
/// (§20.7).
///
/// Gathered whole, so the assist takes one argument and never sees a
/// [`PerspectiveGuide`] — what it needs from the guides is a list of geometry,
/// not a list of guides, and nothing about a name, an eye or an opacity
/// survives the crossing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scaffold {
    /// Every shown axis of every guide handed in.
    pub axes: Vec<AxisPencil>,
    /// Every plane *both* of whose axes are shown.
    pub planes: Vec<AxisPlane>,
}

impl Scaffold {
    /// What a run of guides puts up. Nothing shown allocates nothing, which is
    /// the common case — most strokes are drawn without a grid.
    ///
    /// **`guides` is what this client draws**, not the document's whole roster:
    /// a guide's eye is per-client (§20.5), so the caller drops the ones it has
    /// shut before handing them over. That is the one place the visibility
    /// filter is applied — [`pencils`](PerspectiveGuide::pencils) and
    /// [`planes`](PerspectiveGuide::planes) gate on the document's own controls
    /// and know nothing about an eye — and the caller is the engine, which is
    /// the only side holding both halves.
    ///
    /// Takes an iterator rather than a slice for exactly that reason: the
    /// filtered roster is a `filter` over the document's, and asking for a slice
    /// would make every snap collect one. Both lists are filled in the **one**
    /// pass that costs, which is also what keeps the iterator from needing to be
    /// `Clone` — `rpds`'s is not.
    pub fn of<'a>(guides: impl IntoIterator<Item = &'a PerspectiveGuide>) -> Self {
        let mut out = Self::default();
        for g in guides {
            out.axes.extend(g.pencils().into_iter().flatten());
            out.planes.extend(g.planes().into_iter().flatten());
        }
        out
    }
}

/// The image of one pair plane's directions — a **vanishing trace**: the
/// straight vanishing line of the rectilinear lens, or the circle the fisheye
/// bows it into (§20.8). One curve either way, so the shader carries a kind
/// beside four numbers rather than two pipelines.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PairTrace {
    /// `normal · p + offset = 0`, with `normal` unit — evaluating is signed
    /// canvas-px distance.
    Line { normal: Vec2, offset: f32 },
    /// `|p − center| = radius`, canvas px.
    Circle { center: Vec2, radius: f32 },
}

impl PairTrace {
    /// How far canvas point `p` lies from the trace, canvas px — the one
    /// number both kinds answer, which is why the shader draws them with one
    /// `stroke_cov` and the overlay hit-tests them with one comparison
    /// (§20.4, §20.5).
    pub fn distance(self, p: Vec2) -> f32 {
        match self {
            PairTrace::Line { normal, offset } => (normal.dot(p) + offset).abs(),
            PairTrace::Circle { center, radius } => (p.distance(center) - radius).abs(),
        }
    }
}

/// The derived, draw-ready guide: what the compositor's guide pass uniform
/// carries (§20.4). All canvas-space; the pass adds the view mapping.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GuideScene {
    /// Center of view, canvas px.
    pub center: Vec2,
    /// Focal length, canvas px.
    pub focal: f32,
    /// The lens the fragment pass must invert (§20.8).
    pub lens: Lens,
    /// The dressed view-cone rings about the center, canvas px: the 45° ring,
    /// and the 90° ring where the lens has one ([`PerspectiveGuide::rings`]).
    pub rings: (f32, Option<f32>),
    /// The lattice's corner in camera space, in cells (§20.3) — what the six
    /// families of guide lines step away from. `None` for a guide whose lattice
    /// names no grid, and then no fan is drawn
    /// ([`PerspectiveGuide::corner`]).
    pub lattice: Option<Vec3>,
    /// Master opacity, 0..=1.
    pub opacity: f32,
    /// World axes in camera space. Axis `i` carries **two** families of guide
    /// lines, one for each pair plane it lies in, and both are read off the
    /// ray's components along the *other* two axes (§20.3) — so the fan
    /// arithmetic never computes a vanishing point and never branches on
    /// whether one is finite.
    pub dirs: [Vec3; 3],
    /// Per-**axis** opacity (0 = the axis is nowhere on the canvas). What the
    /// markers belonging to an axis alone follow — its vanishing points — and
    /// derived rather than set: an axis is drawn when either pair plane it is a
    /// side of is ([`PerspectiveGuide::is_drawn`]).
    pub axis_alpha: [f32; 3],
    /// Per-**pair-plane** opacity (0 = the plane is switched off), pair `k`
    /// being the plane axes `k` and `k + 1` span. What everything that lives on
    /// a plane follows: the two fans that rule it, its vanishing trace and its
    /// station point.
    ///
    /// Two arrays because the pass asks two genuinely different questions, and
    /// neither answers the other: switching one plane off leaves both of its
    /// axes on the screen, still ruling the other planes they border, so a dark
    /// plane does not darken an axis. Only losing *both* of an axis's planes
    /// does that.
    pub pair_alpha: [f32; 3],
    /// Vanishing trace of pair `(k, k+1)`; `None` when it is at infinity.
    pub lines: [Option<PairTrace>; 3],
    /// Where each axis's direction images, canvas px; `None` off the lens.
    pub vps: [Option<Vec2>; 3],
    /// Where each axis's *opposite* direction images — the second pole, which
    /// only the fisheye separates from the first (§20.8).
    pub anti_vps: [Option<Vec2>; 3],
    /// Station point of pair `(k, k+1)`, canvas px; rectilinear only.
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
        let Some(PairTrace::Line { normal, offset }) = s.lines[2] else {
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
        assert!(g.scene().lattice.is_some());
        g.lattice = Vec3::ZERO;
        assert!(g.corner().is_none());
        assert!(g.scene().lattice.is_none());
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
        let s = g.scene();
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
        assert!(g.scene().vps[0].is_none());
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
                assert_eq!(g.scene().axis_alpha[i] > 0.0, expected, "planes {pairs:?}");
                // A plane answers for itself, and for nothing else.
                assert_eq!(g.planes()[i].is_some(), pairs[i], "planes {pairs:?}");
                assert_eq!(g.scene().pair_alpha[i] > 0.0, pairs[i], "planes {pairs:?}");
            }
        }
    }

    // --- the planes circles are drawn on (§20.7) ---------------------------

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
    /// vanishes there. Checking it through [`PairTrace::distance`] — the same
    /// expression the overlay hit-tests with — is what ties the index the drag
    /// uses to the curve the hand can actually reach for.
    #[test]
    fn the_horizon_of_an_axis_runs_between_the_other_two_vanishing_points() {
        let g = guide(0.5, 0.35, 0.2);
        let s = g.scene();
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
            g.dragged(from, to, held).scene().vps[1].is_none(),
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

    fn fisheye(yaw: f32, pitch: f32, roll: f32) -> PerspectiveGuide {
        PerspectiveGuide {
            lens: Lens::Fisheye,
            ..guide(yaw, pitch, roll)
        }
    }

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
        let s = g.scene();
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
        let s = fisheye(0.0, 0.0, 0.0).scene();
        // Pair 0 spans (X, Y): the picture plane. Its trace is the equator —
        // the 90° ring.
        let Some(PairTrace::Circle { center, radius }) = s.lines[0] else {
            panic!("the transverse pair should trace a circle");
        };
        assert!((center - s.center).length() < 1e-3);
        assert!((radius - 2.0 * s.focal).abs() < 1e-2);
        // Pairs containing the view axis trace straight lines through c.
        for k in [1, 2] {
            let Some(PairTrace::Line { normal, offset }) = s.lines[k] else {
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

    /// A fisheye guide puts up no scaffold: its lines are arcs, and the assist
    /// snaps strokes to straight pencils and flat charts — offering those
    /// would bend a considered line onto geometry the guide does not draw.
    #[test]
    fn a_fisheye_guide_offers_no_scaffold() {
        let g = fisheye(0.5, 0.35, 0.2);
        let up = Scaffold::of(std::slice::from_ref(&g));
        assert!(up.axes.is_empty());
        assert!(up.planes.is_empty());
    }
}
