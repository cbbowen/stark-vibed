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
//! A guide is not an aid for the hand holding the pen, the way the pan and the
//! zoom are. A perspective set up over a drawing is part of the drawing's
//! construction: it is what the artist reasons in, it is worth as much care as a
//! layer, and losing it on reload — or leaving a collaborator unable to see the
//! scaffold the work is being built on — is losing work. So a guide is a document
//! entity, with an id, saved in the log, replicated to peers and undoable (§20.5).
//!
//! The one thing that stayed per-client is **whether a guide is drawn**. That
//! genuinely is about the hand rather than about the drawing: shutting a
//! guide's eye to see the picture underneath must not reach across the session,
//! must not be saved, and must not cost an undo step. It lives beside the pan
//! and the zoom in `stark-engine`'s `Session`, which is also where the two are
//! combined into one per-client reading of the roster (`GuideInfo`).
//!
//! And it defaults to **not drawn**, which is the same sentence read forwards: a
//! document carries every perspective anyone ever built over it, so laying them all
//! on the canvas the moment it opens would make the scaffolding into something you
//! have to clear away. The construction is kept; looking at it is a thing you ask
//! for. Nothing here knows that — a camera has no eye — but it is why the
//! derivations below gate on the guide's own controls and never on whether it is on
//! screen.
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

use glam::{Quat, Vec2, Vec3};

use super::action::ActionId;
use crate::{finite_in, finite_or};

mod camera;
mod conic;

// The derivations' own public types, lifted so `document::guide` still means one
// thing from outside — the split is a matter of which file to read, not of where a
// type lives (§20.5).
pub use camera::{AxisPencil, CursorRay, Halfplane, PlaneTrace};
pub use conic::AxisPlane;

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

/// The rotation a guide is held to: itself if it is already one, renormalized if it
/// names a direction, and the identity if it names none.
///
/// **The upper bound is the one that bites.** `length_squared` overflows `f32` at a
/// component of ~1.8e19, and `Quat::normalize` then divides by an infinite length and
/// returns `Quat(0, 0, 0, 0)` — finite, so the funnel reports it clean, and not a
/// rotation, so `Mat3::from_quat` of it is the zero matrix. Worse, it is not *stable*:
/// a second pass reads that zero as "no direction" and answers `IDENTITY`, so a guide
/// logged through one funnel reloads as a different guide through the next — the
/// load-is-a-small-edit the §20.5 bargain rests on not happening.
///
/// A quaternion that large is corruption or a hostile peer rather than a pose anyone
/// stated, so it lands where the too-small ones land instead of being rescaled to
/// recover a direction nobody meant.
fn pose(q: Quat) -> Quat {
    let sq = q.length_squared();
    match q {
        q if q.is_finite() && q.is_normalized() => q,
        q if q.is_finite() && sq.is_finite() && sq > MIN_QUAT_SQ => q.normalize(),
        _ => Quat::IDENTITY,
    }
}

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

/// How nearly the hand may point along an axis before that axis has no cursor
/// ray (§20.9). The number is `sin θ` between the axis and the eye's ray, both
/// unit, so it is an angle rather than a distance and means the same under
/// either lens.
///
/// This is where the pointer has arrived at the axis's own vanishing point, and
/// the ray is undefined there for a reason no tolerance can fix: *every* line of
/// the pencil passes through that point, so there is no one line to draw. What
/// the epsilon adds is that the *approach* is undefined too — the plane's normal
/// is the cross product of two nearly parallel unit vectors, and its direction
/// dissolves into rounding well before its length reaches zero, which a ray
/// drawn off it would report as a spin through a full turn.
///
/// The band it retires is `RAY_EPS · f / a.z` canvas px around the vanishing
/// point, so it grows with the focal length: a tenth of a pixel at the short end
/// of the panel's range and a little over one at the long end (`FOCAL_RANGE` is
/// 120–12000), always under the ten-pixel disc drawn over that point. Against
/// the other side of the trade — the cross product of two unit vectors carries a
/// few ulps of absolute error, so this leaves three orders of magnitude of
/// headroom, and the residual wobble at the threshold itself is under a
/// thousandth of a radian.
const RAY_EPS: f32 = 1e-4;

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
/// to resync when a log is picked back up. [`LayerId`](super::LayerId) is the same
/// answer with one field more.
///
/// An [`ActionId`] is already the log's total-order key `(lamport, actor)`, so
/// it is already globally unique: two actions cannot share one, therefore two
/// guides cannot share a `GuideId`. That makes "no two guides carry the same
/// id" a property of the representation rather than a rule a call site could
/// forget, which is what §1 asks for wherever a guarantee can be made
/// structural.
///
/// A guide needs nothing beside the action id, where a layer does: one `AddGuide`
/// mints exactly one guide, so there is no *which one* to say — where
/// `DuplicateLayer` mints one per layer of a subtree, which is what
/// [`LayerId`](super::LayerId)'s `k` answers.
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
        let default = Self::default();
        Self {
            center: if self.center.is_finite() {
                self.center
            } else {
                default.center
            },
            focal: finite_or(self.focal, default.focal).max(MIN_FOCAL),
            // Left untouched once it is a rotation, so this is idempotent. A
            // quaternion that is not one — un-normalized by a drag's accumulated
            // error past the tolerance, or `NaN` — is renormalized, and one with
            // no direction left to normalize towards looks down the world Z axis,
            // which is the identity pose and the honest answer to "no rotation
            // was stated".
            rotation: pose(self.rotation),
            lens: self.lens,
            lattice: if self.lattice.is_finite() {
                self.lattice
            } else {
                default.lattice
            },
            opacity: finite_in(self.opacity, 1.0, (0.0, 1.0)),
            pairs: self.pairs,
        }
    }
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
    pub lines: [Option<PlaneTrace>; 3],
    /// Where each axis's direction images, canvas px; `None` off the lens.
    pub vps: [Option<Vec2>; 3],
    /// Where each axis's *opposite* direction images — the second pole, which
    /// only the fisheye separates from the first (§20.8).
    pub anti_vps: [Option<Vec2>; 3],
    /// Station point of pair `(k, k+1)`, canvas px; rectilinear only.
    pub stations: [Option<Vec2>; 3],
    /// The **cursor ray** of each axis: the line from that axis's vanishing
    /// point through the point under the pointer, which is where the world line
    /// through that point, parallel to the axis, images (§20.9). Under the
    /// fisheye it is the arc through both poles
    /// ([`PerspectiveGuide::axis_ray`]).
    ///
    /// Follows [`axis_alpha`](Self::axis_alpha), like the vanishing point it
    /// runs from: a ray belongs to an axis alone, being a line in no pair plane.
    /// `None` where the pointer is off the canvas, and for the one pose that has
    /// no ray to name — the hand resting exactly on the vanishing point.
    pub rays: [Option<CursorRay>; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Ellipse;

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

    /// **A rotation that is already one is left exactly alone**, and one that is
    /// not settles in a single pass.
    ///
    /// The rotation arm is the only repair in this crate that is not a clamp, and
    /// the whole "run at mint *and* at state entry without a load becoming a small
    /// edit" bargain rests on [`Quat::normalize`] landing inside
    /// [`Quat::is_normalized`]'s tolerance — so the second pass has nothing left to
    /// do. Bit-for-bit, because a pose nudged by an ulp on every load is still a
    /// document that changes when it is opened.
    #[test]
    fn a_rotation_already_normal_is_untouched_and_a_stretched_one_settles_at_once() {
        let unit = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.3);
        let g = PerspectiveGuide {
            rotation: unit,
            ..Default::default()
        }
        .sanitized();
        assert_eq!(
            g.rotation.to_array(),
            unit.to_array(),
            "a unit quaternion must come through the funnel untouched",
        );

        // Twice as long: a drag's accumulated error past the tolerance.
        let stretched = Quat::from_xyzw(unit.x * 2.0, unit.y * 2.0, unit.z * 2.0, unit.w * 2.0);
        assert!(!stretched.is_normalized(), "the fixture must need repair");
        let once = PerspectiveGuide {
            rotation: stretched,
            ..Default::default()
        }
        .sanitized();
        assert!(once.rotation.is_normalized());
        assert!(
            (once.rotation.dot(unit).abs() - 1.0).abs() < 1e-5,
            "renormalizing must keep the pose, not just the length",
        );
        assert_eq!(
            once.sanitized().rotation.to_array(),
            once.rotation.to_array(),
            "a second pass moved a rotation the first pass repaired",
        );
    }

    /// A quaternion with **no direction left to normalize towards** reads as the
    /// identity pose — looking down the world Z axis, the 1-point case — rather
    /// than as whatever the division amplifies its rounding into.
    #[test]
    fn a_rotation_with_no_direction_left_reads_as_the_identity_pose() {
        let tiny = MIN_QUAT_SQ.sqrt() * 0.5;
        for q in [
            Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            Quat::from_xyzw(tiny, 0.0, 0.0, 0.0),
            Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0),
            Quat::from_xyzw(0.0, f32::INFINITY, 0.0, 0.0),
        ] {
            let g = PerspectiveGuide {
                rotation: q,
                ..Default::default()
            }
            .sanitized();
            assert_eq!(g.rotation, Quat::IDENTITY, "{q:?} named a pose");
        }
    }

    /// The camera's own numbers: the focal length floors where [`ray`] and
    /// [`project`] stop being a projection at all, and the two vectors fall back
    /// to the default pose rather than reaching a uniform lane as a `NaN` (§20.4).
    ///
    /// [`ray`]: PerspectiveGuide::ray
    /// [`project`]: PerspectiveGuide::project
    #[test]
    fn a_camera_floors_its_focal_length_and_falls_back_for_a_place_it_cannot_read() {
        let d = PerspectiveGuide::default();
        for bad in [0.0, -900.0, 0.5, f32::NAN, f32::NEG_INFINITY] {
            let g = PerspectiveGuide {
                focal: bad,
                ..Default::default()
            }
            .sanitized();
            assert!(g.focal >= MIN_FOCAL, "focal {bad} sanitized to {}", g.focal);
        }
        // A finite focal length already past the floor is left where it is.
        assert_eq!(
            PerspectiveGuide {
                focal: 42.0,
                ..Default::default()
            }
            .sanitized()
            .focal,
            42.0,
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let g = PerspectiveGuide {
                center: Vec2::new(120.0, bad),
                lattice: Vec3::new(bad, 4.0, 8.0),
                ..Default::default()
            }
            .sanitized();
            assert_eq!(g.center, d.center, "a center holding {bad} was kept");
            assert_eq!(g.lattice, d.lattice, "a lattice holding {bad} was kept");
        }
    }

    /// [`Lens`] is read by variant **name**, not position (§8).
    ///
    /// Two unit variants, which is the quiet case: there is no payload to arrive
    /// mangled, so a positional read would open every saved rectilinear guide as a
    /// fisheye and every fisheye as a rectilinear, with nothing in the file — and
    /// nothing in the loader — able to say so.
    #[test]
    fn a_lens_is_read_by_variant_name_not_position() {
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "Lens")]
        enum Old {
            Fisheye,
            Rectilinear,
        }

        let read = |old: &Old| {
            carbonite::from_slice_static::<Lens>(&carbonite::to_vec_static(old).expect("encodes"))
                .expect("a declaration order this build does not use still reads")
        };

        assert_eq!(read(&Old::Rectilinear), Lens::Rectilinear);
        assert_eq!(read(&Old::Fisheye), Lens::Fisheye);
    }
}
