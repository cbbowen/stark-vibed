//! Drawing guides — the perspective grid (§20).
//!
//! One projective camera behind "1-point", "2-point" and "3-point" alike: the
//! state is a **center of view** (principal point), a **focal length** and an
//! orientation, and the case is only how many world axes the view direction is
//! parallel to (§20.1). Everything an artist is shown is derived from it — each
//! axis's **vanishing point**, allowed to be at infinity (§20.2); each pair's
//! **vanishing line**; the **lattice** of squared planes the fans measure out
//! (§20.3); the **45° circle** of radius `focal` (§20.1); and the three
//! **station points**, the eye rotated into the picture plane about each
//! vanishing line (§20.2).
//!
//! Projection is through a [`Lens`], so the stereographic **fisheye** (§20.8) is
//! one field: the fans, the orbit drag, the snap and the locks are stated in
//! direction space and carry over untouched, while straight guide lines bow into
//! exact circles and both poles of every axis come into view.
//!
//! A guide is **document state** (§20.5) — an id, saved in the log, replicated
//! and undoable — because a perspective set up over a drawing is part of the
//! drawing's construction rather than an aid for the hand, the way the pan and
//! the zoom are. The one per-client thing is *whether a guide is drawn*, which
//! lives beside them in `stark-engine`'s `Session` and defaults to off. Nothing
//! here knows about it: the derivations below gate on the guide's own controls
//! and never on whether it is on screen.
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
//! [`PerspectiveGuide::scene`] derives the [`GuideScene`] the compositor's guide
//! pass draws (§20.4); [`PerspectiveGuide::dragged`] is the direct manipulation
//! (§20.5); [`PerspectiveGuide::pencils`] and [`PerspectiveGuide::planes`] are
//! what the drawing assist holds a snapped stroke to — the axes a line is aimed
//! along (§20.6) and the planes a loop is a circle on (§20.7), gathered as a
//! [`Scaffold`].
//!
//! **None of it needs a pixel**, which is what puts the derivations here beside
//! the fact rather than in the engine (§2, §20.5). What the engine keeps is the
//! GPU part — packing a [`GuideScene`] into the guide pass's uniform — and the
//! session part, the visibility filter deciding which guides reach
//! [`Scaffold::of`] at all.

use serde::{Deserialize, Serialize};

use glam::{Quat, Vec2, Vec3};

use super::action::ActionId;
use crate::sanitize::{finite_in, finite_or};

mod camera;
mod conic;
#[cfg(test)]
mod fixtures;

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
/// direction images at infinity and the whole overlay is one uniform of `inf`.
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
/// component of ~1.8e19, and `Quat::normalize` then returns `Quat(0, 0, 0, 0)` —
/// finite, so the funnel reports it clean, and not a rotation. Worse, it is not
/// idempotent: a second pass reads that zero as "no direction" and answers
/// `IDENTITY`, so the guide would reload as a different guide (§20.5). A quaternion
/// that large is corruption rather than a pose anyone stated, so it lands at the
/// identity with the too-small ones instead of being rescaled.
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
/// At the vanishing point *every* line of the pencil passes through the pointer,
/// so there is no one line to draw; the epsilon covers the approach too, where
/// the plane normal is the cross product of two nearly parallel unit vectors and
/// its direction dissolves into rounding well before its length reaches zero.
///
/// The band it retires is `RAY_EPS · f / a.z` canvas px around the vanishing
/// point, so it grows with the focal length: a tenth of a pixel at the short end
/// of the panel's range and a little over one at the long end (`FOCAL_RANGE` is
/// 120–12000), always inside the ten-pixel disc drawn over that point.
const RAY_EPS: f32 = 1e-4;

/// How far into a plane its chart reaches, in units of the plane's own distance
/// from the eye (§20.7). Past it a point is treated as not being on the plane at
/// all.
///
/// A horizon rather than a tolerance: plane distance and the image's distance
/// from the vanishing line are reciprocal, so at a thousand eye-heights the rest
/// of the plane images inside about a pixel of the line — and the chart never
/// answers with astronomical coordinates as the divisor goes to zero.
const PLANE_REACH: f32 = 1e3;

/// Below this much normal component along the view axis, a fisheye pair trace
/// is drawn as its limiting straight line rather than as a circle (§20.8).
///
/// Coarser than [`LINE_EPS`] on purpose: the circle's radius is `2f/|m.z|`, and
/// at radii near ten million pixels the f32 distance-to-ring subtraction cancels
/// catastrophically. The line substituted deviates from the true circle by
/// `f·|m.z|` at most, under a pixel at this threshold.
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
    /// scaled to agree with the rectilinear lens at the center of view. Alone
    /// among the fisheye mappings it is conformal and takes circles to circles,
    /// so every guide line is an exact canvas circle. Directions are *not*
    /// projective here — an axis and its negation land at two points — which is
    /// why a 1-point pose shows the classical **5-point** grid (§20.8).
    Fisheye,
}

impl Lens {
    /// The radii of the dressed view-cone rings, per unit focal length:
    /// the 45° ring, and the 90° ring where the lens has one.
    ///
    /// Rectilinear: `tan 45° = 1`, so the 45° ring *is* the focal length and the
    /// 90° cone is at infinity. Fisheye: `2·tan(θ/2)` puts 45° at `2(√2 − 1)`
    /// and the 90° ring — the rim of the forward hemisphere, in which the
    /// classical 5-point grid is drawn — at exactly `2`.
    pub fn ring_factors(self) -> (f32, Option<f32>) {
        match self {
            Lens::Rectilinear => (1.0, None),
            Lens::Fisheye => (2.0 * (std::f32::consts::SQRT_2 - 1.0), Some(2.0)),
        }
    }
}

/// Stable identifier for a drawing guide within a document (§20.5).
///
/// **The id of the action that added it.** An [`ActionId`] is the log's
/// total-order key `(lamport, actor)` and so already globally unique, which makes
/// "no two guides carry the same id" a property of the representation rather than
/// a rule a call site could forget (§1) — there is no counter here and nothing to
/// resync when a log is picked back up.
///
/// One `AddGuide` mints exactly one guide, so unlike
/// [`LayerId`](super::LayerId) there is no *which one* left to say.
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
/// undoable like any other edit. Not here: the guide's name, which the roster
/// carries beside it — the same split
/// [`SetLayerName`](super::ActionKind::SetLayerName) draws — and whether it is
/// drawn, which is per-client and lives in the session.
///
/// `Copy`, and worth keeping so: a drag rebuilds one of these per pointer sample
/// from the pose the press started in, and every derivation below takes `&self`
/// and answers with something new.
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
    /// and "1/2/3-point" names nothing but the count of finite vanishing points.
    pub rotation: Quat,
    /// The lens the guide projects through (§20.8): classical straight-line
    /// perspective, or the curvilinear fisheye. Everything else about the
    /// camera — orientation, center, focal — means the same thing under both,
    /// which is what makes this one field a *toggle* rather than a mode.
    pub lens: Lens,
    /// The **lattice** the fans measure out (§20.3): how far each of the three
    /// squared planes lies from the eye, in *cells*, one component per world
    /// axis — equivalently the corner they meet at, as a displacement from the
    /// eye in world-axis coordinates.
    ///
    /// **The grid's phase is the eye, not the corner** (§20.3), so nothing about
    /// this vector needs to be round for the lines to be the grid's own.
    ///
    /// A zero component puts the eye *inside* the plane normal to that axis,
    /// whose grid then images onto its own vanishing line and has nothing to
    /// show; the pass declines it. A zero *vector* is that for all three at once,
    /// and [`corner`](Self::corner) declines it at the source.
    pub lattice: Vec3,
    /// Master opacity of the whole overlay.
    pub opacity: f32,
    /// Which of the three **pair planes** are drawn — pair `k` being the plane
    /// axes `k` and `k + 1` span, so the three are XY, YZ and ZX (§20.3), the
    /// names the bar's chips wear.
    ///
    /// Planes rather than axes because a guide line is a line *in* a plane: three
    /// axis toggles could only reach none, one plane or all three, and "the ground
    /// and the near wall but not the far one" would not be sayable.
    ///
    /// Everything a plane carries follows its flag: the two fans that rule it,
    /// its vanishing trace (and with it the [`horizon`](Self::horizons) that
    /// turns about its normal), and its station point. An **axis** is drawn
    /// when either plane it is a side of is — that is what its vanishing point
    /// and its stroke [`pencil`](Self::pencils) follow, since an axis with no
    /// plane left draws no line anywhere.
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
    /// three planes to have a grid to show at all.
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
    /// Every field here reaches a shader (§20.4), so a `NaN` in any of them is
    /// not a wrong picture but no picture — and it would be *saved*.
    ///
    /// **Idempotent**, which is what lets it run both at mint and on the way into
    /// state without a load becoming a small edit: every step is a clamp or a
    /// replacement, and a rotation already inside [`Quat::is_normalized`]'s
    /// tolerance is left exactly alone.
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
            // Left untouched once it is a rotation, so this is idempotent. One
            // with no direction left to normalize towards looks down the world Z
            // axis, the honest answer to "no rotation was stated".
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
/// [`PerspectiveGuide`]: what it needs is a list of geometry, not a list of
/// guides, and nothing about a name, an eye or an opacity survives the crossing.
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
    /// a guide's eye is per-client (§20.5), so the caller — the engine, the only
    /// side holding both halves — drops the ones it has shut before handing them
    /// over. [`pencils`](PerspectiveGuide::pencils) and
    /// [`planes`](PerspectiveGuide::planes) gate on the document's own controls
    /// and know nothing about an eye.
    ///
    /// Takes an iterator rather than a slice so that filtered roster need not be
    /// collected; both lists are filled in one pass, so it need not be `Clone`.
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
    /// Separate from [`axis_alpha`](Self::axis_alpha) because switching one plane
    /// off leaves both of its axes on the screen, still ruling the other planes
    /// they border; only losing *both* of an axis's planes darkens the axis.
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
    /// Follows [`axis_alpha`](Self::axis_alpha), like the vanishing point it runs
    /// from: a ray belongs to an axis alone, being a line in no pair plane.
    /// `None` where the pointer is off the canvas, and where the hand rests
    /// exactly on the vanishing point.
    pub rays: [Option<CursorRay>; 3],
}

#[cfg(test)]
mod tests {
    use super::fixtures::fisheye;
    use super::*;

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
    /// The "run at mint *and* at state entry without a load becoming a small edit"
    /// bargain rests on [`Quat::normalize`] landing inside
    /// [`Quat::is_normalized`]'s tolerance. Bit-for-bit, because a pose nudged by
    /// an ulp on every load is still a document that changes when it is opened.
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
    /// Two unit variants is the quiet case: there is no payload to arrive mangled,
    /// so a positional read would silently open every saved rectilinear guide as a
    /// fisheye and every fisheye as a rectilinear.
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
