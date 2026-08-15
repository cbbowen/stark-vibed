//! Canvas geometry: tile coordinates and the pan/zoom view transform.
//!
//! Canvas space is in pixels with x to the right and y downward. Tile `(i, j)`
//! covers the square `[i*TILE_SIZE, (i+1)*TILE_SIZE) × [j*TILE_SIZE, ...)`.
//! The infinite canvas (§6) is realized by tiles being sparse and
//! addressed by signed integer coordinates.

use std::f32::consts::{FRAC_PI_2, TAU};

pub use glam::{Affine2, Mat2, Vec2};

/// Eigenvalues of the symmetric 2×2 `[[sxx, sxy], [sxy, syy]]`, larger first, with the
/// unit eigenvector of the larger — in closed form, since a 2×2 needs no iteration.
///
/// The eigenvector is read off whichever column of `M − λ₂I` is longer: both span the
/// same line, and taking the longer is what keeps it defined when the matrix is nearly
/// isotropic (a circle, where the axes are genuinely arbitrary but must still be
/// *some* orthogonal pair).
///
/// Here rather than beside either caller because both a scatter of samples and a conic
/// are the same 2×2 question: [`assist`](crate::assist) reads an ellipse off the second
/// moments of a trace, and [`guides`](crate::guides) reads one off the quadratic part
/// of a conic (§20.7).
pub(crate) fn principal_axis(sxx: f32, sxy: f32, syy: f32) -> (f32, f32, Option<Vec2>) {
    let half_trace = 0.5 * (sxx + syy);
    let disc = (0.25 * (sxx - syy).powi(2) + sxy * sxy).max(0.0).sqrt();
    let (major, minor) = (half_trace + disc, half_trace - disc);
    let c0 = Vec2::new(sxx - minor, sxy);
    let c1 = Vec2::new(sxy, syy - minor);
    let v = if c0.length_squared() >= c1.length_squared() {
        c0
    } else {
        c1
    };
    (major, minor, v.try_normalize().or(Some(Vec2::X)))
}

/// An axis-aligned-in-its-own-frame ellipse: where it is, how big, and how it is
/// turned.
///
/// Here beside [`principal_axis`] and for the same reason. Two modules arrive at an
/// ellipse from opposite directions — [`assist`](crate::assist) reads one off the
/// second moments of a hand-drawn loop, [`guides`](crate::guides) off the quadratic
/// part of a conic (§20.7) — and both were passing it around as a bare
/// `(Vec2, Vec2, f32)`, which says nothing about which `Vec2` is which and left the
/// convergence test in `assist` comparing `a.0` against `b.1`.
///
/// `radii` is **major first**, which the triple could not state and every producer
/// had to promise in prose.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ellipse {
    pub center: Vec2,
    /// Semi-axes, major first, in the frame `angle` turns to.
    pub radii: Vec2,
    /// How far the major axis is turned from +x, in radians.
    pub angle: f32,
}

impl Ellipse {
    pub fn new(center: Vec2, radii: Vec2, angle: f32) -> Self {
        Self {
            center,
            radii,
            angle,
        }
    }

    /// The mean semi-axis — the ellipse's own scale, which is what a tolerance on one
    /// has to be measured against if it is to mean the same thing at any size.
    pub fn scale(&self) -> f32 {
        0.5 * (self.radii.x + self.radii.y)
    }
}

/// Apron (halo) width in pixels carried around each tile's interior, replicated
/// from the neighboring canvas content (§6.4). The compositor samples a
/// tile's interior with bilinear filtering; without an apron the filter clamps at
/// the tile edge instead of reaching into the neighbor, leaving a visible seam at
/// every boundary under sub-pixel pan or non-1:1 zoom (the seam is then amplified
/// by the media pass's height→normal gradient). One pixel is all bilinear needs;
/// widen this if a future media effect needs more neighbor context.
pub const TILE_APRON: u32 = 1;

/// Physical edge length of a tile's channel textures: interior plus an apron on
/// every side. Tiles are stored at this size; only the interior is presented.
pub const TILE_TEX: u32 = 256;

/// Edge length of a square tile's *interior*, in canvas pixels (§6.1).
/// This is the addressing stride: tile `(i, j)` owns canvas
/// `[i*TILE_SIZE, (i+1)*TILE_SIZE)` — aprons (below) overlap neighbors and are
/// not owned.
pub const TILE_SIZE: u32 = TILE_TEX - 2 * TILE_APRON;

/// Maps a tile's interior quad corner (`∈ [0, 1]`) to a UV coordinate in the
/// apron'd texture: `uv = corner * INTERIOR_UV_SCALE + INTERIOR_UV_BIAS`. The
/// compositor and presenter sample only the interior sub-rect; bilinear taps at
/// the interior edge then fall into the apron (neighbor content), not a clamp.
pub const INTERIOR_UV_SCALE: f32 = TILE_SIZE as f32 / TILE_TEX as f32;
pub const INTERIOR_UV_BIAS: f32 = TILE_APRON as f32 / TILE_TEX as f32;

/// Integer address of a tile on the infinite canvas.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
}

impl TileCoord {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Canvas-space position of this tile's top-left corner, in pixels.
    pub fn origin(self) -> Vec2 {
        Vec2::new(
            self.x as f32 * TILE_SIZE as f32,
            self.y as f32 * TILE_SIZE as f32,
        )
    }
}

/// An inclusive tile-coordinate rectangle. `min > max` on either axis is the
/// empty rect ([`EMPTY`](Self::EMPTY)), which overlaps nothing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileRect {
    pub min: (i32, i32),
    pub max: (i32, i32),
}

impl TileRect {
    /// The whole infinite canvas — what an action claims when it cannot say
    /// anything narrower.
    pub const ALL: TileRect = TileRect {
        min: (i32::MIN, i32::MIN),
        max: (i32::MAX, i32::MAX),
    };

    /// The rect that reaches nothing.
    pub const EMPTY: TileRect = TileRect {
        min: (1, 1),
        max: (0, 0),
    };

    pub fn intersects(&self, other: &TileRect) -> bool {
        self.min.0 <= self.max.0
            && other.min.0 <= other.max.0
            && self.min.0 <= other.max.0
            && other.min.0 <= self.max.0
            && self.min.1 <= other.max.1
            && other.min.1 <= self.max.1
    }

    pub fn contains(&self, c: TileCoord) -> bool {
        self.min.0 <= c.x && c.x <= self.max.0 && self.min.1 <= c.y && c.y <= self.max.1
    }

    /// The smallest rect holding both — what a batch of draws culls against when it
    /// wants one draw list rather than one per member (the eyedropper's trace,
    /// §18.0.2).
    ///
    /// Widening is the safe direction: a cull may name tiles a pass then draws
    /// nothing for, and may never omit one it needed. [`EMPTY`](Self::EMPTY) is
    /// inverted (`min > max`) so that it is the identity here rather than a corner
    /// the union has to stretch to reach.
    pub fn union(self, other: TileRect) -> TileRect {
        if self.min.0 > self.max.0 || self.min.1 > self.max.1 {
            return other;
        }
        if other.min.0 > other.max.0 || other.min.1 > other.max.1 {
            return self;
        }
        TileRect {
            min: (self.min.0.min(other.min.0), self.min.1.min(other.min.1)),
            max: (self.max.0.max(other.max.0), self.max.1.max(other.max.1)),
        }
    }

    /// The tiles the canvas box `[lo, hi]` reaches, grown by `ring` tiles on
    /// every side.
    ///
    /// **The one quantizer**, and the reason it is one: this arithmetic was
    /// written out five times across `document/`, and the copies disagreed in
    /// exactly the place that matters. `NaN as i32` is 0, so a `clamp`-then-cast
    /// version answered "one tile at the origin" for a box it could not measure;
    /// a bare `as i32` on an out-of-range index wrapped it to a tile somewhere
    /// else entirely. Both are silent, and both point the unsafe way for the two
    /// things that ask: a footprint that under-claims diverges peers (§12.6), and
    /// a tile cover that under-counts is enumerated rather than refused.
    ///
    /// So the arithmetic is `i64` and saturating throughout, and the answer is
    /// `None` for a box that is not finite or falls outside the grid an `i32`
    /// tile index can address (past ~5×10¹¹ canvas px). What to *do* about that
    /// differs by caller — claim everything, or refuse — which is why this
    /// returns the question rather than picking one.
    ///
    /// Callers pad `lo`/`hi` themselves for whatever their pass reads past its
    /// own geometry: a tip's radius, the apron band, a coverage ramp. `ring` is
    /// for whole tiles rewritten around what is drawn.
    pub fn covering(lo: Vec2, hi: Vec2, ring: i32) -> Option<TileRect> {
        if !(lo.is_finite() && hi.is_finite()) {
            return None;
        }
        let ring = i64::from(ring);
        let index = |v: f32| ((v / TILE_SIZE as f32).floor()) as i64;
        let min = |v: f32| i32::try_from(index(v).saturating_sub(ring)).ok();
        let max = |v: f32| i32::try_from(index(v).saturating_add(ring)).ok();
        Some(TileRect {
            min: (min(lo.x)?, min(lo.y)?),
            max: (max(hi.x)?, max(hi.y)?),
        })
    }

    /// How many tiles this covers — saturating, so [`ALL`](Self::ALL) reports
    /// more than any budget will allow rather than wrapping to a small number.
    ///
    /// Exists to be asked **before** [`coords`](Self::coords) is walked: the box
    /// is quadratic in whatever produced it, so a drag at far zoom-out can name
    /// more tiles than there is memory to list, and finding that out by listing
    /// them is not an option.
    pub fn count(self) -> u64 {
        let span = |a: i32, b: i32| (i64::from(b) - i64::from(a) + 1).max(0) as u64;
        span(self.min.0, self.max.0).saturating_mul(span(self.min.1, self.max.1))
    }

    /// Every tile in the rect, row-major. Empty for an empty rect.
    pub fn coords(self) -> impl Iterator<Item = TileCoord> {
        (self.min.1..=self.max.1)
            .flat_map(move |y| (self.min.0..=self.max.0).map(move |x| TileCoord::new(x, y)))
    }
}

/// A pixel size (e.g. a render target's dimensions).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Extent2 {
    pub width: u32,
    pub height: u32,
}

impl Extent2 {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// The pan/zoom/rotate/mirror transform applied when presenting the canvas to a
/// surface (§6.4). This is session state and is never historized.
///
/// Everything here is *how you are looking at* the painting rather than anything
/// about it, which is what makes turning the canvas and holding it up to a mirror
/// the same kind of act as panning: per-client, never logged, never sent, and
/// invisible to replay (§18.1.2). Two people sharing a drawing can
/// have it at different angles.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ViewTransform {
    /// Canvas-space point shown at the center of the viewport.
    pub center: Vec2,
    /// Zoom factor: screen pixels per canvas pixel (1.0 = 1:1).
    pub zoom: f32,
    /// How far the canvas is turned on screen, in radians, positive **clockwise**
    /// (the sense of a canvas rotated on an easel, in a y-down frame).
    ///
    /// Applied after [`flip_h`](Self::flip_h), so the mirror is about the canvas's
    /// own vertical axis whatever angle it is being viewed at — which is what makes
    /// "flip it to check the drawing" mean the same thing at every rotation.
    pub rotation: f32,
    /// Whether the canvas is mirrored left↔right on screen — the oldest trick for
    /// catching a drawing error, since the eye stops recognising what it expected to
    /// see and starts seeing what is there.
    ///
    /// Only the horizontal one, and that is not a limitation: a vertical flip is this
    /// one composed with a half turn, and the pair "mirror + angle" says which of
    /// those two the artist meant. It also keeps the mirror a *toggle* rather than
    /// two independent booleans that can both be on.
    pub flip_h: bool,
    /// Size of the target surface, in pixels.
    pub viewport: Extent2,
}

impl ViewTransform {
    /// A 1:1 upright view centered on the canvas origin for a viewport of `size`.
    pub fn identity(size: Extent2) -> Self {
        Self {
            center: Vec2::ZERO,
            zoom: 1.0,
            rotation: 0.0,
            flip_h: false,
            viewport: size,
        }
    }

    /// The same view rendered at `n` samples per axis: the viewport and the zoom
    /// scaled together, everything else left alone (§6.4).
    ///
    /// Scaling *both* is what makes supersampling invisible to every pass. The
    /// canvas→NDC map comes out identical (the zoom and the viewport divide out), so
    /// the same canvas pixel lands at the same place in the picture; and anything a
    /// shader measures in target px — an outline's width, a matte's edge fade, a
    /// guide's line — is measured against the scaled zoom, so it comes out `n` times
    /// wider in a picture that is about to be `n` times smaller. Which is the same
    /// width, drawn with `n²` samples of coverage instead of one.
    pub fn supersampled(self, n: u32) -> Self {
        Self {
            zoom: self.zoom * n as f32,
            viewport: Extent2::new(self.viewport.width * n, self.viewport.height * n),
            ..self
        }
    }

    /// The canvas→screen linear map with the zoom divided out: the mirror, then the
    /// rotation. Orthogonal (`|det| = 1`), so its transpose is its inverse — which is
    /// the whole reason the view keeps an angle and a flag rather than a free matrix
    /// that would have to be inverted, and could drift away from being a rigid motion.
    ///
    /// Frontend chrome that draws a canvas-space shape over the canvas hands exactly
    /// this to CSS: a rotated frame, or the transform widget's ellipse, is its
    /// canvas-space shape composed with this.
    pub fn orientation(self) -> Mat2 {
        Mat2::from_angle(self.rotation) * self.mirror()
    }

    /// The mirror alone, without the turn.
    fn mirror(self) -> Mat2 {
        if self.flip_h {
            Mat2::from_cols(Vec2::new(-1.0, 0.0), Vec2::Y)
        } else {
            Mat2::IDENTITY
        }
    }

    /// The full canvas→screen linear map: [`orientation`](Self::orientation) scaled
    /// by the zoom.
    pub fn linear(self) -> Mat2 {
        self.orientation() * self.zoom
    }

    /// The screen→canvas linear map. The transpose rather than a general inverse,
    /// because [`orientation`](Self::orientation) is orthogonal.
    pub fn inverse_linear(self) -> Mat2 {
        self.orientation().transpose() * (1.0 / self.zoom)
    }

    /// A screen-pixel *delta* as a canvas-space delta — a drag, in other words, which
    /// is a direction rather than a place and so does not go through the centre.
    pub fn canvas_delta(self, screen: Vec2) -> Vec2 {
        self.inverse_linear() * screen
    }

    /// Linear map from canvas pixels to normalized device coordinates:
    /// `ndc = m * canvas_pos + translate`. Derivation in §6.4.
    ///
    /// The y row is negated because canvas y is downward while NDC y is upward. An
    /// upright, unmirrored view leaves `m` diagonal and this is exactly the scale it
    /// always was.
    pub fn canvas_to_ndc(self) -> (Mat2, Vec2) {
        let vw = self.viewport.width.max(1) as f32;
        let vh = self.viewport.height.max(1) as f32;
        // The screen→NDC scale with the zoom already folded in, so an upright view
        // computes bit-for-bit what it did before rotation existed.
        let d = Vec2::new(2.0 * self.zoom / vw, -2.0 * self.zoom / vh);
        let o = self.orientation();
        // Scaling the *rows* of `o` by `d`, which in column-major is each column
        // multiplied componentwise.
        let m = Mat2::from_cols(o.x_axis * d, o.y_axis * d);
        (m, -(m * self.center))
    }

    /// Inverse of the view: map a screen-pixel position (origin top-left) to a
    /// canvas-space point. Used to turn pointer input into stroke samples.
    pub fn screen_to_canvas(self, screen: Vec2) -> Vec2 {
        self.canvas_delta(screen - self.half()) + self.center
    }

    /// Forward of [`screen_to_canvas`](Self::screen_to_canvas): a canvas-space
    /// point to its screen-pixel position (origin top-left). Used by frontend
    /// chrome that has to sit over a canvas-space feature — the frame's handles
    /// (§15.7).
    pub fn canvas_to_screen(self, canvas: Vec2) -> Vec2 {
        self.linear() * (canvas - self.center) + self.half()
    }

    /// The canvas-space **bounding box** of everything the viewport shows.
    ///
    /// A bound rather than the region itself, because under rotation the region is a
    /// turned rectangle and its callers — framing what you are looking at, falling
    /// back to "the visible canvas" when nothing is painted — all want a rect. It
    /// therefore covers a little more than is really on screen at an angle, which is
    /// the safe direction for both.
    pub fn visible_bounds(self) -> (Vec2, Vec2) {
        let half = self.half();
        let (a, b) = (
            self.canvas_delta(half),
            self.canvas_delta(Vec2::new(half.x, -half.y)),
        );
        // The four corners are ±a and ±b, so the bound is the componentwise larger.
        let extent = a.abs().max(b.abs());
        (self.center - extent, self.center + extent)
    }

    /// Half the viewport, in screen px — the centre of the surface.
    fn half(self) -> Vec2 {
        Vec2::new(self.viewport.width as f32, self.viewport.height as f32) * 0.5
    }

    /// The rotation that would put the canvas-space direction `up` at the top of the
    /// screen. `None` for a direction with no length to speak of, which asks for
    /// nothing.
    ///
    /// A question rather than an instruction, and stated as a direction rather than an
    /// angle, because that is what a drag gives — and because an angle would have to
    /// be measured against a zero the mirror keeps moving. What the frontend does with
    /// the answer is the frontend's: the navigator eases toward it and snaps it to a
    /// quarter turn, both of which are properties of dragging with a hand rather than
    /// of the view (see `panels::navigator`).
    pub fn rotation_for_up(self, up: Vec2) -> Option<f32> {
        // The mirror is applied before the turn, so where `up` *would* fall is
        // independent of the rotation this is asking about.
        let shown = self.mirror() * up;
        if shown.length_squared() < 1e-12 {
            return None;
        }
        // A vector's angle grows with `rotation`, and screen up is −y (angle −π/2),
        // so the turn wanted is the difference of the two.
        Some((-FRAC_PI_2 - shown.to_angle()).rem_euclid(TAU))
    }

    /// Turn the canvas to `radians`, normalised to one revolution so the value stays
    /// bounded however many turns it is given.
    pub fn set_rotation(&mut self, radians: f32) {
        self.rotation = radians.rem_euclid(TAU);
    }

    /// Mirror what is **on screen**, left↔right — the flip an artist means when they
    /// hold the drawing up to a mirror to check it.
    ///
    /// Screen-relative, not canvas-relative, which is the whole point: at any angle it
    /// swaps what is on the left of the screen with what is on the right, so the check
    /// means the same thing however the easel is turned. Mirroring about the canvas's
    /// own axis instead would, on a canvas turned a quarter, swap top for bottom —
    /// technically a flip, and not the one anyone asked for.
    ///
    /// Reflecting the *result* keeps the view a rotation-and-a-mirror rather than
    /// becoming a free matrix, because a reflection can be pushed back through a
    /// rotation: `M·R(θ) = R(−θ)·M`. So the whole operation is to negate the angle and
    /// toggle the mirror — and doing it twice is exactly the identity, which is what
    /// makes it a toggle rather than a setting.
    pub fn mirror_screen_h(&mut self) {
        self.set_rotation(-self.rotation);
        self.flip_h = !self.flip_h;
    }

    /// Scale the zoom by `factor` while keeping the canvas point under `anchor`
    /// (a screen-pixel position) pinned in place — cursor-anchored zoom.
    pub fn zoom_about(&mut self, anchor: Vec2, factor: f32) {
        self.pinch(anchor, anchor, factor, 0.0)
    }

    /// Move, scale and turn the view **in one act**, about a screen-pixel point: the
    /// canvas point under `anchor` ends up under `to`, having been scaled by `scale`
    /// and turned by `turn` radians clockwise about it. The two-finger gesture
    /// (§18.1.7), and — with `to == anchor` and no turn — the wheel's zoom.
    ///
    /// One command rather than a pan, a zoom and a turn, because a pinch is one motion
    /// of one pair of fingers and the three are not independent: each of the three
    /// anchors against the view it is applied to, so sending them in sequence would
    /// have the second and third re-anchor against a view the hand never saw, and the
    /// point being held would slide out from under it. Composed here, what the fingers
    /// hold is held exactly.
    ///
    /// The mirror is left alone and the turn adds straight onto the angle, because the
    /// gesture is stated in **screen** terms: a twist clockwise on the glass is a twist
    /// clockwise on the screen at any angle and either handedness — the same
    /// screen-relative sense [`mirror_screen_h`](Self::mirror_screen_h) is defined in.
    /// (`R(δ)·R(θ)·M = R(θ+δ)·M`, so it stays a rotation-and-a-mirror.)
    pub fn pinch(&mut self, anchor: Vec2, to: Vec2, scale: f32, turn: f32) {
        // The canvas point the gesture is holding, read through the view as it stands.
        let held = self.screen_to_canvas(anchor);
        self.set_rotation(self.rotation + turn);
        self.zoom = (self.zoom * scale).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        // ...and put back under the hand, through the view as it now is. Solving
        // `screen_to_canvas(to) == held` for the centre, which is the one degree of
        // freedom left once the angle and the zoom are set.
        self.center = held - self.canvas_delta(to - self.half());
    }

    const MIN_ZOOM: f32 = 0.05;
    const MAX_ZOOM: f32 = 64.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `covering` is the one quantizer five call sites across `document/` now
    /// share, so the box it names has to be exactly the tiles the canvas box
    /// reaches — no more (a fill would rewrite tiles it never covered) and no
    /// fewer (a footprint would under-claim, which diverges peers).
    #[test]
    fn covering_names_exactly_the_tiles_a_box_reaches() {
        let side = TILE_SIZE as f32;
        // Wholly inside tile (0, 0).
        let one = TileRect::covering(Vec2::splat(4.0), Vec2::splat(9.0), 0).unwrap();
        assert_eq!((one.min, one.max), ((0, 0), (0, 0)));
        assert_eq!(one.count(), 1);
        assert_eq!(one.coords().collect::<Vec<_>>(), vec![TileCoord::new(0, 0)]);

        // Reaching one texel into the next tile on both axes.
        let two = TileRect::covering(Vec2::splat(4.0), Vec2::splat(side), 0).unwrap();
        assert_eq!((two.min, two.max), ((0, 0), (1, 1)));
        assert_eq!(two.count(), 4);

        // The ring grows it a whole tile on every side.
        let ringed = TileRect::covering(Vec2::splat(4.0), Vec2::splat(9.0), 1).unwrap();
        assert_eq!((ringed.min, ringed.max), ((-1, -1), (1, 1)));
        assert_eq!(ringed.count(), 9);

        // Negative coordinates floor rather than truncate toward zero — the tile
        // left of the origin is -1, and a `as i32` cast would have said 0.
        let left = TileRect::covering(Vec2::splat(-1.0), Vec2::splat(-1.0), 0).unwrap();
        assert_eq!((left.min, left.max), ((-1, -1), (-1, -1)));
    }

    /// A union may only ever **widen**: it is what a batch of draws culls against,
    /// and a cull that named fewer tiles than one of its members needed would drop
    /// paint from that member's pass.
    #[test]
    fn a_union_holds_everything_both_rects_did() {
        let a = TileRect::covering(Vec2::splat(0.0), Vec2::splat(1.0), 0).unwrap();
        let b = TileRect::covering(Vec2::splat(-300.0), Vec2::splat(-299.0), 0).unwrap();
        let u = a.union(b);
        for r in [a, b] {
            for c in r.coords() {
                assert!(u.contains(c), "{c:?} was in a member and not in the union");
            }
        }
        assert_eq!(u, b.union(a), "the union does not depend on the order");

        // `EMPTY` is inverted rather than a corner, so it is the identity here — a
        // rect that reaches nothing must not stretch a union to the origin.
        assert_eq!(TileRect::EMPTY.union(a), a);
        assert_eq!(a.union(TileRect::EMPTY), a);
        assert_eq!(TileRect::EMPTY.union(TileRect::EMPTY), TileRect::EMPTY);
    }

    /// The two answers that must be refused rather than given silently and wrongly:
    /// a box that cannot be measured, and one that cannot be addressed.
    #[test]
    fn covering_refuses_what_it_cannot_measure_or_address() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(TileRect::covering(Vec2::new(bad, 0.0), Vec2::ZERO, 0), None);
            assert_eq!(TileRect::covering(Vec2::ZERO, Vec2::new(0.0, bad), 0), None);
        }
        // Small box, absurd position: the count would pass and only the tile
        // index is impossible, which is exactly where a bare cast wrapped.
        let far = 1.0e30;
        assert_eq!(
            TileRect::covering(Vec2::splat(far), Vec2::splat(far + 1.0), 0),
            None
        );
    }

    /// `count` is asked *before* `coords` is walked, so it must not wrap: the
    /// whole plane has to report more than any budget will allow.
    #[test]
    fn the_whole_plane_counts_as_more_than_anyone_will_allow() {
        assert_eq!(TileRect::ALL.count(), u64::MAX);
        assert_eq!(TileRect::EMPTY.count(), 0);
        assert_eq!(TileRect::EMPTY.coords().count(), 0);
        assert!(TileRect::ALL.contains(TileCoord::new(0, 0)));
        assert!(!TileRect::EMPTY.contains(TileCoord::new(0, 0)));
        assert!(!TileRect::EMPTY.intersects(&TileRect::ALL));
    }

    /// An upright, unmirrored view of `size` at `zoom`, centred on `center`.
    fn view(center: Vec2, zoom: f32, size: Extent2) -> ViewTransform {
        ViewTransform {
            center,
            zoom,
            ..ViewTransform::identity(size)
        }
    }

    #[test]
    fn viewport_center_maps_to_view_center() {
        let view = view(Vec2::new(123.0, -45.0), 2.0, Extent2::new(800, 600));
        let center_px = Vec2::new(400.0, 300.0);
        let mapped = view.screen_to_canvas(center_px);
        assert!((mapped - view.center).length() < 1e-3, "got {mapped:?}");
    }

    #[test]
    fn zoom_scales_screen_to_canvas_distance() {
        let view = view(Vec2::ZERO, 2.0, Extent2::new(800, 600));
        // 100 screen px right of center is 50 canvas px at 2x zoom.
        let p = view.screen_to_canvas(Vec2::new(500.0, 300.0));
        assert!((p - Vec2::new(50.0, 0.0)).length() < 1e-3, "got {p:?}");
    }

    #[test]
    fn zoom_about_keeps_cursor_point_fixed() {
        let mut view = view(Vec2::new(10.0, 20.0), 1.0, Extent2::new(800, 600));
        let anchor = Vec2::new(620.0, 130.0); // arbitrary off-center cursor
        let canvas_under = view.screen_to_canvas(anchor);
        view.zoom_about(anchor, 2.5);
        let after = view.screen_to_canvas(anchor);
        assert!(
            (after - canvas_under).length() < 1e-3,
            "anchor drifted: {canvas_under:?} -> {after:?}"
        );
        assert!((view.zoom - 2.5).abs() < 1e-4);
    }

    /// What the two-finger gesture promises (§18.1.7): the canvas stays stuck to the
    /// fingers. Stated as the property rather than as a formula, because the property
    /// is the whole reason the pan, the zoom and the turn are one command — and it is
    /// what a caller composing them by hand would silently lose.
    #[test]
    fn a_pinch_leaves_the_canvas_under_both_fingers() {
        // Fingers before and after, in screen px, arbitrarily placed and moved.
        let from = [Vec2::new(240.0, 500.0), Vec2::new(560.0, 260.0)];
        let to = [Vec2::new(310.0, 380.0), Vec2::new(500.0, 120.0)];
        for flip_h in [false, true] {
            for turn in [0.0, 0.4, FRAC_PI_2, 2.5, TAU - 0.1] {
                let mut v = ViewTransform {
                    center: Vec2::new(-30.0, 90.0),
                    zoom: 1.75,
                    rotation: turn,
                    flip_h,
                    viewport: Extent2::new(800, 600),
                };
                let held = from.map(|f| v.screen_to_canvas(f));
                // Exactly what the frontend measures off the pair: the midpoint it
                // travelled between, how much further apart the fingers ended up, and
                // how far the line between them swung.
                let (u, w) = (from[1] - from[0], to[1] - to[0]);
                v.pinch(
                    0.5 * (from[0] + from[1]),
                    0.5 * (to[0] + to[1]),
                    w.length() / u.length(),
                    u.angle_to(w),
                );
                for (canvas, finger) in held.into_iter().zip(to) {
                    let now = v.canvas_to_screen(canvas);
                    assert!(
                        (now - finger).length() < 1e-2,
                        "flip {flip_h}, turn {turn}: {canvas:?} should be under \
                         {finger:?}, is under {now:?}"
                    );
                }
            }
        }
    }

    /// A pinch broken into steps has to land where the whole one would — the fingers
    /// report one at a time, so every gesture *is* a sequence of steps, and a
    /// composition that drifted would leave the canvas sliding under a hand that had
    /// come back to where it started.
    #[test]
    fn a_pinch_taken_in_steps_lands_where_one_step_would() {
        let path = [
            [Vec2::new(200.0, 400.0), Vec2::new(500.0, 300.0)],
            [Vec2::new(220.0, 380.0), Vec2::new(540.0, 260.0)],
            [Vec2::new(260.0, 300.0), Vec2::new(600.0, 240.0)],
            [Vec2::new(200.0, 400.0), Vec2::new(500.0, 300.0)], // back to the start
        ];
        let start = ViewTransform {
            center: Vec2::new(12.0, -7.0),
            zoom: 0.8,
            rotation: 1.1,
            flip_h: true,
            viewport: Extent2::new(800, 600),
        };
        let mut v = start;
        for pair in path.windows(2) {
            let (u, w) = (pair[0][1] - pair[0][0], pair[1][1] - pair[1][0]);
            v.pinch(
                0.5 * (pair[0][0] + pair[0][1]),
                0.5 * (pair[1][0] + pair[1][1]),
                w.length() / u.length(),
                u.angle_to(w),
            );
        }
        // The fingers ended where they began, so the view must have too.
        assert!(
            (v.center - start.center).length() < 1e-2,
            "centre {:?}",
            v.center
        );
        assert!((v.zoom - start.zoom).abs() < 1e-4, "zoom {}", v.zoom);
        assert!(
            (v.rotation - start.rotation).abs() < 1e-4,
            "rotation {}",
            v.rotation
        );
    }

    /// The turn and the mirror are still a *view*: screen→canvas and canvas→screen
    /// have to remain each other's inverse, or a stroke would land somewhere other
    /// than under the pen (§18.1.2).
    #[test]
    fn the_two_directions_invert_each_other_at_any_orientation() {
        let probes = [
            Vec2::new(0.0, 0.0),
            Vec2::new(37.0, -12.5),
            Vec2::new(-400.0, 250.0),
        ];
        for flip_h in [false, true] {
            for turn in [0.0, 0.4, FRAC_PI_2, 2.5, TAU - 0.1] {
                let v = ViewTransform {
                    center: Vec2::new(-30.0, 90.0),
                    zoom: 1.75,
                    rotation: turn,
                    flip_h,
                    viewport: Extent2::new(800, 600),
                };
                for p in probes {
                    let round = v.screen_to_canvas(v.canvas_to_screen(p));
                    assert!(
                        (round - p).length() < 1e-2,
                        "flip {flip_h}, turn {turn}: {p:?} -> {round:?}"
                    );
                }
                // And the NDC map agrees with the screen map, since the shaders use
                // one and every chrome overlay uses the other.
                let (m, t) = v.canvas_to_ndc();
                for p in probes {
                    let s = v.canvas_to_screen(p);
                    let ndc = m * p + t;
                    let from_screen = Vec2::new(
                        2.0 * s.x / v.viewport.width as f32 - 1.0,
                        1.0 - 2.0 * s.y / v.viewport.height as f32,
                    );
                    assert!(
                        (ndc - from_screen).length() < 1e-4,
                        "flip {flip_h}, turn {turn}: ndc {ndc:?} vs {from_screen:?}"
                    );
                }
            }
        }
    }

    /// An upright, unmirrored view has to go through the matrix path *bit for bit* as
    /// it would through a plain scale-and-offset — the golden images are blessed
    /// against it, and a one-ulp drift in the raster transform is exactly the kind of
    /// change that shows up as a mysterious failure a year later.
    #[test]
    fn an_upright_view_maps_exactly_as_a_plain_scale_does() {
        let v = view(Vec2::new(123.5, -45.25), 2.5, Extent2::new(801, 597));
        let (m, t) = v.canvas_to_ndc();
        let vw = v.viewport.width as f32;
        let vh = v.viewport.height as f32;
        let scale = Vec2::new(2.0 * v.zoom / vw, -2.0 * v.zoom / vh);
        assert_eq!(m.x_axis, Vec2::new(scale.x, 0.0));
        assert_eq!(m.y_axis, Vec2::new(0.0, scale.y));
        assert_eq!(t, -v.center * scale);
    }

    /// The gesture: the direction dragged becomes the direction that points up.
    #[test]
    fn the_rotation_for_a_direction_puts_it_at_the_top_of_the_screen() {
        for flip_h in [false, true] {
            for up in [
                Vec2::new(0.0, -1.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(-3.0, 4.0),
                Vec2::new(0.7, 0.7),
            ] {
                let mut v = ViewTransform {
                    flip_h,
                    ..ViewTransform::identity(Extent2::new(400, 300))
                };
                v.set_rotation(v.rotation_for_up(up).expect("a direction"));
                let shown = (v.linear() * up).normalize();
                assert!(
                    (shown - Vec2::new(0.0, -1.0)).length() < 1e-4,
                    "flip {flip_h}, up {up:?} came out pointing {shown:?}"
                );
            }
        }
        // A drag that has gone nowhere asks for nothing, rather than for whatever
        // `atan2(0, 0)` happens to be.
        let v = ViewTransform::identity(Extent2::new(400, 300));
        assert!(v.rotation_for_up(Vec2::ZERO).is_none());
    }

    /// The mirror is **screen**-relative: at any angle it swaps the left of the screen
    /// with the right. Canvas-relative would, on a canvas turned a quarter, swap top
    /// for bottom instead — a flip, but not the one anyone asked for.
    #[test]
    fn mirroring_swaps_the_screens_left_and_right_at_any_angle() {
        for turn in [0.0, FRAC_PI_2, 1.1, TAU - 0.3] {
            let mut v = ViewTransform {
                rotation: turn,
                ..ViewTransform::identity(Extent2::new(400, 300))
            };
            // Where four canvas directions land on screen, before and after.
            let probes = [Vec2::X, Vec2::Y, Vec2::new(-2.0, 1.0), Vec2::new(3.0, 4.0)];
            let before: Vec<Vec2> = probes.iter().map(|p| v.linear() * *p).collect();
            v.mirror_screen_h();
            for (p, was) in probes.iter().zip(&before) {
                let now = v.linear() * *p;
                assert!(
                    (now - Vec2::new(-was.x, was.y)).length() < 1e-3,
                    "turn {turn}: {p:?} was at {was:?}, mirrored to {now:?}"
                );
            }
            // And twice is the identity — which is what makes it a toggle.
            v.mirror_screen_h();
            for (p, was) in probes.iter().zip(&before) {
                assert!(
                    (v.linear() * *p - *was).length() < 1e-3,
                    "turn {turn}: {p:?}"
                );
            }
        }
    }

    /// What the navigator's marker and "fit to view" both rest on.
    #[test]
    fn visible_bounds_cover_the_viewport_at_any_orientation() {
        let v = ViewTransform {
            center: Vec2::new(10.0, -20.0),
            zoom: 2.0,
            rotation: 0.6,
            flip_h: true,
            viewport: Extent2::new(400, 300),
        };
        let (min, max) = v.visible_bounds();
        let (w, h) = (400.0, 300.0);
        for corner in [
            Vec2::new(0.0, 0.0),
            Vec2::new(w, 0.0),
            Vec2::new(0.0, h),
            Vec2::new(w, h),
        ] {
            let c = v.screen_to_canvas(corner);
            assert!(
                c.x >= min.x - 1e-3
                    && c.x <= max.x + 1e-3
                    && c.y >= min.y - 1e-3
                    && c.y <= max.y + 1e-3,
                "corner {corner:?} -> {c:?} outside {min:?}..{max:?}"
            );
        }
        // Upright, it is exactly the plain half-viewport-over-zoom rect.
        let upright = view(Vec2::ZERO, 2.0, Extent2::new(400, 300));
        assert_eq!(
            upright.visible_bounds(),
            (Vec2::new(-100.0, -75.0), Vec2::new(100.0, 75.0))
        );
    }
}
