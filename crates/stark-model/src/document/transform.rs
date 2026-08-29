//! Planning for the transforms of selected paint (§16): the whole-plane
//! affine, and the rect-scoped perspective (§16.8) and warp (§16.9).
//!
//! Pure tile-level geometry — which tiles are cut, which are rewritten, which
//! source quads land on each — mirroring `stark-engine`'s `Selection::plan`
//! split: the CPU decides *what*, `stark-engine`'s
//! `gpu::transform::TransformRenderer` does the GPU work. Everything here is a deterministic function of the tile-coordinate
//! sets, the selection's shape, and the map's few floats, so peers and
//! replays always agree — including about rejection.

use serde::{Deserialize, Serialize};

use super::warp::WarpMap;
use crate::geom::{Affine2, Mat2, Vec2};

/// Largest number of paint tiles one transform may rewrite (~650 MB of transient
/// tile allocation at the worst). A transform that would exceed it is rejected
/// whole rather than clipped — the same stance as
/// [`MAX_SELECTION_TILES`](super::selection::MAX_SELECTION_TILES), for the same
/// reason: a silently half-moved painting is worse than a refused move.
pub const MAX_TRANSFORM_TILES: usize = 1024;

/// Whether an affine is usable at all: finite, and not collapsing the plane to a
/// line (paint would silently vanish into a zero-area image — refusing is better,
/// and cheap to agree on).
pub fn affine_usable(affine: Affine2) -> bool {
    let finite = affine.matrix2.is_finite() && affine.translation.is_finite();
    finite && affine.matrix2.determinant().abs() > f32::EPSILON
}

/// The three transform families a gesture can commit (§16, §16.8, §16.9). The
/// affine acts on the whole plane; perspective and warp act on the paint under
/// the author's mask **within their source rect**, leaving everything outside
/// untouched — which is what keeps a distant homography's horizon, or a mesh's
/// edge, from ever reaching paint the gesture was not about.
#[derive(Clone, Debug, PartialEq)]
pub enum TransformMap {
    Affine(Affine2),
    Perspective(PerspectiveMap),
    Warp(WarpMap),
}

impl TransformMap {
    /// Whether the map may be applied at all — finite, orientation-preserving
    /// where that matters, not collapsing paint into a line. Deterministic, so
    /// peers and replays agree about rejection (§16.1).
    pub fn usable(&self) -> bool {
        match self {
            TransformMap::Affine(a) => affine_usable(*a),
            TransformMap::Perspective(p) => p.usable(),
            TransformMap::Warp(w) => w.usable(),
        }
    }
}

/// A projective transform of the selected paint inside `[min, max]` (§16.8):
/// the homography carrying the rect's corners to `corners`, exactly — the wire
/// form is the corners the hand placed, and every peer re-derives the same
/// matrix from them. Straight lines stay straight, which is the whole reason
/// this is not a warp special case (a bilinear mesh bends diagonals).
///
/// Wire format note (§8): these field *names* are what a saved map is read back by,
/// so renaming one needs a `#[serde(alias)]`. Order is free.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct PerspectiveMap {
    /// Source rect, canvas px.
    pub min: Vec2,
    pub max: Vec2,
    /// Images of the rect's corners, in the shader's `(vi & 1, vi >> 1 & 1)`
    /// order: (min), (max.x, min.y), (min.x, max.y), (max).
    pub corners: [Vec2; 4],
}

/// The corners of `[min, max]` in [`PerspectiveMap::corners`]'s order. One
/// shared formula, because "identity" is defined as "the corners *are* these
/// values", bitwise.
pub fn rect_corners(min: Vec2, max: Vec2) -> [Vec2; 4] {
    [min, Vec2::new(max.x, min.y), Vec2::new(min.x, max.y), max]
}

impl PerspectiveMap {
    /// Whether this perspective may be applied: finite, a proper rect, and a
    /// strictly convex, positively oriented target quad. Convexity is not
    /// taste — it is exactly the condition under which the homography keeps
    /// the whole source rect on the near side of its horizon (`w > 0`), so
    /// nothing inside the rect can be flung through infinity.
    pub fn usable(&self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && self.max.x > self.min.x
            && self.max.y > self.min.y
            && self.corners.iter().all(|c| c.is_finite())
            && convex_positive(&self.corners)
            && self.homographies().is_some()
    }

    /// The forward map, for chrome that draws the transformed space (grid
    /// lines are straight, so two endpoints per line suffice). `None` when
    /// unusable.
    pub fn forward(&self) -> Option<Homography> {
        self.homographies().map(|(f, _)| f)
    }

    /// A conservative bound on where the map carries paint: the target quad's
    /// own bounding box (a homography maps the rect *onto* the quad). `None` when
    /// the quad cannot be measured.
    ///
    /// **An `Option` for [`WarpMap::image_aabb`](super::warp::WarpMap::image_aabb)'s
    /// reason**, and it took the footprint to make that visible. The fold below is
    /// `min`/`max`, which return the *non*-NaN operand, so a non-finite corner used
    /// to step straight over the box and leave it looking tight — and
    /// `footprint.rs` passed the result through as `Some(...)`, where the warp arm
    /// beside it fell back to the whole layer. That was safe only because `usable`
    /// refuses such a map at `apply`, which is honesty resting on a refusal in
    /// another file — the very thing `MergeLayerDown`'s footprint declines to do.
    /// Said this way the two arms are the same claim for the same reason, and a
    /// caller with no `apply` behind it cannot get a tight box out of a map that
    /// has none.
    pub fn image_aabb(&self) -> Option<(Vec2, Vec2)> {
        self.corners.iter().all(|c| c.is_finite()).then(|| {
            let lo = self.corners.iter().fold(self.corners[0], |a, p| a.min(*p));
            let hi = self.corners.iter().fold(self.corners[0], |a, p| a.max(*p));
            (lo, hi)
        })
    }

    /// Forward and inverse homographies, derived deterministically from the
    /// corners. Three tiers, each preserving more exactness than the last
    /// (§16.4):
    ///
    /// - **corners untouched** — both maps are the literal identity matrix, so
    ///   the fragment's tap arithmetic is exact and identity is a no-op;
    /// - **a parallelogram target** — the map is affine; it is built and
    ///   inverted through [`Affine2`], the same arithmetic the affine action
    ///   trusts, and embedded with a `(0, 0, 1)` bottom row so the projective
    ///   divide is by exactly 1;
    /// - **a general quad** — derived in f64 (square-to-quad, composed with
    ///   the rect normalization; the inverse is the adjugate) and rounded to
    ///   f32 once. f64 on the CPU is deterministic, so peers agree.
    pub fn homographies(&self) -> Option<(Homography, Homography)> {
        let base = rect_corners(self.min, self.max);
        if self.corners == base {
            return Some((Homography::IDENTITY, Homography::IDENTITY));
        }
        let c = self.corners;
        if c[3] - c[2] == c[1] - c[0] {
            let size = self.max - self.min;
            let m = Mat2::from_cols((c[1] - c[0]) / size.x, (c[2] - c[0]) / size.y);
            // Written as a negated positive test so a NaN determinant rejects.
            let invertible = m.determinant().abs() > f32::EPSILON;
            if !invertible {
                return None;
            }
            let fwd = Affine2::from_mat2_translation(m, c[0] - m * self.min);
            let inv = fwd.inverse();
            if !(fwd.matrix2.is_finite() && fwd.translation.is_finite())
                || !(inv.matrix2.is_finite() && inv.translation.is_finite())
            {
                return None;
            }
            return Some((Homography::from_affine(fwd), Homography::from_affine(inv)));
        }

        // General quad, in f64. Square-to-quad after Heckbert: the boundary
        // order is 00 → 10 → 11 → 01.
        let p: [[f64; 2]; 4] = [
            [c[0].x as f64, c[0].y as f64],
            [c[1].x as f64, c[1].y as f64],
            [c[3].x as f64, c[3].y as f64],
            [c[2].x as f64, c[2].y as f64],
        ];
        let sx = p[0][0] - p[1][0] + p[2][0] - p[3][0];
        let sy = p[0][1] - p[1][1] + p[2][1] - p[3][1];
        let (dx1, dy1) = (p[1][0] - p[2][0], p[1][1] - p[2][1]);
        let (dx2, dy2) = (p[3][0] - p[2][0], p[3][1] - p[2][1]);
        let den = dx1 * dy2 - dx2 * dy1;
        if den == 0.0 {
            return None;
        }
        let g = (sx * dy2 - sy * dx2) / den;
        let h = (dx1 * sy - dy1 * sx) / den;
        // The projective weight at the unit-square corners is affine in (u, v),
        // so positivity at the corners is positivity everywhere on the rect.
        // Every corner must sit strictly on the near side (a NaN weight fails
        // the positive test and rejects, which is why it is spelled this way).
        let near_side = [1.0, 1.0 + g, 1.0 + h, 1.0 + g + h]
            .iter()
            .all(|w| *w > 1e-9);
        if !near_side {
            return None;
        }
        let unit = [
            [
                p[1][0] - p[0][0] + g * p[1][0],
                p[3][0] - p[0][0] + h * p[3][0],
                p[0][0],
            ],
            [
                p[1][1] - p[0][1] + g * p[1][1],
                p[3][1] - p[0][1] + h * p[3][1],
                p[0][1],
            ],
            [g, h, 1.0],
        ];
        let (w, hh) = (
            (self.max.x - self.min.x) as f64,
            (self.max.y - self.min.y) as f64,
        );
        let (mx, my) = (self.min.x as f64, self.min.y as f64);
        let norm = [
            [1.0 / w, 0.0, -mx / w],
            [0.0, 1.0 / hh, -my / hh],
            [0.0, 0.0, 1.0],
        ];
        let fwd = mat3_mul(&unit, &norm);
        let inv = mat3_adjugate(&fwd);
        Some((Homography::from_f64(&fwd)?, Homography::from_f64(&inv)?))
    }
}

/// Whether a target quad is strictly convex and positively oriented (the same
/// orientation as the rect it images — canvas axes, y down). A crossed or
/// reflected quad would run the map through its own horizon.
fn convex_positive(c: &[Vec2; 4]) -> bool {
    let b = [c[0], c[1], c[3], c[2]];
    (0..4).all(|i| {
        let e0 = b[(i + 1) % 4] - b[i];
        let e1 = b[(i + 2) % 4] - b[(i + 1) % 4];
        e0.perp_dot(e1) > 1e-6
    })
}

/// A plane projective map as the rows of its 3×3 matrix, in canvas px:
/// `x' = (r0 · (x, y, 1)) / w`, `y' = (r1 · (x, y, 1)) / w`,
/// `w = r2 · (x, y, 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography {
    pub rows: [[f32; 3]; 3],
}

impl Homography {
    pub const IDENTITY: Self = Self {
        rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    /// An affine embedded with a literal `(0, 0, 1)` bottom row, so applying
    /// it divides by exactly 1 and keeps the affine's tap exactness.
    pub fn from_affine(a: Affine2) -> Self {
        Self {
            rows: [
                [a.matrix2.x_axis.x, a.matrix2.y_axis.x, a.translation.x],
                [a.matrix2.x_axis.y, a.matrix2.y_axis.y, a.translation.y],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Rounded to f32 after normalizing by the largest element, so the nine
    /// numbers stay in a healthy float range whatever scale the derivation
    /// worked at. `None` if anything fails to survive the trip.
    fn from_f64(m: &[[f64; 3]; 3]) -> Option<Self> {
        let scale = m.iter().flatten().fold(0.0f64, |a, v| a.max(v.abs()));
        if !(scale > 0.0 && scale.is_finite()) {
            return None;
        }
        let mut rows = [[0.0f32; 3]; 3];
        for (r, row) in m.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                let v = (v / scale) as f32;
                if !v.is_finite() {
                    return None;
                }
                rows[r][c] = v;
            }
        }
        Some(Self { rows })
    }

    /// The map at `p`. Callers stay on the near side of the horizon (the
    /// planner only maps points inside the source rect, where `usable`
    /// guarantees `w > 0`).
    pub fn apply(&self, p: Vec2) -> Vec2 {
        let r = &self.rows;
        let w = r[2][0] * p.x + r[2][1] * p.y + r[2][2];
        Vec2::new(
            (r[0][0] * p.x + r[0][1] * p.y + r[0][2]) / w,
            (r[1][0] * p.x + r[1][1] * p.y + r[1][2]) / w,
        )
    }
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// The adjugate — a scalar multiple of the inverse, which is all a homography
/// needs (it is only ever used projectively).
fn mat3_adjugate(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = |r: usize, s: usize| -> f64 {
        let (r0, r1) = match r {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let (c0, c1) = match s {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let minor = m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
        if (r + s).is_multiple_of(2) {
            minor
        } else {
            -minor
        }
    };
    // Adjugate = transpose of the cofactor matrix.
    [
        [c(0, 0), c(1, 0), c(2, 0)],
        [c(0, 1), c(1, 1), c(2, 1)],
        [c(0, 2), c(1, 2), c(2, 2)],
    ]
}
