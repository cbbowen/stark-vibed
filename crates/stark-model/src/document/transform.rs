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

    /// The same map, restated in a layer frame placed at `frame` on the canvas
    /// (§14.12): the conjugation `T(−f) ∘ M ∘ T(f)` — what the paint side of a
    /// transform applies while the mask side keeps the canvas map. For the
    /// rect-scoped families every defining point simply shifts, since the map
    /// *is* its point correspondences; the affine composes, exactly for the
    /// whole-pixel frames and translations the exactness invariants are about
    /// (§16.4), since integer sums in `f32` are exact to 2²⁴.
    pub fn under_translation(&self, translation: crate::geom::IVec2) -> Self {
        if translation == crate::geom::IVec2::ZERO {
            return self.clone();
        }
        let f = translation.as_vec2();
        match self {
            TransformMap::Affine(a) => TransformMap::Affine(
                Affine2::from_translation(-f) * *a * Affine2::from_translation(f),
            ),
            TransformMap::Perspective(p) => TransformMap::Perspective(PerspectiveMap {
                min: p.min - f,
                max: p.max - f,
                corners: p.corners.map(|c| c - f),
            }),
            TransformMap::Warp(w) => TransformMap::Warp(w.translated(-f)),
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
    /// The forward and inverse maps of a **usable** perspective — the whole gate
    /// and the solve, once. `None` for anything `apply` would refuse.
    ///
    /// The gate and the solve are one function because they were two: a caller
    /// that wanted only the matrices reached past the gate and got a map `apply`
    /// rejects, and one that wanted both paid the f64 general-quad solve twice.
    /// Everything below is defined in terms of this.
    pub fn resolve(&self) -> Option<(Homography, Homography)> {
        let gated = self.min.is_finite()
            && self.max.is_finite()
            && self.max.x > self.min.x
            && self.max.y > self.min.y
            && self.corners.iter().all(|c| c.is_finite())
            && convex_positive(&self.corners);
        gated.then(|| self.homographies()).flatten()
    }

    /// Whether this perspective may be applied: finite, a proper rect, and a
    /// strictly convex, positively oriented target quad. Convexity is not
    /// taste — it is exactly the condition under which the homography keeps
    /// the whole source rect on the near side of its horizon (`w > 0`), so
    /// nothing inside the rect can be flung through infinity.
    pub fn usable(&self) -> bool {
        self.resolve().is_some()
    }

    /// The forward map, for chrome that draws the transformed space (grid
    /// lines are straight, so two endpoints per line suffice). `None` when
    /// unusable — the same "unusable" `apply` means, which is why it is
    /// [`resolve`](Self::resolve) and not the bare solve.
    pub fn forward(&self) -> Option<Homography> {
        self.resolve().map(|(f, _)| f)
    }

    /// A conservative bound on where the map carries paint: the target quad's
    /// own bounding box (a homography maps the rect *onto* the quad). `None` when
    /// the quad cannot be measured.
    ///
    /// **An `Option` for [`WarpMap::image_aabb`](super::warp::WarpMap::image_aabb)'s
    /// reason.** The fold below is `min`/`max`, which return the *non*-NaN operand, so
    /// a non-finite corner would step straight over the box and leave it looking
    /// tight. Answering `None` instead means a caller with no `apply` behind it cannot
    /// get a tight box out of a map that has none — rather than resting on `usable`
    /// refusing the map in another file.
    pub fn image_aabb(&self) -> Option<(Vec2, Vec2)> {
        self.corners.iter().all(|c| c.is_finite()).then(|| {
            let lo = self.corners.iter().fold(self.corners[0], |a, p| a.min(*p));
            let hi = self.corners.iter().fold(self.corners[0], |a, p| a.max(*p));
            (lo, hi)
        })
    }

    /// [`resolve`](Self::resolve)'s solve half — **private**, because on its own it
    /// answers a weaker question than any caller wants: `near_side` tests `w > 0`,
    /// which says nothing about orientation, so a reflected quad derives cleanly here
    /// and is refused there.
    ///
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
    fn homographies(&self) -> Option<(Homography, Homography)> {
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
///
/// The threshold is **absolute, on an area in px²**, so it is a floor on how thin a
/// quad may be at canvas scale — not the dimensionless one `near_side` applies to a
/// projective weight. The two are the same condition only up to that assumption.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A convex, positively oriented quad that is neither the rect nor a
    /// parallelogram — the general-quad tier, and the only one that runs the f64
    /// solve and the adjugate (§16.4).
    const TRAPEZOID: [Vec2; 4] = [
        Vec2::new(50.0, 0.0),
        Vec2::new(150.0, 0.0),
        Vec2::new(0.0, 200.0),
        Vec2::new(200.0, 200.0),
    ];

    fn map(min: Vec2, max: Vec2, corners: [Vec2; 4]) -> PerspectiveMap {
        PerspectiveMap { min, max, corners }
    }

    /// **Tier 1: untouched corners derive the literal identity matrix** (§16.4).
    ///
    /// `assert_eq!` rather than a tolerance, because what rides on it is the
    /// fragment's tap arithmetic: an identity that is only identity to 1e-7
    /// resamples every texel of a transform nobody asked for.
    #[test]
    fn untouched_corners_derive_the_literal_identity() {
        for (min, max) in [
            (Vec2::ZERO, Vec2::splat(256.0)),
            (Vec2::new(-64.5, 32.25), Vec2::new(192.75, 288.125)),
            (Vec2::new(-1e5, -1e5), Vec2::new(1e5, 1e5)),
        ] {
            let p = map(min, max, rect_corners(min, max));
            assert!(p.usable(), "{min:?}..{max:?} should be usable");
            let (fwd, inv) = p.resolve().expect("a rect maps to itself");
            assert_eq!(fwd, Homography::IDENTITY);
            assert_eq!(inv, Homography::IDENTITY);
            // …and applying it is a no-op to the bit, which is the claim the
            // matrix is only a proxy for.
            for probe in [min, max, 0.5 * (min + max), Vec2::new(min.x, max.y)] {
                assert_eq!(fwd.apply(probe).to_array(), probe.to_array());
            }
        }
    }

    /// **Tier 2: a parallelogram target rides the affine arithmetic** (§16.4) —
    /// bottom row exactly `(0, 0, 1)`, so the projective divide is by exactly 1.
    ///
    /// A `(0, 0, ~1)` row would still draw the right picture and still lose the
    /// affine's exactness, which is why this is an equality on the row rather
    /// than a check that the map is nearly affine.
    #[test]
    fn a_parallelogram_target_keeps_a_unit_bottom_row() {
        let (min, max) = (Vec2::ZERO, Vec2::new(100.0, 50.0));
        // A shear and a whole-pixel shift: a parallelogram, not the rect, so the
        // identity short-circuit cannot be what answers.
        let shear = |c: Vec2| c + Vec2::new(0.5 * c.y + 16.0, -32.0);
        let p = map(min, max, rect_corners(min, max).map(shear));
        assert_ne!(p.corners, rect_corners(min, max), "not the identity tier");
        let (fwd, inv) = p.resolve().expect("a parallelogram is usable");
        assert_eq!(fwd.rows[2], [0.0, 0.0, 1.0]);
        assert_eq!(inv.rows[2], [0.0, 0.0, 1.0]);
        for (src, dst) in rect_corners(min, max).into_iter().zip(p.corners) {
            assert_eq!(fwd.apply(src).to_array(), dst.to_array(), "{src:?}");
        }
    }

    /// **Tier 3: the adjugate really is the inverse**, over the rect's interior.
    ///
    /// [`mat3_adjugate`] is right or catastrophically wrong with nothing in
    /// between — a transposed cofactor or one flipped sign puts the paint
    /// somewhere else entirely — and a round trip over a grid is the cheap
    /// statement of that. Nothing else in this crate runs it.
    #[test]
    fn a_general_quad_inverts_across_its_whole_source_rect() {
        let (min, max) = (Vec2::new(-40.0, 20.0), Vec2::new(160.0, 120.0));
        let p = map(min, max, TRAPEZOID);
        let (fwd, inv) = p.resolve().expect("a trapezoid is usable");
        assert_ne!(fwd.rows[2], [0.0, 0.0, 1.0], "this is the projective tier");
        // The corners land where the hand put them…
        for (src, dst) in rect_corners(min, max).into_iter().zip(p.corners) {
            assert!((fwd.apply(src) - dst).length() < 1e-2, "{src:?}");
        }
        // …and the inverse undoes the forward everywhere between them.
        for i in 0..=8 {
            for j in 0..=8 {
                let t = Vec2::new(i as f32 / 8.0, j as f32 / 8.0);
                let src = min + t * (max - min);
                let round = inv.apply(fwd.apply(src));
                assert!(
                    (round - src).length() < 1e-2,
                    "{src:?} round-tripped to {round:?}",
                );
            }
        }
    }

    /// [`convex_positive`] is the horizon condition spelled as a shape test: it
    /// takes a strictly convex, positively oriented quad and refuses the ways one
    /// stops being that.
    ///
    /// Reached directly rather than through [`PerspectiveMap::usable`] because
    /// the mirrored quad is a *parallelogram*: the solve derives a clean matrix
    /// for it and `near_side` says nothing about orientation, so this predicate
    /// is the only thing in front of it.
    #[test]
    fn convexity_accepts_a_trapezoid_and_refuses_a_folded_quad() {
        let base = rect_corners(Vec2::ZERO, Vec2::splat(100.0));
        assert!(convex_positive(&TRAPEZOID));
        assert!(convex_positive(&base));
        // Mirrored: convex, wound the other way.
        assert!(!convex_positive(&[base[1], base[0], base[3], base[2]]));
        // Crossed: one pair swapped, so two edges meet in the middle.
        assert!(!convex_positive(&[base[1], base[0], base[2], base[3]]));
        // Collapsed onto a line: not thin, gone.
        assert!(!convex_positive(&[base[0], base[1], base[0], base[1]]));
    }

    /// The gate half of [`PerspectiveMap::resolve`], which is not the solve half:
    /// an empty or inverted source rect, and a corner nobody can measure, are
    /// refused before the solve sees them.
    ///
    /// The empty rect is the sharp case — its corners *are* `rect_corners`', so
    /// without the gate in front of it the identity short-circuit would answer
    /// [`Homography::IDENTITY`] for a rect with no interior.
    #[test]
    fn a_map_with_no_source_rect_or_no_finite_corner_is_refused() {
        let min = Vec2::new(10.0, 10.0);
        for max in [
            min,                       // empty both ways
            Vec2::new(10.0, 90.0),     // empty in x
            Vec2::new(90.0, 10.0),     // empty in y
            Vec2::new(-90.0, 90.0),    // inverted in x
            Vec2::new(90.0, -90.0),    // inverted in y
            Vec2::new(f32::NAN, 90.0), // unmeasurable
            Vec2::new(90.0, f32::INFINITY),
        ] {
            let p = map(min, max, rect_corners(min, max));
            assert!(!p.usable(), "max {max:?} leaves no source rect");
            assert!(p.resolve().is_none());
            assert!(p.forward().is_none());
        }
        // A proper rect whose *image* has a corner nobody can place.
        let (min, max) = (Vec2::ZERO, Vec2::splat(100.0));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut corners = rect_corners(min, max);
            corners[2].y = bad;
            let p = map(min, max, corners);
            assert!(!p.usable(), "a {bad} corner should be refused");
            assert!(p.image_aabb().is_none(), "…and it has no measurable box");
        }
    }

    /// [`Homography::from_f64`] normalizes by the largest element, so the nine
    /// numbers stay in a healthy float range whatever scale the derivation worked
    /// at — and refuses a matrix with no scale to normalize by.
    ///
    /// Reached directly because it is private and the solve never hands it the
    /// degenerate cases: through a caller they are one `None` among several.
    #[test]
    fn a_matrix_is_normalized_by_its_largest_element_or_refused() {
        let scaled = Homography::from_f64(&[[2.0, 0.0, 0.0], [0.0, -4.0, 0.0], [0.0, 0.0, 8.0]])
            .expect("a finite matrix with a scale");
        assert_eq!(
            scaled.rows,
            [[0.25, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]],
        );
        // The same matrix a billion times larger is the same projective map, and
        // arrives as the same nine numbers.
        let huge = Homography::from_f64(&[[2e9, 0.0, 0.0], [0.0, -4e9, 0.0], [0.0, 0.0, 8e9]])
            .expect("scale is normalized away");
        assert_eq!(huge.rows, scaled.rows);
        // Nothing to normalize by.
        assert!(Homography::from_f64(&[[0.0; 3]; 3]).is_none());
        assert!(Homography::from_f64(&[[f64::NAN; 3]; 3]).is_none());
        // A scale that is not a number, and one element that is not.
        assert!(Homography::from_f64(&[[f64::INFINITY, 0.0, 0.0], [0.0; 3], [0.0; 3]]).is_none());
        let mut one_bad = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        one_bad[1][2] = f64::NAN;
        assert!(Homography::from_f64(&one_bad).is_none());
        // And the property every derived map inherits from it.
        let (fwd, inv) = map(Vec2::new(-40.0, 20.0), Vec2::new(160.0, 120.0), TRAPEZOID)
            .resolve()
            .expect("a trapezoid is usable");
        for m in [fwd, inv] {
            let largest = m.rows.iter().flatten().fold(0.0f32, |a, v| a.max(v.abs()));
            assert_eq!(largest, 1.0, "{m:?} was not normalized");
        }
    }
}
