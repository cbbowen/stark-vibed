//! Canvas geometry: the tile grid the document is addressed in.
//!
//! **The tile grid only.** How the canvas is being *looked at* — pan, zoom,
//! rotation, the mirror — is session state that is never logged and never sent,
//! so it is `stark-engine`'s `view` (§18.1.2); this crate is the document and
//! nothing else (§2). What stays is what a saved log is addressed in: a footprint
//! quantizes against `TILE_SIZE` (§12.6) and an apron sits one texel inside it
//! (§6.4).
//!
//! Canvas space is in pixels with x to the right and y downward. Tile `(i, j)`
//! covers the square `[i*TILE_SIZE, (i+1)*TILE_SIZE) × [j*TILE_SIZE, ...)`.
//! The infinite canvas (§6) is realized by tiles being sparse and
//! addressed by signed integer coordinates.

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
/// are the same 2×2 question: `stark-engine`'s `assist` reads an ellipse off the second
/// moments of a trace, and its `guides` reads one off the quadratic part of a conic
/// (§20.7). Both callers are in the other crate, which is what makes this `pub`.
pub fn principal_axis(sxx: f32, sxy: f32, syy: f32) -> (f32, f32, Option<Vec2>) {
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
/// ellipse from opposite directions — `stark-engine`'s `assist` reads one off the
/// second moments of a hand-drawn loop, its `guides` off the quadratic
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

    /// Whether this rect reaches nothing — inverted (`min > max`) on **either**
    /// axis.
    ///
    /// One definition, asked by everything here that has to treat an empty rect as
    /// empty. It exists because the emptiness test used to be spelled inline and
    /// per axis: [`intersects`](Self::intersects) guarded x and not y, so a rect
    /// empty only in y intersected everything, while [`union`](Self::union)
    /// guarded both. The fields are public, so the disagreement was reachable from
    /// outside — and this is a footprint predicate, where §12.6 survives a rect
    /// claiming too much and cannot survive one claiming too little.
    pub const fn is_empty(self) -> bool {
        self.min.0 > self.max.0 || self.min.1 > self.max.1
    }

    pub fn intersects(&self, other: &TileRect) -> bool {
        !self.is_empty()
            && !other.is_empty()
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
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
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

    /// **Empty is empty on either axis**, and every predicate has to agree about
    /// it. `intersects` used to guard `min > max` on x alone, so a rect inverted
    /// only in y reached everything — where `union` treated the same rect as the
    /// identity. The fields are public, so the disagreement was reachable, and
    /// this is the predicate the commutation gate rests on: a footprint that
    /// intersects what it does not touch costs the fast path, but a rect that
    /// *claims* to touch nothing while testing positive is the §12.6 direction
    /// with no pixel able to show it.
    #[test]
    fn a_rect_inverted_on_either_axis_reaches_nothing() {
        let real = TileRect::covering(Vec2::ZERO, Vec2::splat(9.0), 0).unwrap();
        let inverted = [
            TileRect::EMPTY,
            // Empty in x alone, and in y alone — the case that used to slip past.
            TileRect {
                min: (5, 0),
                max: (0, 10),
            },
            TileRect {
                min: (0, 5),
                max: (10, 0),
            },
        ];
        for empty in inverted {
            assert!(empty.is_empty(), "{empty:?} should be empty");
            for other in [real, TileRect::ALL, TileRect::EMPTY] {
                assert!(
                    !empty.intersects(&other),
                    "{empty:?} reaches nothing, so it cannot meet {other:?}",
                );
                assert!(!other.intersects(&empty), "…and from the other side");
            }
            // …and it is the identity of the union, not a corner it stretches to.
            assert_eq!(empty.union(real), real);
            assert_eq!(real.union(empty), real);
            assert_eq!(empty.count(), 0);
            assert!(!empty.contains(TileCoord::new(0, 0)));
        }
        assert!(!real.is_empty());
        assert!(real.intersects(&real));
    }
}

// —— tile cover: the geometry every masked pass shares ————————————————————————
//
// A selection, a fill and a transform all have to answer the same question — which
// tiles does this canvas box touch, and is that more than I am willing to walk — and
// they have to answer it *identically*, because a fill's written tiles and its
// footprint are required to be the same tiles (§12.6).
//
// It lives here rather than in `document::selection`, where it grew up, because the
// answer is a fact about the tile grid and not about any one of the three. That the
// GPU side of all three reaches for it (`gpu::fill`, `gpu::selection`) was the
// standing hint; the crate split turned the hint into a `pub`.

/// The tiles whose *texture* (interior + apron) overlaps the canvas box
/// `[lo, hi]`, grown by `ring` tiles — [`TileRect::covering`] with this module's
/// padding, since a tile's texture starts one apron before its interior and a box
/// that reaches into the apron band still touches the neighbour.
///
/// `None` — a refusal, not a clamp — for a box that is not finite or not
/// addressable. That is the only acceptable answer here: a clamp would rasterize
/// a *different* region, and these coordinates arrive from files and peers, where
/// the only tolerable disagreement between two clients is none (§6.8).
pub fn tile_box(lo: Vec2, hi: Vec2, ring: i32) -> Option<TileRect> {
    let apron = Vec2::splat(TILE_APRON as f32);
    TileRect::covering(lo - apron, hi + apron, ring)
}

/// Tiles whose *texture* (interior + apron) overlaps the canvas box `[lo, hi]`,
/// expanded by `ring` tiles on every side — `None` when there would be more than
/// `budget` of them.
///
/// **Counted before it is walked**, which is what makes an absurd box a clean
/// refusal instead of a hang: the box is quadratic in the drag, so a marquee at far
/// zoom-out (or an op arriving from a file or a peer) can name more tiles than
/// there is memory to list, and finding that out by listing them is not an option.
/// Same stance and same shape as `document::transform`'s `quad_reached_tiles`, which counts its
/// candidates against its own budget before enumerating, for the same reason.
pub fn tiles_covering(lo: Vec2, hi: Vec2, ring: i32, budget: usize) -> Option<Vec<TileCoord>> {
    tiles_of(tile_box(lo, hi, ring)?, budget)
}

/// The coordinates of an **already-quantized** rect, `None` when there would be
/// more than `budget` of them — [`tiles_covering`]'s second half, for the caller
/// that has its own reason to hold the `TileRect`.
///
/// The fill is that caller: its written tile set and its footprint have to be the
/// same tiles (§12.6), so the rect is derived once by
/// `document::fill_bounds` and quantized once, and only *then*
/// walked. Counted before it is walked for [`tiles_covering`]'s own reason.
pub fn tiles_of(rect: TileRect, budget: usize) -> Option<Vec<TileCoord>> {
    let count = rect.count();
    if count > budget as u64 {
        return None;
    }
    let mut out = Vec::with_capacity(count as usize);
    out.extend(rect.coords());
    Some(out)
}

/// The lasso's closed edge list, as `selection.wesl` reads it: one texel per edge
/// holding `(a.xy, b.xy)` in canvas px. Empty for a polygon that cannot enclose area.
pub fn lasso_edges(points: &[Vec2]) -> Vec<[f32; 4]> {
    if points.len() < 3 {
        return Vec::new();
    }
    (0..points.len())
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            [a.x, a.y, b.x, b.y]
        })
        .collect()
}

/// The tile geometry a mask tile is rasterized over: its texture's top-left in canvas
/// px (the interior origin, shifted out by the apron — §6.4).
pub fn mask_tex_origin(coord: TileCoord) -> Vec2 {
    coord.origin() - Vec2::splat(TILE_APRON as f32)
}

/// The mask tile's edge length, for the shaders that place it in a region.
pub const MASK_TEX: u32 = TILE_TEX;

#[cfg(test)]
mod cover_tests {
    use super::*;

    /// Coordinates arrive from files and peers. A non-finite bound, or one past
    /// what the `i32` tile grid can address, is refused — never wrapped into a tile
    /// index pointing somewhere else, which is what an `as i32` cast would do.
    #[test]
    fn unrepresentable_boxes_are_refused_rather_than_wrapped() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(tile_box(Vec2::new(bad, 0.0), Vec2::splat(10.0), 0), None);
            assert_eq!(tile_box(Vec2::ZERO, Vec2::new(0.0, bad), 0), None);
        }
        // Well past `i32::MAX` tiles from the origin, but a *small* box — so the
        // count would happily pass and only the addressing is impossible.
        let far = 1.0e30;
        assert_eq!(tile_box(Vec2::splat(far), Vec2::splat(far + 1.0), 0), None);
        assert_eq!(
            tile_box(Vec2::splat(-far), Vec2::splat(-far + 1.0), 0),
            None
        );
    }

    #[test]
    fn lasso_edges_close_the_loop() {
        let pts = vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
        let edges = lasso_edges(&pts);
        assert_eq!(edges.len(), 3);
        assert_eq!(
            edges[2],
            [0.0, 1.0, 0.0, 0.0],
            "last edge returns to the start"
        );
        assert!(
            lasso_edges(&pts[..2]).is_empty(),
            "a segment encloses nothing"
        );
    }
}
