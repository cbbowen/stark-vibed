//! Planning for the affine transform of selected paint (§16).
//!
//! Pure tile-level geometry — which tiles are cut, which are rewritten, which
//! source quads land on each — mirroring [`Selection::plan`]'s split: the CPU
//! decides *what*, [`crate::gpu::transform::TransformRenderer`] does the GPU
//! work. Everything here is a deterministic function of the tile-coordinate
//! sets, the selection's shape, and the affine's six floats, so peers and
//! replays always agree — including about rejection.

use std::collections::BTreeMap;

use rpds::HashTrieMap;

use super::selection::Selection;
use crate::geom::{Affine2, TILE_APRON, TILE_SIZE, TileCoord, Vec2};
use crate::gpu::tile::TilePairHandle;

/// Largest number of paint tiles one transform may rewrite (~650 MB of transient
/// tile allocation at the worst). A transform that would exceed it is rejected
/// whole rather than clipped — the same stance as
/// [`MAX_SELECTION_TILES`](super::selection::MAX_SELECTION_TILES), for the same
/// reason: a silently half-moved painting is worse than a refused move.
pub const MAX_TRANSFORM_TILES: usize = 1024;

/// Ceiling on candidate destination tiles *examined* (before the exact
/// quad-vs-tile test). A pathological affine — a huge scale, an extreme shear —
/// can make a quad's bounding box cover astronomically many tiles that the quad
/// itself never touches; planning must refuse such an action without first
/// walking that box. Deterministic like every other bound here.
const CANDIDATE_BUDGET: usize = 16 * MAX_TRANSFORM_TILES;

/// Whether an affine is usable at all: finite, and not collapsing the plane to a
/// line (paint would silently vanish into a zero-area image — refusing is better,
/// and cheap to agree on).
pub fn affine_usable(affine: Affine2) -> bool {
    let finite = affine.matrix2.is_finite() && affine.translation.is_finite();
    finite && affine.matrix2.determinant().abs() > f32::EPSILON
}

/// How the author's selection stands over one populated tile.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Class {
    /// No mask tile and `outside = 0`: the transform does not touch this tile
    /// (though its texels may still *receive* moved paint).
    Untouched,
    /// A mask tile exists: some fraction of this tile's paint moves.
    Partial,
    /// No mask tile and `outside = 1`: all of it moves.
    Full,
}

fn classify(selection: &Selection, coord: TileCoord) -> Class {
    match selection.tile(coord) {
        Some(_) => Class::Partial,
        None if selection.outside() > 0.5 => Class::Full,
        None => Class::Untouched,
    }
}

/// The tile-level consequence of one transform on a layer's paint.
pub(crate) struct TransformPlan {
    /// Destination tile → the source tiles whose transformed interior quads
    /// reach its texture rect (possibly none: a cut with no incoming paint).
    /// Sorted by destination coordinate; each source list sorted too.
    pub rewrites: Vec<(TileCoord, Vec<TileCoord>)>,
    /// Fully-selected source tiles nothing lands back on: removed from the map.
    /// (A rewrite would write all zeros, and an all-zero tile pollutes `bounds`
    /// and holds pool memory that "no tile" would not.)
    pub drops: Vec<TileCoord>,
}

/// Plan one transform of `tiles` under `selection`. `None` rejects the action —
/// unusable affine, or more work than the caps allow — leaving the document
/// untouched, deterministically.
pub(crate) fn plan_paint(
    tiles: &HashTrieMap<TileCoord, TilePairHandle>,
    selection: &Selection,
    affine: Affine2,
) -> Option<TransformPlan> {
    if !affine_usable(affine) {
        return None;
    }
    // Sorted for a deterministic plan (the persistent map iterates unordered).
    let mut coords: Vec<TileCoord> = tiles.keys().copied().collect();
    coords.sort();

    let mut rewrites: BTreeMap<TileCoord, Vec<TileCoord>> = BTreeMap::new();
    let mut sources: Vec<(TileCoord, Class)> = Vec::new();
    for coord in coords {
        match classify(selection, coord) {
            Class::Untouched => {}
            class @ Class::Partial => {
                // A partial tile is always rewritten: its own cut happens even if
                // no quad reaches it.
                rewrites.entry(coord).or_default();
                sources.push((coord, class));
            }
            class @ Class::Full => sources.push((coord, class)),
        }
    }

    let mut candidates = 0usize;
    for (coord, _) in &sources {
        for dest in reached_tiles(affine, *coord, &mut candidates)? {
            let list = rewrites.entry(dest).or_default();
            if list.last() != Some(coord) {
                list.push(*coord);
            }
            if rewrites.len() > MAX_TRANSFORM_TILES {
                return None;
            }
        }
    }

    let drops = sources
        .iter()
        .filter(|(c, class)| *class == Class::Full && !rewrites.contains_key(c))
        .map(|(c, _)| *c)
        .collect();
    Some(TransformPlan {
        rewrites: rewrites.into_iter().collect(),
        drops,
    })
}

/// The tile-level consequence of one transform on the author's selection mask:
/// pure Replace — the old tiles are dropped and the destinations rasterized
/// afresh. `outside` is untouched: an affine of the whole plane is the whole
/// plane, so the coverage at infinity cannot change.
pub(crate) struct MaskPlan {
    /// Destination mask tile → the source mask tiles reaching it.
    pub rewrites: Vec<(TileCoord, Vec<TileCoord>)>,
}

/// Plan the selection mask's own move under the same affine. Bounded by
/// [`MAX_SELECTION_TILES`](super::selection::MAX_SELECTION_TILES) — a rejection
/// here rejects the whole action, so the paint never moves out from under its
/// selection.
pub(crate) fn plan_mask(selection: &Selection, affine: Affine2) -> Option<MaskPlan> {
    if !affine_usable(affine) {
        return None;
    }
    let mut coords: Vec<TileCoord> = selection.tiles().map(|(c, _)| *c).collect();
    coords.sort();

    let mut rewrites: BTreeMap<TileCoord, Vec<TileCoord>> = BTreeMap::new();
    let mut candidates = 0usize;
    for coord in coords {
        for dest in reached_tiles(affine, coord, &mut candidates)? {
            let list = rewrites.entry(dest).or_default();
            if list.last() != Some(&coord) {
                list.push(coord);
            }
            if rewrites.len() > super::selection::MAX_SELECTION_TILES {
                return None;
            }
        }
    }
    Some(MaskPlan {
        rewrites: rewrites.into_iter().collect(),
    })
}

/// The corners of `coord`'s *interior* under `affine`, in canvas px — the quad
/// the parcel pass draws. Corner order matches the shader's vertex indices
/// (`corner = (vi & 1, vi >> 1 & 1)`).
pub(crate) fn quad_corners(affine: Affine2, coord: TileCoord) -> [Vec2; 4] {
    let o = coord.origin();
    let s = TILE_SIZE as f32;
    [
        affine.transform_point2(o),
        affine.transform_point2(o + Vec2::new(s, 0.0)),
        affine.transform_point2(o + Vec2::new(0.0, s)),
        affine.transform_point2(o + Vec2::new(s, s)),
    ]
}

/// Destination tiles whose *texture* rect (interior + apron) the transformed
/// interior quad of `coord` actually reaches, by exact convex intersection —
/// a bounding-box test would mint tiles a rotated quad never touches, and every
/// minted tile is a real allocation. `None` when the search itself would exceed
/// [`CANDIDATE_BUDGET`].
fn reached_tiles(
    affine: Affine2,
    coord: TileCoord,
    candidates: &mut usize,
) -> Option<Vec<TileCoord>> {
    let quad = quad_corners(affine, coord);
    let lo = quad.iter().fold(quad[0], |a, p| a.min(*p));
    let hi = quad.iter().fold(quad[0], |a, p| a.max(*p));

    // Tiles whose texture rect [origin − apron, origin + TILE_SIZE + apron]
    // overlaps the quad's AABB. A fragment is only produced where the quad covers
    // a texel center, but the rect test keeps the half-open bookkeeping simple —
    // the exact test below prunes the rest.
    let tile = TILE_SIZE as f32;
    let apron = TILE_APRON as f32;
    let x0 = ((lo.x - apron) / tile).floor() as i64;
    let x1 = ((hi.x + apron) / tile).floor() as i64;
    let y0 = ((lo.y - apron) / tile).floor() as i64;
    let y1 = ((hi.y + apron) / tile).floor() as i64;
    let count = ((x1 - x0 + 1).max(0) as usize).checked_mul((y1 - y0 + 1).max(0) as usize)?;
    *candidates = candidates.checked_add(count)?;
    if *candidates > CANDIDATE_BUDGET {
        return None;
    }

    let mut out = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let c = TileCoord::new(x as i32, y as i32);
            let min = c.origin() - Vec2::splat(apron);
            let max = c.origin() + Vec2::splat(tile + apron);
            if quad_intersects_rect(&quad, min, max) {
                out.push(c);
            }
        }
    }
    Some(out)
}

/// Exact convex intersection between a parallelogram (an affine image of a rect)
/// and an axis-aligned rect, by separating axes: the rect's two axes and the
/// parallelogram's two edge normals.
fn quad_intersects_rect(quad: &[Vec2; 4], min: Vec2, max: Vec2) -> bool {
    // Rect axes: the quad's AABB against the rect.
    let qlo = quad.iter().fold(quad[0], |a, p| a.min(*p));
    let qhi = quad.iter().fold(quad[0], |a, p| a.max(*p));
    if qhi.x < min.x || qlo.x > max.x || qhi.y < min.y || qlo.y > max.y {
        return false;
    }
    // Parallelogram edge normals (two unique directions: 0→1 and 0→2).
    for edge in [quad[1] - quad[0], quad[2] - quad[0]] {
        let n = Vec2::new(-edge.y, edge.x);
        let (qmin, qmax) = project(quad.iter().copied(), n);
        let rect = [min, Vec2::new(max.x, min.y), Vec2::new(min.x, max.y), max];
        let (rmin, rmax) = project(rect.into_iter(), n);
        if qmax < rmin || qmin > rmax {
            return false;
        }
    }
    true
}

fn project(points: impl Iterator<Item = Vec2>, axis: Vec2) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for p in points {
        let d = p.dot(axis);
        min = min.min(d);
        max = max.max(d);
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translation(x: f32, y: f32) -> Affine2 {
        Affine2::from_translation(Vec2::new(x, y))
    }

    #[test]
    fn unusable_affines_are_rejected() {
        assert!(!affine_usable(Affine2::from_scale(Vec2::new(0.0, 1.0))));
        assert!(!affine_usable(Affine2::from_translation(Vec2::new(
            f32::NAN,
            0.0
        ))));
        assert!(affine_usable(Affine2::IDENTITY));
        assert!(affine_usable(Affine2::from_angle(0.7)));
    }

    #[test]
    fn identity_quad_reaches_itself_and_its_ring_edges() {
        // The interior quad of (0,0) under identity overlaps only tiles whose
        // texture rect (apron included) it touches: itself and the 8 neighbours
        // (their aprons reach 1 px into it).
        let mut n = 0;
        let reached = reached_tiles(Affine2::IDENTITY, TileCoord::new(0, 0), &mut n).unwrap();
        assert!(reached.contains(&TileCoord::new(0, 0)));
        assert_eq!(reached.len(), 9);
    }

    #[test]
    fn rotated_quad_prunes_far_corners() {
        // A large 45°-rotated quad's AABB covers far more tiles than the diamond
        // touches; the exact test must prune the AABB's corner tiles.
        let rot = Affine2::from_mat2_translation(
            glam::Mat2::from_angle(std::f32::consts::FRAC_PI_4) * 6.0,
            Vec2::ZERO,
        );
        let mut aabb = 0;
        let reached = reached_tiles(rot, TileCoord::new(4, 4), &mut aabb).unwrap();
        assert!(reached.len() < aabb, "exact test pruned nothing");
    }

    #[test]
    fn integer_translation_reaches_the_shifted_neighbourhood() {
        // A whole-tile shift right by one tile: (0,0)'s quad lands exactly on
        // (1,0)'s interior, reaching it and its ring.
        let mut n = 0;
        let reached = reached_tiles(
            translation(TILE_SIZE as f32, 0.0),
            TileCoord::new(0, 0),
            &mut n,
        )
        .unwrap();
        assert!(reached.contains(&TileCoord::new(1, 0)));
        assert!(!reached.contains(&TileCoord::new(3, 0)));
        assert_eq!(reached.len(), 9);
    }

    #[test]
    fn extreme_scale_blows_the_candidate_budget() {
        let mut n = 0;
        assert!(
            reached_tiles(
                Affine2::from_scale(Vec2::splat(1e4)),
                TileCoord::new(0, 0),
                &mut n
            )
            .is_none(),
            "a 10_000× scale of one tile must be refused, not enumerated"
        );
    }

    #[test]
    fn empty_inputs_plan_trivially() {
        // No paint and no mask tiles: both plans succeed and do nothing — the
        // classification and GPU paths are exercised by tests/transform.rs,
        // which has a device to mint real tile handles with.
        let tiles: HashTrieMap<TileCoord, TilePairHandle> = HashTrieMap::new();
        let sel = Selection::everything();
        let plan = plan_paint(&tiles, &sel, translation(10_000.0, 0.0)).unwrap();
        assert!(plan.rewrites.is_empty() && plan.drops.is_empty());
        let mask = plan_mask(&sel, translation(300.0, 0.0)).unwrap();
        assert!(mask.rewrites.is_empty());
    }
}
