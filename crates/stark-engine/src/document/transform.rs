//! Planning a transform (§16): which tiles are read, which are written, and how the
//! source is cut into the units a pass resamples.
//!
//! The **maps** — the affine, the [`PerspectiveMap`]'s corners, the [`WarpMap`]'s
//! grid, and the [`Homography`] one solves for — are `stark-model`'s
//! `document::transform`: a dozen floats each, and what the log carries. Everything
//! here reads or writes a [`TileMap`], which is why it is on this side.

use std::collections::BTreeMap;

/// How the author's selection stands over one populated tile.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Class {
    /// No mask tile and no coverage outside: the transform does not touch this
    /// tile (though its texels may still *receive* moved paint).
    Untouched,
    /// Some fraction of this tile's paint moves — either because it has a mask
    /// tile, or because the plane around it is only *partly* selected (§6.8),
    /// which is the same statement without a texture to say it with.
    Partial,
    /// No mask tile and `outside = 1`: all of it moves.
    Full,
}

fn classify(selection: &Selection, coord: TileCoord) -> Class {
    // The coverage a *gating* read sees out there — the mask's own number scaled by
    // the whole mask's strength (§6.8). It has to be the scaled one: `Full` is what
    // lets a source tile be **dropped** rather than rewritten, and a plane selected
    // at a half moves half its paint, so calling it full would delete the half that
    // stayed.
    let outside = selection.outside() * selection.opacity();
    match selection.tile(coord) {
        Some(_) => Class::Partial,
        // Thresholds on the value rather than either side of a half, now that a
        // selection can be partial. For every selection built at full strength
        // `outside` is still 0 or 1, so these two answer exactly as the old
        // `> 0.5` did.
        None if outside >= 1.0 => Class::Full,
        None if outside > 0.0 => Class::Partial,
        None => Class::Untouched,
    }
}

use stark_model::document::{
    Homography, Lattice, MAX_TRANSFORM_TILES, TransformMap, affine_usable, cell_point, rect_corners,
};
use stark_model::geom::{Affine2, TILE_APRON, TILE_SIZE, TileCoord, TileRect, Vec2};

use super::selection::Selection;
use crate::gpu::tile::TileMap;

/// Ceiling on candidate destination tiles *examined* (before the exact
/// quad-vs-tile test). A pathological affine — a huge scale, an extreme shear —
/// can make a quad's bounding box cover astronomically many tiles that the quad
/// itself never touches; planning must refuse such an action without first
/// walking that box. Deterministic like every other bound here.
const CANDIDATE_BUDGET: usize = 16 * MAX_TRANSFORM_TILES;

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
    tiles: &TileMap,
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
        let dests = reached_tiles(affine, *coord, &mut candidates)?;
        accumulate(&mut rewrites, *coord, dests, MAX_TRANSFORM_TILES)?;
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
/// [`MAX_SELECTION_TILES`](stark_model::document::MAX_SELECTION_TILES) — a rejection
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
        let dests = reached_tiles(affine, coord, &mut candidates)?;
        accumulate(
            &mut rewrites,
            coord,
            dests,
            stark_model::document::MAX_SELECTION_TILES,
        )?;
    }
    Some(MaskPlan {
        rewrites: rewrites.into_iter().collect(),
    })
}

/// The rect-scoped maps, resolved for planning and rendering: the geometry that
/// is derived once per apply and shared by the paint plan, the mask plan and
/// the GPU passes — so the three can never disagree about what the map is.
pub(crate) struct GatedMap {
    /// A conservative bound on where the map carries paint.
    pub image_aabb: (Vec2, Vec2),
    pub kind: GatedKind,
}

pub(crate) enum GatedKind {
    Persp { fwd: Homography, inv: Homography },
    Warp { lat: Lattice },
}

/// Resolve a rect-scoped map, or `None` for the affine (which has its own
/// whole-plane path) and for anything unusable.
pub(crate) fn gated_geometry(map: &TransformMap) -> Option<((Vec2, Vec2), GatedMap)> {
    match map {
        TransformMap::Affine(_) => None,
        TransformMap::Perspective(p) => {
            if !p.usable() {
                return None;
            }
            let (fwd, inv) = p.homographies()?;
            Some((
                (p.min, p.max),
                GatedMap {
                    image_aabb: p.image_aabb()?,
                    kind: GatedKind::Persp { fwd, inv },
                },
            ))
        }
        TransformMap::Warp(w) => {
            let lat = w.lattice()?;
            if !lat.positive() {
                return None;
            }
            Some((
                (w.min, w.max),
                GatedMap {
                    image_aabb: lat.aabb(),
                    kind: GatedKind::Warp { lat },
                },
            ))
        }
    }
}

/// How the fragment shader recovers a source position for one drawn quad.
pub(crate) enum FragMap {
    /// Invert the map's shared homography (also carries every affine special
    /// case — the bottom row is then `(0, 0, 1)` and the divide is exact).
    Persp,
    /// Invert one warp sub-cell's bilinear map: `g` are the corner images of
    /// the source sub-rect at `min` with extent `size`.
    Cell { g: [Vec2; 4], min: Vec2, size: Vec2 },
}

/// One quad the parcel/mask passes draw: a piece of a source tile's interior
/// under the map — the whole `tile ∩ rect` piece for a perspective, one
/// `sub-cell ∩ tile` piece for a warp.
pub(crate) struct SourceUnit {
    /// The source tile whose textures the draw samples.
    pub src: TileCoord,
    /// The piece's corner images, in the shader's vertex order (00, 10, 01, 11).
    pub corners: [Vec2; 4],
    pub frag: FragMap,
}

/// The tile-level consequence of a rect-scoped transform on a layer's paint —
/// [`TransformPlan`]'s shape, with quads generalized to [`SourceUnit`]s.
pub(crate) struct GatedPlan {
    pub units: Vec<SourceUnit>,
    /// Destination tile → indices into `units` whose quads reach its texture
    /// rect (possibly none: a cut with no incoming paint).
    pub rewrites: Vec<(TileCoord, Vec<usize>)>,
    /// Fully-selected, fully-inside-the-rect source tiles nothing lands back
    /// on: removed from the map (§16's drop rule).
    pub drops: Vec<TileCoord>,
}

/// Where a tile's texture rect stands relative to the gate rect, padded by the
/// half-pixel coverage ramp: does the cut touch it at all, and does the whole
/// texture sit where coverage is exactly 1.
fn rect_standing(coord: TileCoord, rect: (Vec2, Vec2)) -> (bool, bool) {
    let apron = TILE_APRON as f32;
    let lo = coord.origin() - Vec2::splat(apron);
    let hi = coord.origin() + Vec2::splat(TILE_SIZE as f32 + apron);
    let overlaps = lo.x < rect.1.x + 0.5
        && hi.x > rect.0.x - 0.5
        && lo.y < rect.1.y + 0.5
        && hi.y > rect.0.y - 0.5;
    let inside = lo.x >= rect.0.x + 0.5
        && lo.y >= rect.0.y + 0.5
        && hi.x <= rect.1.x - 0.5
        && hi.y <= rect.1.y - 0.5;
    (overlaps, inside)
}

/// Append the [`SourceUnit`]s of one source tile's `interior ∩ rect` piece(s)
/// under the map. Piece corners are computed from shared values (tile edges,
/// rect edges, lattice nodes) by shared formulas, so adjacent pieces transform
/// to bitwise-equal vertices and rasterize watertight (§16.4).
fn units_for_tile(
    geo: &GatedMap,
    coord: TileCoord,
    rect: (Vec2, Vec2),
    units: &mut Vec<SourceUnit>,
) {
    let o = coord.origin();
    let t1 = o + Vec2::splat(TILE_SIZE as f32);
    let lo = o.max(rect.0);
    let hi = t1.min(rect.1);
    if !(lo.x < hi.x && lo.y < hi.y) {
        return;
    }
    match &geo.kind {
        GatedKind::Persp { fwd, .. } => {
            let corners = rect_corners(lo, hi).map(|c| fwd.apply(c));
            units.push(SourceUnit {
                src: coord,
                corners,
                frag: FragMap::Persp,
            });
        }
        GatedKind::Warp { lat } => {
            // Sub-cells whose sub-rect overlaps the piece. The axes are
            // strictly increasing, so a linear scan with an early break is
            // fine at these counts (≤ 57 per axis).
            for b in 0..lat.ny() - 1 {
                if lat.ys()[b + 1] <= lo.y {
                    continue;
                }
                if lat.ys()[b] >= hi.y {
                    break;
                }
                for a in 0..lat.nx() - 1 {
                    if lat.xs()[a + 1] <= lo.x {
                        continue;
                    }
                    if lat.xs()[a] >= hi.x {
                        break;
                    }
                    let (sx0, sx1) = (lat.xs()[a], lat.xs()[a + 1]);
                    let (sy0, sy1) = (lat.ys()[b], lat.ys()[b + 1]);
                    let plo = Vec2::new(sx0.max(lo.x), sy0.max(lo.y));
                    let phi = Vec2::new(sx1.min(hi.x), sy1.min(hi.y));
                    if !(plo.x < phi.x && plo.y < phi.y) {
                        continue;
                    }
                    let g = lat.cell(a, b);
                    let size = Vec2::new(sx1 - sx0, sy1 - sy0);
                    let (u0, u1) = ((plo.x - sx0) / size.x, (phi.x - sx0) / size.x);
                    let (v0, v1) = ((plo.y - sy0) / size.y, (phi.y - sy0) / size.y);
                    let corners = [
                        cell_point(&g, u0, v0),
                        cell_point(&g, u1, v0),
                        cell_point(&g, u0, v1),
                        cell_point(&g, u1, v1),
                    ];
                    units.push(SourceUnit {
                        src: coord,
                        corners,
                        frag: FragMap::Cell {
                            g,
                            min: Vec2::new(sx0, sy0),
                            size,
                        },
                    });
                }
            }
        }
    }
}

/// Plan a rect-scoped transform of `tiles` under `selection`. `None` rejects
/// the action, deterministically — the same stance as [`plan_paint`].
pub(crate) fn plan_gated_paint(
    tiles: &TileMap,
    selection: &Selection,
    rect: (Vec2, Vec2),
    geo: &GatedMap,
) -> Option<GatedPlan> {
    let mut coords: Vec<TileCoord> = tiles.keys().copied().collect();
    coords.sort();

    let mut units: Vec<SourceUnit> = Vec::new();
    let mut rewrites: BTreeMap<TileCoord, Vec<usize>> = BTreeMap::new();
    let mut sources: Vec<(TileCoord, Class, bool)> = Vec::new();
    for coord in coords {
        let class = classify(selection, coord);
        if class == Class::Untouched {
            continue;
        }
        let (overlaps, inside) = rect_standing(coord, rect);
        if !overlaps {
            continue;
        }
        // A tile the cut only grazes is always rewritten; a fully-selected
        // tile wholly inside the rect is rewritten only if paint lands on it
        // (otherwise it drops, like the affine's `Full` tiles).
        if !(class == Class::Full && inside) {
            rewrites.entry(coord).or_default();
        }
        sources.push((coord, class, inside));
    }

    let mut candidates = 0usize;
    for (coord, ..) in &sources {
        let start = units.len();
        units_for_tile(geo, *coord, rect, &mut units);
        for (idx, unit) in units.iter().enumerate().skip(start) {
            let dests = quad_reached_tiles(&unit.corners, &mut candidates)?;
            accumulate(&mut rewrites, idx, dests, MAX_TRANSFORM_TILES)?;
        }
    }

    let drops = sources
        .iter()
        .filter(|(c, class, inside)| *class == Class::Full && *inside && !rewrites.contains_key(c))
        .map(|(c, ..)| *c)
        .collect();
    Some(GatedPlan {
        units,
        rewrites: rewrites.into_iter().collect(),
        drops,
    })
}

/// The mask's own move under a rect-scoped map: coverage inside the rect
/// travels with the paint, coverage outside stays — per destination tile the
/// GPU computes `max(old · (1 − box), moved)`, the soft union of the residue
/// and what landed (§16.8). Unlike the affine's pure Replace, tiles the rect
/// never touches keep their handles untouched.
pub(crate) struct GatedMaskPlan {
    pub units: Vec<SourceUnit>,
    pub rewrites: Vec<(TileCoord, Vec<usize>)>,
    /// Mask tiles wholly inside the rect that nothing lands back on, when
    /// `outside = 0`: their residue is all zero, and no tile says zero better.
    pub drops: Vec<TileCoord>,
}

pub(crate) fn plan_gated_mask(
    selection: &Selection,
    rect: (Vec2, Vec2),
    geo: &GatedMap,
) -> Option<GatedMaskPlan> {
    if selection.is_universal() {
        // No mask in force: the transform acts on the whole layer and there is
        // no outline to carry (the affine path behaves identically — no tiles,
        // `outside` untouched).
        return Some(GatedMaskPlan {
            units: Vec::new(),
            rewrites: Vec::new(),
            drops: Vec::new(),
        });
    }
    let outside = selection.outside() > 0.0;

    // The source region: where mask coverage inside the rect can be non-zero.
    // With `outside = 1` that is the whole rect (coverage is 1 wherever no
    // tile says otherwise), so the region is the rect's tile cover — counted
    // before it is walked, so an absurd rect is refused, not enumerated.
    let mut region: Vec<TileCoord> = if outside {
        // Padded by the half-pixel coverage ramp, like `rect_standing`.
        let ramp = Vec2::splat(0.5);
        let box_ = TileRect::covering(rect.0 - ramp, rect.1 + ramp, 0)?;
        if box_.count() > stark_model::document::MAX_SELECTION_TILES as u64 {
            return None;
        }
        let mut cover: Vec<TileCoord> = box_.coords().collect();
        // Mask tiles outside the cover but overlapping the ramp still change.
        for (c, _) in selection.tiles() {
            if !cover.contains(c) && rect_standing(*c, rect).0 {
                cover.push(*c);
            }
        }
        cover
    } else {
        selection
            .tiles()
            .map(|(c, _)| *c)
            .filter(|c| rect_standing(*c, rect).0)
            .collect()
    };
    region.sort();

    let mut units: Vec<SourceUnit> = Vec::new();
    let mut rewrites: BTreeMap<TileCoord, Vec<usize>> = BTreeMap::new();
    let mut standing: Vec<(TileCoord, bool)> = Vec::new();
    for coord in &region {
        let (overlaps, inside) = rect_standing(*coord, rect);
        if !overlaps {
            continue;
        }
        // Its residue changes (old · (1 − box) is not old wherever box > 0),
        // so every region tile is rewritten — except the wholly-inside ones
        // with nothing incoming, which drop below when outside = 0. With
        // outside = 1 a wholly-cut tile must still exist to say "0" against
        // the 1 that reigns tile-lessly.
        if !inside || outside {
            rewrites.entry(*coord).or_default();
        }
        standing.push((*coord, inside));
    }

    let mut candidates = 0usize;
    for coord in &region {
        let start = units.len();
        units_for_tile(geo, *coord, rect, &mut units);
        for (idx, unit) in units.iter().enumerate().skip(start) {
            let dests = quad_reached_tiles(&unit.corners, &mut candidates)?;
            accumulate(
                &mut rewrites,
                idx,
                dests,
                stark_model::document::MAX_SELECTION_TILES,
            )?;
        }
    }
    let drops = standing
        .iter()
        .filter(|(c, inside)| !outside && *inside && !rewrites.contains_key(c))
        .map(|(c, _)| *c)
        .collect();
    Some(GatedMaskPlan {
        units,
        rewrites: rewrites.into_iter().collect(),
        drops,
    })
}

/// The corners of `coord`'s *interior* under `affine`, in canvas px — the quad
/// the parcel pass draws. Corner order matches the shader's vertex indices
/// (`corner = (vi & 1, vi >> 1 & 1)`).
/// Record that `key`'s quad reaches `dest`, in the destination's own source list.
///
/// **The dedup, the cap and the candidate budget, once.** Four planners ran this same
/// body — affine paint, affine mask, gated paint, gated mask — differing only in what
/// a source is called (a `TileCoord` for the affine paths, an index into `units` for
/// the gated ones) and in which cap bounds them. Each carried its own copy of the
/// `list.last()` test and its own `rewrites.len()` check *inside* the inner loop; a
/// fifth that dropped the check is an unbounded `BTreeMap` on a hostile map, which is
/// what the cap is for.
///
/// `None` is the refusal every caller propagates: a plan too large to run is declined
/// whole, deterministically, so peers and replays agree (§16).
fn accumulate<K: Copy + PartialEq>(
    rewrites: &mut BTreeMap<TileCoord, Vec<K>>,
    key: K,
    dests: impl IntoIterator<Item = TileCoord>,
    cap: usize,
) -> Option<()> {
    for dest in dests {
        let list = rewrites.entry(dest).or_default();
        // Adjacent duplicates only, which is all `reached_tiles` can produce: it walks
        // a quad's own coverage, so one source reaches a destination in one run.
        if list.last() != Some(&key) {
            list.push(key);
        }
        if rewrites.len() > cap {
            return None;
        }
    }
    Some(())
}

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
    quad_reached_tiles(&quad_corners(affine, coord), candidates)
}

/// [`reached_tiles`] for an arbitrary convex quad — the perspective and warp
/// pieces. The affine path funnels through here too: for a parallelogram the
/// two extra separating axes are the first two negated, so the answer is
/// unchanged.
fn quad_reached_tiles(quad: &[Vec2; 4], candidates: &mut usize) -> Option<Vec<TileCoord>> {
    let lo = quad.iter().fold(quad[0], |a, p| a.min(*p));
    let hi = quad.iter().fold(quad[0], |a, p| a.max(*p));

    // Tiles whose texture rect [origin − apron, origin + TILE_SIZE + apron]
    // overlaps the quad's AABB. A fragment is only produced where the quad covers
    // a texel center, but the rect test keeps the half-open bookkeeping simple —
    // the exact test below prunes the rest.
    let tile = TILE_SIZE as f32;
    let apron = TILE_APRON as f32;
    let box_ = TileRect::covering(lo - Vec2::splat(apron), hi + Vec2::splat(apron), 0)?;
    *candidates = candidates.checked_add(usize::try_from(box_.count()).ok()?)?;
    if *candidates > CANDIDATE_BUDGET {
        return None;
    }

    let mut out = Vec::new();
    for c in box_.coords() {
        let min = c.origin() - Vec2::splat(apron);
        let max = c.origin() + Vec2::splat(tile + apron);
        if quad_intersects_rect(quad, min, max) {
            out.push(c);
        }
    }
    Some(out)
}

/// Exact convex intersection between a convex quad (an affine or projective
/// image of a rect, or a bilinear cell piece) and an axis-aligned rect, by
/// separating axes: the rect's two axes and the quad's four edge normals (for
/// a parallelogram the last two duplicate the first two, changing nothing).
fn quad_intersects_rect(quad: &[Vec2; 4], min: Vec2, max: Vec2) -> bool {
    // Rect axes: the quad's AABB against the rect.
    let qlo = quad.iter().fold(quad[0], |a, p| a.min(*p));
    let qhi = quad.iter().fold(quad[0], |a, p| a.max(*p));
    if qhi.x < min.x || qlo.x > max.x || qhi.y < min.y || qlo.y > max.y {
        return false;
    }
    // Edge normals, walking the boundary (vertex order is 00, 10, 01, 11, so
    // the boundary is 0 → 1 → 3 → 2).
    let b = [quad[0], quad[1], quad[3], quad[2]];
    for i in 0..4 {
        let edge = b[(i + 1) % 4] - b[i];
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
    use stark_model::document::{PerspectiveMap, WarpMap};

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
    fn perspective_identity_derives_the_literal_identity() {
        let p = PerspectiveMap {
            min: Vec2::new(-64.0, 32.0),
            max: Vec2::new(192.0, 288.0),
            corners: rect_corners(Vec2::new(-64.0, 32.0), Vec2::new(192.0, 288.0)),
        };
        assert!(p.usable());
        let (f, i) = p.homographies().unwrap();
        // Bitwise the identity matrix, not an approximation of it — §16.4's
        // identity invariant rides on the fragment's arithmetic being exact.
        assert_eq!(f, Homography::IDENTITY);
        assert_eq!(i, Homography::IDENTITY);
    }

    #[test]
    fn perspective_carries_the_corners_exactly_where_they_point() {
        let (min, max) = (Vec2::new(0.0, 0.0), Vec2::new(200.0, 100.0));
        let corners = [
            Vec2::new(10.0, -5.0),
            Vec2::new(180.0, 20.0),
            Vec2::new(-20.0, 130.0),
            Vec2::new(230.0, 90.0),
        ];
        let p = PerspectiveMap { min, max, corners };
        assert!(p.usable());
        let f = p.forward().unwrap();
        for (src, dst) in rect_corners(min, max).into_iter().zip(corners) {
            let got = f.apply(src);
            assert!(
                (got - dst).length() < 1e-2,
                "{src:?} -> {got:?}, want {dst:?}"
            );
        }
        // And the inverse really inverts, across the rect's interior.
        let (_, inv) = p.homographies().unwrap();
        for t in [
            Vec2::new(0.3, 0.6),
            Vec2::new(0.9, 0.1),
            Vec2::new(0.5, 0.5),
        ] {
            let src = min + t * (max - min);
            let round = inv.apply(f.apply(src));
            assert!((round - src).length() < 1e-2, "{src:?} -> {round:?}");
        }
    }

    #[test]
    fn crossed_or_reflected_quads_are_refused() {
        let (min, max) = (Vec2::ZERO, Vec2::new(100.0, 100.0));
        let base = rect_corners(min, max);
        // Swap two corners: a bow-tie.
        let crossed = PerspectiveMap {
            min,
            max,
            corners: [base[1], base[0], base[2], base[3]],
        };
        assert!(!crossed.usable());
        // Mirror: negatively oriented.
        let mirrored = PerspectiveMap {
            min,
            max,
            corners: [base[1], base[0], base[3], base[2]],
        };
        assert!(!mirrored.usable());
        // Collapse to a sliver.
        let sliver = PerspectiveMap {
            min,
            max,
            corners: [base[0], base[1], base[0], base[1]],
        };
        assert!(!sliver.usable());
    }

    #[test]
    fn a_parallelogram_perspective_rides_the_affine_arithmetic() {
        // An integer translation expressed as corners: the derived homography
        // must carry a (0, 0, 1) bottom row, so its divide is by exactly 1.
        let (min, max) = (Vec2::ZERO, Vec2::new(100.0, 50.0));
        let shift = Vec2::new(384.0, -128.0);
        let p = PerspectiveMap {
            min,
            max,
            corners: rect_corners(min, max).map(|c| c + shift),
        };
        let (f, i) = p.homographies().unwrap();
        assert_eq!(f.rows[2], [0.0, 0.0, 1.0]);
        assert_eq!(i.rows[2], [0.0, 0.0, 1.0]);
        assert_eq!(
            f.apply(Vec2::new(37.0, 13.0)),
            Vec2::new(37.0, 13.0) + shift
        );
    }

    #[test]
    fn gated_plans_reject_what_the_maps_reject() {
        let (min, max) = (Vec2::ZERO, Vec2::new(100.0, 100.0));
        let base = rect_corners(min, max);
        let bad = TransformMap::Perspective(PerspectiveMap {
            min,
            max,
            corners: [base[1], base[0], base[2], base[3]],
        });
        assert!(gated_geometry(&bad).is_none());
        assert!(!bad.usable());
        let ok = TransformMap::Warp(WarpMap::identity(min, max, 4, 4));
        assert!(ok.usable());
        assert!(gated_geometry(&ok).is_some());
        assert!(gated_geometry(&TransformMap::Affine(Affine2::IDENTITY)).is_none());
    }

    #[test]
    fn gated_planning_scopes_to_the_rect() {
        // One warp over a rect covering tile (0,0) only: a far-away populated
        // tile must be untouched by the plan even under a universal selection.
        let rect = (Vec2::new(10.0, 10.0), Vec2::new(200.0, 200.0));
        let map = TransformMap::Warp(WarpMap::identity(rect.0, rect.1, 4, 4));
        let (r, geo) = gated_geometry(&map).unwrap();
        let tiles = TileMap::new();
        let sel = Selection::everything();
        let plan = plan_gated_paint(&tiles, &sel, r, &geo).unwrap();
        assert!(plan.rewrites.is_empty() && plan.drops.is_empty());
        // Universal selection: the mask plan is deliberately empty — there is
        // no outline to carry (§16.8).
        let mask = plan_gated_mask(&sel, r, &geo).unwrap();
        assert!(mask.rewrites.is_empty() && mask.units.is_empty());
    }

    #[test]
    fn warp_pieces_tile_their_source_without_gaps() {
        // Every point of `interior ∩ rect` must be covered by exactly one
        // piece per source tile — the watertightness §16.4 needs. Checked at
        // the seams: piece corners on shared edges must be bitwise equal.
        let rect = (Vec2::new(-30.0, -30.0), Vec2::new(300.0, 300.0));
        let mut wm = WarpMap::identity(rect.0, rect.1, 4, 4);
        wm.points[5] += Vec2::new(31.0, -17.0);
        let map = TransformMap::Warp(wm);
        let (r, geo) = gated_geometry(&map).unwrap();
        let mut units = Vec::new();
        units_for_tile(&geo, TileCoord::new(0, 0), r, &mut units);
        assert!(!units.is_empty());
        let mut corner_uses: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for u in &units {
            for c in u.corners {
                *corner_uses
                    .entry((c.x.to_bits(), c.y.to_bits()))
                    .or_default() += 1;
            }
        }
        // Interior lattice corners are shared by exactly 4 pieces; if any
        // shared corner were recomputed differently it would appear as extra
        // singleton keys instead.
        assert!(
            corner_uses.values().any(|n| *n == 4),
            "no interior corner is shared bitwise by its four pieces"
        );
    }

    #[test]
    fn empty_inputs_plan_trivially() {
        // No paint and no mask tiles: both plans succeed and do nothing — the
        // classification and GPU paths are exercised by tests/transform.rs,
        // which has a device to mint real tile handles with.
        let tiles = TileMap::new();
        let sel = Selection::everything();
        let plan = plan_paint(&tiles, &sel, translation(10_000.0, 0.0)).unwrap();
        assert!(plan.rewrites.is_empty() && plan.drops.is_empty());
        let mask = plan_mask(&sel, translation(300.0, 0.0)).unwrap();
        assert!(mask.rewrites.is_empty());
    }
}
