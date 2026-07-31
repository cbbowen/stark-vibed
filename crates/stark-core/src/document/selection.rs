//! Selections: the soft, sparse mask that gates where tools may act (§6.8).
//!
//! A selection is *not* a shape — it is a coverage field over the canvas, in exactly
//! the same sparse tile map the paint lives in. That is what makes it "flexible": a
//! rectangle, an ellipse and a freehand lasso are just three ways to produce coverage,
//! they combine with the running selection through the ordinary boolean set ops, and
//! the result is a per-texel *fraction* — so feathered and antialiased edges are the
//! normal case rather than a special one, and any future producer (select-by-colour, a
//! painted quick-mask, a loaded alpha channel) drops into the same representation.
//!
//! Two properties fall out of storing it as tiles:
//!
//! - **The infinite canvas still works.** Tiles are sparse, and the coverage that
//!   reigns where there is no tile is carried as a single flag ([`Selection::outside`]).
//!   "No selection" is `outside = 1` with no tiles at all — free — and so is its
//!   inverse, which is what lets `Invert` stay a constant-cost operation on an
//!   unbounded canvas instead of an impossible one.
//! - **History and collaboration are free.** The map is persistent (`rpds`), so a
//!   `DocState` snapshot with a selection costs the same handful of `Arc` bumps it
//!   always did, and a mask texture returns to the pool when the last version
//!   referencing it drops (§5.2).
//!
//! What is *stored in the log* is the op, not the mask: [`SelectionOp`] is a few
//! floats (or a decimated polyline), and every peer rasterizes it the same way from
//! the same shader. That keeps the action log compact and replay exact, and it is why
//! the selection lives in [`DocState`](super::DocState) rather than in the session —
//! a stroke's pixels depend on it, so replay must be able to reconstruct it.

use rpds::HashTrieMap;
use serde::{Deserialize, Serialize};

use crate::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};
use crate::gpu::tile::MaskHandle;

/// Largest number of mask tiles one op may rasterize (~64 MB of R8 coverage). An op
/// that would exceed it is rejected rather than truncated — a silently clipped
/// selection is worse than none, and [`SelectionShape::All`] already expresses
/// "everything" at zero cost.
pub const MAX_SELECTION_TILES: usize = 1024;

/// A region-producing shape, in canvas space (§6.8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SelectionShape {
    /// The whole canvas. Costs no tiles — this is how "select all" / "deselect" is
    /// expressed, and it is the only shape that can be unbounded.
    All,
    /// Axis-aligned rectangle.
    Rect { min: Vec2, max: Vec2 },
    /// Axis-aligned ellipse.
    Ellipse { center: Vec2, radii: Vec2 },
    /// A closed freehand polygon — the lasso. Decimated before it is recorded (the
    /// mask shader is O(vertices) per texel), so this stays small in the log.
    Lasso(Vec<Vec2>),
}

impl SelectionShape {
    /// A rectangle from two opposite corners, in any order.
    pub fn rect_from_corners(a: Vec2, b: Vec2) -> Self {
        Self::Rect {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// An ellipse inscribed in the rectangle spanned by two opposite corners.
    pub fn ellipse_from_corners(a: Vec2, b: Vec2) -> Self {
        let (min, max) = (a.min(b), a.max(b));
        Self::Ellipse {
            center: (min + max) * 0.5,
            radii: (max - min) * 0.5,
        }
    }

    /// The shape's canvas-space bounding box, or `None` when it is unbounded
    /// ([`Self::All`]) or degenerate (a lasso with no vertices).
    pub fn bounds(&self) -> Option<(Vec2, Vec2)> {
        match self {
            Self::All => None,
            Self::Rect { min, max } => Some((*min, *max)),
            Self::Ellipse { center, radii } => {
                let r = radii.abs();
                Some((*center - r, *center + r))
            }
            Self::Lasso(points) => {
                let mut it = points.iter();
                let first = *it.next()?;
                Some(it.fold((first, first), |(lo, hi), p| (lo.min(*p), hi.max(*p))))
            }
        }
    }

    /// The coverage this shape has arbitrarily far from its bounding box: 1 for
    /// [`Self::All`], 0 for everything else. Only ever 0 or 1, which is what keeps
    /// [`Selection::outside`] a flag rather than a value.
    fn coverage_outside(&self) -> bool {
        matches!(self, Self::All)
    }
}

/// How an op combines with the selection already in force (§6.8). The
/// per-texel algebra is the soft-set one, so it degrades to ordinary booleans on hard
/// edges and stays sensible on feathered ones:
/// `Replace = s`, `Union = max(p, s)`, `Subtract = p·(1−s)`, `Intersect = p·s`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionMode {
    /// Discard the running selection and take the shape's coverage.
    #[default]
    Replace,
    /// Add to it (shift-drag).
    Union,
    /// Cut out of it (alt-drag).
    Subtract,
    /// Keep only the overlap (shift+alt-drag).
    Intersect,
}

impl SelectionMode {
    /// The mode's discriminant as the mask shader sees it (`selection.wesl`).
    fn code(self) -> f32 {
        match self {
            Self::Replace => 0.0,
            Self::Union => 1.0,
            Self::Subtract => 2.0,
            Self::Intersect => 3.0,
        }
    }

    /// Combine two coverages under this mode — the CPU twin of the shader's algebra,
    /// used to carry [`Selection::outside`] (where there is no tile to rasterize).
    fn combine(self, prev: bool, shape: bool) -> bool {
        match self {
            Self::Replace => shape,
            Self::Union => prev || shape,
            Self::Subtract => prev && !shape,
            Self::Intersect => prev && shape,
        }
    }
}

/// One logged edit to the selection (§6.8): a shape, how it combines, and
/// how soft its edge is. Compact enough to live in the action log and on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionOp {
    pub mode: SelectionMode,
    pub shape: SelectionShape,
    /// Edge softness in canvas px: the width of the coverage ramp across the
    /// boundary. 0 still antialiases (the ramp floors at one pixel).
    pub feather: f32,
}

impl SelectionOp {
    pub fn new(mode: SelectionMode, shape: SelectionShape, feather: f32) -> Self {
        Self {
            mode,
            shape,
            feather: feather.max(0.0),
        }
    }

    /// Select everything — i.e. *deselect*, since a selection covering the whole
    /// canvas is indistinguishable from having none (and just as cheap).
    pub fn select_all() -> Self {
        Self::new(SelectionMode::Replace, SelectionShape::All, 0.0)
    }

    /// The shape/feather packed for `selection.wesl`'s uniform: `(b, c)` where `b`
    /// carries the analytic shape's parameters and `c` the kind/mode/edge count.
    pub(crate) fn shader_params(&self, edges: usize) -> ([f32; 4], [f32; 4]) {
        let (kind, b) = match &self.shape {
            SelectionShape::All => (0.0, [0.0; 4]),
            SelectionShape::Rect { min, max } => (1.0, [min.x, min.y, max.x, max.y]),
            SelectionShape::Ellipse { center, radii } => {
                (2.0, [center.x, center.y, radii.x.abs(), radii.y.abs()])
            }
            SelectionShape::Lasso(_) => (3.0, [0.0; 4]),
        };
        (b, [kind, self.mode.code(), edges as f32, 0.0])
    }
}

/// The document's selection: a sparse coverage mask plus the coverage that reigns
/// everywhere it has no tile (§6.8).
#[derive(Clone)]
pub struct Selection {
    tiles: HashTrieMap<TileCoord, MaskHandle>,
    /// Whether canvas outside [`Self::tiles`] is selected. Only ever fully in or
    /// fully out: every combine rule maps `{0,1}²` into `{0,1}`, and the only shape
    /// with non-zero coverage at infinity is `All`.
    outside: bool,
    /// A conservative analytic bounding box of the selected coverage, in canvas px
    /// — `None` when the selection is unbounded (`outside`) or its extent is not
    /// analytically known. Carried through the op algebra so the transform chrome
    /// has a rect to hang its handles on (§16); nothing about the
    /// mask itself depends on it. Conservative means coverage ⊆ hull, never that
    /// the hull is tight: `Subtract` keeps the previous hull, `Intersect`
    /// intersects boxes.
    hull: Option<(Vec2, Vec2)>,
}

impl Default for Selection {
    fn default() -> Self {
        Self::everything()
    }
}

impl Selection {
    /// The unrestricted selection: no mask tiles, everything selected. This is the
    /// state of a fresh document and of an explicit "deselect".
    pub fn everything() -> Self {
        Self {
            tiles: HashTrieMap::new(),
            outside: true,
            hull: None,
        }
    }

    /// A conservative canvas-space bounding box of the selected coverage, or
    /// `None` when the selection is unbounded or its extent is unknown — see the
    /// field docs. Chrome-facing: the transform handles anchor to it.
    pub fn hull(&self) -> Option<(Vec2, Vec2)> {
        self.hull
    }

    /// Whether nothing is masked, so tools act everywhere. Callers use this to skip
    /// the mask machinery entirely (and the UI to hide the outline).
    pub fn is_universal(&self) -> bool {
        self.tiles.is_empty() && self.outside
    }

    /// Whether the selection excludes at least part of the canvas — i.e. whether
    /// there is anything to show and to gate against.
    pub fn is_active(&self) -> bool {
        !self.is_universal()
    }

    /// Coverage where there is no mask tile, as the shaders want it.
    pub fn outside(&self) -> f32 {
        if self.outside { 1.0 } else { 0.0 }
    }

    /// This tile's mask, if the selection has one here.
    pub fn tile(&self, coord: TileCoord) -> Option<&MaskHandle> {
        self.tiles.get(&coord)
    }

    /// Every mask tile, in no particular order.
    pub fn tiles(&self) -> impl Iterator<Item = (&TileCoord, &MaskHandle)> {
        self.tiles.iter()
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.size()
    }

    /// The mask map itself — cloning it is a handful of `Arc` bumps, which is how the
    /// renderer builds the next selection on top of this one.
    pub(crate) fn tile_map(&self) -> &HashTrieMap<TileCoord, MaskHandle> {
        &self.tiles
    }

    pub(crate) fn from_parts(
        tiles: HashTrieMap<TileCoord, MaskHandle>,
        outside: bool,
        hull: Option<(Vec2, Vec2)>,
    ) -> Self {
        // The hull's meaning is "coverage ⊆ hull"; an unbounded selection has no
        // such box, whatever a caller computed.
        let hull = if outside { None } else { hull };
        Self {
            tiles,
            outside,
            hull,
        }
    }

    /// Plan the effect of `op` on this selection: which tiles have to be rasterized,
    /// which of the current ones survive untouched, and the new `outside` flag.
    /// Pure — the GPU work is [`crate::gpu::selection::SelectionRenderer::apply`].
    pub(crate) fn plan(&self, op: &SelectionOp) -> Option<SelectionPlan> {
        let shape_outside = op.shape.coverage_outside();
        let outside = op.mode.combine(self.outside, shape_outside);

        // A shape that reaches to infinity has no boundary to rasterize: the result is
        // the constant `outside` everywhere the previous mask was constant, and the
        // previous tiles survive under the combine (Union with All swallows them,
        // Intersect with All keeps them, and so on).
        if shape_outside {
            return Some(match op.mode {
                // `s = 1` everywhere ⇒ Replace/Union give all-selected, Subtract
                // all-deselected: constant, no tiles at all.
                SelectionMode::Replace | SelectionMode::Union | SelectionMode::Subtract => {
                    SelectionPlan {
                        keep_prev: false,
                        rasterize: Vec::new(),
                        outside,
                        hull: None,
                    }
                }
                // `p · 1 = p`: the selection is unchanged.
                SelectionMode::Intersect => SelectionPlan {
                    keep_prev: true,
                    rasterize: Vec::new(),
                    outside,
                    hull: self.hull,
                },
            });
        }

        let (lo, hi) = op.shape.bounds()?;
        // The hull under the soft-set algebra — conservative, per the field docs.
        // The op's coverage reaches half its (floored) feather ramp past the shape
        // boundary, so the box is padded to keep coverage ⊆ hull literal.
        let pad = Vec2::splat(op.feather.max(1.0) * 0.5);
        let (slo, shi) = (lo - pad, hi + pad);
        let hull = match op.mode {
            SelectionMode::Replace => Some((slo, shi)),
            // An unbounded previous hull stays unbounded under union.
            SelectionMode::Union => self.hull.map(|(a, b)| (a.min(slo), b.max(shi))),
            // Subtracting can only shrink coverage: the old box still contains it.
            SelectionMode::Subtract => self.hull,
            SelectionMode::Intersect => match self.hull {
                None => Some((slo, shi)),
                Some((a, b)) => Some((a.max(slo), b.min(shi))),
            },
        };
        // Pad by the feather ramp (plus the pixel of antialiasing that is always
        // there), then by a whole tile: the extra ring gives the mask a margin of
        // constant coverage, which is what lets the outline pass find the boundary by
        // differencing and keeps a feathered edge from being clipped at the tile the
        // shape happens to end in.
        let pad = op.feather.max(1.0) + TILE_APRON as f32 + 1.0;
        let rasterize = tiles_covering(lo - Vec2::splat(pad), hi + Vec2::splat(pad), 1);

        // Whether the previous mask survives outside the rasterized set. Under Union
        // and Subtract it does — the result there is `max(p, 0) = p` and `p·(1−0) = p`
        // — so those ops only rewrite the tiles the shape actually reaches. Replace
        // and Intersect collapse everything else to the constant `outside`, so the old
        // tiles are dropped rather than rewritten: a Replace over a big old selection
        // costs the new shape's tiles, not the union of both.
        let keep_prev = match op.mode {
            SelectionMode::Replace | SelectionMode::Intersect => false,
            SelectionMode::Union | SelectionMode::Subtract => true,
        };
        if rasterize.len() > MAX_SELECTION_TILES {
            return None;
        }
        Some(SelectionPlan {
            keep_prev,
            rasterize,
            outside,
            hull,
        })
    }

    /// Plan an inversion: every existing tile flips, and so does `outside`. The
    /// hull of a newly-bounded result (inverting an unbounded selection with
    /// holes) is not analytically known, so it falls back to the flipped tiles'
    /// own extent — coarse, but still a box the coverage lives inside.
    pub(crate) fn plan_invert(&self) -> SelectionPlan {
        let rasterize: Vec<TileCoord> = self.tiles.keys().copied().collect();
        let outside = !self.outside;
        let hull = (!outside)
            .then(|| {
                let mut it = rasterize.iter();
                let first = *it.next()?;
                let (lo, hi) = it.fold((first, first), |(lo, hi), c| {
                    (
                        TileCoord::new(lo.x.min(c.x), lo.y.min(c.y)),
                        TileCoord::new(hi.x.max(c.x), hi.y.max(c.y)),
                    )
                });
                let apron = Vec2::splat(TILE_APRON as f32);
                Some((
                    lo.origin() - apron,
                    hi.origin() + Vec2::splat(TILE_SIZE as f32) + apron,
                ))
            })
            .flatten();
        SelectionPlan {
            keep_prev: false,
            rasterize,
            outside,
            hull,
        }
    }
}

/// The tile-level consequence of applying an op — see [`Selection::plan`].
pub(crate) struct SelectionPlan {
    /// Whether the previous mask tiles carry over (those not in `rasterize`).
    pub keep_prev: bool,
    /// Tiles to rasterize afresh.
    pub rasterize: Vec<TileCoord>,
    pub outside: bool,
    /// The result's analytic hull — see [`Selection::hull`].
    pub hull: Option<(Vec2, Vec2)>,
}

/// Tiles whose *texture* (interior + apron) overlaps the canvas box `[lo, hi]`,
/// expanded by `ring` tiles on every side.
pub(crate) fn tiles_covering(lo: Vec2, hi: Vec2, ring: i32) -> Vec<TileCoord> {
    let tile = TILE_SIZE as f32;
    // A tile's texture starts one apron before its interior, so a box that reaches
    // into the apron band still touches the neighbour.
    let apron = TILE_APRON as f32;
    let x0 = ((lo.x - apron) / tile).floor() as i32 - ring;
    let x1 = ((hi.x + apron) / tile).floor() as i32 + ring;
    let y0 = ((lo.y - apron) / tile).floor() as i32 - ring;
    let y1 = ((hi.y + apron) / tile).floor() as i32 + ring;
    let mut out = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            out.push(TileCoord::new(x, y));
        }
    }
    out
}

/// The lasso's closed edge list, as `selection.wesl` reads it: one texel per edge
/// holding `(a.xy, b.xy)` in canvas px. Empty for a polygon that cannot enclose area.
pub(crate) fn lasso_edges(points: &[Vec2]) -> Vec<[f32; 4]> {
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

/// Drop points closer than `min_step` canvas px to the last kept one — the lasso's
/// per-texel cost is linear in vertex count, and raw pointer samples are far denser
/// than a mask boundary needs. The first and last samples are always kept, so the
/// gesture's extent survives decimation.
pub fn decimate(points: &[Vec2], min_step: f32) -> Vec<Vec2> {
    let mut out: Vec<Vec2> = Vec::with_capacity(points.len());
    for p in points {
        if out.last().is_none_or(|q| q.distance(*p) >= min_step) {
            out.push(*p);
        }
    }
    if let (Some(last), Some(end)) = (out.last().copied(), points.last().copied())
        && last != end
    {
        out.push(end);
    }
    out
}

/// The tile geometry a mask tile is rasterized over: its texture's top-left in canvas
/// px (the interior origin, shifted out by the apron — §6.4).
pub(crate) fn mask_tex_origin(coord: TileCoord) -> Vec2 {
    coord.origin() - Vec2::splat(TILE_APRON as f32)
}

/// The mask tile's edge length, for the shaders that place it in a region.
pub(crate) const MASK_TEX: u32 = TILE_TEX;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_matches_boolean_algebra() {
        for p in [false, true] {
            for s in [false, true] {
                assert_eq!(SelectionMode::Replace.combine(p, s), s);
                assert_eq!(SelectionMode::Union.combine(p, s), p || s);
                assert_eq!(SelectionMode::Subtract.combine(p, s), p && !s);
                assert_eq!(SelectionMode::Intersect.combine(p, s), p && s);
            }
        }
    }

    #[test]
    fn replacing_with_all_deselects() {
        let sel = Selection::everything();
        let plan = sel.plan(&SelectionOp::select_all()).expect("planned");
        assert!(plan.outside);
        assert!(plan.rasterize.is_empty());
    }

    #[test]
    fn rect_plan_covers_the_shape_and_a_ring() {
        let sel = Selection::everything();
        let op = SelectionOp::new(
            SelectionMode::Replace,
            SelectionShape::rect_from_corners(Vec2::new(10.0, 10.0), Vec2::new(20.0, 20.0)),
            0.0,
        );
        let plan = sel.plan(&op).expect("planned");
        // The shape sits inside tile (0,0); with the one-tile ring that is 3×3.
        assert_eq!(plan.rasterize.len(), 9);
        assert!(plan.rasterize.contains(&TileCoord::new(0, 0)));
        assert!(!plan.outside, "replace leaves everything else deselected");
    }

    #[test]
    fn oversized_shape_is_rejected() {
        let sel = Selection::everything();
        let huge = 100_000.0;
        let op = SelectionOp::new(
            SelectionMode::Replace,
            SelectionShape::rect_from_corners(Vec2::ZERO, Vec2::splat(huge)),
            0.0,
        );
        assert!(sel.plan(&op).is_none());
    }

    #[test]
    fn decimate_keeps_ends_and_thins_the_middle() {
        let pts: Vec<Vec2> = (0..100).map(|i| Vec2::new(i as f32 * 0.5, 0.0)).collect();
        let thinned = decimate(&pts, 3.0);
        assert_eq!(thinned.first(), pts.first());
        assert_eq!(thinned.last(), pts.last());
        assert!(thinned.len() < pts.len() / 4, "got {}", thinned.len());
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
