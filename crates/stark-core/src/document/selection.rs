//! Selections: the soft, sparse mask that gates where tools may act (§6.8).
//!
//! A selection is *not* a shape — it is a coverage field over the canvas, in exactly
//! the same sparse tile map the paint lives in. That is what makes it "flexible": a
//! rectangle, an ellipse and a freehand lasso are just three ways to produce coverage,
//! they combine with the running selection through the ordinary boolean set ops, and
//! the result is a per-texel *fraction* — so feathered and antialiased edges are the
//! normal case rather than a special one, and any future producer (select-by-color, a
//! painted quick-mask, a loaded alpha channel) drops into the same representation.
//!
//! Two properties fall out of storing it as tiles:
//!
//! - **The infinite canvas still works.** Tiles are sparse, and the coverage that
//!   reigns where there is no tile is carried as a single number
//!   ([`Selection::outside`]). "No selection" is `outside = 1` with no tiles at all
//!   — free — and so is its inverse, which is what lets `Invert` stay a
//!   constant-cost operation on an unbounded canvas instead of an impossible one.
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

use crate::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, TileRect, Vec2};
use crate::gpu::tile::{MaskHandle, MaskMap};

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
    /// [`Self::All`], 0 for everything else.
    ///
    /// Only ever 0 or 1 even though coverage is now scaled by
    /// [`SelectionOp::opacity`] — the unbounded shape is pinned to full strength by
    /// [`SelectionOp::new`], for the reason given there.
    fn coverage_outside(&self) -> f32 {
        if matches!(self, Self::All) { 1.0 } else { 0.0 }
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
    ///
    /// Literally the soft-set expressions above, on `f32` — not a boolean twin of
    /// them. A boolean twin is sound only while every coverage in play is 0 or 1, and
    /// a partial selection ([`SelectionOp::opacity`]) makes the real algebra the only
    /// one that answers.
    fn combine(self, prev: f32, shape: f32) -> f32 {
        match self {
            Self::Replace => shape,
            Self::Union => prev.max(shape),
            Self::Subtract => prev * (1.0 - shape),
            Self::Intersect => prev * shape,
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
    /// The coverage the shape lands at where it fully covers, in `0..=1` — how
    /// *strongly* this region is selected, the Select panel's Opacity slider.
    ///
    /// The mask is a coverage field and every tool already acts through it in
    /// proportion (§6.8), so a partial selection needs nothing new anywhere
    /// downstream: a brush deposits at that fraction, a fill lands at it, a
    /// transform carries it. Feather says the same thing about the *edge* and this
    /// says it about the whole region — one is a ramp, the other a level, and they
    /// multiply.
    ///
    /// Pinned to 1 for [`SelectionShape::All`], which cannot carry a strength: the
    /// unbounded shape is the deselect primitive, and its coverage lands in
    /// [`Selection::outside`] where a shape has no boundary to rasterize. Rather
    /// than grow a rewrite-every-tile path for a state the UI has no way to ask
    /// for — "select all, at a half" is not a control anywhere — the constructor
    /// refuses to build it. A partial `outside` is still reachable, by inverting a
    /// partial selection, and that path is exact.
    pub opacity: f32,
}

impl SelectionOp {
    pub fn new(mode: SelectionMode, shape: SelectionShape, feather: f32) -> Self {
        Self::at(mode, shape, feather, 1.0)
    }

    /// [`Self::new`] at a partial strength — see [`Self::opacity`].
    pub fn at(mode: SelectionMode, shape: SelectionShape, feather: f32, opacity: f32) -> Self {
        let unbounded = matches!(shape, SelectionShape::All);
        Self {
            mode,
            shape,
            feather: feather.max(0.0),
            opacity: if unbounded {
                1.0
            } else {
                opacity.clamp(0.0, 1.0)
            },
        }
    }

    /// Select everything — i.e. *deselect*, since a selection covering the whole
    /// canvas is indistinguishable from having none (and just as cheap).
    pub fn select_all() -> Self {
        Self::new(SelectionMode::Replace, SelectionShape::All, 0.0)
    }

    /// The shape/feather packed for `selection.wesl`'s uniform: `(b, c)` where `b`
    /// carries the analytic shape's parameters and `c` the kind/mode/edge
    /// count/opacity.
    pub(crate) fn shader_params(&self, edges: usize) -> ([f32; 4], [f32; 4]) {
        let (kind, b) = match &self.shape {
            SelectionShape::All => (0.0, [0.0; 4]),
            SelectionShape::Rect { min, max } => (1.0, [min.x, min.y, max.x, max.y]),
            SelectionShape::Ellipse { center, radii } => {
                (2.0, [center.x, center.y, radii.x.abs(), radii.y.abs()])
            }
            SelectionShape::Lasso(_) => (3.0, [0.0; 4]),
        };
        (b, [kind, self.mode.code(), edges as f32, self.opacity])
    }
}

/// The document's selection: a sparse coverage mask plus the coverage that reigns
/// everywhere it has no tile (§6.8).
#[derive(Clone)]
pub struct Selection {
    tiles: MaskMap,
    /// The coverage that reigns on canvas outside [`Self::tiles`].
    ///
    /// A value rather than a flag, since a selection can be partial
    /// ([`SelectionOp::opacity`]): inverting one leaves the whole plane selected at
    /// the strength the region had, which no boolean can say. Ops themselves still
    /// only ever put 0 or 1 here — the only shape with coverage at infinity is
    /// `All`, pinned to full strength — so the in-between values come from
    /// [`Self::plan_invert`] alone.
    outside: f32,
    /// The strongest coverage anywhere in the mask, and so the level whose *half*
    /// is the boundary.
    ///
    /// **Visualization, and the reflection invert needs.** The outline pass finds
    /// the contour by differencing the mask (`overlay.wesl`), which needs to know
    /// what "fully selected" means here — a selection at 0.4 has no 0.5-contour at
    /// all, and the marching ants would simply vanish. Inversion needs the same
    /// number for a different reason: the complement of a region selected at 0.4 is
    /// its outside selected at 0.4, which is `level − m` and not `1 − m`.
    ///
    /// Conservative in the same sense [`Self::hull`] is: coverage ≤ level, never
    /// that the level is reached. `Intersect` multiplies the two peaks, which is an
    /// upper bound unless they peak in the same place. A selection built only from
    /// full-strength ops has `level == 1`, and every expression below collapses to
    /// the plain hard-edged answer for it.
    level: f32,
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
            outside: 1.0,
            level: 1.0,
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
    ///
    /// A partial plane is *not* universal: `outside = 0.5` with no tiles gates every
    /// tool to half strength everywhere, which is a mask doing its job rather than
    /// the absence of one.
    pub fn is_universal(&self) -> bool {
        self.tiles.is_empty() && self.outside >= 1.0
    }

    /// Whether the selection excludes at least part of the canvas — i.e. whether
    /// there is anything to show and to gate against.
    pub fn is_active(&self) -> bool {
        !self.is_universal()
    }

    /// Coverage where there is no mask tile, as the shaders want it.
    pub fn outside(&self) -> f32 {
        self.outside
    }

    /// The level whose half the outline traces, and that inversion reflects
    /// through — see the field docs.
    pub fn level(&self) -> f32 {
        self.level
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
    pub(crate) fn tile_map(&self) -> &MaskMap {
        &self.tiles
    }

    pub(crate) fn from_parts(
        tiles: MaskMap,
        outside: f32,
        level: f32,
        hull: Option<(Vec2, Vec2)>,
    ) -> Self {
        // The hull's meaning is "coverage ⊆ hull"; a selection with coverage at
        // infinity has no such box, whatever a caller computed. *Any* coverage, not
        // just full: a plane selected at a half still reaches everywhere.
        let hull = if outside > 0.0 { None } else { hull };
        Self {
            tiles,
            outside,
            level,
            hull,
        }
    }

    /// Plan the effect of `op` on this selection: which tiles have to be rasterized,
    /// which of the current ones survive untouched, and the new `outside` flag.
    /// Pure — the GPU work is [`crate::gpu::selection::SelectionRenderer::apply`].
    pub(crate) fn plan(&self, op: &SelectionOp) -> Option<SelectionPlan> {
        let shape_outside = op.shape.coverage_outside();
        let outside = op.mode.combine(self.outside, shape_outside);
        // The result's peak. The same algebra as the coverage for three of the four
        // modes; `Subtract` is the exception, and deliberately so — subtracting
        // *removes* coverage, so what survives peaks no higher than it did, wherever
        // the op did not reach. Running `combine` here would have said a full-strength
        // Subtract flattens the level to zero, which is true only of the texels it
        // covered. See [`Self::level`] on why an upper bound is the right kind of
        // answer.
        //
        // The one corner: a region selected at opacity 0 is an empty selection at
        // level 0, and reflecting through 0 makes its inverse empty too. That is what
        // "the complement, at the strength in play" reduces to when the strength is
        // none, and it is reachable only by deliberately asking to select nothing —
        // where Deselect is the way back.
        let level = match op.mode {
            SelectionMode::Replace => op.opacity,
            SelectionMode::Union => self.level.max(op.opacity),
            SelectionMode::Subtract => self.level,
            SelectionMode::Intersect => self.level * op.opacity,
        };

        // A shape that reaches to infinity has no boundary to rasterize: the result is
        // the constant `outside` everywhere the previous mask was constant, and the
        // previous tiles survive under the combine (Union with All swallows them,
        // Intersect with All keeps them, and so on). `All` is pinned to full strength
        // ([`SelectionOp::opacity`]), which is exactly what keeps each of these four
        // constant and this branch as cheap as it always was.
        if shape_outside > 0.0 {
            return Some(match op.mode {
                // `s = 1` everywhere ⇒ Replace/Union give all-selected, Subtract
                // all-deselected: constant, no tiles at all.
                SelectionMode::Replace | SelectionMode::Union | SelectionMode::Subtract => {
                    SelectionPlan {
                        keep_prev: false,
                        rasterize: Vec::new(),
                        outside,
                        level,
                        hull: None,
                    }
                }
                // `p · 1 = p`: the selection is unchanged.
                SelectionMode::Intersect => SelectionPlan {
                    keep_prev: true,
                    rasterize: Vec::new(),
                    outside,
                    level,
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
        //
        // The cap rides *inside* the cover, not on its length: an op naming more
        // tiles than MAX_SELECTION_TILES is refused without the list ever being
        // built, which is the difference between refusing a shape the size of the
        // explored canvas and dying trying to describe it.
        let pad = op.feather.max(1.0) + TILE_APRON as f32 + 1.0;
        let rasterize = tiles_covering(
            lo - Vec2::splat(pad),
            hi + Vec2::splat(pad),
            1,
            MAX_SELECTION_TILES,
        )?;

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
        Some(SelectionPlan {
            keep_prev,
            rasterize,
            outside,
            level,
            hull,
        })
    }

    /// Plan an inversion: every existing tile reflects through [`Self::level`], and
    /// so does `outside`. The hull of a newly-bounded result (inverting an unbounded
    /// selection with holes) is not analytically known, so it falls back to the
    /// flipped tiles' own extent — coarse, but still a box the coverage lives inside.
    ///
    /// `level − m` rather than `1 − m`, so the complement of a region selected at
    /// 0.4 is its outside selected at 0.4 rather than at full strength. The two agree
    /// for any selection built at full strength, where `level == 1`.
    pub(crate) fn plan_invert(&self) -> SelectionPlan {
        let rasterize: Vec<TileCoord> = self.tiles.keys().copied().collect();
        let outside = self.level - self.outside;
        let hull = (outside <= 0.0)
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
            level: self.level,
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
    pub outside: f32,
    /// The result's peak coverage — see [`Selection::level`].
    pub level: f32,
    /// The result's analytic hull — see [`Selection::hull`].
    pub hull: Option<(Vec2, Vec2)>,
}

/// The tiles whose *texture* (interior + apron) overlaps the canvas box
/// `[lo, hi]`, grown by `ring` tiles — [`TileRect::covering`] with this module's
/// padding, since a tile's texture starts one apron before its interior and a box
/// that reaches into the apron band still touches the neighbour.
///
/// `None` — a refusal, not a clamp — for a box that is not finite or not
/// addressable. That is the only acceptable answer here: a clamp would rasterize
/// a *different* region, and these coordinates arrive from files and peers, where
/// the only tolerable disagreement between two clients is none (§6.8).
pub(crate) fn tile_box(lo: Vec2, hi: Vec2, ring: i32) -> Option<TileRect> {
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
/// Same stance and same shape as `transform::quad_reached_tiles`, which counts its
/// candidates against its own budget before enumerating, for the same reason.
pub(crate) fn tiles_covering(
    lo: Vec2,
    hi: Vec2,
    ring: i32,
    budget: usize,
) -> Option<Vec<TileCoord>> {
    let rect = tile_box(lo, hi, ring)?;
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

    /// The soft-set algebra still degrades to the booleans on hard coverages — the
    /// property that lets a feathered mask and a hard one be the same code path.
    #[test]
    fn combine_matches_boolean_algebra_on_hard_coverage() {
        let hard = |b: bool| if b { 1.0 } else { 0.0 };
        for p in [false, true] {
            for s in [false, true] {
                let (pf, sf) = (hard(p), hard(s));
                assert_eq!(SelectionMode::Replace.combine(pf, sf), hard(s));
                assert_eq!(SelectionMode::Union.combine(pf, sf), hard(p || s));
                assert_eq!(SelectionMode::Subtract.combine(pf, sf), hard(p && !s));
                assert_eq!(SelectionMode::Intersect.combine(pf, sf), hard(p && s));
            }
        }
    }

    /// The unbounded shape cannot carry a strength: `outside` is where its coverage
    /// lands, and a partial plane there would need a rewrite of every tile the
    /// selection has (see [`SelectionOp::opacity`]). The constructor is what makes
    /// that unrepresentable rather than a rule `plan` would have to remember.
    #[test]
    fn select_all_is_pinned_to_full_strength() {
        let op = SelectionOp::at(SelectionMode::Replace, SelectionShape::All, 0.0, 0.25);
        assert_eq!(op.opacity, 1.0);
        // A bounded shape keeps whatever it was given, clamped.
        let rect = SelectionShape::rect_from_corners(Vec2::ZERO, Vec2::splat(10.0));
        assert_eq!(
            SelectionOp::at(SelectionMode::Replace, rect.clone(), 0.0, 0.25).opacity,
            0.25
        );
        assert_eq!(
            SelectionOp::at(SelectionMode::Replace, rect, 0.0, 4.0).opacity,
            1.0
        );
    }

    /// Inverting reflects through the level, so the complement of a region selected
    /// at 0.4 is its outside selected at 0.4 — and inverting twice is the identity,
    /// which `1 − m` would not have been.
    #[test]
    fn inverting_a_partial_selection_keeps_its_strength() {
        let sel = Selection::from_parts(HashTrieMap::new(), 0.0, 0.4, None);
        let once = sel.plan_invert();
        assert_eq!(once.outside, 0.4);
        assert_eq!(once.level, 0.4);

        let flipped = Selection::from_parts(HashTrieMap::new(), once.outside, once.level, None);
        assert_eq!(
            flipped.plan_invert().outside,
            0.0,
            "invert is an involution"
        );
    }

    /// Subtracting cannot raise the level, and — unlike the coverage algebra — does
    /// not flatten it either: paint outside the subtracted region is still selected
    /// as strongly as it was.
    #[test]
    fn subtracting_leaves_the_level_where_it_was() {
        let half = SelectionOp::at(
            SelectionMode::Replace,
            SelectionShape::rect_from_corners(Vec2::ZERO, Vec2::splat(64.0)),
            0.0,
            0.5,
        );
        let sel = Selection::everything();
        assert_eq!(sel.plan(&half).expect("planned").level, 0.5);

        let sel = Selection::from_parts(HashTrieMap::new(), 0.0, 0.5, None);
        let cut = SelectionOp::new(
            SelectionMode::Subtract,
            SelectionShape::rect_from_corners(Vec2::splat(8.0), Vec2::splat(16.0)),
            0.0,
        );
        assert_eq!(sel.plan(&cut).expect("planned").level, 0.5);
    }

    #[test]
    fn replacing_with_all_deselects() {
        let sel = Selection::everything();
        let plan = sel.plan(&SelectionOp::select_all()).expect("planned");
        assert_eq!(plan.outside, 1.0);
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
        assert_eq!(
            plan.outside, 0.0,
            "replace leaves everything else deselected"
        );
    }

    /// The cover is *counted* before it is walked, so a box far too large to list
    /// is a refusal rather than a hang. 10⁷ canvas px is ~1.5×10⁹ tiles: the old
    /// enumerate-then-check would have pushed every one of them before finding out.
    #[test]
    fn an_astronomical_cover_is_refused_without_being_enumerated() {
        let huge = 1.0e7;
        assert_eq!(
            tiles_covering(
                Vec2::splat(-huge),
                Vec2::splat(huge),
                1,
                MAX_SELECTION_TILES
            ),
            None
        );
        // And the budget is the real bound, not a rounding of it: a cover of
        // exactly `budget` tiles is served, one past it is refused.
        let side = TILE_SIZE as f32;
        let span = |n: f32| Vec2::new(n * side - 2.0, n * side - 2.0);
        let n = |v: Vec2, b: usize| tiles_covering(Vec2::splat(2.0), v, 0, b).map(|t| t.len());
        assert_eq!(n(span(32.0), 32 * 32), Some(32 * 32));
        assert_eq!(n(span(32.0), 32 * 32 - 1), None);
    }

    /// Coordinates arrive from files and peers. A non-finite bound, or one past
    /// what the `i32` tile grid can address, is refused — never wrapped into a tile
    /// index pointing somewhere else, which is what the old `as i32` cast did.
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
