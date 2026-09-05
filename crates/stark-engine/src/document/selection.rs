//! The selection **mask** (§6.8): the soft coverage field every tool acts through,
//! and the plan that rasterizes an edit to it.
//!
//! The **op** — what an edit to the mask says, which is what the log carries and
//! what peers replay — is `stark-model`'s `document::selection`. This side is the
//! field the op produces, and it holds tiles.

use rpds::HashTrieMap;

use stark_model::document::{MAX_SELECTION_TILES, SelectionMode, SelectionOp};
use stark_model::geom::{TILE_APRON, TileCoord, Vec2, tiles_covering};

use crate::gpu::tile::{MaskHandle, MaskMap};

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
    /// The **whole** mask's opacity, on top of the shape arithmetic (§6.8) — the
    /// selection bar's Opacity slider, and what
    /// [`ActionKind::SetSelectionOpacity`](stark_model::document::ActionKind) sets.
    ///
    /// Not in the tiles, and that is the point. The mask holds whatever the ops
    /// made it; this says how strongly it is *read*, so moving it costs no
    /// rasterization and applies to a region already drawn — which is the whole
    /// feature. [`SelectionOp::opacity`] is the same question asked of one shape,
    /// baked into that shape's coverage where it was struck; the two multiply.
    ///
    /// What a reader does with the product is the reader's law, and there are two
    /// (§6.8): paint that is *minted* takes it as the other factor of the brush's
    /// opacity ceiling — `stroke_constants` folds it into the stroke's ceiling, and
    /// a fill's stated coverage is scaled by it — and paint that is *moved* takes it
    /// as the fraction of height that moves (the transform's cut, the loop's lift
    /// and deposit of carried paint).
    ///
    /// Carried through the op algebra — an op says where the mask is, never how
    /// strongly it is read — so a region drawn over a dimmed selection is dimmed
    /// too, and a universal mask keeps the number as well: set before anything is
    /// selected, it is the strength the coming region will take, and until one is
    /// drawn it is the whole canvas taking paint at that strength — the dial's
    /// other factor, everywhere. The one op that resets it is a **deselect**,
    /// `Replace` with `All` ([`Self::plan`]), which hands the canvas back at full
    /// strength so a dimming never outlives its selection by accident.
    opacity: f32,
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
            opacity: 1.0,
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

    /// Whether this is [`Self::everything`] exactly — nothing masked *and* full
    /// strength: the state of a fresh document, the one a deselect lands on, and
    /// the one `DocState` does not store. A universal mask read below 1 is not
    /// it: it masks nothing and gates everything, which is a state with a number
    /// to keep (see [`opacity`](Self::opacity)).
    pub fn is_everything(&self) -> bool {
        self.is_universal() && self.opacity >= 1.0
    }

    /// Coverage where there is no mask tile, as the shaders want it. The mask's own
    /// number: [`Self::opacity`] is not folded in, because the two kinds of reader
    /// multiply it in under different laws (see there).
    pub fn outside(&self) -> f32 {
        self.outside
    }

    /// The whole mask's opacity — see the field docs. What the selection bar's slider
    /// shows, and what every gating reader multiplies its coverage by — under a
    /// universal mask too, where it is the dial's other factor everywhere.
    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    /// The same selection read at `opacity` — [`ActionKind::SetSelectionOpacity`](stark_model::document::ActionKind)'s
    /// whole effect. No tile moves, which is what makes it retroactive.
    ///
    /// Unclamped here: the number arrives through `ActionKind::sanitized`, the one
    /// funnel an action passes into the document through, and a second bound would
    /// be a second policy to keep in step (§8). A universal mask takes it like any
    /// other — see the field.
    pub(crate) fn with_opacity(&self, opacity: f32) -> Self {
        Self::from_parts(
            self.tiles.clone(),
            self.outside,
            self.level,
            opacity,
            self.hull,
        )
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

    /// Whether two selections are the same state, for the fold's audit (§12.6).
    ///
    /// **Written as a destructuring `let`, and that is the whole point.** This is
    /// what `document::audit` asks to decide whether a selection differed, so it is
    /// the one place that has to learn about a new field — and a `let` over the
    /// whole struct stops compiling until somebody teaches it, where a chain of
    /// accessor comparisons would silently keep answering about the fields it already
    /// knew.
    ///
    /// `hull` is deliberately *not* compared, and the exclusion is a decision rather
    /// than an omission: it is a conservative box carried through the op algebra —
    /// `Subtract` keeps the previous one, `Intersect` intersects — so it is
    /// path-dependent by construction, and nothing about the mask depends on it
    /// (§16 hangs transform handles on it and that is all). `level` *is* compared,
    /// conservative in the same sense but load-bearing: it decides the outline
    /// contour and the reflection [`plan_invert`](Self::plan_invert) takes.
    pub(crate) fn same(&self, other: &Self) -> bool {
        let Self {
            tiles,
            outside,
            level,
            opacity,
            hull: _,
        } = self;
        *outside == other.outside
            && *level == other.level
            && *opacity == other.opacity
            && tiles.size() == other.tiles.size()
            && tiles
                .iter()
                .all(|(c, h)| other.tiles.get(c).is_some_and(|o| o.same(h)))
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
        opacity: f32,
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
            opacity,
            hull,
        }
    }

    /// Plan the effect of `op` on this selection: which tiles have to be rasterized,
    /// which of the current ones survive untouched, and the new `outside` flag.
    /// Pure — the GPU work is [`crate::gpu::selection::SelectionRenderer::apply`].
    pub(crate) fn plan(&self, op: &SelectionOp) -> Option<SelectionPlan> {
        let shape_outside = op.shape().coverage_outside();
        let outside = op.mode().combine(self.outside, shape_outside);
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
        let level = match op.mode() {
            SelectionMode::Replace => op.opacity(),
            SelectionMode::Union => self.level.max(op.opacity()),
            SelectionMode::Subtract => self.level,
            SelectionMode::Intersect => self.level * op.opacity(),
        };

        // A shape that reaches to infinity has no boundary to rasterize: the result is
        // the constant `outside` everywhere the previous mask was constant, and the
        // previous tiles survive under the combine (Union with All swallows them,
        // Intersect with All keeps them, and so on). `All` is pinned to full strength
        // ([`SelectionOp::opacity`]), which is exactly what keeps each of these four
        // constant and this branch as cheap as it always was.
        if shape_outside > 0.0 {
            return Some(match op.mode() {
                // `s = 1` everywhere ⇒ Replace/Union give all-selected, Subtract
                // all-deselected: constant, no tiles at all.
                SelectionMode::Replace | SelectionMode::Union | SelectionMode::Subtract => {
                    SelectionPlan {
                        keep_prev: false,
                        rasterize: Vec::new(),
                        outside,
                        level,
                        // A deselect — Replace with All, the one op whose whole
                        // meaning is "hand the canvas back" — lands on full
                        // strength; the opacity rides through every other op
                        // (see the field). Decided here rather than in the fold
                        // so replay, undo and a peer's copy agree by construction.
                        opacity: if op.mode() == SelectionMode::Replace {
                            1.0
                        } else {
                            self.opacity
                        },
                        hull: None,
                    }
                }
                // `p · 1 = p`: the selection is unchanged.
                SelectionMode::Intersect => SelectionPlan {
                    keep_prev: true,
                    rasterize: Vec::new(),
                    outside,
                    level,
                    opacity: self.opacity,
                    hull: self.hull,
                },
            });
        }

        let (lo, hi) = op.shape().bounds()?;
        // The hull under the soft-set algebra — conservative, per the field docs.
        // The op's coverage reaches half its (floored) feather ramp past the shape
        // boundary, so the box is padded to keep coverage ⊆ hull literal.
        let pad = Vec2::splat(op.feather().max(1.0) * 0.5);
        let (slo, shi) = (lo - pad, hi + pad);
        let hull = match op.mode() {
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
        let pad = op.feather().max(1.0) + TILE_APRON as f32 + 1.0;
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
        let keep_prev = match op.mode() {
            SelectionMode::Replace | SelectionMode::Intersect => false,
            SelectionMode::Union | SelectionMode::Subtract => true,
        };
        Some(SelectionPlan {
            keep_prev,
            rasterize,
            outside,
            level,
            opacity: self.opacity,
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
        // Sorted, because the map is an `rpds::HashTrieMap` under a `RandomState`
        // and so hands its keys out in an order that differs run to run. No pixel
        // depends on it — each tile is an independent clear-and-draw, and the hull
        // below is a fold that does not care — but every other planner names its
        // tiles in the row-major order `TileRect::coords` walks, and an invariant
        // that holds everywhere is worth more than the sort costs on a set bounded
        // by `MAX_SELECTION_TILES`.
        //
        // By `(y, x)` rather than `TileCoord`'s own `Ord`, which orders by x first:
        // the point is to name the same order the covering planners do, and that one
        // is y-outer.
        let mut rasterize: Vec<TileCoord> = self.tiles.keys().copied().collect();
        rasterize.sort_unstable_by_key(|c| (c.y, c.x));
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
                Some((lo.texture_box().0, hi.texture_box().1))
            })
            .flatten();
        SelectionPlan {
            keep_prev: false,
            rasterize,
            outside,
            level: self.level,
            opacity: self.opacity,
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
    /// The result's overall opacity — the previous one, always. An op says where
    /// the mask is; how strongly it is read is a separate question with a separate
    /// action (§6.8), and carrying it here is what keeps a region redrawn over a
    /// dimmed selection dimmed. (A deselect lands on 1 all the same — [`Selection::plan`]
    /// pins `Replace` there, which is the only op that hands the canvas back.)
    pub opacity: f32,
    /// The result's analytic hull — see [`Selection::hull`].
    pub hull: Option<(Vec2, Vec2)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::SelectionShape;
    use stark_model::geom::TILE_SIZE;

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
        assert_eq!(op.opacity(), 1.0);
        // A bounded shape keeps whatever it was given, clamped.
        let rect = SelectionShape::rect_from_corners(Vec2::ZERO, Vec2::splat(10.0));
        assert_eq!(
            SelectionOp::at(SelectionMode::Replace, rect.clone(), 0.0, 0.25).opacity(),
            0.25
        );
        assert_eq!(
            SelectionOp::at(SelectionMode::Replace, rect, 0.0, 4.0).opacity(),
            1.0
        );
    }

    /// Inverting reflects through the level, so the complement of a region selected
    /// at 0.4 is its outside selected at 0.4 — and inverting twice is the identity,
    /// which `1 − m` would not have been.
    #[test]
    fn inverting_a_partial_selection_keeps_its_strength() {
        let sel = Selection::from_parts(HashTrieMap::new(), 0.0, 0.4, 1.0, None);
        let once = sel.plan_invert();
        assert_eq!(once.outside, 0.4);
        assert_eq!(once.level, 0.4);

        let flipped =
            Selection::from_parts(HashTrieMap::new(), once.outside, once.level, 1.0, None);
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

        let sel = Selection::from_parts(HashTrieMap::new(), 0.0, 0.5, 1.0, None);
        let cut = SelectionOp::new(
            SelectionMode::Subtract,
            SelectionShape::rect_from_corners(Vec2::splat(8.0), Vec2::splat(16.0)),
            0.0,
        );
        assert_eq!(sel.plan(&cut).expect("planned").level, 0.5);
    }

    /// A deselect is the one op that resets the mask's opacity; a universal mask
    /// otherwise keeps whatever it was read at, that being the strength the next
    /// region will take (§6.8).
    #[test]
    fn replacing_with_all_deselects_and_lands_on_full_strength() {
        let sel = Selection::everything();
        let plan = sel.plan(&SelectionOp::select_all()).expect("planned");
        assert_eq!(plan.outside, 1.0);
        assert!(plan.rasterize.is_empty());

        let dimmed = Selection::everything().with_opacity(0.4);
        assert!(dimmed.is_universal() && !dimmed.is_everything());
        assert_eq!(dimmed.opacity(), 0.4, "a universal mask keeps its opacity");
        assert_eq!(
            dimmed
                .plan(&SelectionOp::select_all())
                .expect("planned")
                .opacity,
            1.0,
            "a deselect lands on full strength"
        );
        let all = SelectionOp::new(SelectionMode::Union, SelectionShape::All, 0.0);
        assert_eq!(
            dimmed.plan(&all).expect("planned").opacity,
            0.4,
            "only a deselect resets it"
        );
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
}
