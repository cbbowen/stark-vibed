//! Planning a fill (§18.0.4): which tiles a [`FillOp`] writes, once the author's
//! mask has had its say.
//!
//! The op itself is `stark-model`'s `document::fill`. Only the planning is here,
//! and only because it needs the [`Selection`] — which holds tiles.

use stark_model::document::{FillOp, MAX_FILL_TILES, SelectionShape, fill_bounds};
use stark_model::geom::{TileCoord, TileRect, tiles_of};

use super::selection::Selection;

/// Which tiles a fill writes, given the author's selection as its gate. Sorted, so
/// the plan is deterministic (the mask's tile map iterates unordered).
///
/// `None` refuses the whole action, deterministically:
///
/// - **Unbounded** — [`SelectionShape::All`] with nothing selected, or with a
///   selection that reaches everywhere at *any* strength. There is no
///   rectangle to fill, and picking one silently (the frame? the layer's bounds?)
///   would be a different fill on every client. This is §18.0.4's wrinkle, answered by
///   refusing rather than by inventing a boundary.
/// - **Too large** — more than [`MAX_FILL_TILES`].
///
/// A shape that encloses nothing yields an empty plan, not a refusal: a stray click
/// is a fill of nothing, which is a no-op rather than an error.
pub(crate) fn plan(op: &FillOp, gate: &Selection) -> Option<Vec<TileCoord>> {
    let bounded = gate.outside() <= 0.0;
    let mut coords: Vec<TileCoord> = match fill_bounds(op) {
        // A bounded shape: the tiles its coverage can reach, minus any the gate
        // masks out entirely. Filtering rather than letting the shader write zeros
        // is what keeps a fill inside a small selection from rewriting the whole
        // rectangle it was dragged over.
        //
        // Quantized from [`fill_bounds`] — the same box the footprint quantizes,
        // by the same `TileRect::covering` — so the tiles this writes and the
        // tiles the action declares cannot be two different sets. See there.
        Some((lo, hi)) => {
            let reach = TileRect::covering(lo, hi, 0)?;
            if bounded {
                // Walk the **gate**, not the shape's box. The two intersect to the
                // same set either way, but only the gate is bounded in advance (by
                // `MAX_SELECTION_TILES`): the box is quadratic in the drag, so a
                // rectangle swept at far zoom-out over a small selection would cost
                // millions of coordinates to describe an answer of a dozen.
                gate.tiles()
                    .map(|(c, _)| *c)
                    .filter(|c| reach.contains(*c))
                    .collect()
            } else {
                // Nothing selected: the shape's own cover is the only bound there
                // is, so the cap has to ride inside it (see `tiles_of`).
                tiles_of(reach, MAX_FILL_TILES)?
            }
        }
        // `All`, or a lasso with no vertices. Only the gate can bound these.
        None => {
            if !matches!(op.shape(), SelectionShape::All) {
                return Some(Vec::new());
            }
            if !bounded {
                return None;
            }
            gate.tiles().map(|(c, _)| *c).collect()
        }
    };
    if coords.len() > MAX_FILL_TILES {
        return None;
    }
    coords.sort();
    Some(coords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::Srgb;
    use stark_model::document::SelectionShape;
    use stark_model::geom::Vec2;

    fn rect(lo: f32, hi: f32) -> SelectionShape {
        SelectionShape::rect_from_corners(Vec2::splat(lo), Vec2::splat(hi))
    }

    #[test]
    fn filling_the_selection_needs_a_selection() {
        let op = FillOp::of_selection(Srgb::new([1.0; 3]));
        // Nothing selected: unbounded, and refused rather than guessed at.
        assert!(plan(&op, &Selection::everything()).is_none());
    }

    #[test]
    fn a_bounded_shape_fills_without_a_selection() {
        let op = FillOp::new(rect(0.0, 10.0), 0.0, Srgb::new([1.0; 3]), 1.0);
        let coords = plan(&op, &Selection::everything()).expect("bounded");
        assert!(!coords.is_empty());
    }

    #[test]
    fn an_empty_lasso_fills_nothing_rather_than_failing() {
        let op = FillOp::new(
            SelectionShape::Lasso(Vec::new()),
            0.0,
            Srgb::new([1.0; 3]),
            1.0,
        );
        assert_eq!(plan(&op, &Selection::everything()), Some(Vec::new()));
    }

    #[test]
    fn an_enormous_fill_is_refused() {
        let op = FillOp::new(rect(0.0, 1.0e6), 0.0, Srgb::new([1.0; 3]), 1.0);
        assert!(plan(&op, &Selection::everything()).is_none());
    }

    /// **The footprint has to name every tile the plan writes** — §12.6, and the
    /// first of CLAUDE.md's rules that break silently.
    ///
    /// It did not. The two derived the box separately, and the plan's was an apron
    /// wider, so a fill whose padded bound fell within a pixel of a tile boundary
    /// wrote a tile its action never declared: a peer-diverging under-claim through
    /// the commutation gate, and a tile undo could not take back, since
    /// `patch::tile_diff` bounds the restore by the declared rect on purpose.
    ///
    /// Swept across a **whole tile stride** rather than checked at one alignment,
    /// because one alignment is precisely what hid it — `tests/footprint.rs` drives
    /// its fills at (40, 40)–(80, 80), which is nowhere near a boundary. At a
    /// quarter-pixel step several samples land inside the one-pixel window where
    /// the two answers used to differ.
    #[test]
    fn the_footprint_names_every_tile_the_plan_writes() {
        use stark_model::document::fill_rect;
        use stark_model::geom::TILE_SIZE;

        let mut steps = 0;
        for feather in [0.0, 3.0, 40.0] {
            // Off an integer index rather than an accumulator: the same positions
            // exactly (a quarter is representable), but the sweep's *end* no longer
            // depends on rounding not having drifted across 1024 additions.
            for step in 0..TILE_SIZE * 4 {
                let at = step as f32 * 0.25;
                let op = FillOp::new(
                    SelectionShape::rect_from_corners(Vec2::splat(at), Vec2::splat(at + 40.0)),
                    feather,
                    Srgb::new([1.0; 3]),
                    1.0,
                );
                let declared = fill_rect(&op);
                for c in plan(&op, &Selection::everything()).expect("bounded") {
                    assert!(
                        declared.contains(c),
                        "a fill at {at} (feather {feather}) writes {c:?}, which its \
                         footprint {declared:?} does not declare",
                    );
                }
                steps += 1;
            }
        }
        assert!(
            steps > 3000,
            "the sweep has to be fine enough to land in the gap"
        );
    }

    #[test]
    fn feather_widens_the_written_region_by_half_its_ramp() {
        let hard = FillOp::new(rect(0.0, 10.0), 0.0, Srgb::new([1.0; 3]), 1.0);
        let soft = FillOp::new(rect(0.0, 10.0), 512.0, Srgb::new([1.0; 3]), 1.0);
        let n = |op| plan(op, &Selection::everything()).expect("bounded").len();
        assert!(n(&soft) > n(&hard));
    }
}
