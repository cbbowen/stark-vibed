//! Filling a region with paint (§18.0.4).
//!
//! A fill is the **fifth thing a shape gesture can do**. Rect, ellipse and lasso do
//! not produce selections — they produce *coverage*, and [`SelectionMode`] is only
//! the four ways that coverage can land on the selection mask. Landing it on the
//! **paint** instead is [`ShapeAction::Fill`], and everything the four combine modes
//! already had comes with it: the same shapes, the same analytic rasterizer, the
//! same feather slider (a feathered fill is not a new feature, it is the existing
//! one pointed somewhere else).
//!
//! Two consequences worth stating, because they are what make this cheap:
//!
//! - **The selection bounds the fill.** §6.8 already makes the mask the gate
//!   every tool acts through, so a fill is gated identically to a brush stroke —
//!   which is also the answer to the wrinkle that stopped fill being built: a flood
//!   fill of an unbounded plane is undefined, and here the selection is what bounds
//!   it. A fill with *neither* a bounded shape nor a bounded selection is refused
//!   ([`plan`] returns `None`), deterministically, so peers and replays agree.
//! - **A fill deposits paint, not colour.** The parcel it lands carries the brush's
//!   own [`add`](super::BrushDynamics::add) as its height, so a filled region has
//!   real thickness: it takes the light, it can be glazed over, and a lift brush can
//!   scrape it back. It stacks by the shared parcel law (`paint_common.wesl`), the
//!   very law a stroke deposits through — so a fill cannot drift from a very slow
//!   brush covering the same area.
//!
//! Pure CPU geometry, like [`super::transform`]: [`plan`] decides *which* tiles, and
//! [`crate::gpu::fill::FillRenderer`] does the GPU work.

use serde::{Deserialize, Serialize};

use super::selection::{Selection, SelectionMode, SelectionShape, tile_box, tiles_covering};
use crate::geom::{TILE_APRON, TileCoord, Vec2};

/// Largest number of paint tiles one fill may write. The same stance and roughly
/// the same size as [`MAX_TRANSFORM_TILES`](super::transform::MAX_TRANSFORM_TILES):
/// a fill that would exceed it is refused whole rather than clipped, because a
/// silently half-filled region is worse than a refused fill — and, being a pure
/// function of the op and the mask's tile set, refused identically everywhere.
pub const MAX_FILL_TILES: usize = 1024;

/// What the next shape gesture does with the region it encloses (§6.8,
/// §18.0.4) — the "action" the Select panel's chip row picks.
///
/// One enum rather than a mode plus a flag: the chips are five answers to a single
/// question, and modelling them as five values is what keeps "exactly one is lit"
/// structural instead of a rule two pieces of state have to be kept agreeing on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeAction {
    /// Combine the region into the author's selection mask, this way.
    Select(SelectionMode),
    /// Fill the region with paint instead of selecting it.
    Fill,
}

impl Default for ShapeAction {
    fn default() -> Self {
        Self::Select(SelectionMode::default())
    }
}

impl ShapeAction {
    /// The combine mode this action selects with, or `None` if it does not select.
    pub fn mode(self) -> Option<SelectionMode> {
        match self {
            Self::Select(mode) => Some(mode),
            Self::Fill => None,
        }
    }

    /// Whether this action edits the selection — i.e. whether the marquee modifiers
    /// (shift / alt) mean anything for it.
    pub fn is_select(self) -> bool {
        matches!(self, Self::Select(_))
    }
}

/// One logged fill (§18.0.4): a region, and the parcel of paint to
/// lay in it. Compact enough for the action log and the wire, exactly like
/// [`SelectionOp`](super::SelectionOp) — the shape travels, never the tiles, and
/// every peer rasterizes it identically from the same shader.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FillOp {
    /// The region to fill. [`SelectionShape::All`] means "the selection", which is
    /// the selection bar's Fill button; it is refused when nothing is selected,
    /// since the canvas is unbounded.
    pub shape: SelectionShape,
    /// Edge softness in canvas px, read exactly as a selection op's is: 0 still
    /// antialiases.
    pub feather: f32,
    /// Straight (un-premultiplied) sRGB RGBA, like
    /// [`BrushParams::color`](super::BrushParams::color). The alpha is the paint's
    /// **per-unit opacity**, not a coverage — a thin wash and a thick one differ in
    /// `height`, not here (§6.1).
    pub color: [f32; 4],
    /// Paint **height** laid where coverage is full, taken from the brush's
    /// [`add`](super::BrushDynamics::add) so a fill and the brush in hand lay the
    /// same amount of paint. Partial coverage lays proportionally less, which is
    /// what makes a feathered edge a thinning of the paint rather than a fade of
    /// its colour.
    pub height: f32,
}

impl FillOp {
    pub fn new(shape: SelectionShape, feather: f32, color: [f32; 4], height: f32) -> Self {
        Self {
            shape,
            feather,
            color: color.map(|c| c.clamp(0.0, 1.0)),
            height: height.max(0.0),
        }
    }

    /// Fill whatever is selected — the selection bar's button. Bounded by the mask
    /// alone, so [`plan`] refuses it when there is no mask.
    pub fn of_selection(color: [f32; 4], height: f32) -> Self {
        Self::new(SelectionShape::All, 0.0, color, height)
    }

    /// How far past the shape's own boundary its coverage can reach, in canvas px.
    ///
    /// The rasterizer's ramp is `clamp(0.5 − sd/w, 0, 1)` with `w = max(feather, 1)`
    /// (`selection.wesl`), so coverage is exactly zero beyond `w/2` — and the apron
    /// carries a tile's write one band further. Tighter than
    /// [`Selection::plan`](super::Selection)'s padding on purpose: that one is sized
    /// so the *outline* pass can find a boundary by differencing, and reusing it
    /// here would ring every fill with a band of all-zero paint tiles that would
    /// then pollute `bounds` and hold pool memory.
    fn reach(&self) -> f32 {
        self.feather.max(1.0) * 0.5 + TILE_APRON as f32 + 1.0
    }
}

/// Which tiles a fill writes, given the author's selection as its gate. Sorted, so
/// the plan is deterministic (the mask's tile map iterates unordered).
///
/// `None` refuses the whole action, deterministically:
///
/// - **Unbounded** — [`SelectionShape::All`] with nothing selected. There is no
///   rectangle to fill, and picking one silently (the frame? the layer's bounds?)
///   would be a different fill on every client. This is §18.0.4's wrinkle, answered by
///   refusing rather than by inventing a boundary.
/// - **Too large** — more than [`MAX_FILL_TILES`].
///
/// A shape that encloses nothing yields an empty plan, not a refusal: a stray click
/// is a fill of nothing, which is a no-op rather than an error.
pub(crate) fn plan(op: &FillOp, gate: &Selection) -> Option<Vec<TileCoord>> {
    let bounded = gate.outside() <= 0.5;
    let mut coords: Vec<TileCoord> = match op.shape.bounds() {
        // A bounded shape: the tiles its coverage can reach, minus any the gate
        // masks out entirely. Filtering rather than letting the shader write zeros
        // is what keeps a fill inside a small selection from rewriting the whole
        // rectangle it was dragged over.
        Some((lo, hi)) => {
            let pad = Vec2::splat(op.reach());
            let (lo, hi) = (lo - pad, hi + pad);
            if bounded {
                // Walk the **gate**, not the shape's box. The two intersect to the
                // same set either way, but only the gate is bounded in advance (by
                // `MAX_SELECTION_TILES`): the box is quadratic in the drag, so a
                // rectangle swept at far zoom-out over a small selection would cost
                // millions of coordinates to describe an answer of a dozen.
                let reach = tile_box(lo, hi, 0)?;
                gate.tiles()
                    .map(|(c, _)| *c)
                    .filter(|c| reach.contains(*c))
                    .collect()
            } else {
                // Nothing selected: the shape's own cover is the only bound there
                // is, so the cap has to ride inside it (see `tiles_covering`).
                tiles_covering(lo, hi, 0, MAX_FILL_TILES)?
            }
        }
        // `All`, or a lasso with no vertices. Only the gate can bound these.
        None => {
            if !matches!(op.shape, SelectionShape::All) {
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

/// The canvas-space box a fill can write, for its
/// [`Footprint`](super::footprint::Footprint) — `None` when it is bounded only by
/// the selection, which the footprint then has to claim as the whole layer.
pub(crate) fn fill_bounds(op: &FillOp) -> Option<(Vec2, Vec2)> {
    let (lo, hi) = op.shape.bounds()?;
    let pad = Vec2::splat(op.reach());
    Some((lo - pad, hi + pad))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(lo: f32, hi: f32) -> SelectionShape {
        SelectionShape::rect_from_corners(Vec2::splat(lo), Vec2::splat(hi))
    }

    #[test]
    fn filling_the_selection_needs_a_selection() {
        let op = FillOp::of_selection([1.0; 4], 0.6);
        // Nothing selected: unbounded, and refused rather than guessed at.
        assert!(plan(&op, &Selection::everything()).is_none());
    }

    #[test]
    fn a_bounded_shape_fills_without_a_selection() {
        let op = FillOp::new(rect(0.0, 10.0), 0.0, [1.0; 4], 0.6);
        let coords = plan(&op, &Selection::everything()).expect("bounded");
        assert!(!coords.is_empty());
    }

    #[test]
    fn an_empty_lasso_fills_nothing_rather_than_failing() {
        let op = FillOp::new(SelectionShape::Lasso(Vec::new()), 0.0, [1.0; 4], 0.6);
        assert_eq!(plan(&op, &Selection::everything()), Some(Vec::new()));
    }

    #[test]
    fn an_enormous_fill_is_refused() {
        let op = FillOp::new(rect(0.0, 1.0e6), 0.0, [1.0; 4], 0.6);
        assert!(plan(&op, &Selection::everything()).is_none());
    }

    #[test]
    fn feather_widens_the_written_region_by_half_its_ramp() {
        let hard = FillOp::new(rect(0.0, 10.0), 0.0, [1.0; 4], 0.6);
        let soft = FillOp::new(rect(0.0, 10.0), 512.0, [1.0; 4], 0.6);
        let n = |op| plan(op, &Selection::everything()).expect("bounded").len();
        assert!(n(&soft) > n(&hard));
    }
}
