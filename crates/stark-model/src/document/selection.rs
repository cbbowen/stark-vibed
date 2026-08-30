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
//!   (`stark-engine`'s `Selection::outside`). "No selection" is `outside = 1` with no tiles at all
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
//! the selection lives in `stark-engine`'s `DocState` rather than in the session —
//! a stroke's pixels depend on it, so replay must be able to reconstruct it.

use serde::{Deserialize, Serialize};

use crate::geom::Vec2;
use crate::{at_least_zero, clamp01};

/// Largest number of mask tiles one op may rasterize (~64 MB of R8 coverage). An op
/// that would exceed it is rejected rather than truncated — a silently clipped
/// selection is worse than none, and [`SelectionShape::All`] already expresses
/// "everything" at zero cost.
pub const MAX_SELECTION_TILES: usize = 1024;

/// Most vertices one lasso may carry. A bound on what a mask pass *costs*, where
/// [`MAX_SELECTION_TILES`] bounds how much of the canvas it covers — the two are
/// independent, and only the first was held.
///
/// Two things ride on it. The edge list is uploaded as an `N×1` texture
/// (`gpu::selection::edge_texture`), so `N` has to stay inside the smallest
/// `maxTextureDimension1D` any WebGPU adapter guarantees (8192) or the op fails
/// validation instead of rasterizing; and `selection.wesl` walks every edge **per
/// texel**, so `N` prices the pass linearly across a million-texel tile set.
///
/// The first is held where the texture is built rather than only claimed here —
/// `gpu::selection`'s `MIN_MAX_TEXTURE_DIM_1D` asserts against this constant, which is
/// why it is re-exported from `document` at all.
///
/// Four thousand is far past any loop a hand draws: the frontend decimates live input
/// to one vertex per `LASSO_MIN_STEP` (2 px), so this is a boundary some eight
/// kilopixels long. What it defends against is not a drawing but a document — a lasso
/// is a `Vec` whose length a file or a peer states.
pub const MAX_LASSO_POINTS: usize = 4096;

/// A region-producing shape, in canvas space (§6.8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
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

    /// The shape's canvas-space bounding box, or `None` when there is no box to
    /// give: the shape is unbounded ([`Self::All`]), degenerate (a lasso with no
    /// vertices), or **not measurable** (any coordinate non-finite).
    ///
    /// The third case is the one worth naming. A `None` here declines a selection op
    /// outright (`Selection::plan` returns `None`, and the caller leaves the mask
    /// alone), fills nothing on the fill path (`document::fill::plan`'s `None` arm),
    /// and claims the whole layer in a footprint (`fill_rect`) — a deterministic
    /// refusal, an empty write and an over-claim, which are the safe answers in their
    /// three directions.
    ///
    /// **The lasso is tested rather than folded**, exactly as `stroke_rect` is and for
    /// its reason (§12.6): `f32::min`/`max` return the *non*-NaN operand, so a fold
    /// steps over a bad vertex and leaves the box looking tight — it then quantizes
    /// cleanly, and the NaN reaches `selection.wesl`'s coverage ramp, where `clamp` on
    /// a NaN is unspecified. That is two clients disagreeing about a mask, which §6.8
    /// says may not happen.
    pub fn bounds(&self) -> Option<(Vec2, Vec2)> {
        let finite = |lo: Vec2, hi: Vec2| (lo.is_finite() && hi.is_finite()).then_some((lo, hi));
        match self {
            Self::All => None,
            Self::Rect { min, max } => finite(*min, *max),
            Self::Ellipse { center, radii } => {
                let r = radii.abs();
                finite(*center - r, *center + r)
            }
            Self::Lasso(points) => {
                let mut it = points.iter();
                let first = *it.next()?;
                let (lo, hi) = it.try_fold((first, first), |(lo, hi), p| {
                    p.is_finite().then(|| (lo.min(*p), hi.max(*p)))
                })?;
                finite(lo, hi)
            }
        }
    }

    /// The same shape with no more vertices than a mask pass can carry — the
    /// shape's half of the funnel [`SelectionOp::at`] and
    /// [`FillOp::with_paint`](super::fill::FillOp::with_paint) are.
    ///
    /// Only the lasso has anything to hold. The analytic shapes carry four floats
    /// each and are already answered by [`bounds`](Self::bounds), which refuses what
    /// it cannot measure rather than rounding it into a different rectangle.
    ///
    /// **Decimated rather than refused**, on `Gradient`'s argument (§19): a long
    /// loop describes a perfectly good region and simply describes it with more
    /// vertices than the pass can carry, so refusing would unload a document over
    /// something this can answer. Evenly around the cycle, which is what keeps the
    /// loop a loop — the first vertex is kept and the closing edge is implicit, so
    /// there is no end to pin the way a ramp's is.
    ///
    /// The index goes through `pick_index` because `usize` is 32 bits in the browser:
    /// spelled `i * points.len() / MAX_LASSO_POINTS` here, the product wraps on a loop
    /// past ~1.05M vertices onto a valid-but-wrong index — a different polygon than a
    /// native peer reads from the same bytes.
    pub fn sanitized(self) -> Self {
        match self {
            Self::Lasso(points) if points.len() > MAX_LASSO_POINTS => Self::Lasso(
                (0..MAX_LASSO_POINTS)
                    .map(|i| points[crate::pick_index(i, points.len(), MAX_LASSO_POINTS)])
                    .collect(),
            ),
            shape => shape,
        }
    }

    /// The coverage this shape has arbitrarily far from its bounding box: 1 for
    /// [`Self::All`], 0 for everything else.
    ///
    /// Only ever 0 or 1 even though coverage is now scaled by
    /// [`SelectionOp::opacity`] — the unbounded shape is pinned to full strength by
    /// [`SelectionOp::new`], for the reason given there.
    pub fn coverage_outside(&self) -> f32 {
        if matches!(self, Self::All) { 1.0 } else { 0.0 }
    }
}

/// How an op combines with the selection already in force (§6.8). The
/// per-texel algebra is the soft-set one, so it degrades to ordinary booleans on hard
/// edges and stays sensible on feathered ones:
/// `Replace = s`, `Union = max(p, s)`, `Subtract = p·(1−s)`, `Intersect = p·s`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
pub enum SelectionMode {
    /// Discard the running selection and take the shape's coverage.
    #[default]
    Replace,
    /// Add to it (shift-drag).
    ///
    /// `max(p, s)`, with no exception for `p = 1` — a union with the unrestricted
    /// selection *is* the unrestricted selection, and every peer rasterizing this op
    /// agrees. A *gesture* asking to add to nothing means something else, and is
    /// resolved to [`Self::Replace`] before it becomes an op (§6.8), so what reaches
    /// this enum is always what was meant.
    Union,
    /// Cut out of it (alt-drag).
    Subtract,
    /// Keep only the overlap (shift+alt-drag).
    Intersect,
}

impl SelectionMode {
    /// Combine two coverages under this mode — the CPU twin of the shader's algebra,
    /// used to carry `stark-engine`'s `Selection::outside` (where there is no tile to rasterize).
    ///
    /// Literally the soft-set expressions above, on `f32` — not a boolean twin of
    /// them. A boolean twin is sound only while every coverage in play is 0 or 1, and
    /// a partial selection ([`SelectionOp::opacity`]) makes the real algebra the only
    /// one that answers.
    pub fn combine(self, prev: f32, shape: f32) -> f32 {
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
///
/// **Deserialization funnels through [`SelectionOp::at`]**, so an op that arrives from
/// a file or a peer holds the same invariants as one a gesture built: non-negative
/// feather, opacity in `0..=1`, and full strength on the unbounded shape. Same device
/// as [`Gradient`](crate::gradient::Gradient)'s `try_from` — a funnel is worth nothing
/// if there is a second door.
///
/// `Raw` mirrors the fields **in order**, so the encoding is unchanged (§8) —
/// `an_op_from_the_wire_is_normalized` pins that.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
#[serde(from = "RawSelectionOp", into = "RawSelectionOp")]
#[carbonite(as = "RawSelectionOp")]
pub struct SelectionOp {
    mode: SelectionMode,
    shape: SelectionShape,
    /// Edge softness in canvas px: the width of the coverage ramp across the
    /// boundary. 0 still antialiases (the ramp floors at one pixel).
    feather: f32,
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
    /// `stark-engine`'s `Selection::outside` where a shape has no boundary to rasterize. Rather
    /// than grow a rewrite-every-tile path for a state the UI has no way to ask
    /// for — "select all, at a half" is not a control anywhere — the constructor
    /// refuses to build it. A partial `outside` is still reachable, by inverting a
    /// partial selection, and that path is exact.
    opacity: f32,
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
            // The shape holds its own invariant, so this gate does not have to know
            // what one is (§1).
            shape: shape.sanitized(),
            // A length, so `at_least_zero` rather than a bare `max`: the floor alone
            // lands a NaN on 0 and lets an *infinity* through, and an infinitely wide
            // coverage ramp is a half-selected plane (see there).
            feather: at_least_zero(feather, 0.0),
            opacity: if unbounded { 1.0 } else { clamp01(opacity) },
        }
    }

    /// Select everything — i.e. *deselect*, since a selection covering the whole
    /// canvas is indistinguishable from having none (and just as cheap).
    pub fn select_all() -> Self {
        Self::new(SelectionMode::Replace, SelectionShape::All, 0.0)
    }

    /// How this op combines with the selection already in force.
    pub fn mode(&self) -> SelectionMode {
        self.mode
    }

    /// The region it produces coverage from.
    pub fn shape(&self) -> &SelectionShape {
        &self.shape
    }

    /// Edge softness in canvas px — non-negative, by [`Self::at`].
    pub fn feather(&self) -> f32 {
        self.feather
    }

    /// The coverage the shape lands at where it fully covers — in `0..=1`, and
    /// exactly 1 on the unbounded shape, by [`Self::at`].
    pub fn opacity(&self) -> f32 {
        self.opacity
    }
}

/// The wire shape of a [`SelectionOp`], which is the same shape — its only job is
/// to be the type `#[serde(from)]` deserializes *before* the constructor runs.
///
/// Named in both directions by the type above, for `RawFillOp`'s reason (§8): a schema
/// describes reading and writing at once, so the representation is stated once and a
/// one-sided conversion is refused.
#[derive(Serialize, Deserialize, carbonite::Schema)]
#[serde(rename = "SelectionOp")]
struct RawSelectionOp {
    mode: SelectionMode,
    shape: SelectionShape,
    feather: f32,
    opacity: f32,
}

impl From<RawSelectionOp> for SelectionOp {
    fn from(raw: RawSelectionOp) -> Self {
        Self::at(raw.mode, raw.shape, raw.feather, raw.opacity)
    }
}

impl From<SelectionOp> for RawSelectionOp {
    /// A rename and nothing more: an op in hand already holds the invariants
    /// [`SelectionOp::at`] promises.
    fn from(op: SelectionOp) -> Self {
        Self {
            mode: op.mode,
            shape: op.shape,
            feather: op.feather,
            opacity: op.opacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An op decoded from a file or a peer holds what the constructor promises.
    ///
    /// Three separate lies a hostile or corrupt log could tell: a negative feather, an
    /// opacity outside `0..=1`, and a partial strength on the unbounded shape, which
    /// has no boundary to rasterize and would need a rewrite-every-tile path that does
    /// not exist (see [`SelectionOp::opacity`]).
    #[test]
    fn an_op_from_the_wire_is_normalized() {
        let wire = |op: &SelectionOp| carbonite::to_vec_static(op).expect("encodes");
        let back = |b: &[u8]| carbonite::from_slice_static::<SelectionOp>(b).expect("decodes");

        // Built by hand in the states the constructor refuses, then round-tripped.
        let hostile = SelectionOp {
            mode: SelectionMode::Replace,
            shape: SelectionShape::All,
            feather: -5.0,
            opacity: 0.5,
        };
        let landed = back(&wire(&hostile));
        assert_eq!(landed.feather, 0.0, "a negative feather is floored");
        assert_eq!(
            landed.opacity, 1.0,
            "the unbounded shape cannot carry a partial strength",
        );

        let hot = SelectionOp {
            mode: SelectionMode::Union,
            shape: SelectionShape::Rect {
                min: Vec2::ZERO,
                max: Vec2::splat(10.0),
            },
            feather: f32::NAN,
            opacity: 9.0,
        };
        let landed = back(&wire(&hot));
        assert_eq!(landed.opacity, 1.0, "opacity is clamped into range");
        assert_eq!(landed.feather, 0.0, "a NaN feather lands on zero, not NaN");

        // A NaN *opacity* is the one `f32::clamp` would let through, which is why
        // this gate spells the bound `clamp01` and not `clamp`.
        let nan_strength = SelectionOp {
            opacity: f32::NAN,
            ..hot
        };
        assert_eq!(back(&wire(&nan_strength)).opacity, 0.0);

        // …and an ordinary op comes through bit for bit: the funnel is idempotent
        // on anything this engine wrote, or every load would be a small edit.
        let clean = SelectionOp::at(
            SelectionMode::Subtract,
            SelectionShape::Ellipse {
                center: Vec2::splat(4.0),
                radii: Vec2::splat(9.0),
            },
            2.5,
            0.4,
        );
        assert_eq!(back(&wire(&clean)), clean);
    }

    /// **A lasso is decimated to something a mask pass can carry**, and the funnel
    /// is where that happens — the frontend's `LASSO_MIN_STEP` decimates live input
    /// and says nothing about what a file or a peer states the length of.
    ///
    /// Two things ride on the bound (see [`MAX_LASSO_POINTS`]): the edge list is
    /// uploaded as an `N×1` texture, so an over-long one fails wgpu validation
    /// instead of rasterizing, and `selection.wesl` walks every edge per texel, so
    /// `N` prices the pass across a million-texel tile set.
    #[test]
    fn a_lasso_longer_than_a_mask_pass_can_carry_is_decimated() {
        let long: Vec<Vec2> = (0..MAX_LASSO_POINTS * 3)
            .map(|i| {
                let t = i as f32 * 0.01;
                Vec2::new(t.cos() * 500.0, t.sin() * 500.0)
            })
            .collect();
        let op = SelectionOp::new(
            SelectionMode::Replace,
            SelectionShape::Lasso(long.clone()),
            0.0,
        );
        let SelectionShape::Lasso(kept) = &op.shape else {
            panic!("a lasso stays a lasso")
        };
        assert_eq!(kept.len(), MAX_LASSO_POINTS);
        assert_eq!(kept[0], long[0], "the loop still starts where it started");
        // Evenly around the cycle and in order, so the polygon keeps its shape
        // rather than collapsing onto one arc of it.
        assert!(
            kept.windows(2).all(|w| w[0] != w[1]),
            "no vertex is taken twice",
        );

        // Idempotent: a decimated loop is already short enough, so a second pass
        // through the funnel must not thin it again (§8 — a load is not an edit).
        let again = SelectionOp::new(SelectionMode::Replace, op.shape.clone(), 0.0);
        assert_eq!(again.shape, op.shape);

        // …and an ordinary loop is untouched.
        let small = SelectionShape::Lasso(long[..8].to_vec());
        assert_eq!(
            SelectionOp::new(SelectionMode::Replace, small.clone(), 0.0).shape,
            small,
        );
    }

    /// The mirror type **is** the wire shape (§8), so what is left to check is that
    /// the pair of conversions is the identity on an op the constructor already
    /// blessed — the property the funnel would otherwise quietly break.
    #[test]
    fn the_wire_shape_is_the_op_field_for_field() {
        let op = SelectionOp::at(
            SelectionMode::Intersect,
            SelectionShape::Rect {
                min: Vec2::new(1.0, 2.0),
                max: Vec2::new(3.0, 4.0),
            },
            1.5,
            0.25,
        );
        let bytes = carbonite::to_vec_static(&op).expect("encodes as its representation");
        assert_eq!(
            carbonite::from_slice_static::<SelectionOp>(&bytes).expect("decodes"),
            op,
        );
    }
}
