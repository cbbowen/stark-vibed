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
//! the selection lives in [`DocState`](super::DocState) rather than in the session —
//! a stroke's pixels depend on it, so replay must be able to reconstruct it.

use serde::{Deserialize, Serialize};

use super::brush::clamp01;
use crate::geom::Vec2;

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
    pub fn coverage_outside(&self) -> f32 {
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
/// **Deserialization funnels through [`SelectionOp::at`]**, so an op that arrives
/// from a file or a peer holds the same invariants as one a gesture built:
/// non-negative feather, opacity in `0..=1`, and full strength on the unbounded
/// shape. The derived impl walked past all three — including the last, whose whole
/// doc below is an argument that the state must not exist — which left the gate
/// guarding only the caller who was already going to be reasonable. Same device as
/// [`Gradient`](crate::gradient::Gradient)'s `try_from`, and same reason: the funnel
/// is worth nothing if there is a second door.
///
/// `Raw` mirrors the fields **in order**, so the encoding is unchanged (§8) —
/// `an_op_from_the_wire_is_normalized` pins that.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "RawSelectionOp")]
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
    /// `stark-engine`'s `Selection::outside` where a shape has no boundary to rasterize. Rather
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
            // `max` rather than `clamp`, so a NaN lands on 0 — see `clamp01`.
            feather: feather.max(0.0),
            opacity: if unbounded { 1.0 } else { clamp01(opacity) },
        }
    }

    /// Select everything — i.e. *deselect*, since a selection covering the whole
    /// canvas is indistinguishable from having none (and just as cheap).
    pub fn select_all() -> Self {
        Self::new(SelectionMode::Replace, SelectionShape::All, 0.0)
    }
}

/// The wire shape of a [`SelectionOp`], which is the same shape — its only job is
/// to be the type `#[serde(from)]` deserializes *before* the constructor runs.
/// Fields in declaration order, since postcard writes them that way and nothing
/// here may change the encoding (§8).
#[derive(Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An op decoded from a file or a peer holds what the constructor promises.
    ///
    /// Three separate lies a hostile or corrupt log could tell, all of which the
    /// derived `Deserialize` used to pass straight through to a shader: a negative
    /// feather, an opacity outside `0..=1`, and — the one whose absence
    /// [`SelectionOp::opacity`] spends a paragraph justifying — a partial strength
    /// on the unbounded shape, which has no boundary to rasterize and would need a
    /// rewrite-every-tile path that does not exist.
    #[test]
    fn an_op_from_the_wire_is_normalized() {
        let wire = |op: &SelectionOp| postcard::to_allocvec(op).expect("encodes");
        let back = |b: &[u8]| postcard::from_bytes::<SelectionOp>(b).expect("decodes");

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

        // A NaN *opacity* is the one `f32::clamp` would have let through — both of
        // its comparisons against NaN are false — which is why this gate spells the
        // bound `clamp01` (`max`-then-`min`) and not `clamp`.
        let nan_strength = SelectionOp {
            opacity: f32::NAN,
            ..hot.clone()
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

    /// The mirror type must not move the encoding. Postcard writes a struct's
    /// fields in order with no names, so a field reordered in `RawSelectionOp`
    /// would decode every saved op as a different one, silently (§8) — the same
    /// failure `place_encodes_as_an_option_layer_id` guards for `Place`.
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
        let bytes = postcard::to_allocvec(&op).expect("encodes");
        // Field for field, in order, with nothing around them.
        let mut by_hand = Vec::new();
        by_hand.extend(postcard::to_allocvec(&op.mode).expect("encodes"));
        by_hand.extend(postcard::to_allocvec(&op.shape).expect("encodes"));
        by_hand.extend(postcard::to_allocvec(&op.feather).expect("encodes"));
        by_hand.extend(postcard::to_allocvec(&op.opacity).expect("encodes"));
        assert_eq!(bytes, by_hand);
    }
}
