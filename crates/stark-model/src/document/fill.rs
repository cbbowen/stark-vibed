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
//!   (`stark-engine`'s `document::fill::plan` returns `None`), deterministically, so peers and replays agree.
//! - **A fill deposits paint, not color.** The parcel it lands is fully opaque
//!   paint of a real thickness — enough of it to *be* the coverage asked for
//!   ([`FillOp::opacity`]) — so a filled region takes the light, can be glazed
//!   over, and a lift brush can scrape it back. It stacks by the shared parcel law
//!   (`paint_common.wesl`), the very law a stroke deposits through.
//!
//! Pure CPU geometry, like `document::transform`: `stark-engine`'s
//! `document::fill::plan` decides *which* tiles, and `stark-engine`'s
//! `gpu::fill::FillRenderer` does the GPU work.

use serde::{Deserialize, Serialize};

use super::paint::{GradientParcel, Parcel};
use super::selection::{SelectionMode, SelectionShape};
use crate::Srgb;
use crate::geom::{TILE_APRON, Vec2};
use crate::sanitize::{at_least_zero, clamp01};

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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, carbonite::Schema)]
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
///
/// **Deserialization funnels through [`FillOp::with_paint`]**, for
/// [`SelectionOp`](super::SelectionOp)'s reason and by the same device: the
/// constructor clamps the parcel's color and the opacity, and the derived impl
/// walked past both. A fill lands paint whose mass is the *inverse* of the
/// coverage law (§6.1), so an opacity of 1.0000001 from a corrupt log is not a
/// slightly-too-strong fill — it is an infinite mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
#[serde(from = "RawFillOp", into = "RawFillOp")]
#[carbonite(as = "RawFillOp")]
pub struct FillOp {
    /// The region to fill. [`SelectionShape::All`] means "the selection", which is
    /// the selection bar's Fill button; it is refused when nothing is selected,
    /// since the canvas is unbounded.
    shape: SelectionShape,
    /// Edge softness in canvas px, read exactly as a selection op's is: 0 still
    /// antialiases.
    feather: f32,
    /// The paint to lay — one color, or a ramp read from position (§22.4).
    paint: Parcel,
    /// How strongly the fill covers where coverage is full: its **visible alpha**,
    /// in `0..=1`. The Select panel's Opacity slider, and the only strength knob a
    /// fill has.
    ///
    /// Stated as coverage rather than as a height because that is the question the
    /// control answers — *how much of what is underneath still shows?* — and
    /// because a height cannot answer it. Coverage is `1 − exp(−K·mass)` (§6.1),
    /// which approaches 1 asymptotically, so "opaque" is not a thickness anyone can
    /// pick off a slider: at the brush's own flow it is 95%, and the last 5% is a
    /// dozen more flow's worth. The shader inverts that law instead
    /// (`fill.wesl`, the same inverse `slab.wesl` merges through) and lays
    /// **fully opaque paint of exactly the mass this asks for** — so 1 covers, ½
    /// covers half, and the number on the slider is the number on the canvas.
    ///
    /// Partial coverage — a feathered edge, or the selection gating the fill —
    /// scales it, and since mass and thickness are the same thing for opaque
    /// paint, a feathered edge is still a *thinning* of the paint rather than a
    /// fade of its color.
    ///
    /// One opacity for the whole fill, gradient or not: the ramp varies the
    /// *color* of the paint, never how much of it there is — a transition in
    /// thickness would read as a lighting feature, not a color one (§22.4).
    opacity: f32,
}

impl FillOp {
    pub fn new(shape: SelectionShape, feather: f32, color: Srgb, opacity: f32) -> Self {
        Self::with_paint(shape, feather, Parcel::Solid(color), opacity)
    }

    pub fn with_paint(shape: SelectionShape, feather: f32, paint: Parcel, opacity: f32) -> Self {
        Self {
            // Both payloads hold their own invariants, so this gate states no bound
            // of its own beyond the two scalars that are the *fill's* (§1).
            shape: shape.sanitized(),
            // Floored like a selection op's: `reach` scales it into the fill's
            // claimed box, so a negative one shrinks the box the footprint promised —
            // the §12.6 direction — and a NaN one unquantizes it entirely. A length,
            // so `at_least_zero` rather than a bare `max`, which passes an infinity.
            feather: at_least_zero(feather, 0.0),
            paint: paint.sanitized(),
            opacity: clamp01(opacity),
        }
    }

    /// Fill whatever is selected — the selection bar's button. Bounded by the mask
    /// alone, so `stark-engine`'s `document::fill::plan` refuses it when there is no mask.
    ///
    /// **At full opacity, and not by default — by construction.** This fill's whole
    /// region is the selection, so how strongly it lands is already written into the
    /// mask it comes through (§6.8): the Opacity slider dims the *selection*, and a
    /// fill that dimmed itself as well would apply it twice. Taking no opacity
    /// parameter is how that is said once.
    pub fn of_selection(color: Srgb) -> Self {
        Self::new(SelectionShape::All, 0.0, color, 1.0)
    }

    /// Fill whatever is selected with a gradient — the selection bar's gradient
    /// mode (§22.4). The same bound, the same refusal and the same full strength as
    /// [`Self::of_selection`].
    pub fn gradient_of_selection(parcel: GradientParcel) -> Self {
        Self::with_paint(SelectionShape::All, 0.0, Parcel::Gradient(parcel), 1.0)
    }

    /// The region to fill.
    pub fn shape(&self) -> &SelectionShape {
        &self.shape
    }

    /// Edge softness in canvas px — non-negative, by [`Self::with_paint`].
    pub fn feather(&self) -> f32 {
        self.feather
    }

    /// The paint to lay — one color, or a ramp on an axis the pass can evaluate.
    pub fn paint(&self) -> &Parcel {
        &self.paint
    }

    /// How strongly the fill covers where coverage is full — in `0..=1`, by
    /// [`Self::with_paint`]. A fill lands paint whose mass is the *inverse* of the
    /// coverage law (§6.1), so this being in range is not a matter of degree:
    /// 1.0000001 is an infinite mass.
    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    /// The same fill, its geometry shifted whole by `by` — the shape and, for a
    /// gradient, the axis the ramp is read from, which is position too (§22.4).
    /// What brings a canvas-space gesture into the layer's frame at mint
    /// (§14.12); feather, paint and opacity are not positions and ride through.
    ///
    /// Through [`Self::with_paint`], so the result holds the op's invariants
    /// however far the shift lands — a coordinate pushed past `f32`'s horizon is
    /// then refused by the same gates an unmeasurable shape already meets.
    pub fn translated(&self, by: Vec2) -> Self {
        Self::with_paint(
            self.shape.translated(by),
            self.feather,
            self.paint.translated(by),
            self.opacity,
        )
    }

    /// How far past the shape's own boundary its coverage can reach, in canvas px.
    ///
    /// The rasterizer's ramp is `clamp(0.5 − sd/w, 0, 1)` with `w = max(feather, 1)`
    /// (`selection.wesl`), so coverage is exactly zero beyond `w/2` — and the apron
    /// carries a tile's write one band further. Tighter than `stark-engine`'s
    /// `Selection::plan` padding on purpose: that one is sized
    /// so the *outline* pass can find a boundary by differencing, and reusing it
    /// here would ring every fill with a band of all-zero paint tiles that would
    /// then pollute `bounds` and hold pool memory.
    fn reach(&self) -> f32 {
        self.feather.max(1.0) * 0.5 + TILE_APRON as f32 + 1.0
    }
}

/// The canvas-space box a fill can write: its shape's own box, grown by
/// everything the pass reaches past it — the coverage ramp (`FillOp`'s reach),
/// and then the apron band, because a tile's texture starts one apron before its
/// interior and a box reaching into that band still touches the neighbour.
///
/// `None` when the fill is bounded only by the selection — [`SelectionShape::All`],
/// or a lasso with no vertices — which `stark-engine`'s `document::fill::plan` then bounds by the mask and the
/// footprint has to claim as the whole layer.
///
/// **One box, quantized twice, and that is the whole point of this function.**
/// `stark-engine`'s `document::fill::plan` turns it into the tiles a fill writes and
/// [`fill_rect`](super::footprint::fill_rect) turns it into the tiles the action
/// declares, and those must be the same tiles: a footprint naming fewer than the plan
/// writes is the §12.6 under-claim, which diverges peers through the commutation gate
/// and — because `stark-engine`'s `document::patch::tile_diff` bounds the undo diff by
/// the rect the action declared — leaves behind on undo exactly the tiles it failed to
/// name.
///
/// The two still quantize *separately* because they answer differently for a box that
/// cannot be quantized at all — the footprint claims everything, the plan refuses —
/// which is the question `TileRect::covering` returns rather than picking. What they
/// must not do is derive the box differently, which is why it is derived here.
pub fn fill_bounds(op: &FillOp) -> Option<(Vec2, Vec2)> {
    let (lo, hi) = op.shape.bounds()?;
    let pad = Vec2::splat(op.reach() + TILE_APRON as f32);
    Some((lo - pad, hi + pad))
}

/// **The fill's wire shape**, which [`FillOp`] names in both directions —
/// `serde(from/into)` for the encoding, `carbonite(as)` for the schema (§8). So a
/// saved fill's columns and field *names* come from here, and the rename back to
/// `FillOp` is what keeps a document from recording the mirror's name.
///
/// Both directions, because a schema describes reading and writing at once and a
/// one-sided conversion is refused outright. That the conversion is `From` and never
/// `TryFrom` is the whole design — a hostile value is *clamped* on the way in, not
/// rejected, because a fill that will not open is worse than one whose opacity was
/// pulled into range.
#[derive(Serialize, Deserialize, carbonite::Schema)]
#[serde(rename = "FillOp")]
struct RawFillOp {
    shape: SelectionShape,
    feather: f32,
    paint: Parcel,
    opacity: f32,
}

impl From<RawFillOp> for FillOp {
    fn from(raw: RawFillOp) -> Self {
        Self::with_paint(raw.shape, raw.feather, raw.paint, raw.opacity)
    }
}

impl From<FillOp> for RawFillOp {
    /// The way out, which does nothing but rename: an op in hand already holds what
    /// the constructor promises, so writing it is a move, field for field.
    fn from(op: FillOp) -> Self {
        Self {
            shape: op.shape,
            feather: op.feather,
            paint: op.paint,
            opacity: op.opacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fill decoded from a file or a peer holds what the constructor promises.
    ///
    /// The opacity is the sharp one: the shader inverts the coverage law
    /// `1 − exp(−K·mass)` to find the mass that covers this much (§6.1), and that
    /// inverse runs to infinity at 1. A log carrying 1.0000001 is not a slightly
    /// strong fill, it is a `NaN` texel — and the derived impl passed it through.
    #[test]
    fn a_fill_from_the_wire_is_normalized() {
        let round = |op: &FillOp| {
            carbonite::from_slice_static::<FillOp>(&carbonite::to_vec_static(op).expect("encodes"))
                .expect("decodes")
        };
        // The **paint** is not among the hostile values here, and cannot be: a
        // `Parcel::Solid` holds an `Srgb`, which has no constructor that admits one
        // out of the cube. That half is asked of the bytes instead, in
        // `color::tests::a_color_from_the_wire_is_finite_and_bounded`.
        let hostile = FillOp {
            shape: SelectionShape::Rect {
                min: Vec2::ZERO,
                max: Vec2::splat(8.0),
            },
            feather: -3.0,
            paint: Parcel::Solid(Srgb::new([0.25, 0.5, 0.75])),
            opacity: 40.0,
        };
        let landed = round(&hostile);
        assert_eq!(landed.opacity, 1.0);
        assert_eq!(landed.feather, 0.0);

        // A NaN opacity is the one `f32::clamp` would let through, which is why this
        // gate spells the bound `clamp01`. The shader inverts the coverage law through
        // it, so a NaN there is a NaN mass on every texel of the region.
        let nan_strength = FillOp {
            opacity: f32::NAN,
            ..hostile.clone()
        };
        assert_eq!(round(&nan_strength).opacity, 0.0);

        // An infinite feather is the other half of that: `max(0.0)` floors a NaN and
        // passes an infinity, and an infinitely wide coverage ramp is a
        // half-selected plane (`crate::sanitize::at_least_zero`).
        let wide = FillOp {
            feather: f32::INFINITY,
            ..hostile
        };
        assert_eq!(round(&wide).feather, 0.0);

        // Idempotent on anything this engine wrote.
        let clean = FillOp::new(SelectionShape::All, 2.0, Srgb::new([0.25, 0.5, 0.75]), 0.6);
        assert_eq!(round(&clean), clean);
    }

    /// The mirror type **is** the wire shape, and the compiler is what says so (§8).
    ///
    /// `FillOp` declares [`RawFillOp`] as its representation in both directions, so
    /// the schema, the columns and the bytes are the mirror's — there is no second
    /// layout to drift from, and a field present on one side only will not compile.
    /// What is left to test is that the round trip is the identity on a normalized op,
    /// which is the property the two `From` impls have to have between them.
    #[test]
    fn the_wire_shape_is_the_op_field_for_field() {
        let op = FillOp::new(
            SelectionShape::Rect {
                min: Vec2::new(1.0, 2.0),
                max: Vec2::new(3.0, 4.0),
            },
            1.5,
            Srgb::new([0.1, 0.2, 0.3]),
            0.25,
        );
        let bytes = carbonite::to_vec_static(&op).expect("encodes as its representation");
        assert_eq!(
            carbonite::from_slice_static::<FillOp>(&bytes).expect("decodes"),
            op,
            "a normalized op round-trips unchanged through its mirror",
        );
    }
}
