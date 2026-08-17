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
//! - **A fill deposits paint, not color.** The parcel it lands is fully opaque
//!   paint of a real thickness — enough of it to *be* the coverage asked for
//!   ([`FillOp::opacity`]) — so a filled region takes the light, can be glazed
//!   over, and a lift brush can scrape it back. It stacks by the shared parcel law
//!   (`paint_common.wesl`), the very law a stroke deposits through.
//!
//! Pure CPU geometry, like [`super::transform`]: [`plan`] decides *which* tiles, and
//! [`crate::gpu::fill::FillRenderer`] does the GPU work.

use serde::{Deserialize, Serialize};

use super::selection::{SelectionMode, SelectionShape};
use crate::geom::{TILE_APRON, Vec2};
use crate::gradient::Gradient;

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

/// What paint a fill lays: the same parcel everywhere, or one that varies with
/// canvas position (§22.4). This is the seam §18.0.4 named — a gradient is not a
/// new pipeline, it is a fill whose parcel reads its latent from position — so
/// the region, the gate, the stacking law and the footprint are all [`FillOp`]'s,
/// untouched.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Parcel {
    /// One color everywhere. Straight sRGB, and **color only**: how strongly a
    /// fill covers is [`FillOp::opacity`], one number for the whole fill, so a
    /// parcel says *what* paint and never *how much* of it (§6.1).
    Solid([f32; 3]),
    /// A color ramp read from canvas position (§22.4).
    Gradient(GradientParcel),
}

/// The gradient half of a [`Parcel`]: which ramp, along what axis (§22.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradientParcel {
    /// The ramp — embedded **by value**, the way a stroke embeds its brush
    /// color, so the document stays self-contained and replayable with no
    /// reference into anyone's browser-local library (§22.3).
    pub gradient: Gradient,
    /// Where `t = 0` and `t = 1` sit on the canvas.
    pub axis: GradientAxis,
}

/// The geometry mapping canvas position to ramp position — the shape the
/// composing drag draws (§22.4). Beyond either end the ramp holds its end stop:
/// a gradient fill covers its whole region, the axis only says where the
/// transition lives.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GradientAxis {
    /// `t` grows from `from` to `to` along the line joining them, constant on
    /// perpendiculars.
    Linear { from: Vec2, to: Vec2 },
    /// `t` grows with distance from `center`, reaching 1 at `radius`.
    Radial { center: Vec2, radius: f32 },
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
    /// The paint to lay — one color, or a ramp read from position (§22.4).
    pub paint: Parcel,
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
    pub opacity: f32,
}

impl FillOp {
    pub fn new(shape: SelectionShape, feather: f32, color: [f32; 3], opacity: f32) -> Self {
        Self::with_paint(shape, feather, Parcel::Solid(color), opacity)
    }

    pub fn with_paint(shape: SelectionShape, feather: f32, paint: Parcel, opacity: f32) -> Self {
        let paint = match paint {
            Parcel::Solid(c) => Parcel::Solid(c.map(|c| c.clamp(0.0, 1.0))),
            gradient => gradient,
        };
        Self {
            shape,
            feather,
            paint,
            opacity: opacity.clamp(0.0, 1.0),
        }
    }

    /// Fill whatever is selected — the selection bar's button. Bounded by the mask
    /// alone, so [`plan`] refuses it when there is no mask.
    ///
    /// **At full opacity, and not by default — by construction.** This fill's whole
    /// region is the selection, so how strongly it lands is already written into the
    /// mask it comes through (§6.8): the Opacity slider dims the *selection*, and a
    /// fill that dimmed itself as well would apply it twice. Taking no opacity
    /// parameter is how that is said once.
    pub fn of_selection(color: [f32; 3]) -> Self {
        Self::new(SelectionShape::All, 0.0, color, 1.0)
    }

    /// Fill whatever is selected with a gradient — the selection bar's gradient
    /// mode (§22.4). The same bound, the same refusal and the same full strength as
    /// [`Self::of_selection`].
    pub fn gradient_of_selection(parcel: GradientParcel) -> Self {
        Self::with_paint(SelectionShape::All, 0.0, Parcel::Gradient(parcel), 1.0)
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

/// The canvas-space box a fill can write: its shape's own box, grown by
/// everything the pass reaches past it — the coverage ramp ([`FillOp::reach`]),
/// and then the apron band, because a tile's texture starts one apron before its
/// interior and a box reaching into that band still touches the neighbour.
///
/// `None` when the fill is bounded only by the selection — [`SelectionShape::All`],
/// or a lasso with no vertices — which [`plan`] then bounds by the mask and the
/// footprint has to claim as the whole layer.
///
/// **One box, quantized twice, and that is the whole point of this function.**
/// [`plan`] turns it into the tiles a fill writes and
/// [`fill_rect`](super::footprint::fill_rect) turns it into the tiles the action
/// declares, and those must be the same tiles: a footprint naming fewer than the
/// plan writes is the §12.6 under-claim, which diverges peers through the
/// commutation gate and — because `patch::paint_rect` deliberately bounds the undo
/// diff by the very rect the action declared — leaves behind on undo exactly the
/// tiles it failed to name.
///
/// They were two boxes until they weren't. `plan` padded by the apron (through
/// `selection::tile_box`) and the footprint did not, from identical inputs, so the
/// plan named a tile the footprint omitted whenever the padded bound fell within a
/// pixel of a tile boundary — which is every fill aligned to the tile grid. That is
/// the drift `TileRect::covering` was introduced to end, surviving one wrapper out;
/// the fix is not a matching pad in the second place but the removal of the second
/// place. The two still quantize *separately* because they answer differently for a
/// box that cannot be quantized at all — the footprint claims everything, the plan
/// refuses — which is the question `covering` returns rather than picking.
pub fn fill_bounds(op: &FillOp) -> Option<(Vec2, Vec2)> {
    let (lo, hi) = op.shape.bounds()?;
    let pad = Vec2::splat(op.reach() + TILE_APRON as f32);
    Some((lo - pad, hi + pad))
}
