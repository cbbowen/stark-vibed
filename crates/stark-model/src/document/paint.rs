//! **What paint to lay**: one color everywhere, or a ramp read from canvas position
//! (§22.4).
//!
//! Its own module because two actions lay paint and neither owns the answer. A
//! [`Fill`](super::ActionKind::Fill) lands a [`Parcel`] through `fill.wesl`; a matte
//! layer stands in one through `matte.wesl` (§15.4). Both reach the same
//! `ramp_common::ramp_position`, both take their colors from the same picker, and
//! both are gated by the same `sanitized`.
//!
//! # It was two types
//!
//! `MattePaint` was a second enum with the same two cases, and its `sanitized` was
//! word for word this one's — under a comment saying so, and saying why it had not
//! been merged: the two "reached the file at different times and their wire shapes
//! were written differently", so merging was a save-format change and worth doing on
//! its own rather than as a side effect. This is that change, taken while §19's beta
//! rung is unclaimed and the format is still free to move.
//!
//! What it buys is not the deleted lines. It is that a matte and a fill can no longer
//! *diverge*: the gap that comment was written to explain had been open long enough
//! for one of the two to clamp a solid color and hand a ramp's forty-eight floats
//! through untouched, from the same picker into the same texel.

use serde::{Deserialize, Serialize};

use crate::Srgb;
use crate::geom::Vec2;
use crate::gradient::Gradient;

/// What paint to lay: the same parcel everywhere, or one that varies with canvas
/// position (§22.4). This is the seam §18.0.4 named — a gradient is not a
/// new pipeline, it is a fill whose parcel reads its latent from position — so
/// the region, the gate, the stacking law and the footprint are all [`FillOp`](super::FillOp)'s,
/// untouched.
///
/// # What a matte adds, which is nothing
///
/// A matte layer stands in a parcel too, and the one thing it does differently is
/// carry no strength of its own: a matte's transparency is its *layer* opacity
/// (§15.3) and its paint is a full-strength coat. That is why a solid keeps three
/// channels rather than four and why nothing here has a per-unit opacity. A fill
/// states its strength in [`FillOp::opacity`](super::FillOp::opacity) instead, one
/// number for the whole fill — so this type says **what** paint, and each action says
/// how much of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum Parcel {
    /// One color everywhere. Straight sRGB, and **color only**: how strongly a
    /// fill covers is [`FillOp::opacity`](super::FillOp::opacity), one number for the whole fill, so a
    /// parcel says *what* paint and never *how much* of it (§6.1).
    Solid(Srgb),
    /// A color ramp read from canvas position (§22.4).
    Gradient(GradientParcel),
}

impl Parcel {
    /// The color a one-swatch summary shows: the solid itself, or the ramp's start —
    /// the stop the axis anchors on.
    ///
    /// A parcel that cannot say *where* its transition goes still knows exactly what
    /// color it starts from, which is the same reading [`sanitized`](Self::sanitized)
    /// gives an unusable axis.
    pub fn swatch(&self) -> Srgb {
        match self {
            Self::Solid(c) => *c,
            Self::Gradient(GradientParcel { gradient, .. }) => gradient.sample(0.0),
        }
    }

    /// The same paint with every color inside the sRGB cube and an axis the ramp pass
    /// can evaluate — the parcel's half of the funnel
    /// [`FillOp::with_paint`](super::FillOp::with_paint) is, and of
    /// [`ActionKind::sanitized`](super::ActionKind::sanitized) where a matte's paint
    /// comes through.
    ///
    /// Held here rather than at those gates because it is a fact about *paint*, and
    /// there is now one kind of paint for both of them to lay. It was written twice —
    /// once for a fill, once for a matte, word for word — and before that `FillOp`
    /// clamped a `Solid` and passed a `Gradient` through untouched: a solid's three
    /// floats guarded and a ramp's forty-eight not, from the same picker, into the
    /// same texel. Two copies of a gate is how that happens twice.
    ///
    /// **An unusable axis degrades the parcel to the ramp's anchor**, rather than
    /// being clamped into a different axis or refusing the fill. That is the honest
    /// reading of a gradient nobody can place: `swatch` already calls the first stop
    /// "the stop the axis anchors on", so a parcel that cannot say *where* the
    /// transition goes still knows exactly what color it starts from. Deterministic,
    /// and it cannot make a `NaN`.
    pub fn sanitized(self) -> Self {
        match self {
            // Nothing to hold: an `Srgb` is inside the cube by construction, and a
            // ramp's stops are `Srgb`s. What is left is the *axis*, which is the
            // one thing here a type could not answer.
            Self::Solid(_) => self,
            Self::Gradient(GradientParcel { gradient, axis }) if axis.usable() => {
                Self::Gradient(GradientParcel { gradient, axis })
            }
            Self::Gradient(GradientParcel { gradient, .. }) => Self::Solid(gradient.sample(0.0)),
        }
    }
}

/// The gradient half of a [`Parcel`]: which ramp, along what axis (§22.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
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
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum GradientAxis {
    /// `t` grows from `from` to `to` along the line joining them, constant on
    /// perpendiculars.
    Linear { from: Vec2, to: Vec2 },
    /// `t` grows with distance from `center`, reaching 1 at `radius`.
    Radial { center: Vec2, radius: f32 },
}

impl GradientAxis {
    /// Whether `ramp_common::ramp_position` can evaluate this axis at all.
    ///
    /// **Finiteness only**, because that is the only thing the shader does not
    /// already handle: `ramp_position` floors both denominators at `1e-6`, so a
    /// zero-length line and a zero radius are degenerate-but-defined (everything
    /// lands at `t = 0`). A non-finite coordinate is the case it cannot floor — the
    /// guard is a `max`, which is unspecified on a `NaN`, and the `clamp` after it
    /// no better. That is a texel-wide disagreement between two clients rasterizing
    /// the same log.
    pub fn usable(&self) -> bool {
        match self {
            Self::Linear { from, to } => from.is_finite() && to.is_finite(),
            Self::Radial { center, radius } => center.is_finite() && radius.is_finite(),
        }
    }
}
