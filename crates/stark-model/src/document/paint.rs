//! **What paint to lay**: one color everywhere, or a ramp read from canvas position
//! (§22.4).
//!
//! Its own module because two actions lay paint and neither owns the answer — a
//! [`Fill`](super::ActionKind::Fill) through `fill.wesl`, a matte layer through
//! `matte.wesl` (§15.4). One type, so a matte and a fill cannot diverge over what
//! paint is.

use serde::{Deserialize, Serialize};

use crate::Srgb;
use crate::geom::Vec2;
use crate::gradient::Gradient;

/// What paint to lay: the same parcel everywhere, or one that varies with canvas
/// position (§22.4). A gradient is not a new pipeline but a fill whose parcel reads
/// its latent from position (§18.0.4), so the region, the gate, the stacking law and
/// the footprint stay [`FillOp`](super::FillOp)'s.
///
/// **A parcel says what paint, never how much.** Nothing here carries a per-unit
/// opacity: a fill states its strength in
/// [`FillOp::opacity`](super::FillOp::opacity), and a matte's is its *layer* opacity
/// (§15.3), its paint a full-strength coat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum Parcel {
    /// One color everywhere, straight sRGB and **color only** — how strongly a fill
    /// covers is [`FillOp::opacity`](super::FillOp::opacity) (§6.1).
    Solid(Srgb),
    /// A color ramp read from canvas position (§22.4).
    Gradient(GradientParcel),
}

impl Parcel {
    /// The color a one-swatch summary shows: the solid itself, or the ramp's start —
    /// the stop the axis anchors on, and the same reading
    /// [`sanitized`](Self::sanitized) gives an unusable axis.
    pub fn swatch(&self) -> Srgb {
        match self {
            Self::Solid(c) => *c,
            Self::Gradient(GradientParcel { gradient, .. }) => gradient.sample(0.0),
        }
    }

    /// The same paint with an axis the ramp pass can evaluate — the parcel's half of
    /// [`FillOp::with_paint`](super::FillOp::with_paint) and of
    /// [`ActionKind::sanitized`](super::ActionKind::sanitized), where a matte's paint
    /// comes through.
    ///
    /// **An unusable axis degrades the parcel to the ramp's anchor** —
    /// `Solid(gradient.sample(0.0))` — rather than being clamped into a different axis
    /// or refusing the fill. Deterministic, and it cannot make a `NaN`.
    pub fn sanitized(self) -> Self {
        match self {
            // An `Srgb` is finite and bounded by construction, and a ramp's stops
            // are `Srgb`s, so the axis is all that is left to hold.
            Self::Solid(_) => self,
            Self::Gradient(GradientParcel { gradient, axis }) if axis.usable() => {
                Self::Gradient(GradientParcel { gradient, axis })
            }
            Self::Gradient(GradientParcel { gradient, .. }) => Self::Solid(gradient.sample(0.0)),
        }
    }

    /// The same paint shifted whole by `by` (§14.12): a solid has no position
    /// and rides through; a gradient's axis is read from position and moves.
    pub fn translated(&self, by: crate::geom::Vec2) -> Self {
        match self {
            Self::Solid(_) => self.clone(),
            Self::Gradient(GradientParcel { gradient, axis }) => Self::Gradient(GradientParcel {
                gradient: gradient.clone(),
                axis: axis.translated(by),
            }),
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

/// The geometry mapping canvas position to ramp position (§22.4). Beyond either end
/// the ramp holds its end stop: a gradient fill covers its whole region, and the axis
/// only says where the transition lives.
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
    /// **Finiteness only.** The shader floors both denominators at `1e-6`, so a
    /// zero-length line and a zero radius are degenerate-but-defined; a non-finite
    /// coordinate is the case it cannot floor, since its guard is a `max` and that is
    /// unspecified on a `NaN` — a texel-wide disagreement between two clients
    /// rasterizing the same log.
    pub fn usable(&self) -> bool {
        match self {
            Self::Linear { from, to } => from.is_finite() && to.is_finite(),
            Self::Radial { center, radius } => center.is_finite() && radius.is_finite(),
        }
    }

    /// The same axis shifted whole by `by` (§14.12) — position is all an axis is,
    /// apart from the radial's radius, which is a length and stays.
    pub fn translated(&self, by: Vec2) -> Self {
        match self {
            Self::Linear { from, to } => Self::Linear {
                from: *from + by,
                to: *to + by,
            },
            Self::Radial { center, radius } => Self::Radial {
                center: *center + by,
                radius: *radius,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::GradientStop;

    const ANCHOR: [f32; 3] = [0.2, 0.4, 0.6];

    /// A two-stop ramp whose first stop is [`ANCHOR`] — what an unusable axis has to
    /// leave behind.
    fn ramp() -> Gradient {
        Gradient::new(vec![
            GradientStop {
                t: 0.0,
                color: Srgb::new(ANCHOR),
            },
            GradientStop {
                t: 1.0,
                color: Srgb::new([0.9, 0.1, 0.05]),
            },
        ])
        .expect("a two-stop ramp")
    }

    /// **An unusable axis degrades the parcel to the ramp's anchor** (§22.4) — to
    /// exactly `Solid(gradient.sample(0.0))`, the first stop bit for bit, and not
    /// merely to *some* solid. The specific color is the claim, since it is what a
    /// matte layer standing in this parcel shows.
    #[test]
    fn a_gradient_nobody_can_place_degrades_to_the_ramps_anchor() {
        let anchor = Parcel::Solid(Srgb::new(ANCHOR));
        for axis in [
            GradientAxis::Linear {
                from: Vec2::new(f32::NAN, 0.0),
                to: Vec2::splat(64.0),
            },
            GradientAxis::Linear {
                from: Vec2::ZERO,
                to: Vec2::new(0.0, f32::INFINITY),
            },
            GradientAxis::Radial {
                center: Vec2::new(0.0, f32::NEG_INFINITY),
                radius: 32.0,
            },
            GradientAxis::Radial {
                center: Vec2::ZERO,
                radius: f32::NAN,
            },
        ] {
            let parcel = Parcel::Gradient(GradientParcel {
                gradient: ramp(),
                axis,
            });
            assert!(!axis.usable(), "{axis:?} should be unplaceable");
            assert_eq!(
                parcel.clone().sanitized(),
                anchor,
                "{axis:?} should leave the ramp's anchor",
            );
            // The same reading `swatch` gives it, which is what makes the degraded
            // parcel indistinguishable from the summary the picker was showing.
            assert_eq!(parcel.swatch(), Srgb::new(ANCHOR));
            // Idempotent by construction, since what is left is a solid.
            assert_eq!(anchor.clone().sanitized(), anchor);
        }
    }

    /// A parcel the ramp pass can evaluate comes through **untouched** — including the
    /// degenerate-but-defined cases the shader floors for itself (a zero-length line,
    /// a zero radius).
    #[test]
    fn a_placeable_parcel_is_left_alone() {
        for axis in [
            GradientAxis::Linear {
                from: Vec2::new(-10.0, 4.0),
                to: Vec2::new(200.0, 4.0),
            },
            GradientAxis::Linear {
                from: Vec2::ZERO,
                to: Vec2::ZERO,
            },
            GradientAxis::Radial {
                center: Vec2::splat(12.0),
                radius: 0.0,
            },
        ] {
            let parcel = Parcel::Gradient(GradientParcel {
                gradient: ramp(),
                axis,
            });
            assert_eq!(parcel.clone().sanitized(), parcel, "{axis:?} was repaired");
        }
        // A solid has no axis to be unusable.
        let solid = Parcel::Solid(Srgb::new([0.25, 0.5, 0.75]));
        assert_eq!(solid.clone().sanitized(), solid);
        assert_eq!(solid.swatch(), Srgb::new([0.25, 0.5, 0.75]));
    }
}
