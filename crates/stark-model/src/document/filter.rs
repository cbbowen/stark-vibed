//! Filter layers (§21): a layer whose content is a **function of what is
//! composited beneath it** rather than paint of its own. Making it a layer is what
//! buys visibility, opacity, ordering, naming, removal, undo, save, replay and
//! collaboration for free — as with the matte (§15.3).
//!
//! Two consequences of sitting in the stack. **A filter reaches exactly as far as its
//! own stack**: it reads the accumulator that stack has built, so at the root it
//! filters the whole painting and inside a group it filters that group, which makes
//! "filter just this layer" that layer carrying the filter — the gesture §14.4 spends
//! on clipping. The `clip` a filter layer *does* take (§21.4.1) bounds what the pass
//! may write, never what it reads. **Layer opacity is filter strength**: the result is
//! mixed against the untouched backdrop by it, so a zero-opacity filter is the
//! identity (§21.4).
//!
//! Nothing here touches a tile — a filter's whole effect is one fullscreen pass at
//! composite time.

use serde::{Deserialize, Serialize};

use crate::sanitize::finite_in;

/// What a filter layer does to the stack beneath it (§21.2), and the seam the rest of
/// §21.7 lands on.
///
/// A variant is matched by *name*, not by position, so a new kind may be inserted
/// wherever it reads best without disturbing the filters in saved files (§8,
/// `ActionKind`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum Filter {
    /// Exposure, contrast, saturation and hue, applied in Oklab (§21.5).
    Color(ColorAdjust),
    /// The spectrum pulled apart across the picture — the lens's dispersion as the
    /// integral it is, not as three shifted copies (§21.10).
    Chromatic(ChromaticAberration),
    /// The stack beneath repainted by its lightness: Oklab `L` indexes the ramp, and
    /// the ramp's color is what the paint becomes (§21.11). `None` — no ramp chosen
    /// yet — is the neutral, since a gradient map with *any* ramp is already an edit.
    GradientMap(Option<crate::gradient::Gradient>),
    /// Everything beneath spread through a lens's circle of confusion — a true
    /// convolution of the light, not a Gaussian approximation of one (§21.12).
    FocalBlur(FocalBlur),
}

impl Filter {
    /// Every filter this build offers, at its neutral setting — the list the "new
    /// filter" picker is built from, in the order it should offer them.
    pub const ALL: [Filter; 4] = [
        Filter::Color(ColorAdjust::NEUTRAL),
        Filter::Chromatic(ChromaticAberration::NEUTRAL),
        Filter::GradientMap(None),
        Filter::FocalBlur(FocalBlur::NEUTRAL),
    ];

    /// What this filter is called, in the panel and in the layer row.
    pub fn label(&self) -> &'static str {
        match self {
            Filter::Color(_) => "Color",
            Filter::Chromatic(_) => "Chromatic aberration",
            Filter::GradientMap(_) => "Gradient map",
            Filter::FocalBlur(_) => "Focal blur",
        }
    }

    /// Whether this filter reads **neighbouring** texels rather than only the one it
    /// writes — the point/resampling split §21.3.1 draws.
    ///
    /// The merge (§14.11.7) asks it, and the answer is a law rather than a preference:
    /// every pass that writes tiles must be a pure function of canvas position (§6.4),
    /// which a gather over tiles is not at any apron width. So a filter that resamples
    /// cannot be baked into the paint beneath it.
    pub fn resamples(&self) -> bool {
        match self {
            Filter::Color(_) | Filter::GradientMap(_) => false,
            Filter::Chromatic(_) | Filter::FocalBlur(_) => true,
        }
    }

    /// Whether this filter changes anything at all. A neutral filter still costs its
    /// pass — the compositor cannot know that `1.0` is the identity of a gain — so
    /// this is what lets the draw list leave it out (§21.3).
    pub fn is_neutral(&self) -> bool {
        match self {
            Filter::Color(c) => *c == ColorAdjust::NEUTRAL,
            // The spread alone: at zero no wavelength moves, whatever the angle
            // points at.
            Filter::Chromatic(c) => c.spread == 0.0,
            // There is no identity ramp to compare against — even black-to-white
            // repaints every color — so absence is this kind's only neutral.
            Filter::GradientMap(g) => g.is_none(),
            // The radius alone: at zero no light moves, whatever aperture it would
            // have moved through.
            Filter::FocalBlur(b) => b.radius == 0.0,
        }
    }

    /// The same **kind** of filter at its neutral setting — what the bar's
    /// "Neutral" chip puts back, said once here rather than per panel arm.
    #[must_use]
    pub fn neutral(&self) -> Self {
        match self {
            Filter::Color(_) => Filter::Color(ColorAdjust::NEUTRAL),
            Filter::Chromatic(_) => Filter::Chromatic(ChromaticAberration::NEUTRAL),
            Filter::GradientMap(_) => Filter::GradientMap(None),
            Filter::FocalBlur(_) => Filter::FocalBlur(FocalBlur::NEUTRAL),
        }
    }

    /// The same filter with every parameter finite and in range — the funnel every
    /// filter passes through on its way into the document.
    ///
    /// Applied where the action is minted (`Engine::process`) and again where a filter
    /// enters state (`DocState::set_filter` / `insert_filter`), since a loaded file or
    /// a remote peer reaches state without passing through `process`. Idempotent. A
    /// `NaN` saturation reaching a fullscreen pass poisons every texel of the frame,
    /// and nothing downstream can notice.
    #[must_use]
    pub fn sanitized(self) -> Self {
        match self {
            Filter::Color(c) => Filter::Color(c.sanitized()),
            Filter::Chromatic(c) => Filter::Chromatic(c.sanitized()),
            // A `Gradient` holds its structural invariants by construction (§22.1) and
            // its stops' range by `Srgb`, so this arm has nothing to repair.
            Filter::GradientMap(g) => Filter::GradientMap(g),
            Filter::FocalBlur(b) => Filter::FocalBlur(b.sanitized()),
        }
    }
}

/// A color adjustment, applied in **Oklab** (§21.5): in a perceptual space lightness,
/// chroma and hue are separable, so moving one knob leaves the other two where they
/// were.
///
/// **Three of them are one gesture.** [`hue`](Self::hue),
/// [`saturation`](Self::saturation) and [`tint`](Self::tint) are one affine map of the
/// `(a, b)` plane — `ab' = tint + saturation · R(hue) · ab` — applied in the order they
/// are listed: rotate, scale, translate. The panel shows them as one directed circle
/// (§21.6).
///
/// [`NEUTRAL`](Self::NEUTRAL) is the identity, and the value a filter layer is created
/// holding.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct ColorAdjust {
    /// Exposure, in **stops**: the light beneath is scaled by `2^exposure`, so `+1` is
    /// twice the light and `-1` is half.
    ///
    /// Applied to *light* (the normalized XYZ the blend modes combine in, §18.0.4)
    /// rather than to Oklab's `L`, because doubling light is what an exposure is.
    pub exposure: f32,
    /// Contrast: a gain on Oklab `L` about mid-grey. `1` is the identity, `0` flattens
    /// the picture to mid-grey, `2` doubles the spread.
    ///
    /// The pivot is fixed (`stark-engine`'s `filters::CONTRAST_PIVOT`) rather than the
    /// picture's own mean, so what the slider does cannot depend on what is underneath.
    pub contrast: f32,
    /// Saturation: a gain on Oklab chroma — the distance of `(a, b)` from the
    /// achromatic axis. `1` is the identity, `0` is a greyscale that keeps every
    /// lightness exactly where it was, and past `1` is a boost.
    pub saturation: f32,
    /// Hue rotation of the Oklab `(a, b)` plane, in **radians**, clockwise from red
    /// toward yellow. Radians is the unit the engine states angles in; degrees are a
    /// frontend presentation.
    pub hue: f32,
    /// A color cast: `(a, b)` added to Oklab **after** the rotation and the chroma
    /// gain. `[0, 0]` is the identity.
    ///
    /// It is therefore exactly **the color a grey becomes** — the one thing the other
    /// three cannot do, a gain and a rotation both fixing the achromatic axis — which
    /// is what makes `saturation: 0` plus a tint a duotone rather than a greyscale.
    /// Stated as the plane's own two axes rather than as a chroma and an angle, so
    /// there is no second convention to disagree with [`hue`](Self::hue).
    pub tint: [f32; 2],
}

impl ColorAdjust {
    /// The identity: no exposure, unity gains, no rotation, no cast.
    pub const NEUTRAL: Self = Self {
        exposure: 0.0,
        contrast: 1.0,
        saturation: 1.0,
        hue: 0.0,
        tint: [0.0, 0.0],
    };

    /// The widest each knob may be dialled — the range a frontend's slider spans and
    /// the range [`sanitized`](Self::sanitized) holds a log entry to. Bounded at all
    /// because a fullscreen pass has no coverage to hide behind.
    pub const EXPOSURE: (f32, f32) = (-4.0, 4.0);
    pub const CONTRAST: (f32, f32) = (0.0, 2.0);
    pub const SATURATION: (f32, f32) = (0.0, 2.0);
    /// A full turn either way, so every rotation is reachable and none is reachable
    /// twice by more than a lap.
    pub const HUE: (f32, f32) = (-std::f32::consts::PI, std::f32::consts::PI);
    /// Per component, on each axis of the `(a, b)` plane. `0.16` is about as far from
    /// the achromatic axis as the sRGB gamut itself reaches at mid-grey; past it every
    /// texel is out of gamut on the same side and the pass returns a flat wash.
    pub const TINT: (f32, f32) = (-0.16, 0.16);

    /// Every knob finite and in range — see [`Filter::sanitized`]. A non-finite value
    /// falls back to that knob's **neutral** rather than to a bound: `NaN` says nothing
    /// about which end of the range was meant.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            exposure: finite_in(self.exposure, 0.0, Self::EXPOSURE),
            contrast: finite_in(self.contrast, 1.0, Self::CONTRAST),
            saturation: finite_in(self.saturation, 1.0, Self::SATURATION),
            hue: finite_in(self.hue, 0.0, Self::HUE),
            // Per component, so one unusable axis of a cast does not throw away the
            // other.
            tint: [
                finite_in(self.tint[0], 0.0, Self::TINT),
                finite_in(self.tint[1], 0.0, Self::TINT),
            ],
        }
    }
}

impl Default for ColorAdjust {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

/// Chromatic aberration: the lens's dispersion, as an **integral over the shifted
/// spectrum** rather than three shifted copies (§21.10).
///
/// Two numbers, because they describe the lens: how far the spectrum is pulled apart,
/// and along which axis. Both are stated **in canvas terms** (canvas px, canvas angle),
/// so the fringes scale with a zoom and turn with a rotation the way the paint does.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct ChromaticAberration {
    /// How far the red end of the spectrum lands from the blue end, in **canvas px** —
    /// the full width of the fringe an edge grows. `0` is the identity: no wavelength
    /// moves, whatever the angle says.
    pub spread: f32,
    /// The axis the spectrum spreads along, in **radians**, canvas space: the direction
    /// the blue end is carried, with the red end opposite. The two ends part
    /// symmetrically, so the picture itself stays put.
    pub angle: f32,
}

impl ChromaticAberration {
    /// The identity: nothing spreads.
    pub const NEUTRAL: Self = Self {
        spread: 0.0,
        angle: 0.0,
    };

    /// The widest each knob may be dialled — the range a frontend's slider spans and
    /// the range [`sanitized`](Self::sanitized) holds a log entry to. The spread's
    /// ceiling is also a promise to the renderer: it is what keeps the pass's capped
    /// tap budget out of reach at working zooms (§21.10).
    pub const SPREAD: (f32, f32) = (0.0, 128.0);
    /// A full turn either way, so every axis is reachable and none is reachable
    /// twice by more than a lap.
    pub const ANGLE: (f32, f32) = (-std::f32::consts::PI, std::f32::consts::PI);

    /// Every knob finite and in range — see [`Filter::sanitized`]. A non-finite value
    /// falls back to that knob's **neutral**, for [`ColorAdjust::sanitized`]'s reason.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            spread: finite_in(self.spread, 0.0, Self::SPREAD),
            angle: finite_in(self.angle, 0.0, Self::ANGLE),
        }
    }
}

impl Default for ChromaticAberration {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

/// The aperture's **shape**: what one point of light becomes (§21.12).
///
/// **A variant is a mechanism, not a look.** Each is the continuous family one piece of
/// a lens sweeps out — the iris closing onto its blades, the secondary mirror growing
/// across the middle, the anamorphic element squeezing — so the named looks (hexagonal
/// bokeh, the mirror lens's doughnut, the cat's eye) are *settings* here rather than
/// entries, and every variant carries a knob.
///
/// **Every variant is contained in the disc of the blur's radius**, and that is a
/// contract: the FFT pads its planes by the radius on each side so the circular
/// convolution's wrap-around lands in zeros (§21.12), and a shape reaching past it would
/// carry one edge of the picture onto the other. The radius is therefore each shape's
/// *circumradius* — a polygon's vertices and an oval's long axis both land on it.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum Aperture {
    /// A circular aperture, with as much of its middle as one likes taken out of it:
    /// the lens wide open at `obstruction` 0, and a catadioptric — mirror — lens's
    /// doughnut as it grows.
    Disc {
        /// The central shadow's share of the radius, held to
        /// [`OBSTRUCTION`](Aperture::OBSTRUCTION). `0` is the plain disc and the
        /// neutral — what a file written before the aperture existed meant by saying
        /// nothing.
        ///
        /// The ceiling is the renderer's: past it the rim is thin enough that
        /// decimating the convolution (§21.12) would sample it rather than resolve it.
        obstruction: f32,
    },
    /// The iris stopped down onto its blades — a regular polygon of `count` sides,
    /// turned by `angle`. One n-gon behind pentagon, hexagon and octagon, as §20 puts
    /// one camera behind 1-, 2- and 3-point perspective.
    Blades {
        /// How many blades, held to [`BLADES`](Aperture::BLADES). Below three there is
        /// no polygon, and much past a dozen there is no telling one from the
        /// [`Disc`](Self::Disc) that is already a variant.
        count: u32,
        /// The iris's turn, in radians and **canvas space** — the frame
        /// [`ChromaticAberration::angle`] is stated in, and for the same reason (§6.4).
        ///
        /// A polygon's own symmetry makes this periodic in `2π/count`, but the range
        /// spans a full turn like every other angle here: it is the hand's range, not
        /// the shape's.
        angle: f32,
    },
    /// The oval an anamorphic lens makes: a cylindrical front element squeezes the
    /// circle of confusion along one axis, and every out-of-focus highlight stretches
    /// with it.
    Oval {
        /// The long axis's length over the short one's, held to
        /// [`SQUEEZE`](Aperture::SQUEEZE) — `2` is what anamorphic cinema means by the
        /// word. The long axis is the radius, so squeezing narrows the oval rather than
        /// growing it.
        squeeze: f32,
        /// The long axis's direction, in radians and **canvas space** — see
        /// [`Blades`](Self::Blades)'s own turn.
        angle: f32,
    },
}

impl Aperture {
    /// How many blades an iris may be given. Three is the floor a polygon has;
    /// twelve is where one stops being tellable from a [`Disc`](Self::Disc) at any
    /// radius the blur can be dialled to.
    pub const BLADES: (u32, u32) = (3, 12);

    /// How far a shape that has a direction may be turned: a whole turn.
    pub const ANGLE: (f32, f32) = (-std::f32::consts::PI, std::f32::consts::PI);

    /// The central obstruction's range — see [`Aperture::Disc`].
    pub const OBSTRUCTION: (f32, f32) = (0.0, 0.9);

    /// The anamorphic squeeze's range — see [`Aperture::Oval`]. The floor is `1`, the
    /// round oval, because a squeeze under 1 is the *same* set of ovals turned a
    /// quarter turn.
    pub const SQUEEZE: (f32, f32) = (1.0, 4.0);

    /// Every shape this build offers, at the setting it is *given* when picked — the
    /// list the bar's run of buttons is built from, in order: the disc first, being the
    /// neutral, then the two ways of departing from it. The numbers are defaults, not
    /// fudge constants — the plain lens, the commonest six-bladed iris, the 2×
    /// anamorphic squeeze.
    pub const ALL: [Aperture; 3] = [
        Aperture::Disc { obstruction: 0.0 },
        Aperture::Blades {
            count: 6,
            angle: 0.0,
        },
        Aperture::Oval {
            squeeze: 2.0,
            angle: 0.0,
        },
    ];

    /// What this shape is called, in the bar and in the layer row: the name of the
    /// **family**, not of the setting — an obstructed disc is still "Disc".
    pub fn label(&self) -> &'static str {
        match self {
            Aperture::Disc { .. } => "Disc",
            Aperture::Blades { .. } => "Blades",
            Aperture::Oval { .. } => "Oval",
        }
    }

    /// Whether these are the same **shape**, whatever each is set to — what a picker
    /// lights a chip by, since re-picking the shape already showing must not throw
    /// its knobs away.
    pub fn same_shape(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    /// Every parameter finite and in range — see [`Filter::sanitized`]; a non-finite
    /// value lands on the knob's own identity.
    ///
    /// **The shape itself survives** whatever its numbers do: a `NaN` squeeze is a
    /// broken oval, not a request for a disc.
    #[must_use]
    pub fn sanitized(self) -> Self {
        match self {
            Aperture::Disc { obstruction } => Aperture::Disc {
                obstruction: finite_in(obstruction, 0.0, Self::OBSTRUCTION),
            },
            // A count has no `NaN` to fall back from, so the clamp is the whole repair.
            Aperture::Blades { count, angle } => Aperture::Blades {
                count: count.clamp(Self::BLADES.0, Self::BLADES.1),
                angle: finite_in(angle, 0.0, Self::ANGLE),
            },
            Aperture::Oval { squeeze, angle } => Aperture::Oval {
                // `1` is the un-squeezed oval — this knob's own identity.
                squeeze: finite_in(squeeze, 1.0, Self::SQUEEZE),
                angle: finite_in(angle, 0.0, Self::ANGLE),
            },
        }
    }
}

impl Default for Aperture {
    /// The unobstructed disc — what a file written before the aperture existed means
    /// by saying nothing (see [`FocalBlur::aperture`]).
    fn default() -> Self {
        Aperture::Disc { obstruction: 0.0 }
    }
}

/// Focal blur: everything beneath spread through a lens's **circle of confusion** — a
/// true convolution of the light with the aperture's shape, computed by FFT, not a
/// Gaussian standing in for one (§21.12).
///
/// A size and a shape. The size is stated **in canvas px** for the chromatic spread's
/// reason (§21.10): the bokeh belongs to the artwork, so it scales with a zoom and holds
/// its size in an export.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct FocalBlur {
    /// The radius of the circle of confusion, in **canvas px**. `0` is the identity: no
    /// light moves.
    ///
    /// It is also every [`Aperture`]'s circumradius, which is what lets the shape change
    /// without the FFT's padding being recomputed — see there.
    pub radius: f32,
    /// The shape the light is spread through. [`Aperture::Disc`] by default, which
    /// is what a file written before this field existed meant by saying nothing.
    #[serde(default)]
    pub aperture: Aperture,
}

impl FocalBlur {
    /// The identity: nothing spreads.
    pub const NEUTRAL: Self = Self {
        radius: 0.0,
        aperture: Aperture::Disc { obstruction: 0.0 },
    };

    /// The widest the radius may be dialled — the range a frontend's slider spans and
    /// the range [`sanitized`](Self::sanitized) holds a log entry to. The ceiling is
    /// also a promise to the renderer: the FFT pads its scratch by the on-screen radius
    /// (§21.12), and this bound is what keeps that padding affordable at working zooms.
    pub const RADIUS: (f32, f32) = (0.0, 128.0);

    /// Every knob finite and in range — see [`Filter::sanitized`]. A non-finite value
    /// falls back to that knob's **neutral**, for [`ColorAdjust::sanitized`]'s reason.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            radius: finite_in(self.radius, 0.0, Self::RADIUS),
            aperture: self.aperture.sanitized(),
        }
    }
}

impl Default for FocalBlur {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A filter arriving from a file or a peer reaches **every texel** of the frame, so
    /// a value that is not a number must not survive the way in.
    #[test]
    fn sanitizing_replaces_the_unusable_with_the_identity() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let wild = Filter::Color(ColorAdjust {
                exposure: bad,
                contrast: bad,
                saturation: bad,
                hue: bad,
                tint: [bad, bad],
            });
            assert_eq!(wild.sanitized(), Filter::Color(ColorAdjust::NEUTRAL));
            let wild = Filter::Chromatic(ChromaticAberration {
                spread: bad,
                angle: bad,
            });
            assert_eq!(
                wild.sanitized(),
                Filter::Chromatic(ChromaticAberration::NEUTRAL),
            );
            // The shape survives what its numbers do not: the fallback is per knob —
            // the radius to 0, the squeeze to the 1 that is its own identity — and the
            // variant stays.
            let wild = Filter::FocalBlur(FocalBlur {
                radius: bad,
                aperture: Aperture::Oval {
                    squeeze: bad,
                    angle: bad,
                },
            });
            assert_eq!(
                wild.sanitized(),
                Filter::FocalBlur(FocalBlur {
                    radius: 0.0,
                    aperture: Aperture::Oval {
                        squeeze: 1.0,
                        angle: 0.0,
                    },
                }),
            );
            let wild = Filter::FocalBlur(FocalBlur {
                radius: bad,
                aperture: Aperture::Disc { obstruction: 0.0 },
            });
            assert_eq!(wild.sanitized(), Filter::FocalBlur(FocalBlur::NEUTRAL));
        }
        // …and an ordinary out-of-range value is *clamped* rather than neutralized:
        // a slider pushed past its stop still means "as far as it goes".
        let hot = Filter::Color(ColorAdjust {
            exposure: 40.0,
            saturation: -3.0,
            tint: [5.0, -5.0],
            ..ColorAdjust::NEUTRAL
        });
        assert_eq!(
            hot.sanitized(),
            Filter::Color(ColorAdjust {
                exposure: ColorAdjust::EXPOSURE.1,
                saturation: ColorAdjust::SATURATION.0,
                tint: [ColorAdjust::TINT.1, ColorAdjust::TINT.0],
                ..ColorAdjust::NEUTRAL
            }),
        );
        // …and one unusable axis of a cast does not take the other with it: the pair
        // is two numbers the shader adds, not one quantity.
        let half = Filter::Color(ColorAdjust {
            tint: [f32::NAN, 0.08],
            ..ColorAdjust::NEUTRAL
        });
        assert_eq!(
            half.sanitized(),
            Filter::Color(ColorAdjust {
                tint: [0.0, 0.08],
                ..ColorAdjust::NEUTRAL
            }),
        );
        let wide = Filter::Chromatic(ChromaticAberration {
            spread: 1000.0,
            angle: 9.0,
        });
        assert_eq!(
            wide.sanitized(),
            Filter::Chromatic(ChromaticAberration {
                spread: ChromaticAberration::SPREAD.1,
                angle: ChromaticAberration::ANGLE.1,
            }),
        );
        let deep = Filter::FocalBlur(FocalBlur {
            radius: 1e6,
            aperture: Aperture::Disc { obstruction: 0.0 },
        });
        assert_eq!(
            deep.sanitized(),
            Filter::FocalBlur(FocalBlur {
                radius: FocalBlur::RADIUS.1,
                aperture: Aperture::Disc { obstruction: 0.0 },
            }),
        );
        // An aperture out of range is clamped the same way, per shape and per
        // parameter — a blade count included, which has no `NaN` to neutralize and
        // so is only ever clamped.
        let absurd = Filter::FocalBlur(FocalBlur {
            radius: 4.0,
            aperture: Aperture::Blades {
                count: 0,
                angle: 9.0,
            },
        });
        assert_eq!(
            absurd.sanitized(),
            Filter::FocalBlur(FocalBlur {
                radius: 4.0,
                aperture: Aperture::Blades {
                    count: Aperture::BLADES.0,
                    angle: Aperture::ANGLE.1,
                },
            }),
        );
        let pinched = Filter::FocalBlur(FocalBlur {
            radius: 4.0,
            aperture: Aperture::Disc { obstruction: 5.0 },
        });
        assert_eq!(
            pinched.sanitized(),
            Filter::FocalBlur(FocalBlur {
                radius: 4.0,
                aperture: Aperture::Disc {
                    obstruction: Aperture::OBSTRUCTION.1,
                },
            }),
        );
        // Both ends of the count, since a clamp is two bounds and only one of them
        // was above.
        let many = Filter::FocalBlur(FocalBlur {
            radius: 4.0,
            aperture: Aperture::Blades {
                count: 1000,
                angle: 0.0,
            },
        });
        assert_eq!(
            many.sanitized(),
            Filter::FocalBlur(FocalBlur {
                radius: 4.0,
                aperture: Aperture::Blades {
                    count: Aperture::BLADES.1,
                    angle: 0.0,
                },
            }),
        );
        // **The squeeze floor is the containment contract**: the long axis is the
        // radius the renderer padded its planes by, so a squeeze below 1 would put the
        // *other* axis outside that padding and wrap one edge of the picture onto the
        // other (§21.12). The shader floors it too; this pins that a log never asks it
        // to.
        let stretched = Filter::FocalBlur(FocalBlur {
            radius: 4.0,
            aperture: Aperture::Oval {
                squeeze: 0.25,
                angle: 0.0,
            },
        });
        assert_eq!(
            stretched.sanitized(),
            Filter::FocalBlur(FocalBlur {
                radius: 4.0,
                aperture: Aperture::Oval {
                    squeeze: Aperture::SQUEEZE.0,
                    angle: 0.0,
                },
            }),
        );
    }

    /// Zero spread is the identity **whatever the angle says**, so the draw list stays
    /// free to drop the pass (§21.3's "neutral is dropped" rule).
    #[test]
    fn an_unspread_chromatic_filter_is_neutral_at_any_angle() {
        let aimed = Filter::Chromatic(ChromaticAberration {
            spread: 0.0,
            angle: 2.0,
        });
        assert!(aimed.is_neutral());
        assert_eq!(
            aimed.clone().sanitized(),
            aimed,
            "sanitizing must not disturb it"
        );
    }

    /// A radius-0 blur is the identity **whatever aperture it is set to** — the
    /// chromatic filter's rule above, from the other direction.
    #[test]
    fn an_unradiused_blur_is_neutral_at_any_aperture() {
        for aperture in Aperture::ALL {
            let shaped = Filter::FocalBlur(FocalBlur {
                radius: 0.0,
                aperture,
            });
            assert!(
                shaped.is_neutral(),
                "{aperture:?} at radius 0 is not an edit"
            );
            assert_eq!(
                shaped.clone().sanitized(),
                shaped,
                "every shape `ALL` offers is already in range",
            );
        }
    }

    /// A gradient map's stops are inside the cube **before the sanitizer sees them** —
    /// a stop's color is an [`Srgb`](crate::Srgb), and there is no way to build one
    /// outside it. What is left to check is that the ramp survives hot values without
    /// degenerating, and that the sanitizer is the identity on it.
    #[test]
    fn a_gradient_maps_stops_are_bounded_before_it_is_sanitized() {
        use crate::Srgb;
        use crate::gradient::{Gradient, GradientStop};
        let hot = Gradient::new(vec![
            GradientStop {
                t: 0.0,
                color: Srgb::new([-2.0, 0.5, 1e30]),
            },
            GradientStop {
                t: 1.0,
                color: Srgb::new([0.25, 2.0, 0.75]),
            },
        ])
        .expect("clamping must not degenerate the ramp");
        assert_eq!(hot.stops()[0].color.get(), [-2.0, 0.5, Srgb::EXTENT]);
        assert_eq!(hot.stops()[1].color.get(), [0.25, 2.0, 0.75]);

        let filter = Filter::GradientMap(Some(hot));
        assert_eq!(
            filter.clone().sanitized(),
            filter,
            "there is nothing left for the sanitizer to do to a ramp",
        );
    }

    /// A freshly added filter must change nothing: adding one is a step you take
    /// *before* deciding what it does, and a filter that darkened the painting on
    /// creation would be a destructive act dressed as an organizational one.
    #[test]
    fn a_new_filter_is_the_identity() {
        for filter in Filter::ALL {
            assert!(filter.is_neutral(), "{} is not neutral", filter.label());
            assert_eq!(
                filter.clone().sanitized(),
                filter,
                "{} is not stable",
                filter.label()
            );
        }
    }

    /// A [`Filter`] is read by variant **name**, not position (§8), so the picker may
    /// gain a kind anywhere in [`Filter::ALL`] and a saved filter layer still means the
    /// adjustment it meant. `Old` is the same four cases in a different order, and every
    /// one must read back with its payload intact.
    #[test]
    fn a_filter_is_read_by_variant_name_not_position() {
        use crate::Srgb;
        use crate::gradient::{Gradient, GradientStop};

        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "Filter")]
        enum Old {
            FocalBlur(FocalBlur),
            GradientMap(Option<Gradient>),
            Chromatic(ChromaticAberration),
            Color(ColorAdjust),
        }

        let read = |old: &Old| {
            carbonite::from_slice_static::<Filter>(&carbonite::to_vec_static(old).expect("encodes"))
                .expect("a declaration order this build does not use still reads")
        };

        let color = ColorAdjust {
            exposure: 0.75,
            ..ColorAdjust::NEUTRAL
        };
        assert_eq!(read(&Old::Color(color)), Filter::Color(color));
        let cast = ChromaticAberration {
            spread: 3.0,
            angle: 0.5,
        };
        assert_eq!(read(&Old::Chromatic(cast)), Filter::Chromatic(cast));
        assert_eq!(read(&Old::GradientMap(None)), Filter::GradientMap(None));
        let ramp = Gradient::new(vec![
            GradientStop {
                t: 0.0,
                color: Srgb::BLACK,
            },
            GradientStop {
                t: 1.0,
                color: Srgb::WHITE,
            },
        ])
        .expect("a two-stop ramp");
        assert_eq!(
            read(&Old::GradientMap(Some(ramp.clone()))),
            Filter::GradientMap(Some(ramp)),
            "the stops travel with the name, not with an index",
        );
        let blur = FocalBlur {
            radius: 12.0,
            aperture: Aperture::Blades {
                count: 6,
                angle: 0.25,
            },
        };
        assert_eq!(read(&Old::FocalBlur(blur)), Filter::FocalBlur(blur));
    }

    /// The same for [`Aperture`], the sharpest case in the log: three payload-bearing
    /// variants reached through a `#[serde(default)]` field, so a file written before
    /// the aperture existed and one written after take **different paths** through the
    /// same reconciliation. Both are checked here.
    #[test]
    fn an_aperture_is_read_by_variant_name_and_an_older_blur_reads_the_bare_disc() {
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "Aperture")]
        enum Old {
            Oval { squeeze: f32, angle: f32 },
            Blades { count: u32, angle: f32 },
            Disc { obstruction: f32 },
        }

        let read = |old: &Old| {
            carbonite::from_slice_static::<Aperture>(
                &carbonite::to_vec_static(old).expect("encodes"),
            )
            .expect("a declaration order this build does not use still reads")
        };

        assert_eq!(
            read(&Old::Disc { obstruction: 0.4 }),
            Aperture::Disc { obstruction: 0.4 },
        );
        assert_eq!(
            read(&Old::Blades {
                count: 8,
                angle: -0.5
            }),
            Aperture::Blades {
                count: 8,
                angle: -0.5
            },
            "both fields travel with the name, not with an index",
        );
        assert_eq!(
            read(&Old::Oval {
                squeeze: 2.0,
                angle: 1.25
            }),
            Aperture::Oval {
                squeeze: 2.0,
                angle: 1.25
            },
        );

        // The other path: a blur written before the field existed says nothing about
        // a shape, and its absence has to mean the unobstructed disc — the neutral
        // [`FocalBlur::aperture`] promises — rather than refusing the file.
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "FocalBlur")]
        struct OldBlur {
            radius: f32,
        }
        let older = carbonite::from_slice_static::<FocalBlur>(
            &carbonite::to_vec_static(&OldBlur { radius: 12.0 }).expect("encodes"),
        )
        .expect("a blur from before the aperture still loads");
        assert_eq!(
            older,
            FocalBlur {
                radius: 12.0,
                aperture: Aperture::Disc { obstruction: 0.0 },
            },
            "an absent aperture means the lens wide open",
        );
    }
}
