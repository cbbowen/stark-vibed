//! Filter layers (§21): a layer whose content is a **function of what is
//! composited beneath it** rather than paint of its own.
//!
//! A paint layer adds light to the stack and a matte covers it; a filter *rewrites*
//! it. That is one more thing a layer can be, and — as with the matte (§15.3) —
//! making it a layer is what buys visibility, opacity, ordering, naming, removal,
//! undo, save, replay and collaboration without a single new concept.
//!
//! Two properties fall out of the stack rather than being designed, and both are
//! worth knowing before reading [`Filter`]:
//!
//! - **A filter reaches exactly as far as its own stack.** It reads the accumulator
//!   its stack has built so far, so at the root it filters the whole painting and
//!   inside a group it filters that group — which means "filter just this layer" is
//!   that layer carrying the filter, the same single gesture §14.4 spends on
//!   clipping. There is no clip-to-layer-below mode to invent — and the `clip` a
//!   filter layer *does* take (§21.4.1) is not one: it bounds what the pass may
//!   write, never what it reads, so scope is still position and nothing else.
//! - **Layer opacity is filter strength.** The pass mixes its result against the
//!   untouched backdrop by the layer's opacity, so a half-opacity filter is half the
//!   adjustment and a zero-opacity one is the identity — which is what fading a
//!   layer already means everywhere else (§21.4).
//!
//! What a filter is *not* is a brush. Nothing here touches a tile: the whole of a
//! filter's effect is one fullscreen pass at composite time, so it costs no GPU
//! memory, is free to re-tune, and is undone by dropping one action.

use serde::{Deserialize, Serialize};

/// What a filter layer does to the stack beneath it (§21.2).
///
/// Three variants, because three are built. The enum is the seam the rest of §21.7
/// lands on — motion blur, outline, glow — and per this codebase's own precedent
/// (§1) no variant appears here before it does something to a pixel. **Appended
/// only**: postcard encodes an enum by index (§8), so a variant inserted above an
/// existing one would silently rename every filter in every saved file.
///
/// `Clone`, not `Copy`, since the gradient map arrived: a ramp is a stop list, and
/// the enum's old `Copy` was always documented as lasting until the first variant
/// that carries more than a handful of numbers. A filter is still read once per
/// render and once per projection, so the clones stay countable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    /// Exposure, contrast, saturation and hue, applied in Oklab (§21.5).
    Color(ColorAdjust),
    /// The spectrum pulled apart across the picture — the lens's dispersion as the
    /// integral it is, not as three shifted copies (§21.10).
    Chromatic(ChromaticAberration),
    /// The stack beneath repainted by its lightness: Oklab `L` indexes the ramp,
    /// and the ramp's color is what the paint becomes (§21.11). `None` — no ramp
    /// chosen yet — is the neutral, the same shape as a chromatic filter whose
    /// spread has not left zero: a gradient map with *any* ramp is already an
    /// edit, so the only setting a freshly added one may hold is none at all.
    GradientMap(Option<crate::gradient::Gradient>),
}

impl Filter {
    /// Every filter this build offers, at its neutral setting — the list the "new
    /// filter" picker is built from, in the order it should offer them.
    pub const ALL: [Filter; 3] = [
        Filter::Color(ColorAdjust::NEUTRAL),
        Filter::Chromatic(ChromaticAberration::NEUTRAL),
        Filter::GradientMap(None),
    ];

    /// What this filter is called, in the panel and in the layer row.
    pub fn label(&self) -> &'static str {
        match self {
            Filter::Color(_) => "Color",
            Filter::Chromatic(_) => "Chromatic aberration",
            Filter::GradientMap(_) => "Gradient map",
        }
    }

    /// Whether this filter reads **neighbouring** texels rather than only the one it
    /// writes — the point/resampling split §21.3.1 draws, as a value rather than as
    /// prose.
    ///
    /// It is asked by the merge (§14.11.7), and what it decides there is not a
    /// preference but a law: every pass that writes tiles must be a pure function of
    /// canvas position (§6.4), because that is what makes a tile's apron
    /// bit-identical to its neighbour's interior without a copy pass. A point filter
    /// is such a function; a gather over tiles is not, at any apron width, since its
    /// reach is the document's to set. So a filter that resamples cannot be baked
    /// into the paint beneath it, and this is where that is said once.
    ///
    /// An exhaustive match rather than a default, so a new kind has to answer the
    /// question — a resampling kind that inherited "no" would merge to a seam at
    /// every tile boundary, which is the one failure the apron exists to rule out.
    pub fn resamples(&self) -> bool {
        match self {
            Filter::Color(_) | Filter::GradientMap(_) => false,
            Filter::Chromatic(_) => true,
        }
    }

    /// Whether this filter changes anything at all. A neutral filter still costs its
    /// pass — the compositor cannot know that `1.0` is the identity of a gain — so
    /// this is what lets the draw list leave it out (§21.3).
    pub fn is_neutral(&self) -> bool {
        match self {
            Filter::Color(c) => *c == ColorAdjust::NEUTRAL,
            // The spread alone: at zero spread every wavelength lands where it
            // started, whatever the angle points at — so an angle dialled before
            // the spread is not an edit yet, and the draw list rightly spends
            // nothing on it.
            Filter::Chromatic(c) => c.spread == 0.0,
            // No ramp, no map. There is no "identity ramp" to compare against —
            // even black-to-white repaints every color with its greyscale — so
            // the absence is the one neutral this kind has.
            Filter::GradientMap(g) => g.is_none(),
        }
    }

    /// The same **kind** of filter at its neutral setting — what the bar's
    /// "Neutral" chip puts back, said once here rather than per panel arm.
    pub fn neutral(&self) -> Self {
        match self {
            Filter::Color(_) => Filter::Color(ColorAdjust::NEUTRAL),
            Filter::Chromatic(_) => Filter::Chromatic(ChromaticAberration::NEUTRAL),
            Filter::GradientMap(_) => Filter::GradientMap(None),
        }
    }

    /// The same filter with every parameter finite and in range — the funnel every
    /// filter passes through on its way into the document.
    ///
    /// Applied in two places, and both matter. Where the action is **minted**
    /// (`Engine::process`), so the log records what was actually applied and replay
    /// puts it back rather than re-deriving it. And where a filter **enters state**
    /// (`DocState::set_filter` / `insert_filter`), because a loaded file or a
    /// remote peer reaches state without ever passing through `process` —
    /// idempotent for any log this engine wrote, and the only line of defence
    /// against one it did not. The bounds matter more than they look: a `NaN`
    /// saturation reaches a fullscreen pass and poisons every texel of the frame,
    /// and nothing downstream can notice.
    pub fn sanitized(self) -> Self {
        match self {
            Filter::Color(c) => Filter::Color(c.sanitized()),
            Filter::Chromatic(c) => Filter::Chromatic(c.sanitized()),
            // A `Gradient`'s structural invariants (two stops, ascending, finite)
            // are already held by construction — deserialization funnels through
            // `Gradient::new` — so what is left to hold is the *range*: stop
            // colors are straight sRGB, and a finite 1e30 would reach every
            // texel just as surely as a NaN saturation. Clamping to the cube and
            // re-funnelling keeps the promise one gate deep; a ramp that somehow
            // degenerates under it falls back to the neutral, the one answer
            // that cannot make a picture worse.
            Filter::GradientMap(g) => Filter::GradientMap(g.and_then(|g| {
                let stops = g
                    .stops()
                    .iter()
                    .map(|s| crate::gradient::GradientStop {
                        t: s.t,
                        color: s.color.map(|c| c.clamp(0.0, 1.0)),
                    })
                    .collect();
                crate::gradient::Gradient::new(stops)
            })),
        }
    }
}

/// A color adjustment: the knobs that between them cover "make this read warmer /
/// flatter / stronger" (§21.5).
///
/// All of them are applied in **Oklab**, which is the whole reason they are these
/// and not the seven a levels dialog would offer: in a perceptual space, lightness,
/// chroma and hue are separable, so moving one leaves the other two where they were.
/// Saturation in sRGB shifts hue; contrast in sRGB shifts saturation. Here neither
/// does, and that is what makes a slider mean one thing.
///
/// **Three of them are one gesture.** [`hue`](Self::hue),
/// [`saturation`](Self::saturation) and [`tint`](Self::tint) are a rotation, a scale
/// and a translation of the same `(a, b)` plane — between them the whole affine map
/// `ab' = tint + saturation · R(hue) · ab` — which is why the panel shows them as one
/// directed circle rather than as three tracks (§21.6). The order is fixed and it is
/// the order they are listed in: rotate, scale, then translate. Nothing here depends
/// on that being *presented* as a circle; the circle depends on it.
///
/// [`NEUTRAL`](Self::NEUTRAL) is the identity, and it is the value a filter layer is
/// created holding — a new filter changes nothing until it is dialled, which is what
/// keeps "add a filter" from being a destructive act.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorAdjust {
    /// Exposure, in **stops**: the light beneath is scaled by `2^exposure`, so `+1`
    /// is twice the light and `-1` is half.
    ///
    /// Stops rather than a percentage because that is the unit the quantity is
    /// linear in — a stop is the same change everywhere in the range, where "+20%
    /// brightness" is a different change in the shadows than in the highlights. It
    /// is applied to *light* (the same normalized XYZ the blend modes combine in,
    /// §18.0.4) rather than to Oklab's `L`, because doubling light is what an
    /// exposure is and cube-rooting it first would make the number mean nothing.
    pub exposure: f32,
    /// Contrast: a gain on Oklab `L` about mid-grey. `1` is the identity, `0`
    /// flattens the picture to mid-grey, `2` doubles the spread.
    ///
    /// About mid-grey rather than about the picture's own mean, which is the other
    /// thing this could have meant: a pivot that depends on the image makes the
    /// adjustment depend on what is underneath the filter, so moving a layer below
    /// it would change what the slider does. See `stark-engine`'s `filters::CONTRAST_PIVOT`.
    pub contrast: f32,
    /// Saturation: a gain on Oklab chroma — the distance of `(a, b)` from the
    /// achromatic axis. `1` is the identity, `0` is a greyscale conversion that
    /// keeps every lightness exactly where it was, and past `1` is a boost.
    ///
    /// A greyscale at `0` for free, and *correctly*: dropping chroma in Oklab is the
    /// desaturation a luminance-weighted RGB average only approximates.
    pub saturation: f32,
    /// Hue rotation of the Oklab `(a, b)` plane, in **radians**, clockwise from
    /// red toward yellow.
    ///
    /// Radians because that is the unit the engine states angles in
    /// ([`ViewCommand::SetRotation`](crate::command::ViewCommand::SetRotation)); the
    /// frontend offers degrees, which is a way of *presenting* an angle.
    pub hue: f32,
    /// A color cast: `(a, b)` added to Oklab **after** the rotation and the chroma
    /// gain. `[0, 0]` is the identity.
    ///
    /// Which makes it exactly **the color a grey becomes**: an achromatic texel
    /// arrives at the origin of the `(a, b)` plane and nothing before this moves it,
    /// so the pair is the chroma the whole picture is pushed toward. That is the one
    /// thing the other three cannot do — a gain and a rotation both fix the
    /// achromatic axis, so no setting of them tones a grey — and it is what makes
    /// `saturation: 0` plus a tint a duotone rather than only a greyscale.
    ///
    /// Added last for the reason [the struct docs](Self) give, and stated as the
    /// plane's own two axes rather than as a chroma and an angle, because that is
    /// what the shader adds and what the panel drags: a polar spelling would need a
    /// convention for the angle at zero chroma, and would be a second place for an
    /// angle to disagree with [`hue`](Self::hue).
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
    /// the range [`sanitized`](Self::sanitized) holds a log entry to.
    ///
    /// Bounded at all because a fullscreen pass has no coverage to hide behind: a
    /// value from a file or a peer reaches every texel of the frame, so the numbers
    /// that reach the shader have to be numbers.
    pub const EXPOSURE: (f32, f32) = (-4.0, 4.0);
    pub const CONTRAST: (f32, f32) = (0.0, 2.0);
    pub const SATURATION: (f32, f32) = (0.0, 2.0);
    /// A full turn either way, so every rotation is reachable and none is reachable
    /// twice by more than a lap.
    pub const HUE: (f32, f32) = (-std::f32::consts::PI, std::f32::consts::PI);
    /// Per component, on each axis of the `(a, b)` plane.
    ///
    /// `0.16` is about as far from the achromatic axis as the sRGB gamut itself
    /// reaches at mid-grey, so it is the point at which a cast has stopped being a
    /// cast and become a color: past it every texel in the picture is out of gamut
    /// on the same side, and the pass returns a flat wash whatever was underneath.
    /// A square bound rather than a disc for the reason the pair is Cartesian at all
    /// — it is what the shader adds — and the corners it admits are reachable
    /// settings rather than a region needing its own rule.
    pub const TINT: (f32, f32) = (-0.16, 0.16);

    /// Every knob finite and in range — see [`Filter::sanitized`].
    ///
    /// A non-finite value falls back to the **neutral** setting for that knob rather
    /// than to a bound: `NaN` says nothing about which end of the range was meant,
    /// and the identity is the one answer that cannot make a picture worse.
    pub fn sanitized(self) -> Self {
        let clamp = |v: f32, neutral: f32, (lo, hi): (f32, f32)| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                neutral
            }
        };
        Self {
            exposure: clamp(self.exposure, 0.0, Self::EXPOSURE),
            contrast: clamp(self.contrast, 1.0, Self::CONTRAST),
            saturation: clamp(self.saturation, 1.0, Self::SATURATION),
            hue: clamp(self.hue, 0.0, Self::HUE),
            // Per component, so one unusable axis of a cast does not throw away the
            // other — the same reason each scalar knob falls back on its own.
            tint: [
                clamp(self.tint[0], 0.0, Self::TINT),
                clamp(self.tint[1], 0.0, Self::TINT),
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
/// spectrum** rather than the three shifted copies most applications draw (§21.10).
///
/// Two numbers describe the whole effect because they describe the lens: how far
/// the spectrum is pulled apart, and along which axis. Everything else — the
/// asymmetric spread of the blue end, the rainbow ordering of the fringe, the fact
/// that a flat field is untouched — is the physics, computed in the pass rather
/// than parameterized.
///
/// Both are stated **in canvas terms** (canvas px, canvas angle): the fringes
/// belong to the artwork, so they scale with a zoom and turn with a rotation the
/// way the paint does. The trip into screen texels is the encoder's, per frame,
/// through the view's own linear map.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChromaticAberration {
    /// How far the red end of the spectrum lands from the blue end, in **canvas
    /// px** — the full width of the fringe an edge grows. `0` is the identity: no
    /// wavelength moves, whatever the angle says.
    ///
    /// A distance rather than a unitless "amount" because that is what the knob
    /// does to the picture, and because bounding it is what keeps the pass's tap
    /// budget honest (§21.10).
    pub spread: f32,
    /// The axis the spectrum spreads along, in **radians**, canvas space: the
    /// direction the blue end is carried, with the red end opposite. The picture
    /// itself stays put — the two ends part symmetrically around it.
    ///
    /// Radians because that is the unit the engine states angles in; the frontend
    /// offers degrees, which is a way of *presenting* an angle (§21.5's own
    /// argument for the hue knob).
    pub angle: f32,
}

impl ChromaticAberration {
    /// The identity: nothing spreads.
    pub const NEUTRAL: Self = Self {
        spread: 0.0,
        angle: 0.0,
    };

    /// The widest each knob may be dialled — the range a frontend's slider spans
    /// and the range [`sanitized`](Self::sanitized) holds a log entry to.
    ///
    /// The spread's ceiling is also a promise to the renderer: the pass buys taps
    /// in proportion to the on-screen dispersion and caps them (§21.10), and this
    /// bound is what keeps the cap out of reach at working zooms.
    pub const SPREAD: (f32, f32) = (0.0, 128.0);
    /// A full turn either way, so every axis is reachable and none is reachable
    /// twice by more than a lap.
    pub const ANGLE: (f32, f32) = (-std::f32::consts::PI, std::f32::consts::PI);

    /// Every knob finite and in range — see [`Filter::sanitized`]. A non-finite
    /// value falls back to the **neutral** setting for that knob, for
    /// [`ColorAdjust::sanitized`]'s reason: `NaN` says nothing about which end of
    /// the range was meant, and the identity cannot make a picture worse.
    pub fn sanitized(self) -> Self {
        let clamp = |v: f32, neutral: f32, (lo, hi): (f32, f32)| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                neutral
            }
        };
        Self {
            spread: clamp(self.spread, 0.0, Self::SPREAD),
            angle: clamp(self.angle, 0.0, Self::ANGLE),
        }
    }
}

impl Default for ChromaticAberration {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A filter arriving from a file or a peer reaches **every texel** of the frame,
    /// so the one thing that must not survive the way in is a value that is not a
    /// number. A `NaN` here would come back as a white or black frame with nothing
    /// able to say why.
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
    }

    /// Zero spread is the identity **whatever the angle says** — at spread 0 no
    /// wavelength moves, so an angle dialled first is not yet an edit and the draw
    /// list must be free to drop the pass (§21.3's "neutral is dropped" rule,
    /// which anything less would quietly break for one knob ordering).
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

    /// The gradient map's sanitizer holds the one thing `Gradient::new` does not:
    /// the **range** of a stop color. Structure (two stops, ascending, finite) is
    /// the constructor's promise; a finite color outside the sRGB cube is still a
    /// value a hostile file can carry, and it reaches every texel of the frame.
    #[test]
    fn sanitizing_a_gradient_map_clamps_its_stops_to_the_cube() {
        use crate::gradient::{Gradient, GradientStop};
        let hot = Gradient::new(vec![
            GradientStop {
                t: 0.0,
                color: [-2.0, 0.5, 1e30],
            },
            GradientStop {
                t: 1.0,
                color: [0.25, 2.0, 0.75],
            },
        ])
        .unwrap();
        let Filter::GradientMap(Some(g)) = Filter::GradientMap(Some(hot)).sanitized() else {
            panic!("a clampable ramp must survive sanitizing as a ramp");
        };
        assert_eq!(g.stops()[0].color, [0.0, 0.5, 1.0]);
        assert_eq!(g.stops()[1].color, [0.25, 1.0, 0.75]);
        // …and an in-range ramp comes through bit for bit: sanitizing is
        // idempotent on anything this engine wrote.
        let clean = Filter::GradientMap(Some(g));
        assert_eq!(clean.clone().sanitized(), clean);
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
}
