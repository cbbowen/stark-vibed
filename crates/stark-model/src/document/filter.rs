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

use crate::finite_in;

/// What a filter layer does to the stack beneath it (§21.2).
///
/// Four variants, because four are built. The enum is the seam the rest of §21.7
/// lands on — motion blur, outline, glow — and per this codebase's own precedent
/// (§1) no variant appears here before it does something to a pixel. A new one may go
/// wherever it reads best — a variant is matched by *name*, not by position, so
/// inserting one does not disturb the filters in saved files (§8, `ActionKind`).
///
/// `Clone` rather than `Copy`, because a gradient map carries a stop list. A filter is
/// still read once per render and once per projection, so the clones stay
/// countable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
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
            Filter::Chromatic(_) | Filter::FocalBlur(_) => true,
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
            // The radius alone, on the chromatic spread's argument: at zero no
            // light moves, whatever aperture it would have moved through — so a
            // shape picked before the radius is dialled is not an edit either.
            Filter::FocalBlur(b) => b.radius == 0.0,
        }
    }

    /// The same **kind** of filter at its neutral setting — what the bar's
    /// "Neutral" chip puts back, said once here rather than per panel arm.
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
            // A `Gradient`'s structural invariants (two stops, ascending, finite) are
            // held by construction — deserialization repairs into them (§22.1) —
            // and its stops' *range* is held by `Srgb`. So this arm has nothing to do,
            // and stays an arm rather than a catch-all: it is where a gradient map's
            // knobs would go if it grew one, and the match is exhaustive for
            // `sanitized`'s usual reason.
            Filter::GradientMap(g) => Filter::GradientMap(g),
            Filter::FocalBlur(b) => Filter::FocalBlur(b.sanitized()),
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
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
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
    /// (`stark-engine`'s `ViewCommand::SetRotation`); the
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
        Self {
            exposure: finite_in(self.exposure, 0.0, Self::EXPOSURE),
            contrast: finite_in(self.contrast, 1.0, Self::CONTRAST),
            saturation: finite_in(self.saturation, 1.0, Self::SATURATION),
            hue: finite_in(self.hue, 0.0, Self::HUE),
            // Per component, so one unusable axis of a cast does not throw away the
            // other — the same reason each scalar knob falls back on its own.
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
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
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
/// The convolution is shape-blind by construction — everything after the kernel's
/// one compute pass multiplies transforms, whatever they are transforms *of* — so a
/// shape costs a signed distance in `make_kernel` and nothing else. What this enum
/// spends its judgement on is therefore not which shapes are affordable but which
/// ones a lens actually makes.
///
/// **A variant is a mechanism, not a look.** Each is the continuous family one piece
/// of a lens sweeps out — the iris closing onto its blades, the secondary mirror
/// growing across the middle, the anamorphic element squeezing — so the named looks
/// (hexagonal bokeh, the mirror lens's doughnut, the cat's eye) are *settings* here
/// rather than entries. That is what keeps three variants from being the six a list
/// of looks would have needed, and it is why every one of them carries a knob.
///
/// **Every variant is contained in the disc of the blur's radius**, and that is a
/// contract rather than an aesthetic: the FFT pads its planes by the radius on each
/// side so the circular convolution's wrap-around lands in zeros (§21.12), and a
/// shape reaching past it would carry one edge of the picture onto the other. So the
/// radius is each shape's *circumradius* — a polygon's vertices and an oval's long
/// axis both land on it — which is also what makes the shapes comparable: at one
/// radius they are one bokeh, stopped down differently.
///
/// Each variant carries the parameters its geometry can hold and no others, which is
/// the representation ruling out a class rather than a check remembering to (§1):
/// [`Disc`](Self::Disc) is rotationally symmetric however it is obstructed, so there
/// is no turned disc to express and no arm anywhere that has to ignore an angle.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum Aperture {
    /// A circular aperture, with as much of its middle as one likes taken out of it:
    /// the lens wide open at `obstruction` 0, and a catadioptric — mirror — lens's
    /// doughnut as it grows.
    ///
    /// **One variant, because it is one lens.** A mirror lens's aperture is not a
    /// different shape from a plain one; it is a plain one with the secondary
    /// mirror's shadow across the middle, and the family between them is continuous.
    /// Splitting it in two would put a seam in the middle of a knob and give the
    /// enum two spellings of the setting where they meet — the same argument
    /// [`Blades`](Self::Blades) makes for putting one n-gon behind three named
    /// polygons.
    Disc {
        /// The central shadow's share of the radius, held to
        /// [`OBSTRUCTION`](Aperture::OBSTRUCTION). `0` is the plain disc, and the
        /// neutral: it is what every file written before the aperture existed meant
        /// by saying nothing.
        ///
        /// The ceiling is the renderer's as much as the eye's: past it the rim is
        /// thin enough that decimating the convolution (§21.12) would sample it
        /// rather than resolve it, and the bound is what keeps that trade honest for
        /// the one setting of the one shape that has a thin part.
        obstruction: f32,
    },
    /// The iris stopped down onto its blades — a regular polygon of `count` sides,
    /// turned by `angle`.
    ///
    /// One n-gon behind pentagon, hexagon and octagon, which is what a 5-, 6- and
    /// 8-bladed iris each make: the unification §20 makes of 1-, 2- and 3-point
    /// perspective, for the same reason — named cases differing in one number are
    /// one case with the number exposed.
    Blades {
        /// How many blades, held to [`BLADES`](Aperture::BLADES). Below three there
        /// is no polygon, and much past a dozen there is no telling one from the
        /// [`Disc`](Self::Disc) that is already a variant.
        count: u32,
        /// The iris's turn, in radians and **canvas space** — the frame
        /// [`ChromaticAberration::angle`] is stated in, for the same reason (§6.4):
        /// the bokeh belongs to the artwork, so the polygon turns when the canvas is
        /// turned. The renderer makes the trip per frame.
        ///
        /// A polygon's own symmetry makes this periodic in `2π/count`, but the range
        /// spans a full turn like every other angle here: it is the hand's range,
        /// not the shape's.
        angle: f32,
    },
    /// The oval an anamorphic lens makes: a cylindrical front element squeezes the
    /// circle of confusion along one axis, and every out-of-focus highlight stretches
    /// with it.
    Oval {
        /// The long axis's length over the short one's, held to
        /// [`SQUEEZE`](Aperture::SQUEEZE) — `2` is the anamorphic cinema means when
        /// it says the word. The long axis is the radius, so squeezing narrows the
        /// oval rather than growing it.
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

    /// How far a shape that has a direction may be turned — a whole turn, stated
    /// here rather than borrowed from [`ChromaticAberration::ANGLE`]: that the two
    /// agree is a fact about angles, not a coupling between two filters that would
    /// then have to be kept.
    pub const ANGLE: (f32, f32) = (-std::f32::consts::PI, std::f32::consts::PI);

    /// The central obstruction's range — see [`Aperture::Disc`].
    pub const OBSTRUCTION: (f32, f32) = (0.0, 0.9);

    /// The anamorphic squeeze's range — see [`Aperture::Oval`].
    ///
    /// The floor is `1` — the round oval — rather than anything below it, because
    /// a squeeze under 1 is the *same* set of ovals turned a quarter turn, and one
    /// shape with two spellings is what an angle beside a ratio is for.
    pub const SQUEEZE: (f32, f32) = (1.0, 4.0);

    /// Every shape this build offers, at the setting it is *given* when picked — the
    /// list the bar's run of buttons is built from, in the order it should offer
    /// them: the disc first because it is the neutral and what a lens does when
    /// nothing is stopping it down, then the two ways of departing from it.
    ///
    /// The numbers are defaults, not fudge constants: unobstructed is the plain
    /// lens, six blades is the commonest iris, and 2× is the anamorphic squeeze.
    /// The mirror lens is a *setting* of the first rather than a fourth entry here,
    /// which is the one thing this list gives up by merging the two: the doughnut is
    /// a knob away rather than a click away, and the knob's own words are where it
    /// is now advertised.
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

    /// What this shape is called, in the bar and in the layer row.
    ///
    /// The name of the **family**, not of the setting: an obstructed disc is still
    /// on the "Disc" chip, because a chip whose word changed under the knob beside
    /// it would stop being the thing that says which of the three is lit.
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

    /// Every parameter finite and in range — see [`Filter::sanitized`]. A non-finite
    /// value lands on the knob's own identity for [`ColorAdjust::sanitized`]'s
    /// reason: `NaN` says nothing about which end was meant.
    ///
    /// **The shape itself survives** whatever its numbers do. A variant is a choice
    /// rather than a measurement, so there is no unusable value to read it off — a
    /// `NaN` squeeze is a broken oval, not a request for a disc — and repairing the
    /// one number is the whole of the repair.
    pub fn sanitized(self) -> Self {
        match self {
            Aperture::Disc { obstruction } => Aperture::Disc {
                obstruction: finite_in(obstruction, 0.0, Self::OBSTRUCTION),
            },
            // A count is an integer, so it has no `NaN` to fall back from and the
            // clamp is the whole of it: a zero-bladed iris out of a log this engine
            // did not write becomes the triangle that is the fewest a polygon has.
            Aperture::Blades { count, angle } => Aperture::Blades {
                count: count.clamp(Self::BLADES.0, Self::BLADES.1),
                angle: finite_in(angle, 0.0, Self::ANGLE),
            },
            Aperture::Oval { squeeze, angle } => Aperture::Oval {
                // A `NaN` squeeze falls back to `1` — the un-squeezed oval — exactly
                // as a `NaN` contrast falls back to the gain that changes nothing.
                squeeze: finite_in(squeeze, 1.0, Self::SQUEEZE),
                angle: finite_in(angle, 0.0, Self::ANGLE),
            },
        }
    }
}

impl Default for Aperture {
    /// The unobstructed disc — what a file written before the aperture existed means
    /// by saying nothing (see [`FocalBlur::aperture`]).
    ///
    /// Written out rather than derived: `#[derive(Default)]` can only mark a *unit*
    /// variant, and the disc stopped being one when the mirror lens joined it.
    fn default() -> Self {
        Aperture::Disc { obstruction: 0.0 }
    }
}

/// Focal blur: everything beneath spread through a lens's **circle of confusion** —
/// a true convolution of the light with the aperture's shape, computed by FFT, not a
/// Gaussian standing in for one (§21.12).
///
/// A size and a shape. The size is stated **in canvas px** for the chromatic
/// spread's reason (§21.10): the bokeh belongs to the artwork, so it scales with a
/// zoom and holds its size in an export, and the trip into accumulator texels is the
/// renderer's, per frame, through the view's own linear map.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct FocalBlur {
    /// The radius of the circle of confusion, in **canvas px**. `0` is the
    /// identity: no light moves.
    ///
    /// A radius rather than a diameter because that is what a lens formula hands
    /// over, and a distance rather than a unitless "amount" because that is what
    /// the knob does to the picture — and because bounding it is what keeps the
    /// pass's padded FFT scratch honest (§21.12).
    ///
    /// It is also every [`Aperture`]'s circumradius, which is what lets the shape
    /// change without the padding being recomputed — see there.
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

    /// The widest the radius may be dialled — the range a frontend's slider spans
    /// and the range [`sanitized`](Self::sanitized) holds a log entry to.
    ///
    /// The ceiling is also a promise to the renderer, as the chromatic `SPREAD`'s
    /// is: the FFT pads its scratch by the on-screen radius (§21.12), and this
    /// bound is what keeps that padding affordable at working zooms.
    pub const RADIUS: (f32, f32) = (0.0, 128.0);

    /// Every knob finite and in range — see [`Filter::sanitized`]. A non-finite
    /// value falls back to the **neutral** setting, for
    /// [`ColorAdjust::sanitized`]'s reason: `NaN` says nothing about which end of
    /// the range was meant, and the identity cannot make a picture worse.
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
            // The blur is the one kind whose settings are not all numbers, and the
            // shape survives what its numbers do not: a `NaN` squeeze says nothing
            // about which aperture was chosen, so the fallback is per knob — the
            // radius to 0, the squeeze to the 1 that is its own identity — and the
            // variant stays. Which is also invisible either way, an oval at squeeze
            // 1 being a disc; the reason to prefer it is that it is the smaller
            // repair, not that it looks different.
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
        // **The squeeze floor is the containment contract**, not a matter of taste:
        // the long axis is the radius the renderer padded its planes by, so a
        // squeeze below 1 would put the *other* axis outside that padding and wrap
        // one edge of the picture onto the other (§21.12). The shader floors it too,
        // which is what makes the guarantee structural — this pins that a log never
        // asks it to.
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

    /// A radius-0 blur is the identity **whatever aperture it is set to** — the
    /// chromatic filter's rule above, arrived at from the other direction: at zero
    /// radius no light moves, so the shape is not an edit yet and the draw list
    /// must still be free to drop the pass.
    ///
    /// Worth its own test because the aperture is the first thing a focal blur
    /// holds besides its radius, and "is this filter neutral?" is exactly the
    /// question a new field is most likely to be added to by reflex.
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

    /// A gradient map's stops are inside the cube **before the sanitizer sees
    /// them**, which is what the newtype bought.
    ///
    /// A stop's color is an [`Srgb`](crate::Srgb), and there is no way to build one
    /// outside the cube — so what is left to check is that the *ramp* survives being
    /// given hot values (the clamp must not turn two stops into a degenerate one) and
    /// that the sanitizer is the identity on it.
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
}
