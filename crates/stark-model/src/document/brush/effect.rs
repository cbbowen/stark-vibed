//! What a stroke **does** to the canvas (§6.2, §6.12, §6.13): the four effects, and
//! the knobs each carries because it is the one in force.
//!
//! The tool's identity as a sum rather than a mode flag beside knobs it would
//! silently veto — everything outside [`BrushEffect`] shapes the swept extent, and
//! the effect is what that extent then does with it. The four flux axes, the color
//! jitter and each effect's own rate are here; the tip that carries them is
//! [`BrushParams`](super::BrushParams).
//!
//! Reads `modulation`, since an effect's own rates are mapped where the rates live,
//! and nothing above it.

use serde::{Deserialize, Serialize};

use super::modulation::{
    EraseModulations, LiquifyModulations, Modulations, PaintModulations, WetModulations,
};
use crate::sanitize::{at_least_zero, clamp01, finite_in, finite_or};

/// How a [`BrushEffect::Wet`] brush interacts with paint already on the canvas
/// (§6.2). One **unified tool** within that effect: every axis is a flux on the
/// single conserved quantity — paint **height** (the amount; §6.1) — and the axes
/// compose freely. [`add`](Self::add) is the only *source* (the brush's own
/// paint); the rest move paint that is already on the canvas, so with `add = 0`
/// the tool conserves height (it only moves paint around). The everyday add-only
/// brush is not a corner of this space — it is [`BrushEffect::Paint`], a separate
/// effect that carries no fluxes at all.
///
/// **The axes say what the tool does; [`WetEffect::flow`] says how hard.** Every
/// axis here is quoted at the neutral flow of 1, and the effect's flow scales the
/// whole loop — mint, exchange and bleed together — which is what keeps "Flow"
/// meaning the same thing on a wet brush as on every other effect (§6.2).
///
/// Two axes are **vertical** flux between the canvas and a transient
/// per-stroke *tool* reservoir — Lagrangian, giving crisp long-range *directed*
/// transport:
/// - [`lift`](Self::lift)       — lift canvas paint up onto the tool,
/// - [`deposit`](Self::deposit) — lay tool paint back down.
///
/// One is **lateral** flux within the canvas itself, never touching the tool:
/// - [`bleed`](Self::bleed) — the paint under the tip diffuses towards its
///   neighbours (a blur brush alone; wet-softening under `add`).
///
/// `lift`-only is a scraper — it takes paint as a knife does, by the *amount*;
/// the tool an artist calls an eraser acts on what the eye sees instead and is
/// [`BrushEffect::Erase`] (§6.12). `lift`+`deposit` (`add = 0`) is a
/// conservative smudge; `bleed`-only a blur; `add`-only ordinary paint. All flow
/// runs with fixed iteration counts, so replay stays deterministic (§6.2).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct BrushDynamics {
    /// The brush's own paint laid directly: the tool's only source term, in
    /// [0, 1]. 0 = lays none (pure manipulation of existing paint), 1 = the full
    /// mint — a pass at the neutral flow lays a full-thickness deposit.
    ///
    /// What reaches the canvas per unit of swept optical depth (§6.1) is
    /// `add · flow` ([`WetEffect::flow`]): this axis is the *share* of the
    /// brush's own paint in what the tool does, and the flow is how hard the
    /// tool does all of it. At `add = 1` a wet brush lays exactly the paint
    /// [`BrushEffect::Paint`] would at the same flow, so switching a brush
    /// between the two laying effects does not re-interpret its Flow slider.
    ///
    /// A *rate*, not a quantity — this source never runs out on its own. For a stroke
    /// that runs dry as it travels see
    /// [`BrushParams::drain`](super::BrushParams::drain); for a finite carried glob
    /// that depletes as it is laid see [`charge`](Self::charge).
    #[serde(default)]
    pub add: f32,
    /// Canvas paint **lifted** onto the tool per step, as a fraction of the paint present,
    /// in [0, 1]: 0 = none, 1 = lift it all (scrape clean). Vertical flux canvas → tool.
    #[serde(default)]
    pub lift: f32,
    /// Tool paint **deposited** back per step, as a fraction of the paint on the tool, in
    /// [0, 1]: 0 = hold it all (an eraser fills but never lays back), 1 = lay it all
    /// immediately. Vertical flux tool → canvas.
    #[serde(default)]
    pub deposit: f32,
    /// Initial paint **pre-loaded onto the tool** reservoir before the stroke starts, as a
    /// height (the "load a glob on the palette knife" param). 0 = the tool starts empty (the
    /// historical behaviour). It depletes as the tool [`deposit`](Self::deposit)s and refills
    /// as it [`lift`](Self::lift)s — a finite carried amount, unlike the inexhaustible
    /// [`add`](Self::add) source (§6.2). Not scaled by [`WetEffect::flow`]:
    /// the glob is what was scooped, not how hard the hand works it.
    #[serde(default)]
    pub charge: f32,
    /// Canvas paint **diffusing under the tip**, in [0, 1]. The one **lateral** flux,
    /// and it is internal to the canvas — the tool neither takes nor gives (§6.2).
    ///
    /// Unlike its three neighbours the axis is **a diffusivity, not a rate**: it is
    /// linear in `D`, quoted in radius² per pass of the tip, so scrubbing spreads paint
    /// as `σ = sqrt(2·D·τ)` — further the longer you work at it, as a blender does,
    /// rather than converging on a fixed blur. 1 is `D = 0.04`, about `0.28 · radius`
    /// of σ for one pass.
    ///
    /// Quoting it against the **radius** rather than in pixels is what makes it
    /// resolution- and size-independent: the same setting is the same look on any
    /// brush, the property the tapers get from being quoted in radii. The engine
    /// realises `D` by choosing how far the stencil reaches and how hard it relaxes
    /// (`stroke::budget::bleed_stencil`) — a rate alone cannot, because the share that
    /// crosses per step clips at 1 and the axis would stop meaning anything well below
    /// full crank.
    ///
    /// Alone it is a blur brush; alongside [`add`](Self::add) it melts the ridges of the
    /// strokes being painted over instead of leaving their height profile embossed
    /// through the new paint.
    #[serde(default)]
    pub bleed: f32,
}

impl Default for BrushDynamics {
    /// The everyday brush: lay the brush's own paint at the full share, manipulate
    /// nothing — so what the default wet brush lays is decided by its flow alone,
    /// exactly as a paint brush's is.
    fn default() -> Self {
        Self {
            add: 1.0,
            lift: 0.0,
            deposit: 0.0,
            charge: 0.0,
            bleed: 0.0,
        }
    }
}

impl BrushDynamics {
    /// Every axis a number, and the four fractions inside the `[0, 1]` their own docs
    /// quote them in — see [`BrushParams::sanitized`](super::BrushParams::sanitized).
    ///
    /// `charge` is floored but not capped, because it has no documented ceiling
    /// *here*: it is a height, and a bound this crate does not own is not a bound
    /// it may invent — clamping to one would silently rewrite documents whose
    /// brushes were legitimately past a *slider's* end. The rate headroom a hot
    /// brush wants lives on [`WetEffect::flow`], which is unbounded for the same
    /// reason the other effects' rates are.
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        Self {
            add: clamp01(finite_or(self.add, d.add)),
            lift: clamp01(finite_or(self.lift, d.lift)),
            deposit: clamp01(finite_or(self.deposit, d.deposit)),
            charge: at_least_zero(self.charge, d.charge),
            bleed: clamp01(finite_or(self.bleed, d.bleed)),
        }
    }
}

/// The kind of noise field driving [`ColorDynamics`] (§6.2). Each kind is
/// baked into a small tileable 2-D texture (`noise.rs`) — per stroke, from the
/// stroke's seed, and on the CPU, so lookups are cheap and deterministic across
/// replay, peers, and builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default, carbonite::Schema)]
pub enum NoiseKind {
    /// Uncorrelated per-texel randomness — a fine speckle.
    White,
    /// Smooth organic gradient noise (a seamlessly tiling simplex-class noise) —
    /// soft, flowing variation.
    #[default]
    Simplex,
    /// Cellular (Worley F1) noise on a seamlessly tiling jittered grid — mottled
    /// patches with creases where cells meet, like pigment settling in clumps.
    Voronoi,
    /// The discrete form of [`Self::Voronoi`]: each cell one flat color offset,
    /// with a hard edge to its neighbours — crystalline facets rather than a
    /// gradient. All three channels share the same cells, so the facets are
    /// whole polygons of one color.
    Mosaic,
}

/// Color dynamics (color jitter): lets the applied color vary **across the
/// brush and along the stroke** (§6.2). A 3-channel tileable 2-D noise
/// field is sampled in the stroke's **own** frame — `(lateral offset from the
/// centreline, arc length)`, both in canvas px — so the variation belongs to the
/// gesture rather than to the patch of canvas under it: one axis spreads the
/// color across the extent, the other evolves it along the stroke. The three
/// noise channels offset the three color channels *of the current color space*
/// (Oklab `L, a, b`; Mixbox pigment concentrations). The field is baked from
/// the per-stroke `seed`, so each stroke lays its own, deterministically.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct ColorDynamics {
    /// Which noise field to sample.
    pub noise: NoiseKind,
    /// Frequency scale per lookup axis (across the stroke, along it): 1 = one
    /// noise tile per `stark-engine`'s `noise::NOISE_TILE_PX` px; higher = finer
    /// variation along that axis; 0 = constant along that axis.
    pub frequency: [f32; 2],
    /// Noise amplitude per color channel, in the color space's own units
    /// (noise is signed, so a channel wanders ±amplitude). All 0 = off — the
    /// exact historical constant-color deposit.
    ///
    /// Floored at 0 and capped at [`Srgb::EXTENT`](crate::Srgb::EXTENT) by
    /// [`sanitized`](ColorDynamics::sanitized) — the one knob here with a ceiling,
    /// because it is the one this crate *owns*. It is an offset in a color space
    /// this crate defines, so the bound is the same one `Srgb` already states for a
    /// color's distance from zero, and for the same reason: far past any display
    /// gamut, and small enough that a half-float tile cannot overflow through any
    /// pass. A rate's ceiling is a slider's and so may not be invented here
    /// ([`BrushDynamics::sanitized`]); a color-space offset's is not.
    pub amplitude: [f32; 3],
}

impl Default for ColorDynamics {
    fn default() -> Self {
        Self {
            noise: NoiseKind::default(),
            frequency: [1.0; 2],
            amplitude: [0.0; 3],
        }
    }
}

impl ColorDynamics {
    /// Whether the jitter has any effect (any channel amplitude non-zero).
    pub fn is_active(&self) -> bool {
        self.amplitude.iter().any(|a| *a != 0.0)
    }

    /// Every number a number — see
    /// [`BrushParams::sanitized`](super::BrushParams::sanitized). Both are floored at
    /// zero (an amplitude is a distance the channel wanders either way, a frequency a
    /// scale), and only the amplitude is capped: see its own doc for why this is the
    /// bound the crate owns and the frequency's is not.
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        let extent = crate::Srgb::EXTENT;
        Self {
            noise: self.noise,
            frequency: [
                at_least_zero(self.frequency[0], d.frequency[0]),
                at_least_zero(self.frequency[1], d.frequency[1]),
            ],
            amplitude: [
                finite_in(self.amplitude[0], d.amplitude[0], (0.0, extent)),
                finite_in(self.amplitude[1], d.amplitude[1], (0.0, extent)),
                finite_in(self.amplitude[2], d.amplitude[2], (0.0, extent)),
            ],
        }
    }
}

/// The **painting** effect (§6.2): the brush lays its own paint and nothing
/// else — the everyday brush, and the whole of the swept fast path's antialiased
/// deposit. A brush that also *works* what is already there — the smudge, the
/// knife, the blur — is [`BrushEffect::Wet`], a different tool with different
/// available features rather than this one at other rates.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct PaintEffect {
    /// The pigment: straight **sRGB**, components in [0, 1], converted to the
    /// Oklab working space at stamp time (§6.5). Nothing here says how much of it
    /// arrives — the paint a brush lays is per-unit opaque, and "how much shows"
    /// is [`opacity`](Self::opacity) below, a ceiling on the finished stroke —
    /// and there is no fourth component for the same reason: an alpha here scaled
    /// the *material* and so answered a question no digital artist was asking
    /// (§6.2).
    ///
    /// Inside the effect rather than on [`BrushParams`](super::BrushParams), because
    /// this is the one brush parameter only painting consumes: an eraser lays nothing a
    /// color could be a property of (§6.12), so a stored erase stroke carries no color
    /// at all. The *hand* still has one while the eraser is in force — that is frontend
    /// state (`stark-dioxus-frontend`'s `BrushConfig`), which remembers the whole paint
    /// effect across the toggle and hands the color to fills besides.
    ///
    /// A file from before this field replays its strokes in the default brush's
    /// pigment (black) — the bare `#[serde(default)]` (§8).
    #[serde(default)]
    pub color: [f32; 3],
    /// How much of a **full stroke** this stroke is, in [0, 1] — a ceiling on the
    /// stroke's final laid amount, [`EraseEffect::opacity`]'s law run in the laying
    /// direction. The whole stroke's parcel is scaled as one deposit: the coverage
    /// `w` it would have laid saturates at 1 however long the stroke works one
    /// spot, its visible alpha becomes `opacity · w`, and the height is scaled
    /// through the slab law (§6.1) to the amount that shows exactly that — so at
    /// 0.5 a saturated stroke covers half, scrubbing walks its soft edge toward
    /// the cap rather than past it, and a stroke crossing itself never outruns the
    /// dial. The knob a digital artist calls Opacity, beside the
    /// [`flow`](Self::flow) that is the rate.
    #[serde(default = "PaintEffect::default_opacity")]
    pub opacity: f32,
    /// The paint **height** laid per unit of swept optical depth (§6.1) —
    /// this effect's one rate, playing exactly the role [`WetEffect::flow`]
    /// plays for wet paint and [`EraseEffect::flow`] for the eraser: laying is
    /// the whole of what this effect does, so its flow scales the whole effect
    /// as every flow does. Its own field rather than a reading of a
    /// [`BrushDynamics`], for the eraser's reason: this effect has no fluxes,
    /// and a struct of knobs it would silently veto is the shape
    /// [`BrushEffect`]'s own doc forbids.
    ///
    /// A *rate*, not a quantity — it never runs out on its own; see
    /// [`BrushParams::drain`](super::BrushParams::drain) for a stroke that does.
    #[serde(default = "PaintEffect::default_flow")]
    pub flow: f32,
    /// Color dynamics (color jitter) — how the applied color varies across the
    /// brush and along the stroke (§6.2). Historized (it changes stored
    /// pixels); the default (amplitude 0) is the constant color.
    #[serde(default)]
    pub color_dynamics: ColorDynamics,
    /// The pen mappings onto this effect's own rates. The tip's mappings live on
    /// [`BrushParams::modulation`](super::BrushParams::modulation).
    #[serde(default)]
    pub modulation: PaintModulations,
}

impl Default for PaintEffect {
    /// The everyday brush: black, a full stroke, plain flow, the constant color.
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0],
            opacity: 1.0,
            flow: Self::default_flow(),
            color_dynamics: ColorDynamics::default(),
            modulation: PaintModulations::default(),
        }
    }
}

impl PaintEffect {
    /// A paint effect of just a color — the shorthand a test or a preset reaches
    /// for when everything but the pigment is the default.
    pub fn colored(color: [f32; 3]) -> Self {
        Self {
            color,
            ..Self::default()
        }
    }

    /// The opacity a brush gets when it does not say
    /// ([`opacity`](Self::opacity)): a full stroke — for
    /// `#[serde(default = "…")]`, which takes a path to call and cannot name a
    /// constant.
    fn default_opacity() -> f32 {
        1.0
    }

    /// The flow a brush gets when it does not say ([`flow`](Self::flow)) —
    /// shared with [`WetEffect::default`], so the two effects' everyday
    /// brushes lay the same paint.
    fn default_flow() -> f32 {
        0.6
    }
}

/// The **wet** effect (§6.2): the brush lays its own paint *and works what is
/// already there* through the sequential lift/deposit loop — the smudge, the
/// knife, the blur, the loaded brush, and every mixture of them.
///
/// A separate effect rather than a [`PaintEffect`] at other rates, because the
/// two are different tools with different available features, not one tool on
/// two budgets: wet strokes mix with the canvas and carry a reservoir, and in
/// exchange their deposit is point-sampled where paint's is antialiased through
/// the pixel-footprint filter (§6.2) — a trade a brush should make by identity,
/// not by a rate crossing zero.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct WetEffect {
    /// The pigment — [`PaintEffect::color`], and its doc holds here unchanged.
    #[serde(default)]
    pub color: [f32; 3],
    /// The ceiling on a full stroke — [`PaintEffect::opacity`]'s law, **inexact
    /// here by nature**: what the stroke moves it must move whole — conservation
    /// (§6.1) — and once fresh paint is smeared into the picture there is no
    /// longer a "this stroke's share" for a ceiling to scale. The knob scales
    /// what the brush **mints** — the `add` paint and the
    /// [`charge`](BrushDynamics::charge)'s glob — by the same fraction, which
    /// agrees with the ceiling to first order in the amount laid and exactly
    /// at 1.
    #[serde(default = "PaintEffect::default_opacity")]
    pub opacity: f32,
    /// This effect's **overall rate** — how hard a pass of the tip works, playing
    /// exactly the role [`PaintEffect::flow`] plays for paint and
    /// [`EraseEffect::flow`] for the eraser: it scales *everything* the tool does
    /// per pass. The mint is `add · flow` of height per unit swept optical depth;
    /// the exchange runs at `flow` times its per-pass exponents, so a pass at
    /// flow ½ lifts and lays back exactly what half a pass at flow 1 would; the
    /// bleed diffuses `flow` times the diffusivity. 1 is the neutral pass; 0 is a
    /// brush that does nothing at all.
    ///
    /// Its own field beside [`dynamics`](Self::dynamics) rather than an axis
    /// inside it, because it is a different *kind* of number: the axes are the
    /// tool's identity — what a smudge, a knife, a loaded brush *is* — and the
    /// flow is the hand's intensity, the knob the Flow slider and the tuning
    /// drag move whichever effect is in force (§6.2). On a blend brush the
    /// slider therefore blends more or less, instead of turning a blender into
    /// a paint brush — which is what it did while the slider wrote the source
    /// axis.
    ///
    /// A *rate*, floored but not capped
    /// ([`BrushParams::sanitized`](super::BrushParams::sanitized)): the frontend's
    /// `MAX_FLOW` is where a slider stops, not where the quantity stops meaning
    /// something.
    #[serde(default = "PaintEffect::default_flow")]
    pub flow: f32,
    /// The source share and the four fluxes — the unified natural-media tool
    /// (§6.2), quoted at the neutral flow.
    #[serde(default)]
    pub dynamics: BrushDynamics,
    /// Color dynamics — [`PaintEffect::color_dynamics`], unchanged here.
    #[serde(default)]
    pub color_dynamics: ColorDynamics,
    /// The pen mappings onto this effect's own rates ([`WetModulations`]).
    #[serde(default)]
    pub modulation: WetModulations,
}

impl Default for WetEffect {
    /// The everyday wet brush: black, a full stroke, [`PaintEffect`]'s own
    /// default flow over the default dynamics — which lay paint and move none
    /// until a flux is turned up, so the two effects' everyday brushes lay the
    /// same paint.
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0],
            opacity: 1.0,
            flow: PaintEffect::default_flow(),
            dynamics: BrushDynamics::default(),
            color_dynamics: ColorDynamics::default(),
            modulation: WetModulations::default(),
        }
    }
}

/// The **erasing** effect (§6.12): the stroke lays nothing and instead
/// removes what the eye sees. The same swept extent a deposit would rasterize is
/// accumulated across the whole stroke, read as the coverage `w` it would have
/// covered the canvas by, and the paint's **visible** opacity is scaled by
/// `1 − opacity` — inverted through the slab law (§6.1) into the height
/// that shows exactly that much. That inversion is the point: `opacity = 0.5`
/// under a saturated stroke leaves half the opacity that was there, where the
/// `lift` axis — the scraper — removes half the *height*, of which thick paint
/// shows almost nothing.
///
/// The brush's color is ignored — an eraser lays nothing the color could be a
/// property of. The tip, the tooth, the jitter, the tapers and the drain all
/// still apply: they shape *where and how fast* the extent builds, whatever the
/// effect then does with it.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct EraseEffect {
    /// How much of the visible opacity a saturated stroke removes, in [0, 1] — a
    /// **ceiling, not a rate**. `w` saturates at 1 however long a stroke works one
    /// spot, so a stroke removes at most this fraction of what it finds; scrubbing
    /// walks the stroke's soft edge toward the cap rather than eating past it.
    pub opacity: f32,
    /// The rate: how fast a pass builds `w` — the eraser's own flow, playing
    /// exactly the role [`PaintEffect::flow`] plays for paint, and so the knob an
    /// airbrush-style eraser turns down. Its own field rather than a reading of the
    /// paint effect's, so switching a brush's effect never re-interprets a number
    /// that meant something else.
    pub flow: f32,
    /// The pen mappings onto this effect's own rate and ceiling
    /// ([`EraseModulations`]).
    #[serde(default)]
    pub modulation: EraseModulations,
}

impl Default for EraseEffect {
    /// The plain full eraser: everything the stroke covers, removed, building at
    /// the rate one clean pass saturates.
    fn default() -> Self {
        Self {
            opacity: 1.0,
            flow: 1.0,
            modulation: EraseModulations::default(),
        }
    }
}

/// The **liquify** effect (§6.13): the stroke drags the picture itself. The
/// paint under the tip — color, per-unit opacity and height together — follows
/// the travel as a resample of the field, so structure *moves* where the wet
/// loop's smudge would mix it toward a mean: an edge dragged is that edge,
/// displaced, and a texture rides along whole. Nothing is minted and nothing
/// is exchanged — the brush carries no reservoir, no pigment, and no color a
/// jitter could wander.
///
/// A separate effect rather than a [`WetEffect`] at some rate, for the
/// eraser's reason: warping and smearing are different tools with different
/// available features, not one tool on two settings. A smudge trades paint
/// through the tool and conserves height; a warp reparameterizes the canvas
/// and preserves *composition* pointwise instead — every value the stroke
/// leaves is one the field held nearby (§6.13 is where that trade is argued).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct LiquifyEffect {
    /// The **follow fraction**, in [0, 1] — this effect's one rate: how much of
    /// the tip's own travel the paint under full coverage keeps up with, per
    /// pass. At 1 the paint under the tip's core moves with the hand; lower and
    /// it slips behind, so a light setting nudges where a full one carries.
    /// Texels the tip covers more thinly — the shoulder of a soft tip, the gaps
    /// of a textured stamp — follow proportionally less, which is what makes
    /// the falloff the *tip's* rather than a second knob here.
    ///
    /// **The quoted range is load-bearing, not taste**: the renderer's per-segment
    /// gather reads a snapshot whose margin is sized by the segment's own travel, so
    /// "paint cannot outrun the brush" is what keeps every read inside it (§6.13). Both
    /// doors hold it — [`BrushParams::sanitized`](super::BrushParams::sanitized) for
    /// what arrives, and [`BrushEffect::set_flow`] for what a slider writes.
    ///
    /// A fraction of *travel*, so scrubbing keeps carrying — there is no
    /// ceiling a worked spot saturates at, which is why this effect has no
    /// opacity knob at all ([`BrushEffect::opacity`]).
    #[serde(default = "LiquifyEffect::default_strength")]
    pub strength: f32,
    /// The pen mappings onto this effect's own rate ([`LiquifyModulations`]).
    #[serde(default)]
    pub modulation: LiquifyModulations,
}

impl Default for LiquifyEffect {
    /// The plain full drag: the paint under the tip keeps pace with the hand.
    fn default() -> Self {
        Self {
            strength: Self::default_strength(),
            modulation: LiquifyModulations::default(),
        }
    }
}

impl LiquifyEffect {
    /// The follow fraction a brush gets when it does not say
    /// ([`strength`](Self::strength)): the full drag — for
    /// `#[serde(default = "…")]`, which takes a path to call and cannot name a
    /// constant.
    fn default_strength() -> f32 {
        1.0
    }
}

/// What a stroke of this brush **does** (§6.2, §6.12, §6.13) — the tool's
/// identity, as a sum rather than a mode flag beside knobs it would silently
/// veto. Everything outside this enum — the tip, the tooth, the jitter, the
/// tapers, the drain — shapes the swept extent; the effect is what that extent
/// then does to the canvas, and each variant carries exactly the knobs that
/// exist while it is the one in force.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum BrushEffect {
    /// Lay paint (§6.2) — the swept, antialiased deposit.
    Paint(PaintEffect),
    /// Lay paint and work what is already there (§6.2) — the sequential
    /// wet-mixing loop.
    Wet(WetEffect),
    /// Remove visible opacity (§6.12).
    Erase(EraseEffect),
    /// Drag the picture itself (§6.13) — the paint under the tip follows
    /// the travel as a warp of the field, structure carried rather than mixed.
    Liquify(LiquifyEffect),
}

impl Default for BrushEffect {
    /// The everyday brush.
    fn default() -> Self {
        Self::Paint(PaintEffect::default())
    }
}

impl BrushEffect {
    /// A wet effect of a pigment and its axes, **at the neutral flow** — the
    /// shorthand a test reaches for when the color dynamics and the pen mappings
    /// are the defaults. Flow 1 rather than the default brush's, so the axes a
    /// test states *are* the effective per-pass rates it will measure.
    ///
    /// The color is a parameter and not a default on purpose: the pigment lives
    /// *inside* the effect, so a constructor that let a caller build one
    /// without saying a color would be the door through which
    /// `effect: wet_with(..), ..brush(color, r)` silently paints black —
    /// the spread's colored effect replaced whole, with nothing left to say so.
    pub fn wet_with(color: [f32; 3], dynamics: BrushDynamics) -> Self {
        Self::Wet(WetEffect {
            color,
            flow: 1.0,
            dynamics,
            ..WetEffect::default()
        })
    }

    /// [`Paint`](Self::Paint) of just a color — [`wet_with`](Self::wet_with)'s
    /// swept sibling, from the pigment side.
    pub fn painted(color: [f32; 3]) -> Self {
        Self::Paint(PaintEffect::colored(color))
    }

    /// The effect's **overall rate** — "Flow" in the UI, whichever effect is in
    /// force: how hard a pass of the tip works. How much a laying stroke lays
    /// ([`PaintEffect::flow`]), how hard a wet stroke both lays and works the
    /// canvas ([`WetEffect::flow`]), how fast an eraser's bite builds
    /// ([`EraseEffect::flow`]), how hard a liquify stroke drags
    /// ([`LiquifyEffect::strength`]). One meaning with one knob per effect,
    /// which is what lets the brush panel's Flow slider and the tuning drag
    /// tune the tool in hand without asking which kind it is — and without the
    /// slider changing what the tool *is* (that is the effect's own knobs:
    /// [`BrushDynamics`]' axes above all).
    pub fn flow(&self) -> f32 {
        match self {
            Self::Paint(p) => p.flow,
            Self::Wet(w) => w.flow,
            Self::Erase(e) => e.flow,
            Self::Liquify(l) => l.strength,
        }
    }

    /// Write the effect's overall rate — [`flow`](Self::flow)'s other half.
    ///
    /// The three laying rates take the number as given: their ceilings are a
    /// slider's, and a bound this crate does not own is not one it may invent
    /// ([`BrushDynamics::sanitized`]). [`LiquifyEffect::strength`] is the one that
    /// *is* owned here — `[0, 1]` is what keeps the per-segment gather inside the
    /// snapshot its own travel sized (§6.13) — so this door holds it, as
    /// `BrushConfig::params` and `max_flow` already do on the frontend's side.
    ///
    /// Spelled as `sanitized` spells it, so the two doors land a `NaN` in the same
    /// place: full drag, the setting that cannot make a stroke do nothing.
    pub fn set_flow(&mut self, flow: f32) {
        match self {
            Self::Paint(p) => p.flow = flow,
            Self::Wet(w) => w.flow = flow,
            Self::Erase(e) => e.flow = flow,
            Self::Liquify(l) => l.strength = clamp01(finite_or(flow, 1.0)),
        }
    }

    /// The effect's **opacity** — the ceiling on what a saturated stroke does,
    /// whichever effect is in force: how much of a full stroke it lays
    /// ([`PaintEffect::opacity`], [`WetEffect::opacity`]) or removes
    /// ([`EraseEffect::opacity`]). [`flow`](Self::flow)'s sibling, and one
    /// question for one slider for the same reason.
    pub fn opacity(&self) -> f32 {
        match self {
            Self::Paint(p) => p.opacity,
            Self::Wet(w) => w.opacity,
            Self::Erase(e) => e.opacity,
            // A warp has no ceiling for a dial to set: the follow is a fraction
            // of *travel*, so scrubbing keeps carrying the way the bleed keeps
            // buying distance (§6.13), and there is no saturated stroke for an
            // opacity to be a fraction of. 1 is the identity every consumer of
            // this number — the integrate, the mask's fold — expects of "no
            // ceiling".
            Self::Liquify(_) => 1.0,
        }
    }

    /// Write the effect's opacity — [`opacity`](Self::opacity)'s other half.
    /// A no-op on [`Liquify`](Self::Liquify), which has no such knob — the
    /// editor never shows the dial while it is in force, so nothing writes
    /// here; stated as an arm rather than left to a wildcard so a fourth
    /// effect has to answer for itself.
    pub fn set_opacity(&mut self, opacity: f32) {
        match self {
            Self::Paint(p) => p.opacity = opacity,
            Self::Wet(w) => w.opacity = opacity,
            Self::Erase(e) => e.opacity = opacity,
            Self::Liquify(_) => {}
        }
    }

    /// The steepest response across the effect's own pen mappings — the effect's
    /// share of [`BrushParams::max_slope`](super::BrushParams::max_slope).
    pub fn max_slope(&self) -> f32 {
        match self {
            Self::Paint(p) => p.modulation.max_slope(),
            Self::Wet(w) => w.modulation.max_slope(),
            Self::Erase(e) => e.modulation.max_slope(),
            Self::Liquify(l) => l.modulation.max_slope(),
        }
    }

    /// Whether the pen drives this effect's [`opacity`](Self::opacity) — the
    /// one mapping the renderer has to know about *as a brush*, because a stroke
    /// that carries it accumulates one more lane per touched tile and takes the
    /// carried-parcel path whatever the dial says (§6.2). A pure function of the
    /// brush, like every path decision, so a live tail and its commit agree.
    pub fn opacity_modulated(&self) -> bool {
        match self {
            Self::Paint(p) => p.modulation.opacity.is_some(),
            Self::Wet(w) => w.modulation.opacity.is_some(),
            Self::Erase(e) => e.modulation.opacity.is_some(),
            Self::Liquify(_) => false,
        }
    }

    /// Every number a number and every quoted range held — the effect's share of
    /// [`BrushParams::sanitized`](super::BrushParams::sanitized).
    pub fn sanitized(self) -> Self {
        match self {
            Self::Paint(p) => Self::Paint(PaintEffect {
                color: crate::Srgb::new(p.color).get(),
                // In `[0, 1]` by the field's own doc, for the erase twin's
                // reason: a ceiling on the fraction laid, meaningless past 1.
                opacity: clamp01(finite_or(p.opacity, 1.0)),
                // Floored but not capped, for `BrushDynamics::flow`'s reason: a
                // rate, whose ceiling is a slider's.
                flow: at_least_zero(p.flow, PaintEffect::default_flow()),
                color_dynamics: p.color_dynamics.sanitized(),
                modulation: p.modulation.sanitized(),
            }),
            Self::Wet(w) => Self::Wet(WetEffect {
                color: crate::Srgb::new(w.color).get(),
                opacity: clamp01(finite_or(w.opacity, 1.0)),
                // Floored but not capped, for `PaintEffect::flow`'s reason: a
                // rate, whose ceiling is a slider's.
                flow: at_least_zero(w.flow, PaintEffect::default_flow()),
                dynamics: w.dynamics.sanitized(),
                color_dynamics: w.color_dynamics.sanitized(),
                modulation: w.modulation.sanitized(),
            }),
            Self::Erase(e) => Self::Erase(EraseEffect {
                // In `[0, 1]` by the field's own doc: the removal `opacity` is
                // a fraction of the visible opacity, and past 1 it would ask for
                // less than none.
                opacity: clamp01(finite_or(e.opacity, 1.0)),
                // Floored but not capped, for `PaintEffect::flow`'s reason: a
                // rate, whose ceiling is a slider's.
                flow: at_least_zero(e.flow, 1.0),
                modulation: e.modulation.sanitized(),
            }),
            Self::Liquify(l) => Self::Liquify(LiquifyEffect {
                // In `[0, 1]` by the field's own doc, and here the range is a
                // *renderer* invariant rather than a semantic one: the gather's
                // snapshot margin is sized by the segment's travel, and a
                // follow past 1 would read outside it (§6.13). Capped, unlike
                // the flows, because this crate owns that bound.
                strength: clamp01(finite_or(l.strength, 1.0)),
                modulation: l.modulation.sanitized(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The liquify strength's quoted `[0, 1]` is a **renderer** invariant, not
    /// taste (§6.13): the warp's snapshot margin is sized by the segment's own
    /// travel, so a follow past 1 would read outside it. The sanitize is the one
    /// door a wire or file value comes through, so it is where the bound is
    /// pinned — a value past 1 is nonsense, not a stronger drag.
    #[test]
    fn a_liquify_strength_is_held_to_the_range_the_gather_is_sized_by() {
        for (dirty, clean) in [(1.5, 1.0), (-0.25, 0.0), (f32::NAN, 1.0), (0.4, 0.4)] {
            let b = BrushEffect::Liquify(LiquifyEffect {
                strength: dirty,
                ..LiquifyEffect::default()
            })
            .sanitized();
            let BrushEffect::Liquify(l) = b else {
                panic!("sanitize must not change the effect's identity");
            };
            assert_eq!(
                l.strength, clean,
                "strength {dirty} must sanitize to {clean}"
            );
        }
    }
}
