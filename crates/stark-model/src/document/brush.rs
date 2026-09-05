//! The brush (§6.2, §6.6): what a stroke is *made with*, as opposed to what a
//! stroke *is*.
//!
//! Everything here is configuration a [`StrokeRecord`](super::action::StrokeRecord)
//! carries — the tip's shape and orientation, the four flux axes that decide how
//! it meets paint already on the canvas, the color jitter, the tapers, and the
//! mapping from pen input to all of the above. None of it is an action; it is the
//! payload one action kind happens to hold, which is why it lives beside
//! `action.rs` rather than inside it.
//!
//! Two properties hold across the whole module, and most of the design follows
//! from them:
//!
//! - **A modulation can only ever scale a parameter down** ([`Modulation`]). Every
//!   bound the renderer derives from these numbers — the frozen-span radius bound,
//!   the region fit, the choice of render path, the flattener's exchange step — is
//!   stated against the brush's own values and stays sound without any part of it
//!   knowing that modulation exists.
//! - **Every number here decides stored pixels**, so replay, goldens and peers have
//!   to agree on it to the last bit (§12.1). That is why the response curve is
//!   rational rather than a `powf`, and why the values that arrive from files,
//!   presets and peers are clamped on the way in rather than trusted.

use serde::{Deserialize, Serialize};

use crate::{at_least_zero, clamp01, finite_or};

/// The brush tip shape (§6.6).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub enum BrushShape {
    /// Procedural soft disc. Edge softness in [0, 1): 0 = very soft, ~1 = hard
    /// edge — meaningful only here, since it is what shapes this tip's falloff.
    Round { hardness: f32 },
    /// A sampled coverage mask, referenced by content id (an imported image).
    Stamp(crate::AssetId),
}

impl Default for BrushShape {
    fn default() -> Self {
        Self::Round {
            hardness: Self::DEFAULT_HARDNESS,
        }
    }
}

impl BrushShape {
    /// The round tip's hardness fallback: used both as [`Default`] and when a
    /// `Stamp` asset fails to resolve and rendering falls back to the round tip
    /// (§6.6), which by construction has no hardness of its own to fall
    /// back on.
    pub const DEFAULT_HARDNESS: f32 = 0.5;

    /// The same tip with its hardness inside the `[0, 1)` it is quoted in — see
    /// [`BrushParams::sanitized`]. A `Stamp` carries no number to hold.
    pub fn sanitized(self) -> Self {
        match self {
            Self::Round { hardness } => Self::Round {
                hardness: clamp01(hardness),
            },
            Self::Stamp(id) => Self::Stamp(id),
        }
    }
}

/// What sets the brush shape's orientation as it sweeps along the stroke
/// (§6.6). The swept-depth integral runs along the stroke's travel direction, so the
/// shape is looked up in a per-orientation prefix-τ texture indexed by the *relative*
/// angle between the shape's native axis and the travel direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default, carbonite::Schema)]
pub enum OrientationSource {
    /// The shape's native axis tracks the stroke tangent — the relative angle is always
    /// 0, so the extent always faces along the motion (the historical behaviour).
    #[default]
    FollowStroke,
    /// The shape stays pinned to the pen's orientation (the tilt azimuth) in canvas
    /// space; as the stroke curves under a fixed pen the extent angle stays put,
    /// like a calligraphy nib.
    Pen,
}

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
    /// that runs dry as it travels see [`BrushParams::drain`]; for a finite carried
    /// glob that depletes as it is laid see [`charge`](Self::charge).
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
    /// Every axis a number, and the four fractions inside the `[0, 1]` their own
    /// docs quote them in — see [`BrushParams::sanitized`].
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

    /// Every number a number — see [`BrushParams::sanitized`]. Amplitudes and
    /// frequencies are floored at zero (an amplitude is a distance the channel
    /// wanders either way, a frequency a scale) and capped by neither, for
    /// [`BrushDynamics::sanitized`]'s reason.
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        Self {
            noise: self.noise,
            frequency: [
                at_least_zero(self.frequency[0], d.frequency[0]),
                at_least_zero(self.frequency[1], d.frequency[1]),
            ],
            amplitude: [
                at_least_zero(self.amplitude[0], d.amplitude[0]),
                at_least_zero(self.amplitude[1], d.amplitude[1]),
                at_least_zero(self.amplitude[2], d.amplitude[2]),
            ],
        }
    }
}

/// What a [`Modulation`] reads off the pen (§6.2).
///
/// Both are already carried per point of the fitted curve
/// ([`ControlPoint`](crate::path::ControlPoint)) and interpolated per swept segment,
/// so a source here costs the renderer nothing to evaluate and nothing to store.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default, carbonite::Schema)]
pub enum ModSource {
    /// Pen pressure: 0 (barely touching) … 1 (pressed home). A mouse reports 1, so a
    /// pressure-driven brush reads as *pressed* rather than as absent — which is why
    /// this is the one that can be the default.
    #[default]
    Pressure,
    /// How far the pen is leaned over: the **length** of the tilt vector, 0 (upright)
    /// … 1 (flat on the page).
    ///
    /// A mouse reports 0, so a tilt-driven parameter sits at its
    /// [`floor`](Modulation::floor) for the whole stroke. That is the honest reading
    /// of "the pen is upright", not a degenerate case to special-case away — a brush
    /// meant to be usable without a tablet says so by leaving the floor off zero.
    Tilt,
}

/// The pen's state at one point of a stroke, as the modulations read it: both
/// sources reduced to the [0, 1] each is quoted in.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PenState {
    pub pressure: f32,
    /// `tilt.length()` — the pen's lean, which is what a modulation asks about.
    /// The *direction* it leans is a separate question, and the one
    /// [`OrientationSource::Pen`] already answers.
    pub tilt: f32,
}

/// One input driving one brush parameter, as a **multiplier** (§6.2).
///
/// The value that reaches the renderer is `param · factor(input)`, and the factor is
/// held to `[0, 1]` by construction: a modulation can only ever scale a parameter
/// *down* from the value its slider shows. **That bound is the whole design.** Every
/// guarantee the rest of the engine derives from `BrushParams` — the frozen-span
/// rule's radius bound (`gpu::stroke::safe_frozen`), the region fit, the choice of
/// render path (`gpu::stroke::dynamics_setup`), the flattener's exchange step — is
/// stated against the brush's own numbers, and stays sound with no part of it
/// learning that modulation exists. A remap that could also scale *up* would put a
/// correction into every one of those places, and a missed one is a stroke that
/// renders differently live and committed (§1.3).
///
/// It costs nothing in expressiveness: a pencil that widens as the pen leans over is
/// the widest radius on the slider with `source = Tilt` and `floor` at the narrow
/// end. The slider is the maximum, and the pen takes it away.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct Modulation {
    /// Which pen axis drives it.
    pub source: ModSource,
    /// The factor at **zero** input, in [0, 1]: how much of the parameter survives a
    /// feather-light touch (or an upright pen). 0 = the parameter vanishes there, 1 =
    /// the modulation does nothing at all.
    pub floor: f32,
    /// Response shape in [-1, 1]. 0 is linear; positive responds **early** (most of
    /// the parameter's range is reached in the first part of the input's), negative
    /// **late**.
    pub curve: f32,
}

/// Clamp on [`Modulation::curve`]'s underlying bias, and so on how steep a response
/// can be: the factor's slope reaches `(1 − k)/k` at one end of the input and
/// `k/(1 − k)` at the other, which at 0.1 is **9**.
///
/// A bound rather than taste. A segment sweeps at *one* value of every parameter, so
/// a steep response is paid for in segments — `gpu::stroke::budget::flatten_tolerance`
/// divides the attribute budget by [`BrushParams::max_slope`] to keep a modulated ramp
/// from drawing as a staircase, exactly as the taper's own slope buys pieces
/// (`gpu::stroke::segments::Taper`). Unbounded steepness would be an unbounded bill.
const MIN_BIAS: f32 = 0.1;

impl Modulation {
    /// A plain linear modulation from `source`, scaling the parameter all the way to
    /// zero — what "size follows pressure" has always meant here.
    pub const fn linear(source: ModSource) -> Self {
        Self {
            source,
            floor: 0.0,
            curve: 0.0,
        }
    }

    /// The multiplier in force at `pen`.
    pub fn factor(&self, pen: PenState) -> f32 {
        let x = match self.source {
            ModSource::Pressure => pen.pressure,
            ModSource::Tilt => pen.tilt,
        };
        let floor = clamp01(self.floor);
        // `floor = 0` leaves this `0.0 + 1.0 * shape(x)`, which is `shape(x)` to the
        // bit — so the default brush's radius is exactly the product it always was,
        // and every golden holds.
        floor + (1.0 - floor) * self.shape(clamp01(x))
    }

    /// The response curve: a **rational** bias, `x / (m(1 − x) + 1)`, monotone from
    /// (0, 0) to (1, 1) for every `m > −1`.
    ///
    /// Rational rather than the usual `xᵞ` because this decides stored pixels, so
    /// replay, goldens and peers have to agree on it to the last bit (§12.1) — the
    /// same requirement that makes `taper_profile` a polynomial. IEEE-754 pins
    /// `+ − × ÷` to a correctly-rounded result; `powf` is not specified at all.
    fn shape(&self, x: f32) -> f32 {
        let m = self.bias();
        if m == 0.0 {
            // The linear case, exactly — not `x / 1.0`, so nothing depends on the
            // division being the identity it happens to be.
            return x;
        }
        x / (m * (1.0 - x) + 1.0)
    }

    /// The bias the response and its slope bound are both written in, from the
    /// `[-1, 1]` knob the UI shows: `curve` mapped affinely onto `[0, 1]` and held
    /// off the ends by `MIN_BIAS`.
    ///
    /// One function rather than the expression twice, because [`bias`](Self::bias)
    /// and [`max_slope`](Self::max_slope) have to agree: the flattener buys segments
    /// against the slope this pins, and an under-estimate is the staircase `MIN_BIAS`
    /// exists to prevent.
    ///
    /// `curve = 0` lands on `k = 0.5` and so on `m = 0` — every step of which is
    /// exact in binary, which is what makes the linear case unconditional rather
    /// than lucky.
    fn k(&self) -> f32 {
        (0.5 * (self.curve.clamp(-1.0, 1.0) + 1.0)).clamp(MIN_BIAS, 1.0 - MIN_BIAS)
    }

    /// `m` in [`shape`](Self::shape), from [`k`](Self::k).
    fn bias(&self) -> f32 {
        1.0 / self.k() - 2.0
    }

    /// The same mapping with both knobs in the range they are quoted in.
    ///
    /// [`factor`](Self::factor) and [`max_slope`](Self::max_slope) already clamp
    /// what they read, so this changes no pixel — it is here so that what is
    /// *stored* is what the sliders can show, and so a `NaN` curve cannot reach a
    /// panel that would render a slider at an impossible position.
    pub fn sanitized(self) -> Self {
        Self {
            source: self.source,
            floor: clamp01(self.floor),
            // `finite_or` first: `clamp` returns the NaN, and 0 is this knob's own
            // neutral (a linear response).
            curve: finite_or(self.curve, 0.0).clamp(-1.0, 1.0),
        }
    }

    /// A bound on `|d factor / d input|`, for the flattener (see `MIN_BIAS`).
    ///
    /// `shape`'s derivative is `(m + 1)/(m(1 − x) + 1)²`, monotone in `x`, so it is
    /// largest at one end or the other: `k/(1 − k)` at 0 and `(1 − k)/k` at 1. The
    /// floor scales the whole factor by `1 − floor`, so it scales the slope too.
    pub fn max_slope(&self) -> f32 {
        let k = self.k();
        let ends = (k / (1.0 - k)).max((1.0 - k) / k);
        (1.0 - clamp01(self.floor)) * ends
    }
}

/// The multiplier for one modulation target — 1 exactly where there is no mapping.
fn mod_factor(m: Option<Modulation>, pen: PenState) -> f32 {
    m.map_or(1.0, |m| m.factor(pen))
}

/// The steepest response across a set of targets — how much finer the path has
/// to be flattened for a modulated ramp to stay smooth (see `MIN_BIAS` and
/// `gpu::stroke::budget::flatten_tolerance`).
///
/// 1 for the unmodulated brush *and* for the plain linear mappings, so the
/// everyday brush flattens on exactly the budget it always did.
fn mod_slope(targets: &[Option<Modulation>]) -> f32 {
    targets
        .iter()
        .flatten()
        .map(Modulation::max_slope)
        .fold(1.0, f32::max)
}

/// The pen mappings every brush has, whatever its [`effect`](BrushParams::effect)
/// (§6.2): the targets that shape the **tip** — its size, its elongation,
/// and how far it settles into the substrate's tooth. The targets that drive an
/// effect's own rates live *with* the effect ([`PaintModulations`],
/// [`EraseModulations`]), so a mapping cannot name a knob its brush does not have.
///
/// Exactly the parameters that already vary **per swept segment**, and no others. A
/// segment carries one size, one set of rates and one orientation (§6.6), so
/// these are the quantities a modulation can reach without changing what a segment
/// *is*; `hardness` (baked into the prefix-τ texture per value) and `charge` (an
/// initial condition, not a rate) cannot be modulated at all, and are left out rather
/// than carried as knobs that would do nothing.
///
/// `None` on a target is not "a modulation with no effect" — it is skipped entirely,
/// so the parameter reaches the renderer as the exact float the slider holds.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct BrushModulations {
    /// Scales [`BrushParams::size`].
    pub size: Option<Modulation>,
    /// Scales [`ToothParams::give`] — how far the tool settles into the canvas's
    /// substrate (§6.4). Mapped to pressure this is the charcoal behaviour: barely
    /// touching the paper the tip has no give and prints the peaks alone, and borne
    /// down it presses past the falls it was bridging, so the grain fills in.
    pub tooth_give: Option<Modulation>,
    /// Scales [`BrushParams::stretch`] — how far the extent elongates along the
    /// brush's facing axis (§6.6). Mapped to [`ModSource::Tilt`] with
    /// [`OrientationSource::Pen`] this is the pencil behaviour: lean the pen over and
    /// the contact patch draws out along the lean, exactly as a real tip's does.
    pub stretch: Option<Modulation>,
}

impl BrushModulations {
    /// The everyday brush: **size follows pressure**, linearly, all the way to
    /// nothing. Stated here rather than wired into the segment generator, which is
    /// what lets it be turned off or pointed somewhere else like any other mapping.
    ///
    /// Not [`Default`], which is "no mapping at all". Both are wanted:
    /// [`BrushParams::default`] takes this, while a preset that means to hold its
    /// width whatever the hand does asks for the empty set and gets it.
    pub const PRESSURE_SIZE: Self = Self {
        size: Some(Modulation::linear(ModSource::Pressure)),
        tooth_give: None,
        stretch: None,
    };

    /// [`Self::PRESSURE_SIZE`] as a function, for `#[serde(default = "…")]` — which
    /// takes a path to call and cannot name a constant. Its one caller is
    /// [`BrushParams::modulation`], which is where the argument for it is written
    /// out.
    pub fn pressure_size() -> Self {
        Self::PRESSURE_SIZE
    }

    pub fn size(&self, pen: PenState) -> f32 {
        mod_factor(self.size, pen)
    }
    pub fn tooth_give(&self, pen: PenState) -> f32 {
        mod_factor(self.tooth_give, pen)
    }
    pub fn stretch(&self, pen: PenState) -> f32 {
        mod_factor(self.stretch, pen)
    }

    /// Every target at once, in the order they are declared above.
    ///
    /// Destructured rather than read field by field, and that is the whole reason
    /// it is written this way: a pattern with no `..` is exhaustive, so adding a
    /// target to the struct stops this compiling until it is listed here. Left as
    /// field accesses, a new target would simply be missing from
    /// [`max_slope`](Self::max_slope) — and an under-estimated slope is not an
    /// error anywhere, it is a modulated ramp quietly drawn as a staircase.
    /// [`PaintModulations::all`] and [`EraseModulations::all`] strike the same
    /// bargain for their targets.
    fn all(&self) -> [Option<Modulation>; 3] {
        let Self {
            size,
            tooth_give,
            stretch,
        } = *self;
        [size, tooth_give, stretch]
    }

    /// Whether any target is mapped.
    pub fn is_active(&self) -> bool {
        self.all().iter().any(Option::is_some)
    }

    /// Every mapped target sanitized, the unmapped ones left unmapped.
    pub fn sanitized(self) -> Self {
        let [size, tooth_give, stretch] = self.all().map(|m| m.map(Modulation::sanitized));
        Self {
            size,
            tooth_give,
            stretch,
        }
    }

    /// The steepest response across these targets (`mod_slope`).
    pub fn max_slope(&self) -> f32 {
        mod_slope(&self.all())
    }
}

/// The pen mappings whose targets exist only while **painting** (§6.2) —
/// the [`PaintEffect`]'s rate and its ceiling. With the effect it modulates
/// ([`PaintEffect::modulation`]) rather than beside the tip's own mappings, so a
/// mapping cannot name a knob its brush does not have.
///
/// The ceiling is a target with a cost the rate does not have: a segment's
/// share of the stroke's coverage has to be remembered at the segment's own
/// ceiling, so a stroke that maps it carries one more accumulator lane per
/// touched tile (`stamp.wesl`'s ceiling lane, §6.2). A brush that leaves it
/// unmapped never pays for it.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct PaintModulations {
    /// Scales [`PaintEffect::flow`] — the brush's own paint, "Flow" in the UI.
    pub flow: Option<Modulation>,
    /// Scales [`PaintEffect::opacity`] — how much of a full stroke this stretch
    /// of it is. Mapped to pressure the mark is full where the pen bore down
    /// and faint where it eased off; where a stroke covers a spot twice, the
    /// spot shows the **larger** of the two ceilings (§6.2, the ceiling lane),
    /// so a light pass back over a heavy mark leaves it, and a heavy pass over
    /// a light one fills it in.
    #[serde(default)]
    pub opacity: Option<Modulation>,
}

impl PaintModulations {
    pub fn flow(&self, pen: PenState) -> f32 {
        mod_factor(self.flow, pen)
    }
    pub fn opacity(&self, pen: PenState) -> f32 {
        mod_factor(self.opacity, pen)
    }

    /// Every target, exhaustively — [`BrushModulations::all`]'s bargain.
    fn all(&self) -> [Option<Modulation>; 2] {
        let Self { flow, opacity } = *self;
        [flow, opacity]
    }

    /// Whether any target is mapped.
    pub fn is_active(&self) -> bool {
        self.all().iter().any(Option::is_some)
    }

    /// Every mapped target sanitized, the unmapped ones left unmapped.
    pub fn sanitized(self) -> Self {
        let [flow, opacity] = self.all().map(|m| m.map(Modulation::sanitized));
        Self { flow, opacity }
    }

    /// The steepest response across these targets (`mod_slope`).
    pub fn max_slope(&self) -> f32 {
        mod_slope(&self.all())
    }
}

/// The pen mappings whose targets exist only while **working wet paint** — the
/// effect's own flow and ceiling, and the four rates of [`BrushDynamics`]
/// (§6.2). With the effect they modulate ([`WetEffect::modulation`]) for
/// [`PaintModulations`]' reason.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct WetModulations {
    /// Scales [`WetEffect::flow`] — the whole of what the tool does, "Flow" in
    /// the UI: mapped to pressure, a light touch lays less *and* smears less.
    pub flow: Option<Modulation>,
    /// Scales [`WetEffect::opacity`] — [`PaintModulations::opacity`]'s target,
    /// reaching what the brush **mints** (§6.2): the `add` paint's ceiling. The
    /// `charge` glob is scaled by the dial alone, being minted once before the
    /// pen has moved; moved paint is under no ceiling at all.
    #[serde(default)]
    pub opacity: Option<Modulation>,
    /// Scales [`BrushDynamics::add`] — the brush's own paint alone, for a brush
    /// that lays more under the pen without working the canvas harder.
    pub add: Option<Modulation>,
    /// Scales [`BrushDynamics::lift`].
    pub lift: Option<Modulation>,
    /// Scales [`BrushDynamics::deposit`].
    pub deposit: Option<Modulation>,
    /// Scales [`BrushDynamics::bleed`].
    pub bleed: Option<Modulation>,
}

impl WetModulations {
    pub fn flow(&self, pen: PenState) -> f32 {
        mod_factor(self.flow, pen)
    }
    pub fn opacity(&self, pen: PenState) -> f32 {
        mod_factor(self.opacity, pen)
    }
    pub fn add(&self, pen: PenState) -> f32 {
        mod_factor(self.add, pen)
    }
    pub fn lift(&self, pen: PenState) -> f32 {
        mod_factor(self.lift, pen)
    }
    pub fn deposit(&self, pen: PenState) -> f32 {
        mod_factor(self.deposit, pen)
    }
    pub fn bleed(&self, pen: PenState) -> f32 {
        mod_factor(self.bleed, pen)
    }

    /// Every target, exhaustively — [`BrushModulations::all`]'s bargain.
    fn all(&self) -> [Option<Modulation>; 6] {
        let Self {
            flow,
            opacity,
            add,
            lift,
            deposit,
            bleed,
        } = *self;
        [flow, opacity, add, lift, deposit, bleed]
    }

    /// Whether any target is mapped.
    pub fn is_active(&self) -> bool {
        self.all().iter().any(Option::is_some)
    }

    /// Every mapped target sanitized, the unmapped ones left unmapped.
    pub fn sanitized(self) -> Self {
        let [flow, opacity, add, lift, deposit, bleed] =
            self.all().map(|m| m.map(Modulation::sanitized));
        Self {
            flow,
            opacity,
            add,
            lift,
            deposit,
            bleed,
        }
    }

    /// The steepest response across these targets (`mod_slope`).
    pub fn max_slope(&self) -> f32 {
        mod_slope(&self.all())
    }
}

/// The pen mappings whose targets exist only while **erasing** (§6.12): the
/// eraser's rate and its ceiling, [`PaintModulations`]' pair on the removing
/// side, carried the same way (the erase sweep's ceiling lane).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct EraseModulations {
    /// Scales [`EraseEffect::flow`] — how fast a light touch feathers the bite in.
    pub flow: Option<Modulation>,
    /// Scales [`EraseEffect::opacity`] — how much of what it finds this stretch
    /// of the stroke may remove. Mapped to pressure, a light touch thins and a
    /// heavy one clears.
    #[serde(default)]
    pub opacity: Option<Modulation>,
}

impl EraseModulations {
    pub fn flow(&self, pen: PenState) -> f32 {
        mod_factor(self.flow, pen)
    }
    pub fn opacity(&self, pen: PenState) -> f32 {
        mod_factor(self.opacity, pen)
    }

    /// Every target, exhaustively — [`BrushModulations::all`]'s bargain.
    fn all(&self) -> [Option<Modulation>; 2] {
        let Self { flow, opacity } = *self;
        [flow, opacity]
    }

    /// Whether any target is mapped.
    pub fn is_active(&self) -> bool {
        self.all().iter().any(Option::is_some)
    }

    /// Every mapped target sanitized, the unmapped ones left unmapped.
    pub fn sanitized(self) -> Self {
        let [flow, opacity] = self.all().map(|m| m.map(Modulation::sanitized));
        Self { flow, opacity }
    }

    /// The steepest response across these targets (`mod_slope`).
    pub fn max_slope(&self) -> f32 {
        mod_slope(&self.all())
    }
}

/// The pen mappings whose targets exist only while **liquifying** (§6.13) —
/// the one rate a [`LiquifyEffect`] has. With the effect it modulates for
/// [`PaintModulations`]' reason: a mapping cannot name a knob its brush does
/// not have.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct LiquifyModulations {
    /// Scales [`LiquifyEffect::strength`] — how hard the paint follows the tip.
    /// Mapped to pressure this is the natural finger: barely touching, the
    /// picture slides a little; borne down, it keeps pace with the hand.
    pub strength: Option<Modulation>,
}

impl LiquifyModulations {
    pub fn strength(&self, pen: PenState) -> f32 {
        mod_factor(self.strength, pen)
    }

    /// Every target, exhaustively — [`BrushModulations::all`]'s bargain.
    fn all(&self) -> [Option<Modulation>; 1] {
        let Self { strength } = *self;
        [strength]
    }

    /// Whether any target is mapped.
    pub fn is_active(&self) -> bool {
        self.all().iter().any(Option::is_some)
    }

    /// Every mapped target sanitized, the unmapped ones left unmapped.
    pub fn sanitized(self) -> Self {
        let [strength] = self.all().map(|m| m.map(Modulation::sanitized));
        Self { strength }
    }

    /// The steepest response across these targets (`mod_slope`).
    pub fn max_slope(&self) -> f32 {
        mod_slope(&self.all())
    }
}

/// The brush's two knobs against the canvas substrate's tooth (§6.4) — one
/// pair, because they are the two halves of one contact model: how far the tip
/// settles, and how sharply it stops. The *substrate* is document state
/// ([`SubstrateId`](crate::SubstrateId)) — a pencil and a loaded brush on the same
/// canvas see the same grain — which is why only these two knobs live on the brush.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct ToothParams {
    /// How much **give** this tool has, in [0, 1]: 1 = infinite give — the tip
    /// follows every fall, the substrate does not break the mark up at all, and
    /// this is the default; 0 = no give at all — it rides the very tops of the
    /// grain, so the mark is what a dry brush leaves.
    ///
    /// **The give, not its inverse, and that is what makes the pen mapping mean
    /// something.** A [`Modulation`] can only ever scale a parameter *down*, so a knob
    /// quoted as the depth of the bite would have made light pressure the solid mark
    /// and a hard press the dry one — backwards for the one mapping this axis exists
    /// for. Quoted as the give, pressure reads the way a hand expects:
    /// [`BrushModulations::tooth_give`] mapped to pressure is the charcoal, barely
    /// touching the paper it prints the peaks alone, and borne down it presses past
    /// the falls it was bridging and the grain fills in.
    ///
    /// It does cost the slider its usual direction — the *interesting* end of this one
    /// is the left. That is the trade, and the mapping is worth more than the habit.
    ///
    /// What it scales is the exposure the stroke presents per unit swept optical
    /// depth, gated per texel by whether the substrate clears the level this tool
    /// settles to (`paint_common.wesl::tooth_gate`). Inert on a `Flat` canvas
    /// whatever it says, because the substrate's rise is 0 there — so the axis is
    /// orthogonal to every golden that paints on `Flat`.
    ///
    /// `#[serde(default = "…")]` and not a bare `#[serde(default)]`: 0 is *maximum*
    /// tooth at this end of the knob, so a file that does not mention the field has
    /// to be told the full give it meant rather than handed a zero (§8).
    #[serde(default = "ToothParams::default_give")]
    pub give: f32,
    /// The **width of the contact transition**, in the rise's own units — height per
    /// reach of travel (§6.4). The band of rise, either side of the follow limit
    /// [`give`](Self::give) sets, over which a texel goes from taking none of the
    /// tip's paint to taking all of it.
    ///
    /// A hard threshold — 0 — is a binary indicator per texel: correct in the mean,
    /// and at canvas resolution it aliases into hard-edged speckle that reads as
    /// dither rather than as tooth. Too wide and the faces are smeared into a flat
    /// grey and the grain stops reading at all.
    /// [`DEFAULT_SOFTNESS`](Self::DEFAULT_SOFTNESS) is the bundled substrates' own
    /// interquartile rise, so the transition spans the grain's natural variation —
    /// which is what a paint that *sits on* the substrate wants. A charcoal or a
    /// soft graphite does not sit on the substrate: the stick crumbles into the
    /// valleys instead of spanning them, so its contact comes on gradually over
    /// several times that band, and a knob is the only way to say so.
    ///
    /// **A width, so it has no ceiling this crate owns** — the same reading as the
    /// flow and the drain ([`BrushParams::sanitized`]). Past about twice the
    /// encodable rise range the whole distribution is inside the band and the gate is
    /// a flat scale factor; that is where the frontend's slider stops, not where the
    /// quantity stops meaning something.
    ///
    /// `#[serde(default = "…")]` rather than a bare `#[serde(default)]`, because a
    /// file saved before this field meant the constant the shader used to carry —
    /// and 0, which is what a plain default hands back, is the hard threshold and
    /// not that (§8).
    #[serde(default = "ToothParams::default_softness")]
    pub softness: f32,
}

impl Default for ToothParams {
    fn default() -> Self {
        Self {
            give: Self::DEFAULT_GIVE,
            softness: Self::DEFAULT_SOFTNESS,
        }
    }
}

impl ToothParams {
    /// The give a brush has when it does not say ([`give`](Self::give)): all of it,
    /// so the tip follows every fall and the substrate breaks nothing up — the mark
    /// a brush made before the tooth existed, to the bit.
    ///
    /// Named rather than spelled `1.0` at the places that need it, because *which*
    /// end of this knob is the inert one is the fact worth being able to look up.
    pub const DEFAULT_GIVE: f32 = 1.0;

    /// [`DEFAULT_GIVE`](Self::DEFAULT_GIVE) as a function, for `#[serde(default = "…")]`.
    fn default_give() -> f32 {
        Self::DEFAULT_GIVE
    }

    /// The contact transition a brush gets when it does not say
    /// ([`softness`](Self::softness)) — the bundled substrates' own interquartile
    /// rise, so the band spans the grain's natural variation.
    ///
    /// **The one place this number is written.** It was `paint_common.wesl`'s
    /// `const TOOTH_SOFTNESS` until the knob existed; the shader takes it as a
    /// uniform now and declares nothing, which is what keeps the default from being
    /// a host transcription of a shader constant (§6.10).
    pub const DEFAULT_SOFTNESS: f32 = 0.5;

    /// [`DEFAULT_SOFTNESS`](Self::DEFAULT_SOFTNESS) as a function, for
    /// `#[serde(default = "…")]` — which takes a path to call and cannot name a
    /// constant.
    fn default_softness() -> f32 {
        Self::DEFAULT_SOFTNESS
    }

    /// Both knobs held to what they are: the give to its quoted `[0, 1]`, the
    /// softness a finite non-negative width — [`BrushParams::sanitized`]'s rules,
    /// stated where the fields live.
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        Self {
            give: clamp01(finite_or(self.give, d.give)),
            softness: at_least_zero(self.softness, d.softness),
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
    /// Inside the effect rather than on [`BrushParams`], because this is the one
    /// brush parameter only painting consumes: an eraser lays nothing a color
    /// could be a property of (§6.12), so a stored erase stroke carries no color
    /// at all. The *hand* still has one while the eraser is in force — that is
    /// frontend state (`stark-dioxus-frontend`'s `BrushConfig`), which remembers the
    /// whole paint effect across the toggle and hands the color to fills besides.
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
    /// [`BrushParams::drain`] for a stroke that does.
    #[serde(default = "PaintEffect::default_flow")]
    pub flow: f32,
    /// Color dynamics (color jitter) — how the applied color varies across the
    /// brush and along the stroke (§6.2). Historized (it changes stored
    /// pixels); the default (amplitude 0) is the constant color.
    #[serde(default)]
    pub color_dynamics: ColorDynamics,
    /// The pen mappings onto this effect's own rates. The tip's mappings live on
    /// [`BrushParams::modulation`].
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
    /// A *rate*, floored but not capped ([`BrushParams::sanitized`]): the
    /// frontend's `MAX_FLOW` is where a slider stops, not where the quantity
    /// stops meaning something.
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
    /// **The quoted range is load-bearing, not taste** ([`BrushParams::sanitized`]
    /// clamps to it): the renderer's per-segment gather reads a snapshot whose
    /// margin is sized by the segment's own travel, so "paint cannot outrun the
    /// brush" is what keeps every read inside it (§6.13).
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
    pub fn set_flow(&mut self, flow: f32) {
        match self {
            Self::Paint(p) => p.flow = flow,
            Self::Wet(w) => w.flow = flow,
            Self::Erase(e) => e.flow = flow,
            Self::Liquify(l) => l.strength = flow,
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
    /// share of [`BrushParams::max_slope`].
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
    /// [`BrushParams::sanitized`].
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

/// Brush configuration: what a stroke's record carries — the shape of the
/// tip, how the swept extent builds, and the effect it has on the canvas.
/// Every field here is read whatever the [`effect`](Self::effect); a knob only
/// one effect consumes lives inside that effect's own variant, the pigment
/// ([`PaintEffect::color`]) above all.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct BrushParams {
    /// How large the mark is, in canvas pixels at full pressure: **the disc the
    /// mark fits in**, for every shape (§6.6). Not a radius — a stamp is
    /// normalized to this reach whatever its own aspect, which is what lets one
    /// number mean the same thing across a round tip and an asset.
    pub size: f32,
    /// Brush tip shape (§6.6).
    pub shape: BrushShape,
    /// How far the extent **elongates along the brush's facing axis** (§6.6), in
    /// `[0, 1)`: the tip is stretched by [`elongation`](Self::elongation)
    /// `s = 1/(1 − stretch)` along that axis and left alone across it, so 0 is the
    /// extent the shape draws and 0.5 is one twice as long as it is wide.
    ///
    /// **The axis is [`orientation`](Self::orientation)'s**, not a second direction to
    /// set. That is the whole of why this is one number: the brush already says which
    /// way it faces, and a tip elongates along the way it faces. With
    /// [`OrientationSource::Pen`] the axis is the pen's tilt azimuth, so a
    /// [`ModSource::Tilt`] mapping onto this is the pencil — lean the pen and the
    /// contact patch draws out along the lean, exactly as a real conical tip's does,
    /// which is also why the mapping is a reciprocal (a cone leaning at `θ` contacts
    /// over `1/cos θ`) rather than a straight ramp. Held there it is a chisel nib, off
    /// a round tip and with no stamp asset at all.
    ///
    /// The renderer never stretches the *mask*: a swept integral of a stretched
    /// extent is the unstretched one read at another angle, over another travel,
    /// with a factor on the result (§6.6), so the prefix-τ volume the brush already
    /// binds is the volume this reads. What it does cost is extent area — the tip
    /// reaches `s` times as far along its axis, so the tiles a segment touches, and
    /// the dynamics loop's dispatch over them, grow with it. That is what
    /// [`MAX_ELONGATION`](Self::MAX_ELONGATION) bounds.
    #[serde(default)]
    pub stretch: f32,
    /// Length of the stroke's **leading taper** — the run over which the tip widens
    /// from a point to its full [`size`](Self::size) — in *units of `size`*,
    /// so 4.0 means four brush sizes of taper (§6.2). 0 = no taper: the
    /// stroke starts at full width, which is the historical behaviour.
    ///
    /// In radii rather than canvas px so a brush keeps its *look* as it is resized:
    /// scale the tip up and the whole mark scales with it, instead of a taper that
    /// was the shape of the stroke at radius 8 becoming a blunt nub at radius 80.
    /// This is what lets a brush mimic an inker's entry stroke.
    #[serde(default)]
    pub start_taper_length: f32,
    /// Length of the stroke's **trailing taper**, in units of
    /// [`size`](Self::size) — [`start_taper_length`](Self::start_taper_length)
    /// measured back from the end of the stroke, for the exit of an inked line.
    ///
    /// Together the two are held to the stroke's own length: if they would overlap
    /// they are scaled down in proportion, so a short flick is a small pointed mark
    /// rather than a sliver (see `gpu::stroke::segments::Taper`).
    #[serde(default)]
    pub end_taper_length: f32,
    /// Reservoir depletion per **radius** travelled: the stroke thins as paint runs
    /// out (§6.2). 0 = inexhaustible — which is what a pen, a charcoal
    /// stick, or an ordinary digital brush wants; a physical loaded brush wants a
    /// small positive value, and 1 is a tool bone dry one radius past the press.
    ///
    /// In radii rather than canvas px for the reason the tapers are
    /// ([`start_taper_length`](Self::start_taper_length)), and it is the stronger
    /// case of the two: [`size`](Self::size) is meant to be a pure *scale* on
    /// the mark, and a falloff quoted in canvas px is exactly what that scale does
    /// not carry — enlarge such a brush and it runs dry a fraction of the way into
    /// its own tip, which is not a bigger version of anything.
    /// [`drain_px`](Self::drain_px) is where it becomes the per-px rate every render
    /// path reads. Shared by both effects: an eraser runs dry as honestly as a
    /// loaded brush does.
    pub drain: f32,
    /// The brush's two knobs against the canvas substrate's tooth (§6.4).
    #[serde(default)]
    pub tooth: ToothParams,
    /// The deposit jitter (§6.2) — color dynamics' sibling for the *amount*: every
    /// texel of a stroke scales the exposure it presents by a factor uniform in
    /// `(1 − ε, 1 + ε)`, `ε` in `[0, 1]` (past 1 the gate would go negative, which
    /// is not a stronger setting but a meaningless one), keyed on the canvas texel
    /// and the stroke's seed. What it buys is freedom from banding: what the
    /// exchange loop's iterative accumulation would pile into coherent bands lands
    /// as per-texel dither, because neighbouring texels accumulate at decorrelated
    /// phases. 0 is the exact gate 1 — bit-identical to the unjittered deposit.
    ///
    /// Outside the [`effect`](Self::effect) because it is not one effect's knob: it
    /// gates the exposure *every* path presents — the swept fast path, the stamp
    /// loop and the erase pass alike — so it sits beside [`tooth`](Self::tooth),
    /// the other per-texel factor on what a stroke presents.
    #[serde(default = "BrushParams::default_jitter")]
    pub jitter: f32,
    /// What orients the shape as it sweeps (§6.6) — the successor to the old
    /// `follow_path`/`angle_jitter` knobs: `FollowStroke` is the former `follow_path =
    /// true`.
    #[serde(default)]
    pub orientation: OrientationSource,
    /// What a stroke of this brush **does**: paint, or erase. The tool's identity,
    /// and each variant carries the knobs that exist only under it — see
    /// [`BrushEffect`].
    #[serde(default)]
    pub effect: BrushEffect,
    /// The pen mappings every brush has, whatever its effect — the tip's own
    /// (§6.2). [`BrushModulations::PRESSURE_SIZE`] by default: the pressure →
    /// radius scaling, held here as data so a preset can drop it or aim it
    /// elsewhere. An effect's own rates are mapped where they live
    /// ([`PaintEffect::modulation`], [`EraseEffect::modulation`]).
    ///
    /// `#[serde(default = "…")]` and not a bare `#[serde(default)]`, for
    /// [`ToothParams::give`]'s reason (§8): [`BrushModulations`]'s own `Default` is
    /// **no mapping at all**, and a bare attribute would hand a file that predates
    /// this field the empty set — which is not what its absence meant. Before the
    /// mappings were data, the pressure → radius scaling was wired into the segment
    /// generator, so every such stroke replays at full width with its taper gone:
    /// a silent, pixel-changing read of an older log, which is the one thing the
    /// format's name-based reconciliation exists to prevent.
    #[serde(default = "BrushModulations::pressure_size")]
    pub modulation: BrushModulations,
}

impl Default for BrushParams {
    fn default() -> Self {
        Self {
            size: 16.0,
            shape: BrushShape::default(),
            stretch: 0.0,
            start_taper_length: 0.0,
            end_taper_length: 0.0,
            drain: 0.0,
            tooth: ToothParams::default(),
            jitter: Self::DEFAULT_JITTER,
            orientation: OrientationSource::default(),
            effect: BrushEffect::default(),
            modulation: BrushModulations::PRESSURE_SIZE,
        }
    }
}

impl BrushParams {
    /// The [`PaintEffect`] in force, on a brush that lays plain paint — the
    /// reading every consumer of a paint-only knob goes through, so "this brush
    /// does not have that knob" is a `None` rather than a number that lies.
    pub fn paint(&self) -> Option<&PaintEffect> {
        match &self.effect {
            BrushEffect::Paint(p) => Some(p),
            BrushEffect::Wet(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// [`paint`](Self::paint), writable.
    pub fn paint_mut(&mut self) -> Option<&mut PaintEffect> {
        match &mut self.effect {
            BrushEffect::Paint(p) => Some(p),
            BrushEffect::Wet(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// The [`WetEffect`] in force, on a brush that works wet paint —
    /// [`paint`](Self::paint) for the loop's own knobs.
    pub fn wet(&self) -> Option<&WetEffect> {
        match &self.effect {
            BrushEffect::Wet(w) => Some(w),
            BrushEffect::Paint(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// [`wet`](Self::wet), writable.
    pub fn wet_mut(&mut self) -> Option<&mut WetEffect> {
        match &mut self.effect {
            BrushEffect::Wet(w) => Some(w),
            BrushEffect::Paint(_) | BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// The brush turned wet in place, and its effect handed back — the editor's
    /// own gesture when a flux slider is first raised on a plain brush, and the
    /// shorthand a test builds a smearing brush with.
    ///
    /// What both kinds hold carries over — the pigment, the opacity and the flow
    /// (each with its mapping), the color dynamics — and the fluxes start at
    /// zero with `add` at its full share, so wetting a brush and touching
    /// nothing lays the paint it always laid. A brush already wet is handed back
    /// untouched; an eraser keeps its opacity and its flow — the rate means the
    /// same thing on every effect — and takes the default wet brush for the
    /// rest.
    pub fn make_wet(&mut self) -> &mut WetEffect {
        self.effect = match self.effect {
            BrushEffect::Wet(w) => BrushEffect::Wet(w),
            BrushEffect::Paint(p) => BrushEffect::Wet(WetEffect {
                color: p.color,
                opacity: p.opacity,
                flow: p.flow,
                dynamics: BrushDynamics::default(),
                color_dynamics: p.color_dynamics,
                modulation: WetModulations {
                    flow: p.modulation.flow,
                    opacity: p.modulation.opacity,
                    ..WetModulations::default()
                },
            }),
            BrushEffect::Erase(e) => BrushEffect::Wet(WetEffect {
                opacity: e.opacity,
                flow: e.flow,
                modulation: WetModulations {
                    flow: e.modulation.flow,
                    opacity: e.modulation.opacity,
                    ..WetModulations::default()
                },
                ..WetEffect::default()
            }),
            // A liquify brush shares nothing the wet effect could keep — no
            // pigment, no ceiling, and its one rate means "follow", not "lay" —
            // so it takes the default wet brush whole, as the eraser nearly does.
            BrushEffect::Liquify(_) => BrushEffect::Wet(WetEffect::default()),
        };
        self.wet_mut().expect("just made wet")
    }

    /// The [`EraseEffect`] in force, on a brush that erases — [`paint`](Self::paint)
    /// from the other side.
    pub fn erase(&self) -> Option<&EraseEffect> {
        match &self.effect {
            BrushEffect::Erase(e) => Some(e),
            BrushEffect::Paint(_) | BrushEffect::Wet(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// [`erase`](Self::erase), writable.
    pub fn erase_mut(&mut self) -> Option<&mut EraseEffect> {
        match &mut self.effect {
            BrushEffect::Erase(e) => Some(e),
            BrushEffect::Paint(_) | BrushEffect::Wet(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// The [`LiquifyEffect`] in force, on a brush that drags the picture —
    /// [`paint`](Self::paint) from the fourth side (§6.13).
    pub fn liquify(&self) -> Option<&LiquifyEffect> {
        match &self.effect {
            BrushEffect::Liquify(l) => Some(l),
            BrushEffect::Paint(_) | BrushEffect::Wet(_) | BrushEffect::Erase(_) => None,
        }
    }

    /// [`liquify`](Self::liquify), writable.
    pub fn liquify_mut(&mut self) -> Option<&mut LiquifyEffect> {
        match &mut self.effect {
            BrushEffect::Liquify(l) => Some(l),
            BrushEffect::Paint(_) | BrushEffect::Wet(_) | BrushEffect::Erase(_) => None,
        }
    }

    /// The pigment a stroke of this brush lays — the laying effect's own color,
    /// and `None` on an eraser, which lays nothing a color could be a property
    /// of (§6.12). The one spelling of "whichever effect carries the color",
    /// so a consumer cannot ask the paint side alone and silently read black
    /// off a wet brush.
    pub fn pigment(&self) -> Option<[f32; 3]> {
        match &self.effect {
            BrushEffect::Paint(p) => Some(p.color),
            BrushEffect::Wet(w) => Some(w.color),
            // Neither lays anything a color could be a property of: the eraser
            // removes (§6.12), the liquify brush moves what is already
            // colored (§6.13).
            BrushEffect::Erase(_) | BrushEffect::Liquify(_) => None,
        }
    }

    /// The color dynamics a stroke of this brush jitters with: the laying
    /// effect's, and the inactive default on an eraser — which has no color to
    /// wander (§6.12). By value because [`ColorDynamics`] is small and `Copy`,
    /// and a borrow would force every eraser call site through a `static`
    /// default.
    pub fn color_dynamics(&self) -> ColorDynamics {
        match &self.effect {
            BrushEffect::Paint(p) => p.color_dynamics,
            BrushEffect::Wet(w) => w.color_dynamics,
            BrushEffect::Erase(_) | BrushEffect::Liquify(_) => ColorDynamics::default(),
        }
    }

    /// The steepest response across **every** active pen mapping — the tip's and
    /// the effect's own — which is what the flattener buys segments against
    /// (`gpu::stroke::budget::flatten_tolerance`). One derivation, so a target
    /// added to either half cannot be missing from the budget.
    pub fn max_slope(&self) -> f32 {
        self.modulation.max_slope().max(self.effect.max_slope())
    }

    /// The deposit jitter a brush gets when it does not say
    /// ([`jitter`](Self::jitter)), sized between two floors. It must clear the f16
    /// tile quantum (relative ≈ 2⁻¹¹ ≈ 0.05%) by a wide margin, so the exchange
    /// loop's stores land at decorrelated phases of the f16 lattice instead of
    /// ratcheting a region coherently into bands — 1% is ~20 quanta of displacement
    /// wherever a gradient is gentle enough to band at all. And it must stay under what reads as texture: at ±1% a strong deposit
    /// shifts by at most a couple of 8-bit levels per texel, at the threshold of
    /// visibility.
    pub const DEFAULT_JITTER: f32 = 0.01;

    /// [`DEFAULT_JITTER`](Self::DEFAULT_JITTER) as a function, for
    /// `#[serde(default = "…")]`.
    fn default_jitter() -> f32 {
        Self::DEFAULT_JITTER
    }

    /// The two taper lengths in **canvas px**: the stored lengths (in radii) scaled
    /// by [`size`](Self::size). Negative or non-finite lengths read as 0 — the
    /// fields arrive from files, presets and peers, and a taper is a length.
    pub fn taper_px(&self) -> (f32, f32) {
        // `f32::max` returns the non-NaN operand, so this also normalizes NaN to 0.
        let r = self.size.max(0.0);
        let px = |len: f32| {
            let px = len.max(0.0) * r;
            if px.is_finite() { px } else { 0.0 }
        };
        (px(self.start_taper_length), px(self.end_taper_length))
    }

    /// The drain falloff in **canvas px⁻¹**: the stored rate (per radius, see
    /// [`drain`](Self::drain)) over [`size`](Self::size) — [`taper_px`](Self::taper_px)
    /// for the reciprocal quantity, and guarding itself the same way, because the
    /// number arrives from files, presets and peers.
    ///
    /// A radius of zero has no reciprocal, so it reads as **inexhaustible** rather
    /// than as the infinity a shader would turn into a NaN falloff. That is the
    /// honest answer as well as the safe one: a tip with no width lays nothing, and
    /// what lays nothing cannot run out.
    pub fn drain_px(&self) -> f32 {
        // `f32::max` returns the non-NaN operand, so a NaN in either field lands on
        // the `is_finite` fallback below by way of a NaN quotient — as does 0/0.
        let px = self.drain.max(0.0) / self.size.max(0.0);
        if px.is_finite() { px } else { 0.0 }
    }

    /// Whether either end of a stroke with this brush tapers.
    pub fn tapers(&self) -> bool {
        let (start, end) = self.taper_px();
        start > 0.0 || end > 0.0
    }

    /// The furthest the extent may be drawn out along its facing axis — what
    /// [`elongation`](Self::elongation) saturates at, and so the factor by which the
    /// worst-case tip outgrows its own radius.
    ///
    /// A bound on *area*, which is why there is one at all: every tile the stretched
    /// tip reaches is a tile the stroke is rasterized into and the dynamics loop
    /// dispatches over, so `s` prices the stroke roughly linearly. Eight is already a
    /// pen laid almost flat; past it the mark stops reading as a wider stroke and
    /// starts reading as a smear the length of the tip.
    pub const MAX_ELONGATION: f32 = 8.0;

    /// The stretch knob's own top: the value at which
    /// [`elongation`](Self::elongation) reaches [`MAX_ELONGATION`](Self::MAX_ELONGATION)
    /// and the knob stops meaning anything (§6.6).
    ///
    /// Here rather than inline at the frontend's stretch slider, because a slider is
    /// only one of the ways a value reaches this field — a file and a peer are two
    /// more, and neither passes through a panel. `MAX_FLOW`'s own doc makes this
    /// argument for the *drag* bindings; the wire is the third way.
    pub const MAX_STRETCH: f32 = 1.0 - 1.0 / Self::MAX_ELONGATION;

    /// The same brush with every number a number, and every number that has a
    /// documented range inside it — the funnel a brush passes through on its way
    /// into the document, exactly as [`Filter::sanitized`](super::Filter::sanitized)
    /// is for a filter (§21.5) and for the same two reasons.
    ///
    /// **It clamps only where this crate already states a range.** The wet axes
    /// (`add` included), the tooth's *give*, either effect's *opacity*, the hardness and the color are
    /// quoted in `[0, 1]` by their own field docs, and so is the deposit
    /// [`jitter`](Self::jitter), whose gate goes negative past 1; the stretch
    /// saturates at [`MAX_STRETCH`](Self::MAX_STRETCH) by construction. Everything
    /// else — the radius, the flows, the drain, the charge, the tapers, the color
    /// dynamics, the tooth's *softness* — is required to be a finite, non-negative
    /// number and nothing more, because the ceilings those have are a *frontend's*
    /// slider ends rather than facts about the quantity, and clamping a document to
    /// one this crate does not own would rewrite brushes that were never wrong.
    ///
    /// Every guard this replaces stays where it is. `taper_px`, `drain_px`,
    /// `elongation` and `stroke_rect` defend themselves against values that never
    /// came through here, which is what keeps an extent honest for a record built
    /// by hand in a test or arriving down a path this funnel does not cover (§12.6).
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        Self {
            size: at_least_zero(self.size, d.size),
            shape: self.shape.sanitized(),
            // Bounded at the knob's own saturation point rather than at 1: past
            // `MAX_STRETCH` the reciprocal is already pinned, so a larger value
            // stored is a number that cannot mean what it says.
            stretch: finite_or(self.stretch, d.stretch).clamp(0.0, Self::MAX_STRETCH),
            start_taper_length: at_least_zero(self.start_taper_length, d.start_taper_length),
            end_taper_length: at_least_zero(self.end_taper_length, d.end_taper_length),
            drain: at_least_zero(self.drain, d.drain),
            tooth: self.tooth.sanitized(),
            // In `[0, 1]` by the field's own doc: the gate `1 + 2ε·centered` is
            // positive for every ε ≤ 1 and meaningless past it.
            jitter: clamp01(finite_or(self.jitter, d.jitter)),
            orientation: self.orientation,
            effect: self.effect.sanitized(),
            modulation: self.modulation.sanitized(),
        }
    }

    /// [`stretch`](Self::stretch) as the factor the extent is drawn out by along
    /// the facing axis: `s = 1/(1 − stretch)`, clamped to
    /// [`MAX_ELONGATION`](Self::MAX_ELONGATION).
    ///
    /// **Exactly 1 at `stretch = 0`**, which is the whole reason the knob is quoted as
    /// the reciprocal's argument rather than as `s` itself: a brush that never heard of
    /// stretch — and one whose modulation is sitting at a zero floor because the pen is
    /// upright or there is no pen — takes the renderer's identity path bit for bit.
    ///
    /// Takes the modulated knob rather than reading [`stretch`](Self::stretch), because
    /// what a [`Modulation`] scales is the knob and not the factor: scaling `s` towards
    /// 0 would *shrink* the tip across its axis at a low tilt, where scaling the knob
    /// walks `s` back to 1 and leaves the shape alone.
    ///
    /// `min`-then-`max` rather than `clamp`, for `clamp01`'s reason and with more
    /// riding on it: `clamp` returns the NaN where these return the other operand, and
    /// the NaN would reach a lane the shaders divide by.
    #[expect(
        clippy::manual_clamp,
        reason = "min-then-max is the NaN policy; `clamp` returns the NaN these reject"
    )]
    pub fn elongation(stretch: f32) -> f32 {
        // Bounded before the divide rather than clamped after it, so a knob past 1 —
        // or a negative one, which is not a squash but no stretch at all — lands on a
        // real factor instead of on an infinity or an inside-out tip.
        //
        // `min` first and `max` second, and that order is the NaN policy: `f32::min`
        // and `f32::max` return the non-NaN operand (the argument at
        // `clamp01`), so this way a NaN knob falls out as the *identity* and the
        // other way it would fall out as the widest extent the brush can ask for.
        1.0 / (1.0 - stretch).min(1.0).max(1.0 / Self::MAX_ELONGATION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pen(x: f32) -> PenState {
        PenState {
            pressure: x,
            tilt: x,
        }
    }

    /// A sweep of curve/floor pairs covering both directions of the bias and both
    /// ends of the floor, plus the exact defaults.
    fn shapes() -> Vec<Modulation> {
        let mut out = Vec::new();
        for source in [ModSource::Pressure, ModSource::Tilt] {
            for curve in [-1.5, -1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0, 1.5] {
                for floor in [0.0, 0.15, 0.5, 0.9, 1.0] {
                    out.push(Modulation {
                        source,
                        floor,
                        curve,
                    });
                }
            }
        }
        out
    }

    /// The property every bound elsewhere in the engine rests on: a modulation can
    /// only take a parameter *away* (see [`Modulation`]). If this ever fails, the
    /// frozen-span radius bound, the region fit and the exchange step are all being
    /// computed against a number the renderer can exceed.
    /// **A brush from before the mappings were data still tapers with pressure.**
    ///
    /// The save format reconciles a file's schema against today's types by *name*
    /// (§8), so a field this build added arrives from its serde default — and that
    /// default is the file's only way of saying what its absence meant. For
    /// [`BrushParams::modulation`] the absence meant the pressure → radius scaling
    /// the segment generator used to carry, not [`BrushModulations`]'s own
    /// `Default`, which is no mapping at all.
    ///
    /// The two disagree by a whole taper: `mod_factor(None, ..)` is exactly 1, so
    /// the empty set replays every such stroke at full width. Nothing else in the
    /// suite can see it — the field is present on everything this build writes.
    #[test]
    fn a_brush_from_before_the_mappings_were_data_still_follows_pressure() {
        /// `BrushParams` as it stood before `modulation` — spelled with the names
        /// the real type carries, since names are what carbonite reconciles on, and
        /// carrying only the fields that have no default of their own.
        #[derive(Serialize, Deserialize, carbonite::Schema)]
        #[serde(rename = "BrushParams")]
        struct OldBrush {
            size: f32,
            shape: BrushShape,
            drain: f32,
        }

        let bytes = carbonite::to_vec_static(&OldBrush {
            size: 24.0,
            shape: BrushShape::default(),
            drain: 0.5,
        })
        .expect("encode the old shape");
        let back =
            carbonite::from_slice_static::<BrushParams>(&bytes).expect("an older shape loads");

        assert_eq!(back.size, 24.0, "what both shapes share comes through");
        assert_eq!(
            back.modulation,
            BrushModulations::PRESSURE_SIZE,
            "an absent modulation means the mapping the generator used to carry",
        );
        // Said as the picture rather than as the value: what the wrong default
        // costs is a stroke that no longer thins as the hand lifts.
        assert!(
            back.modulation.size(pen(0.25)) < back.modulation.size(pen(1.0)),
            "a lighter press must still make a narrower stroke",
        );
    }

    #[test]
    fn a_factor_never_leaves_the_unit_interval() {
        for m in shapes() {
            // Deliberately fed values outside what a pen reports, and NaN: these
            // arrive from files, presets and peers.
            for x in [-5.0, -0.001, 0.0, 0.3, 0.7, 1.0, 1.001, 9.0, f32::NAN] {
                let f = m.factor(pen(x));
                assert!(
                    (0.0..=1.0).contains(&f),
                    "factor({x}) = {f} escaped [0, 1] for {m:?}"
                );
            }
        }
    }

    /// `curve = 0` is the identity, **to the bit** — which is what keeps the default
    /// brush's radius the exact product `radius · pressure · taper` it always was,
    /// and every golden with it.
    #[test]
    fn a_linear_mapping_is_the_bare_input() {
        let m = Modulation::linear(ModSource::Pressure);
        for i in 0..=1000 {
            let x = i as f32 / 1000.0;
            assert_eq!(
                m.factor(pen(x)).to_bits(),
                x.to_bits(),
                "the linear mapping moved {x}"
            );
        }
    }

    /// No mapping is not a mapping with no effect: the parameter is untouched.
    #[test]
    fn an_unmapped_target_is_exactly_one() {
        let none = WetModulations::default();
        assert!(!none.is_active());
        for x in [0.0, 0.25, 1.0] {
            assert_eq!(none.flow(pen(x)), 1.0);
            assert_eq!(none.lift(pen(x)), 1.0);
        }
        assert_eq!(PaintModulations::default().flow(pen(0.25)), 1.0);
        assert_eq!(EraseModulations::default().flow(pen(0.25)), 1.0);
        assert_eq!(PaintModulations::default().opacity(pen(0.25)), 1.0);
        assert_eq!(EraseModulations::default().opacity(pen(0.25)), 1.0);
        // …and the everyday brush maps size alone.
        let m = BrushModulations::PRESSURE_SIZE;
        assert_eq!(m.size(pen(0.4)).to_bits(), 0.4f32.to_bits());
        assert_eq!(m.tooth_give(pen(0.4)), 1.0);
    }

    /// The ceiling is a target on every effect that has a ceiling, and the
    /// brush says so as one bit (`opacity_modulated`) — the bit the renderer
    /// routes on. Turning a brush wet keeps the mapping, like the flow's.
    #[test]
    fn a_mapped_ceiling_is_a_fact_about_the_brush() {
        let steep = Modulation {
            source: ModSource::Pressure,
            floor: 0.0,
            curve: 1.0,
        };
        let mut b = BrushParams::default();
        assert!(!b.effect.opacity_modulated());
        b.paint_mut()
            .expect("the default brush paints")
            .modulation
            .opacity = Some(steep);
        assert!(b.effect.opacity_modulated());
        // Counted by the flattener's budget like every other target.
        assert!(
            b.max_slope() > 1.0,
            "a steep ceiling response costs segments"
        );

        b.make_wet();
        assert_eq!(b.wet().expect("wet").modulation.opacity, Some(steep));
        assert!(b.effect.opacity_modulated());

        b.effect = BrushEffect::Erase(EraseEffect {
            modulation: EraseModulations {
                opacity: Some(steep),
                ..EraseModulations::default()
            },
            ..EraseEffect::default()
        });
        assert!(b.effect.opacity_modulated());
        b.make_wet();
        assert_eq!(b.wet().expect("wet").modulation.opacity, Some(steep));

        b.effect = BrushEffect::Liquify(LiquifyEffect::default());
        assert!(
            !b.effect.opacity_modulated(),
            "a warp has no ceiling to drive"
        );
    }

    /// The curve is a response, so it has to be one: monotone, at the floor when the
    /// pen gives nothing, and at the full parameter when it gives everything.
    #[test]
    fn the_response_is_monotone_between_its_two_ends() {
        for m in shapes() {
            let floor = m.floor.clamp(0.0, 1.0);
            assert!(
                (m.factor(pen(0.0)) - floor).abs() < 1e-6,
                "zero input should give the floor for {m:?}"
            );
            assert!(
                (m.factor(pen(1.0)) - 1.0).abs() < 1e-6,
                "full input should give the whole parameter for {m:?}"
            );
            let mut prev = f32::NEG_INFINITY;
            for i in 0..=200 {
                let f = m.factor(pen(i as f32 / 200.0));
                assert!(f >= prev - 1e-6, "the response fell back at {i} for {m:?}");
                prev = f;
            }
        }
    }

    /// [`Modulation::max_slope`] is what the flattener buys segments against
    /// (`gpu::stroke::flatten_tolerance`), so it has to be a true bound rather than a
    /// typical value — an under-estimate draws a ramp as a staircase.
    #[test]
    fn max_slope_bounds_the_response() {
        for m in shapes() {
            let bound = m.max_slope();
            let h = 1.0 / 4096.0;
            for i in 0..4096 {
                let x = i as f32 * h;
                let slope = (m.factor(pen(x + h)) - m.factor(pen(x))) / h;
                assert!(
                    slope <= bound + 1e-3,
                    "slope {slope} at {x} exceeds the bound {bound} for {m:?}"
                );
            }
        }
    }

    /// The unmodulated brush and every plain linear mapping cost the flattener
    /// nothing: `attribute / 1.0` is the budget it always had.
    #[test]
    fn a_linear_brush_pays_no_extra_flattening() {
        assert_eq!(BrushParams::default().max_slope(), 1.0);
        assert_eq!(BrushModulations::PRESSURE_SIZE.max_slope(), 1.0);
        // A steep one does pay, and the bill is bounded (`MIN_BIAS`) — and it
        // reaches the brush's own slope through either half: the tip's mappings
        // or the effect's.
        let steep = Modulation {
            source: ModSource::Tilt,
            floor: 0.0,
            curve: -1.0,
        };
        let mut through_effect = BrushParams::default();
        through_effect
            .paint_mut()
            .expect("the default brush paints")
            .modulation
            .flow = Some(steep);
        let mut through_tip = BrushParams::default();
        through_tip.modulation.stretch = Some(steep);
        for slope in [through_effect.max_slope(), through_tip.max_slope()] {
            assert!(
                (1.0..=9.0 + 1e-4).contains(&slope),
                "an extreme curve should cost something, and a bounded something: {slope}"
            );
        }
    }

    /// **Nothing that is not a number survives the funnel**, on any field.
    ///
    /// Driven off a poison list applied to every field in turn rather than one
    /// assertion each, so a field added to the brush has an obvious place to be
    /// added and no way to be quietly exempt — the device
    /// `a_view_never_stores_a_number_it_cannot_use` uses for the view's mutators,
    /// and for the same reason: what is being checked is a *class*.
    #[test]
    fn a_sanitized_brush_holds_no_number_a_shader_cannot_use() {
        /// The default brush's paint effect, writable — what a poke at a
        /// paint-only knob reaches through.
        fn paint(b: &mut BrushParams) -> &mut PaintEffect {
            b.paint_mut().expect("the default brush paints")
        }
        /// The brush turned wet and its effect handed back — what a poke at a
        /// wet-only knob reaches through.
        fn wet(b: &mut BrushParams) -> &mut WetEffect {
            b.effect = BrushEffect::Wet(WetEffect::default());
            b.wet_mut().expect("just made wet")
        }
        type Poke = (&'static str, fn(&mut BrushParams, f32));
        let pokes: [Poke; 21] = [
            ("radius", |b, f| b.size = f),
            ("drain", |b, f| b.drain = f),
            ("erase.opacity", |b, f| {
                b.effect = BrushEffect::Erase(EraseEffect {
                    opacity: f,
                    ..EraseEffect::default()
                })
            }),
            ("erase.flow", |b, f| {
                b.effect = BrushEffect::Erase(EraseEffect {
                    flow: f,
                    ..EraseEffect::default()
                })
            }),
            ("tooth.give", |b, f| b.tooth.give = f),
            ("tooth.softness", |b, f| b.tooth.softness = f),
            ("stretch", |b, f| b.stretch = f),
            ("start_taper", |b, f| b.start_taper_length = f),
            ("end_taper", |b, f| b.end_taper_length = f),
            ("color.r", |b, f| paint(b).color[0] = f),
            ("paint.opacity", |b, f| paint(b).opacity = f),
            ("paint.flow", |b, f| paint(b).flow = f),
            ("wet.flow", |b, f| wet(b).flow = f),
            ("wet.add", |b, f| wet(b).dynamics.add = f),
            ("wet.lift", |b, f| wet(b).dynamics.lift = f),
            ("wet.deposit", |b, f| wet(b).dynamics.deposit = f),
            ("wet.charge", |b, f| wet(b).dynamics.charge = f),
            ("wet.bleed", |b, f| wet(b).dynamics.bleed = f),
            ("jitter.amplitude", |b, f| {
                paint(b).color_dynamics.amplitude[1] = f
            }),
            ("jitter.frequency", |b, f| {
                paint(b).color_dynamics.frequency[0] = f
            }),
            ("hardness", |b, f| {
                b.shape = BrushShape::Round { hardness: f }
            }),
        ];
        // Every stored number, gathered per effect — the shared fields, the rate
        // either effect carries, and whichever effect's own knobs this brush has.
        let numbers = |b: &BrushParams| {
            let mut v = vec![
                b.size,
                b.drain,
                b.tooth.give,
                b.tooth.softness,
                b.stretch,
                b.start_taper_length,
                b.end_taper_length,
                b.effect.flow(),
                b.effect.opacity(),
            ];
            if let Some(w) = b.wet() {
                v.extend([
                    w.dynamics.add,
                    w.dynamics.lift,
                    w.dynamics.deposit,
                    w.dynamics.charge,
                    w.dynamics.bleed,
                ]);
            }
            v
        };
        let unit = |b: &BrushParams| {
            let mut v = vec![b.tooth.give, b.effect.opacity()];
            if let Some(w) = b.wet() {
                v.extend([
                    w.dynamics.add,
                    w.dynamics.lift,
                    w.dynamics.deposit,
                    w.dynamics.bleed,
                ]);
            }
            v
        };
        // The colors are neither of the two lists above: a brush color is extended
        // sRGB, so it may be negative and may pass 1 — what it may not be is
        // unbounded or a `NaN` (§6.5, `Srgb`).
        let colors = |b: &BrushParams| {
            let mut v = vec![];
            v.extend(b.paint().map(|p| p.color).unwrap_or_default());
            v.extend(b.wet().map(|w| w.color).unwrap_or_default());
            v
        };
        for (name, poke) in pokes {
            for f in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -9.0, 1e30] {
                let mut brush = BrushParams::default();
                poke(&mut brush, f);
                let clean = brush.sanitized();
                // Every number is a number…
                for v in numbers(&clean) {
                    assert!(v.is_finite(), "{name} = {f} left a non-finite brush");
                    assert!(v >= 0.0, "{name} = {f} left {v}, which is negative");
                }
                // …the ones this crate quotes in [0, 1] are in it…
                for v in unit(&clean) {
                    assert!((0.0..=1.0).contains(&v), "{name} = {f} escaped [0, 1]");
                }
                // …a color is finite and bounded, in either direction…
                for v in colors(&clean) {
                    let bound = crate::Srgb::EXTENT;
                    assert!((-bound..=bound).contains(&v), "{name} = {f} left {v}");
                }
                // …and the stretch cannot outrun its own saturation point.
                assert!((0.0..=BrushParams::MAX_STRETCH).contains(&clean.stretch));
                // Idempotent, or a load would be a small edit every time.
                assert_eq!(clean.sanitized(), clean, "{name} = {f}");
            }
        }
        // An ordinary brush comes through **bit for bit**: this runs on replay, so
        // anything it nudged would move a golden.
        let ordinary = BrushParams {
            size: 40.0,
            stretch: 0.5,
            tooth: ToothParams {
                give: 0.25,
                // Past the default band, and legitimately so: a soft stick spans
                // the grain rather than sitting on it, and the ceiling is a
                // slider's.
                softness: 0.3,
            },
            effect: BrushEffect::Wet(WetEffect {
                opacity: 0.85,
                flow: 2.5, // past the frontend's slider, and legitimately so
                dynamics: BrushDynamics {
                    add: 0.7,
                    lift: 1.0,
                    bleed: 0.95,
                    ..BrushDynamics::default()
                },
                ..WetEffect::default()
            }),
            ..BrushParams::default()
        };
        assert_eq!(ordinary.sanitized(), ordinary);
        // …an ordinary plain-paint brush too, its flow past the slider included…
        let plain = BrushParams {
            effect: BrushEffect::Paint(PaintEffect {
                opacity: 0.85,
                flow: 2.5,
                ..PaintEffect::default()
            }),
            ..BrushParams::default()
        };
        assert_eq!(plain.sanitized(), plain);
        // …and so does an ordinary eraser, its own rate past 1 included.
        let eraser = BrushParams {
            effect: BrushEffect::Erase(EraseEffect {
                opacity: 0.5,
                flow: 2.0,
                ..EraseEffect::default()
            }),
            ..BrushParams::default()
        };
        assert_eq!(eraser.sanitized(), eraser);
    }

    /// **Flow is one knob with one meaning on every effect** (§6.2): the overall
    /// rate, read and written through whichever effect is in force — and carried
    /// whole across [`BrushParams::make_wet`], so wetting a brush moves no
    /// slider. What a wet brush *is* — its axes — is untouched by it.
    #[test]
    fn flow_is_the_one_overall_rate_whatever_the_effect() {
        // Paint → Wet: the rate and its pen mapping carry, the axes start at the
        // defaults, so the brush lays the paint it always laid.
        let mut b = BrushParams::default();
        b.effect.set_flow(1.7);
        assert_eq!(b.effect.flow(), 1.7);
        let w = b.make_wet();
        assert_eq!(w.flow, 1.7, "wetting a brush must not move the Flow slider");
        assert_eq!(w.dynamics, BrushDynamics::default());
        assert_eq!(b.effect.flow(), 1.7, "…and the enum reads the same knob");
        // The slider scales the overall rate, never the axes: a blend brush
        // stays a blend brush at any flow.
        let d = &mut b.wet_mut().expect("wet").dynamics;
        d.add = 0.0;
        d.lift = 0.5;
        b.effect.set_flow(0.3);
        let w = b.wet().expect("wet");
        assert_eq!((w.dynamics.add, w.dynamics.lift), (0.0, 0.5));
        assert_eq!(w.flow, 0.3);
        // Erase → Wet: the rate carries there too — flow means the same thing
        // on both sides of that switch now.
        let mut e = BrushParams {
            effect: BrushEffect::Erase(EraseEffect {
                opacity: 0.5,
                flow: 2.0,
                ..EraseEffect::default()
            }),
            ..BrushParams::default()
        };
        let w = e.make_wet();
        assert_eq!((w.flow, w.opacity), (2.0, 0.5));
    }

    /// **`radius` is a pure scale on the mark**, which is the whole of why `drain` is
    /// quoted per radius (§6.2): enlarge the tip and the stroke has to run dry
    /// proportionally further along, not at the same canvas distance.
    ///
    /// Stated on the *reach* — the travel at which the load reaches zero,
    /// `1/drain_px` — because that is the length the falloff actually draws, and the
    /// claim is that it grows with the tip instead of standing still. A per-canvas-px
    /// `drain` fails this at every radius but the one it was tuned at.
    #[test]
    fn a_bigger_brush_runs_dry_proportionally_further() {
        let at = |radius: f32| BrushParams {
            size: radius,
            drain: 0.25,
            ..BrushParams::default()
        };
        for radius in [1.0, 8.0, 16.0, 100.0, 1000.0] {
            let reach = 1.0 / at(radius).drain_px();
            assert!(
                (reach - 4.0 * radius).abs() <= 1e-3 * radius,
                "0.25 per radius must reach four radii at radius {radius}, not {reach} px"
            );
        }
        // A tip with no width cannot run out, so the reciprocal that does not exist
        // reads as inexhaustible rather than as an infinity the shader would turn into
        // a NaN falloff — the same class the poison test above rules out for the
        // stored fields, asked of the derived one.
        for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let b = BrushParams {
                size: radius,
                ..at(16.0)
            };
            assert_eq!(b.drain_px(), 0.0, "radius {radius} must drain nothing");
        }
        // And zero stays zero at every size, so a brush that never heard of drain
        // takes the shader's identity path whatever it is scaled to.
        for radius in [1.0, 16.0, 1000.0] {
            let b = BrushParams {
                drain: 0.0,
                ..at(radius)
            };
            assert_eq!(b.drain_px(), 0.0);
        }
    }

    /// A modulation is a pure function of floats, so replay, goldens and peers agree
    /// on it — which is only true while it stays clear of the unspecified library
    /// transcendentals (`Modulation::shape`). Cheap standing check that the value is
    /// reproducible within a build at least.
    #[test]
    fn a_factor_is_reproducible() {
        for m in shapes() {
            for i in 0..=64 {
                let x = i as f32 / 64.0;
                assert_eq!(m.factor(pen(x)).to_bits(), m.factor(pen(x)).to_bits());
            }
        }
    }

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

    /// A liquify brush survives the save format's own encode/decode whole — the
    /// schema-by-name reconciliation (§8) carries the new variant and both its
    /// fields, mapping included.
    #[test]
    fn a_liquify_brush_round_trips_through_the_save_format() {
        let b = BrushParams {
            effect: BrushEffect::Liquify(LiquifyEffect {
                strength: 0.65,
                modulation: LiquifyModulations {
                    strength: Some(Modulation::linear(ModSource::Pressure)),
                },
            }),
            ..BrushParams::default()
        };
        let bytes = carbonite::to_vec_static(&b).expect("encode a liquify brush");
        let back = carbonite::from_slice_static::<BrushParams>(&bytes).expect("decode it back");
        assert_eq!(back, b);
    }
}
