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

/// How a brush interacts with paint already on the canvas (§6.2). One
/// **unified tool**, not a mode switch: every axis is a flux on the single conserved
/// quantity — paint **height** (the amount; §6.1) — and the axes compose freely.
/// [`add`](Self::add) is the only *source* (the brush's own paint); the rest move paint
/// that is already on the canvas, so with `add = 0` the tool conserves height (it only
/// moves paint around). The everyday brush is just `add` with the rest 0 (the default).
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
/// `lift`-only is an eraser; `lift`+`deposit` (`add = 0`) a conservative smudge;
/// `bleed`-only a blur; `add`-only ordinary paint. All flow runs with fixed
/// iteration counts, so replay stays deterministic (§6.2).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct BrushDynamics {
    /// The brush's own paint laid directly: the paint **height** deposited per unit of
    /// swept optical depth (§6.1), and the tool's only source term. 0 = lays
    /// none (pure manipulation of existing paint), 1 = a heavy full-thickness deposit.
    ///
    /// A *rate*, not a quantity — this source never runs out on its own. For a stroke
    /// that runs dry as it travels see [`BrushParams::drain`]; for a finite carried
    /// glob that depletes as it is laid see [`charge`](Self::charge).
    ///
    /// **It means the same amount of paint whatever the other three are doing.** The
    /// axes below decide whether the stroke goes through the swept fast path or the
    /// sequential stamp loop (§6.2), so a gain applied on one path and not the other
    /// would make nudging [`deposit`](Self::deposit) off zero change the flow of a
    /// slider that has nothing to do with it.
    #[serde(default)]
    pub flow: f32,
    /// Canvas paint **lifted** onto the tool per step, as a fraction of the paint present,
    /// in [0, 1]: 0 = none, 1 = lift it all (scrape clean). Vertical flux canvas → tool.
    #[serde(default)]
    pub lift: f32,
    /// Tool paint **deposited** back per step, as a fraction of the paint on the tool, in
    /// [0, 1]: 0 = hold it all (an eraser fills but never lays back), 1 = lay it all
    /// immediately. Vertical flux tool → canvas.
    #[serde(default)]
    pub deposit: f32,
    /// The deposit jitter (§6.2) — color dynamics' sibling for the *amount*: every
    /// texel of a stroke scales the exposure it presents by a factor uniform in
    /// `(1 − ε, 1 + ε)`, `ε` in `[0, 1]` (past 1 the gate would go negative, which
    /// is not a stronger setting but a meaningless one), keyed on the canvas texel
    /// and the stroke's seed. What it buys is freedom from banding: what the
    /// exchange loop's iterative accumulation would pile into coherent bands lands
    /// as per-texel dither, because neighbouring texels accumulate at decorrelated
    /// phases. 0 is the exact gate 1 — bit-identical to the unjittered deposit.
    #[serde(default = "BrushDynamics::default_deposit_jitter")]
    pub deposit_jitter: f32,
    /// Initial paint **pre-loaded onto the tool** reservoir before the stroke starts, as a
    /// height (the "load a glob on the palette knife" param). 0 = the tool starts empty (the
    /// historical behaviour). It depletes as the tool [`deposit`](Self::deposit)s and refills
    /// as it [`lift`](Self::lift)s — a finite carried amount, unlike the inexhaustible
    /// [`add`](Self::add) source (§6.2).
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
    /// The everyday brush: lay the brush's own paint, manipulate nothing.
    fn default() -> Self {
        Self {
            flow: 0.6,
            lift: 0.0,
            deposit: 0.0,
            charge: 0.0,
            bleed: 0.0,
            deposit_jitter: Self::DEFAULT_DEPOSIT_JITTER,
        }
    }
}

impl BrushDynamics {
    /// Every axis a number, and the three fractions inside the `[0, 1]` their own
    /// docs quote them in — see [`BrushParams::sanitized`].
    ///
    /// `add` and `charge` are floored but not capped, because neither has a
    /// documented ceiling *here*: `add` is a rate rather than a fraction (the
    /// frontend's `MAX_FLOW` is where a slider stops, not where the quantity stops
    /// meaning something) and `charge` is a height. A bound this crate does not
    /// own is not a bound it may invent — clamping to one would silently rewrite
    /// documents whose brushes were legitimately past a *slider's* end.
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        Self {
            flow: at_least_zero(self.flow, d.flow),
            lift: clamp01(finite_or(self.lift, d.lift)),
            deposit: clamp01(finite_or(self.deposit, d.deposit)),
            charge: at_least_zero(self.charge, d.charge),
            bleed: clamp01(finite_or(self.bleed, d.bleed)),
            // In `[0, 1]` by the field's own doc: the gate `1 + 2ε·centered` is
            // positive for every ε ≤ 1 and meaningless past it.
            deposit_jitter: clamp01(finite_or(self.deposit_jitter, d.deposit_jitter)),
        }
    }

    /// The deposit jitter a brush gets when it does not say
    /// ([`deposit_jitter`](Self::deposit_jitter)), sized between two floors. It
    /// must clear the f16 tile quantum (relative ≈ 2⁻¹¹ ≈ 0.05%) by a wide
    /// margin, so the exchange loop's stores land at decorrelated phases of the
    /// f16 lattice instead of ratcheting a region coherently into bands — 1% is
    /// ~20 quanta of displacement wherever a gradient is gentle enough to band at
    /// all. And it must stay under what reads as texture: at ±1% a strong deposit
    /// shifts by at most a couple of 8-bit levels per texel, at the threshold of
    /// visibility.
    pub const DEFAULT_DEPOSIT_JITTER: f32 = 0.01;

    /// [`DEFAULT_DEPOSIT_JITTER`](Self::DEFAULT_DEPOSIT_JITTER) as a function, for
    /// `#[serde(default = "…")]`.
    fn default_deposit_jitter() -> f32 {
        Self::DEFAULT_DEPOSIT_JITTER
    }
}

/// The kind of noise field driving [`ColorDynamics`] (§6.2). Each kind
/// is baked once into a small tileable 2-D texture (`noise.rs`), so lookups are
/// cheap and deterministic across replay, peers, and builds.
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
/// (Oklab `L, a, b`; Mixbox pigment concentrations). The per-stroke `seed`
/// translates the lookup so each stroke draws a fresh part of the field,
/// deterministically.
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
/// divides the attribute budget by [`Modulations::max_slope`] to keep a modulated ramp
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

    /// `m` in [`shape`](Self::shape), from the `[-1, 1]` knob the UI shows. `curve = 0`
    /// lands on `k = 0.5` and so on `m = 0` — every step of which is exact in binary,
    /// which is what makes the linear case unconditional rather than lucky.
    fn bias(&self) -> f32 {
        let k = (0.5 * (self.curve.clamp(-1.0, 1.0) + 1.0)).clamp(MIN_BIAS, 1.0 - MIN_BIAS);
        1.0 / k - 2.0
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
        let k = (0.5 * (self.curve.clamp(-1.0, 1.0) + 1.0)).clamp(MIN_BIAS, 1.0 - MIN_BIAS);
        let ends = (k / (1.0 - k)).max((1.0 - k) / k);
        (1.0 - clamp01(self.floor)) * ends
    }
}

/// Which brush parameters the pen drives, and how (§6.2) — the mapping from pen
/// input to brush parameter that makes one tool a brush and another a palette knife.
///
/// Exactly the parameters that already vary **per swept segment**, and no others. A
/// segment carries one size, one set of paint rates and one orientation (§6.6), so
/// these are the quantities a modulation can reach without changing what a segment
/// *is*; `hardness` (baked into the prefix-τ texture per value) and `charge` (an
/// initial condition, not a rate) cannot be modulated at all, and are left out rather
/// than carried as knobs that would do nothing.
///
/// `None` on a target is not "a modulation with no effect" — it is skipped entirely,
/// so the parameter reaches the renderer as the exact float the slider holds.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct Modulations {
    /// Scales [`BrushParams::size`].
    pub size: Option<Modulation>,
    /// Scales [`BrushDynamics::flow`] — the brush's own paint, "Flow" in the UI.
    pub flow: Option<Modulation>,
    /// Scales [`BrushDynamics::lift`].
    pub lift: Option<Modulation>,
    /// Scales [`BrushDynamics::deposit`].
    pub deposit: Option<Modulation>,
    /// Scales [`BrushDynamics::bleed`].
    pub bleed: Option<Modulation>,
    /// Scales [`BrushParams::tooth_give`] — how far the tool settles into the canvas's
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

impl Modulations {
    /// The everyday brush: **size follows pressure**, linearly, all the way to
    /// nothing. Stated here rather than wired into the segment generator, which is
    /// what lets it be turned off or pointed somewhere else like any other mapping.
    ///
    /// Not [`Default`], which is "no mapping at all". Both are wanted:
    /// [`BrushParams::default`] takes this, while a preset that means to hold its
    /// width whatever the hand does asks for the empty set and gets it.
    pub const PRESSURE_SIZE: Self = Self {
        size: Some(Modulation::linear(ModSource::Pressure)),
        flow: None,
        lift: None,
        deposit: None,
        bleed: None,
        tooth_give: None,
        stretch: None,
    };

    /// [`Self::PRESSURE_SIZE`] as a function, for `#[serde(default = "…")]` — which
    /// takes a path to call and cannot name a constant.
    pub fn pressure_size() -> Self {
        Self::PRESSURE_SIZE
    }

    /// The multiplier for one target — 1 exactly where there is no modulation.
    fn factor(m: Option<Modulation>, pen: PenState) -> f32 {
        m.map_or(1.0, |m| m.factor(pen))
    }

    pub fn size(&self, pen: PenState) -> f32 {
        Self::factor(self.size, pen)
    }
    pub fn flow(&self, pen: PenState) -> f32 {
        Self::factor(self.flow, pen)
    }
    pub fn lift(&self, pen: PenState) -> f32 {
        Self::factor(self.lift, pen)
    }
    pub fn deposit(&self, pen: PenState) -> f32 {
        Self::factor(self.deposit, pen)
    }
    pub fn bleed(&self, pen: PenState) -> f32 {
        Self::factor(self.bleed, pen)
    }
    pub fn tooth_give(&self, pen: PenState) -> f32 {
        Self::factor(self.tooth_give, pen)
    }
    pub fn stretch(&self, pen: PenState) -> f32 {
        Self::factor(self.stretch, pen)
    }

    /// Every target at once, in the order they are declared above.
    ///
    /// Destructured rather than read field by field, and that is the whole reason
    /// it is written this way: a pattern with no `..` is exhaustive, so adding a
    /// target to the struct stops this compiling until it is listed here. Left as
    /// field accesses, a new target would simply be missing from
    /// [`max_slope`](Self::max_slope) — and an under-estimated slope is not an
    /// error anywhere, it is a modulated ramp quietly drawn as a staircase.
    fn all(&self) -> [Option<Modulation>; 7] {
        let Self {
            size,
            flow,
            lift,
            deposit,
            bleed,
            tooth_give,
            stretch,
        } = *self;
        [size, flow, lift, deposit, bleed, tooth_give, stretch]
    }

    /// Whether any target is mapped.
    pub fn is_active(&self) -> bool {
        self.all().iter().any(Option::is_some)
    }

    /// Every mapped target sanitized, the unmapped ones left unmapped.
    ///
    /// Destructured through [`all`](Self::all) and rebuilt positionally, so a
    /// target added to the struct arrives here already covered rather than being
    /// silently skipped — the same bargain `all` makes for
    /// [`max_slope`](Self::max_slope).
    pub fn sanitized(self) -> Self {
        let [size, flow, lift, deposit, bleed, tooth_give, stretch] =
            self.all().map(|m| m.map(Modulation::sanitized));
        Self {
            size,
            flow,
            lift,
            deposit,
            bleed,
            tooth_give,
            stretch,
        }
    }

    /// The steepest response across every active target — how much finer the path has
    /// to be flattened for a modulated ramp to stay smooth (see `MIN_BIAS` and
    /// `gpu::stroke::budget::flatten_tolerance`).
    ///
    /// 1 for the unmodulated brush *and* for the plain linear mappings, so the
    /// everyday brush flattens on exactly the budget it always did.
    pub fn max_slope(&self) -> f32 {
        self.all()
            .iter()
            .flatten()
            .map(Modulation::max_slope)
            .fold(1.0, f32::max)
    }
}

/// Brush configuration. `color` is straight **sRGB** RGBA; it is converted to
/// the Oklab working space at stamp time (§6.5).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, carbonite::Schema)]
pub struct BrushParams {
    /// Straight (un-premultiplied) sRGB RGBA, components in [0, 1].
    pub color: [f32; 4],
    /// Stamp radius in canvas pixels at full pressure.
    pub size: f32,
    /// Reservoir depletion per **radius** travelled: the stroke thins as paint runs
    /// out (§6.2). 0 = inexhaustible — which is what a pen, a charcoal
    /// stick, or an ordinary digital brush wants; a physical loaded brush wants a
    /// small positive value, and 1 is a tool bone dry one radius past the press.
    ///
    /// In radii rather than canvas px for the reason the tapers are
    /// ([`start_taper_length`](Self::start_taper_length)), and it is the stronger
    /// case of the two: [`radius`](Self::radius) is meant to be a pure *scale* on
    /// the mark, and a falloff quoted in canvas px is exactly what that scale does
    /// not carry — enlarge such a brush and it runs dry a fraction of the way into
    /// its own tip, which is not a bigger version of anything.
    /// [`drain_px`](Self::drain_px) is where it becomes the per-px rate both render
    /// paths read.
    pub drain: f32,
    /// Brush tip shape (§6.6).
    pub shape: BrushShape,
    /// What orients the shape as it sweeps (§6.6) — the successor to the old
    /// `follow_path`/`angle_jitter` knobs: `FollowStroke` is the former `follow_path =
    /// true`. `#[serde(default)]` so documents saved before this field (which instead
    /// carried `follow_path`, now ignored on load) come in as `FollowStroke`.
    #[serde(default)]
    pub orientation: OrientationSource,
    /// How much of its own paint the brush lays, and how it manipulates paint already
    /// on the canvas (§6.2) — the unified tool. `#[serde(default)]` so
    /// documents saved before this field load as the everyday `add`-only brush.
    #[serde(default)]
    pub dynamics: BrushDynamics,
    /// Color dynamics (color jitter) — how the applied color varies across the
    /// brush and along the stroke (§6.2). Historized (it changes stored
    /// pixels); the default (amplitude 0) is the historical constant color.
    #[serde(default)]
    pub color_dynamics: ColorDynamics,
    /// Length of the stroke's **leading taper** — the run over which the tip widens
    /// from a point to its full [`radius`](Self::radius) — in *units of `radius`*,
    /// so 4.0 means four brush radii of taper (§6.2). 0 = no taper: the
    /// stroke starts at full width, which is the historical behaviour.
    ///
    /// In radii rather than canvas px so a brush keeps its *look* as it is resized:
    /// scale the tip up and the whole mark scales with it, instead of a taper that
    /// was the shape of the stroke at radius 8 becoming a blunt nub at radius 80.
    /// This is what lets a brush mimic an inker's entry stroke.
    #[serde(default)]
    pub start_taper_length: f32,
    /// Length of the stroke's **trailing taper**, in units of
    /// [`radius`](Self::radius) — [`start_taper_length`](Self::start_taper_length)
    /// measured back from the end of the stroke, for the exit of an inked line.
    ///
    /// Together the two are held to the stroke's own length: if they would overlap
    /// they are scaled down in proportion, so a short flick is a small pointed mark
    /// rather than a sliver (see `gpu::stroke::segments::Taper`).
    #[serde(default)]
    pub end_taper_length: f32,
    /// What the pen drives, and how (§6.2) — the mapping from pen input to brush
    /// parameter. [`Modulations::PRESSURE_SIZE`] by default: the pressure → radius
    /// scaling, held here as data so a preset can drop it or aim it elsewhere.
    #[serde(default)]
    pub modulation: Modulations,
    /// How much **give** this tool has against the canvas substrate's tooth (§6.4), in
    /// [0, 1]: 1 = infinite give — the tip follows every fall, the substrate does not
    /// break the mark up at all, and this is the historical behaviour and the default;
    /// 0 = no give at all — it rides the very tops of the grain, so the mark is what a
    /// dry brush leaves.
    ///
    /// **The give, not its inverse, and that is what makes the pen mapping mean
    /// something.** A [`Modulation`] can only ever scale a parameter *down*, so a knob
    /// quoted as the depth of the bite would have made light pressure the solid mark
    /// and a hard press the dry one — backwards for the one mapping this axis exists
    /// for. Quoted as the give, pressure reads the way a hand expects:
    /// [`Modulations::tooth_give`] mapped to pressure is the charcoal, barely touching
    /// the paper it prints the peaks alone, and borne down it presses past the falls
    /// it was bridging and the grain fills in.
    ///
    /// It does cost the slider its usual direction — the *interesting* end of this one
    /// is the left. That is the trade, and the mapping is worth more than the habit.
    ///
    /// The *substrate* is document state ([`SubstrateId`](crate::SubstrateId)) — a pencil and a loaded brush
    /// on the same canvas see the same tooth, which is why the grain lives there and
    /// only this knob lives on the brush. What it scales is the paint the brush lays
    /// per unit swept optical depth, gated per texel by whether the substrate clears the
    /// level this tool settles to (`paint_common.wesl::tooth_gate`).
    ///
    /// **How far the tip settles, not how sharply it stops.** How abruptly a texel
    /// goes from bridged to pressed on is the other half of contact, and it is
    /// [`tooth_softness`](Self::tooth_softness).
    ///
    /// Inert on a `Flat` canvas whatever this says, because `Surface::relief` is
    /// 0 there — so the axis is orthogonal to every golden that paints on `Flat`, the
    /// same way the media pass's substrate already is.
    ///
    /// `#[serde(default = "…")]` and not a bare `#[serde(default)]`: 0 is *maximum*
    /// tooth at this end of the knob, so a file that does not mention the field has
    /// to be told the full give it meant rather than handed a zero (§8).
    #[serde(default = "BrushParams::default_tooth_give")]
    pub tooth_give: f32,
    /// The **width of the contact transition**, in the rise's own units — height per
    /// reach of travel (§6.4). The band of rise, either side of the follow limit
    /// [`tooth_give`](Self::tooth_give) sets, over which a texel goes from taking
    /// none of the tip's paint to taking all of it.
    ///
    /// A hard threshold — 0 — is a binary indicator per texel: correct in the mean,
    /// and at canvas resolution it aliases into hard-edged speckle that reads as
    /// dither rather than as tooth. Too wide and the faces are smeared into a flat
    /// grey and the grain stops reading at all.
    /// [`DEFAULT_TOOTH_SOFTNESS`](Self::DEFAULT_TOOTH_SOFTNESS) is the bundled
    /// substrates' own interquartile rise, so the transition spans the grain's natural
    /// variation — which is what a paint that *sits on* the substrate wants, and it is
    /// where this number lived when it was a constant in `paint_common.wesl`. A
    /// charcoal or a soft graphite does not sit on the substrate: the stick crumbles
    /// into the valleys instead of spanning them, so its contact comes on gradually
    /// over several times that band, and a knob is the only way to say so.
    ///
    /// **A width, so it has no ceiling this crate owns** — the same reading as the
    /// flow and the drain (see [`sanitized`](Self::sanitized)). Past about twice the
    /// encodable rise range the whole distribution is inside the band and the gate is
    /// a flat scale factor; that is where the frontend's slider stops, not where the
    /// quantity stops meaning something.
    ///
    /// `#[serde(default = "…")]` rather than a bare `#[serde(default)]`, because a
    /// file saved before this field meant the constant the shader used to carry —
    /// and 0, which is what a plain default hands back, is the hard threshold and
    /// not that (§8).
    #[serde(default = "BrushParams::default_tooth_softness")]
    pub tooth_softness: f32,
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
}

impl Default for BrushParams {
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0, 1.0],
            size: 16.0,
            drain: 0.0,
            shape: BrushShape::default(),
            orientation: OrientationSource::default(),
            dynamics: BrushDynamics::default(),
            color_dynamics: ColorDynamics::default(),
            start_taper_length: 0.0,
            end_taper_length: 0.0,
            modulation: Modulations::PRESSURE_SIZE,
            tooth_give: Self::DEFAULT_TOOTH_GIVE,
            tooth_softness: Self::DEFAULT_TOOTH_SOFTNESS,
            stretch: 0.0,
        }
    }
}

impl BrushParams {
    /// The give a brush has against the substrate when it does not say
    /// ([`tooth_give`](Self::tooth_give)): all of it, so the tip follows every fall and
    /// the substrate breaks nothing up — the mark a brush made before the tooth
    /// existed, to the bit.
    ///
    /// Named rather than spelled `1.0` at the four places that need it, because *which*
    /// end of this knob is the inert one is the fact worth being able to look up.
    pub const DEFAULT_TOOTH_GIVE: f32 = 1.0;

    /// [`DEFAULT_TOOTH_GIVE`](Self::DEFAULT_TOOTH_GIVE) as a function, for
    /// `#[serde(default = "…")]`.
    fn default_tooth_give() -> f32 {
        Self::DEFAULT_TOOTH_GIVE
    }

    /// The contact transition a brush gets when it does not say
    /// ([`tooth_softness`](Self::tooth_softness)) — the bundled substrates' own
    /// interquartile rise, so the band spans the grain's natural variation.
    ///
    /// **The one place this number is written.** It was
    /// `paint_common.wesl`'s `const TOOTH_SOFTNESS` until the knob existed; the shader
    /// takes it as a uniform now and declares nothing, which is what keeps the default
    /// from being a host transcription of a shader constant (§6.10).
    pub const DEFAULT_TOOTH_SOFTNESS: f32 = 0.5;

    /// [`DEFAULT_TOOTH_SOFTNESS`](Self::DEFAULT_TOOTH_SOFTNESS) as a function, for
    /// `#[serde(default = "…")]` — which takes a path to call and cannot name a
    /// constant.
    fn default_tooth_softness() -> f32 {
        Self::DEFAULT_TOOTH_SOFTNESS
    }

    /// The two taper lengths in **canvas px**: the stored lengths (in radii) scaled
    /// by [`radius`](Self::radius). Negative or non-finite lengths read as 0 — the
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
    /// [`drain`](Self::drain)) over [`radius`](Self::radius) — [`taper_px`](Self::taper_px)
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
    /// Here rather than in the frontend, where it was spelled inline as the
    /// stretch slider's end, because a slider is only one of the ways a value
    /// reaches this field — a file and a peer are two more, and neither passes
    /// through a panel. `MAX_FLOW`'s own doc already makes this argument for the
    /// *drag* bindings ("a knob reachable two ways must have one range"); the
    /// wire is simply the third way.
    pub const MAX_STRETCH: f32 = 1.0 - 1.0 / Self::MAX_ELONGATION;

    /// The same brush with every number a number, and every number that has a
    /// documented range inside it — the funnel a brush passes through on its way
    /// into the document, exactly as [`Filter::sanitized`](super::Filter::sanitized)
    /// is for a filter (§21.5) and for the same two reasons.
    ///
    /// **It clamps only where this crate already states a range.** The three pickup
    /// axes, the tooth's *give*, the hardness and the color are quoted in `[0, 1]` by
    /// their own field docs; the stretch saturates at [`MAX_STRETCH`](Self::MAX_STRETCH)
    /// by construction. Everything else — the radius, the flow, the drain, the
    /// charge, the tapers, the jitter, the tooth's *softness* — is required to be a finite, non-negative
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
            color: self.color.map(clamp01),
            size: at_least_zero(self.size, d.size),
            drain: at_least_zero(self.drain, d.drain),
            shape: self.shape.sanitized(),
            orientation: self.orientation,
            dynamics: self.dynamics.sanitized(),
            color_dynamics: self.color_dynamics.sanitized(),
            start_taper_length: at_least_zero(self.start_taper_length, d.start_taper_length),
            end_taper_length: at_least_zero(self.end_taper_length, d.end_taper_length),
            modulation: self.modulation.sanitized(),
            tooth_give: clamp01(finite_or(self.tooth_give, d.tooth_give)),
            // Floored, never capped, for `add`'s reason: a transition width is a
            // length in the rise's units rather than a fraction, and the only ceiling
            // it has is a slider's.
            tooth_softness: at_least_zero(self.tooth_softness, d.tooth_softness),
            // Bounded at the knob's own saturation point rather than at 1: past
            // `MAX_STRETCH` the reciprocal is already pinned, so a larger value
            // stored is a number that cannot mean what it says.
            stretch: finite_or(self.stretch, d.stretch).clamp(0.0, Self::MAX_STRETCH),
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
        let none = Modulations::default();
        assert!(!none.is_active());
        for x in [0.0, 0.25, 1.0] {
            assert_eq!(none.flow(pen(x)), 1.0);
            assert_eq!(none.lift(pen(x)), 1.0);
        }
        // …and the everyday brush maps size alone.
        let m = Modulations::PRESSURE_SIZE;
        assert_eq!(m.size(pen(0.4)).to_bits(), 0.4f32.to_bits());
        assert_eq!(m.flow(pen(0.4)), 1.0);
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
        assert_eq!(Modulations::default().max_slope(), 1.0);
        assert_eq!(Modulations::PRESSURE_SIZE.max_slope(), 1.0);
        // A steep one does pay, and the bill is bounded (`MIN_BIAS`).
        let steep = Modulations {
            flow: Some(Modulation {
                source: ModSource::Tilt,
                floor: 0.0,
                curve: -1.0,
            }),
            ..Modulations::default()
        };
        let slope = steep.max_slope();
        assert!(
            (1.0..=9.0 + 1e-4).contains(&slope),
            "an extreme curve should cost something, and a bounded something: {slope}"
        );
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
        type Poke = (&'static str, fn(&mut BrushParams, f32));
        let pokes: [Poke; 17] = [
            ("radius", |b, f| b.size = f),
            ("drain", |b, f| b.drain = f),
            ("tooth_give", |b, f| b.tooth_give = f),
            ("tooth_softness", |b, f| b.tooth_softness = f),
            ("stretch", |b, f| b.stretch = f),
            ("start_taper", |b, f| b.start_taper_length = f),
            ("end_taper", |b, f| b.end_taper_length = f),
            ("color.r", |b, f| b.color[0] = f),
            ("color.a", |b, f| b.color[3] = f),
            ("dynamics.add", |b, f| b.dynamics.flow = f),
            ("dynamics.lift", |b, f| b.dynamics.lift = f),
            ("dynamics.deposit", |b, f| b.dynamics.deposit = f),
            ("dynamics.charge", |b, f| b.dynamics.charge = f),
            ("dynamics.bleed", |b, f| b.dynamics.bleed = f),
            ("jitter.amplitude", |b, f| b.color_dynamics.amplitude[1] = f),
            ("jitter.frequency", |b, f| b.color_dynamics.frequency[0] = f),
            ("hardness", |b, f| {
                b.shape = BrushShape::Round { hardness: f }
            }),
        ];
        let unit = |b: &BrushParams| {
            [
                b.dynamics.lift,
                b.dynamics.deposit,
                b.dynamics.bleed,
                b.tooth_give,
                b.color[0],
                b.color[3],
            ]
        };
        for (name, poke) in pokes {
            for f in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -9.0, 1e30] {
                let mut brush = BrushParams::default();
                poke(&mut brush, f);
                let clean = brush.sanitized();
                // Every number is a number…
                for v in [
                    clean.size,
                    clean.drain,
                    clean.tooth_give,
                    clean.tooth_softness,
                    clean.stretch,
                    clean.start_taper_length,
                    clean.end_taper_length,
                    clean.dynamics.flow,
                    clean.dynamics.lift,
                    clean.dynamics.deposit,
                    clean.dynamics.charge,
                    clean.dynamics.bleed,
                ] {
                    assert!(v.is_finite(), "{name} = {f} left a non-finite brush");
                    assert!(v >= 0.0, "{name} = {f} left {v}, which is negative");
                }
                // …the ones this crate quotes in [0, 1] are in it…
                for v in unit(&clean) {
                    assert!((0.0..=1.0).contains(&v), "{name} = {f} escaped [0, 1]");
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
            tooth_give: 0.25,
            // Past the default band, and legitimately so: a soft stick spans the
            // grain rather than sitting on it, and the ceiling is a slider's.
            tooth_softness: 0.3,
            dynamics: BrushDynamics {
                flow: 2.5, // past the frontend's slider, and legitimately so
                lift: 1.0,
                bleed: 0.95,
                ..BrushDynamics::default()
            },
            ..BrushParams::default()
        };
        assert_eq!(ordinary.sanitized(), ordinary);
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
}
