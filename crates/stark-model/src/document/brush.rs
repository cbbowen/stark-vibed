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

/// The brush tip shape (§6.6).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
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
}

/// What sets the brush shape's orientation as it sweeps along the stroke
/// (§6.6). The swept-depth integral runs along the stroke's travel direction, so the
/// shape is looked up in a per-orientation prefix-τ texture indexed by the *relative*
/// angle between the shape's native axis and the travel direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OrientationSource {
    /// The shape's native axis tracks the stroke tangent — the relative angle is always
    /// 0, so the footprint always faces along the motion (the historical behaviour).
    #[default]
    FollowStroke,
    /// The shape stays pinned to the pen's orientation (the tilt azimuth) in canvas
    /// space; as the stroke curves under a fixed pen the footprint angle stays put,
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
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
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
            add: 0.6,
            lift: 0.0,
            deposit: 0.0,
            charge: 0.0,
            bleed: 0.0,
        }
    }
}

/// The kind of noise field driving [`ColorDynamics`] (§6.2). Each kind
/// is baked once into a small tileable 2-D texture (`noise.rs`), so lookups are
/// cheap and deterministic across replay, peers, and builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NoiseKind {
    /// Uncorrelated per-texel randomness — grainy speckle.
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
/// color across the footprint, the other evolves it along the stroke. The three
/// noise channels offset the three color channels *of the current color space*
/// (Oklab `L, a, b`; Mixbox pigment concentrations). The per-stroke `seed`
/// translates the lookup so each stroke draws a fresh part of the field,
/// deterministically.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
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
}

/// What a [`Modulation`] reads off the pen (§6.2).
///
/// Both are already carried per point of the fitted curve
/// ([`ControlPoint`](crate::path::ControlPoint)) and interpolated per swept segment,
/// so a source here costs the renderer nothing to evaluate and nothing to store.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
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
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
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

/// `x` into [0, 1], with NaN landing on 0 — these arrive from files, presets and
/// peers, and a NaN factor would propagate straight into a radius.
///
/// `max`-then-`min` rather than `clamp`, which is what makes that true:
/// `f32::max`/`min` return the non-NaN operand where `clamp` returns the NaN. Same
/// argument as [`BrushParams::taper_px`], and the reason clippy's suggestion here is
/// the wrong one.
#[allow(clippy::manual_clamp)]
fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// Which brush parameters the pen drives, and how (§6.2) — the mapping from pen
/// input to brush parameter that makes one tool a brush and another a palette knife.
///
/// Exactly the parameters that already vary **per swept segment**, and no others. A
/// segment carries one radius, one set of paint rates and one orientation (§6.6), so
/// these are the quantities a modulation can reach without changing what a segment
/// *is*; `hardness` (baked into the prefix-τ texture per value) and `charge` (an
/// initial condition, not a rate) cannot be modulated at all, and are left out rather
/// than carried as knobs that would do nothing.
///
/// `None` on a target is not "a modulation with no effect" — it is skipped entirely,
/// so the parameter reaches the renderer as the exact float the slider holds.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Modulations {
    /// Scales [`BrushParams::radius`].
    pub size: Option<Modulation>,
    /// Scales [`BrushDynamics::add`] — the brush's own paint, "Flow" in the UI.
    pub flow: Option<Modulation>,
    /// Scales [`BrushDynamics::lift`].
    pub lift: Option<Modulation>,
    /// Scales [`BrushDynamics::deposit`].
    pub deposit: Option<Modulation>,
    /// Scales [`BrushDynamics::bleed`].
    pub bleed: Option<Modulation>,
    /// Scales [`BrushParams::tooth`] — how deep the tool bites into the canvas's
    /// weave (§6.4). Mapped to pressure this is the charcoal behaviour: bear down and
    /// the tip flattens into the valleys, so the grain fills in.
    pub tooth: Option<Modulation>,
    /// Scales [`BrushParams::stretch`] — how far the footprint elongates along the
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
        tooth: None,
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
    pub fn tooth(&self, pen: PenState) -> f32 {
        Self::factor(self.tooth, pen)
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
            tooth,
            stretch,
        } = *self;
        [size, flow, lift, deposit, bleed, tooth, stretch]
    }

    /// Whether any target is mapped.
    pub fn is_active(&self) -> bool {
        self.all().iter().any(Option::is_some)
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
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushParams {
    /// Straight (un-premultiplied) sRGB RGBA, components in [0, 1].
    pub color: [f32; 4],
    /// Stamp radius in canvas pixels at full pressure.
    pub radius: f32,
    /// Reservoir depletion per canvas pixel travelled: the stroke thins as paint
    /// runs out (§6.2). 0 = inexhaustible — which is what a pen, a
    /// charcoal stick, or an ordinary digital brush wants; a physical loaded
    /// brush wants a small positive value.
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
    #[serde(default = "Modulations::pressure_size")]
    pub modulation: Modulations,
    /// How deeply this tool bites into the **canvas surface's tooth** (§6.4), in
    /// [0, 1]: 0 = the tip reaches everywhere and the ground does not break the mark
    /// up at all (the historical behaviour, and the default); 1 = it touches only the
    /// very tops of the weave, so the mark is what a dry brush leaves.
    ///
    /// The *ground* is document state ([`SurfaceId`](crate::SurfaceId)) — a pencil and a loaded brush
    /// on the same canvas see the same tooth, which is why the grain lives there and
    /// only this knob lives on the brush. What it scales is the paint the brush lays
    /// per unit swept optical depth, gated per texel by whether the ground clears the
    /// level this tool presses to (`paint_common.wesl::tooth_gate`).
    ///
    /// Exactly 0 on a `Flat` canvas whatever this says, because `Surface::relief` is
    /// 0 there — so the axis is orthogonal to every golden that paints on `Flat`, the
    /// same way the media pass's weave already is.
    #[serde(default)]
    pub tooth: f32,
    /// How far the footprint **elongates along the brush's facing axis** (§6.6), in
    /// `[0, 1)`: the tip is stretched by [`elongation`](Self::elongation)
    /// `s = 1/(1 − stretch)` along that axis and left alone across it, so 0 is the
    /// footprint the shape draws and 0.5 is one twice as long as it is wide.
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
    /// footprint is the unstretched one read at another angle, over another travel,
    /// with a factor on the result (§6.6), so the prefix-τ volume the brush already
    /// binds is the volume this reads. What it does cost is footprint area — the tip
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
            radius: 16.0,
            drain: 0.0,
            shape: BrushShape::default(),
            orientation: OrientationSource::default(),
            dynamics: BrushDynamics::default(),
            color_dynamics: ColorDynamics::default(),
            start_taper_length: 0.0,
            end_taper_length: 0.0,
            modulation: Modulations::PRESSURE_SIZE,
            tooth: 0.0,
            stretch: 0.0,
        }
    }
}

impl BrushParams {
    /// The two taper lengths in **canvas px**: the stored lengths (in radii) scaled
    /// by [`radius`](Self::radius). Negative or non-finite lengths read as 0 — the
    /// fields arrive from files, presets and peers, and a taper is a length.
    pub fn taper_px(&self) -> (f32, f32) {
        // `f32::max` returns the non-NaN operand, so this also normalizes NaN to 0.
        let r = self.radius.max(0.0);
        let px = |len: f32| {
            let px = len.max(0.0) * r;
            if px.is_finite() { px } else { 0.0 }
        };
        (px(self.start_taper_length), px(self.end_taper_length))
    }

    /// Whether either end of a stroke with this brush tapers.
    pub fn tapers(&self) -> bool {
        let (start, end) = self.taper_px();
        start > 0.0 || end > 0.0
    }

    /// The furthest the footprint may be drawn out along its facing axis — what
    /// [`elongation`](Self::elongation) saturates at, and so the factor by which the
    /// worst-case tip outgrows its own radius.
    ///
    /// A bound on *area*, which is why there is one at all: every tile the stretched
    /// tip reaches is a tile the stroke is rasterized into and the dynamics loop
    /// dispatches over, so `s` prices the stroke roughly linearly. Eight is already a
    /// pen laid almost flat; past it the mark stops reading as a wider stroke and
    /// starts reading as a smear the length of the tip.
    pub const MAX_ELONGATION: f32 = 8.0;

    /// [`stretch`](Self::stretch) as the factor the footprint is drawn out by along
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
    #[allow(clippy::manual_clamp)]
    pub fn elongation(stretch: f32) -> f32 {
        // Bounded before the divide rather than clamped after it, so a knob past 1 —
        // or a negative one, which is not a squash but no stretch at all — lands on a
        // real factor instead of on an infinity or an inside-out tip.
        //
        // `min` first and `max` second, and that order is the NaN policy: `f32::min`
        // and `f32::max` return the non-NaN operand (the argument at
        // `clamp01`), so this way a NaN knob falls out as the *identity* and the
        // other way it would fall out as the widest footprint the brush can ask for.
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
