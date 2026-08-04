//! Actions: committed, deterministic, replayable document mutations (§4).
//!
//! An [`Action`] is the unit the timeline stores/replays and (later) the unit
//! serialized to disk. Every action carries a globally-unique [`ActionId`] so
//! the same records work unchanged in a future replicated, multi-peer log
//! (§4, §12) — we pay that tiny cost from the first commit.

use serde::{Deserialize, Serialize};

use super::layer::{BlendMode, LayerId, MatteRegion};
use super::selection::SelectionOp;
use super::state::DocState;
use crate::geom::Vec2;
use crate::gpu::SurfaceId;
use crate::gpu::selection::SelectionRenderer;
use crate::gpu::stroke::StrokeRenderer;
use crate::gpu::tile::TilePool;

/// Identifies the author of an action: one local user, or a peer (§4).
/// Maps to an iroh `NodeId` when collaborating; a fixed value when solo.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActorId(pub u64);

impl ActorId {
    /// The fixed author id used when not collaborating. When a document is
    /// first shared, its solo-authored actions are rewritten to the sharer's
    /// real actor id (so the sharer can still undo them, §12.3);
    /// after that every action in a shared log carries a peer-derived id.
    pub const SOLO: ActorId = ActorId(0);
}

/// Globally-unique action id; also the total order key `(lamport, actor)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActionId {
    pub lamport: u64,
    pub actor: ActorId,
}

/// The tool a gesture drives. Tools become an open registry later (§10).
///
/// Only [`Brush`](Self::Brush) ever reaches a [`StrokeRecord`]: the selection tools
/// produce a [`SelectionOp`] instead of a stroke (§6.8). They share the
/// enum — and so the pointer-gesture plumbing — because from the frontend's point of
/// view they are the same interaction: press, drag, release.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tool {
    #[default]
    Brush,
    /// Rectangular marquee.
    SelectRect,
    /// Elliptical marquee.
    SelectEllipse,
    /// Freehand lasso.
    SelectLasso,
}

impl Tool {
    /// Whether this tool edits the selection rather than the paint.
    pub fn is_selection(self) -> bool {
        matches!(
            self,
            Tool::SelectRect | Tool::SelectEllipse | Tool::SelectLasso
        )
    }
}

/// The brush tip shape (§6.6).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BrushShape {
    /// Procedural soft disc. Edge softness in [0, 1): 0 = very soft, ~1 = hard
    /// edge — meaningful only here, since it is what shapes this tip's falloff.
    Round { hardness: f32 },
    /// A sampled coverage mask, referenced by content id (an imported image).
    Stamp(crate::assets::AssetId),
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
    /// sequential stamp loop (§6.2), and the loop used to scale this by a gain
    /// of 2 — so nudging [`deposit`](Self::deposit) off zero doubled the flow of a
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
    /// Canvas paint **diffusing under the tip** per step, as a fraction of full smoothing,
    /// in [0, 1]: 0 = the paint under the stroke holds still, 1 = it relaxes towards its
    /// neighbourhood as fast as the loop can carry. The one **lateral** flux, and it is
    /// internal to the canvas — the tool neither takes nor gives (§6.2).
    ///
    /// The neighbourhood is **a fixed fraction of the tip** (`BLEED_REACH`,
    /// dynamics.wesl), not a pixel count, so the axis is resolution-independent: at
    /// full crank a pass of the tip blurs by roughly a third of the radius whatever
    /// the canvas resolution — the same property the tapers get from being quoted
    /// in radii.
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
    /// The discrete form of [`Self::Voronoi`]: each cell one flat colour offset,
    /// with a hard edge to its neighbours — crystalline facets rather than a
    /// gradient. All three channels share the same cells, so the facets are
    /// whole polygons of one colour.
    Mosaic,
}

/// Colour dynamics (colour jitter): lets the applied colour vary **across the
/// brush and along the stroke** (§6.2). A 3-channel tileable 2-D noise
/// field is sampled in the stroke's **own** frame — `(lateral offset from the
/// centreline, arc length)`, both in canvas px — so the variation belongs to the
/// gesture rather than to the patch of canvas under it: one axis spreads the
/// colour across the footprint, the other evolves it along the stroke. The three
/// noise channels offset the three colour channels *of the current colour space*
/// (Oklab `L, a, b`; Mixbox pigment concentrations). The per-stroke `seed`
/// translates the lookup so each stroke draws a fresh part of the field,
/// deterministically.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorDynamics {
    /// Which noise field to sample.
    pub noise: NoiseKind,
    /// Frequency scale per lookup axis (across the stroke, along it): 1 = one
    /// noise tile per [`crate::noise::NOISE_TILE_PX`] px; higher = finer
    /// variation along that axis; 0 = constant along that axis.
    pub frequency: [f32; 2],
    /// Noise amplitude per colour channel, in the colour space's own units
    /// (noise is signed, so a channel wanders ±amplitude). All 0 = off — the
    /// exact historical constant-colour deposit.
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
/// rule's radius bound ([`safe_frozen`](crate::gpu::stroke::safe_frozen)), the
/// region fit, the choice of render path
/// ([`dynamics_setup`](crate::gpu::stroke)), the flattener's exchange step — is
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
/// a steep response is paid for in segments — [`flatten_tolerance`] divides the
/// attribute budget by [`Modulations::max_slope`] to keep a modulated ramp from
/// drawing as a staircase, exactly as the taper's own slope buys pieces
/// (`gpu::stroke::segments::Taper`). Unbounded steepness would be an unbounded bill.
///
/// [`flatten_tolerance`]: crate::gpu::stroke::flatten_tolerance
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

    /// A bound on `|d factor / d input|`, for the flattener (see [`MIN_BIAS`]).
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
}

impl Modulations {
    /// The everyday brush: **size follows pressure**, linearly, all the way to
    /// nothing — the one mapping that used to be wired into the segment generator,
    /// now stated where every other one is.
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

    /// Every target at once, in the order they are declared above.
    fn all(&self) -> [Option<Modulation>; 6] {
        [
            self.size,
            self.flow,
            self.lift,
            self.deposit,
            self.bleed,
            self.tooth,
        ]
    }

    /// Whether any target is mapped.
    pub fn is_active(&self) -> bool {
        self.all().iter().any(Option::is_some)
    }

    /// The steepest response across every active target — how much finer the path has
    /// to be flattened for a modulated ramp to stay smooth (see [`MIN_BIAS`] and
    /// [`flatten_tolerance`](crate::gpu::stroke::flatten_tolerance)).
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
    /// Colour dynamics (colour jitter) — how the applied colour varies across the
    /// brush and along the stroke (§6.2). Historized (it changes stored
    /// pixels); the default (amplitude 0) is the historical constant colour.
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
    /// parameter. [`Modulations::PRESSURE_SIZE`] by default, which is the pressure →
    /// radius scaling that was previously wired into the segment generator with no
    /// way to turn it off or point it anywhere else.
    #[serde(default = "Modulations::pressure_size")]
    pub modulation: Modulations,
    /// How deeply this tool bites into the **canvas surface's tooth** (§6.4), in
    /// [0, 1]: 0 = the tip reaches everywhere and the ground does not break the mark
    /// up at all (the historical behaviour, and the default); 1 = it touches only the
    /// very tops of the weave, so the mark is what a dry brush leaves.
    ///
    /// The *ground* is document state ([`SurfaceId`]) — a pencil and a loaded brush
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
}

/// A fully-recorded stroke: enough to replay it bit-for-bit (§4).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrokeRecord {
    pub layer: LayerId,
    pub tool: Tool,
    pub brush: BrushParams,
    /// The fitted stroke curve: the control points the raw pointer samples were
    /// smoothed and simplified down to (§6.2), an order of magnitude
    /// fewer points and all that is needed to reconstruct the stroke. The raw
    /// samples are never stored — not in the file, not in the action log, not
    /// on the wire.
    pub path: Vec<crate::path::ControlPoint>,
    /// Seed for any brush jitter, making replay reproducible. Unused by the MVP
    /// brush but recorded so the format is stable.
    pub seed: u64,
}

/// What an action does to the document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActionKind {
    CommitStroke(StrokeRecord),
    AddLayer {
        id: LayerId,
        /// Whose carried stack to add it to — `None` for the document's own
        /// (§14.8).
        carrier: Option<LayerId>,
        above: Option<LayerId>,
    },
    /// Remove a layer **and everything it carries**: the subtree is the group
    /// (§14.2). Promoting what it carried instead is a
    /// [`MoveLayer`](Self::MoveLayer), which is what "release" is spelled with.
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    /// Move a layer — with everything it carries — into the stack carried by
    /// `carrier` (the document's own when `None`), directly above `above`.
    ///
    /// The **only** structural move, covering all three gestures at once
    /// (§14.8): reorder is `carrier` unchanged, *carry* is `carrier`
    /// set, *release* is `carrier` cleared. Declined deterministically when it
    /// would make a layer carry its own ancestor — see
    /// [`DocState::move_layer`](super::state::DocState::move_layer) for why the
    /// log's total order is all the cycle protection this needs.
    MoveLayer {
        id: LayerId,
        carrier: Option<LayerId>,
        above: Option<LayerId>,
    },
    /// Undo **as a logged action** (§5.4, §12.3): a fact peers can see
    /// and order, meaning "derive the document as if `target` were absent".
    /// Redo is an `Undo` of an `Undo`. Emitted only in shared sessions; solo
    /// undo stays pure timeline navigation and never logs one.
    ///
    /// Deliberately **not interpreted by [`Action`]'s `apply`** — undo needs the
    /// whole log, not just the prior state, so the timeline layer resolves
    /// which actions are *effective* (see [`super::timeline::effective_actions`])
    /// and only ever materializes those. Appended last so postcard decoding of
    /// older files is unaffected.
    Undo(ActionId),

    /// Switch the canvas surface (§6.4).
    ///
    /// Logged rather than kept as a view setting because the surface feeds the
    /// document: which canvas a piece was painted on is part of what it is, and
    /// replay has to reconstruct it. Appended last so
    /// postcard decoding of older files is unaffected; documents saved before this
    /// existed simply never contain one and keep the surface from `CanvasMeta`.
    SetSurface(SurfaceId),
    /// Edit the selection mask (§6.8). Historized because a stroke's
    /// pixels depend on the mask in force when it was drawn — replaying the log has
    /// to put the same mask back. Only the **op** travels (a few floats, or a
    /// decimated polyline); every peer rasterizes it identically from the same
    /// shader, so the log stays compact and convergence is unaffected.
    Select(SelectionOp),
    /// Swap selected for unselected everywhere (§6.8).
    InvertSelection,

    /// Add a **matte** layer — a region filled with a flat colour
    /// (§15.2). A frame is one of these on top of the stack; the
    /// same action serves comic gutters and opaque grounds once the region
    /// generalizes (P4). Appended last, like every variant before it, so postcard
    /// — which encodes an enum by variant *index* — keeps decoding older files.
    AddMatte {
        id: LayerId,
        carrier: Option<LayerId>,
        above: Option<LayerId>,
        region: MatteRegion,
        /// Straight sRGB, like `BrushParams::color` — converted to working-space
        /// channels at composite time, so the log is colour-space independent.
        color: [f32; 3],
    },
    /// Move a matte's rect — the frame drag's commit. One action per drag, not
    /// per pointer move: the gesture accumulates in session state and commits on
    /// release, so fifty tweaks are fifty undo steps rather than five thousand.
    SetMatteRect(LayerId, Vec2, Vec2),
    /// Recolour a matte (straight sRGB).
    SetMatteColor(LayerId, [f32; 3]),
    /// Set the canvas substrate colour — the ground the paint sits on, straight
    /// sRGB (§15.5). Logged because the ground a piece was painted on
    /// is part of what it is; it was previously a view setting, so the paper colour
    /// of a painting was not saved at all.
    SetBackground([f32; 3]),

    /// Affine transform of the selected paint on `layer` (§16):
    /// cut what the **author's** selection holds, resample it once under
    /// `affine`, stack it back over what remained — and carry the author's mask
    /// along with it, so the moved region stays selected. A universal selection
    /// moves the whole layer. Six floats in the log; every peer re-derives the
    /// same tiles from them. Appended last so postcard keeps decoding older
    /// files.
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the
    /// affine is unusable or the rewrite exceeds the tile caps — see
    /// [`super::transform`].
    Transform {
        layer: LayerId,
        affine: crate::geom::Affine2,
    },

    /// Name a layer, or with `None` take its name away again so it falls back to
    /// being described by its place in the stack.
    ///
    /// Logged like every other layer property: a name is part of the document —
    /// it is saved, it is replicated, and taking one back is an undo step, which
    /// is what makes a mistyped rename recoverable the same way a mis-set opacity
    /// is. Carries a `String` rather than the `Arc<str>` the state holds, because
    /// this is the file and wire form, where a shared pointer means nothing.
    /// Appended last so postcard — which encodes an enum by variant *index* —
    /// keeps decoding older files.
    SetLayerName(LayerId, Option<String>),

    /// Fill a region of `layer` with paint (§18.0.4).
    ///
    /// The fifth thing a shape gesture can do, alongside the four ways it can
    /// combine into the selection — see [`ShapeAction`](super::fill::ShapeAction).
    /// Gated and keyed exactly as a stroke is: the **author's** mask, taken off the
    /// state being folded over, bounds the fill, and the actor comes from the
    /// action's own id. A matte or absent layer refuses it, like a stroke.
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the fill
    /// would be unbounded — [`SelectionShape::All`] with nothing selected — or would
    /// exceed [`MAX_FILL_TILES`](super::fill::MAX_FILL_TILES). Appended last so
    /// postcard, which encodes an enum by variant *index*, keeps decoding older
    /// files.
    Fill {
        layer: LayerId,
        op: super::fill::FillOp,
    },

    /// Clip a layer to the paint beneath it, or stop (§14.4).
    ///
    /// A presentation property like the blend mode it is applied beside, and
    /// logged like one: it changes what the document *looks* like, so replay and
    /// peers both have to reproduce it. On the base of a group it clips the whole
    /// group to what lies under the group (§14.4.3) — the same outward reading its
    /// blend mode gets, which is why this needs no second action for groups.
    SetLayerClip(LayerId, bool),

    /// Perspective transform of the selected paint on `layer` within the map's
    /// source rect (§16.8): cut what the **author's** selection holds inside the
    /// rect, resample it once under the homography the map's corners define,
    /// stack it back over what remained — and carry the covered part of the
    /// author's mask along, unioned with what stayed outside the rect. Twelve
    /// floats in the log; every peer re-derives the same matrix from them.
    /// Appended last so postcard keeps decoding older files.
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the
    /// map is unusable — a non-convex target quad, a degenerate rect — or the
    /// rewrite exceeds the tile caps (§16.1).
    TransformPerspective {
        layer: LayerId,
        map: super::transform::PerspectiveMap,
    },

    /// Warp of the selected paint on `layer` within the mesh's source rect
    /// (§16.9): the same cut/stack/carry as
    /// [`TransformPerspective`](Self::TransformPerspective), under the smooth
    /// surface through the map's control grid. The log carries only the grid —
    /// a few dozen floats — and every peer subdivides it identically. Appended
    /// last so postcard keeps decoding older files.
    ///
    /// Deterministically **rejected** when the mesh folds (any sub-cell's
    /// Jacobian runs non-positive), is malformed, or exceeds the tile caps.
    TransformWarp {
        layer: LayerId,
        map: super::warp::WarpMap,
    },

    /// Copy a layer — **and everything it carries** — into its own stack,
    /// directly above the layer it was copied from (§14.8). The subtree travels
    /// as one for the reason [`RemoveLayer`](Self::RemoveLayer)'s does: the
    /// subtree *is* the group (§14.2).
    ///
    /// `ids` pairs every layer of that subtree, in composite order, with the id
    /// its copy takes; the first pair's source is the layer being duplicated.
    /// The copies' ids are minted by the author and travel in the log for the
    /// reason [`AddLayer`](Self::AddLayer)'s does — a replay must mint what the
    /// run that recorded it minted, and two peers duplicating at once must not
    /// land on one id (§17.9).
    ///
    /// Naming the **sources** as well is what lets the footprint be honest: a
    /// copy is a function of every tile and every property of every layer it
    /// copies, so a duplicate does not commute with a stroke or a rename inside
    /// the group, and an action that named only the root could not say so
    /// (§12.6).
    ///
    /// Deterministically **rejected** (the document is left unchanged) when the
    /// subtree holds a layer `ids` does not name — which is what a concurrent add
    /// into the group looks like from here. Appended last so postcard, which
    /// encodes an enum by variant *index*, keeps decoding older files.
    DuplicateLayer {
        ids: Vec<(LayerId, LayerId)>,
    },
}

impl ActionKind {
    /// What this action *is*, in two or three words — the caption a history
    /// scrubber puts on the step it is about to cross (§18.2.4).
    ///
    /// A `&'static str` and nothing more: a timeline showing a hundred steps needs
    /// them by the hundred, and the point of the caption is to tell a stroke from a
    /// layer change at a glance, not to describe either. Anything richer — which
    /// layer, what colour — is what the canvas beside it is for.
    pub fn label(&self) -> &'static str {
        match self {
            ActionKind::CommitStroke(_) => "Stroke",
            ActionKind::Fill { .. } => "Fill",
            ActionKind::Transform { .. } => "Transform",
            ActionKind::TransformPerspective { .. } => "Perspective",
            ActionKind::TransformWarp { .. } => "Warp",
            ActionKind::Select(_) => "Select",
            ActionKind::InvertSelection => "Invert selection",
            ActionKind::AddLayer { .. } => "Add layer",
            ActionKind::DuplicateLayer { .. } => "Duplicate layer",
            ActionKind::RemoveLayer(_) => "Remove layer",
            ActionKind::MoveLayer { .. } => "Reorder layer",
            ActionKind::SetLayerBlend(..) => "Blend mode",
            ActionKind::SetLayerClip(..) => "Clip layer",
            ActionKind::SetLayerOpacity(..) => "Layer opacity",
            ActionKind::SetLayerVisible(..) => "Layer visibility",
            ActionKind::SetLayerName(..) => "Rename layer",
            ActionKind::AddMatte { .. } => "Add frame",
            ActionKind::SetMatteRect(..) => "Move frame",
            ActionKind::SetMatteColor(..) => "Frame colour",
            ActionKind::SetBackground(_) => "Canvas colour",
            ActionKind::SetSurface(_) => "Canvas surface",
            ActionKind::Undo(_) => "Undo",
        }
    }
}

/// A committed document mutation with its identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub id: ActionId,
    pub kind: ActionKind,
}

/// Side-channel passed to [`history::Action::apply`]: the GPU resources needed
/// to render a stroke (§5). It owns cheap `Arc`-backed clones, so it
/// has no borrow lifetime — which is what lets it be the `Action::Context`.
pub struct ApplyCtx {
    pub pool: TilePool,
    pub stroke: StrokeRenderer,
    pub assets: crate::assets::AssetStore,
    pub selection: SelectionRenderer,
    pub transform: crate::gpu::transform::TransformRenderer,
    pub fill: crate::gpu::fill::FillRenderer,
    /// The device, so a canvas surface can be built here on demand.
    pub gpu: crate::gpu::context::GpuContext,
    /// The canvas surfaces and the bytes registered for them (§6.4).
    ///
    /// It lives here, rather than beside the compositor it also feeds, because the
    /// **deposit reads it**: the tooth gates the paint a stroke lays by the ground
    /// under it, so which surface a stroke sees is part of applying that stroke.
    /// Asked with [`DocState::surface`](super::state::DocState) *as the log stood at
    /// that action*, which is the whole reason `SetSurface` was made a logged action
    /// rather than a view setting — a stroke from before a mid-document switch
    /// deposits against the ground it was actually painted on, on replay and on a
    /// peer alike.
    pub surfaces: crate::gpu::registry::Registry<SurfaceId>,
}

impl ApplyCtx {
    /// The canvas surface `id` names, built on demand ([`Registry::get`]).
    ///
    /// Returns an owned handle rather than a borrow — `Surface` is a pair of
    /// reference-counted wgpu objects, so the clone is two atomic bumps — because
    /// the caller then borrows other fields of `self` to build the scene around it.
    ///
    /// [`Registry::get`]: crate::gpu::registry::Registry::get
    pub fn surface(&mut self, id: SurfaceId) -> crate::gpu::surface::Surface {
        let gpu = self.gpu.clone();
        self.surfaces.get(&gpu, id).clone()
    }
}

impl history::Action for Action {
    type State = DocState;
    type Context = ApplyCtx;
    // GPU work reports failure via wgpu's device error callbacks, not return
    // values, and tile allocation never fails — so applying an action is
    // genuinely infallible here (§5).
    type Error = std::convert::Infallible;
    /// An action commutes with everything its [`Footprint`] is disjoint from
    /// (§12.6) — which is what lets the history splice an undone
    /// action out past a peer's unrelated work instead of replaying it.
    ///
    /// [`Footprint`]: super::footprint::Footprint
    type Centralizer<'a> = super::footprint::Footprint;

    /// Remove this action's effect by restoring what it wrote from
    /// `previous_state` — the values under its footprint, nothing more, so the
    /// edits of commuting actions applied after it survive. Tiles come back as
    /// the same shared handles (copy-on-write means identity is equality), so
    /// this re-renders nothing.
    fn inverse(&self, previous_state: &DocState, state: &mut DocState) {
        *state = super::patch::unapply(self, previous_state, state);
    }

    fn apply(&self, state: DocState, ctx: &mut ApplyCtx) -> Result<DocState, Self::Error> {
        Ok(match &self.kind {
            // A matte has no tile map, so a stroke targeting one is refused
            // rather than swallowed or magically rasterized (§15.7).
            // Refusing here (not only in the frontend) is what keeps replay and
            // peers agreeing about a log that contains such a stroke.
            ActionKind::CommitStroke(rec) => {
                // Cloned out of the tree before anything is rebuilt: the tile map
                // is persistent, so this is a handful of `Arc` bumps, and it is
                // what keeps the borrow of `state` from outliving the rewrite.
                match state.layer(rec.layer).and_then(|l| l.tiles()).cloned() {
                    Some(base) => {
                        // The **author's** selection, as it stood at this point in
                        // the log, gates the stroke (§6.8,
                        // §17.3). Read from the state being folded over,
                        // so replay reproduces it exactly; keyed by the author, so a
                        // collaborator's lasso never clips this stroke.
                        let selection = state.selection_of(self.id.actor);
                        // The ground this stroke was painted on, as the log stood
                        // here — not as it stands now (§6.4). The tooth gates the
                        // deposit by it, so a mid-document `SetSurface` changes what
                        // comes *after* it and nothing before, on replay exactly as
                        // it did live.
                        let surface = ctx.surface(state.surface);
                        let tiles = ctx.stroke.render(
                            crate::gpu::stroke::StrokeScene {
                                pool: &ctx.pool,
                                assets: &ctx.assets,
                                base: &base,
                                selection: &selection,
                                surface: &surface,
                            },
                            rec,
                        );
                        state.map_layer(rec.layer, |l| l.with_tiles(tiles))
                    }
                    // Absent layer, or a matte — a matte has no tile map, so a
                    // stroke targeting one is refused rather than swallowed
                    // (§15.7). Refusing here and not only in the
                    // frontend is what keeps replay and peers agreeing about a
                    // log that happens to contain such a stroke.
                    None => state,
                }
            }
            ActionKind::AddLayer { id, carrier, above } => {
                state.insert_layer(*id, *carrier, *above)
            }
            // The copy's tiles are the shared handles the source already holds, so
            // duplicating a layer costs no GPU memory until one of the two is
            // painted on — copy-on-write is what makes this a cheap action rather
            // than a re-render of everything under it (§5.2).
            ActionKind::DuplicateLayer { ids } => state.duplicate_layer(ids),
            ActionKind::RemoveLayer(id) => state.remove_layer(*id),
            ActionKind::SetLayerBlend(id, blend) => state.set_layer_blend(*id, *blend),
            ActionKind::SetLayerClip(id, clip) => state.set_layer_clip(*id, *clip),
            ActionKind::SetLayerOpacity(id, opacity) => state.set_layer_opacity(*id, *opacity),
            ActionKind::SetLayerVisible(id, visible) => state.set_layer_visible(*id, *visible),
            ActionKind::SetLayerName(id, name) => {
                state.set_layer_name(*id, name.as_deref().map(Into::into))
            }
            ActionKind::MoveLayer { id, carrier, above } => state.move_layer(*id, *carrier, *above),
            // Resolved at the timeline layer (effective-sequence filtering); an
            // `Undo` should never be materialized through `apply`. Identity, so
            // a stray one is harmless rather than wrong.
            ActionKind::Undo(_) => state,
            // The author's own selection, and only ever the author's: the key is
            // taken from `self.id.actor`, never from the payload, so an action
            // cannot address anyone else's mask (§17.3).
            //
            // An op too large to rasterize (see `MAX_SELECTION_TILES`) leaves the
            // selection alone — deterministically, since the bound is a pure
            // function of the op, so peers and replays agree.
            ActionKind::Select(op) => {
                let prev = state.selection_of(self.id.actor);
                match ctx.selection.apply(&ctx.pool, &prev, op) {
                    Some(selection) => state.with_selection(self.id.actor, selection),
                    None => {
                        tracing::warn!("selection op too large to rasterize; ignored");
                        state
                    }
                }
            }
            ActionKind::InvertSelection => {
                let prev = state.selection_of(self.id.actor);
                let selection = ctx.selection.invert(&ctx.pool, &prev);
                state.with_selection(self.id.actor, selection)
            }
            ActionKind::SetSurface(id) => state.with_surface(*id),
            ActionKind::AddMatte {
                id,
                carrier,
                above,
                region,
                color,
            } => state.insert_matte(*id, *carrier, *above, *region, *color),
            ActionKind::SetMatteRect(id, min, max) => state.set_matte_rect(*id, *min, *max),
            ActionKind::SetMatteColor(id, color) => state.set_matte_color(*id, *color),
            ActionKind::SetBackground(rgb) => state.with_background(*rgb),
            // Cut the author's selected paint, restack it under the affine, and
            // carry the author's mask with it (§16). Gated and
            // keyed exactly as a stroke is: the mask comes off the state being
            // folded over, the actor off the action's own id. A matte or absent
            // layer refuses it, like a stroke; an unusable or oversized transform
            // is rejected deterministically, so peers and replays agree.
            ActionKind::Transform { layer, affine } => transform_apply(
                state,
                ctx,
                self.id.actor,
                *layer,
                &crate::document::transform::TransformMap::Affine(*affine),
            ),
            // The rect-scoped siblings (§16.8, §16.9): identical shape — cut,
            // restack, carry the mask — differing only in the map handed to
            // the renderer.
            ActionKind::TransformPerspective { layer, map } => transform_apply(
                state,
                ctx,
                self.id.actor,
                *layer,
                &crate::document::transform::TransformMap::Perspective(*map),
            ),
            ActionKind::TransformWarp { layer, map } => transform_apply(
                state,
                ctx,
                self.id.actor,
                *layer,
                &crate::document::transform::TransformMap::Warp(map.clone()),
            ),
            // Lay a parcel of paint through the region's coverage, gated by the
            // author's selection — the same gate a stroke passes through, so a fill
            // is clipped by a selection exactly as a brush is
            // (§18.0.4). Refused on a matte or absent layer like a stroke; refused
            // deterministically when unbounded or oversized, so peers and replays
            // agree about a log that contains one.
            ActionKind::Fill { layer, op } => {
                match state.layer(*layer).and_then(|l| l.tiles()).cloned() {
                    Some(base) => {
                        let selection = state.selection_of(self.id.actor);
                        match ctx.fill.apply(&ctx.pool, &base, &selection, op) {
                            Some(tiles) => state.map_layer(*layer, |l| l.with_tiles(tiles)),
                            None => {
                                tracing::warn!(
                                    "fill rejected (unbounded region or too many tiles); ignored"
                                );
                                state
                            }
                        }
                    }
                    None => state,
                }
            }
        })
    }
}

/// The shared body of the three transform actions (§16): cut the author's
/// selected paint, restack it under `map`, and carry the author's mask with it.
/// Gated and keyed exactly as a stroke is — the mask comes off the state being
/// folded over, the actor off the action's own id. A matte or absent layer
/// refuses it, like a stroke; an unusable or oversized map is rejected
/// deterministically, so peers and replays agree.
fn transform_apply(
    state: DocState,
    ctx: &ApplyCtx,
    actor: ActorId,
    layer: LayerId,
    map: &crate::document::transform::TransformMap,
) -> DocState {
    match state.layer(layer).and_then(|l| l.tiles()).cloned() {
        Some(base) => {
            let selection = state.selection_of(actor);
            match ctx.transform.apply(&ctx.pool, &base, &selection, map) {
                Some((tiles, moved_selection)) => state
                    .map_layer(layer, |l| l.with_tiles(tiles))
                    .with_selection(actor, moved_selection),
                None => {
                    tracing::warn!("transform rejected (unusable map or too many tiles); ignored");
                    state
                }
            }
        }
        None => state,
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
