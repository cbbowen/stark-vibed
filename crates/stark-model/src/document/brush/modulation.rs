//! The pen mappings (§6.2): what a [`Modulation`] reads off the pen, the response
//! it shapes that into, and the five records naming which of a brush's knobs one
//! may be pointed at.
//!
//! **A modulation can only ever scale a parameter down.** Every bound the renderer
//! derives from a brush's numbers — the frozen-span radius bound, the region fit,
//! the choice of render path, the flattener's exchange step — is stated against the
//! brush's own values and stays sound without any part of it knowing that
//! modulation exists.
//!
//! The layer the other two sit on: nothing here reaches an effect or
//! [`BrushParams`](super::BrushParams) in code, which is what lets both reach it.

use serde::{Deserialize, Serialize};

use crate::sanitize::{clamp01, finite_or};

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
    /// [`OrientationSource::Pen`](super::OrientationSource::Pen) already answers.
    pub tilt: f32,
}

/// Both sources reading the same number — how a test states a pen. Here rather than
/// in one test module because the mappings are exercised from two of them.
#[cfg(test)]
pub(super) fn pen(x: f32) -> PenState {
    PenState {
        pressure: x,
        tilt: x,
    }
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
/// A bound rather than taste. A segment sweeps at *one* value of every parameter, so a
/// steep response is paid for in segments — `gpu::stroke::budget::flatten_tolerance`
/// divides the attribute budget by
/// [`BrushParams::max_slope`](super::BrushParams::max_slope) to keep a modulated ramp
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

/// The algebra the five mapping records share (§6.2): a fixed set of targets, each
/// either unmapped or one [`Modulation`].
///
/// The records stay five distinct types — each is its own save-format record, and
/// "a mapping cannot name a knob its brush does not have" is what having five of
/// them buys. What they have no reason to state five times is the arithmetic over
/// those targets: whether any is mapped, sanitizing them, and the steepest response
/// across them are each a pure function of [`all`](Self::all).
///
/// **[`all`](Self::all) and [`from_all`](Self::from_all) are inverses, and both are
/// exhaustive.** A pattern with no `..` and a literal with no `..` are each refused
/// by the compiler once a target is added to the struct, so a new target has to be
/// listed in two places before this builds — where reading the fields one by one
/// would leave it quietly missing from [`max_slope`](Self::max_slope), and an
/// under-estimated slope is not an error anywhere, it is a modulated ramp drawn as a
/// staircase.
///
/// `N` is the trait's parameter rather than an associated const because an
/// associated const cannot be an array length without `generic_const_exprs`.
pub(super) trait Modulations<const N: usize>: Copy + Sized {
    /// Every target at once, in the order the record declares them.
    fn all(&self) -> [Option<Modulation>; N];

    /// The record those targets make — [`all`](Self::all) read backwards.
    fn from_all(all: [Option<Modulation>; N]) -> Self;

    /// Every mapped target sanitized, the unmapped ones left unmapped.
    fn sanitized(self) -> Self {
        Self::from_all(self.all().map(|m| m.map(Modulation::sanitized)))
    }

    /// The steepest response across these targets (`mod_slope`).
    fn max_slope(&self) -> f32 {
        mod_slope(&self.all())
    }
}

/// The pen mappings every brush has, whatever its
/// [`effect`](super::BrushParams::effect) (§6.2): the targets that shape the **tip** —
/// its size, its elongation, and how far it settles into the substrate's tooth. The
/// targets that drive an effect's own rates live *with* the effect
/// ([`PaintModulations`], [`EraseModulations`]), so a mapping cannot name a knob its
/// brush does not have.
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
    /// Scales [`BrushParams::size`](super::BrushParams::size).
    pub size: Option<Modulation>,
    /// Scales [`ToothParams::give`](super::ToothParams::give) — how far the tool
    /// settles into the canvas's substrate (§6.4). Mapped to pressure this is the
    /// charcoal behaviour: barely touching the paper the tip has no give and prints the
    /// peaks alone, and borne down it presses past the falls it was bridging, so the
    /// grain fills in.
    pub tooth_give: Option<Modulation>,
    /// Scales [`BrushParams::stretch`](super::BrushParams::stretch) — how far the
    /// extent elongates along the brush's facing axis (§6.6). Mapped to
    /// [`ModSource::Tilt`] with
    /// [`OrientationSource::Pen`](super::OrientationSource::Pen) this is the pencil
    /// behaviour: lean the pen over and the contact patch draws out along the lean,
    /// exactly as a real tip's does.
    pub stretch: Option<Modulation>,
}

impl BrushModulations {
    /// The everyday brush: **size follows pressure**, linearly, all the way to
    /// nothing. Stated here rather than wired into the segment generator, which is
    /// what lets it be turned off or pointed somewhere else like any other mapping.
    ///
    /// Not [`Default`], which is "no mapping at all". Both are wanted:
    /// [`BrushParams::default`](super::BrushParams::default) takes this, while a preset
    /// that means to hold its width whatever the hand does asks for the empty set and
    /// gets it.
    pub const PRESSURE_SIZE: Self = Self {
        size: Some(Modulation::linear(ModSource::Pressure)),
        tooth_give: None,
        stretch: None,
    };

    /// [`Self::PRESSURE_SIZE`] as a function, for `#[serde(default = "…")]` — which
    /// takes a path to call and cannot name a constant. Its one caller is
    /// [`BrushParams::modulation`](super::BrushParams::modulation), which is where the
    /// argument for it is written out.
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
}

impl Modulations<3> for BrushModulations {
    fn all(&self) -> [Option<Modulation>; 3] {
        let Self {
            size,
            tooth_give,
            stretch,
        } = *self;
        [size, tooth_give, stretch]
    }

    fn from_all([size, tooth_give, stretch]: [Option<Modulation>; 3]) -> Self {
        Self {
            size,
            tooth_give,
            stretch,
        }
    }
}

/// The pen mappings whose targets exist only while **painting** (§6.2) — the
/// [`PaintEffect`](super::PaintEffect)'s rate and its ceiling. With the effect it
/// modulates ([`PaintEffect::modulation`](super::PaintEffect::modulation)) rather than
/// beside the tip's own mappings, so a mapping cannot name a knob its brush does not
/// have.
///
/// The ceiling is a target with a cost the rate does not have: a segment's
/// share of the stroke's coverage has to be remembered at the segment's own
/// ceiling, so a stroke that maps it carries one more accumulator lane per
/// touched tile (`stamp.wesl`'s ceiling lane, §6.2). A brush that leaves it
/// unmapped never pays for it.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct PaintModulations {
    /// Scales [`PaintEffect::flow`](super::PaintEffect::flow) — the brush's own paint,
    /// "Flow" in the UI.
    pub flow: Option<Modulation>,
    /// Scales [`PaintEffect::opacity`](super::PaintEffect::opacity) — how much of a
    /// full stroke this stretch of it is. Mapped to pressure the mark is full where the
    /// pen bore down and faint where it eased off; where a stroke covers a spot twice,
    /// the spot shows the **larger** of the two ceilings (§6.2, the ceiling lane), so a
    /// light pass back over a heavy mark leaves it, and a heavy pass over a light one
    /// fills it in.
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
}

impl Modulations<2> for PaintModulations {
    fn all(&self) -> [Option<Modulation>; 2] {
        let Self { flow, opacity } = *self;
        [flow, opacity]
    }

    fn from_all([flow, opacity]: [Option<Modulation>; 2]) -> Self {
        Self { flow, opacity }
    }
}

/// The pen mappings whose targets exist only while **working wet paint** — the effect's
/// own flow and ceiling, and the four rates of [`BrushDynamics`](super::BrushDynamics)
/// (§6.2). With the effect they modulate
/// ([`WetEffect::modulation`](super::WetEffect::modulation)) for [`PaintModulations`]'
/// reason.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct WetModulations {
    /// Scales [`WetEffect::flow`](super::WetEffect::flow) — the whole of what the tool
    /// does, "Flow" in the UI: mapped to pressure, a light touch lays less *and* smears
    /// less.
    pub flow: Option<Modulation>,
    /// Scales [`WetEffect::opacity`](super::WetEffect::opacity) —
    /// [`PaintModulations::opacity`]'s target, reaching what the brush **mints**
    /// (§6.2): the `add` paint's ceiling. The `charge` glob is scaled by the dial
    /// alone, being minted once before the pen has moved; moved paint is under no
    /// ceiling at all.
    #[serde(default)]
    pub opacity: Option<Modulation>,
    /// Scales [`BrushDynamics::add`](super::BrushDynamics::add) — the brush's own paint
    /// alone, for a brush that lays more under the pen without working the canvas
    /// harder.
    pub add: Option<Modulation>,
    /// Scales [`BrushDynamics::lift`](super::BrushDynamics::lift).
    pub lift: Option<Modulation>,
    /// Scales [`BrushDynamics::deposit`](super::BrushDynamics::deposit).
    pub deposit: Option<Modulation>,
    /// Scales [`BrushDynamics::bleed`](super::BrushDynamics::bleed).
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
}

impl Modulations<6> for WetModulations {
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

    fn from_all([flow, opacity, add, lift, deposit, bleed]: [Option<Modulation>; 6]) -> Self {
        Self {
            flow,
            opacity,
            add,
            lift,
            deposit,
            bleed,
        }
    }
}

/// The pen mappings whose targets exist only while **erasing** (§6.12): the
/// eraser's rate and its ceiling, [`PaintModulations`]' pair on the removing
/// side, carried the same way (the erase sweep's ceiling lane).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct EraseModulations {
    /// Scales [`EraseEffect::flow`](super::EraseEffect::flow) — how fast a light touch
    /// feathers the bite in.
    pub flow: Option<Modulation>,
    /// Scales [`EraseEffect::opacity`](super::EraseEffect::opacity) — how much of what
    /// it finds this stretch of the stroke may remove. Mapped to pressure, a light
    /// touch thins and a heavy one clears.
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
}

impl Modulations<2> for EraseModulations {
    fn all(&self) -> [Option<Modulation>; 2] {
        let Self { flow, opacity } = *self;
        [flow, opacity]
    }

    fn from_all([flow, opacity]: [Option<Modulation>; 2]) -> Self {
        Self { flow, opacity }
    }
}

/// The pen mappings whose targets exist only while **liquifying** (§6.13) — the one
/// rate a [`LiquifyEffect`](super::LiquifyEffect) has. With the effect it modulates for
/// [`PaintModulations`]' reason: a mapping cannot name a knob its brush does not have.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default, carbonite::Schema)]
pub struct LiquifyModulations {
    /// Scales [`LiquifyEffect::strength`](super::LiquifyEffect::strength) — how hard
    /// the paint follows the tip. Mapped to pressure this is the natural finger: barely
    /// touching, the picture slides a little; borne down, it keeps pace with the hand.
    pub strength: Option<Modulation>,
}

impl LiquifyModulations {
    pub fn strength(&self, pen: PenState) -> f32 {
        mod_factor(self.strength, pen)
    }
}

impl Modulations<1> for LiquifyModulations {
    fn all(&self) -> [Option<Modulation>; 1] {
        let Self { strength } = *self;
        [strength]
    }

    fn from_all([strength]: [Option<Modulation>; 1]) -> Self {
        Self { strength }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let none = WetModulations::default();
        assert!(
            none.all().iter().all(Option::is_none),
            "the fixture maps nothing"
        );
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
