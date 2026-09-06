//! The vocabulary a stroke is carried in — a [`Sweep`], the [`Paint`] rates over it,
//! the [`Segment`] that is both, a [`BleedFire`]'s window — and the one funnel a
//! record becomes those through (§6.2).
//!
//! Both render paths flatten through [`generate_segments_in`], so both see the same
//! segments for the same record — which is what lets a live tail and the commit that
//! replaces it agree pixel for pixel. Everything downstream of it reads these types
//! and never the record: what the tiles a piece touches are measured from
//! ([`region`](super::region)), what the dynamics loop dispatches over
//! ([`dynamics`](super::dynamics)), and what the sweep shader is instanced with.

use stark_model::document::{BrushParams, OrientationSource, PenState, StrokeRecord};
use stark_model::geom::Vec2;

use super::StrokeSpans;
use super::budget::lambda;

/// The fixtures every suite in [`stroke`](super) builds its sweeps with — here because
/// this is the module that owns the type they build (see the module's own docs).
#[cfg(test)]
pub(super) mod testing;

// Two subjects a segment is *built from* and does not read back: the taper's
// profile and its cut, and the stretch's map. Each is its own file with its own
// tests; this one is the vocabulary and the funnel.
mod stretch;
pub(super) mod taper;

pub(super) use stretch::Stretch;
use taper::Taper;

/// **Where a tip goes, and how wide it is** — everything the shaders need to unroll
/// one sweep, and nothing about what the tip is doing while it travels.
///
/// The centreline is a **circular arc**, not a chord: `start` and `dir` give the
/// frame it leaves in, `curvature` bends it, and `length` measures along it. A
/// straight sweep is `curvature == 0` and is what every quantity below reduces to,
/// exactly — see [`crate::path::fit_arc`].
///
/// Split from the paint rates ([`Paint`]) because the module has two things that are
/// swept and only one of them paints. A [`BleedFire`]'s window is a stretch of lateral
/// diffusion at a standing tip: it has a start, a bend and a travel, and it has no
/// `add`, no `lift` and no `deposit`. Built as a whole [`Segment`] instead, it would
/// carry the crossing segment's rates for `dynamics_plan` to zero back out lane by
/// lane. Every box and every rect in this module is a function of this half alone, so
/// they take it alone and cannot read a rate that is not there.
#[derive(Copy, Clone)]
pub(super) struct Sweep {
    pub(super) start: Vec2,
    /// Unit travel tangent **at the segment's start** — the x axis of the frame the
    /// sweep is integrated in. On a curved segment the tangent turns as the tip
    /// travels; this is where it begins.
    pub(super) dir: Vec2,
    /// Signed curvature of the centreline (1/canvas px), positive turning towards
    /// the left of `dir`. Exactly 0 for a straight sweep, which both render paths
    /// branch on — so a stroke the arc fit declines to bend is bit-identical to one
    /// drawn before arcs existed (§6.2).
    pub(super) curvature: f32,
    /// The tip's own radius in canvas px at the segment's **midpoint** — the mean of
    /// its two ends, and the reference every rate the host measures is denominated in
    /// (the bleed cadence, the stencil's diffusivity, the touch-down dab).
    ///
    /// The *reference*, because a segment does not have a radius: the tip is a
    /// function of travel, and [`radius_ramp`](Self::radius_ramp) is the rest of that function.
    ///
    /// It is also **the frame the sweep is unrolled in** — region px per brush-local
    /// unit, the units everything the shaders read out of a brush-local coordinate is
    /// in. The two were separate quantities while a pen-oriented stamp's prefix-τ
    /// volume was padded to keep a turned mask's corners; a canonical mask's content
    /// lies inside the disc inscribed in its square (`stark_assetid::coverage`, §6.6),
    /// a rotation maps that disc to itself, so every volume is baked unpadded and the
    /// frame is the tip for every brush there is.
    pub(super) radius: f32,
    /// How much the tip grows across this segment, as a fraction of
    /// [`radius`](Self::radius): `(r_end − r_start) / radius`, so the tip in force a
    /// fraction `u` of the way through is `radius · (1 + ramp·(u − ½))`. **Zero for a
    /// segment whose tip does not change**, which is every segment of an untapered
    /// brush the pen is not driving — and the shaders branch on that zero, so those
    /// strokes render exactly as they did before a ramp existed.
    ///
    /// Why a segment carries one at all: the tip is scaled by the taper and by the
    /// size modulation, both of which vary *with distance travelled*, and a sweep at
    /// a single radius puts a C⁰ break in the stroke's outline at every segment
    /// boundary. Subdividing shrinks the break but cannot remove it — and a step in
    /// an edge is visible far below the pixel it is quantized to, which is what made
    /// a 500 px tapered tip draw as a comb of ~5 px sawteeth (2026-08-14). Carried as
    /// a ramp the outline is continuous *by construction*: adjacent segments agree on
    /// the radius at the knot they share, because both compute it from the same pen
    /// and the same taper at the same arc length.
    ///
    /// **The same agreement is what the deposit rests on**, one level down. The two
    /// ends of a segment's swept span are denominated against `r_start` and `r_end`
    /// rather than against the reference radius (`stamp_common::Sweep::span`), so one
    /// segment's trailing coordinate and the next one's leading coordinate are the same
    /// expression — and a point's exposure over a whole pass comes out the mask's row
    /// total whatever the cut. Denominated both ends in the reference instead, the
    /// exposure rippled at the flattener's own cadence, which is what printed a tapered
    /// stamp brush as a fan of stepped streaks (2026-08-20).
    ///
    /// **`|ramp| < 2` structurally**, so the tip in force is positive everywhere
    /// without a clamp: the ends are floored at `0.5` by [`generate_segments_in`], and
    /// `|r₁ − r₀| < r₁ + r₀` for any two positive radii.
    ///
    /// Relative rather than absolute (px) because the tip and the frame the sweep is
    /// unrolled in are one radius — [`radius`](Self::radius) is both.
    pub(super) radius_ramp: f32,
    /// How far from the centreline this tip's deposit can land, in canvas px.
    ///
    /// Scaled by the segment's **widest** tip rather than its mean, since the ramp
    /// makes those different numbers and this one bounds a box: under-reporting it is
    /// a stroke clipped at a tile boundary (see [`coverage_bounds`](super::region::coverage_bounds)).
    ///
    /// Every shape is swept over brush-local `|x| ≤ 1, |y| ≤ 1` — the whole domain of
    /// the prefix-τ volume — but nothing any shape can paint lies outside the **disc**
    /// inscribed in that square, at any orientation: the round tip by construction
    /// (`tips::round_coverage`), an image stamp by its canonical form's reach
    /// normalization (`stark_assetid::coverage`, §6.6). So the widest tip's radius,
    /// drawn out by the stretch, is the exact bound — where this once carried a `√2`
    /// for a mask that might fill its corners, and every stamp's boxes were that much
    /// larger than what they held.
    pub(super) reach: f32,
    /// Arc length of the centreline (canvas px) — the tip's own travel, which is the
    /// measure every rate in both paths is denominated in.
    pub(super) length: f32,
    /// Shape orientation for this segment as a fraction of a full turn ∈ [0, 1): the
    /// relative angle between the shape's native axis and **the direction the sweep is
    /// integrated along**, used to pick the prefix-τ orientation layer. 0 for
    /// follow-stroke (§6.6).
    ///
    /// That direction is the travel for every unstretched tip, and
    /// [`Stretch::turns`] round from it for a stretched one — which is the sense in
    /// which a stretch is *another slice of the same volume* rather than another bake.
    pub(super) orient: f32,
    /// How far this segment's tip is drawn out along its facing axis, as the map from
    /// the reference travel frame into the frame the prefix-τ volume is read in
    /// (§6.6). [`Stretch::NONE`] for every brush that does not ask, which is the
    /// identity the shaders branch on.
    pub(super) stretch: Stretch,
    /// Arc length from the stroke start to this sweep's start (canvas px) — the
    /// third axis of the color-dynamics noise lookup (§6.2).
    pub(super) dist: f32,
}

impl Sweep {
    /// The tip in force a fraction `u` of this sweep's travel in, canvas px — the
    /// host's statement of the ramp `stamp_common::radius_ramp_scale` applies, so the two
    /// definitions can be read against each other.
    ///
    /// `u` is clamped, like the shader's: past either end the tip has, as far as this
    /// sweep is concerned, stopped at the end it reached.
    ///
    /// Only the tests ask — the renderer evaluates the ramp on the GPU, per fragment,
    /// and the one thing the *host* needs from it (the widest tip, which sizes the
    /// coverage box) is [`widest_tip`](Self::widest_tip).
    #[cfg(test)]
    pub(super) fn tip_at(&self, u: f32) -> f32 {
        self.radius * (1.0 + self.radius_ramp * (u.clamp(0.0, 1.0) - 0.5))
    }

    /// The widest tip this sweep reaches, canvas px.
    ///
    /// Spelled as the shader spells it (`stamp_common::sweep_vertex`'s `hull`) rather
    /// than as `max(r_start, r_end)`, which it equals algebraically. The two can differ
    /// by an ulp in floats, and this one has to come out **no smaller** than the strip
    /// the GPU rasterizes: it is what [`coverage_bounds`](super::region::coverage_bounds) grows the box by, and a box
    /// narrower than its own geometry is a stroke clipped at a tile boundary.
    fn widest_tip(&self) -> f32 {
        self.radius * (1.0 + 0.5 * self.radius_ramp.abs())
    }
}

/// The brush's paint rates **as the pen asked for them here** (§6.2): the axes of
/// [`BrushDynamics`](stark_model::document::BrushDynamics) and the effect's flow,
/// each scaled by whatever
/// [`BrushModulations`](stark_model::document::BrushModulations) and its effect-side
/// siblings map onto it, plus the tooth's depth.
///
/// They live on the segment rather than on the stroke because that is now what they
/// are — the pen attributes they follow are interpolated per segment, and the
/// flattener already holds their step to
/// [`FlattenTolerance::attribute`](crate::path::FlattenTolerance::attribute). Each is
/// at most the brush's own value, never more, which is what lets every bound taken
/// against `rec.brush` stay a bound (see [`Modulation`](stark_model::document::Modulation)).
///
/// `charge` is absent on purpose: it is the tool's *initial* load, one number for the
/// whole stroke, and there is no per-segment version of it to carry.
#[derive(Copy, Clone)]
pub(super) struct Paint {
    pub(super) add: f32,
    /// The exchange rates as the shader runs them — `flow · λ(axis)` (§6.2), each
    /// vertical axis as the pen asked for it here through
    /// [`lambda`](super::budget::lambda), then the wet effect's modulated flow on
    /// the exponent. Folded here, where `add` and `bleed` already take the flow,
    /// so that nothing downstream holds a factor it could apply twice: the axes
    /// and the flow do not survive past this struct. Zero — no transfer — on
    /// every effect but wet.
    pub(super) lambda_lift: f32,
    pub(super) lambda_deposit: f32,
    pub(super) bleed: f32,
    /// The liquify effect's follow fraction as the pen asked for it here
    /// (§6.13) — [`LiquifyEffect::strength`](stark_model::document::LiquifyEffect)
    /// modulated, and nonzero only on a liquify stroke. The plan divides the
    /// tip's peak τ density out before it reaches the shader's lane
    /// (`dynamics::plan`), because the tip is resolved there and not here.
    pub(super) drag: f32,
    /// How much give this segment's tip has against the canvas substrate (§6.4) — the
    /// brush's `tooth_give`, likewise modulated. Not a paint rate: it gates `add` per
    /// *texel* from the substrate under it, in the shader.
    ///
    /// The give alone. How wide the transition around it is is `tooth_softness`, and
    /// that one is not modulated and not here — it rides `StrokeConstants` with the
    /// other per-stroke numbers both paths read.
    ///
    /// **A modulation scales this down towards the driest tip**, which is the whole
    /// reason the knob is quoted as the give: it is what makes a pressure mapping the
    /// charcoal rather than its opposite (`BrushModulations::tooth_give`).
    pub(super) tooth_give: f32,
    /// The ceiling's **modulation factor** as the pen asked for it here (§6.2):
    /// what this segment's share of the stroke's coverage is capped at, as a
    /// fraction of the effect's own dial. The dial itself — with the mask's
    /// opacity folded in — stays a stroke constant (`StrokeConstants::opacity`);
    /// only the pen's factor rides the segment, so a brush with the target
    /// unmapped carries exactly 1 and the ceiling lane holds the plain coverage.
    ///
    /// The **mean of the segment's two ends**, like [`Sweep::radius`] and for its
    /// reason — see [`opacity_ramp`](Self::opacity_ramp).
    pub(super) opacity: f32,
    /// How much the ceiling's factor changes across this segment: `o_end −
    /// o_start`, so the factor a fraction `u` of the way through is
    /// `opacity + opacity_ramp·(u − ½)` (`paint_common::ceiling_at`). **Zero for a
    /// segment whose ceiling does not change**, which is every segment of every
    /// brush the pen does not drive that way, and the shaders branch on that zero.
    ///
    /// Why a segment carries one at all is [`Sweep::radius_ramp`]'s argument, one target
    /// over: read once per segment the ceiling is piecewise constant, and a stroke
    /// drawn at a realistic report rate is a handful of segments wide — so the
    /// mark came out in bands, stepping at every cut. Carried as a ramp the
    /// ceiling is continuous *by construction*: adjacent segments agree at the
    /// knot they share, because both read the pen there at the same arc length.
    ///
    /// **Absolute where the radius ramp is relative**, which is the one way the
    /// two differ: a ceiling is a fraction rather than a scale, so the interpolant
    /// stays inside `[min, max]` of two numbers already in `[0, 1]` and nothing
    /// here has to defend a positive product.
    pub(super) opacity_ramp: f32,
    /// How the `add` source rate changes across this segment — `end − start`, read
    /// at a texel's own travel by `stamp_common::add_of` and
    /// `dynamics.wesl::rate_across`. [`opacity_ramp`](Self::opacity_ramp)'s
    /// construction, and its argument: a rate held constant across a segment steps
    /// at every cut.
    ///
    /// It costs the deposit nothing to be exact about: the deposit is `∫ add dτ`
    /// over the segment, `add` is one function of arc length whichever side of a
    /// knot reads it, and a definite integral cut in two is the sum of its pieces.
    /// So this is as independent of the flattening as the constant it replaces, and
    /// nearer the integral than a midpoint sample.
    pub(super) add_ramp: f32,
    /// How the tooth's give changes across this segment (§6.4) — the same
    /// construction, on the one gate that is per *texel* rather than per parcel.
    pub(super) tooth_give_ramp: f32,
    /// How the liquify follow changes across this segment (§6.13) — likewise. The
    /// travel bound it is held to is the segment's, and interpolating between two
    /// ends that both respect it cannot leave it.
    pub(super) drag_ramp: f32,
}

impl Default for Paint {
    /// Every rate at zero — a tool doing nothing — the ceiling factor at its
    /// **neutral 1** (a scale, so a zero would not be "none" but a ceiling that
    /// admits nothing), and the tooth at **full give**, which is the same statement
    /// about the substrate: a tip that follows every fall deposits exactly what
    /// it would with no substrate under it at all.
    ///
    /// Written out rather than derived, because the knob runs the other way
    /// (`ToothParams::give`) and a derived zero here would be the *driest* tip
    /// there is — a default that gates paint away rather than one that does nothing.
    fn default() -> Self {
        Self {
            add: 0.0,
            lambda_lift: 0.0,
            lambda_deposit: 0.0,
            bleed: 0.0,
            drag: 0.0,
            tooth_give: stark_model::document::ToothParams::DEFAULT_GIVE,
            opacity: 1.0,
            opacity_ramp: 0.0,
            add_ramp: 0.0,
            tooth_give_ramp: 0.0,
            drag_ramp: 0.0,
        }
    }
}

/// One swept segment of the stroke: where the tip went, and what it was doing.
#[derive(Copy, Clone)]
pub(super) struct Segment {
    pub(super) sweep: Sweep,
    pub(super) paint: Paint,
}

/// One crossing of the bleed cadence (§6.2): which segment it fires after, the stretch
/// of path it relaxes over, and the one axis it uses.
///
/// A named type rather than the `(usize, Segment)` it was, and a [`Sweep`] rather than
/// a whole segment. A firing lays no paint — `dynamics_plan` zeroes every vertical rate
/// on the slot it becomes — so carrying rates here meant copying five numbers in for
/// the sole purpose of writing them back out, and left `radius_ramp: 0.0` as a field a window
/// had to remember not to set. It cannot set one now.
#[derive(Copy, Clone)]
pub(super) struct BleedFire {
    /// Index into the segments this firing was derived from — the segment it follows.
    pub(super) after: usize,
    pub(super) window: Sweep,
    /// The crossing segment's modulated `bleed`, which is the only axis a firing reads.
    pub(super) bleed: f32,
}

// Per-segment instance data for the sweep shader, generated from `stamp.wesl`'s own
// vertex parameters (§6.10) — including the prose on each lane, which now lives
// beside the declaration that decides how it is read.
pub(super) use stark_shaders::mirror::stamp::SegmentInstance;

// --- swept arcs ----------------------------------------------------------------
//
// The arc a flattened edge stands for is [`crate::path::fit_arc`]'s, called here with
// the very cap the flattener called it with (`FlattenTolerance::max_arc_curvature`,
// set by [`flatten_budget`](super::budget::flatten_budget) from
// [`MAX_TIP_TURN`](super::MAX_TIP_TURN)). One function, one rule, so the geometry the
// flattener priced is the geometry that gets swept — and neither can spend the
// positional budget on a primitive the other does not use.

/// Where the pen was, at the point a segment samples it.
///
/// Attributes are constant across a swept segment, so they are taken at its
/// **midpoint** rather than its start — with adaptive flattening a segment can be
/// long, and start-sampling would lag every ramp by half a segment.
///
/// Both axes are already clamped to what a pen can report, because the fitter clamps
/// the *curve* and not just the control polygon (`PathFitter::path`), so the
/// modulations below are honest without a second guard.
struct At {
    pos: Vec2,
    pressure: f32,
    tilt: Vec2,
}

/// The arc one segment travels along, as [`generate_segments_in`]'s builder is handed
/// it — the path half of a [`Sweep`], before the tip is measured onto it.
///
/// `dir` is the tangent the sweep *starts* along — the frame's x axis — while
/// `mid_dir` is the one at the midpoint, the same midpoint-sampling argument applied
/// to the one attribute that reads a direction. They are the same vector on a
/// straight segment.
///
/// `dist` is the exception to that rule: it is the arc length at the segment's
/// **start**, because the shader adds the fragment's own offset along the travel to
/// it (`stamp_common.wesl`).
struct Track {
    dir: Vec2,
    mid_dir: Vec2,
    curvature: f32,
    length: f32,
    dist: f32,
}

/// A segment's two **ends**, for the two quantities that ramp across it rather
/// than being sampled at its midpoint: the tip ([`Sweep::radius_ramp`]) and the pen the
/// ceiling's factor is read from ([`Paint::opacity_ramp`]).
///
/// One value because they are one question — what is in force at each end — and
/// because both rest on the same property: each is a function of arc length
/// alone, so the shared end of two adjacent segments resolves to the same number
/// on both sides. Bundling them is what keeps a third such quantity from being
/// threaded through as a fourth argument.
struct Ends {
    /// The tip in canvas px at the start and at the end, the taper's own factor
    /// at each already folded in — which only the caller knows.
    tip: (f32, f32),
    /// The pen as the modulations read it at those same two points.
    pen: (PenState, PenState),
}

/// The effect's rates at one pen reading — what [`Paint`] carries, before the two
/// ends are averaged and ramped. Named lanes, for the reason the plan's `Slot` is
/// (`plan.rs`): a positional tuple of five `f32`s was destructured three times per
/// segment with `_` placeholders, and nothing checked the positions agreed.
#[derive(Clone, Copy)]
struct Rates {
    add: f32,
    lambda_lift: f32,
    lambda_deposit: f32,
    bleed: f32,
    drag: f32,
}

impl Rates {
    const ZERO: Self = Rates {
        add: 0.0,
        lambda_lift: 0.0,
        lambda_deposit: 0.0,
        bleed: 0.0,
        drag: 0.0,
    };
}

/// Build swept segments from the fitted control points (§6.2): flatten
/// the curve adaptively, then make each polyline edge a segment. This is where the
/// brush's fixed numbers become the per-segment ones the shaders read: the radius
/// follows the size mapping and the stroke's start/end tapers, and each paint rate
/// follows whatever [`BrushModulations`](stark_model::document::BrushModulations) points at it
/// (§6.2). **It is the only place a modulation is resolved** — both render paths
/// flatten through here, so a live tail and the commit that replaces it cannot read
/// the pen differently.
///
/// **The `drain` falloff is deliberately not here.** It is a function of arc length
/// alone, and every shader that reads a segment already knows the arc length of the
/// fragment it is shading (`dist` plus the fragment's own offset along the travel), so
/// it is evaluated there instead of being baked in per segment. That is not a
/// micro-optimization: a per-segment factor makes the paint laid depend on where the
/// segment boundaries happened to fall, which is the one thing §6.2 works to keep out
/// of the deposit. Evaluated per fragment it drops out of the sum entirely — the
/// stroke lays `a(arc) · Στ`, and `Στ` is already independent of the cut — so the
/// flattener need not buy accuracy for it with segments
/// (see [`flatten_budget`](super::budget::flatten_budget)).
///
/// Returns the range's segments plus the arc length at its end — measured on the
/// emitted polyline rather than recomputed, so the range that resumes from it starts
/// on the exact accumulator these segments were built with.
///
/// One thing here is measured against the stroke's **whole** length, which only a
/// range that reaches its final span knows: the trailing taper (a range that stops
/// short takes the leading taper alone, [`Taper::resolve`]). It is sound rather than
/// approximate, and [`safe_frozen`](super::safe_frozen) is the one rule that makes
/// it so.
///
/// A stroke deposits exactly its own travel — a swept deposit is a definite
/// integral, so a press that has not moved lays nothing, and the tool says so
/// honestly rather than padding a minimum (the retired `DAB_TRAVEL` dwell): the
/// hover's mark previews what a press would lay before it is made (§18.1.10),
/// and a release that cannot deposit commits nothing (`Session::end_stroke`).
///
/// And the stroke's own travel begins at the record's **marker**
/// ([`StrokeRecord::start`]): the curve may extend back through the run-up —
/// motion from before the press, fitted in so the entry's direction and
/// curvature are measured rather than guessed (§6.2) — and the flattening
/// leaves everything before the marker out
/// ([`flatten_spans_from`](crate::path::flatten_spans_from)). Because the trim
/// happens here, in the funnel both render paths share, the run-up is invisible
/// to everything downstream: `dist` reads 0 at the marker, so the tapers, the
/// `drain` falloff and the dynamics loop all measure the stroke from where the
/// press happened, exactly as they measured it when the curve began there.
pub(super) fn generate_segments_in(
    rec: &StrokeRecord,
    tol: crate::path::FlattenTolerance,
    spans: StrokeSpans,
) -> (Vec<Segment>, f32) {
    let b = &rec.brush;
    let dist0 = spans.dist;
    let reaches_end = spans.range.end >= crate::path::span_count(rec.path.len());
    let pts = crate::path::flatten_spans_from(&rec.path, rec.start, spans.range, dist0, tol);
    let end_dist = pts.last().map_or(dist0, |p| p.dist);
    // One segment per flattened edge at least; the taper only ever cuts an edge into
    // more. Reserved rather than grown from nothing because a commit's stroke runs to
    // thousands of these, each ~100 bytes, on the interactive path.
    let mut segs = Vec::with_capacity(pts.len().saturating_sub(1));
    if pts.is_empty() {
        return (segs, end_dist);
    }
    let taper = Taper::resolve(b, reaches_end.then_some(end_dist));

    // The tip in force for one pen reading and one taper factor, in canvas px.
    //
    // The size mapping and the taper both scale the tip; the floor keeps a tapered tip
    // a hairline at its very point rather than a degenerate zero-width sweep (which
    // would also divide by zero in the dynamics loop's reservoir cadence, and would let
    // a segment's ramp reach the `|ramp| = 2` its positivity rests on). With the
    // default brush the mapping is pressure, linearly, so this is the product it has
    // always been — to the bit (`Modulation::factor`).
    //
    // `tap` is the taper's radius factor, which only the caller can know: it is
    // measured against the *whole* stroke and a partial range does not have one.
    let tip_at = |pen: PenState, tap: f32| (b.size * b.modulation.size(pen) * tap).max(0.5);

    // The ceiling's factor for one pen reading (§6.2). Read at the segment's two
    // **ends** rather than at its midpoint, where every rate below is read: a rate
    // is a per-segment quantity by nature — `add` scales what this segment lays —
    // where a ceiling caps what the whole stroke shows, so a value held constant
    // across a segment steps at every cut and the mark comes out in bands. The
    // shaders interpolate it across the sweep the way they interpolate the tip
    // ([`Paint::opacity_ramp`]).
    //
    // The effect is a fact about the *brush*, so it is resolved to its variant once
    // here and once for `rates` below, not per pen sample. The arms' closures are
    // temporaries the `let` extends to the end of the function; `move`, because a
    // borrow of the arm's binding would not be.
    let ceiling_at: &dyn Fn(PenState) -> f32 = match &b.effect {
        stark_model::document::BrushEffect::Paint(p) => {
            &move |pen: PenState| p.modulation.opacity(pen)
        }
        stark_model::document::BrushEffect::Wet(w) => {
            &move |pen: PenState| w.modulation.opacity(pen)
        }
        stark_model::document::BrushEffect::Erase(e) => {
            &move |pen: PenState| e.modulation.opacity(pen)
        }
        // A warp has no ceiling for the pen to drive (`BrushEffect::opacity`).
        stark_model::document::BrushEffect::Liquify(_) => &|_: PenState| 1.0,
    };

    // The rates are the effect's own, at its own pen mappings. Paint and erase have
    // one rate each — paint's `add` is its flow, an eraser's the rate its bite
    // builds at (§6.12) — and only a wet brush carries fluxes; which is a statement
    // about the *brush*, so every segment of every stroke answers it the same way
    // (`dynamics_setup`'s purity argument).
    //
    // A wet brush's flow scales everything the tool does (§6.2), and all of it is
    // scaled here. What is linear in exposure takes the factor outright — the `add`
    // mint, and the `bleed` diffusivity (which `bleed_stencil` still clamps at its
    // own calibrated top, so a hot flow saturates the blur rather than out-reaching
    // the stencil). The vertical fractions cannot be scaled before their `ln`, so
    // they become λs first and the factor lands on the exponent — one pass at flow
    // f trades exactly what f passes at flow 1 would.
    let rates: &dyn Fn(PenState) -> Rates = match &b.effect {
        stark_model::document::BrushEffect::Paint(p) => &move |pen: PenState| Rates {
            add: p.flow * p.modulation.flow(pen),
            ..Rates::ZERO
        },
        stark_model::document::BrushEffect::Wet(w) => &move |pen: PenState| {
            let flow = w.flow * w.modulation.flow(pen);
            Rates {
                add: w.dynamics.add * w.modulation.add(pen) * flow,
                lambda_lift: flow * lambda(w.dynamics.lift * w.modulation.lift(pen)),
                lambda_deposit: flow * lambda(w.dynamics.deposit * w.modulation.deposit(pen)),
                bleed: w.dynamics.bleed * w.modulation.bleed(pen) * flow,
                drag: 0.0,
            }
        },
        stark_model::document::BrushEffect::Erase(e) => &move |pen: PenState| Rates {
            add: e.flow * e.modulation.flow(pen),
            ..Rates::ZERO
        },
        stark_model::document::BrushEffect::Liquify(l) => &move |pen: PenState| Rates {
            drag: l.strength * l.modulation.strength(pen),
            ..Rates::ZERO
        },
    };

    // `ends` is what is in force at the segment's two ends — the tip, where the radius
    // *ramp* comes from ([`Sweep::radius_ramp`]), and the pen, where the ceiling's does
    // ([`Paint::opacity_ramp`]). Everything else is sampled at the midpoint, `at`: the
    // rates below are applied per segment and the midpoint is the reading whose error
    // is second order where either end's would be first.
    let make = |at: At, track: Track, ends: Ends| {
        // The pen as the modulations read it, at this segment's own attributes
        // (§6.2). `Modulation::factor` clamps anyway.
        let pen = PenState {
            pressure: at.pressure,
            tilt: at.tilt.length(),
        };
        let m = &b.modulation;
        let (r0, r1) = ends.tip;
        let (o0, o1) = (ceiling_at(ends.pen.0), ceiling_at(ends.pen.1));
        // The mean rather than the midpoint *sample*, and that is what makes the ramp
        // exact at both ends: `radius·(1 ± ramp/2)` is then `r1` and `r0` themselves,
        // so two adjacent segments — which computed the tip at their shared knot from
        // the same pen and the same taper — agree on it to the bit, and the outline
        // has no step to alias. A midpoint sample would miss both ends by the size
        // mapping's own curvature, which is exactly the C⁰ break being removed.
        //
        // The two coincide, bit for bit, wherever the tip does not change — which is
        // every segment of an untapered brush at constant pressure.
        let radius = (r0 + r1) * 0.5;
        // The facing axis this segment's stretch runs along is the one its orientation
        // already names, so the two are solved together and the volume slice the
        // shaders read is the sum (`Stretch`).
        let orient = orientation_turns(b.orientation, track.mid_dir, at.tilt);
        let elong = BrushParams::elongation(b.stretch * m.stretch(pen));
        let stretch = Stretch::solve(elong, orient);
        let mut sweep = Sweep {
            start: at.pos,
            dir: track.dir,
            curvature: track.curvature,
            radius,
            radius_ramp: (r1 - r0) / radius,
            // Filled from `widest_tip` below, which needs the ramp this initializer is
            // still building — and must be *that* expression rather than one equal to
            // it, since this bounds the strip the GPU draws.
            reach: 0.0,
            length: track.length,
            orient: (orient + stretch.turns).rem_euclid(1.0),
            stretch,
            dist: track.dist,
        };
        // **The tip's own disc, drawn out by the stretch.** `A` maps the disc every
        // shape's paint lies inside ([`Sweep::reach`]) into a region no point of which
        // is further out than `‖A‖ = elongation` times where it started, so one factor
        // bounds every angle — which is what this has to be, since it grows an
        // axis-aligned box (`coverage_bounds`).
        //
        // Plus the antialiasing rim: the pixel footprint's box filter deposits up to
        // half a px past the disc (§6.2), and the strip the GPU rasterizes carries
        // `AA_RIM_PX` of margin for it (`stamp_common::sweep_vertex`) — so the box
        // must too, or the rim px of a tile another tile's apron rewrites would be
        // drawn by one and skipped by the other (§6.4).
        sweep.reach = sweep.widest_tip() * elong + stark_shaders::mirror::stamp_common::AA_RIM_PX;
        // **Which of the rates ride the ends, and which the midpoint.** A rate the
        // shaders apply per *texel* — `add`, the liquify follow, and the tooth's
        // give below — is read at both ends and interpolated between them, for
        // [`Paint::add_ramp`]'s reason. The three the *exchange* solves with are
        // read at the midpoint and stay constant across the segment, because that
        // solve is one problem per dispatch whose two halves are complements and
        // whose tool half has no canvas position to vary along (`dynamics.wesl`).
        // What holds their step down is the flattener, which already buys segments
        // against [`BrushParams::max_slope`](stark_model::document::BrushParams::max_slope).
        let mid = rates(pen);
        let rate0 = rates(ends.pen.0);
        let rate1 = rates(ends.pen.1);
        let give0 = b.tooth.give * m.tooth_give(ends.pen.0);
        let give1 = b.tooth.give * m.tooth_give(ends.pen.1);
        Segment {
            sweep,
            paint: Paint {
                // The mean of the two ends, like the radius and the ceiling: it is
                // what makes the ramp exact at both, so two adjacent segments —
                // which read the pen at the knot they share from the same sample —
                // agree there to the bit.
                add: (rate0.add + rate1.add) * 0.5,
                add_ramp: rate1.add - rate0.add,
                lambda_lift: mid.lambda_lift,
                lambda_deposit: mid.lambda_deposit,
                bleed: mid.bleed,
                drag: (rate0.drag + rate1.drag) * 0.5,
                drag_ramp: rate1.drag - rate0.drag,
                tooth_give: (give0 + give1) * 0.5,
                tooth_give_ramp: give1 - give0,
                opacity: (o0 + o1) * 0.5,
                opacity_ramp: o1 - o0,
            },
        }
    };

    for w in pts.windows(2) {
        let (a, c) = (w[0], w[1]);
        let v = c.pos - a.pos;
        let chord = v.length();
        if chord < 1e-5 {
            continue;
        }
        // The edge as an arc rather than a chord (see [`segment_arc`]): same
        // endpoints, but leaving along the curve's own tangent, so the swept outline
        // does not break its curvature at every joint. Curvature 0 comes back for a
        // straight or barely-curved edge, and everything below reduces to the chord
        // case exactly.
        let crate::path::Arc {
            dir,
            curvature: kappa,
            length: len,
        } = crate::path::fit_arc(a.vel, v, tol.max_arc_curvature);
        // One flattened edge is one segment wherever the taper is flat — which is
        // everywhere on an untapered brush, so nothing below changes those strokes
        // by a bit. Inside a taper it is cut into pieces fine enough that the radius
        // steps smoothly, the same length bound `drain` and the reservoir cadence
        // ask of the *fitter* (`flatten_tolerance`), except paid only near the ends
        // instead of over the whole stroke. The pieces are sub-*arcs*: they inherit
        // the edge's curvature and are stepped along it, so cutting an edge up still
        // traces exactly the same centreline.
        let n = taper.pieces(a.dist, len);
        let step = len / n as f32;
        // The pen, linearly across the flattened edge — the interpolation the
        // *flattener* priced (`FlattenTolerance::attribute`), read here at whichever
        // fraction of the edge is wanted.
        let pen_at = |u: f32| {
            (
                a.pressure + (c.pressure - a.pressure) * u,
                a.tilt + (c.tilt - a.tilt) * u,
            )
        };
        // …as the modulations read it. Both quantities that ramp across a segment
        // are functions of this alone, so they are pinned to one sample per end
        // rather than each taking its own.
        let pen_state = |(pressure, tilt): (f32, Vec2)| PenState {
            pressure,
            tilt: Vec2::length(tilt),
        };
        for k in 0..n {
            let (u0, u1) = (k as f32 / n as f32, (k + 1) as f32 / n as f32);
            let (pressure, tilt) = pen_at((u0 + u1) * 0.5);
            let along = step * k as f32;
            let dist = a.dist + along;
            let (pos, tan) = crate::path::arc_at(a.pos, dir, kappa, along);
            let (_, mid_tan) = crate::path::arc_at(a.pos, dir, kappa, along + step * 0.5);
            // The tip at the piece's two ends, from the pen and the taper *there*.
            // Both are functions of arc length alone, so the shared end of two
            // adjacent pieces — and of two adjacent flattened edges, where `u1` of one
            // is `u0` of the next at the same `dist` — resolves to the same number on
            // both sides. That agreement is what makes the outline continuous
            // ([`Sweep::radius_ramp`]); it is not approached, it is the same expression
            // evaluated twice.
            let (pen0, pen1) = (pen_state(pen_at(u0)), pen_state(pen_at(u1)));
            let ends = Ends {
                tip: (
                    tip_at(pen0, taper.factor(dist)),
                    tip_at(pen1, taper.factor(dist + step)),
                ),
                pen: (pen0, pen1),
            };
            segs.push(make(
                At {
                    pos,
                    pressure,
                    tilt,
                },
                Track {
                    dir: tan,
                    mid_dir: mid_tan,
                    curvature: kappa,
                    length: step,
                    dist,
                },
                ends,
            ));
        }
    }

    (segs, end_dist)
}

/// The shape's orientation for a segment, as a fraction of a full turn ∈ [0, 1): the
/// relative angle between the shape's native axis and the travel direction `dir`, which
/// picks the prefix-τ orientation layer (§6.6).
///
/// - [`OrientationSource::FollowStroke`]: the shape tracks the tangent, so the relative
///   angle is always 0 (the historical behaviour; for a round tip it is moot anyway).
/// - [`OrientationSource::Pen`]: the shape is pinned to the pen's azimuth (the tilt
///   direction) in canvas space, so relative to the travel direction it is `α − φ` — as
///   the stroke curves the extent angle stays fixed in the world, like a nib.
pub(super) fn orientation_turns(source: OrientationSource, dir: Vec2, tilt: Vec2) -> f32 {
    match source {
        OrientationSource::FollowStroke => 0.0,
        OrientationSource::Pen => {
            let alpha = tilt.y.atan2(tilt.x); // pen azimuth (0 when the pen is upright / mouse)
            let phi = dir.y.atan2(dir.x); // travel direction
            ((alpha - phi) / std::f32::consts::TAU).rem_euclid(1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::budget::{MAX_TIP_TURN, flatten_tolerance};
    use super::super::region::{coverage_bounds, segment_end};
    use super::testing::{
        assert_outline_is_continuous, record, smearing, tapered_record, whole, whole_segments,
    };
    use super::*;
    use stark_model::document::BrushShape;
    // --- what varies across a segment ------------------------------------

    /// **Every pen target the shaders read per texel is continuous at the knots
    /// two segments share** (§6.2).
    ///
    /// This is the whole content of the ramps, stated where it can be checked
    /// exactly rather than looked for in pixels. A value carried as one number per
    /// segment steps at every cut, and a stroke at the rate a hand reports is a
    /// handful of segments wide — which is how a modulated `opacity` came to draw
    /// bands and a modulated radius a comb of sawteeth. Carried as a mean and a
    /// difference, the value at a segment's end is the same expression as the value
    /// at the next one's start: both read the pen at that knot, from the same
    /// sample of the same fitted curve.
    ///
    /// Held to a rounding rather than to the bit, and the gap is the *carrying*
    /// rather than the values: a mean and a difference are each one rounding of the
    /// pair, so reconstructing an end returns it to within an ulp instead of
    /// exactly. What that leaves this free to catch is a **step**, which is the
    /// pen's own change between two knots — five orders of magnitude above the
    /// bound below, and what every one of these targets used to have.
    ///
    /// The three the exchange solves with are deliberately absent: `lift`,
    /// `deposit` and `bleed` are one rate per dispatch (see [`Paint`]), and what
    /// bounds their step is the flattener's attribute budget.
    #[test]
    fn every_per_texel_target_is_continuous_across_a_knot() {
        // A pen that moves through the whole range, so no mapping sits still.
        let pts: Vec<stark_model::path::ControlPoint> = (0..12)
            .map(|i| {
                let t = i as f32 / 11.0;
                stark_model::path::ControlPoint {
                    pos: Vec2::new(-120.0 + 240.0 * t, 30.0 * (t * 5.0).sin()),
                    pressure: 0.05 + 0.9 * t,
                    tilt: Vec2::new(0.8 * (1.0 - t), 0.0),
                    time: t,
                }
            })
            .collect();
        let mapped = |source| {
            Some(stark_model::document::Modulation {
                source,
                floor: 0.1,
                curve: 0.4,
            })
        };
        use stark_model::document::ModSource::{Pressure, Tilt};

        // Every effect, so every target that has a ramp is exercised by the one
        // check — including the two that only one effect has.
        let mut wet = BrushParams {
            size: 40.0,
            ..BrushParams::default()
        };
        wet.modulation.size = mapped(Pressure);
        wet.modulation.tooth_give = mapped(Tilt);
        wet.tooth.give = 0.6;
        {
            let w = wet.make_wet();
            w.modulation.flow = mapped(Pressure);
            w.modulation.add = mapped(Tilt);
            w.modulation.opacity = mapped(Pressure);
        }
        let mut liquify = BrushParams {
            size: 40.0,
            effect: stark_model::document::BrushEffect::Liquify(
                stark_model::document::LiquifyEffect {
                    strength: 0.9,
                    modulation: stark_model::document::LiquifyModulations {
                        strength: mapped(Pressure),
                    },
                },
            ),
            ..BrushParams::default()
        };
        liquify.modulation.size = mapped(Tilt);

        for brush in [wet, liquify] {
            let rec = StrokeRecord {
                layer: stark_model::document::LayerId::ROOT,
                brush,
                path: pts.clone(),
                seed: 0,
                start: 0.0,
                translation: stark_model::geom::IVec2::ZERO,
            };
            let tol = flatten_tolerance(&rec.brush);
            let (segs, _) = generate_segments_in(&rec, tol, StrokeSpans::whole(&rec));
            assert!(segs.len() > 4, "the fixture must cut into several segments");

            for (i, pair) in segs.windows(2).enumerate() {
                let (a, b) = (&pair[0], &pair[1]);
                // Each target at the end of `a` and at the start of `b`.
                let ends = [
                    (
                        "radius",
                        a.sweep.radius * (1.0 + 0.5 * a.sweep.radius_ramp),
                        b.sweep.radius * (1.0 - 0.5 * b.sweep.radius_ramp),
                    ),
                    (
                        "add",
                        a.paint.add + 0.5 * a.paint.add_ramp,
                        b.paint.add - 0.5 * b.paint.add_ramp,
                    ),
                    (
                        "tooth_give",
                        a.paint.tooth_give + 0.5 * a.paint.tooth_give_ramp,
                        b.paint.tooth_give - 0.5 * b.paint.tooth_give_ramp,
                    ),
                    (
                        "opacity",
                        a.paint.opacity + 0.5 * a.paint.opacity_ramp,
                        b.paint.opacity - 0.5 * b.paint.opacity_ramp,
                    ),
                    (
                        "drag",
                        a.paint.drag + 0.5 * a.paint.drag_ramp,
                        b.paint.drag - 0.5 * b.paint.drag_ramp,
                    ),
                ];
                for (what, left, right) in ends {
                    let slack = 1e-6 * left.abs().max(1.0);
                    assert!(
                        (left - right).abs() <= slack,
                        "{what} steps at the knot between segments {i} and {}: \
                         {left} then {right}",
                        i + 1,
                    );
                }
            }
        }
    }

    /// **The taper was only half the problem.** `size` follows the pen, so a pressure
    /// ramp scales the tip with distance travelled exactly as a taper does — and the
    /// flattener's `attribute` bound is a step in *pressure*, which on a big brush is
    /// a large step in px: 0.91 px of radius at 46, 2.4 px at 120, 9.9 px at 500
    /// (measured 2026-08-14). The corpus's own `pressure_ramp` case is radius 46, so
    /// the suite could not see it.
    ///
    /// The ramp fixes both at once, and that is the point of fixing it *there*: it is
    /// not a taper feature, it is the statement that a segment's tip is a function of
    /// travel rather than a value. No second rule was added for the pen.
    #[test]
    fn a_pressure_ramp_on_a_huge_brush_has_no_step_either() {
        let path: Vec<stark_model::path::ControlPoint> = (0..=24)
            .map(|i| {
                let t = i as f32 / 24.0;
                let mut cp = stark_model::path::ControlPoint::at(Vec2::new(t * 4000.0, 0.0));
                cp.pressure = 0.05 + 0.95 * t;
                cp
            })
            .collect();
        let rec = StrokeRecord {
            layer: stark_model::document::LayerId::ROOT,
            brush: BrushParams {
                size: 500.0,
                drain: 0.0,
                ..BrushParams::default()
            },
            path,
            seed: 0,
            start: 0.0,
            translation: stark_model::geom::IVec2::ZERO,
        };
        let segs = whole(&rec);
        assert_outline_is_continuous(&segs);
        // The tip really does sweep the range the pen asked for — otherwise the
        // continuity above would be the trivial one a constant tip has.
        let widest = segs.iter().fold(0.0f32, |m, s| m.max(s.radius));
        let narrowest = segs.iter().fold(f32::MAX, |m, s| m.min(s.radius));
        assert!(
            widest > 400.0 && narrowest < 60.0,
            "the pen drove the tip from {narrowest} to {widest}, which is not a ramp",
        );
    }

    /// A tip that does not change carries **no** ramp, exactly — which is what makes
    /// every stroke that came before this change render as it did, to the bit: the
    /// shaders branch on that zero (`stamp_common::radius_ramp_scale`), and a zero that were
    /// merely small would take the general path and round differently.
    #[test]
    fn a_tip_that_holds_still_carries_no_ramp() {
        // No taper, no size modulation, full pressure throughout.
        let mut rec = tapered_record(40.0, 0.0, 0.0, 900.0);
        rec.brush.modulation = stark_model::document::BrushModulations::default();
        for s in whole(&rec) {
            assert_eq!(s.radius_ramp, 0.0, "an unvarying tip picked up a ramp");
            assert_eq!(s.radius, 40.0, "an unvarying tip changed size");
        }
    }

    /// A stroke sweeps exactly its own travel, however short. The retired
    /// `DAB_TRAVEL` dwell used to top every stroke up to 0.6 radii — a fabricated
    /// minimum that overrode precisely the entry geometry the fit now carries
    /// honestly (§6.2's run-up). A short flick is short; the deposit says so.
    #[test]
    fn a_short_stroke_sweeps_only_its_own_travel() {
        let radius = 20.0;
        for len in [0.5f32, 2.0, 6.0, 11.5, 12.5, 40.0] {
            let rec = tapered_record(radius, 0.0, 0.0, len);
            let travel: f32 = whole(&rec).iter().map(|s| s.length).sum();
            assert!(
                (travel - len).abs() < 0.05,
                "a {len}px stroke swept {travel}"
            );
        }
    }

    // --- swept arcs -------------------------------------------------------

    /// A stroke bending through `sweep` radians of a circle of radius `curve_radius`.
    fn curved_record(radius: f32, curve_radius: f32, sweep: f32) -> StrokeRecord {
        let path: Vec<stark_model::path::ControlPoint> = (0..=12)
            .map(|i| {
                let t = i as f32 / 12.0 * sweep;
                stark_model::path::ControlPoint::at(Vec2::new(
                    curve_radius * t.sin(),
                    curve_radius * (1.0 - t.cos()),
                ))
            })
            .collect();
        StrokeRecord {
            layer: stark_model::document::LayerId::ROOT,
            brush: BrushParams {
                size: radius,
                drain: 0.0,
                ..BrushParams::default()
            },
            path,
            seed: 0,
            start: 0.0,
            translation: stark_model::geom::IVec2::ZERO,
        }
    }

    /// Densely sampled points of the true curve — the ground truth the two stand-ins
    /// below are measured against. Fifty times tighter than the render budget, so its
    /// own flattening error is nowhere near what is being compared.
    fn dense(rec: &StrokeRecord) -> Vec<Vec2> {
        let tol = flatten_tolerance(&rec.brush).relaxed(0.02);
        crate::path::flatten(&rec.path, tol)
            .into_iter()
            .map(|s| s.pos)
            .collect()
    }

    /// The largest distance from any point of the true curve to `poly`.
    fn deviation(curve: &[Vec2], poly: &[Vec2]) -> f32 {
        let to_seg = |p: Vec2, a: Vec2, b: Vec2| {
            let ab = b - a;
            let len2 = ab.length_squared();
            let t = if len2 < 1e-12 {
                0.0
            } else {
                ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
            };
            (p - (a + ab * t)).length()
        };
        curve
            .iter()
            .map(|&p| {
                poly.windows(2)
                    .map(|w| to_seg(p, w[0], w[1]))
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0, f32::max)
    }

    /// `fit_arc` on a genuine circular arc recovers the circle it came from — the
    /// case the whole construction is pinned to, since a flattened edge of a smooth
    /// stroke is one to second order.
    #[test]
    fn the_arc_fit_recovers_a_circle() {
        for (r, theta) in [(50.0f64, 0.08f64), (300.0, 0.05), (900.0, 0.02)] {
            // An arc of radius `r` turning `theta`, starting at the origin heading +x.
            // Built in f64: the far endpoint's lateral offset is `r(1 − cos θ)`, and
            // forming that in f32 would cancel away enough digits to swamp the
            // tolerances below — an artifact of the test's own construction, not of
            // the fit it is checking.
            let start_dir = Vec2::new(1.0, 0.0);
            let end = Vec2::new((r * theta.sin()) as f32, (r * (1.0 - theta.cos())) as f32);
            let (r, theta) = (r as f32, theta as f32);
            let crate::path::Arc {
                dir,
                curvature: kappa,
                length: len,
            } = crate::path::fit_arc(start_dir, end, f32::INFINITY);
            assert!(kappa != 0.0, "r={r} θ={theta}: fitted straight");
            assert!(
                (kappa - 1.0 / r).abs() < 1e-4 / r,
                "r={r}: curvature {kappa} is not 1/{r}"
            );
            assert!(
                (len - r * theta).abs() < 1e-3 * len,
                "r={r}: arc length {len} is not {}",
                r * theta
            );
            // And walking it lands exactly on the far end, which is what makes
            // consecutive segments meet.
            let (landed, _) = crate::path::arc_at(Vec2::ZERO, dir, kappa, len);
            assert!(
                (landed - end).length() < 1e-3,
                "r={r}: the arc ends at {landed:?}, not {end:?}"
            );
        }
    }

    /// The claim the change exists for: swept **arcs** track the fitted curve far
    /// more closely than the chords they replace, at the same segment count.
    ///
    /// The chord's error is the flattener's positional budget by construction — that
    /// is what the budget *is* — so this is really a statement about what a segment
    /// can be asked to do without being made shorter. Measured on the curves below,
    /// the arcs land ~4× closer; the residual is the fitted spline's own curvature
    /// *variation* across a segment, which a single arc cannot follow and which the
    /// flattener's `angle` bound is what actually limits.
    ///
    /// The ratio understates the visible gain, because the amplitude is not what the
    /// eye is picking up: a chord sweep breaks the outline's curvature at every joint
    /// and creases it on the inside of a turn, and an arc sweep does neither. That is
    /// what facets are, and it is not something a distance metric sees.
    #[test]
    fn arcs_track_the_curve_far_closer_than_chords() {
        for curve_radius in [200.0f32, 600.0, 2000.0] {
            let rec = curved_record(8.0, curve_radius, 1.2);
            let segs = whole(&rec);
            let curve = dense(&rec);
            assert!(
                segs.iter().any(|s| s.curvature != 0.0),
                "r={curve_radius}: nothing was bent at all"
            );

            let chords: Vec<Vec2> = segs
                .iter()
                .map(|s| s.start)
                .chain(segs.last().map(segment_end))
                .collect();
            // Each arc sampled finely, so a point-to-polyline distance measures the
            // arc itself rather than its own chord.
            let mut arcs = Vec::new();
            for s in &segs {
                for i in 0..16 {
                    arcs.push(
                        crate::path::arc_at(
                            s.start,
                            s.dir,
                            s.curvature,
                            s.length * i as f32 / 16.0,
                        )
                        .0,
                    );
                }
            }
            arcs.extend(segs.last().map(segment_end));

            let chord_err = deviation(&curve, &chords);
            let arc_err = deviation(&curve, &arcs);
            assert!(
                arc_err < 0.35 * chord_err,
                "r={curve_radius}: arcs are off by {arc_err}, chords by {chord_err}"
            );
        }
    }

    /// Consecutive segments meet: each one's arc *ends* where the next one starts.
    /// Nothing in the deposit re-derives a segment's end — the shaders sweep from
    /// `start` along the arc for `length` — so a gap here would be a seam of missing
    /// paint at every joint, and an overlap a double deposit.
    #[test]
    fn segments_meet_end_to_start_along_their_arcs() {
        let rec = curved_record(12.0, 400.0, 1.5);
        let segs = whole(&rec);
        assert!(segs.len() > 4, "not enough segments to join up");
        for (i, w) in segs.windows(2).enumerate() {
            let gap = (segment_end(&w[0]) - w[1].start).length();
            assert!(
                gap < 1e-2,
                "segment {i} ends {gap}px from where {} starts",
                i + 1
            );
        }
    }

    /// A straight stroke is not bent, and the arc machinery leaves it on exactly the
    /// floats it was on before — the same no-op guarantee the taper's subdivision has.
    #[test]
    fn a_straight_stroke_is_never_bent() {
        let rec = tapered_record(18.0, 0.0, 0.0, 900.0);
        let segs = whole(&rec);
        assert!(
            segs.iter().all(|s| s.curvature == 0.0),
            "a straight stroke picked up curvature"
        );
        for s in &segs {
            assert_eq!(
                segment_end(s),
                s.start + s.dir * s.length,
                "a straight segment's end moved off its chord"
            );
        }
    }

    /// A tip too fat for the turn it is sweeping falls back to a straight segment.
    ///
    /// The bound exists because both shaders sweep a curved segment by unrolling the
    /// annulus about its centre of curvature, and that approximation degrades as the
    /// tip grows against the curve's own radius — see
    /// [`MAX_TIP_TURN`](super::super::budget::MAX_TIP_TURN).
    #[test]
    fn a_fat_tip_on_a_tight_turn_sweeps_straight() {
        let curve_radius = 60.0;
        let fat = 50.0;
        for s in whole(&curved_record(fat, curve_radius, 1.5)) {
            assert!(
                s.radius * s.curvature.abs() <= MAX_TIP_TURN,
                "a segment sweeps an arc of radius {} under a {} tip",
                1.0 / s.curvature.abs(),
                s.radius
            );
        }
        // And the curve really is tight enough for that to have bitten: under a fine
        // tip the same path keeps curvature the fat one had to give up. Without this
        // the assertion above would pass on any straight line.
        let fine = whole(&curved_record(2.0, curve_radius, 1.5));
        assert!(
            fine.iter().any(|s| fat * s.curvature.abs() > MAX_TIP_TURN),
            "the test curve is too gentle to exercise the guard"
        );
    }

    /// The flattener and the segment generator agree, edge for edge, on whether a
    /// piece of curve is swept as an arc or as a chord.
    ///
    /// This is what makes the positional budget mean anything. `path::within` prices
    /// an edge against whatever `fit_arc` returns for it, and the sweep is built from
    /// whatever `fit_arc` returns for it — so if the two ever called it with different
    /// caps, an edge could be *measured* as a well-tracked arc and then *drawn* as a
    /// chord that misses the curve by several times the allowance. Routing both through
    /// one function with one cap is what rules that out; this pins that they do.
    #[test]
    fn the_flattener_and_the_sweep_agree_on_which_edges_bend() {
        for radius in [2.0f32, 18.0, 50.0, 120.0] {
            for curve_radius in [80.0f32, 300.0, 1200.0, 5000.0] {
                let rec = curved_record(radius, curve_radius, 1.4);
                let tol = flatten_tolerance(&rec.brush);
                let pts = crate::path::flatten(&rec.path, tol);
                let segs = whole(&rec);
                // Untapered, so it is one segment per flattened edge — except the
                // degenerate ones the generator drops, which the clamped end condition
                // always produces a few of (its outermost spans are squashed to nearly
                // nothing).
                let edges: Vec<_> = pts
                    .windows(2)
                    .filter(|w| (w[1].pos - w[0].pos).length() >= 1e-5)
                    .collect();
                assert_eq!(segs.len(), edges.len(), "r={radius} R={curve_radius}");
                for (i, (w, s)) in edges.iter().zip(&segs).enumerate() {
                    let want =
                        crate::path::fit_arc(w[0].vel, w[1].pos - w[0].pos, tol.max_arc_curvature);
                    assert_eq!(
                        want.curvature, s.curvature,
                        "r={radius} R={curve_radius} edge {i}: the flattener priced \
                         curvature {} and the sweep drew {}",
                        want.curvature, s.curvature
                    );
                }
                // And the cap really is enforced, not merely never reached.
                assert!(
                    segs.iter()
                        .all(|s| s.curvature.abs() <= tol.max_arc_curvature + 1e-9),
                    "r={radius} R={curve_radius}: a segment bends past the sweepable cap"
                );
            }
        }
    }

    /// Every box a segment is measured by contains its whole arc, not just its two
    /// ends. Under-reporting here is a clipped stroke: `region::cover` would leave a
    /// tile out of the render, and the dynamics loop would dispatch a rect too small
    /// for its own extent.
    #[test]
    fn the_coverage_box_contains_the_whole_arc() {
        let rec = curved_record(10.0, 150.0, 2.4);
        let segs = whole(&rec);
        assert!(segs.iter().any(|s| s.curvature != 0.0), "nothing bent");
        for (i, s) in segs.iter().enumerate() {
            let (lo, hi) = coverage_bounds(s);
            for k in 0..=32 {
                let (p, _) =
                    crate::path::arc_at(s.start, s.dir, s.curvature, s.length * k as f32 / 32.0);
                // Every point of the arc, plus the tip riding along it.
                let r = Vec2::splat(s.radius);
                assert!(
                    (p - r).x >= lo.x
                        && (p - r).y >= lo.y
                        && (p + r).x <= hi.x
                        && (p + r).y <= hi.y,
                    "segment {i}: the arc escapes its own coverage box at {p:?}"
                );
            }
        }
    }

    /// A stamp's box holds its **disc** exactly — no more, at any angle.
    ///
    /// Nothing any canonical shape can paint lies outside the disc inscribed in its
    /// mask square ([`Sweep::reach`]), so the disc's rim is the exact frontier the
    /// box must contain, and the box that contains only it is the tight one: a `√2`
    /// margin for a corner a canonical mask cannot occupy was every stamp stroke
    /// paying up to double the region area for texels its prefix taps difference to
    /// zero. Containment failing clips the stroke at a tile boundary;
    /// tightness failing is the tax coming back.
    ///
    /// Swept at a range of angles because axis-aligned travel is where a wrong bound
    /// and the right one agree, and those were the strokes that always looked right.
    #[test]
    fn a_stamps_box_is_its_disc_exactly() {
        let radius = 24.0f32;
        let stamp = BrushShape::Stamp(stark_model::AssetId([7u8; 32]));
        for k in 0..16 {
            // Angles off the axes as well as on them, from the series (see
            // `sin_series`) so the case is the same on every platform.
            let theta = k as f64 * std::f64::consts::TAU / 16.0;
            let dir = Vec2::new(cos_series(theta) as f32, sin_series(theta) as f32);
            let mut b = BrushParams {
                size: radius,
                ..BrushParams::default()
            };
            b.shape = stamp;
            let rec = record(b, &[Vec2::ZERO, dir * 200.0]);
            for (i, s) in whole(&rec).iter().enumerate() {
                let (lo, hi) = coverage_bounds(s);
                let end = segment_end(s);
                // Containment: the rim of the tip's disc at each end of the travel,
                // every point of it a texel the deposit may reach. A tolerance well
                // under a texel is what says "contains" here; nothing downstream can
                // resolve less, and both consumers add their own margin
                // (`TILE_APRON`, `RECT_MARGIN`) on top.
                const SLACK: f32 = 1e-3;
                for base in [s.start, end] {
                    for j in 0..16 {
                        let phi = j as f64 * std::f64::consts::TAU / 16.0;
                        let rim = base
                            + Vec2::new(cos_series(phi) as f32, sin_series(phi) as f32) * s.radius;
                        assert!(
                            rim.x >= lo.x - SLACK
                                && rim.y >= lo.y - SLACK
                                && rim.x <= hi.x + SLACK
                                && rim.y <= hi.y + SLACK,
                            "angle {k}/16, segment {i}: the disc's rim {rim:?} falls \
                             outside its own coverage box {lo:?}..{hi:?}",
                        );
                    }
                }
                // Tightness: the box is the travel's own, grown by one radius plus
                // the antialiasing rim ([`Sweep::reach`]) — not by `√2` of one.
                let rim = stark_shaders::mirror::stamp_common::AA_RIM_PX;
                let tight_lo = s.start.min(end) - Vec2::splat(s.radius + rim);
                let tight_hi = s.start.max(end) + Vec2::splat(s.radius + rim);
                assert!(
                    (lo - tight_lo).length() <= SLACK && (hi - tight_hi).length() <= SLACK,
                    "angle {k}/16, segment {i}: box {lo:?}..{hi:?} is not the tight \
                     {tight_lo:?}..{tight_hi:?}",
                );
            }
        }
    }

    // --- segment budget ----------------------------------------------------

    /// `sin` and `cos` from their Maclaurin series in plain f64 arithmetic.
    ///
    /// Not for accuracy — the curves below only have to be representative shapes. The
    /// library versions are not specified to the last bit and may differ between
    /// platforms, and these decide *control points*, so a knot differing by an ulp
    /// could flip a subdivision decision and fail this test on someone else's machine.
    /// Basic IEEE arithmetic is exactly specified, which rules that out rather than
    /// hoping — the same argument that makes `taper::taper_profile` a polynomial (§12.1).
    fn sin_series(x: f64) -> f64 {
        let (x2, mut term, mut acc) = (x * x, x, x);
        for k in 1..10 {
            term *= -x2 / (((2 * k) * (2 * k + 1)) as f64);
            acc += term;
        }
        acc
    }

    fn cos_series(x: f64) -> f64 {
        let (x2, mut term, mut acc) = (x * x, 1.0, 1.0);
        for k in 1..10 {
            term *= -x2 / (((2 * k - 1) * (2 * k)) as f64);
            acc += term;
        }
        acc
    }

    /// `n + 1` knots along a curve given by its **heading** — the tangent angle as a
    /// function of arc length, stepped into positions. Curvature is that function's
    /// derivative, which is what lets the curved cases below state their curvature
    /// directly instead of implying it through a parameterization.
    fn by_heading(n: usize, length: f64, theta: impl Fn(f64) -> f64) -> Vec<Vec2> {
        const STEPS_PER_KNOT: usize = 16;
        let ds = length / (n * STEPS_PER_KNOT) as f64;
        let (mut x, mut y) = (0.0f64, 0.0f64);
        let mut pts = vec![Vec2::new(0.0, 0.0)];
        for i in 0..n * STEPS_PER_KNOT {
            let t = theta((i as f64 + 0.5) * ds); // midpoint: symmetric about an inflection
            x += cos_series(t) * ds;
            y += sin_series(t) * ds;
            if (i + 1) % STEPS_PER_KNOT == 0 {
                pts.push(Vec2::new(x as f32, y as f32));
            }
        }
        pts
    }

    /// **How many segments the flattener spends on a stroke — pinned, on purpose.**
    ///
    /// This is a change-detector test and it is meant to be one. **Updating these
    /// numbers is a normal thing to do:** if a change moves them and you have decided
    /// the new geometry is right, paste in the new counts and say why in the commit.
    /// The test is not asserting that any particular number is correct — it is making
    /// sure a number cannot move *silently*, because nothing else here would notice.
    ///
    /// **A wet brush's flow scales the whole of what a segment resolves** (§6.2):
    /// what is linear in exposure — the mint and the bleed — takes the factor
    /// outright, the vertical fractions ride untouched beside the factor the
    /// plan puts on their λs, and every other effect resolves the neutral 1.
    /// This is the seam the flow/add split lives on: get it wrong and the Flow
    /// slider is back to being a paint knob on a blend brush.
    #[test]
    fn a_wet_flow_scales_the_rates_a_segment_resolves() {
        use stark_model::document::{BrushDynamics, BrushEffect};
        let wet = |flow: f32| {
            let mut rec = record(
                BrushParams {
                    size: 20.0,
                    effect: BrushEffect::wet_with(
                        [0.0; 3],
                        BrushDynamics {
                            add: 0.5,
                            lift: 0.6,
                            deposit: 0.4,
                            bleed: 0.2,
                            ..BrushDynamics::default()
                        },
                    ),
                    ..BrushParams::default()
                },
                &[Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)],
            );
            rec.brush.wet_mut().expect("a wet brush").flow = flow;
            whole_segments(&rec)[0].paint
        };
        let one = wet(1.0);
        assert_eq!(
            (one.add, one.lambda_lift, one.lambda_deposit, one.bleed),
            (0.5, lambda(0.6), lambda(0.4), 0.2),
            "at the neutral flow the rates are the axes' own, exactly",
        );
        let two = wet(2.0);
        assert_eq!(two.add, 1.0, "the mint is linear in exposure");
        assert_eq!(two.bleed, 0.4, "…and so is the diffusivity");
        assert_eq!(
            (two.lambda_lift, two.lambda_deposit),
            (2.0 * lambda(0.6), 2.0 * lambda(0.4)),
            "the flow lands on the λ after its ln, never on the axis before it",
        );
        // Flow 0 is a brush that does nothing at all: no mint, no bleed, and no
        // exchange.
        let zero = wet(0.0);
        assert_eq!(
            (zero.add, zero.lambda_lift, zero.lambda_deposit, zero.bleed),
            (0.0, 0.0, 0.0, 0.0),
        );
        // A plain paint brush's flow *is* its `add`, and it has no exchange for a
        // factor to reach.
        let plain = record(
            BrushParams {
                size: 20.0,
                ..BrushParams::default()
            },
            &[Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)],
        );
        let p = whole_segments(&plain)[0].paint;
        assert_eq!((p.add, p.lambda_lift, p.lambda_deposit), (0.6, 0.0, 0.0));
    }

    /// Segment count is the loop's unit of cost. Every dispatch in the dynamics path
    /// is charged per segment (`dynamics.wesl`), so the budgets below are the dial
    /// between quality and time, and they are set from five different quantities that
    /// have nothing to do with one another. A change to any one of them moves a stroke
    /// nobody was thinking about: the cases are chosen so that each is dominated by a
    /// *different* budget, and the one that moves tells you which.
    ///
    /// Every count is reported in one pass rather than failing at the first, so a
    /// deliberate retuning gives you the whole new table to paste in from one run.
    ///
    /// These are CPU-side and float-deterministic (§12.1) — the same reason replay and
    /// peers agree on geometry — so a count that differs *per machine* is a bug in that
    /// determinism, not a tolerance to loosen.
    /// The exchange budget means the same thing to every brush
    /// (`budget::flatten_budget`). These are properties of the rule, not
    /// measured counts — unlike the table below, a failure here is a bug rather than a
    /// retuning.
    #[test]
    fn the_exchange_budget_scales_with_the_transfer_rate() {
        use stark_model::document::BrushDynamics;
        let at = |lift: f32, deposit: f32, charge: f32| {
            flatten_tolerance(&BrushParams {
                size: 100.0,
                effect: stark_model::document::BrushEffect::wet_with(
                    [0.0; 3],
                    BrushDynamics {
                        lift,
                        deposit,
                        charge,
                        ..BrushDynamics::default()
                    },
                ),
                ..BrushParams::default()
            })
            .max_len
        };

        // The calibration point: `lift = deposit = 0.95` is quoted at exactly
        // `RESERVOIR_EXCHANGE_STEP`, which is what leaves the goldens that use it alone.
        assert!(
            (at(0.95, 0.95, 0.0) - 12.5).abs() < 0.05,
            "calibration moved: {}",
            at(0.95, 0.95, 0.0)
        );

        // Halving the rate doubles the travel: `−ln((1−a)(1−b))` is what the step is
        // inversely proportional to, so squaring the retained fractions halves it.
        // (0.95 → 0.9975 has half the rate of 0.95 → 0.95 per axis.)
        let slow = at(0.7775, 0.7775, 0.0); // (1−a)² = 0.05 ⇒ half the rate of 0.95
        assert!(
            (slow - 25.0).abs() < 0.2,
            "the step is not inverse in the rate: {slow}"
        );

        // Monotone in each axis on its own — more trading, shorter segments.
        assert!(at(0.9, 0.0, 0.0) < at(0.5, 0.0, 0.0));
        assert!(at(0.0, 0.9, 0.0) < at(0.0, 0.5, 0.0));

        // `charge` is a starting load, not a rate. A brush that only charges never
        // enters the exchange at all (`exchange_at`'s no-trading branch), so it is
        // bounded by the structural ceiling and nothing else.
        assert_eq!(at(0.0, 0.0, 1.0), 100.0);
        // …and it does not tighten a brush that *does* trade.
        assert!((at(0.95, 0.95, 1.0) - at(0.95, 0.95, 0.0)).abs() < f32::EPSILON);

        // A plain paint brush — no wet variant at all — is not capped by this at
        // all; a wet brush that trades nothing still runs the loop, so it keeps
        // the structural ceiling. The cap is the variant's now (§6.2).
        assert!(
            flatten_tolerance(&BrushParams {
                size: 100.0,
                ..BrushParams::default()
            })
            .max_len
                > 100.0
        );
        assert_eq!(at(0.0, 0.0, 0.0), 100.0);

        // Never a tightening: a brush that trades *faster* than the reference is left at
        // the reference step, so no setting pays more than it did before the scaling.
        assert!((at(1.0, 1.0, 0.0) - at(0.95, 0.95, 0.0)).abs() < 0.05);
        assert!(at(0.99, 0.99, 0.0) >= at(0.95, 0.95, 0.0) - 0.05);

        // The effect's flow multiplies both λs (§6.2), so the budget charges it as
        // part of the rate: half the flow earns double the travel, twice the flow
        // clamps at the reference floor rather than paying more, and a flow of
        // zero trades nothing and keeps the structural ceiling alone.
        let at_flow = |lift: f32, deposit: f32, flow: f32| {
            let mut b = BrushParams {
                size: 100.0,
                effect: stark_model::document::BrushEffect::wet_with(
                    [0.0; 3],
                    BrushDynamics {
                        lift,
                        deposit,
                        ..BrushDynamics::default()
                    },
                ),
                ..BrushParams::default()
            };
            b.wet_mut().expect("a wet brush").flow = flow;
            flatten_tolerance(&b).max_len
        };
        assert!((at_flow(0.7775, 0.7775, 0.5) - 50.0).abs() < 0.5);
        assert!((at_flow(0.7775, 0.7775, 2.0) - 12.5).abs() < 0.05);
        assert_eq!(at_flow(0.95, 0.95, 0.0), 100.0);
    }

    #[test]
    fn the_segment_budget_is_what_it_was() {
        // Three curves, shared across brushes so that a difference between two rows on
        // the same path is the brush's doing and nothing else. Each is 400px of arc.
        //
        // The tip radii below are 20 and 80, so `max_arc_curvature` (MAX_TIP_TURN /
        // radius) sits at 0.005 and 0.00125 respectively — the curvatures are picked
        // around those two thresholds.
        let straight = vec![Vec2::new(0.0, 0.0), Vec2::new(400.0, 0.0)];
        // Constant curvature 0.004: inside what a radius-20 tip may sweep as an arc,
        // outside what a radius-80 tip may, so the same curve is priced both ways.
        let arc = by_heading(24, 400.0, |s| 0.004 * s);
        // An Euler spiral **through its inflection**: curvature linear in arc length,
        // running −0.006 → +0.006 with the zero at the middle. It is the one shape that
        // exercises the whole of `fit_arc` in a single stroke — a sign change, the
        // degenerate straight case exactly at the inflection, and the
        // `max_arc_curvature` threshold crossed once on each side (at |κ| = 0.005 for a
        // radius-20 tip), so the fitter alternates between arcs and chords along it.
        // Heading is the integral of curvature: ∫(a·s + b) ds with the constant chosen
        // to put the inflection at the halfway point.
        let spiral = by_heading(24, 400.0, |s| 0.5 * 0.00003 * s * s - 0.006 * s);

        let cases: &[(&str, usize, StrokeRecord)] = &[
            // `position` and `angle` alone: a straight line satisfies both everywhere,
            // so this is the floor — one segment per flattened span, and the number to
            // compare every other row against.
            (
                "straight, plain tip",
                3,
                record(
                    BrushParams {
                        size: 20.0,
                        ..BrushParams::default()
                    },
                    &straight,
                ),
            ),
            // `max_len` from the exchange budget. `smearing()` trades at `lift = deposit = 0.8`,
            // which the rate scaling prices at 0.233 · radius = 4.7px over 400px — not the
            // 2.5px the 0.95 calibration point would cost.
            // **This is the row a reservoir-cadence retuning moves**, and the reason
            // the dynamics path costs what it does. Subdivision is by bisection, so a
            // count sits at or above the length bound's own `400/4.7 = 86` rather than
            // exactly on it.
            (
                "straight, smearing tip",
                118,
                record(smearing(20.0), &straight),
            ),
            // The same cadence on a tip four times as fat. The cap is a fraction of the
            // radius, so this row and the one above stand in the radius ratio — which
            // is what identifies the cadence, rather than something else, as what sets
            // them both.
            (
                "straight, fat smearing tip",
                30,
                record(smearing(80.0), &straight),
            ),
            // `drain` costs **nothing**, which is the point of this row: the falloff is
            // evaluated per fragment from its own arc length, so it asks the flattener
            // for no segments at all and this comes out identical to the smearing row
            // above. Bought per segment it would bind at `0.02 / drain_px` = 4px, and
            // for a quantity that is exact rather than merely finely sampled.
            (
                "straight, draining tip",
                118,
                record(
                    BrushParams {
                        // Per radius, so 0.1 on this 20px tip is 0.005 per canvas px.
                        drain: 0.1,
                        ..smearing(20.0)
                    },
                    &straight,
                ),
            ),
            // The taper, and **not the most expensive row in the table** — the point
            // of carrying the radius as a ramp. A segment holds the taper's slope
            // exactly, so the cut buys only the sagitta of a chord across the
            // profile's own curvature: `len·√(|r''|/8E)` pieces, a second-order term
            // where a per-segment constant radius buys a first-order one. Charging the
            // first order instead costs 121 pieces here, and 211 if the step is
            // denominated in the radius *factor* rather than in px. Nothing about the
            // curve is driving this one — it is the same straight line as the
            // 3-segment row above.
            (
                "straight, tapered tip",
                15,
                record(
                    BrushParams {
                        size: 20.0,
                        start_taper_length: 2.0,
                        end_taper_length: 3.0,
                        ..BrushParams::default()
                    },
                    &straight,
                ),
            ),
            // `angle` (0.1 rad): 0.004 × 400 = 1.6 radians of turning, so ≥ 16 segments
            // however large the curve is drawn.
            (
                "arc, plain tip",
                31,
                record(
                    BrushParams {
                        size: 20.0,
                        ..BrushParams::default()
                    },
                    &arc,
                ),
            ),
            // The same curve under a tip too fat to sweep it as an arc, so `fit_arc`
            // hands back chords instead. **It costs exactly the same**, and that is the
            // point of keeping both rows: at this curvature `angle` binds first, so the
            // arc/chord choice changes what a segment *is* without changing how many
            // there are. If a change to `MAX_TIP_TURN` or to how a too-tight edge is
            // priced ever makes these two diverge, that is worth knowing about.
            (
                "arc, fat tip",
                31,
                record(
                    BrushParams {
                        size: 80.0,
                        ..BrushParams::default()
                    },
                    &arc,
                ),
            ),
            ("arc, smearing tip", 103, record(smearing(20.0), &arc)),
            // The Euler spiral: `angle` again over 1.2 radians of total turning, but
            // with the fitter crossing the arc/chord threshold on each side of a
            // genuine inflection. Cheaper than the arc because it turns one way and
            // then back, rather than accumulating.
            (
                "euler spiral, plain tip",
                26,
                record(
                    BrushParams {
                        size: 20.0,
                        ..BrushParams::default()
                    },
                    &spiral,
                ),
            ),
            (
                "euler spiral, fat tip",
                26,
                record(
                    BrushParams {
                        size: 80.0,
                        ..BrushParams::default()
                    },
                    &spiral,
                ),
            ),
            // Back to `max_len`: the cadence asks for more than the spiral's own shape
            // does, so a smearing tip pays the same price on a curve as on a line.
            (
                "euler spiral, smearing tip",
                98,
                record(smearing(20.0), &spiral),
            ),
        ];

        let mut moved = Vec::new();
        for (name, expected, rec) in cases {
            let got = whole(rec).len();
            if got != *expected {
                moved.push(format!("  {name}: {expected} -> {got}"));
            }
        }
        assert!(
            moved.is_empty(),
            "the segment budget moved (update the counts if this was deliberate):\n{}",
            moved.join("\n")
        );
    }
    /// The reservation in `generate_segments_in` is sized on "each ~100 bytes"; this
    /// is the number, so the comment cannot drift from the struct a second time.
    #[test]
    fn a_segment_is_the_size_the_reservation_assumes() {
        assert_eq!(std::mem::size_of::<Segment>(), 104);
    }
}
