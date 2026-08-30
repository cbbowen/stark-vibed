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

/// The fixtures every suite in [`stroke`](super) builds its sweeps with — here because
/// this is the module that owns the type they build (see the module's own docs).
#[cfg(test)]
pub(super) mod testing;

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
    /// function of travel, and [`ramp`](Self::ramp) is the rest of that function.
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
    pub(super) ramp: f32,
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
    /// host's statement of the ramp `stamp_common::ramp_scale` applies, so the two
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
        self.radius * (1.0 + self.ramp * (u.clamp(0.0, 1.0) - 0.5))
    }

    /// The widest tip this sweep reaches, canvas px.
    ///
    /// Spelled as the shader spells it (`stamp_common::sweep_vertex`'s `hull`) rather
    /// than as `max(r_start, r_end)`, which it equals algebraically. The two can differ
    /// by an ulp in floats, and this one has to come out **no smaller** than the strip
    /// the GPU rasterizes: it is what [`coverage_bounds`](super::region::coverage_bounds) grows the box by, and a box
    /// narrower than its own geometry is a stroke clipped at a tile boundary.
    fn widest_tip(&self) -> f32 {
        self.radius * (1.0 + 0.5 * self.ramp.abs())
    }
}

/// The brush's paint rates **as the pen asked for them here** (§6.2): the four axes of
/// [`BrushDynamics`](stark_model::document::BrushDynamics) scaled by whatever
/// [`BrushModulations`](stark_model::document::BrushModulations) and its effect-side
/// siblings map onto each, plus the tooth's depth.
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
    pub(super) lift: f32,
    pub(super) deposit: f32,
    pub(super) bleed: f32,
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
}

impl Default for Paint {
    /// Every rate at zero — a tool doing nothing — and the tooth at **full give**,
    /// which is the same statement about the substrate: a tip that follows every fall
    /// deposits exactly what it would with no substrate under it at all.
    ///
    /// Written out rather than derived, because the knob runs the other way
    /// (`ToothParams::give`) and a derived zero here would be the *driest* tip
    /// there is — a default that gates paint away rather than one that does nothing.
    fn default() -> Self {
        Self {
            add: 0.0,
            lift: 0.0,
            deposit: 0.0,
            bleed: 0.0,
            tooth_give: stark_model::document::ToothParams::DEFAULT_GIVE,
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
/// the sole purpose of writing them back out, and left `ramp: 0.0` as a field a window
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
// set by [`flatten_tolerance`](super::budget::flatten_tolerance) from
// [`MAX_TIP_TURN`](super::MAX_TIP_TURN)). One function, one rule, so the geometry the
// flattener priced is the geometry that gets swept — and neither can spend the
// positional budget on a primitive the other does not use.

/// The taper's radius profile: the fraction of the brush's radius in force `t` of
/// the way through a taper (§6.2).
///
/// `f(t) = t(3 − t²)/2` — the cubic pinned by `f(0) = 0`, `f(1) = 1`, `f'(1) = 0`,
/// monotone on `[0, 1]`, and within 2% of `sin(πt/2)` everywhere. Both end
/// conditions are the point:
///
/// * `f'(1) = 0` is what makes the taper *smooth*. The taper meets the stroke's
///   full-width body there, and any profile with a slope left at the join (`√t`,
///   plain `t`) puts a visible crease across the stroke where the two meet — the
///   one artifact that would give the trick away.
/// * `f'(0) = 3/2` is what makes it a **point** rather than a blunt cap or a
///   hairline. The outline leaves the tip as a straight wedge, which is what an
///   inked entry stroke looks like; `smoothstep`'s `f'(0) = 0` instead holds the
///   width near zero for a tenth of the taper and reads as a whisker with a bulge
///   behind it.
///
/// A polynomial rather than the sine it approximates because it has to be
/// bit-identical across platforms: the taper decides stored pixels, so replay,
/// goldens and peers all have to agree on it (§12.1), and `sin` is not
/// specified to the last bit.
fn taper_profile(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    0.5 * t * (3.0 - t * t)
}

/// The largest `d/dt` [`taper_profile`] reaches, at `t = 0`. Used to bound how far
/// the radius can move across one swept segment.
const TAPER_MAX_SLOPE: f32 = 1.5;

/// The largest `|d²/dt²|` [`taper_profile`] reaches, at `t = 1` (`f'' = −3t`). What
/// bounds the error of drawing the profile as a **straight ramp** across a segment
/// ([`Sweep::ramp`]) rather than as the curve it is — the only part of the taper's
/// shape a ramp does not already carry exactly.
const TAPER_MAX_CURVATURE: f32 = 3.0;

/// How far the drawn outline may sit from the true cone, in **canvas px**, where the
/// tip is too hard (or too thin) for its own falloff to hide anything.
///
/// A segment's tip is a straight ramp ([`Sweep::ramp`]) across a profile that is
/// cubic, so what a cut has to buy is the *sagitta* of that chord — a second-order
/// quantity, where before the ramp existed it was the whole first-order step. The
/// budget is the flattener's own [`position`](crate::path::FlattenTolerance::position):
/// the taper's edge is as much drawn geometry as the centreline is, and gets the same
/// sub-pixel promise.
///
/// The history is the point of the constant. It was a step in the radius **factor**
/// (2%), which is a px bound that scales with the brush: invisible at radius 20
/// (0.4 px), a comb of ~5 px sawteeth at radius 500 (2026-08-14, the repro capture).
/// Denominating it in px fixed the artifact but priced smoothness at
/// `radius / 0.7` pieces — ~700 per zone on a hard 500 px tip, and it still bound
/// nothing for the pen-driven half of the same problem. Carrying the variation as a
/// ramp instead makes both first-order terms exact and leaves only this.
const TAPER_OUTLINE_PX: f32 = crate::path::FLATTEN_TOLERANCE.position;

/// Where the tip's own falloff is wider than the floor, the outline budget grows with
/// it: a quarter of the shoulder ([`shoulder_per_radius`](super::budget::shoulder_per_radius)),
/// the same resolvable-feature bound
/// [`extent_cell`](super::budget::extent_cell) coarsens against — an outline
/// error the coverage blurs over more than this cannot print as a scallop. What keeps
/// a fat *soft* brush from buying smoothness its edge could not show.
const TAPER_SHOULDER_SLACK: f32 = 0.25;

// **Why there is no cap on `|ramp|` itself**, which is the first thing a reader will
// look for beside the bound above.
//
// Because a large ramp is no longer the deposit's problem either. It used to be: the
// sweep's travel axis was denominated in the segment's *reference* radius, so a tip
// that is `1 ± ramp/2` of that over the segment's two halves booked its exposure
// through a measure off by the same fraction — over-counting one half, under-counting
// the other, cancelling only to first order. The shader now denominates the two ends of
// the span against the tips actually in force at them (`stamp_common::Sweep::span`), so
// adjacent segments agree at the knot they share exactly as their outlines do, and a
// point's total exposure over a pass is the mask's row total whatever the cut.
//
// That matters most exactly where no cut could have helped. Cut an edge whose tip
// starts at the taper's point into `n` uniform pieces and piece `k` spans radius
// `[kΔ/n, (k+1)Δ/n]`, so its ramp is `1/(k + ½)` — **independent of `n`**. The first
// piece sits at the structural limit of 2 whatever it is subdivided to, the second at
// 0.67, the fourth at 0.22. Subdividing an edge that reaches a point buys nothing but
// segments, which is exactly the trap the px-denominated rule fell into: it charged
// ~700 pieces per zone for a first-order term the ramp now carries exactly, and spent
// most of them where the mark is a hairline.
//
// What is left is the *lateral* axis, which cannot be made cut-free at all: a point's
// offset across the tip is measured against a tip that grows while the tip passes over
// it, and a prefix row is one row. The shader freezes it at the moment the tip is
// closest to the point, which is right where it matters — the outline is then exactly
// the taper's own profile — and first-order in the segment length elsewhere. That term
// *is* bought by subdivision, and it is what the sagitta bound above ends up paying
// for as well.
//
// Away from the point the ramp is small without being asked: in the body of a taper
// the outline bound above already puts the radius change per piece at a few percent of
// the tip. So the ramp is bounded where it matters and unbounded where it cannot
// matter, and the one guarantee that has to hold everywhere — `|ramp| < 2`, which is
// what keeps the tip positive at both ends, and now also what keeps both span scales
// `1 ∓ ramp/2` positive — is structural rather than enforced ([`Sweep::ramp`]).

/// Cap on the pieces one flattened edge is cut into for the taper — a backstop on a
/// pathological brush rather than a quality knob.
///
/// The px step makes a whole taper's cost radius-dependent: a shoulderless tip pays
/// up to `TAPER_MAX_SLOPE · radius / TAPER_STEP_PX` pieces (~1000 at the 500 px
/// radius cap), spread over every edge in the zone, and a soft tip logarithmically
/// less. One *edge* only ever needs a fraction of that (the flattener and the
/// exchange cadence bound edge length well under a zone), so the cap binds on
/// nothing real — but a degenerate brush (a taper shorter than the tip is wide, cut
/// from a knot-starved polyline) is clamped here rather than allowed to name its own
/// segment count.
const TAPER_MAX_PIECES: usize = 512;

/// A stroke's taper, resolved for one span range (§6.2).
///
/// Both lengths are in canvas px here, already scaled out of
/// [`BrushParams::taper_px`] and — crucially — already **fitted to the stroke**: if
/// the two zones together are longer than the stroke, both are scaled down in
/// proportion so they exactly meet. The stroke then reaches full width at one point
/// instead of never reaching it, which is what keeps a quick flick a small pointed
/// mark rather than a sliver, continuously as the stroke grows.
#[derive(Copy, Clone, Debug)]
pub(super) struct Taper {
    /// Leading taper length (canvas px); 0 = none.
    start: f32,
    /// Trailing taper length (canvas px); 0 = none.
    end: f32,
    /// Arc length of the whole stroke, for measuring back from its end. Only read
    /// when `end > 0`.
    total: f32,
    /// The brush's nominal radius (canvas px) — what a factor step scales by to
    /// become the px step the subdivision is actually bounding. Nominal rather than
    /// modulated is conservative: pressure only ever scales the tip *down*
    /// ([`Modulation`](stark_model::document::Modulation)), and the real step with it.
    radius: f32,
    /// The tip's shoulder width per unit radius
    /// ([`shoulder_per_radius`](super::budget::shoulder_per_radius)) — what lets the
    /// step relax where the falloff blurs the join anyway.
    shoulder: f32,
}

impl Taper {
    /// The taper in force for a range, given the stroke's total arc length — or
    /// `None` if this range stops short of the stroke's end and so cannot know it.
    ///
    /// A range that does not reach the end gets the **leading taper alone,
    /// uncompressed**. That is not a guess: the engine refuses to freeze any span
    /// that is within the trailing taper's reach of the live end, or that could
    /// still be compressed ([`safe_frozen`](super::safe_frozen)), so a
    /// partial range is one where both of those factors are exactly 1 — and the
    /// commit, which sees the whole stroke, computes the same 1 for it.
    fn resolve(b: &BrushParams, total: Option<f32>) -> Self {
        let (start, end) = b.taper_px();
        let radius = b.size.max(0.5);
        let shoulder = super::budget::shoulder_per_radius(&b.shape);
        match total {
            Some(total) if start + end > total => {
                // Scaled in proportion, so the two zones meet at one point.
                let k = total / (start + end);
                Self {
                    start: start * k,
                    end: end * k,
                    total,
                    radius,
                    shoulder,
                }
            }
            Some(total) => Self {
                start,
                end,
                total,
                radius,
                shoulder,
            },
            None => Self {
                start,
                end: 0.0,
                total: f32::INFINITY,
                radius,
                shoulder,
            },
        }
    }

    /// The fraction of the brush's radius in force at arc length `dist`.
    fn factor(&self, dist: f32) -> f32 {
        let mut f = 1.0;
        if self.start > 0.0 {
            f *= taper_profile(dist / self.start);
        }
        if self.end > 0.0 {
            f *= taper_profile((self.total - dist) / self.end);
        }
        f
    }

    /// Each zone's `TAPER_MAX_SLOPE / length`, or 0 for a zone the interval
    /// `[dist, dist + len]` cannot reach — which is what keeps the extra subdivision
    /// paid only near the ends of the stroke. Both bounds below are built from the
    /// pair, so which zones are in play is decided in one place.
    fn zone_slopes(&self, dist: f32, len: f32) -> (f32, f32) {
        let s = if self.start > 0.0 && dist < self.start {
            TAPER_MAX_SLOPE / self.start
        } else {
            0.0
        };
        let e = if self.end > 0.0 && self.total - (dist + len) < self.end {
            TAPER_MAX_SLOPE / self.end
        } else {
            0.0
        };
        (s, e)
    }

    /// A bound on `|d² factor / d dist²|` anywhere in `[dist, dist + len]` — what a
    /// straight radius ramp has to be cut fine enough to track ([`Sweep::ramp`]).
    ///
    /// The product rule, term for term: `(f_s·f_e)'' = f_s''·f_e + 2 f_s' f_e' +
    /// f_s·f_e''`, and both factors are ≤ 1, so the two curvatures add and the cross
    /// term is twice the product of the slopes. The cross term is only ever nonzero on
    /// a stroke short enough for its two zones to overlap, where it is exactly the
    /// term that would otherwise be missed.
    fn curvature_bound(&self, dist: f32, len: f32) -> f32 {
        let (ss, se) = self.zone_slopes(dist, len);
        let cs = if ss > 0.0 {
            TAPER_MAX_CURVATURE / (self.start * self.start)
        } else {
            0.0
        };
        let ce = if se > 0.0 {
            TAPER_MAX_CURVATURE / (self.end * self.end)
        } else {
            0.0
        };
        cs + ce + 2.0 * ss * se
    }

    /// How many swept segments a flattened edge of length `len` starting at `dist`
    /// has to be cut into. 1 — no cut at all — wherever the taper is flat, which is
    /// everywhere on an untapered brush, so this path is bit-identical to having no
    /// taper code.
    ///
    /// **The first-order variation is not what is being bought here.** A segment
    /// carries the taper's slope exactly, as its ramp ([`Sweep::ramp`]), and two
    /// adjacent segments agree on the radius at the knot they share — so the outline
    /// is continuous however coarse the cut. What is left is one second-order term:
    /// the ramp is a **chord** across a cubic profile, and the outline bows off it by
    /// the sagitta `|r''|·h²/8`.
    ///
    /// That is the whole rule. See the note above [`TAPER_OUTLINE_PX`] for why there
    /// is no companion bound on the ramp's own magnitude — near a taper's point it is
    /// a constant no subdivision can move, and everywhere else this bound has already
    /// made it small.
    fn pieces(&self, dist: f32, len: f32) -> usize {
        // Only the zones the interval reaches bend the radius at all, so an edge in
        // the stroke's body — and every edge of an untapered brush — is one segment,
        // bit-identical to having no taper code.
        let curvature = self.curvature_bound(dist, len) * self.radius;
        if curvature <= 0.0 {
            return 1;
        }
        // The narrowest tip on this edge, which is where the budget is tightest:
        // `factor` is a product of one rising and one falling profile, each monotone
        // on the interval, so its minimum over the interval is at an end.
        let r_lo = self.radius * self.factor(dist).min(self.factor(dist + len));
        // The sub-pixel floor, relaxed by whatever the tip's own falloff can blur over.
        let budget = TAPER_OUTLINE_PX.max(TAPER_SHOULDER_SLACK * self.shoulder * r_lo);
        // `h ≤ √(8·budget/|r''|)`, so the count is `len/h`. Float → int casts saturate
        // in Rust, so a nonsense length cannot wrap here.
        let n = len * (curvature / (8.0 * budget)).sqrt();
        (n.ceil() as usize).clamp(1, TAPER_MAX_PIECES)
    }
}

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
/// (see [`flatten_tolerance`](super::budget::flatten_tolerance)).
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
    // more. Reserved rather than grown from nothing because a `Segment` is 80 bytes and
    // a commit's stroke runs to thousands of them, on the interactive path.
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
    let tip_at = |pressure: f32, tilt: Vec2, tap: f32| {
        let pen = PenState {
            pressure,
            tilt: tilt.length(),
        };
        (b.size * b.modulation.size(pen) * tap).max(0.5)
    };

    // `ends` is the tip at the segment's two ends — where the radius *ramp* comes from
    // ([`Sweep::ramp`]). Everything else is sampled at the midpoint, `at`: the rates
    // below are applied per segment and the midpoint is the reading whose error is
    // second order where either end's would be first.
    let make = |at: At, track: Track, ends: (f32, f32)| {
        // The pen as the modulations read it, at this segment's own attributes
        // (§6.2). `Modulation::factor` clamps anyway.
        let pen = PenState {
            pressure: at.pressure,
            tilt: at.tilt.length(),
        };
        let m = &b.modulation;
        let (r0, r1) = ends;
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
            ramp: (r1 - r0) / radius,
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
        sweep.reach = sweep.widest_tip() * elong;
        // The rates are the effect's own, at its own pen mappings — the one place
        // per segment the enum is asked. An eraser's `add` is its flow (the rate
        // its bite builds at, §6.12) and it has no fluxes to carry; which is
        // a statement about the *brush*, so every segment of every stroke answers
        // it the same way (`dynamics_setup`'s purity argument).
        let (add, lift, deposit, bleed) = match &b.effect {
            stark_model::document::BrushEffect::Paint(p) => (
                p.dynamics.flow * p.modulation.flow(pen),
                p.dynamics.lift * p.modulation.lift(pen),
                p.dynamics.deposit * p.modulation.deposit(pen),
                p.dynamics.bleed * p.modulation.bleed(pen),
            ),
            stark_model::document::BrushEffect::Erase(e) => {
                (e.flow * e.modulation.flow(pen), 0.0, 0.0, 0.0)
            }
        };
        Segment {
            sweep,
            paint: Paint {
                add,
                lift,
                deposit,
                bleed,
                tooth_give: b.tooth.give * m.tooth_give(pen),
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
            // ([`Sweep::ramp`]); it is not approached, it is the same expression
            // evaluated twice.
            let (p0, t0) = pen_at(u0);
            let (p1, t1) = pen_at(u1);
            let ends = (
                tip_at(p0, t0, taper.factor(dist)),
                tip_at(p1, t1, taper.factor(dist + step)),
            );
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

/// **The stretch of the tip along its facing axis, as the renderer reads it** (§6.6):
/// the map carrying a point's place in the tip's reference travel frame into the frame
/// the prefix-τ volume is looked up in, once the extent is drawn out by
/// [`BrushParams::elongation`].
///
/// The whole feature is here, and it costs three floats and no new texture.
/// Stretching the tip by `s` along a canvas axis `û` is the linear map
/// `A = R_û·diag(s, 1)·R_ûᵀ` on the extent, and the deposit is that extent's
/// integral as it is dragged along the travel `t̂`. Substituting `q = A⁻¹p` turns that
/// integral into one of the **unstretched** extent — dragged along
/// `v̂ = normalize(A⁻¹t̂)` instead of `t̂`, over a travel `m = |A⁻¹t̂|` times as long,
/// with `1/m` on the result. Every one of those is something the existing volume
/// already answers: it is indexed by the angle between the mask's native axis and the
/// direction of integration, so a different direction is a different *slice*, not a
/// different bake.
///
/// That holds because the axis is the brush's **facing** axis
/// ([`orientation_turns`]'s), which is what makes the whole of it fit in the volume the
/// brush already binds:
///
/// - `FollowStroke` faces along the tangent, so `û = t̂` and therefore `v̂ = t̂` — the
///   relative angle stays 0 and the single identity layer still serves.
/// - A round tip is rotation-invariant: one slice answers every angle (§6.6).
/// - `Pen` on a stamp already reads the stack of every angle, so a shifted slice
///   index is free.
///
/// An axis free of the facing one would break all three at once — a follow-stroke
/// stamp would need the rotatable stack it never builds — which is why there is no
/// second direction to set.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct Stretch {
    /// `m` — the segment's travel measured in the stretched frame, as a multiple of
    /// its travel in the tip's own. Also the divisor a prefix difference read in that
    /// frame is worth in the tip's, which is why the shaders carry the one number and
    /// not the pair.
    pub(super) travel: f32,
    /// The lookup frame's travel coordinate picks up this much of the point's lateral
    /// offset — the leading edge of an obliquely-stretched tip is not square to its
    /// own travel. 0 wherever the facing axis is along or across the travel.
    pub(super) shear: f32,
    /// What the point's lateral offset is scaled by. **Below 1 for any real stretch**,
    /// which is the direction that matters: the mask's `|y| ≤ 1` then stands for a
    /// wider band of canvas, and that is the stroke getting fatter.
    pub(super) lateral: f32,
    /// How much further round the volume's orientation axis to read, in turns — the
    /// angle from `t̂` to `v̂`, which is the whole of what "another slice" means above.
    /// Added to [`Sweep::orient`] rather than carried, so the shaders see one angle.
    pub(super) turns: f32,
}

impl Stretch {
    /// The identity: the tip as its mask draws it. What every brush without a stretch
    /// gets, and **exactly** the neutral element of every expression that reads one —
    /// `travel = 1`, `shear = 0`, `lateral = 1` leave the shaders' arithmetic an
    /// identity in floats, not merely close to one.
    pub(super) const NONE: Self = Self {
        travel: 1.0,
        shear: 0.0,
        lateral: 1.0,
        turns: 0.0,
    };

    /// Solve the map for an elongation `s` along a facing axis sitting `orient` turns
    /// round from the travel direction — which is [`orientation_turns`]'s own answer,
    /// for both sources: `FollowStroke` faces along the tangent and reports 0, and
    /// `Pen` reports the azimuth relative to the travel, which is the same angle.
    ///
    /// Short-circuited at `s = 1` rather than left to fall out of the general path.
    /// The trigonometry below *does* return the identity there — `A⁻¹` is `I`, `v` is
    /// `(1, 0)` — but a brush with no stretch should not be relying on `atan2(0, 1)`
    /// being exactly zero to render what it always rendered.
    pub(super) fn solve(elongation: f32, orient: f32) -> Self {
        // A non-finite elongation takes the same exit as an absent one, since the
        // values here arrive from files, presets and peers: a NaN would otherwise reach
        // a lane the shaders divide by.
        if !elongation.is_finite() || elongation <= 1.0 {
            return Self::NONE;
        }
        // `A⁻¹ = I + (k − 1)·ûûᵀ` in the travel frame, where `k = 1/s` and `û` is the
        // facing axis expressed there. Symmetric, so its first column is `(a00, a01)`
        // and that column is `v = A⁻¹t̂` — the direction the sweep is integrated along
        // once the stretch is taken out.
        let k = 1.0 / elongation;
        let (sn, cs) = (orient * std::f32::consts::TAU).sin_cos();
        let a00 = 1.0 + (k - 1.0) * cs * cs;
        let a01 = (k - 1.0) * cs * sn;
        let a11 = 1.0 + (k - 1.0) * sn * sn;
        let m = (a00 * a00 + a01 * a01).sqrt();
        // `M = R(v̂ → x̂)·A⁻¹`, whose first column is `(m, 0)` by construction — so the
        // map is upper triangular and three numbers state it. Its determinant is
        // `det A⁻¹ = k`, which is where `lateral` comes from without a second dot
        // product, and which says the same thing the shape does: a tip stretched `s`
        // along one axis covers `s` times the mask per unit of the frame.
        let (vx, vy) = (a00 / m, a01 / m);
        Self {
            travel: m,
            shear: vx * a01 + vy * a11,
            lateral: k / m,
            turns: -a01.atan2(a00) / std::f32::consts::TAU,
        }
    }

    /// The box in the tip's **reference travel frame** that holds everything the mask
    /// can put on the canvas, as a multiple of the frame radius: `(along, across)`.
    ///
    /// **The shaders' — `stamp_common::stretch_hull` is this function**, and only they
    /// need it: it is the sweep strip and the dynamics loop's rim test that are drawn
    /// in the reference travel frame, where the host's boxes are canvas-aligned and
    /// take [`Sweep::reach`] instead. So this side is `#[cfg(test)]`, existing to hold
    /// the shader to a formula rather than to be called in anger — the derivation is
    /// short enough to restate and wrong enough to matter, since under-reporting it is
    /// a stroke cut off along a straight line where its own geometry ran out.
    ///
    /// A point takes paint only where the map lands it inside the mask's `|x| ≤ 1,
    /// |y| ≤ 1`, so `|y| ≤ 1/lateral` and `|x| ≤ (1 + |shear|/lateral)/travel`.
    /// `(1, 1)` exactly for [`NONE`](Self::NONE).
    #[cfg(test)]
    pub(super) fn hull(&self) -> (f32, f32) {
        let across = 1.0 / self.lateral;
        ((1.0 + self.shear.abs() * across) / self.travel, across)
    }
}

#[cfg(test)]
mod tests {
    use super::super::budget::{MAX_TIP_TURN, flatten_tolerance};
    use super::super::region::{coverage_bounds, segment_end};
    use super::super::safe_frozen;
    use super::testing::{record, smearing};
    use super::*;
    use stark_model::document::BrushShape;

    // --- tapers ----------------------------------------------------------

    /// A straight stroke `len` px long with a tapered brush of `radius`.
    fn tapered_record(radius: f32, start: f32, end: f32, len: f32) -> StrokeRecord {
        // Enough control points that the curve has spans to freeze part of, and
        // straight so arc length is the chord and the taper zones are easy to reason
        // about.
        let path: Vec<stark_model::path::ControlPoint> = (0..=12)
            .map(|i| stark_model::path::ControlPoint::at(Vec2::new(i as f32 / 12.0 * len, 0.0)))
            .collect();
        StrokeRecord {
            layer: stark_model::document::LayerId::ROOT,
            brush: BrushParams {
                size: radius,
                drain: 0.0,
                start_taper_length: start,
                end_taper_length: end,
                ..BrushParams::default()
            },
            path,
            seed: 0,
            start: 0.0,
        }
    }

    /// Every segment of a stroke, as [`Segment`]s — what the chunker and the tile
    /// walks are handed.
    fn whole_segments(rec: &StrokeRecord) -> Vec<Segment> {
        generate_segments_in(rec, flatten_tolerance(&rec.brush), StrokeSpans::whole(rec)).0
    }

    /// The same, as bare [`Sweep`]s.
    ///
    /// Almost everything below is a claim about *geometry* — where the tip went and how
    /// wide it was — so it is asked of the half that carries geometry. A test that
    /// wanted a paint rate would have to say so by using [`whole_segments`], which is
    /// the point of the split being visible here too.
    fn whole(rec: &StrokeRecord) -> Vec<Sweep> {
        sweeps(whole_segments(rec))
    }

    fn sweeps(segs: Vec<Segment>) -> Vec<Sweep> {
        segs.into_iter().map(|s| s.sweep).collect()
    }

    /// The profile's two end conditions are the whole design (see [`taper_profile`]),
    /// so they are asserted rather than left to the formula: pinned at both ends,
    /// monotone in between, and *flat* where it meets the stroke's full-width body —
    /// which is what makes the join invisible.
    #[test]
    fn the_taper_profile_is_pinned_flat_at_the_join_and_monotone() {
        assert_eq!(taper_profile(0.0), 0.0, "the tip is a point");
        assert_eq!(taper_profile(1.0), 1.0, "the join is full width");
        assert_eq!(taper_profile(2.0), 1.0, "past the join it stays full width");

        let mut prev = 0.0;
        for i in 1..=200 {
            let f = taper_profile(i as f32 / 200.0);
            assert!(f > prev, "not monotone at t = {}", i as f32 / 200.0);
            prev = f;
        }
        // Numerical slope over the last 1% of the taper, as a multiple of the
        // average: ~0 means the curve arrives flat. (Exactly `f'(1) = 0`.)
        let slope = (taper_profile(1.0) - taper_profile(0.99)) / 0.01;
        assert!(slope < 0.05, "the taper meets the body with slope {slope}");
        // And leaves the tip as a wedge, not a whisker: a `smoothstep`-shaped profile
        // would be under 0.03 here.
        let tip_slope = (taper_profile(0.01) - taper_profile(0.0)) / 0.01;
        assert!(tip_slope > 1.0, "the tip is blunt, slope {tip_slope}");
    }

    /// What the taper does to a stroke: pointed at both ends, full width in between,
    /// and no step in the outline between the two.
    ///
    /// Asked of the tip at the stroke's actual **ends** rather than of the first and
    /// last segments' radii, which is a distinction the radius ramp makes real: a
    /// segment's `radius` is its midpoint, and since the cut does not have to buy the
    /// first order, the segment holding an end point can be long enough that its
    /// midpoint is nowhere near one. What is at the point is `tip_at(0)`.
    #[test]
    fn a_tapered_stroke_narrows_at_both_ends() {
        let radius = 20.0;
        let rec = tapered_record(radius, 4.0, 6.0, 900.0);
        let segs = whole(&rec);
        let first = segs.first().expect("segments").tip_at(0.0);
        let last = segs.last().expect("segments").tip_at(1.0);
        let widest = segs
            .iter()
            .fold(0.0f32, |m, s| m.max(s.tip_at(0.0)).max(s.tip_at(1.0)));

        assert!(first < 0.1 * radius, "the start is not a point: {first}");
        assert!(last < 0.1 * radius, "the end is not a point: {last}");
        assert!(
            (widest - radius).abs() < 1e-3,
            "the body should reach full radius, got {widest}"
        );
        // The outline the segments describe has no step in it.
        assert_outline_is_continuous(&segs);
    }

    /// **The property the ramp exists to have** (§6.2, [`Sweep::ramp`]): consecutive
    /// segments agree on the tip at the knot they share, so the stroke's outline has
    /// no C⁰ break to alias — at any brush size, and however coarsely the taper is
    /// cut.
    ///
    /// Stated as an agreement between neighbours rather than as a bound on a step,
    /// because that is the difference between the two designs. A per-segment radius
    /// can only ever make the step *small*; a ramp makes it zero, since both sides
    /// evaluate the same pen and the same taper at the same arc length.
    ///
    /// The tolerance is not slack for the taper: within a flattened edge the two are
    /// the identical float expression and agree to the bit. It covers the edge
    /// *boundaries*, where the arc length a segment measures its taper at is
    /// accumulated along the polyline and the two sides can differ by an ulp or two of
    /// a large number.
    fn assert_outline_is_continuous(segs: &[Sweep]) {
        for (i, w) in segs.windows(2).enumerate() {
            let (before, after) = (w[0].tip_at(1.0), w[1].tip_at(0.0));
            let tol = 1e-3 * before.max(after).max(1.0);
            assert!(
                (before - after).abs() <= tol,
                "segment {i} ends at radius {before} and {} begins at {after} — \
                 the outline has a step in it",
                i + 1,
            );
        }
    }

    /// The one bound on the ramp that has to hold everywhere, and the reason
    /// `stamp_common::ramp_scale` needs no clamp: `|ramp| < 2`, so the tip is positive
    /// at both ends of every segment. Structural rather than enforced — it follows
    /// from flooring both ends at half a px — so this checks the algebra rather than a
    /// rule that could be forgotten.
    fn assert_tips_stay_positive(segs: &[Sweep]) {
        for (i, s) in segs.iter().enumerate() {
            assert!(
                s.ramp.abs() < 2.0,
                "segment {i} ramps by {}, which puts a tip at or past zero",
                s.ramp,
            );
            assert!(
                s.tip_at(0.0) > 0.0 && s.tip_at(1.0) > 0.0,
                "segment {i} has a non-positive tip at one end",
            );
        }
    }

    /// The outline at the size that strains it: a radius-500 brush with long tapers.
    /// Swept at one radius per segment, the cut can only make the step between segments
    /// smaller and never zero, so the point draws as a comb of ~5 px sawteeth; the
    /// radius ramp is what removes the step rather than shrinking it.
    ///
    /// It also pins what the ramp buys: the cut does not have to buy the first order,
    /// so a taper costs a logarithmic handful of segments instead of one per `0.7 px`
    /// of radius. The count is the scale-free one — see the sibling test that draws the
    /// same stroke a hundredth the size.
    #[test]
    fn a_huge_brushs_taper_has_no_step_in_its_outline() {
        let mut rec = tapered_record(500.0, 5.0, 11.0, 7600.0);
        rec.brush.shape = BrushShape::Round { hardness: 0.95 };
        let segs = whole(&rec);
        assert_outline_is_continuous(&segs);
        assert_tips_stay_positive(&segs);
        assert!(
            segs.len() < 200,
            "{} segments — the cut is still buying the first order",
            segs.len()
        );
    }

    /// …and the cut is **scale-free**: the same stroke at a hundredth the size costs
    /// the same handful of segments, where the px-denominated rule it replaced charged
    /// the large brush a hundred times the small one for the same picture.
    ///
    /// Quoted as a ratio rather than two counts, because that is the claim — the
    /// absolute numbers move with any retuning of the ramp's own bounds
    /// (`BrushDynamics::radius_ramp`), the independence does not.
    #[test]
    fn a_tapers_cost_does_not_grow_with_the_brush() {
        let count = |radius: f32| {
            let mut rec = tapered_record(radius, 5.0, 11.0, radius * 15.2);
            rec.brush.shape = BrushShape::Round { hardness: 0.95 };
            whole(&rec).len()
        };
        let (small, large) = (count(5.0), count(500.0));
        assert!(
            large <= small * 2,
            "the same stroke costs {small} segments at radius 5 and {large} at 500",
        );
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
    /// shaders branch on that zero (`stamp_common::ramp_scale`), and a zero that were
    /// merely small would take the general path and round differently.
    #[test]
    fn a_tip_that_holds_still_carries_no_ramp() {
        // No taper, no size modulation, full pressure throughout.
        let mut rec = tapered_record(40.0, 0.0, 0.0, 900.0);
        rec.brush.modulation = stark_model::document::BrushModulations::default();
        for s in whole(&rec) {
            assert_eq!(s.ramp, 0.0, "an unvarying tip picked up a ramp");
            assert_eq!(s.radius, 40.0, "an unvarying tip changed size");
        }
    }

    /// A stroke shorter than its own two tapers still reaches full width, at one
    /// point in the middle: the zones are scaled down in proportion rather than
    /// clamped, so a quick flick is a small pointed mark and not an invisible sliver.
    /// And the behaviour is continuous in length — the whole reason to compress
    /// rather than clamp.
    #[test]
    fn short_strokes_compress_their_tapers_instead_of_vanishing() {
        let radius = 16.0;
        for len in [4.0f32, 20.0, 60.0, 160.0, 400.0] {
            let rec = tapered_record(radius, 6.0, 6.0, len);
            let widest = whole(&rec).iter().fold(0.0f32, |m, s| m.max(s.radius));
            assert!(
                widest > 0.9 * radius,
                "a {len}px stroke only reached radius {widest} of {radius}"
            );
        }
        // A click has no length for a taper to run along — and no travel for a
        // deposit to integrate over. It sweeps nothing: what a press will lay is
        // the hover's mark to say before it is made (§18.1.10), and a release
        // with nothing to deposit commits nothing (`Session::end_stroke`).
        let mut dot = tapered_record(radius, 6.0, 6.0, 0.0);
        dot.path.truncate(1);
        assert!(
            whole(&dot).is_empty(),
            "a click swept segments though it has no travel"
        );
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

    /// The load-bearing claim behind [`super::safe_frozen`]: for any prefix it
    /// admits, rendering the stroke as *head + tail* produces the very same swept
    /// segments as rendering it in one pass.
    ///
    /// That is what the live == committed invariant (§1.3) reduces to here. A frozen
    /// head is never redrawn, so if the head's segments differed from the commit's by
    /// even a radius the stroke would visibly change under the pointer at release —
    /// and the taper is exactly the kind of parameter that invites it, being measured
    /// from an end of the stroke that has not been drawn yet.
    #[test]
    fn a_taper_safe_head_plus_tail_is_the_single_pass_stroke() {
        let rec = tapered_record(18.0, 5.0, 9.0, 1200.0);
        let tol = flatten_tolerance(&rec.brush);
        let all = crate::path::span_count(rec.path.len());
        let frozen = safe_frozen(&rec, all);
        assert!(frozen > 0, "nothing could be frozen at all");
        assert!(
            frozen < all,
            "the trailing taper must hold the last spans back"
        );

        let (head, dist) = generate_segments_in(
            &rec,
            tol,
            StrokeSpans {
                range: 0..frozen,
                dist: 0.0,
            },
        );
        let (tail, _) = generate_segments_in(
            &rec,
            tol,
            StrokeSpans {
                range: frozen..all,
                dist,
            },
        );
        let split = sweeps(head.into_iter().chain(tail).collect());
        let one_pass = whole(&rec);

        assert_eq!(
            split.len(),
            one_pass.len(),
            "the split stroke has a different number of segments"
        );
        for (i, (a, b)) in split.iter().zip(&one_pass).enumerate() {
            assert_eq!(a.radius, b.radius, "segment {i}: radius differs (taper)");
            assert_eq!(a.dist, b.dist, "segment {i}: arc length differs");
            assert_eq!(a.length, b.length, "segment {i}: length differs");
            assert_eq!(a.start, b.start, "segment {i}: start differs");
        }
    }

    /// An untapered brush is untouched by any of the above, to the bit: the taper's
    /// subdivision has to be a no-op where the taper is flat, or it would re-cut
    /// every stroke ever drawn and invalidate every golden.
    #[test]
    fn an_untapered_stroke_is_not_subdivided() {
        let rec = tapered_record(18.0, 0.0, 0.0, 900.0);
        let segs = whole(&rec);
        let pts = crate::path::flatten_spans(
            &rec.path,
            0..crate::path::span_count(rec.path.len()),
            0.0,
            flatten_tolerance(&rec.brush),
        );
        assert_eq!(
            segs.len(),
            pts.len() - 1,
            "one segment per flattened edge, with no taper subdivision"
        );
        assert!(
            segs.iter().all(|s| s.radius == 18.0),
            "an untapered stroke is full width throughout"
        );
        assert_eq!(
            safe_frozen(&rec, 7),
            7,
            "an untapered brush holds nothing back from freezing"
        );
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
                // Tightness: the box is the travel's own, grown by one radius — not
                // by `√2` of one.
                let tight_lo = s.start.min(end) - Vec2::splat(s.radius);
                let tight_hi = s.start.max(end) + Vec2::splat(s.radius);
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
    /// hoping — the same argument that makes [`taper_profile`] a polynomial (§12.1).
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
    /// (`budget::flatten_tolerance`). These are properties of the rule, not
    /// measured counts — unlike the table below, a failure here is a bug rather than a
    /// retuning.
    #[test]
    fn the_exchange_budget_scales_with_the_transfer_rate() {
        use stark_model::document::BrushDynamics;
        let at = |lift: f32, deposit: f32, charge: f32| {
            flatten_tolerance(&BrushParams {
                size: 100.0,
                effect: stark_model::document::BrushEffect::paint_with(
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

        // A brush with no dynamics at all is not capped by this at all.
        assert!(at(0.0, 0.0, 0.0) > 100.0);

        // Never a tightening: a brush that trades *faster* than the reference is left at
        // the reference step, so no setting pays more than it did before the scaling.
        assert!((at(1.0, 1.0, 0.0) - at(0.95, 0.95, 0.0)).abs() < 0.05);
        assert!(at(0.99, 0.99, 0.0) >= at(0.95, 0.95, 0.0) - 0.05);
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
}

/// The stretched tip (§6.6).
///
/// One claim is being tested here, and everything else is a reading of it: **the swept
/// integral of an extent drawn out along an axis is the integral of the *undrawn*
/// extent, along another direction, over another travel, times a constant.** That is
/// what lets a stretch cost three floats and no new texture, and it is exactly the kind
/// of claim that is either exact or quietly wrong by a few percent everywhere — the mask
/// is still swept, the stroke still looks like a stroke, and the profile it draws is not
/// the one the brush names.
///
/// So it is checked against a **direct numerical sweep** of the stretched extent,
/// with a mask that is deliberately neither round nor symmetric: a rotation-invariant
/// tip would satisfy the identity at every angle for the wrong reason (any slice would
/// do), and a symmetric one would hide the shear.
#[cfg(test)]
mod stretch_tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Rotate `p` by `a` radians.
    fn rot(p: (f32, f32), a: f32) -> (f32, f32) {
        let (s, c) = a.sin_cos();
        (p.0 * c - p.1 * s, p.0 * s + p.1 * c)
    }

    /// A stand-in for a brush mask's optical-depth field `κ`, in mask coordinates.
    ///
    /// **Anisotropic and off-centre on purpose.** The identity under test moves the
    /// direction of integration, so a field that reads the same along every direction
    /// would pass it however the slice was chosen; and it puts a shear on the travel
    /// coordinate, which a field symmetric about `y = 0` would leave undetectable.
    /// Smooth and compactly supported so the quadrature below converges quickly.
    fn mask(q: (f32, f32)) -> f32 {
        let r2 = q.0 * q.0 + q.1 * q.1;
        if r2 >= 1.0 {
            return 0.0;
        }
        let lobe = (-6.0 * (q.0 - 0.2) * (q.0 - 0.2) - 2.0 * (q.1 + 0.1) * (q.1 + 0.1)).exp();
        (1.0 - r2) * lobe
    }

    /// The prefix-τ volume's own lookup, evaluated by quadrature rather than baked:
    /// slice `w` holds the mask turned by `+w` turns and integrated along `+x`, so a
    /// read at `(x, y)` is `∫ mask(R(−wτ)·(u, y)) du` up to `x`
    /// (`assets::rotate_layers`, `assets::build_prefix_tau`).
    fn prefix(x: f32, y: f32, w: f32) -> f32 {
        const STEPS: usize = 4000;
        let (lo, hi) = (-1.5f32, x.min(1.5));
        if hi <= lo {
            return 0.0;
        }
        let du = (hi - lo) / STEPS as f32;
        (0..STEPS)
            .map(|i| {
                let u = lo + (i as f32 + 0.5) * du;
                mask(rot((u, y), -w * TAU)) * du
            })
            .sum()
    }

    /// The swept depth at `p`, computed the long way: the stretched extent dragged
    /// along `+x` for `len`, integrated step by step in the travel frame. The thing the
    /// renderer's three floats have to reproduce.
    fn swept_directly(p: (f32, f32), len: f32, elongation: f32, orient: f32) -> f32 {
        const STEPS: usize = 4000;
        let psi = orient * TAU;
        let k = 1.0 / elongation;
        let (sn, cs) = psi.sin_cos();
        // `A⁻¹ = I + (k − 1)·ûûᵀ`, the map the extent is read through.
        let inv = |q: (f32, f32)| {
            let d = (k - 1.0) * (q.0 * cs + q.1 * sn);
            (q.0 + d * cs, q.1 + d * sn)
        };
        let dt = len / STEPS as f32;
        (0..STEPS)
            .map(|i| {
                let t = (i as f32 + 0.5) * dt;
                // The tip's own extent at this instant: the mask turned to face the
                // shape's angle, then drawn out along it.
                mask(rot(inv((p.0 - t, p.1)), -psi)) * dt
            })
            .sum()
    }

    /// The same depth as the renderer takes it: one prefix difference on one slice,
    /// with the map and the gain [`Stretch::solve`] hands back.
    fn swept_through_the_volume(p: (f32, f32), len: f32, elongation: f32, orient: f32) -> f32 {
        let st = Stretch::solve(elongation, orient);
        let slice = (orient + st.turns).rem_euclid(1.0);
        // `stretch_look`, in Rust: the lookup frame's travel coordinate at the sweep's
        // end and at its start, and the lateral offset.
        let x1 = st.travel * p.0 + st.shear * p.1;
        let x0 = x1 - st.travel * len;
        let y = st.lateral * p.1;
        (prefix(x1, y, slice) - prefix(x0, y, slice)) / st.travel
    }

    /// **The whole feature, against a sweep that knows nothing about it.**
    ///
    /// Across elongations, facing angles, sample points and travels: the definite
    /// integral of the stretched extent equals the unstretched volume read at
    /// [`Stretch`]'s slice, over its travel, times its gain. If the derivation is wrong
    /// anywhere — the sign of the slice shift, the shear, the Jacobian — this is where
    /// it shows, because the left-hand side is the picture and the right-hand side is
    /// the renderer.
    #[test]
    fn a_stretched_sweep_is_the_unstretched_volume_read_at_another_slice() {
        let mut worst = 0.0f32;
        for &elongation in &[1.0, 1.5, 2.5, 4.0, 8.0] {
            for &orient in &[0.0, 0.05, 0.125, 0.25, 0.4, 0.5, 0.75, 0.9] {
                for &len in &[0.3, 1.0, 3.0] {
                    for &p in &[
                        (0.0, 0.0),
                        (0.4, 0.3),
                        (-0.6, -0.45),
                        (1.1, 0.2),
                        (0.2, -0.8),
                    ] {
                        let direct = swept_directly(p, len, elongation, orient);
                        let volume = swept_through_the_volume(p, len, elongation, orient);
                        let err = (direct - volume).abs();
                        worst = worst.max(err);
                        assert!(
                            err < 2e-3,
                            "s={elongation} orient={orient} len={len} at {p:?}: \
                             swept {direct}, volume {volume}",
                        );
                    }
                }
            }
        }
        // Both sides are 4000-step midpoint rules over a field with a `1 − r²` kink at
        // the rim, so the floor here is the quadrature's and not the identity's.
        assert!(worst < 2e-3, "worst disagreement {worst}");
    }

    /// The identity is the identity **in floats**, not merely to a tolerance — which is
    /// what lets every existing brush keep its pixels. Checked at the ways a brush says
    /// it does not stretch: the knob at rest, a knob the pen has modulated to nothing,
    /// and the two malformed values that arrive from files, presets and peers.
    #[test]
    fn a_brush_that_does_not_stretch_gets_the_exact_identity() {
        for &orient in &[0.0, 0.125, 0.25, 0.5, 0.9] {
            for &e in &[BrushParams::elongation(0.0), 1.0, 0.5, f32::NAN] {
                assert_eq!(
                    Stretch::solve(e, orient),
                    Stretch::NONE,
                    "elongation {e} at orient {orient} is not the identity",
                );
            }
        }
        // And the identity leaves the shaders' arithmetic exact — `stretch_look`,
        // `stretch_gain` and `stretch_hull` spelled here as they are spelled there.
        let st = Stretch::NONE;
        for &(x, y, lr) in &[(0.37f32, -0.62f32, 1.7f32), (-1.0, 0.0, 0.0)] {
            assert_eq!(st.travel * x + st.shear * y, x);
            assert_eq!(st.lateral * y, y);
            assert_eq!(st.travel * x + st.shear * y - st.travel * lr, x - lr);
        }
        assert_eq!(1.0 / st.travel, 1.0);
        assert_eq!(st.hull(), (1.0, 1.0));
    }

    /// `hull` has to hold **everything the mask can paint**, because what is drawn for
    /// the extent is drawn from it: the sweep strip in the shader, and the tile box
    /// on the host. Under-report it and the stroke is cut off along a straight line
    /// where its own geometry ran out — the failure every under-reported reach lands
    /// on ([`Sweep::reach`]), which a stretch reintroduces at a different scale.
    ///
    /// So: every point of the reference travel frame that the map lands *inside* the
    /// mask's square must be inside the hull.
    #[test]
    fn the_hull_holds_every_point_the_stretched_mask_can_paint() {
        for &elongation in &[1.0, 1.5, 2.5, 4.0, 8.0] {
            for &orient in &[0.0, 0.05, 0.125, 0.25, 0.4, 0.6, 0.875] {
                let st = Stretch::solve(elongation, orient);
                let (along, across) = st.hull();
                // Walk the mask's own square back through the map: its corners are the
                // extremes, but the edges are walked too so nothing rests on that.
                for i in 0..=64 {
                    let f = i as f32 / 32.0 - 1.0;
                    for (mx, my) in [(1.0, f), (-1.0, f), (f, 1.0), (f, -1.0)] {
                        // `stretch_unlook`: the map is upper triangular, so its inverse
                        // is three reciprocals.
                        let y = my / st.lateral;
                        let x = (mx - st.shear * y) / st.travel;
                        assert!(
                            x.abs() <= along + 1e-4 && y.abs() <= across + 1e-4,
                            "s={elongation} orient={orient}: mask ({mx}, {my}) rides at \
                             ({x}, {y}), outside the hull ({along}, {across})",
                        );
                    }
                }
            }
        }
    }

    /// [`Sweep::reach`] is the same promise in canvas px and in every direction at once
    /// — a *box* the segment is drawn into — so it is scaled by the elongation alone
    /// rather than by the hull's two axes. This is why that is sound: the map's operator
    /// norm is exactly the elongation, so no point of the mask lands further out than
    /// `elongation` times where the unstretched tip's did, whichever way it faces.
    #[test]
    fn the_reach_covers_every_texel_the_stretched_mask_can_paint() {
        for &elongation in &[1.0, 2.0, 5.0, 8.0] {
            for &orient in &[0.0, 0.1, 0.25, 0.33, 0.5] {
                let psi = orient * TAU;
                let (sn, cs) = psi.sin_cos();
                // `A = I + (s − 1)·ûûᵀ` — the forward map, mask to canvas.
                let fwd = |q: (f32, f32)| {
                    let d = (elongation - 1.0) * (q.0 * cs + q.1 * sn);
                    (q.0 + d * cs, q.1 + d * sn)
                };
                for i in 0..=64 {
                    let f = i as f32 / 32.0 - 1.0;
                    // The disc's rim is the frontier: nothing any canonical shape
                    // can paint lies outside its disc ([`Sweep::reach`]), so the rim
                    // covering under the map is the whole promise.
                    let q = (f.cos(), f.sin());
                    let p = fwd(q);
                    let d = (p.0 * p.0 + p.1 * p.1).sqrt();
                    assert!(
                        d <= elongation + 1e-4,
                        "s={elongation} orient={orient}: mask {q:?} lands {d} out, \
                         past a reach of {elongation}",
                    );
                }
            }
        }
    }

    /// What the axis is *for*, stated as the two readings a hand would recognise — and
    /// the reason a pencil could not be built out of the size mapping it used to use,
    /// which scales both of these together.
    ///
    /// Lean the pen **along** the stroke and the mark gets heavier without getting
    /// wider; lean it **across** and it gets wider without the centreline getting
    /// heavier per unit travel. Measured off the solved map rather than off a picture,
    /// which is what makes it a statement about the model.
    #[test]
    fn leaning_along_the_stroke_darkens_it_and_leaning_across_widens_it() {
        let s = 3.0;
        // Along the travel: the lookup's lateral axis is untouched, so the profile
        // across the stroke is the shape it was — and a full pass, whose prefix
        // saturates at the row total whatever the travel scaled to, lays `s` times as
        // much.
        let along = Stretch::solve(s, 0.0);
        assert!((along.lateral - 1.0).abs() < 1e-6, "{along:?}");
        assert!((1.0 / along.travel - s).abs() < 1e-5, "gain: {along:?}");
        assert_eq!(along.shear, 0.0, "an axis along the travel cannot shear it");
        assert!(along.turns.abs() < 1e-6, "nor turn the slice: {along:?}");

        // Across it: the mask's own `|y| ≤ 1` now stands for `s` radii of canvas, which
        // is the stroke `s` times wider, and the depth per unit travel is untouched.
        let across = Stretch::solve(s, 0.25);
        assert!((1.0 / across.lateral - s).abs() < 1e-5, "{across:?}");
        assert!((across.travel - 1.0).abs() < 1e-6, "gain: {across:?}");
        assert!(
            across.shear.abs() < 1e-6,
            "an axis across the travel cannot shear it"
        );
        assert!(across.turns.abs() < 1e-6, "nor turn the slice: {across:?}");

        // Obliquely: both, plus the shear that leans the leading edge — the term only
        // an oblique lean has, and the one a per-axis scale could not express.
        let oblique = Stretch::solve(s, 0.125);
        assert!(oblique.shear.abs() > 0.1, "{oblique:?}");
        assert!(oblique.turns.abs() > 0.01, "{oblique:?}");
        // The map's determinant is `1/s` at every angle: a tip drawn out `s` along one
        // axis covers `s` times the mask per unit of the frame, however it is turned.
        for st in [along, across, oblique] {
            assert!((st.travel * st.lateral - 1.0 / s).abs() < 1e-5, "{st:?}");
        }
    }

    /// The knob's own contract: exactly 1 at rest, monotone, and bounded — the last
    /// because the elongation prices the stroke, every tile the drawn-out tip reaches
    /// being one the loop rasterizes and dispatches over.
    #[test]
    fn the_elongation_knob_is_the_identity_at_rest_and_bounded_at_the_top() {
        assert_eq!(BrushParams::elongation(0.0), 1.0);
        assert_eq!(
            BrushParams::elongation(-1.0),
            1.0,
            "a negative knob is no stretch, not a squash"
        );
        assert_eq!(
            BrushParams::elongation(f32::NAN),
            1.0,
            "and neither is a NaN"
        );
        assert_eq!(BrushParams::elongation(1.0), BrushParams::MAX_ELONGATION);
        assert_eq!(BrushParams::elongation(9.0), BrushParams::MAX_ELONGATION);
        let mut prev = 0.0;
        for i in 0..=100 {
            let e = BrushParams::elongation(i as f32 / 100.0);
            assert!(e >= prev, "not monotone at {i}");
            assert!(e <= BrushParams::MAX_ELONGATION);
            prev = e;
        }
    }
}
