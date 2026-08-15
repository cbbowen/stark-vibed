//! Turning a fitted path into swept segments, and the region measurements that decide
//! where the stamp loop cuts one into pieces (§6.2).
//!
//! Both render paths flatten through here, so both see the same segments for the
//! same record — which is what lets a live tail and the commit that replaces it
//! agree pixel for pixel.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use crate::document::{BrushParams, BrushShape, OrientationSource, PenState, StrokeRecord};
use crate::geom::{TILE_APRON, TILE_SIZE, TILE_TEX, TileCoord, Vec2};

use super::StrokeSpans;
use super::budget::{BLEED_TRAVEL_QUANTUM, MAX_REGION_DIM, MAX_STAMPS};

/// One swept segment of the stroke.
///
/// The centreline is a **circular arc**, not a chord: `start` and `dir` give the
/// frame it leaves in, `curvature` bends it, and `length` measures along it. A
/// straight sweep is `curvature == 0` and is what every quantity below reduces to,
/// exactly — see [`crate::path::fit_arc`].
#[derive(Copy, Clone)]
pub(super) struct Segment {
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
    /// **`|ramp| < 2` structurally**, so the tip in force is positive everywhere
    /// without a clamp: the ends are floored at `0.5` by [`generate_segments_in`], and
    /// `|r₁ − r₀| < r₁ + r₀` for any two positive radii.
    ///
    /// Relative rather than absolute (px) because it is then the *same number* for the
    /// tip and for the frame the sweep is unrolled in ([`frame`](Self::frame)): the two
    /// differ by a constant [`frame_scale`], which cancels. One lane serves both.
    pub(super) ramp: f32,
    /// The radius of the **frame the sweep is integrated in**, in canvas px: the one
    /// radius the shaders see, and the tip's own only when the two coincide.
    ///
    /// They part company for a pen-oriented stamp (§6.6). Its prefix-τ volume is padded
    /// so the mask can turn inside it without losing its corners, so the volume's
    /// `[-1, 1]²` is `PEN_PAD` tips wide rather than one — and the sweep has to be
    /// unrolled in a frame that much larger for the mask inside it to land at the
    /// radius the brush asked for. Everything the shader derives from brush-local
    /// coordinates (the swept arc, the color-noise domain, the reservoir's placement)
    /// is in this frame; nothing the host prices is, which is what keeps a nib's dab
    /// and its bleed the size of the tip instead of the size of the box around it.
    pub(super) frame: f32,
    /// How far from the centreline this tip's deposit can land, in canvas px — the
    /// half-extent of its footprint **square**, not of the disc inscribed in it
    /// ([`tip_reach`], scaled by the radius).
    ///
    /// Scaled by the segment's **widest** tip rather than its mean, since the ramp
    /// makes those different numbers and this one bounds a box: under-reporting it is
    /// a stroke clipped at a tile boundary (see [`coverage_bounds`]).
    ///
    /// Every shape is swept over brush-local `|x| ≤ 1, |y| ≤ 1` — that is the whole
    /// domain of the prefix-τ volume — so a shape is free to fill the corners of its
    /// own mask, and an image stamp that does reaches `√2 · radius` from the
    /// centreline. Only the round tip, zero outside its unit disc by construction
    /// ([`round_coverage`]), reaches exactly `radius`.
    pub(super) reach: f32,
    /// Arc length of the centreline (canvas px) — the tip's own travel, which is the
    /// measure every rate in both paths is denominated in.
    pub(super) length: f32,
    /// Shape orientation for this segment as a fraction of a full turn ∈ [0, 1): the
    /// relative angle between the shape's native axis and the travel direction, used to
    /// pick the prefix-τ orientation layer. 0 for follow-stroke (§6.6).
    pub(super) orient: f32,
    /// Arc length from the stroke start to this segment's start (canvas px) — the
    /// third axis of the color-dynamics noise lookup (§6.2).
    pub(super) dist: f32,
    /// The brush's paint rates **as the pen asked for them here** (§6.2): the four
    /// axes of [`BrushDynamics`](crate::document::BrushDynamics) scaled by whatever
    /// [`Modulations`](crate::document::Modulations) maps onto each.
    ///
    /// They live on the segment rather than on the stroke because that is now what
    /// they are — the pen attributes they follow are interpolated per segment, and
    /// the flattener already holds their step to
    /// [`FlattenTolerance::attribute`](crate::path::FlattenTolerance::attribute).
    /// Each is at most the brush's own value, never more, which is what lets every
    /// bound taken against `rec.brush` stay a bound (see
    /// [`Modulation`](crate::document::Modulation)).
    ///
    /// `charge` is absent on purpose: it is the tool's *initial* load, one number for
    /// the whole stroke, and there is no per-segment version of it to carry.
    pub(super) add: f32,
    pub(super) lift: f32,
    pub(super) deposit: f32,
    pub(super) bleed: f32,
    /// How deep this segment's tip bites into the canvas weave (§6.4) — the brush's
    /// `tooth`, likewise modulated. Not a paint rate: it gates `add` per *texel* from
    /// the ground under it, in the shader.
    pub(super) tooth: f32,
}

impl Segment {
    /// The tip in force a fraction `u` of this segment's travel in, canvas px — the
    /// host's statement of the ramp `stamp_common::ramp_scale` applies, so the two
    /// definitions can be read against each other.
    ///
    /// `u` is clamped, like the shader's: past either end the tip has, as far as this
    /// segment is concerned, stopped at the end it reached.
    ///
    /// Only the tests ask — the renderer evaluates the ramp on the GPU, per fragment,
    /// and the one thing the *host* needs from it (the widest tip, which sizes the
    /// coverage box) is [`widest_tip`](Self::widest_tip).
    #[cfg(test)]
    pub(super) fn tip_at(&self, u: f32) -> f32 {
        self.radius * (1.0 + self.ramp * (u.clamp(0.0, 1.0) - 0.5))
    }

    /// The widest tip this segment reaches, canvas px.
    ///
    /// Spelled as the shader spells it (`stamp_common::sweep_vertex`'s `hull`) rather
    /// than as `max(r_start, r_end)`, which it equals algebraically. The two can differ
    /// by an ulp in floats, and this one has to come out **no smaller** than the strip
    /// the GPU rasterizes: it is what [`coverage_bounds`] grows the segment's box by,
    /// and a box narrower than its own geometry is a stroke clipped at a tile boundary.
    fn widest_tip(&self) -> f32 {
        self.radius * (1.0 + 0.5 * self.ramp.abs())
    }
}

// Per-segment instance data for the sweep shader, generated from `stamp.wesl`'s own
// vertex parameters (§6.10) — including the prose on each lane, which now lives
// beside the declaration that decides how it is read.
pub(super) use stark_shaders::mirror::stamp::SegmentInstance;

/// Generate the round tip's coverage: the soft disc whose *swept* profile across the
/// stroke is `1 − |y|^h`, for `h = 1/(1 − hardness)` and `y` the distance from the
/// centreline in radii.
///
/// The profile is what is being designed here, not the footprint. What `hardness`
/// names is how the *stroke* falls off from its centreline; the tip that produces it
/// is whatever it has to be, and it is not the profile's own shape — a swept deposit
/// composes in **optical depth**, so a full pass lays `1 − exp(−τ(y))` where `τ` is
/// this mask's `κ = −ln(1 − coverage)` integrated along the travel axis
/// ([`build_prefix_tau`](crate::assets::build_prefix_tau)). Ask instead for the field
/// whose row integrals are
///
/// ```text
/// τ(y) = −h·ln|y|        (so 1 − exp(−τ(y)) = 1 − |y|^h, as wanted)
/// ```
///
/// and that is an Abel transform, which inverts in closed form: the radial
///
/// ```text
/// κ(r) = (h/π)·acos(r)/r,   r < 1
/// ```
///
/// has exactly those integrals. So the tip is `1 − exp(−κ(r))` and the profile is
/// arrived at rather than approached. Rate scales the exponent rather than leaving the
/// family — a pass at strength `a` lays `1 − |y|^(a·h)`, the same shape at another
/// hardness — and the field is radially symmetric, as a round tip's ought to be.
///
/// What this replaces aimed at the same profile through the *linear* integral: a
/// `1 − r^h` disc divided by its own chord half-length, `1/√(1 − y²)`. The log in
/// between is what it did not account for, and it is not a small correction, because
/// `−ln(1 − c)` weights the high-coverage core far above the rim: the stroke came out
/// fuller than its hardness named everywhere, by 0.08 in coverage at `hardness = 0`
/// and by 0.54 at `hardness = 0.9`, with the whole falloff crushed into the last few
/// texels of the rim — and on a hard tip the flanks left the mask above coverage 1
/// entirely, where the clamp ate the overshoot.
///
/// `κ` diverges at the centre, as it must for a profile that reaches exactly 1 there,
/// so the core saturates against that same 0.999 clamp and lands a shade under 1
/// instead. Outside it the profile is exact to a thousandth (`tests`, below).
pub(super) fn round_coverage(hardness: f32, res: u32) -> Vec<f32> {
    let h = 1.0 / (1.0 - hardness).max(0.01);
    let mut cov = vec![0.0f32; (res * res) as usize];
    for y in 0..res {
        let fy = (y as f32 + 0.5) / res as f32 * 2.0 - 1.0;
        for x in 0..res {
            let fx = (x as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let r = (fx * fx + fy * fy).sqrt();
            // Zero outside the disc; `+∞` at a centre exactly hit (`acos(0)/0`), which
            // is the one place the profile asks for a coverage of exactly 1.
            let kappa = if r < 1.0 {
                h * r.acos() / (std::f32::consts::PI * r)
            } else {
                0.0
            };
            cov[(y * res + x) as usize] = 1.0 - (-kappa).exp();
        }
    }
    cov
}

// --- swept arcs ----------------------------------------------------------------
//
// The arc a flattened edge stands for is [`crate::path::fit_arc`]'s, called here with
// the very cap the flattener called it with (`FlattenTolerance::max_arc_curvature`,
// set by [`flatten_tolerance`](super::flatten_tolerance) from
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
/// ([`Segment::ramp`]) rather than as the curve it is — the only part of the taper's
/// shape a ramp does not already carry exactly.
const TAPER_MAX_CURVATURE: f32 = 3.0;

/// The travel a stroke deposits at its very shortest, in radii of the tip in force
/// (§6.2) — the **touch-down dab**.
///
/// A swept deposit is a definite integral over travel, so a press that has not moved
/// integrates over nothing and lays nothing, and the first pixel of a drag lays a
/// twentieth of one radius' worth. Taken literally that is a tool that draws nothing
/// until the hand has moved a tip's width, which is not what pressing a loaded brush
/// to paper does.
///
/// So a stroke travels at least this far: whatever it is short by is made up by a
/// *dwell* segment swept about the stroke's own midpoint. A click is the limiting
/// case — the whole dab, centred on the one point that was pressed — and the dwell
/// shrinks to nothing by the time the tip has travelled this far under its own steam,
/// so the mark grows continuously from a dot into a stroke instead of jumping between
/// the two.
///
/// The value is what the mark *looks* like rather than a bound on anything: 0.6
/// radii of dwell is a little over a third of the optical depth a full pass lays,
/// which the slab law (§6.1) renders as a dot around 90% as opaque as the stroke the
/// same brush draws — a dab, and a round-looking one, since it stretches the tip's
/// own footprint by less than a third.
pub(super) const DAB_TRAVEL: f32 = 0.6;

/// How far the drawn outline may sit from the true cone, in **canvas px**, where the
/// tip is too hard (or too thin) for its own falloff to hide anything.
///
/// A segment's tip is a straight ramp ([`Segment::ramp`]) across a profile that is
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
/// [`footprint_cell`](super::budget::footprint_cell) coarsens against — an outline
/// error the coverage blurs over more than this cannot print as a scallop. What keeps
/// a fat *soft* brush from buying smoothness its edge could not show.
const TAPER_SHOULDER_SLACK: f32 = 0.25;

// **Why there is no cap on `|ramp|` itself**, which is the first thing a reader will
// look for beside the bound above.
//
// A large ramp costs accuracy in the *deposit*, not in the outline: the sweep's travel
// axis is denominated in the segment's reference radius (`stamp_common::Sweep`), so a
// tip that is `1 ± ramp/2` of that over the segment's two halves books its exposure
// through a measure off by the same fraction — over-counting one half, under-counting
// the other, and cancelling to first order because the reference is the midpoint.
//
// Where that residual is largest, no cut can reduce it. Cut an edge whose tip starts
// at the taper's point into `n` uniform pieces and piece `k` spans radius
// `[kΔ/n, (k+1)Δ/n]`, so its ramp is `1/(k + ½)` — **independent of `n`**. The first
// piece sits at the structural limit of 2 whatever it is subdivided to, the second at
// 0.67, the fourth at 0.22. Subdividing an edge that reaches a point buys nothing but
// segments, which is exactly the trap the px-denominated rule fell into: it charged
// ~700 pieces per zone for a first-order term the ramp now carries exactly, and spent
// most of them where the mark is a hairline.
//
// Away from the point the ramp is small without being asked: in the body of a taper
// the outline bound above already puts the radius change per piece at a few percent of
// the tip. So the ramp is bounded where it matters and unbounded where it cannot
// matter, and the one guarantee that has to hold everywhere — `|ramp| < 2`, which is
// what keeps the tip positive at both ends — is structural rather than enforced
// ([`Segment::ramp`]).

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
    /// ([`Modulation`](crate::document::Modulation)), and the real step with it.
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
        let radius = b.radius.max(0.5);
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
    /// straight radius ramp has to be cut fine enough to track ([`Segment::ramp`]).
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
    /// carries the taper's slope exactly, as its ramp ([`Segment::ramp`]), and two
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

/// The arc one segment sweeps along.
///
/// `dir` is the tangent the sweep *starts* along — the frame's x axis — while
/// `mid_dir` is the one at the midpoint, the same midpoint-sampling argument applied
/// to the one attribute that reads a direction. They are the same vector on a
/// straight segment.
///
/// `dist` is the exception to that rule: it is the arc length at the segment's
/// **start**, because the shader adds the fragment's own offset along the travel to
/// it (`stamp_common.wesl`).
struct Sweep {
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
/// follows whatever [`Modulations`](crate::document::Modulations) points at it
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
/// flattener no longer has to buy accuracy for it with segments
/// (see [`flatten_tolerance`](super::flatten_tolerance)).
///
/// Returns the range's segments plus the arc length at its end — measured on the
/// emitted polyline rather than recomputed, so the range that resumes from it starts
/// on the exact accumulator these segments were built with.
///
/// Two things here are measured against the stroke's **whole** length, which only a
/// range that reaches its final span knows: the trailing taper (a range that stops
/// short takes the leading taper alone, [`Taper::resolve`]) and the touch-down dab
/// ([`DAB_TRAVEL`], which a partial range never has). Both are sound rather than
/// approximate, and [`safe_frozen`](super::safe_frozen) is the one rule that makes
/// them so.
pub(super) fn generate_segments_in(
    rec: &StrokeRecord,
    tol: crate::path::FlattenTolerance,
    spans: StrokeSpans,
) -> (Vec<Segment>, f32) {
    let b = &rec.brush;
    let dist0 = spans.dist;
    let reaches_end = spans.range.end >= crate::path::span_count(rec.path.len());
    let from_start = spans.range.start == 0;
    let pts = crate::path::flatten_spans(&rec.path, spans.range, dist0, tol);
    let end_dist = pts.last().map_or(dist0, |p| p.dist);
    let mut segs = Vec::new();
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
        (b.radius * b.modulation.size(pen) * tap).max(0.5)
    };

    // `ends` is the tip at the segment's two ends — where the radius *ramp* comes from
    // ([`Segment::ramp`]). Everything else is sampled at the midpoint, `at`: the rates
    // below are applied per segment and the midpoint is the reading whose error is
    // second order where either end's would be first.
    let make = |at: At, sweep: Sweep, ends: (f32, f32)| {
        // The pen as the modulations read it, at this segment's own attributes
        // (§6.2). `Modulation::factor` clamps anyway.
        let pen = PenState {
            pressure: at.pressure,
            tilt: at.tilt.length(),
        };
        let m = &b.modulation;
        let d = b.dynamics;
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
        let mut seg = Segment {
            start: at.pos,
            dir: sweep.dir,
            curvature: sweep.curvature,
            radius,
            ramp: (r1 - r0) / radius,
            frame: radius * frame_scale(b),
            // Filled from `widest_tip` below, which needs the ramp this initializer is
            // still building — and must be *that* expression rather than one equal to
            // it, since this bounds the strip the GPU draws.
            reach: 0.0,
            length: sweep.length,
            orient: orientation_turns(b.orientation, sweep.mid_dir, at.tilt),
            dist: sweep.dist,
            add: d.add * m.flow(pen),
            lift: d.lift * m.lift(pen),
            deposit: d.deposit * m.deposit(pen),
            bleed: d.bleed * m.bleed(pen),
            tooth: b.tooth * m.tooth(pen),
        };
        seg.reach = seg.widest_tip() * tip_reach(&b.shape);
        seg
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
        // no longer breaks its curvature at every joint. Curvature 0 comes back for a
        // straight or barely-curved edge and everything below reduces to what it was.
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
            // ([`Segment::ramp`]); it is not approached, it is the same expression
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
                Sweep {
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

    // The touch-down dab ([`DAB_TRAVEL`]): a stroke that has not yet travelled a dab's
    // worth sweeps the difference about its own **midpoint**, so a click leaves the
    // whole dab centred where it was pressed and every longer stroke grows out of that
    // one continuously. Centred rather than led from the start point because a dab has
    // no direction to lead in: a click has no tangent at all (the fitter gives a lone
    // knot none), and swept from the point it would read as a short dash in whatever
    // direction the fallback happened to name.
    //
    // Only a range that is the **whole** stroke may add it, for the trailing taper's
    // reason exactly — the length it is measured against is the stroke's, which a
    // partial range does not know. And for the same reason it is sound rather than
    // approximate: [`safe_frozen`](super::safe_frozen) refuses to freeze anything
    // until the stroke has travelled a whole dab, so a partial range is always one
    // whose dab is zero, and the commit computes zero for it too.
    //
    // `dab_bound` is the longest dwell any stroke of this brush could owe — the size
    // mapping and the taper only ever scale the tip *down* (a modulation is a factor
    // in [0, 1] by construction, `Modulation`), and the fitter clamps the pen
    // attributes to the curve as well as to its control points. So a stroke past it
    // is past every dab, and an ordinary stroke leaves here without so much as
    // walking its own polyline.
    let dab_bound = DAB_TRAVEL * b.radius.max(0.5);
    if reaches_end && from_start && end_dist < dab_bound {
        // `range.start == 0` is what makes `end_dist` the whole stroke's arc length:
        // `dist` is the arc *before* the range, and before the first span there is none.
        let mid = end_dist * 0.5;
        let (pos, dir, pressure, tilt) = sample_at(&pts, mid);
        // The tip in force at the midpoint — where a stroke short enough to want a dab
        // is at its widest, its two compressed taper zones meeting there (`Taper`). A
        // click is the limit of that: zero length compresses both zones to nothing, so
        // `factor` is exactly 1 and a tapered brush dots at full size rather than
        // leaving the invisible speck a taper read literally would give.
        let tap = taper.factor(mid);
        let tip = tip_at(pressure, tilt, tap);
        let dwell = DAB_TRAVEL * tip - end_dist;
        if dwell > 0.0 {
            segs.insert(
                0,
                make(
                    At {
                        pos: pos - dir * (dwell * 0.5),
                        pressure,
                        tilt,
                    },
                    Sweep {
                        dir,
                        mid_dir: dir,
                        curvature: 0.0,
                        length: dwell,
                        // The dwell is *at* the stroke, not before it: it must not run
                        // the arc-length clock — which `drain` and the color noise are
                        // measured on — backwards past the stroke's own start.
                        dist: (mid - dwell * 0.5).max(0.0),
                    },
                    // A dab does not ramp: it is one tip pressed at one point, and the
                    // travel it sweeps is a dwell rather than a stretch of the path.
                    (tip, tip),
                ),
            );
        }
    }
    (segs, end_dist)
}

/// The stroke's state at arc length `arc` along a flattened polyline: position, unit
/// travel direction, pressure and tilt. Only the touch-down dab asks
/// ([`DAB_TRAVEL`]), and only ever about the midpoint of a stroke that is at most a
/// dab long.
///
/// `+x` where the stroke has no direction to give — a click, whose one knot the
/// fitter leaves with no tangent, and a press that reported the same position twice.
/// Which direction that is cannot matter, because the dab is swept symmetrically
/// about the point it is asked for.
fn sample_at(pts: &[crate::path::IntermediateSample], arc: f32) -> (Vec2, Vec2, f32, Vec2) {
    for w in pts.windows(2) {
        let (a, c) = (w[0], w[1]);
        let span = c.dist - a.dist;
        if span > 0.0 && arc <= c.dist {
            let t = ((arc - a.dist) / span).clamp(0.0, 1.0);
            let v = c.pos - a.pos;
            let len = v.length();
            return (
                a.pos + v * t,
                if len > 1e-5 {
                    v / len
                } else {
                    Vec2::new(1.0, 0.0)
                },
                a.pressure + (c.pressure - a.pressure) * t,
                a.tilt + (c.tilt - a.tilt) * t,
            );
        }
    }
    let p = pts.last().expect("a flattened range is never empty here");
    (p.pos, Vec2::new(1.0, 0.0), p.pressure, p.tilt)
}

/// How far a tip of this shape deposits from the centreline, as a multiple of the
/// radius in force (§6.6) — the half-extent of its footprint square, which is what
/// [`Segment::reach`] scales.
///
/// The sweep integrates every shape over brush-local `|x| ≤ 1, |y| ≤ 1`, so what
/// separates the two answers is not how the mask is *drawn* but whether the shape can
/// occupy the corners of its own square. Stated as a property of the shape rather than
/// measured off the mask because it has to be the same number on both sides of a
/// commit and on every peer, and a bound is exactly what the callers want: a box that
/// is a little large costs fragments that difference their prefix taps to zero, and one
/// that is a little small is a clipped stroke.
pub(super) fn tip_reach(shape: &BrushShape) -> f32 {
    match shape {
        // Exactly zero outside its unit disc, by construction (`round_coverage`), so
        // the corners of its square hold nothing to lose.
        BrushShape::Round { .. } => 1.0,
        // An imported mask may be opaque to the very corner texel. Swept along a
        // diagonal that square's canvas box is `√2` times as wide as the disc's, which
        // is what a bound taken off the radius alone was cutting off at the tile
        // boundary.
        BrushShape::Stamp(_) => std::f32::consts::SQRT_2,
    }
}

/// The frame the sweep is integrated in for this brush, as a multiple of the tip's own
/// radius (§6.6) — what [`Segment::frame`] scales, and 1 for everything but a
/// pen-oriented stamp.
///
/// A property of the brush rather than of the shape alone, because it is the pair
/// `(shape, orientation)` that decides which prefix-τ volume gets bound: a stamp that
/// follows the stroke reads an unpadded identity layer and a stamp pinned to the pen
/// reads the padded stack. Resolved from the same pair on both sides, so the frame the
/// renderer sweeps in is the frame the volume was baked for
/// ([`AssetStore::prefix_view`](crate::assets::AssetStore::prefix_view)).
pub(super) fn frame_scale(b: &BrushParams) -> f32 {
    match (b.shape, b.orientation) {
        // A disc is its own rotation, so its volume is one slice with nothing to turn
        // inside it — whatever the orientation source says.
        (BrushShape::Round { .. }, _) => 1.0,
        (BrushShape::Stamp(_), OrientationSource::FollowStroke) => 1.0,
        (BrushShape::Stamp(_), OrientationSource::Pen) => crate::assets::pen_frame_scale(),
    }
}

/// The shape's orientation for a segment, as a fraction of a full turn ∈ [0, 1): the
/// relative angle between the shape's native axis and the travel direction `dir`, which
/// picks the prefix-τ orientation layer (§6.6).
///
/// - [`OrientationSource::FollowStroke`]: the shape tracks the tangent, so the relative
///   angle is always 0 (the historical behaviour; for a round tip it is moot anyway).
/// - [`OrientationSource::Pen`]: the shape is pinned to the pen's azimuth (the tilt
///   direction) in canvas space, so relative to the travel direction it is `α − φ` — as
///   the stroke curves the footprint angle stays fixed in the world, like a nib.
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

/// Call `f(segment index, tile)` for every tile whose *texture* (interior + apron) a
/// segment's swept capsule overlaps, in segment order.
///
/// The apron is included in the reach so a stroke landing within a tile's interior
/// but inside a neighbour's apron band re-renders that neighbour too, keeping the
/// shared apron/interior overlap bit-identical (§6.4).
///
/// **A segment writes exactly zero outside the tiles this names**, which is what lets
/// [`tiles_with_segments`] hand each tile a subset rather than the whole stroke. The
/// rasterized geometry does reach further — the shaders sweep a generous angular
/// margin so a round cap is never clipped — but out there a fragment differences two
/// prefix-τ taps that are equal and writes nothing at all (see [`coverage_bounds`]).
/// Zero through the `over` blend and zero through the additive one are both exact
/// identities, so which segments a tile is handed cannot change what lands in it.
fn for_each_touched(segments: &[Segment], mut f: impl FnMut(usize, TileCoord)) {
    let tile = TILE_SIZE as f32;
    for (i, s) in segments.iter().enumerate() {
        let (lo, hi) = segment_bounds(s);
        let (x0, x1) = ((lo.x / tile).floor() as i32, (hi.x / tile).floor() as i32);
        let (y0, y1) = ((lo.y / tile).floor() as i32, (hi.y / tile).floor() as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                f(i, TileCoord::new(x, y));
            }
        }
    }
}

/// Tiles whose texture any segment's swept capsule overlaps — [`for_each_touched`]
/// with the segments forgotten — **plus the bleed firings' windows**. What the
/// dynamics path wants, which sizes a region from the tiles alone.
///
/// The windows are in the walk because they write: a firing's sweep is walked back
/// along the crossing segment's own arc, up to one [`BLEED_TRAVEL_QUANTUM`] before
/// the segment it fires after (`plan::bleed_fires`) — and for the first segment of
/// a piece or a live-tail range that stretch lies behind every segment box here,
/// with one apron texel of margin. Left out of the accounting (as they were until
/// 2026-08-11, while [`snapshot_size`](super::dynamics) *did* take them), the flux
/// written there was silently clipped by the region's bounds check, and a rewritten
/// tile's apron could diverge from an unrewritten neighbour's interior — a §6.4
/// break in exactly the configuration `tests/seam.rs` does not draw.
pub(super) fn affected_tiles(
    segments: &[Segment],
    fires: &[(usize, Segment)],
) -> BTreeSet<TileCoord> {
    let mut coords = BTreeSet::new();
    for_each_touched(segments, |_, c| {
        coords.insert(c);
    });
    for (_, w) in fires {
        for_each_touched(std::slice::from_ref(w), |_, c| {
            coords.insert(c);
        });
    }
    coords
}

/// The same walk, keeping **which** segments reach each tile.
///
/// This is what the swept path draws from. Drawing every segment into every tile made
/// a stroke cost `segments × tiles` vertex invocations, nearly all of them on quads
/// that fall outside the tile being rendered and are discarded after being shaded —
/// and a tapered brush spends ~211 segments on a straight line, so a long stroke
/// crossing a document's worth of tiles paid for the product of two large numbers. Per
/// tile the cost is now the segments that actually reach it, and over a stroke the
/// total is `Σ tiles-per-segment`: the segment count times a small constant, since a
/// segment is at most a tip wide.
///
/// The indices come out ascending, because the walk is in segment order — which
/// matters, since the color target's blend is `over` and therefore ordered. Each tile
/// sees the stroke's own order over the subset that reaches it.
pub(super) fn tiles_with_segments(segments: &[Segment]) -> BTreeMap<TileCoord, Vec<u32>> {
    let mut map: BTreeMap<TileCoord, Vec<u32>> = BTreeMap::new();
    for_each_touched(segments, |i, c| map.entry(c).or_default().push(i as u32));
    map
}

/// Where a segment's centreline ends — along the arc, not along the chord.
pub(super) fn segment_end(s: &Segment) -> Vec2 {
    crate::path::arc_at(s.start, s.dir, s.curvature, s.length).0
}

/// The canvas box one segment's swept coverage occupies — the arc, grown by the tip
/// that rides along it.
///
/// The rasterized geometry reaches further than this at the caps (the shaders sweep a
/// generous angular margin so the round end is never clipped), but every fragment out
/// there differences two prefix taps to exactly zero and writes nothing. What a box
/// has to contain is where the deposit *lands*, which is within the tip's
/// [`reach`](Segment::reach) of the arc.
///
/// **The tip's reach, not its radius.** The two are the same number only for a shape
/// that stays inside the disc inscribed in its mask; a stamp that fills the corners
/// reaches `√2` times as far, and swept along a diagonal that difference is a whole
/// corner of the footprint. Under-reporting it here is a stroke clipped at a tile
/// boundary — `for_each_touched` leaves the tile out of the render (or leaves this
/// segment out of a tile another segment brought in), and the dynamics loop dispatches
/// a rect too small for its own footprint.
pub(super) fn coverage_bounds(s: &Segment) -> (Vec2, Vec2) {
    let end = segment_end(s);
    let reach = Vec2::splat(s.reach + crate::path::arc_sagitta(s.curvature, s.length));
    (s.start.min(end) - reach, s.start.max(end) + reach)
}

/// [`coverage_bounds`] grown by the apron a rewritten tile's neighbours reach into
/// (§6.4). The one place that reach is defined: [`affected_tiles`] enumerates the
/// tiles it touches, [`chunk_segments`] accumulates it into the region a run of
/// segments needs, and those two answers have to be the same rectangle.
fn segment_bounds(s: &Segment) -> (Vec2, Vec2) {
    let (lo, hi) = coverage_bounds(s);
    let apron = Vec2::splat(TILE_APRON as f32);
    (lo - apron, hi + apron)
}

/// The size of the region [`region_rect`] would build for a coverage box, without
/// building the tile set.
///
/// Same rectangle, reached by bounding box rather than by enumerating tiles:
/// [`chunk_segments`] asks this question once per segment while it walks a stroke,
/// and `affected_tiles` costs a set insert per tile per segment — on a long stroke,
/// the very cost the incremental repaint exists to avoid.
fn region_of(lo: Vec2, hi: Vec2) -> (u32, u32) {
    let tile = TILE_SIZE as f32;
    // The tile block the coverage spans, measured between tile origins.
    let span = |a: f32, b: f32| ((b / tile).floor() - (a / tile).floor()) * tile;
    (
        span(lo.x, hi.x) as u32 + TILE_TEX,
        span(lo.y, hi.y) as u32 + TILE_TEX,
    )
}

/// Split a stroke's segments into consecutive runs, each of which the stamp loop can
/// evolve inside one [`MAX_REGION_DIM`]-bounded region (§6.2).
///
/// The loop works on a 1:1 copy of the canvas under the stroke, so a stroke that
/// crosses the document would want a region the size of the document. It does not
/// have to have one: the loop is *sequential*, so running the first run of segments
/// and then the second — each over its own region, the second compositing what the
/// first wrote back — is the same computation as running them all over one region.
/// The same segments in the same order, and the state that threads between them is
/// the reservoir, which is brush-local and says nothing about where the stroke is.
/// That is the identical argument that lets a live tail resume a frozen head
/// ([`ToolState`](super::ToolState)); a piece is just a cut the renderer makes for
/// itself rather than one the fitter made for it.
///
/// Greedy: extend the run until one more segment would push its region past
/// [`MAX_REGION_DIM`], or its dispatch batch past [`MAX_STAMPS`]. A run always holds
/// at least one segment — one tip's own footprint is the floor no subdivision gets
/// under, which is what [`segment_fits_region`] gates on instead.
///
/// A segment is measured **with its own bleed firings** ([`affected_tiles`]'s
/// reason): a window can reach back a quantum before the segment it fires after, so
/// a piece's region must hold everything the piece will write, windows included —
/// the same rectangle [`region_rect`] then builds from the tiles the pair names.
pub(super) fn chunk_segments(
    segments: &[Segment],
    fires: &[(usize, Segment)],
) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    let mut start = 0;
    let mut pending = fires.iter().peekable();
    for (i, s) in segments.iter().enumerate() {
        let (mut slo, mut shi) = segment_bounds(s);
        while let Some((_, w)) = pending.next_if(|(after, _)| *after == i) {
            let (wlo, whi) = segment_bounds(w);
            (slo, shi) = (slo.min(wlo), shi.max(whi));
        }
        let (glo, ghi) = (lo.min(slo), hi.max(shi));
        let (w, h) = region_of(glo, ghi);
        if i > start && (w > MAX_REGION_DIM || h > MAX_REGION_DIM || i - start >= MAX_STAMPS) {
            runs.push(start..i);
            (start, lo, hi) = (i, slo, shi);
        } else {
            (lo, hi) = (glo, ghi);
        }
    }
    if start < segments.len() {
        runs.push(start..segments.len());
    }
    runs
}

/// Whether one segment of `b`'s swept footprint fits a region.
///
/// [`chunk_segments`] can cut a stroke as fine as a single segment, but no finer: the
/// reservoir pickup reduces over the whole tip at once, so the region can never be
/// smaller than one footprint. A brush too fat for that is the one thing left that
/// sends a dynamics stroke to the plain swept deposit — and, unlike the whole-stroke
/// measurement this replaced, it is decided from the brush alone, so it costs nothing
/// to re-ask on every pointer move and cannot answer differently for a piece than for
/// the stroke it belongs to.
///
/// Bounded rather than measured, since it has to hold for segments that do not exist
/// yet: radius peaks at the brush's own (pressure only scales it down), travel at the
/// flattening cap — or at the [`DAB_TRAVEL`] radii a touch-down dab sweeps, which
/// ignores the cap — and a coverage box of a given extent spans at most one tile more
/// than it covers, whichever tile boundary it happens to straddle.
pub(super) fn segment_fits_region(b: &BrushParams, tol: crate::path::FlattenTolerance) -> bool {
    let radius = b.radius.max(0.5);
    // The chord is what `path::within` caps; the arc over it is longer, and bows a
    // sagitta out of its own box. Both are bounded by the turn a segment may bend
    // through (`MAX_HALF_TURN_SIN`) — under 2% and under 5% of the chord — so a
    // single margin covers the pair with room to spare.
    //
    // A bleeding brush's segment also carries its firings, whose windows reach up
    // to one quantum back past its start ([`chunk_segments`]) — so the floor the
    // chunker cannot get under is that much longer for it.
    let bleed = if b.dynamics.bleed > 0.0 {
        BLEED_TRAVEL_QUANTUM * radius
    } else {
        0.0
    };
    let length = tol.max_len.max(DAB_TRAVEL * radius) * 1.1 + bleed;
    // The tip's reach rather than its radius, for [`coverage_bounds`]' reason: a
    // stamp that fills its mask's corners occupies a `√2`-wider box, and this is the
    // bound that decides whether the loop may draw the brush at all.
    let extent = length + 2.0 * (radius * tip_reach(&b.shape) + TILE_APRON as f32);
    let worst = (extent / TILE_SIZE as f32).ceil().max(0.0) as u32 * TILE_SIZE + TILE_TEX;
    worst <= MAX_REGION_DIM
}

/// The region the stamp loop evolves for a stroke piece's affected `coords`: exactly
/// the tile block they span, grown by one apron on each side so the write-back can
/// slice whole `TILE_TEX` blocks out of it — plus the *list* of tiles to composite
/// into it, which is those tiles and the one-tile ring around them (§6.4).
///
/// The ring is in the tile list but deliberately **not** in the rectangle. Its whole
/// job is to give a rewritten tile's apron the neighbour interior it overlaps, and an
/// apron is [`TILE_APRON`] texels — so extending the rectangle by a whole *tile* on
/// every side, as it once did, paid for roughly 4× the region to fill a one-texel
/// band. Ring tiles that fall outside the rectangle simply clip when composited. On a
/// live tail, which covers a handful of tiles and is redrawn on every pointer move,
/// that difference is most of the cost of the whole path.
///
/// Returns `None` if `coords` is empty. The size is [`chunk_segments`]'s business,
/// not this one's — it hands over pieces that fit by construction.
pub(super) fn region_rect(coords: &BTreeSet<TileCoord>) -> Option<RegionRect> {
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for c in coords {
        lo = lo.min(c.origin());
        hi = hi.max(c.origin());
    }
    if !lo.x.is_finite() {
        return None;
    }
    let w = (hi.x - lo.x) as u32 + TILE_TEX;
    let h = (hi.y - lo.y) as u32 + TILE_TEX;
    let mut halo: BTreeSet<TileCoord> = BTreeSet::new();
    for c in coords {
        for dy in -1..=1 {
            for dx in -1..=1 {
                halo.insert(TileCoord::new(c.x + dx, c.y + dy));
            }
        }
    }
    let origin = lo - Vec2::splat(TILE_APRON as f32);
    Some(RegionRect {
        halo: halo.into_iter().collect(),
        lo,
        origin,
        w,
        h,
    })
}

/// What [`region_rect`] measures for a piece: the region rectangle the stamp loop
/// evolves, and the tiles composited into it.
pub(super) struct RegionRect {
    /// The tiles to composite: the affected set plus the one-tile ring around it, so
    /// rewritten tiles' aprons read real neighbour content (§6.4).
    pub(super) halo: Vec<TileCoord>,
    /// The top-left affected tile's origin — the region's *interior* origin, which
    /// the write-back measures each tile's offset against.
    pub(super) lo: Vec2,
    /// The region rectangle's top-left in canvas px: [`lo`](Self::lo) less one
    /// apron — what every slot's coordinates are measured from.
    pub(super) origin: Vec2,
    /// The rectangle's extent in texels.
    pub(super) w: u32,
    pub(super) h: u32,
}

#[cfg(test)]
mod tests {
    use super::super::budget::{MAX_TIP_TURN, flatten_tolerance};
    use super::super::safe_frozen;
    use super::*;

    // --- the round tip ---------------------------------------------------

    /// The whole claim [`round_coverage`] makes, checked where it is a claim: a full
    /// pass of the tip lays `1 − |y|^h` across the stroke.
    ///
    /// Swept through the very integral the GPU volume is built from — the row sum
    /// `assets::build_prefix_tau` does, sharing its `tau_of` so the clamp cannot drift
    /// between the two — which is what makes this a test of the tip rather than of a
    /// restatement of it. Inside `|y| < 0.2` the profile is past 0.99 for every
    /// hardness and the clamped core takes over, so that is where the pin stops; the
    /// centre's saturation is the tip's one documented departure.
    #[test]
    fn the_round_tip_sweeps_to_the_profile_its_hardness_names() {
        const RES: u32 = crate::gpu::stroke::tips::ROUND_RES;
        for hardness in [0.0, 0.25, 0.5, 0.8, 0.95] {
            let h = 1.0 / (1.0 - hardness);
            let cov = round_coverage(hardness, RES);
            for row in 0..RES {
                let y = ((row as f32 + 0.5) / RES as f32 * 2.0 - 1.0).abs();
                if y < 0.2 {
                    continue;
                }
                // The row's optical depth, as the sweep sees it after the tip has
                // passed over: every column of the mask, in brush-local width.
                let tau: f32 = (0..RES)
                    .map(|x| crate::assets::tau_of(cov[(row * RES + x) as usize]))
                    .sum::<f32>()
                    * (2.0 / RES as f32);
                let laid = 1.0 - (-tau).exp();
                let want = 1.0 - y.powf(h);
                assert!(
                    (laid - want).abs() < 2e-3,
                    "hardness {hardness}: at y = {y:.4} the sweep lays {laid:.5}, \
                     not the {want:.5} its profile names",
                );
            }
        }
    }

    // --- tapers ----------------------------------------------------------

    /// A straight stroke `len` px long with a tapered brush of `radius`.
    fn tapered_record(radius: f32, start: f32, end: f32, len: f32) -> StrokeRecord {
        // Enough control points that the curve has spans to freeze part of, and
        // straight so arc length is the chord and the taper zones are easy to reason
        // about.
        let path: Vec<crate::path::ControlPoint> = (0..=12)
            .map(|i| crate::path::ControlPoint::at(Vec2::new(i as f32 / 12.0 * len, 0.0)))
            .collect();
        StrokeRecord {
            layer: crate::document::LayerId(0),
            brush: BrushParams {
                radius,
                drain: 0.0,
                start_taper_length: start,
                end_taper_length: end,
                ..BrushParams::default()
            },
            path,
            seed: 0,
        }
    }

    fn whole(rec: &StrokeRecord) -> Vec<Segment> {
        generate_segments_in(rec, flatten_tolerance(&rec.brush), StrokeSpans::whole(rec)).0
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
    /// last segments' radii, which is the distinction the ramp introduced: a segment's
    /// `radius` is its midpoint, and now that the cut no longer has to buy the first
    /// order the segment holding the point can be long enough that its midpoint is
    /// nowhere near one. What is at the point is `tip_at(0)`.
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

    /// **The property the ramp exists to have** (§6.2, [`Segment::ramp`]): consecutive
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
    fn assert_outline_is_continuous(segs: &[Segment]) {
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
    fn assert_tips_stay_positive(segs: &[Segment]) {
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

    /// The 2026-08-14 defect at the size that showed it: a radius-500 brush with long
    /// tapers (the repro capture's shape) drew its point as a comb of ~5 px sawteeth,
    /// because a segment swept at one radius and the cut could only make the step
    /// between them smaller, never zero.
    ///
    /// It also pins what the ramp *bought*: the cut no longer has to buy the first
    /// order, so a taper costs a logarithmic handful of segments instead of one per
    /// `0.7 px` of radius. The count is the scale-free one — see the sibling test that
    /// draws the same stroke a hundredth the size.
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
    /// absolute numbers move with any retuning of [`MAX_RADIUS_RAMP`], the
    /// independence does not.
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
        let path: Vec<crate::path::ControlPoint> = (0..=24)
            .map(|i| {
                let t = i as f32 / 24.0;
                let mut cp = crate::path::ControlPoint::at(Vec2::new(t * 4000.0, 0.0));
                cp.pressure = 0.05 + 0.95 * t;
                cp
            })
            .collect();
        let rec = StrokeRecord {
            layer: crate::document::LayerId(0),
            brush: BrushParams {
                radius: 500.0,
                drain: 0.0,
                ..BrushParams::default()
            },
            path,
            seed: 0,
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
        rec.brush.modulation = crate::document::Modulations::default();
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
        // A click has no length for a taper to run along at all, and still dots at
        // full size — the limit of the compression above rather than a special case:
        // zero length scales both zones to nothing, so the profile is 1 at the dab.
        let mut dot = tapered_record(radius, 6.0, 6.0, 0.0);
        dot.path.truncate(1);
        let segs = whole(&dot);
        assert_eq!(segs.len(), 1, "a click is one swept dab");
        assert_eq!(segs[0].radius, radius, "a tapered brush should still dot");
    }

    // --- the touch-down dab ------------------------------------------------

    /// A click leaves a dab **centred on the point that was pressed**, of the travel
    /// [`DAB_TRAVEL`] names.
    ///
    /// The centring is the whole of it. A click has no tangent — the fitter leaves a
    /// lone knot without one — so a dab swept *from* the point goes off in whatever
    /// direction the fallback happens to name, and reads as a short dash rather than a
    /// dot: a full tip's width of travel, all of it on one side, on a mark only two
    /// radii across. Swept about the point, the same travel is a dot a little wider
    /// than it is tall, and the arbitrary direction stops being visible at all.
    #[test]
    fn a_click_dabs_symmetrically_about_the_point() {
        let radius = 20.0;
        let at = Vec2::new(37.0, -11.0);
        let rec = record(
            BrushParams {
                radius,
                ..BrushParams::default()
            },
            &[at],
        );
        let segs = whole(&rec);
        assert_eq!(segs.len(), 1, "a click is one swept dab");
        let dab = segs[0];
        assert_eq!(dab.curvature, 0.0, "a dab does not bend");
        assert!(
            (dab.length - DAB_TRAVEL * radius).abs() < 1e-4,
            "the dab swept {} of the {} it owes",
            dab.length,
            DAB_TRAVEL * radius
        );
        let centre = dab.start + dab.dir * (dab.length * 0.5);
        assert!(
            (centre - at).length() < 1e-4,
            "the dab is centred at {centre:?}, not on the {at:?} that was pressed"
        );
    }

    /// The dab and the stroke are one continuum, not two cases: a stroke sweeps
    /// `max(travel, dab)`, so the mark grows out of the dot instead of replacing it.
    ///
    /// The jump this rules out was the visible one. A press deposited a dab; the first
    /// pixel of movement made the stroke "long enough" to stand on its own and
    /// deposited a twentieth of one, so the dot vanished the instant the hand moved
    /// and came back only once the stroke had travelled a tip's width.
    #[test]
    fn a_short_stroke_is_topped_up_to_a_whole_dab() {
        let radius = 20.0;
        let dab = DAB_TRAVEL * radius;
        for len in [0.0f32, 0.5, 2.0, 6.0, dab - 0.5, dab + 0.5, 40.0] {
            let rec = tapered_record(radius, 0.0, 0.0, len);
            let segs = whole(&rec);
            let travel: f32 = segs.iter().map(|s| s.length).sum();
            assert!(
                (travel - len.max(dab)).abs() < 0.05,
                "a {len}px stroke swept {travel}, not the {} it owes",
                len.max(dab)
            );
            // And the dwell is swept about the stroke's own midpoint, so it can only
            // fatten the mark symmetrically — never lead it off in one direction.
            if len < dab {
                let d = segs[0];
                let centre = d.start + d.dir * (d.length * 0.5);
                assert!(
                    (centre - Vec2::new(len * 0.5, 0.0)).length() < 0.05,
                    "a {len}px stroke's dab sits at {centre:?}, not on its midpoint"
                );
            }
        }
    }

    /// Nothing may freeze while the dab is still in play, and this is what says so.
    ///
    /// The dab is measured against the *whole* stroke's travel, exactly as the
    /// trailing taper is against its length, so a span frozen before the stroke has
    /// outrun its dab would keep a dab the commit does not draw — live == committed
    /// (§1.3) failing where it cannot be repainted. Held back until the frozen prefix
    /// alone is a dab long, which proves the whole stroke is.
    #[test]
    fn nothing_freezes_until_the_stroke_has_outrun_its_dab() {
        let radius = 60.0; // dab = 36px, so a stroke can be many spans and still owe one
        let untapered = |len: f32| {
            let rec = tapered_record(radius, 0.0, 0.0, len);
            let all = crate::path::span_count(rec.path.len());
            (safe_frozen(&rec, all), rec)
        };
        let (frozen, _) = untapered(20.0);
        assert_eq!(frozen, 0, "a stroke inside its own dab froze a span");
        let (frozen, rec) = untapered(600.0);
        assert!(frozen > 0, "a long stroke never froze anything");

        // And what it admits really is dab-free: the head it hands over renders the
        // same segments the commit does, which is the property the whole rule is for.
        let tol = flatten_tolerance(&rec.brush);
        let all = crate::path::span_count(rec.path.len());
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
        let one_pass = whole(&rec);
        assert_eq!(
            head.len() + tail.len(),
            one_pass.len(),
            "the split re-cut it"
        );
        for (i, (a, b)) in head.iter().chain(&tail).zip(&one_pass).enumerate() {
            assert_eq!(a.start, b.start, "segment {i}: start differs");
            assert_eq!(a.length, b.length, "segment {i}: length differs");
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
        let split: Vec<Segment> = head.into_iter().chain(tail).collect();
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
        let path: Vec<crate::path::ControlPoint> = (0..=12)
            .map(|i| {
                let t = i as f32 / 12.0 * sweep;
                crate::path::ControlPoint::at(Vec2::new(
                    curve_radius * t.sin(),
                    curve_radius * (1.0 - t.cos()),
                ))
            })
            .collect();
        StrokeRecord {
            layer: crate::document::LayerId(0),
            brush: BrushParams {
                radius,
                drain: 0.0,
                ..BrushParams::default()
            },
            path,
            seed: 0,
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
    /// ends. Under-reporting here is a clipped stroke: `affected_tiles` would leave a
    /// tile out of the render, and the dynamics loop would dispatch a rect too small
    /// for its own footprint.
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

    /// A tip's box has to contain the **square** it is swept over, not the disc
    /// inscribed in that square — and at a diagonal those are different boxes.
    ///
    /// This is the bug the `reach` field exists for. Every shape is integrated over
    /// brush-local `|x| ≤ 1, |y| ≤ 1`, so a stamp is free to paint the corners of its
    /// own mask; measured against the radius alone, a stroke running at 45° claimed a
    /// box `√2` too small on both axes, and `for_each_touched` then left tiles — or
    /// segments within a tile — out of the render. What it looks like is a stroke sliced
    /// off along a tile boundary, which no golden would attribute to the tip's width.
    ///
    /// Swept at a range of angles because the axis-aligned ones are exactly where the
    /// two answers agree, and those were the ones that always looked right.
    #[test]
    fn a_square_tip_claims_the_corners_it_can_paint() {
        let radius = 24.0f32;
        let stamp = BrushShape::Stamp(crate::assets::AssetId([7u8; 32]));
        assert_eq!(
            tip_reach(&BrushShape::default()),
            1.0,
            "a round tip is a disc"
        );
        for k in 0..16 {
            // Angles off the axes as well as on them, from the series (see
            // `sin_series`) so the case is the same on every platform.
            let theta = k as f64 * std::f64::consts::TAU / 16.0;
            let dir = Vec2::new(cos_series(theta) as f32, sin_series(theta) as f32);
            let mut b = BrushParams {
                radius,
                ..BrushParams::default()
            };
            b.shape = stamp;
            let rec = record(b, &[Vec2::ZERO, dir * 200.0]);
            for (i, s) in whole(&rec).iter().enumerate() {
                let (lo, hi) = coverage_bounds(s);
                let perp = Vec2::new(-s.dir.y, s.dir.x);
                let end = segment_end(s);
                // The four corners of the tip's square at each end of the travel —
                // every one of them a texel the deposit may reach.
                for base in [s.start, end] {
                    for sx in [-1.0f32, 1.0] {
                        for sy in [-1.0f32, 1.0] {
                            let c = base + s.dir * (sx * s.radius) + perp * (sy * s.radius);
                            // At exactly 45° the box is *tight* — the corner is on its
                            // edge — and the two sides reach `r·√2` by different
                            // arithmetic, so they land an ulp or so apart. A tolerance
                            // well under a texel is what says "contains" here; nothing
                            // downstream can resolve less, and both consumers add their
                            // own margin (`TILE_APRON`, `RECT_MARGIN`) on top.
                            const SLACK: f32 = 1e-3;
                            assert!(
                                c.x >= lo.x - SLACK
                                    && c.y >= lo.y - SLACK
                                    && c.x <= hi.x + SLACK
                                    && c.y <= hi.y + SLACK,
                                "angle {k}/16, segment {i}: the tip's corner {c:?} \
                                 falls outside its own coverage box {lo:?}..{hi:?}",
                            );
                        }
                    }
                }
            }
        }
    }

    // --- segment budget ----------------------------------------------------

    /// A stroke through `pts` with `brush`, as a path of plain full-pressure knots.
    fn record(brush: BrushParams, pts: &[Vec2]) -> StrokeRecord {
        StrokeRecord {
            layer: crate::document::LayerId(0),
            brush,
            path: pts
                .iter()
                .map(|p| crate::path::ControlPoint::at(*p))
                .collect(),
            seed: 0,
        }
    }

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

    /// A brush that manipulates paint, so the stroke takes the dynamics loop.
    fn smearing(radius: f32) -> BrushParams {
        use crate::document::BrushDynamics;
        BrushParams {
            radius,
            dynamics: BrushDynamics {
                lift: 0.8,
                deposit: 0.8,
                ..BrushDynamics::default()
            },
            ..BrushParams::default()
        }
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
        use crate::document::BrushDynamics;
        let at = |lift: f32, deposit: f32, charge: f32| {
            flatten_tolerance(&BrushParams {
                radius: 100.0,
                dynamics: BrushDynamics {
                    lift,
                    deposit,
                    charge,
                    ..BrushDynamics::default()
                },
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
                        radius: 20.0,
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
            // above. It used to bind at `0.02 / drain` = 4px, and now costs nothing at all —
            // for a quantity that is exact rather than merely finely sampled.
            (
                "straight, draining tip",
                118,
                record(
                    BrushParams {
                        drain: 0.005,
                        ..smearing(20.0)
                    },
                    &straight,
                ),
            ),
            // The taper, and **no longer the most expensive row in the table** — the
            // point of carrying the radius as a ramp. A segment holds the taper's
            // slope exactly, so the cut buys only the sagitta of a chord across the
            // profile's own curvature: `len·√(|r''|/8E)` pieces, a second-order term
            // where the two rules before it bought a first-order one. It was 211 when
            // the step was denominated in the radius *factor*, 121 when it was
            // denominated in px (which fixed a 500 px tip's sawteeth by charging
            // ~700 pieces a zone for them), and 15 now. Nothing about the curve is
            // driving this one — it is the same straight line as the 3-segment row
            // above.
            (
                "straight, tapered tip",
                15,
                record(
                    BrushParams {
                        radius: 20.0,
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
                        radius: 20.0,
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
                        radius: 80.0,
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
                        radius: 20.0,
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
                        radius: 80.0,
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

    // --- region measurement ----------------------------------------------

    /// A segment carrying only what the region measurements read.
    fn seg(start: Vec2, end: Vec2, radius: f32) -> Segment {
        let v = end - start;
        let length = v.length();
        Segment {
            start,
            dir: if length > 0.0 {
                v / length
            } else {
                Vec2::new(1.0, 0.0)
            },
            curvature: 0.0,
            radius,
            // A tip that holds still: these cases are about how the measurements
            // combine boxes, and a ramp would put a second variable in every box.
            ramp: 0.0,
            // A round tip's frame and reach, both the radius: these cases are about how
            // the measurements combine boxes, not about how wide one shape is.
            frame: radius,
            reach: radius,
            length,
            orient: 0.0,
            dist: 0.0,
            // The region measurements are geometry: they read the frame and the
            // radius and nothing else, so the paint rates are left at zero rather
            // than given values that would imply they were consulted.
            add: 0.0,
            lift: 0.0,
            deposit: 0.0,
            bleed: 0.0,
            tooth: 0.0,
        }
    }

    /// The union of every segment's — and firing window's — [`segment_bounds`], as
    /// [`chunk_segments`] accumulates it.
    fn measured(segments: &[Segment], fires: &[(usize, Segment)]) -> Option<(u32, u32)> {
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for s in segments.iter().chain(fires.iter().map(|(_, w)| w)) {
            let (slo, shi) = segment_bounds(s);
            (lo, hi) = (lo.min(slo), hi.max(shi));
        }
        lo.x.is_finite().then(|| region_of(lo, hi))
    }

    /// The per-tile segment lists cover exactly the tiles [`affected_tiles`] names, and
    /// a tile's list holds exactly the segments whose bounds reach it — in stroke
    /// order, which the `over` blend on the color target makes load-bearing.
    ///
    /// The swept path draws from these lists instead of drawing every segment into
    /// every tile, so an omission here is missing paint and a re-ordering is a
    /// different picture. Both are the kind of thing a golden would show as "the stroke
    /// looks a bit wrong" without saying why.
    #[test]
    fn the_per_tile_lists_hold_exactly_the_segments_that_reach_each_tile() {
        let tile = TILE_SIZE as f32;
        let segments: Vec<Segment> = (0..40)
            .map(|i| {
                let t = i as f32;
                seg(
                    Vec2::new(t * 31.0 - 200.0, (t * 0.4).sin() * 300.0),
                    Vec2::new((t + 1.0) * 31.0 - 200.0, ((t + 1.0) * 0.4).sin() * 300.0),
                    4.0 + (i % 5) as f32 * 9.0,
                )
            })
            .collect();

        let map = tiles_with_segments(&segments);
        assert_eq!(
            map.keys().copied().collect::<BTreeSet<_>>(),
            affected_tiles(&segments, &[]),
            "the two walks disagree on which tiles a stroke touches",
        );
        assert!(map.len() > 4, "not enough tiles to be an interesting case");

        for (coord, idx) in &map {
            assert!(
                idx.windows(2).all(|w| w[0] < w[1]),
                "tile {coord:?}'s segments are not in stroke order",
            );
            // The list against the membership test itself, segment by segment: a tile
            // is in a segment's block exactly when the segment is in the tile's list.
            for (i, s) in segments.iter().enumerate() {
                let (lo, hi) = segment_bounds(s);
                let inside = (lo.x / tile).floor() <= coord.x as f32
                    && coord.x as f32 <= (hi.x / tile).floor()
                    && (lo.y / tile).floor() <= coord.y as f32
                    && coord.y as f32 <= (hi.y / tile).floor();
                assert_eq!(
                    idx.contains(&(i as u32)),
                    inside,
                    "tile {coord:?} and segment {i} disagree about reaching one another",
                );
            }
        }

        // And the whole point: the listed pairs are far fewer than the product the
        // swept path used to shade.
        let listed: usize = map.values().map(Vec::len).sum();
        assert!(
            listed < map.len() * segments.len() / 4,
            "{listed} listed pairs against a {} product — the grouping is not buying \
             anything on this case",
            map.len() * segments.len(),
        );
    }

    /// [`chunk_segments`] decides where to cut a stroke by measuring the region a run
    /// of segments would need with [`region_of`], but the render that follows sizes
    /// the actual textures from [`region_rect`]. They are two ways of measuring one
    /// rectangle — bounding box versus enumerated tiles — so they have to agree
    /// exactly. If the bounding box ever under-reported, a piece would allocate past
    /// [`MAX_REGION_DIM`]; if it over-reported, strokes would be cut into more pieces
    /// than they need, each paying for its own region composite.
    #[test]
    fn the_chunker_measures_the_region_the_render_builds() {
        let tile = TILE_SIZE as f32;
        let cases: Vec<(&str, Vec<Segment>)> = vec![
            (
                "a dot",
                vec![seg(Vec2::new(10.0, 10.0), Vec2::new(10.5, 10.0), 4.0)],
            ),
            (
                "one tile-aligned span",
                vec![seg(Vec2::ZERO, Vec2::new(tile, 0.0), 1.0)],
            ),
            (
                "across the origin, into negative tiles",
                vec![seg(Vec2::new(-300.0, -140.0), Vec2::new(220.0, 90.0), 12.0)],
            ),
            (
                "a fat tip, whose radius reaches past its endpoints",
                vec![seg(Vec2::new(500.0, 500.0), Vec2::new(505.0, 500.0), 90.0)],
            ),
            (
                "several segments, extremes in different ones",
                vec![
                    seg(Vec2::new(0.0, 0.0), Vec2::new(120.0, 30.0), 3.0),
                    seg(Vec2::new(120.0, 30.0), Vec2::new(-90.0, 400.0), 20.0),
                    seg(Vec2::new(-90.0, 400.0), Vec2::new(700.0, -60.0), 8.0),
                ],
            ),
        ];
        for (what, segments) in cases {
            let want = region_rect(&affected_tiles(&segments, &[])).map(|r| (r.w, r.h));
            assert_eq!(
                measured(&segments, &[]),
                want,
                "region size disagrees for {what}"
            );
        }
        assert_eq!(measured(&[], &[]), None, "no segments is not a region");
    }

    /// **The accounting covers a firing window's reach back past the piece** — the
    /// 2026-08-11 regression, pinned where it is exact. A window is walked back
    /// along its crossing segment's own arc and can start up to a
    /// [`BLEED_TRAVEL_QUANTUM`] before the piece's first segment
    /// (`plan::bleed_fires`); the margin the segment boxes leave is one apron
    /// texel, so a bleeding tip wider than a few px reaches ground no segment box
    /// names whenever its box falls within a quantum of a tile origin. Both halves
    /// must take the windows: the tile walk (the region rectangle and the
    /// write-back follow it — a tile it misses is flux silently clipped and an
    /// apron/interior seam), and the chunker (a piece's region must hold everything
    /// the piece writes).
    #[test]
    fn a_windows_reach_back_is_in_the_tiles_and_the_region() {
        let tile = TILE_SIZE as f32;
        let radius = 40.0;
        let bq = BLEED_TRAVEL_QUANTUM * radius;
        // The piece's first segment, placed so its own coverage box starts 3 px
        // past a tile origin — inside the window's reach, outside the apron's.
        let x0 = 2.0 * tile + radius + TILE_APRON as f32 + 3.0;
        let s = seg(Vec2::new(x0, 8.0), Vec2::new(x0 + 50.0, 8.0), radius);
        // Its firing's window, one quantum of arc ending where the segment starts —
        // the shape `bleed_fires` emits for the first segment of a range.
        let w = seg(Vec2::new(x0 - bq, 8.0), Vec2::new(x0, 8.0), radius);
        let fires = vec![(0usize, w)];

        let without = affected_tiles(&[s], &[]);
        let with = affected_tiles(&[s], &fires);
        let window_tiles = affected_tiles(&[w], &[]);
        assert!(
            window_tiles.iter().any(|c| !without.contains(c)),
            "the window does not reach past the segment boxes — the case has gone \
             soft and pins nothing",
        );
        assert!(
            window_tiles.iter().all(|c| with.contains(c)),
            "a tile the window writes is missing from the walk",
        );
        // And the chunker measures the very region the render then builds from the
        // tiles — fires on both sides of the relation, like the segments always were.
        assert_eq!(
            measured(&[s], &fires),
            region_rect(&with).map(|r| (r.w, r.h)),
            "the chunker and the region disagree once the windows are counted",
        );
        assert_eq!(
            chunk_segments(&[s], &fires),
            vec![0..1],
            "one segment and its firing are one piece",
        );
    }

    /// What [`chunk_segments`] promises the loop: the pieces tile the stroke in order
    /// (so the sequence of segments the loop walks is unchanged — the whole reason
    /// cutting it is sound), and every piece actually fits the region bound the cut
    /// exists to respect.
    #[test]
    fn the_chunks_tile_the_stroke_and_each_one_fits() {
        // A stroke far longer than one region in both axes, and a fat tip whose own
        // footprint eats a good part of the budget.
        let segments: Vec<Segment> = (0..600)
            .map(|i| {
                let t = i as f32;
                let a = Vec2::new(t * 9.0 - 400.0, (t * 0.05).sin() * 1500.0);
                let b = Vec2::new((t + 1.0) * 9.0 - 400.0, ((t + 1.0) * 0.05).sin() * 1500.0);
                seg(a, b, 60.0)
            })
            .collect();
        let runs = chunk_segments(&segments, &[]);
        assert!(runs.len() > 1, "an oversized stroke should be cut up");

        let mut next = 0;
        for run in &runs {
            assert_eq!(run.start, next, "the pieces leave a gap or overlap");
            next = run.end;
            let (w, h) = measured(&segments[run.clone()], &[]).expect("a piece is never empty");
            assert!(
                w <= MAX_REGION_DIM && h <= MAX_REGION_DIM,
                "piece {run:?} needs a {w}x{h} region",
            );
            assert!(run.len() <= MAX_STAMPS, "piece {run:?} overruns the batch");
        }
        assert_eq!(next, segments.len(), "the pieces do not cover the stroke");
    }

    /// The floor the chunker cannot get under: one segment's own footprint. A brush
    /// whose tip fits is drawn by the loop however long the stroke gets, which is the
    /// whole point of cutting it into pieces; one whose tip does not is the only case
    /// left that degrades to the swept deposit.
    #[test]
    fn the_gate_admits_any_brush_whose_own_tip_fits() {
        let fits = |radius: f32| {
            let mut b = BrushParams {
                radius,
                ..BrushParams::default()
            };
            b.dynamics.lift = 0.5;
            segment_fits_region(&b, flatten_tolerance(&b))
        };
        assert!(fits(1.0), "a hairline tip fits");
        assert!(fits(120.0), "the largest tip the UI offers fits");
        assert!(
            !fits(MAX_REGION_DIM as f32),
            "a tip wider than the whole region cannot fit"
        );
    }
}
