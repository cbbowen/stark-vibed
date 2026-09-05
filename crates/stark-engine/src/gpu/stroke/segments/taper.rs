//! The stroke's taper (§6.2): the radius profile at either end of a stroke, and how
//! finely a flattened edge inside a taper is cut so a straight radius ramp
//! ([`Sweep::radius_ramp`](super::Sweep::radius_ramp)) tracks the curve it stands for.
//!
//! Nothing here reads a rate or a sweep: [`generate_segments_in`](super::generate_segments_in)
//! asks [`Taper`] for a factor and a piece count, and that is the whole interface.

use stark_model::document::BrushParams;

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
/// ([`Sweep::radius_ramp`](super::Sweep::radius_ramp)) rather than as the curve it is — the only part of the taper's
/// shape a ramp does not already carry exactly.
const TAPER_MAX_CURVATURE: f32 = 3.0;

/// How far the drawn outline may sit from the true cone, in **canvas px**, where the
/// tip is too hard (or too thin) for its own falloff to hide anything.
///
/// A segment's tip is a straight ramp ([`Sweep::radius_ramp`](super::Sweep::radius_ramp)) across a profile that is
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
/// it: a quarter of the shoulder ([`shoulder_per_radius`](crate::gpu::stroke::budget::shoulder_per_radius)),
/// the same resolvable-feature bound
/// [`extent_cell`](crate::gpu::stroke::budget::extent_cell) coarsens against — an outline
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
// `1 ∓ ramp/2` positive — is structural rather than enforced ([`Sweep::radius_ramp`]).

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
pub(in crate::gpu::stroke) struct Taper {
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
    /// ([`shoulder_per_radius`](crate::gpu::stroke::budget::shoulder_per_radius)) — what lets the
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
    /// still be compressed ([`safe_frozen`](crate::gpu::stroke::safe_frozen)), so a
    /// partial range is one where both of those factors are exactly 1 — and the
    /// commit, which sees the whole stroke, computes the same 1 for it.
    pub(super) fn resolve(b: &BrushParams, total: Option<f32>) -> Self {
        let (start, end) = b.taper_px();
        let radius = b.size.max(0.5);
        let shoulder = crate::gpu::stroke::budget::shoulder_per_radius(&b.shape);
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
    pub(super) fn factor(&self, dist: f32) -> f32 {
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
    /// straight radius ramp has to be cut fine enough to track ([`Sweep::radius_ramp`](super::Sweep::radius_ramp)).
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
    /// carries the taper's slope exactly, as its ramp ([`Sweep::radius_ramp`](super::Sweep::radius_ramp)), and two
    /// adjacent segments agree on the radius at the knot they share — so the outline
    /// is continuous however coarse the cut. What is left is one second-order term:
    /// the ramp is a **chord** across a cubic profile, and the outline bows off it by
    /// the sagitta `|r''|·h²/8`.
    ///
    /// That is the whole rule. See the note above [`TAPER_OUTLINE_PX`] for why there
    /// is no companion bound on the ramp's own magnitude — near a taper's point it is
    /// a constant no subdivision can move, and everywhere else this bound has already
    /// made it small.
    pub(super) fn pieces(&self, dist: f32, len: f32) -> usize {
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

#[cfg(test)]
mod tests {
    use super::super::testing::{assert_outline_is_continuous, sweeps, tapered_record, whole};
    use super::super::{Sweep, generate_segments_in};
    use super::*;
    use crate::gpu::stroke::budget::flatten_tolerance;
    use crate::gpu::stroke::{StrokeSpans, safe_frozen};
    use stark_model::document::BrushShape;

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

    /// The one bound on the ramp that has to hold everywhere, and the reason
    /// `stamp_common::radius_ramp_scale` needs no clamp: `|ramp| < 2`, so the tip is positive
    /// at both ends of every segment. Structural rather than enforced — it follows
    /// from flooring both ends at half a px — so this checks the algebra rather than a
    /// rule that could be forgotten.
    fn assert_tips_stay_positive(segs: &[Sweep]) {
        for (i, s) in segs.iter().enumerate() {
            assert!(
                s.radius_ramp.abs() < 2.0,
                "segment {i} ramps by {}, which puts a tip at or past zero",
                s.radius_ramp,
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

    /// The load-bearing claim behind [`safe_frozen`]: for any prefix it
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
}
