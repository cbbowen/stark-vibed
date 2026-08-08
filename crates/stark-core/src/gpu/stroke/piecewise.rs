//! Continuous piecewise-linear functions of arc length, and the two operations a
//! swept strip needs from them: **pointwise min/max**, and the **merged knot set**
//! (§6.2).
//!
//! Nothing here touches the GPU — it is float arithmetic over a handful of knots,
//! which is what lets the strip builder's properties be pinned exactly rather than
//! eyeballed on a render.
//!
//! # Why a strip needs this at all
//!
//! A swept segment is drawn as geometry whose vertices carry the **range of travel**
//! each covers, and the fragment turns that range into deposit with a single prefix-τ
//! difference. The rasterizer interpolates between vertices *linearly*, so the
//! quantities a vertex carries have to be linear between the vertices too — which
//! means a vertex has to sit at every place where any of them bends. Miss one and the
//! error is not a rounding: it is a straight line drawn across a corner, and it prints
//! as a facet.
//!
//! So the builder describes each quantity as a [`Piecewise`] over arc length, asks for
//! [`merged_knots`] across all of them at once, and emits a vertex pair per knot. Every
//! interpolant is then exact by construction, and the vertex count follows the stroke's
//! own complexity rather than a fixed tessellation guess.
//!
//! # Why min and max in particular
//!
//! Where the tip is wider than the curve it travels, its footprint overlaps itself: one
//! canvas point is covered by more than one stretch of the stroke. Summing those
//! stretches' prefix differences double-counts the overlap, and picking one of them is
//! discontinuous. Taking the **earliest start and the latest end** — one merged
//! interval — is neither: it is continuous in position, monotone, and exact wherever
//! the coverage happens not to overlap (consecutive ranges telescope,
//! `[a,b] ∪ [b,c] = [a,c]`). Where it does overlap it counts the gap between passes as
//! if it were covered, which is the approximation this whole approach is built on.
//!
//! The min and max of two piecewise-linear functions are themselves piecewise-linear,
//! but only if the point where the two **cross** becomes a knot — that is the corner,
//! and it generally falls strictly inside a piece of both operands.

// The strip builder that spends all of this lands next; until it does, the only
// caller is the test module below. Scoped to this file and temporary on purpose —
// `clippy -D warnings` is a CI gate, and the alternative to saying so here is a red
// branch.
#![allow(dead_code)]

/// A continuous piecewise-linear function of one real variable, held as its knots and
/// **constant-extended** outside the outermost pair.
///
/// The extension is not a convenience: a quantity is asked for at positions past the
/// stretch of stroke that defined it (a tip's footprint reaches a radius beyond the
/// travel at each end), and the honest value there is the one the end knot carries —
/// the tip is not changing any more, because there is no more stroke for it to change
/// along.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Piecewise {
    /// Knots in strictly increasing `x`. Never empty.
    knots: Vec<(f32, f32)>,
}

impl Piecewise {
    /// A function through `knots`, which are sorted and de-duplicated here rather than
    /// being required to arrive that way — the builders that produce them work in the
    /// stroke's own terms (a taper's start, a cap's reach) and have no natural order
    /// between them.
    ///
    /// Duplicated abscissae keep the **first** value given, so a caller may state a
    /// preferred value at a join without having to know whether some other rule already
    /// put a knot there. Returns `None` for an empty set, which is a caller bug rather
    /// than a value: a function of no knots has no value anywhere.
    pub(super) fn new(knots: impl IntoIterator<Item = (f32, f32)>) -> Option<Self> {
        let mut knots: Vec<(f32, f32)> = knots.into_iter().filter(|k| k.0.is_finite()).collect();
        if knots.is_empty() {
            return None;
        }
        knots.sort_by(|a, b| a.0.total_cmp(&b.0));
        knots.dedup_by(|b, a| a.0 == b.0);
        Some(Self { knots })
    }

    /// The constant function — a quantity the stroke does not move.
    pub(super) fn constant(y: f32) -> Self {
        Self {
            knots: vec![(0.0, y)],
        }
    }

    /// The value at `x`, linear between knots and constant outside them.
    pub(super) fn evaluate(&self, x: f32) -> f32 {
        let n = self.knots.len();
        if x <= self.knots[0].0 {
            return self.knots[0].1;
        }
        if x >= self.knots[n - 1].0 {
            return self.knots[n - 1].1;
        }
        // The first knot strictly past `x`; the guards above put it in `1..n`.
        let i = self.knots.partition_point(|k| k.0 <= x);
        let (x0, y0) = self.knots[i - 1];
        let (x1, y1) = self.knots[i];
        let d = x1 - x0;
        if d <= 0.0 {
            return y0;
        }
        y0 + (y1 - y0) * ((x - x0) / d)
    }

    /// Where this function bends.
    pub(super) fn knots(&self) -> &[(f32, f32)] {
        &self.knots
    }

    /// The pointwise larger of the two, **exactly** — including the corner where they
    /// cross.
    ///
    /// Both operands are linear between consecutive merged knots, so the maximum is too
    /// *except* where they swap order inside such an interval. That crossing is a
    /// corner of the result and nothing else will put a knot there, so it is solved for
    /// and inserted. Without it the result is a straight line drawn across the corner —
    /// which is exactly the facet this module exists to keep out of the geometry.
    pub(super) fn pointwise_max(&self, other: &Self) -> Self {
        self.pointwise(other, true)
    }

    /// The pointwise smaller of the two, exactly — see [`pointwise_max`](Self::pointwise_max).
    pub(super) fn pointwise_min(&self, other: &Self) -> Self {
        self.pointwise(other, false)
    }

    fn pointwise(&self, other: &Self, want_max: bool) -> Self {
        let pick = |a: f32, b: f32| if want_max { a.max(b) } else { a.min(b) };
        let xs = merged_knots(&[self, other]);
        let mut out: Vec<(f32, f32)> = Vec::with_capacity(xs.len() * 2);
        for w in xs.windows(2) {
            let (a, b) = (w[0], w[1]);
            out.push((a, pick(self.evaluate(a), other.evaluate(a))));
            // Both are linear across `(a, b)`, so they cross at most once and only if
            // their difference changes sign strictly inside. Solved by linear
            // interpolation on that difference rather than by forming slopes, so a
            // near-parallel pair cannot divide by an almost-zero.
            let (d0, d1) = (
                self.evaluate(a) - other.evaluate(a),
                self.evaluate(b) - other.evaluate(b),
            );
            if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
                let t = d0 / (d0 - d1);
                if t > 0.0 && t < 1.0 {
                    let x = a + (b - a) * t;
                    out.push((x, pick(self.evaluate(x), other.evaluate(x))));
                }
            }
        }
        let last = *xs.last().expect("a merged knot set is never empty");
        out.push((last, pick(self.evaluate(last), other.evaluate(last))));
        Self::new(out).expect("a non-empty input yields a non-empty function")
    }
}

/// Every abscissa at which **any** of `fs` bends, in increasing order and without
/// duplicates.
///
/// This is the sample set a strip's vertices have to sit on: between two consecutive
/// entries every one of `fs` is linear, so a rasterizer interpolating linearly between
/// vertices placed here reproduces all of them exactly at once. Placing vertices on any
/// one function's knots alone would be exact for that one and wrong for the rest.
pub(super) fn merged_knots(fs: &[&Piecewise]) -> Vec<f32> {
    let mut xs: Vec<f32> = fs
        .iter()
        .flat_map(|f| f.knots.iter().map(|k| k.0))
        .collect();
    xs.sort_by(f32::total_cmp);
    xs.dedup();
    xs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pw(k: &[(f32, f32)]) -> Piecewise {
        Piecewise::new(k.iter().copied()).expect("knots")
    }

    /// The two ends extend as constants, and the middle interpolates — the shape every
    /// quantity a strip carries is described in.
    #[test]
    fn a_function_is_linear_between_its_knots_and_constant_outside_them() {
        let f = pw(&[(2.0, 2.0), (4.0, 6.0)]);
        assert_eq!(f.evaluate(-100.0), 2.0, "left of the first knot");
        assert_eq!(f.evaluate(2.0), 2.0);
        assert_eq!(f.evaluate(3.0), 4.0, "halfway");
        assert_eq!(f.evaluate(4.0), 6.0);
        assert_eq!(f.evaluate(100.0), 6.0, "right of the last knot");
    }

    /// A single knot is the constant function, which is what an unmodulated quantity
    /// arrives as — and the case an `evaluate` written around "find the bracketing
    /// pair" gets wrong.
    #[test]
    fn one_knot_is_a_constant() {
        let f = Piecewise::constant(3.0);
        for x in [-10.0f32, 0.0, 10.0] {
            assert_eq!(f.evaluate(x), 3.0);
        }
    }

    /// Knots may arrive in any order and may collide; the builders that make them work
    /// in the stroke's terms and have no order between them.
    #[test]
    fn knots_are_sorted_and_deduplicated() {
        let f = pw(&[(4.0, 6.0), (2.0, 2.0), (4.0, 99.0)]);
        assert_eq!(
            f.knots(),
            &[(2.0, 2.0), (4.0, 6.0)],
            "first value at a tie wins"
        );
        assert_eq!(f.evaluate(3.0), 4.0);
    }

    /// **The property the strip's vertex placement rests on**: between two consecutive
    /// merged knots every function is linear, so one vertex set serves all of them.
    #[test]
    fn merged_knots_are_the_union_of_every_bend() {
        let a = pw(&[(0.0, 0.0), (2.0, 1.0)]);
        let b = pw(&[(1.0, 5.0), (3.0, 5.0)]);
        assert_eq!(merged_knots(&[&a, &b]), vec![0.0, 1.0, 2.0, 3.0]);
        // Halfway across each merged interval, a straight line between the interval's
        // ends reproduces the function exactly — which is what the rasterizer will draw.
        for f in [&a, &b] {
            for w in merged_knots(&[&a, &b]).windows(2) {
                let mid = 0.5 * (w[0] + w[1]);
                let lerped = 0.5 * (f.evaluate(w[0]) + f.evaluate(w[1]));
                assert!(
                    (f.evaluate(mid) - lerped).abs() < 1e-6,
                    "a merged interval is not linear for one of the functions"
                );
            }
        }
    }

    /// **The corner has to become a knot.** Where two lines cross inside a piece, the
    /// max bends there and nothing else in either operand marks it — so a result that
    /// only carries the operands' own knots draws a straight line across the corner.
    #[test]
    fn a_crossing_becomes_a_knot_of_the_maximum() {
        // Two lines crossing at x = 1, y = 1: neither has a knot there.
        let a = pw(&[(0.0, 0.0), (2.0, 2.0)]);
        let b = pw(&[(0.0, 2.0), (2.0, 0.0)]);
        let m = a.pointwise_max(&b);
        assert!(
            m.knots().iter().any(|k| (k.0 - 1.0).abs() < 1e-6),
            "the crossing is not a knot of the max: {:?}",
            m.knots()
        );
        assert!((m.evaluate(1.0) - 1.0).abs() < 1e-6);
        // …and without it the max would read 1.0 at the corner but 2.0 at both ends,
        // so a straight line between the ends would be wrong by a whole unit here.
        assert!((m.evaluate(0.5) - 1.5).abs() < 1e-6);
    }

    /// Min and max agree with the operands wherever they are sampled, and are exactly
    /// piecewise-linear — checked by the same midpoint test the merged knots get, since
    /// that is the property the geometry actually depends on.
    #[test]
    fn min_and_max_agree_pointwise_and_stay_piecewise_linear() {
        let a = pw(&[(2.0, 2.0), (4.0, 6.0), (6.0, -1.0), (8.0, 5.0)]);
        let b = pw(&[(1.0, 2.0), (3.0, 6.0), (7.0, 5.0)]);
        for (got, want_max) in [(a.pointwise_max(&b), true), (a.pointwise_min(&b), false)] {
            for i in 0..=180 {
                let x = -1.0 + i as f32 * 0.05;
                let want = if want_max {
                    a.evaluate(x).max(b.evaluate(x))
                } else {
                    a.evaluate(x).min(b.evaluate(x))
                };
                assert!(
                    (got.evaluate(x) - want).abs() < 1e-5,
                    "at {x}: got {}, want {want}",
                    got.evaluate(x)
                );
            }
            for w in got.knots().windows(2) {
                let mid = 0.5 * (w[0].0 + w[1].0);
                let lerped = 0.5 * (w[0].1 + w[1].1);
                assert!(
                    (got.evaluate(mid) - lerped).abs() < 1e-5,
                    "the result bends between its own knots"
                );
            }
        }
    }

    /// Parallel lines never cross, and the solve must not invent a knot by dividing by
    /// an almost-zero — the case a slope-form intersection gets wrong.
    #[test]
    fn parallel_functions_add_no_knots() {
        let a = pw(&[(0.0, 0.0), (4.0, 4.0)]);
        let b = pw(&[(0.0, 1.0), (4.0, 5.0)]);
        let m = a.pointwise_max(&b);
        assert_eq!(m, b, "the max of two parallel lines is the upper one");
    }

    /// Touching without crossing is not a corner either: the max is still one of the
    /// two lines throughout, and an inserted knot there would be harmless but a sign
    /// the sign test is using `<=` where it means `<`.
    #[test]
    fn functions_that_touch_without_crossing_add_no_knots() {
        let a = pw(&[(0.0, 0.0), (2.0, 2.0), (4.0, 0.0)]);
        let b = pw(&[(0.0, 2.0), (2.0, 2.0), (4.0, 2.0)]);
        let m = a.pointwise_max(&b);
        for i in 0..=40 {
            let x = i as f32 * 0.1;
            let want = a.evaluate(x).max(b.evaluate(x));
            assert!((m.evaluate(x) - want).abs() < 1e-6, "at {x}");
        }
    }
}
