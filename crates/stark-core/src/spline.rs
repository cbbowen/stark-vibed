//! Clamped cardinal **cubic** B-splines over `f32`, and the least-squares solve
//! [`path::PathFitter`](crate::path::PathFitter) drives (§6.2).
//!
//! The knot vector is fixed by two constraints, which together remove it as a thing
//! anyone has to represent:
//!
//! - **Clamped**: the first and last control points are repeated `DEGREE` times, so
//!   the curve starts and ends *on* them. The repeats are simulated by an index view
//!   (`CubicBSpline::knot_row`) rather than stored.
//! - **Cardinal**: the interior knots are equally spaced, so the curve is
//!   parameterized over `[0, num_spans()]` with span `k` covering `[k, k + 1]`, and a
//!   single basis matrix serves every span.
//!
//! Only what the stroke fitter uses is here: evaluation, and a windowed least-squares
//! solve with a frozen prefix, a held tail, and an optional curvature penalty. This
//! was a generic crate (`spline-fit`) with an EM fit, a closest-point search and a
//! monotonic assignment search; `PathFitter` declares the correspondence rather than
//! searching for it, so it used none of that. Monomorphizing to `f32` and to cubic
//! also drops `generic_const_exprs` — the `{P + 1}` arithmetic was the only use.

use nalgebra::{Cholesky, Const, Dyn, OMatrix, SMatrix, SVector};

/// Degree of the curve. Cubic: C² continuity, and a basis local enough that a sample
/// touches only four control points — which is what lets the fit solve a window
/// instead of the whole polygon.
const DEGREE: usize = 3;

/// Order = degree + 1: the number of control points supporting any one span, and the
/// dimension of the basis matrix.
const ORDER: usize = DEGREE + 1;

/// A clamped cardinal cubic B-spline in `D` dimensions.
pub struct CubicBSpline<const D: usize> {
    control_points: OMatrix<f32, Dyn, Const<D>>,
}

/// What one [`CubicBSpline::fit_channels`] is solved against: where each value sits on
/// the curve, what it is, and how much of the subject it stands for.
#[derive(Copy, Clone)]
pub struct Observations<'a, const E: usize> {
    /// Curve parameter each value is assigned to. The caller declares the
    /// correspondence; nothing here searches for it.
    pub ts: &'a [f32],
    pub values: &'a [[f32; E]],
    /// **How much of the thing being fitted each value stands for** — see
    /// `CubicBSpline::m_step` for why a fit over sampled values needs one at all.
    ///
    /// Empty means one each, which is the plain sum-over-values fit and what a caller
    /// whose values are already an even sampling of its subject wants. Otherwise the
    /// same length as `values`.
    pub weights: &'a [f32],
}

impl<'a, const E: usize> Observations<'a, E> {
    /// Values with no measure of their own — an even sampling, weighted one each.
    pub fn even(ts: &'a [f32], values: &'a [[f32; E]]) -> Self {
        Self {
            ts,
            values,
            weights: &[],
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("a B-spline needs at least 2 control points (got {0})")]
pub struct NotEnoughControlPoints(pub usize);

impl<const D: usize> CubicBSpline<D> {
    pub fn from_control_points(
        control_points: OMatrix<f32, Dyn, Const<D>>,
    ) -> Result<Self, NotEnoughControlPoints> {
        let m = control_points.nrows();
        if m < 2 {
            return Err(NotEnoughControlPoints(m));
        }
        Ok(Self { control_points })
    }

    /// Number of control points.
    pub fn num_control_points(&self) -> usize {
        self.control_points.nrows()
    }

    /// Number of polynomial spans; the spline is parameterized over `[0, num_spans()]`.
    pub fn num_spans(&self) -> usize {
        // The clamped view repeats each endpoint knot `DEGREE` times (that is,
        // `DEGREE - 1` extra copies per end), and a degree-p B-spline over v control
        // points has v - p fully supported spans.
        self.control_points.nrows() + 2 * (DEGREE - 1) - DEGREE
    }

    /// Row backing index `i` of the conceptual clamped control sequence, in which the
    /// first and last rows each appear `DEGREE` times. This view simulates the
    /// endpoint duplicates instead of storing them.
    fn knot_row(&self, i: usize) -> usize {
        i.saturating_sub(DEGREE - 1)
            .min(self.control_points.nrows() - 1)
    }

    /// Span index and local coordinate `u ∈ [0, 1]` for `t`, clamped to the domain.
    fn span_and_local(&self, t: f32) -> (usize, f32) {
        let spans = self.num_spans();
        let tf = t.clamp(0.0, spans as f32);
        let k = (tf.floor() as usize).min(spans - 1);
        (k, tf - k as f32)
    }

    /// The uniform B-spline basis matrix: entry `(a, i)` is the coefficient of `u^i`
    /// in the polynomial weight of conceptual control point `k + a` on any span `k`,
    /// with `u = t - k`. Thanks to the duplicating knot view, this one matrix serves
    /// every span.
    fn basis_matrix() -> SMatrix<f32, ORDER, ORDER> {
        // Cox–de Boor on a uniform integer knot vector, carried out on polynomial
        // coefficients (rows: basis index a; columns: powers of u):
        // N_a^q(u) = ((u + q - a) N_{a-1}^{q-1}(u) + (a + 1 - u) N_a^{q-1}(u)) / q.
        let mut m = SMatrix::<f32, ORDER, ORDER>::zeros();
        m[(0, 0)] = 1.0;
        for q in 1..=DEGREE {
            let mut next = SMatrix::<f32, ORDER, ORDER>::zeros();
            // Reciprocal taken in f64 and then narrowed, not divided in f32: `1/3` differs
            // in the last bit between the two, and this feeds every fitted control point.
            // Bit-identical output is what keeps goldens and replay stable (§9).
            let inv_q = (1.0 / q as f64) as f32;
            for a in 0..=q {
                if a >= 1 {
                    let c = (q - a) as f32;
                    for i in 0..q {
                        let v = m[(a - 1, i)];
                        next[(a, i)] += v * c;
                        next[(a, i + 1)] += v;
                    }
                }
                if a < q {
                    let c = (a + 1) as f32;
                    for i in 0..q {
                        let v = m[(a, i)];
                        next[(a, i)] += v * c;
                        next[(a, i + 1)] -= v;
                    }
                }
                for i in 0..=q {
                    next[(a, i)] *= inv_q;
                }
            }
            m = next;
        }
        m
    }

    /// The vector `[1, u, u², u³]`.
    fn u_powers(u: f32) -> SVector<f32, ORDER> {
        let mut powers = SVector::<f32, ORDER>::zeros();
        let mut acc = 1.0;
        for i in 0..ORDER {
            powers[i] = acc;
            acc *= u;
        }
        powers
    }

    /// The point on the curve at parameter `t`.
    pub fn evaluate(&self, t: f32) -> SVector<f32, D> {
        // De Boor's algorithm. Only the `DEGREE + 1` conceptual control points
        // k..=k+DEGREE support span k; their backing rows come from the duplicating
        // knot view. On the uniform (cardinal) integer knot vector the recurrence
        // weights reduce to `alpha = (p + u - a) / (p - r + 1)`, so the basis matrix
        // never has to be formed.
        let (k, u) = self.span_and_local(t);
        let mut d: [SVector<f32, D>; ORDER] = std::array::from_fn(|a| {
            let r = self.knot_row(k + a);
            SVector::<f32, D>::from_fn(|dim, _| self.control_points[(r, dim)])
        });
        for r in 1..=DEGREE {
            let denom = (DEGREE - r + 1) as f32;
            for a in (r..=DEGREE).rev() {
                let alpha = (DEGREE as f32 + u - a as f32) / denom;
                d[a] = d[a - 1] * (1.0 - alpha) + d[a] * alpha;
            }
        }
        d[DEGREE]
    }

    /// Least-squares control values for `E` per-point channels sampled at `ts`.
    ///
    /// Used both for the geometry itself (`E = 2`) and for the pen channels riding
    /// along with it (`E = 4`: pressure, tilt x/y, time). Deliberately *not* one fit
    /// in `D + E` dimensions — folding channels into the geometry would let a
    /// pressure ramp pull on the parameterization and distort the curve to buy error
    /// in a quantity that has no length. The geometry decides `ts` alone and the
    /// channels solve against it, which also means no weight is needed to reconcile
    /// pixels with whatever units the channels are in.
    ///
    /// The first `frozen` rows and the last `tail` rows of `prior` come back
    /// unchanged; only the window between them is solved for. `smoothing` charges the
    /// control polygon for curvature (see `Self::m_step`).
    ///
    /// `prior` may be **shorter** than the control polygon — the usual case for an
    /// incremental caller whose geometry has just grown and whose channels have not.
    /// The missing rows are seeded by repeating the last one, which only sets where
    /// the proximal ridge is centered: those rows are past `frozen`, so the solve
    /// determines them.
    ///
    /// # Panics
    ///
    /// Panics if `prior` has more rows than there are control points, if `obs`'s
    /// parameters and values differ in length, if a non-empty set of weights is not
    /// that same length, or if `frozen` exceeds the control-point count.
    pub fn fit_channels<const E: usize>(
        &self,
        obs: Observations<'_, E>,
        frozen: usize,
        tail: usize,
        prior: &OMatrix<f32, Dyn, Const<E>>,
        smoothing: f32,
    ) -> OMatrix<f32, Dyn, Const<E>> {
        let m = self.control_points.nrows();
        assert!(
            prior.nrows() <= m,
            "channel prior has {} rows for {m} control points",
            prior.nrows()
        );
        assert!(frozen <= m, "cannot freeze {frozen} of {m} channel rows");
        assert_eq!(
            obs.ts.len(),
            obs.values.len(),
            "every assigned parameter needs a value"
        );
        assert!(
            obs.weights.is_empty() || obs.weights.len() == obs.values.len(),
            "{} weights for {} values",
            obs.weights.len(),
            obs.values.len()
        );
        let have = prior.nrows();
        let grown =
            OMatrix::<f32, Dyn, Const<E>>::from_fn_generic(Dyn(m), Const::<E>, |j, d| {
                match (j < have, have) {
                    (true, _) => prior[(j, d)],
                    (false, 0) => 0.0,
                    (false, h) => prior[(h - 1, d)],
                }
            });
        self.m_step(obs, &grown, frozen, tail, smoothing)
    }

    /// The least-squares solve itself: normal equations over the free window, ridge-
    /// regularized towards `prior`, solved by Cholesky.
    ///
    /// **Every value carries a weight, and one each is a claim rather than a neutral
    /// default.** A plain sum minimizes the error *per value supplied*, which is the
    /// right objective only where the values are an even sampling of whatever is being
    /// fitted. Where they are not, the fit is to the sampling and not to the subject,
    /// and wherever the two disagree the sampling wins on sheer count. `weights` is how
    /// a caller states what each value stands for — see
    /// `path::arc_weights` for the case that forced it — and `n`
    /// becomes their sum, so the two knobs scaled by it below (the smoothing's average
    /// data pull, the ridge's floor) go on meaning what they meant.
    fn m_step<const E: usize>(
        &self,
        obs: Observations<'_, E>,
        prior: &OMatrix<f32, Dyn, Const<E>>,
        frozen: usize,
        tail: usize,
        smoothing: f32,
    ) -> OMatrix<f32, Dyn, Const<E>> {
        let Observations {
            ts,
            values: points,
            weights,
        } = obs;
        let basis = Self::basis_matrix();
        let m = prior.nrows();
        // The weight the system carries, which is the point count exactly when the
        // weights are the implicit ones.
        let n: f32 = if weights.is_empty() {
            points.len() as f32
        } else {
            weights.iter().sum()
        };
        if frozen + tail >= m || points.is_empty() {
            // Nothing left to solve for, or nothing to solve against. With no points
            // the normal equations are all ridge, whose solution is `prior` exactly —
            // and taking that here also keeps `lambda` (scaled by `n`) off zero, which
            // it could never escalate away from.
            return prior.clone();
        }
        // The normal equations are assembled over a **window**, not the whole polygon.
        //
        // Only rows `frozen..m - tail` are solved for, and a cubic B-spline's basis is
        // local: a point in span `k` touches rows `k - 2 ..= k + 1`, so any point that
        // reaches a free row touches nothing below `frozen - (ORDER - 1)`. Everything
        // before that is a frozen row coupled to no free one, contributing zero. An
        // `m x m` matrix therefore spends nearly all of itself on structural zeros —
        // 160KB of them at 200 control points, memset four times per sample, which was
        // most of the cost of a long stroke and the reason the per-update time grew
        // with the length of the whole thing rather than with the window.
        let base = frozen.saturating_sub(ORDER - 1);
        let w = m - base;
        let mut btb = OMatrix::<f32, Dyn, Dyn>::zeros(w, w);
        let mut btp = OMatrix::<f32, Dyn, Const<E>>::zeros_generic(Dyn(w), Const::<E>);
        for (i, (&t, p)) in ts.iter().zip(points).enumerate() {
            let (k, u) = self.span_and_local(t);
            // Cannot reach a free row, so contributes nothing to what is solved.
            if self.knot_row(k + ORDER - 1) < frozen {
                continue;
            }
            // Scales this value's whole row of the design matrix, which is what a
            // weighted least squares is: `Σ qᵢ‖residualᵢ‖²`.
            let q = weights.get(i).copied().unwrap_or(1.0);
            let wts = basis * Self::u_powers(u);
            for a in 0..ORDER {
                let ra = self.knot_row(k + a) - base;
                for b in 0..ORDER {
                    btb[(ra, self.knot_row(k + b) - base)] += q * wts[a] * wts[b];
                }
                for d in 0..E {
                    btp[(ra, d)] += q * wts[a] * p[d];
                }
            }
        }
        // Bending energy `Σ ‖P₍ⱼ₋₁₎ − 2Pⱼ + P₍ⱼ₊₁₎‖²`, weighted against the *average*
        // data pull per control point so the knob means the same thing whatever the
        // point count and polygon length are. Quadratic in the control points, so it
        // is just another symmetric band added to the normal matrix — and its target
        // is zero curvature, so the right-hand side is untouched.
        //
        // This is what stops the curve wandering where no point is assigned. Note it
        // lands in `btb` *before* the frozen block is folded into the right-hand side
        // below, so it also couples the first free control point to the frozen ones —
        // which is what makes the free tail continue smoothly out of the committed
        // prefix instead of being free to start off in any direction at all. Only the
        // triples that touch a solved row are in the window; the rest act on frozen
        // rows alone.
        if smoothing > 0.0 && m >= 3 {
            // Ratio in f64 then narrowed — see the note in `basis_matrix`.
            let sw = smoothing * ((n as f64 / m as f64) as f32);
            let c = [1.0f32, -2.0, 1.0];
            for j in (base + 1).max(1)..m - 1 {
                let idx = [j - 1 - base, j - base, j + 1 - base];
                for (a, &ca) in c.iter().enumerate() {
                    for (b, &cb) in c.iter().enumerate() {
                        btb[(idx[a], idx[b])] += sw * ca * cb;
                    }
                }
            }
        }
        // The free block of the system, with the frozen rows' contribution moved to
        // the right-hand side. With `frozen == 0` this is the whole system, unchanged.
        let free = m - frozen - tail;
        let mut lambda = n * f32::EPSILON.sqrt();
        for _ in 0..64 {
            let f0 = frozen - base;
            let mut lhs = btb.view((f0, f0), (free, free)).into_owned();
            let mut rhs = btp.rows(f0, free).into_owned();
            if f0 > 0 {
                rhs -= btb.view((f0, 0), (free, f0)) * prior.rows(base, f0);
            }
            if tail > 0 {
                rhs -= btb.view((f0, w - tail), (free, tail)) * prior.rows(m - tail, tail);
            }
            for j in 0..free {
                lhs[(j, j)] += lambda;
                for d in 0..E {
                    rhs[(j, d)] += prior[(frozen + j, d)] * lambda;
                }
            }
            if let Some(chol) = Cholesky::new(lhs) {
                let solved = chol.solve(&rhs);
                if frozen == 0 && tail == 0 {
                    return solved;
                }
                let mut out = prior.clone();
                out.rows_mut(frozen, free).copy_from(&solved);
                return out;
            }
            lambda *= 10.0;
        }
        unreachable!("ridge-regularized normal equations are positive definite")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spline<const D: usize>(pts: &[[f32; D]]) -> CubicBSpline<D> {
        let m =
            OMatrix::<f32, Dyn, Const<D>>::from_fn_generic(Dyn(pts.len()), Const::<D>, |i, j| {
                pts[i][j]
            });
        CubicBSpline::from_control_points(m).expect("enough knots")
    }

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol})");
    }

    /// Control points at x = 0..5 on the x-axis (domain [0, 7]). By the linear
    /// precision of the uniform basis, x(t) = t - 1 exactly on the interior spans
    /// t ∈ [2, 5], easing in from x=0 at t=0 and out to x=5 at t=7 at the clamped ends.
    fn line_spline() -> CubicBSpline<2> {
        let pts: Vec<[f32; 2]> = (0..6).map(|x| [x as f32, 0.0]).collect();
        spline(&pts)
    }

    fn wiggle_spline() -> CubicBSpline<2> {
        spline(&[
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, -2.0],
            [3.0, 2.0],
            [4.0, -2.0],
            [5.0, 0.0],
        ])
    }

    #[test]
    fn basis_is_a_partition_of_unity() {
        let m = CubicBSpline::<2>::basis_matrix();
        for i in 0..ORDER {
            let sum: f32 = (0..ORDER).map(|a| m[(a, i)]).sum();
            assert_close(sum, if i == 0 { 1.0 } else { 0.0 }, 1e-6);
        }
    }

    /// The Cox–de Boor construction must reproduce the textbook uniform cubic basis.
    #[test]
    fn basis_matches_uniform_cubic() {
        let m = CubicBSpline::<2>::basis_matrix();
        let u = 0.5f32;
        let expected = [
            (1.0 - u).powi(3) / 6.0,
            (3.0 * u.powi(3) - 6.0 * u.powi(2) + 4.0) / 6.0,
            (-3.0 * u.powi(3) + 3.0 * u.powi(2) + 3.0 * u + 1.0) / 6.0,
            u.powi(3) / 6.0,
        ];
        let powers = CubicBSpline::<2>::u_powers(u);
        for (a, e) in expected.iter().enumerate() {
            let got: f32 = (0..ORDER).map(|i| m[(a, i)] * powers[i]).sum();
            assert_close(got, *e, 1e-6);
        }
    }

    #[test]
    fn evaluate_interpolates_endpoints() {
        let s = wiggle_spline();
        let start = s.evaluate(0.0);
        let end = s.evaluate(s.num_spans() as f32);
        assert_close(start[0], 0.0, 1e-6);
        assert_close(start[1], 0.0, 1e-6);
        assert_close(end[0], 5.0, 1e-6);
        assert_close(end[1], 0.0, 1e-6);
    }

    #[test]
    fn evaluate_is_linear_on_interior_spans() {
        let s = line_spline();
        for i in 0..=12 {
            let t = 2.0 + i as f32 * 0.25;
            let v = s.evaluate(t);
            assert_close(v[0], t - 1.0, 1e-5);
            assert_close(v[1], 0.0, 1e-6);
        }
        assert_close(s.evaluate(0.0)[0], 0.0, 1e-6);
        assert_close(s.evaluate(7.0)[0], 5.0, 1e-6);
    }

    #[test]
    fn evaluate_of_constant_spline_is_constant() {
        let s = spline(&[[1.0, 2.0]; 5]);
        for i in 0..=24 {
            let v = s.evaluate(i as f32 * 0.25);
            assert_close(v[0], 1.0, 1e-6);
            assert_close(v[1], 2.0, 1e-6);
        }
    }

    /// Parameters spread across the domain, as a stand-in for a real assignment.
    fn spread(s: &CubicBSpline<2>, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| s.num_spans() as f32 * i as f32 / (n - 1) as f32)
            .collect()
    }

    /// A channel that *is* a spline over these knots must be recovered by the solve:
    /// sample it, fit the samples, get the control values back.
    ///
    /// To ~1%, not to rounding. The proximal ridge is `n · √ε`, and `√ε` is 3.4e-4 in
    /// f32 against 1.5e-8 in f64 — some 2300× more pull towards `prior` (zero here)
    /// for the same data. That bias, not the quality of the fit, sets this bound. It
    /// is measured in f32 because that is what the engine solves in; an f64 test would
    /// hold the solver to a tolerance no caller ever sees.
    #[test]
    fn channel_fitting_recovers_an_exact_channel() {
        let s = wiggle_spline();
        let m = s.num_control_points();
        let truth = OMatrix::<f32, Dyn, Const<1>>::from_fn_generic(Dyn(m), Const::<1>, |j, _| {
            0.3 + 0.7 * (j as f32 * 1.7).sin()
        });
        let basis = CubicBSpline::<2>::basis_matrix();
        let channel_at = |t: f32| {
            let (k, u) = s.span_and_local(t);
            let w = basis * CubicBSpline::<2>::u_powers(u);
            (0..ORDER)
                .map(|a| truth[(s.knot_row(k + a), 0)] * w[a])
                .sum::<f32>()
        };
        let ts = spread(&s, 60);
        let values: Vec<[f32; 1]> = ts.iter().map(|&t| [channel_at(t)]).collect();

        let empty = OMatrix::<f32, Dyn, Const<1>>::zeros_generic(Dyn(0), Const::<1>);
        let got = s.fit_channels(Observations::even(&ts, &values), 0, 0, &empty, 0.0);
        assert_eq!(got.nrows(), m);
        for j in 0..m {
            assert_close(got[(j, 0)], truth[(j, 0)], 1e-2);
        }
    }

    /// With nothing left to fit against — every point retired into the frozen prefix,
    /// which the incremental fitter reaches routinely — the solve is all ridge, and
    /// must hand the prior straight back rather than divide by a `lambda` that `n = 0`
    /// pinned to zero.
    #[test]
    fn a_solve_with_no_points_returns_the_prior() {
        let s = wiggle_spline();
        let m = s.num_control_points();
        let prior = OMatrix::<f32, Dyn, Const<2>>::from_fn_generic(Dyn(m), Const::<2>, |j, d| {
            (j * 2 + d) as f32
        });
        let got = s.fit_channels(Observations::even(&[], &[]), 0, 0, &prior, 0.0);
        assert_eq!(got, prior);
    }

    /// The frozen prefix comes back untouched, and a prior shorter than the
    /// (since-grown) control polygon is accepted rather than required.
    #[test]
    fn frozen_channel_rows_come_back_untouched() {
        let s = wiggle_spline();
        let m = s.num_control_points();
        let ts = spread(&s, 40);
        let values: Vec<[f32; 2]> = (0..ts.len())
            .map(|i| [i as f32 / ts.len() as f32, (i as f32 * 0.3).cos()])
            .collect();

        // A prior two rows short of the polygon: the tail is seeded, not required.
        let short =
            OMatrix::<f32, Dyn, Const<2>>::from_fn_generic(Dyn(m - 2), Const::<2>, |j, d| {
                (j + d) as f32 * 0.25
            });
        let frozen = 3;
        let got = s.fit_channels(Observations::even(&ts, &values), frozen, 0, &short, 0.0);
        assert_eq!(got.nrows(), m);
        for j in 0..frozen {
            for d in 0..2 {
                assert_eq!(got[(j, d)], short[(j, d)], "frozen row {j} moved");
            }
        }
        // And the free rows actually moved off their seed — the solve did something.
        let moved = (frozen..m).any(|j| (0..2).any(|d| got[(j, d)] != short[(j.min(m - 3), d)]));
        assert!(moved, "no free channel row was solved for");
    }

    /// The held tail is the other half of the window: `PathFitter` pins the stroke's
    /// last control point under the pointer and solves around it.
    #[test]
    fn a_held_tail_comes_back_untouched() {
        let s = wiggle_spline();
        let m = s.num_control_points();
        let ts = spread(&s, 30);
        let values: Vec<[f32; 2]> = ts.iter().map(|&t| [t, -t]).collect();
        let prior = OMatrix::<f32, Dyn, Const<2>>::from_fn_generic(Dyn(m), Const::<2>, |j, d| {
            (j + d) as f32
        });
        let got = s.fit_channels(Observations::even(&ts, &values), 1, 1, &prior, 0.0);
        for d in 0..2 {
            assert_eq!(got[(0, d)], prior[(0, d)], "frozen head moved");
            assert_eq!(got[(m - 1, d)], prior[(m - 1, d)], "held tail moved");
        }
    }

    #[test]
    fn too_few_control_points_is_an_error() {
        let one = OMatrix::<f32, Dyn, Const<2>>::zeros_generic(Dyn(1), Const::<2>);
        assert!(CubicBSpline::from_control_points(one).is_err());
    }
}
