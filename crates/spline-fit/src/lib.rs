#![feature(generic_const_exprs)]
// `generic_const_exprs` is incomplete by definition; it is used deliberately, for the
// `{P + 1}` basis-matrix arithmetic, and is one of the two reasons the workspace is
// pinned to nightly (see rust-toolchain.toml).
#![allow(incomplete_features)]
#![allow(type_alias_bounds)]
// Dimension loops here are over a const-generic `D` and index parallel arrays by
// coordinate (`point[d]`, `p[d]`). The iterator rewrite clippy wants obscures which
// axis is which, and there is rarely a single slice to iterate over.
#![allow(clippy::needless_range_loop)]
#![cfg_attr(feature = "nightly", feature(lazy_type_alias))]

use nalgebra::{Cholesky, Const, Dim, Dyn, OMatrix, SMatrix, SVector, convert};
use thiserror::Error;

mod incremental;
mod poly;
mod variable;
pub use incremental::IncrementalFit;
use poly::Poly;
pub use variable::Variable;

/// Knots with the the following constraints:
/// * Clamped: The first and last knots are duplicated so they appear `degree` times. This
///   forces the curve to start and end on these points respectively (a further copy would
///   instead pin the curve to a constant span at each end). The duplicates are simulated
///   by an index view over the stored knots ([`Self::knot_row`]) rather than stored.
/// * Cardinal: The internal (i.e. not duplicated endpoint) knots are equally-spaced.
struct ClampedCardinalKnots;

/// A clamped cardinal B-spline of degree N (order N+1) in dimension D.
pub struct ClampedCardinalBSpline<T: Variable, D: Dim, P: Dim> {
    control_points: OMatrix<T, Dyn, D>,
    _knots: ClampedCardinalKnots,
    degree: P,
}

#[derive(Debug, Error)]
pub enum FromControlPointsError {
    #[error("At least 2 control points are required (got {m})")]
    NotEnoughControlPoints { m: usize },
}

impl<T: Variable, D: Dim, P: Dim> ClampedCardinalBSpline<T, D, P> {
    pub fn degree(&self) -> usize {
        self.degree.value()
    }

    pub fn order(&self) -> usize {
        self.degree() + 1
    }

    pub fn from_control_points(
        degree: P,
        control_points: OMatrix<T, Dyn, D>,
    ) -> Result<Self, FromControlPointsError> {
        {
            let (m, _) = control_points.shape();
            if m < 2 {
                return Err(FromControlPointsError::NotEnoughControlPoints { m });
            }
        }
        let knots = ClampedCardinalKnots;
        Ok(ClampedCardinalBSpline {
            control_points,
            _knots: knots,
            degree,
        })
    }

    /// The control points, one per row.
    pub fn control_points(&self) -> &OMatrix<T, Dyn, D> {
        &self.control_points
    }

    /// Number of control points.
    pub fn num_control_points(&self) -> usize {
        self.control_points.nrows()
    }

    /// The parameter up to which the curve is determined by the first `frozen` control
    /// points: nothing done to the remaining control points — including *appending* new
    /// ones — can change the curve over `[0, frozen_until(frozen)]`.
    ///
    /// The result is negative when `frozen < 1`, since no parameter at all is pinned down
    /// by an empty prefix. This is the boundary an incremental fit retires points against:
    /// a point assigned at or before it has a final position and a final error.
    pub fn frozen_until(&self, frozen: usize) -> T {
        // Conceptual index `i` is backed by row `i - (degree - 1)` (clamped both ends), so
        // span `k` — supported by indices `k..=k + degree` — reads no row past `k + 1`, and
        // is fixed as soon as rows `0..=k + 1` are, i.e. once `frozen >= k + 2`. Spans
        // `0..=frozen - 2` are thus fixed, which is the parameter interval `[0, frozen - 1]`.
        // The end-of-sequence clamp only ever maps an index to a *lower* row, and appending
        // control points relaxes it, so the bound holds under growth too.
        convert::<f64, T>(frozen as f64 - 1.0).min(convert(self.num_spans() as f64))
    }

    /// Number of polynomial spans; the spline is parameterized over `[0, num_spans()]`.
    pub fn num_spans(&self) -> usize {
        // The clamped view repeats each endpoint knot `degree` times (that is,
        // `degree - 1` extra copies per end), and a degree-p B-spline over v control
        // points has v - p fully supported spans.
        let p = self.degree();
        self.control_points.nrows() + 2 * p.saturating_sub(1) - p
    }

    /// Row of `knots` backing index `i` of the conceptual clamped control sequence,
    /// in which the first and last rows each appear `degree` times. This view
    /// simulates the endpoint duplicates instead of storing them.
    fn knot_row(&self, i: usize) -> usize {
        let extra = self.degree().saturating_sub(1);
        i.saturating_sub(extra).min(self.control_points.nrows() - 1)
    }

    /// Span index and local coordinate `u ∈ [0, 1]` for parameter `t`, clamped to the domain.
    fn span_and_local(&self, t: T) -> (usize, T) {
        let spans = self.num_spans();
        let tf = t.max(T::zero()).min(convert(spans as f64));
        let k = tf.floor().to_usize().unwrap_or(0).min(spans - 1);
        (k, tf - convert(k as f64))
    }
}

#[derive(Clone, Copy)]
pub struct VariableTolerance<T> {
    abs: T,
}

impl<T: Variable> VariableTolerance<T> {
    pub fn new(abs: T) -> Self {
        assert!(
            abs > T::zero(),
            "absolute variable tolerance must be positive"
        );
        VariableTolerance { abs }
    }
}

impl<T: Variable> Default for VariableTolerance<T> {
    fn default() -> Self {
        VariableTolerance {
            abs: T::default_epsilon().sqrt(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct FitTolerance<T> {
    variable: VariableTolerance<T>,
    rel_metric: T,
    abs_metric: T,
    smoothing: T,
}

impl<T: Variable> FitTolerance<T> {
    pub fn new(abs_variable: T, rel_metric: T, abs_metric: T) -> Self {
        assert!(
            rel_metric >= T::zero(),
            "relative metric tolerance must be non-negative"
        );
        FitTolerance {
            variable: VariableTolerance::new(abs_variable),
            rel_metric,
            abs_metric,
            smoothing: T::zero(),
        }
    }

    /// Penalize curvature in the control polygon, as a fraction of the pull the data
    /// itself exerts (so it is scale-free, and `0` — the default — is the plain
    /// least-squares fit).
    ///
    /// **What this is for.** The fit minimizes the distance from each *point to the
    /// curve*, and nothing else. Nothing charges the curve for where it goes when no
    /// point is looking: a stretch of parameter that no point is assigned to is
    /// unconstrained, so the curve may take an arbitrary excursion through empty
    /// space and pay nothing, as long as every point still has some nearby piece of
    /// curve. The one-sided objective has no opinion about it, and neither does the
    /// monotone assignment. This shows up wherever the data thins out — a fast pen,
    /// or a run of points the assignment pools onto one parameter — and in an
    /// *incremental* fit the excursion is then frozen in and becomes permanent.
    ///
    /// The missing term is a regularizer, and the standard one is bending energy:
    /// `Σ ‖P₍ⱼ₋₁₎ − 2Pⱼ + P₍ⱼ₊₁₎‖²`, the discrete `∫‖C″‖²`. It is quadratic in the
    /// control points, so it adds straight into the normal equations as another
    /// symmetric band and costs the solve nothing. A control point with no data is
    /// then determined — it goes where its neighbours' straight continuation puts it
    /// — for any positive weight, while one the data does constrain is biased only
    /// in proportion to this fraction.
    ///
    /// It is a genuine trade: bending energy pulls a curved stroke very slightly
    /// flatter. Keep it small (a few percent) — it only has to beat *nothing*.
    pub fn with_smoothing(self, smoothing: T) -> Self {
        assert!(smoothing >= T::zero(), "smoothing must be non-negative");
        FitTolerance { smoothing, ..self }
    }
}

impl<T: Variable> Default for FitTolerance<T> {
    fn default() -> Self {
        let eps = T::default_epsilon().sqrt();
        FitTolerance {
            variable: VariableTolerance::default(),
            rel_metric: eps,
            abs_metric: eps,
            // Off: the plain least-squares fit, so a caller gets what it asked for
            // and nothing it did not. See [`FitTolerance::with_smoothing`].
            smoothing: T::zero(),
        }
    }
}

impl<T> From<FitTolerance<T>> for VariableTolerance<T> {
    fn from(val: FitTolerance<T>) -> Self {
        val.variable
    }
}

/// The part of an incremental fit that has stopped moving, and which a refit therefore
/// treats as given (see [`CardinalCubicBSpline::refit_monotonic`] and [`IncrementalFit`]).
///
/// [`Settled::none`] means nothing has settled, which makes a refit an ordinary
/// warm-started fit.
#[derive(Clone, Copy, Debug)]
pub struct Settled<T> {
    /// Number of leading control points held at their current values. The refit solves
    /// only for the rest, and the curve these determine — the parameters up to
    /// [`ClampedCardinalBSpline::frozen_until`] — cannot move.
    pub control_points: usize,
    /// A lower bound on the assignment: no point may be placed before this parameter.
    ///
    /// This is how *retired* points are accounted for. Once leading points are dropped
    /// from the fit — their assignment lies in the frozen region, so neither it nor their
    /// error can change — the points still being fit must go on respecting the monotone
    /// ordering against them, which is exactly `t >= after` for `after` the last retired
    /// parameter. Leave it non-positive while every point is still in the fit.
    pub after: T,
    /// Number of **trailing** control points held at their current values, the mirror of
    /// `control_points` at the other end.
    ///
    /// This is how an endpoint becomes a *constraint of the solve* rather than an
    /// override of it. Setting the last control point after a fit
    /// ([`ClampedCardinalBSpline::set_control_point`]) leaves every other row where the
    /// unconstrained solve put it, so the polygon has a step in it at the join and the
    /// curve swings through it — a kink that always sits at the end of the stroke,
    /// because that is the only place the override applies. Holding the row here instead
    /// lets the rest of the polygon solve *around* it.
    pub tail: usize,
}

impl<T: Variable> Settled<T> {
    /// Nothing is frozen and no point has been retired.
    pub fn none() -> Self {
        Settled {
            control_points: 0,
            after: T::zero(),
            tail: 0,
        }
    }
}

impl<T: Variable> Default for Settled<T> {
    fn default() -> Self {
        Self::none()
    }
}

impl<T: Variable, const D: usize, const P: usize> ClampedCardinalBSpline<T, Const<D>, Const<P>>
where
    [(); P + 1]: Sized,
{
    /// The uniform B-spline basis matrix: entry `(a, i)` is the coefficient of `u^i`
    /// in the polynomial weight of the conceptual control point `k + a` on any span
    /// `k`, with `u = t - k`. Thanks to the duplicating knot view, this single matrix
    /// serves every span. Stack-allocated when the degree is a compile-time constant.
    fn basis_matrix(&self) -> SMatrix<T, { P + 1 }, { P + 1 }> {
        let p = self.degree();
        // Cox–de Boor on a uniform integer knot vector, carried out on polynomial
        // coefficients (rows: basis index a; columns: powers of u):
        // N_a^q(u) = ((u + q - a) N_{a-1}^{q-1}(u) + (a + 1 - u) N_a^{q-1}(u)) / q.
        let mut m = SMatrix::zeros();
        m[(0, 0)] = T::one();
        for q in 1..=p {
            let mut next = SMatrix::zeros();
            let inv_q: T = convert(1.0 / q as f64);
            for a in 0..=q {
                if a >= 1 {
                    let c: T = convert((q - a) as f64);
                    for i in 0..q {
                        let v = m[(a - 1, i)];
                        next[(a, i)] += v * c;
                        next[(a, i + 1)] += v;
                    }
                }
                if a < q {
                    let c: T = convert((a + 1) as f64);
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

    /// The vector `[1, u, u^2, .., u^degree]`.
    fn u_powers(&self, u: T) -> SVector<T, { P + 1 }> {
        let mut powers = SVector::<_, { P + 1 }>::zeros();
        let mut acc = T::one();
        for i in 0..powers.len() {
            powers[i] = acc;
            acc *= u;
        }
        powers
    }

    fn evaluate_with_basis(&self, basis: &SMatrix<T, { P + 1 }, { P + 1 }>, t: T) -> SVector<T, D> {
        let (k, u) = self.span_and_local(t);
        let w = basis * self.u_powers(u);
        let mut out = SVector::<T, D>::zeros();
        for a in 0..w.len() {
            let r = self.knot_row(k + a);
            for d in 0..D {
                out[d] += self.control_points[(r, d)] * w[a];
            }
        }
        out
    }

    pub fn evaluate(&self, t: T) -> SVector<T, D> {
        // De Boor's algorithm. Only the `p + 1` conceptual control points k..=k+p
        // support span k; their backing rows come from the duplicating knot view.
        // On the uniform (cardinal) integer knot vector the recurrence weights
        // reduce to `alpha = (p + u - a) / (p - r + 1)`, so the whole basis matrix
        // never has to be formed.
        let p = self.degree();
        let (k, u) = self.span_and_local(t);
        let mut d: [SVector<T, D>; P + 1] = std::array::from_fn(|a| {
            let r = self.knot_row(k + a);
            SVector::<T, D>::from_fn(|dim, _| self.control_points[(r, dim)])
        });
        let pf: T = convert(p as f64);
        for r in 1..=p {
            let denom: T = convert((p - r + 1) as f64);
            for a in (r..=p).rev() {
                let alpha = (pf + u - convert(a as f64)) / denom;
                d[a] = d[a - 1] * (T::one() - alpha) + d[a] * alpha;
            }
        }
        d[p]
    }

    /// Overwrite control point `j`, overriding whatever the fit put there.
    ///
    /// The use this exists for is **pinning an endpoint**. The clamped end condition makes
    /// the first and last control points the curve's two ends, but a least-squares fit does
    /// not pin them to the data: a stretch of parameter with no point assigned to it costs
    /// nothing, so the curve is free to start before the first point and run on past the
    /// last. Where the caller knows where the curve must begin or end — a drawn stroke
    /// starts under the pointer, not near it — setting that control point and freezing it
    /// ([`Settled::control_points`], [`IncrementalFit::freeze`]) states the constraint, and
    /// the rest of the fit then solves around it exactly.
    ///
    /// This does not refit. Any assignment or error already computed describes the previous
    /// control points until the next fit runs.
    ///
    /// # Panics
    ///
    /// Panics if `j` is out of range.
    pub fn set_control_point(&mut self, j: usize, value: [T; D]) {
        assert!(
            j < self.control_points.nrows(),
            "control point {j} out of range"
        );
        for (d, v) in value.into_iter().enumerate() {
            self.control_points[(j, d)] = v;
        }
    }

    /// Evaluates the spline at each parameter of `ts`.
    ///
    /// Faster than repeated [`Self::evaluate`] calls: the basis matrix is computed
    /// once for the whole batch instead of per evaluation.
    pub fn evaluate_many(
        &self,
        ts: impl IntoIterator<Item = T>,
    ) -> impl Iterator<Item = SVector<T, D>> {
        let basis = self.basis_matrix();
        ts.into_iter()
            .map(move |t| self.evaluate_with_basis(&basis, t))
    }
}

/// A clamped cardinal cubic B-spline in dimension D.
pub type CardinalCubicBSpline<T: Variable, D: Dim> = ClampedCardinalBSpline<T, D, Const<3>>;

/// Per-span polynomials of the curve in the local coordinate `u = t - k`: the curve
/// `c` and its derivative `dc` per dimension, and `g = C' · C`. None of these depend
/// on a query point `p` — the half-derivative of the squared distance to `p` is
/// `f = C' · (C - p) = g - Σ_d p_d · dc_d` — so one cache serves any number of
/// closest-point queries against the same knots.
struct SpanPolys<T: Variable, const D: usize> {
    c: [Poly<T, 4>; D],
    dc: [Poly<T, 3>; D],
    g: Poly<T, 6>,
}

impl<T: Variable, const D: usize> CardinalCubicBSpline<T, Const<D>> {
    /// The per-span polynomials of the whole curve; see [`SpanPolys`].
    fn span_polys(&self) -> Vec<SpanPolys<T, D>> {
        let bm = self.basis_matrix();
        let basis: [Poly<T, 4>; 4] = std::array::from_fn(|a| Poly::from_fn(|i, _| bm[(a, i)]));
        (0..self.num_spans())
            .map(|k| {
                let c: [Poly<T, 4>; D] = std::array::from_fn(|d| {
                    let mut p = Poly::zeros();
                    for (a, b) in basis.iter().enumerate() {
                        p += b * self.control_points[(self.knot_row(k + a), d)];
                    }
                    p
                });
                let dc: [Poly<T, 3>; D] = std::array::from_fn(|d| poly::derivative(&c[d]));
                let mut g = Poly::<T, 6>::zeros();
                for d in 0..D {
                    g += poly::mul(&dc[d], &c[d]);
                }
                SpanPolys { c, dc, g }
            })
            .collect()
    }

    /// Per-span axis-aligned bounding box of the control points supporting each span,
    /// as `(min, max)` corners. By the convex-hull property the curve over span `k` lies
    /// within the hull — hence the box — of its `degree + 1` control points, so the
    /// squared distance from a query point to this box is a lower bound on the squared
    /// distance to the curve over that span. Cheap (no root-finding), and the bound drives
    /// the range pruning in [`Self::best_ordered_assignment`].
    fn span_control_bboxes(&self) -> Vec<([T; D], [T; D])> {
        let order = self.degree() + 1;
        (0..self.num_spans())
            .map(|k| {
                let mut lo = [T::zero(); D];
                let mut hi = [T::zero(); D];
                for a in 0..order {
                    let r = self.knot_row(k + a);
                    for d in 0..D {
                        let v = self.control_points[(r, d)];
                        if a == 0 {
                            lo[d] = v;
                            hi[d] = v;
                        } else {
                            lo[d] = lo[d].min(v);
                            hi[d] = hi[d].max(v);
                        }
                    }
                }
                (lo, hi)
            })
            .collect()
    }

    /// The squared distance from `point` to the curve at parameter `t`.
    fn distance_sq(&self, spans: &[SpanPolys<T, D>], t: T, point: &[T; D]) -> T {
        let (k, u) = self.span_and_local(t);
        let q = spans[k].c.map(|c| poly::eval(&c, u));
        q.into_iter()
            .zip(point.iter().cloned())
            .map(|(q, p)| (q - p).powi(2))
            .fold(T::zero(), T::add)
    }

    /// [`Self::locally_closest_points`] against a precomputed span-polynomial cache.
    fn locally_closest_in(
        &self,
        spans: &[SpanPolys<T, D>],
        point: &[T; D],
        tol: VariableTolerance<T>,
    ) -> Vec<(T, T)> {
        self.locally_closest_in_range(spans, point, 0, spans.len(), tol)
    }

    /// [`Self::locally_closest_in`] restricted to the contiguous span range
    /// `[span_lo, span_hi)`. The caller must have proven — via a geometric distance bound
    /// — that no locally-closest point *needed by the assignment* lies outside the range;
    /// this method then skips root-finding in the pruned prefix/suffix of spans.
    ///
    /// Exactness rests on two facts. (1) Every minimum strictly inside the range is
    /// classified identically to the full search: `f` has constant sign between
    /// consecutive critical points, and the range's first/last critical points bracket
    /// the same sign intervals the full search would sample. (2) The range ends
    /// `span_lo`/`span_hi` act as reportable minima only when they coincide with the true
    /// domain ends (`0` / `num_spans`); an interior cut is never itself reported, and a
    /// beneficial minimum can never sit on such a cut (that would put the curve near
    /// `point` at the cut knot, making the adjacent pruned span beneficial too — a
    /// contradiction). With the full range this reduces to the unbounded search exactly.
    fn locally_closest_in_range(
        &self,
        spans: &[SpanPolys<T, D>],
        point: &[T; D],
        span_lo: usize,
        span_hi: usize,
        tol: VariableTolerance<T>,
    ) -> Vec<(T, T)> {
        // `f = C' · (C - point)` is half the derivative of the squared distance; its
        // roots are the critical points. The scale that sets `sign_tol` is taken over all
        // spans (cheap, no root-finding) so it — and thus the classification — does not
        // depend on the search range.
        let f_at_coeffs = |sp: &SpanPolys<T, D>| -> Poly<T, 6> {
            let mut f = sp.g;
            for d in 0..D {
                for i in 0..3 {
                    f[i] -= sp.dc[d][i] * point[d];
                }
            }
            f
        };
        let mut f_scale = T::zero();
        for sp in spans.iter() {
            f_scale = f_at_coeffs(sp).iter().fold(f_scale, |m, c| m.max(c.abs()));
        }

        // Critical points within the searched span range only.
        let mut roots: Vec<T> = Vec::new();
        for k in span_lo..span_hi {
            let f = f_at_coeffs(&spans[k]);
            let start = roots.len();
            poly::roots_in_unit_interval(&f, tol.abs, &mut roots);
            for r in &mut roots[start..] {
                *r += convert::<f64, T>(k as f64);
            }
        }

        // Candidate minimizers: interior critical points plus the range ends,
        // deduplicated (a root can be reported by both spans sharing a knot).
        roots.sort_by(|x, y| x.partial_cmp(y).expect("parameters are finite"));
        let t_lo: T = convert(span_lo as f64);
        let t_hi: T = convert(span_hi as f64);
        let merge_tol = T::default_epsilon().sqrt() * convert::<f64, T>(8.0);
        let mut candidates = vec![t_lo];
        for r in roots {
            if r - *candidates.last().expect("nonempty") > merge_tol && t_hi - r > merge_tol {
                candidates.push(r);
            }
        }
        candidates.push(t_hi);

        // Since the candidates include every critical point in the range, f has constant
        // sign between consecutive candidates; sample it at the interval midpoints. A
        // candidate is a local minimum iff the distance is non-increasing into it and
        // non-decreasing out of it.
        let sign_tol = f_scale * T::default_epsilon() * convert::<f64, T>(64.0);
        let half = convert::<f64, T>(0.5);
        let f_at = |t: T| {
            let (k, u) = self.span_and_local(t);
            let mut v = poly::eval(&spans[k].g, u);
            for d in 0..D {
                v -= point[d] * poly::eval(&spans[k].dc[d], u);
            }
            v
        };
        let signs: Vec<i8> = candidates
            .windows(2)
            .map(|w| {
                let v = f_at((w[0] + w[1]) * half);
                if v > sign_tol {
                    1
                } else if v < -sign_tol {
                    -1
                } else {
                    0
                }
            })
            .collect();

        let left_is_end = span_lo == 0;
        let right_is_end = span_hi == spans.len();
        let mut minima = Vec::new();
        for (j, &t) in candidates.iter().enumerate() {
            // A range end that is not a true domain end is an artificial cut, never a
            // minimum in its own right.
            if (j == 0 && !left_is_end) || (j == candidates.len() - 1 && !right_is_end) {
                continue;
            }
            let decreasing_into = j == 0 || signs[j - 1] <= 0;
            let increasing_out = j == candidates.len() - 1 || signs[j] >= 0;
            if decreasing_into && increasing_out {
                minima.push((t, self.distance_sq(spans, t, point)));
            }
        }
        minima
    }

    /// Points on the spline that are locally closest to the provided `point`.
    ///
    /// Domain endpoints count as locally closest when the distance increases into the
    /// interior, so the result is never empty (the globally closest point is always
    /// included).
    ///
    /// Returns: an iteration over all tuples `(t, e)`, ascending in `t`, where each `t`
    /// locally minimizes the squared distance `e` to `point`.
    pub fn locally_closest_points(
        &self,
        point: &[T; D],
        tol: VariableTolerance<T>,
    ) -> impl Iterator<Item = (T, T)> + use<T, D> {
        self.locally_closest_in(&self.span_polys(), point, tol)
            .into_iter()
    }

    /// For each point, the parameters of every critical point of the squared-distance
    /// derivative `f`, across all spans, ascending — i.e. the raw root-finder output
    /// *before* the (cheap) minima classification of [`Self::locally_closest_points`].
    ///
    /// The per-span root-finding this performs is ~90% of an E-step (see PERF.md), so
    /// this method isolates that hot work for benchmarking.
    pub fn all_critical_points(&self, points: &[[T; D]], tol: VariableTolerance<T>) -> Vec<Vec<T>> {
        let spans = self.span_polys();
        points
            .iter()
            .map(|p| {
                let mut out = Vec::new();
                for (k, sp) in spans.iter().enumerate() {
                    let mut f = sp.g;
                    for d in 0..D {
                        for i in 0..3 {
                            f[i] -= sp.dc[d][i] * p[d];
                        }
                    }
                    let start = out.len();
                    poly::roots_in_unit_interval(&f, tol.abs, &mut out);
                    for r in &mut out[start..] {
                        *r += convert::<f64, T>(k as f64);
                    }
                }
                out
            })
            .collect()
    }

    /// The locally-closest point (a local minimum of the squared distance to `point`)
    /// with the smallest parameter strictly greater than `after`, or `None` when none
    /// exists. Passing `after < 0` makes the starting domain endpoint `t = 0` eligible.
    ///
    /// This is the lazy counterpart of [`Self::locally_closest_points`]: walking it from
    /// `after < 0` reproduces that method's full list, but it stops at the *first* minimum
    /// it finds. Because finding the next critical point (`poly::first_root_after`) is
    /// much cheaper than finding all of them, it is the primitive to reach for whenever
    /// only the next locally-closest point is needed rather than the whole set.
    pub fn next_locally_closest_point(
        &self,
        point: &[T; D],
        after: T,
        tol: VariableTolerance<T>,
    ) -> Option<(T, T)> {
        self.next_locally_closest_in(&self.span_polys(), point, after, tol)
    }

    /// [`Self::next_locally_closest_point`] against a precomputed span-polynomial cache.
    fn next_locally_closest_in(
        &self,
        spans: &[SpanPolys<T, D>],
        point: &[T; D],
        after: T,
        tol: VariableTolerance<T>,
    ) -> Option<(T, T)> {
        let n_spans = spans.len();
        let t_end: T = convert(n_spans as f64);
        let half = convert::<f64, T>(0.5);

        // Per-span half-derivative of the squared distance, `f = g - Σ_d p_d · dc_d`
        // (see [`SpanPolys`]). Roots of `f` are the critical points; the sign of `f`
        // between consecutive critical points tells us which are minima. Built on demand so
        // the lazy scan below forms `f` only for the spans it reaches, and the scale below
        // takes its exact max over all spans without ever allocating an all-span buffer.
        let f_of = |k: usize| -> Poly<T, 6> {
            let mut f = spans[k].g;
            for d in 0..D {
                for i in 0..3 {
                    f[i] -= spans[k].dc[d][i] * point[d];
                }
            }
            f
        };
        let f_scale = (0..n_spans).fold(T::zero(), |m, k| {
            f_of(k).iter().fold(m, |m, c| m.max(c.abs()))
        });
        let sign_tol = f_scale * T::default_epsilon() * convert::<f64, T>(64.0);
        // Critical points within `merge_tol` of each other or of a domain endpoint are
        // treated as coincident, matching the deduplication in `locally_closest_in`.
        let merge_tol = T::default_epsilon().sqrt() * convert::<f64, T>(8.0);
        let sign = |t: T| -> i8 {
            let (k, u) = self.span_and_local(t);
            let v = poly::eval(&f_of(k), u);
            if v > sign_tol {
                1
            } else if v < -sign_tol {
                -1
            } else {
                0
            }
        };

        // The smallest critical point strictly greater than `x` (and clear of an
        // endpoint), or `None`. Scans spans left to right, finding at most one root per
        // span via `first_root_after` — the far spans are never touched once a critical
        // point turns up early.
        let next_crit = |x: T| -> Option<T> {
            let start = self.span_and_local(x.max(T::zero())).0;
            for k in start..n_spans {
                let base: T = convert(k as f64);
                let mut lo = x - base;
                let fk = f_of(k);
                while let Some(r) = poly::first_root_after(&fk, lo, tol.abs) {
                    let t = base + r;
                    // Fold critical points that land on a domain endpoint into that
                    // endpoint, and skip anything not clearly past `x`.
                    if t <= merge_tol || t_end - t <= merge_tol || t - x <= merge_tol {
                        lo = r;
                        continue;
                    }
                    return Some(t);
                }
            }
            None
        };

        // A critical point `c` is a local minimum when the distance is non-increasing
        // into it (`left_sign <= 0`) and non-decreasing out of it (`right_sign >= 0`);
        // the domain endpoints drop the vacuous side. We advance one critical point at a
        // time, carrying the sign of the interval to its left.
        let mut c = next_crit(after);
        let mut left_sign;
        if after < T::zero() {
            // The endpoint `t = 0` is the first candidate; it minimizes iff the distance
            // increases into the interior.
            let right = c.unwrap_or(t_end);
            let s = sign(right * half);
            if s >= 0 {
                return Some((T::zero(), self.distance_sq(spans, T::zero(), point)));
            }
            left_sign = s;
        } else {
            let right = c.unwrap_or(t_end);
            left_sign = sign((after.max(T::zero()) + right) * half);
        }
        loop {
            match c {
                None => {
                    // The far endpoint minimizes iff the distance is non-increasing into
                    // it; only report it when it actually lies past `after`.
                    return (left_sign <= 0 && after < t_end)
                        .then(|| (t_end, self.distance_sq(spans, t_end, point)));
                }
                Some(cc) => {
                    let c_next = next_crit(cc);
                    let right_sign = sign((cc + c_next.unwrap_or(t_end)) * half);
                    if left_sign <= 0 && right_sign >= 0 {
                        return Some((cc, self.distance_sq(spans, cc, point)));
                    }
                    left_sign = right_sign;
                    c = c_next;
                }
            }
        }
    }

    /// For each point, the contiguous span range `[lo, hi)` its closest-point search may
    /// safely be confined to. A span is *pruned* only when a lower bound on the total
    /// error of every assignment that places the point there exceeds a feasible
    /// incumbent — so no minimum belonging to an optimal assignment is ever dropped, and
    /// [`Self::best_ordered_assignment`]'s result is unchanged.
    ///
    /// The lower bound decomposes over the points. For point `i` placed in span `k`:
    /// `Σ_{j<i} minlb_j(0) + lb(k, p_i) + Σ_{j>i} minlb_j(k)`, where `lb(k, p)` is the
    /// bounding-box distance of span `k` and `minlb_j(k) = min_{k'≥k} lb(k', p_j)` respects
    /// the monotone constraint (points after `i` sit in spans `≥ k`). Each term is a valid
    /// lower bound, so the sum is; if it exceeds the incumbent the span cannot host an
    /// optimal placement of point `i`. Spans that survive form a contiguous range around
    /// the point's neighborhood; a point with none (numerically degenerate) falls back to
    /// the full domain.
    ///
    /// `after` is the assignment floor (see [`Settled::after`]); it enters only through the
    /// incumbent, which must be a *feasible* assignment of the constrained problem to be an
    /// upper bound on its optimum. The per-span lower bounds are unconstrained, hence still
    /// lower bounds once the floor rules assignments out, so the pruning stays sound.
    fn search_ranges(
        &self,
        spans: &[SpanPolys<T, D>],
        points: &[[T; D]],
        lb: &impl Fn(usize, &[T; D]) -> T,
        n_spans: usize,
        tol: VariableTolerance<T>,
        after: T,
    ) -> Vec<(usize, usize)> {
        let n = points.len();
        // The suffix-minimum bounding-box distance for a point: `minlb[k] = min_{k'≥k}`.
        let min_lb_row = |p: &[T; D]| -> Vec<T> {
            let mut row = vec![T::zero(); n_spans];
            for k in (0..n_spans).rev() {
                let here = lb(k, p);
                row[k] = if k + 1 < n_spans {
                    here.min(row[k + 1])
                } else {
                    here
                };
            }
            row
        };

        // A cheap feasible monotone assignment gives an incumbent upper bound on the
        // optimum. Each point takes the cheaper of pinning at the predecessor's parameter
        // or its first locally-closest point beyond it (found lazily — the far spans go
        // untouched, and `f` is built on demand rather than into a per-call all-span Vec).
        // Looser than the true optimum, but only affects how much we prune. Starting the
        // walk at the floor keeps the incumbent feasible when one is in force.
        let mut incumbent = T::zero();
        let mut prev = if after > T::zero() {
            after
        } else {
            convert::<f64, T>(-1.0)
        };
        for p in points {
            let mut best: Option<(T, T)> =
                (prev >= T::zero()).then(|| (prev, self.distance_sq(spans, prev, p)));
            if let Some((t, e)) = self.next_locally_closest_in(spans, p, prev, tol)
                && best.is_none_or(|(_, be)| e < be)
            {
                best = Some((t, e));
            }
            let (t, e) = best.expect("a nonempty domain always has a closest point");
            incumbent += e;
            prev = t;
        }
        // A little slack keeps borderline spans (where the bound rounds against us),
        // erring toward the exact full-search result; over-keeping only costs speed.
        let ceil =
            incumbent + incumbent * T::default_epsilon().sqrt() + T::default_epsilon().sqrt();

        // Each point's suffix-min bounding-box row, formed once and shared by both passes.
        let rows: Vec<Vec<T>> = points.iter().map(min_lb_row).collect();

        // Pass 1: total suffix-min bound per span and each point's global minimum.
        let mut tot = vec![T::zero(); n_spans];
        let mut global_min_lb = vec![T::zero(); n];
        for (i, row) in rows.iter().enumerate() {
            global_min_lb[i] = row[0];
            for k in 0..n_spans {
                tot[k] += row[k];
            }
        }

        // Pass 2: for each point, the beneficial spans, using running cumulative sums for
        // the prefix (`Σ_{j<i}`) terms so the suffix `Σ_{j>i}` follows from `tot`.
        let mut cum = vec![T::zero(); n_spans];
        let mut prefix_lb = T::zero();
        let mut ranges = Vec::with_capacity(n);
        for (i, p) in points.iter().enumerate() {
            let row = &rows[i];
            let (mut lo, mut hi) = (None, 0usize);
            for k in 0..n_spans {
                let after = (tot[k] - cum[k] - row[k]).max(T::zero());
                let bound = prefix_lb + lb(k, p) + after;
                if bound <= ceil {
                    lo.get_or_insert(k);
                    hi = k + 1;
                }
            }
            ranges.push(match lo {
                Some(lo) => (lo, hi),
                None => (0, n_spans),
            });
            for k in 0..n_spans {
                cum[k] += row[k];
            }
            prefix_lb += global_min_lb[i];
        }
        ranges
    }

    /// Optimal monotonic assignment from points to variables.
    ///
    /// Returns: a tuple of the assignments for each point and the total squared error.
    pub fn best_ordered_assignment(
        &self,
        points: &[[T; D]],
        tol: VariableTolerance<T>,
    ) -> (Vec<T>, T) {
        self.best_ordered_assignment_after(points, tol, T::zero())
    }

    /// [`Self::best_ordered_assignment`] with the assignment floored at `after`: no point
    /// may be placed before that parameter (see [`Settled::after`]). A non-positive floor
    /// is no constraint at all and takes the identical path as the unfloored solver.
    ///
    /// The floor is enforced end to end rather than by post-filtering: the pruning's
    /// incumbent walks from it, each point's span range is clipped to it, the floor itself
    /// joins the candidate set of every point whose range reaches it (it may well be the
    /// constrained optimum, and it is not a local minimum the search would report), and
    /// both the DP's bands and the tied-run post-pass's domain start there.
    fn best_ordered_assignment_after(
        &self,
        points: &[[T; D]],
        tol: VariableTolerance<T>,
        after: T,
    ) -> (Vec<T>, T) {
        // This implementation uses a generalization of Dynamic Time Warping (DTW), treating the assignment problem as a graph search over vertices `(i, t)` with edges from `(i, t)` -> `(i+1, t_prime)` where `t_prime` is either `t` or one of `locally_closest_points(points[i])`. This structure guarantees the graph is acyclic.
        // A post-pass then re-optimizes each run of tied points: the run's optimal shared parameter minimizes the distance to the run's centroid, which the per-point candidate set cannot represent.
        if points.is_empty() {
            return (Vec::new(), T::zero());
        }
        // One span-polynomial cache serves every closest-point query and curve
        // evaluation below; the knots don't change within this call.
        let spans = self.span_polys();

        // Geometric range pruning. The expensive part of candidate generation is the
        // per-span root-finder; most of it is wasted on spans whose curve lies far from
        // the point. A per-span bounding-box distance gives a cheap lower bound `lb(k, p)`
        // on the squared distance from `p` to the curve over span `k`. Using it we shrink
        // each point's search to the contiguous span range that could hold a minimum
        // belonging to some *optimal* assignment — never dropping a beneficial one (see
        // `search_ranges`), so the DP result is unchanged.
        let bboxes = self.span_control_bboxes();
        let n_spans = spans.len();
        let lb = |k: usize, p: &[T; D]| -> T {
            let (lo, hi) = &bboxes[k];
            let mut acc = T::zero();
            for d in 0..D {
                let over = (lo[d] - p[d]).max(p[d] - hi[d]).max(T::zero());
                acc += over * over;
            }
            acc
        };
        let mut ranges = self.search_ranges(&spans, points, &lb, n_spans, tol, after);

        // The floor rules out every span before the one holding it, so clip the ranges to
        // it: the constrained optimum lies at `t >= after` and so cannot be in them.
        let floored = after > T::zero();
        let span_min = if floored {
            self.span_and_local(after).0
        } else {
            0
        };
        if floored {
            for (lo, hi) in &mut ranges {
                *lo = (*lo).max(span_min);
                *hi = (*hi).max(span_min + 1);
            }
        }

        let mut candidates = self.candidates_in_ranges(&spans, points, &ranges, tol);

        // The DP bands on parameter windows; a geometric span range `[lo, hi)` doubles as the
        // proven parameter interval for point `i`'s optimum, so pass its ends as parameters.
        let param_bands: Vec<(T, T)> = ranges
            .iter()
            .map(|&(lo, hi)| (convert::<f64, T>(lo as f64).max(after), convert(hi as f64)))
            .collect();
        if floored {
            // Drop candidates the floor forbids, and seed it into the candidate set of every
            // point whose band starts there — a point the floor actually binds may well be
            // optimal *at* it, and that is not a local minimum the search reports. Points
            // proven to belong in a later span keep a band above the floor and are left
            // alone, so every candidate stays inside its own band (the DP requires it).
            for (cand, &(band_lo, _)) in candidates.iter_mut().zip(&param_bands) {
                cand.retain(|&t| t >= after);
                if band_lo <= after {
                    cand.push(after);
                }
            }
        }
        // Every point must offer the DP at least one state, or the path through it is
        // unreachable and there is no assignment at all. Two things can empty a set: a
        // pruned range that lies strictly inside the domain and holds no local minimum
        // (the range ends only count as minima where they are the true domain ends), and
        // the floor filtering above. Neither is a pathology — a *straight* curve produces
        // the first routinely, since a degenerate bounding box gives the pruning nothing
        // to separate spans by. The band's own lower end is always a feasible stand-in:
        // in band by construction, and at or above the floor.
        for (cand, &(band_lo, _)) in candidates.iter_mut().zip(&param_bands) {
            if cand.is_empty() {
                cand.push(band_lo);
            }
        }
        let domain = (after.max(T::zero()), convert(n_spans as f64));
        self.assignment_from_candidates(&spans, points, &candidates, &param_bands, domain, tol)
    }

    /// Each point's candidate parameters: its locally-closest points sought only within
    /// its pruned span range `ranges[i]`, as bare parameters. The seam between range
    /// pruning and the DP.
    fn candidates_in_ranges(
        &self,
        spans: &[SpanPolys<T, D>],
        points: &[[T; D]],
        ranges: &[(usize, usize)],
        tol: VariableTolerance<T>,
    ) -> Vec<Vec<T>> {
        points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let (span_lo, span_hi) = ranges[i];
                self.locally_closest_in_range(spans, p, span_lo, span_hi, tol)
                    .into_iter()
                    .map(|(t, _)| t)
                    .collect()
            })
            .collect()
    }

    /// The DP over the assignment graph plus the tied-run post-pass, given each point's
    /// candidate parameters and the parameter window `param_bands[i]` its assignment is
    /// confined to. Split out from [`Self::best_ordered_assignment`] so the candidate
    /// generation (which the geometric pruning changes) can be exercised independently of
    /// the search that consumes it.
    ///
    /// `param_bands[i] = (lo, hi)` is the closed parameter interval point `i`'s assignment
    /// may occupy; the DP bands its grid sweep to the grid points inside it (see below). The
    /// caller passes each point's pruned span window `[lo, hi)` verbatim (the integer ends as
    /// parameters), which `search_ranges` proved contains the optimum. Passing
    /// `(0, num_spans)` for every point disables the banding and runs the plain full-grid DP.
    ///
    /// Every own candidate of point `i` must lie within `param_bands[i]` (the caller
    /// guarantees this by generating candidates over the same window, or filtering to it);
    /// out-of-band candidates would index the band-sized parent rows out of bounds.
    ///
    /// `domain = (lo, hi)` bounds the tied-run post-pass's boundary blocks — the first block
    /// may move down to `lo`, the last up to `hi` (interior blocks are bounded by their
    /// neighbors) — so the post-pass cannot escape the assignment floor or the domain end.
    fn assignment_from_candidates(
        &self,
        spans: &[SpanPolys<T, D>],
        points: &[[T; D]],
        candidates: &[Vec<T>],
        param_bands: &[(T, T)],
        domain: (T, T),
        tol: VariableTolerance<T>,
    ) -> (Vec<T>, T) {
        // A point is assigned either one of its own candidates or pinned to its
        // predecessor's parameter, so every reachable parameter is in the sorted
        // union of all candidates. These are the `t` values of the graph vertices.
        let mut grid: Vec<T> = candidates.iter().flatten().copied().collect();
        grid.sort_by(|a, b| a.partial_cmp(b).expect("parameters are finite"));
        grid.dedup();
        let grid_index = |t: T| {
            grid.binary_search_by(|x| x.partial_cmp(&t).expect("parameters are finite"))
                .expect("candidate parameters are in the grid")
        };

        // The curve at each grid point, evaluated once and reused by every point's cost.
        let curve: Vec<SVector<T, D>> = grid
            .iter()
            .map(|&t| {
                let (k, u) = self.span_and_local(t);
                SVector::from_fn(|d, _| poly::eval(&spans[k].c[d], u))
            })
            .collect();
        let cost = |i: usize, s: usize| {
            let mut acc = T::zero();
            for d in 0..D {
                let diff = curve[s][d] - points[i][d];
                acc += diff * diff;
            }
            acc
        };

        // Banded DP. Each point's assignment is confined to the parameter window
        // `param_bands[i] = [lo_i, hi_i]`, which the geometric pruning *proved* contains its
        // optimum, so no considered assignment leaves it. Point i's DP states are therefore
        // confined to the grid indices inside that window — its "band". With the windows
        // narrow (a few spans), each point sweeps a handful of states rather than the whole
        // grid, turning the O(n·grid) (≈ O(n²)) sweep into ≈ O(n·band). This is loss-free:
        // every optimal path stays in the bands, so only unreachable/dominated out-of-window
        // states are skipped. Full windows `(0, num_spans)` give `band[i] = 0..grid.len()`,
        // reproducing the plain DP.
        let band: Vec<(usize, usize)> = param_bands
            .iter()
            .map(|&(lo_p, hi_p)| {
                (
                    grid.partition_point(|&x| x < lo_p),
                    grid.partition_point(|&x| x <= hi_p),
                )
            })
            .collect();

        // dp[s]: least total error over the points so far with the current point at
        // grid[s]; unreachable states are None. Only entries inside the current point's
        // written region `[cur_lo, cur_hi)` (a subset of its band) are ever read; stale
        // entries outside it are never touched. `next`/`prefix_best` are reused per point.
        let mut dp: Vec<Option<T>> = vec![None; grid.len()];
        let (mut cur_lo, mut cur_hi) = (grid.len(), 0usize);
        for &t in &candidates[0] {
            let s = grid_index(t);
            dp[s] = Some(cost(0, s));
            cur_lo = cur_lo.min(s);
            cur_hi = cur_hi.max(s + 1);
        }
        // Parents, one band-wide row per point i ≥ 1: the state of point i - 1 on the best
        // path to (i, s) is `parents[row_start[i] + (s - band[i].0)]`. Rows are band-sized
        // (not grid-sized) so the whole DP is ≈ O(n·band) in time and space.
        let mut parents: Vec<usize> = Vec::new();
        let mut row_start: Vec<usize> = vec![usize::MAX; points.len()];
        let mut next: Vec<Option<T>> = vec![None; grid.len()];
        let mut prefix_best: Vec<Option<(T, usize)>> = vec![None; grid.len()];
        for (i, own_candidates) in candidates.iter().enumerate().skip(1) {
            let (clo, chi) = band[i];
            row_start[i] = parents.len();
            let row = row_start[i];
            parents.resize(row + (chi - clo), usize::MAX);
            for v in &mut next[clo..chi] {
                *v = None;
            }
            // Prefix-best over the predecessor's written window `[cur_lo, cur_hi)`:
            // `prefix_best[s]` is the cheapest predecessor state at or before `s`.
            let mut best: Option<(T, usize)> = None;
            for s in cur_lo..cur_hi {
                if let Some(v) = dp[s]
                    && best.is_none_or(|(bv, _)| v < bv)
                {
                    best = Some((v, s));
                }
                prefix_best[s] = best;
            }
            let (mut nlo, mut nhi) = (grid.len(), 0usize);
            // Stay pinned at the predecessor's parameter, where it lies in this point's band.
            for s in clo.max(cur_lo)..chi.min(cur_hi) {
                if let Some(v) = dp[s] {
                    next[s] = Some(v + cost(i, s));
                    parents[row + (s - clo)] = s;
                    nlo = nlo.min(s);
                    nhi = nhi.max(s + 1);
                }
            }
            // Or move to one of this point's own candidates, from the best predecessor at
            // or before it (monotonicity). A candidate with no predecessor at or before it
            // is unreachable and skipped.
            for &t in own_candidates {
                let s = grid_index(t);
                if s < cur_lo {
                    continue;
                }
                if let Some((v, from)) = prefix_best[s.min(cur_hi - 1)] {
                    let total = v + cost(i, s);
                    if next[s].is_none_or(|nv| total < nv) {
                        next[s] = Some(total);
                        parents[row + (s - clo)] = from;
                        nlo = nlo.min(s);
                        nhi = nhi.max(s + 1);
                    }
                }
            }
            std::mem::swap(&mut dp, &mut next);
            (cur_lo, cur_hi) = (nlo, nhi);
        }

        // Best final state, then walk the parents back.
        let (mut s, mut best) = (usize::MAX, None::<T>);
        for idx in cur_lo..cur_hi {
            if let Some(v) = dp[idx]
                && best.is_none_or(|bv| v < bv)
            {
                best = Some(v);
                s = idx;
            }
        }
        best.expect("the first point's candidates are never empty");
        let mut states = vec![s; points.len()];
        for i in (1..points.len()).rev() {
            s = parents[row_start[i] + (s - band[i].0)];
            states[i - 1] = s;
        }
        let mut ts: Vec<T> = states.into_iter().map(|s| grid[s]).collect();

        // Post-pass: the DP can only place a run of points tied by the ordering
        // constraint on one of the individual candidates, but the run's best shared
        // parameter minimizes its *summed* distance — equivalently (up to a constant)
        // the distance to the run's centroid, which generally lies between the
        // individual candidates. Re-optimize each maximal run of equal parameters
        // between its neighbors' parameters, repeating while runs move (runs that
        // collide merge into one). Only strict improvements are taken, so the total
        // error decreases monotonically and this terminates.
        for _ in 0..=points.len() {
            let mut changed = false;
            let mut blocks: Vec<(usize, usize)> = Vec::new();
            let mut start = 0;
            for i in 1..=ts.len() {
                if i == ts.len() || ts[i] != ts[start] {
                    blocks.push((start, i));
                    start = i;
                }
            }
            for (b, &(b0, b1)) in blocks.iter().enumerate() {
                let lo = if b == 0 { domain.0 } else { ts[b0 - 1] };
                let hi = if b + 1 == blocks.len() {
                    domain.1
                } else {
                    ts[b1]
                };
                let block_cost = |t: T| {
                    let (k, u) = self.span_and_local(t);
                    let c: [T; D] = std::array::from_fn(|d| poly::eval(&spans[k].c[d], u));
                    let mut acc = T::zero();
                    for p in &points[b0..b1] {
                        for d in 0..D {
                            let diff = c[d] - p[d];
                            acc += diff * diff;
                        }
                    }
                    acc
                };
                // The block's summed distance is |block| * (distance to its centroid)
                // plus a constant, so the closest points to the centroid are the
                // interior candidates; the bounds are the boundary candidates.
                let inv_len = T::one() / convert::<f64, T>((b1 - b0) as f64);
                let centroid: [T; D] = std::array::from_fn(|d| {
                    let mut acc = T::zero();
                    for p in &points[b0..b1] {
                        acc += p[d];
                    }
                    acc * inv_len
                });
                // A singleton block's centroid is the point itself, so its closest
                // parameters are exactly the candidates already found for it; reuse
                // them instead of re-running the full closest-point search.
                let block_closest: Vec<T> = if b1 - b0 == 1 {
                    candidates[b0].clone()
                } else {
                    self.locally_closest_in(spans, &centroid, tol)
                        .into_iter()
                        .map(|(t, _)| t)
                        .collect()
                };
                let block_candidates = block_closest
                    .into_iter()
                    .filter(|&t| lo <= t && t <= hi)
                    .chain([lo, hi]);
                let mut best_t = ts[b0];
                let mut best_cost = block_cost(best_t);
                for t in block_candidates {
                    let c = block_cost(t);
                    if c < best_cost {
                        best_cost = c;
                        best_t = t;
                    }
                }
                if best_t != ts[b0] {
                    changed = true;
                    for t in &mut ts[b0..b1] {
                        *t = best_t;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let total = ts.iter().zip(points).fold(T::zero(), |acc, (&t, p)| {
            acc + self.distance_sq(spans, t, p)
        });
        (ts, total)
    }

    /// Fits a cardinal B-spline to points.
    ///
    /// Computes the cardinal B-spline with `m` control points that best fits the provided `points`, which are assumed to be ordered along the spline. Returns the spline and a monotonic assignment from each point to the time.
    ///
    /// `rel_tol` trades accuracy for speed: the fitting loop stops once an iteration
    /// improves the total squared error by no more than `rel_tol` times the error
    /// (plus an absolute floor near machine precision). `0.0` gives maximum quality;
    /// around `1e-3` is severalfold faster at a small cost in error.
    ///
    /// Returns: a tuple of the cardinal B-spline, the monotonic assignment, and the total (squared) error.
    ///
    /// # Panics
    ///
    /// Panics if `m < 2`, if `rel_tol` is negative, or if `points` is empty.
    pub fn fit_monotonic(m: usize, tol: FitTolerance<T>, points: &[[T; D]]) -> (Self, Vec<T>, T) {
        // This implementation leverages an expectation-maximization loop that starts with a uniform assignment and then iteratively:
        // * Holds the assignment constant and optimizes the control points that minimize the distance to the points. This is a simple quadratic optimization problem.
        // * Holds the control points constant and optimizes the monotonic assignment from points to input variables.
        // This process repeats until convergence.
        assert!(m >= 2, "at least 2 control points are required (got {m})");
        assert!(!points.is_empty(), "cannot fit a spline to zero points");
        let (mut spline, ts) = Self::polyline_init(m, points);
        let (ts, err) = spline.em_fit_in_place(ts, tol, points, Settled::none());
        (spline, ts, err)
    }

    /// Refits this spline to `points`, holding what has already [`Settled`] fixed.
    ///
    /// This is [`Self::fit_monotonic`] for a fit that is being built up rather than done in
    /// one shot: instead of starting from a polyline initialization it warm-starts from the
    /// spline it is called on, and instead of solving for every control point it solves
    /// only for the ones past `settled.control_points`. The frozen prefix comes back
    /// bit-identical, so the curve over `[0, self.frozen_until(settled.control_points)]` is
    /// unchanged — that part of the fit is final, whatever arrives later.
    ///
    /// Everything else is the ordinary solver: the same E-step, the same SQUAREM-accelerated
    /// EM loop, the same tolerances. Only the M-step's linear system shrinks, and (when the
    /// caller has retired points, see [`Settled::after`]) the assignment gains a floor.
    ///
    /// Returns the assignment of `points` and its total squared error, mirroring
    /// [`Self::fit_monotonic`] minus the spline, which is `self`.
    ///
    /// # The incremental loop
    ///
    /// The intended use is to grow a spline as points arrive, freezing control points once
    /// they settle. [`IncrementalFit`] drives exactly this loop and is the easier entry
    /// point; reach past it to these primitives when you want a different growth or freezing
    /// policy. The whole of it is:
    ///
    /// ```
    /// # use nalgebra::Const;
    /// # use spline_fit::{CardinalCubicBSpline, FitTolerance, Settled};
    /// # let batches: Vec<Vec<[f64; 2]>> = (0..8)
    /// #     .map(|b| (0..25).map(|i| {
    /// #         let t = (b * 25 + i) as f64 * 0.02;
    /// #         [t, (t * 2.0).sin()]
    /// #     }).collect())
    /// #     .collect();
    /// # let tol = FitTolerance::<f64>::default();
    /// # let control_points = |n: usize| (n / 10).max(2);
    /// let (mut spline, _, _) =
    ///     CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(2, tol, &batches[0]);
    /// let mut live = batches[0].clone();
    /// let mut seen = live.len();
    /// let mut settled = Settled::none();
    ///
    /// for batch in &batches[1..] {
    ///     seen += batch.len();
    ///     live.extend_from_slice(batch);
    ///     spline.extend_control_points(control_points(seen), batch);
    ///     let (ts, _error) = spline.refit_monotonic(settled, tol, &live);
    ///
    ///     // Freeze all but the last four control points — those are the ones the incoming
    ///     // data still pulls on — and retire the points whose assignment has landed in the
    ///     // curve the frozen ones determine. Those points cannot move again, so the next
    ///     // refit need not carry them; what remains must still be ordered after them,
    ///     // which is what `after` records.
    ///     settled.control_points = spline.num_control_points().saturating_sub(4);
    ///     let until = spline.frozen_until(settled.control_points);
    ///     let done = ts.partition_point(|&t| t <= until);
    ///     if done > 0 {
    ///         settled.after = ts[done - 1];
    ///         live.drain(..done);
    ///     }
    /// }
    /// # assert!(live.len() < 60, "most points should have been retired");
    /// ```
    ///
    /// Each update then costs O(live points), not O(every point seen).
    ///
    /// # Panics
    ///
    /// Panics if `settled.control_points` exceeds the number of control points, or if
    /// `points` is empty.
    pub fn refit_monotonic(
        &mut self,
        settled: Settled<T>,
        tol: FitTolerance<T>,
        points: &[[T; D]],
    ) -> (Vec<T>, T) {
        let m = self.control_points.nrows();
        assert!(
            settled.control_points <= m,
            "cannot freeze {} of {m} control points",
            settled.control_points
        );
        assert!(
            settled.tail <= m,
            "cannot hold {} trailing of {m} control points",
            settled.tail
        );
        assert!(!points.is_empty(), "cannot fit a spline to zero points");
        // The assignment a refit starts from is an E-step against the incoming (warm)
        // control points. A full fit instead starts from the uniform assignment of its
        // polyline initialization, which would throw away the very state a refit exists to
        // reuse — and would send the first M-step somewhere the frozen prefix cannot follow.
        let ts = self.e_step(points, tol.into(), settled.after).0;
        self.em_fit_in_place(ts, tol, points, settled)
    }

    /// Appends control points until there are `m` of them, the new ones tracing `along` —
    /// typically the points added since the last fit, which is where the curve has to grow
    /// to reach. Does nothing if there are already `m` or more.
    ///
    /// The existing control points keep their values, so this is safe to call on a spline
    /// with a frozen prefix: appending extends the domain by one span per control point and
    /// leaves the curve below [`Self::frozen_until`] alone. It is only an initialization —
    /// the appended points are the refit's starting guess, not a constraint on it.
    pub fn extend_control_points(&mut self, m: usize, along: &[[T; D]]) {
        let m0 = self.control_points.nrows();
        if m <= m0 {
            return;
        }
        let added = m - m0;
        let n = along.len();
        let old = &self.control_points;
        // Row `m0 - 1 + j` of `added` takes the point `j / added` of the way along the new
        // data, so the last appended control point lands on the last point — which is where
        // the clamped end of the curve wants to be. Without new data to trace there is
        // nothing better to say than "carry on from the last control point".
        let grown = OMatrix::<T, Dyn, Const<D>>::from_fn_generic(Dyn(m), Const::<D>, |j, d| {
            if j < m0 {
                return old[(j, d)];
            }
            if n == 0 {
                return old[(m0 - 1, d)];
            }
            let idx = (j - m0 + 1) as f64 / added as f64 * (n - 1) as f64;
            let i0 = (idx.floor() as usize).min(n - 1);
            let i1 = (i0 + 1).min(n - 1);
            let w: T = convert(idx - i0 as f64);
            along[i0][d] * (T::one() - w) + along[i1][d] * w
        });
        self.control_points = grown;
    }

    /// The polyline initialization shared by the fitting entry points: `m` control points
    /// tracing the point sequence at evenly spaced fractions (these also serve as the
    /// proximal prior for control points the assignment leaves unsupported), and a uniform
    /// initial assignment over the domain.
    fn polyline_init(m: usize, points: &[[T; D]]) -> (Self, Vec<T>) {
        let n = points.len();
        let control_points =
            OMatrix::<T, Dyn, Const<D>>::from_fn_generic(Dyn(m), Const::<D>, |j, d| {
                let idx = j as f64 / (m - 1) as f64 * (n - 1) as f64;
                let i0 = (idx.floor() as usize).min(n - 1);
                let i1 = (i0 + 1).min(n - 1);
                let w: T = convert(idx - i0 as f64);
                points[i0][d] * (T::one() - w) + points[i1][d] * w
            });
        let spline =
            ClampedCardinalBSpline::from_control_points(Const::<3>, control_points).unwrap();
        let spans = spline.num_spans();
        let ts: Vec<T> = (0..n)
            .map(|i| convert(spans as f64 * i as f64 / (n - 1).max(1) as f64))
            .collect();
        (spline, ts)
    }

    /// One EM M-step: the control points minimizing the squared error at the fixed
    /// assignment `ts`, via the normal equations `B'B K = B'P`. A tiny proximal ridge
    /// toward `prior` keeps the system positive definite when the assignment leaves
    /// some basis functions unsupported; at an EM fixed point (`prior` equal to the
    /// result) its bias cancels exactly. Reads no state from `self.control_points`,
    /// so it can be evaluated for any candidate assignment.
    ///
    /// The first `frozen` rows of `prior` are held at their values instead of being solved
    /// for. Splitting `K = [F; X]` with `F` given, the normal equations reduce to the
    /// smaller system `(B'B)_xx X = (B'P)_x − (B'B)_xf F` — still the exact minimizer over
    /// the free control points, and the frozen ones come back out bit-identical. Every
    /// other part of the fit (the E-step, the acceleration, the tolerances) is oblivious to
    /// the freezing: the extrapolation in particular sees zero movement on those rows and
    /// so leaves them alone by construction.
    ///
    /// The payload dimension `E` is independent of the spline's own `D`: the assignment is
    /// the only thing the M-step reads from the geometry, so the same solve fits any number
    /// of per-point channels against it. The fitting loop uses `E = D`; [`Self::fit_channels`]
    /// is the same solve for data that rides the geometry without shaping it.
    fn m_step<const E: usize>(
        &self,
        ts: &[T],
        prior: &OMatrix<T, Dyn, Const<E>>,
        points: &[[T; E]],
        frozen: usize,
        tail: usize,
        smoothing: T,
    ) -> OMatrix<T, Dyn, Const<E>> {
        // The uniform basis serves every span thanks to the duplicating knot view.
        let basis = self.basis_matrix();
        let order = self.order();
        let m = prior.nrows();
        let n = points.len();
        if frozen + tail >= m || n == 0 {
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
        // reaches a free row touches nothing below `frozen - (order - 1)`. Everything
        // before that is a frozen row coupled to no free one, contributing zero. An
        // `m x m` matrix therefore spends nearly all of itself on structural zeros —
        // 160KB of them at 200 control points, memset four times per sample, which was
        // most of the cost of a long stroke and the reason the per-update time grew
        // with the length of the whole thing rather than with the window.
        let base = frozen.saturating_sub(order - 1);
        let w = m - base;
        let mut btb = OMatrix::<T, Dyn, Dyn>::zeros(w, w);
        let mut btp = OMatrix::<T, Dyn, Const<E>>::zeros_generic(Dyn(w), Const::<E>);
        for (&t, p) in ts.iter().zip(points) {
            let (k, u) = self.span_and_local(t);
            // Cannot reach a free row, so contributes nothing to what is solved.
            if self.knot_row(k + order - 1) < frozen {
                continue;
            }
            let wts = basis * self.u_powers(u);
            for a in 0..order {
                let ra = self.knot_row(k + a) - base;
                for b in 0..order {
                    btb[(ra, self.knot_row(k + b) - base)] += wts[a] * wts[b];
                }
                for d in 0..E {
                    btp[(ra, d)] += wts[a] * p[d];
                }
            }
        }
        // Bending energy `Σ ‖P₍ⱼ₋₁₎ − 2Pⱼ + P₍ⱼ₊₁₎‖²`, weighted against the *average*
        // data pull per control point so the knob means the same thing whatever the
        // point count and polygon length are. Quadratic in the control points, so it
        // is just another symmetric band added to the normal matrix — and its target
        // is zero curvature, so the right-hand side is untouched.
        //
        // This is what stops the curve wandering where no point is assigned; see
        // [`FitTolerance::with_smoothing`]. Note it lands in `btb` *before* the
        // frozen block is folded into the right-hand side below, so it also couples
        // the first free control point to the frozen ones — which is what makes the
        // free tail continue smoothly out of the committed prefix instead of being
        // free to start off in any direction at all. Only the triples that touch a
        // solved row are in the window; the rest act on frozen rows alone.
        if smoothing > T::zero() && m >= 3 {
            let sw = smoothing * convert::<f64, T>(n as f64 / m as f64);
            let c = [T::one(), convert::<f64, T>(-2.0), T::one()];
            for j in (base + 1).max(1)..m - 1 {
                let idx = [j - 1 - base, j - base, j + 1 - base];
                for (a, &ca) in c.iter().enumerate() {
                    for (b, &cb) in c.iter().enumerate() {
                        btb[(idx[a], idx[b])] += sw * ca * cb;
                    }
                }
            }
        }
        // The free block of the system, with the frozen rows' contribution moved to the
        // right-hand side. With `frozen == 0` this is the whole system, unchanged.
        let free = m - frozen - tail;
        let mut lambda: T = convert::<f64, T>(n as f64) * T::default_epsilon().sqrt();
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
            lambda *= convert::<f64, T>(10.0);
        }
        unreachable!("ridge-regularized normal equations are positive definite")
    }

    /// Control values for `E` per-point channels that ride along with the geometry —
    /// pressure, a timestamp, anything measured *at* the points but not part of what the
    /// curve is fitted to.
    ///
    /// Given `values[i]` for the point assigned to `ts[i]`, this returns the control values
    /// whose B-spline — the same basis, the same knots, the same parameterization as `self`
    /// — is the least-squares fit of that data. Evaluating it at a parameter therefore reads
    /// the channel *where the curve is*, so a caller can carry the channels on the control
    /// points and interpolate them exactly as it interpolates position.
    ///
    /// This is deliberately not a fit in `D + E` dimensions. Folding channels into the
    /// geometry would let them pull on the assignment — a pressure ramp would stretch the
    /// parameterization the way a longer path does, distorting the curve to buy error in a
    /// quantity that has no length. Here the geometry decides `ts` alone and the channels
    /// solve against it, which also means no weight is needed to reconcile pixels with
    /// whatever units the channels are in.
    ///
    /// `frozen` and `prior` work exactly as in the geometric refit ([`Self::refit_monotonic`]):
    /// the first `frozen` rows of `prior` come back unchanged, so channels stay frozen in
    /// step with the control points they sit on. Pass `0` and an empty prior for a one-shot
    /// solve.
    ///
    /// `prior` may be **shorter** than the control polygon — the usual case in an
    /// incremental fit, where the geometry has just grown ([`Self::extend_control_points`])
    /// and the channels have not. The missing rows are seeded by repeating the last one,
    /// which only sets where the proximal ridge is centered: those rows are past `frozen`,
    /// so the solve determines them.
    ///
    /// # Panics
    ///
    /// Panics if `prior` has more rows than there are control points, if `ts` and `values`
    /// differ in length, or if `frozen` exceeds the control-point count.
    pub fn fit_channels<const E: usize>(
        &self,
        ts: &[T],
        values: &[[T; E]],
        frozen: usize,
        prior: &OMatrix<T, Dyn, Const<E>>,
    ) -> OMatrix<T, Dyn, Const<E>> {
        let m = self.control_points.nrows();
        assert!(
            prior.nrows() <= m,
            "channel prior has {} rows for {m} control points",
            prior.nrows()
        );
        assert!(frozen <= m, "cannot freeze {frozen} of {m} channel rows");
        assert_eq!(
            ts.len(),
            values.len(),
            "every assigned parameter needs a value"
        );
        self.fit_channels_smoothed(ts, values, frozen, 0, prior, T::zero())
    }

    /// [`Self::fit_channels`] with an explicit held tail and curvature penalty — the
    /// same solve the geometry uses, exposed for a caller that already knows the
    /// correspondence and so needs no assignment search at all.
    pub fn fit_channels_smoothed<const E: usize>(
        &self,
        ts: &[T],
        values: &[[T; E]],
        frozen: usize,
        tail: usize,
        prior: &OMatrix<T, Dyn, Const<E>>,
        smoothing: T,
    ) -> OMatrix<T, Dyn, Const<E>> {
        let m = self.control_points.nrows();
        let have = prior.nrows();
        let grown =
            OMatrix::<T, Dyn, Const<E>>::from_fn_generic(Dyn(m), Const::<E>, |j, d| {
                match (j < have, have) {
                    (true, _) => prior[(j, d)],
                    (false, 0) => T::zero(),
                    (false, h) => prior[(h - 1, d)],
                }
            });
        // Passenger channels are not regularized here: they are one-dimensional and
        // bounded by the data, and a caller that wants them smoothed can say so by
        // pre-smoothing its values.
        self.m_step(ts, &grown, values, frozen, tail, smoothing)
    }

    /// One EM E-step: the monotonic assignment and its squared error for the current control
    /// points, with the assignment floored at `after` (see [`Settled::after`]).
    fn e_step(&self, points: &[[T; D]], tol: VariableTolerance<T>, after: T) -> (Vec<T>, T) {
        self.best_ordered_assignment_after(points, tol, after)
    }

    /// The EM loop of [`Self::fit_monotonic`], starting from the control points already in
    /// `spline` and the assignment `ts`.
    ///
    /// The plain EM map `K ↦ M(E(K))` (M-step then E-step) converges quickly at first
    /// and then crawls along a slow *linear* tail — the classic setting for SQUAREM
    /// (Varadhan & Roland 2008). Each cycle takes two ordinary EM steps
    /// `K₀ → K₁ → K₂`, then jumps to the vector-extrapolated point
    /// `Kₐ = K₀ − 2α r + α² v` with `r = K₁ − K₀`, `v = K₂ − 2K₁ + K₀`, and the S3
    /// steplength `α = −‖r‖/‖v‖` (clamped to `α ≤ −1`, where `α = −1` reproduces the
    /// plain iterate `K₂`). The objective is the E-step error `L(K)`, which the M-step's
    /// proximal ridge keeps monotone (`L(K₂) ≤ L(K₁) ≤ L(K₀)`); the accelerated point is
    /// accepted only when `L(Kₐ) ≤ L(K₂)`, so acceleration can never do worse than plain
    /// EM. Each cycle costs three E-steps (the expensive part) but leaps far along the
    /// tail, cutting total E-steps severalfold.
    ///
    /// `settled` fixes a prefix of the control points and floors the assignment; with
    /// [`Settled::none`] this is the plain fit. Freezing changes nothing structurally — the
    /// M-step simply solves a smaller system (see [`Self::m_step`]) — so an incremental fit
    /// gets the same convergence behavior, acceleration and all.
    fn em_fit_in_place(
        &mut self,
        ts: Vec<T>,
        tol: FitTolerance<T>,
        points: &[[T; D]],
        settled: Settled<T>,
    ) -> (Vec<T>, T) {
        let spline = self;
        let m = spline.control_points.nrows();
        let two = convert::<f64, T>(2.0);
        let one = T::one();
        // Cap total E-step evaluations (the dominant cost), matching the old
        // 100-iteration ceiling; acceleration keeps runs far below it.
        const MAX_EVALS: usize = 100;

        // First EM step from the caller's assignment, its proximal prior the incoming
        // (polyline-initialized) control points. Then the initial objective.
        let mut k0 = spline.m_step(
            &ts,
            &spline.control_points,
            points,
            settled.control_points,
            settled.tail,
            tol.smoothing,
        );
        spline.control_points = k0.clone();
        let (mut ts0, mut l0) = spline.e_step(points, tol.into(), settled.after);
        let mut evals = 1usize;

        let converged = |from: T, to: T| from - to <= tol.rel_metric * from + tol.abs_metric;
        let mut prev_l: Option<T> = None;
        loop {
            // Diminishing returns: stop once a whole cycle buys less than rel_tol of
            // the error (the absolute floor terminates exactly-fittable data).
            if prev_l.is_some_and(|pl| converged(pl, l0)) || evals >= MAX_EVALS {
                break;
            }
            prev_l = Some(l0);

            // EM step 1: K1 = M(E(K0)); E(K0) is the assignment ts0 we already hold.
            let k1 = spline.m_step(
                &ts0,
                &k0,
                points,
                settled.control_points,
                settled.tail,
                tol.smoothing,
            );
            spline.control_points = k1.clone();
            let (ts1, l1) = spline.e_step(points, tol.into(), settled.after);
            evals += 1;
            if converged(l0, l1) || evals >= MAX_EVALS {
                (k0, ts0, l0) = (k1, ts1, l1);
                continue;
            }

            // EM step 2: K2 = M(E(K1)).
            let k2 = spline.m_step(
                &ts1,
                &k1,
                points,
                settled.control_points,
                settled.tail,
                tol.smoothing,
            );
            spline.control_points = k2.clone();
            let (ts2, l2) = spline.e_step(points, tol.into(), settled.after);
            evals += 1;

            // Extrapolation coefficients r·r and v·v (Frobenius).
            let (mut rr, mut vv) = (T::zero(), T::zero());
            for j in 0..m {
                for d in 0..D {
                    let r = k1[(j, d)] - k0[(j, d)];
                    let v = k2[(j, d)] - two * k1[(j, d)] + k0[(j, d)];
                    rr += r * r;
                    vv += v * v;
                }
            }
            if rr > T::zero() && vv > T::zero() && evals < MAX_EVALS {
                // S3 steplength, clamped to α ∈ [−1e6, −1]: the magnitude floor keeps
                // the step at least as long as plain EM; the ceiling avoids overflow
                // from a degenerate near-zero v (such wild steps are rejected below).
                let mut alpha = -(rr / vv).sqrt();
                alpha = alpha.min(-one).max(convert::<f64, T>(-1.0e6));
                let mut ka = OMatrix::<T, Dyn, Const<D>>::zeros_generic(Dyn(m), Const::<D>);
                for j in 0..m {
                    for d in 0..D {
                        let r = k1[(j, d)] - k0[(j, d)];
                        let v = k2[(j, d)] - two * k1[(j, d)] + k0[(j, d)];
                        ka[(j, d)] = k0[(j, d)] - two * alpha * r + alpha * alpha * v;
                    }
                }
                spline.control_points = ka.clone();
                let (tsa, la) = spline.e_step(points, tol.into(), settled.after);
                evals += 1;
                if la <= l2 {
                    (k0, ts0, l0) = (ka, tsa, la);
                    continue;
                }
            }
            // Safeguard: the extrapolation didn't beat two plain EM steps; take K2.
            (k0, ts0, l0) = (k2, ts2, l2);
        }

        spline.control_points = k0;
        (ts0, l0)
    }

    /// Adaptively fits a cardinal B-spline to points.
    ///
    /// Computes the cardinal B-spline that best fits the provided `points`, which are assumed to be ordered along the spline. The number of control points is chosen adaptively using `control_point_cost`. Returns the spline and a monotonic assignment from each point to the time.
    ///
    /// `rel_tol` is the per-fit accuracy/speed tradeoff of [`Self::fit_monotonic`].
    ///
    /// Returns: a tuple of the cardinal B-spline, the optimal assignment, and the total (squared) error.
    ///
    /// # Panics
    ///
    /// Panics if `control_point_cost` is not positive, if `rel_tol` is negative, or if `points` is empty.
    pub fn fit_monotonic_adaptive(
        control_point_cost: T,
        tol: FitTolerance<T>,
        points: &[[T; D]],
    ) -> (Self, Vec<T>, T) {
        // This implementation works be repeatedly calling `fit_monotonic`. It starts with an exponential search: `m = 2, m = 4, m = 8, ...` until the increase in the number of knots, scaled by `control_point_cost`, exceeds the reduction in total error. It then performs a binary search between the last two values with the same criterion.
        // Each fit is cold-started: warm-starting knots from the previous fit's curve
        // was measured to be both slower and worse (see PERF.md) — the polyline
        // initialization threads the knots through the data, which is a better basin
        // than the coarser fit's oversmoothed curve.
        // A positive cost guarantees termination: the total error reduction available
        // is bounded by the initial error, while the cost of another doubling grows
        // without bound.
        assert!(
            control_point_cost > T::zero(),
            "control_point_cost must be positive"
        );
        let worth = |from: usize, from_err: T, to: usize, to_err: T| {
            from_err - to_err > control_point_cost * convert((to - from) as f64)
        };

        let mut lo = 2;
        let mut best = Self::fit_monotonic(lo, tol, points);
        let mut hi = loop {
            let hi = lo * 2;
            let fit = Self::fit_monotonic(hi, tol, points);
            if !worth(lo, best.2, hi, fit.2) {
                break hi;
            }
            lo = hi;
            best = fit;
        };
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let fit = Self::fit_monotonic(mid, tol, points);
            if worth(lo, best.2, mid, fit.2) {
                lo = mid;
                best = fit;
            } else {
                hi = mid;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spline<const D: usize>(pts: &[[f64; D]]) -> ClampedCardinalBSpline<f64, Const<D>, Const<3>> {
        let m =
            OMatrix::<f64, Dyn, Const<D>>::from_fn_generic(Dyn(pts.len()), Const::<D>, |i, j| {
                pts[i][j]
            });
        ClampedCardinalBSpline::from_control_points(Const::<3>, m).expect("enough knots")
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol})");
    }

    /// Coontrol points at x = 0..5 on the x-axis (domain [0, 7]). By the linear precision of
    /// the uniform basis, x(t) = t - 1 exactly on the interior spans t ∈ [2, 5],
    /// easing in from x=0 at t=0 and out to x=5 at t=7 at the clamped ends.
    fn line_spline() -> ClampedCardinalBSpline<f64, Const<2>, Const<3>> {
        let pts: Vec<[f64; 2]> = (0..6).map(|x| [x as f64, 0.0]).collect();
        spline(&pts)
    }

    fn wiggle_spline() -> ClampedCardinalBSpline<f64, Const<2>, Const<3>> {
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
        let s = wiggle_spline();
        let m = s.basis_matrix();
        for i in 0..4 {
            let sum: f64 = (0..4).map(|a| m[(a, i)]).sum();
            assert_close(sum, if i == 0 { 1.0 } else { 0.0 }, 1e-12);
        }
    }

    #[test]
    fn basis_matches_uniform_cubic() {
        let s = spline(&[[0.0]; 2]);
        let m = s.basis_matrix();
        let u: f64 = 0.5;
        let expected = [
            (1.0 - u).powi(3) / 6.0,
            (3.0 * u.powi(3) - 6.0 * u.powi(2) + 4.0) / 6.0,
            (-3.0 * u.powi(3) + 3.0 * u.powi(2) + 3.0 * u + 1.0) / 6.0,
            u.powi(3) / 6.0,
        ];
        for (a, e) in expected.iter().enumerate() {
            let row: Poly<f64, 4> = Poly::from_fn(|i, _| m[(a, i)]);
            assert_close(poly::eval(&row, u), *e, 1e-12);
        }
    }

    #[test]
    fn evaluate_interpolates_endpoints() {
        let s = wiggle_spline();
        let start = s.evaluate(0.0);
        let end = s.evaluate(s.num_spans() as f64);
        assert_close(start[0], 0.0, 1e-12);
        assert_close(start[1], 0.0, 1e-12);
        assert_close(end[0], 5.0, 1e-12);
        assert_close(end[1], 0.0, 1e-12);
    }

    #[test]
    fn evaluate_is_linear_on_interior_spans() {
        let s = line_spline();
        for i in 0..=12 {
            let t = 2.0 + i as f64 * 0.25;
            let v = s.evaluate(t);
            assert_close(v[0], t - 1.0, 1e-12);
            assert_close(v[1], 0.0, 1e-12);
        }
        assert_close(s.evaluate(0.0)[0], 0.0, 1e-12);
        assert_close(s.evaluate(7.0)[0], 5.0, 1e-12);
    }

    #[test]
    fn evaluate_of_constant_spline_is_constant() {
        let s = spline(&[[1.0, 2.0]; 5]);
        for i in 0..=24 {
            let v = s.evaluate(i as f64 * 0.25);
            assert_close(v[0], 1.0, 1e-12);
            assert_close(v[1], 2.0, 1e-12);
        }
    }

    #[test]
    fn closest_point_on_a_straight_line() {
        let s = line_spline();
        let result: Vec<_> = s
            .locally_closest_points(&[1.5, 2.0], Default::default())
            .collect();
        assert_eq!(result.len(), 1);
        assert_close(result[0].0, 2.5, 1e-8);
        assert_close(result[0].1, 4.0, 1e-8);
    }

    #[test]
    fn closest_point_clamps_to_domain_ends() {
        let s = line_spline();

        let before: Vec<_> = s
            .locally_closest_points(&[-1.0, 1.0], Default::default())
            .collect();
        assert_eq!(before.len(), 1);
        assert_close(before[0].0, 0.0, 1e-8);
        assert_close(before[0].1, 2.0, 1e-8);

        let after: Vec<_> = s
            .locally_closest_points(&[6.0, 1.0], Default::default())
            .collect();
        assert_eq!(after.len(), 1);
        assert_close(after[0].0, 7.0, 1e-8);
        assert_close(after[0].1, 2.0, 1e-8);
    }

    /// Local minima of the sampled squared distance over a dense grid.
    fn brute_force_minima(
        s: &ClampedCardinalBSpline<f64, Const<2>, Const<3>>,
        point: &[f64; 2],
        steps: usize,
    ) -> Vec<(f64, f64)> {
        let t_end = s.num_spans() as f64;
        let d = |t: f64| {
            let v = s.evaluate(t);
            (v[0] - point[0]).powi(2) + (v[1] - point[1]).powi(2)
        };
        let samples: Vec<f64> = (0..=steps)
            .map(|i| d(t_end * i as f64 / steps as f64))
            .collect();
        let mut minima = Vec::new();
        for i in 0..=steps {
            let left_ok = i == 0 || samples[i] <= samples[i - 1];
            let right_ok = i == steps || samples[i] <= samples[i + 1];
            let strict = (i > 0 && samples[i] < samples[i - 1])
                || (i < steps && samples[i] < samples[i + 1])
                || steps == 0;
            if left_ok && right_ok && strict {
                minima.push((t_end * i as f64 / steps as f64, samples[i]));
            }
        }
        minima
    }

    #[test]
    fn locally_closest_points_match_brute_force() {
        let s = wiggle_spline();
        let queries = [
            [2.5, 0.0],
            [0.5, 1.0],
            [2.0, 3.0],
            [4.0, -3.0],
            [-1.0, -1.0],
            [6.0, 2.0],
        ];
        let steps = 20_000;
        let t_tol = 3.0 * s.num_spans() as f64 / steps as f64;
        let mut max_count = 0;
        for q in &queries {
            let found: Vec<_> = s.locally_closest_points(q, Default::default()).collect();
            let expected = brute_force_minima(&s, q, steps);
            assert_eq!(
                found.len(),
                expected.len(),
                "query {q:?}: found {found:?}, expected {expected:?}"
            );
            for ((t, e), (bt, be)) in found.iter().zip(&expected) {
                assert_close(*t, *bt, t_tol);
                assert_close(*e, *be, 1e-6);
                // The reported error must agree with evaluating the spline directly.
                let v = s.evaluate(*t);
                let direct = (v[0] - q[0]).powi(2) + (v[1] - q[1]).powi(2);
                assert_close(*e, direct, 1e-9);
            }
            max_count = max_count.max(found.len());
        }
        assert!(
            max_count >= 2,
            "expected some query to have multiple minima"
        );
    }

    #[test]
    fn degenerate_spline_reports_endpoints() {
        let s = spline(&[[1.0, 2.0]; 5]);
        // Every point of the spline is (1, 2), so the distance is constant in t and
        // both domain endpoints are reported as (non-strict) minima.
        let far: Vec<_> = s
            .locally_closest_points(&[4.0, 6.0], Default::default())
            .collect();
        assert!(!far.is_empty());
        for (_, e) in far {
            assert_close(e, 25.0, 1e-9);
        }

        let on: Vec<_> = s
            .locally_closest_points(&[1.0, 2.0], Default::default())
            .collect();
        assert!(!on.is_empty());
        for (_, e) in on {
            assert_close(e, 0.0, 1e-9);
        }
    }

    #[test]
    fn next_locally_closest_reproduces_full_list() {
        let s = wiggle_spline();
        let queries = [
            [2.5, 0.0],
            [0.5, 1.0],
            [2.0, 3.0],
            [4.0, -3.0],
            [-1.0, -1.0],
            [6.0, 2.0],
        ];
        for q in &queries {
            let full: Vec<_> = s.locally_closest_points(q, Default::default()).collect();
            // Walking the lazy primitive from before the domain reproduces the full list.
            let mut lazy = Vec::new();
            let mut after = -1.0;
            while let Some((t, e)) = s.next_locally_closest_point(q, after, Default::default()) {
                assert!(t > after, "next parameter must advance: {t} !> {after}");
                lazy.push((t, e));
                after = t;
            }
            assert_eq!(
                full.len(),
                lazy.len(),
                "query {q:?}: full {full:?}, lazy {lazy:?}"
            );
            for ((ft, fe), (lt, le)) in full.iter().zip(&lazy) {
                assert_close(*ft, *lt, 1e-6);
                assert_close(*fe, *le, 1e-6);
            }
        }
    }

    #[test]
    fn next_locally_closest_starts_after_the_given_parameter() {
        let s = line_spline();
        // On a straight line there is a single minimum; querying past it finds nothing.
        let (t, _) = s
            .next_locally_closest_point(&[1.5, 2.0], -1.0, Default::default())
            .expect("a minimum exists");
        assert_close(t, 2.5, 1e-8);
        assert!(
            s.next_locally_closest_point(&[1.5, 2.0], t, Default::default())
                .is_none()
        );
    }

    fn squared_distance(
        s: &ClampedCardinalBSpline<f64, Const<2>, Const<3>>,
        t: f64,
        p: &[f64; 2],
    ) -> f64 {
        let v = s.evaluate(t);
        (v[0] - p[0]).powi(2) + (v[1] - p[1]).powi(2)
    }

    /// Least total error over the assignment graph by exhaustive enumeration:
    /// each point takes one of its own candidates at or after the previous
    /// parameter, or stays pinned at the previous parameter.
    fn enumerate_best_assignment(
        s: &ClampedCardinalBSpline<f64, Const<2>, Const<3>>,
        points: &[[f64; 2]],
    ) -> f64 {
        fn go(
            s: &ClampedCardinalBSpline<f64, Const<2>, Const<3>>,
            candidates: &[Vec<f64>],
            points: &[[f64; 2]],
            i: usize,
            prev: f64,
            acc: f64,
            best: &mut f64,
        ) {
            if i == points.len() {
                *best = best.min(acc);
                return;
            }
            let mut options: Vec<f64> = candidates[i]
                .iter()
                .copied()
                .filter(|&t| t >= prev)
                .collect();
            if prev.is_finite() {
                options.push(prev);
            }
            for t in options {
                let cost = squared_distance(s, t, &points[i]);
                go(s, candidates, points, i + 1, t, acc + cost, best);
            }
        }
        let candidates: Vec<Vec<f64>> = points
            .iter()
            .map(|p| {
                s.locally_closest_points(p, Default::default())
                    .map(|(t, _)| t)
                    .collect()
            })
            .collect();
        let mut best = f64::INFINITY;
        go(s, &candidates, points, 0, f64::NEG_INFINITY, 0.0, &mut best);
        best
    }

    #[test]
    fn ordered_assignment_recovers_points_on_the_spline() {
        let s = wiggle_spline();
        let ts = [0.2, 0.7, 1.3, 2.0, 2.9];
        let pts: Vec<[f64; 2]> = ts
            .iter()
            .map(|&t| {
                let v = s.evaluate(t);
                [v[0], v[1]]
            })
            .collect();
        let (assignment, err) = s.best_ordered_assignment(&pts, Default::default());
        assert_close(err, 0.0, 1e-9);
        for (a, &t) in assignment.iter().zip(&ts) {
            assert_close(*a, t, 1e-6);
        }
    }

    #[test]
    fn ordered_assignment_pins_out_of_order_points() {
        let s = line_spline();
        // Independently, the closest parameters would be t=3 then t=2, which is out
        // of order; the points tie, and the tied pair's optimal shared parameter is
        // the projection of their centroid: t=2.5, with error 2 * (0.5^2 + 1).
        let (assignment, err) =
            s.best_ordered_assignment(&[[2.0, 1.0], [1.0, 1.0]], Default::default());
        assert_close(assignment[0], 2.5, 1e-8);
        assert_close(assignment[1], 2.5, 1e-8);
        assert_close(err, 2.5, 1e-8);
    }

    #[test]
    fn ordered_assignment_pools_a_reversed_run() {
        let s = line_spline();
        // Fully reversed points: the isotonic optimum ties all three at their
        // centroid's projection t=2, with error (1+1) + (0+1) + (1+1).
        let points = [[2.0, 1.0], [1.0, 1.0], [0.0, 1.0]];
        let (assignment, err) = s.best_ordered_assignment(&points, Default::default());
        for a in &assignment {
            assert_close(*a, 2.0, 1e-8);
        }
        assert_close(err, 5.0, 1e-8);
    }

    #[test]
    fn ordered_assignment_matches_graph_enumeration() {
        let s = wiggle_spline();
        let point_sets: [&[[f64; 2]]; 3] = [
            &[[2.5, 0.0], [2.0, 3.0], [3.5, -3.0], [6.0, 2.0]],
            &[[3.0, 2.5], [1.0, 2.5]],
            &[[0.5, 1.0], [2.5, 0.0], [2.0, -3.0], [4.5, 1.0]],
        ];
        for points in point_sets {
            let (assignment, err) = s.best_ordered_assignment(points, Default::default());
            for w in assignment.windows(2) {
                assert!(w[0] <= w[1], "assignment not monotonic: {assignment:?}");
            }
            // The reported error is achieved by the reported assignment...
            let direct: f64 = assignment
                .iter()
                .zip(points)
                .map(|(&t, p)| squared_distance(&s, t, p))
                .sum();
            assert_close(err, direct, 1e-9);
            // ...and is no worse than the best path through the assignment graph
            // (the tied-run post-pass can improve on the graph optimum).
            let graph_best = enumerate_best_assignment(&s, points);
            assert!(
                err <= graph_best + 1e-9,
                "{err} worse than graph optimum {graph_best}"
            );
        }
    }

    #[test]
    fn bounded_candidate_pruning_matches_full_search() {
        // The geometric range pruning must never change the optimum: for many random
        // splines and (possibly wild, out-of-order) point sets, the pruned candidate
        // generation yields the identical assignment and error as the full search.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        let tol = VariableTolerance::default();
        let mut pruned_some = false;
        for trial in 0..400 {
            let m = 2 + (rng() * 7.0) as usize; // 2..=8 control points
            let ctrl: Vec<[f64; 2]> = (0..m)
                .map(|_| [rng() * 10.0 - 5.0, rng() * 10.0 - 5.0])
                .collect();
            let s = spline(&ctrl);
            let n = 1 + (rng() * 8.0) as usize; // 1..=8 points
            let pts: Vec<[f64; 2]> = (0..n)
                .map(|_| [rng() * 14.0 - 7.0, rng() * 14.0 - 7.0])
                .collect();

            // Bounded (pruned) path.
            let (a_b, e_b) = s.best_ordered_assignment(&pts, tol);
            // Full path: unpruned candidates through the identical DP + post-pass.
            let spans = s.span_polys();
            let full: Vec<Vec<f64>> = pts
                .iter()
                .map(|p| {
                    s.locally_closest_in(&spans, p, tol)
                        .into_iter()
                        .map(|(t, _)| t)
                        .collect()
                })
                .collect();
            // Full domain ranges disable the DP banding, so this is also a differential
            // check of the banded DP (pruned path) against the plain full-grid DP.
            let full_bands = vec![(0.0, s.num_spans() as f64); pts.len()];
            let domain = (0.0, s.num_spans() as f64);
            let (a_f, e_f) =
                s.assignment_from_candidates(&spans, &pts, &full, &full_bands, domain, tol);
            // The pruned path and the full search are the same f64 computation, so they
            // agree to round-off.
            let (err_tol, param_tol) = (1e-9, 1e-6);
            pruned_some |= s
                .search_ranges(
                    &spans,
                    &pts,
                    &|k, p: &[f64; 2]| {
                        let b = s.span_control_bboxes();
                        let (lo, hi) = &b[k];
                        (0..2)
                            .map(|d| (lo[d] - p[d]).max(p[d] - hi[d]).max(0.0).powi(2))
                            .sum::<f64>()
                    },
                    s.num_spans(),
                    tol,
                    0.0,
                )
                .iter()
                .any(|&(lo, hi)| lo > 0 || hi < s.num_spans());

            assert!(
                (e_b - e_f).abs() <= err_tol * (1.0 + e_f.abs()),
                "trial {trial}: bounded err {e_b} != full err {e_f}"
            );
            assert_eq!(a_b.len(), a_f.len());
            for (x, y) in a_b.iter().zip(&a_f) {
                assert_close(*x, *y, param_tol);
            }
        }
        assert!(pruned_some, "the bound never pruned — test is vacuous");
    }

    #[test]
    fn ordered_assignment_trivial_cases() {
        let s = wiggle_spline();

        let (assignment, err) = s.best_ordered_assignment(&[], Default::default());
        assert!(assignment.is_empty());
        assert_eq!(err, 0.0);

        let query = [2.0, 3.0];
        let (assignment, err) = s.best_ordered_assignment(&[query], Default::default());
        let (best_t, best_e) = s
            .locally_closest_points(&query, Default::default())
            .min_by(|x, y| x.1.partial_cmp(&y.1).expect("finite"))
            .expect("nonempty");
        assert_eq!(assignment.len(), 1);
        let param_tol = 1e-9;
        assert_close(assignment[0], best_t, param_tol);
        assert_close(err, best_e, param_tol);
    }

    #[test]
    fn fit_recovers_a_spline_from_its_own_samples() {
        let target = wiggle_spline();
        let t_end = target.num_spans() as f64;
        let n = 30;
        let ts: Vec<f64> = (0..n).map(|i| t_end * i as f64 / (n - 1) as f64).collect();
        let pts: Vec<[f64; 2]> = ts
            .iter()
            .map(|&t| {
                let v = target.evaluate(t);
                [v[0], v[1]]
            })
            .collect();
        let (fitted, assignment, err) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(
            6,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &pts,
        );
        assert_close(err, 0.0, 1e-8);
        for (a, &t) in assignment.iter().zip(&ts) {
            assert_close(*a, t, 1e-4);
        }
        // The fitted spline matches the target everywhere, not just at the samples.
        for i in 0..=30 {
            let t = t_end * i as f64 / 30.0;
            let (f, g) = (fitted.evaluate(t), target.evaluate(t));
            assert_close(f[0], g[0], 1e-4);
            assert_close(f[1], g[1], 1e-4);
        }
    }

    #[test]
    fn fit_has_linear_precision() {
        // Points on a straight line are reproduced exactly.
        let n = 12;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| [5.0 * i as f64 / (n - 1) as f64, 1.0])
            .collect();
        let (fitted, assignment, err) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(
            5,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &pts,
        );
        assert_close(err, 0.0, 1e-8);
        for w in assignment.windows(2) {
            assert!(w[0] <= w[1], "assignment not monotonic: {assignment:?}");
        }
        for (&t, p) in assignment.iter().zip(&pts) {
            let v = fitted.evaluate(t);
            assert_close(v[0], p[0], 1e-4);
            assert_close(v[1], p[1], 1e-4);
        }
    }

    #[test]
    fn fit_reports_consistent_error_and_monotonic_assignment() {
        // Deterministic "noise" around the wiggle spline's shape.
        let target = wiggle_spline();
        let n = 25;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let t = target.num_spans() as f64 * i as f64 / (n - 1) as f64;
                let v = target.evaluate(t);
                let bump = 0.05 * (i as f64 * 2.399).sin();
                [v[0] + bump, v[1] - bump]
            })
            .collect();
        let tol = FitTolerance {
            rel_metric: 1e-3,
            ..Default::default()
        };
        let (fitted, assignment, err) =
            CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(6, tol, &pts);
        for w in assignment.windows(2) {
            assert!(w[0] <= w[1], "assignment not monotonic: {assignment:?}");
        }
        // The reported error is achieved by the reported assignment on the
        // reported spline...
        let direct: f64 = assignment
            .iter()
            .zip(&pts)
            .map(|(&t, p)| squared_distance(&fitted, t, p))
            .sum();
        assert_close(err, direct, 1e-9);
        // ...and the fit is no worse than the generating spline's own best
        // assignment of these points.
        let (_, target_err) = target.best_ordered_assignment(&pts, tol.into());
        assert!(
            err <= target_err + 1e-6,
            "{err} worse than the generating spline's {target_err}"
        );
    }

    #[test]
    fn fit_fewer_points_than_control_point() {
        // Underdetermined: the proximal prior pins the unsupported directions and
        // the three points are still interpolated.
        let pts = [[0.0, 0.0], [2.0, 1.0], [5.0, -1.0]];
        let (fitted, assignment, err) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(
            6,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &pts,
        );
        assert_close(err, 0.0, 1e-4);
        for (&t, p) in assignment.iter().zip(&pts) {
            let v = fitted.evaluate(t);
            assert_close(v[0], p[0], 1e-2);
            assert_close(v[1], p[1], 1e-2);
        }
    }

    #[test]
    fn fit_single_point() {
        let (fitted, assignment, err) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(
            4,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &[[1.5, -2.0]],
        );
        assert_eq!(assignment.len(), 1);
        assert_close(err, 0.0, 1e-9);
        let v = fitted.evaluate(assignment[0]);
        assert_close(v[0], 1.5, 1e-6);
        assert_close(v[1], -2.0, 1e-6);
    }

    #[test]
    fn adaptive_fit_recovers_a_spline_from_its_own_samples() {
        let target = wiggle_spline();
        let t_end = target.num_spans() as f64;
        let n = 30;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let v = target.evaluate(t_end * i as f64 / (n - 1) as f64);
                [v[0], v[1]]
            })
            .collect();
        // A small per-control-point cost admits enough control points to reproduce the target.
        let (fitted, assignment, err) =
            CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic_adaptive(
                1e-6,
                FitTolerance {
                    rel_metric: 0.0,
                    abs_metric: 0.0,
                    ..Default::default()
                },
                &pts,
            );
        assert_close(err, 0.0, 1e-6);
        for w in assignment.windows(2) {
            assert!(w[0] <= w[1], "assignment not monotonic: {assignment:?}");
        }
        // The reported error is achieved by the reported assignment.
        let direct: f64 = assignment
            .iter()
            .zip(&pts)
            .map(|(&t, p)| squared_distance(&fitted, t, p))
            .sum();
        assert_close(err, direct, 1e-9);
    }

    #[test]
    fn adaptive_fit_keeps_a_line_at_two_control_points() {
        // Collinear points are fit exactly by the 2-knot spline, so no larger knot
        // count can buy any error reduction.
        let n = 12;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| [5.0 * i as f64 / (n - 1) as f64, 1.0])
            .collect();
        let (fitted, _, err) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic_adaptive(
            1e-3,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &pts,
        );
        assert_eq!(fitted.control_points.nrows(), 2);
        assert_close(err, 0.0, 1e-8);
    }

    #[test]
    fn adaptive_fit_large_cost_stays_minimal() {
        let target = wiggle_spline();
        let n = 20;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let v = target.evaluate(target.num_spans() as f64 * i as f64 / (n - 1) as f64);
                [v[0], v[1]]
            })
            .collect();
        // A cost dwarfing any possible error reduction forbids all growth.
        let tol = FitTolerance {
            rel_metric: 0.0,
            abs_metric: 0.0,
            ..Default::default()
        };
        let (fitted, _, err) =
            CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic_adaptive(1e6, tol, &pts);
        assert_eq!(fitted.control_points.nrows(), 2);
        let (_, expected_ts, expected_err) =
            CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(2, tol, &pts);
        assert_eq!(expected_ts.len(), n);
        assert_close(err, expected_err, 1e-9);
    }

    #[test]
    fn adaptive_fit_beats_or_matches_the_minimal_fit() {
        // Noisy wiggle samples: whatever knot count is chosen, the combined
        // objective (error + cost * control_points) is no worse than staying at 2 control_points.
        let target = wiggle_spline();
        let n = 25;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let t = target.num_spans() as f64 * i as f64 / (n - 1) as f64;
                let v = target.evaluate(t);
                let bump = 0.05 * (i as f64 * 2.399).sin();
                [v[0] + bump, v[1] - bump]
            })
            .collect();
        let cost = 0.01;
        let tol = FitTolerance {
            rel_metric: 1e-3,
            ..Default::default()
        };
        let (fitted, _, err) =
            CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic_adaptive(cost, tol, &pts);
        let m = fitted.control_points.nrows();
        assert!(m >= 2);
        let (_, _, base_err) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(2, tol, &pts);
        assert!(
            err + cost * m as f64 <= base_err + cost * 2.0 + 1e-9,
            "objective {} at m={m} worse than {} at m=2",
            err + cost * m as f64,
            base_err + cost * 2.0
        );
    }

    #[test]
    #[should_panic(expected = "control_point_cost must be positive")]
    fn adaptive_fit_rejects_nonpositive_cost() {
        let _ = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic_adaptive(
            0.0,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &[[0.0, 0.0]],
        );
    }

    #[test]
    #[should_panic(expected = "zero points")]
    fn adaptive_fit_rejects_no_points() {
        let _ = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic_adaptive(
            1.0,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "at least 2 control points")]
    fn fit_rejects_too_few_control_points() {
        let _ = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(
            1,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &[[0.0, 0.0]],
        );
    }

    #[test]
    #[should_panic(expected = "absolute variable tolerance must be positive")]
    fn fit_tolerance_rejects_zero_abs_variable() {
        let _ = FitTolerance::new(0.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "relative metric tolerance must be non-negative")]
    fn fit_tolerance_rejects_negative_rel_metric() {
        let _ = FitTolerance::new(1.0, -1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "zero points")]
    fn fit_rejects_no_points() {
        let _ = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(
            4,
            FitTolerance {
                rel_metric: 0.0,
                ..Default::default()
            },
            &[],
        );
    }

    /// Deterministic noisy samples of the wiggle spline over `[from, to]` of its domain.
    fn wiggle_samples(n: usize, from: f64, to: f64) -> Vec<[f64; 2]> {
        let target = wiggle_spline();
        let t_end = target.num_spans() as f64;
        (0..n)
            .map(|i| {
                let t = t_end * (from + (to - from) * i as f64 / (n - 1) as f64);
                let v = target.evaluate(t);
                let bump = 0.05 * (i as f64 * 2.399).sin();
                [v[0] + bump, v[1] - bump]
            })
            .collect()
    }

    #[test]
    fn a_refit_cannot_move_the_frozen_curve() {
        let pts = wiggle_samples(40, 0.0, 1.0);
        let tol = FitTolerance::<f64>::default();
        let (mut s, _, _) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(8, tol, &pts);
        let frozen = 4;
        let until = s.frozen_until(frozen);
        assert_close(until, 3.0, 1e-12);

        let before_control = s.control_points().clone();
        let before_curve: Vec<_> = (0..=30)
            .map(|i| s.evaluate(until * i as f64 / 30.0))
            .collect();

        // Refit against quite different data: without the freeze it would drag the whole
        // curve, so anything that survives really is pinned by the frozen prefix.
        let other = wiggle_samples(40, 0.0, 1.0)
            .iter()
            .map(|p| [p[0] * 0.5 - 1.0, p[1] + 2.0])
            .collect::<Vec<_>>();
        let settled = Settled {
            control_points: frozen,
            after: 0.0,
            tail: 0,
        };
        let (ts, _) = s.refit_monotonic(settled, tol, &other);
        assert_eq!(ts.len(), other.len());

        // The frozen control points come back bit-identical...
        for j in 0..frozen {
            for d in 0..2 {
                assert_eq!(s.control_points()[(j, d)], before_control[(j, d)]);
            }
        }
        // ...so does the curve they determine, exactly...
        for (i, before) in before_curve.iter().enumerate() {
            let after = s.evaluate(until * i as f64 / 30.0);
            assert_eq!(after[0], before[0]);
            assert_eq!(after[1], before[1]);
        }
        // ...and the free control points did move (otherwise this proves nothing).
        let moved = (frozen..s.num_control_points())
            .any(|j| (0..2).any(|d| s.control_points()[(j, d)] != before_control[(j, d)]));
        assert!(moved, "the free control points should have refit");
    }

    #[test]
    fn a_fully_frozen_refit_only_assigns() {
        let pts = wiggle_samples(30, 0.0, 1.0);
        let tol = FitTolerance::<f64>::default();
        let (mut s, ts, err) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(6, tol, &pts);
        let before = s.control_points().clone();
        let settled = Settled {
            control_points: s.num_control_points(),
            after: 0.0,
            tail: 0,
        };
        let (refit_ts, refit_err) = s.refit_monotonic(settled, tol, &pts);
        assert_eq!(*s.control_points(), before);
        // Nothing can improve, so this is just the E-step of the fit we already had.
        assert_close(refit_err, err, 1e-9);
        for (a, b) in refit_ts.iter().zip(&ts) {
            assert_close(*a, *b, 1e-6);
        }
    }

    #[test]
    fn a_refit_with_nothing_settled_holds_a_converged_fit() {
        let pts = wiggle_samples(40, 0.0, 1.0);
        let tol = FitTolerance::<f64> {
            rel_metric: 0.0,
            ..Default::default()
        };
        let (mut s, _, err) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(7, tol, &pts);
        // A fixed point of the EM map: refitting it can only improve on it, never undo it.
        let (_, refit_err) = s.refit_monotonic(Settled::none(), tol, &pts);
        assert!(
            refit_err <= err + 1e-9,
            "refit err {refit_err} worse than the converged {err}"
        );
    }

    /// A channel that is an exact cubic-B-spline function of the parameter must come back
    /// exactly, and evaluating the returned control values with the same basis must
    /// reproduce it at the assigned parameters.
    #[test]
    fn channel_fitting_recovers_an_exact_channel() {
        let s = wiggle_spline();
        let m = s.num_control_points();
        // Ground-truth channel control values, arbitrary but fixed.
        let truth = OMatrix::<f64, Dyn, Const<1>>::from_fn_generic(Dyn(m), Const::<1>, |j, _| {
            0.3 + 0.7 * (j as f64 * 1.7).sin()
        });
        // Sample that channel spline at a spread of parameters, then re-fit from the samples.
        let basis = s.basis_matrix();
        let channel_at = |t: f64| {
            let (k, u) = s.span_and_local(t);
            let w = basis * s.u_powers(u);
            (0..w.len())
                .map(|a| truth[(s.knot_row(k + a), 0)] * w[a])
                .sum::<f64>()
        };
        let ts: Vec<f64> = (0..60)
            .map(|i| s.num_spans() as f64 * i as f64 / 59.0)
            .collect();
        let values: Vec<[f64; 1]> = ts.iter().map(|&t| [channel_at(t)]).collect();

        let empty = OMatrix::<f64, Dyn, Const<1>>::zeros_generic(Dyn(0), Const::<1>);
        let got = s.fit_channels(&ts, &values, 0, &empty);
        assert_eq!(got.nrows(), m);
        for j in 0..m {
            assert_close(got[(j, 0)], truth[(j, 0)], 1e-6);
        }
    }

    /// With nothing left to fit against — every point retired into the frozen prefix, which
    /// an incremental fit reaches routinely — the solve is all ridge, and must hand the
    /// prior straight back rather than divide by a `lambda` that `n = 0` pinned to zero.
    #[test]
    fn a_solve_with_no_points_returns_the_prior() {
        let s = wiggle_spline();
        let m = s.num_control_points();
        let prior = OMatrix::<f64, Dyn, Const<2>>::from_fn_generic(Dyn(m), Const::<2>, |j, d| {
            (j * 2 + d) as f64
        });
        let got = s.fit_channels(&[], &[], 0, &prior);
        assert_eq!(got, prior);
    }

    /// The channel solve honours a frozen prefix exactly as the geometric one does, and
    /// accepts a prior shorter than the (since-grown) control polygon.
    #[test]
    fn frozen_channel_rows_come_back_untouched() {
        let s = wiggle_spline();
        let m = s.num_control_points();
        let pts = wiggle_samples(40, 0.0, 1.0);
        let (ts, _) = s.best_ordered_assignment(&pts, VariableTolerance::default());
        let values: Vec<[f64; 2]> = (0..pts.len())
            .map(|i| [i as f64 / pts.len() as f64, (i as f64 * 0.3).cos()])
            .collect();

        // A prior two rows short of the polygon: the tail is seeded, not required.
        let short =
            OMatrix::<f64, Dyn, Const<2>>::from_fn_generic(Dyn(m - 2), Const::<2>, |j, d| {
                (j + d) as f64 * 0.25
            });
        let frozen = 3;
        let got = s.fit_channels(&ts, &values, frozen, &short);
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

    #[test]
    fn the_assignment_floor_is_respected() {
        let s = wiggle_spline();
        let pts = wiggle_samples(25, 0.0, 1.0);
        let tol = VariableTolerance::default();
        for &after in &[0.5, 1.0, 2.75, 4.0] {
            let (ts, err) = s.best_ordered_assignment_after(&pts, tol, after);
            assert!(
                ts.iter().all(|&t| t >= after - 1e-9),
                "assignment fell below the floor {after}: {ts:?}"
            );
            assert!(ts.windows(2).all(|w| w[0] <= w[1] + 1e-9));
            let direct: f64 = ts
                .iter()
                .zip(&pts)
                .map(|(&t, p)| squared_distance(&s, t, p))
                .sum();
            assert_close(err, direct, 1e-9);
        }
    }

    #[test]
    fn floored_assignment_matches_an_unpruned_constrained_search() {
        // The floor must be enforced without disturbing the optimum it still admits: for
        // random splines, point sets and floors, the production path (pruned ranges, banded
        // DP) must agree with the plain full-grid DP over candidates filtered to the floor.
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        let mut rng = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        let tol = VariableTolerance::default();
        for trial in 0..300 {
            let m = 2 + (rng() * 7.0) as usize;
            let ctrl: Vec<[f64; 2]> = (0..m)
                .map(|_| [rng() * 10.0 - 5.0, rng() * 10.0 - 5.0])
                .collect();
            let s = spline(&ctrl);
            let n = 1 + (rng() * 8.0) as usize;
            let pts: Vec<[f64; 2]> = (0..n)
                .map(|_| [rng() * 14.0 - 7.0, rng() * 14.0 - 7.0])
                .collect();
            let t_end = s.num_spans() as f64;
            let after = rng() * t_end;

            let (a_b, e_b) = s.best_ordered_assignment_after(&pts, tol, after);
            assert!(
                a_b.iter().all(|&t| t >= after - 1e-9),
                "trial {trial}: assignment fell below the floor {after}"
            );

            // Reference: every locally-closest point, kept only where the floor allows and
            // seeded with the floor itself, through the identical DP and post-pass.
            let spans = s.span_polys();
            let full: Vec<Vec<f64>> = pts
                .iter()
                .map(|p| {
                    let mut c: Vec<f64> = s
                        .locally_closest_in(&spans, p, tol)
                        .into_iter()
                        .map(|(t, _)| t)
                        .filter(|&t| t >= after)
                        .collect();
                    c.push(after);
                    c
                })
                .collect();
            let bands = vec![(after, t_end); pts.len()];
            let (_, e_f) =
                s.assignment_from_candidates(&spans, &pts, &full, &bands, (after, t_end), tol);

            let err_tol = 1e-9;
            assert!(
                (e_b - e_f).abs() <= err_tol * (1.0 + e_f.abs()),
                "trial {trial}: floored err {e_b} != reference {e_f} (after {after})"
            );
        }
    }

    #[test]
    fn extending_control_points_leaves_the_frozen_curve_alone() {
        let pts = wiggle_samples(30, 0.0, 1.0);
        let tol = FitTolerance::<f64>::default();
        let (mut s, _, _) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(6, tol, &pts);
        let m0 = s.num_control_points();
        // Everything the existing control points determine survives growth.
        let until = s.frozen_until(m0);
        let before: Vec<_> = (0..=20)
            .map(|i| s.evaluate(until * i as f64 / 20.0))
            .collect();

        let new_points = wiggle_samples(10, 1.0, 1.6);
        s.extend_control_points(m0 + 4, &new_points);
        assert_eq!(s.num_control_points(), m0 + 4);
        assert_eq!(s.num_spans(), m0 + 5);
        for (i, b) in before.iter().enumerate() {
            let a = s.evaluate(until * i as f64 / 20.0);
            assert_eq!(a[0], b[0]);
            assert_eq!(a[1], b[1]);
        }
        // The appended control points trace the new data, ending on its last point.
        let last = new_points.last().expect("nonempty");
        assert_close(s.control_points()[(m0 + 3, 0)], last[0], 1e-12);
        assert_close(s.control_points()[(m0 + 3, 1)], last[1], 1e-12);
        // And a shorter (or equal) request is a no-op.
        s.extend_control_points(m0, &new_points);
        assert_eq!(s.num_control_points(), m0 + 4);
    }

    #[test]
    #[should_panic(expected = "cannot freeze 9 of 6 control points")]
    fn refit_rejects_freezing_more_than_it_has() {
        let pts = wiggle_samples(20, 0.0, 1.0);
        let tol = FitTolerance::<f64>::default();
        let (mut s, _, _) = CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(6, tol, &pts);
        let _ = s.refit_monotonic(
            Settled {
                control_points: 9,
                after: 0.0,
                tail: 0,
            },
            tol,
            &pts,
        );
    }

    #[test]
    fn evaluate_batch_matches_evaluate() {
        let s = wiggle_spline();
        let t_end = s.num_spans() as f64;
        let ts: Vec<f64> = (0..=40).map(|i| t_end * i as f64 / 40.0).collect();
        // `evaluate_many` shares the basis-matrix path while `evaluate` uses De
        // Boor's algorithm, so they agree only up to floating-point rounding.
        for (v, &t) in s.evaluate_many(ts.iter().copied()).zip(&ts) {
            let direct = s.evaluate(t);
            assert_close(v[0], direct[0], 1e-12);
            assert_close(v[1], direct[1], 1e-12);
        }
    }
}
