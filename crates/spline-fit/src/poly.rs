//! Dense-polynomial utilities used by the spline routines.
//!
//! Polynomials are stack-allocated coefficient vectors in the monomial basis,
//! ascending: `p[i]` is the coefficient of `x^i`.

use nalgebra::{RealField, SVector, convert};

pub type Poly<F, const N: usize> = SVector<F, N>;

/// Evaluates `p` at `x` (Horner's method).
pub fn eval<F: RealField + Copy, const N: usize>(p: &Poly<F, N>, x: F) -> F {
    p.iter().rev().fold(F::zero(), |acc, &c| acc * x + c)
}

/// The derivative of `p`.
pub fn derivative<F: RealField + Copy, const N: usize>(p: &Poly<F, N>) -> Poly<F, { N - 1 }>
where
    [(); N - 1]: Sized,
{
    Poly::from_fn(|i, _| p[i + 1] * convert::<f64, F>((i + 1) as f64))
}

/// The product of `a` and `b`.
pub fn mul<F: RealField + Copy, const N: usize, const M: usize>(
    a: &Poly<F, N>,
    b: &Poly<F, M>,
) -> Poly<F, { N + M - 1 }>
where
    [(); N + M - 1]: Sized,
{
    let mut out = Poly::zeros();
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

fn binomial(n: usize, k: usize) -> f64 {
    let mut b = 1.0;
    for i in 0..k {
        b = b * (n - i) as f64 / (i + 1) as f64;
    }
    b
}

/// Bernstein-basis coefficients of `p` over `[0, 1]`.
fn to_bernstein<F: RealField + Copy, const N: usize>(p: &Poly<F, N>) -> Poly<F, N> {
    let deg = N - 1;
    Poly::from_fn(|i, _| {
        let mut b = F::zero();
        for (j, &c) in p.iter().take(i + 1).enumerate() {
            b += c * convert(binomial(i, j) / binomial(deg, j));
        }
        b
    })
}

/// Splits Bernstein coefficients at the interval midpoint (de Casteljau).
fn split_half<F: RealField + Copy, const N: usize>(bern: &Poly<F, N>) -> (Poly<F, N>, Poly<F, N>) {
    let half = convert::<f64, F>(0.5);
    let mut tmp = *bern;
    let mut left = Poly::zeros();
    let mut right = Poly::zeros();
    left[0] = tmp[0];
    right[N - 1] = tmp[N - 1];
    for lvl in 1..N {
        for i in 0..(N - lvl) {
            tmp[i] = (tmp[i] + tmp[i + 1]) * half;
        }
        left[lvl] = tmp[0];
        right[N - 1 - lvl] = tmp[N - lvl - 1];
    }
    (left, right)
}

const MAX_DEPTH: u32 = 60;

/// Recursive-Descartes root isolation over `[a, b]`, where `bern` holds the Bernstein
/// coefficients of `p` reparametrized to that interval (so `bern[0] = p(a)`,
/// `bern[N-1] = p(b)`). By the convex-hull property and Descartes' rule of signs in the
/// Bernstein basis:
/// * coefficients all one strict sign ⇒ no root ⇒ discard;
/// * one sign change with the endpoints strictly straddling zero ⇒ a single simple root
///   that the endpoints bracket ⇒ solve it directly with safeguarded Newton;
/// * otherwise split at the midpoint and recurse, isolating each root into its own
///   subinterval.
///
/// Intervals that narrow to `tol` (or hit the depth cap) without isolating a lone sign
/// change emit a polished midpoint — this covers clustered or even-multiplicity roots
/// that never separate. Cannot miss a root (every non-discarded interval is subdivided).
// A root-isolation recursion: the polynomial, its derivative and Bernstein form, the
// bracket, and the three stopping controls. Bundling them would only rename the noise.
#[allow(clippy::too_many_arguments)]
fn isolate<F: RealField + Copy, const N: usize>(
    p: &Poly<F, N>,
    dp: &Poly<F, { N - 1 }>,
    bern: &Poly<F, N>,
    a: F,
    b: F,
    tol: F,
    depth: u32,
    out: &mut Vec<F>,
) where
    [(); N - 1]: Sized,
{
    // Descartes in the Bernstein basis: with no sign change there is no root strictly
    // inside (a, b). A control polygon that only *touches* zero at an endpoint is this
    // case — and the endpoint is itself a root. This happens for every query point at a
    // clamped span whose tangent vanishes at the domain boundary (the first/last span),
    // where `f = C'·(C − p)` has `f(endpoint) = 0`. Record the zero endpoint(s) directly
    // instead of subdividing toward the zero, which otherwise recurses to `tol` down the
    // interval's spine (≈10× the cost of every other span at the clamped ends).
    let changes = sign_changes(bern);
    if changes == 0 {
        // The clamped vanishing-tangent case gives an *exact* zero here (`f(endpoint) = 0`
        // from `P0 − P0` control differences); a de-Casteljau cluster boundary is only
        // ever tiny-but-nonzero, so the exact test isolates the intended case.
        if bern[0] == F::zero() {
            out.push(a);
        }
        if bern[N - 1] == F::zero() {
            out.push(b);
        }
        return;
    }
    let ends_bracket = (bern[0] > F::zero()) != (bern[N - 1] > F::zero())
        && bern[0] != F::zero()
        && bern[N - 1] != F::zero();
    if ends_bracket && changes == 1 {
        out.push(newton_bracketed(p, dp, a, b, tol));
        return;
    }
    let half = convert::<f64, F>(0.5);
    let mid = (a + b) * half;
    if b - a <= tol || depth >= MAX_DEPTH {
        out.push(polish(p, dp, mid, tol));
        return;
    }
    let (l, r) = split_half(bern);
    isolate(p, dp, &l, a, mid, tol, depth + 1, out);
    isolate(p, dp, &r, mid, b, tol, depth + 1, out);
}

/// Number of sign changes in a coefficient sequence, ignoring zeros. In the Bernstein
/// basis this is the Descartes bound: the number of roots in `(0, 1)`, with multiplicity,
/// is at most this and has the same parity.
fn sign_changes<F: RealField + Copy, const N: usize>(bern: &Poly<F, N>) -> usize {
    let mut changes = 0;
    let mut prev = 0i32;
    for c in bern.iter() {
        let s = if *c > F::zero() {
            1
        } else if *c < F::zero() {
            -1
        } else {
            0
        };
        if s != 0 {
            if prev != 0 && s != prev {
                changes += 1;
            }
            prev = s;
        }
    }
    changes
}

/// Safeguarded (bisection-guarded) Newton for the single root of `p` in `[x1, x2]`,
/// given that the endpoints bracket it. Newton where it stays in the bracket and reduces
/// the step fast enough, bisection otherwise — so it converges for any bracketed root.
/// The caller uses this only when the Bernstein sign pattern proves exactly one simple
/// root exists in the interval (one sign change, endpoints strictly opposite in sign).
fn newton_bracketed<F: RealField + Copy, const N: usize>(
    p: &Poly<F, N>,
    dp: &Poly<F, { N - 1 }>,
    x1: F,
    x2: F,
    tol: F,
) -> F
where
    [(); N - 1]: Sized,
{
    let two = convert::<f64, F>(2.0);
    let half = convert::<f64, F>(0.5);
    // Orient the bracket so `f(xl) < 0 < f(xh)`.
    let (mut xl, mut xh) = if eval(p, x1) < F::zero() {
        (x1, x2)
    } else {
        (x2, x1)
    };
    let mut rts = (x1 + x2) * half;
    let mut dxold = (x2 - x1).abs();
    let mut dx = dxold;
    let mut f = eval(p, rts);
    let mut df = eval(dp, rts);
    for _ in 0..60 {
        // Bisect when the Newton iterate would jump outside the bracket or is
        // shrinking the step too slowly; otherwise take the Newton step.
        let out_of_range = ((rts - xh) * df - f) * ((rts - xl) * df - f) > F::zero();
        if out_of_range || (two * f).abs() > (dxold * df).abs() || df == F::zero() {
            dxold = dx;
            dx = half * (xh - xl);
            rts = xl + dx;
            if xl == rts {
                return rts;
            }
        } else {
            dxold = dx;
            dx = f / df;
            let temp = rts;
            rts -= dx;
            if temp == rts {
                return rts;
            }
        }
        if dx.abs() < tol {
            return rts;
        }
        f = eval(p, rts);
        df = eval(dp, rts);
        if f < F::zero() {
            xl = rts;
        } else {
            xh = rts;
        }
    }
    rts
}

/// Newton refinement of a bracketed root, constrained near its bracket.
fn polish<F: RealField + Copy, const N: usize>(
    p: &Poly<F, N>,
    dp: &Poly<F, { N - 1 }>,
    x0: F,
    tol: F,
) -> F
where
    [(); N - 1]: Sized,
{
    let lo = (x0 - tol - tol).max(F::zero());
    let hi = (x0 + tol + tol).min(F::one());
    let mut x = x0;
    for _ in 0..12 {
        let d = eval(dp, x);
        if d == F::zero() {
            break;
        }
        let next = (x - eval(p, x) / d).max(lo).min(hi);
        let moved = (next - x).abs();
        x = next;
        if moved <= F::default_epsilon() {
            break;
        }
    }
    x
}

/// Appends all real roots of `p` in `[0, 1]` to `out`, ascending, deduplicated.
///
/// `out`'s existing contents are preserved; appending into a caller-owned buffer
/// keeps repeated root-finding allocation-free. An identically zero polynomial
/// reports no roots.
pub fn roots_in_unit_interval<F: RealField + Copy, const N: usize>(
    p: &Poly<F, N>,
    tol: F,
    out: &mut Vec<F>,
) where
    [(); N - 1]: Sized,
{
    let scale = p.iter().fold(F::zero(), |m, c| m.max(c.abs()));
    if scale == F::zero() {
        return;
    }
    // Roots are invariant under scaling; normalizing keeps the sign tests in `isolate`
    // meaningful regardless of the coefficients' magnitude.
    let normalized = p.map(|c| c / scale);
    let bern = to_bernstein(&normalized);

    // By the convex-hull property, coefficients all of one strict sign mean no root — the
    // common case, taken before touching the derivative or the recursion.
    if bern.iter().all(|c| *c > F::zero()) || bern.iter().all(|c| *c < F::zero()) {
        return;
    }
    // Otherwise isolate every root by recursive Descartes: single-root subintervals are
    // solved directly by Newton, only genuinely multi-root regions keep subdividing.
    let dp = derivative(&normalized);
    let start = out.len();
    isolate(&normalized, &dp, &bern, F::zero(), F::one(), tol, 0, out);
    if out.len() == start {
        return;
    }
    out[start..].sort_by(|x, y| x.partial_cmp(y).expect("roots are finite"));
    // Deduplicate the appended range: keep the first of each near-equal cluster.
    let merge = tol * convert::<f64, F>(4.0);
    let mut last = start;
    for r in (start + 1)..out.len() {
        if out[r] - out[last] > merge {
            last += 1;
            out[last] = out[r];
        }
    }
    out.truncate(last + 1);
}

/// The smallest real root of `p` strictly greater than `lo` within `[0, 1]`, or `None`
/// if there is none.
///
/// Isolates roots left-to-right by the same recursive-Descartes scheme as
/// [`roots_in_unit_interval`], but prunes every subinterval lying entirely at or below
/// `lo` and returns the moment the leftmost qualifying root is isolated — so it never
/// pays to find the later roots. This makes "the next critical point after `t`" much
/// cheaper than the full set when only the next one is needed. An identically zero
/// polynomial reports no root.
pub fn first_root_after<F: RealField + Copy, const N: usize>(
    p: &Poly<F, N>,
    lo: F,
    tol: F,
) -> Option<F>
where
    [(); N - 1]: Sized,
{
    let scale = p.iter().fold(F::zero(), |m, c| m.max(c.abs()));
    if scale == F::zero() {
        return None;
    }
    let normalized = p.map(|c| c / scale);
    let bern = to_bernstein(&normalized);
    if bern.iter().all(|c| *c > F::zero()) || bern.iter().all(|c| *c < F::zero()) {
        return None;
    }
    let dp = derivative(&normalized);
    isolate_first(&normalized, &dp, &bern, F::zero(), F::one(), lo, tol, 0)
}

/// Leftmost root of `p` above `lo`, over `[a, b]` whose Bernstein coefficients are
/// `bern`. Same isolation logic as [`isolate`], with two changes for laziness: a
/// subinterval entirely at or below `lo` (`b <= lo`) is discarded outright, and the
/// recursion returns as soon as the left branch yields a qualifying root, so the right
/// branch (and all later spans, via the caller) is never explored.
#[allow(clippy::too_many_arguments)]
fn isolate_first<F: RealField + Copy, const N: usize>(
    p: &Poly<F, N>,
    dp: &Poly<F, { N - 1 }>,
    bern: &Poly<F, N>,
    a: F,
    b: F,
    lo: F,
    tol: F,
    depth: u32,
) -> Option<F>
where
    [(); N - 1]: Sized,
{
    if b <= lo {
        return None;
    }
    // No interior sign change ⇒ no root inside (a, b); only a zero endpoint (a clamped
    // vanishing-tangent span, see `isolate`) is a root. Return the leftmost that clears
    // `lo`, without subdividing toward the zero.
    let changes = sign_changes(bern);
    if changes == 0 {
        if bern[0] == F::zero() && a > lo {
            return Some(a);
        }
        if bern[N - 1] == F::zero() && b > lo {
            return Some(b);
        }
        return None;
    }
    let ends_bracket = (bern[0] > F::zero()) != (bern[N - 1] > F::zero())
        && bern[0] != F::zero()
        && bern[N - 1] != F::zero();
    if ends_bracket && changes == 1 {
        // Exactly one root in (a, b); keep it only if it clears `lo`.
        let r = newton_bracketed(p, dp, a, b, tol);
        return (r > lo).then_some(r);
    }
    let half = convert::<f64, F>(0.5);
    let mid = (a + b) * half;
    if b - a <= tol || depth >= MAX_DEPTH {
        let r = polish(p, dp, mid, tol);
        return (r > lo).then_some(r);
    }
    let (l, r) = split_half(bern);
    // Left branch first: any root it isolates is the leftmost overall.
    if let Some(root) = isolate_first(p, dp, &l, a, mid, lo, tol, depth + 1) {
        return Some(root);
    }
    isolate_first(p, dp, &r, mid, b, lo, tol, depth + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} != {b} (tol {tol})");
    }

    fn roots<const N: usize>(p: &Poly<f64, N>) -> Vec<f64>
    where
        [(); N - 1]: Sized,
    {
        let mut out = Vec::new();
        roots_in_unit_interval(p, f64::EPSILON.sqrt(), &mut out);
        out
    }

    #[test]
    fn eval_and_derivative() {
        // p(x) = 1 + 2x + 3x^2
        let p = Poly::from([1.0, 2.0, 3.0]);
        assert_close(eval(&p, 2.0), 17.0, 1e-12);
        assert_eq!(derivative(&p), Poly::from([2.0, 6.0]));
    }

    #[test]
    fn mul_matches_expansion() {
        // (1 + x)(2 - x) = 2 + x - x^2
        assert_eq!(
            mul(&Poly::from([1.0, 1.0]), &Poly::from([2.0, -1.0])),
            Poly::from([2.0, 1.0, -1.0])
        );
    }

    #[test]
    fn finds_simple_roots() {
        // (x - 0.3)(x - 0.7) = 0.21 - x + x^2
        let roots = roots(&Poly::from([0.21, -1.0, 1.0]));
        assert_eq!(roots.len(), 2);
        assert_close(roots[0], 0.3, 1e-9);
        assert_close(roots[1], 0.7, 1e-9);
    }

    #[test]
    fn appends_and_preserves_existing_output() {
        let mut out = vec![-7.0];
        roots_in_unit_interval(
            &Poly::from([0.21, -1.0, 1.0]),
            f64::EPSILON.sqrt(),
            &mut out,
        );
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], -7.0);
        assert_close(out[1], 0.3, 1e-9);
        assert_close(out[2], 0.7, 1e-9);
    }

    #[test]
    fn no_roots_outside_interval() {
        // (x - 2)(x + 1) = -2 - x + x^2
        assert!(roots(&Poly::from([-2.0, -1.0, 1.0])).is_empty());
    }

    #[test]
    fn single_interior_root_fast_path() {
        // One simple root in (0, 1) with the endpoints straddling it — the Descartes
        // single-root case (one Bernstein sign change) solved by safeguarded Newton.
        // f(x) = x - 0.4 (linear, one root); and a quintic with a lone interior root.
        assert_eq!(roots(&Poly::from([-0.4, 1.0])).len(), 1);
        assert_close(roots(&Poly::from([-0.4, 1.0]))[0], 0.4, 1e-12);
        // (x - 0.37)(x^2 + 1)(x^2 + 2): only x = 0.37 is real and in (0, 1).
        let p = mul(&Poly::from([-0.37, 1.0]), &Poly::from([1.0, 0.0, 1.0]));
        let p = mul(&p, &Poly::from([2.0, 0.0, 1.0]));
        let r = roots(&p);
        assert_eq!(r.len(), 1);
        assert_close(r[0], 0.37, 1e-9);
    }

    #[test]
    fn triple_root_reported_once() {
        // (x - 0.5)^3
        let factor = Poly::from([-0.5, 1.0]);
        let p = mul(&mul(&factor, &factor), &factor);
        let roots = roots(&p);
        assert_eq!(roots.len(), 1);
        assert_close(roots[0], 0.5, 1e-5);
    }

    #[test]
    fn endpoint_roots() {
        // x(1 - x): roots at both interval endpoints.
        let roots = roots(&Poly::from([0.0, 1.0, -1.0]));
        assert_eq!(roots.len(), 2);
        assert_close(roots[0], 0.0, 1e-9);
        assert_close(roots[1], 1.0, 1e-9);
    }

    #[test]
    fn zero_polynomial_has_no_roots() {
        assert!(roots(&Poly::from([0.0, 0.0, 0.0])).is_empty());
    }

    #[test]
    fn first_root_after_finds_the_next_root() {
        // (x - 0.2)(x - 0.5)(x - 0.8): three simple roots.
        let p = mul(&Poly::from([-0.2, 1.0]), &Poly::from([-0.5, 1.0]));
        let p: Poly<f64, 4> = mul(&p, &Poly::from([-0.8, 1.0]));
        let tol = f64::EPSILON.sqrt();
        assert_close(first_root_after(&p, -1.0, tol).unwrap(), 0.2, 1e-9);
        assert_close(first_root_after(&p, 0.3, tol).unwrap(), 0.5, 1e-9);
        assert_close(first_root_after(&p, 0.65, tol).unwrap(), 0.8, 1e-9);
        assert!(first_root_after(&p, 0.9, tol).is_none());
        // Re-solving from a returned root advances past it (deterministic dedup).
        let r0 = first_root_after(&p, -1.0, tol).unwrap();
        assert_close(first_root_after(&p, r0, tol).unwrap(), 0.5, 1e-9);
    }

    #[test]
    fn first_root_after_enumeration_matches_full_solve() {
        // Walking `first_root_after` from below the interval reproduces the full,
        // deduplicated root set that `roots_in_unit_interval` returns.
        let polys: [Poly<f64, 6>; 2] = [
            // (x-0.15)(x-0.4)(x-0.6)(x-0.85) padded to degree 5.
            {
                let q = mul(&Poly::from([-0.15, 1.0]), &Poly::from([-0.4, 1.0]));
                let q = mul(&q, &Poly::from([-0.6, 1.0]));
                let q = mul(&q, &Poly::from([-0.85, 1.0]));
                Poly::from_fn(|i, _| if i < 5 { q[i] } else { 0.0 })
            },
            // A quintic with a lone interior root and complex partners.
            {
                let q = mul(&Poly::from([-0.37, 1.0]), &Poly::from([1.0, 0.0, 1.0]));
                mul(&q, &Poly::from([2.0, 0.0, 1.0]))
            },
        ];
        let tol = f64::EPSILON.sqrt();
        for p in &polys {
            let mut full = Vec::new();
            roots_in_unit_interval(p, tol, &mut full);
            let mut lazy = Vec::new();
            let mut lo = -1.0;
            while let Some(r) = first_root_after(p, lo, tol) {
                lazy.push(r);
                lo = r;
            }
            assert_eq!(full.len(), lazy.len(), "full {full:?} vs lazy {lazy:?}");
            for (a, b) in full.iter().zip(&lazy) {
                assert_close(*a, *b, 1e-9);
            }
        }
    }

    #[test]
    fn first_root_after_zero_polynomial_has_no_root() {
        assert!(
            first_root_after(&Poly::from([0.0, 0.0, 0.0]), -1.0, f64::EPSILON.sqrt()).is_none()
        );
    }

    #[test]
    fn quintic_with_three_interior_roots() {
        // (x-0.2)(x-0.5)(x-0.8)(x^2+1)
        let p = mul(&Poly::from([-0.2, 1.0]), &Poly::from([-0.5, 1.0]));
        let p = mul(&p, &Poly::from([-0.8, 1.0]));
        let p = mul(&p, &Poly::from([1.0, 0.0, 1.0]));
        let roots = roots(&p);
        assert_eq!(roots.len(), 3);
        assert_close(roots[0], 0.2, 1e-9);
        assert_close(roots[1], 0.5, 1e-9);
        assert_close(roots[2], 0.8, 1e-9);
    }
}
