//! Math the replay path may use (§12.1): functions built only from IEEE-754's
//! correctly rounded operations (`+ − × ÷ √`), so native and wasm builds compute
//! them to the same bit.
//!
//! What decides where a stroke's segments are cut, and what they deposit, has to
//! agree to the last bit across replay, goldens and peers. The library
//! transcendentals (`sin`, `atan2`, …) are not specified to that — each platform's
//! libm may round differently — so anything a replay computes from a record reaches
//! for these instead.

/// `sin(x)` as a truncated Maclaurin series, **for small `x` only**: within an f32 ulp
/// (worst relative error 8.1e-8 against an f64 reference) for `|x| ≤ asin(0.2)`, the
/// range [`fit_arc`](crate::path::fit_arc) guards its callers to.
pub(crate) fn sin_small(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 / 5040.0)))
}

/// `1 − cos(x)` as a series, over [`sin_small`]'s range (worst relative error 1.2e-7).
///
/// Evaluated directly rather than by subtraction, which is not a nicety: near zero
/// `cos(x)` rounds to exactly 1, and the difference cancels every digit it had.
pub(crate) fn versin_small(x: f32) -> f32 {
    let x2 = x * x;
    x2 * (0.5 - x2 * (1.0 / 24.0 - x2 / 720.0))
}

/// `asin(u)/u` as a series, for `|u| ≤ 0.2` (worst relative error 1.3e-7).
pub(crate) fn asin_over_x(u: f32) -> f32 {
    let u2 = u * u;
    1.0 + u2 * (1.0 / 6.0 + u2 * (3.0 / 40.0 + u2 * (15.0 / 336.0)))
}

/// `acos(v.clamp(-1, 1))` to within ~1e-7 absolute: a 7-degree minimax polynomial
/// times a square root (DirectXMath's `XMScalarAcos`).
///
/// **A verbatim copy of glam 0.33's `acos_approx_f32`**, which `Vec2::angle_to` is
/// built on — so `flatten`'s turn test, which used `angle_to`, cuts exactly where it
/// did, now without depending on glam keeping that implementation.
pub(crate) fn acos_approx(v: f32) -> f32 {
    let nonnegative = v >= 0.0;
    let x = v.abs();
    let mut omx = 1.0 - x;
    if omx < 0.0 {
        omx = 0.0;
    }
    let root = omx.sqrt();
    #[expect(
        clippy::approx_constant,
        reason = "the polynomial's own constant term, copied bit for bit"
    )]
    let mut result =
        ((((((-0.001_262_491_1 * x + 0.006_670_09) * x - 0.017_088_126) * x + 0.030_891_88) * x
            - 0.050_174_303)
            * x
            + 0.088_978_99)
            * x
            - 0.214_598_8)
            * x
            + 1.570_796_3;
    result *= root;
    if nonnegative {
        result
    } else {
        std::f32::consts::PI - result
    }
}

/// `atan2(y, x)` in radians, within 1.5 ulp of the true angle: fdlibm's `atan2f` (Sun,
/// via FreeBSD), which reduces to one quotient and [`atan_nonneg`].
///
/// Signed zeros and the axes are exact — `atan2(±0, x > 0)` is `±0`, `atan2(±0, x < 0)`
/// is `±π`, and a zero `x` gives `±π/2` — so a caller asking for "no turn" gets exactly
/// zero. NaN in, NaN out.
pub(crate) fn atan2(y: f32, x: f32) -> f32 {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    /// `π − PI`: the part of π an f32 cannot hold, for the left half-plane.
    const PI_LO: f32 = f32::from_bits(0xb3bb_bd2e);
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    let left = x.is_sign_negative();
    let (ax, ay) = (x.abs(), y.abs());
    let a = if ay == 0.0 {
        if left { PI } else { 0.0 }
    } else if ax == 0.0 {
        FRAC_PI_2
    } else if ax == f32::INFINITY {
        match (ay == f32::INFINITY, left) {
            (true, false) => FRAC_PI_4,
            (true, true) => 3.0 * FRAC_PI_4,
            (false, false) => 0.0,
            (false, true) => PI,
        }
    } else if ay == f32::INFINITY {
        FRAC_PI_2
    } else {
        // The binary exponent of `|y/x|`, near enough: past 2²⁶ either way the
        // quotient is an axis to within half an ulp, and is not formed.
        let k = (ay.to_bits() as i32 - ax.to_bits() as i32) >> 23;
        if k > 26 {
            FRAC_PI_2 + 0.5 * PI_LO
        } else if k < -26 && left {
            PI
        } else {
            let z = atan_nonneg(ay / ax);
            if left { PI - (z - PI_LO) } else { z }
        }
    };
    if y.is_sign_negative() { -a } else { a }
}

/// `atan(x)` for `x ≥ 0`: fdlibm's `atanf`. The argument is reduced against one of
/// four breakpoints (`atan(½)`, `atan(1)`, `atan(3/2)`, `π/2`, each split into a high
/// and a low part), and what remains is at most 7/16 and taken by an odd polynomial.
fn atan_nonneg(x: f32) -> f32 {
    const ATAN_HI: [f32; 4] = [
        f32::from_bits(0x3eed_6338),
        f32::from_bits(0x3f49_0fda),
        f32::from_bits(0x3f7b_985e),
        f32::from_bits(0x3fc9_0fda),
    ];
    const ATAN_LO: [f32; 4] = [
        f32::from_bits(0x31ac_3769),
        f32::from_bits(0x3322_2168),
        f32::from_bits(0x3314_0fb4),
        f32::from_bits(0x33a2_2168),
    ];
    const AT: [f32; 5] = [
        f32::from_bits(0x3eaa_aaa9),
        f32::from_bits(0xbe4c_ca98),
        f32::from_bits(0x3e11_f50d),
        f32::from_bits(0xbdda_1247),
        f32::from_bits(0x3d7c_ac25),
    ];
    let bits = x.to_bits();
    let (id, x) = if bits < 0x3ee0_0000 {
        // Below 7/16, and below 2⁻¹² `atan(x)` rounds to `x` itself.
        if bits < 0x3980_0000 {
            return x;
        }
        (None, x)
    } else if bits < 0x3f30_0000 {
        (Some(0), (2.0 * x - 1.0) / (2.0 + x))
    } else if bits < 0x3f98_0000 {
        (Some(1), (x - 1.0) / (x + 1.0))
    } else if bits < 0x401c_0000 {
        (Some(2), (x - 1.5) / (1.0 + 1.5 * x))
    } else if bits < 0x4c80_0000 {
        (Some(3), -1.0 / x)
    } else {
        return ATAN_HI[3] + ATAN_LO[3];
    };
    let z = x * x;
    let w = z * z;
    let s1 = z * (AT[0] + w * (AT[2] + w * AT[4]));
    let s2 = w * (AT[1] + w * AT[3]);
    match id {
        None => x - x * (s1 + s2),
        Some(i) => ATAN_HI[i] - ((x * (s1 + s2) - ATAN_LO[i]) - x),
    }
}

/// `(sin 2πt, cos 2πt)` for an angle `t` in **turns**, within 2.5 ulp of each, exact at
/// every quarter turn.
///
/// The quarter turn nearest `t` is split off exactly — `4t` is exact, and so is its
/// distance to the nearest integer — which leaves at most an eighth of a turn for two
/// series; the quarter only swaps and negates them. Non-finite `t` gives NaN.
pub(crate) fn sin_cos_turns(t: f32) -> (f32, f32) {
    let u = 4.0 * t;
    let quarter = u.round();
    let a = (u - quarter) * 0.25 * std::f32::consts::TAU;
    let a2 = a * a;
    let s =
        a * (1.0 - a2 * (1.0 / 6.0 - a2 * (1.0 / 120.0 - a2 * (1.0 / 5040.0 - a2 / 362_880.0))));
    let c = 1.0
        - a2 * (0.5
            - a2 * (1.0 / 24.0 - a2 * (1.0 / 720.0 - a2 * (1.0 / 40_320.0 - a2 / 3_628_800.0))));
    match quarter.rem_euclid(4.0) as u8 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How many f32 ulps `got` lies from the f64 `want`, measured at `want`.
    fn ulps(got: f32, want: f64) -> f64 {
        let at = (want as f32).abs();
        let ulp = f64::from(f32::from_bits(at.to_bits() + 1) - at);
        (f64::from(got) - want).abs() / ulp
    }

    /// Over every direction — a fine sweep of angles, each at magnitudes from 10⁻³⁰ to
    /// 10³⁰ — and a spread of bit patterns, [`atan2`] is within 1.5 ulp of f64's
    /// (1.24 measured).
    #[test]
    fn atan2_is_within_an_ulp_and_a_half() {
        let mut worst = 0.0f64;
        let mut check = |y: f32, x: f32| {
            let want = f64::from(y).atan2(f64::from(x));
            let err = ulps(atan2(y, x), want);
            assert!(
                err <= 1.5,
                "atan2({y:e}, {x:e}) = {} against {want}: {err} ulp",
                atan2(y, x)
            );
            worst = worst.max(err);
        };
        for i in 0..20_000 {
            let theta = -std::f64::consts::PI + std::f64::consts::TAU * f64::from(i) / 20_000.0;
            for r in [1e-30f64, 1e-7, 1.0, 3.7, 1e7, 1e30] {
                check((r * theta.sin()) as f32, (r * theta.cos()) as f32);
            }
        }
        // A spread of raw patterns, positive and negative, normal and subnormal.
        let mut s = 0x9e37_79b9u32;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let (y, x) = (f32::from_bits(s), f32::from_bits(s.rotate_left(11)));
            if y.is_finite() && x.is_finite() {
                check(y, x);
            }
        }
        assert!(worst > 0.0, "the sweep measured nothing");
    }

    /// The signs, the zeros and the axes come out exact, as §12.1 needs of "no turn".
    #[test]
    fn atan2_is_exact_on_the_axes() {
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        let inf = f32::INFINITY;
        let cases = [
            (0.0, 1.0, 0.0),
            (-0.0, 1.0, -0.0),
            (0.0, 0.0, 0.0),
            (-0.0, 0.0, -0.0),
            (0.0, -0.0, PI),
            (-0.0, -0.0, -PI),
            (0.0, -3.0, PI),
            (-0.0, -3.0, -PI),
            (2.0, 0.0, FRAC_PI_2),
            (-2.0, -0.0, -FRAC_PI_2),
            (inf, inf, FRAC_PI_4),
            (-inf, inf, -FRAC_PI_4),
            (inf, -inf, 3.0 * FRAC_PI_4),
            (-inf, -inf, -3.0 * FRAC_PI_4),
            (5.0, inf, 0.0),
            (-5.0, inf, -0.0),
            (5.0, -inf, PI),
            (-5.0, -inf, -PI),
            (inf, 7.0, FRAC_PI_2),
            (-inf, -7.0, -FRAC_PI_2),
            (1.0, 1.0, FRAC_PI_4),
        ];
        for (y, x, want) in cases {
            assert_eq!(atan2(y, x).to_bits(), want.to_bits(), "atan2({y}, {x})");
        }
        assert!(atan2(f32::NAN, 1.0).is_nan() && atan2(1.0, f32::NAN).is_nan());
    }

    /// `(sin 2πt, cos 2πt)` within 2.5 ulp of f64's over several turns (2.28
    /// measured), exact wherever the answer is 0 or ±1.
    #[test]
    fn sin_cos_turns_is_close_and_exact_at_quarters() {
        // The reference reduced as the function under test is — exactly, in f64 — so
        // that a true zero is zero and not f64's `sin(π)`.
        let reference = |t: f32| {
            let u = 4.0 * f64::from(t);
            let q = u.round();
            let (s, c) = ((u - q) * 0.25 * std::f64::consts::TAU).sin_cos();
            match q.rem_euclid(4.0) as u8 {
                0 => (s, c),
                1 => (c, -s),
                2 => (-s, -c),
                _ => (-c, s),
            }
        };
        let check = |got: f32, want: f64, t: f32| {
            if want == 0.0 {
                assert_eq!(got, 0.0, "t = {t}: {got} where the answer is 0");
            } else {
                let err = ulps(got, want);
                assert!(err <= 2.5, "t = {t}: {got} against {want}: {err} ulp");
            }
        };
        for i in -400_000..=400_000 {
            let t = i as f32 / 163_840.0 + 1e-3;
            let (s, c) = sin_cos_turns(t);
            let (ws, wc) = reference(t);
            check(s, ws, t);
            check(c, wc, t);
        }
        for q in -8..=8 {
            let t = q as f32 * 0.25;
            let (s, c) = sin_cos_turns(t);
            let (ws, wc) = reference(t);
            assert_eq!((s, c), (ws as f32, wc as f32), "a quarter turn, t = {t}");
        }
    }
}
