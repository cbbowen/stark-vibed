//! Math the replay path may use (§12.1): functions built only from IEEE-754's exact or
//! correctly rounded operations (`+ − × ÷ √`, `round`, `%`, sign and bit
//! manipulation), so native and wasm builds compute them to the same bit.
//!
//! What decides where a stroke's segments are cut, and what they deposit, has to
//! agree to the last bit across replay, goldens and peers. The library
//! transcendentals (`sin`, `atan2`, …) are not specified to that — each platform's
//! libm may round differently. These cover the arc an edge is swept as, the
//! flattener's turn, the pen's orientation, a stretched tip's map, the tooth's bearing
//! and the stack a stamp's mask is turned through.
//!
//! Two kinds of function live here. The series — `sin_small` and its neighbours — are
//! written for the narrow range one caller reduced to, and are only as accurate as that
//! range needs. [`ln`], [`exp`], [`pow`] and [`cos_quadrant`] are ports of fdlibm's own
//! float routines, because the segment budget, the tip bakes, the stamp volumes, the
//! bleed solve, the tooth kernel and a placed image's alpha-to-height each read a library
//! transcendental over a domain no series covers cheaply. They round as fdlibm does,
//! which is within an ulp — not as the host libm
//! does, so porting a call site moves its pixels by an ulp once.
//!
//! f64 appears in [`cos_quadrant`]'s kernels, as it does in fdlibm's: a wider
//! correctly-rounded operation is as reproducible as a narrow one, and the trigonometric
//! kernels want the headroom.
//!
//! **Nothing on a host libm writes a tile any more.** The last one was the color
//! conversion, and it could not be closed from here — `stark-model` is upstream of this
//! crate (§2). It is closed in `stark_model::color` instead, by the **`libm` crate**:
//! fdlibm in Rust, bit-portable because it is the same source on every target, which is
//! what the vendored mixbox already ran the identical sRGB transfer through. So every
//! stroke's brush color, every fill parcel and every texel of a placed image now reaches
//! a tile the same way on every platform.
//!
//! **Still on libm but deciding a composited frame rather than a stored tile**: the focal
//! blur's axis (`gpu/composite/blur.rs` — its `atan2` could be [`atan2`] today), the
//! chromatic aberration's spread (`gpu/composite/filter.rs`) and the environment
//! convolution's direction (`gpu/environment.rs`).
//!
//! **What this list cannot reach is the GPU's own transcendentals**, and one of them does
//! decide a stored tile: `lib/color.wesl`'s `linear_srgb_to_oklab` runs a `pow` for its
//! cube root — reached from `blend.wesl` and `filter_common.wesl` — and `merge` writes
//! **stored** tiles through the blend pass that calls it (§15). WGSL
//! specifies no rounding for `pow`, so that is not a libm question and no host choice
//! closes it — which is the same fact `STARK_SKIP_GOLDEN` exists for, a golden being
//! blessed on one adapter. Named here so "nothing writes a tile" is not read wider than
//! the host it is about.
//!
//! Four things that look like remainders and are not. `powi` is not a libm call: a
//! runtime exponent lowers to `compiler_builtins`' `__powisf2`, which is binary
//! exponentiation by multiplication and the same Rust source on every target. The libm in
//! `assist` is **pre-record** — the assist decides what gets recorded, so it owes
//! agreement within one process and nothing across them — and so is [`tow`](crate::tow)'s,
//! which says so in its own module doc, and so are the guides': what
//! `guide::camera::dragged` and `guide::conic::circle_seen` compute is *recorded* rather
//! than derived from a record. And the CIE lobe in the engine's `filters` is on no render
//! path at all — it paints the spectrum bar the dispersion pad draws.

/// `sin(x)` as a truncated Maclaurin series, **for small `x` only**: within 1.03 ulp
/// (worst relative error 9.1e-8 against an f64 reference) for `|x| ≤ 0.428`, which is
/// `2·asin(0.2) + 0.025` — the whole turn [`arc_at`](crate::path::arc_at) takes across
/// an edge [`fit_arc`](crate::path::fit_arc) bent, and a bleed firing's walk back past
/// it.
pub(crate) fn sin_small(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 / 5040.0)))
}

/// `1 − cos(x)` as a series, over [`sin_small`]'s range: within 5 ulp (worst relative
/// error 4.2e-7) there, and within 2 ulp (1.5e-7) for the half turn
/// [`arc_sagitta`](crate::path::arc_sagitta) asks, `|x| ≤ asin(0.2)`. The first term
/// dropped, `x⁸/8!`, is what the wider range pays.
///
/// Evaluated directly rather than by subtraction, which is not a nicety: near zero
/// `cos(x)` rounds to exactly 1, and the difference cancels every digit it had.
pub(crate) fn versin_small(x: f32) -> f32 {
    let x2 = x * x;
    x2 * (0.5 - x2 * (1.0 / 24.0 - x2 / 720.0))
}

/// `asin(u)/u` as a series, for `|u| ≤ 0.2` (worst relative error 1.4e-7).
pub(crate) fn asin_over_x(u: f32) -> f32 {
    let u2 = u * u;
    1.0 + u2 * (1.0 / 6.0 + u2 * (3.0 / 40.0 + u2 * (15.0 / 336.0)))
}

/// `acos(v.clamp(-1, 1))` to within ~1e-7 absolute: a 7-degree minimax polynomial
/// times a square root (DirectXMath's `XMScalarAcos`).
///
/// **Bit for bit glam 0.33's `acos_approx_f32`**, which `Vec2::angle_to` is built on:
/// `flatten`'s turn test decides cuts through it, so any other arccosine moves them.
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
/// via FreeBSD), which reduces to one quotient and `atan_nonneg`.
///
/// Signed zeros and the axes are exact — `atan2(±0, x > 0)` is `±0`, `atan2(±0, x < 0)`
/// is `±π`, and a zero `x` gives `±π/2` — so a caller asking for "no turn" gets exactly
/// zero. NaN in, NaN out.
pub fn atan2(y: f32, x: f32) -> f32 {
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
pub fn sin_cos_turns(t: f32) -> (f32, f32) {
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

/// `ln(x)`, within 1 ulp: fdlibm's `logf` (Sun, via FreeBSD), which splits off the
/// exponent exactly, reduces the significand onto `[√2/2, √2)` and takes
/// `2·atanh(f/(2+f))` by an even polynomial.
///
/// Asserted over the whole f32 range, subnormals included (0.80 measured). `ln(1)` is
/// exactly `+0` and `ln(0)` exactly `−∞`; a negative argument gives the one canonical
/// NaN rather than a computed one, whose payload is the one thing about a NaN no platform
/// agrees on.
pub fn ln(mut x: f32) -> f32 {
    /// `ln 2` split so `k·ln2` carries more than f32 has room for.
    const LN2_HI: f32 = f32::from_bits(0x3f31_7180);
    const LN2_LO: f32 = f32::from_bits(0x3717_f7d1);
    /// `|(log(1+s) − log(1−s))/s − Lg(s)| < 2^-34.24` over the reduced range.
    const LG: [f32; 4] = [
        f32::from_bits(0x3f2a_aaaa),
        f32::from_bits(0x3ecc_ce13),
        f32::from_bits(0x3e91_e9ee),
        f32::from_bits(0x3e78_9e26),
    ];
    let mut ix = x.to_bits();
    let mut k = 0i32;
    if ix < 0x0080_0000 || (ix >> 31) != 0 {
        if ix << 1 == 0 {
            return f32::NEG_INFINITY;
        }
        if (ix >> 31) != 0 {
            return f32::NAN;
        }
        // Subnormal: scaled up by 2²⁵ and the scaling charged to the exponent.
        k -= 25;
        x *= f32::from_bits(0x4c00_0000);
        ix = x.to_bits();
    } else if ix >= 0x7f80_0000 {
        return x;
    } else if ix == 0x3f80_0000 {
        return 0.0;
    }
    // Onto [√2/2, √2), where `f` is at most 0.415 and `s` at most 0.1716.
    ix += 0x3f80_0000 - 0x3f35_04f3;
    k += (ix >> 23) as i32 - 0x7f;
    ix = (ix & 0x007f_ffff) + 0x3f35_04f3;
    x = f32::from_bits(ix);
    let f = x - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG[1] + w * LG[3]);
    let t2 = z * (LG[0] + w * LG[2]);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let dk = k as f32;
    s * (hfsq + r) + dk * LN2_LO - hfsq + f + dk * LN2_HI
}

/// `exp(x)`, within 1 ulp: fdlibm's `expf` (Sun, via FreeBSD) — `x = k·ln2 + r` with
/// `|r| ≤ ½ln2`, then a rational approximation of `exp(r)` scaled by `2^k`.
///
/// Asserted over the whole range that does not saturate, `[−103.98, 88.72]` (0.87
/// measured where the answer is normal, 0.74 where it is subnormal). `exp(0)` is exactly
/// 1; past either end it is `+∞` or `+0`.
pub fn exp(mut x: f32) -> f32 {
    /// `ln 2`, split so `k·ln2_hi` is exact for every `k` in range.
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e);
    const INV_LN2: f32 = f32::from_bits(0x3fb8_aa3b);
    /// `|x·(exp x + 1)/(exp x − 1) − p(x)| < 2^-27.74` over `[−0.34568, 0.34568]`.
    const P1: f32 = f32::from_bits(0x3e2a_aa8f);
    const P2: f32 = f32::from_bits(0xbb35_5215);
    let bits = x.to_bits();
    let negative = (bits >> 31) != 0;
    let hx = bits & 0x7fff_ffff;
    if hx >= 0x42ae_ac50 {
        // |x| past 87.34, where one of the two ends may already have been reached.
        if hx > 0x7f80_0000 {
            return x;
        }
        if !negative && hx >= 0x42b1_7218 {
            return f32::INFINITY;
        }
        if negative && hx >= 0x42cf_f1b5 {
            return 0.0;
        }
    }
    let k: i32;
    let (hi, lo);
    if hx > 0x3eb1_7218 {
        // |x| over ½ln2, so there is a multiple of ln2 to take out.
        k = if hx > 0x3f85_1592 {
            // |x| over 1½ln2: round `x/ln2` by adding a half and truncating.
            (INV_LN2 * x + if negative { -0.5 } else { 0.5 }) as i32
        } else if negative {
            -1
        } else {
            1
        };
        let kf = k as f32;
        hi = x - kf * LN2_HI;
        lo = kf * LN2_LO;
        x = hi - lo;
    } else if hx > 0x3900_0000 {
        // |x| over 2⁻¹⁴, but under the reduction's threshold.
        k = 0;
        hi = x;
        lo = 0.0;
    } else {
        return 1.0 + x;
    }
    let xx = x * x;
    let c = x - xx * (P1 + xx * P2);
    let y = 1.0 + (x * c / (2.0 - c) - lo + hi);
    if k == 0 { y } else { scalbn(y, k) }
}

/// `x^y` for a **non-negative base**, within 1 ulp: fdlibm's `powf` (Sun, via FreeBSD),
/// which forms `log2(x)` and `2^(y·log2 x)` each in two f32 words, so the product keeps
/// the bits a single-word `exp(y·ln x)` would lose.
///
/// Asserted over the base and exponent ranges the tip bakes ask for — bases in
/// `(0, 100]`, exponents in `[−1, 34]` — plus subnormal bases and exponents out to 2e9
/// (0.82 measured). Exact where the answer is algebraic: `y = 0`, `x = 1`, `y = ±1`,
/// `y = 2`, `y = ½`, and `x = 0` or `+∞` against either sign of `y`. `−0` counts as `+0`.
///
/// **A negative base is not covered.** fdlibm's rules for one turn on the parity of an
/// integer `y`, nothing here asks for them, and a port that quietly answered `|x|^y`
/// instead would be worse than no port: this answers NaN, and a debug build asserts.
/// A NaN base propagates instead, as it does through the rest of this module.
///
/// One shortcut of fdlibm's is deliberately left out: past `|y| = 2²⁷` it declares an
/// over/underflow for any base further than 8 ulps from 1, and 8 ulps is not far
/// enough — `2²⁷ · 8·2⁻²⁴` is 64, well inside f32's range, so `powf(1 − 9ulp, −1.5e8)`
/// comes back `+∞` where the answer is 8.8e34. The two-word product below is exact in
/// `y` at any magnitude, so the general path serves that corner and the test measures
/// it there.
pub fn pow(x: f32, y: f32) -> f32 {
    debug_assert!(
        x.is_nan() || x >= 0.0,
        "detmath::pow covers a non-negative base only, not {x}"
    );
    /// `exp(r) − 1 − r` over the reduced range, as `2^z`'s rational core.
    const P: [f32; 5] = [
        f32::from_bits(0x3e2a_aaab),
        f32::from_bits(0xbb36_0b61),
        f32::from_bits(0x388a_b355),
        f32::from_bits(0xb5dd_ea0e),
        f32::from_bits(0x3331_bb4c),
    ];
    /// `ln 2`, whole and split into a head that multiplies exactly.
    const LG2: f32 = f32::from_bits(0x3f31_7218);
    const LG2_H: f32 = f32::from_bits(0x3f31_7200);
    const LG2_L: f32 = f32::from_bits(0x35bf_be8c);
    /// `−(128 − log2(overflow + ½ulp))`: how far past 128 the exponent may land and
    /// still round back inside.
    const OVT: f32 = f32::from_bits(0x3338_aa3c);

    let hy = y.to_bits() as i32;
    let iy = hy & 0x7fff_ffff;
    // `x⁰ = 1` and `1^y = 1`, whatever the other argument is — a NaN included.
    if iy == 0 || x.to_bits() == 0x3f80_0000 {
        return 1.0;
    }
    // Out of domain, or a NaN either side.
    if x < 0.0 || x.is_nan() || iy > 0x7f80_0000 {
        return f32::NAN;
    }
    let ix = (x.to_bits() as i32) & 0x7fff_ffff;
    if iy == 0x7f80_0000 {
        // A ±∞ exponent: which of 0 and ∞ by which side of 1 the base is on.
        return if (ix > 0x3f80_0000) == (hy > 0) {
            f32::INFINITY
        } else {
            0.0
        };
    }
    if iy == 0x3f80_0000 {
        return if hy > 0 { x } else { 1.0 / x };
    }
    if hy == 0x4000_0000 {
        return x * x;
    }
    if hy == 0x3f00_0000 {
        return x.sqrt();
    }
    if ix == 0 || ix == 0x7f80_0000 {
        // `0^y` and `∞^y`: the magnitude itself, inverted for a negative exponent.
        let z = f32::from_bits(ix as u32);
        return if hy < 0 { 1.0 / z } else { z };
    }

    let (t1, t2) = log2_two_words(x);

    // `y·log2(x)`, with `y` split so the head product is exact.
    let y1 = f32::from_bits(y.to_bits() & HEAD);
    let p_l = (y - y1) * t1 + y * t2;
    let mut p_h = y1 * t1;
    let z = p_l + p_h;
    let j = z.to_bits() as i32;
    if j > 0x4300_0000 || (j == 0x4300_0000 && p_l + OVT > z - p_h) {
        return f32::INFINITY;
    }
    if (j as u32) > 0xc316_0000 || ((j as u32) == 0xc316_0000 && p_l <= z - p_h) {
        return 0.0;
    }

    // `2^(p_h + p_l)`: the integer part split off exactly, the rest by the polynomial.
    let i = j & 0x7fff_ffff;
    let mut k = (i >> 23) - 0x7f;
    let mut n = 0i32;
    if i > 0x3f00_0000 {
        // |z| over ½, so there is an integer to take out: `n = round(z)`.
        n = j + (0x0080_0000 >> (k + 1));
        k = ((n & 0x7fff_ffff) >> 23) - 0x7f;
        let t = f32::from_bits(n as u32 & !(0x007f_ffff >> k));
        n = ((n & 0x007f_ffff) | 0x0080_0000) >> (23 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = f32::from_bits((p_l + p_h).to_bits() & 0xffff_8000);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    let mut z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let c = z - t * (P[0] + t * (P[1] + t * (P[2] + t * (P[3] + t * P[4]))));
    let r = (z * c) / (c - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    let j = (z.to_bits() as i32) + (n << 23);
    if (j >> 23) <= 0 {
        // A subnormal answer, where the exponent cannot simply be pasted on.
        scalbn(z, n)
    } else {
        f32::from_bits(j as u32)
    }
}

/// The bottom 12 bits of an f32, dropped to leave a head two of which multiply exactly:
/// 12 + 12 fits the 24 a significand has.
const HEAD: u32 = 0xffff_f000;

/// `log2(x)` for a positive finite `x`, as a head and a tail whose sum carries more bits
/// than one f32 holds: the half of fdlibm's `powf` that is a logarithm.
///
/// The exponent comes off exactly and the significand is expanded about whichever of 1
/// and 3/2 it is nearer, so `s = (m − bp)/(m + bp)` is at most 1/5 and an odd polynomial
/// in `s²` covers it. Every head below is a value with its low 12 bits cleared, which is
/// what makes the products that reconstruct the tail exact.
fn log2_two_words(x: f32) -> (f32, f32) {
    /// The two points the expansion is taken about, and `log2` of each in two words.
    const BP: [f32; 2] = [1.0, 1.5];
    const DP_H: [f32; 2] = [0.0, f32::from_bits(0x3f15_c000)];
    const DP_L: [f32; 2] = [0.0, f32::from_bits(0x35d1_cfdc)];
    /// `(3/2)·(log(x) − 2s − (2/3)s³)` in `s²`.
    const L: [f32; 6] = [
        f32::from_bits(0x3f19_999a),
        f32::from_bits(0x3edb_6db7),
        f32::from_bits(0x3eaa_aaab),
        f32::from_bits(0x3e8b_a305),
        f32::from_bits(0x3e6c_3255),
        f32::from_bits(0x3e53_f142),
    ];
    /// `2/(3 ln 2)`, whole and split into a head plus the tail of that head.
    const CP: f32 = f32::from_bits(0x3f76_384f);
    const CP_H: f32 = f32::from_bits(0x3f76_4000);
    const CP_L: f32 = f32::from_bits(0xb8f6_23c6);

    let mut ax = x;
    let mut ix = x.to_bits() as i32;
    let mut n = 0i32;
    if ix < 0x0080_0000 {
        // Subnormal, scaled up and the scaling charged to the exponent.
        ax *= f32::from_bits(0x4b80_0000);
        n -= 24;
        ix = ax.to_bits() as i32;
    }
    n += (ix >> 23) - 0x7f;
    let j = ix & 0x007f_ffff;
    ix = j | 0x3f80_0000;
    // Which of the two expansion points the significand is nearer; past √3 it is the
    // next octave's turn.
    let k: usize = if j <= 0x1c_c471 {
        0
    } else if j < 0x5d_b3d7 {
        1
    } else {
        n += 1;
        ix -= 0x0080_0000;
        0
    };
    ax = f32::from_bits(ix as u32);
    // `s = (ax − bp)/(ax + bp)`, with a head and the tail the head left behind.
    let u = ax - BP[k];
    let v = 1.0 / (ax + BP[k]);
    let s = u * v;
    let s_h = f32::from_bits(s.to_bits() & HEAD);
    let t_h = f32::from_bits(
        ((((ix as u32) >> 1) & HEAD) | 0x2000_0000) + 0x0040_0000 + ((k as u32) << 21),
    );
    let t_l = ax - (t_h - BP[k]);
    let s_l = v * ((u - s_h * t_h) - s_h * t_l);
    // `log(ax)`, then `3 + s_h² + r` in two words so the product below stays exact.
    let mut s2 = s * s;
    let mut r =
        s2 * s2 * (L[0] + s2 * (L[1] + s2 * (L[2] + s2 * (L[3] + s2 * (L[4] + s2 * L[5])))));
    r += s_l * (s_h + s);
    s2 = s_h * s_h;
    let u_h = f32::from_bits((3.0 + s2 + r).to_bits() & HEAD);
    let u_l = r - ((u_h - 3.0) - s2);
    let u = s_h * u_h;
    let v = s_l * u_h + u_l * s;
    // `2/(3 ln 2)·(s + …)`, so `log2(ax) = n + dp_h + z_h + z_l`.
    let p_h = f32::from_bits((u + v).to_bits() & HEAD);
    let p_l = v - (p_h - u);
    let z_h = CP_H * p_h;
    let z_l = CP_L * p_h + p_l * CP + DP_L[k];
    let t = n as f32;
    let head = f32::from_bits((((z_h + z_l) + DP_H[k]) + t).to_bits() & HEAD);
    (head, z_l - (((head - t) - DP_H[k]) - z_h))
}

/// `x · 2^n`, exact wherever the answer is normal: fdlibm's `scalbnf`. The prescaling
/// is what keeps a subnormal answer rounded once instead of twice.
fn scalbn(mut x: f32, mut n: i32) -> f32 {
    /// `2^127` and `2^-102`, the two prescale steps.
    const UP: f32 = f32::from_bits(0x7f00_0000);
    const DOWN: f32 = f32::from_bits(0x0c80_0000);
    if n > 127 {
        x *= UP;
        n -= 127;
        if n > 127 {
            x *= UP;
            n = (n - 127).min(127);
        }
    } else if n < -126 {
        x *= DOWN;
        n += 102;
        if n < -126 {
            x *= DOWN;
            n = (n + 102).max(-126);
        }
    }
    x * f32::from_bits(((0x7f + n) as u32) << 23)
}

/// `cos(x)` for `|x| ≤ π/2`, within 1 ulp: fdlibm's `cosf` (Sun, via FreeBSD, tuned by
/// Bruce Evans), restricted to the one quadrant — the kernel directly under π/4, and
/// `sin(π/2 − |x|)` above it.
///
/// Asserted over that quadrant and nothing beyond it (0.50 measured). Anything further
/// needs fdlibm's general argument reduction, no caller asks, and a debug build says so;
/// [`sin_cos_turns`] is what an angle over a whole turn goes through.
pub fn cos_quadrant(x: f32) -> f32 {
    debug_assert!(
        x.abs() <= std::f32::consts::FRAC_PI_2,
        "detmath::cos_quadrant covers |x| ≤ π/2 only, not {x}"
    );
    let ax = x.abs();
    let ix = ax.to_bits();
    if ix < 0x3980_0000 {
        // |x| under 2⁻¹², where cos rounds to 1.
        return 1.0;
    }
    if ix <= 0x3f49_0fda {
        return cos_kernel(f64::from(ax));
    }
    sin_kernel(std::f64::consts::FRAC_PI_2 - f64::from(ax))
}

/// `cos(x)` for `|x| ≤ π/4`: fdlibm's `__kernel_cosdf`, whose polynomial is within
/// 2^-34.1 of the cosine over that range.
fn cos_kernel(x: f64) -> f32 {
    const C: [f64; 4] = [
        f64::from_bits(0xbfdf_ffff_fd0c_5e81),
        f64::from_bits(0x3fa5_5553_e105_3a42),
        f64::from_bits(0xbf56_c087_e80f_1e27),
        f64::from_bits(0x3ef9_9342_e0ee_5069),
    ];
    let z = x * x;
    let w = z * z;
    let r = C[2] + z * C[3];
    (((1.0 + z * C[0]) + w * C[1]) + (w * z) * r) as f32
}

/// `sin(x)` for `|x| ≤ π/4`: fdlibm's `__kernel_sindf`, within 2^-37.5 of `sin(x)/x`
/// over that range.
fn sin_kernel(x: f64) -> f32 {
    const S: [f64; 4] = [
        f64::from_bits(0xbfc5_5555_54cb_ac77),
        f64::from_bits(0x3f81_1110_896e_fbb2),
        f64::from_bits(0xbf2a_00f9_e2ca_e774),
        f64::from_bits(0x3ec6_cd87_8c3b_46a7),
    ];
    let z = x * x;
    let w = z * z;
    let r = S[2] + z * S[3];
    let s = z * x;
    ((x + s * (S[0] + z * S[1])) + s * w * r) as f32
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
    /// Every `stride`th f32 whose bits lie in `bits`, as a sweep that reaches the
    /// subnormals and the top of the range alike without enumerating 2³¹ values.
    fn sweep(bits: std::ops::RangeInclusive<u32>, stride: u32) -> impl Iterator<Item = f32> {
        (*bits.start()..=*bits.end())
            .step_by(stride as usize)
            .map(f32::from_bits)
    }

    /// [`ln`] is within 1 ulp of f64's (0.80 measured) over the whole f32 range,
    /// subnormals included, and sampled finer over the decades `ln_keep` and the bleed
    /// solve read — `[0.05, 1]`.
    #[test]
    fn ln_is_within_an_ulp() {
        let mut worst = 0.0f64;
        let mut check = |x: f32| {
            let want = f64::from(x).ln();
            let err = ulps(ln(x), want);
            assert!(
                err <= 1.0,
                "ln({x:e}) = {} against {want}: {err} ulp",
                ln(x)
            );
            worst = worst.max(err);
        };
        for x in sweep(1..=0x7f7f_ffff, 1009) {
            check(x);
        }
        for x in sweep(0.05f32.to_bits()..=1.0f32.to_bits(), 97) {
            check(x);
        }
        assert!(worst > 0.0, "the sweep measured nothing");
    }

    /// The ends are exact, so a caller reading `ln(1 − axis)` at `axis = 0` gets exactly
    /// "no transfer" rather than an ulp of one.
    #[test]
    fn ln_is_exact_at_the_ends() {
        assert_eq!(ln(1.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert_eq!(ln(-0.0), f32::NEG_INFINITY);
        assert_eq!(ln(f32::INFINITY), f32::INFINITY);
        assert!(ln(-1.0).is_nan(), "a negative argument is not a number");
        assert!(ln(f32::NAN).is_nan(), "NaN in, NaN out");
    }

    /// [`exp`] is within 1 ulp of f64's — 0.87 measured where the answer is normal, 0.74
    /// where it is subnormal, which is the corner the `scalbn` prescale exists for. The
    /// sweep is over bit patterns, so it spends most of its samples where the tip bakes
    /// and the tooth kernel do: `|x|` between 2⁻¹⁴ and a few tens.
    #[test]
    fn exp_is_within_an_ulp() {
        let (mut worst, mut worst_sub) = (0.0f64, 0.0f64);
        let mut check = |x: f32| {
            let want = f64::from(x).exp();
            let err = ulps(exp(x), want);
            assert!(
                err <= 1.0,
                "exp({x:e}) = {} against {want}: {err} ulp",
                exp(x)
            );
            if want >= f64::from(f32::MIN_POSITIVE) {
                worst = worst.max(err);
            } else {
                worst_sub = worst_sub.max(err);
            }
        };
        // Down to −104 and no further: over 88.73 the answer is +∞ and under −103.98 it
        // is +0, and the f64 reference saturates to neither, so the sweep stops where the
        // ulp measure stops meaning anything.
        for x in sweep(1..=88.0f32.to_bits(), 1009) {
            check(x);
            check(-x);
        }
        for x in sweep(88.0f32.to_bits()..=104.0f32.to_bits(), 97) {
            check(-x);
        }
        assert!(worst > 0.0 && worst_sub > 0.0, "the sweep measured nothing");
    }

    /// Zero and the two saturation ends are exact.
    #[test]
    fn exp_is_exact_at_zero_and_at_the_ends() {
        assert_eq!(exp(0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(exp(-0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp(89.0), f32::INFINITY);
        assert_eq!(exp(-200.0), 0.0);
        assert!(exp(f32::NAN).is_nan(), "NaN in, NaN out");
    }

    /// [`pow`] is within 1 ulp of f64's (0.82 measured) over the bases and exponents the
    /// tip bakes ask for, and stays there at the far ends of the exponent range, where
    /// fdlibm takes its `|y| > 2²⁷` branch instead.
    #[test]
    fn pow_is_within_an_ulp() {
        let mut worst = 0.0f64;
        let mut check = |x: f32, y: f32| {
            let want = f64::from(x).powf(f64::from(y));
            // Only where the answer is a normal f32: outside that it is the saturation
            // the exactness test states, not an approximation to be measured.
            if !(f64::from(f32::MIN_POSITIVE)..=f64::from(f32::MAX)).contains(&want) {
                return;
            }
            let err = ulps(pow(x, y), want);
            assert!(
                err <= 1.0,
                "pow({x:e}, {y:e}) = {} against {want}: {err} ulp",
                pow(x, y)
            );
            worst = worst.max(err);
        };
        // The bakes' own domain: bases from a texel of the rim clamp up past the
        // shoulder scan's own base, exponents from `p − ½` at the soft end to `1/p` at
        // the hard one.
        for x in sweep(1e-8f32.to_bits()..=100.0f32.to_bits(), 199_999) {
            for i in 0..=1400 {
                check(x, -1.0 + 35.0 * i as f32 / 1400.0);
            }
        }
        // And the corners no bake reaches: a subnormal base, and the exponents past 2²⁷
        // whose shortcut this does not carry, against a base an ulp or two from 1.
        for x in sweep(1..=0x0080_0000, 4099) {
            for y in [-3.0f32, -0.25, 0.3, 1.7, 5.0] {
                check(x, y);
            }
        }
        for i in -600i32..=600 {
            let x = f32::from_bits((1.0f32.to_bits() as i32 + i) as u32);
            for y in [-2e9f32, -2e8, -1.5e8, 1.5e8, 2e8, 2e9] {
                check(x, y);
            }
        }
        assert!(worst > 0.0, "the sweep measured nothing");
    }

    /// Where the answer is algebraic [`pow`] hands it back exactly, and where it
    /// saturates it saturates — both of which a series would only approach.
    #[test]
    fn pow_is_exact_where_the_answer_is_algebraic() {
        for x in [0.0f32, 0.5, 1.0, 3.7, f32::INFINITY, f32::NAN] {
            assert_eq!(pow(x, 0.0).to_bits(), 1.0f32.to_bits(), "{x}⁰");
        }
        for y in [
            0.0f32,
            -2.5,
            7.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert_eq!(pow(1.0, y).to_bits(), 1.0f32.to_bits(), "1^{y}");
        }
        for x in [f32::MIN_POSITIVE, 0.25, 1.75, 9.0, 1e30] {
            assert_eq!(pow(x, 1.0).to_bits(), x.to_bits(), "{x}¹");
            assert_eq!(pow(x, -1.0).to_bits(), (1.0 / x).to_bits(), "{x}⁻¹");
            assert_eq!(pow(x, 2.0).to_bits(), (x * x).to_bits(), "{x}²");
            assert_eq!(pow(x, 0.5).to_bits(), x.sqrt().to_bits(), "√{x}");
        }
        assert_eq!(pow(0.0, 2.5), 0.0);
        assert_eq!(pow(-0.0, 2.5), 0.0, "−0 counts as +0");
        assert_eq!(pow(0.0, -1.5), f32::INFINITY);
        assert_eq!(pow(f32::INFINITY, 2.5), f32::INFINITY);
        assert_eq!(pow(f32::INFINITY, -2.5), 0.0);
        assert_eq!(pow(2.0, f32::INFINITY), f32::INFINITY);
        assert_eq!(pow(0.5, f32::INFINITY), 0.0);
        assert_eq!(pow(2.0, f32::NEG_INFINITY), 0.0);
        assert_eq!(pow(0.5, f32::NEG_INFINITY), f32::INFINITY);
        assert_eq!(pow(2.0, 200.0), f32::INFINITY, "past the top");
        assert_eq!(pow(2.0, -200.0), 0.0, "past the bottom");
        assert!(pow(f32::NAN, 2.5).is_nan(), "NaN in, NaN out");
        assert!(pow(2.5, f32::NAN).is_nan(), "NaN in, NaN out");
    }

    /// A negative base is outside what is ported, and a debug build says so rather than
    /// answering `|x|^y`.
    #[test]
    #[should_panic(expected = "non-negative base")]
    fn pow_refuses_a_negative_base() {
        let _ = pow(-2.0, 3.0);
    }

    /// [`cos_quadrant`] is within 1 ulp of f64's (0.50 measured) over the quadrant it
    /// covers, either sign.
    #[test]
    fn cos_quadrant_is_within_an_ulp() {
        let mut worst = 0.0f64;
        let mut check = |x: f32| {
            let want = f64::from(x).cos();
            let err = ulps(cos_quadrant(x), want);
            assert!(
                err <= 1.0,
                "cos({x:e}) = {} against {want}: {err} ulp",
                cos_quadrant(x)
            );
            worst = worst.max(err);
        };
        for x in sweep(0..=std::f32::consts::FRAC_PI_2.to_bits(), 521) {
            check(x);
            check(-x);
        }
        assert_eq!(cos_quadrant(0.0).to_bits(), 1.0f32.to_bits(), "cos 0");
        assert!(worst > 0.0, "the sweep measured nothing");
    }

    /// Past a quarter turn the reduction this does not carry would be needed, and a
    /// debug build says so.
    #[test]
    #[should_panic(expected = "|x| ≤ π/2")]
    fn cos_quadrant_refuses_a_wider_angle() {
        let _ = cos_quadrant(2.0);
    }
}
