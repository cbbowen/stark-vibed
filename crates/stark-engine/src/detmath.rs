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
