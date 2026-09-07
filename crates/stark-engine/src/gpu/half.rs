//! IEEE-754 half-precision, both ways (§6.1, §9).
//!
//! The encoder is **signed**, because one of its callers is not radiance: a placed
//! image's tile channels carry an Oklab latent whose `a` and `b` axes run either side of
//! zero (§23), and clamping those to zero desaturates every imported photograph towards
//! green — silently, in the one code path with no shader to read.

/// Encode an `f32` to IEEE-754 half-precision bits (round-to-nearest-even).
///
/// Magnitudes are clamped to the half-float max, so no input produces an infinity, and
/// subnormals flush to zero — negligible at the scales either caller works at
/// (radiance, and a color channel).
pub fn f32_to_f16(x: f32) -> u16 {
    let sign = u16::from(x.is_sign_negative()) << 15;
    // The NaN case is tested rather than clamped: `clamp` *returns* the NaN, whose
    // exponent field is `0xff` and would come out of the arithmetic below as a large
    // finite half. Zero is the one answer a channel or a radiance sample can survive.
    let magnitude = if x.is_nan() {
        0.0
    } else {
        x.abs().min(65504.0)
    };
    let bits = magnitude.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign; // zero / subnormal → ±0
    }
    let mant = bits & 0x7f_ffff;
    let half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let round = u16::from(rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1));
    sign | (((exp as u16) << 10 | half_mant) + round)
}

/// Decode an IEEE-754 half-precision float to `f32`.
///
/// General where [`f32_to_f16`] is not: this reads whatever a GPU wrote, including
/// the subnormals and infinities the encoder never produces.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = match exp {
        0 => (mant as f32) * 2f32.powi(-24), // subnormal (and zero)
        0x1f => {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 { -val } else { val }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_encoding_roundtrips() {
        // Decode our f16 bits back to f32 and check a few representative radiance
        // values land within half-float precision.
        let half_to_f32 = |h: u16| -> f32 {
            let exp = ((h >> 10) & 0x1f) as i32;
            let mant = (h & 0x3ff) as f32;
            if exp == 0 {
                return 0.0;
            }
            (1.0 + mant / 1024.0) * 2.0f32.powi(exp - 15)
        };
        for &v in &[0.0f32, 0.25, 1.0, 2.5, 18.0, 500.0] {
            let back = half_to_f32(f32_to_f16(v));
            assert!(
                (back - v).abs() <= v.max(1.0) * 0.001 + 1e-3,
                "f16({v}) -> {back}"
            );
        }
    }

    /// **The encoder is signed**, and the two directions here are inverses over the
    /// whole range rather than over the non-negative half — the property a placed
    /// image's tile channels depend on (§23), an Oklab latent's `a` and `b` axes running
    /// either side of zero. Folding a negative onto `+0` fails in the way that is
    /// hardest to see — no error, no NaN, just every photograph pulled towards green —
    /// so it is pinned against the *decoder in this file* rather than a table of bits.
    #[test]
    fn the_pair_round_trips_through_zero() {
        for &v in &[
            0.0f32,
            -0.0,
            0.25,
            -0.25,
            1.0,
            -1.0,
            0.0009765625,
            -0.5171,
            2.5,
            -18.0,
            500.0,
            -500.0,
        ] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!(
                (back - v).abs() <= v.abs().max(1.0) * 0.001,
                "f16({v}) -> {back}",
            );
            assert_eq!(
                back.is_sign_negative(),
                v.is_sign_negative(),
                "the sign of {v} did not survive",
            );
        }
        // Out of range in both directions, and a NaN, all land on something finite
        // rather than on an infinity a target would have to interpret.
        for &v in &[f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30, f32::NAN] {
            assert!(f16_to_f32(f32_to_f16(v)).is_finite(), "{v}");
        }
    }
}
