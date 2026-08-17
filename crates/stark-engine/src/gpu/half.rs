//! IEEE-754 half-precision, both ways (§6.1, §9).
//!
//! Two functions that had drifted apart into different files — the encoder in the
//! environment prefilter, the decoder in readback — doing inverse halves of one
//! conversion. They are here together so the asymmetry between them is visible
//! rather than accidental: the encoder is deliberately *not* general, because what
//! it encodes is radiance.

/// Encode a non-negative `f32` to IEEE-754 half-precision bits (round-to-nearest-
/// even). Environment radiance is ≥ 0, so the sign bit is always clear; values are
/// clamped to the half-float max (no infinities), and subnormals flush to zero.
pub(crate) fn f32_to_f16(x: f32) -> u16 {
    let x = x.clamp(0.0, 65504.0);
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return 0; // zero / subnormal → 0 (negligible for radiance)
    }
    let mant = bits & 0x7f_ffff;
    let half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let round = u16::from(rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1));
    ((exp as u16) << 10 | half_mant) + round
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
        assert_eq!(f32_to_f16(-1.0), 0); // negatives clamp to 0
    }
}
