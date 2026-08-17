//! Oklab working color space (§6.5).
//!
//! Color enters as straight sRGB (the picker / image space), is stored and
//! blended in **Oklab** for perceptually uniform mixing, and is converted back
//! to display only in the media pass. These conversions are fixed constants
//! shared with the WGSL side (`shaders/lib/color.wesl`), so ingest and present are
//! reproducible across runs and peers — required by golden tests (§9) and
//! convergence (§12).
//!
//! Oklab transform after Björn Ottosson.
//!
//! One further piece of the same shader library lives here for the same reason: the
//! light space `linear_to_light`/`light_to_linear` state (§18.0.4). The dispersion
//! spectrum that used to sit beside it is in `stark-engine`'s `dispersion`, because
//! it reads the shader mirror (§6.10) and this crate compiles without the shaders.

/// sRGB transfer function: gamma-encoded component in `[0,1]` → linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB transfer function: linear component → gamma-encoded.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Linear sRGB `[r,g,b]` → Oklab `[L,a,b]`.
pub fn linear_srgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    [
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    ]
}

/// Oklab `[L,a,b]` → linear sRGB `[r,g,b]`.
pub fn oklab_to_linear_srgb(lab: [f32; 3]) -> [f32; 3] {
    let [ll, aa, bb] = lab;
    let l_ = ll + 0.396_337_78 * aa + 0.215_803_76 * bb;
    let m_ = ll - 0.105_561_346 * aa - 0.063_854_17 * bb;
    let s_ = ll - 0.089_484_18 * aa - 1.291_485_5 * bb;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// Straight sRGB RGBA in `[0,1]` → Oklab `[L,a,b]` + unchanged alpha.
pub fn srgb_to_oklab(rgba: [f32; 4]) -> [f32; 4] {
    let lin = [
        srgb_to_linear(rgba[0]),
        srgb_to_linear(rgba[1]),
        srgb_to_linear(rgba[2]),
    ];
    let lab = linear_srgb_to_oklab(lin);
    [lab[0], lab[1], lab[2], rgba[3]]
}

/// Oklab `[L,a,b]` + alpha → straight sRGB RGBA in `[0,1]`.
pub fn oklab_to_srgb(laba: [f32; 4]) -> [f32; 4] {
    let lin = oklab_to_linear_srgb([laba[0], laba[1], laba[2]]);
    [
        linear_to_srgb(lin[0]),
        linear_to_srgb(lin[1]),
        linear_to_srgb(lin[2]),
        laba[3],
    ]
}

/// Linear sRGB → XYZ normalized to D65 white — the space light is combined and
/// scaled in (see `lib/color.wesl`, which holds the argument). Each row of the sRGB
/// primaries divided by its own sum, so `(1,1,1)` maps to `(1,1,1)`.
pub fn linear_to_light(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = c;
    [
        0.433_939_5 * r + 0.376_207_8 * g + 0.189_852_8 * b,
        0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b,
        0.017_756_6 * r + 0.109_467_1 * g + 0.872_776_3 * b,
    ]
}

/// The exact inverse of [`linear_to_light`].
pub fn light_to_linear(c: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = c;
    [
        3.079_954_5 * x - 1.537_138_5 * y - 0.542_816 * z,
        -0.921_258_3 * x + 1.876_010_8 * y + 0.045_247_7 * z,
        0.052_887_3 * x - 0.204_025_9 * y + 1.151_138_5 * z,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 4], b: [f32; 4], eps: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= eps)
    }

    #[test]
    fn light_roundtrip() {
        for c in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.5, 0.8]] {
            let back = light_to_linear(linear_to_light(c));
            assert!(
                c.iter().zip(back).all(|(x, y)| (x - y).abs() <= 1e-4),
                "roundtrip {c:?} -> {back:?}"
            );
        }
    }

    #[test]
    fn srgb_oklab_roundtrip() {
        for c in [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.2, 0.5, 0.8, 0.5],
        ] {
            let back = oklab_to_srgb(srgb_to_oklab(c));
            assert!(close(c, back, 1e-3), "roundtrip {c:?} -> {back:?}");
        }
    }

    #[test]
    fn gray_has_no_chroma() {
        let lab = srgb_to_oklab([0.5, 0.5, 0.5, 1.0]);
        assert!(
            lab[1].abs() < 1e-3 && lab[2].abs() < 1e-3,
            "gray a,b ~ 0: {lab:?}"
        );
    }
}
