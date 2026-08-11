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
//! Two further pieces of the same shader library live here for the same reason: the
//! light space `linear_to_light`/`light_to_linear` state (§18.0.4), and the
//! dispersion spectrum the chromatic filter integrates over (§21.10) — the host
//! needs the latter to *draw* a fringe it is about to ask the pass for.

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

// —— the dispersion spectrum (§21.10) ————————————————————————————————————————
//
// The chromatic filter's own colour science, on the host: which wavelength lands
// where along a fringe, and what the eye makes of it. The **pass** is the copy that
// runs (`filter_common.wesl`'s `ca_lambda` / `ca_weight`); this one exists so the
// frontend can *draw* the fringe it is about to ask for — the spectrum bar in the
// filter bar's dispersion pad (§21.6) is painted with these very colours, which is
// what makes it a statement about the render rather than a rainbow.
//
// The two ends and the Cauchy span come through the build-time mirror (§6.10) rather
// than being transcribed, so the range this samples and the range the pass integrates
// cannot drift; `dispersion_lambda_spans_the_visible` ties the three together.

/// The reddest wavelength the dispersion integral carries, in nm — parameter `s = 0`.
pub const LAMBDA_RED: f32 = stark_shaders::mirror::filter_common::CA_LAMBDA_RED;
/// The bluest, at `s = 1`.
pub const LAMBDA_BLUE: f32 = stark_shaders::mirror::filter_common::CA_LAMBDA_BLUE;

/// The wavelength (nm) whose refraction lands at dispersion parameter `s ∈ [0, 1]`
/// — 0 the red end of the fringe, 1 the blue.
///
/// Cauchy's law inverted, exactly as the pass inverts it: `(λ_red/λ)²` runs linearly
/// across the range, which is what makes the taps uniform in *displacement* and the
/// blue end spread farther than an equal run of the red (§21.10).
pub fn dispersion_lambda(s: f32) -> f32 {
    LAMBDA_RED / (1.0 + s * stark_shaders::mirror::filter_common::CA_CAUCHY_SPAN).sqrt()
}

/// One lobe of the CIE fit below: a piecewise Gaussian, its two flanks falling at
/// their own rates.
fn lobe(x: f32, mu: f32, s1: f32, s2: f32) -> f32 {
    let t = (x - mu) / if x < mu { s1 } else { s2 };
    (-0.5 * t * t).exp()
}

/// The eye's response to the wavelength at dispersion parameter `s`, as **linear
/// sRGB**: the CIE 1931 colour-matching functions (the Wyman–Sloan–Shirley analytic
/// fit), normalized to D65 and clamped at zero.
///
/// A *response*, not a colour — its absolute scale means nothing, only its shape
/// along `s`. In the pass that is why each channel of the gather divides by its own
/// summed weight; a caller drawing the spectrum normalizes for the same reason.
pub fn dispersion_weight(s: f32) -> [f32; 3] {
    let l = dispersion_lambda(s);
    let x = 1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7)
        - 0.065 * lobe(l, 501.1, 20.4, 26.2);
    let y = 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1);
    let z = 1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8);
    let lin = light_to_linear([x / 0.9505, y, z / 1.089]);
    [lin[0].max(0.0), lin[1].max(0.0), lin[2].max(0.0)]
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

    /// The three mirrored constants (§6.10) are one statement in three parts: the
    /// Cauchy span *is* `(λ_red/λ_blue)² − 1`, so inverting it at `s = 1` has to land
    /// on the blue end. Editing any one of them in the shader and not the others
    /// fails here rather than quietly narrowing the spectrum the frontend draws.
    #[test]
    fn dispersion_lambda_spans_the_visible() {
        assert!((dispersion_lambda(0.0) - LAMBDA_RED).abs() < 1e-3);
        assert!((dispersion_lambda(1.0) - LAMBDA_BLUE).abs() < 1e-2);
    }

    /// A fringe is a rainbow: red at one end, blue at the other, and nothing negative
    /// or dark in between — the properties the drawn spectrum bar relies on.
    #[test]
    fn dispersion_weight_is_a_rainbow() {
        let red = dispersion_weight(0.0);
        let blue = dispersion_weight(1.0);
        assert!(red[0] > red[2], "the red end is red: {red:?}");
        assert!(blue[2] > blue[0], "the blue end is blue: {blue:?}");
        for i in 0..=32 {
            let w = dispersion_weight(i as f32 / 32.0);
            assert!(w.iter().all(|c| *c >= 0.0), "clamped at zero: {w:?}");
            assert!(w.iter().any(|c| *c > 1e-3), "carries light: {w:?}");
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
