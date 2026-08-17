//! The dispersion spectrum (§21.10).
//!
//! The chromatic filter's own color science, on the host: which wavelength lands
//! where along a fringe, and what the eye makes of it. The **pass** is the copy that
//! runs (`filter_common.wesl`'s `ca_lambda` / `ca_weight`); this one exists so the
//! frontend can *draw* the fringe it is about to ask for — the spectrum bar in the
//! filter bar's dispersion pad (§21.6) is painted with these very colors, which is
//! what makes it a statement about the render rather than a rainbow.
//!
//! The two ends and the Cauchy span come through the build-time mirror (§6.10) rather
//! than being transcribed, so the range this samples and the range the pass integrates
//! cannot drift; `dispersion_lambda_spans_the_visible` ties the three together.
//!
//! In `stark-engine` rather than beside the colorimetry it uses, because it reads
//! the build-time shader mirror (§6.10) — and `stark-model` compiles without the
//! shaders at all. The split is the mirror rule stating itself: what the `.wesl`
//! file says belongs on the side that has the `.wesl` file.

use stark_model::color::light_to_linear;

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
/// sRGB**: the CIE 1931 color-matching functions (the Wyman–Sloan–Shirley analytic
/// fit), normalized to D65 and clamped at zero.
///
/// A *response*, not a color — its absolute scale means nothing, only its shape
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
}
