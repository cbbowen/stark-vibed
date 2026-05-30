//! Kubelka–Munk pigment model (DESIGN.md §6.7).
//!
//! The pigment color space stores four channels — the accumulated optical
//! *amounts* of four real pigments (Phthalo Blue, Quinacridone Magenta, Hansa
//! Yellow, Titanium White). This module is the CPU side: the per-pigment
//! absorption `K` and scattering `S` coefficients (in linear RGB), the picker
//! conversions (RGB → a non-negative least-squares pigment mix, and a pigment
//! mix → its rendered RGB), all mirroring the K–M math in `media_pigment.wesl`.

use crate::color::{linear_to_srgb, srgb_to_linear};

/// Number of pigments. Order: Phthalo Blue, Quinacridone Magenta, Hansa Yellow,
/// Titanium White. Must match `media_pigment.wesl`.
pub const PIGMENTS: usize = 4;

/// Per-pigment absorption `K` per unit amount, in linear RGB. Tuned by eye so
/// each pigment's masstone (`R∞`) reads true; stains have low `S` (transparent),
/// Titanium White has high `S` (opaque, hiding).
const K: [[f32; 3]; PIGMENTS] = [
    [9.60, 2.10, 0.134], // Phthalo Blue
    [0.074, 6.27, 0.553], // Quinacridone Magenta
    [0.021, 0.125, 6.27], // Hansa Yellow
    [0.021, 0.018, 0.016], // Titanium White
];

/// Per-pigment scattering `S` per unit amount, in linear RGB.
const S: [[f32; 3]; PIGMENTS] = [
    [0.40, 0.40, 0.40], // Phthalo Blue (staining → low scatter)
    [0.40, 0.40, 0.40], // Quinacridone Magenta
    [0.50, 0.50, 0.40], // Hansa Yellow
    [6.00, 6.00, 6.00], // Titanium White (opaque → high scatter)
];

/// Kubelka–Munk reflectance of an infinitely thick layer, one channel.
fn r_infinite(k: f32, s: f32) -> f32 {
    let ratio = k / s.max(1e-6);
    (1.0 + ratio - (ratio * ratio + 2.0 * ratio).sqrt()).clamp(0.0, 1.0)
}

/// A pigment's masstone (full-strength reflectance) in linear RGB.
fn masstone(p: usize) -> [f32; 3] {
    [
        r_infinite(K[p][0], S[p][0]),
        r_infinite(K[p][1], S[p][1]),
        r_infinite(K[p][2], S[p][2]),
    ]
}

/// Straight sRGB → four non-negative pigment concentrations (DESIGN.md §6.7).
///
/// An approximate least-squares fit: find `c ≥ 0` minimizing
/// `‖Σ c_p·masstone_p − target‖²` in linear RGB, so the familiar RGB picker maps
/// onto the pigment basis. The actual rendered color comes from K–M mixing of
/// these concentrations, which is more faithful than this linear fit implies.
pub fn srgb_to_pigments(rgb: [f32; 3]) -> [f32; 4] {
    let target = [
        srgb_to_linear(rgb[0]),
        srgb_to_linear(rgb[1]),
        srgb_to_linear(rgb[2]),
    ];
    let basis: [[f32; 3]; PIGMENTS] = [masstone(0), masstone(1), masstone(2), masstone(3)];

    // Normal equations: AᵀA c = Aᵀt, solved by projected gradient (c ≥ 0).
    let mut ata = [[0.0f32; PIGMENTS]; PIGMENTS];
    let mut atb = [0.0f32; PIGMENTS];
    for i in 0..PIGMENTS {
        for j in 0..PIGMENTS {
            ata[i][j] = (0..3).map(|c| basis[i][c] * basis[j][c]).sum();
        }
        atb[i] = (0..3).map(|c| basis[i][c] * target[c]).sum();
    }
    // Step size ≤ 1/‖AᵀA‖; trace is a safe (over)estimate of the top eigenvalue.
    let trace: f32 = (0..PIGMENTS).map(|i| ata[i][i]).sum();
    let step = 1.0 / trace.max(1e-6);

    let mut c = [0.0f32; PIGMENTS];
    for _ in 0..400 {
        for i in 0..PIGMENTS {
            let grad: f32 = (0..PIGMENTS).map(|j| ata[i][j] * c[j]).sum::<f32>() - atb[i];
            c[i] = (c[i] - step * grad).max(0.0);
        }
    }
    c
}

/// Four pigment concentrations → the masstone RGB they render to (picker
/// readout / export). Forward K–M of the mix at full hiding.
pub fn pigments_to_srgb(c: [f32; 4]) -> [f32; 3] {
    let mut k = [0.0f32; 3];
    let mut s = [0.0f32; 3];
    for p in 0..PIGMENTS {
        for ch in 0..3 {
            k[ch] += c[p] * K[p][ch];
            s[ch] += c[p] * S[p][ch];
        }
    }
    let lin = [
        r_infinite(k[0], s[0]),
        r_infinite(k[1], s[1]),
        r_infinite(k[2], s[2]),
    ];
    [
        linear_to_srgb(lin[0]),
        linear_to_srgb(lin[1]),
        linear_to_srgb(lin[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_pigment_round_trips_in_gamut() {
        // A pure pigment's masstone should map back near that pigment.
        for p in 0..PIGMENTS {
            let rgb = pigments_to_srgb({
                let mut c = [0.0; 4];
                c[p] = 1.0;
                c
            });
            let fit = srgb_to_pigments(rgb);
            assert!(fit[p] > 0.3, "pigment {p} should dominate its own masstone: {fit:?}");
        }
    }

    #[test]
    fn white_is_light_and_opaque() {
        let white = pigments_to_srgb([0.0, 0.0, 0.0, 1.0]);
        assert!(white.iter().all(|&c| c > 0.85), "titanium white masstone: {white:?}");
    }
}
