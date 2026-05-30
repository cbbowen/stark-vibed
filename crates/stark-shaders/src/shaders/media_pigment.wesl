// Media pass B for the PIGMENT color space (DESIGN.md §6.7).
//
// The composited color channels are accumulated optical *amounts* of four
// pigments (Phthalo Blue, Quinacridone Magenta, Hansa Yellow, Titanium White).
// We mix their Kubelka–Munk K and S (amount-weighted), evaluate the reflectance
// of the finite paint layer over the background substrate, and hand off to the
// shared lighting/composite tail (`media_common.wesl`). White's high scattering
// gives physical hiding power; thin stains stay transparent.

import package::media_common::{m, comp_color, comp_aux, fullscreen_clip, finish};

// Per-pigment K (absorption) and S (scattering) in linear RGB — mirror pigment.rs.
const K0 = vec3<f32>(9.60, 2.10, 0.134);    // Phthalo Blue
const K1 = vec3<f32>(0.074, 6.27, 0.553);   // Quinacridone Magenta
const K2 = vec3<f32>(0.021, 0.125, 6.27);   // Hansa Yellow
const K3 = vec3<f32>(0.021, 0.018, 0.016);  // Titanium White
const S0 = vec3<f32>(0.40, 0.40, 0.40);
const S1 = vec3<f32>(0.40, 0.40, 0.40);
const S2 = vec3<f32>(0.50, 0.50, 0.40);
const S3 = vec3<f32>(6.00, 6.00, 6.00);

// Reflectance of a finite Kubelka–Munk layer (mixed K, S = absorption/scattering
// optical thickness) over a substrate of reflectance `rg`. As S→0 (thin/clear)
// this → rg; as S grows (white) it → the masstone R∞.
fn km_finite(k: vec3<f32>, s: vec3<f32>, rg: vec3<f32>) -> vec3<f32> {
    let ss = max(s, vec3<f32>(1e-4));
    let a = 1.0 + k / ss;
    let b = sqrt(max(a * a - 1.0, vec3<f32>(0.0)));
    let bs = clamp(b * ss, vec3<f32>(1e-3), vec3<f32>(20.0)); // b·S (scatter thickness)
    let d = b / tanh(bs);                                     // b·coth(b·S)
    return clamp((1.0 - rg * (a - d)) / (a - rg + d), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(frag.xy));
    let dim = vec2<i32>(textureDimensions(comp_aux));

    let amounts = textureLoad(comp_color, p, 0); // four pigment amounts
    let wet = textureLoad(comp_aux, p, 0).y;
    let total = amounts.x + amounts.y + amounts.z + amounts.w;

    // Amount-weighted Kubelka–Munk mix → reflectance over the substrate.
    let k = amounts.x * K0 + amounts.y * K1 + amounts.z * K2 + amounts.w * K3;
    let s = amounts.x * S0 + amounts.y * S1 + amounts.z * S2 + amounts.w * S3;
    let base = km_finite(k, s, m.bg.rgb);

    // Paint presence → how much the lit layer shows over bare background.
    let coverage = 1.0 - exp(-total);
    return finish(base, coverage, wet, p, dim);
}
