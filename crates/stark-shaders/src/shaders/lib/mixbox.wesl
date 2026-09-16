// Mixbox's round trip (§6.7, §18.0.4): a pigment mixture out to light, and light back
// into a mixture. The three pigment shaders share it.
//
// A binding-free leaf, like every module under `lib/` (§2), and it stays one because
// the LUT texture and its sampler arrive as **arguments**, the way `lib/sample.wesl`
// takes the texture it reads — so each importer owns the slot it declares.
//
// Mixbox 2.0 (c) 2022 Secret Weapons, authors Sarka Sochorova and Ondrej Jamriska;
// licensed CC BY-NC 4.0, see vendor/mixbox/LICENSE. `mixbox_eval_polynomial` is
// transpiled from that GLSL at build time; the inverse below is hand-written, because
// that generator works on pure arithmetic and this is a texture lookup.

import package::gen::mixbox_poly::{mixbox_eval_polynomial};
import package::lib::color::{srgb_to_linear, linear_to_srgb};

// The two halves of a Mixbox color: three of the four trained pigments' concentrations
// (`c3 = 1 − (c0+c1+c2)` is derived), and the residual the polynomial cannot express
// (§6.7).
struct Pigment {
    conc: vec3<f32>,
    res: vec3<f32>,
}

// A pigment mixture as **linear sRGB**: Mixbox's `latent_to_rgb`, `poly(c) + r`, with
// its clamp, linearized.
//
// Linear rather than sRGB because that is where every caller goes next — the media pass
// lights it, the blend pass carries it out to XYZ, the filter pass to Oklab or straight
// into the spectral sum.
fn mixbox_to_linear(p: Pigment) -> vec3<f32> {
    let srgb = clamp(mixbox_eval_polynomial(p.conc) + p.res, vec3<f32>(0.0), vec3<f32>(1.0));
    return srgb_to_linear(srgb);
}

// Linear sRGB → a pigment mixture: `mixbox_rgb_to_latent`, spelled out.
//
// The LUT is indexed by sRGB on `[0,1]`, so the clamp precedes it — `max(…, 0)` because
// an adjustment can step outside the gamut and a summed spectrum can dip below zero.
// The residual is then what the polynomial could not say about the color that arrived.
fn mixbox_from_linear(lut: texture_2d<f32>, samp: sampler, lin: vec3<f32>) -> Pigment {
    let srgb = clamp(
        linear_to_srgb(max(lin, vec3<f32>(0.0))),
        vec3<f32>(0.0), vec3<f32>(1.0),
    );
    let conc = mixbox_srgb_to_concentrations(lut, samp, srgb);
    return Pigment(conc, srgb - mixbox_eval_polynomial(conc));
}

// sRGB → pigment concentrations, ported from `mixbox_rgb_to_latent` in the vendored
// GLSL. The concentrations alone: the residual is `mixbox_from_linear`'s to compute,
// being a difference against the polynomial rather than anything the table holds.
//
// The LUT is a 64³ RGB cube unrolled into an 8×8 grid of 64×64 slices. `x` and `y`
// interpolate in hardware (the sampler is Linear, and a slice's samples span
// `[x0+0.5, x0+63.5]`, so a fetch never reaches a neighbouring slice); `z` picks two
// slices and mixes them here.
//
// The original's `MIXBOX_LUT(0.5/512).b < 0.1` probe — a flip test for APIs whose
// texture origin is bottom-left — is dropped: wgpu's origin is top-left and the PNG
// rows are uploaded in order, so the mapping is direct.
fn mixbox_srgb_to_concentrations(
    lut: texture_2d<f32>, samp: sampler, rgb_in: vec3<f32>,
) -> vec3<f32> {
    let rgb = clamp(rgb_in, vec3<f32>(0.0), vec3<f32>(1.0));

    let x = rgb.r * 63.0;
    let y = rgb.g * 63.0;
    let z = rgb.b * 63.0;

    let iz = floor(z);
    let x0 = (iz % 8.0) * 64.0;
    let y0 = floor(iz / 8.0) * 64.0;
    let x1 = ((iz + 1.0) % 8.0) * 64.0;
    let y1 = floor((iz + 1.0) / 8.0) * 64.0;

    let uv0 = vec2<f32>(x0 + x + 0.5, y0 + y + 0.5) / 512.0;
    let uv1 = vec2<f32>(x1 + x + 0.5, y1 + y + 0.5) / 512.0;

    // At `iz == 63` the second slice is off the end of the grid; the sampler clamps
    // it and `z - iz` is 0 there, so it contributes nothing — as in the original.
    return mix(
        textureSampleLevel(lut, samp, uv0, 0.0).rgb,
        textureSampleLevel(lut, samp, uv1, 0.0).rgb,
        z - iz,
    );
}
