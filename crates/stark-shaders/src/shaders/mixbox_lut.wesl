// Mixbox 2.0 sRGB → pigment concentrations, ported from `mixbox_rgb_to_latent` in
// the vendored GLSL (vendor/mixbox/shaders/mixbox.glsl). Mixbox 2.0 (c) 2022 Secret
// Weapons, authors Sarka Sochorova and Ondrej Jamriska; licensed CC BY-NC 4.0, see
// vendor/mixbox/LICENSE.
//
// This is the inverse of `mixbox_poly`'s `mixbox_eval_polynomial`, and it exists for
// one caller: the blend pass, which has to express a combination of *light* back as
// a mixture of *pigment* (§18.0.4). It is the only reason the LUT is on the GPU at
// all: everywhere else the direction is one-way — the CPU converts a picked color
// with the `mixbox` crate, and the GPU only ever evaluates the polynomial forward.
//
// Hand-written rather than transpiled by `build.rs` like `package::gen::mixbox_poly`:
// that generator works because the polynomial is pure arithmetic with no bindings, and
// this function is nothing but a texture lookup, whose bindings are ours.
//
// The concentrations are the whole answer here. Mixbox's full latent also carries a
// residual (`rgb − poly(c)`) that makes the round trip exact; the engine drops it
// everywhere so three concentrations fit alongside coverage (§6.7), and
// dropping it here keeps the blend agreeing with `rgb_to_channels` on the CPU.

// **Bindings 5 and 6 of group 0, which is a claim on someone else's group.** This
// module has no pipeline of its own — its one caller is `blend_mixbox.wesl`, whose
// other bindings come from `blend_common.wesl` at 0–4 and from itself at 7–8 (the two
// residual channels, §6.7). The three files have to agree on where each
// stops and the next starts, and nothing but this comment says so. If `blend_common`
// ever grows a sixth binding, it collides here, and the error will name a mangled
// identifier rather than any of the files.
@group(0) @binding(5) var pigment_lut: texture_2d<f32>;
@group(0) @binding(6) var pigment_samp: sampler;

// The LUT is a 64³ RGB cube unrolled into an 8×8 grid of 64×64 slices. `x` and `y`
// interpolate in hardware (the sampler is Linear, and a slice's samples span
// `[x0+0.5, x0+63.5]`, so a fetch never reaches a neighbouring slice); `z` picks two
// slices and mixes them here.
//
// The original's `MIXBOX_LUT(0.5/512).b < 0.1` probe — a flip test for APIs whose
// texture origin is bottom-left — is dropped: wgpu's origin is top-left and the PNG
// rows are uploaded in order, so the mapping is direct.
fn mixbox_srgb_to_concentrations(rgb_in: vec3<f32>) -> vec3<f32> {
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
        textureSampleLevel(pigment_lut, pigment_samp, uv0, 0.0).rgb,
        textureSampleLevel(pigment_lut, pigment_samp, uv1, 0.0).rgb,
        z - iz,
    );
}
