// Presentation resolve (§6.4): box-average the supersampled render down to the
// target. The last pass of a zoomed-out render, and absent entirely at 1:1.
//
// Why the whole pipeline is supersampled rather than the tiles being mip-filtered:
// minifying the *paint* is only one of three places presentation undersamples, and
// the other two cannot be fixed by a prefiltered texture at all.
//
//   · The tiles — `composite.wesl` takes one bilinear tap per fragment, so at
//     zoom z it reads one texel in every 1/z² the fragment covers.
//   · The canvas weave — `media_common.wesl::surface_at` samples a periodic height
//     field in canvas space, which moirés the moment a screen pixel spans more than
//     one weave period.
//   · The relief shading — the normal is a finite difference of the height field
//     over *screen* pixels, fed through a specular lobe and a tonemap. Shading is a
//     nonlinear function of the height, so the correct minified pixel is the average
//     of the shading and not the shading of the average: an impasto ridge thinner
//     than a screen pixel sparkles and crawls under a pan however well the height
//     that produced it was filtered.
//
// Supersampling is the one mechanism that covers all three, plus the outline and the
// guide overlay drawn over them, and it leaves the tile representation — aprons,
// the seam invariant, the copy-on-write pool — untouched.
//
// **The average is taken in light, not in code values.** The media pass ends in
// `linear_to_srgb`, and the target carries display-encoded sRGB in a non-sRGB format
// (§6.5), so averaging its texels directly would mix numbers on a curve and drag
// every edge toward the dark side of it. Decode, average, re-encode. The colour is
// weighted by alpha and the alpha averaged flat, which is what an unpremultiplied
// downsample is: on the opaque path every alpha is 1 and this is the plain mean, and
// on a cut-out export (§15.6) a fragment that is nine-tenths nothing contributes a
// tenth of a colour rather than a full one.

import package::color::{linear_to_srgb, srgb_to_linear};

struct Resolve {
    // n.x = samples per axis. The rest is the 16-byte stride a uniform member takes
    // anyway; it stays a float because that is the only scalar this file needs.
    n: vec4<f32>,
}

@group(0) @binding(0) var<uniform> r: Resolve;
@group(0) @binding(1) var src: texture_2d<f32>;

const EPS: f32 = 1e-5;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let n = i32(r.n.x);
    // The source is exactly `n` times the target in each axis, so this block is
    // always in bounds and nothing has to be clamped.
    let base = vec2<i32>(floor(pos.xy)) * n;
    var rgb = vec3<f32>(0.0);
    var a = 0.0;
    for (var j = 0; j < n; j++) {
        for (var i = 0; i < n; i++) {
            let t = textureLoad(src, base + vec2<i32>(i, j), 0);
            rgb += srgb_to_linear(t.rgb) * t.a;
            a += t.a;
        }
    }
    return vec4<f32>(linear_to_srgb(rgb / max(a, EPS)), a / f32(n * n));
}
