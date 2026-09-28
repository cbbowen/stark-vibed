// Presentation resolve (§6.4): box-average the supersampled render down to the
// target. The last pass of a zoomed-out render, and absent entirely at 1:1.
//
// Why the whole pipeline is supersampled rather than the tiles being mip-filtered:
// minifying the *paint* is only one of three places presentation undersamples, and
// the other two cannot be fixed by a prefiltered texture at all.
//
//   · The tiles — `composite.wesl` takes one bilinear tap per fragment, so at
//     zoom z it reads one texel in every 1/z² the fragment covers.
//   · The canvas substrate — `media_common.wesl::substrate_height_at` samples a periodic height
//     field in canvas space, which moirés the moment a screen pixel spans more than
//     one substrate period.
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
// the surface's display encode, and the target carries encoded color in a non-sRGB
// format (§6.5), so averaging its texels directly would mix numbers on a curve and
// drag every edge toward the dark side of it. Decode, average, re-encode — through
// the selector the media pass encoded by (`lib/display.wesl`), so the pair is an
// identity whichever transfer the surface speaks. The color is weighted by alpha
// and the alpha averaged flat, which is what an unpremultiplied downsample is: on
// the opaque path every alpha is 1 and this is the plain mean, and on a cut-out
// export (§15.6) a fragment that is nine-tenths nothing contributes a tenth of a
// color rather than a full one.
//
// And the re-encode is a second walk down to 8-bit codes, so it carries its own
// quantization dither (§6.5): the average recovered the depth the source texels'
// dither spread across the block, and storing it back undithered would hand the
// banding straight back.

import package::lib::display::{decode_display, encode_display};
import package::lib::noise::{dither2};
import package::lib::quad::{fullscreen_clip};

struct Resolve {
    // Samples per axis.
    //
    // `@align(16)` is the block's alignment rather than this lane's: WGSL requires a
    // struct in the uniform address space to be 16-byte aligned and has no
    // struct-level attribute to say it with, so the first member carries it — which
    // is also what keeps the block a whole 16-byte row, its fourth lane spare.
    @align(16) samples: f32,
    // One code of the target's encoding, the same number `media_common.wesl` scales
    // its own store's dither by (§6.5).
    dither: f32,
    // The display transfer the media pass encoded by — `lib/display.wesl`'s
    // selector, as a float.
    transfer: f32,
}

@group(0) @binding(0) var<uniform> r: Resolve;
@group(0) @binding(1) var src: texture_2d<f32>;

// The weighted mean's denominator guard: a target block that is entirely
// transparent has no color to average, and returns black rather than a quotient of
// two zeros. Local and named for its own job — it is neither of
// `paint_common`'s two guards (this alpha is neither a tile's per-unit opacity nor a
// composited coverage, but the box filter's own accumulated weight), and a bare `EPS`
// would not say which of the three a reader is looking at.
const WEIGHT_EPS: f32 = 1e-5;

// This pass's dither identity — distinct from `media_common.wesl`'s, so the two
// stores share no pattern.
const DITHER_SEED: u32 = 1u;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let n = i32(r.samples);
    let transfer = u32(r.transfer);
    // The source is exactly `n` times the target in each axis, so this block is
    // always in bounds and nothing has to be clamped.
    let base = vec2<i32>(floor(pos.xy)) * n;
    var rgb = vec3<f32>(0.0);
    var a = 0.0;
    for (var j = 0; j < n; j++) {
        for (var i = 0; i < n; i++) {
            let t = textureLoad(src, base + vec2<i32>(i, j), 0);
            rgb += decode_display(transfer, t.rgb) * t.a;
            a += t.a;
        }
    }
    // The header's second walk down to 8 bits: the same law as the media pass's
    // store — one luma lane, one alpha lane, ±half a code — at this pass's own
    // pixel grid and seed.
    let d = dither2(vec2u(pos.xy), DITHER_SEED) * r.dither;
    return vec4<f32>(
        encode_display(transfer, rgb / max(a, WEIGHT_EPS)) + d.x,
        a / f32(n * n) + d.y,
    );
}
