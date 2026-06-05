// Separable Gaussian bleed for the Wet brush (DESIGN.md §6.2). Replaces the slow
// explicit (Jacobi) diffusion — which spread only ~√iterations px per stroke, so a
// dozen passes still bled just a few px — with a **two-pass separable Gaussian** whose
// radius scales with `bleed`. A 1-D Gaussian of σ in two passes equals ~σ² Jacobi
// iterations, so paint bleeds a wide radius in 2 passes instead of a few px in 12.
//
// Pass 1 (horizontal, `combine`=0) is a pure blur. Pass 2 (vertical, `combine`=1)
// finishes the 2-D Gaussian and mixes it back over the *original* region, gated by the
// footprint rate × wetness — so the bleed stays localized to where the brush bore down
// and is identity where the rate is 0 (the halo / tile aprons), keeping the region
// write-back seam-free. Diffusing the premultiplied colour conserves pigment (symmetric
// kernel); thickness diffuses too, so wet impasto levels. `textureLoad` (exact texels)
// keeps it deterministic.

struct Params {
    knobs: vec4<f32>, // dir.x, dir.y, sigma (px), combine (0 = pure blur, 1 = +mix)
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src_color: texture_2d<f32>;
@group(0) @binding(2) var src_aux: texture_2d<f32>;
@group(0) @binding(3) var orig_color: texture_2d<f32>;
@group(0) @binding(4) var orig_aux: texture_2d<f32>;
@group(0) @binding(5) var rate_tex: texture_2d<f32>; // footprint: opacity in .a

// Max kernel radius (px) — the loop bound; covers `sigma` up to ≈ MAX_RADIUS / 3
// (kept in sync with `MAX_SIGMA` in gpu/stroke.rs).
const MAX_RADIUS: i32 = 42;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let dim = vec2<i32>(textureDimensions(src_color));
    let c = vec2<i32>(floor(frag.xy));
    let dir = vec2<i32>(i32(p.knobs.x), i32(p.knobs.y)); // (1,0) horizontal | (0,1) vertical
    let sigma = max(p.knobs.z, 1e-4);
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    // Uniform loop bound (sigma is a uniform → no divergence); a tight radius keeps
    // the cost down for small sigma.
    let radius = min(i32(ceil(3.0 * sigma)), MAX_RADIUS);

    var sum_c = vec4<f32>(0.0);
    var sum_a = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = -radius; i <= radius; i = i + 1) {
        let w = exp(-f32(i * i) * inv2s2);
        let tc = clamp(c + dir * i, vec2<i32>(0), dim - vec2<i32>(1));
        sum_c = sum_c + w * textureLoad(src_color, tc, 0);
        sum_a = sum_a + w * textureLoad(src_aux, tc, 0);
        wsum = wsum + w;
    }
    let blur_c = sum_c / wsum;
    let blur_a = sum_a / wsum;

    var out: FsOut;
    if p.knobs.w < 0.5 {
        // Horizontal pass: pure 1-D blur.
        out.color = blur_c;
        out.aux = blur_a;
    } else {
        // Vertical pass + combine: blend the full Gaussian over the original, gated by
        // how hard the brush bore down (rate) and how wet the paint is.
        let oc = textureLoad(orig_color, c, 0);
        let oa = textureLoad(orig_aux, c, 0);
        let wet = clamp(oa.y / max(oc.a, 1e-4), 0.0, 1.0);
        let r = clamp(textureLoad(rate_tex, c, 0).a * wet, 0.0, 1.0);
        out.color = mix(oc, blur_c, r);
        out.aux = mix(oa, blur_a, r);
    }
    return out;
}
