// Wet-on-wet diffusion (DESIGN.md §6.2, "Wet-on-wet diffusion"). One ping-pong
// iteration over a composited *region* of the stroke's tiles: each texel relaxes
// toward its neighbours' average, weighted by how wet it is and how hard the brush
// bore down there (the footprint `rate`). The Wet brush runs this a fixed number of
// times after depositing, then the region is sliced back into copy-on-write tiles.
//
// In the normalized representation (§6.1) color is the latent pigment premultiplied
// by opacity and aux is (thickness, wet·opacity). Diffusing the *premultiplied*
// color is the correct translucent-paint blend (it conserves pigment under a
// symmetric kernel); thickness diffuses too, so wet impasto levels. Sampled with
// `textureLoad` (1:1, no filtering) so it is deterministic per region texel.

struct Params {
    knobs: vec4<f32>, // strength, _, _, _
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src_color: texture_2d<f32>;
@group(0) @binding(2) var src_aux: texture_2d<f32>;
@group(0) @binding(3) var rate_tex: texture_2d<f32>; // footprint: opacity in .a

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

    let color_c = textureLoad(src_color, c, 0);
    let aux_c = textureLoad(src_aux, c, 0);

    // Diffusion rate at this texel: localized to the brush footprint (`rate.a`) and
    // weighted by wetness (only wet paint flows). Straight wet = (wet·opacity)/opacity.
    let op = color_c.a;
    let wet = clamp(aux_c.y / max(op, 1e-4), 0.0, 1.0);
    let r = clamp(p.knobs.x * textureLoad(rate_tex, c, 0).a * wet, 0.0, 1.0);

    // 4-neighbour average (clamped at the region edge → reflecting-ish boundary).
    let l = clamp(c + vec2<i32>(-1, 0), vec2<i32>(0), dim - vec2<i32>(1));
    let rt = clamp(c + vec2<i32>(1, 0), vec2<i32>(0), dim - vec2<i32>(1));
    let u = clamp(c + vec2<i32>(0, -1), vec2<i32>(0), dim - vec2<i32>(1));
    let d = clamp(c + vec2<i32>(0, 1), vec2<i32>(0), dim - vec2<i32>(1));

    let blur_color = 0.25 * (textureLoad(src_color, l, 0) + textureLoad(src_color, rt, 0)
        + textureLoad(src_color, u, 0) + textureLoad(src_color, d, 0));
    let blur_aux = 0.25 * (textureLoad(src_aux, l, 0) + textureLoad(src_aux, rt, 0)
        + textureLoad(src_aux, u, 0) + textureLoad(src_aux, d, 0));

    var out: FsOut;
    out.color = mix(color_c, blur_color, r);
    out.aux = mix(aux_c, blur_aux, r);
    return out;
}
