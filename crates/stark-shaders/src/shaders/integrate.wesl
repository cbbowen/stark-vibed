// Stroke *integrate* pass (DESIGN.md §6.2/§6.1). Every brush rasterizes its swept
// footprint into a *cleared scratch* tile (the stamp pass), then this fullscreen
// pass merges that scratch into a fresh copy-on-write tile over the base:
// `new = f(base, scratch)`, written to the color+aux MRT.
//
// It is a render pass (not compute) because the tile channels are `Rgba16Float` /
// `Rg16Float`, which are not core-WebGPU storage-texture formats. Everything is
// sampled with `textureLoad` (1:1 with the destination — no filtering), so the
// result is a deterministic function of canvas position and a tile's apron is
// rewritten identically to its neighbour's interior (the seam invariant, §6.4).
//
// In the normalized representation (§6.1): color = latent·opacity with opacity in
// `.a`, aux = (height, wet). The merge is the ordinary premultiplied-"over" on
// color and an additive sum of height/wet. Brush dynamics (the sequential stamp
// loop, §6.2) will replace this with richer read-modify-write physics; this plain
// merge is the pure-`add` fast path they must reduce to.

@group(0) @binding(0) var base_color: texture_2d<f32>;
@group(0) @binding(1) var base_aux: texture_2d<f32>;
@group(0) @binding(2) var scratch_color: texture_2d<f32>;
@group(0) @binding(3) var scratch_aux: texture_2d<f32>;

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
    let c = vec2<i32>(floor(frag.xy));
    let base_c = textureLoad(base_color, c, 0);
    let base_a = textureLoad(base_aux, c, 0);
    let s_c = textureLoad(scratch_color, c, 0); // stroke slab: latent·opacity, opacity in .a
    // Wide scratch aux: (laid height, laid wet·op, footprint coverage √cov, unused) —
    // see stamp_oklab.wesl's fs_main for the channel-by-channel story.
    let s_a = textureLoad(scratch_aux, c, 0);

    var out: FsOut;
    // Colour: lay the stroke slab over the base (premultiplied "over").
    out.color = s_c + base_c * (1.0 - s_c.a);
    // Height/wet accumulate: height is the amount of paint (§6.1).
    out.aux = vec4<f32>(base_a.x + s_a.x, base_a.y + s_a.y, 0.0, 0.0);
    return out;
}
