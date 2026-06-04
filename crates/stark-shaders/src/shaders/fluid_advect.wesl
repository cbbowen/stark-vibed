// Fluid advection (DESIGN.md §6.2, "Fluid"). One semi-Lagrangian ping-pong step over
// the composited stroke region: each texel back-traces along the (static, injected)
// velocity field and bilinearly samples the paint it came from, so wet paint is
// dragged in the brush's direction. Run a fixed number of times after depositing,
// then the region is sliced back into copy-on-write tiles (the wet brush's machinery).
//
// 16-bit-float region textures are filterable in core WebGPU, so the back-trace uses a
// linear+clamp sampler. Where velocity is 0 (outside the footprint / in the halo) the
// back-trace lands on the texel itself → identity, so the write-back stays seam-free.

struct Params {
    knobs: vec4<f32>, // dt (canvas px per unit velocity), _, _, _
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var velocity: texture_2d<f32>;
@group(0) @binding(2) var src_color: texture_2d<f32>;
@group(0) @binding(3) var src_aux: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

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
    let dims = vec2<f32>(textureDimensions(src_color));
    let uv = frag.xy / dims;
    let vel = textureSampleLevel(velocity, samp, uv, 0.0).xy;
    // The region is 1:1 with canvas px, so velocity (px) is a texel offset directly.
    let src = (frag.xy - vel * p.knobs.x) / dims;

    var out: FsOut;
    out.color = textureSampleLevel(src_color, samp, src, 0.0);
    out.aux = textureSampleLevel(src_aux, samp, src, 0.0);
    return out;
}
