// Brush stamp rasterization for the Oklab color space (DESIGN.md §6.2, §6.6).
//
// One instanced quad per dab, writing two render targets of a tile:
//   target 0: color (Rgba16Float) — premultiplied Oklab (L,a,b) + coverage in a
//   target 1: aux   (Rg16Float)   — (height, wet), blended additively
//
// The instance layout is color-space-agnostic (channels + aux deposits); this
// shader interprets the channels as Oklab and writes premultiplied "over".
// Coverage is a procedural soft disc (mode 0, with `hardness`) or a sampled
// brush mask (mode 1). Each stamp is rotated by its (cos, sin).

struct TileXform {
    // params.xy = tile origin (canvas px); params.z = 2 / TILE_SIZE;
    // params.w = mode (0 = round/procedural, 1 = sampled mask).
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;
@group(1) @binding(0) var mask_tex: texture_2d<f32>;
@group(1) @binding(1) var mask_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) quad: vec2<f32>,    // local stamp coord in [-1, 1] (unrotated)
    @location(1) uv: vec2<f32>,      // mask sample coord in [0, 1]
    @location(2) oklab: vec3<f32>,   // pigment color (Oklab L,a,b)
    @location(3) params: vec4<f32>,  // hardness, flow_eff, height, wet
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec2<f32>,
    @location(1) shape: vec4<f32>,  // radius, hardness, flow_eff, unused
    @location(2) ch: vec4<f32>,     // color channels (Oklab in xyz)
    @location(3) aux: vec2<f32>,    // height, wet
    @location(4) rot: vec2<f32>,    // cos, sin
) -> VsOut {
    let corner = vec2<f32>(
        f32(vi & 1u) * 2.0 - 1.0,
        f32((vi >> 1u) & 1u) * 2.0 - 1.0,
    );
    let rotated = vec2<f32>(
        corner.x * rot.x - corner.y * rot.y,
        corner.x * rot.y + corner.y * rot.x,
    );
    let canvas_pos = center + rotated * shape.x;
    let s = xf.params.z;
    let ndc = (canvas_pos - xf.params.xy) * vec2<f32>(s, -s) + vec2<f32>(-1.0, 1.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.quad = corner;
    out.uv = corner * 0.5 + 0.5;
    out.oklab = ch.xyz;
    out.params = vec4<f32>(shape.y, shape.z, aux.x, aux.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let hardness = in.params.x;
    let flow = in.params.y;
    var coverage: f32;
    if xf.params.w < 0.5 {
        let d = length(in.quad);
        coverage = flow * (1.0 - smoothstep(min(hardness, 0.99), 1.0, d));
    } else {
        coverage = flow * textureSample(mask_tex, mask_samp, in.uv).r;
    }

    var out: FsOut;
    out.color = vec4<f32>(in.oklab * coverage, coverage);
    out.aux = vec4<f32>(in.params.z * coverage, in.params.w * coverage, 0.0, 0.0);
    return out;
}
