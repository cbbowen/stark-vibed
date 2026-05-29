// Brush stamp rasterization (DESIGN.md §6.2), multi-channel.
//
// One instanced quad per brush stamp, writing two render targets of a tile:
//   target 0: color  (Rgba16Float) — premultiplied Oklab, blended "over"
//   target 1: aux    (Rg16Float)   — (height, wet), blended additively
//
// Stamps thus build a continuous stroke in color while piling up impasto height
// and wetness. Color arrives already in Oklab from the CPU (DESIGN.md §6.5).

struct TileXform {
    // params.xy = this tile's origin in canvas pixels.
    // params.z  = 2 / TILE_SIZE (canvas px -> NDC scale). params.w unused.
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) quad: vec2<f32>,    // local stamp coord in [-1, 1]
    @location(1) oklab: vec3<f32>,   // pigment color (Oklab)
    @location(2) deposit: vec4<f32>, // hardness, flow_eff, height, wet
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec2<f32>,
    @location(1) shape: vec4<f32>,  // radius, hardness, flow_eff, height
    @location(2) col: vec4<f32>,    // okL, oka, okb, wet
) -> VsOut {
    let corner = vec2<f32>(
        f32(vi & 1u) * 2.0 - 1.0,
        f32((vi >> 1u) & 1u) * 2.0 - 1.0,
    );
    let canvas_pos = center + corner * shape.x;
    let s = xf.params.z;
    let ndc = (canvas_pos - xf.params.xy) * vec2<f32>(s, -s) + vec2<f32>(-1.0, 1.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.quad = corner;
    out.oklab = col.xyz;
    out.deposit = vec4<f32>(shape.y, shape.z, shape.w, col.w);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let d = length(in.quad);
    let hardness = min(in.deposit.x, 0.99);
    let flow = in.deposit.y;
    let coverage = flow * (1.0 - smoothstep(hardness, 1.0, d));

    var out: FsOut;
    out.color = vec4<f32>(in.oklab * coverage, coverage);
    out.aux = vec4<f32>(in.deposit.z * coverage, in.deposit.w * coverage, 0.0, 0.0);
    return out;
}
