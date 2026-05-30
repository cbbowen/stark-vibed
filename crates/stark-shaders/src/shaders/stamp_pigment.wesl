// Swept-segment brush rasterization for the PIGMENT color space (DESIGN.md §6.7).
//
// Shares all geometry/sweep math with the other color spaces via
// `stamp_common.wesl`; here the four color channels are pigment *amounts*
// deposited **additively**: amount = flow·depth (the swept geometric optical
// depth, which sums exactly across a stroke's segments under additive blending),
// and each pigment channel writes concentration·amount. The media pass
// (`media_pigment.wesl`) renders the accumulated amounts via Kubelka–Munk.
//   target 0: color (Rgba16Float) — four pigment amounts, additive
//   target 1: aux   (Rg16Float)   — (height, wet)·amount, additive

import package::stamp_common::{VsOut, FsOut, sweep_vertex, swept_depth};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) start: vec2<f32>,
    @location(1) dir: vec2<f32>,
    @location(2) geom: vec4<f32>,
    @location(3) ch: vec4<f32>,
    @location(4) aux: vec2<f32>,
) -> VsOut {
    return sweep_vertex(vi, start, dir, geom, ch, aux);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let flow = in.misc.y;
    let amount = flow * swept_depth(in);

    var out: FsOut;
    out.color = in.ch * amount;
    out.aux = vec4<f32>(in.misc.z * amount, in.misc.w * amount, 0.0, 0.0);
    return out;
}
