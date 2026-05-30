// Swept-segment brush rasterization for the OKLAB color space (DESIGN.md §6.2).
//
// Shares all geometry/sweep math with the other color spaces via
// `stamp_common.wesl`; here the fragment turns the swept optical depth into a
// premultiplied-Oklab deposit with coverage in alpha (the over-blend across
// overlapping segment quads then sums depth exactly):
//   target 0: color (Rgba16Float) — premultiplied Oklab (L,a,b) + coverage in a
//   target 1: aux   (Rg16Float)   — (height, wet)·coverage

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
    let coverage = 1.0 - exp(-flow * swept_depth(in));

    var out: FsOut;
    out.color = vec4<f32>(in.ch.xyz * coverage, coverage);
    out.aux = vec4<f32>(in.misc.z * coverage, in.misc.w * coverage, 0.0, 0.0);
    return out;
}
