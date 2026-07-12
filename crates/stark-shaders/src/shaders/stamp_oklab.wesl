// Swept-segment brush rasterization for the OKLAB color space (DESIGN.md §6.2).
//
// Shares all geometry/sweep math with the other color spaces via
// `stamp_common.wesl`. The footprint deposits into a *cleared scratch* tile in the
// normalized representation (DESIGN.md §6.1): colour is premultiplied by **opacity**
// (not coverage), height lives in the aux. The brush's soft footprint coverage is
// split as `√cov` across the opacity and the thickness, so when the media shader
// combines them into the visible alpha (`1 − exp(−K·op·T)`) the coverage is applied
// exactly once. Within a stroke the colour target over-blends (premult-consistent)
// and the aux target accumulates additively.
//   target 0: color (Rgba16Float) — premultiplied Oklab (L,a,b)·op + opacity in a
//   target 1: aux   (Rgba16Float, the wide *scratch* layout — persistent tiles keep only
//             (height, wet)) — (height, wet·op, footprint coverage √cov, unused)

import package::stamp_common::{VsOut, FsOut, sweep_vertex, swept_depth, surface_tooth, brush_color};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) start: vec2<f32>,
    @location(1) dir: vec2<f32>,
    @location(2) geom: vec4<f32>,
    @location(3) aux: vec4<f32>,
    @location(4) extra: vec4<f32>,
) -> VsOut {
    return sweep_vertex(vi, start, dir, geom, aux, extra);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let flow = in.misc.y;
    let cov = surface_tooth(in.misc.z, 1.0 - exp(-flow * swept_depth(in)), in.clip.xy);
    let scov = sqrt(cov); // split the footprint coverage across opacity and thickness
    let op = in.op * scov;

    var out: FsOut;
    // Premultiplied by opacity: the colour target over-blends, so per-arc rates sum
    // through the overlapping segment quads to the local deposit.
    out.color = vec4<f32>(brush_color() * op, op);
    // The wide scratch aux (DESIGN §6.1):
    //   .x = laid height (per-segment attr `in.misc.z`; additive over the overlap →
    //        builds impasto);
    //   .y = laid wet;
    //   .z = footprint coverage (√cov), summed across the overlapping quads;
    //   .w = unused.
    // Height is the amount of paint; the colour alpha is per-unit opacity only and
    // never represents amount (§6.1).
    out.aux = vec4<f32>(in.misc.z * scov, in.misc.w * op, scov, 0.0);
    return out;
}
