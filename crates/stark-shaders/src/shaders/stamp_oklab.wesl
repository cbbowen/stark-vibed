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
//             (height, wet)) — (own add-height, own wet·op, footprint coverage √cov,
//             deposited height·√cov); the channel-by-channel story is in `fs_main`

import package::stamp_common::{VsOut, FsOut, sweep_vertex, swept_depth, surface_tooth, reservoir_color, reservoir_dep_height, add_frac};

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

    // Colour laid here is the reservoir's deposit payload (premultiplied): for a plain
    // brush the flat reservoir's `add·brush`; for a lift/deposit brush the mixer's
    // `deposit·tool + add·brush` (the brush's own colour is folded in by the mixer, since
    // one colour texture carries both). The sweep colour target over-blends it, so per-arc
    // rates sum through the overlapping segment quads to the local deposit.
    let ch = reservoir_color(in.u, in.local);
    let op = in.op * scov;

    let dep_h = reservoir_dep_height(in.u, in.local); // tool deposited height (per-column rate)
    let af = add_frac();

    var out: FsOut;
    out.color = vec4<f32>(ch.xyz * op, ch.a * op);
    // The wide scratch aux separates the height contributions so the integrate can treat them
    // differently (DESIGN §6.1/§6.2):
    //   .x = the brush's *own* add-height (per-segment attr `in.misc.z`, gated by `add`),
    //        laid plain-paint style (additive over the overlap → builds impasto);
    //   .y = the brush's own wet;
    //   .z = footprint coverage (√cov), summed for the integrate's per-pixel contact —
    //        which both feathers the tool deposit and drives the lift's exact
    //        multiplicative removal (`lift × contact` of the base height) — and the ridge;
    //   .w = the tool's **deposited height** (≥ 0), which the integrate divides by the
    //        coverage and adds to the base. Height is the amount of paint; the colour
    //        alpha is per-unit opacity only and never represents amount (§6.1).
    out.aux = vec4<f32>(af * in.misc.z * scov, af * in.misc.w * op, scov, dep_h * scov);
    return out;
}
