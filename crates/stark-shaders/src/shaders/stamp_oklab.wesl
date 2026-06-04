// Swept-segment brush rasterization for the OKLAB color space (DESIGN.md §6.2).
//
// Shares all geometry/sweep math with the other color spaces via
// `stamp_common.wesl`. The footprint deposits into a *cleared scratch* tile in the
// normalized representation (DESIGN.md §6.1): colour is premultiplied by **opacity**
// (not coverage), thickness lives in the aux. The brush's soft footprint coverage is
// split as `√cov` across the opacity and the thickness, so when the media shader
// combines them into the visible alpha (`1 − exp(−K·op·T)`) the coverage is applied
// exactly once. Within a stroke the colour target over-blends (premult-consistent)
// and the aux target accumulates additively.
//   target 0: color (Rgba16Float) — premultiplied Oklab (L,a,b)·op + opacity in a
//   target 1: aux   (Rg16Float)   — (thickness, wet·op)

import package::stamp_common::{VsOut, FsOut, sweep_vertex, swept_depth, surface_tooth, reservoir_color, is_carry};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) start: vec2<f32>,
    @location(1) dir: vec2<f32>,
    @location(2) geom: vec4<f32>,
    @location(3) aux: vec4<f32>,
) -> VsOut {
    return sweep_vertex(vi, start, dir, geom, aux);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let flow = in.misc.y;
    let cov = surface_tooth(in.misc.z, 1.0 - exp(-flow * swept_depth(in)), in.clip.xy);
    let scov = sqrt(cov); // split the footprint coverage across opacity and thickness

    // Each part of the tip lays the color its lateral band picked up (DESIGN §6.2).
    let ch = reservoir_color(in.u, in.local);

    // Color is premultiplied by opacity. For a normal/mixer brush the reservoir holds
    // straight colour `(C, 1)` and the opacity is the brush's own (`in.op`). For the
    // carry knife the reservoir already holds premultiplied dragged paint `(C·d, d)`,
    // so the deposit is just that scaled by √cov — the dragged amount `d` is the
    // opacity, fading the smear as the knife empties.
    var col: vec4<f32>;
    if is_carry() {
        col = vec4<f32>(ch.xyz * scov, ch.a * scov);
    } else {
        let op = in.op * scov;
        col = vec4<f32>(ch.xyz * op, op);
    }
    let thick = in.misc.z * scov; // thickness laid here (height rate · √cov)

    var out: FsOut;
    out.color = col;                                  // premult by opacity; over-blends
    out.aux = vec4<f32>(thick, in.misc.w * col.a, 0.0, 0.0); // additive thickness, wet·op
    return out;
}
