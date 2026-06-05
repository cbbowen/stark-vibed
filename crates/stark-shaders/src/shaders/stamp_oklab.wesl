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
//   target 1: aux   (Rg16Float)   — (thickness, wet·op)

import package::stamp_common::{VsOut, FsOut, sweep_vertex, swept_depth, surface_tooth, reservoir_color, reservoir_aux, add_frac};

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

    // Each part of the tip lays the color its lateral band carries (DESIGN §6.2). The
    // reservoir alpha is the carried *presence*: 1 for a plain brush (flat reservoir
    // `(C, 1)`), or the smear's per-band load `(C·load, load)`. Either way the deposit
    // is `(C · presence · op · √cov, presence · op · √cov)` — premultiplied by opacity,
    // with the smear automatically fading as its load runs out.
    let ch = reservoir_color(in.u, in.local);
    let op = in.op * scov;
    // Thickness laid here is one continuous sum (no branch): the brush's own height rate
    // (`in.misc.z`, gated by the `add` axis) plus the smear-lifted canvas height the
    // reservoir carries (a rate already normalized by the segment overlap, zero where
    // nothing was lifted). Both ride `√cov`, so the footprint coverage applies once.
    // Height is independent of opacity (DESIGN §6.1).
    let lifted = reservoir_aux(in.u, in.local); // smear-lifted height rate (canvas paint)
    let thick = (add_frac() * in.misc.z + lifted) * scov;

    var out: FsOut;
    out.color = vec4<f32>(ch.xyz * op, ch.a * op);
    // aux = (thickness, wet·op, **smear-lifted thickness alone**, _). The wide scratch aux
    // carries the lifted part separately so the integrate can scrape exactly what the smear
    // re-lays — moving paint, not duplicating it (DESIGN §6.2). Persistent tiles keep only
    // (thickness, wet); the integrate writes that compact form out.
    out.aux = vec4<f32>(thick, in.misc.w * out.color.a, lifted * scov, 0.0);
    return out;
}
