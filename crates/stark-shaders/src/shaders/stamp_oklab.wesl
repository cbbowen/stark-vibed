// Swept-segment brush rasterization for the OKLAB color space (DESIGN.md §6.2).
//
// Shares all geometry/sweep math with the other color spaces via
// `stamp_common.wesl`; here the fragment turns the swept optical depth into a
// premultiplied-Oklab deposit with coverage in alpha (the over-blend across
// overlapping segment quads then sums depth exactly):
//   target 0: color (Rgba16Float) — premultiplied Oklab (L,a,b) + coverage in a
//   target 1: aux   (Rg16Float)   — (height, wet)·coverage

import package::stamp_common::{VsOut, FsOut, sweep_vertex, swept_depth, surface_tooth, reservoir_color, is_carry};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) start: vec2<f32>,
    @location(1) dir: vec2<f32>,
    @location(2) geom: vec4<f32>,
    @location(3) aux: vec2<f32>,
) -> VsOut {
    return sweep_vertex(vi, start, dir, geom, aux);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let flow = in.misc.y;
    let base = 1.0 - exp(-flow * swept_depth(in));
    let height = in.misc.z;
    let coverage = surface_tooth(height, base, in.clip.xy);

    // Each part of the tip lays the color its lateral band picked up (DESIGN §6.2).
    let ch = reservoir_color(in.u, in.local);

    // Colour alpha carries the deposit's *amount*: for every additive brush the
    // reservoir alpha is 1, so this is the usual `(ch·coverage, coverage)`. For the
    // carry knife the reservoir holds premultiplied film `(ch·film, film)`, so this
    // becomes `(ch·film·coverage, film·coverage)` — a premult-consistent film — and
    // the aux instead carries the contact coverage the combine scrapes with.
    var out: FsOut;
    out.color = vec4<f32>(ch.xyz * coverage, ch.a * coverage);
    let amount = select(in.misc.z, 1.0, is_carry()); // carry: contact coverage; else height
    out.aux = vec4<f32>(amount * coverage, in.misc.w * coverage, 0.0, 0.0);
    return out;
}
