// Swept-segment brush rasterization for the OKLAB color space (DESIGN.md §6.2).
//
// Shares all geometry/sweep math with the other color spaces via
// `stamp_common.wesl`. The footprint deposits into a *cleared scratch* tile in the
// normalized representation (DESIGN.md §6.1): colour is premultiplied by **opacity**
// (not coverage), height lives in the aux. Within a stroke the colour target
// over-blends (premult-consistent) and the aux target accumulates additively.
//
// Every deposit here is a function of the segment's **swept optical depth** `τ`
// alone, in one of exactly two forms — additive in `τ`, or `1 − exp(−k·τ)` — because
// those are the two that compose *independently of how the path was cut into
// segments*. `τ` itself sums exactly across overlapping quads (that is the whole
// point of the prefix-τ construction), so an amount linear in it sums to the true
// path integral, and a rate exponential in it multiplies out to `exp(−k·Στ)`. Any
// other shape — notably a `√`, which is what the opacity/thickness split used to
// apply per segment — makes the result depend on the *number* of segments: N pieces
// of the same travel deposit `Σ√(τ/N) = √(Nτ)`, so the stroke silently gains weight
// with sampling density. That is fine only while sampling is uniform; it is why
// adaptive flattening (`path::flatten`) needs this form, and it makes the fast path
// agree with the stamp loop's `add` axis, which was always linear in exposure
// (`dynamics.wesl::deposit`).
//
// The cost of dropping the `√` split: the footprint's coverage now reaches the
// media's visible alpha through *both* factors of `op · T`, so a soft tip's faint
// edge falls off as `τ²` rather than `τ`. Tip hardness absorbs that; sampling
// independence is not something a parameter can buy back.
//   target 0: color (Rgba16Float) — premultiplied Oklab (L,a,b)·op + opacity in a
//   target 1: aux   (Rgba16Float, the wide *scratch* layout — persistent tiles keep only
//             (height, wet)) — (height, wet, footprint coverage, unused)

import package::stamp_common::{VsOut, FsOut, sweep_vertex, swept_depth, brush_color};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) start: vec2<f32>,
    @location(1) dir: vec2<f32>,
    @location(2) geom: vec4<f32>,
    @location(3) extra: vec4<f32>,
) -> VsOut {
    return sweep_vertex(vi, start, dir, geom, extra);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    // The segment's swept optical depth — the footprint's geometric coverage
    // integrated along the travel. Everything below is built from this.
    let tau = swept_depth(in);
    let cov = 1.0 - exp(-tau);
    // Per-unit opacity enters as an *absorption rate* rather than a factor, so the
    // colour target's over-blend multiplies out to exactly `exp(−op · Στ)` — the
    // stroke's coverage is the same whether the path was cut into one segment or a
    // hundred.
    let op = 1.0 - exp(-in.op * tau);

    var out: FsOut;
    // Premultiplied by opacity: the colour target over-blends, so per-arc rates sum
    // through the overlapping segment quads to the local deposit. The colour is
    // per-fragment: colour dynamics jitter it across the footprint and along the
    // stroke (DESIGN.md §6.2).
    out.color = vec4<f32>(brush_color(in) * op, op);
    // The wide scratch aux (DESIGN §6.1):
    //   .x = laid height — linear in τ, so the additive target sums to the true path
    //        integral of the footprint and a second pass builds impasto in full
    //        (a saturating form would flatten out instead);
    //   .y = laid wet — always 0. The brush has no wetness knob, so nothing sources
    //        the gloss channel; the channel itself stays in the tile layout and in
    //        the media pass's roughness term (media_common.wesl), which is why this
    //        writes an explicit zero rather than the channel being removed.
    //   .z = footprint coverage (read by nothing today; kept for the write-back);
    //   .w = unused.
    // Height is the amount of paint; the colour alpha is per-unit opacity only and
    // never represents amount (§6.1).
    out.aux = vec4<f32>(in.amount * tau, 0.0, cov, 0.0);
    return out;
}
