// The translucent-slab law, **both ways** (§6.1): a stored tile expanded into what it
// composites to, and a composite stored back as a tile.
//
// Two entry points and one binding layout, because they are one law read in two
// directions. `fs_expand` is `composite.wesl`'s fragment stage with the view taken out
// — the same `1 − exp(−K·opacity·height)`, the same re-premultiply by coverage — and
// `fs_store` inverts it.
//
// # What this is for
//
// Merging a layer down through a **blend mode** (§14.11). The blend algebra is defined
// on *composited* channels — coverage in alpha — while a tile carries per-unit opacity
// (§6.1), so the two cannot meet without a conversion. These are that conversion, and
// having them lets a merge run the **real blend pass**, on tile-sized targets, rather
// than a second copy of its algebra: expand both layers, run
// `blend_oklab`/`blend_mixbox` unchanged, store the result. A merged layer therefore
// cannot drift from the stack it replaced, which is the same argument the eyedropper
// makes for sampling through the compositor rather than beside it (§18.0.2).
//
// # Why the inverse exists at all
//
// A tile's `(opacity, height)` pair has exactly the freedom to name any
// `(coverage, height)` a stack can reach, and the mapping is one-to-one given the
// height: coverage fixes the **optical mass** `M = −ln(1 − w)/K`, and the opacity is
// `M/H`. Nothing is fitted; the pair is what §6.1 conserves and the mass is what adds.
//
// `M ≤ H`, so the stored opacity always lands in `[0,1]` — not by a clamp but by
// construction. Every merge's alpha is `1 − (1−wb)(1−ws)`, so its mass is `Mb + Ms`,
// and each side's own mass is at most its own height (that is what an opacity in
// `[0,1]` means). A clipped merge takes the backdrop's coverage alone, whose mass is
// at most the backdrop's height, which is at most the sum. There is no case left.
//
// Deliberately **not** run for a `Normal` merge, which `merge.wesl` settles in tile
// space directly: that path needs no colour conversion, keeps its exact
// passthrough branches, and is what an ordinary merge-down takes.

import package::lib::paint_common::{COVERAGE_EPS, OPACITY_K, PAINT_EPS};
import package::lib::quad::{fullscreen_clip};

struct Slab {
    // .x = the layer opacity to fold in, read by `fs_expand` alone — pass A scales a
    // leaf layer's tiles by it, so an expansion that ignored it would describe a
    // layer at full strength. `fs_store` writes tiles at opacity 1 and leaves the
    // slider to the surviving layer, so it reads nothing here. .yzw unused.
    p: vec4<f32>,
}

@group(0) @binding(0) var<uniform> s: Slab;
@group(0) @binding(1) var in_color: texture_2d<f32>;
@group(0) @binding(2) var in_aux: texture_2d<f32>;
// The residual half of the colour (§6.7), carried through both directions on the
// **colour's** alpha — the residual target's own is a duplicate, kept only because the
// fixed-function "over" that fills these reads each target's own.
@if(resid) @group(0) @binding(3) var in_resid: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @if(resid) @location(2) resid: vec4<f32>,
}

// A stored tile → what pass A would put in the accumulator for it, alone.
//
// Line for line `composite.wesl::fs_main` with the vertex plumbing removed: the same
// slab weight, the same scale by the layer's opacity, the same re-premultiply so the
// alpha carries coverage. It is a separate shader only because that one samples
// through the view transform and this one reads a tile 1:1 — including its **apron**,
// which is what keeps the merged tile's edge bit-identical to its neighbour's interior
// with no copy pass (§6.4).
@fragment
fn fs_expand(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));
    let c = textureLoad(in_color, p, 0);
    let h = textureLoad(in_aux, p, 0).x;
    let opl = s.p.x;

    // Coverage: per-unit opacity and thickness through the slab law, faded by the
    // layer's own slider — which scales the finished weight, not the mass.
    let w = (1.0 - exp(-OPACITY_K * c.a * h)) * opl;
    let latent = c.rgb / max(c.a, PAINT_EPS);

    var out: FsOut;
    out.color = vec4<f32>(latent * w, w);
    out.aux = vec4<f32>(h * opl, 0.0, 0.0, 0.0);
    @if(resid) let r = textureLoad(in_resid, p, 0);
    @if(resid) out.resid = vec4<f32>(r.rgb / max(c.a, PAINT_EPS) * w, w);
    return out;
}

// A composited pair → the tile that composites to it, at opacity 1.
//
// The inverse of the above, and the only place this engine reads a coverage back as a
// mass. The guard under `1 − w` is what keeps a fully-covered texel finite: coverage
// approaches 1 asymptotically and never reaches it, so a texel that rounds there came
// from paint thick enough that the capped mass is still far below its own height —
// which is why the cap cannot push the stored opacity out of range (see the header).
@fragment
fn fs_store(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));
    let c = textureLoad(in_color, p, 0);
    let height = textureLoad(in_aux, p, 0).x;
    let w = c.a;

    let mass = -log(max(1.0 - w, PAINT_EPS)) / OPACITY_K;
    let opacity = mass / max(height, PAINT_EPS);
    let latent = c.rgb / max(w, COVERAGE_EPS);

    var out: FsOut;
    out.color = vec4<f32>(latent * opacity, opacity);
    out.aux = vec4<f32>(height, 0.0, 0.0, 0.0);
    // The residual un-premultiplies by the **colour's** coverage and rides the same
    // opacity: it is the rest of the same colour, not a second one (§6.7).
    @if(resid) let r = textureLoad(in_resid, p, 0);
    @if(resid) out.resid = vec4<f32>(r.rgb / max(w, COVERAGE_EPS) * opacity, opacity);
    return out;
}
