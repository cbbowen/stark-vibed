// Region → tile write-back for the brush-dynamics stamp loop (§6.2).
//
// After the stamp loop evolves the stroke's canvas region, each affected tile is
// sliced back out as a fresh copy-on-write tile: a fullscreen pass over the tile's
// full TILE_TEX target (interior + apron) that `textureLoad`s the region 1:1 and
// narrows the wide region aux to the persistent single (height). Because every
// rewritten tile's block is cut from the *same* region texture, a tile's apron is
// bit-identical to its neighbour's interior over their overlap — the seam
// invariant (§6.4) holds by construction.

import package::lib::quad::{fullscreen_clip};

struct Params {
    // .xy = the tile texture's top-left, in region texels (integral). .zw unused.
    offset: vec4<f32>,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var region_color: texture_2d<f32>;
@group(0) @binding(2) var region_aux: texture_2d<f32>;
// The region's residual channel (§6.7), sliced out 1:1 like the colour it
// belongs to.
@if(resid) @group(0) @binding(3) var region_resid: texture_2d<f32>;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @if(resid) @location(2) resid: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy)) + vec2<i32>(p.offset.xy);
    var out: FsOut;
    out.color = textureLoad(region_color, c, 0);
    // The persistent aux is one channel (height); the extra components are dropped
    // by the single-component target.
    out.aux = vec4<f32>(textureLoad(region_aux, c, 0).x, 0.0, 0.0, 0.0);
    @if(resid) out.resid = textureLoad(region_resid, c, 0);
    return out;
}
