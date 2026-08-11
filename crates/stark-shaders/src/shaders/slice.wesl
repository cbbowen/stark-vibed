// Region-aux narrowing for the stamp loop's write-back (§6.2).
//
// After the loop evolves the stroke's canvas region, each affected tile is written
// back as a fresh copy-on-write tile. Colour and residual leave the region as plain
// texture copies — their tile formats are the region's own, and a copy is bit-exact
// — so the one channel that changes representation is the aux: the wide region aux
// narrows to the persistent single channel (height). This pass does that narrowing
// once over the whole region; the per-tile copies then cut their TILE_TEX blocks
// out of its result exactly as they cut colour out of the region itself. Because
// every rewritten tile's block is cut from the same region-sized textures, a tile's
// apron is bit-identical to its neighbour's interior over their overlap — the seam
// invariant (§6.4) holds by construction.
//
// Rounding: what lands here is a `textureLoad` of an f16 texel, so the render-target
// write returns it to the f16 lattice point it already sits on — bit-identical to
// the per-tile slice draws this pass replaced, which wrote the same loads through
// the same target format, one render pass per tile.

import package::lib::quad::{fullscreen_clip};

@group(0) @binding(0) var region_aux: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // The persistent aux is one channel (height); the extra components are dropped
    // by the single-component target.
    let a = textureLoad(region_aux, vec2<i32>(floor(frag.xy)), 0);
    return vec4<f32>(a.x, 0.0, 0.0, 0.0);
}
