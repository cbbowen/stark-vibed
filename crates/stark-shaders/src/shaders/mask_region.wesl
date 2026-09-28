// Selection mask → stroke region (§6.8, §6.2).
//
// The brush-dynamics stamp loop works over one 1:1 canvas *region* rather than
// per-tile targets, so the selection mask has to be gathered into a region-sized R8
// texture the loop can read alongside the color and aux regions. The region is
// tile-aligned by construction, so each mask tile's full `TILE_TEX` block lands at an
// integral offset and is copied 1:1 with `textureLoad` — no filtering, no drift.
//
// Tiles the selection has no mask for are simply not drawn: the pass clears to the
// coverage that reigns outside the mask's tile set, so the "everything outside is
// deselected" (or, when inverted, "selected") case costs nothing.
//
// The coverage alone: the mask's overall opacity (§6.8) reaches the loop inside its
// ceiling (`Stamp::opacity`), not through this texture — a copy that carried it too
// would gate the plane at its square.

import package::lib::quad::{quad_corner};

struct Region {
    // The region's size in px.
    //
    // `@align(16)` is the block's alignment rather than this member's: WGSL requires a
    // uniform struct to be 16-byte aligned and has no struct-level attribute for it, so
    // the first member says so — and the block keeps its whole 16-byte row, the fourth
    // lane spare (§6.10).
    @align(16) size: vec2<f32>,
    // One mask tile's texture edge, `TILE_TEX` — the block each instance copies.
    tile_tex: f32,
}

@group(0) @binding(0) var<uniform> r: Region;
@group(1) @binding(0) var mask: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // The tile texture's top-left in region px. Constant across the quad.
    @location(0) @interpolate(flat) origin: vec2<f32>,
}

// One mask tile.
struct MaskInstance {
    // The tile texture's top-left in region px.
    @location(0) origin: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: MaskInstance) -> VsOut {
    let px = inst.origin + quad_corner(vi) * r.tile_tex;
    var out: VsOut;
    out.clip = vec4<f32>(px.x / r.size.x * 2.0 - 1.0, 1.0 - px.y / r.size.y * 2.0, 0.0, 1.0);
    out.origin = inst.origin;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = vec2<i32>(floor(in.clip.xy) - in.origin);
    return vec4<f32>(textureLoad(mask, c, 0).r, 0.0, 0.0, 0.0);
}
