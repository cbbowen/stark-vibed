// Selection mask → stroke region (§6.8, §6.2).
//
// The brush-dynamics stamp loop works over one 1:1 canvas *region* rather than
// per-tile targets, so the selection mask has to be gathered into a region-sized R8
// texture the loop can read alongside the colour and aux regions. The region is
// tile-aligned by construction, so each mask tile's full `TILE_TEX` block lands at an
// integral offset and is copied 1:1 with `textureLoad` — no filtering, no drift.
//
// Tiles the selection has no mask for are simply not drawn: the pass clears to the
// coverage that reigns outside the mask's tile set, so the "everything outside is
// deselected" (or, when inverted, "selected") case costs nothing.

struct Region {
    // .xy = region size in px, .z = TILE_TEX, .w unused.
    a: vec4<f32>,
}

@group(0) @binding(0) var<uniform> r: Region;
@group(1) @binding(0) var mask: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // The tile texture's top-left in region px. Constant across the quad.
    @location(0) @interpolate(flat) origin: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    // The tile texture's top-left in region px.
    @location(0) origin: vec2<f32>,
) -> VsOut {
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let px = origin + corner * r.a.z;
    var out: VsOut;
    out.clip = vec4<f32>(px.x / r.a.x * 2.0 - 1.0, 1.0 - px.y / r.a.y * 2.0, 0.0, 1.0);
    out.origin = origin;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = vec2<i32>(floor(in.clip.xy) - in.origin);
    return vec4<f32>(textureLoad(mask, c, 0).r, 0.0, 0.0, 0.0);
}
