// Region → tile write-back for the brush-dynamics stamp loop (DESIGN.md §6.2).
//
// After the stamp loop evolves the stroke's canvas region, each affected tile is
// sliced back out as a fresh copy-on-write tile: a fullscreen pass over the tile's
// full TILE_TEX target (interior + apron) that `textureLoad`s the region 1:1 and
// narrows the wide region aux to the persistent (height, wet). Because every
// rewritten tile's block is cut from the *same* region texture, a tile's apron is
// bit-identical to its neighbour's interior over their overlap — the seam
// invariant (§6.4) holds by construction.

struct Params {
    // .xy = the tile texture's top-left, in region texels (integral). .zw unused.
    offset: vec4<f32>,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var region_color: texture_2d<f32>;
@group(0) @binding(2) var region_aux: texture_2d<f32>;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy)) + vec2<i32>(p.offset.xy);
    var out: FsOut;
    out.color = textureLoad(region_color, c, 0);
    let a = textureLoad(region_aux, c, 0);
    out.aux = vec4<f32>(a.x, a.y, 0.0, 0.0);
    return out;
}
