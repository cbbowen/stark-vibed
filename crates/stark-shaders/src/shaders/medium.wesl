// Mutable-medium combine pass (DESIGN.md §6.2, "Mutable medium"). A *medium* brush
// replaces the additive/over deposit with a read-modify-write: rasterize the
// stroke's footprint into a cleared *scratch* tile, then this fullscreen pass reads
// the base tile + the scratch and writes `new = f(base, scratch)` into a fresh
// copy-on-write tile's color+aux MRT.
//
// It's a render pass (not compute) because `aux` is `Rg16Float`, which is not a
// storage-texture format. Everything is sampled with `textureLoad` (1:1 with the
// destination — no filtering), so the result is a deterministic function of canvas
// position and a tile's apron is rewritten identically to its neighbour's interior
// (the same property that lets the additive deposit render aprons; §6.4).

struct Params {
    // mode (0 = over / Dry-equivalent, 1 = knife), bite, load, _unused.
    mode: vec4<f32>,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var base_color: texture_2d<f32>;
@group(0) @binding(2) var base_aux: texture_2d<f32>;
@group(0) @binding(3) var scratch_color: texture_2d<f32>;
@group(0) @binding(4) var scratch_aux: texture_2d<f32>;

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
    let c = vec2<i32>(floor(frag.xy));
    let base_c = textureLoad(base_color, c, 0);
    let base_a = textureLoad(base_aux, c, 0);
    let s_c = textureLoad(scratch_color, c, 0); // footprint: premult color, coverage in .a
    let s_a = textureLoad(scratch_aux, c, 0);    // footprint: (height, wet)·coverage

    var out: FsOut;
    if p.mode.x < 0.5 {
        // OVER — reproduces the additive/over deposit exactly (premultiplied "over"
        // for color, additive for aux). The Phase-0 equivalence check.
        out.color = s_c + base_c * (1.0 - s_c.a);
        out.aux = base_a + s_a;
    } else {
        // KNIFE — scrape paint off in proportion to `bite` × footprint coverage,
        // then lay a `load` fraction of the knife's own film over what remains.
        let bite = p.mode.y;
        let load = p.mode.z;
        let factor = clamp(bite * s_c.a, 0.0, 1.0);
        var col = base_c * (1.0 - factor);
        var aux = base_a * (1.0 - factor);
        let film = s_c * load; // premultiplied film (loaded color × coverage × load)
        out.color = film + col * (1.0 - film.a);
        out.aux = aux + s_a * load;
    }
    return out;
}
