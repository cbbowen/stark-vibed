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
    // mode (0 = over / Dry-equivalent, 1 = knife), bite, load, tooth·relief.
    mode: vec4<f32>,
    // tile texture origin in canvas px (.xy), inv surface-tile (.z), carry flag (.w).
    // .xyz feed the knife's tooth gate (sampling the canvas surface in canvas
    // space); .w switches the knife film between its own loaded color (0) and the
    // dragged paint the carry scan deposited into the scratch (1).
    surf: vec4<f32>,
    // extra.x = ridge strength (lateral pile-up); yzw reserved.
    extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var base_color: texture_2d<f32>;
@group(0) @binding(2) var base_aux: texture_2d<f32>;
@group(0) @binding(3) var scratch_color: texture_2d<f32>;
@group(0) @binding(4) var scratch_aux: texture_2d<f32>;
@group(0) @binding(5) var surface: texture_2d<f32>; // canvas bump (height), canvas-space
@group(0) @binding(6) var surface_samp: sampler;

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
        // then lay a `load` fraction of the knife's own film over what remains. The
        // scrape is gated by the canvas surface (tooth·relief): the knife clears the
        // weave's peaks (hs→1, full bite) and leaves paint in its valleys (hs→0),
        // revealing the canvas texture. On `Flat` (relief 0) the gate is a no-op.
        let bite = p.mode.y;
        let load = p.mode.z;
        let tooth = p.mode.w;
        let canvas = p.surf.xy + frag.xy;
        let hs = textureSampleLevel(surface, surface_samp, canvas * p.surf.z, 0.0).r;
        let gate = 1.0 - tooth * (1.0 - hs);
        let factor = clamp(bite * s_c.a * gate, 0.0, 1.0);
        let col = base_c * (1.0 - factor);
        let aux = base_a * (1.0 - factor);

        if p.surf.w < 0.5 {
            // Own-color film: lay a `load` fraction of the footprint as the knife's
            // loaded color.
            let film = s_c * load; // premultiplied (loaded color × coverage × load)
            out.color = film + col * (1.0 - film.a);
            out.aux = aux + s_a * load;
        } else {
            // CARRY — the footprint scratch already holds the *dragged* paint, laid
            // by the carry scan + sweep: `s_c.xyz` is the film colour premultiplied by
            // (amount × coverage), `s_a.x` is that amount × coverage (its opacity),
            // and `s_c.a` is the contact coverage the scrape above used. Because the
            // amount is stored per lateral band, the knife's overhang past a paint
            // edge (empty bands) lays nothing — no black fringe. Re-lay the premult
            // film over the scraped base; the scrape lightened the source, so the
            // paint is conserved — lifted here, dropped downstream as the load fades.
            let film_a = clamp(s_a.x, 0.0, 1.0);
            out.color = vec4<f32>(s_c.xyz, film_a) + col * (1.0 - film_a);
            out.aux = aux + vec4<f32>(film_a, film_a, 0.0, 0.0);
        }

        // Lateral pile-up: paint shoved aside heaps into the footprint's soft
        // transition band — `4·cov·(1-cov)` peaks at coverage 0.5 (the edge), is
        // zero at the fully-covered core and outside — forming a raised lip along
        // the knife's path. It only rises where there is paint to push (scaled by
        // the base height) and is a pure function of the *local* coverage — no
        // neighbour reads — so a tile's apron stays bit-identical to the neighbour's
        // interior and the seam invariant holds (§6.4).
        let edge = 4.0 * s_c.a * (1.0 - s_c.a);
        out.aux.x = out.aux.x + p.extra.x * edge * base_a.x;
    }
    return out;
}
