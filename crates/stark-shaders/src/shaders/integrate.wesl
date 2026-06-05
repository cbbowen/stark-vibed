// Stroke *integrate* pass (DESIGN.md §6.2/§6.1). Every brush rasterizes its swept
// footprint into a *cleared scratch* tile (the stamp pass), then this fullscreen
// pass merges that scratch into a fresh copy-on-write tile over the base:
// `new = f(base, scratch)`, written to the color+aux MRT.
//
// It is a render pass (not compute) because the tile channels are `Rgba16Float` /
// `Rg16Float`, which are not core-WebGPU storage-texture formats. Everything is
// sampled with `textureLoad` (1:1 with the destination — no filtering), so the
// result is a deterministic function of canvas position and a tile's apron is
// rewritten identically to its neighbour's interior (the seam invariant, §6.4).
//
// In the normalized representation (§6.1): color = latent·opacity with opacity in
// `.a`, aux = (height, wet). The substrate composite is therefore the ordinary
// premultiplied-"over" on color (premultiplied by opacity) and an additive sum of
// height/wet — which is exactly what the old direct deposit did, now explicit so
// brush dynamics (knife, blend modes) can plug in as new `mode`s.

struct Params {
    // mode.x (0 = Normal, 1 = Knife), bite, load (film fraction), tooth·relief.
    mode: vec4<f32>,
    // tile texture origin in canvas px (.xy), inv surface-tile (.z), ridge (.w) —
    // .xyz feed the knife's tooth gate (sampling the surface in canvas space).
    surf: vec4<f32>,
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
    let s_c = textureLoad(scratch_color, c, 0); // stroke slab: latent·opacity, opacity in .a
    let s_a = textureLoad(scratch_aux, c, 0);    // stroke slab: (thickness, wet·opacity)

    var out: FsOut;
    if p.mode.x < 0.5 {
        // NORMAL — composite the stroke slab over the base: premultiplied "over" on
        // the opacity-premultiplied color, additive on thickness and wet.
        out.color = s_c + base_c * (1.0 - s_c.a);
        out.aux = base_a + s_a;
    } else {
        // KNIFE — scrape, then lay the knife's footprint as a film, then pile ridges.
        // The new representation makes the scrape trivially premult-consistent: scaling
        // the opacity-premultiplied color *and* the thickness by `(1−factor)` lightens
        // the paint with no dark fringe (the bug the old coverage-premult model had).
        let bite = p.mode.y;
        let load = p.mode.z;   // own-color film fraction (Rust forces 1 for carry)
        let tooth = p.mode.w;  // tooth·relief (0 on Flat → gate is a no-op)
        let canvas = p.surf.xy + frag.xy;
        let hs = textureSampleLevel(surface, surface_samp, canvas * p.surf.z, 0.0).r;
        let gate = 1.0 - tooth * (1.0 - hs);
        // The scratch opacity is the knife's contact (and, for carry, the dragged
        // amount it lays — they coincide, which is fine: a knife that has run out of
        // carried paint is also barely bearing down).
        let contact = s_c.a;
        let factor = clamp(bite * contact * gate, 0.0, 1.0);

        let scraped_c = base_c * (1.0 - factor);
        let scraped_a = base_a * (1.0 - factor);

        // Film: lay a `load` fraction of the footprint (own colour, or the dragged
        // paint the carry scan put in the scratch) over the scraped base.
        let film = s_c * load;
        out.color = film + scraped_c * (1.0 - film.a);
        out.aux = scraped_a + s_a * load;

        // Lateral pile-up: paint shoved aside heaps into the footprint's transition
        // band — `4·contact·(1−contact)` peaks at the soft edge, zero at the covered
        // core and outside — scaled by the base paint present. A pure function of the
        // local contact (no neighbour reads), so the tile apron stays seamless (§6.4).
        let edge = 4.0 * contact * (1.0 - contact);
        out.aux.x = out.aux.x + p.surf.w * edge * base_a.x;
    }
    return out;
}
