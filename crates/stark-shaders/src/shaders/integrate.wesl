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
    // mode.x (0 = Normal, 1 = Dry); the other components are unused. The Dry branch conserves
    // paint by adding the mixer's per-pixel **net height transfer** (deposited − lifted,
    // carried in the scratch aux's `.w`, coverage-normalized by `.z`) to the base — height is
    // the amount of paint; the colour alpha is per-unit opacity only (DESIGN §6.1).
    mode: vec4<f32>,
    // tile texture origin in canvas px (.xy), inv surface-tile (.z), ridge (.w) —
    // .xyz feed the Dry scrape's tooth gate (sampling the surface in canvas space).
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
        // DRY — paint is moved by transferring **height** (the amount; DESIGN §6.1), not by
        // scaling alpha. The mixer emitted, per column, the conserved net height transfer
        // (deposited − lifted) into the scratch aux's `.w`, laid additively across the
        // overlapping segment quads; `.z` is the summed footprint coverage. Dividing recovers
        // the per-pixel transfer at the canvas-height scale, which we add to the base. With
        // `add=0` what's deposited here was lifted elsewhere → total canvas+tool height is
        // conserved (the source lightens because its *thickness* drops, not its alpha).
        let cov = s_a.z;                 // accumulated footprint coverage (Σ√cov)
        // The brush's soft footprint coverage at this pixel ∈ [0,1] (1 under the core, →0 at
        // the feathered edge). The net transfer was laid through the soft footprint then
        // accumulated across the overlapping quads, so `s_a.w / cov` recovers the per-column
        // amount but *divides out the soft edge* — leaving a hard cutoff at the footprint
        // boundary. Re-multiplying by `contact` restores the brush shape: the lift/deposit
        // feathers with the tip instead of stopping abruptly (it still ≈ the per-column amount
        // under the core where contact ≈ 1, so conservation holds).
        let contact = 1.0 - exp(-cov);
        let net = (s_a.w / max(cov, 1e-4)) * contact; // conserved transfer, brush-feathered

        // Colour: lay the deposit (deposit·tool + add·brush) over the base (premultiplied
        // "over"). The lift does NOT touch colour — removing paint reduces height, and the
        // media's `1−exp(−opacity·thickness)` makes the source fade as its thickness drops.
        out.color = s_c + base_c * (1.0 - s_c.a);
        // Height = base + the brush's own add-height (`s_a.x`, impasto) + the conserved net
        // transfer. Wet rides the own attribute (`s_a.y`).
        out.aux = vec4<f32>(max(base_a.x + s_a.x + net, 0.0), base_a.y + s_a.y, 0.0, 0.0);

        // Lateral pile-up: paint shoved aside heaps into the footprint's transition band —
        // `4·contact·(1−contact)` peaks at the soft edge, zero at the covered core and
        // outside — scaled by the base paint present. A pure function of the local coverage
        // (no neighbour reads), so the tile apron stays seamless (§6.4).
        let edge = 4.0 * contact * (1.0 - contact);
        out.aux.x = out.aux.x + p.surf.w * edge * base_a.x;
    }
    return out;
}
