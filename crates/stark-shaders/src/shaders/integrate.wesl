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
    // mode.x (0 = Normal, 1 = Dry); mode.y = the `lift` axis. The Dry branch removes
    // `lift × contact` of the base height **per-pixel** (exact — the base is right here,
    // so an eraser leaves no banded residue) and adds the mixer's deposited height
    // (scratch aux `.w`, coverage-normalized by `.z`) — height is the amount of paint;
    // the colour alpha is per-unit opacity only (DESIGN §6.1).
    mode: vec4<f32>,
    // ridge (lateral pile-up strength) in .x; .yzw unused.
    ridge: vec4<f32>,
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
    let s_c = textureLoad(scratch_color, c, 0); // stroke slab: latent·opacity, opacity in .a
    // Wide scratch aux: (own add-height, own wet·op, footprint coverage Σ√cov, net height
    // transfer·√cov) — see stamp_oklab.wesl's fs_main for the channel-by-channel story.
    let s_a = textureLoad(scratch_aux, c, 0);

    var out: FsOut;
    if p.mode.x < 0.5 {
        // NORMAL — composite the stroke slab over the base: premultiplied "over" on
        // the opacity-premultiplied color, additive on thickness and wet.
        out.color = s_c + base_c * (1.0 - s_c.a);
        out.aux = base_a + s_a;
    } else {
        // DRY — paint is moved by transferring **height** (the amount; DESIGN §6.1), not by
        // scaling alpha. The two directions are split deliberately:
        //
        //   LIFT is **multiplicative and per-pixel**: remove `lift × contact` of the base
        //   height, where `contact` saturates the exact swept coverage (`.z`). The base is
        //   right here, so the removal is exact — a `lift = 1` eraser takes the core to
        //   zero with nothing left behind (the old reconstruction from the mixer's 16
        //   lateral bands smeared point samples of the base height across the brush width,
        //   leaving residue an eraser could never clean up). Multiplicative also can't
        //   overshoot, so no clamp absorbs error.
        //
        //   DEPOSIT stays the mixer's banded approximation: `.w` (deposited height rate ≥ 0,
        //   laid through the soft footprint and accumulated over the overlapping quads) ÷
        //   coverage recovers the per-column amount, × `contact` re-feathers the brush
        //   shape. Diffuse carried paint tolerates the banding that the subtraction didn't.
        //
        // The tool's banded intake and the exact removal agree in expectation, so with
        // `add = 0` paint still moves rather than appearing/vanishing (approximately
        // conserved; the source lightens because its *thickness* drops, not its alpha).
        let cov = s_a.z;                 // accumulated footprint coverage (Σ√cov)
        let contact = 1.0 - exp(-cov);
        let lift_frac = clamp(p.mode.y * contact, 0.0, 1.0);
        let dep = (s_a.w / max(cov, 1e-4)) * contact;

        // Colour: lay the deposit (deposit·tool + add·brush) over the base (premultiplied
        // "over"). The lift does NOT touch colour — removing paint reduces height, and the
        // media's `1−exp(−opacity·thickness)` makes the source fade as its thickness drops.
        out.color = s_c + base_c * (1.0 - s_c.a);
        // Height = lift-scraped base + the brush's own add-height (`s_a.x`, impasto) + the
        // tool's deposit. Wet rides the own attribute (`s_a.y`).
        out.aux = vec4<f32>(base_a.x * (1.0 - lift_frac) + s_a.x + dep, base_a.y + s_a.y, 0.0, 0.0);

        // Lateral pile-up: paint shoved aside heaps into the footprint's transition band —
        // `4·contact·(1−contact)` peaks at the soft edge, zero at the covered core and
        // outside — scaled by the base paint present. A pure function of the local coverage
        // (no neighbour reads), so the tile apron stays seamless (§6.4).
        let edge = 4.0 * contact * (1.0 - contact);
        out.aux.x = out.aux.x + p.ridge.x * edge * base_a.x;
    }
    return out;
}
