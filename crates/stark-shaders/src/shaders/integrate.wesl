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
    // mode.x = the constant `load` (lift) axis; mode.y = ridge strength; mode.z unused;
    // mode.w selects the **per-pixel scrape** path. There is one unified merge (no
    // Normal/Dry branch): the lift removes `load × contact` of the base height **per-pixel**
    // (exact — the base is right here, so an eraser leaves no banded residue), the deposit
    // adds the mixer's height (scratch aux `.w`, coverage-normalized by `.z`), and ridge
    // redistributes paint laterally. With `load = 0` and a flat reservoir (deposit 0) it
    // reduces exactly to the old additive "over" deposit — height is the amount of paint;
    // the colour alpha is per-unit opacity only (DESIGN §6.1).
    //
    // mode.w = 0: `load` is constant — use mode.x (every existing brush; unchanged path).
    // mode.w = 1: pen pressure modulated the scrape — read the coverage-weighted per-pixel
    //             load from `scrape_tex` instead (the palette knife; DESIGN §6.2).
    mode: vec4<f32>,
}

// Strength of the ridge doublet at `ridge = 1` (the band's pile/scrape peaks ≈ this ×
// the local base height). Strong enough that the per-bristle corrugation reads, not just
// the round outer rim.
const RIDGE_GAIN: f32 = 3.0;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var base_color: texture_2d<f32>;
@group(0) @binding(2) var base_aux: texture_2d<f32>;
@group(0) @binding(3) var scratch_color: texture_2d<f32>;
@group(0) @binding(4) var scratch_aux: texture_2d<f32>;
// Per-pixel scrape: Σ(load·√cov) over the overlapping segment quads (the palette knife's
// pressure-modulated removal). Only read when `mode.w = 1`; a 1×1 dummy is bound otherwise.
@group(0) @binding(5) var scrape_tex: texture_2d<f32>;

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

    // ONE unified merge — paint moves by transferring **height** (the amount; DESIGN §6.1),
    // never by scaling alpha. The directions are split deliberately:
    //
    //   LIFT (`load`, mode.x) is **multiplicative and per-pixel**: remove `load × contact`
    //   of the base height, where `contact` saturates the exact swept coverage (`.z`). The
    //   base is right here, so the removal is exact — a `load = 1` eraser takes the core to
    //   zero with nothing left behind (a banded reconstruction across the brush width would
    //   leave residue an eraser could never clean up). Multiplicative can't overshoot, so no
    //   clamp absorbs error.
    //
    //   DEPOSIT is the mixer's banded approximation: `.w` (deposited height rate ≥ 0, laid
    //   through the soft footprint and accumulated over the overlapping quads) ÷ coverage
    //   recovers the per-column amount, × `contact` re-feathers the brush shape. Diffuse
    //   carried paint tolerates the banding that the subtraction didn't.
    //
    // With `load = 0` and a flat reservoir (`.w = 0`) this is exactly `out.aux = base + s_a`
    // and `out.color = s_c over base` — the old additive deposit. The tool's banded intake
    // and the exact removal agree in expectation, so with `add = 0` paint moves rather than
    // appearing/vanishing (approximately conserved; the source lightens because its
    // *thickness* drops, not its alpha).
    let cov = s_a.z;                 // accumulated footprint coverage (Σ√cov)
    let contact = 1.0 - exp(-cov);
    // The effective `load`: constant (mode.x) for every existing brush, or the per-pixel
    // coverage-weighted load Σ(load·√cov)/Σ√cov when pressure modulated the scrape (mode.w).
    // The latter reduces to the former when load is constant, so the only difference is that
    // it now varies along the stroke — still exact and residue-free per pixel (DESIGN §6.2).
    var load_eff = p.mode.x;
    if p.mode.w > 0.5 {
        load_eff = textureLoad(scrape_tex, c, 0).x / max(cov, 1e-4);
    }
    let lift_frac = clamp(load_eff * contact, 0.0, 1.0);
    let dep = (s_a.w / max(cov, 1e-4)) * contact;

    // Colour: lay the deposit (deposit·tool + add·brush) over the base (premultiplied
    // "over"). The lift does NOT touch colour — removing paint reduces height, and the
    // media's `1−exp(−opacity·thickness)` makes the source fade as its thickness drops.
    var out: FsOut;
    out.color = s_c + base_c * (1.0 - s_c.a);
    // Height = lift-scraped base + the brush's own add-height (`s_a.x`, impasto) + the
    // tool's deposit. Wet rides the own attribute (`s_a.y`).
    out.aux = vec4<f32>(base_a.x * (1.0 - lift_frac) + s_a.x + dep, base_a.y + s_a.y, 0.0, 0.0);

    // Lateral pile-up (`ridge`, mode.y): paint shoved aside heaps into an impasto lip at the
    // stroke rim. Exact mass conservation of a lateral move needs neighbour reads, which the
    // seam-free pass forbids (a tile's apron must be a pure function of canvas position,
    // §6.4). Instead a **doublet** across the soft transition band — pile on the outer side,
    // scrape just inside — is odd about the band's midpoint, so it is approximately zero-mean
    // over the footprint (bounded by the local paint, never the old unbounded additive lip):
    //   edge = 4·contact·(1−contact) localizes to the band; (1−2·contact) flips sign across
    //   it (+ at the low-coverage outer rim, − at the covered inner core).
    let edge = 4.0 * contact * (1.0 - contact);
    let doublet = edge * (1.0 - 2.0 * contact);
    out.aux.x = max(out.aux.x + p.mode.y * RIDGE_GAIN * base_a.x * doublet, 0.0);
    return out;
}
