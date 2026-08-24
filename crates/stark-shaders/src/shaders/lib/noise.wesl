// GPU noise, two tiers with two jobs (§6.2, §6.5).
//
// **Artistic noise is baked, quantization noise is hashed.** The color-dynamics
// field wants smooth, tileable spatial structure, so it is baked once on the CPU
// into a small `Rgba8Snorm` texture (`stark-engine`'s `noise.rs`) and the GPU only
// samples it — that is what keeps a *pattern* bit-identical across GPUs, runs and
// peers, the same determinism contract as the sRGB↔Oklab constants (§6.5).
// Quantization dither wants the opposite — white noise, one fresh value per
// target pixel — and a baked texture of it would be this hash with a footprint.
// The hash holds the same contract on its own terms: integer arithmetic is exact
// on every backend, and the one float step below is a mantissa placement and a
// Sterbenz-exact subtraction, so no operation ever rounds.
//
// There is deliberately nothing else here — no `noise_N_to_M` wrapper family. A
// hash the GPU never evaluates cannot change a pixel, which makes it exactly the
// scaffolding this project declines to carry; a new shape joins this file
// alongside the pass that first reads it. The CPU `pcg4d` in `noise.rs` is a
// different thing: it bakes the textures `color_jitter` samples, and has no GPU
// counterpart to agree with.

// Per-channel color offset at a **stroke-local** position: sample the baked
// tileable 2-D noise field (signed, ≈[-1,1] per channel) and scale each channel by
// its amplitude.
//   local: (lateral offset from the stroke centreline, arc length along the
//          stroke), both in canvas px — the brush's own frame, so the pattern
//          rides the stroke instead of the canvas underneath it. Both are
//          functions of the fragment's canvas position and the segment, so the
//          deposit stays a pure function of the two and tile aprons remain
//          bit-consistent (§6.4).
//   freq:  per-axis frequency .xy, with 1/NOISE_TILE_PX pre-folded into .z.
//          A `vec3` because that is what the lookup is: the fourth lane the two
//          callers used to spend on it was never read.
//   amp:   per color-channel amplitude (0 = that channel holds still).
//   off:   the per-stroke lookup translation (derived from the stroke seed).
// The sampler must wrap (repeat) on both axes.
fn color_jitter(
    tex: texture_2d<f32>,
    s: sampler,
    local: vec2<f32>,
    freq: vec3<f32>,
    amp: vec3<f32>,
    off: vec2<f32>,
) -> vec3<f32> {
    let q = local * freq.xy * freq.z + off;
    return amp * textureSampleLevel(tex, s, q, 0.0).xyz;
}

// Core integer hash — Jarzynski & Olano, "Hash Functions for GPU Rendering"
// (JCGT 2020). Square 3×3: take the first M output lanes for M outputs.
fn _pcg3d(v_in: vec3u) -> vec3u {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// Two centered uniform lanes at target pixel `px` — quantization dither for a
// store that walks f32 down to a shallow encoding (§6.5). Scaled by one code
// step of the target's encoding at the call site, it makes the store's rounding
// stochastic: a value between two codes splits between them in exact proportion,
// so a ramp too gentle for the codes renders as mixture instead of bands.
//
// The swing is **half a code either way, and strictly inside it** — not the
// audio-style full-code triangle. Half, so a value the encode lands exactly on a
// code — black, white, anything the tonemap clamped — renders that code at every
// pixel rather than speckling its neighbours. Strictly inside, by construction:
// the hash's top 23 bits are placed as a [1,2) mantissa with the low bit forced,
// so every lane is an odd multiple of 2⁻²³ — a symmetric, zero-mean set that
// cannot produce ±0.5 itself, which is what would land a code-point value on the
// rounding tie between two codes. Both float steps are exact (the placement does
// no arithmetic; subtracting 1.5 from [1,2) is exact by Sterbenz), so the value
// is the same on every backend.
//
// Keyed on the integer pixel rather than a float position so no derived
// quantity's ULPs can re-roll the noise; `seed` separates passes writing the
// same coordinates.
fn dither2(px: vec2u, seed: u32) -> vec2f {
    let h = _pcg3d(vec3u(px, seed));
    let m = (h.xy >> vec2u(9u)) | vec2u(0x3f800001u);
    return bitcast<vec2f>(m) - 1.5;
}
