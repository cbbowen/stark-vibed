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

// A centered uniform lane from one hash word: odd multiples of 2⁻²³ in
// (−0.5, 0.5) — symmetric and so zero-mean *exactly*, with both endpoints
// unreachable by construction: the word's top 23 bits are placed as a [1,2)
// mantissa with the low bit forced. Both float steps are exact (the placement
// does no arithmetic; subtracting 1.5 from [1,2) is exact by Sterbenz), so the
// value is the same on every backend.
fn _centered(h: u32) -> f32 {
    return bitcast<f32>((h >> 9u) | 0x3f800001u) - 1.5;
}

// Two centered uniform lanes at target pixel `px` — quantization dither for a
// store that walks f32 down to a shallow encoding (§6.5). Scaled by one code
// step of the target's encoding at the call site, it makes the store's rounding
// stochastic: a value between two codes splits between them in exact proportion,
// so a ramp too gentle for the codes renders as mixture instead of bands.
//
// The swing is **half a code either way, and strictly inside it** (`_centered`)
// — not the audio-style full-code triangle. Half, so a value the encode lands
// exactly on a code — black, white, anything the tonemap clamped — renders that
// code at every pixel rather than speckling its neighbours; strictly inside, so
// a code-point value can never land on the rounding tie between two codes.
//
// Keyed on the integer pixel rather than a float position so no derived
// quantity's ULPs can re-roll the noise; `seed` separates passes writing the
// same coordinates.
fn dither2(px: vec2u, seed: u32) -> vec2f {
    let h = _pcg3d(vec3u(px, seed));
    return vec2f(_centered(h.x), _centered(h.y));
}

// The deposit jitter (§6.2): the multiplicative gate a stroke's paint takes at
// canvas texel `texel` — `color_jitter`'s sibling, perturbing the *amount* where
// that perturbs the color. Uniform in `(1 − eps, 1 + eps)`, mean exactly 1 (the
// `_centered` set is symmetric), and exactly 1.0 when `eps` is 0, which is what
// keeps a jitter-free stroke bit-identical.
//
// It multiplies the swept **exposure**, beside the substrate tooth and under the
// tooth's own two laws (§6.2, §6.4): a factor on the exposure keeps the deposit's
// two composing forms intact where a factor on the finished parcel would not, and
// the tool's side of the exchange books the gate's *mean* — for the tooth the
// bearing fraction, here exactly 1, so the tool's dispatch does not change at all
// and the books balance in expectation over the extent, by the same argument.
//
// Keyed on the **absolute canvas texel** and a per-stroke seed, so the gate is
// one field however the stroke is cut, folded or replayed: constant along the
// stroke, it factors out of every per-segment sum (§6.4, live == committed), and
// per-stroke seeding is what keeps repeated glazes from compounding one pattern.
// White per texel rather than the baked field's smooth structure, because its job
// is the field's opposite: decorrelating neighbouring texels' accumulation, so
// what the iterative exchange would pile into a visible band lands as dither.
fn deposit_jitter(texel: vec2<i32>, seed: u32, eps: f32) -> f32 {
    if eps <= 0.0 {
        return 1.0;
    }
    let h = _pcg3d(vec3u(bitcast<vec2u>(texel), seed));
    return 1.0 + 2.0 * eps * _centered(h.x);
}
