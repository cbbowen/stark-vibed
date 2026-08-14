// The color-dynamics noise lookup (§6.2).
//
// **The GPU does not generate noise, it reads it.** Every field is baked once on
// the CPU into a small tileable `Rgba8Snorm` texture (`stark-core`'s `noise.rs`),
// because that is what keeps it bit-identical across GPUs, runs and peers — the
// same determinism contract as the sRGB↔Oklab constants (§6.5). So the whole of
// this module is one texture sample and a scale.
//
// It used to carry a hash library alongside: `_pcg2d`/`_pcg3d`/`_pcg4d` (Jarzynski
// & Olano, JCGT 2020), a `_unit` bit-twiddle, and seven `noise_N_to_M` wrappers
// over them — 85 of the file's 111 lines, with **no caller anywhere in the tree**.
// `media_common` imported one of them and never used it, which was the only
// reference the family had. A hash the GPU never evaluates cannot change a pixel,
// which makes it exactly the scaffolding this project declines to carry: it went
// the way of `tooth`, `drag` and `wetness`, and comes back with the pass that
// needs it. The live CPU `pcg4d` in `noise.rs` is untouched — it bakes the
// textures this samples, and never had a GPU counterpart to agree with.

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
//   amp:   per color-channel amplitude (0 = that channel holds still).
//   off:   the per-stroke lookup translation (derived from the stroke seed).
// The sampler must wrap (repeat) on both axes.
fn color_jitter(
    tex: texture_2d<f32>,
    s: sampler,
    local: vec2<f32>,
    freq: vec4<f32>,
    amp: vec3<f32>,
    off: vec2<f32>,
) -> vec3<f32> {
    let q = local * freq.xy * freq.z + off;
    return amp * textureSampleLevel(tex, s, q, 0.0).xyz;
}
