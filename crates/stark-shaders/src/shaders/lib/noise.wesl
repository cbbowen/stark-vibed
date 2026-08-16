// The color-dynamics noise lookup (§6.2).
//
// **The GPU does not generate noise, it reads it.** Every field is baked once on
// the CPU into a small tileable `Rgba8Snorm` texture (`stark-core`'s `noise.rs`),
// because that is what keeps it bit-identical across GPUs, runs and peers — the
// same determinism contract as the sRGB↔Oklab constants (§6.5). So the whole of
// this module is one texture sample and a scale.
//
// There is deliberately no GPU hash family here — no `pcg2d`/`pcg3d`/`pcg4d`, no
// `noise_N_to_M` wrappers. A hash the GPU never evaluates cannot change a pixel,
// which makes it exactly the scaffolding this project declines to carry; it belongs
// here only alongside the pass that needs it. The CPU `pcg4d` in `noise.rs` is a
// different thing: it bakes the textures this samples, and has no GPU counterpart to
// agree with.

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
