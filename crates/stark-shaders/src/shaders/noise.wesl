// Core integer hashes — Jarzynski & Olano, "Hash Functions for GPU Rendering" (JCGT 2020).
// Square NxN: take the first M output lanes for M outputs.

fn _pcg2d(v_in: vec2u) -> vec2u {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> vec2u(16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> vec2u(16u);
    return v;
}

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

fn _pcg4d(v_in: vec4u) -> vec4u {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> vec4u(16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

// u32 -> uniform f32 in [0, 1). 
fn _unit(h: u32) -> f32 {
    // Builds a float in [1, 2) from the top 23 bits, then shifts down. Guaranteed half-open, exactly 2^23 levels, no f32 rounding bias.
    // return bitcast<f32>((h >> 9u) | 0x3f800000u) - 1.0;

    // Can produce denormalized values and exactly 1.0.
    return f32(h) * 2.3283064365386963e-10;  // h / 2^32
}

// --- the family, each parameterized by a u32 seed ---

fn noise_1_to_1(p: f32, seed: u32) -> f32 {
    let h = _pcg2d(vec2u(bitcast<u32>(p), seed));
    return _unit(h.x);
}

fn noise_2_to_1(p: vec2f, seed: u32) -> f32 {
    let h = _pcg3d(vec3u(bitcast<u32>(p.x), bitcast<u32>(p.y), seed));
    return _unit(h.x);
}

fn noise_2_to_2(p: vec2f, seed: u32) -> vec2f {
    let h = _pcg3d(vec3u(bitcast<u32>(p.x), bitcast<u32>(p.y), seed));
    return vec2f(_unit(h.x), _unit(h.y));
}

fn noise_2_to_3(p: vec2f, seed: u32) -> vec3f {
    let h = _pcg3d(vec3u(bitcast<u32>(p.x), bitcast<u32>(p.y), seed));
    return vec3f(_unit(h.x), _unit(h.y), _unit(h.z));
}

fn noise_2_to_4(p: vec2f, seed: u32) -> vec4f {
    let h = _pcg4d(vec4u(bitcast<u32>(p.x), bitcast<u32>(p.y), seed, 0x9e3779b9u));
    return vec4f(_unit(h.x), _unit(h.y), _unit(h.z), _unit(h.w));
}

fn noise_3_to_3(p: vec3f, seed: u32) -> vec3f {
    let h = _pcg4d(vec4u(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z), seed));
    return vec3f(_unit(h.x), _unit(h.y), _unit(h.z));
}

fn noise_3_to_4(p: vec3f, seed: u32) -> vec4f {
    let h = _pcg4d(vec4u(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z), seed));
    return vec4f(_unit(h.x), _unit(h.y), _unit(h.z), _unit(h.w));
}

// --- colour dynamics (DESIGN.md §6.2) ---------------------------------------------
// Per-channel colour offset at a **stroke-local** position: sample the baked
// tileable 2-D noise field (signed, ≈[-1,1] per channel; built on the CPU in
// stark-core's noise.rs) and scale each channel by its amplitude.
//   local: (lateral offset from the stroke centreline, arc length along the
//          stroke), both in canvas px — the brush's own frame, so the pattern
//          rides the stroke instead of the canvas underneath it. Both are
//          functions of the fragment's canvas position and the segment, so the
//          deposit stays a pure function of the two and tile aprons remain
//          bit-consistent (§6.4).
//   freq:  per-axis frequency .xy, with 1/NOISE_TILE_PX pre-folded into .z.
//   amp:   per colour-channel amplitude (0 = that channel holds still).
//   off:   the per-stroke lookup translation (derived from the stroke seed).
// The sampler must wrap (repeat) on both axes.
fn color_jitter(
    tex: texture_2d<f32>,
    s: sampler,
    local: vec2f,
    freq: vec4f,
    amp: vec3f,
    off: vec2f,
) -> vec3f {
    let q = local * freq.xy * freq.z + off;
    return amp * textureSampleLevel(tex, s, q, 0.0).xyz;
}