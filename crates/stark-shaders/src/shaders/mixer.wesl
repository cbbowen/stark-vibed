// Wet-mixing reservoir scan (DESIGN.md §6.2, "Wet mixing & brush dynamics").
//
// A single serial pass walks the stroke's flattened segments in order, lifting
// wet paint from the pre-composited base region into a small reservoir and
// writing each segment's resulting color back into the instance buffer (the
// sweep shader then deposits it). Running this on the GPU keeps pickup off the
// CPU entirely — WebGPU cannot block for a buffer map, so a CPU readback is
// impossible (it panics in getMappedRange).
//
// `decode` mirrors `ColorSpace::decode_canvas` in colorspace.rs; the reservoir
// math mirrors `Reservoir::step` in gpu/stroke.rs. Keep all three in sync.

struct Params {
    brush_ch: vec4<f32>,    // the brush's own color channels
    origin_dims: vec4<f32>, // region origin.xy (canvas px), region dims.xy (px)
    knobs: vec4<f32>,       // pickup, color_inject, flatten_step, reservoir_capacity
    counts: vec4<u32>,      // segment_count, color_space (0=oklab, 1=pigment), _, _
}

// Must match `SegmentInstance` (gpu/stroke.rs): std430 layout, 64 bytes.
struct Instance {
    start: vec2<f32>,
    dir: vec2<f32>,
    geom: vec4<f32>, // radius, length, flow, _
    ch: vec4<f32>,
    aux: vec2<f32>,  // height, wet
    pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var region_color: texture_2d<f32>;
@group(0) @binding(2) var region_aux: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> instances: array<Instance>;

struct Decoded {
    ch: vec4<f32>,  // per-unit pickup channels
    amount: f32,    // paint present, in [0, 1]
}

// Inverse of the deposit's accumulation (mirrors ColorSpace::decode_canvas).
fn decode(stored: vec4<f32>, space: u32) -> Decoded {
    var out: Decoded;
    if space == 1u {
        // Pigment: additively accumulated amounts → normalized mixture + K–M-style
        // coverage 1 − exp(−Σ).
        let sum = stored.x + stored.y + stored.z + stored.w;
        if sum <= 1e-4 {
            return Decoded(vec4<f32>(0.0), 0.0);
        }
        return Decoded(stored / sum, 1.0 - exp(-sum));
    }
    // Oklab: premultiplied (L·c, a·c, b·c, c) → un-premultiply.
    let c = stored.w;
    if c <= 1e-4 {
        return Decoded(vec4<f32>(0.0), 0.0);
    }
    out.ch = vec4<f32>(stored.xyz / c, 0.0);
    out.amount = min(c, 1.0);
    return out;
}

@compute @workgroup_size(1)
fn main() {
    let count = params.counts.x;
    let space = params.counts.y;
    let origin = params.origin_dims.xy;
    let dims = vec2<i32>(i32(params.origin_dims.z), i32(params.origin_dims.w));
    let pickup = params.knobs.x;
    let inject_rate = params.knobs.y;
    let flatten_step = params.knobs.z;
    let capacity = params.knobs.w;

    var ch = params.brush_ch; // reservoir starts dipped in the brush color
    var load = 1.0;

    for (var i = 0u; i < count; i = i + 1u) {
        let pos = instances[i].start;
        let len = instances[i].geom.y;
        let steps = len / flatten_step;

        // Sample the base region at the segment start (nearest texel). The region
        // is 1:1 with canvas over [origin, origin+dims], so texel = pos − origin.
        let texel = clamp(vec2<i32>(floor(pos - origin)), vec2<i32>(0), dims - vec2<i32>(1));
        let dec = decode(textureLoad(region_color, texel, 0), space);
        let wet = textureLoad(region_aux, texel, 0).y;

        // PICKUP: mass-weighted blend toward the canvas (convex combination).
        let lift = clamp(pickup * wet * dec.amount * steps, 0.0, 1.0);
        let total = load + lift;
        if total > 1e-4 {
            ch = (ch * load + dec.ch * lift) / total;
        }
        load = min(total, capacity);

        // Refill toward the brush's own color (0 = pure smudge that only carries).
        let inj = clamp(inject_rate * steps, 0.0, 1.0);
        ch = ch * (1.0 - inj) + params.brush_ch * inj;

        instances[i].ch = ch;
    }
}
