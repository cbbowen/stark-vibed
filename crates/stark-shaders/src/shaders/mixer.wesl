// Wet-mixing reservoir scan (DESIGN.md §6.2, "Wet mixing & brush dynamics").
//
// A serial pass walks the stroke's flattened segments in order, lifting wet paint
// from the pre-composited base region into a small reservoir and writing the
// resulting color into a reservoir *texture* (one column per segment × one row per
// lateral band), which the sweep shader then samples — linear filtering blends the
// bands for free, so the deposit shader needn't know how many there are. Running
// this on the GPU keeps pickup off the CPU entirely — WebGPU cannot block for a
// buffer map, so a CPU readback is impossible (it panics in getMappedRange).
//
// The reservoir math mirrors `encode_mixer` in gpu/stroke.rs.

struct Params {
    brush_ch: vec4<f32>,    // the brush's own color channels
    origin_dims: vec4<f32>, // region origin.xy (canvas px), region dims.xy (px)
    knobs: vec4<f32>,       // pickup, color_inject, flatten_step, reservoir_capacity
    counts: vec4<u32>,      // segment_count, _, _, _
}

// Must match `SegmentInstance` (gpu/stroke.rs): std430 layout, 48 bytes. Read-only
// here — the per-segment colors now go to the reservoir texture, not back into the
// instances.
struct Instance {
    start: vec2<f32>,
    dir: vec2<f32>,
    geom: vec4<f32>,  // radius, length, flow, reservoir column (u; unused here)
    aux: vec2<f32>,   // height, wet
    pad: vec2<f32>,
}

// Lateral reservoir bands across the brush width, so each part of the tip carries
// the color it rolled over instead of one color for the whole brush (DESIGN.md
// §6.2). Band `j` sits at brush-local y = (2j+1)/BANDS − 1 (the row's texel center
// mapped onto [-1,1]); the deposit samples the texture at the matching y, so the
// two stay aligned with no shared offset constant. The reservoir texture's height
// is `BANDS` (set in gpu/stroke.rs).
const BANDS: u32 = 16u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var region_color: texture_2d<f32>;
@group(0) @binding(2) var region_aux: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var reservoir: texture_storage_2d<rgba16float, write>;

struct Decoded {
    ch: vec4<f32>,  // per-unit pickup channels
    amount: f32,    // paint present, in [0, 1]
}

// Un-premultiply a stored color texel into (per-unit channels, coverage). Every
// color space stores premultiplied channels with coverage in alpha (Oklab L,a,b;
// Mixbox pigment concentrations c0,c1,c2), so one decode serves all.
fn decode(stored: vec4<f32>) -> Decoded {
    let c = stored.w;
    if c <= 1e-4 {
        return Decoded(vec4<f32>(0.0), 0.0);
    }
    return Decoded(vec4<f32>(stored.xyz / c, 0.0), min(c, 1.0));
}

// One serial reservoir per lateral band, run in parallel: invocation `j` walks the
// whole stroke maintaining band j's reservoir, sampling the canvas at band j's
// lateral offset from the path and writing reservoir texel (segment i, band j).
// The bands are independent (distinct reservoirs, distinct output texels), so they
// need no synchronization.
@compute @workgroup_size(BANDS)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let j = lid.x;
    let count = params.counts.x;
    let origin = params.origin_dims.xy;
    let dims = vec2<i32>(i32(params.origin_dims.z), i32(params.origin_dims.w));
    let pickup = params.knobs.x;
    let inject_rate = params.knobs.y;
    let flatten_step = params.knobs.z;
    let capacity = params.knobs.w;

    // Band j's lateral position across the tip, in radius units ∈ [-1, 1].
    let lateral = (2.0 * f32(j) + 1.0) / f32(BANDS) - 1.0;

    var ch = params.brush_ch; // reservoir starts dipped in the brush color
    var load = 1.0;

    for (var i = 0u; i < count; i = i + 1u) {
        let pos = instances[i].start;
        let dir = instances[i].dir;
        let radius = instances[i].geom.x;
        let len = instances[i].geom.y;
        let steps = len / flatten_step;

        // Sample the base region at this band's lateral offset from the path. The
        // region is 1:1 with canvas over [origin, origin+dims].
        let perp = vec2<f32>(-dir.y, dir.x);
        let sample = pos + perp * (lateral * radius);
        let texel = clamp(vec2<i32>(floor(sample - origin)), vec2<i32>(0), dims - vec2<i32>(1));
        let dec = decode(textureLoad(region_color, texel, 0));
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

        textureStore(reservoir, vec2<i32>(i32(i), i32(j)), vec4<f32>(ch.xyz, 1.0));
    }
}
