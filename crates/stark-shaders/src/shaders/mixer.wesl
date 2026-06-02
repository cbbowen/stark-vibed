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
    counts: vec4<u32>,      // segment_count, lead-in columns, reservoir width, _
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

// Reservoir state carried along the walk.
struct Res {
    ch: vec4<f32>,
    load: f32,
}

// One step of the reservoir recurrence: lift wet paint from the canvas at `sample`
// (region px) into the reservoir, then refill toward the brush color. `steps` is
// the arc length covered, in flatten steps. Reads the knobs/region/brush globals.
fn pickup(prev: Res, sample: vec2<f32>, steps: f32) -> Res {
    let origin = params.origin_dims.xy;
    let dims = vec2<i32>(i32(params.origin_dims.z), i32(params.origin_dims.w));
    let texel = clamp(vec2<i32>(floor(sample - origin)), vec2<i32>(0), dims - vec2<i32>(1));
    let dec = decode(textureLoad(region_color, texel, 0));
    let wet = textureLoad(region_aux, texel, 0).y;

    // PICKUP: mass-weighted blend toward the canvas (convex combination).
    let lift = clamp(params.knobs.x * wet * dec.amount * steps, 0.0, 1.0);
    let total = prev.load + lift;
    var ch = prev.ch;
    if total > 1e-4 {
        ch = (prev.ch * prev.load + dec.ch * lift) / total;
    }
    let load = min(total, params.knobs.w); // knobs.w = reservoir capacity

    // Refill toward the brush's own color (0 = pure smudge that only carries).
    let inj = clamp(params.knobs.y * steps, 0.0, 1.0);
    ch = ch * (1.0 - inj) + params.brush_ch * inj;
    return Res(ch, load);
}

// One serial reservoir per lateral band, run in parallel: invocation `j` walks the
// whole stroke maintaining band j's reservoir and writing reservoir texel (column,
// band j). The walk is extended a half-brush *before* the first segment and *after*
// the last (the `lead`/lead-out pad columns), continuing to pick up canvas along the
// path tangent there — so the brush's round end caps blend smoothly into the stroke
// (they share the same continuous reservoir) instead of clamping to a flat slab of
// the edge segment's color. The bands are independent (distinct reservoirs, distinct
// output texels), so they need no synchronization.
@compute @workgroup_size(BANDS)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let j = lid.x;
    let count = params.counts.x;
    let lead = params.counts.y;
    let width = params.counts.z;
    let flatten_step = params.knobs.z;

    // Band j's lateral position across the tip, in radius units ∈ [-1, 1].
    let lateral = (2.0 * f32(j) + 1.0) / f32(BANDS) - 1.0;

    var res = Res(params.brush_ch, 1.0); // starts dipped in the brush color

    // Lead-in: extrapolate behind the first segment along its tangent, picking up
    // canvas there, so the start cap is part of the same continuous walk.
    let s0 = instances[0].start;
    let d0 = instances[0].dir;
    let r0 = instances[0].geom.x;
    let perp0 = vec2<f32>(-d0.y, d0.x);
    for (var k = 0u; k < lead; k = k + 1u) {
        let back = f32(lead - k) * flatten_step; // distance behind the start
        let sample = s0 - d0 * back + perp0 * (lateral * r0);
        res = pickup(res, sample, 1.0);
        textureStore(reservoir, vec2<i32>(i32(k), i32(j)), vec4<f32>(res.ch.xyz, 1.0));
    }

    // The stroke proper, one column per segment.
    for (var i = 0u; i < count; i = i + 1u) {
        let pos = instances[i].start;
        let dir = instances[i].dir;
        let radius = instances[i].geom.x;
        let perp = vec2<f32>(-dir.y, dir.x);
        let steps = instances[i].geom.y / flatten_step;
        res = pickup(res, pos + perp * (lateral * radius), steps);
        textureStore(reservoir, vec2<i32>(i32(lead + i), i32(j)), vec4<f32>(res.ch.xyz, 1.0));
    }

    // Lead-out: extrapolate ahead of the last segment along its tangent.
    let li = count - 1u;
    let dl = instances[li].dir;
    let rl = instances[li].geom.x;
    let perpl = vec2<f32>(-dl.y, dl.x);
    let endp = instances[li].start + dl * instances[li].geom.y; // last segment's end
    for (var k = lead + count; k < width; k = k + 1u) {
        let fwd = f32(k - (lead + count) + 1u) * flatten_step; // distance past the end
        let sample = endp + dl * fwd + perpl * (lateral * rl);
        res = pickup(res, sample, 1.0);
        textureStore(reservoir, vec2<i32>(i32(k), i32(j)), vec4<f32>(res.ch.xyz, 1.0));
    }
}
