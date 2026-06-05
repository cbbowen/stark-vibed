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
    knobs: vec4<f32>,       // smear, add, flatten_step, reservoir_capacity
    counts: vec4<u32>,      // segment_count, lead-in columns, reservoir width, _
}

// Must match `SegmentInstance` (gpu/stroke.rs): std430 layout, 48 bytes. Read-only:
// the scan writes only the reservoir texture (per-band premultiplied film), nothing
// goes back into the instances.
struct Instance {
    start: vec2<f32>,
    dir: vec2<f32>,
    geom: vec4<f32>,  // radius, length, flow, reservoir column (u; unused here)
    aux: vec4<f32>,   // height, wet, opacity, _ (the deposit lays the brush's own height)
}

// Per-flatten-step decay of the carried "presence" (load) when pickup/inject stop, so
// a smear runs out on bare canvas over a tail of roughly `1/(1−DECAY)` steps (DESIGN
// §6.2) — unifying the old mixer's continuous smear with the knife's drag-and-fade.
const DECAY: f32 = 0.90;

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
// Companion reservoir carrying the paint *height* (amount) the tool holds — stored as a
// per-stamp deposit *rate* so the additive deposit reconstructs it without magnifying it
// by the segment overlap (DESIGN §6.1/§6.2).
@group(0) @binding(5) var aux_reservoir: texture_storage_2d<rgba16float, write>;

// Reservoir state carried along the walk: the carried paint and a "presence" load.
// `ch.xyz` is the per-unit colour channels (Oklab L,a,b or Mixbox c0,c1,c2); `ch.w` is
// the smear-lifted canvas *height* as a deposit rate (the brush's own height is laid
// separately) — both ride the same mass-weighted recurrence.
struct Res {
    ch: vec4<f32>,
    load: f32,
}

// One step of the unified reservoir recurrence at `sample` (region px), covering
// `steps` flatten-steps of arc. The carried presence decays over distance, is
// replenished by **smear** (mass-weighted pickup of the canvas) and **add** (the
// brush's own colour), and the carried colour is the mass-weighted blend. Over solid
// wet paint the load saturates → a full-strength smear; on bare canvas it decays →
// the smear fades out. Reads the knobs/region/brush globals.
fn step(prev: Res, sample: vec2<f32>, steps: f32, hnorm: f32) -> Res {
    let origin = params.origin_dims.xy;
    let dims = vec2<i32>(i32(params.origin_dims.z), i32(params.origin_dims.w));
    let texel = clamp(vec2<i32>(floor(sample - origin)), vec2<i32>(0), dims - vec2<i32>(1));
    let col = textureLoad(region_color, texel, 0); // premult colour, opacity in .w
    let amount = min(col.w, 1.0);                    // canvas opacity-presence
    let canvas_h = textureLoad(region_aux, texel, 0).x; // canvas paint height (absolute)

    // Picked-up canvas paint: colour as per-unit channels (.xyz, un-premult by presence)
    // and the lifted **height** (.w). The reservoir carries *only* the smear-lifted canvas
    // height — the brush's own height is laid separately at deposit time from the per-
    // segment attribute (it's already a per-stamp rate), so it isn't folded in here. The
    // lifted height is stored as a deposit rate (× `hnorm`) so the additive stamp puts
    // back the absolute lifted amount, not magnified by the segment overlap. Height is
    // independent of opacity, so it is NOT divided by presence.
    var in_ch = vec4<f32>(0.0);
    if amount > 1e-4 {
        in_ch = vec4<f32>(col.xyz / amount, canvas_h * hnorm);
    }
    let br_ch = vec4<f32>(params.brush_ch.xyz, 0.0); // brush's own colour; no height here

    let lift = clamp(params.knobs.x * amount * steps, 0.0, 1.0); // smear pickup mass
    let gain = clamp(params.knobs.y * steps, 0.0, 1.0);          // add (own paint) mass
    let decayed = prev.load * pow(DECAY, steps);
    var load = decayed + lift + gain;
    var ch = prev.ch;
    if load > 1e-4 {
        ch = (prev.ch * decayed + in_ch * lift + br_ch * gain) / load;
    }
    load = min(load, params.knobs.w); // knobs.w = reservoir capacity
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
// Store the reservoir state as premultiplied film `(ch·load, load)` — empty bands
// (load 0) deposit nothing, and linear filtering between a loaded and an empty column
// stays a dimmer version of the same colour rather than fading through black.
fn store_res(res: Res, col: u32, band: u32) {
    let p = vec2<i32>(i32(col), i32(band));
    textureStore(reservoir, p, vec4<f32>(res.ch.xyz * res.load, res.load));
    // The smear-lifted height the deposit lays, premultiplied by presence so it fades to
    // zero as the tool empties. `ch.w` already carries the `hnorm` deposit-rate factor, so
    // the additive stamp reconstructs the absolute lifted height — not magnified by the
    // segment overlap, zero on an empty tool.
    textureStore(aux_reservoir, p, vec4<f32>(res.ch.w * res.load, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(BANDS)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let j = lid.x;
    let count = params.counts.x;
    let lead = params.counts.y;
    let width = params.counts.z;
    let flatten_step = params.knobs.z;

    // Band j's lateral position across the tip, in radius units ∈ [-1, 1].
    let lateral = (2.0 * f32(j) + 1.0) / f32(BANDS) - 1.0;

    var res = Res(vec4<f32>(0.0), 0.0); // starts empty; pickup/add fill it

    // Lead-in: extrapolate behind the first segment along its tangent, picking up
    // canvas there, so the start cap is part of the same continuous walk.
    let s0 = instances[0].start;
    let d0 = instances[0].dir;
    let r0 = instances[0].geom.x;
    let perp0 = vec2<f32>(-d0.y, d0.x);
    let hn0 = flatten_step / (2.0 * r0); // 1 / (sample count) at the start
    for (var k = 0u; k < lead; k = k + 1u) {
        let back = f32(lead - k) * flatten_step; // distance behind the start
        res = step(res, s0 - d0 * back + perp0 * (lateral * r0), 1.0, hn0);
        store_res(res, k, j);
    }

    // The stroke proper, one column per segment.
    for (var i = 0u; i < count; i = i + 1u) {
        let pos = instances[i].start;
        let dir = instances[i].dir;
        let radius = instances[i].geom.x;
        let perp = vec2<f32>(-dir.y, dir.x);
        let steps = instances[i].geom.y / flatten_step;
        res = step(res, pos + perp * (lateral * radius), steps, flatten_step / (2.0 * radius));
        store_res(res, lead + i, j);
    }

    // Lead-out: extrapolate ahead of the last segment along its tangent.
    let li = count - 1u;
    let dl = instances[li].dir;
    let rl = instances[li].geom.x;
    let perpl = vec2<f32>(-dl.y, dl.x);
    let endp = instances[li].start + dl * instances[li].geom.y; // last segment's end
    let hnl = flatten_step / (2.0 * rl); // 1 / (sample count) at the end
    for (var k = lead + count; k < width; k = k + 1u) {
        let fwd = f32(k - (lead + count) + 1u) * flatten_step; // distance past the end
        res = step(res, endp + dl * fwd + perpl * (lateral * rl), 1.0, hnl);
        store_res(res, k, j);
    }
}
