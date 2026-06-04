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
    // Mixer mode: pickup, color_inject, flatten_step, reservoir_capacity.
    // Knife mode:  bite,   carry,        flatten_step, reservoir_capacity.
    knobs: vec4<f32>,
    counts: vec4<u32>,      // segment_count, lead-in columns, reservoir width, mode (0=mixer, 1=knife)
}

// Must match `SegmentInstance` (gpu/stroke.rs): std430 layout, 48 bytes. Read-only:
// both modes write only the reservoir texture (the knife stores its per-band film
// amount in the reservoir alpha, so nothing goes back into the instances).
struct Instance {
    start: vec2<f32>,
    dir: vec2<f32>,
    geom: vec4<f32>,  // radius, length, flow, reservoir column (u; unused here)
    aux: vec4<f32>,   // height, wet, opacity, _ (unused here)
}

const MODE_KNIFE: u32 = 1u;

// Knife carry: fraction of the reservoir load re-laid as film per flatten-step.
// Past the scraped paint the knife picks up nothing, so the load — and thus the
// deposited film — decays as `(1 - DEPOSIT_RATE)^steps`, fading the dragged paint
// out over a tail of roughly `radius / DEPOSIT_RATE` flatten-steps (DESIGN.md §6.2).
const DEPOSIT_RATE: f32 = 0.12;

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

// One step of the *knife* carry recurrence: scrape paint off the canvas at
// `sample` into the reservoir (mass-weighted, `bite`-driven), with no brush-color
// refill — the knife only ever holds what it lifted. Depletion (re-laying the film)
// is handled by the caller so it can record the deposited amount (DESIGN.md §6.2).
fn pickup_knife(prev: Res, sample: vec2<f32>, steps: f32) -> Res {
    let origin = params.origin_dims.xy;
    let dims = vec2<i32>(i32(params.origin_dims.z), i32(params.origin_dims.w));
    let texel = clamp(vec2<i32>(floor(sample - origin)), vec2<i32>(0), dims - vec2<i32>(1));
    let dec = decode(textureLoad(region_color, texel, 0));

    // Always-wet medium: the scrape lifts in proportion to bite × paint present.
    let lift = clamp(params.knobs.x * dec.amount * steps, 0.0, 1.0);
    let total = prev.load + lift;
    var ch = prev.ch;
    if total > 1e-4 {
        ch = (prev.ch * prev.load + dec.ch * lift) / total;
    }
    return Res(ch, min(total, params.knobs.w)); // knobs.w = capacity
}

// Film amount re-laid this step: a fixed fraction of the current load per arc step.
fn deposit_amt(load: f32, steps: f32) -> f32 {
    return clamp(DEPOSIT_RATE * steps, 0.0, 1.0) * load;
}

// What to write into a reservoir texel after one step, by mode.
struct Stored {
    texel: vec4<f32>, // mixer: (carried color, 1); knife: premult film (ch·film, film)
    load: f32,        // the (possibly depleted) reservoir load to carry forward
}

// Mixer: the texel is just the carried color (alpha unused by the deposit). Knife:
// re-lay a `DEPOSIT_RATE`-fraction of the load, scaled by `carry`, as the film — and
// store it **premultiplied** so empty bands (load 0 → film 0 → texel 0) deposit
// nothing and linear filtering between a loaded and an empty column stays a dimmer
// version of the same colour rather than fading through black.
fn stored(res: Res, steps: f32, is_knife: bool) -> Stored {
    if !is_knife {
        return Stored(vec4<f32>(res.ch.xyz, 1.0), res.load);
    }
    let film = deposit_amt(res.load, steps) * params.knobs.y; // knobs.y = carry
    return Stored(vec4<f32>(res.ch.xyz * film, film), res.load - deposit_amt(res.load, steps));
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
    let is_knife = params.counts.w == MODE_KNIFE;

    // Band j's lateral position across the tip, in radius units ∈ [-1, 1].
    let lateral = (2.0 * f32(j) + 1.0) / f32(BANDS) - 1.0;

    // The mixer starts dipped in the brush color; the knife starts empty (it holds
    // only paint it scrapes off the canvas).
    var res = Res(params.brush_ch, 1.0);
    if is_knife {
        res = Res(vec4<f32>(0.0), 0.0);
    }

    // Lead-in: extrapolate behind the first segment along its tangent, picking up
    // canvas there, so the start cap is part of the same continuous walk.
    let s0 = instances[0].start;
    let d0 = instances[0].dir;
    let r0 = instances[0].geom.x;
    let perp0 = vec2<f32>(-d0.y, d0.x);
    for (var k = 0u; k < lead; k = k + 1u) {
        let back = f32(lead - k) * flatten_step; // distance behind the start
        let sample = s0 - d0 * back + perp0 * (lateral * r0);
        if is_knife { res = pickup_knife(res, sample, 1.0); } else { res = pickup(res, sample, 1.0); }
        let st = stored(res, 1.0, is_knife);
        res.load = st.load;
        textureStore(reservoir, vec2<i32>(i32(k), i32(j)), st.texel);
    }

    // The stroke proper, one column per segment.
    for (var i = 0u; i < count; i = i + 1u) {
        let pos = instances[i].start;
        let dir = instances[i].dir;
        let radius = instances[i].geom.x;
        let perp = vec2<f32>(-dir.y, dir.x);
        let steps = instances[i].geom.y / flatten_step;
        let sample = pos + perp * (lateral * radius);
        if is_knife { res = pickup_knife(res, sample, steps); } else { res = pickup(res, sample, steps); }
        let st = stored(res, steps, is_knife);
        res.load = st.load;
        textureStore(reservoir, vec2<i32>(i32(lead + i), i32(j)), st.texel);
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
        if is_knife { res = pickup_knife(res, sample, 1.0); } else { res = pickup(res, sample, 1.0); }
        let st = stored(res, 1.0, is_knife);
        res.load = st.load;
        textureStore(reservoir, vec2<i32>(i32(k), i32(j)), st.texel);
    }
}
