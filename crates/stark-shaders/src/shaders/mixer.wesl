// Conservative brush-dynamics reservoir scan (DESIGN.md §6.2, "Wet mixing & brush
// dynamics").
//
// A serial pass walks the stroke's flattened segments in order, maintaining an explicit
// **tool reservoir** per lateral band: each step **lifts** a fraction of the canvas paint
// onto the tool, **deposits** a fraction of the tool's paint back onto the canvas, and
// **adds** the brush's own paint straight to the canvas. The walk writes two textures the
// sweep/deposit shader then samples (linear filtering blends the bands for free):
//   1. `reservoir`     — the deposit's *colour*: the tool's tint weighted by its height
//                        share of the resulting paint, with the brush's own `add` colour
//                        composited over it (premultiplied; see `step`).
//   2. `aux_reservoir` — `.x` the **deposited height** (≥ 0, a per-flatten-step rate);
//                        `.yzw` unused. The lift's *subtraction* deliberately does NOT
//                        ride this banded texture: the integrate removes
//                        `lift × contact` of the base height per-pixel — exact, where
//                        the banded reconstruction of `lift × canvas_h` left residue an
//                        eraser could never clean up. The tool's intake here (banded)
//                        and that exact removal agree in expectation, so paint stays
//                        approximately conserved with `add = 0` (DESIGN §6.1/§6.2).
//
// Running this on the GPU keeps pickup off the CPU entirely — WebGPU cannot block for a
// buffer map, so a CPU readback is impossible (it panics in getMappedRange).
//
// `encode_mixer` in gpu/stroke.rs encodes this pass (and the region composite feeding it).

struct Params {
    brush_ch: vec4<f32>,    // the brush's own colour channels (.xyz); .w = `add` axis
    origin_dims: vec4<f32>, // region origin.xy (canvas px), region dims.xy (px)
    knobs: vec4<f32>,       // charge (tool pre-load height), brush opacity, flatten_step, _
    counts: vec4<u32>,      // segment_count, lead-in columns, reservoir width, _
}

// Must match `SegmentInstance` (gpu/stroke.rs): std430 layout, 64 bytes. Read-only:
// the scan writes only the reservoir textures, nothing goes back into the instances.
struct Instance {
    start: vec2<f32>,
    dir: vec2<f32>,
    geom: vec4<f32>,  // radius, length, flow, reservoir start column (u; unused here)
    aux: vec4<f32>,   // height, wet, opacity (the brush's own paint, for the `add` axis),
                      // reservoir u-advance per radius (used by the deposit; unused here)
    extra: vec4<f32>, // orientation (turns; used by the deposit), load (scrape rate),
                      // deposit rate, _
}

// Lateral reservoir bands across the brush width, so each part of the tip carries the
// colour it rolled over instead of one colour for the whole brush (DESIGN.md §6.2). Band
// `j` sits at brush-local y = (2j+1)/BANDS − 1 (the row's texel center mapped onto
// [-1,1]); the deposit samples the texture at the matching y, so the two stay aligned with
// no shared offset constant. The reservoir texture's height is `BANDS` (set in stroke.rs).
const BANDS: u32 = 64u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var region_color: texture_2d<f32>;
@group(0) @binding(2) var region_aux: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var reservoir: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var aux_reservoir: texture_storage_2d<rgba16float, write>;

// The tool reservoir carried along the walk. `h` is the **amount** of paint the tool holds
// = HEIGHT (the conserved quantity; DESIGN §6.1). `ch.xyz` is the per-unit latent colour and
// `op` the per-unit opacity (a material property, *not* an amount) — both carried as
// height-weighted blends of the paint lifted off the canvas. The tool starts empty and
// fills only by lifting canvas height. Wetness is deliberately NOT carried: a dry tool
// wicks moisture rather than transporting it, so the lift leaves the canvas wet channel
// in place and the deposit lays only the brush's own wet (the `add`-gated per-segment
// attribute in the deposit shader).
struct Tool {
    ch: vec3<f32>,
    op: f32,
    h: f32,
}

// What `step` emits for one reservoir texel, alongside the updated tool. The deposit colour
// is the tool's tint weighted by its height share of the resulting paint (see the colour
// block in `step`), with the brush's own `add` colour composited over it, laid through the
// sweep colour target's *over*-blend. `dep_h` is the **deposited height** as a
// per-flatten-step *rate* (the segment's amount ÷ its `steps`, so short segments store the
// same length-independent value as full ones; see `step`). The integrate adds it to the
// base; the lift's matching removal is computed per-pixel there (`lift × contact`), not
// here — with `add = 0` what is deposited was lifted elsewhere, so paint moves and the
// total canvas+tool height is (approximately) conserved (DESIGN §6.1/§6.2).
struct StepOut {
    tool: Tool,
    dep_rgb: vec3<f32>, // deposit+add premultiplied colour (over-blended)
    dep_op: f32,        // deposit+add per-unit opacity (over-blended)
    dep_h: f32,         // deposited height (≥ 0), as a per-flatten-step rate
}

// One step of the reservoir recurrence at `sample` (region px), covering `steps`
// flatten-steps of arc. `bru` is the brush's own paint for this segment (height, wet,
// opacity), its colour laid by the `add` axis. `lift_rate`/`deposit_rate` are this
// segment's per-flatten-step scrape/deposit fractions (pressure/tilt modulated; the
// instance `extra.yz`). Reads the knobs/region/brush globals.
fn step(prev: Tool, sample: vec2<f32>, steps: f32, bru: vec3<f32>, lift_rate: f32, deposit_rate: f32) -> StepOut {
    let origin = params.origin_dims.xy;
    let dims = vec2<i32>(i32(params.origin_dims.z), i32(params.origin_dims.w));
    let texel = clamp(vec2<i32>(floor(sample - origin)), vec2<i32>(0), dims - vec2<i32>(1));
    let col = textureLoad(region_color, texel, 0); // premult colour, per-unit opacity in .w
    let a = min(col.w, 1.0);                         // canvas per-unit opacity
    let canvas_h = textureLoad(region_aux, texel, 0).x; // canvas paint AMOUNT (height)
    var canvas_ch = vec3<f32>(0.0);
    if a > 1e-4 {
        canvas_ch = col.xyz / a; // un-premultiply to per-unit latent colour
    }

    let lift = clamp(lift_rate * steps, 0.0, 1.0);         // fraction of canvas height lifted
    let deposit = clamp(deposit_rate * steps, 0.0, 1.0);   // fraction of tool height deposited
    let add = clamp(params.brush_ch.w * steps, 0.0, 1.0);  // brush's own-paint fraction

    var tool = prev;

    // --- Lift: move a fraction of the canvas HEIGHT onto the tool. The tool's per-unit
    // colour/opacity become the height-weighted blend of what it holds and what it picks
    // up (opacity is per-unit, so it averages — it is not accumulated like an amount).
    let lift_h = lift * canvas_h;
    let new_h = tool.h + lift_h;
    if new_h > 1e-4 {
        tool.ch = (tool.ch * tool.h + canvas_ch * lift_h) / new_h;
        tool.op = (tool.op * tool.h + a * lift_h) / new_h;
    }
    tool.h = new_h;

    // --- Deposit: move a fraction of the tool's HEIGHT back onto the canvas. On bare canvas
    // (no lift) the tool empties geometrically → the smear fades; over paint the lift refills
    // it. `deposit` replaces the old fixed decay.
    let dep_h = deposit * tool.h;
    tool.h = tool.h - dep_h;

    // --- Colour: the tool's tint weighted by the deposit's **share of the resulting
    // paint** — `dep_h / (remaining + dep_h)`, the height-fraction of the post-transfer
    // paint that came off the tool — with the brush's own `add` colour composited over it
    // (one colour texture carries both). Colour must blend by *amount* (height), not by
    // per-unit opacity alone (§6.1): without the share weighting, a `deposit = 0` eraser
    // would re-tint the paint it passes with carried colour at full alpha while laying
    // nothing, and a tiny deposit would disproportionately re-tint thick paint. With it,
    // an eraser leaves colour untouched (share 0), a deposit onto bare canvas lays the
    // full tint (share 1), and everything between mixes in proportion. `add` here only
    // contributes the own *colour*; the own height rides the per-segment attribute in the
    // deposit shader, gated by the same `add` axis.
    let remaining = canvas_h - lift_h; // canvas height left after the lift (lift ≤ 1)
    let share = dep_h / max(remaining + dep_h, 1e-4);
    let tint = tool.op * share;
    let ao = add * bru.z; // own-paint over-coverage (per-unit opacity × add)
    var out: StepOut;
    out.tool = tool;
    out.dep_rgb = params.brush_ch.xyz * ao + tool.ch * tint * (1.0 - ao);
    out.dep_op = ao + tint * (1.0 - ao);
    // The tool advanced over `steps` flatten-steps of arc; the deposit amount above is
    // proportional to that. Emit the per-flatten-step **rate** (÷ steps) so the reservoir
    // holds a length-independent value — the arc each segment covers is reapplied by the
    // deposit's footprint accumulation. Without this, the short remainder segment at the
    // stroke end (and any short segment on a curve) gets a fractional-step amount → a
    // one-column dip → a visible seam (DESIGN §6.2).
    out.dep_h = dep_h / max(steps, 1e-4); // per-flatten-step deposited height
    return out;
}

// Store one reservoir column for band `j`. The deposit colour goes to `reservoir`
// (premultiplied; the sweep's colour target over-blends it) and the deposited height
// rate to `aux_reservoir.x` (DESIGN §6.1/§6.2).
fn store_res(s: StepOut, col: u32, band: u32) {
    let p = vec2<i32>(i32(col), i32(band));
    textureStore(reservoir, p, vec4<f32>(s.dep_rgb, s.dep_op));
    textureStore(aux_reservoir, p, vec4<f32>(s.dep_h, 0.0, 0.0, 0.0));
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

    // Pre-charge: the tool starts carrying `charge` height of the brush's own paint (the
    // palette-knife glob), its per-unit colour/opacity the brush's own. Depositing depletes
    // it, scraping refills it (DESIGN §6.2). `charge = 0` → an empty tool (the historical
    // behaviour: the `if new_h > 1e-4` guard in `step` makes the initial colour moot).
    let charge = params.knobs.x;
    let brush_op = params.knobs.y;
    var tool = Tool(params.brush_ch.xyz, brush_op, charge);

    // Lead-in: extrapolate behind the first segment along its tangent, lifting/depositing
    // there, so the start cap is part of the same continuous walk.
    let s0 = instances[0].start;
    let d0 = instances[0].dir;
    let r0 = instances[0].geom.x;
    let perp0 = vec2<f32>(-d0.y, d0.x);
    let bru0 = instances[0].aux.xyz;
    let lift0 = instances[0].extra.y;
    let dep0 = instances[0].extra.z;
    for (var k = 0u; k < lead; k = k + 1u) {
        let back = f32(lead - k) * flatten_step; // distance behind the start
        let s = step(tool, s0 - d0 * back + perp0 * (lateral * r0), 1.0, bru0, lift0, dep0);
        tool = s.tool;
        store_res(s, k, j);
    }

    // The stroke proper, one column per segment.
    for (var i = 0u; i < count; i = i + 1u) {
        let pos = instances[i].start;
        let dir = instances[i].dir;
        let radius = instances[i].geom.x;
        let perp = vec2<f32>(-dir.y, dir.x);
        let steps = instances[i].geom.y / flatten_step;
        let s = step(tool, pos + perp * (lateral * radius), steps, instances[i].aux.xyz, instances[i].extra.y, instances[i].extra.z);
        tool = s.tool;
        store_res(s, lead + i, j);
    }

    // Lead-out: extrapolate ahead of the last segment along its tangent.
    let li = count - 1u;
    let dl = instances[li].dir;
    let rl = instances[li].geom.x;
    let perpl = vec2<f32>(-dl.y, dl.x);
    let endp = instances[li].start + dl * instances[li].geom.y; // last segment's end
    let brul = instances[li].aux.xyz;
    let liftl = instances[li].extra.y;
    let depl = instances[li].extra.z;
    // The deposit maps the last segment's *end* (local.x = length/radius) to column
    // `lead+count`, so the first lead-out column continues from `endp` itself (fwd = 0) —
    // not one step past it, which would shift the cap and leave a one-segment seam at the end.
    for (var k = lead + count; k < width; k = k + 1u) {
        let fwd = f32(k - (lead + count)) * flatten_step; // distance past the end (0 at the end)
        let s = step(tool, endp + dl * fwd + perpl * (lateral * rl), 1.0, brul, liftl, depl);
        tool = s.tool;
        store_res(s, k, j);
    }
}
