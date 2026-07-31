// Brush dynamics: the **sequential swept-exchange loop** (DESIGN.md §6.2).
//
// A stroke that manipulates existing paint (lift / deposit / charge) runs as an
// ordered loop, all inside one compute pass (implicit barriers between dispatches
// give the sequencing; usage scopes are per-dispatch, so the region may be
// sampled by one dispatch and storage-written by the next — no copies):
//
//   - `pickup` (at reservoir cadence, every `spacing·radius` of travel) — one
//     thread per **tool reservoir** texel: sample the (evolving) canvas region
//     under that part of the tip, lift paint onto the tool, deplete the tool by
//     the upcoming deposit. Ping-pongs the reservoir.
//   - per flattened segment: `bake` (integrate the reservoir along the travel
//     axis, so the deposit can read what the tip presented over a whole pass
//     rather than at one point), `snapshot` (copy the segment's footprint texels
//     to an `under` scratch, so the exchange can read-modify-write the region),
//     then `deposit` — one thread per footprint texel.
//
// **The canvas-side exchange is swept, not stamped** — the same definite-integral
// formulation as the plain deposit: a texel's *exposure* to the segment is the
// prefix-τ difference `e(x) = prefix(u) − prefix(u − d)` (the brush's coverage
// integrated along the travel), and every exchange rate is exponential in
// exposure, `1 − exp(λ·e)` with `λ = ln(1 − axis)`. Exposures add across the
// overlapping quads of consecutive segments, so the rates compose **exactly**:
// the whole stroke applies `1 − (1−axis)^∫e` — the continuous path integral,
// with no per-stamp dabbing regardless of spacing.
//
// The reservoir is a real 2-D texture in brush-local coordinates, so each part of
// the tip carries what *it* rolled through; because pickup reads the region as
// already modified by earlier segments, smearing is genuinely order-dependent —
// including across the stroke's own trail.
//
// **The reservoir's own evolution is ramped, not stepped.** The canvas-side
// exchange above composes exactly, but the tool state it serves changes at
// discrete instants: `pickup` lands a whole interval's reload between two
// segments, and at high lift × deposit that is a large fraction of the load —
// so consecutive segments served visibly different tools, and the stroke showed
// crescent seams at the reservoir cadence (level sets of a texel's partial
// exposure, i.e. the tip's edge). `pickup` therefore also writes its reload as a
// separate **gain parcel**, and the bake serves `R + G·(γ − ½)` with
// `γ = (t0 + s)/L` the fraction of the pickup interval elapsed at the moment of
// service — the reload fades in linearly across the interval (mean-preserving:
// the average serving equals the stepped one, so stroke feel is unchanged; a
// 1-D simulation of this loop puts the seam ripple at ~1.7% stepped vs ~0.5%
// ramped, with the peak seam gradient down ~4×). `γ` is linear in the serving
// column, so it rides the same prefix trick as everything else — two extra
// baked prefixes (`G` and `G·q`), no per-fragment loop. The matching
// `deplete` step stays multiplicative (it composes exactly across any cut); its
// residual against the deposit's exposure currency is bounded by cutting
// segments at *half* the cadence (`stroke::flatten_tolerance`).
//
// Conservation (DESIGN.md §6.1): paint moves by transferring **height** — the one
// conserved quantity. Colour and per-unit opacity ride as optical-mass
// (opacity·height) weighted blends; `add` is the only source term. Both sides of
// each transfer integrate the same rate over the same footprint, so with
// `add = 0` total height (canvas + tool) is conserved up to resampling error.
//
// Determinism: segment/pickup slots are fixed CPU-side per stroke; every dispatch
// is a pure function of what it reads. Replay and live == committed hold.

import package::stamp_common::{prefix_at};
import package::noise::{color_jitter};
import package::paint_common::{blend_latent, EPS};

struct Stamp {
    // a: position .xy (region px) + frame x-axis .zw (unit vector). For a segment
    //    slot: the segment start and its travel tangent (the prefix-τ frame). For
    //    a pickup slot: the stamp centre and the *shape* frame (travel rotated by
    //    the orientation, §6.6) — the reservoir's own frame.
    a: vec4<f32>,
    // b: radius (px), segment length / radius (0 for pickup), λ_lift, λ_deposit —
    //    λ = ln(1 − axis) ≤ 0, clamped away from −∞.
    b: vec4<f32>,
    // c: the brush's own colour channels .xyz + per-unit opacity .w (drained).
    c: vec4<f32>,
    // d: under-rect origin .xy (region px, integral); orientation (turns ∈ [0,1),
    //    picks the prefix-τ slice) .z; exposure scale .w (pickup: the travel since
    //    the last pickup in radius units; segments: 1 — exposure comes from the
    //    prefix difference directly).
    d: vec4<f32>,
    // e: the `add` rate per unit exposure: height .x (.y unused). For a segment
    //    slot, the gain-ramp coordinates: travel into the current pickup interval at
    //    the segment start .z and the interval's nominal length .w (radius units).
    e: vec4<f32>,
    // f–h: colour dynamics (DESIGN.md §6.2) for the brush's own `add` paint —
    // the same field/domain as the fast path's jitter, so the two paths agree.
    // f: noise frequency per lookup axis (across the stroke, along it) .xy,
    //    1/NOISE_TILE_PX in .z; .w unused.
    f: vec4<f32>,
    // g: per colour-channel noise amplitude .xyz (all 0 = off); arc length at
    //    the segment start (canvas px) in .w.
    g: vec4<f32>,
    // h: per-stroke noise lookup translation .xy; .zw unused.
    h: vec4<f32>,
}

@group(0) @binding(0) var<uniform> st: Stamp;

// The stroke's canvas region: premultiplied colour (latent·op, op) + wide aux
// (height, _, _, _). Read by snapshot (textureLoad) and pickup (bilinear).
@group(0) @binding(1) var region_color: texture_2d<f32>;
@group(0) @binding(2) var region_aux: texture_2d<f32>;
// Footprint snapshot targets (written by `snapshot`).
@group(0) @binding(3) var under_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var under_aux_w: texture_storage_2d<rgba16float, write>;
// Bilinear clamp sampler for the region / reservoir / coverage lookups.
@group(0) @binding(5) var samp: sampler;
// The brush tip's coverage mask over brush-local [-1,1]² (R8; §6.6) — the
// reservoir texel's own footprint weight (pickup only; the canvas side uses the
// prefix-τ exposure instead).
@group(0) @binding(6) var cov_tex: texture_2d<f32>;
// The tool reservoir ping-pong: colour = (latent .xyz, per-unit opacity .w),
// aux = (carried height, _, _, _). `src` is read this dispatch, `dst`
// written; the host swaps the bind group at each pickup.
@group(0) @binding(7) var brush_src_color: texture_2d<f32>;
@group(0) @binding(8) var brush_src_aux: texture_2d<f32>;
@group(0) @binding(9) var brush_dst_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var brush_dst_aux_w: texture_storage_2d<rgba16float, write>;
// The footprint snapshot (read by `deposit`).
@group(0) @binding(11) var under_color: texture_2d<f32>;
@group(0) @binding(12) var under_aux: texture_2d<f32>;
// The region, written in place by `deposit`.
@group(0) @binding(13) var region_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var region_aux_w: texture_storage_2d<rgba16float, write>;
// The colour-dynamics noise field + its repeat sampler (deposit only): jitters
// the brush's own `add` paint exactly like the fast path's stamp (DESIGN §6.2).
@group(0) @binding(15) var dyn_noise_tex: texture_2d<f32>;
@group(0) @binding(16) var dyn_noise_samp: sampler;
// The reservoir's **swept prefix** in the travel frame (written by `bake`, read by
// `deposit`): running integrals of the carried paint weighted by the footprint's
// optical-depth density, for the full reservoir `R` and for the last pickup's gain
// parcel `G` (plain and travel-weighted — the two prefixes the linear reload ramp
// decomposes into; see the header). Packed to stay within the four storage
// textures core WebGPU guarantees a stage:
//   load = (∫h_R·dτ,      ∫m_R·dτ,      ∫h_G·dτ,      unused)
//   latm = (∫lat_R·m_R·dτ,                             ∫m_G·dτ)
//   g    = (∫lat_G·m_G·dτ,                             ∫h_G·q·dτ)
//   gq   = (∫lat_G·m_G·q·dτ,                           ∫m_G·q·dτ)
// with `q` the serving column's travel-frame coordinate (radius units). fp32
// because each fragment takes a *difference* of two prefix sums, exactly as the
// prefix-τ volume does, and for the same cancellation reason.
@group(0) @binding(17) var bake_load_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(18) var bake_latm_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(19) var bake_load: texture_2d<f32>;
@group(0) @binding(20) var bake_latm: texture_2d<f32>;
@group(0) @binding(26) var bake_g_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(27) var bake_gq_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(28) var bake_g: texture_2d<f32>;
@group(0) @binding(29) var bake_gq: texture_2d<f32>;
// The selection's coverage over the region (DESIGN.md §6.8), 1:1 with it — or a 1×1
// texture holding a constant when there is nothing to mask. Both the vertical fluxes
// are gated by it: `pickup` scales what it lifts, `deposit` lerps its whole
// read-modify-write back towards the untouched region, so paint outside the selection
// is neither taken nor laid and the two sides of the transfer still balance.
@group(0) @binding(21) var sel_mask: texture_2d<f32>;
// The last pickup's **gain parcel** (written by `pickup`, read by `bake`): what the
// tool lifted over the preceding interval, kept separately from the reservoir it
// was also added to so the bake can ramp it in across the interval instead of
// serving it as a step (see the header). Same layout as the reservoir pair:
// colour = (lifted latent .xyz, its per-unit opacity .w), aux = (height, _, _, _).
// Not ping-ponged: each pickup overwrites it whole, `deplete` leaves it alone (the
// undrained tail of a ramping gain is second-order), and nothing else writes it.
@group(0) @binding(22) var gain_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(23) var gain_aux_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(24) var gain_color: texture_2d<f32>;
@group(0) @binding(25) var gain_aux: texture_2d<f32>;
// (The prefix-τ volume rides at group(1) binding(0) via the stamp_common import.)

const TAU: f32 = 6.283185307179586;

// Deplete a bit slower than the math would indicate to account for error. Depleting
// too slowly feels better than depleting too quickly.
// TODO: Compute the right value for this from the current parameters.
const DEPLETE_FACTOR: f32 = 0.75;

// How much of a *constant deposition rate* survives on the canvas at the end of a
// segment of exposure `e`: `∫₀ᵉ exp(λ_lift·(e−s)) ds`, the amount laid at each
// point of the pass discounted by the lift still acting on it for the rest of the
// pass. Reduces to `e` when the brush does not lift.
//
// This is what makes the exchange **independent of how the path was cut into
// segments** — the property adaptive flattening (`path::flatten`) depends on and
// uniform 2px sampling used to hide. Within a segment the exchange is an ODE,
// `dh/de = λ_lift·h + source`; the canvas-side `lift` alone composes exactly
// (`h·exp(λ·e)` is multiplicative), but a source term added *after* the lift is a
// one-step Euler integration of it, and one coarse step is not N fine ones. Using
// the ODE's own solution instead makes any cut of the path agree exactly:
//
//     h·A₁ + B₁ then ·A₂ + B₂  ==  h·A₁A₂ + B₁₊₂     with A = exp(λ·e), B = k·W(e)
//
// Getting this wrong is visible, not subtle: the previous `1 − exp(λ_dep·e)` per
// segment dumped the tool's load in the first quad that touched a spot, so a
// carrying stroke ran dry early and showed arcs at the segment spacing. It only
// shows once the tool is actually carrying paint — the `add` axis was already a
// bare `e`, correct only because it had no lift to survive.
//
// This is the uniform-reservoir case of what `bake` integrates: with `R ≡ 1`, the
// swept prefix there evaluates to exactly this window. The two are one expression
// seen from either end — which is why the deposit multiplies them together and
// nothing else is needed to make a pass composable.
fn laid_window(e: f32) -> f32 {
    let k = -st.b.z; // the lift rate; 0 when the brush does not lift
    if k < 1e-6 {
        return e;
    }
    return (1.0 - exp(-k * e)) / k;
}

// The footprint's **optical depth** at brush-local `local` — the currency every
// transfer in this loop is measured in, on both sides. The canvas side gets it from
// the prefix-τ volume's differences; the tool side, which has no prefix to
// difference, converts its own coverage the same way the volume was built
// (`assets::build_prefix_tau`), clamp included, so the two agree.
fn tau_at(local: vec2<f32>) -> f32 {
    return -log(1.0 - min(coverage_at(local), 0.999));
}

// Footprint coverage at brush-local `local` ∈ [-1,1]² (0 outside — image masks may
// have non-zero borders, so the clamp sampler alone can't be trusted at the edge).
fn coverage_at(local: vec2<f32>) -> f32 {
    if any(abs(local) >= vec2<f32>(1.0, 1.0)) {
        return 0.0;
    }
    return textureSampleLevel(cov_tex, samp, local * 0.5 + vec2<f32>(0.5, 0.5), 0.0).r;
}

// (`blend_latent` — the parcel-stacking law — is shared with the transform's
// combine and lives in paint_common.wesl.)

// Bilinear lookup of a baked prefix row at travel-frame `(x, y)`, with the same
// asymmetric edge semantics as `prefix_at`: 0 before the tip's leading edge, and
// clamped to the row total after its trailing edge. Hand-rolled on `textureLoad`
// for the same reasons the prefix-τ lookup is (fp32 is not filterable in core
// WebGPU, and the travel axis is asymmetric).
fn bake_at(tex: texture_2d<f32>, x: f32, y: f32) -> vec4<f32> {
    if x <= -1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let dim = vec2<f32>(textureDimensions(tex));
    let ux = clamp(x * 0.5 + 0.5, 0.0, 1.0) * dim.x - 0.5;
    let uy = clamp(y * 0.5 + 0.5, 0.0, 1.0) * dim.y - 0.5;
    let x0 = floor(ux);
    let y0 = floor(uy);
    let fx = ux - x0;
    let fy = uy - y0;
    let xi0 = clamp(i32(x0), 0, i32(dim.x) - 1);
    let xi1 = clamp(i32(x0) + 1, 0, i32(dim.x) - 1);
    let yi0 = clamp(i32(y0), 0, i32(dim.y) - 1);
    let yi1 = clamp(i32(y0) + 1, 0, i32(dim.y) - 1);
    let p00 = textureLoad(tex, vec2<i32>(xi0, yi0), 0);
    let p10 = textureLoad(tex, vec2<i32>(xi1, yi0), 0);
    let p01 = textureLoad(tex, vec2<i32>(xi0, yi1), 0);
    let p11 = textureLoad(tex, vec2<i32>(xi1, yi1), 0);
    return mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);
}

// ---- 0. Bake: the reservoir's swept prefix along the travel axis. ----------------
//
// During a segment the tip does not present one texel to a canvas point — it drags
// a whole *range* of texels across it. A point at travel-frame `(xl, yl)` is served
// by tool texel `(xl − s, yl)` as the tip advances by `s`, so what it receives is
//
//     ∫₀^lr R(xl − s, yl) · τ(xl − s, yl) ds
//
// a convolution along travel between the reservoir `R` and the footprint's
// optical-depth density. Sampling `R` once mid-pass (what this used to do) is a
// one-tap midpoint rule for that integral: first order, with error growing in
// segment length, which is why it could only ever be traded against step size.
//
// The integral has the same shape as the exposure itself, so it yields to the same
// trick: **`prefix_at` is this integral with `R ≡ 1`**. Accumulate `R·dτ` along the
// row and the deposit becomes a difference of two prefix sums — O(1), and exact for
// any segment length. Baked per segment, so the tip's orientation (the reservoir
// lives in the shape frame, §6.6) and the reservoir's own evolution are both folded
// in with no approximation. `dτ` comes from the prefix-τ volume's own differences,
// so bake and exposure integrate numerically identical footprints.
//
// One **workgroup per row**, one thread per column, with the running integral
// built by a shared-memory inclusive scan (Hillis–Steele, log₂(BAKE_RES) steps).
// The loop this replaced — one thread walking its whole row — put 128 threads on
// the device per dispatch and a 128-iteration serial chain of texture reads on
// each, and since the bake runs once per segment inside the sequential stamp
// loop, that near-idle dispatch dominated the whole dynamics path. The scan
// computes the same prefix sums (up to float re-association) with every lane of
// a full-width dispatch busy.

// Must equal the host's BAKE_RES (the bake textures' side).
const BAKE_RES: u32 = 128u;

var<workgroup> ws_e: array<f32, BAKE_RES>;
var<workgroup> ws_load: array<vec4<f32>, BAKE_RES>;
var<workgroup> ws_latm: array<vec4<f32>, BAKE_RES>;
var<workgroup> ws_g: array<vec4<f32>, BAKE_RES>;
var<workgroup> ws_gq: array<vec4<f32>, BAKE_RES>;

@compute @workgroup_size(128) // = BAKE_RES: thread i scans column i of its row
fn bake(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let dim = textureDimensions(bake_load_w);
    let row = wid.y; // dispatched as (1, dim.y, 1), so every row is in range
    let i = lid.x;
    let yl = (f32(row) + 0.5) / f32(dim.y) * 2.0 - 1.0;
    let theta = st.d.z * TAU;
    let cs = cos(theta);
    let sn = sin(theta);

    // This column's share of the exposure: the prefix-τ difference across it.
    // Each thread reads its right edge; the left edge is the neighbour's.
    let x1 = (f32(i) + 1.0) / f32(dim.x) * 2.0 - 1.0;
    ws_e[i] = prefix_at(x1, yl, st.d.z);
    workgroupBarrier();
    var de = ws_e[i];
    if i > 0u {
        de = de - ws_e[i - 1u];
    }
    de = max(de, 0.0);

    // The tool texel over this column. The reservoir is stored in the shape
    // frame, so rotate out of travel first.
    let xc = (f32(i) + 0.5) / f32(dim.x) * 2.0 - 1.0;
    let sl = vec2<f32>(xc * cs + yl * sn, -xc * sn + yl * cs);
    let buv = clamp(sl * 0.5 + vec2<f32>(0.5, 0.5), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let b_c = textureSampleLevel(brush_src_color, samp, buv, 0.0);
    let b_a = textureSampleLevel(brush_src_aux, samp, buv, 0.0);
    let g_c = textureSampleLevel(gain_color, samp, buv, 0.0);
    let g_a = textureSampleLevel(gain_aux, samp, buv, 0.0);

    // Carried height and optical mass (opacity·height), each weighted by the
    // exposure they are presented with; latent rides as a mass weight so a
    // ratio of the two prefixes recovers it (§6.1). The gain parcel gets the
    // same treatment twice — plain and travel-weighted (`xc`) — which is what
    // lets the deposit evaluate the linear reload ramp as prefix differences.
    let m = b_c.a * b_a.x;
    let mg = g_c.a * g_a.x;
    ws_load[i] = vec4<f32>(b_a.x, m, g_a.x, 0.0) * de;
    ws_latm[i] = vec4<f32>(b_c.rgb * m, mg) * de;
    ws_g[i] = vec4<f32>(g_c.rgb * mg, g_a.x * xc) * de;
    ws_gq[i] = vec4<f32>(g_c.rgb * (mg * xc), mg * xc) * de;
    workgroupBarrier();

    // Inclusive scan over the four accumulators together.
    for (var off = 1u; off < BAKE_RES; off = off << 1u) {
        var v_load = ws_load[i];
        var v_latm = ws_latm[i];
        var v_g = ws_g[i];
        var v_gq = ws_gq[i];
        if i >= off {
            v_load += ws_load[i - off];
            v_latm += ws_latm[i - off];
            v_g += ws_g[i - off];
            v_gq += ws_gq[i - off];
        }
        workgroupBarrier();
        ws_load[i] = v_load;
        ws_latm[i] = v_latm;
        ws_g[i] = v_g;
        ws_gq[i] = v_gq;
        workgroupBarrier();
    }

    // Out-of-range stores (a texture narrower than the workgroup) are no-ops.
    let t = vec2<i32>(i32(i), i32(row));
    textureStore(bake_load_w, t, ws_load[i]);
    textureStore(bake_latm_w, t, ws_latm[i]);
    textureStore(bake_g_w, t, ws_g[i]);
    textureStore(bake_gq_w, t, ws_gq[i]);
}

// ---- 1. Snapshot the region under the segment's footprint rect. -----------------

@compute @workgroup_size(8, 8)
fn snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_color_w);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let rt = vec2<i32>(st.d.xy) + t;
    let rdim = vec2<i32>(textureDimensions(region_color));
    var c = vec4<f32>(0.0);
    var a = vec4<f32>(0.0);
    if rt.x >= 0 && rt.y >= 0 && rt.x < rdim.x && rt.y < rdim.y {
        c = textureLoad(region_color, rt, 0);
        a = textureLoad(region_aux, rt, 0);
    }
    textureStore(under_color_w, t, c);
    textureStore(under_aux_w, t, a);
}

// ---- 2. Pickup: canvas → tool, plus the tool's own depletion. --------------------

@compute @workgroup_size(8, 8)
fn pickup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bdim = textureDimensions(brush_src_color);
    if gid.x >= bdim.x || gid.y >= bdim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    // This reservoir texel's brush-local position and its spot on the canvas.
    let local = (vec2<f32>(t) + vec2<f32>(0.5, 0.5)) / vec2<f32>(bdim) * 2.0 - vec2<f32>(1.0, 1.0);
    let rot = st.a.zw;
    let world = st.a.xy
        + (local.x * rot + local.y * vec2<f32>(-rot.y, rot.x)) * st.b.x;

    // This texel's exposure over the interval since the last pickup: its own
    // optical depth × the travel (radius units). In τ, like the canvas side and
    // like `deplete` — the canvas gives up `1 − exp(λ_lift·∫τ)`, so a tool that
    // measured its gain in bare coverage would receive a fraction of what the
    // canvas lost and quietly destroy paint.
    let e = tau_at(local) * st.d.w;

    let rdim = vec2<f32>(textureDimensions(region_color));
    // Outside the selection the canvas gives nothing up (§6.8) — the matching
    // reduction on the canvas side is the deposit's own mask lerp, so height stays
    // conserved. Bilinear like the region taps below: this texel is over an arbitrary
    // sub-pixel spot, and a 1×1 constant mask reads the same at every uv.
    let lift = (1.0 - exp(st.b.z * e))
        * textureSampleLevel(sel_mask, samp, world / rdim, 0.0).r;

    let rc = textureSampleLevel(region_color, samp, world / rdim, 0.0);
    let ra = textureSampleLevel(region_aux, samp, world / rdim, 0.0);

    let b_c = textureLoad(brush_src_color, t, 0); // (latent, op) — straight, not premult
    let b_a = textureLoad(brush_src_aux, t, 0);   // (height, _, _, _)

    // Only the canvas→tool direction happens here. The tool's own draining is per
    // *segment* (`deplete`), because it has to advance with travel for the exchange
    // to be independent of how the path was cut — at pickup cadence, every segment
    // in an interval would deposit from the same undepleted load.
    var h = b_a.x;
    var m = b_c.a * h;

    // Fold in the paint lifted off the canvas: height moves, colour/opacity
    // ride as optical-mass-weighted blends (§6.1).
    let take_h = ra.x * lift;
    let take_m = rc.a * take_h;
    let canvas_lat = rc.rgb / max(rc.a, EPS);
    let lat = blend_latent(b_c.rgb, m, canvas_lat, take_m);
    h += take_h;
    m += take_m;

    textureStore(brush_dst_color_w, t, vec4<f32>(lat, m / max(h, EPS)));
    textureStore(brush_dst_aux_w, t, vec4<f32>(h, 0.0, 0.0, 0.0));
    // The reload just folded in, again as its own parcel — the bake reads it back
    // out of the reservoir sum so the deposit can ramp it in across the coming
    // interval instead of serving it as a step (see the header). Same convention
    // as the reservoir pair: straight latent + per-unit opacity, mass = op·height.
    textureStore(gain_color_w, t, vec4<f32>(canvas_lat, rc.a));
    textureStore(gain_aux_w, t, vec4<f32>(take_h, 0.0, 0.0, 0.0));
}

// ---- 2b. Deplete: the tool drains as it gives paint up, per segment. ------------
//
// The counterpart to what `deposit` lays. Multiplicative in the texel's own
// exposure to the segment (its footprint weight × the travel), so the tool's state
// after a run of segments is `exp(λ_dep · Σeᵢ)` — the same wherever the path was
// cut. Doing this at pickup cadence instead is what left a carrying stroke
// depositing the same undepleted load once per segment and scalloping between
// reloads. Latent and per-unit opacity are untouched: the tool holds *less* paint,
// not different paint (§6.1).
@compute @workgroup_size(8, 8)
fn deplete(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bdim = textureDimensions(brush_src_color);
    if gid.x >= bdim.x || gid.y >= bdim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let local = (vec2<f32>(t) + vec2<f32>(0.5, 0.5)) / vec2<f32>(bdim) * 2.0 - vec2<f32>(1.0, 1.0);
    // In optical depth, like every other transfer here: `deposit` integrates the
    // tool's decay over the canvas-side exposure, which is the prefix-τ integral, so
    // a tool draining in bare coverage would disagree with it by a factor that grows
    // with segment length — the very partition dependence this dispatch removes.
    let keep = exp(DEPLETE_FACTOR * st.b.w * tau_at(local) * st.b.y);
    let b_c = textureLoad(brush_src_color, t, 0);
    let b_a = textureLoad(brush_src_aux, t, 0);
    textureStore(brush_dst_color_w, t, b_c);
    textureStore(brush_dst_aux_w, t, vec4<f32>(b_a.x * keep, 0.0, 0.0, 0.0));
}

// ---- 3. Deposit: the swept exchange over one segment. ----------------------------

@compute @workgroup_size(8, 8)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_color);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let rt = vec2<i32>(st.d.xy) + t;
    let rdim = vec2<i32>(textureDimensions(region_color_w));
    if rt.x < 0 || rt.y < 0 || rt.x >= rdim.x || rt.y >= rdim.y {
        return;
    }
    // Travel-frame brush-local coords (x in radius units from the segment start).
    let p = vec2<f32>(rt) + vec2<f32>(0.5, 0.5);
    let rel = (p - st.a.xy) / st.b.x;
    let dir = st.a.zw;
    let xl = rel.x * dir.x + rel.y * dir.y;
    let yl = -rel.x * dir.y + rel.y * dir.x;

    // The swept exposure — the brush's coverage integrated as it travels this
    // segment, as a difference of prefix sums (identical to the plain deposit's
    // `swept_depth`). Overlapping segment quads add their exposures, so the
    // exponential rates below compose exactly across the whole stroke.
    let e = max(prefix_at(xl, yl, st.d.z) - prefix_at(xl - st.b.y, yl, st.d.z), 0.0);
    if e <= 0.0 {
        return; // outside the swept footprint: the region keeps its value
    }
    // The selection's coverage here (§6.8). Fully deselected texels are exactly the
    // "keeps its value" case, so they cost nothing.
    let mdim = vec2<i32>(textureDimensions(sel_mask));
    let sel = textureLoad(sel_mask, min(rt, mdim - vec2<i32>(1, 1)), 0).r;
    if sel <= 0.0 {
        return;
    }
    let lift = 1.0 - exp(st.b.z * e);
    // The brush's own `add` is a *constant* source, so it rides the plain window.
    let laid = laid_window(e);
    // The tool's is not: it drains as it gives paint up (`deplete`), while what it
    // has already laid is itself being lifted. Over the segment that is a pair of
    // coupled linear ODEs — `dR/de = λ_dep·R`, `dh/de = λ_lift·h + k_dep·R` — whose
    // solution is the fraction of the *entering* load that ends up on the canvas:
    //
    //     k_dep · (exp(λ_dep·e) − exp(λ_lift·e)) / (λ_dep − λ_lift)
    //
    // Being the exact solution, it composes over any cut of the path, and it is
    // bounded by 1 on its own — a segment cannot lay more than the tool brought,
    // which is what a per-segment rate applied to an undepleted reservoir could.
    let ld = st.b.w;
    let ll = st.b.z;
    let split = ld - ll;
    var dep = -ld * e * exp(ll * e); // λ_dep → λ_lift limit
    if abs(split) > 1e-4 {
        dep = -ld * (exp(ld * e) - exp(ll * e)) / split;
    }

    let uc = textureLoad(under_color, t, 0);
    let ua = textureLoad(under_aux, t, 0);
    // The tool's carried paint over this spot, as the tip actually presented it:
    // the swept prefix differenced across the segment (see `bake`) gives the
    // exposure-weighted integral of the reservoir over the whole pass, so dividing
    // by the exposure recovers the mean load that pass delivered. Exact for any
    // segment length — no mid-pass sample, no length-dependent error.
    let d_load = bake_at(bake_load, xl, yl) - bake_at(bake_load, xl - st.b.y, yl);
    let d_latm = bake_at(bake_latm, xl, yl) - bake_at(bake_latm, xl - st.b.y, yl);
    let d_g = bake_at(bake_g, xl, yl) - bake_at(bake_g, xl - st.b.y, yl);
    let d_gq = bake_at(bake_gq, xl, yl) - bake_at(bake_gq, xl - st.b.y, yl);
    // The reload ramp (see the header): the last pickup's gain parcel is served as
    // `G·(γ − ½)` on top of the reservoir it was already added to, with
    // `γ = (t0 + s)/L` the fraction of the pickup interval elapsed when the column
    // at `q` served this texel. `s = xl − q` is linear in `q`, so the integral
    // splits into the plain and travel-weighted `G` prefixes baked above:
    //     Σ G·(γ − ½)·dτ = ((t0 + xl)/L − ½)·ΔG − ΔGq/L.
    // Mean-preserving by construction — over a whole interval the ramp serves
    // exactly the stepped amount — so this changes the seams, not the stroke.
    let rc0 = (st.e.z + xl) / st.e.w - 0.5;
    let rc1 = 1.0 / st.e.w;
    // Height and mass are non-negative and the ramp is a fraction of the parcel
    // already inside the reservoir sum, so a total below zero can only be
    // rounding. The *latent* is not: these are colour-space channels (Oklab a/b
    // are signed — green carries a negative a), so its prefix is not monotonic and
    // clamping the difference would shear the chroma off whatever the tool is
    // carrying.
    let inv_e = 1.0 / e;
    let tool_h = max(d_load.x + rc0 * d_load.z - rc1 * d_g.w, 0.0) * inv_e;
    let tool_m = max(d_load.y + rc0 * d_latm.w - rc1 * d_gq.w, 0.0) * inv_e;
    let latm = d_latm.xyz + rc0 * d_g.xyz - rc1 * d_gq.xyz;
    let tool_op = tool_m / max(tool_h, EPS);
    let tool_lat = latm * inv_e / max(tool_m, EPS);

    // The lift's removal: height leaves; the remaining paint's latent and per-unit
    // opacity are unchanged — the source fades because its *thickness* drops,
    // never because its alpha is scaled (§6.1).
    var h = ua.x * (1.0 - lift);
    var m = uc.a * h;
    let canvas_lat = uc.rgb / max(uc.a, EPS);

    // The brush's own colour at this texel: the stroke colour offset by the
    // colour-dynamics field, sampled at the same stroke-local (lateral, arc)
    // domain as the fast path — both in canvas px (the travel frame's y is in
    // radius units, so it scales by the radius).
    var own_lat = st.c.xyz;
    if any(st.g.xyz != vec3<f32>(0.0, 0.0, 0.0)) {
        let lateral = yl * st.b.x;
        let arc = st.g.w + xl * st.b.x;
        own_lat += color_jitter(
            dyn_noise_tex, dyn_noise_samp, vec2<f32>(lateral, arc),
            st.f, st.g.xyz, st.h.xy,
        );
    }

    // The parcel this segment lays: the tool's deposit + the brush's own `add`
    // paint — the latter linear in exposure, i.e. exactly the plain deposit's
    // path-integral height.
    let dep_h = tool_h * dep;
    let add_h = st.e.x * laid;
    let par_h = dep_h + add_h;
    let par_m = tool_op * dep_h + st.c.w * add_h;
    var par_lat = own_lat;
    if par_m > EPS {
        par_lat = (tool_lat * (tool_op * dep_h) + own_lat * (st.c.w * add_h)) / par_m;
    }

    let lat = blend_latent(canvas_lat, m, par_lat, par_m);
    h += par_h;
    m += par_m;
    let op = m / max(h, EPS);

    // Lerp the whole exchange back towards the snapshot by the selection coverage: at
    // the mask's feathered edge the texel receives that fraction of everything this
    // segment did — the lift as well as the deposit — which is exactly the fraction
    // `pickup` scaled its own side of the transfer by.
    textureStore(region_color_w, rt, mix(uc, vec4<f32>(lat * op, op), sel));
    textureStore(region_aux_w, rt, mix(ua, vec4<f32>(h, 0.0, 0.0, 0.0), sel));
}
