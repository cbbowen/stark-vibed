// Brush dynamics: the **sequential swept-exchange loop** (§6.2).
//
// A stroke that manipulates existing paint (lift / deposit / charge) runs as an
// ordered loop, all inside one compute pass (implicit barriers between dispatches
// give the sequencing; usage scopes are per-dispatch, so the region may be
// sampled by one dispatch and storage-written by the next — no copies):
//
//   - per flattened segment, in order: `bake` (integrate the reservoir along the
//     travel axis, so the deposit can read what the tip presented over a whole pass
//     rather than at one point), `exchange` — one thread per **tool reservoir**
//     texel, which settles the tool's half of the transfer — `snapshot` (copy the
//     segment's footprint texels to an `under` scratch, so the deposit can
//     read-modify-write the region) and `deposit`, one thread per footprint texel.
//     `exchange` ping-pongs the reservoir.
//
// **`exchange` runs before `deposit`, not after.** The two are the two halves of one
// transfer, and a transfer is a statement about a *single* pre-state: both dispatches
// evaluate the same kernel (`exchange_at`) from the same segment-entry canvas and the
// same segment-entry reservoir, and hand back complementary fractions of it. Reading
// the region as the deposit had already left it — which is what the tool used to do —
// makes the tool lift from a canvas that has, from its point of view, not happened yet,
// and the halves stop adding up (see `exchange_at`).
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
// **The tool exchanges once per segment, at the same cadence as everything else.**
// The canvas side of the transfer is swept per segment (above), so the tool side has
// to be too, or the tool gains what the canvas gave up only some segments later. That
// lag used to be real: `pickup` ran on its own coarse cadence, every `spacing·radius`
// of travel, and a stroke that stopped mid-interval left its last footprint stripped
// by a `deposit` whose paint the tool had not yet been credited with — a tip-shaped
// patch of missing paint, worst exactly where `lift` is high.
//
// So `exchange` does the whole of the tool's side in one dispatch, per segment,
// against the very same segment the `deposit` accounts for. Nothing is deferred, so
// nothing is left owing at the end of a stroke.
//
// This also retires a good deal of machinery. The old cadence had to hide its own
// discreteness — a whole interval's reload landing between two segments is a large
// fraction of the load at high lift × deposit, and consecutive segments served
// visibly different tools — which took a separate **gain parcel**, a linear ramp
// serving it in across the interval, and two extra baked prefixes to evaluate that
// ramp per fragment. A per-segment exchange has no step to hide: the segment length
// *is* the reload step, and `stroke::flatten_tolerance` already caps it at a fraction
// of the tip. All of it is gone.
//
// Conservation (§6.1): paint moves by transferring **height** — the one
// conserved quantity. Colour and per-unit opacity ride as optical-mass
// (opacity·height) weighted blends; `add` is the only source term. Both sides of
// each transfer integrate the same rate over the same footprint, so with
// `add = 0` total height (canvas + tool) is conserved up to resampling error.
//
// Determinism: segment/pickup slots are fixed CPU-side per stroke; every dispatch
// is a pure function of what it reads. Replay and live == committed hold.

import package::stamp_common::{prefix_at, Sweep, sweep_at};
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
    //    picks the prefix-τ slice) .z; .w unused.
    d: vec4<f32>,
    // e: the `add` rate per unit exposure: height .x; the segment's signed curvature
    //    (1/region px, 0 = a straight sweep) .y; the segment's midpoint .zw (region
    //    px), which is where `exchange` samples the canvas.
    e: vec4<f32>,
    // f–h: colour dynamics (§6.2) for the brush's own `add` paint —
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
// the brush's own `add` paint exactly like the fast path's stamp (§6.2).
@group(0) @binding(15) var dyn_noise_tex: texture_2d<f32>;
@group(0) @binding(16) var dyn_noise_samp: sampler;
// The reservoir's **swept prefix** in the travel frame (written by `bake`, read by
// `deposit`): running integrals of the carried paint weighted by the footprint's
// optical-depth density.
//   load = (∫h·dτ, ∫m·dτ, unused, unused)
//   latm = (∫lat·m·dτ,        unused)
// fp32 because each fragment takes a *difference* of two prefix sums, exactly as the
// prefix-τ volume does, and for the same cancellation reason. Two targets where there
// were four: the other pair only ever carried the retired reload ramp.
@group(0) @binding(17) var bake_load_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(18) var bake_latm_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(19) var bake_load: texture_2d<f32>;
@group(0) @binding(20) var bake_latm: texture_2d<f32>;
// The selection's coverage over the region (§6.8), 1:1 with it — or a 1×1
// texture holding a constant when there is nothing to mask. Both the vertical fluxes
// are gated by it: `exchange` scales what it lifts, `deposit` lerps its whole
// read-modify-write back towards the untouched region, so paint outside the selection
// is neither taken nor laid and the two sides of the transfer still balance.
@group(0) @binding(21) var sel_mask: texture_2d<f32>;
// (The prefix-τ volume rides at group(1) binding(0) via the stamp_common import.)

const TAU: f32 = 6.283185307179586;

// ---- The exchange kernel ---------------------------------------------------------
//
// Over one segment a canvas point and the tool above it are a pair of coupled boxes
// trading the one conserved quantity — height (§6.1) — at rates proportional to their
// exposure `e`, with the brush's own `add` a source into the canvas box:
//
//     dh/de = −k_lift·h + k_dep·R + A          (canvas)
//     dR/de = +k_lift·h − k_dep·R              (tool)
//
// where `k = −λ ≥ 0`. Their sum moves only by the source, so with `add = 0` the pair
// conserves height *identically*, not approximately.
//
// **Both halves of the loop evaluate this one solution**, over the same exposure and
// from the same segment-entry state, and take complementary shares of it — which is
// what makes the whole exchange **independent of how the path was cut into segments**,
// the property adaptive flattening (`path::flatten`) depends on and uniform 2px
// sampling used to hide. The system is linear and autonomous in `e`, so running it
// over `e₁` and then `e₂` is running it over `e₁+e₂`; exposures add across the
// overlapping quads of consecutive segments, so the whole stroke applies the
// continuous path integral whatever the spacing.
//
// Getting this wrong is not subtle. What was here before solved only *half* the pair —
// the canvas relaxing towards a tool that never took anything back, `1 − exp(λ_lift·e)`
// lifted off and `k_dep(exp(λ_dep·e) − exp(λ_lift·e))/(λ_dep − λ_lift)` laid on — and
// then let the tool lift from the region as the `deposit` had already left it. The two
// sides therefore disagreed about how much had changed hands by `O(lift²)` per
// segment, and paint that goes nowhere has to show:
//
//   * at `lift = deposit = 0.95` and a segment of half a radius, 39% of the total
//     `canvas + tool` height vanished at *every* segment boundary. What survives is
//     front-loaded into the first quad that touches a spot, which draws arcs at
//     exactly the segment spacing through anything thick enough to see them.
//   * a stroke that stops leaves its last footprint stripped by a lift the tool was
//     never credited with — a tip-shaped disc of missing paint, worst where `lift` is
//     high, and unmistakable because the tip's own hardness edge is frozen into it.
//
// The pair has one non-zero eigenvalue: it relaxes towards the split `k_dep : k_lift`
// at rate `s = k_lift + k_dep`. So every coefficient below is written in terms of the
// single window
//
//     w(e) = (1 − exp(−s·e)) / s
//
// which is also the plain `∫₀ᵉ exp(−k(e−u)) du` the source rides when nothing takes it
// back, and the uniform-reservoir case of what `bake` integrates: with `R ≡ 1` the
// swept prefix there evaluates to exactly this. One expression seen from three ends.

// How one segment's exposure `e` splits the two boxes. Every field is a fraction of an
// entering quantity, and the tool's share is the complement of each — so the canvas
// takes `keep·h + dep·R + add_w·A` and the tool takes `(1−keep)·h + (1−dep)·R + src_w·A`,
// which sum to `h + R + e·A`: everything the pair started with, plus everything the
// source minted, and not a texel of height more or less.
struct Exchange {
    // Of the canvas's own height, what is still on the canvas.
    keep: f32,
    // Of the tool's entering load, what is now on the canvas.
    dep: f32,
    // Canvas height laid per unit of the brush's own `add` rate…
    add_w: f32,
    // …and what the tool lifted straight back off it, so that `add_w + src_w == e`.
    src_w: f32,
}

fn exchange_at(e: f32) -> Exchange {
    let kl = -st.b.z; // the lift rate; 0 when the brush does not lift
    let kd = -st.b.w; // the deposit rate
    let s = kl + kd;
    var out: Exchange;
    if s < 1e-6 {
        // Neither box trades, so the source is all the canvas gets — and it keeps it.
        out.keep = 1.0;
        out.dep = 0.0;
        out.add_w = e;
        out.src_w = 0.0;
        return out;
    }
    let w = (1.0 - exp(-s * e)) / s;
    out.keep = 1.0 - kl * w;
    out.dep = kd * w;
    // `kl/s ≤ 1` bounds the second term, so no division here can run away as `s → 0`;
    // the branch above is for exactness, not for conditioning.
    out.add_w = (kd * e + kl * w) / s;
    out.src_w = kl * (e - w) / s;
    return out;
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

    // Carried height and optical mass (opacity·height), each weighted by the
    // exposure they are presented with; latent rides as a mass weight so a
    // ratio of the two prefixes recovers it (§6.1).
    let m = b_c.a * b_a.x;
    ws_load[i] = vec4<f32>(b_a.x, m, 0.0, 0.0) * de;
    ws_latm[i] = vec4<f32>(b_c.rgb * m, 0.0) * de;
    workgroupBarrier();

    // Inclusive scan over both accumulators together.
    for (var off = 1u; off < BAKE_RES; off = off << 1u) {
        var v_load = ws_load[i];
        var v_latm = ws_latm[i];
        if i >= off {
            v_load += ws_load[i - off];
            v_latm += ws_latm[i - off];
        }
        workgroupBarrier();
        ws_load[i] = v_load;
        ws_latm[i] = v_latm;
        workgroupBarrier();
    }

    // Out-of-range stores (a texture narrower than the workgroup) are no-ops.
    let t = vec2<i32>(i32(i), i32(row));
    textureStore(bake_load_w, t, ws_load[i]);
    textureStore(bake_latm_w, t, ws_latm[i]);
}

// ---- 1. Snapshot the region under the segment's footprint rect. -----------------

// This texel's place in the segment's travel frame — the one the whole exchange is
// measured in. Straight or curved, `sweep_at` hands back the same three quantities
// (see stamp_common.wesl), so nothing downstream branches on the segment's shape.
//
// `st.b.y` carries the travel as `length / radius`, so the arc length it was built
// from is recovered by multiplying back through.
fn segment_frame(rt: vec2<i32>) -> Sweep {
    let p = vec2<f32>(rt) + vec2<f32>(0.5, 0.5);
    return sweep_at(p, st.a.xy, st.a.zw, st.b.x, st.b.y * st.b.x, st.e.y);
}

// Whether a region texel is provably outside the segment's sweep — the span
// x ∈ (-1, lr+1), |y| < 1 in the travel frame — so its exposure is exactly 0 and
// `deposit` leaves it untouched. The dispatch rect is the sweep's AABB, so at a
// diagonal orientation up to half its texels are these corners; the test rejects
// them with a few ALU ops instead of the prefix-τ taps `deposit` would otherwise
// spend proving e = 0. `snapshot` may skip exactly the same texels because
// `deposit` reads the snapshot only after its own e > 0 check, and the two compute
// the identical frame from the identical uniform — same floats, same verdict.
fn outside_sweep(sw: Sweep) -> bool {
    // Beyond the lateral edges the prefix rows are the footprint's own zero
    // border; past the trailing edge both taps clamp to the row total. Either
    // way the difference is exactly 0. The lateral test comes first because it is
    // also what rejects a texel near a tight arc's centre of curvature, where the
    // travel coordinate has no meaning to test.
    return abs(sw.local.y) >= 1.0 || sw.local.x <= -1.0 || sw.local.x - sw.lr >= 1.0;
}

@compute @workgroup_size(8, 8)
fn snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_color_w);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let rt = vec2<i32>(st.d.xy) + t;
    if outside_sweep(segment_frame(rt)) {
        return; // deposit never reads these texels of the snapshot
    }
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

// ---- 2. Exchange: the tool's own side of the transfer, per segment. -------------
//
// One thread per **tool reservoir** texel, run *before* the `deposit` it is the other
// half of, so that both read the canvas and the reservoir as the segment found them.
// It takes the complement of every share `exchange_at` hands the canvas:
//
//   * **retain** `1 − dep` of its entering load — the rest the deposit has on the
//     canvas. Latent and per-unit opacity are untouched: the tool holds *less* paint,
//     not different paint (§6.1).
//   * **lift** `1 − keep` of what is under it — which is what makes smearing
//     genuinely order-dependent, including across the stroke's own trail, since the
//     region it reads is the one earlier segments have already worked over.
//   * **take back** `src_w` of the brush's own `add`, the share of the newly minted
//     paint the tip lifts straight off again. Without it a high-`lift` brush would
//     mint `A·e` of height, lay `add_w·A` and destroy the difference.
//
// All three share the one exposure, and it is the *same* segment travel the canvas
// side integrates over, so nothing the canvas gave up waits to be credited. That is
// the whole reason this runs per segment rather than on a cadence of its own: the
// tool used to be reloaded every `spacing·radius` while the canvas was stripped every
// segment, and a stroke ending mid-interval left the difference behind as a
// tip-shaped patch of missing paint.
//
// The canvas is sampled at the segment's **midpoint** (`st.e.zw`) rather than either
// end — the midpoint rule for a lift that is really swept along the segment, and
// second-order where an endpoint would be first.
@compute @workgroup_size(8, 8)
fn exchange(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bdim = textureDimensions(brush_src_color);
    if gid.x >= bdim.x || gid.y >= bdim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let local = (vec2<f32>(t) + vec2<f32>(0.5, 0.5)) / vec2<f32>(bdim) * 2.0 - vec2<f32>(1.0, 1.0);

    // This texel's exposure to the segment: its own optical depth × the travel
    // (radius units). In τ, like the canvas side — the canvas gives up
    // `1 − keep(∫τ)`, so a tool that measured its gain in bare coverage would
    // receive a fraction of what the canvas lost and quietly destroy paint.
    let e = tau_at(local) * st.b.y;
    let ex = exchange_at(e);

    let b_c = textureLoad(brush_src_color, t, 0); // (latent, op) — straight, not premult
    let b_a = textureLoad(brush_src_aux, t, 0);   // (height, _, _, _)

    // What it keeps of its own load. Exactly the complement of the fraction the
    // `deposit` after us lays from it, over the very exposure it lays over — so there
    // is no correction factor.
    //
    // There used to be one (`DEPLETE_FACTOR = 0.75`, "deplete a bit slower than the
    // math would indicate to account for error"). The error it was accounting for was
    // real but misdiagnosed: the two sides were solving different equations, so they
    // disagreed about how much had changed hands and the tool over-drained. Draining
    // slower papered over that — but a factor below 1 means the tool keeps a fraction
    // of every deposit it makes, which compounds, so paint grew without bound over a
    // long stroke. With both sides now taking complementary shares of one solution
    // there is nothing left to correct, and the exact rate is also the conserving one.
    var h = b_a.x * (1.0 - ex.dep);
    var m = b_c.a * h;

    // Then lift. The tool's shape frame is the travel frame turned by the shape's
    // orientation (§6.6) — the same rotation `bake` applies to read the reservoir.
    let theta = st.d.z * TAU;
    let cs = cos(theta);
    let sn = sin(theta);
    let dir = st.a.zw;
    let rot = vec2<f32>(dir.x * cs - dir.y * sn, dir.x * sn + dir.y * cs);
    let world = st.e.zw + (local.x * rot + local.y * vec2<f32>(-rot.y, rot.x)) * st.b.x;

    let rdim = vec2<f32>(textureDimensions(region_color));
    // Outside the selection the canvas gives nothing up and the source mints nothing
    // (§6.8) — the matching reduction on the canvas side is the deposit's own mask
    // lerp, so height stays conserved. Bilinear like the region taps below: this texel
    // is over an arbitrary sub-pixel spot, and a 1×1 constant mask reads the same at
    // every uv.
    let sel = textureSampleLevel(sel_mask, samp, world / rdim, 0.0).r;

    let rc = textureSampleLevel(region_color, samp, world / rdim, 0.0);
    let ra = textureSampleLevel(region_aux, samp, world / rdim, 0.0);

    // Height moves; colour and per-unit opacity ride as optical-mass-weighted
    // blends (§6.1). Two parcels land on the tool — what it took off the canvas, and
    // the share of the brush's own `add` it lifted back before that paint had settled
    // — each stacked with its own latent, so a tip smearing red through white carries
    // the mixture rather than an average of the two errands.
    let take_h = ra.x * (1.0 - ex.keep) * sel;
    let take_m = rc.a * take_h;
    let canvas_lat = rc.rgb / max(rc.a, EPS);
    var lat = blend_latent(b_c.rgb, m, canvas_lat, take_m);
    h += take_h;
    m += take_m;

    let src_h = st.e.x * ex.src_w * sel;
    let src_m = st.c.w * src_h;
    lat = blend_latent(lat, m, st.c.xyz, src_m);
    h += src_h;
    m += src_m;

    textureStore(brush_dst_color_w, t, vec4<f32>(lat, m / max(h, EPS)));
    textureStore(brush_dst_aux_w, t, vec4<f32>(h, 0.0, 0.0, 0.0));
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
    // The AABB's corners are outside the sweep and provably at zero exposure
    // (`outside_sweep`): reject them before the prefix-τ taps below spend ~10 loads
    // apiece computing that zero.
    let sw = segment_frame(rt);
    if outside_sweep(sw) {
        return;
    }
    let xl = sw.local.x;
    let yl = sw.local.y;

    // The swept exposure — the brush's coverage integrated as it travels this
    // segment, as a difference of prefix sums (identical to the plain deposit's
    // `swept_depth`). Overlapping segment sweeps add their exposures, so the
    // exponential rates below compose exactly across the whole stroke.
    //
    // `dpre` is that difference in *this texel's own* measure of travel and `e` is it
    // in the tip's, which on a curved segment differ by `sw.warp` (see `sweep_at`).
    // Both are needed: every exchange rate below is per unit tip travel, but the bake
    // was integrated against the same prefix and its ratios must be taken in the
    // measure it was integrated in — where the factor cancels rather than squaring.
    let dpre = max(prefix_at(xl, yl, st.d.z) - prefix_at(xl - sw.lr, yl, st.d.z), 0.0);
    if dpre <= 0.0 {
        return; // outside the swept footprint: the region keeps its value
    }
    let e = dpre * sw.warp;
    // The selection's coverage here (§6.8). Fully deselected texels are exactly the
    // "keeps its value" case, so they cost nothing.
    let mdim = vec2<i32>(textureDimensions(sel_mask));
    let sel = textureLoad(sel_mask, min(rt, mdim - vec2<i32>(1, 1)), 0).r;
    if sel <= 0.0 {
        return;
    }
    // The canvas's share of the segment's transfer (`exchange_at`), of which the tool
    // took the complement in the dispatch just before this one — off the same region
    // and the same reservoir, both as the segment found them.
    let ex = exchange_at(e);

    let uc = textureLoad(under_color, t, 0);
    let ua = textureLoad(under_aux, t, 0);
    // The tool's carried paint over this spot, as the tip actually presented it:
    // the swept prefix differenced across the segment (see `bake`) gives the
    // exposure-weighted integral of the reservoir over the whole pass, so dividing
    // by the exposure recovers the mean load that pass delivered. Exact for any
    // segment length — no mid-pass sample, no length-dependent error.
    let d_load = bake_at(bake_load, xl, yl) - bake_at(bake_load, xl - sw.lr, yl);
    let d_latm = bake_at(bake_latm, xl, yl) - bake_at(bake_latm, xl - sw.lr, yl);
    // Divided by the prefix difference in the *same* measure the bake integrated
    // against, not by the tip-travel exposure: these three are means of the load the
    // pass delivered, so the curved sweep's warp cancels between numerator and
    // denominator instead of appearing twice.
    //
    // Height and mass are non-negative, so a total below zero can only be rounding.
    // The *latent* is not: these are colour-space channels (Oklab a/b are signed —
    // green carries a negative a), so its prefix is not monotonic and clamping the
    // difference would shear the chroma off whatever the tool is carrying.
    let inv_pre = 1.0 / dpre;
    let tool_h = max(d_load.x, 0.0) * inv_pre;
    let tool_m = max(d_load.y, 0.0) * inv_pre;
    let latm = d_latm.xyz;
    let tool_op = tool_m / max(tool_h, EPS);
    let tool_lat = latm * inv_pre / max(tool_m, EPS);

    // The lift's removal: height leaves; the remaining paint's latent and per-unit
    // opacity are unchanged — the source fades because its *thickness* drops,
    // never because its alpha is scaled (§6.1).
    var h = ua.x * ex.keep;
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
    let dep_h = tool_h * ex.dep;
    let add_h = st.e.x * ex.add_w;
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
