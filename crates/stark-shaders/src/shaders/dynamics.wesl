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
//   - per flattened segment: `snapshot` (copy the segment's footprint texels to
//     an `under` scratch, so the exchange can read-modify-write the region), then
//     `deposit` — one thread per footprint texel.
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
// Conservation (DESIGN.md §6.1): paint moves by transferring **height** — the one
// conserved quantity. Colour and per-unit opacity ride as optical-mass
// (opacity·height) weighted blends; `add` is the only source term. Both sides of
// each transfer integrate the same rate over the same footprint, so with
// `add = 0` total height (canvas + tool) is conserved up to resampling error.
//
// Determinism: segment/pickup slots are fixed CPU-side per stroke; every dispatch
// is a pure function of what it reads. Replay and live == committed hold.

import package::stamp_common::{prefix_at};

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
    // e: `add` rates per unit exposure: height .x, wet .y. .zw unused.
    e: vec4<f32>,
}

@group(0) @binding(0) var<uniform> st: Stamp;

// The stroke's canvas region: premultiplied colour (latent·op, op) + wide aux
// (height, wet, _, _). Read by snapshot (textureLoad) and pickup (bilinear).
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
// aux = (carried height, carried wet, _, _). `src` is read this dispatch, `dst`
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
// (The prefix-τ volume rides at group(1) binding(0) via the stamp_common import.)

const EPS: f32 = 1e-5;
const TAU: f32 = 6.283185307179586;
// Matches the media pass's translucent-slab law `1 − exp(−K·opacity·thickness)`
// (media_common.wesl), so the deposit's covering behaviour agrees with what the
// viewer sees: a parcel's blend weight is its visible alpha.
const OPACITY_K: f32 = 1.0;

// Footprint coverage at brush-local `local` ∈ [-1,1]² (0 outside — image masks may
// have non-zero borders, so the clamp sampler alone can't be trusted at the edge).
fn coverage_at(local: vec2<f32>) -> f32 {
    if any(abs(local) >= vec2<f32>(1.0, 1.0)) {
        return 0.0;
    }
    return textureSampleLevel(cov_tex, samp, local * 0.5 + vec2<f32>(0.5, 0.5), 0.0).r;
}

// Blend a parcel of paint (latent `lat1`, optical mass `m1` = opacity·height) onto
// resident paint (`lat0`, `m0`). The parcel lands on top, so its weight is its own
// visible alpha and the resident paint shows through the remainder — thick deposits
// cover, thin glazes tint, and for small masses this reduces to mass-weighted
// mixing. Falls back to the resident latent when both masses vanish.
fn blend_latent(lat0: vec3<f32>, m0: f32, lat1: vec3<f32>, m1: f32) -> vec3<f32> {
    let w1 = 1.0 - exp(-OPACITY_K * m1);
    let w0 = (1.0 - w1) * (1.0 - exp(-OPACITY_K * m0));
    let w = w0 + w1;
    if w <= EPS {
        return lat0;
    }
    return (lat0 * w0 + lat1 * w1) / w;
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
    // footprint weight × the travel (radius units). The same exponential-in-
    // exposure law as the canvas side, so depletion matches what the segment
    // deposits lay over the same interval.
    let e = coverage_at(local) * st.d.w;
    let lift = 1.0 - exp(st.b.z * e);
    let dep = 1.0 - exp(st.b.w * e);

    let rdim = vec2<f32>(textureDimensions(region_color));
    let rc = textureSampleLevel(region_color, samp, world / rdim, 0.0);
    let ra = textureSampleLevel(region_aux, samp, world / rdim, 0.0);

    let b_c = textureLoad(brush_src_color, t, 0); // (latent, op) — straight, not premult
    let b_a = textureLoad(brush_src_aux, t, 0);   // (height, wet, _, _)

    // Deplete the tool by what the upcoming segments deposit (they read this
    // post-pickup reservoir — see `deposit`)…
    let keep = 1.0 - dep;
    var h = b_a.x * keep;
    var wet = b_a.y * keep;
    var m = b_c.a * h;

    // …then fold in the paint lifted off the canvas: height moves, colour/opacity
    // ride as optical-mass-weighted blends (§6.1).
    let take_h = ra.x * lift;
    let take_m = rc.a * take_h;
    let canvas_lat = rc.rgb / max(rc.a, EPS);
    let lat = blend_latent(b_c.rgb, m, canvas_lat, take_m);
    h += take_h;
    wet += ra.y * lift;
    m += take_m;

    textureStore(brush_dst_color_w, t, vec4<f32>(lat, m / max(h, EPS)));
    textureStore(brush_dst_aux_w, t, vec4<f32>(h, wet, 0.0, 0.0));
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
    let lift = 1.0 - exp(st.b.z * e);
    let dep = 1.0 - exp(st.b.w * e);

    let uc = textureLoad(under_color, t, 0);
    let ua = textureLoad(under_aux, t, 0);
    // The tool's carried paint over this spot: the reservoir lives in the *shape*
    // frame (travel rotated by the orientation), and the tip's mid-pass position
    // over this texel is `xl − len/2` along the travel.
    let theta = st.d.z * TAU;
    let cs = cos(theta);
    let sn = sin(theta);
    let xc = clamp(xl - st.b.y * 0.5, -1.0, 1.0);
    let shape_local = vec2<f32>(xc * cs + yl * sn, -xc * sn + yl * cs);
    let buv = clamp(shape_local * 0.5 + vec2<f32>(0.5, 0.5), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let b_c = textureSampleLevel(brush_src_color, samp, buv, 0.0);
    let b_a = textureSampleLevel(brush_src_aux, samp, buv, 0.0);

    // The lift's removal: height (and wet) leave; the remaining paint's latent and
    // per-unit opacity are unchanged — the source fades because its *thickness*
    // drops, never because its alpha is scaled (§6.1).
    var h = ua.x * (1.0 - lift);
    var wet = ua.y * (1.0 - lift);
    var m = uc.a * h;
    let canvas_lat = uc.rgb / max(uc.a, EPS);

    // The parcel this segment lays: the tool's deposit + the brush's own `add`
    // paint — the latter linear in exposure, i.e. exactly the plain deposit's
    // path-integral height.
    let dep_h = b_a.x * dep;
    let add_h = st.e.x * e;
    let par_h = dep_h + add_h;
    let par_m = b_c.a * dep_h + st.c.w * add_h;
    var par_lat = st.c.xyz;
    if par_m > EPS {
        par_lat = (b_c.rgb * (b_c.a * dep_h) + st.c.xyz * (st.c.w * add_h)) / par_m;
    }
    let par_wet = b_a.y * dep + st.e.y * e;

    let lat = blend_latent(canvas_lat, m, par_lat, par_m);
    h += par_h;
    wet += par_wet;
    m += par_m;
    let op = m / max(h, EPS);

    textureStore(region_color_w, rt, vec4<f32>(lat * op, op));
    textureStore(region_aux_w, rt, vec4<f32>(h, wet, 0.0, 0.0));
}
