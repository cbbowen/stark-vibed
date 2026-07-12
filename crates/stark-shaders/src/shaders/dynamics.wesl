// Brush dynamics: the **sequential stamp loop** (DESIGN.md §6.2).
//
// A stroke that manipulates existing paint (load / deposit / charge) runs as an
// ordered loop over stamps. Per stamp, three small compute dispatches — all in one
// compute pass, so the whole loop is a handful of encoder calls and zero copies:
//
//   1. `snapshot` — copy the region texels under the stamp's footprint into a
//      scratch `under` texture, so the deposit can read-modify-write the region
//      (WebGPU forbids reading a texture the same dispatch writes).
//   2. `pickup`   — one thread per **brush reservoir** texel: sample the (evolving)
//      canvas region under that part of the tip, lift paint onto the tool, deplete
//      the tool by this stamp's deposit. Writes the ping-pong reservoir.
//   3. `deposit`  — one thread per footprint texel: remove the lifted paint from
//      the canvas and lay the tool's deposit + the brush's own `add` paint over it.
//      Writes the region in place (reading the pre-stamp `under` snapshot).
//
// The reservoir is a real 2-D texture in brush-local coordinates, so each part of
// the tip carries what *it* rolled through. Because pickup reads the region as
// already modified by earlier stamps, smearing is genuinely order-dependent —
// including across a stroke's own trail — with no banded 1-D approximation.
//
// Conservation (DESIGN.md §6.1): paint moves by transferring **height** — the one
// conserved quantity. Colour and per-unit opacity ride along as optical-mass
// (opacity·height) weighted blends; `add` is the only source term. Both sides of
// each transfer use the same rate expression evaluated over the same footprint, so
// with `add = 0` total height (canvas + tool) is conserved up to resampling error.
//
// Determinism: stamp positions/rates are fixed CPU-side per stroke; every dispatch
// is a pure function of the snapshot it reads; ordering is guaranteed by the
// implicit barriers between dispatches. Replay and live == committed hold.

struct Stamp {
    // a: stamp centre .xy (region px) + the brush frame's x-axis .zw (unit vector —
    //    travel tangent, or the pen azimuth for a pinned nib, §6.6).
    a: vec4<f32>,
    // b: radius (px), lift fraction, deposit fraction, add height (all per-stamp).
    b: vec4<f32>,
    // c: the brush's own colour channels .xyz + per-unit opacity .w.
    c: vec4<f32>,
    // d: under-rect origin .xy (region px, integral), add wet, unused.
    d: vec4<f32>,
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
// The brush tip's coverage mask over brush-local [-1,1]² (R8; §6.6).
@group(0) @binding(6) var cov_tex: texture_2d<f32>;
// The tool reservoir ping-pong: colour = (latent .xyz, per-unit opacity .w),
// aux = (carried height, carried wet, _, _). `src` is read this stamp, `dst`
// written; the host swaps the bind group each stamp.
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

const EPS: f32 = 1e-5;
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

// ---- 1. Snapshot the region under the stamp's footprint. -----------------------

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

// ---- 2. Pickup: canvas → tool, plus the tool's own depletion. -------------------

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
    let cov = coverage_at(local);

    let rdim = vec2<f32>(textureDimensions(region_color));
    let rc = textureSampleLevel(region_color, samp, world / rdim, 0.0);
    let ra = textureSampleLevel(region_aux, samp, world / rdim, 0.0);

    let b_c = textureLoad(brush_src_color, t, 0); // (latent, op) — straight, not premult
    let b_a = textureLoad(brush_src_aux, t, 0);   // (height, wet, _, _)

    let lift = st.b.y * cov;
    let dep = st.b.z * cov;

    // Deplete the tool by what this stamp deposits (the deposit pass lays the same
    // `dep · carried` from the same pre-stamp reservoir — see `deposit`)…
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

// ---- 3. Deposit: tool + `add` source → canvas, and the lift's removal. ----------

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
    // This region texel's brush-local position (inverse of the pickup mapping).
    let p = vec2<f32>(rt) + vec2<f32>(0.5, 0.5);
    let rel = (p - st.a.xy) / st.b.x;
    let rot = st.a.zw;
    let local = vec2<f32>(rel.x * rot.x + rel.y * rot.y, -rel.x * rot.y + rel.y * rot.x);
    let cov = coverage_at(local);
    if cov <= 0.0 {
        return; // outside the footprint: the region keeps its value (no store)
    }

    let uc = textureLoad(under_color, t, 0);
    let ua = textureLoad(under_aux, t, 0);
    // The tool's carried paint over this spot — the same pre-stamp reservoir the
    // pickup read, so the depletion there and the deposit here agree exactly.
    let buv = clamp(local * 0.5 + vec2<f32>(0.5, 0.5), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let b_c = textureSampleLevel(brush_src_color, samp, buv, 0.0);
    let b_a = textureSampleLevel(brush_src_aux, samp, buv, 0.0);

    let lift = st.b.y * cov;
    let dep = st.b.z * cov;

    // The lift's removal: height (and wet) leave; the remaining paint's latent and
    // per-unit opacity are unchanged — the source fades because its *thickness*
    // drops, never because its alpha is scaled (§6.1).
    var h = ua.x * (1.0 - lift);
    var wet = ua.y * (1.0 - lift);
    var m = uc.a * h;
    let canvas_lat = uc.rgb / max(uc.a, EPS);

    // The parcel this stamp lays: the tool's deposit + the brush's own `add` paint.
    let dep_h = b_a.x * dep;
    let add_h = st.b.w * cov;
    let par_h = dep_h + add_h;
    let par_m = b_c.a * dep_h + st.c.w * add_h;
    var par_lat = st.c.xyz;
    if par_m > EPS {
        par_lat = (b_c.rgb * (b_c.a * dep_h) + st.c.xyz * (st.c.w * add_h)) / par_m;
    }
    let par_wet = b_a.y * dep + st.d.z * cov;

    let lat = blend_latent(canvas_lat, m, par_lat, par_m);
    h += par_h;
    wet += par_wet;
    m += par_m;
    let op = m / max(h, EPS);

    textureStore(region_color_w, rt, vec4<f32>(lat * op, op));
    textureStore(region_aux_w, rt, vec4<f32>(h, wet, 0.0, 0.0));
}
