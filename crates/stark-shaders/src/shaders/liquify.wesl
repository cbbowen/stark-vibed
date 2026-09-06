// The liquify field (§6.13): a stroke as a **homeomorphism** of the canvas, kept
// as a displacement field composed one segment at a time and applied to the
// picture by one resample per piece — three compute kernels over the wet loop's
// region machinery (`dynamics.wesl`'s pieces, composite and write-back), reading
// the same per-slot uniform and sweep frame (`dynamics_common.wesl`).
//
// Bindings: group 0 is partitioned with `dynamics_common` — its slot at 0, the
// region's write targets and the selection mask — and this module's own start at
// 1. The prefix volume at group 1 (`stamp_common`) is bound to the tip's
// **coverage prefix** here, where every deposit binds its prefix-τ.

import package::dynamics_common::{
    st, TILE_WG, travel_px, rate_across, arc_at, stroke_drain, segment_frame, segment_look,
    swept_pre, outside_look, region_color_w, region_aux_w, sel_mask,
};
@if(resid) import package::dynamics_common::{region_resid_w};
import package::stamp_common::{stretch_gain, sweep_tangent, Sweep};
import package::lib::sample::{load1};
import package::lib::store::{f16_nearest4};

// ---- Bindings -------------------------------------------------------------------
//
// A liquify stroke never touches `region_color` in place. What it evolves is the
// run's **displacement field** over the region — `d(x)` in canvas px, `rg32float`,
// where the picture at `x` is the run's pristine base at `x + d(x)` — and the
// picture is resampled through it once per piece, by `warp_apply`, from the **base
// composite**: the run's base tiles laid 1:1 like the region, but grown by the
// run's reach so every `x + d(x)` lands inside it. Three groups of bindings, one
// per kernel: the field's own snapshot (`snapshot_field` copies the field under a
// segment's square so `warp` can gather it while storage-writing it, exactly as the
// deposit's snapshot works), the composition (`warp`: field in through the
// snapshot, field out), and the resample (`warp_apply`: field and base in, the
// region's color, aux and residual out — the same targets the write-back slices).
@group(0) @binding(1) var field: texture_2d<f32>;
@group(0) @binding(2) var field_w: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var under_field: texture_2d<f32>;
@group(0) @binding(4) var under_field_w: texture_storage_2d<rg32float, write>;
@group(0) @binding(5) var base_color: texture_2d<f32>;
@group(0) @binding(6) var base_aux: texture_2d<f32>;
@if(resid) @group(0) @binding(7) var base_resid: texture_2d<f32>;

// ---- The field: compose, then resample once. -------------------------------------
//
// A liquify stroke is a **homeomorphism** of the canvas, and it is kept as one: a
// displacement field `d` over the region, `rg32float` canvas px, such that the
// picture at `x` is the run's pristine base at `x + d(x)`. Each segment is a small
// backward step `ψ(x) = x − v(x)`, with `v` the follow along the local travel
// tangent, and the stroke's map composes as the field:
//
//     d'(x) = d(x − v(x)) − v(x)         (`warp`, one bilinear gather of the field)
//
// — a gather of the *field*, which is smooth, and therefore nearly lossless where a
// gather of the picture blurs an edge a little per segment until nothing sharp is
// left. The picture is resampled through the final field exactly once per piece,
// from the base composite (`warp_apply`), so what a hard edge pays is one
// generation however many strokes and segments moved it.
//
// **The follow is the tip's own coverage, swept.** `v = follow · t̂`, and the follow
// is `strength · ∫coverage` over the texel's stretch of the pass — the difference
// of the **coverage prefix** across it (`swept_pre`, on the volume the liquify
// path binds at `prefix_tex` in the prefix-τ's place: the same mask, integrated
// linearly). Exact for any cut of the path, as every deposit's exposure is, so
// consecutive segments' follows add to the pass without printing their cadence;
// and it is the mask *as it is*, so a soft tip carries its shoulder gently, a hard
// one drags its disc whole, and a stamp drags by its picture.
//
// **Every step is a contraction where the tip lets it be.** What can fold the map
// `x ↦ x − v(x)` is the follow's gradient *along* the travel, which is
// `strength · (coverage ahead − coverage behind)` across the step — the tip's
// leading edge climbing under the paint ahead of it. The host's segment budget
// holds that climb under half (`budget::liquify_len`, the tip's **rise**), which
// makes each step a bijection of the plane and the composition one. The gradient
// *across* the travel — the shoulder's slope — is a shear, and a shear is a
// bijection at any slope, so it is not priced: a hard rim is a shear line, not a
// fold, and no step length widens it. The selection multiplies the follow *after*
// (§6.8), so a hard mask edge is a deliberate tear along the mask, and a feathered
// one a homeomorphism again. The tooth and the jitter are deliberately **not**
// read: a follow gated by the substrate's every step is a field of tears.
//
// Sequenced in place through the snapshot, so a stroke composes over **its own
// trail** when it crosses it — the order-dependence the loop's every kernel has.

// A follow no texel could express — under a 2⁻¹² px nudge. Returning without the
// store keeps a near-inert drag from re-storing the field once per segment, which is
// the deposit's rewrite guard in this kernel's terms (§6.2).
const WARP_MIN_FOLLOW: f32 = 0.000244140625;

// The step at region texel `rt`: how far its paint keeps pace with the tip across
// this slot, in px, along the local travel tangent. Zero — without a store — where
// the tip's pass never reaches it.
//
// `strength · ∫coverage`, in the tip's own measure: `swept_pre` reads the coverage
// prefix's difference across the texel's stretch of travel in the lookup frame's
// measure, `sw.warp` and `stretch_gain` bring it back to the tip's (§6.6), and the
// radius makes it px. At strength 1 a texel the tip's core slides over keeps pace
// with the hand exactly. Capped at the pass: the prefix cannot exceed it, so the
// `min` is the wall for rounding — what keeps the gather inside the snapshot the
// slot's travel sized. The drain runs the drag dry as it runs a deposit dry (§6.2),
// and the selection scales the follow rather than blending the result (§6.8).
fn warp_follow(sw: Sweep, lk: vec3<f32>, rt: vec2<i32>) -> f32 {
    let sel = load1(sel_mask, rt).r;
    if sel <= 0.0 {
        return 0.0;
    }
    let e = swept_pre(lk) * sw.warp * stretch_gain(st.stretch);
    if e <= 0.0 {
        return 0.0;
    }
    let arc = arc_at(sw);
    let rate = min(rate_across(st.drag, st.drag_ramp, arc), 1.0);
    return min(rate * e * st.radius, travel_px()) * stroke_drain(arc) * sel;
}

// Copy the field under a warp slot's square to the snapshot, so `warp` can gather it
// while storage-writing it. The whole square: the gather pulls from *upstream* of
// any one texel's sweep test, and a clamped tap has to land on real content — or on
// the identity the out-of-region zero below states.
@compute @workgroup_size(TILE_WG, TILE_WG)
fn snapshot_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_field_w);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(i32(gid.x), i32(gid.y));
    let rt = st.rect_origin + t;
    let rdim = vec2<i32>(textureDimensions(field));
    var d = vec4<f32>(0.0);
    if rt.x >= 0 && rt.y >= 0 && rt.x < rdim.x && rt.y < rdim.y {
        d = textureLoad(field, rt, 0);
    }
    textureStore(under_field_w, t, d);
}

// Compose one segment's step into the field: `d'(x) = d(x − v) − v`.
@compute @workgroup_size(TILE_WG, TILE_WG)
fn warp(@builtin(global_invocation_id) gid: vec3<u32>) {
    // A slot whose modulated strength has fallen to zero moves nothing and must
    // store nothing — the plan still dispatches it, for the reason a zero-rate
    // bleed firing dispatches: the plan stays a pure function of the segmentation.
    if st.drag <= 0.0 {
        return;
    }
    let udim = textureDimensions(under_field);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(i32(gid.x), i32(gid.y));
    let rt = st.rect_origin + t;
    let rdim = vec2<i32>(textureDimensions(field_w));
    if rt.x < 0 || rt.y < 0 || rt.x >= rdim.x || rt.y >= rdim.y {
        return;
    }
    let sw = segment_frame(rt);
    // Provably outside the sweep, in the frame the mask is read in (§6.6): the
    // prefix difference is exactly 0 there, at the cost of its taps.
    let lk = segment_look(sw);
    if outside_look(lk) {
        return;
    }
    let follow = warp_follow(sw, lk, rt);
    if follow < WARP_MIN_FOLLOW {
        return;
    }
    let v = sweep_tangent(st.dir, sw.turn) * follow;
    // Where this texel's paint stood before the step, in snapshot texels; the field
    // there says where *that* came from. Clamped into the scratch — whose every
    // texel the slot's snapshot wrote — as the wall for whatever rounding remains:
    // the follow is at most the pass, which keeps the tap inside the sweep's box.
    let u = vec2<f32>(t) - v;
    let base = floor(u);
    let f = u - base;
    let hi = vec2<i32>(udim) - vec2<i32>(1, 1);
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0, 0), hi);
    let i1 = clamp(vec2<i32>(base) + vec2<i32>(1, 1), vec2<i32>(0, 0), hi);
    let d = warp_tap(under_field, i0, i1, f).xy;
    textureStore(field_w, rt, vec4<f32>(d - v, 0.0, 0.0));
}

// Resample the region through the field, once: `out(x) = base(x + d(x))`, every
// channel on one map — the color premultiplied as the region holds it, the aux and
// the residual beside it — so what lands is the base a little away, whole.
//
// **Catmull-Rom, clamped, because this is the one resample there is.** The field
// composes losslessly, so the only softening a liquified edge ever pays is this
// pass's, and a cubic keeps a hard edge a texel sharper than a tent would. Its
// negative lobes ring, and on premultiplied lanes ringing is a value no paint could
// have — so each lane is clamped to the range of the four texels nearest the tap
// (`warp_tap_cubic`), the anti-ringing clamp, which also keeps every lane inside
// what its neighbourhood holds.
//
// **The identity is exact.** Where `d` is 0 the tap sits on a texel centre, the
// cubic weights are exactly (0, 1, 0, 0), and `f16_nearest4` of a value already on
// the f16 lattice is that value — so a texel the run never moved is stored
// bit-for-bit, whichever adapter converts the store (`lib::store`). A drag through a
// field that does not vary is the identity by the clamp alone: equal neighbours
// bound the tap to themselves.
@compute @workgroup_size(TILE_WG, TILE_WG)
fn warp_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rdim = textureDimensions(region_color_w);
    if gid.x >= rdim.x || gid.y >= rdim.y {
        return;
    }
    let rt = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = textureLoad(field, rt, 0).xy;
    // The source in base texels: this texel's own place in the base composite, then
    // the displacement. The integer part is exact in f32 (a composite is at most a
    // few thousand texels wide), so an unmoved texel lands on itself exactly.
    let src = vec2<f32>(rt + st.base_offset) + d;
    let bdim = vec2<i32>(textureDimensions(base_color));
    let base = vec2<i32>(floor(src));
    let f = src - floor(src);
    // Every tap is clamped into the base composite, the no-flux wall (§6.13): the
    // composite is grown by the run's reach, so a tap only meets the wall through
    // rounding — or through the cubic's outer ring, which reaches a texel further.
    let hi = bdim - vec2<i32>(1, 1);
    textureStore(region_color_w, rt, f16_nearest4(warp_tap_cubic(base_color, base, f, hi)));
    textureStore(region_aux_w, rt, f16_nearest4(warp_tap_cubic(base_aux, base, f, hi)));
    @if(resid) textureStore(region_resid_w, rt, f16_nearest4(warp_tap_cubic(base_resid, base, f, hi)));
}

// Catmull-Rom weights along one axis at fraction `f` ∈ [0, 1), for the taps at
// −1, 0, 1, 2 texels from the base. Exactly (0, 1, 0, 0) at `f = 0`, which is what
// the identity rests on; summing to 1 up to rounding elsewhere.
fn cubic_weights(f: f32) -> vec4<f32> {
    let f2 = f * f;
    let f3 = f2 * f;
    return vec4<f32>(
        -0.5 * f + f2 - 0.5 * f3,
        1.0 - 2.5 * f2 + 1.5 * f3,
        0.5 * f + 2.0 * f2 - 1.5 * f3,
        -0.5 * f2 + 0.5 * f3,
    );
}

// One Catmull-Rom tap of the resample: the separable 4×4 sum about `base` at
// fraction `f`, clamped per lane to the range of the central 2×2 — the four texels
// a bilinear tap would have mixed. The ring the cubic adds is what sharpens the
// edge; the clamp is what keeps it from inventing a value no neighbour holds.
// Hand-rolled on `textureLoad` for `warp_tap`'s reasons.
fn warp_tap_cubic(tex: texture_2d<f32>, base: vec2<i32>, f: vec2<f32>, hi: vec2<i32>) -> vec4<f32> {
    let wx = cubic_weights(f.x);
    let wy = cubic_weights(f.y);
    var sum = vec4<f32>(0.0);
    var lo_v = vec4<f32>(3.4e38);
    var hi_v = vec4<f32>(-3.4e38);
    for (var j = 0; j < 4; j = j + 1) {
        let y = clamp(base.y + j - 1, 0, hi.y);
        var row = vec4<f32>(0.0);
        for (var i = 0; i < 4; i = i + 1) {
            let x = clamp(base.x + i - 1, 0, hi.x);
            let v = textureLoad(tex, vec2<i32>(x, y), 0);
            row = row + wx[i] * v;
            if (i == 1 || i == 2) && (j == 1 || j == 2) {
                lo_v = min(lo_v, v);
                hi_v = max(hi_v, v);
            }
        }
        sum = sum + wy[j] * row;
    }
    return clamp(sum, lo_v, hi_v);
}

// One bilinear tap of the field's gather: the four texels the clamp admitted, mixed
// in the separable form. Hand-rolled on `textureLoad` for the prefix lookup's
// reasons — fp32 weights that are a pure function of the inputs, so the gather is
// bit-reproducible across adapters (§12.1) and needs no FLOAT32_FILTERABLE anywhere,
// which the field's `rg32float` could not have offered.
fn warp_tap(tex: texture_2d<f32>, i0: vec2<i32>, i1: vec2<i32>, f: vec2<f32>) -> vec4<f32> {
    return mix(
        mix(
            textureLoad(tex, vec2<i32>(i0.x, i0.y), 0),
            textureLoad(tex, vec2<i32>(i1.x, i0.y), 0),
            f.x,
        ),
        mix(
            textureLoad(tex, vec2<i32>(i0.x, i1.y), 0),
            textureLoad(tex, vec2<i32>(i1.x, i1.y), 0),
            f.x,
        ),
        f.y,
    );
}
