// Merging a layer **down** onto the one beneath it (§14.11): two tiles in, one tile
// out, carrying exactly the pixels the compositor would have produced from the two.
//
// The whole feature rests on one identity, and it is not a new one. Pass A gives a
// layer the weight `1 − exp(−K·opacity·height)` — its **visible alpha**, the
// translucent-slab law — and stacks the weighted colors with premultiplied "over"
// while summing the heights (`composite.wesl`). `paint_common.wesl`'s `blend_latent`
// stacks a *parcel* onto resident paint by that very law, because a parcel must cover
// resident paint exactly as much as the viewer sees it cover. The two are the same
// arithmetic, so the tile this pass writes composites to the same texels the pair did:
//
//     H = h₀ + h₁                       heights add — paint stacks (§6.1)
//     M = m₀ + m₁                       optical masses add, since
//                                       1 − e^{−M} = 1 − (1−w₀)(1−w₁) is "over"
//     opacity = M / H                   the height-weighted mean of the two
//     latent  = blend_latent(…)         "over" on the weighted colors
//
// **That the merged opacity is a mean is the reason this is exact rather than close.**
// A tile stores per-unit opacity and height separately (§6.1), and the pair of them
// has exactly the freedom needed to express any (coverage, height) the stack can
// reach — conserving height, which is the invariant, and letting opacity be whatever
// the conservation implies. Nothing is fitted and no constant is tuned.
//
// The layer **opacity** slider is folded in here rather than left on the merged layer,
// so a merge of two half-faded layers is one layer at full strength that looks the
// same. Pass A applies it to both the height and the weight (`aux = h·opl`,
// `w = (1−e^{−K·op·h})·opl`), which is not a slab of any opacity — so the fold runs
// through `optical_mass`, inverting the law to recover the mass that *does* produce
// that weight. The inverse is exact and lands in range: `M ≤ h·opl` for every
// `opacity ≤ 1`, so the mean above stays a per-unit opacity in `[0,1]`.
//
// Two laws, because a **clipped** layer is not stacked but *deleted outside its
// backdrop* (§14.4, `blend_common.wesl::merge`) — see `fs_main`'s two branches.
//
// Unlike the transform's combine there is no cut and no mask: both inputs are whole
// tiles of the same coordinate, and either may be the 1×1 zero where its layer has no
// tile there (§6.8's pattern, `lib/sample.wesl`).

import package::lib::latent::{Latent, latent_blend, latent_lerp, latent_of, latent_scale, latent_unpremult};
import package::lib::paint_common::{OPACITY_K, PAINT_EPS};
import package::lib::quad::{fullscreen_clip};
import package::lib::sample::{load1};

struct Merge {
    // .x = the lower layer's opacity, .y = the upper layer's — folded into the
    // result, which is why the merged layer is left at full opacity.
    // .z = 1 when the upper layer is **clipped** to the lower (§14.4), else 0.
    // .w unused.
    p: vec4<f32>,
}

@group(0) @binding(0) var<uniform> m: Merge;
// The destination: the layer being merged *into*, which is the one beneath.
@group(0) @binding(1) var lower_color: texture_2d<f32>;
@group(0) @binding(2) var lower_aux: texture_2d<f32>;
// The source: the layer being merged *down*.
@group(0) @binding(3) var upper_color: texture_2d<f32>;
@group(0) @binding(4) var upper_aux: texture_2d<f32>;
// The residual channels (§6.7), read through the same clamped `load1` — a layer with
// no tile here binds the 1×1 zero for these exactly as it does for the color.
@if(resid) @group(0) @binding(5) var lower_resid: texture_2d<f32>;
@if(resid) @group(0) @binding(6) var upper_resid: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @if(resid) @location(2) resid: vec4<f32>,
}

// The optical mass `M` whose slab weight `1 − exp(−K·M)` is what pass A would give a
// texel of this layer: the stored `opacity · height`, scaled by the layer's opacity
// slider the way pass A scales it.
//
// The slider is **not** a factor of the mass — pass A multiplies the finished weight
// by it (`composite.wesl`), which is a different curve — so recovering the mass means
// inverting the law: `M = −ln(1 − opl·(1 − e^{−K·m})) / K`.
//
// Exact at `opl = 1` by an explicit branch rather than by the arithmetic agreeing to a
// last bit: that is the overwhelmingly common case, and taking it verbatim is what
// keeps a merge of two ordinary layers bit-identical to `fill.wesl`'s stacking law and
// free of a log/exp round trip.
fn optical_mass(opacity: f32, height: f32, opl: f32) -> f32 {
    let mass = opacity * height;
    if opl >= 1.0 {
        return mass;
    }
    // `1 − opl·(1 − e^{−K·m}) ≥ 1 − opl > 0` for any slider below 1, so the guard is
    // only reached by a slider within `PAINT_EPS` of full — where the branch above is
    // the answer anyway, to well past what an f16 tile can store.
    let weight = opl * (1.0 - exp(-OPACITY_K * mass));
    return -log(max(1.0 - weight, PAINT_EPS)) / OPACITY_K;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy));
    let lc = load1(lower_color, c);
    let la = load1(lower_aux, c).x;
    let uc = load1(upper_color, c);
    let ua = load1(upper_aux, c).x;
    // The two colors, each as one value (§6.7, `lib::latent`); the raw pair is kept
    // whole for the two exact passthroughs below. One gate covers every residual read
    // in the pass, and nothing after it is written twice.
    var lower = latent_of(lc.rgb, vec3<f32>(0.0));
    var upper = latent_of(uc.rgb, vec3<f32>(0.0));
    var lr = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var ur = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    @if(resid) {
        lr = load1(lower_resid, c);
        ur = load1(upper_resid, c);
        lower.res = lr.rgb;
        upper.res = ur.rgb;
    }

    let opl0 = m.p.x;
    let opl1 = m.p.y;
    // Heights as pass A would have accumulated them: the slider scales the *amount of
    // paint* directly (`aux = h·opl`), unlike the weight.
    let h0 = la * opl0;
    let h1 = ua * opl1;
    let m0 = optical_mass(lc.a, la, opl0);
    let m1 = optical_mass(uc.a, ua, opl1);

    var out: FsOut;
    // Nothing of the upper layer lands here, and the lower layer is already carrying
    // the answer. An exact passthrough rather than the general arithmetic, which is
    // the same argument `fill.wesl` makes at its own untouched branch: the f32→f16
    // store rounds in an implementation-defined direction, so merely *recomputing* a
    // resident value can land an ulp off it — and a merge of a small stroke layer onto
    // a large one runs this branch over nearly every texel it rewrites.
    //
    // Both laws below agree here (a clipped layer with no paint deletes nothing), so
    // one guard covers both.
    if h1 <= 0.0 && opl0 >= 1.0 {
        out.color = lc;
        out.aux = vec4<f32>(la, 0.0, 0.0, 0.0);
        @if(resid) out.resid = lr;
        return out;
    }
    // The mirror of it, and the case that makes tile-level pass-through nearly moot:
    // the destination has no paint here, so under "over" the answer is the source's
    // own texel. It matters because tiles are coarse — two strokes that never meet
    // still share the tiles their bounding boxes cross — so without this branch a
    // merge of two disjoint marks would rewrite every texel of one of them for
    // nothing.
    //
    // Not taken under `Clip`, where an absent backdrop deletes the source rather than
    // passing it through; that case falls to the general arithmetic below, which
    // returns nothing for it.
    if h0 <= 0.0 && opl1 >= 1.0 && m.p.z <= 0.5 {
        out.color = uc;
        out.aux = vec4<f32>(ua, 0.0, 0.0, 0.0);
        @if(resid) out.resid = ur;
        return out;
    }

    let col0 = latent_unpremult(lower, lc.a, PAINT_EPS);
    let col1 = latent_unpremult(upper, uc.a, PAINT_EPS);

    if m.p.z > 0.5 {
        // **Clipped** (§14.4): the upper layer exists only where the lower one does,
        // and it is a *deletion* rather than a scale on its alpha. Reading
        // `blend_common.wesl::merge` at `MODE_NORMAL` with `clip = 1` and a backdrop
        // that is exactly this tile:
        //
        //     αo  = αb                     the backdrop's coverage, untouched
        //     Co  = αb·(αs·Cs + (1−αs)·Cb) which is that same coverage over a lerp
        //     ho  = hb + hs·αb             height suppressed where the backdrop is not
        //
        // The coverage being untouched is what makes the stored opacity simple: the
        // merged texel has the lower layer's mass over a taller slab.
        let v0 = 1.0 - exp(-OPACITY_K * m0);
        let v1 = 1.0 - exp(-OPACITY_K * m1);
        let h = h0 + h1 * v0;
        let op = m0 / max(h, PAINT_EPS);
        // The residual rides the same lerp on the same weight — it is the rest of the
        // same color (§6.7), which one call states.
        let clipped = latent_scale(latent_lerp(col0, col1, v1), op);
        out.color = vec4<f32>(clipped.lat, op);
        out.aux = vec4<f32>(h, 0.0, 0.0, 0.0);
        @if(resid) out.resid = vec4<f32>(clipped.res, op);
        return out;
    }

    // Plain "over": the header's four lines, and the very arithmetic `fill.wesl` lays
    // a parcel of paint with.
    let h = h0 + h1;
    let mass = m0 + m1;
    let op = mass / max(h, PAINT_EPS);
    let merged = latent_scale(latent_blend(col0, m0, col1, m1), op);
    out.color = vec4<f32>(merged.lat, op);
    out.aux = vec4<f32>(h, 0.0, 0.0, 0.0);
    @if(resid) out.resid = vec4<f32>(merged.res, op);
    return out;
}
