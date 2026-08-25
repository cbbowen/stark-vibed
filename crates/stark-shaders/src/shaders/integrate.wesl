// Stroke *integrate* pass (§6.2/§6.1). Every brush rasterizes its swept
// extent into a *cleared scratch* tile (the stamp pass), then this fullscreen
// pass merges that scratch into a fresh copy-on-write tile over the base:
// `new = f(base, scratch)`, written to the color+aux MRT.
//
// It is a render pass (not compute) because the tile channels are `Rgba16Float` /
// `R16Float`, which are not core-WebGPU storage-texture formats. Everything is
// sampled with `textureLoad` (1:1 with the destination — no filtering), so the
// result is a deterministic function of canvas position and a tile's apron is
// rewritten identically to its neighbour's interior (the seam invariant, §6.4).
//
// In the normalized representation (§6.1): color = latent·opacity with opacity in
// `.a`, aux = (height). The merge is the **shared parcel law** (`paint_common.wesl`)
// — the very one a fill lands through and the one the stamp loop's `deposit` stacks
// its own `add` paint with, so a fast-path stroke cannot drift from a
// `lift = deposit = 0` one over the same travel. Height sums (it is the amount of
// paint), optical mass sums, and the parcel covers resident paint by its own visible
// alpha `1 − exp(−K·opacity·height)`.
//
// **Not a premultiplied-"over" weighed by the scratch's alpha.** Such an alpha is
// built from the brush's per-unit opacity alone, so the `add` rate (Flow) — the thing
// that decides how much paint the segment actually lays — would reach the *height* and
// never the color. Over bare canvas that hides, since visible alpha is `op · height`
// there and the height is right; over existing paint nothing hides it, and a brush at
// 5% flow repaints the canvas at full strength in one pass. It is the defect §6.3
// names in the layer composite — covering by opacity rather than by
// opacity × thickness — one level further down.

import package::lib::latent::{Latent, latent_blend, latent_of, latent_scale, latent_unpremult};
import package::lib::paint_common::{OPACITY_K, PAINT_EPS};
import package::lib::quad::{fullscreen_clip};
import package::lib::sample::{load1};

struct Integrate {
    // The paint effect's opacity (`PaintEffect::opacity`, §6.2) in `.x`. 1 is the
    // identity, and taken as an exact branch below. Incorporates the selection's strength (`Selection::strength`, §6.8): the whole mask's opacity, which every *gating* read of the mask is scaled by and which the mask texture itself does not carry.
    params: vec4<f32>,
}

// The layer's resident paint under this tile — its own tile, or the 1×1 zero
// standing in where the layer has none. The clamped load below reads both, which is
// what lets a stroke onto bare canvas run this same shader with nothing bound for it
// (§6.8's pattern, as `fill.wesl` and `transform.wesl` use it).
@group(0) @binding(0) var base_color: texture_2d<f32>;
@group(0) @binding(1) var base_aux: texture_2d<f32>;
@group(0) @binding(2) var scratch_color: texture_2d<f32>;
@group(0) @binding(3) var scratch_aux: texture_2d<f32>;
// The selection's coverage over this tile (§6.8): 1 where the brush may
// paint, 0 where it may not, feathered in between. Either this tile's own mask —
// 1:1 with the destination, apron included — or a 1×1 texture holding the constant
// coverage outside the mask's tile set; the clamped load below reads both.
@group(0) @binding(4) var selection: texture_2d<f32>;
// The residual halves of the two colors above (§6.7) — the layer's resident
// paint and the stroke's parcel. Bound only by a space that has a residual, and
// read through the same clamped `load1`/`textureLoad` pair as the latents, so a
// stroke onto bare canvas reads the 1×1 zero here exactly as it does for `base_color`.
@if(resid) @group(0) @binding(5) var base_resid: texture_2d<f32>;
@if(resid) @group(0) @binding(6) var scratch_resid: texture_2d<f32>;
@group(0) @binding(7) var<uniform> ig: Integrate;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @if(resid) @location(2) resid: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy));
    let base_c = load1(base_color, c);
    let base_a = load1(base_aux, c);
    // The stroke's parcel: latent premultiplied by its visible alpha, that alpha in
    // `.a`. Wide scratch aux: (laid height, optical mass, unused, unused) — see
    // stamp.wesl's fs_main for the channel-by-channel story.
    let s_c = textureLoad(scratch_color, c, 0);
    let s_a = textureLoad(scratch_aux, c, 0);
    // The two colors, each as one value (§6.7, `lib::latent`). **The gate is on the
    // load and nowhere else**: a residual *texture* exists only in a space that has
    // one, so fetching it is an `@if` no bundle can absorb — but everything between
    // here and the stores is then written once, for both halves.
    var base_col = latent_of(base_c.rgb, vec3<f32>(0.0));
    var s_col = latent_of(s_c.rgb, vec3<f32>(0.0));
    // Kept whole for the passthrough below, which must not recompute *anything* — the
    // residual target carries its own copy of the alpha, and reading the color's would
    // be a recomputation however equal the two are today.
    var base_r = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    @if(resid) {
        base_r = load1(base_resid, c);
        base_col.res = base_r.rgb;
        s_col.res = textureLoad(scratch_resid, c, 0).rgb;
    }

    var out: FsOut;
    // The stroke's extent missed this texel. An exact branch rather than the
    // general arithmetic, so a texel a stroke merely passed *near* — every tile it
    // overlaps is rewritten whole — passes through bit-for-bit. The f32→f16 store's
    // rounding is implementation-defined, so recomputing a resident value through
    // the `m/h` round trip below can land an ulp off it, and a hundred strokes
    // across a tile would each nudge the paint they never touched. The same argument
    // `fill.wesl` makes for its untouched branch.
    if s_a.x == 0.0 && s_a.y == 0.0 && s_c.a == 0.0 {
        out.color = base_c;
        out.aux = vec4<f32>(base_a.x, 0.0, 0.0, 0.0);
        // Passed through whole, for the reason above: the residual goes through the
        // same `m/h` round trip below and would drift by an ulp per stroke too.
        @if(resid) out.resid = base_r;

        return out;
    }

    // Resident paint: height, optical mass (per-unit opacity × height), latent.
    let h0 = base_a.x;
    let m0 = base_c.a * h0;
    let col0 = latent_unpremult(base_col, base_c.a, PAINT_EPS);
    // The stroke's parcel, accumulated across the overlapping segment quads. `.a` is
    // already its visible alpha (the segments' `1 − exp(−K·m)` over-blend multiplies
    // out to `1 − exp(−K·Σm)`), which is what recovers the latent; the mass itself
    // rides unsaturated in the aux, because the merged per-unit opacity is `m/h` and
    // a saturated total cannot answer that.
    var h1 = s_a.x;
    var m1 = s_a.y;
    let col1 = latent_unpremult(s_col, s_c.a, PAINT_EPS);

    // Gate by the selection: the tile keeps its base value where the mask is 0 and
    // takes the stroke where it is 1 (§6.8). A *lerp of the result*, not a scaling of
    // the stroke's coverage — a half-covered mask texel must read as half of the
    // finished paint, which for an opaque brush is nothing like half its optical
    // depth. That makes the mask's own antialiasing/feather the stroke's edge.
    // Scaled by the whole mask's strength: the coverage a mask *holds* is what its
    // ops struck, and how strongly that gates is one number on top of it (§6.8) —
    // so the two are only a mask together, and this is where they meet.
    let effective_opacity = ig.params.x * load1(selection, c).r;

    // The paint effect's opacity (§6.2): the whole parcel scaled as one laid
    // amount, `erase.wesl`'s arithmetic run in the laying direction. The coverage
    // the parcel would have laid saturates at 1 however long the stroke worked
    // this spot — when it spans the pieces of a live stroke, the carry holds the
    // accumulated parcel and re-derives every piece from pristine paint
    // (`gpu::stroke::swept`), because a scaled coverage is neither of §6.2's two
    // composable forms — so `opacity·w` is a ceiling, not a rate: a stroke
    // crossing itself never outruns the dial.
    //
    // An exact branch at 1, `integrate.wesl`'s own untouched-texel reason one
    // level up: the round trip below lands an ulp off now and then, and every
    // full-opacity stroke would pay it.
    if effective_opacity != 1.0 {
        // The parcel's own coverage, from the unsaturated mass — the same number
        // the color target's `.a` carries, except that the saturated copy cannot
        // be inverted back.
        let w = 1.0 - exp(-OPACITY_K * m1);
        // The scaled coverage, inverted through the slab law into the mass that
        // shows exactly that much. `1 − opacity·w ≥ 1 − opacity > 0` on this
        // branch, so nothing cancels and the log has an argument.
        let ms = -log(1.0 - effective_opacity * w) / OPACITY_K;
        // Less of the same parcel (§6.1): the height falls in proportion, so the
        // per-unit opacity of what arrives is unchanged — a half-opacity stroke
        // is half a stroke, not full-bodied translucent material.
        h1 = h1 * (ms / max(m1, PAINT_EPS));
        m1 = ms;
    }

    // Stack the parcel on the resident paint (§6.1): height and mass are conserved
    // sums, the per-unit opacity is what the two together imply, and the latent
    // blends by the shared law — thick deposits cover, thin glazes tint.
    let h = h0 + h1;
    let mm = m0 + m1;
    let op = mm / max(h, PAINT_EPS);
    // The residual stacks through the *same* law on the *same* masses — it is the rest
    // of the same color, so anything that weighted it differently would have the merged
    // paint reproduce neither of the two colors that made it (§6.7). One call, so "the
    // same law" is a fact about the code rather than a claim about two lines.
    let merged = latent_scale(latent_blend(col0, m0, col1, m1), op);
    let merged_c = vec4<f32>(merged.lat, op);
    let merged_a = vec4<f32>(h, 0.0, 0.0, 0.0);

    out.color = merged_c;
    out.aux = merged_a;
    @if(resid) out.resid = vec4<f32>(merged.res, op);
    return out;
}
