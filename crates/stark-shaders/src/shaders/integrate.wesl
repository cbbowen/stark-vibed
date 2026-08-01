// Stroke *integrate* pass (§6.2/§6.1). Every brush rasterizes its swept
// footprint into a *cleared scratch* tile (the stamp pass), then this fullscreen
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
// It used to be a plain premultiplied-"over" weighed by the scratch's alpha, and
// that alpha was built from the brush's per-unit opacity alone — so the `add` rate
// (Flow), the thing that decides how much paint the segment actually lays, reached
// the *height* and never the colour. Over bare canvas the media pass hid it, because
// visible alpha is `op · height` there and the height was right; over existing paint
// nothing did, and a brush at 5% flow repainted the canvas at full strength in one
// pass. It is the same defect §6.3 names in the layer composite — covering by
// opacity rather than by opacity × thickness — one level further down.

import package::paint_common::{blend_latent, EPS};

@group(0) @binding(0) var base_color: texture_2d<f32>;
@group(0) @binding(1) var base_aux: texture_2d<f32>;
@group(0) @binding(2) var scratch_color: texture_2d<f32>;
@group(0) @binding(3) var scratch_aux: texture_2d<f32>;
// The selection's coverage over this tile (§6.8): 1 where the brush may
// paint, 0 where it may not, feathered in between. Either this tile's own mask —
// 1:1 with the destination, apron included — or a 1×1 texture holding the constant
// coverage outside the mask's tile set; the clamped load below reads both.
@group(0) @binding(4) var selection: texture_2d<f32>;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy));
    let base_c = textureLoad(base_color, c, 0);
    let base_a = textureLoad(base_aux, c, 0);
    // The stroke's parcel: latent premultiplied by its visible alpha, that alpha in
    // `.a`. Wide scratch aux: (laid height, optical mass, unused, unused) — see
    // stamp_oklab.wesl's fs_main for the channel-by-channel story.
    let s_c = textureLoad(scratch_color, c, 0);
    let s_a = textureLoad(scratch_aux, c, 0);

    var out: FsOut;
    // The stroke's footprint missed this texel. An exact branch rather than the
    // general arithmetic, so a texel a stroke merely passed *near* — every tile it
    // overlaps is rewritten whole — passes through bit-for-bit. The f32→f16 store's
    // rounding is implementation-defined, so recomputing a resident value through
    // the `m/h` round trip below can land an ulp off it, and a hundred strokes
    // across a tile would each nudge the paint they never touched. The same argument
    // `fill.wesl` makes for its untouched branch.
    if s_a.x == 0.0 && s_a.y == 0.0 && s_c.a == 0.0 {
        out.color = base_c;
        out.aux = vec4<f32>(base_a.x, 0.0, 0.0, 0.0);
        return out;
    }

    // Resident paint: height, optical mass (per-unit opacity × height), latent.
    let h0 = base_a.x;
    let m0 = base_c.a * h0;
    let lat0 = base_c.rgb / max(base_c.a, EPS);
    // The stroke's parcel, accumulated across the overlapping segment quads. `.a` is
    // already its visible alpha (the segments' `1 − exp(−K·m)` over-blend multiplies
    // out to `1 − exp(−K·Σm)`), which is what recovers the latent; the mass itself
    // rides unsaturated in the aux, because the merged per-unit opacity is `m/h` and
    // a saturated total cannot answer that.
    let h1 = s_a.x;
    let m1 = s_a.y;
    let lat1 = s_c.rgb / max(s_c.a, EPS);

    // Stack the parcel on the resident paint (§6.1): height and mass are conserved
    // sums, the per-unit opacity is what the two together imply, and the latent
    // blends by the shared law — thick deposits cover, thin glazes tint.
    let h = h0 + h1;
    let mm = m0 + m1;
    let op = mm / max(h, EPS);
    let merged_c = vec4<f32>(blend_latent(lat0, m0, lat1, m1) * op, op);
    let merged_a = vec4<f32>(h, 0.0, 0.0, 0.0);

    // Gate by the selection: the tile keeps its base value where the mask is 0 and
    // takes the stroke where it is 1 (§6.8). A *lerp of the result*, not a scaling of
    // the stroke's coverage — a half-covered mask texel must read as half of the
    // finished paint, which for an opaque brush is nothing like half its optical
    // depth. That makes the mask's own antialiasing/feather the stroke's edge.
    let sdim = vec2<i32>(textureDimensions(selection));
    let sel = textureLoad(selection, min(c, sdim - vec2<i32>(1, 1)), 0).r;

    out.color = mix(base_c, merged_c, sel);
    out.aux = mix(vec4<f32>(base_a.x, 0.0, 0.0, 0.0), merged_a, sel);
    return out;
}
