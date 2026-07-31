// Affine transform of the selected paint (§16).
//
// Two passes per rewritten tile, plus one for the mask:
//
// - **Parcel** (`vs_quad` + `fs_parcel`): each selected source tile's *interior*
//   is one quad under the affine, drawn into the destination tile's scratch pair.
//   Source interiors tile the plane, so their images tile the transformed plane:
//   quads share transformed edges bitwise and rasterization is watertight — every
//   destination texel receives **at most one** parcel, which is what lets this be
//   a plain write with no blending and no order dependence. Sampling is bilinear;
//   the 1-px apron feeds the filter's edge taps (§6.4).
// - **Combine** (`vs_fill` + `fs_combine`): cut the base by its own mask — the
//   lift law, `h₀ = h·(1−m)` with latent and per-unit opacity untouched
//   (§6.1) — then stack the parcel by the shared parcel-deposit law
//   (paint_common.wesl). A texel neither cut nor pasted passes through
//   bit-exactly, so a far-away transform provably cannot disturb paint.
// - **Mask** (`vs_quad` + `fs_mask`): the author's selection mask resampled under
//   the same affine, so the selection follows the paint it holds.
//
// Every output is a pure function of canvas position, so a tile's apron is
// rewritten identically to its neighbour's interior (the §6.4 seam invariant).

import package::paint_common::{blend_latent, EPS};

struct Quad {
    // a: destination tile texture's top-left in canvas px .xy; .z = TILE_TEX.
    a: vec4<f32>,
    // m: the forward affine's linear part, rows: canvas.x = m.x·p.x + m.y·p.y + t.x,
    //    canvas.y = m.z·p.x + m.w·p.y + t.y. Vertex-stage only: it places the quad.
    m: vec4<f32>,
    // t: the forward translation .xy; the source tile *texture* origin .zw
    //    (interior origin − apron, canvas px).
    t: vec4<f32>,
    // u: source interior origin .xy (canvas px); .z = TILE_SIZE (interior edge).
    u: vec4<f32>,
    // i/j: the inverse affine (rows / translation), fragment-stage. The fragment
    //    maps its own canvas centre back through the inverse rather than
    //    interpolating a uv across the quad: attribute interpolation plus the
    //    sampler's fixed-point uv conversion can land an exact-centre tap a
    //    fraction off, and the exactness invariants (identity, integer translate,
    //    flips — §16.4) live or die on those taps being exact.
    i: vec4<f32>,
    j: vec4<f32>,
}

@group(0) @binding(0) var<uniform> q: Quad;
// Bilinear clamp sampler for the source taps.
@group(0) @binding(1) var samp: sampler;

// The source tile being carried: its channels and the selection mask over it.
// `src_mask` is either the mask tile 1:1 with the source (apron included) or a
// 1×1 constant (fully-selected tiles); a clamped sample reads both.
@group(1) @binding(0) var src_color: texture_2d<f32>;
@group(1) @binding(1) var src_aux: texture_2d<f32>;
@group(1) @binding(2) var src_mask: texture_2d<f32>;

// Combine inputs (a separate pipeline; these bindings are unused by the quad
// passes and vice versa). Every input may be a real tile 1:1 with the
// destination *or* a 1×1 constant — virgin-canvas destinations bind zero base
// channels, cut-only tiles bind zero parcels — so all five read via clamped
// loads.
@group(0) @binding(2) var base_color: texture_2d<f32>;
@group(0) @binding(3) var base_aux: texture_2d<f32>;
@group(0) @binding(4) var base_mask: texture_2d<f32>;
@group(0) @binding(5) var parcel_color: texture_2d<f32>;
@group(0) @binding(6) var parcel_aux: texture_2d<f32>;

// One source-tile interior as a transformed quad, placed in the destination
// tile's texture (origin = interior origin − apron, edge TILE_TEX). The quad
// only provides *coverage*; the fragment derives its own source tap.
@vertex
fn vs_quad(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    // The source interior corner in canvas px. Corner arithmetic is exact in f32,
    // so two quads sharing a source edge transform to bitwise-equal vertices and
    // rasterize watertight.
    let p = q.u.xy + corner * q.u.z;
    let c = vec2<f32>(
        q.m.x * p.x + q.m.y * p.y + q.t.x,
        q.m.z * p.x + q.m.w * p.y + q.t.y,
    );
    let tex_px = (c - q.a.xy) / q.a.z;
    return vec4<f32>(tex_px.x * 2.0 - 1.0, 1.0 - tex_px.y * 2.0, 0.0, 1.0);
}

// The source uv for this fragment: its canvas centre mapped back through the
// inverse affine into the source tile's texture. For the identity, an integer
// translation, or an axis flip about the origin this is exact fixed-grid
// arithmetic — texel centres map to texel centres and the bilinear tap
// degenerates to the stored value, which is what makes those transforms
// byte-exact. Edge taps fall into the source's apron (real neighbour content),
// never a clamp.
fn src_uv(frag: vec2<f32>) -> vec2<f32> {
    let p = q.a.xy + frag;
    let s = vec2<f32>(
        q.i.x * p.x + q.i.y * p.y + q.j.x,
        q.i.z * p.x + q.i.w * p.y + q.j.y,
    );
    return (s - q.t.zw) / q.a.z;
}

struct ParcelOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// The moved parcel: `(latent, op, h·m)` — the cut takes height with it, the
// paint's latent and per-unit opacity ride along unscaled (the lift law,
// §16.2).
@fragment
fn fs_parcel(@builtin(position) frag: vec4<f32>) -> ParcelOut {
    let uv = src_uv(frag.xy);
    let c = textureSampleLevel(src_color, samp, uv, 0.0);
    let h = textureSampleLevel(src_aux, samp, uv, 0.0).x;
    let m = textureSampleLevel(src_mask, samp, uv, 0.0).r;
    var out: ParcelOut;
    out.color = c;
    out.aux = vec4<f32>(h * m, 0.0, 0.0, 0.0);
    return out;
}

// The selection mask itself, carried under the same affine (§16.1).
@fragment
fn fs_mask(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let m = textureSampleLevel(src_mask, samp, src_uv(frag.xy), 0.0).r;
    return vec4<f32>(m, 0.0, 0.0, 0.0);
}

@vertex
fn vs_fill(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@fragment
fn fs_combine(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy));
    // Base and mask clamp to their own extent, so a real tile and the 1×1
    // constants read through the same code (§6.8's pattern).
    let bdim = vec2<i32>(textureDimensions(base_color));
    let bc = textureLoad(base_color, min(c, bdim - vec2<i32>(1, 1)), 0);
    let adim = vec2<i32>(textureDimensions(base_aux));
    let ba = textureLoad(base_aux, min(c, adim - vec2<i32>(1, 1)), 0);
    let mdim = vec2<i32>(textureDimensions(base_mask));
    let m = textureLoad(base_mask, min(c, mdim - vec2<i32>(1, 1)), 0).r;
    let pdim = vec2<i32>(textureDimensions(parcel_color));
    let pc = textureLoad(parcel_color, min(c, pdim - vec2<i32>(1, 1)), 0);
    let hdim = vec2<i32>(textureDimensions(parcel_aux));
    let ph = textureLoad(parcel_aux, min(c, hdim - vec2<i32>(1, 1)), 0).x;

    var out: FsOut;
    // The two ends of the coverage ramp take **exact** branches rather than the
    // general arithmetic. Not an optimization: each is algebraically what the
    // arithmetic below computes, but the f32→f16 render-target store's rounding
    // is implementation-defined, so a value that merely *recomputes* the answer
    // can land one f16 ulp off it. The exactness invariants
    // (§16.4) hold exactly on fully-selected and fully-
    // unselected texels; inside a feather ramp they hold to one f16 ulp.
    //
    // Untouched: not under the selection, and no parcel landed here.
    if m == 0.0 && pc.a == 0.0 && ph == 0.0 {
        out.color = bc;
        out.aux = vec4<f32>(ba.x, 0.0, 0.0, 0.0);
        return out;
    }
    // Fully selected: the cut leaves nothing (`h₀ = 0`), so the result *is* the
    // parcel.
    if m == 1.0 {
        out.color = pc;
        out.aux = vec4<f32>(ph, 0.0, 0.0, 0.0);
        return out;
    }

    // The cut: thickness leaves, the remaining paint's latent and per-unit
    // opacity are untouched (the lift law, §6.1).
    let h0 = ba.x * (1.0 - m);
    let m0 = bc.a * h0;
    let lat0 = bc.rgb / max(bc.a, EPS);
    // The parcel: (latent, op, height) as fs_parcel wrote it.
    let h1 = ph;
    let m1 = pc.a * h1;
    let lat1 = pc.rgb / max(pc.a, EPS);

    let h = h0 + h1;
    let mm = m0 + m1;
    let op = mm / max(h, EPS);
    let lat = blend_latent(lat0, m0, lat1, m1);
    out.color = vec4<f32>(lat * op, op);
    out.aux = vec4<f32>(h, 0.0, 0.0, 0.0);
    return out;
}
