// Transforms of the selected paint: the whole-plane affine (§16), and the
// rect-scoped perspective (§16.8) and warp (§16.9).
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

import package::lib::paint_common::{blend_latent, PAINT_EPS};
import package::lib::quad::{fullscreen_clip};
import package::lib::sample::{load1};

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

// ---- The rect-scoped maps: perspective (§16.8) and warp (§16.9). ----
//
// Same architecture as the affine — forward-rasterized source pieces, an
// inverse map per fragment — with two differences: piece corners arrive
// precomputed (the CPU shares corner values between adjacent pieces, which is
// what keeps rasterization watertight when the map is not affine), and every
// deposit is additionally scaled by the source rect's coverage `box(s)`, so
// paint outside the rect is neither cut nor carried.

struct Gated {
    // a: destination tile texture's top-left in canvas px .xy; .z = TILE_TEX;
    //    .w = fragment mode: 0 = homography (i0..i2 are the inverse's rows —
    //    affine pieces embed a (0,0,1) bottom row, so their divide is by
    //    exactly 1 and the §16.4 tap exactness carries over); 1 = inverse
    //    bilinear over the cell corners in g0/g1.
    a: vec4<f32>,
    // The piece's corner images in canvas px: (00, 10) and (01, 11).
    c0: vec4<f32>,
    c1: vec4<f32>,
    // Inverse homography rows (mode 0): s = (i0·p̃, i1·p̃) / (i2·p̃).
    i0: vec4<f32>,
    i1: vec4<f32>,
    i2: vec4<f32>,
    // Warp cell corner images (mode 1): (g00, g10) and (g01, g11).
    g0: vec4<f32>,
    g1: vec4<f32>,
    // Warp cell source sub-rect: min .xy, size .zw.
    s: vec4<f32>,
    // The map's source rect: min .xy, max .zw — the coverage gate.
    r: vec4<f32>,
    // Source tile texture origin .xy (interior origin − apron, canvas px).
    t: vec4<f32>,
}

@group(0) @binding(0) var<uniform> qg: Gated;

// Coverage of an axis-aligned box with a half-pixel antialiasing ramp: exactly
// 1 deeper than half a pixel inside, exactly 0 outside, a 1-px ramp on the
// boundary. A pure function of canvas position, so the cut side and the paste
// side compute the identical gate and identity recombines exactly (§16.4).
fn box_coverage(p: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    let d = min(p - lo, hi - p);
    return clamp(min(d.x, d.y) + 0.5, 0.0, 1.0);
}

// The piece's corners arrive precomputed rather than transformed here: under a
// warp adjacent pieces belong to *different* bilinear cells, and only the CPU,
// evaluating shared corners once from shared lattice values, can guarantee the
// bitwise equality watertight rasterization needs.
@vertex
fn vs_gated(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var c: vec2<f32>;
    switch vi {
        case 0u: { c = qg.c0.xy; }
        case 1u: { c = qg.c0.zw; }
        case 2u: { c = qg.c1.xy; }
        default: { c = qg.c1.zw; }
    }
    let tex_px = (c - qg.a.xy) / qg.a.z;
    return vec4<f32>(tex_px.x * 2.0 - 1.0, 1.0 - tex_px.y * 2.0, 0.0, 1.0);
}

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// The source canvas position for this fragment: its own canvas centre mapped
// back through the inverse of the piece's map.
fn gated_src(frag: vec2<f32>) -> vec2<f32> {
    let p = qg.a.xy + frag;
    if qg.a.w < 0.5 {
        // Homography (or embedded affine, whose w is exactly 1).
        let w = qg.i2.x * p.x + qg.i2.y * p.y + qg.i2.z;
        return vec2<f32>(
            (qg.i0.x * p.x + qg.i0.y * p.y + qg.i0.z) / w,
            (qg.i1.x * p.x + qg.i1.y * p.y + qg.i1.z) / w,
        );
    }
    // Inverse bilinear over the cell's corner images. Everything is computed
    // in differences (cell-sized numbers), so large canvas coordinates never
    // meet a cross product. The quadratic uses the numerically stable split;
    // validity (positive Jacobian, §16.9) puts exactly one root in the cell.
    let e = qg.g0.zw - qg.g0.xy;
    let f = qg.g1.xy - qg.g0.xy;
    let g = qg.g1.zw - qg.g0.zw - qg.g1.xy + qg.g0.xy;
    let h = p - qg.g0.xy;
    let k2 = cross2(g, f);
    let k1 = cross2(e, f) + cross2(h, g);
    let k0 = cross2(h, e);
    var v: f32;
    if abs(k2) < 1e-10 {
        v = -k0 / k1;
    } else {
        let disc = max(k1 * k1 - 4.0 * k0 * k2, 0.0);
        let sq = sqrt(disc);
        var s: f32;
        if k1 >= 0.0 { s = 1.0; } else { s = -1.0; }
        let q = -0.5 * (k1 + s * sq);
        let va = q / k2;
        var vb: f32;
        if abs(q) > 1e-20 { vb = k0 / q; } else { vb = va; }
        if abs(va - 0.5) <= abs(vb - 0.5) { v = va; } else { v = vb; }
    }
    let den = e + g * v;
    var u: f32;
    if abs(den.x) >= abs(den.y) {
        u = (h.x - f.x * v) / den.x;
    } else {
        u = (h.y - f.y * v) / den.y;
    }
    return qg.s.xy + vec2<f32>(u, v) * qg.s.zw;
}

// The moved parcel under a rect-scoped map: the lift law's `(latent, op,
// h·m·box)` — coverage of the author's mask *and* of the source rect scale the
// height, never the colour (§16.2).
@fragment
fn fs_parcel_gated(@builtin(position) frag: vec4<f32>) -> ParcelOut {
    let s = gated_src(frag.xy);
    let uv = (s - qg.t.xy) / qg.a.z;
    let c = textureSampleLevel(src_color, samp, uv, 0.0);
    let h = textureSampleLevel(src_aux, samp, uv, 0.0).x;
    let m = textureSampleLevel(src_mask, samp, uv, 0.0).r * box_coverage(s, qg.r.xy, qg.r.zw);
    var out: ParcelOut;
    out.color = c;
    out.aux = vec4<f32>(h * m, 0.0, 0.0, 0.0);
    return out;
}

// The moved selection coverage — drawn with **max** blending over the residue
// `fs_mask_base` lays down, the soft union of what stayed and what landed
// (§16.8).
@fragment
fn fs_mask_gated(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let s = gated_src(frag.xy);
    let uv = (s - qg.t.xy) / qg.a.z;
    let m = textureSampleLevel(src_mask, samp, uv, 0.0).r * box_coverage(s, qg.r.xy, qg.r.zw);
    return vec4<f32>(m, 0.0, 0.0, 0.0);
}

// The residue: what the destination's old coverage keeps where the rect's cut
// did not reach — `old · (1 − box)`. Fullscreen, drawn before the moved quads.
@fragment
fn fs_mask_base(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = qg.a.xy + frag.xy;
    let m = load1(src_mask, vec2<i32>(floor(frag.xy))).r;
    return vec4<f32>(m * (1.0 - box_coverage(p, qg.r.xy, qg.r.zw)), 0.0, 0.0, 0.0);
}

@vertex
fn vs_fill(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// The combine's one map-dependent fact: whether the cut is gated by a source
// rect (perspective/warp), and by which. `a.z = 0` is the affine's whole-plane
// cut, arithmetically untouched from before the gate existed.
struct Combine {
    // a: destination tile texture origin .xy; .z = 1 to gate the cut.
    a: vec4<f32>,
    // r: the gate rect, min .xy, max .zw.
    r: vec4<f32>,
}

@group(0) @binding(7) var<uniform> qc: Combine;

@fragment
fn fs_combine(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy));
    // Base and mask clamp to their own extent, so a real tile and the 1×1
    // constants read through the same code (§6.8's pattern, `lib::sample::load1`).
    let bc = load1(base_color, c);
    let ba = load1(base_aux, c);
    var m = load1(base_mask, c).r;
    if qc.a.z > 0.5 {
        // The cut only reaches inside the map's source rect (§16.8): the
        // effective coverage is m·box, the same gate the parcel side applied
        // at the source — so an identity map recombines exactly.
        m = m * box_coverage(qc.a.xy + frag.xy, qc.r.xy, qc.r.zw);
    }
    let pc = load1(parcel_color, c);
    let ph = load1(parcel_aux, c).x;

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
    let lat0 = bc.rgb / max(bc.a, PAINT_EPS);
    // The parcel: (latent, op, height) as fs_parcel wrote it.
    let h1 = ph;
    let m1 = pc.a * h1;
    let lat1 = pc.rgb / max(pc.a, PAINT_EPS);

    let h = h0 + h1;
    let mm = m0 + m1;
    let op = mm / max(h, PAINT_EPS);
    let lat = blend_latent(lat0, m0, lat1, m1);
    out.color = vec4<f32>(lat * op, op);
    out.aux = vec4<f32>(h, 0.0, 0.0, 0.0);
    return out;
}
