// Region fill: lay a parcel of paint through a coverage mask (§18.0.4).
//
// One fullscreen pass per written tile, and deliberately the *smallest* shader in
// the engine, because a fill has almost nothing of its own. The coverage it lands
// through was rasterized by `selection.wesl` — the identical analytic shapes, the
// identical feather ramp — and the way the parcel lands on resident paint is the
// shared law from `paint_common.wesl`, the very one the brush deposits through.
// What is left here is the multiplication of the two masks and the stack.
//
// The two masks are separate bindings rather than premultiplied on the CPU because
// they mean different things: `region` is what this gesture drew, `gate` is the
// author's selection (§6.8) — the same gate a stroke passes through, so a
// fill is clipped by a selection exactly as a brush is. Either may be a real tile
// 1:1 with the destination *or* a 1×1 constant (`SelectionRenderer::mask_for`), and
// the clamped loads read both through the same code.
//
// Unlike the transform's combine there is no cut: a fill *adds* paint, so resident
// height is untouched and the parcel stacks over it.
//
// The parcel's latent may vary with **canvas position** — the gradient fill
// (§22.4). The stops arrive already converted to this space's channels, so the
// ramp interpolates in the working space: Oklab documents blend the way the
// library's strip previews (§22.1), and Mixbox documents blend the way the
// *paint* mixes — a pigment ramp, which is the point of having pigments. Height
// does not vary: the ramp is a transition in colour, not in thickness.

import package::lib::paint_common::{blend_latent, PAINT_EPS};
import package::lib::quad::{fullscreen_clip};
import package::lib::sample::{load1};

// Most stops a gradient parcel may carry — the fitter's own cap
// (`gradient::MAX_STOPS`, asserted equal host-side §6.10).
const MAX_GRADIENT_STOPS: u32 = 16;

struct Fill {
    // .xyz = a solid parcel's latent in this space's colour channels (Oklab
    // L,a,b or Mixbox concentrations); .w = its per-unit opacity. Read only
    // when `p.y == 0` — kept in the same lanes as before the gradient existed,
    // so a solid fill's arithmetic is bit-identical to what the goldens pinned.
    c: vec4<f32>,
    // .x = paint height laid at full coverage; .y = gradient stop count (0 = a
    // solid parcel; a gradient always has ≥ 2, §22.1); .z = the axis kind
    // (0 linear, 1 radial); .w = a gradient parcel's per-unit opacity.
    p: vec4<f32>,
    // A solid parcel colour's **residual** (.xyz) — the half of it the space's
    // three channels cannot express (§6.7); .w unused. Zero for every space but
    // Mixbox. Declared unconditionally: a uniform is mirrored into Rust from the
    // unlinked source (§6.10), which has no feature set to evaluate.
    r: vec4<f32>,
    // The gradient's axis, canvas px: linear = from.xy, to.xy; radial =
    // center.xy, radius, unused.
    axis: vec4<f32>,
    // The gradient's stops: latent .xyz, ramp position t in .w — ascending,
    // first at 0, last at 1 (`Gradient`'s invariants, held by construction).
    // The length is MAX_GRADIENT_STOPS written out — the mirror generator
    // (§6.10) cannot resolve a named const in a type position.
    stop_c: array<vec4<f32>, 16>,
    // And their residuals (.xyz; .w unused), index-aligned with `stop_c`.
    stop_r: array<vec4<f32>, 16>,
}

// Where this tile's texture sits on the canvas (top-left texel, canvas px) —
// per tile where `Fill` is shared by the whole fill. Only the gradient reads
// it: a solid parcel is the same everywhere, but a ramp is a function of
// canvas position, and evaluating it *as* one is what keeps a tile's apron
// bit-identical to its neighbour's interior (§6.4) — the same argument as
// `selection.wesl`'s origin.
struct Tile {
    origin: vec4<f32>,
}

@group(0) @binding(0) var<uniform> f: Fill;
@group(0) @binding(6) var<uniform> tile: Tile;
@group(0) @binding(1) var base_color: texture_2d<f32>;
@group(0) @binding(2) var base_aux: texture_2d<f32>;
// The shape's own coverage, and the author's selection gating it.
@group(0) @binding(3) var region: texture_2d<f32>;
@group(0) @binding(4) var gate: texture_2d<f32>;
// The resident paint's residual channel (§6.7), read through the same
// clamped `load1` — bare canvas binds the 1×1 zero here as it does for the colour.
@if(resid) @group(0) @binding(5) var base_resid: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @if(resid) @location(2) resid: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy));
    let bc = load1(base_color, c);
    let ba = load1(base_aux, c).x;
    let cov = load1(region, c).r * load1(gate, c).r;
    @if(resid) let br = load1(base_resid, c);

    var out: FsOut;
    // Nothing lands here. An exact branch rather than the general arithmetic: a
    // texel outside the fill must pass through bit-for-bit, so a fill provably
    // cannot disturb paint beyond its own coverage — the same argument
    // `transform.wesl`'s untouched branch makes, and for the same reason (the
    // f32→f16 store's rounding is implementation-defined, so merely *recomputing*
    // the resident value can land an ulp off it).
    if cov == 0.0 {
        out.color = bc;
        out.aux = vec4<f32>(ba, 0.0, 0.0, 0.0);
        @if(resid) out.resid = br;
        return out;
    }

    // Resident paint, unchanged: a fill adds, it does not lift.
    let h0 = ba;
    let m0 = bc.a * h0;
    let lat0 = bc.rgb / max(bc.a, PAINT_EPS);

    // The parcel at this texel: the solid lanes verbatim, or the ramp read from
    // canvas position. The solid path takes the exact values it always did, so
    // a solid fill cannot have moved by an ulp for the gradient existing.
    var parcel_lat = f.c.rgb;
    var parcel_res = f.r.rgb;
    var parcel_op = f.c.w;
    let stops = u32(f.p.y);
    if stops >= 2 {
        let pos = tile.origin.xy + frag.xy;
        var t: f32;
        if f.p.z < 0.5 {
            // Linear: project onto the axis; constant on perpendiculars. The
            // guard keeps a degenerate axis at t = 0 rather than dividing by it.
            let d = f.axis.zw - f.axis.xy;
            let len2 = max(dot(d, d), 1e-6);
            t = clamp(dot(pos - f.axis.xy, d) / len2, 0.0, 1.0);
        } else {
            // Radial: distance from the centre, 1 at the radius. Clamped both
            // ways, so past either end the ramp holds its end stop (§22.4).
            t = clamp(distance(pos, f.axis.xy) / max(f.axis.z, 1e-6), 0.0, 1.0);
        }
        // The segment t falls in: the last stop at or before it. Stops are
        // ascending with endpoints at 0 and 1, so j+1 always exists.
        var j = 0u;
        for (var i = 1u; i + 1u < stops; i++) {
            if f.stop_c[i].w <= t {
                j = i;
            }
        }
        let a = f.stop_c[j];
        let b = f.stop_c[j + 1u];
        // Two stops sharing a position are a hard edge; the guard reads its
        // earlier side, matching `Gradient::sample`.
        let span = max(b.w - a.w, 1e-6);
        let s = clamp((t - a.w) / span, 0.0, 1.0);
        parcel_lat = mix(a.xyz, b.xyz, s);
        parcel_res = mix(f.stop_r[j].xyz, f.stop_r[j + 1u].xyz, s);
        parcel_op = f.p.w;
    }

    // Coverage scales the *height* — the amount of paint — never the per-unit
    // opacity, which is a property of the pigment and not of how much of it
    // there is (§6.1).
    let h1 = f.p.x * cov;
    let m1 = parcel_op * h1;

    let h = h0 + h1;
    let mm = m0 + m1;
    let op = mm / max(h, PAINT_EPS);
    let lat = blend_latent(lat0, m0, parcel_lat, m1);
    out.color = vec4<f32>(lat * op, op);
    out.aux = vec4<f32>(h, 0.0, 0.0, 0.0);
    // The residual stacks through the same law on the same two masses — it is the
    // rest of the same colour (§6.7).
    @if(resid) let res0 = br.rgb / max(bc.a, PAINT_EPS);
    @if(resid) out.resid = vec4<f32>(blend_latent(res0, m0, parcel_res, m1) * op, op);
    return out;
}
