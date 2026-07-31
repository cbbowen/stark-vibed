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

import package::paint_common::{blend_latent, EPS};

struct Fill {
    // .xyz = the parcel's latent in this space's colour channels (Oklab L,a,b or
    // Mixbox concentrations); .w = its per-unit opacity.
    c: vec4<f32>,
    // .x = paint height laid at full coverage; .yzw unused.
    p: vec4<f32>,
}

@group(0) @binding(0) var<uniform> f: Fill;
@group(0) @binding(1) var base_color: texture_2d<f32>;
@group(0) @binding(2) var base_aux: texture_2d<f32>;
// The shape's own coverage, and the author's selection gating it.
@group(0) @binding(3) var region: texture_2d<f32>;
@group(0) @binding(4) var gate: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// Load clamped to the bound texture's own extent, so a real tile and a 1×1
// constant read identically (§6.8's pattern).
fn load1(t: texture_2d<f32>, c: vec2<i32>) -> vec4<f32> {
    let dim = vec2<i32>(textureDimensions(t));
    return textureLoad(t, min(c, dim - vec2<i32>(1, 1)), 0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let c = vec2<i32>(floor(frag.xy));
    let bc = load1(base_color, c);
    let ba = load1(base_aux, c).x;
    let cov = load1(region, c).r * load1(gate, c).r;

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
        return out;
    }

    // Resident paint, unchanged: a fill adds, it does not lift.
    let h0 = ba;
    let m0 = bc.a * h0;
    let lat0 = bc.rgb / max(bc.a, EPS);
    // The parcel. Coverage scales the *height* — the amount of paint — never the
    // per-unit opacity, which is a property of the pigment and not of how much of
    // it there is (§6.1).
    let h1 = f.p.x * cov;
    let m1 = f.c.w * h1;

    let h = h0 + h1;
    let mm = m0 + m1;
    let op = mm / max(h, EPS);
    let lat = blend_latent(lat0, m0, f.c.rgb, m1);
    out.color = vec4<f32>(lat * op, op);
    out.aux = vec4<f32>(h, 0.0, 0.0, 0.0);
    return out;
}
