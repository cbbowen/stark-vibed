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
// does not vary: the ramp is a transition in color, not in thickness.

import package::lib::latent::{Latent, latent_blend, latent_lerp, latent_of, latent_scale, latent_unpremult};
import package::lib::paint_common::{OPACITY_K, OPAQUE_MASS, PAINT_EPS};
import package::lib::quad::{fullscreen_clip};
import package::lib::ramp_common::{ramp_position};
import package::lib::sample::{load1};

// Most stops a gradient parcel may carry — the fitter's own cap
// (`gradient::MAX_STOPS`, asserted equal host-side §6.10).
const MAX_GRADIENT_STOPS: u32 = 16;

// Where a fill declares "opaque" to be is `paint_common::OPAQUE_MASS`, imported
// above rather than declared here: the host builds an imported image's tiles by
// this same inversion with no shader to import from (§23), so the number has to be
// readable from both sides. The matte slab's thickness is the identical arithmetic
// for the identical reason (`matte.wesl`, §15.4).

struct Fill {
    // .xyz = a solid parcel's latent in this space's color channels (Oklab
    // L,a,b or Mixbox concentrations), read only when `p.y == 0`; .w unused —
    // a parcel carries color alone, the strength is `p.x` below.
    c: vec4<f32>,
    // .x = the fill's **opacity**: the visible alpha it covers with where
    // coverage is full (`FillOp::opacity`); .y = gradient stop count (0 = a
    // solid parcel; a gradient always has ≥ 2, §22.1); .z = the axis kind
    // (0 linear, 1 radial); .w = the author's selection opacity
    // (`Selection::opacity`, §6.8) — the whole mask's, which `gate`'s texels do
    // not carry.
    p: vec4<f32>,
    // A solid parcel color's **residual** (.xyz) — the half of it the space's
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
// clamped `load1` — bare canvas binds the 1×1 zero here as it does for the color.
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
    // The gate at the mask's opacity — a *gating* read, so the whole mask's
    // opacity multiplies it (§6.8): a fill is a saturated parcel, so its stated
    // coverage times the mask *is* the opacity ceiling a stroke reads off the same
    // mask (`integrate.wesl`), with `w = 1`. On `cov` rather than folded into
    // `f.p.x`, though the two are the same product: the exact passthrough below
    // keys on `cov == 0`, and a mask read at opacity 0 must take that branch
    // rather than recompute the resident paint and land an ulp off it.
    let cov = load1(region, c).r * load1(gate, c).r * f.p.w;
    // The base's color, both halves (§6.7, `lib::latent`). The gate is on the load —
    // a residual texture exists only in a space that has one — and the arithmetic
    // below is then written once.
    var base = latent_of(bc.rgb, vec3<f32>(0.0));
    // Kept whole for the passthrough, which must not recompute anything: the residual
    // target carries its own copy of the alpha.
    var br = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    @if(resid) {
        br = load1(base_resid, c);
        base.res = br.rgb;
    }

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
    let col0 = latent_unpremult(base, bc.a, PAINT_EPS);

    // The parcel at this texel: the solid lanes verbatim, or the ramp read from
    // canvas position. The solid path takes the exact values it always did, so
    // a solid fill cannot have moved by an ulp for the gradient existing.
    // The parcel's color. Both halves come off the uniform, which declares the
    // residual whether or not the space uses it (§6.10) — so this needs no gate at all.
    var parcel = latent_of(f.c.rgb, f.r.rgb);
    let stops = u32(f.p.y);
    if stops >= 2 {
        let pos = tile.origin.xy + frag.xy;
        let t = ramp_position(pos, f.axis, f.p.z);
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
        parcel = latent_lerp(
            latent_of(a.xyz, f.stop_r[j].xyz),
            latent_of(b.xyz, f.stop_r[j + 1u].xyz),
            s,
        );
    }

    // What the fill asks for at this texel: its opacity through its own region and
    // through the author's gate. A **coverage** — the quantity the eye reads and
    // the one the slider is labelled with — not yet an amount of paint.
    let want = min(f.p.x * cov, 1.0 - exp(-OPACITY_K * OPAQUE_MASS));

    // The paint that lands. A fill lays **fully opaque paint**: per-unit opacity 1,
    // so its optical mass and its height are one number, and that number is
    // whatever the slab law needs in order to *be* `want` — the inversion
    // `1 − exp(−K·m) = w` ⇒ `m = −ln(1 − w)/K`, the same one a blended merge-down
    // runs (`slab.wesl::fs_store`). The cap above is what keeps the logarithm
    // finite, exactly as `fs_store`'s guard does.
    //
    // Two things fall out, and both are the point. A fill at opacity 1 *covers* —
    // no amount of the brush's flow ever did, since flow enters through the same
    // exponential and 95% is what its whole range buys. And coverage now scales
    // the *paint* while landing linearly on the *canvas*: a feathered edge thins
    // from eight units to none, and reads as precisely the ramp
    // `selection.wesl` rasterized, because `want` is linear in `cov`.
    let m1 = -log(1.0 - want) / OPACITY_K;
    let h1 = m1;

    let h = h0 + h1;
    let mm = m0 + m1;
    let op = mm / max(h, PAINT_EPS);
    // The residual stacks through the same law on the same two masses — it is the rest
    // of the same color (§6.7), and one call is what says so.
    let merged = latent_scale(latent_blend(col0, m0, parcel, m1), op);
    out.color = vec4<f32>(merged.lat, op);
    out.aux = vec4<f32>(h, 0.0, 0.0, 0.0);
    @if(resid) out.resid = vec4<f32>(merged.res, op);
    return out;
}
