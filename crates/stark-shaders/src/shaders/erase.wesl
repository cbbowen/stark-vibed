// Stroke *erase* pass (§6.12). An erase stroke rasterizes the same swept
// extent every brush does (`stamp.wesl::fs_erase`) into a per-stroke accumulator —
// one channel, the stroke's transparency mass, summed across segments *and* across
// the pieces of a live stroke — and this fullscreen pass turns that whole extent on
// the layer's resident paint at once: where the stroke would have covered the
// canvas by `w`, the paint's **visible** opacity is scaled by `1 − opacity·w`,
// and the height becomes whatever the slab law (§6.1) needs in order to show
// exactly that much. Acting on the visible opacity and inverting is the whole
// point of the pass: a `lift`-built eraser removes *height*, of which thick
// paint's coverage shows almost nothing (`1 − exp(−K·op·h)` is flat up there), so
// its knob never meant anything in the units an artist reads.
//
// Reading the **accumulated** extent rather than a per-piece parcel is what makes
// `opacity` a ceiling rather than a rate: `w` saturates at 1 however long the
// stroke works one spot, so one stroke removes at most `opacity` of what it
// finds — and a range cut anywhere gives the pieces' render exactly the whole's,
// because the accumulator carries the same sums either way (§6.2's additive form,
// held across pieces by `StrokeCarry`).
//
// A render pass with `textureLoad`s 1:1 to the destination, like `integrate.wesl`
// and for its reasons: the result is a deterministic function of canvas position,
// so a tile's apron is rewritten identically to its neighbour's interior (the seam
// invariant, §6.4).
//
// What the pass writes: **the height alone**. The per-unit opacity is a property
// of the pigment, not of how much of it there is (§6.1), and the latent and the
// residual are the pigment — erased paint is *less of the same paint*, so color
// and residual pass through untouched and only the amount falls.

import package::lib::paint_common::{OPACITY_K, PAINT_EPS};
import package::lib::quad::{fullscreen_clip};
import package::lib::sample::{load1};

struct Erase {
    // The erase opacity (`EraseEffect::opacity`, §6.12) in `.x`, and the selection's
    // strength (`Selection::strength`, §6.8) in `.y` — the whole mask's opacity,
    // which the mask texture does not carry and every gating read of it is scaled
    // by. `.zw` unused — the uniform's 16-byte round (§6.10).
    params: vec4<f32>,
}

// The layer's resident paint under this tile — its own tile, or the 1×1 zero
// standing in where the layer has none, read through the clamped load below
// (§6.8's pattern). In practice the erase renderer never draws a tile the layer
// does not have — erasing nothing is nothing — but the stand-in keeps the
// contract the same as every other tile pass.
@group(0) @binding(0) var base_color: texture_2d<f32>;
@group(0) @binding(1) var base_aux: texture_2d<f32>;
// The stroke's accumulated transparency mass (`stamp.wesl::fs_erase`), 1:1 with
// the destination, apron included.
@group(0) @binding(2) var accum: texture_2d<f32>;
// The selection's coverage over this tile (§6.8): 1 where the stroke may act,
// 0 where it may not, feathered in between — this tile's own mask or the 1×1
// constant, exactly as `integrate.wesl` binds it.
@group(0) @binding(3) var selection: texture_2d<f32>;
@group(0) @binding(4) var<uniform> e: Erase;
// The resident color's residual half (§6.7), in a space that has one. Bound
// so it can pass through the rewrite whole — an erase changes no pigment.
@if(resid) @group(0) @binding(5) var base_resid: texture_2d<f32>;

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
    let m_e = textureLoad(accum, c, 0).x;

    var base_r = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    @if(resid) { base_r = load1(base_resid, c); }

    var out: FsOut;
    out.color = base_c;
    @if(resid) out.resid = base_r;

    // The stroke's extent missed this texel: pass the height through exactly,
    // for `integrate.wesl`'s reason — the arithmetic below rounds through the
    // f32→f16 store, and a hundred strokes across a tile would each nudge paint
    // they never touched.
    if m_e == 0.0 {
        out.aux = vec4<f32>(base_a.x, 0.0, 0.0, 0.0);
        return out;
    }

    // Resident paint: height, optical mass, and what shows through it.
    let h0 = base_a.x;
    let m0 = base_c.a * h0;
    let t0 = exp(-OPACITY_K * m0);

    // The coverage the stroke's extent would have covered this texel by — the
    // slab law over the accumulated mass, so the eraser's edge is the very
    // profile its brush would have painted.
    let w = 1.0 - exp(-OPACITY_K * m_e);

    // Opacity incorporating the brush opacity and selection mask.
    //
    // Gate by the selection (§6.8): a lerp of the *result*, `integrate.wesl`'s
    // rule. Color and residual are the base's either way, so the height is the
    // only lane with two answers to mix.
    // At the mask's own strength, exactly as `integrate.wesl` reads it: the tiles
    // hold what the ops struck, and how strongly that gates is one number on top.
    let effective_opacity = e.params.x * load1(selection, c).r;

    // What must remain visible is `(1 − t0)·(1 − opacity·w)`; carried as what
    // shows *through* — the old transmittance plus the newly uncovered share —
    // so the small quantity is the one computed and nothing cancels where the
    // paint is nearly opaque.
    let through = t0 + (1.0 - t0) * effective_opacity * w;
    // Inverted through the slab law into the mass that shows exactly that much
    // (`fill.wesl`'s inversion, run the other way). `through >= t0` by
    // construction, so `m1 <= m0` up to rounding — the `min` states it outright,
    // and it is also what answers a `through` that underflowed to zero under
    // extreme impasto: an eraser must never add paint.
    let m1 = min(-log(through) / OPACITY_K, m0);
    // The same per-unit opacity, less of it: only the amount falls (§6.1).
    let h1 = h0 * (m1 / max(m0, PAINT_EPS));

    out.aux = vec4<f32>(h1, 0.0, 0.0, 0.0);
    return out;
}
