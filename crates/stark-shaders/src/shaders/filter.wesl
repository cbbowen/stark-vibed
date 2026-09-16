// Filter layers (§21), in whichever working space the document is in (§6.7).
//
// **One module, linked twice.** The adjustment, the strength mix, the chromatic
// gather's weights and the focal blur's planes are `filter_common.wesl`'s and are the
// same in every space; a space supplies only its channels ↔ **Oklab** (the space the
// knobs are defined in, §21.5) and, for the two filters that resample, its channels ↔
// **linear light** (a sum over wavelengths being a sum of light). Plus, where it has
// one, the residual that rides beside them.
//
// The five conversions below are the whole of the difference. In Oklab three of them
// are the identity or nearly so — the channels are already Oklab, so a point filter
// pays no bracket at all — and in pigment each costs the polynomial and its inverse
// LUT, per texel for a point filter and per *tap* for the gather.
//
// One consequence the medium owns: pigment cannot be brighter than the light falling on
// it, and the LUT is defined on `[0,1]` sRGB, so a positive **exposure** saturates at
// white in a pigment document rather than pushing past it into the media pass's
// highlight roll-off. Paint does not glow.

import package::lib::color::{linear_to_light, oklab_to_linear_srgb, linear_srgb_to_oklab};
@if(!pigment) import package::lib::color::{floor_light};
@if(pigment) import package::lib::mixbox::{Pigment, mixbox_to_linear, mixbox_from_linear};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS, PAINT_EPS};
import package::filter_common::{
    f, FILTER_CHROMATIC, FILTER_FOCAL_BLUR, back_color, back_aux, back_samp,
    filtered, resolve, weigh, is_clipped, ca_taps, ca_s, ca_uv, ca_weight, ca_lum,
    BlurOut, blur_pad, blur_read, blur_src,
};

// **Group 1, whole**: `filter_common` owns group 0 entire and this module owns this one
// entire (§6.10). A colorimetric filter declares nothing here and binds an empty layout
// in its place.
@if(pigment) @group(1) @binding(0) var pigment_lut: texture_2d<f32>;
@if(pigment) @group(1) @binding(1) var pigment_samp: sampler;
@if(pigment) @group(1) @binding(2) var back_resid: texture_2d<f32>;

// The targets this space's filter writes: `filter_common::FsOut`'s two, plus the
// residual where the space has one.
struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @if(pigment) @location(2) resid: vec4<f32>,
}

// A color as this space stores it: its three channels, and — in a pigment space — the
// rest of the same color (§6.7).
struct Chan {
    lat: vec3<f32>,
    @if(pigment) res: vec3<f32>,
}

// —— what a space supplies ———————————————————————————————————————————————————————
//
// Five, not two, because the two resampling filters arrive in **light** and the point
// filters arrive in **Oklab**, and only one of those trips is free to be undone: a
// colorimetric space's channels *are* Oklab, so `to_lab`/`from_lab` must be the
// identity there rather than a round trip that happens to land close.

@if(!pigment) fn to_lab(c: Chan) -> vec3<f32> {
    return c.lat;
}

@if(!pigment) fn from_lab(lab: vec3<f32>) -> Chan {
    return Chan(lab);
}

// A tap's light, as linear sRGB. Floored **in light**, not per channel: light does not
// go negative, but a wide-gamut color's sRGB channel does (§6.5).
@if(!pigment) fn tap_linear(c: Chan) -> vec3<f32> {
    return floor_light(oklab_to_linear_srgb(c.lat));
}

@if(!pigment) fn to_light(c: Chan) -> vec3<f32> {
    return max(linear_to_light(oklab_to_linear_srgb(c.lat)), vec3<f32>(0.0));
}

@if(!pigment) fn from_linear(lin: vec3<f32>) -> Chan {
    return Chan(linear_srgb_to_oklab(lin));
}

@if(pigment) fn to_lab(c: Chan) -> vec3<f32> {
    return linear_srgb_to_oklab(mixbox_to_linear(Pigment(c.lat, c.res)));
}

@if(pigment) fn from_lab(lab: vec3<f32>) -> Chan {
    return from_linear(oklab_to_linear_srgb(lab));
}

// Already non-negative: `mixbox_to_linear` clamps to the sRGB cube on the way through,
// so there is nothing left to floor.
@if(pigment) fn tap_linear(c: Chan) -> vec3<f32> {
    return mixbox_to_linear(Pigment(c.lat, c.res));
}

@if(pigment) fn to_light(c: Chan) -> vec3<f32> {
    return linear_to_light(mixbox_to_linear(Pigment(c.lat, c.res)));
}

// The residual is **recomputed** here rather than carried: it is what the polynomial
// could not say about the color that arrived, and the color that arrived is the
// adjusted one (§6.7).
@if(pigment) fn from_linear(lin: vec3<f32>) -> Chan {
    let p = mixbox_from_linear(pigment_lut, pigment_samp, lin);
    return Chan(p.conc, p.res);
}

// —— this module's own ————————————————————————————————————————————————————————————

// `filter_common::resolve` with the residual beside it — the whole tail of a filter
// that has no opinion about coverage, which is every point filter and every clipped one
// (§21.4). Both halves take the identical strength mix on the identical coverage, and
// both ride the alpha `resolve` already decided.
fn resolve_chan(cb: vec4<f32>, was: Chan, now: Chan, h: f32) -> FsOut {
    let base = resolve(cb, was.lat, now.lat, h);
    var out: FsOut;
    out.color = base.color;
    out.aux = base.aux;
    @if(pigment) out.resid = vec4<f32>(weigh(was.res, now.res) * cb.a, cb.a);
    return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (channels · coverage, coverage)
    @if(pigment) let rb = textureLoad(back_resid, p, 0);
    let h = textureLoad(back_aux, p, 0).x;

    if f.kind == FILTER_CHROMATIC {
        // The spectral gather (§21.10): each tap is one band of the dispersed spectrum
        // — its light in linear sRGB premultiplied by the coverage it arrived with,
        // plus that coverage and height on the scalar weight — normalized per channel
        // at the end so a flat region comes back exact.
        let dim = vec2<f32>(textureDimensions(back_color));
        let n = ca_taps(f.disp);
        var lin_acc = vec3<f32>(0.0); // Σ w · light · coverage (premultiplied light)
        var w_acc = vec3<f32>(0.0); // Σ w, the per-channel partition of unity
        var cov_acc = 0.0; // Σ y · coverage
        var h_acc = 0.0; // Σ y · height
        var y_acc = 0.0; // Σ y
        for (var i = 0u; i < n; i++) {
            let s = ca_s(i, n);
            let uv = ca_uv(frag.xy, s, dim);
            let cs = textureSampleLevel(back_color, back_samp, uv, 0.0);
            @if(pigment) let rs = textureSampleLevel(back_resid, back_samp, uv, 0.0);
            let hs = textureSampleLevel(back_aux, back_samp, uv, 0.0).x;
            let w = ca_weight(s);
            let y = ca_lum(w);
            var tap: Chan;
            tap.lat = cs.rgb / max(cs.a, COVERAGE_EPS);
            @if(pigment) tap.res = rs.rgb / max(cs.a, COVERAGE_EPS);
            lin_acc += w * tap_linear(tap) * cs.a;
            w_acc += w;
            cov_acc += y * cs.a;
            h_acc += y * hs;
            y_acc += y;
        }
        let a = cov_acc / y_acc;
        // Out of light and back into this space's channels, once, for the summed
        // spectrum.
        let now = from_linear(lin_acc / w_acc / max(a, COVERAGE_EPS));
        // **Clipped**: the fringe's color lands, its displaced coverage does not
        // (§21.4). The gather's own `a` is spent on un-premultiplying the light it
        // summed and then dropped, so what is left is a color for the coverage that was
        // already here — which is exactly the point filters' tail, taken verbatim.
        if is_clipped() {
            let cov = max(cb.a, COVERAGE_EPS);
            var was: Chan;
            was.lat = cb.rgb / cov;
            @if(pigment) was.res = rb.rgb / cov;
            return resolve_chan(cb, was, now, h);
        }
        // The strength mix on the raw premultiplied lanes — color, coverage and height
        // together, because this filter moved all three — so strength 0 is `cb` itself,
        // bit for bit (§21.4).
        var out: FsOut;
        out.color = mix(cb, vec4<f32>(now.lat * a, a), f.strength);
        out.aux = vec4<f32>(mix(h, h_acc / y_acc, f.strength), 0.0, 0.0, 0.0);
        @if(pigment) out.resid = mix(rb, vec4<f32>(now.res * a, a), f.strength);
        return out;
    }

    if f.kind == FILTER_FOCAL_BLUR {
        // The convolution already ran in light (§21.12, `blur.wesl`): one read per lane
        // — exact, or interpolated up from a decimated transform — and the trip back
        // into this space's channels. Like the gather, this filter moved color,
        // coverage and height together.
        let blurred = blur_read(frag.xy);
        let now = from_linear(blurred.lin);
        if is_clipped() {
            let cov = max(cb.a, COVERAGE_EPS);
            var was: Chan;
            was.lat = cb.rgb / cov;
            @if(pigment) was.res = rb.rgb / cov;
            return resolve_chan(cb, was, now, h);
        }
        var out: FsOut;
        out.color = mix(cb, vec4<f32>(now.lat * blurred.cov, blurred.cov), f.strength);
        out.aux = vec4<f32>(mix(h, blurred.h, f.strength), 0.0, 0.0, 0.0);
        @if(pigment) out.resid = mix(rb, vec4<f32>(now.res * blurred.cov, blurred.cov), f.strength);
        return out;
    }

    let cov = max(cb.a, COVERAGE_EPS);
    var was: Chan;
    was.lat = cb.rgb / cov;
    @if(pigment) was.res = rb.rgb / cov;
    return resolve_chan(cb, was, from_lab(filtered(to_lab(was))), h);
}

// The focal blur's decode (§21.12): the accumulator laid into `blur.wesl`'s
// spatial-domain planes — premultiplied XYZ light, coverage, height, and the border
// weight — over this layer's padded FFT extent, zeros past the accumulator's edge,
// decimated by the scale in `f.disp.x` when the on-screen radius demands it.
//
// Space-specific for the chromatic gather's reason: a texel has to go through this
// space's own decode to *be* light. In pigment the residual is **consumed** here and
// recomputed at the re-entry, so nothing of it rides the transform.
@fragment
fn fs_blur_decode(@builtin(position) frag: vec4<f32>) -> BlurOut {
    let src = blur_src(frag.xy);
    if !src.inside {
        return blur_pad();
    }
    var cb: vec4<f32>;
    var h: f32;
    @if(pigment) var rb: vec4<f32>;
    if src.load {
        cb = textureLoad(back_color, src.p, 0);
        @if(pigment) rb = textureLoad(back_resid, src.p, 0);
        h = textureLoad(back_aux, src.p, 0).x;
    } else {
        cb = textureSampleLevel(back_color, back_samp, src.uv, 0.0);
        @if(pigment) rb = textureSampleLevel(back_resid, back_samp, src.uv, 0.0);
        h = textureSampleLevel(back_aux, back_samp, src.uv, 0.0).x;
    }
    var c: Chan;
    c.lat = cb.rgb / max(cb.a, COVERAGE_EPS);
    @if(pigment) c.res = rb.rgb / max(cb.a, COVERAGE_EPS);
    return BlurOut(
        vec4<f32>(to_light(c) * cb.a, cb.a),
        vec4<f32>(h, 1.0, 0.0, 0.0),
    );
}

// —— the merge (§14.11.7) ————————————————————————————————————————————————————————
//
// One **tile's** worth of a point filter: what a filter layer merged into the paint
// beneath it does to that paint's stored channels.
//
// It is the fragment above with one substitution, and the substitution is the whole
// argument for merging a filter at all. A tile carries a *per-unit opacity* where the
// accumulator carries *coverage* (§6.1), and both carry the channels premultiplied by
// their own — but the filter is defined on the **un-premultiplied** channels, and for a
// layer composited alone those are the same three numbers either way: pass A's slab law
// scales coverage and height and leaves the color where it found it. So the merge needs
// no trip out through `slab.wesl` and back; it un-premultiplies by the tile's own alpha
// (`PAINT_EPS`, the tile-space guard) and tails through the very `resolve` the screen
// takes.
//
// **Why only a point filter may come here.** Every pass that writes tiles must be a
// pure function of canvas position (§6.4) — that is what keeps a tile's apron
// bit-identical to its neighbour's interior without a copy pass, and `tests/seam.rs`
// guards it. This one is: the texel's own channels in, its own channels out. A
// **gather** is not, and no width of apron would make it so at an arbitrary spread —
// which is why the merge rule refuses one (§14.11.7) rather than this file growing a
// branch for it.
@fragment
fn fs_tile(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));
    let c = textureLoad(back_color, p, 0); // (channels · opacity, per-unit opacity)
    let h = textureLoad(back_aux, p, 0).x;
    let opacity = max(c.a, PAINT_EPS);
    var was: Chan;
    was.lat = c.rgb / opacity;
    @if(pigment) was.res = textureLoad(back_resid, p, 0).rgb / opacity;
    return resolve_chan(c, was, from_lab(filtered(to_lab(was))), h);
}
