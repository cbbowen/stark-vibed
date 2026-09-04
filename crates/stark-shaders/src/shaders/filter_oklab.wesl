// Filter layers for the OKLAB color space (§21). The composited channels are
// premultiplied Oklab with the stack's *coverage* in alpha (composite.wesl) — the
// very space `filter_common`'s adjustment is defined in, so the point filters need
// no space-specific conversion at all: the channels go straight in and come
// straight back. (Exposure's one excursion to linear light happens inside
// `adjust_color`, where every space pays it or not identically.)
//
// The chromatic gather (§21.10) is the one thing with a space-specific body here:
// its integral runs in linear light, so each tap pays this space's Oklab→linear
// trip and the result pays the trip back — the same bracket, per tap instead of
// per texel. Shorter than its Mixbox twin by exactly the residual and the pigment
// round trip, which is the point: in a colorimetric space the three channels are
// the whole color, and they are already Oklab (§6.7).

import package::lib::color::{oklab_to_linear_srgb, linear_srgb_to_oklab, linear_to_light, floor_light};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS, PAINT_EPS};
import package::filter_common::{
    FsOut, f, FILTER_CHROMATIC, FILTER_FOCAL_BLUR, back_color, back_aux, back_samp,
    filtered, resolve, is_clipped, ca_taps, ca_s, ca_uv, ca_weight, ca_lum,
    BlurOut, blur_pad, blur_read, blur_src,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (Oklab · coverage, coverage)
    let h = textureLoad(back_aux, p, 0).x;

    if f.kind == FILTER_CHROMATIC {
        // The spectral gather (§21.10): each tap is one band of the dispersed
        // spectrum — its light in linear sRGB premultiplied by the coverage it
        // arrived with, plus that coverage and height on the scalar weight —
        // normalized per channel at the end so a flat region comes back exact.
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
            let hs = textureSampleLevel(back_aux, back_samp, uv, 0.0).x;
            let w = ca_weight(s);
            let y = ca_lum(w);
            let lab_s = cs.rgb / max(cs.a, COVERAGE_EPS);
            // Floored in light, not per channel: light does not go negative, but a
            // wide-gamut color's sRGB channel does (§6.5).
            lin_acc += w * floor_light(oklab_to_linear_srgb(lab_s)) * cs.a;
            w_acc += w;
            cov_acc += y * cs.a;
            h_acc += y * hs;
            y_acc += y;
        }
        let a = cov_acc / y_acc;
        let lab = linear_srgb_to_oklab(lin_acc / w_acc / max(a, COVERAGE_EPS));
        // **Clipped**: the fringe's color lands, its displaced coverage does not
        // (§21.4). The gather's own `a` is spent on un-premultiplying the light it
        // summed and then dropped, so what is left is a color for the coverage that
        // was already here — which is exactly the point filters' tail, taken
        // verbatim rather than restated.
        if is_clipped() {
            return resolve(cb, cb.rgb / max(cb.a, COVERAGE_EPS), lab, h);
        }
        // The strength mix runs on the raw premultiplied lanes — color, coverage
        // and height together, because this filter moved all three — so strength 0
        // is `cb` itself, bit for bit, the same exactness `weigh` gives the point
        // filters (§21.4).
        var out: FsOut;
        out.color = mix(cb, vec4<f32>(lab * a, a), f.strength);
        out.aux = vec4<f32>(mix(h, h_acc / y_acc, f.strength), 0.0, 0.0, 0.0);
        return out;
    }

    if f.kind == FILTER_FOCAL_BLUR {
        // The convolution already ran (§21.12, `blur.wesl`): one read per lane —
        // exact, or interpolated up from a decimated transform — and the trip
        // back from light into this space's channels. Like the gather, this
        // filter moved color, coverage and height together, so the unclipped
        // strength mix runs on the raw premultiplied lanes and the clipped one
        // tails through the point filters' own ending.
        let b = blur_read(frag.xy);
        let lab = linear_srgb_to_oklab(b.lin);
        if is_clipped() {
            return resolve(cb, cb.rgb / max(cb.a, COVERAGE_EPS), lab, h);
        }
        var out: FsOut;
        out.color = mix(cb, vec4<f32>(lab * b.cov, b.cov), f.strength);
        out.aux = vec4<f32>(mix(h, b.h, f.strength), 0.0, 0.0, 0.0);
        return out;
    }

    let chan = cb.rgb / max(cb.a, COVERAGE_EPS);
    let out_chan = filtered(chan);
    return resolve(cb, chan, out_chan, h);
}

// The focal blur's decode (§21.12): the accumulator laid into `blur.wesl`'s
// spatial-domain planes — premultiplied XYZ light, coverage, height, and the
// border weight — over this layer's padded FFT extent, zeros past the
// accumulator's edge, decimated by the scale in `f.disp.x` when the on-screen
// radius demands it. Space-specific for the chromatic gather's reason: a texel
// has to go through this space's own decode to *be* light, and here that is one
// Oklab trip.
@fragment
fn fs_blur_decode(@builtin(position) frag: vec4<f32>) -> BlurOut {
    let src = blur_src(frag.xy);
    if !src.inside {
        return blur_pad();
    }
    var cb: vec4<f32>;
    var h: f32;
    if src.load {
        cb = textureLoad(back_color, src.p, 0);
        h = textureLoad(back_aux, src.p, 0).x;
    } else {
        cb = textureSampleLevel(back_color, back_samp, src.uv, 0.0);
        h = textureSampleLevel(back_aux, back_samp, src.uv, 0.0).x;
    }
    let lab = cb.rgb / max(cb.a, COVERAGE_EPS);
    // Floored as light, for the gather's reason (§6.5).
    let light = max(linear_to_light(oklab_to_linear_srgb(lab)), vec3<f32>(0.0));
    return BlurOut(
        vec4<f32>(light * cb.a, cb.a),
        vec4<f32>(h, 1.0, 0.0, 0.0),
    );
}

// —— the merge (§14.11.7) ————————————————————————————————————————————————————
//
// One **tile's** worth of a point filter: what a filter layer merged into the paint
// beneath it does to that paint's stored channels.
//
// It is the fragment above with one substitution, and the substitution is the whole
// argument for merging a filter at all. A tile carries a *per-unit opacity* where the
// accumulator carries *coverage* (§6.1), and both carry the channels premultiplied by
// their own — but the filter is defined on the **un-premultiplied** channels, and for
// a layer composited alone those are the same three numbers either way: pass A's slab
// law scales coverage and height and leaves the color where it found it. So the merge
// needs no trip out through `slab.wesl` and back; it un-premultiplies by the tile's own
// alpha (`PAINT_EPS`, the tile-space guard) and tails through the very `resolve` the
// screen takes.
//
// **Why only a point filter may come here.** Every pass that writes tiles must be a
// pure function of canvas position (§6.4) — that is what keeps a tile's apron
// bit-identical to its neighbour's interior without a copy pass, and `tests/seam.rs`
// guards it. This one is: the texel's own channels in, its own channels out, so the
// apron texel and the neighbour's interior are the same input and come out the same
// number. A **gather** is not, and no width of apron would make it so at an arbitrary
// spread — which is why the merge rule refuses one (§14.11.7) rather than this file
// growing a branch for it.
@fragment
fn fs_tile(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));
    let c = textureLoad(back_color, p, 0); // (Oklab · opacity, per-unit opacity)
    let h = textureLoad(back_aux, p, 0).x;
    let chan = c.rgb / max(c.a, PAINT_EPS);
    return resolve(c, chan, filtered(chan), h);
}
