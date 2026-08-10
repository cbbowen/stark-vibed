// Filter layers for the OKLAB colour space (§21). The composited channels are
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
// the whole colour, and they are already Oklab (§6.7).

import package::lib::color::{oklab_to_linear_srgb, linear_srgb_to_oklab};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::filter_common::{
    FsOut, f, FILTER_CHROMATIC, back_color, back_aux, back_samp, filtered, resolve,
    ca_taps, ca_s, ca_uv, ca_weight, ca_lum,
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
            // `max(…, 0)`: the accumulator can hold an out-of-gamut mixture whose
            // linear form dips below zero, and light does not go negative.
            lin_acc += w * max(oklab_to_linear_srgb(lab_s), vec3<f32>(0.0)) * cs.a;
            w_acc += w;
            cov_acc += y * cs.a;
            h_acc += y * hs;
            y_acc += y;
        }
        let a = cov_acc / y_acc;
        let lab = linear_srgb_to_oklab(lin_acc / w_acc / max(a, COVERAGE_EPS));
        // The strength mix runs on the raw premultiplied lanes — colour, coverage
        // and height together, because this filter moved all three — so strength 0
        // is `cb` itself, bit for bit, the same exactness `weigh` gives the point
        // filters (§21.4).
        var out: FsOut;
        out.color = mix(cb, vec4<f32>(lab * a, a), f.strength);
        out.aux = vec4<f32>(mix(h, h_acc / y_acc, f.strength), 0.0, 0.0, 0.0);
        return out;
    }

    let chan = cb.rgb / max(cb.a, COVERAGE_EPS);
    let out_chan = filtered(chan);
    return resolve(cb, chan, out_chan, h);
}
