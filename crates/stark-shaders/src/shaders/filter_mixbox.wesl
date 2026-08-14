// Filter layers for the **Mixbox** pigment color space (§21, §6.7). Same
// adjustment as every other space; what differs is that a channel triple here is a
// mixture of pigments rather than a color, so the trip out to Oklab is Mixbox's
// polynomial and the trip back is its LUT — exactly as `blend_mixbox.wesl` reaches
// light, and for the same reason. The chromatic gather (§21.10) pays the same
// bracket per *tap*: each band of the dispersed spectrum is decoded `poly(c) + r`
// to light before it is summed, and the summed light re-enters as a mixture once.
//
// One consequence worth naming, because it is the same one the blend pass carries:
// pigment cannot be brighter than the light falling on it, and the LUT is defined on
// `[0,1]` sRGB, so a positive **exposure** here saturates at white rather than
// pushing past it into the media pass's highlight roll-off the way it does in an
// Oklab document. That is not a limitation being tolerated; a pigment document says
// what a pigment can be, and paint does not glow.

import package::lib::color::{
    srgb_to_linear, linear_to_srgb, linear_srgb_to_oklab, oklab_to_linear_srgb,
};
import package::gen::mixbox_poly::{mixbox_eval_polynomial};
import package::mixbox_lut::{mixbox_srgb_to_concentrations};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::filter_common::{
    f, FILTER_CHROMATIC, back_color, back_aux, back_samp, filtered, resolve, weigh,
    ca_taps, ca_s, ca_uv, ca_weight, ca_lum,
};

// The backdrop's residual half (§6.7). **Binding 7 of group 0** — `filter_common`
// owns 0–3 and `mixbox_lut` 5–6, the same partition `blend_mixbox.wesl` works
// under, and as that file says nothing but this comment holds the three apart.
// Unconditional, with no `@if`: this module is reached only by the space that has a
// residual.
@group(0) @binding(7) var back_resid: texture_2d<f32>;

// The three targets a pigment document's filter writes. `filter_common::FsOut` is
// the two an Oklab one writes; this is that plus the residual, and `fs_main` fills
// the shared pair from `resolve` so there is still one definition of the algebra.
struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @location(2) resid: vec4<f32>,
}

// A latent (concentrations + residual) → Oklab, `filter_common`'s interface.
// Mixbox's `latent_to_rgb` is `poly(c) + r`, and taking the polynomial alone is
// what makes a black backdrop go out as mid-grey (§6.7).
fn latent_to_lab(conc: vec3<f32>, res: vec3<f32>) -> vec3<f32> {
    let srgb = clamp(mixbox_eval_polynomial(conc) + res, vec3<f32>(0.0), vec3<f32>(1.0));
    return linear_srgb_to_oklab(srgb_to_linear(srgb));
}

// A latent (concentrations + residual) → **linear light**, the space the chromatic
// integral sums in (§21.10): the same `poly(c) + r` as above, stopping at linear
// sRGB because a sum over wavelengths is a sum of light, not of lightness.
fn latent_to_linear(conc: vec3<f32>, res: vec3<f32>) -> vec3<f32> {
    let srgb = clamp(mixbox_eval_polynomial(conc) + res, vec3<f32>(0.0), vec3<f32>(1.0));
    return srgb_to_linear(srgb);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (concentrations · coverage, coverage)
    let rb = textureLoad(back_resid, p, 0); // (residual · coverage, coverage)
    let h = textureLoad(back_aux, p, 0).x;

    if f.kind == FILTER_CHROMATIC {
        // The spectral gather (§21.10), as `filter_oklab.wesl` writes it — the one
        // difference is the decode: a tap is a pigment mixture, so reaching its
        // light costs the polynomial plus the residual, per tap.
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
            let rs = textureSampleLevel(back_resid, back_samp, uv, 0.0);
            let hs = textureSampleLevel(back_aux, back_samp, uv, 0.0).x;
            let w = ca_weight(s);
            let y = ca_lum(w);
            let conc = cs.rgb / max(cs.a, COVERAGE_EPS);
            let res = rs.rgb / max(cs.a, COVERAGE_EPS);
            lin_acc += w * latent_to_linear(conc, res) * cs.a;
            w_acc += w;
            cov_acc += y * cs.a;
            h_acc += y * hs;
            y_acc += y;
        }
        let a = cov_acc / y_acc;
        // Out of light and back into pigment, once, for the summed spectrum: the
        // LUT is indexed by sRGB on `[0,1]`, so the clamp precedes it.
        let srgb = clamp(
            linear_to_srgb(max(lin_acc / w_acc / max(a, COVERAGE_EPS), vec3<f32>(0.0))),
            vec3<f32>(0.0), vec3<f32>(1.0),
        );
        let out_conc = mixbox_srgb_to_concentrations(srgb);
        // The rest of `mixbox_rgb_to_latent`: what the polynomial cannot say about
        // the color the gather arrived at (§6.7).
        let out_res = srgb - mixbox_eval_polynomial(out_conc);
        // The strength mix on the raw premultiplied lanes — both halves of the
        // color, the coverage and the height together, so strength 0 is the
        // backdrop bit for bit (§21.4).
        var out: FsOut;
        out.color = mix(cb, vec4<f32>(out_conc * a, a), f.strength);
        out.resid = mix(rb, vec4<f32>(out_res * a, a), f.strength);
        out.aux = vec4<f32>(mix(h, h_acc / y_acc, f.strength), 0.0, 0.0, 0.0);
        return out;
    }

    // Both halves un-premultiply by the *color's* coverage — the residual target's
    // own alpha is the same number, duplicated only because the fixed-function
    // "over" that filled these targets reads each one's alpha.
    let chan = cb.rgb / max(cb.a, COVERAGE_EPS);
    let res = rb.rgb / max(cb.a, COVERAGE_EPS);

    let lab = filtered(latent_to_lab(chan, res));
    // `max(…, 0)` before the sRGB encode, then a clamp: the adjustment can step
    // outside the gamut, and the LUT is indexed by sRGB.
    let srgb = clamp(
        linear_to_srgb(max(oklab_to_linear_srgb(lab), vec3<f32>(0.0))),
        vec3<f32>(0.0), vec3<f32>(1.0),
    );
    let out_chan = mixbox_srgb_to_concentrations(srgb);
    // The rest of `mixbox_rgb_to_latent`: what the polynomial cannot say about the
    // color the filter arrived at. Without it a filter that landed on black would
    // store a mixture the media pass renders `#383838`, and every filtered texel
    // would be pulled toward the polynomial's reachable set.
    let out_res = srgb - mixbox_eval_polynomial(out_chan);

    let base = resolve(cb, chan, out_chan, h);

    var out: FsOut;
    out.color = base.color;
    out.aux = base.aux;
    // The residual takes the identical strength mix on the identical coverage, and
    // rides the alpha `resolve` already decided (§6.7).
    out.resid = vec4<f32>(weigh(res, out_res) * cb.a, cb.a);
    return out;
}
