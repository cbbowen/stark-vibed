// Filter layers for the **Mixbox** pigment colour space (§21, §6.7). Same
// adjustment as every other space; what differs is that a channel triple here is a
// mixture of pigments rather than a colour, so the trip out to Oklab is Mixbox's
// polynomial and the trip back is its LUT — exactly as `blend_mixbox.wesl` reaches
// light, and for the same reason.
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
import package::filter_common::{back_color, back_aux, filtered, resolve, weigh};

// The backdrop's residual half (§6.7). **Binding 7 of group 0** — `filter_common`
// owns 0–2 and `mixbox_lut` 5–6, the same partition `blend_mixbox.wesl` works
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

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (concentrations · coverage, coverage)
    let rb = textureLoad(back_resid, p, 0); // (residual · coverage, coverage)
    // Both halves un-premultiply by the *colour's* coverage — the residual target's
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
    // colour the filter arrived at. Without it a filter that landed on black would
    // store a mixture the media pass renders `#383838`, and every filtered texel
    // would be pulled toward the polynomial's reachable set.
    let out_res = srgb - mixbox_eval_polynomial(out_chan);

    let h = textureLoad(back_aux, p, 0).x;
    let base = resolve(cb, chan, out_chan, h);

    var out: FsOut;
    out.color = base.color;
    out.aux = base.aux;
    // The residual takes the identical strength mix on the identical coverage, and
    // rides the alpha `resolve` already decided (§6.7).
    out.resid = vec4<f32>(weigh(res, out_res) * cb.a, cb.a);
    return out;
}
