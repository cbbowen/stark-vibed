// Media pass B for the Mixbox color space (§6.7). The composited color
// is premultiplied Mixbox latent concentrations (c0, c1, c2) with coverage in
// alpha (c3 = 1 − (c0+c1+c2) is derived), and the latent's **residual** rides
// alongside in a channel of its own. We un-premultiply both, evaluate Mixbox's
// pigment polynomial to sRGB, add the residual back, convert to linear, and hand off
// to the shared lighting/composite tail.
//
// **The residual is what makes this reproduce a color at all.** Four trained
// pigments do not span sRGB, so `mixbox_eval_polynomial` alone reaches neither black
// — pure black comes out `#383838`, its concentrations being a near-neutral mixture
// the polynomial can only render as mid-grey — nor the saturated corners, where it is
// off by as much as 0.39. Mixbox carries that gap in `rgb − poly(c)`, and this engine
// dropped it for as long as a tile had room for only three concentrations plus
// coverage. The gap is not recoverable from the concentrations, either: `rgb → c` is
// many-to-one (70 sRGB colors share one quantized triple, spanning 0.38 of the
// cube), so no function of `c` could ever have stood in for the stored value.
//
// `mixbox_eval_polynomial` is generated at build time from the vendored Mixbox
// GLSL (vendor/mixbox, CC BY-NC 4.0).

import package::lib::color::{srgb_to_linear};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::gen::mixbox_poly::{mixbox_eval_polynomial};
import package::media_common::{
    comp_color, comp_aux, m, is_transparent, over_substrate, paint_vis_at, finish,
};

// **Binding 7 of group 0, past where `media_common` stops** (0–6). Declared here
// rather than there so an Oklab document's media layout does not carry a texture it
// has no residual to put in — the same reason `blend_mixbox` owns its own 7 and 8.
@group(0) @binding(7) var comp_resid: texture_2d<f32>;

// The residual half of the paint-over-substrate composite, on the same `vis` the
// concentrations take (§6.7). A separate function rather than a wider
// `over_substrate` because only this space has a residual, and the alpha the other
// one returns is already the answer for both.
fn over_substrate_resid(paint: vec3<f32>, vis: f32) -> vec3<f32> {
    if is_transparent() {
        return paint;
    }
    return m.bg_resid.rgb * (1.0 - vis) + paint * vis;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(frag.xy));
    let dim = vec2<i32>(textureDimensions(comp_aux));

    let cc = textureLoad(comp_color, p, 0);    // (concentrations · coverage, coverage)
    let vis = paint_vis_at(p, dim);            // accumulated coverage, substrate-corrected
    let rr = textureLoad(comp_resid, p, 0);     // (residual · coverage, coverage)
    let conc = cc.rgb / max(cc.a, COVERAGE_EPS); // un-premultiply coverage → concentrations
    // Un-premultiplied by the *color's* coverage, which is the same number the
    // residual target carries in its own alpha — that duplicate exists only because
    // the fixed-function "over" reads each target's own alpha.
    let res = rr.rgb / max(cc.a, COVERAGE_EPS);
    // Both halves of the Mixbox latent mix linearly, so composite paint-over-substrate
    // in latent space — the concentrations and the residual on the same coverage.
    let over = over_substrate(conc, vis);      // substrate, or a transparent cut-out
    let over_res = over_substrate_resid(res, vis);

    // Evaluate Mixbox's pigment polynomial and add the residual back — `poly(c) + r`
    // is Mixbox's `latent_to_rgb`, and the clamp is its clamp. Then linearize for
    // lighting.
    let srgb = clamp(mixbox_eval_polynomial(over.rgb) + over_res, vec3<f32>(0.0), vec3<f32>(1.0));
    // `vis` doubles as the gloss ramp: the paint film is uniformly specular, so how
    // much sheen a fragment shows is just how much of it is paint (media_common).
    return finish(srgb_to_linear(srgb), vis, over.a, frag.xy, dim);
}
