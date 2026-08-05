// Media pass B for the Mixbox color space (§6.7). The composited color
// is premultiplied Mixbox latent concentrations (c0, c1, c2) with coverage in
// alpha (c3 = 1 − (c0+c1+c2) is derived; the residual is dropped — see
// §6.7). We un-premultiply, evaluate Mixbox's pigment polynomial to sRGB, convert
// to linear, and hand off to the shared lighting/composite tail.
//
// `mixbox_eval_polynomial` is generated at build time from the vendored Mixbox
// GLSL (vendor/mixbox, CC BY-NC 4.0).

import package::lib::color::{srgb_to_linear};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::gen::mixbox_poly::{mixbox_eval_polynomial};
import package::media_common::{comp_color, comp_aux, over_substrate, paint_vis_at, finish};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(frag.xy));
    let dim = vec2<i32>(textureDimensions(comp_aux));

    let cc = textureLoad(comp_color, p, 0);    // (concentrations · coverage, coverage)
    let vis = paint_vis_at(p, dim);            // accumulated coverage, weave-corrected
    let conc = cc.rgb / max(cc.a, COVERAGE_EPS); // un-premultiply coverage → concentrations
    // Mixbox latent mixes linearly, so composite paint-over-substrate in latent space.
    let over = over_substrate(conc, vis);      // substrate, or a transparent cut-out

    // Evaluate Mixbox's pigment polynomial → sRGB, then linearize for lighting.
    let srgb = clamp(mixbox_eval_polynomial(over.rgb), vec3<f32>(0.0), vec3<f32>(1.0));
    // `vis` doubles as the gloss ramp: the paint film is uniformly specular, so how
    // much sheen a fragment shows is just how much of it is paint (media_common).
    return finish(srgb_to_linear(srgb), vis, over.a, frag.xy, dim);
}
