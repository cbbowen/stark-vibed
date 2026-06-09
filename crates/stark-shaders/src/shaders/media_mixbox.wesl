// Media pass B for the Mixbox color space (DESIGN.md §6.7). The composited color
// is premultiplied Mixbox latent concentrations (c0, c1, c2) with coverage in
// alpha (c3 = 1 − (c0+c1+c2) is derived; the residual is dropped — see DESIGN
// §6.7). We un-premultiply, evaluate Mixbox's pigment polynomial to sRGB, convert
// to linear, and hand off to the shared lighting/composite tail.
//
// `mixbox_eval_polynomial` is generated at build time from the vendored Mixbox
// GLSL (vendor/mixbox, CC BY-NC 4.0).

import package::color::{srgb_to_linear};
import package::mixbox_poly::{mixbox_eval_polynomial};
import package::media_common::{comp_color, comp_aux, fullscreen_clip, background_color, paint_vis_at, finish};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(frag.xy));
    let dim = vec2<i32>(textureDimensions(comp_aux));

    let cc = textureLoad(comp_color, p, 0);    // (concentrations · opacity, opacity)
    let vis = paint_vis_at(p, dim);            // translucent-slab visible alpha
    let conc = cc.rgb / max(cc.a, 1e-4);       // un-premultiply opacity → concentrations
    // Mixbox latent mixes linearly, so composite paint-over-substrate in latent space.
    let latent = background_color().rgb * (1.0 - vis) + conc * vis;
    let wet = clamp(textureLoad(comp_aux, p, 0).y, 0.0, 1.0) * vis;

    // Evaluate Mixbox's pigment polynomial → sRGB, then linearize for lighting.
    let srgb = clamp(mixbox_eval_polynomial(latent), vec3<f32>(0.0), vec3<f32>(1.0));
    return finish(srgb_to_linear(srgb), wet, frag.xy, dim);
}
