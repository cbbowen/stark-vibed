// Media pass B for the OKLAB color space (DESIGN.md §6.3): the "old masters"
// payoff. Color is premultiplied Oklab with *opacity* in alpha and thickness in the
// aux (the normalized representation, §6.1); the visible alpha combines them as a
// translucent slab. We un-premultiply, composite the paint over the substrate by
// that alpha, convert Oklab → linear sRGB, and hand off to the shared lighting tail.

import package::color::{oklab_to_linear_srgb};
import package::media_common::{comp_color, comp_aux, fullscreen_clip, background_color, paint_vis_at, finish};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(frag.xy));
    let dim = vec2<i32>(textureDimensions(comp_aux));

    let cc = textureLoad(comp_color, p, 0);    // (Oklab · opacity, opacity)
    let vis = paint_vis_at(p, dim);            // translucent-slab visible alpha
    let lab = cc.rgb / max(cc.a, 1e-4);        // un-premultiply opacity → latent Oklab
    let latent = background_color().rgb * (1.0 - vis) + lab * vis;
    let wet = clamp(textureLoad(comp_aux, p, 0).y, 0.0, 1.0) * vis;

    return finish(oklab_to_linear_srgb(latent), wet, frag.xy, dim);
}
