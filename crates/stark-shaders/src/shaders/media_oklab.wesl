// Media pass B for the OKLAB color space (DESIGN.md §6.3): the "old masters"
// payoff. Color is premultiplied Oklab with *opacity* in alpha and thickness in the
// aux (the normalized representation, §6.1); the visible alpha combines them as a
// translucent slab. We un-premultiply, composite the paint over the substrate by
// that alpha, convert Oklab → linear sRGB, and hand off to the shared lighting tail.

import package::color::{oklab_to_linear_srgb};
import package::media_common::{comp_color, comp_aux, fullscreen_clip, over_substrate, paint_vis_at, finish};

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
    let over = over_substrate(lab, vis);       // substrate, or a transparent cut-out

    // `vis` doubles as the gloss ramp: the paint film is uniformly specular, so how
    // much sheen a fragment shows is just how much of it is paint (media_common).
    return finish(oklab_to_linear_srgb(over.rgb), vis, over.a, frag.xy, dim);
}
