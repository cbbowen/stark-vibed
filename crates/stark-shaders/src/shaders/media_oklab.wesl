// Media pass B for the OKLAB color space (DESIGN.md §6.3): the "old masters"
// payoff. The composited color is premultiplied Oklab with coverage in alpha;
// we un-premultiply, convert Oklab → linear sRGB, and hand off to the shared
// lighting/composite tail (`media_common.wesl`).

import package::color::{oklab_to_linear_srgb};
import package::media_common::{comp_color, comp_aux, fullscreen_clip, finish};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(frag.xy));
    let dim = vec2<i32>(textureDimensions(comp_aux));

    let color = textureLoad(comp_color, p, 0);
    let coverage = color.a;
    let wet = textureLoad(comp_aux, p, 0).y;

    // Un-premultiply Oklab, then convert to linear sRGB for lighting.
    let base = oklab_to_linear_srgb(color.rgb / max(coverage, 1e-4));
    return finish(base, coverage, wet, p, dim);
}
