// Layer blending for the OKLAB colour space (MISSING_FEATURES.md §0.4). The
// composited channels are premultiplied Oklab with per-unit *opacity* in alpha
// (DESIGN.md §6.1), so the only space-specific work is Oklab ↔ normalized XYZ; all
// of the light algebra and the compositing tail live in `blend_common`.

import package::color::{oklab_to_linear_srgb, linear_srgb_to_oklab};
import package::blend_common::{
    EPS, FsOut, back_color, src_color, fullscreen_clip, combine, merge,
    linear_to_light, light_to_linear,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (Oklab · opacity, opacity)
    let cs = textureLoad(src_color, p, 0);
    let chan_b = cb.rgb / max(cb.a, EPS);
    let chan_s = cs.rgb / max(cs.a, EPS);

    let blended = linear_srgb_to_oklab(light_to_linear(combine(
        linear_to_light(oklab_to_linear_srgb(chan_b)),
        linear_to_light(oklab_to_linear_srgb(chan_s)),
    )));

    return merge(cb, cs, blended, p);
}
