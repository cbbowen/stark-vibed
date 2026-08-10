// Layer blending for the OKLAB colour space (§18.0.4). The
// composited channels are premultiplied Oklab with the layer's *coverage* in
// alpha (composite.wesl), so the only space-specific work is Oklab ↔ normalized
// XYZ; all of the light algebra and the compositing tail live in `blend_common`.

import package::lib::color::{
    oklab_to_linear_srgb, linear_srgb_to_oklab, linear_to_light, light_to_linear,
};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::blend_common::{
    FsOut, back_color, back_aux, src_color, src_aux, combine, merge,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (Oklab · coverage, coverage)
    let cs = textureLoad(src_color, p, 0);
    let chan_b = cb.rgb / max(cb.a, COVERAGE_EPS);
    let chan_s = cs.rgb / max(cs.a, COVERAGE_EPS);

    let blended = linear_srgb_to_oklab(light_to_linear(combine(
        linear_to_light(oklab_to_linear_srgb(chan_b)),
        linear_to_light(oklab_to_linear_srgb(chan_s)),
    )));

    let hb = textureLoad(back_aux, p, 0).x;
    let hs = textureLoad(src_aux, p, 0).x;
    return merge(cb, cs, blended, hb, hs);
}
