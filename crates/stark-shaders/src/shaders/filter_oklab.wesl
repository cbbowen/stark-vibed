// Filter layers for the OKLAB colour space (§21). The composited channels are
// premultiplied Oklab with the stack's *coverage* in alpha (composite.wesl), so the
// only space-specific work is Oklab ↔ normalized XYZ; the adjustment itself and the
// strength mix live in `filter_common`.
//
// Shorter than its Mixbox twin by exactly the residual, which is the point: in a
// colorimetric space the three channels are the whole colour, so there is no second
// half to carry through the round trip (§6.7).

import package::lib::color::{oklab_to_linear_srgb, linear_srgb_to_oklab, linear_to_light, light_to_linear};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::filter_common::{FsOut, back_color, back_aux, filtered, resolve};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (Oklab · coverage, coverage)
    let chan = cb.rgb / max(cb.a, COVERAGE_EPS);

    let out_chan = linear_srgb_to_oklab(light_to_linear(filtered(
        linear_to_light(oklab_to_linear_srgb(chan)),
    )));

    let h = textureLoad(back_aux, p, 0).x;
    return resolve(cb, chan, out_chan, h);
}
