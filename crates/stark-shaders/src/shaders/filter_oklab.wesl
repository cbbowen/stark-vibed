// Filter layers for the OKLAB colour space (§21). The composited channels are
// premultiplied Oklab with the stack's *coverage* in alpha (composite.wesl) — the
// very space `filter_common`'s adjustment is defined in, so there is no
// space-specific conversion at all: the channels go straight in and come straight
// back. (Exposure's one excursion to linear light happens inside `adjust_color`,
// where every space pays it or not identically.)
//
// Shorter than its Mixbox twin by exactly the residual and the pigment round trip,
// which is the point: in a colorimetric space the three channels are the whole
// colour, and they are already Oklab (§6.7).

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

    let out_chan = filtered(chan);

    let h = textureLoad(back_aux, p, 0).x;
    return resolve(cb, chan, out_chan, h);
}
