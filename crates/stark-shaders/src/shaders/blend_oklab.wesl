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
    FsOut, back_color, back_aux, src_color, src_aux,
    added_light, combine, is_emissive, merge, merge_added,
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
    let hb = textureLoad(back_aux, p, 0).x;
    let hs = textureLoad(src_aux, p, 0).x;

    // **Nothing beneath**: with no backdrop every mode is the source, and `merge`
    // gives it back to the bit — its `mix(…, αb)` at `αb = 0` is the source term
    // verbatim, with no trip out to light and back to round it (`blend_common`).
    // `chan_s` stands in for the blend function's opinion, which that `mix` discards.
    if cb.a <= 0.0 {
        return merge(cb, cs, chan_s, hb, hs);
    }

    let light_b = linear_to_light(oklab_to_linear_srgb(chan_b));
    let light_s = linear_to_light(oklab_to_linear_srgb(chan_s));

    // The emissive modes weigh coverage in **emission**, where the combination is
    // addition, so they compose one sum rather than a nest of interpolations
    // (`added_light`). The whole output colour falls out of that sum, so there is no
    // Porter-Duff triple left to run — only the conversion back to Oklab.
    if is_emissive() {
        let x = added_light(light_b, cb.a, light_s, cs.a);
        return merge_added(cb, cs, linear_srgb_to_oklab(light_to_linear(x)), hb, hs);
    }
    let blended = linear_srgb_to_oklab(light_to_linear(combine(light_b, light_s)));
    return merge(cb, cs, blended, hb, hs);
}
