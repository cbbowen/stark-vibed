// Layer blending for the OKLAB color space (§18.0.4). The
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
    combined_light, is_normal, merge, merge_added,
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

    // Two ways out without the trip to light and back, and both are exact rather
    // than merely cheap. **Nothing beneath**: with no backdrop every mode is the
    // source, and `merge` gives it back to the bit — its `mix(…, αb)` at `αb = 0` is
    // the source term verbatim. **`Normal`**: it covers rather than combines, so there
    // is no blend function to be affine anywhere and the working space is where it
    // belongs. Reached only by a clipped layer or a faded group (§14.4).
    //
    // `chan_s` stands in for the blend function's opinion, which neither branch reads.
    if cb.a <= 0.0 || is_normal() {
        return merge(cb, cs, chan_s, hb, hs);
    }

    // Everything else is settled in the mode's own space, where its blend function is
    // affine and Porter-Duff's coverage weighting therefore commutes with it
    // (`combined_light`). The whole output color falls out of that one expansion, so
    // there is no triple left to run here — only the conversion back to Oklab.
    let x = combined_light(
        linear_to_light(oklab_to_linear_srgb(chan_b)),
        cb.a,
        linear_to_light(oklab_to_linear_srgb(chan_s)),
        cs.a,
    );
    return merge_added(cb, cs, linear_srgb_to_oklab(light_to_linear(x)), hb, hs);

}
