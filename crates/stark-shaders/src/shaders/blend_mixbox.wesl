// Layer blending for the **Mixbox** pigment color space (§18.0.4,
// §6.7). Same light algebra as every other space; what differs is that a
// channel triple here is a mixture of pigments rather than a color, so the trip out
// to light and back is `lib/mixbox.wesl`'s round trip — the polynomial one way, the
// LUT the other.
//
// That round trip is why the LUT is bound here at all. Combining light is not a
// pigment operation — there is no mixture of Cadmium Red and Ultramarine that *is*
// two lamps added together — so the only way to say what a glow layer did in a
// pigment document is to work out the resulting light and ask which mixture would
// have produced it.
//
// One honest consequence: pigment cannot be brighter than the light falling on it,
// and the LUT is defined on `[0,1]` sRGB, so a `Radiance` layer's overshoot (see
// `BlendMode::Drago`) clips at white here instead of surviving into the media pass's
// highlight roll-off the way it does in an Oklab document. That is not a bug being
// tolerated; it is what the medium says. Paint does not glow.
//
// `Multiply` is the one the medium has no complaint about — a glaze *is* a pigment
// operation — but it still takes the same road out to light and back, and that is
// deliberate rather than a missed shortcut. Multiplying the concentrations directly
// would be cheaper and would mean nothing: `c` is how much of a pigment is present,
// not how much light gets through, so their product is not a quantity. Two glazes
// attenuate the light passing through them, which is a statement about light, so it
// is settled in light and the LUT is asked which mixture ended up that color.

import package::lib::color::{linear_to_light, light_to_linear};
import package::lib::mixbox::{Pigment, mixbox_to_linear, mixbox_from_linear};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::blend_common::{
    b, back_color, back_aux, src_color, src_aux,
    combined_light, is_normal, merge, merge_added, merge_triple,
};

// Mixbox's inverse LUT and the sampler that reads it — bound here because this is one
// of the two passes that has to express a combination of *light* back as a mixture of
// pigment. **Bindings 5 and 6 of group 0**, past where `blend_common` stops (0–4).
@group(0) @binding(5) var pigment_lut: texture_2d<f32>;
@group(0) @binding(6) var pigment_samp: sampler;

// The residual halves of the two colors (§6.7). **Bindings 7 and 8 of
// group 0**, and nothing but this comment holds the two modules apart. Unconditional,
// with no `@if`: this module is reached only by the space that has a residual.
@group(0) @binding(7) var back_resid: texture_2d<f32>;
@group(0) @binding(8) var src_resid: texture_2d<f32>;

// The three targets a pigment document's blend writes. `blend_common::FsOut` is the
// two an Oklab one writes; this is that plus the residual, and `fs_main` fills the
// shared pair from `merge` so there is still one definition of the algebra.
struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @location(2) resid: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (concentrations · coverage, coverage)
    let cs = textureLoad(src_color, p, 0);
    let rb = textureLoad(back_resid, p, 0); // (residual · coverage, coverage)
    let rs = textureLoad(src_resid, p, 0);
    // Both halves un-premultiply by the *color's* coverage — the residual target's
    // own alpha is the same number, duplicated only because the fixed-function "over"
    // that filled these targets reads each one's alpha.
    let chan_b = cb.rgb / max(cb.a, COVERAGE_EPS);
    let chan_s = cs.rgb / max(cs.a, COVERAGE_EPS);
    let res_b = rb.rgb / max(cb.a, COVERAGE_EPS);
    let res_s = rs.rgb / max(cs.a, COVERAGE_EPS);

    let hb = textureLoad(back_aux, p, 0).x;
    let hs = textureLoad(src_aux, p, 0).x;

    var out: FsOut;
    // Three ways out without the trip to light and back, and here more than anywhere
    // that trip is worth skipping: it is a LUT lookup that would re-quantize a latent
    // which is already exact. **Nothing beneath** — every mode is the source, given
    // back to the bit by `merge`'s `mix(..., ab)` at `ab = 0`. **Nothing above** — the
    // backdrop, given back to the bit at `as = 0`. **`Normal`** — it covers rather than
    // combines, so it has no blend function to be affine anywhere and the working space
    // is where it belongs. Both halves of the color take the branch.
    if cb.a <= 0.0 || cs.a <= 0.0 || is_normal() {
        let base = merge(cb, cs, chan_s, hb, hs);
        out.color = base.color;
        out.aux = base.aux;
        out.resid = vec4<f32>(
            merge_triple(rb.rgb, cb.a, rs.rgb * b.opacity, cs.a * b.opacity, res_s),
            base.color.a,
        );
        return out;
    }

    // Everything else is settled in the mode's own space (`combined_light`), and a
    // pigment document then asks the LUT which mixture ended up that color. That the
    // residual falls out of the same answer rather than being merged separately is a
    // simplification the one-expansion law buys: the two halves are one color, and
    // here they are decoded from one.
    //
    // **The residual is what makes the trip back from light exact** — the LUT's
    // concentrations alone cannot reach every color, and a blend landing on black would
    // store a mixture the media pass renders `#383838`.
    let x = combined_light(
        linear_to_light(mixbox_to_linear(Pigment(chan_b, res_b))),
        cb.a,
        linear_to_light(mixbox_to_linear(Pigment(chan_s, res_s))),
        cs.a,
    );
    let out_p = mixbox_from_linear(pigment_lut, pigment_samp, light_to_linear(x));
    let base = merge_added(cb, cs, out_p.conc, hb, hs);
    out.color = base.color;
    out.aux = base.aux;
    out.resid = vec4<f32>(out_p.res * base.color.a, base.color.a);
    return out;

}
