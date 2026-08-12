// Layer blending for the **Mixbox** pigment colour space (§18.0.4,
// §6.7). Same light algebra as every other space; what differs is that a
// channel triple here is a mixture of pigments rather than a colour, so the trip out
// to light is Mixbox's polynomial and the trip back is its LUT.
//
// That round trip is the reason `mixbox_lut.wesl` exists. Combining light is not a
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
// is settled in light and the LUT is asked which mixture ended up that colour.

import package::lib::color::{
    srgb_to_linear, linear_to_srgb, linear_to_light, light_to_linear,
};
import package::gen::mixbox_poly::{mixbox_eval_polynomial};
import package::mixbox_lut::{mixbox_srgb_to_concentrations};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::blend_common::{
    b, back_color, back_aux, src_color, src_aux,
    added_light, combine, is_emissive, merge, merge_added, merge_triple,
};

// The residual halves of the two colours (§6.7). **Bindings 7 and 8 of
// group 0** — `blend_common` owns 0–4 and `mixbox_lut` 5–6, and as that file says,
// nothing but this comment holds the three apart. Unconditional, with no `@if`: this
// module is reached only by the space that has a residual.
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

// A latent (concentrations + residual) → normalized XYZ. Mixbox's `latent_to_rgb` is
// `poly(c) + r`, and taking the polynomial alone here is what made a black layer go
// out to light as mid-grey (§6.7).
fn latent_to_light(conc: vec3<f32>, res: vec3<f32>) -> vec3<f32> {
    let srgb = clamp(mixbox_eval_polynomial(conc) + res, vec3<f32>(0.0), vec3<f32>(1.0));
    return linear_to_light(srgb_to_linear(srgb));
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
    // Both halves un-premultiply by the *colour's* coverage — the residual target's
    // own alpha is the same number, duplicated only because the fixed-function "over"
    // that filled these targets reads each one's alpha.
    let chan_b = cb.rgb / max(cb.a, COVERAGE_EPS);
    let chan_s = cs.rgb / max(cs.a, COVERAGE_EPS);
    let res_b = rb.rgb / max(cb.a, COVERAGE_EPS);
    let res_s = rs.rgb / max(cs.a, COVERAGE_EPS);

    let hb = textureLoad(back_aux, p, 0).x;
    let hs = textureLoad(src_aux, p, 0).x;

    var out: FsOut;
    // **Nothing beneath**: every mode is the source, given back to the bit by the
    // `mix(…, αb)` at `αb = 0` — and, here more than anywhere, without the trip out
    // to light, which is a LUT lookup that would re-quantize a latent that is already
    // exact (`blend_common::merge`). Both halves of the colour take the same branch.
    if cb.a <= 0.0 {
        let base = merge(cb, cs, chan_s, hb, hs);
        out.color = base.color;
        out.aux = base.aux;
        out.resid = vec4<f32>(
            merge_triple(rb.rgb, cb.a, rs.rgb * b.opacity, cs.a * b.opacity, res_s),
            base.color.a,
        );
        return out;
    }

    // `max(…, 0)` before the sRGB encode: the combination is componentwise in XYZ,
    // which can step just outside the gamut on very saturated pairs, and the LUT is
    // indexed by sRGB. `mixbox_srgb_to_concentrations` clamps the top end itself.
    // The rest of `mixbox_rgb_to_latent` is the residual: what the polynomial could
    // not say about the colour the light works out to. **The trip back from light is
    // exact** — it was the LUT's concentrations alone, so a blend that landed on black
    // stored a mixture the media pass would render `#383838`, and every blend result
    // was pulled toward the polynomial's reachable set.
    // The emissive modes settle the whole colour in **emission**, where combining is
    // addition (`added_light`), so a pigment document asks the LUT which mixture ended
    // up that colour and is finished — there is no Porter-Duff triple left to run on
    // either half. That the residual falls out of the same answer rather than being
    // merged separately is a simplification the additive law buys: the two halves are
    // one colour, and here they are decoded from one.
    if is_emissive() {
        let x = added_light(
            latent_to_light(chan_b, res_b), cb.a,
            latent_to_light(chan_s, res_s), cs.a,
        );
        let srgb = clamp(
            linear_to_srgb(max(light_to_linear(x), vec3<f32>(0.0))),
            vec3<f32>(0.0), vec3<f32>(1.0),
        );
        let conc = mixbox_srgb_to_concentrations(srgb);
        let base = merge_added(cb, cs, conc, hb, hs);
        out.color = base.color;
        out.aux = base.aux;
        out.resid = vec4<f32>((srgb - mixbox_eval_polynomial(conc)) * base.color.a, base.color.a);
        return out;
    }

    let light = combine(latent_to_light(chan_b, res_b), latent_to_light(chan_s, res_s));
    let srgb = clamp(
        linear_to_srgb(max(light_to_linear(light), vec3<f32>(0.0))),
        vec3<f32>(0.0), vec3<f32>(1.0),
    );
    let blended = mixbox_srgb_to_concentrations(srgb);
    let blended_res = srgb - mixbox_eval_polynomial(blended);
    let base = merge(cb, cs, blended, hb, hs);
    out.color = base.color;
    out.aux = base.aux;
    // The residual takes the identical three-term merge on the identical weights, and
    // rides the alpha `merge` already decided (§6.7).
    out.resid = vec4<f32>(
        merge_triple(rb.rgb, cb.a, rs.rgb * b.opacity, cs.a * b.opacity, blended_res),
        base.color.a,
    );
    return out;
}
