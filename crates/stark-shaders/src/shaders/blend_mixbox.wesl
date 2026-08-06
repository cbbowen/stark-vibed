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

import package::lib::color::{srgb_to_linear, linear_to_srgb};
import package::gen::mixbox_poly::{mixbox_eval_polynomial};
import package::mixbox_lut::{mixbox_srgb_to_concentrations};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::blend_common::{
    FsOut, back_color, back_aux, src_color, src_aux, combine, merge,
    linear_to_light, light_to_linear,
};

// Concentrations → normalized XYZ. Mixbox's polynomial answers in sRGB.
fn conc_to_light(conc: vec3<f32>) -> vec3<f32> {
    let srgb = clamp(mixbox_eval_polynomial(conc), vec3<f32>(0.0), vec3<f32>(1.0));
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
    let chan_b = cb.rgb / max(cb.a, COVERAGE_EPS);
    let chan_s = cs.rgb / max(cs.a, COVERAGE_EPS);

    let light = combine(conc_to_light(chan_b), conc_to_light(chan_s));
    // `max(…, 0)` before the sRGB encode: the combination is componentwise in XYZ,
    // which can step just outside the gamut on very saturated pairs, and the LUT is
    // indexed by sRGB. `mixbox_srgb_to_concentrations` clamps the top end itself.
    let blended = mixbox_srgb_to_concentrations(
        linear_to_srgb(max(light_to_linear(light), vec3<f32>(0.0)))
    );

    let hb = textureLoad(back_aux, p, 0).x;
    let hs = textureLoad(src_aux, p, 0).x;
    return merge(cb, cs, blended, hb, hs);
}
