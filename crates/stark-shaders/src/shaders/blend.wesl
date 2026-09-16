// Layer blending, in whichever working space the document is in (§18.0.4, §6.7). One
// fullscreen draw merging an isolated layer's composited channels into the accumulator
// through a non-`Normal` blend mode.
//
// **One module, linked twice.** The light algebra, the Porter-Duff tail and the
// bindings are `blend_common.wesl`'s and are the same in every space; what a space
// supplies is the pair below — its channels out to normalized XYZ light, and light back
// into its channels — plus, where it has one, the residual that rides beside them.
// `@if(pigment)` is the whole of the difference, and the Oklab build never names the
// vendored polynomial at all: `wesl` evaluates the gate before it resolves the import,
// so a build without Mixbox does not look it up.
//
// A pigment space pays a round trip here that a colorimetric one does not, and it is
// not an implementation cost. Combining light is not a pigment operation — there is no
// mixture of Cadmium Red and Ultramarine that *is* two lamps added together — so the
// only way to say what a glow layer did in a pigment document is to work out the
// resulting light and ask which mixture would have produced it.
//
// Two consequences the medium owns rather than the code. Pigment cannot be brighter
// than the light falling on it, and the LUT is defined on `[0,1]` sRGB, so a `Radiance`
// layer's overshoot (see `BlendMode::Drago`) clips at white there instead of surviving
// into the media pass's highlight roll-off. And `Multiply` — a glaze, which *is* a
// pigment operation — still takes the road out to light, because `c` is how much of a
// pigment is present rather than how much light gets through, so multiplying
// concentrations would not be a quantity. Two glazes attenuate light, which is a
// statement about light.

import package::lib::color::{linear_to_light, light_to_linear};
@if(!pigment) import package::lib::color::{oklab_to_linear_srgb, linear_srgb_to_oklab};
@if(pigment) import package::lib::mixbox::{Pigment, mixbox_to_linear, mixbox_from_linear};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::blend_common::{
    b, back_color, back_aux, src_color, src_aux,
    combined_light, is_normal, merge, merge_added, merge_triple,
};

// **Group 1, whole**: `blend_common` owns group 0 entire and this module owns this one
// entire, so neither has to know where the other stops (§6.10). A colorimetric blend
// declares nothing here and binds an empty layout in its place.
@if(pigment) @group(1) @binding(0) var pigment_lut: texture_2d<f32>;
@if(pigment) @group(1) @binding(1) var pigment_samp: sampler;
@if(pigment) @group(1) @binding(2) var back_resid: texture_2d<f32>;
@if(pigment) @group(1) @binding(3) var src_resid: texture_2d<f32>;

// The targets this space's blend writes: `blend_common::FsOut`'s two, plus the residual
// where the space has one. `fs_main` fills the shared pair from `merge` so there is
// still one definition of the algebra.
struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @if(pigment) @location(2) resid: vec4<f32>,
}

// A color as this space stores it: its three channels, and — in a pigment space — the
// rest of the same color (§6.7).
struct Chan {
    lat: vec3<f32>,
    @if(pigment) res: vec3<f32>,
}

// —— what a space supplies ———————————————————————————————————————————————————————

@if(!pigment) fn to_light(c: Chan) -> vec3<f32> {
    return linear_to_light(oklab_to_linear_srgb(c.lat));
}

@if(pigment) fn to_light(c: Chan) -> vec3<f32> {
    return linear_to_light(mixbox_to_linear(Pigment(c.lat, c.res)));
}

@if(!pigment) fn from_light(x: vec3<f32>) -> Chan {
    return Chan(linear_srgb_to_oklab(light_to_linear(x)));
}

// **The residual is what makes the trip back exact** — the LUT's concentrations alone
// cannot reach every color, and a blend landing on black would store a mixture the
// media pass renders `#383838`. That it falls out of the same decode rather than being
// merged separately is what the one-expansion law buys (§6.7).
@if(pigment) fn from_light(x: vec3<f32>) -> Chan {
    let p = mixbox_from_linear(pigment_lut, pigment_samp, light_to_linear(x));
    return Chan(p.conc, p.res);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (channels · coverage, coverage)
    let cs = textureLoad(src_color, p, 0);
    @if(pigment) let rb = textureLoad(back_resid, p, 0); // (residual · coverage, coverage)
    @if(pigment) let rs = textureLoad(src_resid, p, 0);
    // Both halves un-premultiply by the *color's* coverage — the residual target's own
    // alpha is the same number, duplicated only because the fixed-function "over" that
    // filled these targets reads each one's alpha.
    var chan_b: Chan;
    var chan_s: Chan;
    chan_b.lat = cb.rgb / max(cb.a, COVERAGE_EPS);
    chan_s.lat = cs.rgb / max(cs.a, COVERAGE_EPS);
    @if(pigment) chan_b.res = rb.rgb / max(cb.a, COVERAGE_EPS);
    @if(pigment) chan_s.res = rs.rgb / max(cs.a, COVERAGE_EPS);

    let hb = textureLoad(back_aux, p, 0).x;
    let hs = textureLoad(src_aux, p, 0).x;

    var out: FsOut;
    // Three ways out without the trip to light and back, and all are exact rather than
    // merely cheap — in a pigment document it is a LUT lookup that would re-quantize a
    // latent which is already exact. **Nothing beneath** — every mode is the source,
    // given back to the bit by `merge`'s `mix(…, αb)` at `αb = 0`. **Nothing above** —
    // the backdrop, given back to the bit at `αs = 0`, where the trip would clamp an
    // emission the backdrop may exceed. **`Normal`** — it covers rather than combines,
    // so it has no blend function to be affine anywhere and the working space is where
    // it belongs; reached only by a clipped layer or a faded group (§14.4).
    //
    // `chan_s.lat` stands in for the blend function's opinion, which no branch reads.
    if cb.a <= 0.0 || cs.a <= 0.0 || is_normal() {
        let base = merge(cb, cs, chan_s.lat, hb, hs);
        out.color = base.color;
        out.aux = base.aux;
        @if(pigment) out.resid = vec4<f32>(
            merge_triple(rb.rgb, cb.a, rs.rgb * b.opacity, cs.a * b.opacity, chan_s.res),
            base.color.a,
        );
        return out;
    }

    // Everything else is settled in the mode's own space, where its blend function is
    // affine and Porter-Duff's coverage weighting therefore commutes with it
    // (`combined_light`). The whole output color falls out of that one expansion, so
    // there is no triple left to run here — only the conversion back.
    let now = from_light(combined_light(to_light(chan_b), cb.a, to_light(chan_s), cs.a));
    let base = merge_added(cb, cs, now.lat, hb, hs);
    out.color = base.color;
    out.aux = base.aux;
    @if(pigment) out.resid = vec4<f32>(now.res * base.color.a, base.color.a);
    return out;
}
