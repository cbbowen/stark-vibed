// Media pass B (§6.3): the "old masters" payoff, in whichever working space the
// document is in (§6.7).
//
// Color arrives premultiplied by *coverage* — pass A weighs each layer by its slab
// visible alpha (composite.wesl), so alpha here is the stack's accumulated coverage. We
// un-premultiply, composite the paint over the substrate by that coverage
// (substrate-sunk paint discounted, `paint_vis_at`), convert to linear sRGB, and hand
// off to the shared lighting tail.
//
// **One module, linked twice.** The lighting, the substrate and the display encode are
// `media_common.wesl`'s; a space supplies only the one conversion, and in a pigment
// space the residual that rides beside its channels.

@if(!pigment) import package::lib::color::{oklab_to_linear_srgb};
@if(pigment) import package::lib::mixbox::{Pigment, mixbox_to_linear};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS};
import package::media_common::{
    comp_color, comp_aux, m, is_transparent, over_substrate, paint_vis_at, finish,
};

// **Group 1, whole**: `media_common` owns group 0 entire and this module owns this one
// (§6.10). A space with no residual declares nothing here and binds an empty layout in
// its place, rather than carrying a texture it has nothing to put in.
@if(pigment) @group(1) @binding(0) var comp_resid: texture_2d<f32>;

// The residual half of the paint-over-substrate composite, on the same `vis` the
// channels take (§6.7). A separate function rather than a wider `over_substrate`
// because only a pigment space has one, and the alpha the other returns already answers
// for both.
@if(pigment) fn over_substrate_resid(paint: vec3<f32>, vis: f32) -> vec3<f32> {
    if is_transparent() {
        return paint;
    }
    return m.bg_resid.rgb * (1.0 - vis) + paint * vis;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(frag.xy));
    let dim = vec2<i32>(textureDimensions(comp_aux));

    let cc = textureLoad(comp_color, p, 0);      // (channels · coverage, coverage)
    let vis = paint_vis_at(p, dim);              // accumulated coverage, substrate-corrected
    let chan = cc.rgb / max(cc.a, COVERAGE_EPS); // un-premultiply coverage → the channels
    let over = over_substrate(chan, vis);        // substrate, or a transparent cut-out

    // Both halves of a pigment color mix linearly, so the residual composites over the
    // substrate on the same coverage — un-premultiplied by the *color's*, which is the
    // same number its own target carries, that duplicate existing only because the
    // fixed-function "over" reads each target's own alpha.
    @if(pigment) let rr = textureLoad(comp_resid, p, 0);
    @if(pigment) let over_res = over_substrate_resid(rr.rgb / max(cc.a, COVERAGE_EPS), vis);

    // The one line a space owns: its channels as linear sRGB. In pigment that is
    // Mixbox's `latent_to_rgb`, `poly(c) + r` — and **the residual is what makes it
    // reproduce a color at all**, four trained pigments not spanning sRGB (§6.7).
    @if(!pigment) let lin = oklab_to_linear_srgb(over.rgb);
    @if(pigment) let lin = mixbox_to_linear(Pigment(over.rgb, over_res));

    // `vis` doubles as the gloss ramp: the paint film is uniformly specular, so how much
    // sheen a fragment shows is just how much of it is paint (media_common).
    return finish(lin, vis, over.a, frag.xy, dim);
}
