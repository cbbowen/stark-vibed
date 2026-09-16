// Filter layers for the **Mixbox** pigment color space (§21, §6.7). Same
// adjustment as every other space; what differs is that a channel triple here is a
// mixture of pigments rather than a color, so the trip out to Oklab and back is
// `lib/mixbox.wesl`'s round trip — exactly as `blend_mixbox.wesl` reaches light, and
// for the same reason. The chromatic gather (§21.10) pays the same bracket per *tap*:
// each band of the dispersed spectrum is decoded to light before it is summed, and the
// summed light re-enters as a mixture once.
//
// One consequence worth naming, because it is the same one the blend pass carries:
// pigment cannot be brighter than the light falling on it, and the LUT is defined on
// `[0,1]` sRGB, so a positive **exposure** here saturates at white rather than
// pushing past it into the media pass's highlight roll-off the way it does in an
// Oklab document. That is not a limitation being tolerated; a pigment document says
// what a pigment can be, and paint does not glow.

import package::lib::color::{
    linear_srgb_to_oklab, oklab_to_linear_srgb, linear_to_light,
};
import package::lib::mixbox::{Pigment, mixbox_to_linear, mixbox_from_linear};
import package::lib::quad::{fullscreen_clip};
import package::lib::paint_common::{COVERAGE_EPS, PAINT_EPS};
import package::filter_common::{
    f, FILTER_CHROMATIC, FILTER_FOCAL_BLUR, back_color, back_aux, back_samp,
    filtered, resolve, weigh, is_clipped, ca_taps, ca_s, ca_uv, ca_weight, ca_lum,
    BlurOut, blur_pad, blur_read, blur_src,
};

// Mixbox's inverse LUT and the sampler that reads it, on the same terms
// `blend_mixbox.wesl` binds them: an adjustment is a statement about light, so a
// pigment document has to ask which mixture the adjusted light would have been.
// **Bindings 5 and 6 of group 0**, past where `filter_common` stops (0–4, and 8).
@group(0) @binding(5) var pigment_lut: texture_2d<f32>;
@group(0) @binding(6) var pigment_samp: sampler;

// The backdrop's residual half (§6.7). **Binding 7 of group 0** — and nothing but
// this comment holds the two modules apart. Unconditional, with no `@if`: this module
// is reached only by the space that has a residual.
@group(0) @binding(7) var back_resid: texture_2d<f32>;

// The three targets a pigment document's filter writes. `filter_common::FsOut` is
// the two an Oklab one writes; this is that plus the residual, and `fs_main` fills
// the shared pair from `resolve` so there is still one definition of the algebra.
struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @location(2) resid: vec4<f32>,
}

// A pigment mixture → Oklab, `filter_common`'s interface.
fn latent_to_lab(p: Pigment) -> vec3<f32> {
    return linear_srgb_to_oklab(mixbox_to_linear(p));
}

// [`mixbox_from_linear`] for a color that arrives in the space the point filters work
// in — `latent_to_lab`'s inverse, and the leg the merge takes too.
fn lab_to_latent(lab: vec3<f32>) -> Pigment {
    return mixbox_from_linear(pigment_lut, pigment_samp, oklab_to_linear_srgb(lab));
}

// `filter_common::resolve` with the residual beside it — the whole tail of a filter
// that has no opinion about coverage, which is every point filter and every clipped
// one (§21.4). Both halves of the color take the identical strength mix on the
// identical coverage, and both ride the alpha `resolve` already decided (§6.7).
//
// One definition, two callers: the point path below and the clipped gather, which is
// the same argument `filter_common::resolve` makes for being shared between the two
// spaces. The residual's own target alpha is `cb.a` too — duplicated only because
// the fixed-function "over" that filled these targets reads each one's.
fn resolve_latent(cb: vec4<f32>, was: Pigment, now: Pigment, h: f32) -> FsOut {
    let base = resolve(cb, was.conc, now.conc, h);
    var out: FsOut;
    out.color = base.color;
    out.aux = base.aux;
    out.resid = vec4<f32>(weigh(was.res, now.res) * cb.a, cb.a);
    return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_clip(vi);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));

    let cb = textureLoad(back_color, p, 0); // (concentrations · coverage, coverage)
    let rb = textureLoad(back_resid, p, 0); // (residual · coverage, coverage)
    let h = textureLoad(back_aux, p, 0).x;

    if f.kind == FILTER_CHROMATIC {
        // The spectral gather (§21.10), as `filter_oklab.wesl` writes it — the one
        // difference is the decode: a tap is a pigment mixture, so reaching its
        // light costs the polynomial plus the residual, per tap.
        let dim = vec2<f32>(textureDimensions(back_color));
        let n = ca_taps(f.disp);
        var lin_acc = vec3<f32>(0.0); // Σ w · light · coverage (premultiplied light)
        var w_acc = vec3<f32>(0.0); // Σ w, the per-channel partition of unity
        var cov_acc = 0.0; // Σ y · coverage
        var h_acc = 0.0; // Σ y · height
        var y_acc = 0.0; // Σ y
        for (var i = 0u; i < n; i++) {
            let s = ca_s(i, n);
            let uv = ca_uv(frag.xy, s, dim);
            let cs = textureSampleLevel(back_color, back_samp, uv, 0.0);
            let rs = textureSampleLevel(back_resid, back_samp, uv, 0.0);
            let hs = textureSampleLevel(back_aux, back_samp, uv, 0.0).x;
            let w = ca_weight(s);
            let y = ca_lum(w);
            let conc = cs.rgb / max(cs.a, COVERAGE_EPS);
            let res = rs.rgb / max(cs.a, COVERAGE_EPS);
            lin_acc += w * mixbox_to_linear(Pigment(conc, res)) * cs.a;
            w_acc += w;
            cov_acc += y * cs.a;
            h_acc += y * hs;
            y_acc += y;
        }
        let a = cov_acc / y_acc;
        // Out of light and back into pigment, once, for the summed spectrum.
        let gathered = mixbox_from_linear(
            pigment_lut, pigment_samp, lin_acc / w_acc / max(a, COVERAGE_EPS),
        );
        // **Clipped**: the fringe's color lands, its displaced coverage does not
        // (§21.4) — `filter_oklab.wesl`'s branch, with the residual carried through
        // the same tail the point path takes.
        if is_clipped() {
            let cov = max(cb.a, COVERAGE_EPS);
            return resolve_latent(cb, Pigment(cb.rgb / cov, rb.rgb / cov), gathered, h);
        }
        // The strength mix on the raw premultiplied lanes — both halves of the
        // color, the coverage and the height together, so strength 0 is the
        // backdrop bit for bit (§21.4).
        var out: FsOut;
        out.color = mix(cb, vec4<f32>(gathered.conc * a, a), f.strength);
        out.resid = mix(rb, vec4<f32>(gathered.res * a, a), f.strength);
        out.aux = vec4<f32>(mix(h, h_acc / y_acc, f.strength), 0.0, 0.0, 0.0);
        return out;
    }

    if f.kind == FILTER_FOCAL_BLUR {
        // The convolution already ran in light (§21.12, `blur.wesl`); what is
        // pigment's alone is the one re-entry — the inverse LUT with the residual
        // recomputed for the arrived-at color, exactly the gather's ending (§6.7).
        let blurred = blur_read(frag.xy);
        let l = mixbox_from_linear(pigment_lut, pigment_samp, blurred.lin);
        if is_clipped() {
            let cov = max(cb.a, COVERAGE_EPS);
            return resolve_latent(cb, Pigment(cb.rgb / cov, rb.rgb / cov), l, h);
        }
        var out: FsOut;
        out.color = mix(cb, vec4<f32>(l.conc * blurred.cov, blurred.cov), f.strength);
        out.resid = mix(rb, vec4<f32>(l.res * blurred.cov, blurred.cov), f.strength);
        out.aux = vec4<f32>(mix(h, blurred.h, f.strength), 0.0, 0.0, 0.0);
        return out;
    }

    // Both halves un-premultiply by the *color's* coverage — the residual target's
    // own alpha is the same number, duplicated only because the fixed-function
    // "over" that filled these targets reads each one's alpha.
    let cov = max(cb.a, COVERAGE_EPS);
    let was = Pigment(cb.rgb / cov, rb.rgb / cov);

    return resolve_latent(cb, was, lab_to_latent(filtered(latent_to_lab(was))), h);
}

// The focal blur's decode (§21.12): `filter_oklab.wesl`'s, with the one
// substitution pigment always makes — a texel's light costs the polynomial plus
// its residual (§6.7). The residual is *consumed* here and *recomputed* in
// `fs_main`'s re-entry, so nothing of it rides the transform.
@fragment
fn fs_blur_decode(@builtin(position) frag: vec4<f32>) -> BlurOut {
    let src = blur_src(frag.xy);
    if !src.inside {
        return blur_pad();
    }
    var cb: vec4<f32>;
    var rb: vec4<f32>;
    var h: f32;
    if src.load {
        cb = textureLoad(back_color, src.p, 0);
        rb = textureLoad(back_resid, src.p, 0);
        h = textureLoad(back_aux, src.p, 0).x;
    } else {
        cb = textureSampleLevel(back_color, back_samp, src.uv, 0.0);
        rb = textureSampleLevel(back_resid, back_samp, src.uv, 0.0);
        h = textureSampleLevel(back_aux, back_samp, src.uv, 0.0).x;
    }
    let conc = cb.rgb / max(cb.a, COVERAGE_EPS);
    let res = rb.rgb / max(cb.a, COVERAGE_EPS);
    let light = linear_to_light(mixbox_to_linear(Pigment(conc, res)));
    return BlurOut(
        vec4<f32>(light * cb.a, cb.a),
        vec4<f32>(h, 1.0, 0.0, 0.0),
    );
}

// —— the merge (§14.11.7) ——————————————————————————————————————————————————————
//
// `filter_oklab.wesl`'s `fs_tile`, in pigment: one tile's stored latent adjusted in
// place, for a filter layer merged into the paint beneath it. That file's header
// carries the argument — why a tile needs no trip through `slab.wesl`, and why only a
// **point** filter may come here (§6.4's seam rule, which a gather cannot keep).
//
// The one thing this adds is the one thing pigment always adds: the color has two
// halves, so both un-premultiply by the tile's own opacity and both come back through
// `resolve_latent` (§6.7).
@fragment
fn fs_tile(@builtin(position) frag: vec4<f32>) -> FsOut {
    let p = vec2<i32>(floor(frag.xy));
    let c = textureLoad(back_color, p, 0); // (concentrations · opacity, opacity)
    let r = textureLoad(back_resid, p, 0); // (residual · opacity, opacity)
    let h = textureLoad(back_aux, p, 0).x;
    let opacity = max(c.a, PAINT_EPS);
    let was = Pigment(c.rgb / opacity, r.rgb / opacity);
    return resolve_latent(c, was, lab_to_latent(filtered(latent_to_lab(was))), h);
}
