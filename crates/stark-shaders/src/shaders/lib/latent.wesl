// A color, both halves (§6.7) — and the arithmetic that moves one.
//
// A binding-free leaf, like every module under `lib/` (§2).
//
// **The residual is not a second quantity, it is the rest of the same color.** Every
// pass in this engine that touches a latent touches the residual at the same point, on
// the same weights, through the same law — and until this module existed, every one of
// them said so *twice*: a line for the latent and an `@if(resid)` line beside it,
// spelling the same expression against different names. A hundred and twenty-three of
// those gates across ten modules, and the correspondence between each pair was held by
// nothing but the two lines being adjacent.
//
// That is a hazard in the direction the type system cannot see. Writing
// `blend_latent(lat0, m0, lat1, m1)` and forgetting its partner compiles, links,
// renders, and is invisible in the default color space — Oklab has no residual to drop
// — so it fails only in a pigment document, as a color that reproduces neither of the
// two that made it.
//
// So the pair is one value, and the law is written once against it. `blend_latent`
// below is **not exported**: the parcel law is reachable only as [`latent_blend`], on a
// whole color, because there was no caller in the tree that legitimately wanted half of
// one. A pass can no more forget the residual than it can forget the green channel.
//
// **What is still gated is the boundary**, and that is the honest remainder: a residual
// *texture* exists only in a space that has one, so loading and storing it are `@if`s
// no bundle can absorb. Everything between a load and a store is now unconditional.

import package::lib::paint_common::{blend_latent};

// The two halves of one color: the space's channels, and what they cannot express.
//
// Premultiplied or straight as the caller pleases — every operation here is linear in
// both halves and neither knows nor cares, which is exactly why one type serves the
// tile representation, the parcel and the composited stack alike. What names the
// convention is the *call*: [`latent_unpremult`] is the conversion between them.
struct Latent {
    lat: vec3<f32>,
    @if(resid) res: vec3<f32>,
}

// A color from its two halves.
//
// Both arguments are read in every build; the residual is dropped where the space has
// none. That is what lets a caller hand over a value it holds unconditionally — the
// brush's own `st.brush_lat` / `st.brush_res`, which the uniform declares whether or
// not the space uses them (§6.10) — with no `@if` of its own.
fn latent_of(lat: vec3<f32>, res: vec3<f32>) -> Latent {
    var o: Latent;
    o.lat = lat;
    @if(resid) o.res = res;
    return o;
}

// Black, in both halves — the additive identity, for an accumulator.
fn latent_zero() -> Latent {
    return latent_of(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0));
}

// The **parcel-stacking law** (§6.1), on a whole color: `b` lands on top of `a`, each
// weighted by its own optical mass.
//
// The one way to evaluate it. `paint_common::blend_latent` states the law for a triple
// and is imported here and nowhere else, so the mass weights are computed once and both
// halves ride them — which is the property that makes the merged paint reproduce the
// two colors that made it rather than an average of one and a stale copy of the other.
fn latent_blend(a: Latent, m0: f32, b: Latent, m1: f32) -> Latent {
    var o: Latent;
    o.lat = blend_latent(a.lat, m0, b.lat, m1);
    // The identical blend: `blend_latent`'s weights are a function of `m0` and `m1`
    // alone, so calling it again with the residual is the same mixture (§6.7).
    @if(resid) o.res = blend_latent(a.res, m0, b.res, m1);
    return o;
}

// A premultiplied color divided back out by the coverage or opacity it was multiplied
// by — **the color's own**, for both halves.
//
// The residual target carries a duplicate of that alpha, and dividing by *it* is the
// bug this signature rules out: the two are the same number only until a pass writes
// one of them through a different blend. Taking the alpha as an argument makes the
// choice once, for both halves.
//
// Divided, not scaled by a reciprocal. `x / y` and `x * (1/y)` are two roundings apart,
// and every call site this replaces wrote the division — on paths whose exactness
// invariants are asserted to the byte (§16.4).
fn latent_unpremult(c: Latent, alpha: f32, eps: f32) -> Latent {
    return latent_div(c, max(alpha, eps));
}

fn latent_scale(a: Latent, s: f32) -> Latent {
    var o: Latent;
    o.lat = a.lat * s;
    @if(resid) o.res = a.res * s;
    return o;
}

// Divided rather than scaled by a reciprocal: the two are not the same in floats, and
// every call site this replaces wrote the division.
fn latent_div(a: Latent, d: f32) -> Latent {
    var o: Latent;
    o.lat = a.lat / d;
    @if(resid) o.res = a.res / d;
    return o;
}

fn latent_add(a: Latent, b: Latent) -> Latent {
    var o: Latent;
    o.lat = a.lat + b.lat;
    @if(resid) o.res = a.res + b.res;
    return o;
}

fn latent_sub(a: Latent, b: Latent) -> Latent {
    var o: Latent;
    o.lat = a.lat - b.lat;
    @if(resid) o.res = a.res - b.res;
    return o;
}

// `a·wa + b·wb` — the weighted sum a two-part parcel is mixed by, before it is divided
// through by the mass those weights add to.
fn latent_wsum(a: Latent, wa: f32, b: Latent, wb: f32) -> Latent {
    var o: Latent;
    o.lat = a.lat * wa + b.lat * wb;
    @if(resid) o.res = a.res * wa + b.res * wb;
    return o;
}

fn latent_lerp(a: Latent, b: Latent, t: f32) -> Latent {
    var o: Latent;
    o.lat = mix(a.lat, b.lat, t);
    @if(resid) o.res = mix(a.res, b.res, t);
    return o;
}

// Whether either half holds anything — the question a lateral flux asks before calling
// a texel moved (`dynamics.wesl`'s bleed).
//
// **The residual joins the verdict rather than riding on the latent's.** A flux that
// moved residual and no latent is possible: two colors with equal concentrations and
// different residuals is precisely what the channel exists to tell apart, and calling
// that texel unmoved would drop the transfer.
fn latent_any(a: Latent) -> bool {
    var hit = any(a.lat != vec3<f32>(0.0, 0.0, 0.0));
    @if(resid) hit = hit || any(a.res != vec3<f32>(0.0, 0.0, 0.0));
    return hit;
}

