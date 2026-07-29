// The **layer blend** pass (MISSING_FEATURES.md §0.4, DESIGN.md §6.3). One
// fullscreen draw that merges an isolated layer's composited channels into the
// accumulator through a non-`Normal` blend mode.
//
// Everything here is space-independent: the bindings, the light algebra, and the
// Porter-Duff tail. Each colour space supplies only `to_light` / `from_light` — its
// channels ↔ *normalized CIE XYZ* — and calls `merge`. That split is the same one
// `media_common.wesl` makes, and for the same reason: there is one blend model, not
// one per colour space.
//
//
// WHY THESE MODES, AND WHY NOT SCREEN
//
// Screen is `a + b − ab`. It is what you get by inverting a multiply, and it
// describes nothing: no optical process produces it, and it flattens the top of the
// range into the chalky white that reads as "digital glow" at a glance.
//
// Light does exactly one thing when you combine it: it adds. But the numbers in a
// layer are not light — they are light that has *already* been through a tone curve
// on its way to being displayable. So the honest combination is to undo the curve,
// add, and put the curve back:
//
//     f(a, b) = T(T⁻¹(a) + T⁻¹(b))
//
// Every mode below is that sentence with a different `T`. Writing them as
// conjugations of `+` is what buys the properties that matter in practice:
// commutativity, associativity, and a neutral element — so stacking three glow
// layers is order-independent and regrouping them changes nothing, which is what
// real lamps do and what a blend mode has to do if a painting is going to survive
// being reordered.
//
// `multiply` is the same sentence read the other way round, and it is the reason the
// paragraph above can be so rude about Screen. Take `T(x) = e^{-x}`, so `T⁻¹(y) =
// −log y`. Then the conjugation collapses to plain `f(a,b) = ab`, and the quantity
// that adds is **optical density** — which is Beer-Lambert, exactly what two stacked
// filters, two glazes, or two sheets of stained glass do to the light passing
// through them. Screen is this function conjugated *again* by `x ↦ 1−x`, and that
// second conjugation is the step with no physical referent: nothing in optics
// complements a transmittance. So of the two, multiply is the real one and Screen is
// the shadow it casts, which is why one is here and the other is not.
//
// The dual runs all the way down. The emissive modes add light and have **black** as
// their identity; multiply adds density and has **white** as its identity, black
// being an annihilator instead. Both are still commutative and associative, so the
// stack-reordering guarantee is unchanged — a glaze does not care what order it was
// laid down in either.
//
//
// WHY NORMALIZED XYZ
//
// The combination runs in CIE XYZ divided by the display white (D65), not in the
// working space and not in RGB.
//
//   · **Linear in light.** Oklab's L is a perceptual cube root and Mixbox's channels
//     are pigment concentrations; adding either is adding something that is not
//     light, and `T⁻¹` would be undoing a curve that was never applied.
//   · **Non-negative.** Every real colour has non-negative tristimulus values. Both
//     curves need that: Reinhard's denominator and Drago's logarithm are only
//     defined on light that exists. Linear RGB does not have the property — an
//     out-of-gamut colour has a negative channel — which is precisely the case that
//     would produce NaN.
//   · **White is 1 on all three axes.** The sRGB→XYZ rows sum to the white point, so
//     dividing each row by its own sum makes an in-gamut colour land in `[0,1]³` and
//     makes "1" mean the same thing on X, Y and Z. Without it, Z tops out at 1.089
//     and Reinhard's `1 − ab` changes sign. Multiply needs it even more directly:
//     white is its *identity*, and in raw XYZ multiplying by white would scale a
//     colour by `(0.9505, 1.0, 1.089)` — every glaze over white paper coming back
//     dimmer and colder than it went in.
//   · **Primary-independent.** Blending in RGB makes the result a function of the
//     display's primaries. Two apps with different working primaries would disagree
//     about what "glow" means; XYZ does not have an opinion.

const EPS: f32 = 1e-4;

// Blend-mode codes. Mirrored by `blend_code` in `gpu/composite.rs` — this is the
// shader ABI for `BlendMode`, so it lives next to the shader rather than in the
// enum, which has no business knowing a `u32`.
const MODE_REINHARD: u32 = 1u;
const MODE_DRAGO: u32 = 2u;
const MODE_MULTIPLY: u32 = 3u;

struct Blend {
    // Which mode (see the codes above).
    mode: u32,
    // Drago's curve bend, in units of display white (`document::DRAGO_K`). Passed
    // rather than hardcoded so the constant has exactly one home, on the Rust side
    // where the doc comment and the tests can reach it.
    k: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> b: Blend;
// The accumulator: everything below this layer, already composited.
@group(0) @binding(1) var back_color: texture_2d<f32>;
@group(0) @binding(2) var back_aux: texture_2d<f32>;
// The isolated layer: this layer alone, on nothing.
@group(0) @binding(3) var src_color: texture_2d<f32>;
@group(0) @binding(4) var src_aux: texture_2d<f32>;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// Fullscreen triangle clip position from a vertex index. A local copy rather than
// `media_common`'s: importing that module would drag its whole `@group(0)` in, and
// this pass binds something else entirely.
fn fullscreen_clip(vi: u32) -> vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Linear sRGB → XYZ normalized to D65 white. These are the sRGB primaries' rows,
// each divided by its own sum (= the white's tristimulus value), so `(1,1,1)` maps
// to `(1,1,1)`.
fn linear_to_light(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.4339395, 0.3762078, 0.1898528)),
        dot(c, vec3<f32>(0.2126729, 0.7151522, 0.0721750)),
        dot(c, vec3<f32>(0.0177566, 0.1094671, 0.8727763)),
    );
}

// The exact inverse: the XYZ→linear-sRGB matrix with column *j* scaled by the white
// point's *j*th component, undoing the row normalization above.
fn light_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>( 3.0799545, -1.5371385, -0.5428160)),
        dot(c, vec3<f32>(-0.9212583,  1.8760108,  0.0452477)),
        dot(c, vec3<f32>( 0.0528873, -0.2040259,  1.1511385)),
    );
}

// Reinhard: `T(x) = x/(1+x)`, so `T⁻¹(y) = y/(1−y)` and
//
//     f(a,b) = (a + b − 2ab) / (1 − ab)
//
// evaluated here in terms of the *complements* `u = 1−a`, `v = 1−b`, which is the
// same function:
//
//     1 − f = uv / (u + v − uv)
//
// That form is the one to compute. The direct expression is 0/0 at `a = b = 1` —
// two white layers, the single most likely thing an author does — whereas this one
// reads `0 / max(0, ε)` and lands on exactly 1, which is the right answer. It also
// makes the mode's defining property visible: the numerator carries a factor of both
// complements, so the result reaches 1 only when an input already is 1. No finite
// stack of glow layers clips.
fn reinhard(a: f32, bb: f32) -> f32 {
    let u = 1.0 - a;
    let v = 1.0 - bb;
    return 1.0 - u * v / max(u + v - u * v, 1e-6);
}

// Drago: `T(x) = k·log(1 + x/k)`, so `T⁻¹(y) = k(e^{y/k} − 1)` and
//
//     f(a,b) = k·log(e^{a/k} + e^{b/k} − 1)
//
// factored by the larger input so no exponential is ever raised on a large positive
// argument:
//
//     f = m + k·log(1 + e^{(n−m)/k} − e^{−m/k}),   m = max(a,b),  n = min(a,b)
//
// Both exponentials now have non-positive arguments, so the whole thing is stable at
// any `k`, and the log's argument stays ≥ `e^{(n−m)/k} > 0`. Unlike `reinhard` this
// overshoots: two whites give `1 + k·log 2`. That is deliberate — see `BlendMode`.
fn drago(a: f32, bb: f32, k: f32) -> f32 {
    let m = max(a, bb);
    let n = min(a, bb);
    return m + k * log(1.0 + exp((n - m) / k) - exp(-m / k));
}

// Combine two light values, channel by channel.
//
// Inputs are clamped into `[0,1]` first: the curves are defined on light between
// black and display white, and a channel outside that came from a colour outside the
// gamut, not from light that is really there. Multiply wants the clamp for a blunter
// reason — two negative channels multiply to a *positive* one, so an out-of-gamut
// pair would come back brighter from the one mode that only ever darkens. The
// *output* is deliberately not clamped — Drago's overshoot is the feature, and the
// half-float composite target carries it into the media pass's highlight roll-off.
fn combine(back: vec3<f32>, src: vec3<f32>) -> vec3<f32> {
    let x = clamp(back, vec3<f32>(0.0), vec3<f32>(1.0));
    let y = clamp(src, vec3<f32>(0.0), vec3<f32>(1.0));
    // Multiply is `T(T⁻¹(a) + T⁻¹(b))` for `T(x) = e^{-x}` — adding optical density,
    // which is what a glaze does. No helper: unlike the emissive curves there is
    // nothing to rearrange for stability, because there is nothing to be unstable.
    if b.mode == MODE_MULTIPLY {
        return x * y;
    }
    if b.mode == MODE_DRAGO {
        return vec3<f32>(drago(x.x, y.x, b.k), drago(x.y, y.y, b.k), drago(x.z, y.z, b.k));
    }
    return vec3<f32>(reinhard(x.x, y.x), reinhard(x.y, y.y), reinhard(x.z, y.z));
}

// Compose the blended layer onto the accumulator (the tail every mode shares).
//
// `cb` / `cs` are the two premultiplied samples and `cr` is the *un-premultiplied*
// blended colour the caller derived. This is W3C compositing's source-over with a
// blend function, in the app's own channel semantics: alpha here is the layer's
// **coverage** — its slab visible alpha, accumulated by pass A (composite.wesl) —
// which is exactly the weight `Normal` gives a layer, so a mode that changes only
// the colour changes only the colour.
//
// The aux is summed unconditionally, mode or no mode. Aux is *amount of paint*, and
// paint stacks whatever its colour does — height conservation is an invariant of the
// medium, not of the blend (§6.1). One consequence worth naming: a **matte** inside a
// non-`Normal` group keeps the additive merge, so it will not erase the relief under
// it the way a `Normal` matte does (FRAME_DESIGN.md §4.2). A frame that screens light
// is not a thing anyone has asked for; if it becomes one, this is the line.
// `blended` is `B(Cb, Cs)` un-premultiplied — the mode's opinion, before any of it
// is weighed against how much of the two layers is actually there.
//
// The interpolation `mix(…, αb)` is W3C's `(1 − αb)·Cs + αb·B(Cb, Cs)`: blend
// against what is genuinely underneath, and where nothing is, the layer is simply
// itself. It runs in the **working space** rather than in light, which is not a
// detail — it means a layer over empty canvas takes the identity branch and never
// makes the trip out to light and back. In a Mixbox document, where that trip is a
// LUT lookup which drops the latent residual, it is the difference between
// "unchanged" and "slightly recoloured for nothing".
//
// The source term is written `cs.rgb` and not `Cs·αs`, which is the same number:
// `Cs` was only ever `cs.rgb` divided by `αs` so the blend function could see a
// colour rather than a premultiplied one, and multiplying it back would round-trip a
// division for no reason. Spelling it this way makes the identity *exact* — a mode
// over an empty stack is bit-for-bit the `Normal` render, which `tests/blend.rs`
// asserts to the byte — instead of exact-to-a-least-significant-bit.
//
// The aux is summed unconditionally, mode or no mode. Aux is *amount of paint*, and
// paint stacks whatever its colour does — height conservation is an invariant of the
// medium, not of the blend (DESIGN.md §6.1). One consequence worth naming: a **matte**
// inside a non-`Normal` group keeps the additive merge, so it will not erase the
// relief under it the way a `Normal` matte does (FRAME_DESIGN.md §4.2). A frame that
// screens light is not a thing anyone has asked for; if it becomes one, this is the
// line that has to learn the difference.
fn merge(cb: vec4<f32>, cs: vec4<f32>, blended: vec3<f32>, p: vec2<i32>) -> FsOut {
    var out: FsOut;
    out.color = vec4<f32>(
        mix(cs.rgb, blended * cs.a, cb.a) + cb.rgb * (1.0 - cs.a),
        cs.a + cb.a * (1.0 - cs.a),
    );
    out.aux = vec4<f32>(
        textureLoad(back_aux, p, 0).x + textureLoad(src_aux, p, 0).x,
        0.0, 0.0, 0.0,
    );
    return out;
}
