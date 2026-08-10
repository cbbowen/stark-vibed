// The **layer blend** pass (§18.0.4, §6.3). One
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
//
// The conversion pair itself is `lib/color.wesl`'s `linear_to_light` /
// `light_to_linear`. It lived here while this was the only pass that worked in
// light; the filter pass (§21) scales light too, and a module in `lib` is where a
// binding-free conversion two pipelines share belongs — importing it out of *this*
// file would have dragged this one's bindings along with it.

// (The un-premultiply guard this pass needs is `COVERAGE_EPS`, in
// `lib/paint_common.wesl` — it used to be a local `EPS` here at 1e-4 while four
// other modules had an `EPS` at 1e-5. The importers take it directly.)

// Blend-mode codes. Mirrored by `blend_code` in `gpu/composite.rs` — this is the
// shader ABI for `BlendMode`, so it lives next to the shader rather than in the
// enum, which has no business knowing a `u32`.
//
// `MODE_NORMAL` reaches this pass only when the layer is **clipped**
// (§14.4): an unclipped normal layer needs no isolation and draws
// straight into the accumulator under fixed-function `over`, which is the fast
// path an ordinary document never leaves.
const MODE_NORMAL: u32 = 0u;
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
    // Whether this layer is **clipped** to the paint beneath it: 0 or 1
    // (§14.4). Read by `merge`, which is where the whole of clipping
    // lives.
    clip: u32,
    // The group's own opacity. For a **group** this is the only place opacity can
    // be applied: its members overlap, so fading each of them is not the same
    // thing as fading their composite (§14.7). A leaf layer's
    // opacity rides on its tiles instead (composite.wesl) and this is 1.0 —
    // exactly, so the arithmetic below is unchanged for it.
    opacity: f32,
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
    // Normal's blend function is "the source" — it covers rather than combines.
    // Reached only by a clipped layer, and `merge` does not read what this
    // returns for it: taking the source *premultiplied* instead is the same
    // number without a round trip out of the working space and back, which is
    // what keeps a clipped layer over opaque paint identical to an unclipped one.
    if b.mode == MODE_NORMAL {
        return y;
    }
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
//
// CLIPPING (§14.4)
//
// It is a deletion, not a scale — and that distinction is the design. Reading
// "multiply by the opacity of what it composites onto" literally, as `αs ← αs·αb`,
// invents coverage: at `αb = 0.5, αs = 1` it returns alpha 0.75, so a clipped layer
// makes the group *more* opaque than the group is and the backdrop shows through
// paint that should be solid. Expanding the mix instead shows the three terms this
// function is actually made of,
//
//     rgb = (1−αb)·αs·Cs  +  αb·αs·B(Cb,Cs)  +  cb.rgb·(1−αs)
//     a   =    αs·(1−αb)  +  αb·αs           +  αb·(1−αs)
//
// and clipping is: drop the first one from each. The alpha then collapses to
// exactly `αb` — the backdrop's coverage is untouched by anything clipped to it,
// which is the property that makes clipping composable and which the scaled-alpha
// version does not have. Note the tail keeps the *unmodified* `αs`: inside the
// backdrop's region the source still covers `αs` of it.
//
// The two weights below are written so that the unclipped case is **exactly** the
// arithmetic this function performed before clipping existed — `x * 1.0` and
// `x * keep` with `keep` an exact 1.0 — rather than something that agrees to a
// least-significant bit. `tests/blend.rs` compares byte-for-byte, and a `mix` that
// reconstructs 1.0 from `(1−αb) + αb` would not survive it.
//
// The aux is summed, mode or no mode. Aux is *amount of paint*, and paint stacks
// whatever its colour does — height conservation is an invariant of the medium, not
// of the blend (§6.1). Clipping is the one thing that reaches it: a clipped
// layer's colour is suppressed outside the backdrop, and height that stayed behind
// would light as relief with no paint under it — ghost impasto (§14.4.2). So the source
// height carries the same weight the source's coverage does.
//
// One consequence worth naming: a **matte** inside a non-`Normal` group keeps the
// additive merge, so it will not erase the relief under it the way a `Normal` matte
// does (§15.4.2). A frame that screens light is not a thing anyone has
// asked for; if it becomes one, this is the line that has to learn the difference.
// `hb` / `hs` are the two aux heights. They arrive as arguments like everything else
// rather than being loaded from `back_aux` / `src_aux` in here, which is what this
// function used to do — and the `p: vec2<i32>` it needed to do it was the only reason
// a compositing routine took a texel coordinate at all. Half its inputs came down the
// parameter list and half were fetched behind the caller's back, at a position the
// caller had already computed. Now the whole of the algebra below is a function of its
// arguments and the mode uniform, and the two colour spaces read identically.
fn merge(cb: vec4<f32>, cs_raw: vec4<f32>, blended: vec3<f32>, hb: f32, hs: f32) -> FsOut {
    // The group's opacity, applied to its composited result — premultiplied
    // colour and coverage scale together, which is the same thing
    // `composite.wesl` does to a leaf layer's tiles one at a time.
    let cs = cs_raw * b.opacity;
    let unclipped = b.clip == 0u;
    // How much of the source survives overall — 1 unclipped, the backdrop's own
    // coverage when clipped.
    let keep = select(cb.a, 1.0, unclipped);
    var out: FsOut;
    out.color = vec4<f32>(
        merge_triple(cb.rgb, cb.a, cs.rgb, cs.a, blended),
        cs.a * keep + cb.a * (1.0 - cs.a),
    );
    out.aux = vec4<f32>(hb + hs * keep * b.opacity, 0.0, 0.0, 0.0);
    return out;
}

/// One premultiplied colour triple through the three-term merge above, given the
/// backdrop's and (opacity-scaled) source's own alphas.
///
/// Split out of [`merge`] because a **pigment** document runs it twice: once on the
/// concentrations and once on the residual beside them, which is not a second colour
/// to combine but the rest of the same one (§6.7). The alpha is decided
/// once, by `merge`, and both triples ride it — so this returns only the triple.
///
/// Everything that made the unclipped case exact stays exact here: `outside` and the
/// `Normal` shortcut are the same expressions in the same order, so `tests/blend.rs`'s
/// byte-for-byte comparison is unaffected by the extraction.
fn merge_triple(
    cb_rgb: vec3<f32>, cb_a: f32, cs_rgb: vec3<f32>, cs_a: f32, blended: vec3<f32>,
) -> vec3<f32> {
    // The weight on source that lands where the backdrop is *not*: all of it, or
    // none of it.
    let outside = select(0.0, 1.0, b.clip == 0u);
    // `B(Cb,Cs)·αs`, premultiplied. For `Normal` that product is `cs.rgb` by
    // definition, and taking it verbatim rather than dividing by `αs` and
    // multiplying it back is what makes a **clipped normal layer over opaque
    // paint** bit-identical to the fixed-function `over` an unclipped one gets.
    let src_term = select(blended * cs_a, cs_rgb, b.mode == MODE_NORMAL);
    return mix(cs_rgb * outside, src_term, cb_a) + cb_rgb * (1.0 - cs_a);
}
