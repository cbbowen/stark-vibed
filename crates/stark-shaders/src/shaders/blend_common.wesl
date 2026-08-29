// The **layer blend** pass (§18.0.4, §6.3). One
// fullscreen draw that merges an isolated layer's composited channels into the
// accumulator through a non-`Normal` blend mode.
//
// Everything here is space-independent: the bindings, the light algebra, and the
// Porter-Duff tail. Each color space supplies only `to_light` / `from_light` — its
// channels ↔ *normalized CIE XYZ* — and calls `merge`. That split is the same one
// `media_common.wesl` makes, and for the same reason: there is one blend model, not
// one per color space.
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
//   · **Non-negative.** Every real color has non-negative tristimulus values. Both
//     curves need that: Reinhard's denominator and Drago's logarithm are only
//     defined on light that exists. Linear RGB does not have the property — an
//     out-of-gamut color has a negative channel — which is precisely the case that
//     would produce NaN.
//   · **White is 1 on all three axes.** The sRGB→XYZ rows sum to the white point, so
//     dividing each row by its own sum makes an in-gamut color land in `[0,1]³` and
//     makes "1" mean the same thing on X, Y and Z. Without it, Z tops out at 1.089
//     and Reinhard's `1 − ab` changes sign. Multiply needs it even more directly:
//     white is its *identity*, and in raw XYZ multiplying by white would scale a
//     color by `(0.9505, 1.0, 1.089)` — every glaze over white paper coming back
//     dimmer and colder than it went in.
//   · **Primary-independent.** Blending in RGB makes the result a function of the
//     display's primaries. Two apps with different working primaries would disagree
//     about what "glow" means; XYZ does not have an opinion.
//
// The conversion pair itself is `lib/color.wesl`'s `linear_to_light` /
// `light_to_linear`. The filter pass (§21) scales light too, and a module in `lib`
// is where a binding-free conversion two pipelines share belongs — importing it out
// of *this* file would drag this one's bindings along with it.

// (The un-premultiply guard this pass needs is `COVERAGE_EPS`, in
// `lib/paint_common.wesl`. The importers take it directly, so there is one guard on
// this boundary rather than a per-module `EPS` at whatever magnitude each picked.)

// Blend-mode codes. This is the shader ABI for `BlendMode`, so it lives next to the
// shader rather than in the enum, which has no business knowing a `u32`. The host
// reads these very declarations through the generated mirror and maps the enum onto
// them in `blend_code` (`gpu/composite/blend.rs`) — a mapping, not a second copy of
// the numbers (§6.10).
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
    // Drago's curve bend, in units of display white — **the layer's own**, carried by
    // `BlendMode::Drago` (§18.0.4). Every other mode branches before reading it and
    // sends `document::DRAGO_K` here, since the uniform has one field and no way to
    // spell "absent".
    //
    // The shader implements the mode's whole *family* of curves rather than one of
    // them, so where `k` comes from — a constant or a per-layer setting — is entirely
    // the host's business.
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

// ---- the emissive modes' conjugating curve, as a pair --------------------------
//
// `Reinhard` and `Drago` are `f(a,b) = T(T⁻¹(a) + T⁻¹(b))`, and this is `T⁻¹` and `T`
// — written as the pair rather than as a collapsed two-argument form, because **the
// addition in the middle is the part that has to be reachable**.
// Collapsed, the curve can only combine two values at full strength; separated, the
// quantity that adds is a value in its own hand, so coverage and opacity can weigh it
// (`added_light`) and a stack of any depth is one sum.
//
// `emission` is the linear light a display-referred value stands for, and it is where
// the two modes actually differ. Everything else about them is the addition.

// The largest emission `T⁻¹` will report, as its reciprocal. Reinhard's inverse is
// `y/(1−y)`, which is unbounded at display white — the single most likely input an
// author produces — so it needs a floor under the denominator, and this is it. At
// 1e-6 the cap is 1e6, whose tonemap is 0.999999: white comes back white to five
// places past what an 8-bit target can show, and a sum of several stays finite in
// f32 rather than reaching the `inf/inf` that would make a NaN of the whole texel.
//
// A collapsed form working in the complements would land `a = b = 1` on exactly 1.
// That exactness is the one thing the floor gives up, and it is given up for a reason:
// the complement form cannot express "half a lamp".
const EMISSION_EPS: f32 = 1e-6;

// `T⁻¹`: the linear emission behind a display-referred light value.
//
// Clamped into `[0,1]` first, for `combine`'s reason: the curves are defined on light
// between black and display white, and a channel outside that came from a color
// outside the gamut rather than from light that is really there. Drago needs no floor
// — with the input clamped, `e^{y/k}` tops out at `e^{1/k}`, which for any sane `k`
// is a small number — so the guard is Reinhard's alone.
fn emission(x: vec3<f32>) -> vec3<f32> {
    let y = clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
    if b.mode == MODE_DRAGO {
        return b.k * (exp(y / b.k) - vec3<f32>(1.0));
    }
    return y / max(vec3<f32>(1.0) - y, vec3<f32>(EMISSION_EPS));
}

// `T`: the display-referred light value an emission tonemaps to.
//
// Deliberately **not** clamped on the way out. Reinhard's curve is asymptotic, so it
// cannot pass 1 however much light it is given; Drago's is a logarithm and does, and
// that overshoot is the feature — the half-float composite target carries it into the
// media pass's highlight roll-off (see `BlendMode::Drago`).
fn tonemap(e: vec3<f32>) -> vec3<f32> {
    let x = max(e, vec3<f32>(0.0));
    if b.mode == MODE_DRAGO {
        return b.k * log(vec3<f32>(1.0) + x / b.k);
    }
    return x / (vec3<f32>(1.0) + x);
}

// Whether this mode combines by **adding light** — the emissive half of the family.
//
// What it decides is which space the merge below runs in, and that is a property of
// the mode rather than of the layer: the emissive modes are affine in *emission*, and
// `Multiply` is affine in the light itself.
fn is_emissive() -> bool {
    return b.mode == MODE_REINHARD || b.mode == MODE_DRAGO;
}

// Whether this mode **covers** rather than combines — the one that needs no light at
// all, and the only one `merge` still settles in the working space.
//
// Reached only by a clipped layer or a faded group (§14.4): an unclipped `Normal` layer
// at full opacity is the absence of a pass.
fn is_normal() -> bool {
    return b.mode == MODE_NORMAL;
}


// The coverage a merge leaves behind, clip included (§14.4.1) — one definition,
// because `combined_light` divides by the very number `merge` writes into alpha, and
// the two disagreeing would spread the light over the wrong area.
fn merge_alpha(cb_a: f32, cs_a: f32) -> f32 {
    let keep = select(cb_a, 1.0, b.clip == 0u);
    return cs_a * keep + cb_a * (1.0 - cs_a);
}

// Compose the blended layer onto the accumulator (the tail every mode shares).
//
// `cb` / `cs` are the two premultiplied samples and `cr` is the *un-premultiplied*
// blended color the caller derived. This is W3C compositing's source-over with a
// blend function, in the app's own channel semantics: alpha here is the layer's
// **coverage** — its slab visible alpha, accumulated by pass A (composite.wesl) —
// which is exactly the weight `Normal` gives a layer, so a mode that changes only
// the color changes only the color.
//
// `blended` is `B(Cb, Cs)` un-premultiplied — the mode's opinion, before any of it
// is weighed against how much of the two layers is actually there.
//
// The interpolation `mix(…, αb)` is W3C's `(1 − αb)·Cs + αb·B(Cb, Cs)`: blend
// against what is genuinely underneath, and where nothing is, the layer is simply
// itself. It runs in the **working space**, which is right for `Multiply` and wrong
// for the emissive modes — those take `added_light` instead. What this handles is
// `Normal` (reached only by a clipped layer) and the one mode whose blend function
// is bilinear, so that working-space mixing *is* its linear quantity.
//
// The source term is written `cs.rgb` and not `Cs·αs`, which is the same number:
// `Cs` is just `cs.rgb` divided by `αs` so the blend function can see a color rather
// than a premultiplied one, and multiplying it back would round-trip a division for
// no reason. Spelling it this way is what makes an **empty backdrop**
// exact: `mix(x, y, 0)` is `x` to the bit, so a mode over an empty stack is
// bit-for-bit the `Normal` render, which `tests/blend.rs` asserts to the byte. The
// space shaders route `αb = 0` here for exactly that reason rather than through
// `added_light`, whose round trip out to light and back could not promise it — and
// in a Mixbox document that trip is a LUT lookup which drops the latent residual, so
// it is the difference between "unchanged" and "slightly recolored for nothing".
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
// The two weights below are written so that an **unclipped** layer runs exactly the
// arithmetic it would with clipping out of the picture entirely — `x * 1.0` and
// `x * keep` with `keep` an exact 1.0 — rather than something that agrees only to a
// least-significant bit. `tests/blend.rs` compares byte-for-byte, and a `mix` that
// reconstructs 1.0 from `(1−αb) + αb` would not survive it.
//
// The aux is summed, mode or no mode. Aux is *amount of paint*, and paint stacks
// whatever its color does — height conservation is an invariant of the medium, not
// of the blend (§6.1). Clipping is the one thing that reaches it: a clipped
// layer's color is suppressed outside the backdrop, and height that stayed behind
// would light as relief with no paint under it — ghost impasto (§14.4.2). So the source
// height carries the same weight the source's coverage does.
//
// One consequence worth naming: a **matte** inside a non-`Normal` group keeps the
// additive merge, so it will not erase the relief under it the way a `Normal` matte
// does (§15.4.2). A frame that screens light is not a thing anyone has
// asked for; if it becomes one, this is the line that has to learn the difference.
// `hb` / `hs` are the two aux heights. They arrive as arguments like everything else
// rather than being loaded from `back_aux` / `src_aux` in here: **the whole of the
// algebra below is a function of its arguments and the mode uniform**, so the two
// color spaces read identically and nothing is fetched behind the caller's back at a
// position the caller has already computed. It is also why no compositing routine
// here takes a texel coordinate.
fn merge(cb: vec4<f32>, cs_raw: vec4<f32>, blended: vec3<f32>, hb: f32, hs: f32) -> FsOut {
    // The group's opacity, applied to its composited result — premultiplied
    // color and coverage scale together, which is the same thing
    // `composite.wesl` does to a leaf layer's tiles one at a time.
    let cs = cs_raw * b.opacity;
    let unclipped = b.clip == 0u;
    // How much of the source survives overall — 1 unclipped, the backdrop's own
    // coverage when clipped.
    let keep = select(cb.a, 1.0, unclipped);
    var out: FsOut;
    out.color = vec4<f32>(
        merge_triple(cb.rgb, cb.a, cs.rgb, cs.a, blended),
        merge_alpha(cb.a, cs.a),
    );
    out.aux = vec4<f32>(hb + hs * keep * b.opacity, 0.0, 0.0, 0.0);
    return out;
}

// [`merge`] for the emissive path, whose color the caller has already worked out.
//
// The two differ in one line and share the rest, which is the point: coverage and
// height are facts about how much paint is there, and no blend mode has an opinion
// about either. What the emissive modes change is the *color* law, so that is all
// this replaces — `chan` arrives un-premultiplied from `added_light`, converted back
// into the caller's own channels, and is premultiplied here by the very alpha
// `added_light` divided by.
fn merge_added(cb: vec4<f32>, cs_raw: vec4<f32>, chan: vec3<f32>, hb: f32, hs: f32) -> FsOut {
    let cs = cs_raw * b.opacity;
    let keep = select(cb.a, 1.0, b.clip == 0u);
    let a = merge_alpha(cb.a, cs.a);
    var out: FsOut;
    out.color = vec4<f32>(chan * a, a);
    out.aux = vec4<f32>(hb + hs * keep * b.opacity, 0.0, 0.0, 0.0);
    return out;
}

/// One premultiplied color triple through the three-term merge above, given the
/// backdrop's and (opacity-scaled) source's own alphas.
///
/// Split out of [`merge`] because a **pigment** document runs it twice: once on the
/// concentrations and once on the residual beside them, which is not a second color
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


// **The whole color algebra of every combining mode**, in one expansion.
//
// This is `merge_triple`'s three-term Porter-Duff expansion — the same one, the same
// weights — evaluated in **the space where the mode's blend function is affine**, and
// that phrase is the entire content of this function. Everything else is which space
// that is:
//
//   · The emissive modes are addition conjugated by a curve, so they are affine in
//     **emission** (`T⁻¹` of the light), where the expansion collapses outright:
//
//         (1−αb)·αs·Ls + αb·αs·(Lb+Ls) + αb·(1−αs)·Lb  =  αs·Ls + αb·Lb
//
//     — premultiplied emission simply adds.
//   · `Multiply` is Beer-Lambert, so it is bilinear in **light**, light being a
//     transmittance there (`T(x) = e^{-x}` makes the added quantity optical density).
//     The expansion does not collapse, but it does not need to: bilinear is affine in
//     each argument, which is all that matters.
//
// **Why the space is the whole question.** Porter-Duff weighs each layer by its
// coverage, and a weighted average commutes with the blend function only where that
// function is affine. Run it anywhere else — the working space, say — and `B` gets
// applied to the accumulator's coverage-*averaged* color, which is not the average of
// `B`. That is not a rounding difference: it made stacking order matter by 20 levels
// wherever coverage was partial, and it made two same-mode layers impossible to merge
// into one (§18.0.4, §14.11).
//
// It is also the physically right rule rather than the convenient one. Averaging over
// a partly-covered pixel is linear in the quantity that is *there*: radiance for a
// lamp, transmitted light for a glaze. Both are the mode's own space.
//
// Returns the **un-premultiplied** output light, so the caller converts one triple back
// into its own channels. `cs_raw_a` is the source's stored coverage, scaled here by the
// group opacity exactly as `merge` scales it.
fn combined_light(xb: vec3<f32>, cb_a: f32, xs: vec3<f32>, cs_raw_a: f32) -> vec3<f32> {
    let cs_a = cs_raw_a * b.opacity;
    // The coverage the merge leaves behind — the very number `merge` writes into
    // alpha, so the light is spread over exactly the area it will be read back from.
    let a = max(merge_alpha(cb_a, cs_a), EMISSION_EPS);
    if is_emissive() {
        let lb = emission(xb);
        let ls = emission(xs);
        return tonemap(merge_triple(lb * cb_a, cb_a, ls * cs_a, cs_a, lb + ls) / a);
    }
    // Clamped into `[0,1]` first: the curves are defined on light between black and
    // display white, and a channel outside that came from a color outside the gamut
    // rather than from light that is really there. Multiply wants it for a blunter
    // reason — two negative channels multiply to a *positive* one, so an out-of-gamut
    // pair would come back brighter from the one mode that only ever darkens.
    let lb = clamp(xb, vec3<f32>(0.0), vec3<f32>(1.0));
    let ls = clamp(xs, vec3<f32>(0.0), vec3<f32>(1.0));
    return merge_triple(lb * cb_a, cb_a, ls * cs_a, cs_a, lb * ls) / a;
}
