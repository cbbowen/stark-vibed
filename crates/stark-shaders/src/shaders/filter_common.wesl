// The **filter layer** pass (§21). One fullscreen draw that reads the accumulator
// its stack has built so far, rewrites the colour, and writes the result back.
//
// This is the blend pass's shape with one input instead of two: a filter layer has
// no content of its own, so there is nothing to isolate and nothing to merge — it
// *is* the backdrop, adjusted. The ping-pong is the same and for the same reason
// (a texture cannot be both read and written), which is why a filter costs exactly
// what a blended layer costs and reuses the very same scratch (`composite/blend.rs`).
//
// Everything here is space-independent: the bindings, the adjustment, and the
// strength mix. Each colour space supplies only its channels ↔ **Oklab** — the
// space the knobs are defined in (§21.5) — and calls `filtered`. That is the split
// `blend_common.wesl` and `media_common.wesl` already make, and for the same reason:
// there is one filter model, not one per colour space. Oklab rather than light as
// the interface because it is where the adjustment happens anyway: an Oklab
// document then passes its channels straight in, instead of a trip out to XYZ that
// the first thing inside would exactly undo.
//
//
// WHY THE ACCUMULATOR IS NOT A COLOUR
//
// Pass A's colour target holds the layer stack **premultiplied by coverage**, and
// its aux holds the *amount of paint* (§6.1). A filter has opinions about the first
// and none about the second:
//
//   · Coverage is not the filter's business. A filter says what colour the paint
//     already there should be, not how much of it there is, so alpha comes out
//     exactly as it went in — and the adjustment runs on the *un*-premultiplied
//     channels, so a half-covered texel is adjusted like a full one rather than
//     like a darker one.
//   · Height is not the filter's business either. Relief is a property of the
//     medium, so the aux is copied across verbatim. That is a real copy rather than
//     a skipped attachment: the ping-pong means the pass's output targets are not
//     the ones it read, so an aux left unwritten would be whatever the previous
//     bounce left there.
//
//
// WHY THE ADJUSTMENT IS IN OKLAB
//
// Because in a perceptual space lightness, chroma and hue are separable, and that
// separability is the whole promise a colour slider makes. Saturation in sRGB shifts
// hue; contrast in sRGB shifts saturation; a "greyscale" that is a weighted RGB
// average moves lightness around. In Oklab a chroma gain is a chroma gain.
//
// Exposure is the exception, and it is one on purpose: it is applied to **light**,
// before the trip into Oklab. Doubling light is what an exposure *is* — a stop is a
// factor of two on the photons — and Oklab's `L` is the cube root of roughly that,
// so scaling `L` by `2^n` would be a number with no referent. It is computed on
// linear sRGB rather than on XYZ, and that is the same operation: the two encodings
// differ by a fixed linear matrix, and a scalar gain commutes with it.

import package::lib::color::{oklab_to_linear_srgb, linear_srgb_to_oklab, light_to_linear};

// Filter codes — the shader ABI for `document::Filter`, so they live next to the
// shader rather than on the enum, which has no business knowing a `u32`. Mirrored
// into the host by the generator (§6.10, `build.rs` `CONSTS`); `FilterDraw::new`
// (`gpu/composite/group.rs`) maps each `Filter` variant through the mirror, so an
// unmirrored kind cannot compile host-side.
const FILTER_COLOR: u32 = 0u;
const FILTER_CHROMATIC: u32 = 1u;

struct Filter {
    // Which filter this layer is (see the codes above).
    kind: u32,
    // How much of the filtered result lands, mixed against the untouched backdrop:
    // the layer's own opacity (§21.4). At 0 this pass is the identity — which the
    // draw list already spends nothing on, since a layer at zero opacity never
    // reaches the compositor at all.
    strength: f32,
    // The chromatic filter's dispersion: the displacement from where the red end of
    // the spectrum lands to where the blue end does, in **accumulator texels**
    // (§21.10). Derived by the encoder each frame from the filter's own two numbers
    // (a spread in canvas px and a canvas-space angle, `params.xy`) taken through
    // the view's canvas→screen linear map — the fringes belong to the artwork, so
    // they must turn, mirror and scale with it rather than with the window, which
    // is the same statement `media_common`'s `surf_m` makes about the weave. View
    // state, never document state; zero for every other kind.
    disp: vec2<f32>,
    // The filter's own numbers, read according to `kind`. For `FILTER_COLOR`
    // (§21.5): `.x` exposure in stops, `.y` contrast, `.z` saturation, `.w` hue
    // rotation in radians. For `FILTER_CHROMATIC` (§21.10): `.x` spread in canvas
    // px, `.y` angle in radians — the pair `disp` above is derived from, carried so
    // the uniform states the document's numbers as well as the view's.
    //
    // One lane rather than a member per knob, because what a lane *means* is the
    // filter's to say: the next kind reads these four floats as its own four, and
    // neither the host struct nor the bind group layout has to learn about it.
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> f: Filter;
// The accumulator: everything composited beneath this layer in its own stack.
@group(0) @binding(1) var back_color: texture_2d<f32>;
@group(0) @binding(2) var back_aux: texture_2d<f32>;
// Bilinear, clamp-to-edge — how the chromatic gather (§21.10) reads the pair
// above between texels. On **binding 3**, one of the two slots the blend pass
// spends on the isolated source this pass does not have; the point filters keep
// their exact `textureLoad`s and never touch it.
@group(0) @binding(3) var back_samp: sampler;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// Oklab `L` of mid-grey (sRGB 0.5), which the contrast gain pivots about. Mirrored
// by `document::CONTRAST_PIVOT`, whose unit test derives it rather than trusting it.
//
// A fixed pivot rather than the picture's own mean: a pivot that depends on what is
// underneath would make the slider do something different every time a layer below
// it changed, which is not what a contrast control is.
const CONTRAST_PIVOT: f32 = 0.5981;

// The colour adjustment (§21.5), in Oklab.
//
// In and out in Oklab so the colour spaces share it — and so the space the knobs
// are defined in is the interface, rather than a light interface whose first and
// last steps here would cancel a caller's own conversions.
fn adjust_color(lab_in: vec3<f32>, p: vec4<f32>) -> vec3<f32> {
    // Exposure, in stops, on the light itself rather than on `L` — see the header.
    // The trip stops at linear sRGB because a scalar gain commutes with the fixed
    // matrix between it and XYZ. `max(…, 0)` because the accumulator can hold a
    // Drago overshoot or an out-of-gamut mixture whose linear form dips below zero,
    // and Oklab's signed cube root would happily carry that through into a hue no
    // colour has.
    let exposed = max(oklab_to_linear_srgb(lab_in) * exp2(p.x), vec3<f32>(0.0));
    var lab = linear_srgb_to_oklab(exposed);

    // Contrast: a gain on lightness about mid-grey.
    lab.x = CONTRAST_PIVOT + (lab.x - CONTRAST_PIVOT) * p.y;

    // Hue, then chroma — one rotation and one scale of the same `(a, b)` plane, so
    // the order between them cannot matter, and doing the rotation first keeps the
    // scale reading as "how far from grey" rather than "how far from grey, turned".
    let c = cos(p.w);
    let s = sin(p.w);
    let ab = vec2<f32>(lab.y * c - lab.z * s, lab.y * s + lab.z * c) * p.z;
    lab = vec3<f32>(lab.x, ab.x, ab.y);

    return lab;
}

// One texel's worth of a **point** filter: the backdrop in Oklab, the filtered
// Oklab out.
//
// The dispatch on `kind` is here rather than in each space's `fs_main` so that
// adding a point filter is adding an arm to *one* function — the same reason
// `combine` holds all four blend modes rather than one per shader. A filter that
// reads *neighbouring* texels cannot come through here — one Oklab in is not
// enough to say where its light went — so a resampling kind (`FILTER_CHROMATIC`,
// §21.10) branches in each space's `fs_main` instead, where the space's own
// decode is in reach per tap.
fn filtered(lab: vec3<f32>) -> vec3<f32> {
    // One arm, so this is a `return` and not a chain — but it is written as a guard
    // rather than as the body, because the next filter is an arm above it and the
    // fallthrough has to be somebody's.
    if f.kind == FILTER_COLOR {
        return adjust_color(lab, f.params);
    }
    return lab;
}

// How much of the adjustment lands: the layer's opacity, as a mix from what was
// there to what the filter made of it (§21.4).
//
// It runs in the **working space** rather than in light, and that is what makes
// strength 0 exactly the identity rather than the identity plus a round trip's
// rounding: at `strength = 0` the result is `was` itself, bit for bit, whatever the
// filter did with it. A pigment document weighs its residual with the same call, on
// the same number, for the reason `merge_triple` exists (§6.7) — a residual is not a
// second colour to fade but the rest of the same one.
fn weigh(was: vec3<f32>, now: vec3<f32>) -> vec3<f32> {
    return mix(was, now, f.strength);
}

// The filtered channels premultiplied back, with the coverage and the height put
// back untouched.
//
// `chan` is un-premultiplied and `cb` is the raw sample, so the premultiply is by
// `cb.a` — the same coverage that came in, because a filter changes what colour the
// paint is and not how much of it there is.
fn resolve(cb: vec4<f32>, chan: vec3<f32>, out_chan: vec3<f32>, h: f32) -> FsOut {
    var out: FsOut;
    out.color = vec4<f32>(weigh(chan, out_chan) * cb.a, cb.a);
    out.aux = vec4<f32>(h, 0.0, 0.0, 0.0);
    return out;
}

// —— chromatic aberration (§21.10) ——————————————————————————————————————————
//
// Not three shifted copies of the picture but the integral they approximate: every
// wavelength of the light leaving the painting is displaced by its own refraction,
// and what a channel then holds is that continuum weighed by the eye's response,
//
//     out(x) = ∫ w(λ) · image(x − d(λ)) dλ,   d(λ) along `f.disp`.
//
// The pieces below say what `d` and `w` are; each colour space's `fs_main` owns the
// gather loop itself, because a tap has to go through that space's own decode to
// *be* light (§21.10). Three properties are load-bearing:
//
//   · **The taps are uniform in displacement, not in wavelength.** Dispersion in
//     glass is Cauchy's `n(λ) ≈ A + B/λ²`, so a wavelength's displacement is linear
//     in `1/λ²` and the blue end of the spectrum spreads farther than an equal run
//     of the red end — sampling evenly along the fringe and asking `ca_lambda`
//     which wavelength landed there puts the taps where the picture is, and gets
//     the asymmetry of a real lens for free.
//   · **The weights form an exact partition of unity.** Each channel of the gather
//     is divided by that channel's own summed weight, so wherever the image is
//     locally flat the pass returns it exactly — for any tap count, with no
//     normalization constant to tune (and none to get wrong).
//   · **The integral runs in linear light.** A sum over wavelengths is a sum of
//     light; summing Oklab or concentrations would mean nothing. Each space decodes
//     a tap to linear sRGB, accumulates, and re-encodes once at the end.

// The wavelengths bounding the integral, in nm: where the eye's responses still
// carry meaningful weight. Wider buys nothing — the weights outside are ~0, so a
// tap out there is a tap wasted.
const CA_LAMBDA_RED: f32 = 680.0;
const CA_LAMBDA_BLUE: f32 = 420.0;
// `(CA_LAMBDA_RED / CA_LAMBDA_BLUE)² − 1`: the span of the Cauchy dispersion term
// across the range above, which `ca_lambda` inverts.
const CA_CAUCHY_SPAN: f32 = 1.6213152;

// How many taps the gather spends: enough that neighbouring taps land within a
// texel of each other (bilinear filtering closes the gap), floored so even a
// sub-texel dispersion still integrates over a few spectral bands rather than
// collapsing to a hue cast, and capped so an extreme spread at an extreme zoom
// stays one affordable pass — past the cap the taps spread apart and the spectrum
// quantizes gently, which the bound on the document-side knob keeps rare (§21.10).
fn ca_taps(disp: vec2<f32>) -> u32 {
    return clamp(u32(ceil(length(disp))) + 3u, 4u, 64u);
}

// Tap `i` of `n` as the dispersion parameter `s ∈ [0, 1]`: 0 the red end, 1 the
// blue, uniform in displacement (see the header).
fn ca_s(i: u32, n: u32) -> f32 {
    return f32(i) / f32(n - 1u);
}

// The wavelength whose refraction lands at parameter `s` — Cauchy inverted:
// `(λ_red/λ)²` runs linearly from 1 at `s = 0` to `(λ_red/λ_blue)²` at `s = 1`.
fn ca_lambda(s: f32) -> f32 {
    return CA_LAMBDA_RED * inverseSqrt(1.0 + s * CA_CAUCHY_SPAN);
}

// One lobe of the CIE fit below: a piecewise Gaussian, its two flanks falling at
// their own rates — the shape of the measured response curves.
fn ca_lobe(x: f32, mu: f32, s1: f32, s2: f32) -> f32 {
    let t = (x - mu) / select(s2, s1, x < mu);
    return exp(-0.5 * t * t);
}

// The eye's response to wavelength `ca_lambda(s)`, as linear sRGB: the CIE 1931
// colour-matching functions (the Wyman–Sloan–Shirley multi-lobe analytic fit),
// normalized to the D65 white the light space is stated against and taken through
// the shared XYZ→sRGB matrix. Clamped at zero — a monochromatic colour sits outside
// the sRGB gamut, and a negative weight would ring where a fringe should glow. The
// clamp shifts each channel's total, and the per-channel normalization in the
// gather is what absorbs that without a constant to tune.
fn ca_weight(s: f32) -> vec3<f32> {
    let l = ca_lambda(s);
    let x = 1.056 * ca_lobe(l, 599.8, 37.9, 31.0)
        + 0.362 * ca_lobe(l, 442.0, 16.0, 26.7)
        - 0.065 * ca_lobe(l, 501.1, 20.4, 26.2);
    let y = 0.821 * ca_lobe(l, 568.8, 46.9, 40.5)
        + 0.286 * ca_lobe(l, 530.9, 16.3, 31.1);
    let z = 1.217 * ca_lobe(l, 437.0, 11.8, 36.0)
        + 0.681 * ca_lobe(l, 459.0, 26.0, 13.8);
    let light = vec3<f32>(x, y, z) / vec3<f32>(0.9505, 1.0, 1.089);
    return max(light_to_linear(light), vec3<f32>(0.0));
}

// The scalar weight a tap's **coverage and height** ride on: the luminance of its
// spectral weight, so the amount of paint travels with the light that shows it.
// Both must travel (§21.10): the media pass makes paint visible only where coverage
// *and* height exist, so a fringe displaced past a stroke's edge with neither would
// be a fringe that cannot be seen.
fn ca_lum(w: vec3<f32>) -> f32 {
    return dot(w, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Where tap `s` samples, as the UV `back_samp` takes: the fragment *minus* the
// displacement the wavelength's image was carried by, red end at `−disp/2`, blue at
// `+disp/2` — a symmetric spread, so the picture stays put and the fringes part
// around it. (Gathering is the transpose of scattering: blue light shifted toward
// `+disp` is found by looking back along it.)
fn ca_uv(frag: vec2<f32>, s: f32, dim: vec2<f32>) -> vec2<f32> {
    return (frag - f.disp * (s - 0.5)) / dim;
}
