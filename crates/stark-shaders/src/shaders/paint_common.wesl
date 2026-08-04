// The parcel-stacking law shared by every pass that restacks *existing* paint
// (§6.1): the brush-dynamics deposit (dynamics.wesl) and the transform's
// combine (transform.wesl). One definition so the two cannot drift — a parcel must
// land the same way whether a brush carried it or a transform did.

const EPS: f32 = 1e-5;

// Matches the media pass's translucent-slab law `1 − exp(−K·opacity·thickness)`
// (media_common.wesl), so a parcel's covering behaviour agrees with what the
// viewer sees: its blend weight is its visible alpha.
const OPACITY_K: f32 = 1.0;

// Blend a parcel of paint (latent `lat1`, optical mass `m1` = opacity·height) onto
// resident paint (`lat0`, `m0`). The parcel lands on top, so its weight is its own
// visible alpha and the resident paint shows through the remainder — thick deposits
// cover, thin glazes tint, and for small masses this reduces to mass-weighted
// mixing. Falls back to the resident latent when both masses vanish.
fn blend_latent(lat0: vec3<f32>, m0: f32, lat1: vec3<f32>, m1: f32) -> vec3<f32> {
    let w1 = 1.0 - exp(-OPACITY_K * m1);
    let w0 = (1.0 - w1) * (1.0 - exp(-OPACITY_K * m0));
    let w = w0 + w1;
    if w <= EPS {
        return lat0;
    }
    return (lat0 * w0 + lat1 * w1) / w;
}

// --- the deposition tooth (§6.4) -------------------------------------------------
//
// Paint lands where the tip touches the ground, and on a rough ground the tip
// touches the peaks before the valleys. `tooth_gate` is that contact, per texel: the
// fraction of the brush's own paint (`add`) this texel receives, given the canvas
// surface's height `s` there and how deep into the weave the tool reaches.
//
// **The model is the substrate's bearing-area curve** (Abbott–Firestone): the
// fraction of a rough surface standing above a given level. The tip presses to a
// level `θ`, and a texel takes paint where the ground clears it. Per texel this is
// one *sample* of that curve, so the mean over a footprint is the true contact
// fraction — a prediction that can be checked against the height map's own histogram
// (tests/tooth.rs) rather than a curve tuned until it looked right, which is what
// §6.4 asked for before this could return.
//
// **It scales the exposure, not the transfer**, and that one choice is what makes it
// compose and what makes it conserve. Every rate in this engine is a function of
// swept optical depth τ — additive in it, or `1 − exp(−k·τ)` — because those are the
// two forms independent of how the path was cut (§6.2). Multiply τ itself by a
// per-texel `g` and every one of them stays in its form: `exp(−k·g·τ₁)·exp(−k·g·τ₂) =
// exp(−k·g·(τ₁+τ₂))`, and `Σᵢ g·τᵢ = g·Στᵢ`, because `g` belongs to the *canvas* and
// not to the segment. Applied to the finished shares instead — `g·(1 − exp(−k·τ))` —
// none of that holds, and a toothed lift would fade at a rate that depended on where
// the flattener happened to cut.
//
// The same choice is what closes the books. The exchange's four shares are exact
// complements of one exposure (`dynamics.wesl::Exchange`), so scaling the exposure
// keeps them complementary and the canvas takes exactly what the tool gives up. The
// two sides evaluate `g` differently — the canvas per texel, the tool against the
// ground's mean bearing (`Surface::bearing`), since a reservoir cell has no ground of
// its own — and they agree in expectation over any footprint spanning many grain
// features. That residual is the same order as the mean-field freeze either side of
// the kernel already carries.
//
// It is a pure function of **canvas position**, so a tile's apron is still
// bit-identical to its neighbour's interior with no copy pass (§6.4), and — the
// reason the mark reads as paper rather than as noise — successive strokes catch on
// the same peaks and register with each other.
//
// It gates **height, never the per-unit opacity** (§6.1): the tooth decides how much
// paint arrives, not what the pigment is.

// Width of the contact transition, in the height map's own [0, 1] units.
//
// A hard threshold is a binary indicator per texel: correct in the mean, and at
// canvas resolution it aliases into hard-edged speckle that reads as dither rather
// than as tooth. Too wide and the bearing curve is smeared into a flat grey and the
// grain stops reading at all. 0.15 is a little under a quarter of the useful range.
// **Must match `TOOTH_SOFTNESS` in `gpu::surface`.**
const TOOTH_SOFTNESS: f32 = 0.15;

// The ground height at a canvas position, sampled **nearest** rather than bilinear.
//
// Not a shortcut — bilinear would be wrong twice over. The map is minified (about two
// texels per canvas px at `SURFACE_TILE_PX`), so filtering averages the peaks the
// tooth exists to catch on and washes the grain toward its mean. And the distribution
// the shader draws from has to be the map's own histogram, because that histogram is
// what the CPU integrates to get the bearing fraction the tool books against; a
// filtered tap draws from a narrower distribution and the two halves of the transfer
// stop agreeing. Nearest also sidesteps the reduced-precision filter weights
// `stamp_common.wesl::prefix_slice` documents, so the tap is bit-reproducible.
//
// `dims` is the map's size; the modulo is the Repeat the sampler would have done.
fn ground_at(uv: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    let d = vec2<f32>(dims);
    return vec2<i32>(fract(uv - floor(uv)) * d);
}

// The contact level the tip presses to, from the `tooth` knob.
//
// Mapped so both ends are **exact** rather than nearly so. `tooth = 0` puts the whole
// transition band below `s = 0`, so every texel is in full contact and the gate is
// 1.0 to the bit — a brush with no tooth deposits precisely what it always did, and
// the same is true of a `Flat` canvas, whose `relief` of 0 zeroes the knob before it
// gets here. `tooth = 1` puts the band above `s = 1`, so only the very top of the
// height map's range gets anything.
fn tooth_level(tooth: f32) -> f32 {
    return tooth * (1.0 + TOOTH_SOFTNESS) - 0.5 * TOOTH_SOFTNESS;
}

// The fraction of its paint a texel receives, given the ground height `s ∈ [0, 1]`
// there. A cubic smoothstep, for the reason `taper_profile` is a polynomial: this
// decides stored pixels, and replay, goldens and peers have to agree on it to the
// last bit (§12.1).
//
// **Mirrored on the CPU** in `gpu::surface`, which averages it over a height map's
// histogram to get the bearing fraction the tool books against. The two must be the
// same function or the two halves of a toothed transfer disagree; the conservation
// pair in `tests/dynamics.rs` is what notices if they drift.
fn tooth_gate(s: f32, tooth: f32) -> f32 {
    let t = clamp((s - tooth_level(tooth)) / TOOTH_SOFTNESS + 0.5, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
