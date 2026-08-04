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
// Two things follow from where it is applied rather than from what it computes, and
// both are why it goes here rather than onto a coverage:
//
//   * It multiplies **swept optical depth τ**, in the slot `stroke_drain` occupies.
//     `add·g·τ` is still additive in τ and `1 − exp(−K·op·add·g·τ)` is still of the
//     permitted form (§6.2), and `g` does not depend on the segment — so
//     `Σᵢ g·τᵢ = g·Στᵢ` and the gate is exactly independent of how the path was cut.
//   * It is a pure function of **canvas position**, so a tile's apron is still
//     bit-identical to its neighbour's interior with no copy pass (§6.4).
//
// It gates **height, never the per-unit opacity** (§6.1): the tooth decides how much
// paint arrives, not what the pigment is. And it gates only the brush's own `add`
// source — the tool's `deposit` is one half of a transfer whose other half already
// happened, so gating it would destroy paint the tool has given up.

// Width of the contact transition, in the height map's own [0, 1] units.
//
// A hard threshold is a binary indicator per texel: correct in the mean, and at
// canvas resolution it aliases into hard-edged speckle that reads as dither rather
// than as tooth. Too wide and the bearing curve is smeared into a flat grey and the
// grain stops reading at all. 0.15 is a little under a quarter of the useful range.
const TOOTH_SOFTNESS: f32 = 0.15;

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
fn tooth_gate(s: f32, tooth: f32) -> f32 {
    let t = clamp((s - tooth_level(tooth)) / TOOTH_SOFTNESS + 0.5, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
