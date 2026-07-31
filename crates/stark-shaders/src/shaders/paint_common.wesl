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
