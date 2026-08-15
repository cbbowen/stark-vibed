// The parcel-stacking law shared by every pass that restacks *existing* paint
// (§6.1): the brush-dynamics deposit (dynamics.wesl) and the transform's
// combine (transform.wesl). One definition so the two cannot drift — a parcel must
// land the same way whether a brush carried it or a transform did.

// The two guards, named for **which representation** they guard, because they used
// not to be.
//
// Both were spelled `EPS`, in five modules, at two different values: 1e-5 in
// `paint_common`, `media_common`, `composite` and `resolve`, and 1e-4 in
// `blend_common` — so `fill.wesl` importing `EPS` from one and `blend_oklab.wesl`
// importing `EPS` from the other got different numbers under one name, with nothing
// at either call site to say so. The values below are unchanged; only the names are
// new, and they are chosen so the call site says what it is protecting.
//
// The distinction is real and not a tolerance to be split. A **tile** carries
// per-unit opacity in alpha and height in aux (§6.1); a **composited target**
// carries the stack's coverage in alpha (pass A, `composite.wesl`). Un-premultiplying
// the two divides by different quantities on different scales, which is why they
// were always two numbers even while they shared a name.

// The tile/paint representation: the mass or height below which there is no paint
// to speak of, and the per-unit opacity below which a latent cannot be recovered.
// One constant because all three are the same representation at the same scale.
const PAINT_EPS: f32 = 1e-5;

// Un-premultiplying a **composited** target by its coverage (`blend_*.wesl`,
// `media_*.wesl`). Coarser than `PAINT_EPS` deliberately: coverage saturates towards
// 1 and the interesting failure is a fragment with almost none of it, where the
// quotient explodes.
const COVERAGE_EPS: f32 = 1e-4;

// The translucent-slab law's build-up rate: `1 − exp(−K·opacity·thickness)`.
//
// **The one definition.** It was written out four times — here, `composite.wesl`,
// `matte.wesl` and `media_common.wesl` — three of them carrying a "must match"
// comment at the very constant a shared import was already available for
// (`stamp.wesl` imported this one). The `media_common` copy had by then
// gone dead outright: its own comment said it "remains only for the weave-sink
// correction", and that correction is a power law with no `K` in it.
//
// Every place the slab law is evaluated now reads this: the parcel stacking below,
// pass A's per-layer coverage (`composite.wesl`), the matte slab (`matte.wesl`),
// and the stamp's per-segment visible alpha (`stamp.wesl`). That a parcel's
// blend weight *is* its visible alpha — that it covers resident paint exactly as
// much as the viewer sees it cover — is the property those four agreeing is worth.
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
    if w <= PAINT_EPS {
        return lat0;
    }
    return (lat0 * w0 + lat1 * w1) / w;
}

// --- the deposition tooth (§6.4) -------------------------------------------------
//
// Paint lands where the tip touches the ground, and what a *dragged* tip touches is
// not a level set of the ground's height — it is decided by the ground's **slope
// along the travel**. A moving tip has give: it sinks after ground that falls away
// beneath it and is pressed up by ground that rises to meet it, so it bears on the
// near face of every bump and bridges the lee side behind it. That is why a dry
// brush prints the leading edges of the grain rather than a speckle of its summits,
// and it is why a height threshold — which reads the same mark whichever way the
// stroke runs — was the wrong field to gate on. `tooth_gate` is that contact, per
// texel: the fraction of its paint a texel receives, given the **rise ahead** of the
// tip there (`rise_ahead` — the height's derivative along the travel, taken across
// the contact's own reach) and how much give the tool has left.
//
// **`tooth` is the give, inverted** (`tooth_level`). The gate thresholds the rise
// against the steepest fall the tip can still follow, and the knob walks that limit
// from infinite (tooth = 0: the tip tracks any fall — the surface is ignored,
// exactly) through zero (tooth = ½: the tip holds its level, touching whatever is
// flat or rising and bridging every fall) to *demanding* ground that rises at the
// contact scale (tooth = 1: only the leading faces of the grain print — the dry
// mark, still a mark). The field is directional outright: run the same stroke
// backwards and the rise negates, so the mark moves to the other side of every
// feature (tests/tooth.rs).
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
// It is a pure function of **canvas position and the tip's travel there**, and the
// second half changes nothing structural: travel is recovered from the segment's own
// frame at the fragment's canvas position (`stamp_common.wesl::sweep_at`), so within
// one pass it is still position that decides everything. A tile's apron stays
// bit-identical to its neighbour's interior with no copy pass (§6.4); the gate still
// factors out of the sum over a segment's overlapping quads, so `preview == committed`
// survives; and — the reason the mark reads as paper rather than as noise —
// successive strokes over the same ground in the same direction catch on the same
// faces and register with each other. What is deliberately *not* invariant is the
// stroke run the other way: that is the physics, not a leak.
//
// It gates **height, never the per-unit opacity** (§6.1): the tooth decides how much
// paint arrives, not what the pigment is.

// Width of the contact transition, in the rise's own units — height per reach of
// travel.
//
// A hard threshold is a binary indicator per texel: correct in the mean, and at
// canvas resolution it aliases into hard-edged speckle that reads as dither rather
// than as tooth. Too wide and the faces are smeared into a flat grey and the grain
// stops reading at all. 0.06 is the bundled grounds' own interquartile rise, so the
// transition spans the grain's natural variation.
// **Must match `TOOTH_SOFTNESS` in `gpu::surface`.**
const TOOTH_SOFTNESS: f32 = 0.06;

// The contact scale: the rise, per reach of travel, that a full-tooth tip demands of
// the ground before it presses — and the unit `tooth_level`'s follow limit is
// measured in. Measured against the bundled grounds (mean |rise| 0.037–0.090), not
// picked; see `gpu::surface::TOOTH_RISE`, which it **must match**.
const TOOTH_RISE: f32 = 0.05;

// The half-range the rise bytes span, as a fraction of the height range — the scale
// `gpu::surface::encode_rise` quantizes against, and which `rise_ahead` below has to
// decode with. The host reads this declaration rather than keeping a copy (§6.10),
// which is why it is named here instead of folded into `rise_ahead`'s arithmetic as
// the literals `255.0 / 512.0` and `0.25`. Naming it costs nothing: both spellings
// fold to exactly 0.498046875 and 0.25, and 0.25 is a power of two, so the arithmetic
// is identical to the bit.
//
// Real rises are small, so spending the byte on a narrow range is what keeps the
// tooth's transition tonal rather than stepped.
const RISE_LIMIT: f32 = 0.25;

// The ground texel at a canvas position, sampled **nearest** rather than bilinear.
//
// Not a shortcut — bilinear would be wrong twice over. The map is minified (about two
// texels per canvas px at `SURFACE_TILE_PX`), so filtering averages the faces the
// tooth exists to catch on and washes the grain toward its mean. And the distribution
// the shader draws from has to be the map's own, because that distribution is what the
// CPU integrates to get the bearing fraction the tool books against; a filtered tap
// draws from a narrower one and the two halves of the transfer stop agreeing. Nearest
// also sidesteps the reduced-precision filter weights
// `stamp_common.wesl::prefix_slice` documents, so the tap is bit-reproducible.
//
// The one tap serves the whole gate: the ground texture carries the rise ahead in
// `gb` (its `r` is the height, which is the media pass's business, §6.3), so the
// texel the gate thresholds is the map's own stored byte — which is what lets the
// CPU tabulate the same distribution, byte for byte, and keep booking the tool's
// half against the real thing.
//
// `dims` is the map's size; the modulo is the Repeat the sampler would have done.
fn ground_at(uv: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    let d = vec2<f32>(dims);
    return vec2<i32>(fract(uv - floor(uv)) * d);
}

// The **rise ahead** at a ground texel `g`, under a tip whose travel here is the
// unit tangent `travel`: how much higher the ground the tip is about to meet stands
// than the ground here, `ahead·d̂` — the height's derivative along the travel, taken
// across the contact's own reach.
//
// `ahead` is baked into the map (`gpu::surface::pack_ground`) — how much higher the
// ground stands one `TOOTH_REACH` further on along each canvas axis, filtered at the
// scale contact happens at and already scaled to canvas px, so nothing here depends on
// how finely the weave was stored. The bytes span ±`RISE_LIMIT` = 0.25 of the height
// range (real rises are small; spending the byte there is what keeps the transition
// tonal rather than stepped — `gpu::surface::RISE_LIMIT`, which the folded constants
// here **must match**). All that is left at draw time is the projection onto the
// direction of travel, and that projection is the whole model: positive on a bump's
// near face, where the tip bears early; negative past the summit, where it bridges;
// negated outright when the same stroke runs the other way.
//
// **Mirrored on the CPU** in `gpu::surface::rise_ahead`, decode constants included,
// because the tool's half of the transfer is booked against this quantity's
// distribution.
fn rise_ahead(g: vec4<f32>, travel: vec2<f32>) -> f32 {
    // `byte·(L/128) − L`, with the `byte/255` the sampler already applied folded back
    // out — the exact inverse of `gpu::surface::encode_rise`'s `128 + 128·rise/L`.
    let ahead = g.gb * (255.0 * RISE_LIMIT / 128.0) - vec2<f32>(RISE_LIMIT, RISE_LIMIT);
    return dot(ahead, travel);
}

// The steepest fall the tip can still follow, negated — the level the gate
// thresholds the rise against, from the `tooth` knob.
//
// `TOOTH_RISE · (2 − 1/tooth)`, and the three stations of that map are the model:
// −∞ at `tooth = 0` (infinite give — the transition band runs out below any rise the
// map can encode, so the knob's bottom meets the callers' `tooth <= 0` guard
// *continuously*: a pen mapping sweeping through zero cannot pop), 0 at `tooth = ½`
// (no give left going down: the tip holds its level, touches flat and rising ground,
// bridges every fall), +`TOOTH_RISE` at `tooth = 1` (contact demands a rise at the
// contact scale — the leading faces alone). A `Flat` canvas never gets here: its
// `relief` of 0 zeroes the uv scale upstream and the gate is exactly 1.
//
// The floor under the division is inert, not a tuning: every level below −0.39 is
// already past the steepest fall the encoding can express plus the whole transition
// band, and the floor's level is −3.05 — it exists so the arithmetic stays finite,
// which WGSL does not promise for a true 1/0.
fn tooth_level(tooth: f32) -> f32 {
    return TOOTH_RISE * (2.0 - 1.0 / max(tooth, 0.01));
}

// The fraction of its paint a texel receives, given the rise `d` of the ground along
// the tip's travel there (`rise_ahead`). A cubic smoothstep, for the reason
// `taper_profile` is a polynomial: this decides stored pixels, and replay, goldens
// and peers have to agree on it to the last bit (§12.1).
//
// **Mirrored on the CPU** in `gpu::surface`, which averages it over the map's rise
// distribution to get the bearing fraction the tool books against. The two must be
// the same function or the two halves of a toothed transfer disagree; the
// conservation pair in `tests/dynamics.rs` is what notices if they drift.
fn tooth_gate(d: f32, tooth: f32) -> f32 {
    let t = clamp((d - tooth_level(tooth)) / TOOTH_SOFTNESS + 0.5, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
