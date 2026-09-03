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
// gone dead outright: its own comment said it "remains only for the substrate-sink
// correction", and that correction is a power law with no `K` in it.
//
// Every place the slab law is evaluated now reads this: the parcel stacking below,
// pass A's per-layer coverage (`composite.wesl`), the matte slab (`matte.wesl`),
// and the stamp's per-segment visible alpha (`stamp.wesl`). That a parcel's
// blend weight *is* its visible alpha — that it covers resident paint exactly as
// much as the viewer sees it cover — is the property those four agreeing is worth.
const OPACITY_K: f32 = 1.0;

// Where "opaque" is, as optical mass — the finite number that stands for the end of a
// coverage slider.
//
// Coverage is `1 − exp(−K·mass)` and so reaches 1 only in the limit, which means
// anything asking for *full* coverage has to name a mass rather than inherit one: the
// inversion `m = −ln(1 − w)/K` runs to infinity at `w = 1`. Eight units leaves under a
// thousandth of what is beneath showing, which is opaque by any reading of it.
//
// Beside `OPACITY_K` because it is the same law's other end, and shared because it is
// asked from both sides of this crate: `fill.wesl` inverts the law here (§18.0.4), and
// the host builds an imported image's tiles by the same arithmetic with no shader at
// all (§23) — a duplicate there would be a second declaration of "opaque" that nothing
// checks, in Rust, where the WESL could not even be read for comparison.
const OPAQUE_MASS: f32 = 8.0;

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
// Paint lands where the tip touches the substrate, and what a *dragged* tip touches is
// not a level set of the substrate's height — it is decided by the substrate's **slope
// along the travel**. A moving tip has give: it sinks after substrate that falls away
// beneath it and is pressed up by substrate that rises to meet it, so it bears on the
// near face of every bump and bridges the lee side behind it. That is why a dry
// brush prints the leading edges of the grain rather than a speckle of its summits,
// and it is why a height threshold — which reads the same mark whichever way the
// stroke runs — was the wrong field to gate on. `tooth_gate` is that contact, per
// texel: the fraction of its paint a texel receives, given the **rise ahead** of the
// tip there (`rise_ahead` — the height's derivative along the travel, taken across
// the contact's own reach) and how much give the tool has left.
//
// **The knob `tooth_level` reads *is* that give** (`ToothParams::give`). The
// gate thresholds the rise against the steepest fall the tip can still follow, and the
// knob walks that limit from *demanding* substrate that rises at the contact scale
// (give = 0: only the leading faces of the grain print — the dry mark, still a mark)
// through zero (give = ½: the tip holds its level, touching whatever is flat or rising
// and bridging every fall) to infinite (give = 1: the tip tracks any fall — the
// substrate is ignored, exactly, which is the default). Quoted as the give and not as
// its inverse because a modulation only ever scales *down*, and the mapping this axis
// exists for is pressure: a tip barely touching the paper has no give and prints the
// peaks alone, and one borne down has all of it. The field is directional outright:
// run the same stroke backwards and the rise negates, so the mark moves to the other
// side of every feature (tests/tooth.rs).
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
// substrate's mean bearing (`Surface::bearing`), since a reservoir cell has no substrate of
// its own — and they agree in expectation over any extent spanning many grain
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
// successive strokes over the same substrate in the same direction catch on the same
// faces and register with each other. What is deliberately *not* invariant is the
// stroke run the other way: that is the physics, not a leak.
//
// It gates **height, never the per-unit opacity** (§6.1): the tooth decides how much
// paint arrives, not what the pigment is.

// The narrowest contact transition that is still a *number* rather than a division by
// zero — the floor under `tooth_gate`'s band.
//
// Inert, not a tuning, in exactly `tooth_level`'s sense: the rise arrives decoded off
// an 8-bit lattice whose step is `2·RISE_LIMIT/255 ≈ 0.002`, so any band an order of
// magnitude under that already gates every texel the same way a hard threshold would.
// A brush asking for 0 softness gets the hard threshold it asked for; what it does not
// get is the `0/0` at the one rise that sits exactly on the limit, which WGSL does not
// promise anything about.
const TOOTH_SOFTNESS_FLOOR: f32 = 1e-4;

// The contact scale: the rise, per reach of travel, that a tip with no give demands of
// the substrate before it presses — and the unit `tooth_level`'s follow limit is
// measured in. Measured against the bundled substrates (mean |rise| 0.037–0.090), not
// picked; see `gpu::substrate::TOOTH_RISE`, which it **must match**.
const TOOTH_RISE: f32 = 0.05;

// The half-range the rise bytes span, as a fraction of the height range — the scale
// `gpu::substrate::encode_rise` quantizes against, and which `rise_ahead` below has to
// decode with. The host reads this declaration rather than keeping a copy (§6.10),
// which is why it is named here instead of folded into `rise_ahead`'s arithmetic as
// the literals `255.0 / 512.0` and `0.25`. Naming it costs nothing: both spellings
// fold to exactly 0.498046875 and 0.25, and 0.25 is a power of two, so the arithmetic
// is identical to the bit.
//
// Real rises are small, so spending the byte on a narrow range is what keeps the
// tooth's transition tonal rather than stepped.
const RISE_LIMIT: f32 = 0.25;

// The substrate texel at a canvas position, sampled **nearest** rather than bilinear.
//
// Not a shortcut — bilinear would be wrong twice over. The map is minified (about two
// texels per canvas px at `SUBSTRATE_TILE_PX`), so filtering averages the faces the
// tooth exists to catch on and washes the grain toward its mean. And the distribution
// the shader draws from has to be the map's own, because that distribution is what the
// CPU integrates to get the bearing fraction the tool books against; a filtered tap
// draws from a narrower one and the two halves of the transfer stop agreeing. Nearest
// also sidesteps the reduced-precision filter weights
// `stamp_common.wesl::prefix_slice` documents, so the tap is bit-reproducible.
//
// The one tap serves the whole gate: the substrate texture carries the rise ahead in
// `gb` (its `r` is the height, which is the media pass's business, §6.3), so the
// texel the gate thresholds is the map's own stored byte — which is what lets the
// CPU tabulate the same distribution, byte for byte, and keep booking the tool's
// half against the real thing.
//
// `dims` is the map's size; the modulo is the Repeat the sampler would have done.
fn substrate_texel_at(uv: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    let d = vec2<f32>(dims);
    return vec2<i32>(fract(uv - floor(uv)) * d);
}

// The **rise ahead** at a substrate texel `g`, under a tip whose travel here is the
// unit tangent `travel`: how much higher the substrate the tip is about to meet stands
// than the substrate here, `ahead·d̂` — the height's derivative along the travel, taken
// across the contact's own reach.
//
// `ahead` is baked into the map (`gpu::substrate::pack_substrate`) — how much higher the
// substrate stands one `TOOTH_REACH` further on along each canvas axis, filtered at the
// scale contact happens at and already scaled to canvas px, so nothing here depends on
// how finely the substrate was stored. The bytes span ±`RISE_LIMIT` = 0.25 of the height
// range (real rises are small; spending the byte there is what keeps the transition
// tonal rather than stepped — `gpu::substrate::RISE_LIMIT`, which the folded constants
// here **must match**). All that is left at draw time is the projection onto the
// direction of travel, and that projection is the whole model: positive on a bump's
// near face, where the tip bears early; negative past the summit, where it bridges;
// negated outright when the same stroke runs the other way.
//
// **Mirrored on the CPU** in `gpu::substrate::rise_ahead`, decode constants included,
// because the tool's half of the transfer is booked against this quantity's
// distribution.
fn rise_ahead(g: vec4<f32>, travel: vec2<f32>) -> f32 {
    // `byte·(L/128) − L`, with the `byte/255` the sampler already applied folded back
    // out — the exact inverse of `gpu::substrate::encode_rise`'s `128 + 128·rise/L`.
    let ahead = g.gb * (255.0 * RISE_LIMIT / 128.0) - vec2<f32>(RISE_LIMIT, RISE_LIMIT);
    return dot(ahead, travel);
}

// The steepest fall the tip can still follow, negated — the level the gate
// thresholds the rise against, from the `tooth_give` knob.
//
// `TOOTH_RISE · (2 − 1/(1 − give))`, and the three stations of that map are the model:
// +`TOOTH_RISE` at `give = 0` (no give at all: contact demands a rise at the contact
// scale, so the leading faces alone print), 0 at `give = ½` (no give left going *down*:
// the tip holds its level, touches flat and rising substrate, bridges every fall),
// −∞ as `give → 1` (infinite give — the transition band runs out below any rise the
// map can encode, so the knob's top meets the callers' `tooth_give >= 1` guard
// *continuously*: a pen mapping sweeping through it cannot pop). A `Flat` canvas never
// gets here: its `relief` of 0 zeroes the uv scale upstream and the gate is exactly 1.
//
// The floor under the division is inert, not a tuning: every level below −0.39 is
// already past the steepest fall the encoding can express plus the whole transition
// band, and the floor's level is −3.05 — it exists so the arithmetic stays finite,
// which WGSL does not promise for a true 1/0.
fn tooth_level(give: f32) -> f32 {
    return TOOTH_RISE * (2.0 - 1.0 / max(1.0 - give, 0.01));
}

// The fraction of its paint a texel receives, given the rise `d` of the substrate along
// the tip's travel there (`rise_ahead`). A cubic smoothstep, for the reason
// `taper_profile` is a polynomial: this decides stored pixels, and replay, goldens
// and peers have to agree on it to the last bit (§12.1).
//
// **Two knobs, and they are different questions.** `give` is where the threshold
// sits — how steep a fall the tip still follows — and `softness` is how wide the band
// around it is: how gradually a texel goes from taking none of the tip's paint to
// taking all of it. A paint that sits on the substrate has a narrow band and prints
// the faces; a charcoal or a soft graphite crumbles into the valleys instead of
// spanning them, and its contact comes on over several times that width
// (`ToothParams::softness`, which is where the band's default lives — the host
// carries it, so this file declares no constant for it).
//
// **Mirrored on the CPU** in `gpu::substrate`, which averages it over the map's rise
// distribution to get the bearing fraction the tool books against. The two must be
// the same function or the two halves of a toothed transfer disagree; the
// conservation pair in `tests/dynamics.rs` is what notices if they drift.
fn tooth_gate(d: f32, give: f32, softness: f32) -> f32 {
    let t = clamp(
        (d - tooth_level(give)) / max(softness, TOOTH_SOFTNESS_FLOOR) + 0.5,
        0.0,
        1.0,
    );
    return t * t * (3.0 - 2.0 * t);
}

// ---- The ceiling under the pen (§6.2) --------------------------------------------
//
// When the pen drives a stroke's opacity there is no one ceiling to scale the
// parcel by, and what a texel should show is the **largest** ceiling any paint
// covering it asked for — at every point of the texel, then averaged over it. With
// `c(θ)` the coverage laid by segments whose ceiling is at least `θ`, that average
// is `∫₀¹ c(θ) dθ`, and `c(θ) = 1 − exp(−K·M(θ))` with `M(θ)` the mass laid at or
// above `θ`. Every `M(θ)` is a plain sum of per-segment masses gated on the
// segment's factor, so a blend can accumulate it — which is what makes a *max* law
// reachable by fixed-function blending at all, where a running max of anything is
// not.
//
// The lane samples `M` at two levels, `LEVEL_1` and `LEVEL_2`, and beside each the
// first moment of the mass above it; the total mass and total moment ride the aux.
// Between two levels the mass is placed at its own **mean** factor, which is exact
// wherever all the mass in a third sits at one factor — a pen that holds still, a
// mouse — and is a mean within a third otherwise. So a crossing whose two ceilings
// fall in different thirds of the dial shows the higher one exactly, and one whose
// two fall in the same third shows their mass-weighted mean, never more than a third
// of the dial from either.

// The two levels: thirds of the dial.
const LEVEL_1: f32 = 0.33333334;
const LEVEL_2: f32 = 0.6666667;

// One fragment's share of the lane: its mass above each level, and its moment
// above each — `(M₁, M₂, Q₁, Q₂)` increments for a factor `o` laying mass `m`.
fn level_sums(o: f32, m: f32) -> vec4<f32> {
    let g1 = step(LEVEL_1, o);
    let g2 = step(LEVEL_2, o);
    return vec4<f32>(m * g1, m * g2, o * m * g1, o * m * g2);
}

// The claimed coverage: the point-wise max of the ceilings, averaged over the
// texel, from the total mass `s0`, the total moment `s1` and the lane `lv` — each
// third's mass placed at its mean factor, and `∫ c(θ) dθ` summed across them. Every
// sum is clamped against its neighbour: they are f16 accumulations of the same
// increments, and a mass above a level can round a hair past the mass above the
// level below it.
fn claimed_coverage(s0: f32, s1: f32, lv: vec4<f32>) -> f32 {
    let m1 = min(lv.x, s0);
    let m2 = min(lv.y, m1);
    let q1 = min(lv.z, s1);
    let q2 = min(lv.w, q1);
    let c0 = 1.0 - exp(-OPACITY_K * s0);
    let c1 = 1.0 - exp(-OPACITY_K * m1);
    let c2 = 1.0 - exp(-OPACITY_K * m2);
    // Each third's mean factor, held to its third. A third with no mass has
    // `c` equal on both sides of it, so where its mean lands decides nothing.
    let mu1 = clamp((s1 - q1) / max(s0 - m1, PAINT_EPS), 0.0, LEVEL_1);
    let mu2 = clamp((q1 - q2) / max(m1 - m2, PAINT_EPS), LEVEL_1, LEVEL_2);
    let mu3 = clamp(q2 / max(m2, PAINT_EPS), LEVEL_2, 1.0);
    return mu1 * c0 + (mu2 - mu1) * c1 + (mu3 - mu2) * c2;
}

// The ceiling's factor where a point sits along one segment (§6.2): the segment's
// two **ends**, linearly across its travel — the mean the segment carries, and the
// difference between them.
//
// A ceiling read once per segment is piecewise constant, and a stroke drawn at a
// realistic report rate is a handful of segments wide, so what the eye sees is
// bands: `opacity` was the one pen target whose value stepped at every cut. This
// is `Sweep::ramp`'s answer for the radius, in the one form that differs —
// absolute rather than relative, opacity being a fraction rather than a scale, so
// nothing here has to defend a positive product.
//
// `x` is the point's travel from the segment's start and `lr` the segment's own,
// both in whatever units the caller's frame measures travel in. The clamp past
// either end is `ramp_scale`'s and buys the same thing: two adjacent segments
// computed their shared end from the same pen at the same arc length, so the
// overhanging caps agree there rather than each running off along its own slope.
// A segment with no travel at all — the pen-up settle — reads the mean.
fn ceiling_at(mean: f32, ramp: f32, x: f32, lr: f32) -> f32 {
    // Every segment of a brush whose ceiling the pen does not drive takes this
    // branch, and takes it as an exact identity.
    if ramp == 0.0 || lr <= 0.0 {
        return mean;
    }
    return mean + ramp * (clamp(x / lr, 0.0, 1.0) - 0.5);
}
