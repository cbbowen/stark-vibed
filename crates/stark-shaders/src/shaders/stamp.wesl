// Swept-segment brush rasterization (§6.2).
//
// **Color-space agnostic, and named accordingly.** This was `stamp_oklab.wesl`,
// which was never true: `colorspace.rs` hands it to the Mixbox space as well, since
// the deposit is the same premultiplied "over" whatever the three channels mean —
// the shader never interprets them. The name read as the surviving half of a
// `stamp_mixbox.wesl` pair that has never existed, which is a worse thing for a name
// to do than merely be long. (`blend_oklab`/`blend_mixbox` and
// `media_oklab`/`media_mixbox` are genuine pairs; those keep their suffixes.)
//
// Shares all geometry/sweep math with both spaces via `stamp_common.wesl`. The extent deposits into a *cleared scratch* tile, which
// holds the **parcel this stroke is laying** rather than a tile of paint: color
// premultiplied by the parcel's visible alpha, and the amounts that alpha was built
// from (height, optical mass) in the aux. `integrate.wesl` is what turns the one
// parcel that comes out into paint on the canvas, in the normalized representation
// (§6.1). Within a stroke the color target over-blends (premult-consistent) and the
// aux target accumulates additively.
//
// Every deposit here is a function of the segment's **swept optical depth** `τ`
// alone, in one of exactly two forms — additive in `τ`, or `1 − exp(−k·τ)` — because
// those are the two that compose *independently of how the path was cut into
// segments*. `τ` itself sums exactly across overlapping quads (that is the whole
// point of the prefix-τ construction), so an amount linear in it sums to the true
// path integral, and a rate exponential in it multiplies out to `exp(−k·Στ)`. Any
// other shape — a `√`, say — makes the result depend on the *number* of segments: N
// pieces of the same travel deposit `Σ√(τ/N) = √(Nτ)`, so the stroke silently gains
// weight with sampling density. That is fine only while sampling is uniform; it is
// why adaptive flattening (`path::flatten`) needs this form, and it is what makes the
// fast path agree with the stamp loop's `add` axis, which is linear in exposure
// (`dynamics.wesl::deposit`).
//
// **What a segment lays is a parcel of paint, not a coverage** (§6.1). Its
// amount is `add · τ` of height and its per-unit opacity is the brush's own; the two
// meet only in the slab law `1 − exp(−K · opacity · height)`, which is what the
// color target carries so that `integrate.wesl` can stack the accumulated parcel
// through the shared law in `paint_common.wesl`. Emitting the opacity *alone* there
// would be the defect §6.3 names in the layer composite, one level down: a fragment
// whose brush lays almost no paint would arrive at the integrate claiming near-full
// coverage, so a low-`add` stroke would draw as nothing over bare canvas (the media
// pass catches that, via the height) and replace the color outright over paint
// (nothing catches that at all). It would also cost the faint edge of a soft tip a
// factor of `τ`, coverage reaching visible alpha through *both* factors of
// `op · height`; weighing the parcel by its own visible alpha is what keeps it linear
// in `τ`.
//   target 0: color (Rgba16Float) — the parcel's visible alpha in `.a`, with the
//             Oklab (L,a,b) latent premultiplied by it; over-blended, so the latent
//             a stroke crossing itself carries is ordered rather than averaged
//   target 1: aux   (Rgba16Float, the wide *scratch* layout — persistent tiles keep only
//             (height)) — (height, optical mass, unused, unused)

import package::stamp_common::{
    VsOut, FsOut, sweep_vertex, sweep_of, swept_depth, brush_color, stroke_arc, stroke_drain,
    substrate_tooth, jitter_of, ceiling_of, add_of, xf,
};
import package::lib::paint_common::{OPACITY_K, level_sums};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    // Per-segment instance data: what actually varies from segment to segment.
    // `drain` is the one rate that does not, and rides the `TileXform` uniform
    // because it is recovered per *fragment* from its own arc length rather than
    // being a segment constant at all.
    @location(0) start: vec2<f32>,
    // Unit tangent at the segment start.
    @location(1) dir: vec2<f32>,
    // The segment's **reference** radius (its midpoint tip), its arc length, and its
    // radius ramp — `(r_end − r_start)/radius`, so the tip a fraction `u` of the way
    // through is `radius·(1 + ramp·(u − ½))`. A segment does not have *a* radius: the
    // taper and the size modulation both scale the tip with distance travelled, and
    // carrying only one value per segment puts a step in the stroke's outline at every
    // cut (§6.2, `Segment::ramp`). Zero wherever the tip holds still, which is the
    // branch that keeps those strokes bit-identical.
    @location(2) geom: vec3<f32>,
    // Orientation (turns ∈ [0,1)), arc length at the segment start, signed
    // curvature, and the `add` source rate — the one paint rate the swept path
    // reads, here because a modulation can point at it (§6.2).
    @location(3) extra: vec4<f32>,
    // How much give this segment's tip has against the canvas substrate (§6.4). Its own
    // attribute rather than a lane of `extra`, because `extra` is full and a
    // five-float vector would be three unused lanes to carry one number.
    //
    // The give only. How wide the contact transition around it is
    // (`ToothParams::softness`) does not vary segment to segment, so it rides
    // the stroke uniform instead (`stamp_common::TileXform::paint`).
    @location(4) tooth_give: f32,
    // The tip drawn out along its facing axis (§6.6): the travel scale, the shear and
    // the lateral scale that carry a point of the reference travel frame into the
    // frame the prefix-τ volume is read in (`stamp_common::stretch_look`).
    // `(1, 0, 1)` — the identity, and exactly so — on every brush that does not stretch.
    //
    // Three lanes and not one knob, because what the fragment needs is the *solved*
    // map: the angle and the elongation cost a `sin_cos`, a normalize and an `atan2`
    // to turn into it, and this is per segment where that would be per texel in the
    // dynamics loop beside it. The host solves it once (`segments::Stretch::solve`).
    @location(5) stretch: vec3<f32>,
    // The ceiling's factor across this segment (§6.2): the mean of its
    // two ends, and the difference between them — what the fragment reads at its
    // own travel, the way the radius ramp beside it is read. `(1, 0)` on every
    // brush whose opacity the pen does not drive. Its own attribute for
    // `tooth_give`'s reason: `extra` is full.
    @location(6) opacity: vec2<f32>,
    // How the `add` source rate and the tooth's give change across this segment
    // (§6.2) — their means ride `extra.w` and `tooth_give`. `(0, 0)` on a brush the
    // pen does not drive that way.
    @location(7) ramps: vec2<f32>,
) -> VsOut {
    return sweep_vertex(vi, start, dir, geom, extra, tooth_give, stretch, opacity, ramps);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    // Where this fragment sits in the segment's travel frame — straight or curved,
    // the rest of this shader cannot tell the difference.
    let sw = sweep_of(in);
    // The segment's swept optical depth — the extent's geometric coverage
    // integrated along the travel. Everything below is built from this.
    let tau = swept_depth(in, sw);
    // The brush's remaining load here. A function of the fragment's own arc length
    // along the stroke, not of the segment it happens to be drawn by, so every rate
    // below stays independent of how the path was cut (`stroke_drain`).
    let drain = stroke_drain(stroke_arc(in, sw));

    // The parcel this segment lays. `add` is the brush's only source term (§6.1),
    // so it — and not the per-unit opacity — is what decides how much paint arrives;
    // a brush at zero flow lays a parcel of nothing, and the integrate below leaves
    // the canvas alone. Height is linear in `τ`, and the per-unit opacity is a
    // property of the pigment rather than of how much of it there is, so the mass
    // (their product) is linear in `τ` too.
    // The substrate's share of this texel (§6.4): where the substrate falls away faster than
    // this tip can follow, the paint simply does not land. A per-*fragment*
    // factor on `τ`, exactly like `drain` — so the deposit stays linear in `τ` and
    // exactly independent of how the path was cut, which a factor on the finished
    // parcel would not be.
    let tooth = substrate_tooth(in, sw);
    // The deposit jitter (§6.2): a second per-fragment factor on `τ`, beside the
    // tooth and under the same law — constant along the stroke at this texel, so
    // it factors out of the sum over the segments and the deposit stays exactly
    // independent of how the path was cut.
    let jitter = jitter_of(in);
    // `add` comes off the *instance*: it is the one paint rate the swept path reads,
    // and the pen can drive it (§6.2), so it varies segment to segment exactly as the
    // radius does. Constant within the segment either way, which is all the argument
    // above needs — the deposit is still linear in this segment's `τ`.
    //
    // The tooth gates the **height** and not the opacity beside it (§6.1), and the
    // jitter follows it: both decide how much paint arrives, not what the pigment is.
    let h = add_of(in, sw) * drain * tooth * jitter * tau;
    // Per-unit opacity: exactly 1, faded only by the drain — the paint a brush lays
    // is opaque per unit (§6.2). "How much shows" is not a material property any
    // more: the effect's own opacity scales the finished parcel where it lands
    // (`integrate.wesl`), so a half-opacity stroke is half a stroke rather than
    // full-bodied translucent material.
    let op = drain;
    let m = op * h;

    var out: FsOut;
    // The parcel's **visible alpha**: the slab law `1 − exp(−K·op·height)` that
    // `paint_common.wesl` weighs a parcel by, evaluated per segment. Being of the
    // form `1 − exp(−k·τ)` it is exactly what the color target's premultiplied
    // "over" composes — the complements multiply out to `1 − exp(−K·Σm)`, the whole
    // stroke's parcel, whether the path was cut into one segment or a hundred. The
    // color it carries is per-fragment: color dynamics jitter it across the
    // extent and along the stroke (§6.2), and the "over" keeps that jitter
    // *ordered*, so a stroke crossing itself covers rather than averages.
    let w = 1.0 - exp(-OPACITY_K * m);
    out.color = vec4<f32>(brush_color(in, sw) * w, w);
    // The ceiling lane (`FsOut::ceiling`): this segment's mass and moment above
    // each ceiling level, at its own factor — and the moment of *all* its mass in
    // the aux's spare `.z`, beside the mass it is the moment of. The dial is not
    // here — it is a stroke constant the integrate scales the claim by.
    var moment = 0.0;
    @if(ceiling) {
        let o = ceiling_of(in, sw);
        out.ceiling = level_sums(o, m);
        moment = o * m;
    }
    // The wide scratch aux (§6.1), both channels additive in τ so they sum to the
    // true path integral of the extent:
    //   .x = laid height — the amount of paint, so a second pass builds impasto in
    //        full (a saturating form would flatten out instead);
    //   .y = the parcel's optical mass (opacity · height) — the currency the parcel
    //        law blends in. It rides here *unsaturated* because `.a` above has
    //        already saturated, and the integrate needs the total to recover the
    //        merged per-unit opacity `m/h`;
    //   .z  = the mass's first moment over the ceiling factor, on a stroke whose
    //        ceiling the pen drives (§6.2) — zero on every other;
    //   .w  = unused.
    // Height is the amount of paint; the color alpha is per-unit opacity only and
    // never represents amount (§6.1). Height is also the *only* thing a persistent
    // tile keeps: gloss is a uniform property of paint, not a stored channel, so
    // there is nothing here that a brush could source it from.
    out.aux = vec4<f32>(h, m, moment, 0.0);
    // The parcel's residual (§6.7), premultiplied by the same `w` and over-blended by
    // the same rule as the color beside it: the pair is one color, and anything that
    // treated them as two quantities to combine differently would take the paint off
    // the sRGB color it was picked as.
    //
    // **Not jittered**, where the latent above is: color dynamics perturb the
    // *pigment mixture*, and the residual is the part of the picked color no mixture
    // of those pigments reaches. Jittering it would be jittering the correction rather
    // than the paint — and the correction has no direction in pigment space to be
    // jittered along.
    @if(resid) out.resid = vec4<f32>(xf.resid.xyz * w, w);
    return out;
}

// The erase pass's accumulator target (§6.12): one channel, the stroke's
// **transparency mass** summed additively across the overlapping segment quads —
// and, unlike the paint scratch beside it, across the *pieces* of a live stroke,
// which is what lets `erase.wesl` read the whole stroke's extent at once.
struct FsErase {
    @location(0) mass: vec4<f32>,
    // The erase's ceiling lane (§6.12) — `FsOut::ceiling` on the removing side,
    // with the moment of the whole mass beside it in a lane of its own, since
    // the accumulator has one channel and no spare. `erase.wesl` reads the pair
    // in place of the accumulated extent's coverage on a stroke whose ceiling
    // the pen drives.
    @if(ceiling) @location(1) ceiling: vec4<f32>,
    @if(ceiling) @location(2) moment: vec4<f32>,
}

// The erase variant of the sweep (§6.12): the same swept extent as
// [`fs_main`], laid as **transparency** rather than paint. What it writes is the
// mass of a parcel whose per-unit opacity is exactly 1 — so mass and height are one
// number, the arithmetic `fill.wesl` already makes for fully opaque paint — built
// from the same per-fragment chain as the deposit's height: `add` (the flow, and the
// eraser's one rate), the drain, the tooth and the jitter, all factors on `τ`, so
// the accumulated total is independent of how the path was cut (§6.2's two forms).
//
// **The brush's color is deliberately absent** — an eraser lays nothing for a
// color to be a property of (`brush_color` is the one read of `xf.color`, and this
// entry never makes it).
@fragment
fn fs_erase(in: VsOut) -> FsErase {
    let sw = sweep_of(in);
    let tau = swept_depth(in, sw);
    let drain = stroke_drain(stroke_arc(in, sw));
    let tooth = substrate_tooth(in, sw);
    let jitter = jitter_of(in);

    var out: FsErase;
    let mass = add_of(in, sw) * drain * tooth * jitter * tau;
    out.mass = vec4<f32>(mass, 0.0, 0.0, 0.0);
    @if(ceiling) {
        let o = ceiling_of(in, sw);
        out.ceiling = level_sums(o, mass);
        out.moment = vec4<f32>(o * mass, 0.0, 0.0, 0.0);
    }
    return out;
}

// The ceiling lane alone (§6.2): what the stamp loop draws per segment into its
// region, between that segment's deposit and the next's, so the loop's claim
// advances by the very sums the swept sweep accumulates — one fragment law for
// both paths. The mass is `fs_main`'s, gated the same way.
struct FsLevels {
    @location(0) levels: vec4<f32>,
}

@fragment
fn fs_levels(in: VsOut) -> FsLevels {
    let sw = sweep_of(in);
    let tau = swept_depth(in, sw);
    let drain = stroke_drain(stroke_arc(in, sw));
    let tooth = substrate_tooth(in, sw);
    let jitter = jitter_of(in);
    // **The mass, exactly as `fs_main` forms it** — the height, then faded again by
    // the per-unit opacity the drain leaves (§6.1). Both `drain`s are load-bearing
    // and the second is easy to read as a slip: what the lane holds has to be the
    // same quantity the aux's own total is, because `claimed_coverage` reads the
    // two against each other and clamps one by the other. Written as the height and
    // then the mass, the way the deposit writes it, so the pair cannot drift.
    let h = add_of(in, sw) * drain * tooth * jitter * tau;
    let m = drain * h;
    var out: FsLevels;
    out.levels = level_sums(ceiling_of(in, sw), m);
    return out;
}
