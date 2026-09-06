// The **slot** every dynamics kernel reads and the frame it is read in — shared by
// the wet loop (`dynamics.wesl`, §6.2) and the liquify field (`liquify.wesl`,
// §6.13), which run the same region machinery around different kernels off one
// per-slot uniform and one sweep frame.
//
// A binding-owning shared module, like `stamp_common`: it declares the slot uniform
// at `@group(0) @binding(0)` and the four other group-0 slots both roots write or
// gate by — the region's write targets and the selection mask — so each artifact's
// group 0 is partitioned between this module's five and the root's own, which
// `build.rs` checks (`bindings_do_not_collide`). The prefix volume at group 1 rides
// in through `stamp_common`, and it is a different volume for the two roots: the
// prefix-τ for every deposit, the coverage prefix for the liquify follow — the same
// lookup either way (`swept_pre`), which is the point of sharing it.
//
// Nothing here is a kernel. The mirror generates `Stamp` and `TILE_WG` under
// `mirror::dynamics_common` (§6.10), and both hosts fill the one uniform.

import package::stamp_common::{
    prefix_span, radius_ramp_apply, stretch_look, Sweep, sweep_at,
};
import package::lib::paint_common::{linear_across};

// One slot of the sequential loop — a painting segment, a bleed firing or the pen-up
// settle (`SlotKind`), all read through the same declaration.
//
// **Named members, not packed lanes.** This was twelve anonymous `vec4`s called `a`–`l`
// with the lane map written out twice — once as thirty accessor functions here and
// once as `gpu::stroke::dynamics::plan`'s `Slot::pack` — and the compiler could see
// neither. It checked the *size*; nothing checked the *meaning*, so swapping two lanes
// between the two sides compiled clean on both. Now the mirror generates
// `Stamp { travel_radii, .. }` from this declaration (§6.10) and the field names are
// the contract, with the generated `offset_of` assertions holding the layout.
//
// The packing bought nothing it cost: WGSL's uniform rules constrain array strides and
// a struct's own alignment, not scalar members — `filter_common.wesl`'s `Filter` has
// mixed `u32`/`f32`/`vec2` members and mirrors correctly — and this order has no
// interior padding at all, so it is the same 192 bytes the lanes were.
//
// The members that are **integral by construction** say so in the type. `rect_origin`,
// `cell_anchor`, `cell_px` and `bleed_reach` were floats the shader converted at every
// read (`i32(st.e.z)`), which is a truncation to reason about on a quantity that has no
// fraction to lose.
struct Stamp {
    // The sweep's start (region px) and its travel tangent (unit vector) — the prefix-τ
    // frame, and the same two fields for all three slot kinds. A **bleed** slot's sweep
    // is the firing's travel window rather than a stretch of painting; a **settle**
    // slot's is the tip's own extent, so its tangent is the direction the pen was
    // going when it left the page. (The tool's *shape* frame — this turned by
    // `orientation`, §6.6 — is derived where it is needed, in `bake` and `exchange`,
    // not carried here.)
    start: vec2<f32>,
    dir: vec2<f32>,
    // The tip's radius — region px per brush-local unit, and so the frame every
    // brush-local coordinate is read in: the arc, the color-noise domain, the
    // reservoir's placement. One radius for every brush (§6.6): a mask's content
    // lies inside the disc inscribed in its square, so its volume is baked unpadded
    // and the frame the sweep is unrolled in *is* the tip.
    radius: f32,
    // Segment length as a multiple of `radius`. 0 on a settle slot, which is a
    // break of contact rather than a stretch of travel.
    travel_radii: f32,
    // How much the tip grows across this slot's sweep, as a fraction of `radius`
    // (§6.2, `Segment::radius_ramp`), so the tip a fraction `u` of the way through is
    // `radius·(1 + radius_ramp·(u − ½))`. A segment does not have one radius —
    // the taper and the size modulation both scale the tip with distance travelled —
    // and a sweep at a single radius steps the stroke's outline at every cut.
    //
    // Its neutral value is 0, unlike `tooth_bearing` and `cell_px`, because it is a
    // *difference* rather than a scale: a zeroed lane is a tip that holds still, which
    // is what a bleed window and a settle both are, and what every untapered stroke at
    // constant pressure is.
    radius_ramp: f32,
    // The three exchange rates as the host packs them: `λ = ln(1 − axis) ≤ 0`, clamped
    // away from −∞. Every consumer wants `k = −λ ≥ 0`, which is what `lift_rate` and
    // its two neighbours below hand back — spelling the negation at each site is an
    // easy sign to lose in a kernel where the sign is the whole physics.
    lambda_lift: f32,
    lambda_deposit: f32,
    // The lateral canvas diffusion rate. Nonzero ONLY on a dedicated **bleed slot**
    // (§6.2): a quad the host emits per crossing of the bleed cadence, whose sweep IS
    // the firing's travel window and whose vertical rates and source are all zero. Real
    // segments carry 0 here and take the plain-deposit path bit-for-bit.
    lambda_bleed: f32,
    // The segment's signed curvature, 1/region px; 0 is a straight sweep.
    curvature: f32,
    // The liquify effect's follow rate (§6.13): the brush's modulated `strength`,
    // the fraction of the pass a texel under the tip's core keeps up with — the
    // whole pass at 1, and less wherever the tip's coverage is (`warp_follow`).
    // Nonzero ONLY on a **warp slot**; every other kind carries 0.
    drag: f32,
    // The brush's own color channels. **Undrained** — the falloff is applied per
    // fragment from its own arc length, not baked in per segment (see
    // `stroke_drain`). No per-unit opacity beside them: minted paint is opaque per
    // unit (§6.2), faded only by the drain.
    brush_lat: vec3<f32>,
    // The stroke's ceiling (§6.2): the paint effect's opacity with the selection
    // mask's folded in (`stroke_constants`, §6.8) — the per-stroke ceiling on
    // what the brush's own paint builds to. Applied where the mint lands
    // (`lay_parcel`), as prefix differences of the capped law over the running
    // raw totals the region aux's spare lanes carry — the very arithmetic the
    // swept integrate applies to its whole accumulated parcel, so a brush whose
    // axes are all zero but for a whisper of `deposit` lays exactly what the
    // fast path lays. The mask's per-texel coverage is the same ceiling's third
    // factor, taken from `sel_mask` at the texel. 1 is the identity, taken as
    // an exact branch.
    opacity: f32,
    // The brush color's **residual** — the half of it the space's three channels cannot
    // express (§6.7). Undrained, like `brush_lat`. Zero for every space but Mixbox.
    //
    // Declared unconditionally, where every texture that carries it is `@if(resid)`:
    // this struct is mirrored into Rust from the *unlinked* source (§6.10), which has
    // no feature set to evaluate, so a member that came and went with the variant would
    // give one Rust struct two layouts.
    brush_res: vec3<f32>,
    // The `add` source rate — height per unit exposure, undrained like `brush_lat`.
    // The effect's opacity is already folded in (§6.2): on this path the ceiling
    // cannot be exact — moved paint moves whole — so it scales what the brush mints.
    add: f32,
    // Color dynamics (§6.2) for the brush's own `add` paint — the same field and domain
    // as the fast path's jitter, so the two paths agree. Frequency per lookup axis
    // (across the stroke, along it) in `.xy`, with 1/NOISE_TILE_PX pre-folded into `.z`.
    noise_freq: vec3<f32>,
    // Arc length at the segment start, canvas px.
    arc_at_start: f32,
    // Per color-channel noise amplitude; all zero is jitter off.
    noise_amp: vec3<f32>,
    // The `drain` falloff per canvas px (`BrushParams::drain_px`: the brush's own
    // knob is per *radius*, and the host divides before it gets here).
    drain: f32,
    // The tip **drawn out along its facing axis** (§6.6), solved into the map that
    // carries a point of the reference travel frame into the frame the prefix-τ volume
    // and the bake rows are indexed in: the travel scale, the shear, and the lateral
    // scale (`stamp_common::stretch_look`).
    //
    // `(1, 0, 1)` — the identity, and exactly so in floats — on every brush that does
    // not stretch, which is what keeps this whole loop bit-for-bit what it was. Its
    // neutral value is that triple and not zeroes: a zeroed member is not "no stretch"
    // but a tip of no width and infinite gain.
    //
    // The rotation a stretch implies is **not** here, and that is the point of the
    // arrangement: a stretch changes the direction the sweep is integrated along, and
    // `orientation` already carries the change, because the volume is indexed by
    // exactly that angle. So a stretched tip is another slice of the same volume, and
    // nothing in this file has a second texture to bind or a second angle to turn by.
    stretch: vec3<f32>,
    // The shape's orientation in turns ∈ [0,1) — which picks the prefix-τ slice (§6.6)
    // and is the rotation `bake` and `exchange` take out of the travel frame.
    orientation: f32,
    // The map from this dispatch's *region* texel to the substrate:
    // `uv = rt·substrate_uv_scale + substrate_uv_bias`, with the piece's own canvas origin
    // already folded into the bias. The scale is zero on a substrate with no relief, which
    // is what makes a `Flat` canvas render exactly what it always did.
    substrate_uv_bias: vec2<f32>,
    // Where this dispatch's extent rect sits in the region, in texels.
    rect_origin: vec2<i32>,
    // The cell grid's canvas anchor: the region's canvas origin modulo the cell, so
    // `region texel + cell_anchor` is congruent to the texel's canvas position modulo
    // the cell. That is the whole §6.4 story for this pass — cell boundaries land on
    // the same canvas texels whichever region this piece, or the live fold's next one,
    // happens to put around them, without the shader ever learning a canvas coordinate
    // too large for an f32 to hold exactly.
    cell_anchor: vec2<i32>,
    // The deposition tooth (§6.4): how much give this segment's tip has against the
    // substrate. 1 is all of it, and means the substrate does not gate anything.
    //
    // Neutral at 1 rather than 0, like `tooth_bearing` and `cell_px`, and for a reason
    // of its own: this knob runs the other way (`ToothParams::give`), so a zeroed
    // lane is not "no tooth" but the driest tip there is.
    tooth_give: f32,
    // How wide the contact transition around that level is, in the rise's own units
    // (`ToothParams::softness`, §6.4) — how gradually a texel goes from taking
    // none of the tip's paint to taking all of it. A stroke constant, unlike the give
    // beside it: what the tip is made of does not change under the hand. It is a
    // member here rather than a second uniform for `drain`'s reason — this loop's
    // uniform is per *slot*, so a stroke constant simply rides every slot.
    tooth_softness: f32,
    // The tooth's **bearing fraction** — the share of the substrate a tip with this
    // `tooth_give` and `tooth_softness` stands on, averaged over the whole substrate,
    // along **this segment's own
    // direction of travel** (`Surface::bearing`, §6.4). It is what the *tool* books its
    // half of the transfer against: a canvas texel scales its exposure by the substrate
    // under itself, and a reservoir cell, which has no substrate of its own, scales by
    // this. 1 where there is nothing to bite.
    tooth_bearing: f32,
    substrate_uv_scale: f32,
    // The **extent cell** (§6.2): its edge in texels, ≥ 1 — the square the coarse
    // pair (`cell_hoist`/`deposit_coarse`) evaluates the exchange laws once per. 1 (and
    // `cell_anchor` zero) on every slot the host keeps on the exact path: small tips,
    // hard tips, bleed and settle slots, none of which read this.
    cell_px: i32,
    // The **bleed stencil's longest tap** in texels, ≥ 1 — nonzero ONLY on a bleed
    // slot, and solved per firing from the diffusivity the axis asks for
    // (`stroke::budget::bleed_stencil`). It arrives as a number rather than a rule so
    // that every thread of a flux pair reads one uniform and derives one set of
    // integers, which is what the antisymmetry rests on.
    bleed_reach: i32,
    // The deposit jitter (§6.2): the canvas texel this region's texel (0,0) sits on.
    // The jitter's domain is the absolute canvas grid, like the substrate's — and
    // integral, exact in an i32 where a canvas coordinate might not be in an f32,
    // which is what lets this loop key a hash on canvas position at all (§6.4).
    canvas_origin: vec2<i32>,
    // …with the gate's half-range (0 = off, and the deposit is bit-identical) and
    // the stroke's own seed for it.
    jitter_eps: f32,
    jitter_seed: u32,
    // The ceiling's modulation factor across this slot (§6.2): the mean of its two
    // ends, and — in `opacity_ramp` — the difference between them, which
    // `paint_common::linear_across` reads at the texel's own travel exactly as the
    // swept sweep does. Read only where `ceiling_lane` says so; `(1, 0)` — a
    // scale's neutral and no ramp — everywhere else. Here at the tail with its
    // flag rather than beside `opacity`, where a scalar after the `vec3` lanes'
    // filler would open a 16-byte hole.
    opacity_mod: f32,
    opacity_ramp: f32,
    // How the slot's other **per-texel** pen targets change across its travel
    // (§6.2): the `add` source rate, the tooth's give, and the liquify follow, each
    // as the difference between the slot's two ends, read by `rate_across` at the
    // texel's own arc. Their means ride `add`, `tooth_give` and `drag`, so a brush
    // the pen does not drive this way carries zeroes and every read is an identity.
    //
    // `lambda_lift`, `lambda_deposit` and `lambda_bleed` have no ramp beside them
    // and cannot: the exchange is one solve per dispatch whose two halves are
    // complements, and the tool half lives in the shape frame with no canvas
    // position to vary along. What bounds their step is the flattener's attribute
    // budget instead (`BrushParams::max_slope`).
    add_ramp: f32,
    tooth_give_ramp: f32,
    drag_ramp: f32,
    // Whether the stroke's ceiling is pen-driven (§6.2): 1 routes the mint through
    // the ceiling lane the region aux's `.w` carries (`lay_parcel`), reading
    // `opacity_mod` per slot; 0 is the dial's own law, bit for bit. A stroke
    // constant, like `opacity` — every slot of a stroke says the same, since which
    // lane a texel's claim lives in cannot change segment to segment.
    ceiling_lane: u32,
    // Where this region's texel (0, 0) sits in the **base composite** the liquify
    // apply resamples from (§6.13) — the base is the region grown by the run's reach
    // on every side, so this is that margin, in texels. Read by `warp_apply` alone.
    base_offset: vec2<i32>,
}

@group(0) @binding(0) var<uniform> st: Stamp;

// ---- Reading the slot ------------------------------------------------------------
//
// What is left of the accessor block the packed lanes needed: the four readings that
// are arithmetic rather than a name. Every other member is read as `st.<name>`, since
// there is now nothing between the reader and the declaration to decode.

// The segment's travel in px — the product the arc arithmetic wants.
fn travel_px() -> f32 { return st.travel_radii * st.radius; }

// One of this slot's ramped rates at a point, addressed by its **arc** (§6.2,
// `paint_common::linear_across`).
//
// By arc rather than by the travel frame's own `x`, because that is the one
// coordinate both halves of a transfer can name. The canvas half reads a texel's
// arc off the segment frame; the tool half reads its reservoir texel's, which sits
// somewhere else entirely (`exchange`) — and the two must agree on the rate for the
// same reason they must agree on the drain: what one hands over the other takes
// back, and a pair that read different rates would not balance.
fn rate_across(mean: f32, delta: f32, arc: f32) -> f32 {
    return linear_across(mean, delta, arc - st.arc_at_start, travel_px());
}

// A texel's arc along the stroke, from the frame the sweep was unrolled in: the
// segment's own start plus how far into it this point sits. The travel axis is
// denominated in the *reference* radius (`stamp_common::Sweep`), which is what
// `st.radius` is.
fn arc_at(sw: Sweep) -> f32 {
    return st.arc_at_start + sw.local.x * st.radius;
}

// The tile kernels' workgroup side: every entry point below that walks a rectangle of
// region texels is dispatched over `ceil(extent / TILE_WG)` groups per axis.
//
// Declared here so the host reads it from the mirror rather than transcribing it
// (§6.10), which is what `dynamics.wesl`'s `BAKE_RES` does for the scan's width. It was
// the literal `8` in seven `@workgroup_size` attributes and six host expressions;
// changing it for a tuning experiment meant finding all six by hand, and missing the
// one that turns groups back into texels (`cell_geometry`) under-counts the cells the
// hoist writes — which `deposit_coarse` then reads as the previous segment's
// leftovers. A wrong picture, not a validation error.
const TILE_WG: u32 = 8u;

// The brush's remaining load at a point `arc` canvas px along the stroke: `drain`
// fades what it lays, linearly with distance travelled, until it runs dry (§6.2).
//
// **Every consumer evaluates this at its own position rather than per segment**, which
// is what keeps the paint laid independent of how the path was cut — the same property
// the exposure integral has, and for the same reason it matters. The falloff is linear
// in arc length, so sampling it where the paint actually lands is exact, not merely
// closer — which is why `gpu::stroke::budget::flatten_tolerance` need not buy accuracy
// for it with a `0.02 / drain` cap on segment length.
//
// **Bounded at a full load, not only at an empty one.** A texel is measured at its own
// arc, and the texels behind the stroke's *start* sit at a negative one — the tip's
// trailing half covers them at pen-down — so the unbounded form hands them more than a
// full brush. That is meaningless on the load, and worse on the per-unit opacity it
// multiplies: `src_m = drain·src_h` lands mass above the height carrying it,
// and `m ≤ h` is the invariant the region's storage rests on, since what it stores is
// `m / h`. Both paths bound it identically (`stamp_common.wesl::stroke_drain`), so a
// drained brush lays the same start cap whichever one draws it.
fn stroke_drain(arc: f32) -> f32 {
    return clamp(1.0 - st.drain * arc, 0.0, 1.0);
}

// The region, written in place by `deposit` and whole by `warp_apply` — the
// targets the write-back slices for either root.
@group(0) @binding(13) var region_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var region_aux_w: texture_storage_2d<rgba16float, write>;

// The selection's coverage over the region (§6.8), 1:1 with it — or a 1×1
// texture holding a constant when there is nothing to mask. The coverage alone:
// the mask's overall opacity is inside `st.opacity`. Both the vertical fluxes are
// gated by it, under the two laws a mask has (`lay_parcel`): paint that *moves*
// moves in that fraction — `exchange` scales what it lifts, the deposit takes the
// same fraction of the lift and of the tool's carried paint, so paint outside the
// selection is neither taken nor laid and the two sides of the transfer still
// balance — and paint that is *minted* is capped at `st.opacity · sel`, the swept
// integrate's own reading of the same texel.
@group(0) @binding(21) var sel_mask: texture_2d<f32>;

// The region's residual write target (§6.7), beside its color partner above; the
// read side, `region_resid`, is the wet loop's own. `@if(resid)` like every
// residual slot: Oklab declares none of them.
@if(resid) @group(0) @binding(24) var region_resid_w: texture_storage_2d<rgba16float, write>;

// This texel's place in the segment's travel frame — the one the whole exchange is
// measured in. Straight or curved, `sweep_at` hands back the same three quantities
// (see stamp_common.wesl), so nothing downstream branches on the segment's shape.
//
// `st.travel_radii` carries the travel as `length / radius`, so the arc length it was built
// from is recovered by multiplying back through.
//
// **The single place the loop builds a frame**, which is what lets the radius ramp
// (§6.2, `Segment::radius_ramp`) be applied in exactly one line: `deposit`, its coarse twin,
// the bleed flux, `snapshot` and `settle` all come through here, so all five agree
// about where the sweep's rim is — and `outside_sweep`, which several of them use to
// skip work the others must then also skip, is computed from the same frame. Applying
// it anywhere further down would be five chances to disagree.
//
// The tool *reservoir* deliberately does not ramp: `bake` and `exchange` place a
// brush-local grid, which is the tool's own and does not stretch as the tip does, and
// they reach the canvas through `st.radius` rather than through a `Sweep` at all.
// What ramps is the canvas-side coverage — the prefix-τ difference, which is what the
// stroke's outline is made of.
fn segment_frame(rt: vec2<i32>) -> Sweep {
    let p = vec2<f32>(rt) + vec2<f32>(0.5, 0.5);
    let sw = sweep_at(p, st.start, st.dir, st.radius, travel_px(), st.curvature);
    return radius_ramp_apply(sw, st.radius_ramp);
}

// Where that frame lands in the one the prefix-τ volume and the bake rows are indexed
// in, once the tip is drawn out along its facing axis (§6.6, `stretch_look`): the
// travel coordinate at the slot's end .x, at its start .y, and the lateral offset .z.
//
// **The frame above stays the tip's own and this is layered on top**, for the reason
// the fast path's `swept_depth` gives: everything else read out of a frame here — the
// arc a texel sits at, its lateral offset for the color noise, the tip in force, the
// canvas position a reservoir texel is dragged to — is a property of where the *tip*
// went, and only the lookup into the mask is re-coordinated.
fn segment_look(sw: Sweep) -> vec3<f32> {
    return stretch_look(sw, st.stretch);
}

// This texel's swept exposure over the slot, **in the lookup frame's measure** — the
// prefix-τ difference across its own stretch of travel, clamped at zero because a
// difference of two nearly equal prefixes is rounding out past the extent.
//
// One function because four callers want exactly this number and would each have had
// to apply the stretch: `deposit`, its coarse hoist, the bleed flux's neighbour tap,
// and `snapshot` by way of `outside_look`. A stretch applied in three places out of
// four is an extent whose flux is antisymmetric with itself and not with its
// neighbour — the one property the bleed pair's conservation rests on.
//
// `stretch_gain` is deliberately *not* folded in: this measure is the one the bake was
// integrated in, so the ratios `deposit` takes against it cancel rather than square.
// Its callers apply the gain where they want the tip's own measure instead.
//
// **It takes the look, not the frame**, so a caller that also tests the rim computes
// `segment_look` once. `deposit` used to build it three times per texel — inside the
// rim test, inside here, and again for the bake rows — and the bleed ladder built two
// more for each of its thirty-six neighbours.
fn swept_pre(l: vec3<f32>) -> f32 {
    return max(prefix_span(l.x, l.y, l.z, st.orientation), 0.0);
}

// Whether a region texel is provably outside the segment's sweep — the span
// x ∈ (-1, lr+1), |y| < 1 in the travel frame — so its exposure is exactly 0 and
// `deposit` leaves it untouched. The dispatch rect is the sweep's AABB, so at a
// diagonal orientation up to half its texels are these corners; the test rejects
// them with a few ALU ops instead of the prefix-τ taps `deposit` would otherwise
// spend proving e = 0. `snapshot` may skip exactly the same texels because
// `deposit` reads the snapshot only after its own e > 0 check, and the two compute
// the identical frame from the identical uniform — same floats, same verdict.
//
// Taken in the **lookup** frame, which is where those rims are: a stretched tip paints
// out past the rim its own radius names, and a rejection keyed on the reference frame
// would cut the mark off along a straight line exactly there.
//
// Beyond the lateral edges the prefix rows are the extent's own zero border; past
// the trailing edge both taps clamp to the row total. Either way the difference is
// exactly 0. The lateral test comes first because it is also what rejects a texel near
// a tight arc's centre of curvature, where the travel coordinate has no meaning to test.
fn outside_look(l: vec3<f32>) -> bool {
    return abs(l.z) >= 1.0 || l.x <= -1.0 || l.y >= 1.0;
}

// [`outside_look`] for a caller that wants nothing else out of the frame — `snapshot`,
// and the coarse deposit, which reads its exposure from the cell scratch instead.
fn outside_sweep(sw: Sweep) -> bool {
    return outside_look(segment_look(sw));
}
