// Brush dynamics: the **sequential swept-exchange loop** (§6.2).
//
// A stroke that manipulates existing paint (lift / deposit / charge) runs as an
// ordered loop, all inside one compute pass (implicit barriers between dispatches
// give the sequencing; usage scopes are per-dispatch, so the region may be
// sampled by one dispatch and storage-written by the next — no copies):
//
//   - per flattened segment, in order: `bake` (integrate the reservoir along the
//     travel axis, so the deposit can read what the tip presented over a whole pass
//     rather than at one point),
//     `exchange` — one thread per **tool reservoir** texel, which settles the tool's
//     half of the transfer, and which also carries the `snapshot` (copy the segment's
//     extent texels to an `under` scratch, so the deposit can read-modify-write the
//     region) in the tail of its own grid — and `deposit`, one thread per extent
//     texel. `exchange` ping-pongs the reservoir, so a segment cycles it once.
//
// **The snapshot is not a step of the loop, it is the read side of the deposit**, and
// it depends on nothing the exchange writes. Giving it a dispatch of its own bought a
// barrier neither pass wanted, so it rides in `exchange`'s grid instead: four
// serialized dispatches per segment where there were five. The pen-up keeps the
// standalone entry point, because there the `settle` *reads* what the snapshot writes —
// and it runs a `bake` of its own, whose zero-travel slot stores the remaining pass's
// delivery integral rather than a per-segment window (see `settle`).
//
// **`exchange` runs before `deposit`, not after.** The two are the two halves of one
// transfer, and a transfer is a statement about a *single* pre-state: both dispatches
// evaluate the same kernel (`exchange_at`) from the same segment-entry canvas and the
// same segment-entry reservoir, and hand back complementary fractions of it. Reading
// the region as the deposit has already left it would make the tool lift from a canvas
// that has, from its point of view, not happened yet, and the halves would stop adding
// up (see `exchange_at`).
//
// **The canvas-side exchange is swept, not stamped** — the same definite-integral
// formulation as the plain deposit: a texel's *exposure* to the segment is the
// prefix-τ difference `e(x) = prefix(u) − prefix(u − d)` (the brush's coverage
// integrated along the travel), and every exchange rate is exponential in
// exposure, `1 − exp(λ·e)` with `λ = ln(1 − axis)`. Exposures add across the
// overlapping quads of consecutive segments, so the rates compose **exactly**:
// the whole stroke applies `1 − (1−axis)^∫e` — the continuous path integral,
// with no per-stamp dabbing regardless of spacing.
//
// The reservoir is a real 2-D texture in brush-local coordinates, so each part of
// the tip carries what *it* rolled through; because `exchange` reads the region as
// already modified by earlier segments, smearing is genuinely order-dependent —
// including across the stroke's own trail.
//
// **The tool exchanges once per segment, at the same cadence as everything else.**
// The canvas side of the transfer is swept per segment (above), so the tool side has
// to be too, or the tool gains what the canvas gave up only some segments later. On a
// coarser cadence of its own — a reload every `spacing·radius` of travel, say — a
// stroke stopping mid-interval leaves its last extent stripped by a `deposit` whose
// paint the tool has not yet been credited with: a tip-shaped patch of missing paint,
// worst exactly where `lift` is high.
//
// So `exchange` does the whole of the tool's side in one dispatch, per segment,
// against the very same segment the `deposit` accounts for. Nothing is deferred, so
// nothing is left owing at the end of a stroke.
//
// Matching the cadences is also what keeps this loop small. A coarser reload has to
// hide its own discreteness — a whole interval's load landing between two segments is
// a large fraction of it at high lift × deposit, and consecutive segments would be
// served visibly different tools — which takes a **gain parcel**, a linear ramp
// serving the load in across the interval, and two extra baked prefixes to evaluate
// that ramp per fragment. A per-segment exchange has no step to hide: the segment
// length *is* the reload step, and `stroke::budget::flatten_tolerance` already caps it
// at a fraction of the tip.
//
// Conservation (§6.1): paint moves by transferring **height** — the one
// conserved quantity. Color and per-unit opacity ride as optical-mass
// (opacity·height) weighted blends; `add` is the only source term. Both sides of
// each transfer integrate the same rate over the same extent, so with
// `add = 0` total height (canvas + tool) is conserved up to resampling error.
//
// Determinism: the slot sequence — segment, bleed and settle alike (`SlotKind`) —
// is fixed CPU-side per stroke; every dispatch is a pure function of what it reads.
// Replay and live == committed hold.

import package::stamp_common::{
    prefix_at, prefix_span, ramp_apply, stretch_gain, stretch_look, stretch_unlook, Sweep,
    sweep_at, sweep_tangent,
};
import package::lib::noise::{color_jitter, jitter};
import package::lib::latent::{
    Latent, latent_add, latent_any, latent_blend, latent_div, latent_lerp, latent_of,
    latent_scale, latent_sub, latent_unpremult, latent_wsum, latent_zero,
};
import package::lib::paint_common::{substrate_texel_at, rise_ahead, tooth_gate, PAINT_EPS};
import package::lib::sample::{load1, bilerp_taps, bilerp_mix};
import package::lib::store::{f16_nearest4};

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
    // The radius of the **frame the sweep is unrolled in** — region px per brush-local
    // unit. The frame, not the tip: the two are the same for every brush but a
    // pen-oriented stamp, whose prefix-τ volume is padded so the mask can turn inside
    // it without losing its corners (§6.6). There the volume's `[-1, 1]²` is
    // `frame_scale` tips wide, and everything read out of a brush-local coordinate —
    // the arc, the color-noise domain, the reservoir's placement — is in that larger
    // frame.
    frame_radius: f32,
    // Segment length as a multiple of `frame_radius`. 0 on a settle slot, which is a
    // break of contact rather than a stretch of travel.
    travel_radii: f32,
    // How much the tip grows across this slot's sweep, as a fraction of `frame_radius`
    // (§6.2, `Segment::ramp`), so the tip a fraction `u` of the way through is
    // `frame_radius·(1 + radius_ramp·(u − ½))`. A segment does not have one radius —
    // the taper and the size modulation both scale the tip with distance travelled —
    // and a sweep at a single radius steps the stroke's outline at every cut.
    //
    // Its neutral value is 0, unlike `frame_scale` and `cell_px`, because it is a
    // *difference* rather than a scale: a zeroed lane is a tip that holds still, which
    // is what a bleed window and a settle both are, and what every untapered stroke at
    // constant pressure is.
    radius_ramp: f32,
    // How many tips wide the frame `frame_radius` measures is (§6.6), and so the one
    // conversion between the frame's units and the **mask's**. Exactly 1 for every
    // unpadded volume, which is every brush but a pen-oriented stamp, so a slot that
    // never heard of padding reads its extent exactly as it always did.
    //
    // The canvas side never needs it: a prefix-τ difference is an absolute optical
    // depth, integrated at the mask's own column width whatever box was baked around it
    // (`assets::build_prefix_tau`). Only the tool side, which has no prefix to
    // difference and reconstructs its exposure from the raw coverage, has to know.
    //
    // Neutral at 1 rather than 0, like `tooth_bearing` and `cell_px`: the lane is a
    // *scale*, and a zeroed one is not "no padding" but "no tip".
    frame_scale: f32,
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
    // The brush's own color channels and per-unit opacity. **Undrained** — the falloff
    // is applied per fragment from its own arc length, not baked in per segment (see
    // `stroke_drain`).
    brush_lat: vec3<f32>,
    brush_op: f32,
    // The brush color's **residual** — the half of it the space's three channels cannot
    // express (§6.7). Undrained, like `brush_lat`. Zero for every space but Mixbox.
    //
    // Declared unconditionally, where every texture that carries it is `@if(resid)`:
    // this struct is mirrored into Rust from the *unlinked* source (§6.10), which has
    // no feature set to evaluate, so a member that came and went with the variant would
    // give one Rust struct two layouts.
    brush_res: vec3<f32>,
    // The `add` source rate — height per unit exposure, undrained like `brush_op`.
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
    // The per-stroke noise lookup translation.
    noise_off: vec2<f32>,
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
    // of its own: this knob runs the other way (`BrushParams::tooth_give`), so a zeroed
    // lane is not "no tooth" but the driest tip there is.
    tooth_give: f32,
    // How wide the contact transition around that level is, in the rise's own units
    // (`BrushParams::tooth_softness`, §6.4) — how gradually a texel goes from taking
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
}

@group(0) @binding(0) var<uniform> st: Stamp;

// ---- Reading the slot ------------------------------------------------------------
//
// What is left of the accessor block the packed lanes needed: the four readings that
// are arithmetic rather than a name. Every other member is read as `st.<name>`, since
// there is now nothing between the reader and the declaration to decode.

// The segment's travel in px — the product the arc arithmetic wants.
fn travel_px() -> f32 { return st.travel_radii * st.frame_radius; }

// The three exchange rates, as `k = −λ ≥ 0`. `lift` and `deposit` are the vertical
// pair; `bleed` is the lateral one, and is nonzero only on a dedicated bleed slot.
fn lift_rate() -> f32 { return -st.lambda_lift; }
fn deposit_rate() -> f32 { return -st.lambda_deposit; }
fn bleed_rate() -> f32 { return -st.lambda_bleed; }

// The stroke's canvas region: premultiplied color (latent·op, op) + wide aux
// (height, _, _, _). Read by `snapshot` (textureLoad) and `exchange` (bilinear).
@group(0) @binding(1) var region_color: texture_2d<f32>;
@group(0) @binding(2) var region_aux: texture_2d<f32>;
// Extent snapshot targets (written by `snapshot`).
@group(0) @binding(3) var under_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var under_aux_w: texture_storage_2d<rgba16float, write>;
// Bilinear clamp sampler for the region / reservoir / coverage lookups.
@group(0) @binding(5) var samp: sampler;
// The brush tip's coverage mask over brush-local [-1,1]² (R8; §6.6) — the
// reservoir texel's own extent weight. Read by the **tool** side only
// (`exchange`); the canvas side uses the prefix-τ exposure instead.
@group(0) @binding(6) var cov_tex: texture_2d<f32>;
// The tool reservoir ping-pong: color = (latent .xyz, per-unit opacity .w),
// aux = (carried height, _, _, _). `src` is read this dispatch, `dst` written;
// the host swaps the bind group after each pass that writes it — `exchange`,
// so a painting segment cycles it once. A bleed or settle slot
// writes it not at all and leaves it where the previous segment put it.
@group(0) @binding(7) var brush_src_color: texture_2d<f32>;
@group(0) @binding(8) var brush_src_aux: texture_2d<f32>;
@group(0) @binding(9) var brush_dst_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var brush_dst_aux_w: texture_storage_2d<rgba16float, write>;
// The extent snapshot (read by `deposit`).
@group(0) @binding(11) var under_color: texture_2d<f32>;
@group(0) @binding(12) var under_aux: texture_2d<f32>;
// The region, written in place by `deposit`.
@group(0) @binding(13) var region_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var region_aux_w: texture_storage_2d<rgba16float, write>;
// The color-dynamics noise field + its repeat sampler (deposit only): jitters
// the brush's own `add` paint exactly like the fast path's stamp (§6.2).
@group(0) @binding(15) var dyn_noise_tex: texture_2d<f32>;
@group(0) @binding(16) var dyn_noise_samp: sampler;
// The reservoir's **swept prefix** in the travel frame (written by `bake`, read by
// `deposit`): running integrals of the carried paint weighted by the extent's
// optical-depth density.
//   load = (∫h·dτ, ∫m·dτ, unused, unused)
//   latm = (∫lat·m·dτ,        unused)
// fp32 because each fragment takes a *difference* of two prefix sums, exactly as the
// prefix-τ volume does, and for the same cancellation reason. Two targets where there
// were four: the other pair only ever carried the retired reload ramp.
@group(0) @binding(17) var bake_load_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(18) var bake_latm_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(19) var bake_load: texture_2d<f32>;
@group(0) @binding(20) var bake_latm: texture_2d<f32>;
// The selection's coverage over the region (§6.8), 1:1 with it — or a 1×1
// texture holding a constant when there is nothing to mask. Both the vertical fluxes
// are gated by it: `exchange` scales what it lifts, `deposit` lerps its whole
// read-modify-write back towards the untouched region, so paint outside the selection
// is neither taken nor laid and the two sides of the transfer still balance.
@group(0) @binding(21) var sel_mask: texture_2d<f32>;

// The canvas substrate's tileable height map (§6.4) — height in `r` (the media pass's),
// the rise one reach ahead in `gb` — whose rise along the travel scales the exposure
// a canvas texel sees. Read nearest (`substrate_texel_at`), so no sampler. Named
// by `deposit` and `settle`, the two entry points that put paint on the canvas.
@group(0) @binding(22) var substrate_tex: texture_2d<f32>;

// ---- The residual channel (§6.7) -------------------------------------------------
//
// One texture beside each of the color textures above, carrying the half of the
// color the space's three channels cannot express. A residual is not a second
// quantity to conserve but the rest of the same color, so each of these is read,
// written, swept and blended at exactly the point its color partner is, on the
// same masses and the same shares — there is no rate, no prefix and no bound here
// that the latent does not already have.
//
// All `@if(resid)`: Oklab reproduces every sRGB color in three channels, so it
// declares none of these and its bind group layout is shorter rather than padded
// with 1×1 stand-ins.
//
// The region's residual, premultiplied by the same per-unit opacity `region_color`
// carries in its `.a` — the alpha is not duplicated here because nothing samples
// this texture without sampling that one.
@if(resid) @group(0) @binding(23) var region_resid: texture_2d<f32>;
@if(resid) @group(0) @binding(24) var region_resid_w: texture_storage_2d<rgba16float, write>;
// The extent snapshot's residual (written by `snapshot`, read by `deposit` and
// `settle`), premultiplied like the region it is copied from.
@if(resid) @group(0) @binding(25) var under_resid: texture_2d<f32>;
@if(resid) @group(0) @binding(26) var under_resid_w: texture_storage_2d<rgba16float, write>;
// The tool reservoir's residual, **straight** (not premultiplied), like the latent
// in `brush_src_color.rgb` beside it. Ping-ponged by the same host swap.
@if(resid) @group(0) @binding(27) var brush_src_resid: texture_2d<f32>;
@if(resid) @group(0) @binding(28) var brush_dst_resid_w: texture_storage_2d<rgba16float, write>;
// The residual's swept prefix — `∫res·m·dτ`, the same optical-mass weight and the
// same fp32 differencing as `bake_latm`, so the ratio that recovers it is the ratio
// that recovers the latent.
@if(resid) @group(0) @binding(29) var bake_rlm_w: texture_storage_2d<rgba32float, write>;
@if(resid) @group(0) @binding(30) var bake_rlm: texture_2d<f32>;

// ---- The extent-cell scratch (§6.2, coarse deposit) ---------------------------
//
// Written by `cell_hoist` — one thread per canvas-anchored cell of the segment's
// extent, plus a one-cell border the bilinear read needs — and read back by
// `deposit_coarse`, which is the whole trade: the two prefix-τ differences and the six
// `bake_at` taps run once per cell instead of once per texel. fp32 like the bake it
// distils, and for one more reason besides precision: **exact zeros are semantic
// here**. A cell whose centre the sweep never covers stores exposure 0, which is what
// lets `deposit_coarse` return before touching the region once all four of a texel's
// neighbours are zero — the same "keeps its value" contract the exact kernel's
// `dpre <= 0` gives, on which the f16 re-store guard rests.
//
// Every lane past the exposure is **premultiplied by the weight that averages it**, so
// the four taps blend as masses and not as means (see the coarse pair below):
//
//   tool = (e, e·height, e·mass, —)  with e the exposure in the tip's measure
//   lat  = (e·mass·latent, —)        the same mass those means are taken against
@group(0) @binding(31) var cell_tool_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(32) var cell_lat_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(33) var cell_tool: texture_2d<f32>;
@group(0) @binding(34) var cell_lat: texture_2d<f32>;
// The residual's cell mean (§6.7), recovered against the same masses at the same
// centre — one more channel of the same color, exactly as everywhere else.
@if(resid) @group(0) @binding(35) var cell_res_w: texture_storage_2d<rgba32float, write>;
@if(resid) @group(0) @binding(36) var cell_res: texture_2d<f32>;

// ---- The bleed pair's mobility, hoisted (§6.2) -----------------------------------
//
// Written by `bleed_weight` — one thread per snapshot texel — and read by `deposit`'s
// ladder, which asks for it thirty-six times a texel and, before this, *recomputed* it
// thirty-six times: a sweep frame, a rim test and a prefix-τ difference per tap, each
// of them the answer some neighbouring thread was deriving at the same moment for the
// same texel. R32Float, because it is one number and the difference that makes it is
// taken in the fp32 the prefix volume is kept in.
//
// **It also makes the pair's antisymmetry structural.** The flux is a difference two
// threads must agree on to the bit, and the argument for that used to be that both
// derive `w` from one uniform through one function — true, and an argument. Now both
// *read the same texel*, so there is nothing left to argue about.
@group(0) @binding(37) var bleed_w_w: texture_storage_2d<r32float, write>;
@group(0) @binding(38) var bleed_w: texture_2d<f32>;
// (The prefix-τ volume rides at group(1) binding(0) via the stamp_common import.)

const TAU: f32 = 6.283185307179586;

// ---- The exchange kernel ---------------------------------------------------------
//
// Over one segment a canvas point and the tool above it are a pair of coupled boxes
// trading the one conserved quantity — height (§6.1) — at rates proportional to their
// exposure `e`, with the brush's own `add` a source into the canvas box:
//
//     dh/de = −k_lift·h + k_dep·R + A          (canvas)
//     dR/de = +k_lift·h − k_dep·R              (tool)
//
// where `k = −λ ≥ 0`. Their sum moves only by the source, so with `add = 0` the pair
// conserves height *identically*, not approximately.
//
// **Both halves of the loop evaluate this one solution**, over the same exposure and
// from the same segment-entry state, and take complementary shares of it — which is
// what makes the whole exchange **independent of how the path was cut into segments**,
// the property adaptive flattening (`path::flatten`) depends on and which uniform 2px
// sampling would only ever hide. The system is linear and autonomous in `e`, so running it
// over `e₁` and then `e₂` is running it over `e₁+e₂`; exposures add across the
// overlapping quads of consecutive segments, so the whole stroke applies the
// continuous path integral whatever the spacing.
//
// Getting this wrong is not subtle. Solving only *half* the pair — the canvas relaxing
// towards a tool that never takes anything back, `1 − exp(λ_lift·e)` lifted off and
// `k_dep(exp(λ_dep·e) − exp(λ_lift·e))/(λ_dep − λ_lift)` laid on — and then letting the
// tool lift from the region as the `deposit` has already left it makes the two sides
// disagree about how much changed hands by `O(lift²)` per segment. Paint that goes
// nowhere has to show:
//
//   * at `lift = deposit = 0.95` and a segment of half a radius, 39% of the total
//     `canvas + tool` height vanished at *every* segment boundary. What survives is
//     front-loaded into the first quad that touches a spot, which draws arcs at
//     exactly the segment spacing through anything thick enough to see them.
//   * a stroke that stops leaves its last extent stripped by a lift the tool was
//     never credited with — a tip-shaped disc of missing paint, worst where `lift` is
//     high, and unmistakable because the tip's own hardness edge is frozen into it.
//
// **The two directions are solved as decoupled, not as a closed pair, and that is a
// statement about the geometry rather than a linearisation.** The pair above describes
// a canvas point and *one* reservoir cell held over it for the whole segment, which is
// not what happens: the point slides through a whole stream of cells, each pairing
// lasting the instant the cell is overhead. Over any single pairing the partner cannot
// meaningfully relax, so what the point sees is `−k_lift·h` with the `+k_dep·R` supplied
// by a *different* cell each moment — and symmetrically for the cell. Taking that limit
// splits the system into two one-sided decays:
//
//     h(e) = h₀·exp(−k_lift·e)          each side's own loss, closed form
//     R(e) = R₀·exp(−k_dep·e)
//
// with each side's *gain* the integral of the other's loss over the track they shared.
//
// **This conserves exactly, and the proof is a change of variable rather than an
// appeal to the matrix being stochastic.** Take the deposit direction. Cell `u` loses
// `R₀(u)(1 − exp(−k_dep·τ(u)·lr))`; the canvas gains
// `∫ k_dep·τ(x−p)·R₀(x−p)·exp(−k_dep·τ(x−p)·p) dp` along the pass. Integrating that
// over `x` and substituting `u = x − p` gives back `∫R₀(u)(1 − exp(−k_dep·τ(u)·lr))du`
// — the tool's loss, to the last term. The lift direction is the mirror image, with
// `q = P(1) − P(u)` as the substitution: `∫k_lift·τ(u)·exp(−k_lift·q)du` telescopes to
// `1 − exp(−k_lift·E)`, which is exactly the fraction the canvas gave up. So each
// direction balances on its own, and neither needs the other to be evaluated at the
// same instant for the books to close.
//
// What this buys over the pair is that the pair's `keep` **saturates at `k_lift/s`** —
// half the canvas retained however long the segment, because the tool it is coupled to
// keeps handing paint back — and that plateau is modelling a coupling that is not
// there. The sliding form runs to zero instead. Measured, it converges to the same
// answer as the pair and gets there faster.
//
// What it does *not* fix is the mean-field freeze either side of it: the canvas the
// tool lifts from is still the segment's entry canvas, so the lift never sees the
// deposit the same pass laid down. That residue is `O(k_lift·k_dep·e²)` — second order
// in the rates, where the pair kernel's splitting error is first — and it is what
// `RESERVOIR_EXCHANGE_STEP` is left bounding.
//
// The one thing to be careful of: the two sides evaluate their exponentials at
// *different* quadratures of the same exposure (the canvas differences the prefix-τ,
// the tool takes `τ(u)·lr`), and an exponential that saturates harder turns the same
// small quadrature disagreement into a larger height disagreement than the pair
// kernel's gentler `1 − k·w(e)` would. This is *not* a loss of column-stochasticity:
// the shares sum to one exactly, and it is the exposures underneath them that only
// agree approximately.

// How one segment's exposure `e` splits the two boxes. Every field is a fraction of an
// entering quantity, and the tool's share is the complement of each — so the canvas
// takes `keep·h + dep·R + add_w·A` and the tool takes `(1−keep)·h + (1−dep)·R + src_w·A`,
// which sum to `h + R + e·A`: everything the pair started with, plus everything the
// source minted, and not a texel of height more or less.
struct Exchange {
    // Of the canvas's own height, what is still on the canvas.
    keep: f32,
    // Of the tool's entering load, what is now on the canvas.
    dep: f32,
    // Canvas height laid per unit of the brush's own `add` rate…
    add_w: f32,
    // …and what the tool lifted straight back off it, so that `add_w + src_w == e`.
    src_w: f32,
}

// The tool's share of the minted source, as a fraction of the exposure:
// `ψ(x) = 1 − (1 − e^{−x})/x` at `x = k_lift·e`, summed as its own series
// `Σ_{n≥1} (−1)^{n+1}·xⁿ/(n+1)!` rather than as that difference.
//
// **The difference is the whole bug.** `(1 − e^{−x})/x` is a quotient of two
// quantities that both vanish with `x`, and subtracting it from 1 throws away every
// digit the quotient's leading `1 − x/2 + …` occupies: at `lift = 0.002`
// (`x ≈ 3e−4` for a segment's exposure) the true share is `x/2 ≈ 1.5e−4` and the
// error in the difference is a whole float ulp of 1 — `6e−8`, but *of the wrong
// sign about half the time*, and `ψ` is what the tool takes **back** off the canvas.
// A negative share is the tool minting height it never lifted, which is the one thing
// this loop's books cannot absorb: `deposit` empties the reservoir every segment, so
// with a high `deposit` the residue is not a rounding error on a large load, it *is*
// the load. The reservoir goes negative, and every reciprocal downstream — the
// tool's own `m / max(h, PAINT_EPS)`, the region's `rc.rgb / rc.a`, the deposit's
// `latm / max(tool_m, PAINT_EPS)` — is then a quotient by a negative or vanishing
// denominator, which is the wild color that prints as a speckle of blown-out texels.
// Reproduced at `lift = 0.002, deposit = 0.95`; clean at `lift = 0`, at `lift = 0.5`
// and at any small `deposit`, which is the fingerprint of a cancellation and not of
// the physics (`a_barely_lifting_brush_does_not_mint_on_its_tool`).
//
// The series has no such digits to lose: every term is computed from `x` alone, so
// the result carries `x`'s own relative accuracy and is non-negative for `x ≥ 0`
// **by construction** rather than by a clamp. Truncated after `x⁶/5040`, which is
// under an ulp of the result at the crossover below.
fn src_share(x: f32) -> f32 {
    return x * (1.0 / 2.0
        - x * (1.0 / 6.0
        - x * (1.0 / 24.0
        - x * (1.0 / 120.0
        - x * (1.0 / 720.0
        - x * (1.0 / 5040.0))))));
}

// Where the series hands over to the closed form. Not a tolerance: it is the point
// either side of which the *other* form is the conditioned one. Below it the series
// converges in a handful of terms and the difference cancels; above it `1 − e^{−x}`
// has left the cancellation behind (it is already `≥ 0.39`) while the series needs
// ever more terms. Both forms are good to an ulp here, so nothing steps across it.
const SRC_SERIES_MAX_X: f32 = 0.5;

fn exchange_at(e: f32) -> Exchange {
    let kl = lift_rate(); // the lift rate; 0 when the brush does not lift
    let kd = deposit_rate(); // the deposit rate
    var out: Exchange;
    // Each side's own one-sided decay over the exposure it saw. No `s = kl + kd`
    // anywhere: the two directions do not share a relaxation, which is the whole point.
    out.keep = exp(-kl * e);
    out.dep = 1.0 - exp(-kd * e);
    // The source rides the lift's decay and nothing else — `∫₀ᵉ A·exp(−k_lift·(e−q))dq`
    // — so what settles is `(1 − exp(−k_lift·e))/k_lift` per unit rate, and the tool
    // lifted the rest of the `e·A` minted back off before it set.
    //
    // **Whichever of the two shares is the small one is the one computed**, and the
    // other is the remainder — so the subtraction only ever takes a small number off a
    // large one and neither share can come out negative. A brush that does not lift
    // keeps all of it, which is the limit of that quotient and not a special case in
    // the physics: `x = 0` takes the series with every term zero, so the old
    // `kl < 1e-6` guard against a 0/0 has nothing left to guard.
    let x = kl * e;
    if x < SRC_SERIES_MAX_X {
        out.src_w = e * src_share(x);
        out.add_w = e - out.src_w;
    } else {
        out.add_w = (1.0 - out.keep) / kl; // `x ≥ 0.5` and `e > 0`, so `kl > 0`
        out.src_w = e - out.add_w;
    }
    return out;
}

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
// multiplies: `src_m = brush_op·drain·src_h` lands mass above the height carrying it,
// and `m ≤ h` is the invariant the region's storage rests on, since what it stores is
// `m / h`. Both paths bound it identically (`stamp_common.wesl::stroke_drain`), so a
// drained brush lays the same start cap whichever one draws it.
fn stroke_drain(arc: f32) -> f32 {
    return clamp(1.0 - st.drain * arc, 0.0, 1.0);
}

// The share of the tip's contact a canvas texel gets from the substrate under it (§6.4)
// — the same `rise_ahead`/`tooth_gate` pair the swept fast path applies, from the same
// map, so a brush's tooth reads the same whether or not some other axis put it on this
// path.
//
// `rt` is in region px; `st.substrate_uv_bias` folds the piece's own canvas origin into
// the map, so the value is a pure function of canvas position exactly as the fast
// path's is.
// `turn` is how far the tip's heading has swung by this point (`segment_frame(rt)`),
// which `sweep_tangent` turns into the direction the rise is read along —
// after the guards, so an ordinary brush never pays for the rotation. A settle slot
// carries zero travel, so its heading is the direction the pen was going when it left
// the page, which is the substrate the pass it is finishing was laying through.
//
// Both guards return exactly 1.0: a brush with no tooth, or a substrate with no relief,
// leaves every rate below the float it always was. The first is also what keeps the
// **bleed slots** clear of all of this — they carry `tooth_give = 1` (`bleed_fires`), so a
// lateral flux is never gated by the substrate and never reads a direction. That is not
// bookkeeping: bleed is wet paint spreading sideways on the canvas, not a tip dragged
// over it, so there is no travel to read the substrate's rise along, and gating it would
// break the flux's antisymmetry (the two threads of a pair sit over different substrate).
fn substrate_tooth(rt: vec2<i32>, turn: f32) -> f32 {
    if st.substrate_uv_scale <= 0.0 || st.tooth_give >= 1.0 {
        return 1.0;
    }
    let uv = vec2<f32>(rt) * st.substrate_uv_scale + st.substrate_uv_bias;
    let dims = textureDimensions(substrate_tex);
    let g = textureLoad(substrate_tex, substrate_texel_at(uv, dims), 0);
    return tooth_gate(rise_ahead(g, sweep_tangent(st.dir, turn)), st.tooth_give, st.tooth_softness);
}

// The deposit jitter's gate at region texel `rt` (§6.2) —
// `lib::noise::jitter` at the absolute canvas texel, so a live piece, its
// commit and a replay read the identical gate however the region is placed, and
// the swept path's `jitter_of` reads the same one (the loop's `add` axis
// is that path's deposit, and the two must lay the same paint).
//
// Applied wherever `substrate_tooth` is — it is the same shape of gate, a
// per-texel factor on the exposure — and nowhere else. Never on the bleed pair,
// whose antisymmetry a per-texel gate would break exactly as the tooth would
// (`bleed_weight`); and never on the tool side, which books the gate's mean the
// way it books `tooth_bearing` — and the jitter's mean is exactly 1, so the
// tool's dispatch does not change at all.
fn jitter_at(rt: vec2<i32>) -> f32 {
    return jitter(st.canvas_origin + rt, st.jitter_seed, st.jitter_eps);
}

// The extent's **optical depth** for a given coverage — the currency every
// transfer in this loop is measured in, on both sides. The canvas side gets it from
// the prefix-τ volume's differences; the tool side, which has no prefix to
// difference, converts its own coverage the same way the volume was built
// (`assets::build_prefix_tau`), clamp included, so the two agree.
fn tau_of(cov: f32) -> f32 {
    return -log(1.0 - min(cov, 0.999));
}

// Extent coverage at brush-local `local` ∈ [-1,1]² (0 outside — image masks may
// have non-zero borders, so the clamp sampler alone can't be trusted at the edge).
fn coverage_at(local: vec2<f32>) -> f32 {
    if any(abs(local) >= vec2<f32>(1.0, 1.0)) {
        return 0.0;
    }
    return textureSampleLevel(cov_tex, samp, local * 0.5 + vec2<f32>(0.5, 0.5), 0.0).r;
}

// (`blend_latent` — the parcel-stacking law — is shared with the transform's
// combine and lives in paint_common.wesl.)

// Bilinear lookup of a baked prefix row at travel-frame `(x, y)`, with the same
// asymmetric edge semantics as `prefix_at`: 0 before the tip's leading edge, and
// clamped to the row total after its trailing edge. Hand-rolled on `textureLoad`
// for the same reasons the prefix-τ lookup is, which is why the tap arithmetic is
// shared with it (`lib::sample::bilerp_taps`); only the leading-edge guard below
// is this lookup's own.
fn bake_at(tex: texture_2d<f32>, x: f32, y: f32) -> vec4<f32> {
    if x <= -1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let b = bilerp_taps(vec2<f32>(textureDimensions(tex)), vec2<f32>(x, y));
    return bilerp_mix(
        b,
        textureLoad(tex, vec2<i32>(b.i0.x, b.i0.y), 0),
        textureLoad(tex, vec2<i32>(b.i1.x, b.i0.y), 0),
        textureLoad(tex, vec2<i32>(b.i0.x, b.i1.y), 0),
        textureLoad(tex, vec2<i32>(b.i1.x, b.i1.y), 0),
    );
}

// (The `wick` pass — pass 0 of this loop, a lateral flux letting the tool's paint
// migrate within the tip — stood here from 2026-07-31 to 2026-08-10. It existed to
// suppress the lift-end shoulder ring `golden_lift_end_regression` pins; once the
// pen-up `settle` became the remaining pass's exact delivery integral, the ring no
// longer returns without it — measured on a fifteen-radius drained smear, wick on
// vs off: no lateral rise above 1 level, fade past the stroke end monotone both
// ways, worst 4 levels anywhere. §6.2 records the history and the measurement.)

// ---- 1. Bake: the reservoir's swept prefix along the travel axis. ----------------
//
// During a segment the tip does not present one texel to a canvas point — it drags
// a whole *range* of texels across it. A point at travel-frame `(xl, yl)` is served
// by tool texel `(xl − s, yl)` as the tip advances by `s`, so what it receives is
//
//     ∫₀^lr R(xl − s, yl) · τ(xl − s, yl) ds
//
// a convolution along travel between the reservoir `R` and the extent's
// optical-depth density. Sampling `R` once mid-pass would be a one-tap midpoint rule
// for that integral: first order, with error growing in segment length, and so
// tradeable only against step size.
//
// The integral has the same shape as the exposure itself, so it yields to the same
// trick: **`prefix_at` is this integral with `R ≡ 1`**. Accumulate `R·dτ` along the
// row and the deposit becomes a difference of two prefix sums — O(1), and exact for
// any segment length. Baked per segment, so the tip's orientation (the reservoir
// lives in the shape frame, §6.6) and the reservoir's own evolution are both folded
// in with no approximation. `dτ` comes from the prefix-τ volume's own differences,
// so bake and exposure integrate numerically identical extents.
//
// One **workgroup per row**, one thread per column, with the running integral
// built by a shared-memory inclusive scan (Hillis–Steele, log₂(BAKE_RES) steps).
// One thread walking its whole row instead puts 128 threads on the device per
// dispatch and a 128-iteration serial chain of texture reads on each — and since
// the bake runs once per segment inside the sequential stamp loop, that near-idle
// dispatch dominates the whole dynamics path. The scan computes the same prefix
// sums (up to float re-association) with every lane of a full-width dispatch busy.

// Must equal the host's BAKE_RES (the bake textures' side).
const BAKE_RES: u32 = 128u;

var<workgroup> ws_e: array<f32, BAKE_RES>;
var<workgroup> ws_load: array<vec4<f32>, BAKE_RES>;
// The mass-weighted color, both halves in one lane (§6.7, `lib::latent`). The residual
// is scanned in the same sweep and costs the same 2 KB per column it always did — 6.5 KB
// of the 16 KB downlevel workgroup budget at BAKE_RES = 128, so the channel costs the
// scan no width — but the scan itself no longer knows there are two of them.
var<workgroup> ws_lm: array<Latent, BAKE_RES>;

@compute @workgroup_size(128) // = BAKE_RES: thread i scans column i of its row
fn bake(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let dim = textureDimensions(bake_load_w);
    let row = wid.y; // dispatched as (1, dim.y, 1), so every row is in range
    let i = lid.x;
    let yl = (f32(row) + 0.5) / f32(dim.y) * 2.0 - 1.0;
    let theta = st.orientation * TAU;
    let cs = cos(theta);
    let sn = sin(theta);

    // This column's share of the exposure: the prefix-τ difference across it.
    // Each thread reads its right edge; the left edge is the neighbour's.
    let x1 = (f32(i) + 1.0) / f32(dim.x) * 2.0 - 1.0;
    ws_e[i] = prefix_at(x1, yl, st.orientation);
    workgroupBarrier();
    var de = ws_e[i];
    if i > 0u {
        de = de - ws_e[i - 1u];
    }
    de = max(de, 0.0);

    // The tool texel over this column. The reservoir is stored in the shape
    // frame, so rotate out of travel first.
    let xc = (f32(i) + 0.5) / f32(dim.x) * 2.0 - 1.0;
    let sl = vec2<f32>(xc * cs + yl * sn, -xc * sn + yl * cs);
    let buv = clamp(sl * 0.5 + vec2<f32>(0.5, 0.5), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let b_c = textureSampleLevel(brush_src_color, samp, buv, 0.0);
    let b_a = textureSampleLevel(brush_src_aux, samp, buv, 0.0);

    // Carried height and optical mass (opacity·height), each weighted by the
    // exposure they are presented with; latent rides as a mass weight so a
    // ratio of the two prefixes recovers it (§6.1). The bare exposure rides in
    // `.z`, so the prefix `P` itself comes back with the same reads — the segment
    // deposit never looks at it, but the pen-up `settle` walks its delivery
    // integral against exactly this row.
    let m = b_c.a * b_a.x;
    ws_load[i] = vec4<f32>(b_a.x, m, 1.0, 0.0) * de;
    // The residual rides the identical mass and the identical exposure weight, so a
    // ratio against `bake_latm`'s own denominator recovers it (§6.7) — which one lane
    // makes true by construction rather than by two lines agreeing.
    var b_col = latent_of(b_c.rgb, vec3<f32>(0.0));
    @if(resid) b_col.res = textureSampleLevel(brush_src_resid, samp, buv, 0.0).rgb;
    ws_lm[i] = latent_scale(b_col, m * de);
    workgroupBarrier();

    // Inclusive scan over the accumulators together.
    for (var off = 1u; off < BAKE_RES; off = off << 1u) {
        var v_load = ws_load[i];
        var v_lm = ws_lm[i];
        if i >= off {
            v_load += ws_load[i - off];
            v_lm = latent_add(v_lm, ws_lm[i - off]);
        }
        workgroupBarrier();
        ws_load[i] = v_load;
        ws_lm[i] = v_lm;
        workgroupBarrier();
    }

    // Out-of-range stores (a texture narrower than the workgroup) are no-ops.
    let t = vec2<i32>(i32(i), i32(row));
    textureStore(bake_load_w, t, ws_load[i]);
    textureStore(bake_latm_w, t, vec4<f32>(ws_lm[i].lat, 0.0));
    @if(resid) textureStore(bake_rlm_w, t, vec4<f32>(ws_lm[i].res, 0.0));
}

// ---- 2. Snapshot the region under the segment's extent rect. -----------------

// This texel's place in the segment's travel frame — the one the whole exchange is
// measured in. Straight or curved, `sweep_at` hands back the same three quantities
// (see stamp_common.wesl), so nothing downstream branches on the segment's shape.
//
// `st.travel_radii` carries the travel as `length / radius`, so the arc length it was built
// from is recovered by multiplying back through.
//
// **The single place the loop builds a frame**, which is what lets the radius ramp
// (§6.2, `Segment::ramp`) be applied in exactly one line: `deposit`, its coarse twin,
// the bleed flux, `snapshot` and `settle` all come through here, so all five agree
// about where the sweep's rim is — and `outside_sweep`, which several of them use to
// skip work the others must then also skip, is computed from the same frame. Applying
// it anywhere further down would be five chances to disagree.
//
// The tool *reservoir* deliberately does not ramp: `bake` and `exchange` place a
// brush-local grid, which is the tool's own and does not stretch as the tip does, and
// they reach the canvas through `st.frame_radius` rather than through a `Sweep` at all.
// What ramps is the canvas-side coverage — the prefix-τ difference, which is what the
// stroke's outline is made of.
fn segment_frame(rt: vec2<i32>) -> Sweep {
    let p = vec2<f32>(rt) + vec2<f32>(0.5, 0.5);
    let sw = sweep_at(p, st.start, st.dir, st.frame_radius, travel_px(), st.curvature);
    return ramp_apply(sw, st.radius_ramp);
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

// One extent texel's copy. Factored out because the segment loop runs this from the
// tail of `exchange`'s grid rather than from a dispatch of its own — see there — while
// the pen-up still needs it as a standalone pass.
fn snapshot_at(gx: u32, gy: u32) {
    let udim = textureDimensions(under_color_w);
    if gx >= udim.x || gy >= udim.y {
        return;
    }
    let t = vec2<i32>(i32(gx), i32(gy));
    let rt = st.rect_origin + t;
    if outside_sweep(segment_frame(rt)) {
        return; // deposit never reads these texels of the snapshot
    }
    let rdim = vec2<i32>(textureDimensions(region_color));
    var c = vec4<f32>(0.0);
    var a = vec4<f32>(0.0);
    @if(resid) var r = vec4<f32>(0.0);
    if rt.x >= 0 && rt.y >= 0 && rt.x < rdim.x && rt.y < rdim.y {
        c = textureLoad(region_color, rt, 0);
        a = textureLoad(region_aux, rt, 0);
        @if(resid) r = textureLoad(region_resid, rt, 0);
    }
    textureStore(under_color_w, t, c);
    textureStore(under_aux_w, t, a);
    @if(resid) textureStore(under_resid_w, t, r);
}

// The standalone pass, which only the pen-up now dispatches: the `settle` reads the
// snapshot, so unlike the segment case it cannot share a dispatch with its producer.
@compute @workgroup_size(8, 8)
fn snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    snapshot_at(gid.x, gid.y);
}

// (**There is no sub-step count here, and its absence is the measurement.** The canvas
// side gets its sweep exactly, from the prefix-τ; the tool side has no prefix to
// difference — the canvas under it is an arbitrary evolving field — so its exposure is a
// quadrature, and the question was how fine. An `EXCHANGE_STEPS` constant stood here at
// 1, with a loop and a divide that could not move a pixel until someone raised it.
//
// The sliding kernel puts no obstacle in the way of raising it: each direction is a
// plain exponential in exposure, and `exp(−k·e/N)^N = exp(−k·e)` exactly, so splitting
// the exposure changes the tool's own arithmetic not at all. All N > 1 buys is sampling
// the canvas at several points along the texel's track instead of at its midpoint, and
// that is measured — both as sub-steps and as an average over the track — at less than
// the 8-bit noise floor. The midpoint tap is simply not where the error is;
// `RESERVOIR_EXCHANGE_STEP` is. So the loop is gone rather than pinned, and this is the
// note that says what it would take to bring it back: a per-sub-step share of the
// exposure, the tip's frame re-evaluated at each one, and a reason.
//
// A closed-pair kernel would forbid the split outright — over each sub-step the tool
// lifts its share of a canvas held fixed, N times over, while the `deposit` gives up a
// single share of the whole exposure, so the two halves stop being complements of one
// solution and the transfer stops balancing. That the sliding form permits it is worth
// recording even while nothing uses it.)

// `sin(x)` and `1 − cos(x)` as the same truncated Maclaurin series `path::arc_at` uses,
// and for the same reason it does: the turn across one segment is *tiny* — the host
// caps `radius · |curvature|` at `MAX_TIP_TURN` — and `1 − cos(x)` formed by
// subtraction at that scale cancels away most of its digits before it is divided
// through by an equally tiny curvature. Evaluating the versine directly is what keeps
// the quotient meaningful.
//
// Getting this wrong is not a rounding difference, it is garbage: the position comes
// out arbitrarily far from the tip, `exchange` lifts from somewhere else entirely, and
// the reservoir fills with paint that was never under the brush. It shows as bright
// streaks combed out along the stroke.
fn sin_small(x: f32) -> f32 {
    let x2 = x * x;
    return x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 / 5040.0)));
}

fn versin_small(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * (0.5 - x2 * (1.0 / 24.0 - x2 / 720.0));
}

// The tip's centre and travel tangent a fraction `f` along this segment — the arc
// `path::arc_at` walks, run forwards, with the same sign convention (curvature turns
// towards the left of `dir`). A straight segment never touches the arc arithmetic.
fn tip_at(f: f32) -> vec4<f32> {
    let start = st.start;
    let dir = st.dir;
    let s = f * travel_px(); // travel = (length / radius) · radius · f
    let k = st.curvature;
    if k == 0.0 {
        return vec4<f32>(start + dir * s, dir.x, dir.y);
    }
    let perp = vec2<f32>(-dir.y, dir.x);
    let turn = k * s;
    let sn = sin_small(turn);
    let vs = versin_small(turn);
    return vec4<f32>(
        start + dir * (sn / k) + perp * (vs / k),
        dir.x * (1.0 - vs) + perp.x * sn,
        dir.y * (1.0 - vs) + perp.y * sn,
    );
}

// ---- 3. Exchange: the tool's own side of the transfer, per segment. -------------
//
// One thread per **tool reservoir** texel, run *before* the `deposit` it is the other
// half of, so that both read the canvas and the reservoir as the segment found them.
// It takes the complement of every share `exchange_at` hands the canvas:
//
//   * **retain** `1 − dep` of its entering load — the rest the deposit has on the
//     canvas. Latent and per-unit opacity are untouched: the tool holds *less* paint,
//     not different paint (§6.1).
//   * **lift** `1 − keep` of what is under it — which is what makes smearing
//     genuinely order-dependent, including across the stroke's own trail, since the
//     region it reads is the one earlier segments have already worked over.
//   * **take back** `src_w` of the brush's own `add`, the share of the newly minted
//     paint the tip lifts straight off again. Without it a high-`lift` brush would
//     mint `A·e` of height, lay `add_w·A` and destroy the difference.
//
// All three share the one exposure, and it is the *same* segment travel the canvas
// side integrates over, so nothing the canvas gave up waits to be credited. That is
// the whole reason this runs per segment rather than on a cadence of its own: reload
// the tool every `spacing·radius` while the canvas is stripped every segment and a
// stroke ending mid-interval leaves the difference behind as a tip-shaped patch of
// missing paint.
//
// **The tool's side is swept too.** A reservoir texel does not hover over one spot for
// a segment — it is dragged along a track `lr` radii long, over a canvas that is
// changing under it, and the canvas side integrates over exactly that track through
// the prefix-τ. A single bilinear tap at the segment's midpoint would be the midpoint
// rule, second-order in the segment length, and the one thing in this loop whose
// answer depended on how finely the path happened to be cut.
//
// That dependence is visible, and it is *not* the kind that hides behind a tolerance:
// the flattener's segment lengths follow the pointer samples, and a hand varies its
// speed, so the same gesture drawn twice would cut the tool's decay differently and
// the smear would fade at a different rate. Measured on a pure smear dragged out of a
// band of paint, the same 300px stroke drawn with 2, 6, 20 and 80 pointer samples came
// out up to 11 levels apart, and 15% away from the converged answer.
//
// So the canvas is read where the texel actually **is** — at the midpoint of the track
// it is dragged along this segment, with the tip's frame evaluated there, so a curved
// sweep turns under the tool as it goes. Reading it at the segment's *start* instead is
// the error that measurement found; splitting the track finer than its midpoint is not
// (see the note above `exchange_at`'s callers).
// **The segment's `snapshot` rides in the tail of this dispatch.** The two are
// independent — the snapshot only reads the region, which the previous segment's
// `deposit` last wrote, and writes the `under` scratch; the exchange reads that same
// region and writes the reservoir — so a barrier between them would pay for an
// ordering neither needs. Both must finish before `deposit`, and sharing one dispatch
// is the cheapest way to say exactly that.
//
// Workgroups `[0, ex_groups)` on x are the reservoir; the rest are the extent,
// re-based onto its own grid. The grid's y extent is whatever the taller of the two
// needs, so a few groups fall outside both and return on the bounds checks — a handful
// of early-out threads against one fewer serialized dispatch per segment, out of five.
@compute @workgroup_size(8, 8)
fn exchange(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bdim = textureDimensions(brush_src_color);
    let ex_groups = (bdim.x + 7u) / 8u;
    if wid.x >= ex_groups {
        snapshot_at((wid.x - ex_groups) * 8u + lid.x, wid.y * 8u + lid.y);
        return;
    }
    let gid = wid.xy * 8u + lid.xy;
    if gid.x >= bdim.x || gid.y >= bdim.y {
        return;
    }
    let t = vec2<i32>(gid);
    let local = (vec2<f32>(t) + vec2<f32>(0.5, 0.5)) / vec2<f32>(bdim) * 2.0 - vec2<f32>(1.0, 1.0);

    let b_c = textureLoad(brush_src_color, t, 0); // (latent, op) — straight, not premult
    let b_a = textureLoad(brush_src_aux, t, 0);   // (height, _, _, _)
    var h = b_a.x;
    var m = b_c.a * h;
    // The reservoir's carried color, both halves — straight, not premultiplied
    // (§6.7, `lib::latent`). The gate is on the load; the two blends below are one call
    // each.
    var col = latent_of(b_c.rgb, vec3<f32>(0.0));
    @if(resid) col.res = textureLoad(brush_src_resid, t, 0).rgb;

    // This texel's exposure to the whole segment: its own optical depth × the travel
    // (radius units). In τ, like the canvas side — the canvas gives up
    // `1 − keep(∫τ)`, so a tool that measured its gain in bare coverage would
    // receive a fraction of what the canvas lost and quietly destroy paint.
    //
    // Scaled by the **bearing fraction** (§6.4): the tip only trades where it touches,
    // and a reservoir cell has no substrate of its own to ask — it is dragged over a
    // fresh stretch of canvas at every sub-step. So it books against the substrate's
    // mean *for the way this segment is going*, which the host read off the substrate's
    // per-direction table (`Surface::bearing`, `st.tooth_bearing`), while each canvas
    // texel scales
    // by the substrate ahead of itself. The two agree in expectation over any extent spanning many
    // grain features — which is every usable tip — and that is what keeps a toothed
    // smear conserving. Exactly 1 when there is no tooth, so an ordinary brush is
    // untouched.
    //
    // **Both halves are in the mask's units, not the frame's** (§6.6). A reservoir texel
    // tiles the frame, which for a padded volume is `st.frame_scale` tips wide — so the
    // mask under it sits at `local · st.frame_scale`, and the track it is dragged along is
    // that many more of the tip's own radii than of the frame's. The canvas side is
    // already in mask units, its prefix having been integrated at the mask's column
    // width. Exactly 1 for an unpadded volume, so nothing here moves for any brush but a
    // nib.
    //
    // Left in the frame's units the tool weights itself by a mask stretched over its own
    // padding — picking up from an extent `st.frame_scale` wider than the one it lays —
    // and books the exposure for it in the wrong units on top. What that does *today* is
    // very little, and the reason is worth stating so nobody reads the correction as
    // load-bearing where it is not: the paint it misplaces lands in the padding, and
    // `deposit` reads the reservoir back through `bake`'s τ-weighted prefix, which is
    // zero out there. So it is stranded rather than deposited, and a zero-angle nib
    // renders within 2 levels either way (`assets.rs`). It is written correctly because
    // the *representation* has to be consistent — the day anything reads the reservoir
    // by something other than the mask's own τ, a tool a third too wide stops being
    // invisible, and nothing about that day would point back here.
    //
    // **And by the tip's stretch** (§6.6), which is the third thing in the same
    // sentence: a mask texel on a tip drawn out along its facing axis stands over more
    // canvas, so the track it crosses is that many fewer of its *own* widths. The
    // factor is `travel` and not `1/travel` for the reason the canvas side's is the
    // reciprocal — the two are opposite sides of one substitution, and it is what makes
    // the books balance: a stretch multiplies the canvas's total exposure by `s` and
    // the area each reservoir texel answers for by the same `s`.
    let e = tau_of(coverage_at(local * st.frame_scale))
        * st.travel_radii * st.stretch.x * st.frame_scale * st.tooth_bearing;
    let ex = exchange_at(e);
    let rdim = vec2<f32>(textureDimensions(region_color));
    // Where this reservoir texel rides, in the tip's own reference frame — the two
    // steps `bake` takes to get here, run backwards. The reservoir is stored in the
    // **shape** frame, so turn it by the orientation into the frame the sweep is looked
    // up in (`bake`'s own rotation, inverted), then take the stretch back out of that
    // (`stretch_unlook`) to land in the frame the travel tangent below spans. Both are
    // per-texel constants, so they are lifted out of the sub-step loop that used to
    // turn the tangent instead.
    let theta = st.orientation * TAU;
    let cs = cos(theta);
    let sn = sin(theta);
    let seat = stretch_unlook(
        vec2<f32>(local.x * cs - local.y * sn, local.x * sn + local.y * cs),
        st.stretch,
    );

    {
        // The midpoint of the track this texel is dragged along.
        let f = 0.5;
        let tip = tip_at(f);
        let dir = tip.zw;
        let world = tip.xy + (seat.x * dir + seat.y * vec2<f32>(-dir.y, dir.x)) * st.frame_radius;
        let uv = world / rdim;

        // Outside the selection the canvas gives nothing up and the source mints
        // nothing (§6.8) — the matching reduction on the canvas side is the deposit's
        // own mask lerp, so height stays conserved. Bilinear like the region taps
        // below: this texel is over an arbitrary sub-pixel spot, and a 1×1 constant
        // mask reads the same at every uv.
        let sel = textureSampleLevel(sel_mask, samp, uv, 0.0).r;
        let rc = textureSampleLevel(region_color, samp, uv, 0.0);
        let ra = textureSampleLevel(region_aux, samp, uv, 0.0);
        // The identical bilinear tap at the identical uv — the residual rides the
        // color it belongs to and never its own weights.
        var rcol = latent_of(rc.rgb, vec3<f32>(0.0));
        @if(resid) rcol.res = textureSampleLevel(region_resid, samp, uv, 0.0).rgb;

        // What it keeps of its own load. Exactly the complement of the fraction the
        // `deposit` after us lays from it, over the very exposure it lays over — so
        // there is no correction factor.
        //
        // **No fudge factor belongs here.** A constant below 1 ("deplete a bit slower
        // to account for error") means the tool keeps a fraction of every deposit it
        // makes, which compounds — paint grows without bound over a long stroke. Such
        // a factor only ever compensates for the two sides solving *different*
        // equations and disagreeing about how much changed hands; with both taking
        // complementary shares of one solution there is nothing to correct, and the
        // exact rate is also the conserving one.
        h = h * (1.0 - ex.dep);
        m = m * (1.0 - ex.dep);

        // Height moves; color and per-unit opacity ride as optical-mass-weighted
        // blends (§6.1). Two parcels land on the tool — what it took off the canvas,
        // and the share of the brush's own `add` it lifted back before that paint had
        // settled — each stacked with its own latent, so a tip smearing red through
        // white carries the mixture rather than an average of the two errands.
        let take_h = ra.x * (1.0 - ex.keep) * sel;
        let take_m = rc.a * take_h;
        h = h + take_h;
        // Only mix a color in when there is optical mass to carry it. The region is
        // stored premultiplied, so recovering a latent is a division by the opacity —
        // and out in a stroke's faded tail that opacity is a rounding error, so the
        // quotient is enormous. `blend_latent` weights it by its own vanishing mass,
        // but "enormous × almost nothing" is not nothing: it leaves a few tenths of a
        // wild color on the tool, which the deposit then combs out along the stroke as
        // thin bright streaks. Walking the track meets that case more often than a
        // single mid-segment tap would: it samples the ends as well, where the canvas
        // is emptiest, and eight times as often — hence the guard.
        if take_m > PAINT_EPS {
            // The same stack on the same two masses for both halves, against the
            // *entering* `m` — one call, so it cannot be the identical blend for one
            // half and something else for the other (§6.7).
            col = latent_blend(col, m, latent_div(rcol, rc.a), take_m);
        }
        m = m + take_m;

        // The brush's remaining load **where this texel actually is**, not where the
        // segment starts. The reservoir lives in the shape frame, so turning its
        // position back into travel — the inverse of the `rot` above, which works out
        // to `local.x·cos − local.y·sin` — gives its offset along the stroke, and the
        // tip's own travel this far into the segment gives the rest.
        //
        // It has to be this point and not the segment's, or the two halves of the
        // source stop balancing: the `deposit` hands the canvas `add_w · A` at the
        // canvas texel's arc, and this takes back `src_w · A` at the tool texel's. The
        // two are the same place at the same moment, so they must read the same `A`.
        let arc = st.arc_at_start + f * travel_px() + (local.x * cs - local.y * sn) * st.frame_radius;
        let drain = stroke_drain(arc);
        let src_h = st.add * drain * ex.src_w * sel;
        let src_m = st.brush_op * drain * src_h;
        // The brush's own paint. Its residual is **not** jittered, for the reason
        // `stamp.wesl` gives — color dynamics perturb the pigment mixture, and the
        // residual is precisely what no mixture of those pigments reaches — and here
        // neither half is, so the pair comes straight off the uniform, which declares
        // the residual whether or not the space uses it (§6.10): no gate at all.
        col = latent_blend(col, m, latent_of(st.brush_lat, st.brush_res), src_m);
        h = h + src_h;
        m = m + src_m;
    }

    textureStore(brush_dst_color_w, t, f16_nearest4(vec4<f32>(col.lat, m / max(h, PAINT_EPS))));
    textureStore(brush_dst_aux_w, t, f16_nearest4(vec4<f32>(h, 0.0, 0.0, 0.0)));
    @if(resid) textureStore(brush_dst_resid_w, t, f16_nearest4(vec4<f32>(col.res, 0.0)));
}

// ---- 4. Deposit: the swept exchange over one segment. ----------------------------

// The `bleed` axis (§6.2): the paint already on the canvas diffusing laterally under
// the tip. One share of a texel's difference from each neighbour crosses per
// **firing**, keyed on the pair's swept exposure over the firing's travel window —
// `1 − exp(−k_bleed·e)`, the same saturating form as every other rate, so
// consecutive windows compose to first order and a texel the tip never covers never
// moves.
//
// **The bleed runs on dedicated slots, not on the painting segments — a travel
// cadence, realised as geometry.** Real slow input is fitted at a control point per
// pointer sample (a measured repro: 177 knots over 68 px, segments of 0.39 px), and
// at that cut a per-segment bleed is broken twice over: the flux
// `share·w·Δ ≈ 1e-4` sits under the f16 ULP of the heights it edits, so stores
// either snap it away or ratchet a whole ULP into a directional drift (20 levels of
// ghost at 176 spans against 2 at 44); and the per-texel exposure `dpre` is a
// prefix difference over 3e-4 radii — cancellation noise, whose arbitrary sign
// draws crescents of misclassified texels through the extent. So the host emits
// one **bleed slot** per crossing of `BLEED_TRAVEL_QUANTUM` of absolute arc
// (`stroke::dynamics::dynamics_plan`): a quad whose sweep IS the firing's travel
// window — the last quantum of path, carrying the crossing segment's own curvature
// so the relaxed band follows the paint rather than cutting the corner off it — with
// every vertical rate and the source zeroed, and λ_bleed alone nonzero. This dispatch
// then needs nothing special: it sweeps an arc the way every painting segment does
// (`sweep_at`), and the slot's own `dpre`, over half a radius of travel, is the window's
// exposure and is numerically healthy; participation, pairing and the no-flux wall
// all sit at the window-sweep's rim, where the weight fades to zero smoothly.
// Keyed on absolute arc, the cadence is a pure function of the record, so every
// cut of the same path fires identically (§6.2, live == committed) — and painting
// segments carry λ_bleed = 0, so between firings the canvas takes the no-bleed
// path bit-for-bit.
//
// **Flux form**: both threads of a pair compute the same number from
// the same `under` snapshot — the neighbour's exposure is re-derived from the same
// uniform, so the same floats — and apply it with opposite signs. The redistribution
// is internal to the canvas, so it conserves height and optical mass to rounding
// (not to the bit: the delta rides through this thread's own
// `keep` multiply and selection lerp below) and needs no entry in the tool's books.
//
// `min(w, w_n)` is the pair's mobility, and it carries three boundaries at once:
// a neighbour outside the sweep has `w_n = 0`, so no paint leaks to a texel whose
// own thread never writes (which is also why the snapshot's `outside_sweep` skip
// still covers every texel this reads); the sweep's rim fades the flux out
// smoothly; and the un-written partner of a pair straddling `e = 0` agrees the
// flux was zero. The neighbour's selection gates its side the same way the final
// `sel` lerp gates this one, so a feathered mask scales the pair's flux by both
// coverages and deselected paint neither gives nor takes (§6.8).
//
// **The axis is a diffusivity, and the reach is how it is spent.** Per unit of
// exposure the stencil injects `σ² ≈ 2·k_bleed·Σ(share·d²)` texels² per axis, so
// what it realises is a diffusion coefficient `D = k_bleed·Σ(share·d²)` — a *rate*
// times a *second moment*. Only one of those two factors has room in it. The blend
// `w = 1 − exp(−k·e)` clips at 1, and a firing can never move more than its share of
// the difference, so at a fixed stencil `D` has a hard ceiling of `Σ(share·d²)` per
// firing however hard the rate is driven. Measured on a 40 px tip, whose window
// carries `e ≈ 1.7`: the whole of `bleed` from 0.95 to 1.0 takes `w` from 0.52 to
// 0.99, ×1.9 in `D` and ×1.4 in distance, and past that nothing. Driving the rate
// alone past its λ-clamp therefore buys nothing visible at all.
//
// That ceiling is not a stability bound, it is the stencil's own geometry, and
// `Σ(share·d²)` is quadratic in the reach. So the host is given the diffusivity and
// **solves for the pair**: it picks the reach that delivers the variance this
// window's travel calls for at a fixed, well-conditioned blend, and derives the rate
// that lands the window's nominal exposure on that blend
// (`stroke::budget::bleed_stencil`). The reach arrives in the slot; `bleed` is then
// linear in `D` across its whole travel, and `σ = sqrt(2·D·τ)` — real diffusion, with
// the distance growing as the root of the scrubbing, like a blender.
//
// Holding the blend near ½ rather than at saturation is the other half of the win.
// The worst-case eigenvalue is `1 − 8·Σshare·w = 1 − w` (Σshare = 1/8): at `w → 1` the
// operator annihilates its worst mode — a hard local average, not a Laplacian — and
// firings stop composing. Half leaves the margin the model is written against.
//
// The reach is quoted as a fraction of the tip's radius rather than in px, which is
// what makes `bleed` mean the same *look* at every brush size — the same property the
// tapers get from being quoted in radii. Its own ceiling is the extent: a tap that
// leaves the sweep has `w_n = 0` and carries nothing, so past about half the radius
// the long tap is truncated for most of the tip and the delivered `D` falls short of
// the asked-for one. Past that the honest move is a finer cadence, not a longer tap.

// **The taps are a ladder up the reach, not a few isolated scales**, and that is what
// makes one firing read as a blur rather than as a copy.
//
// A tap at ±d moves a share of each texel's difference from the texel d away, so on
// its own it lays a *displaced ghost* of the source: against a hard edge — the shape
// this axis is most often asked to soften — the response is that edge again, d away,
// at the tap's share. Three scales (1 px, a quarter of the reach, the reach) put 62%
// of the shed on the longest tap alone, so every firing printed a hard copy of the
// edge a whole reach out, and a texel is only covered by `2/BLEED_TRAVEL_QUANTUM`
// firings — a handful — so the sum kept that structure instead of filling in. A
// bleed-only pass across a painted bar came out as stacked slabs a reach tall, with
// the step between them as sharp as the bar's own edge.
//
// Rungs at `j·reach/BLEED_LADDER_TAPS`, `j = 1..=TAPS`, sharing
// [`BLEED_SHARE_LADDER`] equally, spread that same shed over the whole interval: one
// firing's response to an edge is a staircase of `TAPS` steps of `reach/TAPS` rather
// than one step of `reach`, and consecutive firings — which land at different reaches,
// since the reach follows the window — fill the treads. It costs taps, which the
// firing has room for (the pass is a small share of a stroke's frame; measured at
// radius 500, the whole bleed is ~2 ms of a ~25 ms replay), and it costs *second
// moment*: a uniform ladder carries `(T+1)(2T+1)/(6T²) → 1/3` of what the same share
// carries out at the reach, so the cadence is finer by the same factor to keep the
// diffusivity the knob means (`budget::BLEED_TRAVEL_QUANTUM`).
//
// The 1 px near tap survives all of that, for the reason it was introduced: a stencil
// of ±d taps alone decouples the grid into d² sublattices — the checkerboard
// generalized, whose eigenvalue such a stencil leaves at 1 — so sub-d texture would
// ride through a "blur" untouched. The near taps couple every texel to its actual
// neighbours, so every non-zero frequency strictly decays. Total shed is still bounded
// by `4·Σshare·w = w/2` (Σshare = 1/8, unchanged): no firing can overshoot, and
// positivity of height and mass survives any rate.
//
// Declared as scalars because the **host** computes `Σ(share·d²)` to solve for the
// reach, and a second copy of these numbers over there is a way for the two sides to
// disagree about how much a firing diffuses without anything failing. They are
// generated from these declarations instead (§6.10).
const BLEED_SHARE_NEAR: f32 = 1.0 / 64.0;
const BLEED_SHARE_LADDER: f32 = 7.0 / 64.0;
// Rungs of the ladder. Eight: the ripple a bleed-only pass leaves across a hard edge
// falls 2.4 → 1.9 → 1.3 → 1.1 → 1.0 levels rms as this goes 3 → 4 → 6 → 8 → 12, so
// this is where it flattens. Their distances are `j·reach/TAPS` under *integer*
// division, floored and held at ≥ 1 — the host's `stencil_moment` builds the identical
// set, including the rungs that collapse onto each other on a tip whose reach is only
// a few texels.
const BLEED_LADDER_TAPS: i32 = 8;
const BLEED_OFFS = array<vec2<i32>, 4>(
    vec2<i32>(1, 0),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 1),
    vec2<i32>(0, -1),
);

// One thread per snapshot texel: this texel's share of a bleed firing, for the ladder
// to read back.
//
// **Dispatched over the whole snapshot square, not the firing's rect.** The ladder
// clamps a tap into `[0, udim)` — the scratch's own extent — so a texel outside the
// dispatched rect can still be read, and reading one this pass had not written would be
// reading whatever the *previous* firing left there. Covering the square is what makes
// that impossible rather than unlikely; the texels beyond the rect cost one rim test
// each and store the exact zero they would have computed.
//
// The gate is deliberately not here. `w` is the *un-toothed* weight — a firing carries
// `tooth_give = 1` by construction (`bleed_fires`), because bleed is wet paint spreading
// sideways rather than a tip dragged over grain, and gating it would break the
// antisymmetry outright: the two threads of a pair sit over different substrate. Computing
// it once, here, is what enforces that, where the deposit's own `w_t` used to be free to
// pick up a tooth its neighbour's `w_n` never had.
@compute @workgroup_size(8, 8)
fn bleed_weight(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(bleed_w_w);
    if gid.x >= dim.x || gid.y >= dim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let sw = segment_frame(st.rect_origin + t);
    let lk = segment_look(sw);
    var w = 0.0;
    if !outside_look(lk) {
        w = 1.0 - exp(-bleed_rate() * swept_pre(lk) * sw.warp * stretch_gain(st.stretch));
    }
    textureStore(bleed_w_w, t, vec4<f32>(w, 0.0, 0.0, 0.0));
}

// ---- What the two deposit kernels share ------------------------------------------
//
// `deposit` and `deposit_coarse` are **one model at two resolutions** (§6.2): the exact
// kernel differences the bake at every texel, the coarse one blends four cell centres
// the hoist evaluated the same expressions at, and from the exchange solve onward there
// is nothing left to differ about. That was a hundred lines written out twice with a
// comment on the second saying it "must stay so, line for line" — an invariant no
// reader could check and no compiler could hold, on the one kernel where a divergence
// stays invisible until a brush radius flips the host's choice of path.
//
// So it is said once, below, and the two kernels are what they claim to be: the same
// tail reached by two front halves. The three bundles are what a tail needs and a front
// half produces.

// The three snapshot texels a paint-laying kernel reads, and writes back through.
//
// Read as a bundle because the store `mix`es against all three: a texel outside the
// selection keeps exactly what it had, which is these.
struct Under {
    c: vec4<f32>,
    a: vec4<f32>,
    @if(resid) r: vec4<f32>,
}

fn under_at(t: vec2<i32>) -> Under {
    var u: Under;
    u.c = textureLoad(under_color, t, 0);
    u.a = textureLoad(under_aux, t, 0);
    @if(resid) u.r = textureLoad(under_resid, t, 0);
    return u;
}

// The tool's mean load over the pass it made across this texel — the front half's whole
// output, recovered from the bake by `deposit` and from the cell scratch by
// `deposit_coarse`, by expressions that are the same ratio taken in the same measure.
struct ToolMean {
    h: f32,
    m: f32,
    col: Latent,
}

// The canvas as this slot found it, in the form the transfer acts on: height, per-unit
// opacity, and the latent (and residual) un-premultiplied out of it.
//
// `bled` is part of the *verdict*, not of the state: it says a lateral flux moved
// something, which is a change to store even when the vertical transfer was the
// identity. The coarse kernel never sets it, and that is structural rather than an
// omission — the host keeps every bleed slot on the exact path.
struct Canvas {
    h: f32,
    op: f32,
    col: Latent,
    bled: bool,
}

fn canvas_at(u: Under) -> Canvas {
    var c: Canvas;
    c.h = u.a.x;
    c.op = u.c.a;
    // Un-premultiplied by the **color's** alpha, both halves, in one call — the
    // residual target's own is a duplicate, and dividing by it is the mistake this
    // signature cannot express (`lib::latent`).
    c.col = latent_unpremult(under_color_of(u), u.c.a, PAINT_EPS);
    c.bled = false;
    return c;
}

// The snapshot's premultiplied color, as one value. The `@if` is on the *field*, which
// exists only in a residual build, and this is the one place it is read.
fn under_color_of(u: Under) -> Latent {
    var col = latent_of(u.c.rgb, vec3<f32>(0.0));
    @if(resid) col.res = u.r.rgb;
    return col;
}

// The shared tail: the lift, the parcel this slot lays, and the store.
//
// Everything here is per texel *by model* and stays so at either resolution — the
// selection coverage, the snapshot, the arc the drain fades along, the color-dynamics
// jitter, and the re-store guard. What the two kernels hand in is the exposure's
// solved shares (`ex`) and the tool's mean (`tool`); everything else they compute
// identically, which is why it is computed here.
fn lay_parcel(
    rt: vec2<i32>,
    sw: Sweep,
    ex: Exchange,
    sel: f32,
    tool: ToolMean,
    canvas: Canvas,
    u: Under,
) {
    // The lift's removal: height leaves; the remaining paint's latent and per-unit
    // opacity are unchanged — the source fades because its *thickness* drops,
    // never because its alpha is scaled (§6.1).
    var h = canvas.h * ex.keep;
    var m = canvas.op * h;

    // How far along the stroke this texel sits, and what the brush has left to lay
    // there. The same value whichever segment is drawing it — segment `i` starts at
    // `dᵢ` and sees this texel at `(arc − dᵢ)/radius` — so both rates below come out
    // independent of how the path was cut (`stroke_drain`). It is also the domain the
    // color-dynamics lookup runs on, so there is one arc, not two.
    //
    // The travel frame's `y` is in units of the tip **in force here** and scales by
    // `sw.radius`; its `x` is in the segment's reference units and scales by
    // `st.frame_radius`. The two axes are denominated differently on a ramping tip
    // (`stamp_common::Sweep`).
    let arc = st.arc_at_start + sw.local.x * st.frame_radius;
    let drain = stroke_drain(arc);
    let own_op = st.brush_op * drain;

    // The brush's own color at this texel: the stroke color offset by the
    // color-dynamics field, sampled at the same stroke-local (lateral, arc) domain as
    // the fast path — both in canvas px.
    var own_lat = st.brush_lat;
    if any(st.noise_amp != vec3<f32>(0.0, 0.0, 0.0)) {
        let lateral = sw.local.y * sw.radius;
        own_lat += color_jitter(
            dyn_noise_tex, dyn_noise_samp, vec2<f32>(lateral, arc),
            st.noise_freq, st.noise_amp, st.noise_off,
        );
    }

    // The parcel this slot lays: the tool's deposit + the brush's own `add` paint —
    // the latter linear in exposure, i.e. exactly the plain deposit's path-integral
    // height.
    // The tool's per-unit opacity, from the two means it carries. Recovered here
    // rather than carried, so the one expression answers for both front halves.
    let tool_op = tool.m / max(tool.h, PAINT_EPS);
    let dep_h = tool.h * ex.dep;
    // No tooth factor here, and none on the deposit above: both already carry it,
    // because `ex` was solved at the exposure the substrate let through. Multiplying
    // again would charge the substrate twice — and would take the deposit out of the
    // complement the tool's own dispatch computed, which is what would destroy paint.
    let add_h = st.add * drain * ex.add_w;
    let par_h = dep_h + add_h;
    let par_m = tool_op * dep_h + own_op * add_h;

    // A texel this dispatch did not change must not be **re-stored** (§6.2). The
    // write itself is not neutral: a storage write re-encodes f32 → f16, and a
    // backend may truncate that conversion toward zero (D3D12's f32→f16 does), so
    // re-storing an algebraically identical (premult, op, height) walks a channel
    // down one ULP per rewrite wherever the recompute rounds a hair low — and
    // `premult = (premult/op)·op` rounds low about half the time for any op ≠ 1.
    // Iterated over the sub-pixel segments a slow hand is fitted at, that is a
    // *directional* drift: measured at +28 8-bit levels of ghost over a 177-knot
    // stroke whose every rate was zero. The guard is semantic, not a value
    // comparison: `keep = 1` says the lift took nothing (exact — `exp(±0)` is
    // exactly 1, and any nonzero rate gives `keep < 1`), a zero parcel says
    // nothing landed, no flux says nothing moved sideways. What remains is the
    // identity, whose only observable effect is the rounding above.
    if ex.keep == 1.0 && par_h == 0.0 && par_m == 0.0 && !canvas.bled {
        return;
    }
    // The brush's own color. Its residual is **not** jittered, where the latent above
    // is (see `exchange`), so the pair is assembled here rather than by `own_lat`.
    let own = latent_of(own_lat, st.brush_res);
    var parcel = own;
    if par_m > PAINT_EPS {
        // The same two masses weight both halves, so the parcel is one color rather
        // than two independently averaged ones — which one call states and two lines
        // could only promise.
        parcel = latent_div(
            latent_wsum(tool.col, tool_op * dep_h, own, own_op * add_h),
            par_m,
        );
    }

    let col = latent_blend(canvas.col, m, parcel, par_m);
    h += par_h;
    m += par_m;
    let op = m / max(h, PAINT_EPS);

    // Lerp the whole exchange back towards the snapshot by the selection coverage: at
    // the mask's feathered edge the texel receives that fraction of everything this
    // slot did — the lift as well as the deposit — which is exactly the fraction
    // `exchange` scaled its own side of the transfer by.
    // On the f16 lattice before it goes down (`lib::store`). The guard above skips the
    // stores that change nothing at all; this is what keeps the ones that *do* land
    // from each shaving a ULP off the region as a segment sweep passes over it.
    textureStore(region_color_w, rt, f16_nearest4(mix(u.c, vec4<f32>(col.lat * op, op), sel)));
    textureStore(region_aux_w, rt, f16_nearest4(mix(u.a, vec4<f32>(h, 0.0, 0.0, 0.0), sel)));
    // Premultiplied by the same `op` and lerped by the same `sel` — the residual is
    // stored exactly as the color beside it is.
    @if(resid) textureStore(
        region_resid_w, rt, f16_nearest4(mix(u.r, vec4<f32>(col.res * op, 0.0), sel)),
    );
}

@compute @workgroup_size(8, 8)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_color);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let rt = st.rect_origin + t;
    let rdim = vec2<i32>(textureDimensions(region_color_w));
    if rt.x < 0 || rt.y < 0 || rt.x >= rdim.x || rt.y >= rdim.y {
        return;
    }
    // Travel-frame brush-local coords (x in radius units from the segment start).
    // The AABB's corners are outside the sweep and provably at zero exposure
    // (`outside_sweep`): reject them before the prefix-τ taps below spend ~10 loads
    // apiece computing that zero.
    let sw = segment_frame(rt);
    // Where that frame lands in the one the prefix and the bake rows are indexed in,
    // once the tip's stretch is taken into account (§6.6). Built once and used three
    // times over — the rim test, the exposure and the bake rows all read it — where
    // each used to build its own. The frame itself stays the tip's own: the arc and
    // the noise domain are measured off that.
    let lk = segment_look(sw);
    if outside_look(lk) {
        return;
    }

    // The swept exposure — the brush's coverage integrated as it travels this
    // segment, as a difference of prefix sums (identical to the plain deposit's
    // `swept_depth`). Overlapping segment sweeps add their exposures, so the
    // exponential rates below compose exactly across the whole stroke.
    //
    // `dpre` is that difference in *this texel's own* measure of travel and `e` is it
    // in the tip's, which differ by `sw.warp` on a curved segment (see `sweep_at`) and
    // by `stretch_gain` on a stretched tip. Both are needed: every exchange rate below
    // is per unit tip travel, but the bake was integrated against the same prefix and
    // its ratios must be taken in the measure it was integrated in — where the factors
    // cancel rather than squaring.
    let dpre = swept_pre(lk);
    if dpre <= 0.0 {
        return; // outside the swept extent: the region keeps its value
    }
    // The substrate's share of this texel (§6.4), and the deposit jitter beside it
    // (§6.2). Both scale the **exposure**, so the whole exchange below runs on the
    // contact this texel actually had: the four shares stay exact complements of one
    // exposure, so the canvas takes precisely what the tool gives up, and each stays
    // in the form that composes across segment cuts. Scaling the finished shares
    // instead would do neither.
    let e = dpre * sw.warp * stretch_gain(st.stretch)
        * substrate_tooth(rt, sw.turn) * jitter_at(rt);
    // The selection's coverage here (§6.8). Fully deselected texels are exactly the
    // "keeps its value" case, so they cost nothing.
    let sel = load1(sel_mask, rt).r;
    if sel <= 0.0 {
        return;
    }
    // The canvas's share of the segment's transfer (`exchange_at`), of which the tool
    // took the complement in the dispatch just before this one — off the same region
    // and the same reservoir, both as the segment found them.
    let ex = exchange_at(e);

    let u = under_at(t);
    // The tool's carried paint over this spot, as the tip actually presented it:
    // the swept prefix differenced across the segment (see `bake`) gives the
    // exposure-weighted integral of the reservoir over the whole pass, so dividing
    // by the exposure recovers the mean load that pass delivered. Exact for any
    // segment length — no mid-pass sample, no length-dependent error.
    let d_load = bake_at(bake_load, lk.x, lk.z) - bake_at(bake_load, lk.y, lk.z);
    let d_latm = bake_at(bake_latm, lk.x, lk.z) - bake_at(bake_latm, lk.y, lk.z);
    // Divided by the prefix difference in the *same* measure the bake integrated
    // against, not by the tip-travel exposure: these three are means of the load the
    // pass delivered, so the curved sweep's warp cancels between numerator and
    // denominator instead of appearing twice.
    //
    // Height and mass are non-negative, so a total below zero can only be rounding.
    // The *latent* is not: these are color-space channels (Oklab a/b are signed —
    // green carries a negative a), so its prefix is not monotonic and clamping the
    // difference would shear the chroma off whatever the tool is carrying.
    let inv_pre = 1.0 / dpre;
    // The residual's mean rides the same mass and the same measure — unclamped like the
    // latent, and for the same reason: a residual is signed (`rgb − poly(c)` is negative
    // on all three channels at black, §6.7), so clamping its prefix difference would
    // shear the color it exists to carry.
    var d_col = latent_of(d_latm.xyz, vec3<f32>(0.0));
    @if(resid) d_col.res = (bake_at(bake_rlm, lk.x, lk.z) - bake_at(bake_rlm, lk.y, lk.z)).xyz;
    var tool: ToolMean;
    tool.h = max(d_load.x, 0.0) * inv_pre;
    tool.m = max(d_load.y, 0.0) * inv_pre;
    tool.col = latent_div(latent_scale(d_col, inv_pre), max(tool.m, PAINT_EPS));

    // The canvas as this segment found it — or, when the brush bleeds, that state
    // relaxed towards its neighbours first (see `BLEED_REACH` above). Ahead of the
    // lift rather than behind it: the flux's stability bound is a fraction of the
    // *entering* height, and paint the lift has already taken must not also be shed
    // sideways. The tool's own half sampled the un-bled region, so the two sides of
    // the vertical transfer disagree by O(k_bleed·k_lift·e²) — second order, the
    // same class as the mean-field freeze the loop already carries (`exchange_at`).
    var canvas = canvas_at(u);
    let kb = bleed_rate(); // the bleed rate; nonzero only on a dedicated bleed slot
    if kb > 0.0 {
        // This texel's own mobility, read from the same scratch its neighbours read it
        // from (`bleed_weight`) rather than recomputed from `e`. Two threads of a pair
        // now agree because they load one texel, not because two derivations of one
        // formula are argued to land on the same float.
        let w_t = textureLoad(bleed_w, t, 0).r;
        // Entering quantities in the mass-weighted form every flux moves (§6.1):
        // height, optical mass, latent·mass — the same formula both threads of a
        // pair evaluate, which is what makes the flux antisymmetric.
        let m0 = u.c.a * u.a.x;
        let lm0 = latent_scale(under_color_of(u), u.a.x);
        var dh = 0.0;
        var dm = 0.0;
        var dlm = latent_zero();
        // Rung 0 is the 1 px near tap; rungs 1..=TAPS climb the reach the host solved
        // for this firing (see `BLEED_SHARE_NEAR` above). Every thread derives the same
        // integers from the same uniform, so a pair at any rung still computes one
        // number — the antisymmetry the conservation rests on.
        let reach = max(st.bleed_reach, 1);
        let rungs = BLEED_LADDER_TAPS + 1;
        let per_rung = BLEED_SHARE_LADDER / f32(BLEED_LADDER_TAPS);
        for (var k = 0; k < rungs; k = k + 1) {
            var dist = 1;
            var share = BLEED_SHARE_NEAR;
            if k > 0 {
                dist = max((reach * k + BLEED_LADDER_TAPS / 2) / BLEED_LADDER_TAPS, 1);
                share = per_rung;
            }
            for (var i = 0; i < 4; i = i + 1) {
                // Clamped into the snapshot rect. A clamped tap lands on the rect's
                // border, which the margins put provably outside the sweep, so its
                // exposure — and with it the flux — is exactly zero: a no-flux
                // wall, at every scale.
                let tn = clamp(
                    t + BLEED_OFFS[i] * dist,
                    vec2<i32>(0, 0),
                    vec2<i32>(udim) - vec2<i32>(1, 1),
                );
                if all(tn == t) {
                    continue; // a tiny tip's collapsed scale, or the wall itself
                }
                let rtn = st.rect_origin + tn;
                // The neighbour's mobility — **the very texel its own thread reads for
                // itself**, so the pair computes one number by construction. This is
                // where the ladder used to spend a sweep frame, a rim test and a
                // prefix-τ difference, thirty-six times over, on an answer the thread
                // one tap away was deriving at the same moment (`bleed_weight`).
                let w_n = textureLoad(bleed_w, tn, 0).r;
                let seln = load1(sel_mask, rtn).r;
                let c = share * min(w_t, w_n) * seln;
                if c > 0.0 {
                    let ucn = textureLoad(under_color, tn, 0);
                    let uan = textureLoad(under_aux, tn, 0);
                    // The neighbour's color, both halves. The same share of the same
                    // pair's difference carries them: the residual diffuses with the
                    // paint that carries it, not on a rate of its own (§6.7).
                    var ncol = latent_of(ucn.rgb, vec3<f32>(0.0));
                    @if(resid) ncol.res = textureLoad(under_resid, tn, 0).rgb;
                    dh += c * (uan.x - u.a.x);
                    dm += c * (ucn.a * uan.x - m0);
                    dlm = latent_add(
                        dlm,
                        latent_scale(latent_sub(latent_scale(ncol, uan.x), lm0), c),
                    );
                }
            }
        }
        // The residual joins the verdict rather than riding on the latent's — a flux
        // that moved residual and no latent is possible, and calling that texel unbled
        // would drop the transfer (`latent_any`).
        canvas.bled = dh != 0.0 || dm != 0.0 || latent_any(dlm);
        canvas.h = u.a.x + dh;
        let m1 = m0 + dm;
        canvas.op = m1 / max(canvas.h, PAINT_EPS);
        canvas.col = latent_div(latent_add(lm0, dlm), max(m1, PAINT_EPS));
    }

    // The lift, the parcel and the store — the half this kernel shares with its coarse
    // twin, said once (`lay_parcel`).
    lay_parcel(rt, sw, ex, sel, tool, canvas, u);
}

// ---- 4b. The coarse deposit: hoisting the exchange to the tip's shoulder. --------
//
// `deposit` above evaluates its expensive per-texel quantities — two prefix-τ
// differences, six `bake_at` taps, the divides that turn them into reservoir means —
// on a field that cannot vary faster than the tip itself resolves. A round tip's
// finest feature is its **shoulder**, the coverage falloff `3·(1−hardness)·radius`
// wide; at a 500 px radius and mid hardness that is hundreds of texels, and the
// per-texel taps are re-deriving a locally constant answer at every one of them.
// Measured, the deposit is ALU/latency-bound on exactly those taps (~1 ns/texel, an
// order under streaming), so hoisting them is the win — not resampling the result.
//
// So the host may split the dispatch in two wherever the shoulder allows
// (`budget::extent_cell`): `cell_hoist` runs one thread per **cell** — a c×c
// square of canvas texels — and evaluates the taps once at the cell's centre;
// `deposit_coarse` then runs the same per-texel grid the exact kernel would,
// **interpolating** its neighbouring cell centres back into a per-texel value. What
// stays per-texel is what is per-texel *by model*: the substrate's tooth (the canvas's
// resolution, never the cell's), the selection coverage, the snapshot reads, the
// color-dynamics jitter, the arc the drain fades along, and the store with its
// re-store guard.
//
// **The read is bilinear, and that is not a polish pass.** A nearest read makes every
// hoisted quantity piecewise constant over the cell, so the coverage falloff prints
// as a staircase of c-wide treads on a canvas-anchored grid — a grid of fat pixels,
// plainly visible at c = 5 even though the levels barely move, because the eye reads
// the *edges* an axis-aligned step introduces and not the amplitude. Interpolating
// costs three more loads per scratch texture and drops the error from O(c·|∇|) to
// O(c²·|∇²|), which is what makes the shoulder bound in `extent_cell` (four cells
// across the falloff) mean what it claims.
//
// What is interpolated is **premultiplied by the exposure that weights it**, for the
// same reason compositing interpolates premultiplied alpha: a cell whose centre the
// sweep never covers carries no mean at all, not a mean of zero, and lerping toward
// its zero would drag the tool's height and color down at every rim. So the hoist
// stores `(e, e·h, e·m)` and the latent as `e·m·lat`, and the deposit recovers the
// means by the same divides the exact kernel does — an unexposed neighbour then
// contributes zero *weight* rather than a zero *value*, and a texel whose four
// neighbours are all unexposed still reads exactly 0 and takes the early exit.
//
// **The cell grid is anchored to the canvas, not to the region** (§6.4). A cell is
// `floor(canvas / c)`, reached here as `floor((rt + anchor) / c)` with
// `anchor = region origin mod c` (`Stamp::k`) — congruent arithmetic on small
// integers, so two pieces with different region origins, or a live fold and the
// commit that replaces it, agree on every cell boundary without the shader holding a
// canvas coordinate an f32 might round. The cell *centre* is likewise a pure function
// of the cell and the segment, so the hoisted means are as replay-deterministic as
// the per-texel values they stand in for.
//
// What the approximation costs, and where it is bounded: the exposure a texel sees is
// its cell centres' bilinear blend, so the coverage falloff is rendered as a piecewise
// bilinear chord of the true one — no step anywhere, and an error second order in the
// cell, bounded because the falloff is at least 4c wide wherever the host engages this
// path at all (`extent_cell`) and a tip with no shoulder never leaves the exact
// kernel. At the extent's rim the blend reaches up to a cell past where the
// per-texel integral would stop, fading linearly to nothing over that last cell rather
// than ending on a tread; the two sides of the transfer then disagree by the rim
// ring's worth of exposure — the same order as the mean-field freeze the loop already
// carries, and it fades with the coverage itself. The bleed ladder is absent by
// construction rather than omitted: bleed runs on dedicated slots, and the host keeps
// every bleed and settle slot on the exact path.

// Floor division, componentwise. WGSL's `/` on i32 truncates toward zero, and the
// cell grid must round the same way either side of zero — a dispatch rect's origin
// can sit a margin below the region's edge — or the anchor stops meaning anything
// exactly where two rects meet.
fn div_floor2(n: vec2<i32>, c: i32) -> vec2<i32> {
    let q = n / c;
    return q - vec2<i32>(
        select(0, 1, n.x - q.x * c < 0),
        select(0, 1, n.y - q.y * c < 0),
    );
}

// The scratch's origin: one cell *below* the first the dispatch rect touches, because
// a texel in that first cell's leading half interpolates against the cell before it.
// Both kernels derive it from the same uniform, so the two agree on which texel of the
// scratch is which cell, and the host counts the same two-cell surplus into the grid
// it dispatches (`cell_geometry`).
fn cell_base() -> vec2<i32> {
    return div_floor2(st.rect_origin + st.cell_anchor, st.cell_px) - vec2<i32>(1, 1);
}

// One thread per cell: the exact kernel's front half, evaluated at the cell's centre
// and stored for `deposit_coarse` to read back. Threads past the cells the host
// counted for this rect evaluate a real (unused) cell or store out of bounds, which
// WGSL defines as a no-op — the dispatch rounding needs no guard of its own.
@compute @workgroup_size(8, 8)
fn cell_hoist(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = vec2<i32>(gid.xy);
    let c = st.cell_px;
    let a = st.cell_anchor;
    // The cell's centre, in region px: the cell spans canvas `[k·c, (k+1)·c)`, i.e.
    // region `[k·c − a, (k+1)·c − a)`, and this is that span's midpoint — a position,
    // not a texel index, so no half-texel offset applies on top.
    let p = (vec2<f32>(cell_base() + ci) + vec2<f32>(0.5, 0.5)) * f32(c) - vec2<f32>(a);
    let sw = sweep_at(p, st.start, st.dir, st.frame_radius, travel_px(), st.curvature);
    var tool = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var lat = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    @if(resid) var res = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let lk = segment_look(sw);
    if !outside_look(lk) {
        // The swept exposure and the baked reservoir means, exactly as `deposit`
        // takes them — same taps, same measure, same clamps — at one position.
        let dpre = swept_pre(lk);
        if dpre > 0.0 {
            let d_load = bake_at(bake_load, lk.x, lk.z) - bake_at(bake_load, lk.y, lk.z);
            let d_latm = bake_at(bake_latm, lk.x, lk.z) - bake_at(bake_latm, lk.y, lk.z);
            // The two measure conversions `deposit` applies to the exposure and to
            // nothing else, taken together — the warp of a curved sweep and the gain of
            // a stretched tip. **Applied to the loads as well as to `e`**, which is
            // what keeps the ratio `deposit_coarse` takes the same ratio `deposit`
            // takes: there the loads are divided by a bare `dpre` that carries
            // neither factor, so here they must carry both and cancel.
            let wg = sw.warp * stretch_gain(st.stretch);
            // Everything below the exposure is stored **premultiplied by it** — the
            // weight the interpolation has to carry, so an unexposed neighbour costs
            // a lerp weight and not a value. `deposit_coarse` divides them back out
            // with the exact kernel's own expressions, so a texel at a cell centre
            // recovers precisely what `deposit` would have computed there.
            let e = dpre * wg;
            let load = wg * max(d_load.xy, vec2<f32>(0.0, 0.0));
            tool = vec4<f32>(e, load.x, load.y, 0.0);
            lat = vec4<f32>(d_latm.xyz * wg, 0.0);
            @if(resid) {
                let d_rlm = bake_at(bake_rlm, lk.x, lk.z) - bake_at(bake_rlm, lk.y, lk.z);
                res = vec4<f32>(d_rlm.xyz * wg, 0.0);
            }
        }
    }
    textureStore(cell_tool_w, ci, tool);
    textureStore(cell_lat_w, ci, lat);
    @if(resid) textureStore(cell_res_w, ci, res);
}

// A texel's read of the cell scratch: the four cell centres around it and their
// bilinear weights, in scratch coordinates. The centre of cell `k` sits at region
// position `(k + ½)·c − a`, so the continuous cell coordinate of the texel whose
// centre is `rt + ½` is `(rt + a + ½)/c − ½`, and the four cells are its floor and
// that cell's upper neighbours. `cell_base`'s border ring is what keeps every one of
// them a cell the hoist actually wrote.
struct CellTap {
    // The lower-left cell, in the scratch's own coordinates.
    at: vec2<i32>,
    // Weights for (0,0), (1,0), (0,1), (1,1) — summing to 1.
    w: vec4<f32>,
}

fn cell_tap(rt: vec2<i32>) -> CellTap {
    let c = st.cell_px;
    let a = st.cell_anchor;
    let u = (vec2<f32>(rt + a) + vec2<f32>(0.5, 0.5)) / f32(c) - vec2<f32>(0.5, 0.5);
    let lo = floor(u);
    let f = u - lo;
    let g = vec2<f32>(1.0, 1.0) - f;
    return CellTap(
        vec2<i32>(lo) - cell_base(),
        vec4<f32>(g.x * g.y, f.x * g.y, g.x * f.y, f.x * f.y),
    );
}

fn cell_lerp(tex: texture_2d<f32>, tap: CellTap) -> vec4<f32> {
    return textureLoad(tex, tap.at, 0) * tap.w.x
        + textureLoad(tex, tap.at + vec2<i32>(1, 0), 0) * tap.w.y
        + textureLoad(tex, tap.at + vec2<i32>(0, 1), 0) * tap.w.z
        + textureLoad(tex, tap.at + vec2<i32>(1, 1), 0) * tap.w.w;
}

// The exact `deposit` with its front half read from the cell scratch. Every
// departure from that kernel is one of two things: a per-texel quantity replaced by
// the blend of its cells' (the exposure, the tool means), or the bleed ladder gone —
// the host keeps bleed slots on the exact path, so `bled` here is structurally false
// and the re-store guard loses nothing. Everything else — the frame, the tooth, the
// selection, the arc, the jitter, the parcel algebra, the guard, the stores — is
// line for line the same, and must stay so: the two kernels are one model at two
// resolutions, and a divergence between them is a `preview == committed` bug waiting
// for the radius that flips the host's choice.
@compute @workgroup_size(8, 8)
fn deposit_coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_color);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let rt = st.rect_origin + t;
    let rdim = vec2<i32>(textureDimensions(region_color_w));
    if rt.x < 0 || rt.y < 0 || rt.x >= rdim.x || rt.y >= rdim.y {
        return;
    }
    let sw = segment_frame(rt);
    if outside_sweep(sw) {
        return;
    }

    // This texel's four cells and their weights.
    let tap = cell_tap(rt);
    let cell = cell_lerp(cell_tool, tap);
    // The exposure in the tip's measure (`dpre · warp`), blended from the centres.
    // Zero is the hoist's "outside the swept extent" verdict at all four of them,
    // and takes the same exit.
    if cell.x <= 0.0 {
        return; // no cell around this texel saw exposure: the region keeps its value
    }
    // The substrate's share of this texel (§6.4) and the deposit jitter (§6.2) — per
    // texel, never per cell: both gates are the canvas's resolution, and averaging
    // either over a cell would smear exactly the per-texel structure it exists for.
    let e = cell.x * substrate_tooth(rt, sw.turn) * jitter_at(rt);
    let sel = load1(sel_mask, rt).r;
    if sel <= 0.0 {
        return;
    }
    let ex = exchange_at(e);

    // The means, recovered from the blended masses by the exact kernel's own
    // expressions — `inv_e` standing where `deposit` writes `inv_pre`, the two
    // differing by the warp both sides of the ratio carry.
    let inv_e = 1.0 / cell.x;
    var cell_col = latent_of(cell_lerp(cell_lat, tap).xyz, vec3<f32>(0.0));
    @if(resid) cell_col.res = cell_lerp(cell_res, tap).xyz;
    var tool: ToolMean;
    tool.h = cell.y * inv_e;
    tool.m = cell.z * inv_e;
    // The exact kernel's own expression, on the blended cell means (`deposit`).
    tool.col = latent_div(latent_scale(cell_col, inv_e), max(tool.m, PAINT_EPS));

    // The lift, the parcel and the store — the exact kernel's own tail, called rather
    // than restated (`lay_parcel`). There is no bleed to fold in first: the host keeps
    // every bleed slot on the exact path, so `Canvas::bled` is structurally false here
    // and the re-store guard loses nothing by it.
    let u = under_at(t);
    lay_parcel(rt, sw, ex, sel, tool, canvas_at(u), u);
}

// ---- 5. Settle: the transfer the pen-up breaks. ----------------------------------
//
// A stroke stops with the tip still **in contact and out of equilibrium**: the tool
// is holding more than the canvas under it (or less, for an eraser), and that excess
// is exactly the transfer the tip would have gone on making. Everywhere else on the
// trail the tip left smoothly — a point sees the whole extent pass over it, ending
// with the trailing rim where `τ` has fallen back to 0 — but the last extent never
// gets that, so what was in flight is stranded. The reservoir is per-stroke, so
// stranded means gone: a tip-shaped disc short of the trail behind it, and (for an
// eraser) a tip-shaped smudge of paint that should have been taken.
//
// So the pen-up settles the pair — one dispatch, one thread per canvas texel of the
// final extent, the same `exchange_at` kernel as every segment. What it needs is an
// exposure, and it is bounded by the pass on **both** sides:
//
//     owed = prefix(l),   received = rowtotal − prefix(l)
//
// * `received` is what the tip has already given this texel. A point the tip has
//   barely reached has no film to break — and without this bound the settle steps
//   against untouched canvas a radius *ahead* of where the pen came up. Measured on a
//   smear over a flat field that step is 40% of the paint's own range; on an eraser it
//   is a fully-scraped disc with a hard rim.
// * `owed` is what it still had to give. A point the tip had all but finished with has
//   nothing left to hand over — and without this bound the settle steps against the
//   *trail*, which got no settle at all, right where the two meet.
//
// Both are readings of the **same prefix-τ volume** the swept deposit integrates
// against, and that is the whole reason this comes out smooth. The instantaneous
// depth `τ(l)` would do neither: `κ = −ln(1−coverage)` has a knee wherever coverage
// approaches 1, so a settle keyed on it turns on across the few pixels of the tip's
// shoulder and prints the tip's own edge. `prefix` is that same `τ` *integrated along
// the travel*, so it ramps across the whole radius — which is why the brush's own
// `add`, which is linear in it, caps smoothly, and the exchange, which saturates
// exponentially in it, does not.
//
// **The two bounds are combined with a soft `min`, not a `min`** (§6.2), because a
// `min` of two ramps is continuous and its *slope* is not. The two cross at the tip
// centre, where `owed = received = rowtotal/2`, and the exposure turns a corner there.
// Everything below is exponential in that exposure, so a corner in it is a corner in
// the height the settle leaves — and a corner in height is a **step in the substrate
// normal**, which the media pass (§6.3) prints as a hard line straight across the
// middle of the last extent, perpendicular to the travel. It is not subtle at any
// impasto worth the name: on a 64 px smear into thick paint the specular jumped 107
// levels across one texel at exactly `xl = 0`, against 2 anywhere along the trail. That
// the kernel is deep in saturation there is no defence — it is true of the *value* and
// says nothing about the derivative, which is the thing the light actually reads.
//
// So the handover is the **p-norm soft minimum** of the two, `(owed⁻ᵖ + received⁻ᵖ)⁻¹ᐟᵖ`
// — written below as `min · (1 + (min/max)ᵖ)⁻¹ᐟᵖ`, which is the same number and never
// divides by the small one. It is smooth in every derivative, it is `≤ min` everywhere
// (so both bullets above keep their ceiling exactly), and it agrees with `min` to
// within `(min/max)ᵖ` — so it is indistinguishable from it wherever one bound is
// clearly the binding one, and only softens where they are comparable.
//
// Both bounds vanish on the extent's rim (`owed` at the trailing edge, `received` at
// the leading one, and the row total itself laterally), so the settle fades to nothing
// all the way round and adds no edge of its own.
//
// The tool is not written back: nothing reads it after this. A range that does *not*
// reach the end of the stroke never runs this dispatch — it hands its reservoir on to
// the range that resumes instead — so the settle happens once, at the pen-up, and a
// live tail computes the same one its commit will.
// How sharply the settle's two bounds hand over to each other (see `settle`) — the
// exponent of the p-norm soft minimum. `∞` is a plain `min`; `1` is their harmonic mean.
//
// **Both ends of this are a defect, and they are different defects**, which is what
// makes 4 an optimum rather than a floor. Push it up and the handover, still smooth in
// the exact arithmetic, concentrates all of its curvature into a fraction of a texel —
// and a render that samples a normal at texel centres cannot tell that from the corner
// it was meant to replace. Push it down and the settle stops settling: a texel just
// behind the tip centre has all but finished its pass, so what is left in flight there
// is nearly the whole of what is still owed, and a soft minimum that discounts it too
// far strands exactly the load this dispatch exists to deliver.
//
// Measured on the three tests that read those two failures — the crease under a sky
// with structure in it (`the_settle_leaves_no_crease_across_the_last_stamp`), and the
// two that read the pen-up point itself (`an_eraser_takes_the_paint_it_stopped_on`,
// `a_carrying_stroke_ends_without_breaking_its_own_slope`), in 8-bit levels:
//
//     p          1     2     3     4     5     6     ∞ (= min)
//     crease     2     2     2     3    87   105   107
//     eraser     —     5     3     2     2     —     0
//     carry     12     9     8     7     6     6     5
//
// 4 is the last value before the crease collapses back, and it buys the two settle
// readings to within 2 levels of what the bare `min` delivered. Their bounds are 8 and
// 10 and an un-settled stroke measures 18 and 17, so there is room on that side; the
// crease has none on the other.
const SETTLE_BOUND_SHARPNESS: f32 = 4.0;

@compute @workgroup_size(8, 8)
fn settle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_color);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let rt = st.rect_origin + t;
    let rdim = vec2<i32>(textureDimensions(region_color_w));
    if rt.x < 0 || rt.y < 0 || rt.x >= rdim.x || rt.y >= rdim.y {
        return;
    }
    // The slot carries zero travel and zero curvature, so the "sweep" is the tip's own
    // extent and `outside_sweep` reduces to the brush-local unit square — the same
    // rejection `snapshot` uses to decide which texels it bothers to copy.
    let sw = segment_frame(rt);
    // The settle's whole geometry is the extent's, so unlike the deposit it works
    // in the lookup frame throughout: the row it walks is a bake row, and the point it
    // walks to is where the stretch put it (§6.6). The slot carries no travel, so the
    // range's two ends coincide and `.x` is the point — and the rim test is a reading
    // of the same frame, so it is built once.
    let lk = segment_look(sw);
    if outside_look(lk) {
        return;
    }
    let xl = lk.x;
    let yl = lk.z;

    let total = prefix_at(1.0, yl, st.orientation);
    let owed = prefix_at(xl, yl, st.orientation);
    let received = total - owed;
    let lo = min(owed, received);
    let hi = max(owed, received);
    if lo <= 0.0 {
        return; // the extent's rim: nothing in flight, nothing to settle
    }
    // The soft `min` of the two bounds (see above), taken as `lo·(1 + (lo/hi)ᵖ)^(−1/p)`
    // rather than as `(owed⁻ᵖ + received⁻ᵖ)^(−1/p)`: the same value, with the reciprocal
    // of the *small* bound — which vanishes at both rims of the extent — never formed.
    //
    // Into the tip's own measure by the same gain the deposit applies, because
    // `exchange_at` below meets it with the raw rates: `owed` and `received` are both
    // readings of a prefix in the lookup frame, and the soft `min` is homogeneous of
    // degree one in the pair, so one factor on the result is the whole conversion.
    let e = lo * pow(1.0 + pow(lo / hi, SETTLE_BOUND_SHARPNESS), -1.0 / SETTLE_BOUND_SHARPNESS)
        * stretch_gain(st.stretch);
    let sel = load1(sel_mask, rt).r;
    if sel <= 0.0 {
        return;
    }
    // Gated like every other deposit (§6.4), the jitter included (§6.2) — the settle
    // completes the same pass the segments were laying, so it must read the same
    // per-texel contact they did: a tip lifted off a rough substrate was
    // bridging the faces that fell away from its travel, so what it would have
    // delivered there stays on it. Nothing
    // is destroyed — the tool is discarded at pen-up carrying whatever it still holds,
    // which is what a knife lifted off a canvas does.
    let ex = exchange_at(e * substrate_tooth(rt, sw.turn) * jitter_at(rt));

    // The parcel is **what the rest of the pass would have delivered**, not the cell
    // overhead. At the end of a long smear the reservoir is radially skewed: the
    // interior cells trade fast, so they sit near equilibrium with the thin mid-pass
    // canvas under them — nearly empty — while the trailing shoulder, whose τ is a
    // hundredth of the interior's, still carries paint lifted hundreds of px back and
    // pays it out slowly. That slow payout is precisely what the visible trail is made
    // of, so pairing each canvas texel with the (empty) cell directly above it deposits
    // almost nothing where the continued pass would have slid the loaded trailing
    // cells over every interior point: the whole extent printed as a tip-shaped
    // disc of *missing* paint, stepped exactly where this kernel saturates.
    //
    // So the parcel is the sliding kernel's own gain integral over the remaining
    // pass, cell by cell along this row of the pen-up's `bake`:
    //
    //     delivered(xl) = k_dep · ∫₋₁^xl R·dτ · exp(−k_dep·τ(u)·(xl−u)) · exp(−k_lift·P(u))
    //
    // The first exponential is the cell **spending itself en route** — everything it
    // lays on the points between its pen-up position and this one comes out of the
    // same load — which is what confines a saturated interior cell to the few px
    // under it while letting a shoulder cell, whose τ is a hundredth of that, carry
    // its payout the whole way across the extent. The second is the parcel's
    // **survival**: cells nearer the rim serve later, so what cell `u` lays here is
    // then re-lifted by everything after it, `P` the same prefix `owed` reads.
    // Dropping the en-route factor is not an option even though the artifact's ring
    // cells barely feel it: for a brush with `deposit ≫ lift` the survival factor no
    // longer saturates the sum, and a settle without the spend-down would mint
    // several times the tool's whole load. With it, a cell's payout summed over
    // every point it serves telescopes to at most its own load — the same
    // change-of-variable bound the segment kernel rests on (`exchange_at`) — so the
    // settle cannot mint, for any pair of rates, by construction.
    //
    // `min(owed, received)` still bounds it: the delivery above assumes the pass
    // completes, and on the leading half it must not (that is the bound's whole
    // argument). The bound lands as the **ratio of the two deposit saturations**,
    // `dep(e) / dep(owed)` — 1 exactly on the trailing half, fading with `received`
    // on the leading one — scaling the whole delivery rather than cutting the pass
    // where the truncation falls. Cutting is the more literal reading and was tried
    // first: the ring serves *last*, so a truncated pass drops its entire payout the
    // moment `received` dips below `owed`, and the settle prints a fresh cliff down
    // the middle of the extent. A ratio of two smooth readings of one prefix
    // moves nothing discontinuously, and it can only shrink the parcel, so the
    // no-minting bound survives it.
    //
    // A per-texel walk of the row, not a prefix — the en-route factor couples the
    // cell to the point it serves, so no single baked integral carries it. The
    // settle runs once per stroke, which is what makes O(BAKE_RES) taps per texel
    // the right trade here where the per-segment deposit could never afford it.
    //
    // **The rates carry the stretch here, where `e` above carried it** — and it is the
    // same conversion said the other way round. Everything the walk meets them with
    // (`owed`, the column exposures `d_e`, the running prefix `pr`, the loads `d_h`)
    // is a reading of the bake in the *lookup* frame, so rather than gaining half a
    // dozen quantities one by one, gain the two rates that multiply all of them. That
    // also keeps `share` below a ratio of two shares in one measure. Exactly the
    // unstretched rates when there is no stretch.
    let g = stretch_gain(st.stretch);
    let kd = deposit_rate() * g;
    let kl = lift_rate() * g;
    let bdimf = vec2<f32>(textureDimensions(bake_load));
    // The bake row pair this texel sits between — `bake_at`'s own addressing, kept
    // linear through the differencing below (a difference of blends is a blend of
    // differences).
    let uy = clamp(yl * 0.5 + 0.5, 0.0, 1.0) * bdimf.y - 0.5;
    let fy = uy - floor(uy);
    let yi0 = clamp(i32(floor(uy)), 0, i32(bdimf.y) - 1);
    let yi1 = clamp(i32(floor(uy)) + 1, 0, i32(bdimf.y) - 1);
    let dx = 2.0 / bdimf.x; // one bake column, radius units
    var prev = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    // The walk's two color accumulators, each both halves (§6.7, `lib::latent`): the
    // previous column's mass-weighted color, and the parcel being built.
    var prev_lm = latent_zero();
    var par_h = 0.0;
    var par_m = 0.0;
    var par_lm = latent_zero();
    for (var i = 0; i < i32(BAKE_RES); i = i + 1) {
        let xr = (f32(i) + 1.0) / bdimf.x * 2.0 - 1.0;
        // The fraction of this column that lies below the point — columns ascend, so
        // the first one wholly above it ends the walk.
        let cover = clamp((xl - (xr - dx)) / dx, 0.0, 1.0);
        if cover <= 0.0 {
            break;
        }
        let cur = mix(
            textureLoad(bake_load, vec2<i32>(i, yi0), 0),
            textureLoad(bake_load, vec2<i32>(i, yi1), 0),
            fy,
        );
        var cur_lm = latent_of(
            mix(
                textureLoad(bake_latm, vec2<i32>(i, yi0), 0),
                textureLoad(bake_latm, vec2<i32>(i, yi1), 0),
                fy,
            ).xyz,
            vec3<f32>(0.0),
        );
        @if(resid) cur_lm.res = mix(
            textureLoad(bake_rlm, vec2<i32>(i, yi0), 0),
            textureLoad(bake_rlm, vec2<i32>(i, yi1), 0),
            fy,
        ).xyz;
        // Height and mass prefixes are monotone, so a negative difference is
        // rounding; the latent's is not (Oklab a/b are signed) and stays unclamped.
        let d_h = max(cur.x - prev.x, 0.0);
        let d_m = max(cur.y - prev.y, 0.0);
        let d_e = max(cur.z - prev.z, 0.0);
        let d_lm = latent_sub(cur_lm, prev_lm);
        let pr = cur.z;
        prev = cur;
        prev_lm = cur_lm;
        if d_e <= 0.0 {
            continue; // outside the extent: nothing carried, nothing served
        }
        let tau = d_e / dx;
        let xc = xr - 0.5 * dx;
        let en_route = exp(-kd * tau * max(xl - xc, 0.0));
        let survive = exp(-kl * max(pr - 0.5 * d_e, 0.0));
        let c = kd * en_route * survive * cover;
        par_h += d_h * c;
        par_m += d_m * c;
        // The same delivery coefficient on both halves — en-route spend-down and
        // survival act on the parcel, and the residual is part of the parcel, not a
        // passenger.
        par_lm = latent_add(par_lm, latent_scale(d_lm, c));
    }
    // The `min(owed, received)` bound, as the smooth share it lands as (see above).
    let share = ex.dep / max(1.0 - exp(-kd * owed), 1e-6);
    par_h *= share;
    par_m *= share;
    par_lm = latent_scale(par_lm, share);

    // The deposit's rewrite guard, for the same reason (see there): a settle whose
    // rates are all zero settles nothing, and must not re-round the texel.
    if ex.keep == 1.0 && par_h == 0.0 && par_m == 0.0 {
        return;
    }

    // The snapshot, as the deposit reads it — one bundle, one gate (`under_at`).
    let u = under_at(t);
    let uc = u.c;
    let ua = u.a;

    // The same parcel stacking as the deposit, with the delivery integral as the
    // parcel: no `add` source, because a settle is a break of contact and not a
    // stretch of travel, and the source is a rate per unit of travel. Color comes
    // back out of the optical-mass weight it was integrated under, so the parcel
    // carries the mixture of the cells that would have served this point, each in
    // proportion to what it would have delivered (§6.1).
    var h = ua.x * ex.keep;
    var m = uc.a * h;
    let canvas_col = latent_unpremult(under_color_of(u), uc.a, PAINT_EPS);
    var parcel = canvas_col;
    if par_m > PAINT_EPS {
        parcel = latent_div(par_lm, par_m);
    }
    let col = latent_blend(canvas_col, m, parcel, par_m);
    h += par_h;
    m += par_m;
    let op = m / max(h, PAINT_EPS);

    textureStore(region_color_w, rt, f16_nearest4(mix(uc, vec4<f32>(col.lat * op, op), sel)));
    textureStore(region_aux_w, rt, f16_nearest4(mix(ua, vec4<f32>(h, 0.0, 0.0, 0.0), sel)));
    @if(resid) textureStore(
        region_resid_w, rt, f16_nearest4(mix(u.r, vec4<f32>(col.res * op, 0.0), sel)),
    );
}
