// Brush dynamics: the **sequential swept-exchange loop** (§6.2).
//
// A stroke that manipulates existing paint (lift / deposit / charge) runs as an
// ordered loop, all inside one compute pass (implicit barriers between dispatches
// give the sequencing; usage scopes are per-dispatch, so the region may be
// sampled by one dispatch and storage-written by the next — no copies):
//
//   - per flattened segment, in order: `wick` (let the paint on the tool migrate
//     between neighbouring parts of the tip, before either half of the transfer reads
//     it), `bake` (integrate the reservoir along the travel axis, so the deposit can
//     read what the tip presented over a whole pass rather than at one point),
//     `exchange` — one thread per **tool reservoir** texel, which settles the tool's
//     half of the transfer, and which also carries the `snapshot` (copy the segment's
//     footprint texels to an `under` scratch, so the deposit can read-modify-write the
//     region) in the tail of its own grid — and `deposit`, one thread per footprint
//     texel. `wick` and `exchange` each ping-pong the reservoir, so a segment cycles it
//     twice.
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
// the region as the deposit had already left it — which is what the tool used to do —
// makes the tool lift from a canvas that has, from its point of view, not happened yet,
// and the halves stop adding up (see `exchange_at`).
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
// to be too, or the tool gains what the canvas gave up only some segments later. That
// lag used to be real: a separate `pickup` pass — retired, and the only mention of
// that name left in the tree — ran on its own coarse cadence, every `spacing·radius`
// of travel, and a stroke that stopped mid-interval left its last footprint stripped
// by a `deposit` whose paint the tool had not yet been credited with — a tip-shaped
// patch of missing paint, worst exactly where `lift` is high.
//
// So `exchange` does the whole of the tool's side in one dispatch, per segment,
// against the very same segment the `deposit` accounts for. Nothing is deferred, so
// nothing is left owing at the end of a stroke.
//
// This also retires a good deal of machinery. The old cadence had to hide its own
// discreteness — a whole interval's reload landing between two segments is a large
// fraction of the load at high lift × deposit, and consecutive segments served
// visibly different tools — which took a separate **gain parcel**, a linear ramp
// serving it in across the interval, and two extra baked prefixes to evaluate that
// ramp per fragment. A per-segment exchange has no step to hide: the segment length
// *is* the reload step, and `stroke::budget::flatten_tolerance` already caps it at a fraction
// of the tip. All of it is gone.
//
// Conservation (§6.1): paint moves by transferring **height** — the one
// conserved quantity. Colour and per-unit opacity ride as optical-mass
// (opacity·height) weighted blends; `add` is the only source term. Both sides of
// each transfer integrate the same rate over the same footprint, so with
// `add = 0` total height (canvas + tool) is conserved up to resampling error.
//
// Determinism: the slot sequence — segment, bleed and settle alike (`SlotKind`) —
// is fixed CPU-side per stroke; every dispatch is a pure function of what it reads.
// Replay and live == committed hold.

import package::stamp_common::{prefix_at, Sweep, sweep_at, sweep_tangent};
import package::lib::noise::{color_jitter};
import package::lib::paint_common::{blend_latent, ground_at, rise_ahead, tooth_gate, PAINT_EPS};
import package::lib::sample::{load1, bilerp_taps, bilerp_mix};

struct Stamp {
    // a: the sweep's start .xy (region px) + its travel tangent .zw (unit vector) —
    //    the prefix-τ frame, and the same two fields for all three slot kinds
    //    (`SlotKind`). A **bleed** slot's sweep is the firing's travel window rather
    //    than a stretch of painting; a **settle** slot's is the tip's own footprint,
    //    so its tangent is the direction the pen was going when it left the page.
    //    (The tool's *shape* frame — this turned by the orientation, §6.6 — is
    //    derived where it is needed, in `bake` and `exchange`, not carried here.)
    a: vec4<f32>,
    // b: radius (px), segment length / radius (0 on a settle slot, which is a break
    //    of contact rather than a stretch of travel), λ_lift, λ_deposit —
    //    λ = ln(1 − axis) ≤ 0, clamped away from −∞.
    b: vec4<f32>,
    // c: the brush's own colour channels .xyz + per-unit opacity .w. **Undrained** —
    //    the falloff is applied per fragment from its own arc length, not baked in
    //    per segment (see `stroke_drain`).
    c: vec4<f32>,
    // d: under-rect origin .xy (region px, integral); orientation (turns ∈ [0,1),
    //    picks the prefix-τ slice) .z; the `drain` falloff per canvas px .w.
    d: vec4<f32>,
    // e: the `add` rate per unit exposure, **undrained** like `c.w`: height .x; the
    //    segment's signed curvature (1/region px, 0 = a straight sweep) .y.
    //
    //    `.zw` carried the segment's midpoint, which is where `exchange` used to
    //    sample the canvas — one bilinear tap for the whole segment. It walks the
    //    texel's own track through `tip_at` now, so nothing reads the lane. It stays
    //    because the lane map is an ABI: the host's `Stamp` is generated from this
    //    declaration (§6.10), so reclaiming the lane moves both sides at once — worth
    //    doing the next time this struct is touched for a reason of its own.
    e: vec4<f32>,
    // f–h: colour dynamics (§6.2) for the brush's own `add` paint —
    // the same field/domain as the fast path's jitter, so the two paths agree.
    // f: noise frequency per lookup axis (across the stroke, along it) .xy,
    //    1/NOISE_TILE_PX in .z; .w unused. (The wick's step is not here: its kernel is
    //    a compile-time binomial and its cadence is the host's, see `wick_axis`.)
    f: vec4<f32>,
    // g: per colour-channel noise amplitude .xyz (all 0 = off); arc length at
    //    the segment start (canvas px) in .w.
    g: vec4<f32>,
    // h: per-stroke noise lookup translation .xy; the tooth's **bearing fraction** in
    //    .z — the share of the ground a tip with this `tooth` stands on, averaged over
    //    the whole ground, along **this segment's own direction of travel**
    //    (`Surface::bearing`, §6.4). It is what the *tool* books
    //    its half of the transfer against: a canvas texel scales its exposure by the
    //    ground under itself, and a reservoir cell, which has no ground of its own,
    //    scales by this. 1 where there is nothing to bite. λ_bleed — the lateral
    //    canvas diffusion rate (≤ 0, like the two λs in `b`) — in .w. Nonzero ONLY
    //    on a dedicated **bleed slot** (§6.2): a quad the host emits per crossing of
    //    the bleed cadence, whose sweep IS the firing's travel window and whose
    //    vertical rates and source are all zero. Real segments carry 0 here and take
    //    the plain-deposit path bit-for-bit.
    h: vec4<f32>,
    // i: the deposition tooth (§6.4). How little give this segment's tip has .x
    //    (0 = the ground does not gate anything), and the map from
    //    this dispatch's *region* texel to the weave: `uv = rt·.y + .zw`, with the
    //    piece's own origin already folded into the bias. `.y` is zero on a ground
    //    with no relief, which is what makes a `Flat` canvas render exactly what it
    //    always did.
    i: vec4<f32>,
}

@group(0) @binding(0) var<uniform> st: Stamp;

// ---- Reading the slot ------------------------------------------------------------
//
// The nine lanes above are packed `vec4`s because that is what a uniform wants, but
// nothing downstream should have to know which. Every field is read through one of
// these, so the loop's arithmetic says `travel_px()` and `lift_rate()` rather than
// `st.b.y * st.b.x` and `-st.b.z` — the two spellings that cost the most to decode,
// and appear five and two times respectively.
//
// This is a rename, not a repacking. The `Stamp` layout is an ABI: the host's mirror
// is generated from the declaration above (§6.10), so moving a lane moves
// `gpu::stroke::dynamics`'s `SLOT` with it. What these buy is that the *lane map*
// exists in exactly one place — here, next to the struct that documents it — instead
// of being re-read from a comment at each of the fifty-odd use sites.
//
// The two rate accessors also fold in the negation. `λ = ln(1 − axis) ≤ 0` is what
// the host packs, and every consumer wants `k = −λ ≥ 0`; spelling that at each site
// is an easy sign to lose in a kernel where the sign is the whole physics.

// The sweep's start and its travel tangent, in region px (see `Stamp::a`).
fn seg_start() -> vec2<f32> { return st.a.xy; }
fn seg_dir() -> vec2<f32> { return st.a.zw; }

// The tip's radius in region px, its travel as a multiple of that radius, and the
// travel in px — which is the product, and is what the arc arithmetic wants.
fn radius() -> f32 { return st.b.x; }
fn travel_radii() -> f32 { return st.b.y; }
fn travel_px() -> f32 { return st.b.y * st.b.x; }

// The three exchange rates, as `k = −λ ≥ 0`. `lift` and `deposit` are the vertical
// pair; `bleed` is the lateral one, and is nonzero only on a dedicated bleed slot.
fn lift_rate() -> f32 { return -st.b.z; }
fn deposit_rate() -> f32 { return -st.b.w; }
fn bleed_rate() -> f32 { return -st.h.w; }

// The brush's own paint: its latent channels and its per-unit opacity, both
// **undrained** — the falloff is applied per fragment from its own arc (`stroke_drain`).
fn brush_lat() -> vec3<f32> { return st.c.xyz; }
fn brush_op() -> f32 { return st.c.w; }

// The `add` source rate — height per unit exposure, undrained like `brush_op`.
fn add_rate() -> f32 { return st.e.x; }

// The segment's signed curvature, 1/region px; 0 is a straight sweep.
fn curvature() -> f32 { return st.e.y; }

// Where this dispatch's footprint rect sits in the region, and the shape's
// orientation in turns ∈ [0,1) — which picks the prefix-τ slice (§6.6) and is the
// rotation `bake` and `exchange` take out of the travel frame.
fn under_origin() -> vec2<i32> { return vec2<i32>(st.d.xy); }
fn orientation() -> f32 { return st.d.z; }

// The `drain` falloff per canvas px, and the arc length the segment starts at.
fn drain_rate() -> f32 { return st.d.w; }
fn arc_at_start() -> f32 { return st.g.w; }

// Colour dynamics (§6.2): the noise field's frequency, per-channel amplitude and
// per-stroke translation. Amplitude all-zero is jitter off.
fn noise_freq() -> vec4<f32> { return st.f; }
fn noise_amp() -> vec3<f32> { return st.g.xyz; }
fn noise_off() -> vec2<f32> { return st.h.xy; }

// The tooth (§6.4): how little give this tip has, the **bearing fraction** the tool
// books against (1 where there is nothing to bite), and the region-texel → weave map
// `uv = rt·scale + bias`, whose scale is zero on a ground with no relief.
fn tooth_knob() -> f32 { return st.i.x; }
fn tooth_bearing() -> f32 { return st.h.z; }
fn weave_uv_scale() -> f32 { return st.i.y; }
fn weave_uv_bias() -> vec2<f32> { return st.i.zw; }

// The stroke's canvas region: premultiplied colour (latent·op, op) + wide aux
// (height, _, _, _). Read by `snapshot` (textureLoad) and `exchange` (bilinear).
@group(0) @binding(1) var region_color: texture_2d<f32>;
@group(0) @binding(2) var region_aux: texture_2d<f32>;
// Footprint snapshot targets (written by `snapshot`).
@group(0) @binding(3) var under_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var under_aux_w: texture_storage_2d<rgba16float, write>;
// Bilinear clamp sampler for the region / reservoir / coverage lookups.
@group(0) @binding(5) var samp: sampler;
// The brush tip's coverage mask over brush-local [-1,1]² (R8; §6.6) — the
// reservoir texel's own footprint weight. Read by the **tool** side only (`wick`
// and `exchange`); the canvas side uses the prefix-τ exposure instead.
@group(0) @binding(6) var cov_tex: texture_2d<f32>;
// The tool reservoir ping-pong: colour = (latent .xyz, per-unit opacity .w),
// aux = (carried height, _, _, _). `src` is read this dispatch, `dst` written;
// the host swaps the bind group after each pass that writes it — `wick` and
// `exchange`, so a painting segment cycles it twice. A bleed or settle slot
// writes it not at all and leaves it where the previous segment put it.
@group(0) @binding(7) var brush_src_color: texture_2d<f32>;
@group(0) @binding(8) var brush_src_aux: texture_2d<f32>;
@group(0) @binding(9) var brush_dst_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var brush_dst_aux_w: texture_storage_2d<rgba16float, write>;
// The footprint snapshot (read by `deposit`).
@group(0) @binding(11) var under_color: texture_2d<f32>;
@group(0) @binding(12) var under_aux: texture_2d<f32>;
// The region, written in place by `deposit`.
@group(0) @binding(13) var region_color_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var region_aux_w: texture_storage_2d<rgba16float, write>;
// The colour-dynamics noise field + its repeat sampler (deposit only): jitters
// the brush's own `add` paint exactly like the fast path's stamp (§6.2).
@group(0) @binding(15) var dyn_noise_tex: texture_2d<f32>;
@group(0) @binding(16) var dyn_noise_samp: sampler;
// The reservoir's **swept prefix** in the travel frame (written by `bake`, read by
// `deposit`): running integrals of the carried paint weighted by the footprint's
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

// The canvas surface's tileable ground (§6.4) — height in `r` (the media pass's),
// the rise one reach ahead in `gb` — whose rise along the travel scales the exposure
// a canvas texel sees. Read nearest (`ground_at`), so no sampler. Named
// by `deposit` and `settle`, the two entry points that put paint on the canvas.
@group(0) @binding(22) var surface_tex: texture_2d<f32>;
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
// the property adaptive flattening (`path::flatten`) depends on and uniform 2px
// sampling used to hide. The system is linear and autonomous in `e`, so running it
// over `e₁` and then `e₂` is running it over `e₁+e₂`; exposures add across the
// overlapping quads of consecutive segments, so the whole stroke applies the
// continuous path integral whatever the spacing.
//
// Getting this wrong is not subtle. What was here before solved only *half* the pair —
// the canvas relaxing towards a tool that never took anything back, `1 − exp(λ_lift·e)`
// lifted off and `k_dep(exp(λ_dep·e) − exp(λ_lift·e))/(λ_dep − λ_lift)` laid on — and
// then let the tool lift from the region as the `deposit` had already left it. The two
// sides therefore disagreed about how much had changed hands by `O(lift²)` per
// segment, and paint that goes nowhere has to show:
//
//   * at `lift = deposit = 0.95` and a segment of half a radius, 39% of the total
//     `canvas + tool` height vanished at *every* segment boundary. What survives is
//     front-loaded into the first quad that touches a spot, which draws arcs at
//     exactly the segment spacing through anything thick enough to see them.
//   * a stroke that stops leaves its last footprint stripped by a lift the tool was
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
// small quadrature disagreement into a larger height disagreement than the pair's
// gentler `1 − k·w(e)` did. That is the real content of the old note that the sliding
// kernel "gives up column-stochasticity" — the shares still sum to one exactly; it is
// the exposures underneath them that only agree approximately.

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
    // lifted the rest of the `e·A` minted back off before it set. A brush that does not
    // lift keeps all of it, which is the limit of that quotient and not a special case
    // in the physics; the branch is only because the quotient is 0/0 in floats.
    if kl < 1e-6 {
        out.add_w = e;
    } else {
        out.add_w = (1.0 - out.keep) / kl;
    }
    out.src_w = e - out.add_w;
    return out;
}

// The brush's remaining load at a point `arc` canvas px along the stroke: `drain`
// fades what it lays, linearly with distance travelled, until it runs dry (§6.2).
//
// **Every consumer evaluates this at its own position rather than per segment**, which
// is what keeps the paint laid independent of how the path was cut — the same property
// the exposure integral has, and for the same reason it matters. The falloff is linear
// in arc length, so sampling it where the paint actually lands is exact, not merely
// closer; it used to be baked in at the segment midpoint and bought back with a
// `0.02 / drain` cap on segment length (`gpu::stroke::budget::flatten_tolerance`).
fn stroke_drain(arc: f32) -> f32 {
    return max(1.0 - drain_rate() * arc, 0.0);
}

// The share of the tip's contact a canvas texel gets from the ground under it (§6.4)
// — the same `rise_ahead`/`tooth_gate` pair the swept fast path applies, from the same
// map, so a brush's tooth reads the same whether or not some other axis put it on this
// path.
//
// `rt` is in region px; `weave_uv_bias()` folds the piece's own canvas origin into
// the map, so the value is a pure function of canvas position exactly as the fast
// path's is.
// `turn` is how far the tip's heading has swung by this point (`segment_frame(rt)`),
// which `sweep_tangent` turns into the direction the rise is read along —
// after the guards, so an ordinary brush never pays for the rotation. A settle slot
// carries zero travel, so its heading is the direction the pen was going when it left
// the page, which is the ground the pass it is finishing was laying through.
//
// Both guards return exactly 1.0: a brush with no tooth, or a ground with no relief,
// leaves every rate below the float it always was. The first is also what keeps the
// **bleed slots** clear of all of this — they carry `tooth = 0` (`bleed_fires`), so a
// lateral flux is never gated by the weave and never reads a direction. That is not
// bookkeeping: bleed is wet paint spreading sideways on the canvas, not a tip dragged
// over it, so there is no travel to read the ground's rise along, and gating it would
// break the flux's antisymmetry (the two threads of a pair sit over different ground).
fn surface_tooth(rt: vec2<i32>, turn: f32) -> f32 {
    if weave_uv_scale() <= 0.0 || tooth_knob() <= 0.0 {
        return 1.0;
    }
    let uv = vec2<f32>(rt) * weave_uv_scale() + weave_uv_bias();
    let dims = textureDimensions(surface_tex);
    let g = textureLoad(surface_tex, ground_at(uv, dims), 0);
    return tooth_gate(rise_ahead(g, sweep_tangent(seg_dir(), turn)), tooth_knob());
}

// The footprint's **optical depth** for a given coverage — the currency every
// transfer in this loop is measured in, on both sides. The canvas side gets it from
// the prefix-τ volume's differences; the tool side, which has no prefix to
// difference, converts its own coverage the same way the volume was built
// (`assets::build_prefix_tau`), clamp included, so the two agree.
fn tau_of(cov: f32) -> f32 {
    return -log(1.0 - min(cov, 0.999));
}

// Footprint coverage at brush-local `local` ∈ [-1,1]² (0 outside — image masks may
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

// ---- 0. Wick: paint migrating within the tip. ------------------------------------
//
// The reservoir is a grid of independent cells: each one trades with the canvas
// directly beneath it and with nothing else. Real paint on a tool does not hold still
// like that — it creeps between the bristles — and leaving that out is visible,
// because the *rate* each cell trades at varies enormously across the tip.
//
// `exchange` keys a cell's exposure on its own optical depth, and τ is flat across the
// interior (the 0.999 coverage clamp caps it) and then falls off a cliff over the few
// texels of a hard tip's shoulder — at `hardness = 0.95` the shoulder cells exchange
// some fifty times slower than the interior. So when a stroke's own paint runs out
// (`drain`) and the tip is smearing what it picked up, the interior cells empty in step
// with the fading trail while the shoulder ring goes on holding paint lifted hundreds
// of pixels back. Nothing relaxes it, and the deposit lays each lateral row's load back
// at its own offset, so the ring prints: a rim of surplus paint at the shoulder with a
// scraped groove just inside it, curving into a tip-shaped chevron where the stroke
// stops. `golden_lift_end_regression` pins it.
//
// So the cells trade with their four neighbours, in **flux form**: a pair swaps a share
// of its difference, one thread adding what the other subtracts. That makes the whole
// pass a pure internal redistribution — total height and optical mass over the tool are
// unchanged to the last bit the float arithmetic keeps — which is why it can sit in the
// loop without touching the conservation argument the transfer rests on (§6.1).
//
// **It runs before `bake`, not between `bake` and `exchange`.** Those two are the two
// halves of one transfer and only add up if both read the reservoir as the segment
// found it; wicking between them would let the canvas receive a share of one reservoir
// while the tool gave up a share of another, and the halves would disagree by the flux.
// Ahead of both, it is simply part of what the tool arrives at the segment carrying.
//
// **What the flux chases is concentration, not height.** A tool holds more paint where
// there is more of it in contact, so the state a tip relaxes towards is `height ∝
// coverage` — its own shape — and the thing that actually equalises is height *per unit
// coverage*. Driving the flux on raw height instead would make uniform height the
// equilibrium, which is a different brush: it pumps paint out of the interior into the
// barely-touching shoulder, and the whole stroke lays down ~5% of a radius wider with a
// softer edge. Measured, not reasoned about — that version widened this test's body
// edge from 15 to 32 levels before the concentration form replaced it.
//
// The mobility is `min` of the two cells' coverage — symmetric, so the flux stays
// antisymmetric, and vanishing where the tip does. That second property matters twice
// over: no paint is dragged out into the zero-coverage corners of the square reservoir,
// where nothing would ever deposit it again, and a cell out of contact carries nothing
// along. It is also what bounds the scheme: a cell can shed at most `min(c_n, c_t)/c_t
// ≤ 1` of its own concentration to each neighbour.

// How readily paint migrates within the tip, per unit of travel in radius units.
//
// Tuned against `golden_lift_end_regression`, measuring the ring as the worst rise
// above a groove along any lateral profile — a stroke's paint may only fall off as you
// walk away from its axis, so any rise is a rim sitting outside a scrape. Rates 1, 2,
// 4, 8 gave 2.0, 1.67, 1.0, 1.67 levels. 4 is the optimum from both ends: below it the
// worst rise is still *at the stroke end*, which is the ring itself, and above it the
// ring is gone but the rim the wick feeds starts registering instead.
//
// There is a real cost on the other side, which is why this is not simply set high.
// Wicking hands paint to shoulder cells that used to hold almost none — the same rate
// disparity that stranded the ring also starved them — so they now deposit, and the
// stroke's own edge softens: the same test's body edge climbs 25 → 29 → 33 → 36 levels
// across the rates above. Arguably the starved rim was the older defect of the two, but
// it is a visible change to every paint-manipulating brush and not one to spend for
// free, which is what makes the optimum an optimum rather than a floor.
//
// Read it as a **variance rate**: travelling `lr` injects `σ² = 0.5 · WICK_RATE · lr`
// texels² per axis, and since variance adds under composition, the total over a stroke
// is proportional to the distance travelled however the path was cut and however the
// cadence below chops it up. It was 8 while the retired saturating form was silently
// delivering a fraction of nominal at the step this loop actually runs at; with the
// scheme honest the same smoothing costs about half the number.
//
// It also sets the wick's **cadence**: one pass of the kernel below carries
// `WICK_HALF / WICK_RATE` of travel, which is what `stroke::WICK_TRAVEL_QUANTUM` must
// equal.
const WICK_RATE: f32 = 4.0;

// **The wick keeps its own cadence, and that is what bounds the step.** It used to run
// once per segment, whatever a segment happened to be, and carry however much travel
// that was by pushing the flux's *reach* outwards — `d = sqrt(WICK_RATE·lr)`, on the
// argument that variance is `Σ w·d²`, so moving the same weight further buys `d²` times
// the smoothing under the same `Σw ≤ 1` stability bound.
//
// The argument is true about variance and wrong about *conditioning*, and the failure is
// not gradual. A four-point stencil at integer distance `d` couples a cell only to cells
// of the same parity in `d`, so at even reach the grid's two sublattices stop talking:
// the checkerboard mode `(−1)^(x+y)` has eigenvalue exactly **1** and never decays,
// while `Σw = 1` leaves the cell *none* of its own value. That is not a corner —
// `exchange_travel` caps at `MAX_EXCHANGE_TRAVEL = 1`, and `WICK_RATE · 1 = 4` puts the
// reach at exactly 2, so **every brush at the relaxed end of the range ran there**.
// Measured across `lift = deposit`: the checkerboard eigenvalue is 0 at 0.95/0.95 (the
// point `WICK_RATE` was tuned at, and the only well-conditioned one), −0.86 at 0.8/0.8,
// and 1.0 for anything relaxed to the cap.
//
// So the reach is gone and the *cadence* carries the step instead: the host wicks once
// per `WICK_TRAVEL_QUANTUM` of travel (`stroke::WICK_TRAVEL_QUANTUM`, in radii),
// independent of where the segment boundaries fell, and each step is one fixed
// nearest-neighbour stencil sized to exactly that quantum. Sub-stepping in time was
// rejected before on the grounds that this loop is dispatch-bound — but the cadence is
// *slower* than the old one wherever the loop is expensive (a tight brush's segments are
// shorter than the quantum, so half of them now skip the pass entirely) and faster only
// where segments are long and therefore few. Total wick dispatches over a stroke are
// `travel / quantum`, which no longer depends on the segmentation at all.
//
// **The stencil is a separable binomial, run as two 1-D passes**, and the quantum is
// whatever that kernel's variance buys. The 1-D binomial `C(2m, m+k) / 4^m` has
// variance `m/2` and frequency response `cos^{2m}(θ/2)` — exactly zero at Nyquist, so
// the checkerboard mode is annihilated rather than merely damped, at every `m`. Setting
// `m/2 = 0.5·WICK_RATE·lr` gives the travel one pass carries:
//
//     lr = m / WICK_RATE      →      WICK_TRAVEL_QUANTUM
//
// **`m = 1` is the stencil this replaced**, exactly: `C(2, 1+k)/4 = [¼, ½, ¼]`, which
// is the old `[w, 1−2w, w]` at its `w = ¼` limit. So this is a strict generalisation of
// a tuned behaviour rather than a new one, and the total variance a stroke receives is
// unchanged — only how it is parcelled out.
//
// The two passes are each internally conservative (a 1-D symmetric flux exchange, one
// thread adding what its partner subtracts), so composing them conserves exactly, and
// the concentration form is re-derived between them. That is why this can be two passes
// at all: a non-separable 2-D kernel of the same variance would need `(2m+1)² − 1` taps
// in one pass — 80 at `m = 4` — to save a single dispatch.
//
// **`m` is deliberately not pushed as far as it will go.** Widening trades cost for a
// coarser cadence, and the cadence has a price the variance argument does not see: a
// firing happens at the *start of the segment* the boundary fell in, so its position
// jitters by up to one segment length, and a kernel carrying four times the variance
// per firing amplifies that jitter fourfold. Measured on
// `a_carried_stroke_is_independent_of_how_the_path_was_cut`:
//
//     m       1     2     4     8
//     levels  2     3     4     5      (the test's bound is 4)
//
// `m = 2` is where that stops being free: it halves the taps per unit of travel for one
// level, and is dispatch-neutral. `m = 4` would halve the dispatches too, for another
// level — but the previous constant to trade this same quantity, `RESERVOIR_EXCHANGE_STEP`,
// was tightened *away* from that level once already, so doubling it back for a wick
// saving is not the trade to make here.
//
// **The place to widen is the march, not this loop**, which is the other reason to hold
// at 2. Here both axes are dispatches and so must share a cadence; in stroke space the
// along-travel pass runs inside a lateral row's own workgroup for free, so only the
// across-row pass is a barrier and only *it* wants a wide kernel. Committing to a wide
// `m` now would bank the jitter on both axes to save dispatches on one.
//
// Offsets stay *integer*, which is load-bearing rather than convenient: the pair
// `(x, x+d)` is then symmetric, so both threads compute one number and apply it with
// opposite signs. Sampling a neighbour at a fractional distance through the bilinear
// sampler would quietly break conservation, because the partner is a blend of four cells
// whose reciprocal flux does not mirror it.

// Half-width of the 1-D binomial, and the kernel itself indexed by `|k|`. **Must match
// `stroke::WICK_TRAVEL_QUANTUM`**, which is `WICK_HALF / WICK_RATE`.
const WICK_HALF: i32 = 2;
const WICK_KERNEL = array<f32, 3>(
    6.0 / 16.0,
    4.0 / 16.0,
    1.0 / 16.0,
);

// One neighbour pair's parcel for this cell. `weight` is the share of the pair's
// concentration difference that crosses, and the mobility folded in on top converts
// that concentration back into the height it moves. Every factor is symmetric in the
// pair, so the neighbour computes the identical number and applies it with the opposite
// sign — which is what conserves the total exactly.
fn wick_pair(
    t: vec2<i32>,
    off: vec2<i32>,
    bdim: vec2<u32>,
    cov_t: f32,
    inv_t: f32,
    self_load: Load,
    weight: f32,
) -> Load {
    let tn = t + off;
    let n = load_at(tn, bdim);
    let cov_n = coverage_at(brush_local_at(tn, bdim));
    let f = weight * min(cov_t, cov_n);
    let inv_n = 1.0 / max(cov_n, WICK_MIN_COV);
    var d: Load;
    d.h = f * (n.h * inv_n - self_load.h * inv_t);
    d.m = f * (n.m * inv_n - self_load.m * inv_t);
    d.lm = f * (n.lm * inv_n - self_load.lm * inv_t);
    return d;
}

// Coverage floor for the concentration quotient. Cells below it are out of contact —
// nothing ever deposits from them and the mobility above already zeroes their flux — so
// this only keeps the division finite, it never decides a pixel.
const WICK_MIN_COV: f32 = 1e-4;

// What one reservoir cell carries, in the quantities the flux moves: height, optical
// mass (opacity·height), and the latent riding as a mass weight — the same
// mass-weighted form the rest of the loop blends colour in, so a cell that gains paint
// from a neighbour gains that neighbour's colour in proportion to what it brought.
struct Load {
    h: f32,
    m: f32,
    lm: vec3<f32>,
}

fn load_at(t: vec2<i32>, bdim: vec2<u32>) -> Load {
    let tc = clamp(t, vec2<i32>(0, 0), vec2<i32>(bdim) - vec2<i32>(1, 1));
    let c = textureLoad(brush_src_color, tc, 0);
    let a = textureLoad(brush_src_aux, tc, 0);
    var out: Load;
    out.h = a.x;
    out.m = c.a * a.x;
    out.lm = c.rgb * out.m;
    return out;
}

// A reservoir texel's brush-local coordinates, clamped into the grid — so a cell on the
// border reads *itself* as its outside neighbour, the difference is exactly zero, and
// the rim is a no-flux wall with no special case.
fn brush_local_at(t: vec2<i32>, bdim: vec2<u32>) -> vec2<f32> {
    let tc = clamp(t, vec2<i32>(0, 0), vec2<i32>(bdim) - vec2<i32>(1, 1));
    return (vec2<f32>(tc) + vec2<f32>(0.5, 0.5)) / vec2<f32>(bdim) * 2.0 - vec2<f32>(1.0, 1.0);
}

// One 1-D pass of the binomial along `axis`. Both entry points below are this with a
// different axis, and running the two in sequence realises the separable 2-D kernel.
//
// Nothing here couples the two axes, which is the property the stroke-space march
// depends on: the along-travel pass touches only cells of the same lateral row, so a
// workgroup that owns a row can run it with no barrier, and only the across-row pass
// has to be a dispatch of its own.
fn wick_axis(gid: vec3<u32>, axis: vec2<i32>) {
    let bdim = textureDimensions(brush_src_color);
    if gid.x >= bdim.x || gid.y >= bdim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let self_load = load_at(t, bdim);
    let cov_t = coverage_at(brush_local_at(t, bdim));
    let inv_t = 1.0 / max(cov_t, WICK_MIN_COV);

    // Every share is taken against the *entering* state on both sides, never against a
    // running total: that is what makes this thread's flux the exact negative of the
    // one its neighbour computes for the same pair — the mobility, the weight and the
    // two concentrations are all symmetric in the pair, so the two threads compute one
    // number and disagree only in sign.
    var d: Load;
    d.h = 0.0;
    d.m = 0.0;
    d.lm = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 1; k <= WICK_HALF; k = k + 1) {
        let w = WICK_KERNEL[k];
        let a = wick_pair(t, axis * k, bdim, cov_t, inv_t, self_load, w);
        let b = wick_pair(t, axis * -k, bdim, cov_t, inv_t, self_load, w);
        d.h += a.h + b.h;
        d.m += a.m + b.m;
        d.lm += a.lm + b.lm;
    }

    let h = self_load.h + d.h;
    let m = self_load.m + d.m;
    let lm = self_load.lm + d.lm;
    // Colour comes back out of the mass weight it travelled as. With no mass there is
    // no colour to recover, so the cell keeps the latent it had rather than a quotient
    // of two vanishing numbers (§6.1).
    var lat = textureLoad(brush_src_color, t, 0).rgb;
    if m > PAINT_EPS {
        lat = lm / m;
    }
    textureStore(brush_dst_color_w, t, vec4<f32>(lat, m / max(h, PAINT_EPS)));
    textureStore(brush_dst_aux_w, t, vec4<f32>(h, 0.0, 0.0, 0.0));
}

// Along the tip's travel axis — the one a stroke-space row could run internally.
@compute @workgroup_size(8, 8)
fn wick_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    wick_axis(gid, vec2<i32>(1, 0));
}

// Across it — the one that couples lateral rows, and so must stay a dispatch.
@compute @workgroup_size(8, 8)
fn wick_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    wick_axis(gid, vec2<i32>(0, 1));
}

// ---- 1. Bake: the reservoir's swept prefix along the travel axis. ----------------
//
// During a segment the tip does not present one texel to a canvas point — it drags
// a whole *range* of texels across it. A point at travel-frame `(xl, yl)` is served
// by tool texel `(xl − s, yl)` as the tip advances by `s`, so what it receives is
//
//     ∫₀^lr R(xl − s, yl) · τ(xl − s, yl) ds
//
// a convolution along travel between the reservoir `R` and the footprint's
// optical-depth density. Sampling `R` once mid-pass (what this used to do) is a
// one-tap midpoint rule for that integral: first order, with error growing in
// segment length, which is why it could only ever be traded against step size.
//
// The integral has the same shape as the exposure itself, so it yields to the same
// trick: **`prefix_at` is this integral with `R ≡ 1`**. Accumulate `R·dτ` along the
// row and the deposit becomes a difference of two prefix sums — O(1), and exact for
// any segment length. Baked per segment, so the tip's orientation (the reservoir
// lives in the shape frame, §6.6) and the reservoir's own evolution are both folded
// in with no approximation. `dτ` comes from the prefix-τ volume's own differences,
// so bake and exposure integrate numerically identical footprints.
//
// One **workgroup per row**, one thread per column, with the running integral
// built by a shared-memory inclusive scan (Hillis–Steele, log₂(BAKE_RES) steps).
// The loop this replaced — one thread walking its whole row — put 128 threads on
// the device per dispatch and a 128-iteration serial chain of texture reads on
// each, and since the bake runs once per segment inside the sequential stamp
// loop, that near-idle dispatch dominated the whole dynamics path. The scan
// computes the same prefix sums (up to float re-association) with every lane of
// a full-width dispatch busy.

// Must equal the host's BAKE_RES (the bake textures' side).
const BAKE_RES: u32 = 128u;

var<workgroup> ws_e: array<f32, BAKE_RES>;
var<workgroup> ws_load: array<vec4<f32>, BAKE_RES>;
var<workgroup> ws_latm: array<vec4<f32>, BAKE_RES>;

@compute @workgroup_size(128) // = BAKE_RES: thread i scans column i of its row
fn bake(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let dim = textureDimensions(bake_load_w);
    let row = wid.y; // dispatched as (1, dim.y, 1), so every row is in range
    let i = lid.x;
    let yl = (f32(row) + 0.5) / f32(dim.y) * 2.0 - 1.0;
    let theta = orientation() * TAU;
    let cs = cos(theta);
    let sn = sin(theta);

    // This column's share of the exposure: the prefix-τ difference across it.
    // Each thread reads its right edge; the left edge is the neighbour's.
    let x1 = (f32(i) + 1.0) / f32(dim.x) * 2.0 - 1.0;
    ws_e[i] = prefix_at(x1, yl, orientation());
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
    ws_latm[i] = vec4<f32>(b_c.rgb * m, 0.0) * de;
    workgroupBarrier();

    // Inclusive scan over both accumulators together.
    for (var off = 1u; off < BAKE_RES; off = off << 1u) {
        var v_load = ws_load[i];
        var v_latm = ws_latm[i];
        if i >= off {
            v_load += ws_load[i - off];
            v_latm += ws_latm[i - off];
        }
        workgroupBarrier();
        ws_load[i] = v_load;
        ws_latm[i] = v_latm;
        workgroupBarrier();
    }

    // Out-of-range stores (a texture narrower than the workgroup) are no-ops.
    let t = vec2<i32>(i32(i), i32(row));
    textureStore(bake_load_w, t, ws_load[i]);
    textureStore(bake_latm_w, t, ws_latm[i]);
}

// ---- 2. Snapshot the region under the segment's footprint rect. -----------------

// This texel's place in the segment's travel frame — the one the whole exchange is
// measured in. Straight or curved, `sweep_at` hands back the same three quantities
// (see stamp_common.wesl), so nothing downstream branches on the segment's shape.
//
// `travel_radii()` carries the travel as `length / radius`, so the arc length it was built
// from is recovered by multiplying back through.
fn segment_frame(rt: vec2<i32>) -> Sweep {
    let p = vec2<f32>(rt) + vec2<f32>(0.5, 0.5);
    return sweep_at(p, seg_start(), seg_dir(), radius(), travel_px(), curvature());
}

// Whether a region texel is provably outside the segment's sweep — the span
// x ∈ (-1, lr+1), |y| < 1 in the travel frame — so its exposure is exactly 0 and
// `deposit` leaves it untouched. The dispatch rect is the sweep's AABB, so at a
// diagonal orientation up to half its texels are these corners; the test rejects
// them with a few ALU ops instead of the prefix-τ taps `deposit` would otherwise
// spend proving e = 0. `snapshot` may skip exactly the same texels because
// `deposit` reads the snapshot only after its own e > 0 check, and the two compute
// the identical frame from the identical uniform — same floats, same verdict.
fn outside_sweep(sw: Sweep) -> bool {
    // Beyond the lateral edges the prefix rows are the footprint's own zero
    // border; past the trailing edge both taps clamp to the row total. Either
    // way the difference is exactly 0. The lateral test comes first because it is
    // also what rejects a texel near a tight arc's centre of curvature, where the
    // travel coordinate has no meaning to test.
    return abs(sw.local.y) >= 1.0 || sw.local.x <= -1.0 || sw.local.x - sw.lr >= 1.0;
}

// One footprint texel's copy. Factored out because the segment loop runs this from the
// tail of `exchange`'s grid rather than from a dispatch of its own — see there — while
// the pen-up still needs it as a standalone pass.
fn snapshot_at(gx: u32, gy: u32) {
    let udim = textureDimensions(under_color_w);
    if gx >= udim.x || gy >= udim.y {
        return;
    }
    let t = vec2<i32>(i32(gx), i32(gy));
    let rt = under_origin() + t;
    if outside_sweep(segment_frame(rt)) {
        return; // deposit never reads these texels of the snapshot
    }
    let rdim = vec2<i32>(textureDimensions(region_color));
    var c = vec4<f32>(0.0);
    var a = vec4<f32>(0.0);
    if rt.x >= 0 && rt.y >= 0 && rt.x < rdim.x && rt.y < rdim.y {
        c = textureLoad(region_color, rt, 0);
        a = textureLoad(region_aux, rt, 0);
    }
    textureStore(under_color_w, t, c);
    textureStore(under_aux_w, t, a);
}

// The standalone pass, which only the pen-up now dispatches: the `settle` reads the
// snapshot, so unlike the segment case it cannot share a dispatch with its producer.
@compute @workgroup_size(8, 8)
fn snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    snapshot_at(gid.x, gid.y);
}

// How many sub-steps `exchange` walks its texel's track in. The canvas side gets its
// sweep exactly, from the prefix-τ; the tool side has no prefix to difference — the
// canvas under it is an arbitrary evolving field — so it is a quadrature, and this is
// how fine.
//
// **Pinned at 1, and for a different reason than it used to be.** Under the closed-pair
// kernel raising it was outright *wrong* — over each sub-step the tool lifted its share
// of a canvas held fixed, N times over, while the `deposit` gave up a single share of
// the whole exposure, so the two halves stopped being complements of one solution and
// the transfer stopped balancing (N = 8 landed 12 levels rms from where N = 1 converged,
// even at a segment length four times finer than the loop runs at).
//
// The sliding kernel has no such objection: each direction is a plain exponential in
// exposure, and `exp(−k·e/N)^N = exp(−k·e)` exactly, so splitting the exposure changes
// the tool's own arithmetic not at all. All N > 1 buys is sampling the canvas at
// several points along the texel's track instead of one — and that was measured, both
// as sub-steps and as an average over the track, and moves the result by less than the
// 8-bit noise floor. The midpoint tap is simply not where the error is; see
// `RESERVOIR_EXCHANGE_STEP`, which is. So it stays at 1 because the taps are not worth
// paying for, and it is now free to raise if that ever changes.
const EXCHANGE_STEPS: u32 = 1u;

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
    let start = seg_start();
    let dir = seg_dir();
    let s = f * travel_px(); // travel = (length / radius) · radius · f
    let k = curvature();
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
// the whole reason this runs per segment rather than on a cadence of its own: the
// tool used to be reloaded every `spacing·radius` while the canvas was stripped every
// segment, and a stroke ending mid-interval left the difference behind as a
// tip-shaped patch of missing paint.
//
// **The tool's side is swept too.** A reservoir texel does not hover over one spot for
// a segment — it is dragged along a track `lr` radii long, over a canvas that is
// changing under it, and the canvas side integrates over exactly that track through
// the prefix-τ. This used to be a single bilinear tap at the segment's midpoint: the
// midpoint rule, second-order in the segment length, and the only thing left in the
// loop whose answer depended on how finely the path happened to be cut.
//
// That dependence is visible, and it is *not* the kind that hides behind a tolerance:
// the flattener's segment lengths follow the pointer samples, and a hand varies its
// speed, so the same gesture drawn twice cut the tool's decay differently and the
// smear faded at a different rate. Measured on a pure smear dragged out of a band of
// paint, the same 300px stroke drawn with 2, 6, 20 and 80 pointer samples came out up
// to 11 levels apart, and 15% away from the converged answer.
//
// So the track is walked in [`EXCHANGE_STEPS`] sub-steps, each with its own canvas tap
// and its own share of the exposure. The kernel composes exactly, so splitting the
// exposure changes nothing on its own — what the sub-steps buy is the *canvas* being
// read where the texel actually was, which is the whole error term. The tip's frame is
// re-evaluated at each one, so a curved sweep turns under the tool as it goes.
// **The segment's `snapshot` rides in the tail of this dispatch.** The two are
// independent — the snapshot only reads the region, which the previous segment's
// `deposit` last wrote, and writes the `under` scratch; the exchange reads that same
// region and writes the reservoir — so the barrier that used to separate them was
// paying for an ordering neither needed. Both must finish before `deposit`, and
// sharing one dispatch is the cheapest way to say exactly that.
//
// Workgroups `[0, ex_groups)` on x are the reservoir; the rest are the footprint,
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
    var lat = b_c.rgb;

    // This texel's exposure to the whole segment: its own optical depth × the travel
    // (radius units). In τ, like the canvas side — the canvas gives up
    // `1 − keep(∫τ)`, so a tool that measured its gain in bare coverage would
    // receive a fraction of what the canvas lost and quietly destroy paint.
    //
    // Scaled by the **bearing fraction** (§6.4): the tip only trades where it touches,
    // and a reservoir cell has no ground of its own to ask — it is dragged over a
    // fresh stretch of canvas at every sub-step. So it books against the ground's
    // mean *for the way this segment is going*, which the host read off the ground's
    // per-direction table (`Surface::bearing`, `tooth_bearing()`), while each canvas
    // texel scales
    // by the ground ahead of itself. The two agree in expectation over any footprint spanning many
    // grain features — which is every usable tip — and that is what keeps a toothed
    // smear conserving. Exactly 1 when there is no tooth, so an ordinary brush is
    // untouched.
    let e_step =
        tau_of(coverage_at(local)) * travel_radii() * tooth_bearing() / f32(EXCHANGE_STEPS);
    let ex = exchange_at(e_step);
    let rdim = vec2<f32>(textureDimensions(region_color));
    // The tool's shape frame is the travel frame turned by the shape's orientation
    // (§6.6) — the same rotation `bake` applies to read the reservoir.
    let theta = orientation() * TAU;
    let cs = cos(theta);
    let sn = sin(theta);

    for (var i = 0u; i < EXCHANGE_STEPS; i = i + 1u) {
        // Midpoint of this sub-step's stretch of the track.
        let f = (f32(i) + 0.5) / f32(EXCHANGE_STEPS);
        let tip = tip_at(f);
        let dir = tip.zw;
        let rot = vec2<f32>(dir.x * cs - dir.y * sn, dir.x * sn + dir.y * cs);
        let world = tip.xy + (local.x * rot + local.y * vec2<f32>(-rot.y, rot.x)) * radius();
        let uv = world / rdim;

        // Outside the selection the canvas gives nothing up and the source mints
        // nothing (§6.8) — the matching reduction on the canvas side is the deposit's
        // own mask lerp, so height stays conserved. Bilinear like the region taps
        // below: this texel is over an arbitrary sub-pixel spot, and a 1×1 constant
        // mask reads the same at every uv.
        let sel = textureSampleLevel(sel_mask, samp, uv, 0.0).r;
        let rc = textureSampleLevel(region_color, samp, uv, 0.0);
        let ra = textureSampleLevel(region_aux, samp, uv, 0.0);

        // What it keeps of its own load. Exactly the complement of the fraction the
        // `deposit` after us lays from it, over the very exposure it lays over — so
        // there is no correction factor.
        //
        // There used to be one (`DEPLETE_FACTOR = 0.75`, "deplete a bit slower than the
        // math would indicate to account for error"). The error it was accounting for
        // was real but misdiagnosed: the two sides were solving different equations, so
        // they disagreed about how much had changed hands and the tool over-drained.
        // Draining slower papered over that — but a factor below 1 means the tool keeps
        // a fraction of every deposit it makes, which compounds, so paint grew without
        // bound over a long stroke. With both sides now taking complementary shares of
        // one solution there is nothing left to correct, and the exact rate is also the
        // conserving one.
        h = h * (1.0 - ex.dep);
        m = m * (1.0 - ex.dep);

        // Height moves; colour and per-unit opacity ride as optical-mass-weighted
        // blends (§6.1). Two parcels land on the tool — what it took off the canvas,
        // and the share of the brush's own `add` it lifted back before that paint had
        // settled — each stacked with its own latent, so a tip smearing red through
        // white carries the mixture rather than an average of the two errands.
        let take_h = ra.x * (1.0 - ex.keep) * sel;
        let take_m = rc.a * take_h;
        h = h + take_h;
        // Only mix a colour in when there is optical mass to carry it. The region is
        // stored premultiplied, so recovering a latent is a division by the opacity —
        // and out in a stroke's faded tail that opacity is a rounding error, so the
        // quotient is enormous. `blend_latent` weights it by its own vanishing mass,
        // but "enormous × almost nothing" is not nothing: it leaves a few tenths of a
        // wild colour on the tool, which the deposit then combs out along the stroke as
        // thin bright streaks. The single mid-segment tap this replaced could hit that
        // too, but once per segment and at the midpoint; walking the track samples the
        // ends as well, where the canvas is emptiest, and eight times as often.
        if take_m > PAINT_EPS {
            lat = blend_latent(lat, m, rc.rgb / rc.a, take_m);
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
        let arc = arc_at_start() + f * travel_px() + (local.x * cs - local.y * sn) * radius();
        let drain = stroke_drain(arc);
        let src_h = add_rate() * drain * ex.src_w * sel;
        let src_m = brush_op() * drain * src_h;
        lat = blend_latent(lat, m, brush_lat(), src_m);
        h = h + src_h;
        m = m + src_m;
    }

    textureStore(brush_dst_color_w, t, vec4<f32>(lat, m / max(h, PAINT_EPS)));
    textureStore(brush_dst_aux_w, t, vec4<f32>(h, 0.0, 0.0, 0.0));
}

// ---- 4. Deposit: the swept exchange over one segment. ----------------------------

// The `bleed` axis (§6.2): the paint already on the canvas diffusing laterally under
// the tip. One share of a texel's difference from each neighbour crosses per
// **firing**, keyed on the pair's swept exposure over the firing's travel window —
// `1 − exp(−k_bleed·e)`, the same saturating form as every other rate, so
// consecutive windows compose to first order and a texel the tip never covers never
// moves.
//
// **The bleed runs on dedicated slots, not on the painting segments — the wick's
// cadence, realised as geometry.** Real slow input is fitted at a control point per
// pointer sample (a measured repro: 177 knots over 68 px, segments of 0.39 px), and
// at that cut a per-segment bleed is broken twice over: the flux
// `share·w·Δ ≈ 1e-4` sits under the f16 ULP of the heights it edits, so stores
// either snap it away or ratchet a whole ULP into a directional drift (20 levels of
// ghost at 176 spans against 2 at 44); and the per-texel exposure `dpre` is a
// prefix difference over 3e-4 radii — cancellation noise, whose arbitrary sign
// draws crescents of misclassified texels through the footprint. So the host emits
// one **bleed slot** per crossing of `BLEED_TRAVEL_QUANTUM` of absolute arc
// (`stroke::dynamics::dynamics_plan`): a straight quad whose sweep IS the firing's travel
// window — the chord over the last quantum of path — with every vertical rate and
// the source zeroed, and λ_bleed alone nonzero. This dispatch then needs nothing
// special: the slot's own `dpre`, over half a radius of travel, is the window's
// exposure and is numerically healthy; participation, pairing and the no-flux wall
// all sit at the window-sweep's rim, where the weight fades to zero smoothly.
// Keyed on absolute arc, the cadence is a pure function of the record, so every
// cut of the same path fires identically (§6.2, live == committed) — and painting
// segments carry λ_bleed = 0, so between firings the canvas takes the no-bleed
// path bit-for-bit.
//
// **Flux form, like the wick**: both threads of a pair compute the same number from
// the same `under` snapshot — the neighbour's exposure is re-derived from the same
// uniform, so the same floats — and apply it with opposite signs. The redistribution
// is internal to the canvas, so it conserves height and optical mass to rounding
// (not to the bit, as the wick does: the delta rides through this thread's own
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
// **The reach scales with the tip, or the axis is decorative.** A stencil is
// bounded: one application moves at most `Σshare·d²` texels² of variance, and the
// per-firing blend saturates at 1, so a 1-texel stencil tops out near σ ≈ 1 px
// per pass of the tip whatever the rate — imperceptible under any brush big
// enough to blur with (measured: cranking the knob past its 0.95 λ-clamp changed
// nothing visible, because nothing past the clamp *can*). So the longest tap is
// a fixed fraction of the radius, which is what makes `bleed` mean the same
// *look* at every brush size — the same property the tapers get from being
// quoted in radii. At full crank a pass of the tip buys σ ≈ 0.3·radius.
const BLEED_REACH: f32 = 0.25;

// Three tap scales per direction — 1 px, a quarter of the reach, the reach —
// because a sparse pair of ±d taps alone decouples the grid into d² sublattices
// (the wick's parity failure, generalized): sub-d texture would ride through a
// "blur" untouched. The 1 px taps couple every texel to its actual neighbours, so
// every non-zero frequency strictly decays; the long taps carry the variance; the
// middle scale fills the response in between. Shares are per tap, 12 taps total:
// worst-case eigenvalue is `1 − 8·Σshare·w = 1 − w ≥ 0` (Σshare = 1/8), so at
// full saturation the worst mode is annihilated rather than flipped — the same
// margin the 5-point form had, spent on reach instead of slack. Total shed is
// bounded by `4·Σshare·w = w/2`: no segment can overshoot, and positivity of
// height and mass survives any rate.
//
// Per unit of exposure, in the composing regime, the stencil injects
// `σ² ≈ 2·k_bleed·Σ(share·d²)` texels² per axis; at saturation each firing
// contributes its capped share and scrubbing keeps buying variance linearly in
// travel — like a real blender, more passes blur further.
const BLEED_SHARES = array<f32, 3>(
    1.0 / 64.0,
    2.0 / 64.0,
    5.0 / 64.0,
);
const BLEED_OFFS = array<vec2<i32>, 4>(
    vec2<i32>(1, 0),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 1),
    vec2<i32>(0, -1),
);

@compute @workgroup_size(8, 8)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_color);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let rt = under_origin() + t;
    let rdim = vec2<i32>(textureDimensions(region_color_w));
    if rt.x < 0 || rt.y < 0 || rt.x >= rdim.x || rt.y >= rdim.y {
        return;
    }
    // Travel-frame brush-local coords (x in radius units from the segment start).
    // The AABB's corners are outside the sweep and provably at zero exposure
    // (`outside_sweep`): reject them before the prefix-τ taps below spend ~10 loads
    // apiece computing that zero.
    let sw = segment_frame(rt);
    if outside_sweep(sw) {
        return;
    }
    let xl = sw.local.x;
    let yl = sw.local.y;

    // The swept exposure — the brush's coverage integrated as it travels this
    // segment, as a difference of prefix sums (identical to the plain deposit's
    // `swept_depth`). Overlapping segment sweeps add their exposures, so the
    // exponential rates below compose exactly across the whole stroke.
    //
    // `dpre` is that difference in *this texel's own* measure of travel and `e` is it
    // in the tip's, which on a curved segment differ by `sw.warp` (see `sweep_at`).
    // Both are needed: every exchange rate below is per unit tip travel, but the bake
    // was integrated against the same prefix and its ratios must be taken in the
    // measure it was integrated in — where the factor cancels rather than squaring.
    let dpre = max(
        prefix_at(xl, yl, orientation()) - prefix_at(xl - sw.lr, yl, orientation()),
        0.0,
    );
    if dpre <= 0.0 {
        return; // outside the swept footprint: the region keeps its value
    }
    // The ground's share of this texel (§6.4). It scales the **exposure**, so the whole
    // exchange below runs on the contact this texel actually had: the four shares stay
    // exact complements of one exposure, so the canvas takes precisely what the tool
    // gives up, and each stays in the form that composes across segment cuts. Scaling
    // the finished shares instead would do neither.
    let e = dpre * sw.warp * surface_tooth(rt, sw.turn);
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

    let uc = textureLoad(under_color, t, 0);
    let ua = textureLoad(under_aux, t, 0);
    // The tool's carried paint over this spot, as the tip actually presented it:
    // the swept prefix differenced across the segment (see `bake`) gives the
    // exposure-weighted integral of the reservoir over the whole pass, so dividing
    // by the exposure recovers the mean load that pass delivered. Exact for any
    // segment length — no mid-pass sample, no length-dependent error.
    let d_load = bake_at(bake_load, xl, yl) - bake_at(bake_load, xl - sw.lr, yl);
    let d_latm = bake_at(bake_latm, xl, yl) - bake_at(bake_latm, xl - sw.lr, yl);
    // Divided by the prefix difference in the *same* measure the bake integrated
    // against, not by the tip-travel exposure: these three are means of the load the
    // pass delivered, so the curved sweep's warp cancels between numerator and
    // denominator instead of appearing twice.
    //
    // Height and mass are non-negative, so a total below zero can only be rounding.
    // The *latent* is not: these are colour-space channels (Oklab a/b are signed —
    // green carries a negative a), so its prefix is not monotonic and clamping the
    // difference would shear the chroma off whatever the tool is carrying.
    let inv_pre = 1.0 / dpre;
    let tool_h = max(d_load.x, 0.0) * inv_pre;
    let tool_m = max(d_load.y, 0.0) * inv_pre;
    let latm = d_latm.xyz;
    let tool_op = tool_m / max(tool_h, PAINT_EPS);
    let tool_lat = latm * inv_pre / max(tool_m, PAINT_EPS);

    // The canvas as this segment found it — or, when the brush bleeds, that state
    // relaxed towards its neighbours first (see `BLEED_REACH` above). Ahead of the
    // lift rather than behind it: the flux's stability bound is a fraction of the
    // *entering* height, and paint the lift has already taken must not also be shed
    // sideways. The tool's own half sampled the un-bled region, so the two sides of
    // the vertical transfer disagree by O(k_bleed·k_lift·e²) — second order, the
    // same class as the mean-field freeze the loop already carries (`exchange_at`).
    var h0 = ua.x;
    var op0 = uc.a;
    var lat0 = uc.rgb / max(uc.a, PAINT_EPS);
    var bled = false;
    let kb = bleed_rate(); // the bleed rate; nonzero only on a dedicated bleed slot
    if kb > 0.0 {
        let w_t = 1.0 - exp(-kb * e);
        // Entering quantities in the mass-weighted form every flux moves (§6.1):
        // height, optical mass, latent·mass — the same formula both threads of a
        // pair evaluate, which is what makes the flux antisymmetric.
        let m0 = uc.a * ua.x;
        let lm0 = uc.rgb * ua.x;
        var dh = 0.0;
        var dm = 0.0;
        var dlm = vec3<f32>(0.0, 0.0, 0.0);
        // The tap distances, per BLEED_REACH above. Every thread derives the same
        // three integers from the same uniform, so a pair at any scale still
        // computes one number — the antisymmetry the conservation rests on.
        let reach = max(i32(round(BLEED_REACH * radius())), 1);
        var dists = array<i32, 3>(1, max(reach / 4, 1), reach);
        for (var k = 0; k < 3; k = k + 1) {
            for (var i = 0; i < 4; i = i + 1) {
                // Clamped into the snapshot rect. A clamped tap lands on the rect's
                // border, which the margins put provably outside the sweep, so its
                // exposure — and with it the flux — is exactly zero: a no-flux
                // wall, like the wick's rim, at every scale.
                let tn = clamp(
                    t + BLEED_OFFS[i] * dists[k],
                    vec2<i32>(0, 0),
                    vec2<i32>(udim) - vec2<i32>(1, 1),
                );
                if all(tn == t) {
                    continue; // a tiny tip's collapsed scale, or the wall itself
                }
                let rtn = under_origin() + tn;
                // The neighbour's own exposure, from the same uniform its own thread
                // reads it — same floats, same verdict, so the pair agrees on its flux.
                let swn = segment_frame(rtn);
                var w_n = 0.0;
                if !outside_sweep(swn) {
                    let dpn = max(
                        prefix_at(swn.local.x, swn.local.y, orientation())
                            - prefix_at(swn.local.x - swn.lr, swn.local.y, orientation()),
                        0.0,
                    );
                    w_n = 1.0 - exp(-kb * dpn * swn.warp);
                }
                let seln = load1(sel_mask, rtn).r;
                let c = BLEED_SHARES[k] * min(w_t, w_n) * seln;
                if c > 0.0 {
                    let ucn = textureLoad(under_color, tn, 0);
                    let uan = textureLoad(under_aux, tn, 0);
                    dh += c * (uan.x - ua.x);
                    dm += c * (ucn.a * uan.x - m0);
                    dlm += c * (ucn.rgb * uan.x - lm0);
                }
            }
        }
        bled = dh != 0.0 || dm != 0.0 || any(dlm != vec3<f32>(0.0, 0.0, 0.0));
        h0 = ua.x + dh;
        let m1 = m0 + dm;
        op0 = m1 / max(h0, PAINT_EPS);
        lat0 = (lm0 + dlm) / max(m1, PAINT_EPS);
    }

    // The lift's removal: height leaves; the remaining paint's latent and per-unit
    // opacity are unchanged — the source fades because its *thickness* drops,
    // never because its alpha is scaled (§6.1).
    var h = h0 * ex.keep;
    var m = op0 * h;
    let canvas_lat = lat0;

    // The brush's own colour at this texel: the stroke colour offset by the
    // colour-dynamics field, sampled at the same stroke-local (lateral, arc)
    // domain as the fast path — both in canvas px (the travel frame's y is in
    // radius units, so it scales by the radius).
    // How far along the stroke this texel sits, and what the brush has left to lay
    // there. The same value whichever segment is drawing it — segment `i` starts at
    // `dᵢ` and sees this texel at `(arc − dᵢ)/radius` — so both rates below come out
    // independent of how the path was cut (`stroke_drain`). It is also the domain the
    // colour-dynamics lookup already ran on, so there is one arc, not two.
    let arc = arc_at_start() + xl * radius();
    let drain = stroke_drain(arc);
    let own_op = brush_op() * drain;

    var own_lat = brush_lat();
    if any(noise_amp() != vec3<f32>(0.0, 0.0, 0.0)) {
        let lateral = yl * radius();
        own_lat += color_jitter(
            dyn_noise_tex, dyn_noise_samp, vec2<f32>(lateral, arc),
            noise_freq(), noise_amp(), noise_off(),
        );
    }

    // The parcel this segment lays: the tool's deposit + the brush's own `add`
    // paint — the latter linear in exposure, i.e. exactly the plain deposit's
    // path-integral height.
    let dep_h = tool_h * ex.dep;
    // No tooth factor here, and none on the deposit above: both already carry it,
    // because `ex` was solved at the exposure the ground let through. Multiplying
    // again would charge the weave twice — and would take the deposit out of the
    // complement the tool's own dispatch computed, which is what would destroy paint.
    let add_h = add_rate() * drain * ex.add_w;
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
    if ex.keep == 1.0 && par_h == 0.0 && par_m == 0.0 && !bled {
        return;
    }
    var par_lat = own_lat;
    if par_m > PAINT_EPS {
        par_lat = (tool_lat * (tool_op * dep_h) + own_lat * (own_op * add_h)) / par_m;
    }

    let lat = blend_latent(canvas_lat, m, par_lat, par_m);
    h += par_h;
    m += par_m;
    let op = m / max(h, PAINT_EPS);

    // Lerp the whole exchange back towards the snapshot by the selection coverage: at
    // the mask's feathered edge the texel receives that fraction of everything this
    // segment did — the lift as well as the deposit — which is exactly the fraction
    // `exchange` scaled its own side of the transfer by.
    textureStore(region_color_w, rt, mix(uc, vec4<f32>(lat * op, op), sel));
    textureStore(region_aux_w, rt, mix(ua, vec4<f32>(h, 0.0, 0.0, 0.0), sel));
}

// ---- 5. Settle: the transfer the pen-up breaks. ----------------------------------
//
// A stroke stops with the tip still **in contact and out of equilibrium**: the tool
// is holding more than the canvas under it (or less, for an eraser), and that excess
// is exactly the transfer the tip would have gone on making. Everywhere else on the
// trail the tip left smoothly — a point sees the whole footprint pass over it, ending
// with the trailing rim where `τ` has fallen back to 0 — but the last footprint never
// gets that, so what was in flight is stranded. The reservoir is per-stroke, so
// stranded means gone: a tip-shaped disc short of the trail behind it, and (for an
// eraser) a tip-shaped smudge of paint that should have been taken.
//
// So the pen-up settles the pair — one dispatch, one thread per canvas texel of the
// final footprint, the same `exchange_at` kernel as every segment. What it needs is an
// exposure, and there is exactly one choice that does not simply move the discontinuity
// somewhere else. **The transfer a break of contact settles is bounded by the pass on
// both sides:**
//
//     e = min(owed, received),   owed = prefix(l),  received = rowtotal − prefix(l)
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
// Both bounds vanish on the footprint's rim (`owed` at the trailing edge, `received`
// at the leading one, and the row total itself laterally), so the settle fades to
// nothing all the way round and adds no edge of its own. They cross at the tip centre,
// where the kernel is already deep in saturation and the kink costs nothing.
//
// The tool is not written back: nothing reads it after this. A range that does *not*
// reach the end of the stroke never runs this dispatch — it hands its reservoir on to
// the range that resumes instead — so the settle happens once, at the pen-up, and a
// live tail computes the same one its commit will.
@compute @workgroup_size(8, 8)
fn settle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let udim = textureDimensions(under_color);
    if gid.x >= udim.x || gid.y >= udim.y {
        return;
    }
    let t = vec2<i32>(gid.xy);
    let rt = under_origin() + t;
    let rdim = vec2<i32>(textureDimensions(region_color_w));
    if rt.x < 0 || rt.y < 0 || rt.x >= rdim.x || rt.y >= rdim.y {
        return;
    }
    // The slot carries zero travel and zero curvature, so the "sweep" is the tip's own
    // footprint and `outside_sweep` reduces to the brush-local unit square — the same
    // rejection `snapshot` used to decide which texels it bothered to copy.
    let sw = segment_frame(rt);
    if outside_sweep(sw) {
        return;
    }
    let xl = sw.local.x;
    let yl = sw.local.y;

    let total = prefix_at(1.0, yl, orientation());
    let owed = prefix_at(xl, yl, orientation());
    let e = min(owed, total - owed);
    if e <= 0.0 {
        return; // the footprint's rim: nothing in flight, nothing to settle
    }
    let sel = load1(sel_mask, rt).r;
    if sel <= 0.0 {
        return;
    }
    // Gated like every other deposit (§6.4): a tip lifted off a rough ground was
    // bridging the faces that fell away from its travel, so what it would have
    // delivered there stays on it. Nothing
    // is destroyed — the tool is discarded at pen-up carrying whatever it still holds,
    // which is what a knife lifted off a canvas does.
    let ex = exchange_at(e * surface_tooth(rt, sw.turn));

    // The parcel is **what the rest of the pass would have delivered**, not the cell
    // overhead. At the end of a long smear the reservoir is radially skewed: the
    // interior cells trade fast, so they sit near equilibrium with the thin mid-pass
    // canvas under them — nearly empty — while the trailing shoulder, whose τ is a
    // hundredth of the interior's, still carries paint lifted hundreds of px back and
    // pays it out slowly. That slow payout is precisely what the visible trail is made
    // of, so pairing each canvas texel with the (empty) cell directly above it deposits
    // almost nothing where the continued pass would have slid the loaded trailing
    // cells over every interior point: the whole footprint printed as a tip-shaped
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
    // its payout the whole way across the footprint. The second is the parcel's
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
    // the middle of the footprint. A ratio of two smooth readings of one prefix
    // moves nothing discontinuously, and it can only shrink the parcel, so the
    // no-minting bound survives it.
    //
    // A per-texel walk of the row, not a prefix — the en-route factor couples the
    // cell to the point it serves, so no single baked integral carries it. The
    // settle runs once per stroke, which is what makes O(BAKE_RES) taps per texel
    // the right trade here where the per-segment deposit could never afford it.
    let kd = deposit_rate();
    let kl = lift_rate();
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
    var prev_lm = vec3<f32>(0.0, 0.0, 0.0);
    var par_h = 0.0;
    var par_m = 0.0;
    var par_lm = vec3<f32>(0.0, 0.0, 0.0);
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
        let cur_lm = mix(
            textureLoad(bake_latm, vec2<i32>(i, yi0), 0),
            textureLoad(bake_latm, vec2<i32>(i, yi1), 0),
            fy,
        ).xyz;
        // Height and mass prefixes are monotone, so a negative difference is
        // rounding; the latent's is not (Oklab a/b are signed) and stays unclamped.
        let d_h = max(cur.x - prev.x, 0.0);
        let d_m = max(cur.y - prev.y, 0.0);
        let d_e = max(cur.z - prev.z, 0.0);
        let d_lm = cur_lm - prev_lm;
        let pr = cur.z;
        prev = cur;
        prev_lm = cur_lm;
        if d_e <= 0.0 {
            continue; // outside the footprint: nothing carried, nothing served
        }
        let tau = d_e / dx;
        let xc = xr - 0.5 * dx;
        let en_route = exp(-kd * tau * max(xl - xc, 0.0));
        let survive = exp(-kl * max(pr - 0.5 * d_e, 0.0));
        let c = kd * en_route * survive * cover;
        par_h += d_h * c;
        par_m += d_m * c;
        par_lm += d_lm * c;
    }
    // The `min(owed, received)` bound, as the smooth share it lands as (see above).
    let share = ex.dep / max(1.0 - exp(-kd * owed), 1e-6);
    par_h *= share;
    par_m *= share;
    par_lm *= share;

    // The deposit's rewrite guard, for the same reason (see there): a settle whose
    // rates are all zero settles nothing, and must not re-round the texel.
    if ex.keep == 1.0 && par_h == 0.0 && par_m == 0.0 {
        return;
    }

    let uc = textureLoad(under_color, t, 0);
    let ua = textureLoad(under_aux, t, 0);

    // The same parcel stacking as the deposit, with the delivery integral as the
    // parcel: no `add` source, because a settle is a break of contact and not a
    // stretch of travel, and the source is a rate per unit of travel. Colour comes
    // back out of the optical-mass weight it was integrated under, so the parcel
    // carries the mixture of the cells that would have served this point, each in
    // proportion to what it would have delivered (§6.1).
    var h = ua.x * ex.keep;
    var m = uc.a * h;
    let canvas_lat = uc.rgb / max(uc.a, PAINT_EPS);
    var par_lat = canvas_lat;
    if par_m > PAINT_EPS {
        par_lat = par_lm / par_m;
    }
    let lat = blend_latent(canvas_lat, m, par_lat, par_m);
    h += par_h;
    m += par_m;
    let op = m / max(h, PAINT_EPS);

    textureStore(region_color_w, rt, mix(uc, vec4<f32>(lat * op, op), sel));
    textureStore(region_aux_w, rt, mix(ua, vec4<f32>(h, 0.0, 0.0, 0.0), sel));
}
