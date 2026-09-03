// Shared swept-segment scaffolding for the stamp pass (§6.2), imported
// by the per-color-space stamp shaders. Everything here is color-space-agnostic:
// the tile transform + prefix-τ bindings, the oriented-quad vertex, and the
// swept optical-depth lookup. Each variant differs only in how the fragment
// turns that depth into channel deposits (see stamp.wesl; the Mixbox space
// reuses it, since its deposit is the same premultiplied-"over").

import package::lib::noise::{color_jitter, jitter};
import package::lib::paint_common::{substrate_texel_at, rise_ahead, tooth_gate, ceiling_at};
import package::lib::sample::{bilerp_taps, bilerp_mix1};

struct TileXform {
    // params.xy = the target's top-left in canvas px — a tile *texture*'s (interior
    // origin − apron), or the stamp loop's region's; params.z unused (the NDC
    // scale is `ndc` below, per axis); params.w = half
    // the deposit's own footprint in canvas px — 0.5 on the 1× path, 0.5/s when
    // the sweep rasterizes s subsamples per px into a supersampled scratch
    // (§6.2). Only `swept_depth`'s lateral window reads it: every other
    // per-fragment quantity is a function of the interpolated canvas position,
    // which the finer raster grid supplies on its own.
    params: vec4<f32>,
    // The brush's color in the color space's channels (.xyz); .w unused. Constant
    // across the stroke — color dynamics (below) jitter the channels per fragment.
    // There is no per-unit opacity here: the paint a brush lays is per-unit opaque
    // (§6.2), faded only by `drain` per fragment (see `stroke_drain`), and "how
    // much shows" is the effect's own opacity, applied where the whole parcel
    // lands (`integrate.wesl`).
    color: vec4<f32>,
    // The brush color's **residual** (.xyz) — what the space's three channels cannot
    // express (§6.7). Zero for every space but Mixbox, whose pigment
    // polynomial reaches neither black nor the saturated corners on its own; .w unused.
    //
    // Declared unconditionally, where the target that carries it is `@if(resid)`: a
    // uniform is mirrored into Rust from the *unlinked* source (§6.10), which has no
    // feature set to evaluate, so a member that came and went with the variant would
    // give one Rust struct two layouts. Sixteen bytes of zeroes once per stroke is the
    // price; the per-texel channel, which is the one that would matter, Oklab still
    // does not allocate.
    resid: vec4<f32>,
    // The `drain` falloff per canvas px in .x (`BrushParams::drain_px` — the brush's
    // own knob is quoted per *radius*, and the host divides), the canvas →
    // substrate-tile uv scale (1/SUBSTRATE_TILE_PX, zero when the substrate has no relief
    // to gate with) in .y, and the **width of the tooth's contact transition** in the
    // rise's own units in .z (`ToothParams::softness`, §6.4); .w unused. All
    // three stroke constants in the only sense that matters — each is a function of
    // something the fragment already knows, so it is evaluated there (`stroke_drain`,
    // `substrate_tooth`). The tooth's *depth* is not among them, and neither is `add`:
    // both are pen-driven (§6.2) and ride the segment instance, like the radius. What
    // the tip is made of does not change under the hand; how hard it is pressed does.
    paint: vec4<f32>,
    // Color dynamics (§6.2): noise frequency per lookup axis
    // (across the stroke, along it) in .xy with 1/NOISE_TILE_PX pre-folded into
    // .z; .w unused.
    noise_freq: vec4<f32>,
    // Per color-channel noise amplitude .xyz (all 0 = jitter off); .w unused.
    noise_amp: vec4<f32>,
    // The deposit jitter (§6.2): the gate's half-range (0 = off, and then the
    // deposit is bit-identical) and the stroke's own seed for it — a draw off
    // the same record seed the noise tile (`noise_tex`) is baked from.
    jitter_eps: f32,
    jitter_seed: u32,
    // Canvas px → NDC per axis: `2 / target size`. `(2/TILE_TEX, 2/TILE_TEX)` on a
    // tile; the stamp loop's ceiling pass draws the same segments into its
    // region, which is not square (§6.2).
    ndc: vec2<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;
// Prefix-τ volume (§6.2, §6.6): x = travel, y = lateral, array layer = shape
// orientation (one slice per relative angle, sampled with wrapping). Rg32Float:
// `r` is the running travel integral of optical depth, `g` the lateral prefix of
// `r` — what lets `prefix_span_box` read the deposit's exact box average over the
// pixel's own footprint as one more difference (§6.2). Round tips are
// rotation-invariant and have a single slice; image stamps carry up to
// MAX_ORIENTATION_LAYERS. A 2D array (not a 3D texture) so the mask keeps its full size.
@group(1) @binding(0) var prefix_tex: texture_2d_array<f32>;
// The color-dynamics noise field (§6.2): a tileable 2-D tile
// (three signed channels) sampled at (lateral, arc) with a repeat sampler.
// A 1×1 zero tile is bound when the brush's jitter is off.
@group(2) @binding(0) var noise_tex: texture_2d<f32>;
@group(2) @binding(1) var noise_samp: sampler;
// The canvas substrate's tileable height map (§6.4) — height in `r` (the media pass's),
// the rise one reach ahead in `gb` — whose rise along the travel gates the deposit.
// Read in *canvas* space, so the grain belongs to the canvas and successive strokes
// in one direction register with each other; that is also what keeps the gate a pure
// function of canvas position and the tile aprons bit-consistent. A `Flat` canvas
// binds its 1×1 texel and arrives here with the uv scale zeroed, so the gate is
// exactly 1.
//
// No sampler: the tap is nearest (`substrate_texel_at`), for the reasons given there.
@group(2) @binding(2) var substrate_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // The fragment's canvas position. The canvas→NDC map is affine, so this
    // interpolates exactly, and every brush-local quantity is recovered from it in
    // the fragment (`sweep_at`) rather than carried down from the vertices — which is
    // what lets one segment be a curve instead of a straight quad.
    @location(0) canvas: vec2<f32>,
    // The instance, forwarded verbatim. Flat: these are per-segment constants, and
    // interpolating them across a curved sweep would be meaningless.
    @location(1) @interpolate(flat) start: vec2<f32>,
    @location(2) @interpolate(flat) dir: vec2<f32>,
    // The **reference** radius, the arc length, and the radius ramp — how much the tip
    // grows across the segment as a fraction of that radius (§6.2, `Segment::ramp`).
    // Zero on a tip that does not change, which is the branch every untapered stroke
    // at constant pressure takes.
    @location(3) @interpolate(flat) geom: vec3<f32>,
    // Shape orientation in turns ∈ [0,1) (the relative angle between the shape's
    // native axis and the travel direction, picking the prefix-τ slice, §6.6); the
    // arc length of the segment's start from the stroke start (canvas px); the
    // segment's signed curvature (1/canvas px); and the `add` source rate — height
    // per unit swept optical depth, as the pen asked for it here (§6.2).
    @location(4) @interpolate(flat) extra: vec4<f32>,
    // How much give this segment's tip has against the substrate (§6.4). Per segment,
    // not per stroke, because the pen can drive it — bearing down gives the tip the
    // give to press past the falls it was bridging, so the grain fills in: the charcoal
    // behaviour. 1 is all of it, and the gate short-circuits. The band the contact is
    // soft over is the *stroke's* (`xf.paint.z`), for the reason given there.
    @location(5) @interpolate(flat) tooth_give: f32,
    // **The stretch of the tip's travel this point is swept by**, in radius units:
    // where the point sits in the tip frame when the segment ends .x, and where when it
    // starts .y — the tip advancing carries the point backwards through its own
    // extent, so `.y` is the larger. The deposit is one prefix-τ difference across
    // it (`swept_depth`), and `sweep_of` reads the pair off as `Sweep::span`.
    //
    // Carried from the vertices rather than recovered per fragment, and that is the
    // whole point rather than a saving. A range recovered from the fragment's own
    // position can only ever be *one segment's*, because it is read out of that
    // segment's rigid frame; a range that arrives interpolated can be any function the
    // vertices agree on — including the merged `[min start, max end]` of a whole run of
    // segments, which is what lets a stroke be drawn as one strip instead of a quad per
    // cut (§6.2). Nothing merges yet; the mechanism does.
    @location(6) range: vec2<f32>,
    // The rest of the point's place in that frame: its lateral offset in radius units
    // .x, the measure conversion a curved sweep needs .y (`Sweep::warp`), and how far
    // the tip's heading has turned to reach it .z (`Sweep::turn`).
    @location(7) frame: vec3<f32>,
    // The tip's stretch along its facing axis, forwarded verbatim (`stretch_look`).
    // Flat: a per-segment constant, like the instance lanes above.
    @location(8) @interpolate(flat) stretch: vec3<f32>,
    // The ceiling's modulation factor across this segment (§6.2): the mean of its
    // two ends in `.x` and the difference between them in `.y`, which
    // [`ceiling_of`] reads at the fragment's own travel. Flat, like the radius
    // ramp beside it and for its reason: the pair is the segment's, and what
    // varies across the sweep is recovered per fragment from the frame rather
    // than interpolated — the travel fraction is not affine in canvas position on
    // a curved sweep.
    @location(9) @interpolate(flat) opacity: vec2<f32>,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    // The parcel's residual, premultiplied by the same visible alpha the color is
    // and over-blended by the same rule — a residual is not a separate quantity to
    // combine but the rest of the same color (§6.7).
    @if(resid) @location(2) resid: vec4<f32>,
    // The **ceiling lane** (§6.2, `paint_common::level_sums`): this segment's mass
    // above each of the two ceiling levels, and its moment above each — additive,
    // like the aux, since every one of the four is a sum over segments. What the
    // integrate reads off it is the largest ceiling any paint at the texel asked
    // for, averaged over the texel (`claimed_coverage`). Only the variant a
    // pen-driven ceiling builds carries it; location 3 whether or not there is a
    // residual, so the host's lane index is the shader's.
    @if(ceiling) @location(3) ceiling: vec4<f32>,
}

// A canvas point's place in the frame the sweep is integrated in.
struct Sweep {
    // Brush-local: .x the travel coordinate in units of the segment's *reference*
    // radius (0 at the segment start, growing the way the tip goes), .y the lateral
    // offset in units of the radius **in force where this point is** — which is the
    // reference one only on a segment whose tip does not ramp (`ramp_apply`).
    //
    // The two axes are denominated differently on purpose. `.x` is conjugate to `lr`
    // and to the segment's own arc, so the pair must share one frame and that frame is
    // the reference (`stroke_arc`); `.y` indexes the coverage profile across the
    // stroke, which is what the outline *is*, so it has to follow the tip that is
    // actually there. What the prefix table is read at is neither — it is `span`,
    // where each end carries the tip in force at its own moment.
    local: vec2<f32>,
    // The segment's travel **in this point's own measure**, radius units. On a curved
    // sweep a point on the outside of the turn sees the tip cross more of its own
    // frame than a point on the inside does; this is that amount.
    lr: f32,
    // **The two travel coordinates the prefix difference is taken across**: where the
    // point sits in the tip's frame when the segment starts .x, and when it ends .y
    // (so `.x ≥ .y` — the tip advancing carries the point backwards through its own
    // extent). `swept_depth` is `prefix(.x) − prefix(.y)`, and everything below the
    // frame reads the pair from here rather than rebuilding it from `local.x` and
    // `lr`.
    //
    // A pair and not a base-plus-width, because **the two ends are denominated
    // against different tips** on a ramping segment (`ramp_apply`): the point's offset
    // when the segment starts is worth `r_start` px and when it ends `r_end`, and
    // dividing both by the reference radius is what made a taper's deposit depend on
    // where the flattener cut it (§6.2). Denominated this way the pair *telescopes*:
    // one segment's `.y` and the next one's `.x` are the same expression at the knot
    // they share, so a point's total exposure over a pass is the mask's row total
    // whatever the cut — which is the property the outline already had and the deposit
    // did not.
    //
    // Exactly `(local.x, local.x − lr)` wherever the tip does not ramp.
    span: vec2<f32>,
    // Tip travel per unit of the point's own measure — `|R|/ρ`, and exactly 1 on a
    // straight sweep. Every depth read out of a prefix difference is in the point's
    // measure and has to be converted back by this before it means anything.
    warp: f32,
    // How far the tip's heading has turned from the segment's start by the time it
    // reaches this point, signed the way the arc goes — 0 on a straight sweep. Feed it
    // to `sweep_tangent` for the direction the tip is actually going here, which is
    // what the deposition tooth reads the substrate ahead along (§6.4).
    //
    // The **angle** rather than the tangent, because the rotation is the expensive half
    // and almost nobody wants it: `outside_sweep` rejects on the frame alone, and the
    // bleed flux builds a dozen of these per thread to read its neighbours' exposure.
    // The angle falls out of the arc arithmetic already done, so carrying it is free
    // and the sines are paid for only where a substrate is actually being gated.
    //
    // Local rather than the segment's start tangent, for the reason everything else
    // here is evaluated per fragment: a segment may bend by tens of degrees, and a
    // gate keyed on the start tangent would make the mark depend on where the
    // flattener happened to cut a curve. The local tangent is the same field however
    // finely the curve is chopped, so the tooth stays a property of the path.
    turn: f32,
    // The tip radius **in force where this point is**, in the same px the frame is
    // measured in — the segment's reference radius on a tip that does not ramp, and
    // the ramp read at this point's own travel otherwise (`ramp_apply`).
    //
    // What turns a brush-local lateral offset back into px: `local.y · radius`. The
    // travel axis does *not* convert through this one — `local.x` is in reference
    // units and pairs with the reference radius (`stroke_arc`).
    radius: f32,
}

// ---- the stretched tip (§6.6) ----------------------------------------------------
//
// A brush may draw its extent **out along the axis it faces** — what makes a leaned
// pencil lay a broad mark and an upright one a fine line. Stretching by `s` along a
// canvas axis `û` is the linear map `A = R_û·diag(s, 1)·R_ûᵀ` on the extent, and
// substituting `q = A⁻¹p` into the swept integral turns it into an integral of the
// **unstretched** extent: along `v̂ = normalize(A⁻¹t̂)` rather than the travel `t̂`,
// over a travel `m = |A⁻¹t̂|` times as long, with `1/m` on the result.
//
// Every one of those the existing prefix-τ volume already answers, which is why there
// is no second texture here and no rotation in these functions. The *direction* is a
// slice of the volume's own orientation axis, and the host has already folded the
// difference into the orientation it sends (`segments::Stretch`); what is left is a
// change of coordinates on the way into the lookup, and it is upper triangular, so
// three numbers state it:
//
//   .x  `m`      — the travel scale, and the divisor a depth read in the lookup frame
//                  is worth in the tip's own
//   .y  shear    — how much of the point's lateral offset the lookup's travel
//                  coordinate picks up. Zero unless the facing axis is oblique to the
//                  travel, and it is what makes an obliquely stretched tip's leading
//                  edge lean rather than stay square
//   .z  lateral  — what the point's lateral offset is scaled by. **Below 1 for a real
//                  stretch**, so the mask's `|y| ≤ 1` stands for a wider band of
//                  canvas — which is the stroke getting fatter
//
// `vec3(1, 0, 1)` is the tip as its own mask draws it, and every expression below is
// an identity in floats there rather than merely close to one: a brush that never
// heard of stretch renders bit for bit what it always did.

// Where a point of the tip's reference travel frame lands in the frame the prefix-τ
// volume is read in: the travel coordinate where the segment **starts** .x, where it
// **ends** .y, and the lateral offset .z. A prefix difference across `.xy` at `.z`
// is this point's swept depth, in that frame's measure.
fn stretch_look(sw: Sweep, st: vec3<f32>) -> vec3<f32> {
    // The shear is the *same* at both ends — it is a property of where the point sits
    // across the tip, not of how far the tip has gone — so it is formed once and added
    // to each. Written this way round rather than as `x1 − st.x·lr` because the two
    // ends no longer share a denominator (`Sweep::span`).
    let sh = st.y * sw.local.y;
    return vec3<f32>(st.x * sw.span.x + sh, st.x * sw.span.y + sh, st.z * sw.local.y);
}

// The same map run backwards: from a point of the lookup frame to where it rides in
// the tip's reference travel frame. What the dynamics loop's `exchange` needs, because
// it walks the other way — from a reservoir texel, which lives in the mask's own grid,
// out to the canvas the tip is dragging it over.
//
// The inverse of an upper triangular map is upper triangular, so this is three
// reciprocals and no solve, and it is the identity on `vec3(1, 0, 1)` like everything
// else here.
fn stretch_unlook(look: vec2<f32>, st: vec3<f32>) -> vec2<f32> {
    let y = look.y / st.z;
    return vec2<f32>((look.x - st.y * y) / st.x, y);
}

// What a unit of depth read in that frame is worth in the tip's own — the Jacobian
// `1/m` the substitution leaves behind. It is the whole of why an elongated tip lays
// more paint: over a full pass the prefix saturates at the row total whatever the
// travel was scaled to, so the deposit comes out `s` times what the unstretched tip
// would lay, which is exactly the extra mask the stretch dragged over the point.
fn stretch_gain(st: vec3<f32>) -> f32 {
    return 1.0 / st.x;
}

// The box in the reference travel frame that holds everything the mask can paint, as a
// multiple of the frame radius: travel .x, lateral .y.
//
// A point takes paint only where the map above lands it inside the mask's own
// `|x| ≤ 1, |y| ≤ 1`, so `|y| ≤ 1/lateral` across and `|x| ≤ (1 + |shear|/lateral)/m`
// along. What needs it is the geometry drawn *for* the extent — the sweep strip
// below, and the dynamics loop's rim test — because a stretched tip paints outside the
// strip an unstretched one would have been given, and a fragment never rasterized is a
// stroke cut off along a straight line. `segments::Stretch::hull` is this same
// function on the host, where it bounds the tiles the segment is drawn into.
//
// `vec2(1, 1)` exactly when there is no stretch.
fn stretch_hull(st: vec3<f32>) -> vec2<f32> {
    let across = 1.0 / st.z;
    return vec2<f32>((1.0 + abs(st.y) * across) / st.x, across);
}

// The tip in force a fraction `u` of the way through a ramping segment, as a multiple
// of the segment's reference radius: `1 + ramp·(u − ½)` (§6.2, `Segment::ramp`).
//
// Positive for every `u ∈ [0,1]` without a clamp, because the host floors both ends of
// the ramp at half a px and `|r₁ − r₀| < r₁ + r₀` — so `|ramp| < 2` structurally.
//
// Clamped in `u` rather than extrapolated: past either end of the segment the tip has,
// as far as this segment is concerned, stopped at the end it reached. That is also what
// keeps the overhanging caps of two adjacent segments agreeing about the knot they
// share, instead of each running off along its own slope.
fn ramp_scale(ramp: f32, x: f32, lr: f32) -> f32 {
    // Every segment of an untapered brush at constant pressure takes this branch, and
    // it is an exact identity — those strokes render as they did before ramps existed,
    // to the bit. It also carries the settle slot, whose travel is zero.
    if ramp == 0.0 || lr <= 0.0 {
        return 1.0;
    }
    return 1.0 + ramp * (clamp(x / lr, 0.0, 1.0) - 0.5);
}

// Re-denominate a sweep's lateral axis against the tip actually in force there.
//
// Split out of `sweep_at` rather than folded into it because the two callers need it at
// different moments. The dynamics loop builds its frame per fragment and applies this
// immediately (`dynamics::segment_frame`). The stamp path builds the frame at the
// *vertices* and interpolates it, and this correction is not affine in canvas position
// — it is a ratio of two affine functions — so interpolating its result would be wrong
// where interpolating `sweep_at`'s is exact. There it is applied after the interpolation
// instead (`sweep_of`), on quantities that survive it.
fn ramp_apply(sw: Sweep, ramp: f32) -> Sweep {
    // The identity, taken as a branch rather than as a `/1` and a `*1`. Every segment
    // of an untapered brush at constant pressure comes through here, and the dynamics
    // loop's bleed ladder builds thirty-six of these frames per thread, so the two
    // reciprocals below are worth not paying. Guarded on `lr` as well as on the ramp,
    // for the reason `ramp_scale` is: a slot with no travel has no ends to tell apart,
    // and denominating its two span coordinates differently would open a prefix
    // difference where the tip never moved.
    if ramp == 0.0 || sw.lr <= 0.0 {
        return sw;
    }
    let s = ramp_scale(ramp, sw.local.x, sw.lr);
    var out = sw;
    out.local.y = sw.local.y / s;
    out.radius = sw.radius * s;
    // **And the two ends of the span against the tips that are actually there** — the
    // one at the segment's start against `r_start`, the one at its end against `r_end`
    // (`Sweep::span`). The exposure is the integral of the extent over the offsets
    // the point occupied relative to the tip, and that offset is measured in whatever
    // tip was in force at the moment: `r(0)` when the sweep opens and `r(len)` when it
    // closes. Dividing both by the reference radius instead over-counts the segment's
    // narrow half and under-counts its fat one, cancelling only to first order — so a
    // taper's deposit rippled at the flattener's own cadence, worst exactly where the
    // ramp is largest and no subdivision can shrink it (the taper's point).
    //
    // The two scales are `1 ∓ ramp/2` — `ramp_scale`'s value at either end, written
    // out because these are the segment's ends and not this point's position. Both are
    // positive without a clamp for the reason the tip is: `|ramp| < 2` structurally
    // ([`Segment::ramp`]).
    out.span = vec2<f32>(sw.span.x / (1.0 - 0.5 * ramp), sw.span.y / (1.0 + 0.5 * ramp));
    return out;
}

// The tip's heading `turn` into the segment, from the tangent it left the start with.
//
// A rotation about the centre of curvature turns every point's tangent by the same
// angle, so this is the point's own travel as much as the tip's. Straight sweeps take
// the branch and get `dir` back untouched, which is the overwhelmingly common case and
// bit-exact besides.
fn sweep_tangent(dir: vec2<f32>, turn: f32) -> vec2<f32> {
    if turn == 0.0 {
        return dir;
    }
    let c = cos(turn);
    let s = sin(turn);
    return vec2<f32>(dir.x * c - dir.y * s, dir.x * s + dir.y * c);
}

// Where `p` sits in the travel frame of one swept segment (§6.2).
//
// A straight sweep is the plain rotation into (along, across). A **curved** one is
// the annulus about the centre of curvature *unrolled* into that same straight frame:
// the point keeps its distance from the centreline, and its travel coordinate is arc
// length measured at its own radius — which is the natural thing, because in the tip's
// rotating frame the point really does slide along a line of constant offset at speed
// `ρω` while the tip travels at `Rω`.
//
// That ratio is the whole content of `warp`. The prefix-τ volume integrates a
// extent dragged across a point at unit speed; here the drag is `ρ/|R|` times the
// tip's own travel, so a prefix difference comes out in the point's measure and
// `warp` puts it back into the tip's. Concretely the swept depth is
// `(|R|/ρ)·(prefix(x) − prefix(x − lr))`, and for a hard disc that evaluates to
// `|R|·Δφ_covered` — the tip travel while the point was under it, which is what it
// has to be. It is also why the inside of a turn takes more paint than the outside.
//
// Everything is written to stay well-conditioned as the curvature vanishes (no large
// radius is ever subtracted from another), but the `kappa == 0` branch is still exact:
// the host emits a literal zero for any segment it declines to bend, so a straight
// stroke never touches the arc arithmetic at all.
fn sweep_at(
    p: vec2<f32>,
    start: vec2<f32>,
    dir: vec2<f32>,
    radius: f32,
    len: f32,
    kappa: f32,
) -> Sweep {
    let perp = vec2<f32>(-dir.y, dir.x);
    let w = p - start;
    let along = dot(w, dir);
    let across = dot(w, perp);

    var out: Sweep;
    out.radius = radius;
    if kappa == 0.0 {
        out.local = vec2<f32>(along, across) / radius;
        out.lr = len / radius;
        out.warp = 1.0;
        out.turn = 0.0;
        out.span = vec2<f32>(out.local.x, out.local.x - out.lr);
        return out;
    }
    let ak = abs(kappa);
    // ρ/|R| — the point's own turning radius as a fraction of the centreline's.
    let g = length(kappa * w - perp);
    // The angle swept from the segment's start, positive the way the tip travels.
    // Both arguments are the exact ones scaled by the positive quantity R², and the
    // travel sign is folded into the first, so no branch on the turn direction is
    // needed and nothing large is differenced.
    let delta = atan2(ak * along, 1.0 - kappa * across);
    // The lateral offset, `sign(κ)(|R| − ρ)`, rearranged through `1 − g²` so the two
    // radii are never subtracted from one another.
    let lateral = (2.0 * across - kappa * dot(w, w)) / (1.0 + g);

    out.local = vec2<f32>(g * delta / (ak * radius), lateral / radius);
    out.lr = len * g / radius;
    out.warp = 1.0 / g;
    // `delta` is already the angle swept from the start, positive the way the tip
    // travels, so the turn direction is the only thing left to restore.
    out.turn = sign(kappa) * delta;
    out.span = vec2<f32>(out.local.x, out.local.x - out.lr);
    return out;
}

// How many cross-sections the swept geometry is built from. The sweep is a curve, so
// its extent is an annular *sector*, and this is how finely that sector is
// approximated by a triangle strip. Purely a fill-rate knob — the deposit itself is
// computed per fragment from the exact arc — so it is small, and the strip is
// circumscribed (below) so no value of it can clip the stroke.
const SWEEP_SLICES: u32 = 8u;
// = 2 * (SWEEP_SLICES + 1); the host's draw call must ask for exactly this many.
const SWEEP_VERTS: u32 = 18u;

// How far outside the tip's own reach the deposit can land, canvas px: the pixel
// footprint's box filter (`swept_depth`) reads half a px past `|y| = 1`, and a px of
// margin keeps the strip's two bounds one inequality apart from the filter's. Grows
// the strip's rims below and the host's per-segment reach (`segments::Sweep::reach`,
// through the generated mirror) — the two statements of the same bound, and a
// fragment either one under-reports is a stroke clipped along a straight line.
const AA_RIM_PX: f32 = 1.0;

// Build the swept geometry for one segment: a strip of `SWEEP_SLICES` quads covering
// brush-local x ∈ [-1, lr+1], y ∈ [-1, 1] — a radius of margin at each end for the
// round cap, plus [`AA_RIM_PX`] on every side for the pixel footprint's half-px reach
// — bent around the segment's arc and mapped into the tile's apron'd target.
//
// Nothing here has to be tight, only *conservative*: a fragment outside the extent
// differences two prefix taps to zero and writes nothing. So the outer rim is pushed
// out to circumscribe the true arc and the inner rim pulled in, and both errors cost
// a few fragments rather than a clipped stroke. What it must not do is cover a texel
// twice, which is why the host caps `radius · |curvature|` (`MAX_TIP_TURN`): the
// sector stays a simple polygon only while the tip's inner rim clears the centre.
fn sweep_vertex(
    vi: u32,
    start: vec2<f32>,
    dir: vec2<f32>,   // unit tangent at the segment start
    geom: vec3<f32>,  // reference radius, arc length, radius ramp
    extra: vec4<f32>, // orientation (turns ∈ [0,1)), arc length at start, curvature, `add`
    tooth_give: f32, // how far this tip settles into the substrate (§6.4)
    stretch: vec3<f32>, // the tip drawn out along its facing axis (§6.6)
    opacity: vec2<f32>, // the ceiling's factor across the segment (§6.2)
) -> VsOut {
    let radius = geom.x;
    let len = geom.y;
    let kappa = extra.z;
    let perp = vec2<f32>(-dir.y, dir.x);

    // The strip is built at the segment's **widest** tip, not its reference one: a
    // ramping segment's extent reaches `radius·(1 + |ramp|/2)` at the fat end, and
    // the strip has only to be conservative — a fragment outside the true extent
    // differences its prefix taps to zero and writes nothing, while one the strip
    // misses is a clipped stroke. Exactly `radius` when the tip does not ramp, so no
    // existing stroke pays a fragment for this.
    let hull = radius * (1.0 + 0.5 * abs(geom.z));

    // ...and at its **stretched** reach on each axis (`stretch_hull`). A tip drawn out
    // along its facing axis paints outside the strip its radius alone would have been
    // given, and a fragment that is never rasterized is a stroke cut off along a
    // straight line — the failure every under-reported reach lands on
    // (`segments::Sweep::reach`, §6.6). Plus the filter's rim ([`AA_RIM_PX`]): the
    // pixel footprint reads half a px past the extent, so the strip carries a px of
    // margin on both axes for it.
    let hf = stretch_hull(stretch);
    let ha = hull * hf.x + AA_RIM_PX; // the travel margin at each end
    let hl = hull * hf.y + AA_RIM_PX; // the lateral rim

    let u = f32(vi >> 1u) / f32(SWEEP_SLICES); // 0..1 along the travel
    let side = select(-1.0, 1.0, (vi & 1u) != 0u); // the two lateral rims

    var canvas: vec2<f32>;
    if kappa == 0.0 {
        // The straight case is one quad's worth of geometry, cut into slices: the
        // travel runs from one margin's reach to the other's, in canvas px — a
        // radius of the *widest* tip plus the filter's rim at each end, for the
        // same reason the lateral rim carries both.
        let cx = -ha + u * (len + 2.0 * ha);
        canvas = start + dir * cx + perp * (side * hl);
    } else {
        let ak = abs(kappa);
        let sk = sign(kappa);
        let abs_r = 1.0 / ak;
        // The tip's own turn, plus the angular margin that carries the round cap past
        // each end. The margin is measured at the *inner* rim, where a radius of
        // travel subtends the largest angle, so it covers both rims. Built on the
        // widest tip, like the straight branch, and `max`'d away from a degenerate
        // denominator should a hull ever reach the centre of curvature — `MAX_TIP_TURN`
        // caps `radius·|curvature|` at a tenth, so this has an order of magnitude in
        // hand even at the largest ramp a segment can carry.
        // Measured with the stretched reaches for the same reason the straight branch
        // is: the travel margin has to carry the cap past each end, and the angle it
        // subtends is largest at the inner rim, which the lateral reach sets.
        let margin = ha / max(abs_r - hl, 1e-6);
        let span = len * ak + 2.0 * margin;
        let delta = -margin + u * span;
        // Circumscribe: a chord of the outer rim would cut inside the true arc and
        // clip the stroke, so push it out by the half-slice it spans. The inner rim
        // gets the same factor the other way, which only over-covers.
        let half = 0.5 * span / f32(SWEEP_SLICES);
        let c = cos(half);
        var rho = abs_r - sk * side * hl;
        rho = select(rho * c, rho / c, sk * side < 0.0);
        // The same frame `sweep_at` unrolls, run forwards: `ρ sin δ` along the start
        // tangent and `sign(κ)(|R| − ρ cos δ)` across it.
        canvas = start
            + dir * (rho * sin(delta))
            + perp * (sk * (abs_r - rho * cos(delta)));
    }

    let ndc = (canvas - xf.params.xy) * vec2<f32>(xf.ndc.x, -xf.ndc.y) + vec2<f32>(-1.0, 1.0);

    // The exact frame *at this vertex*, from the same arc arithmetic the fragment used
    // to run. On a straight sweep every quantity below is affine in canvas position, so
    // interpolating it across the strip reproduces it exactly and this is a pure move
    // of the work off the fragment. On a curved one it is a chord across `atan2`, which
    // is what `SWEEP_SLICES` now buys in *accuracy* as well as in coverage — the strip
    // is circumscribed either way, so no slice count can clip the stroke.
    let sw = sweep_at(canvas, start, dir, radius, len, kappa);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.canvas = canvas;
    out.start = start;
    out.dir = dir;
    out.geom = geom;
    out.extra = extra;
    out.tooth_give = tooth_give;
    out.range = vec2<f32>(sw.local.x - sw.lr, sw.local.x);
    out.frame = vec3<f32>(sw.local.y, sw.warp, sw.turn);
    out.stretch = stretch;
    out.opacity = opacity;
    return out;
}


// This fragment's place in its segment's travel frame, as the vertices agreed it.
//
// The travel coordinate is the range's second lane — where the point sits when the
// segment *starts*, which is the far end of the stretch it is swept through — and the
// travel it was swept by is the range's own width. Stated that way round because it is
// the range that is fundamental: `swept_depth` wants two prefix arguments, and
// everything else here is a reading of the same pair.
// The ramp is applied *here*, after the interpolation, for the reason `ramp_apply`
// gives: the varyings above are affine in canvas position on a straight sweep and the
// correction is not, so the strip carries the reference frame and the fragment
// re-denominates it. A segment whose tip does not ramp takes `ramp_apply`'s identity
// branch and reads exactly the interpolated values, bit for bit.
fn sweep_of(in: VsOut) -> Sweep {
    var out: Sweep;
    out.local = vec2<f32>(in.range.y, in.frame.x);
    out.lr = in.range.y - in.range.x;
    // The range **is** the reference span, and taken straight rather than rebuilt from
    // `local.x` and `lr` — the vertices agreed on both ends, so differencing one out of
    // the other and back is a rounding the interpolation did not ask for.
    out.span = in.range.yx;
    out.warp = in.frame.y;
    out.turn = in.frame.z;
    out.radius = in.geom.x;
    return ramp_apply(out, in.geom.z);
}

// The brush's color channels (per the color space) at this fragment: the
// stroke-constant color, offset by the color-dynamics noise field
// (§6.2). The lookup domain is **stroke-local** — (lateral offset from the
// centreline, arc length along the stroke), both in canvas px — so the pattern
// belongs to the stroke rather than to the canvas it lands on: the same gesture
// carries the same color variation wherever it is drawn. Both coordinates are
// still functions of the fragment's canvas position and the segment, so the
// deposit remains a pure function of the two and tile aprons stay bit-consistent
// (§6.4).
fn brush_color(in: VsOut, sw: Sweep) -> vec3<f32> {
    if all(xf.noise_amp.xyz == vec3<f32>(0.0, 0.0, 0.0)) {
        return xf.color.xyz;
    }
    // Brush-local y is in units of the radius **in force here**, so it converts back
    // through that one and not the segment's reference — they differ on a ramping tip,
    // and the product is the fragment's true lateral offset either way. The noise
    // domain is px on both axes, so the pattern keeps one scale whatever size the tip
    // is.
    let lateral = sw.local.y * sw.radius;
    let jitter = color_jitter(
        noise_tex, noise_samp, vec2<f32>(lateral, stroke_arc(in, sw)),
        xf.noise_freq.xyz, xf.noise_amp.xyz,
    );
    return xf.color.xyz + jitter;
}

// How far along the stroke this fragment is, in canvas px: the segment's own start
// distance plus the fragment's offset along the travel (brush-local x is in radius
// units). A function of the fragment's canvas position and the segment alone, so it
// keeps tile aprons bit-consistent (§6.4) like everything else here.
//
// The same value comes out whichever segment is being drawn — segment `i` starts at
// `dᵢ` and sees this fragment at `(arc − dᵢ)/radius`, which adds back to `arc` — and
// that is the whole point of measuring the paint rates against it.
//
// Through the segment's **reference** radius, not the tip in force (`Sweep::radius`).
// The travel axis is conjugate to it — `local.x` was denominated in the reference when
// the frame was built, and `lr` with it — so converting back through anything else
// would not return the px the fragment actually sits at.
fn stroke_arc(in: VsOut, sw: Sweep) -> f32 {
    return in.extra.y + sw.local.x * in.geom.x;
}

// The brush's remaining load at this fragment: `drain` fades the paint it lays,
// linearly with distance travelled, until the stroke runs dry (§6.2).
//
// **Evaluated per fragment, and that is load-bearing rather than tidy.** As a
// per-segment constant it made the paint laid depend on where the segment boundaries
// fell — the one property the swept formulation exists to avoid, and the flattener had
// to buy accuracy back at `0.02 / drain` px per segment, which for a fast-draining
// brush dominated every other budget. Per fragment the stroke lays `a(arc) · Στ`,
// where `Στ` is already independent of the cut, so the whole dependence is gone and
// the cap with it. It is also exact rather than merely better: the falloff is linear
// in arc length, and a linear function sampled at the fragment's own position needs no
// correction at all.
//
// Bounded at a full load as well as at an empty one: the fragments behind the stroke's
// *start* sit at a negative arc — the tip's trailing half covers them at pen-down — and
// an unbounded falloff hands them more than a full brush, which is not a load at all
// and puts `op > 1` on the paint it multiplies. Both paths bound it identically, so a
// drained brush lays the same start cap whichever one draws it
// (`dynamics.wesl::stroke_drain` states what the loop's storage does with the excess).
fn stroke_drain(arc: f32) -> f32 {
    return clamp(1.0 - xf.paint.x * arc, 0.0, 1.0);
}

// The share of its paint this fragment's texel receives from the substrate under it
// (§6.4) — the deposition tooth, `paint_common.wesl::tooth_gate` fed the rise this
// fragment's tip meets along its travel there (`rise_ahead`).
//
// Sampled at the fragment's own canvas position, and read ahead along the tip's travel
// *there* (`sw.tangent`), which the fragment recovers from the segment's frame at that
// same position. So it is still a pure function of canvas position within a pass: a
// tile's apron is rewritten identically to its neighbour's interior with no copy pass,
// and the gate factors out of the sum over one segment's overlapping quads exactly as
// before, leaving the deposit as independent of the flattening as it was.
//
// `xf.paint.y` is zero on a canvas with no relief (`Surface::relief`), so the guard
// returns exactly 1.0 before the map is read: a `Flat` document deposits bit-for-bit
// what it always did whatever the brush's tooth says.
fn substrate_tooth(in: VsOut, sw: Sweep) -> f32 {
    let uv_scale = xf.paint.y;
    if uv_scale <= 0.0 || in.tooth_give >= 1.0 {
        return 1.0;
    }
    let dims = textureDimensions(substrate_tex);
    let g = textureLoad(substrate_tex, substrate_texel_at(in.canvas * uv_scale, dims), 0);
    return tooth_gate(rise_ahead(g, sweep_tangent(in.dir, sw.turn)), in.tooth_give, xf.paint.z);
}

// The ceiling's factor at this fragment (§6.2): the segment's two ends read at the
// fragment's own travel (`paint_common::ceiling_at`). Exactly `in.opacity.x` on a
// brush whose ceiling the pen does not drive, which is the branch that keeps those
// strokes bit-identical.
fn ceiling_of(in: VsOut, sw: Sweep) -> f32 {
    return ceiling_at(in.opacity.x, in.opacity.y, sw.local.x, sw.lr);
}

// The deposit jitter's gate at this fragment (§6.2, `lib::noise::jitter`):
// keyed on the fragment's own canvas texel, a nearest read of an integer grid
// exactly as the substrate tap above is, and a pure function of canvas position
// the same way — so tile aprons stay bit-consistent, and the dynamics loop's
// `jitter_at`, which reaches the same texel through its region's origin,
// reads the identical gate (the loop must: its `add` axis is this path's deposit).
fn jitter_of(in: VsOut) -> f32 {
    return jitter(vec2<i32>(floor(in.canvas)), xf.jitter_seed, xf.jitter_eps);
}

// Bilinear lookup of one orientation slice `zi` at brush-local (xl, yl) ∈ [-1,1]: 0
// before the leading edge, clamp-to-row-total past the trailing edge.
//
// Hand-rolled rather than taken from a ClampToEdge sampler with linear filtering,
// for the three reasons `lib::sample::bilerp_taps` documents — which is where the
// tap arithmetic now lives, shared with the one other lookup that needs it
// (`dynamics::bake_at`). Two of those reasons are worth pinning to this caller in
// particular:
//
//   · prefix_tex is fp32 (Rg32Float) because each fragment takes the *difference* of
//     two prefix sums (see `swept_depth`), which is sensitive to catastrophic
//     cancellation, and a row's total κ can be large for opaque brushes. A
//     filterable 16-bit format would band exactly where the differencing is most
//     sensitive.
//   · The travel (x) axis is asymmetric — 0 before the leading edge,
//     clamp-to-row-total after the trailing one — and ClampToEdge would read the
//     column-0 value across the whole start margin instead of 0. So the
//     `xl <= -1` guard in `prefix_at` would be required whatever the sampler did.
fn prefix_slice(xl: f32, yl: f32, zi: i32) -> f32 {
    let b = bilerp_taps(vec2<f32>(textureDimensions(prefix_tex)), vec2<f32>(xl, yl));
    return bilerp_mix1(
        b,
        textureLoad(prefix_tex, vec2<i32>(b.i0.x, b.i0.y), zi, 0).r,
        textureLoad(prefix_tex, vec2<i32>(b.i1.x, b.i0.y), zi, 0).r,
        textureLoad(prefix_tex, vec2<i32>(b.i0.x, b.i1.y), zi, 0).r,
        textureLoad(prefix_tex, vec2<i32>(b.i1.x, b.i1.y), zi, 0).r,
    );
}

// One slice's reading at brush-local (xl, yl), with the travel axis's leading-edge
// rule: 0 before the tip has reached the point at all.
//
// Split from [`prefix_at`] so that [`prefix_span`] can apply the lateral guard and the
// slice selection once for the *pair* of readings it differences.
fn prefix_edge(xl: f32, yl: f32, zi: i32) -> f32 {
    if xl <= -1.0 {
        return 0.0;
    }
    return prefix_slice(xl, yl, zi);
}

// The `g` channel's reading at brush-local (xl, yl): the **lateral prefix of the
// travel prefix**, `∫₋₁^yl prefix(xl, ·)` in brush units, with `prefix_edge`'s
// leading-edge rule. Baked at the midpoint rule, so a bilinear read at a row's
// centre is the exact integral up to it (`assets::build_prefix_tau`); the lateral
// clamp is the right boundary here rather than a hazard — a tap past either rim
// reads the column's empty start or its total, which is what a window hanging off
// the mask's edge means.
fn prefix2_edge(xl: f32, yl: f32, zi: i32) -> f32 {
    if xl <= -1.0 {
        return 0.0;
    }
    let b = bilerp_taps(vec2<f32>(textureDimensions(prefix_tex)), vec2<f32>(xl, yl));
    return bilerp_mix1(
        b,
        textureLoad(prefix_tex, vec2<i32>(b.i0.x, b.i0.y), zi, 0).g,
        textureLoad(prefix_tex, vec2<i32>(b.i1.x, b.i0.y), zi, 0).g,
        textureLoad(prefix_tex, vec2<i32>(b.i0.x, b.i1.y), zi, 0).g,
        textureLoad(prefix_tex, vec2<i32>(b.i1.x, b.i1.y), zi, 0).g,
    );
}

// One slice's swept depth integrated over the lateral window [y_lo, y_hi]: the
// lateral prefix differenced across the window at each travel end, then across the
// travel pair. Differencing in y first is the conditioned order — each pair is two
// nearby readings of one column, subtracted before anything scales them.
fn prefix2_box(x1: f32, x0: f32, y_hi: f32, y_lo: f32, zi: i32) -> f32 {
    return (prefix2_edge(x1, y_hi, zi) - prefix2_edge(x1, y_lo, zi))
        - (prefix2_edge(x0, y_hi, zi) - prefix2_edge(x0, y_lo, zi));
}

// Trilinear lookup of the prefix-τ volume at brush-local (xl, yl) ∈ [-1,1] and shape
// orientation `w` (turns ∈ [0,1)). The orientation axis wraps (a full turn is periodic),
// so the two bracketing slices are chosen modulo the slice count. The x,y bilinear is
// hand-rolled for the same reasons (fp32 isn't filterable here, the asymmetric travel-edge
// boundary, fp32 precision); 0 before the leading edge, clamp-to-row-total past the
// trailing edge. A single-slice (round) volume collapses to that one slice for every `w`.
fn prefix_at(xl: f32, yl: f32, w: f32) -> f32 {
    return prefix_span(xl, -2.0, yl, w);
}

// **The difference of two prefix readings across a stretch of travel** — the swept
// depth, and what every caller of this volume actually wants: `prefix(x1) − prefix(x0)`
// at one lateral offset on one orientation.
//
// Written as the span rather than as two lookups because everything except the travel
// coordinate is shared between them, and paying for it twice was the shape of the
// hottest lookup the dynamics loop has (`deposit` takes this difference at every
// extent texel, and the bleed ladder takes it thirty-six more times to read its
// neighbours' exposure). The lateral guard, `textureNumLayers`, and the whole
// orientation-slice selection now happen once.
//
// **Differencing inside the slice mix is also the better-conditioned order.**
// `mix(a₁,b₁,t) − mix(a₀,b₀,t)` and `mix(a₁−a₀, b₁−b₀, t)` are the same number
// exactly; in floats the second subtracts two readings that are close to each other
// before they are scaled, where the first forms two large interpolants and differences
// those. The whole reason this volume is fp32 is that this subtraction is where the
// digits go (`prefix_slice`).
//
// `x0 <= -1` is the plain single lookup: nothing is subtracted, because the tip had not
// reached the point when the span opened. `prefix_at` passes `-2` and means exactly that.
fn prefix_span(x1: f32, x0: f32, yl: f32, w: f32) -> f32 {
    // Outside the extent's own square there is no shape to integrate: every mask is
    // swept over brush-local `|x| ≤ 1, |y| ≤ 1` and is zero beyond it — the round tip
    // by construction, a canonical stamp because its content lies inside the disc
    // inscribed in that square (`stark_assetid::coverage`). Stated rather than left
    // to the mask's border row,
    // because `bilerp_taps` **clamps** — so a lateral tap past the edge silently reads
    // the outermost row instead, and that row is not quite zero: the round tip's
    // `κ = (h/π)·acos(r)/r` is still small-but-positive at the last texel's `r ≈ 0.996`.
    //
    // It cost nothing while every fragment the sweep rasterized sat inside the square,
    // which is how it survived. A segment whose tip *ramps* is drawn over a strip built
    // on its widest radius (`sweep_vertex`), so on its narrow half there is now a band
    // of real fragments out past `|y| = 1` — and all of them were differencing the same
    // clamped row into a faint, hard-edged rectangle of paint the width of the quad.
    if abs(yl) >= 1.0 {
        return 0.0;
    }
    let nz = i32(textureNumLayers(prefix_tex));
    // A round tip's volume is one slice, and that slice is the answer at every
    // orientation. Worth a branch rather than letting the general path stand:
    // that path reads the same slice twice — sixteen loads where eight suffice — and
    // its `mix(a, a, fw)` is not even the identity in floats (two roundings), so the
    // collapse is the more exact reading as well as the cheaper one.
    if nz == 1 {
        return prefix_edge(x1, yl, 0) - prefix_edge(x0, yl, 0);
    }
    let fl = fract(w) * f32(nz);
    let l0f = floor(fl);
    let fw = fl - l0f;
    let z0 = (i32(l0f) % nz + nz) % nz;
    let z1 = (z0 + 1) % nz;
    let a = prefix_edge(x1, yl, z0) - prefix_edge(x0, yl, z0);
    let b = prefix_edge(x1, yl, z1) - prefix_edge(x0, yl, z1);
    return mix(a, b, fw);
}

// [`prefix_span`] **box-filtered across the lateral axis**: the swept depth averaged
// over the window `yl ± dy` (brush units) rather than read at `yl` alone — the pixel's
// own footprint, when the caller passes half a canvas px (§6.2). One more prefix
// difference, on the volume's `g` channel, exactly as the travel axis has always been
// integrated; twice the loads of the point read.
//
// The filter is linear in τ, so it commutes with the sum over a stroke's overlapping
// segments — a box average of `Στ` is the sum of box averages — and the deposit stays
// exactly as independent of the flattener's cuts as the point read (§6.2's law). It
// conserves the profile's lateral integral wherever the window sits (adjacent windows
// telescope), which is what keeps a sub-px tip's paint on the canvas instead of
// vanishing between texel centres — height conserved, §6.1's own invariant.
fn prefix_span_box(x1: f32, x0: f32, yl: f32, w: f32, dy: f32) -> f32 {
    // The filtered support: the window must overlap the extent's |y| < 1.
    if abs(yl) - dy >= 1.0 {
        return 0.0;
    }
    let y_hi = yl + dy;
    let y_lo = yl - dy;
    let norm = 0.5 / dy;
    let nz = i32(textureNumLayers(prefix_tex));
    if nz == 1 {
        return prefix2_box(x1, x0, y_hi, y_lo, 0) * norm;
    }
    let fl = fract(w) * f32(nz);
    let l0f = floor(fl);
    let fw = fl - l0f;
    let z0 = (i32(l0f) % nz + nz) % nz;
    let z1 = (z0 + 1) % nz;
    let a = prefix2_box(x1, x0, y_hi, y_lo, z0);
    let b = prefix2_box(x1, x0, y_hi, y_lo, z1);
    return mix(a, b, fw) * norm;
}

// Swept geometric optical depth over the fragment's segment: the brush extent
// integrated as it travels, as a difference of prefix sums on the segment's orientation
// slice. Sums exactly across a stroke's overlapping segment sweeps (under "over" for τ,
// or additively for amounts), reconstructing the continuous path integral — each
// segment contributes the definite integral over its own stretch of the path, which is
// as true of an arc as it was of a chord.
//
// The difference comes out in the fragment's own measure of travel; `warp` converts it
// back to the tip's (see `sweep_at`), and is 1 on a straight sweep.
fn swept_depth(in: VsOut, sw: Sweep) -> f32 {
    let w = in.extra.x;
    // The one place the tip's stretch is applied on this path (§6.6), and deliberately
    // the last thing done. Everything else the fragment reads out of its frame — the
    // arc it sits at, the color-noise domain, the tip in force, the substrate it is gated
    // by — is a property of where the *tip* went, and the stretch is a property of the
    // extent the tip carries. So the frame stays the tip's own the whole way down
    // and only the lookup into the mask is re-coordinated, which is also what keeps
    // `stroke_arc` measuring canvas px rather than the lookup frame's own units.
    let l = stretch_look(sw, in.stretch);
    // Half the fragment's own footprint in the lookup frame's lateral units
    // (§6.2's pixel footprint): brush-local y is denominated in the tip in force,
    // drawn out by the stretch's lateral scale. `xf.params.w` is half a canvas px
    // on the 1× path and half a *subsample* when the sweep supersamples — each
    // fragment filters over the patch of canvas it will answer for, and the
    // integrate's resolve supplies the rest of the pixel. A function of the
    // fragment's frame and the stroke's own constant, like everything here, so
    // the deposit stays a pure function of canvas position (§6.4).
    let dy = xf.params.w * in.stretch.z / sw.radius;
    var depth: f32;
    // Box-filtered only where the window spans at least a mask texel. Below that the
    // differenced lateral prefix degenerates into a fixed one-texel blur *scaled by
    // the radius* — softer than the px asked for, and growing with the tip — while a
    // tip that large resolves its own profile over many px anyway.
    if dy * f32(textureDimensions(prefix_tex).y) >= 1.0 {
        depth = prefix_span_box(l.x, l.y, l.z, w, dy);
    } else {
        depth = prefix_span(l.x, l.y, l.z, w);
    }
    return max(depth, 0.0) * sw.warp * stretch_gain(in.stretch);
}
