// Shared swept-segment scaffolding for the stamp pass (§6.2), imported
// by the per-color-space stamp shaders. Everything here is color-space-agnostic:
// the tile transform + prefix-τ bindings, the oriented-quad vertex, and the
// swept optical-depth lookup. Each variant differs only in how the fragment
// turns that depth into channel deposits (see stamp_oklab.wesl; the Mixbox space
// reuses it, since its deposit is the same premultiplied-"over").

import package::noise::{color_jitter};

struct TileXform {
    // params.xy = tile *texture* top-left (interior origin − apron) in canvas px;
    // params.z = 2 / TILE_TEX (full interior+apron maps to NDC); params.w unused.
    params: vec4<f32>,
    // The brush's colour in the color space's channels (.xyz); .w unused. Constant
    // across the stroke — per-segment variation (pressure, drain) rides the instance
    // attributes, and colour dynamics (below) jitter it per fragment.
    color: vec4<f32>,
    // Colour dynamics (§6.2): noise frequency per lookup axis
    // (across the stroke, along it) in .xy with 1/NOISE_TILE_PX pre-folded into
    // .z; .w unused.
    noise_freq: vec4<f32>,
    // Per colour-channel noise amplitude .xyz (all 0 = jitter off); .w unused.
    noise_amp: vec4<f32>,
    // Per-stroke noise lookup translation .xy (from the stroke seed); .zw unused.
    noise_off: vec4<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;
// Prefix-τ volume (§6.2, §6.6): x = travel, y = lateral, array layer = shape
// orientation (one slice per relative angle, sampled with wrapping). Round tips are
// rotation-invariant and have a single slice; image stamps carry up to
// MAX_ORIENTATION_LAYERS. A 2D array (not a 3D texture) so the mask keeps its full size.
@group(1) @binding(0) var prefix_tex: texture_2d_array<f32>;
// The colour-dynamics noise field (§6.2): a tileable 2-D tile
// (three signed channels) sampled at (lateral, arc) with a repeat sampler.
// A 1×1 zero tile is bound when the brush's jitter is off.
@group(2) @binding(0) var noise_tex: texture_2d<f32>;
@group(2) @binding(1) var noise_samp: sampler;

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
    // radius, arc length, the paint height laid per unit swept optical depth (the
    // brush's `add` source faded by its remaining load — the one amount knob,
    // §6.1/§6.2), and the brush opacity at this segment.
    @location(3) @interpolate(flat) geom: vec4<f32>,
    // Shape orientation in turns ∈ [0,1) (the relative angle between the shape's
    // native axis and the travel direction, picking the prefix-τ slice, §6.6); the
    // arc length of the segment's start from the stroke start (canvas px); the
    // segment's signed curvature (1/canvas px).
    @location(4) @interpolate(flat) extra: vec4<f32>,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// A canvas point's place in the frame the sweep is integrated in.
struct Sweep {
    // Brush-local: .x the travel coordinate in radius units (0 at the segment start,
    // growing the way the tip goes), .y the lateral offset in radius units.
    local: vec2<f32>,
    // The segment's travel **in this point's own measure**, radius units. On a curved
    // sweep a point on the outside of the turn sees the tip cross more of its own
    // frame than a point on the inside does; this is that amount.
    lr: f32,
    // Tip travel per unit of the point's own measure — `|R|/ρ`, and exactly 1 on a
    // straight sweep. Every depth read out of a prefix difference is in the point's
    // measure and has to be converted back by this before it means anything.
    warp: f32,
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
// footprint dragged across a point at unit speed; here the drag is `ρ/|R|` times the
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
    if kappa == 0.0 {
        out.local = vec2<f32>(along, across) / radius;
        out.lr = len / radius;
        out.warp = 1.0;
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
    return out;
}

// How many cross-sections the swept geometry is built from. The sweep is a curve, so
// its footprint is an annular *sector*, and this is how finely that sector is
// approximated by a triangle strip. Purely a fill-rate knob — the deposit itself is
// computed per fragment from the exact arc — so it is small, and the strip is
// circumscribed (below) so no value of it can clip the stroke.
const SWEEP_SLICES: u32 = 8u;
// = 2 * (SWEEP_SLICES + 1); the host's draw call must ask for exactly this many.
const SWEEP_VERTS: u32 = 18u;

// Build the swept geometry for one segment: a strip of `SWEEP_SLICES` quads covering
// brush-local x ∈ [-1, lr+1], y ∈ [-1, 1] — a radius of margin at each end for the
// round cap — bent around the segment's arc and mapped into the tile's apron'd target.
//
// Nothing here has to be tight, only *conservative*: a fragment outside the footprint
// differences two prefix taps to zero and writes nothing. So the outer rim is pushed
// out to circumscribe the true arc and the inner rim pulled in, and both errors cost
// a few fragments rather than a clipped stroke. What it must not do is cover a texel
// twice, which is why the host caps `radius · |curvature|` (`MAX_TIP_TURN`): the
// sector stays a simple polygon only while the tip's inner rim clears the centre.
fn sweep_vertex(
    vi: u32,
    start: vec2<f32>,
    dir: vec2<f32>,   // unit tangent at the segment start
    geom: vec4<f32>,  // radius, arc length, amount (height per unit τ), opacity
    extra: vec4<f32>, // orientation (turns ∈ [0,1)), arc length at start, curvature, _
) -> VsOut {
    let radius = geom.x;
    let len = geom.y;
    let kappa = extra.z;
    let perp = vec2<f32>(-dir.y, dir.x);

    let u = f32(vi >> 1u) / f32(SWEEP_SLICES); // 0..1 along the travel
    let side = select(-1.0, 1.0, (vi & 1u) != 0u); // the two lateral rims

    var canvas: vec2<f32>;
    if kappa == 0.0 {
        // The straight case is one quad's worth of geometry, cut into slices.
        let lr = len / radius;
        let cx = -1.0 + u * (lr + 2.0);
        canvas = start + dir * (cx * radius) + perp * (side * radius);
    } else {
        let ak = abs(kappa);
        let sk = sign(kappa);
        let abs_r = 1.0 / ak;
        // The tip's own turn, plus the angular margin that carries the round cap past
        // each end. The margin is measured at the *inner* rim, where a radius of
        // travel subtends the largest angle, so it covers both rims.
        let margin = radius / (abs_r - radius);
        let span = len * ak + 2.0 * margin;
        let delta = -margin + u * span;
        // Circumscribe: a chord of the outer rim would cut inside the true arc and
        // clip the stroke, so push it out by the half-slice it spans. The inner rim
        // gets the same factor the other way, which only over-covers.
        let half = 0.5 * span / f32(SWEEP_SLICES);
        let c = cos(half);
        var rho = abs_r - sk * side * radius;
        rho = select(rho * c, rho / c, sk * side < 0.0);
        // The same frame `sweep_at` unrolls, run forwards: `ρ sin δ` along the start
        // tangent and `sign(κ)(|R| − ρ cos δ)` across it.
        canvas = start
            + dir * (rho * sin(delta))
            + perp * (sk * (abs_r - rho * cos(delta)));
    }

    let s = xf.params.z;
    let ndc = (canvas - xf.params.xy) * vec2<f32>(s, -s) + vec2<f32>(-1.0, 1.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.canvas = canvas;
    out.start = start;
    out.dir = dir;
    out.geom = geom;
    out.extra = extra;
    return out;
}

// This fragment's place in its segment's travel frame.
fn sweep_of(in: VsOut) -> Sweep {
    return sweep_at(in.canvas, in.start, in.dir, in.geom.x, in.geom.y, in.extra.z);
}

// The brush's colour channels (per the color space) at this fragment: the
// stroke-constant colour, offset by the colour-dynamics noise field
// (§6.2). The lookup domain is **stroke-local** — (lateral offset from the
// centreline, arc length along the stroke), both in canvas px — so the pattern
// belongs to the stroke rather than to the canvas it lands on: the same gesture
// carries the same colour variation wherever it is drawn. Both coordinates are
// still functions of the fragment's canvas position and the segment, so the
// deposit remains a pure function of the two and tile aprons stay bit-consistent
// (§6.4).
fn brush_color(in: VsOut, sw: Sweep) -> vec3<f32> {
    if all(xf.noise_amp.xyz == vec3<f32>(0.0, 0.0, 0.0)) {
        return xf.color.xyz;
    }
    let radius = in.geom.x;
    // Brush-local y is in radius units; the noise domain is px on both axes, so
    // the pattern keeps one scale whatever size the tip is.
    let lateral = sw.local.y * radius;
    let arc = in.extra.y + sw.local.x * radius;
    let jitter = color_jitter(
        noise_tex, noise_samp, vec2<f32>(lateral, arc),
        xf.noise_freq, xf.noise_amp.xyz, xf.noise_off.xy,
    );
    return xf.color.xyz + jitter;
}

// Bilinear lookup of one orientation slice `zi` at brush-local (xl, yl) ∈ [-1,1]: 0
// before the leading edge, clamp-to-row-total past the trailing edge.
//
// Why this is hand-rolled instead of a ClampToEdge sampler with linear filtering:
//   1. Format. prefix_tex is R32Float (we need fp32 because each fragment takes
//      the *difference* of two prefix sums — see swept_depth — which is sensitive
//      to catastrophic cancellation, and a row's total κ can be large for opaque
//      brushes). Hardware linear filtering of a 32-bit-float texture requires the
//      FLOAT32_FILTERABLE feature, which is optional on WebGPU (our primary
//      target) and not enabled by this device. A filterable 16-bit format would
//      band exactly where the differencing is most sensitive.
//   2. Boundary semantics. The travel (x) axis is asymmetric: 0 before the
//      leading edge, clamp-to-row-total after the trailing edge. A single sampler
//      address mode can't express that — ClampToEdge would read the column-0
//      value across the whole start margin instead of 0 — so the `xl <= -1` guard
//      in `prefix_at` would be required regardless.
//   3. Precision. Fixed-function bilinear uses reduced-precision fractional
//      weights on many GPUs; the fp32 `mix` below keeps full precision for the
//      cancellation-sensitive difference.
fn prefix_slice(xl: f32, yl: f32, zi: i32) -> f32 {
    let dim = vec2<f32>(textureDimensions(prefix_tex));
    let ux = clamp(xl * 0.5 + 0.5, 0.0, 1.0) * dim.x - 0.5;
    let uy = clamp(yl * 0.5 + 0.5, 0.0, 1.0) * dim.y - 0.5;
    let x0 = floor(ux);
    let y0 = floor(uy);
    let fx = ux - x0;
    let fy = uy - y0;
    let xi0 = clamp(i32(x0), 0, i32(dim.x) - 1);
    let xi1 = clamp(i32(x0) + 1, 0, i32(dim.x) - 1);
    let yi0 = clamp(i32(y0), 0, i32(dim.y) - 1);
    let yi1 = clamp(i32(y0) + 1, 0, i32(dim.y) - 1);
    let p00 = textureLoad(prefix_tex, vec2<i32>(xi0, yi0), zi, 0).r;
    let p10 = textureLoad(prefix_tex, vec2<i32>(xi1, yi0), zi, 0).r;
    let p01 = textureLoad(prefix_tex, vec2<i32>(xi0, yi1), zi, 0).r;
    let p11 = textureLoad(prefix_tex, vec2<i32>(xi1, yi1), zi, 0).r;
    return mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);
}

// Trilinear lookup of the prefix-τ volume at brush-local (xl, yl) ∈ [-1,1] and shape
// orientation `w` (turns ∈ [0,1)). The orientation axis wraps (a full turn is periodic),
// so the two bracketing slices are chosen modulo the slice count. The x,y bilinear is
// hand-rolled for the same reasons (R32Float isn't filterable, the asymmetric travel-edge
// boundary, fp32 precision); 0 before the leading edge, clamp-to-row-total past the
// trailing edge. A single-slice (round) volume collapses to that one slice for every `w`.
fn prefix_at(xl: f32, yl: f32, w: f32) -> f32 {
    if xl <= -1.0 {
        return 0.0;
    }
    let nz = i32(textureNumLayers(prefix_tex));
    let fl = fract(w) * f32(nz);
    let l0f = floor(fl);
    let fw = fl - l0f;
    let z0 = (i32(l0f) % nz + nz) % nz;
    let z1 = (z0 + 1) % nz;
    let a = prefix_slice(xl, yl, z0);
    let b = prefix_slice(xl, yl, z1);
    return mix(a, b, fw);
}

// Swept geometric optical depth over the fragment's segment: the brush footprint
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
    let depth = prefix_at(sw.local.x, sw.local.y, w) - prefix_at(sw.local.x - sw.lr, sw.local.y, w);
    return max(depth, 0.0) * sw.warp;
}
