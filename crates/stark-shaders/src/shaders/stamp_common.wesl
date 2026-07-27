// Shared swept-segment scaffolding for the stamp pass (DESIGN.md §6.2), imported
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
    // Colour dynamics (DESIGN.md §6.2): noise frequency per lookup axis
    // (across the stroke, along it) in .xy with 1/NOISE_TILE_PX pre-folded into
    // .z; .w unused.
    noise_freq: vec4<f32>,
    // Per colour-channel noise amplitude .xyz (all 0 = jitter off); .w unused.
    noise_amp: vec4<f32>,
    // Per-stroke noise lookup translation .xy (from the stroke seed); .zw unused.
    noise_off: vec4<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;
// Prefix-τ volume (DESIGN.md §6.2, §6.6): x = travel, y = lateral, array layer = shape
// orientation (one slice per relative angle, sampled with wrapping). Round tips are
// rotation-invariant and have a single slice; image stamps carry up to
// MAX_ORIENTATION_LAYERS. A 2D array (not a 3D texture) so the mask keeps its full size.
@group(1) @binding(0) var prefix_tex: texture_2d_array<f32>;
// The colour-dynamics noise field (DESIGN.md §6.2): a tileable 2-D tile
// (three signed channels) sampled at (lateral, arc) with a repeat sampler.
// A 1×1 zero tile is bound when the brush's jitter is off.
@group(2) @binding(0) var noise_tex: texture_2d<f32>;
@group(2) @binding(1) var noise_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>, // brush-local (x in radius units from start, y in [-1,1])
    @location(1) lr: f32, // segment length in radius units (the sweep's travel)
    // Paint height laid per unit swept optical depth — the brush's `add` source, faded
    // by its remaining load. The one amount knob (DESIGN.md §6.1, §6.2).
    @location(2) amount: f32,
    @location(3) op: f32, // brush opacity at this segment (drives the opacity channel)
    // Shape orientation in turns ∈ [0,1): the relative angle between the shape's native
    // axis and the travel direction, picking the prefix-τ orientation slice (§6.6).
    @location(4) orient: f32,
    // Arc length of the segment's start from the stroke start (.x, canvas px) and the
    // segment radius (.y) — together they recover the fragment's own arc length for the
    // colour-dynamics lookup.
    @location(5) sinfo: vec2<f32>,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

// Build the oriented quad for one swept segment. The quad spans brush-local
// x ∈ [-1, lr+1], y ∈ [-1, 1] (a radius of margin each side) and is placed onto
// the canvas along the segment's tangent, mapped into the tile's apron'd target.
fn sweep_vertex(
    vi: u32,
    start: vec2<f32>,
    dir: vec2<f32>,   // unit tangent
    geom: vec4<f32>,  // radius, length, amount (height per unit τ), opacity
    extra: vec4<f32>, // orientation (turns ∈ [0,1)), arc length at segment start, _, _
) -> VsOut {
    let radius = geom.x;
    let lr = geom.y / radius; // segment length in radius units

    let cx = select(-1.0, lr + 1.0, (vi & 1u) != 0u);
    let cy = select(-1.0, 1.0, (vi >> 1u) != 0u);

    let perp = vec2<f32>(-dir.y, dir.x);
    let canvas = start + dir * (cx * radius) + perp * (cy * radius);
    let s = xf.params.z;
    let ndc = (canvas - xf.params.xy) * vec2<f32>(s, -s) + vec2<f32>(-1.0, 1.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.local = vec2<f32>(cx, cy);
    out.lr = lr;
    out.amount = geom.z;
    out.op = geom.w;
    out.orient = extra.x;
    out.sinfo = vec2<f32>(extra.y, radius);
    return out;
}

// The brush's colour channels (per the color space) at this fragment: the
// stroke-constant colour, offset by the colour-dynamics noise field (DESIGN.md
// §6.2). The lookup domain is **stroke-local** — (lateral offset from the
// centreline, arc length along the stroke), both in canvas px — so the pattern
// belongs to the stroke rather than to the canvas it lands on: the same gesture
// carries the same colour variation wherever it is drawn. Both coordinates are
// still functions of the fragment's canvas position and the segment, so the
// deposit remains a pure function of the two and tile aprons stay bit-consistent
// (§6.4). The arc is clamped to the segment body, which makes it *continuous*
// across the overlapping quads of consecutive segments (a trailing margin's arc
// equals the next segment's start).
fn brush_color(in: VsOut) -> vec3<f32> {
    if all(xf.noise_amp.xyz == vec3<f32>(0.0, 0.0, 0.0)) {
        return xf.color.xyz;
    }
    let radius = in.sinfo.y;
    // Brush-local y is in radius units; the noise domain is px on both axes, so
    // the pattern keeps one scale whatever size the tip is.
    let lateral = in.local.y * radius;
    let arc = in.sinfo.x + clamp(in.local.x, 0.0, in.lr) * radius;
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
// slice. Sums exactly across a stroke's overlapping segment quads (under "over" for τ, or
// additively for amounts), reconstructing the continuous path integral.
fn swept_depth(in: VsOut) -> f32 {
    let depth = prefix_at(in.local.x, in.local.y, in.orient) - prefix_at(in.local.x - in.lr, in.local.y, in.orient);
    return max(depth, 0.0);
}
