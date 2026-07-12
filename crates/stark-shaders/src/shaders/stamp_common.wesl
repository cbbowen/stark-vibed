// Shared swept-segment scaffolding for the stamp pass (DESIGN.md §6.2), imported
// by the per-color-space stamp shaders. Everything here is color-space-agnostic:
// the tile transform + prefix-τ bindings, the oriented-quad vertex, and the
// swept optical-depth lookup. Each variant differs only in how the fragment
// turns that depth into channel deposits (see stamp_oklab.wesl; the Mixbox space
// reuses it, since its deposit is the same premultiplied-"over").

struct TileXform {
    // params.xy = tile *texture* top-left (interior origin − apron) in canvas px;
    // params.z = 2 / TILE_TEX (full interior+apron maps to NDC); params.w unused.
    params: vec4<f32>,
    // The brush's colour in the color space's channels (.xyz); .w unused. Constant
    // across the stroke — per-segment variation (pressure, drain) rides the instance
    // attributes instead.
    color: vec4<f32>,
    // surf.x = inv surface-tile (canvas px → bump uv); surf.y = tooth; .zw unused.
    surf: vec4<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;
// Prefix-τ volume (DESIGN.md §6.2, §6.6): x = travel, y = lateral, array layer = shape
// orientation (one slice per relative angle, sampled with wrapping). Round tips are
// rotation-invariant and have a single slice; image stamps carry up to
// MAX_ORIENTATION_LAYERS. A 2D array (not a 3D texture) so the mask keeps its full size.
@group(1) @binding(0) var prefix_tex: texture_2d_array<f32>;
// The canvas surface (tileable bump/height), sampled in canvas space so the
// weave gates deposition consistently regardless of which tile a fragment lands
// in (DESIGN.md §6.4).
@group(2) @binding(0) var surface_tex: texture_2d<f32>;
@group(2) @binding(1) var surface_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>, // brush-local (x in radius units from start, y in [-1,1])
    @location(1) misc: vec4<f32>,  // length/radius, flow, height(thickness rate), wet
    @location(2) op: f32, // brush opacity at this segment (drives the opacity channel)
    // Shape orientation in turns ∈ [0,1): the relative angle between the shape's native
    // axis and the travel direction, picking the prefix-τ orientation slice (§6.6).
    @location(3) orient: f32,
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
    geom: vec4<f32>,  // radius, length, flow, _
    aux: vec4<f32>,   // height(thickness rate), wet, opacity, _
    extra: vec4<f32>, // orientation (turns ∈ [0,1)), _, _, _
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
    out.misc = vec4<f32>(lr, geom.z, aux.x, aux.y);
    out.op = aux.z;
    out.orient = extra.x;
    return out;
}

// The brush's colour channels (per the color space), constant over the stroke.
fn brush_color() -> vec3<f32> {
    return xf.color.xyz;
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
    let depth = prefix_at(in.local.x, in.local.y, in.orient) - prefix_at(in.local.x - in.misc.x, in.local.y, in.orient);
    return max(depth, 0.0);
}

// Gate `coverage` by the canvas tooth at the fragment's canvas position (DESIGN.md
// §6.4): paint adheres to the weave's peaks and skips its valleys when light, and
// the effect fades as coverage builds (valleys fill). `frag_xy` is the fragment's
// framebuffer pixel; the tile texture is 1:1 with canvas, so canvas = tex_origin +
// frag_xy.
fn surface_tooth(height: f32, coverage: f32, frag_xy: vec2<f32>) -> f32 {
    // TODO: Find a more natural and physically-accurate way to implement this.
    // While this is a pass-through stub, NO shader reads the canvas surface at
    // stamp/integrate time — the group(2) `surface_tex`/`surface_samp` bindings above
    // are placeholders for this TODO, and the weave currently shows only through the
    // media pass's `thickness = height − surface` and relief normals (DESIGN §6.3/§6.4).
    return coverage;
}
