// Swept-segment brush rasterization for the Oklab color space (DESIGN.md §6.2).
//
// One instanced quad per stroke segment. The quad covers the segment's swept
// footprint; the fragment computes the brush *swept along the segment* as a
// difference of the precomputed prefix-τ texture, converts optical depth to
// coverage (α = 1 − exp(−flow·depth)), and writes:
//   target 0: color (Rgba16Float) — premultiplied Oklab (L,a,b) + coverage in a
//   target 1: aux   (Rg16Float)   — (height, wet) scaled by coverage
// Overlapping segment quads composite with premultiplied "over", which sums
// optical depth exactly, so the stroke reconstructs the continuous path integral.

struct TileXform {
    // params.xy = tile origin (canvas px); params.z = 2 / TILE_SIZE; params.w unused.
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;
// Prefix-τ: per row, the running integral of κ=−ln(1−coverage) along the brush's
// travel axis, in brush-local units (x ∈ [-1,1]). Sampled via textureLoad.
@group(1) @binding(0) var prefix_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>, // brush-local (x in radius units from start, y in [-1,1])
    @location(1) ch: vec3<f32>,    // pigment color (Oklab L,a,b)
    @location(2) misc: vec4<f32>,  // length/radius, flow, height, wet
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) start: vec2<f32>,
    @location(1) dir: vec2<f32>,    // unit tangent
    @location(2) geom: vec4<f32>,   // radius, length, flow, unused
    @location(3) ch: vec4<f32>,     // color channels (Oklab in xyz)
    @location(4) aux: vec2<f32>,    // height, wet
) -> VsOut {
    let radius = geom.x;
    let lr = geom.y / radius; // segment length in radius units

    // Quad spans local x ∈ [-1, lr+1], y ∈ [-1, 1] (a radius of margin each side).
    let cx = select(-1.0, lr + 1.0, (vi & 1u) != 0u);
    let cy = select(-1.0, 1.0, (vi >> 1u) != 0u);

    let perp = vec2<f32>(-dir.y, dir.x);
    let canvas = start + dir * (cx * radius) + perp * (cy * radius);
    let s = xf.params.z;
    let ndc = (canvas - xf.params.xy) * vec2<f32>(s, -s) + vec2<f32>(-1.0, 1.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.local = vec2<f32>(cx, cy);
    out.ch = ch.xyz;
    out.misc = vec4<f32>(lr, geom.z, aux.x, aux.y);
    return out;
}

// Bilinear lookup of the prefix-τ at brush-local (xl, yl), xl,yl ∈ [-1,1].
// Below the brush (xl ≤ -1) the running integral is 0; past it, it clamps to the
// row total (the last column), which is what an "over" beyond the tip should be.
//
// Why this is hand-rolled instead of a ClampToEdge sampler with linear filtering:
//   1. Format. prefix_tex is R32Float (we need fp32 because each fragment takes
//      the *difference* of two prefix sums — see fs_main — which is sensitive to
//      catastrophic cancellation, and a row's total κ can be large for opaque
//      brushes). Hardware linear filtering of a 32-bit-float texture requires the
//      FLOAT32_FILTERABLE feature, which is optional on WebGPU (our primary
//      target) and not enabled by this device. A filterable 16-bit format would
//      band exactly where the differencing is most sensitive.
//   2. Boundary semantics. The travel (x) axis is asymmetric: 0 before the
//      leading edge, clamp-to-row-total after the trailing edge. A single sampler
//      address mode can't express that — ClampToEdge would read the column-0
//      value across the whole start margin instead of 0 — so the `xl <= -1` guard
//      below would be required regardless.
//   3. Precision. Fixed-function bilinear uses reduced-precision fractional
//      weights on many GPUs; the fp32 `mix` below keeps full precision for the
//      cancellation-sensitive difference.
// The fetch cost (8 textureLoads/fragment) is negligible: these textures are tiny
// and cache-hot, and stamping is bound by blend/fill-rate, not these lookups.
fn prefix_at(xl: f32, yl: f32) -> f32 {
    if xl <= -1.0 {
        return 0.0;
    }
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
    let p00 = textureLoad(prefix_tex, vec2<i32>(xi0, yi0), 0).r;
    let p10 = textureLoad(prefix_tex, vec2<i32>(xi1, yi0), 0).r;
    let p01 = textureLoad(prefix_tex, vec2<i32>(xi0, yi1), 0).r;
    let p11 = textureLoad(prefix_tex, vec2<i32>(xi1, yi1), 0).r;
    return mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let px = in.local.x;
    let py = in.local.y;
    let lr = in.misc.x;
    let flow = in.misc.y;

    // Swept optical depth over the segment, then over → coverage.
    let depth = prefix_at(px, py) - prefix_at(px - lr, py);
    let coverage = 1.0 - exp(-flow * max(depth, 0.0));

    var out: FsOut;
    out.color = vec4<f32>(in.ch * coverage, coverage);
    out.aux = vec4<f32>(in.misc.z * coverage, in.misc.w * coverage, 0.0, 0.0);
    return out;
}
