// Separable Gaussian conditioning of the injected velocity field (DESIGN.md §6.2, the
// **drag** axis). The stroke injects velocity as a sum of overlapping per-segment capsule
// quads, which gives it a hard footprint edge and a fine per-segment ripple in magnitude.
// A real flow varies on the brush's scale, not the segment's, so two passes of a 1-D
// Gaussian (σ ∝ brush radius) smooth the field: they soften the footprint-edge velocity
// step (whose sharp divergence would otherwise pile paint at the drag boundary) and wash
// out the per-segment ripple, while keeping the large-scale convergence (paint genuinely
// piling at the leading edge). It is a cheap step toward incompressibility — lowering the
// spurious divergence without a full pressure projection. (The advection's own monotonicity
// is handled by its CFL limit, not here.) Velocity is transient, so this touches no stored
// pixels and stays replay-deterministic.

struct Params {
    knobs: vec4<f32>, // dir.x, dir.y (1,0 = horizontal | 0,1 = vertical), sigma (px), _
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d<f32>;

const MAX_RADIUS: i32 = 24;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<i32>(textureDimensions(src));
    let c = vec2<i32>(floor(frag.xy));
    let dir = vec2<i32>(i32(p.knobs.x), i32(p.knobs.y));
    let sigma = max(p.knobs.z, 1e-4);
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let radius = min(i32(ceil(3.0 * sigma)), MAX_RADIUS);

    var sum = vec2<f32>(0.0);
    var wsum = 0.0;
    for (var i = -radius; i <= radius; i = i + 1) {
        let w = exp(-f32(i * i) * inv2s2);
        let tc = clamp(c + dir * i, vec2<i32>(0), dim - vec2<i32>(1));
        sum = sum + w * textureLoad(src, tc, 0).xy;
        wsum = wsum + w;
    }
    return vec4<f32>(sum / wsum, 0.0, 0.0);
}
