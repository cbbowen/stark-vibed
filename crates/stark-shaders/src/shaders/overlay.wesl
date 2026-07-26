// Selection outline — the "marching ants" (DESIGN.md §6.8).
//
// Drawn over the finished image, one instanced quad per mask tile, in the same
// canvas→NDC frame as the compositor. The outline is recovered from the mask itself
// rather than from the shape that produced it, so it costs nothing to keep correct
// under union/subtract/intersect: the boundary is the mask's 0.5-contour, and the
// distance to it is the classic first-order estimate `(m − 0.5) / |∇m|`. Taking the
// gradient at exactly one canvas pixel and converting the distance to *screen* pixels
// with the zoom gives a line of constant on-screen width at any zoom, and one that
// stays thin over a feathered edge (where |∇m| is small, so points a little way off
// the contour are already far in these units) instead of smearing across the ramp.

struct View {
    st: vec4<f32>,   // scale.xy, translate.xy (canvas px → NDC)
    misc: vec4<f32>, // tile_size (interior, canvas px), uv_scale, uv_bias, zoom
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var samp: sampler;
@group(1) @binding(0) var mask: texture_2d<f32>;

/// Half-width of the drawn outline, in screen px.
const HALF_WIDTH: f32 = 1.0;
/// Dash period along the screen diagonal, in px.
const DASH: f32 = 8.0;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @location(0) origin: vec2<f32>) -> VsOut {
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let canvas_pos = origin + corner * view.misc.x;
    var out: VsOut;
    out.clip = vec4<f32>(canvas_pos * view.st.xy + view.st.zw, 0.0, 1.0);
    // Interior sub-rect of the apron'd mask texture, exactly as the compositor maps
    // a colour tile — so the taps below reach into the apron (the neighbour's real
    // coverage) at the tile edge rather than clamping.
    out.uv = corner * view.misc.y + view.misc.z;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // uv step of one canvas px: (TILE_SIZE/TILE_TEX) / TILE_SIZE = 1/TILE_TEX.
    let d = view.misc.y / view.misc.x;
    let m = textureSample(mask, samp, in.uv).r;
    let mx0 = textureSample(mask, samp, in.uv - vec2<f32>(d, 0.0)).r;
    let mx1 = textureSample(mask, samp, in.uv + vec2<f32>(d, 0.0)).r;
    let my0 = textureSample(mask, samp, in.uv - vec2<f32>(0.0, d)).r;
    let my1 = textureSample(mask, samp, in.uv + vec2<f32>(0.0, d)).r;

    // Coverage gradient per canvas px. Flat mask (interior or exterior) ⇒ no contour
    // anywhere near, so nothing to draw.
    let g = vec2<f32>(mx1 - mx0, my1 - my0) * 0.5;
    let gl = length(g);
    if gl < 1e-4 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let dist_px = abs((m - 0.5) / gl) * view.misc.w;
    let alpha = 1.0 - smoothstep(HALF_WIDTH - 0.5, HALF_WIDTH + 0.5, dist_px);
    if alpha <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    // Alternating black/white along the screen diagonal: legible over paint of any
    // value, which a single-colour outline is not.
    let dash = step(0.5, fract((in.clip.x + in.clip.y) / DASH));
    return vec4<f32>(vec3<f32>(dash), alpha);
}
