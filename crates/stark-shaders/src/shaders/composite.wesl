// Compositing pass A (DESIGN.md §6.3): draw each visible tile's channels into
// viewport-sized offscreen targets. Color composites with premultiplied "over",
// aux (height, wet) accumulates additively. The media pass then lights the
// result. Vertex setup matches the present pass: one instanced quad per tile.

struct View {
    st: vec4<f32>,   // scale.xy, translate.xy  (canvas px -> NDC)
    misc: vec4<f32>, // tile_size, (unused)
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var samp: sampler;
@group(1) @binding(0) var tile_color: texture_2d<f32>;
@group(1) @binding(1) var tile_aux: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @location(0) origin: vec2<f32>) -> VsOut {
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let scale = view.st.xy;
    let translate = view.st.zw;
    let canvas_pos = origin + corner * view.misc.x;
    let ndc = canvas_pos * scale + translate;

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = corner;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var out: FsOut;
    out.color = textureSample(tile_color, samp, in.uv);
    out.aux = textureSample(tile_aux, samp, in.uv);
    return out;
}
