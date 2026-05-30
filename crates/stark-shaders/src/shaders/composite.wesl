// Compositing pass A (DESIGN.md §6.3): draw each visible tile's channels into
// viewport-sized offscreen targets. Color composites with premultiplied "over",
// aux (height, wet) accumulates additively. The media pass then lights the
// result. Vertex setup matches the present pass: one instanced quad per tile.

struct View {
    st: vec4<f32>,   // scale.xy, translate.xy  (canvas px -> NDC)
    // misc.x = tile_size (interior, canvas px). The tile texture carries an apron
    // around the interior, so we sample only the interior sub-rect and let the
    // bilinear filter reach into the apron at the edges (kills tile seams):
    //   uv = corner * misc.y + misc.z,  misc.y = TILE_SIZE/TILE_TEX, misc.z = APRON/TILE_TEX.
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var samp: sampler;
@group(1) @binding(0) var tile_color: texture_2d<f32>;
@group(1) @binding(1) var tile_aux: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) opacity: f32,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) origin: vec2<f32>,
    @location(1) opacity: f32,
) -> VsOut {
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let scale = view.st.xy;
    let translate = view.st.zw;
    let canvas_pos = origin + corner * view.misc.x;
    let ndc = canvas_pos * scale + translate;

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = corner * view.misc.y + view.misc.z; // interior sub-rect of the apron'd texture
    out.opacity = opacity;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    // Scaling premultiplied color (and additive aux) by layer opacity yields a
    // correct Normal-blend "layer at opacity% over below".
    var out: FsOut;
    out.color = textureSample(tile_color, samp, in.uv) * in.opacity;
    out.aux = textureSample(tile_aux, samp, in.uv) * in.opacity;
    return out;
}
