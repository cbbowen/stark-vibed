// Canvas -> surface presentation pass (DESIGN.md §6.4).
//
// Draws one screen-aligned quad per visible tile. The unit quad is generated
// from the vertex index (triangle-strip, 4 verts); the per-tile canvas-space
// origin arrives as an instance attribute. The view uniform maps canvas pixels
// to clip space, encoding pan (translate) and zoom (scale).

struct View {
    // Packed into vec4s for an unambiguous 32-byte std140 layout that matches
    // the Rust ViewUniform exactly. ndc = canvas_pos * scale + translate.
    st: vec4<f32>,   // scale.xy, translate.xy
    // misc.x = tile_size (interior). Tiles carry an apron, so sample the interior
    // sub-rect: uv = corner * misc.y + misc.z (misc.y = TILE_SIZE/TILE_TEX,
    // misc.z = APRON/TILE_TEX); bilinear then reaches into the apron at edges.
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var samp: sampler;
@group(1) @binding(0) var tile_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) origin: vec2<f32>,
) -> VsOut {
    // Triangle-strip corners: (0,0) (1,0) (0,1) (1,1).
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let scale = view.st.xy;
    let translate = view.st.zw;
    let canvas_pos = origin + corner * view.misc.x;
    let ndc = canvas_pos * scale + translate;

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = corner * view.misc.y + view.misc.z; // interior sub-rect of the apron'd texture
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // TODO(step 4, DESIGN.md §6.5): tiles store Oklab; convert Oklab -> display
    // here. For the skeleton the color channel holds linear RGBA and is passed
    // through, with any encoding handled by the target texture format.
    return textureSample(tile_tex, samp, in.uv);
}
