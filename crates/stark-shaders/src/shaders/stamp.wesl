// Brush stamp rasterization (DESIGN.md §6.2).
//
// Renders one instanced quad per brush stamp into a single tile's color
// texture. Each stamp is a soft radial dab; stamps blend over one another with
// premultiplied-alpha "over", so a dense run of stamps builds a continuous
// stroke. Output is premultiplied (matches the tile color channel and the
// present pass).
//
// Color is linear RGBA for the MVP; it becomes Oklab in step 4 (DESIGN.md §6.5).

struct TileXform {
    // params.xy = this tile's origin in canvas pixels.
    // params.z  = 2 / TILE_SIZE (canvas px -> NDC scale). params.w unused.
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> xf: TileXform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) quad: vec2<f32>,   // local stamp coord in [-1, 1]
    @location(1) color: vec4<f32>,  // straight (un-premultiplied) RGBA
    @location(2) hf: vec2<f32>,     // hardness, flow
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec2<f32>,
    @location(1) shape: vec4<f32>,  // radius, hardness, flow, (unused)
    @location(2) color: vec4<f32>,
) -> VsOut {
    // Triangle-strip quad corners: (-1,-1) (1,-1) (-1,1) (1,1).
    let corner = vec2<f32>(
        f32(vi & 1u) * 2.0 - 1.0,
        f32((vi >> 1u) & 1u) * 2.0 - 1.0,
    );
    let radius = shape.x;
    let canvas_pos = center + corner * radius;

    let s = xf.params.z;
    let ndc = (canvas_pos - xf.params.xy) * vec2<f32>(s, -s) + vec2<f32>(-1.0, 1.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.quad = corner;
    out.color = color;
    out.hf = shape.yz;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.quad);
    // Solid out to `hardness`, fading to zero at the stamp edge (d = 1).
    let hardness = min(in.hf.x, 0.99);
    let flow = in.hf.y;
    let coverage = flow * (1.0 - smoothstep(hardness, 1.0, d)) * in.color.a;
    // Premultiplied output.
    return vec4<f32>(in.color.rgb * coverage, coverage);
}
