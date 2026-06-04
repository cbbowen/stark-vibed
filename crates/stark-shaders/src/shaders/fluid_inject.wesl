// Fluid velocity injection (DESIGN.md §6.2, "Fluid"). Rasterizes the stroke's swept
// segments as oriented capsule quads into a velocity region (Rg16Float, vx/vy),
// additively, so each footprint stamps `dir · strength · falloff` — a velocity field
// pointing along the brush's motion, strongest at the centre and 0 at the brush
// radius. The advect pass then drags the wet paint along this field. Quad geometry
// matches `sweep_vertex` (stamp_common), but mapped to the region's NDC.

struct View {
    st: vec4<f32>,    // scale.xy, translate.xy  (canvas px → region NDC)
    knobs: vec4<f32>, // velocity magnitude (canvas px / iteration), _, _, _
}

@group(0) @binding(0) var<uniform> v: View;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>, // capsule coords: x along axis (radius units), y lateral [-1,1]
    @location(1) lr: f32,          // segment length in radius units
    @location(2) vel: vec2<f32>,   // dir · magnitude (constant per quad)
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) start: vec2<f32>,
    @location(1) dir: vec2<f32>,
    @location(2) geom: vec4<f32>,
) -> VsOut {
    let radius = geom.x;
    let lr = geom.y / radius;
    let cx = select(-1.0, lr + 1.0, (vi & 1u) != 0u);
    let cy = select(-1.0, 1.0, (vi >> 1u) != 0u);

    let perp = vec2<f32>(-dir.y, dir.x);
    let canvas = start + dir * (cx * radius) + perp * (cy * radius);
    let ndc = canvas * v.st.xy + v.st.zw;

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.local = vec2<f32>(cx, cy);
    out.lr = lr;
    out.vel = dir * v.knobs.x;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Capsule falloff: distance to the segment axis in radius units, 0 at the radius.
    let bx = clamp(in.local.x, 0.0, in.lr);
    let d = length(vec2<f32>(in.local.x - bx, in.local.y));
    let falloff = clamp(1.0 - d, 0.0, 1.0);
    return vec4<f32>(in.vel * falloff, 0.0, 0.0);
}
