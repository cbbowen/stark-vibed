// Shared scaffolding for the media/lighting pass (DESIGN.md §6.3), imported by
// the per-color-space media shaders. Each variant only differs in how it derives
// the un-lit linear *base color* from the composited channels; everything else —
// the bindings, the fullscreen vertex, the impasto normal-from-height, the
// directional lighting + wet gloss, and the composite-over-background + sRGB
// encode — lives here so it is written once.

import package::color::{linear_to_srgb};

struct Media {
    light: vec4<f32>,  // light dir xyz (normalized in shader), height_strength
    bg: vec4<f32>,     // background (substrate) in latent color space
    shade: vec4<f32>,  // ambient, diffuse_k, spec_strength, shininess
    surf_a: vec4<f32>, // canvas_origin.xy (canvas px at pixel 0), canvas_per_px, inv_tile
    surf_b: vec4<f32>, // surface_strength, _, _, _
}

@group(0) @binding(0) var<uniform> m: Media;
@group(0) @binding(1) var comp_color: texture_2d<f32>;
@group(0) @binding(2) var comp_aux: texture_2d<f32>;
@group(0) @binding(3) var surface: texture_2d<f32>; // tileable canvas bump (height)
@group(0) @binding(4) var surface_samp: sampler;

// Hardness of the paint's emergence above the surface.
const PAINT_OPACITY: f32 = 100.0;

// Fullscreen triangle clip position from a vertex index.
fn fullscreen_clip(vi: u32) -> vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Absolute canvas surface height at screen pixel `p` (sampled in canvas space, so
// the weave is fixed to the canvas). `Flat` is height 0, so it adds no relief.
fn surface_at(p: vec2<i32>) -> f32 {
    let canvas = m.surf_a.xy + vec2<f32>(p) * m.surf_a.z; // surf_a.z = canvas px / screen px
    let hs = textureSampleLevel(surface, surface_samp, canvas * m.surf_a.w, 0.0).r;
    return m.surf_b.x * hs; // surf_b.x = surface_strength
}

// Accumulated paint level (absolute height the paint has built up to, DESIGN §6.4).
fn paint_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    let c = clamp(p, vec2<i32>(0), dim - vec2<i32>(1));
    return textureLoad(comp_aux, c, 0).x;
}

// The visible top surface: paint floats the surface up where it has pooled above
// it, and the canvas pokes through where the paint is thinner. So uniform thick
// paint reads flat (its own level, no weave), while thin paint reveals the weave.
fn height_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    return max(surface_at(p), paint_at(p, dim));
}

// The thickness of the paint.
fn thickness_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    return max(paint_at(p, dim) - surface_at(p), 0);
}

// Determines the opacity of the paint at a given location based on its thickness there.
fn paint_opacity_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    let thickness = thickness_at(p, dim);
    return 1.0 - exp(-PAINT_OPACITY * thickness);
}

fn background_color() -> vec4<f32> {
    return m.bg;
}

// Surface normal from the screen-space height gradient (impasto relief).
fn surface_normal(p: vec2<i32>, dim: vec2<i32>) -> vec3<f32> {
    let strength = m.light.w;
    let hl = height_at(p + vec2<i32>(-1, 0), dim);
    let hr = height_at(p + vec2<i32>(1, 0), dim);
    let hu = height_at(p + vec2<i32>(0, -1), dim);
    let hd = height_at(p + vec2<i32>(0, 1), dim);
    return normalize(vec3<f32>(-(hr - hl) * strength, -(hd - hu) * strength, 1.0));
}

// Light the un-lit linear `base` color (directional + wet gloss, impasto from the
// height field at pixel `p`) and encode to display sRGB. The single shared tail of
// every media pass; the paint-over-background composite already happened in the
// caller's latent space, so `base` is the final un-lit color.
fn finish(base: vec3<f32>, wet: f32, p: vec2<i32>, dim: vec2<i32>) -> vec4<f32> {
    let n = surface_normal(p, dim);
    let light_dir = normalize(m.light.xyz);
    let diffuse = max(dot(n, light_dir), 0.0);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let half_v = normalize(light_dir + view_dir);
    let spec = pow(max(dot(n, half_v), 0.0), m.shade.w) * m.shade.z * clamp(wet, 0.0, 1.0);
    let shade = m.shade.x + m.shade.y * diffuse;
    return vec4<f32>(base * shade + vec3<f32>(spec), 1.0);
}
