// Shared scaffolding for the media/lighting pass (DESIGN.md §6.3), imported by
// the per-color-space media shaders. Each variant only differs in how it derives
// the un-lit linear *base color* from the composited channels; everything else —
// the bindings, the fullscreen vertex, the impasto normal-from-height, the
// directional lighting + wet gloss, and the composite-over-background + sRGB
// encode — lives here so it is written once.

import package::color::{linear_to_srgb};

struct Media {
    light: vec4<f32>,  // light dir xyz (normalized in shader), height_strength
    bg: vec4<f32>,     // background (substrate) linear-RGB
    shade: vec4<f32>,  // ambient, diffuse_k, spec_strength, shininess
    surf_a: vec4<f32>, // canvas_origin.xy (canvas px at pixel 0), canvas_per_px, inv_tile
    surf_b: vec4<f32>, // surface_strength, _, _, _
}

@group(0) @binding(0) var<uniform> m: Media;
@group(0) @binding(1) var comp_color: texture_2d<f32>;
@group(0) @binding(2) var comp_aux: texture_2d<f32>;
@group(0) @binding(3) var surface: texture_2d<f32>; // tileable canvas bump (height)
@group(0) @binding(4) var surface_samp: sampler;

// Fullscreen triangle clip position from a vertex index.
fn fullscreen_clip(vi: u32) -> vec4<f32> {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Total relief height at screen pixel `p`: the painted impasto (`comp_aux.x`)
// plus the canvas surface bump sampled in canvas space (so the weave is fixed to
// the canvas and shows even where there is no paint).
fn height_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    let c = clamp(p, vec2<i32>(0), dim - vec2<i32>(1));
    let impasto = textureLoad(comp_aux, c, 0).x;

    let canvas = m.surf_a.xy + vec2<f32>(p) * m.surf_a.z; // surf_a.z = canvas px / screen px
    let hs = textureSampleLevel(surface, surface_samp, canvas * m.surf_a.w, 0.0).r - 0.5;
    return impasto + m.surf_b.x * hs; // surf_b.x = surface_strength
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

// Light the un-lit linear `base` paint color (directional + wet gloss, impasto
// from the height field at pixel `p`), composite over the background by
// `coverage`, and encode to display sRGB. The single shared tail of every media
// pass; only `base`/`coverage` are color-space-specific.
fn finish(base: vec3<f32>, coverage: f32, wet: f32, p: vec2<i32>, dim: vec2<i32>) -> vec4<f32> {
    let n = surface_normal(p, dim);
    let light_dir = normalize(m.light.xyz);
    let diffuse = max(dot(n, light_dir), 0.0);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let half_v = normalize(light_dir + view_dir);
    let spec = pow(max(dot(n, half_v), 0.0), m.shade.w) * m.shade.z * clamp(wet, 0.0, 1.0);

    let shade = m.shade.x + m.shade.y * diffuse;
    let lit = base * shade + vec3<f32>(spec);
    // The bare substrate catches the same relief lighting, so the canvas weave is
    // visible where paint is thin or absent. Normalize by the shading a *flat*
    // surface would give, so a flat surface leaves the background unchanged and
    // only relief (weave) modulates it — keeping the flat default orthogonal.
    let shade_flat = m.shade.x + m.shade.y * max(normalize(m.light.xyz).z, 0.0);
    let bg_lit = m.bg.rgb * (shade / max(shade_flat, 1e-4));
    let out_lin = mix(bg_lit, lit, coverage);
    return vec4<f32>(linear_to_srgb(out_lin), 1.0);
}
