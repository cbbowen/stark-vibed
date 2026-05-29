// Media pass B (DESIGN.md §6.3): the "old masters" payoff.
//
// Reads the composited Oklab color and (height, wet) aux, derives a surface
// normal from the height field, applies directional lighting so impasto ridges
// catch the light, adds wet gloss, converts Oklab -> display sRGB, and composites
// the lit paint over the background. This is the single shader stage intended
// for iteration (Kubelka-Munk, granulation, varnish, ...).

import package::color::{oklab_to_linear_srgb, linear_to_srgb};

struct Media {
    light: vec4<f32>,  // light dir xyz (will be normalized), height_strength
    bg: vec4<f32>,     // background linear-RGB, unused
    shade: vec4<f32>,  // ambient, diffuse_k, spec_strength, shininess
}

@group(0) @binding(0) var<uniform> m: Media;
@group(0) @binding(1) var comp_color: texture_2d<f32>;
@group(0) @binding(2) var comp_aux: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle.
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn height_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    let c = clamp(p, vec2<i32>(0), dim - vec2<i32>(1));
    return textureLoad(comp_aux, c, 0).x;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(frag.xy));
    let dim = vec2<i32>(textureDimensions(comp_aux));

    let color = textureLoad(comp_color, p, 0);
    let aux = textureLoad(comp_aux, p, 0);
    let coverage = color.a;
    let wet = aux.y;

    // Surface normal from the height gradient (screen space).
    let strength = m.light.w;
    let hl = height_at(p + vec2<i32>(-1, 0), dim);
    let hr = height_at(p + vec2<i32>(1, 0), dim);
    let hu = height_at(p + vec2<i32>(0, -1), dim);
    let hd = height_at(p + vec2<i32>(0, 1), dim);
    let n = normalize(vec3<f32>(-(hr - hl) * strength, -(hd - hu) * strength, 1.0));

    let light_dir = normalize(m.light.xyz);
    let diffuse = max(dot(n, light_dir), 0.0);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let half_v = normalize(light_dir + view_dir);
    let spec = pow(max(dot(n, half_v), 0.0), m.shade.w) * m.shade.z * clamp(wet, 0.0, 1.0);

    // Un-premultiply Oklab, convert to linear sRGB, then light it.
    let oklab = color.rgb / max(coverage, 1e-4);
    let base = oklab_to_linear_srgb(oklab);
    let shade = m.shade.x + m.shade.y * diffuse;
    let lit = base * shade + vec3<f32>(spec);

    // Composite lit paint over the background, then encode to display sRGB.
    let out_lin = mix(m.bg.rgb, lit, coverage);
    return vec4<f32>(linear_to_srgb(out_lin), 1.0);
}
