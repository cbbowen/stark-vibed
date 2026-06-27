// Shared scaffolding for the media/lighting pass (DESIGN.md §6.3), imported by
// the per-color-space media shaders. Each variant only differs in how it derives
// the un-lit linear *base color* from the composited channels; everything else —
// the bindings, the fullscreen vertex, the impasto normal-from-height, the
// directional lighting + wet gloss, and the composite-over-background + sRGB
// encode — lives here so it is written once.

import package::color::{linear_to_srgb};
import package::noise::{noise_2_to_4};

struct Media {
    light: vec4<f32>,  // _, _, _, height_strength (relief slope; xyz unused under IBL)
    bg: vec4<f32>,     // background (substrate) in latent color space
    shade: vec4<f32>,  // exposure, diffuse_lod, wet_gloss, max_lod
    surf_a: vec4<f32>, // canvas_origin.xy (canvas px at pixel 0), canvas_per_px, inv_tile
    surf_b: vec4<f32>, // surface_strength, normal_dither, _, _
}

@group(0) @binding(0) var<uniform> m: Media;
@group(0) @binding(1) var comp_color: texture_2d<f32>;
@group(0) @binding(2) var comp_aux: texture_2d<f32>;
@group(0) @binding(3) var surface: texture_2d<f32>; // tileable canvas bump (height)
@group(0) @binding(4) var surface_samp: sampler;
// The HDR lighting environment (equirectangular, mipped) for image-based lighting
// (DESIGN.md §6.3): diffuse irradiance from a high mip in the normal direction, wet
// specular from a gloss-selected mip in the reflection direction.
@group(0) @binding(5) var env: texture_2d<f32>;
@group(0) @binding(6) var env_samp: sampler;

const PI: f32 = 3.14159265359;

// Opacity build-up rate (DESIGN.md §6.1, normalized representation). A paint texel
// stores a per-unit *opacity* (color.a) and a *thickness* (aux.x); its visible alpha
// is the translucent-slab law `1 − exp(−OPACITY_K · opacity · thickness)`. Calibrated
// so a normal opaque stroke reads solid.
const OPACITY_K: f32 = 1.0;

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

fn paint_height_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    let c = clamp(p, vec2<i32>(0), dim - vec2<i32>(1));
    return textureLoad(comp_aux, c, 0).x;
}

// The paint's visible alpha at `p`: its per-unit opacity (color.a) combined with its
// thickness (aux.x) as a translucent slab. 0 on bare canvas, → 1 for thick opaque
// paint. This is the single source of "how much paint shows", replacing the old
// coverage × thickness-opacity double-gate (DESIGN.md §6.1).
fn paint_vis_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    let c = clamp(p, vec2<i32>(0), dim - vec2<i32>(1));
    let op = textureLoad(comp_color, c, 0).a;
    // Paint thickness is paint height - surface height.
    let t = max(textureLoad(comp_aux, c, 0).x - surface_at(p), 0);
    return 1.0 - exp(-OPACITY_K * op * t);
}

// The relief height for impasto normals: the canvas surface plus the paint's
// thickness, gated by how visible the paint is so a near-transparent tail doesn't
// emboss a ghost ridge around the stroke (DESIGN.md §6.4).
fn height_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    return max(surface_at(p), paint_height_at(p, dim));
}

fn background_color() -> vec4<f32> {
    return m.bg;
}

// Surface normal from the height gradient (impasto relief). The gradient is taken
// over screen pixels but divided by the canvas px they span (`surf_a.z`), so the
// slope is per *canvas* px — the relief (and its gloss) is zoom-invariant rather
// than steepening as you zoom out.
fn surface_normal(p: vec2f, dim: vec2<i32>) -> vec3<f32> {
    let i = vec2<i32>(floor(p));
    let strength = m.light.w / m.surf_a.z;
    // Normal dithering (canvas-anchored noise, scaled by `normal_dither`): breaks up
    // banding in the relief shading. Position-seeded, so it is NOT translation
    // invariant — the seam tests set the amplitude to 0, like `surface_strength`.
    let noise = noise_2_to_4(m.surf_a.xy + p * m.surf_a.z, 0x12778282) * (m.surf_b.y / 64.0);
    let hl = height_at(i + vec2<i32>(-1, 0), dim) + noise.x;
    let hr = height_at(i + vec2<i32>(1, 0), dim) + noise.y;
    let hu = height_at(i + vec2<i32>(0, -1), dim) + noise.z;
    let hd = height_at(i + vec2<i32>(0, 1), dim) + noise.w;
    return normalize(vec3<f32>(-(hr - hl) * strength, -(hd - hu) * strength, 1.0));
}

// Equirectangular lookup: world direction (y-up) → environment UV. Longitude wraps
// (Repeat), latitude clamps; the top row (v=0) is straight up (+Y).
fn equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(0.5 + atan2(dir.x, -dir.z) / (2.0 * PI), 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI);
}

// Canvas-space normal → world direction for the environment lookup. The canvas is
// an easel *facing the viewer*, so a flat canvas (+Z) looks straight ahead into the
// environment (the equirect's center, on the horizon) — not up at the pole, where
// the equirect is degenerate and tiny normal tilts would jump erratically across
// the seam (saturating relief and defeating `height_strength`). Screen +X → world
// +X (right); screen +Y is down → world −Y; +Z (toward viewer) → −Z (forward).
fn env_dir(n: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(n.x, n.y, -n.z);
}

fn sample_env(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    return textureSampleLevel(env, env_samp, equirect_uv(dir), lod).rgb;
}

// Fresnel–Schlick with a roughness floor (Lagarde): the reflectance rises from F0
// at normal incidence to ~1 at grazing, but a rough surface can't reflect a sharp
// grazing rim, so the ceiling is clamped down by roughness.
fn fresnel_schlick_roughness(cos_theta: f32, f0: f32, roughness: f32) -> f32 {
    return f0 + (max(1.0 - roughness, f0) - f0) * pow(1.0 - cos_theta, 5.0);
}

// Analytic environment BRDF (Karis/Lazarov split-sum fit): the (scale, bias) to
// apply to F0 for prefiltered specular IBL, folding the Smith geometric term and
// the Fresnel integral into a closed form so no precomputed LUT is needed.
fn env_brdf_approx(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2<f32>(a004 * -1.04 + r.z, a004 * 1.04 + r.w);
}

// Khronos "PBR Neutral" tonemap: compresses HDR into [0,1] while preserving hue,
// desaturating only the brightest highlights — gentler than ACES on saturated
// colors. WGSL has no `if`-expression and parameters are immutable, so the work is
// done on a local `var`.
fn commerce_tonemap(color: vec3<f32>) -> vec3<f32> {
    const start_compression = 0.8 - 0.04;
    const desaturation = 0.15;

    var c = color;
    let x = min(c.r, min(c.g, c.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    c = c - offset;

    let peak = max(c.r, max(c.g, c.b));
    if peak < start_compression {
        return c;
    }

    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    c = c * (new_peak / peak);

    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(c, vec3<f32>(new_peak), g);
}

// ACES filmic tonemap (Narkowicz fit): compress HDR scene-linear radiance into the
// displayable [0,1] range with a filmic toe/shoulder, so the environment's bright
// lights roll off gracefully instead of clipping to flat white. Applied before the
// sRGB encode (DESIGN.md §6.3).
fn tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Light the un-lit linear `base` color by image-based lighting from the HDR
// environment (DESIGN.md §6.3): diffuse irradiance sampled (blurred) in the relief
// normal direction, plus a wet specular reflection that sharpens with `wet` so
// impasto ridges glint with the studio lights. `base` is already the composited
// paint-over-substrate color (the over happened in the caller's latent space).
fn finish(base: vec3<f32>, wet: f32, p: vec2f, dim: vec2<i32>) -> vec4<f32> {
    let n = surface_normal(p, dim);
    let exposure = m.shade.x;
    let diffuse_lod = m.shade.y;
    let wet_gloss = m.shade.z; // how smooth/glossy wet paint becomes, 0..1
    let max_lod = m.shade.w;   // highest mip (roughness = 1 samples here)

    // Orthographic view V = (0,0,1), so N·V is just the normal's z.
    let n_dot_v = clamp(n.z, 1e-4, 1.0);
    // Paint is a dielectric (F0 ≈ 0.04); wetter paint is smoother (lower roughness),
    // dry paint and the bare canvas are rough → matte.
    let f0 = 0.04;
    let roughness = clamp(1.0 - wet_gloss * clamp(wet, 0.0, 1.0), 0.045, 1.0);

    // Cook–Torrance split-sum image-based lighting:
    //  · diffuse — hemispherical irradiance (heavily-blurred env) in the normal dir;
    //  · specular — the env prefiltered in the reflection direction, at a mip chosen
    //    by roughness, weighted by the analytic env-BRDF (Smith G + Fresnel integral).
    // Energy is conserved: the diffuse keeps only what the specular Fresnel doesn't.
    let irradiance = sample_env(env_dir(n), diffuse_lod);
    let refl = 2.0 * n.z * n - vec3<f32>(0.0, 0.0, 1.0);
    let prefiltered = sample_env(env_dir(refl), roughness * max_lod);

    let brdf = env_brdf_approx(n_dot_v, roughness);
    let ks = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let kd = 1.0 - ks;

    let diffuse = kd * base * irradiance;
    let specular = prefiltered * (f0 * brdf.x + brdf.y);

    let lit = (diffuse + specular) * exposure;
    return vec4<f32>(linear_to_srgb(commerce_tonemap(lit)), 1.0);
}
