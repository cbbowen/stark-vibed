// Shared scaffolding for the media/lighting pass (§6.3), imported by
// the per-color-space media shaders. Each variant only differs in how it derives
// the un-lit linear *base color* from the composited channels; everything else —
// the bindings, the fullscreen vertex, the impasto normal-from-height, the
// directional lighting + paint gloss, and the composite-over-background + display
// encode — lives here so it is written once.

import package::lib::display::{encode_display};
import package::lib::noise::{dither2};
import package::lib::paint_common::{PAINT_EPS};

struct Media {
    // The background (substrate) in the latent color space. Three channels and no
    // fourth: the substrate is opaque paint under everything, and the alpha
    // `over_substrate` carries out is the literal 1 it writes there.
    bg: vec3<f32>,
    // The substrate color's **residual**, the half of it the space's three
    // channels cannot express (§6.7). Zero for every space but
    // Mixbox — and a black substrate is exactly the case that made this channel
    // necessary, since Mixbox's pigment polynomial renders black as `#383838`.
    //
    // Unconditional, like `comp_resid` below and for the same two reasons: a uniform
    // is mirrored into Rust from the unlinked source (§6.10), which cannot evaluate a
    // feature, and this pass has one bind group layout across both spaces.
    //
    // `@size(16)` holds open the lane this triple leaves spare: `exposure` follows at a
    // 4-byte alignment, so the member's own stride is the only thing that can say it.
    // `bg` above needs none — this member's own 16 places it (§6.10).
    @size(16) bg_resid: vec3<f32>,
    // The reciprocal of the irradiance a *flat* canvas receives (§6.3).
    exposure: f32,
    // The mip the diffuse irradiance is sampled from — blurred to a hemispherical
    // average.
    diffuse_lod: f32,
    // How smooth and glossy the paint film is, 0..1.
    gloss: f32,
    // The relief slope. A shading term with no light *direction* to hang it off: the
    // pass is image-based lit (§6.5), so the environment carries the direction and
    // this uniform carries only what shading multiplies by.
    height_strength: f32,
    // The canvas px at screen pixel 0.
    canvas_origin: vec2<f32>,
    // The screen→canvas map's length *scale* — rotation and mirroring leave lengths
    // alone — which is what `height_strength` wants, and what it would be wrong to
    // take from `view_m` below.
    canvas_per_px: f32,
    // The reciprocal of the canvas px one period of the substrate field spans.
    inv_tile: f32,
    substrate_strength: f32,
    // 0 or 1 — see `is_transparent`.
    transparent: f32,
    // One code of the target's encoding, or 0 (`media::dither_step`, §6.5).
    dither_step: f32,
    // The screen→canvas linear map, column-major: `.xy` is the first column and
    // `.zw` the second. What carries a fragment into canvas space when the view is
    // turned or mirrored (§18.1.2).
    view_m: vec4<f32>,
    // The transfer the target expects — `lib/display.wesl`'s selector, as a float
    // (§6.5).
    transfer: f32,
    // The headroom the tonemap compresses into — times SDR white; 1 on any SDR
    // surface and on every export (§6.5).
    headroom: f32,
}

@group(0) @binding(0) var<uniform> m: Media;
@group(0) @binding(1) var comp_color: texture_2d<f32>;
@group(0) @binding(2) var comp_aux: texture_2d<f32>; // paint height (r), the only aux channel
@group(0) @binding(3) var substrate: texture_2d<f32>; // tileable canvas bump (height)
@group(0) @binding(4) var substrate_samp: sampler;
// The HDR lighting environment (equirectangular, mipped) for image-based lighting
// (§6.3): diffuse irradiance from a high mip in the normal direction,
// specular from a gloss-selected mip in the reflection direction.
@group(0) @binding(5) var env: texture_2d<f32>;
@group(0) @binding(6) var env_samp: sampler;
// **The composited residual is `media.wesl`'s own `@group(1)`**, declared there: it
// (§6.7). It is not here because Oklab has no residual to bind — a
// declaration in this shared module would put the texture in *both* spaces' layouts
// and leave the colorimetric one binding a placeholder it never samples.

const PI: f32 = 3.14159265359;

// This pass's dither identity: `resolve.wesl` runs the same law over the same
// pixel grid when the view is supersampled, and the two stores must not share a
// pattern. Any two distinct values do.
const DITHER_SEED: u32 = 0u;

// Absolute canvas substrate height at screen pixel `p` (sampled in canvas space, so
// the substrate is fixed to the canvas). `Flat` is height 0, so it adds no relief.
fn substrate_height_at(p: vec2<i32>) -> f32 {
    // The substrate belongs to the canvas, so it pans, zooms, turns and mirrors with it.
    let inv = mat2x2<f32>(m.view_m.xy, m.view_m.zw);
    // The fragment's *centre*, like every other sample: at the corner the substrate
    // sits half a texel up-and-left of the pixel it shades, which is invisible on
    // its own and stops the view's mirror being an exact reflection.
    let canvas = m.canvas_origin + inv * (vec2<f32>(p) + 0.5);
    let hs = textureSampleLevel(substrate, substrate_samp, canvas * m.inv_tile, 0.0).r;
    return m.substrate_strength * hs;
}

fn paint_height_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    let c = clamp(p, vec2<i32>(0), dim - vec2<i32>(1));
    return textureLoad(comp_aux, c, 0).x;
}

// The paint's visible alpha at `p`. Pass A weighs each layer by its own slab
// visible alpha (composite.wesl), so the accumulated color alpha *is* the stack's
// coverage — read, not re-derived; re-deriving it from the stack totals is what
// let one layer cover by its opacity alone (the §6.3 defect). 0 on bare canvas,
// → 1 for thick opaque paint.
//
// What is still applied here is the substrate sink: paint below the canvas substrate
// height doesn't show, so the coverage of the mass that filled the substrate is
// discounted by the height fraction that clears it. `1 − (1−w)^(t/h)` — for a
// single layer this is exactly `1 − exp(−K·op·t)`, the pre-coverage law, because
// the slab is multiplicative in mass; for a stack it charges the sink to the
// bottom of the pile, which is where the paint that filled the substrate is.
fn paint_vis_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    let c = clamp(p, vec2<i32>(0), dim - vec2<i32>(1));
    let w = textureLoad(comp_color, c, 0).a;
    let h = textureLoad(comp_aux, c, 0).x;
    let t = max(h - substrate_height_at(p), 0.0);
    // Both guards return the limit the power law approaches; the second also
    // dodges `pow(0, 0)` (opaque paint entirely below the substrate), which WGSL
    // leaves undefined.
    if h <= PAINT_EPS {
        return 0.0;
    }
    if t <= 0.0 {
        return 0.0;
    }
    return 1.0 - pow(max(1.0 - w, 0.0), t / h);
}

// The relief height for impasto normals: the canvas substrate plus the paint's
// thickness, gated by how visible the paint is so a near-transparent tail doesn't
// emboss a ghost ridge around the stroke (§6.4).
fn height_at(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    return max(substrate_height_at(p), paint_height_at(p, dim));
}

fn substrate_color() -> vec3<f32> {
    return m.bg;
}

// Whether this is a transparent export: the substrate is skipped entirely and the
// paint's own visible alpha becomes the image's alpha (§15.6). A real
// branch rather than an alpha tweak — compositing over the substrate and *then*
// setting alpha would leave every bare-canvas texel carrying substrate color at
// zero alpha, which fringes the moment the PNG is composited over anything else.
fn is_transparent() -> bool {
    return m.transparent > 0.5;
}

// The substrate a fragment composites over, and the alpha it ends up with. On a
// transparent export the paint keeps its own color (no substrate mixed in) and
// carries `vis` out as alpha; otherwise it is opaque over the substrate.
fn over_substrate(paint: vec3<f32>, vis: f32) -> vec4<f32> {
    if is_transparent() {
        return vec4<f32>(paint, vis);
    }
    return vec4<f32>(substrate_color() * (1.0 - vis) + paint * vis, 1.0);
}

// Surface normal from the height gradient (impasto relief). The gradient is taken
// over screen pixels but divided by the canvas px they span (`canvas_per_px`), so the
// slope is per *canvas* px — the relief (and its gloss) is zoom-invariant rather
// than steepening as you zoom out.
fn relief_normal(p: vec2<f32>, dim: vec2<i32>) -> vec3<f32> {
    let i = vec2<i32>(floor(p));
    let strength = m.height_strength / m.canvas_per_px;
    let hl = height_at(i + vec2<i32>(-1, 0), dim);
    let hr = height_at(i + vec2<i32>(1, 0), dim);
    let hu = height_at(i + vec2<i32>(0, -1), dim);
    let hd = height_at(i + vec2<i32>(0, 1), dim);
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
    return vec3<f32>(n.x, -n.y, -n.z);
}

fn sample_env(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    return textureSampleLevel(env, env_samp, equirect_uv(dir), lod).rgb;
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

// Khronos "PBR Neutral" tonemap, re-tuned so the pass is a *reference* rather than a
// look: it compresses HDR into [0,1] preserving hue, desaturating only the brightest
// highlights. Two things differ from the stock Khronos curve, both so that a color
// already inside display range survives unchanged (§6.3):
//
//   · `black_point` is the sheen the BRDF actually added at this fragment, passed in
//     rather than assumed to be F0 = 0.04. Khronos hardcodes 0.04 because their
//     pipeline's specular floor *is* F0; ours is the split-sum's `f0·brdf.x + brdf.y`,
//     roughly 0.016 for dry paint, and subtracting 0.04 of it crushed every midtone.
//   · The highlight knee sits at 1.0 instead of 0.8, so nothing representable is
//     reshaped on its way to the display. Only what genuinely overflows — the
//     environment's lights glinting off glossy impasto — is rolled off.
//
// WGSL has no `if`-expression and parameters are immutable, so the work is done on a
// local `var`.
fn commerce_tonemap(color: vec3<f32>, black_point: f32) -> vec3<f32> {
    const desaturation = 0.15;
    // In the shifted space the knee moves down with the offset, exactly as Khronos's
    // `0.8 - 0.04` does — theirs is this same expression with both constants fixed.
    let start_compression = 1.0 - black_point;

    var c = color;
    // Lift the sheen off the darkest channel, so paint over a black substrate reads
    // black — by at most the sheen, and never by a negative channel: a wide-gamut
    // color has one, and it is chroma, not light to lift (§6.5).
    c = c - clamp(min(c.r, min(c.g, c.b)), 0.0, black_point);

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

// The same curve for a display that can show `headroom` times SDR white (§6.5): run
// in that display's units and scaled back, so the knee sits at `headroom` and only
// what overflows *this* surface is rolled off. At `headroom = 1` it is
// `commerce_tonemap` exactly, which is what keeps every golden where it was.
fn tonemap_to(color: vec3<f32>, black_point: f32, headroom: f32) -> vec3<f32> {
    return headroom * commerce_tonemap(color / headroom, black_point / headroom);
}

// Light the un-lit linear `base` color by image-based lighting from the HDR
// environment (§6.3): diffuse irradiance sampled (blurred) in the relief
// normal direction, plus a specular reflection that sharpens with `paint` so
// impasto ridges glint with the studio lights. `base` is already the composited
// paint-over-substrate color (the over happened in the caller's latent space), and
// `paint` is the visible alpha that composite used — the paint film's own share of
// the fragment. Specularity is a **uniform property of the paint**, not of a stored
// channel: paint is glossy everywhere it is, and the substrate behind it never is.
fn finish(base: vec3<f32>, paint: f32, alpha: f32, p: vec2<f32>, dim: vec2<i32>) -> vec4<f32> {
    let n = relief_normal(p, dim);
    // Orthographic view V = (0,0,1), so N·V is just the normal's z.
    let n_dot_v = clamp(n.z, 1e-4, 1.0);
    // Paint is a dielectric (F0 ≈ 0.04) and is smoother than the canvas it sits on;
    // the bare canvas is rough → matte. Ramping by the paint's *visible alpha* is
    // what keeps the gloss on the film rather than the substrate: a thin glaze reads
    // almost as matte as the substrate it barely covers, thick paint carries the
    // full sheen, and bare canvas carries none.
    let f0 = 0.04;
    let roughness = clamp(1.0 - m.gloss * clamp(paint, 0.0, 1.0), 0.045, 1.0);

    // Cook–Torrance split-sum image-based lighting:
    //  · diffuse — hemispherical irradiance (heavily-blurred env) in the normal dir;
    //  · specular — the env prefiltered in the reflection direction, at a mip chosen
    //    by roughness, weighted by the analytic env-BRDF (Smith G + Fresnel integral).
    // Energy is conserved: the diffuse keeps only what the specular doesn't.
    //
    // Both ends of the roughness ramp land on `m.diffuse_lod`, not on the 1×1 top of the
    // chain: a fully rough lobe *is* the hemispherical average, so it has to read the
    // same blur the diffuse does. Sampling a blurrier level than the diffuse made the
    // dielectric floor brighter than the sheen `finish` subtracts back off, which
    // showed up as black paint that would not go black.
    let irradiance = sample_env(env_dir(n), m.diffuse_lod);
    let refl = 2.0 * n.z * n - vec3<f32>(0.0, 0.0, 1.0);
    let prefiltered = sample_env(env_dir(refl), roughness * m.diffuse_lod);

    let brdf = env_brdf_approx(n_dot_v, roughness);
    // How much energy the specular lobe actually carries off. The diffuse keeps the
    // rest — `1 - spec_energy`, not `1 - fresnel`: the split-sum's `brdf` already
    // integrates Fresnel, so taking a second Schlick term out of the diffuse was
    // double-counting it and quietly losing ~2.4% of every color.
    let spec_energy = f0 * brdf.x + brdf.y;
    let kd = 1.0 - spec_energy;

    let diffuse = kd * base * irradiance;
    let specular = prefiltered * spec_energy;

    let lit = (diffuse + specular) * m.exposure;
    // The sheen to lift back off: what the specular contributes where the reflection
    // sees nothing brighter than the ambient the diffuse already sampled. A glint
    // reflecting an actual light sits far above this, so gloss survives the subtraction
    // — only the uniform dielectric floor over the whole canvas comes off.
    let sheen = spec_energy * dot(irradiance, vec3<f32>(0.2126, 0.7152, 0.0722)) * m.exposure;
    // Straight (un-premultiplied) alpha: PNG stores straight, and the color here is
    // the paint's own, not paint-times-coverage. Compressed into the display's
    // headroom and encoded in the transfer this surface reads (§6.5).
    let encoded = encode_display(u32(m.transfer), tonemap_to(lit, sheen, m.headroom));
    // Quantization dither (§6.5): the store below is the pipeline's one walk from
    // f32 down to the target's 8-bit codes, so its rounding is made stochastic —
    // ±half a code, uniform, `lib::noise`'s law — and a ramp too gentle for the
    // codes renders as mixture instead of bands. One lane shared across RGB, so
    // the noise is luma-only (chroma noise is the visible kind at this amplitude);
    // the second covers alpha, whose soft edges band the same way on a cut-out
    // export (§15.6). `dither_step` is 0 for a target deep enough not to band,
    // which turns all of this off.
    let d = dither2(vec2u(p), DITHER_SEED) * m.dither_step;
    return vec4<f32>(encoded + d.x, alpha + d.y);
}
