//! HDR **environment maps** for image-based lighting (§6.3).
//!
//! A studio (or any) HDR is decoded from its Radiance RGBE file into a linear-RGB
//! equirectangular image, then used to light the painting: the media pass samples
//! it in the substrate-normal direction (diffuse irradiance) and the view-reflection
//! direction (the paint's specular), so impasto relief catches the environment's lights.
//!
//! Like [`super::substrate::SubstrateMap`], the bytes come from the frontend at runtime
//! (the engine embeds none); decoding and prefiltering happen here, on the CPU,
//! once per environment.

use serde::{Deserialize, Serialize};

use crate::gpu::context::GpuContext;
use crate::gpu::half::f32_to_f16;

mod hdr;

use hdr::decode_hdr;

/// Which environment a document is lit by. A view setting (not historized): it
/// changes how the canvas *looks*, never the stored pixels. The set is open —
/// future uploaded HDRs slot in here.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EnvironmentId {
    /// The procedural **reference** light: achromatic, generated on the fly, no HDR
    /// file. A soft overhead key over an ambient dome — enough directionality that
    /// impasto relief still reads, but no color cast, so paint reads as its own
    /// hue. This is what you switch to when you want to judge color rather than
    /// enjoy the room; it is also the fallback before any HDR's bytes arrive.
    #[default]
    Neutral,
    /// The bundled `ferndale_studio` HDR.
    Ferndale,
    BloemHill,
    KloofendalOvercast,
    QwantaniDusk,
}

/// A decoded, prefiltered environment ready for image-based lighting: an
/// equirectangular `Rgba16Float` texture with a full mip chain (each level a box
/// downsample of the last). The media pass samples a high mip in the substrate-normal
/// direction for diffuse irradiance and a gloss-selected mip in the reflection
/// direction for the paint's specular (§6.3). Cloning is cheap (Arc-backed wgpu
/// handles), so it can live alongside the [`super::substrate::SubstrateMap`].
#[derive(Clone)]
pub struct Environment {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// Mip levels, so the media shader can pick the diffuse (very blurred) LOD.
    pub mip_count: u32,
    /// Which mip the media pass reads for diffuse irradiance. Computed here, beside
    /// [`Self::flat_irradiance`], because the two must agree: the normalization is
    /// only exact if the CPU samples the level the shader will.
    pub diffuse_lod: u32,
    /// The irradiance a **flat** canvas receives — the diffuse mip sampled in the one
    /// direction an untilted normal faces (dead ahead, the equirect's centre), which
    /// is exactly what `finish` in `media_common.wesl` looks up when the relief is
    /// flat. **The media pass shades by its reciprocal**, which is what makes "a flat
    /// canvas reads its own albedo" true in *any* environment, procedural or HDR —
    /// and is the whole of the normalization now that no light carries an exposure of
    /// its own (§6.3).
    ///
    /// Not the whole-image mean luminance, which only approximates it: a mean over
    /// equirect texels over-weights the poles and includes light no front-facing canvas
    /// ever sees, leaving a flat patch ~13% dark under the procedural environment.
    pub flat_irradiance: f32,
}

impl Environment {
    /// The procedural reference light (no HDR): a soft overhead key over an ambient
    /// dome, generated here rather than shipped as a file. Deliberately achromatic —
    /// every texel is grey — so it lights the relief without tinting the paint, which
    /// is what makes it usable as a color reference next to [`Self::load`]ed HDRs.
    pub fn neutral(ctx: &GpuContext) -> Self {
        let (px, w, h) = neutral_equirect();
        Self::from_equirect(ctx, &px, w, h)
    }

    /// Decode a Radiance HDR and prefilter it for lighting — or `Err` with what the
    /// decoder made of the bytes.
    ///
    /// **Fallible because the bytes come from outside**: an environment is fetched at
    /// runtime and handed straight in, so a truncated download or a file that is not
    /// an `.hdr` at all reaches here — the class §5 exists to remove, and one
    /// [`hdr`]'s own header says this path has to be
    /// defensive about. [`Resource::decode`] is the door a caller learns at;
    /// [`Resource::build`] is where a byte string that got past it anyway degrades to
    /// the procedural light rather than killing the renderer.
    ///
    /// [`Resource::decode`]: crate::gpu::registry::Resource::decode
    ///
    /// [`Engine::register_environment`]: crate::Engine::register_environment
    /// [`Resource::build`]: crate::gpu::registry::Resource::build
    pub fn load(ctx: &GpuContext, hdr_bytes: &[u8]) -> Result<Self, String> {
        let (px, w, h) = decode_hdr(hdr_bytes)?;
        Ok(Self::from_equirect(ctx, &px, w, h))
    }

    /// Upload a linear-RGB equirect image as a mipped `Rgba16Float` texture, the
    /// mip chain box-downsampled on the CPU (it is built once per environment).
    fn from_equirect(ctx: &GpuContext, base: &[[f32; 3]], w: u32, h: u32) -> Self {
        let mip_count = 32 - (w.max(h)).leading_zeros(); // floor(log2(max))+1
        let diffuse_lod = diffuse_lod(mip_count);
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark environment"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Build the chain, capturing the diffuse level on the way past: the exposure
        // normalization has to read the same texels the shader will.
        let (mut level, mut lw, mut lh) = (base.to_vec(), w, h);
        let mut flat_irradiance = 1.0f32;
        for mip in 0..mip_count {
            write_mip(ctx, &texture, mip, &level, lw, lh);
            if mip == diffuse_lod {
                flat_irradiance = luminance(sample_center(&level, lw, lh)).max(1e-3);
            }
            if mip + 1 < mip_count {
                let (next, nw, nh) = downsample(&level, lw, lh);
                level = next;
                lw = nw;
                lh = nh;
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark environment sampler"),
            // Longitude wraps; latitude clamps. Trilinear so the LOD blends.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            view,
            sampler,
            mip_count,
            diffuse_lod,
            flat_irradiance,
        }
    }
}

/// Which mip stands in for diffuse irradiance: blurred enough to pass for a cosine
/// convolution, but not the 1×1 average — some directionality has to survive or
/// relief stops reading at all.
fn diffuse_lod(mip_count: u32) -> u32 {
    mip_count.saturating_sub(3)
}

/// Rec.709 luminance.
fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Bilinearly sample a mip level at its centre, `uv = (0.5, 0.5)` — the equirect
/// direction a flat, front-facing canvas looks in. Mirrors the GPU sampler: half-texel
/// offsets, `Repeat` across longitude, `ClampToEdge` across latitude.
fn sample_center(px: &[[f32; 3]], w: u32, h: u32) -> [f32; 3] {
    let (fw, fh) = (w as f32, h as f32);
    let (x, y) = (0.5 * fw - 0.5, 0.5 * fh - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let at = |ix: i32, iy: i32| -> [f32; 3] {
        let u = ix.rem_euclid(w as i32) as u32; // longitude wraps
        let v = iy.clamp(0, h as i32 - 1) as u32; // latitude clamps
        px[(v * w + u) as usize]
    };
    let (x0, y0) = (x0 as i32, y0 as i32);
    let mut out = [0.0f32; 3];
    for (dx, dy, weight) in [
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ] {
        let c = at(x0 + dx, y0 + dy);
        for i in 0..3 {
            out[i] += weight * c[i];
        }
    }
    out
}

/// Generate the `Neutral` reference environment as a linear-RGB equirect image,
/// returning `(pixels, width, height)` in the same form [`decode_hdr`] produces — so
/// it feeds the identical prefilter and the identical shader, and "neutral" costs a
/// procedure rather than a second lighting path or a checked-in `.hdr`.
///
/// Every texel is grey; the only variation is directional, so relief still catches
/// the light while paint keeps its own hue.
fn neutral_equirect() -> (Vec<[f32; 3]>, u32, u32) {
    const W: u32 = 256;
    const H: u32 = 128;
    // A front-overhead key direction (y-up), normalized.
    let kd = {
        let k = [0.28f32, 0.9, 0.34];
        let n = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
        [k[0] / n, k[1] / n, k[2] / n]
    };
    let mut px = vec![[0.0f32; 3]; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let dir = equirect_dir((x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / H as f32);
            let up = dir.1.max(0.0);
            // Soft ambient fill (smooth → dominates the blurred diffuse tone).
            let ambient = 0.5 + 0.3 * up;
            // A gentle, broad overhead key — soft enough not to clip flats or
            // throw a harsh white rim, but enough that relief still catches it.
            let cosang = dir.0 * kd[0] + dir.1 * kd[1] + dir.2 * kd[2];
            let softbox = smoothstep(0.78, 0.98, cosang) * 1.6;
            let l = ambient + softbox;
            px[(y * W + x) as usize] = [l, l, l]; // achromatic: no cast on the paint
        }
    }
    (px, W, H)
}

/// Upload one mip level (linear RGB → `Rgba16Float`, alpha = 1).
fn write_mip(ctx: &GpuContext, texture: &wgpu::Texture, mip: u32, px: &[[f32; 3]], w: u32, h: u32) {
    let mut data = Vec::with_capacity(px.len() * 4);
    for c in px {
        data.extend_from_slice(&f32_to_f16(c[0]).to_le_bytes());
        data.extend_from_slice(&f32_to_f16(c[1]).to_le_bytes());
        data.extend_from_slice(&f32_to_f16(c[2]).to_le_bytes());
        data.extend_from_slice(&f32_to_f16(1.0).to_le_bytes());
    }
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: mip,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 8), // 4 channels × 2 bytes
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

/// Box-downsample an equirect image by 2× (each axis ≥ 1). Averaging adjacent
/// texels is a crude prefilter, but with the orthographic view's near-`+Z` normals
/// it reads well, and the diffuse term only needs the heavily-blurred high mips.
fn downsample(src: &[[f32; 3]], w: u32, h: u32) -> (Vec<[f32; 3]>, u32, u32) {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let mut out = vec![[0.0f32; 3]; (nw * nh) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0.0f32; 3];
            let mut n = 0.0f32;
            for dy in 0..(h / nh).max(1) {
                for dx in 0..(w / nw).max(1) {
                    let sx = (x * (w / nw).max(1) + dx).min(w - 1);
                    let sy = (y * (h / nh).max(1) + dy).min(h - 1);
                    let s = src[(sy * w + sx) as usize];
                    acc[0] += s[0];
                    acc[1] += s[1];
                    acc[2] += s[2];
                    n += 1.0;
                }
            }
            out[(y * nw + x) as usize] = [acc[0] / n, acc[1] / n, acc[2] / n];
        }
    }
    (out, nw, nh)
}

/// Equirect texel UV → world direction (y-up). The forward map (in the shader) is
/// `u = 0.5 + atan2(x,-z)/2π`, `v = 0.5 - asin(y)/π`; this is its inverse.
fn equirect_dir(u: f32, v: f32) -> (f32, f32, f32) {
    let theta = (u - 0.5) * std::f32::consts::TAU;
    let phi = (0.5 - v) * std::f32::consts::PI; // +π/2 at top (+Y)
    let cp = phi.cos();
    (cp * theta.sin(), phi.sin(), -cp * theta.cos())
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl crate::gpu::registry::Resource for EnvironmentId {
    type Gpu = Environment;

    /// A light is built from its own bytes and nothing else, so the two keys the
    /// registry distinguishes are one here (`gpu::registry`).
    type Content = Self;

    fn content(self) -> Self {
        self
    }

    /// `Neutral` is procedural — a soft overhead key over an ambient dome — so the
    /// canvas is lit before any HDR arrives (§6.3).
    fn is_builtin(self) -> bool {
        self == EnvironmentId::Neutral
    }

    /// **Nothing**, deliberately, where the substrate keeps its decoded height field.
    /// An environment is built exactly once per registration — its `Gpu` is the mip
    /// chain, and no second thing is derived from the same bytes — so keeping the
    /// decode would be several megabytes of float image held for a build that already
    /// happened. The decode still runs at the door, in
    /// [`decode`](crate::gpu::registry::Resource::decode); what is not kept is its
    /// result.
    type Decoded = ();

    fn decode(bytes: &[u8]) -> std::result::Result<(), stark_model::DocError> {
        decode_hdr(bytes)
            .map(|_| ())
            .map_err(|e| stark_model::DocError::Asset(format!("lighting environment: {e}")))
    }

    /// Every light is normalized by its own `flat_irradiance` and shown at no exposure
    /// beyond that (§6.3), so building one is a decode and a mip chain and nothing else.
    fn build(
        self,
        gpu: &GpuContext,
        registered: Option<crate::gpu::registry::Registered<'_, Self>>,
    ) -> Environment {
        match registered {
            // Bytes that will not decode fall back to the procedural light, exactly as
            // bytes that never arrived do — the canvas stays lit and says so, where a
            // panic here would take the document with it. `Registry::register` refuses
            // them at the door; reaching this arm means bytes were registered by some
            // other path.
            Some(r) if !self.is_builtin() => Environment::load(gpu, r.bytes).unwrap_or_else(|e| {
                tracing::warn!(
                    ?self,
                    "environment will not decode, lighting neutrally: {e}"
                );
                Environment::neutral(gpu)
            }),
            _ => Environment::neutral(gpu),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of `Neutral` is that it is a *reference*: it may shape the
    /// light, but it must not tint it. A color cast here would silently bias every
    /// judgement made against it.
    #[test]
    fn neutral_environment_is_achromatic_and_directional() {
        let (px, w, h) = neutral_equirect();
        assert_eq!(px.len(), (w * h) as usize);
        for c in &px {
            assert!(
                c[0] > 0.0 && c[0].is_finite(),
                "non-positive radiance {c:?}"
            );
            assert_eq!(
                [c[0], c[0]],
                [c[1], c[2]],
                "neutral must be grey, got {c:?}"
            );
        }
        // Still lit from somewhere: a uniform dome would flatten all relief away.
        let lum = |c: &[f32; 3]| c[0];
        let min = px.iter().map(lum).fold(f32::INFINITY, f32::min);
        let max = px.iter().map(lum).fold(0.0f32, f32::max);
        assert!(max > min * 1.5, "too flat to read relief: {min}..{max}");
    }
}
