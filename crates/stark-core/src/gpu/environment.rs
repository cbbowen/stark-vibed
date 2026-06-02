//! HDR **environment maps** for image-based lighting (DESIGN.md §6.3).
//!
//! A studio (or any) HDR is decoded from its Radiance RGBE file into a linear-RGB
//! equirectangular image, then used to light the painting: the media pass samples
//! it in the surface-normal direction (diffuse irradiance) and the view-reflection
//! direction (wet specular), so impasto relief catches the environment's lights.
//!
//! Like [`super::surface::Surface`], the bytes come from the frontend at runtime
//! (the engine embeds none); decoding and prefiltering happen here, on the CPU,
//! once per environment.

use serde::{Deserialize, Serialize};

use crate::gpu::context::GpuContext;

/// Which environment a document is lit by. A view setting (not historized): it
/// changes how the canvas *looks*, never the stored pixels. The set is open —
/// future uploaded HDRs slot in here.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EnvironmentId {
    /// A procedural, no-HDR default: a soft overhead key over ambient fill, so the
    /// canvas is still lit (and relief still reads) before any HDR is registered.
    #[default]
    Studio,
    /// The bundled `ferndale_studio` HDR.
    Ferndale,
}

/// Decode a Radiance RGBE (`#?RADIANCE`, `FORMAT=32-bit_rle_rgbe`) file into a
/// linear-RGB equirectangular image (row-major, top row first). Returns
/// `(pixels, width, height)`.
///
/// Supports the new-style per-scanline RLE (the common case for ≥8px-wide images)
/// and falls back to flat RGBE quads otherwise. Errors on a malformed header.
pub fn decode_hdr(bytes: &[u8]) -> Result<(Vec<[f32; 3]>, u32, u32), String> {
    let mut pos = 0usize;

    // --- Header: text lines until a blank line, then the resolution line. ---
    let line = |pos: &mut usize| -> String {
        let start = *pos;
        while *pos < bytes.len() && bytes[*pos] != b'\n' {
            *pos += 1;
        }
        let s = String::from_utf8_lossy(&bytes[start..*pos]).into_owned();
        *pos += 1; // skip '\n'
        s
    };

    let magic = line(&mut pos);
    if !magic.starts_with("#?") {
        return Err(format!("hdr: bad magic {magic:?}"));
    }
    // Consume header lines until the blank separator.
    loop {
        if pos >= bytes.len() {
            return Err("hdr: unexpected EOF in header".into());
        }
        let l = line(&mut pos);
        if l.is_empty() {
            break;
        }
    }

    // Resolution line, e.g. "-Y 512 +X 1024". We only support the standard
    // top-down, left-right orientation (`-Y h +X w`), which HDRIs use.
    let res = line(&mut pos);
    let parts: Vec<&str> = res.split_whitespace().collect();
    if parts.len() != 4 || parts[0] != "-Y" || parts[2] != "+X" {
        return Err(format!("hdr: unsupported resolution line {res:?}"));
    }
    let h: u32 = parts[1].parse().map_err(|_| "hdr: bad height")?;
    let w: u32 = parts[3].parse().map_err(|_| "hdr: bad width")?;
    let (wu, hu) = (w as usize, h as usize);

    let mut out = vec![[0.0f32; 3]; wu * hu];
    let mut scan = vec![[0u8; 4]; wu]; // one scanline of RGBE
    for y in 0..hu {
        read_scanline(bytes, &mut pos, &mut scan, wu)?;
        let row = &mut out[y * wu..(y + 1) * wu];
        for (px, rgbe) in row.iter_mut().zip(scan.iter()) {
            *px = rgbe_to_linear(*rgbe);
        }
    }
    Ok((out, w, h))
}

/// Read one scanline of `w` RGBE pixels into `scan`, advancing `pos`. Handles the
/// new-style RLE header (`0x02 0x02 hi lo`) per channel, else flat/old quads.
fn read_scanline(bytes: &[u8], pos: &mut usize, scan: &mut [[u8; 4]], w: usize) -> Result<(), String> {
    // New-style RLE is only used for widths in [8, 0x7fff] and is flagged by a
    // leading 0x02 0x02 with the width in the next two bytes.
    let new_rle = w >= 8
        && w < 0x8000
        && *pos + 4 <= bytes.len()
        && bytes[*pos] == 2
        && bytes[*pos + 1] == 2
        && ((bytes[*pos + 2] as usize) << 8 | bytes[*pos + 3] as usize) == w;

    if !new_rle {
        // Flat RGBE quads (old-style RLE — repeats flagged by R=G=B=1 — is rare
        // for modern HDRIs; we read straight quads, which covers the non-RLE case).
        for px in scan.iter_mut().take(w) {
            if *pos + 4 > bytes.len() {
                return Err("hdr: EOF in flat scanline".into());
            }
            px.copy_from_slice(&bytes[*pos..*pos + 4]);
            *pos += 4;
        }
        return Ok(());
    }
    *pos += 4; // consume the RLE scanline header

    // Four channel planes (R, G, B, E), each run-length encoded across the row.
    for ch in 0..4 {
        let mut x = 0usize;
        while x < w {
            if *pos >= bytes.len() {
                return Err("hdr: EOF in RLE channel".into());
            }
            let count = bytes[*pos] as usize;
            *pos += 1;
            if count > 128 {
                // A run: (count - 128) copies of the next byte.
                let n = count - 128;
                if *pos >= bytes.len() || x + n > w {
                    return Err("hdr: bad RLE run".into());
                }
                let v = bytes[*pos];
                *pos += 1;
                for i in 0..n {
                    scan[x + i][ch] = v;
                }
                x += n;
            } else {
                // A literal: `count` raw bytes.
                if *pos + count > bytes.len() || x + count > w {
                    return Err("hdr: bad RLE literal".into());
                }
                for i in 0..count {
                    scan[x + i][ch] = bytes[*pos + i];
                }
                *pos += count;
                x += count;
            }
        }
    }
    Ok(())
}

/// RGBE → linear RGB. The shared exponent `e` scales the mantissa by `2^(e-136)`
/// (128 bias + 8 mantissa bits); `e == 0` is exact black. The `+0.5` centers each
/// mantissa in its quantization bucket.
fn rgbe_to_linear(rgbe: [u8; 4]) -> [f32; 3] {
    let e = rgbe[3];
    if e == 0 {
        return [0.0; 3];
    }
    let f = 2.0f32.powi(e as i32 - 136);
    [
        (rgbe[0] as f32 + 0.5) * f,
        (rgbe[1] as f32 + 0.5) * f,
        (rgbe[2] as f32 + 0.5) * f,
    ]
}

/// A decoded, prefiltered environment ready for image-based lighting: an
/// equirectangular `Rgba16Float` texture with a full mip chain (each level a box
/// downsample of the last). The media pass samples a high mip in the surface-normal
/// direction for diffuse irradiance and a gloss-selected mip in the reflection
/// direction for wet specular (DESIGN.md §6.3). Cloning is cheap (Arc-backed wgpu
/// handles), so it can live alongside the [`super::surface::Surface`].
#[derive(Clone)]
pub struct Environment {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// Mip levels, so the media shader can pick the diffuse (very blurred) LOD.
    pub mip_count: u32,
    /// Mean luminance of the environment. The media pass divides exposure by this
    /// so any environment (procedural or HDR) is normalized to a neutral level —
    /// a flat surface reads ~its albedo regardless of how bright the HDR is.
    pub mean_luminance: f32,
}

impl Environment {
    /// The procedural default (no HDR): a soft overhead key over ambient fill, so
    /// the canvas is lit and impasto/weave relief still reads before — or without —
    /// any HDR being registered.
    pub fn studio(ctx: &GpuContext) -> Self {
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
                px[(y * W + x) as usize] = [l, l * 0.99, l * 0.95]; // slightly warm
            }
        }
        Self::from_equirect(ctx, &px, W, H)
    }

    /// Decode a Radiance HDR and prefilter it for lighting.
    pub fn load(ctx: &GpuContext, hdr_bytes: &[u8]) -> Self {
        let (px, w, h) = decode_hdr(hdr_bytes).expect("environment: decode HDR");
        Self::from_equirect(ctx, &px, w, h)
    }

    /// Upload a linear-RGB equirect image as a mipped `Rgba16Float` texture, the
    /// mip chain box-downsampled on the CPU (it is built once per environment).
    fn from_equirect(ctx: &GpuContext, base: &[[f32; 3]], w: u32, h: u32) -> Self {
        // Mean luminance (Rec.709) for exposure normalization; guard against 0.
        let mean_luminance = (base
            .iter()
            .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
            .sum::<f32>()
            / base.len() as f32)
            .max(1e-3);

        let mip_count = 32 - (w.max(h)).leading_zeros(); // floor(log2(max))+1
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark environment"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let (mut level, mut lw, mut lh) = (base.to_vec(), w, h);
        for mip in 0..mip_count {
            write_mip(ctx, &texture, mip, &level, lw, lh);
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
        Self { view, sampler, mip_count, mean_luminance }
    }
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
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
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

/// Encode a non-negative `f32` to IEEE-754 half-precision bits (round-to-nearest-
/// even). Environment radiance is ≥ 0, so the sign bit is always clear; values are
/// clamped to the half-float max (no infinities), and subnormals flush to zero.
fn f32_to_f16(x: f32) -> u16 {
    let x = x.clamp(0.0, 65504.0);
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return 0; // zero / subnormal → 0 (negligible for radiance)
    }
    let mant = bits & 0x7f_ffff;
    let half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let round = u16::from(rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1));
    ((exp as u16) << 10 | half_mant) + round
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_bundled_studio_hdr() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../stark-ui/assets/environment/ferndale_studio_11_1k.hdr"
        ))
        .expect("read studio HDR");
        let (pixels, w, h) = decode_hdr(&bytes).expect("decode HDR");
        assert_eq!((w, h), (1024, 512));
        assert_eq!(pixels.len(), (w * h) as usize);
        // All finite and non-negative; a studio HDR has some bright (>1) values.
        assert!(pixels.iter().flatten().all(|c| c.is_finite() && *c >= 0.0));
        let max = pixels.iter().flatten().cloned().fold(0.0f32, f32::max);
        assert!(max > 1.0, "studio HDR should contain values >1 (got max {max})");
    }

    #[test]
    fn f16_encoding_roundtrips() {
        // Decode our f16 bits back to f32 and check a few representative radiance
        // values land within half-float precision.
        let half_to_f32 = |h: u16| -> f32 {
            let exp = ((h >> 10) & 0x1f) as i32;
            let mant = (h & 0x3ff) as f32;
            if exp == 0 {
                return 0.0;
            }
            (1.0 + mant / 1024.0) * 2.0f32.powi(exp - 15)
        };
        for &v in &[0.0f32, 0.25, 1.0, 2.5, 18.0, 500.0] {
            let back = half_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.max(1.0) * 0.001 + 1e-3, "f16({v}) -> {back}");
        }
        assert_eq!(f32_to_f16(-1.0), 0); // negatives clamp to 0
    }
}
