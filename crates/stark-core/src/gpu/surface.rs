//! The physical canvas **surface** — a tileable height/bump map that affects both
//! paint deposition (tooth: dry strokes catch on the weave) and media shading
//! (the relief catches light), DESIGN.md §6.4.
//!
//! It is a single global, color-space-independent texture sampled in *canvas*
//! space (so the weave pans and zooms with the canvas), shared by the stroke
//! renderer (deposition) and the compositor (shading). Cloning is cheap — wgpu
//! views/samplers are reference-counted.

use serde::{Deserialize, Serialize};

use crate::gpu::context::{GpuContext, MAX_TEXTURE_DIM_2D};

/// Canvas pixels spanned by one full tile of the surface texture. The bump wraps
/// (Repeat sampling), so this sets the apparent weave scale; both the deposition
/// and shading passes must use the same value for the texture to line up.
pub const SURFACE_TILE_PX: f32 = 1024.0;

/// Which physical surface a document is painted on. Saved in `CanvasMeta` (§8)
/// because the surface affects *deposition* (tooth), so replay needs it to be
/// reproducible. The set is open — future custom/uploaded surfaces slot in here.
///
/// Non-`Flat` surfaces need image bytes, which the *frontend* fetches at runtime
/// and registers with the engine ([`crate::Engine::register_surface`]) — the
/// engine embeds nothing (DESIGN.md §6.4, §Inputs).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SurfaceId {
    /// Perfectly smooth: full height everywhere, so tooth is a no-op and the
    /// constant height has zero gradient (no relief). Paint behaves exactly as if
    /// there were no surface — the orthogonal default.
    #[default]
    Flat,
    /// The built-in tileable linen canvas weave.
    Linen,
}

/// A canvas surface: a single-channel height texture plus a tiling sampler.
#[derive(Clone)]
pub struct Surface {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// 1.0 if this is a real (image) surface with weave to interact with, 0.0 for
    /// the procedural `Flat`. Lets effects keyed on surface relief (e.g. the knife's
    /// tooth-gated scrape, §6.2) be a no-op on `Flat`, whose height is a constant 0.
    pub relief: f32,
}

impl Surface {
    /// A perfectly smooth surface: a 1×1 *zero-height* texel. Paint always stands
    /// above it (so it shows everywhere) and the constant height has zero gradient
    /// (no relief) — exactly equivalent to having no surface (DESIGN.md §6.4).
    pub fn flat(ctx: &GpuContext) -> Self {
        Self {
            relief: 0.0,
            ..Self::from_height(ctx, &[0u8], 1, 1)
        }
    }

    /// Decode a grayscale PNG height map into an `R8Unorm` tileable texture.
    pub fn load(ctx: &GpuContext, png_bytes: &[u8]) -> Self {
        let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
        let mut reader = decoder.read_info().expect("surface: read png info");
        let size = reader
            .output_buffer_size()
            .expect("surface: png output size");
        let mut buf = vec![0u8; size];
        let info = reader.next_frame(&mut buf).expect("surface: decode png frame");
        let (w, h) = (info.width, info.height);

        // Collapse to one height byte per texel (the source is 8-bit grayscale,
        // but accept the common color types defensively).
        let n = (w * h) as usize;
        let height: Vec<u8> = match info.color_type {
            png::ColorType::Grayscale => buf[..n].to_vec(),
            png::ColorType::GrayscaleAlpha => buf.chunks_exact(2).map(|p| p[0]).collect(),
            png::ColorType::Rgb => buf.chunks_exact(3).map(|p| p[0]).collect(),
            png::ColorType::Rgba => buf.chunks_exact(4).map(|p| p[0]).collect(),
            other => panic!("surface: unsupported PNG color type {other:?}"),
        };

        // Fit within the device texture limit (integer-factor box downsample).
        let (height, w, h) = downsample_to_limit(height, w, h, MAX_TEXTURE_DIM_2D);
        Self::from_height(ctx, &height, w, h)
    }

    /// Upload a single-channel height field as an `R8Unorm` tileable texture.
    fn from_height(ctx: &GpuContext, height: &[u8], w: u32, h: u32) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark surface bump"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            texture.as_image_copy(),
            height,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("stark surface sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { view, sampler, relief: 1.0 }
    }
}

/// Box-downsample a single-channel image by the smallest integer factor that
/// brings both edges within `limit`. An integer factor keeps a tileable texture
/// tileable; `factor == 1` returns the input unchanged.
fn downsample_to_limit(src: Vec<u8>, w: u32, h: u32, limit: u32) -> (Vec<u8>, u32, u32) {
    let factor = w.div_ceil(limit).max(h.div_ceil(limit)).max(1);
    if factor == 1 {
        return (src, w, h);
    }
    let (nw, nh) = (w / factor, h / factor);
    let area = factor * factor;
    let mut out = vec![0u8; (nw * nh) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    let sx = x * factor + dx;
                    let sy = y * factor + dy;
                    sum += src[(sy * w + sx) as usize] as u32;
                }
            }
            out[(y * nw + x) as usize] = (sum / area) as u8;
        }
    }
    (out, nw, nh)
}
