//! The physical canvas **surface** — a tileable height/bump map that affects both
//! media shading
//! (the relief catches light), §6.4.
//!
//! It is a single global, color-space-independent texture sampled in *canvas*
//! space (so the weave pans and zooms with the canvas), shared by the stroke
//! renderer (deposition) and the compositor (shading). Cloning is cheap — wgpu
//! views/samplers are reference-counted.

use serde::{Deserialize, Serialize};

use crate::assets::{AssetId, encode_gray_png};
use crate::error::{EngineError, Result};
use crate::gpu::context::{GpuContext, MAX_TEXTURE_DIM_2D};

/// Canvas pixels spanned by one full tile of the surface texture. The bump wraps
/// (Repeat sampling), so this sets the apparent weave scale; both the deposition
/// and shading passes must use the same value for the texture to line up.
pub const SURFACE_TILE_PX: f32 = 1024.0;

/// Canvas px → surface-tile uv, which is all a shader needs of [`SURFACE_TILE_PX`].
///
/// A function rather than a second constant so the deposition tooth (§6.4) and the
/// media pass cannot end up quoting different scales: the weave the paint catches on
/// and the weave the light catches on have to be the *same* weave, or the highlights
/// sit beside the grain instead of on it.
pub fn grain_uv_scale() -> f32 {
    1.0 / SURFACE_TILE_PX
}

/// Which physical surface a document is painted on. Saved in `CanvasMeta` (§8)
/// because which canvas a piece was painted on is part of the document, so it is
/// reproducible.
///
/// **Two variants, and the split is the point.** `Flat` is procedural and needs no
/// bytes; every other ground *is* its bytes, named by the hash of them. There is no
/// third case — no ground named by a label whose image the engine would have to be
/// told about separately — because that case is exactly the one that can go missing
/// (§6.4). A peer, a save file or a replay that meets an
/// [`Image`](Self::Image) id it has never seen can always ask for it by content, and
/// verify what comes back; a ground called "Gesso" could only be looked up in a
/// table the asker might not have, and the miss was silent — the tooth read a flat
/// stand-in and baked it into the tiles.
///
/// So this is the same bargain brush shapes already make (§6.6): the id
/// comes *from* the image ([`Engine::import_surface`](crate::Engine::import_surface)),
/// which is what makes "built-in" a property of the frontend's asset list and of
/// nothing downstream. The engine still embeds no image bytes.
#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum SurfaceId {
    /// Perfectly smooth: full height everywhere, so the
    /// constant height has zero gradient (no relief). Paint behaves exactly as if
    /// there were no surface — the orthogonal default.
    #[default]
    Flat,
    /// A height map, named by the BLAKE3 hash of its canonical decoded form
    /// ([`surface_id`]). Covers the grounds that ship with the app and the ones a
    /// user brings, identically — the engine cannot tell them apart, which is why
    /// neither can go missing in a way the other wouldn't.
    Image(AssetId),
}

/// A canvas surface: a single-channel height texture plus a tiling sampler.
#[derive(Clone)]
pub struct Surface {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// 1.0 if this is a real (image) surface with weave to interact with, 0.0 for
    /// the procedural `Flat`. Lets effects keyed on surface relief (e.g. the knife's
    /// scrape, §6.2) be a no-op on `Flat`, whose height is a constant 0.
    pub relief: f32,
    /// The height map's own histogram: the fraction of texels at each of the 256
    /// levels an `R8Unorm` can hold.
    ///
    /// It exists to answer one question — [`bearing`](Self::bearing) — and that
    /// question is what makes a **toothed smear conserve paint** (§6.4). The canvas
    /// side of the exchange gates each texel by the ground under *it*; the tool has
    /// no per-texel ground, so it books its side against the mean, and the mean of a
    /// gate over a height field is a sum over this histogram. Exact rather than
    /// sampled, which is why the shaders tap the map with nearest and not bilinear:
    /// the distribution they draw from is then this one, texel for texel.
    ///
    /// `Arc` so a `Surface` stays two atomic bumps to clone — it is cloned per
    /// stroke, and 1 KB per clone is not the shape of this type.
    hist: std::sync::Arc<[f32; 256]>,
}

/// Width of the tooth's contact transition, in the height map's [0, 1] units.
///
/// **Must match `TOOTH_SOFTNESS` in `paint_common.wesl`.** The pair below is a
/// deliberate mirror of the shader's, and the mirror is load-bearing rather than
/// convenient: the canvas evaluates the gate per texel on the GPU while the tool
/// books its side against [`Surface::bearing`] on the CPU, so if the two functions
/// disagree the two halves of the transfer disagree and a smear stops conserving.
/// That is also what guards it — `tests/dynamics.rs`'s conservation pair is sensitive
/// to exactly this, so a drift here fails a test rather than quietly leaking paint.
const TOOTH_SOFTNESS: f32 = 0.15;

/// The contact level the tip presses to, from the `tooth` knob (see
/// `paint_common.wesl::tooth_level`).
fn tooth_level(tooth: f32) -> f32 {
    tooth * (1.0 + TOOTH_SOFTNESS) - 0.5 * TOOTH_SOFTNESS
}

/// The share of its paint a texel at ground height `s` receives
/// (`paint_common.wesl::tooth_gate`).
fn tooth_gate(s: f32, tooth: f32) -> f32 {
    let t = ((s - tooth_level(tooth)) / TOOTH_SOFTNESS + 0.5).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Surface {
    /// A perfectly smooth surface: a 1×1 *zero-height* texel. Paint always stands
    /// above it (so it shows everywhere) and the constant height has zero gradient
    /// (no relief) — exactly equivalent to having no surface (§6.4).
    pub fn flat(ctx: &GpuContext) -> Self {
        Self {
            relief: 0.0,
            ..Self::from_height(ctx, &[0u8], 1, 1)
        }
    }

    /// The **bearing fraction** at a given tooth: what share of the ground stands
    /// above the level a tip with this `tooth` presses to, averaged over the whole
    /// height map (§6.4).
    ///
    /// This is the model's own central quantity — the Abbott–Firestone curve
    /// evaluated at one level — and it is where paint conservation comes from. A
    /// canvas texel gates its half of the exchange by the ground under itself; the
    /// tool has no ground of its own, so it gates its half by this. The two agree in
    /// expectation over any footprint that spans many grain features, which is every
    /// usable tip, and the residual is the ground's sampling fluctuation under the
    /// tip — the same order as the mean-field freeze the loop already carries.
    ///
    /// Exactly 1 where there is nothing to bite: no tooth, or no relief. Not
    /// *approximately* 1 — the sum over a histogram would land a rounding error away,
    /// and that error would be a systematic leak on every stroke of every ordinary
    /// brush.
    pub fn bearing(&self, tooth: f32) -> f32 {
        if tooth <= 0.0 || self.relief <= 0.0 {
            return 1.0;
        }
        let mut mean = 0.0;
        for (level, share) in self.hist.iter().enumerate() {
            mean += share * tooth_gate(level as f32 / 255.0, tooth);
        }
        mean
    }

    /// Build from a height-map PNG. The bytes reached the registry through
    /// [`canonicalize`] or [`identify`], which decoded them once already, so a
    /// failure here is a broken invariant rather than bad input — hence the
    /// `expect` rather than a `Result` the caller would have nothing to do with.
    pub fn load(ctx: &GpuContext, png_bytes: &[u8]) -> Self {
        let (w, h, height) = canonical_height(png_bytes).expect("surface: registered bytes decode");
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
        let mut counts = [0u32; 256];
        for byte in height {
            counts[*byte as usize] += 1;
        }
        let n = height.len().max(1) as f32;
        let hist = std::array::from_fn(|i| counts[i] as f32 / n);
        Self {
            view,
            sampler,
            relief: 1.0,
            hist: std::sync::Arc::new(hist),
        }
    }
}

/// Import a height map: the id that names it, and the canonical bytes to keep
/// beside it. Re-encoded from the decoded height, so what is stored, bundled into a
/// save file and sent to a peer is the form the id actually names — reload it and
/// you land on the same id.
///
/// The engine's entry point is
/// [`Engine::import_surface`](crate::Engine::import_surface).
pub fn canonicalize(png_bytes: &[u8]) -> Result<(SurfaceId, Vec<u8>)> {
    let (w, h, height) = canonical_height(png_bytes)?;
    let id = SurfaceId::Image(surface_id(w, h, &height));
    Ok((id, encode_gray_png(w, h, &height)?))
}

/// The id of an already-canonical height map — bytes out of a save file or off a
/// peer, which are kept verbatim. Derived rather than taken on trust: a ground whose
/// bytes did not hash to the id that asked for them is a ground that would silently
/// deposit the wrong tooth, so the caller gets the id the bytes *are* and compares.
pub fn identify(png_bytes: &[u8]) -> Result<SurfaceId> {
    let (w, h, height) = canonical_height(png_bytes)?;
    Ok(SurfaceId::Image(surface_id(w, h, &height)))
}

/// Content id of a canonical height field: the hash of its dimensions and texels.
///
/// Over the *decoded, downsampled* field rather than the file bytes, for the reason
/// [`AssetId`] names a brush's coverage the same way — it is what actually drives
/// pixels, so two peers who encoded the same weave differently converge on one id.
/// That this is deterministic across peers rests on [`MAX_TEXTURE_DIM_2D`] being a
/// fixed constant rather than a device query: were the downsample factor to follow
/// the adapter's real limit, the same PNG would canonicalize differently on two
/// machines and the id would stop naming one thing.
fn surface_id(width: u32, height: u32, texels: &[u8]) -> AssetId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&width.to_le_bytes());
    hasher.update(&height.to_le_bytes());
    hasher.update(texels);
    AssetId(*hasher.finalize().as_bytes())
}

/// Decode a height-map PNG to its canonical form: one height byte per texel,
/// box-downsampled by an integer factor to fit [`MAX_TEXTURE_DIM_2D`].
///
/// Channel 0, not luminance — a height map's grey *is* its height, so an RGB source
/// carries it in red and weighting the channels would tilt the ground.
fn canonical_height(png_bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| EngineError::Asset(e.to_string()))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| EngineError::Asset("surface: missing png size".into()))?;
    let mut buf = vec![0u8; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| EngineError::Asset(e.to_string()))?;
    let (w, h) = (info.width, info.height);

    // Collapse to one height byte per texel (the source is 8-bit grayscale, but
    // accept the common color types defensively).
    let n = (w * h) as usize;
    let height: Vec<u8> = match info.color_type {
        png::ColorType::Grayscale => buf[..n].to_vec(),
        png::ColorType::GrayscaleAlpha => buf.as_chunks::<2>().0.iter().map(|p| p[0]).collect(),
        png::ColorType::Rgb => buf.as_chunks::<3>().0.iter().map(|p| p[0]).collect(),
        png::ColorType::Rgba => buf.as_chunks::<4>().0.iter().map(|p| p[0]).collect(),
        other => {
            return Err(EngineError::Asset(format!(
                "surface: unsupported PNG color type {other:?}"
            )));
        }
    };

    let (height, w, h) = downsample_to_limit(height, w, h, MAX_TEXTURE_DIM_2D);
    Ok((w, h, height))
}

/// Box-downsample a single-channel image by the smallest integer factor that
/// brings both edges within `limit`. An integer factor keeps a tileable texture
/// tileable; `factor == 1` returns the input unchanged.
pub(crate) fn downsample_to_limit(src: Vec<u8>, w: u32, h: u32, limit: u32) -> (Vec<u8>, u32, u32) {
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

impl crate::gpu::registry::Resource for SurfaceId {
    type Gpu = Surface;

    /// `Flat` is a 1x1 full-height texel: a constant
    /// height has zero gradient, so it is exactly equivalent to having no surface
    /// (§6.4). It is the only ground with no bytes behind it, which is what makes
    /// "the id names an image the holder may not have yet" a question with exactly
    /// one shape.
    fn is_builtin(self) -> bool {
        matches!(self, SurfaceId::Flat)
    }

    fn build(self, gpu: &GpuContext, bytes: Option<&[u8]>) -> Surface {
        match bytes {
            Some(bytes) if !self.is_builtin() => Surface::load(gpu, bytes),
            _ => Surface::flat(gpu),
        }
    }
}
