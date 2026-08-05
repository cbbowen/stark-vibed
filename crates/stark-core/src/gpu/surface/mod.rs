//! The physical canvas **surface** — a tileable ground that decides both where paint
//! lands (the deposition tooth) and how the relief catches light, §6.4.
//!
//! It is a single global, color-space-independent texture sampled in *canvas*
//! space (so the weave pans and zooms with the canvas), shared by the stroke
//! renderer (deposition) and the compositor (shading). Cloning is cheap — wgpu
//! views/samplers are reference-counted.
//!
//! The texture carries the height in `R` — the media pass's relief, §6.3 — and, in
//! `GB`, the **rise the ground makes one [`TOOTH_REACH`] ahead** along each canvas
//! axis ([`pack_ground`]). The rise is the whole of the deposition model: what a
//! dragged tip contacts is not a level set of the height but the *slope of the ground
//! along its own travel* — it is pressed up by ground rising to meet it and left
//! hanging by ground falling away — so paint catches on the near faces of the grain
//! and bridges the lee sides. The rise is baked here, once, because nothing at draw
//! time should be recomputing a filter — which also means the filter can be as
//! carefully chosen as the model deserves.

use serde::{Deserialize, Serialize};

use crate::assets::AssetId;
use crate::gpu::context::GpuContext;

mod import;
mod tooth;

use import::canonical_height;
pub use import::{canonicalize, identify};
use tooth::{Bearing, pack_ground};

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
    /// Perfectly smooth: zero height everywhere, so the
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

/// A canvas surface: the ground texture (height + the rise ahead) plus a tiling
/// sampler.
#[derive(Clone)]
pub struct Surface {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// 1.0 if this is a real (image) surface with weave to interact with, 0.0 for
    /// the procedural `Flat`. Lets effects keyed on surface relief (e.g. the knife's
    /// scrape, §6.2) be a no-op on `Flat`, whose height is a constant 0.
    pub relief: f32,
    /// What share of the ground a tip stands on, per tooth and direction of travel
    /// — the table that makes a toothed smear conserve paint ([`Bearing`]).
    bearing: Bearing,
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

    /// The **bearing fraction** at a given tooth and direction of travel (§6.4) —
    /// [`Bearing::at`], with the one thing that is the *surface's* business rather
    /// than the model's: a ground with no relief has nothing to bite, whatever the
    /// tooth, and answers exactly 1.
    pub fn bearing(&self, tooth: f32, dir: [f32; 2]) -> f32 {
        if self.relief <= 0.0 {
            return 1.0;
        }
        self.bearing.at(tooth, dir)
    }

    /// Build from a height-map PNG. The bytes reached the registry through
    /// [`canonicalize`] or [`identify`], which decoded them once already, so a
    /// failure here is a broken invariant rather than bad input — hence the
    /// `expect` rather than a `Result` the caller would have nothing to do with.
    pub fn load(ctx: &GpuContext, png_bytes: &[u8]) -> Self {
        let (w, h, height) = canonical_height(png_bytes).expect("surface: registered bytes decode");
        Self::from_height(ctx, &height, w, h)
    }

    /// Upload a height field as the `Rgba8Unorm` **ground** texture — height in `R`,
    /// the rise ahead in `GB` ([`pack_ground`]) — and tabulate the bearing
    /// curve it implies.
    fn from_height(ctx: &GpuContext, height: &[u8], w: u32, h: u32) -> Self {
        let ground = pack_ground(height, w, h);
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark surface ground"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            texture.as_image_copy(),
            &ground,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
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
        Self {
            view,
            sampler,
            relief: 1.0,
            bearing: Bearing::tabulate(&ground),
        }
    }
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
