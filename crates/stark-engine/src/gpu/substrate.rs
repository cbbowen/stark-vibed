//! The physical canvas **substrate** — a tileable height map that decides both where paint
//! lands (the deposition tooth) and how the relief catches light, §6.4.
//!
//! It is a single global, color-space-independent texture sampled in *canvas*
//! space (so the substrate pans and zooms with the canvas), shared by the stroke
//! renderer (deposition) and the compositor (shading). Cloning is cheap — wgpu
//! views/samplers are reference-counted.
//!
//! The texture carries the height in `R` — the media pass's relief, §6.3 — and, in
//! `GB`, the **rise the substrate makes one [`TOOTH_REACH`] ahead** along each canvas
//! axis ([`pack_substrate`]). The rise is the whole of the deposition model: what a
//! dragged tip contacts is not a level set of the height but the *slope of the substrate
//! along its own travel* — it is pressed up by substrate rising to meet it and left
//! hanging by substrate falling away — so paint catches on the near faces of the grain
//! and bridges the lee sides. The rise is baked here, once, because nothing at draw
//! time should be recomputing a filter — which also means the filter can be as
//! carefully chosen as the model deserves.

use crate::gpu::context::GpuContext;
use stark_model::{SubstrateId, SubstrateScale};

mod import;
mod tooth;

use import::canonical_height;
pub use import::{canonicalize, identify};
use tooth::{Bearing, pack_substrate};

/// Canvas pixels spanned by one full tile of the substrate texture **at natural
/// scale**. The bump wraps (Repeat sampling), so this sets how coarse the grain reads;
/// both the deposition and shading passes must use the same value for the texture to
/// line up.
///
/// The document's [`SubstrateScale`] multiplies it — see [`Substrate`], which is the pair
/// everything downstream actually reads.
pub const SUBSTRATE_TILE_PX: f32 = 1024.0;

/// **A canvas substrate as the renderer builds it: which map, and how large it is
/// laid** (§6.4).
///
/// The pair, not the id alone, because the map that gets baked is a function of
/// both. The rise a tip meets is a difference taken across [`TOOTH_REACH`] *canvas
/// px* expressed in the map's own texels (`tooth::pack_substrate`), so laying the same
/// substrate at twice the size halves that span and changes what the tooth bites — and
/// the bearing table read off the result changes with it. Baking one map and scaling
/// the lookup would be the compensating fudge §1 rules out: it would report the rise
/// over six px as if it were the rise over three.
///
/// So this is the registry's key, while the *bytes* stay keyed by the [`SubstrateId`]
/// alone (`gpu::registry`): one height map, a bake per scale it is laid at.
///
/// [`TOOTH_REACH`]: tooth
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Substrate {
    /// Which substrate — the document's `SubstrateId`, the thing the bytes are named by.
    pub id: SubstrateId,
    /// How large it is laid, as the document says (`DocState::substrate_scale`).
    pub scale: SubstrateScale,
}

impl Substrate {
    /// The substrate `id` at natural size — what an engine with no document opinion
    /// stands on, and what every caller that has only an id means.
    pub fn new(id: SubstrateId) -> Self {
        Self {
            id,
            scale: SubstrateScale::NATURAL,
        }
    }

    /// Canvas px spanned by one full tile of this substrate's map.
    fn tile_px(self) -> f32 {
        SUBSTRATE_TILE_PX * self.scale.factor()
    }

    /// Canvas px → substrate-tile uv: `1 / tile_px`.
    ///
    /// **The one definition**, which both passes that sample the substrate reach through
    /// — the deposition tooth by way of the bake it was built for
    /// ([`SubstrateMap::uv_scale`]), the media pass by asking the document (see there).
    /// The substrate the paint catches on and the substrate the light catches on have to be
    /// the *same* substrate, or the highlights sit beside the grain instead of on it.
    pub fn uv_scale(self) -> f32 {
        1.0 / self.tile_px()
    }
}

/// A canvas substrate: the substrate texture (height + the rise ahead) plus a tiling
/// sampler.
#[derive(Clone)]
pub struct SubstrateMap {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// [`Substrate::uv_scale`] for the pair this map was baked for.
    ///
    /// Carried rather than recomputed because it is a fact *about this bake*: the
    /// deposit samples the rise channels, which were measured over a reach in the
    /// texels this scale implies, so reading them at any other uv would be reading
    /// the wrong substrate. The media pass asks the document instead, and the reason it
    /// may is on [`Substrate::uv_scale`].
    pub uv_scale: f32,
    /// 1.0 if this is a real (image) substrate with substrate to interact with, 0.0 for
    /// the procedural `Flat`. Lets effects keyed on substrate relief (e.g. the knife's
    /// scrape, §6.2) be a no-op on `Flat`, whose height is a constant 0.
    pub relief: f32,
    /// What share of the substrate a tip stands on, per tooth and direction of travel
    /// — the table that makes a toothed smear conserve paint ([`Bearing`]).
    bearing: Bearing,
}

impl SubstrateMap {
    /// A perfectly smooth substrate: a 1×1 *zero-height* texel. Paint always stands
    /// above it (so it shows everywhere) and the constant height has zero gradient
    /// (no relief) — exactly equivalent to having no substrate (§6.4).
    pub fn flat(ctx: &GpuContext, substrate: Substrate) -> Self {
        Self {
            relief: 0.0,
            ..Self::from_height(ctx, substrate, &[0u8], 1, 1)
        }
    }

    /// The **bearing fraction** at a given tooth — the tip's give and the width of its
    /// contact transition ([`BrushParams::tooth_give`](stark_model::document::BrushParams::tooth_give),
    /// [`tooth_softness`](stark_model::document::BrushParams::tooth_softness)) — and
    /// direction of travel (§6.4).
    ///
    /// [`Bearing::at`], with the one thing that is the *substrate's* business rather
    /// than the model's: a substrate with no relief has nothing to bite, whatever the
    /// tooth, and answers exactly 1.
    pub fn bearing(&self, give: f32, softness: f32, dir: [f32; 2]) -> f32 {
        if self.relief <= 0.0 {
            return 1.0;
        }
        self.bearing.at(give, softness, dir)
    }

    /// Upload a height field as the `Rgba8Unorm` **substrate** texture — height in `R`,
    /// the rise ahead in `GB` ([`pack_substrate`]) — and tabulate the bearing
    /// curve it implies.
    fn from_height(ctx: &GpuContext, substrate: Substrate, height: &[u8], w: u32, h: u32) -> Self {
        let packed = pack_substrate(height, w, h, substrate.tile_px());
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stark substrate map"),
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
            &packed,
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
            label: Some("stark substrate sampler"),
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
            uv_scale: substrate.uv_scale(),
            relief: 1.0,
            bearing: Bearing::tabulate(&packed),
        }
    }
}

impl crate::gpu::registry::Resource for Substrate {
    type Gpu = SubstrateMap;

    /// The bytes are the *substrate's*, and every scale it may be laid at shares them
    /// (`gpu::registry`).
    type Content = SubstrateId;

    fn content(self) -> SubstrateId {
        self.id
    }

    /// `Flat` is a 1x1 full-height texel: a constant
    /// height has zero gradient, so it is exactly equivalent to having no substrate
    /// (§6.4). It is the only substrate with no bytes behind it, which is what makes
    /// "the id names an image the holder may not have yet" a question with exactly
    /// one shape.
    fn is_builtin(self) -> bool {
        matches!(self.id, SubstrateId::Flat)
    }

    /// The decoded height field, kept by the registry so the **bake per scale** this
    /// substrate is laid at reads it instead of re-decoding the PNG (§6.4). One
    /// height map, one decode, however many sizes the document lays it at.
    type Decoded = stark_assetid::Canonical;

    fn decode(bytes: &[u8]) -> std::result::Result<Self::Decoded, stark_model::DocError> {
        canonical_height(bytes)
    }

    fn build(
        self,
        gpu: &GpuContext,
        registered: Option<crate::gpu::registry::Registered<'_, Self>>,
    ) -> SubstrateMap {
        match registered {
            Some(r) if !self.is_builtin() => {
                let f = r.decoded;
                SubstrateMap::from_height(gpu, self, &f.texels, f.width, f.height)
            }
            _ => SubstrateMap::flat(gpu, self),
        }
    }
}
