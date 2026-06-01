//! Pluggable color spaces (DESIGN.md §6.7).
//!
//! Tile channels are color-space-agnostic: tools deposit values and only assume
//! they blend linearly. A [`ColorSpace`] gives those channels meaning — the tile
//! texture layout, how dabs combine (blend), the picker conversions to/from RGB,
//! and the GPU shaders that deposit (stamp) and present (media) them.
//!
//! A document has one color space, selected by [`ColorSpaceId`] in `CanvasMeta`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::color;

/// Identifies a color space; serialized in the save format (`CanvasMeta`, §8).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpaceId {
    Oklab,
    Mixbox,
}

impl ColorSpaceId {
    /// Construct the color space implementation for this id.
    pub fn make(self) -> Arc<dyn ColorSpace> {
        match self {
            ColorSpaceId::Oklab => Arc::new(OkLabColorSpace),
            ColorSpaceId::Mixbox => Arc::new(MixboxColorSpace),
        }
    }
}

/// A color space: tile layout + blend + picker conversions + GPU shaders.
pub trait ColorSpace {
    fn id(&self) -> ColorSpaceId;

    /// Tile color channel texture format.
    fn color_format(&self) -> wgpu::TextureFormat;
    /// Tile auxiliary channel texture format (height/wet/…).
    fn aux_format(&self) -> wgpu::TextureFormat;
    /// Blend for the color target when stamping/compositing.
    fn color_blend(&self) -> wgpu::BlendState;
    /// Blend for the aux target.
    fn aux_blend(&self) -> wgpu::BlendState;

    /// Straight display RGB → the space's four color channels (pre-coverage).
    fn rgb_to_channels(&self, rgb: [f32; 3]) -> [f32; 4];
    /// The space's color channels → straight display RGB (picker readout/export).
    fn channels_to_rgb(&self, channels: [f32; 4]) -> [f32; 3];

    /// WGSL for the stamp deposit pass (color + aux MRT outputs) — DESIGN §6.2.
    fn stamp_shader(&self) -> &'static str;
    /// WGSL for the media/lighting + present pass — DESIGN §6.3.
    fn media_shader(&self) -> &'static str;
}

/// Premultiplied "over" — the standard alpha compositing blend.
fn over() -> wgpu::BlendState {
    wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
}

/// Additive accumulation (`One, One`).
fn additive() -> wgpu::BlendState {
    let c = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState { color: c, alpha: c }
}

/// The default perceptual color space: premultiplied Oklab `(L, a, b)` with
/// coverage in the color alpha, height/wet in a two-channel aux (DESIGN.md §6.5).
pub struct OkLabColorSpace;

impl ColorSpace for OkLabColorSpace {
    fn id(&self) -> ColorSpaceId {
        ColorSpaceId::Oklab
    }

    fn color_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba16Float
    }
    fn aux_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rg16Float
    }
    fn color_blend(&self) -> wgpu::BlendState {
        over()
    }
    fn aux_blend(&self) -> wgpu::BlendState {
        additive()
    }

    fn rgb_to_channels(&self, rgb: [f32; 3]) -> [f32; 4] {
        let lin = [
            color::srgb_to_linear(rgb[0]),
            color::srgb_to_linear(rgb[1]),
            color::srgb_to_linear(rgb[2]),
        ];
        let lab = color::linear_srgb_to_oklab(lin);
        [lab[0], lab[1], lab[2], 1.0]
    }

    fn channels_to_rgb(&self, channels: [f32; 4]) -> [f32; 3] {
        let lin = color::oklab_to_linear_srgb([channels[0], channels[1], channels[2]]);
        [
            color::linear_to_srgb(lin[0]),
            color::linear_to_srgb(lin[1]),
            color::linear_to_srgb(lin[2]),
        ]
    }

    fn stamp_shader(&self) -> &'static str {
        stark_shaders::stamp_oklab()
    }
    fn media_shader(&self) -> &'static str {
        stark_shaders::media_oklab()
    }
}

/// Experimental **Mixbox** pigment-mixing space (DESIGN.md §6.7). Colors are
/// stored as Mixbox latent pigment *concentrations* `(c0, c1, c2)` — the fourth,
/// `c3 = 1 − (c0+c1+c2)`, is derived, and the latent residual is dropped so the
/// three concentrations fit alongside coverage. Because the latent mixes linearly,
/// the ordinary premultiplied-"over" deposit *is* Mixbox mixing (blue over yellow
/// → green), so the layout, blends, and stamp shader are identical to Oklab; only
/// the media pass differs (it evaluates Mixbox's pigment polynomial).
///
/// Conversions use the vendored `mixbox` crate (CC BY-NC 4.0; `vendor/mixbox`).
pub struct MixboxColorSpace;

impl ColorSpace for MixboxColorSpace {
    fn id(&self) -> ColorSpaceId {
        ColorSpaceId::Mixbox
    }

    fn color_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba16Float
    }
    fn aux_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rg16Float
    }
    fn color_blend(&self) -> wgpu::BlendState {
        over()
    }
    fn aux_blend(&self) -> wgpu::BlendState {
        additive()
    }

    fn rgb_to_channels(&self, rgb: [f32; 3]) -> [f32; 4] {
        // Mixbox latent = [c0, c1, c2, c3, residual…]; keep the concentrations.
        let z = mixbox::float_rgb_to_latent(&rgb);
        [z[0], z[1], z[2], 1.0]
    }

    fn channels_to_rgb(&self, channels: [f32; 4]) -> [f32; 3] {
        // Rebuild a residual-free latent and evaluate the pigment polynomial.
        let (c0, c1, c2) = (channels[0], channels[1], channels[2]);
        let latent = [c0, c1, c2, 1.0 - (c0 + c1 + c2), 0.0, 0.0, 0.0];
        mixbox::latent_to_float_rgb(&latent)
    }

    fn stamp_shader(&self) -> &'static str {
        // Deposit is premultiplied-over of the channels — identical to Oklab.
        stark_shaders::stamp_oklab()
    }
    fn media_shader(&self) -> &'static str {
        stark_shaders::media_mixbox()
    }
}
