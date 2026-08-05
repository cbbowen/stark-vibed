//! Pluggable color spaces (§6.7).
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
    /// Tile auxiliary channel texture format (paint height).
    fn aux_format(&self) -> wgpu::TextureFormat;
    /// Blend for the color target when stamping/compositing.
    fn color_blend(&self) -> wgpu::BlendState;
    /// Blend for the aux target.
    fn aux_blend(&self) -> wgpu::BlendState;

    /// Straight display RGB → the space's four color channels (pre-coverage).
    fn rgb_to_channels(&self, rgb: [f32; 3]) -> [f32; 4];
    /// The space's color channels → straight display RGB (picker readout/export).
    fn channels_to_rgb(&self, channels: [f32; 4]) -> [f32; 3];

    /// WGSL for the stamp deposit pass (color + aux MRT outputs) — §6.2.
    fn stamp_shader(&self) -> &'static str;
    /// WGSL for the media/lighting + present pass — §6.3.
    fn media_shader(&self) -> &'static str;
    /// WGSL for the per-layer blend pass — §18.0.4. One isolated
    /// layer merged into the accumulator through a light-combining mode.
    ///
    /// A space needs its own variant because blending happens in *light* (normalized
    /// CIE XYZ) while the targets hold channels, so the pass is bracketed by this
    /// space's conversion out and back. The algebra between them is shared.
    fn blend_shader(&self) -> &'static str;

    /// Whether [`blend_shader`](Self::blend_shader) needs Mixbox's pigment LUT bound
    /// (`mixbox_lut.wesl`).
    ///
    /// A property of the space rather than a flag on the pass: coming *back* from
    /// light is a closed-form matrix in a colorimetric space and a table lookup in a
    /// pigment one, and only the pigment case pays for the texture. Everything else
    /// binds a 1×1 placeholder, so there is one bind group layout.
    fn needs_pigment_lut(&self) -> bool {
        false
    }
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

/// The default perceptual color space: Oklab `(L, a, b)` premultiplied by the
/// paint's *opacity* (in the color alpha), with paint height in a one-channel aux
/// (§6.5/§6.1). The media pass derives visible alpha from opacity × thickness,
/// where thickness is the difference between paint height and surface height.
pub struct OkLabColorSpace;

impl ColorSpace for OkLabColorSpace {
    fn id(&self) -> ColorSpaceId {
        ColorSpaceId::Oklab
    }

    fn color_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba16Float
    }
    fn aux_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::R16Float
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
        stark_shaders::stamp()
    }
    fn media_shader(&self) -> &'static str {
        stark_shaders::media_oklab()
    }
    fn blend_shader(&self) -> &'static str {
        stark_shaders::blend_oklab()
    }
}

/// Experimental **Mixbox** pigment-mixing space (§6.7). Colors are
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
        wgpu::TextureFormat::R16Float
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
        stark_shaders::stamp()
    }
    fn media_shader(&self) -> &'static str {
        stark_shaders::media_mixbox()
    }
    fn blend_shader(&self) -> &'static str {
        stark_shaders::blend_mixbox()
    }
    /// The one space that needs it: expressing combined *light* back as a pigment
    /// mixture is Mixbox's LUT, the inverse of the polynomial the media pass runs
    /// forward (§18.0.4).
    fn needs_pigment_lut(&self) -> bool {
        true
    }
}
