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
///
/// **Every variant is unconditional, including one whose implementation a build may
/// not carry** (`Mixbox`, behind the `mixbox` cargo feature). Postcard encodes enums
/// by index (§8): a build that `cfg`'d a variant away would renumber every variant
/// appended after it, so the same bytes would name different spaces in two builds of
/// the same version — a wire-format break that nothing would report, which is the
/// class §19 exists to rule out. An id is therefore always nameable and always
/// serializable, and whether it can be *honoured* is [`Self::make`]'s answer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpaceId {
    Oklab,
    Mixbox,
}

impl ColorSpaceId {
    /// Construct the color space implementation for this id, or `None` when this
    /// build does not carry it.
    ///
    /// `None` is reachable only for [`ColorSpaceId::Mixbox`] without the `mixbox`
    /// feature. It is deliberately not a fallback to Oklab: the two spaces read the
    /// same tile bytes as different colors, so opening a pigment document through a
    /// colorimetric space would render every pixel wrong while looking like it
    /// worked. Failing is the honest answer, and
    /// [`EngineError::UnsupportedColorSpace`](crate::error::EngineError::UnsupportedColorSpace)
    /// is where it lands.
    pub fn make(self) -> Option<Arc<dyn ColorSpace>> {
        match self {
            ColorSpaceId::Oklab => Some(Arc::new(OkLabColorSpace)),
            #[cfg(feature = "mixbox")]
            ColorSpaceId::Mixbox => Some(Arc::new(MixboxColorSpace)),
            #[cfg(not(feature = "mixbox"))]
            ColorSpaceId::Mixbox => None,
        }
    }

    /// Whether this build can open a document in this space — [`Self::make`] without
    /// building anything. What a frontend asks to decide which spaces to offer.
    pub fn available(self) -> bool {
        self.make().is_some()
    }

    /// Every id this build can actually open, in a stable order — the list a "new
    /// document" picker is built from.
    pub fn all_available() -> impl Iterator<Item = ColorSpaceId> {
        [ColorSpaceId::Oklab, ColorSpaceId::Mixbox]
            .into_iter()
            .filter(|id| id.available())
    }
}

/// A color space: tile layout + blend + picker conversions + GPU shaders.
pub trait ColorSpace {
    fn id(&self) -> ColorSpaceId;

    /// Tile color channel texture format.
    fn color_format(&self) -> wgpu::TextureFormat;
    /// Tile auxiliary channel texture format (paint height).
    fn aux_format(&self) -> wgpu::TextureFormat;
    /// The tile's **residual** channel format, or `None` for a space that has no
    /// residual (§6.7).
    ///
    /// A residual is the part of a color the space's three channels cannot express.
    /// A colorimetric space has none — Oklab's `(L, a, b)` reproduces every sRGB
    /// color exactly — so it allocates no third texture, and the passes that carry
    /// tile color are built in a variant without one (`stark_shaders`'s
    /// `RESID_ENTRY_POINTS`). A *pigment* space has one necessarily: four trained
    /// pigments do not span sRGB.
    ///
    /// `Some` costs eight bytes a texel and a third render target through every pass
    /// that writes a tile. `None` is not an optimization but a statement — that this
    /// space's channels are the whole color.
    fn resid_format(&self) -> Option<wgpu::TextureFormat> {
        None
    }

    /// Whether this space carries a residual — [`resid_format`](Self::resid_format)
    /// as the flag the shader variants and bind groups actually branch on.
    fn has_resid(&self) -> bool {
        self.resid_format().is_some()
    }
    /// Blend for the color target when stamping/compositing.
    fn color_blend(&self) -> wgpu::BlendState;
    /// Blend for the aux target.
    fn aux_blend(&self) -> wgpu::BlendState;

    /// Straight display RGB → the space's four color channels (pre-coverage).
    fn rgb_to_channels(&self, rgb: [f32; 3]) -> [f32; 4];
    /// Straight display RGB → the **residual** those channels leave behind (§6.7).
    ///
    /// Zero for a space with no [`resid_format`](Self::resid_format), and zero
    /// *exactly*: the default here is not a placeholder for an unimplemented
    /// conversion but the true answer, since such a space's channels already say the
    /// whole color. Callers write it unconditionally into a uniform lane, and for
    /// Oklab that lane is genuinely zeroes rather than an unread field.
    fn rgb_to_resid(&self, _rgb: [f32; 3]) -> [f32; 3] {
        [0.0; 3]
    }
    /// The space's color channels **and residual** → straight display RGB (picker
    /// readout/export). The inverse of the two functions above, taken together.
    fn channels_to_rgb(&self, channels: [f32; 4], resid: [f32; 3]) -> [f32; 3];

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
    /// WGSL for the **filter layer** pass — §21. The accumulator beneath a filter
    /// layer, read and rewritten.
    ///
    /// A space needs its own variant for the reason
    /// [`blend_shader`](Self::blend_shader) does, and it is the same reason twice: a
    /// color adjustment is a statement about light and about perceived color,
    /// while the targets hold channels, so the pass is bracketed by this space's
    /// conversion out and back. The adjustment between them is shared.
    fn filter_shader(&self) -> &'static str;

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

    /// `resid` is ignored, and is always `[0.0; 3]` for this space: Oklab reproduces
    /// every sRGB color, so there is nothing left over to add back.
    fn channels_to_rgb(&self, channels: [f32; 4], _resid: [f32; 3]) -> [f32; 3] {
        let lin = color::oklab_to_linear_srgb([channels[0], channels[1], channels[2]]);
        [
            color::linear_to_srgb(lin[0]),
            color::linear_to_srgb(lin[1]),
            color::linear_to_srgb(lin[2]),
        ]
    }

    fn stamp_shader(&self) -> &'static str {
        stark_shaders::stamp(false)
    }
    fn media_shader(&self) -> &'static str {
        stark_shaders::media_oklab()
    }
    fn blend_shader(&self) -> &'static str {
        stark_shaders::blend_oklab()
    }
    fn filter_shader(&self) -> &'static str {
        stark_shaders::filter_oklab()
    }
}

/// **Mixbox** pigment-mixing space (§6.7). Colors are stored as Mixbox
/// latent pigment *concentrations* `(c0, c1, c2)` — the fourth, `c3 = 1 −
/// (c0+c1+c2)`, is derived — **plus the latent's residual in a third tile texture**.
/// Because the latent mixes linearly, the ordinary premultiplied-"over" deposit *is*
/// Mixbox mixing (blue over yellow → green), so the blends and the stamp law are the
/// same as Oklab's; what differs is the third channel and the media pass, which
/// evaluates Mixbox's pigment polynomial and adds the residual back.
///
/// **The residual is not optional.** Four trained pigments do not span sRGB, so the
/// polynomial alone reaches neither black — whose concentrations render as `#383838`
/// — nor the saturated corners, where it is off by up to 0.39 (mean 0.05 over the
/// cube). This engine dropped it for as long as a tile held only three concentrations
/// plus coverage, and no cheaper recovery exists: `rgb → c` is many-to-one, with up
/// to 70 sRGB colors sharing one quantized triple across 0.38 of the cube, so the
/// residual is not a function of the channels stored beside it.
///
/// Conversions use the vendored `mixbox` crate (CC BY-NC 4.0; `vendor/mixbox`),
/// which is why this whole space is behind the `mixbox` cargo feature: the licence is
/// non-commercial, so a build has to be able to leave it out entirely rather than
/// merely not reach it. [`ColorSpaceId::Mixbox`] still exists there — see
/// [`ColorSpaceId::make`] for why the *id* cannot be gated even though the
/// implementation can.
#[cfg(feature = "mixbox")]
pub struct MixboxColorSpace;

#[cfg(feature = "mixbox")]
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
    /// The residual's three components premultiplied by the same per-unit opacity the
    /// concentrations are, with that opacity duplicated into the fourth: not
    /// redundancy for its own sake, but because the fixed-function "over" reads *each*
    /// target's own alpha, so the target carrying the residual has to hold it too.
    fn resid_format(&self) -> Option<wgpu::TextureFormat> {
        Some(wgpu::TextureFormat::Rgba16Float)
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

    fn rgb_to_resid(&self, rgb: [f32; 3]) -> [f32; 3] {
        // The other half of the same latent: `rgb − poly(c)`, which is what makes the
        // round trip below exact rather than approximate.
        let z = mixbox::float_rgb_to_latent(&rgb);
        [z[4], z[5], z[6]]
    }

    fn channels_to_rgb(&self, channels: [f32; 4], resid: [f32; 3]) -> [f32; 3] {
        // Reassemble the latent and evaluate `poly(c) + r` — Mixbox's own
        // `latent_to_rgb`, so a color picked and read back is the color picked.
        let (c0, c1, c2) = (channels[0], channels[1], channels[2]);
        let latent = [
            c0,
            c1,
            c2,
            1.0 - (c0 + c1 + c2),
            resid[0],
            resid[1],
            resid[2],
        ];
        mixbox::latent_to_float_rgb(&latent)
    }

    fn stamp_shader(&self) -> &'static str {
        // Deposit is premultiplied-over of the channels — the same law as Oklab's,
        // run over one more target.
        stark_shaders::stamp(true)
    }
    fn media_shader(&self) -> &'static str {
        stark_shaders::media_mixbox()
    }
    fn blend_shader(&self) -> &'static str {
        stark_shaders::blend_mixbox()
    }
    fn filter_shader(&self) -> &'static str {
        stark_shaders::filter_mixbox()
    }
    /// The one space that needs it: expressing combined or adjusted *light* back as
    /// a pigment mixture is Mixbox's LUT, the inverse of the polynomial the media
    /// pass runs forward (§18.0.4, §21).
    fn needs_pigment_lut(&self) -> bool {
        true
    }
}
