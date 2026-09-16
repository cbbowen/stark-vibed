//! Pluggable color spaces (§6.7).
//!
//! Tile channels are color-space-agnostic: tools deposit values and only assume
//! they blend linearly. A [`ColorSpace`] gives those channels meaning — the tile
//! texture layout, how dabs combine (blend), the picker conversions to/from RGB,
//! and the GPU shaders that deposit (stamp) and present (media) them.
//!
//! A document has one color space, selected by [`ColorSpaceId`] in `CanvasMeta`.

use std::sync::Arc;

use stark_model::Srgb;
use stark_model::{ColorSpaceId, color};
use stark_shaders::{Lane, Pigment, Resid};

/// Construct the color space implementation for `id`, or `None` when this
/// build does not carry it.
///
/// `None` is reachable only for [`ColorSpaceId::Mixbox`] without the `mixbox`
/// feature, and is deliberately not a fallback to Oklab: the two spaces read the
/// same tile bytes as different colors, so opening a pigment document through a
/// colorimetric space would render every pixel wrong while looking like it worked.
/// Callers turn it into
/// [`DocError::UnsupportedColorSpace`](stark_model::DocError::UnsupportedColorSpace).
///
/// A free function rather than a method on the id, because the id is
/// `stark-model`'s and an inherent impl may only be written where the type is (§2).
pub fn make(id: ColorSpaceId) -> Option<Arc<dyn ColorSpace>> {
    match id {
        ColorSpaceId::Oklab => Some(Arc::new(OkLabColorSpace)),
        #[cfg(feature = "mixbox")]
        ColorSpaceId::Mixbox => Some(Arc::new(MixboxColorSpace)),
        #[cfg(not(feature = "mixbox"))]
        ColorSpaceId::Mixbox => None,
    }
}

/// Whether this build can open a document in `id`'s space — [`make`] without
/// building anything. What a frontend asks to decide which spaces to offer.
///
/// The feature gate is the whole answer, and both read it, so a space [`make`]
/// cannot build is a space this reports unavailable.
pub fn available(id: ColorSpaceId) -> bool {
    match id {
        ColorSpaceId::Oklab => true,
        ColorSpaceId::Mixbox => cfg!(feature = "mixbox"),
    }
}

/// Every id this build can actually open, in a stable order — the list a "new
/// document" picker is built from.
pub fn all_available() -> impl Iterator<Item = ColorSpaceId> {
    [ColorSpaceId::Oklab, ColorSpaceId::Mixbox]
        .into_iter()
        .filter(|id| available(*id))
}

/// A color as the working space stores it: the channels a tile's color target holds,
/// and the residual a pigment space's third target holds beside them (§6.7).
///
/// Three and three, not four and four: the fourth lane of each target is per-unit
/// opacity — a property of the *paint*, not of the color (§6.1) — and call sites
/// write it themselves rather than take it from a conversion with no opinion on it.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Latent {
    /// The space's three color channels, pre-coverage: Oklab's `L, a, b`, or the
    /// first three Mixbox pigment concentrations.
    pub lat: [f32; 3],
    /// What those channels cannot say, added back on the way out — `[0.0; 3]`
    /// **exactly** in a space with no [`resid_format`](ColorSpace::resid_format),
    /// which is the true answer rather than a placeholder: such a space's channels
    /// already carry the whole color.
    pub res: [f32; 3],
}

/// A color space: tile layout + blend + picker conversions + GPU shaders.
pub trait ColorSpace {
    fn id(&self) -> ColorSpaceId;

    /// Tile color channel texture format.
    ///
    /// Defaulted, along with [`aux_format`](Self::aux_format),
    /// [`color_blend`](Self::color_blend) and [`aux_blend`](Self::aux_blend): an
    /// override here is a format the tile pool was never sized for. What actually
    /// distinguishes a space is [`resid_format`](Self::resid_format).
    fn color_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba16Float
    }
    /// Tile auxiliary channel texture format (paint height).
    fn aux_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::R16Float
    }
    /// The tile's **residual** channel format, or `None` for a space that has no
    /// residual (§6.7).
    ///
    /// A residual is the part of a color the space's three channels cannot express.
    /// A colorimetric space has none — Oklab's `(L, a, b)` reproduces every sRGB
    /// color exactly — so it allocates no third texture, and the passes that carry
    /// tile color are built in a variant without one ([`Resid::Without`]). A *pigment*
    /// space has one necessarily: four trained pigments do not span sRGB.
    ///
    /// `Some` costs eight bytes a texel and a third render target through every pass
    /// that writes a tile. `None` is a statement, not an optimization: this space's
    /// channels are the whole color.
    fn resid_format(&self) -> Option<wgpu::TextureFormat> {
        None
    }

    /// Whether this space carries a residual — [`resid_format`](Self::resid_format)
    /// as the flag the bind groups actually branch on.
    fn has_resid(&self) -> bool {
        self.resid_format().is_some()
    }

    /// Which build of the three passes bracketed by a space's own conversion this space
    /// reads (§6.7).
    ///
    /// **Declared, not derived from [`resid_format`](Self::resid_format)**, though the
    /// two agree for both spaces there are: a residual is a fact about how much of a
    /// color three channels can hold, and this is a fact about what those channels
    /// *are*. Deriving one from the other would make a colorimetric space with a
    /// residual unstateable rather than merely absent.
    fn pigment(&self) -> Pigment {
        Pigment::Oklab
    }

    /// Which build of a pass that carries a tile's color this space reads (§6.7).
    ///
    /// **Derived** from [`resid_format`](Self::resid_format), like
    /// [`has_resid`](Self::has_resid), rather than declared beside it — a space with
    /// two answers to one question is the drift this whole boundary is about. The
    /// `cfg` is what the derivation costs: [`Resid::With`] is generated only in a build
    /// that linked the residual variants, which is the same build a space with a
    /// `resid_format` can exist in.
    fn resid(&self) -> Resid {
        #[cfg(feature = "mixbox")]
        if self.has_resid() {
            return Resid::With;
        }
        Resid::Without
    }
    /// Blend for the color target when stamping/compositing.
    fn color_blend(&self) -> wgpu::BlendState {
        over()
    }
    /// Blend for the aux target — additive, because what it carries is an *amount*
    /// of paint and amounts add (§6.1).
    fn aux_blend(&self) -> wgpu::BlendState {
        additive()
    }

    /// Straight display RGB → the space's color channels **and** the residual they
    /// leave behind (§6.7), in one conversion.
    ///
    /// An [`Srgb`] is finite and bounded by construction, so an implementation never
    /// has to wonder whether it was handed a `NaN`. It may well be **outside the sRGB
    /// cube** — extended sRGB is what a document carries (§6.5) — which is a
    /// colorimetric space's to represent and a pigment space's to hold.
    ///
    /// **One method because it is one evaluation**: in Mixbox the concentrations and
    /// the remainder come out of a single run of the pigment polynomial, and the
    /// conversion is per *texel* on a placed image (`gpu::place`).
    fn rgb_to_latent(&self, rgb: Srgb) -> Latent;
    /// The space's color channels **and residual** → straight display RGB (picker
    /// readout/export). The inverse of the two functions above, taken together.
    fn channels_to_rgb(&self, channels: [f32; 4], resid: [f32; 3]) -> [f32; 3];

    /// The stamp deposit pass (color + aux MRT outputs) — §6.2.
    /// [`Lane::Ceiling`] asks for the variant that also accumulates the ceiling lane,
    /// the fourth target a stroke whose opacity the pen drives sweeps into.
    ///
    /// The generated record whole, because both spaces link the same module
    /// (`stamp.wesl`) and so share its type — which the three below do not.
    fn stamp_shader(&self, lane: Lane) -> &'static stark_shaders::Stamp;
    /// The media/lighting + present pass — §6.3.
    ///
    /// Defaulted, with the two below: the three are one module linked twice along
    /// [`pigment`](Self::pigment), so what a space picks is the axis and not the
    /// shader.
    fn media_shader(&self) -> &'static stark_shaders::Media {
        stark_shaders::media(self.pigment())
    }
    /// The per-layer blend pass — §18.0.4. One isolated layer merged into the
    /// accumulator through a light-combining mode.
    ///
    /// A space needs its own build because blending happens in *light* (normalized CIE
    /// XYZ) while the targets hold channels, so the pass is bracketed by this space's
    /// conversion out and back. The algebra between them is shared.
    fn blend_shader(&self) -> &'static stark_shaders::Blend {
        stark_shaders::blend(self.pigment())
    }
    /// The **filter layer** pass — §21. The accumulator beneath a filter layer, read
    /// and rewritten.
    ///
    /// A space needs its own build for the reason [`blend_shader`](Self::blend_shader)
    /// does: a color adjustment is a statement about light and about perceived color,
    /// while the targets hold channels.
    fn filter_shader(&self) -> &'static stark_shaders::Filter {
        stark_shaders::filter(self.pigment())
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
/// where thickness is the difference between paint height and substrate height.
pub struct OkLabColorSpace;

impl ColorSpace for OkLabColorSpace {
    fn id(&self) -> ColorSpaceId {
        ColorSpaceId::Oklab
    }

    fn rgb_to_latent(&self, rgb: Srgb) -> Latent {
        let lin = [
            color::srgb_to_linear(rgb[0]),
            color::srgb_to_linear(rgb[1]),
            color::srgb_to_linear(rgb[2]),
        ];
        Latent {
            lat: color::linear_srgb_to_oklab(lin),
            // Oklab reproduces every sRGB color, so there is nothing left over.
            res: [0.0; 3],
        }
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

    fn stamp_shader(&self, lane: Lane) -> &'static stark_shaders::Stamp {
        stark_shaders::stamp(self.resid(), lane)
    }
}

/// **Mixbox** pigment-mixing space (§6.7). Colors are stored as Mixbox
/// latent pigment *concentrations* `(c0, c1, c2)` — the fourth, `c3 = 1 −
/// (c0+c1+c2)`, is derived — **plus the latent's residual in a third tile texture**.
/// The latent mixes linearly, so the ordinary premultiplied-"over" deposit *is*
/// Mixbox mixing (blue over yellow → green) and the blends and stamp law match
/// Oklab's; what differs is the third channel and the media pass, which evaluates
/// Mixbox's pigment polynomial and adds the residual back.
///
/// **The residual is not optional.** Four trained pigments do not span sRGB, so the
/// polynomial alone reaches neither black — whose concentrations render as `#383838`
/// — nor the saturated corners, where it is off by up to 0.39 (mean 0.05 over the
/// cube). No cheaper recovery exists: `rgb → c` is many-to-one, with up to 70 sRGB
/// colors sharing one quantized triple across 0.38 of the cube, so the residual is
/// not a function of the channels stored beside it.
///
/// Conversions use the vendored `mixbox` crate (CC BY-NC 4.0; `vendor/mixbox`),
/// which is why this whole space is behind the `mixbox` cargo feature: the licence is
/// non-commercial, so a build has to be able to leave it out entirely rather than
/// merely not reach it. [`ColorSpaceId::Mixbox`] still exists there — see
/// [`make`] for why the *id* cannot be gated even though the
/// implementation can.
#[cfg(feature = "mixbox")]
pub struct MixboxColorSpace;

#[cfg(feature = "mixbox")]
impl ColorSpace for MixboxColorSpace {
    fn id(&self) -> ColorSpaceId {
        ColorSpaceId::Mixbox
    }

    /// The residual's three components premultiplied by the same per-unit opacity the
    /// concentrations are, with that opacity duplicated into the fourth: not
    /// redundancy for its own sake, but because the fixed-function "over" reads *each*
    /// target's own alpha, so the target carrying the residual has to hold it too.
    fn resid_format(&self) -> Option<wgpu::TextureFormat> {
        Some(wgpu::TextureFormat::Rgba16Float)
    }

    fn rgb_to_latent(&self, rgb: Srgb) -> Latent {
        // The pigment space *is* the sRGB cube: the polynomial was trained on it and
        // the LUT is indexed by it, so a wide-gamut color is held to the cube here
        // (§6.5, §6.7) — the one door a document's color enters this space by.
        let rgb = rgb.map(|c| c.clamp(0.0, 1.0));
        // Mixbox latent = [c0, c1, c2, c3, residual…]. The concentrations and the
        // remainder `rgb − poly(c)` — which is what makes the round trip below exact
        // rather than approximate — are two halves of **one** evaluation.
        let z = mixbox::float_rgb_to_latent(&rgb);
        Latent {
            lat: [z[0], z[1], z[2]],
            res: [z[4], z[5], z[6]],
        }
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

    fn pigment(&self) -> Pigment {
        Pigment::Mixbox
    }

    fn stamp_shader(&self, lane: Lane) -> &'static stark_shaders::Stamp {
        // Deposit is premultiplied-over of the channels — the same law as Oklab's,
        // run over one more target.
        stark_shaders::stamp(self.resid(), lane)
    }
}
