//! What the frame's target *means* (§6.5): the transfer its texels are encoded in,
//! and how far above SDR white the surface can show. Paint is reflectance and never
//! exceeds white; the light glinting off it does (§6.3), and headroom is where that
//! goes instead of into the tonemap's roll-off.

use stark_model::color::Gamut;
use stark_shaders::mirror::display as d;

/// How a target's texels encode light — `lib/display.wesl`'s selector on the host.
///
/// Read off the surface configuration, never the format: an `Rgba16Float` canvas is
/// [`ExtendedSrgb`](Self::ExtendedSrgb) under the web's extended tone mapping and
/// [`Linear`](Self::Linear) on a native scRGB swapchain, and wgpu encodes for
/// neither.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Transfer {
    /// The sRGB OETF over `[0,1]` — every 8-bit sRGB target, and every export.
    #[default]
    Srgb,
    /// The sRGB OETF continued past 1 and mirrored through 0 — an fp16 canvas under
    /// the web's `"extended"` tone mapping.
    ExtendedSrgb,
    /// Linear light, `1.0` = SDR white — scRGB, the native HDR surface. Unbounded on
    /// sRGB primaries, so it carries a wide-gamut color in a negative channel.
    Linear,
    /// Display P3 primaries under the sRGB OETF over `[0,1]` — an 8-bit
    /// `display-p3` canvas: wide gamut, no headroom.
    DisplayP3,
    /// Display P3 primaries under the extended OETF — an fp16 `display-p3` canvas
    /// under extended tone mapping.
    ExtendedDisplayP3,
}

impl Transfer {
    /// The shader's selector, as the float lane the media and resolve uniforms carry.
    pub(super) fn lane(self) -> f32 {
        let id = match self {
            Self::Srgb => d::TRANSFER_SRGB,
            Self::ExtendedSrgb => d::TRANSFER_EXTENDED_SRGB,
            Self::Linear => d::TRANSFER_LINEAR,
            Self::DisplayP3 => d::TRANSFER_DISPLAY_P3,
            Self::ExtendedDisplayP3 => d::TRANSFER_EXTENDED_DISPLAY_P3,
        };
        id as f32
    }

    /// Whether a surface read in this transfer can show anything above SDR white.
    pub fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::ExtendedSrgb | Self::Linear | Self::ExtendedDisplayP3
        )
    }

    /// The gamut a surface read in this transfer can show (§6.5) — what a picker
    /// fits its wheel to. scRGB is unbounded and so wide; it is credited with P3,
    /// which is what the displays that offer it have.
    pub fn gamut(self) -> Gamut {
        match self {
            Self::Srgb | Self::ExtendedSrgb => Gamut::Srgb,
            Self::Linear | Self::DisplayP3 | Self::ExtendedDisplayP3 => Gamut::DisplayP3,
        }
    }
}

/// The display a render is presented on (§6.5). A view setting like
/// [`MediaParams`](super::MediaParams), per engine — and one an export never reads:
/// a render for a file is [`SDR`](Self::SDR) by the attachments it is drawn through
/// (`Engine::render_view`), so a file written from an HDR or wide-gamut session is
/// the picture an sRGB viewer sees.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Output {
    transfer: Transfer,
    /// Held to `[1, ∞)`: a knee below white would reshape colors the surface can show.
    headroom: f32,
}

impl Output {
    /// sRGB, nothing above white: what every engine opens on and every export is.
    pub const SDR: Self = Self {
        transfer: Transfer::Srgb,
        headroom: 1.0,
    };

    pub fn new(transfer: Transfer, headroom: f32) -> Self {
        let headroom = if headroom.is_finite() {
            headroom.max(1.0)
        } else {
            1.0
        };
        Self { transfer, headroom }
    }

    pub fn transfer(self) -> Transfer {
        self.transfer
    }

    /// Times SDR white the surface can show — the knee the tonemap compresses into.
    pub fn headroom(self) -> f32 {
        self.headroom
    }
}

impl Default for Output {
    fn default() -> Self {
        Self::SDR
    }
}

/// Whether `format` is one of the 8-bit targets: what an export is drawn into, what
/// the display dither is for (§6.5), and what can carry nothing above white whatever
/// it is asked for. The one list, so those three cannot disagree.
pub(super) fn is_eight_bit(format: wgpu::TextureFormat) -> bool {
    use wgpu::TextureFormat as F;
    matches!(format, F::Rgba8Unorm | F::Bgra8Unorm)
}

/// The 8-bit format a render that is read back to the CPU is drawn in (§15.6): the
/// screen's when that is already 8-bit — which keeps a browser's `Bgra8Unorm` export
/// as it was — else `Rgba8Unorm`.
pub(crate) fn export_format(screen: wgpu::TextureFormat) -> wgpu::TextureFormat {
    if is_eight_bit(screen) {
        screen
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headroom_is_held_to_at_least_white() {
        assert_eq!(Output::new(Transfer::Linear, 0.5).headroom(), 1.0);
        assert_eq!(Output::new(Transfer::Linear, f32::NAN).headroom(), 1.0);
        assert_eq!(Output::new(Transfer::Linear, 4.0).headroom(), 4.0);
        assert_eq!(Output::default(), Output::SDR);
    }

    #[test]
    fn a_transfer_says_what_its_surface_can_show() {
        assert!(!Transfer::Srgb.is_hdr());
        assert!(!Transfer::DisplayP3.is_hdr());
        assert!(Transfer::ExtendedDisplayP3.is_hdr() && Transfer::Linear.is_hdr());
        assert_eq!(Transfer::ExtendedSrgb.gamut(), Gamut::Srgb);
        assert_eq!(Transfer::DisplayP3.gamut(), Gamut::DisplayP3);
    }

    #[test]
    fn an_export_is_always_eight_bit() {
        use wgpu::TextureFormat as F;
        assert_eq!(export_format(F::Bgra8Unorm), F::Bgra8Unorm);
        assert_eq!(export_format(F::Rgba8Unorm), F::Rgba8Unorm);
        assert_eq!(export_format(F::Rgba16Float), F::Rgba8Unorm);
        assert_eq!(export_format(F::Rgb10a2Unorm), F::Rgba8Unorm);
        assert!(!is_eight_bit(F::Rgba16Float));
    }
}
