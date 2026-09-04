//! What the frame's target *means* (§6.5): the transfer its texels are encoded in,
//! and how far above SDR white the surface can show. Paint is reflectance and never
//! exceeds white; the light glinting off it does (§6.3), and headroom is where that
//! goes instead of into the tonemap's roll-off.

use stark_shaders::mirror::display as d;

/// How a target's texels encode light — `lib/display.wesl`'s selector on the host.
///
/// Read off the surface configuration, never the format: an `Rgba16Float` canvas is
/// [`ExtendedSrgb`](Self::ExtendedSrgb) under the web's extended tone mapping and
/// [`Linear`](Self::Linear) on a native scRGB swapchain, and wgpu encodes for
/// neither.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Transfer {
    /// The sRGB OETF over `[0,1]` — every 8-bit target, and every export.
    #[default]
    Srgb,
    /// The sRGB OETF continued past 1 and mirrored through 0 — an fp16 canvas under
    /// the web's `"extended"` tone mapping.
    ExtendedSrgb,
    /// Linear light, `1.0` = SDR white — scRGB, the native HDR surface.
    Linear,
}

impl Transfer {
    /// The shader's selector, as the float lane the media and resolve uniforms carry.
    pub(super) fn lane(self) -> f32 {
        let id = match self {
            Self::Srgb => d::TRANSFER_SRGB,
            Self::ExtendedSrgb => d::TRANSFER_EXTENDED_SRGB,
            Self::Linear => d::TRANSFER_LINEAR,
        };
        id as f32
    }
}

/// The display a render is presented on (§6.5). A view setting like
/// [`MediaParams`](super::MediaParams), per engine — and one an 8-bit render never
/// reads: it is [`SDR`](Self::SDR) by construction (`Compositor::render`), so a file
/// written from an HDR session is the picture an SDR viewer sees.
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
/// the display dither is for (§6.5), and what is rendered [`Output::SDR`] whatever
/// the screen is showing. The one list, so those three cannot disagree.
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
    fn an_export_is_always_eight_bit() {
        use wgpu::TextureFormat as F;
        assert_eq!(export_format(F::Bgra8Unorm), F::Bgra8Unorm);
        assert_eq!(export_format(F::Rgba8Unorm), F::Rgba8Unorm);
        assert_eq!(export_format(F::Rgba16Float), F::Rgba8Unorm);
        assert_eq!(export_format(F::Rgb10a2Unorm), F::Rgba8Unorm);
        assert!(!is_eight_bit(F::Rgba16Float));
    }
}
