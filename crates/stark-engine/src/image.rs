//! A simple CPU-side RGBA8 image, used for export and golden-image testing
//! (§9). Tightly packed, top-left origin, 4 bytes per pixel.

/// An 8-bit RGBA image read back from the GPU.
#[derive(Clone, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, row-major, no padding.
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        // In `u64` and not `u32`: an 8192² export is 268 MB and a caller may ask for
        // more, where `width * height * 4` wraps at 4 GB — so the check that exists to
        // catch a mis-sized buffer would pass on one, or panic in the multiply with
        // the wrong message.
        debug_assert_eq!(
            pixels.len() as u64,
            u64::from(width) * u64::from(height) * 4,
            "an RgbaImage is row-major with no padding: {width}x{height} needs four bytes a texel"
        );
        Self {
            width,
            height,
            pixels,
        }
    }

    /// The RGBA bytes at `(x, y)`. Panics if out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        // In `u64` like [`RgbaImage::new`]'s check, and for its reason: at the export
        // ceiling `max_export_dim` now reports (`max_texture_dimension_2d`, commonly
        // 32768), `y * width + x` crosses `u32::MAX` in the last rows — so the index
        // this returns would wrap to a pixel somewhere else in the image.
        let i = (u64::from(y) * u64::from(self.width) + u64::from(x)) as usize * 4;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    /// Build from bytes read back off a render target of `format`, normalizing the
    /// channel order.
    ///
    /// A readback returns the target's bytes *in the target's own order*, and this
    /// type is RGBA by definition, so a BGRA target has to be swizzled. That
    /// distinction is invisible on the machines the tests run on — they render to
    /// `Rgba8Unorm` — but a browser substrate is typically `Bgra8Unorm`, and the
    /// result was an exported PNG with red and blue swapped: salmon paper came out
    /// pale blue, orange paint came out blue. Green, black and white all survive a
    /// R↔B swap unchanged, which is what made it look like a color-space bug
    /// rather than a byte-order one.
    pub fn from_target_bytes(
        width: u32,
        height: u32,
        mut bytes: Vec<u8>,
        format: wgpu::TextureFormat,
    ) -> Self {
        use wgpu::TextureFormat::{Bgra8Unorm, Bgra8UnormSrgb};
        if matches!(format, Bgra8Unorm | Bgra8UnormSrgb) {
            for texel in bytes.as_chunks_mut::<4>().0 {
                texel.swap(0, 2);
            }
        }
        Self::new(width, height, bytes)
    }

    /// Encode as an RGBA PNG — the export format (§15.6).
    ///
    /// Alpha is written straight (un-premultiplied), which is what the media pass
    /// produces and what PNG stores, so a transparent export drops into any
    /// compositor without a fringe.
    pub fn to_png(&self) -> crate::Result<Vec<u8>> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, self.width, self.height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder
                .write_header()
                .map_err(crate::error::ExportError::Encode)?;
            writer
                .write_image_data(&self.pixels)
                .map_err(crate::error::ExportError::Encode)?;
        }
        Ok(out)
    }
}

impl std::fmt::Debug for RgbaImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RgbaImage({}x{})", self.width, self.height)
    }
}
