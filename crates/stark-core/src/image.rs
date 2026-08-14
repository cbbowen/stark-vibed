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
        debug_assert_eq!(pixels.len(), (width * height * 4) as usize);
        Self {
            width,
            height,
            pixels,
        }
    }

    /// The RGBA bytes at `(x, y)`. Panics if out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
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
    /// `Rgba8Unorm` — but a browser surface is typically `Bgra8Unorm`, and the
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
                .map_err(|e| crate::EngineError::Export(e.to_string()))?;
            writer
                .write_image_data(&self.pixels)
                .map_err(|e| crate::EngineError::Export(e.to_string()))?;
        }
        Ok(out)
    }
}

impl std::fmt::Debug for RgbaImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RgbaImage({}x{})", self.width, self.height)
    }
}
