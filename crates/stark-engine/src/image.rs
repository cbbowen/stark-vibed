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

    /// Encode as a JPEG at `quality` (1–100) — the other export format (§15.6).
    ///
    /// The alpha channel is dropped: JPEG has nowhere to put it. So the *caller*
    /// decides what stands where nothing was painted, by rendering onto the
    /// substrate rather than onto transparency — encoding a transparent render
    /// would bake in whatever straight color sits under its alpha-0 texels.
    ///
    /// At 90 and above the encoder keeps full chroma (4:4:4); below that it
    /// subsamples, which smears exactly the colored edges a painting is made of.
    pub fn to_jpeg(&self, quality: u8) -> crate::Result<Vec<u8>> {
        use crate::error::ExportError;
        let (Ok(width), Ok(height)) = (u16::try_from(self.width), u16::try_from(self.height))
        else {
            return Err(ExportError::JpegTooLarge {
                width: self.width,
                height: self.height,
            }
            .into());
        };
        let mut out = Vec::new();
        let encoder = jpeg_encoder::Encoder::new(&mut out, quality);
        encoder
            .encode(&self.pixels, width, height, jpeg_encoder::ColorType::Rgba)
            .map_err(ExportError::EncodeJpeg)?;
        Ok(out)
    }
}

impl std::fmt::Debug for RgbaImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RgbaImage({}x{})", self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::RgbaImage;
    use crate::error::{EngineError, ExportError};

    #[test]
    fn jpeg_encodes_a_complete_stream() {
        let img = RgbaImage::new(2, 2, vec![255; 16]);
        let bytes = img.to_jpeg(90).expect("a 2x2 image must encode");
        // SOI at the front, EOI at the back — a decoder refuses anything less.
        assert_eq!(&bytes[..2], [0xFF, 0xD8], "JPEG must start with SOI");
        assert_eq!(
            &bytes[bytes.len() - 2..],
            [0xFF, 0xD9],
            "JPEG must end with EOI"
        );
    }

    #[test]
    fn jpeg_refuses_a_dimension_past_its_format() {
        // One texel past what 16 bits can say. PNG would take this; JPEG cannot.
        let img = RgbaImage::new(65536, 1, vec![0; 65536 * 4]);
        assert!(
            matches!(
                img.to_jpeg(90),
                Err(EngineError::Export(ExportError::JpegTooLarge {
                    width: 65536,
                    ..
                }))
            ),
            "a 65536-wide image must be refused as JpegTooLarge"
        );
    }
}
