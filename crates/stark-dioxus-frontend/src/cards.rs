//! A gallery card's picture as this frontend shows one (§6.4, §6.6), the `data:` URL
//! every picture made from bytes travels in, and a swatch's color.
//!
//! What the picture *is* belongs to `stark_ui::assets::card`; only the DOM encoding (PNG,
//! base64, a `background-image`) is here.

use stark_ui::assets::{Card, Ink};

/// A card's `background-image` declaration, written as `none` when there is no picture
/// yet rather than omitted: an inline style merges per property, so a declaration left off
/// a reused node would keep another card's picture.
pub fn thumb_style(url: Option<&str>) -> String {
    match url {
        Some(url) => format!("background-image: url({url});"),
        None => "background-image: none;".to_string(),
    }
}

/// A swatch's `background` declaration for a straight sRGB color — a well, a loupe, the
/// hex field's patch. Percentages to two places, finer than a 10-bit channel.
pub fn swatch_style([r, g, b]: [f32; 3]) -> String {
    format!(
        "background: rgb({:.2}% {:.2}% {:.2}%);",
        r * 100.0,
        g * 100.0,
        b * 100.0
    )
}

/// What a `data:` URL says its bytes are.
#[derive(Clone, Copy)]
pub enum Mime {
    Png,
    Bmp,
}

impl Mime {
    fn as_str(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Bmp => "image/bmp",
        }
    }
}

/// `bytes` as a base64 `data:` URL of type `mime`.
pub fn bytes_url(mime: Mime, bytes: &[u8]) -> String {
    format!(
        "data:{};base64,{}",
        mime.as_str(),
        crate::base64::encode(bytes)
    )
}

/// A `data:` URL for `card`, or `None` if the encode failed.
pub fn data_url(card: Card) -> Option<String> {
    encode_png(card).map(|png| bytes_url(Mime::Png, &png))
}

/// `card` as PNG bytes, or `None` if the encode failed.
///
/// - **Coverage** is white ink with the field in alpha, so the panel shows through where a
///   stamp lays nothing; grayscale + alpha, since the ink is constant.
/// - **Height** is opaque grey: a substrate's lows are ground, not holes.
pub fn encode_png(card: Card) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, card.width, card.height);
        encoder.set_color(match card.ink {
            Ink::Coverage => png::ColorType::GrayscaleAlpha,
            Ink::Height => png::ColorType::Grayscale,
        });
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::High);
        let mut writer = encoder.write_header().ok()?;
        match card.ink {
            Ink::Coverage => {
                let pixels: Vec<u8> = card.texels.iter().flat_map(|&c| [u8::MAX, c]).collect();
                writer.write_image_data(&pixels).ok()?;
            }
            Ink::Height => writer.write_image_data(&card.texels).ok()?,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bytes_url_names_its_type_and_carries_base64() {
        assert_eq!(bytes_url(Mime::Png, b"foo"), "data:image/png;base64,Zm9v");
        assert_eq!(bytes_url(Mime::Bmp, b""), "data:image/bmp;base64,");
    }
}
