//! A gallery card's picture, as this frontend puts one on screen (§6.4, §6.6).
//!
//! What the picture *is* — the canonical field an id names, reduced to a card's size,
//! and whether it is coverage or height — is `stark_ui::assets::card`. All that is
//! left here is the encoding, and it is a DOM idiom: a `background-image` takes a URL,
//! so the texels become a PNG and the PNG becomes base64. The native frontend hands
//! the same numbers to a texture instead, which is exactly why the numbers and not the
//! encoding are what moved down.

use stark_ui::assets::{Card, Ink};

/// A `data:` URL for `card`, or `None` if the encode failed.
///
/// Two channel layouts for the two readings, and neither is a style choice:
///
/// - **Coverage** is white ink with the field in the alpha channel, so the panel shows
///   through where a stamp lays nothing. Grayscale + alpha rather than RGBA because
///   the ink is a constant — the only channel carrying anything is alpha, which is
///   what makes a compressed mask a couple of kilobytes rather than a couple of
///   hundred.
/// - **Height** is opaque grey. A substrate has no gaps: its low ground is as much a
///   part of it as its high ground, and drawing the lows transparent would show a
///   canvas full of holes.
pub fn data_url(card: Card) -> Option<String> {
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
    Some(format!(
        "data:image/png;base64,{}",
        crate::platform::base64_encode(&out)
    ))
}
