//! The icons the controls wear, as this toolkit draws them (§11, §25).
//!
//! **Which glyph a control wears is `stark_chrome::icons`', prose and all.** What is
//! here is the carrier, and it is a different one from the web's: over there an icon
//! is inlined into the DOM so `fill="currentColor"` resolves against the control,
//! here it is rasterized to an **alpha mask** and tinted by the element's own text
//! colour. The two arrive at the same behaviour by different routes, which is what
//! made the catalog worth sharing and the carrier not.
//!
//! # SVG, not a conversion pipeline
//!
//! wgpui vendors resvg and rasterizes on demand: `svg()` takes a path, the renderer
//! asks an [`AssetSource`] for the bytes, and `render_alpha_mask` keeps **only the
//! alpha channel** of what resvg produced. So the `fill` in the file is irrelevant —
//! what survives is coverage — and the mask goes into the R8 monochrome atlas, tinted
//! per draw. One file covers a resting chip, a lit chip and a dim one, exactly as it
//! does in the browser.
//!
//! That is why there is no build-time SVG-to-PNG step. A rasterized icon would have
//! to be baked at a size, and a panel that is laid out in logical px on a display
//! whose scale factor is a runtime fact has no such size to bake at.

use std::borrow::Cow;

use stark_chrome::icons::Icon;
use wgpui::{AssetSource, IntoElement, Result, SharedString, Styled, px, rgb, svg};

/// The path prefix an icon is asked for under.
///
/// A namespace rather than a bare stem, so a second kind of asset served through the
/// same source is a second prefix rather than a collision.
const PREFIX: &str = "icons/";

/// Serves the shipped icons to wgpui's SVG renderer.
///
/// Registered once on the `Application` (`crate::main`), which is what hands it to
/// the `SvgRenderer`. Nothing else in this app goes through an `AssetSource`: the
/// brush shapes and canvas substrates are content the *document* names and are
/// resolved by content id instead (`crate::assets`), which is a different question
/// with a different answer.
pub struct Icons;

impl AssetSource for Icons {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        let Some(stem) = path.strip_prefix(PREFIX) else {
            return Ok(None);
        };
        // `None` rather than an error for an unknown stem: the renderer treats a
        // missing asset as nothing to draw, and the catalog's own test is what rules
        // out a name with no file behind it.
        Ok(stark_chrome::icons::by_stem(stem).map(|text| Cow::Borrowed(text.as_bytes())))
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(stark_chrome::icons::ALL
            .iter()
            .map(|icon| SharedString::from(format!("{PREFIX}{}", icon.0)))
            .collect())
    }
}

/// The path the renderer asks for an icon under.
fn path(icon: Icon) -> SharedString {
    SharedString::from(format!("{PREFIX}{}", icon.0))
}

/// The size a glyph is drawn at beside a word, logical px.
///
/// Matched to the `text_xs` the chips wear rather than to the row's height: an icon
/// that outgrows its label reads as a picture with a caption, where these are labels
/// with a mark.
pub const SIZE: f32 = 14.0;

/// An icon at the ordinary size, in the colour the caller gives.
///
/// The colour is an argument rather than inherited: wgpui's `svg()` reads
/// `style.text.color` off its *own* element, so a glyph does not pick up the colour
/// of the row around it the way an inlined one does. Passing it is the whole of the
/// difference, and it is worth stating at each call site anyway — a lit chip and a
/// dim one differ in exactly this.
pub fn icon(mark: Icon, color: u32) -> impl IntoElement {
    svg()
        .path(path(mark))
        .size(px(SIZE))
        .flex_none()
        .text_color(rgb(color))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every icon the catalog names is served, and served as the file the catalog
    /// says — the whole of what this source owes.
    #[test]
    fn every_catalogued_icon_is_served() {
        for mark in stark_chrome::icons::ALL {
            let served = Icons
                .load(&path(*mark))
                .expect("the source does not fail")
                .expect("every catalogued icon is served");
            assert_eq!(served.as_ref(), mark.svg().unwrap().as_bytes());
        }
    }

    /// A path outside the prefix is not this source's, and a stem with no file is
    /// nothing to draw rather than a failure.
    #[test]
    fn anything_else_is_simply_not_here() {
        assert!(Icons.load("substrate/Linen.png").unwrap().is_none());
        assert!(Icons.load("icons/no-such-glyph").unwrap().is_none());
    }
}
