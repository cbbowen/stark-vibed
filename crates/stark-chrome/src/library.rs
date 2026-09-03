//! What a **browser-held asset library** is made of — the parts both of them share.
//!
//! There are two, and both are a frontend's (`shapes` and `substrates` in the web one):
//! the brush stamps a user brought in (§6.6) and the canvas substrates (§6.4). They are the same object twice over. An
//! entry is a canonical image keyed by its content id, the id is the whole of the
//! reference, the rows go to `localStorage` and the bytes to the blob store beside
//! them (§25.6), and each is shown as a card the size of a fingernail.
//!
//! What differs is what the card is a **picture of**, and that is the one thing left
//! in each module: a stamp's card is the coverage it will lay, a substrate's is the
//! height field it will bite. Neither is derivable from the other — the same grayscale
//! PNG canonicalizes to the same id under both readings, so a shared cache would hand
//! one library the other's picture. Hence [`Thumbs`], which is a cache each declares
//! *one of*, rather than a table this module owns.
//!
//! Everything here is off the browser: it is arithmetic over decoded images and a
//! `Mutex`, and it is tested on the host like the rest of the crate.

use std::sync::Mutex;

use stark_model::AssetId;

/// The longest edge of a gallery thumbnail, in texels: 2× the 52-px card that shows
/// it, so it stays crisp on a dense display, and two orders of magnitude off the
/// [`MAX_SHAPE_DIM`](stark_assetid::MAX_SHAPE_DIM) or
/// [`MAX_SUBSTRATE_DIM`](stark_assetid::MAX_SUBSTRATE_DIM) the field itself may run to. That
/// difference is the whole reason [`reduce`] exists — a full-size map rides into the
/// DOM as half a megabyte of base64, per card, for a picture the size of a fingernail.
pub const THUMB_DIM: u32 = 128;

/// One library's gallery thumbnails, keyed by the content id each is a picture of.
///
/// An id *names* a field (§19), so a thumbnail is a pure function of the id and of
/// which library is drawing it: there is no invalidation to get wrong and nothing to
/// evict. Scanned by `PartialEq` over a `Vec` rather than hashed, like
/// `Renderer::builtins` and the frontend's `thumbs::lookup` — the list is as long as the user
/// has entries.
///
/// **One per library, never shared.** See the module note: two readings of one PNG
/// land on one id and two different pictures.
pub struct Thumbs(Mutex<Vec<(AssetId, String)>>);

impl Thumbs {
    /// `const`, because every consumer of this declares one as a `static`: a
    /// library's cache is per-library and lives as long as the app does.
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    /// The remembered data URL for `id`, as a value — so a caller's miss path is not
    /// holding the lock while it decodes.
    pub fn get(&self, id: AssetId) -> Option<String> {
        let thumbs = self.0.lock().ok()?;
        thumbs
            .iter()
            .find(|(k, _)| *k == id)
            .map(|(_, u)| u.clone())
    }

    /// Remember `url` as the picture of `id`. A poisoned lock simply forgets it: the
    /// cache is a saving, and re-encoding a thumbnail is cheaper than taking a panel
    /// down over one.
    pub fn put(&self, id: AssetId, url: String) {
        if let Ok(mut thumbs) = self.0.lock() {
            thumbs.push((id, url));
        }
    }
}

impl Default for Thumbs {
    /// Deferring to [`new`](Self::new) rather than deriving, which would drop the
    /// `const` the `static`s need. Here because this is a library's public type now
    /// and a `new` without a `Default` is a surprise in one.
    fn default() -> Self {
        Self::new()
    }
}

/// Box-average a single-channel field down by the smallest integer factor that brings
/// both edges within [`THUMB_DIM`]; `factor == 1` returns it unchanged.
///
/// The same reduction `stark_assetid` applies at the identity cap, written again here
/// rather than reached for: that one is part of what an id *means* (§19) and is frozen
/// at its own limit, and this one is a picture for a 52-px card. Sharing the function
/// would tie a stylesheet's idea of a thumbnail to the file format's idea of an asset.
pub fn reduce(field: stark_assetid::Canonical) -> stark_assetid::Canonical {
    let (w, h) = (field.width, field.height);
    let factor = w.div_ceil(THUMB_DIM).max(h.div_ceil(THUMB_DIM)).max(1);
    if factor == 1 {
        return field;
    }
    let (nw, nh) = (w / factor, h / factor);
    let area = factor * factor;
    let mut texels = vec![0u8; (nw * nh) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    let i = ((y * factor + dy) * w + (x * factor + dx)) as usize;
                    sum += field.texels[i] as u32;
                }
            }
            texels[(y * nw + x) as usize] = (sum / area) as u8;
        }
    }
    stark_assetid::Canonical {
        width: nw,
        height: nh,
        texels,
    }
}

/// A card's `background-image` declaration — written out as `none` when there is no
/// picture yet (a built-in still fetching, bytes that would not decode) rather than
/// omitted.
///
/// An inline style merges per property, so a declaration left off a reused node is
/// stranded at its last value instead of cleared — which would leave one card wearing
/// another's picture.
pub fn thumb_style(url: Option<&str>) -> String {
    match url {
        Some(url) => format!("background-image: url({url});"),
        None => "background-image: none;".to_string(),
    }
}

/// A human name from a picked file's name: the stem, tidied. `fallback` is what an
/// empty one becomes, and it is the caller's because it says which library was being
/// added to.
pub fn display_name(file_name: &str, fallback: &str) -> String {
    let stem = file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(file_name)
        .trim();
    if stem.is_empty() {
        fallback.to_string()
    } else {
        stem.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field arrives capped at the identity contract's own limit and leaves here at
    /// a card's size — the reduction is what keeps a data URL out of the DOM at a
    /// hundred times the size the picture is shown at.
    #[test]
    fn a_thumbnail_is_reduced_to_the_size_the_card_shows_it_at() {
        let full = stark_assetid::Canonical {
            width: 1024,
            height: 512,
            texels: vec![128; 1024 * 512],
        };
        let small = reduce(full);
        assert_eq!((small.width, small.height), (128, 64));
        assert!(
            small.texels.iter().all(|&t| t == 128),
            "a flat field stays flat"
        );
    }

    /// A field already inside the cap is handed back untouched, factor and all — the
    /// `factor == 1` early return, which is what keeps a small stamp crisp instead of
    /// putting it through an averaging pass that could only blur it.
    #[test]
    fn a_field_already_small_enough_is_left_alone() {
        let small = stark_assetid::Canonical {
            width: 64,
            height: 64,
            texels: (0..64 * 64).map(|i| i as u8).collect(),
        };
        let out = reduce(small.clone());
        assert_eq!((out.width, out.height), (64, 64));
        assert_eq!(out.texels, small.texels);
    }

    /// The stem, and the caller's word when there is no stem to take.
    #[test]
    fn a_name_comes_off_the_file_and_falls_back_to_the_callers_word() {
        assert_eq!(display_name("Coarse Linen.png", "x"), "Coarse Linen");
        assert_eq!(display_name("no-extension", "x"), "no-extension");
        assert_eq!(
            display_name(".png", "Imported substrate"),
            "Imported substrate"
        );
    }
}
