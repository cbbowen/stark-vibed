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
///
/// Generic in what a picture *is*, because the two frontends carry one differently: a
/// data URL for a `background-image`, a texture handle for an element tree. The cache
/// policy is the part worth agreeing on, and it is all here.
pub struct Thumbs<T>(Mutex<Vec<(AssetId, T)>>);

impl<T: Clone> Thumbs<T> {
    /// `const`, because every consumer of this declares one as a `static`: a
    /// library's cache is per-library and lives as long as the app does.
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    /// The remembered picture of `id`, as a value — so a caller's miss path is not
    /// holding the lock while it decodes.
    pub fn get(&self, id: AssetId) -> Option<T> {
        let thumbs = self.entries();
        thumbs
            .iter()
            .find(|(k, _)| *k == id)
            .map(|(_, u)| u.clone())
    }

    /// Remember `picture` as the picture of `id`, replacing whatever was under it.
    ///
    /// Replace rather than push: a caller's miss path is a decode, a reduce and an
    /// encode, and the web frontend's runs inside a render body — so two renders that
    /// both miss both arrive here with the same id, and an unconditional push would
    /// grow the list a duplicate per frame until the first copy shadowed the rest.
    pub fn put(&self, id: AssetId, picture: T) {
        let mut thumbs = self.entries();
        match thumbs.iter_mut().find(|(k, _)| *k == id) {
            Some((_, held)) => *held = picture,
            None => thumbs.push((id, picture)),
        }
    }

    /// The list, **poisoning ignored**.
    ///
    /// There is no invariant here for a panic to have broken: a thumbnail is a pure
    /// function of the id and of which library is drawing it (see the type doc), so
    /// the worst a torn write leaves is a picture that is still correct. Honouring
    /// the poison would disable the cache permanently — every gallery card a full
    /// decode-plus-reduce-plus-encode on every render, for the rest of the run, from
    /// one panic anywhere else in the process.
    fn entries(&self) -> std::sync::MutexGuard<'_, Vec<(AssetId, T)>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl<T: Clone> Default for Thumbs<T> {
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

    /// One id, one entry, however many times it is remembered — the gallery's miss
    /// path is a render body, so the same id arrives here again before the first
    /// answer has been shown.
    #[test]
    fn remembering_a_picture_twice_keeps_one_of_it() {
        let thumbs: Thumbs<u8> = Thumbs::new();
        let id = AssetId([1; 32]);
        thumbs.put(id, 7);
        thumbs.put(id, 9);
        assert_eq!(thumbs.get(id), Some(9), "the later picture is the one held");
        assert_eq!(thumbs.entries().len(), 1, "and it did not grow a duplicate");
    }

    /// A panic elsewhere in the process does not cost the cache. There is no
    /// invariant a torn write could have broken — see [`Thumbs::entries`] — and
    /// honouring the poison would mean re-encoding every card on every render for
    /// the rest of the run.
    #[test]
    fn a_poisoned_lock_still_answers() {
        let thumbs: Thumbs<u8> = Thumbs::new();
        let id = AssetId([2; 32]);
        thumbs.put(id, 3);
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = thumbs.0.lock().expect("not poisoned yet");
            panic!("something else went wrong while the lock was held");
        }));
        assert!(poisoned.is_err(), "the panic was the point");
        assert!(thumbs.0.is_poisoned());
        assert_eq!(thumbs.get(id), Some(3));
        thumbs.put(AssetId([4; 32]), 5);
        assert_eq!(thumbs.get(AssetId([4; 32])), Some(5));
    }
}
