//! Content-addressed **pictures**: images placed into the document as paint (§23).
//!
//! [`AssetStore`](crate::assets::AssetStore)'s sibling, and deliberately a separate
//! store rather than a second map inside it. The two share a sentence — content
//! addressed by the hash of a decoded canonical form, so the log references 32 bytes
//! and the pixels ride beside it — and share nothing else: a brush shape is baked
//! into a prefix-τ volume and a coverage texture the moment it lands, where a picture
//! has **no GPU residency at all**. The tiles it becomes are built on the CPU
//! (`gpu::place`) and it is never bound as a texture, which is what leaves its size
//! bounded by the document (`MAX_PICTURE_DIM`) rather than by
//! `max_texture_dimension_2d`.
//!
//! So what is kept is the decoded picture and the bytes it was canonicalized to —
//! the first because replay needs it and the second because the save file and a peer
//! do. Behind an [`Arc`] in both directions: a placement is re-applied on every
//! replay and every history splice that crosses it, and a photograph is not a thing
//! to clone per lookup.

use std::collections::hash_map::{Entry, HashMap};
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::unpoisoned;

use stark_assetid::Picture;
use stark_model::AssetId;

/// One loaded picture: what replay reads, and what a bundle or a peer is given.
struct Held {
    picture: Arc<Picture>,
    /// The canonical RGBA PNG the id names, retained so the asset can be bundled
    /// into the save file (§8) and handed to a peer without re-encoding.
    bytes: Vec<u8>,
}

/// The pictures this session holds, by content id.
///
/// `Clone` (`Arc`-backed) so it rides in the `Action::Context` beside the tile pool
/// and the renderers (§5).
#[derive(Clone, Default)]
pub struct PictureStore {
    inner: Arc<Mutex<HashMap<AssetId, Held>>>,
}

impl PictureStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Import an image (any PNG this build decodes), returning its content id.
    ///
    /// The id names the **decoded, capped** picture rather than the file bytes, so
    /// two people who encoded the same photograph differently land on one id and a
    /// reload of the stored form lands back on it (§19). The stored bytes are
    /// re-encoded from that canonical form for the same reason.
    pub fn import(&self, png_bytes: &[u8]) -> Result<AssetId> {
        self.load(png_bytes, None)
    }

    /// Re-insert a picture from its saved (already-canonical RGBA PNG) bytes when
    /// loading a document or receiving one from a peer — keeps the bytes verbatim.
    ///
    /// The id is still **derived** rather than taken on trust, exactly as
    /// `surface::identify` derives a ground's: bytes that do not hash to the id that
    /// asked for them are a different picture, and a caller comparing what it got
    /// against what it wanted is the whole of how that is caught (§19).
    pub fn insert_bytes(&self, bytes: &[u8]) -> Result<AssetId> {
        self.load(bytes, Some(bytes.to_vec()))
    }

    fn load(&self, decode_from: &[u8], store_bytes: Option<Vec<u8>>) -> Result<AssetId> {
        // Decode, cap and hash are the identity contract's, not ours (§19).
        let picture = stark_assetid::picture(decode_from)?;
        let id = picture.id();
        let mut inner = unpoisoned(self.inner.lock());
        if let Entry::Vacant(slot) = inner.entry(id) {
            let bytes = match store_bytes {
                Some(b) => b,
                None => picture.encode()?,
            };
            slot.insert(Held {
                picture: Arc::new(picture),
                bytes,
            });
        }
        Ok(id)
    }

    /// The picture `id` names, if this session holds it.
    ///
    /// An `Arc`, so the caller reads it without holding the store's lock across a
    /// tile walk that visits sixty-five thousand texels a tile.
    pub fn get(&self, id: AssetId) -> Option<Arc<Picture>> {
        unpoisoned(self.inner.lock())
            .get(&id)
            .map(|held| held.picture.clone())
    }

    /// Whether `id` is loaded in this store.
    pub fn contains(&self, id: AssetId) -> bool {
        unpoisoned(self.inner.lock()).contains_key(&id)
    }

    /// The canonical PNG bytes of one picture, if loaded — what a save file bundles
    /// and what a peer is served.
    pub fn bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        unpoisoned(self.inner.lock())
            .get(&id)
            .map(|held| held.bytes.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32, tint: u8) -> Vec<u8> {
        Picture {
            width: w,
            height: h,
            pixels: (0..w * h)
                .flat_map(|i| [(i as u8).wrapping_add(tint), 40, 90, 255])
                .collect(),
        }
        .encode()
        .expect("encode")
    }

    /// Content addressing, which is the whole point of the store: the same picture
    /// imported twice is one entry under one id, and a different one is a different
    /// id. This is what makes placing the same reference photo on two layers — or a
    /// peer that already holds it — cost nothing.
    #[test]
    fn the_same_picture_imported_twice_is_one_entry() {
        let store = PictureStore::new();
        let a = store.import(&png(8, 6, 0)).expect("import");
        let b = store.import(&png(8, 6, 0)).expect("import again");
        let other = store.import(&png(8, 6, 7)).expect("a different picture");
        assert_eq!(a, b);
        assert_ne!(a, other);
        assert_eq!(unpoisoned(store.inner.lock()).len(), 2);
    }

    /// The stored bytes round-trip to the same id, which is what keeps a document's
    /// own bundle matching the ids its log references (§19).
    #[test]
    fn the_stored_bytes_reload_to_the_same_id() {
        let store = PictureStore::new();
        let id = store.import(&png(9, 5, 3)).expect("import");
        let bytes = store.bytes(id).expect("held");

        let second = PictureStore::new();
        assert_eq!(second.insert_bytes(&bytes).expect("reload"), id);
        assert_eq!(
            second.get(id).expect("held").pixels,
            store.get(id).unwrap().pixels
        );
    }

    /// A store that does not hold a picture says so, rather than answering with an
    /// empty one — the distinction `apply` turns into "leave the layer empty and
    /// warn" instead of "place nothing and look successful" (§23).
    #[test]
    fn an_absent_picture_is_absent() {
        let store = PictureStore::new();
        let id = AssetId([9; 32]);
        assert!(!store.contains(id));
        assert!(store.get(id).is_none());
        assert!(store.bytes(id).is_none());
    }
}
