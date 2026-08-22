//! The user's custom brush-shape library (§6.6, build-order step 11).
//!
//! An entry is a **canonical grayscale PNG keyed by content id** — exactly the
//! bytes the engine's `AssetStore` holds, bundles into save files, and serves
//! to session peers, so one representation flows everywhere. The library
//! itself is frontend state: the engine's store is per-document (populated on
//! import and load), while the library follows this browser across documents
//! (the `identity` bargain — where storage is unavailable it degrades to a
//! per-session library and breaks nothing).
//!
//! **The library is kept in two stores, split down the middle of an entry**
//! ([`StoredShape`] and [`ShapeEntry`], §25.6). The name and the id are text and go
//! to `localStorage` with the settings and the chord table; the PNG is bytes and goes
//! to IndexedDB under the id that names it. They were one record once, the PNG
//! base64'd inline — which put half a megabyte per imported shape into a five-megabyte
//! text store that ten other records are also spending, and re-encoded the whole
//! library on the painting thread every time one changed. `crate::storage` has the
//! rest of that argument. What it means here is that reading the library is a fetch
//! ([`load`] is `async` and awaited once at start), and that every write puts the
//! bytes and the row down in the order that leaves the two agreeing.
//!
//! Import runs through [`crate::platform::normalize_shape_image`], so anything
//! the browser can decode becomes a shape, downscaled to the engine cap and
//! auto-inverted when it reads as dark ink on light paper. The engine then
//! canonicalizes (grayscale, content-hashed), and the canonical bytes — not
//! the user's file — are what the library keeps.
//!
//! Every task here is `spawn_forever` and every result lands in root-owned
//! signals: imports are started from the brush editor, and closing a modal
//! must not cancel or dangle them (see `collab.rs` for the same rule).
//!
//! The gallery's picture of a shape is [`thumbnail`], and it serves the app's
//! own stamps as well as the library's: a card is drawn from the **coverage** an
//! id names rather than from the file it was authored from, so where the author
//! put that coverage — in alpha, or in the value over a black ground — stops
//! being something the app has to be careful about.

use std::sync::Mutex;

use dioxus::dioxus_core::spawn_forever;
use dioxus::prelude::*;
use stark_model::AssetId;
use stark_model::document::BrushShape;

use crate::platform::{base64_encode, normalize_shape_image};
use crate::state::{AppState, update_brush};
use crate::storage::{self, Store};

/// One custom shape in the library, **with its bytes in hand**.
///
/// Not the stored type any more, and the split is the point: the row is text and the
/// PNG is not, so they are kept in two stores and this is what they add up to
/// ([`StoredShape`], `crate::storage`). Constructing one means the bytes are here —
/// [`load`] drops a row whose blob is gone rather than admitting a byte-less entry —
/// so `ensure` and `thumbnail` have nothing to check and there is no half-loaded
/// shape for the gallery to draw a blank card for.
#[derive(Clone, PartialEq)]
pub struct ShapeEntry {
    /// Display name, defaulted from the imported file's stem.
    pub name: String,
    /// Canonical grayscale PNG (what the engine stores under `id`).
    pub png: Vec<u8>,
    /// Content id of `png`. If an engine upgrade ever re-canonicalizes differently,
    /// [`select`] heals the entry from the id the import actually returns.
    pub id: AssetId,
}

/// One row of the stored library: **a name and an id, and no bytes at all.**
///
/// The id is the whole of the reference — it *names* the PNG (§19), which is what
/// makes a row this small enough to keep in a text store that the settings, the chord
/// table and the tour's ledger are also spending. The bytes it names are a
/// [`storage::Blob`] under the same record's key; see `crate::storage` for what
/// putting them there bought and what it cost.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StoredShape {
    name: String,
    #[serde(with = "storage::hex")]
    id: AssetId,
}

impl storage::Entry for StoredShape {
    const STORE: Store = Store::Shapes;
}

impl storage::Blob for ShapeEntry {
    const STORE: Store = Store::Shapes;
}

/// The longest edge of a gallery thumbnail, in texels: 2× the 52-px card that
/// shows it, so it stays crisp on a dense display, and two orders of magnitude
/// off the [`MAX_SHAPE_DIM`](stark_assetid::MAX_SHAPE_DIM) the mask itself may
/// run to. That difference is the whole reason the reduction exists — a full-size
/// mask rides into the DOM as half a megabyte of base64, per card, for a picture
/// the size of a fingernail.
const THUMB_DIM: u32 = 128;

/// Gallery thumbnails, keyed by the content id each one is a picture of.
///
/// An id *names* a coverage field (§19), so a thumbnail is a pure function of the
/// id: there is no invalidation to get wrong and nothing to evict. Scanned by
/// `PartialEq` over a `Vec` rather than hashed, like `Renderer::builtins` and
/// `crate::thumbs::lookup` — the list is as long as the user has shapes.
static THUMBS: Mutex<Vec<(AssetId, String)>> = Mutex::new(Vec::new());

/// A `background-image` data URL showing what the shape `id` names **covers**,
/// whichever way its source encoded that.
///
/// A shape says its coverage in luminance × alpha (`stark_assetid::coverage`), so
/// white on opaque black and an alpha cut-out are two spellings of one field, and
/// the engine reads them as one. The gallery read neither. A built-in was drawn
/// from the file it was authored from, so its card carried whatever ground the
/// author had left behind it — the app's own stamps only looked right because they
/// had been drawn with alpha on purpose, which made that a requirement nothing
/// states and nothing checks. An imported shape was drawn from its canonical
/// bytes, which are grayscale and therefore *always* opaque: every one of them
/// arrived as a black tile with a shape faintly on it, however its source had been
/// authored.
///
/// So the card is drawn from the coverage instead: white ink, the coverage field
/// in the alpha channel. Every spelling lands on the same picture because they
/// already land on the same id, and what the gallery shows is what the stroke
/// will stamp.
///
/// Bytes come from the engine's store first and the library second — [`ensure`]'s
/// order, for [`ensure`]'s reason: a built-in is only ever in the engine, and a
/// shape imported in an earlier session is only ever in the library until it is
/// picked. `None` while a built-in's fetch is still in flight (the same moment its
/// card is not yet clickable), or if the bytes do not decode.
pub fn thumbnail(state: AppState, id: AssetId) -> Option<String> {
    if let Some(url) = cached(id) {
        return Some(url);
    }
    let bytes = {
        let renderer = state.renderer;
        let guard = renderer.read();
        guard.as_ref().and_then(|r| r.asset_bytes(id))
    }
    .or_else(|| {
        state
            .shapes
            .entries
            .read()
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.png.clone())
    })?;
    let url = encode_thumb(&bytes)?;
    if let Ok(mut thumbs) = THUMBS.lock() {
        thumbs.push((id, url.clone()));
    }
    Some(url)
}

/// [`THUMBS`] under its lock, as a value — so the miss path below is not holding it
/// while it decodes.
fn cached(id: AssetId) -> Option<String> {
    let thumbs = THUMBS.lock().ok()?;
    thumbs
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, u)| u.clone())
}

/// Decode `png` to its coverage, reduce it to a thumbnail, and encode that as white
/// ink with the coverage in alpha.
///
/// **Grayscale + alpha**, because the ink is a constant: the only channel carrying
/// anything is the alpha one, which is what makes a compressed mask a couple of
/// kilobytes rather than a couple of hundred. White rather than the panel's ink so
/// the picture is the shape's and not the stylesheet's — a card sets the ground it
/// sits on, and this is the paint.
fn encode_thumb(png: &[u8]) -> Option<String> {
    let stark_assetid::Canonical {
        width,
        height,
        texels,
    } = reduce(stark_assetid::coverage(png).ok()?);
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::GrayscaleAlpha);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::High);
        let mut writer = encoder.write_header().ok()?;
        let pixels: Vec<u8> = texels.iter().flat_map(|&c| [u8::MAX, c]).collect();
        writer.write_image_data(&pixels).ok()?;
    }
    Some(format!("data:image/png;base64,{}", base64_encode(&out)))
}

/// Box-average a coverage field down by the smallest integer factor that brings both
/// edges within [`THUMB_DIM`]; `factor == 1` returns it unchanged.
///
/// The same reduction `stark_assetid` applies at the identity cap, written again here
/// rather than reached for: that one is part of what an id *means* (§19) and is
/// frozen at its own limit, and this one is a picture for a 52-px card. Sharing the
/// function would tie a stylesheet's idea of a thumbnail to the file format's idea of
/// an asset.
fn reduce(cov: stark_assetid::Canonical) -> stark_assetid::Canonical {
    let (w, h) = (cov.width, cov.height);
    let factor = w.div_ceil(THUMB_DIM).max(h.div_ceil(THUMB_DIM)).max(1);
    if factor == 1 {
        return cov;
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
                    sum += cov.texels[i] as u32;
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

/// Populate the library signal from storage. Called once at app start.
///
/// Two reads, because the library is kept in two stores: the rows out of the text
/// one, then their PNGs out of the blob one in a single batch (`crate::storage`).
/// A row whose bytes are not there is **dropped**, and the library written back
/// without it — IndexedDB is evictable under storage pressure, so that is a state to
/// expect rather than one that only follows a crash, and an entry that cannot be
/// painted with is not a shape. Leaving it in the list would show a card that draws
/// nothing and reports "failed to load" every time it is clicked.
///
/// Awaited before the first thing that resolves a stamp id — `presets::apply_first`,
/// in `main` — which is what the ordering there is for.
pub async fn load(state: AppState) {
    let rows = storage::load_list::<StoredShape>().unwrap_or_default();
    let ids: Vec<AssetId> = rows.iter().map(|r| r.id).collect();
    let blobs = storage::blob_load_all::<ShapeEntry>(&ids).await;

    let kept: Vec<ShapeEntry> = rows
        .into_iter()
        .zip(blobs)
        .filter_map(|(row, png)| {
            png.map(|png| ShapeEntry {
                name: row.name,
                png,
                id: row.id,
            })
        })
        .collect();
    if kept.len() != ids.len() {
        tracing::warn!(
            "{} shape(s) had lost their image and were dropped from the library",
            ids.len() - kept.len()
        );
        persist(&kept);
    }

    let mut entries = state.shapes.entries;
    entries.set(kept);
}

/// Import an image file as a new shape: normalize in the browser, canonicalize
/// in the engine, add to the library, select it, and seed any live session so
/// peers can fetch it the moment a stroke (or live preview) references it.
pub fn import_file(state: AppState, file_name: String, bytes: Vec<u8>) {
    let mut notice = state.shapes.notice;
    notice.set(None);
    spawn_forever(async move {
        let (png, inverted) = match normalize_shape_image(bytes).await {
            Ok(v) => v,
            Err(e) => {
                notice.set(Some(format!("Couldn't import “{file_name}”: {e}.")));
                return;
            }
        };
        let imported = {
            let renderer = state.renderer;
            let guard = renderer.read();
            match guard.as_ref() {
                Some(r) => r
                    .import_brush_id(&png)
                    .map(|id| (id, r.asset_bytes(id).unwrap_or(png))),
                None => Err("the canvas is still starting".to_string()),
            }
        };
        let (id, canonical) = match imported {
            Ok(v) => v,
            Err(e) => {
                notice.set(Some(format!("Couldn't import “{file_name}”: {e}.")));
                return;
            }
        };

        let name = display_name(&file_name);
        let mut entries = state.shapes.entries;
        let known = entries.read().iter().any(|e| e.id == id);
        if !known {
            // Bytes before the row that names them (`storage::blob_save`): the other
            // order can leave a library pointing at a shape that was never stored.
            storage::blob_save::<ShapeEntry>(id, &canonical).await;
            entries.write().push(ShapeEntry {
                name: name.clone(),
                png: canonical.clone(),
                id,
            });
            persist(&entries.read());
        }
        seed_session(state, id, canonical);
        update_brush(state, |b| b.shape = BrushShape::Stamp(id));
        notice.set(match (known, inverted) {
            (true, _) => Some(format!(
                "“{name}” is already in your library — selected it."
            )),
            (false, true) => Some(format!(
                "“{name}” read as dark ink on light paper, so it was inverted — white now paints."
            )),
            (false, false) => None,
        });
    });
}

/// Import files dropped onto the shape gallery. Each file reads and imports
/// independently; when several are dropped the last import to finish holds the
/// brush (arbitrary but harmless — every one lands in the library).
pub fn import_dropped(state: AppState, files: Vec<dioxus::html::FileData>) {
    for file in files {
        spawn_forever(async move {
            let name = file.name();
            match file.read_bytes().await {
                Ok(bytes) => import_file(state, name, bytes.to_vec()),
                Err(e) => {
                    let mut notice = state.shapes.notice;
                    notice.set(Some(format!("Couldn't read “{name}”: {e}.")));
                }
            }
        });
    }
}

/// An entry id as hex — a stable rsx `key` for gallery items, and the same spelling
/// the entry is stored under (`storage::hex`).
pub fn id_hex(id: AssetId) -> String {
    id.to_hex()
}

/// Make `id` the active brush shape, importing the entry's bytes into the
/// engine first when this document hasn't seen it yet (content-addressing
/// makes a repeat import free).
pub fn select(state: AppState, id: AssetId) {
    if let Some(actual) = ensure(state, id) {
        update_brush(state, |b| b.shape = BrushShape::Stamp(actual));
    }
}

/// Make sure `id` is usable as a stamp: its bytes in this document's engine and
/// seeded to any live session, returning the id to reference it by — healed
/// when the stored id predates a canonicalization change. `None` when the bytes
/// are nowhere to be found (neither the engine nor the library has them), the
/// canvas isn't up yet, or the import failed. Shared by the gallery's [`select`]
/// and by preset application (`crate::presets`), which resurrects stamp shapes
/// from snapshots taken in earlier documents.
pub fn ensure(state: AppState, id: AssetId) -> Option<AssetId> {
    // Already in this document's engine (imported here, or arrived with a loaded
    // file): just make sure a live session can serve it.
    let engine_bytes = {
        let renderer = state.renderer;
        let guard = renderer.read();
        guard.as_ref().and_then(|r| r.asset_bytes(id))
    };
    if let Some(bytes) = engine_bytes {
        seed_session(state, id, bytes);
        return Some(id);
    }

    let entry = state
        .shapes
        .entries
        .read()
        .iter()
        .find(|e| e.id == id)
        .cloned()?;
    let imported = {
        let renderer = state.renderer;
        let guard = renderer.read();
        guard.as_ref()?.import_brush_id(&entry.png)
    };
    let actual = match imported {
        Ok(actual) => actual,
        Err(e) => {
            let mut notice = state.shapes.notice;
            notice.set(Some(format!("“{}” failed to load: {e}.", entry.name)));
            return None;
        }
    };
    if actual != entry.id {
        // The stored id predates a canonicalization change; heal it in place. The
        // signal moves now — the caller is about to paint with `actual` — but the two
        // stores are rewritten in a task, in the order `storage::blob_save` states:
        // the bytes under their new name, then the row, then the old name dropped.
        let mut entries = state.shapes.entries;
        if let Some(e) = entries.write().iter_mut().find(|e| e.id == entry.id) {
            e.id = actual;
        }
        let (stale, bytes) = (entry.id, entry.png.clone());
        spawn_forever(async move {
            storage::blob_save::<ShapeEntry>(actual, &bytes).await;
            persist(&entries.read());
            storage::blob_remove::<ShapeEntry>(stale).await;
        });
    }
    seed_session(state, actual, entry.png);
    Some(actual)
}

/// Drop an entry from the library. Paint already on canvases is untouched —
/// the engine's per-document store keeps every imported asset, and save files
/// bundle whatever strokes reference. A brush currently holding the removed
/// shape falls back to the round tip.
pub fn remove(state: AppState, id: AssetId) {
    let mut entries = state.shapes.entries;
    entries.write().retain(|e| e.id != id);
    persist(&entries.read());
    // The row first, then the bytes (`storage::blob_save`): a crash between the two
    // strands some bytes, which costs space; the other order strands the *row*, which
    // costs a shape that cannot be painted with.
    spawn_forever(async move { storage::blob_remove::<ShapeEntry>(id).await });

    let selected = state
        .obs
        .peek()
        .as_ref()
        .map(|o| o.brush.shape == BrushShape::Stamp(id))
        .unwrap_or(false);
    if selected {
        update_brush(state, |b| b.shape = BrushShape::default());
    }
}

/// Register the shape with a live session's mirror so peers can fetch it by
/// hash. A no-op when solo; idempotent when repeated (content-addressed).
fn seed_session(state: AppState, id: AssetId, bytes: Vec<u8>) {
    if let Some(broadcaster) = state
        .collab
        .session
        .read()
        .as_ref()
        .map(|s| s.broadcaster())
    {
        broadcaster.add_content(stark_net::AssetNeed::Brush(id), bytes);
    }
}

/// A human name from a picked file's name: the stem, tidied.
fn display_name(file_name: &str) -> String {
    let stem = file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(file_name)
        .trim();
    if stem.is_empty() {
        "Imported shape".to_string()
    } else {
        stem.to_string()
    }
}

// --- persistence ----------------------------------------------------------
//
// The library is kept in two stores and this is the seam between them: the rows here,
// the bytes at each site that changes one (`storage::blob_save`, and the order it
// states). Everything else — the format, the key, the skip-a-damaged-row rule and the
// failure policy — is `crate::storage`'s, as it was when the bytes rode along.
//
// Only the rows are written from here, and they are written whole: a library of a few
// dozen names and ids is a couple of kilobytes, which is what the split was for. What
// used to happen at this line was a base64 of every PNG in the library, on the thread
// the canvas paints on, per change.

/// Write the library's rows — [`ShapeEntry`] narrowed to what is durable about it.
///
/// The bytes are not this function's to write. An entry reaching here always has
/// them stored already, because every caller put them there first.
fn persist(entries: &[ShapeEntry]) {
    let rows: Vec<StoredShape> = entries
        .iter()
        .map(|e| StoredShape {
            name: e.name.clone(),
            id: e.id,
        })
        .collect();
    storage::save_list(&rows);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64×64 disc, written the two ways a brush shape may be authored: `alpha`
    /// puts the coverage in the alpha channel over transparent black, `value` puts
    /// it in the color over an opaque black ground.
    fn disc(alpha: bool) -> Vec<u8> {
        const N: u32 = 64;
        let mut pixels = Vec::with_capacity((N * N * 4) as usize);
        for y in 0..N {
            for x in 0..N {
                let (dx, dy) = (x as f32 - 31.5, y as f32 - 31.5);
                let r = (dx * dx + dy * dy).sqrt() / 30.0;
                let cov = (255.0 * (1.0 - r).clamp(0.0, 1.0)) as u8;
                pixels.extend_from_slice(&if alpha {
                    [255, 255, 255, cov]
                } else {
                    [cov, cov, cov, 255]
                });
            }
        }
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, N, N);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&pixels).unwrap();
        }
        out
    }

    /// The thumbnail is taken over the *coverage*, so the two spellings of one disc
    /// draw the same card — which is the whole point: an imported shape cannot look
    /// out of place beside a built-in for having put its coverage somewhere else.
    #[test]
    fn a_shape_draws_the_same_card_however_its_coverage_was_authored() {
        let (a, v) = (encode_thumb(&disc(true)), encode_thumb(&disc(false)));
        assert!(a.is_some());
        assert_eq!(a, v);
    }

    /// And the card the two agree on has a *transparent* ground, not a black one:
    /// where the disc does not cover, nothing is drawn, so the panel behind the card
    /// shows through however the source spelled that.
    #[test]
    fn where_a_shape_does_not_cover_the_thumbnail_is_clear() {
        let url = encode_thumb(&disc(false)).expect("a thumbnail");
        let b64 = url
            .strip_prefix("data:image/png;base64,")
            .expect("a png data url");
        let bytes = crate::platform::base64_decode(b64).expect("base64");

        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
            .read_info()
            .unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.color_type, png::ColorType::GrayscaleAlpha);

        // The corner is outside the disc; the centre is the whole of it.
        let at = |x: u32, y: u32| buf[((y * info.width + x) * 2 + 1) as usize];
        assert_eq!(at(0, 0), 0, "the corner is clear");
        // Not 255: the disc falls off from its exact centre, and the centre texel is
        // half a texel off it. What is being pinned is the axis, not the peak.
        assert!(
            at(info.width / 2, info.height / 2) > 240,
            "the centre covers"
        );
    }

    /// A mask arrives capped at the identity contract's `MAX_SHAPE_DIM` and leaves
    /// here at a card's size — the reduction is what keeps a data URL out of the DOM
    /// at a hundred times the size the picture is shown at.
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
}
