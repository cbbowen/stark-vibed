//! The user's custom brush-shape library (§6.6, build-order step 11).
//!
//! An entry is a **canonical grayscale PNG keyed by content id** — exactly the
//! bytes the engine's `AssetStore` holds, bundles into save files, and serves
//! to session peers, so one representation flows everywhere. The library
//! itself is frontend state: the engine's store is per-document (populated on
//! import and load), while the library follows this browser across documents
//! via `localStorage` (the `identity` bargain — where storage is unavailable
//! it degrades to a per-session library and breaks nothing).
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

/// One custom shape in the library — and, unchanged, one stored entry.
///
/// The live type *is* the stored type: every field here is durable, so a second struct
/// to map it onto would be a copy with nothing to say. The two `with` adapters are
/// what JSON cannot hold by itself (`crate::storage`) — the bytes as base64, the
/// content id as the hex Stark spells an id in everywhere else.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShapeEntry {
    /// Display name, defaulted from the imported file's stem.
    pub name: String,
    /// Canonical grayscale PNG (what the engine stores under `id`).
    #[serde(with = "storage::b64")]
    pub png: Vec<u8>,
    /// Content id of `png`. Persisted alongside the bytes; if an engine
    /// upgrade ever re-canonicalizes differently, [`select`] heals the entry
    /// from the id the import actually returns.
    #[serde(with = "storage::hex")]
    pub id: AssetId,
}

impl storage::Entry for ShapeEntry {
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
pub fn load(state: AppState) {
    let mut entries = state.shapes.entries;
    entries.set(read_storage());
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
        // The stored id predates a canonicalization change; heal it in place.
        let mut entries = state.shapes.entries;
        if let Some(e) = entries.write().iter_mut().find(|e| e.id == entry.id) {
            e.id = actual;
        }
        persist(&entries.read());
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

    let selected = state
        .obs
        .read()
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
// [`ShapeEntry`] is the stored entry, so there is nothing here but the two calls: the
// format, the encodings and the skip-a-damaged-entry rule are all `crate::storage`'s.

fn persist(entries: &[ShapeEntry]) {
    storage::save_list(entries);
}

/// What this browser has stored. An empty library and a browser that has stored
/// nothing are the same thing here — unlike the preset rack, nothing is seeded —
/// so the two answers are folded into one.
fn read_storage() -> Vec<ShapeEntry> {
    storage::load_list().unwrap_or_default()
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
