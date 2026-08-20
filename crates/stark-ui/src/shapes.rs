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

/// A `background-image` data URL for an entry's thumbnail. Callers should
/// memoize per entry — the PNG is re-encoded to base64 each call.
pub fn data_url(entry: &ShapeEntry) -> String {
    format!("data:image/png;base64,{}", base64_encode(&entry.png))
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
    storage::save(Store::Shapes, entries);
}

/// What this browser has stored. An empty library and a browser that has stored
/// nothing are the same thing here — unlike the preset rack, nothing is seeded —
/// so the two answers are folded into one.
fn read_storage() -> Vec<ShapeEntry> {
    storage::load_list(Store::Shapes).unwrap_or_default()
}
