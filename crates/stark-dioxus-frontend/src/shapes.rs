//! The brush stamps a person brought in (§6.6): picking one, and what the brush does
//! when one leaves the library.
//!
//! The library itself — reading it in, importing into it, its cards, and making an
//! entry usable in this document — is `crate::library`, which keeps the canvas
//! substrates the same way. An entry is a **canonical grayscale PNG keyed by content
//! id**: the bytes the engine's `AssetStore` holds, bundles into save files and serves
//! to peers, so one representation flows everywhere. The library follows this browser
//! across documents, where the engine's store is per-document.

use dioxus::prelude::*;
use stark_model::AssetId;
use stark_model::document::BrushShape;
use stark_ui::assets::Shapes;

use crate::library;
use crate::state::{AppState, update_brush};

/// Make `id` the brush's stamp, importing the entry's bytes into this document first
/// when it has not seen them — what a gallery pick does, and what an import does once
/// the file is in the library.
pub fn select(state: AppState, id: AssetId) {
    if let Some(actual) = library::ensure::<Shapes>(state, id) {
        update_brush(state, |b, _| b.shape = BrushShape::Stamp(actual));
    }
}

/// Drop an entry from the library, putting a brush that held it back on the round tip.
pub fn remove(state: AppState, id: AssetId) {
    library::remove::<Shapes>(state, id);
    if state.brush.peek().shape == BrushShape::Stamp(id) {
        update_brush(state, |b, _| b.shape = BrushShape::default());
    }
}
