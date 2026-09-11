//! The files this build bundles, by the catalog path that names them (§6.4, §6.6,
//! §12.4).
//!
//! Which assets ship, what each is called and how it is described are
//! `stark_ui::assets`' two catalogs. What neither can carry is the `Asset` a file is
//! fetched through: `asset!` is a proc macro that demands a path literal inside this
//! crate, so each file is spelled here once more, as a literal — and the tests check
//! that the two spellings name the same files.
//!
//! **Fetched at runtime** rather than compiled in, so megabytes of stamps and height
//! maps stay out of the wasm binary. That is why an id is only knowable once the bytes
//! arrive, and why a picker has to be told when they do ([`watch`]).

use dioxus::prelude::*;

use crate::state::AppState;

/// Every bundled file, by its catalog path: both catalogs, one table.
const BUNDLED: &[(&str, Asset)] = &[
    (
        "shape/Worn_Bristles.png",
        asset!("/assets/shape/Worn_Bristles.png"),
    ),
    ("shape/Flat.png", asset!("/assets/shape/Flat.png")),
    ("shape/Pencil.png", asset!("/assets/shape/Pencil.png")),
    ("substrate/Linen.png", asset!("/assets/substrate/Linen.png")),
    ("substrate/Rough.png", asset!("/assets/substrate/Rough.png")),
];

/// The bundled file at a catalog path.
fn bundled_at(path: &str) -> Option<Asset> {
    BUNDLED
        .iter()
        .find(|(bundled, _)| *bundled == path)
        .map(|(_, asset)| *asset)
}

/// The bytes of the bundled file at a catalog path — a same-origin read — or `None`,
/// having said why.
///
/// Every caller takes `None` as "not available" rather than as an error to surface: a
/// shape that will not fetch leaves its card blank while the rest load, a substrate
/// leaves the canvas smooth, and owed content is asked of a peer instead
/// (`crate::builtin_ids`).
pub async fn fetch_bytes(path: &str) -> Option<Vec<u8>> {
    let Some(asset) = bundled_at(path) else {
        tracing::warn!(path, "no file is bundled at this catalog path");
        return None;
    };
    match dioxus::asset_resolver::read_asset_bytes(asset).await {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            tracing::warn!(path, "could not fetch a shipped asset: {e}");
            None
        }
    }
}

/// Subscribe the caller to a shipped asset landing in the main engine — and to nothing
/// else the engine does.
///
/// A picker listing shipped assets has to redraw when one lands, and used to learn it
/// by reading the renderer signal, which is written on every command. Two moments
/// instead: the renderer arriving, holding every shape and the opening substrate it
/// imported on its way up ([`renderer_ready`](crate::state::Signals::renderer_ready)),
/// and each substrate map fetched after that ([`landed`]).
pub fn watch(state: AppState) {
    let _ = (state.renderer_ready)();
    let _ = (state.shipped_landed)();
}

/// Say that a shipped asset has just landed in the main engine, waking what [`watch`]es.
pub fn landed(state: AppState) {
    let mut landed = state.shipped_landed;
    let count = *landed.peek();
    landed.set(count.wrapping_add(1));
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use stark_ui::assets::{SHIPPED_SHAPES, SHIPPED_SUBSTRATES};

    use super::BUNDLED;

    fn catalogued() -> BTreeSet<&'static str> {
        SHIPPED_SHAPES
            .iter()
            .chain(SHIPPED_SUBSTRATES)
            .filter_map(|row| row.path)
            .collect()
    }

    fn bundled() -> BTreeSet<&'static str> {
        BUNDLED.iter().map(|(path, _)| *path).collect()
    }

    /// Every file a catalog row names is bundled here: a row without its file is a card
    /// that never draws and a preset that never resolves.
    #[test]
    fn every_file_a_catalog_names_is_bundled() {
        let missing: Vec<_> = catalogued().difference(&bundled()).copied().collect();
        assert!(
            missing.is_empty(),
            "the catalogs name {missing:?} and no `asset!` here does"
        );
    }

    /// And every bundled file is offered by some catalog row, once: a file nothing
    /// fetches is weight the deploy still carries.
    #[test]
    fn every_bundled_file_is_offered_once() {
        let stray: Vec<_> = bundled().difference(&catalogued()).copied().collect();
        assert!(
            stray.is_empty(),
            "{stray:?} is bundled but no catalog row offers it"
        );
        assert_eq!(bundled().len(), BUNDLED.len(), "a file is bundled twice");
    }
}
