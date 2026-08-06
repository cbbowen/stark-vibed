//! What the app already has, by content id (§6.6, §6.4, §12.4).
//!
//! Every asset under `assets/shape` and `assets/surface` is hashed at build time
//! (`build.rs`) into [`BUILTIN_IDS`], so this build knows the id of a bundled
//! brush shape or canvas ground **without fetching it**. Both catalogs are
//! explicit that an id is otherwise only knowable once the bytes have arrived;
//! this is the table that breaks the circularity.
//!
//! What it buys: a peer that switches to a ground this app ships with names it by
//! content id like any other, and the receiver — knowing it can resolve that id
//! from its own bundle — declines the transfer instead of pulling megabytes over
//! the network for bytes sitting next to its binary.
//!
//! It is *only* a table of ids. Resolving one to bytes is still a fetch, just a
//! local one; nothing here embeds an image.

use stark_assetid::AssetId;

include!(concat!(env!("OUT_DIR"), "/builtin_ids.rs"));

/// The bundled file behind a content id, if this build ships it — the reverse of
/// [`resolvable`], for actually making good on what it promised.
pub fn asset_for(id: AssetId) -> Option<dioxus::prelude::Asset> {
    let path = BUILTIN_IDS
        .iter()
        .find(|(_, i)| *i == id)
        .map(|(p, _)| *p)?;
    crate::builtins::SHAPES
        .iter()
        .find(|s| s.path == path)
        .map(|s| s.asset)
        .or_else(|| {
            crate::grounds::GROUNDS
                .iter()
                .find(|g| g.path == Some(path))
                .and_then(|g| g.asset)
        })
}

/// Every content id this build can resolve out of its own bundle.
///
/// Handed to a session at join time so the host can leave them out of the
/// snapshot: the joiner is not saying "I have these loaded", it is saying "I can
/// get these without you" (§12.4).
pub fn resolvable() -> Vec<AssetId> {
    BUILTIN_IDS.iter().map(|(_, id)| *id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalogs name their files as literals for `asset!` and again as a path
    /// for this table, because a proc macro needs a literal and a lookup needs a
    /// string. Two spellings of one filename is exactly the kind of thing that
    /// drifts, so it is checked rather than trusted: every catalog row must
    /// resolve, and every bundled asset must be claimed by a row.
    #[test]
    fn the_catalogs_and_the_manifest_name_the_same_files() {
        let claimed: Vec<&str> = crate::builtins::SHAPES
            .iter()
            .map(|s| s.path)
            .chain(crate::grounds::GROUNDS.iter().filter_map(|g| g.path))
            .collect();
        for path in &claimed {
            assert!(
                BUILTIN_IDS.iter().any(|(p, _)| p == path),
                "catalog names {path}, which is not in assets/ — the row and the file disagree"
            );
        }
        for (path, _) in BUILTIN_IDS {
            assert!(
                claimed.contains(path),
                "assets/{path} is bundled and hashed but no catalog row offers it"
            );
        }
    }

    /// Distinct files must be distinct content; two rows sharing an id would make
    /// the picker offer one thing twice.
    #[test]
    fn every_bundled_asset_is_distinct_content() {
        for (i, (path, id)) in BUILTIN_IDS.iter().enumerate() {
            for (other, other_id) in &BUILTIN_IDS[i + 1..] {
                assert_ne!(id, other_id, "{path} and {other} hash to one id");
            }
        }
    }
}
