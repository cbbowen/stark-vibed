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
use stark_net::AssetNeed;

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

/// Read content out of this app's own bundle, by content id (§12.4, §8).
///
/// The one place a need becomes bytes without the network: a session settling
/// what a host left out, a session answering `ResolveLocally` mid-stroke, and a
/// lean save file being opened all want exactly this.
///
/// A local read — same-origin on the web, the file the binary shipped beside
/// natively. Anything that will not resolve is simply left out of the result, and
/// what that costs depends on who asked: a session falls back to fetching it off a
/// peer, while a file has nobody to ask and must refuse to open (§6.4).
pub async fn fetch(owed: &[AssetNeed]) -> Vec<(AssetNeed, Vec<u8>)> {
    let mut out = Vec::new();
    for &need in owed {
        let Some(asset) = asset_for(need.content()) else {
            // Not ours to resolve. Either the host omitted something we never
            // promised, or this build's catalog moved under a document that
            // referenced the old one.
            tracing::warn!(?need, "owed content is not in this build's bundle");
            continue;
        };
        match dioxus::asset_resolver::read_asset_bytes(asset).await {
            Ok(bytes) => out.push((need, bytes)),
            Err(e) => tracing::warn!(?need, "could not read owed content locally: {e}"),
        }
    }
    out
}

/// Install one piece of locally-resolved content into the engine, under the id
/// that asked for it — the same two calls the network path makes for a resolved
/// asset, because locally-resolved content is not a different kind of content,
/// only a different way of getting hold of it.
///
/// `accept_surface` re-derives the id and refuses bytes that do not match, so a
/// catalog file that changed out from under a document is caught there rather
/// than deposited through the wrong weave; both wrappers log their own refusal.
pub fn install(r: &mut crate::render::Renderer, need: AssetNeed, bytes: &[u8]) {
    match need.surface() {
        Some(id) => r.accept_surface(id, bytes),
        None => r.import_brush(bytes),
    }
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

    /// **The catalog is append-only.** Every id here has been shipped, so a saved
    /// document may reference one and rely on this build to supply it (§8's
    /// version 6). Re-authoring `Gesso.png` or dropping a shape does not break a
    /// picker — it strands every painting made on it, which will then refuse to
    /// open rather than open wrong.
    ///
    /// Adding a row is free. Changing or removing one is a decision about other
    /// people's files, so it fails here first: add the new asset alongside, and
    /// retire the old one only when nothing can still be pointing at it.
    #[test]
    fn the_shipped_catalog_is_append_only() {
        // Shipped ids, oldest first. Append; do not edit.
        const SHIPPED: &[(&str, &str)] = &[
            (
                "shape/Flat.png",
                "4051f4c5e66e9e7a008a1367d31aaced9e524b06104dd9fc4509abffccd265a1",
            ),
            (
                "shape/Worn_Bristles.png",
                "62b76803f7c06460854d3268cd41d868f271ba1cf54ecc53b7387cb81d83979e",
            ),
            (
                "surface/Gesso.png",
                "0b88d740a6b3f35f57b5f1d6e4064ac7b4ace0d2c2abab417bbcce762602deb6",
            ),
            (
                "surface/Linen.png",
                "9d8105e76895f6e47b456177da890816a2983112548d7d748cd42c5d67cd5dc1",
            ),
        ];
        for (path, want) in SHIPPED {
            let got = BUILTIN_IDS
                .iter()
                .find(|(p, _)| p == path)
                .unwrap_or_else(|| panic!("{path} has shipped and is no longer bundled"));
            assert_eq!(
                got.1.to_hex(),
                *want,
                "{path} has shipped and its content changed; documents painted on the \
                 old one can no longer be opened. Add the new asset as a new row."
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
