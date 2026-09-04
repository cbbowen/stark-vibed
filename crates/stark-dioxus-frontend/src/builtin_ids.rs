//! What the app already has, by content id (§6.6, §6.4, §12.4).
//!
//! The table of ids is `stark_ui::assets` — hashed at build time, in the crate
//! both frontends depend on, so there is one answer to "what does this build already
//! have" rather than one per frontend. What is left here is the two halves that need
//! this frontend's own vocabulary: an id becomes a `dioxus::Asset` to fetch, and a
//! fetched asset becomes an `AssetNeed` the session glue speaks in.
//!
//! What the table buys: a peer that switches to a substrate this app ships with names
//! it by content id like any other, and the receiver — knowing it can resolve that id
//! from its own bundle — declines the transfer instead of pulling megabytes over the
//! network for bytes sitting next to its binary.

use stark_assetid::AssetId;
use stark_model::SubstrateId;
use stark_net::AssetNeed;

/// The bundled file behind a content id, if this build ships it — for actually making
/// good on what `stark_ui::assets::resolvable` promised.
pub fn asset_for(id: AssetId) -> Option<dioxus::prelude::Asset> {
    let path = stark_ui::assets::shipped_at(id)?.path?;
    crate::builtins::bundled_at(path).or_else(|| crate::substrates::bundled_at(path))
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
/// `accept_substrate` re-derives the id and refuses bytes that do not match, so a
/// catalog file that changed out from under a document is caught there rather
/// than deposited through the wrong substrate; both wrappers log their own refusal.
/// **Exhaustive on the need, with no `_` arm**, rather than branching on whether it
/// names a substrate. That is not a style preference: `AssetNeed::substrate()` answers
/// `None` for a brush *and* for a picture (§23), so the two-arm form quietly filed a
/// picture's RGBA bytes in the brush store, where they would decode as luminance ×
/// alpha and be neither. The catalogs ship no pictures, so it could not fire today —
/// which is exactly the kind of latent wrong-bag bug §8 keys the bags apart to
/// prevent, and the reason this refuses instead.
pub fn install(r: &mut crate::render::Renderer, need: AssetNeed, bytes: &[u8]) {
    match need {
        AssetNeed::Brush(_) => r.import_brush(bytes),
        AssetNeed::Substrate(id) => r.accept_substrate(SubstrateId::Image(id), bytes),
        // No build ships a picture: one is by definition something a person brought
        // in, so a catalog naming one is a catalog that is wrong about itself.
        AssetNeed::Picture(id) => {
            tracing::error!(?id, "the shipped catalog cannot resolve a picture")
        }
    }
}

#[cfg(test)]
mod tests {
    /// The catalog names its files as paths and this frontend names them again as
    /// `asset!` literals, because a proc macro needs a literal and a lookup needs a
    /// string. Two spellings of one filename is exactly the kind of thing that drifts.
    ///
    /// The other half — that a catalog path is a file that was actually hashed — is
    /// `stark_ui::assets`', where the id table is. This half cannot move with it:
    /// an `Asset` is a `dioxus` type and the crate below names none.
    #[test]
    fn every_catalog_row_has_the_file_this_frontend_bundles() {
        for row in stark_ui::assets::SHIPPED_SHAPES
            .iter()
            .chain(stark_ui::assets::SHIPPED_SUBSTRATES)
        {
            let Some(path) = row.path else { continue };
            assert!(
                super::asset_for(
                    stark_ui::assets::shipped_id(path).expect("a catalog row is hashed")
                )
                .is_some(),
                "the catalog names {path} and no `asset!` in this frontend does"
            );
        }
    }

    /// And nothing is bundled that no row offers — a file fetched by nothing is
    /// weight in the deploy that the manifest still carries.
    #[test]
    fn nothing_this_frontend_bundles_is_unreachable() {
        let claimed: Vec<&str> = stark_ui::assets::SHIPPED_SHAPES
            .iter()
            .chain(stark_ui::assets::SHIPPED_SUBSTRATES)
            .filter_map(|s| s.path)
            .collect();
        for path in [
            "shape/Worn_Bristles.png",
            "shape/Flat.png",
            "shape/Pencil.png",
        ] {
            assert!(crate::builtins::bundled_at(path).is_some());
            assert!(claimed.contains(&path), "{path} is bundled but unreachable");
        }
        for path in ["substrate/Linen.png", "substrate/Rough.png"] {
            assert!(crate::substrates::bundled_at(path).is_some());
            assert!(claimed.contains(&path), "{path} is bundled but unreachable");
        }
    }
}
