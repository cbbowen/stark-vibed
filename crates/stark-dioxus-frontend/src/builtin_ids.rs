//! What the app already has, by content id (§6.6, §6.4, §12.4).
//!
//! The table of ids is `stark_ui::assets` — hashed at build time, in the crate both
//! frontends depend on, so there is one answer to "what does this build already
//! have" rather than one per frontend. What is left here is the two halves that need
//! this frontend's own vocabulary: an id is read out of this build's bundle
//! (`crate::shipped`), and what was read becomes an import into the engine.
//!
//! What the table buys: a peer that switches to a substrate this app ships with names
//! it by content id like any other, and the receiver — knowing it can resolve that id
//! from its own bundle — declines the transfer instead of pulling megabytes over the
//! network for bytes sitting next to its binary.

use stark_model::SubstrateId;
use stark_net::AssetNeed;

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
        let Some(path) = stark_ui::assets::shipped_at(need.content()).and_then(|row| row.path)
        else {
            // Not ours to resolve. Either the host omitted something we never
            // promised, or this build's catalog moved under a document that
            // referenced the old one.
            tracing::warn!(?need, "owed content is not in this build's bundle");
            continue;
        };
        // A read that fails has said why; the need is left out, for the reason above.
        if let Some(bytes) = crate::shipped::fetch_bytes(path).await {
            out.push((need, bytes));
        }
    }
    out
}

/// Install one piece of content into the engine, under the id that asked for it —
/// read out of this app's own bundle or arrived off a peer, which are two ways of
/// getting hold of one kind of content (§12.4).
///
/// `accept_substrate` and `accept_picture` re-derive the id and refuse bytes that do not
/// match, so a file that changed out from under a document is caught there rather than
/// deposited through the wrong substrate; each wrapper logs its own refusal.
///
/// **Exhaustive on the need, with no `_` arm.** `AssetNeed::substrate()` answers `None`
/// for a brush *and* for a picture (§23), so a two-arm form files a picture's RGBA bytes
/// in the brush store, where they decode as luminance × alpha and are neither (§8).
pub fn install(r: &mut crate::render::Renderer, need: AssetNeed, bytes: &[u8]) {
    match need {
        AssetNeed::Brush(_) => r.import_brush(bytes),
        AssetNeed::Substrate(id) => r.accept_substrate(SubstrateId::Image(id), bytes),
        AssetNeed::Picture(id) => r.accept_picture(id, bytes),
    }
}
