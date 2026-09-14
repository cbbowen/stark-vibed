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

/// Read one piece of content out of this app's own bundle, by content id (§12.4, §8).
///
/// A local read — same-origin on the web, the file the binary shipped beside
/// natively. `None` for content this build does not ship or a read that failed, each
/// logged where it happened. What that costs depends on who asked: a session answering
/// `ResolveLocally` falls back to fetching it off a peer, while a log about to be
/// replayed has to refuse ([`settle`]).
pub async fn fetch(need: AssetNeed) -> Option<Vec<u8>> {
    let Some(path) = stark_ui::assets::shipped_at(need.content()).and_then(|row| row.path) else {
        // Either the host omitted something we never promised, or this build's
        // catalog moved under a document that referenced the old one.
        tracing::warn!(?need, "owed content is not in this build's bundle");
        return None;
    };
    crate::shipped::fetch_bytes(path).await
}

/// Fetch everything a file or a joined session owes before its log is replayed — all
/// of it, or `None` with the shortfall logged.
///
/// All or nothing because a replay cannot wait for what is missing: a substrate that is
/// not registered when its `SetSubstrate` replays deposits every later stroke through the
/// flat stand-in, and those pixels are stored (§6.4).
pub async fn settle(owed: &[AssetNeed]) -> Option<Vec<(AssetNeed, Vec<u8>)>> {
    let mut fetched = Vec::with_capacity(owed.len());
    for &need in owed {
        fetched.push((need, fetch(need).await));
    }
    match all_or_short(fetched) {
        Ok(supplied) => Some(supplied),
        Err(short) => {
            tracing::error!(
                ?short,
                "refused: this uses content this version of Stark does not have"
            );
            None
        }
    }
}

/// [`settle`]'s rule over what came back for each need: every need's bytes, or the
/// needs that came back empty.
fn all_or_short(
    fetched: Vec<(AssetNeed, Option<Vec<u8>>)>,
) -> Result<Vec<(AssetNeed, Vec<u8>)>, Vec<AssetNeed>> {
    let mut supplied = Vec::with_capacity(fetched.len());
    let mut short = Vec::new();
    for (need, bytes) in fetched {
        match bytes {
            Some(bytes) => supplied.push((need, bytes)),
            None => short.push(need),
        }
    }
    if short.is_empty() {
        Ok(supplied)
    } else {
        Err(short)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::AssetId;

    fn brush(n: u8) -> AssetNeed {
        AssetNeed::Brush(AssetId([n; 32]))
    }

    #[test]
    fn everything_fetched_is_supplied_in_order() {
        let fetched = vec![
            (brush(1), Some(vec![1])),
            (AssetNeed::Substrate(AssetId([2; 32])), Some(vec![2, 2])),
        ];
        assert_eq!(
            all_or_short(fetched),
            Ok(vec![
                (brush(1), vec![1]),
                (AssetNeed::Substrate(AssetId([2; 32])), vec![2, 2]),
            ])
        );
    }

    #[test]
    fn one_short_need_refuses_all_and_is_named() {
        let fetched = vec![
            (brush(1), Some(vec![1])),
            (brush(2), None),
            (brush(3), Some(vec![3])),
        ];
        assert_eq!(all_or_short(fetched), Err(vec![brush(2)]));
    }

    #[test]
    fn nothing_owed_settles_to_nothing() {
        assert_eq!(all_or_short(Vec::new()), Ok(Vec::new()));
    }
}
