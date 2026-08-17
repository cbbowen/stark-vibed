//! Which canvas a document is painted on (§6.4) — the id, not the ground.

use serde::{Deserialize, Serialize};

use stark_assetid::AssetId;

/// Which physical surface a document is painted on. Saved in `CanvasMeta` (§8)
/// because which canvas a piece was painted on is part of the document, so it is
/// reproducible.
///
/// **Two variants, and the split is the point.** `Flat` is procedural and needs no
/// bytes; every other ground *is* its bytes, named by the hash of them. There is no
/// third case — no ground named by a label whose image the engine would have to be
/// told about separately — because that case is exactly the one that can go missing
/// (§6.4). A peer, a save file or a replay that meets an
/// [`Image`](Self::Image) id it has never seen can always ask for it by content, and
/// verify what comes back; a ground called "Gesso" could only be looked up in a
/// table the asker might not have, and the miss was silent — the tooth read a flat
/// stand-in and baked it into the tiles.
///
/// So this is the same bargain brush shapes already make (§6.6): the id
/// comes *from* the image (`stark-engine`'s `Engine::import_surface`),
/// which is what makes "built-in" a property of the frontend's asset list and of
/// nothing downstream. The engine still embeds no image bytes.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    carbonite::Schema,
)]
pub enum SurfaceId {
    /// Perfectly smooth: zero height everywhere, so the
    /// constant height has zero gradient (no relief). Paint behaves exactly as if
    /// there were no surface — the orthogonal default.
    #[default]
    Flat,
    /// A height map, named by the BLAKE3 hash of its canonical decoded form
    /// (`stark-engine`'s `surface::identify`). Covers the grounds that ship with the app and the ones a
    /// user brings, identically — the engine cannot tell them apart, which is why
    /// neither can go missing in a way the other wouldn't.
    Image(AssetId),
}
