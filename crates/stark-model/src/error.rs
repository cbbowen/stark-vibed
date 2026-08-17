//! What can go wrong with a *document* (§8, §19) — as opposed to with a renderer.
//!
//! The split is the crate split (§2). A file that will not decode, a version this
//! build is too old for, a space it does not carry, content it was never given: all
//! of those are facts about the document, answerable without a GPU, and a consumer
//! that only ever reads and writes logs — `stark-net`, a headless tool — should not
//! have to match on `NoAdapter` to handle them.
//!
//! `stark-engine`'s `EngineError` keeps the other half (a lost device, an
//! impossible export) and folds this in with `#[from]`, so a caller holding the
//! engine still catches everything in one place.

use thiserror::Error;

use crate::ColorSpaceId;
use crate::content::AssetNeed;

/// Errors produced by reading, writing or resolving a document.
#[derive(Debug, Error)]
pub enum DocError {
    #[error("serialization failed: {0}")]
    Serialize(String),

    #[error("deserialization failed: {0}")]
    Deserialize(String),

    #[error("not a Stark document (bad magic)")]
    BadMagic,

    #[error("unsupported document version {0}")]
    UnsupportedVersion(u32),

    /// The container's compressed body expands past what this build will hold in
    /// memory at once (`io::MAX_DECOMPRESSED`).
    ///
    /// A refusal rather than a decode error, and its own variant rather than an
    /// `Io`, because the bytes may be perfectly well-formed: deflate compresses a
    /// long run to almost nothing, so a small file can name an enormous one, and a
    /// reader that expands first and asks afterwards has already spent the memory.
    /// Documents arrive from peers as well as from disk (§12.4), which is the case
    /// the bound is actually for.
    #[error("document expands to more than {limit} bytes")]
    TooLarge { limit: u64 },

    /// The document was recorded against a different tile stride than this build
    /// addresses the canvas with (`CanvasMeta::tile_size`).
    ///
    /// A sibling of [`Self::UnsupportedVersion`] for [`Self::TooLarge`]'s reason:
    /// the bytes are well-formed and well-understood, and what is wrong is that
    /// they do not mean here what they meant where they were written. Every tile
    /// boundary moves with the stride, so there is no partial reading to fall back
    /// on — and the field was stored on every save and read by nothing until this
    /// existed, so the mismatch used to load clean and render wrong.
    #[error("document uses a tile size of {found}; this build addresses tiles of {expected}")]
    TileSize { expected: u32, found: u32 },

    /// The document names a color space this build does not carry — today only
    /// [`ColorSpaceId::Mixbox`] in a build without the `mixbox` cargo feature.
    ///
    /// A sibling of [`Self::UnsupportedVersion`] rather than a decode error, and for
    /// the same reason: the bytes are perfectly well-formed and perfectly well
    /// understood — the id is a variant every build has, because the save format's
    /// enum indices cannot depend on a feature (§8, §19) — and what is missing is the
    /// implementation behind it. That distinction is what lets a frontend say "this
    /// document needs a Mixbox build" instead of "this file is corrupt".
    ///
    /// Raised by whoever *opens* the document rather than by whoever decodes it: this
    /// crate has no `mixbox` feature to consult (§2), and "can this build honour the
    /// space" is a question about the engine, not about the file.
    #[error(
        "this build does not support the {0:?} color space; \
         it was compiled without the `mixbox` feature"
    )]
    UnsupportedColorSpace(ColorSpaceId),

    #[error("asset decode failed: {0}")]
    Asset(String),

    /// A content id could not be derived — an image that would not decode. Its own
    /// crate because the derivation is the format's identity contract (§19), so it
    /// arrives as its own error and is folded in here.
    #[error("{0}")]
    AssetId(#[from] stark_assetid::AssetError),

    /// A document was asked to replay while content its log names is neither bundled
    /// in the file nor loaded in this engine (§8).
    ///
    /// **Refusing is the point.** Replaying anyway is not a degraded open, it is a
    /// wrong one that persists: a `SetSurface` whose height map is missing deposits
    /// every stroke made on it through the flat stand-in, and those pixels are stored,
    /// so no later arrival un-bakes them (§6.4). Logging and returning `Ok(())` here
    /// would report success and hand back a document that replays perfectly smooth,
    /// with nothing on screen to say which ground it was actually painted through.
    ///
    /// Settle it first — `stark-engine`'s `Engine::unresolved_content` is the bill and
    /// names each need, `import_brush`/`accept_surface` pay it. A collaboration *join*
    /// is the one caller that legitimately starts short and heals, and it does not come
    /// through here.
    #[error("this document names content that is neither bundled nor loaded: {0:?}")]
    MissingContent(Vec<AssetNeed>),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias for the document layer.
pub type Result<T> = std::result::Result<T, DocError>;
