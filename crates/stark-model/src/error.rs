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
    /// The two halves of the container's codec. They carry the `carbonite::Error`
    /// itself rather than its `to_string()`, so `source()` reaches it and a caller
    /// printing the chain gets what actually went wrong — the courtesy every other
    /// variant here already extends (`AssetId`, `Io`).
    ///
    /// Two variants rather than one `#[from]`, because both directions fail with
    /// the same type and which one it was is the useful half of the message: a
    /// serialize failure is this build's bug, a deserialize failure is the file's.
    #[error("serialization failed")]
    Serialize(#[source] carbonite::Error),

    #[error("deserialization failed")]
    Deserialize(#[source] carbonite::Error),

    #[error("not a Stark document (bad magic)")]
    BadMagic,

    /// A document written by a **pre-carbonite** build, whose container carried a
    /// schema version and whose body was postcard (§8.1).
    ///
    /// The one error here that can never be fixed by a newer build, and the reason
    /// the format moved: postcard writes no field names, so those bytes only mean
    /// anything to the exact schema that wrote them, and reading version 6 would
    /// take this build's copy of all thirteen historical layouts. Files are alpha
    /// (§19), so they are named rather than migrated.
    ///
    /// **A tombstone, not a version check.** It is raised by recognizing the old
    /// *container header*, never consulted to decide how to read a current file, and
    /// nothing bumps it — a carbonite document carries no version at all, because
    /// the writer's schema travels with it and is reconciled by name (§8).
    #[error(
        "this document was saved by an older Stark (format version {0}), \
         which this build can no longer read"
    )]
    Legacy(u32),

    /// A container that **arrived from somewhere else** expands past
    /// `io::MAX_DECOMPRESSED` — raised by `DocumentFile::from_untrusted_bytes` and
    /// by nothing else (§8, §12.4).
    ///
    /// A refusal rather than a decode error, and its own variant rather than an
    /// `Io`, because the bytes may be perfectly well-formed: deflate compresses a
    /// long run to almost nothing, so a small file can name an enormous one, and a
    /// reader that expands first and asks afterwards has already spent the memory.
    ///
    /// **Opening the user's own file cannot produce this**, which is the half worth
    /// stating: a document is as large as the artist made it, so the bound is about
    /// who wrote the bytes rather than about how many there are.
    #[error("document expands to more than {limit} bytes")]
    TooLarge { limit: u64 },

    /// The document names a color space this build does not carry — today only
    /// [`ColorSpaceId::Mixbox`] in a build without the `mixbox` cargo feature.
    ///
    /// A refusal rather than a decode error: the bytes are perfectly well-formed and
    /// perfectly well understood — the id is a variant every build has, because the
    /// save format's vocabulary cannot depend on a feature (§8, §19) — and what is
    /// missing is the implementation behind it. That distinction is what lets a
    /// frontend say "this document needs a Mixbox build" instead of "this file is
    /// corrupt".
    ///
    /// Raised by whoever *opens* the document rather than by whoever decodes it: this
    /// crate has no `mixbox` feature to consult (§2), and "can this build honour the
    /// space" is a question about the engine, not about the file.
    #[error(
        "this build does not support the {0:?} color space; \
         it was compiled without the `mixbox` feature"
    )]
    UnsupportedColorSpace(ColorSpaceId),

    /// Content arrived under an id its own bytes do not hash to.
    ///
    /// **The one way a content-addressed store could still hold the wrong thing**
    /// (§19), and so a refusal rather than a repair: `import_*` derives the id from
    /// the bytes and has nothing to disagree with, but `accept_*` is handed both, off
    /// a save file's bundle or a peer's transfer (§8, §12.4). What installing them
    /// anyway would cost differs by kind and is never nothing — a substrate deposits
    /// the wrong tooth into tiles no later arrival un-bakes (§6.4), a picture is
    /// simply a different photograph — so the check is the same at every door.
    #[error("{expected:?} arrived as {actual:?}; refusing to install it")]
    Misnamed {
        expected: crate::AssetNeed,
        actual: crate::AssetNeed,
    },

    /// Registered bytes a store could not read. The detail is whatever the decoder
    /// said, which is the one thing here that genuinely is a sentence.
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
    /// wrong one that persists: a `SetSubstrate` whose height map is missing deposits
    /// every stroke made on it through the flat stand-in, and those pixels are stored,
    /// so no later arrival un-bakes them (§6.4). Logging and returning `Ok(())` here
    /// would report success and hand back a document that replays perfectly smooth,
    /// with nothing on screen to say which substrate it was actually painted through.
    ///
    /// Settle it first — `stark-engine`'s `Engine::unresolved_content` is the bill and
    /// names each need, `import_brush`/`accept_substrate` pay it. A collaboration *join*
    /// is the one caller that legitimately starts short and heals, and it does not come
    /// through here.
    #[error("this document names content that is neither bundled nor loaded: {0:?}")]
    MissingContent(Vec<AssetNeed>),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias for the document layer.
pub type Result<T> = std::result::Result<T, DocError>;
