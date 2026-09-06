//! What can go wrong with a *document* (§8, §19) — as opposed to with a renderer.
//!
//! Everything here is answerable without a GPU, so a consumer that only reads and
//! writes logs (`stark-net`, a headless tool) never has to match on a device error.
//! `stark-engine`'s `EngineError` holds the other half and folds this in with
//! `#[from]`, so a caller holding the engine still catches both in one place.

use thiserror::Error;

use crate::ColorSpaceId;
use crate::content::AssetNeed;

/// Errors produced by reading, writing or resolving a document.
#[derive(Debug, Error)]
pub enum DocError {
    /// The two halves of the container's codec, kept apart because which direction
    /// failed is the useful half: a serialize failure is this build's bug, a
    /// deserialize failure is the file's. Both carry the `carbonite::Error` itself,
    /// so `source()` reaches it.
    #[error("serialization failed")]
    Serialize(#[source] carbonite::Error),

    #[error("deserialization failed")]
    Deserialize(#[source] carbonite::Error),

    /// The body is not a deflate stream this build can inflate — the corrupt or
    /// truncated file. Distinct from [`Deserialize`](Self::Deserialize), which is a
    /// body that *did* inflate and then was not a document.
    #[error("this document is damaged: its body will not decompress")]
    Corrupt(#[source] std::io::Error),

    /// Compression failed on the way out. The sink is a `Vec`, so unreachable in
    /// practice; named rather than `expect`ed.
    #[error("compression failed")]
    Deflate(#[source] std::io::Error),

    #[error("not a Stark document (bad magic)")]
    BadMagic,

    /// A document written by a **pre-carbonite** build, whose container carried a
    /// schema version and whose body was postcard (§8.1).
    ///
    /// The one error here no newer build can fix: postcard writes no field names, so
    /// those bytes only mean anything to the exact schema that wrote them, and files
    /// are alpha (§19).
    ///
    /// **A tombstone, not a version check** — raised by recognizing the old container
    /// header, never consulted to read a current file, and never bumped, since a
    /// carbonite document carries no version at all (§8).
    #[error(
        "this document was saved by an older Stark (format version {0}), \
         which this build can no longer read"
    )]
    Legacy(u32),

    /// A container that **arrived from somewhere else** expands past
    /// `io::MAX_DECOMPRESSED` — raised by `DocumentFile::from_untrusted_bytes` and by
    /// nothing else (§8, §12.4), so opening the user's own file cannot produce it.
    ///
    /// A refusal rather than a decode error: the bytes may be well-formed, since
    /// deflate lets a small file name an enormous one, and a reader that expands
    /// first has already spent the memory.
    #[error("document expands to more than {limit} bytes")]
    TooLarge { limit: u64 },

    /// The document names a color space this build does not carry — today only
    /// [`ColorSpaceId::Mixbox`] in a build without the `mixbox` cargo feature.
    ///
    /// A refusal rather than a decode error: every build has the id, since the save
    /// format's vocabulary cannot depend on a feature (§8, §19); what is missing is
    /// the implementation. Raised by whoever *opens* the document, not by whoever
    /// decodes it — this crate has no `mixbox` feature to consult (§2).
    #[error(
        "this build does not support the {0:?} color space; \
         it was compiled without the `mixbox` feature"
    )]
    UnsupportedColorSpace(ColorSpaceId),

    /// Content arrived under an id its own bytes do not hash to — the one way a
    /// content-addressed store could still hold the wrong thing (§19).
    ///
    /// Only `accept_*` can raise it, being handed id and bytes separately off a save
    /// file's bundle or a peer's transfer (§8, §12.4); `import_*` derives the id from
    /// the bytes. Installing a mismatch is never free: a wrong substrate bakes the
    /// wrong tooth into tiles no later arrival un-bakes (§6.4).
    #[error("{expected:?} arrived as {actual:?}; refusing to install it")]
    Misnamed {
        expected: crate::AssetNeed,
        actual: crate::AssetNeed,
    },

    /// Registered bytes a store could not read. The detail is whatever the decoder
    /// said, which is the one thing here that genuinely is a sentence.
    #[error("asset decode failed: {0}")]
    Asset(String),

    /// A content id could not be derived — an image that would not decode. It arrives
    /// from `stark-assetid`, which owns the format's identity contract (§19).
    #[error("{0}")]
    AssetId(#[from] stark_assetid::AssetError),

    /// A document was asked to replay while content its log names is neither bundled
    /// in the file nor loaded in this engine (§8).
    ///
    /// Replaying anyway would be wrong rather than degraded, and it persists: a
    /// `SetSubstrate` whose height map is missing deposits every stroke through the
    /// flat stand-in, and those pixels are stored (§6.4).
    ///
    /// Settle it first — `stark-engine`'s `Engine::unresolved_content` names each
    /// need, `import_brush`/`accept_substrate` pay it. A collaboration *join*
    /// legitimately starts short and heals, and never comes through here.
    #[error("this document names content that is neither bundled nor loaded: {0:?}")]
    MissingContent(Vec<AssetNeed>),
}

/// Convenience alias for the document layer.
pub type Result<T> = std::result::Result<T, DocError>;
