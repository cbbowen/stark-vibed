//! Engine error type (§5: `Action::Error`, surfaced everywhere).

use thiserror::Error;

/// Errors produced by the engine and its GPU subsystem.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,

    #[error("failed to create GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    #[error("failed to request GPU adapter: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),

    #[error("serialization failed: {0}")]
    Serialize(String),

    #[error("deserialization failed: {0}")]
    Deserialize(String),

    #[error("not a Stark document (bad magic)")]
    BadMagic,

    #[error("unsupported document version {0}")]
    UnsupportedVersion(u32),

    /// The document names a colour space this build does not carry — today only
    /// [`ColorSpaceId::Mixbox`](crate::colorspace::ColorSpaceId::Mixbox) in a build
    /// without the `mixbox` cargo feature.
    ///
    /// A sibling of [`Self::UnsupportedVersion`] rather than a decode error, and for
    /// the same reason: the bytes are perfectly well-formed and perfectly well
    /// understood — the id is a variant every build has, because the save format's
    /// enum indices cannot depend on a feature (§8, §19) — and what is missing is the
    /// implementation behind it. That distinction is what lets a frontend say "this
    /// document needs a Mixbox build" instead of "this file is corrupt".
    #[error(
        "this build does not support the {0:?} color space; \
         it was compiled without the `mixbox` feature"
    )]
    UnsupportedColorSpace(crate::colorspace::ColorSpaceId),

    #[error("asset decode failed: {0}")]
    Asset(String),
    /// A content id could not be derived — an image that would not decode. Its own
    /// crate because the derivation is the format's identity contract (§19), so it
    /// arrives as its own error and is folded in here.
    #[error("{0}")]
    AssetId(#[from] stark_assetid::AssetError),

    #[error("cannot export: {0}")]
    Export(String),

    /// A document was asked to replay while content its log names is neither bundled
    /// in the file nor loaded in this engine (§8).
    ///
    /// **Refusing is the point.** Replaying anyway is not a degraded open, it is a
    /// wrong one that persists: a `SetSurface` whose height map is missing deposits
    /// every stroke made on it through the flat stand-in, and those pixels are stored,
    /// so no later arrival un-bakes them (§6.4). This used to be a `tracing::error!`
    /// with `Ok(())` behind it, which is how a captured bug report came to replay
    /// perfectly smooth and cost an afternoon.
    ///
    /// Settle it first — [`Engine::unresolved_content`](crate::Engine::unresolved_content)
    /// is the bill and names each need, `import_brush`/`accept_surface` pay it. A
    /// collaboration *join* is the one caller that legitimately starts short and heals,
    /// and it does not come through here.
    #[error("this document names content that is neither bundled nor loaded: {0:?}")]
    MissingContent(Vec<crate::content::AssetNeed>),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the engine.
pub type Result<T> = std::result::Result<T, EngineError>;
