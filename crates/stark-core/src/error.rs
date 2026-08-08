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
