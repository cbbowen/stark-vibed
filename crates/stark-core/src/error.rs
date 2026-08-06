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

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the engine.
pub type Result<T> = std::result::Result<T, EngineError>;
