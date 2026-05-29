//! Engine error type (DESIGN.md §5: `Action::Error`, surfaced everywhere).

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

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the engine.
pub type Result<T> = std::result::Result<T, EngineError>;
