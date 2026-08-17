//! Engine error type (§5), surfaced everywhere.
//!
//! The renderer's half. The document's half is `stark-model`'s [`DocError`], which
//! this folds in (§2).

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

    #[error("cannot export: {0}")]
    Export(String),

    /// The GPU failed underneath an operation — a lost device, an exhausted one, or
    /// an error no scope caught (§5).
    ///
    /// **Reported rather than panicked on, because the document survives it.** The
    /// engine's state is an action log in ordinary memory, so a caller told this can
    /// still write the file; the readback path used to discover the same fact with an
    /// `expect`, which on the web is an abort and takes the painting with it.
    ///
    /// Distinct from [`Self::Export`], which is about a request that does not make
    /// sense (a frame too small, a size past the device's limit). That is answerable
    /// by asking for something else; this is not answerable at all, and
    /// [`ObservableState::gpu_failure`](crate::ObservableState::gpu_failure) is where
    /// a frontend learns it is permanent.
    #[error("{0}")]
    Gpu(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Something went wrong with the *document* rather than with the renderer — a
    /// file that will not decode, a space this build lacks, content it was never
    /// given (§2). Folded in so a caller holding the engine still catches everything
    /// in one place, while `stark-net` and any other log-only consumer can match on
    /// [`DocError`] alone and never see `NoAdapter`.
    #[error(transparent)]
    Document(#[from] stark_model::DocError),
}

/// `?` on a content-id failure, in one hop.
///
/// [`DocError`] already folds `AssetError` in, but `From` does not chain: without
/// this, every `?` on an id derivation inside an engine function would have to spell
/// the two steps out. One impl beats a hundred `.map_err`s, and it keeps the error a
/// caller sees identical either way.
impl From<stark_assetid::AssetError> for EngineError {
    fn from(e: stark_assetid::AssetError) -> Self {
        EngineError::Document(e.into())
    }
}

/// Convenience alias used throughout the engine.
pub type Result<T> = std::result::Result<T, EngineError>;
