//! Engine error type (§5), surfaced everywhere.
//!
//! The renderer's half. The document's half is `stark-model`'s [`DocError`](stark_model::DocError), which
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
    Export(#[from] ExportError),

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
    /// by asking for something else; this is not answerable at all.
    ///
    /// **Carries the failure, not a rendering of it.** It was `Gpu(String)`, built by
    /// `to_string()`-ing a [`DeviceFailure`](crate::DeviceFailure) the constructor had
    /// in hand — so a caller could not tell `FailureKind::Lost` (permanent; recovery
    /// means building a new device) from `FailureKind::OutOfMemory` (not necessarily
    /// terminal) without going to
    /// [`ObservableState::gpu_failure`](crate::ObservableState::gpu_failure) to be told
    /// what the error itself had just erased. That second channel is still how a
    /// frontend watches the device; it is no longer how it learns what happened.
    ///
    /// Carried by value rather than as a `#[from]` source: [`DeviceFailure`] is a
    /// plain reportable fact — `Clone`, comparable, free of any borrow of the device
    /// that died — precisely so it can also cross into `ObservableState`, and making
    /// it an `Error` to gain a conversion nothing asks for would be the tail wagging
    /// the dog. There is one construction site, and it names the variant.
    ///
    /// [`DeviceFailure`]: crate::DeviceFailure
    #[error("{0}")]
    Gpu(crate::gpu::DeviceFailure),

    /// A GPU operation failed while the device reports nothing wrong with itself.
    ///
    /// The other half of what [`Self::Gpu`] used to conflate: a readback that came
    /// back empty on a live device is a different fact from a dead one, and only one
    /// of the two is worth retrying.
    #[error("readback failed ({0})")]
    Readback(String),

    /// Something went wrong with the *document* rather than with the renderer — a
    /// file that will not decode, a space this build lacks, content it was never
    /// given (§2). Folded in so a caller holding the engine still catches everything
    /// in one place, while `stark-net` and any other log-only consumer can match on
    /// [`DocError`](stark_model::DocError) alone and never see `NoAdapter`.
    #[error(transparent)]
    Document(#[from] stark_model::DocError),
}

/// Why a render or an export could not be produced — a **request** that does not
/// make sense, or the encoder refusing what it was handed.
///
/// A type rather than a `String` because the two halves want different answers.
/// [`TooSmall`](Self::TooSmall), [`OverLimit`](Self::OverLimit),
/// [`UnusableView`](Self::UnusableView) and [`JpegTooLarge`](Self::JpegTooLarge)
/// are all "ask for something else", and a frontend that wants to *say* what else
/// — clamp the scale, offer the device's limit — needs the numbers rather than a
/// sentence containing them. [`Encode`](Self::Encode) and
/// [`EncodeJpeg`](Self::EncodeJpeg) are not answerable by asking differently at
/// all, and are the arms carrying somebody else's error.
#[derive(Debug, Error)]
pub enum ExportError {
    /// The frame has no area to render: an empty or non-finite bound (§15.6).
    #[error("frame is too small to export ({width:.0} × {height:.0} canvas px)")]
    TooSmall { width: f32, height: f32 },

    /// The scale asked for is not a positive, finite factor.
    #[error("export scale must be positive")]
    BadScale,

    /// The target is larger than this device will allocate a texture for. Both
    /// numbers are here because the useful reply is "then ask for `limit`".
    #[error("{} is {} × {} px; this device's limit is {limit}", what.noun(), size.width, size.height)]
    OverLimit {
        what: Produces,
        size: crate::view::Extent2,
        limit: u32,
    },

    /// The view handed in is not one anything can be rendered through
    /// ([`ViewTransform::usable`](crate::view::ViewTransform::usable)).
    #[error("view must be finite")]
    UnusableView,

    /// JPEG stores each dimension in 16 bits — a limit of the format, not of this
    /// device, which is why it is not an [`OverLimit`](Self::OverLimit).
    #[error("{width} × {height} px does not fit in a JPEG (65535 px a side at most)")]
    JpegTooLarge { width: u32, height: u32 },

    /// The PNG encoder refused the image — an arm that is not about the request.
    #[error("PNG encoding failed: {0}")]
    Encode(#[from] png::EncodingError),

    /// The JPEG encoder refused the image — the other such arm.
    #[error("JPEG encoding failed: {0}")]
    EncodeJpeg(#[from] jpeg_encoder::EncodingError),
}

/// Which of the two things a size limit was measured against, so
/// [`ExportError::OverLimit`] reads as the caller's own question.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Produces {
    /// A framed export at a chosen scale (§15.6).
    Export,
    /// A render of the current view to an image.
    Render,
}

impl Produces {
    fn noun(self) -> &'static str {
        match self {
            Produces::Export => "export",
            Produces::Render => "render",
        }
    }
}

/// `?` on a content-id failure, in one hop.
///
/// [`DocError`](stark_model::DocError) already folds `AssetError` in, but `From` does not chain: without
/// this, every `?` on an id derivation inside an engine function would have to spell
/// the two steps out. One impl beats a hundred `.map_err`s, and it keeps the error a
/// caller sees identical either way.
impl From<stark_assetid::AssetError> for EngineError {
    fn from(e: stark_assetid::AssetError) -> Self {
        EngineError::Document(e.into())
    }
}

/// Convenience alias used throughout the engine.
///
/// **Fixed rather than defaulted.** `Result<T, E = EngineError>` would widen this to
/// a superset of `std::result::Result` and save the two places below that spell the
/// std one out — but a type-parameter default is not used for *inference*, so
/// `stark_engine::Result::Ok(v)` in expression position stops resolving `E`, and that
/// is exactly how a frontend closure pins its error type (`stark-dioxus-frontend`'s
/// `collab`). Two in-crate spellings is the cheaper side of that trade.
pub type Result<T> = std::result::Result<T, EngineError>;
