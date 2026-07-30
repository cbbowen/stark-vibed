//! `stark-net` — the iroh transport for shared multi-user drawings
//! (DESIGN.md §12.4).
//!
//! `stark-core` owns the merge semantics (the `ReplicatedTimeline` CRDT over
//! the action log); this crate owns the wire and nothing else:
//!
//! - **Identity**: an iroh [`iroh::EndpointId`] (a public key) maps
//!   to the engine's [`ActorId`](stark_core::document::ActorId) via
//!   [`actor_from_endpoint_id`].
//! - **Live edits**: each committed [`Action`](stark_core::document::Action) is
//!   broadcast over the [`mesh`] — a sampled path, never pixels.
//! - **Join / catch-up**: a joining peer fetches the session snapshot — the
//!   save-format [`DocumentFile`](stark_core::DocumentFile), which already
//!   bundles referenced brush assets — over a dedicated ALPN, then rides the
//!   mesh tail. Brush blobs a later stroke references are fetched over the
//!   same ALPN on demand (content-addressed, DESIGN.md §6.6).
//!
//! The UI glue is a small pump: drain [`Engine::take_outbox`](stark_core::Engine::take_outbox)
//! into [`CollabSession::broadcast`], and feed [`RemoteEvent`]s into
//! [`Engine::merge_remote`](stark_core::Engine::merge_remote) /
//! [`Engine::import_brush`](stark_core::Engine::import_brush).

mod backend;
pub mod mesh;
mod mirror;
mod proto;
mod session;
mod ticket;
mod transport;

pub use mesh::LinkKind;
pub use session::{
    Broadcaster, CollabSession, NetOptions, PeerLink, RemoteEvent, actor_from_endpoint_id,
};
pub use ticket::SessionTicket;

// Re-exports so frontends don't need a direct iroh dependency for the basics.
pub use iroh::{EndpointId, SecretKey};

/// Errors from session setup and the wire.
#[derive(Debug, thiserror::Error)]
pub enum NetError {
    #[error("endpoint bind failed: {0}")]
    Bind(#[from] iroh::endpoint::BindError),
    #[error("connect failed: {0}")]
    Connect(#[from] iroh::endpoint::ConnectError),
    #[error("connection error: {0}")]
    Connection(#[from] iroh::endpoint::ConnectionError),
    #[error("stream write failed: {0}")]
    Write(#[from] iroh::endpoint::WriteError),
    #[error("stream read failed: {0}")]
    Read(#[from] iroh::endpoint::ReadToEndError),
    #[error("encode/decode failed: {0}")]
    Codec(#[from] postcard::Error),
    #[error("document error: {0}")]
    Document(#[from] stark_core::EngineError),
    #[error("bad ticket: {0}")]
    Ticket(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T, E = NetError> = std::result::Result<T, E>;
