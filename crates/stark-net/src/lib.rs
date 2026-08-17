//! `stark-net` — the iroh transport for shared multi-user drawings
//! (§12.4).
//!
//! `stark-engine` owns the merge semantics (the `ReplicatedTimeline` CRDT over
//! the action log); this crate owns the wire and nothing else:
//!
//! - **Identity**: an iroh [`iroh::EndpointId`] (a public key) maps
//!   to the engine's [`ActorId`](stark_model::document::ActorId) via
//!   [`actor_from_endpoint_id`].
//! - **Live edits**: each committed [`Action`](stark_model::document::Action) is
//!   broadcast over `iroh-gossip` on the session's topic — a sampled path,
//!   never pixels.
//! - **Join / catch-up**: a joining peer fetches the session snapshot — the
//!   save-format [`DocumentFile`](stark_model::DocumentFile), which already
//!   bundles referenced brush assets — over a dedicated ALPN, then rides the
//!   gossip tail. Brush blobs a later stroke references are fetched over the
//!   same ALPN on demand (content-addressed, §6.6).
//! - **Repair**: gossip is a flood, not a delivery guarantee, so members
//!   periodically compare logs with a neighbour and fetch back whatever the flood
//!   dropped (§12.5). Without it a lost `CommitStroke` is a stroke that exists on
//!   some canvases and not others, forever, with both sides believing they are in
//!   sync — see [`reconcile`].
//!
//! The UI glue is a small pump: drain `Engine::take_outbox`
//! into [`CollabSession::broadcast`], and feed the [`RemoteEvent`]s the session's
//! [`Events`] stream yields into
//! `Engine::merge_remote` /
//! `Engine::import_brush`.

mod backend;
mod content;
mod mirror;
mod proto;
mod reconcile;
mod session;
mod ticket;
mod transport;
mod waitlist;

pub use session::{
    AssetNeed, Broadcaster, CollabSession, Events, Joined, LinkKind, NetOptions, PeerLink,
    RemoteEvent, actor_from_endpoint_id,
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
    #[error("stream ended early: {0}")]
    Truncated(#[from] iroh::endpoint::ReadExactError),
    #[error("stream closed: {0}")]
    Closed(#[from] iroh::endpoint::ClosedStream),
    #[error("encode/decode failed: {0}")]
    Codec(#[from] postcard::Error),
    #[error("gossip: {0}")]
    Gossip(#[from] iroh_gossip::api::ApiError),
    #[error("blob transfer failed: {0}")]
    BlobFetch(#[from] iroh_blobs::get::GetError),
    #[error("blob store read failed: {0}")]
    BlobRead(#[from] iroh_blobs::api::ExportBaoError),
    /// Something went wrong with the **document** — a snapshot that will not decode,
    /// a version this build is too old for (§8, §19).
    ///
    /// `stark-model`'s error rather than the engine's, because this crate never holds
    /// an engine: it moves logs and assets, and a lost GPU is not a thing that can
    /// happen to it (§2).
    #[error("document error: {0}")]
    Document(#[from] stark_model::DocError),
    /// The member asked has no session to serve yet — it is still fetching its
    /// own snapshot. Every member is an entry point (§12.4), so the answer is to
    /// ask a different one, not to give up.
    #[error("that session member is still joining; ask another")]
    NotReady,
    #[error("bad ticket: {0}")]
    Ticket(#[from] TicketError),
    #[error("{0}")]
    Other(String),
}

/// Why a pasted link could not be read.
///
/// Typed rather than a string, because every one of these reaches a person who
/// pasted something and needs to know which of four different things went wrong —
/// and because the frontend shows the text verbatim.
#[derive(Debug, thiserror::Error)]
pub enum TicketError {
    #[error("that is not a Stark link — a link starts with the prefix `stark`")]
    NoPrefix,
    #[error("the link is damaged: {0}")]
    Encoding(#[from] data_encoding::DecodeError),
    #[error("the link is damaged or cut short: {0}")]
    Malformed(#[from] postcard::Error),
    /// Named rather than guessed at: past the version byte the fields are a
    /// different shape, so anything else this could say about them would be about
    /// the wrong shape.
    #[error(
        "this link is version {found}; this build speaks {expected} —          both ends need the same version of Stark"
    )]
    Version { found: u8, expected: u8 },
}

pub type Result<T, E = NetError> = std::result::Result<T, E>;
