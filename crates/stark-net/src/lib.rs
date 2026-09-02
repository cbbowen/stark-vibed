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
//! into [`Broadcaster::broadcast`], and feed the [`RemoteEvent`]s the session's
//! [`Events`] stream yields into
//! `Engine::merge_remote` /
//! `Engine::import_brush`.

mod backend;
mod cancel;
mod codec;
mod content;
mod events;
mod mirror;
mod neighbors;
mod proto;
mod reconcile;
mod session;
mod ticket;
mod transport;
mod waitlist;
mod wire;

pub use events::{Events, NetOptions, RemoteEvent, actor_from_endpoint_id};
pub use session::{Broadcaster, CollabSession, Joined, LinkKind, PeerLink};
pub use ticket::SessionTicket;

/// Content a remote action needs before it can be applied faithfully, and which
/// store it belongs in — [`stark_model::AssetNeed`], re-exported so a frontend
/// pumping this transport does not need to name two crates for one idea.
///
/// It lives in the model because it is a fact about the document (§2:
/// `Serialize` ⇒ model) — the *stores* it routes to are the engine's, and
/// loading a file asks the same question a joining peer does.
pub use stark_model::AssetNeed;

// Re-exports so frontends don't need a direct iroh dependency for the basics.
// `TopicId` is among them because it is [`SessionTicket::topic`]'s type, which a
// consumer could not otherwise name.
pub use iroh::{EndpointId, SecretKey};
pub use iroh_gossip::proto::TopicId;

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
    Codec(#[from] carbonite::Error),
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
    /// A member the link names did not answer the dial within its bound —
    /// walked past, so the link's other members get their turn
    /// (`session::join`'s `DIAL_TIMEOUT`).
    #[error("no answer from session member {}", .member.fmt_short())]
    NoAnswer { member: EndpointId },
    /// A response opened with a tag byte this build does not know — a newer
    /// peer's vocabulary, which the [`ALPN`](crate::wire::ALPN) should have
    /// kept from meeting this one.
    #[error("response tagged {tag}, which this build does not know")]
    UnknownTag { tag: u8 },
    #[error("bad ticket: {0}")]
    Ticket(#[from] TicketError),
}

/// Why a pasted link could not be read.
///
/// Typed rather than a string, because every one of these reaches a person who pasted
/// something and needs to know which of several different things went wrong — and
/// because the frontend shows the text verbatim.
#[derive(Debug, thiserror::Error)]
pub enum TicketError {
    #[error("that is not a Stark link — a link starts with the prefix `stark`")]
    NoPrefix,
    #[error("the link is damaged: {0}")]
    Encoding(#[from] data_encoding::DecodeError),
    /// The characters decoded, and what they spell is not a deflate stream (`ticket`'s
    /// `wrap` for why a link is compressed at all).
    ///
    /// The same sentence as [`Malformed`](Self::Malformed) to the person reading it,
    /// because the remedy is the same one — ask for the link again — and a separate case
    /// only because it is a different layer's error type. Which layer noticed is worth
    /// keeping: base64url has no checksum, so a link cut short by a chat client's line
    /// wrap decodes into perfectly legal bytes, and deflate is the first thing that says
    /// otherwise.
    #[error("the link is damaged or cut short: {0}")]
    Compressed(#[from] std::io::Error),
    #[error("the link is damaged or cut short: {0}")]
    Malformed(#[from] carbonite::Error),
    /// The link inflates past the ceiling `ticket` puts on one — a few hundred bytes'
    /// worth of names being what an honest one holds.
    ///
    /// Refused rather than expanded, because deflate's ratio means a string short enough
    /// to paste can name as many megabytes as it likes, and every link arrives from
    /// somewhere else (§8: a file has an unbounded door for the case a link has no
    /// equivalent of).
    #[error("this link is not a session link: it expands past {limit} bytes")]
    TooLarge { limit: u64 },
    /// Named rather than guessed at: past the version byte the fields are a
    /// different shape, so anything else this could say about them would be about
    /// the wrong shape.
    #[error(
        "this link is version {found}; this build speaks {expected} — both ends need the same version of Stark"
    )]
    Version { found: u8, expected: u8 },
    /// The link decoded, and a member it names is not a valid endpoint id.
    ///
    /// Its own case because it is the one damaged-link failure that gets *past* the
    /// encoding: the bytes were well-formed, the version matched, and what is wrong is
    /// a key inside — which no amount of re-pasting will fix.
    #[error("this link names something that is not a Stark peer")]
    NotAnEndpoint,
    /// The link names no member at all — nobody to join through.
    ///
    /// Refused at the parse rather than discovered as a dial with no one to dial:
    /// the person who pasted the link is still looking at the parse's answer.
    #[error("this link is empty — it names no session member to join through")]
    Empty,
}

pub type Result<T, E = NetError> = std::result::Result<T, E>;
