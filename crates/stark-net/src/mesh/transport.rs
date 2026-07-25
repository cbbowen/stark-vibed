//! What the mesh needs from a network, and nothing more.
//!
//! The mesh is deliberately ignorant of iroh, WebRTC, and relays: it asks only
//! for *framed, ordered, bidirectional channels to peers, addressed by 32-byte
//! id*. That keeps the protocol (membership, flooding, dedup, healing) testable
//! against an in-memory transport, and lets the same code run over plain iroh
//! connections or the WebRTC facade's connections.
//!
//! ## `MaybeSend`
//!
//! Native transports are `Send` (tokio moves tasks between threads); the browser
//! WebRTC facade is `Rc`-based and is *not*. Rather than fork the mesh, bounds
//! are written against [`MaybeSend`]/[`MaybeSync`], which are `Send`/`Sync` off
//! wasm and vacuous on it — the standard idiom for single-threaded wasm runtimes.

use std::fmt;
use std::future::Future;

use serde::{Deserialize, Serialize};

/// `Send` everywhere except wasm, where the browser runtime is single-threaded.
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
mod bounds {
    /// `Send` on native targets.
    pub trait MaybeSend: Send {}
    impl<T: Send + ?Sized> MaybeSend for T {}
    /// `Sync` on native targets.
    pub trait MaybeSync: Sync {}
    impl<T: Sync + ?Sized> MaybeSync for T {}
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
mod bounds {
    /// Vacuous on wasm: the browser runtime never moves work across threads.
    pub trait MaybeSend {}
    impl<T: ?Sized> MaybeSend for T {}
    /// Vacuous on wasm.
    pub trait MaybeSync {}
    impl<T: ?Sized> MaybeSync for T {}
}

pub use bounds::{MaybeSend, MaybeSync};

/// A mesh member's identity: the 32 bytes of its public key.
///
/// Deliberately *not* `iroh::EndpointId` — the mesh does not depend on iroh.
/// Transports convert at their boundary.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PeerId(pub [u8; 32]);

impl PeerId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for PeerId {
    /// Short form — enough to tell peers apart in logs.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0[..4] {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PeerId({self})")
    }
}

/// A transport-level failure. The mesh treats every one of these as "this path
/// is down for now" and heals by redialing — it never gives up on a peer.
#[derive(Debug, Clone)]
pub struct MeshTransportError(pub String);

impl MeshTransportError {
    pub fn new(msg: impl fmt::Display) -> Self {
        Self(msg.to_string())
    }
}

impl fmt::Display for MeshTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MeshTransportError {}

pub type TransportResult<T> = std::result::Result<T, MeshTransportError>;

/// Dials peers and surfaces inbound connections.
///
/// `accept` is pull-based; transports whose inbound side is push-based (iroh's
/// `ProtocolHandler`) bridge through a queue.
pub trait MeshTransport: MaybeSend + MaybeSync + 'static {
    type Conn: MeshConn;

    /// This node's identity.
    fn local_id(&self) -> PeerId;

    /// Open a connection to `peer`. Called repeatedly while healing, so it must
    /// be safe to retry.
    fn dial(&self, peer: PeerId) -> impl Future<Output = TransportResult<Self::Conn>> + MaybeSend;

    /// The next inbound connection, or `None` once the transport is closed.
    ///
    /// **`None` must mean permanently finished.** The mesh stops accepting for
    /// good when it sees one, so a transient failure — a peer vanishing
    /// mid-handshake, one bad connection — has to be swallowed and retried
    /// internally. Returning `None` for those would leave the node running and
    /// talking to its existing peers while silently refusing every new one.
    fn accept(&self) -> impl Future<Output = Option<Self::Conn>> + MaybeSend;
}

/// One peer-to-peer channel, split into halves so the reader task and the
/// broadcaster can work concurrently without sharing a lock.
pub trait MeshConn: MaybeSend + 'static {
    type Sender: MeshSender;
    type Recv: MeshRecv;

    /// Who is on the other end.
    fn peer(&self) -> PeerId;

    fn split(self) -> (Self::Sender, Self::Recv);
}

/// The write half. Cloneable so the same channel can be fed from anywhere.
pub trait MeshSender: Clone + MaybeSend + MaybeSync + 'static {
    /// Deliver one whole frame. Implementations must preserve frame boundaries
    /// and ordering; the mesh relies on both.
    fn send(&self, frame: Vec<u8>) -> impl Future<Output = TransportResult<()>> + MaybeSend;

    /// Drop the channel. Idempotent.
    fn close(&self);
}

/// The read half, owned by exactly one reader task per connection.
pub trait MeshRecv: MaybeSend + 'static {
    /// The next whole frame, or `None` at end of stream.
    fn recv(&mut self) -> impl Future<Output = TransportResult<Option<Vec<u8>>>> + MaybeSend;
}
