//! The mesh wire format.
//!
//! Three frames carry everything: who I am and who I know ([`Frame::Hello`]),
//! an application payload to flood ([`Frame::Data`]), and periodic membership
//! gossip ([`Frame::Peers`]). Payloads are opaque bytes — the mesh neither
//! parses nor interprets what it carries.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::transport::PeerId;

/// Identifies one swarm, so a peer can be in several sessions at once and a
/// stray connection from another session is rejected at `Hello`.
///
/// 32 bytes, encoded exactly like `iroh_gossip::TopicId` was, so existing
/// session tickets stay wire-compatible.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TopicId([u8; 32]);

impl TopicId {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for TopicId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0[..4] {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for TopicId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopicId({self})")
    }
}

/// A frame on a mesh connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Frame {
    /// Always the first frame in each direction. Establishes the sender's
    /// identity, guards the swarm boundary, and seeds peer discovery.
    Hello {
        topic: TopicId,
        from: PeerId,
        /// Everyone the sender knows about, so the receiver can widen its view.
        peers: Vec<PeerId>,
    },
    /// An application payload, flooded through the swarm. `(origin, seq)`
    /// identifies it uniquely for deduplication and gap detection.
    Data {
        origin: PeerId,
        seq: u64,
        payload: Vec<u8>,
    },
    /// Membership gossip: peers the sender has learned since `Hello`.
    Peers { peers: Vec<PeerId> },
}

impl Frame {
    pub(crate) fn encode(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }
}
