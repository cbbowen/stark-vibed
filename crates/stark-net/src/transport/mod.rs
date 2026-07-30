//! Bindings from the [`mesh`](crate::mesh) onto real networks.
//!
//! The mesh owns the protocol and knows nothing about iroh or WebRTC; these
//! modules are the only place the two meet. [`iroh`] supplies the
//! [`MeshTransport`](crate::mesh::MeshTransport) — dial a peer, accept peers,
//! move framed bytes — and nothing else.

// `::iroh` — inside this module the bare name `iroh` is the child module below.
use ::iroh::EndpointId;

use crate::mesh::PeerId;

/// The mesh's ALPN, distinct from the catch-up/asset protocol. Shared by both
/// transports so either side of a connection can speak it.
pub(crate) const MESH_ALPN: &[u8] = b"stark/mesh/0";

/// The mesh addresses peers by raw key bytes; iroh wraps the same bytes.
pub(crate) fn to_peer_id(id: EndpointId) -> PeerId {
    PeerId(*id.as_bytes())
}

/// `None` if the bytes are not a valid public key — a peer we could never dial.
pub(crate) fn to_endpoint_id(peer: PeerId) -> Option<EndpointId> {
    EndpointId::from_bytes(peer.as_bytes()).ok()
}

pub(crate) mod iroh;

// WebRTC on the single endpoint (`webrtc2`): not a transport of its own — the
// mesh still rides `iroh` above — just the bootstrap that gives the endpoint
// direct WebRTC paths.
#[cfg(feature = "webrtc2")]
pub(crate) mod direct;
