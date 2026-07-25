//! Bindings from the [`mesh`](crate::mesh) onto real networks.
//!
//! The mesh owns the protocol and knows nothing about iroh or WebRTC; these
//! modules are the only place the two meet. Each supplies a
//! [`MeshTransport`](crate::mesh::MeshTransport) — dial a peer, accept peers,
//! move framed bytes — and nothing else.
//!
//! Exactly one is compiled, chosen by the same condition as
//! [`backend`](crate::backend): the WebRTC facade on wasm with the `webrtc`
//! feature, plain iroh otherwise.

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

#[cfg(not(all(feature = "webrtc", target_family = "wasm", target_os = "unknown")))]
pub(crate) mod iroh;

#[cfg(all(feature = "webrtc", target_family = "wasm", target_os = "unknown"))]
pub(crate) mod webrtc;
