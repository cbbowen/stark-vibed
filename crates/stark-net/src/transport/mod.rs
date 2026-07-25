//! Bindings from the [`mesh`](crate::mesh) onto real networks.
//!
//! The mesh owns the protocol and knows nothing about iroh or WebRTC; these
//! modules are the only place the two meet. Each supplies a
//! [`MeshTransport`](crate::mesh::MeshTransport) — dial a peer, accept peers,
//! move framed bytes — and nothing else.

pub(crate) mod iroh;

#[cfg(all(feature = "webrtc", target_family = "wasm", target_os = "unknown"))]
pub(crate) mod webrtc;
