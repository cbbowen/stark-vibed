//! Extra transports under the session's endpoint.
//!
//! The live wire itself is `iroh-gossip` on the plain endpoint; this module
//! holds what augments it. WebRTC on the single endpoint (`webrtc` feature):
//! not a wire of its own — just the bootstrap that gives the endpoint direct
//! WebRTC *paths*, which every protocol on it (gossip, catch-up) then rides.

#[cfg(feature = "webrtc")]
pub(crate) mod direct;
