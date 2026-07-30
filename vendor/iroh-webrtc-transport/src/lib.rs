//! WebRTC data channels bridged to iroh custom transport, with JSEP signaling pluggable via [`Signaling`].
//!
//! Use [`QuicSignaling`] + [`negotiate_dc_as_offerer`] / [`negotiate_dc_as_answerer`] for iroh QUIC
//! streams (JSEP over QUIC uses [`JSEP_SIGNALING_ALPN`]), then hand the negotiated peer to
//! `WebRtcTransport::attach_data_channel`. The peer type and WebRTC stack are per-target with the
//! same negotiate/attach API shape: native uses str0m on a UDP socket (`Str0mPeer`); wasm uses the
//! browser's `RTCPeerConnection` (`WebRtcPeer`, plus a `WebPeerConfig` for STUN servers). The
//! two-message offer/answer protocol carries ICE inside the SDP, so browser and native negotiate
//! with each other.

mod bridge;
mod endpoint;
mod jsep_alpn;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
mod jsep_core;
mod jsep_envelope;
mod jsep_quic;
mod jsep_signaling;
mod sender;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
mod str0m_peer;
mod transport;
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
mod web_peer;

pub use bridge::{custom_addr_from_opaque_data, AttachOptions};
pub use jsep_alpn::JSEP_SIGNALING_ALPN;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub use jsep_core::{negotiate_dc_as_answerer, negotiate_dc_as_offerer};
pub use jsep_envelope::SignalEnvelope;
pub use jsep_quic::QuicSignaling;
pub use jsep_signaling::Signaling;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub use str0m_peer::Str0mPeer;
pub use transport::{WebRtcTransport, WEBRTC_TRANSPORT_ID};
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub use web_peer::{negotiate_dc_as_answerer, negotiate_dc_as_offerer, WebPeerConfig, WebRtcPeer};