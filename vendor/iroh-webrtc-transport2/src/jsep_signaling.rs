//! Transport abstraction for JSEP: move [`SignalEnvelope`] between peers without tying negotiation to QUIC or WebSocket.

use async_trait::async_trait;

use crate::jsep_envelope::SignalEnvelope;

/// Sends and receives JSON [`SignalEnvelope`] values. Framing is defined by each implementation
/// ([`crate::QuicSignaling`], etc.).
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
#[async_trait]
pub trait Signaling: Send {
    async fn send_envelope(&mut self, env: &SignalEnvelope) -> anyhow::Result<()>;
    async fn recv_envelope(&mut self) -> anyhow::Result<SignalEnvelope>;
}

/// Browser variant of [`Signaling`]: wasm is single-threaded and iroh's stream
/// types are not `Send` there, so neither the trait nor its futures require it.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
#[async_trait(?Send)]
pub trait Signaling {
    async fn send_envelope(&mut self, env: &SignalEnvelope) -> anyhow::Result<()>;
    async fn recv_envelope(&mut self) -> anyhow::Result<SignalEnvelope>;
}
