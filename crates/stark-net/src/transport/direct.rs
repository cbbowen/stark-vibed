//! WebRTC bootstrap for the single-endpoint backend (`webrtc` feature).
//!
//! The session's ordinary iroh endpoint carries a WebRTC
//! [`CustomTransport`](::iroh::endpoint::transports::CustomTransport)
//! (vendor/iroh-webrtc-transport). This module owns everything around it:
//!
//! - **Addressing.** A peer's WebRTC [`CustomAddr`] is derived from its
//!   [`EndpointId`] ([`custom_addr_for`]), so no ticket or wire format carries
//!   WebRTC addresses — knowing a peer's id is knowing its custom addr. It
//!   also makes the addr peer-level rather than channel-level: if crossed
//!   simultaneous bootstraps ever leave the two sides sending on different
//!   channels, both still resolve to the same addr and iroh's path validation
//!   is unaffected.
//! - **Signaling.** JSEP (SDP offer/answer, ICE inside the SDP) over one iroh
//!   bi-stream on [`SIGNALING_ALPN`] — which rides the relay on the web, and
//!   whatever exists natively. [`JsepProto`] is the answering side, registered
//!   on the session router.
//! - **Bootstrap.** [`ensure_direct`] is called for every mesh link (both
//!   ends, dial and accept). The side with the smaller endpoint id offers; the
//!   other side answers via [`JsepProto`]. After the channel attaches, the
//!   offerer re-announces the peer's custom addr with
//!   `Endpoint::add_addr` (a patched-iroh API), which opens the custom path on
//!   every live connection to that peer — mesh and catch-up conns *migrate*
//!   onto WebRTC, keeping their relay paths as fallback. Path opening is
//!   mutual, so the answerer needs no announcement of its own.
//!
//! Nothing here is in the dial or data path: if bootstrap fails (no WebRTC,
//! peer without the feature, handshake failure), connections simply stay on
//! the paths they already have.

use std::sync::Arc;
use std::time::Duration;

use ::iroh::endpoint::Connection;
use ::iroh::protocol::{AcceptError, ProtocolHandler};
use ::iroh::{Endpoint, EndpointAddr, EndpointId, TransportAddr};
use iroh_base::CustomAddr;
use iroh_webrtc_transport::{
    AttachOptions, QuicSignaling, WebRtcTransport, custom_addr_from_opaque_data,
};

/// JSEP signaling for the session's WebRTC channels, distinct from the mesh
/// and catch-up ALPNs.
pub(crate) const SIGNALING_ALPN: &[u8] = b"stark/webrtc-sig/0";

const DC_LABEL: &str = "stark/webrtc";

/// Give up on one bootstrap attempt after this long (covers the QUIC dial,
/// SDP exchange, ICE, and the channel opening).
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

/// Delays before each bootstrap attempt.
const ATTEMPT_DELAYS: [Duration; 3] = [
    Duration::ZERO,
    Duration::from_secs(1),
    Duration::from_secs(4),
];

/// A peer's WebRTC custom addr, derived from its endpoint id.
pub(crate) fn custom_addr_for(id: EndpointId) -> CustomAddr {
    custom_addr_from_opaque_data(id.as_bytes())
}

/// The transport for this session's endpoint; its advertised custom addr is
/// derived from our own endpoint id.
pub(crate) fn make_transport(local: EndpointId) -> Arc<WebRtcTransport> {
    Arc::new(WebRtcTransport::new(local.as_bytes().to_vec()))
}

/// `addr` plus the WebRTC custom addr derived from its id.
pub(crate) fn with_custom_addr(mut addr: EndpointAddr) -> EndpointAddr {
    let custom = TransportAddr::Custom(custom_addr_for(addr.id));
    addr.addrs.insert(custom);
    addr
}

// The negotiated peer type and the negotiate calls differ per target (str0m on
// native, the browser's RTCPeerConnection on wasm); the protocol is identical.

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
async fn negotiate_offer(
    sig: &mut QuicSignaling,
) -> anyhow::Result<iroh_webrtc_transport::Str0mPeer> {
    iroh_webrtc_transport::negotiate_dc_as_offerer(sig, DC_LABEL).await
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
async fn negotiate_answer(
    sig: &mut QuicSignaling,
) -> anyhow::Result<iroh_webrtc_transport::Str0mPeer> {
    iroh_webrtc_transport::negotiate_dc_as_answerer(sig).await
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
async fn negotiate_offer(
    sig: &mut QuicSignaling,
) -> anyhow::Result<iroh_webrtc_transport::WebRtcPeer> {
    iroh_webrtc_transport::negotiate_dc_as_offerer(
        sig,
        DC_LABEL,
        &iroh_webrtc_transport::WebPeerConfig::default(),
    )
    .await
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
async fn negotiate_answer(
    sig: &mut QuicSignaling,
) -> anyhow::Result<iroh_webrtc_transport::WebRtcPeer> {
    iroh_webrtc_transport::negotiate_dc_as_answerer(
        sig,
        &iroh_webrtc_transport::WebPeerConfig::default(),
    )
    .await
}

/// Answer one JSEP offer arriving on `conn` and attach the channel.
async fn answer_one(transport: &Arc<WebRtcTransport>, conn: &Connection) -> anyhow::Result<()> {
    let remote = conn.remote_id();
    let (send, recv) = conn.accept_bi().await?;
    let mut sig = QuicSignaling::new(send, recv);
    let peer = negotiate_answer(&mut sig).await?;
    transport.attach_data_channel(peer, custom_addr_for(remote), AttachOptions::default())?;
    tracing::debug!(remote = %remote.fmt_short(), "webrtc channel attached (answerer)");
    Ok(())
}

/// Answers JSEP offers and attaches the negotiated channel (native: inline in
/// the handler, whose future is `Send`).
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
#[derive(Debug, Clone)]
pub(crate) struct JsepProto {
    transport: Arc<WebRtcTransport>,
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
impl JsepProto {
    pub fn new(transport: Arc<WebRtcTransport>) -> Self {
        Self { transport }
    }
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
impl ProtocolHandler for JsepProto {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        answer_one(&self.transport, &conn)
            .await
            .map_err(|e| AcceptError::from_boxed(e.into()))?;
        // Hold the signaling connection open until the offerer closes it; it
        // is their signal that bootstrap (attach + addr announcement) is done.
        conn.closed().await;
        Ok(())
    }
}

/// Answers JSEP offers and attaches the negotiated channel. On wasm the
/// browser negotiation is `!Send` (`JsFuture`, `Rc`) but iroh requires `Send`
/// handler futures, so the handler only queues the connection; a local worker
/// task does the negotiating.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
#[derive(Debug, Clone)]
pub(crate) struct JsepProto {
    worker: tokio::sync::mpsc::UnboundedSender<Arc<Connection>>,
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
impl JsepProto {
    pub fn new(transport: Arc<WebRtcTransport>) -> Self {
        let (worker, mut inbox) = tokio::sync::mpsc::unbounded_channel::<Arc<Connection>>();
        // Exits when the router (and with it every JsepProto clone) is
        // dropped. Sequential is fine: one channel per peer, rarely.
        n0_future::task::spawn(async move {
            while let Some(conn) = inbox.recv().await {
                if let Err(e) = answer_one(&transport, &conn).await {
                    tracing::debug!("webrtc answer failed: {e:#}");
                }
            }
        });
        Self { worker }
    }
}

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
impl ProtocolHandler for JsepProto {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        let conn = Arc::new(conn);
        if self.worker.send(conn.clone()).is_err() {
            return Ok(());
        }
        // Hold the connection open for the worker (returning would tear it
        // down); the offerer closes it when bootstrap is done.
        conn.closed().await;
        Ok(())
    }
}

/// Fire-and-forget: make sure a WebRTC channel to `remote` exists or is being
/// bootstrapped. Called on every mesh link, both ends, as often as links come
/// and go — cheap when the channel already exists.
pub(crate) fn ensure_direct(
    endpoint: &Endpoint,
    transport: &Arc<WebRtcTransport>,
    remote: EndpointId,
) {
    // Exactly one side offers — the smaller endpoint id — the other answers
    // via [`JsepProto`]. Mesh links are mutual, so the offerer always learns
    // of the peer.
    if endpoint.id().as_bytes() >= remote.as_bytes() {
        return;
    }
    if transport.is_attached(&custom_addr_for(remote)) {
        return;
    }
    let endpoint = endpoint.clone();
    let transport = transport.clone();
    n0_future::task::spawn(async move {
        for delay in ATTEMPT_DELAYS {
            if !delay.is_zero() {
                n0_future::time::sleep(delay).await;
            }
            if transport.is_attached(&custom_addr_for(remote)) {
                return;
            }
            let attempt = bootstrap(&endpoint, &transport, remote);
            match n0_future::time::timeout(ATTEMPT_TIMEOUT, attempt).await {
                Ok(Ok(())) => {
                    tracing::debug!(remote = %remote.fmt_short(), "webrtc channel attached (offerer)");
                    return;
                }
                Ok(Err(e)) => {
                    tracing::debug!(remote = %remote.fmt_short(), "webrtc bootstrap attempt failed: {e:#}");
                }
                Err(_) => {
                    tracing::debug!(remote = %remote.fmt_short(), "webrtc bootstrap attempt timed out");
                }
            }
        }
        tracing::debug!(
            remote = %remote.fmt_short(),
            "webrtc bootstrap gave up; traffic stays on existing paths"
        );
    });
}

/// One bootstrap attempt: dial JSEP, negotiate as offerer, attach, announce.
async fn bootstrap(
    endpoint: &Endpoint,
    transport: &Arc<WebRtcTransport>,
    remote: EndpointId,
) -> anyhow::Result<()> {
    // Dialing with the custom addr also records it in iroh's remote map; the
    // dial itself rides whatever paths already work (relay on the web).
    let mixed = with_custom_addr(EndpointAddr::new(remote));
    let conn = endpoint.connect(mixed.clone(), SIGNALING_ALPN).await?;
    let (send, recv) = conn.open_bi().await?;
    let mut sig = QuicSignaling::new(send, recv);
    let peer = negotiate_offer(&mut sig).await?;
    transport.attach_data_channel(peer, custom_addr_for(remote), AttachOptions::default())?;
    conn.close(0u32.into(), b"signaling done");
    // Now that the channel exists, re-announce the custom addr: the patched
    // iroh opens it as a path on every live connection to this peer, and the
    // path selector migrates traffic onto it once it validates. Path opening
    // is mutual, so this one call serves both ends.
    endpoint.add_addr(mixed).await;
    Ok(())
}
