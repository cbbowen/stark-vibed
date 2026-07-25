//! The mesh over browser WebRTC, via the vendored `iroh-webrtc-transport`
//! facade (wasm only, behind the `webrtc` feature).
//!
//! # Why the facade rather than a custom transport on our own endpoint
//!
//! A WebRTC custom transport cannot simply be *added* to the session's iroh
//! endpoint: WebRTC signalling has to ride an existing connection (the relay),
//! so iroh learns the relay path first and then never attempts the custom one —
//! measured, see `vendor/iroh-webrtc-transport/tests/single_endpoint_webrtc.rs`.
//! The facade solves this by keeping a relay endpoint for signalling and a
//! separate relay-cleared endpoint for traffic, and dialing the latter
//! explicitly.
//!
//! That shape only works for protocols built on *explicit* dial/accept — which
//! is precisely why the live wire is now the [`mesh`](crate::mesh) instead of
//! `iroh-gossip`.
//!
//! # Fallback
//!
//! Connections are dialed [`webrtc_preferred`](BrowserDialOptions::webrtc_preferred):
//! direct WebRTC when the handshake succeeds, iroh relay when it does not. So a
//! peer behind a hostile NAT still joins, just with relay latency — and because
//! the mesh floods, one peer stuck on relay never isolates anyone else.
//!
//! # Status
//!
//! Compiled but not yet selected by [`CollabSession`](crate::CollabSession),
//! which still builds the iroh transport on every target. Switching over also
//! means serving the catch-up/asset ALPN through the facade — it owns its
//! endpoints and exposes no `Endpoint`/`Router` — and none of that can be
//! exercised outside a browser. See the `webrtc` feature note in `Cargo.toml`.

// Everything here is reachable only once the session selects this transport.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::str::FromStr;

use iroh::EndpointId;
use iroh_webrtc_transport::browser::{
    BrowserDialOptions, BrowserWebRtcAcceptor, BrowserWebRtcConnection, BrowserWebRtcNode,
    BrowserWebRtcStream,
};
use n0_future::task;

use crate::mesh::{
    MeshConn, MeshRecv, MeshSender, MeshTransport, MeshTransportError, PeerId, TransportResult,
};

use super::iroh::to_endpoint_id;

/// The mesh ALPN, shared with the iroh transport so either side can speak it.
pub(crate) const ALPN: &[u8] = super::iroh::ALPN;

/// Matches the iroh transport's ceiling.
const MAX_FRAME: usize = 1024 * 1024;

/// Facade errors are `JsValue`s; take them by `Debug` so this crate needs no
/// `wasm-bindgen` dependency of its own.
fn js_err(e: impl std::fmt::Debug) -> MeshTransportError {
    MeshTransportError::new(format!("{e:?}"))
}

fn peer_to_string(peer: PeerId) -> TransportResult<String> {
    to_endpoint_id(peer)
        .map(|id| id.to_string())
        .ok_or_else(|| MeshTransportError::new("bad peer id"))
}

fn string_to_peer(s: &str) -> TransportResult<PeerId> {
    EndpointId::from_str(s)
        .map(|id| PeerId(*id.as_bytes()))
        .map_err(|e| MeshTransportError::new(format!("bad remote endpoint id: {e}")))
}

/// Dials and accepts mesh connections through the browser WebRTC facade.
pub(crate) struct WebRtcMeshTransport {
    node: BrowserWebRtcNode,
    local: PeerId,
    acceptor: BrowserWebRtcAcceptor,
}

impl WebRtcMeshTransport {
    /// Wrap a facade node that was built with `.accept_facade(ALPN)`.
    pub async fn new(node: BrowserWebRtcNode) -> TransportResult<Self> {
        let local = string_to_peer(node.endpoint_id())?;
        let acceptor = node.accept(ALPN).await.map_err(js_err)?;
        Ok(Self { node, local, acceptor })
    }
}

impl MeshTransport for WebRtcMeshTransport {
    type Conn = WebRtcConn;

    fn local_id(&self) -> PeerId {
        self.local
    }

    async fn dial(&self, peer: PeerId) -> TransportResult<WebRtcConn> {
        let remote = peer_to_string(peer)?;
        let conn = self
            .node
            // Direct when WebRTC negotiates, relay when it does not.
            .dial(&remote, ALPN, BrowserDialOptions::webrtc_preferred())
            .await
            .map_err(js_err)?;
        // The dialer opens the stream; the mesh writes its greeting at once, so
        // the peer's `accept_bi` resolves immediately.
        let stream = conn.open_bi().await.map_err(js_err)?;
        Ok(WebRtcConn { peer, conn, stream })
    }

    async fn accept(&self) -> Option<WebRtcConn> {
        loop {
            let conn = match self.acceptor.accept().await {
                Ok(Some(conn)) => conn,
                Ok(None) => return None,
                Err(e) => {
                    tracing::debug!("webrtc accept failed: {e:?}");
                    return None;
                }
            };
            let peer = match string_to_peer(conn.remote_endpoint_id()) {
                Ok(peer) => peer,
                Err(e) => {
                    tracing::warn!("ignoring webrtc peer with unusable id: {e}");
                    continue;
                }
            };
            match conn.accept_bi().await {
                Ok(stream) => return Some(WebRtcConn { peer, conn, stream }),
                // One failed handshake should not stop us accepting others.
                Err(e) => tracing::debug!("webrtc stream accept failed: {e:?}"),
            }
        }
    }
}

pub(crate) struct WebRtcConn {
    peer: PeerId,
    conn: BrowserWebRtcConnection,
    stream: BrowserWebRtcStream,
}

impl MeshConn for WebRtcConn {
    type Sender = WebRtcSender;
    type Recv = WebRtcRecv;

    fn peer(&self) -> PeerId {
        self.peer
    }

    fn split(self) -> (WebRtcSender, WebRtcRecv) {
        (
            WebRtcSender {
                conn: self.conn,
                stream: self.stream.clone(),
            },
            WebRtcRecv {
                stream: self.stream,
                buffer: Rc::new(RefCell::new(Vec::new())),
            },
        )
    }
}

#[derive(Clone)]
pub(crate) struct WebRtcSender {
    conn: BrowserWebRtcConnection,
    stream: BrowserWebRtcStream,
}

impl MeshSender for WebRtcSender {
    async fn send(&self, frame: Vec<u8>) -> TransportResult<()> {
        let len = u32::try_from(frame.len())
            .map_err(|_| MeshTransportError::new("frame too large"))?;
        // One write: the facade's stream is a byte stream, so the length prefix
        // and body must not be able to interleave with another frame.
        let mut out = Vec::with_capacity(4 + frame.len());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&frame);
        self.stream.send_all(&out).await.map_err(js_err)
    }

    fn close(&self) {
        let conn = self.conn.clone();
        task::spawn(async move {
            let _ = conn.close("mesh closed").await;
        });
    }
}

pub(crate) struct WebRtcRecv {
    stream: BrowserWebRtcStream,
    /// Chunks arrive at arbitrary boundaries; frames are reassembled here.
    buffer: Rc<RefCell<Vec<u8>>>,
}

impl MeshRecv for WebRtcRecv {
    async fn recv(&mut self) -> TransportResult<Option<Vec<u8>>> {
        loop {
            if let Some(frame) = take_frame(&self.buffer)? {
                return Ok(Some(frame));
            }
            match self.stream.read_chunk().await.map_err(js_err)? {
                Some(chunk) => self.buffer.borrow_mut().extend_from_slice(&chunk),
                // End of stream. A partial frame here just means the peer went
                // away mid-write; the mesh will redial.
                None => return Ok(None),
            }
        }
    }
}

/// Split one whole length-prefixed frame off the front of `buffer`, if it has
/// arrived in full.
fn take_frame(buffer: &Rc<RefCell<Vec<u8>>>) -> TransportResult<Option<Vec<u8>>> {
    let mut buffer = buffer.borrow_mut();
    if buffer.len() < 4 {
        return Ok(None);
    }
    let len = u32::from_le_bytes(buffer[..4].try_into().expect("4 bytes")) as usize;
    if len > MAX_FRAME {
        return Err(MeshTransportError::new(format!(
            "peer announced a {len}-byte frame"
        )));
    }
    if buffer.len() < 4 + len {
        return Ok(None);
    }
    let frame = buffer[4..4 + len].to_vec();
    buffer.drain(..4 + len);
    Ok(Some(frame))
}
