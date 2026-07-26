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
//! # Two wires, one node
//!
//! The facade owns its endpoints and exposes no `Endpoint`/`Router`, so the
//! catch-up/asset protocol cannot ride an iroh `ProtocolHandler` here. Instead
//! [`serve_catchup`] accepts that ALPN through the facade and answers with
//! [`proto::answer`] — the same protocol the iroh backend serves, only the
//! plumbing differs.

use std::cell::RefCell;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use iroh::EndpointId;
use iroh_webrtc_transport::browser::{
    BrowserDialOptions, BrowserWebRtcAcceptor, BrowserWebRtcConnection, BrowserWebRtcNode,
    BrowserWebRtcStream,
};
use n0_future::task;

use crate::mesh::{
    MeshConn, MeshRecv, MeshSender, MeshTransport, MeshTransportError, PeerId, TransportResult,
};
use crate::mirror::Mirror;

use super::{MESH_ALPN as ALPN, to_endpoint_id};

/// Matches the iroh transport's ceiling.
const MAX_FRAME: usize = 1024 * 1024;

/// Pause between accept retries, and how many consecutive failures mean this
/// node itself is finished rather than one peer having dropped.
const ACCEPT_RETRY_DELAY: Duration = Duration::from_millis(250);
const ACCEPT_GIVE_UP_AFTER: u32 = 40;

enum Retry {
    Again,
    GiveUp,
}

/// Decide whether a failed accept is worth retrying. Individual failures are
/// expected — peers close tabs — so only a long unbroken run of them means the
/// node is gone.
async fn retry_accept(failures: &mut u32, what: &str, error: impl std::fmt::Debug) -> Retry {
    *failures += 1;
    if *failures >= ACCEPT_GIVE_UP_AFTER {
        tracing::warn!("giving up accepting {what} connections after {failures}: {error:?}");
        return Retry::GiveUp;
    }
    tracing::debug!("{what} accept failed ({failures}), retrying: {error:?}");
    n0_future::time::sleep(ACCEPT_RETRY_DELAY).await;
    Retry::Again
}

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
        Ok(Self {
            node,
            local,
            acceptor,
        })
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
        let mut failures = 0u32;
        loop {
            let conn = match self.acceptor.accept().await {
                Ok(Some(conn)) => {
                    failures = 0;
                    conn
                }
                Ok(None) => return None,
                // A peer vanishing (a closed browser tab) surfaces here as an
                // error. Retry: giving up would stop this node accepting *any*
                // future peer, so one departure would quietly end the session
                // for everyone who has not joined yet.
                Err(e) => match retry_accept(&mut failures, "mesh", e).await {
                    Retry::Again => continue,
                    Retry::GiveUp => return None,
                },
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
        let len =
            u32::try_from(frame.len()).map_err(|_| MeshTransportError::new("frame too large"))?;
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

/// Serve the catch-up/asset protocol over the facade, for as long as the node
/// lives. Every peer is a provider, so a joiner can bootstrap from any member.
pub(crate) fn serve_catchup(node: BrowserWebRtcNode, mirror: Arc<Mutex<Mirror>>) {
    task::spawn(async move {
        let acceptor = match node.accept(crate::proto::ALPN).await {
            Ok(acceptor) => acceptor,
            Err(e) => {
                tracing::error!("cannot serve catch-up over webrtc: {e:?}");
                return;
            }
        };
        let mut failures = 0u32;
        loop {
            match acceptor.accept().await {
                Ok(Some(conn)) => {
                    failures = 0;
                    // One task per peer: a slow fetch must not stall the others.
                    task::spawn(serve_catchup_conn(conn, mirror.clone()));
                }
                Ok(None) => break,
                // Never stop serving because one peer went away: this is the
                // path newcomers use to fetch the snapshot, so abandoning it
                // would make this member unjoinable for the rest of the session.
                Err(e) => match retry_accept(&mut failures, "catch-up", e).await {
                    Retry::Again => continue,
                    Retry::GiveUp => break,
                },
            }
        }
        tracing::debug!("catch-up server stopped");
    });
}

/// Answer requests on one connection until the peer goes away.
async fn serve_catchup_conn(conn: BrowserWebRtcConnection, mirror: Arc<Mutex<Mirror>>) {
    loop {
        let Ok(stream) = conn.accept_bi().await else {
            return;
        };
        // The peer half-closes after its request, so this returns the whole of it.
        let Ok(request) = stream.read_to_end().await else {
            return;
        };
        let response = match crate::proto::decode_request(&request)
            .and_then(|req| crate::proto::answer(&mirror, req))
        {
            Ok(response) => response,
            Err(e) => {
                tracing::warn!("bad catch-up request over webrtc: {e}");
                return;
            }
        };
        if stream.send_all(&response).await.is_err() {
            return;
        }
        let _ = stream.close_send().await;
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
