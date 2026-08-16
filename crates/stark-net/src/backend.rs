//! How a session binds to the network and reaches its peers.
//!
//! Binding the stack — the endpoint, the router, gossip — plus opening
//! catch-up connections, minting the ticket address, and shutting down.
//! [`session`](crate::session) drives it and owns the protocol above it.
//!
//! The backend is an ordinary iroh [`Endpoint`](iroh::Endpoint) and
//! [`Router`](iroh::protocol::Router) on every target; in the browser iroh
//! rides its relay (WebSocket) transport. With the **`webrtc`** feature the
//! endpoint additionally carries a WebRTC custom transport: connections
//! establish over whatever works and migrate onto a WebRTC path once a data
//! channel is bootstrapped (see [`transport::direct`](crate::transport::direct))
//! — which is what gives the *web* direct connections.

#[cfg(not(target_arch = "wasm32"))]
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
#[cfg(feature = "webrtc")]
use std::sync::Arc;
use std::time::Duration;

use iroh::endpoint::{Connection, presets};
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey, TransportAddr};
use iroh_blobs::store::mem::MemStore;
use iroh_blobs::{BlobsProtocol, Hash};
use iroh_gossip::net::Gossip;

use crate::Result;
use crate::mirror::Served;
use crate::proto::{self, CollabProto, Request};
use crate::session::NetOptions;

/// How long to wait for relay/publish readiness before minting a ticket.
const ONLINE_TIMEOUT: Duration = Duration::from_secs(15);

/// A long stroke's fitted control points can be sizeable, so the gossip
/// message ceiling sits well past any plausible single action (paths are
/// RDP-simplified; pixels never ride gossip).
const MAX_MESSAGE_SIZE: usize = 256 * 1024;

pub(crate) struct Bound {
    pub dialer: Dialer,
    pub gossip: Gossip,
    pub shutdown: Shutdown,
}

pub(crate) async fn bind(served: Served, opts: &NetOptions) -> Result<Bound> {
    let secret = opts.secret.clone().unwrap_or_else(SecretKey::generate);
    // The WebRTC custom transport rides the same endpoint; peers derive
    // its addr from our endpoint id (see transport::direct).
    #[cfg(feature = "webrtc")]
    let webrtc = crate::transport::direct::make_transport(secret.public());

    let builder = if opts.local_only {
        Endpoint::builder(presets::Minimal).secret_key(secret)
    } else {
        Endpoint::builder(presets::N0).secret_key(secret)
    };
    #[cfg(feature = "webrtc")]
    let builder = builder.add_custom_transport(webrtc.clone());
    let endpoint = builder.bind().await?;

    let gossip = Gossip::builder()
        .max_message_size(MAX_MESSAGE_SIZE)
        .spawn(endpoint.clone());
    // Brush assets: an in-memory blob store served over the blobs ALPN.
    // The store actor lives as long as any client handle (the Dialer's).
    let blobs: iroh_blobs::api::Store = MemStore::new().into();

    let router = Router::builder(endpoint.clone())
        .accept(iroh_gossip::ALPN, gossip.clone())
        .accept(iroh_blobs::ALPN, BlobsProtocol::new(&blobs, None))
        .accept(proto::ALPN, CollabProto { served });
    #[cfg(feature = "webrtc")]
    let router = router.accept(
        crate::transport::direct::SIGNALING_ALPN,
        crate::transport::direct::JsepProto::new(endpoint.clone(), webrtc.clone()),
    );
    let router = router.spawn();

    Ok(Bound {
        dialer: Dialer {
            endpoint: endpoint.clone(),
            blobs,
            #[cfg(feature = "webrtc")]
            webrtc,
        },
        gossip,
        shutdown: Shutdown { endpoint, router },
    })
}

#[derive(Clone)]
pub(crate) struct Dialer {
    endpoint: Endpoint,
    blobs: iroh_blobs::api::Store,
    #[cfg(feature = "webrtc")]
    webrtc: Arc<iroh_webrtc_transport::WebRtcTransport>,
}

impl Dialer {
    pub fn local_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// Fire-and-forget: bootstrap a WebRTC channel to `remote` unless one
    /// exists (or we are the answering side). Called on every gossip
    /// neighbor; connections migrate onto the channel once it attaches.
    pub fn ensure_direct(&self, remote: EndpointId) {
        #[cfg(feature = "webrtc")]
        crate::transport::direct::ensure_direct(&self.endpoint, &self.webrtc, remote);
        #[cfg(not(feature = "webrtc"))]
        let _ = remote;
    }

    /// The address traffic to `remote` currently travels over — the remote
    /// map's selected path. `None` when no path selection exists (no live
    /// connection). Sampled per call; paths migrate.
    pub async fn selected_addr(&self, remote: EndpointId) -> Option<TransportAddr> {
        self.endpoint
            .remote_info(remote)
            .await?
            .selected_addr()
            .cloned()
    }

    /// If a WebRTC channel to this peer is already attached, add its
    /// (derived) custom addr so the connection starts direct. Never before
    /// it attaches: a custom path opened against an unattached channel
    /// fails validation and blocks the later real open (see
    /// transport::direct::bootstrap).
    fn dial_addr(&self, addr: EndpointAddr) -> EndpointAddr {
        #[cfg(feature = "webrtc")]
        if self
            .webrtc
            .is_attached(&crate::transport::direct::custom_addr_for(addr.id))
        {
            return crate::transport::direct::with_custom_addr(addr);
        }
        addr
    }

    /// Connecting also teaches the endpoint how to reach `addr`, which is
    /// what later lets gossip dial the same peer by bare id.
    pub async fn open(&self, addr: EndpointAddr) -> Result<Catchup> {
        let conn = self
            .endpoint
            .connect(self.dial_addr(addr), proto::ALPN)
            .await?;
        Ok(Catchup { conn })
    }

    /// Store `bytes` in the local blob store so peers can fetch them.
    /// Returns immediately with the content hash (computed synchronously —
    /// broadcasts referencing it need not wait); the store insert runs
    /// detached.
    pub fn add_blob(&self, bytes: bytes::Bytes) -> Hash {
        let hash = Hash::new(&bytes);
        let blobs = self.blobs.clone();
        n0_future::task::spawn(async move {
            if let Err(e) = blobs.add_bytes(bytes).await {
                tracing::warn!("storing a blob failed: {e}");
            }
        });
        hash
    }

    /// Fetch one blob from `provider`, hash-verified into the local store
    /// (so this peer can serve it onward), and return its bytes.
    pub async fn fetch_blob(&self, provider: EndpointId, hash: Hash) -> Result<bytes::Bytes> {
        let conn = self
            .endpoint
            .connect(
                self.dial_addr(EndpointAddr::new(provider)),
                iroh_blobs::ALPN,
            )
            .await?;
        self.blobs.remote().fetch(conn, hash).await?;
        Ok(self.blobs.get_bytes(hash).await?)
    }

    /// With public infrastructure, wait (bounded) for the relay handshake so
    /// the ticket carries a relay URL; local-only tickets carry the bound
    /// sockets, loopback-normalized.
    pub async fn ticket_addr(&self, opts: &NetOptions) -> Result<EndpointAddr> {
        if !opts.local_only {
            // `online()` pends forever with no WAN; bound wait, then best effort.
            let _ = n0_future::time::timeout(ONLINE_TIMEOUT, self.endpoint.online()).await;
            return Ok(self.endpoint.addr());
        }
        // Local-only is native-only: a browser has no UDP sockets to
        // advertise, so there it yields a bare-id ticket that only
        // same-machine tests could ever have used anyway.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut addr = EndpointAddr::new(self.endpoint.id());
            for sock in self.endpoint.bound_sockets() {
                let sock = if sock.ip().is_unspecified() {
                    let loopback: IpAddr = if sock.is_ipv4() {
                        Ipv4Addr::LOCALHOST.into()
                    } else {
                        Ipv6Addr::LOCALHOST.into()
                    };
                    SocketAddr::new(loopback, sock.port())
                } else {
                    sock
                };
                addr = addr.with_ip_addr(sock);
            }
            Ok(addr)
        }
        #[cfg(target_arch = "wasm32")]
        Ok(EndpointAddr::new(self.endpoint.id()))
    }
}

pub(crate) struct Catchup {
    conn: Connection,
}

impl Catchup {
    pub async fn request(&self, req: Request) -> Result<Vec<u8>> {
        proto::request(&self.conn, req).await
    }

    pub async fn close(self) {
        self.conn.close(0u8.into(), b"done");
    }
}

/// Closing the stack down, from either end of a session's life.
///
/// `Clone` and idempotent (`Router::shutdown` and `Endpoint::close` both no-op
/// once run) so that the *setup* path can hold one too: every fallible step of
/// `host` / `join` happens after the endpoint exists, and dropping the pieces
/// closes none of them — the endpoint keeps its relay connection and the gossip,
/// blob and router actors keep running. A user retrying a bad link would
/// accumulate one of each per attempt.
#[derive(Clone)]
pub(crate) struct Shutdown {
    endpoint: Endpoint,
    router: Router,
}

impl Shutdown {
    pub async fn run(&self) {
        if let Err(e) = self.router.shutdown().await {
            tracing::warn!("router shutdown: {e}");
        }
        self.endpoint.close().await;
    }
}
