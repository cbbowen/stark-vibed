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

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use iroh::endpoint::{Connection, presets};
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey, TransportAddr};
use iroh_blobs::store::mem::MemStore;
use iroh_blobs::{BlobsProtocol, Hash};
use iroh_gossip::net::Gossip;
use tokio::sync::Notify;

use crate::Result;
use crate::content::ContentSource;
use crate::mirror::Served;
use crate::proto::{self, CollabProto, Request};
use crate::session::NetOptions;

/// How long to wait for relay/publish readiness before minting a ticket.
const ONLINE_TIMEOUT: Duration = Duration::from_secs(15);

/// A long stroke's fitted control points can be sizeable, so the gossip
/// message ceiling sits well past any plausible single *drawn* action (paths are
/// RDP-simplified).
///
/// It was a quarter of this, which is close enough to a long stroke at a fine
/// tolerance to be worth widening — and crossing it used to mean the action was
/// gone from every peer that was watching, permanently, because a failed broadcast
/// had nowhere to be retried from. It is survivable now: the sender mirrors before
/// it sends, so [`reconcile`](crate::reconcile) hands the action to whoever sweeps
/// next. A ceiling that a legal document action can cross is still a ceiling that
/// will be crossed, so this is a delay to avoid rather than a loss to prevent.
///
/// **Pixels never ride gossip**, and that is a property of the vocabulary rather than
/// of this number: every kind of content a log can name — a brush shape, a canvas
/// ground, a placed picture (§23) — is named by a content id and moved over the blob
/// ALPN, so what crosses here is always a description. A placed picture is the case
/// that could most easily have been otherwise, and the one this ceiling decided: it is
/// megabytes, it would not fit, and raising the ceiling to suit the largest photograph
/// anyone might paste would size every peer's gossip buffer for it too.
const MAX_MESSAGE_SIZE: usize = 1024 * 1024;

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
            blob_conns: Arc::default(),
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
    /// Live blob connections, by provider.
    ///
    /// A resolver's retry loop asks the same provider on every round, and a
    /// ground's rounds are unbounded — so without this, one 20 KB PNG could cost a
    /// QUIC handshake per round, forever. The retry exists for the case where the
    /// provider is slow, which is exactly when handshaking again costs most.
    ///
    /// A connection that errors is dropped rather than kept: the next round should
    /// re-establish it, not keep asking down a path that has already failed.
    blob_conns: Arc<Mutex<HashMap<EndpointId, Connection>>>,
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

    /// Teach the endpoint how to reach a session member a link names, without
    /// dialing it — what lets gossip bootstrap from *every* member a ticket
    /// names rather than only the one the catch-up connection reached.
    ///
    /// Custom (WebRTC) hops are dropped rather than taught. `add_addr` opens
    /// custom addrs as paths on any live connection to that remote, and a custom
    /// path opened before this side's channel attaches fails validation and
    /// blocks the later real open (see `transport::direct::bootstrap`). Minting
    /// already leaves them off a ticket's extra members — a peer derives one
    /// from the endpoint id itself — so this holds the same line for links
    /// minted by anyone else.
    pub async fn learn(&self, member: &EndpointAddr) {
        let member = EndpointAddr::from_parts(
            member.id,
            member
                .addrs
                .iter()
                .filter(|addr| !matches!(addr, TransportAddr::Custom(_)))
                .cloned(),
        );
        if member.addrs.is_empty() {
            // Nothing to teach: a bare id resolves through address lookup when
            // something dials it, and is not worth a resolve round here.
            return;
        }
        self.endpoint.add_addr(member).await;
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

/// The real content source: hash-verified blob fetches over the session's
/// endpoint. See [`ContentSource`] for why the resolver names the trait and not
/// this.
impl ContentSource for Dialer {
    /// Fetched into the local store as well as returned, so this peer can serve
    /// the content onward.
    async fn fetch_blob(&self, provider: EndpointId, hash: Hash) -> Result<bytes::Bytes> {
        let conn = self.blob_conn(provider).await?;
        if let Err(e) = self.blobs.remote().fetch(conn, hash).await {
            self.blob_conns
                .lock()
                .expect("blob connections poisoned")
                .remove(&provider);
            return Err(e.into());
        }
        Ok(self.blobs.get_bytes(hash).await?)
    }
}

impl Dialer {
    /// The blob connection to `provider`, reused if one is still open.
    async fn blob_conn(&self, provider: EndpointId) -> Result<Connection> {
        let cached = self
            .blob_conns
            .lock()
            .expect("blob connections poisoned")
            .get(&provider)
            .filter(|conn| conn.close_reason().is_none())
            .cloned();
        if let Some(conn) = cached {
            return Ok(conn);
        }
        let conn = self
            .endpoint
            .connect(
                self.dial_addr(EndpointAddr::new(provider)),
                iroh_blobs::ALPN,
            )
            .await?;
        self.blob_conns
            .lock()
            .expect("blob connections poisoned")
            .insert(provider, conn.clone());
        Ok(conn)
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

/// A session's stop signal, held by everything it spawned.
///
/// What this replaces is an inference. The unbounded retries used to key off
/// `Waitlist::is_live` — "does the UI still hold the `Events` receiver" — which is
/// a frontend convention standing in for a session fact, and left
/// [`CollabSession::shutdown`](crate::CollabSession::shutdown) cancelling nothing
/// it had spawned. Both facts are real and either one ends the work, so the
/// resolver now asks both; this is the one that shutting down actually controls.
///
/// [`sleep`](Cancel::sleep) is the reason it carries a `Notify` rather than being
/// a bare flag: a ground's backoff widens to half a minute, and a session that has
/// ended should not wait out the rest of one.
#[derive(Debug, Clone, Default)]
pub(crate) struct Cancel(Arc<CancelInner>);

#[derive(Debug, Default)]
struct CancelInner {
    stopped: AtomicBool,
    woken: Notify,
}

impl Cancel {
    /// End the session's background work. Idempotent.
    pub fn stop(&self) {
        self.0.stopped.store(true, Ordering::Relaxed);
        self.0.woken.notify_waiters();
    }

    pub fn stopped(&self) -> bool {
        self.0.stopped.load(Ordering::Relaxed)
    }

    /// Sleep, unless the session ends first — `false` if it did.
    pub async fn sleep(&self, duration: Duration) -> bool {
        // Registered before the check, so a `stop` racing this cannot slip between
        // the two and leave the sleeper waiting on a notification already sent.
        let woken = self.0.woken.notified();
        if self.stopped() {
            return false;
        }
        n0_future::future::race(
            async {
                n0_future::time::sleep(duration).await;
                true
            },
            async {
                woken.await;
                false
            },
        )
        .await
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
