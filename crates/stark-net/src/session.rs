//! A live shared-drawing session over iroh (DESIGN.md §12.4).
//!
//! One [`CollabSession`] per shared document. The engine stays on the UI
//! thread; the session runs the network side (the mesh receive loop, catch-up
//! server) on spawned tasks and talks to the engine through two thin streams:
//!
//! ```text
//! engine.take_outbox() ──────────► session.broadcast(action) ──► mesh
//! mesh/ALPN ──► RemoteEvent ──► engine.merge_remote / import_brush
//! ```
//!
//! Live actions ride the [`mesh`](crate::mesh) — a flood-with-deduplication
//! broadcast over explicit peer connections. It replaces `iroh-gossip`, whose
//! self-dialing swarm cannot run over WebRTC (see `crate::mesh` for why), while
//! keeping the properties that matter: every peer receives every action, once,
//! even without a direct connection to its author.

#[cfg(not(target_arch = "wasm32"))]
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use iroh::endpoint::presets;
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use n0_future::task;
use stark_core::document::{Action, ActionKind, ActorId, BrushShape};
use stark_core::{AssetId, DocumentFile};
use tokio::sync::mpsc;

use crate::mesh::{Mesh, MeshConfig, MeshEvent, TopicId};
use crate::mirror::Mirror;
use crate::proto::{self, AssetResponse, CollabProto, Request, Wire};
use crate::ticket::SessionTicket;
use crate::transport::iroh::{to_peer_id, IrohMeshTransport};
use crate::Result;

/// A long stroke's fitted control points can be sizeable, so the ceiling sits
/// well past any plausible single action (paths are RDP-simplified; pixels
/// never ride the mesh).
const MAX_MESH_FRAME: usize = 256 * 1024;

/// How long to wait for relay/publish readiness before minting a ticket.
const ONLINE_TIMEOUT: Duration = Duration::from_secs(15);

/// How long a joiner waits to meet the swarm before fetching the snapshot.
const JOIN_TIMEOUT: Duration = Duration::from_secs(10);
const JOIN_POLL: Duration = Duration::from_millis(50);

/// Attempts (with delay) to fetch a brush asset from a peer — it may still be
/// fetching the blob itself.
const ASSET_RETRIES: u32 = 5;
const ASSET_RETRY_DELAY: Duration = Duration::from_millis(300);

/// Map an iroh endpoint identity to the engine's author id (DESIGN.md §12.4:
/// "an iroh node id *is* the `ActorId`"). `ActorId` is 8 bytes to keep every
/// action id small, so this takes the key's first 8 bytes — collisions across
/// the handful of peers in a drawing session are negligible (birthday bound
/// ≈ n²/2⁶⁵), and a collision would only merge two peers' undo scopes.
pub fn actor_from_endpoint_id(id: EndpointId) -> ActorId {
    let bytes = id.as_bytes();
    ActorId(u64::from_le_bytes(bytes[..8].try_into().expect("32-byte key")))
}

/// Something a peer did, to be applied to the local engine. Apply in order:
/// assets arrive before the action that references them.
#[derive(Debug, Clone)]
pub enum RemoteEvent {
    /// A content-addressed brush image a remote stroke references — feed to
    /// [`Engine::import_brush`](stark_core::Engine::import_brush) first.
    Asset { bytes: Vec<u8> },
    /// A committed remote action — feed to
    /// [`Engine::merge_remote`](stark_core::Engine::merge_remote).
    Action(Action),
}

/// Connectivity configuration for a session.
#[derive(Debug, Default, Clone)]
pub struct NetOptions {
    /// Reuse a persisted identity; a fresh key is generated otherwise.
    pub secret: Option<SecretKey>,
    /// Skip the public n0 relay + address-lookup infrastructure and rely on
    /// the ticket's direct socket addresses only — for LAN use and tests.
    pub local_only: bool,
}

impl NetOptions {
    /// LAN/test configuration: no relays, no external lookups.
    pub fn local() -> Self {
        Self {
            local_only: true,
            ..Self::default()
        }
    }
}

/// A live shared session: broadcasts local actions, serves joiners and asset
/// requests, and surfaces remote edits as [`RemoteEvent`]s.
pub struct CollabSession {
    endpoint: Endpoint,
    router: Router,
    topic: TopicId,
    mesh: Mesh,
    mirror: Arc<Mutex<Mirror>>,
    events: Option<mpsc::UnboundedReceiver<RemoteEvent>>,
    ticket_addr: EndpointAddr,
}

impl CollabSession {
    /// Start sharing `doc` (the host side). `doc` should come from
    /// [`Engine::document_file`](stark_core::Engine::document_file) *after*
    /// [`Engine::start_collaboration`](stark_core::Engine::start_collaboration)
    /// with [`actor_from_endpoint_id`] of this session's identity — generate a
    /// [`SecretKey`] first and pass it in `opts` so the actor id is known
    /// before binding.
    pub async fn host(doc: DocumentFile, opts: NetOptions) -> Result<Self> {
        let (endpoint, router, mirror, transport) =
            bind_stack(Mirror::from_file(&doc), &opts).await?;
        // A fresh random 32-byte topic — a secret key is a convenient CSPRNG.
        let topic = TopicId::from_bytes(SecretKey::generate().to_bytes());
        // The first member starts the swarm alone; joiners bootstrap from it.
        let (mesh, events) = Mesh::spawn(transport, mesh_config(topic), []);
        let ticket_addr = reachable_addr(&endpoint, &opts).await;
        Ok(Self::finish(endpoint, router, topic, mesh, events, mirror, ticket_addr))
    }

    /// Join an existing session from a ticket. Returns the session and the
    /// snapshot to load via
    /// [`Engine::join_collaboration`](stark_core::Engine::join_collaboration)
    /// (with [`CollabSession::actor_id`] as the actor).
    pub async fn join(ticket: &SessionTicket, opts: NetOptions) -> Result<(Self, DocumentFile)> {
        let empty = Mirror::from_file(&DocumentFile::new(Vec::new()));
        let (endpoint, router, mirror, transport) = bind_stack(empty, &opts).await?;

        // Connect first: this also teaches the endpoint the peer's address, so
        // the mesh can dial it by bare id below.
        let conn = endpoint.connect(ticket.addr.clone(), proto::ALPN).await?;

        // Enter the live swarm *before* fetching the snapshot: everything
        // before the join is in the snapshot, everything after rides the mesh,
        // and the overlap deduplicates by action id. The ticket's peer is our
        // one contact; the rest of the swarm arrives through peer exchange.
        let (mesh, events) = Mesh::spawn(
            transport,
            mesh_config(ticket.topic),
            [to_peer_id(ticket.addr.id)],
        );
        wait_for_swarm(&mesh).await;

        let snapshot = proto::request(&conn, Request::Snapshot).await?;
        conn.close(0u8.into(), b"joined");
        let file = DocumentFile::from_bytes(&snapshot)?;
        *mirror.lock().expect("mirror poisoned") = Mirror::from_file(&file);

        let ticket_addr = reachable_addr(&endpoint, &opts).await;
        let session =
            Self::finish(endpoint, router, ticket.topic, mesh, events, mirror, ticket_addr);
        Ok((session, file))
    }

    fn finish(
        endpoint: Endpoint,
        router: Router,
        topic: TopicId,
        mesh: Mesh,
        mesh_events: mpsc::UnboundedReceiver<MeshEvent>,
        mirror: Arc<Mutex<Mirror>>,
        ticket_addr: EndpointAddr,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        task::spawn(recv_loop(endpoint.clone(), mesh_events, mirror.clone(), tx));
        Self {
            endpoint,
            router,
            topic,
            mesh,
            mirror,
            events: Some(rx),
            ticket_addr,
        }
    }

    /// The ticket others use to join — every member can hand one out (it
    /// points at *this* peer), so the session survives the host leaving.
    pub fn ticket(&self) -> SessionTicket {
        SessionTicket {
            addr: self.ticket_addr.clone(),
            topic: self.topic,
        }
    }

    /// The author id this session's identity maps to.
    pub fn actor_id(&self) -> ActorId {
        actor_from_endpoint_id(self.endpoint.id())
    }

    /// The stream of remote edits. Take it once and pump it into the engine.
    pub fn take_events(&mut self) -> Option<mpsc::UnboundedReceiver<RemoteEvent>> {
        self.events.take()
    }

    /// A cheap, `Clone` handle for feeding the session from elsewhere (e.g. a
    /// UI task that can't borrow the session across an `await`).
    pub fn broadcaster(&self) -> Broadcaster {
        Broadcaster {
            mesh: self.mesh.clone(),
            mirror: self.mirror.clone(),
        }
    }

    /// Broadcast one locally-committed action (from
    /// [`Engine::take_outbox`](stark_core::Engine::take_outbox)) to the swarm.
    pub async fn broadcast(&self, action: Action) -> Result<()> {
        self.broadcaster().broadcast(action).await
    }

    /// Register a brush image so joiners and asset requests can be served
    /// (call alongside [`Engine::import_brush`](stark_core::Engine::import_brush)).
    pub fn add_asset(&self, id: AssetId, bytes: Vec<u8>) {
        self.mirror
            .lock()
            .expect("mirror poisoned")
            .insert_asset(id, bytes);
    }

    /// Leave the session gracefully.
    pub async fn shutdown(self) {
        self.mesh.shutdown();
        if let Err(e) = self.router.shutdown().await {
            tracing::warn!("router shutdown: {e}");
        }
        self.endpoint.close().await;
    }
}

/// A detached publishing handle onto a [`CollabSession`]: broadcast actions and
/// register assets without holding the session itself. All clones share the
/// same mesh and mirror.
#[derive(Clone)]
pub struct Broadcaster {
    mesh: Mesh,
    mirror: Arc<Mutex<Mirror>>,
}

impl Broadcaster {
    /// See [`CollabSession::broadcast`].
    pub async fn broadcast(&self, action: Action) -> Result<()> {
        self.mirror
            .lock()
            .expect("mirror poisoned")
            .insert(action.clone());
        let bytes = postcard::to_allocvec(&Wire::Action(action))?;
        self.mesh
            .broadcast(bytes)
            .await
            .map_err(|e| crate::NetError::Other(e.to_string()))?;
        Ok(())
    }

    /// See [`CollabSession::add_asset`].
    pub fn add_asset(&self, id: AssetId, bytes: Vec<u8>) {
        self.mirror
            .lock()
            .expect("mirror poisoned")
            .insert_asset(id, bytes);
    }
}

fn mesh_config(topic: TopicId) -> MeshConfig {
    MeshConfig {
        max_frame: MAX_MESH_FRAME,
        ..MeshConfig::new(topic)
    }
}

/// Bind the endpoint and the protocol stack shared by host and joiner: the mesh
/// (live actions) and the catch-up/asset protocol, on one router.
async fn bind_stack(
    mirror: Mirror,
    opts: &NetOptions,
) -> Result<(Endpoint, Router, Arc<Mutex<Mirror>>, IrohMeshTransport)> {
    let secret = opts.secret.clone().unwrap_or_else(SecretKey::generate);
    let endpoint = if opts.local_only {
        Endpoint::builder(presets::Minimal).secret_key(secret).bind().await?
    } else {
        Endpoint::builder(presets::N0).secret_key(secret).bind().await?
    };
    let (transport, mesh_proto) = IrohMeshTransport::new(endpoint.clone());
    let mirror = Arc::new(Mutex::new(mirror));
    let router = Router::builder(endpoint.clone())
        .accept(crate::transport::iroh::ALPN, mesh_proto)
        .accept(
            proto::ALPN,
            CollabProto {
                mirror: mirror.clone(),
            },
        )
        .spawn();
    Ok((endpoint, router, mirror, transport))
}

/// Wait (bounded) until the mesh has met at least one peer, so a joiner does
/// not miss the actions committed while it was fetching the snapshot. Best
/// effort: joining still proceeds if the swarm is slow, since the snapshot plus
/// later mesh traffic still converges.
async fn wait_for_swarm(mesh: &Mesh) {
    let deadline = JOIN_TIMEOUT.as_millis() / JOIN_POLL.as_millis().max(1);
    for _ in 0..deadline {
        match mesh.neighbors().await {
            Ok(neighbors) if !neighbors.is_empty() => return,
            Ok(_) => {}
            Err(_) => return,
        }
        n0_future::time::sleep(JOIN_POLL).await;
    }
    tracing::warn!("joined without meeting a peer yet; relying on catch-up");
}

/// The address peers should dial, for tickets. With public infrastructure we
/// wait (bounded) for the relay handshake so the ticket carries a relay URL;
/// local-only tickets carry the bound sockets (loopback-normalized).
async fn reachable_addr(endpoint: &Endpoint, opts: &NetOptions) -> EndpointAddr {
    if !opts.local_only {
        // `online()` pends forever with no WAN; bound wait, then best effort.
        let _ = n0_future::time::timeout(ONLINE_TIMEOUT, endpoint.online()).await;
        return endpoint.addr();
    }
    // Local-only is native-only: a browser has no UDP sockets to advertise
    // (wasm iroh is relay-borne), so `local_only` there yields a bare-id
    // ticket that only same-machine tests could ever have used anyway.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut addr = EndpointAddr::new(endpoint.id());
        for sock in endpoint.bound_sockets() {
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
        addr
    }
    #[cfg(target_arch = "wasm32")]
    EndpointAddr::new(endpoint.id())
}

/// The mesh receive loop: decode, resolve asset dependencies, mirror, forward
/// to the engine.
async fn recv_loop(
    endpoint: Endpoint,
    mut events: mpsc::UnboundedReceiver<MeshEvent>,
    mirror: Arc<Mutex<Mirror>>,
    tx: mpsc::UnboundedSender<RemoteEvent>,
) {
    while let Some(event) = events.recv().await {
        let (origin, from, payload) = match event {
            MeshEvent::Received { origin, from, payload } => (origin, from, payload),
            MeshEvent::Lagged { origin } => {
                // Dropped messages: peers converge again on the next snapshot
                // fetch; flag it loudly for now (DESIGN.md §12.5).
                tracing::warn!(%origin, "mesh lagged; some remote actions may be missing");
                continue;
            }
            MeshEvent::NeighborUp(peer) => {
                tracing::debug!(%peer, "peer joined the session");
                continue;
            }
            MeshEvent::NeighborDown(peer) => {
                tracing::debug!(%peer, "peer left the session");
                continue;
            }
        };

        let wire: Wire = match postcard::from_bytes(&payload) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("undecodable mesh payload: {e}");
                continue;
            }
        };
        let Wire::Action(action) = wire;

        // Resolve the stroke's brush image before surfacing the action so the
        // engine can render it faithfully (a miss degrades to the round tip
        // rather than blocking the log). The origin authored the stroke and so
        // definitely has the asset; the neighbour that forwarded it may not.
        if let Some(id) = referenced_asset(&action)
            && !mirror.lock().expect("mirror poisoned").has_asset(id)
        {
            let sources = asset_sources(origin, from);
            match fetch_asset(&endpoint, &sources, id).await {
                Some(bytes) => {
                    mirror
                        .lock()
                        .expect("mirror poisoned")
                        .insert_asset(id, bytes.clone());
                    if tx.send(RemoteEvent::Asset { bytes }).is_err() {
                        return;
                    }
                }
                None => {
                    tracing::warn!("brush asset {id:?} unavailable; stroke will fall back")
                }
            }
        }

        let fresh = mirror
            .lock()
            .expect("mirror poisoned")
            .insert(action.clone());
        if fresh && tx.send(RemoteEvent::Action(action)).is_err() {
            return;
        }
    }
}

/// Who to ask for a brush image: its author first, then whoever delivered the
/// action (which may have it cached, and is known to be reachable).
fn asset_sources(origin: crate::mesh::PeerId, from: crate::mesh::PeerId) -> Vec<EndpointId> {
    let mut ids = Vec::with_capacity(2);
    for peer in [origin, from] {
        if let Some(id) = crate::transport::iroh::to_endpoint_id(peer)
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    }
    ids
}

/// The brush image a stroke depends on, if any (DESIGN.md §6.6).
fn referenced_asset(action: &Action) -> Option<AssetId> {
    match &action.kind {
        ActionKind::CommitStroke(rec) => match rec.brush.shape {
            BrushShape::Stamp(id) => Some(id),
            BrushShape::Round => None,
        },
        _ => None,
    }
}

/// Fetch a content-addressed brush image, trying each source in turn and
/// retrying (a peer may still be fetching it itself).
async fn fetch_asset(
    endpoint: &Endpoint,
    sources: &[EndpointId],
    id: AssetId,
) -> Option<Vec<u8>> {
    for attempt in 0..ASSET_RETRIES {
        if attempt > 0 {
            n0_future::time::sleep(ASSET_RETRY_DELAY).await;
        }
        for &source in sources {
            match try_fetch_asset(endpoint, source, id).await {
                Ok(Some(bytes)) => return Some(bytes),
                Ok(None) => continue,
                Err(e) => tracing::debug!("asset fetch attempt {attempt} failed: {e}"),
            }
        }
    }
    None
}

async fn try_fetch_asset(
    endpoint: &Endpoint,
    from: EndpointId,
    id: AssetId,
) -> Result<Option<Vec<u8>>> {
    let conn = endpoint
        .connect(EndpointAddr::new(from), proto::ALPN)
        .await?;
    let response = proto::request(&conn, Request::Asset(id)).await?;
    conn.close(0u8.into(), b"done");
    let AssetResponse(bytes) = postcard::from_bytes(&response)?;
    Ok(bytes)
}

impl std::fmt::Debug for CollabSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollabSession")
            .field("topic", &self.topic)
            .field("endpoint", &self.endpoint.id())
            .finish_non_exhaustive()
    }
}
