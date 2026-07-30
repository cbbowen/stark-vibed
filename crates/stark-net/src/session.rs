//! A live shared-drawing session over iroh (DESIGN.md §12.4).
//!
//! One [`CollabSession`] per shared document. The engine stays on the UI
//! thread; the session runs the network side (the gossip receive loop, catch-up
//! server) on spawned tasks and talks to the engine through two thin streams:
//!
//! ```text
//! engine.take_outbox() ──────────► session.broadcast(action) ──► gossip
//! gossip/ALPN ──► RemoteEvent ──► engine.merge_remote / import_brush
//! ```
//!
//! Live actions ride `iroh-gossip` on the session's topic: every peer receives
//! every action, once, even without a direct connection to its author. Gossip
//! shares the session's one endpoint with the catch-up ALPN — and, with the
//! `webrtc` feature, with the WebRTC custom transport that gives browsers
//! direct paths (each new gossip neighbor triggers a channel bootstrap; see
//! [`transport::direct`](crate::transport)).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use iroh::{EndpointAddr, EndpointId, SecretKey, TransportAddr};
use iroh_blobs::Hash;
use iroh_gossip::api::{Event as GossipEvent, GossipReceiver, GossipSender};
pub use iroh_gossip::proto::TopicId;
use n0_future::{StreamExt, task};
use stark_core::document::{Action, ActionKind, ActorId, BrushShape};
use stark_core::peer::{GestureFrame, PeerFrame};
use stark_core::{AssetId, DocumentFile};
use tokio::sync::mpsc;

use crate::Result;
use crate::backend::{self, Dialer, Shutdown};
use crate::mirror::Mirror;
use crate::proto::{Request, Stamped, Wire};
use crate::ticket::SessionTicket;

/// [`AssetId`] → the blob hash its canonical bytes transfer under. An asset id
/// names the *decoded coverage* (encoding-independent), so it is not directly
/// fetchable over blobs; whoever holds the bytes knows both names and carries
/// the translation in [`Stamped`].
type AssetHashes = Arc<Mutex<HashMap<AssetId, Hash>>>;

/// How long a joiner waits to meet the swarm before fetching the snapshot.
const JOIN_TIMEOUT: Duration = Duration::from_secs(10);

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
    ActorId(u64::from_le_bytes(
        bytes[..8].try_into().expect("32-byte key"),
    ))
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
    /// A peer's presence — feed to
    /// [`Engine::merge_presence`](stark_core::Engine::merge_presence)
    /// (PEER_DESIGN.md §4). Unlike an action this may be dropped freely: nothing in
    /// the log refers to it, so losing one costs a frame of someone else's cursor
    /// and nothing else.
    Presence { actor: ActorId, frame: PeerFrame },
}

/// How a direct connection to a session member currently travels. Gossip
/// links may migrate (relay first, then direct once hole punching or a WebRTC
/// bootstrap lands), so this is sampled at query time rather than recorded at
/// connection time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    /// A peer-to-peer WebRTC data channel (the browser's direct path).
    WebRtc,
    /// A hole-punched UDP path (native iroh's direct path).
    Direct,
    /// Via an iroh relay server.
    Relay,
    /// No path selected yet.
    Unknown,
}

/// How this client's connection to one session member currently travels —
/// the answer to "are we peer-to-peer or riding a relay?".
///
/// Keyed by the member's [`ActorId`] so the UI can join it against the
/// presence roster. Members with no entry are not gossip neighbors of this
/// client; their traffic is forwarded by whoever is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerLink {
    /// The author id the peer's endpoint identity maps to.
    pub actor: ActorId,
    /// How the connection reaches them right now.
    pub kind: LinkKind,
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
    local_id: EndpointId,
    shutdown: Shutdown,
    topic: TopicId,
    dialer: Dialer,
    sender: GossipSender,
    neighbors: Arc<Mutex<HashSet<EndpointId>>>,
    asset_hashes: AssetHashes,
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
        let mirror = Arc::new(Mutex::new(Mirror::from_file(&doc)));
        let bound = backend::bind(mirror.clone(), &opts).await?;
        // A fresh random 32-byte topic — a secret key is a convenient CSPRNG.
        let topic = TopicId::from_bytes(SecretKey::generate().to_bytes());
        // The first member starts the swarm alone; joiners bootstrap from it.
        let sub = bound
            .gossip
            .subscribe(topic, Vec::new())
            .await
            .map_err(|e| crate::NetError::Other(e.to_string()))?;
        let ticket_addr = bound.dialer.ticket_addr(&opts).await?;
        Self::finish(
            bound.dialer,
            bound.shutdown,
            topic,
            sub,
            mirror,
            ticket_addr,
        )
    }

    /// Join an existing session from a ticket. Returns the session and the
    /// snapshot to load via
    /// [`Engine::join_collaboration`](stark_core::Engine::join_collaboration)
    /// (with [`CollabSession::actor_id`] as the actor).
    pub async fn join(ticket: &SessionTicket, opts: NetOptions) -> Result<(Self, DocumentFile)> {
        let mirror = Arc::new(Mutex::new(Mirror::from_file(
            &DocumentFile::new(Vec::new()),
        )));
        let bound = backend::bind(mirror.clone(), &opts).await?;

        // Open the catch-up connection first: this also teaches the endpoint
        // the peer's address, so gossip can dial it by bare id below.
        let catchup = bound.dialer.open(ticket.addr.clone()).await?;

        // Enter the live swarm *before* fetching the snapshot: everything
        // before the join is in the snapshot, everything after rides gossip,
        // and the overlap deduplicates by action id. The ticket's peer is our
        // one bootstrap; the rest of the swarm arrives through gossip's
        // membership exchange. Best effort: joining still proceeds if the
        // swarm is slow, since the snapshot plus later traffic still converges.
        let mut sub = bound
            .gossip
            .subscribe(ticket.topic, vec![ticket.addr.id])
            .await
            .map_err(|e| crate::NetError::Other(e.to_string()))?;
        if n0_future::time::timeout(JOIN_TIMEOUT, sub.joined())
            .await
            .is_err()
        {
            tracing::warn!("joined without meeting a peer yet; relying on catch-up");
        }

        let snapshot = catchup.request(Request::Snapshot).await?;
        catchup.close().await;
        let file = DocumentFile::from_bytes(&snapshot)?;
        *mirror.lock().expect("mirror poisoned") = Mirror::from_file(&file);

        let ticket_addr = bound.dialer.ticket_addr(&opts).await?;
        let session = Self::finish(
            bound.dialer,
            bound.shutdown,
            ticket.topic,
            sub,
            mirror,
            ticket_addr,
        )?;
        Ok((session, file))
    }

    fn finish(
        dialer: Dialer,
        shutdown: Shutdown,
        topic: TopicId,
        sub: iroh_gossip::api::GossipTopic,
        mirror: Arc<Mutex<Mirror>>,
        ticket_addr: EndpointAddr,
    ) -> Result<Self> {
        let local_id = dialer.local_id()?;
        let (sender, receiver) = sub.split();
        // Seed with the neighbors met before the receive loop takes over
        // (typically the bootstrap peer a joiner already awaited).
        let neighbors: HashSet<EndpointId> = receiver.neighbors().collect();
        for &peer in &neighbors {
            dialer.ensure_direct(peer);
        }
        let neighbors = Arc::new(Mutex::new(neighbors));
        // Every asset already known (the hosted document's, or the joiner's
        // snapshot's) enters the blob store so this peer can serve it, and the
        // hash map so this peer's own strokes referencing it can broadcast the
        // transfer hash.
        let asset_hashes: AssetHashes = Arc::new(Mutex::new(
            mirror
                .lock()
                .expect("mirror poisoned")
                .assets()
                .into_iter()
                .map(|(id, bytes)| (id, dialer.add_blob(bytes)))
                .collect(),
        ));
        let (tx, rx) = mpsc::unbounded_channel();
        // The receive loop is the only thing that dials afterwards (to fetch
        // brush assets and bootstrap WebRTC), so it takes the dialer with it.
        task::spawn(recv_loop(
            dialer.clone(),
            receiver,
            neighbors.clone(),
            asset_hashes.clone(),
            mirror.clone(),
            tx,
        ));
        Ok(Self {
            local_id,
            shutdown,
            topic,
            dialer,
            sender,
            neighbors,
            asset_hashes,
            mirror,
            events: Some(rx),
            ticket_addr,
        })
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
        actor_from_endpoint_id(self.local_id)
    }

    /// The stream of remote edits. Take it once and pump it into the engine.
    pub fn take_events(&mut self) -> Option<mpsc::UnboundedReceiver<RemoteEvent>> {
        self.events.take()
    }

    /// A cheap, `Clone` handle for feeding the session from elsewhere (e.g. a
    /// UI task that can't borrow the session across an `await`).
    pub fn broadcaster(&self) -> Broadcaster {
        Broadcaster {
            local_id: self.local_id,
            sender: self.sender.clone(),
            dialer: self.dialer.clone(),
            neighbors: self.neighbors.clone(),
            asset_hashes: self.asset_hashes.clone(),
            mirror: self.mirror.clone(),
        }
    }

    /// Broadcast one locally-committed action (from
    /// [`Engine::take_outbox`](stark_core::Engine::take_outbox)) to the swarm.
    pub async fn broadcast(&self, action: Action) -> Result<()> {
        self.broadcaster().broadcast(action).await
    }

    /// Register a brush image so joiners can be served and peers can fetch it
    /// (call alongside [`Engine::import_brush`](stark_core::Engine::import_brush)).
    pub fn add_asset(&self, id: AssetId, bytes: Vec<u8>) {
        self.broadcaster().add_asset(id, bytes);
    }

    /// See [`Broadcaster::links`].
    pub async fn links(&self) -> Vec<PeerLink> {
        self.broadcaster().links().await
    }

    /// Leave the session gracefully.
    pub async fn shutdown(self) {
        self.shutdown.run().await;
    }
}

/// A detached publishing handle onto a [`CollabSession`]: broadcast actions and
/// register assets without holding the session itself. All clones share the
/// same gossip topic and mirror.
#[derive(Clone)]
pub struct Broadcaster {
    local_id: EndpointId,
    sender: GossipSender,
    dialer: Dialer,
    neighbors: Arc<Mutex<HashSet<EndpointId>>>,
    asset_hashes: AssetHashes,
    mirror: Arc<Mutex<Mirror>>,
}

impl Broadcaster {
    /// See [`CollabSession::broadcast`].
    pub async fn broadcast(&self, action: Action) -> Result<()> {
        self.mirror
            .lock()
            .expect("mirror poisoned")
            .insert(action.clone());
        let asset = referenced_asset(&action);
        self.publish_wire(Wire::Action(action), asset).await
    }

    /// Publish this client's presence (PEER_DESIGN.md §4).
    ///
    /// Deliberately *not* mirrored: presence is not part of the document, so it is
    /// never served to a joiner and never reaches a file. And deliberately
    /// best-effort — a frame that cannot be sent is dropped rather than retried,
    /// because the next one supersedes it anyway.
    pub async fn publish(&self, frame: PeerFrame) -> Result<()> {
        let asset = referenced_presence_asset(&frame);
        self.publish_wire(Wire::Presence(frame), asset).await
    }

    async fn publish_wire(&self, wire: Wire, asset: Option<AssetId>) -> Result<()> {
        // Attach the blob hash for the referenced brush image, so receivers
        // that lack it know what to fetch. Registered before the stroke could
        // have been drawn (add_asset accompanies the import), so the lookup
        // only misses for the round tip (no asset at all).
        let asset = asset.and_then(|id| {
            self.asset_hashes
                .lock()
                .expect("asset hashes poisoned")
                .get(&id)
                .copied()
        });
        let stamped = Stamped {
            origin: self.local_id,
            asset,
            wire,
        };
        let bytes = postcard::to_allocvec(&stamped)?;
        self.sender
            .broadcast(bytes.into())
            .await
            .map_err(|e| crate::NetError::Other(e.to_string()))
    }

    /// See [`CollabSession::add_asset`].
    pub fn add_asset(&self, id: AssetId, bytes: Vec<u8>) {
        self.mirror
            .lock()
            .expect("mirror poisoned")
            .insert_asset(id, bytes.clone());
        let hash = self.dialer.add_blob(bytes);
        self.asset_hashes
            .lock()
            .expect("asset hashes poisoned")
            .insert(id, hash);
    }

    /// How each gossip-neighbor session member is reached right now — direct
    /// (WebRTC or hole-punched UDP) or via a relay. Sampled per call; a link
    /// migrates from relay to direct when hole punching or a WebRTC bootstrap
    /// lands, so poll rather than cache.
    pub async fn links(&self) -> Vec<PeerLink> {
        let neighbors: Vec<EndpointId> = self
            .neighbors
            .lock()
            .expect("neighbors poisoned")
            .iter()
            .copied()
            .collect();
        let mut links = Vec::with_capacity(neighbors.len());
        for id in neighbors {
            let kind = match self.dialer.selected_addr(id).await {
                Some(TransportAddr::Custom(_)) => LinkKind::WebRtc,
                Some(TransportAddr::Ip(_)) => LinkKind::Direct,
                Some(TransportAddr::Relay(_)) => LinkKind::Relay,
                _ => LinkKind::Unknown,
            };
            links.push(PeerLink {
                actor: actor_from_endpoint_id(id),
                kind,
            });
        }
        links
    }
}

/// The gossip receive loop: decode, resolve asset dependencies, mirror,
/// forward to the engine. Also maintains the neighbor set and kicks off the
/// WebRTC bootstrap for every new neighbor.
async fn recv_loop(
    dialer: Dialer,
    mut gossip: GossipReceiver,
    neighbors: Arc<Mutex<HashSet<EndpointId>>>,
    asset_hashes: AssetHashes,
    mirror: Arc<Mutex<Mirror>>,
    tx: mpsc::UnboundedSender<RemoteEvent>,
) {
    while let Some(event) = gossip.next().await {
        let message = match event {
            Ok(GossipEvent::Received(message)) => message,
            Ok(GossipEvent::Lagged) => {
                // Dropped messages: peers converge again on the next snapshot
                // fetch; flag it loudly for now (DESIGN.md §12.5). A lagged
                // *presence* stream needs no recovery at all — the author re-sends
                // its whole gesture on the next resync frame (PEER_DESIGN.md §5).
                tracing::warn!("gossip lagged; some remote actions may be missing");
                continue;
            }
            Ok(GossipEvent::NeighborUp(peer)) => {
                tracing::debug!(%peer, "gossip neighbor up");
                neighbors.lock().expect("neighbors poisoned").insert(peer);
                dialer.ensure_direct(peer);
                continue;
            }
            Ok(GossipEvent::NeighborDown(peer)) => {
                tracing::debug!(%peer, "gossip neighbor down");
                neighbors.lock().expect("neighbors poisoned").remove(&peer);
                continue;
            }
            Err(e) => {
                tracing::debug!("gossip receiver closed: {e}");
                return;
            }
        };

        let Stamped {
            origin,
            asset: asset_hash,
            wire,
        } = match postcard::from_bytes(&message.content) {
            Ok(stamped) => stamped,
            Err(e) => {
                tracing::warn!("undecodable gossip payload: {e}");
                continue;
            }
        };
        let from = message.delivered_from;

        let action = match wire {
            Wire::Action(action) => action,
            // Presence bypasses the mirror entirely: it is not part of the document,
            // so it is never served to a joiner and never reaches a file.
            Wire::Presence(frame) => {
                // A live stroke's head names its brush image just like the
                // eventual commit will. Resolve it detached — presence must
                // never wait on a fetch — so the rest of the gesture renders
                // with the real shape as soon as the bytes land; until then
                // the receiver's preview degrades to the round tip.
                if let Some(asset) = referenced_presence_asset(&frame)
                    && !mirror.lock().expect("mirror poisoned").has_asset(asset)
                    && let Some(hash) = require_hash(asset, asset_hash)
                {
                    task::spawn(resolve_asset(
                        dialer.clone(),
                        asset_sources(origin, from),
                        asset,
                        hash,
                        asset_hashes.clone(),
                        mirror.clone(),
                        tx.clone(),
                    ));
                }
                let event = RemoteEvent::Presence {
                    actor: actor_from_endpoint_id(origin),
                    frame,
                };
                if tx.send(event).is_err() {
                    return;
                }
                continue;
            }
        };

        // Resolve the stroke's brush image before surfacing the action so the
        // engine can render it faithfully (a miss degrades to the round tip
        // rather than blocking the log). The origin authored the stroke and so
        // definitely has the asset; the neighbour that forwarded it may not.
        if let Some(id) = referenced_asset(&action)
            && !mirror.lock().expect("mirror poisoned").has_asset(id)
            && let Some(hash) = require_hash(id, asset_hash)
        {
            // Awaited (not spawned): the Asset event must reach the engine
            // before the Action that references it.
            resolve_asset(
                dialer.clone(),
                asset_sources(origin, from),
                id,
                hash,
                asset_hashes.clone(),
                mirror.clone(),
                tx.clone(),
            )
            .await;
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
fn asset_sources(origin: EndpointId, from: EndpointId) -> Vec<EndpointId> {
    let mut ids = vec![origin];
    if from != origin {
        ids.push(from);
    }
    ids
}

/// The transfer hash for a referenced-but-missing asset, or a warning: without
/// it the image cannot be fetched (a sender from before its own import
/// completed — which `add_asset` ordering prevents — or a version mismatch).
fn require_hash(id: AssetId, hash: Option<Hash>) -> Option<Hash> {
    if hash.is_none() {
        tracing::warn!("missing brush asset {id:?} arrived without a transfer hash");
    }
    hash
}

/// The brush image a *live* remote gesture depends on, if any: a stroke's head
/// frame carries the full `BrushParams` (PEER_DESIGN.md §5). Only head/resync
/// frames name it — delta frames extend the path of a head already seen.
fn referenced_presence_asset(frame: &PeerFrame) -> Option<AssetId> {
    match &frame.gesture {
        Some(GestureFrame::Stroke {
            head: Some(head), ..
        }) => match head.brush.shape {
            BrushShape::Stamp(id) => Some(id),
            BrushShape::Round => None,
        },
        _ => None,
    }
}

/// Fetch a missing brush image over blobs, mirror it and record its transfer
/// hash (so this peer can serve and announce it onward), and surface it to
/// the engine. The action path awaits this — assets must precede the action
/// that references them — while the presence path runs it detached. Two
/// concurrent resolvers are harmless: the blob is content-addressed, so the
/// second insert and import are idempotent.
async fn resolve_asset(
    dialer: Dialer,
    sources: Vec<EndpointId>,
    id: AssetId,
    hash: Hash,
    asset_hashes: AssetHashes,
    mirror: Arc<Mutex<Mirror>>,
    tx: mpsc::UnboundedSender<RemoteEvent>,
) {
    match fetch_asset(&dialer, &sources, hash).await {
        Some(bytes) => {
            mirror
                .lock()
                .expect("mirror poisoned")
                .insert_asset(id, bytes.clone());
            asset_hashes
                .lock()
                .expect("asset hashes poisoned")
                .insert(id, hash);
            let _ = tx.send(RemoteEvent::Asset { bytes });
        }
        None => tracing::warn!("brush asset {id:?} unavailable; stroke will fall back"),
    }
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

/// Fetch a brush image blob, trying each source in turn and retrying (a peer
/// may still be fetching it itself). The transfer is hash-verified by blobs.
async fn fetch_asset(dialer: &Dialer, sources: &[EndpointId], hash: Hash) -> Option<Vec<u8>> {
    for attempt in 0..ASSET_RETRIES {
        if attempt > 0 {
            n0_future::time::sleep(ASSET_RETRY_DELAY).await;
        }
        for &source in sources {
            match dialer.fetch_blob(source, hash).await {
                Ok(bytes) => return Some(bytes),
                Err(e) => tracing::debug!("asset fetch attempt {attempt} failed: {e}"),
            }
        }
    }
    None
}

impl std::fmt::Debug for CollabSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollabSession")
            .field("topic", &self.topic)
            .field("endpoint", &self.local_id)
            .finish_non_exhaustive()
    }
}
