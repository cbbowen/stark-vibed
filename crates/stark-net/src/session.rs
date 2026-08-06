//! A live shared-drawing session over iroh (§12.4).
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

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use iroh::{EndpointAddr, EndpointId, SecretKey, TransportAddr};
use iroh_blobs::Hash;
use iroh_gossip::api::{Event as GossipEvent, GossipReceiver, GossipSender};
pub use iroh_gossip::proto::TopicId;
use n0_future::{StreamExt, task};
use stark_core::document::{Action, ActionKind, ActorId, BrushShape};
use stark_core::peer::{GestureFrame, PeerFrame};
use stark_core::{AssetId, DocumentFile, SurfaceId};
use tokio::sync::mpsc;

use crate::Result;
use crate::backend::{self, Bound, Dialer, Shutdown};
use crate::mirror::Mirror;
use crate::proto::{Request, Stamped, Wire};
use crate::ticket::SessionTicket;
use crate::waitlist::{Admit, Waitlist};

/// How long a joiner waits to meet the swarm before fetching the snapshot.
const JOIN_TIMEOUT: Duration = Duration::from_secs(10);

/// How many presence frames may sit queued for the engine at once, across all
/// peers.
///
/// Presence is a latch, so a deep queue is not resilience — it is staleness. The
/// engine drops any frame that does not advance `(boot, seq)` anyway (§17.5), so
/// everything queued behind the newest is work the UI will do and throw away.
/// This cap is what stops a UI that has fallen behind from accumulating them
/// without bound: roughly half a second of five peers stroking at 30 Hz, long
/// enough to ride out a slow frame and short enough that what does arrive is
/// still current.
const PRESENCE_QUEUE: usize = 64;

/// Rounds spent fetching a *brush* image before giving up and letting the stroke
/// draw with the round tip. A ground is never given up on — see
/// [`resolve_asset`].
const BRUSH_ATTEMPTS: u32 = 5;
/// The first delay between fetch rounds — a source may still be fetching the
/// blob itself. It doubles up to the cap, so an unbounded retry settles into an
/// occasional poll rather than a busy one.
const ASSET_RETRY_DELAY: Duration = Duration::from_millis(300);
const ASSET_RETRY_MAX_DELAY: Duration = Duration::from_secs(30);

/// Map an iroh endpoint identity to the engine's author id (§12.4:
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

/// Content a remote action needs before it can be applied faithfully, and which
/// store it belongs in (§6.6, §6.4).
///
/// Both arms are the same 32-byte content hash and travel the same way; the split
/// exists because the two decode differently at the far end — a brush mask is
/// luminance × alpha, a ground is channel 0 — so the receiver has to be told which
/// it is being handed. It is *told* rather than left to guess, and it is told by the
/// action that referenced the content, which is the only thing that actually knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetNeed {
    /// A brush shape a stroke stamps with.
    Brush(AssetId),
    /// The canvas ground a `SetSurface` moves the document onto — named by the
    /// [`AssetId`] inside its [`SurfaceId`], the only kind of ground there is bytes
    /// to move for. Missing it is worse than missing a brush: an unresolved shape
    /// degrades to the round tip and the stroke is still visibly a stroke, whereas an
    /// unresolved ground silently drops the deposition tooth (§6.4) and bakes a
    /// smooth deposit into tiles that no later arrival un-bakes.
    Ground(AssetId),
}

impl AssetNeed {
    /// The need a document moving onto `surface` creates — `None` for `Flat`,
    /// which is procedural, has no bytes to move, and so is never waited on.
    ///
    /// This is the only place a ground's `Flat` case is answered. Past it the
    /// need carries an [`AssetId`], so every question about it — what hash it
    /// transfers under, which store it belongs in, whether this peer holds it —
    /// has an answer instead of an answer and a special case.
    pub fn ground(surface: SurfaceId) -> Option<Self> {
        match surface {
            SurfaceId::Flat => None,
            SurfaceId::Image(id) => Some(AssetNeed::Ground(id)),
        }
    }

    /// The id the bytes transfer under.
    pub(crate) fn content(self) -> AssetId {
        match self {
            AssetNeed::Brush(id) | AssetNeed::Ground(id) => id,
        }
    }

    /// The surface a `Ground` need names, for the receiver's
    /// [`Engine::accept_surface`](stark_core::Engine::accept_surface).
    pub fn surface(self) -> Option<SurfaceId> {
        match self {
            AssetNeed::Brush(_) => None,
            AssetNeed::Ground(id) => Some(SurfaceId::Image(id)),
        }
    }
}

/// Something a peer did, to be applied to the local engine. Apply in order:
/// assets arrive before the action that references them.
#[derive(Debug, Clone)]
pub enum RemoteEvent {
    /// Content a remote action references, resolved off a peer — feed to the store
    /// `need` names before the action that wanted it: a brush image to
    /// [`Engine::import_brush`](stark_core::Engine::import_brush), a canvas ground to
    /// [`Engine::accept_surface`](stark_core::Engine::accept_surface).
    Asset { need: AssetNeed, bytes: Bytes },
    /// A committed remote action — feed to
    /// [`Engine::merge_remote`](stark_core::Engine::merge_remote).
    Action(Action),
    /// A peer's presence — feed to
    /// [`Engine::merge_presence`](stark_core::Engine::merge_presence)
    /// (§17.4). Unlike an action this may be dropped freely: nothing in
    /// the log refers to it, so losing one costs a frame of someone else's cursor
    /// and nothing else.
    Presence { actor: ActorId, frame: PeerFrame },
}

/// The stream of remote edits, handed out once by [`CollabSession::host`] /
/// [`CollabSession::join`]. Pump it into the engine.
///
/// It is a type rather than a bare channel because the presence quota is
/// accounted here: a slot is taken when a frame is queued and freed when one is
/// handed over, so "queued presence" means the same number on both ends without
/// the consumer knowing there is a quota at all.
#[derive(Debug)]
pub struct Events {
    rx: mpsc::UnboundedReceiver<RemoteEvent>,
    presence: Arc<PresenceQuota>,
}

impl Events {
    /// The next remote event, or `None` once the session has ended.
    pub async fn recv(&mut self) -> Option<RemoteEvent> {
        let event = self.rx.recv().await?;
        self.took(&event);
        Some(event)
    }

    /// The next remote event if one is already queued — for a pump driven by
    /// something other than this stream, such as a frame clock.
    pub fn try_recv(&mut self) -> Option<RemoteEvent> {
        let event = self.rx.try_recv().ok()?;
        self.took(&event);
        Some(event)
    }

    fn took(&self, event: &RemoteEvent) {
        if matches!(event, RemoteEvent::Presence { .. }) {
            self.presence.release();
        }
    }
}

/// Slots for presence frames in flight to the engine (see [`PRESENCE_QUEUE`]).
#[derive(Debug, Default)]
struct PresenceQuota(AtomicUsize);

impl PresenceQuota {
    /// Take a slot, or `false` when the engine is already this far behind — in
    /// which case the frame is dropped, which is the one thing presence is
    /// allowed to have happen to it. Nothing in the log refers to it, the newest
    /// frame supersedes it, and the author re-sends its whole gesture on the
    /// next resync frame (§17.4, §17.5).
    fn reserve(&self) -> bool {
        self.0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |queued| {
                (queued < PRESENCE_QUEUE).then_some(queued + 1)
            })
            .is_ok()
    }

    fn release(&self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
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
    /// Everything publishing needs, which is everything the session needs but
    /// four — so the session holds one rather than assembling one per call.
    broadcaster: Broadcaster,
    shutdown: Shutdown,
    topic: TopicId,
    ticket_addr: EndpointAddr,
}

impl CollabSession {
    /// Start sharing `doc` (the host side). `doc` should come from
    /// [`Engine::document_file`](stark_core::Engine::document_file) *after*
    /// [`Engine::start_collaboration`](stark_core::Engine::start_collaboration)
    /// with [`actor_from_endpoint_id`] of this session's identity — generate a
    /// [`SecretKey`] first and pass it in `opts` so the actor id is known
    /// before binding.
    pub async fn host(doc: DocumentFile, opts: NetOptions) -> Result<(Self, Events)> {
        let mirror = Arc::new(Mutex::new(Mirror::from_file(&doc)));
        let bound = backend::bind(mirror.clone(), &opts).await?;
        // A fresh random 32-byte topic — a secret key is a convenient CSPRNG.
        let topic = TopicId::from_bytes(SecretKey::generate().to_bytes());
        // The first member starts the swarm alone; joiners bootstrap from it.
        let sub = bound.gossip.subscribe(topic, Vec::new()).await?;
        let ticket_addr = bound.dialer.ticket_addr(&opts).await?;
        Self::finish(bound, topic, sub, mirror, ticket_addr)
    }

    /// Join an existing session from a ticket. Returns the session and the
    /// snapshot to load via
    /// [`Engine::join_collaboration`](stark_core::Engine::join_collaboration)
    /// (with [`CollabSession::actor_id`] as the actor).
    pub async fn join(
        ticket: &SessionTicket,
        opts: NetOptions,
    ) -> Result<(Self, Events, DocumentFile)> {
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
            .await?;
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
        let (session, events) = Self::finish(bound, ticket.topic, sub, mirror, ticket_addr)?;
        Ok((session, events, file))
    }

    fn finish(
        bound: Bound,
        topic: TopicId,
        sub: iroh_gossip::api::GossipTopic,
        mirror: Arc<Mutex<Mirror>>,
        ticket_addr: EndpointAddr,
    ) -> Result<(Self, Events)> {
        let Bound {
            dialer, shutdown, ..
        } = bound;
        let local_id = dialer.local_id();
        let (sender, receiver) = sub.split();
        // Seed with the neighbors met before the receive loop takes over
        // (typically the bootstrap peer a joiner already awaited).
        let neighbors: HashSet<EndpointId> = receiver.neighbors().collect();
        for &peer in &neighbors {
            dialer.ensure_direct(peer);
        }
        let neighbors = Arc::new(Mutex::new(neighbors));
        // Every piece of content already known (the hosted document's, or the
        // joiner's snapshot's) enters the blob store so this peer can serve it, and
        // its transfer hash is recorded so this peer's own actions referencing it can
        // broadcast one. Brush images and canvas grounds alike — both are content an
        // action can be waiting on.
        mirror
            .lock()
            .expect("mirror poisoned")
            .seed_blobs(|bytes| dialer.add_blob(bytes));
        let (tx, rx) = mpsc::unbounded_channel();
        let presence = Arc::new(PresenceQuota::default());
        let waitlist = Arc::new(Waitlist::new(mirror, tx.clone()));
        // The receive loop is the only thing that dials afterwards (to fetch
        // brush assets and bootstrap WebRTC), so it takes the dialer with it.
        task::spawn(recv_loop(
            dialer.clone(),
            receiver,
            neighbors.clone(),
            waitlist.clone(),
            presence.clone(),
            tx,
        ));
        let session = Self {
            broadcaster: Broadcaster {
                local_id,
                sender,
                dialer,
                neighbors,
                waitlist,
            },
            shutdown,
            topic,
            ticket_addr,
        };
        Ok((session, Events { rx, presence }))
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
        actor_from_endpoint_id(self.broadcaster.local_id)
    }

    /// A cheap, `Clone` handle for feeding the session from elsewhere (e.g. a
    /// UI task that can't borrow the session across an `await`).
    pub fn broadcaster(&self) -> Broadcaster {
        self.broadcaster.clone()
    }

    /// Broadcast one locally-committed action (from
    /// [`Engine::take_outbox`](stark_core::Engine::take_outbox)) to the swarm.
    pub async fn broadcast(&self, action: Action) -> Result<()> {
        self.broadcaster.broadcast(action).await
    }

    /// Register content so joiners can be served and peers can fetch it — a brush
    /// image alongside
    /// [`Engine::import_brush`](stark_core::Engine::import_brush), a canvas ground
    /// alongside [`Engine::import_surface`](stark_core::Engine::import_surface).
    ///
    /// Call it *before* committing an action that references the content: the
    /// broadcast attaches a transfer hash looked up here, and an action that goes out
    /// without one leaves receivers unable to fetch what it needs. Getting that order
    /// wrong logs an error naming the content, from the client that committed it —
    /// the fault is only visible there, since what it produces at the far end is
    /// indistinguishable from content that has not arrived yet.
    pub fn add_content(&self, need: AssetNeed, bytes: impl Into<Bytes>) {
        self.broadcaster.add_content(need, bytes);
    }

    /// See [`Broadcaster::links`].
    pub async fn links(&self) -> Vec<PeerLink> {
        self.broadcaster.links().await
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
    waitlist: Arc<Waitlist>,
}

impl Broadcaster {
    /// See [`CollabSession::broadcast`].
    pub async fn broadcast(&self, action: Action) -> Result<()> {
        self.waitlist.published(action.clone());
        let asset = referenced_asset(&action);
        self.publish_wire(Wire::Action(action), asset).await
    }

    /// Publish this client's presence (§17.4).
    ///
    /// Deliberately *not* mirrored: presence is not part of the document, so it is
    /// never served to a joiner and never reaches a file. And deliberately
    /// best-effort — a frame that cannot be sent is dropped rather than retried,
    /// because the next one supersedes it anyway.
    pub async fn publish(&self, frame: PeerFrame) -> Result<()> {
        let asset = referenced_presence_asset(&frame);
        self.publish_wire(Wire::Presence(frame), asset).await
    }

    async fn publish_wire(&self, wire: Wire, need: Option<AssetNeed>) -> Result<()> {
        // Attach the blob hash for the referenced content, so receivers that lack
        // it know what to fetch. `add_content` accompanies the import, for a
        // ground as for a brush, so a registered lookup is the normal case and a
        // miss means that ordering was broken.
        let asset = need.and_then(|need| {
            let hash = self.waitlist.transfer_hash(need.content());
            if hash.is_none() {
                // What a call site that committed before registering looks like
                // from here. Reported at the fault rather than left to surface
                // as a warning on someone else's canvas: with no transfer hash
                // the receiver cannot fetch the bytes at all, and for a ground
                // that is a permanent divergence (§6.4).
                //
                // Sent anyway. The action is already committed locally, so
                // withholding it would guarantee the divergence this is warning
                // about rather than merely risk it — and a peer that gets the
                // content some other way still converges.
                tracing::error!(
                    "broadcasting a payload referencing unregistered {need:?}; \
                     add_content must precede the commit that references it"
                );
            }
            hash
        });
        let stamped = Stamped {
            origin: self.local_id,
            asset,
            wire,
        };
        let bytes = postcard::to_allocvec(&stamped)?;
        Ok(self.sender.broadcast(bytes.into()).await?)
    }

    /// See [`CollabSession::add_content`].
    pub fn add_content(&self, need: AssetNeed, bytes: impl Into<Bytes>) {
        let bytes = bytes.into();
        let hash = self.dialer.add_blob(bytes.clone());
        // Through the waitlist, not straight into the mirror: a remote action
        // may already be parked on exactly this content, and a local import
        // satisfies it as well as a fetch would.
        self.waitlist.imported(need, bytes, hash);
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

/// The gossip receive loop: decode, park what is waiting on content, mirror,
/// forward to the engine. Also maintains the neighbor set and kicks off the
/// WebRTC bootstrap for every new neighbor.
///
/// The loop itself never waits on the network. An action that references
/// content this peer lacks is parked on the [`Waitlist`] and released by the
/// resolver that fetches it; everything else keeps flowing past.
async fn recv_loop(
    dialer: Dialer,
    mut gossip: GossipReceiver,
    neighbors: Arc<Mutex<HashSet<EndpointId>>>,
    waitlist: Arc<Waitlist>,
    presence: Arc<PresenceQuota>,
    tx: mpsc::UnboundedSender<RemoteEvent>,
) {
    while let Some(event) = gossip.next().await {
        let message = match event {
            Ok(GossipEvent::Received(message)) => message,
            Ok(GossipEvent::Lagged) => {
                // Dropped messages: peers converge again on the next snapshot
                // fetch; flag it loudly for now (§12.5). A lagged
                // *presence* stream needs no recovery at all — the author re-sends
                // its whole gesture on the next resync frame (§17.5).
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
                // eventual commit will. Claimed detached — presence must never
                // wait on a fetch — so the rest of the gesture renders with the
                // real shape as soon as the bytes land; until then the
                // receiver's preview degrades to the round tip. The commit that
                // follows names the same content and parks behind *this*
                // resolver rather than starting a second one.
                if let Some(need) = referenced_presence_asset(&frame)
                    && let Some(hash) = hash_or_warn(need, asset_hash)
                    && waitlist.claim_detached(need)
                {
                    task::spawn(resolve_asset(
                        dialer.clone(),
                        asset_sources(origin, from),
                        need,
                        hash,
                        waitlist.clone(),
                    ));
                }
                // Dropped rather than queued when the engine is already
                // `PRESENCE_QUEUE` frames behind: a frame the UI would reach
                // late is one the engine rejects as stale anyway.
                if !presence.reserve() {
                    tracing::trace!("presence frame dropped; the engine is behind");
                    continue;
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

        // Whatever the action references has to reach the engine first, so the
        // engine can apply it faithfully. The action waits for it — parked, not
        // awaited here, so nothing else in the session waits with it. The origin
        // authored the action and so definitely holds the content; the neighbour
        // that forwarded it may not.
        //
        // Falling through the `if` applies the action now, which covers both an
        // action that references nothing and one whose sender attached no
        // transfer hash: there is nothing to fetch, so parking would be parking
        // forever, and the kind's fallback is the best that is available.
        if let Some(need) = referenced_asset(&action)
            && let Some(hash) = hash_or_warn(need, asset_hash)
        {
            match waitlist.claim(need, &action) {
                Admit::Ready => {}
                Admit::Waiting => continue,
                Admit::Fetch => {
                    task::spawn(resolve_asset(
                        dialer.clone(),
                        asset_sources(origin, from),
                        need,
                        hash,
                        waitlist.clone(),
                    ));
                    continue;
                }
            }
        }

        waitlist.accept(action);
        if !waitlist.is_live() {
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

/// The transfer hash for referenced content, or a warning: without it the bytes
/// cannot be fetched (a sender from before its own import completed — which
/// `add_content` ordering prevents — or a version mismatch).
fn hash_or_warn(need: AssetNeed, hash: Option<Hash>) -> Option<Hash> {
    if hash.is_none() {
        tracing::warn!("missing {need:?} arrived without a transfer hash");
    }
    hash
}

/// The brush image a *live* remote gesture depends on, if any: a stroke's head
/// frame carries the full `BrushParams` (§17.5). Only head/resync
/// frames name it — delta frames extend the path of a head already seen.
fn referenced_presence_asset(frame: &PeerFrame) -> Option<AssetNeed> {
    match &frame.gesture {
        Some(GestureFrame::Stroke {
            head: Some(head), ..
        }) => match head.brush.shape {
            BrushShape::Stamp(id) => Some(AssetNeed::Brush(id)),
            BrushShape::Round { .. } => None,
        },
        _ => None,
    }
}

/// Fetch missing content over blobs, mirror it and record its transfer hash (so
/// this peer can serve and announce it onward), surface it to the engine, and
/// release every action parked behind it.
///
/// How long it tries is the one place the two kinds differ, and the difference
/// is §6.4's. An unresolved **brush** degrades to the round tip and the stroke
/// is still visibly a stroke, so after [`BRUSH_ATTEMPTS`] it gives up and lets
/// the action through. An unresolved **ground** has no acceptable fallback:
/// applying the `SetSurface` against `Flat` bakes a smooth deposit into stored
/// tiles that no later arrival un-bakes. So it never gives up — the action
/// simply waits, and nothing else waits with it. Strokes that merged ahead of it
/// are replayed against the real ground when it lands, because an action
/// arriving out of order is exactly what makes the timeline resync (§12.6).
async fn resolve_asset(
    dialer: Dialer,
    sources: Vec<EndpointId>,
    need: AssetNeed,
    hash: Hash,
    waitlist: Arc<Waitlist>,
) {
    let attempts = match need {
        AssetNeed::Brush(_) => Some(BRUSH_ATTEMPTS),
        AssetNeed::Ground(_) => None,
    };
    let Some(bytes) = fetch_asset(&dialer, &sources, hash, attempts, &waitlist).await else {
        // Only a brush gives up (a ground retries until the session ends), so
        // releasing here is releasing to the round-tip fallback.
        tracing::warn!("{need:?} unavailable; the stroke will draw with the round tip");
        waitlist.abandoned(need);
        return;
    };
    // Recording the transfer hash with the bytes is what lets this peer announce
    // the content onward on its own actions.
    waitlist.resolved(need, bytes, hash);
}

/// The content an action depends on, if any: the brush image a stroke stamps with
/// (§6.6), or the ground a `SetSurface` moves onto (§6.4).
///
/// The ground arm is what stops two clients diverging over the tooth. Before it, a
/// `SetSurface` naming a ground the receiver had never fetched was applied anyway —
/// the registry fell back to `Flat`, and every stroke after it deposited as though
/// the canvas were smooth. It reads like an asset-loading problem, and it was: the
/// ground was simply not on the list of things an action could be waiting for.
fn referenced_asset(action: &Action) -> Option<AssetNeed> {
    match &action.kind {
        ActionKind::CommitStroke(rec) => match rec.brush.shape {
            BrushShape::Stamp(id) => Some(AssetNeed::Brush(id)),
            BrushShape::Round { .. } => None,
        },
        ActionKind::SetSurface(id) => AssetNeed::ground(*id),
        _ => None,
    }
}

/// Fetch one content blob, trying each source in turn on a widening backoff (a
/// source may still be fetching it itself). `attempts` caps the rounds; `None`
/// retries until the content arrives or the session ends. The transfer is
/// hash-verified by blobs.
async fn fetch_asset(
    dialer: &Dialer,
    sources: &[EndpointId],
    hash: Hash,
    attempts: Option<u32>,
    waitlist: &Waitlist,
) -> Option<Bytes> {
    let mut delay = ASSET_RETRY_DELAY;
    let mut round = 0u32;
    loop {
        for &source in sources {
            match dialer.fetch_blob(source, hash).await {
                Ok(bytes) => return Some(bytes),
                Err(e) => tracing::debug!("asset fetch round {round} failed: {e}"),
            }
        }
        round = round.saturating_add(1);
        if attempts.is_some_and(|max| round >= max) {
            return None;
        }
        // The engine is gone; an uncapped retry would otherwise outlive it.
        if !waitlist.is_live() {
            return None;
        }
        n0_future::time::sleep(delay).await;
        delay = (delay * 2).min(ASSET_RETRY_MAX_DELAY);
    }
}

impl std::fmt::Debug for CollabSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollabSession")
            .field("topic", &self.topic)
            .field("endpoint", &self.broadcaster.local_id)
            .finish_non_exhaustive()
    }
}
