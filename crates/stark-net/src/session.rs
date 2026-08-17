//! A live shared-drawing session over iroh (§12.4).
//!
//! One [`CollabSession`] per shared document. The engine stays on the UI
//! thread; the session runs the network side on spawned tasks — the gossip
//! receive loop, the send queue, the catch-up server, the reconciler — and talks
//! to the engine through two thin streams:
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
//!
//! *Once*, but not *reliably*: a flood can drop a message, and what it drops
//! [`reconcile`](crate::reconcile) fetches back. Nothing in here treats gossip as
//! a delivery guarantee.

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
use stark_engine::DocumentFile;
use stark_engine::document::{Action, ActorId, BrushShape};
use stark_engine::peer::{GestureFrame, PeerFrame};
use stark_model::AssetId;
use tokio::sync::mpsc;

use crate::Result;
use crate::backend::{self, Bound, Cancel, Dialer, Shutdown};
use crate::content::Resolver;
use crate::mirror::{Mirror, Served};
use crate::proto::{Request, Stamped, StampedRef, Wire, WireRef};
use crate::reconcile::{Prompt, Reconciler};
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
/// store it belongs in — [`stark_engine::AssetNeed`], re-exported so a frontend
/// pumping this transport does not need to name two crates for one idea.
///
/// It lives in the engine because the engine is what has the two stores, and
/// because loading a file asks the same question a joining peer does.
pub use stark_engine::AssetNeed;

/// Something a peer did, to be applied to the local engine. Apply in order:
/// assets arrive before the action that references them.
#[derive(Debug, Clone)]
pub enum RemoteEvent {
    /// Content a remote action references, resolved off a peer — feed to the store
    /// `need` names before the action that wanted it: a brush image to
    /// [`Engine::import_brush`](stark_engine::Engine::import_brush), a canvas ground to
    /// [`Engine::accept_surface`](stark_engine::Engine::accept_surface).
    Asset { need: AssetNeed, bytes: Bytes },
    /// A committed remote action — feed to
    /// [`Engine::merge_remote`](stark_engine::Engine::merge_remote).
    Action(Action),
    /// A peer's presence — feed to
    /// [`Engine::merge_presence`](stark_engine::Engine::merge_presence)
    /// (§17.4). Unlike an action this may be dropped freely: nothing in
    /// the log refers to it, so losing one costs a frame of someone else's cursor
    /// and nothing else.
    Presence { actor: ActorId, frame: PeerFrame },
    /// A remote action needs content this client said it could resolve itself
    /// ([`NetOptions::resolvable`]) — the promise being called in.
    ///
    /// Read the bytes from wherever you promised they were and hand them back
    /// with [`CollabSession::add_content`]; the action waiting on them is released
    /// when you do. Nothing else is expected of you and nothing is on fire.
    ///
    /// **Ignoring it is safe.** After a short grace period the transport dials a
    /// peer for the content exactly as it would have without the promise, so a
    /// frontend that cannot deliver — or does not handle this at all — loses a
    /// little time and nothing else.
    ResolveLocally { need: AssetNeed },
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
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |queued| {
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
    /// Content this client can produce **without asking anyone** — the ids of the
    /// assets that ship with its build (§12.4).
    ///
    /// It does two things, and they are the same promise read from either end. A
    /// joiner's list is left out of the snapshot it is sent ([`Joined::owed`] is
    /// the bill). And for the rest of the session, a remote action naming one of
    /// these raises [`RemoteEvent::ResolveLocally`] instead of dialling a peer —
    /// so a collaborator switching to a ground this app ships with costs a read
    /// from disk rather than megabytes over the wire.
    ///
    /// Empty promises nothing, and is the safe default: every id then travels the
    /// wire as it would with no local catalog at all.
    pub resolvable: Vec<AssetId>,
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

/// What [`CollabSession::join`] hands back.
///
/// A struct rather than a tuple because of `owed`: a fourth positional value
/// that must be acted on *before* the third is replayed is exactly the kind of
/// thing a caller destructures past.
pub struct Joined {
    pub session: CollabSession,
    pub events: Events,
    /// The snapshot to load via
    /// [`Engine::join_collaboration`](stark_engine::Engine::join_collaboration) —
    /// **after** `owed` is settled.
    pub document: DocumentFile,
    /// Content `document`'s log names that the host left out, because this client
    /// said it could resolve it locally (`resolvable`).
    ///
    /// **Install every one of these into the engine before replaying the
    /// document.** Replay reads the ground in force when each stroke was made
    /// (§6.4), so a `SetSurface` whose height map is not registered yet replays
    /// against the flat stand-in and bakes a smooth deposit that no later arrival
    /// un-bakes — the divergence content-addressing exists to prevent, arrived at
    /// by way of an optimization.
    ///
    /// If a promise cannot be kept, do nothing: the log still names the content,
    /// so the ordinary blob fetch pulls it off a peer exactly as it would have
    /// without the promise. Being wrong here costs a transfer, not a picture.
    pub owed: Vec<AssetNeed>,
}

/// A live shared session: broadcasts local actions, serves joiners and asset
/// requests, and surfaces remote edits as [`RemoteEvent`]s.
pub struct CollabSession {
    /// Everything publishing needs, which is everything the session needs but
    /// four — so the session holds one rather than assembling one per call.
    broadcaster: Broadcaster,
    shutdown: Shutdown,
    /// Stops what the session spawned. Held here because this is what owns the
    /// session's lifetime; see [`Cancel`].
    cancel: Cancel,
    topic: TopicId,
    ticket_addr: EndpointAddr,
}

impl CollabSession {
    /// Start sharing `doc` (the host side). `doc` should come from
    /// [`Engine::document_file`](stark_engine::Engine::document_file) *after*
    /// [`Engine::start_collaboration`](stark_engine::Engine::start_collaboration)
    /// with [`actor_from_endpoint_id`] of this session's identity — generate a
    /// [`SecretKey`] first and pass it in `opts` so the actor id is known
    /// before binding.
    pub async fn host(doc: DocumentFile, opts: NetOptions) -> Result<(Self, Events)> {
        let served = Served::default();
        let bound = backend::bind(served.clone(), &opts).await?;
        let shutdown = bound.shutdown.clone();
        closing_on_error(shutdown, Self::hosting(bound, served, doc, &opts)).await
    }

    /// Join an existing session from a ticket.
    ///
    /// [`NetOptions::resolvable`] is what this client can produce without asking
    /// anyone; the host leaves it out of the snapshot, and [`Joined::owed`] is the
    /// bill for that.
    pub async fn join(ticket: &SessionTicket, opts: NetOptions) -> Result<Joined> {
        let served = Served::default();
        let bound = backend::bind(served.clone(), &opts).await?;
        let shutdown = bound.shutdown.clone();
        closing_on_error(shutdown, Self::joining(bound, served, ticket, &opts)).await
    }

    /// Everything about hosting that can fail after the endpoint exists.
    async fn hosting(
        bound: Bound,
        served: Served,
        doc: DocumentFile,
        opts: &NetOptions,
    ) -> Result<(Self, Events)> {
        let mirror = Arc::new(Mutex::new(Mirror::from_file(&doc)));
        // A fresh random 32-byte topic — a secret key is a convenient CSPRNG.
        let topic = TopicId::from_bytes(SecretKey::generate().to_bytes());
        // The first member starts the swarm alone; joiners bootstrap from it.
        let sub = bound.gossip.subscribe(topic, Vec::new()).await?;
        let ticket_addr = bound.dialer.ticket_addr(opts).await?;
        Self::finish(
            bound,
            served,
            topic,
            sub,
            mirror,
            ticket_addr,
            &opts.resolvable,
        )
    }

    /// Everything about joining that can fail after the endpoint exists.
    async fn joining(
        bound: Bound,
        served: Served,
        ticket: &SessionTicket,
        opts: &NetOptions,
    ) -> Result<Joined> {
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

        let request = if opts.resolvable.is_empty() {
            Request::Snapshot
        } else {
            Request::SnapshotWithout(opts.resolvable.clone())
        };
        let snapshot = catchup.request(request).await?;
        catchup.close().await;
        let file = DocumentFile::from_bytes(&snapshot)?;
        // What the log names but the bundle no longer carries — the bill for the
        // promise. Worked out by the file itself rather than here, because it is
        // the same question loading a document off disk asks, and one definition
        // of "what does this log need" is what stops the two drifting.
        let owed = file.unbundled_content();
        let mirror = Arc::new(Mutex::new(Mirror::from_file(&file)));

        let ticket_addr = bound.dialer.ticket_addr(opts).await?;
        let (session, events) = Self::finish(
            bound,
            served,
            ticket.topic,
            sub,
            mirror,
            ticket_addr,
            &opts.resolvable,
        )?;
        Ok(Joined {
            session,
            events,
            document: file,
            owed,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        bound: Bound,
        served: Served,
        topic: TopicId,
        sub: iroh_gossip::api::GossipTopic,
        mirror: Arc<Mutex<Mirror>>,
        ticket_addr: EndpointAddr,
        resolvable: &[AssetId],
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
        // Only now is there a session to serve, and only now does the catch-up
        // protocol have anything to answer with: seeded, so what a snapshot names
        // can also be fetched piecemeal afterwards.
        served.publish(mirror.clone());
        let (tx, rx) = mpsc::unbounded_channel();
        let presence = Arc::new(PresenceQuota::default());
        let waitlist = Arc::new(Waitlist::new(mirror, tx.clone(), resolvable));
        // The receive loop is the only thing that dials afterwards (to fetch
        // brush assets and bootstrap WebRTC), so it takes the dialer with it.
        let cancel = Cancel::default();
        let wiring = Wiring {
            resolver: Resolver::new(
                dialer.clone(),
                neighbors.clone(),
                waitlist.clone(),
                cancel.clone(),
            ),
            dialer: dialer.clone(),
            neighbors: neighbors.clone(),
            waitlist: waitlist.clone(),
            cancel: cancel.clone(),
            // Anti-entropy, for what the flood loses. Raised by the receive loop
            // when the swarm reports it outran this peer, and swept on a slow
            // cadence regardless, since most losses announce themselves to nobody.
            prompt: Prompt::default(),
        };
        let (outgoing, queue) = mpsc::unbounded_channel();
        task::spawn(send_loop(sender.clone(), queue));
        task::spawn(Reconciler::new(wiring.clone()).run());
        task::spawn(recv_loop(wiring, receiver, presence.clone(), tx));
        let session = Self {
            broadcaster: Broadcaster {
                local_id,
                sender,
                outgoing,
                dialer,
                neighbors,
                waitlist,
            },
            shutdown,
            cancel,
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
    /// [`Engine::take_outbox`](stark_engine::Engine::take_outbox)) to the swarm.
    ///
    /// Returns once the action is mirrored and queued; the session's one send task
    /// puts it on the wire. That task is what makes the wire order the order things
    /// were committed in — a caller spawning a send per dispatch raced two of them
    /// onto the same sender, and every inversion cost a timeline resync on every
    /// receiver.
    pub fn broadcast(&self, action: Action) -> Result<()> {
        self.broadcaster.broadcast(action)
    }

    /// Register content so joiners can be served and peers can fetch it — a brush
    /// image alongside
    /// [`Engine::import_brush`](stark_engine::Engine::import_brush), a canvas ground
    /// alongside [`Engine::import_surface`](stark_engine::Engine::import_surface).
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
    ///
    /// The stop signal goes first and the stack second: a resolver mid-backoff has
    /// to be told the session is over, not discover it by failing to dial.
    pub async fn shutdown(self) {
        self.cancel.stop();
        self.shutdown.run().await;
    }
}

/// A detached publishing handle onto a [`CollabSession`]: broadcast actions and
/// register assets without holding the session itself. All clones share the
/// same gossip topic and mirror.
#[derive(Clone)]
pub struct Broadcaster {
    local_id: EndpointId,
    /// Presence goes straight out: it is best-effort by design and on its own
    /// cadence, so it must not queue behind a long stroke.
    sender: GossipSender,
    /// Committed actions go through one queue drained by one task, which is what
    /// makes their wire order the order they were committed in.
    outgoing: mpsc::UnboundedSender<Bytes>,
    dialer: Dialer,
    neighbors: Arc<Mutex<HashSet<EndpointId>>>,
    waitlist: Arc<Waitlist>,
}

impl Broadcaster {
    /// See [`CollabSession::broadcast`].
    pub fn broadcast(&self, action: Action) -> Result<()> {
        let need = stark_engine::action_content(&action);
        let bytes = self.encode(WireRef::Action(&action), need)?;
        // Mirrored after encoding and before queueing, which is what lets the action
        // be moved rather than duplicated — and mirrored whether or not it ever goes
        // out, since that is what lets `reconcile` hand it to a peer that missed it.
        self.waitlist.published(action);
        // Queued, not sent. The caller returns immediately and the session's one
        // send task puts it on the wire; a closed queue means the session is over,
        // which is not this caller's problem.
        let _ = self.outgoing.send(bytes);
        Ok(())
    }

    /// Publish this client's presence (§17.4).
    ///
    /// Deliberately *not* mirrored: presence is not part of the document, so it is
    /// never served to a joiner and never reaches a file. And deliberately
    /// best-effort — a frame that cannot be sent is dropped rather than retried,
    /// because the next one supersedes it anyway.
    pub async fn publish(&self, frame: PeerFrame) -> Result<()> {
        let need = referenced_presence_asset(&frame);
        let bytes = self.encode(WireRef::Presence(&frame), need)?;
        Ok(self.sender.broadcast(bytes).await?)
    }

    /// Stamp and encode one payload, attaching the transfer hash for whatever
    /// content it references.
    fn encode(&self, wire: WireRef<'_>, need: Option<AssetNeed>) -> Result<Bytes> {
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
        let stamped = StampedRef {
            origin: self.local_id,
            asset,
            wire,
        };
        Ok(postcard::to_allocvec(&stamped)?.into())
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

/// The handles a session's background work runs on — what the receive loop and the
/// reconciler both need, assembled once by `finish` instead of threaded to each of
/// them a field at a time.
#[derive(Clone)]
pub(crate) struct Wiring {
    pub dialer: Dialer,
    pub neighbors: Arc<Mutex<HashSet<EndpointId>>>,
    pub waitlist: Arc<Waitlist>,
    pub resolver: Resolver<Dialer>,
    pub cancel: Cancel,
    pub prompt: Prompt,
}

/// Run the fallible tail of session setup, closing the stack if it does not
/// finish.
///
/// Every step that can fail — dialling the ticket's peer, subscribing, fetching
/// and decoding the snapshot, minting the ticket address — happens *after* the
/// endpoint exists, and dropping [`Bound`] closes none of it: the endpoint keeps
/// its relay connection and the gossip, blob and router actors keep running. The
/// expected failures are the ones that repeat (a link from another build, a host
/// that has gone), so without this a user retrying accumulates a stack per
/// attempt — in a browser tab with a hard ceiling.
async fn closing_on_error<T>(
    shutdown: Shutdown,
    setup: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    match setup.await {
        Ok(ready) => Ok(ready),
        Err(e) => {
            shutdown.run().await;
            Err(e)
        }
    }
}

/// The session's one send task: committed actions reach the wire in the order they
/// were queued, which is the order they were committed in.
///
/// A failed send is not retried here. The action is already in this peer's mirror,
/// so the next member to sweep collects it ([`reconcile`](crate::reconcile)) — and
/// retrying in place would hold every action behind it up for a peer that is about
/// to ask for it anyway.
async fn send_loop(sender: GossipSender, mut queue: mpsc::UnboundedReceiver<Bytes>) {
    while let Some(bytes) = queue.recv().await {
        if let Err(e) = sender.broadcast(bytes).await {
            tracing::warn!("broadcast failed: {e}; a peer's sweep will recover it");
        }
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
    wiring: Wiring,
    mut gossip: GossipReceiver,
    presence: Arc<PresenceQuota>,
    tx: mpsc::UnboundedSender<RemoteEvent>,
) {
    let Wiring {
        dialer,
        neighbors,
        waitlist,
        resolver,
        prompt,
        ..
    } = wiring;
    while let Some(event) = gossip.next().await {
        let message = match event {
            Ok(GossipEvent::Received(message)) => message,
            Ok(GossipEvent::Lagged) => {
                // The swarm outran this peer and skipped the rest. Which actions
                // went is not knowable from here, so this is one of the two things
                // reconciliation exists for: compare logs with a neighbour shortly,
                // once what is still in flight has landed (§12.5). A lagged
                // *presence* stream needs no recovery at all — the author re-sends
                // its whole gesture on the next resync frame (§17.5).
                tracing::warn!("gossip lagged; reconciling to recover what was missed");
                prompt.raise();
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
                    task::spawn(resolver.clone().resolve(need, hash, origin, from));
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

        // Two identities travel with every action and nothing upstream ties them
        // together: `origin` picks who to fetch its content from, while
        // `id.actor` owns its undo scope (§12.3) and half the total order key.
        // Gossip reports only the delivering neighbour, so `origin` is
        // self-declared — the same trust the payload already carries (§12.5) —
        // which makes this a consistency check and not authentication.
        //
        // Dropped rather than accepted, because an action whose author is wrong
        // is one whose undo scope is wrong: applying it puts something in the log
        // that the peer who appears to have written it cannot take back.
        if actor_from_endpoint_id(origin) != action.id.actor {
            tracing::warn!(
                origin = %origin.fmt_short(),
                actor = ?action.id.actor,
                "dropping an action whose author does not match its sender"
            );
            continue;
        }

        // Whatever the action references has to reach the engine first, so the
        // engine can apply it faithfully. The action waits for it — parked, not
        // awaited here, so nothing else in the session waits with it. The origin
        // authored the action and so definitely holds the content; the neighbour
        // that forwarded it may not.
        match Admission::of(&action, asset_hash, &waitlist) {
            Admission::Ready => waitlist.accept(action),
            Admission::Waiting => continue,
            Admission::Fetching { need, hash } => {
                task::spawn(resolver.clone().resolve(need, hash, origin, from));
                continue;
            }
        }
        if !waitlist.is_live() {
            return;
        }
    }
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

/// What has to happen before an arriving action can be applied.
///
/// One place decides, because two ways in must not decide differently. An action
/// off the gossip flood and one recovered by
/// [`reconcile`](crate::reconcile) need the same thing: a `SetSurface` applied
/// before its ground has landed bakes a smooth deposit into stored tiles that no
/// later arrival un-bakes (§6.4), and it does not matter which door it came
/// through.
pub(crate) enum Admission {
    /// Nothing is missing — apply it. The caller keeps the action, the way
    /// [`Admit`] leaves it with them.
    Ready,
    /// Parked behind a fetch already in flight; it is released with that one.
    Waiting,
    /// Parked, and the caller must start the fetch.
    Fetching { need: AssetNeed, hash: Hash },
}

impl Admission {
    /// Decide, parking the action if it has to wait.
    ///
    /// [`Ready`](Admission::Ready) covers both an action that references nothing
    /// and one that arrived without a transfer hash: there is nothing to fetch, so
    /// parking would be parking forever, and the kind's fallback is the best
    /// available.
    pub fn of(action: &Action, hash: Option<Hash>, waitlist: &Waitlist) -> Self {
        let Some(need) = stark_engine::action_content(action) else {
            return Self::Ready;
        };
        let Some(hash) = hash_or_warn(need, hash) else {
            return Self::Ready;
        };
        match waitlist.claim(need, action) {
            Admit::Ready => Self::Ready,
            Admit::Waiting => Self::Waiting,
            Admit::Fetch => Self::Fetching { need, hash },
        }
    }
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

impl std::fmt::Debug for CollabSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollabSession")
            .field("topic", &self.topic)
            .field("endpoint", &self.broadcaster.local_id)
            .finish_non_exhaustive()
    }
}
