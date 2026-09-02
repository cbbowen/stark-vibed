//! A live shared-drawing session over iroh (§12.4).
//!
//! One [`CollabSession`] per shared document. The engine stays on the UI
//! thread; the session runs the network side on spawned tasks — the gossip
//! receive loop ([`inbox`]), the send queue ([`publish`]), the catch-up server,
//! the reconciler — and talks to the engine through two thin streams:
//!
//! ```text
//! engine.take_outbox() ──────► broadcaster.broadcast(action) ──► gossip
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

mod inbox;
mod join;
mod publish;

pub use publish::{Broadcaster, LinkKind, PeerLink};

use std::sync::Arc;

use iroh::{EndpointAddr, SecretKey};
use iroh_gossip::proto::TopicId;
use n0_future::task;
use stark_model::document::ActorId;
use stark_model::{AssetId, AssetNeed, DocumentFile};
use tokio::sync::mpsc;

use crate::Result;
use crate::backend::{self, Bound, Shutdown};
use crate::cancel::Cancel;
use crate::events::{Events, NetOptions, PresenceQuota, actor_from_endpoint_id};
use crate::mirror::{Mirror, Served, SharedMirror};
use crate::neighbors::Neighbors;
use crate::reconcile::{Prompt, Reconciler, Wiring};
use crate::ticket::SessionTicket;
use crate::waitlist::Waitlist;
use crate::wire::Request;

use inbox::recv_loop;
use join::{JOIN_TIMEOUT, closing_on_error, fetch_snapshot};
use publish::send_loop;

/// What [`CollabSession::join`] hands back.
///
/// A struct rather than a tuple because of `owed`: a fourth positional value
/// that must be acted on *before* the third is replayed is exactly the kind of
/// thing a caller destructures past.
pub struct Joined {
    pub session: CollabSession,
    pub events: Events,
    /// The snapshot to load via
    /// `Engine::join_collaboration` —
    /// **after** `owed` is settled.
    pub document: DocumentFile,
    /// Content `document`'s log names that the host left out, because this client
    /// said it could resolve it locally (`resolvable`).
    ///
    /// **Install every one of these into the engine before replaying the
    /// document.** Replay reads the substrate in force when each stroke was made
    /// (§6.4), so a `SetSubstrate` whose height map is not registered yet replays
    /// against the flat stand-in and bakes a smooth deposit that no later arrival
    /// un-bakes — the divergence content-addressing exists to prevent, arrived at
    /// by way of an optimization.
    ///
    /// If a promise cannot be kept, do nothing: the log still names the content,
    /// so the ordinary blob fetch pulls it off a peer exactly as it would have
    /// without the promise. Being wrong here costs a transfer, not a picture.
    pub owed: Vec<AssetNeed>,
}

/// A live shared session: its lifecycle — the spawned loops, the stop signal,
/// the teardown. Everything sent *into* it (actions, content, presence) and
/// everything asked *of* it (tickets, link kinds) goes through the
/// [`Broadcaster`] handle [`broadcaster`](Self::broadcaster) clones out; remote
/// edits arrive as [`RemoteEvent`](crate::RemoteEvent)s on the [`Events`]
/// stream handed out at setup.
pub struct CollabSession {
    /// Everything publishing needs, which is everything the session needs but
    /// two — so the session holds one rather than assembling one per call.
    broadcaster: Broadcaster,
    shutdown: Shutdown,
    /// Stops what the session spawned. Held here because this is what owns the
    /// session's lifetime; see [`Cancel`].
    cancel: Cancel,
}

impl CollabSession {
    /// Start sharing `doc` (the host side). `doc` should come from
    /// `Engine::document_file` *after*
    /// `Engine::start_collaboration`
    /// with [`actor_from_endpoint_id`] of this session's identity — generate a
    /// [`SecretKey`] first and pass it in `opts` so the actor id is known
    /// before binding.
    pub async fn host(doc: DocumentFile, opts: NetOptions) -> Result<(Self, Events)> {
        let served = Served::default();
        let bound = backend::bind(served.clone(), &opts).await?;
        let cancel = bound.cancel.clone();
        let shutdown = bound.shutdown.clone();
        closing_on_error(cancel, shutdown, Self::hosting(bound, served, doc, &opts)).await
    }

    /// Join an existing session from a ticket.
    ///
    /// [`NetOptions::resolvable`] is what this client can produce without asking
    /// anyone; the host leaves it out of the snapshot, and [`Joined::owed`] is the
    /// bill for that.
    pub async fn join(ticket: &SessionTicket, opts: NetOptions) -> Result<Joined> {
        let served = Served::default();
        let bound = backend::bind(served.clone(), &opts).await?;
        let cancel = bound.cancel.clone();
        let shutdown = bound.shutdown.clone();
        closing_on_error(
            cancel,
            shutdown,
            Self::joining(bound, served, ticket, &opts),
        )
        .await
    }

    /// Everything about hosting that can fail after the endpoint exists.
    async fn hosting(
        bound: Bound,
        served: Served,
        doc: DocumentFile,
        opts: &NetOptions,
    ) -> Result<(Self, Events)> {
        let mirror = SharedMirror::new(Mirror::from_file(&doc));
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
        // Teach the endpoint every member the link names before anything dials:
        // gossip below bootstraps from all of them by bare id, and an id is only
        // dialable once some address for it is known.
        for member in &ticket.members {
            bound.dialer.learn(member).await;
        }

        // Enter the live swarm *before* fetching the snapshot: everything
        // before the join is in the snapshot, everything after rides gossip,
        // and the overlap deduplicates by action id. Every member the link
        // names is a bootstrap candidate; the rest of the swarm arrives through
        // gossip's membership exchange. Best effort: joining still proceeds if
        // the swarm is slow, since the snapshot plus later traffic still
        // converges.
        let bootstrap = ticket.members.iter().map(|member| member.id).collect();
        let mut sub = bound.gossip.subscribe(ticket.topic, bootstrap).await?;
        if n0_future::time::timeout(JOIN_TIMEOUT, sub.joined())
            .await
            .is_err()
        {
            tracing::warn!("joined without meeting a peer yet; relying on catch-up");
        }

        let request = if opts.resolvable.is_empty() {
            Request::Snapshot
        } else {
            // The canonical order the server also imposes — what lets every
            // joiner running the same build hit the host's encode cache instead
            // of costing it a re-encode, however its catalog was enumerated.
            let mut have = opts.resolvable.clone();
            have.sort_unstable();
            have.dedup();
            Request::SnapshotWithout(have)
        };
        let snapshot = fetch_snapshot(&bound.dialer, &ticket.members, request).await?;
        // The untrusted door: these bytes are a peer's, and deflate's ratio means a
        // few kilobytes of them can name as many gigabytes as they like (§8, §12.4).
        let file = DocumentFile::from_untrusted_bytes(&snapshot)?;
        // What the log names but the bundle no longer carries — the bill for the
        // promise. Worked out by the file itself rather than here, because it is
        // the same question loading a document off disk asks, and one definition
        // of "what does this log need" is what stops the two drifting.
        let owed = file.unbundled_content();
        let mirror = SharedMirror::new(Mirror::from_file(&file));

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

    fn finish(
        bound: Bound,
        served: Served,
        topic: TopicId,
        sub: iroh_gossip::api::GossipTopic,
        mirror: SharedMirror,
        ticket_addr: EndpointAddr,
        resolvable: &[AssetId],
    ) -> Result<(Self, Events)> {
        let Bound {
            dialer,
            shutdown,
            cancel,
            ..
        } = bound;
        let local_id = dialer.local_id();
        let (sender, receiver) = sub.split();
        // Seed with the neighbors met before the receive loop takes over
        // (typically the bootstrap peer a joiner already awaited).
        let neighbors: Neighbors = receiver.neighbors().collect();
        for peer in neighbors.snapshot() {
            dialer.ensure_direct(peer);
        }
        // Every piece of content already known (the hosted document's, or the
        // joiner's snapshot's) enters the blob store so this peer can serve it, and
        // its transfer hash is recorded so this peer's own actions referencing it can
        // broadcast one. Brush images and canvas substrates alike — both are content an
        // action can be waiting on.
        mirror.lock().seed_blobs(|bytes| dialer.add_blob(bytes));
        // Only now is there a session to serve, and only now does the catch-up
        // protocol have anything to answer with: seeded, so what a snapshot names
        // can also be fetched piecemeal afterwards.
        served.publish(mirror.clone());
        let (tx, rx) = mpsc::unbounded_channel();
        let presence = Arc::new(PresenceQuota::default());
        let waitlist = Arc::new(Waitlist::new(mirror, tx.clone(), resolvable));
        // The receive loop is the only thing that dials afterwards (to fetch
        // brush assets and bootstrap WebRTC), so it takes the dialer with it.
        let wiring = Wiring {
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
        task::spawn(send_loop(cancel.clone(), sender.clone(), queue));
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
                topic,
                ticket_addr,
            },
            shutdown,
            cancel: cancel.clone(),
        };
        Ok((
            session,
            Events {
                rx,
                presence,
                cancel,
            },
        ))
    }

    /// The author id this session's identity maps to.
    pub fn actor_id(&self) -> ActorId {
        actor_from_endpoint_id(self.broadcaster.local_id)
    }

    /// A cheap, `Clone` handle for feeding the session — the whole sending and
    /// asking surface ([`Broadcaster::broadcast`], [`Broadcaster::add_content`],
    /// [`Broadcaster::ticket`], [`Broadcaster::links`]) lives on it, so a UI
    /// task never has to borrow the session across an `await`.
    pub fn broadcaster(&self) -> Broadcaster {
        self.broadcaster.clone()
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

/// Dropping the session ends it: the same signal-then-stack pair as
/// [`CollabSession::shutdown`], with the stack spawned because drop cannot
/// block. `shutdown` consumes `self`, so the graceful path runs the pair a
/// second time here — both halves are idempotent, and this is why they are.
impl Drop for CollabSession {
    fn drop(&mut self) {
        self.cancel.stop();
        // Spawning needs a runtime, and Drop can run without one — a session
        // moved to a plain thread, or a process on its way out. A Drop that
        // can panic aborts mid-unwind, so degrade instead: the stop signal
        // alone still ends every loop at its next cancel check, and only the
        // endpoint's explicit close is lost — which the exiting process does
        // anyway. wasm's spawn is spawn_local and needs no runtime handle.
        #[cfg(not(target_family = "wasm"))]
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let shutdown = self.shutdown.clone();
        task::spawn(async move { shutdown.run().await });
    }
}

impl std::fmt::Debug for CollabSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollabSession")
            .field("topic", &self.broadcaster.topic)
            .field("endpoint", &self.broadcaster.local_id)
            .finish_non_exhaustive()
    }
}

/// Teardown is the [`Cancel`], not a convention: a session that is merely
/// dropped must still end. The graceful path is what the integration tests
/// exercise; this is the other end of the session's life.
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// Dropping a `CollabSession` without `shutdown()` ends the session:
    /// `Events::recv` delivers its documented `None`, and the stack the session
    /// spawned dies — observed as the endpoint closing.
    #[tokio::test(flavor = "multi_thread")]
    async fn dropping_the_session_ends_it() {
        let (session, mut events) =
            CollabSession::host(DocumentFile::new(Vec::new()), NetOptions::local())
                .await
                .expect("host");
        // The one fact drop cannot report: whether the stack actually died.
        let dialer = session.broadcaster.dialer.clone();

        drop(session);

        // (a) The event stream drains and ends rather than hanging.
        let ended = tokio::time::timeout(Duration::from_secs(10), async {
            while events.recv().await.is_some() {}
        })
        .await;
        assert!(
            ended.is_ok(),
            "Events::recv must yield None once the session has ended"
        );
        assert!(events.try_recv().is_none());

        // (b) The teardown drop spawned closes the endpoint.
        let closed = tokio::time::timeout(Duration::from_secs(10), async {
            while !dialer.is_closed() {
                n0_future::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            closed.is_ok(),
            "dropping the session must close the endpoint"
        );
    }
}
