//! Getting the content an action names: how hard to try, and who to ask (§12.4).
//!
//! This is the other half of [`waitlist`](crate::waitlist). The waitlist decides
//! *whether* an action has to wait and holds what is waiting; this decides how
//! long to spend making the wait end, and it is the half where the two kinds of
//! content stop behaving alike.
//!
//! The asymmetry is §6.4's. An unresolved **brush** degrades to the round tip and
//! the stroke is still visibly a stroke, so after [`BRUSH_ATTEMPTS`] rounds the
//! fetch gives up and lets the action through. An unresolved **ground** has no
//! acceptable fallback: applying the `SetSurface` against `Flat` bakes a smooth
//! deposit into stored tiles that no later arrival un-bakes. So a ground is never
//! given up on — the action simply waits, and nothing else waits with it. Strokes
//! that merged ahead of it are replayed against the real ground when it lands,
//! because an action arriving out of order is exactly what makes the timeline
//! resync (§12.6).

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use iroh::EndpointId;
use iroh_blobs::Hash;

use crate::Result;
use crate::backend::Cancel;
use crate::session::AssetNeed;
use crate::waitlist::Waitlist;

/// Rounds spent fetching a *brush* image before giving up and letting the stroke
/// draw with the round tip. A ground is never given up on.
const BRUSH_ATTEMPTS: u32 = 5;

/// The first delay between fetch rounds — a source may still be fetching the blob
/// itself. It doubles up to the cap, so an unbounded retry settles into an
/// occasional poll rather than a busy one.
const ASSET_RETRY_DELAY: Duration = Duration::from_millis(300);
const ASSET_RETRY_MAX_DELAY: Duration = Duration::from_secs(30);

/// How long a resolver waits for the frontend to make good on
/// [`RemoteEvent::ResolveLocally`](crate::RemoteEvent::ResolveLocally) before
/// dialling a peer anyway.
///
/// Generous, because the cost of being early is a transfer this whole mechanism
/// exists to avoid, while the cost of being late is a wait on an action that is
/// parked regardless — and a read from the app's own bundle that takes longer
/// than this is one that was probably never going to arrive.
const LOCAL_GRACE: Duration = Duration::from_secs(3);

/// Where content comes from — and the seam the policy above is tested through.
///
/// Everything in this file is timing and giving up: how many rounds a brush gets,
/// that a ground gets unlimited ones, how long the frontend has to keep a promise,
/// how the backoff widens, which peers are asked and in what order. None of it is
/// reachable through two real endpoints, because a test cannot ask a live peer to
/// fail four times and then succeed, or to be gone for the first two rounds and
/// back for the third. That is why none of it used to be tested.
///
/// [`Dialer`](crate::backend::Dialer) is the production implementation; the tests
/// at the bottom supply peers that fail, vanish, or answer late.
pub(crate) trait ContentSource: Clone + 'static {
    /// Fetch one blob from `provider`, hash-verified, and return its bytes.
    fn fetch_blob(
        &self,
        provider: EndpointId,
        hash: Hash,
    ) -> impl std::future::Future<Output = Result<Bytes>>;
}

/// Everything a fetch needs except what it is fetching. Cloned per resolver.
#[derive(Clone)]
pub(crate) struct Resolver<S> {
    source: S,
    /// This peer's gossip neighbors, live — the swarm a widening search reaches
    /// past the two peers that named the content. See [`Resolver::providers`].
    neighbors: Arc<Mutex<HashSet<EndpointId>>>,
    waitlist: Arc<Waitlist>,
    cancel: Cancel,
}

/// How a fetch ended, which is three things and not two: giving up releases the
/// parked actions to their fallback, and being shut down must not, because there
/// is no longer anyone to release them to and a ground's fallback is a wrong
/// picture (§6.4).
enum Outcome {
    Got(Bytes),
    GaveUp,
    Cancelled,
}

impl<S: ContentSource> Resolver<S> {
    pub fn new(
        source: S,
        neighbors: Arc<Mutex<HashSet<EndpointId>>>,
        waitlist: Arc<Waitlist>,
        cancel: Cancel,
    ) -> Self {
        Self {
            source,
            neighbors,
            waitlist,
            cancel,
        }
    }

    /// Fetch missing content, mirror it and record its transfer hash (so this peer
    /// can serve and announce it onward), surface it to the engine, and release
    /// every action parked behind it.
    ///
    /// `origin` authored the message and so definitely holds the content; `from`
    /// is the neighbour that delivered it, which may not.
    pub async fn resolve(self, need: AssetNeed, hash: Hash, origin: EndpointId, from: EndpointId) {
        // First refusal to the frontend, when it said it ships with this content: a
        // ground the app bundles is a read from its own files, not megabytes off a
        // peer (§12.4). Asking costs one event and a wait on an action that is parked
        // either way; being ignored costs `LOCAL_GRACE` and then behaves exactly as it
        // would have.
        if self.waitlist.is_local(need) {
            self.waitlist.ask_locally(need);
            if !self.cancel.sleep(LOCAL_GRACE).await {
                return;
            }
            if self.waitlist.holds(need) {
                // `add_content` recorded it and released whatever was parked; there is
                // nothing left for this resolver to do.
                tracing::debug!(?need, "resolved from the frontend's own bundle");
                return;
            }
            tracing::debug!(?need, "local resolution did not arrive; asking a peer");
        }

        let attempts = match need {
            AssetNeed::Brush(_) => Some(BRUSH_ATTEMPTS),
            AssetNeed::Ground(_) => None,
        };
        match self.fetch(hash, attempts, origin, from).await {
            // Recording the transfer hash with the bytes is what lets this peer
            // announce the content onward on its own actions.
            Outcome::Got(bytes) => self.waitlist.resolved(need, bytes, hash),
            // Only a brush gives up, so releasing here is releasing to the round tip.
            Outcome::GaveUp => {
                tracing::warn!("{need:?} unavailable; the stroke will draw with the round tip");
                self.waitlist.abandoned(need);
            }
            // Deliberately nothing: the session is over, and for a ground the
            // "fallback" would be the flat stand-in this all exists to avoid.
            Outcome::Cancelled => tracing::debug!(?need, "resolver stopped with the session"),
        }
    }

    /// Try each source in turn on a widening backoff (a source may still be
    /// fetching the content itself). `attempts` caps the rounds; `None` retries
    /// until the content arrives or the session ends. The transfer is
    /// hash-verified by blobs.
    async fn fetch(
        &self,
        hash: Hash,
        attempts: Option<u32>,
        origin: EndpointId,
        from: EndpointId,
    ) -> Outcome {
        let mut delay = ASSET_RETRY_DELAY;
        let mut round = 0u32;
        loop {
            for provider in self.providers(round, origin, from) {
                if !self.wanted() {
                    return Outcome::Cancelled;
                }
                match self.source.fetch_blob(provider, hash).await {
                    Ok(bytes) => return Outcome::Got(bytes),
                    Err(e) => tracing::debug!("asset fetch round {round} failed: {e}"),
                }
            }
            round = round.saturating_add(1);
            if attempts.is_some_and(|max| round >= max) {
                return Outcome::GaveUp;
            }
            if !self.cancel.sleep(delay).await || !self.wanted() {
                return Outcome::Cancelled;
            }
            delay = (delay * 2).min(ASSET_RETRY_MAX_DELAY);
        }
    }

    /// Who to ask this round.
    ///
    /// The author first — it referenced the content, so it holds it — then whoever
    /// delivered the message, which may have it cached and is known to be
    /// reachable. From the second round on, the rest of the swarm joins them:
    /// every member seeded its blob store from the snapshot it joined on
    /// (`Mirror::seed_blobs`), so content whose author has since left is still held
    /// by everyone who was there when it arrived.
    ///
    /// That matters most for a ground, because a ground is never given up on.
    /// Asking only the two peers that named it meant a `SetSurface` whose author
    /// and forwarder had both left waited for the life of the session, dialling two
    /// dead endpoints on a widening backoff, while the bytes sat on every other
    /// member.
    ///
    /// Re-read per round rather than captured once, so a member who joins during a
    /// long retry becomes a candidate.
    fn providers(&self, round: u32, origin: EndpointId, from: EndpointId) -> Vec<EndpointId> {
        let mut ids = vec![origin];
        if from != origin {
            ids.push(from);
        }
        if round > 0 {
            let neighbors = self.neighbors.lock().expect("neighbors poisoned");
            let swarm: Vec<EndpointId> = neighbors
                .iter()
                .copied()
                .filter(|id| !ids.contains(id))
                .collect();
            ids.extend(swarm);
        }
        ids
    }

    /// Whether there is still anyone to resolve for. Two different facts, either
    /// of which ends the work: the session may have been shut down, or the engine
    /// may simply have stopped listening.
    fn wanted(&self) -> bool {
        !self.cancel.stopped() && self.waitlist.is_live()
    }
}

/// The policy this file is, exercised against peers that behave the way real ones
/// do only rarely and never on demand.
///
/// Time is paused throughout: the backoff widens to half a minute, so a test that
/// waited for it would be measuring `tokio::time` rather than the retry.
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use iroh::SecretKey;
    use stark_engine::DocumentFile;
    use stark_engine::document::{Action, ActionId, ActionKind, ActorId};
    use stark_model::AssetId;
    use tokio::sync::mpsc;

    use super::*;
    use crate::mirror::Mirror;
    use crate::session::RemoteEvent;
    use crate::waitlist::Admit;

    /// A peer whose blob store holds `content` — after `fails_first` refusals.
    #[derive(Clone, Default)]
    struct Peer {
        content: Option<Bytes>,
        fails_first: usize,
    }

    /// A source with a script. Counts what it was asked, so a test can assert the
    /// *order* peers were tried in and not just the outcome.
    #[derive(Clone, Default)]
    struct Fake {
        peers: Arc<Mutex<HashMap<EndpointId, Peer>>>,
        asked: Arc<Mutex<Vec<EndpointId>>>,
        rounds: Arc<AtomicUsize>,
    }

    impl Fake {
        fn holding(&self, id: EndpointId, bytes: &Bytes, fails_first: usize) -> &Self {
            self.peers.lock().unwrap().insert(
                id,
                Peer {
                    content: Some(bytes.clone()),
                    fails_first,
                },
            );
            self
        }

        fn asked(&self) -> Vec<EndpointId> {
            self.asked.lock().unwrap().clone()
        }
    }

    impl ContentSource for Fake {
        async fn fetch_blob(&self, provider: EndpointId, _hash: Hash) -> Result<Bytes> {
            self.asked.lock().unwrap().push(provider);
            self.rounds.fetch_add(1, Ordering::Relaxed);
            let mut peers = self.peers.lock().unwrap();
            let Some(peer) = peers.get_mut(&provider) else {
                return Err(crate::NetError::Other("no such peer".into()));
            };
            if peer.fails_first > 0 {
                peer.fails_first -= 1;
                return Err(crate::NetError::Other("not yet".into()));
            }
            peer.content
                .clone()
                .ok_or_else(|| crate::NetError::Other("nothing here".into()))
        }
    }

    fn endpoint(tag: u8) -> EndpointId {
        SecretKey::from_bytes(&[tag; 32]).public()
    }

    fn action(lamport: u64) -> Action {
        Action {
            id: ActionId {
                lamport,
                actor: ActorId(1),
            },
            kind: ActionKind::SetBackground([0.0; 3]),
        }
    }

    struct Harness {
        resolver: Resolver<Fake>,
        fake: Fake,
        neighbors: Arc<Mutex<HashSet<EndpointId>>>,
        waitlist: Arc<Waitlist>,
        cancel: Cancel,
        events: mpsc::UnboundedReceiver<RemoteEvent>,
    }

    fn harness(resolvable: &[AssetId]) -> Harness {
        let mirror = Arc::new(Mutex::new(Mirror::from_file(
            &DocumentFile::new(Vec::new()),
        )));
        let (tx, events) = mpsc::unbounded_channel();
        let waitlist = Arc::new(Waitlist::new(mirror, tx, resolvable));
        let neighbors = Arc::new(Mutex::new(HashSet::new()));
        let fake = Fake::default();
        let cancel = Cancel::default();
        Harness {
            resolver: Resolver::new(
                fake.clone(),
                neighbors.clone(),
                waitlist.clone(),
                cancel.clone(),
            ),
            fake,
            neighbors,
            waitlist,
            cancel,
            events,
        }
    }

    fn drain(rx: &mut mpsc::UnboundedReceiver<RemoteEvent>) -> Vec<RemoteEvent> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            out.push(event);
        }
        out
    }

    fn brush() -> AssetNeed {
        AssetNeed::Brush(AssetId([1; 32]))
    }

    fn ground() -> AssetNeed {
        AssetNeed::Ground(AssetId([2; 32]))
    }

    fn payload() -> (Bytes, Hash) {
        let bytes = Bytes::from_static(b"the content");
        let hash = Hash::new(&bytes);
        (bytes, hash)
    }

    /// A brush that never arrives costs the round tip; an action parked forever
    /// costs the stroke. So the giving up has to actually happen, and it has to
    /// release.
    #[tokio::test(start_paused = true)]
    async fn a_brush_gives_up_after_its_rounds_and_lets_the_stroke_through() {
        let mut h = harness(&[]);
        let (_, hash) = payload();
        assert!(matches!(
            h.waitlist.claim(brush(), &action(1)),
            Admit::Fetch
        ));

        h.resolver
            .clone()
            .resolve(brush(), hash, endpoint(1), endpoint(1))
            .await;

        assert_eq!(
            h.fake.rounds.load(Ordering::Relaxed),
            BRUSH_ATTEMPTS as usize,
            "one ask per round, and then it stops"
        );
        let events = drain(&mut h.events);
        assert!(
            matches!(&events[0], RemoteEvent::Action(a) if a.id == action(1).id),
            "the stroke is released to its fallback"
        );
        assert_eq!(events.len(), 1, "and no asset, because none arrived");
    }

    /// The other half of the asymmetry (§6.4): a ground has no acceptable
    /// fallback, so it waits — for as long as it takes, past any number of rounds
    /// a brush would have given up after.
    #[tokio::test(start_paused = true)]
    async fn a_ground_waits_however_long_it_takes() {
        let mut h = harness(&[]);
        let (bytes, hash) = payload();
        assert!(matches!(
            h.waitlist.claim(ground(), &action(1)),
            Admit::Fetch
        ));
        // Silent for four times as many rounds as a brush is allowed.
        h.fake
            .holding(endpoint(1), &bytes, BRUSH_ATTEMPTS as usize * 4);

        h.resolver
            .clone()
            .resolve(ground(), hash, endpoint(1), endpoint(1))
            .await;

        let events = drain(&mut h.events);
        assert!(
            matches!(&events[0], RemoteEvent::Asset { need, .. } if *need == ground()),
            "the ground arrives before what waited on it"
        );
        assert!(matches!(&events[1], RemoteEvent::Action(a) if a.id == action(1).id));
    }

    /// The swarm holds what the author had: every member seeded its blob store
    /// from the snapshot it joined on. Without this a ground whose author and
    /// forwarder have both left waits out the session while the bytes sit on
    /// everyone else.
    #[tokio::test(start_paused = true)]
    async fn the_swarm_is_asked_after_the_two_peers_that_named_the_content() {
        let mut h = harness(&[]);
        let (bytes, hash) = payload();
        h.waitlist.claim(ground(), &action(1));
        // The author and the forwarder are gone. A third member was there.
        h.fake.holding(endpoint(3), &bytes, 0);
        h.neighbors.lock().unwrap().insert(endpoint(3));

        h.resolver
            .clone()
            .resolve(ground(), hash, endpoint(1), endpoint(2))
            .await;

        assert_eq!(
            h.fake.asked(),
            vec![
                // Round one is the two who named it, in that order...
                endpoint(1),
                endpoint(2),
                // ...and only then does the search widen.
                endpoint(1),
                endpoint(2),
                endpoint(3),
            ]
        );
        assert!(
            drain(&mut h.events)
                .iter()
                .any(|e| matches!(e, RemoteEvent::Asset { .. }))
        );
    }

    /// A ground retries without limit, so the stop signal is the only thing that
    /// ends it — and it must end it without releasing the parked action, whose
    /// fallback is the flat stand-in.
    #[tokio::test(start_paused = true)]
    async fn shutting_down_stops_a_ground_that_would_retry_forever() {
        let mut h = harness(&[]);
        let (_, hash) = payload();
        h.waitlist.claim(ground(), &action(1));

        let resolving = tokio::spawn(h.resolver.clone().resolve(
            ground(),
            hash,
            endpoint(1),
            endpoint(1),
        ));
        // Let it get well into the backoff, then end the session.
        tokio::time::sleep(Duration::from_secs(120)).await;
        assert!(!resolving.is_finished(), "still trying, as a ground does");
        h.cancel.stop();

        tokio::time::timeout(Duration::from_secs(1), resolving)
            .await
            .expect("the resolver stops with the session")
            .expect("and does not panic");
        assert!(
            drain(&mut h.events).is_empty(),
            "nothing released: there is no one to release it to, and no fallback worth taking"
        );
    }

    /// The promise is asked first and, when kept, costs no fetch at all — which is
    /// the entire point of it (§12.4).
    #[tokio::test(start_paused = true)]
    async fn a_kept_promise_costs_no_fetch() {
        let mut h = harness(&[ground().content()]);
        let (bytes, hash) = payload();
        h.waitlist.claim(ground(), &action(1));

        let waitlist = h.waitlist.clone();
        let supplied = bytes.clone();
        let resolving = tokio::spawn(h.resolver.clone().resolve(
            ground(),
            hash,
            endpoint(1),
            endpoint(1),
        ));
        // What the frontend does on `ResolveLocally`, inside the grace period.
        tokio::time::sleep(LOCAL_GRACE / 2).await;
        waitlist.imported(ground(), supplied, hash);
        resolving.await.expect("resolver");

        assert_eq!(h.fake.asked(), Vec::new(), "nobody was dialled");
        let events = drain(&mut h.events);
        assert!(matches!(&events[0], RemoteEvent::ResolveLocally { need } if *need == ground()));
        assert!(
            matches!(&events[1], RemoteEvent::Action(a) if a.id == action(1).id),
            "and the import released what was parked"
        );
    }

    /// Ignoring `ResolveLocally` has to be safe: the grace period passes and the
    /// resolver behaves exactly as it would have without the promise.
    #[tokio::test(start_paused = true)]
    async fn an_ignored_promise_falls_back_to_the_network() {
        let mut h = harness(&[brush().content()]);
        let (bytes, hash) = payload();
        h.waitlist.claim(brush(), &action(1));
        h.fake.holding(endpoint(1), &bytes, 0);

        h.resolver
            .clone()
            .resolve(brush(), hash, endpoint(1), endpoint(1))
            .await;

        assert_eq!(h.fake.asked(), vec![endpoint(1)], "asked, after the wait");
        let events = drain(&mut h.events);
        assert!(matches!(&events[0], RemoteEvent::ResolveLocally { .. }));
        assert!(matches!(&events[1], RemoteEvent::Asset { .. }));
        assert!(matches!(&events[2], RemoteEvent::Action(_)));
    }
}
