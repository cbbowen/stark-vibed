//! What an arriving action waits for, and what waits behind it (§12.4).
//!
//! An action naming content this peer does not hold cannot be applied
//! faithfully: a stroke would stamp with the round tip instead of its shape
//! (§6.6), and a `SetSurface` would move the document onto a flat stand-in,
//! after which every stroke bakes a smooth deposit into stored tiles that no
//! later arrival un-bakes (§6.4). So the action waits for the content.
//!
//! It waits *here*, beside the gossip receive loop, rather than inside it. What
//! an action needs ordering against is the content it names and nothing else:
//! [`merge_remote`](stark_engine::Engine::merge_remote) is idempotent by id and
//! order-insensitive, and an action landing behind newer ones makes the timeline
//! resync, which replays them against what it changed (§12.6). Waiting inside
//! the loop — which is what this replaces — stalled every other peer's actions,
//! every presence frame and the neighbor bookkeeping behind one unreachable
//! blob.
//!
//! Parking is also what de-duplicates the fetch: the same content is routinely
//! named twice, once by a live gesture's head frame and again by the commit it
//! becomes, and only the first of them starts a resolver.
//!
//! Two invariants live in here rather than at the call sites, which is the whole
//! reason the type exists:
//!
//! - **Nothing parks behind a resolver that has already finished.** Claiming and
//!   delivering both take the two locks, so the "is it here yet?" test and the
//!   park that follows it cannot straddle a delivery. An action that lost that
//!   race would wait for the life of the session.
//! - **Content reaches the engine before the actions that named it.** Releasing
//!   is not something a caller does after delivering; it is what delivering
//!   *is*.
//!
//! The mirror handle lives here for a related reason: this is the only place
//! that locks both mutexes, so the order it locks them in — parked, then mirror
//! — is a fact about one file instead of a convention every call site could
//! break.
//!
//! What this file does *not* decide is how hard to try. Whether an action waits is
//! here; how long the fetch it waits on keeps going, and who it asks, is
//! [`content`](crate::content) — the two halves of one question, split where the
//! brush and the ground stop behaving alike.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use iroh_blobs::Hash;
use stark_model::AssetId;
use stark_model::document::{Action, ActionId};
use tokio::sync::mpsc;

use crate::mirror::Mirror;
use crate::session::{AssetNeed, RemoteEvent};

/// What the receive loop should do with an action that references content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Admit {
    /// The content is already here — apply the action now.
    Ready,
    /// Parked behind a resolver already in flight; it is released with it.
    Waiting,
    /// Parked, and this caller must start the resolver.
    Fetch,
}

pub(crate) struct Waitlist {
    mirror: Arc<Mutex<Mirror>>,
    /// Needs with a resolver in flight, and the actions parked behind each. An
    /// empty vec is a live fetch nothing is waiting on — a presence head's.
    parked: Mutex<HashMap<AssetNeed, Vec<Action>>>,
    events: mpsc::UnboundedSender<RemoteEvent>,
    /// What the frontend promised it could produce without the network
    /// ([`NetOptions::resolvable`](crate::NetOptions::resolvable)). Fixed for the
    /// life of the session: it is a property of the build, not of what has been
    /// loaded so far.
    resolvable: HashSet<AssetId>,
}

impl Waitlist {
    pub fn new(
        mirror: Arc<Mutex<Mirror>>,
        events: mpsc::UnboundedSender<RemoteEvent>,
        resolvable: &[AssetId],
    ) -> Self {
        Self {
            mirror,
            parked: Mutex::new(HashMap::new()),
            events,
            resolvable: resolvable.iter().copied().collect(),
        }
    }

    /// Whether the frontend claims it can produce this content itself — the test
    /// that decides whether a resolver asks it before it dials.
    pub fn is_local(&self, need: AssetNeed) -> bool {
        self.resolvable.contains(&need.content())
    }

    /// Whether the content has arrived since — how a resolver finds out that the
    /// frontend made good on [`RemoteEvent::ResolveLocally`].
    pub fn holds(&self, need: AssetNeed) -> bool {
        self.mirror.lock().expect("mirror poisoned").has(need)
    }

    /// Ask the frontend for content it said it could produce.
    pub fn ask_locally(&self, need: AssetNeed) {
        let _ = self.events.send(RemoteEvent::ResolveLocally { need });
    }

    /// Whether the engine is still listening. A resolver that may retry
    /// indefinitely polls this so it cannot outlive the session.
    pub fn is_live(&self) -> bool {
        !self.events.is_closed()
    }

    /// Decide what to do with `action`, which references `need`. Borrowed, and
    /// cloned only into the park: the common answer is [`Admit::Ready`], where
    /// the caller keeps the action it already had.
    pub fn claim(&self, need: AssetNeed, action: &Action) -> Admit {
        let mut parked = self.parked.lock().expect("waitlist poisoned");
        if self.mirror.lock().expect("mirror poisoned").has(need) {
            return Admit::Ready;
        }
        match parked.entry(need) {
            Entry::Occupied(mut slot) => {
                slot.get_mut().push(action.clone());
                Admit::Waiting
            }
            Entry::Vacant(slot) => {
                slot.insert(vec![action.clone()]);
                Admit::Fetch
            }
        }
    }

    /// The same decision for content nothing is waiting on — a live gesture's
    /// brush image, which must never block on a fetch (§17.5): the preview
    /// degrades to the round tip and upgrades when the bytes land. `true` when
    /// this caller must start the resolver.
    pub fn claim_detached(&self, need: AssetNeed) -> bool {
        let mut parked = self.parked.lock().expect("waitlist poisoned");
        if self.mirror.lock().expect("mirror poisoned").has(need) {
            return false;
        }
        match parked.entry(need) {
            Entry::Occupied(_) => false,
            Entry::Vacant(slot) => {
                slot.insert(Vec::new());
                true
            }
        }
    }

    /// Content fetched off a peer: hand it to the engine, then release
    /// everything parked behind it — in that order, which is the point.
    pub fn resolved(&self, need: AssetNeed, bytes: Bytes, hash: Hash) {
        let parked = self.take_parked(need, Some((bytes.clone(), hash)));
        if self.events.send(RemoteEvent::Asset { need, bytes }).is_ok() {
            self.release(parked);
        }
    }

    /// Content this client imported locally. The engine already holds it, so
    /// only the parked actions are released — but they *are* released, because a
    /// local import satisfies a remote action's need exactly as a fetch does.
    pub fn imported(&self, need: AssetNeed, bytes: Bytes, hash: Hash) {
        let parked = self.take_parked(need, Some((bytes, hash)));
        self.release(parked);
    }

    /// The hash content transfers under, for a broadcast to attach.
    pub fn transfer_hash(&self, id: AssetId) -> Option<Hash> {
        self.mirror
            .lock()
            .expect("mirror poisoned")
            .transfer_hash(id)
    }

    /// Nothing could be fetched: release the parked actions to whatever fallback
    /// their kind has. Only ever reached for a brush — a ground is not given up
    /// on (see `resolve_asset`).
    pub fn abandoned(&self, need: AssetNeed) {
        let parked = self.take_parked(need, None);
        self.release(parked);
    }

    /// A remote action arriving with everything it needs: mirror it and surface
    /// it to the engine, unless it has been seen before.
    pub fn accept(&self, action: Action) {
        self.release(vec![action]);
    }

    /// Which of `theirs` this peer does not hold — what a reconciliation sweep is
    /// asking (see [`reconcile`](crate::reconcile)). Routed through here rather
    /// than off a second mirror handle, so this file stays the only one holding
    /// one and the lock order stays a fact about one file.
    pub fn missing_from(&self, theirs: &[ActionId]) -> Vec<ActionId> {
        self.mirror
            .lock()
            .expect("mirror poisoned")
            .missing_from(theirs)
    }

    /// A locally-committed action on its way out to the swarm: mirror it so this
    /// peer can serve it to a joiner, and surface nothing — the engine authored
    /// it and has applied it already.
    pub fn published(&self, action: Action) {
        self.mirror.lock().expect("mirror poisoned").insert(action);
    }

    /// Record content, if any, and detach everything parked behind `need`.
    fn take_parked(&self, need: AssetNeed, content: Option<(Bytes, Hash)>) -> Vec<Action> {
        let mut parked = self.parked.lock().expect("waitlist poisoned");
        if let Some((bytes, hash)) = content {
            self.mirror
                .lock()
                .expect("mirror poisoned")
                .insert_content(need, bytes, hash);
        }
        parked.remove(&need).unwrap_or_default()
    }

    /// Mirror actions and surface them in order, skipping ones already seen.
    fn release(&self, actions: Vec<Action>) {
        for action in actions {
            let fresh = self
                .mirror
                .lock()
                .expect("mirror poisoned")
                .insert_cloned(&action);
            if fresh && self.events.send(RemoteEvent::Action(action)).is_err() {
                return;
            }
        }
    }
}

/// The parking mechanism, without a network under it.
///
/// These are here rather than in an integration test because the properties
/// worth pinning are about *this* type — that a claim and a delivery cannot
/// interleave wrongly, that the asset precedes what waited on it — and reaching
/// them through two iroh endpoints would test the swarm instead, more slowly and
/// less exactly.
#[cfg(test)]
mod tests {
    use stark_engine::DocumentFile;
    use stark_model::AssetId;
    use stark_model::document::{ActionId, ActionKind, ActorId};

    use super::*;

    fn setup() -> (Waitlist, mpsc::UnboundedReceiver<RemoteEvent>) {
        with_resolvable(&[])
    }

    /// A waitlist whose frontend claims it can produce `resolvable` itself.
    fn with_resolvable(resolvable: &[AssetId]) -> (Waitlist, mpsc::UnboundedReceiver<RemoteEvent>) {
        let mirror = Arc::new(Mutex::new(Mirror::from_file(
            &DocumentFile::new(Vec::new()),
        )));
        let (tx, rx) = mpsc::unbounded_channel();
        (Waitlist::new(mirror, tx, resolvable), rx)
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

    fn need(tag: u8) -> AssetNeed {
        AssetNeed::Brush(AssetId([tag; 32]))
    }

    fn content(tag: u8) -> (Bytes, Hash) {
        let bytes = Bytes::from(vec![tag; 16]);
        let hash = Hash::new(&bytes);
        (bytes, hash)
    }

    fn drain(rx: &mut mpsc::UnboundedReceiver<RemoteEvent>) -> Vec<RemoteEvent> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            out.push(event);
        }
        out
    }

    #[test]
    fn an_action_waits_for_its_content_and_arrives_behind_it() {
        let (waitlist, mut rx) = setup();
        assert!(matches!(waitlist.claim(need(1), &action(1)), Admit::Fetch));
        assert!(drain(&mut rx).is_empty(), "nothing surfaces while parked");

        let (bytes, hash) = content(1);
        waitlist.resolved(need(1), bytes, hash);

        // The order is the whole point: applying the action first is what bakes a
        // flat deposit for a ground (§6.4).
        let events = drain(&mut rx);
        assert!(
            matches!(events[0], RemoteEvent::Asset { .. }),
            "asset first"
        );
        assert!(matches!(&events[1], RemoteEvent::Action(a) if a.id == action(1).id));
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn content_already_held_does_not_park_at_all() {
        let (waitlist, _rx) = setup();
        let (bytes, hash) = content(1);
        waitlist.imported(need(1), bytes, hash);
        assert!(matches!(waitlist.claim(need(1), &action(1)), Admit::Ready));
    }

    /// A live gesture's head and the commit it becomes name the same shape, and
    /// only one of them should be dialing for it.
    #[test]
    fn a_second_claimant_rides_the_first_resolver() {
        let (waitlist, mut rx) = setup();
        assert!(
            waitlist.claim_detached(need(1)),
            "the head starts the fetch"
        );
        assert!(
            !waitlist.claim_detached(need(1)),
            "a repeated head starts nothing further"
        );
        assert!(matches!(
            waitlist.claim(need(1), &action(1)),
            Admit::Waiting
        ));

        let (bytes, hash) = content(1);
        waitlist.resolved(need(1), bytes, hash);
        assert_eq!(drain(&mut rx).len(), 2, "one asset, one released action");
    }

    /// A local import satisfies a remote action's need exactly as a fetch does —
    /// and if it did not release, the action would wait for a fetch of bytes
    /// already in hand.
    #[test]
    fn a_local_import_releases_what_was_waiting_on_it() {
        let (waitlist, mut rx) = setup();
        assert!(matches!(waitlist.claim(need(1), &action(1)), Admit::Fetch));

        let (bytes, hash) = content(1);
        waitlist.imported(need(1), bytes, hash);

        // No `Asset` event: the engine did the importing, so it already holds them.
        let events = drain(&mut rx);
        assert!(matches!(&events[0], RemoteEvent::Action(a) if a.id == action(1).id));
        assert_eq!(events.len(), 1);
    }

    /// Giving up must still let the action through — a brush that never arrives
    /// costs the round tip, whereas an action parked forever costs the stroke.
    #[test]
    fn abandoning_a_fetch_releases_to_the_fallback() {
        let (waitlist, mut rx) = setup();
        waitlist.claim(need(1), &action(1));
        waitlist.abandoned(need(1));

        let events = drain(&mut rx);
        assert!(matches!(&events[0], RemoteEvent::Action(a) if a.id == action(1).id));
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn parking_one_need_does_not_hold_up_another() {
        let (waitlist, mut rx) = setup();
        waitlist.claim(need(1), &action(1));
        waitlist.claim(need(2), &action(2));

        let (bytes, hash) = content(2);
        waitlist.resolved(need(2), bytes, hash);

        let ids: Vec<_> = drain(&mut rx)
            .iter()
            .filter_map(|e| match e {
                RemoteEvent::Action(a) => Some(a.id),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![action(2).id], "only the resolved need released");
    }

    /// Gossip delivers the same action more than once, and a duplicate that
    /// parked would otherwise surface a second time when its content landed.
    #[test]
    fn a_duplicate_action_surfaces_once() {
        let (waitlist, mut rx) = setup();
        waitlist.claim(need(1), &action(1));
        waitlist.claim(need(1), &action(1));

        let (bytes, hash) = content(1);
        waitlist.resolved(need(1), bytes, hash);
        assert_eq!(drain(&mut rx).len(), 2, "one asset, one action");

        waitlist.accept(action(1));
        assert!(drain(&mut rx).is_empty(), "and never again");
    }

    /// The promise is per-id, not a blanket setting: content the frontend did not
    /// claim still goes straight to the network, and content it did claim is
    /// asked for first.
    #[test]
    fn only_promised_content_is_asked_of_the_frontend() {
        let (waitlist, _rx) = with_resolvable(&[need(1).content()]);
        assert!(waitlist.is_local(need(1)));
        assert!(!waitlist.is_local(need(2)));
    }

    /// A local commit is mirrored so joiners get it, but must not come back at
    /// the engine that authored it.
    #[test]
    fn a_published_action_is_mirrored_but_not_surfaced() {
        let (waitlist, mut rx) = setup();
        waitlist.published(action(1));
        assert!(drain(&mut rx).is_empty());

        // Mirrored, so the same action arriving from a peer is not fresh.
        waitlist.accept(action(1));
        assert!(drain(&mut rx).is_empty());
    }
}
