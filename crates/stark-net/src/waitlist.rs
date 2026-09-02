//! What an arriving action waits for, and what waits behind it (§12.4).
//!
//! An action naming content this peer does not hold cannot be applied
//! faithfully: a stroke would stamp with the round tip instead of its shape
//! (§6.6), and a `SetSubstrate` would move the document onto a flat stand-in,
//! after which every stroke bakes a smooth deposit into stored tiles that no
//! later arrival un-bakes (§6.4). So the action waits for the content.
//!
//! It waits *here*, beside the gossip receive loop, rather than inside it. What
//! an action needs ordering against is the content it names and nothing else:
//! `Engine::merge_remote` is idempotent by id and
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
//! brush and the substrate stop behaving alike.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use bytes::Bytes;
use iroh_blobs::Hash;
use stark_model::document::{Action, ActionId};
use stark_model::{AssetId, AssetNeed};
use tokio::sync::mpsc;

use crate::events::RemoteEvent;
use crate::mirror::SharedMirror;

/// What one claim decided — the parking primitive [`Waitlist::admit`] composes.
/// Arriving traffic goes through `admit`; this is also how the resolver's tests
/// park a fixture without a wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Admit {
    /// The content is already here — apply the action now.
    Ready,
    /// Parked behind a resolver already in flight; it is released with it.
    Waiting,
    /// Parked, and this caller must start the resolver.
    Fetch,
}

/// A fetch [`Waitlist::admit`] cannot start itself: only the caller has the
/// endpoints to ask (the author, the delivering neighbour) and the spawn hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MustFetch {
    pub need: AssetNeed,
    pub hash: Hash,
}

pub(crate) struct Waitlist {
    mirror: SharedMirror,
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
        mirror: SharedMirror,
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
        self.mirror.lock().has(need)
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

    /// The one door an arriving action goes through: decide what must happen
    /// before it can be applied, and do all of it except start a fetch.
    ///
    /// One place decides, because two ways in must not decide differently. An
    /// action off the gossip flood and one recovered by
    /// [`reconcile`](crate::reconcile) need the same thing: a `SetSubstrate`
    /// applied before its substrate has landed bakes a smooth deposit into
    /// stored tiles that no later arrival un-bakes (§6.4), and it does not
    /// matter which door it came through.
    ///
    /// `None` is handled — accepted, or parked behind a resolver already in
    /// flight and released with it. Accepted covers an action that references
    /// nothing *and* one that arrived without a transfer hash: there is nothing
    /// to fetch, so parking would be parking forever, and the kind's fallback
    /// is the best available. [`Some`] means parked, and this caller must start
    /// the resolver — only it knows which endpoints to ask.
    #[must_use = "a dropped MustFetch is a fetch that never starts, and a parked action never released"]
    pub fn admit(&self, action: Action, hash: Option<Hash>) -> Option<MustFetch> {
        let Some(need) = stark_model::action_content(&action) else {
            self.accept(action);
            return None;
        };
        let Some(hash) = hash_or_warn(need, hash) else {
            self.accept(action);
            return None;
        };
        match self.claim(need, &action) {
            Admit::Ready => {
                self.accept(action);
                None
            }
            Admit::Waiting => None,
            Admit::Fetch => Some(MustFetch { need, hash }),
        }
    }

    /// [`admit`](Self::admit) for content nothing will wait on — a live
    /// gesture's brush image, which must never block on a fetch (§17.5): the
    /// preview degrades to the round tip and upgrades when the bytes land.
    /// [`Some`] when this caller must start the resolver.
    #[must_use = "a dropped MustFetch is a fetch that never starts"]
    pub fn admit_detached(&self, need: AssetNeed, hash: Option<Hash>) -> Option<MustFetch> {
        let hash = hash_or_warn(need, hash)?;
        let mut parked = self.parked.lock().expect("waitlist poisoned");
        if self.mirror.lock().has(need) {
            return None;
        }
        match parked.entry(need) {
            Entry::Occupied(_) => None,
            Entry::Vacant(slot) => {
                slot.insert(Vec::new());
                Some(MustFetch { need, hash })
            }
        }
    }

    /// Decide what to do with `action`, which references `need` — [`admit`]'s
    /// parking primitive. Borrowed, and cloned only into the park: the common
    /// answer is [`Admit::Ready`], where the caller keeps the action it already
    /// had.
    ///
    /// [`admit`]: Self::admit
    pub fn claim(&self, need: AssetNeed, action: &Action) -> Admit {
        let mut parked = self.parked.lock().expect("waitlist poisoned");
        if self.mirror.lock().has(need) {
            return Admit::Ready;
        }
        match parked.entry(need) {
            Entry::Occupied(mut slot) => {
                // The flood redelivers, and a recovered copy can race the park
                // (`missing_from` subtracts parked ids only when the next sweep
                // asks, so a copy already in flight still lands here): the same
                // action must not park twice however many doors it comes through.
                if slot.get().iter().all(|a| a.id != action.id) {
                    slot.get_mut().push(action.clone());
                }
                Admit::Waiting
            }
            Entry::Vacant(slot) => {
                slot.insert(vec![action.clone()]);
                Admit::Fetch
            }
        }
    }

    /// Content fetched off a peer: hand it to the engine, then release
    /// everything parked behind it — in that order, which is the point. The
    /// engine having dropped its receiver changes none of the mirroring: the
    /// content and the actions are recorded either way, so this peer can still
    /// serve them onward.
    pub fn resolved(&self, need: AssetNeed, bytes: Bytes, hash: Hash) {
        let parked = self.take_parked(need, Some((bytes.clone(), hash)));
        let _ = self.events.send(RemoteEvent::Asset { need, bytes });
        self.release(parked);
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
        self.mirror.lock().transfer_hash(id)
    }

    /// Nothing could be fetched: release the parked actions to whatever fallback
    /// their kind has. Only ever reached for a brush — a substrate is not given up
    /// on (see [`Resolver::resolve`](crate::content::Resolver::resolve)).
    pub fn abandoned(&self, need: AssetNeed) {
        let parked = self.take_parked(need, None);
        self.release(parked);
    }

    /// A remote action arriving with everything it needs: mirror it and hand it
    /// to the engine, unless it has been seen before.
    pub fn accept(&self, action: Action) {
        self.release(vec![action]);
    }

    /// Which of `theirs` this peer does not hold — what a reconciliation sweep is
    /// asking (see [`reconcile`](crate::reconcile)). Routed through here rather
    /// than off a second mirror handle, so this file stays the only one holding
    /// one and the lock order stays a fact about one file.
    ///
    /// Parked counts as held: reporting it missing would only park another copy.
    /// It does not count for serving — it is not in the mirror until released.
    pub fn missing_from(&self, theirs: &[ActionId]) -> Vec<ActionId> {
        // Each lock in the file's order and neither across the diff: the walk is
        // O(m log n) over a member's whole digest, and the mirror's lock is one
        // the receive loop takes per arriving action. An action that lands
        // between the two snapshots is reported missing at worst, and a
        // re-claimed copy de-duplicates by id ([`claim`](Self::claim)).
        let parked: Vec<ActionId> = {
            let parked = self.parked.lock().expect("waitlist poisoned");
            parked.values().flatten().map(|a| a.id).collect()
        };
        let view = self.mirror.lock().log_view();
        let mut missing = view.missing_from(theirs);
        missing.retain(|id| !parked.contains(id));
        missing
    }

    /// A locally-committed action on its way out to the swarm: mirror it so this
    /// peer can serve it to a joiner, and hand nothing to the engine — it
    /// authored the action and has applied it already.
    pub fn published(&self, action: Action) {
        self.mirror.lock().insert(action);
    }

    /// Record content, if any, and detach everything parked behind `need`.
    fn take_parked(&self, need: AssetNeed, content: Option<(Bytes, Hash)>) -> Vec<Action> {
        let mut parked = self.parked.lock().expect("waitlist poisoned");
        if let Some((bytes, hash)) = content {
            self.mirror.lock().insert_content(need, bytes, hash);
        }
        parked.remove(&need).unwrap_or_default()
    }

    /// Mirror actions and hand them to the engine in order, skipping ones
    /// already seen. Mirroring never depends on the channel: the engine having
    /// stopped listening is not the session being over, and an action dropped
    /// short of the mirror is a hole in every log this peer serves a joiner —
    /// one no reconciliation from here can fill (§12).
    fn release(&self, actions: Vec<Action>) {
        for action in actions {
            let fresh = self.mirror.lock().insert_cloned(&action);
            if fresh {
                let _ = self.events.send(RemoteEvent::Action(action));
            }
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

/// The parking mechanism, without a network under it.
///
/// These are here rather than in an integration test because the properties
/// worth pinning are about *this* type — that a claim and a delivery cannot
/// interleave wrongly, that the asset precedes what waited on it — and reaching
/// them through two iroh endpoints would test the swarm instead, more slowly and
/// less exactly.
#[cfg(test)]
mod tests {
    use stark_model::AssetId;
    use stark_model::DocumentFile;
    use stark_model::Srgb;
    use stark_model::document::{
        ActionId, ActionKind, ActorId, BrushParams, BrushShape, LayerId, StrokeRecord,
    };
    use stark_model::geom::IVec2;

    use super::*;
    use crate::mirror::Mirror;

    fn setup() -> (Waitlist, mpsc::UnboundedReceiver<RemoteEvent>) {
        with_resolvable(&[])
    }

    /// A waitlist whose frontend claims it can produce `resolvable` itself.
    fn with_resolvable(resolvable: &[AssetId]) -> (Waitlist, mpsc::UnboundedReceiver<RemoteEvent>) {
        let mirror = SharedMirror::new(Mirror::from_file(&DocumentFile::new(Vec::new())));
        let (tx, rx) = mpsc::unbounded_channel();
        (Waitlist::new(mirror, tx, resolvable), rx)
    }

    fn id(lamport: u64) -> ActionId {
        ActionId {
            lamport,
            actor: ActorId(1),
        }
    }

    /// An action that references no content at all.
    fn action(lamport: u64) -> Action {
        Action {
            id: id(lamport),
            kind: ActionKind::SetSubstrateColor(Srgb::new([0.0; 3])),
        }
    }

    /// An action naming the brush behind `need(tag)` — the door derives the
    /// need from the action, so a test cannot pair them inconsistently.
    fn action_needing(lamport: u64, tag: u8) -> Action {
        Action {
            id: id(lamport),
            kind: ActionKind::CommitStroke(StrokeRecord {
                layer: LayerId::ROOT,
                brush: BrushParams {
                    shape: BrushShape::Stamp(AssetId([tag; 32])),
                    ..BrushParams::default()
                },
                path: Vec::new(),
                seed: 0,
                start: 0.0,
                translation: IVec2::ZERO,
            }),
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
        let (bytes, hash) = content(1);
        assert_eq!(
            waitlist.admit(action_needing(1, 1), Some(hash)),
            Some(MustFetch {
                need: need(1),
                hash
            }),
            "the first claimant is told to start the fetch"
        );
        assert!(
            drain(&mut rx).is_empty(),
            "nothing reaches the engine while parked"
        );

        waitlist.resolved(need(1), bytes, hash);

        // The order is the whole point: applying the action first is what bakes a
        // flat deposit for a substrate (§6.4).
        let events = drain(&mut rx);
        assert!(
            matches!(events[0], RemoteEvent::Asset { .. }),
            "asset first"
        );
        assert!(matches!(&events[1], RemoteEvent::Action(a) if a.id == id(1)));
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn content_already_held_does_not_park_at_all() {
        let (waitlist, mut rx) = setup();
        let (bytes, hash) = content(1);
        waitlist.imported(need(1), bytes, hash);
        assert_eq!(waitlist.admit(action_needing(1, 1), Some(hash)), None);
        let events = drain(&mut rx);
        assert!(
            matches!(&events[0], RemoteEvent::Action(a) if a.id == id(1)),
            "admitted straight through"
        );
        assert_eq!(events.len(), 1);
    }

    /// A live gesture's head and the commit it becomes name the same shape, and
    /// only one of them should be dialing for it.
    #[test]
    fn a_second_claimant_rides_the_first_resolver() {
        let (waitlist, mut rx) = setup();
        let (bytes, hash) = content(1);
        assert!(
            waitlist.admit_detached(need(1), Some(hash)).is_some(),
            "the head starts the fetch"
        );
        assert!(
            waitlist.admit_detached(need(1), Some(hash)).is_none(),
            "a repeated head starts nothing further"
        );
        assert!(
            waitlist.admit(action_needing(1, 1), Some(hash)).is_none(),
            "the commit parks behind the head's resolver"
        );

        waitlist.resolved(need(1), bytes, hash);
        assert_eq!(drain(&mut rx).len(), 2, "one asset, one released action");
    }

    /// A local import satisfies a remote action's need exactly as a fetch does —
    /// and if it did not release, the action would wait for a fetch of bytes
    /// already in hand.
    #[test]
    fn a_local_import_releases_what_was_waiting_on_it() {
        let (waitlist, mut rx) = setup();
        let (bytes, hash) = content(1);
        assert!(waitlist.admit(action_needing(1, 1), Some(hash)).is_some());

        waitlist.imported(need(1), bytes, hash);

        // No `Asset` event: the engine did the importing, so it already holds them.
        let events = drain(&mut rx);
        assert!(matches!(&events[0], RemoteEvent::Action(a) if a.id == id(1)));
        assert_eq!(events.len(), 1);
    }

    /// Giving up must still let the action through — a brush that never arrives
    /// costs the round tip, whereas an action parked forever costs the stroke.
    #[test]
    fn abandoning_a_fetch_releases_to_the_fallback() {
        let (waitlist, mut rx) = setup();
        let (_, hash) = content(1);
        assert!(waitlist.admit(action_needing(1, 1), Some(hash)).is_some());
        waitlist.abandoned(need(1));

        let events = drain(&mut rx);
        assert!(matches!(&events[0], RemoteEvent::Action(a) if a.id == id(1)));
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn parking_one_need_does_not_hold_up_another() {
        let (waitlist, mut rx) = setup();
        assert!(
            waitlist
                .admit(action_needing(1, 1), Some(content(1).1))
                .is_some()
        );
        assert!(
            waitlist
                .admit(action_needing(2, 2), Some(content(2).1))
                .is_some()
        );

        let (bytes, hash) = content(2);
        waitlist.resolved(need(2), bytes, hash);

        let ids: Vec<_> = drain(&mut rx)
            .iter()
            .filter_map(|e| match e {
                RemoteEvent::Action(a) => Some(a.id),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![id(2)], "only the resolved need released");
    }

    /// Gossip delivers the same action more than once, and a duplicate that
    /// parked would otherwise reach the engine a second time when its content
    /// landed.
    #[test]
    fn a_duplicate_action_surfaces_once() {
        let (waitlist, mut rx) = setup();
        let (bytes, hash) = content(1);
        assert!(waitlist.admit(action_needing(1, 1), Some(hash)).is_some());
        assert!(waitlist.admit(action_needing(1, 1), Some(hash)).is_none());

        waitlist.resolved(need(1), bytes, hash);
        assert_eq!(drain(&mut rx).len(), 2, "one asset, one action");

        waitlist.accept(action_needing(1, 1));
        assert!(drain(&mut rx).is_empty(), "and never again");
    }

    /// An action naming content the mirror lacks, arriving with no transfer
    /// hash: admitted anyway. There is nothing to fetch, parking would be
    /// parking forever, and the kind's fallback is the best available. Pinned
    /// here because the only other pin was a GPU integration test.
    #[test]
    fn no_transfer_hash_admits_rather_than_parks_forever() {
        let (waitlist, mut rx) = setup();
        assert_eq!(
            waitlist.admit(action_needing(1, 1), None),
            None,
            "nothing for the caller to start"
        );

        let events = drain(&mut rx);
        assert!(
            matches!(&events[0], RemoteEvent::Action(a) if a.id == id(1)),
            "released to the kind's fallback, not parked"
        );
        assert_eq!(events.len(), 1);
    }

    /// A parked action is not in the mirror yet, so without the subtraction a
    /// reconcile sweep reports it missing, re-claims it, and parks another copy
    /// — one per sweep for as long as the fetch stalls.
    #[test]
    fn a_reclaimed_parked_action_parks_once_and_counts_as_held() {
        let (waitlist, mut rx) = setup();
        let (bytes, hash) = content(1);
        assert!(waitlist.admit(action_needing(1, 1), Some(hash)).is_some());
        assert!(waitlist.admit(action_needing(1, 1), Some(hash)).is_none());

        assert!(
            waitlist.missing_from(&[id(1)]).is_empty(),
            "parked counts as held: reporting it missing is what re-claims it"
        );

        waitlist.resolved(need(1), bytes, hash);
        let events = drain(&mut rx);
        assert!(matches!(events[0], RemoteEvent::Asset { .. }));
        assert!(matches!(&events[1], RemoteEvent::Action(a) if a.id == id(1)));
        assert_eq!(events.len(), 2, "one asset, one action — not one per claim");
    }

    /// The engine dropping its receiver is not the session ending: what arrives
    /// must still enter the mirror, or a joiner served by this peer gets a log
    /// with holes no reconciliation from here can fill.
    #[test]
    fn a_closed_channel_still_mirrors_what_is_released() {
        let (waitlist, rx) = setup();
        let (bytes, hash) = content(1);
        assert!(waitlist.admit(action_needing(1, 1), Some(hash)).is_some());
        assert!(waitlist.admit(action_needing(2, 1), Some(hash)).is_none());
        drop(rx);

        waitlist.resolved(need(1), bytes, hash);
        assert!(
            waitlist.missing_from(&[id(1), id(2)]).is_empty(),
            "the whole batch entered the mirror with no one listening"
        );

        assert!(waitlist.admit(action(3), None).is_none());
        assert!(
            waitlist.missing_from(&[id(3)]).is_empty(),
            "and so does an action that never had to park"
        );
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
