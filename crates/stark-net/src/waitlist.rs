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
//! [`merge_remote`](stark_core::Engine::merge_remote) is idempotent by id and
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

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Arc, Mutex};

use stark_core::document::Action;
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
}

impl Waitlist {
    pub fn new(mirror: Arc<Mutex<Mirror>>, events: mpsc::UnboundedSender<RemoteEvent>) -> Self {
        Self {
            mirror,
            parked: Mutex::new(HashMap::new()),
            events,
        }
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
    pub fn resolved(&self, need: AssetNeed, bytes: Vec<u8>) {
        let parked = self.take_parked(need, Some(&bytes));
        if self.events.send(RemoteEvent::Asset { need, bytes }).is_ok() {
            self.release(parked);
        }
    }

    /// Content this client imported locally. The engine already holds it, so
    /// only the parked actions are released — but they *are* released, because a
    /// local import satisfies a remote action's need exactly as a fetch does.
    pub fn imported(&self, need: AssetNeed, bytes: Vec<u8>) {
        let parked = self.take_parked(need, Some(&bytes));
        self.release(parked);
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

    /// A locally-committed action on its way out to the swarm: mirror it so this
    /// peer can serve it to a joiner, and surface nothing — the engine authored
    /// it and has applied it already.
    pub fn published(&self, action: Action) {
        self.mirror.lock().expect("mirror poisoned").insert(action);
    }

    /// Record content, if any, and detach everything parked behind `need`.
    fn take_parked(&self, need: AssetNeed, bytes: Option<&[u8]>) -> Vec<Action> {
        let mut parked = self.parked.lock().expect("waitlist poisoned");
        if let Some(bytes) = bytes {
            self.mirror
                .lock()
                .expect("mirror poisoned")
                .insert_content(need, bytes.to_vec());
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
                .insert(action.clone());
            if fresh && self.events.send(RemoteEvent::Action(action)).is_err() {
                return;
            }
        }
    }
}
