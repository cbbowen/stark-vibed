//! The session's live gossip-neighbor set — its own module for the reason
//! [`cancel`](crate::cancel) is: everything holds one (ticket minting, the link
//! badges, the widening fetch, the sweep partner, and the receive loop that
//! maintains it), and the lower modules that do must not import `session`.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard};

use iroh::EndpointId;

/// A shared handle to the neighbor set: clones see one set.
#[derive(Debug, Clone, Default)]
pub(crate) struct Neighbors(Arc<Mutex<HashSet<EndpointId>>>);

impl Neighbors {
    /// The one lock site, so the poison policy is stated once: nothing under
    /// this lock runs code that can panic short of a bug in this crate.
    fn lock(&self) -> MutexGuard<'_, HashSet<EndpointId>> {
        self.0.lock().expect("neighbors poisoned")
    }

    pub fn insert(&self, peer: EndpointId) {
        self.lock().insert(peer);
    }

    pub fn remove(&self, peer: EndpointId) {
        self.lock().remove(&peer);
    }

    /// The membership right now, in no particular order.
    pub fn snapshot(&self) -> Vec<EndpointId> {
        self.lock().iter().copied().collect()
    }

    /// The membership, sorted by key bytes. One order per membership is
    /// load-bearing twice: a ticket's text changes only when its members do
    /// (the frontend re-mints on a cadence and rewrites the invitation only on
    /// change), and the reconciler's rotation visits every member instead of
    /// chasing hash order, which moves as the set changes. The link badges take
    /// it too, so a polled roster keeps one order between polls.
    pub fn snapshot_sorted(&self) -> Vec<EndpointId> {
        let mut ids = self.snapshot();
        ids.sort_unstable_by_key(|id| *id.as_bytes());
        ids
    }
}

impl FromIterator<EndpointId> for Neighbors {
    fn from_iter<T: IntoIterator<Item = EndpointId>>(iter: T) -> Self {
        Self(Arc::new(Mutex::new(iter.into_iter().collect())))
    }
}
