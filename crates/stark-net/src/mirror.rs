//! The session mirror: a CPU-side copy of the shared log + assets, so the
//! transport can serve joining peers without touching the engine (which lives
//! on the UI thread and owns the GPU). Assets also live in the blob store for
//! peer fetches; the mirror's copy is what snapshots bundle.
//!
//! The mirror sees every action exactly once — the initial snapshot, local
//! commits via [`Broadcaster::broadcast`](crate::Broadcaster::broadcast),
//! and remote actions from gossip — so any peer can bootstrap any other.
//!
//! Content is held as [`Bytes`], which is what lets the mirror's copy, the blob
//! store's and the one handed to the engine be the same allocation.
//!
//! **Nothing expensive happens under the lock.** The receive loop takes it for
//! every arriving action and every broadcast takes it to look up a transfer hash,
//! so a joiner arriving mid-session must not stall this peer's painting for the
//! size of the session. The log is persistent, so taking a [`Snapshot`] is a
//! refcount bump; copying the actions out of it, copying the asset payloads into a
//! [`DocumentFile`] and encoding the result all happen off the lock, and the result
//! is remembered so the next joiner asking the same question pays for none of it.
//! Reconciliation answers go the same way: a [`LogView`] is taken under the lock
//! and the id walks and action clones happen off it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use bytes::Bytes;
use iroh_blobs::Hash;
use rpds::RedBlackTreeMapSync;
use stark_model::document::{Action, ActionId};
use stark_model::{AssetId, AssetNeed, BuildId, CanvasMeta, DocumentFile};

/// The mirror as the catch-up server sees it: absent until this peer is a session
/// member.
///
/// A peer binds its endpoint — and with it the catch-up protocol — *before* it has
/// anything to serve. A joiner has not fetched its own snapshot yet; a host has not
/// seeded its blob store. An empty [`Mirror`] and a complete one are the same value,
/// so a request arriving in that window would be answered with a well-formed, empty,
/// silently-wrong document, and the joiner that got it would ride the gossip tail
/// believing it had the painting.
///
/// So publishing is a step of its own, taken once the session is real, and until it
/// is taken there is nothing here to mistake for a session. The joiner turned away
/// asks another member — every one of them is an entry point (§12.4).
#[derive(Debug, Clone, Default)]
pub(crate) struct Served(Arc<OnceLock<SharedMirror>>);

impl Served {
    /// Hand the session's mirror to the catch-up server — the moment this peer
    /// starts being a member rather than becoming one.
    pub fn publish(&self, mirror: SharedMirror) {
        assert!(
            self.0.set(mirror).is_ok(),
            "one session, one published mirror"
        );
    }

    /// The mirror to answer from, or `None` while this peer is still joining.
    pub fn get(&self) -> Option<&SharedMirror> {
        self.0.get()
    }
}

/// The shared handle to a session's [`Mirror`]: clones see one mirror, and
/// [`lock`](Self::lock) is the only door through it.
#[derive(Debug, Clone)]
pub(crate) struct SharedMirror(Arc<Mutex<Mirror>>);

impl SharedMirror {
    pub fn new(mirror: Mirror) -> Self {
        Self(Arc::new(Mutex::new(mirror)))
    }

    /// The one lock site, so the poison policy is stated once: nothing under
    /// this lock runs code that can panic short of a bug in this crate.
    pub fn lock(&self) -> MutexGuard<'_, Mirror> {
        self.0.lock().expect("mirror poisoned")
    }
}

#[derive(Debug)]
pub(crate) struct Mirror {
    build: BuildId,
    canvas: CanvasMeta,
    /// Sorted by [`ActionId`] — iteration yields the total order.
    ///
    /// Persistent (`rpds`) for the reason `DocState`'s maps are (CLAUDE.md): the
    /// whole purpose of keeping the log here is to hand it to a joiner, and the
    /// lock it is kept behind is one the receive loop takes per action. Cloning
    /// the handle is a refcount bump, so a snapshot costs the lock nothing that
    /// grows; the per-action copy happens off it, in [`Snapshot::into_file`].
    ///
    /// The `Sync` flavour, unlike `DocState`'s maps: those live on the UI thread and
    /// spend `Rc`, while this one is shared between the receive loop, the catch-up
    /// server and every resolver.
    actions: RedBlackTreeMapSync<ActionId, Action>,
    /// Every piece of content the log names, keyed by the [`AssetNeed`] that
    /// says which store its bytes belong in — the same bag the save file keeps
    /// (`DocumentFile::content`, §8). The kind rides the key: a brush mask and
    /// a substrate are both grayscale PNGs that decode differently, and the key
    /// is what stops one store being handed the other's bytes to reinterpret.
    content: HashMap<AssetNeed, Bytes>,
    /// The blob hash each piece of content transfers under.
    ///
    /// An [`AssetId`] names the *decoded coverage* (encoding-independent), so it is
    /// not itself fetchable over blobs; the transfer hash is the BLAKE3 of the bytes
    /// as they move. Held beside the content rather than in a map of its own, so
    /// there is one thing to keep in step with an import instead of two.
    hashes: HashMap<AssetId, Hash>,
    /// Bumped by every change a snapshot would show, so [`Encoded`] can tell
    /// whether it is still the answer.
    revision: u64,
    encoded: Option<Encoded>,
}

/// The last snapshot this peer encoded, and what it was encoded for.
///
/// Encoding one is the most expensive thing the mirror does: [`DocumentFile`] owns
/// its payloads, so every asset is copied out of the [`Bytes`] holding it, and the
/// whole container is then encoded into a third buffer — megabytes twice over
/// for a session with imported substrates, on a peer that is also painting.
///
/// Most joins in a session ask the identical question. Peers running the same build
/// send the same `resolvable` list, and the log has usually not moved in the seconds
/// between two people opening the same link.
#[derive(Debug)]
struct Encoded {
    revision: u64,
    have: Vec<AssetId>,
    bytes: Bytes,
}

/// A session snapshot's parts, taken out of the mirror under its lock.
///
/// Every field is a refcount bump: the log is persistent and the payloads are
/// [`Bytes`]. Copying the actions out, copying the payloads into the save-format
/// container — which owns them — and encoding the result all happen afterwards, in
/// [`Snapshot::into_file`], with the lock released.
pub(crate) struct Snapshot {
    build: BuildId,
    canvas: CanvasMeta,
    actions: RedBlackTreeMapSync<ActionId, Action>,
    content: Vec<(AssetNeed, Bytes)>,
    /// What the mirror stood at when this was taken — the key its encoding is
    /// remembered under.
    pub revision: u64,
}

impl Snapshot {
    /// Leave out content the joiner said it can resolve without help — the assets
    /// that ship with its build (§12.4).
    ///
    /// Only the *payloads* go; the log is untouched, so the document still names
    /// everything it names and a joiner that cannot make its promise good still
    /// has an id to fetch by. That is what keeps this an optimization: the worst
    /// case of a wrong claim is the transfer that would have happened anyway.
    pub fn without(mut self, have: &[AssetId]) -> Self {
        if have.is_empty() {
            return self;
        }
        let have: std::collections::HashSet<AssetId> = have.iter().copied().collect();
        let before = self.content.len();
        // By the id the bytes are named under, whichever kind carries it — the
        // promise is a list of content ids, not needs.
        self.content
            .retain(|(need, _)| !have.contains(&need.content()));
        let spared = before - self.content.len();
        if spared > 0 {
            tracing::debug!(spared, "omitted content the joiner can resolve locally");
        }
        self
    }

    pub fn into_file(self) -> DocumentFile {
        let mut file = DocumentFile::new(self.actions.iter().map(|(_, a)| a.clone()).collect());
        file.app_build = self.build;
        file.canvas = self.canvas;
        // The container owns its payloads, so this is the per-asset copy the
        // mirror's lock must not cover. The bag's shape is already the file's (§8).
        file.content = self
            .content
            .into_iter()
            .map(|(need, b)| (need, b.to_vec()))
            .collect();
        file
    }
}

impl Mirror {
    pub fn from_file(file: &DocumentFile) -> Self {
        Self {
            build: file.app_build.clone(),
            canvas: file.canvas.clone(),
            actions: file.actions.iter().map(|a| (a.id, a.clone())).collect(),
            // The bundle is already keyed by [`AssetNeed`] (§8), so the bag moves
            // over whole. A `Flat` substrate cannot appear: it names no content,
            // so `AssetNeed` has no variant for it (`AssetNeed::for_substrate`).
            content: file
                .content
                .iter()
                .map(|(need, bytes)| (*need, Bytes::from(bytes.clone())))
                .collect(),
            hashes: HashMap::new(),
            revision: 0,
            encoded: None,
        }
    }

    /// The full session snapshot (§8 == §12.4's join payload): total-ordered
    /// actions + every piece of content the log names.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            build: self.build.clone(),
            canvas: self.canvas.clone(),
            actions: self.actions.clone(),
            content: self
                .content
                .iter()
                .map(|(need, b)| (*need, b.clone()))
                .collect(),
            revision: self.revision,
        }
    }

    /// The encoding of the snapshot `have` asks for, if the one remembered is
    /// still the answer.
    pub fn encoded_for(&self, have: &[AssetId]) -> Option<Bytes> {
        let cached = self.encoded.as_ref()?;
        (cached.revision == self.revision && cached.have == have).then(|| cached.bytes.clone())
    }

    /// Remember an encoding for the next joiner to ask the same question.
    ///
    /// Dropped if the log moved while it was being encoded — off the lock, which is
    /// the point, so it can. The bytes are still right for the revision they were
    /// taken at; they are just not right for the one anyone will ask about next.
    pub fn remember(&mut self, revision: u64, have: &[AssetId], bytes: Bytes) {
        if revision == self.revision {
            self.encoded = Some(Encoded {
                revision,
                have: have.to_vec(),
                bytes,
            });
        }
    }

    /// Record an action this peer holds the only copy of — a local commit already
    /// encoded onto the wire. Returns whether it was new.
    pub fn insert(&mut self, action: Action) -> bool {
        if self.actions.contains_key(&action.id) {
            return false;
        }
        self.actions.insert_mut(action.id, action);
        self.revision = self.revision.wrapping_add(1);
        true
    }

    /// The same, for a caller that keeps its own copy — the receive loop, which
    /// has to hand the action to the engine after mirroring it.
    ///
    /// Cloned only when the action is new. Gossip delivers duplicates as a matter
    /// of course (it is a flood), and a duplicate's clone would be dropped by the
    /// line after the one that made it.
    pub fn insert_cloned(&mut self, action: &Action) -> bool {
        !self.actions.contains_key(&action.id) && self.insert(action.clone())
    }

    /// What the reconciliation queries read, snapshotted under the lock: the
    /// log handle is a refcount bump and the hash table a small copy, where the
    /// queries themselves walk the whole log or clone actions — the work the
    /// module header forbids under a lock the receive loop shares per action.
    pub fn log_view(&self) -> LogView {
        LogView {
            actions: self.actions.clone(),
            hashes: self.hashes.clone(),
        }
    }

    /// Record content a peer may ask for, under the need that names it and the
    /// hash it transfers under.
    pub fn insert_content(&mut self, need: AssetNeed, bytes: Bytes, hash: Hash) {
        self.content.insert(need, bytes);
        self.hashes.insert(need.content(), hash);
        // A snapshot bundles payloads, so this moves what one would say.
        self.revision = self.revision.wrapping_add(1);
    }

    /// Whether this peer already holds what `need` names — the test that decides
    /// whether an arriving action has to wait on a fetch.
    pub fn has(&self, need: AssetNeed) -> bool {
        self.content.contains_key(&need)
    }

    /// The hash content transfers under, for a broadcast to attach so receivers
    /// that lack it know what to fetch.
    pub fn transfer_hash(&self, id: AssetId) -> Option<Hash> {
        self.hashes.get(&id).copied()
    }

    /// Hand every piece of content this peer already holds to the blob store, and
    /// record what it transfers under — the session-start seed, from the hosted
    /// document or a joiner's snapshot alike.
    ///
    /// Every kind goes in, keyed by its content hash: a substrate's transfer id
    /// is the [`AssetId`] inside its [`SubstrateId`](stark_model::SubstrateId), because both are the same BLAKE3
    /// of the same canonical bytes. The blob store only ever moves bytes, so it has
    /// no need to know which kind it is holding — that is the receiver's question,
    /// answered by the action that referenced them.
    pub fn seed_blobs(&mut self, add: impl Fn(Bytes) -> Hash) {
        let hashes: Vec<(AssetId, Hash)> = self
            .content
            .iter()
            .map(|(need, bytes)| (need.content(), add(bytes.clone())))
            .collect();
        self.hashes.extend(hashes);
    }
}

/// The log as reconciliation reads it, detached from the mirror's lock
/// ([`Mirror::log_view`]). The queries here are the ones too expensive for the
/// lock: a full-log id walk, an O(m log n) diff, per-action clones.
pub(crate) struct LogView {
    actions: RedBlackTreeMapSync<ActionId, Action>,
    hashes: HashMap<AssetId, Hash>,
}

impl LogView {
    /// Every action id held, in total order — this peer's half of a
    /// reconciliation digest.
    pub fn action_ids(&self) -> Vec<ActionId> {
        self.actions.keys().copied().collect()
    }

    /// Which of `theirs` this peer does not hold — the answer reconciliation is
    /// looking for, worked out against the digest a member sent.
    pub fn missing_from(&self, theirs: &[ActionId]) -> Vec<ActionId> {
        theirs
            .iter()
            .filter(|id| !self.actions.contains_key(id))
            .copied()
            .collect()
    }

    /// The named actions, each with the hash its content transfers under, for a
    /// member recovering what the flood dropped. Ids not held are skipped — the
    /// asker's digest may be older than this peer's own trimming of it.
    ///
    /// Plain pairs rather than the wire's shape: the store answers the question,
    /// and [`proto::answer`](crate::proto::answer) spells it for the wire.
    pub fn recover(&self, ids: &[ActionId]) -> Vec<(Action, Option<Hash>)> {
        ids.iter()
            .filter_map(|id| self.actions.get(id))
            .map(|action| {
                (
                    action.clone(),
                    stark_model::action_content(action)
                        .and_then(|need| self.hashes.get(&need.content()).copied()),
                )
            })
            .collect()
    }
}

/// The collapsed content bag: one map keyed by [`AssetNeed`], round-tripping
/// through the same-shaped bag the save file carries.
#[cfg(test)]
mod tests {
    use stark_model::Srgb;
    use stark_model::document::{ActionKind, ActorId};

    use super::*;

    fn file_with_content() -> DocumentFile {
        let action = Action {
            id: ActionId {
                lamport: 1,
                actor: ActorId(1),
            },
            kind: ActionKind::SetSubstrateColor(Srgb::new([0.0; 3])),
        };
        let mut file = DocumentFile::new(vec![action]);
        file.content = vec![
            (AssetNeed::Brush(AssetId([1; 32])), vec![1, 2, 3]),
            (AssetNeed::Substrate(AssetId([2; 32])), vec![4, 5]),
            (AssetNeed::Picture(AssetId([3; 32])), vec![6]),
        ];
        file
    }

    /// What a file bundles, one entry per kind, is what a snapshot of the
    /// mirror built from it hands the next joiner — nothing dropped, nothing
    /// reinterpreted, however the map iterates.
    #[test]
    fn the_content_bag_survives_file_to_snapshot_to_file() {
        let file = file_with_content();

        let back = Mirror::from_file(&file).snapshot().into_file();

        assert_eq!(back.actions.len(), 1);
        let mut content = back.content;
        content.sort_unstable_by_key(|(need, _)| *need);
        assert_eq!(content, file.content);
    }

    /// The promise subtracts by content id across every kind, and takes only
    /// payloads — the log still names what was omitted.
    #[test]
    fn without_drops_exactly_the_promised_ids() {
        let file = file_with_content();

        let trimmed = Mirror::from_file(&file)
            .snapshot()
            .without(&[AssetId([2; 32]), AssetId([3; 32])])
            .into_file();

        assert_eq!(
            trimmed.content,
            vec![(AssetNeed::Brush(AssetId([1; 32])), vec![1, 2, 3])]
        );
        assert_eq!(trimmed.actions.len(), 1, "only payloads go; the log stays");
    }
}
