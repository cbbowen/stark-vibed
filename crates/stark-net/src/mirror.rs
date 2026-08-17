//! The session mirror: a CPU-side copy of the shared log + assets, so the
//! transport can serve joining peers without touching the engine (which lives
//! on the UI thread and owns the GPU). Assets also live in the blob store for
//! peer fetches; the mirror's copy is what snapshots bundle.
//!
//! The mirror sees every action exactly once — the initial snapshot, local
//! commits via [`CollabSession::broadcast`](crate::CollabSession::broadcast),
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

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use bytes::Bytes;
use iroh_blobs::Hash;
use rpds::RedBlackTreeMapSync;
use stark_model::AssetId;
use stark_model::SurfaceId;
use stark_model::document::{Action, ActionId};
use stark_model::{BuildId, CanvasMeta, DocumentFile};

use crate::session::AssetNeed;

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
pub(crate) struct Served(Arc<OnceLock<Arc<Mutex<Mirror>>>>);

impl Served {
    /// Hand the session's mirror to the catch-up server — the moment this peer
    /// starts being a member rather than becoming one.
    pub fn publish(&self, mirror: Arc<Mutex<Mirror>>) {
        assert!(
            self.0.set(mirror).is_ok(),
            "one session, one published mirror"
        );
    }

    /// The mirror to answer from, or `None` while this peer is still joining.
    pub fn get(&self) -> Option<&Mutex<Mirror>> {
        self.0.get().map(|mirror| &**mirror)
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
    assets: HashMap<AssetId, Bytes>,
    /// The canvas grounds the log names, as canonical height maps (§6.4).
    ///
    /// Kept apart from `assets` for the reason the save file keeps them apart: the
    /// two are both grayscale PNGs and both content-addressed, but a brush mask
    /// decodes as luminance × alpha and a ground as channel 0, so one bag would hand
    /// each store the other's bytes to reinterpret.
    surfaces: HashMap<AssetId, Bytes>,
    /// The pictures the log places (§23). A third map for the second one's reason:
    /// the three decode differently, so a single bag would hand each store the
    /// others' bytes to reinterpret.
    pictures: HashMap<AssetId, Bytes>,
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
/// for a session with imported grounds, on a peer that is also painting.
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
    assets: Vec<(AssetId, Bytes)>,
    surfaces: Vec<(AssetId, Bytes)>,
    pictures: Vec<(AssetId, Bytes)>,
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
        let have: std::collections::HashSet<&AssetId> = have.iter().collect();
        let omit = |id: &AssetId| have.contains(id);
        let before = self.assets.len() + self.surfaces.len() + self.pictures.len();
        self.assets.retain(|(id, _)| !omit(id));
        self.surfaces.retain(|(id, _)| !omit(id));
        self.pictures.retain(|(id, _)| !omit(id));
        let spared: usize =
            before - (self.assets.len() + self.surfaces.len() + self.pictures.len());
        if spared > 0 {
            tracing::debug!(spared, "omitted content the joiner can resolve locally");
        }
        self
    }

    pub fn into_file(self) -> DocumentFile {
        let mut file = DocumentFile::new(self.actions.iter().map(|(_, a)| a.clone()).collect());
        file.app_build = self.build;
        file.canvas = self.canvas;
        file.assets = self
            .assets
            .into_iter()
            .map(|(id, b)| (id, b.to_vec()))
            .collect();
        file.surfaces = self
            .surfaces
            .into_iter()
            .map(|(id, b)| (SurfaceId::Image(id), b.to_vec()))
            .collect();
        file.pictures = self
            .pictures
            .into_iter()
            .map(|(id, b)| (id, b.to_vec()))
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
            assets: file
                .assets
                .iter()
                .map(|(id, b)| (*id, Bytes::from(b.clone())))
                .collect(),
            // A `Flat` entry would carry no bytes and name no content; the save
            // format cannot produce one, and skipping it is what keeps every
            // ground in here fetchable.
            surfaces: file
                .surfaces
                .iter()
                .filter_map(|(id, b)| Some((ground_content_id(*id)?, Bytes::from(b.clone()))))
                .collect(),
            pictures: file
                .pictures
                .iter()
                .map(|(id, b)| (*id, Bytes::from(b.clone())))
                .collect(),
            hashes: HashMap::new(),
            revision: 0,
            encoded: None,
        }
    }

    /// The full session snapshot (§8 == §12.4's join payload): total-ordered
    /// actions + every known brush asset and canvas ground.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            build: self.build.clone(),
            canvas: self.canvas.clone(),
            actions: self.actions.clone(),
            assets: self.assets.iter().map(|(id, b)| (*id, b.clone())).collect(),
            surfaces: self
                .surfaces
                .iter()
                .map(|(id, b)| (*id, b.clone()))
                .collect(),
            pictures: self
                .pictures
                .iter()
                .map(|(id, b)| (*id, b.clone()))
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
        if self.actions.contains_key(&action.id) {
            return false;
        }
        self.actions.insert_mut(action.id, action.clone());
        self.revision = self.revision.wrapping_add(1);
        true
    }

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
    pub fn recover(&self, ids: &[ActionId]) -> Vec<crate::proto::Recovered> {
        ids.iter()
            .filter_map(|id| self.actions.get(id))
            .map(|action| crate::proto::Recovered {
                action: action.clone(),
                hash: stark_model::action_content(action)
                    .and_then(|need| self.transfer_hash(need.content())),
            })
            .collect()
    }

    /// Record content a peer may ask for, under the id that names it and the
    /// hash it transfers under.
    pub fn insert_content(&mut self, need: AssetNeed, bytes: Bytes, hash: Hash) {
        match need {
            AssetNeed::Brush(id) => {
                self.assets.insert(id, bytes);
            }
            AssetNeed::Ground(id) => {
                self.surfaces.insert(id, bytes);
            }
            AssetNeed::Picture(id) => {
                self.pictures.insert(id, bytes);
            }
        }
        self.hashes.insert(need.content(), hash);
        // A snapshot bundles payloads, so this moves what one would say.
        self.revision = self.revision.wrapping_add(1);
    }

    /// Whether this peer already holds what `need` names — the test that decides
    /// whether an arriving action has to wait on a fetch.
    pub fn has(&self, need: AssetNeed) -> bool {
        match need {
            AssetNeed::Brush(id) => self.assets.contains_key(&id),
            AssetNeed::Ground(id) => self.surfaces.contains_key(&id),
            AssetNeed::Picture(id) => self.pictures.contains_key(&id),
        }
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
    /// Both kinds go in, keyed by their common content hash: a ground's transfer id
    /// is the [`AssetId`] inside its [`SurfaceId`], because both are the same BLAKE3
    /// of the same canonical bytes. The blob store only ever moves bytes, so it has
    /// no need to know which kind it is holding — that is the receiver's question,
    /// answered by the action that referenced them.
    pub fn seed_blobs(&mut self, add: impl Fn(Bytes) -> Hash) {
        let assets = self.assets.iter().map(|(id, b)| (*id, b.clone()));
        let grounds = self.surfaces.iter().map(|(id, b)| (*id, b.clone()));
        let pictures = self.pictures.iter().map(|(id, b)| (*id, b.clone()));
        let hashes: Vec<(AssetId, Hash)> = assets
            .chain(grounds)
            .chain(pictures)
            .map(|(id, bytes)| (id, add(bytes)))
            .collect();
        self.hashes.extend(hashes);
    }
}

/// The content hash a ground transfers under — `None` for `Flat`, which is
/// procedural and has no bytes to move.
pub(crate) fn ground_content_id(id: SurfaceId) -> Option<AssetId> {
    match id {
        SurfaceId::Flat => None,
        SurfaceId::Image(asset) => Some(asset),
    }
}
