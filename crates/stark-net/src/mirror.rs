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
//! store's and the one handed to the engine be the same allocation, and what
//! keeps [`Mirror::snapshot`] off the lock for longer than the log takes.

use std::collections::{BTreeMap, HashMap};

use bytes::Bytes;
use iroh_blobs::Hash;
use stark_core::document::{Action, ActionId};
use stark_core::{AssetId, BuildId, CanvasMeta, DocumentFile, SurfaceId};

use crate::session::AssetNeed;

#[derive(Debug)]
pub(crate) struct Mirror {
    build: BuildId,
    canvas: CanvasMeta,
    /// Sorted by [`ActionId`] — iteration yields the total order.
    actions: BTreeMap<ActionId, Action>,
    assets: HashMap<AssetId, Bytes>,
    /// The canvas grounds the log names, as canonical height maps (§6.4).
    ///
    /// Kept apart from `assets` for the reason the save file keeps them apart: the
    /// two are both grayscale PNGs and both content-addressed, but a brush mask
    /// decodes as luminance × alpha and a ground as channel 0, so one bag would hand
    /// each store the other's bytes to reinterpret.
    surfaces: HashMap<AssetId, Bytes>,
    /// The blob hash each piece of content transfers under.
    ///
    /// An [`AssetId`] names the *decoded coverage* (encoding-independent), so it is
    /// not itself fetchable over blobs; the transfer hash is the BLAKE3 of the bytes
    /// as they move. Held beside the content rather than in a map of its own, so
    /// there is one thing to keep in step with an import instead of two.
    hashes: HashMap<AssetId, Hash>,
}

/// A session snapshot's parts, cloned out of the mirror. Every clone here is a
/// refcount bump or a `BTreeMap` walk, so the caller's lock covers the log and
/// nothing else; turning it into the save-format container — which copies the
/// bytes, since [`DocumentFile`] owns its payloads — happens off the lock in
/// [`Snapshot::into_file`].
pub(crate) struct Snapshot {
    build: BuildId,
    canvas: CanvasMeta,
    actions: Vec<Action>,
    assets: Vec<(AssetId, Bytes)>,
    surfaces: Vec<(AssetId, Bytes)>,
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
        let omit = |id: &AssetId| have.contains(id);
        let before = self.assets.len() + self.surfaces.len();
        self.assets.retain(|(id, _)| !omit(id));
        self.surfaces.retain(|(id, _)| !omit(id));
        let spared: usize = before - (self.assets.len() + self.surfaces.len());
        if spared > 0 {
            tracing::debug!(spared, "omitted content the joiner can resolve locally");
        }
        self
    }

    pub fn into_file(self) -> DocumentFile {
        let mut file = DocumentFile::new(self.actions);
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
            hashes: HashMap::new(),
        }
    }

    /// The full session snapshot (§8 == §12.4's join payload): total-ordered
    /// actions + every known brush asset and canvas ground.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            build: self.build.clone(),
            canvas: self.canvas.clone(),
            actions: self.actions.values().cloned().collect(),
            assets: self.assets.iter().map(|(id, b)| (*id, b.clone())).collect(),
            surfaces: self
                .surfaces
                .iter()
                .map(|(id, b)| (*id, b.clone()))
                .collect(),
        }
    }

    /// Record an action; returns whether it was new.
    pub fn insert(&mut self, action: Action) -> bool {
        self.actions.insert(action.id, action).is_none()
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
        }
        self.hashes.insert(need.content(), hash);
    }

    /// Whether this peer already holds what `need` names — the test that decides
    /// whether an arriving action has to wait on a fetch.
    pub fn has(&self, need: AssetNeed) -> bool {
        match need {
            AssetNeed::Brush(id) => self.assets.contains_key(&id),
            AssetNeed::Ground(id) => self.surfaces.contains_key(&id),
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
        let hashes: Vec<(AssetId, Hash)> = assets
            .chain(grounds)
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
