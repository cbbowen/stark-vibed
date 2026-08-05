//! The session mirror: a CPU-side copy of the shared log + assets, so the
//! transport can serve joining peers without touching the engine (which lives
//! on the UI thread and owns the GPU). Assets also live in the blob store for
//! peer fetches; the mirror's copy is what snapshots bundle.
//!
//! The mirror sees every action exactly once — the initial snapshot, local
//! commits via [`CollabSession::broadcast`](crate::CollabSession::broadcast),
//! and remote actions from gossip — so any peer can bootstrap any other.

use std::collections::{BTreeMap, HashMap};

use stark_core::document::{Action, ActionId};
use stark_core::{AssetId, BuildId, CanvasMeta, DocumentFile, SurfaceId};

use crate::session::AssetNeed;

#[derive(Debug)]
pub(crate) struct Mirror {
    build: BuildId,
    canvas: CanvasMeta,
    /// Sorted by [`ActionId`] — iteration yields the total order.
    actions: BTreeMap<ActionId, Action>,
    assets: HashMap<AssetId, Vec<u8>>,
    /// The canvas grounds the log names, as canonical height maps (§6.4).
    ///
    /// Kept apart from `assets` for the reason the save file keeps them apart: the
    /// two are both grayscale PNGs and both content-addressed, but a brush mask
    /// decodes as luminance × alpha and a ground as channel 0, so one bag would hand
    /// each store the other's bytes to reinterpret.
    surfaces: HashMap<SurfaceId, Vec<u8>>,
}

impl Mirror {
    pub fn from_file(file: &DocumentFile) -> Self {
        Self {
            build: file.app_build.clone(),
            canvas: file.canvas.clone(),
            actions: file.actions.iter().map(|a| (a.id, a.clone())).collect(),
            assets: file.assets.iter().cloned().collect(),
            surfaces: file.surfaces.iter().cloned().collect(),
        }
    }

    /// The full session snapshot, as the save-format container (§8 ==
    /// §12.4's join payload): total-ordered actions + every known brush asset
    /// and canvas ground.
    pub fn document_file(&self) -> DocumentFile {
        let mut file = DocumentFile::new(self.actions.values().cloned().collect());
        file.app_build = self.build.clone();
        file.canvas = self.canvas.clone();
        file.assets = self.assets.iter().map(|(id, b)| (*id, b.clone())).collect();
        file.surfaces = self
            .surfaces
            .iter()
            .map(|(id, b)| (*id, b.clone()))
            .collect();
        file
    }

    /// Record an action; returns whether it was new.
    pub fn insert(&mut self, action: Action) -> bool {
        self.actions.insert(action.id, action).is_none()
    }

    /// Record content a peer may ask for, under the id that names it.
    pub fn insert_content(&mut self, need: AssetNeed, bytes: Vec<u8>) {
        match need {
            AssetNeed::Brush(id) => {
                self.assets.insert(id, bytes);
            }
            AssetNeed::Ground(id) => {
                self.surfaces.insert(id, bytes);
            }
        }
    }

    /// Whether this peer already holds what `need` names — the test that decides
    /// whether an arriving action has to wait on a fetch.
    pub fn has(&self, need: AssetNeed) -> bool {
        match need {
            AssetNeed::Brush(id) => self.assets.contains_key(&id),
            AssetNeed::Ground(id) => self.surfaces.contains_key(&id),
        }
    }

    /// Every piece of content this peer can serve, paired with the id it transfers
    /// under — for seeding the blob store at session start.
    ///
    /// Both kinds, keyed by their common content hash: a ground's transfer id is the
    /// [`AssetId`] inside its [`SurfaceId`], because both are the same BLAKE3 of the
    /// same canonical bytes. The blob store only ever moves bytes, so it has no need
    /// to know which kind it is holding — that is the receiver's question, answered
    /// by the action that referenced them.
    pub fn contents(&self) -> Vec<(AssetId, Vec<u8>)> {
        let assets = self.assets.iter().map(|(id, b)| (*id, b.clone()));
        let grounds = self
            .surfaces
            .iter()
            .filter_map(|(id, b)| Some((ground_content_id(*id)?, b.clone())));
        assets.chain(grounds).collect()
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
