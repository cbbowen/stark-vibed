//! The session mirror: a CPU-side copy of the shared log + assets, so the
//! transport can serve joining peers and asset requests without touching the
//! engine (which lives on the UI thread and owns the GPU).
//!
//! The mirror sees every action exactly once — the initial snapshot, local
//! commits via [`CollabSession::broadcast`](crate::CollabSession::broadcast),
//! and remote actions from gossip — so any peer can bootstrap any other.

use std::collections::{BTreeMap, HashMap};

use stark_core::document::{Action, ActionId};
use stark_core::{AssetId, BuildId, CanvasMeta, DocumentFile};

#[derive(Debug)]
pub(crate) struct Mirror {
    build: BuildId,
    canvas: CanvasMeta,
    /// Sorted by [`ActionId`] — iteration yields the total order.
    actions: BTreeMap<ActionId, Action>,
    assets: HashMap<AssetId, Vec<u8>>,
}

impl Mirror {
    pub fn from_file(file: &DocumentFile) -> Self {
        Self {
            build: file.app_build.clone(),
            canvas: file.canvas.clone(),
            actions: file.actions.iter().map(|a| (a.id, a.clone())).collect(),
            assets: file.assets.iter().cloned().collect(),
        }
    }

    /// The full session snapshot, as the save-format container (DESIGN.md §8 ==
    /// §12.4's join payload): total-ordered actions + every known brush asset.
    pub fn document_file(&self) -> DocumentFile {
        let mut file = DocumentFile::new(self.actions.values().cloned().collect());
        file.app_build = self.build.clone();
        file.canvas = self.canvas.clone();
        file.assets = self.assets.iter().map(|(id, b)| (*id, b.clone())).collect();
        file
    }

    /// Record an action; returns whether it was new.
    pub fn insert(&mut self, action: Action) -> bool {
        self.actions.insert(action.id, action).is_none()
    }

    pub fn insert_asset(&mut self, id: AssetId, bytes: Vec<u8>) {
        self.assets.insert(id, bytes);
    }

    pub fn has_asset(&self, id: AssetId) -> bool {
        self.assets.contains_key(&id)
    }

    pub fn asset(&self, id: AssetId) -> Option<Vec<u8>> {
        self.assets.get(&id).cloned()
    }
}
