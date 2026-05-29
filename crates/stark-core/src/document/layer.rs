//! Layers (DESIGN.md §5.1). A layer is a sparse, persistent map of populated
//! tiles plus a blend mode. Layer compositing across blend modes arrives in
//! step 4; for now layers stack with `Normal` over.

use rpds::HashTrieMap;
use serde::{Deserialize, Serialize};

use crate::geom::TileCoord;
use crate::gpu::tile::TileHandle;

/// Stable identifier for a layer within a document.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerId(pub u64);

/// How a layer combines with the layers below it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    Normal,
}

/// A single layer: a persistent (structurally-shared) map of tiles plus its
/// presentation properties.
#[derive(Clone)]
pub struct Layer {
    pub id: LayerId,
    pub blend: BlendMode,
    /// Layer opacity in [0, 1].
    pub opacity: f32,
    /// Whether the layer contributes to the composite.
    pub visible: bool,
    /// Only populated tiles exist — this sparsity is the infinite canvas.
    pub tiles: HashTrieMap<TileCoord, TileHandle>,
}

impl Layer {
    pub fn new(id: LayerId) -> Self {
        Self {
            id,
            blend: BlendMode::Normal,
            opacity: 1.0,
            visible: true,
            tiles: HashTrieMap::new(),
        }
    }
}
