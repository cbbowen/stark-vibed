//! `DocState`: the versioned document state (DESIGN.md §5.1).
//!
//! `DocState` is the `history` crate's `State`, so cloning it must be cheap: it
//! holds `rpds` persistent collections whose clone is a handful of `Arc` bumps.
//! The heavy GPU memory lives behind `TileHandle`s shared across versions, and
//! is reclaimed when the last version referencing a tile drops (DESIGN.md §5.2).

use rpds::Vector;

use super::layer::{BlendMode, Layer, LayerId};
use crate::geom::TileCoord;

/// Inclusive tile-coordinate bounding box of all populated tiles (DESIGN.md §6),
/// i.e. the explored extent of the infinite canvas.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CanvasBounds {
    range: Option<(TileCoord, TileCoord)>,
}

impl CanvasBounds {
    /// The `(min, max)` inclusive tile range, or `None` if nothing is painted.
    pub fn tile_range(&self) -> Option<(TileCoord, TileCoord)> {
        self.range
    }

    fn include(&mut self, c: TileCoord) {
        self.range = Some(match self.range {
            None => (c, c),
            Some((min, max)) => (
                TileCoord::new(min.x.min(c.x), min.y.min(c.y)),
                TileCoord::new(max.x.max(c.x), max.y.max(c.y)),
            ),
        });
    }
}

/// The full document: an ordered stack of layers and the explored bounds.
#[derive(Clone)]
pub struct DocState {
    pub layers: Vector<Layer>,
    pub bounds: CanvasBounds,
}

impl DocState {
    /// An empty document with a single starting layer.
    pub fn with_layer(id: LayerId) -> Self {
        Self {
            layers: Vector::new().push_back(Layer::new(id)),
            bounds: CanvasBounds::default(),
        }
    }

    /// Index of the layer with the given id, if present.
    pub fn layer_index(&self, id: LayerId) -> Option<usize> {
        self.layers.iter().position(|l| l.id == id)
    }

    /// The layer at `index`. Panics if out of range (callers hold a valid index).
    pub fn layer_at(&self, index: usize) -> &Layer {
        self.layers.get(index).expect("layer index in range")
    }

    /// Replace the layer at `index`, returning a new document with bounds
    /// recomputed from all layers' populated tiles.
    pub fn with_layer_at(&self, index: usize, layer: Layer) -> Self {
        let layers = self
            .layers
            .set(index, layer)
            .expect("layer index in range");
        Self::from_layers(layers)
    }

    /// Insert a new empty layer directly above `above` (or on top if `None`).
    pub fn insert_layer(&self, id: LayerId, above: Option<LayerId>) -> Self {
        let at = match above {
            Some(target) => self.layer_index(target).map_or(self.layers.len(), |i| i + 1),
            None => self.layers.len(),
        };
        // rpds::Vector has no insert-at; rebuild with the new layer spliced in.
        let mut layers = Vector::new();
        for (i, l) in self.layers.iter().enumerate() {
            if i == at {
                layers = layers.push_back(Layer::new(id));
            }
            layers = layers.push_back(l.clone());
        }
        if at >= self.layers.len() {
            layers = layers.push_back(Layer::new(id));
        }
        Self::from_layers(layers)
    }

    /// Remove the layer with the given id (no-op if absent).
    pub fn remove_layer(&self, id: LayerId) -> Self {
        let mut layers = Vector::new();
        for l in self.layers.iter() {
            if l.id != id {
                layers = layers.push_back(l.clone());
            }
        }
        Self::from_layers(layers)
    }

    /// Set the blend mode of a layer (no-op if absent).
    pub fn set_layer_blend(&self, id: LayerId, blend: BlendMode) -> Self {
        self.map_layer(id, |l| Layer { blend, ..l })
    }

    /// Set a layer's opacity, clamped to [0, 1] (no-op if absent).
    pub fn set_layer_opacity(&self, id: LayerId, opacity: f32) -> Self {
        self.map_layer(id, |l| Layer {
            opacity: opacity.clamp(0.0, 1.0),
            ..l
        })
    }

    /// Set a layer's visibility (no-op if absent).
    pub fn set_layer_visible(&self, id: LayerId, visible: bool) -> Self {
        self.map_layer(id, |l| Layer { visible, ..l })
    }

    /// Move layer `id` to sit immediately above `above`, or on top if `None`.
    /// The layer keeps its tiles, so the painting moves with it.
    pub fn move_layer(&self, id: LayerId, above: Option<LayerId>) -> Self {
        let Some(moved) = self.layers.iter().find(|l| l.id == id).cloned() else {
            return self.clone();
        };
        let remaining: Vec<Layer> = self.layers.iter().filter(|l| l.id != id).cloned().collect();
        let at = match above {
            Some(target) => remaining
                .iter()
                .position(|l| l.id == target)
                .map_or(remaining.len(), |i| i + 1),
            None => remaining.len(),
        };
        let mut layers = Vector::new();
        for (i, l) in remaining.iter().enumerate() {
            if i == at {
                layers = layers.push_back(moved.clone());
            }
            layers = layers.push_back(l.clone());
        }
        if at >= remaining.len() {
            layers = layers.push_back(moved);
        }
        Self::from_layers(layers)
    }

    fn map_layer(&self, id: LayerId, f: impl FnOnce(Layer) -> Layer) -> Self {
        match self.layer_index(id) {
            Some(idx) => self.with_layer_at(idx, f(self.layer_at(idx).clone())),
            None => self.clone(),
        }
    }

    fn from_layers(layers: Vector<Layer>) -> Self {
        let mut bounds = CanvasBounds::default();
        for layer in layers.iter() {
            for coord in layer.tiles.keys() {
                bounds.include(*coord);
            }
        }
        Self { layers, bounds }
    }
}
