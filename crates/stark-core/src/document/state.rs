//! `DocState`: the versioned document state (DESIGN.md §5.1).
//!
//! `DocState` is the `history` crate's `State`, so cloning it must be cheap: it
//! holds `rpds` persistent collections whose clone is a handful of `Arc` bumps.
//! The heavy GPU memory lives behind `TileHandle`s shared across versions, and
//! is reclaimed when the last version referencing a tile drops (DESIGN.md §5.2).

use std::sync::Arc;

use rpds::{HashTrieMap, Vector};

use super::action::ActorId;
use super::layer::{BlendMode, Layer, LayerContent, LayerId, MatteRegion};
use super::selection::Selection;
use crate::geom::{TileCoord, Vec2};
use crate::gpu::SurfaceId;

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

/// The full document: an ordered stack of layers, the explored bounds, and the
/// selection masks that gate where tools may act.
#[derive(Clone)]
pub struct DocState {
    pub layers: Vector<Layer>,
    pub bounds: CanvasBounds,
    /// The active selection **of each actor** (DESIGN.md §6.8, PEER_DESIGN.md §3).
    ///
    /// Document state, not session state: a stroke's pixels depend on the mask it
    /// was drawn through, so replay has to be able to reconstruct it — which is why
    /// selection edits are logged actions like any other. But *whose* mask is the
    /// author's, not the document's: one collaborator's lasso must not clip
    /// another's brush. So the mask is owned per actor, keyed by the very
    /// [`ActorId`] that orders the log, which is what makes "only its owner may
    /// change it" structural rather than a rule a call site could forget — the key
    /// comes from `Action::id.actor` and there is no way to write anyone else's.
    ///
    /// An absent entry is the unrestricted selection, so an actor who never selects
    /// costs nothing and a solo document has at most one entry.
    selections: HashTrieMap<ActorId, Selection>,
    /// The physical canvas surface (DESIGN.md §6.4). Document state: which canvas
    /// a piece was painted on is part of what the document *is*, it is saved, and
    /// reopening on a different weave would be a different painting. Today it is
    /// read only by the media pass (§6.3), so a switch changes no stored pixel —
    /// but it is logged like any other edit, which is what would let a future
    /// deposition gate read it without becoming a history change.
    pub surface: SurfaceId,
    /// The canvas substrate colour — the ground the paint sits on — as straight
    /// sRGB (FRAME_DESIGN.md §5).
    ///
    /// Document state on the same argument §6.4 makes for the weave: which ground
    /// a piece was painted on is part of what it *is*, and it must be saved. It
    /// was previously a view setting the frontend owned, which meant the paper
    /// colour of a painting was simply not stored anywhere.
    ///
    /// Distinct from a matte layer, which is a slab of opaque *paint*: the
    /// substrate sits under everything, is lit, and the canvas weave shows through
    /// it (the media pass composites paint over it, DESIGN.md §6.3).
    pub background: [f32; 3],
}

/// The default substrate: a light neutral grey ground. Neutral on purpose — an HDR
/// lights the scene warm, and a warm ground on top of that reads noticeably red.
/// Grey rather than near-white so paint has somewhere to go in *both* directions:
/// a highlight can read lighter than the bare canvas.
pub const DEFAULT_BACKGROUND: [f32; 3] = [0.85, 0.85, 0.85];

/// The canvas a new document starts on (DESIGN.md §6.4): linen, not `Flat`. The
/// weave is what paint has to sit in — its relief is read by the stroke pass
/// whether or not `MediaParams::surface_strength` makes the light show it — so it
/// is the honest starting substrate, and `Flat` is the deliberate switch away.
///
/// Distinct from `SurfaceId::default()`, which stays `Flat`: that is the *builtin*
/// the surface registry falls back to before any bytes arrive (the frontend fetches
/// the linen height map at runtime — DESIGN.md §6.6).
pub const DEFAULT_SURFACE: SurfaceId = SurfaceId::Linen;

impl DocState {
    /// An empty document with a single starting layer and nothing masked.
    pub fn with_layer(id: LayerId) -> Self {
        Self {
            layers: Vector::new().push_back(Layer::new(id)),
            bounds: CanvasBounds::default(),
            selections: HashTrieMap::new(),
            surface: DEFAULT_SURFACE,
            background: DEFAULT_BACKGROUND,
        }
    }

    /// The same document on a different canvas surface (DESIGN.md §6.4).
    pub fn with_surface(&self, surface: SurfaceId) -> Self {
        Self {
            surface,
            ..self.clone()
        }
    }

    /// The same document on a different substrate colour (FRAME_DESIGN.md §5).
    pub fn with_background(&self, background: [f32; 3]) -> Self {
        Self {
            background,
            ..self.clone()
        }
    }

    /// `actor`'s selection mask (DESIGN.md §6.8, PEER_DESIGN.md §3). An actor with
    /// no entry has selected nothing, which *is* the unrestricted selection.
    ///
    /// Returned by value because that is what the callers want and it costs a
    /// persistent-map clone — a handful of `Arc` bumps, the same price as cloning
    /// the `DocState` it came out of.
    pub fn selection_of(&self, actor: ActorId) -> Selection {
        self.selections
            .get(&actor)
            .cloned()
            .unwrap_or_else(Selection::everything)
    }

    /// Whether `actor` has a selection in force — the cheap test, with no clone.
    pub fn has_selection(&self, actor: ActorId) -> bool {
        self.selections
            .get(&actor)
            .is_some_and(Selection::is_active)
    }

    /// Every actor with a selection in force, in no particular order. Universal
    /// selections are never stored, so nothing here is empty.
    pub fn selections(&self) -> impl Iterator<Item = (ActorId, &Selection)> {
        self.selections.iter().map(|(a, s)| (*a, s))
    }

    /// The same document with `actor`'s selection replaced (DESIGN.md §6.8).
    ///
    /// A universal selection is *removed* rather than stored, so "no selection" has
    /// exactly one representation: `selections()` never yields an empty mask, and an
    /// actor who deselects stops costing anything again.
    pub fn with_selection(&self, actor: ActorId, selection: Selection) -> Self {
        let selections = if selection.is_universal() {
            self.selections.remove(&actor)
        } else {
            self.selections.insert(actor, selection)
        };
        Self {
            selections,
            ..self.clone()
        }
    }

    /// Index of the layer with the given id, if present.
    pub fn layer_index(&self, id: LayerId) -> Option<usize> {
        self.layers.iter().position(|l| l.id == id)
    }

    /// What the layer with the given id is called, or `None` if it is unnamed —
    /// or absent, which reads the same way here: neither has a name to give.
    pub fn layer_name(&self, id: LayerId) -> Option<&str> {
        self.layers
            .iter()
            .find(|l| l.id == id)
            .and_then(|l| l.name.as_deref())
    }

    /// The layer at `index`. Panics if out of range (callers hold a valid index).
    pub fn layer_at(&self, index: usize) -> &Layer {
        self.layers.get(index).expect("layer index in range")
    }

    /// Replace the layer at `index`, returning a new document with bounds
    /// recomputed from all layers' populated tiles.
    pub fn with_layer_at(&self, index: usize, layer: Layer) -> Self {
        let layers = self.layers.set(index, layer).expect("layer index in range");
        self.with_layers(layers)
    }

    /// Insert a new empty paint layer directly above `above` (or on top if `None`).
    pub fn insert_layer(&self, id: LayerId, above: Option<LayerId>) -> Self {
        self.insert(Layer::new(id), above)
    }

    /// Insert a matte layer directly above `above` (or on top if `None`) —
    /// FRAME_DESIGN.md §2. A frame is one of these on top of the stack.
    pub fn insert_matte(
        &self,
        id: LayerId,
        above: Option<LayerId>,
        region: MatteRegion,
        color: [f32; 3],
    ) -> Self {
        self.insert(Layer::matte(id, region, color), above)
    }

    fn insert(&self, layer: Layer, above: Option<LayerId>) -> Self {
        let at = match above {
            Some(target) => self
                .layer_index(target)
                .map_or(self.layers.len(), |i| i + 1),
            None => self.layers.len(),
        };
        // rpds::Vector has no insert-at; rebuild with the new layer spliced in.
        let mut layers = Vector::new();
        for (i, l) in self.layers.iter().enumerate() {
            if i == at {
                layers = layers.push_back(layer.clone());
            }
            layers = layers.push_back(l.clone());
        }
        if at >= self.layers.len() {
            layers = layers.push_back(layer);
        }
        self.with_layers(layers)
    }

    /// Move a matte layer's rect (the frame drag's commit). A no-op on a paint
    /// layer or an absent id.
    pub fn set_matte_rect(&self, id: LayerId, min: Vec2, max: Vec2) -> Self {
        self.map_layer(id, |l| match &l.content {
            LayerContent::Matte { region, color } => Layer {
                content: LayerContent::Matte {
                    region: region.with_rect(min, max),
                    color: *color,
                },
                ..l.clone()
            },
            LayerContent::Paint(_) => l,
        })
    }

    /// Set a matte layer's fill colour (straight sRGB). A no-op on a paint layer
    /// or an absent id.
    pub fn set_matte_color(&self, id: LayerId, color: [f32; 3]) -> Self {
        self.map_layer(id, |l| match &l.content {
            LayerContent::Matte { region, .. } => Layer {
                content: LayerContent::Matte {
                    region: *region,
                    color,
                },
                ..l.clone()
            },
            LayerContent::Paint(_) => l,
        })
    }

    /// Remove the layer with the given id (no-op if absent).
    pub fn remove_layer(&self, id: LayerId) -> Self {
        let mut layers = Vector::new();
        for l in self.layers.iter() {
            if l.id != id {
                layers = layers.push_back(l.clone());
            }
        }
        self.with_layers(layers)
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

    /// Set (or, with `None`, clear) a layer's name — no-op if absent.
    ///
    /// Takes whatever it is given: the name is normalized once where the action is
    /// minted ([`Engine::process`](crate::Engine::process)), so a replay of the log
    /// puts back exactly what was recorded rather than re-deriving it from rules
    /// that may have changed since.
    pub fn set_layer_name(&self, id: LayerId, name: Option<Arc<str>>) -> Self {
        self.map_layer(id, |l| Layer {
            name: name.clone(),
            ..l
        })
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
        self.with_layers(layers)
    }

    fn map_layer(&self, id: LayerId, f: impl FnOnce(Layer) -> Layer) -> Self {
        match self.layer_index(id) {
            Some(idx) => self.with_layer_at(idx, f(self.layer_at(idx).clone())),
            None => self.clone(),
        }
    }

    /// Rebuild from a new layer stack: bounds are recomputed from every populated
    /// tile, and the selections carry over — they are orthogonal to the layer stack
    /// (a mask applies to whatever is painted through it, §6.8).
    ///
    /// Bounds are **paint-only**: a matte covers the infinite plane, so counting
    /// it would make `bounds` unbounded and break both "frame to content" and
    /// export's no-frame fallback (FRAME_DESIGN.md §6). `Layer::tiles` is empty
    /// for a matte, so this falls out rather than needing a branch.
    ///
    /// `pub(crate)` for the timeline's patch restore (DESIGN.md §12.6), which
    /// splices layer records back into the stack.
    pub(crate) fn with_layers(&self, layers: Vector<Layer>) -> Self {
        let mut bounds = CanvasBounds::default();
        for coord in layers
            .iter()
            .filter_map(Layer::tiles)
            .flat_map(|t| t.keys())
        {
            bounds.include(*coord);
        }
        Self {
            layers,
            bounds,
            selections: self.selections.clone(),
            surface: self.surface,
            background: self.background,
        }
    }
}
