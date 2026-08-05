//! `DocState`: the versioned document state (§5.1).
//!
//! `DocState` is the `history` crate's `State`, so cloning it must be cheap: it
//! holds `rpds` persistent collections whose clone is a handful of `Arc` bumps.
//! The heavy GPU memory lives behind `TileHandle`s shared across versions, and
//! is reclaimed when the last version referencing a tile drops (§5.2).

use std::sync::Arc;

use rpds::{HashTrieMap, Vector};

use super::action::ActorId;
use super::layer::{BlendMode, Layer, LayerContent, LayerId, MatteRegion};
use super::selection::Selection;
use crate::geom::{TileCoord, Vec2};
use crate::gpu::SurfaceId;

/// Inclusive tile-coordinate bounding box of all populated tiles (§6),
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

/// Where a layer sits in the tree: whose stack it is in, and how far up that
/// stack (§14.8).
///
/// Enough to put a removed layer back exactly where it was, and no more. A full
/// index path would say the same thing in a form that goes stale the moment
/// anything below it moves; a carrier **id** does not, because ids are stable
/// and a carrier that has itself been removed is a case the restore has to
/// handle anyway.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerSite {
    /// The layer whose carried stack this one is in, or `None` for the root
    /// stack.
    pub carrier: Option<LayerId>,
    /// Position within that stack, counting from the bottom.
    pub index: usize,
}

/// The full document: a **tree** of layers (each may carry others — a group is
/// a layer with a non-empty `carries`, §14.2), the explored
/// bounds, and the selection masks that gate where tools may act.
#[derive(Clone)]
pub struct DocState {
    /// The root stack, bottom-to-top. The tree lives *inside* it: a layer's
    /// [`carries`](Layer::carries) is the group it is the base of.
    pub layers: Vector<Layer>,
    pub bounds: CanvasBounds,
    /// The active selection **of each actor** (§6.8, §17.3).
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
    /// The physical canvas surface (§6.4). Document state: which canvas
    /// a piece was painted on is part of what the document *is*, it is saved, and
    /// reopening on a different weave would be a different painting.
    ///
    /// Read by the media pass (§6.3) *and* by the deposition tooth (§6.4) — which is
    /// what having logged it bought. A switch is an ordinary action, so a stroke
    /// deposits against the ground the log stood on when it was made, on replay and
    /// on a peer alike, and no history rule had to be invented to say so.
    pub surface: SurfaceId,
    /// The canvas substrate colour — the ground the paint sits on — as straight
    /// sRGB (§15.5).
    ///
    /// Document state on the same argument §6.4 makes for the weave: which ground
    /// a piece was painted on is part of what it *is*, and it must be saved. It
    /// was previously a view setting the frontend owned, which meant the paper
    /// colour of a painting was simply not stored anywhere.
    ///
    /// Distinct from a matte layer, which is a slab of opaque *paint*: the
    /// substrate sits under everything, is lit, and the canvas weave shows through
    /// it (the media pass composites paint over it, §6.3).
    pub background: [f32; 3],
}

/// The default substrate: a light neutral grey ground. Neutral on purpose — an HDR
/// lights the scene warm, and a warm ground on top of that reads noticeably red.
/// Grey rather than near-white so paint has somewhere to go in *both* directions:
/// a highlight can read lighter than the bare canvas.
pub const DEFAULT_BACKGROUND: [f32; 3] = [0.85, 0.85, 0.85];

/// The canvas a document starts on when nobody says otherwise (§6.4): `Flat`, the
/// one ground that is procedural.
///
/// It used to be linen, and that could not survive grounds being content-addressed
/// (§6.4): a `SurfaceId` now *is* a height map, and the engine embeds no image
/// bytes, so core naming linen would be core naming an image it cannot produce. The
/// old constant papered over exactly that — a fresh document claimed to be on linen
/// while the registry rendered the flat stand-in until the frontend's fetch landed,
/// which is the same "named ground, absent bytes" gap the peer divergence came
/// through.
///
/// So the choice of opening ground belongs to whoever holds the bytes: the frontend
/// imports linen at startup and opens the document on it
/// (`Engine::new_document`), and a document that never gets told starts smooth
/// — which is at least *true*.
pub const DEFAULT_SURFACE: SurfaceId = SurfaceId::Flat;

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

    /// The same document on a different canvas surface (§6.4).
    pub fn with_surface(&self, surface: SurfaceId) -> Self {
        Self {
            surface,
            ..self.clone()
        }
    }

    /// The same document on a different substrate colour (§15.5).
    pub fn with_background(&self, background: [f32; 3]) -> Self {
        Self {
            background,
            ..self.clone()
        }
    }

    /// `actor`'s selection mask (§6.8, §17.3). An actor with
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

    /// The same document with `actor`'s selection replaced (§6.8).
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

    /// The layer with this id, wherever in the tree it sits.
    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find_map(|l| l.find(id))
    }

    /// Whether a layer with this id exists at all.
    pub fn contains_layer(&self, id: LayerId) -> bool {
        self.layer(id).is_some()
    }

    /// Every layer in **composite order** — each stack bottom-to-top, a group's
    /// base before what it carries — with its depth (0 at the root stack).
    ///
    /// The single traversal every reader shares: the UI projection, the draw
    /// list, the bounds. Compare `Layer::visit`, which is this for one subtree.
    ///
    /// The order is what answers **"does anything composite beneath this
    /// layer?"** — the one predicate behind both relational properties, since a
    /// layer's blend mode and its clip are live exactly when it holds and go
    /// inert together where it fails (§14.4.3). It fails in exactly one place:
    /// the first layer visited, the bottom of the root stack, where a mode is the
    /// identity and a clip would erase the layer. Every other layer has either a
    /// lower sibling or the content of the layer carrying it — a carrier's own
    /// content is the bottom of its group, beneath everything it carries, which
    /// is why a group's base can never be the layer without a backdrop however
    /// the group itself sits. `Engine::observe` reads it off this walk for free;
    /// a standalone predicate searched the tree per layer to say the same thing.
    pub fn visit(&self, f: &mut impl FnMut(&Layer, usize)) {
        for l in self.layers.iter() {
            l.visit(0, f);
        }
    }

    /// What the layer with the given id is called, or `None` if it is unnamed —
    /// or absent, which reads the same way here: neither has a name to give.
    pub fn layer_name(&self, id: LayerId) -> Option<&str> {
        self.layer(id).and_then(|l| l.name.as_deref())
    }

    /// Which layer carries `id`, or `None` when it sits in the root stack (or
    /// does not exist).
    pub fn carrier_of(&self, id: LayerId) -> Option<LayerId> {
        fn walk(layers: &Vector<Layer>, id: LayerId, carrier: Option<LayerId>) -> Option<LayerId> {
            for l in layers.iter() {
                if l.id == id {
                    return carrier;
                }
                if let Some(found) = walk(&l.carries, id, Some(l.id)) {
                    return Some(found);
                }
            }
            None
        }
        walk(&self.layers, id, None)
    }

    /// Where `id` sits, for a restore to put it back (§14.8).
    pub fn site_of(&self, id: LayerId) -> Option<LayerSite> {
        fn walk(
            layers: &Vector<Layer>,
            id: LayerId,
            carrier: Option<LayerId>,
        ) -> Option<LayerSite> {
            if let Some(index) = layers.iter().position(|l| l.id == id) {
                return Some(LayerSite { carrier, index });
            }
            layers.iter().find_map(|l| walk(&l.carries, id, Some(l.id)))
        }
        walk(&self.layers, id, None)
    }

    /// Insert a new empty paint layer into the stack carried by `carrier` (the
    /// root stack when `None`), directly above `above` — or on top of that stack
    /// when `above` is absent or lives somewhere else.
    pub fn insert_layer(
        &self,
        id: LayerId,
        carrier: Option<LayerId>,
        above: Option<LayerId>,
    ) -> Self {
        self.insert(Layer::new(id), carrier, above)
    }

    /// Insert a matte layer the same way — §15.2. A frame is one of
    /// these on top of the stack.
    pub fn insert_matte(
        &self,
        id: LayerId,
        carrier: Option<LayerId>,
        above: Option<LayerId>,
        region: MatteRegion,
        color: [f32; 3],
    ) -> Self {
        self.insert(Layer::matte(id, region, color), carrier, above)
    }

    fn insert(&self, layer: Layer, carrier: Option<LayerId>, above: Option<LayerId>) -> Self {
        let layers = match carrier {
            None => Some(splice(&self.layers, above, &layer)),
            // Into the carrier's own stack. An unknown carrier inserts nowhere:
            // silently landing the layer at the root instead would be a
            // different document on a client whose tree is one action behind,
            // and "nothing happened" is at least the same nothing everywhere.
            Some(c) => map_in(&self.layers, c, &mut |l: &Layer| {
                l.with_carries(splice(&l.carries, above, &layer))
            }),
        };
        match layers {
            Some(layers) => self.with_layers(layers),
            None => self.clone(),
        }
    }

    /// Copy a layer — and everything it carries — and splice the copy into the
    /// source's own stack, directly above it (§14.8).
    ///
    /// `ids` pairs every layer of the subtree with the id its copy takes, root
    /// first; see [`ActionKind::DuplicateLayer`] for why the map travels in the
    /// action rather than being minted here. The copy is the source *record*, id
    /// replaced — same tiles, same name, same opacity — because that is what a
    /// duplicate is. The name comes along rather than being decorated into
    /// "Sky copy": a name in the document is the author's own word, and inventing
    /// one here would be the engine writing a name nobody typed (see
    /// [`Layer::name`]).
    ///
    /// Nothing happens — deterministically, so peers and replays agree — when the
    /// source is absent, or when the subtree holds a layer `ids` does not name.
    ///
    /// [`ActionKind::DuplicateLayer`]: super::action::ActionKind::DuplicateLayer
    pub fn duplicate_layer(&self, ids: &[(LayerId, LayerId)]) -> Self {
        let Some((source, _)) = ids.first() else {
            return self.clone();
        };
        let Some(layer) = self.layer(*source) else {
            return self.clone();
        };
        let Some(copy) = copy_subtree(layer, ids) else {
            return self.clone();
        };
        self.insert(copy, self.carrier_of(*source), Some(*source))
    }

    /// Put `layer` back at `site` — the inverse of removing it
    /// (§14.8). It comes back carrying whatever it carried, because a `Layer` owns
    /// its subtree.
    ///
    /// A site whose carrier has since been removed falls back to the top of the
    /// root stack: the layer is restored rather than lost, which is the property
    /// undo needs, and the alternative — dropping it — would be unrecoverable.
    pub(crate) fn restore_layer(&self, site: &LayerSite, layer: Layer) -> Self {
        let at = |stack: &Vector<Layer>| site.index.min(stack.len());
        let layers = match site.carrier {
            None => Some(insert_at(&self.layers, at(&self.layers), &layer)),
            Some(c) => map_in(&self.layers, c, &mut |l: &Layer| {
                l.with_carries(insert_at(&l.carries, at(&l.carries), &layer))
            })
            .or_else(|| Some(insert_at(&self.layers, self.layers.len(), &layer))),
        };
        self.with_layers(layers.expect("restore always produces a stack"))
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
            LayerContent::Paint(_) => l.clone(),
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
            LayerContent::Paint(_) => l.clone(),
        })
    }

    /// Remove the layer with the given id **and everything it carries** (no-op
    /// if absent).
    ///
    /// The subtree goes as one because the subtree *is* the group
    /// (§14.2): removing a base and leaving what stood on it would
    /// leave layers whose blend modes and clips were written against a backdrop
    /// that no longer exists. Promoting what it carries is a different
    /// operation, and it has its own command — see [`Self::move_layer`], which
    /// is what "release" is spelled with.
    pub fn remove_layer(&self, id: LayerId) -> Self {
        match remove_in(&self.layers, id) {
            Some((layers, _)) => self.with_layers(layers),
            None => self.clone(),
        }
    }

    /// Set the blend mode of a layer (no-op if absent).
    pub fn set_layer_blend(&self, id: LayerId, blend: BlendMode) -> Self {
        self.map_layer(id, |l| Layer { blend, ..l.clone() })
    }

    /// Set whether a layer clips to the paint beneath it (no-op if absent) —
    /// §14.4.
    pub fn set_layer_clip(&self, id: LayerId, clip: bool) -> Self {
        self.map_layer(id, |l| Layer { clip, ..l.clone() })
    }

    /// Set a layer's opacity, clamped to [0, 1] (no-op if absent).
    pub fn set_layer_opacity(&self, id: LayerId, opacity: f32) -> Self {
        self.map_layer(id, |l| Layer {
            opacity: opacity.clamp(0.0, 1.0),
            ..l.clone()
        })
    }

    /// Set a layer's visibility (no-op if absent).
    pub fn set_layer_visible(&self, id: LayerId, visible: bool) -> Self {
        self.map_layer(id, |l| Layer {
            visible,
            ..l.clone()
        })
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
            ..l.clone()
        })
    }

    /// Move layer `id` into the stack carried by `carrier` (the root stack when
    /// `None`), immediately above `above` — or on top of that stack when `above`
    /// is absent or lives elsewhere. The layer keeps its tiles **and everything
    /// it carries**, so a whole group travels as one.
    ///
    /// This is the *only* structural move, and it is deliberately one operation
    /// rather than three (§14.8): reordering within a stack is
    /// `carrier` unchanged, **carrying** a layer onto another is `carrier` set,
    /// and **releasing** it is `carrier` cleared. There is nothing a "group"
    /// command would do that this does not already say.
    ///
    /// Two ways it declines, both silent and both deterministic — which is what
    /// matters, since peers replay this from a log and must all decline
    /// identically:
    ///
    /// - **A cycle.** Carrying a layer onto its own descendant (or onto itself)
    ///   would detach the subtree from the document entirely. Two peers can
    ///   concurrently ask for the two halves of one; the total order
    ///   `(lamport, actor)` means the second to apply sees the first's result and
    ///   refuses, so no tree-CRDT cycle machinery is needed (§17.9).
    /// - **An unknown carrier**, for the reason [`Self::insert`] gives.
    pub fn move_layer(
        &self,
        id: LayerId,
        carrier: Option<LayerId>,
        above: Option<LayerId>,
    ) -> Self {
        let Some(subtree) = self.layer(id) else {
            return self.clone();
        };
        // Nothing inside the moved subtree may become its carrier — including
        // the layer itself.
        if carrier.is_some_and(|c| subtree.find(c).is_some()) {
            return self.clone();
        }
        let Some((remaining, moved)) = remove_in(&self.layers, id) else {
            return self.clone();
        };
        let layers = match carrier {
            None => Some(splice(&remaining, above, &moved)),
            Some(c) => map_in(&remaining, c, &mut |l: &Layer| {
                l.with_carries(splice(&l.carries, above, &moved))
            }),
        };
        match layers {
            Some(layers) => self.with_layers(layers),
            None => self.clone(),
        }
    }

    /// Rewrite one layer in place, wherever it is in the tree (no-op if absent).
    ///
    /// `pub(crate)` and taking the layer by reference because every caller is a
    /// property setter or an `apply` arm that wants the record it is replacing.
    pub(crate) fn map_layer(&self, id: LayerId, f: impl FnOnce(&Layer) -> Layer) -> Self {
        let mut f = Some(f);
        let mut apply = |l: &Layer| (f.take().expect("map_layer runs once"))(l);
        match map_in(&self.layers, id, &mut apply) {
            Some(layers) => self.with_layers(layers),
            None => self.clone(),
        }
    }

    /// Rebuild from a new layer tree: bounds are recomputed from every populated
    /// tile **at every depth**, and the selections carry over — they are
    /// orthogonal to the layer stack (a mask applies to whatever is painted
    /// through it, §6.8).
    ///
    /// Bounds are **paint-only**: a matte covers the infinite plane, so counting
    /// it would make `bounds` unbounded and break both "frame to content" and
    /// export's no-frame fallback (§15.6). `Layer::tiles` is empty
    /// for a matte, so this falls out rather than needing a branch.
    ///
    /// `pub(crate)` for the timeline's patch restore (§12.6), which
    /// splices layer records back into the tree.
    pub(crate) fn with_layers(&self, layers: Vector<Layer>) -> Self {
        let mut bounds = CanvasBounds::default();
        fn walk(layers: &Vector<Layer>, bounds: &mut CanvasBounds) {
            for l in layers.iter() {
                for coord in l.tiles().into_iter().flat_map(|t| t.keys()) {
                    bounds.include(*coord);
                }
                walk(&l.carries, bounds);
            }
        }
        walk(&layers, &mut bounds);
        Self {
            layers,
            bounds,
            selections: self.selections.clone(),
            surface: self.surface,
            background: self.background,
        }
    }
}

// ---- Tree surgery ---------------------------------------------------------
//
// Four operations over a `Vector<Layer>` and its nested stacks, each returning
// `None` when the id it was given is not in the tree — which is what lets the
// callers above turn "no such layer" into a clean no-op rather than a panic or
// a half-applied edit. Free functions rather than methods because they recurse
// into stacks that are not a `DocState`'s root.
//
// `None` therefore means **exactly** "no such layer", and nothing else, which is
// why `Vector::set`'s own `None` is `expect`ed rather than propagated: it reports
// an out-of-range index, and every index here came from `position`/`enumerate`
// over the very stack being written. Passing it through would spell an impossible
// failure as the one case the callers act on — a removal or a rename that
// silently did not happen, on a document that says it did.
//
// `&mut dyn FnMut` rather than a generic closure: the recursion would otherwise
// instantiate a fresh copy of the function per level and never terminate at
// compile time.

/// What `Vector::set` failing would mean here — see the section note above.
const IN_RANGE: &str = "index came from this stack";

/// `layers` with the layer `id` replaced by `f` applied to it, wherever it sits.
fn map_in(
    layers: &Vector<Layer>,
    id: LayerId,
    f: &mut dyn FnMut(&Layer) -> Layer,
) -> Option<Vector<Layer>> {
    if let Some(i) = layers.iter().position(|l| l.id == id) {
        let replaced = f(layers.get(i).expect(IN_RANGE));
        return Some(layers.set(i, replaced).expect(IN_RANGE));
    }
    for (i, l) in layers.iter().enumerate() {
        if let Some(carries) = map_in(&l.carries, id, f) {
            return Some(layers.set(i, l.with_carries(carries)).expect(IN_RANGE));
        }
    }
    None
}

/// `layer` and everything it carries, re-identified through `ids` — or `None`
/// if the map does not name every layer in the subtree, which is the one way a
/// duplicate declines.
///
/// All-or-nothing on purpose: a partially re-identified subtree would share ids
/// with the layers it was copied from, and two layers under one id is the state
/// [`LayerId::mint`] exists to make impossible.
fn copy_subtree(layer: &Layer, ids: &[(LayerId, LayerId)]) -> Option<Layer> {
    let id = ids
        .iter()
        .find_map(|(src, copy)| (*src == layer.id).then_some(*copy))?;
    let mut carries = Vector::new();
    for l in layer.carries.iter() {
        carries = carries.push_back(copy_subtree(l, ids)?);
    }
    Some(Layer {
        id,
        carries,
        ..layer.clone()
    })
}

/// `layers` without the layer `id`, plus the subtree that came out of it.
fn remove_in(layers: &Vector<Layer>, id: LayerId) -> Option<(Vector<Layer>, Layer)> {
    if let Some(i) = layers.iter().position(|l| l.id == id) {
        let taken = layers.get(i).expect(IN_RANGE).clone();
        let mut out = Vector::new();
        for (j, l) in layers.iter().enumerate() {
            if j != i {
                out = out.push_back(l.clone());
            }
        }
        return Some((out, taken));
    }
    for (i, l) in layers.iter().enumerate() {
        if let Some((carries, taken)) = remove_in(&l.carries, id) {
            let layers = layers.set(i, l.with_carries(carries)).expect(IN_RANGE);
            return Some((layers, taken));
        }
    }
    None
}

/// `stack` with `layer` directly above `above`, or on top when `above` is absent
/// or is not a member of this stack.
fn splice(stack: &Vector<Layer>, above: Option<LayerId>, layer: &Layer) -> Vector<Layer> {
    let at = match above {
        Some(target) => stack
            .iter()
            .position(|l| l.id == target)
            .map_or(stack.len(), |i| i + 1),
        None => stack.len(),
    };
    insert_at(stack, at, layer)
}

/// `stack` with `layer` at `index`, counting from the bottom. `rpds::Vector` has
/// no insert-at, so this rebuilds.
fn insert_at(stack: &Vector<Layer>, index: usize, layer: &Layer) -> Vector<Layer> {
    let mut out = Vector::new();
    for (i, l) in stack.iter().enumerate() {
        if i == index {
            out = out.push_back(layer.clone());
        }
        out = out.push_back(l.clone());
    }
    if index >= stack.len() {
        out = out.push_back(layer.clone());
    }
    out
}
