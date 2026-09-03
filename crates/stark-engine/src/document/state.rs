//! `DocState`: the versioned document state (§5.1).
//!
//! `DocState` is the `history` crate's `State`, so cloning it must be cheap: it
//! holds `rpds` persistent collections whose clone is a handful of `Arc` bumps.
//! The heavy GPU memory lives behind `TileHandle`s shared across versions, and
//! is reclaimed when the last version referencing a tile drops (§5.2).

use std::sync::Arc;

use rpds::{HashTrieMap, Vector};

use super::layer::{CompositeParams, Layer, LayerContent};
use super::selection::Selection;
use stark_model::Srgb;
use stark_model::document::ActorId;
use stark_model::document::Filter;
use stark_model::document::{
    BlendMode, GuideId, LayerId, MatteRegion, Parcel, PerspectiveGuide, Place,
};
use stark_model::geom::{TileCoord, Vec2};
use stark_model::{SubstrateId, SubstrateScale};

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

    /// The extent spanning every tile in `coords` — empty for none.
    pub(crate) fn of_tiles<'a>(coords: impl Iterator<Item = &'a TileCoord>) -> Self {
        let mut out = Self::default();
        for c in coords {
            out.include(*c);
        }
        out
    }

    /// Grow to contain `other` as well.
    ///
    /// The **join** that makes a document's extent the union of its layers' own,
    /// which is what lets a layer whose tiles did not change contribute the box
    /// it already knows instead of being walked again ([`PaintTiles`]). A box is
    /// a rectangle, so containing another's two corners contains all of it.
    ///
    /// [`PaintTiles`]: super::layer::PaintTiles
    pub(crate) fn union(&mut self, other: Self) {
        if let Some((min, max)) = other.range {
            self.include(min);
            self.include(max);
        }
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

    /// The extent as it sits on the canvas once the layer's frame is placed at
    /// `by` (§14.12) — tile-granular, rounded **outward**, since a pixel offset
    /// that is not a multiple of the tile size lands a tile range astride two.
    ///
    /// Conservative in the direction bounds already are: the box is a cover, so a
    /// tile of slack costs a slightly looser "frame to content" and nothing else.
    /// Saturating, so a frame parked at the integer horizon clamps rather than
    /// wraps the box to the other side of the canvas.
    pub(crate) fn shifted(self, by: stark_model::geom::IVec2) -> Self {
        use stark_model::geom::TILE_SIZE;
        let t = TILE_SIZE as i32;
        let floor = |d: i32| d.div_euclid(t);
        let ceil = |d: i32| d.saturating_add(t - 1).div_euclid(t);
        Self {
            range: self.range.map(|(min, max)| {
                (
                    TileCoord::new(
                        min.x.saturating_add(floor(by.x)),
                        min.y.saturating_add(floor(by.y)),
                    ),
                    TileCoord::new(
                        max.x.saturating_add(ceil(by.x)),
                        max.y.saturating_add(ceil(by.y)),
                    ),
                )
            }),
        }
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
///
/// # Who may produce one
///
/// **The readers are `pub`; every mutator is `pub(crate)`.** A `DocState` is supposed
/// to be a fold of the action log and nothing else — that is the founding sentence of
/// the whole design (§1) — and while the ~two dozen setters below were public the
/// answer to "who can make one that is not" was "anything that links this crate".
///
/// Inside the crate they have exactly three callers, and each is the log: `apply`
/// (the fold), `patch` (its inverse), and the unlogged drag previews in `engine`,
/// which fold the very action their release will commit
/// (`document::apply::preview_of`). Nothing outside names one — checked against
/// `stark-dioxus-frontend`, the integration tests, the benches and `stark-net`, which
/// between them use only the readers.
#[derive(Clone)]
pub struct DocState {
    /// The root stack, bottom-to-top. The tree lives *inside* it: a layer's
    /// [`carries`](Layer::carries) is the group it is the base of.
    ///
    /// Private, with [`root`](Self::root) to read it, because [`bounds`] is
    /// derived from it: the two are one fact in two forms, and a struct literal
    /// that set the first without the second would be a document whose extent
    /// disagrees with its paint. Only this module can write either, and only
    /// [`with_layers`](Self::with_layers) writes them together.
    ///
    /// [`bounds`]: Self::bounds
    layers: Vector<Layer>,
    /// The explored extent of the infinite canvas — the union of every layer's
    /// own (§6), **derived** from `layers` and never set beside it.
    ///
    /// Worth the privacy rather than trusted to convention: this is what "frame
    /// to content" and export's no-frame fallback measure (§15.6), so a value out
    /// of step with the paint is a wrongly-cropped export, and nothing about the
    /// pixels would say so.
    bounds: CanvasBounds,
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
    /// The physical canvas substrate (§6.4). Document state: which canvas
    /// a piece was painted on is part of what the document *is*, it is saved, and
    /// reopening on a different substrate would be a different painting.
    ///
    /// Read by the media pass (§6.3) *and* by the deposition tooth (§6.4) — which is
    /// what having logged it bought. A switch is an ordinary action, so a stroke
    /// deposits against the substrate the log stood on when it was made, on replay and
    /// on a peer alike, and no history rule had to be invented to say so.
    pub substrate: SubstrateId,
    /// **How large that substrate is laid** (§6.4) — the other half of the same fact.
    ///
    /// Document state on `substrate`'s own argument, and it is not a view setting for
    /// the reason `substrate` is not: the tooth reads the substrate's rise over a reach in
    /// canvas px, so the scale decides what a stroke *deposits*, and a deposit is
    /// stored. A stroke replayed from before a change bites the substrate at the size it
    /// was laid at when it was made.
    pub substrate_scale: SubstrateScale,
    /// The canvas substrate color — the substrate the paint sits on — as straight
    /// sRGB (§15.5).
    ///
    /// Document state on the same argument §6.4 makes for the substrate: which substrate a
    /// piece was painted on is part of what it *is*, and it must be saved. A view
    /// setting the frontend owned would leave the paper color of a painting stored
    /// nowhere at all.
    ///
    /// Distinct from a matte layer, which is a slab of opaque *paint*: the
    /// substrate sits under everything, is lit, and the canvas substrate shows through
    /// it (the media pass composites paint over it, §6.3).
    pub substrate_color: Srgb,
    /// The **drawing guides** the artist has set up, in roster order (§20.5).
    ///
    /// Document state on the argument the substrate above makes, and it took a
    /// while to see it: a perspective built over a drawing is part of the
    /// drawing's construction, not a preference about how it is being looked at.
    /// Held as view state, it was lost on reload and invisible to collaborators —
    /// work thrown away by the same reasoning that would have thrown away the
    /// paper color.
    ///
    /// Whether a guide is *drawn* is the half that genuinely is per-client, and it
    /// is not here: it lives beside the pan and the zoom in `Session`, and the two
    /// are combined into one reading of the roster by `Engine::observe` (§20.5).
    ///
    /// Private where `substrate` and `background` are `pub`, for
    /// [`bounds`](Self::bounds)' reason rather than theirs: those two are a value,
    /// this is a **list with an invariant** — no two entries share an id — and the
    /// insert that holds it is the only way in.
    guides: Vector<Guide>,
}

/// One guide in the roster: the camera, and what the artist calls it (§20.5).
///
/// The engine's half of the `GuideId`/`PerspectiveGuide` pair, and the whole of
/// what it adds is the name — held as an `Arc<str>` for the reason
/// [`Layer::name`](super::layer::Layer) is one, where the action carrying it
/// holds a `String` because that is what goes on the wire.
///
/// A struct rather than the `(GuideId, Option<Arc<str>>, PerspectiveGuide)` it
/// would otherwise be, on `Ellipse`'s argument: a tuple of three says nothing
/// about which of them is which.
#[derive(Clone, Debug, PartialEq)]
pub struct Guide {
    pub id: GuideId,
    /// What the artist calls this guide, or `None` for one never named — the
    /// panel then shows its *place* in the roster ("Perspective 2"), which is a
    /// description rather than a name and is why it is not stored as one (the
    /// same distinction [`Layer::name`](super::layer::Layer) draws).
    ///
    /// Normalized on the way in by the engine's command handler, the same funnel
    /// a layer's name passes through (`normalize_name`), so "either absent or
    /// something you can read" is a property of the document rather than a habit
    /// of whichever frontend collected it.
    pub name: Option<Arc<str>>,
    /// The camera, and how densely to dress it — everything §20 derives from.
    pub camera: PerspectiveGuide,
}

/// The default substrate: a light neutral grey substrate. Neutral on purpose — an HDR
/// lights the scene warm, and a warm substrate on top of that reads noticeably red.
/// Grey rather than near-white so paint has somewhere to go in *both* directions:
/// a highlight can read lighter than the bare canvas.
pub const DEFAULT_SUBSTRATE_COLOR: Srgb = Srgb::new([0.85, 0.85, 0.85]);

/// The canvas a document starts on when nobody says otherwise (§6.4): `Flat`, the
/// one substrate that is procedural.
///
/// **Core cannot name a woven substrate here.** Substrates are content-addressed (§6.4) — a
/// `SubstrateId` *is* a height map — and the engine embeds no image bytes, so a default
/// of linen would be core naming an image it cannot produce. A fresh document would
/// claim to be on linen while the registry rendered the flat stand-in until the
/// frontend's fetch landed: the "named substrate, absent bytes" gap that diverges peers.
///
/// So the choice of opening substrate belongs to whoever holds the bytes: the frontend
/// imports linen at startup and opens the document on it
/// (`Engine::new_document`), and a document that never gets told starts smooth
/// — which is at least *true*.
pub const DEFAULT_SUBSTRATE: SubstrateId = SubstrateId::Flat;

impl DocState {
    /// An empty document with a single starting layer and nothing masked.
    ///
    /// Built by *inserting* that layer rather than by naming the pair directly,
    /// so [`with_layers`](Self::with_layers) stays the one place `bounds` is ever
    /// written. The literal below is the only `DocState` with no layers at all,
    /// and its empty extent is the one that needs no deriving.
    pub fn with_layer(id: LayerId) -> Self {
        Self {
            layers: Vector::new(),
            bounds: CanvasBounds::default(),
            selections: HashTrieMap::new(),
            substrate: DEFAULT_SUBSTRATE,
            substrate_scale: SubstrateScale::NATURAL,
            substrate_color: DEFAULT_SUBSTRATE_COLOR,
            guides: Vector::new(),
        }
        .insert_layer(id, None, None)
    }

    /// The root stack, bottom-to-top — the tree's top level, each layer carrying
    /// its own subtree. The compositor walks it to build the draw list.
    ///
    /// Compare [`visit`](Self::visit), which flattens the whole tree in composite
    /// order and is what most readers want.
    pub fn root(&self) -> &Vector<Layer> {
        &self.layers
    }

    /// The explored extent of the canvas: the union of every layer's painted
    /// tiles, at every depth (§6). Empty for a document with no paint — and for
    /// one holding nothing but mattes, which cover the infinite plane and so
    /// contribute no extent at all (§15.6).
    pub fn bounds(&self) -> CanvasBounds {
        self.bounds
    }

    /// The same document on a different canvas substrate (§6.4).
    pub(crate) fn with_substrate(&self, substrate: SubstrateId) -> Self {
        Self {
            substrate,
            ..self.clone()
        }
    }

    /// **The canvas substrate as the renderer asks for it** (§6.4): the substrate and the
    /// size it is laid at, together.
    ///
    /// The two are one fact and are read as one everywhere downstream — the deposit,
    /// the media pass, the registry's key — so putting them together happens here and
    /// not at each of them. A `substrate` read without its scale beside it is the shape
    /// of the bug this method exists to make hard to write.
    pub fn substrate(&self) -> crate::gpu::substrate::Substrate {
        crate::gpu::substrate::Substrate {
            id: self.substrate,
            scale: self.substrate_scale,
        }
    }

    /// The same document with its substrate laid at a different size (§6.4).
    pub(crate) fn with_substrate_scale(&self, substrate_scale: SubstrateScale) -> Self {
        Self {
            substrate_scale,
            ..self.clone()
        }
    }

    /// The same document on a different substrate color (§15.5).
    pub(crate) fn with_substrate_color(&self, substrate_color: Srgb) -> Self {
        Self {
            substrate_color,
            ..self.clone()
        }
    }

    // --- the drawing-guide roster (§20.5) -----------------------------------

    /// The guides the artist has set up, in roster order.
    ///
    /// Order changes no pixel — every guide a client draws is drawn, whichever
    /// row it sits on — so this is a `Vector` for the arrangement's sake alone,
    /// which is the artist's own and is kept for the reason a layer's name is.
    /// Lookup is a scan, and deliberately: a roster is a handful of entries,
    /// where a map would cost the order this exists to hold.
    pub fn guides(&self) -> &Vector<Guide> {
        &self.guides
    }

    /// The guide with this id, or `None` if it is not there — which is what a
    /// concurrent removal looks like from every action that names one.
    pub fn guide(&self, id: GuideId) -> Option<&Guide> {
        self.guides.iter().find(|g| g.id == id)
    }

    /// The roster with a new guide directly after `after`, or at its head when
    /// nothing is named (§20.5).
    ///
    /// An anchor that is not there lands the guide at the head — deterministic,
    /// taken the same way by every peer, and the same shape [`Place::Above`]
    /// takes for a sibling that has gone.
    pub(crate) fn insert_guide(
        &self,
        id: GuideId,
        camera: PerspectiveGuide,
        after: Option<GuideId>,
        name: Option<Arc<str>>,
    ) -> Self {
        // An id already in the roster is refused rather than doubled. It cannot
        // arise — a `GuideId` *is* an action's id and an action is applied once —
        // but the roster is a list rather than a map, so nothing about the
        // representation says so, and a lookup answering the first of two would
        // be the quiet kind of divergence.
        if self.guide(id).is_some() {
            return self.clone();
        }
        let at = slot_after(&self.guides, after);
        Self {
            guides: insert_at(&self.guides, at, &Guide { id, name, camera }),
            ..self.clone()
        }
    }

    /// The roster without `id`. A no-op when it is already gone.
    pub(crate) fn remove_guide(&self, id: GuideId) -> Self {
        let Some(i) = self.guides.iter().position(|g| g.id == id) else {
            return self.clone();
        };
        Self {
            guides: remove_at(&self.guides, i),
            ..self.clone()
        }
    }

    /// Reshape a guide — the whole camera at once (§20.5). A no-op on a guide
    /// that is not there.
    pub(crate) fn set_guide(&self, id: GuideId, camera: PerspectiveGuide) -> Self {
        self.map_guide(id, |g| Guide {
            camera,
            ..g.clone()
        })
    }

    /// Name a guide, or with `None` take its name away so it goes back to being
    /// described by its place in the roster.
    pub(crate) fn set_guide_name(&self, id: GuideId, name: Option<Arc<str>>) -> Self {
        self.map_guide(id, |g| Guide {
            name: name.clone(),
            ..g.clone()
        })
    }

    /// Move a guide so it sits directly after `after`, or at the roster's head.
    ///
    /// Taken out and put back, **in that order**, which is the only reading under
    /// which `after` is an anchor in the roster the drag was drawn against: the
    /// slot is re-derived against the shortened list, so an anchor above the row
    /// that left keeps the place it names rather than one past it.
    pub(crate) fn move_guide(&self, id: GuideId, after: Option<GuideId>) -> Self {
        let Some(from) = self.guides.iter().position(|g| g.id == id) else {
            return self.clone();
        };
        let moved = self.guides.get(from).expect(IN_RANGE).clone();
        let rest = remove_at(&self.guides, from);
        let to = slot_after(&rest, after);
        Self {
            guides: insert_at(&rest, to, &moved),
            ..self.clone()
        }
    }

    /// The whole roster, replaced — what a patch puts back when an undo crosses a
    /// guide edit. [`Resource::Guides`] is one resource, so a patch restores one
    /// thing (§12.6).
    ///
    /// [`Resource::Guides`]: stark_model::document::Resource::Guides
    pub(crate) fn with_guides(&self, guides: Vector<Guide>) -> Self {
        Self {
            guides,
            ..self.clone()
        }
    }

    /// Rewrite one guide in place, leaving the roster's order alone. A guide that
    /// is not there is left alone too, which is how every action naming one
    /// declines.
    fn map_guide(&self, id: GuideId, f: impl FnOnce(&Guide) -> Guide) -> Self {
        let Some(i) = self.guides.iter().position(|g| g.id == id) else {
            return self.clone();
        };
        let guide = f(self.guides.get(i).expect(IN_RANGE));
        Self {
            guides: self.guides.set(i, guide).expect(IN_RANGE),
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

    /// Every actor with a selection stored, in no particular order — one in force,
    /// or a universal mask read below full strength; never
    /// [`Selection::everything`] itself ([`with_selection`](Self::with_selection)).
    pub fn selections(&self) -> impl Iterator<Item = (ActorId, &Selection)> {
        self.selections.iter().map(|(a, s)| (*a, s))
    }

    /// The same document with `actor`'s selection replaced (§6.8).
    ///
    /// [`Selection::everything`] is *removed* rather than stored, so "no selection"
    /// has exactly one representation and an actor who deselects stops costing
    /// anything again. A universal mask read below full strength *is* stored: it
    /// masks nothing — [`has_selection`](Self::has_selection) still says no — but
    /// its number is the strength the next region will take, and the whole
    /// canvas's until then (`Selection::opacity`).
    pub(crate) fn with_selection(&self, actor: ActorId, selection: Selection) -> Self {
        let selections = if selection.is_everything() {
            self.selections.remove(&actor)
        } else {
            self.selections.insert(actor, selection)
        };
        Self {
            selections,
            ..self.clone()
        }
    }

    /// The same document with `actor`'s mask read at `opacity` (§6.8) — no tile
    /// moves, which is what lets the slider reach a region already drawn. With
    /// nothing selected it is the strength the next region will take, and the
    /// whole canvas's until one is drawn.
    pub(crate) fn with_selection_opacity(&self, actor: ActorId, opacity: f32) -> Self {
        self.with_selection(actor, self.selection_of(actor).with_opacity(opacity))
    }

    /// Find `id` anywhere in the tree: the layer record, and where it sits
    /// (§14.8).
    ///
    /// **The one search this module does.** [`layer`](Self::layer),
    /// [`site_of`](Self::site_of) and [`carrier_of`](Self::carrier_of) are
    /// projections of it, and the two halves genuinely travel together — a
    /// duplicate needs the record to copy *and* the place to put the copy beside,
    /// and asking separately is two walks of one tree to one layer.
    fn locate(&self, id: LayerId) -> Option<(&Layer, LayerSite)> {
        fn walk(
            layers: &Vector<Layer>,
            id: LayerId,
            carrier: Option<LayerId>,
        ) -> Option<(&Layer, LayerSite)> {
            if let Some(index) = layers.iter().position(|l| l.id == id) {
                let layer = layers.get(index).expect(IN_RANGE);
                return Some((layer, LayerSite { carrier, index }));
            }
            layers.iter().find_map(|l| walk(&l.carries, id, Some(l.id)))
        }
        walk(&self.layers, id, None)
    }

    /// The layer with this id, wherever in the tree it sits.
    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.locate(id).map(|(layer, _)| layer)
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
    pub fn visit<'a>(&'a self, f: &mut impl FnMut(&'a Layer, usize)) {
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
    /// does not exist). The two read the same way here: neither is inside
    /// anything.
    pub fn carrier_of(&self, id: LayerId) -> Option<LayerId> {
        self.locate(id).and_then(|(_, site)| site.carrier)
    }

    /// Where `id` sits, for a restore to put it back (§14.8).
    pub fn site_of(&self, id: LayerId) -> Option<LayerSite> {
        self.locate(id).map(|(_, site)| site)
    }

    /// Whether `carrier` names a layer that may not carry: a **filter** (§21.2).
    ///
    /// A group's members composite *over* its base, and a filter rewrites only what
    /// is *beneath* it in its own stack — so a filter that carried layers could
    /// never reach them, and every consumer (the draw list, the projection, the
    /// panel) would need a rule for an arrangement that means nothing. The rule is
    /// ruled out as a class instead: the state refuses to hold it, here and in
    /// [`Self::move_layer`], so no action can create it and nothing downstream has
    /// to ask.
    fn cannot_carry(&self, carrier: Option<LayerId>) -> bool {
        carrier.is_some_and(|c| self.layer(c).is_some_and(|l| l.filter().is_some()))
    }

    /// Insert a new empty paint layer into the stack carried by `carrier` (the
    /// root stack when `None`), directly above `above` — or on top of that stack
    /// when `above` is absent or lives somewhere else.
    pub(crate) fn insert_layer(
        &self,
        id: LayerId,
        carrier: Option<LayerId>,
        above: Option<LayerId>,
    ) -> Self {
        self.insert(Layer::new(id), carrier, Place::from(above))
    }

    /// Insert a matte layer the same way — §15.2. A frame is one of
    /// these on top of the stack; a substrate ([`MatteRegion::Everything`]) is one
    /// at the bottom, which is why this takes the full [`Place`] where the
    /// other inserts keep the two-state anchor (§15.5).
    ///
    /// **Declined, deterministically, for a region nobody can measure**
    /// ([`MatteRegion::usable`]) — the frame's half of the gate an unusable affine
    /// already passes through (§16.1). A rect cannot be clamped into a repaired
    /// rect without reframing the piece, so it is refused instead, and every peer
    /// and every replay refuses the same one. The paint beside it *is* clamped, by
    /// `ActionKind::sanitized` on the way in, because a color out of range does have
    /// a nearest legal value.
    pub(crate) fn insert_matte(
        &self,
        id: LayerId,
        carrier: Option<LayerId>,
        at: Place,
        region: MatteRegion,
        paint: Parcel,
    ) -> Self {
        if !region.usable() {
            return self.clone();
        }
        self.insert(Layer::matte(id, region, paint), carrier, at)
    }

    /// Insert a **filter** layer the same way — §21. It rewrites whatever is
    /// composited beneath it in the stack it lands in.
    ///
    /// Sanitized on the way in, as [`set_filter`](Self::set_filter) is — see there.
    pub(crate) fn insert_filter(
        &self,
        id: LayerId,
        carrier: Option<LayerId>,
        above: Option<LayerId>,
        filter: Filter,
    ) -> Self {
        self.insert(
            Layer::filter_layer(id, filter.sanitized()),
            carrier,
            Place::from(above),
        )
    }

    /// Land a floated child at the **foot** of its source's carried stack
    /// (§16.12) — directly over the paint it was cut from, which is what keeps
    /// the picture unchanged (§14.1). The child arrives whole: tiles, frame and
    /// identity params are the fold's to build, so this takes the record rather
    /// than re-deriving it from an id.
    pub(crate) fn insert_float(&self, source: LayerId, child: Layer) -> Self {
        self.insert(child, Some(source), Place::Bottom)
    }

    fn insert(&self, layer: Layer, carrier: Option<LayerId>, above: Place) -> Self {
        // **An id the document already holds is refused rather than doubled.** It
        // cannot arise from an action this engine mints — a `LayerId` *is* an action's
        // id and an action is applied once (§17.9) — but the tree is a `Vector` rather
        // than a map, so nothing about the representation says so, and the ids arrive
        // in the payload of an action that may have come off a file or a wire. Two
        // layers under one id is the state §17.9 is about: `locate`, `map_in`,
        // `remove_in` and `Layer::find` would each answer with whichever comes first,
        // which is the quiet kind of divergence. The same refusal `insert_guide`
        // makes, for the same reason and against the same shape.
        if self.contains_layer(layer.id) {
            return self.clone();
        }
        // A filter never carries (§21.2): what a group carries composites *over*
        // its base, so a filter base could reach none of it — declined here, in
        // state, so replay and peers agree and no path can build the arrangement.
        if self.cannot_carry(carrier) {
            return self.clone();
        }
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
    /// [`ActionKind::DuplicateLayer`]: stark_model::document::ActionKind::DuplicateLayer
    pub(crate) fn duplicate_layer(&self, ids: &[(LayerId, LayerId)]) -> Self {
        let Some((source, _)) = ids.first() else {
            return self.clone();
        };
        // **Every copy id has to be new and distinct.** `copy_subtree` below looks
        // each source up and takes the id it is paired with; it has no way to notice
        // that two sources were paired with one id, or that a copy names a layer the
        // document already holds. An action this engine mints cannot say either —
        // `commit_minting` draws them from its own action id at distinct `k` — but
        // this one arrived in a payload, off a file or a wire, and two layers under
        // one id is exactly what §17.9 is about. Declined deterministically, so peers
        // agree about the refusal, which is the bargain every other decline here makes.
        let copies: Vec<LayerId> = ids.iter().map(|(_, copy)| *copy).collect();
        let fresh = copies
            .iter()
            .enumerate()
            .all(|(i, c)| !self.contains_layer(*c) && !copies[..i].contains(c));
        if !fresh {
            return self.clone();
        }
        // The record to copy and the stack to copy it into, from one search.
        let Some((layer, site)) = self.locate(*source) else {
            return self.clone();
        };
        let Some(copy) = copy_subtree(layer, ids) else {
            return self.clone();
        };
        self.insert(copy, site.carrier, Place::Above(*source))
    }

    /// Put `layer` back at `site` — the inverse of removing it
    /// (§14.8). It comes back carrying whatever it carried, because a `Layer` owns
    /// its subtree.
    ///
    /// A site whose carrier has since been removed falls back to the top of the
    /// root stack: the layer is restored rather than lost, which is the property
    /// undo needs, and the alternative — dropping it — would be unrecoverable.
    ///
    /// **Already present is already restored.** The same refusal [`Self::insert`] and
    /// [`Self::insert_guide`] make, against the same shape — and reachable here
    /// rather than only from a crafted log: undoing a group's removal captures one
    /// `Resource::Layer` per layer in the subtree, and the group's own op puts the
    /// whole subtree back before its children's ops are read (`patch::capture`). Each
    /// of those would then insert a layer the ancestor had already returned. The
    /// duplicate was collapsed downstream — `PatchOp::Structure`'s rebuild is keyed by
    /// a map — so it never survived a whole patch; it existed in between, where
    /// `map_in`, `remove_in` and `Layer::find` all answer with whichever copy comes
    /// first. Declined here instead, so the arrangement cannot be built at all.
    pub(crate) fn restore_layer(&self, site: &LayerSite, layer: Layer) -> Self {
        if self.contains_layer(layer.id) {
            return self.clone();
        }
        let at = |stack: &Vector<Layer>| site.index.min(stack.len());
        let layers = match site.carrier {
            None => Some(insert_at(&self.layers, at(&self.layers), &layer)),
            // A site whose carrier has become a filter (a crafted log — no action
            // this engine mints can produce one, see `cannot_carry`) falls back
            // exactly as a removed carrier does: restored rather than lost.
            Some(_) if self.cannot_carry(site.carrier) => {
                Some(insert_at(&self.layers, self.layers.len(), &layer))
            }
            Some(c) => map_in(&self.layers, c, &mut |l: &Layer| {
                l.with_carries(insert_at(&l.carries, at(&l.carries), &layer))
            })
            .or_else(|| Some(insert_at(&self.layers, self.layers.len(), &layer))),
        };
        self.with_layers(layers.expect("restore always produces a stack"))
    }

    /// Rewrite a layer's **content**, where `f` accepts it — the shape every content
    /// setter below takes.
    ///
    /// `f` answers `None` for content this edit does not apply to, which is what
    /// makes "which kinds accept this edit" a *parameter* rather than a pattern
    /// repeated per setter. Four of them spelled out the same match arm, the same
    /// `Layer { content, ..l.clone() }`, and the same "every other kind is left
    /// alone" — so a new [`LayerContent`] variant had to be visited four times to
    /// say the same nothing.
    ///
    /// Deliberately narrower than [`map_layer`](Self::map_layer): a setter that can
    /// only change `content` cannot reach `composite`, `visible` or `name` by
    /// accident. `set_layer_blend` and its neighbours are *not* written through this,
    /// and that is the distinction rather than an omission — they write
    /// `composite`, and read `content` only to refuse (§21.4).
    fn map_content(
        &self,
        id: LayerId,
        f: impl FnOnce(&LayerContent) -> Option<LayerContent>,
    ) -> Self {
        self.map_layer(id, |l| match f(&l.content) {
            Some(content) => Layer {
                content,
                ..l.clone()
            },
            None => l.clone(),
        })
    }

    /// Move a matte layer's rect (the frame drag's commit). A no-op on a paint
    /// layer, an absent id — or a region that has no rect to move
    /// ([`MatteRegion::with_rect`]).
    pub(crate) fn set_matte_rect(&self, id: LayerId, min: Vec2, max: Vec2) -> Self {
        self.map_content(id, |c| match c {
            // Refused rather than repaired when the rect cannot be measured, for
            // [`insert_matte`](Self::insert_matte)'s reason — spelled as the same
            // `None` this arm already returns for a layer that is not a matte, since
            // both are "this action names something that is not there".
            LayerContent::Matte { region, paint } => {
                let region = region.with_rect(min, max);
                region.usable().then(|| LayerContent::Matte {
                    region,
                    paint: paint.clone(),
                })
            }
            LayerContent::Paint(_) | LayerContent::Filter(_) => None,
        })
    }

    /// Replace a matte layer's region wholesale — undo's restore path
    /// (`patch::PatchOp::Matte`), which must put back the
    /// *value* rather than route through a rect the region may not have. A
    /// no-op on a paint layer or an absent id, like every setter here.
    pub(crate) fn set_matte_region(&self, id: LayerId, region: MatteRegion) -> Self {
        self.map_content(id, |c| match c {
            LayerContent::Matte { paint, .. } => Some(LayerContent::Matte {
                region,
                paint: paint.clone(),
            }),
            LayerContent::Paint(_) | LayerContent::Filter(_) => None,
        })
    }

    /// Set a matte layer's paint — a flat color or a gradient ramp (§15.4,
    /// §22.4). A no-op on a paint layer or an absent id.
    pub(crate) fn set_matte_paint(&self, id: LayerId, paint: Parcel) -> Self {
        self.map_content(id, |c| match c {
            LayerContent::Matte { region, .. } => Some(LayerContent::Matte {
                region: *region,
                paint: paint.clone(),
            }),
            LayerContent::Paint(_) | LayerContent::Filter(_) => None,
        })
    }

    /// Retune a filter layer — §21. A no-op on anything that is not one, and on an
    /// absent id, like every other content setter here.
    ///
    /// The whole filter travels rather than one knob, on the argument
    /// [`set_guide`](Self::set_guide) makes for a guide's whole camera (§20.5): the
    /// frontend reads the current settings off the projection, moves one of them, and
    /// sends the value back, so a filter that grows a parameter needs no command of
    /// its own — and neither does the next filter kind.
    ///
    /// **Sanitized here**, where every filter enters state — not only where a local
    /// command is minted (§21.5). A loaded file's replay and a remote peer's action
    /// reach this through `ActionKind::apply` without ever passing
    /// `Engine::process`, and a fullscreen pass has no coverage to hide behind: a
    /// NaN smuggled in by a crafted or corrupted log would poison every texel of
    /// the frame. Idempotent for any log this engine wrote, so replay still puts
    /// back exactly what was applied.
    pub(crate) fn set_filter(&self, id: LayerId, filter: Filter) -> Self {
        self.map_content(id, |c| match c {
            LayerContent::Filter(_) => Some(LayerContent::Filter(filter.sanitized())),
            LayerContent::Paint(_) | LayerContent::Matte { .. } => None,
        })
    }

    /// `id` and everything it carries, at any depth, in composite order — root
    /// first — or `None` when there is no such layer.
    ///
    /// **The one derivation of "what a group is"**, and both of the actions that name
    /// a whole subtree are minted from it. A removal takes
    /// [`Self::carried_ids`] (this without the root) and `apply` checks the action
    /// against the same walk before folding; a duplication takes this one, and
    /// `duplicate_layer` declines unless `ids` names exactly the subtree *it* walks.
    /// Either way two walks that had to agree would be the §12.6 hazard, one level
    /// down — and the duplicate path had its own copy in `engine.rs` until now.
    pub fn subtree_ids(&self, id: LayerId) -> Option<Vec<LayerId>> {
        let layer = self.layer(id)?;
        let mut out = Vec::new();
        layer.visit(0, &mut |l, _| out.push(l.id));
        Some(out)
    }

    /// The layers `id` carries, at any depth, in composite order — or `None` when
    /// there is no such layer. Empty for a leaf.
    ///
    /// [`Self::subtree_ids`] without the root, which is the form
    /// [`ActionKind::RemoveLayer`]'s `carried` is minted in.
    ///
    /// [`ActionKind::RemoveLayer`]: stark_model::document::ActionKind::RemoveLayer
    pub fn carried_ids(&self, id: LayerId) -> Option<Vec<LayerId>> {
        let mut ids = self.subtree_ids(id)?;
        ids.remove(0);
        Some(ids)
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
    ///
    /// Unconditional, and it has three callers that want it that way: the undo
    /// patch putting a layer back to absent, the merge folding a source away, and
    /// `apply`'s arm — which asks [`carried_ids`](Self::carried_ids) *first*,
    /// because whether the action may remove this subtree is a question about the
    /// action's footprint rather than about the tree surgery.
    pub(crate) fn remove_layer(&self, id: LayerId) -> Self {
        match remove_in(&self.layers, id) {
            Some((layers, _)) => self.with_layers(layers),
            None => self.clone(),
        }
    }

    /// Set the blend mode of a layer (no-op if absent).
    ///
    /// A no-op on a **filter** too, like paint on a matte (§15.7): a filter has no
    /// source to meet a backdrop, and — since a filter never carries (§21.2) — no
    /// group whose merge the mode could describe either. Refusing here rather than
    /// in a frontend rule is what keeps a stored-but-unreadable mode out of every
    /// replayed and replicated document.
    ///
    /// **Not** the rule its neighbour [`Self::set_layer_clip`] follows, and the two
    /// docs are worth reading together: a clip asks a question a filter can answer
    /// (§21.4.1), so only the mode is turned away here.
    ///
    /// The mode is **sanitized on the way in**, as a filter is (§21.5): a mode now
    /// carries numbers of its own, and a file or a peer reaches this without passing
    /// through `Engine::process`.
    pub(crate) fn set_layer_blend(&self, id: LayerId, blend: BlendMode) -> Self {
        let blend = blend.sanitized();
        self.map_layer(id, |l| match &l.content {
            LayerContent::Filter(_) => l.clone(),
            LayerContent::Paint(_) | LayerContent::Matte { .. } => Layer {
                composite: CompositeParams {
                    blend,
                    ..l.composite
                },
                ..l.clone()
            },
        })
    }

    /// Set whether a layer clips to the paint beneath it (no-op if absent) —
    /// §14.4.
    ///
    /// **Live on a filter**, unlike [`Self::set_layer_blend`] beside it, and the
    /// asymmetry is the point (§21.4). A mode says how a *source* meets a backdrop
    /// and a filter has no source, so there is nothing for one to describe. A clip
    /// says where the layer is allowed to land, and that question survives having no
    /// source: a filter is confined to the coverage it read, so it may say what color
    /// the paint already there should be and never where there is paint. For a point
    /// filter that is what it does anyway — the flag is a bit-exact no-op — and for a
    /// filter that *gathers* (§21.10) it is the difference between a fringe held
    /// inside the silhouette and one spilling past it.
    pub(crate) fn set_layer_clip(&self, id: LayerId, clip: bool) -> Self {
        self.map_layer(id, |l| Layer {
            composite: CompositeParams {
                clip,
                ..l.composite
            },
            ..l.clone()
        })
    }

    /// Set a layer's opacity, clamped to [0, 1] (no-op if absent).
    pub(crate) fn set_layer_opacity(&self, id: LayerId, opacity: f32) -> Self {
        self.map_layer(id, |l| Layer {
            composite: CompositeParams {
                opacity: opacity.clamp(0.0, 1.0),
                ..l.composite
            },
            ..l.clone()
        })
    }

    /// Put each named layer's frame at a place on the canvas (§14.12) — absent
    /// entries are skipped, like every property write on a layer that is not
    /// there, and each entry stands alone: a concurrent removal of one member
    /// does not strand the rest of a group mid-move.
    ///
    /// A no-op on a **filter**, like a stroke aimed at one (§21.4): it has
    /// nothing that sits anywhere. A matte moves like paint — its rect and
    /// gradient axis are stated in the layer's frame (§15.2), so the write
    /// carries them with it. Refused here, in the fold
    /// ([`Layer::is_translatable`]), so a log that contains one reads the same
    /// on every peer.
    pub(crate) fn translate_layers(&self, moves: &[(LayerId, stark_model::geom::IVec2)]) -> Self {
        let mut state = self.clone();
        for (id, to) in moves {
            state = state.map_layer(*id, |l| {
                if l.is_translatable() {
                    Layer {
                        translation: *to,
                        ..l.clone()
                    }
                } else {
                    l.clone()
                }
            });
        }
        state
    }

    /// Set a layer's visibility (no-op if absent).
    pub(crate) fn set_layer_visible(&self, id: LayerId, visible: bool) -> Self {
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
    pub(crate) fn set_layer_name(&self, id: LayerId, name: Option<Arc<str>>) -> Self {
        self.map_layer(id, |l| Layer {
            name: name.clone(),
            ..l.clone()
        })
    }

    /// Move layer `id` into the stack carried by `carrier` (the root stack when
    /// `None`), at `at` — above a named sibling, on top, or at the foot. The layer
    /// keeps its tiles **and everything it carries**, so a whole group travels as
    /// one.
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
    pub(crate) fn move_layer(&self, id: LayerId, carrier: Option<LayerId>, at: Place) -> Self {
        // Declined before anything is taken apart: a filter never carries (§21.2)
        // — see `insert`, which refuses the same carrier for the same reason.
        if self.cannot_carry(carrier) {
            return self.clone();
        }
        // Taken out first, so the cycle check below reads the subtree the
        // removal already had to find — rather than searching for it again, and
        // then searching for it again to remove it.
        let Some((remaining, moved)) = remove_in(&self.layers, id) else {
            return self.clone();
        };
        // Nothing inside the moved subtree may become its carrier — including
        // the layer itself. Declining here leaves `self` untouched: the removal
        // above built a new stack rather than mutating this one.
        if carrier.is_some_and(|c| moved.find(c).is_some()) {
            return self.clone();
        }
        let layers = match carrier {
            None => Some(splice(&remaining, at, &moved)),
            Some(c) => map_in(&remaining, c, &mut |l: &Layer| {
                l.with_carries(splice(&l.carries, at, &moved))
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

    /// Rebuild from a new layer tree: bounds are the **union of every layer's
    /// own extent at every depth**, and the selections carry over — they are
    /// orthogonal to the layer stack (a mask applies to whatever is painted
    /// through it, §6.8).
    ///
    /// A union of boxes each layer already knows ([`PaintTiles`]), not a walk of
    /// every populated tile in the document. That is what keeps this off the
    /// per-action cost curve: this runs on *every* layer mutation, including the
    /// property setters and structural moves that cannot change a tile set at
    /// all, and re-deriving the whole extent from tiles made a rename cost what a
    /// stroke costs — and a replay of `n` actions cost `n ×` the document.
    /// Walking the tree is unavoidable (a layer can be anywhere in it), but that
    /// is `O(layers)`, which is dozens, where the tiles are thousands.
    ///
    /// Bounds are **paint-only**: a matte covers the infinite plane, so counting
    /// it would make `bounds` unbounded and break both "frame to content" and
    /// export's no-frame fallback (§15.6). A matte has no tiles and so no extent,
    /// so this falls out rather than needing a branch.
    ///
    /// `pub(crate)` for the timeline's patch restore (§12.6), which
    /// splices layer records back into the tree.
    ///
    /// [`PaintTiles`]: super::layer::PaintTiles
    pub(crate) fn with_layers(&self, layers: Vector<Layer>) -> Self {
        let mut bounds = CanvasBounds::default();
        fn walk(layers: &Vector<Layer>, bounds: &mut CanvasBounds) {
            for l in layers.iter() {
                // Where the tiles sit on the *canvas*: the layer's own extent,
                // placed by its frame (§14.12). Each layer's own, flat — see
                // `Layer::translation` for why nothing accumulates down the tree.
                bounds.union(l.bounds().shifted(l.translation));
                walk(&l.carries, bounds);
            }
        }
        walk(&layers, &mut bounds);
        Self {
            layers,
            bounds,
            selections: self.selections.clone(),
            substrate: self.substrate,
            substrate_scale: self.substrate_scale,
            substrate_color: self.substrate_color,
            guides: self.guides.clone(),
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
/// [`LayerId`]'s shape exists to make impossible (§17.9).
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
        return Some((remove_at(layers, i), taken));
    }
    for (i, l) in layers.iter().enumerate() {
        if let Some((carries, taken)) = remove_in(&l.carries, id) {
            let layers = layers.set(i, l.with_carries(carries)).expect(IN_RANGE);
            return Some((layers, taken));
        }
    }
    None
}

/// `stack` with `layer` at `place`: directly above a named sibling, at the foot,
/// or on top — which is also where a sibling this stack does not contain lands.
fn splice(stack: &Vector<Layer>, place: Place, layer: &Layer) -> Vector<Layer> {
    let at = match place {
        Place::Above(target) => stack
            .iter()
            .position(|l| l.id == target)
            .map_or(stack.len(), |i| i + 1),
        Place::Top => stack.len(),
        Place::Bottom => 0,
    };
    insert_at(stack, at, layer)
}

/// `v` with `item` at `index`, counting from the bottom. `rpds::Vector` has no
/// insert-at, so this rebuilds.
///
/// Generic because the argument is about the collection and not about what is in
/// it: a layer stack and the guide roster (§20.5) are both ordered lists whose
/// only structural edit is putting one entry somewhere.
fn insert_at<T: Clone>(v: &Vector<T>, index: usize, item: &T) -> Vector<T> {
    let mut out = Vector::new();
    for (i, l) in v.iter().enumerate() {
        if i == index {
            out = out.push_back(item.clone());
        }
        out = out.push_back(l.clone());
    }
    if index >= v.len() {
        out = out.push_back(item.clone());
    }
    out
}

/// `v` without the entry at `index` — [`insert_at`]'s other half, and rebuilding
/// for its reason.
fn remove_at<T: Clone>(v: &Vector<T>, index: usize) -> Vector<T> {
    let mut out = Vector::new();
    for (i, l) in v.iter().enumerate() {
        if i != index {
            out = out.push_back(l.clone());
        }
    }
    out
}

/// The index the anchor `after` names in `roster`: one past the guide it names,
/// or the head when it names nothing — or names a guide that is no longer there,
/// which every peer reads the same way (§20.5).
fn slot_after(roster: &Vector<Guide>, after: Option<GuideId>) -> usize {
    after
        .and_then(|id| roster.iter().position(|g| g.id == id))
        .map_or(0, |i| i + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::document::{DRAGO_K, DRAGO_K_RANGE};

    fn at(x: i32, y: i32) -> TileCoord {
        TileCoord::new(x, y)
    }

    fn span(coords: &[TileCoord]) -> CanvasBounds {
        CanvasBounds::of_tiles(coords.iter())
    }

    #[test]
    fn an_extent_spans_every_tile_it_was_given() {
        assert_eq!(span(&[]).tile_range(), None);
        assert_eq!(
            span(&[at(3, -2)]).tile_range(),
            Some((at(3, -2), at(3, -2)))
        );
        assert_eq!(
            span(&[at(3, -2), at(-1, 4), at(0, 0)]).tile_range(),
            Some((at(-1, -2), at(3, 4))),
            "the box is the corner-wise span, not the tiles' own hull"
        );
    }

    /// `DocState::bounds` is the union of its layers' own extents, so the join
    /// has to behave like one: identity on the empty box, and containing both
    /// arguments however they are ordered. If this ever failed, a document's
    /// bounds would depend on the order its layers happen to be visited in.
    #[test]
    fn the_union_is_a_join_with_the_empty_box_as_identity() {
        let empty = CanvasBounds::default();
        let a = span(&[at(0, 0), at(2, 1)]);
        let b = span(&[at(-4, 5)]);

        let mut grown = a;
        grown.union(empty);
        assert_eq!(grown.tile_range(), a.tile_range(), "empty adds nothing");

        let mut from_empty = empty;
        from_empty.union(a);
        assert_eq!(from_empty.tile_range(), a.tile_range(), "…either way round");

        let mut ab = a;
        ab.union(b);
        let mut ba = b;
        ba.union(a);
        assert_eq!(ab.tile_range(), ba.tile_range(), "order cannot matter");
        assert_eq!(ab.tile_range(), Some((at(-4, 0), at(2, 5))));
        // …and it agrees with spanning the tiles directly, which is what the
        // per-layer boxes replaced.
        assert_eq!(
            ab.tile_range(),
            span(&[at(0, 0), at(2, 1), at(-4, 5)]).tile_range(),
        );
    }

    use stark_model::document::ColorAdjust;

    const BASE: LayerId = LayerId::ROOT;
    const FILTER: LayerId = LayerId::solo(1);
    const OTHER: LayerId = LayerId::solo(2);

    /// A document with one paint layer and one filter above it in the root stack.
    fn with_filter() -> DocState {
        DocState::with_layer(BASE).insert_filter(
            FILTER,
            None,
            None,
            Filter::Color(ColorAdjust::NEUTRAL),
        )
    }

    /// A filter never carries (§21.2): every way a layer can be attached under a
    /// carrier declines a filter carrier, deterministically, so no log can build
    /// the arrangement — which is what lets the renderer and the projection treat
    /// "filter" and "plain member of a stack" as the same thing.
    #[test]
    fn a_filter_refuses_to_carry() {
        let state = with_filter();

        let inserted = state.insert_layer(OTHER, Some(FILTER), None);
        assert!(
            inserted
                .layer(FILTER)
                .expect("filter exists")
                .carries
                .is_empty(),
            "insert_layer attached a child to a filter",
        );
        assert!(
            inserted.layer(OTHER).is_none(),
            "the refused insert should add nothing anywhere",
        );

        let moved = state.move_layer(BASE, Some(FILTER), Place::Top);
        assert!(
            moved
                .layer(FILTER)
                .expect("filter exists")
                .carries
                .is_empty(),
            "move_layer attached a child to a filter",
        );
        assert!(
            moved.layer(BASE).is_some(),
            "the refused move must not lose the layer",
        );
        assert_eq!(
            moved.carrier_of(BASE),
            None,
            "the refused move must leave the layer where it was",
        );
    }

    /// **A blend is refused on a filter and a clip is not**, and the two halves are
    /// one test because the pair is where the difference is legible (§21.4.1). A
    /// mode is structurally inert there — no source, and (per the test above) never
    /// a group — so state refuses to store it, exactly as a matte refuses paint
    /// (§15.7); refused in state so replayed and replicated documents agree with
    /// local ones. A clip is not about a source at all: it bounds what the pass may
    /// write, which a filter can answer, so the flag is stored and the compositor
    /// reads it.
    #[test]
    fn a_filter_refuses_a_blend_and_takes_a_clip() {
        let state = with_filter()
            .set_layer_blend(FILTER, BlendMode::Multiply)
            .set_layer_clip(FILTER, true);
        let filter = state.layer(FILTER).expect("filter exists");
        assert_eq!(
            filter.composite.blend,
            BlendMode::Normal,
            "a filter stored a blend mode"
        );
        assert!(filter.composite.clip, "a filter dropped its clip");
        // …and it comes back off again: a flag that could only be set would be a
        // trap, since clearing it is how a spilling fringe is asked for.
        assert!(
            !state
                .set_layer_clip(FILTER, false)
                .layer(FILTER)
                .expect("filter exists")
                .composite
                .clip,
        );
        // …while a paint layer beside it still takes both.
        let state = state
            .set_layer_blend(BASE, BlendMode::Multiply)
            .set_layer_clip(BASE, true);
        let base = state.layer(BASE).expect("base exists");
        assert_eq!(base.composite.blend, BlendMode::Multiply);
        assert!(base.composite.clip);
    }

    /// Every filter is sanitized where it **enters state**, not only where a local
    /// command is minted — the path a crafted or corrupted file's replay and a
    /// nonconforming peer's action take (§21.5). A NaN that got through would reach
    /// every texel of the frame with nothing able to say why.
    #[test]
    fn a_filter_is_sanitized_wherever_it_enters_state() {
        let wild = Filter::Color(ColorAdjust {
            exposure: f32::NAN,
            contrast: f32::INFINITY,
            saturation: -5.0,
            hue: f32::NEG_INFINITY,
            tint: [9.0, f32::NAN],
        });
        let expect = wild.clone().sanitized();
        assert_ne!(wild, expect, "the wild filter must need sanitizing");

        let inserted = DocState::with_layer(BASE).insert_filter(FILTER, None, None, wild.clone());
        assert_eq!(
            inserted.layer(FILTER).and_then(Layer::filter),
            Some(expect.clone()),
            "insert_filter installed an unsanitized filter",
        );

        let set = with_filter().set_filter(FILTER, wild);
        assert_eq!(
            set.layer(FILTER).and_then(Layer::filter),
            Some(expect),
            "set_filter installed an unsanitized filter",
        );
    }

    /// And so is a **blend mode**, now that one carries numbers of its own
    /// (§18.0.4) — the same argument as the test above, down the same path: a
    /// replayed file and a peer's action reach state without passing through
    /// `Engine::process`, and a NaN bend reaches the blend pass, which is fullscreen.
    #[test]
    fn a_blend_mode_is_sanitized_wherever_it_enters_state() {
        let wild = BlendMode::Drago { k: f32::NAN };
        let state = DocState::with_layer(BASE).set_layer_blend(BASE, wild);
        assert_eq!(
            state.layer(BASE).map(|l| l.composite.blend),
            Some(BlendMode::Drago { k: DRAGO_K }),
            "set_layer_blend installed an unusable bend",
        );
        let state = state.set_layer_blend(BASE, BlendMode::Drago { k: 1e9 });
        assert_eq!(
            state.layer(BASE).map(|l| l.composite.blend),
            Some(BlendMode::Drago { k: DRAGO_K_RANGE.1 }),
            "set_layer_blend installed a bend past the end of its range",
        );
    }
}
