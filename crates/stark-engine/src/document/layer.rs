//! What a layer *holds* (§5.1, §14, §21): its tiles, its matte or its filter, and
//! the layers it carries.
//!
//! The other half — [`LayerId`], [`BlendMode`], [`Place`](stark_model::document::Place),
//! [`Parcel`] and the rest of what a layer *is* as a fact about the document — is
//! `stark-model`'s `document::layer`. The line is the usual one (§2): those are in
//! the log, these hold tiles.

use stark_model::document::Filter;
use std::rc::Rc;
use std::sync::Arc;

use rpds::{HashTrieMap, Vector};

use stark_model::document::{BlendMode, LayerId, MatteRegion, Parcel};
use stark_model::geom::IVec2;

use super::liquify::LiquifyRun;
use super::state::CanvasBounds;
use crate::gpu::tile::TileMap;

/// How something — a layer together with everything it carries, or a composited
/// group — **meets what lies beneath it** (§14.4.3).
///
/// The three travel as one value because every rule about them is a rule about all
/// three at once:
///
/// - They are stated **against a backdrop**, so they are vacuous where there is none
///   — the foot of the root stack, where a mode is the identity and a clip would
///   erase the layer; opacity is the only one that still does anything.
/// - They belong to the **group as a whole**, never to its base. A group's members
///   composite over its base (§14.1), so the base's own content is a *member*: it
///   draws with [`IDENTITY`](Self::IDENTITY) and these are applied once, to the
///   result.
/// - They decide the compositor's **fast path** together ([`is_free`](Self::is_free)):
///   a layer needs isolating if *any* of them does something.
///
/// `Layer` and [`CompositeGroup`] hold the same value, so the render path never has
/// to take them apart. The projection ([`LayerInfo`]) keeps them flat instead — that
/// is a list of fields for a panel to hang one widget on each.
///
/// [`CompositeGroup`]: crate::gpu::CompositeGroup
/// [`LayerInfo`]: crate::LayerInfo
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CompositeParams {
    pub blend: BlendMode,
    /// Clip to the coverage of what this composites onto (§14.4).
    pub clip: bool,
    /// Opacity in [0, 1], applied to the **composited whole**.
    pub opacity: f32,
}

impl CompositeParams {
    /// Meeting the backdrop by plain premultiplied "over" at full strength — which is
    /// to say not interacting with it at all.
    ///
    /// The value a group's **base** composites with, and the value a fresh layer
    /// carries. Also `Default`, so the two struct-update tails are the same value;
    /// the constant exists because "the identity" is what the call sites mean.
    pub const IDENTITY: Self = Self {
        blend: BlendMode::Normal,
        clip: false,
        opacity: 1.0,
    };

    /// Whether these do nothing, so what they describe can draw straight into the
    /// accumulator instead of being isolated and merged (§6.3, §14.7).
    ///
    /// One predicate over all three rather than three tests at each call site: a
    /// layer needs isolating if *any* of them does something.
    pub fn is_free(self) -> bool {
        self.blend.is_normal() && !self.clip && self.opacity >= 1.0
    }
}

impl Default for CompositeParams {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A paint layer's tiles **and the extent they span** (§6) — one value, because
/// the extent is a function of the map and the two must never disagree.
///
/// Both fields are private and the only constructor derives the extent, so the pair
/// cannot be built inconsistently. `bounds` is what "frame to content" and export's
/// no-frame fallback measure (§15.6), so a stale one is a wrongly-cropped export.
/// Keeping it here also makes `DocState::bounds` a union of boxes each layer already
/// knows, so a mutation that leaves a layer's tiles alone pays nothing for it.
#[derive(Clone)]
pub struct PaintTiles {
    map: TileMap,
    bounds: CanvasBounds,
    revision: u64,
    /// The liquify run the layer's picture is composed through (§6.13), if any — see
    /// [`LiquifyRun`]. What the map holds at the run's written tiles is the run's
    /// base resampled through its field, and the next liquify stroke composes into
    /// that rather than resampling it again.
    ///
    /// Only a liquify stroke writes it ([`Layer::with_painted`]); every other paint
    /// edit carries it forward unchanged ([`Layer::with_tiles`]), which is what
    /// keeps a paint stroke commuting with a liquify stroke beyond its reach
    /// (§12.6, `document::liquify`).
    liquify: Option<Rc<LiquifyRun>>,
}

/// Where [`PaintTiles::revision`] comes from. Process-wide and monotonic, so a
/// number is never handed out twice however many documents, engines or history
/// versions are alive.
static TILE_REVISION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl PaintTiles {
    /// The tiles, with their extent and their revision derived once, and the liquify
    /// run they are composed through — `None` for a picture no run stands behind.
    fn new(map: TileMap, liquify: Option<Rc<LiquifyRun>>) -> Self {
        Self {
            bounds: CanvasBounds::of_tiles(map.keys()),
            // `Relaxed` is the whole requirement: this is asked for a *distinct*
            // number, never for an ordering between threads.
            revision: TILE_REVISION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            map,
            liquify,
        }
    }

    /// The sparse tile map itself.
    pub fn map(&self) -> &TileMap {
        &self.map
    }

    /// The liquify run these tiles are composed through (§6.13), if any.
    pub fn liquify(&self) -> Option<&Rc<LiquifyRun>> {
        self.liquify.as_ref()
    }

    /// A number that changes exactly when these tiles do — **the cheapest sound
    /// answer to "is this the same picture?"** for anything caching a render of
    /// them (§14.6: the layer panel's thumbnails).
    ///
    /// Derived in `new`, the only way to install a map, so a writer cannot forget to
    /// move it.
    ///
    /// **Why not the map's pointer.** A cache stores its key and compares it later,
    /// by which time the allocation it named may have been freed and a different map
    /// built at the same address; the key would match and the picture would be stale.
    /// A counter that only ever goes up cannot collide with its own past.
    ///
    /// **What undo does with it.** The revision travels with the value, so rewinding
    /// something that left the tiles alone — a blend mode, an opacity, a rename —
    /// restores these same tiles under this same number. Rewinding past a *stroke*
    /// yields a fresh number rather than the pre-stroke one coming back: the sound
    /// direction, since a fresh key costs one re-render where a stale one would cost
    /// a wrong picture. Both are pinned by `tests/layers.rs`.
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

/// What a layer is made of (§15.2).
#[derive(Clone)]
pub enum LayerContent {
    /// Painted tiles. Only populated ones exist — this sparsity is the infinite
    /// canvas.
    Paint(PaintTiles),
    /// A procedural region filled with a [`Parcel`] — one flat color, or a
    /// gradient ramp (§22.4). The paint converts to working-space
    /// channels at composite time, so the log stays independent of whether the
    /// document is Oklab or Mixbox. A matte has no alpha of its own: its
    /// transparency *is* its layer opacity. Both halves state their geometry in the
    /// layer's frame, placed by [`Layer::translation`] on the way out (§14.12,
    /// §15.2).
    ///
    /// Physically a flat, opaque *coat of paint*: the compositor gives it a constant
    /// thickness, so its interior lights flat (zero height gradient) while its
    /// boundary catches light the way any stroke edge does — a graded wash varies the
    /// paint's color, never its thickness (§22.4). §15.4 says why it must write the
    /// aux target at all, and why its blend there is `over` rather than additive.
    Matte { region: MatteRegion, paint: Parcel },
    /// A **function of what is composited beneath it** in its own stack (§21).
    ///
    /// It holds no tiles and no region: its whole effect is one fullscreen pass at
    /// composite time that reads the accumulator and writes it back adjusted. The
    /// accumulator it reads is *its own stack's*, so how far it reaches is decided by
    /// where it sits in the tree rather than by a mode of its own (§21.2).
    ///
    /// Its layer opacity is the filter's **strength**, mixed against the untouched
    /// backdrop, so fading a filter layer means what fading any other layer means
    /// (§21.4). Its **blend mode** is refused: a mode describes how a *source* meets
    /// a backdrop, and a filter has no source — it *is* the backdrop, rewritten.
    ///
    /// Its **clip** is live, and the asymmetry with the mode beside it is the whole
    /// of §21.4.1. A clip does not ask about a source; it says where the layer is
    /// allowed to land, and that survives having none — the filter's result exists
    /// only where the backdrop it read had coverage. So a clipped filter hands
    /// coverage and height back exactly as it found them: it may say what color the
    /// paint already there should be, never where there is paint. Inert for a filter
    /// that is a function of one texel, live for one that displaces (§21.10).
    Filter(Filter),
}

/// The canvas extent of everything a layer **carries** (§14.12), derived with
/// [`Layer::carries`] and kept beside it — the second level of the argument
/// [`PaintTiles`] makes. A subtree's box is a function of its children's boxes, so
/// the document's is a union over the root stack rather than a walk of the tree
/// (`DocState::with_layers`).
///
/// Only [`Layer::with_carries`] mints one. The field it lives in is `pub(crate)`
/// rather than private because struct update refuses a private field, and
/// `apply.rs` builds two layers as `Layer { .., ..base }` — copying `base`'s box.
/// What the type rules out is a literal that names *both*: nothing outside this
/// module can spell a fresh one.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CarriedBounds(CanvasBounds);

/// A single layer: its content, what it carries, and its presentation
/// properties.
///
/// **A group is a layer with a non-empty [`carries`](Self::carries)**, and there
/// is no other kind (§14.2). One sentence covers the whole model:
/// a layer's [`composite`](Self::composite) params describe how it *together with
/// everything it carries* meets what lies beneath it.
///
/// That splits the properties in two, and the split is why there is no separate
/// group object to own a second copy of anything:
///
/// - **[`composite`](Self::composite)** — blend, clip and opacity, which are about
///   the backdrop and therefore belong to the layer *plus its subtree* rather than
///   to its own content. At the bottom of a stack there is no backdrop inside the
///   group, so they are vacuous there and are free to describe the group's own merge
///   outward: `merge()` with an empty backdrop is provably the `Normal` result, so
///   the slot could not express anything to begin with (`blend_common.wesl`, pinned
///   to the byte by `tests/blend.rs`).
/// - **Intrinsic** — `visible`, `name`, which mean the same thing whatever is under
///   the layer.
///
/// Opacity sits in the first group though it reads as intrinsic. It is applied at the
/// same step as the other two, to the group's composited whole (§14.7); held as a
/// separate field it would also reach the base's own content, and a group base at 0.5
/// would draw its paint at 0.25. The base composites with
/// [`CompositeParams::IDENTITY`], so there is one place that fade can be applied.
#[derive(Clone)]
pub struct Layer {
    pub id: LayerId,
    /// How this layer — **and everything it carries** — meets what lies beneath it
    /// (§14.4.3). One value rather than three fields: see [`CompositeParams`].
    ///
    /// The **clip** is the one worth restating here (§14.4). The layer exists only
    /// where there is paint under it *in its own stack*: it inherits the alpha of
    /// everything composited below it there, not of "the nearest layer that is not
    /// itself clipped". There is no chain to trace, because the group is what bounds
    /// *below* — clipping to exactly one layer is that layer carrying this one. It is
    /// not a scale on the source's alpha; see `blend_common.wesl` for why that is the
    /// wrong operation and what the right one is.
    pub composite: CompositeParams,
    /// Whether the layer contributes to the composite.
    pub visible: bool,
    /// What the author called this layer, or `None` for one that has never been
    /// named.
    ///
    /// Absent rather than pre-filled with "Layer 3": an unnamed layer is *described*
    /// by its position in the stack, and storing the generated text would freeze one
    /// moment's description into the document and make it look deliberate.
    ///
    /// `Arc<str>` because every `observe()` projects the name and nothing edits it in
    /// place.
    pub name: Option<Arc<str>>,
    pub content: LayerContent,
    /// The layers carried on this one, **bottom-to-top** — the group this layer
    /// is the base of (§14.2). Empty for a layer that carries nothing.
    ///
    /// They composite *over* this layer's own content and beneath whatever comes
    /// after the group, so this order is render order and panel order alike: the
    /// panel draws them indented **above** the base, which is where they land.
    ///
    /// A `Vector` because the whole tree is cloned per document version, so every
    /// level of it has to be persistent (§5.1).
    pub carries: Vector<Layer>,
    /// The canvas extent of `carries`, derived with it — see [`CarriedBounds`].
    pub(crate) carried: CarriedBounds,
    /// Where this layer's frame sits on the canvas, in whole pixels (§14.12): its
    /// tiles are keyed in the layer's own frame, and the compositor, the pick and
    /// the bounds add this on the way out. A matte's geometry — its rect and its
    /// gradient axis — is stated in the same frame (§15.2), so a matte moves the
    /// same way. Moving a layer touches no tile.
    ///
    /// **Intrinsic, and deliberately not inherited** down [`carries`](Self::carries):
    /// a footprint is built from an action alone and could not name the ancestors an
    /// inherited offset would make every stroke read (§12.6) — so the gesture that
    /// moves a group writes every member instead (`ActionKind::TranslateLayers`).
    /// Not part of [`CompositeParams`] either: it forces no isolation, so a
    /// translated layer still joins a plain run.
    pub translation: IVec2,
}

impl Layer {
    /// Whether the compositor draws this layer at all — **and with it, everything it
    /// carries**, since the group *is* the layer (§14.3).
    ///
    /// The draw list's first cull, and the one predicate every other cull in the
    /// renderer and the projection must agree with (`LayerInfo::has_underlay`'s doc
    /// says it "follows the renderer").
    ///
    /// Fully transparent counts as hidden because it is: the compositor multiplies by
    /// this weight, so zero contributes nothing and the subtree beneath it is a draw
    /// list nobody can see.
    pub fn is_shown(&self) -> bool {
        self.visible && self.composite.opacity > 0.0
    }

    /// Whether this layer's **own content** puts anything into the accumulator: a
    /// matte always covers, paint only once painted, a filter never — it rewrites
    /// what is already there and adds nothing (§21.3).
    ///
    /// Says nothing about what the layer carries; [`Self::is_shown`] is the other
    /// half.
    pub fn draws_content(&self) -> bool {
        match &self.content {
            LayerContent::Matte { .. } => true,
            LayerContent::Paint(_) => self.tiles().is_some_and(|t| !t.is_empty()),
            LayerContent::Filter(_) => false,
        }
    }

    /// An empty paint layer, carrying nothing.
    pub fn new(id: LayerId) -> Self {
        Self {
            id,
            composite: CompositeParams::IDENTITY,
            visible: true,
            name: None,
            content: LayerContent::Paint(PaintTiles::new(HashTrieMap::new(), None)),
            carries: Vector::new(),
            carried: CarriedBounds(CanvasBounds::default()),
            translation: IVec2::ZERO,
        }
    }

    /// A paint layer whose own extent is `bounds`, with no tiles behind it — for the
    /// tree-level bounds tests, which have no device to lay a tile with. Skips the
    /// map/box pairing [`PaintTiles::new`] derives, which the GPU suite pins
    /// (`tests/layers.rs`).
    #[cfg(test)]
    pub(crate) fn spanning(id: LayerId, bounds: CanvasBounds) -> Self {
        let tiles = PaintTiles {
            bounds,
            ..PaintTiles::new(HashTrieMap::new(), None)
        };
        Self {
            content: LayerContent::Paint(tiles),
            ..Self::new(id)
        }
    }

    /// A matte layer over `region`, filled with `paint` (§15.4).
    pub fn matte(id: LayerId, region: MatteRegion, paint: Parcel) -> Self {
        Self {
            content: LayerContent::Matte { region, paint },
            ..Self::new(id)
        }
    }

    /// A filter layer running `filter` over the stack beneath it (§21).
    pub fn filter_layer(id: LayerId, filter: Filter) -> Self {
        Self {
            content: LayerContent::Filter(filter),
            ..Self::new(id)
        }
    }

    /// This layer's painted tiles, or `None` if it holds none — an `Option` rather
    /// than an empty map, so a matte or a filter cannot silently read as an empty
    /// paint layer.
    pub fn tiles(&self) -> Option<&TileMap> {
        match &self.content {
            LayerContent::Paint(tiles) => Some(tiles.map()),
            LayerContent::Matte { .. } | LayerContent::Filter(_) => None,
        }
    }

    /// The extent of this layer's **own** painted tiles, already derived
    /// ([`PaintTiles`]) — what `DocState`'s bounds union together.
    ///
    /// Empty for a matte: a matte covers the infinite plane, so counting it would
    /// make the document's bounds unbounded and break both "frame to content" and
    /// export's no-frame fallback (§15.6).
    pub fn bounds(&self) -> CanvasBounds {
        match &self.content {
            LayerContent::Paint(tiles) => tiles.bounds,
            LayerContent::Matte { .. } | LayerContent::Filter(_) => CanvasBounds::default(),
        }
    }

    /// A number that changes exactly when this layer's own painted tiles do
    /// ([`PaintTiles::revision`]), or `None` for a layer that holds no tiles.
    ///
    /// `None` says "there is no thumbnail here": a matte's content is a rect and a
    /// color, a filter's is nothing at all, and neither has a picture of its own to
    /// cache.
    pub fn content_revision(&self) -> Option<u64> {
        match &self.content {
            LayerContent::Paint(tiles) => Some(tiles.revision()),
            LayerContent::Matte { .. } | LayerContent::Filter(_) => None,
        }
    }

    /// The matte region this layer fills, if it is a matte.
    pub fn matte_region(&self) -> Option<MatteRegion> {
        match &self.content {
            LayerContent::Matte { region, .. } => Some(*region),
            LayerContent::Paint(_) | LayerContent::Filter(_) => None,
        }
    }

    /// The filter this layer runs over the stack beneath it, if it is one (§21).
    pub fn filter(&self) -> Option<Filter> {
        // Cloned out: a filter is read once per render and once per projection
        // (§21.7), and a ramp's stop list is small — the borrow a reference
        // would hand back is not worth threading through every consumer.
        match &self.content {
            LayerContent::Filter(f) => Some(f.clone()),
            LayerContent::Paint(_) | LayerContent::Matte { .. } => None,
        }
    }

    /// Whether strokes may be painted onto this layer. Neither a matte nor a filter
    /// has a tile map, so a stroke targeting one is refused rather than silently
    /// swallowed or magically rasterized (§15.7, §21.4).
    pub fn is_paintable(&self) -> bool {
        matches!(self.content, LayerContent::Paint(_))
    }

    /// Whether this layer has a frame to move — whether writing
    /// [`translation`](Self::translation) places anything (§14.12). Paint stands
    /// somewhere and so does a matte, whose rect and gradient axis are stated in
    /// the layer's frame (§15.2); a filter has nothing that sits anywhere (§21).
    ///
    /// One predicate rather than two matches, because the fold and the gesture's
    /// subtree expansion must leave out exactly the same layers
    /// (`DocState::translate_layers`, `Engine::translate_moves`).
    pub fn is_translatable(&self) -> bool {
        match &self.content {
            LayerContent::Paint(_) | LayerContent::Matte { .. } => true,
            LayerContent::Filter(_) => false,
        }
    }

    /// The same layer with its painted tiles replaced. A no-op on anything with no
    /// tiles to replace — and on a map that is the one already there.
    ///
    /// **The identity case keeps the revision.** `PaintTiles::new` mints a fresh
    /// [`PaintTiles::revision`] and every thumbnail in the layer panel is keyed on it
    /// (§14.6), so a caller that hands back the map it was given would re-render all
    /// of them to show the same picture — which a merge with a neutral filter,
    /// *defined* as leaving the destination's texels alone (§14.11.7), does exactly.
    ///
    /// `ptr_eq` rather than a comparison: the map is persistent, so sharing a root is
    /// exactly "these are the same tiles". Two maps that are equal without sharing a
    /// root still mint — the safe direction, costing a re-render rather than a stale
    /// thumbnail.
    ///
    /// **The liquify run rides along untouched** (§6.13): only
    /// [`with_painted`](Self::with_painted) replaces it, which is what lets a paint
    /// edit commute with a liquify stroke beyond its reach (§12.6). Whether the run
    /// it leaves is stale is the next liquify stroke's question, answered by tile
    /// identity ([`LiquifyRun::is_fresh`]).
    pub fn with_tiles(&self, tiles: TileMap) -> Self {
        match &self.content {
            LayerContent::Paint(current) if current.map().ptr_eq(&tiles) => self.clone(),
            LayerContent::Paint(current) => Self {
                content: LayerContent::Paint(PaintTiles::new(tiles, current.liquify.clone())),
                ..self.clone()
            },
            LayerContent::Matte { .. } | LayerContent::Filter(_) => self.clone(),
        }
    }

    /// The same layer with its painted tiles replaced **and its liquify run set** —
    /// what a liquify stroke's render lands (§6.13). `run` is the run the new tiles
    /// are composed through; `None` says the render was not a liquify stroke's and
    /// the run in place stays, which is [`with_tiles`](Self::with_tiles) exactly.
    pub fn with_painted(&self, tiles: TileMap, run: Option<Rc<LiquifyRun>>) -> Self {
        match run {
            Some(run) => match &self.content {
                LayerContent::Paint(_) => Self {
                    content: LayerContent::Paint(PaintTiles::new(tiles, Some(run))),
                    ..self.clone()
                },
                LayerContent::Matte { .. } | LayerContent::Filter(_) => self.clone(),
            },
            None => self.with_tiles(tiles),
        }
    }

    /// The same layer with its liquify run replaced and its tiles left alone — the
    /// undo patch's restore of the run alone (§12.6, `patch::PatchOp::LiquifyRun`).
    /// A no-op on anything without tiles, and on the run already there.
    pub(crate) fn with_liquify_run(&self, run: Option<Rc<LiquifyRun>>) -> Self {
        match &self.content {
            LayerContent::Paint(current) if LiquifyRun::same(current.liquify(), run.as_ref()) => {
                self.clone()
            }
            LayerContent::Paint(current) => Self {
                content: LayerContent::Paint(PaintTiles {
                    liquify: run,
                    ..current.clone()
                }),
                ..self.clone()
            },
            LayerContent::Matte { .. } | LayerContent::Filter(_) => self.clone(),
        }
    }

    /// The liquify run this layer's picture is composed through (§6.13), or `None`
    /// for a layer with no tiles or no run.
    pub fn liquify_run(&self) -> Option<&Rc<LiquifyRun>> {
        match &self.content {
            LayerContent::Paint(tiles) => tiles.liquify(),
            LayerContent::Matte { .. } | LayerContent::Filter(_) => None,
        }
    }

    /// Whether this layer carries any others — i.e. whether it is a **group**
    /// (§14.2). There is no other kind of group, so this is the
    /// whole test.
    pub fn is_group(&self) -> bool {
        !self.carries.is_empty()
    }

    /// The extent of this layer **and everything it carries**, placed on the
    /// canvas (§14.12) — what `DocState::bounds` unions over the root stack.
    ///
    /// O(1): the own box came with the tiles ([`PaintTiles`]), the carried box
    /// with the children ([`CarriedBounds`]). The own half is placed here rather
    /// than cached, since `translation` and `content` are written by literals
    /// and a box that depended on them could be left behind.
    pub fn subtree_bounds(&self) -> CanvasBounds {
        let mut out = self.bounds().shifted(self.translation);
        out.union(self.carried.0);
        out
    }

    /// The same layer carrying `carries` instead — and knowing their extent,
    /// which is O(children) here because each child already knows its own.
    pub fn with_carries(&self, carries: Vector<Layer>) -> Self {
        let mut carried = CanvasBounds::default();
        for l in carries.iter() {
            carried.union(l.subtree_bounds());
        }
        Self {
            carries,
            carried: CarriedBounds(carried),
            ..self.clone()
        }
    }

    /// This layer and everything it carries, in **composite order**: the base
    /// first, then each carried subtree in turn. `depth` counts levels below
    /// this one, so the receiver is always visited at `0`.
    ///
    /// One traversal for every reader — the projection, the bounds, the draw
    /// list, the peers' layer index — so composite order is answered in one place.
    ///
    /// The borrow is the **tree's**, not each call's, so a walk may keep the layers
    /// it has seen and answer a question about a layer's lower sibling or its carrier
    /// without searching for either ([`MergeSite`](super::merge::MergeSite)).
    pub fn visit<'a>(&'a self, depth: usize, f: &mut impl FnMut(&'a Layer, usize)) {
        f(self, depth);
        for l in self.carries.iter() {
            l.visit(depth + 1, f);
        }
    }

    /// The layer with this id within this subtree, the receiver included.
    pub fn find(&self, id: LayerId) -> Option<&Layer> {
        if self.id == id {
            return Some(self);
        }
        self.carries.iter().find_map(|l| l.find(id))
    }
}
