//! What a layer *holds* (§5.1, §14, §21): its tiles, its matte or its filter, and
//! the layers it carries.
//!
//! The other half — [`LayerId`], [`BlendMode`], [`Place`], [`Parcel`] and the
//! rest of what a layer *is* as a fact about the document — is `stark-model`'s
//! `document::layer`. The line is the usual one (§2): those are in the log, these
//! hold tiles.

use stark_model::document::Filter;
use std::sync::Arc;

use rpds::{HashTrieMap, Vector};

use stark_model::document::{BlendMode, LayerId, MatteRegion, Parcel};

use super::state::CanvasBounds;
use crate::gpu::tile::TileMap;

/// How something — a layer together with everything it carries, or a composited
/// group — **meets what lies beneath it** (§14.4.3).
///
/// The three travel as one value because they are one question asked three ways, and
/// because every rule about them is a rule about all three at once:
///
/// - They are stated **against a backdrop**, so they are vacuous where there is none
///   — the foot of the root stack, where a mode is the identity, a clip would erase
///   the layer, and opacity is the only one that still does anything.
/// - They belong to the **group as a whole**, never to its base. A group's members
///   composite over its base (§14.1), so the base's own content is a *member*: it
///   draws with [`IDENTITY`](Self::IDENTITY) and these are applied once, to the
///   result. Keeping them together is what makes that one assignment rather than
///   three, and it is why a fourth relational property added here needs no second
///   place to be remembered.
/// - They decide the compositor's **fast path** together ([`is_free`](Self::is_free)):
///   a layer needs isolating if *any* of them does something, so no one of them can
///   answer the question alone.
///
/// Grouped in the document rather than only in the draw list, so `Layer` and
/// [`CompositeGroup`] hold the same value and the render path never has to take them
/// apart. The **projection** ([`LayerInfo`]) deliberately keeps them flat: that is a
/// list of fields for a panel to hang one widget on each, and a blend picker has no
/// use for the clip beside it.
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
    /// carries. Also `Default`, so `..CompositeParams::IDENTITY` and
    /// `..Default::default()` are the same struct-update tail; the constant exists
    /// because "the identity" is what the call sites mean, and `default()` reads as
    /// "whatever we happened to pick".
    pub const IDENTITY: Self = Self {
        blend: BlendMode::Normal,
        clip: false,
        opacity: 1.0,
    };

    /// Whether these do nothing, so what they describe can draw straight into the
    /// accumulator instead of being isolated and merged (§6.3, §14.7).
    ///
    /// One predicate over all three rather than three tests at each call site: a
    /// layer needs isolating if *any* of them does something, and a call site that
    /// checked two of them would have found the fast path once too often.
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
/// The pairing is what makes the document's own bounds cheap. `DocState::bounds`
/// is the union of every layer's, and a union needs only each layer's box, so a
/// mutation that leaves a layer's tiles alone — a rename, an opacity, a reorder,
/// a stroke on some *other* layer — contributes the box that layer already knows
/// instead of re-walking its tiles. Only a layer whose map actually changed pays,
/// and it pays for itself alone rather than for the whole document.
///
/// Both fields are private and the only constructor derives the extent, so there
/// is no way to build the pair inconsistently. That matters more than the speed:
/// `bounds` is what "frame to content" and export's no-frame fallback measure
/// (§15.6), so a stale one is a wrongly-cropped export — and the alternative to
/// deriving it here is a rule every writer of a tile map has to remember, which
/// is the kind of rule this codebase spends structure to avoid (§1).
#[derive(Clone)]
pub struct PaintTiles {
    map: TileMap,
    bounds: CanvasBounds,
    revision: u64,
}

/// Where [`PaintTiles::revision`] comes from. Process-wide and monotonic, so a
/// number is never handed out twice however many documents, engines or history
/// versions are alive.
static TILE_REVISION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl PaintTiles {
    /// The tiles, with their extent and their revision derived once.
    fn new(map: TileMap) -> Self {
        Self {
            bounds: CanvasBounds::of_tiles(map.keys()),
            // `Relaxed` is the whole requirement: this is asked for a *distinct*
            // number, never for an ordering between threads.
            revision: TILE_REVISION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            map,
        }
    }

    /// The sparse tile map itself.
    pub fn map(&self) -> &TileMap {
        &self.map
    }

    /// A number that changes exactly when these tiles do — **the cheapest sound
    /// answer to "is this the same picture?"** for anything caching a render of
    /// them (§14.6: the layer panel's thumbnails).
    ///
    /// Derived here, beside `bounds`, and for the identical reason the doc above
    /// gives: `new` is the only way to install a map, so a writer cannot forget to
    /// move it. That is what makes this a fact about the value rather than a
    /// counter a new mutation path could leave behind.
    ///
    /// **Why not the map's pointer.** A tile is never rewritten once committed, so
    /// `TilePairHandle::same` already reads identity as "unchanged" — but that is
    /// sound only between two handles *both held*. A cache stores its key and
    /// compares it later, by which time the allocation it named may have been freed
    /// and a different map built at the same address; the key would match and the
    /// picture would be stale, with nothing anywhere to say so. A counter that only
    /// ever goes up cannot collide with its own past.
    ///
    /// **What undo does with it.** The revision travels with the value, so rewinding
    /// something that left the tiles alone — a blend mode, an opacity, a rename —
    /// restores a version holding these same tiles under this same number, and a
    /// cache keyed on it is untouched by a held Ctrl+Z through a run of property
    /// edits. Rewinding past a *stroke* is the other case, and there the number is
    /// simply a new one rather than the pre-stroke number coming back. That is the
    /// sound direction and deliberately not tightened: a fresh key costs one
    /// re-render, while reusing a stale one would cost a wrong picture. Both are
    /// pinned by `tests/layers.rs`.
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
    /// transparency *is* its layer opacity, which is the whole point of it
    /// being a layer.
    ///
    /// Physically this is a flat, opaque *coat of paint*: the compositor gives it
    /// a constant thickness, so its interior lights flat (zero height gradient —
    /// no substrate, and the paint film's uniform sheen reads as an even wash rather
    /// than a glint) while its boundary catches light the same way any stroke edge
    /// does — a graded wash varies the paint's color, never its thickness, the
    /// same statement the gradient fill makes (§22.4). See §15.4 for
    /// why it must write the aux target at all, and why its blend there is
    /// `over` rather than additive.
    Matte { region: MatteRegion, paint: Parcel },
    /// A **function of what is composited beneath it** in its own stack (§21).
    ///
    /// The one content that is not content: a filter layer holds no tiles and no
    /// region, and its whole effect is one fullscreen pass at composite time that
    /// reads the accumulator and writes it back adjusted. So it costs no GPU memory,
    /// it is free to re-tune, and — because the accumulator it reads is *its own
    /// stack's* — how far it reaches is decided by where it sits in the tree rather
    /// than by a mode of its own (§21.2).
    ///
    /// Its layer opacity is the filter's **strength**, mixed against the untouched
    /// backdrop, so fading a filter layer means what fading any other layer means
    /// (§21.4). Its **blend mode** has nothing to say and is refused: a mode
    /// describes how a *source* meets a backdrop, and a filter has no source — it
    /// *is* the backdrop, rewritten.
    ///
    /// Its **clip** is live, and the asymmetry with the mode beside it is the whole
    /// of §21.4.1. A clip does not ask about a source; it says where the layer is
    /// allowed to land, and that survives having none — the filter's result exists
    /// only where the backdrop it read had coverage. So a clipped filter hands
    /// coverage and height back exactly as it found them: it may say what color the
    /// paint already there should be, never where there is paint. Inert for a filter
    /// that is a function of one texel, and the live case for one that displaces
    /// (§21.10).
    Filter(Filter),
}

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
///   the backdrop, and which therefore belong to the layer *plus its subtree*
///   rather than to its own content. At the bottom of a stack there is no backdrop
///   inside the group, so they are vacuous there and are free to describe the
///   group's own merge outward. That is not an overload: `merge()` with an empty
///   backdrop is provably the `Normal` result, so the slot could not express
///   anything to begin with (`blend_common.wesl`, and `tests/blend.rs` pins it to
///   the byte).
/// - **Intrinsic** — `visible`, `name`, which describe the layer itself and mean
///   the same thing whatever is under it.
///
/// Opacity sits in the first group, though it reads as intrinsic — "how faded is this
/// layer". It is applied at the same step as the other two, to the same thing: the
/// group's composited whole (§14.7). Held as a separate field it would also reach the
/// base's own content, and a group base at 0.5 would draw its paint at 0.25; the base
/// composites with [`CompositeParams::IDENTITY`] so there is one place that fade can
/// be applied.
#[derive(Clone)]
pub struct Layer {
    pub id: LayerId,
    /// How this layer — **and everything it carries** — meets what lies beneath it
    /// (§14.4.3). One value rather than three fields, because every rule about them
    /// is a rule about all three at once: see [`CompositeParams`].
    ///
    /// The **clip** is the one worth restating here (§14.4). The layer exists only
    /// where there is paint under it *in its own stack*: it inherits the alpha of
    /// everything composited below it there, not of "the nearest layer that is not
    /// itself clipped". There is no chain to trace, because the group is what bounds
    /// *below* — clipping to exactly one layer is that layer carrying this one, which
    /// is the same single gesture every other app spells with a separate mode. It is
    /// not a scale on the source's alpha; see `blend_common.wesl` for why that is the
    /// wrong operation and what the right one is.
    pub composite: CompositeParams,
    /// Whether the layer contributes to the composite.
    pub visible: bool,
    /// What the author called this layer, or `None` for one that has never been
    /// named.
    ///
    /// Absent rather than pre-filled with "Layer 3", because the two are
    /// different facts: an unnamed layer is *described* by its position in the
    /// stack, and a frontend that spells that out should keep doing so as the
    /// stack changes. Storing the generated text would freeze one moment's
    /// description into the document and make it look deliberate.
    ///
    /// `Arc<str>` because the name is read far more often than it is written —
    /// every `observe()` projects it — and never edited in place.
    pub name: Option<Arc<str>>,
    pub content: LayerContent,
    /// The layers carried on this one, **bottom-to-top** — the group this layer
    /// is the base of (§14.2). Empty for a layer that carries
    /// nothing, which is every layer until one is dropped onto it.
    ///
    /// They composite *over* this layer's own content and beneath whatever comes
    /// after the group, so this order is render order and panel order alike: the
    /// panel draws them indented **above** the base, which is where they land.
    ///
    /// A `Vector` for the same reason the document's stack is one — the whole
    /// tree is cloned per document version, so every level of it has to be
    /// persistent (§5.1).
    pub carries: Vector<Layer>,
}

impl Layer {
    /// An empty paint layer, carrying nothing.
    pub fn new(id: LayerId) -> Self {
        Self {
            id,
            composite: CompositeParams::IDENTITY,
            visible: true,
            name: None,
            content: LayerContent::Paint(PaintTiles::new(HashTrieMap::new())),
            carries: Vector::new(),
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

    /// This layer's painted tiles, or `None` if it holds none. Deliberately an
    /// `Option` rather than an empty map: "this layer has no tiles" is a real
    /// fact about it, and making callers say what they do about it is what keeps
    /// a matte or a filter from silently reading as an empty paint layer.
    pub fn tiles(&self) -> Option<&TileMap> {
        match &self.content {
            LayerContent::Paint(tiles) => Some(tiles.map()),
            LayerContent::Matte { .. } | LayerContent::Filter(_) => None,
        }
    }

    /// The extent of this layer's **own** painted tiles, already derived
    /// ([`PaintTiles`]) — what `DocState`'s bounds union together.
    ///
    /// Empty for a matte, and that is not a special case to remember: a matte
    /// covers the infinite plane, so counting it would make the document's
    /// bounds unbounded and break both "frame to content" and export's no-frame
    /// fallback (§15.6). Having no tiles, it has no extent, and the right answer
    /// falls out.
    pub fn bounds(&self) -> CanvasBounds {
        match &self.content {
            LayerContent::Paint(tiles) => tiles.bounds,
            LayerContent::Matte { .. } | LayerContent::Filter(_) => CanvasBounds::default(),
        }
    }

    /// A number that changes exactly when this layer's own painted tiles do
    /// ([`PaintTiles::revision`]), or `None` for a layer that holds no tiles.
    ///
    /// The `None` is the same statement [`tiles`](Self::tiles) makes and is load-bearing
    /// in the same way: a matte's content is a rect and a color, a filter's is nothing
    /// at all, and neither has a picture of its own to cache. A caller gets "there is
    /// no thumbnail here" from the same field that would have told it the key, instead
    /// of asking a second question about the layer's kind.
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

    /// The same layer with its painted tiles replaced. A no-op on anything with no
    /// tiles to replace — and on a map that is the one already there.
    ///
    /// **The identity case keeps the revision**, which is the whole reason it is
    /// tested for. `PaintTiles::new` mints a fresh [`PaintTiles::revision`], and every
    /// thumbnail in the layer panel is keyed on it (§14.6) — so a caller that hands
    /// back the map it was given re-renders every one of them to show the same
    /// picture. A merge with a neutral filter is exactly that caller: it is *defined*
    /// as leaving the destination's texels alone (§14.11.7), and said so by passing
    /// them straight through.
    ///
    /// `ptr_eq` rather than a comparison: the map is persistent, so sharing a root is
    /// exactly "these are the same tiles" and costs one pointer test. Two maps that
    /// are equal without sharing a root still mint — the safe direction, since the
    /// cost is a re-render and the alternative is a stale thumbnail.
    pub fn with_tiles(&self, tiles: TileMap) -> Self {
        match &self.content {
            LayerContent::Paint(current) if current.map().ptr_eq(&tiles) => self.clone(),
            LayerContent::Paint(_) => Self {
                content: LayerContent::Paint(PaintTiles::new(tiles)),
                ..self.clone()
            },
            LayerContent::Matte { .. } | LayerContent::Filter(_) => self.clone(),
        }
    }

    /// Whether this layer carries any others — i.e. whether it is a **group**
    /// (§14.2). There is no other kind of group, so this is the
    /// whole test.
    pub fn is_group(&self) -> bool {
        !self.carries.is_empty()
    }

    /// The same layer carrying `carries` instead.
    pub fn with_carries(&self, carries: Vector<Layer>) -> Self {
        Self {
            carries,
            ..self.clone()
        }
    }

    /// This layer and everything it carries, in **composite order**: the base
    /// first, then each carried subtree in turn. `depth` counts levels below
    /// this one, so the receiver is always visited at `0`.
    ///
    /// One traversal for every reader — the projection, the bounds, the draw
    /// list, the peers' layer index — so "what order does a group composite in"
    /// is answered in one place instead of once per caller with a chance to
    /// disagree.
    /// The borrow is the **tree's**, not each call's, so a walk may keep the layers
    /// it has seen — which is what lets a reader answer a question about a layer's
    /// lower sibling or its carrier without searching for either
    /// ([`MergeSite`](super::merge::MergeSite)). A `FnMut(&Layer, usize)` binds its
    /// argument for the call alone and nothing may outlive it.
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
