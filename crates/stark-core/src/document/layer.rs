//! Layers (§5.1, §15.2, §14, §21). A layer is a sparse, persistent map of painted
//! tiles, a **matte** — a procedural region filled with a flat colour — or a
//! **filter**, which is a function of what is composited beneath it rather than
//! content of its own; plus its presentation properties, plus the layers it
//! **carries**.
//!
//! A layer stacks with premultiplied "over" unless its [`BlendMode`] says
//! otherwise or it is [`clip`](Layer::clip)ped, in which case the compositor
//! isolates it and merges it through the mode (§18.0.4). A
//! layer that carries others is a **group** — there is no separate group type —
//! and the same isolation, recursed, is what composites it (§14.7).

use std::sync::Arc;

use rpds::{HashTrieMap, Vector};
use serde::{Deserialize, Serialize};

use super::action::ActorId;
use super::filter::Filter;
use super::state::CanvasBounds;
use crate::geom::Vec2;
use crate::gpu::tile::TileMap;

/// Stable identifier for a layer within a document.
///
/// Ids are **minted from the author**, not from a shared counter — see
/// [`LayerId::mint`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerId(pub u64);

impl LayerId {
    /// Mint the id for `actor`'s `n`th layer (§17.9).
    ///
    /// Two peers adding a layer at the same moment must not mint the same id. A
    /// counter resynced from the log does exactly that — both peers see `n` layers,
    /// both mint `n + 1`, and the log ends up holding two different layers under one
    /// id, which `layer_index` resolves to whichever comes first. That is a genuine
    /// convergence failure, so the id space is partitioned by author instead: a
    /// mixed 32-bit fold of the actor in the high half, the per-actor counter in the
    /// low.
    ///
    /// [`ActorId::SOLO`] maps to high half 0, so a document that was never shared
    /// keeps the small, readable ids it always had — including the root layer's
    /// `LayerId(0)`, which every peer must agree on because it predates any actor.
    pub fn mint(actor: ActorId, n: u64) -> Self {
        let hi = if actor == ActorId::SOLO {
            0
        } else {
            // Never 0: that is SOLO's space, and colliding with it would clash with
            // the layers a document had before it was ever shared.
            mix32(actor.0).max(1)
        };
        LayerId((u64::from(hi) << 32) | (n & 0xFFFF_FFFF))
    }

    /// The per-actor counter this id was minted from — the inverse of the low half
    /// of [`mint`](Self::mint).
    pub fn ordinal(self) -> u64 {
        self.0 & 0xFFFF_FFFF
    }

    /// Whether this id was minted by `actor`, so the engine can resume that actor's
    /// counter from a log without also resuming everyone else's.
    pub fn minted_by(self, actor: ActorId) -> bool {
        self.0 >> 32 == Self::mint(actor, 0).0 >> 32
    }
}

/// Where in a stack a layer lands — the anchor half of a structural move
/// (§14.8).
///
/// [`Above`](Self::Above) names a sibling, and a stack of `n` layers has `n + 1`
/// places to land in, so naming siblings covers all of them but one: the place
/// **under the bottom layer**, which has no sibling below it to be named after.
/// [`Bottom`](Self::Bottom) is that place. Without it a panel could offer every
/// drop position in a stack except its foot — and "put this behind everything"
/// is not an exotic move, it is where a background goes.
///
/// The variant order is load-bearing. This replaced an `Option<LayerId>`, and
/// postcard writes an `Option` as `0` for `None` / `1` for `Some` and an enum as
/// its variant index — so `Top` and `Above` encode exactly as the `None` and
/// `Some` they stand in for, and `Bottom` is an *appended* third variant. That
/// is the one shape of change §8 allows without a format break, and
/// `place_encodes_as_the_option_it_replaced` below is what keeps the
/// claim honest.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Place {
    /// On top of the stack, over everything already in it.
    Top,
    /// Directly above this sibling — or on top, if it is not in this stack.
    Above(LayerId),
    /// At the foot of the stack, under everything already in it.
    Bottom,
}

impl Place {
    /// The sibling this place is stated against, if any — what a footprint has to
    /// name as read (§12.6), since where the move lands depends on where that
    /// layer is.
    pub fn anchor(self) -> Option<LayerId> {
        match self {
            Place::Above(id) => Some(id),
            Place::Top | Place::Bottom => None,
        }
    }
}

impl From<Option<LayerId>> for Place {
    /// The old two-state anchor, which is still what insertion takes: a named
    /// sibling, or the top of the stack.
    fn from(above: Option<LayerId>) -> Self {
        match above {
            Some(id) => Place::Above(id),
            None => Place::Top,
        }
    }
}

/// splitmix64's finalizer, folded to 32 bits: decorrelates the bits an
/// endpoint-derived [`ActorId`] takes verbatim from a public key.
fn mix32(x: u64) -> u32 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 32) as u32
}

/// How a layer combines with the layers below it (§18.0.4).
///
/// Everything past `Normal` combines the two layers' **light** rather than covering
/// one with the other — and none of it is Screen. Screen is `a + b − ab`, which is
/// what falls out of inverting a multiply; it describes no physical process, and it
/// crushes the top of the range into a flat, chalky white that is the giveaway of a
/// digital glow.
///
/// Ours are derived the other way round. Two lights *add* — that is the only thing
/// light does — but the numbers in a layer are not light, they are light that has
/// already been through a tone curve on its way to being displayable. So the honest
/// combination is: undo the curve, add, re-apply it. Every mode here is that same
/// sentence with a different curve `T`:
///
/// ```text
///     f(a, b) = T(T⁻¹(a) + T⁻¹(b))
/// ```
///
/// Being a conjugation of addition is not a technicality — it is the whole
/// guarantee. Each mode is commutative and associative with a neutral element, so
/// three glowing layers give the same result in any order and regrouping them
/// changes nothing, exactly as three real lamps would. Screen happens to share those
/// properties (it is addition conjugated by `1 − e^{-x}`'s cousin), which is *why*
/// it survived; these are what you get when the curve is chosen for how light
/// actually rolls off instead of for algebraic convenience.
///
/// **The guarantee holds at any coverage**, which took getting right: a layer's
/// coverage weighs its *emission* — the quantity that adds — rather than its
/// working-space colour, because applying the curve to a coverage-averaged colour is
/// not the same as averaging the curve, and stacking order used to matter by up to 20
/// levels wherever a stroke was less than solid. See `blend_common.wesl`'s
/// `added_light`, and §18.0.4 for what that costs.
///
/// [`Reinhard`](Self::Reinhard) and [`Drago`](Self::Drago) are the emissive half:
/// they add light and their identity is black. [`Multiply`](Self::Multiply) is the
/// subtractive half — the same construction with `T(x) = e^{-x}`, which makes the
/// added quantity optical density and the identity white. That is the *whole* of
/// what changes between the two halves; the family is one idea, not two — though the
/// halves do weigh coverage in different quantities, each in the one that adds for
/// it: averaging light over an area is linear in light, averaging transmitted light
/// is linear in transmittance, so `Multiply` weighs coverage where it always did.
///
/// The combination happens in **CIE XYZ normalized to the display white**, not in
/// the working colour space and not in RGB: XYZ is linear in light, its components
/// are non-negative for every real colour (which is what makes the curves
/// well-defined), and normalizing by the white point puts an in-gamut colour's
/// components in `[0,1]` — so "1" means the same thing on all three axes. Blending
/// in RGB instead would make the result depend on the display's primaries; blending
/// in Oklab or in pigment concentrations would be adding things that are not light.
///
/// See `blend_common.wesl` for the derivations and `Compositor` for the isolation
/// pass that makes per-layer blending possible at all.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    /// Premultiplied "over": the layer sits on top of what is below it.
    #[default]
    Normal,
    /// **Glow** — addition conjugated by the Reinhard tonemap `T(x) = x/(1+x)`,
    /// which collapses to
    ///
    /// ```text
    ///     f(a, b) = (a + b − 2ab) / (1 − ab)
    /// ```
    ///
    /// Reinhard's curve is asymptotic: no finite amount of light reaches 1. So this
    /// mode **cannot blow out** — stack a hundred glow layers and the result
    /// approaches white without ever clipping, and detail survives everywhere. That
    /// makes it the one to reach for on glazes, mist, rim light and bloom, where
    /// Screen's flat white is exactly the failure.
    Reinhard,
    /// **Radiance** — addition conjugated by Drago's log curve
    /// `T(x) = k·log(1 + x/k)`, which collapses to
    ///
    /// ```text
    ///     f(a, b) = k·log(e^{a/k} + e^{b/k} − 1)
    /// ```
    ///
    /// A log curve has no asymptote, so unlike [`Reinhard`](Self::Reinhard) this one
    /// *does* push past display white where two strong lights coincide — and that
    /// overflow is the point. The composite targets are half-float, so the excess
    /// survives into the media pass and comes back through its highlight roll-off
    /// (§6.3) as a genuine bloom with a filmic shoulder, rather than being
    /// clipped at the blend. Reach for it on flame, specular hits, anything meant to
    /// read as *brighter than the paper*.
    ///
    /// `k` sets how quickly the curve bends: large `k` tends to plain addition,
    /// small `k` tends to `max`. It is fixed at [`DRAGO_K`] rather than exposed —
    /// per-layer blend parameters are the seam a future mapping UI lands on, and
    /// this codebase's precedent (§1) is that no knob appears before something turns it.
    Drago,
    /// **Multiply** — the same construction read the other way round, with
    /// `T(x) = e^{-x}`, which collapses to
    ///
    /// ```text
    ///     f(a, b) = a·b
    /// ```
    ///
    /// The quantity being added is **optical density**, so this is Beer-Lambert:
    /// what two stacked filters, two glazes, or two sheets of stained glass do to
    /// the light passing through them. It is the mode Screen is an inversion *of* —
    /// and of the two it is the one that describes something real, which is why this
    /// is here and Screen is not.
    ///
    /// Everything the emissive modes guarantee still holds, dualised: commutative
    /// and associative, so a stack of glazes is order-independent, but the neutral
    /// element is **white** rather than black. Glaze over bare paper and nothing
    /// happens; glaze over black and nothing shows. Because it runs in normalized
    /// XYZ rather than in RGB, the darkening is a statement about light rather than
    /// about the display's primaries — two saturated glazes cross without the dead
    /// channel that an RGB multiply produces when one primary happens to be near
    /// zero.
    ///
    /// The one mode here that *removes* light, and so the one that never reaches the
    /// media pass's highlight roll-off: its output is in `[0,1]` by construction.
    ///
    /// One consequence to know about. The blend sees the layer stack, not the
    /// **substrate** — the paper is composited in pass B, after all blending
    /// (`media_common.wesl`) — so a glaze laid on bare canvas leaves the paper's own
    /// colour untouched instead of tinting it. On white paper that is exactly right,
    /// white being multiply's identity, and it is why the mode reads correctly to a
    /// painter by default. On a toned ground it is a divergence from what a real
    /// glaze would do, and the fix is not here: it is for the substrate to become the
    /// bottom of the stack rather than a step of the media pass.
    Multiply,
}

/// The bend of [`BlendMode::Drago`]'s log curve, in units of display white. Large
/// `k` tends to plain addition, small `k` tends to `max`.
///
/// Chosen so the two light modes are a genuine choice rather than two settings of
/// one. Take two half-lit layers: [`BlendMode::Reinhard`] gives 0.667, Screen gives
/// 0.75, plain addition gives 1.0 (clipped), and this gives 0.769. So Glow reads
/// distinctly softer than the mode everyone already knows and Radiance reads
/// distinctly hotter, across the whole range instead of only at the extremes — which
/// is what a value near 0.35 gave, and the reason it is not that. At the top, two
/// whites come out at ≈1.36, well into the media pass's highlight roll-off.
pub const DRAGO_K: f32 = 0.6;

impl BlendMode {
    /// Every mode, in the order a frontend should offer them: `Normal` first, then
    /// increasingly emphatic light, then the one that takes light away.
    pub const ALL: [BlendMode; 4] = [Self::Normal, Self::Reinhard, Self::Drago, Self::Multiply];

    /// What this mode is called. The painter-facing name, not the tonemap's — the
    /// curve is how it is *built*, not what it is *for*.
    ///
    /// `Multiply` is the exception that proves it: there the operation's name and the
    /// painter's name are the same word, and it has been that word in every paint
    /// program for thirty years. Renaming it "Glaze" to match its neighbours would be
    /// inventing a synonym for a term of art nobody needs translated.
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Reinhard => "Glow",
            Self::Drago => "Radiance",
            Self::Multiply => "Multiply",
        }
    }

    /// Whether this mode composites under plain premultiplied "over".
    ///
    /// The compositor's fast path: a run of consecutive `Normal` layers needs no
    /// isolation and draws straight into the accumulator, so an ordinary document
    /// costs exactly what it did before blend modes existed (§6.3).
    pub fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }
}

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

/// The region a matte layer fills (§15.2).
///
/// A region is a coverage field over the *infinite* plane, so what matters is its
/// value at infinity — which is what makes the frame case (fill everywhere except
/// a rect) expressible at all, and expressible without a mask.
///
/// It is stored as **geometry, not a rasterized mask**: the fill is evaluated
/// analytically from a signed distance at canvas position, exactly as
/// `selection.wesl` does (§6.8). That costs no tiles (a 4000² frame would
/// otherwise be ~16 MB of mask and could trip `MAX_SELECTION_TILES`), stays exact
/// at any zoom, keeps the log to four floats, and — being a pure function of
/// canvas position — satisfies the §6.4 seam invariant for free.
///
/// One variant, because one is built. This is the seam where the `SelectionOp`
/// algebra lands (§15.9, P4), bringing comic gutters, lasso mattes
/// and whole-plane slabs at once. Per this codebase's own precedent (§1 —
/// `drag` and `wetness` were deleted rather than kept inert, and `bleed` and
/// `tooth` came back only once each had a model), no variant appears here before
/// it does something.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MatteRegion {
    /// Everything *outside* this canvas-space rect — the frame / mat board.
    OutsideRect { min: Vec2, max: Vec2 },
}

impl MatteRegion {
    /// The rect this region is defined against, in canvas px. For `OutsideRect`
    /// this is the *hole* — the piece — which is what export frames against
    /// (§15.6).
    pub fn rect(&self) -> (Vec2, Vec2) {
        match self {
            Self::OutsideRect { min, max } => (*min, *max),
        }
    }

    /// The same region with its rect replaced (the frame drag's commit).
    pub fn with_rect(&self, min: Vec2, max: Vec2) -> Self {
        match self {
            Self::OutsideRect { .. } => Self::OutsideRect { min, max },
        }
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
}

impl PaintTiles {
    /// The tiles, with their extent derived once.
    fn new(map: TileMap) -> Self {
        Self {
            bounds: CanvasBounds::of_tiles(map.keys()),
            map,
        }
    }

    /// The sparse tile map itself.
    pub fn map(&self) -> &TileMap {
        &self.map
    }
}

/// What a layer is made of (§15.2).
#[derive(Clone)]
pub enum LayerContent {
    /// Painted tiles. Only populated ones exist — this sparsity is the infinite
    /// canvas.
    Paint(PaintTiles),
    /// A procedural region filled with a flat colour.
    ///
    /// `color` is **straight sRGB**, like [`BrushParams::color`], and is converted
    /// to working-space channels at composite time — so the log stays independent
    /// of whether the document is Oklab or Mixbox. A matte has no alpha of its
    /// own: its transparency *is* its layer opacity, which is the whole point of
    /// it being a layer.
    ///
    /// Physically this is a flat, opaque *coat of paint*: the compositor gives it
    /// a constant thickness, so its interior lights flat (zero height gradient —
    /// no weave, and the paint film's uniform sheen reads as an even wash rather
    /// than a glint) while its boundary catches light the same way any stroke edge
    /// does. See §15.4 for why it must write the aux target at all,
    /// and why its blend there is `over` rather than additive.
    ///
    /// [`BrushParams::color`]: crate::document::BrushParams::color
    Matte {
        region: MatteRegion,
        color: [f32; 3],
    },
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
    /// (§21.4). Its blend mode and clip, by contrast, have nothing to say: both
    /// describe how a *source* meets a backdrop, and a filter has no source — it
    /// *is* the backdrop, rewritten (§21.4).
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
/// Opacity sits in the first group, and it took a bug to establish that it belongs
/// there. It reads as intrinsic — "how faded is this layer" — but it is applied at
/// the same step as the other two, to the same thing: the group's composited whole
/// (§14.7). Held separately, it was applied to the base's own content *as well*, and
/// a group base at 0.5 drew its paint at 0.25. Now the base composites with
/// [`CompositeParams::IDENTITY`] and there are no longer three chances to get that
/// wrong.
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

    /// A matte layer over `region`, filled with `color` (working-space channels).
    pub fn matte(id: LayerId, region: MatteRegion, color: [f32; 3]) -> Self {
        Self {
            content: LayerContent::Matte { region, color },
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

    /// The matte region this layer fills, if it is a matte.
    pub fn matte_region(&self) -> Option<MatteRegion> {
        match &self.content {
            LayerContent::Matte { region, .. } => Some(*region),
            LayerContent::Paint(_) | LayerContent::Filter(_) => None,
        }
    }

    /// The filter this layer runs over the stack beneath it, if it is one (§21).
    pub fn filter(&self) -> Option<Filter> {
        match &self.content {
            LayerContent::Filter(f) => Some(*f),
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
    /// tiles to replace.
    pub fn with_tiles(&self, tiles: TileMap) -> Self {
        match &self.content {
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
    pub fn visit(&self, depth: usize, f: &mut impl FnMut(&Layer, usize)) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// [`Place`] replaced an `Option<LayerId>` in a logged action, and it claims to
    /// have done so **without changing a byte** of what was already written: postcard
    /// encodes an `Option` as a `0`/`1` discriminant and an enum as its variant index,
    /// so the two existing cases had to keep their indices and the new one had to be
    /// appended (§8).
    ///
    /// Asserted rather than reasoned about, because the failure is silent in exactly
    /// the way §8 warns: reorder the variants and every `MoveLayer` in every saved
    /// document decodes as a *different* move, with nothing in the file able to notice.
    #[test]
    fn place_encodes_as_the_option_it_replaced() {
        let id = LayerId(0x1234_5678_9ABC_DEF0);
        assert_eq!(
            postcard::to_allocvec(&Place::Top).expect("encodes"),
            postcard::to_allocvec(&None::<LayerId>).expect("encodes"),
            "Top must occupy None's discriminant"
        );
        assert_eq!(
            postcard::to_allocvec(&Place::Above(id)).expect("encodes"),
            postcard::to_allocvec(&Some(id)).expect("encodes"),
            "Above must occupy Some's discriminant, payload and all"
        );
        // And the new case is appended, so nothing that existed had to move.
        assert_eq!(
            postcard::to_allocvec(&Place::Bottom).expect("encodes"),
            vec![2u8],
            "Bottom is the third variant"
        );
    }
}
