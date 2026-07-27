//! Layers (DESIGN.md §5.1, FRAME_DESIGN.md §2). A layer is either a sparse,
//! persistent map of painted tiles or a **matte** — a procedural region filled
//! with a flat colour — plus its presentation properties. Layer compositing
//! across blend modes arrives in step 4; for now layers stack with `Normal` over.

use rpds::HashTrieMap;
use serde::{Deserialize, Serialize};

use crate::geom::{TileCoord, Vec2};
use crate::gpu::tile::TilePairHandle;

/// Stable identifier for a layer within a document.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerId(pub u64);

/// How a layer combines with the layers below it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    Normal,
}

/// The region a matte layer fills (FRAME_DESIGN.md §2).
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
/// algebra lands (FRAME_DESIGN.md §9, P4), bringing comic gutters, lasso mattes
/// and whole-plane slabs at once. Per DESIGN's own precedent (`tooth`, `drag`,
/// `bleed` were deleted rather than kept inert), no variant appears here before
/// it does something.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MatteRegion {
    /// Everything *outside* this canvas-space rect — the frame / mat board.
    OutsideRect { min: Vec2, max: Vec2 },
}

impl MatteRegion {
    /// The rect this region is defined against, in canvas px. For `OutsideRect`
    /// this is the *hole* — the piece — which is what export frames against
    /// (FRAME_DESIGN.md §6).
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

/// What a layer is made of (FRAME_DESIGN.md §2).
#[derive(Clone)]
pub enum LayerContent {
    /// Painted tiles. Only populated ones exist — this sparsity is the infinite
    /// canvas.
    Paint(HashTrieMap<TileCoord, TilePairHandle>),
    /// A procedural region filled with a flat colour.
    ///
    /// `color` is **straight sRGB**, like [`BrushParams::color`], and is converted
    /// to working-space channels at composite time — so the log stays independent
    /// of whether the document is Oklab or Mixbox. A matte has no alpha of its
    /// own: its transparency *is* its layer opacity, which is the whole point of
    /// it being a layer.
    ///
    /// Physically this is a flat, opaque *coat of paint*: the compositor gives it
    /// a constant thickness, so its interior lights flat and matte (zero height
    /// gradient — no weave, no gloss) while its boundary catches light the same
    /// way any stroke edge does. See FRAME_DESIGN.md §4 for why it must write the
    /// aux target at all, and why its blend there is `over` rather than additive.
    ///
    /// [`BrushParams::color`]: crate::document::BrushParams::color
    Matte { region: MatteRegion, color: [f32; 3] },
}

/// A single layer: its content plus its presentation properties.
#[derive(Clone)]
pub struct Layer {
    pub id: LayerId,
    pub blend: BlendMode,
    /// Layer opacity in [0, 1].
    pub opacity: f32,
    /// Whether the layer contributes to the composite.
    pub visible: bool,
    pub content: LayerContent,
}

impl Layer {
    /// An empty paint layer.
    pub fn new(id: LayerId) -> Self {
        Self {
            id,
            blend: BlendMode::Normal,
            opacity: 1.0,
            visible: true,
            content: LayerContent::Paint(HashTrieMap::new()),
        }
    }

    /// A matte layer over `region`, filled with `color` (working-space channels).
    pub fn matte(id: LayerId, region: MatteRegion, color: [f32; 3]) -> Self {
        Self {
            content: LayerContent::Matte { region, color },
            ..Self::new(id)
        }
    }

    /// This layer's painted tiles, or `None` if it is a matte. Deliberately an
    /// `Option` rather than an empty map: "this layer has no tiles" is a real
    /// fact about it, and making callers say what they do about it is what keeps
    /// a matte from silently reading as an empty paint layer.
    pub fn tiles(&self) -> Option<&HashTrieMap<TileCoord, TilePairHandle>> {
        match &self.content {
            LayerContent::Paint(tiles) => Some(tiles),
            LayerContent::Matte { .. } => None,
        }
    }

    /// The matte region this layer fills, if it is a matte.
    pub fn matte_region(&self) -> Option<MatteRegion> {
        match &self.content {
            LayerContent::Matte { region, .. } => Some(*region),
            LayerContent::Paint(_) => None,
        }
    }

    /// Whether strokes may be painted onto this layer. A matte has no tile map,
    /// so a stroke targeting one is refused rather than silently swallowed or
    /// magically rasterized (FRAME_DESIGN.md §7).
    pub fn is_paintable(&self) -> bool {
        matches!(self.content, LayerContent::Paint(_))
    }

    /// The same layer with its painted tiles replaced. A no-op on a matte.
    pub fn with_tiles(&self, tiles: HashTrieMap<TileCoord, TilePairHandle>) -> Self {
        match &self.content {
            LayerContent::Paint(_) => Self {
                content: LayerContent::Paint(tiles),
                ..self.clone()
            },
            LayerContent::Matte { .. } => self.clone(),
        }
    }
}
