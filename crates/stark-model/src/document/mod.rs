//! The document: the actions that produce it, and the vocabulary they are written in
//! (§4, §5).
//!
//! The submodules are **crate-private**, and the re-exports below are the API — the
//! same rule `stark-engine`'s `document` keeps, and for the same reason: publishing
//! both gives every type two paths, `document::BrushParams` and
//! `document::brush::BrushParams`, with nothing choosing between them.
//!
//! What is *not* here is the state these actions fold into. `DocState`, `Layer`'s
//! tiles, the `Selection`'s mask and the `Timeline` that materializes them are all
//! `stark-engine`'s, because they hold pixels (§2). Four modules are split down the
//! middle by that line and keep the same file name on both sides:
//!
//! | module | here | there |
//! |---|---|---|
//! | `layer` | `LayerId`, `BlendMode`, `Place`, `MattePaint` | `Layer`, `LayerContent`, `PaintTiles` |
//! | `selection` | `SelectionOp` and its shapes | `Selection`, the mask itself |
//! | `fill` | `FillOp`, `fill_bounds` | `plan`, which needs the mask |
//! | `transform` | the maps, and the homography solve | the tile plans |

pub(crate) mod action;
pub(crate) mod brush;
pub(crate) mod fill;
pub(crate) mod filter;
pub(crate) mod fold;
pub(crate) mod footprint;
pub(crate) mod layer;
pub(crate) mod selection;
pub(crate) mod transform;
pub(crate) mod warp;

pub use action::{Action, ActionId, ActionKind, ActorId, StrokeRecord, Tool};
pub use brush::{
    BrushDynamics, BrushParams, BrushShape, ColorDynamics, ModSource, Modulation, Modulations,
    NoiseKind, OrientationSource, PenState,
};
pub use fill::{FillOp, GradientAxis, GradientParcel, MAX_FILL_TILES, Parcel, ShapeAction};
pub use filter::{ChromaticAberration, ColorAdjust, Filter};
pub use fold::{Logged, Materialize};
/// The commutation vocabulary (§12.6) — what an action reads and writes, and
/// whether two of them can be reordered.
pub use footprint::{Footprint, Prop, Resource, fill_rect, footprint, stroke_rect};
pub use layer::{BlendMode, DRAGO_K, DRAGO_K_RANGE, LayerId, MattePaint, MatteRegion, Place};
pub use selection::{MAX_SELECTION_TILES, SelectionMode, SelectionOp, SelectionShape};
pub use transform::{
    Homography, MAX_TRANSFORM_TILES, PerspectiveMap, TransformMap, affine_usable, rect_corners,
};
pub use warp::{Lattice, MAX_WARP_GRID, Prepared, WarpMap, cell_point};

/// The tiles a fill would write, as a canvas box — `None` when it would be unbounded.
///
/// `pub` where its neighbours are `pub(crate)` because the planner that consumes it
/// is in the other crate now (§2), and so is the pass that must write exactly the
/// tiles this names. That it is one function reaching across is the point: a fill's
/// written tiles and its footprint have to be *the same* tiles (§12.6), so there is
/// one derivation of them and both sides call it.
pub use fill::fill_bounds;
