//! The document: the actions that produce it, and the vocabulary they are written in
//! (§4, §5).
//!
//! The submodules are **crate-private**, and the re-exports below are the API — the
//! same rule `stark-engine`'s `document` keeps: publishing both would give every type
//! two paths, with nothing choosing between them.
//!
//! What is *not* here is the state these actions fold into. `DocState`, `Layer`'s
//! tiles, the `Selection`'s mask and the `Timeline` that materializes them are all
//! `stark-engine`'s, because they hold pixels (§2). Four modules are split down the
//! middle by that line and keep the same file name on both sides:
//!
//! | module | here | there |
//! |---|---|---|
//! | `layer` | `LayerId`, `BlendMode`, `Place`, `Parcel` | `Layer`, `LayerContent`, `PaintTiles` |
//! | `selection` | `SelectionOp` and its shapes | `Selection`, the mask itself |
//! | `fill` | `FillOp`, `fill_bounds` | `plan`, which needs the mask |
//! | `transform` | the maps, and the homography solve | the tile plans |
//!
//! `guide` is the one that is *not* split (§20.5). A drawing guide is document
//! state — logged, saved, replicated, undoable — and everything derived from one
//! is a pure function of the camera, so the derivations sit here beside the fact
//! for the reason `fill_bounds` and the homography solve do. What the engine
//! keeps is the roster's per-client half: whose eye is shut, and the packing of a
//! `GuideScene` into the guide pass's uniform.

pub(crate) mod action;
pub(crate) mod brush;
pub(crate) mod effect;
pub(crate) mod fill;
pub(crate) mod filter;
pub(crate) mod fold;
pub(crate) mod footprint;
pub(crate) mod guide;
pub(crate) mod image;
pub(crate) mod layer;
pub(crate) mod paint;
pub(crate) mod selection;
pub(crate) mod transform;
pub(crate) mod warp;

pub use action::{Action, ActionId, ActionKind, ActionTag, ActorId, StrokeRecord};
pub use brush::{
    BrushDynamics, BrushEffect, BrushModulations, BrushParams, BrushShape, ColorDynamics,
    EraseEffect, EraseModulations, ModSource, Modulation, NoiseKind, OrientationSource,
    PaintEffect, PaintModulations, PenState, ToothParams,
};
pub use fill::{FillOp, MAX_FILL_TILES, ShapeAction};
pub use filter::{ChromaticAberration, ColorAdjust, Filter};
// The undo algebra (§12.3): which actions in a log are effective. The helpers behind
// these — the revival keys, the two target searches — stay inside the module.
pub use effect::{
    Targets, effective_actions, effective_indices, targets, undo_target_of, undone_ids,
};
pub use fold::{Logged, Materialize};
/// The commutation vocabulary (§12.6) — what an action reads and writes, and
/// whether two of them can be reordered.
pub use footprint::{Footprint, Prop, Resource, compute_footprint, fill_rect, stroke_rect};
pub use guide::{
    AxisPencil, AxisPlane, GuideId, GuideScene, Lens, PairTrace, PerspectiveGuide, Scaffold,
};
pub use image::{MAX_IMAGE_TILES, image_tiles};
pub use layer::{BlendMode, DRAGO_K, DRAGO_K_RANGE, LayerId, MatteRegion, Place};
pub use paint::{GradientAxis, GradientParcel, Parcel};
pub use selection::{
    MAX_LASSO_POINTS, MAX_SELECTION_TILES, SelectionMode, SelectionOp, SelectionShape,
};
pub use transform::{
    Homography, MAX_TRANSFORM_TILES, PerspectiveMap, TransformMap, affine_usable, rect_corners,
};
pub use warp::{Lattice, MAX_WARP_GRID, Prepared, WarpMap, cell_point};

/// The tiles a fill would write, as a canvas box — `None` when it would be unbounded.
///
/// `pub` where its neighbours are `pub(crate)` because the planner that consumes it
/// is in the other crate (§2), and so is the pass that must write exactly the tiles
/// this names: a fill's written tiles and its footprint have to be *the same* tiles
/// (§12.6), so there is one derivation and both sides call it.
pub use fill::fill_bounds;
