//! The document: versioned state and the actions that produce it (§4, §5).
//!
//! The submodules are **crate-private**, and the re-exports below are the API.
//! Publishing both gave every type two paths — `document::BrushParams` and
//! `document::brush::BrushParams` — with nothing choosing between them, and the
//! list here is the one that says what the module offers rather than how it is
//! filed. Inside the crate the paths still work, which is what lets `gpu/` reach
//! the `pub(crate)` planners (`fill::plan`, `selection::SelectionPlan`) that were
//! never part of this list to begin with.

pub(crate) mod action;
pub(crate) mod brush;
pub(crate) mod fill;
pub(crate) mod filter;
pub(crate) mod footprint;
pub(crate) mod layer;
pub(crate) mod patch;
pub(crate) mod selection;
pub(crate) mod state;
pub(crate) mod timeline;
pub(crate) mod transform;
pub(crate) mod warp;

pub use action::{Action, ActionId, ActionKind, ActorId, ApplyCtx, StrokeRecord, Tool};
pub use brush::{
    BrushDynamics, BrushParams, BrushShape, ColorDynamics, ModSource, Modulation, Modulations,
    NoiseKind, OrientationSource, PenState,
};
pub use fill::{FillOp, MAX_FILL_TILES, ShapeAction};
pub use filter::{CONTRAST_PIVOT, ChromaticAberration, ColorAdjust, Filter};
/// The commutation vocabulary (§12.6) — what an action reads and writes, and
/// whether two of them can be reordered.
pub use footprint::{Footprint, Prop, Resource, footprint};
pub use layer::{BlendMode, DRAGO_K, Layer, LayerContent, LayerId, MatteRegion, PaintTiles, Place};
pub use selection::{Selection, SelectionMode, SelectionOp, SelectionShape};
pub use state::{CanvasBounds, DEFAULT_BACKGROUND, DEFAULT_SURFACE, DocState, LayerSite};
pub use timeline::{
    LinearTimeline, ReplicatedTimeline, Timeline, TimelineStats, effective_actions,
};
pub use transform::{
    Homography, MAX_TRANSFORM_TILES, PerspectiveMap, TransformMap, affine_usable, rect_corners,
};
pub use warp::{MAX_WARP_GRID, WarpMap};
