//! The document: versioned state and the actions that produce it (§4, §5).

pub mod action;
pub mod brush;
pub mod fill;
pub mod footprint;
pub mod layer;
pub mod patch;
pub mod selection;
pub mod state;
pub mod timeline;
pub mod transform;
pub mod warp;

pub use action::{Action, ActionId, ActionKind, ActorId, ApplyCtx, StrokeRecord, Tool};
pub use brush::{
    BrushDynamics, BrushParams, BrushShape, ColorDynamics, ModSource, Modulation, Modulations,
    NoiseKind, OrientationSource, PenState,
};
pub use fill::{FillOp, MAX_FILL_TILES, ShapeAction};
pub use layer::{BlendMode, DRAGO_K, Layer, LayerContent, LayerId, MatteRegion, PaintTiles};
pub use selection::{Selection, SelectionMode, SelectionOp, SelectionShape};
pub use state::{CanvasBounds, DEFAULT_BACKGROUND, DEFAULT_SURFACE, DocState, LayerSite};
pub use timeline::{
    LinearTimeline, ReplicatedTimeline, Timeline, TimelineStats, effective_actions,
};
pub use transform::{
    Homography, MAX_TRANSFORM_TILES, PerspectiveMap, TransformMap, affine_usable, rect_corners,
};
pub use warp::{MAX_WARP_GRID, WarpMap};
