//! The document: versioned state and the actions that produce it (§4, §5).

pub mod action;
pub mod fill;
pub mod footprint;
pub mod layer;
pub mod patch;
pub mod selection;
pub mod state;
pub mod timeline;
pub mod transform;

pub use action::{
    Action, ActionId, ActionKind, ActorId, ApplyCtx, BrushDynamics, BrushParams, BrushShape,
    ColorDynamics, NoiseKind, OrientationSource, StrokeRecord, Tool,
};
pub use fill::{FillOp, MAX_FILL_TILES, ShapeAction};
pub use layer::{BlendMode, DRAGO_K, Layer, LayerContent, LayerId, MatteRegion};
pub use selection::{Selection, SelectionMode, SelectionOp, SelectionShape};
pub use state::{CanvasBounds, DEFAULT_BACKGROUND, DEFAULT_SURFACE, DocState, LayerSite};
pub use timeline::{
    LinearTimeline, ReplicatedTimeline, Timeline, TimelineStats, effective_actions,
};
pub use transform::{MAX_TRANSFORM_TILES, affine_usable};
