//! The document: versioned state and the actions that produce it (DESIGN.md §4, §5).

pub mod action;
pub mod layer;
pub mod selection;
pub mod state;
pub mod timeline;

pub use action::{
    Action, ActionId, ActionKind, ActorId, ApplyCtx, BrushDynamics, BrushParams, BrushShape,
    ColorDynamics, NoiseKind, OrientationSource, StrokeRecord, Tool,
};
pub use layer::{BlendMode, Layer, LayerContent, LayerId, MatteRegion};
pub use selection::{Selection, SelectionMode, SelectionOp, SelectionShape};
pub use state::{CanvasBounds, DocState};
pub use timeline::{LinearTimeline, ReplicatedTimeline, Timeline, effective_actions};
