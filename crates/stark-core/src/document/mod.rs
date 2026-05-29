//! The document: versioned state and the actions that produce it (DESIGN.md §4, §5).

pub mod action;
pub mod layer;
pub mod state;
pub mod timeline;

pub use action::{
    Action, ActionId, ActionKind, ActorId, ApplyCtx, BrushParams, BrushShape, StrokeRecord, Tool,
};
pub use layer::{BlendMode, Layer, LayerId};
pub use state::{CanvasBounds, DocState};
pub use timeline::{LinearTimeline, Timeline};
