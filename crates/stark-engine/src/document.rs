//! The document: versioned state and the actions that produce it (§4, §5).
//!
//! The **derived** half of the document (§2): the state actions fold into, and the
//! planning that needs it. The actions themselves, and the vocabulary they are
//! written in, are `stark-model`'s `document`.
//!
//! The submodules are **crate-private**, and the re-exports below are the API.
//! Publishing both gave every type two paths — `document::BrushParams` and
//! `document::brush::BrushParams` — with nothing choosing between them, and the
//! list here is the one that says what the module offers rather than how it is
//! filed. Inside the crate the paths still work, which is what lets `gpu/` reach
//! the `pub(crate)` planners (`fill::plan`, `selection::SelectionPlan`) that were
//! never part of this list to begin with.

pub(crate) mod apply;
/// The §12.6 rule, checked on every fold of every debug build — see the module.
pub(crate) mod audit;
pub(crate) mod fill;
pub(crate) mod layer;
pub(crate) mod merge;
pub(crate) mod patch;
pub(crate) mod selection;
pub(crate) mod state;
pub(crate) mod timeline;
pub(crate) mod transform;

pub use apply::{ApplyCtx, PreparedStroke};
#[doc(hidden)]
pub use audit::undeclared;
pub use layer::{CompositeParams, Layer, LayerContent, PaintTiles};
/// Merging a layer down onto the one beneath it (§14.11) — the rule for when that
/// leaves the document looking the same, which is the whole of what a merge promises.
pub use merge::MergePlan;
pub use selection::Selection;
pub use state::{
    CanvasBounds, DEFAULT_SUBSTRATE, DEFAULT_SUBSTRATE_COLOR, DocState, Guide, LayerSite,
};
pub use timeline::{
    LinearTimeline, ReplicatedTimeline, Timeline, TimelineStats, effective_actions,
};
