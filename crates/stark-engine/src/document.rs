//! The document: versioned state and the actions that produce it (§4, §5).
//!
//! The **derived** half of the document (§2): the state actions fold into, and the
//! fold that fills it. The actions themselves, and the vocabulary they are written
//! in, are `stark-model`'s `document`.
//!
//! A GPU pass's plan is not here — it lives beside the pass, in `gpu/`. What stays is
//! the mask's own op algebra, `Selection::plan`, which produces the field rather than
//! a draw list.
//!
//! The submodules are **crate-private**. A frontend needs the two names re-exported
//! `pub` below; the suite reaches the state itself through [`testing`](crate::testing).

pub(crate) mod apply;
/// The §12.6 rule, checked on every fold of every debug build — see the module.
pub(crate) mod audit;
/// The drawing-guide roster (§20.5) — see the module.
pub(crate) mod guides;
pub(crate) mod layer;
pub(crate) mod liquify;
pub(crate) mod merge;
pub(crate) mod patch;
pub(crate) mod selection;
pub(crate) mod state;
pub(crate) mod timeline;

pub use state::{CanvasBounds, DEFAULT_SUBSTRATE_COLOR};

pub(crate) use apply::{ApplyCtx, PreparedStroke};
pub(crate) use guides::Guide;
pub(crate) use layer::{CompositeParams, Layer, LayerContent};
pub(crate) use selection::Selection;
pub(crate) use state::{DEFAULT_SUBSTRATE, DocState};
pub(crate) use timeline::{LinearTimeline, ReplicatedTimeline, Step, Stepped, Timeline};
