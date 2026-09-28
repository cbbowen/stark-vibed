//! The document: the state an action log folds into (§4, §5).
//!
//! The **derived** half of the document (§2). The actions themselves, and the
//! vocabulary they are written in, are `stark-model`'s `document`; the folding is
//! [`fold`](crate::fold)'s.
//!
//! **Nothing here names a renderer, and that is load-bearing** — the layering puts this
//! module below [`gpu`](crate::gpu) and above [`device`](crate::device) alone, so a
//! layer may hold a tile handle while the pass that filled it sits above (§2). The two
//! things the split cost are stated where they went: a GPU pass's plan lives beside the
//! pass, and the fold that needs every renderer lives in `fold/`. What stays is the
//! mask's own op algebra, `Selection::plan`, which produces the field rather than a
//! draw list.
//!
//! The submodules are **crate-private**. A frontend needs the two names re-exported
//! `pub` below; the suite reaches the state itself through [`testing`](crate::testing).

/// The drawing-guide roster (§20.5) — see the module.
pub(crate) mod guides;
pub(crate) mod layer;
pub(crate) mod liquify;
pub(crate) mod patch;
pub(crate) mod selection;
pub(crate) mod state;
/// Which substrate the document stands on and how large — see the module.
pub(crate) mod substrate;

pub use state::{CanvasBounds, DEFAULT_SUBSTRATE_COLOR};

pub(crate) use guides::Guide;
pub(crate) use layer::{CompositeParams, Layer, LayerContent, PaintTiles};
pub(crate) use selection::Selection;
pub(crate) use state::{DEFAULT_SUBSTRATE, DocState};
pub(crate) use substrate::Substrate;
