//! The fold: what happens when an action is applied, and the timeline that orders the
//! applying (§4, §5).
//!
//! [`document`](crate::document) is the state a log folds *into*; this is the folding.
//! The split is where the layering falls (§2): [`ApplyCtx`] holds every renderer there
//! is, so the fold is a tier above [`gpu`](crate::gpu), while the state it writes holds
//! only tile handles and sits below. `document/` naming a renderer was the one edge
//! that made the two a cycle.
//!
//! - [`apply`] — `impl Materialize for DocState`, and the two questions the chrome asks
//!   before a gesture commits (`float_cut_offered`, `preview_of`).
//! - [`merge`] — whether a layer can be merged down without changing the picture
//!   (§14.11). Here rather than beside the state because the answer consults the
//!   transform's own paint plan.
//! - [`timeline`] — the order actions apply in: [`LinearTimeline`] solo,
//!   [`ReplicatedTimeline`] when shared (§5, §12).
//! - [`audit`] — that every `apply` touched only what its `Footprint` declared (§12.6).
//!
//! The submodules are crate-private; the names the rest of the crate spells are
//! re-exported below.

pub(crate) mod apply;
/// The §12.6 rule, checked on every fold of every debug build — see the module.
pub(crate) mod audit;
pub(crate) mod merge;
pub(crate) mod timeline;

pub(crate) use apply::{ApplyCtx, PreparedStroke};
pub(crate) use timeline::{LinearTimeline, ReplicatedTimeline, Step, Stepped, Timeline};
