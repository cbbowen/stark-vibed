//! The GPU passes: stroke rasterization, selection masks, fills, transforms, the
//! canvas substrate's bake, and compositing/media (§6, §7).
//!
//! The device and the resources it hands out are [`device`](crate::device)'s, a tier
//! below — a pass names them, and nothing they name is a pass (§2).
//!
//! `gpu` and its modules are `pub(crate)`, and inside the crate an item is named by
//! its module path. The re-exports below are the names that leave the crate, for the
//! root to offer.

pub(crate) mod composite;
pub(crate) mod environment;
pub(crate) mod fill;
pub(crate) mod merge;
pub(crate) mod pigment;
pub(crate) mod place;
// One check over every pipeline in this subsystem, so it lives above all of them
// rather than in whichever module it happens to name first.
#[cfg(test)]
mod pipeline_coverage;
pub(crate) mod registry;
pub(crate) mod selection;
pub(crate) mod stroke;
pub(crate) mod substrate;
pub(crate) mod transform;

pub use composite::{MediaParams, Offscreen, Output, Transfer};
pub use environment::EnvironmentId;
pub use stroke::{max_stretch, max_tip_reach};
