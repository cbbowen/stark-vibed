//! GPU subsystem: device context, the recycling tile pool, stroke rasterization,
//! selection masks, and compositing/media (§6, §7).
//!
//! `gpu` and its modules are `pub(crate)`, and inside the crate an item is named by
//! its module path. The re-exports below are the names that leave the crate, for the
//! root to offer.

pub(crate) mod channels;
pub(crate) mod composite;
pub(crate) mod context;
pub(crate) mod desc;
pub(crate) mod environment;
pub(crate) mod fill;
pub(crate) mod half;
pub(crate) mod merge;
pub(crate) mod pigment;
pub(crate) mod place;
pub(crate) mod readback;
// One check over every pipeline in this subsystem, so it lives above all of them
// rather than in whichever module it happens to name first.
#[cfg(test)]
mod pipeline_coverage;
pub(crate) mod registry;
pub(crate) mod scratch;
pub(crate) mod selection;
pub(crate) mod stroke;
pub(crate) mod substrate;
pub(crate) mod tile;
pub(crate) mod transform;
pub(crate) mod uniforms;

pub use composite::{MediaParams, Offscreen, Output, Transfer};
pub use context::{DeviceFailure, FailureKind, GpuContext, GpuHealth};
pub use environment::EnvironmentId;
pub use stroke::{max_stretch, max_tip_reach};
