//! The device and what it hands out: the wgpu context, the recycling tile pool, the
//! scratch pool, the descriptor helpers, the channel set, the uniform slots, the
//! readback and the f16 conversions (§6.1).
//!
//! **Below [`document`](crate::document), where [`gpu`](crate::gpu) is above it**, and
//! that is the whole reason the module exists (§2, the module layering). Not one of
//! these names a pass or a pipeline, so the state a log folds into can hold a tile
//! handle without naming the renderers — which is the edge that used to make
//! `document` and `gpu` a cycle.
//!
//! `device` and its modules are `pub(crate)`, and inside the crate an item is named by
//! its module path. The re-exports below are the names that leave the crate, for the
//! root to offer.

pub(crate) mod channels;
pub(crate) mod context;
pub(crate) mod desc;
pub(crate) mod half;
pub(crate) mod readback;
pub(crate) mod scratch;
pub(crate) mod tile;
pub(crate) mod uniforms;

pub use context::{GpuContext, GpuHealth};
