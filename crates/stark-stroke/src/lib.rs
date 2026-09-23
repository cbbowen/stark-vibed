//! The stroke's CPU geometry (CLAUDE.md): what a pointer trace becomes before any
//! pixel exists.
//!
//! [`InputSample`]s go in; [`ControlPoint`](stark_model::path::ControlPoint)s —
//! what a stroke *is* once captured — and the flattened
//! [`IntermediateSample`](path::IntermediateSample)s a brush sweeps along come out.
//! Nothing here touches wgpu, a shader or a tile, which is the whole reason it is a
//! crate: the engine that consumes it is 60k lines behind a GPU device, and the
//! geometry's own tests, properties and fingerprint have no business waiting on them.
//!
//! - [`path`] — the least-squares cubic B-spline fit, the adaptive flattener, the
//!   arc-length map and the input tolerance they are all priced in (§6.2).
//! - `spline` — the B-spline basis underneath: the knot view, the solve, and the only
//!   place nalgebra is named. Crate-private, because nothing above it asks a question
//!   in terms of a knot.
//! - [`assist`] — drag-and-hold shape assist: what a held gesture is recognized as,
//!   how the rest of the drag steers it, and the path it realizes back into (§6.9).
//! - [`tow`] — the towed tip: stroke smoothing as a rope the pointer drags (§6.12).
//! - [`detmath`] — the deterministic series the **post-record** stages use in place
//!   of libm, so a replay and a peer agree to the bit (§12.1).
//!
//! **Which side of the record decides what the math may use.** The fit, the
//! arc-length map, the spline solve and the assist run *before* a record exists, so
//! they only have to agree with themselves within one process; flattening and the arc
//! run at every replay and on every peer, so they go through [`detmath`].
//! `path::fingerprint` is the gate on both halves.

#![warn(unreachable_pub)]

pub mod assist;
pub mod detmath;
mod input;
pub mod path;
pub(crate) mod spline;
pub mod tow;

pub use input::{COORD_LIMIT, InputSample};
