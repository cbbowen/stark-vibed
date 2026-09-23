//! The brush-dynamics path (§6.2): a serial swept-exchange loop that lets a stroke
//! pick paint up off the canvas and put it back down.
//!
//! Sequential, unlike the swept path: what the tip carries into a segment is what the
//! previous segment left on it. It runs entirely on the GPU (no CPU readback, so it
//! works on WebGPU), a per-segment × per-lateral-band reservoir texture standing in
//! for the tip's load.
//!
//! [`plan`] decides what to dispatch and touches no GPU, [`kit`] builds the objects
//! it is dispatched with, and [`run`] records it — checking its working textures out
//! of the stroke-level [`scratch`](crate::gpu::scratch) pool. Whether a stroke comes
//! here at all is [`StrokePlan::of`](super::path::StrokePlan::of)'s answer.

mod bleed;
mod kit;
mod liquify;
mod plan;
mod region;
mod run;
mod tool;
mod trio;

pub(in crate::gpu::stroke) use bleed::BLEED_TRAVEL_QUANTUM;
pub(super) use kit::{DynamicsKit, build_dynamics_kit};
