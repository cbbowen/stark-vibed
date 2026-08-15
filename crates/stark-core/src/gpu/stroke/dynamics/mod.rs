//! The brush-dynamics path (§6.2): a serial swept-exchange loop that lets
//! a stroke pick paint up off the canvas and put it back down.
//!
//! Where the swept path composes by summing optical depth — and so can draw its
//! segments in any order — this one is *sequential* by nature: what the tip carries
//! into a segment is what the previous segment left on it. The loop runs on the GPU
//! (no CPU readback, so it works on WebGPU) with a per-segment x per-lateral-band
//! reservoir texture standing in for the tip's load.
//!
//! The path is three modules, split by what a maintainer is holding in their head:
//! [`plan`] works out what to dispatch and touches no GPU at all, [`kit`] builds the
//! objects it is dispatched with, and [`run`] records it — checking its working
//! textures out of the stroke-level [`scratch`](super::scratch) pool through the
//! submit scope that owns their release. What is left here is the one question
//! asked before any of them — which path a stroke takes at all.

use crate::document::StrokeRecord;

use super::budget::flatten_tolerance;
use super::region::segment_fits_region;

mod bleed;
mod kit;
mod plan;
mod run;
mod slots;

pub(in crate::gpu::stroke) use bleed::BLEED_TRAVEL_QUANTUM;
pub(super) use kit::{DynamicsKit, build_dynamics_kit};

/// fp32, for the same reason the prefix-τ volume is: every fragment reads the baked
/// swept prefix as a *difference* of two prefix sums (§6.2), so f16 would band exactly
/// where the difference is smallest. Shared by the layout that declares the storage
/// texture ([`kit`]) and the run that allocates it ([`run`]).
const BAKE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Which path a stroke takes, as [`dynamics_setup`] decides it.
///
/// The two swept answers are kept apart because they are not the same event: one is
/// the fast path doing its job, the other is the renderer failing to draw the brush
/// it was given. Only the caller knows how loudly to say so, so the distinction is
/// carried out rather than resolved here.
pub(super) enum StrokePath {
    /// Run the sequential stamp loop.
    Loop,
    /// The brush manipulates no paint already on the canvas, so the swept deposit
    /// *is* the whole stroke — one pass, no region, nothing given up.
    Swept,
    /// The brush manipulates paint, but its tip alone wants more than one region, and
    /// the region is the one thing pieces cannot subdivide. The swept deposit draws
    /// what it can, which is the brush's own `add` paint and none of the manipulation.
    TipTooLarge,
}

/// Which path a stroke takes and the budget it flattens at — both decided together,
/// because both are answers about the brush alone and every path needs the second.
pub(super) struct StrokePlan {
    pub(super) path: StrokePath,
    pub(super) tol: crate::path::FlattenTolerance,
}

/// Which path `rec` takes, and the flattening budget if it is the stamp loop.
///
/// **A pure function of the record, and of the brush alone.** This answer has to
/// agree across every render of every piece of the stroke and with the commit that
/// eventually replaces them: a live tail that took the stamp loop while the commit
/// degraded to the swept deposit would redraw the stroke the moment the pointer came
/// up. Asking only about the brush is the strongest form of that guarantee — there is
/// nothing about the piece in hand, or the stroke's length, for it to disagree over —
/// and it is what lets `render_range` re-ask on every pointer move for free.
///
/// It can read that way because the stroke's *size* decides nothing: an oversized
/// stroke is drawn one region-sized piece at a time ([`chunk_segments`]) rather than
/// degraded. All that is left is the floor no subdivision gets under — one segment's
/// own footprint — which is [`segment_fits_region`]'s question.
pub(super) fn dynamics_setup(rec: &StrokeRecord) -> StrokePlan {
    let d = rec.brush.dynamics;
    // The brush's **own** rates, not the modulated ones — and that is sound rather
    // than an oversight the pen could catch out. A modulation is a factor in [0, 1]
    // (`document::Modulation`), so an axis the brush leaves at zero is zero at every
    // point of every stroke it could ever draw, and one it leaves positive is
    // positive *somewhere*. There is no segment this test could be asked about that
    // would answer differently — which is exactly the property the function's
    // contract above needs, and the reason a modulation was built as a multiplier.
    // The same flattened segments whichever path runs, at the same budget: a long
    // stroke costs more pieces, not coarser geometry — and the swept fallback below
    // draws the very segments the loop would have.
    let tol = flatten_tolerance(&rec.brush);
    let path = if d.lift <= 0.0 && d.deposit <= 0.0 && d.charge <= 0.0 && d.bleed <= 0.0 {
        StrokePath::Swept
    } else if segment_fits_region(&rec.brush, tol) {
        StrokePath::Loop
    } else {
        StrokePath::TipTooLarge
    };
    StrokePlan { path, tol }
}
