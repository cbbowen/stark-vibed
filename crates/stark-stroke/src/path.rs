//! Stroke path fitting and adaptive flattening (§6.2).
//!
//! Three representations, deliberately distinct:
//!
//! - [`InputSample`](crate::InputSample) — one raw pointer report, as it
//!   arrived. Never stored: it exists only between the pointer event and the fitter.
//! - [`ControlPoint`](stark_model::path::ControlPoint) — a knot of the fitted stroke
//!   curve, and what a stroke *is* once captured
//!   ([`StrokeRecord::path`](stark_model::document::StrokeRecord)).
//! - [`IntermediateSample`] — a point *of the curve*: position plus its
//!   derivative, and the pen attributes interpolated there. Transient, produced
//!   by [`flatten`](fn@flatten) and consumed by the stamp generator.
//!
//! [`PathFitter`] streams the first into the second: a **least-squares cubic
//! B-spline fit**, grown and refit as samples arrive, with a prefix of control
//! points *frozen* once the incoming data can no longer pull on it. A frozen control
//! point is final whatever is drawn next, so a caller can render the settled part of
//! a live stroke once instead of repainting it on every pointer move (see
//! [`PathFitter::frozen_spans`]).
//!
//! [`flatten`](fn@flatten) expands control points through that same B-spline into a
//! polyline, subdividing only where a straight segment would exceed a bounded error
//! in position, tangent direction and pen attributes — which is why
//! [`IntermediateSample`] carries the derivative rather than position alone.
//!
//! The fit runs *before* a record exists and flattening *after* it, at every replay and
//! on every peer — which is why only the second goes through `detmath`. The crate root
//! states that rule once, for every stage in it.
//!
//! [`span_count`] and [`frozen_spans_for`] sit here rather than in a submodule because
//! they belong to none of them: a span count is a statement about a control *polygon*,
//! which the fit produces and the flattener consumes.

mod arc;
mod arclen;
#[cfg(test)]
mod fingerprint;
mod fit;
mod flatten;
mod tolerance;

// The arc series' domain and the half-turn it caps are `pub` for the one guard that
// ties them to the bleed walk-back that widened the domain — which lives in the engine's
// `gpu::stroke`, since this crate sits below it and cannot name either of that term's
// factors.
pub use arc::{Arc, MAX_ARC_TURN, MAX_HALF_TURN_SIN, arc_at, arc_sagitta, fit_arc};
pub use arclen::{cumulative_lengths, locate};
pub use fit::{PathFitter, fit, fit_at};
pub use flatten::{
    FLATTEN_TOLERANCE, FlattenTolerance, IntermediateSample, flatten, flatten_spans,
    flatten_spans_from, point_at, span_end,
};
pub use tolerance::{DEFAULT_TOLERANCE, InputTolerance, MAX_TOLERANCE, MIN_TOLERANCE};

#[cfg(test)]
pub(crate) use arc::point_segment_distance;
pub(crate) use arclen::{arc_profile, param_at, polyline_length};
// The pen-channel layout, for `assist::realize`, which fits the same channels onto the
// same control polygon.
pub(crate) use fit::{CHANNELS, control_points, pen_row};

/// How many cubic spans the curve through `control_points` has.
///
/// The clamped end condition is expressed by *repeating* each end control point
/// `degree` times in the conceptual control sequence (the clamped knot view,
/// [`crate::spline`]), which pins the curve to them. Those repeats are spans too, so
/// `m` control points give `m + 1` spans rather than the `m - 1` an interpolating
/// spline would — the two extra sit at the ends, each covering the sixth of the first
/// (last) leg that the clamp bends through. Fewer than two control points is not a
/// curve: one is a click, zero is nothing.
pub fn span_count(control_points: usize) -> usize {
    // Asked of the knot view rather than restated as `m + 1`, so the two spellings of
    // one number cannot drift.
    crate::spline::SplineIndex::spans_for(control_points)
}

/// How many spans `frozen` frozen control points settle, out of a path of `total`.
///
/// A span reads at most two control points past its own index, so span `k` is final
/// once control points `0..=k+1` are — hence `frozen - 1`. Separate from
/// [`PathFitter::frozen_spans`] because a *received* stroke asks the same question
/// without the fitter that produced it: a peer knows which control points the sender
/// has stopped resending (§17.5) and needs the same incremental repaint from them.
///
/// **Strictly fewer than [`span_count`]`(total)`, for every `frozen <= total`**, and
/// something downstream depends on it: the stroke renderer captures cross-piece brush
/// state only for a range that does *not* reach the end of the stroke
/// (`gpu::stroke::incremental::Resume`), so a head range equal to the span count would
/// stop carrying the brush forward and the tail would resume from a stale state. The bound
/// holds because `frozen - 1 <= total - 1` and `span_count(total) = total + 1` for a
/// curve. For `total < 2` there is no curve, `span_count` is 0, and no head range
/// exists.
pub fn frozen_spans_for(frozen: usize, total: usize) -> usize {
    let out = frozen.saturating_sub(1).min(span_count(total));
    debug_assert!(
        out < span_count(total) || span_count(total) == 0,
        "a frozen head reaching the stroke's end would stop carrying the brush",
    );
    out
}
