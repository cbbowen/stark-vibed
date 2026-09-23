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
//! **Which side of the record decides what the math may use.** The fit, the arc-length
//! map, the spline solve and the assist run *before* a record exists: the log keeps
//! their control points, not their samples, so they only have to agree with themselves
//! within one process (`preview == committed`, the goldens, the fingerprint). Flattening
//! and the arc run *after* it, at every replay and on every peer, so they fall under
//! §12.1's bit-agreement rule — which is why they use `detmath` and not libm.
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

pub use fit::{PathFitter, fit, fit_with_tolerance};
pub use flatten::{
    FLATTEN_TOLERANCE, FlattenTolerance, IntermediateSample, flatten, flatten_spans_from, point_at,
    span_end,
};
pub use tolerance::{DEFAULT_TOLERANCE, MAX_TOLERANCE, MIN_TOLERANCE};

pub use arc::{Arc, arc_at, arc_sagitta, fit_arc};
// The arc series' domain and the half-turn it caps, for the one guard that ties them
// to the bleed walk-back that widened the domain — which lives in the engine's
// `gpu::stroke`, since this crate sits below it and cannot name either of that term's
// factors.
pub use arc::{MAX_ARC_TURN, MAX_HALF_TURN_SIN};
pub(crate) use arclen::{arc_profile, param_at, polyline_length};
pub use arclen::{cumulative_lengths, locate};
pub use fit::fit_at;
// The pen-channel layout, for `assist::realize`, which fits the same channels onto the
// same control polygon.
pub(crate) use fit::{CHANNELS, control_points, pen_row};
pub use flatten::flatten_spans;
pub use tolerance::InputTolerance;

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

/// What the geometry tests share: builders, the recorded strokes, and the one measure of
/// how far points stray from a polyline.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{IntermediateSample, PathFitter, arc::point_segment_distance};
    use crate::InputSample;
    use stark_model::geom::Vec2;
    use stark_model::path::ControlPoint;

    /// One pointer report at a position, with everything else at rest.
    pub(crate) fn sample(x: f32, y: f32) -> InputSample {
        InputSample::at(Vec2::new(x, y))
    }

    /// One full-pressure control point at a position.
    pub(crate) fn knot(x: f32, y: f32) -> ControlPoint {
        ControlPoint::at(Vec2::new(x, y))
    }

    /// The worst distance (canvas px) from any of `points` to the polyline `poly` — to
    /// its one vertex, for a click's.
    pub(crate) fn max_deviation(
        points: impl IntoIterator<Item = Vec2>,
        poly: &[IntermediateSample],
    ) -> f32 {
        points
            .into_iter()
            .map(|p| match poly {
                [only] => (p - only.pos).length(),
                _ => poly
                    .windows(2)
                    .map(|w| point_segment_distance(p, w[0].pos, w[1].pos))
                    .fold(f32::INFINITY, f32::min),
            })
            .fold(0.0, f32::max)
    }

    /// Feed `samples` one at a time, snapshotting the path and its frozen span count
    /// after every push; the fitter comes back finished.
    pub(crate) fn fit_incrementally(
        samples: &[InputSample],
    ) -> (PathFitter, Vec<(Vec<ControlPoint>, usize)>) {
        let mut f = PathFitter::new();
        let mut snaps = Vec::new();
        for s in samples {
            f.push(*s);
            snaps.push((f.path(), f.frozen_spans()));
        }
        f.finish();
        (f, snaps)
    }

    /// Every recorded stroke in `stark_testdata`, by name.
    pub(crate) const RECORDED: [(&str, &[[f32; 2]]); 6] = [
        ("hairpin", stark_testdata::HAIRPIN_STROKE),
        ("loop", stark_testdata::LOOP_STROKE),
        ("C", stark_testdata::C_STROKE),
        ("spiral", stark_testdata::SPIRAL_STROKE),
        ("big-C", stark_testdata::BIG_C_STROKE),
        ("fast", stark_testdata::FAST_STROKE),
    ];

    /// Recorded positions with pen channels laid over them, so a channel mapped onto
    /// another shows. Integer arithmetic only: the input has to be the same bits on
    /// every platform (`fingerprint`).
    pub(crate) fn with_pen(pts: &[[f32; 2]]) -> Vec<InputSample> {
        pts.iter()
            .enumerate()
            .map(|(i, &[x, y])| {
                // A triangle wave of `period` reports, in `[0, 1]`.
                let tri = |period: usize| {
                    let half = period as f32 / 2.0;
                    1.0 - ((i % period) as f32 - half).abs() / half
                };
                InputSample {
                    pos: Vec2::new(x, y),
                    pressure: 0.2 + 0.6 * tri(64),
                    tilt: Vec2::new(0.5 * tri(50) - 0.25, 0.3 - 0.4 * tri(38)),
                    // Off zero, so the fitter's re-basing onto the stroke's clock shows.
                    time: 1000.0 + i as f64 / 240.0,
                }
            })
            .collect()
    }

    /// `samples` as a hover and a stroke: the first quarter with the pressure lifted
    /// off, to seed as a run-up, and the reports pressed after it.
    pub(crate) fn split_runup(samples: &[InputSample]) -> (Vec<InputSample>, &[InputSample]) {
        let (hover, stroke) = samples.split_at(samples.len() / 4);
        let hover = hover
            .iter()
            .map(|s| InputSample {
                pressure: 0.0,
                ..*s
            })
            .collect();
        (hover, stroke)
    }

    /// The configuration every geometry property runs under: a fixed, modest case
    /// count, so the suite costs the same every run.
    pub(crate) fn cases() -> proptest::test_runner::Config {
        proptest::test_runner::Config::with_cases(64)
    }

    /// Every field of every control point as bits: what "the same path" means to a
    /// test that claims the same pixels.
    pub(crate) fn bits(path: &[ControlPoint]) -> Vec<[u32; 6]> {
        path.iter()
            .map(|k| [k.pos.x, k.pos.y, k.pressure, k.tilt.x, k.tilt.y, k.time].map(f32::to_bits))
            .collect()
    }
}
