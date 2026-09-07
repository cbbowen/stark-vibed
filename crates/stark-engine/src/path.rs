//! Stroke path fitting and adaptive flattening (§6.2).
//!
//! Three representations, deliberately distinct:
//!
//! - [`InputSample`](crate::command::InputSample) — one raw pointer report, as it
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
//! All math here is deterministic, preserving golden / replay / save-load
//! equivalence.
//!
//! [`span_count`] and [`frozen_spans_for`] sit here rather than in one of the three
//! submodules because they belong to none of them: a span count is a statement about
//! a control *polygon*, which the fit produces and the flattener consumes.

mod arc;
mod arclen;
mod fit;
mod flatten;

pub use arc::{Arc, arc_at, arc_sagitta, fit_arc};
// `KNOT_COST` and `KNOT_SPACING` are `pub` for their *doc links* alone (`assist::realize`
// names the spacing, `path`'s own prose names the cost): a doc link cannot follow a
// re-export the compiler has pruned as unused.
pub use fit::{
    DEFAULT_TOLERANCE, KNOT_COST, KNOT_SPACING, MAX_TOLERANCE, MIN_TOLERANCE, PathFitter,
    clamp_tolerance, fit, fit_with_tolerance,
};
// The pen-channel layout and its two conversions, for `assist::realize` — which fits
// the same four channels onto the same control polygon and had its own copy of both.
pub(crate) use arclen::{arc_profile, param_at};
pub(crate) use fit::{CHANNELS, control_point_from};
pub use flatten::{
    FLATTEN_TOLERANCE, FlattenTolerance, IntermediateSample, flatten, flatten_spans,
    flatten_spans_from, point_at, span_end,
};

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
/// (`gpu::stroke::Resume`), so a head range equal to the span count would stop
/// carrying the brush forward and the tail would resume from a stale state. The bound
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

#[cfg(test)]
mod tests {
    /// The worst distance (canvas px) from any sample to the flattened curve fitted
    /// through them.
    fn fit_error(samples: &[InputSample]) -> f32 {
        let path = fit(samples);
        let poly = flatten(&path, FLATTEN_TOLERANCE);
        samples
            .iter()
            .map(|s| {
                poly.windows(2)
                    .map(|w| point_segment_distance(s.pos, w[0].pos, w[1].pos))
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0, f32::max)
    }

    /// Fit `samples` one report at a time, keeping every intermediate path and how much
    /// of it the fitter called frozen.
    fn fit_incrementally(samples: &[InputSample]) -> (PathFitter, Vec<(Vec<ControlPoint>, usize)>) {
        let mut fitter = PathFitter::new();
        let mut snaps = Vec::new();
        for s in samples {
            fitter.push(*s);
            snaps.push((fitter.path().to_vec(), fitter.frozen_spans()));
        }
        (fitter, snaps)
    }

    use super::arc::point_segment_distance;
    use super::*;
    use crate::command::InputSample;
    use stark_model::geom::Vec2;
    use stark_model::path::ControlPoint;

    fn sample(x: f32, y: f32) -> InputSample {
        InputSample::at(Vec2::new(x, y))
    }

    fn knot(x: f32, y: f32) -> ControlPoint {
        ControlPoint::at(Vec2::new(x, y))
    }

    /// **Thinning the window does not cost the fit its accuracy** — the one risk the
    /// bound carries.
    ///
    /// Measured against every report, *including the ones no solve ever saw*: if a
    /// survivor list let the curve wander off the stretches it skipped, this is where
    /// it would show. At 3200 reports the window is thinned throughout; at 200 it is
    /// under budget and untouched.
    ///
    /// Both bars are calibrated from measurement, thinned against unthinned. The
    /// absolute one catches a selection that lost the curve outright; the ratio
    /// catches accuracy that degrades as the reports get denser, which is what
    /// thinning too hard looks like. Measured, thinning moves the fit by under 0.1% at
    /// every rate, and the ~12% spread across the rates is the density policy's own
    /// smoothing, which `KNOT_COST` owns.
    ///
    /// The deviations are large in absolute terms because this curve is far coarser
    /// than [`DEFAULT_TOLERANCE`] is asked to trace (90 px of amplitude over 400 px,
    /// against a knot every 64 px): the fit is *meant* to smooth it.
    #[test]
    fn thinning_the_window_does_not_cost_the_fit_its_accuracy() {
        let curve = |t: f32| Vec2::new(t * 400.0, (t * 2.2).sin() * 90.0);
        let deviation = |n: usize| -> f32 {
            let pts: Vec<InputSample> = (0..n)
                .map(|i| {
                    let p = curve(i as f32 / (n - 1) as f32);
                    sample(p.x, p.y)
                })
                .collect();
            let poly = flatten(&fit(&pts), FLATTEN_TOLERANCE);
            pts.iter()
                .map(|r| {
                    poly.iter()
                        .map(|s| (s.pos - r.pos).length())
                        .fold(f32::MAX, f32::min)
                })
                .fold(0.0f32, f32::max)
        };
        let sparse = deviation(200);
        for n in [200usize, 800, 3200] {
            let worst = deviation(n);
            assert!(
                worst < 35.0,
                "{n} reports fitted {worst:.2}px off the curve they came from",
            );
            assert!(
                worst < sparse * 1.25,
                "{n} reports fitted {worst:.2}px off, against {sparse:.2}px unthinned                  — thinning is costing accuracy as the rate rises",
            );
        }
    }

    #[test]
    fn the_curve_is_pinned_to_its_end_control_points() {
        // What the clamped end condition buys: the stroke starts and finishes at the
        // first and last control point exactly, even though the interior ones are
        // only approached.
        let knots = [
            knot(0.0, 0.0),
            knot(30.0, 40.0),
            knot(90.0, 10.0),
            knot(120.0, 60.0),
        ];
        let poly = flatten(&knots, FLATTEN_TOLERANCE);
        assert!((poly.first().unwrap().pos - knots[0].pos).length() < 1e-4);
        assert!((poly.last().unwrap().pos - knots[3].pos).length() < 1e-4);
    }

    // ---- fitting -------------------------------------------------------

    #[test]
    fn fit_starts_and_ends_under_the_pointer() {
        // A least-squares fit does not pin its ends: an unassigned stretch of
        // parameter costs nothing, so unless the ends are held the curve runs out
        // past both ends of the stroke.
        let pts: Vec<InputSample> = (0..20).map(|i| sample(i as f32, i as f32 * 0.9)).collect();
        let fitted = fit(&pts);
        assert_eq!(fitted.first().unwrap().pos, pts.first().unwrap().pos);
        assert_eq!(fitted.last().unwrap().pos, pts.last().unwrap().pos);
        // And the clamped end condition puts the *curve* there too, not just the
        // control polygon.
        let poly = flatten(&fitted, FLATTEN_TOLERANCE);
        assert!((poly.first().unwrap().pos - pts.first().unwrap().pos).length() < 1e-3);
        assert!((poly.last().unwrap().pos - pts.last().unwrap().pos).length() < 1e-3);
    }

    #[test]
    fn fit_flattens_coarse_staircase() {
        // A 2-px right / 2-px up staircase: a slow diagonal snapped to a 2px device
        // grid, each corner ~1.4px off the diagonal. Fitting is the only smoothing
        // stage, so this is what happens when a control point's price sits *below*
        // the input's own quantization — the zigzag reads as curvature and gets
        // traced. `a_coarser_tolerance_smooths_what_a_finer_one_traces` is the same
        // staircase with the grid declared.
        let mut stair = vec![sample(0.0, 0.0)];
        for i in 0..12 {
            let b = (i * 2) as f32;
            stair.push(sample(b + 2.0, b)); // 2px right
            stair.push(sample(b + 2.0, b + 2.0)); // 2px up
        }
        // The claim is the *shape*, not the control-point count: traced, the curve
        // would sit ~0 from every sample; smoothed, it splits the corners.
        let err = fit_error(&stair);
        assert!(
            (0.5..2.0).contains(&err),
            "err {err} — traced, not smoothed?"
        );
    }

    /// The same 2px staircase, with the grid it was quantized to declared. Told what
    /// the input can actually resolve, the fit stops paying for control points to
    /// explain a zigzag that is not there.
    #[test]
    fn a_coarser_tolerance_smooths_what_a_finer_one_traces() {
        let mut stair = vec![sample(0.0, 0.0)];
        for i in 0..12 {
            let b = (i * 2) as f32;
            stair.push(sample(b + 2.0, b));
            stair.push(sample(b + 2.0, b + 2.0));
        }
        let fine = fit_with_tolerance(&stair, DEFAULT_TOLERANCE).len();
        let coarse = fit_with_tolerance(&stair, 2.0).len();
        // What the *shape* is worth with the quantization taken out: the same reports
        // projected onto the diagonal they are a staircase of. Declaring the grid can
        // only buy back what the zigzag costs, never what the stroke is, so that is
        // the baseline both counts are measured from — a bare ratio of totals would
        // instead move whenever the finer fit improved, which is not under test.
        let ideal = fit_with_tolerance(
            &stair
                .iter()
                .map(|s| {
                    let d = 0.5 * (s.pos.x + s.pos.y);
                    InputSample {
                        pos: Vec2::splat(d),
                        ..*s
                    }
                })
                .collect::<Vec<_>>(),
            DEFAULT_TOLERANCE,
        )
        .len();
        // The premise. At 1px the zigzag is above the declared tolerance, so the fit does
        // pay for some of it; without this the comparison below could be satisfied by
        // both fits being perfect, and the test would have stopped testing anything.
        assert!(
            fine > ideal,
            "nothing left to smooth: {fine} control points over a shape worth {ideal}"
        );
        assert!(
            coarse.saturating_sub(ideal) * 2 <= fine.saturating_sub(ideal),
            "2px grid declared: {coarse} against {fine} control points, over a shape worth {ideal} — not smoothed"
        );
        // And what is left is still the diagonal, not a shortcut across it: every
        // sample is within a step of the curve.
        let poly = flatten(&fit_with_tolerance(&stair, 2.0), FLATTEN_TOLERANCE);
        let worst = stair
            .iter()
            .map(|s| {
                poly.windows(2)
                    .map(|w| point_segment_distance(s.pos, w[0].pos, w[1].pos))
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0, f32::max);
        assert!(worst < 2.0, "smoothed off the diagonal: {worst}px");
    }

    #[test]
    fn fit_keeps_real_corners() {
        // An L-shape: the corner must survive. The polygon starts far coarser than
        // this whole gesture, so keeping it depends on the sagitta test refining a
        // stroke shorter than one control-point interval.
        let pts = [
            sample(0.0, 0.0),
            sample(10.0, 0.0),
            sample(20.0, 0.0),
            sample(20.0, 10.0),
            sample(20.0, 20.0),
        ];
        let poly = flatten(&fit(&pts), FLATTEN_TOLERANCE);
        let nearest = poly
            .iter()
            .map(|p| (p.pos - Vec2::new(20.0, 0.0)).length())
            .fold(f32::INFINITY, f32::min);
        assert!(
            nearest < 3.0,
            "corner lost: curve passes {nearest}px from it"
        );
    }

    /// The guarantee incremental repaint rests on: whatever the fitter reports as
    /// frozen, flattening it now equals flattening it at the end.
    #[test]
    fn committed_knots_are_never_revised() {
        let pts: Vec<InputSample> = (0..60)
            .map(|i| {
                let t = i as f32 * 0.15;
                sample(t * 12.0, (t * 0.9).sin() * 40.0 + (t * 2.7).cos() * 6.0)
            })
            .collect();
        let (fitter, snaps) = fit_incrementally(&pts);
        let full = fitter.path();
        let mut checked_nonempty = false;
        for (path, frozen) in snaps {
            if frozen == 0 {
                continue;
            }
            checked_nonempty = true;
            let then = flatten_spans(&path, 0..frozen, 0.0, FLATTEN_TOLERANCE);
            let now = flatten_spans(&full, 0..frozen, 0.0, FLATTEN_TOLERANCE);
            assert_eq!(then, now, "frozen prefix of {frozen} spans changed");
        }
        assert!(checked_nonempty, "test never observed a frozen span");
    }

    #[test]
    fn a_click_fits_to_one_knot() {
        let fitted = fit(&[sample(3.0, 4.0)]);
        assert_eq!(fitted.len(), 1);
        assert_eq!(fitted[0].pos, Vec2::new(3.0, 4.0));
        // And flattens to a single directionless sample.
        let poly = flatten(&fitted, FLATTEN_TOLERANCE);
        assert_eq!(poly.len(), 1);
        assert_eq!(poly[0].pos, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn empty_input_fits_and_flattens_to_nothing() {
        assert!(fit(&[]).is_empty());
        assert!(flatten(&[], FLATTEN_TOLERANCE).is_empty());
    }

    // ---- flattening ----------------------------------------------------

    #[test]
    fn flatten_spans_tile_the_whole_path() {
        let knots = [
            knot(0.0, 0.0),
            knot(20.0, 30.0),
            knot(60.0, 30.0),
            knot(90.0, 0.0),
            knot(120.0, -40.0),
        ];
        let all = span_count(knots.len());
        let whole = flatten(&knots, FLATTEN_TOLERANCE);
        // Split anywhere: what an incremental renderer does is exactly this, with the
        // cut at `PathFitter::frozen_spans`.
        for cut in 1..all {
            let head = flatten_spans(&knots, 0..cut, 0.0, FLATTEN_TOLERANCE);
            let tail = flatten_spans(
                &knots,
                cut..all,
                head.last().unwrap().dist,
                FLATTEN_TOLERANCE,
            );
            // The ranges share exactly one point, so the pieces concatenate into the
            // whole with no gap and no duplicated segment.
            let joined: Vec<IntermediateSample> =
                head.iter().chain(tail[1..].iter()).copied().collect();
            assert_eq!(joined, whole, "cut at span {cut}");
        }
    }
}
