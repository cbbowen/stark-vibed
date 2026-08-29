//! Stroke path fitting and adaptive flattening (§6.2).
//!
//! Three representations, deliberately distinct:
//!
//! - [`InputSample`](crate::command::InputSample) — one raw pointer report, as it arrived. High frequency,
//!   jittery, and never stored: it exists only between the pointer event and the
//!   fitter.
//! - [`ControlPoint`](stark_model::path::ControlPoint) — a knot of the fitted stroke curve. This is what a stroke
//!   *is* once captured ([`StrokeRecord::path`](stark_model::document::StrokeRecord)):
//!   typically an order of magnitude fewer points than the input arrived as, and
//!   all that is needed to reconstruct the stroke.
//! - [`IntermediateSample`] — a point *of the curve*: position plus its
//!   derivative, and the pen attributes interpolated there. Transient, produced
//!   by [`flatten`] and consumed by the stamp generator.
//!
//! [`PathFitter`] streams the first into the second: a **least-squares cubic
//! B-spline fit**, grown and refit as samples arrive, with a prefix of control
//! points *frozen* once the incoming data can no longer pull on it. Freezing is
//! what makes the fit append-only — a frozen control point is final, whatever is
//! drawn next — so a caller can render the settled part of a live stroke once
//! instead of repainting it on every pointer move (see
//! [`PathFitter::frozen_spans`]).
//!
//! [`flatten`] expands control points through that same B-spline into a polyline,
//! **adaptively**: it subdivides only where a straight segment would exceed a
//! bounded error in position, tangent direction, and pen attributes. A long gentle
//! stroke then costs a handful of segments where uniform arc-length sampling cost
//! hundreds, while a corner still gets the density it needs — the tangent bound
//! buys both, which is why [`IntermediateSample`] carries the derivative rather
//! than position alone.
//!
//! All math here is deterministic, preserving golden / replay / save-load
//! equivalence.
//!
//! **Three files, named for the three the banners already named.** This was one
//! 2,738-line module holding a streaming fitter, an arc primitive and a flattener,
//! which is three subjects and not one long one — every type belongs to exactly one of
//! them (`PathFitter` and `Accepted` to the fit, `Arc` to the arcs,
//! [`FlattenTolerance`] and [`IntermediateSample`] to the flattener), and the
//! constants partition the same way. The public names are re-exported here, so
//! `path::` still spells everything it used to.
//!
//! The tests stay in this file rather than moving with the code, because they are
//! tests of the *pipeline*: most of them fit a stroke and then flatten what came out,
//! and they share one set of builders. Splitting them would put a copy of those in
//! each file.

mod arc;
mod fit;
mod flatten;

pub use arc::{Arc, arc_at, arc_sagitta, fit_arc};
pub use fit::{
    DEFAULT_TOLERANCE, KNOT_COST, KNOT_SPACING, MAX_TOLERANCE, MIN_TOLERANCE, PathFitter,
    clamp_tolerance, fit, fit_with_tolerance,
};
pub(crate) use fit::{arc_profile, param_at};
pub use flatten::{
    FLATTEN_TOLERANCE, FlattenTolerance, IntermediateSample, flatten, flatten_spans,
    flatten_spans_from, frozen_spans_for, point_at, span_count, span_end,
};

#[cfg(test)]
mod tests {
    // The pieces' own internals the pipeline tests reach for. `Accepted`, the window
    // constants and `window_indices` are the fit's bookkeeping — several tests here
    // assert on the *window* rather than on the curve, which is the only way to say
    // "thinning did not cost the fit its accuracy". `span` is the flattener's Bézier
    // conversion, held against `spline`'s own evaluation.
    use super::fit::{
        Accepted, CHANNELS, FREE_CONTROL_POINTS, MAX_WINDOW_SAMPLES, point_segment_distance,
        window_indices,
    };
    use super::flatten::span;
    use super::*;
    use crate::command::InputSample;
    use crate::spline::CubicBSpline;
    use stark_model::geom::Vec2;
    use stark_model::path::ControlPoint;

    fn sample(x: f32, y: f32) -> InputSample {
        InputSample::at(Vec2::new(x, y))
    }

    fn knot(x: f32, y: f32) -> ControlPoint {
        ControlPoint::at(Vec2::new(x, y))
    }

    /// `n` accepted reports `step` apart along the x axis — the shape
    /// [`window_indices`] reads, without a fitter around it.
    fn accepted(n: usize, step: f32) -> Vec<Accepted> {
        (0..n)
            .map(|i| Accepted {
                pos: Vec2::new(i as f32 * step, 0.0),
                channels: [0.0; CHANNELS],
                arc: i as f32 * step,
            })
            .collect()
    }

    /// **Under budget the selection is the identity**, which is what makes this a
    /// change to the pathological cases only. Ordinary painting keeps every report the
    /// window admitted, `arc_weights` reduces to the rule it always was, and the fitted
    /// curve is therefore bit-identical to what it was before the bound existed — so
    /// no golden moves and no recorded stroke re-fits.
    #[test]
    fn a_window_under_budget_keeps_every_report() {
        let pts = accepted(MAX_WINDOW_SAMPLES, 1.0);
        for lo in [0, 1, 7, MAX_WINDOW_SAMPLES - 1] {
            assert_eq!(
                window_indices(&pts, lo),
                (lo..pts.len()).collect::<Vec<_>>(),
                "a window of {} was thinned",
                pts.len() - lo,
            );
        }
    }

    /// Past the budget the window stops growing — the whole point. Both ends survive
    /// whatever the budget: the leading report anchors against the frozen prefix, and
    /// the trailing one is where the pen actually is.
    #[test]
    fn a_dense_window_is_capped_and_keeps_both_ends() {
        for n in [MAX_WINDOW_SAMPLES + 1, 500, 20_000] {
            let pts = accepted(n, 0.05);
            let idx = window_indices(&pts, 0);
            assert!(
                idx.len() <= MAX_WINDOW_SAMPLES + 1,
                "{n} reports produced a window of {}",
                idx.len(),
            );
            assert_eq!(idx.first().copied(), Some(0), "the window lost its head");
            assert_eq!(idx.last().copied(), Some(n - 1), "the window lost the pen");
            assert!(
                idx.windows(2).all(|w| w[0] < w[1]),
                "the survivors are out of order",
            );
        }
    }

    /// The survivors are spread along the **arc**, not along the report index — so a
    /// hand that paused cannot spend the whole budget on the pause.
    ///
    /// Half the reports here sit on one point and the other half cover the distance;
    /// an index-even rule would put half the window in the dwell.
    #[test]
    fn a_dwell_does_not_swallow_the_window() {
        let mut pts = accepted(200, 0.0);
        for (i, p) in pts.iter_mut().enumerate().skip(100) {
            let d = (i - 99) as f32;
            p.pos = Vec2::new(d, 0.0);
            p.arc = d;
        }
        let idx = window_indices(&pts, 0);
        let in_dwell = idx.iter().filter(|&&i| i < 100).count();
        assert!(
            in_dwell <= 2,
            "{in_dwell} of {} survivors were spent on a dwell",
            idx.len(),
        );
    }

    /// **Thinning the window does not cost the fit its accuracy** — the one risk the
    /// bound actually carries.
    ///
    /// Measured against every report, *including the ones no solve ever saw*, which is
    /// the point: if a survivor list let the curve wander off the stretches it skipped,
    /// this is where it would show. A densely-reported mark is exactly the case that
    /// gets thinned — at 3200 reports the window is thinned throughout, at 200 it is
    /// under budget and untouched.
    ///
    /// The bars come from running it both ways, with `MAX_WINDOW_SAMPLES` raised to
    /// disable the bound:
    ///
    /// | reports | thinned | unthinned |
    /// |---------|---------|-----------|
    /// | 200     | 28.461  | 28.481    |
    /// | 800     | 29.898  | 29.992    |
    /// | 3200    | 31.824  | 31.833    |
    ///
    /// So thinning moves the fit by under 0.1% at every rate, and the ~12% spread
    /// across the rates is the density policy's own smoothing — a property this
    /// stroke had before the bound existed and which `KNOT_COST` owns, not this.
    ///
    /// Both bars matter and they catch different things. The absolute one catches a
    /// selection that lost the curve outright; the ratio catches the failure that
    /// would actually be *this* change's fault — accuracy that degrades as the
    /// reports get denser, which is what thinning too hard would look like.
    ///
    /// The deviations are large in absolute terms because this curve is far coarser
    /// than [`DEFAULT_TOLERANCE`] is asked to trace (90 px of amplitude over 400 px,
    /// against a knot every 64 px): the fit is *meant* to smooth it. That is why the
    /// bars are calibrated from measurement rather than picked to look tight.
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

    /// Feed samples one at a time, snapshotting the fit after every push.
    fn fit_incrementally(samples: &[InputSample]) -> (PathFitter, Vec<(Vec<ControlPoint>, usize)>) {
        let mut f = PathFitter::new();
        let mut snaps = Vec::new();
        for s in samples {
            f.push(*s);
            snaps.push((f.path(), f.frozen_spans()));
        }
        f.finish();
        (f, snaps)
    }

    /// **A non-finite report is dropped, not fitted** — and the stroke that comes out
    /// is the one its finite reports describe, exactly as if the bad report had never
    /// arrived.
    ///
    /// Both halves matter and the second is the sharper claim. Merely not panicking
    /// would be satisfied by a fitter that swallowed the whole stroke; what has to
    /// hold is that one bad report costs one report.
    ///
    /// The panic this rules out was real and three subsystems downstream: `arc`
    /// accumulates the step to the bad sample, every curve parameter is derived from
    /// `arc`, so `solve_window`'s normal equations go NaN — and they are then singular at
    /// every ridge, which its solve reports with `unreachable!` because for
    /// *admissible* input it genuinely cannot happen.
    ///
    /// **Finite is not admissible**, which is why the last two rows are finite. A
    /// position a whole `f32` range from its neighbour makes `arc` accumulate an
    /// infinite step and every parameter after it a NaN, by subtraction rather than
    /// by anything the report itself carries — so a gate that asked only
    /// `is_finite` let the same panic through the same door.
    #[test]
    fn an_inadmissible_report_is_dropped_rather_than_fitted() {
        let clean: Vec<InputSample> = (0..24)
            .map(|i| {
                let t = i as f32;
                sample(t * 7.0, (t * 0.4).sin() * 30.0)
            })
            .collect();

        /// One way a report can be unusable: a name for the failure message, and the
        /// channel it poisons.
        type Poison = (&'static str, fn(InputSample) -> InputSample);

        // Every channel a report has, and every way one can be unusable.
        let poisons: [Poison; 7] = [
            ("pos.x", |mut s| {
                s.pos.x = f32::NAN;
                s
            }),
            // Finite, and past the last tile an `i32` can address — the difference
            // between this gate and a plain `is_finite`.
            ("pos.x beyond the grid", |mut s| {
                s.pos.x = 1.0e30;
                s
            }),
            // The boundary itself, which `TileRect::covering` refuses: `COORD_LIMIT`
            // is `2³¹` tiles out, and `2³¹` is one past `i32::MAX`.
            ("pos.y at the limit", |mut s| {
                s.pos.y = crate::command::COORD_LIMIT;
                s
            }),
            ("pos.y", |mut s| {
                s.pos.y = f32::INFINITY;
                s
            }),
            ("pressure", |mut s| {
                s.pressure = f32::NAN;
                s
            }),
            ("tilt", |mut s| {
                s.tilt = Vec2::splat(f32::NAN);
                s
            }),
            ("time", |mut s| {
                s.time = f64::NAN;
                s
            }),
        ];

        let reference = {
            let mut f = PathFitter::new();
            for s in &clean {
                f.push(*s);
            }
            f.finish();
            f.path()
        };

        for (name, poison) in poisons {
            // Injected mid-stroke, where the fitter has state to corrupt, and again
            // as the very first report, which is the case that seeds `t0` and the
            // arc origin.
            for at in [0usize, 12] {
                let mut f = PathFitter::new();
                for (i, s) in clean.iter().enumerate() {
                    if i == at {
                        f.push(poison(*s));
                    }
                    f.push(*s);
                }
                f.finish();
                let got = f.path();
                assert_eq!(
                    got.len(),
                    reference.len(),
                    "{name} at {at} changed the fitted path's shape",
                );
                for (a, b) in got.iter().zip(&reference) {
                    assert!(
                        (a.pos - b.pos).length() < 1e-4
                            && a.pressure.is_finite()
                            && a.tilt.is_finite(),
                        "{name} at {at} moved a control point: {a:?} vs {b:?}",
                    );
                }
            }
        }
    }

    /// The load-bearing link between the two halves of this module: the span form
    /// used to *render* a stored path must be the same curve [`CubicBSpline`] **fitted**.
    /// If these ever diverge, a stroke would be fitted to one curve and drawn as
    /// another — silently, since both are smooth and pass through roughly the same
    /// place.
    #[test]
    fn span_form_matches_the_fitted_spline() {
        use nalgebra::{Const, Dyn, OMatrix};

        for m in 2..9usize {
            let ctrl: Vec<Vec2> = (0..m)
                .map(|j| {
                    let t = j as f32;
                    Vec2::new(t * 13.0 + (t * 2.1).sin() * 4.0, (t * 0.8).cos() * 21.0)
                })
                .collect();
            let knots: Vec<ControlPoint> = ctrl.iter().map(|&p| ControlPoint::at(p)).collect();

            let rows =
                OMatrix::<f32, Dyn, Const<2>>::from_fn_generic(Dyn(m), Const::<2>, |j, d| {
                    if d == 0 { ctrl[j].x } else { ctrl[j].y }
                });
            let reference: CubicBSpline<'_, 2> = CubicBSpline::new(&rows).unwrap();

            // Not a second spelling of the count any more — `span_count` asks
            // `SplineIndex` — so this line is a tautology kept for one thing it still
            // says: that `reference`, built from a matrix rather than from `m`, has the
            // `m` this loop thinks it has. What the loop below checks is the part that
            // was never arithmetic: that the Bézier conversion evaluates to the same
            // curve.
            assert_eq!(
                span_count(m),
                reference.num_spans(),
                "span count disagrees at m = {m}"
            );
            for k in 0..span_count(m) {
                let sp = span(&knots, k);
                for i in 0..=8 {
                    let u = i as f32 / 8.0;
                    let want = reference.evaluate(k as f32 + u);
                    let got = sp.eval(u).pos;
                    let off = (got - Vec2::new(want[0], want[1])).length();
                    assert!(off < 1e-3, "m={m} span {k} at u={u}: off by {off}");
                }
            }
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

    /// Largest distance (canvas px) from any input sample to the fitted curve.
    fn fit_error(samples: &[InputSample]) -> f32 {
        let poly = flatten(&fit(samples), FLATTEN_TOLERANCE);
        samples
            .iter()
            .map(|s| {
                if poly.len() < 2 {
                    return (s.pos - poly[0].pos).length();
                }
                poly.windows(2)
                    .map(|w| point_segment_distance(s.pos, w[0].pos, w[1].pos))
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0, f32::max)
    }

    /// The fit may never ask for more control points than the samples can hold down.
    ///
    /// A fast pen reports tens of pixels apart, and a density policy that reads the
    /// input's curvature will happily ask for detail the data cannot support;
    /// granting it leaves the polygon under-determined and the curve wanders between
    /// the samples it passes through. The bound is structural rather than an explicit
    /// cap: a control point is only taken on if it *measurably* reduces the error, and
    /// one the data cannot see does not.
    #[test]
    fn the_fit_never_outruns_its_data() {
        for step in [4.0f32, 20.0, 50.0] {
            let n = (1500.0 / step) as usize;
            let pts: Vec<InputSample> = (0..n)
                .map(|i| {
                    let t = i as f32 * step;
                    sample(t, (t * 0.004).sin() * 120.0)
                })
                .collect();
            let m = fit(&pts).len();
            assert!(
                m <= pts.len(),
                "step {step}: {m} control points for {n} samples"
            );
            let err = fit_error(&pts);
            assert!(err < 16.0, "step {step}: err {err}");
        }
    }

    /// While a stroke is live, exactly `FREE_CONTROL_POINTS` of the polygon are
    /// still solvable — never zero, and never the whole thing.
    ///
    /// Both halves matter. Freezing that outruns the pointer leaves nothing able to
    /// respond to what is drawn next, so the stroke stops following the pen; freezing
    /// that never happens leaves the whole polygon re-solving on every report, which
    /// is what makes a live stroke wobble along its length. A fixed-size window is
    /// both at once, which is why growth and freezing are the same decision here.
    #[test]
    fn a_live_stroke_keeps_a_fixed_solvable_window() {
        for (name, src) in [
            ("spiral", stark_testdata::SPIRAL_STROKE),
            ("big-C", stark_testdata::BIG_C_STROKE),
            ("fast", stark_testdata::FAST_STROKE),
        ] {
            let pts: Vec<InputSample> = src.iter().map(|&[x, y]| sample(x, y)).collect();
            let mut f = PathFitter::new();
            let mut ever_froze = false;
            for p in &pts {
                f.push(*p);
                let m = f.path().len();
                if m > FREE_CONTROL_POINTS + 1 {
                    let free = m - f.frozen_spans().max(1);
                    assert!(
                        free >= FREE_CONTROL_POINTS,
                        "{name}: only {free} free of {m}"
                    );
                    ever_froze |= f.frozen_spans() > 0;
                }
            }
            assert!(ever_froze, "{name}: nothing ever froze");
        }
    }

    #[test]
    fn fit_starts_and_ends_under_the_pointer() {
        // A least-squares fit does not pin its ends: an unassigned stretch of
        // parameter costs nothing, so without saying so the curve runs out past both
        // ends of the stroke — by 20px on the very data below, before this was fixed.
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
    fn fit_collapses_pixel_staircase() {
        // A diagonal drawn as 1-px right / 1-px up steps.
        let mut stair = Vec::new();
        for i in 0..10 {
            stair.push(sample(i as f32, i as f32));
            stair.push(sample(i as f32 + 1.0, i as f32));
        }
        let fitted = fit(&stair);
        // The staircase hugs the diagonal within ~1px, so it collapses sharply.
        // The substantive claim is the error bound below — that the curve splits the
        // steps rather than following them. The count is a proxy and a loose one: six
        // control points over 10px is denser than the shape needs, but they all sit
        // on the diagonal.
        assert!(
            fitted.len() <= 8,
            "staircase should collapse, got {} points",
            fitted.len()
        );
        assert!(fit_error(&stair) < 1.5, "err {}", fit_error(&stair));
    }

    #[test]
    fn fit_flattens_coarse_staircase() {
        // A 2-px right / 2-px up staircase, e.g. a slow diagonal snapped to a 2px
        // device grid. Each corner sits ~1.4px off the diagonal. Fitting is what
        // smooths here — there is no separate low-pass stage — so this is really a
        // test of what happens when the price a control point pays sits *below* the
        // input's own quantization: the zigzag reads as curvature and gets traced.
        // `a_coarser_tolerance_smooths_what_a_finer_one_traces` is the other half —
        // the same staircase with the grid declared.
        let mut stair = vec![sample(0.0, 0.0)];
        for i in 0..12 {
            let b = (i * 2) as f32;
            stair.push(sample(b + 2.0, b)); // 2px right
            stair.push(sample(b + 2.0, b + 2.0)); // 2px up
        }
        // Nine control points for a 48px staircase, against the two the diagonal
        // underneath it is worth. Most of that gap used to be much wider — the
        // arc-length guess is not the curve's own parameterization, and the residual
        // that mismatch leaves reads as error the growth rule tries to buy away;
        // `SMOOTHING` is what now charges the polygon for the curvature it would buy.
        // What the count never showed either way is the *shape* — the curve splits
        // the corners rather than following them, which is what the error bound below
        // checks.
        // Traced, it would sit ~0 from every sample; smoothed, it splits the corners.
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
        // What the *shape* is worth, with the quantization taken out: the same
        // reports projected onto the diagonal they are a staircase of. Declaring the
        // grid can only buy back what the zigzag costs, never what the stroke is, so
        // that is the baseline both counts are measured from. Priced as a bare ratio
        // of totals instead, this read as a regression the moment `SMOOTHING` moved —
        // the finer fit had got better too, which is not the thing under test.
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

    /// The point of letting the caller state the tolerance: one gesture, drawn at
    /// different zoom levels, fits to one curve.
    ///
    /// Zoom scales canvas coordinates and the input's resolution *in* them by the
    /// same factor, so a declared tolerance makes the fit scale-invariant — the error
    /// price goes as its square and the spacing floor as its first power, which is
    /// exactly how a uniform scaling moves the two quantities they are each compared
    /// against. Priced in canvas px instead, the same stroke bought control points to
    /// trace its own jitter zoomed in and lost real detail zoomed out.
    #[test]
    fn a_declared_tolerance_makes_the_fit_zoom_invariant() {
        let screen: Vec<InputSample> = stark_testdata::BIG_C_STROKE
            .iter()
            .map(|&[x, y]| sample(x, y))
            .collect();
        let reference = fit_with_tolerance(&screen, DEFAULT_TOLERANCE);
        // Powers of two, so scaling the input is exact in f32 and any difference
        // that shows up is the growth rule's and not the arithmetic's.
        for zoom in [0.25f32, 4.0, 16.0] {
            let k = 1.0 / zoom;
            let pts: Vec<InputSample> = screen
                .iter()
                .map(|s| InputSample {
                    pos: s.pos * k,
                    ..*s
                })
                .collect();
            let got = fit_with_tolerance(&pts, DEFAULT_TOLERANCE * k);
            assert_eq!(
                got.len(),
                reference.len(),
                "zoom {zoom}: {} control points against {}",
                got.len(),
                reference.len()
            );
            for (i, (a, b)) in got.iter().zip(&reference).enumerate() {
                let d = (a.pos - b.pos * k).length();
                assert!(d < 1e-3 * k, "zoom {zoom}: control point {i} moved {d}");
            }
        }
    }

    /// The tolerance arrives from a frontend, so it is guarded rather than trusted.
    /// Zero or negative would make a control point free of charge and the spacing
    /// floor zero; huge would leave a stroke never growing the polygon, never
    /// freezing, and re-solving against every sample it ever took. Held to the usable
    /// range, each of these still fits a well-formed polygon bounded by its data (one
    /// growth per report, from a polygon that starts at two).
    #[test]
    fn a_degenerate_tolerance_is_held_to_the_usable_range() {
        let pts: Vec<InputSample> = (0..48).map(|i| sample(i as f32 * 3.0, 0.0)).collect();
        for t in [0.0, -5.0, f32::NAN, f32::INFINITY, 1e9] {
            let m = fit_with_tolerance(&pts, t).len();
            assert!(
                (2..=pts.len() + 1).contains(&m),
                "tolerance {t}: {m} control points"
            );
        }
        // A number that is not one falls back to the default rather than to an end of
        // the range — there is nothing to read from it in either direction.
        for t in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(fit_with_tolerance(&pts, t), fit(&pts), "tolerance {t}");
        }
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

    #[test]
    fn a_straight_stroke_stays_cheap_however_long_it_is() {
        // The arc-length floor is what lets freezing advance on a stroke the fit is
        // already perfect on, so a straight line does cost control points — but at
        // the floor's rate, not the refinement's.
        let long: Vec<InputSample> = (0..400).map(|i| sample(i as f32 * 7.5, 0.0)).collect();
        let fitted = fit(&long);
        // **Known weakness**, and the clearest statement of it: a dead-straight
        // stroke should cost a handful of control points and costs ~400, one per
        // sample. The growth rule is answering honestly — the arc-length guess is not
        // how a clamped B-spline is parameterized, so even exact input leaves a
        // residual, and a control point does reduce it. The fix is a correct
        // arc-to-parameter map, not a different price.
        assert!(
            fitted.len() <= long.len(),
            "more control points than samples"
        );
        assert!(
            fit_error(&long) < 0.5,
            "a straight line should still be straight"
        );
    }

    #[test]
    fn fit_streams_and_batches_alike() {
        let pts: Vec<InputSample> = (0..40)
            .map(|i| {
                let t = i as f32 * 0.2;
                sample(t * 10.0, (t * 1.3).sin() * 25.0)
            })
            .collect();
        let (mut streamed, _) = fit_incrementally(&pts);
        streamed.finish();
        assert_eq!(streamed.path(), fit(&pts));
    }

    /// The guarantee incremental repaint rests on: whatever the fitter reports as
    /// frozen, flattening it now equals flattening it at the end.
    #[test]
    fn committed_knots_are_never_revised() {
        // The core guarantee behind incremental repaint: whatever the fitter
        // reports as frozen, flattening it now equals flattening it at the end.
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
    fn the_curve_leaves_its_start_along_the_first_leg() {
        // The clamped end repeats the first control point three times, so the very
        // first Bézier point is a triple: the derivative *at* u = 0 is zero, and the
        // direction only becomes readable just after. What matters is that it heads
        // down the first leg once it does.
        let knots = [knot(0.0, 0.0), knot(30.0, 0.0), knot(60.0, 20.0)];
        let head = span(&knots, 0).eval(0.25);
        assert!(head.vel.length() > 1e-3, "start derivative {:?}", head.vel);
        assert!(head.vel.normalize().dot(Vec2::X) > 0.99);
    }

    #[test]
    fn flatten_cost_follows_the_polygon_not_the_length() {
        // The point of adaptive sampling: a straight run costs the same however long
        // it is. Uniform arc-length sampling spent 500 samples on this one.
        let short = flatten(&[knot(0.0, 0.0), knot(10.0, 0.0)], FLATTEN_TOLERANCE);
        let long = flatten(&[knot(0.0, 0.0), knot(1000.0, 0.0)], FLATTEN_TOLERANCE);
        assert_eq!(short.len(), long.len());
        // One sample per span plus the start; two control points give three spans
        // (`span_count`), of which the two clamped ends are geometrically slivers.
        assert_eq!(long.len(), span_count(2) + 1, "got {} samples", long.len());
        assert_eq!(long[0].pos, Vec2::ZERO);
        // The end is the last control point, reached through the Bézier conversion,
        // so it lands there to rounding rather than bit-exactly.
        assert!((long.last().unwrap().pos - Vec2::new(1000.0, 0.0)).length() < 1e-3);
        assert!((long.last().unwrap().dist - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn flatten_stays_within_the_position_budget() {
        let knots = [
            knot(0.0, 0.0),
            knot(20.0, 30.0),
            knot(60.0, 30.0),
            knot(90.0, 0.0),
            knot(120.0, -40.0),
        ];
        let poly = flatten(&knots, FLATTEN_TOLERANCE);
        // Every point of the true curve is within the budget of the polyline (a
        // little slack for the midpoint-only test the sampler uses).
        for i in 0..knots.len() - 1 {
            let sp = span(&knots, i);
            for s in 0..=64 {
                let p = sp.eval(s as f32 / 64.0).pos;
                let d = poly
                    .windows(2)
                    .map(|w| point_segment_distance(p, w[0].pos, w[1].pos))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    d < FLATTEN_TOLERANCE.position * 2.0,
                    "curve point {p:?} is {d}px off the polyline",
                );
            }
        }
    }

    #[test]
    fn flatten_spends_samples_where_the_curve_bends() {
        // Two strokes of the same length: one gentle, one tight. The tight one
        // must cost far more samples — the whole point of bounding error rather
        // than arc length.
        let gentle = [knot(0.0, 0.0), knot(200.0, 6.0), knot(400.0, 0.0)];
        let tight = [knot(0.0, 0.0), knot(200.0, 160.0), knot(400.0, 0.0)];
        let g = flatten(&gentle, FLATTEN_TOLERANCE).len();
        let t = flatten(&tight, FLATTEN_TOLERANCE).len();
        assert!(t > g * 3, "gentle {g} samples vs tight {t}");
        // And both are far under what a uniform 2px walk would have cost (~200).
        assert!(g < 40, "a gentle 400px stroke took {g} samples");
    }

    #[test]
    fn flatten_honours_the_length_cap() {
        let knots = [knot(0.0, 0.0), knot(300.0, 0.0)];
        let tol = FlattenTolerance {
            max_len: 10.0,
            ..FLATTEN_TOLERANCE
        };
        let poly = flatten(&knots, tol);
        for w in poly.windows(2) {
            let d = (w[1].pos - w[0].pos).length();
            assert!(d <= 10.0 + 1e-3, "segment of {d}px exceeds the 10px cap");
        }
    }

    #[test]
    fn flatten_splits_on_a_pressure_ramp() {
        // A dead-straight stroke whose pressure sweeps 0 → 1: geometry alone would
        // emit one segment, but radius follows pressure, so it must not.
        let knots: Vec<ControlPoint> = (0..2)
            .map(|i| ControlPoint {
                pressure: i as f32,
                ..knot(i as f32 * 200.0, 0.0)
            })
            .collect();
        let poly = flatten(&knots, FLATTEN_TOLERANCE);
        for w in poly.windows(2) {
            let d = (w[1].pressure - w[0].pressure).abs();
            assert!(
                d <= FLATTEN_TOLERANCE.attribute + 1e-4,
                "pressure step of {d} exceeds the budget",
            );
        }
    }

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

    /// The marker trim (§6.2): [`flatten_spans_from`] starts the polyline at
    /// exactly the asked parameter with the accumulator at `dist0`, leaves
    /// everything behind it out, and still tiles with later ranges — while a
    /// marker at or before the range is the untrimmed call, so every
    /// `start == 0` record keeps the floats it always flattened to.
    #[test]
    fn flattening_from_a_marker_trims_and_only_trims() {
        let knots = [
            knot(0.0, 0.0),
            knot(20.0, 30.0),
            knot(60.0, 30.0),
            knot(90.0, 0.0),
            knot(120.0, -40.0),
        ];
        let all = span_count(knots.len());
        let whole = flatten(&knots, FLATTEN_TOLERANCE);

        let from = 2.4_f32;
        let cut = flatten_spans_from(&knots, from, 0..all, 0.0, FLATTEN_TOLERANCE);
        let first = cut.first().expect("a trimmed polyline still starts");
        assert_eq!(
            first.pos,
            point_at(&knots, from),
            "the polyline must start at the marker"
        );
        assert_eq!(first.dist, 0.0, "the accumulator reads dist0 at the marker");
        assert!(
            cut.windows(2).all(|w| w[0].dist <= w[1].dist),
            "arc must accumulate along the trimmed polyline"
        );
        assert_eq!(
            cut.last().unwrap().pos,
            whole.last().unwrap().pos,
            "the tail past the marker is untouched"
        );

        // A range entirely behind the marker: spans of the curve, none of the
        // stroke. A marker past the whole curve leaves nothing at all.
        assert!(flatten_spans_from(&knots, from, 0..2, 0.0, FLATTEN_TOLERANCE).is_empty());
        assert!(flatten_spans_from(&knots, all as f32, 0..all, 0.0, FLATTEN_TOLERANCE).is_empty());

        // Ranges still tile around a marker mid-range: the cut point is shared.
        let head = flatten_spans_from(&knots, from, 0..4, 0.0, FLATTEN_TOLERANCE);
        let tail = flatten_spans_from(
            &knots,
            from,
            4..all,
            head.last().unwrap().dist,
            FLATTEN_TOLERANCE,
        );
        let joined: Vec<IntermediateSample> =
            head.iter().chain(tail[1..].iter()).copied().collect();
        assert_eq!(
            joined, cut,
            "trimmed head + tail must equal the trimmed whole"
        );
    }

    #[test]
    fn arc_length_accumulates_along_the_polyline() {
        let knots = [knot(0.0, 0.0), knot(40.0, 40.0), knot(80.0, 0.0)];
        let poly = flatten(&knots, FLATTEN_TOLERANCE);
        assert_eq!(poly[0].dist, 0.0);
        for w in poly.windows(2) {
            let step = (w[1].pos - w[0].pos).length();
            assert!((w[1].dist - w[0].dist - step).abs() < 1e-3);
        }
    }

    #[test]
    fn relaxing_the_budget_costs_fewer_samples() {
        let knots = [knot(0.0, 0.0), knot(60.0, 80.0), knot(160.0, 0.0)];
        let fine = flatten(&knots, FLATTEN_TOLERANCE).len();
        let coarse = flatten(&knots, FLATTEN_TOLERANCE.relaxed(8.0)).len();
        assert!(coarse < fine, "relaxed {coarse} vs fine {fine}");
    }
}
