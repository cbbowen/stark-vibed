//! Building a fit up as points arrive, freezing control points as they settle.

use nalgebra::{Const, Dyn, OMatrix};

use crate::{CardinalCubicBSpline, FitTolerance, Settled, Variable};

/// A fit that is built up over time: points are appended in order, the control polygon
/// grows to cover them, and a prefix of control points that has stopped moving is frozen
/// so later updates only solve for the tail.
///
/// The work per update is proportional to the points still *live* — those whose assignment
/// can still change — rather than to every point ever seen, which is what makes a long
/// stream affordable. Freezing is what retires points: once the leading control points are
/// fixed, the curve up to [`ClampedCardinalBSpline::frozen_until`] is final, so the points
/// assigned there have a final position and a final error and drop out of the fit. Their
/// ordering is not forgotten — the points that remain stay constrained to come after them
/// (see [`Settled::after`]).
///
/// Freezing is a *decision*, not an inference, so it is left to the caller: [`Self::freeze`]
/// commits to a prefix, and [`Self::freeze_all_but`] expresses the policy that works well in
/// practice — keep the last few control points free, since those are the ones the incoming
/// data still pulls on, and freeze everything behind them.
///
/// It is tempting to freeze instead on the evidence of [`Self::settled_prefix`] — the
/// control points that stopped moving — but on this spline that does not bootstrap. Nothing
/// pins the parameterization, so while the polygon is entirely free the fit keeps sliding
/// the whole curve by small amounts and the settled prefix stays near zero. Freezing is what
/// stops the sliding: once a prefix is committed, the settled prefix tracks just past it. So
/// drive the loop by the lag and treat movement as a diagnostic (or, with an `eps` scaled
/// generously to the data, as a brake on a lag that is too aggressive).
///
/// ```
/// # use spline_fit::{FitTolerance, IncrementalFit};
/// # let batches: Vec<Vec<[f64; 2]>> = (0..8)
/// #     .map(|b| (0..25).map(|i| {
/// #         let t = (b * 25 + i) as f64 * 0.02;
/// #         [t, (t * 2.0).sin()]
/// #     }).collect())
/// #     .collect();
/// let tol = FitTolerance::default();
/// // One control point per 10 points, and keep the last 4 free — those still move.
/// let control_points = |n: usize| (n / 10).max(2);
/// let mut fit = IncrementalFit::<f64, 2>::new(control_points(batches[0].len()), tol, &batches[0]);
/// for batch in &batches[1..] {
///     fit.extend(control_points(fit.points().len() + batch.len()), batch);
///     fit.freeze_all_but(4);
/// }
/// # assert!(fit.retired() > 100);   // most points have left the fit
/// let (spline, assignment, error) = fit.finish();
/// # assert_eq!(assignment.len(), 200);
/// # assert!(error < 1.0);
/// # let _ = spline.evaluate(0.0);
/// ```
///
/// [`ClampedCardinalBSpline::frozen_until`]: crate::ClampedCardinalBSpline::frozen_until
pub struct IncrementalFit<T: Variable, const D: usize> {
    spline: CardinalCubicBSpline<T, Const<D>>,
    tol: FitTolerance<T>,
    settled: Settled<T>,
    /// Every point, in order; `points[..retired]` are no longer part of the fit.
    points: Vec<[T; D]>,
    /// The assignment of `points`, same length; `ts[..retired]` can no longer change.
    ts: Vec<T>,
    retired: usize,
    /// Squared error of the retired points, which the frozen curve keeps fixed.
    retired_error: T,
    /// Squared error of the live points, as of the last update.
    live_error: T,
    /// Control points as they stood going into the last update, for [`Self::settled_prefix`].
    previous: Option<OMatrix<T, Dyn, Const<D>>>,
}

impl<T: Variable, const D: usize> IncrementalFit<T, D> {
    /// Starts an incremental fit: an ordinary [`CardinalCubicBSpline::fit_monotonic`] of `m`
    /// control points to the points available so far, with nothing frozen yet.
    ///
    /// # Panics
    ///
    /// Panics if `m < 2` or if `points` is empty.
    pub fn new(m: usize, tol: FitTolerance<T>, points: &[[T; D]]) -> Self {
        let (spline, ts, live_error) = CardinalCubicBSpline::fit_monotonic(m, tol, points);
        IncrementalFit {
            spline,
            tol,
            settled: Settled::none(),
            points: points.to_vec(),
            ts,
            retired: 0,
            retired_error: T::zero(),
            live_error,
            previous: None,
        }
    }

    /// Appends `points`, grows the control polygon to `m` control points, and refits the
    /// live tail — leaving the frozen prefix, and the curve it determines, untouched.
    ///
    /// `m` is the caller's growth policy: the control polygon has to lengthen as the curve
    /// does, and how fast is the same accuracy-versus-smoothness tradeoff as choosing `m`
    /// for a one-shot fit. Values at or below the current count leave it as it is; it never
    /// shrinks, since the frozen prefix would have nowhere to go.
    ///
    /// Appending no points is allowed, and simply refits what is already there.
    pub fn extend(&mut self, m: usize, points: &[[T; D]]) {
        self.points.extend_from_slice(points);
        self.spline.extend_control_points(m, points);
        self.previous = Some(self.spline.control_points().clone());
        if self.retired == self.points.len() {
            // Everything so far is final and nothing new arrived; there is nothing to fit.
            return;
        }
        let live = &self.points[self.retired..];
        let (ts, err) = self.spline.refit_monotonic(self.settled, self.tol, live);
        self.ts.truncate(self.retired);
        self.ts.extend(ts);
        self.live_error = err;
    }

    /// The number of leading control points that did not move during the last
    /// [`Self::extend`], to within `eps` in every coordinate — the fit's own evidence about
    /// how much of the curve has settled. Zero before the first update.
    ///
    /// Evidence, not proof, in both directions. A control point that held still can still be
    /// moved by points that arrive later; and one that moved may only have been drifting
    /// with the reparameterization the fit is free to make while nothing is frozen. Control
    /// points that were already frozen going into the update count towards the prefix
    /// trivially, since they could not move at all. See the type docs for how to use it.
    pub fn settled_prefix(&self, eps: T) -> usize {
        let Some(previous) = &self.previous else {
            return 0;
        };
        let current = self.spline.control_points();
        let common = previous.nrows().min(current.nrows());
        (0..common)
            .take_while(|&j| (0..D).all(|d| (current[(j, d)] - previous[(j, d)]).abs() <= eps))
            .count()
    }

    /// Freezes the leading `control_points` control points, and retires every point whose
    /// assignment has landed in the part of the curve they determine.
    ///
    /// Retired points keep their place in [`Self::assignment`] and their share of
    /// [`Self::error`], but leave the fit: later updates neither move them nor pay for them.
    /// Freezing only ever advances — a `control_points` below what is already frozen is
    /// ignored, and one above the control point count is capped.
    pub fn freeze(&mut self, control_points: usize) {
        let frozen = control_points.clamp(
            self.settled.control_points,
            self.spline.num_control_points(),
        );
        self.settled.control_points = frozen;

        // Points assigned within the frozen curve can neither move nor change their error.
        // The last of them becomes the ordering floor for everything still live.
        let until = self.spline.frozen_until(frozen);
        let was = self.retired;
        while self.retired < self.ts.len() && self.ts[self.retired] <= until {
            self.retired_error += self.point_error(self.retired);
            self.settled.after = self.ts[self.retired];
            self.retired += 1;
        }
        if self.retired > was {
            // The retired points' share of the error has moved out of the live total.
            self.live_error = (self.retired..self.points.len())
                .map(|i| self.point_error(i))
                .fold(T::zero(), T::add);
        }
    }

    /// Freezes everything but the last `lag` control points — the lag policy of the type
    /// docs, and the usual way to drive an incremental fit.
    ///
    /// `lag` is the accuracy knob: it is how far back the fit stays revisable, so a larger
    /// lag keeps more of the curve able to respond to what arrives next, at the cost of
    /// carrying more points and solving for more control points on every update.
    pub fn freeze_all_but(&mut self, lag: usize) {
        self.freeze(self.spline.num_control_points().saturating_sub(lag));
    }

    /// The squared distance from point `i` to the curve at its assigned parameter.
    fn point_error(&self, i: usize) -> T {
        let v = self.spline.evaluate(self.ts[i]);
        (0..D)
            .map(|d| (v[d] - self.points[i][d]).powi(2))
            .fold(T::zero(), T::add)
    }

    /// The spline as it currently stands.
    pub fn spline(&self) -> &CardinalCubicBSpline<T, Const<D>> {
        &self.spline
    }

    /// Every point given to the fit, in order.
    pub fn points(&self) -> &[[T; D]] {
        &self.points
    }

    /// The monotonic assignment of [`Self::points`] to curve parameters. Entries before
    /// [`Self::retired`] are final.
    pub fn assignment(&self) -> &[T] {
        &self.ts
    }

    /// The total squared error over every point, retired ones included.
    pub fn error(&self) -> T {
        self.retired_error + self.live_error
    }

    /// How many leading control points are frozen.
    pub fn frozen(&self) -> usize {
        self.settled.control_points
    }

    /// How many leading points have been retired: they are no longer part of the fit, and
    /// their assignment and error are final.
    pub fn retired(&self) -> usize {
        self.retired
    }

    /// The finished spline, the assignment of every point, and the total squared error.
    pub fn finish(self) -> (CardinalCubicBSpline<T, Const<D>>, Vec<T>, T) {
        let error = self.error();
        (self.spline, self.ts, error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wiggly curve sampled in order, as a stream of equal batches.
    fn batches(count: usize, per_batch: usize) -> Vec<Vec<[f64; 2]>> {
        let n = count * per_batch;
        let point = |i: usize| {
            let t = i as f64 / (n - 1) as f64;
            [t * 20.0, (t * 9.0).sin() * 3.0 + (t * 31.0).cos() * 0.4]
        };
        (0..count)
            .map(|b| (0..per_batch).map(|i| point(b * per_batch + i)).collect())
            .collect()
    }

    /// One control point per 10 points, as the doc example's growth policy.
    fn control_points(n: usize) -> usize {
        (n / 10).max(2)
    }

    fn tolerance() -> FitTolerance<f64> {
        FitTolerance::new(1e-6, 1e-4, 1e-9)
    }

    /// Drives the loop the type docs recommend: extend, then freeze all but `lag` control
    /// points.
    fn run(count: usize, per_batch: usize, lag: usize) -> IncrementalFit<f64, 2> {
        let batches = batches(count, per_batch);
        let mut fit = IncrementalFit::new(control_points(per_batch), tolerance(), &batches[0]);
        for batch in &batches[1..] {
            let n = fit.points().len() + batch.len();
            fit.extend(control_points(n), batch);
            fit.freeze_all_but(lag);
        }
        fit
    }

    /// The freezing policies, from aggressive to none, against the one-shot fit of the same
    /// data. Run with `--nocapture` to see what a lag costs; the assertions only hold the
    /// aggressive end of the range to account.
    #[test]
    fn freezing_lag_trades_work_against_error() {
        let batches = batches(8, 25);
        let all: Vec<[f64; 2]> = batches.iter().flatten().copied().collect();
        let m = control_points(all.len());
        let (_, _, batch_err) =
            CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(m, tolerance(), &all);
        eprintln!(
            "one-shot fit of all {} points: m={m} err={batch_err:.6}",
            all.len()
        );
        for lag in [2usize, 4, 8, usize::MAX] {
            let mut fit = IncrementalFit::new(control_points(25), tolerance(), &batches[0]);
            for batch in &batches[1..] {
                let n = fit.points().len() + batch.len();
                fit.extend(control_points(n), batch);
                fit.freeze_all_but(lag);
            }
            eprintln!(
                "lag {lag:>3}: frozen={:>2} retired={:>3}/{} err={:.6}",
                fit.frozen(),
                fit.retired(),
                fit.points().len(),
                fit.error(),
            );
            // Even the most aggressive policy — two free control points, everything else
            // committed — stays in the same league as fitting all the points at once.
            assert!(
                fit.error() <= batch_err * 3.0 + 1e-3,
                "lag {lag}: err {} far off the one-shot fit's {batch_err}",
                fit.error()
            );
            if lag <= 8 {
                assert!(
                    fit.retired() > fit.points().len() / 2,
                    "lag {lag}: only {} of {} points retired",
                    fit.retired(),
                    fit.points().len()
                );
            }
        }
    }

    #[test]
    fn an_incremental_fit_tracks_a_batch_fit() {
        let batches = batches(8, 25);
        let all: Vec<[f64; 2]> = batches.iter().flatten().copied().collect();
        let m = control_points(all.len());

        let fit = run(8, 25, 4);
        assert_eq!(fit.points(), all.as_slice());
        assert_eq!(fit.assignment().len(), all.len());
        assert!(
            fit.assignment().windows(2).all(|w| w[0] <= w[1] + 1e-9),
            "assignment not monotonic: {:?}",
            fit.assignment()
        );
        assert!(fit.frozen() > 0, "nothing ever settled — test is vacuous");
        assert!(fit.retired() > 0, "no points were ever retired");
        assert_eq!(fit.spline().num_control_points(), m);

        // The freezing is what costs accuracy: it commits to control points the batch fit
        // would still have been free to move. It should cost little.
        let (_, _, batch_err) =
            CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(m, tolerance(), &all);
        let (_, _, err) = fit.finish();
        assert!(
            err <= batch_err * 3.0 + 1e-3,
            "incremental err {err} far off the batch fit's {batch_err}"
        );
    }

    /// The squared error of `fit`'s first `n` points, evaluated directly against the curve.
    fn error_of_first(fit: &IncrementalFit<f64, 2>, n: usize) -> f64 {
        fit.assignment()[..n]
            .iter()
            .zip(fit.points())
            .map(|(&t, p)| {
                let v = fit.spline().evaluate(t);
                (v[0] - p[0]).powi(2) + (v[1] - p[1]).powi(2)
            })
            .sum()
    }

    #[test]
    fn a_retired_point_keeps_its_assignment_and_its_error() {
        let batches = batches(8, 25);
        let mut fit = IncrementalFit::new(control_points(25), tolerance(), &batches[0]);
        let mut checked = false;
        for batch in &batches[1..] {
            let retired = fit.retired();
            let assignment_before: Vec<f64> = fit.assignment()[..retired].to_vec();
            let error_before = error_of_first(&fit, retired);

            fit.extend(control_points(fit.points().len() + batch.len()), batch);
            fit.freeze_all_but(4);

            // Retirement is final: neither the parameters nor the curve under them can move
            // again, however far the free tail travels — so the error already booked for
            // them stands, to the last bit.
            assert_eq!(&fit.assignment()[..retired], assignment_before.as_slice());
            assert_eq!(error_of_first(&fit, retired), error_before);
            assert!(fit.retired() >= retired, "retirement must not go backwards");
            checked |= retired > 0;
        }
        assert!(checked, "nothing was ever retired — test is vacuous");
    }

    #[test]
    fn the_settled_prefix_covers_the_frozen_one() {
        // Control points frozen going into an update cannot move during it, so they pass the
        // movement test at any tolerance — which is what makes `settled_prefix` usable as a
        // brake on a lag policy rather than something that can pull the fit backwards.
        let batches = batches(6, 25);
        let mut fit = IncrementalFit::new(control_points(25), tolerance(), &batches[0]);
        assert_eq!(fit.settled_prefix(1e-3), 0, "no update has happened yet");
        let mut checked = false;
        for batch in &batches[1..] {
            let frozen = fit.frozen();
            fit.extend(control_points(fit.points().len() + batch.len()), batch);
            assert!(
                fit.settled_prefix(0.0) >= frozen,
                "settled {} below the {frozen} frozen during the update",
                fit.settled_prefix(0.0)
            );
            checked |= frozen > 0;
            fit.freeze_all_but(4);
        }
        assert!(checked, "nothing was ever frozen — test is vacuous");
    }

    #[test]
    fn never_freezing_matches_refitting_everything() {
        // With `freeze` never called, every point stays live and every control point stays
        // free: the fit is then just a warm-started full fit, and must be at least as good
        // as the cold-started one it can always fall back to.
        let batches = batches(6, 20);
        let all: Vec<[f64; 2]> = batches.iter().flatten().copied().collect();
        let m = control_points(all.len());
        let mut fit = IncrementalFit::new(control_points(20), tolerance(), &batches[0]);
        for batch in &batches[1..] {
            fit.extend(control_points(fit.points().len() + batch.len()), batch);
        }
        assert_eq!(fit.frozen(), 0);
        assert_eq!(fit.retired(), 0);

        let (spline, ts, err) = fit.finish();
        let direct: f64 = ts
            .iter()
            .zip(&all)
            .map(|(&t, p)| {
                let v = spline.evaluate(t);
                (v[0] - p[0]).powi(2) + (v[1] - p[1]).powi(2)
            })
            .sum();
        assert!(
            (err - direct).abs() <= 1e-9 * (1.0 + direct),
            "reported error {err} is not what the assignment achieves ({direct})"
        );
        let (_, _, batch_err) =
            CardinalCubicBSpline::<f64, Const<2>>::fit_monotonic(m, tolerance(), &all);
        assert!(
            err <= batch_err * 1.5 + 1e-6,
            "unfrozen incremental err {err} far off the batch fit's {batch_err}"
        );
    }

    #[test]
    fn freezing_everything_still_assigns_new_points() {
        // The degenerate policy: freeze the whole control polygon. Later points can then
        // only be assigned against a curve that no longer responds to them, which must
        // still produce a valid monotone assignment rather than falling over.
        let batches = batches(4, 20);
        let mut fit = IncrementalFit::new(4, tolerance(), &batches[0]);
        fit.freeze(usize::MAX);
        assert_eq!(fit.frozen(), 4);
        for batch in &batches[1..] {
            fit.extend(6, batch);
            fit.freeze(usize::MAX);
        }
        let (spline, ts, err) = fit.finish();
        assert_eq!(ts.len(), 80);
        assert!(ts.windows(2).all(|w| w[0] <= w[1] + 1e-9));
        assert!(err.is_finite());
        // Growth still happened; it is only the *values* that were frozen.
        assert_eq!(spline.num_control_points(), 6);
    }
}
