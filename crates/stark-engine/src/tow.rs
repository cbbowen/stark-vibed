//! The towed tip — per-brush stroke smoothing (§6.11).
//!
//! The mark is drawn by a tip towed behind the pointer on a string of fixed
//! length. While the pointer wanders within the rope of the tip, the string is
//! slack and the tip is **parked** — jitter and hesitation never move it at
//! all. The moment it comes taut the tip is dragged, and a dragged tip traces
//! the classical pursuit curve, the **tractrix**. The pen-up **parks the tip**:
//! lifting stops pulling the string, it does not reel the tip in, so the mark
//! ends where the rope had towed it to — which is the trace the preview was
//! already showing at the release (§6.11).
//!
//! This is an *input* transform: it sits between the raw pointer reports and
//! [`PathFitter::push`](crate::path::PathFitter::push), one stage upstream of
//! the fitter-to-renderer seam the drawing assist attaches at (§6.9). What
//! lands in `StrokeRecord::path` is the towed, fitted control points, so
//! nothing downstream — record, renderer, save format, replay, peers — learns
//! that smoothing exists.
//!
//! **The tow is integrated exactly, per straight run of the target, not
//! stepped per report.** For a target moving along a line the taut tow has a
//! closed form — the angle `θ` between string and travel obeys
//! `tan(θ/2) = tan(θ₀/2)·exp(−s/L)` — and the slack→taut crossing within a run
//! is a quadratic. The tip's *trajectory* is therefore a function of the
//! pointer's path alone, not of its report clock: cutting a run in two
//! composes, because the exponential is exponential in arc (§6.2's
//! partition-independence discipline, applied to input). The samples emitted
//! along that trajectory do land on a per-run grid, but they are always *on*
//! the trajectory, and the fitter downstream smooths through sampling.
//!
//! Transcendentals (`exp`, `sqrt`) are fine here, and the boundary is worth
//! stating: the tow runs **once, on the originating client, upstream of the
//! record** — the same class of computation as the fitter's own least-squares
//! solve. §12.1's bit-agreement rules (the reason `taper_profile` is a
//! polynomial) reach only what is derived *from* the record.

use crate::command::InputSample;
use stark_model::geom::Vec2;

/// Emission spacing while the tip is being dragged, as a fraction of the rope,
/// measured along the **target's** travel. The tractrix bend mostly completes
/// within a couple of rope-lengths of travel, so a quarter-rope grid gives the
/// fitter several samples across it; the fit smooths through the rest.
///
/// How far the grid runs is [`bend_reach`], and the two are only meaningful
/// together: this one is a fraction *of the rope*, so on its own it makes the cost
/// of a pointer report inverse in the smoothing knob.
const EMIT_SPACING: f32 = 0.25;

/// How far past the taut crossing that grid is worth laying, in canvas px — the
/// distance over which the tip is still *bending*, rather than trailing dead
/// astern on a straight line the fitter would draw from two samples anyway.
///
/// **The spacing is a fraction of the rope, so without this the number of samples
/// one pointer report costs goes as `1/rope`** — and a smoothing knob near zero is
/// a rope near zero. Measured on the frontend's own mapping (`rope_in`: the 0..1
/// amount squared, times 160 screen px), one 4 px report at 8× zoom cost 320 fitter
/// pushes at amount 0.05 and 2000 at amount 0.02 — 15 ms and 445 ms of fitting for
/// a single pointer move, natively. That is the whole of why a near-zero smoothing
/// setting froze: not the fit being slow, but the input stage handing it thousands
/// of samples per report. The loop also stopped terminating once the step fell under
/// the f32 ULP of the distance it was accumulating into.
///
/// The bound is read off the tractrix rather than picked. The tip's offset from its
/// straight asymptote is `2·rope·t/√(1+t²) ≤ 2·rope·t`, and the half-angle decays as
/// `t = t₀·exp(−Δs/rope)` with `t₀ ≤ 1` — so the bend is under the input's own tolerance,
/// and therefore invisible to the fit that tolerance prices ([`clamp_tolerance`](crate::path::clamp_tolerance)),
/// after `rope · ln(2·rope/tolerance)`. Past it the run's final emission is the only one
/// that carries anything.
///
/// Two properties follow, and both are the point:
///
/// * **A report costs `4·ln(2·rope/tolerance) + 1` emissions at worst** — under 30 across
///   the whole reachable range of the knob, at any zoom, because rope and tolerance are
///   both carried through the view and the ratio is what survives. Bounded in the
///   rope rather than inverse in it.
/// * **A rope at or under half the tolerance reaches zero**, so the tow emits exactly one
///   sample per report: the same rate as no tow at all, which is the right answer for
///   a string shorter than the pointer can resolve. The knob's bottom end degrades to
///   the untowed path instead of falling off a cliff into it.
fn bend_reach(rope: f32, tolerance: f32) -> f32 {
    (2.0 * rope / tolerance).ln().max(0.0) * rope
}

/// The in-flight string, for the frontend's overlay (§6.11): drawn from the
/// towed tip to the pointer, sagging while slack and straight while taut.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TowString {
    /// Where the mark is being laid (canvas px).
    pub tip: Vec2,
    /// The last raw report — where the hand is (canvas px).
    pub target: Vec2,
    /// The string's length (canvas px). `|target − tip| < rope` is slack.
    pub rope: f32,
}

/// The towed tip: a stateful transform on the raw [`InputSample`] stream.
///
/// Feed every report through [`to`](Self::to); it emits zero or more towed
/// samples for the fitter (none while the tip is parked). There is nothing to
/// do at pen-up: the last emission already *is* the tip, and lifting the pen
/// stops pulling the string rather than reeling it in (§6.11).
#[derive(Clone, Debug)]
pub struct Tow {
    rope: f32,
    /// The input's own positional resolution, in canvas px — the same tolerance the
    /// fit prices its thresholds in (`PathFitter::with_tolerance`), held to the
    /// same bounds. What it decides here is how far the emission grid runs: see
    /// [`bend_reach`].
    tolerance: f32,
    tip: Vec2,
    /// The last raw report — the target the string runs to. Kept whole (not
    /// just the position) because a run's emissions interpolate the pen
    /// channels across it.
    target: InputSample,
}

impl Tow {
    /// A tow of `rope` canvas px over input that resolves to `tolerance` canvas px,
    /// tip parked on the first report.
    ///
    /// Callers gate construction on `rope > 0` — a rope of zero is no tow, and
    /// the session simply feeds the fitter directly (bit-identical to the
    /// pre-§6.11 path). A rope that is merely *small* needs no gate of its own:
    /// [`bend_reach`] takes it to one emission a report, which is that same
    /// rate.
    ///
    /// # Panics
    ///
    /// In debug, on a `rope` that is not positive and finite. It is the caller's
    /// gate and not a repair here, because the two answers are different tools: a
    /// zero rope means *do not build one of these*, where silently substituting some
    /// positive rope would put a smoothing the artist switched off back into the
    /// stroke. Stated as an assertion so a caller that forgets the gate finds out at
    /// the door rather than in [`bend_reach`]'s arithmetic.
    ///
    /// `tolerance` is the caller's declared input resolution — the one thing the tow
    /// cannot work out for itself, and the same number the fitter is built with —
    /// so it is held to the same bounds by the same function, rather than by a
    /// second copy of them here.
    pub fn new(rope: f32, tolerance: f32, first: InputSample) -> Self {
        debug_assert!(
            rope.is_finite() && rope > 0.0,
            "a tow of {rope} is not a tow; the caller gates on rope > 0"
        );
        Self {
            rope,
            tolerance: crate::path::clamp_tolerance(tolerance),
            tip: first.pos,
            target: first,
        }
    }

    /// The string as the overlay draws it.
    pub fn string(&self) -> TowString {
        TowString {
            tip: self.tip,
            target: self.target.pos,
            rope: self.rope,
        }
    }

    /// Advance the target to `next`, dragging the tip along the straight run
    /// from the previous report; every towed sample is handed to `emit` in
    /// order. A run that never brings the string taut emits nothing — the tip
    /// is parked, which is the dead zone doing its job.
    ///
    /// Attributes ride the pen, not the arc (§6.11): an emitted sample carries
    /// the report stream's own attributes at that point of the *run*, so the
    /// final emission of a run carries exactly `next`'s.
    pub fn to(&mut self, next: InputSample, emit: &mut impl FnMut(InputSample)) {
        let prev = self.target;
        self.target = next;
        let run = next.pos - prev.pos;
        let len = run.length();
        // A report that did not move tows nothing — the same answer the fitter
        // gives a stationary report (§6.2), stated here so the taut math never
        // sees a zero (or garbage) direction.
        if !len.is_normal() || len <= 0.0 {
            return;
        }
        let u = run / len;
        // Float error can leave the tip a hair outside the rope; restore the
        // invariant the closed form assumes before reading the geometry.
        let d = prev.pos - self.tip;
        let dist = d.length();
        if dist > self.rope {
            self.tip = prev.pos - d * (self.rope / dist);
        }
        // Slack: the target runs on alone until the string comes taut — the
        // larger root of `|d + s·u|² = rope²`. One expression covers every
        // case: a target moving *toward* the tip deepens the slack and pays it
        // out on the far side, and a string already taut and receding crosses
        // at `s = 0`.
        let d = prev.pos - self.tip;
        let b = d.dot(u);
        let c = d.length_squared() - self.rope * self.rope; // ≤ 0 by the invariant
        let s0 = -b + (b * b - c).max(0.0).sqrt();
        if s0 >= len {
            return; // still slack at the end of the run: the tip stays parked
        }
        // Taut: the tip is dragged from `s0` to `len`. At the crossing the
        // distance is increasing, so `cos θ₀ ≥ 0` and the half-angle
        // `t = sin θ / (1 + cos θ)` starts in [0, 1]; it then decays as
        // `exp(−s/rope)` — the tractrix, integrated exactly however the
        // pointer's path was cut into reports.
        let p0 = prev.pos + u * s0;
        let w = (p0 - self.tip) / self.rope; // unit: tip → target, on the string
        let cos0 = w.dot(u).clamp(-1.0, 1.0);
        let perp = u.perp();
        let side = w.dot(perp);
        // The side the tip trails on is preserved: θ only decays, never
        // crosses zero.
        let n = if side >= 0.0 { perp } else { -perp };
        let t0 = side.abs() / (1.0 + cos0);
        // Where the tip is `s` into the run, and what the pen said there: the
        // tractrix in closed form, with the attributes read off the *report*
        // stream at that fraction of the run (§6.11).
        let rope = self.rope;
        let at = |s: f32| {
            let t = t0 * (-(s - s0) / rope).exp();
            let m = 1.0 + t * t;
            let w = u * ((1.0 - t * t) / m) + n * (2.0 * t / m);
            let f = s / len;
            InputSample {
                pos: prev.pos + u * s - w * rope,
                pressure: prev.pressure + (next.pressure - prev.pressure) * f,
                tilt: prev.tilt.lerp(next.tilt, f),
                time: prev.time + (next.time - prev.time) * f as f64,
            }
        };
        let mut lay = |tip: &mut Vec2, s: f32| {
            let sample = at(s);
            *tip = sample.pos;
            emit(sample);
        };
        // The grid, laid over the bend and no further ([`bend_reach`]), then the
        // run's end — which is where the tip actually finishes and so is always
        // emitted, grid or no grid.
        //
        // **Counted, not accumulated.** The old loop walked `s += step` until it
        // reached `len`, which is a loop whose trip count is `1/rope` and which
        // does not terminate at all once `step` falls under the f32 ULP of `s`.
        // A count computed up front cannot do either: it is bounded by
        // [`bend_reach`]'s own logarithm, and it is an integer.
        let step = self.rope * EMIT_SPACING;
        let reach = bend_reach(self.rope, self.tolerance).min(len - s0);
        let steps = if step > 0.0 {
            (reach / step) as usize
        } else {
            0
        };
        for i in 1..=steps {
            let s = s0 + i as f32 * step;
            // `steps` truncates, so the last grid point is short of `reach` and
            // therefore of `len`; the guard is for the float that lands on it.
            if s >= len {
                break;
            }
            lay(&mut self.tip, s);
        }
        lay(&mut self.tip, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(x: f32, y: f32) -> InputSample {
        InputSample {
            pos: Vec2::new(x, y),
            pressure: 1.0,
            tilt: Vec2::ZERO,
            time: 0.0,
        }
    }

    /// Feed a polyline through a tow, collecting every emission. At
    /// [`DEFAULT_TOLERANCE`](crate::path::DEFAULT_TOLERANCE), which is what a
    /// replay declares and the coarsest tolerance any of these ropes meets.
    fn run(rope: f32, pts: &[Vec2]) -> (Tow, Vec<Vec2>) {
        let mut tow = Tow::new(
            rope,
            crate::path::DEFAULT_TOLERANCE,
            InputSample::at(pts[0]),
        );
        let mut out = Vec::new();
        for &p in &pts[1..] {
            tow.to(InputSample::at(p), &mut |s| out.push(s.pos));
        }
        (tow, out)
    }

    #[test]
    fn a_straight_tow_settles_to_a_trail_of_one_rope() {
        let (tow, out) = run(40.0, &[Vec2::ZERO, Vec2::new(400.0, 0.0)]);
        // Far from the start the bend has decayed and the tip trails the
        // target by exactly the rope, dead astern.
        let tip = tow.string().tip;
        assert!(
            (tip - Vec2::new(360.0, 0.0)).length() < 0.1,
            "tip at {tip} after a long straight tow with rope 40"
        );
        // And every emitted sample respects the string: never further than the
        // rope from the target's line, never ahead of it.
        for p in out {
            assert!(p.y.abs() < 1e-3, "the tip left the target's own line: {p}");
            assert!(p.x <= 360.0 + 1e-3, "the tip overtook the string: {p}");
        }
    }

    /// §6.11: the trajectory is a function of the pointer's path, not its
    /// report clock — cutting a run into reports composes.
    #[test]
    fn the_tow_composes_across_report_boundaries() {
        // A path with a genuine bend, so the taut phase is exercised on both
        // sides of it.
        let a = Vec2::ZERO;
        let b = Vec2::new(300.0, 0.0);
        let c = Vec2::new(300.0, 300.0);
        let coarse = run(50.0, &[a, b, c]).0.string().tip;
        // The same path delivered in many short reports.
        let mut fine_pts = vec![a];
        for i in 1..=30 {
            fine_pts.push(a.lerp(b, i as f32 / 30.0));
        }
        for i in 1..=30 {
            fine_pts.push(b.lerp(c, i as f32 / 30.0));
        }
        let fine = run(50.0, &fine_pts).0.string().tip;
        assert!(
            (coarse - fine).length() < 1e-2,
            "report cadence moved the tip: {coarse} vs {fine}"
        );
    }

    #[test]
    fn a_wander_inside_the_rope_parks_the_tip() {
        // A pixel staircase and a hesitation loop, all within the rope.
        let jitter: Vec<Vec2> = (0..40)
            .map(|i| {
                let a = i as f32 * 0.7;
                Vec2::new(10.0 * a.cos(), 10.0 * a.sin())
            })
            .collect();
        let (tow, out) = run(40.0, &jitter);
        assert!(out.is_empty(), "a slack string moved the tip: {out:?}");
        assert_eq!(tow.string().tip, jitter[0], "the tip left its park");
    }

    /// The pen-up parks the tip (§6.11): the mark ends where the rope had towed
    /// it to, one string short of the lift point and dead astern of it. The run
    /// out to the hand that a winch would splice on is exactly what must not be
    /// drawn — nothing steered it, and nothing previewed it.
    #[test]
    fn the_pen_up_leaves_the_tip_where_the_rope_towed_it() {
        let lift = Vec2::new(100.0, 30.0);
        let (tow, out) = run(40.0, &[Vec2::ZERO, lift]);
        let end = *out.last().expect("a run well past the rope tows");
        assert_eq!(tow.string().tip, end, "the last emission is not the tip");
        // Straight tow, so the tip lies on the target's own line, a rope behind.
        let gap = (lift - end).length();
        assert!(
            (gap - 40.0).abs() < 1e-2,
            "tip {end} is {gap} from the lift point, not one rope"
        );
    }

    /// A flick shorter than the rope never brings the string taut, so it never
    /// tows — and the pen-up does not rescue it. The mark is the dab that was
    /// on screen for the whole gesture: a tick smaller than the string is
    /// indistinguishable from the wobble the string exists to eat, and a brush
    /// that hatches wants little smoothing or none.
    #[test]
    fn a_flick_inside_the_rope_leaves_only_the_dab() {
        let (tow, out) = run(
            60.0,
            &[Vec2::ZERO, Vec2::new(20.0, 8.0), Vec2::new(35.0, 14.0)],
        );
        assert!(out.is_empty(), "a flick inside the rope towed: {out:?}");
        assert_eq!(
            tow.string().tip,
            Vec2::ZERO,
            "the pen-up moved a parked tip"
        );
    }

    /// The string is an invariant, not a tendency: at every emission the tip
    /// is within one rope of where the target then was, and it never drifts
    /// off the towing side of a one-sided turn.
    #[test]
    fn the_string_never_stretches() {
        let rope = 30.0;
        // A wide zig-zag driven well past the rope on every leg.
        let pts: Vec<Vec2> = (0..12)
            .map(|i| Vec2::new(i as f32 * 80.0, if i % 2 == 0 { 0.0 } else { 120.0 }))
            .collect();
        let mut tow = Tow::new(
            rope,
            crate::path::DEFAULT_TOLERANCE,
            InputSample::at(pts[0]),
        );
        for pair in pts.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            tow.to(InputSample::at(b), &mut |s| {
                // The target at this emission lies somewhere on [a, b]; the
                // string bound holds against the *nearest* point of the run,
                // which is the weakest claim that is always true mid-run.
                let t = ((s.pos - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
                let near = a.lerp(b, t);
                assert!(
                    (s.pos - near).length() <= rope + 1e-2,
                    "tip {} further than the rope from the run [{a}, {b}]",
                    s.pos
                );
            });
        }
    }

    /// **What one pointer report may cost the fitter** — the property
    /// [`bend_reach`] exists for, swept over the whole of what the frontend can
    /// ask for.
    ///
    /// The spacing is a fraction of the rope, so the count was `1/rope` and a
    /// smoothing knob near zero froze the tab: 2000 emissions for one 4 px report
    /// at amount 0.02 and 8x zoom, each one a full least-squares update. The
    /// frontend's own mapping is reproduced here rather than cited, because what is
    /// being pinned is the composition of the two — a rope this module never sees
    /// alone.
    #[test]
    fn one_report_costs_a_bounded_number_of_emissions() {
        // `rope_in` / `tolerance_in`: both the string and the tolerance are the
        // hand's, carried through the view, which is why the ratio that decides the
        // count survives the zoom and the count does not blow up at either end.
        let rope_in = |amount: f32, zoom: f32| amount * amount * 160.0 / zoom;
        let tolerance_in = |res: f32, zoom: f32| res / zoom;
        let mut worst = 0usize;
        for &amount in &[1.0f32, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.001, 1e-6] {
            for &zoom in &[0.05f32, 1.0, 8.0, 64.0] {
                // A mouse at dpr 1 and a pen at dpr 2 — the two ends of what
                // `input_resolution` reports.
                for &res in &[1.0f32, 0.25] {
                    let rope = rope_in(amount, zoom);
                    if rope <= 0.0 {
                        continue;
                    }
                    let mut tow = Tow::new(rope, tolerance_in(res, zoom), sample(0.0, 0.0));
                    // Reports 4 canvas px apart at this zoom: an unhurried hand at a
                    // few hundred hertz, which is the case a per-report walk costs
                    // seconds on.
                    let step = 4.0 / zoom;
                    for i in 1..=40 {
                        let mut n = 0usize;
                        tow.to(sample(i as f32 * step, 0.0), &mut |_| n += 1);
                        worst = worst.max(n);
                        assert!(
                            n <= 32,
                            "amount {amount}, zoom {zoom}, res {res}: one report emitted {n}"
                        );
                    }
                }
            }
        }
        // Not vacuous: a real rope does spend its grid across the bend.
        assert!(
            worst > 4,
            "no case emitted more than {worst} — the sweep is inert"
        );
    }

    /// A rope the pointer cannot resolve is no rope: the tow costs exactly what
    /// the untowed path costs, one emission a report, rather than falling off a
    /// cliff into it at some threshold.
    #[test]
    fn a_rope_under_the_grain_emits_once_a_report() {
        let mut tow = Tow::new(0.4, 1.0, sample(0.0, 0.0));
        for i in 1..=20 {
            let mut n = 0usize;
            tow.to(sample(i as f32 * 4.0, 0.0), &mut |_| n += 1);
            assert_eq!(n, 1, "report {i} emitted {n}");
        }
        // And the tip is still towed — it lags the pointer by the rope. The knob
        // going quiet is about the *sampling*, not about the string.
        let tip = tow.string().tip;
        assert!(
            (tip - Vec2::new(79.6, 0.0)).length() < 1e-2,
            "tip at {tip}, expected a rope short of 80"
        );
    }

    /// Cutting the grid short leaves the tip where the closed form puts it: the
    /// trajectory is exact however few samples are taken along it (§6.11), so a
    /// run far longer than the bend still finishes in the right place.
    #[test]
    fn cutting_the_grid_short_does_not_move_the_tip() {
        // One report covering 4000 px at a 4 px rope — 4000 grid points under the
        // old rule, one under this one.
        let mut long = Tow::new(4.0, 1.0, sample(0.0, 0.0));
        long.to(sample(4000.0, 0.0), &mut |_| {});
        // The same travel delivered in reports short enough that the grid covers
        // every one of them.
        let mut fine = Tow::new(4.0, 1.0, sample(0.0, 0.0));
        for i in 1..=1000 {
            fine.to(sample(i as f32 * 4.0, 0.0), &mut |_| {});
        }
        let (a, b) = (long.string().tip, fine.string().tip);
        assert!((a - b).length() < 1e-2, "coarse tip {a} against fine {b}");
    }

    /// The wiring the session repeats (§6.11): tow into fitter, nothing at
    /// pen-up. A towed, fitted stroke starts at the press and ends at the tip —
    /// a rope short of the lift, with the towed samples in between.
    #[test]
    fn a_towed_stroke_fits_from_press_to_the_tip() {
        use crate::path::PathFitter;
        let mut fitter = PathFitter::new();
        let first = sample(0.0, 0.0);
        let mut tow = Tow::new(25.0, crate::path::DEFAULT_TOLERANCE, first);
        fitter.push(first);
        for i in 1..=60 {
            let s = sample(i as f32 * 4.0, (i as f32 * 0.1).sin() * 3.0);
            tow.to(s, &mut |t| fitter.push(t));
        }
        fitter.finish();
        let path = fitter.path();
        assert!(path.len() >= 2);
        assert_eq!(path[0].pos, Vec2::ZERO);
        let end = path.last().unwrap().pos;
        assert!(
            (end - tow.string().tip).length() < 1e-3,
            "fitted end {end} is not the towed tip {}",
            tow.string().tip
        );
        let lift = Vec2::new(240.0, (6.0f32).sin() * 3.0);
        assert!(
            (end - lift).length() > 1.0,
            "the fit ran on to the lift point {lift}"
        );
    }
}
