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
use crate::geom::Vec2;

/// Emission spacing while the tip is being dragged, as a fraction of the rope,
/// measured along the **target's** travel. The tractrix bend mostly completes
/// within a couple of rope-lengths of travel, so a quarter-rope grid gives the
/// fitter several samples across it; the fit smooths through the rest.
const EMIT_SPACING: f32 = 0.25;

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
    tip: Vec2,
    /// The last raw report — the target the string runs to. Kept whole (not
    /// just the position) because a run's emissions interpolate the pen
    /// channels across it.
    target: InputSample,
}

impl Tow {
    /// A tow of `rope` canvas px, tip parked on the first report.
    ///
    /// Callers gate construction on `rope > 0` — a rope of zero is no tow, and
    /// the session simply feeds the fitter directly (bit-identical to the
    /// pre-§6.11 path).
    pub fn new(rope: f32, first: InputSample) -> Self {
        Self {
            rope,
            tip: first.pos,
            target: first,
        }
    }

    /// Where the mark is being laid.
    pub fn tip(&self) -> Vec2 {
        self.tip
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
        let perp = Vec2::new(-u.y, u.x);
        let side = w.dot(perp);
        // The side the tip trails on is preserved: θ only decays, never
        // crosses zero.
        let n = if side >= 0.0 { perp } else { -perp };
        let t0 = side.abs() / (1.0 + cos0);
        let step = self.rope * EMIT_SPACING;
        let mut s = s0;
        loop {
            s = (s + step).min(len);
            let t = t0 * (-(s - s0) / self.rope).exp();
            let m = 1.0 + t * t;
            let w = u * ((1.0 - t * t) / m) + n * (2.0 * t / m);
            self.tip = prev.pos + u * s - w * self.rope;
            let f = s / len;
            emit(InputSample {
                pos: self.tip,
                pressure: prev.pressure + (next.pressure - prev.pressure) * f,
                tilt: prev.tilt.lerp(next.tilt, f),
                time: prev.time + (next.time - prev.time) * f as f64,
            });
            if s >= len {
                return;
            }
        }
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

    /// Feed a polyline through a tow, collecting every emission.
    fn run(rope: f32, pts: &[Vec2]) -> (Tow, Vec<Vec2>) {
        let mut tow = Tow::new(rope, InputSample::at(pts[0]));
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
        let tip = tow.tip();
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
        let coarse = run(50.0, &[a, b, c]).0.tip();
        // The same path delivered in many short reports.
        let mut fine_pts = vec![a];
        for i in 1..=30 {
            fine_pts.push(a.lerp(b, i as f32 / 30.0));
        }
        for i in 1..=30 {
            fine_pts.push(b.lerp(c, i as f32 / 30.0));
        }
        let fine = run(50.0, &fine_pts).0.tip();
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
        assert_eq!(tow.tip(), jitter[0], "the tip left its park");
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
        assert_eq!(tow.tip(), end, "the last emission is not the tip");
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
        assert_eq!(tow.tip(), Vec2::ZERO, "the pen-up moved a parked tip");
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
        let mut tow = Tow::new(rope, InputSample::at(pts[0]));
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

    /// The wiring the session repeats (§6.11): tow into fitter, nothing at
    /// pen-up. A towed, fitted stroke starts at the press and ends at the tip —
    /// a rope short of the lift, with the towed samples in between.
    #[test]
    fn a_towed_stroke_fits_from_press_to_the_tip() {
        use crate::path::PathFitter;
        let mut fitter = PathFitter::new();
        let first = sample(0.0, 0.0);
        let mut tow = Tow::new(25.0, first);
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
            (end - tow.tip()).length() < 1e-3,
            "fitted end {end} is not the towed tip {}",
            tow.tip()
        );
        let lift = Vec2::new(240.0, (6.0f32).sin() * 3.0);
        assert!(
            (end - lift).length() > 1.0,
            "the fit ran on to the lift point {lift}"
        );
    }
}
