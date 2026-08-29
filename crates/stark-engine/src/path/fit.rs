//! **Fitting**: input samples → control points (§6.2).
//!
//! A streaming least-squares fit — see the module note in [`super`] for where this
//! sits between the three.
//!
//! `pub(super)`, fields and all, and that is what the split cost here: the pipeline
//! tests live in `super` (see the module note there) and several of them assert on the
//! *window* rather than on the curve, which is the only way to say that thinning the
//! window did not cost the fit its accuracy. Nothing outside `path` can name it.

use super::flatten::{frozen_spans_for, span_count};
use crate::command::InputSample;
use crate::spline::{CubicBSpline, Observations};
use nalgebra::{Const, Dyn, OMatrix};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;

/// Control points solved for at the live end of the stroke. Everything behind them
/// is frozen; the pinned endpoint sits inside the window on top of these.
///
/// This is *the* accuracy/stability trade, and small is the point. A short window
/// means the settled stroke cannot move under the pointer — which is what stops a
/// live stroke wobbling along its whole length — and it caps the solve at a handful
/// of unknowns however long the stroke gets. Too short and the curve cannot round a
/// corner before the corner is committed.
pub(super) const FREE_CONTROL_POINTS: usize = 3;

/// What a control point has to earn, in **mean** squared error over the samples in
/// the window, before the fit will take one on — measured in units of the caller's
/// input tolerance, so the price in canvas px² is `KNOT_COST × tolerance²`.
///
/// Every sample is fitted both ways — as the polygon stands, and with one more
/// control point — and the larger fit is adopted only if it buys at least this much.
/// Adopting it is also what *freezes* one, since the window is a fixed size, so this
/// single number decides both how detailed the curve is and how promptly the stroke
/// settles behind the pointer.
/// Measured on the six recorded strokes at `DEFAULT_TOLERANCE`, worst *live* error
/// and control-point count, once the parameterization was corrected (see
/// `arc_profile`):
///
/// | price | C        | hairpin  | loop     | spiral   | big-C    | fast     |
/// |-------|----------|----------|----------|----------|----------|----------|
/// | 0.03  | 0.9px 37 | 1.4px 27 | 1.4px 34 | 2.4px 91 | 1.4px 68 | 0.7px 28 |
/// | 0.06  | 1.2px 25 | 1.5px 23 | 2.3px 25 | 2.2px 56 | 2.2px 52 | 0.8px 18 |
/// | 0.12  | 2.5px 19 | 1.5px 18 | 2.8px 18 | 4.2px 23 | 3.0px 18 | 1.2px 16 |
///
/// The floor is set by the input's own quantization rather than by taste: priced
/// below the jitter, the fit buys control points to *trace* a pixel staircase
/// instead of smoothing through it. That is why the price is denominated in the
/// tolerance and why it scales as its **square** — input landing on a grid of
/// `tolerance` leaves a residual whose *mean square* goes as `tolerance²`, so a
/// price fixed in canvas px² would sit above the jitter at one zoom level and
/// below it at another.
pub const KNOT_COST: f32 = 0.06;

/// Distance after which the polygon gains a control point regardless of error, in
/// units of the caller's input tolerance (`KNOT_SPACING × tolerance` canvas px).
///
/// Not about accuracy: a dead-straight stroke is fitted perfectly by a handful of
/// control points however long it runs, so on error alone it would never gain one,
/// never freeze one, and never let the renderer retire any of it.
///
/// Scaled by the tolerance for the same reason the price is. What it bounds is how
/// long the free window may go without advancing, and the window's job is to average
/// over a certain number of *input reports* — canvas px say nothing about how many
/// of those a stretch of stroke holds.
pub const KNOT_SPACING: f32 = 64.0;

/// The input tolerance assumed when a caller does not supply one: one canvas px,
/// which is a mouse over a 1:1 view — what the two prices above were measured
/// against.
pub const DEFAULT_TOLERANCE: f32 = 1.0;

/// Bounds a supplied tolerance is held to.
///
/// Zero (or negative) would make a control point free and every sample worth one.
/// The ceiling is 64 canvas px per input px — past the most zoomed-out view
/// [`ViewTransform`](crate::view::ViewTransform) allows — and is what keeps the
/// window advancing, and so the per-sample work bounded, rather than letting a
/// stroke never grow a control point at all.
/// `pub` so the benchmark can name the bound rather than restate it: the tolerance
/// sweep in `benches/path.rs` exists to hold the fit's cost flat *to the top of this
/// range*, and a hard-coded 64.0 there would be a second copy of the number that
/// decides what the range is.
pub const MIN_TOLERANCE: f32 = 1.0 / 64.0;

/// See [`MIN_TOLERANCE`].
pub const MAX_TOLERANCE: f32 = 64.0;

/// A caller's declared input resolution held to the bounds above, with
/// [`DEFAULT_TOLERANCE`] for a non-finite one — **the** statement of what a
/// tolerance may be.
///
/// One function rather than a clamp at each door, because the tolerance is read by two
/// stages of the input path and they have to agree about it: the fit prices its two
/// thresholds in it ([`PathFitter::with_tolerance`]), and the towed tip measures
/// against it how far its own bend is still worth sampling (§6.11). Two copies of
/// the bounds would let a rope be measured against a tolerance the fitter had already
/// clamped away.
pub fn clamp_tolerance(tolerance: f32) -> f32 {
    if tolerance.is_finite() {
        tolerance.clamp(MIN_TOLERANCE, MAX_TOLERANCE)
    } else {
        DEFAULT_TOLERANCE
    }
}

/// Curvature penalty on the control polygon, as a fraction of the data's own pull
/// (see [`CubicBSpline::fit_channels`]).
///
/// Least squares charges the curve for being far from a *point*, never for where it
/// goes when no point is near — so a stretch the data does not constrain is free to
/// wander. With the correspondence declared rather than searched this is a much
/// milder problem than it was, but a control point at the very end of the window
/// still has little holding it, and this is what settles it onto its neighbours'
/// continuation.
const SMOOTHING: f32 = 0.05;

/// Per-point channels carried alongside the geometry: pressure, tilt x/y, time.
pub(super) const CHANNELS: usize = 4;

/// Which of [`CHANNELS`] is the clock. The odd one out: the other three are pen state,
/// which the tip's shape follows, while this one is only a stamp on the report — which
/// is why [`PathFitter::solve`] treats the stroke's last one differently.
const TIME_CHANNEL: usize = 3;

type ChannelCtrl = OMatrix<f32, Dyn, Const<CHANNELS>>;
type GeomCtrl = OMatrix<f32, Dyn, Const<2>>;

/// Streams [`InputSample`]s into [`ControlPoint`]s, append-only.
///
/// The fit is a **least-squares clamped cubic B-spline** solved over a *fixed-size
/// window* at the live end: exactly `FREE_CONTROL_POINTS` of them are solved for,
/// everything behind is frozen, and the polygon grows only when the data proves it
/// needs to. There is no assignment search — a sample's place on the curve is
/// declared from how far along the stroke it sits — so the solve is one small linear
/// system and cannot land in a bad local optimum.
///
/// **Growth is the same decision as freezing**, which is what makes the two agree.
/// Each sample is fitted twice: once with the polygon as it stands, and once with
/// one more control point. If the extra one earns its keep ([`KNOT_COST`]) it is
/// adopted — and because the window is a fixed size, adopting it *pushes one out the
/// back*, freezing it. So a control point is committed at exactly the moment the
/// stroke has moved on far enough to justify a new one behind it, rather than on a
/// lag guessed in advance.
///
/// Two properties follow, and both were hard to get any other way:
///
/// * **The stroke stops wobbling.** Only the last few control points can move at
///   all, so the settled part of a live stroke is pixel-stable rather than
///   re-solving under the pointer on every report.
/// * **The system being solved is a constant size.** It is
///   `FREE_CONTROL_POINTS × 2` unknowns however long the stroke is, and only the
///   samples that can reach those rows take part. `spline::solve_window` assembles the
///   normal equations over that window too, so the *arithmetic* per report does not
///   grow with the stroke.
///
///   The **work** per report is not quite constant, and saying so is worth more than
///   the tidier claim that used to stand here. Each report solves two candidate
///   polygons and scores both, and a candidate is a whole `m`-row matrix: growing one
///   ([`grow_rows`]) and measuring its arc profile ([`arc_profile`]) are `O(m)`
///   copies around an `O(1)` solve. Two of those copies per solve have since been
///   removed — the curve is read through a borrow and the fit writes back into the
///   candidate rather than returning a fresh one ([`spline::SplineIndex`]), worth
///   3–21% on `benches/path.rs` — and the four that remain are the reason a long
///   stroke's last report still costs more than its first.
///
///   [`spline::SplineIndex`]: crate::spline::SplineIndex
///
/// Both ends are pinned to the samples they belong to — the clamped end condition
/// makes the first and last control points the curve's endpoints, and least squares
/// does not otherwise hold them there. They are pinned as *constraints* of the solve
/// (held rows), not written over its result, so the rest solves around them.
///
/// Both thresholds in that growth rule are quoted in the caller's **input tolerance**
/// rather than in canvas px (see [`Self::with_tolerance`]) — what counts as jitter to
/// smooth through, as against detail to keep, is a fact about the device and the zoom
/// level, and only the caller knows it. Flattening is untouched by this: its budget
/// is an error against the *curve*, in the canvas px it will be drawn in
/// ([`FlattenTolerance`](super::FlattenTolerance)).
pub struct PathFitter {
    /// Every accepted report, with the distance along the stroke that parameterizes
    /// it. Kept whole — a few hundred is nothing — while the solve only ever looks
    /// at the tail (see `first_live`).
    pts: Vec<Accepted>,
    /// The **run-up** (§6.2): reports from before the stroke's first sample,
    /// awaiting that sample — whose pressure and clock they are converted
    /// against ([`Self::adopt_runup`]). Once adopted they are ordinary rows of
    /// `pts` *ahead of the press*: the curve genuinely extends back through
    /// them, and `start_arc` remembers where the stroke itself begins.
    runup: Vec<InputSample>,
    /// First sample that can still reach a control point being solved for.
    first_live: usize,
    /// Control points: geometry, and the pen channels riding the same knots.
    geom: GeomCtrl,
    attr: ChannelCtrl,
    /// Arc length through the accepted samples (canvas px).
    arc: f32,
    /// Arc profile of the frozen spans, reused across updates (see `arc_profile`).
    settled_profile: Vec<f32>,
    /// Arc at which the polygon last gained a control point.
    grown_at: f32,
    smoothing: f32,
    /// The input's own positional resolution, in canvas px (see [`Self::with_tolerance`]).
    tolerance: f32,
    /// Absolute time of the first sample; channel times are relative to it.
    t0: f64,
    /// Arc at the stroke's own first sample — the press, which the run-up
    /// precedes ([`Self::seed_runup`]). `None` until it arrives; 0 for a stroke
    /// with no run-up. Fixed at the press and never revised: it is measured
    /// along the accepted reports, which are append-only.
    start_arc: Option<f32>,
    /// The marker `start_arc` resolves to on the **adopted** fit's curve
    /// ([`Self::start_on`]) — what the committed record's
    /// [`start`](stark_model::document::StrokeRecord::start) is once the stroke
    /// is finished. A mid-stroke preview reads the as-finished solve's own
    /// instead ([`Self::as_finished`]).
    start_param: f32,
    finished: bool,
}

/// One accepted report: where it is, what the pen said, and how far along it sits.
///
/// `pub(super)`, fields and all, and that is what the split cost here: the pipeline
/// tests live in [`super`] (see the module note there) and several of them assert on
/// the *window* rather than on the curve, which is the only way to say that thinning
/// the window did not cost the fit its accuracy. Nothing outside `path` can name it.
#[derive(Copy, Clone, Debug)]
pub(super) struct Accepted {
    pub(super) pos: Vec2,
    pub(super) channels: [f32; CHANNELS],
    pub(super) arc: f32,
}

impl std::fmt::Debug for PathFitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathFitter")
            .field("samples", &self.pts.len())
            .field("control_points", &self.geom.nrows())
            .field("frozen_spans", &self.frozen_spans())
            .field("tolerance", &self.tolerance)
            .field("arc", &self.arc)
            .field("finished", &self.finished)
            .finish()
    }
}

impl Default for PathFitter {
    fn default() -> Self {
        Self::new()
    }
}

impl PathFitter {
    pub fn new() -> Self {
        Self::with_tolerance(DEFAULT_TOLERANCE)
    }

    /// A fitter for input whose positional resolution is `tolerance` **canvas px**.
    ///
    /// This is the one thing about the input the fit cannot work out for itself: how
    /// far apart two reports have to be before the difference means anything. Canvas
    /// px are the wrong unit for it — the same hand movement covers 64× as many of
    /// them zoomed in as zoomed out, and a pen digitizer resolves far finer than a
    /// mouse does at either — so the caller that owns the view transform and knows
    /// what device is reporting states it, and the fit's two prices ([`KNOT_COST`],
    /// [`KNOT_SPACING`]) are denominated in it.
    ///
    /// Clamped to `[1/64, 64]`; a non-finite tolerance falls back to
    /// [`DEFAULT_TOLERANCE`].
    pub fn with_tolerance(tolerance: f32) -> Self {
        Self {
            pts: Vec::new(),
            runup: Vec::new(),
            first_live: 0,
            geom: GeomCtrl::zeros_generic(Dyn(0), Const::<2>),
            attr: ChannelCtrl::zeros_generic(Dyn(0), Const::<CHANNELS>),
            arc: 0.0,
            settled_profile: Vec::new(),
            grown_at: 0.0,
            smoothing: SMOOTHING,
            tolerance: clamp_tolerance(tolerance),
            t0: 0.0,
            start_arc: None,
            start_param: 0.0,
            finished: false,
        }
    }

    /// Seed the stroke's **run-up**: reports from before its first sample — the
    /// hover trail the engine was already watching (§18.1.10) — adopted as real
    /// leading samples, so the fitted curve *extends back through them* and the
    /// entry's direction and curvature are measured from motion the fit could
    /// otherwise only guess at from its first, tolerance-quantized steps (§6.2).
    ///
    /// **The curve extends; the stroke does not.** Where on the extended curve
    /// the stroke itself begins is recorded — the arc of the first pushed
    /// sample, resolved to a curve parameter by the same map the solve places
    /// samples with ([`Self::start_on`]) — and it lands in the record as
    /// [`StrokeRecord::start`](stark_model::document::StrokeRecord::start),
    /// where the one flattening funnel begins the deposit. The press is then an
    /// *interior* sample: the curve is pinned to the run-up's first report and
    /// to the live tip, and the entry is smoothed **through** the press exactly
    /// as every later report is — which is the point, since a start pinned to
    /// one tolerance-quantized report was the last unsmoothed place on the stroke.
    /// A fitter seeded with nothing is bit-identical to one that never had this
    /// called, and its marker is 0.
    ///
    /// Call before the first [`push`](Self::push); afterwards it is ignored, as
    /// are non-finite reports. The reports' pressures are replaced with the
    /// first pushed sample's at adoption: pressure begins at the press — a
    /// hovering pen reports none — and geometry is what the run-up is evidence
    /// of. Tilt and time are kept; both are continuous through a press.
    pub fn seed_runup(&mut self, samples: &[InputSample]) {
        if !self.pts.is_empty() || self.finished {
            return;
        }
        self.runup = samples
            .iter()
            .copied()
            .filter(|s| s.is_admissible())
            .collect();
    }

    /// Adopt the pending run-up against the stroke's first sample: its reports
    /// walk in as ordinary accepted rows *ahead of the press*, times re-based
    /// on its clock and pressures replaced with its own — see
    /// [`seed_runup`](Self::seed_runup).
    fn adopt_runup(&mut self, first: &InputSample) {
        let pending = std::mem::take(&mut self.runup);
        for s in pending {
            if let Some(prev) = self.pts.last() {
                let step = (s.pos - prev.pos).length();
                // The same zero-step gate `push` applies, for the same reason.
                if step < 1e-6 {
                    continue;
                }
                self.arc += step;
            }
            self.pts.push(Accepted {
                pos: s.pos,
                channels: [first.pressure, s.tilt.x, s.tilt.y, self.rel_time(s.time)],
                arc: self.arc,
            });
        }
    }

    /// Feed one pointer report. Ignored once the stroke is [`finish`](Self::finish)ed,
    /// and ignored if it is not [admissible](InputSample::is_admissible) — the same
    /// "this report carries nothing" answer the zero-length step below gives, for a
    /// report that carries nothing usable.
    ///
    /// **Dropping it is the only total answer.** A NaN position spreads into `arc`,
    /// out of `arc` into every sample's curve parameter, and from there into the
    /// normal equations, which are then singular at every ridge — a state
    /// [`spline`](crate::spline)'s solve reports by panicking, because for admissible
    /// input it cannot arise. Repairing the sample instead would mean inventing a
    /// position the hand never visited; refusing it means the stroke is exactly the
    /// stroke the admissible reports describe.
    ///
    /// [`InputSample::is_admissible`]: crate::command::InputSample::is_admissible
    pub fn push(&mut self, s: InputSample) {
        if self.finished || !s.is_admissible() {
            return;
        }
        if self.pts.is_empty() {
            // The stroke's own first report: the clock every channel time is
            // relative to, and the sample the pending run-up is converted
            // against — after which `pts` holds the run-up and this report is
            // appended behind it like any later one.
            self.t0 = s.time;
            self.adopt_runup(&s);
        }
        // Whether this is the stroke's own first report — where the marker
        // lands, even for a report the spacing gate drops.
        let pressed = self.start_arc.is_none();
        if let Some(prev) = self.pts.last() {
            let step = (s.pos - prev.pos).length();
            // A report that did not move carries no geometry, and a run of them
            // would put several samples at one parameter. Its attributes are no
            // loss: they apply to a zero-length piece of path. A press that
            // coincides with the run-up's newest report still marks the start —
            // the marker is a place on the curve, and the place exists.
            if step < 1e-6 {
                if pressed {
                    self.start_arc = Some(self.arc);
                }
                return;
            }
            self.arc += step;
        }
        self.pts.push(Accepted {
            pos: s.pos,
            channels: self.channels(s),
            arc: self.arc,
        });
        if pressed {
            self.start_arc = Some(self.arc);
        }
        if self.pts.len() < 2 {
            return;
        }
        if self.geom.nrows() < 2 {
            self.grow_to(2);
        }

        // Fit as we are, and fit with one more control point. Keeping the better of
        // the two *is* the growth rule, and adopting the larger one is what freezes
        // the control point it pushes out of the window.
        let m = self.geom.nrows();
        let as_is = self.solve(m);
        let grown = self.solve(m + 1);
        let scored_from = as_is.lo;
        let err_as_is = self.mean_error(&as_is, scored_from);
        let err_grown = self.mean_error(&grown, scored_from);
        // The arc-length term is not about accuracy: a dead-straight stroke is fitted
        // perfectly by a handful of control points forever, so nothing would ever
        // freeze and the renderer could never retire any of it.
        //
        // Both prices are quoted in the input's own units rather than in canvas px —
        // the error one squared, since it is compared against a mean square. See
        // `KNOT_COST`.
        let price = KNOT_COST * self.tolerance * self.tolerance;
        let spacing = KNOT_SPACING * self.tolerance;
        let earns_it = err_as_is - err_grown > price || self.arc - self.grown_at > spacing;
        if earns_it {
            self.grown_at = self.arc;
            self.adopt(grown);
        } else {
            self.adopt(as_is);
        }
    }

    /// Every accepted report's position **from the stroke's own first sample
    /// on**, in order — the **raw trace**, which the drawing assist recognizes
    /// a shape from (§6.9). The run-up is left out: it is evidence about the
    /// entry, not part of the gesture, and a circle drawn after a watched
    /// approach has to read as the circle rather than as the approach with a
    /// circle appended.
    ///
    /// Deliberately the reports rather than [`path`](Self::path): the fit is a curve
    /// pulled *towards* its control points, so those sit off the stroke by design and
    /// asking whether they lie on a circle is asking the wrong question. Nothing
    /// downstream stores these — they are already held here only because the window
    /// solve reads its own tail, and a few hundred `Vec2`s is nothing.
    pub fn trace(&self) -> Vec<Vec2> {
        let from = self.start_arc.unwrap_or(0.0);
        self.pts
            .iter()
            .filter(|s| s.arc >= from)
            .map(|s| s.pos)
            .collect()
    }

    /// End the stroke: commit the whole polygon, so every control point is final.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if self.geom.nrows() >= 2 {
            // One last solve with the window still free — those control points have
            // had no chance to settle against data that will never arrive.
            let last = self.solve(self.geom.nrows());
            self.adopt(last);
        }
    }

    /// The path fitted so far.
    ///
    /// The trailing `FREE_CONTROL_POINTS` may still move; everything before them
    /// is final. After [`finish`](Self::finish) the whole path is.
    pub fn path(&self) -> Vec<ControlPoint> {
        if self.geom.nrows() == 0 {
            // A click: one report, so one point and no curve.
            return self
                .pts
                .first()
                .map(|s| {
                    ControlPoint::clamped(
                        s.pos,
                        s.channels[0],
                        Vec2::new(s.channels[1], s.channels[2]),
                        s.channels[3],
                    )
                })
                .into_iter()
                .collect();
        }
        control_points(&self.geom, &self.attr)
    }

    /// The path and the start marker exactly as [`finish`](Self::finish) would
    /// leave them, as a pure query: the one last solve with the window still
    /// free, *not* adopted — and one solve for both halves, so the marker names
    /// a place on the very curve beside it.
    ///
    /// This is what a live preview must render (§1.3): the stroke that would be
    /// committed if the pen lifted now. The free window's control points sit
    /// elsewhere under a mid-stroke solve — they are still braced for data that a
    /// finished stroke never receives — and that gap is a real change of geometry at
    /// pen-up, sub-pixel but fatal to `preview == committed` wherever a discontinuous
    /// lookup (the tooth's nearest-sampled substrate, §6.4) turns position into a step.
    /// Rendering the as-finished path instead makes `End` a no-op on the record by
    /// construction: nothing is pushed between the last preview and the commit, so
    /// [`finish`](Self::finish) adopts this very solve, bit for bit. The marker
    /// rides the same argument: it is a function of the solve's own arc profile
    /// ([`Self::start_on`]), so preview and commit place it identically.
    pub fn as_finished(&self) -> (Vec<ControlPoint>, f32) {
        if self.finished || self.geom.nrows() < 2 {
            return (self.path(), self.start_param);
        }
        let f = self.solve(self.geom.nrows());
        let start = self.start_on(&f.profile);
        (control_points(&f.geom, &f.attr), start)
    }

    /// [`as_finished`](Self::as_finished)'s path alone, for callers with no use
    /// for the marker.
    pub fn path_as_finished(&self) -> Vec<ControlPoint> {
        self.as_finished().0
    }

    /// How many leading spans of [`path`](Self::path) are settled — their geometry,
    /// and so their flattening, can never change however the stroke continues.
    ///
    /// A span of a clamped cubic B-spline reads at most two control points past its
    /// own index (see [`span_count`]), so span `k` is final once control points
    /// `0..=k+1` are frozen: `f` frozen control points settle `f - 1` spans. This is
    /// the hook for incremental repaint (§6.2) — render
    /// `0..frozen_spans()` once and re-render only the short tail after it.
    pub fn frozen_spans(&self) -> usize {
        if self.finished {
            span_count(self.geom.nrows())
        } else {
            frozen_spans_for(self.frozen(), self.geom.nrows())
        }
    }

    /// How many leading control points of [`path`](Self::path) are final — the
    /// public face of [`frozen`](Self::frozen).
    ///
    /// This is what lets a live gesture be published incrementally
    /// (§17.5): a frozen control point never moves again, so a peer that
    /// has been told about it never has to be told again, and the wire cost of a
    /// stroke follows its tail rather than its length.
    pub fn frozen_points(&self) -> usize {
        let n = self.path_len();
        if self.finished {
            n
        } else {
            self.frozen().min(n)
        }
    }

    /// How many control points [`path`](Self::path) would return, without building
    /// it.
    fn path_len(&self) -> usize {
        if self.geom.nrows() == 0 {
            usize::from(!self.pts.is_empty())
        } else {
            self.geom.nrows()
        }
    }

    /// Control points held at their values: everything but the window at the live
    /// end, and the pinned endpoint inside it.
    fn frozen(&self) -> usize {
        self.geom
            .nrows()
            .saturating_sub(FREE_CONTROL_POINTS + 1)
            .max(usize::from(self.geom.nrows() > 0))
    }

    /// Solve for the free window at a polygon of `m` control points, and report the
    /// squared error it achieves.
    ///
    /// Both ends are pinned first — as held rows, so the solve places the rest of the
    /// window *around* them. Samples map onto the curve's domain by distance
    /// travelled, which puts the first and last exactly on the two ends.
    fn solve(&self, m: usize) -> Fit {
        let m = m.max(2);
        let mut geom = grow_rows(&self.geom, m, |j, d| {
            let p = self.at_fraction(j, m).pos;
            if d == 0 { p.x } else { p.y }
        });
        let mut attr = grow_rows(&self.attr, m, |j, d| self.at_fraction(j, m).channels[d]);
        let first = self.pts[0];
        let last = *self.pts.last().expect("a sample");
        set_row(&mut geom, 0, [first.pos.x, first.pos.y]);
        set_row(&mut geom, m - 1, [last.pos.x, last.pos.y]);
        set_row(&mut attr, 0, first.channels);
        // **The attribute end is held at its neighbour, not pinned to the last report.**
        //
        // The geometry's endpoint has to be the last report — the mark must end where
        // the hand did, and the eye sees that directly. The channels have no such claim
        // on it: nobody can see where a pressure "ends", only the width it produces over
        // the last stretch of stroke. And the last control point is the least-constrained
        // row in the whole polygon, supported on the final span alone, so whatever sits
        // in the last sliver of the domain decides it outright — which is the pen coming
        // off the tablet, the least trustworthy report on the stroke (see
        // [`arc_weights`], which lightens that report's vote everywhere but here, where
        // it is the only vote there is).
        //
        // So the attribute curve leaves the stroke flat: the end continues its
        // neighbour rather than diving for a pressure the hand reported while no longer
        // painting. The neighbour is read from the prior, so it lags the solve by one
        // report and catches up on the next — including at [`Self::finish`], whose last
        // solve is the one [`Self::path_as_finished`] mirrors, so preview and commit see
        // the same lag and agree to the bit (§1.3).
        let held: [f32; CHANNELS] = std::array::from_fn(|d| attr[(m - 2, d)]);
        set_row(&mut attr, m - 1, held);
        // …except the clock, which is not a pen attribute at all. `time` is what the
        // report was stamped with, and the release genuinely happened then; carrying the
        // neighbour's instead would shorten every stroke's recorded duration by a span
        // and quietly skew the timelapse (§8).
        attr[(m - 1, TIME_CHANNEL)] = last.channels[TIME_CHANNEL];

        // The curve as the candidate polygon currently stands, borrowed rather than
        // copied — it is read to measure arc length and then dropped, and the fits
        // below write back into `geom`/`attr` themselves ([`SplineIndex`]).
        let index = {
            let spline: CubicBSpline<'_, 2> =
                CubicBSpline::new(&geom).expect("at least two control points");
            let spans = spline.num_spans() as f32;
            let profile = arc_profile(&spline, &self.settled_profile);
            (spline.index(), spans, profile)
        };
        let (index, spans, profile) = index;
        let total = self.arc.max(1e-6);
        let param = |a: f32| param_at(&profile, spans, a / total);

        // A cubic B-spline's basis is local, so a sample sitting under the frozen
        // prefix cannot influence any row still being solved.
        let frozen = m.saturating_sub(FREE_CONTROL_POINTS + 1).max(1);
        let cutoff = frozen as f32 - 2.0;
        let mut lo = self.first_live.min(self.pts.len() - 1);
        while lo + 1 < self.pts.len() && param(self.pts[lo].arc) < cutoff {
            lo += 1;
        }
        // …and of the reports that *do* reach a free row, at most a bounded number are
        // minimized over — see [`window_indices`], which is the whole of why this
        // costs what it costs rather than what the digitizer charges for it.
        let idx = window_indices(&self.pts, lo);
        let pos: Vec<[f32; 2]> = idx.iter().map(|&i| self.pts[i].pos.to_array()).collect();

        // Distance along the stroke is only a *first guess* at where a sample sits on
        // the curve, because a clamped B-spline is not parameterized by arc: the
        // triple knots at each end squash the first and last spans into a fraction of
        // the leg they cover. Fitted against the raw guess even a dead-straight
        // stroke reads as several px of error, and the growth rule then buys control
        // points to explain it away — 399 of them for a straight line.
        //
        // So the guess is *corrected*: solve, project each sample back onto the curve
        // it just produced, solve again. The projection is a local search around the
        // sample's current parameter, and it is clamped to keep the sequence
        // non-decreasing, so a sample can slide a little along the curve but can
        // never overtake its neighbours — the reordering that makes a searched
        // correspondence dangerous is ruled out by construction.
        let ts: Vec<f32> = idx.iter().map(|&i| param(self.pts[i].arc)).collect();
        // What each report stands for, so the solve minimizes over the *stroke* rather
        // than over the reporting clock ([`arc_weights`]).
        let qs = arc_weights(&self.pts, &idx);
        let vals: Vec<[f32; CHANNELS]> = idx.iter().map(|&i| self.pts[i].channels).collect();
        // Solved **into** the candidate polygon rather than into a fresh one: the
        // prior and the result are the same buffer, which is sound because the solve
        // reads every row it uses as a prior before it writes any
        // ([`SplineIndex::fit_into`]). Returning an owned matrix instead meant a copy
        // of the whole polygon per fit, four fits per pointer report.
        index.fit_into(
            Observations {
                ts: &ts,
                values: &pos,
                weights: &qs,
            },
            &mut geom,
            frozen,
            1,
            self.smoothing,
        );

        // The pen channels ride the same knots at the same parameters, so they are
        // the same solve with a different payload — unsmoothed, since a pressure ramp
        // is not a shape and has no curvature to penalize. Its end is held (the `1`)
        // for the reason set out where that row is written, above.
        index.fit_into(
            Observations {
                ts: &ts,
                values: &vals,
                weights: &qs,
            },
            &mut attr,
            frozen,
            1,
            0.0,
        );
        Fit {
            geom,
            attr,
            lo,
            profile,
        }
    }

    /// Mean **arc-weighted** squared distance from the samples at and after `lo` to
    /// `fit`'s curve.
    ///
    /// Both candidates must be scored over the **same** samples. Each solve drops
    /// the ones its own frozen prefix has swallowed, and the larger polygon freezes
    /// one more — so scoring each on its own slice compares a sum over fewer points
    /// against a sum over more, which the larger one wins every time regardless of
    /// whether it fits better. That made a dead-straight stroke take a control point
    /// per sample. Per-sample rather than total for the same reason in miniature: a
    /// total grows with the window, so a fixed price would mean something different
    /// at every length.
    fn mean_error(&self, fit: &Fit, lo: usize) -> f32 {
        // Borrowed, not copied: scoring a candidate reads its control points and
        // never moves them, so the copy this used to make was a whole polygon per
        // candidate per report to answer one number.
        let spline = CubicBSpline::new(&fit.geom).expect("at least two control points");
        let spans = spline.num_spans() as f32;
        let total = self.arc.max(1e-6);
        let lo = lo.min(self.pts.len() - 1);
        // **The same reports the solve minimized over**, decimated by the same rule.
        // This is the third thing the two have to agree about, beside where the samples
        // sit and what they weigh: scoring the full window while the solve had a
        // bounded one would charge the growth rule for error at reports the solve was
        // never shown ([`window_indices`]).
        let idx = window_indices(&self.pts, lo);
        if idx.is_empty() {
            return 0.0;
        }
        // The same *rule* as the solve's parameters, off a later curve — and the gap is
        // real, so it is written down rather than claimed away. `solve` builds its
        // profile from the polygon as it stands *before* `fit_into` writes back
        // (`index`, above), where this builds one from the spline that came out. Both
        // are `arc_profile` over the same settled prefix, so they agree about what a
        // parameter means and disagree only about which curve it is measured on — a
        // difference of one solve's movement, which is small precisely where the growth
        // rule is deciding not to fire.
        //
        // What the two must not do is use different *maps*: then the growth rule reads
        // one quantity while the solve improves another, and it stops firing where the
        // fit is actually poor — measured at 4-15px on recorded strokes against
        // 0.6-1.6px when they agree. Consistency matters more than accuracy in either.
        //
        // Taking `fit.profile` instead would close the gap outright and drop two of the
        // four curve walks a report costs. It is not done here because `KNOT_COST` was
        // tuned against what this does today, so the change is a re-tune and wants a
        // sitting of its own (`ENGINE_CLEANUP.md`, F3).
        let profile = arc_profile(&spline, &self.settled_profile);
        // Weighted exactly as the solve weights them ([`arc_weights`]), which is the
        // same argument as the paragraph above carried one step further: the two must
        // agree about *which samples matter* as well as about where they sit, or the
        // price is charged for an error the solve was never trying to remove — and a
        // dwell would buy control points to trace itself.
        let qs = arc_weights(&self.pts, &idx);
        let sum: f32 = idx
            .iter()
            .zip(&qs)
            .map(|(&i, q)| {
                let s = self.pts[i];
                let c = spline.evaluate(param_at(&profile, spans, s.arc / total));
                q * ((c[0] - s.pos.x).powi(2) + (c[1] - s.pos.y).powi(2))
            })
            .sum();
        sum / qs.iter().sum::<f32>().max(1e-6)
    }

    fn adopt(&mut self, f: Fit) {
        self.geom = f.geom;
        self.attr = f.attr;
        self.first_live = f.lo;
        // The marker on the curve just adopted — before the profile is cut down
        // to its settled prefix, since the marker may still sit past it.
        self.start_param = self.start_on(&f.profile);
        // Keep only the part of the profile the frozen spans determine: those
        // control points are held, so that length is settled for good.
        let settled = self.frozen_spans() * ARC_SAMPLES_PER_SPAN;
        self.settled_profile = f.profile;
        self.settled_profile
            .truncate((settled + 1).min(self.settled_profile.len()));
    }

    fn grow_to(&mut self, m: usize) {
        self.geom = grow_rows(&self.geom, m, |j, d| {
            let p = self.at_fraction(j, m).pos;
            if d == 0 { p.x } else { p.y }
        });
        self.attr = grow_rows(&self.attr, m, |j, d| self.at_fraction(j, m).channels[d]);
    }

    /// The sample `j / (m - 1)` of the way along the stroke — where a new control
    /// point is seeded. Only the ridge's centre; the solve decides it.
    fn at_fraction(&self, j: usize, m: usize) -> Accepted {
        let want = j as f32 / (m - 1).max(1) as f32 * self.arc;
        let i = self.pts.partition_point(|s| s.arc < want);
        self.pts[i.min(self.pts.len() - 1)]
    }

    fn channels(&self, s: InputSample) -> [f32; CHANNELS] {
        [s.pressure, s.tilt.x, s.tilt.y, self.rel_time(s.time)]
    }

    fn rel_time(&self, t: f64) -> f32 {
        (t - self.t0) as f32
    }

    /// The marker: where `start_arc` sits on the curve `profile` measures, as a
    /// span-unit parameter —
    /// [`StrokeRecord::start`](stark_model::document::StrokeRecord::start)'s
    /// value for that curve.
    ///
    /// Resolved through the same [`param_at`] map the solve places samples
    /// with, so the marker is exactly where the press's report was fitted. It
    /// refines while the entry is still being solved, and settles the moment
    /// the frozen prefix's arc covers it: [`param_at`] reads only profile
    /// entries up to the marker's own arc, and those are carried over verbatim
    /// once frozen ([`arc_profile`]). Which is what lets a renderer bake spans
    /// behind the marker into a cached head (§6.2): whether the marker lies
    /// behind a frozen boundary is settled *by* that boundary freezing, so it
    /// can never move across one afterwards.
    fn start_on(&self, profile: &[f32]) -> f32 {
        let Some(a) = self.start_arc.filter(|a| *a > 0.0) else {
            return 0.0;
        };
        let spans = ((profile.len() - 1) / ARC_SAMPLES_PER_SPAN) as f32;
        param_at(profile, spans, a / self.arc.max(1e-6))
    }

    /// Whether the stroke has any travel of its own — reports accepted after
    /// its first sample. What separates a stroke from a click
    /// (`Session::end_stroke`): a swept deposit is a definite integral over
    /// travel, and the travel that counts starts at the marker — the run-up is
    /// evidence, never deposit, so a click at the end of a watched approach is
    /// still a click.
    pub fn painted(&self) -> bool {
        self.start_arc.is_some_and(|a| self.arc > a)
    }
}

/// **How much stroke each report speaks for**: the arc from halfway back to its
/// predecessor to halfway on to its successor, normalized so the weights average one.
///
/// A pointer reports on a clock, not on a ruler. The same stretch of curve therefore
/// carries as many reports as the hand took time over it, and a least-squares sum over
/// reports is not a fit to the *stroke* — it is a fit to the hand's dwell, which wins
/// wherever the two disagree by sheer count.
///
/// That has a name: **the pen leaving the tablet**. A tablet keeps sampling through the
/// release, so a stroke ends with a run of reports carrying the pressure to zero across
/// a fraction of a pixel of tip drift. They land at the very end of the parameter
/// domain, and unweighted they outvote the whole last span of real curve — measured
/// before this weight existed, the fitted pressure came down over 88 px of a 563 px
/// `LOOP_STROKE` and 134 px of an 838 px `FAST_STROKE`, reaching the tip at 0.80 and
/// 0.52 instead of the 1.0 the hand actually drew. §6.2 says a piece of path with no
/// length deposits nothing, and the renderer honours that to the bit; this is where the
/// claim was being lost. The same effect in miniature is every mid-stroke pause pulling
/// the curve into the jitter it sat in.
///
/// The weight is the trapezoid rule's, which is exactly what turns `Σ residual²` over
/// reports into `∫ residual² ds` over the stroke — the quantity that was meant all
/// along, and one that cannot be shouted down, because a report standing on no path
/// carries no weight however many of it arrive.
///
/// **Rejecting such reports instead does not work, and not for want of a threshold.**
/// A release drifts, so its reports accumulate past any fixed bar and the one that gets
/// through arrives part-decayed; the bar decides *which* release report contaminates
/// the fit, not whether one does. Swept over ×0…×1 of the input tolerance the reach was
/// non-monotone — `C_STROKE` was worse at half a tolerance (19.6 px) than at a quarter
/// (3.3 px) — and at ×1 it also decimated real input, taking `HAIRPIN_STROKE` from 22
/// knots to 15.
///
/// Weighting is not the whole cure by itself either, because at the extreme end of the
/// domain the release is the *only* evidence and a local fit follows the only evidence
/// it has, however light. What closes that is holding the attribute endpoint — see
/// [`PathFitter::solve`].
///
/// Normalized to average one so that the two knobs `solve_window` scales by the weight sum —
/// the smoothing's data pull and the ridge's floor — keep the meanings they were tuned
/// with. Input with no arc at all comes back all ones, which is the unweighted fit
/// exactly.
///
/// Returns the weights for `pts[lo..]` — the solve's window — but measures every one of
/// them against its **true** neighbours, which is why it takes the whole run and an
/// offset rather than a slice. Only the stroke's own first and last reports get a
/// half-interval; the window's leading report has a predecessor and is entitled to it.
/// Reading the slice instead put a half-interval wherever the window happened to begin,
/// which is a fact about the solve's bookkeeping and not about the stroke — enough, on
/// its own, to move `fit_collapses_pixel_staircase` by a control point.
fn arc_weights(pts: &[Accepted], idx: &[usize]) -> Vec<f32> {
    let k = idx.len();
    if pts.len() < 2 || k < 2 {
        return vec![1.0; k];
    }
    let mut q: Vec<f32> = (0..k)
        .map(|j| {
            // Measured against the neighbouring **survivors**, which is what makes the
            // sum a trapezoid rule over the reports actually being fitted rather than
            // over the ones that happened to arrive. Where nothing was decimated the
            // survivors *are* the neighbours and this is the plain rule it always was.
            let hi = if j + 1 < k {
                pts[idx[j + 1]].arc
            } else {
                // The stroke's own last report has no successor, so it gets the one
                // half-interval it is entitled to.
                pts[idx[j]].arc
            };
            let low = if j > 0 {
                pts[idx[j - 1]].arc
            } else {
                // The *window's* leading report is not the stroke's: it has a real
                // predecessor in `pts` and is entitled to it. Only the stroke's own
                // first report is halved, and `saturating_sub` is what says so.
                pts[idx[0].saturating_sub(1)].arc
            };
            // Halved because an interior report spans two half-intervals. An end report
            // has only the one, and `hi == low` on that side already halves it.
            (hi - low) * 0.5
        })
        .collect();
    let total: f32 = q.iter().sum();
    if total <= 1e-6 {
        return vec![1.0; k];
    }
    let scale = k as f32 / total;
    for w in &mut q {
        *w *= scale;
    }
    q
}

/// How many reports the solve will minimize over at once, however many arrive.
///
/// **The bound the window did not have.** `solve`'s window is delimited in *span
/// parameter* — three spans at the live end — and a span is `KNOT_SPACING ×
/// tolerance` canvas px wide, so the count of reports inside it is the product of two
/// things the fitter does not control: how densely the digitizer reports per canvas
/// px, and how far out the view is zoomed. Each scaled the work linearly and the
/// stroke quadratically. Measured on `LOOP_STROKE` before this existed
/// (`benches/path.rs`): per-report throughput fell 19× across a 32× range of report
/// density, so the densest row cost 612× the total of the sparsest for the same mark;
/// and 6.4× across the tolerance range, in the direction nobody would guess — a
/// coarser tolerance produces *fewer* knots and used to cost far more per report.
///
/// **Decimating is sound because of what the weights already are.** [`arc_weights`]
/// exists to turn `Σ residual²` over reports into `∫ residual² ds` over the stroke,
/// so that a hand's dwell cannot outvote geometry by sheer count. That is exactly the
/// property that makes a subset with the trapezoid weights recomputed for it
/// approximate the *same* integral: what is dropped is sampling rate, and the
/// objective was never a function of sampling rate. Rejecting reports would be a
/// different operation and `arc_weights`' own header rules it out — but that argument
/// is about discarding evidence from the integral, not about evaluating it at fewer
/// points.
///
/// **64 is 16 observations per free control point** — the window solves for four —
/// which is ample for least squares over input whose whole problem is jitter. It is
/// set from what the fit needs, and deliberately not from what leaves the goldens
/// alone.
///
/// That distinction has a bill attached, so it is stated rather than implied.
/// [`window_indices`] is the identity under budget and `arc_weights` then reduces to
/// the rule it always was, so a stroke whose window fits is bit-identical to what it
/// was. `LOOP_STROKE` at [`DEFAULT_TOLERANCE`] does **not** fit — the corpus's `taper`
/// case is drawn from it, and re-blessing moved 1.66% of its texels by more than 6
/// levels of 255, against the 12 the corpus itself calls visible. The two renders are
/// indistinguishable; what moved is where a few antialiased edges land, because the
/// fitted curve moved a fraction of a pixel.
///
/// Raising the budget until that golden stopped moving was the obvious alternative and
/// is the wrong shape: it would be a constant chosen to preserve an old output rather
/// than to serve the fit, and it would weaken the bound precisely where the bound is
/// the point — the cost of a report is linear in this number, so a budget of 256 gives
/// back most of the win on the dense strokes that were quadratic. The accuracy this
/// trades is measured rather than argued and it is under 0.1%
/// (`thinning_the_window_does_not_cost_the_fit_its_accuracy`).
pub(super) const MAX_WINDOW_SAMPLES: usize = 64;

/// The reports `solve` minimizes over: `pts[lo..]`, thinned to at most
/// [`MAX_WINDOW_SAMPLES`] chosen evenly along the **arc** they cover.
///
/// Evenly in arc rather than in index, because arc is the axis the objective is an
/// integral over — a hand that paused would otherwise spend the whole budget on the
/// pause. Both ends are kept whatever the budget: the leading report anchors the
/// window against the frozen prefix, and the trailing one is where the pen is now,
/// which is the report the curve is pinned to.
///
/// Deterministic, and a pure function of the accepted reports — so a replay, a peer
/// and a golden all decimate identically, which is what keeps this a performance
/// change rather than a wire-format one (§1).
pub(super) fn window_indices(pts: &[Accepted], lo: usize) -> Vec<usize> {
    let n = pts.len();
    if lo >= n {
        return Vec::new();
    }
    if n - lo <= MAX_WINDOW_SAMPLES {
        return (lo..n).collect();
    }
    let span = pts[n - 1].arc - pts[lo].arc;
    // No arc to spread over — a run of reports at one point, which `push` mostly
    // rejects but a long enough stroke can still round into. Even by index is the
    // same rule read through the only ordering left.
    if span <= 0.0 {
        let stride = (n - lo).div_ceil(MAX_WINDOW_SAMPLES);
        let mut out: Vec<usize> = (lo..n).step_by(stride).collect();
        if out.last() != Some(&(n - 1)) {
            out.push(n - 1);
        }
        return out;
    }
    let step = span / (MAX_WINDOW_SAMPLES - 1) as f32;
    let mut out = Vec::with_capacity(MAX_WINDOW_SAMPLES + 1);
    let mut next = pts[lo].arc;
    for (i, s) in pts.iter().enumerate().take(n - 1).skip(lo) {
        if s.arc >= next {
            out.push(i);
            next = s.arc + step;
        }
    }
    out.push(n - 1);
    out
}

/// The control points a `(geom, attr)` pair stands for — one mapping shared by
/// [`PathFitter::path`] and [`PathFitter::path_as_finished`], so the two cannot
/// disagree about anything but which solve they read.
fn control_points(geom: &GeomCtrl, attr: &ChannelCtrl) -> Vec<ControlPoint> {
    // Through `clamped`, which is where the range a pen can report is stated for
    // every fitter (`ControlPoint::clamped`) — the channels are solved the same way
    // the geometry is, so a point the data barely holds can overshoot them.
    (0..geom.nrows())
        .map(|j| {
            ControlPoint::clamped(
                Vec2::new(geom[(j, 0)], geom[(j, 1)]),
                attr[(j, 0)],
                Vec2::new(attr[(j, 1)], attr[(j, 2)]),
                attr[(j, 3)],
            )
        })
        .collect()
}

/// One candidate fit, before the growth rule has chosen between two of them.
struct Fit {
    geom: GeomCtrl,
    attr: ChannelCtrl,
    lo: usize,
    /// Arc profile of this candidate, to be kept as far as it is now frozen.
    profile: Vec<f32>,
}

/// `rows` lengthened to `m`, with new entries from `seed`.
fn grow_rows<const E: usize>(
    rows: &OMatrix<f32, Dyn, Const<E>>,
    m: usize,
    seed: impl Fn(usize, usize) -> f32,
) -> OMatrix<f32, Dyn, Const<E>> {
    let have = rows.nrows();
    if have >= m {
        return rows.clone();
    }
    OMatrix::<f32, Dyn, Const<E>>::from_fn_generic(Dyn(m), Const::<E>, |j, d| {
        if j < have { rows[(j, d)] } else { seed(j, d) }
    })
}

fn set_row<const E: usize>(rows: &mut OMatrix<f32, Dyn, Const<E>>, j: usize, v: [f32; E]) {
    for (d, x) in v.into_iter().enumerate() {
        rows[(j, d)] = x;
    }
}

/// Samples per span used to measure a curve's own arc length (see `arc_profile`).
const ARC_SAMPLES_PER_SPAN: usize = 4;

/// Cumulative arc length along `curve`, sampled evenly in *parameter*.
///
/// A clamped B-spline is not parameterized by distance: the triple knots at each end
/// squash the first and last spans into a fraction of the leg they cover, so
/// parameter `t` is well short of `t / num_spans` of the way along. Assuming
/// otherwise leaves a residual on input the curve could fit exactly — a straight
/// stroke read as several px of error — and the growth rule then buys control points
/// to explain it away. Sparse input has nothing between samples to hold those extra
/// control points, so they oscillate.
/// `settled` is a profile of the same curve's *frozen* spans, carried over from an
/// earlier update: those spans' control points are held, so their geometry — and so
/// their length — cannot change, and re-walking them every update is the last piece
/// of per-update work that scaled with the whole stroke rather than the window.
pub(crate) fn arc_profile(curve: &CubicBSpline<'_, 2>, settled: &[f32]) -> Vec<f32> {
    let spans = curve.num_spans();
    let n = spans * ARC_SAMPLES_PER_SPAN;
    let keep = settled.len().saturating_sub(1).min(n);
    let mut cum = Vec::with_capacity(n + 1);
    if keep == 0 {
        cum.push(0.0);
    } else {
        cum.extend_from_slice(&settled[..=keep]);
    }
    let step = |i: usize| i as f32 / ARC_SAMPLES_PER_SPAN as f32;
    let mut prev = curve.evaluate(step(keep));
    for i in keep + 1..=n {
        let c = curve.evaluate(step(i));
        let d = ((c[0] - prev[0]).powi(2) + (c[1] - prev[1]).powi(2)).sqrt();
        cum.push(cum[i - 1] + d);
        prev = c;
    }
    cum
}

/// The parameter at which `profile`'s curve is `f` of the way along its own length.
///
/// This is a *global, monotone* reparameterization: one function, applied to every
/// sample alike. That is what separates it from projecting each sample onto the
/// curve independently — samples keep their order and their relative spacing, so
/// they cannot slide past one another or bunch up on the input's jitter, which is
/// how per-sample correction blew strokes up to 45px with loops.
pub(crate) fn param_at(profile: &[f32], spans: f32, f: f32) -> f32 {
    let total = *profile.last().expect("profile is never empty");
    if total <= 1e-6 || profile.len() < 2 {
        return f.clamp(0.0, 1.0) * spans;
    }
    let want = f.clamp(0.0, 1.0) * total;
    let i = profile
        .partition_point(|&c| c < want)
        .clamp(1, profile.len() - 1);
    let (a, b) = (profile[i - 1], profile[i]);
    let u = if b > a { (want - a) / (b - a) } else { 0.0 };
    (((i - 1) as f32 + u) / ARC_SAMPLES_PER_SPAN as f32).min(spans)
}

/// Fit a whole sample sequence in one call at [`DEFAULT_TOLERANCE`] — the batch form
/// of [`PathFitter`], used by replay and tests. Identical output to feeding the same
/// samples one at a time and finishing.
pub fn fit(samples: &[InputSample]) -> Vec<ControlPoint> {
    fit_with_tolerance(samples, DEFAULT_TOLERANCE)
}

/// [`fit`] for input of a stated resolution (see [`PathFitter::with_tolerance`]).
pub fn fit_with_tolerance(samples: &[InputSample], tolerance: f32) -> Vec<ControlPoint> {
    let mut f = PathFitter::with_tolerance(tolerance);
    for s in samples {
        f.push(*s);
    }
    f.finish();
    f.path()
}

pub(super) fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 < 1e-12 {
        0.0
    } else {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    };
    (p - (a + ab * t)).length()
}
