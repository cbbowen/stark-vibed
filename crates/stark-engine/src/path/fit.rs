//! **Fitting**: input samples → control points (§6.2).
//!
//! A streaming least-squares fit — see the module note in [`super`] for where this
//! sits between the three.

use super::arclen::{ARC_SAMPLES_PER_SPAN, arc_profile_into, param_at};
use super::{frozen_spans_for, span_count};
use crate::command::InputSample;
use crate::spline::{CubicBSpline, Observations, SplineIndex};
use nalgebra::{Const, Dyn, OMatrix};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;
use std::cell::RefCell;

/// Control points solved for at the live end of the stroke. Everything behind them
/// is frozen; the pinned endpoint sits inside the window on top of these.
///
/// This is *the* accuracy/stability trade, and small is the point. A short window
/// means the settled stroke cannot move under the pointer — which is what stops a
/// live stroke wobbling along its whole length — and it caps the solve at a handful
/// of unknowns however long the stroke gets. Too short and the curve cannot round a
/// corner before the corner is committed.
const FREE_CONTROL_POINTS: usize = 3;

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
/// Below this, a report is the same place as the one before it: no arc accrues and
/// no sample is recorded. Also the floor under [`PathFitter::arc_total`], so the
/// distance a fit divides by is never smaller than the step it refuses to measure.
const MIN_STEP_PX: f32 = 1e-6;

/// Control points held at their values for a polygon of `m`: everything but the free
/// window at the live end, and the pinned endpoint inside it.
///
/// The one statement of the rule. `PathFitter::frozen` asks it of the polygon as it
/// stands and publishes the answer — `frozen_points` goes on the wire (§17.5) and
/// `frozen_spans` drives the renderer's cached head; `solve` asks it of a *candidate*
/// polygon, which is `m` or `m + 1`. The two drifting means published control points
/// the solve is still moving.
fn frozen_at(m: usize) -> usize {
    m.saturating_sub(FREE_CONTROL_POINTS + 1)
        .max(usize::from(m > 0))
}

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
/// (see [`SplineIndex::fit_channels`](crate::spline::SplineIndex::fit_channels)).
///
/// Least squares charges the curve for being far from a *point*, never for where it
/// goes when no point is near — so a stretch the data does not constrain is free to
/// wander. With the correspondence declared rather than searched this is a much
/// milder problem than it was, but a control point at the very end of the window
/// still has little holding it, and this is what settles it onto its neighbours'
/// continuation.
const SMOOTHING: f32 = 0.05;

/// Per-point channels carried alongside the geometry: pressure, tilt x/y, time.
///
/// **The order is the layout**, and `[0]`/`[1]`/`[2]`/`[3]` are written out at a
/// dozen sites here and in `assist::realize`, which fits the same four the same way.
/// Reordering the array is therefore a change that mis-maps pressure onto tilt in
/// whichever module was not edited, with nothing failing to compile and the fit still
/// converging — so the two ends of it, the only places the indices actually mean
/// anything, is [`control_point_from`] below — the one direction that is genuinely
/// shared. The *reading* direction is not: the fitter reads an `InputSample` with the
/// clock re-based onto the stroke, and `assist::realize` reads a flattened sample with
/// the path's own, so those are two sources rather than one function written twice.
pub(crate) const CHANNELS: usize = 4;

/// Which of [`CHANNELS`] is the clock. The odd one out: the other three are pen state,
/// which the tip's shape follows, while this one is only a stamp on the report — which
/// is why [`PathFitter::solve`] treats the stroke's last one differently.
const TIME_CHANNEL: usize = 3;

/// A solved row back into a [`ControlPoint`] — **through `clamped`**, which is where
/// the range a pen can report is stated for every fitter. The channels are solved the
/// same way the geometry is, so a point the data barely holds can overshoot them, and
/// `assist::realize`'s solve overshoots the same way.
pub(crate) fn control_point_from(pos: Vec2, ch: [f32; CHANNELS]) -> ControlPoint {
    ControlPoint::clamped(pos, ch[0], Vec2::new(ch[1], ch[2]), ch[3])
}

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
///   ([`grow_rows`]) is an `O(m)` copy around an `O(1)` solve, and the two per
///   candidate are the reason a long stroke's last report still costs more than its
///   first. Everything else a report touches is bounded by the window: the arc
///   profile walks only past the settled prefix, and the observation buffers are
///   kept and refilled rather than allocated ([`Scratch`]).
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
    /// Buffers every solve refills rather than reallocates. A `RefCell` because
    /// [`Self::as_finished`] solves under `&self`; taken out for the length of a
    /// solve and put back, so no borrow is held across one.
    scratch: RefCell<Scratch>,
    /// [`Self::as_finished`]'s last answer, keyed by the state it was solved from.
    /// The key is what a report changes — see [`Memo`] — so there is no counter a
    /// new mutation path could forget to bump.
    finished_memo: RefCell<Option<Memo>>,
}

/// One accepted report: where it is, what the pen said, and how far along it sits.
///
#[derive(Copy, Clone, Debug)]
struct Accepted {
    pos: Vec2,
    channels: [f32; CHANNELS],
    arc: f32,
}

/// What [`PathFitter::as_finished`] answered, and from which state.
///
/// Every mutation the fitter has goes through `push` or `finish`, and `finish` takes
/// `as_finished` off this path altogether. A `push` that changes anything either
/// appends to `pts` or — for a press the spacing gate dropped — sets `start_arc`,
/// so the pair is the whole of what the answer can differ by.
struct Memo {
    key: (usize, Option<f32>),
    path: Vec<ControlPoint>,
    start: f32,
}

/// The reports one candidate solve minimizes over, and what the solve reads off
/// them — a pure function of the accepted reports and the candidate's polygon, so
/// the scoring of *both* candidates reads the as-is one rather than rebuilding it.
///
/// A report's worth of these is a few hundred numbers, refilled in place: at up to a
/// thousand reports a second, allocating them afresh was most of a `push`'s heap
/// traffic.
#[derive(Default)]
struct Window {
    /// First accepted report that can still reach a row being solved for.
    lo: usize,
    /// The reports minimized over ([`window_indices`]).
    idx: Vec<usize>,
    /// Their curve parameters.
    ts: Vec<f32>,
    /// What each stands for ([`arc_weights`]).
    qs: Vec<f32>,
    /// Their positions and pen channels, contiguous for the solve.
    pos: Vec<[f32; 2]>,
    vals: Vec<[f32; CHANNELS]>,
    /// Arc profile of the candidate polygon as seeded, which `ts` was read off — and
    /// what the adopted candidate's settled prefix is cut from.
    profile: Vec<f32>,
}

/// The buffers a report's solves share. See [`PathFitter::scratch`].
#[derive(Default)]
struct Scratch {
    /// One per candidate polygon a report solves: as it stands, and one larger.
    windows: [Window; 2],
    /// The scoring's own arc profile ([`PathFitter::mean_error`]).
    scored: Vec<f32>,
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
            scratch: RefCell::default(),
            finished_memo: RefCell::new(None),
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
                if step < MIN_STEP_PX {
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
            if step < MIN_STEP_PX {
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
        let mut scratch = std::mem::take(self.scratch.get_mut());
        let Scratch {
            windows: [as_is_w, grown_w],
            scored,
        } = &mut scratch;
        let as_is = self.solve(m, as_is_w);
        let grown = self.solve(m + 1, grown_w);
        // Both candidates are scored over the **as-is** window — see `mean_error`.
        let err_as_is = self.mean_error(&as_is, as_is_w, scored);
        let err_grown = self.mean_error(&grown, as_is_w, scored);
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
            self.adopt(grown, grown_w);
        } else {
            self.adopt(as_is, as_is_w);
        }
        *self.scratch.get_mut() = scratch;
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
            let mut scratch = std::mem::take(self.scratch.get_mut());
            let w = &mut scratch.windows[0];
            let last = self.solve(self.geom.nrows(), w);
            self.adopt(last, w);
            *self.scratch.get_mut() = scratch;
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
    ///
    /// Memoized ([`Memo`]): the fold and the presence frame both ask, per frame,
    /// from the same state, and the second answer is a clone of the first.
    pub fn as_finished(&self) -> (Vec<ControlPoint>, f32) {
        if self.finished || self.geom.nrows() < 2 {
            return (self.path(), self.start_param);
        }
        let key = (self.pts.len(), self.start_arc);
        if let Some(m) = self
            .finished_memo
            .borrow()
            .as_ref()
            .filter(|m| m.key == key)
        {
            return (m.path.clone(), m.start);
        }
        let mut scratch = self.scratch.take();
        let w = &mut scratch.windows[0];
        let f = self.solve(self.geom.nrows(), w);
        let start = self.start_on(&w.profile);
        let path = control_points(&f.geom, &f.attr);
        *self.scratch.borrow_mut() = scratch;
        *self.finished_memo.borrow_mut() = Some(Memo {
            key,
            path: path.clone(),
            start,
        });
        (path, start)
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
        frozen_at(self.geom.nrows())
    }

    /// The total arc the fit parameterizes against, floored off zero.
    ///
    /// The floor is what keeps `a / total` finite for the first report of a stroke,
    /// where no distance has accumulated yet. Written three times before this, and it
    /// is the denominator every sample's curve parameter goes through — so the three
    /// disagreeing is the one way a sample could be assigned to a different place on
    /// the curve depending on which of them asked.
    fn arc_total(&self) -> f32 {
        self.arc.max(MIN_STEP_PX)
    }

    /// Solve for the free window at a polygon of `m` control points: the candidate
    /// seeded and pinned, the reports that reach its free rows gathered into `w`,
    /// and the geometry and the pen channels each fitted **into** the candidate.
    fn solve(&self, m: usize, w: &mut Window) -> Fit {
        let m = m.max(2);
        let mut fit = self.seed_candidate(m);
        self.observation_window(&fit, w);
        let frozen = frozen_at(m);
        let index = SplineIndex::new(m).expect("at least two control points");
        // Solved **into** the candidate polygon rather than into a fresh one: the
        // prior and the result are the same buffer, which is sound because the solve
        // reads every row it uses as a prior before it writes any
        // ([`SplineIndex::fit_into`]). Returning an owned matrix instead meant a copy
        // of the whole polygon per fit, four fits per pointer report.
        index.fit_into(
            Observations {
                ts: &w.ts,
                values: &w.pos,
                weights: &w.qs,
            },
            &mut fit.geom,
            frozen,
            1,
            self.smoothing,
        );

        // The pen channels ride the same knots at the same parameters, so they are
        // the same solve with a different payload — unsmoothed, since a pressure ramp
        // is not a shape and has no curvature to penalize. Its end is held (the `1`)
        // for the reason set out where that row is written (`seed_candidate`).
        index.fit_into(
            Observations {
                ts: &w.ts,
                values: &w.vals,
                weights: &w.qs,
            },
            &mut fit.attr,
            frozen,
            1,
            0.0,
        );
        fit
    }

    /// The polygon a solve starts from: `m` rows, any new ones seeded along the
    /// stroke, and both ends pinned — as held rows, so the solve places the rest of
    /// the window *around* them. Samples map onto the curve's domain by distance
    /// travelled, which puts the first and last exactly on the two ends.
    fn seed_candidate(&self, m: usize) -> Fit {
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
        // solve is the one [`Self::as_finished`] mirrors, so preview and commit see
        // the same lag and agree to the bit (§1.3).
        let held: [f32; CHANNELS] = std::array::from_fn(|d| attr[(m - 2, d)]);
        set_row(&mut attr, m - 1, held);
        // …except the clock, which is not a pen attribute at all. `time` is what the
        // report was stamped with, and the release genuinely happened then; carrying the
        // neighbour's instead would shorten every stroke's recorded duration by a span
        // and quietly skew the timelapse (§8).
        attr[(m - 1, TIME_CHANNEL)] = last.channels[TIME_CHANNEL];
        Fit { geom, attr }
    }

    /// The reports a solve of `candidate` minimizes over, into `w`: which reach a free
    /// row, where each sits on the curve, and what each weighs.
    fn observation_window(&self, candidate: &Fit, w: &mut Window) {
        let Window {
            lo,
            idx,
            ts,
            qs,
            pos,
            vals,
            profile,
        } = w;
        // The curve as the candidate polygon stands, borrowed rather than copied — it
        // is read to measure arc length and then dropped, and the fits write back into
        // the candidate themselves ([`SplineIndex`]).
        let spline: CubicBSpline<'_, 2> =
            CubicBSpline::new(&candidate.geom).expect("at least two control points");
        let m = candidate.geom.nrows();
        let spans = spline.num_spans() as f32;
        arc_profile_into(&spline, &self.settled_profile, profile);
        let total = self.arc_total();
        let param = |a: f32| param_at(profile, spans, a / total);

        // A cubic B-spline's basis is local, so a sample sitting under the frozen
        // prefix cannot influence any row still being solved. `m` is already `>= 2`
        // here, which is why this and `Self::frozen` are one rule despite one of them
        // having clamped to 1 and the other to 0 for the empty polygon.
        let frozen = frozen_at(m);
        // How far back a frozen row's support reaches, in control points: a cubic's
        // basis touches `ORDER` of them, and the row itself is one of those.
        let cutoff = frozen as f32 - (crate::spline::ORDER - 2) as f32;
        *lo = self.first_live.min(self.pts.len() - 1);
        while *lo + 1 < self.pts.len() && param(self.pts[*lo].arc) < cutoff {
            *lo += 1;
        }
        // …and of the reports that *do* reach a free row, at most a bounded number are
        // minimized over — see [`window_indices`], which is the whole of why this
        // costs what it costs rather than what the digitizer charges for it.
        window_indices(&self.pts, *lo, idx);
        pos.clear();
        pos.extend(idx.iter().map(|&i| self.pts[i].pos.to_array()));

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
        ts.clear();
        ts.extend(idx.iter().map(|&i| param(self.pts[i].arc)));
        // What each report stands for, so the solve minimizes over the *stroke* rather
        // than over the reporting clock ([`arc_weights`]).
        arc_weights(&self.pts, idx, qs);
        vals.clear();
        vals.extend(idx.iter().map(|&i| self.pts[i].channels));
    }

    /// Mean **arc-weighted** squared distance from the reports in `w` to `fit`'s
    /// curve.
    ///
    /// Both candidates must be scored over the **same** samples — the as-is
    /// candidate's window. Each solve drops the ones its own frozen prefix has
    /// swallowed, and the larger polygon freezes one more — so scoring each on its own
    /// slice compares a sum over fewer points against a sum over more, which the
    /// larger one wins every time regardless of whether it fits better. That made a
    /// dead-straight stroke take a control point per sample. Per-sample rather than
    /// total for the same reason in miniature: a total grows with the window, so a
    /// fixed price would mean something different at every length.
    ///
    /// `w` is the solve's own window, not one rebuilt here: **the same reports the
    /// solve minimized over**, decimated by the same rule and weighted exactly as the
    /// solve weighted them ([`window_indices`], [`arc_weights`]). The two have to
    /// agree about which samples matter as well as about where they sit, or the price
    /// is charged for an error the solve was never trying to remove — and a dwell
    /// would buy control points to trace itself.
    fn mean_error(&self, fit: &Fit, w: &Window, profile: &mut Vec<f32>) -> f32 {
        // Borrowed, not copied: scoring a candidate reads its control points and never
        // moves them, so a copy here would be a whole polygon per candidate per report
        // to answer one number.
        let spline = CubicBSpline::new(&fit.geom).expect("at least two control points");
        let spans = spline.num_spans() as f32;
        let total = self.arc_total();
        if w.idx.is_empty() {
            return 0.0;
        }
        // The same *rule* as the solve's parameters, off a later curve — and the gap is
        // real, so it is written down rather than claimed away. The solve reads its
        // parameters off the polygon as *seeded*, before `fit_into` writes back
        // (`w.profile`), where this builds a profile from the spline that came out. Both
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
        // Taking `w.profile` instead would close the gap outright and drop two of the
        // four curve walks a report costs. It is not done here because `KNOT_COST` was
        // tuned against what this does today, so the change is a re-tune and wants a
        // sitting of its own.
        arc_profile_into(&spline, &self.settled_profile, profile);
        let sum: f32 = w
            .idx
            .iter()
            .zip(&w.qs)
            .map(|(&i, q)| {
                let s = self.pts[i];
                let c = spline.evaluate(param_at(profile, spans, s.arc / total));
                q * ((c[0] - s.pos.x).powi(2) + (c[1] - s.pos.y).powi(2))
            })
            .sum();
        sum / w.qs.iter().sum::<f32>().max(1e-6)
    }

    /// Take `f` — solved over `w` — as the polygon.
    fn adopt(&mut self, f: Fit, w: &Window) {
        self.geom = f.geom;
        self.attr = f.attr;
        self.first_live = w.lo;
        // The marker on the curve just adopted — before the profile is cut down
        // to its settled prefix, since the marker may still sit past it.
        self.start_param = self.start_on(&w.profile);
        // Keep only the part of the profile the frozen spans determine: those
        // control points are held, so that length is settled for good.
        let settled = self.frozen_spans() * ARC_SAMPLES_PER_SPAN;
        self.settled_profile.clear();
        self.settled_profile
            .extend_from_slice(&w.profile[..(settled + 1).min(w.profile.len())]);
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

    /// A report's channels, in [`CHANNELS`] order, with the clock re-based onto this
    /// stroke's own start.
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
    /// once frozen ([`arc_profile_into`](super::arclen::arc_profile_into)). Which is what lets a renderer bake spans
    /// behind the marker into a cached head (§6.2): whether the marker lies
    /// behind a frozen boundary is settled *by* that boundary freezing, so it
    /// can never move across one afterwards.
    fn start_on(&self, profile: &[f32]) -> f32 {
        let Some(a) = self.start_arc.filter(|a| *a > 0.0) else {
            return 0.0;
        };
        let spans = ((profile.len() - 1) / ARC_SAMPLES_PER_SPAN) as f32;
        param_at(profile, spans, a / self.arc_total())
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
/// Writes the weights for the reports `idx` names — the solve's window — into `out`,
/// emptied first, but measures every one of them against its **true** neighbours,
/// which is why it takes the whole run rather than a slice. Only the stroke's own first
/// and last reports get a half-interval; the window's leading report has a predecessor
/// and is entitled to it. Reading the slice instead put a half-interval wherever the
/// window happened to begin, which is a fact about the solve's bookkeeping and not
/// about the stroke — enough, on its own, to move `fit_collapses_pixel_staircase` by a
/// control point.
fn arc_weights(pts: &[Accepted], idx: &[usize], out: &mut Vec<f32>) {
    let k = idx.len();
    out.clear();
    if pts.len() < 2 || k < 2 {
        out.resize(k, 1.0);
        return;
    }
    out.extend((0..k).map(|j| {
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
    }));
    let total: f32 = out.iter().sum();
    if total <= 1e-6 {
        out.fill(1.0);
        return;
    }
    let scale = k as f32 / total;
    for w in out.iter_mut() {
        *w *= scale;
    }
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
const MAX_WINDOW_SAMPLES: usize = 64;

/// The reports `solve` minimizes over, into `out` (emptied first): `pts[lo..]`, thinned
/// to at most [`MAX_WINDOW_SAMPLES`] chosen evenly along the **arc** they cover.
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
fn window_indices(pts: &[Accepted], lo: usize, out: &mut Vec<usize>) {
    out.clear();
    let n = pts.len();
    if lo >= n {
        return;
    }
    if n - lo <= MAX_WINDOW_SAMPLES {
        out.extend(lo..n);
        return;
    }
    let span = pts[n - 1].arc - pts[lo].arc;
    // No arc to spread over — a run of reports at one point, which `push` mostly
    // rejects but a long enough stroke can still round into. Even by index is the
    // same rule read through the only ordering left.
    if span <= 0.0 {
        let stride = (n - lo).div_ceil(MAX_WINDOW_SAMPLES);
        out.extend((lo..n).step_by(stride));
        if out.last() != Some(&(n - 1)) {
            out.push(n - 1);
        }
        return;
    }
    let step = span / (MAX_WINDOW_SAMPLES - 1) as f32;
    out.reserve(MAX_WINDOW_SAMPLES + 1);
    let mut next = pts[lo].arc;
    for (i, s) in pts.iter().enumerate().take(n - 1).skip(lo) {
        if s.arc >= next {
            out.push(i);
            next = s.arc + step;
        }
    }
    out.push(n - 1);
}

/// The control points a `(geom, attr)` pair stands for — one mapping shared by
/// [`PathFitter::path`] and [`PathFitter::as_finished`], so the two cannot
/// disagree about anything but which solve they read.
fn control_points(geom: &GeomCtrl, attr: &ChannelCtrl) -> Vec<ControlPoint> {
    (0..geom.nrows())
        .map(|j| {
            control_point_from(
                Vec2::new(geom[(j, 0)], geom[(j, 1)]),
                std::array::from_fn(|d| attr[(j, d)]),
            )
        })
        .collect()
}

/// One candidate polygon, before the growth rule has chosen between two of them.
/// What was read off it to solve it — and what adopting it keeps — is in the
/// [`Window`] it was solved over.
struct Fit {
    geom: GeomCtrl,
    attr: ChannelCtrl,
}

/// `rows` lengthened to `m`, with new entries from `seed`.
fn grow_rows<const E: usize>(
    rows: &OMatrix<f32, Dyn, Const<E>>,
    m: usize,
    seed: impl Fn(usize, usize) -> f32,
) -> OMatrix<f32, Dyn, Const<E>> {
    let have = rows.nrows();
    // The name says "grow", and every caller means it: `solve` is called with the
    // polygon's own row count or one more. Worth stating, because the early return
    // hands back *more* rows than `m` if it is ever called with fewer — and `solve`
    // then writes its pinned endpoint to `m - 1`, which would be the wrong row.
    debug_assert!(
        have <= m,
        "grow_rows shrinks nothing: asked for {m} rows from {have}",
    );
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

#[cfg(test)]
mod tests {
    use super::super::arc::point_segment_distance;
    use super::*;
    use crate::path::{FLATTEN_TOLERANCE, flatten};

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

    /// One pointer report at a position, with everything else at rest. The root's test
    /// module has a copy, and so does `flatten`'s: two lines each against making a test
    /// builder visible across three module boundaries.
    fn sample(x: f32, y: f32) -> InputSample {
        InputSample::at(Vec2::new(x, y))
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

    /// **Under budget the selection is the identity**, which is what makes this a
    /// change to the pathological cases only. Ordinary painting keeps every report the
    /// window admitted, `arc_weights` reduces to the rule it always was, and the fitted
    /// curve is therefore bit-identical to what it was before the bound existed — so
    /// no golden moves and no recorded stroke re-fits.
    #[test]
    fn a_window_under_budget_keeps_every_report() {
        let pts = accepted(MAX_WINDOW_SAMPLES, 1.0);
        let mut idx = Vec::new();
        for lo in [0, 1, 7, MAX_WINDOW_SAMPLES - 1] {
            window_indices(&pts, lo, &mut idx);
            assert_eq!(
                idx,
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
            let mut idx = Vec::new();
            window_indices(&pts, 0, &mut idx);
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
        let mut idx = Vec::new();
        window_indices(&pts, 0, &mut idx);
        let in_dwell = idx.iter().filter(|&&i| i < 100).count();
        assert!(
            in_dwell <= 2,
            "{in_dwell} of {} survivors were spent on a dwell",
            idx.len(),
        );
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
}
