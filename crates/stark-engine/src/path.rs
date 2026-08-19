//! Stroke path fitting and adaptive flattening (§6.2).
//!
//! Three representations, deliberately distinct:
//!
//! - [`InputSample`] — one raw pointer report, as it arrived. High frequency,
//!   jittery, and never stored: it exists only between the pointer event and the
//!   fitter.
//! - [`ControlPoint`] — a knot of the fitted stroke curve. This is what a stroke
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

use std::ops::Range;

use nalgebra::{Const, Dyn, OMatrix};

use crate::command::InputSample;
use crate::spline::{CubicBSpline, Observations};
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
/// [`ViewTransform`](stark_model::geom::ViewTransform) allows — and is what keeps the
/// window advancing, and so the per-sample work bounded, rather than letting a
/// stroke never grow a control point at all.
/// `pub` so the benchmark can name the bound rather than restate it: the tolerance
/// sweep in `benches/path.rs` exists to hold the fit's cost flat *to the top of this
/// range*, and a hard-coded 64.0 there would be a second copy of the number that
/// decides what the range is.
pub const MIN_TOLERANCE: f32 = 1.0 / 64.0;
/// See [`MIN_TOLERANCE`].
pub const MAX_TOLERANCE: f32 = 64.0;

/// Curvature penalty on the control polygon, as a fraction of the data's own pull
/// (see [`CubicBSpline::fit_channels`]).
///
/// Least squares charges the curve for being far from a *point*, never for where it
/// goes when no point is near — so a stretch the data does not constrain is free to
/// wander. With the correspondence declared rather than searched this is a much
/// milder problem than it was, but a control point at the very end of the window
/// still has little holding it, and this is what settles it onto its neighbours'
/// continuation.
const SMOOTHING: f32 = 0.02;

/// Per-point channels carried alongside the geometry: pressure, tilt x/y, time.
const CHANNELS: usize = 4;

/// Which of [`CHANNELS`] is the clock. The odd one out: the other three are pen state,
/// which the tip's shape follows, while this one is only a stamp on the report — which
/// is why [`PathFitter::solve`] treats the stroke's last one differently.
const TIME_CHANNEL: usize = 3;

type ChannelCtrl = OMatrix<f32, Dyn, Const<CHANNELS>>;
type GeomCtrl = OMatrix<f32, Dyn, Const<2>>;

/// A point sampled *from* the curve: where it is, where it is heading, and the pen
/// attributes there.
///
/// `vel` is the derivative of position with respect to the span parameter — its
/// *direction* is the curve tangent, which is what [`flatten`] bounds and what
/// makes corners survive; its magnitude is an artifact of the parameterization and
/// means nothing to consumers.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IntermediateSample {
    pub pos: Vec2,
    pub vel: Vec2,
    pub pressure: f32,
    pub tilt: Vec2,
    pub time: f32,
    /// Arc length from the stroke start (canvas px), measured along the emitted
    /// polyline — the distance axis that the load drain, the color-dynamics
    /// noise, and the tool reservoir are parameterized by (§6.2).
    pub dist: f32,
}

/// The error budget [`flatten`] may spend when it replaces a piece of curve with a
/// straight segment. Every bound is absolute and brush-independent, so flattening
/// stays a pure function of the path — except `max_len`, which is where a caller
/// declares what *it* additionally needs.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FlattenTolerance {
    /// Max distance (canvas px) between a segment and the curve it replaces.
    pub position: f32,
    /// Max turn (radians) of the curve tangent across one segment.
    ///
    /// This is the bound that makes adaptive sampling *safe*: positional flatness
    /// alone is fooled by a symmetric wiggle whose midpoint happens to sit on the
    /// chord, and it says nothing about the direction the brush is swept along
    /// (which orients the footprint, §6.6). It is also what preserves corners —
    /// the tangent turns fastest exactly there, so that is where samples go.
    pub angle: f32,
    /// Max change in a pen attribute — pressure, or the length of the tilt delta —
    /// across one segment. Attributes are constant *within* a swept segment, so
    /// this is what keeps a pressure ramp from becoming a staircase of radii.
    pub attribute: f32,
    /// Max segment length (canvas px); `INFINITY` for "no opinion". The renderer
    /// sets this from quantities that vary with distance travelled but are applied
    /// per segment rather than per fragment — the `drain` falloff, the dynamics
    /// loop's reservoir cadence (see `gpu::stroke::flatten_tolerance`).
    pub max_len: f32,
    /// Tightest arc the caller can actually sweep (1/canvas px); `INFINITY` for "no
    /// opinion". Also a renderer-supplied cap rather than an error bound: the shaders
    /// sweep a curved segment by unrolling the annulus about its centre of curvature
    /// into the straight travel frame, and that stops being accurate once the tip is
    /// an appreciable fraction of the curve's own radius
    /// (see `gpu::stroke::MAX_TIP_TURN`).
    ///
    /// [`fit_arc`] enforces it, so an edge too tight to sweep is *priced* as a chord
    /// as well as drawn as one — which is what stops the budget being spent on a
    /// primitive that never gets used.
    pub max_arc_curvature: f32,
}

/// Default flattening budget. `position` is sub-pixel, so the polyline is
/// indistinguishable from the curve at canvas resolution; `angle` (≈5.7°) keeps
/// the swept direction honest — a full circle costs at least 63 segments however
/// large it is; and a 2% attribute step is far under the overlap of consecutive
/// footprints.
pub const FLATTEN_TOLERANCE: FlattenTolerance = FlattenTolerance {
    position: 0.35,
    angle: 0.1,
    attribute: 0.02,
    max_len: f32::INFINITY,
    max_arc_curvature: f32::INFINITY,
};

impl FlattenTolerance {
    /// The same budget with the *error* bounds scaled by `k`, the length cap left
    /// alone — a cap encodes a hard requirement, not an error. Used to trade
    /// fidelity for a bounded segment count on an extreme stroke.
    pub fn relaxed(self, k: f32) -> Self {
        Self {
            position: self.position * k,
            angle: self.angle * k,
            attribute: self.attribute * k,
            ..self
        }
    }
}

impl Default for FlattenTolerance {
    fn default() -> Self {
        FLATTEN_TOLERANCE
    }
}

/// Max bisections of a single control span: 2^10 segments, the ceiling on what any
/// one span can cost however pathological its knots.
const MAX_SUBDIVISION_DEPTH: u32 = 10;

// ---------------------------------------------------------------------------
// Fitting: input samples → control points
// ---------------------------------------------------------------------------

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
///   samples that can reach those rows take part. `spline::m_step` assembles the
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
/// ([`FlattenTolerance`]).
pub struct PathFitter {
    /// Every accepted report, with the distance along the stroke that parameterizes
    /// it. Kept whole — a few hundred is nothing — while the solve only ever looks
    /// at the tail (see `first_live`).
    pts: Vec<Accepted>,
    /// The **run-up** (§6.2): reports from before the stroke's first sample, at
    /// negative arcs measured back from it — see [`Self::seed_context`]. Never
    /// part of the emitted path; evidence for the solve alone.
    context: Vec<Accepted>,
    /// Run-up reports awaiting the first push, which is what their arcs, times
    /// and pressures are converted against ([`Self::adopt_context`]).
    pending_context: Vec<InputSample>,
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
    finished: bool,
}

/// One accepted report: where it is, what the pen said, and how far along it sits.
#[derive(Copy, Clone, Debug)]
struct Accepted {
    pos: Vec2,
    channels: [f32; CHANNELS],
    arc: f32,
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
            context: Vec::new(),
            pending_context: Vec::new(),
            first_live: 0,
            geom: GeomCtrl::zeros_generic(Dyn(0), Const::<2>),
            attr: ChannelCtrl::zeros_generic(Dyn(0), Const::<CHANNELS>),
            arc: 0.0,
            settled_profile: Vec::new(),
            grown_at: 0.0,
            smoothing: SMOOTHING,
            tolerance: if tolerance.is_finite() {
                tolerance.clamp(MIN_TOLERANCE, MAX_TOLERANCE)
            } else {
                DEFAULT_TOLERANCE
            },
            t0: 0.0,
            finished: false,
        }
    }

    /// Feed one pointer report. Ignored once the stroke is [`finish`](Self::finish)ed,
    /// and ignored if any of its numbers is non-finite
    /// ([`InputSample::is_finite`]) — the same "this report carries nothing" answer
    /// the zero-length step below gives, for a report that carries nothing usable.
    ///
    /// **Dropping it is the only total answer.** A NaN position spreads into `arc`,
    /// out of `arc` into every sample's curve parameter, and from there into the
    /// normal equations, which are then singular at every ridge — a state
    /// [`spline`](crate::spline)'s solve reports by panicking, because for finite
    /// input it cannot arise. Repairing the sample instead would mean inventing a
    /// position the hand never visited; refusing it means the stroke is exactly the
    /// stroke the finite reports describe.
    ///
    /// [`InputSample::is_finite`]: crate::command::InputSample::is_finite
    /// Seed the stroke's **run-up**: reports from before its first sample — the
    /// hover trail the engine was already watching (§18.1.10) — offered to the
    /// solve as evidence of how the hand was moving when the stroke began.
    ///
    /// **Context conditions; it never extends.** The fitted curve still starts
    /// at the first pushed sample, pinned exactly as ever, and the emitted path
    /// is unchanged in kind. What the run-up reaches is the *solve*: its reports
    /// become observation rows on the first span's polynomial extension —
    /// negative parameters (`spline::span_and_local_extended`) — so the entry
    /// direction and curvature of the committed record are informed by motion
    /// the fit could otherwise only guess at from its first, grain-quantized
    /// steps ([`Self::context_rows`]). The rows fall out of the solve on their
    /// own once the entry control points freeze, and a fitter seeded with
    /// nothing is bit-identical to one that never had this called.
    ///
    /// Call before the first [`push`](Self::push); afterwards it is ignored, as
    /// are non-finite reports. The reports' pressures are replaced with the
    /// first pushed sample's at conversion: pressure begins at the press — a
    /// hovering pen reports none — and geometry is what the run-up is evidence
    /// of. Tilt and time are kept; both are continuous through a press.
    ///
    /// A stroke that never grows past its two pinned endpoints has nothing free
    /// for the run-up to condition: a straight tap stays exactly the chord it
    /// always was, which is right — inventing curvature there would fabricate a
    /// mark the hand never made.
    pub fn seed_context(&mut self, samples: &[InputSample]) {
        if !self.pts.is_empty() || self.finished {
            return;
        }
        self.pending_context = samples.iter().copied().filter(|s| s.is_finite()).collect();
    }

    /// Convert the pending run-up against the stroke's first sample: arcs walk
    /// back from it (negative), times are re-based on it, and pressures are
    /// replaced with its own — see [`seed_context`](Self::seed_context).
    fn adopt_context(&mut self, first: &InputSample) {
        let pending = std::mem::take(&mut self.pending_context);
        let mut arc = 0.0;
        let mut ahead = first.pos;
        self.context = pending
            .iter()
            .rev()
            .map(|s| {
                arc -= (ahead - s.pos).length();
                ahead = s.pos;
                Accepted {
                    pos: s.pos,
                    channels: [first.pressure, s.tilt.x, s.tilt.y, self.rel_time(s.time)],
                    arc,
                }
            })
            .collect();
        self.context.reverse();
    }

    pub fn push(&mut self, s: InputSample) {
        if self.finished || !s.is_finite() {
            return;
        }
        match self.pts.last() {
            None => {
                self.t0 = s.time;
                self.adopt_context(&s);
            }
            Some(prev) => {
                let step = (s.pos - prev.pos).length();
                // A report that did not move carries no geometry, and a run of them
                // would put several samples at one parameter. Its attributes are no
                // loss: they apply to a zero-length piece of path.
                if step < 1e-6 {
                    return;
                }
                self.arc += step;
            }
        }
        self.pts.push(Accepted {
            pos: s.pos,
            channels: self.channels(s),
            arc: self.arc,
        });
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

    /// Every accepted report's position, in order — the **raw trace**, which the
    /// drawing assist recognizes a shape from (§6.9).
    ///
    /// Deliberately the reports rather than [`path`](Self::path): the fit is a curve
    /// pulled *towards* its control points, so those sit off the stroke by design and
    /// asking whether they lie on a circle is asking the wrong question. Nothing
    /// downstream stores these — they are already held here only because the window
    /// solve reads its own tail, and a few hundred `Vec2`s is nothing.
    pub fn trace(&self) -> Vec<Vec2> {
        self.pts.iter().map(|s| s.pos).collect()
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
                .map(|s| ControlPoint {
                    pos: s.pos,
                    pressure: s.channels[0].clamp(0.0, 1.0),
                    tilt: clamp_tilt(Vec2::new(s.channels[1], s.channels[2])),
                    time: s.channels[3],
                })
                .into_iter()
                .collect();
        }
        control_points(&self.geom, &self.attr)
    }

    /// The path exactly as [`finish`](Self::finish) would leave it, as a pure query:
    /// the one last solve with the window still free, *not* adopted.
    ///
    /// This is what a live preview must render (§1.3): the stroke that would be
    /// committed if the pen lifted now. The free window's control points sit
    /// elsewhere under a mid-stroke solve — they are still braced for data that a
    /// finished stroke never receives — and that gap is a real change of geometry at
    /// pen-up, sub-pixel but fatal to `preview == committed` wherever a discontinuous
    /// lookup (the tooth's nearest-sampled ground, §6.4) turns position into a step.
    /// Rendering the as-finished path instead makes `End` a no-op on the record by
    /// construction: nothing is pushed between the last preview and the commit, so
    /// [`finish`](Self::finish) adopts this very solve, bit for bit.
    pub fn path_as_finished(&self) -> Vec<ControlPoint> {
        if self.finished || self.geom.nrows() < 2 {
            return self.path();
        }
        let f = self.solve(self.geom.nrows());
        control_points(&f.geom, &f.attr)
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
        let mut pos: Vec<[f32; 2]> = idx.iter().map(|&i| self.pts[i].pos.to_array()).collect();

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
        let mut ts: Vec<f32> = idx.iter().map(|&i| param(self.pts[i].arc)).collect();
        // What each report stands for, so the solve minimizes over the *stroke* rather
        // than over the reporting clock ([`arc_weights`]).
        let mut qs = arc_weights(&self.pts, &idx);
        let mut vals: Vec<[f32; CHANNELS]> = idx.iter().map(|&i| self.pts[i].channels).collect();
        // The run-up's rows, while the entry control points are still free —
        // negative parameters on the first span's polynomial extension
        // ([`Self::seed_context`]). After every real row, so nothing about the
        // window's own bookkeeping moves.
        self.context_rows(frozen, &profile, &mut ts, &mut pos, &mut vals, &mut qs);
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
        // Scored at exactly the parameters the solve minimizes at. If the two use
        // different maps the growth rule reads one quantity while the solve improves
        // another, and it stops firing where the fit is actually poor — measured at
        // 4-15px on recorded strokes against 0.6-1.6px when they agree. Consistency
        // between the two matters more than accuracy in either.
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

    /// Append the run-up's observation rows ([`Self::seed_context`]) — while the
    /// entry control points are still free, and only as deep as the extension
    /// can be trusted.
    ///
    /// Each row sits at a **negative parameter**: the report's arc behind the
    /// stroke's start, carried onto the extension at the curve's **mean** arc
    /// density. Deliberately the mean and not the density the curve begins with:
    /// the clamped ends squash the first span into a fraction of the arc an
    /// interior one covers, but the squash is the triple knot's artifact, and
    /// the extension has no knots — mapped at the start's own density, a whole
    /// run-up fit inside a fraction of a pixel and reached nothing. Depth is
    /// capped at [`CONTEXT_REACH`]. The rows are weighted by the trapezoid rule
    /// among themselves and normalized to average one — the same statement
    /// [`arc_weights`] makes for the window, made locally.
    ///
    /// Not built at all once `frozen` reaches the rows the extension touches:
    /// span 0 reads the first two distinct control points, so past `frozen == 1`
    /// the rows could no longer move anything `m_step` solves for (its own
    /// reach test would drop them anyway).
    #[allow(clippy::too_many_arguments)]
    fn context_rows(
        &self,
        frozen: usize,
        profile: &[f32],
        ts: &mut Vec<f32>,
        pos: &mut Vec<[f32; 2]>,
        vals: &mut Vec<[f32; CHANNELS]>,
        qs: &mut Vec<f32>,
    ) {
        if frozen >= 2 || self.context.is_empty() {
            return;
        }
        let total = *profile.last().expect("profile is never empty");
        if total <= 1e-6 {
            return;
        }
        let spans = (profile.len() - 1) as f32 / ARC_SAMPLES_PER_SPAN as f32;
        let density = spans / total;
        // Context arcs ascend towards zero, so the kept suffix is the near end.
        let from = self
            .context
            .partition_point(|c| c.arc * density < -CONTEXT_REACH);
        let kept = &self.context[from..];
        if kept.is_empty() {
            return;
        }
        let mut w: Vec<f32> = (0..kept.len())
            .map(|j| {
                // The stroke's own start closes the run on the near side, so the
                // newest context report spans a full interval like the rest.
                let hi = if j + 1 < kept.len() {
                    kept[j + 1].arc
                } else {
                    0.0
                };
                let lo = if j > 0 { kept[j - 1].arc } else { kept[j].arc };
                (hi - lo) * 0.5
            })
            .collect();
        let sum: f32 = w.iter().sum();
        if sum <= 1e-6 {
            return;
        }
        let scale = kept.len() as f32 / sum;
        for (c, wj) in kept.iter().zip(&mut w) {
            *wj *= scale;
            ts.push(c.arc * density);
            pos.push(c.pos.to_array());
            vals.push(c.channels);
            qs.push(*wj);
        }
    }
}

/// How deep into the pre-domain extension a run-up row may sit, in spans
/// ([`PathFitter::context_rows`]).
///
/// The bound is on the row's **vote**, not on numeric stability. A row on span
/// 0's extension reaches the solve only through the far control point's basis
/// weight, `|u|³ / 6` — about 1.3 at two spans back, a strong but bounded say,
/// comparable to a couple of on-curve reports — and it grows with the cube from
/// there, until evidence extrapolated far off the curve would outvote the data
/// actually on it.
const CONTEXT_REACH: f32 = 2.0;

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
/// a fraction of a pixel of nib drift. They land at the very end of the parameter
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
/// Normalized to average one so that the two knobs `m_step` scales by the weight sum —
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
const MAX_WINDOW_SAMPLES: usize = 64;

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
fn window_indices(pts: &[Accepted], lo: usize) -> Vec<usize> {
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

/// One candidate fit, before the growth rule has chosen between two of them.
/// The control points a `(geom, attr)` pair stands for — one mapping shared by
/// [`PathFitter::path`] and [`PathFitter::path_as_finished`], so the two cannot
/// disagree about anything but which solve they read.
fn control_points(geom: &GeomCtrl, attr: &ChannelCtrl) -> Vec<ControlPoint> {
    (0..geom.nrows())
        .map(|j| ControlPoint {
            pos: Vec2::new(geom[(j, 0)], geom[(j, 1)]),
            // Clamped to the range a pen can report. The channels are solved the
            // same way the geometry is, and a control point the data barely
            // reaches is held only by the ridge, so it can overshoot the values
            // it was fitted from. For pressure that is a radius, which
            // `generate_segments` multiplies the brush by with no upper bound.
            // Clamping the control values bounds the *curve* and not just the
            // polygon: B-spline bases are non-negative and sum to one, so every
            // evaluated value is a convex combination of them.
            pressure: attr[(j, 0)].clamp(0.0, 1.0),
            tilt: clamp_tilt(Vec2::new(attr[(j, 1)], attr[(j, 2)])),
            time: attr[(j, 3)],
        })
        .collect()
}

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
pub(crate) fn arc_profile(curve: &CubicBSpline<2>, settled: &[f32]) -> Vec<f32> {
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

/// Tilt clamped to the unit disc a pen reports in — the same overshoot guard as the
/// pressure clamp in [`PathFitter::path`], for the channel that steers the footprint
/// rather than sizing it.
pub(crate) fn clamp_tilt(t: Vec2) -> Vec2 {
    let len = t.length();
    if len > 1.0 { t / len } else { t }
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

fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 < 1e-12 {
        0.0
    } else {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    };
    (p - (a + ab * t)).length()
}

// ---------------------------------------------------------------------------
// Arcs: what a flattened edge actually stands for
// ---------------------------------------------------------------------------
//
// [`flatten`] replaces a piece of curve with one edge, and the renderer sweeps that
// edge as a **circular arc** rather than a chord (§6.2). So this is where
// the arc is defined, and the flattening budget below is measured against it — the
// two have to be the same primitive or the budget is describing geometry nobody draws.
//
// Which matters twice over. A chord's error is first order in the turn and its outline
// carries a curvature *impulse* at every joint — a round join of radius `r` spliced
// between two straight runs — and periodic curvature impulses along a silhouette are
// what the eye reads as facets. An arc's error is second order and its joints only
// step the curvature slightly, so the artifact is gone whatever the segment length.
// Measuring against the arc is therefore what lets segments grow: the same declared
// error now buys a much longer edge, and buys it without the facets back.

/// The least sagitta (canvas px) an arc has to buy before an edge is bent at all.
/// Under this the arc is indistinguishable from its chord, so it is reported straight
/// — which keeps a straight or barely-curved stroke on exactly the floats it had
/// before arcs existed.
const MIN_SAGITTA: f32 = 0.01;

/// Cap on `sin(θ/2)` for the turn `θ` one edge may bend through — ~23°, far past the
/// ~5.7° [`FLATTEN_TOLERANCE`]'s `angle` bound admits. A backstop on a pathological
/// edge (a cusp, or a span that bottomed out [`MAX_SUBDIVISION_DEPTH`]) rather than a
/// quality knob: past it the series below stop being the right approximation, and so
/// does the annular sector the shaders rasterize a curved sweep as.
const MAX_HALF_TURN_SIN: f32 = 0.2;

/// One flattened edge, as it will actually be swept.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Arc {
    /// Unit tangent the arc leaves its start along.
    pub dir: Vec2,
    /// Signed curvature (1/canvas px), positive turning left of `dir`. Exactly 0 for
    /// a straight edge, and then `dir` is the chord's own direction — every consumer
    /// branches on that zero to keep the straight case on its original arithmetic.
    pub curvature: f32,
    /// Length along the arc.
    pub length: f32,
}

/// `sin(x)`, `1 − cos(x)` and `asin(u)/u` as **polynomials**, for the same reason
/// [`taper_profile`](crate::gpu::stroke) is one: these decide where segments are cut
/// and so which pixels are stored, and replay, goldens and peers all have to agree on
/// them to the last bit — which the transcendental library functions are not specified
/// to. Each is a truncated Maclaurin series, accurate to better than 1e-7 relative
/// over the range its caller is guarded to ([`MAX_HALF_TURN_SIN`]), well inside the
/// f32 the result lands in.
///
/// `versin` is `1 − cos(x)` evaluated *directly* rather than by subtraction: it is
/// used where `x` is small and the difference would cancel away most of its digits.
fn sin_small(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 / 5040.0)))
}

fn versin_small(x: f32) -> f32 {
    let x2 = x * x;
    x2 * (0.5 - x2 * (1.0 / 24.0 - x2 / 720.0))
}

fn asin_over_x(u: f32) -> f32 {
    let u2 = u * u;
    1.0 + u2 * (1.0 / 6.0 + u2 * (3.0 / 40.0 + u2 * (15.0 / 336.0)))
}

/// The arc standing in for the edge from a curve point with derivative `vel` along
/// chord `v`: the arc that leaves along the **curve's own tangent** and passes through
/// the far end.
///
/// `max_curvature` is the tightest arc the *caller* can actually sweep
/// ([`FlattenTolerance::max_arc_curvature`]); anything tighter comes back straight.
/// That cap is a parameter rather than a constant because it depends on the brush, and
/// it is what keeps [`within`] honest: the flattener prices an edge as whatever this
/// returns, and the renderer sweeps whatever this returns, so the two cannot disagree
/// about which primitive the budget was spent on.
///
/// Fitting from the start tangent rather than from both is what makes it cheap: with
/// `t̂` the unit tangent and `n̂` its left normal, an arc leaving along `t̂` reaches
/// chord `v` iff `κ = 2 (v·n̂)/|v|²`, and `sin(θ/2) = κ|v|/2` falls straight out of the
/// same identity. No angle is ever formed, so nothing here needs a transcendental
/// beyond the series above.
///
/// The end tangent is then *not* pinned to the curve's tangent there, so consecutive
/// arcs still meet with a small kink — but it is second order in the turn where the
/// chord's was first order, which is the whole point.
pub fn fit_arc(vel: Vec2, v: Vec2, max_curvature: f32) -> Arc {
    let chord = v.length();
    if chord < 1e-5 {
        // No chord names no direction. Length 0 makes the edge a point, which is what
        // the caller will do with it either way.
        return Arc {
            dir: Vec2::new(1.0, 0.0),
            curvature: 0.0,
            length: chord,
        };
    }
    let straight = Arc {
        dir: v / chord,
        curvature: 0.0,
        length: chord,
    };
    let speed = vel.length();
    if speed < 1e-12 {
        // A stationary derivative names no tangent (`turn` declines the same case):
        // there is nothing to bend towards.
        return straight;
    }
    let t = vel / speed;
    let n = Vec2::new(-t.y, t.x);
    let curvature = 2.0 * v.dot(n) / (chord * chord);
    if !curvature.is_finite() || curvature == 0.0 {
        return straight;
    }
    // Half the turn, as its sine — the arc is `chord = 2 sin(θ/2)/κ` by construction.
    let u = 0.5 * curvature * chord;
    if u.abs() > MAX_HALF_TURN_SIN || v.dot(t) <= 0.0 || curvature.abs() > max_curvature {
        return straight;
    }
    if arc_sagitta(curvature, chord * asin_over_x(u)) < MIN_SAGITTA {
        return straight;
    }
    Arc {
        dir: t,
        curvature,
        length: chord * asin_over_x(u),
    }
}

/// How far an arc bows away from its own chord: `|R|(1 − cos(θ/2))`, and 0 for a
/// straight edge. What a bounding box has to add, since the two endpoints no longer
/// bound the edge between them.
pub fn arc_sagitta(curvature: f32, length: f32) -> f32 {
    if curvature == 0.0 {
        return 0.0;
    }
    versin_small(0.5 * curvature * length) / curvature.abs()
}

/// Where an arc leaving `start` along `dir` with signed `curvature` has got to after
/// `s` of arc length, and the unit tangent it is travelling along there. The straight
/// case is the exact limit and is taken as one, so a `curvature == 0` edge is stepped
/// by plain addition.
pub fn arc_at(start: Vec2, dir: Vec2, curvature: f32, s: f32) -> (Vec2, Vec2) {
    if curvature == 0.0 {
        return (start + dir * s, dir);
    }
    let turn = curvature * s;
    let sn = sin_small(turn);
    let vs = versin_small(turn);
    let perp = Vec2::new(-dir.y, dir.x);
    (
        start + dir * (sn / curvature) + perp * (vs / curvature),
        dir * (1.0 - vs) + perp * sn,
    )
}

/// Distance from `p` to the arc leaving `start` — radially, which is the true distance
/// for a point near the arc and all [`within`] needs it to be.
fn point_arc_distance(p: Vec2, start: Vec2, arc: &Arc) -> f32 {
    if arc.curvature == 0.0 {
        return point_segment_distance(p, start, start + arc.dir * arc.length);
    }
    let perp = Vec2::new(-arc.dir.y, arc.dir.x);
    let r = 1.0 / arc.curvature;
    let centre = start + perp * r;
    ((p - centre).length() - r.abs()).abs()
}

// ---------------------------------------------------------------------------
// Flattening: control points → intermediate samples
// ---------------------------------------------------------------------------

/// How many cubic spans the curve through `control_points` has.
///
/// The clamped end condition is expressed by *repeating* each end control point
/// `degree` times in the conceptual control sequence (the clamped knot view, [`crate::spline`]),
/// which pins the curve to them. Those repeats are spans too, so `m` control
/// points give `m + 1` spans rather than the `m - 1` an interpolating spline would
/// — the two extra sit at the ends, each covering the sixth of the first (last)
/// leg that the clamp bends through. Fewer than two control points is not a curve:
/// one is a click, zero is nothing.
pub fn span_count(control_points: usize) -> usize {
    match control_points {
        0 | 1 => 0,
        m => m + 1,
    }
}

/// How many spans `frozen` frozen control points settle, out of a path of `total`.
///
/// A span reads at most two control points past its own index, so span `k` is final
/// once control points `0..=k+1` are — hence `frozen - 1`. Split out from
/// [`PathFitter::frozen_spans`] because a *received* stroke has the same question to
/// answer without the fitter that produced it: a peer knows which of its control
/// points are settled (everything the sender has stopped resending,
/// §17.5) and needs the same incremental repaint from them.
pub fn frozen_spans_for(frozen: usize, total: usize) -> usize {
    frozen.saturating_sub(1).min(span_count(total))
}

/// The curve point at the **end** of span `k` — where span `k + 1` picks up. `k`
/// past the last span gives the stroke's own end point.
///
/// One Bézier conversion and one evaluation, with no subdivision at all, which is
/// what lets a caller walk spans back from the live end of a stroke measuring chords
/// without paying for the polyline (see `gpu::stroke::safe_frozen`).
pub fn span_end(knots: &[ControlPoint], k: usize) -> Vec2 {
    match span_count(knots.len()) {
        // Fewer than two control points is not a curve: one is a click, zero nothing.
        0 => knots.first().map_or(Vec2::ZERO, |k| k.pos),
        last => span(knots, k.min(last - 1)).eval(1.0).pos,
    }
}

/// Expand `knots` into a polyline, subdividing only where the error budget
/// requires it (§6.2).
pub fn flatten(knots: &[ControlPoint], tol: FlattenTolerance) -> Vec<IntermediateSample> {
    flatten_spans(knots, 0..span_count(knots.len()), 0.0, tol)
}

/// [`flatten`] restricted to `spans`, with the arc-length accumulator starting at
/// `dist0`.
///
/// The polyline starts at the first span's own start knot, so adjacent ranges
/// share exactly one point and their segments (consecutive pairs) tile the stroke
/// with no gap and no overlap — the shape an incremental renderer wants, together
/// with [`PathFitter::frozen_spans`].
pub fn flatten_spans(
    knots: &[ControlPoint],
    spans: Range<usize>,
    dist0: f32,
    tol: FlattenTolerance,
) -> Vec<IntermediateSample> {
    if knots.is_empty() {
        return Vec::new();
    }
    let last_span = span_count(knots.len()); // one past the last valid span index
    let spans = spans.start.min(last_span)..spans.end.min(last_span);
    if spans.is_empty() {
        // A lone control point (a click): the path is that one point, no direction.
        let k = knots[spans.start.min(knots.len() - 1)];
        return vec![IntermediateSample {
            pos: k.pos,
            vel: Vec2::ZERO,
            pressure: k.pressure,
            tilt: k.tilt,
            time: k.time,
            dist: dist0,
        }];
    }

    let mut out = Vec::with_capacity(spans.len() * 4);
    let first = span(knots, spans.start);
    let mut start = first.eval(0.0);
    start.dist = dist0;
    out.push(start);
    for i in spans {
        let sp = span(knots, i);
        // The span's own start sample: same position as the last emitted point
        // (both are the shared knot, bit-for-bit), but with *this* span's
        // derivative, so the error test compares like with like.
        let mut a = sp.eval(0.0);
        a.dist = out.last().expect("start sample").dist;
        let ends = (
            End { u: 0.0, s: a },
            End {
                u: 1.0,
                s: sp.eval(1.0),
            },
        );
        subdivide(&sp, ends.0, ends.1, MAX_SUBDIVISION_DEPTH, tol, &mut out);
    }
    out
}

/// One end of a candidate segment: a curve parameter and the sample there.
#[derive(Copy, Clone)]
struct End {
    u: f32,
    s: IntermediateSample,
}

/// Emit the polyline for `sp` between two already-evaluated ends. `a` is the last
/// sample in `out`; only `b` and whatever a split produces are appended, so the
/// recursion emits in curve order.
fn subdivide(
    sp: &Span,
    a: End,
    b: End,
    depth: u32,
    tol: FlattenTolerance,
    out: &mut Vec<IntermediateSample>,
) {
    let m = End {
        u: 0.5 * (a.u + b.u),
        s: sp.eval(0.5 * (a.u + b.u)),
    };
    if depth == 0 || within(&a.s, &m.s, &b.s, tol) {
        emit(out, b.s);
        return;
    }
    subdivide(sp, a, m, depth - 1, tol, out);
    subdivide(sp, m, b, depth - 1, tol, out);
}

/// Append `s`, giving it the arc length accumulated along the polyline.
fn emit(out: &mut Vec<IntermediateSample>, mut s: IntermediateSample) {
    let prev = *out.last().expect("the start sample is emitted first");
    s.dist = prev.dist + (s.pos - prev.pos).length();
    out.push(s);
}

/// Is the edge `s0 → s1` an acceptable stand-in for the curve between them? `sm` is
/// the curve at the parametric midpoint.
///
/// The positional test is against the **arc** that edge will be swept as
/// ([`fit_arc`]), not against the chord between the ends. That is not a relaxation of
/// the budget — the number is unchanged, and it is still "max distance between a
/// segment and the curve it replaces". It is the budget finally being spent on the
/// geometry that gets drawn: an arc's error is second order in the turn where a
/// chord's is first order, so the same allowance buys a substantially longer edge.
///
/// The `angle` bound is deliberately left where it was, and with the positional test
/// no longer binding on gentle curves it is usually what does bind now. It earns that:
/// it is the bound that keeps a single midpoint sample honest (a symmetric wiggle can
/// sit on any one sample, but it cannot hide from the end tangents), and it is what
/// holds the *footprint's* orientation still enough — the shape angle is one value per
/// segment on both paths, and the dynamics loop bakes its reservoir in one frame per
/// segment (§6.6).
fn within(
    s0: &IntermediateSample,
    sm: &IntermediateSample,
    s1: &IntermediateSample,
    tol: FlattenTolerance,
) -> bool {
    let v = s1.pos - s0.pos;
    if v.length() > tol.max_len {
        return false;
    }
    let arc = fit_arc(s0.vel, v, tol.max_arc_curvature);
    if point_arc_distance(sm.pos, s0.pos, &arc) > tol.position {
        return false;
    }
    if turn(s0.vel, s1.vel) > tol.angle {
        return false;
    }
    let attr = (s1.pressure - s0.pressure)
        .abs()
        .max((s1.tilt - s0.tilt).length());
    attr <= tol.attribute
}

/// The unsigned angle between two derivatives; 0 where either is stationary and
/// the direction is undefined.
fn turn(a: Vec2, b: Vec2) -> f32 {
    if a.length_squared() < 1e-12 || b.length_squared() < 1e-12 {
        return 0.0;
    }
    (a.x * b.y - a.y * b.x).atan2(a.dot(b)).abs()
}

/// One cubic span of the path, in Bézier form: position *and* every pen attribute,
/// since both are B-splines over the same control polygon (see [`span`]).
struct Span {
    b: [ControlPoint; 4],
}

impl Span {
    /// The curve at `u ∈ [0, 1]`: position, its derivative, and the attributes —
    /// all the Bernstein form of the same four Bézier control points, so an
    /// attribute is read exactly where the curve is rather than lerped across the
    /// span.
    fn eval(&self, u: f32) -> IntermediateSample {
        let v = 1.0 - u;
        let at = blend(
            &self.b,
            [v * v * v, 3.0 * v * v * u, 3.0 * v * u * u, u * u * u],
        );
        let [b0, b1, b2, b3] = [self.b[0].pos, self.b[1].pos, self.b[2].pos, self.b[3].pos];
        let vel = (b1 - b0) * (3.0 * v * v) + (b2 - b1) * (6.0 * v * u) + (b3 - b2) * (3.0 * u * u);
        IntermediateSample {
            pos: at.pos,
            vel,
            pressure: at.pressure,
            tilt: at.tilt,
            time: at.time,
            dist: 0.0,
        }
    }
}

/// Build span `k` of the clamped cubic B-spline through `knots`
/// (requires `k < span_count(knots.len())`).
///
/// Every span of a *uniform* cubic B-spline is the same fixed combination of the
/// four control points supporting it, so the conversion to Bézier form — which is
/// what adaptive sampling wants, for the closed-form derivative — is one constant
/// 4×4 matrix and no knot-spacing arithmetic at all:
///
/// ```text
/// b0 = (Q0 + 4Q1 +  Q2) / 6      b2 = ( Q1 + 2Q2) / 3
/// b1 = (     2Q1 +  Q2) / 3      b3 = ( Q1 + 4Q2 + Q3) / 6
/// ```
///
/// The clamp at the two ends is not a special case here but a consequence of the
/// control sequence [`CubicBSpline`] fits against, in which each end control point
/// appears `degree` times ([`knot_row`]). Repeating `Q0` collapses `b0`, `b1` and
/// `b2` onto it, which is exactly what pins the curve to the first control point
/// and starts it heading down the first leg.
///
/// The attribute channels are B-splines over the same polygon and the same
/// parameterization ([`PathFitter::fit_channels`]), so the identical conversion
/// carries them: one `blend` per Bézier point does position and attributes at once.
fn span(knots: &[ControlPoint], k: usize) -> Span {
    let m = knots.len();
    let q: [ControlPoint; 4] = std::array::from_fn(|a| knots[knot_row(k + a, m)]);
    const SIXTH: f32 = 1.0 / 6.0;
    const THIRD: f32 = 1.0 / 3.0;
    Span {
        b: [
            blend(&q, [SIXTH, 4.0 * SIXTH, SIXTH, 0.0]),
            blend(&q, [0.0, 2.0 * THIRD, THIRD, 0.0]),
            blend(&q, [0.0, THIRD, 2.0 * THIRD, 0.0]),
            blend(&q, [0.0, SIXTH, 4.0 * SIXTH, SIXTH]),
        ],
    }
}

/// The control point backing index `i` of the conceptual clamped sequence, in which
/// the first and last each appear `degree` (= 3) times. This is [`crate::spline`]'s knot
/// view, reproduced here so a stored path can be evaluated without a fitter.
fn knot_row(i: usize, m: usize) -> usize {
    i.saturating_sub(2).min(m - 1)
}

/// A weighted combination of four control points, applied to every field alike.
fn blend(q: &[ControlPoint; 4], w: [f32; 4]) -> ControlPoint {
    let mut out = ControlPoint {
        pos: Vec2::ZERO,
        pressure: 0.0,
        tilt: Vec2::ZERO,
        time: 0.0,
    };
    for (p, w) in q.iter().zip(w) {
        out.pos += p.pos * w;
        out.pressure += p.pressure * w;
        out.tilt += p.tilt * w;
        out.time += p.time * w;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
    /// `arc`, so `m_step`'s normal equations go NaN — and they are then singular at
    /// every ridge, which its solve reports with `unreachable!` because for finite
    /// input it genuinely cannot happen.
    #[test]
    fn a_non_finite_report_is_dropped_rather_than_fitted() {
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
        let poisons: [Poison; 5] = [
            ("pos.x", |mut s| {
                s.pos.x = f32::NAN;
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
        // **Known weakness.** 25 control points for a 48px staircase: the fit is
        // tracing the quantization rather than smoothing it, because the arc-length
        // guess is not the curve's own parameterization and the residual that
        // mismatch leaves reads as error the growth rule tries to buy away. What is
        // still right is the *shape* — the curve splits the corners rather than
        // following them, which is what the error bound below checks.
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
        assert!(
            coarse * 2 <= fine,
            "2px grid declared: {coarse} control points against {fine} — not smoothed"
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
