//! Stroke path fitting and adaptive flattening (DESIGN.md §6.2).
//!
//! Three representations, deliberately distinct:
//!
//! - [`InputSample`] — one raw pointer report, as it arrived. High frequency,
//!   jittery, and never stored: it exists only between the pointer event and the
//!   fitter.
//! - [`ControlPoint`] — a knot of the fitted stroke curve. This is what a stroke
//!   *is* once captured ([`StrokeRecord::path`](crate::document::StrokeRecord)):
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
use serde::{Deserialize, Serialize};

use crate::command::InputSample;
use crate::geom::Vec2;
use crate::spline::CubicBSpline;

/// Control points solved for at the live end of the stroke. Everything behind them
/// is frozen; the pinned endpoint sits inside the window on top of these.
///
/// This is *the* accuracy/stability trade, and small is the point. A short window
/// means the settled stroke cannot move under the pointer — which is what stops a
/// live stroke wobbling along its whole length — and it caps the solve at a handful
/// of unknowns however long the stroke gets. Too short and the curve cannot round a
/// corner before the corner is committed.
const FREE_CONTROL_POINTS: usize = 3;

/// What a control point has to earn, in **mean** squared canvas px of error over
/// the samples in the window, before the fit will take one on.
///
/// Every sample is fitted both ways — as the polygon stands, and with one more
/// control point — and the larger fit is adopted only if it buys at least this much.
/// Adopting it is also what *freezes* one, since the window is a fixed size, so this
/// single number decides both how detailed the curve is and how promptly the stroke
/// settles behind the pointer.
/// Measured on the six recorded strokes, worst *live* error and control-point
/// count, once the parameterization was corrected (see [`arc_profile`]):
///
/// | price | C        | hairpin  | loop     | spiral   | big-C    | fast     |
/// |-------|----------|----------|----------|----------|----------|----------|
/// | 0.03  | 0.9px 37 | 1.4px 27 | 1.4px 34 | 2.4px 91 | 1.4px 68 | 0.7px 28 |
/// | 0.06  | 1.2px 25 | 1.5px 23 | 2.3px 25 | 2.2px 56 | 2.2px 52 | 0.8px 18 |
/// | 0.12  | 2.5px 19 | 1.5px 18 | 2.8px 18 | 4.2px 23 | 3.0px 18 | 1.2px 16 |
///
/// The floor is set by the input's own quantization rather than by taste: priced
/// below the jitter, the fit buys control points to *trace* a pixel staircase
/// instead of smoothing through it.
pub const KNOT_COST: f32 = 0.06;

/// Distance (canvas px) after which the polygon gains a control point regardless of
/// error.
///
/// Not about accuracy: a dead-straight stroke is fitted perfectly by a handful of
/// control points however long it runs, so on error alone it would never gain one,
/// never freeze one, and never let the renderer retire any of it.
pub const KNOT_SPACING: f32 = 64.0;

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

type ChannelCtrl = OMatrix<f32, Dyn, Const<CHANNELS>>;
type GeomCtrl = OMatrix<f32, Dyn, Const<2>>;

/// A control point of the fitted stroke curve — the stored form of a path.
///
/// Distinct from [`InputSample`] on purpose: an input sample is one *pointer
/// report* (raw, jittery, high frequency, discarded once fitted); a control point
/// is one coefficient of the fitted curve (stable, saved to the file and sent to
/// peers). It is a **cubic B-spline** control point, so the curve is pulled
/// towards it rather than through it — only the first and last are on the curve,
/// which the clamped end condition pins them to.
///
/// `time` is seconds since the stroke started rather than an absolute clock —
/// that is what velocity and timelapse want (DESIGN.md §8), and it halves the
/// field.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlPoint {
    pub pos: Vec2,
    pub pressure: f32,
    pub tilt: Vec2,
    pub time: f32,
}

impl ControlPoint {
    /// A full-pressure knot at `pos` (mouse input, or tests).
    pub fn at(pos: Vec2) -> Self {
        Self {
            pos,
            pressure: 1.0,
            tilt: Vec2::ZERO,
            time: 0.0,
        }
    }
}

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
    /// polyline — the distance axis that the load drain, the colour-dynamics
    /// noise, and the tool reservoir are parameterized by (DESIGN.md §6.2).
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
/// window* at the live end: exactly [`FREE_CONTROL_POINTS`] of them are solved for,
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
/// * **The work per sample is constant.** The system is
///   `FREE_CONTROL_POINTS × 2` unknowns however long the stroke is, and only the
///   samples that can reach those rows take part.
///
/// Both ends are pinned to the samples they belong to — the clamped end condition
/// makes the first and last control points the curve's endpoints, and least squares
/// does not otherwise hold them there. They are pinned as *constraints* of the solve
/// (held rows), not written over its result, so the rest solves around them.
pub struct PathFitter {
    /// Every accepted report, with the distance along the stroke that parameterizes
    /// it. Kept whole — a few hundred is nothing — while the solve only ever looks
    /// at the tail (see `first_live`).
    pts: Vec<Accepted>,
    /// First sample that can still reach a control point being solved for.
    first_live: usize,
    /// Control points: geometry, and the pen channels riding the same knots.
    geom: GeomCtrl,
    attr: ChannelCtrl,
    /// Arc length through the accepted samples (canvas px).
    arc: f32,
    /// Arc profile of the frozen spans, reused across updates (see [`arc_profile`]).
    settled_profile: Vec<f32>,
    /// Arc at which the polygon last gained a control point.
    grown_at: f32,
    smoothing: f32,
    knot_cost: f32,
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
        Self::with_params(KNOT_COST)
    }

    /// A fitter with an explicit price per control point (see [`KNOT_COST`]).
    pub fn with_params(knot_cost: f32) -> Self {
        Self {
            pts: Vec::new(),
            first_live: 0,
            geom: GeomCtrl::zeros_generic(Dyn(0), Const::<2>),
            attr: ChannelCtrl::zeros_generic(Dyn(0), Const::<CHANNELS>),
            arc: 0.0,
            settled_profile: Vec::new(),
            grown_at: 0.0,
            smoothing: SMOOTHING,
            knot_cost,
            t0: 0.0,
            finished: false,
        }
    }

    /// Feed one pointer report. Ignored once the stroke is [`finish`](Self::finish)ed.
    pub fn push(&mut self, s: InputSample) {
        if self.finished {
            return;
        }
        match self.pts.last() {
            None => self.t0 = s.time,
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
        let earns_it =
            err_as_is - err_grown > self.knot_cost || self.arc - self.grown_at > KNOT_SPACING;
        if earns_it {
            self.grown_at = self.arc;
            self.adopt(grown);
        } else {
            self.adopt(as_is);
        }
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
    /// The trailing [`FREE_CONTROL_POINTS`] may still move; everything before them
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
        (0..self.geom.nrows())
            .map(|j| ControlPoint {
                pos: Vec2::new(self.geom[(j, 0)], self.geom[(j, 1)]),
                // Clamped to the range a pen can report. The channels are solved the
                // same way the geometry is, and a control point the data barely
                // reaches is held only by the ridge, so it can overshoot the values
                // it was fitted from. For pressure that is a radius, which
                // `generate_segments` multiplies the brush by with no upper bound.
                // Clamping the control values bounds the *curve* and not just the
                // polygon: B-spline bases are non-negative and sum to one, so every
                // evaluated value is a convex combination of them.
                pressure: self.attr[(j, 0)].clamp(0.0, 1.0),
                tilt: clamp_tilt(Vec2::new(self.attr[(j, 1)], self.attr[(j, 2)])),
                time: self.attr[(j, 3)],
            })
            .collect()
    }

    /// How many leading spans of [`path`](Self::path) are settled — their geometry,
    /// and so their flattening, can never change however the stroke continues.
    ///
    /// A span of a clamped cubic B-spline reads at most two control points past its
    /// own index (see [`span_count`]), so span `k` is final once control points
    /// `0..=k+1` are frozen: `f` frozen control points settle `f - 1` spans. This is
    /// the hook for incremental repaint (DESIGN.md §6.2) — render
    /// `0..frozen_spans()` once and re-render only the short tail after it.
    pub fn frozen_spans(&self) -> usize {
        let all = span_count(self.geom.nrows());
        if self.finished {
            all
        } else {
            self.frozen().saturating_sub(1).min(all)
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
        set_row(&mut attr, m - 1, last.channels);

        let spline: CubicBSpline<2> =
            CubicBSpline::from_control_points(geom.clone()).expect("at least two control points");
        let spans = spline.num_spans() as f32;
        let total = self.arc.max(1e-6);
        let profile = arc_profile(&spline, &self.settled_profile);
        let param = |a: f32| param_at(&profile, spans, a / total);

        // A cubic B-spline's basis is local, so a sample sitting under the frozen
        // prefix cannot influence any row still being solved.
        let frozen = m.saturating_sub(FREE_CONTROL_POINTS + 1).max(1);
        let cutoff = frozen as f32 - 2.0;
        let mut lo = self.first_live.min(self.pts.len() - 1);
        while lo + 1 < self.pts.len() && param(self.pts[lo].arc) < cutoff {
            lo += 1;
        }
        let live = &self.pts[lo..];
        let pos: Vec<[f32; 2]> = live.iter().map(|s| [s.pos.x, s.pos.y]).collect();

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
        let ts: Vec<f32> = live.iter().map(|s| param(s.arc)).collect();
        let geom = spline.fit_channels(&ts, &pos, frozen, 1, &geom, self.smoothing);

        // The pen channels ride the same knots at the same parameters, so they are
        // the same solve with a different payload — unsmoothed, since a pressure ramp
        // is not a shape and has no curvature to penalize.
        let vals: Vec<[f32; CHANNELS]> = live.iter().map(|s| s.channels).collect();
        let attr = spline.fit_channels(&ts, &vals, frozen, 0, &attr, 0.0);
        Fit {
            geom,
            attr,
            lo,
            profile,
        }
    }

    /// Mean squared distance from the samples at and after `lo` to `fit`'s curve.
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
        let spline = CubicBSpline::from_control_points(fit.geom.clone())
            .expect("at least two control points");
        let spans = spline.num_spans() as f32;
        let total = self.arc.max(1e-6);
        let live = &self.pts[lo.min(self.pts.len() - 1)..];
        if live.is_empty() {
            return 0.0;
        }
        // Scored at exactly the parameters the solve minimizes at. If the two use
        // different maps the growth rule reads one quantity while the solve improves
        // another, and it stops firing where the fit is actually poor — measured at
        // 4-15px on recorded strokes against 0.6-1.6px when they agree. Consistency
        // between the two matters more than accuracy in either.
        let profile = arc_profile(&spline, &self.settled_profile);
        let sum: f32 = live
            .iter()
            .map(|s| {
                let c = spline.evaluate(param_at(&profile, spans, s.arc / total));
                (c[0] - s.pos.x).powi(2) + (c[1] - s.pos.y).powi(2)
            })
            .sum();
        sum / live.len() as f32
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

/// Samples per span used to measure a curve's own arc length (see [`arc_profile`]).
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
fn arc_profile(curve: &CubicBSpline<2>, settled: &[f32]) -> Vec<f32> {
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
fn param_at(profile: &[f32], spans: f32, f: f32) -> f32 {
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
fn clamp_tilt(t: Vec2) -> Vec2 {
    let len = t.length();
    if len > 1.0 { t / len } else { t }
}

/// Fit a whole sample sequence in one call — the batch form of [`PathFitter`],
/// used by replay and tests. Identical output to feeding the same samples one at
/// a time and finishing.
pub fn fit(samples: &[InputSample]) -> Vec<ControlPoint> {
    let mut f = PathFitter::new();
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

/// Expand `knots` into a polyline, subdividing only where the error budget
/// requires it (DESIGN.md §6.2).
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

/// Is the straight segment `s0 → s1` an acceptable stand-in for the curve between
/// them? `sm` is the curve at the parametric midpoint.
fn within(
    s0: &IntermediateSample,
    sm: &IntermediateSample,
    s1: &IntermediateSample,
    tol: FlattenTolerance,
) -> bool {
    if (s1.pos - s0.pos).length() > tol.max_len {
        return false;
    }
    if point_segment_distance(sm.pos, s0.pos, s1.pos) > tol.position {
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
            let reference: CubicBSpline<2> = CubicBSpline::from_control_points(rows).unwrap();

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
    /// the samples it passes through. It used to take an explicit cap. It is now
    /// structural: a control point is only taken on if it *measurably* reduces the
    /// error, and one the data cannot see does not.
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

    /// While a stroke is live, exactly [`FREE_CONTROL_POINTS`] of the polygon are
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
        // test that `SAGITTA_TOLERANCE` sits above the input's own quantization: set
        // it below and the zigzag reads as curvature and gets traced.
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
