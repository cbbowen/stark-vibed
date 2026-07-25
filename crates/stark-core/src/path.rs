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
//! [`PathFitter`] streams the first into the second — low-pass smoothing to shed
//! pointer/pixel-quantization jitter, then simplification to knots — **append
//! only**: once a knot is committed nothing appended later can move it, so a
//! caller can render the settled part of a live stroke once instead of repainting
//! it on every pointer move (see [`PathFitter::frozen_spans`]).
//!
//! [`flatten`] expands knots through a centripetal Catmull–Rom spline into a
//! polyline, **adaptively**: it subdivides only where a straight segment would
//! exceed a bounded error in position, tangent direction, and pen attributes. A
//! long gentle stroke then costs a handful of segments where uniform arc-length
//! sampling cost hundreds, while a corner still gets the density it needs — the
//! tangent bound buys both, which is why [`IntermediateSample`] carries the
//! derivative rather than position alone.
//!
//! All math here is deterministic, preserving golden / replay / save-load
//! equivalence.

use std::collections::VecDeque;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::command::InputSample;
use crate::geom::Vec2;

/// Default simplification tolerance in canvas pixels. A 1-px-stepped diagonal sits
/// ~0.7px off its ideal line, so ~1px clears that staircase while keeping any
/// feature larger than a pixel — imperceptible smoothing at normal brush sizes.
pub const SIMPLIFY_TOLERANCE: f32 = 1.0;

/// Default smoothing radius in canvas pixels (see [`PathFitter`]). A drawn-slowly
/// diagonal arrives snapped to the device-pixel grid as a right/up staircase whose
/// ~1–2px corners survive simplification; averaging over a few pixels of path pulls
/// them back onto the line. Larger = smoother but rounds intentional detail more.
pub const SMOOTH_RADIUS: f32 = 4.0;

/// A knot of the fitted stroke curve — the stored form of a path.
///
/// Distinct from [`InputSample`] on purpose: an input sample is one *pointer
/// report* (raw, jittery, high frequency, discarded once fitted); a control point
/// is one *curve knot* (fitted, stable, saved to the file and sent to peers).
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

    /// Linear blend of two knots at `f ∈ [0, 1]`.
    fn lerp(self, other: Self, f: f32) -> Self {
        Self {
            pos: self.pos.lerp(other.pos, f),
            pressure: self.pressure + (other.pressure - self.pressure) * f,
            tilt: self.tilt.lerp(other.tilt, f),
            time: self.time + (other.time - self.time) * f,
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
/// Two stages, both incremental:
///
/// 1. **Smooth.** Each sample becomes a triangular-weighted average of its
///    neighbours within [`SMOOTH_RADIUS`] canvas px of *arc length* — distance
///    based, so it smooths the same amount however densely the pointer sampled. A
///    sample is *ripe* (its window complete) once the stroke has travelled a full
///    radius past it, so smoothing lags the pointer by that much and never has to
///    revise what it emitted.
/// 2. **Simplify.** An open window of smoothed points is kept after the last
///    committed knot; while any point in it strays more than the tolerance from
///    the chord *anchor → newest*, the farthest one becomes a knot and the window
///    restarts there. This is Ramer–Douglas–Peucker's test applied to a forward
///    window instead of the whole stroke, which is what makes it streaming — a
///    committed knot is never revisited.
///
/// That window invariant (every open point within tolerance of the chord) is also
/// why the provisional tail [`path`](Self::path) appends is a faithful preview: a
/// live stroke reaches the cursor without waiting for the fit to settle, and it
/// does so within the same tolerance as the rest of the path.
///
/// The stroke's first and last samples are kept exactly, so it starts and ends
/// where the pointer did.
#[derive(Debug)]
pub struct PathFitter {
    smooth_radius: f32,
    tolerance: f32,
    /// Raw samples still needed: those not yet ripe, plus the trailing radius of
    /// ripe ones that their windows still reach back for.
    raw: VecDeque<Raw>,
    /// How many leading `raw` entries have been smoothed and passed on.
    ripe: usize,
    /// Cumulative arc length of the newest raw sample (canvas px).
    arc: f32,
    /// Total samples smoothed so far (only the first is special).
    emitted: usize,
    /// Absolute time of the first sample; knot times are relative to it.
    t0: f64,
    /// Committed knots — final, whatever arrives next.
    knots: Vec<ControlPoint>,
    /// Smoothed points after the last committed knot, still undecided.
    open: Vec<ControlPoint>,
    finished: bool,
}

#[derive(Copy, Clone, Debug)]
struct Raw {
    s: InputSample,
    /// Arc length from the stroke start. `f32` from the start of the stroke, like
    /// every other distance here; only local differences are ever taken.
    arc: f32,
}

impl Default for PathFitter {
    fn default() -> Self {
        Self::new()
    }
}

impl PathFitter {
    pub fn new() -> Self {
        Self::with_params(SMOOTH_RADIUS, SIMPLIFY_TOLERANCE)
    }

    pub fn with_params(smooth_radius: f32, tolerance: f32) -> Self {
        Self {
            smooth_radius,
            tolerance,
            raw: VecDeque::new(),
            ripe: 0,
            arc: 0.0,
            emitted: 0,
            t0: 0.0,
            knots: Vec::new(),
            open: Vec::new(),
            finished: false,
        }
    }

    /// Feed one pointer report. Ignored once the stroke is [`finish`](Self::finish)ed.
    pub fn push(&mut self, s: InputSample) {
        if self.finished {
            return;
        }
        let arc = match self.raw.back() {
            Some(prev) => prev.arc + (s.pos - prev.s.pos).length(),
            None => {
                self.t0 = s.time;
                0.0
            }
        };
        self.raw.push_back(Raw { s, arc });
        self.arc = arc;
        self.ripen(false);
    }

    /// End the stroke: flush the samples still inside the smoothing window and
    /// commit the final knot, exactly where the pointer released.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.ripen(true);
        // Close the open window on its last point. Everything before it is within
        // tolerance of the anchor→last chord (the window invariant), which is
        // exactly what simplification would have discarded.
        if let Some(last) = self.open.last().copied() {
            self.knots.push(last);
            self.open.clear();
        }
    }

    /// The path fitted so far.
    ///
    /// While the stroke is live this is the committed knots plus one
    /// **provisional** knot at the newest sample, so a preview reaches the cursor;
    /// every knot before it is already final. After [`finish`](Self::finish) it is
    /// the whole fitted path.
    pub fn path(&self) -> Vec<ControlPoint> {
        let mut out = self.knots.clone();
        if self.finished {
            return out;
        }
        if let Some(newest) = self.raw.back() {
            let p = self.control_point(newest.s);
            if out
                .last()
                .is_none_or(|k| (k.pos - p.pos).length_squared() > 1e-12)
            {
                out.push(p);
            }
        }
        out
    }

    /// How many leading spans of [`path`](Self::path) are settled — their
    /// flattening can never change, however the stroke continues.
    ///
    /// A centripetal Catmull–Rom span is defined by four knots (`i-1 ..= i+2`), so
    /// span `i` settles as soon as knot `i+2` is committed: `k` committed knots
    /// settle `k - 2` spans, and finishing settles the rest. This is the hook for
    /// incremental repaint (DESIGN.md §6.2) — render `0..frozen_spans()` once and
    /// re-render only the short tail after it as the pointer moves.
    pub fn frozen_spans(&self) -> usize {
        if self.finished {
            self.knots.len().saturating_sub(1)
        } else {
            self.knots.len().saturating_sub(2)
        }
    }

    /// Smooth and pass on every raw sample whose window is complete. With
    /// `final_flush`, the samples still inside the window go too — their forward
    /// window truncated at the stroke's end, and the very last one kept exactly.
    fn ripen(&mut self, final_flush: bool) {
        while self.ripe < self.raw.len() {
            let i = self.ripe;
            if !final_flush && self.arc - self.raw[i].arc < self.smooth_radius {
                break;
            }
            // The endpoints are kept exactly so the stroke begins and ends where
            // the pointer did; everything between is the windowed average.
            let ends = self.emitted == 0 || (final_flush && i + 1 == self.raw.len());
            let cp = if ends || self.smooth_radius <= 0.0 {
                self.control_point(self.raw[i].s)
            } else {
                self.smoothed(i)
            };
            self.ripe += 1;
            self.emitted += 1;
            self.trim();
            self.feed(cp);
        }
    }

    /// The windowed average at raw index `i` (position and pressure; tilt and time
    /// are taken from the sample itself).
    fn smoothed(&self, i: usize) -> ControlPoint {
        let r = self.smooth_radius;
        let ai = self.raw[i].arc;
        let mut wsum = 0.0f32;
        let mut pos = Vec2::ZERO;
        let mut pressure = 0.0f32;
        // Walk outward in both directions until past the radius; `arc` is
        // monotonic, so we can stop early.
        let mut j = i as isize;
        while j >= 0 && ai - self.raw[j as usize].arc < r {
            let e = self.raw[j as usize];
            let w = 1.0 - (ai - e.arc) / r;
            wsum += w;
            pos += e.s.pos * w;
            pressure += e.s.pressure * w;
            j -= 1;
        }
        let mut k = i + 1;
        while k < self.raw.len() && self.raw[k].arc - ai < r {
            let e = self.raw[k];
            let w = 1.0 - (e.arc - ai) / r;
            wsum += w;
            pos += e.s.pos * w;
            pressure += e.s.pressure * w;
            k += 1;
        }
        ControlPoint {
            pos: pos * (1.0 / wsum),
            pressure: pressure / wsum,
            tilt: self.raw[i].s.tilt,
            time: self.rel_time(self.raw[i].s.time),
        }
    }

    /// Drop raw samples no future smoothing window can reach back to.
    fn trim(&mut self) {
        while self.ripe > 0 && self.raw.len() > 1 {
            // The next sample to ripen is either present or still to arrive, in
            // which case its arc is at least the newest one's.
            let next = self.raw.get(self.ripe).map_or(self.arc, |r| r.arc);
            if next - self.raw[0].arc <= self.smooth_radius {
                break;
            }
            self.raw.pop_front();
            self.ripe -= 1;
        }
    }

    /// Offer one smoothed point to the streaming simplifier.
    fn feed(&mut self, p: ControlPoint) {
        let Some(anchor) = self.knots.last().copied() else {
            // The stroke's first point anchors the path.
            self.knots.push(p);
            return;
        };
        self.open.push(p);
        // Split while the window strays from the anchor→newest chord. One split
        // can leave the remainder still straying (the chord moved), hence the loop.
        let mut anchor = anchor;
        loop {
            let last = *self.open.last().expect("just pushed");
            let interior = &self.open[..self.open.len() - 1];
            let mut max_d = 0.0f32;
            let mut idx = 0usize;
            for (i, q) in interior.iter().enumerate() {
                let d = point_segment_distance(q.pos, anchor.pos, last.pos);
                if d > max_d {
                    max_d = d;
                    idx = i;
                }
            }
            if max_d <= self.tolerance {
                break;
            }
            anchor = self.open[idx];
            self.knots.push(anchor);
            self.open.drain(..=idx);
        }
    }

    fn control_point(&self, s: InputSample) -> ControlPoint {
        ControlPoint {
            pos: s.pos,
            pressure: s.pressure,
            tilt: s.tilt,
            time: self.rel_time(s.time),
        }
    }

    fn rel_time(&self, t: f64) -> f32 {
        (t - self.t0) as f32
    }
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

/// Expand `knots` into a polyline, subdividing only where the error budget
/// requires it (DESIGN.md §6.2).
pub fn flatten(knots: &[ControlPoint], tol: FlattenTolerance) -> Vec<IntermediateSample> {
    flatten_spans(knots, 0..knots.len().saturating_sub(1), 0.0, tol)
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
    let last_span = knots.len() - 1; // one past the last valid span index
    let spans = spans.start.min(last_span)..spans.end.min(last_span);
    if spans.is_empty() {
        // A lone knot (a click): the path is that one point, with no direction.
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
        let ends = (End { u: 0.0, s: a }, End { u: 1.0, s: sp.eval(1.0) });
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

/// One cubic span of the path: the curve from knot `i` to knot `i + 1`.
struct Span {
    /// Cubic Bézier control points, equivalent to the centripetal Catmull–Rom
    /// through the four surrounding knots.
    b: [Vec2; 4],
    start: ControlPoint,
    end: ControlPoint,
}

impl Span {
    /// The curve at `u ∈ [0, 1]`: position and its derivative (Bernstein form),
    /// with the pen attributes interpolated linearly across the span.
    fn eval(&self, u: f32) -> IntermediateSample {
        let v = 1.0 - u;
        let [b0, b1, b2, b3] = self.b;
        let pos = b0 * (v * v * v) + b1 * (3.0 * v * v * u) + b2 * (3.0 * v * u * u) + b3 * (u * u * u);
        let vel = (b1 - b0) * (3.0 * v * v) + (b2 - b1) * (6.0 * v * u) + (b3 - b2) * (3.0 * u * u);
        let at = self.start.lerp(self.end, u);
        IntermediateSample {
            pos,
            vel,
            pressure: at.pressure,
            tilt: at.tilt,
            time: at.time,
            dist: 0.0,
        }
    }
}

/// Build span `i` of `knots` (requires `i + 1 < knots.len()`).
///
/// Centripetal Catmull–Rom (α = ½) in its exact cubic Bézier form: the same curve
/// the Barry–Goldman pyramid evaluates, but with the derivative in closed form —
/// which is what adaptive sampling needs — and cheaper per sample.
///
/// The two ends use a **one-sided** tangent (straight down the first/last chord)
/// rather than the usual duplicated phantom knot. Duplicating collapses that knot
/// spacing onto the epsilon floor, which drives the end tangent to nearly zero: a
/// stroke that crawls out of its own start, and a direction the sampler cannot
/// read. One-sided is well conditioned and leaves the curve's interior untouched.
fn span(knots: &[ControlPoint], i: usize) -> Span {
    let n = knots.len();
    let (p1, p2) = (knots[i], knots[i + 1]);
    let d1 = knot_delta(p1.pos, p2.pos);
    let v1 = (p2.pos - p1.pos) / d1;
    let m1 = if i == 0 {
        v1
    } else {
        let p0 = knots[i - 1].pos;
        let d0 = knot_delta(p0, p1.pos);
        (p1.pos - p0) / d0 - (p2.pos - p0) / (d0 + d1) + v1
    };
    let m2 = if i + 2 >= n {
        v1
    } else {
        let p3 = knots[i + 2].pos;
        let d2 = knot_delta(p2.pos, p3);
        (p3 - p2.pos) / d2 - (p3 - p1.pos) / (d1 + d2) + v1
    };
    Span {
        b: [
            p1.pos,
            p1.pos + m1 * (d1 / 3.0),
            p2.pos - m2 * (d1 / 3.0),
            p2.pos,
        ],
        start: p1,
        end: p2,
    }
}

/// Centripetal (α = ½) knot spacing between two control points, floored so
/// coincident knots cannot divide by zero.
fn knot_delta(a: Vec2, b: Vec2) -> f32 {
    (b - a).length().sqrt().max(1e-4)
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

    // ---- fitting -------------------------------------------------------

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
        assert!(
            fitted.len() <= 4,
            "staircase should collapse, got {} points",
            fitted.len()
        );
        assert_eq!(fitted.first().unwrap().pos, stair.first().unwrap().pos);
        assert_eq!(fitted.last().unwrap().pos, stair.last().unwrap().pos);
    }

    #[test]
    fn fit_flattens_coarse_staircase() {
        // A 2-px right / 2-px up staircase, e.g. a slow diagonal snapped to a 2px
        // device grid. Each corner sits ~1.4px off the diagonal — above the
        // simplification tolerance — so without smoothing the whole zigzag stays.
        let mut stair = vec![sample(0.0, 0.0)];
        for i in 0..12 {
            let b = (i * 2) as f32;
            stair.push(sample(b + 2.0, b)); // 2px right
            stair.push(sample(b + 2.0, b + 2.0)); // 2px up
        }
        let mut raw = PathFitter::with_params(0.0, SIMPLIFY_TOLERANCE);
        for s in &stair {
            raw.push(*s);
        }
        raw.finish();
        assert!(
            raw.path().len() > 6,
            "simplification alone keeps the staircase: {} pts",
            raw.path().len()
        );

        // Smoothing pulls the corners back onto the diagonal, so the fit collapses
        // to a near-straight line.
        let fitted = fit(&stair);
        assert!(
            fitted.len() <= 4,
            "smoothed staircase should collapse: {} pts",
            fitted.len()
        );
        // Endpoints are still exactly where the pointer pressed and released.
        assert_eq!(fitted.first().unwrap().pos, stair.first().unwrap().pos);
        assert_eq!(fitted.last().unwrap().pos, stair.last().unwrap().pos);
    }

    #[test]
    fn fit_keeps_real_corners() {
        // An L-shape: the corner must survive, well clear of the tolerance.
        let pts = [
            sample(0.0, 0.0),
            sample(10.0, 0.0),
            sample(20.0, 0.0),
            sample(20.0, 10.0),
            sample(20.0, 20.0),
        ];
        let fitted = fit(&pts);
        let nearest = fitted
            .iter()
            .map(|k| (k.pos - Vec2::new(20.0, 0.0)).length())
            .fold(f32::INFINITY, f32::min);
        assert!(nearest < SMOOTH_RADIUS, "corner lost: nearest knot {nearest}px");
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

    /// The Barry–Goldman evaluation this file used before the Bézier conversion,
    /// kept as the reference the new form must reproduce.
    fn barry_goldman(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, u: f32) -> Vec2 {
        let dt = |a: Vec2, b: Vec2| (b - a).length().powf(0.5).max(1e-4);
        let t0 = 0.0;
        let t1 = t0 + dt(p0, p1);
        let t2 = t1 + dt(p1, p2);
        let t3 = t2 + dt(p2, p3);
        let t = t1 + (t2 - t1) * u;
        let a1 = p0.lerp(p1, (t - t0) / (t1 - t0));
        let a2 = p1.lerp(p2, (t - t1) / (t2 - t1));
        let a3 = p2.lerp(p3, (t - t2) / (t3 - t2));
        let b1 = a1.lerp(a2, (t - t0) / (t2 - t0));
        let b2 = a2.lerp(a3, (t - t1) / (t3 - t1));
        b1.lerp(b2, (t - t1) / (t2 - t1))
    }

    #[test]
    fn bezier_form_matches_barry_goldman() {
        let knots = [
            knot(0.0, 0.0),
            knot(20.0, 35.0),
            knot(70.0, 30.0),
            knot(95.0, -10.0),
            knot(140.0, 5.0),
        ];
        // Interior spans only: the ends deliberately differ (one-sided tangents
        // instead of a duplicated phantom knot).
        for i in 1..knots.len() - 2 {
            let sp = span(&knots, i);
            for s in 0..=8 {
                let u = s as f32 / 8.0;
                let want = barry_goldman(
                    knots[i - 1].pos,
                    knots[i].pos,
                    knots[i + 1].pos,
                    knots[i + 2].pos,
                    u,
                );
                let got = sp.eval(u).pos;
                assert!(
                    (got - want).length() < 1e-3,
                    "span {i} at u={u}: {got:?} != {want:?}",
                );
            }
        }
    }

    #[test]
    fn end_tangents_point_along_the_stroke() {
        // The duplicated-knot construction drove these to ~0; one-sided tangents
        // make the curve leave and arrive at full speed, which is what the angle
        // bound reads.
        let knots = [knot(0.0, 0.0), knot(30.0, 0.0), knot(60.0, 20.0)];
        let head = span(&knots, 0).eval(0.0);
        assert!(head.vel.length() > 1.0, "start derivative {:?}", head.vel);
        assert!(head.vel.normalize().dot(Vec2::X) > 0.99);
        let tail = span(&knots, 1).eval(1.0);
        assert!(tail.vel.length() > 1.0, "end derivative {:?}", tail.vel);
    }

    #[test]
    fn flatten_straight_line_collapses_to_one_segment() {
        // The point of adaptive sampling: a straight run costs nothing, however
        // long it is. Uniform arc-length sampling spent 500 points on this.
        let knots = [knot(0.0, 0.0), knot(1000.0, 0.0)];
        let poly = flatten(&knots, FLATTEN_TOLERANCE);
        assert_eq!(poly.len(), 2, "got {} samples", poly.len());
        assert_eq!(poly[0].pos, Vec2::ZERO);
        assert_eq!(poly[1].pos, Vec2::new(1000.0, 0.0));
        assert!((poly[1].dist - 1000.0).abs() < 1e-3);
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
        let whole = flatten(&knots, FLATTEN_TOLERANCE);
        let head = flatten_spans(&knots, 0..2, 0.0, FLATTEN_TOLERANCE);
        let tail = flatten_spans(
            &knots,
            2..4,
            head.last().unwrap().dist,
            FLATTEN_TOLERANCE,
        );
        // The ranges share exactly one point, so the pieces concatenate into the
        // whole with no gap and no duplicated segment.
        let joined: Vec<IntermediateSample> =
            head.iter().chain(tail[1..].iter()).copied().collect();
        assert_eq!(joined, whole);
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


