//! **Flattening**: control points → intermediate samples (§6.2).
//!
//! Owns the two types the other two speak in — the budget a caller declares
//! ([`FlattenTolerance`]) and the samples that come out ([`IntermediateSample`]).

use super::arc::{fit_arc, point_arc_distance};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;
use std::ops::Range;

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
    // Asked of the knot view rather than restated as `m + 1`. The two agreed, and
    // `span_form_matches_the_fitted_spline` is what said so — a test comparing two
    // spellings of one number, which is the shape that only ever *reports* a drift
    // (§13). `SplineIndex::new` is the same "fewer than two is not a curve" this arm
    // used to spell for itself.
    crate::spline::SplineIndex::new(control_points).map_or(0, |ix| ix.num_spans())
}

/// How many spans `frozen` frozen control points settle, out of a path of `total`.
///
/// A span reads at most two control points past its own index, so span `k` is final
/// once control points `0..=k+1` are — hence `frozen - 1`. Split out from
/// [`PathFitter::frozen_spans`](super::PathFitter::frozen_spans) because a *received* stroke has the same question to
/// answer without the fitter that produced it: a peer knows which of its control
/// points are settled (everything the sender has stopped resending,
/// §17.5) and needs the same incremental repaint from them.
///
/// **Strictly fewer than [`span_count`]`(total)`, for every `frozen <= total`**, and
/// something downstream depends on it. A frozen head's range ends here, and the stroke
/// renderer captures cross-piece brush state only for a range that does *not* reach the
/// end of the stroke (`gpu::stroke::Resume`) — so a head range that could equal the
/// span count would silently stop carrying the brush forward, and the tail would resume
/// from a state one range stale. The bound holds because `frozen - 1 <= total - 1` and
/// `span_count(total) = total + 1` for a curve: the `min` is never the term that binds.
/// For `total < 2` there is no curve, `span_count` is 0, and no head range exists.
pub fn frozen_spans_for(frozen: usize, total: usize) -> usize {
    let out = frozen.saturating_sub(1).min(span_count(total));
    debug_assert!(
        out < span_count(total) || span_count(total) == 0,
        "a frozen head reaching the stroke's end would stop carrying the brush",
    );
    out
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

/// The curve point at parameter `t`, in span units, clamped to the domain — the
/// general position [`span_end`] answers at whole spans. What a caller
/// measuring against a stroke's marker uses
/// ([`StrokeRecord::start`](stark_model::document::StrokeRecord::start)): the
/// marker names a place mid-span, where the deposit begins. A non-finite `t`
/// reads the curve's own start — records arrive from files and peers.
pub fn point_at(knots: &[ControlPoint], t: f32) -> Vec2 {
    match span_count(knots.len()) {
        0 => knots.first().map_or(Vec2::ZERO, |k| k.pos),
        last => {
            let t = if t.is_finite() {
                t.clamp(0.0, last as f32)
            } else {
                0.0
            };
            let k = (t.floor() as usize).min(last - 1);
            span(knots, k).eval(t - k as f32).pos
        }
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
/// with [`PathFitter::frozen_spans`](super::PathFitter::frozen_spans).
pub fn flatten_spans(
    knots: &[ControlPoint],
    spans: Range<usize>,
    dist0: f32,
    tol: FlattenTolerance,
) -> Vec<IntermediateSample> {
    flatten_spans_from(knots, 0.0, spans, dist0, tol)
}

/// [`flatten_spans`] with everything before curve parameter `from` left out —
/// the stroke's marker
/// ([`StrokeRecord::start`](stark_model::document::StrokeRecord::start)),
/// honoured here, in the one place both render paths flatten through, so a live
/// tail, a commit, a replay and a peer all trim identically (§6.2).
///
/// `from` at or before the range's start is the identity — bit for bit, since
/// it then takes the very code path the untrimmed call takes — which is what
/// keeps every record with `start == 0`, including every record from before the
/// marker existed, on exactly the floats it always flattened to. A range
/// entirely behind the marker comes back **empty**: its spans are real spans of
/// the record's curve, but no travel of the *stroke*, and the segment builder
/// treats an empty polyline as exactly that.
///
/// The accumulator therefore reads `dist0` — 0, for a whole-stroke render — at
/// the marker itself, which re-bases every distance-parameterized quantity (the
/// tapers, the `drain` falloff, the color-dynamics noise, the reservoir
/// cadence) onto the stroke's own travel: the run-up is not painted, and it is
/// not aged over either.
///
/// `from` is clamped to the domain and a non-finite `from` reads 0 — records
/// arrive from files and peers, and flattening one must not panic.
pub fn flatten_spans_from(
    knots: &[ControlPoint],
    from: f32,
    spans: Range<usize>,
    dist0: f32,
    tol: FlattenTolerance,
) -> Vec<IntermediateSample> {
    if knots.is_empty() {
        return Vec::new();
    }
    let last_span = span_count(knots.len()); // one past the last valid span index
    let spans = spans.start.min(last_span)..spans.end.min(last_span);
    let from = if from.is_finite() {
        from.clamp(0.0, last_span as f32)
    } else {
        0.0
    };
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
    if from >= spans.end as f32 {
        // The whole range is behind the marker: spans of the curve, none of
        // the stroke.
        return Vec::new();
    }
    // Where the polyline begins: the range's own start, or the marker mid-span.
    let (first_span, u0) = if from > spans.start as f32 {
        let k = (from.floor() as usize).min(last_span - 1);
        (k, from - k as f32)
    } else {
        (spans.start, 0.0)
    };

    let mut out = Vec::with_capacity((spans.end - first_span) * 4);
    let first = span(knots, first_span);
    let mut start = first.eval(u0);
    start.dist = dist0;
    out.push(start);
    for i in first_span..spans.end {
        let sp = span(knots, i);
        let u = if i == first_span { u0 } else { 0.0 };
        // The span's own start sample: same position as the last emitted point
        // (both are the shared knot — or the marker — bit-for-bit), but with
        // *this* span's derivative, so the error test compares like with like.
        let mut a = sp.eval(u);
        a.dist = out.last().expect("start sample").dist;
        let ends = (
            End { u, s: a },
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
pub(super) struct Span {
    b: [ControlPoint; 4],
}

impl Span {
    /// The curve at `u ∈ [0, 1]`: position, its derivative, and the attributes —
    /// all the Bernstein form of the same four Bézier control points, so an
    /// attribute is read exactly where the curve is rather than lerped across the
    /// span.
    pub(super) fn eval(&self, u: f32) -> IntermediateSample {
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
/// appears `degree` times ([`SplineIndex`](crate::spline::SplineIndex)'s knot view).
/// Repeating `Q0` collapses `b0`, `b1` and
/// `b2` onto it, which is exactly what pins the curve to the first control point
/// and starts it heading down the first leg.
///
/// The attribute channels are B-splines over the same polygon and the same
/// parameterization ([`PathFitter::fit_channels`]), so the identical conversion
/// carries them: one `blend` per Bézier point does position and attributes at once.
pub(super) fn span(knots: &[ControlPoint], k: usize) -> Span {
    // The clamped knot view, asked once for the four rows rather than four times. It is
    // `crate::spline`'s and not a copy of it: this file evaluates a *stored* path
    // without the fitter that produced it, which is a reason to have the evaluator here
    // and never was a reason to spell the degree twice.
    //
    // The `expect` is unreachable from every caller. `span_count` is 0 below two
    // control points, and each of `span_end`, `point_at` and `flatten_spans_from`
    // returns on that before it asks for a span at all.
    let view = crate::spline::SplineIndex::new(knots.len())
        .expect("a span implies at least two control points");
    let q: [ControlPoint; 4] = std::array::from_fn(|a| knots[view.knot_row(k + a)]);
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
