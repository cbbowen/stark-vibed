//! **Flattening**: control points → intermediate samples (§6.2).
//!
//! Owns the two types the other two speak in — the budget a caller declares
//! ([`FlattenTolerance`]) and the samples that come out ([`IntermediateSample`]).

use super::arc::{fit_arc, point_arc_distance};
use super::span_count;
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
/// control sequence [`CubicBSpline`](crate::spline::CubicBSpline) fits against, in which each end control point
/// appears `degree` times ([`SplineIndex`](crate::spline::SplineIndex)'s knot view).
/// Repeating `Q0` collapses `b0`, `b1` and
/// `b2` onto it, which is exactly what pins the curve to the first control point
/// and starts it heading down the first leg.
///
/// The attribute channels are B-splines over the same polygon and the same
/// parameterization ([`SplineIndex::fit_channels`](crate::spline::SplineIndex::fit_channels)), so the identical conversion
/// carries them: one `blend` per Bézier point does position and attributes at once.
fn span(knots: &[ControlPoint], k: usize) -> Span {
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

#[cfg(test)]
mod tests {
    use super::super::arc::point_segment_distance;
    use super::*;
    use crate::spline::CubicBSpline;

    /// One control point at a position. See [`sample`]'s note on the copies.
    fn knot(x: f32, y: f32) -> ControlPoint {
        ControlPoint::at(Vec2::new(x, y))
    }

    /// The load-bearing link between the two halves of this module: the span form
    /// used to *render* a stored path must be the same curve [`CubicBSpline`](crate::spline::CubicBSpline) **fitted**.
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
