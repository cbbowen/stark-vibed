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
    /// Arc length from the stroke start (canvas px), measured along the **arcs** the
    /// emitted edges stand for ([`fit_arc`]) — the distance axis that the load drain,
    /// the color-dynamics noise, and the tool reservoir are parameterized by (§6.2).
    ///
    /// Along the arcs and not along the chords between them, because the arc is what
    /// gets swept: the segment builder steps a piece of one out to `dist + length`
    /// and the next edge has to pick up exactly there, or the taper's radius has a
    /// step at every curved joint and a bleed firing's window is scanned twice
    /// (§6.2).
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
    /// Max **arc** length of one segment (canvas px); `INFINITY` for "no opinion". The
    /// renderer sets this from quantities that vary with distance travelled but are
    /// applied per segment rather than per fragment — the `drain` falloff, the dynamics
    /// loop's reservoir cadence (see `gpu::stroke::flatten_tolerance`).
    ///
    /// The arc rather than the chord under it, for the same reason the positional
    /// budget is spent on the arc: that is the primitive the caller will sweep, and a
    /// cap on the chord under-prices the travel the renderer then has to fit in a
    /// region. Unlike the error bounds this is a hard requirement, so it is met by
    /// construction and not merely aimed at — [`flatten`] cuts each span into enough
    /// pieces that the error-driven subdivision inside one can always reach it.
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

/// Max bisections of a single span piece: 2^10 edges, the ceiling on what any one
/// piece can cost however pathological its knots.
///
/// **A backstop on the *error* bounds, which is what it was for.** `position`, `angle`
/// and `attribute` are allowances a pathological span can ask unboundedly much of, so
/// their pursuit has to stop somewhere and a picture drawn a shade off budget is the
/// right thing to hand back. [`FlattenTolerance::max_len`] is not one of those: it is a
/// requirement the caller states, and would be silently overridden here whenever a
/// span wanted more than 1,024 edges of it. [`span_pieces`] is what keeps the two
/// apart — the requirement decides how many pieces a span is cut into, this decides
/// how hard the error bounds are chased inside one.
const MAX_SUBDIVISION_DEPTH: u32 = 10;

/// Edges one exhausted piece costs — what [`MAX_SUBDIVISION_DEPTH`] affords.
const EDGES_PER_PIECE: f32 = (1u32 << MAX_SUBDIVISION_DEPTH) as f32;

/// Ceiling on the pre-split of one span, so a `max_len` no renderer budget can produce
/// cannot ask for an unbounded polyline. 64 pieces is 65,536 edges, which at the
/// renderer's own floor (`gpu::stroke::MIN_SEGMENT_LEN`, 0.5 px) is a 32,768 px span —
/// past anything a fitted stroke holds, and the only reason it is a number at all is
/// that [`flatten`] is reachable with a tolerance nobody's budget built.
const MAX_SPAN_PIECES: u32 = 64;

/// How many uniform parametric pieces a span is cut into *before* the error-driven
/// subdivision runs inside each — the depth floor [`FlattenTolerance::max_len`]'s hard
/// requirement needs, priced so that it costs nothing where there is no requirement.
///
/// **1 whenever [`MAX_SUBDIVISION_DEPTH`] can already reach the cap unaided**, so every
/// stroke the flattener was already cutting correctly comes out bit for bit: the
/// pre-split engages in exactly the regime that was quietly missing the cap and nowhere
/// else.
///
/// The count is taken from a bound on the span's *speed* rather than from its chord,
/// which is what makes the cap met rather than aimed at. The Bernstein weights of a
/// cubic's derivative sum to 1, so `|B′| ≤ 3·max leg` everywhere on the span; a piece
/// of parametric width `h` therefore travels at most `3·max leg·h`, and
/// [`MAX_SUBDIVISION_DEPTH`] halvings inside it leave every edge under `max_len`
/// whatever the parameterization does with the arc length. The subdivision still stops
/// the moment an edge is within budget, so a piece that did not need the depth does not
/// spend it.
fn span_pieces(sp: &Span, max_len: f32) -> u32 {
    let speed = 3.0
        * sp.b
            .windows(2)
            .map(|w| (w[1].pos - w[0].pos).length())
            .fold(0.0, f32::max);
    let afforded = max_len * EDGES_PER_PIECE;
    if speed > afforded {
        ((speed / afforded).ceil() as u32).clamp(1, MAX_SPAN_PIECES)
    } else {
        // `INFINITY` (no opinion), a non-positive cap, and a NaN in either term all
        // land here, which is why the comparison is spelled the way round that lets
        // them: there is no requirement to build a floor under, and the span keeps the
        // arithmetic the error bounds alone give it.
        1
    }
}

/// The curve point at the **end** of span `k` — where span `k + 1` picks up. `k`
/// past the last span gives the stroke's own end point.
///
/// One Bézier conversion and one evaluation, with no subdivision at all, which is
/// what lets a caller walk spans back from the live end of a stroke measuring chords
/// without paying for the polyline (see `gpu::stroke::safe_frozen`).
pub fn span_end(knots: &[ControlPoint], k: usize) -> Vec2 {
    // Fewer than two control points is not a curve: one is a click, zero nothing.
    let Ok(ix) = crate::spline::SplineIndex::new(knots.len()) else {
        return knots.first().map_or(Vec2::ZERO, |k| k.pos);
    };
    let last = ix.num_spans();
    span(ix, knots, k.min(last - 1)).eval(1.0).pos
}

/// The curve point at parameter `t`, in span units, clamped to the domain — the
/// general position [`span_end`] answers at whole spans. What a caller
/// measuring against a stroke's marker uses
/// ([`StrokeRecord::start`](stark_model::document::StrokeRecord::start)): the
/// marker names a place mid-span, where the deposit begins. A non-finite `t`
/// reads the curve's own start — records arrive from files and peers.
pub fn point_at(knots: &[ControlPoint], t: f32) -> Vec2 {
    let Ok(ix) = crate::spline::SplineIndex::new(knots.len()) else {
        return knots.first().map_or(Vec2::ZERO, |k| k.pos);
    };
    let last = ix.num_spans();
    let t = if t.is_finite() {
        t.clamp(0.0, last as f32)
    } else {
        0.0
    };
    let k = (t.floor() as usize).min(last - 1);
    span(ix, knots, k).eval(t - k as f32).pos
}

/// Expand `knots` into a polyline, subdividing only where the error budget
/// requires it (§6.2) — and, where [`FlattenTolerance::max_len`] asks a span for more
/// edges than chasing the error bounds inside one can afford, cutting that span into
/// as many pieces as the requirement takes ([`span_pieces`]).
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
    // Fewer than two control points is not a curve: a lone knot is a click, and the
    // path is that one point with no direction. Asked *before* the range is looked at,
    // because it is a fact about the polygon and the branch below is a fact about the
    // request — conflated, `flatten_spans(knots, 3..3, ..)` on a real curve came back
    // as a stray point at `knots[3]` instead of nothing, which is not a tiling of the
    // stroke (see this function's contract).
    let Ok(ix) = crate::spline::SplineIndex::new(knots.len()) else {
        let k = knots[0];
        return vec![IntermediateSample {
            pos: k.pos,
            vel: Vec2::ZERO,
            pressure: k.pressure,
            tilt: k.tilt,
            time: k.time,
            dist: dist0,
        }];
    };
    let last_span = ix.num_spans(); // one past the last valid span index
    let spans = spans.start.min(last_span)..spans.end.min(last_span);
    let from = if from.is_finite() {
        from.clamp(0.0, last_span as f32)
    } else {
        0.0
    };
    if spans.is_empty() {
        // No spans of a real curve were asked for. Adjacent ranges share exactly one
        // point and their segments tile the stroke, so an empty range contributes
        // none of it — the same answer the "entirely behind the marker" branch below
        // gives, for the same reason.
        return Vec::new();
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
    let first = span(ix, knots, first_span);
    let mut start = first.eval(u0);
    start.dist = dist0;
    out.push(start);
    for i in first_span..spans.end {
        let sp = span(ix, knots, i);
        let u = if i == first_span { u0 } else { 0.0 };
        // The span's own start sample: same position as the last emitted point
        // (both are the shared knot — or the marker — bit-for-bit), but with
        // *this* span's derivative, so the error test compares like with like.
        let mut a = sp.eval(u);
        a.dist = out.last().expect("start sample").dist;
        let pieces = span_pieces(&sp, tol.max_len);
        let mut lo = End { u, s: a };
        for p in 1..=pieces {
            // The last piece ends at `1.0` exactly — and a single piece *is* the span,
            // so a span with no requirement to build a floor under takes the very
            // arithmetic it always took, ends and all.
            let hu = if p == pieces {
                1.0
            } else {
                u + (1.0 - u) * (p as f32 / pieces as f32)
            };
            let hi = End {
                u: hu,
                s: sp.eval(hu),
            };
            subdivide(&sp, lo, hi, MAX_SUBDIVISION_DEPTH, tol, &mut out);
            // The next piece picks up from what was emitted, not from `hi` — the two
            // are the same sample but for the accumulator, which only `emit` fills in.
            lo = End {
                u: hu,
                s: *out.last().expect("a piece emits its own far end"),
            };
        }
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
        emit(out, b.s, tol);
        return;
    }
    subdivide(sp, a, m, depth - 1, tol, out);
    subdivide(sp, m, b, depth - 1, tol, out);
}

/// Append `s`, giving it the arc length accumulated along the edges emitted so far.
///
/// The edge's length is [`fit_arc`]'s, from the **same call** the segment builder
/// makes for that edge — the previous sample's own derivative, this sample's position,
/// the caller's curvature cap. Not a second expression that ought to agree with it:
/// `dist` is what the renderer steps *along* the arc from, so the two would be
/// measuring the same edge with different rulers, and the disagreement — the arc over
/// the chord, up to 0.7% — lands as a step in the taper's radius at every curved joint
/// and as a sliver of path two consecutive segments both scan for bleed firings
/// (§6.2).
fn emit(out: &mut Vec<IntermediateSample>, mut s: IntermediateSample, tol: FlattenTolerance) {
    let prev = *out.last().expect("the start sample is emitted first");
    let arc = fit_arc(prev.vel, s.pos - prev.pos, tol.max_arc_curvature);
    s.dist = prev.dist + arc.length;
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
/// The **length** cap is priced on that same arc, for the plainer reason that it is
/// the length the renderer will travel: a chord under the cap can carry an arc over
/// it, and every consumer of the cap — the reservoir cadence, the region fit — is
/// asking about the travel and not about the shortcut across it (§6.2).
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
    let arc = fit_arc(s0.vel, v, tol.max_arc_curvature);
    if arc.length > tol.max_len {
        return false;
    }
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
    a.angle_to(b).abs()
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
fn span(view: crate::spline::SplineIndex, knots: &[ControlPoint], k: usize) -> Span {
    // The clamped knot view is the caller's, built once. It is `crate::spline`'s and
    // not a copy of it: this file evaluates a *stored* path without the fitter that
    // produced it, which is a reason to have the evaluator here and never was a reason
    // to spell the degree twice.
    //
    // Taken as a parameter rather than rebuilt here, which is what removes the `expect`
    // this carried. "Fewer than two control points is not a curve" is the `Option` the
    // three callers already branch on before asking for a span at all — so the index's
    // existence *is* that branch, rather than a claim about it restated per span. It
    // was rebuilt `1 + spans` times per flatten.
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
                let ix = crate::spline::SplineIndex::new(knots.len()).expect("a curve");
                let sp = span(ix, &knots, k);
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
        let ix = crate::spline::SplineIndex::new(knots.len()).expect("a curve");
        let head = span(ix, &knots, 0).eval(0.25);
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
            let ix = crate::spline::SplineIndex::new(knots.len()).expect("a curve");
            let sp = span(ix, &knots, i);
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

    /// The length cap is a **requirement**, and one span's error-driven subdivision
    /// ceiling ([`MAX_SUBDIVISION_DEPTH`]) is not allowed to quietly override it
    /// (§6.2).
    ///
    /// Two knots is a ~1000 px middle span; at 0.05 px that is ~13,000 edges of a span
    /// halving alone can afford 1,024. The regime is not exotic — the renderer's own
    /// floor is `MIN_SEGMENT_LEN` = 0.5 px, which the same span overruns at 512 px —
    /// so the cap has to be met by construction rather than aimed at, which is what
    /// the pre-split in [`flatten_spans_from`] is for.
    #[test]
    fn flatten_honours_a_cap_finer_than_one_spans_subdivision_ceiling() {
        let knots = [knot(0.0, 0.0), knot(1000.0, 0.0)];
        let tol = FlattenTolerance {
            max_len: 0.05,
            ..FLATTEN_TOLERANCE
        };
        let poly = flatten(&knots, tol);
        let worst = poly
            .windows(2)
            .map(|w| w[1].dist - w[0].dist)
            .fold(0.0, f32::max);
        assert!(
            worst <= tol.max_len + 1e-4,
            "the longest of {} edges travelled {worst}px against a {}px cap",
            poly.len() - 1,
            tol.max_len,
        );
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

    /// **The accumulator is the arcs**, edge for edge (§6.2): `dist` advances by
    /// exactly what [`fit_arc`] reports for the edge — the same call, with the same
    /// three arguments, that the segment builder makes for it — and never goes
    /// backwards.
    ///
    /// Stated over a family and not on one curve, because a *chord* accumulator
    /// satisfies every check a straight stroke can pose and most of what a gentle one
    /// can. What separates the two is a curved edge, so the family varies the bend, the
    /// pen ramp and both caps — including a curvature cap tight enough to send edges
    /// back to being chords, where the two agree again and have to.
    ///
    /// Equality is exact rather than toleranced, and that is the claim: this is not two
    /// derivations of one number that ought to land near each other, it is the same
    /// expression evaluated in the same order.
    #[test]
    fn dist_accumulates_the_arcs_the_edges_stand_for() {
        let mut bent = 0usize;
        let mut over_chord = 0usize;
        for bend in [0.0f32, 3.0, 20.0, 90.0, 260.0] {
            for tol in [
                FLATTEN_TOLERANCE,
                FLATTEN_TOLERANCE.relaxed(6.0),
                FlattenTolerance {
                    max_len: 7.0,
                    ..FLATTEN_TOLERANCE
                },
                FlattenTolerance {
                    max_arc_curvature: 0.004,
                    ..FLATTEN_TOLERANCE
                },
                FlattenTolerance {
                    max_len: 0.4,
                    ..FLATTEN_TOLERANCE.relaxed(3.0)
                },
            ] {
                let knots: Vec<ControlPoint> = (0..6)
                    .map(|i| {
                        let t = i as f32;
                        ControlPoint {
                            pressure: t / 5.0,
                            ..knot(t * 47.0, (t * 0.9).sin() * bend)
                        }
                    })
                    .collect();
                let poly = flatten(&knots, tol);
                assert_eq!(poly[0].dist, 0.0, "the accumulator starts at dist0");
                let (mut arcs, mut chords) = (0.0f32, 0.0f32);
                for w in poly.windows(2) {
                    let arc = fit_arc(w[0].vel, w[1].pos - w[0].pos, tol.max_arc_curvature);
                    arcs += arc.length;
                    chords += (w[1].pos - w[0].pos).length();
                    bent += usize::from(arc.curvature != 0.0);
                    assert_eq!(
                        w[1].dist,
                        w[0].dist + arc.length,
                        "bend {bend}: an edge advanced dist by something other than its arc",
                    );
                    assert!(
                        w[1].dist >= w[0].dist,
                        "bend {bend}: dist went backwards across an edge",
                    );
                }
                assert_eq!(
                    poly.last().expect("a polyline").dist,
                    arcs,
                    "bend {bend}: the final dist is not the sum of the emitted arcs",
                );
                over_chord += usize::from(arcs > chords);
            }
        }
        // The premise. Without a bent edge somewhere the family would be satisfied by
        // the chord accumulator this replaced, and would have stopped testing anything.
        assert!(bent > 0, "no configuration produced a curved edge");
        assert!(
            over_chord > 0,
            "no configuration measured longer along the arcs than along the chords",
        );
    }

    #[test]
    fn relaxing_the_budget_costs_fewer_samples() {
        let knots = [knot(0.0, 0.0), knot(60.0, 80.0), knot(160.0, 0.0)];
        let fine = flatten(&knots, FLATTEN_TOLERANCE).len();
        let coarse = flatten(&knots, FLATTEN_TOLERANCE.relaxed(8.0)).len();
        assert!(coarse < fine, "relaxed {coarse} vs fine {fine}");
    }
}
