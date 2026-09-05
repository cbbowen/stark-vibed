//! The [`Sweep`]/[`Segment`] fixtures the stroke modules' tests are written in (§6.2).
//!
//! Here rather than beside each suite because these build *this* module's central
//! type, and what a fixture leaves neutral is an argument rather than a value: a ramp
//! or a stretch set in a fixture is a second variable in whatever the test is actually
//! measuring, and every caller holds the same ones still for the same reason. Written
//! out once per suite, that argument was three copies to keep in step by hand — and a
//! new field on [`Sweep`] was five edits the compiler could only report one at a time.
//!
//! The rates are the one thing a fixture cannot leave neutral by omission: [`sweep`]
//! hands back the geometry half, which has nowhere to put a rate that would imply one
//! had been consulted, and [`seg`] adds [`Paint::default`]'s zeros — so a test that
//! wants a rate has to set one.
//!
//! The record fixtures and the walks over a whole stroke's segments are here for the
//! same reason, one level up: the taper's tests and the arc and budget tests are two
//! files now, and both measure the very same straight tapered stroke.

use stark_model::document::{BrushDynamics, BrushEffect, BrushParams, LayerId, StrokeRecord};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;

use super::{Paint, Segment, Stretch, Sweep, generate_segments_in};
use crate::gpu::stroke::StrokeSpans;
use crate::gpu::stroke::budget::flatten_tolerance;

/// A straight sweep of `length` from `start` along `dir`, at arc length `dist`.
pub(in crate::gpu::stroke) fn sweep(
    start: Vec2,
    dir: Vec2,
    length: f32,
    radius: f32,
    dist: f32,
) -> Sweep {
    Sweep {
        start,
        dir,
        curvature: 0.0,
        radius,
        // A tip that holds still, so the frame the shader unrolls is the one the
        // caller is measuring against and nothing else.
        radius_ramp: 0.0,
        // A tip that reaches its own radius: what is under test is the arithmetic over
        // sweeps, not how wide any one shape is.
        reach: radius,
        length,
        orient: 0.0,
        // An unstretched tip, for the reason the ramp is zero: an extent drawn out
        // along an axis is a second variable in everything built from these.
        stretch: Stretch::NONE,
        dist,
    }
}

/// The same sweep named by its **endpoints**, at the arc clock's origin.
///
/// The frame is derived rather than given, so a caller whose subject is only where the
/// tip went — the region measurements, which combine boxes — says that and nothing
/// about direction or travel. Derived from [`sweep`] rather than written beside it,
/// because a field added there has to reach this shape too.
pub(in crate::gpu::stroke) fn sweep_between(start: Vec2, end: Vec2, radius: f32) -> Sweep {
    let v = end - start;
    let length = v.length();
    let dir = if length > 0.0 {
        v / length
    } else {
        Vec2::new(1.0, 0.0)
    };
    sweep(start, dir, length, radius, 0.0)
}

/// [`sweep`] as a whole segment — what the chunker, the tile walks and the plan take.
pub(in crate::gpu::stroke) fn seg(
    start: Vec2,
    dir: Vec2,
    length: f32,
    radius: f32,
    dist: f32,
) -> Segment {
    Segment {
        sweep: sweep(start, dir, length, radius, dist),
        paint: Paint::default(),
    }
}

/// [`sweep_between`] as a whole segment.
pub(in crate::gpu::stroke) fn seg_between(start: Vec2, end: Vec2, radius: f32) -> Segment {
    Segment {
        sweep: sweep_between(start, end, radius),
        paint: Paint::default(),
    }
}

/// `n` straight segments of `len` each, running +x from the origin — a stroke cut the
/// way the flattener would cut a steady drag.
pub(in crate::gpu::stroke) fn run(n: usize, len: f32, radius: f32) -> Vec<Segment> {
    (0..n)
        .map(|i| {
            let d = i as f32 * len;
            seg(Vec2::new(d, 0.0), Vec2::new(1.0, 0.0), len, radius, d)
        })
        .collect()
}

/// A stroke through `pts` with `brush`, as a path of plain full-pressure knots.
pub(in crate::gpu::stroke) fn record(brush: BrushParams, pts: &[Vec2]) -> StrokeRecord {
    StrokeRecord {
        layer: LayerId::ROOT,
        brush,
        path: pts.iter().map(|p| ControlPoint::at(*p)).collect(),
        seed: 0,
        start: 0.0,
        translation: stark_model::geom::IVec2::ZERO,
    }
}

/// A wet brush that manipulates paint, so a stroke of it takes the dynamics loop.
pub(in crate::gpu::stroke) fn smearing(radius: f32) -> BrushParams {
    BrushParams {
        size: radius,
        effect: BrushEffect::wet_with(
            [0.0; 3],
            BrushDynamics {
                lift: 0.8,
                deposit: 0.8,
                ..BrushDynamics::default()
            },
        ),
        ..BrushParams::default()
    }
}

/// A straight stroke `len` px long with a tapered brush of `radius`.
pub(in crate::gpu::stroke) fn tapered_record(
    radius: f32,
    start: f32,
    end: f32,
    len: f32,
) -> StrokeRecord {
    // Enough control points that the curve has spans to freeze part of, and
    // straight so arc length is the chord and the taper zones are easy to reason
    // about.
    let path: Vec<stark_model::path::ControlPoint> = (0..=12)
        .map(|i| stark_model::path::ControlPoint::at(Vec2::new(i as f32 / 12.0 * len, 0.0)))
        .collect();
    StrokeRecord {
        layer: stark_model::document::LayerId::ROOT,
        brush: BrushParams {
            size: radius,
            drain: 0.0,
            start_taper_length: start,
            end_taper_length: end,
            ..BrushParams::default()
        },
        path,
        seed: 0,
        start: 0.0,
        translation: stark_model::geom::IVec2::ZERO,
    }
}

/// Every segment of a stroke, as [`Segment`]s — what the chunker and the tile
/// walks are handed.
pub(in crate::gpu::stroke) fn whole_segments(rec: &StrokeRecord) -> Vec<Segment> {
    generate_segments_in(rec, flatten_tolerance(&rec.brush), StrokeSpans::whole(rec)).0
}

/// The same, as bare [`Sweep`]s.
///
/// Almost everything below is a claim about *geometry* — where the tip went and how
/// wide it was — so it is asked of the half that carries geometry. A test that
/// wanted a paint rate would have to say so by using [`whole_segments`], which is
/// the point of the split being visible here too.
pub(in crate::gpu::stroke) fn whole(rec: &StrokeRecord) -> Vec<Sweep> {
    sweeps(whole_segments(rec))
}

pub(in crate::gpu::stroke) fn sweeps(segs: Vec<Segment>) -> Vec<Sweep> {
    segs.into_iter().map(|s| s.sweep).collect()
}

/// **The property the ramp exists to have** (§6.2, [`Sweep::radius_ramp`]): consecutive
/// segments agree on the tip at the knot they share, so the stroke's outline has
/// no C⁰ break to alias — at any brush size, and however coarsely the taper is
/// cut.
///
/// Stated as an agreement between neighbours rather than as a bound on a step,
/// because that is the difference between the two designs. A per-segment radius
/// can only ever make the step *small*; a ramp makes it zero, since both sides
/// evaluate the same pen and the same taper at the same arc length.
///
/// The tolerance is not slack for the taper: within a flattened edge the two are
/// the identical float expression and agree to the bit. It covers the edge
/// *boundaries*, where the arc length a segment measures its taper at is
/// accumulated along the polyline and the two sides can differ by an ulp or two of
/// a large number.
pub(in crate::gpu::stroke) fn assert_outline_is_continuous(segs: &[Sweep]) {
    for (i, w) in segs.windows(2).enumerate() {
        let (before, after) = (w[0].tip_at(1.0), w[1].tip_at(0.0));
        let tol = 1e-3 * before.max(after).max(1.0);
        assert!(
            (before - after).abs() <= tol,
            "segment {i} ends at radius {before} and {} begins at {after} — \
             the outline has a step in it",
            i + 1,
        );
    }
}
