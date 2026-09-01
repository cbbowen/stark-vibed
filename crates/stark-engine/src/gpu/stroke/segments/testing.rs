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

use stark_model::document::{BrushDynamics, BrushEffect, BrushParams, LayerId, StrokeRecord};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;

use super::{Paint, Segment, Stretch, Sweep};

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
        ramp: 0.0,
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
        frame: stark_model::geom::IVec2::ZERO,
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
