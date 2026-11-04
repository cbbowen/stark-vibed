//! What the geometry tests share: builders, the recorded strokes, the one measure of
//! how far points stray from a polyline, and the case count every property runs at.
//!
//! At the crate root rather than under [`path`](crate::path) because the towed tip and
//! the assist lean on it too.

use crate::InputSample;
use crate::path::{IntermediateSample, PathFitter, point_segment_distance};
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;

/// One pointer report at a position, with everything else at rest.
pub(crate) fn sample(x: f32, y: f32) -> InputSample {
    InputSample::at(Vec2::new(x, y))
}

/// One full-pressure control point at a position.
pub(crate) fn knot(x: f32, y: f32) -> ControlPoint {
    ControlPoint::at(Vec2::new(x, y))
}

/// The worst distance (canvas px) from any of `points` to the polyline `poly` — to
/// its one vertex, for a click's.
pub(crate) fn max_deviation(
    points: impl IntoIterator<Item = Vec2>,
    poly: &[IntermediateSample],
) -> f32 {
    points
        .into_iter()
        .map(|p| match poly {
            [only] => (p - only.pos).length(),
            _ => poly
                .windows(2)
                .map(|w| point_segment_distance(p, w[0].pos, w[1].pos))
                .fold(f32::INFINITY, f32::min),
        })
        .fold(0.0, f32::max)
}

/// Feed `samples` one at a time, snapshotting the path and its frozen span count
/// after every push; the fitter comes back finished.
pub(crate) fn fit_incrementally(
    samples: &[InputSample],
) -> (PathFitter, Vec<(Vec<ControlPoint>, usize)>) {
    let mut f = PathFitter::new();
    let mut snaps = Vec::new();
    for s in samples {
        f.push(*s);
        snaps.push((f.path(), f.frozen_spans()));
    }
    f.finish();
    (f, snaps)
}

/// Every recorded stroke in `stark_testdata`, by name.
pub(crate) const RECORDED: [(&str, &[[f32; 2]]); 6] = [
    ("hairpin", stark_testdata::HAIRPIN_STROKE),
    ("loop", stark_testdata::LOOP_STROKE),
    ("C", stark_testdata::C_STROKE),
    ("spiral", stark_testdata::SPIRAL_STROKE),
    ("big-C", stark_testdata::BIG_C_STROKE),
    ("fast", stark_testdata::FAST_STROKE),
];

/// Recorded positions with pen channels laid over them, so a channel mapped onto
/// another shows. Integer arithmetic only: the input has to be the same bits on
/// every platform (`fingerprint`).
pub(crate) fn with_pen(pts: &[[f32; 2]]) -> Vec<InputSample> {
    pts.iter()
        .enumerate()
        .map(|(i, &[x, y])| {
            // A triangle wave of `period` reports, in `[0, 1]`.
            let tri = |period: usize| {
                let half = period as f32 / 2.0;
                1.0 - ((i % period) as f32 - half).abs() / half
            };
            InputSample {
                pos: Vec2::new(x, y),
                pressure: 0.2 + 0.6 * tri(64),
                tilt: Vec2::new(0.5 * tri(50) - 0.25, 0.3 - 0.4 * tri(38)),
                // Off zero, so the fitter's re-basing onto the stroke's clock shows.
                time: 1000.0 + i as f64 / 240.0,
            }
        })
        .collect()
}

/// `samples` as a hover and a stroke: the first quarter with the pressure lifted
/// off, to seed as a run-up, and the reports pressed after it.
pub(crate) fn split_runup(samples: &[InputSample]) -> (Vec<InputSample>, &[InputSample]) {
    let (hover, stroke) = samples.split_at(samples.len() / 4);
    let hover = hover
        .iter()
        .map(|s| InputSample {
            pressure: 0.0,
            ..*s
        })
        .collect();
    (hover, stroke)
}

/// The configuration every geometry property runs under: a fixed, modest case
/// count, so the suite costs the same every run.
pub(crate) fn cases() -> proptest::test_runner::Config {
    proptest::test_runner::Config::with_cases(64)
}

/// Every field of every control point as bits: what "the same path" means to a
/// test that claims the same pixels.
pub(crate) fn bits(path: &[ControlPoint]) -> Vec<[u32; 6]> {
    path.iter()
        .map(|k| [k.pos.x, k.pos.y, k.pressure, k.tilt.x, k.tilt.y, k.time].map(f32::to_bits))
        .collect()
}
