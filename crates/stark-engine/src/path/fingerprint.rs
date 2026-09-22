//! The **fit fingerprint**: `STARK_GOLDEN_EXACT` for the CPU half of a stroke (§6.2).
//! A refactor that changes a hash changes strokes; re-blessing it is a deliberate act.
//!
//! Every recorded stroke is fitted and then flattened, and every bit of the result is
//! hashed. The flatten half matters more: a record stores control points, so the fit
//! only has to agree with itself in-process, while flattening re-runs at every replay
//! and on every peer.

use super::{
    DEFAULT_TOLERANCE, FLATTEN_TOLERANCE, FlattenTolerance, IntermediateSample, PathFitter,
    fit_with_tolerance, flatten_spans_from, span_count,
};
use crate::command::InputSample;
use stark_model::geom::Vec2;
use stark_model::path::ControlPoint;

/// blake3 of every fitted control point's channels and marker, in [`fitted`] order.
const FIT: &str = "47b4adfb6843d7b436dd4ec9000f4232fba00da62581d891f4bfef06349ce207";

/// blake3 of every flattened sample's fields, over [`BUDGETS`], in [`fitted`] order.
const FLATTEN: &str = "9b38c6fbf57c995d2689aa0b3660df6782c2ac89c3515bfb0a6401f6717c438e";

/// The shipped budget, and one with both renderer caps set — `max_len` pre-splits spans
/// and a finite `max_arc_curvature` sends tight edges back to chords, two paths the
/// shipped budget never takes.
const BUDGETS: [FlattenTolerance; 2] = [
    FLATTEN_TOLERANCE,
    FlattenTolerance {
        max_len: 6.0,
        max_arc_curvature: 0.02,
        ..FLATTEN_TOLERANCE
    },
];

/// The recorded positions with pen channels laid over them, so a channel mis-mapped
/// onto another changes the hash. Integer arithmetic only: the input has to be the
/// same bits on every platform the fingerprint runs on.
fn with_pen(pts: &[[f32; 2]]) -> Vec<InputSample> {
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
                // Off zero, so the fitter's re-basing onto the stroke's clock is exercised.
                time: 1000.0 + i as f64 / 240.0,
            }
        })
        .collect()
}

/// Every recorded stroke fitted twice — whole, and with its first quarter seeded as a
/// hovering run-up — as `(path, marker)`.
fn fitted() -> Vec<(Vec<ControlPoint>, f32)> {
    [
        stark_testdata::HAIRPIN_STROKE,
        stark_testdata::LOOP_STROKE,
        stark_testdata::C_STROKE,
        stark_testdata::SPIRAL_STROKE,
        stark_testdata::BIG_C_STROKE,
        stark_testdata::FAST_STROKE,
    ]
    .into_iter()
    .flat_map(|pts| {
        let samples = with_pen(pts);
        let whole = fit_with_tolerance(samples.iter().copied(), DEFAULT_TOLERANCE);

        let (hover, stroke) = samples.split_at(samples.len() / 4);
        let hover: Vec<InputSample> = hover
            .iter()
            .map(|s| InputSample {
                pressure: 0.0,
                ..*s
            })
            .collect();
        let mut f = PathFitter::new();
        f.seed_runup(&hover);
        for s in stroke {
            f.push(*s);
        }
        f.finish();

        [(whole, 0.0), (f.path(), f.start())]
    })
    .collect()
}

fn update_f32s(h: &mut blake3::Hasher, xs: impl IntoIterator<Item = f32>) {
    for x in xs {
        h.update(&x.to_bits().to_le_bytes());
    }
}

fn update_len(h: &mut blake3::Hasher, n: usize) {
    h.update(&(n as u64).to_le_bytes());
}

fn hash_fit(fits: &[(Vec<ControlPoint>, f32)]) -> String {
    let mut h = blake3::Hasher::new();
    for (path, start) in fits {
        update_len(&mut h, path.len());
        for k in path {
            update_f32s(
                &mut h,
                [k.pos.x, k.pos.y, k.pressure, k.tilt.x, k.tilt.y, k.time],
            );
        }
        update_f32s(&mut h, [*start]);
    }
    h.finalize().to_hex().to_string()
}

fn hash_flatten(fits: &[(Vec<ControlPoint>, f32)]) -> String {
    let mut h = blake3::Hasher::new();
    for (path, start) in fits {
        for tol in BUDGETS {
            // Through the marker, as a replay flattens a record.
            let poly: Vec<IntermediateSample> =
                flatten_spans_from(path, *start, 0..span_count(path.len()), 0.0, tol);
            update_len(&mut h, poly.len());
            for s in &poly {
                update_f32s(
                    &mut h,
                    [
                        s.pos.x, s.pos.y, s.vel.x, s.vel.y, s.pressure, s.tilt.x, s.tilt.y, s.time,
                        s.dist,
                    ],
                );
            }
        }
    }
    h.finalize().to_hex().to_string()
}

#[test]
fn the_recorded_strokes_fit_and_flatten_bit_for_bit() {
    let fits = fitted();
    // The premise: a run-up that left no marker would fingerprint nothing of it.
    assert!(
        fits.iter()
            .skip(1)
            .step_by(2)
            .all(|(_, start)| *start > 0.0),
        "a seeded run-up left the marker at the curve's start",
    );
    assert_eq!(
        (hash_fit(&fits).as_str(), hash_flatten(&fits).as_str()),
        (FIT, FLATTEN),
        "(fit, flatten) fingerprints moved: this change alters strokes",
    );
}
