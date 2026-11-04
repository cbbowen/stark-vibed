//! The **fit fingerprint**: `STARK_GOLDEN_EXACT` for the CPU half of a stroke (§6.2).
//! A refactor that changes a hash changes strokes; re-blessing it is a deliberate act.
//!
//! Every recorded stroke is fitted and then flattened, and every bit of the result is
//! hashed. The flatten half matters more: a record stores control points, so the fit
//! only has to agree with itself in-process, while flattening re-runs at every replay
//! and on every peer.

use super::{
    FLATTEN_TOLERANCE, FlattenTolerance, InputTolerance, IntermediateSample, PathFitter, fit_at,
    flatten_spans_from, span_count,
};
use crate::test_support::{RECORDED, bits, split_runup, with_pen};
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

/// Every recorded stroke fitted twice — whole, and after a hovering run-up — as
/// `(path, marker)`.
fn fitted() -> Vec<(Vec<ControlPoint>, f32)> {
    RECORDED
        .into_iter()
        .flat_map(|(_, pts)| {
            let samples = with_pen(pts);
            let whole = fit_at(samples.iter().copied(), InputTolerance::DEFAULT);

            let (hover, stroke) = split_runup(&samples);
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

fn update_bits(h: &mut blake3::Hasher, xs: impl IntoIterator<Item = u32>) {
    for x in xs {
        h.update(&x.to_le_bytes());
    }
}

fn update_f32s(h: &mut blake3::Hasher, xs: impl IntoIterator<Item = f32>) {
    update_bits(h, xs.into_iter().map(f32::to_bits));
}

fn update_len(h: &mut blake3::Hasher, n: usize) {
    h.update(&(n as u64).to_le_bytes());
}

fn hash_fit(fits: &[(Vec<ControlPoint>, f32)]) -> String {
    let mut h = blake3::Hasher::new();
    for (path, start) in fits {
        update_len(&mut h, path.len());
        // Every field of every control point, through the one statement of what "the
        // same path" means to a test.
        update_bits(&mut h, bits(path).into_iter().flatten());
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
