//! CPU benchmarks for the stroke *geometry* pipeline: fitting raw pen reports into
//! a cubic B-spline, and flattening that spline into the polyline the renderer
//! sweeps (§6.2).
//!
//! Separate from `benches/stroke.rs` because these need no adapter and are
//! nanosecond-to-microsecond scale, so they get criterion's ordinary settings and
//! run anywhere — including a CI box with no GPU. They are also the only half of
//! the pipeline whose cost is comparable *between* machines.
//!
//! Recorded input, not synthetic: the fitter's behaviour turns on report spacing
//! (see `stark-testdata`), and an evenly-sampled sine wave exercises the density
//! policy in a way real input never does.
//!
//! The two halves are **not** the same order of magnitude, which is worth knowing
//! before reading a result. Flattening a fitted stroke is tens of microseconds;
//! fitting the reports that produced it is *milliseconds* — 16 ms for `loop`'s 635
//! samples, against 15 µs to flatten the result. On the CPU side of a stroke,
//! essentially all of the time is the fit.
//!
//! Run with `cargo bench -p stark-core --bench path`.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

use stark_core::command::InputSample;
use stark_core::geom::Vec2;
use stark_core::path::{self, DEFAULT_TOLERANCE, FLATTEN_TOLERANCE, FlattenTolerance, PathFitter};

/// The recorded strokes, by the property of the fit each one stresses.
fn cases() -> Vec<(&'static str, Vec<InputSample>)> {
    [
        ("hairpin", stark_testdata::HAIRPIN_STROKE),
        ("spiral", stark_testdata::SPIRAL_STROKE),
        ("loop", stark_testdata::LOOP_STROKE),
        // Sparse reports over a long arc: the opposite end of the density policy
        // from `hairpin`, and the case where each span covers the most input.
        ("fast", stark_testdata::FAST_STROKE),
    ]
    .into_iter()
    .map(|(name, pts)| (name, samples(pts)))
    .collect()
}

fn samples(pts: &[[f32; 2]]) -> Vec<InputSample> {
    pts.iter()
        .map(|&[x, y]| InputSample::at(Vec2::new(x, y)))
        .collect()
}

/// Fitting: raw reports → control points. Where essentially all of a stroke's CPU
/// time goes (see the module note).
fn fit(c: &mut Criterion) {
    let mut g = c.benchmark_group("fit");
    for (name, pts) in cases() {
        // Elements are input *samples*, which is the one quantity a change to the
        // fitter cannot move. Knot count is an output and would silently rescale
        // the throughput figure whenever the density policy changed. It also makes
        // the throughput figure directly meaningful: `path::fit` is the incremental
        // fitter fed in one call, so per-element is per *pointer move*.
        g.throughput(criterion::Throughput::Elements(pts.len() as u64));

        // The batch form, as `replay_stroke` and the peer path use it.
        g.bench_with_input(BenchmarkId::new("batch", name), &pts, |b, pts| {
            b.iter(|| path::fit(black_box(pts)));
        });

        // The live form. `Session::stroke_to` is `fitter.push`, but the preview that
        // follows it calls `preview_record` → `PathFitter::path`, so a gesture pays
        // one whole `path()` *per pointer move* where the batch form pays one for the
        // stroke. That difference is the only thing separating these two lines.
        //
        // As measured it costs nothing — the two run within noise of each other on
        // all four strokes — so this is a guard rather than a known cost: `path()`
        // rebuilds a `Vec` over every knot, and it is one policy change away from
        // being the O(n²) that a per-move call to it looks like.
        g.bench_with_input(BenchmarkId::new("live", name), &pts, |b, pts| {
            b.iter_batched(
                PathFitter::new,
                |mut f| {
                    for s in pts {
                        f.push(*s);
                        black_box(f.path());
                    }
                    f.finish();
                    f.path()
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

/// Flattening: control points → the polyline of swept segments.
///
/// Three budgets, because they subdivide for different reasons:
///
///  * `default` — [`FLATTEN_TOLERANCE`] as-is. What the renderer ships, so this is
///    the line a regression has to be read off.
///  * `len-capped` — a max segment length, which the dynamics path sets from the
///    reservoir exchange step. Splits by distance rather than by error, so it is
///    the only budget whose cost rises on *sparse* input (on `fast` it emits 179
///    segments where `default` emits 65).
///  * `position-bound` — `angle` relaxed until `position` is what actually decides.
///    This is the regime the circular-arc segment was built for, and the only one
///    where it can change the segment count: measured over these four strokes it
///    cuts 20–30% (`hairpin` 86 → 70, `loop` 66 → 51). Under the shipped budget it
///    cuts nothing at all — `angle` (0.1 rad) makes segments short enough that arc
///    and chord are indistinguishable at `position` — so a benchmark that only ran
///    `default` would report the arc code as dead weight.
fn flatten(c: &mut Criterion) {
    let budgets: [(&str, FlattenTolerance); 3] = [
        ("default", FLATTEN_TOLERANCE),
        (
            "len-capped",
            FlattenTolerance {
                max_len: 8.0,
                ..FLATTEN_TOLERANCE
            },
        ),
        (
            "position-bound",
            FlattenTolerance {
                angle: 1.0,
                ..FLATTEN_TOLERANCE
            },
        ),
    ];

    let mut g = c.benchmark_group("flatten");
    for (name, pts) in cases() {
        let knots = path::fit(&pts);
        for (budget, tol) in budgets {
            // Knots in, not segments out — see the note in `fit`.
            g.throughput(criterion::Throughput::Elements(knots.len() as u64));
            g.bench_with_input(
                BenchmarkId::new(budget, name),
                &(knots.clone(), tol),
                |b, (knots, tol)| b.iter(|| path::flatten(black_box(knots), *tol)),
            );
        }
    }
    g.finish();
}

/// Fit + flatten end to end, at the renderer's own entry point: what a replayed
/// stroke pays on the CPU before a single dispatch is recorded.
///
/// Worth its own line rather than being read off the sum: the two halves share no
/// state, so a change that moves work across the boundary (pricing an edge in the
/// fitter instead of subdividing it in the flattener) shows up here as flat while
/// the two component benchmarks move in opposite directions.
fn end_to_end(c: &mut Criterion) {
    let mut g = c.benchmark_group("geometry");
    for (name, pts) in cases() {
        g.throughput(criterion::Throughput::Elements(pts.len() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(name), &pts, |b, pts| {
            b.iter(|| {
                let knots = path::fit_with_tolerance(black_box(pts), DEFAULT_TOLERANCE);
                path::flatten(&knots, FLATTEN_TOLERANCE)
            });
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    // These are microsecond-scale and very quiet — criterion's default 3s warm-up
    // plus 5s measurement buys no extra resolution over this and would put the
    // 24-benchmark suite in the minutes.
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = fit, flatten, end_to_end
}
criterion_main!(benches);
