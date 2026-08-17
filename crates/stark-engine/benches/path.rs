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
//! Two of the groups are **sweeps** rather than cases: `fit-density` and
//! `fit-tolerance` vary the two things a recorded stroke cannot vary — how often
//! the digitizer reported, and how coarse the caller declared its input to be —
//! and both exist to hold a *flat* line. The fit's window is delimited in arc
//! length, so before it was bounded each of those axes scaled the work done per
//! report, and a per-sample throughput figure over fixed recordings could not see
//! it — which is why the sweeps are here and not merely another recorded stroke.
//! `MAX_WINDOW_SAMPLES` in `path.rs` carries the bound and the argument for it.
//!
//! Run with `cargo bench -p stark-engine --bench path`.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

use stark_engine::command::InputSample;
use stark_engine::path::{
    self, DEFAULT_TOLERANCE, FLATTEN_TOLERANCE, FlattenTolerance, MAX_TOLERANCE, MIN_TOLERANCE,
    PathFitter,
};
use stark_model::geom::Vec2;

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

/// The stroke the two sweeps below are run on: one real hand movement, resampled
/// and re-priced rather than replaced, so only the axis under test moves.
const SWEEP_STROKE: &[[f32; 2]] = stark_testdata::LOOP_STROKE;

/// `pts` walked as a polyline and re-emitted at `n` points evenly spaced **in arc
/// length** — the same hand movement as reported by a digitizer of another rate.
///
/// Arc rather than time because that is the axis the fit's window is actually
/// delimited on, and because these recordings carry no clock. At a roughly steady
/// hand the two agree; where they do not, arc is the one the fitter sees.
///
/// Linear between the reports it was given, so this adds no geometry the hand did
/// not draw — a 4× resample is the same curve, reported four times as often, which
/// is exactly the thing being varied.
fn resampled(pts: &[InputSample], n: usize) -> Vec<InputSample> {
    let arcs: Vec<f32> = pts
        .iter()
        .scan((0.0f32, None::<Vec2>), |(acc, prev), s| {
            if let Some(p) = *prev {
                *acc += (s.pos - p).length();
            }
            *prev = Some(s.pos);
            Some(*acc)
        })
        .collect();
    let total = *arcs.last().unwrap_or(&0.0);
    if total <= 0.0 || n < 2 {
        return pts.to_vec();
    }
    (0..n)
        .map(|i| {
            let want = i as f32 / (n - 1) as f32 * total;
            let k = arcs.partition_point(|&a| a < want).clamp(1, pts.len() - 1);
            let (a, b) = (arcs[k - 1], arcs[k]);
            let u = if b > a { (want - a) / (b - a) } else { 0.0 };
            InputSample::at(pts[k - 1].pos.lerp(pts[k].pos, u))
        })
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

/// **What one pointer report costs as the reports get denser** — the axis the four
/// recorded cases above cannot move, because each arrives at whatever rate it was
/// captured at.
///
/// Read as *throughput*, which is the whole point: elements are input samples, so a
/// fit whose per-report cost does not depend on the reporting rate draws a flat line
/// here however many samples the row has. A rising line means the work done per
/// report grows with the rate — which is a stroke that costs the square of the
/// digitizer's frequency, and the difference between a 200 Hz tablet and a 1000 Hz
/// pen drawing the identical mark.
///
/// It is not hypothetical and it is not small: before the window was bounded this
/// spanned 9.9 µs to 31.1 µs per report across these five rows, so the 8× row cost
/// 25× the total of the 1× row for the same stroke.
fn fit_density(c: &mut Criterion) {
    let base = samples(SWEEP_STROKE);
    let mut g = c.benchmark_group("fit-density");
    for mult in [0.25f32, 0.5, 1.0, 2.0, 4.0, 8.0] {
        let pts = resampled(&base, (base.len() as f32 * mult) as usize);
        g.throughput(criterion::Throughput::Elements(pts.len() as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{mult}x")),
            &pts,
            |b, pts| b.iter(|| path::fit(black_box(pts))),
        );
    }
    g.finish();
}

/// **What one pointer report costs as the declared tolerance rises** — i.e. as the
/// artist zooms out, since that is what a frontend sets it from.
///
/// The other axis the recorded cases hold fixed: they all run at
/// [`DEFAULT_TOLERANCE`]. Same stroke and same report count on every row here, so
/// per-element times are directly comparable down the column, and a fit whose cost
/// is a property of the *stroke* rather than of the view draws a flat line.
///
/// This one used to run backwards, which is why it is worth a benchmark rather than
/// a comment: a coarser tolerance produces far *fewer* knots and used to cost several
/// times more CPU per report, because the solver's window is measured in spans and a
/// span is `KNOT_SPACING × tolerance` canvas px wide. Throughput before and after
/// (Kelem/s):
///
/// | tolerance | 1/64  | 0.25 | 1    | 4    | 16   | 64   | collapse |
/// |-----------|-------|------|------|------|------|------|----------|
/// | before    | 105.1 | 73.5 | 32.3 | 18.7 | 16.3 | 16.4 | 6.4×     |
/// | after     | 108.0 | 73.2 | 65.7 | 59.4 | 60.3 | 61.6 | 1.75×    |
///
/// Flat from `tol-1` rightwards now, which is the property worth having: zooming out
/// no longer costs anything. What is left slopes the *other* way, at the fine end,
/// and is the window filling up to its budget rather than the budget being exceeded.
///
/// The top row is [`MAX_TOLERANCE`], not a round number picked to look severe: it is
/// the coarsest input the fitter will accept, so holding the line flat to here is
/// holding it flat everywhere.
fn fit_tolerance(c: &mut Criterion) {
    let pts = samples(SWEEP_STROKE);
    let mut g = c.benchmark_group("fit-tolerance");
    g.throughput(criterion::Throughput::Elements(pts.len() as u64));
    for tol in [
        MIN_TOLERANCE,
        0.25,
        DEFAULT_TOLERANCE,
        4.0,
        16.0,
        MAX_TOLERANCE,
    ] {
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("tol-{tol}")),
            &(pts.clone(), tol),
            |b, (pts, tol)| b.iter(|| path::fit_with_tolerance(black_box(pts), *tol)),
        );
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
    targets = fit, fit_density, fit_tolerance, flatten, end_to_end
}
criterion_main!(benches);
