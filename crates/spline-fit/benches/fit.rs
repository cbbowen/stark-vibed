use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::Const;
use spline_fit::{CardinalCubicBSpline, FitTolerance};
use std::hint::black_box;

/// A wiggly, monotonically-parameterized point sequence for benchmarking.
fn wiggly_points(n: usize) -> Vec<[f32; 2]> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            let x = t * 20.0;
            let y = (t * 12.0).sin() * 3.0 + (t * 37.0).cos() * 0.5;
            [x, y]
        })
        .collect()
}

/// The EM accuracy/speed tradeoff used across the fitting benchmarks.
fn tolerance() -> FitTolerance<f32> {
    const EPS: f32 = 1e-2;
    FitTolerance::new(EPS, EPS, EPS)
}

fn make_spline(m: usize, points: &[[f32; 2]]) -> CardinalCubicBSpline<f32, Const<2>> {
    CardinalCubicBSpline::<f32, Const<2>>::fit_monotonic(m, tolerance(), points).0
}

fn bench_evaluate(c: &mut Criterion) {
    let points = wiggly_points(200);
    let spline = make_spline(20, &points);
    let t_end = spline.num_spans() as f32;
    c.bench_function("evaluate", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for i in 0..100 {
                let t = t_end * i as f32 / 100.0;
                acc += spline.evaluate(black_box(t))[0];
            }
            black_box(acc)
        })
    });
    c.bench_function("evaluate_batch", |b| {
        b.iter(|| {
            let ts = (0..100).map(|i| black_box(t_end * i as f32 / 100.0));
            spline.evaluate_many(ts).map(|v| v[0]).sum::<f32>()
        })
    });
}

fn bench_locally_closest_points(c: &mut Criterion) {
    let points = wiggly_points(200);
    let spline = make_spline(20, &points);
    c.bench_function("locally_closest_points", |b| {
        b.iter(|| {
            spline
                .locally_closest_points(black_box(&[5.0, 1.0]), tolerance().into())
                .count()
        })
    });
}

/// The isolated per-span root-finding work (~90% of an E-step): every point's critical
/// points across all spans.
fn bench_all_critical_points(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_critical_points");
    for n in [10, 100, 1000] {
        let points = wiggly_points(n);
        let spline = make_spline(10, &points);
        group.bench_with_input(BenchmarkId::from_parameter(n), &points, |b, pts| {
            b.iter(|| spline.all_critical_points(black_box(pts), tolerance().into()))
        });
    }
    group.finish();
}

fn bench_best_ordered_assignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("best_ordered_assignment");
    for n in [10, 100, 1000] {
        let points = wiggly_points(n);
        let spline = make_spline(10, &points);
        group.bench_with_input(BenchmarkId::from_parameter(n), &points, |b, pts| {
            b.iter(|| spline.best_ordered_assignment(black_box(pts), tolerance().into()))
        });
    }
    group.finish();
}

fn bench_fit_monotonic(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_monotonic");
    for n in [10, 100, 1000] {
        let points = wiggly_points(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &points, |b, pts| {
            b.iter(|| {
                CardinalCubicBSpline::<f32, Const<2>>::fit_monotonic(
                    10,
                    tolerance(),
                    black_box(pts),
                )
            })
        });
    }
    group.finish();
}

fn bench_fit_monotonic_adaptive(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_monotonic_adaptive");
    for n in [10, 100, 1000] {
        let points = wiggly_points(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &points, |b, pts| {
            b.iter(|| {
                CardinalCubicBSpline::<f32, Const<2>>::fit_monotonic_adaptive(
                    0.1,
                    tolerance(),
                    black_box(pts),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_evaluate,
    bench_locally_closest_points,
    bench_all_critical_points,
    bench_best_ordered_assignment,
    bench_fit_monotonic,
    bench_fit_monotonic_adaptive
);
criterion_main!(benches);
