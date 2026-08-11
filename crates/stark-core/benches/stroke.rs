//! GPU benchmarks for stroke rendering (§6.2) — the numbers that guard
//! against a performance regression in the swept fast path or the brush-dynamics
//! stamp loop.
//!
//! Three things worth knowing before reading a result:
//!
//!  * **Only comparable to another run on the same machine and adapter.** These
//!    time GPU work through a driver, so there is no absolute figure to assert.
//!    The workflow is `cargo bench -p stark-core --bench stroke -- --save-baseline
//!    main` before a change and `-- --baseline main` after; criterion then reports
//!    the change with a confidence interval and flags it as improved/regressed.
//!  * **Every iteration drains the device** (`poll(wait_indefinitely)`) inside the
//!    timed region. Without that, `submit` returns almost immediately and the
//!    benchmark measures command encoding — which is not where the time goes.
//!  * **Radius runs backwards on the dynamics path.** The stamp loop is sequential
//!    per segment, so a *smaller* brush is slower: it flattens to more segments,
//!    and the cost is serialized dispatch latency rather than texel work. Only at
//!    extreme radii does it flip to texel-bound. That is why the radius sweep goes
//!    down to 8 as well as up to 500 — a change can easily improve one end and
//!    regress the other.
//!
//! Run with `cargo bench -p stark-core --bench stroke`.

use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};

use stark_core::Engine;
use stark_core::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushParams, Tool};
use stark_core::engine::headless_engine;
use stark_core::geom::{Extent2, Vec2};
use stark_core::path::DEFAULT_TOLERANCE;

const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const VIEWPORT: Extent2 = Extent2 {
    width: 1024,
    height: 1024,
};

/// Samples in the benchmarked stroke. Long enough that the per-segment costs
/// dominate the fixed per-stroke setup, short enough that a live gesture — which
/// re-renders the in-flight stroke on every one of them — stays inside a sane
/// measurement window.
const N: usize = 240;

/// Same as `tests/common`: a missing adapter is a failure by default, because a
/// benchmark that silently measures nothing is worse than one that does not run.
const ALLOW_NO_GPU: &str = "STARK_ALLOW_NO_GPU";

fn wait_idle(engine: &Engine) {
    engine
        .gpu()
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
}

/// `None` only when this machine has no usable adapter *and* the skip is permitted.
fn engine() -> Option<Engine> {
    match pollster::block_on(headless_engine(TARGET, VIEWPORT)) {
        Ok(e) => Some(e),
        Err(e) if std::env::var(ALLOW_NO_GPU).is_ok_and(|v| v == "1") => {
            eprintln!("skipping GPU benchmarks ({ALLOW_NO_GPU}=1): {e}");
            None
        }
        Err(e) => panic!("no usable GPU adapter: {e}\nset {ALLOW_NO_GPU}=1 to skip"),
    }
}

/// A brush that runs the sequential stamp loop: non-zero `lift`/`deposit` means it
/// picks paint up off the canvas and lays it back down, so it cannot take the swept
/// fast path. Deliberately mid-range values rather than extremes — this is meant to
/// be the everyday smear, not a worst case.
fn smear(radius: f32) -> BrushParams {
    let mut b = BrushParams {
        radius,
        color: [0.8, 0.2, 0.1, 0.9],
        ..BrushParams::default()
    };
    b.dynamics.lift = 0.6;
    b.dynamics.deposit = 0.5;
    b.dynamics.add = 0.5;
    b
}

/// The same brush with dynamics off, which is what puts it on the swept path: one
/// instanced draw over the whole stroke instead of a dispatch chain per segment.
/// The control the dynamics numbers are read against.
fn plain(radius: f32) -> BrushParams {
    BrushParams {
        dynamics: Default::default(),
        ..smear(radius)
    }
}

/// A long wavy stroke crossing many tiles — the shape a real smear gesture takes,
/// and wide enough (1400px) that it leaves the 1024px viewport, so tile work is not
/// bounded by what happens to be on screen.
///
/// Synthetic rather than recorded on purpose: the GPU cost keys off segment count
/// and swept area, and those want to be *fixed* across runs so that a change in the
/// timing is a change in the renderer. `benches/path.rs` is where recorded input
/// matters, because that is where report spacing drives the work.
fn samples() -> Vec<InputSample> {
    (0..N)
        .map(|i| {
            let t = i as f32 / (N - 1) as f32;
            let mut s = InputSample::at(Vec2::new(-700.0 + 1400.0 * t, (t * 12.0).sin() * 220.0));
            s.pressure = 0.7 + 0.3 * (t * 5.0).sin().abs();
            s
        })
        .collect()
}

/// Undo the stroke an iteration just committed, so the next one starts from the same
/// document.
///
/// Not cosmetic. Committed strokes accumulate: the history log grows without bound
/// across a few hundred iterations, and the dynamics path reads the canvas it is
/// smearing, so each pass would lift paint the previous pass laid down. Untimed,
/// because restoring the document is the harness's business and not the renderer's.
fn rewind(engine: &mut Engine) {
    engine.process(DocCommand::Undo);
    wait_idle(engine);
}

/// The whole stroke rendered in one pass — the commit path, and what a replay or a
/// remote peer's stroke costs.
fn commit(c: &mut Criterion) {
    let Some(mut engine) = engine() else { return };
    let pts = samples();
    // Warm-up outside any measurement: the first stroke builds pipelines and the
    // prefix-τ cache, which would otherwise land inside one benchmark's samples and
    // not the others'.
    engine.process(ViewCommand::SetBrush(smear(30.0)));
    engine.replay_stroke(Tool::Brush, &pts);
    wait_idle(&engine);
    rewind(&mut engine);

    let mut g = c.benchmark_group("commit");
    // A commit is milliseconds, not nanoseconds; criterion's default 100 samples
    // would put this group in the minutes for no extra resolution. 12s is what the
    // slowest member (`dynamics/8`, ~50ms) needs to fit 20 linearly-scaled samples;
    // the faster ones spend the same wall clock on more iterations.
    g.sample_size(20)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(12));

    for radius in [8.0f32, 30.0, 100.0, 250.0, 500.0] {
        for (mode, brush) in [("dynamics", smear(radius)), ("swept", plain(radius))] {
            // r=500 on the swept path is one enormous instanced draw with nothing to
            // learn from — the interesting extreme radius is the dynamics one, where
            // the loop flips from dispatch-bound to texel-bound.
            if mode == "swept" && radius > 100.0 {
                continue;
            }
            engine.process(ViewCommand::SetBrush(brush));
            g.bench_function(BenchmarkId::new(mode, radius), |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let t = Instant::now();
                        engine.replay_stroke(Tool::Brush, &pts);
                        wait_idle(&engine);
                        total += t.elapsed();
                        rewind(&mut engine);
                    }
                    total
                });
            });
        }
    }
    g.finish();
}

/// A whole interactive gesture: press, [`N`] pointer moves each drained before the
/// next, release.
///
/// The number the user feels. It is not the commit cost divided up — a live stroke
/// re-renders its unfrozen tail on *every* move, so this also measures how well the
/// freezing boundary is holding (`safe_frozen`, `StrokeCarry`). A change that
/// leaves `commit` flat and moves this one has moved the freezing, and that is
/// exactly the regression hardest to notice by hand.
fn live(c: &mut Criterion) {
    let Some(mut engine) = engine() else { return };
    let pts = samples();
    engine.process(ViewCommand::SetBrush(smear(30.0)));
    engine.replay_stroke(Tool::Brush, &pts);
    wait_idle(&engine);
    rewind(&mut engine);

    let mut g = c.benchmark_group("live");
    // One sample here is 240 drained round-trips — a second and a half at the worst
    // radius — so this is the slowest group by a wide margin. Flat sampling because
    // of it: criterion's default scales iterations linearly across samples, which on
    // a benchmark this slow means the last sample alone runs ten gestures. Flat gives
    // every sample the same (small) iteration count, so the group finishes in
    // roughly the time asked for. Ten samples is criterion's floor, and the spread it
    // reports (~±2%) resolves any regression worth acting on.
    g.sampling_mode(SamplingMode::Flat)
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        // 15s is what the slowest member needs: `dynamics/8` is ~1.4s a gesture, and
        // flat sampling will not run fewer than one gesture per sample.
        .measurement_time(Duration::from_secs(15));

    // 250/500 too, unlike the old sweep: a wide tip's live latency is exactly what
    // this group exists to see, and it was the one thing it could not.
    for radius in [8.0f32, 30.0, 100.0, 250.0, 500.0] {
        for (mode, brush) in [("dynamics", smear(radius)), ("swept", plain(radius))] {
            // Same rationale as `commit`: the swept path at extreme radii is one
            // enormous instanced draw with nothing to learn from.
            if mode == "swept" && radius > 100.0 {
                continue;
            }
            engine.process(ViewCommand::SetBrush(brush));
            g.bench_function(BenchmarkId::new(mode, radius), |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let t = Instant::now();
                        let mut it = pts.iter();
                        engine.process(GestureCommand::Start {
                            tool: Tool::Brush,
                            sample: *it.next().expect("samples"),
                            tolerance: DEFAULT_TOLERANCE,
                        });
                        for s in it {
                            engine.process(GestureCommand::To { sample: *s });
                            // The fold is lazy: `To` marks it stale and the
                            // presentation read rebuilds it. The frontend takes
                            // that read once per rAF; folding per move here is
                            // the per-move latency ceiling this group guards —
                            // and the same work the pre-lazy engine did per `To`,
                            // so older baselines stay comparable.
                            engine.flush_live();
                            // Drain per move, not per gesture: queuing all 240 moves
                            // and waiting once would measure throughput, and what
                            // makes a stroke feel laggy is the latency of one move.
                            wait_idle(&engine);
                        }
                        engine.process(GestureCommand::End);
                        wait_idle(&engine);
                        total += t.elapsed();
                        rewind(&mut engine);
                    }
                    total
                });
            });
        }
    }
    g.finish();
}

criterion_group!(benches, commit, live);
criterion_main!(benches);
