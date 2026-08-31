//! GPU benchmarks for stroke rendering (§6.2) — the numbers that guard
//! against a performance regression in the swept fast path or the brush-dynamics
//! stamp loop.
//!
//! Three things worth knowing before reading a result:
//!
//!  * **Only comparable to another run on the same machine and adapter.** These
//!    time GPU work through a driver, so there is no absolute figure to assert.
//!    The workflow is `cargo bench -p stark-engine --bench stroke -- --save-baseline
//!    main` before a change and `-- --baseline main` after; criterion then reports
//!    the change with a confidence interval and flags it as improved/regressed.
//!  * **Every iteration drains the device** (`poll(wait_indefinitely)`) inside the
//!    timed region. Without that, `submit` returns almost immediately and the
//!    benchmark measures command encoding — which is not where the time goes.
//!  * **Every benchmark prints its phase split** (§7.1). Criterion answers "did the
//!    total move"; the table under each line answers "which phase moved it", for the
//!    exact configuration criterion just timed — which is the pairing the two were
//!    always missing. The rows are CPU time to *record* work: no timestamp queries,
//!    so what the GPU then spent executing it is not in the table (`bench.gpu_wait`
//!    in `examples/stroke_bench` is the nearest thing, and bracketing is still how
//!    GPU time gets divided among dispatches).
//!
//!    Installing the subscriber invalidates baselines saved before it, and that is
//!    the right trade — **and the cost is measured, not assumed.** A span costs
//!    1.4 ns with no subscriber installed and 234 ns with this one (2 M iterations,
//!    release, 2026-08-16), so the densest benchmark here — `commit/dynamics/8`,
//!    ~42 k spans over a 4.6 s window — pays **0.2%**, and `live/dynamics/500` pays
//!    0.07%. Both are two orders of magnitude under the ~15% this box drifts between
//!    runs of identical code, which is the floor any reading here is against anyway.
//!    If a line ever does move with the instrumentation, that is a finding about the
//!    instrumentation and not a cost to be avoided by switching it off.
//!  * **Radius runs backwards on the dynamics path.** The stamp loop is sequential
//!    per segment, so a *smaller* brush is slower: it flattens to more segments,
//!    and the cost is serialized dispatch latency rather than texel work. Only at
//!    extreme radii does it flip to texel-bound. That is why the radius sweep goes
//!    down to 8 as well as up to 500 — a change can easily improve one end and
//!    regress the other.
//!
//! Run with `cargo bench -p stark-engine --bench stroke`.

#![expect(
    clippy::disallowed_types,
    reason = "a criterion bench is native-only by construction; `quanta` is for the code that reaches a tab"
)]

use stark_engine::command::Tool;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};

use stark_engine::Engine;
use stark_engine::command::{DocCommand, GestureCommand, InputSample, ViewCommand};
use stark_engine::headless_engine;
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::timing;
use stark_model::document::BrushParams;
use stark_model::geom::{Extent2, Vec2};

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

fn wait_idle(engine: &Engine) {
    engine
        .gpu()
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
}

/// Install the timing layer as this process's subscriber, once.
///
/// `Once` rather than a call from `main`, because `criterion_main!` writes `main` and
/// there is nowhere in it to put this. A bare registry with nothing else on it: the
/// tables below are printed directly, so a `fmt` layer would only thread engine log
/// lines through criterion's output.
fn install_timing() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::registry::Registry;
        tracing::subscriber::set_global_default(Registry::default().with(timing::layer()))
            .expect("no subscriber has been installed in this process yet");
    });
}

/// Print what the benchmark just run passed through, then start a fresh window.
///
/// Between benchmarks rather than at the end: the histograms are cumulative, so one
/// table for the whole file would average `dynamics/8` together with `swept/100` and
/// describe neither. Called after `bench_function` returns, so the window covers
/// criterion's warm-up as well as its measurement — which is what we want, since
/// shares are the durable reading and a warm-up runs the same phases.
fn phase_table(label: &str) {
    match timing::snapshot() {
        Some(t) => println!("  phases for {label}:\n{t}"),
        None => println!("  (no timing subscriber installed — phases unavailable)"),
    }
    timing::reset();
}

/// `None` only when this machine has no usable adapter *and* the skip is permitted —
/// `stark_engine::testing`'s decision, the same one the test harness makes.
fn engine() -> Option<Engine> {
    stark_engine::testing::or_skip(
        pollster::block_on(headless_engine(TARGET, VIEWPORT)),
        "GPU benchmarks",
    )
}

/// A brush that runs the sequential stamp loop: non-zero `lift`/`deposit` means it
/// picks paint up off the canvas and lays it back down, so it cannot take the swept
/// fast path. Deliberately mid-range values rather than extremes — this is meant to
/// be the everyday smear, not a worst case.
fn smear(radius: f32) -> BrushParams {
    let mut b = BrushParams {
        size: radius,
        effect: stark_model::document::BrushEffect::painted([0.8, 0.2, 0.1]),
        ..BrushParams::default()
    };
    b.make_wet().dynamics.lift = 0.6;
    b.make_wet().dynamics.deposit = 0.5;
    b.make_wet().dynamics.flow = 0.5;
    b
}

/// [`smear`] with the **bleed** axis on — the blender, and the one brush that fires the
/// lateral flux (§6.2).
///
/// Its own case because bleed is the loop's only pass whose cost is a *stencil*: a
/// firing reads nine rungs × four offsets per texel, where every other pass reads a
/// fixed handful. Without a bleeding brush in this file the whole ladder was
/// unmeasured, and the one number anybody had for it came from a hand-timed replay.
fn blender(radius: f32) -> BrushParams {
    let mut b = smear(radius);
    b.make_wet().dynamics.bleed = 0.9;
    b
}

/// The same brush with dynamics off, which is what puts it on the swept path: one
/// instanced draw over the whole stroke instead of a dispatch chain per segment.
/// The control the dynamics numbers are read against.
fn plain(radius: f32) -> BrushParams {
    BrushParams {
        effect: Default::default(),
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
    install_timing();
    let Some(mut engine) = engine() else { return };
    let pts = samples();
    // Warm-up outside any measurement: the first stroke builds pipelines and the
    // prefix-τ cache, which would otherwise land inside one benchmark's samples and
    // not the others'.
    engine.process(ViewCommand::set_brush(smear(30.0)));
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
        for (mode, brush) in [
            ("dynamics", smear(radius)),
            ("blender", blender(radius)),
            ("swept", plain(radius)),
        ] {
            // r=500 on the swept path is one enormous instanced draw with nothing to
            // learn from — the interesting extreme radius is the dynamics one, where
            // the loop flips from dispatch-bound to texel-bound.
            if mode == "swept" && radius > 100.0 {
                continue;
            }
            // The bleed's cost is the stencil, which is per texel — so it is read at
            // the two radii where the loop is texel-bound, and skipped where the
            // dispatch chain dominates and the firing count is what little there is.
            if mode == "blender" && radius < 100.0 {
                continue;
            }
            engine.process(ViewCommand::set_brush(brush));
            // The warm-up above, and the previous configuration's samples, are not
            // this one's: clear them so the table describes the line criterion is
            // about to print and nothing else.
            timing::reset();
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
            phase_table(&format!("commit/{mode}/{radius}"));
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
    install_timing();
    let Some(mut engine) = engine() else { return };
    let pts = samples();
    engine.process(ViewCommand::set_brush(smear(30.0)));
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

    // Out to 250/500, because a wide tip's live latency is exactly what this group
    // exists to see.
    for radius in [8.0f32, 30.0, 100.0, 250.0, 500.0] {
        for (mode, brush) in [("dynamics", smear(radius)), ("swept", plain(radius))] {
            // Same rationale as `commit`: the swept path at extreme radii is one
            // enormous instanced draw with nothing to learn from.
            if mode == "swept" && radius > 100.0 {
                continue;
            }
            engine.process(ViewCommand::set_brush(brush));
            timing::reset();
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
                            rope: 0.0,
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
            phase_table(&format!("live/{mode}/{radius}"));
        }
    }
    g.finish();
}

criterion_group!(benches, commit, live);
criterion_main!(benches);
