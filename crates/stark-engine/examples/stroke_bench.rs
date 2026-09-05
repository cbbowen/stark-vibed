//! Wall-clock benchmark and **phase profile** for the brush-dynamics stroke path
//! (§6.2, §7.1).
//!
//! Measures the case the fast path cannot take: a large brush with non-zero
//! lift/deposit, which runs the sequential swept-exchange loop. Two numbers per
//! scenario, and then a table of where they went:
//!
//!  * **live** — a pointer-move-by-pointer-move gesture, timed per move with the
//!    GPU drained after each one. This is the latency the user feels as lag.
//!  * **commit** — the whole stroke rendered in one pass (`replay_stroke`).
//!  * **phases** — every `timing::span!` the scenario passed through, with its count,
//!    its distribution and its share of the scenario's wall clock.
//!
//! Run with `cargo run --release -p stark-engine --example stroke_bench`.
//!
//! # What the phase table can and cannot tell you
//!
//! Every row is **CPU time to prepare work**, because that is all a `timing` span can
//! see: nothing here issues a GPU timestamp query. So `stroke.loop` is the cost of
//! *recording* the stamp loop's dispatches and not of executing them.
//!
//! That would be a thin result on its own, which is why the drain is instrumented
//! too: `bench.gpu_wait` is `poll(wait_indefinitely)`, so the encode rows plus that
//! one account for the scenario end to end, and the split between "we are encoding"
//! and "we are waiting" is legible at a glance. What the table still cannot do is
//! divide the *waiting* among the dispatches inside it — for that the method is
//! unchanged: bracket by gating a dispatch kind out and re-running, which is what
//! produced the phase shares recorded in the perf notes.
//!
//! The CPU side is worth its own table regardless, and not a formality: fitting a
//! stroke has measured ~350× its flattening and the same order as a whole GPU commit,
//! and `input.fit` is the row that says so.
//!
//! # Against `benches/stroke.rs`
//!
//! That file is the regression gate — criterion, confidence intervals, a saved
//! baseline — and it prints the same phase table under every line it measures. What
//! this one has over it is **speed and the drain**: seconds rather than minutes, a
//! p95 per-move figure, and `bench.gpu_wait`, which criterion has no equivalent of
//! because its timed region is the thing being reported rather than something to
//! account for.
//!
//! It is one run per scenario, so read the shares, which are stable, rather than the
//! totals, which on this class of machine drift 15% between runs of identical code.

#![expect(
    clippy::disallowed_types,
    reason = "a dev harness that only ever runs natively; the app's clock is `quanta` (§7.1)"
)]

use stark_engine::command::Tool;
use std::time::Instant;

use stark_engine::Extent2;
use stark_engine::command::{GestureCommand, InputSample, ViewCommand};
use stark_engine::headless_engine;
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::timing;
use stark_model::document::BrushParams;
use stark_model::geom::Vec2;

const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn wait_idle(engine: &stark_engine::Engine) {
    // Instrumented so the table accounts for the whole scenario: this is the GPU
    // executing everything the encode rows recorded, and without it every share
    // would be a fraction of a wall clock most of which nothing explained.
    timing::span!("bench.gpu_wait");
    engine
        .gpu()
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
}

fn smear_brush(radius: f32) -> BrushParams {
    let mut b = BrushParams {
        size: radius,
        effect: stark_model::document::BrushEffect::painted([0.8, 0.2, 0.1]),
        ..BrushParams::default()
    };
    // At the neutral flow with the axes stated outright, so the measured rates
    // are exactly these numbers — and the figures survive the flow/add split.
    let w = b.make_wet();
    w.flow = 1.0;
    w.dynamics.add = 0.5;
    w.dynamics.lift = 0.6;
    w.dynamics.deposit = 0.5;
    b
}

/// A long wavy stroke crossing many tiles — the shape a real smear gesture takes.
fn samples(n: usize) -> Vec<InputSample> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let x = -700.0 + 1400.0 * t;
            let y = (t * 12.0).sin() * 220.0;
            let mut s = InputSample::at(Vec2::new(x, y));
            s.pressure = 0.7 + 0.3 * (t * 5.0).sin().abs();
            s
        })
        .collect()
}

fn bench_live(engine: &mut stark_engine::Engine, brush: BrushParams, n: usize) -> (f64, f64, f64) {
    engine.process(ViewCommand::set_brush(brush));
    let pts = samples(n);
    let mut it = pts.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: *it.next().unwrap(),
        tolerance: DEFAULT_TOLERANCE,
        rope: 0.0,
    });
    wait_idle(engine);
    let mut per_move = Vec::with_capacity(n);
    let total = Instant::now();
    for s in it {
        let t = Instant::now();
        engine.process(GestureCommand::To { sample: *s });
        // The fold is lazy (`Engine::mark_live_stale`), so the work a frame would do
        // has to be asked for or the loop measures the fitter alone. The frontend
        // takes this read once per rAF; here it is once per move, which is the
        // per-move latency ceiling — and the same thing `benches/stroke.rs` does.
        engine.flush_live();
        wait_idle(engine);
        per_move.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    engine.process(GestureCommand::End);
    wait_idle(engine);
    let total = total.elapsed().as_secs_f64() * 1000.0;
    per_move.sort_by(f64::total_cmp);
    let mean = per_move.iter().sum::<f64>() / per_move.len() as f64;
    let p95 = per_move[(per_move.len() as f64 * 0.95) as usize];
    (total, mean, p95)
}

fn bench_commit(engine: &mut stark_engine::Engine, brush: BrushParams, n: usize) -> f64 {
    engine.process(ViewCommand::set_brush(brush));
    let pts = samples(n);
    wait_idle(engine);
    let t = Instant::now();
    engine.replay_stroke(Tool::Brush, &pts);
    wait_idle(engine);
    t.elapsed().as_secs_f64() * 1000.0
}

/// Print what the scenario just run passed through, then start a fresh window.
///
/// The table itself is `Timings`' own `Display` (§7.1), shared with
/// `benches/stroke.rs`: a column added to one copy and not the other would be a
/// column meaning different things in two places.
///
/// Here the share is of the scenario's whole wall clock, and because the drain is
/// instrumented too the encode rows plus `bench.gpu_wait` add up to roughly 100% —
/// whatever is missing is untimed harness work.
fn phase_table() {
    match timing::snapshot() {
        Some(t) => println!("{t}"),
        None => eprintln!("  (no timing subscriber installed — phases unavailable)"),
    }
    timing::reset();
}

/// Install the timing layer as this process's subscriber.
///
/// A bare registry with nothing else on it: the example prints its own table, so a
/// `fmt` layer would only interleave engine log lines through the numbers.
fn install_timing() {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::registry::Registry;

    tracing::subscriber::set_global_default(Registry::default().with(timing::layer()))
        .expect("no subscriber has been installed in this process yet");
}

fn main() {
    install_timing();
    let viewport = Extent2 {
        width: 1024,
        height: 1024,
    };
    let mut engine =
        pollster::block_on(headless_engine(TARGET, viewport)).expect("headless engine");

    let n = 240;
    // Warm-up: builds prefix/coverage caches and pipelines so the first timed
    // stroke doesn't pay for them. Its phases are thrown away with the reset below
    // rather than reported — a first run's shader compilation is not a phase split.
    bench_commit(&mut engine, smear_brush(30.0), 40);
    timing::reset();

    for radius in [30.0, 60.0, 100.0, 500.0] {
        let commit = bench_commit(&mut engine, smear_brush(radius), n);
        println!("smear r={radius:>5.1}: commit {commit:8.1} ms");
        phase_table();
        let (total, mean, p95) = bench_live(&mut engine, smear_brush(radius), n);
        println!(
            "smear r={radius:>5.1}: live total {total:8.1} ms, \
             per-move mean {mean:6.2} ms, p95 {p95:6.2} ms"
        );
        phase_table();
    }

    // Reference: the same sizes on the swept fast path (no lift/deposit), so the
    // dynamics numbers have something to be compared against.
    for radius in [30.0, 100.0] {
        let mut plain = smear_brush(radius);
        plain.effect = Default::default();
        let commit = bench_commit(&mut engine, plain, n);
        let (total, mean, p95) = bench_live(&mut engine, plain, n);
        println!(
            "plain r={radius:>5.1}: commit {commit:8.1} ms | live total {total:8.1} ms, \
             per-move mean {mean:6.2} ms, p95 {p95:6.2} ms"
        );
        phase_table();
    }

    // A last word on how to read all of the above, printed rather than left to the
    // module comment: the shares are the durable part and the totals are not.
    println!(
        "\nphase rows are CPU time to *record* work; `bench.gpu_wait` is the drain \
         that executes it.\nread the shares — totals on this class of machine drift \
         ~15% between runs of identical code.",
    );
}
