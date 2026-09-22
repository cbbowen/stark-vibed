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
//! baseline — and it runs the same scenario (`benches/scenario`) and prints the same
//! phase table under every line it measures. What this one has over it is **speed**:
//! seconds rather than minutes, and a p95 per-move figure.
//!
//! It is one run per scenario, so read the shares, which are stable, rather than the
//! totals, which on this class of machine drift 15% between runs of identical code.

#![expect(
    clippy::disallowed_types,
    reason = "a dev harness that only ever runs natively; the app's clock is `quanta` (§7.1)"
)]

#[path = "../benches/scenario/mod.rs"]
mod scenario;

use stark_engine::command::Tool;
use std::time::Instant;

use scenario::{N, install_timing, phase_table, plain, samples, smear, wait_idle};
use stark_engine::command::{GestureCommand, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_engine::testing::Probe;
use stark_engine::timing;
use stark_model::document::BrushParams;

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
    engine.replay_stroke(&pts);
    wait_idle(engine);
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    install_timing();
    let Some(mut engine) = scenario::engine() else {
        return;
    };

    // Warm-up: builds prefix/coverage caches and pipelines so the first timed
    // stroke doesn't pay for them. Its phases are thrown away with the reset below
    // rather than reported — a first run's shader compilation is not a phase split.
    bench_commit(&mut engine, smear(30.0), 40);
    timing::reset();

    for radius in [30.0, 60.0, 100.0, 500.0] {
        let commit = bench_commit(&mut engine, smear(radius), N);
        println!("smear r={radius:>5.1}: commit {commit:8.1} ms");
        phase_table(&format!("commit/smear/{radius}"));
        let (total, mean, p95) = bench_live(&mut engine, smear(radius), N);
        println!(
            "smear r={radius:>5.1}: live total {total:8.1} ms, \
             per-move mean {mean:6.2} ms, p95 {p95:6.2} ms"
        );
        phase_table(&format!("live/smear/{radius}"));
    }

    // Reference: the same sizes on the swept fast path (no lift/deposit), so the
    // dynamics numbers have something to be compared against.
    for radius in [30.0, 100.0] {
        let commit = bench_commit(&mut engine, plain(radius), N);
        let (total, mean, p95) = bench_live(&mut engine, plain(radius), N);
        println!(
            "plain r={radius:>5.1}: commit {commit:8.1} ms | live total {total:8.1} ms, \
             per-move mean {mean:6.2} ms, p95 {p95:6.2} ms"
        );
        phase_table(&format!("commit+live/plain/{radius}"));
    }

    // A last word on how to read all of the above, printed rather than left to the
    // module comment: the shares are the durable part and the totals are not.
    println!(
        "\nphase rows are CPU time to *record* work; `bench.gpu_wait` is the drain \
         that executes it.\nread the shares — totals on this class of machine drift \
         ~15% between runs of identical code.",
    );
}
