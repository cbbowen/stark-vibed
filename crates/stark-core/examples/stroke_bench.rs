//! Wall-clock benchmark for the brush-dynamics stroke path (DESIGN.md §6.2).
//!
//! Measures the case the fast path cannot take: a large brush with non-zero
//! lift/deposit, which runs the sequential swept-exchange loop. Two numbers per
//! scenario:
//!
//!  * **live** — a pointer-move-by-pointer-move gesture, timed per move with the
//!    GPU drained after each one. This is the latency the user feels as lag.
//!  * **commit** — the whole stroke rendered in one pass (`replay_stroke`).
//!
//! Run with `cargo run --release -p stark-core --example stroke_bench`.

use std::time::Instant;

use stark_core::command::{GestureCommand, InputSample, ViewCommand};
use stark_core::document::{BrushParams, Tool};
use stark_core::engine::headless_engine;
use stark_core::geom::{Extent2, Vec2};
use stark_core::path::DEFAULT_TOLERANCE;

const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn wait_idle(engine: &stark_core::Engine) {
    engine
        .gpu()
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
}

fn smear_brush(radius: f32) -> BrushParams {
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

fn bench_live(engine: &mut stark_core::Engine, brush: BrushParams, n: usize) -> (f64, f64, f64) {
    engine.process(ViewCommand::SetBrush(brush));
    let pts = samples(n);
    let mut it = pts.iter();
    engine.process(GestureCommand::Start {
        tool: Tool::Brush,
        sample: *it.next().unwrap(),
        tolerance: DEFAULT_TOLERANCE,
    });
    wait_idle(engine);
    let mut per_move = Vec::with_capacity(n);
    let total = Instant::now();
    for s in it {
        let t = Instant::now();
        engine.process(GestureCommand::To { sample: *s });
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

fn bench_commit(engine: &mut stark_core::Engine, brush: BrushParams, n: usize) -> f64 {
    engine.process(ViewCommand::SetBrush(brush));
    let pts = samples(n);
    wait_idle(engine);
    let t = Instant::now();
    engine.replay_stroke(Tool::Brush, &pts);
    wait_idle(engine);
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let viewport = Extent2 {
        width: 1024,
        height: 1024,
    };
    let mut engine =
        pollster::block_on(headless_engine(TARGET, viewport)).expect("headless engine");

    let n = 240;
    // Warm-up: builds prefix/coverage caches and pipelines so the first timed
    // stroke doesn't pay for them.
    bench_commit(&mut engine, smear_brush(30.0), 40);

    for radius in [30.0, 60.0, 100.0, 500.0] {
        let commit = bench_commit(&mut engine, smear_brush(radius), n);
        let (total, mean, p95) = bench_live(&mut engine, smear_brush(radius), n);
        println!(
            "smear r={radius:>5.1}: commit {commit:8.1} ms | live total {total:8.1} ms, \
             per-move mean {mean:6.2} ms, p95 {p95:6.2} ms"
        );
    }

    // Reference: the same sizes on the swept fast path (no lift/deposit), so the
    // dynamics numbers have something to be compared against.
    for radius in [30.0, 100.0] {
        let mut plain = smear_brush(radius);
        plain.dynamics = Default::default();
        let commit = bench_commit(&mut engine, plain.clone(), n);
        let (total, mean, p95) = bench_live(&mut engine, plain, n);
        println!(
            "plain r={radius:>5.1}: commit {commit:8.1} ms | live total {total:8.1} ms, \
             per-move mean {mean:6.2} ms, p95 {p95:6.2} ms"
        );
    }
}
