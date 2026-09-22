//! The stroke the two stroke benchmarks time (§7.1), stated once.
//!
//! `benches/stroke.rs` is the regression gate and `examples/stroke_bench.rs` the quick
//! look, and both run this one scenario — the bench with `mod scenario;`, the example
//! through a `#[path]`. Not a Cargo target itself: `benches/*/mod.rs` matches neither
//! pattern Cargo auto-discovers. A criterion baseline is only comparable to a run of
//! the *same* scenario, so **the values here are the ones the saved baselines were
//! measured on** and are not to be tuned in passing.

use stark_engine::command::InputSample;
use stark_engine::timing;
use stark_engine::{Engine, Extent2, headless_engine};
use stark_model::document::{BrushEffect, BrushParams};
use stark_model::geom::Vec2;

const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const VIEWPORT: Extent2 = Extent2 {
    width: 1024,
    height: 1024,
};

/// Samples in the benchmarked stroke. Long enough that the per-segment costs
/// dominate the fixed per-stroke setup, short enough that a live gesture — which
/// re-renders the in-flight stroke on every one of them — stays inside a sane
/// measurement window.
pub const N: usize = 240;

/// The engine a scenario runs on. `None` only when this machine has no usable
/// adapter *and* the skip is permitted — `stark_engine::testing`'s decision, the same
/// one the test harness makes.
pub fn engine() -> Option<Engine> {
    stark_engine::testing::or_skip(
        pollster::block_on(headless_engine(TARGET, VIEWPORT)),
        "GPU benchmarks",
    )
}

/// Block until the device has executed everything submitted so far.
///
/// Inside the timed region on purpose: without it `submit` returns almost at once and
/// a benchmark measures command encoding, which is not where the time goes. Its span
/// is what lets a phase table account for the scenario end to end — the encode rows
/// plus `bench.gpu_wait` add up to roughly the wall clock.
pub fn wait_idle(engine: &Engine) {
    timing::span!("bench.gpu_wait");
    engine
        .gpu()
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
}

/// Install the timing layer as this process's subscriber, once.
///
/// `Once` because `criterion_main!` writes `main` and there is nowhere in it to put
/// this. A bare registry with nothing else on it: the callers print their own tables,
/// so a `fmt` layer would only thread engine log lines through them.
pub fn install_timing() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::registry::Registry;
        tracing::subscriber::set_global_default(Registry::default().with(timing::layer()))
            .expect("no subscriber has been installed in this process yet");
    });
}

/// Print what the scenario just run passed through, under `label`, then start a
/// fresh window.
///
/// Between scenarios rather than at the end: the histograms are cumulative, so one
/// table for a whole run would average `dynamics/8` together with `swept/100` and
/// describe neither. The table itself is `Timings`' own `Display`.
pub fn phase_table(label: &str) {
    match timing::snapshot() {
        Some(t) => println!("  phases for {label}:\n{t}"),
        None => println!("  (no timing subscriber installed — phases unavailable)"),
    }
    timing::reset();
}

/// A brush that runs the sequential stamp loop: non-zero `lift`/`deposit` means it
/// picks paint up off the canvas and lays it back down, so it cannot take the swept
/// fast path. Deliberately mid-range values rather than extremes — this is meant to
/// be the everyday smear, not a worst case.
pub fn smear(radius: f32) -> BrushParams {
    let mut b = BrushParams {
        size: radius,
        effect: BrushEffect::painted([0.8, 0.2, 0.1]),
        ..BrushParams::default()
    };
    // At the neutral flow with the axes stated outright, so the measured rates are
    // exactly these numbers.
    let w = b.make_wet();
    w.flow = 1.0;
    w.dynamics.add = 0.5;
    w.dynamics.lift = 0.6;
    w.dynamics.deposit = 0.5;
    b
}

/// The same brush with dynamics off, which is what puts it on the swept path: one
/// instanced draw over the whole stroke instead of a dispatch chain per segment.
/// The control the dynamics numbers are read against.
pub fn plain(radius: f32) -> BrushParams {
    BrushParams {
        effect: Default::default(),
        ..smear(radius)
    }
}

/// A long wavy stroke of `n` reports crossing many tiles — the shape a real smear
/// gesture takes, and wide enough (1400px) that it leaves the 1024px viewport, so
/// tile work is not bounded by what happens to be on screen.
///
/// Synthetic rather than recorded on purpose: the GPU cost keys off segment count
/// and swept area, which must be *fixed* across runs so a change in the timing is a
/// change in the renderer. Recorded input matters in `benches/path.rs`, where report
/// spacing drives the work.
pub fn samples(n: usize) -> Vec<InputSample> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let mut s = InputSample::at(Vec2::new(-700.0 + 1400.0 * t, (t * 12.0).sin() * 220.0));
            s.pressure = 0.7 + 0.3 * (t * 5.0).sin().abs();
            s
        })
        .collect()
}
