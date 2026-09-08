//! What the test suite and the benchmarks need of this crate and a shipping
//! frontend does not (§9).
//!
//! **`#[doc(hidden)]`, and the module is the point.** An integration test is a
//! separate crate, so anything it reaches has to be `pub`; the diagnostic methods on
//! [`Engine`](crate::Engine) that only tests call are `pub` for that reason and no
//! other, and marking them says so. The harness's own shared pieces live here so the
//! decision is written once rather than in files that cannot see each other.
//!
//! Not behind a cargo feature, deliberately: a feature would have to be enabled by
//! every command in CLAUDE.md's list and by a self-referential dev-dependency — a
//! second build of the crate to hide a handful of methods a reader is already told
//! to ignore.

/// The recycling tile pool (§6.1), for the one test that drives it directly
/// (`tests/tile_pool.rs`), and the tag `acquire_tex` demands of a caller.
pub use crate::gpu::{AllocSource, TilePool};

/// The environment variable that turns a missing GPU from a failure into a skip.
///
/// **A missing adapter is a failure unless this says otherwise**, and that is the
/// whole of why it exists. A skipped test still reports `ok`, so a suite that quietly
/// stopped finding a device would take the golden, seam and dynamics rounds green
/// having rendered nothing (CLAUDE.md). CI sets it because CI has no adapter; a
/// developer's machine must not.
pub const ALLOW_NO_GPU: &str = "STARK_ALLOW_NO_GPU";

/// Whether the environment permits skipping GPU work. Exactly `"1"`: an empty or
/// misspelt value must not read as permission.
pub fn allowed_to_skip() -> bool {
    std::env::var(ALLOW_NO_GPU).is_ok_and(|v| v == "1")
}

/// **The one place a missing GPU is decided about**: `Some` for something that built,
/// `None` for a permitted skip, and a panic otherwise. `what` names what is being
/// skipped, for the message.
///
/// The *blocking* and the caching stay with each caller — `pollster` is a
/// dev-dependency and has no business in the shipped crate — so what is shared here
/// is the judgement and not the plumbing. Every harness must route through it, or one
/// of them grows a silent skip.
pub fn or_skip<T, E: std::fmt::Display>(built: std::result::Result<T, E>, what: &str) -> Option<T> {
    match built {
        Ok(t) => Some(t),
        Err(e) if allowed_to_skip() => {
            eprintln!("skipping {what} ({ALLOW_NO_GPU}=1): {e}");
            None
        }
        Err(e) => panic!("no usable GPU adapter: {e}\nset {ALLOW_NO_GPU}=1 to skip {what}"),
    }
}

/// The one GPU context a test binary builds, kept for that binary's life.
///
/// **The caching is here; the blocking is still the caller's.** `build` runs at most
/// once, and `pollster` stays a dev-dependency of the suite rather than a dependency
/// of the shipped crate — which is the split the module note above draws, with the
/// cache moved across it and the plumbing left where it was.
///
/// What moving it buys is the crate the static is *proved* in.
/// `OnceLock<Option<GpuContext>>: Sync` unfolds through wgpu's dispatch enum and
/// wgpu-core's `Global` deeply enough to outrun the default recursion limit, and the
/// limit is a **crate** property: this crate raises it once (`lib.rs`), where the
/// forty-odd integration-test crates that share `tests/common` would each have had to
/// raise their own, and a forty-first would have to remember.
///
/// Still one device per test *binary*, which is what it was when `tests/common` held
/// the lock: a static in a linked rlib is per-binary, and each test target is its own
/// binary. What that means for a test — a shared device and driver shader cache, but
/// its own pool, renderers and document — `tests/common` explains at the call site.
///
/// Native only, and not by taste: a `GpuContext` is `Send + Sync` on native — wgpu's
/// handles are, and the health cell is an `Arc<Mutex<..>>` — while on wasm they are
/// `Rc<Cell<..>>` and a `static` could not hold one at all. The suite is native by
/// construction, which is what let `tests/common` keep this without a gate; the gate
/// is what the move across the crate boundary costs (`bench` below is gated for its
/// own reason).
#[cfg(not(target_arch = "wasm32"))]
pub fn shared_context(
    build: impl FnOnce() -> Option<crate::GpuContext>,
) -> Option<&'static crate::GpuContext> {
    static CTX: std::sync::OnceLock<Option<crate::GpuContext>> = std::sync::OnceLock::new();
    CTX.get_or_init(build).as_ref()
}

/// The stroke the two benchmarks time (§7.1), stated once.
///
/// `benches/stroke.rs` is the regression gate and `examples/stroke_bench.rs` the quick
/// look, and both run this one scenario. A criterion baseline is only comparable to a
/// run of the *same* scenario, so **the values here are the ones the saved baselines
/// were measured on** and are not to be tuned in passing.
///
/// Native only: nothing in a browser runs a benchmark, and `poll(wait)` has no meaning
/// there (`gpu::readback` gates the same call).
#[cfg(not(target_arch = "wasm32"))]
pub mod bench {
    use crate::command::InputSample;
    use crate::timing;
    use crate::{Engine, Extent2};
    use stark_model::document::{BrushEffect, BrushParams};
    use stark_model::geom::Vec2;

    pub const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    pub const VIEWPORT: Extent2 = Extent2 {
        width: 1024,
        height: 1024,
    };

    /// Samples in the benchmarked stroke. Long enough that the per-segment costs
    /// dominate the fixed per-stroke setup, short enough that a live gesture — which
    /// re-renders the in-flight stroke on every one of them — stays inside a sane
    /// measurement window.
    pub const N: usize = 240;

    /// Block until the device has executed everything submitted so far.
    ///
    /// Inside the timed region on purpose: without it `submit` returns almost at once
    /// and a benchmark measures command encoding, which is not where the time goes.
    pub fn wait_idle(engine: &Engine) {
        engine
            .gpu()
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }

    /// Install the timing layer as this process's subscriber, once.
    ///
    /// `Once` because `criterion_main!` writes `main` and there is nowhere in it to put
    /// this. A bare registry with nothing else on it: the callers print their own
    /// tables, so a `fmt` layer would only thread engine log lines through them.
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
        // At the neutral flow with the axes stated outright, so the measured rates
        // are exactly these numbers — and the baseline survives the flow/add split.
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
    /// and swept area, which must be *fixed* across runs so a change in the timing is
    /// a change in the renderer. Recorded input matters in `benches/path.rs`, where
    /// report spacing drives the work.
    pub fn samples(n: usize) -> Vec<InputSample> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let mut s =
                    InputSample::at(Vec2::new(-700.0 + 1400.0 * t, (t * 12.0).sin() * 220.0));
                s.pressure = 0.7 + 0.3 * (t * 5.0).sin().abs();
                s
            })
            .collect()
    }
}
