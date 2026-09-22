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
/// (`tests/engine/tile_pool.rs`), and the tag `acquire_tex` demands of a caller.
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
/// limit is a **crate** property: this crate raises it once (`lib.rs`), where every
/// test crate that shares the cache would have had to raise its own, and the next one
/// would have to remember.
///
/// Still one device per test *binary*, which is what it was when `tests/engine/common` held
/// the lock: a static in a linked rlib is per-binary, and each test target is its own
/// binary. What that means for a test — a shared device and driver shader cache, but
/// its own pool, renderers and document — `tests/engine/common` explains at the call site.
///
/// Native only, and not by taste: a `GpuContext` is `Send + Sync` on native — wgpu's
/// handles are, and the health cell is an `Arc<Mutex<..>>` — while on wasm they are
/// `Rc<Cell<..>>` and a `static` could not hold one at all. The suite is native by
/// construction, which is what let `tests/engine/common` keep this without a gate; the gate
/// is what the move across the crate boundary costs.
#[cfg(not(target_arch = "wasm32"))]
pub fn shared_context(
    build: impl FnOnce() -> Option<crate::GpuContext>,
) -> Option<&'static crate::GpuContext> {
    static CTX: std::sync::OnceLock<Option<crate::GpuContext>> = std::sync::OnceLock::new();
    CTX.get_or_init(build).as_ref()
}
