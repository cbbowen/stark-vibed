//! What the test suite and the benchmarks need of this crate and a shipping
//! frontend does not (§9).
//!
//! **`#[doc(hidden)]`, and the module is the point.** An integration test is a
//! separate crate, so anything it reaches has to be `pub` — and the diagnostic methods
//! on [`Engine`](crate::Engine) that only tests call are `pub` for exactly that reason
//! and no other. Marking them says so, and gathering the harness's own shared piece
//! here stops the alternative: one decision written out in three files that cannot see
//! each other.
//!
//! Not behind a cargo feature, deliberately. A feature would have to be enabled by
//! every command in CLAUDE.md's list and by a self-referential dev-dependency, which
//! is a second build of the crate to hide a handful of methods a reader is already
//! told to ignore. What the hidden module buys is honesty about the API's surface;
//! what a feature would buy on top of that is not worth a doubled compile.

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
/// dev-dependency and has no business in the shipped crate — so what is shared here is
/// the judgement and not the plumbing. It was written out three times: the engine
/// harness, the tile pool's own test and the benchmark, each with its own copy of the
/// variable's name and its own wording of the refusal, which is three places for one
/// of them to grow a silent skip.
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
