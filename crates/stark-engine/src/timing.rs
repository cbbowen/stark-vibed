//! Where a frame's time actually goes — measured in the shipped app, not only
//! under a benchmark (§7.1).
//!
//! One rule holds the module together: **a phase is a span, and the span's name is
//! its row.** [`span!`] opens one, the layer [`layer`] builds keeps an HDR histogram
//! of how long each *name* takes — creation to close, so a row is a span's own
//! lifetime and a child's does not eat its parent's — and [`snapshot`] reads those
//! histograms out. That is the whole model, and everything below is a consequence of
//! it.
//!
//! - **Nothing here enumerates the phases.** Adding a `span!` to a new piece of the
//!   pipeline is all it takes for a row to appear in the Timing Stats dialog and in
//!   `examples/stroke_bench`'s table, because both render whatever the histograms
//!   hold rather than a list somebody has to keep in step. The names are dotted
//!   (`stroke.loop`, `render.composite`) so a reader can group them; that is a
//!   naming convention for the eye, **not** a call tree — a span's histogram
//!   aggregates every call site that opened it, and `stroke.range` is entered both
//!   from a commit and from the live fold.
//! - **A row is a distribution, not a number.** `mean` answers "what does this cost",
//!   `p99`/`max` answer "what makes it hitch", and those are different questions —
//!   the second is the one a painter feels and the one an average hides.
//! - **Totals are read against the window, not against each other.** [`Timings`]
//!   carries the wall-clock span it covers, so `count / window` is a rate (frames a
//!   second, pointer samples a second) and `total / window` is the share of wall
//!   time a phase took. Nested spans double-count against each other by
//!   construction; against the window they do not.
//!
//! # The browser's clock is coarse, and that is the design constraint
//!
//! `quanta` reads `performance.now()` on the web, which browsers deliberately
//! quantize — 100 µs in Chromium, and a full **millisecond** in a Firefox that is
//! not cross-origin isolated. A single sub-millisecond phase therefore reads as 0
//! or as one whole quantum there, and no amount of care in this file changes that.
//!
//! Two things follow, and both are load-bearing:
//!
//! - **Instrument phases, not operations.** Every `span!` in the tree wraps
//!   something that is milliseconds when it matters — a fold, a compute-pass
//!   recording, a region composite. A span around a bind-group build would be
//!   honest arithmetic over noise.
//! - **Say what the quantum is.** [`Timings::quantum`] is *measured* — the clock is
//!   watched until it moves — rather than assumed, and the dialog shows it, so a row
//!   reading `0.0 ms` is read as
//!   "under the clock's resolution" rather than as "free". Aggregate figures
//!   survive the quantization that individual ones do not: truncation costs each
//!   sample under half a quantum, so a mean over thousands of samples is good well
//!   below the resolution of any one of them, which is exactly why this module
//!   keeps histograms rather than last-value gauges.
//!
//! Natively — the benchmarks — `quanta` uses the invariant TSC and the same spans
//! resolve to nanoseconds. The instrumentation does not change; only how finely it
//! can be read does.
//!
//! # Why these spans are `info`, and why they carry a target
//!
//! The workspace pins `tracing`'s `release_max_level_info`, so a release build
//! compiles `debug!` and `trace!` away entirely — and instrumentation the shipped
//! app does not carry cannot answer a question about the shipped app. So the spans
//! are `info_span!`, at the same level as the handful of things Stark actually logs.
//!
//! That makes [`TARGET`] the thing that separates them, and it separates them in
//! *both* directions — which is why [`TimingFilter`] is one type with two settings
//! rather than a predicate each layer negates for itself. `TimingFilter::<true>`
//! keeps foreign spans out of the histograms, or every `info_span!` in `iroh` would
//! open a row. `TimingFilter::<false>` keeps these spans out of the log, which is not
//! cosmetic: `tracing_wasm`'s layer calls `performance.mark`/`measure` on **every**
//! span it sees, so an unfiltered console layer would spend two JS calls and two
//! `String`s per phase per frame reporting timings into the devtools timeline.
//!
//! Getting that target right at ~20 call sites is exactly the kind of thing this
//! codebase makes structural rather than remembered, which is why [`span!`] exists
//! and why no call site writes `info_span!` by hand.
//!
//! # What it costs, and why it is always on
//!
//! Opening a span stamps a `u64` into storage the span registry already allocated;
//! closing one takes a lock, hashes a `&'static str` and records a sample. Measured
//! at **234 ns a span** end to end, and **1.4 ns** with no subscriber installed —
//! `cargo test`, or an embedding that never calls [`layer`] — where the macro is an
//! atomic load and a disabled span. At the granularity above that is a few
//! microseconds a frame against a 16 ms budget, and 0.07–0.2% of the stroke
//! benchmarks, which is why they run instrumented (`benches/stroke.rs`).
//!
//! So it is on for everyone, always, and that is the point: a profile you have to
//! rebuild to collect is a profile of a build nobody is using.
//!
//! # Why the layer is written here rather than taken off the shelf
//!
//! `tracing-timing` is the crate for this and Stark used it first. Its **read** path
//! cannot run in a browser — see [`PhaseLayer`] for what exactly goes wrong and what
//! was kept. The short version: what it offers over these forty lines is
//! multi-threaded recording, the browser build has one thread, and the machinery that
//! buys the threads is precisely the machinery that panics without a `std` clock.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use hdrhistogram::Histogram;
use tracing::subscriber::Interest;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, Filter};
use tracing_subscriber::registry::LookupSpan;

use crate::unpoisoned;

/// The `tracing` target every timing span carries.
///
/// [`TimingFilter`] is what keys off it, in both directions — see the module note.
/// Namespaced with `::` rather than `_` so it cannot collide with a module path,
/// which is what an unqualified target defaults to.
pub const TARGET: &str = "stark::timing";

/// Which half of the world a layer wants: the phases that feed the histograms, or
/// everything else.
///
/// `TimingFilter::<true>` is the half that measures; `TimingFilter::<false>` is the
/// half that logs.
///
/// **The two are exact complements**, which is the whole reason this is one type with
/// a flipped parameter rather than a predicate each layer negates for itself. A
/// subscriber stack that puts `<true>` on the histogram layer and `<false>` on the
/// console layer handles every event exactly once, and that stays true if what counts
/// as a timing span ever grows past a target comparison — the definition lives in
/// [`admits`](Self::admits), and neither call site restates it.
///
/// The negative half is not tidiness (see the module note): `tracing_wasm`'s layer
/// spends a `performance.mark` and a formatted `String` on every span it is shown, so
/// a console layer without `TimingFilter::<false>` would report Stark's own
/// instrumentation into the devtools timeline a dozen times a frame, for nobody.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimingFilter<const KEEP_TIMING: bool>;

impl<const KEEP_TIMING: bool> TimingFilter<KEEP_TIMING> {
    /// Whether this filter admits `meta`.
    ///
    /// A comparison against the target, and — as long as that is all it is — one
    /// that depends on nothing but the callsite, which is what lets
    /// [`callsite_enabled`](Filter::callsite_enabled) answer once and for all below.
    /// A future definition that reads anything *dynamic* has to give that up in the
    /// same commit.
    fn admits(&self, meta: &tracing::Metadata<'_>) -> bool {
        (meta.target() == TARGET) == KEEP_TIMING
    }
}

impl<const KEEP_TIMING: bool, S> Filter<S> for TimingFilter<KEEP_TIMING> {
    fn enabled(&self, meta: &tracing::Metadata<'_>, _: &Context<'_, S>) -> bool {
        self.admits(meta)
    }

    /// Answered per callsite rather than per event, so `tracing` caches the decision
    /// and the layer is never asked again about a span it will always take or always
    /// refuse.
    ///
    /// Sound because [`admits`](Self::admits) reads only the metadata, which is
    /// `'static` per callsite. `Interest` here is *per layer* — a `never` from this
    /// filter disables the callsite for the layer it is attached to and leaves the
    /// others alone, which is exactly the arrangement the two settings describe.
    fn callsite_enabled(&self, meta: &'static tracing::Metadata<'static>) -> Interest {
        if self.admits(meta) {
            Interest::always()
        } else {
            Interest::never()
        }
    }
}

/// Open a timing span for the rest of the enclosing block.
///
/// ```ignore
/// fn flush_live(&mut self) {
///     crate::timing::span!("live.fold");
///     // ...measured until this function's block ends.
/// }
/// ```
///
/// **A statement, not an expression**, and deliberately: the guard is bound to a
/// name only the macro can see, so there is no binding for a call site to drop early
/// by writing `let _ = …` — the classic way to turn a span into a no-op that still
/// looks instrumented. Measuring a *part* of a block is spelled the way scope is
/// always spelled, with braces around it.
///
/// The name is the row, so it should read as a phase of the pipeline
/// (`stroke.writeback`) rather than as the function it happens to sit in.
#[macro_export]
macro_rules! __stark_timing_span {
    ($name:expr) => {
        let _stark_timing_guard =
            ::tracing::info_span!(target: $crate::timing::TARGET, $name).entered();
    };
}

#[doc(inline)]
pub use crate::__stark_timing_span as span;

/// The narrowest and widest phase a histogram can distinguish, in nanoseconds.
///
/// A microsecond floor is below every clock Stark runs on — the TSC's tick is under
/// a nanosecond and the browser quantizes to 100 µs or worse — so the floor is never
/// what limits a reading. The ten-second ceiling is not a phase anyone is timing but
/// a *saturation* point: a tab left in a background throttle, or a first-frame shader
/// compile, records at the ceiling instead of resizing the histogram under the
/// recorder.
const HIST_LOW_NS: u64 = 1_000;
const HIST_HIGH_NS: u64 = 10_000_000_000;

/// Significant figures each histogram keeps. Two is ~1% error on any quantile, which
/// is far finer than the browser clock the samples come off — and it costs tens of
/// kilobytes a row, a rounding error against a single tile.
const HIST_SIGFIG: u8 = 2;

/// When a span was created, stashed in the registry's own per-span extensions.
///
/// The registry already allocates that storage for every span it tracks, so a phase
/// costs no allocation of its own — and the reading is the *raw* counter rather than
/// a `Duration`, so the conversion happens once, at close, on the value that is
/// actually recorded.
#[derive(Clone, Copy)]
struct Opened(u64);

/// The layer [`layer`] builds: one HDR histogram per span name, and the clock the
/// samples, the window and the quantum are all taken against.
///
/// # Why this is ~40 lines here and not a dependency
///
/// `tracing-timing` is the obvious crate for this and Stark used it first. Its
/// **read** path cannot run in a browser: draining a `SyncHistogram` means
/// `refresh_timeout`, which calls `std::time::Instant::now()` before it looks at
/// anything, and on `wasm32-unknown-unknown` that is a panic — "time not implemented
/// on this platform". There is no way round it from outside the crate, because
/// `force_synchronize` is the only door to the recorders and it always passes a
/// timeout.
///
/// The whole apparatus that needs it — a channel, a phase counter, a recorder per
/// thread — exists to let many OS threads record without synchronizing. Stark's
/// browser build has one thread. So what is kept is the part that was doing the work
/// (`hdrhistogram`, whose quantiles are the point) and what is dropped is the
/// cross-thread machinery that only ever cost us: `hdrhistogram`'s `sync` feature
/// goes, taking `crossbeam-channel` out of the wasm binary, and there is no lock on
/// span *creation* at all — `tracing-timing` takes a write lock there, and this
/// stamps a `u64` into storage the registry had already allocated.
struct PhaseLayer {
    clock: quanta::Clock,
    /// One histogram per [`span!`] name. Keyed by the `&'static str` the callsite
    /// already owns, so a recorded sample allocates nothing.
    rows: Mutex<HashMap<&'static str, Histogram<u64>>>,
    /// Raw clock reading at the last install or [`reset`] — the start of the window
    /// [`Timings::window`] reports.
    window_start: AtomicU64,
    /// The measured clock resolution, cached after the first read of it
    /// ([`PhaseLayer::quantum`]). Zero until then, which is also the answer for a
    /// clock that never moved.
    quantum_ns: AtomicU64,
}

impl PhaseLayer {
    fn new() -> Self {
        let clock = quanta::Clock::new();
        let window_start = AtomicU64::new(clock.raw());
        Self {
            clock,
            rows: Mutex::new(HashMap::new()),
            window_start,
            quantum_ns: AtomicU64::new(0),
        }
    }

    /// Fold one finished phase in.
    fn record(&self, name: &'static str, ns: u64) {
        let mut rows = unpoisoned(self.rows.lock());
        rows.entry(name)
            .or_insert_with(|| {
                Histogram::new_with_bounds(HIST_LOW_NS, HIST_HIGH_NS, HIST_SIGFIG)
                    .expect("the histogram bounds are constants and satisfy low * 2 <= high")
            })
            // Saturating, so a phase that ran past the ceiling — a first-frame shader
            // compile, a tab returning from a background throttle — lands at the top
            // of the histogram instead of resizing it under the recorder.
            .saturating_record(ns);
    }
}

impl<S> Layer<S> for PhaseLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        _: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Opened(self.clock.raw()));
        }
    }

    /// **Creation to close is the sample**, which for every call site in Stark is the
    /// enclosing block: [`span!`] creates and enters in one statement and the guard
    /// drops at the brace.
    ///
    /// Deliberately *not* accumulated across enter/exit pairs. A phase that were
    /// entered, left and re-entered would then report the time it was on the stack
    /// rather than the time it took, and the two differ only for a span held across
    /// an await — which no `span!` in this crate is, because the macro's guard is not
    /// `Send` and so cannot cross one.
    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let opened = span.extensions().get::<Opened>().copied();
        let Some(Opened(at)) = opened else { return };
        self.record(span.name(), self.clock.delta_as_nanos(at, self.clock.raw()));
    }
}

/// The tracing layer that turns [`span!`] into histograms, filtered to [`TARGET`].
///
/// Compose it into a subscriber stack; the frontend puts the console layer beside
/// it, a benchmark uses it alone:
///
/// ```ignore
/// use tracing_subscriber::layer::SubscriberExt;
/// use tracing_subscriber::registry::Registry;
///
/// let subscriber = Registry::default().with(stark_engine::timing::layer());
/// tracing::subscriber::set_global_default(subscriber).expect("set subscriber");
/// ```
///
/// Called more than once, you get more than one set of histograms, and [`snapshot`]
/// finds whichever the installed subscriber holds — fine for a test that wants its
/// own, a bug anywhere else.
pub fn layer<S>() -> impl Layer<S>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    PhaseLayer::new().with_filter(TimingFilter::<true>)
}

/// The installed [`PhaseLayer`], if a subscriber holds one.
///
/// Every reader goes through here rather than through a handle the caller has to keep,
/// because the layer is handed to a subscriber and the subscriber is global. The
/// `Filtered` wrapper `layer` puts round it forwards a downcast to the layer inside,
/// so the type asked for is the bare one.
fn with_layer<R>(f: impl Fn(&PhaseLayer) -> R) -> Option<R> {
    tracing::dispatcher::get_default(|dispatch| dispatch.downcast_ref::<PhaseLayer>().map(&f))
}

/// One instrumented phase, as it stands right now.
///
/// Durations rather than raw nanoseconds, so a consumer that wants milliseconds asks
/// for milliseconds instead of dividing by a constant it had to look up.
///
/// Five numbers, and no more, because a histogram will give as many quantiles as
/// anyone cares to ask for and an unread one is a column nobody knows the meaning
/// of: how often, what it usually costs, what its bad case costs, what its worst
/// case was, and how much of the session it accounts for. Both consumers show all
/// five. A `p50` sat here unrendered for exactly as long as it took to notice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Phase {
    /// The [`span!`] name, dotted (`stroke.loop`). Owned, because the histograms are
    /// read under a lock the caller must not still be holding when it renders them.
    pub name: String,
    /// How many times the phase ran inside [`Timings::window`].
    pub count: u64,
    /// Arithmetic mean over those runs. Also what [`total`](Self::total) is built
    /// from, which is why the *mean* is the one central figure kept.
    pub mean: Duration,
    /// The bad case — one run in a hundred is at least this slow. The number a
    /// dropped frame is explained by, where the mean is the number a budget is
    /// planned against.
    pub p99: Duration,
    /// The worst single run seen. A shader compile, a tab coming back from a
    /// background throttle, the frame that visibly hitched.
    pub max: Duration,
    /// `mean × count`: the wall time this phase accounts for over the window. What
    /// makes a profile a profile — a 0.3 ms phase run 900 times outranks a 40 ms one
    /// run twice, and only this column says so.
    pub total: Duration,
}

/// Every instrumented phase, and what to read the numbers against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timings {
    /// Sorted by [`Phase::name`], so the dotted prefixes group and the order does not
    /// depend on which phase happened to run first.
    pub phases: Vec<Phase>,
    /// Wall clock since the layer was installed or [`reset`] last ran. The
    /// denominator for every rate and share — see the module note.
    pub window: Duration,
    /// The finest interval the clock behind these numbers can resolve, measured by
    /// watching it until it moves. Shown, not merely known: it is what tells a reader
    /// whether a row reading zero is fast or merely unmeasurable.
    pub quantum: Duration,
}

impl Timings {
    /// How often `name` ran, per second of the window. `None` when the phase is not
    /// instrumented in this build or has not run.
    ///
    /// The end-to-end numbers are exactly this shape: `rate("frame")` is the frame
    /// rate the app actually achieved, and `rate("input.sample")` is how many pointer
    /// reports per second reached the engine — which is the pair the whole latency
    /// question is asked in.
    pub fn rate(&self, name: &str) -> Option<f64> {
        let phase = self.phases.iter().find(|p| p.name == name)?;
        let window = self.window.as_secs_f64();
        (window > 0.0).then(|| phase.count as f64 / window)
    }
}

/// The phase table as text: one indented row per phase and a line saying what the
/// numbers are measured over.
///
/// Here rather than in each benchmark that prints one, because there are two of them
/// (`benches/stroke.rs` and `examples/stroke_bench.rs`) and a column added to one
/// copy and not the other is a column that means different things in two places. The
/// Timing Stats dialog renders its own, and should: a DOM table is not a `Display`.
///
/// **Sorted by total, descending**, unlike [`snapshot`]'s name order. A live table is
/// watched and so wants rows that stay put; this one is read once and the only
/// question being asked of it is which phase is the biggest.
///
/// The share is of [`window`](Self::window) rather than of any parent row, so the
/// rows sum to roughly what the process was doing and whatever is missing is untimed.
/// Nested phases overlap — `stroke.piece` contains the four under it — so the column
/// sums past 100%; the dotted names are what say which rows are inside which.
///
/// Indented by two, because it is a block that belongs under a heading naming the
/// scenario it describes, and every caller has one.
impl std::fmt::Display for Timings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let window = self.window.as_secs_f64();
        let mut phases: Vec<&Phase> = self.phases.iter().collect();
        phases.sort_by_key(|p| std::cmp::Reverse(p.total));
        for p in phases {
            let share = if window > 0.0 {
                p.total.as_secs_f64() / window * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "  {:<18} n={:>5}  mean {:>7.3}  p99 {:>7.3}  max {:>7.3} ms  \
                 total {:>8.1} ms  {share:>5.1}%",
                p.name,
                p.count,
                p.mean.as_secs_f64() * 1e3,
                p.p99.as_secs_f64() * 1e3,
                p.max.as_secs_f64() * 1e3,
                p.total.as_secs_f64() * 1e3,
            )?;
        }
        write!(
            f,
            "  over {window:.2} s of wall clock, clock resolution {:?}",
            self.quantum,
        )
    }
}

/// Read the histograms out of the installed subscriber, or `None` where no
/// subscriber holds a [`layer`] — an embedding that never installed one, or a test.
///
/// One lock, held for as long as it takes to read a quantile off each row and let go
/// — there is nothing to drain and nothing to wait for, which is the point of owning
/// the layer (see [`PhaseLayer`]). Still not for a frame loop: it allocates a `Phase`
/// per row. Twice a second while a dialog is open is what it is sized for.
pub fn snapshot() -> Option<Timings> {
    let mut timings = with_layer(|layer| {
        let phases = {
            let rows = unpoisoned(layer.rows.lock());
            rows.iter()
                .filter(|(_, hist)| !hist.is_empty())
                .map(|(name, hist)| phase(name, hist))
                .collect()
        };
        Timings {
            phases,
            window: layer.clock.delta(
                layer.window_start.load(Ordering::Relaxed),
                layer.clock.raw(),
            ),
            quantum: layer.quantum(),
        }
    })?;
    // Outside the lock: the order is this function's contract, not the map's, and a
    // `HashMap`'s iteration order is not one.
    timings.phases.sort_by(|a, b| a.name.cmp(&b.name));
    Some(timings)
}

/// Start a fresh window: every histogram is emptied and [`Timings::window`] restarts.
///
/// What makes the dialog usable for an experiment rather than only for a session
/// average — clear it, paint one stroke, read the rows that moved. A no-op where no
/// subscriber holds a [`layer`].
///
/// The rows are **dropped**, not cleared in place: a phase that does not run in the
/// new window should be absent from it rather than present with a count of zero, and
/// a table of zeroes is a table nobody reads down.
pub fn reset() {
    with_layer(|layer| {
        unpoisoned(layer.rows.lock()).clear();
        // After the clear, so a sample recorded between the two is counted in the
        // window it actually happened in rather than in one that had not started.
        layer
            .window_start
            .store(layer.clock.raw(), Ordering::Relaxed);
    });
}

/// The finest interval this layer's clock can actually resolve — **measured**, by
/// watching it until it moves, and then cached.
///
/// There is no way to ask for this. Natively it is the TSC's scaled tick; on the web
/// it is whatever quantization the browser applies to `performance.now()`, which
/// varies by vendor, by version and by whether the page is cross-origin isolated. A
/// number this file asserted would be wrong on some browser the day it was written.
///
/// **The step it waits for is a nanosecond of `delta_as_nanos`, not a tick of
/// `raw`**, and the difference is not pedantry: an invariant TSC ticks about three
/// times a nanosecond, so a loop watching the raw counter returns after a step that
/// converts to *zero* nanoseconds — a resolution of "0 ns" that reads as a broken
/// clock when what it means is a clock finer than the unit the histograms are kept
/// in. Nanoseconds are that unit, so nanoseconds are the quantum worth reporting.
///
/// The loop is bounded rather than trusting the clock to move at all: a stopped
/// clock — a hardened browser, a mock — would otherwise spin forever, and answering
/// "at least this coarse" is both true and enough for a caption.
impl PhaseLayer {
    fn quantum(&self) -> Duration {
        let cached = self.quantum_ns.load(Ordering::Relaxed);
        if cached != 0 {
            return Duration::from_nanos(cached);
        }
        // Five attempts, because the *first* step after an arbitrary moment can
        // straddle a boundary and read short; the smallest of several is the quantum.
        const ATTEMPTS: usize = 5;
        // Enough reads to cross a millisecond quantum even on a slow wasm build, and
        // few enough that a frozen clock costs microseconds rather than a hang.
        const MAX_SPINS: usize = 1 << 20;
        let mut best = u64::MAX;
        for _ in 0..ATTEMPTS {
            let start = self.clock.raw();
            let mut spins = 0;
            let mut step = 0;
            while step == 0 && spins < MAX_SPINS {
                step = self.clock.delta_as_nanos(start, self.clock.raw());
                spins += 1;
            }
            if step == 0 {
                // The clock never moved. Say so as a zero rather than as `u64::MAX`
                // dressed up as a duration, and do not cache it — a clock throttled
                // in a background tab can come back.
                return Duration::ZERO;
            }
            best = best.min(step);
        }
        self.quantum_ns.store(best, Ordering::Relaxed);
        Duration::from_nanos(best)
    }
}

/// One histogram, read into a [`Phase`].
fn phase(name: &str, hist: &Histogram<u64>) -> Phase {
    let count = hist.len();
    let mean = hist.mean();
    Phase {
        name: name.to_string(),
        count,
        mean: Duration::from_nanos(mean as u64),
        p99: Duration::from_nanos(hist.value_at_quantile(0.99)),
        max: Duration::from_nanos(hist.max()),
        total: Duration::from_nanos((mean * count as f64) as u64),
    }
}

#[cfg(test)]
mod tests {
    // These are the one place a real elapsed interval is the thing under test: a
    // histogram cannot be checked against a clock that did not move. The suite is
    // native, and the app never blocks the thread it draws on.
    #![expect(
        clippy::disallowed_methods,
        reason = "a histogram test needs the clock to actually advance; native-only"
    )]

    use std::sync::Mutex;
    use std::thread::sleep;

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::registry::Registry;

    use super::*;

    /// Run `f` under a subscriber of this thread's own and return what the
    /// histograms held at the end of it.
    ///
    /// `with_default` rather than `set_global_default`, because the suite is one
    /// process and a global default may be installed once. That makes the layer —
    /// and so the histograms — **per thread**, which is what lets these tests run
    /// beside each other; the two `static`s above are genuinely process-wide, so
    /// nothing here asserts on [`Timings::window`] beyond its sign.
    fn under_a_subscriber(f: impl FnOnce()) -> Timings {
        let subscriber = Registry::default().with(layer());
        tracing::subscriber::with_default(subscriber, || {
            f();
            snapshot().expect("the layer was just installed on this thread")
        })
    }

    fn named<'a>(t: &'a Timings, name: &str) -> Option<&'a Phase> {
        t.phases.iter().find(|p| p.name == name)
    }

    /// **A span's row is its own lifetime, not its lifetime minus its children's.**
    ///
    /// This is the whole model, and the reading every row in the dialog is annotated
    /// with. It follows from `on_close` differencing against the stamp `on_new_span`
    /// left, and from nothing else — no accumulation across enter/exit pairs, no
    /// bubbling into enclosing spans. It is worth a test rather than a comment
    /// because both of those are plausible things for a later hand to add, and either
    /// would leave every number on screen quietly meaning something different: an
    /// outer row would start reading as "time since the innermost thing that last
    /// finished". (It is also the shape `tracing-timing` gets to only by turning off
    /// two of its own defaults, which is how this file used to be written.)
    #[test]
    fn a_span_is_one_row_and_a_child_does_not_eat_the_parent() {
        let t = under_a_subscriber(|| {
            for _ in 0..3 {
                span!("outer");
                sleep(Duration::from_millis(4));
                {
                    span!("inner");
                    sleep(Duration::from_millis(8));
                }
            }
        });

        let outer = named(&t, "outer").expect("the outer span opened a row");
        let inner = named(&t, "inner").expect("the nested span opened its own row");
        assert_eq!(outer.count, 3, "one sample per entry, not per event inside");
        assert_eq!(inner.count, 3);
        // The parent contains the child, so it cannot read as shorter. Sleeps are a
        // floor and never a ceiling, so the assertions are one-sided throughout —
        // a loaded CI box may take 40 ms over a 4 ms sleep and be perfectly correct.
        assert!(
            outer.mean >= inner.mean,
            "outer ({:?}) read as shorter than the inner span it contains ({:?}) — \
             span recursion is on and the child's close reset the parent's clock",
            outer.mean,
            inner.mean,
        );
        assert!(
            inner.mean >= Duration::from_millis(7),
            "inner read {:?} for an 8 ms sleep",
            inner.mean,
        );
        assert!(
            outer.mean >= Duration::from_millis(11),
            "outer read {:?}, which is under the 12 ms it slept for — it is being \
             measured from the inner span's close rather than from its own entry",
            outer.mean,
        );
        // The row keys off the name alone, so a phase entered from two call sites
        // aggregates. That is the documented reading of the table and it is why
        // `total` exists.
        assert_eq!(t.phases.len(), 2, "rows: {:?}", t.phases);
        assert!(t.window > Duration::ZERO);
    }

    /// A layer that remembers the name of every span it was shown — the console
    /// layer's stand-in, since `tracing_wasm`'s does not build off the web.
    #[derive(Clone, Default)]
    struct Seen(std::sync::Arc<Mutex<Vec<String>>>);

    impl Seen {
        fn names(&self) -> Vec<String> {
            self.0
                .lock()
                .expect("no test panicked holding this")
                .clone()
        }
    }

    impl<S> tracing_subscriber::Layer<S> for Seen
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _: &tracing::span::Id,
            _: Context<'_, S>,
        ) {
            self.0
                .lock()
                .expect("no test panicked holding this")
                .push(attrs.metadata().name().to_string());
        }
    }

    /// **[`TimingFilter`]'s two halves partition the world**: every span reaches
    /// exactly one of the two layers, and neither reaches both.
    ///
    /// Both directions matter and they fail differently. A timing span that got past
    /// `logging` would cost a `performance.mark` and a formatted `String` per phase
    /// per frame in the browser, for a devtools entry nobody asked for. A foreign
    /// span that got past `timing` would open a row — every `info_span!` in `iroh`
    /// and `wgpu` — and the dialog would become a list of other people's subsystems
    /// with Stark's phases somewhere in it.
    ///
    /// Through a real subscriber stack rather than by calling `Filter::enabled`
    /// directly, because what has to hold is the arrangement and not the predicate:
    /// `&'static Metadata` only exists at a callsite, and it is the callsite-level
    /// caching (`callsite_enabled`) that would be the subtle thing to get wrong.
    #[test]
    fn the_two_filters_are_exact_complements() {
        let seen = Seen::default();
        let console = seen.clone().with_filter(TimingFilter::<false>);
        let subscriber = Registry::default().with(layer()).with(console);
        let t = tracing::subscriber::with_default(subscriber, || {
            // Braced, because a sample is recorded when the span *closes*: read
            // inside this block, the snapshot would be taken while `ours` is still
            // open and find no row at all.
            {
                span!("ours");
                let _theirs = tracing::info_span!("someone.elses").entered();
                sleep(Duration::from_millis(1));
            }
            snapshot().expect("the layer was just installed on this thread")
        });

        assert_eq!(
            t.phases.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            ["ours"],
            "the histogram layer took a span that is not one of ours",
        );
        assert_eq!(
            seen.names(),
            ["someone.elses"],
            "the logging layer was shown a timing span, or missed an ordinary one",
        );
    }

    /// [`reset`] leaves no counts behind — what makes the dialog usable for an
    /// experiment ("clear, paint one stroke, read what moved") rather than only for
    /// a session average.
    #[test]
    fn reset_empties_the_rows() {
        let subscriber = Registry::default().with(layer());
        tracing::subscriber::with_default(subscriber, || {
            {
                span!("before");
                sleep(Duration::from_millis(1));
            }
            assert!(named(&snapshot().expect("installed"), "before").is_some());

            reset();
            // An emptied histogram is dropped from the snapshot entirely rather than
            // reported with a count of zero: a row that has not run in this window
            // has no distribution to show, and a table of zeroes is a table nobody
            // reads down.
            assert!(
                snapshot().expect("installed").phases.is_empty(),
                "reset left rows behind",
            );

            {
                span!("after");
                sleep(Duration::from_millis(1));
            }
            let t = snapshot().expect("installed");
            assert_eq!(
                t.phases.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
                ["after"],
                "the window after a reset carries only what ran inside it",
            );
        });
    }

    /// Every clock Stark runs on resolves *something*, and [`Timings::quantum`] must
    /// say what — it is the caption that tells a reader whether a row of `0.0 ms` is
    /// fast or merely unmeasurable.
    ///
    /// Bounded rather than asserted exactly, because the answer is the machine's: a
    /// nanosecond off the TSC here, 100 µs or a whole millisecond in a browser.
    #[test]
    fn the_clock_reports_a_resolution_it_can_actually_meet() {
        let q = under_a_subscriber(|| {}).quantum;
        assert!(
            q > Duration::ZERO,
            "the clock reported no resolution at all — either it never moved, or \
             the probe is watching raw ticks instead of nanoseconds again",
        );
        assert!(
            q < Duration::from_millis(5),
            "a {q:?} quantum is coarser than a display frame; every phase in the \
             pipeline would read as 0 or as one whole frame",
        );
    }
}
