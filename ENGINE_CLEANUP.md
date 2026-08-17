# Engine cleanup

Findings from a review of `crates/stark-engine/src` (2026-08-16). Seven items, in
two tiers: three measured, four structural.

Nothing here is producing a wrong pixel today, and the tree was clippy-clean with
no TODOs when this was written. The tier-one items are a real performance problem
that the existing gate cannot see; the tier-two items are places where a rule this
crate states clearly is upheld by convention rather than by construction — which is
the specific failure the "rule out a class rather than enumerate its instances"
convention exists to prevent.

**These are findings, not decisions.** Each carries a suggested direction, but the
measurement is the part worth keeping: the direction is one reading of it.

## How the measurements were taken

A standalone binary against `stark-engine` with `default-features = false`, driving
`PathFitter::push` over a synthetic sine stroke and timing the push loop alone
(best of three, `--release`). Reproducing it needs nothing from this repo but the
public `path` API — `PathFitter::with_tolerance`, `push`, `path` — so it is a
handful of lines rather than a harness worth checking in. What *is* worth checking
in is §F2's benchmark case, which measures the same thing inside the gate.

Figures are one machine, one run; the shares matter, not the absolute nanoseconds.

---

## Tier one — measured

### F1. The solver window is delimited in arc length, so fit cost scales with report rate and with zoom

`path.rs:565` (`solve`), `path.rs:619`–`625` (the window), `path.rs:384` (`push`),
`path.rs:83` (`KNOT_SPACING`)

`PathFitter::push` runs two candidate solves per pointer report — `solve(m)` and
`solve(m + 1)` — each fitting geometry and pen channels separately, and each
candidate then scored by `mean_error`. Six passes, all over `pts[lo..]`.

The count of control points in flight is properly bounded: `FREE_CONTROL_POINTS` is
3, and the freezing rule is what the whole `FrozenHead` machinery rests on (§6.2).
The count of **observations** fed to those solves is not bounded at all. `lo`
advances while `param(pts[lo].arc) < cutoff`, so the window is roughly five spans of
*arc length* — and a span is `KNOT_SPACING × tolerance`, i.e. `64 × tolerance`
canvas px.

So the number of reports re-solved on every report is the product of two things the
engine does not control: how densely the digitizer reports per canvas px, and how
far out the view is zoomed. Each scales the window linearly, which makes each report
linearly more expensive and the whole stroke quadratic.

**One fixed 24 000 px stroke, varying report density.** Knot count barely moves;
total time grows 25×.

| reports | knots | ns/report | total ms |
|--------:|------:|----------:|---------:|
|   1 000 |   335 |     9 874 |      9.9 |
|   2 000 |   373 |    12 613 |     25.2 |
|   4 000 |   386 |    18 271 |     73.1 |
|   8 000 |   398 |    31 068 |    248.5 |

Each doubling of the report rate costs roughly 3× the total. A 200 Hz tablet and a
1000 Hz pen drawing the identical stroke are not in the same complexity class — and
on the web, coalesced pointer events deliver exactly that high rate.

**Identical stroke, identical 4 000 reports, only the declared tolerance moves** —
which is what a frontend sets from the zoom level (`GestureCommand::Start`).

| tolerance | knots | ns/report | total ms |
|----------:|------:|----------:|---------:|
|      0.25 |   743 |    15 047 |     60.2 |
|      0.5  |   386 |    19 250 |     77.0 |
|      1.0  |   200 |    27 769 |    111.1 |
|      2.0  |   102 |    45 820 |    183.3 |
|      4.0  |    58 |    74 537 |    298.1 |
|      8.0  |    37 |   110 800 |    443.2 |
|     16.0  |    24 |   151 436 |    605.7 |

This one runs backwards from intuition: a coarser tolerance produces **31× fewer
knots** and costs **10× more CPU per report**. At tolerance 16 a single pointer
report costs 151 µs, so a burst of ten coalesced samples is a whole frame — and
`MAX_TOLERANCE` is 64, four times past the right-hand end of that table.

It also explains the row `engine/mod.rs` already annotates. `input.fit` is described
there as "the one phase that grows with stroke length rather than with the tail",
attributed to the fitter re-solving its unfrozen prefix. The prefix is bounded; the
*window* is not, and that is the term that grows.

**Suggested direction.** Bound the observation count, not only the control-point
count — and the argument for why that is safe already exists in this file.
`arc_weights` was introduced to turn `Σ residual²` over reports into `∫ residual² ds`
over the stroke, precisely so report density stops being able to outvote geometry
(see its header, and the pen-release measurements in it).

That property is what makes decimation sound here. Subsampling the window to a fixed
budget of observations along arc, with the trapezoid weights recomputed for the
survivors, approximates the same integral the solve already minimizes. Per-report
cost becomes O(1) in both density and tolerance, and the quantity the fit is defined
against is unchanged. This is one bound rather than a rewrite.

Note this is *not* the report rejection `arc_weights`' header rules out. That
argument is about dropping the pen-release reports, where a threshold only decides
which contaminated report gets through. Decimation with weights preserved is a
different operation: it approximates the integral rather than discarding evidence
from it.

### F2. The path benchmark holds fixed both variables that drive F1

`benches/path.rs:33` (`cases`), `benches/path.rs:55` (`fit`)

The fit gate runs four recorded strokes — hairpin, spiral, loop, fast — at their
captured native density, always at `DEFAULT_TOLERANCE`, reporting throughput in
input samples. Every one of those choices is well argued in the file, and together
they make the gate blind to F1.

Report density is fixed by the recording; tolerance is fixed by the constant. Since
the metric is per-sample, a cost that grows with either axis reads as a flat number.
A user on a high-rate pen, or simply zoomed out, pays a multiple of what the gate
measures, and no regression in that multiple could fail it.

**Suggested direction.** Add two sweeps beside the existing cases: one resampling a
single recorded stroke to several report densities, one running a fixed stroke
across the tolerance range up to `MAX_TOLERANCE`. Both are a few lines on the
harness already there, and both would have surfaced F1 as a rising per-element
figure rather than needing to be found by hand.

### F3. Each report copies the whole control polygon four times, plus four arc profiles

`path.rs:875` (`grow_rows`), `path.rs:911` (`arc_profile`), `path.rs:866` (`Fit`)

Secondary to F1, and independent of it. `grow_rows` returns an owned matrix on both
paths — `rows.clone()` when the polygon is already long enough, `from_fn_generic`
when it is not — and `solve` calls it for geometry and for channels, twice per
report. Four full-polygon allocations and copies per pointer sample, with both
candidate `Fit`s alive at once.

Alongside them, `arc_profile` allocates `spans × 4 + 1` floats and
`extend_from_slice`s the settled prefix into it: twice inside `solve`, twice more
inside `mean_error`. The *evaluation* is correctly incremental — it resumes from
`keep` — but the allocation and the prefix copy are not.

Both quantities grow with stroke length, and the frozen prefix is by definition
never revised, so neither copy buys anything.

**Fixed 3 px report spacing, growing stroke length.** Density and window constant;
only the polygon behind them grows.

| reports | knots | ns/report |
|--------:|------:|----------:|
|     500 |    26 |    23 357 |
|   2 000 |   100 |    26 851 |
|   8 000 |   398 |    30 281 |
|  16 000 |   795 |    31 592 |

About 10 ns per knot per report — roughly a third of the cost at 795 knots.

**Suggested direction.** Keep the two candidate polygons as reusable buffers on
`PathFitter` and let `solve` write into them, the way `fit_into` already writes into
`geom` rather than returning a fresh matrix — its comment makes exactly this
argument one level down and then stops at the solve. Have `arc_profile` append into
a retained buffer instead of building a fresh `Vec` over the settled prefix.

---

## Tier two — structure and consistency

### F4. The tile pool contradicts `unpoisoned`'s stated rule, inside one file

`lib.rs:99` (`unpoisoned`), `gpu/tile.rs:714` and `:771` (panic),
`gpu/tile.rs:222` and `:753` (recover), `assets.rs` ×6, `gpu/registry.rs:117`

`crate::unpoisoned` states the rule and its reasoning at length: every mutex in this
crate guards a cache, a free list or a tally, so propagating a poison as a panic
"turns one thread's failure into a dead renderer, which is a worse answer than a
cold cache."

The tile pool then does both. `Drop for GpuTex` recovers the guard, with a comment
explaining that dropping the texture instead would leak; `resident_bytes` recovers.
But `acquire_tex` — the hot path — and `free_count` both
`.expect("tile pool poisoned")`. Same lock, opposite policies, forty lines apart.
`AssetStore` does the same at six sites and `Registry` at one. Eleven sites bypass a
helper that exists and is used eight times.

The consequence is the one the helper's doc rules out: a panic anywhere under the
pool lock makes every later acquire panic, so one thread's failure becomes a
permanently dead renderer.

**Reachability, stated honestly.** The first reading of this was that a GPU
out-of-memory in `create_pooled` would panic inside the critical section and poison
the lock. That is wrong: `install_callbacks` (`gpu/context.rs:233`) installs a
non-panicking `on_uncaptured_error` that routes failures to `GpuHealth`. So this is
a consistency defect with a low-probability failure mode, not a live bug — worth
fixing because the rule already exists, not because something is broken.

**Suggested direction.** Route all eleven sites through `unpoisoned`. Separately
worth a look: `create_pooled` calls `device.create_texture` while holding the pool
lock, which puts a device call inside the critical section on the miss path.

### F5. The submit-ordering rule is carried by five renderers agreeing, not by the pool

`gpu/submit.rs:114` (`TileScope`), `gpu/stroke/scratch.rs:262` (`SubmitScope`),
`gpu/tile.rs:711` (`acquire_tex`)

`TileScope` and `SubmitScope` state the rule well — a pooled resource handed back
before its commands are submitted is the next consumer's, and since `TilePool`'s
trim can `destroy()`, the same mistake now reaches a dangling view rather than
merely wrong pixels. Every current consumer uses one: fill, both merges, selection,
both transforms, and the stroke path.

But `TilePool::acquire_tex` hands out a `TexHandle` whose `Drop` returns it to the
free list unconditionally. Nothing about the pool's API knows a scope exists. A
sixth renderer written without one compiles, runs, and reintroduces a defect that
shows only on large operations and that no test names.

**Suggested direction.** This is the crate's own "rule out a class rather than
enumerate its instances" applied to the one rule that is currently enumerated.
Making the acquire go *through* a scope — so a handle cannot be obtained without one
to hold it — would move the guarantee from five files agreeing to a signature. If
that is too invasive, a debug-only assertion that a released handle's scope has
submitted would at least make the class testable.

### F6. `ObservableState::guides` is the un-shared twin of `Layers`

`engine/mod.rs:153` (`Layers`), `:362` (the field), `:1501` (the clone),
`guides.rs:142` (`PerspectiveGuide::name`)

The layer list got the full treatment: an `Arc<[LayerInfo]>` newtype, a `PartialEq`
with an `Arc::ptr_eq` fast path, and memoization behind `projected_layers` keyed on
counters that already exist. The reasoning is written out and it is right.

The guide list, projected from the same `observe()` after the same commands at the
same pointer rate, is a plain `Vec<PerspectiveGuide>` cloned on line 1501 and
compared element-wise by the derived `PartialEq`.

What makes this worth naming is that the argument was already made and applied one
level too shallow: `PerspectiveGuide::name` is an `Arc<str>` whose doc cites
`LayerInfo::name` and "cloned into `ObservableState` after *every* command". The
per-guide allocation was removed; the per-list one was not.

**Suggested direction.** Give it the same shape as `Layers`. Small in absolute
terms, but it closes a gap between a stated standard and its application.

### F7. Five accessors sit on one O(layers) tree walk

`document/state.rs:269` (`locate`), `:748` (`with_layers`),
`engine/live.rs:595` and `:649`

`locate` is honest about being "the one search this module does", and `layer`,
`contains_layer`, `carrier_of` and `site_of` are all projections of it. `map_layer`
walks too. The projection path has already been de-quadratified once — the
`merge_down` comment records 79 µs at 60 layers against 1.3 µs at 4, fixed by
reading answers off the traversal instead of searching per row.

The apply, patch and preview paths still search per operation; the live fold does
roughly six walks per in-flight gesture per frame. At today's layer counts this is
nothing, and it may never matter. **The shape is flagged, not a measured cost.**

**Suggested direction, if it ever does matter.** The structural answer has an exact
precedent in the same file: `bounds` is derived, private, and written only by
`with_layers`, on the stated grounds that a struct literal setting one without the
other would be a document disagreeing with its own paint. An `id → LayerSite` index
maintained in that same single writer would make lookup O(1) and could not drift,
for the reason `bounds` cannot.

---

## What not to change

Four things checked and deliberately left alone, including one that looks like an
obvious cleanup and is not.

- **Don't split the large files.** `segments.rs` is 2 598 lines with ~130 before its
  first test module; `peer.rs` is 1 011 with 839 of them tests. The counts are prose
  and coverage, not complexity, and splitting on size would cut real seams apart.
- **The prose is the asset.** 38% of the crate is comment (17 450 lines against
  26 103 of code), and it argues rather than describes — nearly every constant
  carries the measurement or the reasoning that set it. It is what makes the code
  reviewable at all.
- **The cache-key discipline.** `DrawKey`, `projected_layers` and `Preview::epoch`
  are keyed on counters something else already maintains, with no invalidation call
  anywhere. A cache that must be told is one a new path can forget to tell.
- **The GPU health path.** `on_uncaptured_error` and the device-lost callback route
  failures to `GpuHealth` and out through `ObservableState::gpu_failure`, so a dead
  device is reported rather than discovered by an abort (§5). This is the design F4
  should be brought in line with.

## One note for the actor move

§7 still has the asynchronous actor loop ahead of it, and `Engine` is the type that
has to survive it: around twenty fields, `impl` blocks across six files, nineteen
`&mut self` public methods against forty-one shared ones. The file split is clean
and the `EngineShared` / `Authoring` / `Preview` groupings have already absorbed
most of what would otherwise be loose state.

The detail worth knowing before that step: `layer_cache` is a `RefCell`, taken
deliberately so `observe` can memoize behind `&self`. That makes `Engine` `Send` but
not `Sync` — fine for an actor owned by a single task, and it rules out handing out
`&Engine` across threads. Worth deciding on purpose rather than discovering from a
trait bound.
