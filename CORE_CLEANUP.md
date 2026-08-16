# `stark-core` cleanup

A review of [crates/stark-core/](crates/stark-core/) — correctness, performance,
and structure — with what to do about each finding and how you would know it
worked.

> **The identifiers here are `C1`–`C8`, deliberately not `§n.m`.** Design-doc
> section numbers are stable and cited from ~1000 places in the source
> (CLAUDE.md); a work list is neither stable nor citable, and giving it §numbers
> would put entries into a namespace that must keep resolving. Where a finding
> contradicts or extends a design section, it says so and cites it.

Nothing here is a defect in the model. The engine's structural habits are the
reason the review was cheap to do: no `unsafe`, no `TODO`s, four `#[allow]`s
across ~49k lines of `src/`, and invariants enforced by types rather than by
convention —
`TileScope::finish(self)`, `Preview::set_doc` owning the epoch, `minted_layers`
exhaustive on purpose. What follows is the edges.

## Ranked

| | Finding | Kind | Size | Status |
|---|---|---|---|---|
| [C1](#c1-there-is-no-gpu-failure-path) | There is no GPU failure path | correctness | medium | **done** |
| [C2](#c2-non-finite-input-reaches-a-guaranteed-panic) | Non-finite input reaches a guaranteed panic | correctness | small | **done** |
| [C3](#c3-the-fitters-per-sample-cost-is-linear-in-stroke-length) | The fitter's per-sample cost is linear in stroke length | performance | medium | **part** |
| [C4](#c4-the-draw-list-is-rebuilt-from-scratch-every-frame) | The draw list is rebuilt from scratch every frame | performance | medium | **part** |
| [C5](#c5-nothing-ever-retires-history) | Nothing ever retires history | correctness | medium | **corrected — blocked** |
| [C6](#c6-every-integration-test-builds-its-own-device) | Every integration test builds its own device | build health | small | **done** |
| [C7](#c7-engine-is-the-crates-one-god-object) | `Engine` is the crate's one god object | structure | large | open |
| [C8](#c8-smaller-things) | Smaller things | mixed | small | open |

### What landed, and what is left

- **C1** — `GpuContext` carries a `GpuHealth` cell fed by both wgpu callbacks;
  projected as `ObservableState::gpu_failure` and `Engine::gpu_failure()`. The
  async readback reports instead of panicking, so `export`/`export_view` yield
  `Result<RgbaImage>`. `read_rgba8_blocking` still panics on purpose — native,
  test-facing.
- **C2** — every `ViewTransform` mutator is total, through one `commit` funnel;
  non-finite reports are dropped at the fitter and at the session door. The
  `unreachable!` in `spline::m_step` is now genuinely unreachable.
- **C3, part** — the curve is read through a borrow and the fit writes back into
  the candidate (`SplineIndex::fit_into`), removing two `O(m)` copies per solve.
  Measured on `benches/path.rs`: −21% `fit/live/fast`, −13% `fit/batch/fast`, −9%
  spiral, −7% hairpin, −8% `fit/batch/loop`, all p < 0.05.
  **Left:** four `O(m)` allocations per report remain — two candidate grows
  (`grow_rows`) and two arc profiles. Removing them means a small buffer pool on
  the fitter so `adopt` swaps the winner in and returns the loser's storage;
  `Fit` and `adopt` have to change shape for it. Worth doing, and worth measuring
  separately, since the numerics must not move.
- **C4, part** — `culled` walks whichever of the layer or the viewport is
  smaller, so a frame's cost follows the viewport rather than the painting.
  **Left:** the draw list is still rebuilt every frame. Caching it on
  `(doc_revision, preview epoch, visible)` needs a cache on `Engine`, which
  `composite_groups` cannot hold behind `&self` today — it is the same borrow
  problem [C7](#c7-engine-is-the-crates-one-god-object) is about, and is probably
  best done after it rather than around it.
- **C6** — one `GpuContext` per test binary. A fresh `headless_engine` is ~338 ms
  against ~22 ms on an existing device, so this removes about
  (386 − 34) × 316 ms of construction; the warm suite is now 111 s.

- **C5** — **not implemented, and the finding is corrected above.** Trimming the
  log would silently drop early strokes from every saved file and from every
  joining peer's log, because the log *is* the document; retention is gated on
  §8's `checkpoints`, which `io.rs` defers. The leak is also smaller than first
  written: `history`'s snapshots are geometrically spaced, so retention is
  `O(log n)` states rather than `O(n)`.

Suggested order for what remains: **C7** (which also unblocks the rest of C4),
then **C8**, then the rest of **C3**, which is independent and can go any time.
**C5** waits on checkpoints.

---

## C1. There is no GPU failure path

`document/action.rs` justifies `type Error = Infallible` with

> GPU work reports failure via wgpu's device error callbacks, not return values,
> and tile allocation never fails — so applying an action is genuinely
> infallible here (§5).

Nothing installs one. A workspace-wide search for `on_uncaptured_error`,
`push_error_scope` and device-lost callbacks returns zero hits; the only `poll`
calls are in `gpu/readback.rs` and the two benches. The first half of that
sentence describes a mechanism that does not exist.

**Where it lands.** `gpu/readback.rs` turns `map_async` failure and `poll`
failure into `expect` panics, and `Engine::export` / `Engine::export_view` return
`impl Future<Output = RgbaImage>` — no `Result` on the future — so a device lost
mid-export aborts the wasm module.

**Why it matters here more than elsewhere.** This is the one failure the
architecture is best placed to survive. The action log is CPU-side and intact, so
a lost device should be recoverable by rebuilding the GPU stack and replaying —
which is `rebuild_gpu_for` plus a load, both of which already exist. Instead it
is an abort with unsaved work, which is the outcome §1's whole "the document is a
list of actions" claim exists to rule out.

**The fix.**

- A health cell on `GpuContext` — `Arc<ArcSwapOption<DeviceFailure>>` or an
  `AtomicU8` — fed by `Device::set_device_lost_callback` and
  `on_uncaptured_error`, installed at both construction sites
  (`GpuContext::headless` and `from_parts`, so a frontend-supplied device is
  covered too).
- Project it into `ObservableState`, so a frontend can stop dispatching and offer
  to save the log rather than discovering the device is gone by painting nothing.
- Make the readback future return `Result<RgbaImage>`; `export` already returns
  `Result` at the call, so this is the *inner* type changing.

`Action::Error` can stay `Infallible` — a lost device is a device-level fact, not
a per-action one, and threading a `Result` through every `apply` arm would buy
nothing. What has to change is the comment: it should point at the callback that
now exists, rather than at one that never did.

**How you would know.** A test that calls `device.destroy()` (native) or drops
the device between a commit and a render, and asserts the engine reports
unhealthy rather than panicking. Today that test cannot be written.

---

## C2. Non-finite input reaches a guaranteed panic

`ViewTransform::pinch` in `geom.rs` does

```rust
self.zoom = (self.zoom * scale).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
```

`f32::clamp` passes NaN straight through — both comparisons are false — so
`ViewCommand::Zoom { factor: NaN }` or `Pinch { scale: NaN }` poisons `zoom`, and
the next line poisons `center`. `CenterOn`, `Pan` and `SetRotation` are unguarded
too (`NaN.rem_euclid(TAU)` is NaN).

**The path from there to a panic**, each step verified:

1. `screen_to_canvas` divides by the poisoned zoom → NaN canvas position.
2. The frontend hands that in as `InputSample.pos`.
3. `PathFitter::push` accepts it: the zero-step filter is `step < 1e-6`, and
   `NaN < 1e-6` is false. `self.arc += NaN`.
4. `PathFitter::solve` maps samples to curve parameters through `arc`, so `ts`
   is NaN; `span_and_local` clamps NaN to NaN and `u_powers` propagates it into
   the basis weights.
5. `spline::m_step` accumulates NaN into `btb`. `Cholesky::new` returns `None`
   for all 64 ridge escalations, because escalating `lambda` cannot repair a NaN.
6. `unreachable!("ridge-regularized normal equations are positive definite")`.

On wasm that is an abort with unsaved work.

**The tell that the class is already half-recognised.** `document/footprint.rs`
checks `!p.pos.is_finite()` on *control points*, and `Engine::export_view`
refuses a non-finite view with `EngineError::Export("view must be finite")`. Both
sit **downstream** of the poisoning, so the view can be permanently wedged and
export then fails forever with no path back — a guard that reports the state
rather than refusing to enter it.

**The fix**, in the shape CLAUDE.md already asks for ("rule out a class rather
than enumerate its instances"):

- Make `ViewTransform`'s mutators total: a non-finite argument leaves the view
  as it was. `pinch` is the only one that needs care, since it composes three
  things; the rest are one line each.
- `PathFitter::push` drops a non-finite sample the way it already drops a
  zero-length step — same branch, one more condition.

Then `m_step`'s `unreachable!` is true, and `export_view`'s finiteness check
becomes a belt-and-braces assertion rather than the only thing standing between a
NaN and a wgpu validation panic.

**How you would know.** Two unit tests: a `ViewTransform` fuzz over
`{NaN, ±inf, 0}` for every mutator asserting the view stays finite, and a
`PathFitter` fed a NaN sample mid-stroke asserting it fits the same path as if
the sample had not arrived. Both are CPU-only, so they cost nothing in the suite.

---

## C3. The fitter's per-sample cost is linear in stroke length

`path.rs` states, of the windowed solve:

> **The work per sample is constant.** The system is
> `FREE_CONTROL_POINTS × 2` unknowns however long the stroke is, and only the
> samples that can reach those rows take part.

The *solve* is. `spline::m_step` windows the normal equations at
`base = frozen - (ORDER - 1)` and documents exactly why ("an `m × m` matrix
therefore spends nearly all of itself on structural zeros — 160KB of them at 200
control points"). The plumbing around it was not brought along.

**What each `push` actually allocates.** Two `solve`s (at `m` and `m+1`) and two
`mean_error`s. Per solve: `grow_rows` for geom and for attr (two full `m × E`
matrices), `geom.clone()` to build the `CubicBSpline`, `fit_channels`'s `grown`
(another full one), and `m_step`'s `prior.clone()` / `out` (two more) — plus an
`arc_profile` `Vec`. Each `mean_error` clones the candidate's geometry again to
rebuild a spline. That is roughly **16 heap allocations of `O(m)` floats per
pointer sample**, where `m` grows with the stroke.

**The number is already in the tree.** `benches/path.rs` records

> Flattening a fitted stroke is tens of microseconds; fitting the reports that
> produced it is *milliseconds* — 16 ms for `loop`'s 635 samples, against 15 µs
> to flatten the result.

16 ms / 635 ≈ **25 µs per sample**, for a least-squares solve over at most 4×2
unknowns. That is an order of magnitude of matrix plumbing on the interactive
drawing path, and it is the top item in the stroke-latency ledger.

**The fix.**

- Hold the two candidate `(geom, attr)` buffers on the `PathFitter` and mutate
  the window in place. Only rows `frozen..m` can move; the prefix is by
  definition unchanged, which is the property the whole freezing design rests on.
- Change `CubicBSpline::fit_channels` / `m_step` to write into a caller-supplied
  `&mut OMatrix` rather than returning an owned one. The `prior.clone()` at the
  top of `m_step` becomes a no-op when the caller passes the prior as the output.
- `mean_error` can score against the candidate's own control points without
  rebuilding a spline — `CubicBSpline` borrows nothing but the matrix.

Deliberately *not* on the list: changing the growth rule, the reparameterization,
or the arc weighting. This finding is about allocation, and any change that moves
a fitted control point is a different change with different goldens.

**How you would know.** `cargo bench -p stark-core --bench path -- --save-baseline main`
before, `-- --baseline main` after. The fit is deterministic and
`tests/stroke.rs` already pins preview-vs-commit, so a change that alters output
fails rather than silently re-blessing.

**Also fix the comment.** "The work per sample is constant" should say what is
constant (the system being solved) and what is not, or become true. See the note
under [C8](#the-documentation).

---

## C4. The draw list is rebuilt from scratch every frame

`Engine::composite_groups` → `composite_stack` → `layer_items` allocates a fresh
`Vec<CompositeItem>` per layer and clones a `TilePairHandle` per visible tile,
per layer, **per frame**.

`TilePairHandle::composite_bg` fixed the wgpu-object churn — and its doc comment
sizes the problem exactly right:

> The visible tile count scales as 1/zoom², so a zoomed-out multi-layer document
> was creating ~10⁵ of them a frame.

The bind groups are cached now. The `Arc` traffic and the `Vec`s are not: the
same ~10⁵ tiles still pay an atomic increment on the way into the list and a
decrement on the way out, every frame, plus a few hundred allocations for the
`Vec`s holding them.

Two independent levers:

**Cache the list.** It changes only with the committed document, the layer
properties, the preview, or the visible rect — and the first two counters already
exist for exactly this kind of question (`doc_revision`, `Preview::epoch`, both
documented as "what a frontend keeping a rendered stand-in watches"). Keying a
cached `Vec<CompositeGroup>` on `(doc_revision, epoch, visible)` makes a
pan-free, edit-free frame free. Failing that, reuse the `Vec`s off an arena on
`Compositor` (`clear()` + refill), which amortizes the allocation even when the
key moves every frame during a pan.

**Stop walking the whole tile map.** `layer_items` iterates `tiles.map()` in full
and filters against `visible`:

```rust
LayerContent::Paint(tiles) => tiles
    .map()
    .iter()
    .filter(|(coord, _)| visible.is_none_or(|r| r.contains(**coord)))
```

`rpds::HashTrieMap::iter` walks the whole trie, so a large painting scrolled to
one corner does `O(painted)` work to keep `O(visible)` tiles — every frame, every
layer. A coarse per-layer index (a `HashTrieMap<SuperCoord, ...>` bucketing
`TileCoord >> k`, or a sorted `Arc<[TileCoord]>` rebuilt on tile change) makes
the cost follow the viewport, which is what §6.3's cull is *for*.

**How you would know.** The cull already has tests
(`the_cull_keeps_every_tile_the_viewport_shows`) that pin correctness, so this is
purely a timing change. A criterion bench over `composite_groups` on a synthetic
20-layer, 10k-tile document, at zoom 0.05 and at 1.0, would give the before/after
and would be a useful gate to keep.

---

## C5. Nothing ever retires history

> **Corrected while implementing.** The finding as first written proposed a
> `Timeline::trim` mapping to `history::forget_actions`. **That would silently
> corrupt saved documents**, and the severity of the leak was overstated. Both
> halves are set out below, because the original is the more obvious reading and
> someone will have it again.

`history::forget_actions` exists upstream and is called nowhere in the workspace.
`LinearTimeline` keeps every action for the session's life. CLAUDE.md says:

> **`DocState` is cheap to clone** … Tiles are copy-on-write, so history
> retention drives GPU memory reclamation for free.

That is true, and it only reclaims if something *retires* history. Nothing does.

### Why the obvious fix is unsafe

**The log is the document** (§1, §8). `Engine::save_bytes` writes
`timeline.clone_actions()` and the file is replayed onto an *empty* document;
`engine/collab.rs` sends that same log to a joining peer. `forget_actions(k)`
folds the oldest `k` actions into `History`'s initial state and drops them from
`actions()` — which is exactly what the save format and the join path read.

So a trimmed session would save a file **missing its early strokes entirely**,
and hand a joining peer a log that can never converge. Not a degraded save: a
wrong one, and one nothing on screen would reveal, because the in-memory state
still shows the whole painting.

The initial state a trim folds into is pixels, and the only place the format has
for pixels is `checkpoints`, which §8 lists and `io.rs` defers ("the advisory
raster `checkpoints` of §8 are deferred"). **Retention is therefore gated on
checkpoints**, not on a `Timeline` method — the method is the last step, not the
first.

### And the leak is smaller than it looked

Worth correcting because it changes the priority. `history` keeps its snapshots
**geometrically spaced**, so the retained `DocState`s are `O(log n)` and not
`O(n)` — and each pins only the tiles that later changed, since copy-on-write
shares the rest. The action log itself does grow linearly, but a `CommitStroke`
is a fitted control-point list: hundreds of bytes, so tens of thousands of
strokes are tens of megabytes of CPU memory.

That is a real cost for a long session on a tab with a hard ceiling, and it is
still worth fixing. It is not the unbounded GPU growth the original text implied.

### What to do instead

1. Land `checkpoints` (§8) — a cached raster of the folded-in initial state, and
   the format's own answer to this question.
2. *Then* add `Timeline::trim`, defaulting to a no-op so `ReplicatedTimeline`
   declines by construction, exactly as it declines `seek`.
3. Say out loud in §5 what it costs: undo depth becomes bounded by memory. With
   checkpoints the *file* still opens to the same picture, so nothing is lost
   from the document — only from the live undo stack.

**How you would know.** A round-trip test is the one that matters and the one
that would have caught the unsafe version: commit N strokes, trim, save, load
into a fresh engine, and assert the picture is identical. A pool-capacity test
measures the win; this measures whether the win cost the document.

---

## C6. Every integration test builds its own device

392 tests across 34 binaries; 386 `engine_or_skip*` call sites. Each goes
`headless_engine` → `Instance::new` → `request_adapter` → `request_device` →
`Engine::new_with_color_space`, which by the engine's own accounting means

> recompiling ~19 shaders and ~30 pipelines and re-decoding every image the app
> has already decoded once.

Per test.

`Engine::new_sharing` already shares exactly that set, and its doc comment
already makes the argument — it was written for the brush editor's preview
canvas, but the test suite is the same bargain at 386×. A per-process donor
engine behind a `OnceLock`, with `new_sharing` for the per-test engine, would cut
the suite's dominant fixed cost without changing a single assertion.

**Two opt-outs to keep standalone**, and they should be named in the harness so
the reason survives:

- `tests/tile_pool.rs` — it asserts pool counts, and `new_sharing` shares the
  pool.
- Anything asserting a non-Oklab color space, since the space is fixed at
  construction and a rebuild is what `new_sharing` explicitly does not do.

**Why it is worth doing beyond the wall clock.** "The test suite is slow — run it
once" is currently a rule contributors have to remember, and it costs real
diligence: it is why counts get grepped from one redirected run rather than
re-queried. Making the suite cheap removes the rule rather than documenting it
harder.

**How you would know.** Time `cargo test --workspace` before and after, once each
(the rule still applies while it is true).

---

## C7. `Engine` is the crate's one god object

4,506 lines across `engine/`, 116 methods (66 public), 16 fields, `&mut self`
throughout. The split by subject is good and the module doc is honest that it is
a split of the *file*, not the type:

> A child module can reach a private field of a struct its parent defines, so
> this is a division of the *file* and not of the type.

That was the right call for readability. What has accumulated since is that the
single borrow is deforming the API:

- `engine/render.rs` clones selection masks so "the borrow of `doc` — and with it
  of `self` — ends before the compositor is borrowed mutably."
- `engine/live.rs::flush_live` needs a comment naming which fields are disjoint
  before it can call `rebuild`.
- `Engine::export` carries a 15-line comment explaining why it cannot be an
  `async fn`: `&mut self` would be held across the await, and a frontend taking
  that borrow from a shared cell would panic with `AlreadyBorrowedMut`. That is a
  frontend-visible contortion caused entirely by the single borrow, and it is
  documented as a design constraint rather than as a symptom.

**Not** a proposal to split `Engine` into pieces the caller assembles — the
"`Engine` owns everything and is the only entry point" promise is worth keeping.
The proposal is to give the two clusters that genuinely do not share mutable
state their own types:

| Unit | Holds | Reads | Writes |
|---|---|---|---|
| presentation | `compositor`, `compositor_pipeline`, `environment` | `&DocState`, `&Session` | its own attachments only |
| document | `apply`, `timeline`, `authoring` | commands | `DocState` |

`render_view` then becomes a function over
`(&mut Presentation, &DocState, &Session)`. The mask clone goes away, the
disjointness comments go away, and `export` can be an ordinary `async fn` because
the future borrows the presentation unit and not the engine.

§7's actor target pushes the same way, and this is the part of §7 that pays
before the actor exists — which matters, since §7 is currently kept as a design
whose justification is entirely prospective:

> If the actor is ever abandoned, §4's discipline loses its main justification
> and should be revisited rather than quietly kept.

**Large, and worth staging.** The presentation unit alone is most of the win and
touches `engine/render.rs` plus the two `Attachments` call sites. The document
unit can wait.

---

## C8. Smaller things

### `rpds::Vector<Layer>` buys nothing

`DocState::layers` is a persistent vector, but every structural op rebuilds it
element-by-element: `insert_at` and `remove_in` both `push_back` the whole stack,
allocating trie nodes rather than one buffer. Clone cost is identical to
`Arc<Vec<Layer>>` (one refcount bump); edit cost is worse.
`Arc::make_mut(&mut layers).insert(i, layer)` is cheaper and simpler, and the
`IN_RANGE` `expect`s that exist to explain `Vector::set`'s `Option` go with it.

`HashTrieMap` for the tile map genuinely earns its keep — thousands of entries,
real structural sharing between versions. `Vector` for dozens of layers does not.

### The crate root has none of the discipline `document/` and `gpu/` have

Both of those declare `pub(crate) mod` plus a curated `pub use` list, each with a
doc comment saying *why* — `gpu/mod.rs`:

> what leaves this module should be a decision, not a consequence of how a file
> happened to be split.

The root is a flat namespace of 20 files with four unrelated roles: pure-CPU
geometry (`geom`, `path`, `spline`, `tow`, `assist` at 1,846 lines, `guides` at
1,841, `noise`), session and presence (`session`, `peer`, `presence`, `command`),
persistence (`io`, `assets`, `content`, `image`), and color (`color`,
`colorspace`, `gradient`). Grouping them with the same curated surface applies
the crate's own stated rule to the one place it is not applied — and makes the
wasm-size story legible, since `assist` + `guides` are ~3,700 lines of CPU-only
code a headless replay never reaches.

### `DocState`'s content setters repeat one shape five times

`set_matte_rect`, `set_matte_region`, `set_matte_paint`, `set_filter` and
`set_layer_blend` each spell out
`map_layer(id, |l| match &l.content { … => Layer { content: …, ..l.clone() }, _ => l.clone() })`.
Each is a place a new `LayerContent` variant must be visited, and the "which
content kinds accept this edit" decision is made five times. One helper that
takes that decision as its argument would make the answer a parameter rather than
a pattern.

### The documentation

Worth saying once, because it cuts both ways.

The comments are why this review was cheap: they record *why*, they name the bug
that motivated the rule, and several — `Pooled` keeping its view, the trim's
quarantine, "where the layer's composite params go" — are better than any design
doc that could have been written for them. **Do not cut them.**

The risk is drift, and [C3](#c3-the-fitters-per-sample-cost-is-linear-in-stroke-length)
is a live instance: "the work per sample is constant" is asserted in a doc
comment, is false, and nothing checks it.

The antidote is already in the codebase, in several places — a comment turned
into an assertion:

- `a_trim_never_drops_below_the_epochs_peak_demand`
- `a_scope_hands_its_scratch_back_as_it_goes`
- `tests/seam.rs`
- `a_census_slot_belongs_to_the_source_that_indexes_it`

Extending that habit to the **performance** claims specifically is the cheap fix:
the fitter's constant-work claim, `composite_stack`'s "an ordinary document is
one group", and the pool's working-set claim are all measurable, and none of them
is measured.
