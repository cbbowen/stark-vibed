# Stroke module cleanup — the review ledger

An architectural review of `crates/stark-core/src/gpu/stroke`, 2026-08-11, on
master at `6fa95e2` (post-wick-removal). Line references are of that date; the
mechanisms are the durable part. Items are ordered by priority within each
section: one suspected correctness gap, one invariant enforced by convention,
then maintainability and performance.

The module's strengths are worth naming so the cleanup does not erode them:
the plan/kit/run/scratch split keeps GPU-free arithmetic testable without an
adapter, the §6.10 generated mirrors kill a class of host/shader drift, and
the structural-fit relations (`rect_extent` → `snapshot_size`,
`cell_scratch_size` → `cell_geometry`) make overruns unrepresentable rather
than checked. Several suggestions below are that same move applied to places
that don't have it yet.

## Correctness

### 1. Bleed fire windows are missing from the region/tile accounting

**Suspected bug — verify with a test before fixing.**

A piece's tile set is `affected_tiles(segments)` and the region rectangle
follows from it (`dynamics/run.rs::draw`), but the bleed firings are computed
*after* that and enter only `snapshot_size`. So a firing's coverage box sizes
the snapshot scratch (pinned by `the_scratch_is_sized_with_the_bleed_windows_in_it`)
but participates in neither `affected_tiles`, nor `chunk_segments`' bounds
accumulation, nor `region_rect`.

The asymmetry matters because a firing's window is deliberately history-free:
`bleed_fires` walks it back along the crossing segment's own arc, up to one
quantum (`BLEED_TRAVEL_QUANTUM = 0.25` radii) **before the segment's start** —
and for the first segment of a piece or of a live-tail range, that stretch
lies behind everything the region was sized from. The margin available is
`TILE_APRON = 1` texel, so any bleeding brush over ~4 px radius can overrun.
Three consequences, in increasing severity:

- The overrunning writes are silently dropped by the shader's `rdim` bounds
  check — the flux near a piece boundary depends on where the region
  rectangle fell, against the spirit of §6.4's "pure function of canvas
  position" rule.
- A firing can write the region's leading apron texel — the band overlapping
  a tile that is *not* in `coords` and won't be rewritten — so a rewritten
  tile's apron can diverge from its unrewritten neighbour's interior:
  exactly the seam `tests/seam.rs` guards, in a configuration it likely
  doesn't cover.
- A live tail starts at a span boundary each pointer move while the commit
  renders the whole stroke, so the two clip the same firing differently: a
  `preview == committed` break (§1.3) for bleeding brushes, of the same
  family as the two `bleed_fires`/`settle_tangent` already fixed — those
  cured the window's *geometry*, but the *accounting* didn't follow.
  (`bleed_firings_do_not_depend_on_where_the_stroke_was_cut` can't see this:
  it tests the plan, and the clipping happens in the run.)

**Fix direction:** compute the fires before the tile walk and fold their
windows' coverage boxes into `for_each_touched` and the chunker's bounds
accumulation (or conservatively grow a bleeding brush's segment bounds by one
quantum). Pin it first with a preview-vs-commit or seam test at a radius where
the quantum spans a tile boundary — a large bleeding brush whose range cut
falls just past a tile origin.

- [x] Test written first (`a_bleeding_strokes_preview_is_its_commit`,
      tests/dynamics.rs, tol 2 — tightest any dynamics stroke gets). **Honest
      finding: it passed on the unfixed code at 1 level worst**, on a
      designed-hostile case (radius 120 pure-bleed across five tile boundaries).
      Two reasons measured out: a whole render's windows never reach before arc 0,
      so only piece/range boundaries are exposed at all, and a cut lands inside
      the quantum-wide vulnerable band only by tile-phase luck. So the fix is
      accounting hygiene and seam-risk closure, not a visible-artifact cure — the
      deterministic pin is the unit test
      `a_windows_reach_back_is_in_the_tiles_and_the_region` (segments.rs), which
      exercises the exact geometry.
- [x] Fires folded in end to end (`3d86447`): computed once per range in
      `render_dynamic`, measured into `chunk_segments`' per-segment bounds,
      walked into `affected_tiles`, sliced per piece with `after` re-keyed;
      `segment_fits_region` charges a bleeding brush the extra quantum.
- [x] No goldens moved — whole renders were never clipped (windows cannot
      precede arc 0), so committed pixels are untouched by construction.

### 2. "Leases return only after submit" is enforced by convention at three sites

The scratch pool's whole reuse argument is "give back only behind the submit"
— stated in `dynamics/scratch.rs`, re-argued in `dynamics/run.rs`
(`flush`/`submit`), and argued a third time, hardest, for the swept path's
scratch pair (`swept.rs`, where releasing early means another tile's paint
vanishing a frame later). The class already has a recorded bug elsewhere: the
`TilePool` free-list-vs-open-encoder incident ("no live handle ≠ no pending
GPU work"), whose class-level fix is still open.

**Fix direction:** a submit-scope type that owns the encoder *and* the
leases, whose only way to release them is a method that consumes it by
calling `queue.submit`. A call site then cannot return a lease early any more
than it can forget a `Footprint` — the representation can't express the wrong
ordering. Design it to cover both `ScratchPool` leases and the swept path's
pooled tile pair (and ideally the `TilePool` case, retiring the open memory
item).

- [x] Submit-scope type; `DynamicsRun` and `render_swept` migrated (`27c3449`).
      The pool moved up to `stroke/scratch.rs` and its `give` went private; the
      two release paths are `SubmitScope` (owns the encoder *and* the leases,
      releases each tier only in the call that submits — `flush` for piece
      leases, `finish` for run leases) and `Kept` (a lease that outlives its run,
      returned on drop under the borrow argument: a run only ever borrows a
      `ToolState` and submits before returning). `DynamicsRun` shed three fields
      and a flag; the swept path's comment-defended `drop` pair became
      `hold`-then-`finish`.
- [x] The swept path's tile-pair — the sharpest TilePool instance of the class —
      now rides `SubmitScope::hold`, which keeps any drop-releases-to-a-pool
      resource alive past the submit. The *general* TilePool campaign
      (`transform::Recording`, the compositor) is outside this module and stays
      on the pool-free-list memory item; `hold` is the template for it.

## Maintainability

### 3. Binding numbers are triple-maintained

The ~37 raw binding indices (`23` region_resid, `27` brush_src_resid, …) must
agree across the WESL declarations, the layouts in `dynamics/kit.rs`, and the
bind-group entries in `dynamics/run.rs::bind_piece`, with margin comments as
the only map. A mismatch is at least a loud validation error, but this is the
same shape of problem the §6.10 generator already solved for the `Stamp`
lanes and `BAKE_RES`.

**Fix direction:** extend `stark-shaders/build/mirror.rs` to emit
binding-index consts from the shader's own `@binding` declarations. Both host
sides then write `binding::REGION_RESID`, and renumbering becomes a one-file
change in the shader.

- [x] `emit_bindings` in `build/mirror.rs` (`96be1c0`): one `u32` const per
      `@binding` declaration, named for its WESL variable, `@if`-gated ones
      included (the unlinked source keeps them); a name collision is a build
      failure. Listed per module in `build.rs::BINDINGS` — `dynamics` for now,
      the mechanism extends by adding a name.
- [x] `kit.rs` layouts and `run.rs` bind groups both name
      `mirror::dynamics::binding::*`; the margin comments that were the only map
      are gone.

### 4. The λ mapping is defined twice, tied by a stale comment

`dynamics/plan.rs`'s `lambda` closure and `budget.rs`'s `rate_of` closure are
the same clamp expression (`ln(1 − axis).max(−20)`, one divided by
`TAU_PER_PASS`), and the comment binding them says "Mirrors `dynamics.rs`'s
own clamp" — a file that no longer exists (the module split into
`dynamics/`; `budget.rs` has the stale reference twice). The flattener
pricing the very rates the shader runs is load-bearing for the exchange-step
budget, so the agreement should be structural.

- [x] One `budget::lambda` over a shared `ln_keep` core (`faff575`); the plan's
      slots and the flattener's pricing both call it, and the two stale
      `dynamics.rs` references now point at the function.

### 5. Stale doc in `snapshot_scratch`

`dynamics/run.rs::snapshot_scratch`'s doc still explains the size as "+3 for
the sampling margin … +2 because …" — the pre-refactor arithmetic that
`plan.rs::snapshot_size`'s doc explicitly describes having replaced with the
`rect_extent` structural fit. The sentence now documents the exact form the
refactor retired.

- [x] Rewritten to state the structural fit (`faff575`).

### 6. Small items

- [x] `region_rect` returns a named `RegionRect` (`faff575`).
- [x] One `unpoisoned` on the stroke module, with a doc that covers both
      arguments (`faff575`).

## Performance

Context: the latency ledger (`STROKE_LATENCY.md`) already ranks the big
levers — the serialized dispatch chain is semantic, the march is a recorded
dead end, `RESERVOIR_EXCHANGE_STEP` scaling is the remaining structural one.
These are only the gaps the review surfaced. **Measure against
`cargo bench -p stark-core --bench stroke` (baseline first) before adopting
any.**

### 7. Textures are pooled; bind groups and buffers are not

Each fold rebuilds ~8–10 bind groups in `bind_piece` and creates fresh
view/stamp/instance buffers, per piece per pointer move — the very
allocation-rate regime the module's own `ScopedResources` doc identifies as
what OOMs a tab. Since `ScratchPool` deliberately hands back the *same*
textures across folds (newest-match-first for exactly this reason), the bind
groups built over them are cacheable alongside the leases; failing that,
pooling buffers with a size quantum like `SNAPSHOT_QUANTUM` is the same trick
already proven for the snapshot.

- [ ] Bench the bind-group/buffer creation share of a live fold, then decide

### 8. `capture_tool` allocates fresh textures every pointer move

2–3 `create_texture` calls per fold on a mid-stroke tail, destroyed by
`ToolState::drop` one fold later. The `Key` already encodes usage, so these
could come from the `ScratchPool` with the lease held by `ToolState`.

- [x] Pooled as `Kept` leases held by `ToolState` (`27c3449`); the eager-destroy
      `Drop` impl is gone — the pool's return-on-drop is behind the resuming
      run's submit by the borrow argument.

### 9. The round-tip cache is a single entry

`tips.rs` argues one entry from the slider-drag working set — sound for one
user, but two brushes alternating hardness re-bake 256² of `acos`/`exp` plus
two texture uploads *per render*. Two peers painting concurrently in a
collab session (§12), or replay interleaving strokes from different brushes,
hits that every frame. A 2–4 entry LRU keeps the eviction story and removes
the thrash.

- [x] `ROUND_TIPS_KEPT = 4`, move-to-back on hit, evict-front (`27c3449`); the
      slider-drag working set still holds, the alternating-brush thrash is gone.

## Deliberate non-findings

- `dynamics_plan` and `render_swept` are long but linear and single-purpose,
  and their invariants are pinned by the strongest tests in the module
  (`every_slot_field_lands_in_the_lane_the_shader_reads_it_from`, the
  chunker/region agreement tests). Splitting them would scatter the shared
  closures (`lambda`, `bearing`, `common`) that make the agreement between
  slot kinds visible in one screen. Leave them.
- The `TipTooLarge` fallback logging per pointer move is deliberate
  (documented at the call site); no change.
- `noise_cache` growing without bound is fine — its key domain is a small
  enum, documented.
