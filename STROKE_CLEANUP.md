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

- [ ] Write the failing test (seam or preview==commit, bleeding brush,
      window crossing the region's leading edge)
- [ ] Fold fires into the tile walk and chunk bounds
- [ ] Re-bless any goldens the fixed flux moves (fix the model, no fudge)

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

- [ ] Submit-scope type; migrate `DynamicsRun` and `render_swept`
- [ ] Extend to `TilePool` acquisitions recorded into open encoders

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

- [ ] Generate `@binding` consts per shader module
- [ ] Migrate `kit.rs` layouts and `run.rs` bind groups to the names

### 4. The λ mapping is defined twice, tied by a stale comment

`dynamics/plan.rs`'s `lambda` closure and `budget.rs`'s `rate_of` closure are
the same clamp expression (`ln(1 − axis).max(−20)`, one divided by
`TAU_PER_PASS`), and the comment binding them says "Mirrors `dynamics.rs`'s
own clamp" — a file that no longer exists (the module split into
`dynamics/`; `budget.rs` has the stale reference twice). The flattener
pricing the very rates the shader runs is load-bearing for the exchange-step
budget, so the agreement should be structural.

- [ ] One shared `lambda`/`rate_of` in `budget.rs`; fix the two stale
      `dynamics.rs` references

### 5. Stale doc in `snapshot_scratch`

`dynamics/run.rs::snapshot_scratch`'s doc still explains the size as "+3 for
the sampling margin … +2 because …" — the pre-refactor arithmetic that
`plan.rs::snapshot_size`'s doc explicitly describes having replaced with the
`rect_extent` structural fit. The sentence now documents the exact form the
refactor retired.

- [ ] Rewrite the doc to point at `rect_extent`/`snapshot_size`

### 6. Small items

- `region_rect`'s 5-tuple return (`segments.rs`) — every caller destructures
  five positional fields; a named struct reads better.
- `unpoisoned` is duplicated verbatim with its doc essay in `tips.rs` and
  `dynamics/scratch.rs`; one copy in a shared spot ends the drift risk
  between the two arguments.

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

- [ ] Pool the tool-state copies (lease lifetime moves into `ToolState`)

### 9. The round-tip cache is a single entry

`tips.rs` argues one entry from the slider-drag working set — sound for one
user, but two brushes alternating hardness re-bake 256² of `acos`/`exp` plus
two texture uploads *per render*. Two peers painting concurrently in a
collab session (§12), or replay interleaving strokes from different brushes,
hits that every frame. A 2–4 entry LRU keeps the eviction story and removes
the thrash.

- [ ] Small LRU keyed by hardness bits

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
