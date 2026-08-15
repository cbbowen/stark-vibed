# `gpu/stroke` — the architectural cleanup ledger

Reviewed 2026-08-15 on branch `gpu-cleanup` (`8952270`), across all 11 files of
`crates/stark-core/src/gpu/stroke` (~8.5k lines, of which ~2.4k are the in-file test
modules), plus the two things it is joined to at the seam: the generated shader
mirrors (`stark-shaders/build/mirror.rs`, §6.10) and `gpu/tile`'s pool.

**Nothing here landed.** This file records what the review found and what it would
cost; every finding is open. Symbol names are the durable part.

This is `COMPOSITE_CLEANUP.md`'s review one directory over, and it finds the mirror
image. There the module's hardest rules were *argued in prose and consumed
positionally*; here most of them are already structural — `SubmitScope` makes a lease
release unforgeable, `dynamics_setup` answers from the brush alone so no two renders
can disagree, `Slot::pack` is pinned field-by-field against the lane the shader reads.
What is left is a shorter list, and it clusters in the places that discipline has not
reached: the GPU-facing half (`kit`/`run`), and the three or four types that carry more
than one subject.

## The one-sentence summary

**The models are separated and tested; the bindings and the boxes are not.** Every
number a brush becomes is CPU float math with a test on it, but the layouts and their
bind groups are two hand-maintained mirrors of one shader-declared fact (§4), one
segment's coverage box is computed four times by three functions that must agree (§3,
§5), and `Segment` carries a bleed window's rates that the slot then zeroes back out
(§2). One correctness hazard is live-but-unreached (§1), and it is the very class
`scratch.rs`'s module doc was written against.

| § | Finding | Kind | Status |
|---|---|---|---|
| 1 | Base tiles reach the pool free list under an open encoder | latent, ordering-held | open |
| 2 | `Segment` is a sweep and a paint parcel in one struct | god-struct, fabricated windows | open |
| 3 | Dispatch rects computed twice; the fit is an assert, not a max | duplication + wasm panic | open |
| 4 | Layouts and bind groups are two hand-written mirrors | ~350 lines, drift class | open |
| 5 | Three answers to "what region do these sweeps need" | invariant by convention | open |
| 6 | The swept path serializes on one shared scratch pair | 2N passes, no overlap | open, measure |
| 7 | 40–60 bind groups per fold on the live tail | allocation rate | open |
| 8 | `segments.rs` is five subjects; `budget.rs` is three | 2.6k-line file | open |
| 9 | Every kit field is `pub(super)`; the renderer has no boundary | encapsulation | open |
| 10 | Smaller correctness and clarity items (four) | mixed | open |
| 11 | The `bake` dispatch shape has never been measured | measure first | open |

Suggested order: **§1, §2, §3, §6, §8, §4**. §1 is one line and closes a known hazard
class. §2 unlocks §3 and §5. §4 is the biggest payoff and wants its own round.

---

## 1. Base tiles reach the pool free list under an open encoder — open

The `pool-free-list-vs-open-encoder` class, live in the dynamics path.
`render_dynamic` walks its pieces at `run.rs`:

```rust
map = run.draw(&map, &segments[piece], &piece_fires, !capture && i == last);
```

`DynamicsRun::draw` records `composite_region`, which **samples `map`'s tiles**, into
the open encoder; it then returns a new map. The assignment drops the old map, and
every `TilePairHandle` the new map replaced goes back to `TilePool`'s free list — with
the commands that read it still only recorded.

Consecutive pieces share the tiles around their cut (the apron guarantees it, which is
`affected_tiles`' whole reason for growing a segment's box), so this is the common
case for any stroke long enough to be chunked, not a corner.

`tile.rs`'s `PoolInner::trim` already knows about this hazard — "a slot that drops
while an unsubmitted encoder still names its view reaches this free list" — and guards
**destruction** behind the epoch. It does not, and cannot, guard **reuse**: the next
`acquire_tex` of the same format is free to hand the texture straight back out.

It does not fire today, and the reason is not a rule. The next statement executed is
the following iteration's `self.scope.flush()`, and nothing acquires from `TilePool` in
between. That is a one-statement margin holding an invariant the rest of this module
makes structural — `swept.rs` holds its scratch pair across exactly this boundary with
`scope.hold(scratch)`, and `scratch.rs`'s module doc spends five paragraphs on why
"no live handle" is not "no pending GPU work".

**What to do.** Hold the map the piece read across the piece's own submit, the same
way the swept path holds its scratch:

```rust
// In `DynamicsRun::draw`, once the composite has been recorded against `base`:
self.scope.hold(base.clone());
```

A `TileMap` clone is an `rpds` map of `Arc` handles, so this costs a refcount per tile
and nothing else — and it makes the ordering the borrow checker's business rather than
the statement order's. The alternative shape, if the clone is unwanted: have `draw`
take `TileMap` by value and hold the whole predecessor, which says the same thing
without the clone.

---

## 2. `Segment` is a sweep and a paint parcel in one struct — open

`Segment` carries ten geometry/frame fields (`start`, `dir`, `curvature`, `radius`,
`ramp`, `frame`, `reach`, `length`, `orient`, `dist`) and five paint rates (`add`,
`lift`, `deposit`, `bleed`, `tooth`). The cost is not the field count; it is that half
the fields are meaningless for one of the two things the type is used to represent.

`bleed_fires` has to fabricate a whole `Segment` for a window that **is not a
segment**. It copies `add`, `lift`, `deposit` and `tooth` in ("the window inherits the
crossing segment's rates"), and `dynamics_plan` then zeroes every one of them back out
when the window becomes a slot ("λ_lift = 0 so the canvas keeps everything, λ_deposit =
0 so the (uninvolved) tool lays nothing, no drain because nothing is laid, no `add`
because this is not a stretch of painting, no tooth because there is no `add` for the
ground to gate"). `ramp: 0.0` needs a nine-line comment explaining why a window cannot
have one — a fact the type could have carried instead.

The same tuple, `(usize, Segment)`, is then threaded unnamed through five signatures:
`affected_tiles`, `chunk_segments`, `snapshot_size`, `dynamics_plan`, `bleed_fires`,
and `DynamicsRun::draw`.

**What to do.** Split the subject the type has two of:

```rust
/// Where a tip goes and how wide it is — everything the shaders unroll a sweep from.
struct Sweep { start, dir, curvature, radius, ramp, frame, reach, length, orient, dist }

/// What the tip is doing while it goes there, as the pen asked for it here (§6.2).
struct Paint { add, lift, deposit, bleed, tooth }

struct Segment { sweep: Sweep, paint: Paint }

/// One crossing of the bleed cadence: which segment it fires after, the stretch of
/// path it relaxes over, and the one axis it uses.
struct BleedFire { after: usize, window: Sweep, bleed: f32 }
```

What it buys:

- `coverage_bounds`, `segment_bounds`, `for_each_touched`, `snapshot_size` and
  `PlanCtx::rect` take `&Sweep` — they never read a rate, and would then be unable to.
- A window **cannot** carry rates it does not mean, so both the copy-in and the
  zero-out go away, and with them the paragraph defending `ramp: 0.0`.
- The unnamed `(usize, Segment)` gets a name at six call sites.
- `render_dynamic`'s per-piece re-key (`fires[lo..hi].iter().map(|(after, w)| (after -
  piece.start, *w)).collect()`) becomes a slice plus an index base, dropping an
  allocation per piece per pointer move.

`bleed_stencil` wants `(bleed, radius, span)`, all of which are on the `Sweep` plus the
one scalar — so `BleedFire` is complete as written.

---

## 3. Dispatch rects are computed twice, and the fit is an assert — open

`snapshot_size` folds `rect_extent` over the piece's coverage boxes to size the
snapshot square; `dispatch_rect` then computes the **real** rect per slot and asserts
it fits:

```rust
assert!(
    w <= rect_extent(span.x) && h <= rect_extent(span.y) && w <= dsize && h <= dsize,
    "a {w}x{h} dispatch rect overruns the {dsize} snapshot scratch",
);
```

`cell_geometry` carries the twin of this against `cell_scratch_size`.

The doc comments are right that this beats what it replaced (a `debug_assert` and a
`min`, which in release silently truncated a footprint). But the shape is still wrong
in two ways. It is a **panic in the render path** — on wasm that aborts the whole app,
for a condition the module elsewhere degrades from with a `tracing::error!` and a
fallback (`StrokePath::TipTooLarge`). And it is only *needed* because the rect is
derived twice: once approximately, to size the scratch, and once exactly, to dispatch.

`coverage_bounds` is in fact evaluated four times per segment on the dynamics path —
in `chunk_segments` (via `segment_bounds`), in `affected_tiles` (via
`for_each_touched`), in `snapshot_size`, and in `dynamics_plan`. Each involves an
`arc_at` and a `sqrt`.

**What to do.** Compute the rects once, and take the scratch square as their maximum:

```rust
/// Every rect the piece will dispatch, and the square that holds all of them.
struct PieceRects { rects: Vec<Rect>, dsize: u32 }
```

The fit stops being a bound to defend and becomes a `max` — no assertion, no
`rect_extent`/`dispatch_rect` duality, and `rect_extent`'s "monotone in `span`, which
is what makes `snapshot_size`'s maximum a bound on every rect rather than on the one
that happened to be widest" stops needing to be true, because nothing depends on it.

`SNAPSHOT_QUANTUM`'s round-up sits on top of the max unchanged. `cell_scratch_size`
gets the same treatment from `dsize`, so `cell_geometry`'s assert goes with it.

Note that `every_dispatch_rect_fits_the_scratch_its_piece_sized` and
`the_scratch_is_sized_with_the_bleed_windows_in_it` are testing exactly the relation
this makes unrepresentable — they should be kept, restated as "the max is over the
rects that were dispatched".

---

## 4. Layouts and bind groups are two hand-written mirrors — open

The largest maintainability liability in the module, and the one with the most lines
behind it.

`kit.rs` hand-lists seven bind group layouts; `DynamicsRun::bind_piece` hand-lists the
seven matching entry sets, ~200 lines of it. Each pair is joined by a magic slice
count:

```rust
][..12 + 4 * usize::from(resid)],   // exchange_bgl
][..12 + 3 * usize::from(resid)],   // deposit_bgl
][..11 + 3 * usize::from(resid)],   // deposit_coarse_bgl
][..9  + 3 * usize::from(resid)],   // settle_bgl
```

All seven are correct today (checked). All seven are recounted by hand on every edit,
and the failure is a wgpu validation error at pipeline creation — loud, but at runtime
and on a GPU, which is the half of the suite CI does not run against real pixels.

The `filterable` argument to `ctex(b::X, true/false)` is a second hand-made decision
per binding that must match how the entry point actually samples it, and nothing checks
that either.

**The declaration already carries everything.** From `dynamics.wesl`:

```wgsl
@group(0) @binding(17) var bake_load_w: texture_storage_2d<rgba32float, write>;
@if(resid) @group(0) @binding(27) var brush_src_resid: texture_2d<f32>;
```

— index, name, WESL type, storage format, **and** the `resid` predicate that the
`[..n + k]` slices are a hand-transcription of. `emit_bindings` in
`stark-shaders/build/mirror.rs` already parses all of it and emits the index as
`binding::BRUSH_SRC_RESID`; which bindings a given entry point reaches is derivable
from the same AST.

**What to do.** Emit the table, not just the index:

```rust
pub struct Binding { pub index: u32, pub kind: BindKind, pub resid: bool }

pub const EXCHANGE: &[Binding] = &[
    Binding { index: 0,  kind: Uniform { min_size: … },       resid: false },
    Binding { index: 27, kind: Texture { filterable: … },     resid: true  },
    …
];
```

`kit.rs` then builds every layout by mapping the table and filtering on `resid`;
`bind_piece` builds every group by supplying a resource per named binding. ~350
hand-written lines collapse, the seven slice counts disappear along with the whole
drift class, and a `resid` binding can no longer be declared in one half and forgotten
in the other.

**The one residue is filterability**, which is not in the declaration — a texture's
type does not say whether the entry point `textureSample`s it or `textureLoad`s it.
Two ways out, in order of preference: derive it from the entry point's AST (the
generator is already walking it for reachability), or annotate it in WESL beside the
`@if`. A host-side override map would work but re-opens a smaller version of the same
hole.

This also closes a test gap for free: with one table, "every layout has exactly the
bindings its bind group supplies" is true by construction rather than untestable
without an adapter.

---

## 5. Three answers to "what region do these sweeps need" — open

The same question is answered three ways, and all three must agree:

| Walk | Where | Produces |
|---|---|---|
| `for_each_touched` → `affected_tiles` | `segments.rs` | the tile set |
| `region_of`, inside `chunk_segments` | `segments.rs` | bbox → region dims |
| `region_rect` | `segments.rs` | tile set → the rect the loop allocates |

`chunk_segments` promises a piece whose region fits `MAX_REGION_DIM`; `region_rect`
builds the region that promise is about, from a different input, by a different route.
The comment says so ("those two answers have to be the same rectangle") and
`the_chunker_measures_the_region_the_render_builds` tests it — but `region_rect` never
**asserts** its own result is within `MAX_REGION_DIM`, so the promise is unchecked at
the exact point it is relied on. A disagreement is a silent oversized allocation.

**What to do.** One accumulator, three readings:

```rust
#[derive(Default)]
struct Coverage { lo: Vec2, hi: Vec2 }

impl Coverage {
    fn add(&mut self, s: &Sweep);            // grows by `segment_bounds`
    fn dims(&self) -> (u32, u32);            // what `region_of` answers
    fn tiles(&self) -> BTreeSet<TileCoord>;  // what `affected_tiles` answers
    fn rect(&self) -> Option<RegionRect>;    // what `region_rect` answers
}
```

`chunk_segments` then extends one of these and asks `dims()`; `draw` builds the same
type from the piece it was handed and asks `rect()`. The two cannot disagree, because
there is one definition of what a set of sweeps covers. Pairs naturally with §3, which
wants the boxes computed once anyway.

---

## 6. The swept path serializes on one shared scratch pair — open, measure

`render_swept` acquires **one** scratch pair for the whole stroke and loops:

```rust
for (i, coord) in coords.iter().enumerate() {
    // sweep this tile's segments into `scratch`, CLEAR load op
    // integrate `scratch` over the base into a fresh CoW tile
}
```

Tile *n+1*'s sweep writes the texture tile *n*'s integrate reads. That is a
write-after-read dependency, so the driver must order them: `2N` render passes run
strictly back to back with no overlap whatever. A live tail crossing ~30 tiles pays 60
serialized passes per pointer move.

The comment defends *sharing* the pair on lifetime grounds — "a pair acquired per tile
and dropped at the end of its iteration goes back on the pool's free list while the
passes naming it are still only recorded" — which is §1's hazard, correctly identified.
But the fix it chose (one pair) is stronger than the hazard needs. A **ring** of pairs,
all held by the scope until `finish`, is equally sound (every sweep pass clears, so no
tile can see what another left) and gives the GPU *k*-way overlap.

At `TILE_TEX = 256` a target is 256² × 8 B = 512 KB, so a ring of 3 costs ~3–4.5 MB
depending on the residual. Against `ScratchPool`'s own 256 MB budget that is noise.

Self-contained and cheap; bracket it with `cargo bench -p stark-core --bench stroke`
before and after. The `stroke-bench-noise` warning in the lift-end notes applies.

The larger version of this change — sweep every tile into a scratch **atlas** in one
pass with per-tile viewports, then one integrate per tile — halves the pass count
outright, but wants the atlas to be sized and pooled and is a much bigger diff. Worth
naming as the direction if the ring measures well.

---

## 7. 40–60 bind groups per fold on the live tail — open

The module frets about WebGPU allocation *rate* in three separate doc comments —
`UNIFORM_STRIDE`'s "the rate is the thing JS GC cannot keep up with", `ScopedResources`,
`ScratchPool`'s reason for destroying rather than dropping — and then builds, per
pointer move:

- `bind_piece`: 8–10 bind groups per piece (snapshot, exchange ×2, bake ×2, deposit,
  settle, and the coarse pair);
- `composite_region`: **one per halo tile**, in the loop over `halo`;
- `render_swept`: one `integrate_bg` per affected tile.

For a live tail that is roughly 40–60 bind groups a frame. The buffers and uniforms
were fixed (one buffer, dynamic offsets); the bind groups were not.

The halo ones are the easy half and the largest count: a composite tile bind group is a
pure function of `(tile identity, layout)` — the tile's own three views and nothing
else. Cache it on the `GpuTile` beside its views, or memoize on the `DynamicsRun` keyed
by `TileCoord` so at least the pieces of one stroke share it.

`render_swept`'s `integrate_bg` genuinely varies per tile (it binds that tile's base
views) and cannot be hoisted without bindless; the `bind_piece` ten are already once per
piece rather than per dispatch. So this finding is really about the halo loop.

Measure before and after — this is a CPU-side cost on the submit thread, so the stroke
bench sees it but a GPU timestamp will not.

---

## 8. `segments.rs` is five subjects; `budget.rs` is three — open

`segments.rs` is 2,636 lines carrying, in order: the round tip's coverage field, the
taper model, segment generation, the tile walk and the coverage boxes, and the region
chunker. Only the middle one is what the file is named for.

| new file | contents |
|---|---|
| `tip.rs` | `round_coverage`, `tip_reach`, `frame_scale`, `orientation_turns` |
| `taper.rs` | `taper_profile`, `Taper`, `TAPER_*`, `DAB_TRAVEL` |
| `segments.rs` | `Segment`/`Sweep`/`Paint`, `generate_segments_in`, `sample_at` |
| `region.rs` | `for_each_touched`, `affected_tiles`, `tiles_with_segments`, `coverage_bounds`, `segment_bounds`, `region_of`, `chunk_segments`, `segment_fits_region`, `region_rect`, `RegionRect` |

`region.rs` is the split that earns it: §5's "these two must be the same rectangle"
invariant becomes file-local, which is the precondition for making it structural.

Separately, `round_coverage` is really an **asset** function. It pairs with
`assets::build_prefix_tau` — its own test sweeps it through that function's `tau_of` to
avoid the clamp drifting between them — and its only consumer is `tips.rs`. It reads
oddly in a file about turning a path into segments, and `stark-assetid`/`assets` is
where its sibling lives.

`budget.rs` (735 lines) holds three models that share nothing but a file:

- the flattening and region caps — `MAX_REGION_DIM`, `MAX_STAMPS`, `MAX_TIP_TURN`,
  `flatten_tolerance`, `exchange_travel`, `lambda`. This is what the file's header
  describes.
- the **bleed diffusion solver** — `BLEED_*`, `bleed_stencil`, `stencil_moment`,
  `STENCIL_MOMENT_PER_REACH2`, ~200 lines plus seven tests. Its only consumer is
  `dynamics/plan.rs`, next to `bleed_fires`, which is the other half of the same model.
- the **coarse deposit cell law** — `footprint_cell`, `shoulder_per_radius`,
  `FOOTPRINT_CELL_MAX`.

Move the bleed solver to `dynamics/bleed.rs` beside `bleed_fires` and the axis has one
home. `shoulder_per_radius` genuinely has two consumers on opposite sides (the cell and
the taper's subdivision) and should stay where both can reach it.

---

## 9. Every kit field is `pub(super)`; the renderer has no boundary — open

`SweptKit`'s six fields and `DynamicsKit`'s twenty-two are all `pub(super)` /
`pub(in crate::gpu::stroke)`, and `render_swept` and `render_dynamic` are
`impl StrokeRenderer` blocks living in sibling modules, reaching into the renderer's
private fields. Everything in the module is a friend of everything.

`SweptKit`'s own doc comment names the problem it was created to fix — "these five sat
loose on `StrokeRenderer` among the caches, so a struct documented as holding 'only
immutable GPU objects' held one path's pipelines by name". The type landed; the
boundary did not. The fields simply moved behind a name that is still fully public to
every file in the tree.

**What to do.** Give each kit the API its type implies — `SweptKit::record(&self, enc,
&SweptJob)`, `DynamicsKit::record_loop(&self, cpass, &plan, &bindings)` — and keep the
only `impl StrokeRenderer` in `mod.rs`. Lower priority than §1–§4 and mostly a
readability change, but it is why "where does this pipeline get bound" is currently a
grep rather than a jump.

---

## 10. Smaller correctness and clarity items — open

**10.1 — `let last = pieces.len() - 1` underflows.** `render_dynamic` relies on
`chunk_segments` never returning empty, which is true (it pushes `start..len` whenever
`segments` is non-empty, and `segments.is_empty()` returned early two screens up) but is
a non-local invariant across two functions. `pieces.len().saturating_sub(1)`, or
restructure so the last piece is named rather than counted.

**10.2 — `TipTooLarge` logs an error per pointer move.** `render_range`'s
`tracing::error!` fires on every render of a record that will never stop failing —
the comment notes this ("It repeats per pointer move, because the gate is re-asked per
render") but names it as a consequence rather than a thing to fix. It is a property of
the *record*, so it should be said once per record, not per frame. The engine has the
stroke id to key on.

**10.3 — `StrokeCarry::dirty` over-reports.** Documented as "everything in the returned
map that differs from `scene.base`", but the swept path reports every tile
`for_each_touched` named — including tiles where every fragment differenced its prefix
taps to zero and the fresh CoW tile is bit-identical to the base. Harmless for §17.6's
purpose (a superset costs a redundant composite, not a wrong picture), but the doc
should say superset if the behaviour is staying.

**10.4 — Comments that have become archaeology.** The density is overwhelmingly an
asset here — the measurements and dead ends recorded on `BLEED_TRAVEL_QUANTUM`,
`RESERVOIR_EXCHANGE_STEP` and `footprint_cell` are the reason those constants are
tunable at all, and none of that should move. But a few passages now document a change
rather than the code: `swept.rs`'s five-line tombstone for a deleted test
("`the_draw_call_and_the_strip_agree_on_the_vertex_count` stood here…"), and the note
above the `Stamp` import recording what a since-deleted host-side copy used to say.
Those belong in the commits.

---

## 11. The `bake` dispatch shape has never been measured — open, measure first

`record_loop` dispatches the reservoir bake as `dispatch_workgroups(1, BAKE_RES, 1)` —
**one workgroup wide** — once per segment, and the loop is sequential so none of them
overlap. A fine stroke is thousands of segments, so this is a long chain of tiny
serialized dispatches.

That is plausibly the dispatch-bound regime the `dynamics-perf-profile` and
`march-vs-master-perf` notes describe at small radius, where the ALU work per segment
is trivial and the win came from doing fewer, bigger things. It may be irreducible —
the sequence is what makes the loop a loop — but the split has not been measured on
master.

**What to do before anything else in the loop.** Get the per-entry-point share of a
live fold at `radius = 8` and `radius = 100`: bake vs exchange vs deposit vs the
composite and write-back either side. Timestamp queries or a coarse A/B (stub one entry
point to a no-op and diff the fold time). If bake is a third of the fold at small
radius, the shape is worth attacking; if it is 5%, §6 and §7 are the whole story and
this closes.

---

## What the review did not find

No bug against a shipped behaviour. Several invariants that would have been easy to get
wrong are already structural, and the review confirmed rather than disturbed them:

- **`dynamics_setup` asks only about the brush**, so a live tail and its commit cannot
  take different paths — and because a modulation is a factor in `[0, 1]` by
  construction, the test on the brush's own rates is exact rather than conservative.
- **`safe_frozen`'s three conditions really do prove what they claim.** Total arc
  length ≥ `arc(head→cut) + arc(cut→tip)` ≥ `start_px + end_px`, both measured on
  chords which under-estimate, so an admitted span's taper compression is 1 and stays 1.
- **The bleed cadence's reach-back is exactly one quantum**, which is what
  `segment_fits_region` and `chunk_segments` budget for: `crossings ≤ length/bq + 1`, so
  `crossings · bq ≤ length + bq`, and `MAX_BLEED_FIRES_PER_SEGMENT` only ever lowers it.
  The cap drops the *oldest* windows, which is the right end to drop.
- **The settle's rect is inside the last segment's coverage box** (a segment's box is
  the tip's square grown by its travel), so nothing has to size the scratch for it
  separately — the comment claiming this is correct.
- **`Slot::pack` is pinned field-by-field** against the lane the shader reads, which is
  the one place in the module where a positional ABI would otherwise be a silent wrong
  picture — and the test says so in those terms.
- **`SubmitScope`'s two lease tiers are the right two.** Run leases (the reservoir
  ping-pong, the bake pair) and piece leases (region, snapshot, cells) is exactly the
  split that keeps a long stroke's peak transient memory at one region, and `Kept`'s
  borrow argument for outliving its run holds.

The `unpoisoned` lock helper and the two hand-rolled LRUs in `tips.rs` are fine at
`ROUND_TIPS_KEPT = 4` and were not worth a finding.

## Verification

None. Nothing has been changed, built or run — this is the review, not the work.
Every claim above about what the code does was read out of the source at `8952270`;
the arithmetic claims in "What the review did not find" were re-derived rather than
taken from the comments that assert them.
