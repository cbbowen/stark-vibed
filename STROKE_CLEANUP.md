# `gpu/stroke` — the architectural cleanup ledger

Reviewed 2026-08-15 on branch `gpu-cleanup` (`8952270`), across all 11 files of
`crates/stark-core/src/gpu/stroke` (~8.5k lines, of which ~2.4k are the in-file test
modules), plus the two things it is joined to at the seam: the generated shader
mirrors (`stark-shaders/build/mirror.rs`, §6.10) and `gpu/tile`'s pool.

**Seven of eleven findings have landed**, in four commits above `8952270`. Three are
open and one landed in part; each says below what it would still cost. Symbol names
are the durable part.

This is `COMPOSITE_CLEANUP.md`'s review one directory over, and it found the mirror
image. There the module's hardest rules were *argued in prose and consumed
positionally*; here most of them were already structural — `SubmitScope` makes a lease
release unforgeable, `dynamics_setup` answers from the brush alone so no two renders
can disagree, `Slot::pack` is pinned field-by-field against the lane the shader reads.
What was left was a shorter list, and it clustered in the places that discipline had
not reached: the GPU-facing half (`kit`/`run`), and the three or four types that
carried more than one subject.

## The one-sentence summary

**The models were separated and tested; the bindings and the boxes were not.** Every
number a brush becomes was CPU float math with a test on it, but the layouts and their
bind groups were two hand-maintained mirrors of one shader-declared fact (§4), one
segment's coverage box was computed four times by three functions that had to agree
(§3, §5), and `Segment` carried a bleed window's rates that the slot then zeroed back
out (§2). One correctness hazard was live-but-unreached (§1), and it was the very
class `scratch.rs`'s module doc was written against.

| § | Finding | Kind | Status |
|---|---|---|---|
| 1 | Base tiles reach the pool free list under an open encoder | latent, ordering-held | **landed** `174f405` |
| 2 | `Segment` is a sweep and a paint parcel in one struct | god-struct, fabricated windows | **landed** `169bd0a` |
| 3 | Dispatch rects computed twice; the fit is an assert, not a max | duplication + wasm panic | **landed** `169bd0a` |
| 4 | Layouts and bind groups are two hand-written mirrors | ~350 lines, drift class | **landed** `140ae11` |
| 5 | Three answers to "what region do these sweeps need" | invariant by convention | **landed** `169bd0a` |
| 6 | The swept path serializes on one shared scratch pair | 2N passes, no overlap | **landed** `169bd0a`, −18/−20% |
| 7 | 40–60 bind groups per fold on the live tail | allocation rate | open |
| 8 | `segments.rs` is five subjects; `budget.rs` is three | 2.6k-line file | **part landed** `951b032` |
| 9 | Every kit field is `pub(super)`; the renderer has no boundary | encapsulation | open |
| 10 | Smaller correctness and clarity items (four) | mixed | **landed** `174f405` |
| 11 | The `bake` dispatch shape has never been measured | measure first | open |

---

## 1. Base tiles reach the pool free list under an open encoder — landed

`DynamicsRun::draw` recorded `composite_region`, which samples `base`'s tiles, into
the piece's encoder; the caller then replaced its `map` with the return value,
dropping at that assignment every tile handle the new map superseded. A
`TilePairHandle`'s drop *is* its release to `TilePool`'s free list, so those textures
became available to the next `acquire_tex` while the commands reading them were still
only recorded.

`PoolInner::trim` guards the irreversible half — it will not *destroy* a slot returned
during the current epoch — but reuse is unguarded by construction, and reuse is enough.
Consecutive pieces share the tiles around their cut, because `affected_tiles` grows
every segment's box by an apron, so this was the ordinary case for any stroke long
enough to be chunked.

It never fired, and the reason was not a rule: the next statement the run executes is
the following piece's `flush`, and nothing acquires in between.

**What landed.** `self.scope.hold(base.clone())` before the composite, which is what
`render_swept` already does with its scratch pair across the identical boundary. The
scope releases it in the call that submits the commands naming it. An `rpds` map of
`Arc` handles costs a refcount per tile and no pixels.

## 2. `Segment` is a sweep and a paint parcel in one struct — landed

Ten geometry fields and five paint rates were one struct, so `bleed_fires` had to
fabricate a whole `Segment` for a window that is not one: it copied `add`, `lift`,
`deposit` and `tooth` in, `dynamics_plan` zeroed every one of them back out lane by
lane, and `ramp: 0.0` needed nine lines explaining why a window must not have one.

**What landed.** `Sweep` (where a tip goes and how wide it is) and `Paint` (what it is
doing while it travels), with `Segment` holding both and `BleedFire { after, window:
Sweep, bleed }` replacing the unnamed `(usize, Segment)` that was threaded through six
signatures. Every box, rect and tile walk takes `&Sweep` and can no longer read a rate;
a window cannot carry one.

The test helpers split the same way, which is the part worth keeping: `whole()` returns
`Vec<Sweep>` and `whole_segments()` returns segments, so a test that wanted a paint rate
would have to say so.

## 3. Dispatch rects computed twice; the fit is an assert, not a max — landed

`snapshot_size` folded a position-independent `rect_extent(span)` over the coverage
boxes to size the scratch; `dispatch_rect` then computed the real rect and *asserted*
it came in under that bound — a panic in the render path, which on wasm aborts the app.
Two derivations of one number, related by an argument and defended by a crash.

**What landed.** `dynamics_plan` walks its sources once to fix the dispatch order,
measures every rect against that walk, and takes `snapshot_square` as their maximum. A
maximum is not a claim about what it was taken over, so `rect_extent` and both
assertions are gone. `cell_geometry`'s survives as a `debug_assert` carrying its
derivation. The plan now returns the square it sized, so the scratch is allocated *from*
the plan rather than the plan checked against the scratch.

`every_dispatch_rect_fits_the_scratch_its_piece_sized` became
`every_dispatch_grid_fits_the_scratch_its_piece_sized`: the surviving claim is one step
out, that a dispatch rounded up to whole 8×8 workgroups still fits — which holds only
because `SNAPSHOT_QUANTUM` is itself a multiple of 8.

## 4. Layouts and bind groups are two hand-written mirrors — landed

Seven entry points had their layouts hand-written in `kit.rs` and their bind groups
hand-written in `run.rs`, aligned by nothing but the order they were written in and
closed by a magic element count per layout (`[..12 + 4 * usize::from(resid)]`, seven of
them). All seven were correct, by attention.

**What landed.** `emit_bindings` now emits a `BINDINGS` table beside the indices,
carrying each slot's kind, its storage format, its uniform's `min_binding_size` and its
`@if(resid)` gate — all read off the declaration. The host writes **one list per entry
point** (`dynamics/slots.rs`) and `desc::layout_for` / `desc::bind_group_for` build both
sides from it. `stor` versus `stor32` is not a choice any more, the element counts are
gone, and `push_resid` went with them.

Filterability stays on the host, and that is not a gap: it is a property of the (entry
point, binding) pair, since `region_color` is `textureLoad`ed by `snapshot` and
`textureSample`d by `exchange`. So a list says `Slot::at` or `Slot::sampled` and nothing
else. Two CPU tests stand behind the hand-written half.

Documented at the project level, which was the other half of the ask: §6.10 gained
"Adding a binding table" and a closing rule that generalizes all five generators, and
CLAUDE.md gained the one-line version.

## 5. Three answers to "what region do these sweeps need" — landed

`affected_tiles` enumerated a tile set, `region_of` turned a box into region dimensions
inside `chunk_segments`, and `region_rect` turned the set back into a rectangle. The
chunker's promise was about the second and the allocation about the third; a comment
asked them to be the same rectangle.

**What landed.** One `Coverage` accumulator with `add`/`union`/`dims`, and a `Covered`
that carries the tile set beside the box from a single walk. `Covered::rect` takes its
**extent** from `Coverage::dims` — the very function the chunker checked — and only its
**halo** from the set, which is sound because `min` of `floor(lo/tile)` is `floor(min
lo/tile)`. That identity is what the rewritten test pins, the agreement itself no longer
being a claim. `rect` also debug-asserts the `MAX_REGION_DIM` the chunker promised,
which was missing at the point it is relied on.

## 6. The swept path serializes on one shared scratch pair — landed, **−18% to −20%**

One pair for the whole stroke is sound (every sweep clears) but serializing: tile *n+1*'s
sweep writes the texture tile *n*'s integrate reads, a write-after-read the driver must
order, so the path's `2N` render passes ran strictly back to back.

**What landed.** A ring of three pairs, round-robin, all held to the submit for the
reason the single pair was held. ~1.5 MB apiece at `TILE_TEX = 256`, against
`ScratchPool`'s 256 MB budget; a stroke touching fewer tiles than the ring takes only
as many pairs as it has tiles.

**Measured**, by setting `SCRATCH_RING` back to 1 — which is exactly the old code — and
running `cargo bench -p stark-core --bench stroke` as an A/B:

| case | change | p |
|---|---|---|
| `live/swept/30` | **−17.8%** | 0.00 |
| `live/swept/100` | **−20.3%** | 0.00 |
| `live/swept/8` | +4.0% (no effect) | 0.26 |
| `commit/swept/8`, `/30`, `/100` | −1.9%, −1.5%, +0.9% (no effect) | 0.51, 0.48, 0.64 |

The shape is the argument confirmed rather than a number to bank. The win is where
there are passes to overlap — a live tail at radius 30 or 100 crosses many tiles — and
absent where there are not: a radius-8 tail covers a handful, and a `commit` is one
pass over the whole stroke rather than a re-render per pointer move. Nothing regressed.

Three is not tuned. Nothing was measured at 2 or 4, and the ceiling is memory rather
than diminishing returns, so there may be a little more here.

## 7. 40–60 bind groups per fold on the live tail — open

The module frets about WebGPU allocation *rate* in three doc comments and then builds,
per pointer move: 8–10 bind groups in `bind_piece`, **one per halo tile** in
`composite_region`, and one `integrate_bg` per affected tile in `render_swept`.

The halo ones are the easy half and the largest count: a composite tile bind group is a
pure function of `(tile identity, layout)` — the tile's own three views and nothing
else. Cache it on the `GpuTile` beside its views, or memoize on the `DynamicsRun` keyed
by `TileCoord` so at least the pieces of one stroke share it.

`render_swept`'s `integrate_bg` genuinely varies per tile and cannot be hoisted without
bindless; `bind_piece`'s ten are already once per piece rather than per dispatch. So
this finding is really about the halo loop.

Measure before and after — it is a CPU-side cost on the submit thread, so the stroke
bench sees it but a GPU timestamp will not.

## 8. `segments.rs` is five subjects; `budget.rs` is three — part landed

**What landed.** `region.rs` (the tile walk, the coverage boxes, `Coverage`/`Covered`,
`chunk_segments`, `segment_fits_region`, `RegionRect`) and `dynamics/bleed.rs` (the
cadence, the stencil solve and their seven tests). Those were the two the finding
singled out: `region.rs` because it is where §5's invariant lives and file-locality is
the precondition for it staying structural, `bleed.rs` because the axis is one model
that was being decided in two places a directory apart. `budget.rs` is now the one
subject its own header describes.

**What is left.** The other two splits of `segments.rs`:

| new file | contents |
|---|---|
| `tip.rs` | `round_coverage`, `tip_reach`, `frame_scale`, `orientation_turns` |
| `taper.rs` | `taper_profile`, `Taper`, `TAPER_*`, `DAB_TRAVEL` |

The same mechanical move, with each one's test block. Lower priority now: at 2.1k lines
`segments.rs` is no longer the file that most needs it.

Still worth saying separately: `round_coverage` is really an **asset** function. It
pairs with `assets::build_prefix_tau` — its own test sweeps it through that function's
`tau_of` to keep the clamp from drifting between them — and its only consumer is
`tips.rs`.

## 9. Every kit field is `pub(super)`; the renderer has no boundary — open

`SweptKit`'s six fields and `DynamicsKit`'s twenty-two are all `pub(super)` /
`pub(in crate::gpu::stroke)`, and `render_swept` and `render_dynamic` are
`impl StrokeRenderer` blocks living in sibling modules, reaching into the renderer's
private fields. Everything in the module is a friend of everything.

`SweptKit`'s own doc comment names the problem it was created to fix — "these five sat
loose on `StrokeRenderer` among the caches" — and the type landed while the boundary did
not.

**What to do.** Give each kit the API its type implies (`SweptKit::record(&self, enc,
&SweptJob)`, `DynamicsKit::record_loop(&self, cpass, &plan, &bindings)`) and keep the
only `impl StrokeRenderer` in `mod.rs`. Mostly a readability change, but it is why
"where does this pipeline get bound" is a grep rather than a jump.

Note that §4 moved the needle here without meaning to: `kit.rs` no longer holds seven
hand-written layouts, so the file is 338 lines rather than 501, and `slots.rs` is now
the thing both sides read. The remaining coupling is the pipelines and the samplers.

## 10. Smaller correctness and clarity items — landed

- `pieces.len() - 1` underflowed on an empty cut. That cut cannot be empty, but the
  proof was two functions from the subtraction; it asks the iterator now.
- `TipTooLarge` shouted on every pointer move. The gate is a pure function of the brush
  — which is what lets a tail and its commit agree for free — so its *answer* is a
  property of the record, and is said once per stroke seed.
- `StrokeCarry::dirty` is a superset of the tiles whose pixels changed, not the set: a
  tile at the edge of the reach can take a fresh CoW tile that differences to zero.
  Narrowing it means comparing pixels, which is the cost it exists to avoid. Documented
  as the superset it is.
- The archaeology comments (`swept.rs`'s tombstone for a deleted test) were left alone.
  On reflection they are cheap, and the surrounding density is the module's main asset.

## 11. The `bake` dispatch shape has never been measured — open

`record_loop` dispatches the reservoir bake as `dispatch_workgroups(1, BAKE_RES, 1)` —
one workgroup wide — once per segment, and the loop is sequential so none of them
overlap. A fine stroke is thousands of segments.

That is plausibly the dispatch-bound regime the `dynamics-perf-profile` and
`march-vs-master-perf` notes describe at small radius. It may be irreducible — the
sequence is what makes the loop a loop — but the split has not been measured on master.

**Before anything else in the loop**, get the per-entry-point share of a live fold at
`radius = 8` and `radius = 100`: bake vs exchange vs deposit vs the composite and
write-back either side. If bake is a third of the fold at small radius the shape is
worth attacking; if it is 5%, §7 is the whole story and this closes.

---

## What the review did not find

No bug against a shipped behaviour. Several invariants that would have been easy to get
wrong were already structural, and the review confirmed rather than disturbed them:

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

Every commit was taken green: `cargo fmt --all --check`, `cargo clippy -p stark-core
--all-targets -D warnings`, and `cargo test --workspace` redirected once to a file per
batch. **831 tests pass with the goldens rendered and compared** (`STARK_SKIP_GOLDEN`
unset, `STARK_ALLOW_NO_GPU` unset), which is the check that matters: §1, §2, §3, §5, §6
and §8 must all be pixel-identical, and are.

§4 additionally ran the second configuration — `cargo clippy --workspace --all-targets
--no-default-features --features stark-net/webrtc` — because that is the `resid = false`
path its whole residual gate drives, and `cargo check -p stark-ui --target
wasm32-unknown-unknown`.

Two tests were added (`slots::tests`) for the one hand-written half §4 leaves behind,
and four rewritten to state what survived the change rather than what it replaced:
`every_dispatch_grid_fits_the_scratch_its_piece_sized`,
`the_scratch_is_sized_with_the_bleed_windows_in_it` (which had been passing on the
quantum's slack), `the_box_and_the_tile_set_measure_the_same_rectangle`, and
`the_per_tile_lists_hold_exactly_the_segments_that_reach_each_tile`.

§6 is the only change here that claims a performance number, and it was measured as
an A/B against `SCRATCH_RING = 1` — which is exactly the code it replaced — rather than
against a different commit: `−17.8%` on `live/swept/30` and `−20.3%` on `live/swept/100`,
both at `p = 0.00`, with no significant movement anywhere else. Nothing else here claims
one. §7 and §11 are open *because* they want measurement first.
