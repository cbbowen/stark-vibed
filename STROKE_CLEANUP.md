# `gpu/stroke` cleanup

A review of `crates/stark-core/src/gpu/stroke` (2026-08-04), kept as a working
list. Ordered by what to do first — the first three compose, so doing them in
order makes each smaller than it looks.

Nothing here is a known-wrong pixel. It is structure, doc drift, and one class of
per-frame allocation churn.

## 1. The `Stamp` slot — **done**

`LoopDispatch.slot` was `[f32; 36]`, built at three positional construction sites
in `dynamics_plan` (segment, bleed, settle) whose only structure was comments
naming shader lanes. The receiving `Stamp { a..i: vec4 }` in `dynamics.wesl` is
nine vec4s; nothing checked the count, the lane alignment, or that the three
sites agreed on which lane meant what.

Replaced with a `#[repr(C)] Pod` `Stamp` mirroring the shader's nine lanes, a
`const SLOT = size_of::<Stamp>()` pinned by a compile-time assert (retiring the
`144` that was duplicated between the buffer window and the layout's
`min_binding_size`), and a `SlotCommon` holding the lanes every slot fills the
same way — so each of the three sites now lists only what differs.

## 2. Split `DynamicsRun::draw` — **done**

520 lines doing six jobs. Now `draw` is 40 lines naming the six steps, over
`composite_region` / `snapshot_scratch` / `upload_plan` / `bind_piece` /
`record_loop` / `write_back`, with `Region`, `Snapshot` and `PieceBindings`
carrying what passes between them. `STRIDE` and the `clear` operations became
module constants rather than locals re-declared per call.

Took item 11's `deposit_bgs` with it: the `(0..1).map(…).collect()` Vec-of-one is
now `PieceBindings.deposit`, and the two genuine ping-pong pairs are `[_; 2]`
arrays instead of `Vec`s.

Still open from this item: `dynamics_plan` keeps its
`#[allow(clippy::too_many_arguments)]` — its 8 args want a
`PieceGeom { region_origin, dsize, channels, surface }` bundle, which is easier
to do alongside item 3.

## 3. `LoopDispatch` should be an enum — **done**

Three slot kinds were encoded as `bleed_only: bool` plus a position
(`segment_slots = plan.len() - usize::from(settle)`, then
`plan.get(segment_slots)`). Now `SlotKind { Segment { wick_steps }, Bleed,
Settle }`, and `record_loop` is a `match` over the whole plan with no index
arithmetic and no `settle` parameter — the pen-up is simply the last slot.

`exchange_groups` went with it: it was `(BRUSH_RES/8, BRUSH_RES/8)` for every
segment slot and `(0, 0)` — never read — for the other two, so it is now the
`RESERVOIR_GROUPS` constant.

Also took the rest of item 2 and two bullets from items 10 and 11:
`dynamics_plan`'s 8 args became a `PlanCtx`, retiring its
`#[allow(clippy::too_many_arguments)]`; the thrice-duplicated coverage-box →
dispatch-rect arithmetic became `PlanCtx::rect`; and that one place now carries
a `debug_assert!` that the rect fits the snapshot scratch, so the `.min(dsize)`
can no longer silently clip a footprint. The whole GPU suite runs it without
firing.

## 4. Per-tile uniform buffer + bind-group churn — **done**

`swept.rs` created a buffer *and* a bind group per affected tile, per render — on
the path that re-renders every pointer move — and `dynamics.rs` did the same for
the slice write-back. Both now lay their per-tile uniforms out as
`UNIFORM_STRIDE` slots in one buffer read through dynamic offsets, exactly as
the stamp buffer already did, so a K-tile stroke builds one buffer and one bind
group instead of 2K objects.

The slice case collapsed particularly well: every tile slices out of the *same*
region, so the two texture bindings beside the uniform never varied — the whole
bind group is now built once per piece.

`UNIFORM_STRIDE` lives in `mod.rs` as the one place the 256-byte alignment and
the reason for it are written down; `XFORM_SLOT` and `SLICE_SLOT` are taken from
their structs' sizes, like `SLOT`.

With `view_buf` and `tile_inst` registered on the piece and the stamp buffer
moved from the run to the piece, **every buffer and texture in the module is now
either registered with `ScopedResources` or owned by a type with its own
`destroy`-on-drop** — the inconsistency this item existed for is gone rather than
argued about. (The stamp buffer's move is safe on the same argument the region
textures already rest on: `flush` submits before it destroys, and WebGPU defers
the free until that submission retires. It also bounds a long stroke's peak
transient cost at one piece, which is what `MAX_REGION_DIM` is for.)

What deliberately stays: one bind group per halo tile in the region composite,
and per-tile `integrate` bind groups in `swept.rs`. Those bind *different
textures* per tile, so there is nothing to fold — and a bind group holds no
allocation of its own.

## 5. Module boundaries — **mostly done**

`mod.rs` went 1046 → 544 lines and is now the renderer it claims to be:
`StrokeRenderer`, `StrokeScene`, `ScopedResources`, `RoundTip`, and the two
constants those actually use.

* **`budget.rs`** (381) — what a stroke is allowed to cost. The cadence constants
  with their measurements and dead ends (`RESERVOIR_EXCHANGE_STEP`,
  `WICK_TRAVEL_QUANTUM`, `BLEED_TRAVEL_QUANTUM`, `MAX_TIP_TURN`), the piece
  ceilings (`MAX_REGION_DIM`, `MAX_STAMPS`), `TAU_PER_PASS`, and the two
  functions that spend them (`flatten_tolerance`, `exchange_travel`). Touches no
  GPU — float arithmetic over a `BrushParams`, which is what lets the
  segment-budget tests pin it exactly.
* **`incremental.rs`** (152) — drawing a stroke in pieces and resuming:
  `ToolState`, `StrokeSpans`, `StrokeCarry`, `safe_frozen`.
* `BRUSH_RES` / `BAKE_RES` / `BAKE_FORMAT` moved into `dynamics.rs`, their only
  user.

Call sites outside the module are unchanged — `mod.rs` re-exports the surface, so
nothing depends on which file an item lives in.

Two things fell out:

* The eleven `super::super::` spellings in `segments.rs`'s tests are gone; the
  test module imports `budget::{MAX_TIP_TURN, flatten_tolerance}` and
  `safe_frozen` by name.
* `safe_frozen` is `pub(crate)` now. Nothing outside `stark-core/src` used it,
  and being `pub` was what made rustdoc object to its doc comment pointing at
  `segments` internals. **The stroke module's rustdoc warnings are 5 → 0.**
  (`StrokeCarry` and `ToolState` have to stay `pub` — they are in `render_range`'s
  signature — so the one link `dirty` made into `segments` became a code span.)

`ViewUniform` and `TileInstance` have since moved to `dynamics.rs`, their only
user — done as a follow-up so the change above stayed a pure move of `mod.rs`'s
contents. Both are private there rather than `pub(super)`, and the four
GPU-uniform mirrors (`ViewUniform`, `TileInstance`, `SliceUniform`, `Stamp`) now
sit together at the top of the file. `swept.rs` exports only `XFORM_SLOT` and
`render_swept`.

## 6. Three caches, three policies, one in the wrong place — **done**

The two round-tip caches are one now: `StrokeRenderer::round_tip`, holding a
`RoundTip { prefix, coverage }` built from a **single** `round_coverage`
evaluation. That was 256² texels of `powf` run twice for the same hardness, once
per texture — and, held apart in two independently-evicting single-entry slots,
the stamp loop could find its prefix hot and its coverage cold and pay for the
field again regardless.

`round_cov` is out of `DynamicsKit`, which was the layering complaint: a mutable
cache inside a struct documented as immutable GPU objects built once. The kit now
holds no `Arc<Mutex<_>>` at all — `dynamics.rs` no longer imports `Arc` or
`Mutex` — so "built once" is a property of the type rather than an intention.

`prefix_view` gained a `coverage_view` sibling, so both "resolve a brush's
texture, asset or generated" helpers sit together and the stamp loop's setup
stopped open-coding the match.

**The original claim about policy was wrong, and the difference is now
documented instead of removed.** Unifying eviction would have been a regression:
hardness is a *continuous slider*, so a grow-forever round-tip cache banks
~320 KB of GPU texture per position a user drags through and never returns it —
single-entry is exactly the working set of "adjust the knob and look". `NoiseKind`
is a small enum, so its cache can hold the whole domain and never evict. Two
policies, two key domains, and each now says which it is and why.

## 7. The two paths derive shared stroke constants independently — **done**

`StrokeRenderer::stroke_constants` resolves them once into a `StrokeConstants`:
the brush's colour in the working space plus its per-unit opacity, the canvas →
weave scale, and the colour-dynamics lookup triplet. Both paths read that struct
— the swept path into its `TileXform`, the loop into `SlotCommon`, which
collapsed to a borrow of it plus the one lane (`grain_bias`) that genuinely
belongs to the piece.

`noise_uniform` / `noise_offset` moved out of `segments.rs` to sit with it, since
"the uniform triplet both paths read" was already their whole charter.

**The justification in the original review was wrong.** This is not about
`preview == committed` — that is about a live tail versus its commit, which take
the *same* path by construction. It is about
`tests/dynamics.rs::a_glaze_lands_the_same_whether_or_not_the_stamp_loop_runs`:
which path a brush takes is decided from axes that have nothing to do with colour
or flow, so nudging `deposit` off zero must not change what the same colour and
the same flow lay down. The two paths have drifted before, by 157 levels, and
both halves of that were quantities of exactly this kind. That test is the guard;
this makes it harder to need.

## 8. `MAX_STAMPS` no longer bounds what its doc claims — **done**

Fixed the sentence, not the chunker. `chunk_segments` caps **segments**, which is
what the constant should say; `dynamics_plan` then adds up to one bleed slot per
segment plus the settle, so a piece plans at most `2 · MAX_STAMPS + 1` slots —
~2.1 MB of uniform buffer. Making the cut count planned slots would couple
`chunk_segments` to the bleed cadence to save a megabyte nothing is short of, so
the factor is now stated and the reason for leaving it alone with it.

Its neighbour had the same class of error: `MAX_REGION_DIM` said a 2048² piece
costs "~34 MB together" for colour and aux. Both are `Rgba16Float` at 8 B/texel,
so each is 32 MiB and the pair is ~67 MB — the figure counted one texture. Now
stated with the arithmetic so it can be checked, plus the note that it is per
*piece*, since `flush` destroys a region as soon as it submits.

Two cross-file invariants asserted in prose nearby both check out:
`WICK_TRAVEL_QUANTUM` = `WICK_HALF / WICK_RATE` = 2/4 = 0.5, and the host's
`BAKE_RES` = the shader's `128u` = its `@workgroup_size(128)`. See item 10 for
making those structural rather than trusted.

## 9. Doc drift — **done**

* `build_dynamics_kit` said "three" compute pipelines / entry points, twice, and
  "one bind group each". It is seven entry points — `snapshot`, `exchange`,
  `wick_x`, `wick_y`, `bake`, `deposit`, `settle` — over five layouts, since the
  two wick axes share `exchange`'s.
* `mod.rs`'s `dynamics` field called the axis `load`; it has been `lift` since
  the rename. That list and the module header both omitted `bleed`, which
  `dynamics_setup` gates on — both now name the four axes the gate actually
  tests.
* Found while sweeping: `snapshot_pipeline` claimed it is dispatched standalone
  "**only for the pen-up**". Bleed slots dispatch it standalone too, and have
  since they were added — the `SlotKind::Bleed` doc even says so. Now stated as
  the two slot kinds with no `exchange` grid to ride in.
* Also fixed the module's one genuinely broken intra-doc link: `safe_frozen`
  pointed at `segments::Taper`, which was private and so unnameable. `Taper` is
  `pub(super)` now, like the `DAB_TRAVEL` and `generate_segments_in` the same
  doc comment links to.

The remaining rustdoc warnings under `gpu/stroke` are all "public documentation
links to private item" on `safe_frozen` and `StrokeCarry::dirty` — those links
resolve, they just point into the module's own internals. Worth a look if item 5
ever moves `safe_frozen` out of `mod.rs`.

## 10. CPU tests for the plan builders — **done**

Nine unit tests in `dynamics.rs`, none needing an adapter, so they run in CI
whether or not there is a GPU. Two extractions made the arithmetic reachable:
`snapshot_size` and `dispatch_rect` are now free functions, with `PlanCtx::rect`
delegating.

Both load-bearing tests were **mutation-checked** rather than merely observed to
pass:

* Reintroducing the historical bug — clamping a bleed window's start to the
  segments in hand instead of walking back along the crossing segment's arc —
  fails `bleed_firings_do_not_depend_on_where_the_stroke_was_cut`.
* Shaving `snapshot_size`'s `+2` fires `dispatch_rect`'s assert through
  `every_dispatch_rect_fits_the_scratch_its_piece_sized` ("a 5x5 dispatch rect
  overruns the 4 snapshot scratch").

The two cross-file invariants are now asserted by reading
`stark_shaders::dynamics()`, plus a third that guards item 1's contract: the
shader's `struct Stamp` must have nine `vec4` lanes and `SLOT` must be those nine.

One thing that could not be checked: `WICK_RATE` is absent from the linked WGSL,
because the WESL linker drops constants that survive only in prose and the shader
computes with `WICK_KERNEL` instead. So the quantum's derivation is asserted
against `WICK_HALF` with the rate written on the host side — which still catches
the realistic change, widening the stencil, and is recorded on the test.

### Original notes

`bleed_fires`, `settle_tangent` and `dynamics_plan`'s rect math are pure,
float-deterministic, and covered only through full GPU renders in
`tests/dynamics.rs`. Three properties are asserted in prose and worth asserting
in code, beside the taper tests in `segments.rs` that already do this for
`generate_segments_in`:

* `bleed_fires`'s headline claim — the firing windows are a pure function of the
  record, independent of the cut. Comparing the `(dist, length)` list for `whole`
  against `head + tail` is a five-line test, and its failure was a visible
  `preview == committed` break.
* ~~Every dispatch rect fits `dsize`~~ — done with item 3, as a `debug_assert!`
  in `PlanCtx::rect` rather than a test: it is now checked on every slot of
  every stroke the suite draws, which is stronger than any case list.
* `settle_tangent` against a trailing cluster of zero-length segments — the exact
  input its doc says produced 0°/−90°/90°/180° on a stroke running at 90°.

Also worth doing here: the module states two cross-file invariants in prose only —
`WICK_TRAVEL_QUANTUM` must equal the shader's `WICK_HALF / WICK_RATE`, and the
host's `BAKE_RES` must equal the shader's. Both hold today (checked while doing
item 8). `stark_shaders::dynamics()` hands back the shader source as a `&str`, so
a test can read the constants straight out of it and assert the pair — CPU-only,
so it runs in CI whether or not there is an adapter.

## 11. Small

* ~~`let deposit_bgs: Vec<_> = (0..1).map(…).collect()`, then always
  `deposit_bgs[0]`~~ — done with item 2.
* `let mut fires = fires.iter().peekable()` shadows the parameter. The consumer
  also relies on `fires` being sorted by segment index — true by construction,
  unstated at the type.
* ~~The coverage-box → dispatch-rect computation is copy-pasted twice with a
  near-variant for the settle~~ — done with item 3, as `PlanCtx::rect`.
* `ScopedResources::is_empty` doubles as "has a piece been recorded yet" in
  `flush`. A `piece_open: bool` says what is meant.
* `swept.rs` recomputes `flatten_tolerance` that `dynamics_setup` already
  computed and discarded on the `Swept` arm. `StrokePath::Swept(tol)` carries it.

## 12. Where the derivations live — **done**

Resolved by the owner: strip the dead ends and the convergence table, keep the
theorem and the derivations in `docs/brush.md` §6.2.

Doing it turned out to be mostly **deduplication**. §6.2 already carried the
column-stochastic transfer-matrix argument, the sliding kernel, the
too-pessimistic-on-both-counts note and the quadrature measurements, nearly
verbatim — and it deferred to the constant for the two things now deleted ("the
convergence table and the four cheaper fixes … are recorded on
`RESERVOIR_EXCHANGE_STEP` itself"). Those two references would have dangled, and
are repaired.

Deleted: the 4×4 convergence table, the four cheaper approaches that do not work,
and every paragraph duplicating §6.2. `RESERVOIR_EXCHANGE_STEP` keeps what a
reader of the *code* needs — what it caps, the rate it is quoted at, what it
bounds, and a `§6.2` cite — going 145 doc lines to 20.

`WICK_TRAVEL_QUANTUM` had the same duplication (its parity/sublattice and
separability derivations are both in §6.2) and was trimmed the same way, keeping
the `WICK_HALF / WICK_RATE` contract and the measured reason it stops at 2.

One thing moved rather than deleted: why the error went unnoticed at a step of
0.5 — the `drain` cap that made every golden render at 13.3 px segments whatever
the step said. Not a dead end and not a derivation, but the lesson generalises
("a golden that does not move is evidence about the test, not about the change"),
so it is now a paragraph in §6.2.

`budget.rs` 393 → 247 lines. Comment-only; the module's rustdoc stays at 0
warnings.

## Original note

`RESERVOIR_EXCHANGE_STEP` carries ~120 lines of doc including a measured 4×4
error table, a theorem about column-stochastic transfer matrices, and four
recorded dead ends; `WICK_TRAVEL_QUANTUM` and `BLEED_TRAVEL_QUANTUM` add ~70
more. The content is worth keeping to the word — but `docs/brush.md` §6.2 already
discusses all three constants by name, so the material is split across two homes
with the deeper half in the file `CLAUDE.md` says is not where design lives.

Moving the derivations into §6.2 and leaving each constant with a short summary
plus a `§` cite would halve `mod.rs`. Against it: proximity is why these
arguments survive refactors. Deliberately left undecided.
