# Stroke cleanup

A reading review of `crates/stark-core/src/gpu/stroke` — the six files behind both
render paths (§6.2). Items are ordered by what they cost if left alone, not by
effort. Nothing here was measured on a GPU or pinned by a new test; where a claim
rests on reasoning rather than a run, it says so.

Cited by item name rather than line number throughout, since the line numbers will
drift out from under this file long before the items do.

| # | Item | Kind | Status |
|---|---|---|---|
| 1 | `settle_tangent` is piece-local | correctness | **done** — `d5eb89a` |
| 2 | `dispatch_rect` clips silently in release | correctness | **done** — `907a92b` |
| 3 | Two stale figures from the bleed retuning | docs | **done** — `93c0285` |
| 4 | The swept path is O(segments × tiles) | performance | **done** — `93c0285` |
| 5 | Bind-group churn in the swept per-tile loop | performance | **deferred** (see below) |
| 6 | `affected_tiles` computed twice | performance | **done** — `93c0285` |
| 7 | `Stamp`'s nine anonymous lanes | architecture | **done** — `d5eb89a` |
| 8 | `dynamics.rs` is five modules | architecture | **done** — `907a92b` |
| 9 | A `DynamicsKit` but no `SweptKit` | architecture | **done** — `8745a11` |
| 10 | Minor | | **done** — `93c0285` |

Nine of ten landed on `stroke-cleanup`. The item bodies below are kept as written —
they are the reasoning each change rests on, and the commits cite them.

**Verification.** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets
-D warnings`, and the wasm build are clean; `cargo test --workspace` is 646 passing,
including all 24 goldens compared against real pixels (not `STARK_SKIP_GOLDEN`, and no
adapter was skipped). **No golden moved**, which is the claim the whole set needed: the
per-tile grouping in #4 and the lane repacking in #7 are meant to be bit-identical, and
#1 changes a frame that no golden's stroke apparently exercises differently.

Three tests were added for properties that had none:
`the_settle_frame_does_not_depend_on_where_the_stroke_was_cut` (which carries its own
non-vacuity check — it keeps the old piece-local walk beside it and asserts *that* one
moves), `every_slot_field_lands_in_the_lane_the_shader_reads_it_from`, and
`the_per_tile_lists_hold_exactly_the_segments_that_reach_each_tile`.

**What #5 still wants.** It was the one item marked "measure before touching", and it
still is: the fix is not cheap (a bindless-style array, or compositing the base into
the scratch so the integrate reads one pair), and nothing here established that the
churn is actually the cost. The asymmetry is on the record; the measurement is not.

---

## 1. `settle_tangent` has the defect `bleed_fires` was fixed for

**It looks its window up among the segments in hand.**

`settle_tangent` (`dynamics.rs`) walks back one radius through the `segments` slice
it is handed. That slice is the **piece's** segments — `render_dynamic` cuts the
range with `chunk_segments` and passes `&segments[piece]` to `DynamicsRun::draw`,
which passes it to `dynamics_plan`, which passes it here. A piece is a cut of the
*range*, and the range is not the record.

The live tail renders `head.spans..all` (`engine::live::render_live_stroke`) and,
because `range.end == span_count`, `capture` is false and the tail therefore **does**
run the settle. The commit renders `0..all`. So when the unfrozen tail carries less
than one brush radius of travel — ordinary for a large tip, since `safe_frozen`
holds spans back only for the taper and the touch-down dab — the tail measures its
tangent over a shorter window than the commit does. On a curving stroke those two
directions differ; the settle's `min(owed, received)` lens is elongated *along* the
tangent, so the fade-out cap changes orientation the moment the pointer comes up.
That is a `preview == committed` break (§1.3) in the one place it cannot be
repainted.

This is the same failure the module has already diagnosed and cured once, for the
bleed windows — the note at the walk in `bleed_fires` says it outright:

> Looking the position up meant clamping to the first segment in hand, so a window
> reaching further back than the range being drawn came out short — and a live tail
> always starts at a span boundary while the commit renders the whole stroke from
> zero, so the two relaxed different amounts of paint at exactly that seam.

The settle never got the same treatment.

**The bleed cure does not transplant.** `bleed_fires` walks back along the crossing
segment's own arc, which is history-free. Doing that here would defeat the whole
point of `settle_tangent`: a hand pauses before it lifts, so the last segments are
degenerate edges whose direction is a rounding error — which is precisely what the
one-radius window exists to average away
(`the_settle_frame_ignores_a_paused_hands_arbitrary_last_edges`).

**The fix that fits** is to derive the tangent from the **record** rather than from
any segment slice: flatten the trailing spans of `rec.path` covering one radius,
inside `render_dynamic`, independent of `spans.range`. `dynamics_plan` already
carries `rec` on `PlanCtx`, so nothing new has to be threaded. That makes the answer
a pure function of the record, which is the same standard `dynamics_setup` and the
bleed cadence are already held to.

A cheaper stopgap — a third condition in `safe_frozen`, that no span freezes within
one radius of the tip — fixes the range cut but *not* a chunk boundary landing near
the stroke's end, so it is a mitigation rather than the cure.

**Confidence.** The call graph is verified; the visible artifact is reasoned, not
observed. A failing test is the first thing to write: render a curving dynamics
stroke whole, then as head + tail, and compare the settle slot's `a.zw`.

---

## 2. `dispatch_rect` degrades to a silent clip in release

`dispatch_rect` `debug_assert!`s that the rect fits the snapshot scratch and then
`min`s it in anyway. Its own doc comment admits what that costs: getting the
four-number argument wrong "clips a footprint rather than failing", and a truncated
footprint is wrong pixels with no signal at all.

The four numbers are `snapshot_size`'s `+3` and `+2` against `dispatch_rect`'s
`RECT_MARGIN` and its outward rounding — an argument spread across two functions
that must agree exactly. Per the convention (**rule out a class rather than
enumerate its instances**), the fix is to make the two share one function that
returns the grown-and-rounded rect, so the scratch is sized by the very call the
dispatch later makes and the pair cannot disagree.

`build_dynamics_kit`'s `debug_assert_eq!(color_space.color_format(), Rgba16Float)`
is the same shape and simpler: it runs once at init, so a real `assert!` costs
nothing.

---

## 3. Two stale figures from the bleed-cadence retuning

`a46b3b0` doubled `MAX_BLEED_FIRES_PER_SEGMENT` 8 → 16 and updated that constant's
own doc, but two dependents were missed:

* **`budget.rs`, `MAX_STAMPS`** still quotes the worst-case stamp buffer at
  "~9.4 MB". That was `9 × 4096 × 256`. At sixteen fires it is `17 × 4096 × 256`
  ≈ **17.8 MB**. Not a bug — the buffer is sized from `plan.len()` — but the figure
  is the entire justification for "not worth chunking around", and it has doubled.
* **`dynamics.rs`, inside `bleed_fires`** still reads "Eight covers a tip down to a
  quarter of the brush", contradicting the constant it is capping against and
  `MAX_BLEED_FIRES_PER_SEGMENT`'s own corrected doc.

---

## 4. The swept path is O(segments × tiles)

`render_swept` draws **every** instance into **every** affected tile. At
`SWEEP_SLICES = 8` that is 18 vertices per segment per tile, and an off-tile quad is
discarded only after its vertices have been shaded.

A live tail is bounded — few segments, few tiles — so this does not hurt painting.
What it hurts is every whole-stroke render: commit, replay, goldens. A tapered brush
costs ~211 segments on a *straight line* (`the_segment_budget_is_what_it_was`), so a
long tapered scribble is thousands of segments across hundreds of tiles, nearly all
of the pairs empty.

**The fix reuses work already done.** `affected_tiles` already visits every
(segment, tile) pair to build its set. Have it return the inverted map instead —
tile → segment indices — build the instance buffer grouped by tile, and draw each
tile's contiguous range. Total instances become `Σ tiles-per-segment`, which is the
segment count times a small constant, rather than `segments × tiles`. The change is
local to `swept.rs` plus one signature in `segments.rs`.

The dynamics path already does the equivalent, per piece, through `chunk_segments`.

---

## 5. Bind-group churn in the swept per-tile loop

The module invests heavily in *not* allocating per tile per pointer move — that is
what `UNIFORM_STRIDE` and `ScopedResources` are for, and the reasoning on
`UNIFORM_STRIDE` is explicit that the allocation **rate** is what JS GC cannot keep
up with.

`render_swept` nonetheless builds a fresh `integrate_bg` per tile per render,
because the base colour/aux views vary per tile. Bind groups have no `destroy()`, so
they are pure GC pressure — exactly the pattern the buffers were restructured away
from.

Worth measuring before acting: the fix is not cheap (a bindless-style array, or
compositing the base into the scratch so the integrate reads one pair). Listed here
so the asymmetry is on the record rather than rediscovered.

---

## 6. `affected_tiles` is computed twice over the whole stroke

`render_dynamic` builds it over *all* segments to fill `StrokeCarry::dirty`, then
`DynamicsRun::draw` rebuilds it per piece. The pieces partition the segments, so the
union of the per-piece sets **is** the whole set: accumulate `dirty` from `draw`'s
return and delete the first call.

Each call is a `BTreeSet` insert per tile per segment — the very cost `region_of`
was introduced to avoid, per its own doc ("*on a long stroke, the very cost the
incremental repaint exists to avoid*").

---

## 7. `Stamp`'s nine anonymous lanes are the module's largest unchecked surface

The uniform is nine `vec4<f32>` named `a`–`i` (`dynamics.wesl`), with all meaning
carried in prose on both sides. `dynamics_plan` fills them at three sites — segment,
bleed, settle — with wholly different semantics per lane: **108 positional float
slots, none of them compiler-checked.** The mirror generator's `offset_of`
assertions check lane *offsets*, not what lives inside a lane, so nothing would
catch `lambda(s.deposit)` and `lambda(s.lift)` written the wrong way round.

That this drifts is recorded in the file's own history. The note above the `Stamp`
import says the previous host-side copy "still described `e.zw` as the midpoint
`exchange` samples the canvas at, some time after the shader had stopped reading the
lane at all" — and `e.w` is *still* documented in the WESL as the last of that lane
going spare.

Two fixes, and both are worth doing:

* **Name the members.** Declare the WESL struct with named scalar members in the
  same order (`radius`, `travel`, `lambda_lift`, `lambda_deposit`, …). Uniform
  layout rules permit it and the packing is unchanged, so the mirror generator emits
  named host fields instead of `b: [f32; 4]`. Every read in the shader and every
  write on the host becomes checkable, and the lane map stops being prose.
* **Constructors, regardless of the above.** Replace the three inline
  `Stamp { … }` literals with `Stamp::segment(…)` / `::bleed(…)` / `::settle(…)`
  beside `SlotCommon`. `dynamics_plan` is ~230 lines that are mostly literal, and
  the three kinds' *differences* — the readable content, and the thing `SlotKind`
  exists to make legible — are buried inside them.

---

## 8. `dynamics.rs` is five modules

At 2485 lines it holds run orchestration (`DynamicsRun`), plan construction
(`dynamics_plan`, `bleed_fires`, `settle_tangent`, `dispatch_rect`,
`snapshot_size`), ~340 lines of pipeline construction (`build_dynamics_kit`), path
selection (`dynamics_setup`) — and `build_integrate_pipeline`, which belongs to the
**swept** path and is here only by accident.

The plan half is pure CPU float math and already carries the module's most
interesting tests: cut-independence of the bleed firings, the rect/scratch fit, the
firings tiling their own arc. That is exactly the virtue `budget.rs` claims for
itself —

> Nothing here touches the GPU. It is float arithmetic over a `BrushParams`, which
> is what lets the segment-budget tests pin it exactly.

— and it applies verbatim to the plan builders, which today can only be reached
through a file that pulls in `wgpu`.

Suggested split:

```
dynamics/plan.rs   pure, GPU-free, testable: the plan builders + their tests
dynamics/kit.rs    build_dynamics_kit
dynamics/run.rs    DynamicsRun: recording, regions, write-back
```

and move `build_integrate_pipeline` to `swept.rs`, beside the pass that uses it.

---

## 9. A `DynamicsKit` but no `SweptKit`

The dynamics path's GPU objects are bundled behind one type. The swept path's
(`pipeline`, `uniform_bgl`, `prefix_bgl`, `integrate_pipeline`, `integrate_bgl`) sit
loose among the caches on a 15-field `StrokeRenderer` that three files `impl`
against. Two extractions turn `mod.rs` into composition rather than storage:

* **`SweptKit`**, mirroring `DynamicsKit`.
* **`TipCache`** (or `BrushAssets`), owning `round_tip`, `noise_cache`,
  `noise_sampler`, `dummy_noise`, `noise_bgl` and the four resolvers `prefix_view` /
  `coverage_view` / `round_tip` / `noise_view`. These are one coherent thing — the
  textures a brush resolves to — and they are the only mutable state on a struct
  otherwise documented as holding "only immutable GPU objects plus `Arc`-backed
  handles".

That last complaint is one the module has already made and fixed once, in the other
direction: `DynamicsKit`'s doc records that it "used to carry the round tip's
coverage cache, the one mutable thing in a struct documented as built-once", moved
out to the renderer. The renderer is now where all of it has piled up.

---

## 10. Minor

* **`bleed_fires` guards after it divides.** It computes `crossings` from `s.dist /
  bq` and only then checks `bq <= 1e-3`. It is safe — a zero `bq` gives `NaN`,
  `NaN < 1.0` is false, and control falls through to the guard — but only by
  accident of NaN ordering. Radius is floored at 0.5 in `generate_segments_in`, so
  the guard is dead code for any real segment; reordering costs nothing and removes
  the puzzle.
* **Poisoned-mutex panics on the render path.** `round_tip` and `noise_cache` both
  `.lock().expect("… poisoned")`. Poisoning here means a panic inside a pure bake
  function, so the cached value cannot be torn; `PoisonError::into_inner` is the
  honest recovery.

---

## What the module looks like now

```
stroke/
  mod.rs           375   composition: the two kits, the tip cache, the entry points
  budget.rs        613   what a stroke may cost (unchanged)
  incremental.rs   152   drawing a stroke in pieces (unchanged)
  segments.rs     1988   path -> segments, and the tile inversion #4 added
  swept.rs         479   the fast path, its kit, and the integrate that was misfiled
  tips.rs          173   what a brush resolves to, and the bakes behind it
  dynamics/
    mod.rs          92   which path a stroke takes at all
    plan.rs       1426   what to dispatch — names no wgpu, and its tests
    kit.rs         422   the objects it is dispatched with
    run.rs        1050   recording: regions, the ping-pong, the write-back
```

The 2967-line `dynamics.rs` is gone, and `mod.rs` is down from 585. What moved is
where a maintainer reads, not what the engine depends on — the module's public surface
is unchanged, and `plan.rs` is the one that earns its own file twice over, being both
the hardest half to reason about and the only one testable on any machine.

## If these are picked up again

Beyond #5, two things this pass noticed and did not act on:

* **`segments.rs` is 1988 lines**, over half of it tests, and it now holds three
  separable things: the round tip's coverage field, the path → segment generator with
  its taper and dab, and the region/tile measurements. The same argument as #8 applies,
  with less force — nothing in it is misfiled, it is just large.
* **`Stamp`'s `e.w` is still spare.** The WESL comment has said so through two rounds
  of edits now. Worth either spending or deleting the lane the next time the struct is
  touched, since a lane documented as free is one nobody checks.

The rule the whole set was held to: **fix the model and re-bless**, never a
compensating constant. Nothing needed re-blessing.
