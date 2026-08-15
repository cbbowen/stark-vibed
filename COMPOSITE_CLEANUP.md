# `gpu/composite` — the architectural cleanup ledger

Reviewed 2026-08-15 on branch `gpu-cleanup` (`c0c9cf0`), across all 10 files of
`crates/stark-core/src/gpu/composite` (~3.3k lines), plus its callers
(`engine/render.rs::composite_stack`, `gpu/merge`) and the three types it leans on
(`channels::ChannelFormats`/`Targets`, `uniforms::UniformSlots`, `desc`).

**Acted on the same day**, one commit per finding. This file records what the review
found, what landed, and what was deliberately left. Symbol names are the durable part.

One finding was not new: §8's shared uniform buffers is `GPU_CLEANUP.md` §7 (deleted
in `c0c9cf0`) restated. Its verdict there — "correct today, holds only while every
render is write-then-submit" — was right; what it missed is that the fix was nearly
free, and the buffers held per-target state anyway.

## The one-sentence summary

**The module's hardest rules were argued in prose and consumed positionally.** The
group tree was walked four times by four functions that had to agree on order, depth
and parity; the passes owned their pipelines but not their encoding, so `mod.rs` held
1464 of the module's 3303 lines; and two mechanisms this crate already owns
(`UniformSlots<T>`, `Targets<'_>`) had one consumer each reimplementing them by hand.
Every fix promoted something that already existed here rather than inventing one —
which is `GPU_CLEANUP.md`'s finding about `gpu/stroke`'s lessons, one directory deeper.

| § | Finding | Status |
|---|---|---|
| 1 | Four traversals of the group tree must agree | **done** — `5eb05f3` |
| 2 | Every pass owns its pipeline; none owns its encoding | **done** — `c71a7f5` |
| 3 | `guides.rs` hand-rolls `UniformSlots` with a literal stride | **done** — `8d7bda1` |
| 4 | Four hand-rolled grow buffers; two allocate for nothing | **done** — `ec82183` |
| 5 | The supersample budget omits the scratch and the residual | **done** — `2e8c3ca` |
| 6 | Blend/filter bind groups rebuilt per frame over stable views | **done** — `8d7bda1`, `53826f9`; regressed and fixed — `4a3f5cd` |
| 7 | Pass A is one draw and one bind group per tile | **not done** — measure first |
| 8 | Smaller correctness and clarity items (six) | **five done** — `eeef655`, `3a5bd1e`; one left |
| 9 | `scratch_needs` has no unit tests | **done** — `5eb05f3` |

---

## 1. Four traversals of the group tree must agree — **done**

The same tree was walked four times, and three of those walks produced something the
fourth consumed **positionally**: `CompositeGroup::items_into` the flat instance
order, `collect` the blend and filter uniform slots, `scratch_needs` the per-level
isolation, and `Compositor::encode_stack` the draws — reconstructing all three
indices as it went with a `Cursors`.

Nothing tied them together. `collect`'s doc asserted it was "the same recursion the
encoder consumes them with", which was true and was a sentence. A `collect` that
failed to recurse into a `Stack` would have rendered every group through its
*sibling's* blend mode, silently, and identically to the correct result whenever the
two modes happened to agree. A `scratch_needs` that disagreed about depth panicked
mid-encode at `expect("a merge without scratch targets")` — a message that did not
cover the **filter** case which depended on the same entry, since without it
`swap == target` and the pass reads and writes one texture.

**What landed.** `composite::plan` is one walk producing a `Vec<Step>`. Each step
names the targets it reads and writes as a `Slot` — `Target`, `Swap(l)`, `Iso(l)` —
and carries the uniform slot it binds. The parity that lands the accumulator in the
caller's own targets, the level a group isolates into, and which slot a merge takes
are byproducts of one pass, so they cannot drift: there is only one order now.
`Compositor::encode_plan` is a `match` over the steps with no recursion, no cursors
and no parity arithmetic.

Being view-free is what makes the plan buildable before the attachments are sized,
which is what §5 needed. The one view-dependent number in pass A — a chromatic
filter's dispersion (§21.10) — is resolved at `upload`, once the sample count has
settled. That is why `Plan::filters` holds *descriptions* where `Plan::blends` holds
uniforms, and it is the only asymmetry in the type.

`items_into`, `scratch_needs`, `collect`, `chromatic_disp`, `Cursors` and
`encode_stack` are all gone.

## 2. Every pass owns its pipeline; none owns its encoding — **done**

`mod.rs`'s header promised "five passes, one per module, each owning its uniform, its
pipeline and the constants only it reads". True of construction, false of encoding:
every `begin_render_pass` in the module lived in `mod.rs`, and `render` alone was 260
lines.

The asymmetry had a cost worth naming. `encode_filter` reached into
`e.p.blend.pigment` — a real and deliberate coupling (both passes ask the LUT the
same question, and an Oklab document binds the same 1×1 stand-in, so there is one
table per space rather than one per pass) that was **invisible from `filter.rs`**,
where a reader would look for it.

**What landed.** `TilePass::encode`, `BlendPass::encode`, `FilterPass::encode`,
`MediaPass::encode` (with the uniform assembly behind a `MediaScene`),
`OverlayPass::encode` (with the instance gathering), `GuidePass::encode` (with the
packing) and `ResolvePass::encode`. The pigment LUT is a parameter of the filter
pass's now. `Supersampled` moved to `resolve.rs` — the set exists only for that pass
— and builds its own bind group and uniform. `Bounce`, which is what a merge and a
filter both need beyond the pipeline kit, lives in `blend.rs` where the level it
names lives, and carries the render pass both encode.

`mod.rs` 1464 → 994 lines; `render` 260 → 90.

## 3. `guides.rs` hand-rolled `UniformSlots` with a literal stride — **done**

`GUIDE_SLOT: u64 = 512` was hand-maintained, with `guide_buf`/`guide_slots` on
`Compositor`, its own `alloc_guides`, its own grow loop and its own offset
arithmetic.

It was *correct*: 512 is exactly what `UniformSlots` computes for a 336-byte `Guide`.
It would have stayed correct until the next time the uniform grew, which had already
happened once (the fisheye's second set of poles, §20.8). Past 512 it under-strides,
and two visible guides read each other's lanes with no compile error, no validation
error, and nothing in a golden to say so.

**What landed.** The pass takes `UniformSlots<GuideUniform>` like the other two.
`UniformSlots::<T>::layout()` and `binding()` put *both* ends of the slot ABI on the
type that already knew the stride — seven layout sites and five bind-group sites had
been restating `size_of::<T>()`, and `gpu::fill` and `gpu::selection` had their own
copies of the entry. The two sizes can no longer disagree.

## 4. Four hand-rolled grow buffers; two allocated for nothing — **done**

`instances`/`instance_cap`, `matte_instances`/`matte_cap`,
`overlay_instances`/`overlay_cap`, `guide_buf`/`guide_slots` — four copies of *check
cap, realloc, update cap, write*.

**What landed.** `InstanceStream<T>` beside `UniformSlots<T>` in `gpu::uniforms`, with
the slot law removed and the growth policy kept: a vertex buffer is indexed by the
draw's own instance range, so there is no alignment quantum and no offset to get
wrong. `Compositor` loses three field pairs.

Independently: `alloc_instances` and `alloc_overlay` went through `create_buffer_init`
with a `vec![Default; count]`, building and uploading a CPU-side vector of placeholder
records that the very next `write_buffer` overwrote in full. The tail past
`items.len()` is stale either way — a draw names its own instance range and never
reaches it — so a bare `create_buffer` says what was already true.

## 5. The supersample budget omitted the scratch and the residual — **done**

`MAX_SUPERSAMPLED_PX = 16 << 20` was documented as "~210 MB in the worst case", and
its enumeration of what scales with the sample count **included** "the blend scratch
if the document has a mode in it". No pixel count can include that: the scratch is two
viewport-sized trios per isolating level, which is a fact about the *document*, and
the residual is 8 more bytes a texel, which is a fact about the *color space*.

The real figures at 16 Mpx: 234 MB flat Oklab, 571 MB with one blend group, 973 MB for
the same in pigment, ~1.6 GB at two levels of nesting. Four to seven times the stated
ceiling, on exactly the documents that are heaviest already, and reallocated in one go
each time a wheel-zoom crosses a threshold.

**What landed.** `MAX_SUPERSAMPLED_BYTES = 224 << 20` — the figure that was always
intended — and `supersample` takes what one supersampled texel of *this* frame costs.
`resolve::attachment_bytes` computes it from `ChannelFormats::bytes_per_px` and the
frame's `Plan::scratch`, which is why `render` builds the plan before choosing the
sample count rather than after.

A flat Oklab document is unchanged, deliberately and by test: 14 bytes a texel into
224 MiB is exactly the 16 Mpx it used to get. What moves is the case the old
arithmetic was wrong about — a nested pigment document on a large window now gives up
samples instead of a gigabyte.

## 6. Blend/filter bind groups rebuilt per frame over stable views — **done**

Two halves. The bind-group **entry** was written out verbatim three times in `mod.rs`,
each restating a `size_of::<T>()` that `UniformSlots<T>` already knew — closed by §3's
`binding()`.

The other half: `encode_blend` and `encode_filter` built a `BindGroup` on every merge
of every frame, over views that do not change. A pass at level `l` reads either that
level's `swap` or the stack's own target, and a merge's source is always that level's
`iso`. Two phases, so two of each per level — however many merges the document has,
and however many frames it is drawn for.

**What landed.** The plan is what made the key available without a lookup: each
bouncing step records which way round the ping-pong was when it was *decided*, so
`Phase { level, back_is_swap }` rides on the step and the encoder indexes straight
into the level's `OnceLock` pair.

**This shipped broken, and the shape of the mistake is worth keeping.** The
justification above was "no key, no eviction policy — `ensure_targets` drops the whole
scratch whenever the accumulator is rebuilt, so the lifetime is exactly the views'".
That is true of the *textures* and false of the group as a whole: it also names the
pass's **uniform buffer**, and a frame with more merges than any before it does not
resize that buffer, it *replaces* it (`UniformSlots::write`). The kept bind group then
pointed at a buffer too small for the offset it was about to be given —

```
Dynamic Offset[0] (256) is out of bounds of [Buffer "stark blend uniform"]
with a size of 256 and a bound range of (offset: 0, size: 16).
```

— which is a validation error rather than a wrong pixel, so **every golden still
passed**. Two or more non-`Normal` layers failed to render at all.

The reason nothing caught it is the reason it is recorded here: it needs *two renders
through one compositor*, the second with more merges than the first. `render_to_image`
— the backbone of the whole suite — takes a fresh `Offscreen` every call, and a fresh
compositor sizes its uniform buffer before it builds anything over it. Only the two
consumers that keep a compositor for the life of the app, the surface and the
navigator, could reach it.

`UniformSlots::write` now reports whether the buffer moved, and `Compositor::upload`
— the one place that knows, and the one both callers go through — drops the scratch's
cached groups when it did. That also covers a case the original design would not have:
the eyedropper shares these uniforms with the screen, so a pick with more merges than
any render can stale the render path's cache. `a_kept_offscreen_survives_a_frame_with_
more_merges_than_the_last` pins it, and fails with the exact validation error above
when the fix is reverted.

## 7. Pass A is one draw and one bind group per tile — **not done**

Not a defect; it is the ceiling on this module. `TILE_SIZE` is 254, so a 2560×1440
view is ~77 tiles per fully-painted layer, and a 20-layer document is ~1540
`set_bind_group` + `draw` pairs per frame. Fine on native. On the web that is ~185k
WebGPU calls a second, and the encode *is* the frame.

The structural fix is for `TilePool` to allocate channel textures as **2D array
layers** rather than standalone textures, so every tile in a run shares one bind group
and `Instance` carries an array index beside `origin`. Pass A becomes one
`draw(0..4, 0..n)` per run. `desc::load_tex_array` already exists (`stroke/swept.rs`
uses it) and the pool already centralizes allocation, so the change is contained to
`tile.rs`, `composite.wesl` and the instance mirror.

**Left deliberately**, on the review's own advice: measure wasm frame time first. The
work is real and speculative in equal measure, and nothing above depends on it. §1
makes it easier rather than harder — a `Step::Draw` already carries a contiguous range
of the flat streams, which is the shape a single instanced draw wants.

## 8. Smaller correctness and clarity items — **five of six done**

**`composite_channels` takes a `Targets`** — `3a5bd1e`. It took three views and then
`debug_assert`ed that the residual's presence matched the color space: re-checking, in
debug builds only, the invariant `ChannelFormats` was written to make unsayable. A
pigment document's pass A writes three attachments, so a caller offering two is
missing one — a validation error the Oklab half of the suite cannot see.
`channel_formats` hands back the whole `ChannelFormats` for the same reason.

**`media` is `media_pass`** — `3a5bd1e`. `CompositorPipeline` reaches
`CompositorPasses` through `Deref` and has a `media()` of its own returning the params,
so `p.media` and `p.media()` were two different things one character apart — and the
field, being the `Deref`'d one, was the half a reader least expects. The `Deref` itself
stays: it buys a genuine spelling saving, and the collision was the whole of its cost.

**`Compositor::new` no longer takes a size** — `3a5bd1e`. It allocated a
full-viewport accumulator that `ensure_targets` overwrote on the very first render,
because only a render knows the zoom and therefore the supersampled size. The
accumulator is an `Option` built there and nowhere else, so the sizing rule lives in
one place rather than two that could disagree. `Offscreen::get` and `GpuBuild` both
lose a parameter that was telling a reader something untrue.

**The uniforms a render writes belong to the renderer** — `eeef655`.
`CompositorPasses` is `Arc`-shared across sibling engines and reached through
`&CompositorPipeline`, and it held the view, media and resolve uniform buffers. Sound
on "each render writes immediately before the submit that reads it, and submits on one
queue are ordered" — a property of the *call sequence*, not of the types. All three
hold per-target state anyway (what this render is looking at, how it is lit, how many
samples it took), which is the sharper argument: two targets disagree about them by
construction. `ViewBindings`, `media_buf` and `Supersampled` now live on the
`Compositor`; what stays shared is the pipelines, the layouts and the tile sampler.

**The media uniform's `light` lane is gone** — `3a5bd1e`. Image-based lighting took
over the direction its `.xyz` once held; only `.w` was read, and only for the relief
slope. `shade.w` was free.

**The opening clear is still its own render pass** — **not done**. When a stack begins
with a merge or a filter, `clear_targets` encodes a pass that only clears. A
`backdrop_empty` flag in `BlendUniform` would let the first bounce skip the read and
fold the clear into its own write. Left because the trade is the wrong way round: a
clear-only render pass is fixed-function and near-free on every backend this runs on,
while the flag adds a lane and a branch to the blend ABI that every golden depends on.
Worth revisiting only if a capture ever shows the clear costing something.

## 9. `scratch_needs` had no unit tests — **done**

The function whose disagreement with `encode_stack` was a mid-frame panic or a
wrong-target render had none, and `tests/groups.rs` reached only one or two levels of
nesting through document scenarios that need an adapter.

**What landed.** §1 made it free, because `Plan` is plain data: eleven tests in
`composite::plan`, of which two are properties run over thirteen shapes.

- `every_slot_is_allocated` — every `Slot` a step names is one the scratch was told to
  allocate, and an `Iso` implies the level *isolates*. This is the invariant that used
  to live in two functions with a matched pair of `if`s holding them together.
- `lands_in_the_callers_targets` — the last step always writes `Slot::Target`. The
  parity claim, which was previously guaranteed only by arithmetic in a comment. It is
  `debug_assert`ed in `encode_plan` too, so a shape the tests do not know still trips
  it at the point of use.

The rest pin the cases: a plain stack is one cleared draw and no scratch, an empty
stack still clears, odd and even bounce counts start on the right side, a stack opening
with a merge clears on its own, a filter-only level never allocates an `iso`, nesting
costs a level per level, a group of plain layers consumes none of its own, the two slot
counters are dense and independent, and an inner merge takes the lower slot.

---

## What the review did not find

No bug against a shipped behaviour. The invariants that matter most were already
structural and none of the above disturbed them: `as_direct_run` returning the run
instead of a `bool` so three call sites cannot disagree about what "direct" implies;
`CompositeGroup::leaf` taking the opacity *off* the merge in the same breath as it
folds it into the items; `ChannelFormats` deciding the residual once from the space;
the process-wide `generation` counter, whose doc is right that a per-pipeline counter
would let a replaced pipeline hand back a value a stale consumer is holding. The
parity trick was correct as written, including for filters and the empty stack — it
was the *derivation* of the count it depends on that was unguarded, not the trick.

`engine/render.rs::composite_stack` already coalesces adjacent direct groups into one
`Run` via `as_direct_run_mut`, so the "consecutive runs cost a render pass each"
concern does not arise: runs are split only where a bounce genuinely separates them.

## Verification

Every commit was taken green: `cargo fmt --all --check`, `cargo clippy --workspace
--all-targets -D warnings`, `cargo test --workspace` redirected once to a file, and
the wasm build. 838 tests pass, with the goldens **rendered and compared** rather than
skipped (`STARK_SKIP_GOLDEN` unset) — which is the check that matters for §1, §2, §3,
§4, §6 and §8, all of which must be and are bit-identical.

§5 is the one change that alters output, and only where the old arithmetic was wrong:
it lowers the sample count for nested or pigment documents on a large zoomed-out
window. No golden is blessed anywhere but `zoom = 1.0`, where `supersample` returns 1
and this is a no-op, so nothing needed re-blessing.

Seventeen tests were added for behaviour that had none: eleven in `composite::plan`
(§9), five in `composite::resolve` — the three existing `supersample` cases carried
across to the byte budget, plus `a_flat_oklab_frame_still_stops_at_sixteen_megapixels`
(§5's compatibility claim) and `the_blend_scratch_is_most_of_what_a_nested_frame_costs`
(why it was worth changing) — and one in `tests/composite.rs` for §6's regression.

**The gap that let §6 ship broken is worth naming on its own**, because it is not
specific to that finding. `render_to_image` takes a fresh `Offscreen` every call, so
the entire golden suite renders each engine **once**. Nothing about *frame N+1* was
tested at all — and "grew since the last frame" is precisely the axis every
grow-on-demand mechanism here lives on (`UniformSlots`, `InstanceStream`, the scratch
levels, the attachments). `a_kept_offscreen_renders_what_a_fresh_one_would` was the
only test in that shape, and it varies size, lighting and color space rather than
*count*. The new test covers merges; the other counts — mattes, guides, filters,
outlined mask tiles — are still only ever rendered once per compositor.
