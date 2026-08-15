# `gpu/composite` — the architectural cleanup ledger

Reviewed 2026-08-15 on branch `gpu-cleanup` (`c0c9cf0`), across all 10 files of
`crates/stark-core/src/gpu/composite` (~3.3k lines), plus its callers
(`engine/render.rs::composite_stack`, `gpu/merge`) and the three types it leans on
(`channels::ChannelFormats`/`Targets`, `uniforms::UniformSlots`, `desc`).

**Nothing here landed.** This file records what the review found and what it would
cost; every finding is open. Symbol names are the durable part.

One finding is not new: §8's shared uniform buffers is
`GPU_CLEANUP.md` §7 (deleted in `c0c9cf0`) restated, and its verdict there —
"correct today, holds only while every render is write-then-submit" — still stands.
It is repeated below because the cheap structural fix was not noted at the time.

## The one-sentence summary

**The module's hardest rules are argued in prose and consumed positionally.** The
group tree is walked four times by four functions that must agree on order, depth
and parity; the passes own their pipelines but not their encoding, so `mod.rs`
holds 1464 of the module's 3303 lines; and two mechanisms this crate already owns
(`UniformSlots<T>`, `Targets<'_>`) have one consumer each that reimplements them by
hand. Every fix below promotes something that already exists here rather than
inventing one — which is `GPU_CLEANUP.md`'s finding about `gpu/stroke`'s lessons,
one directory deeper.

| § | Finding | Kind | Status |
|---|---|---|---|
| 1 | Four traversals of the group tree must agree | invariant by convention | open |
| 2 | Every pass owns its pipeline; none owns its encoding | 1464-line `mod.rs` | open |
| 3 | `guides.rs` hand-rolls `UniformSlots` with a literal stride | silent-corruption path | open |
| 4 | Four hand-rolled grow buffers; two allocate for nothing | duplication + waste | open |
| 5 | The supersample budget omits the scratch and the residual | 3–5× understated ceiling | open |
| 6 | Blend/filter bind groups rebuilt per frame over stable views | allocation rate | open |
| 7 | Pass A is one draw and one bind group per tile | scaling cliff | open, measure first |
| 8 | Smaller correctness and clarity items (six) | mixed | open |
| 9 | `scratch_needs` has no unit tests | test gap | open |

---

## 1. Four traversals of the group tree must agree — open

The same tree is walked four times, each producing something the fourth consumes
**positionally**:

| Walk | Where | Produces |
|---|---|---|
| `CompositeGroup::items_into` | `group.rs` | flat instance order (tile + matte streams) |
| `collect` (inside `prepare_composite`) | `mod.rs` | blend/filter uniform slot order |
| `scratch_needs` | `group.rs` | per-level iso/swap requirements |
| `Compositor::encode_stack` | `mod.rs` | the draws, consuming all three through `Cursors` |

`Cursors` exists *only* to re-derive, at encode time, indices the other three walks
already knew. The comment above `collect` — "collected by the **same recursion**
the encoder consumes them with" — is exactly the class of claim the "rule out a
class rather than enumerate its instances" convention says to make structural
rather than assert.

The failure mode is not a compile error. `collect` failing to recurse into a
`Stack` would render every group through its *sibling's* blend mode, and no golden
distinguishes that when the two modes happen to match. The disagreement that *is*
caught panics mid-encode at `here.expect("a merge without scratch targets")` — and
that message is wrong for the case that also depends on it: a **filter** needs
`here` to be `Some` too, because with no level entry `swap == target`, `cur == alt`,
and the pass reads and writes one texture.

**What to do.** One walk producing a flat instruction list, then a dumb encoder:

```rust
/// Which trio a step names, resolved against the caller's targets at encode time.
enum Slot { Target, Swap(usize), Iso(usize) }

enum Step {
    Draw { into: Slot, tiles: Range<u32>, mattes: Range<u32>, clear: bool },
    Clear { into: Slot },
    Blend { back: Slot, src: Slot, out: Slot, uniform: u32 },
    Filter { back: Slot, out: Slot, uniform: u32 },
}

struct Plan {
    steps: Vec<Step>,
    instances: Vec<Instance>,
    mattes: Vec<MatteInstance>,
    ramps: Vec<Ramp>,
    blends: Vec<BlendUniform>,
    filters: Vec<FilterUniform>,
    scratch: Vec<bool>,      // what `scratch_needs` returns today, as a byproduct
}
```

What it buys:

- The parity dance, `Cursors` and `scratch_needs` all become byproducts of one
  walk. Slot indices are **written into the step**, so they cannot drift from the
  buffer they index.
- `Plan` is pure data with no GPU in it, so the whole of §14.7 — parity lands the
  result in the caller's own targets, a level with only filters allocates no iso, a
  free group costs no pass — becomes **CPU-unit-testable without an adapter**. None
  of it is today, and the GPU half of the suite is the slow, adapter-specific half.
- Adjacent `Draw` steps into the same slot can be coalesced in the plan rather than
  by an encoder noticing.
- `encode_composite` becomes `for step in &plan.steps { … }`, and
  `composite_channels` and `render` provably run the identical pass A — which is
  currently argued in `prepare_composite`'s doc comment.
- `Plan` lives on the `Compositor` and is `clear()`ed per frame, which also removes
  the seven per-frame `Vec` allocations in `prepare_composite` — the very cost
  `items_into`'s "appends rather than returning a `Vec`" comment cites one level
  down.

---

## 2. Every pass owns its pipeline; none owns its encoding — open

`mod.rs`'s own header promises "Five passes, one per module, each owning its
uniform, its pipeline and the constants only it reads". True of construction, false
of encoding: `encode_blend`, `encode_filter`, the overlay's instance gathering, the
guides' packing and bind group, and the media uniform's assembly all live in
`mod.rs`. `Compositor::render` alone is 280 lines.

The asymmetry has a visible cost. `encode_filter` reaches into
`e.p.blend.pigment` — a real and deliberate cross-pass dependency (both passes ask
the LUT the same question, so there is one layout per space rather than one per
pass) that is **invisible from `filter.rs`**, where a reader would look for it.
Give each pass an `encode` taking exactly what it needs and the coupling appears in
a signature:

```rust
impl FilterPass {
    fn encode(&self, ctx: &GpuContext, enc: &mut CommandEncoder,
              pigment: &PigmentLut, slots: &UniformSlots<FilterUniform>,
              back: Targets<'_>, out: Targets<'_>, slot: u32);
}
```

~200 lines leave `mod.rs`, which is then holding what its doc says it holds: the
two structs the passes hang off, and the walk that orders them. Best done after §1,
which makes the move mechanical.

---

## 3. `guides.rs` hand-rolls `UniformSlots` with a literal stride — open

`GUIDE_SLOT: u64 = 512` is hand-maintained, with `guide_buf`/`guide_slots` on
`Compositor` and a manual grow-and-write loop in `render`. `GuideUniform` is 21
`vec4`s = 336 bytes, so `UniformSlots::<GuideUniform>::STRIDE` computes **exactly
512**. It is a drop-in replacement.

Worth doing not for the ~25 lines but for the trap the current shape keeps open.
Add 11 more `vec4`s to `guides.wesl` — the comment records that the fisheye's second
set of poles already forced one bump from 256 — and `GUIDE_SLOT` under-strides.
Every guide past the first then reads a mangled overlap of its neighbour's slot: no
compile error, no validation error, wrong pixels only when two guides are visible at
once. `UniformSlots<T>` deriving its stride from the type is precisely the guarantee
`GPU_CLEANUP.md` §4 promoted this mechanism for; one of its three would-be consumers
never got it.

---

## 4. Four hand-rolled grow buffers; two allocate for nothing — open

`instances`/`instance_cap`, `matte_instances`/`matte_cap`,
`overlay_instances`/`overlay_cap`, `guide_buf`/`guide_slots` — four copies of *check
cap, realloc, update cap, write*. `UniformSlots<T>` is the uniform-side answer; the
vertex side wants its sibling:

```rust
struct InstanceStream<T> { buf: wgpu::Buffer, cap: usize, label: &'static str, _t: PhantomData<T> }
impl<T: bytemuck::Pod> InstanceStream<T> {
    fn write(&mut self, device: &Device, queue: &Queue, items: &[T]) -> bool; // non-empty
    fn slice(&self) -> wgpu::BufferSlice<'_>;
}
```

Independently, and worth fixing on its own: `alloc_instances` (`tiles.rs`) and
`alloc_overlay` (`overlay.rs`) use `create_buffer_init` with `vec![Default; count]`.
That builds and uploads an N-element CPU vector of placeholder values which the very
next `write_buffer` overwrites in full. `alloc_mattes` and `alloc_guides` correctly
use a plain `create_buffer`. Three of the four should match the two that are right.

---

## 5. The supersample budget omits the scratch and the residual — open

`MAX_SUPERSAMPLED_PX` (`resolve.rs`) documents "At 16 Mpx that is ~210 MB in the
worst case, which is the most a painting canvas may quietly take to stop sparkling",
and its enumeration *includes* the blend scratch — but the arithmetic cannot. At
16 Mpx:

| | Oklab (10 B/px) | Mixbox (18 B/px) |
|---|---|---|
| accumulator trio | 168 MB | 302 MB |
| `ss_target` (4 B/px) | 67 MB | 67 MB |
| **per scratch level** (swap + iso) | **+336 MB** | **+604 MB** |

A pigment document with one blend-mode layer, zoomed out on a large window, reaches
~970 MB; two levels of nesting ~1.6 GB. That is not "quietly", and it lands on
exactly the documents that are heaviest already. The crossing is also a cliff —
`ensure_targets` drops and rebuilds every attachment at once.

The ceiling is a **bytes** budget wearing a pixel budget's clothes. `supersample`
already takes `&wgpu::Limits`; give it the two facts it is missing and keep the
budget in bytes:

```rust
fn supersample(size: Extent2, zoom: f32, limits: &Limits,
               bytes_per_px: u32,   // from ChannelFormats
               levels: usize)       // from scratch_needs, already computed this frame
    -> u32
```

That also makes the navigator-versus-window trade the comment describes fall out of
the rule rather than coincide with it.

---

## 6. Blend/filter bind groups rebuilt per frame over stable views — open

`encode_blend` and `encode_filter` each create a `BindGroup` per merge per frame.
But the views they name are entirely determined by `(level, parity)`: at level `l`,
`target(l)`, `swap(l)` and `iso(l)` are fixed for the whole render, and `(cur, alt)`
takes exactly two values. So **at most two blend bind groups and two filter bind
groups per level** exist, ever.

Cache them on `ScratchLevel`, built lazily, invalidated by the event that already
drops the scratch (`ensure_targets`). Same argument `TilePairHandle::composite_bg`
already makes for tiles (`GPU_CLEANUP.md` §1), and it matters more on the web, where
each `create_bind_group` is a JS object plus validation on the frame path.

While there: the slotted-uniform `BindGroupEntry` is written out verbatim three
times in `mod.rs` (matte ramp, blend, filter), each restating a `size_of::<T>()`
that `UniformSlots<T>` already knows. `desc::uniform_entry` covers the whole-buffer
case; the slot case wants `UniformSlots::<T>::binding(&self, binding: u32)`.

---

## 7. Pass A is one draw and one bind group per tile — open, measure first

Not a defect; it is the ceiling on this module. `TILE_SIZE` is 254, so a 2560×1440
view is ~77 tiles per fully-painted layer, and a 20-layer document is ~1540
`set_bind_group` + `draw` pairs per frame. Fine on native. On the web that is
~185k WebGPU calls a second, and the encode *is* the frame.

The structural fix is for `TilePool` to allocate channel textures as **2D array
layers** rather than standalone textures, so every tile in a run shares one bind
group and `Instance` carries an array index beside `origin`. Pass A becomes one
`draw(0..4, 0..n)` per run. `desc::load_tex_array` already exists
(`stroke/swept.rs` uses it) and the pool already centralizes allocation, so the
change is contained to `tile.rs`, `composite.wesl` and the instance mirror.

Real work, and not to be done speculatively — but it is the lever, and the per-tile
bind group is the one thing standing between here and a single draw call.
Measure wasm frame time first; if pass A encode is not the wall, leave it.

---

## 8. Smaller correctness and clarity items — open

**`composite_channels` takes three views where `Targets` exists.** It takes
`color`, `aux`, `resid: Option<_>` and then `debug_assert_eq!`s the residual's
presence against the color space — re-checking, by convention and in debug builds
only, the exact invariant `channels.rs` was written to make structural ("a trio
cannot be built half-residual"). Take `Targets<'_>`; the pick path is the one
caller and already holds a trio.

**`Deref` from `CompositorPipeline` to `CompositorPasses`** buys spelling
stability and costs a real trap: `p.media` is the `MediaPass` (through the `Deref`)
while `p.media()` is `MediaParams`, and `CompositorPipeline::offscreen` reads
`media: &self.media` meaning the pass. Rename the field to `media_pass`, or the
accessor to `media_params()`. Dropping the `Deref` for an explicit `p.passes()`
costs one token per read and makes the split visible where it means something,
which is what the doc says the split is for.

**`Compositor::new` allocates attachments the first render may discard.** The
constructor calls `pipeline.offscreen(size)` and duplicates five of
`ensure_targets`'s assignments; `size` is only advisory, since the first render
decides `ss`. Start empty and let `ensure_targets` be the single place attachments
are built. A real saving for the navigator, whose `Offscreen::get(p, view.viewport)`
passes the *un*-supersampled size and then immediately reallocates at 2–4×.

**Shared mutable uniform buffers behind `&self`** — `GPU_CLEANUP.md` §7 restated.
`p.media.buf`, `p.resolve.buf` and `p.view.buf` live on the `Arc<CompositorPasses>`
and are written per render through the queue. The soundness argument is correct
*for serialized renders*, and nothing in the types enforces serialization while the
whole design intent is several `Compositor`s and several `Engine`s sharing one.
Latent rather than live (the frontend is single-threaded). The note not made last
time is that **the fix is nearly free**: the media *bind group* is already
per-`Compositor`, so moving `media.buf` onto `Compositor` costs one small buffer
each and turns the convention into a type. Same for `resolve.buf`, which is already
paired with a per-`Compositor` bind group inside `ss_target`.

**Dead uniform lanes.** `MediaUniform.light` is written
`[0.0, 0.0, 0.0, height_strength]`, and `media_common.wesl` reads only `m.light.w`
(in its relief `strength`). Three inert lanes under a name describing something
that no longer exists — the "do not add inert scaffolding" convention applied to a
leftover rather than to a new field. Rename to what it is, or fold `.w` into
`shade` and delete the lane.

**The opening clear is its own render pass.** When a stack begins with a merge or a
filter, `clear_targets` encodes a pass that only clears — at supersampled size, a
16 Mpx clear of two or three attachments. Cheap, but avoidable: a `backdrop_empty`
flag in `BlendUniform` would let the first merge skip the read entirely and fold the
clear into its own write.

---

## 9. `scratch_needs` has no unit tests — open

`group.rs`'s tests cover opacity folding thoroughly, and
`collapsing_a_free_group_keeps_its_members_folded_opacities` pins a real historical
bug. But `scratch_needs` — the function whose disagreement with `encode_stack` is a
mid-frame panic or a wrong-target render — has none, and `tests/groups.rs` reaches
only one or two levels of nesting through document scenarios that need an adapter.

If §1 lands, the tests come free and are properties rather than cases:

- every `Slot::Swap(l)` / `Slot::Iso(l)` any step names has an entry in
  `plan.scratch`, and `Iso(l)` implies `scratch[l]`;
- no step's `out` equals its `back`;
- the last step's `out` is `Slot::Target` (the parity claim, currently guaranteed
  only by arithmetic in a comment).

Without §1 they are still worth writing directly against `scratch_needs`.

---

## What the review did not find

No bug against a shipped behaviour. The invariants that matter most are already
structural: `as_direct_run` returning the run instead of a `bool` so three call
sites cannot disagree about what "direct" implies; `CompositeGroup::leaf` taking the
opacity *off* the merge in the same breath as it folds it into the items, so the
double fade is unrepresentable; `ChannelFormats` deciding the residual once from the
space; the process-wide `generation` counter, whose doc is right that a per-pipeline
counter would let a replaced pipeline hand back a value a stale consumer is holding.
The parity trick that lands the final accumulator in the caller's own targets is
correct as written, including for filters and for the empty stack — it is the
*derivation* of the count it depends on (§1) that is unguarded, not the trick.

`engine/render.rs::composite_stack` already coalesces adjacent direct groups into
one `Run` via `as_direct_run_mut`, so the "consecutive runs cost a render pass each"
concern does not arise: runs are only split where a merge genuinely separates them.

## Verification

None — nothing landed. Anything taken from this file should be taken green:
`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -D warnings`,
`cargo test --workspace` redirected once to a file, and the wasm build. §1, §2, §3,
§4 and §6 must all be **bit-identical** and so are checkable by the goldens rendered
rather than skipped; §5 changes which `ss` a zoomed-out view picks and therefore
re-blesses nothing at `zoom = 1.0`, where every golden is.
