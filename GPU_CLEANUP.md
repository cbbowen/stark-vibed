# `gpu/` — the architectural cleanup ledger

Reviewed 2026-08-15 on master (`83ab453`), across all 39 files of
`crates/stark-core/src/gpu` (~18k lines); **acted on the same day** on branch
`gpu-cleanup`, one commit per finding. This file records what the review found,
what landed, and what was deliberately left. Symbol names are the durable part.

Nothing here was a bug report against a shipped behaviour. Every finding was
either a **scaling cliff** (a cost that is fine today and quadratic in something
the artist controls) or a **discipline that exists in one subsystem and has not
reached its siblings**.

## The one-sentence summary

**`gpu/stroke` learned four lessons the hard way — bind-group and buffer
allocation *rate*, submit-ordered release, dynamic-offset uniform slots, and
pooling views alongside textures — and `composite`, `transform`, `merge`, `fill`
and `selection` had learned none of them.** Every finding below was a consequence
of that, and the fix in every case was to promote a type that already existed in
this crate rather than to invent one.

| § | Finding | Status |
|---|---|---|
| 1 | A bind group per tile, per layer, per frame | **done** — `5a90ca8` |
| 2 | O(tiles) scratch held to a single submit | **done** — `9a911d4` |
| 3 | The trim's `destroy()` sharpened an open hazard | **done** — `5ee29ec` |
| 4 | Three uniform mechanisms, one user each | **done**, one site deferred — `98d776b` |
| 5 | The channel trio spelled four ways | **done** — `84d31f2` |
| 6 | Readback GC, `items()`, the repeated pass loop | **done** — `f2a87a6` |
| 7 | Shared uniform buffers across sibling engines | noted, no change |

---

## 1. A bind group per tile, per layer, per frame — **done**

`prepare_composite` rebuilt `tile_bgs` from scratch on every render, and pass C
did the same for every mask tile of every live selection. Visible tile count
scales as 1/zoom², so a zoomed-out multi-layer document was creating ~10⁵ wgpu
objects a frame — on the web a JS object apiece, the allocation *rate*
`ScopedResources` and `Pooled` already exist to keep down (§6.2).
`STROKE_LATENCY.md` step 7 named the same cost from the frame-scheduling side.

**What landed.** A tile's texels are never rewritten once a commit lands —
copy-on-write hands out a fresh tile instead (§5.2), the property
`TilePairHandle::same` already rests on — so a bind group naming its three views
describes it for as long as it exists. It lives on the tile now, in a `OnceLock`
filled on first composite, reclaimed with the textures it names. No cache, no
key, no eviction policy: the lifetime is exactly the tile's.

The layout cannot change under it either, which is the half worth checking before
trusting this: it answers to the compositor's `tile_bgl`, a function of the color
space alone (§6.7); a sibling engine is handed the very same
`Arc<CompositorPasses>` (`new_sharing`), and the one thing that builds a different
one is a color-space rebuild, which replaces the pool and requires an empty
document (`rebuild_gpu_for`).

Mask tiles took the same treatment under a deliberately narrower name,
`overlay_bg`: a mask is bound through three layouts and one slot can only answer
for one of them.

## 2. O(tiles) scratch held to a single submit — **done**

`MergeRenderer::apply` recorded every tile into one encoder and submitted once,
holding all of it live until then. The blended path takes three scratch trios per
tile, and a merge has no cap: ~15 GB at 10k tiles, and ~40,000 render passes in
one command buffer, which is a Windows TDR as much as an allocation failure.
`transform` had the same shape, survivable only because `MAX_TRANSFORM_TILES`
caps it at 1024.

**What landed.** `gpu::submit::TileScope` — the pool-free half of the stroke
path's `SubmitScope`, for the renderers that rewrite whole tiles. Both hand-written
`Recording` types are gone. `finish` takes `self` and `flush` submits before it
releases, so a call site never holds anything loose; `tile_done` marks the one
point where a tile's scratch is safe to hand back, and every `FLUSH_TILES` of
those submits and releases.

Cutting the recording is invisible because tiles are independent — checked at each
of the four call sites, and the two per-operation values that must outlive a flush
(the fill's `ubuf`, the selection's edge texture) are deliberately *not* scoped,
and say so. `a_scope_hands_its_scratch_back_as_it_goes` pins the bound.

The two scopes stayed separate types, and each says why: the stroke's holds
`ScratchPool` leases whose release must be unforgeable and is therefore private to
that module; this one holds pooled handles whose release is their `Drop`.

## 3. The trim's `destroy()` sharpened an open hazard — **done**

"No live handle" is not "no pending GPU work". A texture whose last handle drops
while an unsubmitted encoder still names its view reaches the free list early.
Reuse alone makes that wrong pixels; `destroy()` makes it a dangling view handed
to the next bind group — a device error, from a pool that cannot see which of its
consumers was careful.

**What landed.** Each slot is stamped when handed back, and a trim may only take
slots stamped before the current epoch opened. An epoch is `TRIM_INTERVAL`
acquires and an encoder spans one operation, so a slot that has survived a whole
epoch is long past any encoder that could name it. Reuse is untouched; this delays
only the destroy, and costs a burst one extra epoch before its surplus drains.

Ruling the class out at the pool rather than trusting each consumer is the point:
§2 made every current caller careful, and this makes a future careless one merely
wasteful instead of fatal. `quarantine_passed` is a free function over the stamps
rather than the slots, so the rule is decidable without a GPU.

## 4. Three uniform mechanisms, one user each — **done, one site deferred**

`UniformSlots<T>` and `UNIFORM_SLOT` lived in `composite::blend` and were used
only there; `gpu::stroke` kept a third copy of the law as a bare `UNIFORM_STRIDE`.

**What landed.** `UniformSlots` moves to `gpu::uniforms`, and its would-be users
pick it up: the **matte ramps** (the per-*frame* one, so the only one that mattered
at frame rate — a buffer and a bind group per gradient matte per render become one
of each per frame, and the shared `zero_ramp_bg` disappears, a solid matte's slot
simply being zeroed); the **fill's per-tile origin**; the **selection's per-tile
params**.

**Deferred, and `quad_bg` now says so at the site:** the transform keeps a buffer
per draw. Its draw count is the sum over the plan of the sources reaching each
destination, so it is not known before encoding starts, and growing a slot buffer
mid-encode would reallocate under the bind groups already recorded against it. Its
buffers are at least registered with the `TileScope` now, so they are destroyed at
their submit rather than left for the GC. Slotting them properly means pre-counting
the plan — a change to the plan's shape, not to that function.

## 5. The channel trio spelled four ways — **done**

`merge::Trio`, `transform::Parcel`, `blend::Trio` and bare
`(TexHandle, TexHandle, Option<TexHandle>)` tuples, four sets of accessors, and
the residual's `Option` threaded by hand at about a dozen sites.
`desc::tile_attachments` and `Targets::attachments` were the same function twice.

**What landed.** `gpu::channels` holds three views of one thing: `ChannelFormats`
(the formats, plus `targets()`/`blended()` for pipeline declarations), `Channels`
(owned handles, with `acquire` and `into_tile`), and `Targets` (borrowed, with
`attachments`/`count`), which moved out of `composite::blend` because it was never
pass A's alone. `TilePairHandle::targets()` lets a tile answer as one.
`desc::tile_attachments` is deleted, and eight constructors take the trio as one
value instead of two or three parallel format parameters.

The failure mode that justified the type is a missing attachment on a pigment
document — a validation error rather than a wrong pixel, invisible to the
Oklab-only half of the suite. As a convention it was a dozen call sites; as
`ChannelFormats` a trio cannot be built half-residual.

Two places still spell their targets out, and say why: pass A, whose three do not
share a blend (the height aux is additive where the color and its residual
composite `over`), and the swept path's scratch, whose aux is the wider
`SCRATCH_AUX_FORMAT`.

**Still open, deliberately:** the `[..2 + usize::from(resid)]` idiom survives in
`stroke/swept.rs` and `stroke/dynamics/kit.rs`, where it slices *bind-group entry*
arrays rather than attachments. That is the same rule about the same `Option`, but
`ChannelFormats` does not fit it — the entries are not a channel trio — and
inventing a second abstraction for two sites would cost more than it saves.

## 6. Readback GC, `items()`, the repeated pass loop — **done**

- **Readback buffers waited on the GC.** `take_rows` now takes the buffer by value
  and destroys it. Safe for `ScopedResources`' reason: the copy that filled it has
  already been submitted *and* waited on.
- **`CompositeGroup::items` allocated a `Vec` per group per frame**, on the path §1
  had just been cleaned. It appends into a caller-supplied `Vec` now — what its own
  inner `walk` was already doing.
- **`TileScope::fullscreen_pass`** absorbs what the merge, the fill and the
  transform's combine each spelled out. The selection's rasterize keeps its own: it
  writes a single R8 mask target cleared to the coverage that reigns outside the
  selection, which is neither a channel trio nor a `CLEAR`.

## 7. Shared uniform buffers across sibling engines — noted, no change

`CompositorPasses` shares its `view`, `media` and `resolve` uniform buffers across
sibling engines, sound because each render writes them through the queue
immediately before the submit that reads them, and submits on one queue execute in
order. That argument is correct today and is written down where it lives. It holds
only while every render is write-then-submit with nothing interleaved, which no
current path violates. If a render is ever split so a write and its submit straddle
another render, this becomes a cross-engine corruption with no test that would name
it. Worth a `debug_assert` if that shape is ever proposed.

---

## What the review did not find

The invariants that matter most were already structural rather than checked, and
none of the above disturbed them: `SubmitScope` owning its leases so release cannot
precede a submit; `as_direct_run` returning the run instead of a bool so three call
sites cannot disagree about what "direct" implies; `UniformSlots<T>` deriving its
stride from the type; `Zeroes` making "there is no tile here" one question with one
answer; `TilePool`'s trim policy proved in four tests rather than tuned. The
`AllocSource` census and the `QUAD_STRIP` equality test are both the right shape — a
guarantee the compiler or a test keeps, rather than a comment.

## Verification

Every commit was taken green: `cargo fmt --all`, `cargo clippy --workspace
--all-targets -D warnings`, `cargo test --workspace`, and the wasm build. 815
tests pass, 24 of them goldens rendered and compared rather than skipped — which
is the check that matters for §1 and §5, both of which must be bit-identical and
are.

Two tests were added for behaviour that had none:
`a_scope_hands_its_scratch_back_as_it_goes` (§2's bound) and
`a_trim_never_destroys_a_slot_returned_this_epoch` with
`the_quarantine_takes_the_old_slots_oldest_first` (§3's rule).

One regression was caught in flight by `export_omits_the_selection_outline`: a
layout entry has to become `uniform_slot` in the same breath as the offset appears
at the draw. A validation error rather than a wrong pixel — the good kind.
