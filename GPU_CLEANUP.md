# `gpu/` — the architectural cleanup ledger

Reviewed 2026-08-15 on master (`83ab453`), across all 39 files of
`crates/stark-core/src/gpu` (~18k lines). This file records what the review
found and what to do about it. Symbol names are the durable part; the file
paths are stable, the line numbers are of that date.

Nothing here is a bug report against a shipped behaviour. Every finding is
either a **scaling cliff** (a cost that is fine today and quadratic in
something the artist controls) or a **discipline that exists in one subsystem
and has not reached its siblings**.

## The one-sentence summary

**`gpu/stroke` learned four lessons the hard way — bind-group and buffer
allocation *rate*, submit-ordered release, dynamic-offset uniform slots, and
pooling views alongside textures — and `composite`, `transform`, `merge`,
`fill` and `selection` have learned none of them.** Every finding below is a
consequence of that, and the fix in every case is to promote a type that
already exists in this crate rather than to invent one.

---

## 1. The compositor allocates a bind group per tile, per layer, per frame

**Where.** `composite/mod.rs::prepare_composite` rebuilds `tile_bgs` from
scratch on every call, which is every render. Same shape for the overlay's
mask tiles in `Compositor::render`, and gradient mattes additionally
`create_buffer_init` a fresh uniform buffer per matte per frame.

**Why it is a cliff.** Visible tile count scales as 1/zoom² — a 3840-px
viewport at zoom 0.1 covers ~12,800 tiles per layer, so twenty layers is ~10⁵
bind-group creations per frame. On the web that is a JS object apiece, which
is precisely the allocation *rate* — not amount — that `ScopedResources`
(`stroke/mod.rs`) and `Pooled` (`tile.rs`) were written to keep off the stroke
path. The compositor is the only hot path in the module with no such
discipline at all. `STROKE_LATENCY.md` step 7 names the same thing from the
frame-scheduling side, independently.

**The fix, and why it is unusually clean here.** A tile's texels are never
rewritten once committed — copy-on-write hands out a fresh tile instead
(§5.2) — so **a bind group over a tile is valid for that tile's entire life.**
That means no cache and no eviction policy: put a `OnceLock<wgpu::BindGroup>`
inside `TilePair`, filled on first composite. Its lifetime becomes exactly the
tile's, and the pool reclaims it with the textures.

`TilePairHandle::same` already asserts the identity that makes this sound. The
one thing to write down is that the layout is the color space's and fixed for
the document's life (§6.7) — a color-space rebuild replaces every tile anyway.

The matte ramps should go through `UniformSlots<Ramp>`, which already exists
two files away (see §4 below).

## 2. `merge` and `transform` hold O(tiles) of scratch live until one submit

**Where.** `MergeRenderer::apply` and `apply_filter` record every tile into one
encoder and submit once. The blended path (`encode_blended`) acquires **three**
scratch trios per tile and parks all of them in `Recording::scratch` until that
submit. `TransformRenderer` has the same shape.

**Why it is a cliff.** Merge is documented as infallible — "there is no map to
be unusable and no cap to exceed" — which is true of the *result*: it spans a
union of tiles the document already holds. It is not true of the *scratch*.
Merging two full-canvas layers through a non-`Normal` mode at 10k tiles wants
~15 GB of simultaneously-live pooled textures, and puts ~40,000 render passes
in a single command buffer, which is a Windows TDR waiting to happen.
`apply_filter` has the same shape and explicitly rewrites every tile with no
passthrough-by-handle.

`transform` is bounded by `MAX_TRANSFORM_TILES = 1024`, so ~800 MB peak —
survivable, but for that reason rather than by design.

**The fix is already written.** `stroke::scratch::SubmitScope::flush` submits
and *then* releases piece-scoped leases, which is exactly the rule
`Recording` re-derives by hand in both files. `merge.rs` says so out loud:
"`TransformRenderer`'s `Recording` learned that the hard way; this is the same
guard." Two hand-written copies of a rule the crate has a type for is the
signal.

So: **promote `SubmitScope` out of `gpu::stroke` into `gpu::`**, and have
transform, merge, fill and selection record through it, flushing every N
tiles. Both `Recording` types then delete themselves, peak transient memory
becomes O(N) rather than O(tiles), and command buffers stay driver-sized.

## 3. The trim sharpened an already-open hazard

**Where.** `PoolInner::tick` calls `slot.tex.destroy()` on free-list entries.

**What changed.** A texture reaching the free list while an unsubmitted encoder
still names its view used to be a wrong-pixels bug — the class `SubmitScope`
was built for, and which the `TilePool` general case is on record as still
open on. With `destroy()` in the picture, the same mistake is now a dangling
view handed to a bind group: a device error rather than a silent smear. The
trim is right and should stay; what it needs is for the irreversible half to
be ordered.

**Two fixes, either cheap:**

- **Quarantine one epoch.** An entry returned to `free` is not eligible for
  `destroy()` until it has survived a tick. Reuse stays immediate; only the
  irreversible operation waits.
- **Better: record the submission index** in `give`, and make eligibility a
  comparison against the completed index. That closes the *reuse* half too,
  and is the structural version of what §2's fix achieves by construction.

## 4. Three mechanisms for "vary a uniform across draws", one user each

`UNIFORM_STRIDE` (`stroke/mod.rs`) and `UniformSlots<T>`
(`composite/blend.rs`) both encode the same 256-byte dynamic-offset law, and
neither is used outside the file it was born in. Meanwhile:

- `transform.rs::quad_bg` and `gated_uniform_bg` create a buffer **and** a bind
  group per source quad per destination tile
- `fill.rs` creates a buffer per tile to carry four floats (`TileUniform`)
- `selection.rs::rasterize` creates one per mask tile

`UniformSlots<T>` has no compositor dependency. Move it to `gpu::desc`, let
`UNIFORM_STRIDE` become `UniformSlots::<T>::STRIDE`, and each of the three
sites above collapses to one buffer and one bind group.

## 5. The channel trio is spelled four different ways

`TilePairHandle`, `merge::Trio`, `transform::Parcel` and
`composite::blend::Trio`/`Targets` are all "(color, aux, maybe resid)", each
with its own accessors. `desc::tile_attachments` and `Targets::attachments`
are the same function written twice.

The residual `Option` (§6.7) is then threaded by hand at roughly a dozen sites:
`if resid { entries.push(…) }`, `2 + usize::from(resid.is_some())`,
`.map(|f| pool.acquire_tex(f, …))`, and the `views_of` / `color_of` /
`aux_of` / `resid_of` quartet in `merge.rs`.

**This is the module's most-repeated shape and its most likely source of a
future residual bug.** The failure mode is a missing attachment on a pigment
document, which an Oklab-only test cannot see. A `Channels<T>` — owned
handles, borrowed views, formats — owned by the color space, with
`attachments()`, `count()` and `acquire(pool, source)` on it, collapses all of
it. `Targets` is already most of the right type; it is in the wrong file.

## 6. Smaller, and independent of the above

- **Readback buffers wait on GC.** `readback.rs` unmaps but never `destroy()`s.
  An 8192² export parks 268 MB against the module's own stated doctrine
  (`ScopedResources`: dropping releases the JS handle and waits for GC, which
  is how the tab OOMs). One line after `take_rows`.
- **`CompositeGroup::items()` allocates per group per frame**
  (`composite/group.rs`), and `prepare_composite` `flat_map`s over it. The
  inner `walk` already appends to a caller-supplied `Vec`; expose that instead.
- **Six copies of the same per-tile encode loop** — `fill.rs`, `merge.rs`
  (twice), `transform.rs` (twice), `selection.rs`. Each is: acquire
  destination trio → per-tile uniform → bind group → one pass → insert into the
  CoW map → submit at the end. Findings 2, 4 and 5 all want to land in the same
  place; a single `tile_pass` helper taking a closure for the per-tile bind
  entries is that place.

## 7. Noted, not a finding

`CompositorPasses` shares its `view`, `media` and `resolve` uniform buffers
across sibling engines, sound because each render writes them through the queue
immediately before the submit that reads them, and submits on one queue execute
in order. That argument is correct today and is written down where it lives.
It holds only while every render is write-then-submit with nothing interleaved,
which no current path violates. If a render is ever split so a write and its
submit straddle another render, this becomes a cross-engine corruption with no
test that would name it. Worth a `debug_assert` if that shape is ever proposed.

---

## Order of work

1. **§1** — the only finding on the per-frame path, and the
   `OnceLock`-on-the-tile version is contained and made obviously correct by
   the CoW invariant.
2. **§2** — the only finding that can take the process down, and the type it
   needs already exists; it just lives inside `gpu::stroke`.
3. **§3** — small, and turns a latent class into an impossibility.
4. **§5** then **§4** — in that order, because a `Channels` type is what makes
   the six-way encode-loop fold in §6 worth doing at all.

## What the review did not find

The invariants that matter most are already structural rather than checked, and
none of the above disturbs them: `SubmitScope` owning its leases so release
cannot precede a submit; `as_direct_run` returning the run instead of a bool so
three call sites cannot disagree about what "direct" implies; `UniformSlots<T>`
deriving its stride from the type; `Zeroes` making "there is no tile here" one
question with one answer; `TilePool`'s trim policy proved in four tests rather
than tuned. The `AllocSource` census and the `QUAD_STRIP` equality test are
both the right shape — a guarantee the compiler or a test keeps, rather than a
comment.
