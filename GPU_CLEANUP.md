# GPU module cleanup

A review of `crates/stark-core/src/gpu` (18 files, ~12.6k lines), recorded as a
working plan. Like `DOCUMENT_CLEANUP.md` before it, this file is a checklist with
a lifetime: it is deleted when the passes below are done, because the reasoning
for each change belongs in the commit that made it and in the doc comments around
the code it explains.

The module's arguments are sound where they are written down. Nearly every
problem here is in the **seams** — something the module reasons about correctly in
one place and then re-implements informally in five others. Several findings are
the module's own stated conventions applied to code that predates them.

Three passes, separable and in order of value per unit risk:

- **(a) Latent breaks** — small, high value, no pixels move. §1. **Done.**
- **(b) Helper extraction** — large, purely mechanical, goldens bit-identical. §3.
  **Done.**
- **(c) `composite.rs` split** — structural, no behaviour change. §2.1, §2.6.
  **Done.**

§2.2, §2.4, §2.5 and §4 are what is left. §2.7 is withdrawn — see there.

Run `cargo test --workspace` once after each pass, not after each edit.

---

## 1. Latent breaks — pass (a) — **done**

537 passed, 0 failed, including 25 goldens compared against real pixels (no
`STARK_SKIP_GOLDEN`, no adapter skip) — so the pass moved nothing visible, which
is what it claimed.

Two findings were confirmed by the fix rather than by inspection: the three
drifted byte counts in §1.2 only compiled once corrected, and the scratch-aux
omission in §1.1 was real — and slightly worse than described below. The pool had
no free list for that format *at all*; scratch aux was being served out of the
colour format's, because the two are the same enum variant and therefore the same
`HashMap` key. Correct today, and correct for the wrong reason.

### 1.1 The tile pool's format contract is one coincidence from panicking

`stroke/mod.rs::acquire_scratch` acquires `SCRATCH_AUX_FORMAT`, but the pool is
built with `[cs.color_format(), cs.aux_format(), MASK_FORMAT]` (`engine.rs`).
`acquire_tex` panics `"unsupported format"` on anything else. It works today only
because `SCRATCH_AUX_FORMAT == Rgba16Float == Oklab's `color_format()`.

This is exactly the failure `acquire_tile`'s own doc comment describes as fixed:

> the pool previously hardcoded `R16Float` for aux, which happened to match every
> colour space but would have panicked on the first one that chose otherwise (§6.7)

Same bug, one level up. `build_dynamics_kit` has a `debug_assert_eq!` about it,
which is off in release — where the panic lands.

**Fix.** The pool defines `MASK_FORMAT` and `SCRATCH_AUX_FORMAT` itself, so the
pool guarantees them: `TilePool::new` unions its own constants into whatever the
caller passes. No call site can then forget a format the pool already owns, and
the caller's list shrinks to the one thing it actually knows — the colour space's
two.

### 1.2 Three uniform mirrors have stale byte counts

In a module where those comments *are* the cross-boundary safety mechanism:

| Struct | Comment says | Actually | Drifted when |
|---|---|---|---|
| `composite::ViewUniform` | 32 bytes | 48 | — |
| `composite::MediaUniform` | 80 bytes | 96 | `surf_m` (§18.1.2 view rotation) |
| `composite::GuideUniform` | 240 bytes | 304 | the fisheye's second pole set (§20.8) |

A number in a comment that nothing checks is worse than no number: it reads as
verified. The right answer already exists in this module — `dynamics::SLOT` is
`size_of::<Stamp>()` with `const _: () = assert!(SLOT == 144);`.

**Fix.** Every `Mirrors X (N bytes)` comment gets a `const _: () =
assert!(size_of::<T>() == N)` beside it. The comment keeps naming the WESL struct;
the compiler keeps the number honest.

### 1.3 Two Rust structs mirror one WGSL `View`, with nothing tying them

`composite.rs` and `stroke/dynamics.rs` both declare `ViewUniform` for
`composite.wesl`'s `View` — and the second's doc says it "Mirrors `View` …
**exactly**, including the members this path has no use for". They are identical
(48 bytes) today, and the only thing keeping them so is that sentence.

**Fix.** One definition, beside the shader it mirrors, with a constructor that
fills the three members neither site should be choosing for itself
(`TILE_SIZE`, `INTERIOR_UV_SCALE`, `INTERIOR_UV_BIAS`).

### 1.4 Panic in `Drop`, and a silent leak beside it

`tile.rs`'s `Drop for GpuTex` ends `.expect("source not recorded")`. Practically
unreachable — every acquire records its source — but a panic in `Drop` during an
unwind is an abort, which is a steep price for an unreachable branch. In the same
block, a poisoned pool lock makes the `if let Ok(..)` swallow the texture: it is
never returned to the free list and `capacity` never learns, so the pool quietly
grows a replacement.

**Fix.** Recover from the poison rather than leaking (the pool's state is a free
list and a counter; a panic elsewhere cannot leave either meaning something a
return would violate), and saturate the decrement rather than asserting it.

---

## 2. Architecture — pass (c) and beyond

### 2.1 `composite.rs` is seven unrelated passes in one 2515-line file — **done**

Split into `composite/{mod,view,group,tiles,blend,media,overlay,guides,resolve}.rs`,
each pass owning its uniform, its pipeline and the constants only it reads. 2211
lines (after pass (b)) became 2435 across nine files, the growth being module
headers and `use` blocks; `mod.rs` keeps the two structs and the walk that orders
the passes, and is now one concern rather than seven. 538 tests pass, 25 goldens
against real pixels, and the test-name set is unchanged apart from the three
`supersample` tests moving to `composite::resolve::tests`.

What it was: `CompositorPipeline::new` running to ~540 lines of straight-line
construction across composite, matte, blend, overlay, media, resolve and guides,
each a self-contained unit (shader → BGL → layout → pipeline → buffers) with
nothing separating it from the next. `GuideUniform::pack` in particular was pure
§20 perspective-grid math sitting in the middle of a file about compositing tiles;
it is in `guides.rs` now, and `group.rs` came out with no GPU in it at all.

The `Compositor`/`CompositorPipeline` split (does it depend on the target?) was
already a good line. It just wasn't the only line the file needed.

### 2.2 The pool never returns memory, and the largest allocations bypass it

`PoolInner::free` only grows; `capacity` is monotonically increasing (the field
doc says "available to this pool", the log says "increased capacity" — they mean
different things). A session's peak tile count is permanent GPU residency.

Meanwhile the biggest transients — dynamics regions up to `MAX_REGION_DIM`² × 8 B
× 2 = 67 MB — are created and `destroy()`d per piece and never pooled. Two
opposite policies in one subsystem, neither stated as a decision. Either a
high-water decay on the free lists, or an explicit note on why unbounded
retention is right for tiles specifically.

### 2.3 A `TextureView` is created on every acquire, including recycled ones — **done**

The free list holds `Pooled { tex, view }` now, so a recycled slot brings its view
with it and the common acquire creates no wgpu object at all — a `Vec::pop` and an
`Arc::new`. `Drop` clones the view back (an `Arc` bump) rather than the handle
holding an `Option`, which keeps the read path — once per bind group, per tile, per
frame — a plain borrow.

The rate is what made it worth doing: the swept path acquires `2 + 4·tiles`
textures **per pointer move** (a cleared empty base, then a scratch pair and a
destination pair per affected tile), so a stroke over twenty tiles at pen rate was
creating ~10k views a second.

Two things checked rather than assumed. A recycled view stays valid because it was
built from that exact texture with the default descriptor and recycling hands the
same texture back; the only thing that could invalidate it is `destroy()`, and all
four `destroy()` sites in the crate are on non-pooled textures. And `Drop` clones
into the free list while the handle's own copy dies with it, so a slot holds
exactly one view whether checked out or free.

That check turned up the accessors that made it a *grep* rather than a guarantee:
`TilePairHandle::color`/`aux`, `MaskHandle::texture` and `TexHandle::texture` had
zero callers anywhere, tests included, and were the only route to a pooled
`wgpu::Texture` from outside `tile.rs`. They are gone, so the pool hands out views
and never textures — §2.3's safety is now structural rather than verified.

### 2.4 `AllocSource` is diagnostic-only plumbing threaded through every signature

An 11-variant enum on every `acquire_tex`/`acquire_mask`, a `HashMap` mutation
under the pool lock on the hot path, and a decrement in `Drop` — all to populate
one `tracing::debug!`. If it is worth keeping, put it behind a cfg; if it earns
its place, say so on the enum.

### 2.5 `surface.rs` and `environment.rs` are three concerns each

`surface.rs` (778 lines) holds the `SurfaceId` document type + serde, PNG
canonicalization + BLAKE3 hashing, a CPU mirror of `paint_common.wesl`'s tooth
model (`tooth_gate` / `decode_rise` / `rise_ahead` / `tabulate_bearing` — which is
what all four of its tests exercise), and the GPU upload. `environment.rs` embeds
a complete Radiance RGBE decoder. Neither the decoder nor the tooth model touches
the GPU. `surface/{mod,tooth,import}.rs` would let the shader-mirror half be read
as the physics it is.

Relatedly, `downsample_to_limit` is a generic image utility living in
`gpu::surface` and imported by `assets.rs` for *brush shapes*.

### 2.6 `is_direct() ⇒ Run` is enforced by runtime re-matches — **done**

There were **three** sites, not the two this review found. `CompositeGroup::stack`
and `encode_stack` each asked the `bool` and then re-matched `GroupContent` behind
an `unreachable!` — and `Engine`'s stack builder did the same behind an `if let`
with no `else`, returning `true` either way, so a group that answered "direct"
while holding a `Stack` would have been counted as merged and then silently
dropped.

`as_direct_run` / `as_direct_run_mut` return the run itself, so the test and the
extraction are one step and all three sites are total. `engine.rs` no longer
imports `GroupContent` at all. The one assertion left is in `stack`, immediately
under its own proof, because that path consumes the members while `as_direct_run`
borrows them.

### 2.7 Bind-group churn — **withdrawn; the premise was wrong**

What this item said: the module went to real lengths to avoid per-tile *buffers*
(dynamic offsets, `UNIFORM_STRIDE`, `ScopedResources`) but still builds per-tile
*bind groups* every frame, ~500 a frame at 4K, so cache them keyed on the tile
handle's allocation identity.

Three things are wrong with that.

**The arithmetic.** It assumed 128 px tiles. `TILE_SIZE` is 254 (`TILE_TEX` 256
less two aprons), so a 4K viewport spans ~16×10 tiles per layer, not ~500.

**The cause.** `prepare_composite` does not build one bind group per *visible*
tile. `Engine::layer_items` maps over **every populated tile of every visible
layer**, with no viewport cull anywhere — `visible_bounds` exists but is only used
by the UI's frame panel. So the count scales with the *document*, not the viewport,
and the churn is a symptom rather than the thing.

**It is already a known, deliberate deferral**, not a seam. `docs/rendering.md`
§6.3 carries a "Not yet: damage tracking" box saying exactly this — off-screen
tiles are drawn and clipped by the rasterizer rather than skipped, "fine at current
canvas sizes; the obvious first optimization when it stops being" — and
`docs/roadmap.md` schedules it as *Damage tracking / view-AABB cull*. This review
had no business relisting it as a cleanup.

**So a cache would be the wrong fix at the wrong layer.** It would memoize bind
groups for tiles that should not be drawn at all, and to be correct it would need
keep-alive `Arc`s (or it can ABA on a recycled address) — which pins tiles against
the pool, working directly against §5.1's "history retention drives reclamation".
It would also miss on exactly the tiles a stroke is changing, which are the ones
redrawn most often. A view-AABB cull subsumes it: bind groups, draw calls, instance
entries and `TilePairHandle` clones all become proportional to what is on screen.

Nothing to do here as a cleanup. The cull is a roadmap item and belongs to whoever
decides the roadmap.

---

## 3. Mechanical duplication — pass (b) — **done**

All of it now lives in `gpu/desc.rs`. Net **−1073 lines** across the eight files
touched, against 367 added, and the same 538 tests pass with the same names —
25 goldens compared against real pixels, not skipped.

One risk was worth naming rather than assuming away. `QUAD_STRIP` had to be
spelled out field by field, because a `const` cannot call `Default::default()`,
and it replaced ten pipelines that had spelled their primitive state inline. A
field that disagreed would change what those rasterize with nothing failing to
say so — so `quad_strip_is_the_default_with_a_strip_topology` pins it against
both forms it replaced. Every other field these helpers fill in is one wgpu
validates or one the shader ignores.

All the same shape: a helper that exists once, correctly, and is then re-typed
everywhere else. Roughly 800–1000 lines recoverable, none of it behavioural.

| Pattern | Copies | Where |
|---|---|---|
| BGL texture entry closure (`load_tex` / `sample_tex` / `ctex` / `filter_tex` / `tex_entry`) | 10 | composite ×2, fill, selection, transform ×2, dynamics ×4 |
| `clear_attachment` / `const CLEAR` | 5 | composite ×2, fill, transform, dynamics |
| 1×1 constant texture builder | 5 | fill, transform (byte-identical), selection ×2, pigment |
| Bind-group texture entry (`tex` vs `view_entry` — one function, two names) | 2 | dynamics, composite |
| `UNIFORM_STRIDE` slot packing + `write_buffer` | 3 | swept, dynamics ×2 |
| Grow-buffer-if-needed + write | 5 | `Compositor`: instances, mattes, blend, overlay, guides |

Plus **17 `create_render_pipeline` calls**, each repeating `depth_stencil: None,
multisample: Default::default(), multiview_mask: None, cache: None,
compilation_options: Default::default()` — of which **10 are the identical
fullscreen-triangle shape** (fill, selection rasterize, integrate, blend, media,
resolve, guides, transform combine, transform mask_base, slice).
`dynamics::cpipe` is already the right precedent for compute pipelines;
generalizing it to render pipelines is ~250 lines on its own.

Target: a `gpu/common.rs` holding `tex_entry`, `load_tex_entry`, `storage_entry`,
`uniform_entry`, `clear_attachment`, `load_attachment`, `constant_texture`,
`fullscreen_pipeline`, `SlotBuffer`, `GrowBuffer`.

`FillRenderer` and `TransformRenderer` also each allocate their own
`zero_color`/`zero_aux` for the same two formats, built at the same moment in
`build_gpu`, and each carry `ctx` + `color_format` + `aux_format` + `selection`.
A shared `Palette` cloned into each renderer removes the repetition and the
duplicate textures together.

---

## 4. Smaller cleanups

- **`Registry::ensure` is `get` minus the return value.** `register`/`set` can call
  `self.get(gpu, id);`. Both also do `contains_key` then `[&id]` — two hash lookups.
- **7 `#[allow(clippy::too_many_arguments)]`** (transform ×4, composite ×2,
  segments ×1). `render_gated_parcel` takes 10. `dynamics::PlanCtx` is the pattern
  this module already found; apply it in transform.
- **`transform::apply_affine` and `apply_gated` are ~90 near-parallel lines each** —
  plan, encoder, `src_bgs`, `scratch`, rewrite loop → parcel → acquire → combine →
  insert, drops, mask, submit, `drop(scratch)`. Only the parcel/mask function and
  the gate differ. And `drop(scratch); // now safe to recycle` is a
  correctness-critical ordering rule enforced by a comment: recycling a
  still-referenced scratch inside the same encoder would corrupt silently.
  `ScopedResources` solved this class with a type; this should too.
- **`readback::begin_read` takes `bytes_per_texel` from the caller** when
  `texture.format().block_copy_size(None)` is right there. A mismatch produces
  silently wrong rows.
- **Two hand-rolled f16 codecs** in different files: `readback::f16_to_f32`
  (general) and `environment::f32_to_f16` (lossy, radiance-only). Colocate them so
  the asymmetry is visible, or use `half`.
- **`TilePool::free_count` hardcodes `Rgba16Float`** on an otherwise
  format-generic pool, and is `pub` while documented "for tests".
- **`mod.rs` exposes everything twice** — modules are `pub` *and* flat-re-exported,
  so `gpu::composite::Compositor` and `gpu::Compositor` both work, and `engine.rs`
  uses both styles. `readback` is the one module not re-exported.
- **`render_swept` acquires and clears an `empty` base tile unconditionally**, on
  every pointer move, even when every affected tile already exists in `base`.
- **`MediaUniform`'s field order and its write site disagree** (the struct declares
  `surf_a, surf_b, surf_m`; `render` writes `surf_a, surf_m, surf_b`). Harmless in
  Rust, actively confusing when checking against the WESL.
- **`min_binding_size: None` on most uniform BGL entries.** `swept.rs` gets this
  right with `XFORM_SLOT`; selection, fill, transform and composite's
  view/media/resolve all pass `None`. It is free validation against a truncated
  write.
