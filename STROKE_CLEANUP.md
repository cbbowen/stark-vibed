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

## 5. Module boundaries

`mod.rs` is ~1000 lines, of which the renderer is ~350. The rest is the cadence
constants plus `flatten_tolerance` / `exchange_travel` / `safe_frozen` (~470
lines), and the incremental-render protocol (`StrokeSpans`, `StrokeCarry`,
`ToolState`). The tell is `segments.rs`'s tests reaching through `super::super::`
eleven times. A `budget.rs` would leave `mod.rs` as the renderer it claims to be.

Smaller: `ViewUniform` and `TileInstance` are declared in `swept.rs` and used
only by `dynamics.rs`. `swept.rs` touches neither.

## 6. Three caches, three policies, one in the wrong place

`round_prefix` (single-entry, on the renderer), `noise_cache` (unbounded `Vec`,
on the renderer), and `round_cov` — single-entry but living in `DynamicsKit`, a
struct documented as immutable GPU objects built once.

Both round caches key on `hardness.to_bits()` and both call
`round_coverage(hardness, ROUND_RES)` — 65,536 texels with a `powf` each — so a
dynamics round brush computes the identical coverage field twice. One
`BrushTextures` cache keyed on the shape, returning `{ prefix, coverage }`, fixes
the layering, the duplicate work, and the single-entry thrash when a user
alternates two hardnesses.

## 7. The two paths derive shared stroke constants independently

`grain_uv = surface.relief * grain_uv_scale()` appears in both `swept.rs` and
`dynamics.rs`; so does the `rgb_to_channels` dance. These are exactly the
quantities that must agree between the paths for `preview == committed`, and they
agree by the same line being written twice. A shared
`StrokeConstants::new(rec, surface, color_space)` makes the agreement structural
— the "rule out a class rather than enumerate its instances" convention.

## 8. `MAX_STAMPS` no longer bounds what its doc claims

It says "cap on the segments one piece dispatches, which bounds its stamp uniform
buffer". `chunk_segments` enforces it on segments, but `dynamics_plan` then
appends up to one bleed slot per segment plus a settle slot, so the buffer can
reach `2·MAX_STAMPS + 1` slots. Harmless today — the binding window is one slot
and the buffer is far under `maxBufferSize` — but the stated bound is off by 2×.
Either chunk on planned slots or fix the sentence.

## 9. Doc drift

* `build_dynamics_kit` says "three" compute pipelines / entry points, twice. The
  shader has seven: `wick_x`, `wick_y`, `bake`, `snapshot`, `exchange`,
  `deposit`, `settle`.
* `mod.rs`'s `dynamics` field still calls the axis `load`; it has been `lift`
  since the rename. That list and the module header both omit `bleed`, which
  `dynamics_setup` does gate on.

## 10. CPU tests for the plan builders

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

## 12. Open question: where the derivations live

`RESERVOIR_EXCHANGE_STEP` carries ~120 lines of doc including a measured 4×4
error table, a theorem about column-stochastic transfer matrices, and four
recorded dead ends; `WICK_TRAVEL_QUANTUM` and `BLEED_TRAVEL_QUANTUM` add ~70
more. The content is worth keeping to the word — but `docs/brush.md` §6.2 already
discusses all three constants by name, so the material is split across two homes
with the deeper half in the file `CLAUDE.md` says is not where design lives.

Moving the derivations into §6.2 and leaving each constant with a short summary
plus a `§` cite would halve `mod.rs`. Against it: proximity is why these
arguments survive refactors. Deliberately left undecided.
