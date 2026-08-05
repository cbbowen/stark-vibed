# `stark-core::document` — review findings

A critical read of [`crates/stark-core/src/document/`](crates/stark-core/src/document/)
(11 files, ~6,300 lines) against the invariants [CLAUDE.md](CLAUDE.md) and
[docs/](docs/) state. Ordered by value, not by file.

Status legend: **open** · **in progress** · **done**.

---

## 1. Bugs

### 1.1 `tiles_covering` enumerates before it checks the cap — **done**

`selection.rs::tiles_covering` materialized the whole `Vec<TileCoord>` and only
then let its callers test the limit (`Selection::plan`, `fill::plan`). A rect
spanning ~10⁶ canvas px is 15.5M coords (~124 MB, collected twice on the fill
path); ~10⁷ px is 6×10⁹ pushes — a wedge plus OOM rather than a refusal. It is
reachable from a marquee drag at far zoom-out, and unconditionally from a
`SelectionOp`/`FillOp` in a loaded file or a peer action, which are external
input the module otherwise treats as hostile (`clamp01`, `taper_px`, `shape_ok`).

The fix pattern already existed one file over: `transform.rs::quad_reached_tiles`
counts with `checked_mul` against a budget *before* walking, and uses `as i64`
rather than a saturating `as i32`. `plan_gated_mask` does the same; selection and
fill were the outliers.

Fixed by giving `tiles_covering` a budget and an `Option` return, splitting the
index arithmetic into `tile_box` (finite-checked, `i64`, refusing anything the
`i32` tile grid cannot address), and turning `fill::plan`'s bounded case around
to walk the **gate** — already capped at `MAX_SELECTION_TILES` — filtering by the
shape's box, instead of walking the shape's box filtering by the gate.

### 1.2 `stroke_rect` under-claims on a non-finite radius — open

`footprint.rs:159-184`: a NaN `radius` gives a NaN pad, `f32::clamp` propagates
NaN, and `NaN as i32` is 0 — so the footprint collapses to a single tile at the
origin. Under-claiming is the direction that silently diverges peers (§12.6).
`BrushParams::taper_px` already normalizes NaN for exactly this reason; either
`radius` gets the same treatment or `stroke_pad` falls back to `TileRect::ALL`
when non-finite.

### 1.3 `remove_in` conflates two meanings of `None` — open

`state.rs:620`: `layers.set(i, …)?` propagates a `Vector::set` out-of-range
failure as "layer not found", which callers turn into a silent no-op. It cannot
happen today (the index came from `position`), but the `?` spells an impossible
case as the one case that matters. `.expect("position is in range")` matches the
two lines above it.

---

## 2. Performance

### 2.1 `observe()` costs two full log scans per pointer sample — open

`stark-ui/src/state.rs::dispatch` calls `observe()` on every command, and
`GestureCommand::To` goes through `dispatch` — i.e. at digitizer rate.
`observe()` reads `can_undo`/`can_redo`, and on `ReplicatedTimeline` each of
those calls `undone_ids(&self.log)`: an O(n) pass building a fresh `HashSet`.
Two of them, per pen sample, growing with session length.

`undone_ids` should be maintained incrementally in `resync` (which already
recomputes `effective_indices`), or cached against a log-length/revision stamp.

### 2.2 `has_backdrop` is O(layers²) inside `observe()` — open

`engine.rs:1855` calls `shown.has_backdrop(l.id)` *inside* `shown.visit(…)` — a
full tree walk per layer. `has_backdrop` is exactly `i > 0 || under`, which the
enclosing visit already has in hand; computing it inline makes it free and
deletes `state.rs:267-280`.

### 2.3 `with_layers` recomputes canvas bounds on every layer mutation — open

`state.rs:533-551` walks every layer's every tile key — and `map_layer` routes
*all* the property setters through it, so setting a layer's name re-derives the
bounds of the whole document. O(total tiles) per action, so a scrub or replay
across N actions is O(N·T). The property setters provably cannot change bounds; a
`with_layers_keeping_bounds` for them is a ten-line change that removes the
majority of the cost. Incremental union on tile writes, with a full recompute
only when tiles are *removed*, would finish the job.

### 2.4 `LinearTimeline::applied()` is O(n) per `scrub_range` — open

Called from `scrub_range`, which the timeline panel reads per render.

---

## 3. Architecture

### 3.1 One action variant, four exhaustive matches, three files — open

Adding an `ActionKind` means editing `apply` (`action.rs:945`), `footprint`
(`footprint.rs:190`), `StatePatch::capture` (`patch.rs:87`) and `label`.
Exhaustiveness gets you the *presence* of an arm; nothing checks
*correspondence*, and CLAUDE.md's first "easy to break silently" rule is
precisely that an `apply` must touch only what its footprint declares.
`tests/commute.rs` covers five hand-written scenarios; there is no generic check.

**The cheap, high-value version is a property test, not a refactor**: for a
corpus of actions, apply to a state, diff the result structurally, and assert
every difference lies inside `footprint(action).writes`. `DocState` is cheap to
clone and the resources are enumerable, so this is mechanical — and it converts a
class of silent divergence into a test failure, which is what "rule out a class
rather than enumerate its instances" asks for here. The full version (colocating
apply/footprint/patch per action so the compiler forces all three) is a large
change; do the test first and let it say whether the refactor earns itself.

### 3.2 `action.rs` is two files — open

Lines 67–622 are the brush model — `BrushShape`, `OrientationSource`,
`BrushDynamics`, `ColorDynamics`, `ModSource`/`PenState`/`Modulation`/
`Modulations`, `BrushParams` — 550 lines with nothing to do with the action log,
documented against §6.2/§6.6 rather than §4. Splitting `document/brush.rs` out
leaves `action.rs` at ~700 lines actually about actions, and matches the doc
split the crate already organizes by.

### 3.3 `DocState`'s derived field is public — open

`bounds` is a pure function of `layers`, but both are `pub` (`state.rs:63-103`).
`layers` has exactly one external reader (`engine.rs:1441`); making both private
behind a `root()` accessor makes "bounds tracks layers" structural rather than a
convention `with_layers` happens to uphold.

### 3.4 Two tile-rectangle types and five copies of the quantizer — partly done

`CanvasBounds` (`state.rs:21`) and `TileRect` (`footprint.rs:27`) are the same
inclusive tile box with different empty encodings. And
`|v: f32| ((v / TILE_SIZE as f32).floor().clamp(-1e9, 1e9)) as i32` appears
verbatim three times in `footprint.rs` and twice more as an `as i64` variant in
`transform.rs`/`selection.rs` — the two versions differing in overflow behaviour,
which is where §1.1 lived.

`selection.rs::tile_box` is now the single finite-checked, range-checked
quantizer; the remaining step is to route `footprint.rs`'s three copies and
`transform.rs`'s two through it (as `TileRect::covering(lo, hi, ring)`), and to
fold `CanvasBounds` onto `TileRect`.

### 3.5 Three near-identical tree walks — open

`carrier_of`, `site_of` and `has_backdrop` (`state.rs:228-280`) are the same
recursion returning three projections of one answer. A single
`locate(id) -> Option<{carrier, index, has_backdrop}>` collapses them;
`move_layer` currently walks the tree three times (`layer`, `remove_in`,
`map_in`) for one move.

### 3.6 `StrokeRecord::tool` is write-only — open

Every construction site sets it; nothing reads it. Per the type's own doc only
`Brush` can reach a `StrokeRecord`, so the field can only ever hold one value —
and it is serialized into every stroke in every save. This is the
inert-scaffolding rule (§1) pointing at existing code. Removing it is a postcard
wire break (§8), so it is a decision rather than a cleanup, but it should be a
conscious one.

---

## 4. Small cleanups

- `footprint.rs:200-217`: the `AddLayer` and `AddMatte` arms are byte-identical;
  an or-pattern `AddLayer { id, carrier, above } | AddMatte { id, carrier, above, .. }`
  merges them (`patch.rs` already does exactly this).
- `action.rs:947-950` and `:981-986`: the same "a matte has no tile map, so a
  stroke targeting one is refused" paragraph appears twice inside one match arm.
- Five arms of `apply` repeat
  `state.layer(x).and_then(|l| l.tiles()).cloned()` → match → `warn` → no-op. One
  `fn on_paint_layer(state, layer, f) -> DocState` puts the matte-refusal rule in
  a single place and removes ~40 lines.
- `ReplicatedTimeline::redo_target` (`timeline.rs:521-528`) inlines
  `undo_target`'s body verbatim as `latest_ordinary`; call it.
- `patch::paint_rect` (`patch.rs:287-296`) rebuilds the entire `Footprint` (two
  `Vec` allocations) to search for one rect, and silently falls back to
  `TileRect::ALL` if it does not find one — an over-restore if the two ever
  drift. A shared `footprint::paint_rect_of(kind, layer)` used by both would make
  them one derivation.
- Type aliases for `HashTrieMap<TileCoord, TilePairHandle>` and
  `HashTrieMap<TileCoord, MaskHandle>`; they appear in a dozen signatures.
- `mod.rs` both `pub mod`s every submodule and re-exports the curated set, giving
  two public paths to everything. Nothing outside `document/` uses the submodule
  paths — `pub(crate) mod` would make the re-export list the actual API.
- `Modulations` needs four edits per new target (field, accessor, `all()`, the
  `PRESSURE_SIZE` literal). Not urgent at six targets, but `all()` going stale is
  a wrong `max_slope`, which is a staircased ramp rather than a compile error.
