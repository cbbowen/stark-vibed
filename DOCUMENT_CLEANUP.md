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

### 1.2 `stroke_rect` under-claims on a non-finite radius — **done**

A NaN `radius` gave a NaN pad, `f32::clamp` propagated it, and `NaN as i32` is 0
— so the footprint collapsed to a single tile at the origin, which a distant
stroke would then commute past. Under-claiming is the one direction §12.6 cannot
survive: the fast path splices on the lie and no pixel can show it. Two more of
the same shape sat beside it — `f32::min`/`max` return the *non*-NaN operand, so
a non-finite path point was stepped over and left the bbox looking tight; and
`clamp(-1e9, 1e9)` on a coordinate past the addressable grid clamped *inward*.

Fixed by `TileRect::covering(lo, hi, ring)`, which quantizes in `i64` and answers
`ALL` for any box that is not finite or not addressable, plus an explicit
non-finite test per path point. All three of this file's quantizer copies now go
through it; semantics are unchanged for every finite, in-range input.

### 1.3 The tree surgery conflated two meanings of `None` — **done**

`remove_in` propagated a `Vector::set` out-of-range failure as "layer not
found", which callers turn into a silent no-op; `map_in` did the same twice, by
returning `set`'s `Option` directly. The section's own header already states the
contract these broke — `None` means "no such layer", which is what lets callers
turn it into a clean no-op. Spelling an impossible failure as the one case they
act on would have made a removal or a rename that silently did not happen, on a
document reporting that it did.

Unreachable today (every index comes from `position`/`enumerate` over the stack
being written), so this is a contract repair rather than a behaviour change, and
there is nothing new to test. Now `expect`ed against one shared reason string.

Worth noting while in here: `state.rs` has no unit tests at all, and the four
recursive tree-surgery functions are the part of this module most worth having
them. Covered only behaviourally, through the engine, by `tests/groups.rs` and
`tests/layers.rs`.

---

## 2. Performance

### 2.1 `observe()` cost two full log scans per pointer sample — **done**

`stark-ui/src/state.rs::dispatch` calls `observe()` on every command, and
`GestureCommand::To` goes through `dispatch` — i.e. at digitizer rate.
`observe()` reads `can_undo`/`can_redo`, and on `ReplicatedTimeline` each called
`undone_ids(&self.log)`: an O(n) pass building a fresh `HashSet`, then an O(n)
backwards scan over the log. Two of those per pen sample, growing with session
length — and the log does not change once during a stroke, so every one of them
recomputed the same two answers.

`ReplicatedTimeline` now caches the pair in a `targets` field, making the four
query methods O(1) field reads. `undo_target`/`redo_target` became pure functions
of `(log, actor, undone)`, which is what makes caching them sound, and `resync` —
the single point every log write funnels through — refreshes the pair from the
same `undone` set it materializes against, before any of its own early returns.

Two things fell out. `redo_target` had inlined `undo_target`'s body verbatim as
`latest_ordinary` (§4's bullet); it now calls it, so the predicate has one home.
And `timeline.rs` gained its first unit tests — five over the target resolution,
which had been covered only end-to-end through the engine.

**Not** done here: `resync` still recomputes `effective_indices` (O(n log n)) per
insert, so a session is O(n² log n) in total. That is per *commit* rather than
per pen sample, so it is a different order of problem — but it is the next one in
this file.

### 2.2 `has_backdrop` was O(layers²) inside `observe()` — **done**

`engine.rs` called `shown.has_backdrop(l.id)` *inside* `shown.visit(…)` — a full
recursive tree walk per layer, on the same per-pointer-sample path as §2.1.

The predicate turned out to be answerable without any search. `has_backdrop` is
`index_in_own_stack > 0 || depth > 0`, and composite order visits the bottom of
the root stack **first**, so the only layer it is false for is the first one
visited: every other has either a lower sibling or the content of the layer
carrying it. The projection's own field doc and the test guarding it
(`only_the_bottom_of_the_document_has_no_backdrop`) already said exactly that —
the search was computing a known answer the long way.

So `observe` now reads `!layers.is_empty()` off the traversal, and
`DocState::has_backdrop` is gone; it had one caller, and the "what about an id
that is not in the tree" question it had to answer disappears with it. The design
prose moved to `DocState::visit`, which is where the fact is now derived.

### 2.3 `with_layers` recomputed canvas bounds on every layer mutation — **done**

It walked every layer's every tile key, and `map_layer` routes *all* the property
setters through it — so setting a layer's name re-derived the extent of the whole
document. O(total tiles) per action, and a scrub or replay across N actions was
O(N·T).

Now the extent is memoized **per layer**, in a `PaintTiles` value that pairs a
tile map with the box it spans, and `DocState::bounds` is the union of those
(`CanvasBounds::union`, a join). A mutation that leaves a layer's tiles alone —
every property setter, every structural move, a stroke on some *other* layer —
contributes the box that layer already knows. Only a layer whose map actually
changed re-derives, and it pays for itself rather than for the document. The tree
walk stays, because a layer can be anywhere in it, but it is now O(layers)
(dozens) rather than O(tiles) (thousands).

Correct **by construction** rather than by classification: both of `PaintTiles`'
fields are private and its only constructor derives the extent, so no caller
decides whether its own change grew or shrank the document, and no writer of a
tile map has to remember to refresh anything. That mattered more than the speed —
`bounds` is what "frame to content" and export's no-frame fallback measure
(§15.6), so a stale one is a wrongly-cropped export.

Not pursued: making the *within-layer* rederive incremental too, so a stroke
unions only the tiles it wrote instead of re-spanning its layer. It needs the
writer to report its written extent — either the renderers returning it (which
leaves `document/`) or trusting the action's own footprint rect, which would hang
bounds correctness on §3.1's still-unverified invariant. Worth revisiting after
§3.1, not before.

### 2.4 `LinearTimeline::applied()` is O(n) per `scrub_range` — open

Called from `scrub_range`, which the timeline panel reads per render.

---

## 3. Architecture

### 3.1 One action variant, four exhaustive matches, three files — **test done**

Adding an `ActionKind` means editing `apply` (`action.rs:945`), `footprint`
(`footprint.rs:190`), `StatePatch::capture` (`patch.rs:87`) and `label`.
Exhaustiveness gets you the *presence* of an arm; nothing checks
*correspondence*, and CLAUDE.md's first "easy to break silently" rule is
precisely that an `apply` must touch only what its footprint declares.
`tests/commute.rs` covers five hand-written scenarios; there is no generic check.

**The test is done** (`tests/footprint.rs`); the refactor is still open, and is
now a much smaller question than it was.

`every_action_touches_only_what_it_declares` drives a shared session through
every action kind the engine can commit — all 21 of them, `Undo` excepted —
diffing the document across each commit and insisting every difference lands
inside a resource the action declared. Differences are named in the footprint's
own vocabulary (`Resource`), so coverage is containment rather than
interpretation: a tile by a rect that contains it, a property by its own `Prop`,
existence by `Existence` or by the `StackOrder` that `footprint.rs` deliberately
coarsens a subtree removal into.

Two things keep it from passing by being blind. A second test feeds a
deliberately dishonest footprint — every claimed rect shrunk to one tile at the
origin, which is what the pre-`TileRect::covering` quantizer produced for a
non-finite radius — and asserts the check rejects it. And the run collects the
action kinds it actually reached and asserts the expected set at the end, because
a step that stops committing would otherwise *silence* its coverage instead of
failing.

Out of scope, and stated in the test: `reads` are not checked (a read leaves no
trace in a state diff and needs a different instrument), and `Undo` is excluded
by construction, since it is resolved by the timeline rather than applied — which
is exactly why its footprint is empty.

With the invariant now under test, the remaining refactor question — colocating
apply/footprint/patch per action so the compiler forces all three — is worth
weighing on its own merits rather than as a safety measure.

### 3.2 `action.rs` was two files — **done**

The brush model — `BrushShape`, `OrientationSource`, `BrushDynamics`,
`NoiseKind`, `ColorDynamics`, `ModSource`/`PenState`/`Modulation`/`Modulations`,
`BrushParams` — was 550 lines with nothing to do with the action log, documented
against §6.2/§6.6 rather than §4, and it carried `action.rs`'s entire test module
with it (all seven tests were about `Modulation`).

Now `document/brush.rs`: 739 lines about what a stroke is *made with*, leaving
`action.rs` at 573 about what an action *is*. `Tool` and `StrokeRecord` stayed —
a tool is a gesture's, not a brush's (the selection tools are in that enum), and
a `StrokeRecord` is the action payload that happens to hold a `BrushParams`.

Pure move: no item changed, and the public paths are unchanged because `mod.rs`
re-exported these already. Only `footprint.rs` reached for `BrushParams` by
module path.

One pre-existing broken intra-doc link surfaced and was fixed on the way past
(`SelectionShape::All` in `ActionKind::Fill`'s docs — `SelectionShape` was never
in scope in `action.rs`). `document/` is now clean under
`-D rustdoc::broken_intra_doc_links`. **Two more remain outside it**, both
pre-existing and both outside this review's scope: `Session::publish_due` in
`engine.rs:2112` and `Self::adopt_default_name` in `session.rs:299`.

### 3.3 `DocState`'s derived field was public — **done**

`bounds` is a pure function of `layers`, and both were `pub` — so the invariant
that the two agree was a convention `with_layers` happened to uphold rather than
something the type enforced. It matters because `bounds` is what "frame to
content" and export's no-frame fallback measure (§15.6): a value out of step with
the paint is a wrongly-cropped export, and nothing about the pixels would say so.

Both are private now, behind `root()` and `bounds()`. The three external readers
(one of `layers`, two of `bounds`, all in `engine.rs`) plus one test moved over.

The part that makes it structural rather than cosmetic: `with_layer` now builds
the empty document and *inserts* its first layer, instead of naming the
layers/bounds pair directly. So `with_layers` — which derives the extent — is the
only place `bounds` is ever written, and the one remaining literal describes a
`DocState` with no layers at all, whose empty extent needs no deriving.

### 3.4 Two tile-rectangle types and five copies of the quantizer — **done** (one half deliberately not)

`CanvasBounds` (`state.rs:21`) and `TileRect` (`footprint.rs:27`) are the same
inclusive tile box with different empty encodings. And
`|v: f32| ((v / TILE_SIZE as f32).floor().clamp(-1e9, 1e9)) as i32` appears
verbatim three times in `footprint.rs` and twice more as an `as i64` variant in
`transform.rs`/`selection.rs` — the two versions differing in overflow behaviour,
which is where §1.1 lived.

**The quantizer is now one.** `TileRect` moved to `geom.rs` — it is tile
geometry, not a footprint concept — and carries `covering(lo, hi, ring)`,
`count()` and `coords()`. All five sites go through it: `footprint.rs`'s three
(via a `claim` helper that supplies the "an unmeasurable box claims everything"
answer), `selection.rs`'s `tile_box`/`tiles_covering`, and `transform.rs`'s two.

That last pair still had the original defect: both quantized in `i64` and then
built coordinates with a bare `x as i32`, which **truncates**, so a tile index
past the addressable grid wrapped to a tile somewhere else. Reachable only from
an absurd transform rect, but silent, and the same class as §1.1.

Callers differ in what they *do* about an unmeasurable box — a footprint claims
everything, a cover refuses — so `covering` returns `Option` and each caller
answers for itself at one visible line. The counting is shared too: `count()`
saturates, so `TileRect::ALL` reports more than any budget allows rather than
wrapping to something small and being enumerated.

**`CanvasBounds` is deliberately *not* folded onto `TileRect`.** They have the
same shape and genuinely different meanings: `CanvasBounds` is a *measured*
extent — what is painted — always finite, empty when nothing is; `TileRect` is a
*claimed* region and its most important value is `ALL`, the whole plane. Merging
them would give the document's bounds a representable `ALL`, and bounds is what
"frame to content" and export's no-frame fallback measure (§15.6) — so the merged
type could express a wrong thing that the current pair cannot. §2.3 just spent a
type making that unrepresentable; collapsing it back for a dedup that saves about
fifteen lines is the wrong trade. The duplication that mattered — the arithmetic
— is gone.

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
- ~~`ReplicatedTimeline::redo_target` inlines `undo_target`'s body verbatim as
  `latest_ordinary`; call it.~~ **done** with §2.1.
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
