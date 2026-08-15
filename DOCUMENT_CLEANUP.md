# `document/` — the architectural cleanup ledger

Reviewed 2026-08-15 on `master` (`7001c6a`), across all 14 files of
`crates/stark-core/src/document` (~9.4k lines, of which ~2.0k are the in-file test
modules), plus the three places it is joined to at the seam: `engine/mod.rs`'s
`observe` projection, `gpu::fill::FillRenderer::apply` (the one consumer whose written
tile set has to match a footprint), and `tests/footprint.rs`, which is the only
instrument pointed at the rule this module lives or dies by.

**Eight of eleven findings have landed**, in five commits on `document-cleanup`. One
landed in part and two are open; each says below what it would still cost. Symbol names
are the durable part.

This is `STROKE_CLEANUP.md`'s review one directory over, and it found the same shape
from the other side. There the models were separated and tested while the bindings and
the boxes were not; here the *state* is structural almost everywhere it can be —
`PaintTiles` cannot be built inconsistently, `cannot_carry` rules out a filter carrier
at both entry points, `Place` is pinned byte-for-byte against the `Option` it must
stay wire-compatible with, `LayerId::mint` partitions the id space so two peers
cannot collide. What is *not* structural is everything stated in more than one file,
and that is exactly where both defects were.

## The one-sentence summary

**What one file decided, it decided structurally; what two files decided, they decided
by convention — and every live defect sat on a seam between two files.** A fill's
written tile set was computed twice from the same op, once with the apron and once
without (§1). A merge's write set was declared in `footprint.rs` and made honest by
guards in `merge.rs` (§2). An action's meaning was spelled three times — `apply`,
`footprint`, `capture` — with nothing but Rust's exhaustiveness linking the arms (§4).
The one test that checks correspondence rather than presence missed the three newest
kinds, and its own anti-rot guard was a hand-written list that did not fail when a
variant was added (§3). `action.rs` and `patch.rs`, the two files holding the seam, had
**no in-file tests at all** (§11).

What the seams have now is one source each: the fill's box is one function, the patch is
the footprint's own write list, and the merge's site is either searched for or read off
a walk but decided by one predicate.

| § | Finding | Kind | Status |
|---|---|---|---|
| 1 | A fill's footprint under-claims the tiles its plan writes | **correctness**, §12.6 under-claim | **landed** `3848f71` |
| 2 | `MergeLayerDown` writes three composite params and declares one | latent, guard-held | **landed** `3848f71` |
| 3 | The footprint-honesty run misses three kinds, and its guard rots | coverage | **landed** `3848f71` |
| 4 | Three parallel matches over `ActionKind`, linked only by prose | drift class | **landed** `aabad89` |
| 5 | `apply`'s merge arm is 90 lines of orchestration inside a match | misplaced subject | **landed** `aabad89` |
| 6 | `observe` re-searches the tree per layer for `merge_down` | O(L²) per command | **landed** `8371f1f`, −91% |
| 7 | The commutation scan allocates per comparison and derives itself twice | 2× on the fast path | open |
| 8 | "Everything about this layer" is spelled nine resources at a time | O(n·m) conflict scan | **landed** `7eca152` |
| 9 | A fill allocates tiles where its coverage is provably zero | bounds pollution | open |
| 10 | Smaller correctness and clarity items (four) | mixed | **part landed** `3848f71` |
| 11 | `action.rs` and `patch.rs` carry no in-file tests | coverage | **landed** `9b6048c` |

---

## 1. A fill's footprint under-claimed the tiles its plan writes — landed

**The first of CLAUDE.md's "rules that are easy to break silently", and it was broken.**

Two different tile boxes were derived from one `FillOp`, and they disagreed by an apron.
`fill::plan` padded by `op.reach()` and then called `selection::tile_box`, which pads by
`TILE_APRON` *again* before quantizing — it answers "tiles whose **texture** overlaps
this box", which is what it exists for. `footprint::fill_rect` padded by the same
`op.reach()` (through `fill_bounds`) and then called `claim`, which is
`TileRect::covering` with **no** apron — "tiles whose **interior** overlaps this box".

So the plan's box was `[lo−1, hi+1]` and the footprint's `[lo, hi]`, from identical
inputs. Wherever the padded bound fell within one pixel below a tile boundary — always,
for a fill aligned to the tile grid — `plan` named a tile the footprint did not. And
`FillRenderer::apply` walks `plan`'s coords and ends every iteration with `tiles =
tiles.insert(*coord, dst.into_tile())`: a **fresh handle per planned coord,
unconditionally**, no zero-coverage skip. The written set *was* the plan's set.

Two consequences, and the second needed no second peer: the commutation gate could
splice an undone fill out past an action that overlapped that tile, and — because
`patch::paint_rect` bounded `tile_diff` by the footprint's rect *on purpose* — undoing
that fill left the tile behind, an all-zero tile still counting toward `PaintTiles::bounds`
and so toward "frame to content" and export's no-frame fallback (§15.6).

**What landed.** Not a matching pad in the second place but the removal of the second
place. `fill_bounds` carries the apron and is the one box; `plan` and `fill_rect`
quantize *it*, separately, because they answer differently for a box that cannot be
quantized at all — the footprint claims everything, the plan refuses, which is the
question `TileRect::covering` returns rather than picking. `selection::tiles_of` is
`tiles_covering`'s second half, split out so a caller holding its own rect still counts
before it walks.

**No pixel moved.** The plan's box was already `covering(lo − reach − apron, …)` and
still is; only the footprint widened to meet it.

`the_footprint_names_every_tile_the_plan_writes` pins it, swept across a whole tile
stride at a quarter pixel — one alignment is exactly what hid this, since
`tests/footprint.rs` fills at (40, 40)–(80, 80). Checked against a reproduction of the
original drift, where it fails with `writes TileCoord { x: -1, y: -1 }, which its
footprint TileRect { min: (0, 0), max: (0, 0) } does not declare`.

This was the failure `TileRect::covering` was introduced to end — *"this arithmetic was
written out five times across `document/`, and the copies disagreed in exactly the place
that matters"* — surviving one wrapper out.

## 2. `MergeLayerDown` wrote three composite params and declared one — landed

`apply` assigns `composite: keeps`, a whole `CompositeParams`; the footprint declared
only `Prop(dest, Opacity)`, with blend and clip as *reads*.

It was sound, and only by guards in another file: `merge::plan`'s sibling arm reaches
`keeps.clip = false` only after refusing a clipped destination, and takes `keeps.blend`
from the destination's own. Nothing in `footprint.rs` or `action.rs` said so, and no test
drove the pair (§3).

**What landed.** All three declared. Two are the identity today, so this costs a false
conflict and nothing else — the direction §12.6 says to err in. The alternative was
resting the footprint's honesty on refusals made in `merge.rs`, and on the two merges
that module's header says it means to build next, both of which touch `keeps`.

## 3. The footprint-honesty run missed three kinds, and its guard rotted — landed

`tests/footprint.rs`'s `expected` list named 21 of 25 kinds. `MergeLayerDown`,
`AddFilter` and `SetFilter` were never driven — which is why §2 was invisible. The
list's own header argues that it exists so the test cannot "rot into vacuity one command
at a time", which is right and does not cover the case that happened: a hand-written
list cannot fail for a kind that does not exist yet, so an action added to the enum
*silenced* its own coverage rather than failing.

**What landed.** The list is a `slot` function: an exhaustive match with no `_` arm, so
a new variant stops the file compiling at the match, three lines from the run that has
to drive it. It forces a *visit* rather than proving a correspondence, which is as far
as Rust goes without a derive — the same bargain `Modulations::all`'s `..`-free
destructure makes for a struct's fields. `Undo`'s exemption is *asked for*
(`slot(&ActionKind::Undo(..))`) rather than written down, so it survives a reorder.

The run gained the filter pair — added neutral, then dialled, since a neutral filter is
dropped from the draw list and would hold its footprint to a diff of nothing — and a
merge of a freshly painted layer into `root`.

## 4. Three parallel matches over `ActionKind` — landed

`StatePatch::capture` no longer matches on `ActionKind`. It walks
`footprint(action).writes` and maps each resource to the op that puts it back:
`capture_resource` records a `Resource`, `PatchOp::restore` writes one back, and the
correspondence is the table between them.

`paint_rect` is deleted by the change, which is the tidiest evidence it was the right
one: the helper whose whole job was to ask the footprint for a rect has nothing to do
once the footprint is the driver. `Undo` needs no arm either — it is never materialized,
which is *why* its footprint is empty.

Two things it had to resolve, both real and both handled:

- **Existence is captured first**, by partition, whatever order a footprint lists its
  writes in. A layer has to be back in the tree before its tiles, its properties or the
  tree's shape can be put right — `restore_structure` arranges records it does not
  create. Every current footprint happens to list `Existence` before `StackOrder`;
  relying on that would have made it a rule `footprint.rs` has to remember.
- **`StackOrder` is now captured by add, remove, duplicate and merge**, which previously
  restored placement through `Present`/`Absent` alone. Under the commutation gate the
  suffix holds no structural edit, so the rebuild is an identity — and where it is not,
  it is *more* correct: `PatchOp::Present` falls back to the top of the root stack when
  a site's carrier has gone, and the shape then puts the layer where it belongs.

The exhaustiveness this buys paid off within the hour: §8's new `Resource` variant would
not compile until `capture_resource` said what restoring it means.

## 5. `apply`'s merge arm was 90 lines inside a match — landed

`merge_apply` sits beside `transform_apply`, and the arm is a call. The `unreachable!`
that re-destructured a `MergeKind` the arm had already matched is gone with it, and the
two kinds' difference is said once as an `Option<CompositeParams>` — `None` being "the
destination keeps its own", which is the whole content of a filter merge.

**Not into `merge.rs`, which is where this ledger first said to put it.** That module's
header opens "Pure CPU, no GPU anywhere in this file", and the split the codebase
actually keeps is plan (`document/`) against render (`gpu/`), with the orchestration of
the two in `action.rs` — for fill, for all three transforms, and now for the merge. The
original recommendation would have put a `TilePool` in a file that promises not to hold
one.

Also landed here: the `Fill` arm's comment block had drifted above `MergeLayerDown` and
was documenting the wrong action.

## 6. `observe` re-searched the tree per layer — landed, **−91%**

`Engine::observe` called `merge::plan` per layer, and `plan` opens with `site_of` — a
walk of the whole tree — then reaches for the carrier and the destination the same way.
The projection was quadratic in the layer count.

**Measured**, before and after, on a tree with real nesting and reordering:

| layers | before | after | |
|---|---|---|---|
| 4 | 1.30 µs | 0.62 µs | −52% |
| 20 | 11.34 µs | 2.70 µs | −76% |
| 60 | 79.09 µs | 7.10 µs | **−91%** |

The shape matters more than the constant: before, 3× the layers cost 7× the time; after,
2.6×. The curve is linear. 79 µs on a command path is not a wound today — this is a
projection, not the stroke path, and `dispatch_sample` already skips it mid-gesture —
but the growth was wrong.

**What landed.** `merge::plan` splits in two. `MergeSite` is what a compositing walk
already knows about a candidate — the layer, the layer beneath it, and the two flags
§14.11.2 turns on — and `plan_at` is everything after that, which is a question about the
two layers and never about the tree. `plan` is the search plus `plan_at`, so the action,
the command gate and the panel decide by one function and differ only in how they find
the site.

`observe`'s three parallel per-depth vectors became one `Cursor`, which is also where the
lower sibling and the stack index live. `Layer::visit` and `DocState::visit` thread the
tree's lifetime rather than binding each call's — strictly more permissive, every
existing caller unchanged — which is what lets a walk keep what it has seen.

`an_index_keeps_counting_across_a_carried_stack` pins the one thing the trade risks: a
stack's index has to keep counting after the walk descends into what a lower sibling
carries and comes back. `[ROOT[X], B, C]` with `C` clipped, where an index that reset
over `X` would offer a merge §14.11.2 forbids. Checked against a loosened predicate,
where it fails with `left: Some(LayerId(1)), right: None`.

## 7. The commutation scan allocates per comparison and derives itself twice — open

`Footprint` is two `Vec<Resource>`, and `Centralizer::commutes` calls `footprint(other)`
fresh for every candidate. `ReplicatedTimeline::resync` then walks the suffix building
footprints **purely to populate `TimelineStats`**, immediately before handing the same
work to `remove_action_with`.

Both halves are still open, and the reasons differ:

- **The stats derivation needs an upstream change.** The clean fix is for
  `remove_action_with` to report how far it got, and `history` is an external git
  dependency (`cbbowen/history`) rather than a workspace crate — so it is a PR there,
  not a change here. Gating the derivation on the counters actually being read is the
  local alternative, and it is ugly for what it buys: the derivation early-exits at the
  first conflict, so it is 2× only on the path that was already cheapest.
- **The allocation wants a measurement first**, and §8 moved the goalposts: the typical
  footprint is now one or two entries, and `DuplicateLayer`'s went from 9*n* to *n*. A
  `SmallVec` would remove two small allocations per comparison, but adding a dependency
  on an unmeasured motive is the thing §6 was held to and this should be held to it
  too. Time an undo across a long commuting suffix before deciding.

Caching the footprint beside each entry in `ReplicatedTimeline::log` remains the
strongest version — it is a pure function of an immutable action — and it is also the
largest change, since the log is written in one place but read in several.

## 8. "Everything about this layer" was nine resources — landed

`DuplicateLayer` emitted nine `Resource`s per copied layer and `MergeLayerDown` five per
side, which made `Footprint::conflicts` — a nested scan — quadratic in a number with no
business being large: a twenty-layer duplicate claimed 180 read resources.

**What landed.** `Resource::Layer(id)`: `StackOrder`'s move for a layer instead of the
tree, earning its place the same way. A coarse claim is *more* conservative than the fine
ones it replaces, never less, which is the only direction §12.6 permits. `overlaps` gains
one symmetric arm and a `layer()` projection; a `Paint` rect is not consulted, since the
coarse claim takes all of it.

Three tests, because a coarse claim is exactly the kind that can be quietly too fine:
`a_duplicate_conflicts_with_every_edit_inside_what_it_copies` pins the claim rather than
the spelling, `a_whole_layer_claim_meets_every_finer_claim_on_it` pins the symmetry and
the boundary, and `every_prop_is_named_in_all` guards `Prop::ALL`, which is what the
coarse claim expands to when it is a *write*.

`capture_resource` gained a `Layer` arm because §4's footprint-driven capture refused to
compile without one. No action writes a whole layer today, so it is unreached; it is
written out rather than left to an `unreachable!` because a patch restoring nothing for a
declared write is §12.6 read backwards, and an `unreachable!` is how that would arrive —
as a panic in the undo path, on the day the coarse resource finds a second use.

## 9. A fill allocates tiles where its coverage is provably zero — open, and rescoped

`op.reach()` is `feather/2 + TILE_APRON + 1`, documented as exactly how far coverage can
travel past the shape boundary and deliberately tighter than `Selection::plan`'s padding
so a fill is not ringed with all-zero tiles. `plan` then pads by a second apron and
`FillRenderer::apply` mints a fresh tile for every coord in the box, so the band the
comment argues against is built anyway, one apron thinner.

**Rescoped by §1.** The original write-up treated this as the same defect; it is not.
§1 was a disagreement between two derivations and its fix moved no pixel. This is a
question about how much slack the one derivation should carry, and changing it **moves
which tiles a fill writes** — a golden-touching change, and one whose safety argument
runs through `selection.wesl`'s ramp and the §6.4 seam invariant rather than through
anything in `document/`.

So it wants the seam test pointed at it first, not a cleanup pass. `transform` already
has the shape of the answer in `TransformPlan::drops` ("a rewrite would write all zeros,
and an all-zero tile pollutes `bounds` and holds pool memory that 'no tile' would not").
The conservative behaviour is correct today; it is only wasteful.

## 10. Smaller correctness and clarity items — part landed

**Landed.** `ActionKind::SetFilter`'s doc linked `SetMatteColor`, a variant renamed to
`SetMattePaint`. (`io.rs` and `engine/mod.rs` also name it, but in prose about the wire
history, which is correct and stays.) Worth adding a `cargo doc` build with `-D warnings`
to CI given how much of this module's design lives in its doc comments — not done here,
since it is a CI change rather than a code one and would want its own pass over whatever
else it turns up.

**Left, deliberately.** Two file splits — `state.rs`'s persistent-tree library (`map_in`,
`remove_in`, `splice`, `insert_at`, `copy_subtree` and the `IN_RANGE` contract), and
`transform.rs`'s projective geometry (`Homography`, `PerspectiveMap`, the f64
square-to-quad derivation, `mat3_adjugate`) away from its tile planning. Both are the
mechanical move `region.rs` and `dynamics/bleed.rs` already were, and neither is urgent:
unlike `segments.rs` at 2.6k lines, these are 1.1k and their subjects do not have an
invariant spanning them that file-locality would protect.

**Left, and probably should stay.** `rpds::Vector` for layer stacks buys little at
"dozens" — `insert_at` and `remove_in` rebuild the spine element by element anyway — but
it is correct as it stands, and `Arc<Vec<Layer>>` would touch every tree-surgery function
for a constant factor nobody has measured. It is on the wrong side of the same bar §7's
allocation half is.

## 11. `action.rs` and `patch.rs` carried no in-file tests — landed

Zero of the module's ~2.0k test lines were in the two files holding the seam. `unapply`
is the inverse of every action in the enum, its soundness argument is the subtlest in the
module, and nothing exercised it below the integration level — where `STARK_ALLOW_NO_GPU`
can skip it entirely.

**What landed.** Seven CPU-only tests, no adapter needed, because the property and
structural arms of `apply` are pure `DocState` calls and the states can be built by hand.
`every_property_round_trips` is driven off `Prop::ALL` so the coverage and the enum are
one list; `a_restore_leaves_a_commuting_edit_alone` is the non-adjacency property itself;
`a_move_restores_the_shape_and_keeps_current_records` pins why a move restores a shape
rather than a snapshot; `a_shape_naming_a_layer_twice_is_survived` reaches
`restore_structure`'s `placed` guard, which nothing had.

Worth recording: each property case asserts the action *changed* something before
asserting it came back, and that caught a hole in the test's own comparison —
`Prop::Matte` read as a no-op because the fingerprint compared composite, visibility,
name and filter, none of which include a matte's paint.

`action.rs` still has no in-file tests. Its `apply` needs a GPU context, so its coverage
is `tests/footprint.rs` and `tests/commute.rs` by construction; §3 is what makes that
coverage honest, and there is no cheap unit-level addition worth making.

---

## What the review did not find

No defect in the state model itself. Several invariants that would have been easy to get
wrong are already structural, and the review confirmed rather than disturbed them:

- **`PaintTiles` cannot be built inconsistently.** Both fields private, one constructor,
  the extent derived — so `DocState::bounds` being a union of per-layer boxes is cheap
  *and* cannot go stale, and a rename does not cost what a stroke costs.
- **`DocState::with_layers` is the only writer of `layers` and `bounds`**, and both are
  private for that reason. The "two forms of one fact" hazard is closed by construction.
- **`cannot_carry` rules out a filter carrier at every entry**, `insert` and `move_layer`
  alike, and `restore_layer` handles the crafted-log case by falling back rather than
  losing the layer. Tested at both entries.
- **`undone_ids`' single descending pass is sound**, and — more importantly — remains
  *deterministic* for a malformed log that violates the "an `Undo` outranks its target"
  premise. Determinism is what convergence needs; correctness against a hostile file is a
  different and lesser requirement.
- **`insert`'s fast path is checked against the derivation it skips.**
  `appending_an_ordinary_action_only_extends_the_effective_sequence` and
  `appending_moves_only_its_own_actors_targets` are the right pair: the shortcut and the
  scan must agree, and they are made to say so.
- **`Selection::plan`'s level algebra deliberately departs from its coverage algebra**
  for `Subtract`, and the comment proves it: subtracting flattens the level only on the
  texels it covered, so `combine` would have been wrong there.
- **`Place` and `BlendMode` are pinned byte-for-byte** against the encodings §8 promises,
  rather than reasoned about — the only defence against a reordered variant silently
  re-reading every saved document.
- **`LayerId::mint` partitions by author**, so two peers adding a layer at the same moment
  cannot collide, and `ActorId::SOLO` maps to high half 0 so a never-shared document keeps
  `LayerId(0)`.
- **`TileRect::covering` genuinely closes the NaN/overflow class** — `i64` throughout,
  saturating, `None` rather than a wrap — for every caller. §1 was the one that reached it
  through the wrong wrapper, and does not any more.
- **The warp's deviation-form Hermite gives a bit-exact identity**, asserted bitwise
  rather than approximately, which is what §16.4's identity invariant actually needs.
- **`Selection::from_parts` forces `hull = None` whenever `outside > 0`**, so a caller
  cannot hand it a box that contradicts coverage reaching infinity.

`Modulation`'s rational response curve and its `max_slope` bound were checked against
their tests and are exactly what the flattener needs; `Modulations::all`'s `..`-free
destructure is the pattern §3 and §8 both ended up copying.

---

## Verification

Every commit was taken green: `cargo fmt --all --check`, `cargo clippy --workspace
--all-targets -D warnings`, the second configuration (`--no-default-features --features
stark-net/webrtc`), `cargo check -p stark-ui --target wasm32-unknown-unknown`, and
`cargo test --workspace` redirected once to a file per batch.

**843 tests pass with the goldens rendered and compared** (`STARK_SKIP_GOLDEN` unset,
`STARK_ALLOW_NO_GPU` unset), from a baseline of 831 — twelve added: one for §1, one for
§6, three for §8, seven for §11. Green both in parallel and under
`-- --test-threads=1`.

One run segfaulted in `corpus` with `STATUS_ACCESS_VIOLATION` partway through this work,
and it did **not** recur: `corpus` then passed in parallel three times on this branch and
three times on `master`, and the full workspace passes in parallel. So it was the known
driver-contention flake (`gpu-test-access-violation-flake`) rather than a property of
either tree — but "it reproduces on master" would have been the wrong way to establish
that, and running it three times each way is the right one. The intermediate batches were
taken green serially, which is what made the one-off legible as a one-off at the time.

Two changes claim something a test cannot show, and both were checked by breaking them:

- **§1's regression test was run against a reproduction of the original drift** — `plan`
  one apron wider than the footprint — where it fails with `writes TileCoord { x: -1, y:
  -1 }, which its footprint TileRect { min: (0, 0), max: (0, 0) } does not declare`. A
  test for a §12.6 under-claim that has never seen one is not evidence.
- **§6's index test was run against a loosened predicate** (`index <= 2` for `index ==
  1`), where it fails with `left: Some(LayerId(1)), right: None`.

**§6 is the only change here claiming a performance number**, and it was measured as an
A/B on `observe` itself at 4, 20 and 60 layers — not on `--bench stroke`, which cannot
see a CPU cost on the command path. −52%, −76% and −91%, with the curve going from
quadratic to linear. Nothing else here claims one: §8 is a constant-factor reduction
argued from the resource counts rather than timed, and §7 and §9 are open *because* they
want measurement first.

§1 is the only change that could have moved a pixel and does not: the plan's box is
unchanged and only the footprint widened to meet it, which the golden run confirms.
