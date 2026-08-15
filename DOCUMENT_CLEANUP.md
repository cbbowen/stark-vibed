# `document/` — the architectural cleanup ledger

Reviewed 2026-08-15 on `master` (`7001c6a`), across all 14 files of
`crates/stark-core/src/document` (~9.4k lines, of which ~2.0k are the in-file test
modules), plus the three places it is joined to at the seam: `engine/mod.rs`'s
`observe` projection, `gpu::fill::FillRenderer::apply` (the one consumer whose written
tile set has to match a footprint), and `tests/footprint.rs`, which is the only
instrument pointed at the rule this module lives or dies by.

**Eleven findings, all open.** Two are live correctness defects, one of them
user-visible without collaboration. Symbol names are the durable part.

This is `STROKE_CLEANUP.md`'s review one directory over, and it found the same shape
from the other side. There the models were separated and tested while the bindings and
the boxes were not; here the *state* is structural almost everywhere it can be —
`PaintTiles` cannot be built inconsistently, `cannot_carry` rules out a filter carrier
at both entry points, `Place` is pinned byte-for-byte against the `Option` it must
stay wire-compatible with, `LayerId::mint` partitions the id space so two peers
cannot collide. What is *not* structural is everything stated in more than one file,
and that is exactly where both defects are.

## The one-sentence summary

**What one file decides, it decides structurally; what two files decide, they decide by
convention — and every live defect here sits on a seam between two files.** A fill's
written tile set is computed twice from the same op, once with the apron and once
without (§1). A merge's write set is declared in `footprint.rs` and made honest by
guards in `merge.rs` (§2). An action's meaning is spelled three times — `apply`,
`footprint`, `capture` — with nothing but Rust's exhaustiveness linking the arms (§4).
The one test that checks correspondence rather than presence misses the three newest
kinds, and its own anti-rot guard is a hand-written list that does not fail when a
variant is added (§3). `action.rs` and `patch.rs`, the two files holding the seam, have
**no in-file tests at all** (§11).

| § | Finding | Kind | Status |
|---|---|---|---|
| 1 | A fill's footprint under-claims the tiles its plan writes | **correctness**, §12.6 under-claim | open |
| 2 | `MergeLayerDown` writes three composite params and declares one | latent, guard-held | open |
| 3 | The footprint-honesty run misses three kinds, and its guard rots | coverage | open |
| 4 | Three parallel matches over `ActionKind`, linked only by prose | drift class | open |
| 5 | `apply`'s merge arm is 90 lines of orchestration inside a match | misplaced subject | open |
| 6 | `observe` re-searches the tree per layer for `merge_down` | O(L²) per command | open |
| 7 | The commutation scan allocates per comparison and derives itself twice | 2× on the fast path | open |
| 8 | "Everything about this layer" is spelled nine resources at a time | O(n·m) conflict scan | open |
| 9 | A fill allocates tiles where its coverage is provably zero | bounds pollution | open |
| 10 | Smaller correctness and clarity items (four) | mixed | open |
| 11 | `action.rs` and `patch.rs` carry no in-file tests | coverage | open |

---

## 1. A fill's footprint under-claims the tiles its plan writes — open

**The first of CLAUDE.md's "rules that are easy to break silently", broken today.**

Two different tile boxes are derived from one `FillOp`, and they disagree by an apron:

- `fill::plan` pads by `op.reach()` and then calls `selection::tile_box`, which pads by
  `TILE_APRON` *again* before quantizing — it answers "tiles whose **texture** overlaps
  this box", which is what it exists for.
- `footprint::fill_rect` pads by the same `op.reach()` (through `fill_bounds`) and then
  calls `claim`, which is `TileRect::covering` with **no** apron — "tiles whose
  **interior** overlaps this box".

So the plan's box is `[lo−1, hi+1]` and the footprint's is `[lo, hi]`, from identical
inputs. Wherever the padded bound lands within one pixel below a tile boundary — always,
for a fill aligned to the tile grid — `plan` names a tile the footprint does not.

That gap is written, not merely planned. `FillRenderer::apply` walks `plan`'s coords and
ends every iteration with `tiles = tiles.insert(*coord, dst.into_tile())`: a **fresh
handle per planned coord, unconditionally**, with no zero-coverage skip. The written set
*is* the plan's set.

Two consequences follow by construction, and the second needs no second peer:

- The commutation gate may splice an undone fill out past an action that genuinely
  overlapped that tile. Silent divergence, and pixels cannot show which materialization
  ran — which is the whole point of §12.6.
- `patch::paint_rect` bounds `tile_diff` by the footprint's rect *on purpose* ("asked of
  the footprint itself rather than re-derived, so the two cannot drift"). It therefore
  inherits the under-claim: **undoing that fill leaves the boundary tile behind**, an
  all-zero tile that still counts toward `PaintTiles::bounds`, and so toward "frame to
  content" and export's no-frame fallback (§15.6).

`tests/footprint.rs` does drive two fills, but at `(40,40)–(80,80)` with feather 1.0,
which does not cross the alignment case.

**What to do.** Not a patch to `fill_rect` — one function. A fill's written tile set
should have exactly one definition, and `plan` and `fill_rect` should both read it.
`stroke_rect` already carries this rule in its own doc comment ("Deliberately the *same*
answer the commit's footprint gives, so the fold cannot decide two strokes are
independent where the log would decide they conflict") and `fill_rect`'s comment claims
the same lineage — but it is shared with the live-preview fold and *not* with `plan`,
which is the consumer that matters.

This is the failure `TileRect::covering`'s own doc comment says it was introduced to end:
*"this arithmetic was written out five times across `document/`, and the copies disagreed
in exactly the place that matters."* One copy survived, on the far side of `tile_box`.

**Verification.** A unit test in `fill.rs` at a tile-aligned rect asserting
`plan(op, gate) ⊆ fill_rect(op)` for a sweep of offsets across one tile stride, plus a
fill at such an offset added to `tests/footprint.rs`'s run — which is the instrument that
should have caught this and would have, at the right coordinates.

## 2. `MergeLayerDown` writes three composite params and declares one — open

`ActionKind::apply`'s merge arm assigns `composite: keeps` — blend, clip **and** opacity.
`footprint`'s arm declares only `Resource::Prop(*dest, Prop::Opacity)` as a write; blend
and clip are declared as *reads*.

It is sound today, and only by guards in a different file. `merge::plan`'s sibling arm
reaches `keeps.clip = false` only after refusing `d.composite.clip`, and sets
`keeps.blend` from `d.composite.blend`, so both assignments are the identity; the carrier
arm sets `keeps = d.composite` wholesale. Nothing in `footprint.rs` says so, nothing in
`action.rs` says so, and no test drives the pair (§3).

That is "a rule a call site could forget", which §1 of CLAUDE.md says this codebase
spends structure to avoid. It is also the rule most likely to be quietly invalidated
next: `merge.rs`'s own header names two merges it means to build — two layers sharing a
blend mode, and a source with a mode merged into its carrier — and both touch `keeps`.

**What to do.** Declare `Prop(dest, Blend)` and `Prop(dest, Clip)` as writes. A false
conflict costs the fast path and nothing else, which is the direction §12.6 says to err
in. Alternatively have `apply` assign only the field it means to change, which says the
same thing from the other end and is the smaller diff — but the first is the honest one,
because `keeps` really is the whole triple that speaks for the survivor.

## 3. The footprint-honesty run misses three kinds, and its guard rots — open

`tests/footprint.rs` holds every committed action to its declared writes by diffing the
document across the commit. It is the best instrument in the crate and it is why §1 and
§2 are findings rather than incidents waiting to happen — except that its `expected` list
names 21 kinds and `ActionKind::label` has 25.

Missing: **`MergeLayerDown`, `AddFilter`, `SetFilter`.** (`Undo` is excluded by
construction, and the module note says why.) `MergeLayerDown` has by far the most complex
footprint in the enum — ten reads, five writes, and an `apply` that rewrites tiles *and*
removes a layer *and* reassigns params — and it is the one §2 lives in.

The list's own header explains that it exists so the test cannot "rot into vacuity one
command at a time: a step that stops committing *silences* its own coverage rather than
failing." That argument is exactly right and it does not cover the case that happened:
the list is hand-maintained, so **a new variant does not fail it**. The anti-rot guard
acquired the rot it was built against.

**What to do.** Make the label table the single source. Declare
`ActionKind::LABELS: [&'static str; N]` beside `label()`, have `label()`'s arms return
from it, and have the test assert `seen == LABELS` rather than `seen ⊇ expected`. A new
variant then fails at `label()` (exhaustiveness) *and* at the run (coverage), which is
the same two-sided guard `Modulations::all`'s `..`-free destructure gets for a struct.
Immediately, and independently: drive the three missing kinds.

## 4. Three parallel matches over `ActionKind`, linked only by prose — open

Adding an action means editing five exhaustive matches across three files: the enum,
`label` and `apply` (`action.rs`), `footprint` (`footprint.rs`), and
`StatePatch::capture` (`patch.rs`). Rust's exhaustiveness gets the *presence* of an arm.
Nothing gets *correspondence* — that the footprint's arm names what `apply`'s arm goes on
to touch, and that `capture`'s arm restores what the footprint's arm declared.

`tests/footprint.rs`'s module header states this plainly and treats it as a fact of life.
§1 and §2 are what it costs.

`paint_rect` already has the right instinct and says so: it **asks the footprint** for the
rect rather than re-deriving it, "so the region a restore rewrites is the very region the
action declared; the two cannot drift." Generalize it.

**What to do.** Drive `capture` off `footprint(action).writes` entirely. The mapping is
already one-to-one, resource for op:

| write resource | patch op |
|---|---|
| `Paint(l, rect)` | `tile_diff(l, rect)` |
| `Existence(id)` | `Absent` / `Present` |
| `Prop(id, p)` | the matching op, `p` for `p` |
| `StackOrder` | `Structure` |
| `Selection(a)` / `Surface` / `Background` | the matching op |

That collapses two of the five matches into one, and makes "the patch restores exactly
what the footprint declares" true by construction rather than by inspection — the same
move §5 of the stroke ledger made when three answers to "what region do these sweeps
need" became one `Coverage`.

**Two things it has to resolve**, both checked and neither fatal:

- `AddLayer`/`RemoveLayer`/`DuplicateLayer` would additionally capture `Structure`, which
  they do not today. Under the commutation guarantee the suffix contains no structural
  edit — `StackOrder` is what guarantees it — so the rebuild is an identity. It is
  redundant work, not wrong work, and it can be skipped by letting an `Existence` op in
  the same patch subsume the `StackOrder` one.
- `MergeLayerDown`'s current capture order is load-bearing by comment ("the tile diff
  comes first so the destination is there to take the source back when its site names it
  as its carrier"). A writes-driven order would run `Present(source)` first. Either
  preserve the ordering explicitly or establish that it is unnecessary — the destination
  survives the merge, so it is present either way, and the comment may be describing a
  hazard that no longer exists.

## 5. `apply`'s merge arm is 90 lines of orchestration inside a match — open

Every other arm of `ActionKind::apply` is a line or two delegating to a `DocState` method
or a module-level helper — `transform_apply` is the pattern, shared by the three
transform kinds. `MergeLayerDown` is ~90 lines of GPU orchestration inline, including an
`unreachable!` used to re-destructure a `MergeKind` the arm has already matched on once.

`merge.rs` owns the decision and says so in its header ("Pure CPU, no GPU anywhere in
this file"). It should own the application too: `merge::apply(state, ctx, source, dest)
-> DocState`, with the `MergeKind` handled by one `match` rather than an `if let` plus a
`let … else`, which is where the `unreachable!` comes from and where it goes.

The general form of the finding: `action.rs` is the action **vocabulary** — the enum, its
wire form, its captions — and it is currently also the largest single site of action
**interpretation**. Those are the two subjects `segments.rs` was split for one directory
over.

## 6. `observe` re-searches the tree per layer for `merge_down` — open

`Engine::observe` calls `merge::plan(shown, l.id)` for **every layer** in the projection.
Each call does `site_of` (a full `locate` walk), `layer(source)` (another), and
`layer(carrier)` / `layer(dest)` (up to two more), plus a `MergePlan` allocation and a
`Filter` clone. O(L²) node visits and O(L) allocations per projection.

The field directly above it in the same struct literal is `has_backdrop`, whose comment
reads: *"`Engine::observe` reads it off this walk for free; a standalone predicate
searched the tree per layer to say the same thing."* `merge_down` is that search,
reintroduced two lines later.

`observe` is per-command rather than per-sample — `dispatch_sample` deliberately skips it,
and says why — so this is not a stroke-latency finding. It is the cost curve that matters:
the projection is the one thing every command pays for, and this is the only field on it
derived by a search.

**What to do**, either:

- Derive it inside the `visit` walk, which already holds the carrier chain and the row
  position — everything `plan` re-searches for. `plan` would take a located context
  (`&Layer`, `LayerSite`, and the stack it sits in) rather than an id, which is also what
  makes it testable without a `DocState`.
- Or make it a **request**, on `scrub_range`'s own stated argument: *"asked for only while
  a scrubber is on screen, and putting it in the projection would have every command …
  pay for a number nothing else reads."* `merge_down` is read only when the layer panel
  draws its row. This is the cheaper change and the codebase's own precedent for it.

## 7. The commutation scan allocates per comparison and derives itself twice — open

`Footprint` is `{ reads: Vec<Resource>, writes: Vec<Resource> }`, and
`Centralizer::commutes` calls `footprint(other)` fresh for every candidate action. The
fast path's whole claim is that it is cheaper than a replay, and it currently pays two
heap allocations per action examined.

Worse: `ReplicatedTimeline::resync` builds a footprint for the removed action and one per
suffix action, walking the suffix to find the first conflict — **purely to populate
`TimelineStats`** — and then hands the identical work to `history`'s
`remove_action_with`, which does it again. That is a 2× cost on the exact path §12.6
exists to make cheap, spent on a counter. The comment is candid about why ("The history
doesn't report which path it took, and pixels can't show it — so re-derive it for the
stats"), which makes it a known trade rather than an oversight; it is still the wrong
side of the trade on the path the design is proudest of.

**What to do**, in payoff order:

- Have `remove_action_with` report how far it got, and derive the stats from that. If
  `history` cannot change, gate the derivation on the stats actually being read.
- `SmallVec<[Resource; 4]>` for both fields. The median footprint is one read and two
  writes; this removes the allocation entirely for everything but §8's two kinds.
- Cache the footprint beside each entry in `ReplicatedTimeline::log`. It is a pure
  function of an immutable action, so recomputing it per comparison is recomputing a
  constant — the same argument `Targets` and `undone` are already cached on.

## 8. "Everything about this layer" is spelled nine resources at a time — open

`DuplicateLayer`'s footprint emits nine `Resource`s **per copied layer** — `Existence`,
`Paint(ALL)` and all seven `Prop`s — and `MergeLayerDown` emits five per side. The
reasoning is right and the docs argue it well: a copy really is a function of every tile
and every property of every layer it copies.

But `Footprint::conflicts` is a nested `any`-over-`any`, so it is O(n·m) in the two
footprints' lengths. Duplicating a twenty-layer group produces 180 read resources, and
every commutation test against it is 180 × m `Resource` comparisons — with `Resource`
being a `PartialEq` derive over an enum holding a `TileRect`.

Both actions are saying one thing. **A coarse `Resource::Layer(id)`** that `overlaps`
every `Prop(id, _)`, `Paint(id, _)` and `Existence(id)` says it in one entry, shrinks both
footprints by an order of magnitude, and reads as what the doc comments already claim. It
is the same move `StackOrder` makes for the tree ("One coarse resource: two concurrent
restructures genuinely don't commute, and structural edits are rare enough that finer
granularity would buy nothing").

`Resource::overlaps` is already the place asymmetric containment is expressed, so the
change is local to it plus the two footprint arms.

## 9. A fill allocates tiles where its coverage is provably zero — open

`op.reach()` is `feather/2 + TILE_APRON + 1`, which is documented as *exactly* how far
coverage can travel past the shape boundary — "coverage is exactly zero beyond `w/2`, and
the apron carries a tile's write one band further" — and is deliberately tighter than
`Selection::plan`'s padding for precisely this reason ("reusing it here would ring every
fill with a band of all-zero paint tiles that would then pollute `bounds` and hold pool
memory").

`plan` then pads by a second apron through `tile_box` and returns the whole box, and
`FillRenderer::apply` mints a fresh tile for every coord in it. So the band the comment
argues against is built anyway, one apron thinner.

`transform` has the answer already: `TransformPlan::drops` exists so that "a rewrite would
write all zeros, and an all-zero tile pollutes `bounds` and holds pool memory that 'no
tile' would not." A fill wants the same rule — either prune coords whose coverage is
provably zero before the walk, or let the renderer skip the insert for a tile whose gate
and region masks are both empty.

Fixing §1 by making `fill_rect` match `plan` closes the correctness hole; fixing this one
lets both boxes shrink to the honest reach instead.

## 10. Smaller correctness and clarity items — open

- **A broken intra-doc link.** `ActionKind::SetFilter`'s comment links
  ``[`SetMatteColor`](Self::SetMatteColor)``, a variant renamed to `SetMattePaint`. Latent
  because CI runs fmt, clippy, tests and the wasm build but not `cargo doc` — worth adding
  a doc build with `-D warnings` given how much of this module's design lives in its doc
  comments. (`io.rs` and `engine/mod.rs` also name `SetMatteColor`, but in prose about the
  wire history, which is correct and should stay.)
- **`rpds::Vector` buys nothing for a layer stack.** Stacks are "dozens" by
  `with_layers`'s own account, and `insert_at` and `remove_in` both rebuild the whole
  spine element by element anyway. `Arc<Vec<Layer>>` with copy-on-write would be simpler
  and faster at this size. The thing that genuinely needs persistence — the tile map — is
  already a `HashTrieMap`. Low priority; it is correct as it stands.
- **Two files carry more than one subject.** `state.rs` is `DocState` plus a
  self-contained persistent-tree library (`map_in`, `remove_in`, `splice`, `insert_at`,
  `copy_subtree`, and the `IN_RANGE` contract that governs them). `transform.rs` is
  projective geometry (`Homography`, `PerspectiveMap`, the f64 square-to-quad derivation,
  `mat3_adjugate`) plus tile planning plus separating-axis intersection — three subjects
  sharing only a caller. Both splits are the mechanical move `region.rs` and
  `dynamics/bleed.rs` already were.
- **`StatePatch::restore` clones the state per op**, and each clone is followed by a
  `with_layers` walk of the whole tree. At three ops for a merge this is noise; it is
  worth knowing it is linear in ops rather than constant, since §4 would add ops.

## 11. `action.rs` and `patch.rs` carry no in-file tests — open

Of the module's ~2.0k test lines, **zero** are in the two files that hold the seam §4
describes. `merge.rs` has 383, `transform.rs` 244, `state.rs` 208, `timeline.rs` 180 —
the pure functions are well covered, several of them exemplary. `apply` and `unapply` are
covered only through `tests/footprint.rs` and `tests/commute.rs`, both of which need a
GPU adapter and both of which are skipped under `STARK_ALLOW_NO_GPU`.

The gap that matters is `unapply`. It is the inverse of every action in the enum, its
soundness argument is subtle (the two states are *not* adjacent, and the rect bound on
`tile_diff` is load-bearing rather than an optimization), and nothing exercises it below
the integration level. A CPU-only round-trip — build a `DocState` by hand, apply a
property or structural action, `unapply` it, assert the state is equal — needs no adapter
for every kind except the four that render, and would have caught §2 directly.

`restore_structure`'s two defensive arms are the specific case: the `placed` guard against
a shape naming a layer twice, and the "anything the shape predates keeps stacking on top
of the root" fallback. Both are reachable only from a crafted log, both are argued in
comments, and neither has a test.

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
  texels it covered, so `combine` would have been wrong there. The corner case (level 0
  reflecting to an empty inverse) is named and reachable only on purpose.
- **`Place` and `BlendMode` are pinned byte-for-byte** against the encodings §8 promises,
  rather than reasoned about — which is the only defence against the failure §8 describes,
  where a reordered variant silently re-reads every saved document.
- **`LayerId::mint` partitions by author**, so two peers adding a layer at the same moment
  cannot collide, and `ActorId::SOLO` maps to high half 0 so a never-shared document keeps
  `LayerId(0)`.
- **`TileRect::covering` genuinely closes the NaN/overflow class** — `i64` throughout,
  saturating, `None` rather than a wrap — for every caller except the one §1 is about,
  which reaches it through the wrong wrapper.
- **The warp's deviation-form Hermite gives a bit-exact identity**, and
  `identity_mesh_lattices_exactly_onto_its_base` asserts it bitwise rather than
  approximately, which is what §16.4's identity invariant actually needs.
- **`Selection::from_parts` forces `hull = None` whenever `outside > 0`**, so a caller
  cannot hand it a box that contradicts coverage reaching infinity.

`Modulation`'s rational response curve and its `max_slope` bound were checked against
their tests and are exactly what the flattener needs; `Modulations::all`'s `..`-free
destructure is the pattern §3 should copy.

---

## Verification, when these land

The suite is slow — one `cargo test --workspace` redirected to a file per batch, grepped
rather than re-run (CLAUDE.md, and `test-suite-is-slow-run-once`).

- **§1 and §9 move pixels.** A fill's written tile set changes, so goldens containing one
  must be re-blessed with the model fixed rather than compensated for — and the diff has
  to be *only* the boundary ring, which is the check that says the fix is the fix.
  `STARK_SKIP_GOLDEN` unset, `STARK_ALLOW_NO_GPU` unset.
- **§2, §4, §7, §8 must be pixel-identical.** They change what is declared, restored and
  compared, never what is drawn. Any golden movement is a bug in the change.
- **§3 is the gate for §1 and §2** and should land first: the three missing kinds and the
  `LABELS` guard, so the two correctness fixes have an instrument pointed at them before
  they are made.
- **§6 wants a number.** It is a CPU cost on the command path, so it is not visible to
  `--bench stroke`; time `observe` directly against a document with 4, 20 and 60 layers
  before and after, or it is a refactor claiming a performance motive it never measured.
- **§5, §10, §11** are structural and additive; the ordinary green run covers them.

`cargo clippy --workspace --all-targets -- -D warnings` and the second configuration
(`--no-default-features --features stark-net/webrtc`) throughout, plus
`cargo check -p stark-ui --target wasm32-unknown-unknown` for anything that reaches
`engine/mod.rs` — §6 does.
