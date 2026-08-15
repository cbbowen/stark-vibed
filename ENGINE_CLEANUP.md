# Engine cleanup

A review of `crates/stark-core/src/engine/` — two confirmed defects and the
architectural changes that would stop the next one of their kind. Nothing here is
done; this is the work list.

Functions are cited by name rather than by line, per CLAUDE.md — line numbers rot
and these will move as the list is worked through.

## Two confirmed bugs, one root cause

Both are the same shape: **a `match` over `ActionKind` with a `_ =>` arm, which a
later-added variant walked straight past.** Each was reproduced with a throwaway
integration test before being written down; the reproductions are below, ready to
become regression tests.

### 1. `resync_counters` doesn't know `AddFilter` mints a layer id

`engine/file.rs::resync_counters` notes `AddLayer`, `AddMatte` and
`DuplicateLayer`. But `process_doc_inner`'s `AddFilter` arm mints an id through
`mint_layer` as well, and `DocState::insert` has no duplicate-id guard — so after a
load or a join, `next_layer` resumes below an id already in the log:

```
before save: [LayerId(0), LayerId(1)]
after load:  [LayerId(0), LayerId(1)]
after add:   [LayerId(0), LayerId(1), LayerId(1)]   <-- two layers, one id
```

Reproduced by: add a filter, `save_bytes`, `load_bytes`, add a layer. From there
`layer(id)` finds whichever comes first, so painting, renaming and deleting all
reach the wrong row. Reached through `join_collaboration` — which calls the same
`resync_counters` — it is a convergence failure: exactly the defect §17.9 says
per-client identity was introduced to rule out.

### 2. A remote `MergeLayerDown` strands the active layer

`engine/collab.rs::merge_remote` extracts the removed layer from `RemoveLayer`
alone, with `_ => None`. But `document/action.rs::merge_apply` ends in
`.remove_layer(source)`, so a peer merging down the layer this client is painting on
leaves `session.active_layer` dangling:

```
B's active layer LayerId(10451216376902189057) no longer exists
— every stroke will be silently refused
```

Reproduced by: A shares, paints, adds a layer, paints again; B joins and selects the
top layer; A merges it down; B merges the action. This is verbatim the second defect
§17.9 records as fixed ("a remote `RemoveLayer` could strand the active layer"),
reintroduced by a feature added afterwards.

**The fix is to stop asking which actions remove a layer.** Delete the `removed`
extraction and call `repoint_active_layer()` unconditionally on a successful merge:
it already returns early when the layer still exists, so it costs one
`contains_layer` per remote commit and cannot miss a variant.

## A. Move per-variant questions onto `ActionKind`, exhaustively

Three live wildcards — `engine/collab.rs::merge_remote`,
`engine/file.rs::resync_counters` and `content.rs::action_content` — each keep a
fact about actions *away from* the enum, where adding a variant is silent instead of
loud.

The idiom to follow is already in the tree: `tests/footprint.rs::slot` is exhaustive
with no `_` arm, and the comment beside it explains why — and even names `AddFilter`
as the variant that escaped an under-specified list once before. The two bugs above
are that same comment, one scope over, unheeded.

So put the questions next to `apply`, in `document/action.rs`:

```rust
impl ActionKind {
    /// Every layer id this action *mints*. Exhaustive with no `_` arm: a new
    /// variant that mints one stops this file compiling.
    pub fn minted_layers(&self) -> impl Iterator<Item = LayerId> + '_ { ... }

    /// Whether applying this can remove a layer — what the brush is repointed
    /// after, locally and on a merge alike.
    pub fn removes_layers(&self) -> bool { ... }
}
```

`content.rs::action_content` should lose its wildcard the same way. That one is
latent today — only `CommitStroke` names an asset — but a gradient fill or a matte
carrying an id would silently save an unbundled document, which is the failure the
whole bundling path exists to prevent (§8).

While there: `engine/file.rs::document_file` hand-rolls the "which brush shapes does
this log name" scan that `content::action_content` already answers, and
`referenced_surfaces` hand-rolls the ground scan beside it. Three walks of the log
across two modules answering one question. Fold `document_file` onto
`required_content()` so there is one list to keep right.

## B. `Engine` is a 30-field, 66-public-method god object

Splitting the `impl` across six files was right for readability, and the module doc
is honest that it is "a division of the *file* and not of the type" — but that means
all 122 methods still reach all 30 fields and nothing is enforced. Three tightenings,
in increasing order of effort:

**`build_gpu` returns a 7-tuple**, destructured and assigned field-by-field in two
places (`new_with_color_space`, `rebuild_gpu_for`). Return one
`GpuStack { pool, stroke, transform, fill, merge, compositor, compositor_pipeline }`
held as a single field, and a rebuild becomes one assignment instead of seven a new
subsystem could be left out of.

**The 30-field struct literal is written twice** — `new_with_color_space` and
`new_sharing` — and then reset piecemeal a third time in
`engine/file.rs::reset_document`. Three places that have to agree about what a fresh
session is, where the compiler catches a *missing* field but never a *wrong* one.
Group the authoring scalars:

```rust
struct Authoring { actor: ActorId, clock: u64, next_layer: u64, outbox: Vec<Action>, outbox_enabled: bool }
```

Then both constructors say `authoring: Authoring::solo()`, `reset_document` says the
same, and "did I remember to clear the outbox" stops being a question.

**`outbox` + `outbox_enabled` should be `Option<Vec<Action>>`**, which makes "queued
actions while not sharing" unrepresentable. Related, and already broken:
`is_shared()` reports `outbox_enabled`, but `end_collaboration` leaves the
`ReplicatedTimeline` in place — and `ReplicatedTimeline` takes the trait's default
`scrub_range`, which is `None`. So after leaving a session `is_shared()` says solo
while the history scrubber stays permanently dead. Two notions of "shared" that have
already diverged.

This is also what unblocks the §7 actor migration. `export` and `pick_color` already
work around whole-`&mut self` borrows by cloning what their futures need; that is a
symptom of the single-owner shape, not a design.

## C. Make the two per-command rules structural rather than enumerated

**"A commit supersedes the drag preview."** Written out at 11 call sites, each with
its own paragraph of justification — and **13 of the 24 commit paths omit it**:
`Select`, `InvertSelection`, `SetSurface`, `AddLayer`, `AddMatte`, `AddFilter`,
`DuplicateLayer`, `RemoveLayer`, `MergeLayerDown`, `SetLayerClip`,
`SetLayerVisible`, `SetLayerName`, `MoveLayer`, and the gesture-end commit. Hold a
slider so a preview is installed, delete a layer by keyboard without releasing, and
the canvas goes on showing the pre-delete document.

Move `self.preview.set_doc(None)` into `commit` itself. An unlogged drag is by
definition superseded by any logged change, and `merge_remote` does not route
through `commit`, so a peer's edit still will not cancel a local drag. This is the
"rule out a class rather than enumerate its instances" convention exactly.

**"Don't spend an undo step on a no-op."** Hand-rolled in four different shapes
(`SetLayerOpacity`, `SetLayerBlend`, `SetFilter`, `SetMattePaint`, plus
`SetLayerName` and `SetSurface`), while `SetLayerVisible`, `SetLayerClip`,
`SetMatteRect`, `SetBackground` and `MoveLayer` have none — so toggling the eye to
the value it already holds costs an undo step, and so does `SetBackground` with the
color it already is.

The comments at those sites already worry about "a second rule about what a matte
is, kept somewhere `apply` cannot see". Answer that by putting the predicate where
`apply` lives:

```rust
impl ActionKind {
    /// Whether applying this to `state` would leave it as it found it.
    pub fn is_noop_on(&self, state: &DocState) -> bool { ... }
}
```

and have `commit` consult it once. The engine then holds no rules at all about what
a filter or a matte is.

## D. The eyedropper's per-sample cost

`engine/pick.rs::pick_colors` loops over points and, **per point**, rebuilds the
entire draw list (`composite_groups` — a full layer-tree walk cloning an
`Arc<GpuTile>` per visible tile) and creates two or three textures. `pick_gradient`
traces up to `gradient::MAX_SAMPLES` (128) points, so one gradient capture is 128
tree walks and up to 384 texture creations before the — already batched — readback.

Two independent wins, neither changing what is sampled:

- **Hoist the draw list.** It varies with the point only through
  `visible_tiles(view)`. Take the union of the patch rects once, build `groups`
  once, reuse it for every point: 128 walks become 1.
- **Stop allocating per point.** A patch is at most 65×65, so render them into one
  atlas texture (or a small reused ring) rather than a fresh triple each. That also
  collapses the readback to a single texture.

## E. Smaller items

- `engine/render.rs::export` and `::export_view` duplicate the identical
  render-then-readback tail, including the `use<>` future shape; they differ only in
  how the view is derived. `export` could be `export_view(plan.view())` once the
  device-limit check is shared.
- `debug_samples: Vec<InputSample>` is a real field in shipping builds, gated by a
  runtime `cfg!()` at three sites rather than by `#[cfg]` on the field. It costs
  nothing, but it contradicts "do not add inert scaffolding".
- `contributes()` inside `observe` walks each layer's whole subtree, once per
  visited layer — O(n·depth). Harmless on a flat document, quadratic in a deep chain
  of groups. The same shape as the `merge_down` search the comment beside it already
  celebrates having removed.
- `ObservableState` derives `Clone, Debug` but not `PartialEq`, so the frontend's
  `obs.set()` marks every subscriber dirty even when the projection is identical.
  `LayerInfo` and `MatteInfo` already derive it; the container is the only thing
  missing.

## Suggested order

1. **The two bugs plus A.** Contained, two proven defects behind it, and the
   exhaustive `ActionKind` methods are what stop the third.
2. **C.** Small, and removes a whole class at each of two sites.
3. **B.** The larger refactor. Worth sequencing *before* the §7 actor work rather
   than as a project of its own.
4. **D**, if gradient capture is felt to be slow. **E** as it is passed.
