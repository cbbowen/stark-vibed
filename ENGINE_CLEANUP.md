# Engine cleanup

A review of `crates/stark-core/src/engine/` — the defects it found, the
architectural changes that follow from them, and what was done about each.

**Status: worked through.** Everything below is landed except the two items marked
*not done*, each of which says why. Functions are cited by name rather than by line,
per CLAUDE.md.

## Three confirmed bugs, one root cause — **fixed**

All three were the same shape: **a `match` over `ActionKind` with a `_ =>` arm,
which a later-added variant walked straight past.** §17.9 records two of these
defect *classes* as fixed; each was reopened by a feature that arrived afterwards.
Each was reproduced with a throwaway test before being written down, and each now
has a regression test in the tree.

### 1. `resync_counters` didn't know `AddFilter` mints a layer id

`AddFilter` mints through `mint_layer` like every other `Add`, but the resync
listed three variants and it was not one of them — and `DocState::insert` has no
duplicate-id guard. So a document whose highest ordinal came from a filter reloaded
with a counter that would mint that id a second time:

```
before save: [LayerId(0), LayerId(1)]
after load:  [LayerId(0), LayerId(1)]
after add:   [LayerId(0), LayerId(1), LayerId(1)]   <-- two layers, one id
```

Through `join_collaboration`, which resyncs the same way, it is a convergence
failure outright.

Now asked of `ActionKind::minted_layers`, beside the variants.
*Test:* `filter.rs::a_filters_id_is_not_reused_after_a_reload`.

### 2. Undo of an `AddLayer` stranded the active layer

Found while fixing #3, and the most ordinary of the three: `AddLayer` arms the layer
it added, undo withdraws exactly that layer, and nothing repointed the brush. Add a
layer, change your mind, and every subsequent stroke was silently refused.
*Test:* `layers.rs::undoing_an_add_leaves_the_brush_somewhere_it_can_paint`.

### 3. A remote `MergeLayerDown` stranded the active layer

`merge_remote` keyed the repoint on the `RemoveLayer` variant, but `merge_apply`
ends in `.remove_layer(source)`. A peer merging down the layer you are painting on
left you pointing at nothing.
*Test:* `collab.rs::a_remote_merge_down_does_not_strand_the_active_layer`.

**Both repoints are one now.** The rule is not "which actions remove a layer" but
"the document has been replaced", and `committed_changed` is already the single
funnel every commit, undo/redo step, seek, merge, share, join and reset comes
through. The repoint went there and four keyed call sites went away.

## A. Per-variant questions moved onto `ActionKind` — **done**

Three live wildcards, all now exhaustive with no `_` arm, following
`tests/footprint.rs::slot` — whose own comment names `AddFilter` as the variant
that escaped *it* once:

- `ActionKind::minted_layers` — replaces the list in `resync_counters`.
- `ActionKind::is_noop_on` — see C.
- `content::action_content` — was latent (only `CommitStroke` names an asset), but a
  wildcard answers "needs nothing" for every variant that does not exist yet, and an
  unbundled ground bakes a smooth deposit into tiles no later arrival un-bakes.

`Engine::document_file` also kept a second scan of the log for content, beside the
one `content::action_content`'s doc comment calls "the single definition". Folded
onto `required_content`.

## B. `Engine`'s loose field groups — **done**

- **`build_gpu` returned a seven-tuple**, assigned field-by-field in two places. It
  returns the whole `ApplyCtx` now, so `rebuild_gpu_for` is `self.apply =
  built.apply` and a renderer added to the context is rebuilt by construction. What
  a rebuild *keeps* is `GpuKeep` — the interesting half of that function, stated as
  a list.
- **`ApplyCtx` derives `Clone`**, and `new_sharing` uses it. Cloning nine fields by
  hand meant a renderer added to the context was shared by every engine except the
  preview one.
- **`Authoring`** groups `actor`, `clock`, `next_layer` and the outbox, because they
  move as one thing. `reset_document` is one assignment rather than five.
- **The outbox is `Option<Vec<Action>>`**, so "queued actions that will never be
  sent" is unrepresentable and the queue's presence *is* `is_shared`. It also buys a
  real saving: `commit` cloned its action unconditionally, and a `CommitStroke`
  carries the stroke's whole control-point list — a solo session duplicated one per
  commit to drop it an instruction later.
- **`debug_samples` is `#[cfg]`**, not a `cfg!` around a field that exists anyway.
  That also fixed a live inconsistency: `Start` was ungated while `To` was gated, so
  a shipping build kept the first sample of every stroke and dropped the rest.

**The `is_shared` / `scrub_range` divergence is documented, not fixed.**
`end_collaboration` stops the broadcast but keeps the `ReplicatedTimeline`, which
takes the trait's default `scrub_range` (`None`) — so after leaving a session the
history scrubber stays unavailable until a new or loaded document brings a linear
history. That is *consistent* (there is no seek on a replicated timeline), so the
thing that was lying was `is_shared`, and it now says which question it answers.
Making the scrubber come back means implementing `seek` on `ReplicatedTimeline`,
which is a feature rather than a cleanup.

## C. A setter's two rules are one call — **done**

- **"Don't spend an undo step on nothing"** had four hand-rolled shapes and was
  missing from `SetLayerVisible`, `SetLayerClip`, `SetMatteRect` and
  `SetBackground`. Now `ActionKind::is_noop_on`, beside `apply` — which is also the
  answer to a discomfort three of those arms wrote down themselves ("a second rule
  about what a matte is, kept somewhere `apply` cannot see").
- **"A commit supersedes the drag"** was 11 `preview.set_doc(None)` calls, present
  at 8 commit sites and absent from 13, including the gesture commit. It moved into
  `commit`.
- `settle` is the two together, because a slider released on the value it was
  pressed on must log nothing *and* still drop what it was showing.

*Tests:* `layers.rs::setting_a_value_to_the_one_it_already_holds_is_not_an_edit`,
`layers.rs::a_commit_supersedes_a_drag_it_knows_nothing_about`.

## D. The eyedropper's per-sample cost — **half done, and the review over-claimed it**

The draw list is now built **once** for a trace, culled to the union of every patch
(`patch_view`, `patch_cull`, `TileRect::union`), instead of once per point. That is
the honest expression of "the list does not depend on the point" and it is what the
review asked for.

**But it was not where the time went.** Measured on a 9-layer document, a
128-sample trace: **4.78 ms → 4.65 ms**. The patch cull already held each pass to a
couple of tiles, so the tree walk was never the cost.

What the time is actually in is **128 queue submissions and ~380 texture creations**
— `composite_channels` owns a command encoder and submits it per call. Batching
those into one encoder is a real change to that method's shape: the per-compositor
upload buffers are reused between calls, so N passes in one submission would need
them not to be. **Not done**: 4.65 ms for a one-shot gesture does not buy that
risk. Revisit if the trace ever runs at pointer rate.

## E. Smaller items

- **`export` is `export_view` through the plan's view** — done. The
  render-then-readback tail, including the borrow bargain, was written out twice.
- **`ObservableState` derives `PartialEq`** — done, plus `ViewTransform` and
  `MediaParams`, the only two fields that lacked it. The frontend can now notice
  that a re-publish did not move; taking that up is a `stark-ui` change.
- **`contributes()` in `observe` is O(n·depth)** — **not done**, deliberately. It
  recurses each layer's subtree once per visited row, so a deep chain of groups is
  quadratic. Unlike the `merge_down` search that shared its shape (79 µs at 60
  layers), this one is an allocation-free boolean walk that short-circuits as soon
  as the stack is filled: the pathological case is a few microseconds. Restructuring
  a readable pre-order walk into a post-order accumulation is not worth that.

## What is left

Nothing from this list except the two marked *not done*, both of which are judgement
calls recorded above rather than omissions. The larger thing this cleanup was
sequenced ahead of — the §7 actor migration — is now better placed for it:
`ApplyCtx`, `GpuKeep` and `Authoring` are the seams a channel would be drawn along.
