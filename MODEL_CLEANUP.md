# `stark-model` cleanup ledger

A critical review of `crates/stark-model` (~12.5k lines) and its seams into
`stark-engine`, recorded 2026-08-25 against `6a4d198`.

**Status: settled.** Fourteen of the fifteen findings are fixed on this branch, and
the fifteenth was withdrawn as wrong — see [N](#n-withdrawn-footprints-two-vecs-per-action).
The table below carries the commit that closed each one, and each section keeps its
original diagnosis so the reasoning stays readable next to the fix.

Two things made this the moment to spend it: §19's beta rung has not been claimed, so
the save format was still free to change (which [G](#g-parcel-and-mattepaint-are-the-same-type-twice)
and [H](#h-the-three-content-bags-should-be-one) both needed); and none of the
findings needed a bit-identical result, so a wrong model could be fixed and the
goldens re-blessed. In the event none of them moved a golden.

## What it turned out to be about

The headline finding was [A](#a-commitstroke-under-declares-its-footprint), and the
shape of it is worth keeping: **a stroke reads the substrate and did not say so.**
Both peers took the same splice and converged on the same *wrong* picture, so the
failure was not a disagreement between clients — it was a document that had stopped
being what its own log said, and changed the next time anyone opened the file.

It survived a suite that already covered the splice because `shift_late` re-applies
the single action at each cached state: one action shifted past comes out right by
accident, whatever the footprint says, and only a *run* between two snapshots is left
behind. That is why [B](#b-the-reads-half-of-every-footprint-is-unverified) exists and
why its table carries five strokes.

## The order to spend it in

| # | Finding | Kind | Closed by |
|---|---|---|---|
| [A](#a-commitstroke-under-declares-its-footprint) | `CommitStroke` under-declares its footprint | **correctness** | `3fc17e1` |
| [B](#b-the-reads-half-of-every-footprint-is-unverified) | `Footprint::reads` is structurally unverified | **correctness** | `dbfbfab` |
| [C](#c-lasso-decimation-overflows-usize-on-wasm32) | Lasso decimation overflows `usize` on wasm32 | **correctness** | `fd0e943` |
| [D](#d-max_lasso_points-states-a-coupling-nothing-can-check) | `MAX_LASSO_POINTS`' texture bound is unenforceable | correctness | `fd0e943` |
| [E](#e-a-placement-loses-the-exactness-it-promises) | `image::extent` loses `PlaceImage`'s exactness | correctness | `0203a75` |
| [O](#o-23-rustdoc-warnings-and-no-ci-gate) | 23 rustdoc warnings, no CI gate | code health | `f6c75f5` |
| [I](#i-fillop-and-selectionop-defeat-their-own-funnels) | `FillOp`/`SelectionOp` public fields | architecture | `4e993a9` |
| [H](#h-the-three-content-bags-should-be-one) | Three content bags should be one | architecture | `fa0ffd8`, `2c6985c` |
| [G](#g-parcel-and-mattepaint-are-the-same-type-twice) | `Parcel` ≡ `MattePaint` | architecture | `9dcff3a` |
| [L](#l-timelineresync-recomputes-footprints-for-statistics) | `resync` recomputes cached footprints | performance | `7aeb45c` |
| [M](#m-preparedeval-still-allocates-in-the-loop-it-was-made-for) | `Prepared::eval` allocates per call | performance | `f6fe856` |
| [F](#f-the-actionkind-roster-tax) | The `ActionKind` roster tax | maintainability | `8ad97a8` |
| [J](#j-documentguiders-is-2000-lines-of-two-different-things) | `guide.rs` is two modules in one file | maintainability | `4c15bb9` |
| [K](#k-the-crates-nan-policy-has-five-private-re-implementations) | The NaN policy has five re-implementations | maintainability | `171ea00` |
| [N](#n-withdrawn-footprints-two-vecs-per-action) | ~~`Footprint`'s two `Vec`s per action~~ | **withdrawn** | — |

## What is left

Two follow-ups this branch names but does not take:

- **The rustdoc gate covers `stark-model` only.** The workspace reports ~300
  warnings, most of them the generated shader mirrors' `binding::*` links. Widening
  it is a crate at a time, as each is brought to zero — see
  [O](#o-23-rustdoc-warnings-and-no-ci-gate).
- **A saved gradient *matte* no longer loads**, a deliberate cost of
  [G](#g-parcel-and-mattepaint-are-the-same-type-twice). Solid mattes and gradient
  fills are unaffected. Worth a line in the release notes whenever §19's rung is
  claimed.

---

# 1. Correctness

## A. `CommitStroke` under-declares its footprint

**This is the §12.6 failure CLAUDE.md's first rule warns about, and it is live.**

`crates/stark-model/src/document/footprint.rs:301`

```rust
ActionKind::CommitStroke(rec) => Footprint {
    reads: vec![Resource::Existence(rec.layer), Resource::Selection(actor)],
    writes: vec![Resource::Paint(rec.layer, stroke_rect(rec))],
},
```

`crates/stark-engine/src/document/apply.rs:342`

```rust
// The substrate this stroke was painted on, as the log stood here —
// not as it stands now (§6.4). The tooth gates the deposit by it, ...
let substrate = ctx.substrate(state.substrate());
```

`DocState::substrate()` reads both `substrate` and `substrate_scale`
(`document/state.rs:275`). Both are written by actions declaring
`Resource::Substrate` (`footprint.rs:415`). **No action anywhere reads that
resource** — `grep -rn 'Resource::Substrate'` finds only writers, the patch arm and
the tests.

So `footprint(stroke).conflicts(footprint(SetSubstrate))` is `false`, and the two are
judged to commute. They do not: the tooth gates the deposit by the substrate and its
scale, which is the whole point of §6.4 and the reason the substrate became content
that a replay waits for (§12.4).

### What it costs

`ReplicatedTimeline::resync` (`document/timeline.rs:747`) takes the commuting-splice
path when exactly one materialized action leaves the sequence. Undo a `SetSubstrate`
in a shared session and every stroke after it is spliced past rather than re-rendered
— those tiles keep the tooth of a substrate the log no longer contains. A canonical
replay of the same log renders them differently.

`resync`'s own doc states the premise that fails here:

> Convergence is untouched by the fast paths: disjoint footprints mean the shifted
> materialization computes the *same pixels* the canonical replay would, because
> every `apply` reads only what its footprint declares.

And per §12.6, pixels cannot show which materialization ran — that agreement *is* the
claim. This is the silent-divergence class, not the false-conflict class.

### Coverage today

- `stark-engine/tests/footprint.rs` — checks **writes** only, by state diff.
- `stark-engine/tests/commute.rs` — five hand-written scenarios, none involving the
  substrate (`grep -c Substrate` → 0).
- `docs/collaboration.md:212` lists "the substrate" among the footprint's resources,
  which is true of the vocabulary and false of any action's `reads`.

### Fix

```rust
ActionKind::CommitStroke(rec) => Footprint {
    reads: vec![
        Resource::Existence(rec.layer),
        Resource::Selection(actor),
        // The tooth gates the deposit by the substrate and the scale it is laid
        // at (§6.4), both read off the state being folded over — so a stroke does
        // not commute with the substrate changing under it.
        Resource::Substrate,
    ],
    writes: vec![Resource::Paint(rec.layer, stroke_rect(rec))],
},
```

Add a scenario to `commute.rs` alongside `undo_splices_past_a_rename`: a
`SetSubstrate`, strokes after it, an undo of the substrate — and assert the strokes
re-render (`stats.rebuilds` moved, and the pixels match a canonical replay).

Cost of the fix: a stroke no longer commutes with a substrate change, which is a
false conflict for nobody, because it is a true one.

## B. The `reads` half of every footprint is unverified

`stark-engine/tests/footprint.rs` proves footprint honesty over every `ActionKind`
structurally — but only for writes. Its header concedes the other half:

> And `reads` are not checked: a read is not observable in a state diff, and
> catching an undeclared one needs a different instrument.

An under-declared read is exactly as divergent as an under-declared write:
`Footprint::conflicts` tests `writes×reads` in **both** directions, so a missing read
makes a pair commute that does not. Half of §12.6 is currently prose.

### The instrument exists: read-independence

For each kind, apply the action to two states that differ **only outside**
`reads ∪ writes`, and assert the resulting write-diff is identical. Anything the
action secretly depends on shows up as a difference; anything it legitimately reads
is excluded by construction.

Sketch, driven off the same `stark_testdata::vocabulary` roster the writes test uses,
so it is exhaustive by the same device:

```text
for each kind:
    fp = footprint(action)
    for each resource r not overlapped by fp.reads ∪ fp.writes:
        base_a = a state
        base_b = base_a with r perturbed
        assert diff(base_a, apply(action, base_a)) == diff(base_b, apply(action, base_b))
```

The perturbations are the same vocabulary the writes test already builds diffs in:
a substrate, a substrate scale, a substrate color, another layer's paint, another
actor's selection, a property on an unrelated layer, the guide roster.

**This would have caught [A](#a-commitstroke-under-declares-its-footprint) on its
first run**, and it is the only thing on this list that finds the *next* one. Highest
value item in the ledger.

## C. Lasso decimation overflows `usize` on wasm32

`crates/stark-model/src/document/selection.rs:148`

```rust
Self::Lasso(points) if points.len() > MAX_LASSO_POINTS => Self::Lasso(
    (0..MAX_LASSO_POINTS)
        .map(|i| points[i * points.len() / MAX_LASSO_POINTS])
        .collect(),
),
```

`i` reaches 4095. On `wasm32-unknown-unknown` `usize` is 32 bits, so
`i * points.len()` overflows once `points.len() > 1_048_913` — about 8.4 MB of
`Vec2`, and deflate's ratio means a small file names it (which is the entire premise
of `MAX_DECOMPRESSED`). Both doors reach here: this is `SelectionOp::at`, the
deserialization funnel, so it runs on every op arriving from a file or a peer.

Measured, emulating 32-bit `usize` at `len = 1_100_000`:

```text
i=    0  64-bit=        0  32-bit=        0  same
i= 1000  64-bit=   268554  32-bit=   268554  same
i= 3000  64-bit=   805664  32-bit=   805664  same
i= 4095  64-bit=  1099731  32-bit=    51155  DIVERGES
```

- **Debug wasm**: panic, "attempt to multiply with overflow".
- **Release wasm**: wraps, and the tail of the loop reads vertices from the wrong
  place — a *different polygon* than a native peer decodes from the same bytes.

Two clients rasterizing different masks from one op is the one thing §6.8 says may
not happen, and the browser build is on the wrong side of it.

### Fix

```rust
.map(|i| points[(i as u64 * points.len() as u64 / MAX_LASSO_POINTS as u64) as usize])
```

`crates/stark-model/src/gradient.rs:218` (`thin`) has the same shape and is safe only
by size accident — `15 × len` needs 286M stops, which is 4.6 GB and unreachable.
Widen it anyway, for the reason `TileRect::covering` is `i64` and saturating
throughout: the arithmetic should not depend on a bound held somewhere else.

## D. `MAX_LASSO_POINTS` states a coupling nothing can check

Its doc (`document/selection.rs:44`) says:

> The edge list is uploaded as an `N×1` texture (`gpu::selection::edge_texture`), so
> `N` has to stay inside the smallest `maxTextureDimension1D` any WebGPU adapter
> guarantees (8192) or the op fails validation instead of rasterizing.

`stark-engine/src/gpu/selection.rs:536` sizes that texture straight from
`edges.len()` with no assertion, and the constant is **not re-exported** from
`document` (`selection` is `pub(crate)`, and `document.rs`'s re-export list omits
it) — so `gpu/selection.rs:37` can only name it in prose.

The workspace already has the right pattern one file over,
`stark-engine/src/gpu/fill.rs:79`:

```rust
stark_shaders::mirror::fill::MAX_GRADIENT_STOPS as usize == stark_model::gradient::MAX_STOPS,
```

### Fix

Re-export `MAX_LASSO_POINTS` from `document`, and const-assert it against 8192 beside
`edge_texture`.

## E. A placement loses the exactness it promises

`crates/stark-model/src/document/image.rs:33`

```rust
let lo = Vec2::new(at.x as f32, at.y as f32);
```

`PlaceImage`'s doc makes a promise about resampling:

> **`at` is in whole canvas pixels, and that is a promise about resampling**: the
> image's texels land on canvas pixels one for one, so nothing is filtered and there
> is no sampling loss between the file and the tiles.

Past `|at| > 2^24` the `f32` rounds and the promise breaks silently. `covering` only
refuses past ~5×10¹¹ px, so there is a wide band of placements that are accepted and
wrong. Either bound `at` to what an `f32` addresses exactly and refuse past it (the
stance §16.1 takes for a transform), or quantize the box in integers before it
becomes a `Vec2`.

Low severity — nobody places a picture 16 million pixels from the origin today — but
it is a stated promise with nothing holding it.

---

# 2. Architecture

## F. The `ActionKind` roster tax

A new variant must be visited in about ten places:

| Where | What it answers |
|---|---|
| `document/action.rs` | the variant itself |
| `ActionKind::minted_layers` | which ids it mints (§17.9) |
| `ActionKind::sanitized` | which of its numbers are clamped |
| `ActionKind::label` | its caption |
| `document/footprint.rs::footprint` | what it reads and writes |
| `content.rs::action_content` | what content it names |
| `stark-testdata::vocabulary::{slot, LABELS, KINDS}` | its place in the roster |
| `tests/action_kinds.rs::kinds` | one of it, with numbers |
| `tests/action_kinds.rs::gated_at_apply` | clamped or gated |
| `stark-engine` `apply` / `patch` | what it does, and how to undo it |

The crate handles this about as well as Rust allows — every one is exhaustive with no
`_` arm, and `vocabulary`'s header records exactly what it cost the day two copies of
the roster drifted. But it is still the dominant per-feature cost, and four of those
ten are the *same list written four times*.

### The cheap consolidation

A plain payload-free `ActionTag` enum in the model, with one exhaustive
`ActionKind::tag(&self) -> ActionTag` and a `const ALL: [ActionTag; N]`:

- `label` moves onto `ActionTag` (it never reads a payload today).
- `vocabulary::slot` becomes `tag as usize`.
- `vocabulary::LABELS` and `KINDS` disappear — `ActionTag::ALL` and its length are
  the roster.
- `stark-testdata::vocabulary` shrinks to the one-of-each fixture, which is the part
  that genuinely belongs in a fixtures crate.

Four hand-maintained lists across two crates collapse to one exhaustive match plus
one const array, with no proc macro and no new dependency. The chain
`vocabulary`'s header describes ("no link can be left half-done") survives intact:
a new variant still fails to compile at `tag`, and `ALL`'s declared length is still
what the one-of-each array is checked against.

A derive macro carrying `#[action(label = "…", mints = id, content = image)]` would
subsume `minted_layers` and `action_content` as well, but that is a lot of machinery
for 34 variants and it would hide the arguments the current match arms carry in
comments. Not recommended yet.

## G. `Parcel` and `MattePaint` are the same type twice

`document/layer.rs:544` and `document/fill.rs:107` are word-for-word identical, and
`MattePaint::sanitized`'s own comment says so:

> Word for word [`Parcel::sanitized`], because this is word for word a [`Parcel`]:
> both are "what paint to lay", both are a solid or a ramp on an axis, and both reach
> the same `ramp_common::ramp_position` through different passes. They are two types
> only because they reached the file at different times and their wire shapes were
> written differently (`Gradient { gradient, axis }` against
> `Gradient(GradientParcel)`); **merging them is a save-format change and is worth
> doing on its own**, not as a side effect of closing this gap.

There are no stability guarantees yet. This is the window, and it only gets more
expensive: every document saved between now and the beta rung is one the merge has to
carry an alias for.

Merge into `Parcel`, keep `MattePaint` as a `#[serde(alias)]`-bearing name if the
existing files are worth reading, and delete the duplicate sanitizer, the duplicate
`swatch`, and the paragraph above with them.

## H. The three content bags should be one

`io.rs`'s `DocumentFile` carries three parallel bags:

```rust
pub assets:     Vec<(AssetId, Vec<u8>)>,      // brush shapes
pub substrates: Vec<(SubstrateId, Vec<u8>)>,  // height maps — keyed differently
pub pictures:   Vec<(AssetId, Vec<u8>)>,      // placed images
```

`content.rs::unbundled_content` then builds three `HashSet`s and matches three arms to
enforce the property its own test is named after — *"a need is answered by its own
bag, never by the other one"* — because an `AssetId` is a **content** hash, so one
image imported as a stamp and placed as a picture carries one id in two bags that
cannot stand in for each other.

That property is currently held by a test. `AssetNeed` **already is** "the id, plus
which store it belongs in". One bag keyed by it —

```rust
pub content: Vec<(AssetNeed, Vec<u8>)>,
```

— makes the wrong answer unrepresentable, and deletes the three-way match, the three
`HashSet`s, and `a_need_is_not_answered_by_another_bag` along with it. That is
CLAUDE.md's *"rule out a class rather than enumerate its instances"* applied to the
one place in the crate that still enumerates.

Needs `AssetNeed` to gain `Serialize`/`Deserialize`/`carbonite::Schema` (it is a plain
three-variant enum over `AssetId`, so this is free), and it is a format change —
which is the same window [G](#g-parcel-and-mattepaint-are-the-same-type-twice) wants.
Do them together.

## I. `FillOp` and `SelectionOp` defeat their own funnels

Both types go to real trouble to make the constructor the only door — a `Raw*` mirror
type, `#[serde(from/into)]`, `#[carbonite(as)]`, and a doc paragraph apiece arguing
that *"the funnel is worth nothing if there is a second door"*.

Then every field is `pub`, so the second door is `FillOp { opacity: 5.0, .. }` and it
compiles anywhere in the workspace.

That is why `ActionKind::sanitized` has to re-run both constructors —
`document/action.rs:640` and `:648`:

```rust
ActionKind::Fill { layer, op } => ActionKind::Fill {
    layer,
    op: FillOp::with_paint(op.shape, op.feather, op.paint, op.opacity),
},
ActionKind::Select(op) => {
    ActionKind::Select(SelectionOp::at(op.mode, op.shape, op.feather, op.opacity))
}
```

Compare `gradient.rs:76`, which does it right: `stops` is private, `new` is the only
door, and consequently there is *nothing left* for `Filter::sanitized`'s gradient arm
to do — its comment is a small celebration of exactly that:

> It has been three things on this branch — a loop spelling the bound `f32::clamp`
> (which returns the NaN it exists to catch), then a call to `Gradient::clamped`, and
> now nothing at all. That is the newtype's whole argument in one arm.

### Fix

Make the fields private with accessors. Verified: outside their own `mod tests`,
nothing in the workspace constructs either type by struct literal
(`grep -rn 'SelectionOp {' --include=*.rs crates/` → one hit, in `action.rs`'s own
test). The two `sanitized` arms above then join the "nothing to hold" list, and the
invariant stops being something two gates re-establish.

## J. `document/guide.rs` is 2000 lines of two different things

§20.5's argument for keeping guides unsplit is sound — everything derived from the
camera is a pure function of it, so the derivations belong beside the fact for the
reason `fill_bounds` and the homography solve do.

But the file is the largest in the crate by 15%, and it is two modules:

- **the document fact** (~450 lines): `PerspectiveGuide`, `GuideId`, `Lens`,
  `sanitized`, `dragged`.
- **projective geometry that knows nothing about documents** (~1500 lines):
  `conic_of`, `ellipse_of`, `congruent`, `AxisPlane::chart`, `circle_seen`,
  `circle_behind`, `pair_trace`, `horizons`, `pencils`, `planes`, `scene`,
  `axis_turn`.

A `guide/` directory — `mod.rs` for the fact, `camera.rs` and `conic.rs` for the
derivations — keeps §20.5's claim exactly intact (nothing crosses a crate boundary,
nothing becomes public that was not) while making the file navigable. `document.rs`'s
re-export list does not change.

## K. The crate's NaN policy has five private re-implementations

`lib.rs:83` introduces `clamp01` / `finite_or` / `at_least_zero` under a heading that
is explicit about why they are there:

> **the crate's NaN policy**, in one place because it is one policy.
> [...] One definition, so the policy cannot be half-remembered at the next gate.

Then:

| Site | What it spells |
|---|---|
| `document/filter.rs:269` | a local `clamp` closure |
| `document/filter.rs:353` | the **same** closure, byte for byte |
| `document/guide.rs:362` | a local `fn or` that *is* `finite_or` |
| `document/layer.rs:393` | `if k.is_finite() { k.clamp(..) } else { DRAGO_K }` inline |
| `document/action.rs:695` | the same shape inline, for `SetLayerOpacity` |

The missing helper is the one four of them want:

```rust
/// `x` if it is a number this parameter can be, else `neutral` — [`finite_or`]'s
/// companion for a knob with a range at both ends.
pub(crate) fn finite_in(x: f32, neutral: f32, (lo, hi): (f32, f32)) -> f32 {
    if x.is_finite() { x.clamp(lo, hi) } else { neutral }
}
```

Add it beside the other three, route the five sites through it, and `lib.rs`'s claim
becomes true. (`filter.rs`'s two are the clearest win — they are the same six lines
twice in one file.)

---

# 3. Performance

## L. `timeline::resync` recomputes footprints for statistics

`stark-engine/src/document/timeline.rs:753`

```rust
let commuting = {
    let mut suffix = self.history.actions().skip(diverge);
    let fp = footprint(suffix.next().expect("diverge < mat.len()"));
    suffix.take_while(|a| !fp.conflicts(&footprint(a))).count()
};
```

`History::actions()` yields `&Logged<DocState>`, which **already carries a cached
footprint** — the entire reason `Logged` holds one. From `document/fold.rs:86`:

> `history` builds a centralizer once per removal and then asks it about *each* later
> action, so `Centralizer::commutes` used to rebuild the other action's footprint on
> every comparison — two `Vec` allocations always, a walk of the whole control-point
> list for a stroke, and for a `TransformWarp` an entire fine-lattice solve
> (`WarpMap::image_aabb`, 57×57 nodes at an 8×8 grid). An undo across a warp was
> quadratic in the log for an answer that cannot change.

`Logged` derefs to `Action`, so `footprint(a)` silently resolves to the free function
and re-does all of it — over the whole commuting suffix, **purely to update a
counter**. A `TransformWarp` in that suffix costs a 57×57 lattice solve per undo.

### Fix

```rust
let fp = suffix.next().expect("diverge < mat.len()").footprint();
suffix.take_while(|a| !fp.conflicts(a.footprint())).count()
```

### The model-side lesson

After the caching landed, `pub fn footprint(&Action)` is still the ergonomic path and
`Deref` routes to it silently. Rename the free function `compute_footprint`, or make
`Logged::footprint()` the only reachable one, so the next call site cannot make the
same choice invisibly.

## M. `Prepared::eval` still allocates in the loop it was made for

`document/warp.rs:420` builds a `Vec<Vec2>` per call:

```rust
let row_dev: Vec<Vec2> = (0..rows)
    .map(|j| hermite_axis(&self.deltas[j * cols..(j + 1) * cols], kx, fx))
    .collect();
```

`Prepared` exists precisely to hoist an allocation out of this loop —
`WarpMap::prepared`'s doc:

> **The frontend is that caller, in a loop.** Finding the grabbed point is a search
> over the substrate and drawing the mesh is one `eval` per point of every curve, so
> a drag was rebuilding the whole delta grid — a `Vec` allocation and `cols · rows`
> subtractions — per sample, tens to hundreds of times a frame.

`MAX_WARP_GRID` is 8, so `rows ≤ 9`: this is a `[Vec2; MAX_WARP_GRID as usize + 1]`
plus a length, dependency-free, with no behaviour change (§16.4's identity invariant
is stated bitwise, so the arithmetic order must not move — an array write in the same
order is fine).

Same for `Lattice::basis` and `axis_basis` (`warp.rs:109`, `:383`), both bounded at 9,
which together allocate three `Vec<f32>` per evaluation on the exact-follow drag path.

## N. ~~Withdrawn~~: `Footprint`'s two `Vec`s per action

**This finding was wrong, and the codebase had already answered it.**

The proposal was to replace `Footprint`'s two `Vec<Resource>` with a single
`Box<[Resource]>` and a split index, halving the allocations. It turns out
`stark-model/tests/action_kinds.rs::a_footprint_stays_small_enough_for_a_nested_scan`
exists to settle exactly this, and its doc says so:

> It also settles whether the two `Vec`s should be inline storage: at these lengths
> the allocations are two per commit, amortized against the GPU work a commit already
> does, and the scan is over nine elements. Inline storage would trade ~130 bytes per
> logged action for that, and buy a dependency. **Measured, and not worth it —
> recorded here so the question is not re-opened from intuition.**

Which is what this finding did: re-opened it from intuition, having read the test's
assertions and not its argument. The test is the right shape and the answer stands.

The one thing that did change is the number: `MAX_READS` moved from 2 to 3 when
[A](#a-commitstroke-under-declares-its-footprint) gave a stroke its substrate read.
The claim about storage is untouched.

---

# 4. Code health

## O. 23 rustdoc warnings, and no CI gate

`.github/workflows/ci.yml` runs fmt, clippy `-D warnings`, nextest, doctests and the
wasm build. It does not run `cargo doc`.

In a crate where the doc comments *are* the design record — §-numbers cited from
~1000 places, arguments that exist nowhere else — an unchecked doc graph is a real
correctness surface. `cargo doc -p stark-model --no-deps` reports 23 warnings, 15 of
them broken links. What has already rotted:

### `PlaceImage` documents a design that was reversed

`document/action.rs:438`:

> The pixels are the payload, behind an [`Arc`] and PNG-encoded on the wire — see
> [`ImageRef`](super::ImageRef) for both, and for why this is the one action that
> carries content rather than naming it.

`ImageRef` does not exist. The variant carries `image: crate::AssetId`. And
`docs/images.md:49` spends four paragraphs explaining why carrying it by value **was
wrong** and was undone — it did not fit a gossip message, it did not deduplicate, and
a joining peer could not skip it.

The same stale claim survives in **CLAUDE.md's own doc table**: *"images.md | §23 |
… the one action that carries its content by value"*. Both need the correction.

### `BrushParams::radius` → `size` left the field docs behind

`document/brush.rs:1041` still reads *"Stamp radius in canvas pixels at full
pressure"*, and five sibling field docs link to `Self::radius`, which no longer
exists (`start_taper_length`, `end_taper_length`, `drain`, `taper_px`, `drain_px`).
`docs/brush.md:1272` is the current statement:

> **A brush's `size` names the disc the mark fits in — for every shape.**

On the crate's most-read type.

### The rest

| Count | Warning |
|---|---|
| 5 | unresolved link to `Self::radius` |
| 3 | unresolved link to `Self::add` |
| 3 | `max_slope` links to private `mod_slope` |
| 1 each | `super::ImageRef`, `crate::document::BrushParams::color`, `Tool`, `SelectionOp`, `LayerId`, `Centralizer::commutes`, `ActionKind::sanitized` |
| 1 each | `to_plane` → private `PLANE_REACH`, `new` → private `crate::clamp01`, `from_untrusted_bytes` → private `MAX_DECOMPRESSED`, `corner` → private `LATTICE_EPS` |
| 1 | redundant explicit link target (`color.rs:74`) |

### Fix

Add to CI, after the clippy step:

```yaml
- name: Docs
  env:
    RUSTDOCFLAGS: -D warnings
  run: cargo doc --workspace --no-deps
```

This is a ratchet in exactly the spirit of the workspace lint table's own bar — fix
the 23 sites once and it can never fail this build again, only a later one. The
private-item links are a judgement call (`--document-private-items` is the other
answer); the 15 broken ones are not.
