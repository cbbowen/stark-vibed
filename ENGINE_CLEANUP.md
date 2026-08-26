# `stark-engine` cleanup ledger

A critical review of `crates/stark-engine` (~50.6k source lines, ~25k test lines)
recorded 2026-08-25 against `47fc89b`, the day `MODEL_CLEANUP.md` was settled and
removed. Same terms as that ledger: §19's beta rung is unclaimed, so the save format
and the wire may change, and no finding needs a bit-identical result — a wrong model
is fixed and the goldens re-blessed.

**Status: in progress.** The table carries the commit that closed each finding; a
finding that turns out to be wrong is struck through and kept, as
[N](#n-withdrawn-footprints-two-vecs-per-action) was in the model's ledger.

Two things the work has turned up that are not in any row below, recorded here
because they are what the findings were *for*:

- **A second §12.6 break**, found by the guard [A](#a-a-groups-removelayer-under-declares-its-subtree)
  asked for rather than by reading: `Resource::Substrate` names the substrate *and*
  the scale it is laid at, and `PatchOp::Substrate` carried only the id — so undoing
  a `SetSubstrateScale` through the commuting splice left the scale where the undone
  action had set it. Closed with A in `6313c00`.
- **A build-cache hazard on this machine**, twice: a `rustc`
  `STATUS_STACK_BUFFER_OVERRUN` poisons `target/debug/incremental`, and every unit
  after it fails with "only metadata stub found for `std`" or "can't find crate for
  `stark_engine`". It is not a code failure and no amount of reading the diff finds
  it — `rm -rf target/debug/incremental` and re-run.
- **What the golden hatch was hiding**, measured rather than guessed: six goldens
  were stale, five by ordinary drift and one — `transform_perspective_warp` — by a
  contiguous 22×51 strip at up to 24 levels, which is an edge that moved. Rendering
  it at each commit since it was blessed puts the whole of that change in the
  selection-opacity rework of 2026-08-25, and clears `f6fe856`'s bit-identity claim
  (zero difference across it). Closed with C in `31a9993`.

How it was read: the core — `engine.rs` and its children, `document/state.rs`,
`document/timeline.rs`, `document/apply.rs`, `gpu/tile.rs`, `gpu/submit.rs`,
`command.rs`, the model's `Footprint`/`Materialize` contract, the `history` crate's
cache — by hand; the rest (the stroke path, compositing and tile operations, the
fold and collaboration, input/geometry/services, the test suite) by five parallel
readers whose findings were then re-read in the source. A finding marked
**verified** was re-read that way; one marked *reported* carries its reader's quoted
evidence and was not independently confirmed. Nothing was edited and no test was
run.

## What it turned out to be about

The architecture is sound and the one idea everything follows from is carried
through faithfully: a total fold that refuses deterministically, footprints driving
both commutation and undo, persistent maps of copy-on-write tiles, one renderer used
three ways, a generated shader ABI with no hand-transcribed constant left that a
reader could find. The core is not the problem.

The debt is at the seams, and three things stand out:

- **One real §12.6 hole**, the same shape as the model ledger's
  [A](#a-a-groups-removelayer-under-declares-its-subtree): removing a *group*
  declares the parent and `StackOrder`, and `Resource::overlaps` says an edit inside
  the group commutes with it. The fast-path undo then restores the pre-edit subtree
  while a canonical replay keeps the edit — peers diverge and no pixel says which
  path ran. It survived because `tests/footprint.rs` exempts exactly this case.
- **Three panics reachable from outside the process**, each a tab abort on the web
  with the painting unsaved: a peer's document in a colour space this build lacks, a
  malformed HDR, and an sRGB target format that nothing refuses.
- **The two strongest guards in the suite each have an escape hatch**: the golden
  comparator tolerates 1% of pixels at any magnitude — a tip-sized disc, which is
  the size of every artifact the suite has fought — and one test returns a passing
  pair when there is no GPU at all.

Below that is ordinary host-side duplication (one render loop written three times,
one bind-group match in two files, fifteen hand-written pass descriptors beside the
module whose charter is to end them) and a 39% comment ratio of which a measured
share narrates history, in a codebase whose own rule says comments describe the
present.

## The order to spend it in

| # | Finding | Kind | Closed by |
|---|---|---|---|
| [A](#a-a-groups-removelayer-under-declares-its-subtree) | A group's `RemoveLayer` under-declares its subtree | **correctness** | `6313c00` |
| [B](#b-three-panics-reachable-from-outside-the-process) | Three panics reachable from outside the process | **correctness** | `ae27cb5` |
| [C](#c-the-golden-comparator-and-a-test-that-passes-without-a-gpu) | The golden comparator and a test that passes without a GPU | **correctness** | `31a9993` |
| [D](#d-footprint--apply-correspondence-is-held-by-discipline) | Footprint ↔ apply correspondence is held by discipline | **correctness** | `3974f8f` |
| [E](#e-layerid-is-a-32-bit-fold-of-the-actor) | `LayerId` is a 32-bit fold of the actor | correctness | |
| [F](#f-three-places-where-the-code-and-the-claim-beside-it-disagree) | Three places where the code and the claim beside it disagree | correctness | F2 in `e8d2e78`+ |
| [G](#g-the-compositors-generation-conflates-two-invalidations) | The compositor's `generation` conflates two invalidations | **performance** | `a9edfbf` |
| [H](#h-the-stroke-hot-path-does-not-use-the-plumbing-built-for-it) | The stroke hot path does not use the plumbing built for it | **performance** | |
| [I](#i-the-eyedropper-submits-once-per-sample-point) | The eyedropper submits once per sample point | performance | |
| [J](#j-the-mixbox-lut-runs-twice-per-colour-per-texel-on-a-placed-image) | The Mixbox LUT runs twice per colour, per texel on a placed image | performance | `a9edfbf` |
| [K](#k-the-log-is-cloned-whole-on-every-save) | The log is cloned whole on every save | performance | |
| [L](#l-smaller-measurable-costs) | Smaller measurable costs | performance | `078c4ba` (3 of 14) |
| [M](#m-engine-is-one-type-with-25-fields-and-65k-lines-of-impl) | `Engine` is one type with ~25 fields and ~6.5k lines of `impl` | architecture | |
| [N](#n-boxdyn-timeline-is-a-two-mode-enum-with-silent-defaults) | `Box<dyn Timeline>` is a two-mode enum with silent defaults | architecture | |
| [O](#o-the-accumulated-extent-render-loop-is-written-three-times) | The accumulated-extent render loop is written three times | maintainability | |
| [P](#p-planslot-is-a-hand-maintained-twin-of-the-generated-stamp) | `plan::Slot` is a hand-maintained twin of the generated `Stamp` | maintainability | |
| [Q](#q-the-descriptor-boilerplate-descrs-was-written-to-end) | The descriptor boilerplate `desc.rs` was written to end | maintainability | |
| [R](#r-two-scratch-pools-and-two-submit-scopes-for-one-need) | Two scratch pools and two submit scopes for one need | maintainability | |
| [S](#s-four-copies-of-the-paint-edit-gate-and-uneven-minted-layer-claims) | Four copies of the paint-edit gate, and uneven minted-layer claims | maintainability | `2c6303b` |
| [T](#t-files-and-apis-that-carry-more-than-they-should) | Files and APIs that carry more than they should | maintainability | |
| [U](#u-comments-that-narrate-history-or-describe-code-that-is-gone) | Comments that narrate history, or describe code that is gone | maintainability | `dbf4dad` |
| [V](#v-footprint-reads-are-checked-over-a-hand-picked-vocabulary) | Footprint *reads* are checked over a hand-picked vocabulary | tests | `6313c00`, `3974f8f` |
| [W](#w-64-translation-invariance-is-guarded-for-strokes-only) | §6.4 translation invariance is guarded for strokes only | tests | `PENDING` |
| [X](#x-what-the-suite-observes-only-through-the-lit-composite) | What the suite observes only through the lit composite | tests | |
| [Y](#y-suite-infrastructure) | Suite infrastructure | tests | |

## What is left open on purpose

- **The nightly toolchain** ([T](#t-files-and-apis-that-carry-more-than-they-should)):
  the fix is upstream in `history`, not here.
- **F1 and F3** ([F](#f-three-places-where-the-code-and-the-claim-beside-it-disagree)):
  the flattening determinism story is a decision about what Stark promises across
  platforms, and `mean_error`'s map cannot be changed without re-tuning `KNOT_COST`.
  Both want a sitting rather than a patch.

---

# 1. Correctness

## A. A group's `RemoveLayer` under-declares its subtree

**Verified.** The §12.6 failure CLAUDE.md's first rule warns about, in the shape
the model ledger's own headline finding took: an action touches state its footprint
does not name, and the commutation fast path silently diverges.

`crates/stark-model/src/document/footprint.rs:398`

```rust
ActionKind::RemoveLayer(id) => Footprint {
    reads: Vec::new(),
    writes: vec![
        Resource::Existence(*id),
        Resource::StackOrder,
        Resource::Paint(*id, TileRect::ALL),
    ],
},
```

`crates/stark-engine/src/document/state.rs:802` — `remove_layer` → `remove_in`
takes the **whole subtree**: every carried layer's existence, paint and properties
are written, none declared. The comment above the footprint argues `StackOrder`
covers them. It covers *structural* edits only:

`crates/stark-model/src/document/footprint.rs:149`

```rust
fn overlaps(&self, other: &Resource) -> bool {
    match (self, other) {
        (Resource::Paint(a, ra), Resource::Paint(b, rb)) => a == b && ra.intersects(rb),
        (Resource::Layer(id), other) | (other, Resource::Layer(id)) => {
            other.layer() == Some(*id)
        }
        _ => self == other,
    }
}
```

`StackOrder` vs `Existence(child)` falls to `self == other` → false. `Paint(G, ALL)`
vs `Paint(C, r)` fails `a == b` → false. So `SetLayerOpacity(C, _)` (reads
`Existence(C)`, writes `Prop(C, Opacity)`) and `CommitStroke` on `C` are both judged
to **commute** with `RemoveLayer(G)`.

### What it costs

Log: `AddLayer G`, `AddLayer C carrier=G`, A: `RemoveLayer(G)`, B concurrently:
`SetLayerOpacity(C, 0.5)` (ordered after), A: `Undo(Remove G)`.

- A fresh joiner materializes `[Add G, Add C, SetOpacity(C, 0.5)]` → C at 0.5.
- A's own timeline takes the one-action-removed arm (`document/timeline.rs:745`) →
  `History::remove_action_with` → `shift_late` → `inverse`, which restores
  `PatchOp::Present` with the **old** subtree record (`document/patch.rs:212`) → C at
  1.0. The shifted cached state is then kept as the replay base for later versions.

With a stroke on C instead of an opacity, the stroke is lost on one side. Pixels
cannot show which materialization ran — that is the whole of what §12.6 promises.

### Coverage today

- `crates/stark-engine/tests/footprint.rs:207` exempts exactly this:

  ```rust
  Diff::Named(r @ Resource::Existence(_)) => {
      writes.contains(r) || writes.contains(&Resource::StackOrder)
  }
  ```

  and its `differences` reports a vanished carried layer only as `Existence`, never
  as the tile and prop diffs underneath.
- `tests/commute_pairs.rs:279` removes `LayerId(1)`, a root-level leaf. No row
  removes a group; no row edits inside one.

### Fix

Apply the `DuplicateLayer` precedent ("a footprint is built from the action alone
and cannot go looking"): the action names what it removes —
`RemoveLayer { id, #[serde(default)] carried: Vec<LayerId> }` — declares
`Resource::Layer(x)` as a write for `id` and every carried id, and `apply` declines
when the subtree's id set differs from the named set (a concurrent add into the
group), exactly as `copy_subtree` declines. Bump the ALPN: a meaning change. Delete
the `StackOrder` exemption in `tests/footprint.rs::covered` so `Existence` must be
named exactly; add "remove a group" and "edit inside a group" rows to
`commute_pairs`. Write the general rule into `footprint.rs`'s header: *an action
whose effect spans layers it does not name must carry their ids.* A legacy file with
`carried = []` on a group declines the removal; that is a pixel change, and it is
the honest one — `Option<Vec<_>>` with `None` meaning "remove whatever is there"
keeps the hole open for exactly those actions.

## B. Three panics reachable from outside the process

**Verified.** Each is an `expect` or an unchecked convention on a path a peer, a
file or an embedder can drive; each is an abort on wasm with the document unsaved —
the class `EngineError::Gpu`'s doc and `GpuHealth` exist to rule out.

### B1. A joiner in a colour space this build lacks

`crates/stark-engine/src/engine/collab.rs:100`

```rust
pub fn join_collaboration(&mut self, file: &DocumentFile, identity: impl Into<Identity>) {
    ...
    self.adopt(file);
```

`crates/stark-engine/src/engine/file.rs:133`

```rust
pub(super) fn adopt(&mut self, file: &DocumentFile) {
    ...
    if file.canvas.color_space != self.shared.color_space.id() {
        // A `DocumentFile` reaches here from exactly two places, and both have
        // already settled this: ...
        let cs = crate::colorspace::make(file.canvas.color_space)
            .expect("a document whose space this build lacks is refused before adoption");
```

`require_color_space` is reached only from `load_bytes` (`file.rs:219`); the join
path decodes the file off the wire in `stark-net` and hands it straight through
(`stark-ui/src/collab.rs:161`). A build without `mixbox` — the second configuration
CLAUDE.md says a commercial build ships — joining a Mixbox session aborts.

**Fix.** Rule out the class: `adopt` takes a `ValidatedFile<'_>` newtype that only
`require_color_space` can mint (or returns `Result`, and `join_collaboration`
propagates it to `stark-ui`'s `fail`).

### B2. A malformed HDR

`crates/stark-engine/src/gpu/environment.rs:105`

```rust
pub fn load(ctx: &GpuContext, hdr_bytes: &[u8], exposure: f32) -> Self {
    let (px, w, h) = decode_hdr(hdr_bytes).expect("environment: decode HDR");
```

reached from `register_environment` (`file.rs:564`) → `Registry::register` →
`Resource::build`, with nothing between the frontend's fetch and the `expect`.
`environment/hdr.rs:1` states the intent — "a malformed file must be an error
rather than a panic" — and its only caller unwraps it. Contrast `accept_substrate`
(`file.rs:495`), which runs `identify` *before* `register`, which is what makes
`SubstrateMap::load`'s `expect` a genuine invariant.

**Fix.** `register_environment` decodes at the boundary and returns `Result`;
store the decoded equirect beside the bytes so `build` never decodes.

### B3. An sRGB target format

The media pass hand-encodes sRGB (`media_common.wesl::finish`) and the resolve
decodes and re-encodes; on an sRGB-suffixed target the hardware does both again. The
rule "the target is a non-sRGB format" is enforced in exactly one place,
`stark-ui/src/render.rs:951`, and contradicted in two: `CompositorPipeline::new`
(`gpu/composite.rs:373`) accepts any format, `media.rs::dither_step` lists the sRGB
formats as valid, and `tests/stroke.rs:647` builds an engine on `Rgba8UnormSrgb`. The
wrong path is a gamma-squared image with no error.

**Fix.** `assert!(!target_format.is_srgb())` in `CompositorPipeline::new`; drop the
two sRGB arms from `dither_step`; `tests/stroke.rs:647` → `Rgba8Unorm`.

## C. The golden comparator, and a test that passes without a GPU

**Verified.**

`crates/stark-engine/tests/common/mod.rs:411`

```rust
if d as u8 > tol { bad += 1; }
...
let frac = bad as f64 / total as f64;
if frac > 0.01 { ... panic!(...) }
```

A pixel counts as bad only if it exceeds `tol`, and the panic fires only if more
than 1% do. On a 256² golden that is ~655 pixels free to differ by **any** amount —
a disc of radius ~14 px, which is the size of the lift-end ring, the settle crease
and the stranded-glob hole, each of which has a bespoke test in `dynamics.rs`
because a golden did not catch it. All 19 `assert_golden` sites inherit this. The
corpus battery's own comparator (`tests/common/corpus.rs:502`) is worst-texel —
"steps and seams are loud in the maximum and quiet in the average" — so a corpus
case's *golden* is its weakest check.

`crates/stark-engine/tests/stroke.rs:646`

```rust
let Ok(mut engine) = pollster::block_on(stark_engine::engine::headless_engine(..)) else {
    eprintln!("skipping GPU test");
    return (1.0, 1.0);
};
```

The caller asserts `late < early * 2.0` → `1.0 < 2.0` → `ok`, with no GPU and no
`STARK_ALLOW_NO_GPU` — the exact failure CLAUDE.md's "a skipped test still reports
ok" paragraph names, surviving in one place. It is also the suite's only wall-clock
assertion, inside a 16-wide GPU group.

**Fix.** Worst-texel `assert_golden` (or two-tier: `worst <= tol_hi` and
`frac(> tol) <= 0.1%`), printing the worst pixel's location. Route
`measure_per_move_growth` through `common::shared_context()` so the no-GPU contract
holds everywhere — or drop it in favour of the criterion gate, which is where a
wall-clock claim belongs.

## D. Footprint ↔ apply correspondence is held by discipline

Seven exhaustive matches over `ActionKind` (`apply`, `is_noop_on`,
`compute_footprint`, `sanitized`, `minted_layers`, `tag`, `action_content`) give
*presence*, not *correspondence*: nothing says the arm in `apply` touches what the
arm in `compute_footprint` claims. `document/patch.rs:152` says so itself. The two
integration tests that hold the line ([A](#a-a-groups-removelayer-under-declares-its-subtree)
shows how far) check writes structurally and reads over a hand-picked list
([V](#v-footprint-reads-are-checked-over-a-hand-picked-vocabulary)).

**Fix.** Hand the `Logged`'s footprint into `Materialize::fold` — `unfold` already
receives it — and in debug builds diff the state before and after inside `apply`,
asserting every changed tile, prop and existence lies inside the declared writes.
That checks A's class on every fold of every test rather than in one table. Then
make `commute_pairs` roster-complete ([V](#v-footprint-reads-are-checked-over-a-hand-picked-vocabulary)).

## E. `LayerId` is a 32-bit fold of the actor

**Verified mechanism; collision not observed.**

`crates/stark-model/src/document/layer.rs:51`

```rust
pub fn mint(actor: ActorId, n: u64) -> Self {
    let hi = if actor == ActorId::SOLO { 0 } else { mix32(actor.0).max(1) };
    LayerId((u64::from(hi) << 32) | (n & 0xFFFF_FFFF))
}
```

Two actors whose 32-bit folds coincide mint colliding ids — precisely the "two
layers under one id" state §17.9 exists to rule out, hidden behind a hash and said
nowhere. Rare, but the apparatus around it — `Authoring::next_layer`,
`Engine::next_ordinal`, `resync_counters`' layer half, `ActionKind::minted_layers`,
the re-share counter rules in §17.9 — exists to keep unique a counter that a
structural id would make unique by construction. `GuideId` already takes that
answer: `GuideId(action.id)` (`document/apply.rs:479`).

**Fix.** `LayerId = (ActionId, u32 k)`: `AddLayer`/`AddMatte`/`AddFilter`/
`PlaceImage` mint `k = 0`; `DuplicateLayer` mints `k` per subtree position
(deterministic, since it claims `Layer(src)` for each source); `ROOT_LAYER` a
reserved sentinel. Deletes the counter machinery and the collision class; costs a
wider id per `StrokeRecord` (small beside the path) and a format change, which is
free now.

## F. Three places where the code and the claim beside it disagree

### F1. Libm-free flattening, undone by an `atan2` — verified

`crates/stark-engine/src/path.rs:1316` justifies three hand-rolled Maclaurin series
(`sin_small`, `versin_small`, `asin_over_x`) because the cut decisions "have to agree
to the last bit — which the transcendental library functions are not specified to."
The cut decision itself:

`crates/stark-engine/src/path.rs:1698`

```rust
fn turn(a: Vec2, b: Vec2) -> f32 {
    ...
    (a.x * b.y - a.y * b.x).atan2(a.dot(b)).abs()
}
```

`f32::atan2` is the platform libm. The series buy a guarantee the same function
gives away two lines later. The stroke module has the same shape: a polynomial
`taper_profile` "for bit-identity" (`gpu/stroke/segments.rs:321`) beside `ln_keep`
(`budget.rs:390`), `orientation_turns`, `Stretch::solve` and `ToothTable::at`, all
on `ln`/`atan2`/`acos`/`sin_cos`.

**Fix.** Either finish the guarantee — a libm-free cut test
(`cross² > dot² · tan²(angle)`, `tan²` carried in `FlattenTolerance`) and a
cross-platform cut-position test — or delete the series and the claim. And narrow
the stroke module's determinism claim to what it can keep: same machine, live ==
replay == commit.

### F2. `EnvironmentId::exposure` is a dead match the design doc contradicts — verified

`crates/stark-engine/src/gpu/environment.rs:51`

```rust
pub fn exposure(self) -> f32 {
    match self {
        EnvironmentId::Neutral => 1.0,
        _ => 1.0,
    }
}
```

**Fix.** Delete the method and its plumbing and fix the doc. Do not leave a match whose arms agree.

### F3. `mean_error` scores at a different map than its comment says — *reported*

`crates/stark-engine/src/path.rs:843` says "Scored at exactly the parameters the
solve minimizes at. If the two use different maps the growth rule reads one quantity
while the solve improves another". The reader found that `solve` hands `fit_into`
the `ts` of the *pre-solve* profile (line 739) while `mean_error` recomputes
`arc_profile` from the *post-solve* spline (line 851), ignoring `fit.profile`.
`KNOT_COST` was tuned against whatever this actually does, so changing it is a
re-tune. Either fix the comment or use `fit.profile` — which also drops two of the
four curve walks per pointer report.

### Smaller correctness notes

- `engine/pick.rs:322, 569` — a `debug_assert_eq!` guards a readback decode; in
  release a colour space storing colour otherwise mis-decodes silently. Type the
  readback by `TextureFormat`.
- `document/selection.rs:340` — `plan_invert` rasterizes in `HashTrieMap` order
  (rpds uses `RandomState`). Pixels are unaffected — each tile is an independent
  clear-and-draw — but every other planner sorts; sort this one so the invariant is
  uniform.
- `Resource::Existence` silently also means "kind": `cannot_carry` and
  `set_layer_blend`'s filter refusal read a layer's kind without declaring it. Sound
  only because kind is immutable after creation; say so in `Existence`'s doc so a
  future "convert to matte" action knows it needs a resource.
- `document/apply.rs:380` — `PlaceImage`'s fold depends on an out-of-log store: an
  absent picture yields an empty layer on that peer. Documented as a transport
  contract; worth a line in `Resource`'s docs, since it is the one arm whose
  determinism is not a function of the log alone.
- `engine/collab.rs:68` — `start_collaboration` drops the redo stack silently
  (`clone_actions` is `forgotten + history`, never `redo`). Sharing after a
  scrub-back truncates the future with no signal.
- `command.rs:123` — `InputSample::is_finite` is not "bounded": two finite reports
  with `|Δpos| > f32::MAX` reach `spline.rs:427`'s `unreachable!`. Practically
  unreachable; a magnitude gate closes the class for one comparison.

---

# 2. Performance

## G. The compositor's `generation` conflates two invalidations

**Verified.**

`crates/stark-engine/src/gpu/composite.rs:453`

```rust
pub fn set_substrate(&mut self, substrate: SubstrateMap) {
    self.substrate = substrate;
    self.generation = next_generation();
}
pub fn set_environment(&mut self, environment: Environment) {
    self.environment = environment;
    self.generation = next_generation();
}
```

`crates/stark-engine/src/gpu/composite.rs:551` — `ensure_targets` treats any
generation change as "reallocate the accumulator trio, the supersampled target and
the whole blend scratch": up to the 224 MiB `MAX_SUPERSAMPLED_BYTES` budget on a
zoomed-out view. The only GPU object that depends on the substrate or environment is
the media bind group (`composite/media.rs:316`, `Offscreen.bg`).
`engine/file.rs:545` says the opposite of what happens: "no pipeline or pool
rebuild".

### What it costs

Every undo or redo across a logged `SetSubstrate`/`SetSubstrateScale`, every scale
commit, every light switch and every late-arriving HDR destroys and recreates the
largest allocations the app makes — the destroy/create churn `Attachment`'s own doc
(`composite.rs:1038`) warns kills the GPU process on the web at a rate.

### Fix

Key attachments on `(size, ss, Arc::as_ptr(&passes))` — the colour-space rebuild,
the one case that genuinely needs reallocation, is detectable that way. Keep a
separate `bindings` stamp bumped by `set_substrate`/`set_environment`, and give
`media::Offscreen` a `rebind(..)` that rebuilds only `bg`. The scratch levels' bind
groups name neither the substrate nor the environment and need no invalidation on
that path.

## H. The stroke hot path does not use the plumbing built for it

**Verified.** `gpu/uniforms.rs:13` says `gpu::stroke` "had a third copy of the law
as a bare `UNIFORM_STRIDE` constant" — past tense. `UNIFORM_STRIDE` has 18 uses
under `stroke/`; `UniformSlots` and `InstanceStream` have zero. Per render — which
is per pointer move on a live stroke:

- **Swept/erase**: an instance buffer and a transform buffer + bind group created
  and destroyed (`swept.rs:819-886`), an opacity uniform (`swept.rs:728`), an erase
  uniform (`erase.rs:216`), and two bind groups that are pure functions of
  (brush, seed) (`swept.rs:682`).
- **Dynamics, per piece**: `view_buf` (`run.rs:752`), `tile_inst` (`run.rs:800`),
  `stamp_buf` (`run.rs:955`), 9–11 bind groups, and **one bind group per halo tile**
  (`run.rs:771`) — the exact group the compositor already caches on the tile handle
  (`TilePairHandle::composite_bg`) with a doc explaining that a per-tile-per-frame
  group "was creating ~10⁵ of them a frame".
- **Under any active selection**, the dynamics path creates a region-sized R8 mask
  texture (up to 2048², 4 MB) per piece per move and `destroy()`s it at the submit
  (`run.rs:844` → `selection.rs:387`); never reused.
- `upload_plan` and `sweep_draws` pack slots at a hand-written 256 stride with no
  static check that `size_of::<Stamp>()` fits; `UniformSlots::<T>::STRIDE` already
  computes the padded stride from `T`.

The module's own header (`stroke.rs:70`) says the allocation *rate* is what JS GC
cannot keep up with. `ScopedResources::destroy()` bounds the memory, not the rate.

### Fix

A `Mutex<Streams>` on `StrokeRenderer` — shared across clones like `ScratchPool` —
holding `InstanceStream<SegmentInstance>` and `UniformSlots` for `TileXform`,
`Integrate`, `Stamp` and `ViewUniform`; grow-only, written once per render before
the submit. Sound across clones because every path submits before returning, so a
later `write_buffer` is queue-ordered behind the previous reader. Hand `DynamicsKit`
the compositor's `tile_bgl` (both are built from `COMPOSITE_TILE_SLOTS`) and use
`composite_bg`. Cache the prefix/noise bind groups on `TipCache` beside the views
they wrap. Lease the region mask through `scope.take_piece` (`region_mask` takes the
target view). Delete `UNIFORM_STRIDE`.

## I. The eyedropper submits once per sample point

**Verified.** `crates/stark-engine/src/engine/pick.rs:350` — inside
`for &at in points`: three `offscreen_target` creations and one
`composite_channels` call, which creates an encoder and `queue.submit`s
(`composite.rs:880`). A gradient trace samples up to `gradient::MAX_SAMPLES` = 128
points → 384 textures and 128 submits per pick. The readback side was already
batched (`read_many_rgba16f`) for exactly this latency reason.

Root cause worth naming: `Compositor` has one `ViewBindings` buffer
(`composite/view.rs:88`) written by `queue.write_buffer`. A queue write between two
submits is the only thing ordering them, so each patch's view must be followed by
its own submit — the per-patch submit is a consequence of the view uniform not
being slotted.

**Fix.** `UniformSlots<ViewUniform>`; `composite_channels` takes
`&mut CommandEncoder` and a slot; render every patch of a trace into one
`(N·size)`-wide strip in one submit; one readback.

## J. The Mixbox LUT runs twice per colour, per texel on a placed image

**Verified.** `crates/stark-engine/src/colorspace.rs:267` — `rgb_to_channels`
calls `mixbox::float_rgb_to_latent` and returns `[z0, z1, z2, 1.0]`;
`rgb_to_resid` calls it **again** for `[z4, z5, z6]`. Every caller calls both back
to back — `gpu/place.rs:199` does so **per texel** of a placed image (a 4096²
import evaluates the LUT 33 M times instead of 16 M), and `fill.rs:190`,
`engine/render.rs:322`, `engine/pick.rs:313`, `gpu/stroke.rs:358` likewise. The
fourth lane is `1.0` in both implementations and no caller reads it.

**Fix.** One `fn rgb_to_latent(&self, rgb: Srgb) -> Latent { lat: [f32; 3], res: [f32; 3] }`
— the shader side already has exactly this type (`lib/latent.wesl`).

## K. The log is cloned whole on every save

`Timeline::clone_actions` copies every `Action` — each `CommitStroke`'s
control-point list, the largest thing in the log — on every `save_bytes`,
`document_file` and `start_collaboration` (`engine/file.rs:47`).
`ReplicatedTimeline` stores each action twice (`log` and `history.actions`,
`timeline.rs:656`) plus a third clone for the broadcast (`engine.rs:2100`).
`DocumentFile::new` takes a `Vec<Action>` by value, so the shape is forced from the
model side too.

**Fix.** `Arc<Action>` in the log (the history entry and the outbox share it), or
an iterator into the serializer. Autosave then costs the serialization and nothing
else.

## L. Smaller measurable costs

| Where | What | Change |
|---|---|---|
| `document/patch.rs:340` | `tile_diff` walks the whole layer's tile map per `inverse` per cached state; undoing a stroke over a 5,000-tile layer costs tens of thousands of lookups per cached state to find a handful of tiles. | Iterate `rect.coords()` when bounded; fall back to the walk for `TileRect::ALL`. |
| `engine/collab.rs:68` | `start_collaboration` re-renders the whole document from empty (`ReplicatedTimeline::from_log`); `end_collaboration` adopts for free (`unshare`). The rewrite changes ids, not order; what blocks adoption is that `History` cannot re-key its `Logged` footprints. | `History::map_actions` + `ReplicatedTimeline::from_history`, the counterpart of `unshare`. |
| `gpu/registry.rs:86, 168` | Substrate builds (PNG decode, two Gaussian passes, a 16-direction histogram) and environment builds (HDR decode + CPU mip chain) run under the registry mutex; sibling engines block in `current()`. | Check under the lock, build outside, insert under the lock — the pool's own pattern. |
| `assets.rs:147` | The pen-orientation bake allocates two 64 MiB buffers and runs 16 M `ln` calls inside the store lock on the first pen-oriented stroke. No `asset.*` timing row exists (verified), so the Timing Stats table cannot see it. Cost estimated, not measured. | Add `timing::span!("asset.pen_bake")` first; then a τ LUT for the u8 follow layer, per-layer baking, or bake at import. |
| `session.rs:601, 885` | `as_finished()` — a full extra solve — runs twice per frame from the same fitter state (`gesture_source` and `gesture_view` both reach `fitted`). | Memoize in the fitter, invalidated by `push`/`finish`. |
| `path.rs:477-545` | Per pointer report: two `solve`s and two `mean_error`s, ~25 heap allocations and four curve walks, times up to ~30 tow emissions per report; `grown` is fully built and dropped whenever `as_is` wins; `basis_matrix()` recomputed per `m_step`. *Reported.* | Persistent candidate buffers solved into in place; scratch `Vec`s; `basis_matrix` as a `const`. |
| `session.rs:87` → `assist.rs:671` | Steering an ellipse solves a dense 101×101 Cholesky (bandwidth 4) twice per pointer move. *Reported.* | A banded solve, or first a `timing::span!("assist.steer")` to see if it matters. |
| `document/apply.rs:253` | A neutral-filter merge returns `lower` unchanged but still goes through `with_tiles`, minting a fresh `PaintTiles::revision` for identical tiles — every thumbnail keyed on it re-renders. Same for a `Stack` merge whose `rewritten` set is empty. | Return the layer untouched when nothing was rewritten. |
| `gpu/tile.rs:751` | `format_pools: HashMap<TextureFormat, _>` hashed on the hottest path in the crate (~4 acquires per tile per pointer move). | A fixed array indexed by a channel enum. |
| `gpu/composite.rs:642`, `guides.rs:148` | Matte-ramp and guide bind groups rebuilt every frame. | Cache on `UniformSlots::write`'s `moved`, as `ScratchLevel` does. |
| `gpu/composite/overlay.rs:120` | The selection overlay draws every mask tile of every visible selection, uncullled, while pass A is culled. | Cull by the same `TileRect`. |
| `gpu/selection.rs:245` | `SelectionRenderer::constant` allocates and uploads a texture per call for partial coverage; `mask_for` calls it once per tile of every fill, transform and stroke over an inverted-partial selection. | An `Arc<Mutex<..>>` cache, as `Registry` already uses. |
| `gpu/scratch.rs:70` | `Key.label` is part of the pool key, so identical `Rgba16Float` squares under different labels cannot serve one another, doubling the warm set the 256 MB budget holds. *Reported.* | Drop the label from the key. |
| `assets.rs:187`, `pictures.rs:113` | `bytes()`/`all_bytes()` clone whole PNGs per call (a picture can be megabytes). | `Arc<[u8]>`. |
| `gpu/substrate.rs:137` | The substrate PNG is decoded on `identify`, again on `load`, and again per scale bake. | Store the decoded `Canonical` beside the bytes. |

---

# 3. Architecture

## M. `Engine` is one type with ~25 fields and ~6.5k lines of `impl`

`engine.rs` (2,435) + `render.rs` (1,624) + `live.rs` (731) + `file.rs` (711) +
`pick.rs` (624) + `collab.rs` (353). The module doc is candid: the split "is a
division of the *file* and not of the type: no field moved". Every method shares one
`&mut self` over the timeline, the authoring identity, three memo caches, five
counters, the preview, the peers, the session and two halves of the compositor.
`engine/mod.rs` and `session.rs` are the two most-changed files in the crate (26 and
25 of the last 400 commits).

Symptoms inside it:

- The arm-the-active-layer logic after `AddLayer`, `PlaceImage`, `DuplicateLayer`
  and `MergeLayerDown` is written four times (`engine.rs:1248-1375`).
- `observe(&self)` needs two `RefCell` memo caches (`layer_cache`, `guide_cache`)
  only because it takes `&self`; the frontend always holds `&mut`
  (`stark-ui/src/state.rs:1386`).
- `Compositor`, `CompositorPipeline` and `shared.passes` are held separately with a
  `debug_assert` that they "have not come apart" (`engine.rs:1000`).
- Counters: `doc_revision`, `preview.epoch`, `preview.fold`, `guide_epoch`,
  `gesture_ordinal`, a process-global `TILE_REVISION`, and `history::Version`. Each
  justified; each hand-maintained; `DrawKey` composes three of them.
- `ApplyCtx::prepared` is an out-of-band input threaded through the `history`
  context: set before the push, "accepted" read back as "the slot is empty",
  filtered by `fast_commit`. Verified sound (`is_render_of` compares the record and
  the base map's identity); it is the one place the fold's inputs are not the action
  and the state.

**Fix.** Extract two types along seams the fields already have: an *authored
document* (timeline + `Authoring` + `doc_revision`/`doc_origin` + `commit`,
`settle`, `navigate`, `trim_history`) and a *projection* (the caches, `guide_epoch`,
`observe`). `Engine` becomes the composition of those with `Session`, `Peers`,
`Preview` and the renderer, and each piece's invariants are checkable alone. One
`arm_active(id)` helper. `observe(&mut self)`. A named `Revisions` value so the
counters are visible in one place. If the timeline ever grows a
`push_with(action, extra)` door, `prepared` belongs on it.

## N. `Box<dyn Timeline>` is a two-mode enum with silent defaults

`crates/stark-engine/src/document/timeline.rs:24-140`: `merge` defaults to `false`,
`seek` to `false`, `forget_oldest` to `0`, `undo_as_action`/`redo_as_action` to
`None`. There are exactly two implementations, and the engine already branches on the
mode everywhere it matters (`is_shared()`, `undo_as_action().is_none()`,
`scrub_range().is_none()`). A silent default is the shape of the bug where a new
operation quietly does nothing in one mode; `unshare(self: Box<Self>)` exists only
to get the concrete type back.

**Fix.** `enum Timeline { Linear(LinearTimeline), Replicated(ReplicatedTimeline) }`:
exhaustive matches at the call sites that care, `unshare` a plain method, and
"unsupported here" a visible arm rather than a trait default.

---

# 4. Maintainability

## O. The accumulated-extent render loop is written three times

**Verified.** `gpu/stroke/erase.rs:229-355` and `gpu/stroke/swept.rs:476-636` are
the same ~120-line loop: clone the carried map with `Arc::clone`d accumulators, then
per tile resolve `pristine`, `scratch.keep(..)` a working accumulator, copy the
carried one in, sweep with `LOAD`/`CLEAR`, `acquire_tile`, `mask_for`,
`bind_group_for`, fullscreen integrate, insert, `dirty.push`, `scope.finish()`,
return `Some(carry)`. `swept.rs:334-428` (the stateless ring path) repeats the sweep
and integrate passes a third time, and the integrate bind-group closure is verbatim
at `swept.rs:388` and `swept.rs:584`. The differences are the accumulator set (1 vs
2–3), the integrate pipeline, and whether a missing base tile means "skip" or "bare
canvas". The erase path was added by copying the scaled path; the next effect will
be copied again, and the copies already show small divergences: the "capped"
predicate `k.opacity < 1.0 || selection.is_active()` is written at `swept.rs:270`
and again as `DynamicsRun::capped` (`run.rs:288`), and the `resid` flag is taken
from the tile in hand at four sites while the layout it must match was built from
`color_space.has_resid()`.

**Fix.** One `render_accumulated(&self, scene, rec, k, segments, end_dist, tool,
kind: AccumKind)` where `AccumKind::{Erase, Sweep}` supplies the accumulator keys,
the sweep pipeline and attachments, the integrate bind-group closure and the
missing-tile rule; two helpers `record_sweep(..)` and `record_integrate(..)` serve
all three paths. Fold `capped` and `resid` into `StrokeConstants`, which should take
`&StrokeScene` rather than three loose refs. About −250 lines and one place for the
theorem in `incremental.rs:179`.

## P. `plan::Slot` is a hand-maintained twin of the generated `Stamp`

**Verified.** `gpu/stroke/dynamics/plan.rs:166-390`: a 30-field struct, a 55-line
`Default`, and a `pack()` whose own doc says "**A rename, not a packing.** `Stamp`'s
members are named now". The rename table (`orient`→`orientation`,
`dist`→`arc_at_start`, `ramp`→`radius_ramp`, `bearing`→`tooth_bearing`,
`cell`→`cell_px`, `channels`→`brush_lat`, `resid`→`brush_res`) is exactly the drift
surface §6.10 exists to remove, and it needs an 80-line test to pin it
(`plan.rs:1236`). The generated `Stamp` already has every named member with
`offset_of` assertions.

**Fix.** `impl Stamp { fn neutral() -> Self }` carrying the six non-zero neutrals;
the three arms of `dynamics_plan` build `Stamp { start: p.to_array(), .. }` directly;
keep `the_default_slot_is_neutral_rather_than_zeroed` against `neutral()`; delete
`Slot`, `pack` and the lane-mapping test. About −180 lines.

## Q. The descriptor boilerplate `desc.rs` was written to end

**Counts verified by grep.**

- `fn tex(&wgpu::TextureView) -> BindingResource` is defined six times
  (`blend.rs:50`, `media.rs:36`, `filter.rs:52`, `transform.rs:110`, `fill.rs:64`,
  `merge.rs:74` as `view`).
- The five-field `wgpu::RenderPassDescriptor { label, color_attachments,
  depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
  multiview_mask: None }` appears 15 times in `gpu/` outside the stroke module.
  `desc.rs:1` describes exactly this class as the module's reason to exist, and
  `TileScope::fullscreen_pass` solved it for the tile writers only.
- `merge.rs:357-373` and `merge.rs:515-537` re-spell, arm for arm, the slot→resource
  matches of `blend.rs:176` and `filter.rs:189`. To allow it, `BlendPass`/
  `FilterPass` expose `pipeline`/`bgl`/`pigment`/`sampler`/`tile` and
  `composite.rs:77` re-exports the slot lists "so it must bind the very groups, not
  a second description of them" — the match arms are that second description, and
  the `unreachable!` arm catches a *missing* binding only at run time, on an adapter,
  in the Mixbox half CI does not render.
- "A tile, or the 1×1 zeroes" is derived in `merge.rs:549`, `fill.rs:240` and
  `transform.rs:951` (twice, cloning every view); `desc::Zeroes` and
  `channels::Targets` exist so this has one answer.
- `transform.rs`: `render_parcel` (646-726) and `render_gated_parcel` (845-924) are
  the same function modulo pipeline and group-0 uniform — the cache closure at
  673-699 is byte-for-byte the one at 871-897; `apply_affine`/`apply_gated` likewise.
- `FilterUniform` and `BlendUniform` are assembled in both `composite/plan.rs`
  (341, 269) and `merge.rs` (391, 608); `..Default::default()` fills a lane missed in
  one with zero, silently.
- `merge.rs:83` hand-rolls the one-slot dynamic-offset buffer `UniformSlots` already
  is; `uniforms.rs:28` exports `UNIFORM_SLOT` "for the merge renderer".
- Three owned "trio" types: `blend.rs:262` `Trio`, `media.rs:294` `Offscreen` (the
  same three fields plus `bg`), `channels.rs:114` `Channels`.
- `readback.rs:113` and `:204` duplicate the copy/map/poll/take-rows sequence.
- `PREFIX_SLOTS` declared twice (`swept.rs:31`, `kit.rs:16`); `stamp_shader()`
  compiled into two modules (`swept.rs:145`, `erase.rs:108`) — startup cost on the
  web; four identical "one whole texture" `Extent3d` constants and three
  near-identical tile `Key` builders across `run.rs`/`swept.rs`/`erase.rs`.

**Fix.** In order of payoff: `BlendPass::bind_group(..)` /
`FilterPass::bind_group(..)` as the sole constructors of their groups, fields
private, slot-list re-exports gone (≈ −60 lines in `merge.rs` and the guarantee the
merge cannot drift from the screen); `Zeroes::targets_of`; `desc::tex`,
`desc::begin_pass`, `Targets::begin_pass` (≈ −100); `FilterDraw::uniform` and a
`blend::uniform`; `UniformSlots::<BlendUniform>::new(.., 1)` in the merge; one
`render_parcel_draws` with two thin public paths; `Offscreen { trio: Trio, bg }`
with `Trio` beside `Channels`; one `read_many` with two wrappers; `Key::tile(..)`.
Roughly 600 lines removed, and every remaining site becomes a statement of what
differs.

## R. Two scratch pools and two submit scopes for one need

**Verified.** `render_swept` takes its scratch ring from `TilePool` via
`acquire_scratch` (`stroke.rs:333`) and then needs `swept.rs:306-329` — 27 lines —
to explain the free-list-vs-open-encoder hazard and `swept.rs:436 scope.hold(ring)`
to defuse it: a rule the next call site must remember. `render_swept_scaled` leases
textures of the very same shape from `ScratchPool` via `parcel_key` (`swept.rs:658`),
where the hazard cannot arise by construction (`scratch.rs:12`). `submit.rs:18` and
`scratch.rs:30` each argue the two scope types "may not be collapsed without
weakening one", but `SubmitScope` is already a superset of `TileScope` minus the
`tile_done` cadence and `fullscreen_pass`; the only obstacle is that
`ScratchPool::give` is private to `stroke::scratch`. Smaller asymmetries between the
"siblings": `SubmitScope::encoder()` does not mark the piece open where
`TileScope::encoder()` does; `StrokeCarry::tool` is documented `None` for a range
reaching the stroke's end and honoured by the dynamics path, but the erase and
scaled paths always return `Some`, building a map of leases per move the fold
discards.

**Fix.** Ring → `scope.take_piece(parcel_key(..))`; delete `acquire_scratch`,
`AllocSource::StrokeScratch`, the comment block and the `hold`. Move `ScratchPool` +
`SubmitScope` to `gpu::scratch` and make `TileScope` = `SubmitScope` + the flush
cadence, so the ordering rule lives once. Then `Key` gains a `Buffer` variant and
the buffers of [H](#h-the-stroke-hot-path-does-not-use-the-plumbing-built-for-it)
lease through the same pool.

## S. Four copies of the paint-edit gate, and uneven minted-layer claims

`document/apply.rs`: `CommitStroke` (359), `Fill` (568), `transform_apply` (184)
and `merge_apply` all spell `paint_base → selection_of(actor) → renderer →
map_layer(with_tiles)` with a per-arm `warn!`. And the minted-layer footprints
disagree about what a new layer claims: `PlaceImage` declares `Paint(id, ALL)` +
`Prop(id, Name)` (`footprint.rs:350`); `AddLayer`/`AddMatte`/`AddFilter` and
`DuplicateLayer`'s copies declare only `Existence`. Both are sound — a minted layer's
`Existence` write subsumes everything about it — but it is said two ways, and
`capture_resource`'s `Resource::Layer` write arm (`patch.rs:243`) is "unreached
today" as a result.

**Fix.** One `paint_edit(state, layer, actor, |base, sel| -> Option<TileMap>)`
helper. Declare `Resource::Layer(id)` as the write for every minted layer; the
patch arm becomes reached and tested.

## T. Files and APIs that carry more than they should

- **Public API inflated by test hooks.** 377 `pub fn`. `flush_live`,
  `live_head_count`, `strokes_reused`, `preview_epoch`, `timeline_stats`,
  `render_to_image` are public because integration tests can only reach `pub`. A
  `#[doc(hidden)] pub mod testing` or a `test-support` feature keeps the API honest.
- **Nightly for one trait default.** The whole workspace is on nightly
  (`rust-toolchain.toml`) because the `history` git dependency uses
  `#![feature(associated_type_defaults)]`. Upstream can drop the defaults with no
  loss — Stark writes `type Ctx` explicitly anyway.
- **`path.rs`** (2,693 lines) is a streaming fitter, an arc primitive, a flattener
  and ~900 lines of tests; **`assist.rs`** (1,891) is recognition, adjustment,
  realization and the pen profile, and its own header names "three separable
  pieces". `path/{fit,arc,flatten}.rs`, `assist/{recognize,shape,realize}.rs`.
  `path.rs` also duplicates `SplineIndex`'s knot arithmetic (`span_count`,
  `knot_row`, `span`), pinned only by `span_form_matches_the_fitted_spline`.
- **Stringly errors.** `EngineError::Export(String)` conflates "the request makes no
  sense" (`render.rs:468-612`) with "the PNG encoder failed" (`image.rs:77, 80`);
  `Io` has no producer in `src/`; `DocError::Asset(String)` is the same shape.
  `Export(ExportError)` with `TooSmall`, `OverLimit { size, limit }`,
  `UnusableView`, `Encode(png::EncodingError)`.
- **`spline.rs`** still names `m_step` after deleting the EM it came from, and its
  `prior` parameter is both the prior and the output.
- Pressure and tilt clamps at three sites (`path.rs:596`, `path.rs:1128`,
  `assist.rs:735`) → `ControlPoint::clamped` in the model.
- `live.rs::render_live_stroke` has eight arguments under an `#[expect]`;
  `(actor, ordinal, contested, frozen)` is one `LiveTail` value.
- `timing.rs:533` — the frozen-clock path spins `MAX_SPINS = 2^20` reads of
  `performance.now()` (~100 ms) under a comment saying "microseconds". *Reported.*
- `noise.rs:262` `unit()` returns exactly `1.0` for `h >= 0xFFFF_FF80`; harmless
  here, but `(h >> 8) as f32 * 2^-24` is exact. `image.rs:17` does
  `width * height * 4` in `u32`. `tow.rs::Tow::new` trusts `rope > 0` with no
  `debug_assert`. `merge.rs:347` `encode_filter(tile: Option<_>)` is only ever
  called with `Some`.

## U. Comments that narrate history, or describe code that is gone

**Counts verified by grep.** 19,795 of 50,610 source lines are comments (39%); 88
doc blocks exceed 20 lines; 213 comment lines match "used to / no longer / this
replaced / it was / for a while" (top files: `engine.rs` 20, `path.rs` 18,
`segments.rs` 12). Each keeps a real *why*; each also keeps the archaeology, and the
archaeology is what goes stale. Confirmed-stale, each a trap for the next reader:

- `document/patch.rs:175` — `let _ = action;` under a comment saying the arms read
  it. They don't; `unapply`'s `action` is dead through `capture`.
- `document/selection.rs:379` — says `Selection::from_parts` pins a deselect at
  opacity 1. It doesn't; the pin is in `plan` (`:256`). A reader following the
  comment would reintroduce the pin the ALPN-14 change removed.
- `gpu/merge.rs:125` — says `merge.wesl` only knows the unclipped `Normal` merge;
  `is_direct()` ignores `clip`, `p.z` carries it, and the shader has the branch.
- `gpu/uniforms.rs:13` — says the stroke's stride copy was absorbed
  ([H](#h-the-stroke-hot-path-does-not-use-the-plumbing-built-for-it)).
- `assets.rs:3` — says ids are the BLAKE3 hash of the bytes; they hash the decoded,
  capped coverage, as `import`/`load` say and §19 depends on.
- `document/fill.rs:122` and the model's `fill.rs:222` cite `patch::paint_rect`,
  which does not exist (`tile_diff`); `apply.rs:26, 63` cite `history::Action::apply`
  (`Materialize::fold`); `budget.rs:38` cites `DynamicsRun::flush`; `kit.rs:29`
  links `StrokeRenderer::round_tip` (`TipCache::round_tip`); `run.rs:195`,
  `incremental.rs:236`, `region.rs:453`, `segments.rs:1681` cite `region_of`/
  `affected_tiles`; `segments.rs:1212` cites `MAX_RADIUS_RAMP`. None exist.
- `path.rs:1124` — a doc comment on the wrong item (`Fit` is undocumented,
  `control_points` carries its sentence); `path.rs:984` describes a signature
  `arc_weights` no longer has; `tips.rs:203` — the `RoundTip` doc starts
  mid-sentence; `command.rs:112, 534, 554, 565, 582` link to
  `stark_model::geom::ViewTransform`, which moved.
- `path.rs:2402` — a "known weakness" paragraph describing a fixed bug over an
  assertion (`fitted.len() <= long.len()`) that cannot fail; assert against the arc
  floor instead.
- Test tombstones ("… stood here") at `swept.rs:896` and `plan.rs:1881`;
  `assist.rs:1097` `ring = trace.to_vec()` never mutated — a vestige of the closed
  loop the comment above it says is gone.
- Whole sections of "was": `paint.rs:10` ("# It was two types"), `presence.rs:19`
  (cites commit `77f0f69`), `apply.rs:551`, `footprint.rs:104`, `spline.rs:14, 56,
  260`, `view.rs:7`, `noise.rs:254`, `tow.rs:230`, `path.rs:301, 830`.
- Shader-side, in files this crate consumes: `blend_common.wesl:82` points at a
  module that moved; `mixbox_lut.wesl:17` says the engine drops the residual, which
  §6.7 reversed.

**Fix.** One pass: keep the invariant and the number, drop the "was"; move
measurement history into `docs/` or the memory notes; delete tombstones. Then widen
the rustdoc gate the model ledger's [O] added to this crate so dead identifiers
cannot accumulate silently again.

---

# 5. The test suite

Read against CLAUDE.md's "rules that are easy to break silently":

| Rule | Guard | Strength | Gap |
|---|---|---|---|
| `apply` touches only its `Footprint` (§12.6) | `tests/footprint.rs:313` diffs `DocState` by tile identity for every `ActionKind`, roster from `ActionTag::ALL`; reads only via `commute_pairs.rs` | writes **structural**; reads *enumerated* | [A](#a-a-groups-removelayer-under-declares-its-subtree)'s exemption; [V](#v-footprint-reads-are-checked-over-a-hand-picked-vocabulary) |
| Tile-writing pass is a pure function of canvas position (§6.4) | `seam.rs` ×3; corpus `check_translation` ×16 at worst-texel 4 | strokes only | [W](#w-64-translation-invariance-is-guarded-for-strokes-only) |
| Deposit additive in τ (§6.2) | corpus `check_refinement`; cut-independence tests | indirect | refinement moves the fit as well as the cuts; several `Tol::refine` bounds were "raised to admit" measurements (hairpin 4.0, pressure_ramp 0.7) |
| Conserve height, never alpha (§6.1) | `dynamics.rs`, `opacity.rs`, `erase.rs` via image darkness | proxied | [X](#x-what-the-suite-observes-only-through-the-lit-composite) |
| `preview == committed` (§1.3) | 16-case corpus × 5 checkpoints + commit vs whole render; per-feature tests | broad | fast-commit *off* for one swept brush only; no Mixbox, modulated or stamp+erase case; `rope` never non-zero anywhere |
| `#[serde(default)]` on new log fields (§8) | `stark-model io.rs:658` on two synthesized types | weak | no historical `.stark` fixture corpus; tombstone/ALPN rule unguarded |

## V. Footprint *reads* are checked over a hand-picked vocabulary

`tests/commute_pairs.rs:194` has 19 rows, guarded against vacuity by `>= 17` and
`>= 35` — neither tied to the roster. Kinds absent: `SetSubstrate` (the one whose
undeclared read already cost the model ledger's [A]), `SetMatteRect`,
`SetMattePaint`, `TransformPerspective`, `TransformWarp`, `SetFilter`,
`MergeLayerDown`, `PlaceImage`. A new variant is invisible to the test by
construction. `tests/footprint.rs`'s `differences()` (100-198) is a hand-written
field list of `DocState` and `Layer`; a new field is invisible until someone adds a
line.

**Fix.** Mint one action per `ActionTag::ALL` entry and assert `missed.is_empty()`
as `footprint.rs:785` does; `SetSubstrate` first. Split into two tests if the 240 s
ceiling bites (the cost is documents, not rows). Derive `differences()` from the
same debug-assert [D](#d-footprint--apply-correspondence-is-held-by-discipline)
adds, so there is one enumeration.

## W. §6.4 translation invariance is guarded for strokes only

`seam.rs` covers the swept impasto stroke, the placed image and the dynamics
write-back; the corpus covers 16 stroke configurations. No translation-invariance
test exists for `gpu/fill.rs`, `gpu/transform.rs` (affine, perspective, warp —
`transform.rs:197` moves the *content*, not the tile phase), `gpu/merge.rs` or
`gpu/selection.rs`. And `seam.rs:102` is loose — `worst <= 25 && frac < 0.07` —
where the corpus holds the same swept stroke to 4 levels at every texel.

**Fix.** One table-driven `check_translation`-style test over fill, the three
transforms, merge-down and select; tighten `seam.rs` to the corpus's 4 levels.

## X. What the suite observes only through the lit composite

`dynamics.rs:618` — "Measured as image darkness rather than height, since there is
no height readback". Yet `gpu/readback.rs:204 read_many_rgba16f` exists. Every
conservation, opacity and erase claim is a proxy through tonemapping. The
`debug-unfrozen`-gated live tail, the tow (`rope` never non-zero in any test), the
fast-commit-off path for anything but one swept brush, Mixbox `preview == committed`
and modulated strokes are uncovered.

**Fix.** A `tests/common` helper over `read_many_rgba16f` so conservation and erase
assert on height and alpha directly; run the corpus battery once with
`SetFastCommit(false)`; one tow case in the corpus.

## Y. Suite infrastructure

- Device acquire-or-skip logic exists in four copies (`common/mod.rs:92`,
  `tile_pool.rs:16`, `benches/stroke.rs:115`, `stroke.rs:646` ungated).
- 21 helpers are defined in 2–5 files each (`screen_of` ×5, `texel` ×4, `apart` ×4,
  `is_red` ×4, …); two notions of "is there paint here" coexist with different
  margins (`common::red_dominant` 30 vs `is_red` 60).
- ~97% of integration tests need a GPU; the 310 unit tests in `src/` need one in
  four places. There are no property-based tests anywhere. Where they pay most, all
  GPU-free: a counting `Materialize` (`Ctx = ()`) over random logs of
  `(actor, lamport, footprint)` with undo/redo/late arrival, asserting
  splice == rewind+replay — the invariant `commute.rs` checks in six hand scenarios,
  with shrinking, on the CI box; footprint algebra (`conflicts` symmetric,
  `TileRect::covering` contains every tile a stroke can touch); `PathFitter`
  (NaN-free for any input, `push` == batch, resample invariance).
- Written-to-pass shapes worth revisiting: `dynamics.rs:76, 1099` "almost nothing"
  as `frac < 0.2` of pixels differing by > 40 levels; `reversals <= 6` / `<= 4`
  with no derivation; `commute_pairs`' floors.
- Doc drift: `common/mod.rs:303` `diff_fraction` doc vs code; `corpus.rs:64`
  "thirteen of the fourteen" for 16 cases; `docs/engine.md` §9 lists a bench as a
  test and omits 17 current test files.

---

# 6. Looked wrong, turned out handled

Kept so the next review does not spend the same hour.

- **The apron rule (§6.4) across tile writers.** Merge (`merge.wesl` per-texel
  `load1`), slab, point filters (the chromatic gather is refused at `merge::plan`
  and asserted), fill, the CPU place path, and the transform's forward
  rasterization into the full `TILE_TEX` target all compute each texel from its own
  position.
- **Group opacity applied once.** `CompositeGroup::leaf` folds it into items and
  clears it off the merge in one expression; `stack` collapses with `run`, not
  `leaf`. Clipping suppresses ghost impasto via the aux scale in
  `blend_common.wesl::merge`.
- **Free list vs open encoder in the multi-piece dynamics loop.**
  `run.rs:535 scope.hold(base.clone())` keeps the previous piece's handles alive
  until the flush that submits the reader — structural, not statement order. The
  pool's `destroy()` is quarantined by epoch and wgpu defers it behind submitted
  work.
- **`ApplyCtx::prepared`'s acceptance check** — record equality plus base-map
  identity — is exact for a persistent map.
- **Determinism.** Every `HashMap`/`HashSet` in the fold is lookup-only or sorted
  before iteration (`restore_structure`, `revival_keys`, `fill::plan`, `plan_mask`,
  `plan_gated_mask`); `tile_diff`'s rpds order feeds an order-insensitive patch;
  pass A's tiles are disjoint so draw order is immaterial.
- **wasm32.** No `std::time::Instant` in scope; `quad_reached_tiles` rejects on
  32-bit exactly where the 64-bit path rejects on budget; asset volumes are ≤ 16 M
  entries.
- **`spline.rs:427`'s `unreachable!`** is safe for finite, bounded input; NaN is
  refused at `InputSample::is_finite` and both session entry points.
- **The `fresh` budget seeded across dynamics pieces** cannot disagree in apron
  overlaps: `segment_bounds` grows every box by an apron on both sides, so any
  deposit into an overlap touches both tiles in the same piece.
- **`BRUSH_RES` on the host** is an allocation size, not a transcription;
  `exchange` derives the split from `textureDimensions`.
- **Every un-gated Mixbox test** is either `#[cfg(feature = "mixbox")]` or filtered
  through `colorspace::available`.
- **Fill's temporary region `Selection` outlives its submit** — by declaration
  order (`fill.rs:165` vs `scope.finish()` at `:287`). Sound; `scope.hold(region)`
  would say so.
- **Withdrawn during review:** a reader's claim that hover has no timing row —
  `engine.rs:1567` opens `input.hover`.
