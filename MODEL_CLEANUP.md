# `stark-model` cleanup

A review of `crates/stark-model` as of 2026-08-21 (`f443530`): six defects, five
structural changes and four sweeps, each with the file and line that shows it.

## Status

| | | |
|---|---|---|
| **D1** | gradient stop count | **open** |
| **D2** | "nothing to hold" holds plenty | **open** |
| **D3** | selection shape unvalidated | **open** |
| **D4** | gradient parcel skips a clamp | **open** |
| **D5** | idempotence test claims 31, drives 24 | **open** |
| **D6** | doc block on the wrong item | **open** |
| **A1** | the funnel belongs in the payloads | **open** |
| **A2** | perspective footprint's honesty | **open** |
| **A3** | no `tests/` in the crate that owns §12.6 | **open** |
| **A4** | `Tool` is session state | **open** |
| **A5** | `geom.rs` is two modules | **open** |
| **S1** | unused `tracing` dependency | **open** |
| **S2** | two allocations per logged action | **open** |
| **S3** | `to_bytes` copies three times | **open** |
| **S4** | `Action` clone carries stroke paths | **open** |

The line numbers below are from the review and are **not** updated as the fixes
land — they are what the finding was found at. Follow the named function, not
the number.

Nothing here is a redesign. The crate's spine holds up: `Logged<S>` solving the
orphan-rule problem *and* caching the footprint once per push, `Materialize`
keeping pixels out of the model, `TileRect::covering` as the one quantizer
returning the question rather than picking an answer, and the
exhaustive-match-with-no-`_`-arm device that stops a new variant compiling until
it is visited. There is no `unsafe` in the crate and no `unwrap`, `expect` or
`panic!` outside tests.

What the list is about is narrower and repeats: **the mechanism a stated
invariant claims to rest on is, in these places, not the mechanism holding it
up.** The funnel is exhaustive but its classification is unchecked (**D2**); the
one door refuses NaN but not size (**D1**, **D3**); a compile-time assert pins
the shader against a constant while the data is free to disagree with both
(**D1**); a test's doc comment describes a generality it does not have (**D5**).
In every case the enforcement stops one step short of what the design already
claims, which is why the fixes are small.

The three worth doing first, if only three are done: **D1** (a handful of lines
in `Gradient::new`, and it is a panic reachable from a document you opened),
**D2** (four action kinds carrying floats to a shader through a funnel that
believes it covered them), and **A1** (which is what makes **D2** and **D4**
unable to come back).

## Defects

### D1. `Gradient` has no stop-count invariant, and two consumers index a fixed 16

`Gradient::new` (`gradient.rs:81`) is the type's only door — deserialization
funnels through it by `try_from` — and it enforces two stops, all finite,
distinct endpoints, sorted and rescaled. It does **not** enforce `MAX_STOPS`
(`gradient.rs:206`), which is only the *fitter's* stopping condition
(`gradient.rs:330`). So a file or a peer can carry a ten-thousand-stop ramp and
`Gradient` will call it well-formed.

Every consumer assumes sixteen:

```rust
// stark-engine/src/gpu/fill.rs:194
for (i, stop) in stops.iter().enumerate() {
    uniform.stop_c[i] = ...;   // stop_c: [[f32; 4]; 16]  → panics at i == 16
```

The same shape at `stark-engine/src/engine/render.rs:1080-1084`, for
`MattePaint::Gradient`. `stark-engine/src/gpu/composite/group.rs:165` uses
`.zip()` and survives — then writes `params[0] = g.stops().len()` at line 181,
telling the shader a count larger than the lanes it filled.

The guard that exists is `gpu/fill.rs:71`:

```rust
const _: () = assert!(
    stark_shaders::mirror::fill::MAX_GRADIENT_STOPS as usize == stark_model::gradient::MAX_STOPS
);
```

— which makes the *shader* and the *constant* agree and never looks at the data.
Its comment says an oversized gradient "would truncate silently". It does not
truncate; it panics. Two of the three call sites disagree about the failure mode,
which is the tell that nobody owns the bound.

**The fix is to make it a type invariant**, in `Gradient::new`, so the fixed-16
arrays cannot be overrun and the `zip`-versus-`[i]` asymmetry stops mattering.

The type's own doc already says how, and names this exact invariant while doing
it (`gradient.rs:57-64`):

> But adding a condition — **a stop-count bound**, monotonic lightness — would
> retroactively unload files that were valid when saved, which §19 does not
> permit. A tightened invariant has to arrive as **repair on the way in** (drop
> the offending stops, spread coincident positions, fall back to a two-stop ramp)
> with `new` left refusing for the *authoring* path.

So: truncate past `MAX_STOPS` on the way in, keep `new` refusing the *structural*
failures for the caller who traced a line, exactly as it does now. The paragraph
that anticipated the bound is the one piece of evidence that this is a gap rather
than a decision.

### D2. Four action kinds carry unvalidated floats under "nothing to hold"

`ActionKind::sanitized` (`action.rs:639`) is the crate's centrepiece: one funnel,
no `_` arm, so a variant added later stops it compiling. But the compiler forces
a *visit*, not a correct *answer* — and the "nothing to hold" arm
(`action.rs:698`) currently holds four things that carry numbers to a shader:

| variant | what it carries | gate today |
|---|---|---|
| `SetBackground([f32; 3])` | straight sRGB | none — `with_background` (`stark-engine/src/document/state.rs:259`) stores it raw |
| `SetMattePaint(_, MattePaint)` | `Solid([f32; 3])`, or a whole `Gradient` | none — `MattePaint` has no `sanitized` at all |
| `AddMatte { region, paint }` | the same paint, plus `OutsideRect { min, max }` | none |
| `SetMatteRect(_, Vec2, Vec2)` | canvas geometry | none |

The arm's own comment justifies the omission as "the geometry whose own
`usable`/`affine_usable` gate rejects it at `apply`". Those gates exist for
`Transform`, `TransformPerspective` and `TransformWarp` and for nothing else;
there is no `usable` anywhere for a matte rect. Four variants are covered by a
sentence describing three others.

A NaN background reaches the presenter's clear; a NaN matte color reaches
`matte.wesl`. Neither is caught anywhere downstream, and neither can be seen in
a pixel as anything but "the frame is wrong".

The narrow fix is four arms. The fix that makes it not come back is **A1**.

### D3. A selection's shape is never validated — neither its coordinates nor its size

`SelectionOp::at` (`selection.rs:186`) is documented as the deserialization
funnel and clamps `feather` and `opacity`. It never touches `shape`.

**Non-finite coordinates.** `SelectionShape::bounds()` folds with
`Vec2::min`/`max`, which return the *non*-NaN operand — so a `Lasso` carrying one
NaN vertex yields a tight, finite box. This is precisely the hazard `stroke_rect`
calls out and guards against, one file over (`footprint.rs:265`):

> Tested rather than folded in: `f32::min`/`max` return the *non*-NaN operand, so
> a non-finite point would step straight over the bbox and leave it looking
> tight.

The reasoning was not carried across. The NaN then rides into the edge texture
through `geom::lasso_edges` and into `selection.wesl`, where
`clamp(0.5 - sd/w, 0, 1)` on a NaN is implementation-defined — a divergence in
the one place §6.8 says the only tolerable disagreement between two clients is
none.

**Unbounded vertex count.** `Lasso(Vec<Vec2>)` is decimated at *capture* by
`LASSO_MIN_STEP` (`stark-engine/src/session.rs:29`), which says nothing about a
file or a peer. `edge_texture` then does `width: edges.len() as u32`
(`stark-engine/src/gpu/selection.rs:485`) — past `max_texture_dimension_1d`
that is a wgpu validation error, and the mask shader is O(vertices) *per texel*,
so a hundred-thousand-vertex lasso is a GPU hang from a document that opened
cleanly.

`MAX_SELECTION_TILES` bounds how many tiles an op may rasterize and nothing
bounds the per-texel cost. Both the finiteness test and the vertex cap belong in
`SelectionOp::at`, beside the two clamps already there — and the cap wants the
same repair-not-refuse stance as **D1**.

### D4. `Parcel::Gradient` skips the clamp its sibling gets

```rust
// document/fill.rs:172
let paint = match paint {
    Parcel::Solid(c) => Parcel::Solid(c.map(clamp01)),
    gradient => gradient,          // ← passes through untouched
};
```

`Filter::GradientMap`'s sanitizer (`filter.rs:149`) clamps gradient stop colors
into the sRGB cube, with a good argument for why —

> a finite 1e30 would reach every texel just as surely as a NaN saturation

— and a test pinning it
(`sanitizing_a_gradient_map_clamps_its_stops_to_the_cube`). The identical ramp
inside a `FillOp` gets nothing, and the one inside a `MattePaint` is not
sanitized at all (**D2**). Same data, same shader path, opposite treatment, no
stated reason for the difference.

### D5. The idempotence test claims a generality it does not have

`action.rs:864`:

> Driven off the same one-of-each list the footprint's exhaustiveness device
> uses, so a variant added later is covered here as soon as it is added there.

It is not. `sanitizing_is_idempotent_on_every_kind` walks a hand-written array
(`action.rs:869`) of **24** kinds against the **31** that
`stark-engine/tests/footprint.rs`'s `slot` enumerates. Missing:
`TransformPerspective`, `TransformWarp`, and all five guide variants.

The guide variants are the ones that matter, because `AddGuide` and `SetGuide`
are among the handful that actually *transform* their payload —
`PerspectiveGuide::sanitized` (`guide.rs:358`), whose own doc spends a paragraph
asserting idempotence and reasoning about `Quat::is_normalized`'s tolerance. That
claim is unverified, and the comment above the test is the reason nobody noticed.

Either drive it off `slot`'s list for real (see **A3**) or delete the sentence.
The sentence is worse than no sentence.

### D6. A doc block is attached to the wrong item

`brush.rs:779-803`: three paragraphs describing the `elongation()` *function* —
"Takes the modulated knob rather than reading `stretch`, because what a
`Modulation` scales is the knob and not the factor" — sit on
`const MAX_STRETCH`. Only the last paragraph is about the constant. The real
`elongation` (`brush.rs:848`) gets the `min`-then-`max` NaN note and nothing
else, so the argument for *why the knob is quoted as the reciprocal's argument*
is not findable from the function it is about.

Cheap, and worth doing for the same reason **D5** is: in this crate the prose is
the specification, so prose that has drifted off its subject is the same class of
defect as code that has.

## Architecture

### A1. The funnel belongs in the payload types, not in the match

**D2** and **D4** are the same defect twice, and a third instance is one new
action away. `ActionKind::sanitized` asks the author, per variant, "does this
carry a number?" — and the compiler can check that the question was *asked*, never
that it was answered right.

Move the funnel down. Give `MattePaint`, `MatteRegion`, `SelectionShape` and
`Parcel` a `sanitized()` of their own; then the match arm's only judgement is
"does this payload type have a sanitizer", which the type system answers instead
of the author. The "nothing to hold" arm would then hold only what it says: ids,
flags, places, and the geometry with a `usable` gate at `apply`.

**Stronger, and I think the right endgame:** an `Srgb([f32; 3])` newtype whose
only constructor clamps. `SetBackground`, `MattePaint::Solid`, `Parcel::Solid`,
`GradientStop::color` and `BrushParams::color` are five copies of one clamp
obligation, of which four are currently unmet. A representation that cannot
express the wrong thing removes the obligation rather than tracking it — which is
the crate's own standing preference:

> Rule out a class rather than enumerate its instances.

Note the wire cost is nil: a newtype over `[f32; 3]` with `#[carbonite(as = ...)]`
is the same columns under the same names, the device `FillOp` and `SelectionOp`
already use.

### A2. The perspective footprint's honesty rests on a refusal in another file

```rust
// footprint.rs:439 — always Some(...)
gated_rect((map.min, map.max), Some(map.image_aabb())),
// footprint.rs:447 — an Option, so an unusable map claims TileRect::ALL
gated_rect((map.min, map.max), map.image_aabb()),
```

`PerspectiveMap::image_aabb` (`transform.rs:106`) folds with `min`/`max`, so
non-finite corners produce a *tight* box where the warp arm falls back to
everything.

It is not a §12.6 break today: `usable()` rejects the action at `apply`, so its
true footprint is empty and any claim contains it. But the safety now depends on
a refusal made elsewhere, which is exactly the reasoning `MergeLayerDown`'s own
footprint comment refuses to accept:

> Claiming only the opacity would be resting the footprint's honesty on refusals
> made in *another file*.

Give `PerspectiveMap::image_aabb` the warp's `Option` return and the two arms say
the same thing for the same reason. One line each side, and it removes a standing
invitation to reuse `image_aabb` somewhere that has no `apply` behind it.

### A3. The crate that owns §12.6 has no `tests/` directory

Every test in `stark-model` is an in-file `#[cfg(test)] mod tests`. The two
cross-cutting guards that matter most — footprint honesty, and action-kind
exhaustiveness (`slot`/`KINDS`/`NAMES`) — live in
`stark-engine/tests/footprint.rs`, which needs a GPU.

So the crate's headline invariant — *every `apply` must touch only what its
`Footprint` declares* — cannot be checked in a headless build of the crate that
owns it, and `cargo nextest run -p stark-model` proves considerably less than a
green run suggests. It is also why **D5** was able to drift: the list it claims to
follow is in another crate, behind an adapter.

The *enumeration* is a fact about `ActionKind` and belongs beside it. Splitting
`slot`'s one-of-each list into `stark-model/tests/` — leaving the GPU-driven diff
comparison where it is, since that genuinely needs a device — would let the
model's own suite fail when a variant escapes the funnel, and would give **D5**
something real to be driven off.

### A4. `Tool` is session state living in the document crate

`Tool` (`action.rs:78`) has zero uses inside `stark-model` beyond its definition
and its re-export; every consumer is in `stark-engine` or `stark-ui`. Its own doc
concedes the point:

> **Session state, not document state.** … which of them was in hand is not part
> of what a document *is*.

`lib.rs` advertises `#[derive(Serialize)]` as the *mechanical* placement test —
"that is not a judgement call". It decides one direction only: serializable
implies here. There are roughly sixteen non-serializable types in the crate
placed by judgement, which is fine — `Footprint`, `Homography` and `Lattice` are
all derivations *of* document facts. `Tool` is the one where the judgement went
the other way from the crate's own argument, and it moves to `stark-engine`'s
`command`/`session` without a single other edit.

Worth saying plainly in `lib.rs` either way: the test is a sufficient condition,
not a partition.

### A5. `geom.rs` is two modules under one name

The tile grid — `TileCoord`, `TileRect`, `covering`, `TILE_SIZE` — is genuinely
the document's addressing, and the module header says so ("The tile grid only").
Sharing the file with it:

- `TILE_TEX`, `TILE_APRON`, `INTERIOR_UV_SCALE`, `INTERIOR_UV_BIAS`, `MASK_TEX`,
  `mask_tex_origin` — a texture layout, an apron band and two UV constants that
  exist for the compositor's bilinear taps;
- `lasso_edges`, whose doc says outright "as `selection.wesl` reads it";
- `Ellipse` and `principal_axis`, both of whose callers are in `stark-engine` by
  their own doc comments.

`io.rs:117` argues at length that even `TILE_SIZE` is not a fact about a painting
("An implementation detail is not a fact about a painting"). By that argument the
UV bias certainly is not. This is not urgent and nothing is wrong today — the
model compiles without the shaders either way — but it is the seam along which
the file will keep growing, and naming it now is cheaper than finding it later.

## Sweeps

### S1. `tracing` is an unused dependency

`Cargo.toml:30`. No `tracing::`, no `use tracing`, no `info!`/`warn!`/`debug!`/
`instrument` anywhere in `src/`. It is in the wasm payload and the build graph for
nothing. One line.

### S2. Two heap allocations per logged action, for the life of the history

`Footprint` is `{ reads: Vec<Resource>, writes: Vec<Resource> }`
(`footprint.rs:175`), and every `Logged` holds one from push to drop. Almost
every footprint has at most two reads and four writes; `Resource` is 32 bytes.

A `SmallVec<[Resource; 4]>` removes roughly twenty thousand allocations from a
ten-thousand-action log and — the part that matters more — makes `conflicts`'s
nested scan cache-resident. `conflicts` is the hot one: `history` builds the
centralizer once per removal and then asks it about *every* later action, which
is the whole reason the footprint is cached in `Logged` in the first place.

Measure first: the win is in allocator traffic and locality rather than in a
number a profile hands over, and it costs a dependency the crate does not have
today.

### S3. `to_bytes` copies the whole document three times

`io.rs:214-230`: carbonite writes a `Vec`, deflate writes a second, then a third
is allocated to prepend `MAGIC`. On a document dominated by placed pictures (§23,
and by far the largest thing in the container) that is 3× peak over the encoded
size on every save.

Seeding the encoder with a sink that already holds the magic —
`DeflateEncoder::new(Vec::from(&MAGIC[..]))` — removes the third copy entirely.
The magic stays uncompressed because it is already in the sink, before the
encoder writes a byte; `finish()` hands back the whole container. Four lines, and
`container_roundtrips` already pins the result.

### S4. `Action` clones carry stroke paths by value

`Logged: Clone` clones the `Action`, which for a `CommitStroke` clones a
`Vec<ControlPoint>` — 24 bytes a point, hundreds of points on a long stroke.
Whether that shows up depends on how often `history` clones rather than borrows.

`Arc<StrokeRecord>`, or `Arc<[ControlPoint]>` for the path alone, is the lever if
it does. **Measure before acting** — and note the wire is unaffected either way,
since `Arc<T>` serializes as `T`.

## What was checked and found sound

Recorded so the next reader does not re-derive it:

- **No `unsafe`, and no `unwrap`/`expect`/`panic!`/`unreachable!` outside tests.**
  Every slice index in non-test code is bounds-guarded by construction, and the
  guards were followed rather than assumed — `Gradient::sample`'s
  `partition_point` cannot return 0 given the endpoint invariant, and the `0 =>`
  arm is there anyway (`gradient.rs:112`).
- **`TileRect`'s saturating arithmetic.** `ALL.count()` saturates to `u64::MAX`
  rather than wrapping to zero, `covering` refuses rather than clamping, and the
  float→int casts are the saturating ones. The emptiness predicate is one
  definition consulted by all of `intersects`/`union`/`count`/`contains`, which
  is what the per-axis inline spellings got wrong before.
- **`BrushParams::sanitized` and the NaN policy behind it.** `clamp01` is
  `max`-then-`min` rather than `clamp`, deliberately and with the reason written
  down, and `elongation` orders its `min`/`max` so a NaN knob falls out as the
  *identity* rather than as the widest footprint. The poison-list test covers all
  sixteen fields and asserts idempotence per field.
- **`stroke_pad`'s elongation factor.** The footprint grows with the drawn-out
  tip, the test states it as an inequality against what the renderer will paint
  rather than against a copy of the expression, and the bleed argument for why the
  lateral flux needs no allowance is sound — the no-flux wall is a shader
  invariant, correctly noted as checkable only where the shader is.
- **The container's two doors.** `from_bytes` unbounded, `from_untrusted_bytes`
  capped, the bound applied *before* inflation rather than after, and the legacy
  sniff hanging off the inflate error so a current file is never asked whether it
  looks old. All three are pinned by tests, including the one that matters
  (`a_current_document_is_not_mistaken_for_an_old_one`).
- **`Resource::Layer` as the coarse claim.** Symmetric, meets every finer resource
  of its layer and nothing of another, and the test drives it off `Prop::ALL` so a
  new property is covered as soon as it is added. The `Prop::ALL` visit-forcing
  match is the right device for what Rust can express here.
- **The wire mirrors.** `FillOp`, `SelectionOp` and `Gradient` each state their
  representation with `#[carbonite(as = ...)]` in *both* directions, so a
  one-sided conversion will not compile and there is no second layout to drift
  from. This is why **A1**'s newtype costs nothing on the wire.
- **Dependency direction.** Nothing here names `stark-engine`, there is no build
  script, and no `mixbox` feature exists in this crate to `cfg` a save-format
  variant away — which is the structural version of the rule `colorspace.rs`
  used to have to remember.

## What this was checked against

Reading, not running: this is a review of `crates/stark-model` and its consumers
in `stark-engine`, and no gate was run for it. Every claim above is a citation to
a line in the tree at `f443530`, and the two that are behavioural rather than
structural — the `stop_c[i]` panic in **D1**, and the shader-side NaN in **D3** —
are read off the array's declared length (`fill.wesl:65`, `array<vec4<f32>, 16>`)
and off `selection.wesl`'s ramp rather than observed.

**Both are worth reproducing before they are fixed**, since a fix aimed at the
wrong failure mode is what **D1** already is once:

- **D1** — construct a `Gradient` with seventeen stops through `try_from` (a
  hand-built `Vec<GradientStop>`, since `fit` will not produce one), put it in a
  `FillOp` parcel, and apply the fill. The expectation is an index panic, not a
  truncated ramp.
- **D3** — a `Lasso` with one NaN vertex, committed as a `Select`, and the
  question is what the mask does with it on this adapter versus another. The
  divergence is the finding; the tight bbox is only how it gets there.

Neither needs a document on disk — both are reachable from
`stark-engine`'s test harness with a hand-built action, which is the same route
`tests/footprint.rs` already takes.
