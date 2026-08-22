# `stark-model` cleanup

A review of `crates/stark-model` as of 2026-08-21 (`f443530`): six defects, five
structural changes and four sweeps, each with the file and line that shows it —
and, below each, what actually happened when it was fixed.

**Two more defects were found in the fixing than in the reading.** **D7** by the
test **A3** asked for, on its first run; **D8** by asking the presence wire the
question **A1** had just been asked of the action log. That is the result worth
carrying into the next review: the items that paid best were not the six found by
reading, they were the one that said the crate could not check itself and the one
that moved a rule into a type. Both then found things reading had missed.

## Status

| | | |
|---|---|---|
| **D1** | gradient stop count | **done** — `f7639e5` |
| **D2** | "nothing to hold" holds plenty | **done** — `f7639e5` |
| **D3** | selection shape unvalidated | **done** — `f7639e5` |
| **D4** | gradient parcel skips a clamp | **done** — `f7639e5` |
| **D5** | idempotence test claims 31, drives 24 | **done** — `f7639e5` |
| **D6** | doc block on the wrong item | **done** — `e653851` |
| **D7** | an infinite feather | **done** — `f7639e5`; found in the doing, see below |
| **D8** | a peer's live brush skips the funnel | **done**; found in the doing, see below |
| **A1** | the funnel belongs in the payloads | **done** — `f7639e5` |
| **A2** | perspective footprint's honesty | **done** — `e653851` |
| **A3** | no `tests/` in the crate that owns §12.6 | **done** — `f7639e5` |
| **A4** | `Tool` is session state | **done** — `e653851` |
| **A5** | `geom.rs` is two modules | **done** — `e653851`, and the move; the review was a third wrong, see the item |
| **S1** | unused `tracing` dependency | **done** — `e653851` |
| **S2** | two allocations per logged action | **closed** — measured, not worth it; see the item |
| **S3** | `to_bytes` copies three times | **done** — `e653851` |
| **S4** | `Action` clone carries stroke paths | **open** |

Everything here is on `model-cleanup`, checked against every gate the project
names — both clippy configurations, `cargo nextest run --workspace` (1115 tests,
all passing), the doctests, and the wasm build.

The line numbers below are from the review and are **not** updated as the fixes
land — they are what the finding was found at. Follow the named function, not
the number. Each item carries an **As built** note; several of them record the
doing differing from the sketch, and one (**A5**) records the review being wrong.

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

The three that were worth doing first, and were: **D1** (a handful of lines
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

**As built** (`f7639e5`): `Gradient::new` calls a `thin` that keeps both endpoints
and spreads the rest evenly **across the list rather than across `t`** — a capture
puts stops where the ramp turns, so index spacing already tracks where the
structure is, and thinning by position would spend the budget on whichever stretch
happened to be long. Keeping the ends is what lets it run before the rescale
below it: `lo` and `hi` are the same stops either way. `MAX_STOPS`' own doc now
says it is an invariant of the type rather than a budget of the fitter, and
`gpu/fill.rs`'s const assert says what it actually guards — two constants, never
the data.

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

**As built** (`f7639e5`): the paint is clamped and the geometry is gated, which is
the split the arm's comment was reaching for and got backwards. `MattePaint` and
`Parcel` grew a `sanitized`; `MatteRegion` grew a `usable`, and `insert_matte` and
`set_matte_rect` decline an unmeasurable rect the way `apply` declines an unusable
affine — a frame is a rectangle the artist placed, and there is no other rectangle
that is a repaired version of one nobody can measure. The "nothing to hold" comment
now states the rule rather than an example of it: *a variant belongs here when its
payload is ids, flags, places, `bool`s and `String`s; if it carries a float, it
belongs above, or beside a `usable` this comment can name.*

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

**As built** (`f7639e5`): the two halves landed in different places, and that turned
out to be the point. The **count** is repaired, in a `SelectionShape::sanitized` both
funnels call. The **coordinates** are not: `bounds` refuses what it cannot measure,
and every consumer already had the right answer waiting — a `None` declines the op
at `Selection::plan`, fills nothing at `fill::plan`, and claims the whole layer in a
footprint, which are the safe answers in their three directions. So a bad rect is
gated exactly as a bad affine is, and only the lasso's length is repaired. The
lasso's fold became a `try_fold`, which is the same shape `stroke_rect` uses.

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

**As built** (`f7639e5`): `Gradient::clamped` is the one definition now, called from
all three. It returns `Self` rather than `Option`, since clamping moves no position
and drops no stop — so there is nothing for the funnel to refuse, and the `and_then`
the filter site had to explain away is gone. The loop it replaced spelled the bound
`f32::clamp`, which returns the NaN it exists to catch: unreachable there, because
`Gradient::new` admits no non-finite stop, but the third place in the crate to write
the policy down and the second to write it wrongly — which is the argument for there
being one of it.

The **axis** needed an answer the review had not thought about: `ramp_position`
already floors a zero-length line and a zero radius, so the only unhandled case is
non-finite, and there is no repaired axis to clamp to. It degrades the parcel to
`Solid(gradient.sample(0.0))` — `swatch` already calls the first stop "the stop the
axis anchors on", so a ramp nobody can place still knows what color it starts from.

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

**As built** (`f7639e5`): driven off a real list, in `stark-model/tests/action_kinds.rs`
(**A3**), and `the_list_holds_one_of_every_kind` pins the list to the match so it
cannot drift again. One list serves both runs — `kinds(n)` takes the number every
float in it is built from, so the poisoned run and the idempotence run cover *the
same* set by construction, which is the failure this whole item is about.

One thing the sketch got wrong: there is no single `n` in range for every field (a
tint stops at 0.16, a focal length starts at 1), so a bit-for-bit "ordinary" run
over `kinds` would have been checking the literal chosen in the test rather than the
funnel. The property goldens actually rest on is that each type's **own default** is
a fixed point, which is in range by construction — that is
`every_default_is_already_sanitized`, and it is the stronger claim.

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

### D7. An infinite feather is half a selection, everywhere

**Found by the fix for D5, not by the review** — which is the best argument for
**A3** this document contains.

Both funnels floored the feather with `feather.max(0.0)`, under a comment
explaining that `max` rather than `clamp` is what makes a `NaN` land on 0. True,
and exactly half a guard: `f32::max` passes `+inf` straight through. An infinite
feather reaches `selection.wesl` as a coverage ramp of infinite width, where
`clamp(0.5 - sd/w, 0, 1)` is `0.5` at every texel that is not itself infinitely far
away — a half-selected plane nobody asked for — and `NaN` at the ones that are.

The same half-guard was *not* present in `brush.rs`, which had spelled the pattern
properly as `at_least_zero` (finite first, **then** floored) for every length it
holds. So the fix was to stop having two spellings: `at_least_zero` and `finite_or`
moved to the crate root beside `clamp01`, and every non-negative length in the crate
goes through one of them.

### D8. A peer's live brush never passes the funnel its committed twin does

**Also found in the doing**, by asking of the presence wire the question **A1**
had just been asked of the action log: *which payloads here carry numbers, and what
gates them?*

Three shapes travel in a `GestureFrame`. A `Selection` and a `Fill` carry ops whose
**deserialization** funnels through `SelectionOp::at` and `FillOp::with_paint`, so
they arrive normalized however they were built. A `Stroke` carries a `StrokeHead`
holding a plain `BrushParams`, whose `Deserialize` is derived — and the thing that
sanitizes a brush is `ActionKind::sanitized`, which a presence frame never passes
through, because a live gesture is not an action until it commits.

So a peer's radius sized a dispatch and its pickup rates reached the dynamics loop
at whatever the wire said. Not a convergence bug — presence is per-client and never
replayed (§17.4) — but the same class as everything above, and the live preview is
the half of a shared session an artist actually looks at.

Held at `GestureRx::apply`, which is the one door a frame comes through, rather than
at the decode: `StrokeHead` is `stark-model`'s wire form and has no opinion about
what a renderer can use (§2).

**What this suggests for the next pass.** The three items that found real defects —
**D7**, **D8** and **A3** — were all the same move: take a rule the codebase already
states, and ask it somewhere nobody had. The rule here was §21.5's *one funnel*, and
it turns out to have had two doors all along.

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

**As built** (`f7639e5`): `stark-model/tests/action_kinds.rs`, seven tests, no GPU.
The engine keeps its own `slot` — two exhaustive matches in two crates is not
duplication to remove, since each fails to compile when a variant appears, which is
the entire point of both, and neither crate can see the other's test.

It paid for itself immediately: **D7** is a defect the review missed, and the first
run of `the_funnel_leaves_no_action_holding_a_number_a_shader_cannot_use` caught it.
That test asks its question of the whole action's `Debug` output rather than field by
field, deliberately — a per-field list is one a new field can be left out of, and
what is being checked is a class.

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

**As built** (`e653851`): moved to `stark-engine::command`, beside the command
vocabulary that is the only thing which ever asked what tool was in hand. Thirty-odd
files across four crates changed an import and nothing else, which is the honest
measure of how much the model was using it.

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

**As built** (`e653851`), and **the review was wrong about a third of this item.**
`Ellipse` and `principal_axis` do *not* have both callers in the engine: `guide`
landed in the model whole (§20.5) and reads an ellipse off the quadratic part of a
conic (§20.7), which is fifteen uses in `document/guide.rs`. The review repeated
their doc comments instead of checking them — the same class of defect as **D6**,
found the same way. The comments now say what is true, and the two types stay.

The rest of the item was real and is done. The five genuinely shader-facing items
had **zero** callers in `stark-model` outside `geom.rs` itself and are with `gpu`
now: the interior UVs, `MASK_TEX` and `mask_tex_origin` in `gpu::tile`, which is
already the module about tile textures, and `lasso_edges` in `gpu::selection`
beside the shader that reads its layout — along with the only test it had, which
the model was running for a function the model never called.

`TILE_APRON`, `TILE_TEX` and `TILE_SIZE` stay, and `geom`'s header now says why
rather than leaving it to look like an oversight: the model's own quantization is
written against them (`fill_bounds`' reach, `image_tiles`, `tile_box`), because a
box has to be padded by what a pass reads past it before anyone can ask which tiles
it touches — and `TILE_SIZE` is *derived* from `TILE_TEX`, so the three are one fact
and cannot be split down the middle.

## Sweeps

### S1. `tracing` is an unused dependency

`Cargo.toml:30`. No `tracing::`, no `use tracing`, no `info!`/`warn!`/`debug!`/
`instrument` anywhere in `src/`. It is in the wasm payload and the build graph for
nothing. One line.

**As built** (`e653851`): one line.

### S2. Two heap allocations per logged action, for the life of the history

`Footprint` is `{ reads: Vec<Resource>, writes: Vec<Resource> }`
(`footprint.rs:175`), and every `Logged` holds one from push to drop. Almost
every footprint has at most two reads and four writes; `Resource` is 32 bytes.

A `SmallVec<[Resource; 4]>` removes roughly twenty thousand allocations from a
ten-thousand-action log and — the part that matters more — makes `conflicts`'s
nested scan cache-resident. `conflicts` is the hot one: `history` builds the
centralizer once per removal and then asks it about *every* later action, which
is the whole reason the footprint is cached in `Logged` in the first place.

**Measured** (`f7639e5`), and the answer is no.

`a_footprint_stays_small_enough_for_a_nested_scan` in the new test file prints the
numbers and now pins them. `size_of::<Resource>()` is 32 and `size_of::<Footprint>()`
is 48. Across all thirty-one kinds the widest claim is **2 reads and 7 writes**
(`MergeLayerDown`, which names everything about both layers); everything else is at
most 2 and 3. The one kind that scales with the document is `DuplicateLayer`, and
that is exactly what `Resource::Layer` was introduced to collapse.

So the two allocations are **once per commit**, amortized against the GPU work a
commit already does — not in `conflicts`' loop, which is a scan over at most nine
32-byte values behind one pointer hop. Inline storage would trade roughly 130 bytes
per logged action, plus a dependency the crate does not otherwise want, for removing
two mallocs from a path that already submits a command buffer.

The measurement stays as an assertion rather than a printout, because the thing
worth guarding is the *premise*: a future action whose footprint wanted a dozen
resources would not be wrong, but it would want `Resource::Layer`'s treatment, and
the test is where that is said.

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

**Still open, deliberately.** Unlike **S2** this one cannot be measured from inside
the model: what it turns on is how often `history` clones a `Logged` rather than
borrowing it, which is a fact about the other crate and about a real editing session.
The place to find out is the in-app Timing Stats probe (§7.1) on a document with long
strokes, not a unit test — and it is worth doing after a session that has felt slow
rather than on spec.

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

Every gate the project names, after the last commit on this branch: `cargo fmt
--all --check`; `cargo clippy --workspace --all-targets -D warnings` in **both**
configurations, the default and `--no-default-features --features
stark-net/webrtc`; `cargo nextest run --workspace` — 1115 tests, all passing;
`cargo test --workspace --doc`; and `cargo check -p stark-ui --target
wasm32-unknown-unknown`.

**Not a golden moved**, which is the assertion that matters most here. The funnel
runs on replay, so a clamp that pulled an ordinary value would have shifted every
reference image — and the whole point of `every_default_is_already_sanitized` is to
say that in the suite rather than in a paragraph.

### What was *not* reproduced

The review said the two behavioural claims — the `stop_c[i]` panic in **D1** and the
shader-side NaN in **D3** — were read off declarations rather than observed, and
recommended reproducing them before fixing. **That was not done.** Both fixes went
in on the reading, and the reason is that the reading got sharper rather than
weaker: `fill.wesl:65` declares `array<vec4<f32>, 16>`, the mirror generator turns
that into `[[f32; 4]; 16]`, and `gpu/fill.rs:194` indexes it by `enumerate()` — there
is no arrangement of those three facts in which a seventeenth stop is not an index
past the end.

What that leaves genuinely unverified is narrower than the original claim and worth
stating plainly:

- **D1** — that the panic *fires* rather than being unreachable for some other
  reason (a caller that caps stops before this point, say). The fix makes it
  unreachable either way, so this can only be checked by reverting.
- **D3** — what `selection.wesl` actually *does* with a NaN vertex on this adapter
  versus another. The divergence was the finding; it is now unreachable through the
  op, but the shader's behaviour is still unknown, and it would be worth knowing if
  another path ever feeds that buffer.
- **D7** — the same, for an infinite feather. The arithmetic is written out in
  `at_least_zero`'s doc and follows from `max(feather, 1)`, but nobody has seen the
  half-selected plane.

None of the three is load-bearing for the fixes, which are all "hold the invariant
the type already claimed". They are load-bearing for the *severity* the review
assigned them, and that is the part still taken on reading.

## What is left

Three things, in the order I would take them:

1. **A1's newtype** — `Srgb([f32; 3])`, which turns four `map(clamp01)` calls into
   a type that cannot hold an out-of-range color. Deferred deliberately: it changes
   the wire representation of four payloads at once, so it wants its own commit and
   its own older-shape test rather than the tail of a correctness commit. Note it
   covers four of the five sites, not five — `BrushParams::color` is RGBA and would
   need its own newtype or to stay as it is, which is worth deciding before starting
   rather than discovering halfway.

   **`MattePaint` and `Parcel` merging rides with it.** They are word-for-word the
   same type — a solid or a ramp on an axis, both reaching `ramp_common` — with two
   wire shapes written at different times, which is now said at
   `MattePaint::sanitized` rather than only noticed here. Their two `sanitized`
   implementations are the same fifteen lines twice.

2. **S4** — measured in the app, not in a test, and only after a session that has
   felt slow.

Nothing else is outstanding.
