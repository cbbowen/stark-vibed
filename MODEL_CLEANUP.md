# stark-model cleanup

Findings from a review of `crates/stark-model` — the document crate (§2): the
action log, its vocabulary, the save format and the presence wire.

The crate is disciplined. Invariants are stated where they are held, the
wire-format hazards are enumerated one bump at a time, and the
exhaustive-match-as-tripwire device (`minted_layers`, `action_content`,
`Filter::resamples`, `Modulations::all`) is used consistently and for the right
reason. So almost nothing below is a missing rule. What is below is places where
a rule the crate *states* is not actually enforced, and places where a cost is
paid in a loop that does not need to pay it.

Ordered by what it costs to leave alone:

| # | Finding | Cost of leaving it |
|---|---|---|
| [1](#1-unbundled_content-conflates-the-two-asset-bags) | `unbundled_content` conflates the two asset bags | a document that silently replays wrong, into stored pixels |
| [3](#3-four-strategies-for-one-invariant-and-two-do-not-survive-serde) | Deserialization bypasses two of the four validation strategies | a file or peer writes values no slider can produce |
| [4](#4-the-commutation-hot-path-rebuilds-footprints) | The commutation hot path rebuilds footprints | undo across a warp is O(actions × 3249 nodes) |
| [2](#2-tilerectintersects-guards-emptiness-on-x-but-not-y) | `TileRect::intersects` guards emptiness on x only | latent §12.6 false negative |
| [7](#7-from_bytes-decompresses-unbounded) | `from_bytes` decompresses unbounded | a peer can exhaust memory |
| [9](#9-two-pieces-of-inert-scaffolding), [10](#10-wire_versions-docstring-is-a-150-line-changelog) | Inert scaffolding; a changelog on a `const` | §1 violations, cheap to fix |
| rest | Code health | maintenance drag |

---

## Correctness

### 1. `unbundled_content` conflates the two asset bags

[`crates/stark-model/src/content.rs:141-156`](crates/stark-model/src/content.rs)

`unbundled_content` flattens `assets` and `surfaces` into one
`HashSet<AssetId>`, then filters needs by `need.content()` alone. So
`AssetNeed::Ground(X)` reports as **bundled** when `X` is present only in the
*brush* bag, and vice versa.

This is the exact confusion the format was split to prevent.
`DocumentFile::surfaces` is keyed separately from `assets` because, in its own
words, "the two decode differently — a mask is luminance × alpha, a ground is
channel 0 — so a single bag would hand each store the other's bytes to
reinterpret" ([`io.rs:243`](crates/stark-model/src/io.rs)). Ids are content
hashes, so the same image imported once as a stamp and once as a ground collides
for real rather than theoretically.

The consequence is the one `DocError::MissingContent` exists to rule out
([`error.rs:62-76`](crates/stark-model/src/error.rs)): the bill comes back short,
nothing refuses the replay, and every stroke made on that ground deposits through
the flat stand-in — into tiles that no later arrival un-bakes (§6.4).

**Fix.** Match on the need's kind and check the corresponding bag. `AssetNeed`
already distinguishes them; only this function throws the distinction away.

### 2. `TileRect::intersects` guards emptiness on x but not y

[`crates/stark-model/src/geom.rs:144-151`](crates/stark-model/src/geom.rs)

The predicate tests `min.0 <= max.0` for both rects, then overlap on both axes —
but never `min.1 <= max.1`. A rect empty only in y intersects everything.

`union` handles both axes ([`geom.rs:165`](crates/stark-model/src/geom.rs)), so
the two disagree about what "empty" means. Not reachable today: `covering` is the
only constructor and every caller passes `lo <= hi`. But `min`/`max` are public
fields, and this is a **footprint** predicate — the one place §12.6 says a false
negative cannot be survived, because the fast path would splice on a lie and no
pixel could show it.

**Fix.** One `TileRect::is_empty()`, used by `intersects`, `union` and `contains`
alike. Rules out the class rather than patching the instance (§1).

### 3. Four strategies for one invariant, and two do not survive serde

The crate holds "a value that reaches a shader is a number in range" four
different ways:

| Type | Strategy | Survives deserialization? |
|---|---|---|
| `Gradient` | ctor + `#[serde(try_from)]` funnel | yes |
| `Filter`, `BlendMode` | `sanitized()`, called at two named engine sites | yes |
| `SelectionOp`, `FillOp` | constructor only, public fields | **no** |
| `BrushParams`, `BrushDynamics` | nothing; defended per use site | **no** |

[`SelectionOp::at`](crates/stark-model/src/document/selection.rs) pins
`opacity: 1.0` for `SelectionShape::All` (its doc explains at length why that
state must not exist), clamps opacity and floors feather — and
`#[derive(Deserialize)]` walks straight past all three. A file or a peer can
carry `All` at opacity 0.5, `feather: NaN`, `opacity: 5.0`.
[`FillOp::with_paint`](crates/stark-model/src/document/fill.rs) is the same, and
does not clamp `feather` even in the constructor.

`brush.rs`'s module header claims "the values that arrive from files, presets and
peers are clamped on the way in rather than trusted"
([`brush.rs:20-22`](crates/stark-model/src/document/brush.rs)). Clamping exists
inside `Modulation::factor`, `taper_px` and `elongation`. `BrushDynamics`' five
axes — `add`, `lift`, `deposit`, `charge`, `bleed` — are not bounded anywhere in
the workspace.

Verified: `sanitized()` is implemented for `Filter`, `ColorAdjust`,
`ChromaticAberration` and `BlendMode`, and for nothing else.

**Fix.** Converge on one strategy. `Gradient`'s is strongest because it is
structural — the type cannot exist in a bad state, and §1 prefers ruling out a
class to checking for its instances — at the cost of a `try_from` per type.
`sanitized()` is cheaper and already has engine call sites; its gap is that it
covers two of the six types that need it. Either way the win is that "which types
are validated, and where" stops being something a reader reconstructs per type.

---

## Performance

### 4. The commutation hot path rebuilds footprints

[`crates/stark-model/src/document/fold.rs:117-125`](crates/stark-model/src/document/fold.rs)

Confirmed against the `history` crate (`src/lib.rs:1066-1069`): `for_action`
builds the centralizer **once**, then `commutes(other)` runs inside a
`take_while` over every consecutive later action. The impl calls
`footprint(&other.0)` inside that loop, so each comparison pays:

- two `Vec<Resource>` heap allocations — every action, every time;
- a full walk of the control-point path (`stroke_rect`,
  [`footprint.rs:217`](crates/stark-model/src/document/footprint.rs));
- for `TransformWarp`, **an entire fine-lattice build**. `map.image_aabb()`
  ([`footprint.rs:365`](crates/stark-model/src/document/footprint.rs)) calls
  `lattice()`, which for an 8×8 control grid solves 57×57 nodes with several
  intermediate `Vec`s.

So an undo across a warp action is O(actions × 3249 nodes) — for a footprint that
is a pure function of an action that is not changing.

**Fix, cheapest first.**
1. Give `Footprint` an inline buffer (`SmallVec`-shaped). Every arm but
   `DuplicateLayer` has ≤ 7 resources.
2. Memoize the expensive arms — `WarpMap::image_aabb` is the only one that is
   not O(size of the action).
3. Structurally: compute an action's footprint **once at commit** and carry it
   beside the action. It is a pure function of the action, so a cached copy
   cannot drift — which is the property that makes the cache safe here and would
   not make it safe for a state-dependent quantity.

### 5. `WarpMap::eval` and `basis` rebuild the delta grid per call

[`warp.rs:308-352`](crates/stark-model/src/document/warp.rs)

Both allocate the full `deltas` vector on every call; `basis` allocates three
more. A frontend drawing mesh curves calls `eval` hundreds of times per frame.
The two also carry a verbatim copy of the `locate` closure.

**Fix.** A `WarpMap::prepared() -> Deltas` view, computed once per drag and
borrowed by both — which removes the allocations and the duplicated `locate`
together.

### 6. `Compression::best()` on every save

[`io.rs:267`](crates/stark-model/src/io.rs)

Level 9 is markedly slower than level 6 for near-identical size on smooth,
highly-compressible path data. Saving is user-facing latency.

### 7. `from_bytes` decompresses unbounded

[`io.rs:294`](crates/stark-model/src/io.rs)

`read_to_end` on an attacker-supplied deflate stream. `stark-net` moves logs
between peers, so this is reachable from the network, not only from a file the
user chose. A `Read::take(limit)` costs one line, and the limit is a documented
property of the format rather than a magic number.

---

## Code health

### 8. `ViewTransform` breaks the crate's own boundary rule

[`lib.rs:21-25`](crates/stark-model/src/lib.rs) states the mechanical test: "if a
type is serializable it is a fact about the document and lives here; if it holds
a tile it is a cache and lives there."

`ViewTransform` is not `Serialize`, and its own doc says it "is session state and
is never historized" ([`geom.rs:246`](crates/stark-model/src/geom.rs)). It is 380
lines of this crate, and `stark-net` — the consumer the split was made for (§2) —
has no use for it.

`geom.rs` is really three modules under one name: the tile grid (document
vocabulary), the view transform (session), and the mask/lasso helpers
(engine-facing). Splitting the view out would make the crate doc's rule *true*
rather than aspirational — the same move §2 already made for `ColorSpaceId::make`
and `SelectionOp::shader_params`.

### 9. Two pieces of inert scaffolding

§1: "If a field, slider or shader hook cannot yet change a pixel, leave it out."

- **`CanvasMeta::tile_size`** ([`io.rs:201`](crates/stark-model/src/io.rs)) is
  written on every save and **never read anywhere in the workspace** — grepped;
  the only occurrences are its declaration and its default. A file claiming
  `tile_size: 999` loads silently and renders wrong. Validate it on load, or drop
  it.
- **`BuildId::app_version`** ([`io.rs:193`](crates/stark-model/src/io.rs)) is
  `env!("CARGO_PKG_VERSION")` of *stark-model*, which is `0.0.0` and is never
  bumped. The field exists so "cross-build replay differences are explainable"
  (§8) and records the same constant in every file ever written. Either wire it
  to something that moves, or take it out until there is a version to record.

### 10. `WIRE_VERSION`'s docstring is a 150-line changelog

[`io.rs:33-180`](crates/stark-model/src/io.rs)

The content is the best record in the tree of why each format break was worth
taking — versions 5, 9 and 11 in particular are load-bearing arguments. But it
hangs off a `const u32`, so anyone who hovers the constant gets the whole history.

**Fix.** Move it to `docs/engine.md` §8 and have the constant cite the section —
the convention §1 already sets ("cite sections, not line numbers").

### 11. The crate-root re-exports contradict `document/mod.rs`'s own argument

`document/mod.rs` refuses to publish its submodules because "publishing both
gives every type two paths, `document::BrushParams` and
`document::brush::BrushParams`, with nothing choosing between them"
([`mod.rs:5-7`](crates/stark-model/src/document/mod.rs)).

[`lib.rs:40-53`](crates/stark-model/src/lib.rs) then gives `LayerId`,
`SelectionMode`, `SelectionOp` and `SelectionShape` exactly two paths each, and
the chosen subset looks arbitrary: `LayerId` but not `ActionId`, `SelectionOp`
but not `FillOp`. Pick one level and hold it.

### 12. Smaller items

- **`DocError::Serialize(String)` / `Deserialize(String)`**
  ([`error.rs:21-25`](crates/stark-model/src/error.rs)) stringify postcard errors
  instead of `#[from]`, losing the source chain every other variant preserves.
- **`Lattice` has public fields and methods that panic on a hand-built one**:
  `positive()` does `0..self.ny - 1` (underflows at `ny == 0`), `aabb()` indexes
  `pts[0]`. Private fields plus a constructor rule that out structurally.
- **`Modulations`' seven `Option` fields and seven near-identical accessors**
  ([`brush.rs:379-490`](crates/stark-model/src/document/brush.rs)) collapse to
  `[Option<Modulation>; N]` keyed by a `ModTarget` enum, keeping the
  exhaustiveness tripwire via `ModTarget::ALL` — the device `Prop::ALL` already
  uses.

---

## One invariant worth pinning

`stroke_pad`'s `1.5` factor
([`footprint.rs:206-208`](crates/stark-model/src/document/footprint.rs)) is
documented as covering `√2` for a square stamp swept at an angle. It also has to
cover the **bleed** stencil, whose taps reach `BLEED_REACH_MAX = 0.5 · radius`
(`stark-engine`'s `gpu::stroke::dynamics::bleed`) — and `1.0 + 0.5 = 1.5` is
unlikely to be a coincidence.

The claim appears to hold today: a tap landing outside the sweep has `w_n = 0`
and carries nothing, so bleed's *effect* stays inside `√2 · radius · elongation`,
under the pad. But the margin over the justification the doc actually gives is
only `0.086 · radius · elongation + 4`, the constant that makes it work lives in
`stark-engine` where this crate cannot see it, and
`a_stretched_stroke_claims_the_tiles_its_drawn_out_tip_can_reach` ties the pad to
elongation only.

If the bleed reach ceiling is ever raised, nothing fails. It is a silent §12.6
break — two peers commuting a pair of strokes the paint says overlap, with no
pixel able to show which order ran.

**Fix.** An engine-side test asserting `stroke_pad ≥ tip reach + bleed reach`.
Engine-side because that is the only crate that can name both constants — the
same reason `ColorSpaceId::make` ended up there (§2).
