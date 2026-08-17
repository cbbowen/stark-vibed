# stark-model cleanup

Findings from a review of `crates/stark-model` — the document crate (§2) — and
what was done about each. **Status as of the `model-cleanup` branch.**

The crate was already disciplined: invariants stated where they are held, the
wire-format hazards enumerated one bump at a time, and the
exhaustive-match-as-tripwire device used consistently. So almost none of this was
a missing rule. It was places where a rule the crate *states* was not actually
enforced, and places where a cost was paid in a loop that did not need to pay it.

| # | Finding | Status |
|---|---|---|
| [1](#1-unbundled_content-conflated-the-two-asset-bags) | `unbundled_content` conflated the two asset bags | **fixed** |
| [2](#2-tilerectintersects-guarded-emptiness-on-x-but-not-y) | `TileRect::intersects` guarded emptiness on x only | **fixed** |
| [3](#3-four-strategies-for-one-invariant-two-of-which-did-not-survive-serde) | Deserialization bypassed two of four validation strategies | **fixed** |
| [3b](#3b-the-gates-spelled-their-bounds-clamp) | *(found while fixing 3)* the gates spelled their bounds `f32::clamp` | **fixed** |
| [4](#4-the-commutation-hot-path-rebuilt-footprints) | The commutation hot path rebuilt footprints | **fixed** |
| [5](#5-warpmapeval-rebuilt-the-delta-grid-per-call) | `WarpMap::eval` rebuilt the delta grid per call | **fixed** |
| [6](#6-compressionbest-on-every-save) | `Compression::best()` on every save | **fixed** |
| [7](#7-from_bytes-decompressed-unbounded) | `from_bytes` decompressed unbounded | **fixed** |
| [8](#8-viewtransform-broke-the-crates-own-boundary-rule) | `ViewTransform` broke the crate's own boundary rule | **fixed** |
| [9a](#9a-canvasmetatile_size-was-written-and-never-read) | `CanvasMeta::tile_size` written, never read | **fixed** |
| [9b](#9b-buildidapp_version-records-a-constant) | `BuildId::app_version` records a constant | **not done — see below** |
| [10](#10-wire_versions-docstring-was-a-150-line-changelog) | `WIRE_VERSION`'s 150-line changelog docstring | **fixed** (→ §8.1) |
| [11](#11-the-crate-root-re-exports-contradicted-documentmodrs) | Crate-root re-exports contradicted `document/mod.rs` | **fixed** |
| [12](#12-smaller-items) | `DocError` source chain, `Lattice` panics, `Modulations` shape | **2 of 3 fixed** |
| [∗](#the-invariant-that-turned-out-not-to-need-pinning) | `stroke_pad` ↔ `BLEED_REACH_MAX` | **investigated — no defect** |

---

## Correctness

### 1. `unbundled_content` conflated the two asset bags

`unbundled_content` flattened `assets` and `surfaces` into one
`HashSet<AssetId>`, then filtered needs by `need.content()` alone — so
`AssetNeed::Ground(X)` reported as bundled when `X` was present only in the
*brush* bag.

This was the exact confusion the format was split to prevent: the bags are keyed
apart because their bytes decode differently (a mask is luminance × alpha, a
ground is channel 0), and ids are content hashes, so one image imported as both a
stamp and a ground carries one id in two bags that cannot stand in for each
other. The consequence was the one `DocError::MissingContent` exists to rule out:
the bill came back short, nothing refused the replay, and every stroke made on
that ground deposited through the flat stand-in — into tiles no later arrival
un-bakes (§6.4).

**Fixed** by matching on the need's kind against the corresponding bag,
exhaustively, so a kind added later must name the bag that answers it.

### 2. `TileRect::intersects` guarded emptiness on x but not y

It tested `min.0 <= max.0` for both rects, then overlap on both axes — never
`min.1 <= max.1`. A rect empty only in y intersected everything, while `union`,
from the same inputs, treated it as the identity. The fields are public, so the
disagreement was reachable, and this is the predicate the commutation gate rests
on: §12.6 survives a rect claiming too much and cannot survive the reverse.

**Fixed**: one `TileRect::is_empty()`, asked by everything that has to treat an
empty rect as empty.

### 3. Four strategies for one invariant, two of which did not survive serde

| Type | Strategy | Survived deserialization? |
|---|---|---|
| `Gradient` | ctor + `#[serde(try_from)]` funnel | yes |
| `Filter`, `BlendMode` | `sanitized()` at named sites | yes |
| `SelectionOp`, `FillOp` | constructor only, public fields | **no** |
| `BrushParams`, `BrushDynamics` | nothing | **no** |

`SelectionOp::at` pins `opacity: 1.0` for `SelectionShape::All` — its doc spends a
paragraph on why that state must not exist — and `#[derive(Deserialize)]` walked
straight past it, along with the feather floor and the opacity clamp. `FillOp` the
same. `brush.rs`'s header claimed values from files and peers "are clamped on the
way in"; `BrushDynamics`' five axes were not bounded anywhere in the workspace.

**Fixed** by converging on one shape:

- the two ops funnel deserialization through their own constructors, via a
  field-for-field `Raw` mirror that leaves the encoding untouched (pinned by a
  test, since postcard writes fields in order with no names);
- `BrushParams` grows a `sanitized()`, which clamps **only where this crate
  already states a range** — the pickup axes, tooth, hardness and color in
  `[0, 1]`, stretch at its own saturation point. Radius, flow, drain, charge and
  the tapers are required to be finite and non-negative and nothing more, because
  their ceilings are a frontend's slider ends rather than facts about the
  quantity, and clamping a document to a bound this crate does not own would
  rewrite brushes that were never wrong;
- `ActionKind::sanitized()` gathers all of it into one exhaustive match with no
  `_` arm, and the engine sanitizes an *action* at mint and at state entry rather
  than remembering which payloads have knobs. The three payload-level calls that
  list had accumulated are gone.

### 3b. The gates spelled their bounds `f32::clamp`

Found while writing the tests for 3, and worth its own row because it is the
sharper half. `FillOp::with_paint` and `SelectionOp::at` bounded their values with
`f32::clamp`, which **returns the NaN** — both of its comparisons against one are
false. So the gates caught every hostile value except the one that matters most:
a NaN opacity reaches a fullscreen inversion of the coverage law.

`brush.rs` already had `clamp01` (`max`-then-`min`) with the rationale written
out. It is `pub(crate)` now and is the document's single NaN policy.

### 7. `from_bytes` decompressed unbounded

`read_to_end` on an attacker-supplied deflate stream, reachable from the network
(§12.4), where a few KB can name as many GB as it likes.

**Fixed**: the body is `take`n one byte past a 256 MiB cap and refused as
`DocError::TooLarge` — roughly two orders of magnitude of headroom over anything
a session produces.

---

## Performance

### 4. The commutation hot path rebuilt footprints

`history` builds a centralizer once per removal and then asks it about *each*
later action (`lib.rs:1066-1069`), so `Centralizer::commutes` rebuilt the other
action's footprint per comparison: two `Vec` allocations always, a walk of the
whole control-point list for a stroke, and — for `TransformWarp` — an entire
57×57 fine-lattice solve via `WarpMap::image_aabb`. An undo across a warp was
quadratic in the log for an answer that cannot change.

**Fixed**: `Logged` carries its footprint, computed once at push. Deliberately
computed *after* sanitizing — a footprint built from the raw action could
disagree with the pass wherever a clamp pulls a value down, which is the §12.6
direction.

### 5. `WarpMap::eval` rebuilt the delta grid per call

Both callers call it in a loop: finding the grabbed point is 81 coarse probes
plus six refinement passes of 25, and the mesh overlay is ~400 evaluations a frame
while dragging. `eval` and `basis` also carried a verbatim copy of `locate` each.

**Fixed**: `prepared()` computes the grid once and returns a borrowed view; `eval`
is that view for a single point, so there is one implementation rather than two
that could drift. Asserted bit-for-bit, because §16.4's identity invariant is
stated bitwise and watertightness rests on shared points agreeing exactly.

### 6. `Compression::best()` on every save

**Fixed**: level 6. Level 9 spends a large multiple of the time for a fraction of
a percent on smooth path data, and the bundled PNGs are incompressible either way.

---

## Code health

### 8. `ViewTransform` broke the crate's own boundary rule

`lib.rs` states a mechanical test — `Serialize` means it is the document,
holds a tile means it is a cache — and the Cargo.toml charter says "the document,
and nothing else". `ViewTransform` is neither: it is session state, as its own doc
always said, and `stark-net` (the consumer the split was made for) was compiling
four hundred lines it can never use.

**Fixed**: it is `stark-engine`'s `view` module now. The tile grid stays in the
model, because *that* is document vocabulary — a footprint quantizes against
`TILE_SIZE` and a saved log is addressed in it. The split runs between the canvas
and the eye, which is the line §18.1.2 already draws.

### 9a. `CanvasMeta::tile_size` was written and never read

Recorded on every save, read by nothing in the workspace. Every tile boundary
moves with it, so a file from a build with another stride loaded clean and
rendered wrong — precisely the reproducibility question the field was added to
answer.

**Fixed**: checked on load, refused as `DocError::TileSize`.

### 9b. `BuildId::app_version` records a constant

**Not done, deliberately.** It is `env!("CARGO_PKG_VERSION")` of *stark-model*,
which is `0.0.0` and never moves, so the field that exists to explain cross-build
replay differences says the same thing in every file ever written.

Neither fix is a code change I should make unilaterally:

- **removing it** is a wire-format break (version 13) worth nothing on its own,
  and §8's rule is that breaks are paid for in batches;
- **wiring it to something that moves** needs either a build script — which this
  crate's charter forbids, and which is why `stark-assetid` exists as a separate
  crate at all — or real crate versions, which is a workspace-policy decision.

The honest intermediate would be for the *engine* to supply it at save time, but
`stark-engine` is also `0.0.0`, so that changes nothing today. Left as-is and
flagged rather than papered over.

### 10. `WIRE_VERSION`'s docstring was a 150-line changelog

**Fixed**: it is §8.1 of [docs/engine.md](docs/engine.md) — a table of every bump
plus the four worth more than a row — and the constant cites the section, per §1.
What stays at the constant is the rule the history is evidence for.

### 11. The crate-root re-exports contradicted `document/mod.rs`

`document` publishes a curated list over crate-private submodules precisely so
each type has one path; the crate root then lifted four of them out again, giving
`LayerId` and `SelectionOp` two paths while `ActionId` and `FillOp` had one.

**Fixed**: not one call site in the workspace took the short path, so the four are
gone rather than the other twenty added. `lib.rs` now states which modules are
flat preludes and why `document` is not.

### 12. Smaller items

- **`DocError::Serialize`/`Deserialize`** stringified the postcard error, losing
  the `source()` chain every other variant preserves. **Fixed** — they carry it.
  Still two variants rather than one `#[from]`: which direction failed is the
  useful half of the message.
- **`Lattice`** had public fields and two methods that panic on a lattice they
  could not have produced (`0..ny - 1` underflows, `pts[0]` indexes out of
  bounds). **Fixed** — private fields with accessors, and `WarpMap::lattice` the
  only way to one, so there is no degenerate lattice left to defend against.
- **`Modulations`' seven `Option` fields and seven near-identical accessors**
  would collapse to `[Option<Modulation>; N]` keyed by a `ModTarget` enum.
  **Not done.** It is the one item on this list that is purely cosmetic — the
  `all()` destructure already provides the exhaustiveness tripwire the array
  would, so nothing is unsound or slow — and it reaches into `stark-ui`'s
  `ModRow`, which is a second enum over the same seven targets and the real
  duplication. Worth doing as one change that unifies both, which is a wider
  edit than this branch should carry.

---

## The invariant that turned out not to need pinning

`stroke_pad` pads a stroke's claim by `radius · 1.5 · elongation + 4`, justified
in its doc as covering the `√2` a square stamp sweeps to at an angle. The review
flagged that `1.0 + 0.5` looked too much like the tip plus `BLEED_REACH_MAX` to be
a coincidence, and that the two constants live in crates that cannot see each
other.

I wrote the test asserting `pad ≥ tip reach + bleed reach`. **It failed** at
`radius = 500` — 957 px of reach against 754 of pad.

**The test was wrong, not the pad.** `dynamics.wesl` weighs every bleed tap by
`min(w_t, w_n)` — this texel's mobility and its neighbour's — and `bleed_weight`
writes `w = 0` for any texel outside the sweep. So a texel outside is never
written (its own `w_t` zeroes all of its fluxes) and a tap reaching out of the
sweep carries exactly nothing. The reach sets how far *within* the footprint paint
is carried, never how far the footprint extends: a no-flux wall, at every scale.

So the coincidence is a coincidence, the `1.5` covers `√2` and nothing else, and a
bleed reach raised past it would still be contained. What *would* break the pad is
the wall coming down — a shader invariant, checkable only where the shader is, and
not something a host-side test can pin. The reasoning is now on `stroke_pad` so
the next reader who finds the arithmetic does not have to re-derive the alarm in
order to dismiss it.
