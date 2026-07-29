# Transform — moving paint under an affine

Move / scale / rotate / flip / skew of the selected region of a layer
(MISSING_FEATURES.md §0.3). Today a selection can only *mask*; this makes it
*hold* paint. The document specifies the engine: the action, the resampling
semantics, and the GPU passes. The interactive gesture (handles, snapping,
modifier keys) is deliberately out of scope — it commits exactly one of these
actions on release, the same shape as the frame drag
([FRAME_DESIGN.md](FRAME_DESIGN.md) §7).

It is written against the code as it stands
([selection.rs](crates/stark-core/src/document/selection.rs),
[tile.rs](crates/stark-core/src/gpu/tile.rs),
[integrate.wesl](crates/stark-shaders/src/shaders/integrate.wesl),
[dynamics.wesl](crates/stark-shaders/src/shaders/dynamics.wesl)), and its central
choices are forced by invariants that code already holds.

## 1. The action

```rust
ActionKind::Transform { layer: LayerId, affine: Affine2 }
```

One new variant (appended last — postcard encodes enums by index), carrying a
[`glam::Affine2`](crates/stark-core/src/geom.rs) — six floats in the log.
Semantics: **cut the paint under the author's selection on `layer`, apply the
affine, stack the result back onto what remained — and carry the author's
selection mask along with it.** A universal selection means the whole layer, so
"move layer" and "move selection" are one action, not two.

Everything rides existing machinery:

- **The author's mask gates it**, read from `self.id.actor` like a stroke's
  (DESIGN.md §6.8, PEER_DESIGN.md §3) — a collaborator's lasso never decides
  what *your* transform moves.
- **Undo** is timeline navigation solo and `Undo(target)` shared, untouched.
- **Replay** is exact: the GPU passes are pure functions of the tiles, the mask
  and six floats.
- **A matte layer refuses it**, exactly as it refuses strokes (its geometry
  already moves via `SetMatteRect`).
- **Degenerate and oversized transforms are rejected deterministically** — a
  near-singular matrix (paint would vanish into a line), or a destination that
  would need more than `MAX_TRANSFORM_TILES` rewrites. The bound is a pure
  function of the action and the state, so peers and replays agree, exactly as
  `MAX_SELECTION_TILES` works (§6.8). The document is left unchanged.

The selection moves because the alternative is wrong twice: the outline would sit
over paint that is no longer there, and a second nudge would cut a stale region.
`outside` is unchanged by any affine (an affine of the whole plane is the whole
plane), so the flag machinery is untouched.

## 2. The cut is a lift, not an erase

What does "remove the selected fraction `m` of a texel's paint" mean in the
normalized representation (DESIGN.md §6.1: color = `(latent·op, op)`, aux =
`height`)? The dynamics loop already answers for the eraser: **height leaves;
the remaining paint's latent and per-unit opacity are untouched** — the source
fades because its *thickness* drops, never because its alpha is scaled.

```
h₀ = h · (1 − m)          // the cut: thickness leaves
parcel = (latent, op, h·m) // what the transform now holds
```

The rejected alternative — scaling the premultiplied color by `(1 − m)` too —
fails the invariant that makes this whole feature testable (§4): cutting and
pasting back in place must be the identity. Under the lift law the masses
recombine exactly (`op·h·(1−m) + op·h·m = op·h` at every feather value `m`);
under color scaling the optical mass recombines as `(1−m)² + m²` of itself and a
feathered identity transform visibly thins the paint at its own edge.

## 3. The paste is a parcel deposit

The moved paint lands as a **parcel of existing paint**, so it stacks by the law
the dynamics deposit already uses (`blend_latent`, dynamics.wesl): heights add,
latents blend weighted by visible alpha (`1 − exp(−K·opacity·height)` — the same
translucent-slab constant as the media pass), the new per-unit opacity is total
optical mass over total height. Thick moved paint covers what is under it; a
thin moved glaze tints it. That function moves to a shared
`paint_common.wesl` module so the two consumers cannot drift.

Notably this is *not* the integrate pass's premultiplied-"over": a stroke's
scratch alpha is swept coverage (already an exposure), while a moved parcel's
`op` is per-unit opacity — over-blending by it would let a zero-thickness ghost
occlude real paint.

Scaling deliberately does **not** conserve total paint: doubling a region's size
quadruples its paint, keeping every texel's appearance. That is what a transform
tool means everywhere; conserving mass would visibly thin paint as it grows.
Pure moves conserve height exactly.

## 4. Resampling, and the invariants that pin it

The transformed paint is resampled **once**, bilinearly, forward-rasterized:
each selected source tile's *interior* becomes one quad under the affine, drawn
into each destination tile it touches, sampling the source tile's textures (the
1-px apron feeds the filter's edge taps — §6.4 built exactly this). Source
interiors tile the plane, so their images tile the transformed plane: quads
share transformed edges bitwise (corner arithmetic is exact in f32) and the
rasterizer's fill rules make coverage watertight — **every destination texel
receives at most one parcel**, which is what lets the deposit be a single
pass with no blending and no order dependence.

Three exactness properties are the acceptance tests, all consequences of
sampling at texel centers falling on texel centers:

1. **Identity is a no-op.** Weights `(1,0,0,0)`, parcel = source, restack is
   algebraically the inverse of the cut.
2. **Integer translation is exact** — the painting moves without a single texel
   of resampling loss.
3. **Axis flips and quarter-turns about half-integer centers are exact**, and
   involutions compose to the identity through two separate actions.

Two implementation choices carry these, because "algebraically exact" is not
"byte-exact" on a GPU: the fragment maps its own canvas center back through the
**inverse** affine rather than interpolating a uv across the quad (attribute
interpolation plus the sampler's fixed-point uv conversion can land an
exact-center tap a fraction off), and the combine takes **exact branches at
coverage 0 and 1** — passthrough and parcel respectively — because the
render-target's f32→f16 store rounding is implementation-defined, so even
sub-half-ulp recomputation noise can flip a stored bit. What remains inexact is
exactly what must be: texels strictly inside a feather ramp are genuinely
recomputed (`h·(1−m) + h·m`) and may land one f16 ulp off — at most one display
LSB, pinned by test.

Everything else (rotation, fractional scale) is honest bilinear: one generation
of loss per commit, never compounded during a gesture because the gesture
previews from the *committed* tiles and commits once. Minification below ~½×
aliases (bilinear underfilters); an EWA or mip-chain sampler is a future quality
pass, localized entirely in the parcel shader.

Aprons stay rendered-not-copied (§6.4): every pass here is a pure function of
canvas position, so a destination tile's apron is bit-identical to its
neighbour's interior by construction and the seam test extends to transforms.

## 5. The GPU passes

Per rewritten tile, two passes (`transform.wesl`), the same CoW discipline as
every stroke — old tiles stay valid in old history versions, new tiles come from
the pool:

- **Parcel** — clear a scratch pair to zero, then for each intersecting source
  quad draw it, sampling source color / height / mask bilinearly: out =
  `(color_as_is, height·m)`. Disjointness (§4) makes draw order irrelevant.
- **Combine** — fullscreen `textureLoad` pass: cut the base by its own mask
  (`h₀ = h·(1−m)`), stack the parcel by §3, write color + aux. Base and mask
  bind per-tile textures or the 1×1 constants (§6.8's pattern), so paste onto
  virgin canvas and cut-only tiles are the same shader; a texel neither cut nor
  pasted passes through bit-exactly.

A tile whose mask is constant 1 and which no quad reaches is simply dropped from
the map — the combine would write zeros, and an all-zero tile is worse than no
tile (it would pollute `bounds` and hold pool memory).

The mask transform is one simpler pass: destination mask tiles are cleared to
`outside` and the source mask quads are drawn over, resampling coverage. Mask
tiles carry a ring of constant coverage by construction (`Selection::plan` pads),
so the boundary between rasterized and constant regions is continuous.

CPU planning (`document/transform.rs`) is pure and unit-testable, like
`Selection::plan`: classify each populated tile against the mask (untouched /
partial / fully-selected), build the quad list, intersect transformed quads with
tile rects (exact convex test, not AABB — a loose test would mint empty tiles),
and enforce the caps.

## 6. The gesture: an ellipse, not a box of handles

Most software shows a rectangle with resize grips, tacks rotation on as a knob,
and either hides skew behind a modifier or doesn't offer it. This widget is an
**ellipse** — the image of a reference **circle** under the accumulated linear
map — so the widget's own shape *shows* the transform: it stays a circle
exactly as long as the transform is a similarity (move, turn, uniform scale),
and any eccentricity *is* the distortion. A circle rather than the hull's own
aspect, precisely for that reading — a rectangle-shaped reference would start a
rectangular selection's widget as an ellipse and the shape would say
"distorted" before anything happened. Its radius is the geometric mean of the
hull's half-extents (the area of the hull's inscribed ellipse), so it stays
proportionate for elongated selections. One surface carries the whole affine
group with three gestures, chosen by where the drag starts:

| Region | Gesture | Family |
|---|---|---|
| inside | translate | `x ↦ x + Δ` |
| on the rim | rotate + uniform scale | similarity about the centre |
| outside | directional scale + skew | rank-1: `I + (Δ ⊗ d̂)/λ` |

Each shaping gesture is solved so that **the grabbed point follows the pointer
exactly** within its family — the hand feels attached to the paint, and the two
degrees of freedom of the pointer are always fully spent:

- **Rim.** The unique similarity carrying the grab to the pointer is the complex
  ratio `(p − c)/(p₀ − c)`. Its differential behaviour is exactly the spec:
  motion *tangent* to the ellipse is pure rotation, motion *normal* to it is
  pure uniform scale, anything between blends — no mode to pick, no knob.
- **Outside.** The rank-1 map `G = I + (Δ ⊗ d̂)/λ` (with `d̂` the grab
  direction, `λ` its distance) also carries the grab exactly to the pointer,
  while **pinning the diameter perpendicular to the grab**: radial pull is a
  scale along `d̂`, tangential drag is a shear, and the pinned axis is what
  makes the gesture predictable. Composing these from different directions
  (with the rim's rotations) reaches every orientation-preserving linear map —
  skew and non-axis-aligned scaling are not a bolted-on mode but the same
  vocabulary. Pulling in past the pinned axis is floored at 90% so the
  determinant cannot run through zero mid-drag.
- Hit-testing pulls the pointer back through the linear map into the reference
  circle's space, where every region test is a radius — exact at any
  deformation; the rim band stays a constant *screen* width by scaling with the
  widget's local radius. The gesture is locked at the press (crossing the rim
  mid-drag must not change what the hand is doing), a north dot marks the
  reference "up" (a rotated circle otherwise hides its rotation), and the
  cursor announces the region under the resting pointer.

State is `TransformState { anchor, radius, center, linear: Mat2 }`, affine
`x ↦ center + linear·(x − anchor)`; gestures left-compose world-space factors
onto `linear`, always recomputed from the drag's start (nothing accumulates
per-move). Untouched factors are simply absent: a pure move keeps `linear`
bit-exactly the identity — the pure-translation exactness of §4 with no snapping
heuristics — and a sub-2-screen-px jiggle in any gesture snaps back to its
start, so touching the widget never resamples by accident. The bar carries
**Flip ↔ / Flip ↕ / Done** (flips are world-axis mirrors folded into `linear`;
four mirrors cancel bit-exactly).

Everything before Done is a **lossless preview**: every change runs
`ViewCommand::PreviewTransform` — the same renderer as the commit, over the
committed tiles, into the `doc_preview` slot the frame drag uses — so the screen
shows exactly what Done will produce, a long drag resamples once, and Done
commits a single action (one undo step per gesture). The widget anchors to the
conservative analytic **hull** the selection carries through its op algebra
(`Selection::hull`, `ObservableState::selection_hull`); an unbounded selection
falls back to the painted content's bounds — which is also how "move the whole
layer" arrives for free. While the mode is active a full-viewport catcher owns
the pointer (composing, not painting) but **navigation survives**: middle-drag
and space-drag pan and the wheel zooms, with the canvas's exact bindings; all
gesture maths lives in canvas space, so panning or zooming mid-gesture cannot
corrupt it.

Known rough edges for the UI experiments: no rotation/angle snapping
(shift-for-15° is the convention), no cancel affordance other than Done at
identity, and the keyboard selection shortcuts (Ctrl+D / invert) still fire
mid-mode and change what Done would cut.

## 7. What this deliberately defers

- **Cut / copy / paste** across layers and documents — the parcel machinery is
  the ingredient, the clipboard policy is not designed here.
- **Better minification** (EWA / mips) — a parcel-shader-local upgrade.
- **Snapping** (angles, integer translations) — a gesture-layer refinement.
