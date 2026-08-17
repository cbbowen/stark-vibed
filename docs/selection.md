# Selections, fill, and transform

The soft-mask coverage field every tool acts through, the fifth shape action, and moving paint under an affine, a perspective, or a warp — §6.8, §16.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 6.8 Selections — a soft mask, not a shape

A selection restricts where tools may act. The obvious implementation — remember
the rectangle, clip to it — does not survive contact with the rest of the design:
it cannot express a lasso combined with a rectangle minus an ellipse, it has no
answer for a feathered edge, and nothing to say about "select by color" or
painted quick-mask producers. So a selection here is **not a shape**. It is a
*coverage field* — the same sparse tile map paint lives in, one `R8Unorm`
channel, `TILE_TEX` per tile, aprons and all.

**Representation** (`document/selection.rs`). A `Selection` is a persistent map
of mask tiles plus the coverage that reigns *outside* those tiles. That single
number is what makes the infinite canvas work. "No selection" is `outside = 1`
with no tiles — free — and so is its inverse, which is why `Invert` is a
constant-cost operation on an unbounded canvas rather than an impossible one. Ops
only ever put 0 or 1 there, since the one shape with coverage at infinity is
`All`; the in-between values come from inverting a *partial* selection, below.

**Producers and the algebra.** A `SelectionOp` is a shape (`All` / `Rect` /
`Ellipse` / `Lasso`), a mode, a feather width, and an opacity. Modes are the
soft-set operations, so they degrade to ordinary booleans on hard edges and stay
meaningful on feathered ones:

| Mode | Per-texel |
|---|---|
| `Replace` | `s` |
| `Union` | `max(p, s)` |
| `Subtract` | `p · (1 − s)` |
| `Intersect` | `p · s` |

Rasterization (`selection.wesl`) evaluates the shape **analytically at canvas
position** and takes coverage from a signed distance, so antialiasing and feather
are one knob: the 0.5-contour is the boundary, and the ramp around it spans
`feather` canvas px (floored at one, which *is* the antialiased hard edge). Being
a pure function of canvas position, a tile's apron rasterizes identically to its
neighbour's interior — the §6.4 seam invariant, for free — and the mask can be
resampled at any zoom without ever having been stored at one. The lasso is a
polygon: even-odd crossing for the sign, nearest-edge distance for the magnitude,
with the edge list uploaded as an `N×1` texture.

**Where it applies to the brush.** At the *end* of each stroke path, never by
clipping the footprint:

- The swept fast path masks in the **integrate** pass: `out = mix(base, merged, m)`
  (`integrate.wesl`).
- The dynamics stamp loop masks in **deposit**, lerping its whole
  read-modify-write back toward the pre-segment snapshot, and scales **pickup**'s
  lift by the same coverage (`dynamics.wesl`) — so paint outside the selection is
  neither taken nor laid, and both sides of the transfer still balance (§6.1).

Masking the *result* rather than the stroke's coverage is the whole point. A
half-covered mask texel must read as half of the finished paint; scaling optical
depth by 0.5 instead would barely fade an opaque brush at all, and a feathered
selection would have a hard edge.

Consumers never branch on whether a mask exists: where the selection has no tile,
a **1×1 texture holding the constant** is bound and every read clamps to the
bound texture's own extent. An unmasked document costs one extra texture fetch
and nothing else — which is why the goldens are unchanged.

**Why it lives in `DocState`.** A stroke's pixels depend on the mask in force
when it was drawn, so replay must reconstruct it: the selection is document state
and edits are logged actions. It is **owned** document state —
`DocState.selections` holds one mask per `ActorId`, and `Action::apply` reads the
key off `self.id.actor`, never off the payload (§17.3). What travels is the
**op**, not the mask — a few floats or a decimated polyline — and every peer
rasterizes it identically from the same shader. An op needing more than
`MAX_SELECTION_TILES` masks is rejected (deterministically, so peers agree)
rather than clipped; `All` already expresses "everything" at zero cost.

**Feedback.** A third compositor pass outlines the selection over the lit image
(`overlay.wesl`), one instanced quad per mask tile. The contour is recovered from
the mask itself rather than from the shape that produced it — `(m − h) / |∇m|`
with the gradient taken at one canvas pixel, converted to screen px by the zoom —
so it stays a constant on-screen width at any zoom, stays thin over a feathered
edge, and needs no bookkeeping to survive union/subtract/intersect. `h` is half
the selection's own peak coverage rather than a flat 0.5, which is what keeps a
partial selection visible; see below.

**The shape tools do not only select.** Rect, ellipse and lasso never produced
selections; they produce **coverage**, and the four modes above are only the four
ways that coverage can land on the mask. Landing it on the *paint* instead is a
fifth `ShapeAction` — `Fill` — sharing the shapes, the rasterizer and the feather
outright. Two things follow rather than needing rules: the mask still gates a
fill exactly as it gates a brush, which is what makes a fill of an unbounded
canvas well-defined at all; and the modifiers, which override the *combine mode*,
are inert under `Fill`, which has no combining to do. The one genuinely unbounded
case (fill the selection, with nothing selected) is **refused**,
deterministically, so peers and replays agree; inventing a boundary would be a
different fill on every client.

**A fill deposits paint, not color.** It stacks by the shared parcel law
(`paint_common.wesl`) — the very law a stroke deposits through. A filled region
has real thickness: it takes the light, it can be glazed over, and a lift brush
scrapes it back off (`tests/fill.rs` pins that last one, because it is the whole
difference from a paint bucket). Coverage scales the *paint*, never the per-unit
opacity, so a feathered edge is a thinning of the deposit rather than a fade of
its color. The drag previews as the *paint* rather than an outline — the same
`FillRenderer::apply` the commit makes, over the same base, so
`preview == committed` holds as it does for a stroke.

**How much it covers is one number, and it is a coverage.** `FillOp::opacity`.
A fill used to lay the brush's color alpha at the brush's flow, on the argument
that a fill lays the paint you have in hand. What that actually gave the user was
a Fill button governed by two sliders in another panel, neither labelled for this
job, which between them **could not produce an opaque fill**: visible coverage is
`1 − exp(−K·opacity·height)` (§6.1), so the whole of the flow range at full alpha
buys 95%, and the last 5% is a dozen more flow's worth. A slider that cannot
reach its own top is not a control.

Naming the *coverage* instead, and letting the shader solve for the paint, fixes
both halves at once. `fill.wesl` inverts the slab law — `m = −ln(1 − w)/K`, the
same inversion `slab.wesl` already runs to merge a layer through a blend mode —
and lays fully opaque paint of exactly that mass, capped at the thickness the
matte slab calls opaque (§15.4). So 1 covers, ½ covers half, and the feather ramp
lands on the canvas as precisely the ramp `selection.wesl` rasterized, because the
coverage asked for is linear in the mask and only the paint that delivers it is
not. The brush's color is still the fill's color; only its *alpha* stopped being
consulted, that being a fact about the pigment rather than about how much of the
picture this covers.

**And the two whole-selection fills do not ask at all.** `FillOp::of_selection`
and its gradient sibling take no opacity: their region *is* the selection, so how
strongly they land is already written in the mask they come through. Taking no
parameter is how "the slider applies once" is said structurally rather than
remembered at three call sites.

### A selection has a strength — `SelectionOp::opacity`

The Select panel's **Opacity** slider, above Feather, and the exact counterpart of
it: one says how soft the edge is, the other how strong the whole region is; one
is a ramp, the other a level, and they multiply. Both apply to whichever of the
five actions the row is set to, because both describe the coverage the gesture
*produces*, and the five actions differ only in where that coverage lands.

**It cost nothing downstream, and that is the argument for putting it here.** The
mask has always been a coverage field — feathered edges are already fractional
values — so a partial selection is that field taking 0.4 where it used to take 1.
`selection.wesl` multiplies the shape's ramp by the opacity and every consumer is
untouched: a brush deposits at 0.4 (`integrate.wesl` lerps by the mask), a fill
lands at 0.4, a transform carries 0.4 across. "Affects every tool" is not a list
of tools that were changed; it is the absence of one.

Three things did have to answer for it:

- **The marching ants.** The outline is recovered by differencing the mask, and a
  selection at 0.4 has no 0.5-contour anywhere — the ants would simply not be
  drawn. `Selection::level` carries the mask's peak, per selection and per
  instance (so each collaborator's outline traces its own), and `overlay.wesl`
  contours at half of it. The recovered distance is unchanged by the scaling,
  since `m` and `|∇m|` scale together, so the line lands in the same place at the
  same width whatever strength the selection is drawn at. `level` says where to
  draw a boundary and nothing else.
- **Inversion.** `level − m` rather than `1 − m`, so the complement of a region
  selected at 0.4 is its outside selected at 0.4 rather than at full strength —
  and inverting twice is the identity, which it would not otherwise have been.
- **`Selection::outside`.** A number now, not a flag: inverting a bounded partial
  selection leaves the whole plane selected at 0.4, which no boolean can say. The
  CPU-side combine became the real soft-set algebra rather than a boolean twin of
  it, which is one fewer pair of spellings that can drift.

`SelectionShape::All` is pinned to full strength, by the constructor rather than
by a rule `plan` has to remember. Its coverage lands in `outside` where there is
no boundary to rasterize, so a partial one would need a rewrite of every tile the
selection has — for a state the UI cannot ask for, since the only `All` op is
Deselect. Selections built entirely at full strength have `level == 1` and
`outside ∈ {0,1}`, where every expression above reduces to exactly what it was.

**Selecting is momentary; filling is not.** `Session::end_shape` hands the canvas
back to `Tool::Brush` the moment a *selecting* gesture actually encloses
something. Selecting is a step *towards* painting and is essentially never done
twice in a row, so a modal selection tool charges a deliberate switch-back on the
overwhelmingly common path — and when the user forgets, their next brush gesture
silently redefines the selection instead of painting. A fill *is* painting, and
blocking in is done many times in a row, so it leaves the tool armed. The rule is
one sentence rather than two cases: **the tool disarms when the gesture was a
step towards painting, and stays armed when the gesture was painting.** A gesture
that enclosed nothing (a stray click) leaves the tool armed either way, rather
than punishing a mis-click.

This is engine-side, not chrome: the session owns `tool`, so every frontend gets
the same behaviour and `observe().tool` reports it in the same update that
committed the op. The frontend then needs no "Paint" tool chip at all — *no chip
lit* is the brush, and clicking the lit chip disarms it, so the control that
armed a tool is the one that takes it back. The commands that act on a whole
selection (transform, fill, deselect, invert) live in a small floating bar
mounted only while a selection is in force: they are meaningless without one, and
a bar that is present or absent indicates the canvas is masked more directly than
permanently-visible buttons that happen to be greyed out. Fill appears in both
places — as the fifth action chip and as a button on that bar — which is not a
duplicate but the same word answering the two ways a region can already exist:
one you are drawing now, and one you drew earlier and kept.


## 16. Transform — moving paint under an affine

Move / scale / rotate / flip / skew of the selected region of a layer. A
selection could only *mask*; this makes it *hold* paint.

### 16.1 The action

```rust
ActionKind::Transform { layer: LayerId, affine: Affine2 }
```

One variant, carrying a
`glam::Affine2` — six floats in the log. Semantics: **cut the paint under the
author's selection on `layer`, apply the affine, stack the result back onto what
remained — and carry the author's selection mask along with it.** A universal
selection means the whole layer, so "move layer" and "move selection" are one
action, not two.

Everything rides existing machinery:

- **The author's mask gates it**, read from `self.id.actor` like a stroke's
  (§6.8, §17.3) — a collaborator's lasso never decides what *your* transform
  moves.
- **Undo** is timeline navigation solo and `Undo(target)` shared, untouched.
- **Replay** is exact: the GPU passes are pure functions of the tiles, the mask
  and six floats.
- **A matte layer refuses it**, exactly as it refuses strokes (its geometry
  already moves via `SetMatteRect`).
- **Degenerate and oversized transforms are rejected deterministically** — a
  near-singular matrix (paint would vanish into a line), or a destination needing
  more than `MAX_TRANSFORM_TILES` rewrites. The bound is a pure function of the
  action and the state, so peers and replays agree, exactly as
  `MAX_SELECTION_TILES` works. The document is left unchanged.

The selection moves because the alternative is wrong twice: the outline would sit
over paint that is no longer there, and a second nudge would cut a stale region.
`outside` is unchanged by any affine (an affine of the whole plane is the whole
plane), so the flag machinery is untouched.

### 16.2 The cut is a lift, not an erase

What does "remove the selected fraction `m` of a texel's paint" mean in the
normalized representation (§6.1)? The dynamics loop already answers for the
eraser: **height leaves; the remaining paint's latent and per-unit opacity are
untouched** — the source fades because its *thickness* drops, never because its
alpha is scaled.

```
h₀ = h · (1 − m)          // the cut: thickness leaves
parcel = (latent, op, h·m) // what the transform now holds
```

The rejected alternative — scaling the premultiplied color by `(1 − m)` too —
fails the invariant that makes this feature testable (§16.4): cutting and pasting
back in place must be the identity. Under the lift law the masses recombine
exactly (`op·h·(1−m) + op·h·m = op·h` at every feather value `m`); under color
scaling the optical mass recombines as `(1−m)² + m²` of itself and a feathered
identity transform visibly thins the paint at its own edge.

### 16.3 The paste is a parcel deposit

The moved paint lands as a **parcel of existing paint**, so it stacks by the law
the dynamics deposit uses (`blend_latent`, `paint_common.wesl`): heights add,
latents blend weighted by visible alpha (`1 − exp(−K·opacity·height)` — the same
translucent-slab constant as the media pass), the new per-unit opacity is total
optical mass over total height. Thick moved paint covers what is under it; a thin
moved glaze tints it. That function lives in a shared module so the two consumers
cannot drift.

Notably this is *not* the integrate pass's premultiplied-"over": a stroke's
scratch alpha is swept coverage (already an exposure), while a moved parcel's
`op` is per-unit opacity — over-blending by it would let a zero-thickness ghost
occlude real paint.

Scaling deliberately does **not** conserve total paint: doubling a region's size
quadruples its paint, keeping every texel's appearance. That is what a transform
tool means everywhere; conserving mass would visibly thin paint as it grows. Pure
moves conserve height exactly.

### 16.4 Resampling, and the invariants that pin it

The transformed paint is resampled **once**, bilinearly, forward-rasterized: each
selected source tile's *interior* becomes one quad under the affine, drawn into
each destination tile it touches, sampling the source tile's textures (the 1-px
apron feeds the filter's edge taps — §6.4 built exactly this). Source interiors
tile the plane, so their images tile the transformed plane: quads share
transformed edges bitwise (corner arithmetic is exact in f32) and the
rasterizer's fill rules make coverage watertight — **every destination texel
receives at most one parcel**, which is what lets the deposit be a single pass
with no blending and no order dependence.

Three exactness properties are the acceptance tests, all consequences of sampling
at texel centres falling on texel centres:

1. **Identity is a no-op.** Weights `(1,0,0,0)`, parcel = source, restack is
   algebraically the inverse of the cut.
2. **Integer translation is exact** — the painting moves without a single texel
   of resampling loss.
3. **Axis flips and quarter-turns about half-integer centres are exact**, and
   involutions compose to the identity through two separate actions.

Two implementation choices carry these, because "algebraically exact" is not
"byte-exact" on a GPU: the fragment maps its own canvas centre back through the
**inverse** affine rather than interpolating a uv across the quad (attribute
interpolation plus the sampler's fixed-point uv conversion can land an
exact-centre tap a fraction off), and the combine takes **exact branches at
coverage 0 and 1** — passthrough and parcel respectively — because the render
target's f32→f16 store rounding is implementation-defined, so even sub-half-ulp
recomputation noise can flip a stored bit. What remains inexact is exactly what
must be: texels strictly inside a feather ramp are genuinely recomputed
(`h·(1−m) + h·m`) and may land one f16 ulp off — at most one display LSB, pinned
by test.

Everything else (rotation, fractional scale) is honest bilinear: one generation
of loss per commit, never compounded during a gesture because the gesture
previews from the *committed* tiles and commits once. Minification below ~½×
aliases (bilinear underfilters); an EWA or mip-chain sampler is a future quality
pass, localized entirely in the parcel shader.

Aprons stay rendered-not-copied (§6.4): every pass here is a pure function of
canvas position, so a destination tile's apron is bit-identical to its
neighbour's interior by construction and the seam test extends to transforms.

### 16.5 The GPU passes

Per rewritten tile, two passes (`transform.wesl`), the same CoW discipline as
every stroke — old tiles stay valid in old history versions, new tiles come from
the pool:

- **Parcel** — clear a scratch pair to zero, then for each intersecting source
  quad draw it, sampling source color / height / mask bilinearly: out =
  `(color_as_is, height·m)`. Disjointness (§16.4) makes draw order irrelevant.
- **Combine** — fullscreen `textureLoad` pass: cut the base by its own mask
  (`h₀ = h·(1−m)`), stack the parcel by §16.3, write color + aux. Base and mask
  bind per-tile textures or the 1×1 constants (§6.8's pattern), so paste onto
  virgin canvas and cut-only tiles are the same shader; a texel neither cut nor
  pasted passes through bit-exactly.

A tile whose mask is constant 1 and which no quad reaches is dropped from the map
— the combine would write zeros, and an all-zero tile is worse than no tile (it
would pollute `bounds` and hold pool memory).

The mask transform is one simpler pass: destination mask tiles are cleared to
`outside` and the source mask quads drawn over, resampling coverage. Mask tiles
carry a ring of constant coverage by construction (`Selection::plan` pads), so
the boundary between rasterized and constant regions is continuous.

CPU planning (`document/transform.rs`) is pure and unit-testable, like
`Selection::plan`: classify each populated tile against the mask (untouched /
partial / fully-selected), build the quad list, intersect transformed quads with
tile rects (exact convex test, not AABB — a loose test would mint empty tiles),
and enforce the caps.

### 16.6 The gesture: an ellipse, not a box of handles

Most software shows a rectangle with resize grips, tacks rotation on as a knob,
and either hides skew behind a modifier or does not offer it. This widget is an
**ellipse** — the image of a reference **circle** under the accumulated linear
map — so the widget's own shape *shows* the transform: it stays a circle exactly
as long as the transform is a similarity (move, turn, uniform scale), and any
eccentricity *is* the distortion. A circle rather than the hull's own aspect,
precisely for that reading — a rectangle-shaped reference would start a
rectangular selection's widget as an ellipse and the shape would say "distorted"
before anything happened. Its radius is the geometric mean of the hull's
half-extents (the area of the hull's inscribed ellipse), so it stays proportionate
for elongated selections.

One surface carries the whole affine group with three gestures, chosen by where
the drag starts:

| Region | Gesture | Family |
|---|---|---|
| inside | translate | `x ↦ x + Δ` |
| on the rim | rotate + uniform scale | similarity about the centre |
| outside | directional scale + skew | rank-1: `I + (Δ ⊗ d̂)/λ` |

Each shaping gesture is solved so that **the grabbed point follows the pointer
exactly** within its family — the hand feels attached to the paint, and the
pointer's two degrees of freedom are always fully spent:

- **Rim.** The unique similarity carrying the grab to the pointer is the complex
  ratio `(p − c)/(p₀ − c)`. Its differential behaviour is exactly the spec:
  motion *tangent* to the ellipse is pure rotation, motion *normal* to it is pure
  uniform scale, anything between blends — no mode to pick, no knob.
- **Outside.** The rank-1 map `G = I + (Δ ⊗ d̂)/λ` (with `d̂` the grab direction,
  `λ` its distance) also carries the grab exactly to the pointer while **pinning
  the diameter perpendicular to the grab**: radial pull is a scale along `d̂`,
  tangential drag is a shear, and the pinned axis is what makes the gesture
  predictable. Composing these from different directions (with the rim's
  rotations) reaches every orientation-preserving linear map — skew and
  non-axis-aligned scaling are not a bolted-on mode but the same vocabulary.
  Pulling in past the pinned axis is floored at 90% so the determinant cannot run
  through zero mid-drag.
- Hit-testing pulls the pointer back through the linear map into the reference
  circle's space, where every region test is a radius — exact at any deformation;
  the rim band stays a constant *screen* width by scaling with the widget's local
  radius. The gesture is locked at the press (crossing the rim mid-drag must not
  change what the hand is doing), a north dot marks the reference "up" (a rotated
  circle otherwise hides its rotation), and the cursor announces the region under
  the resting pointer.

State is `TransformState { anchor, radius, center, linear: Mat2 }`, affine
`x ↦ center + linear·(x − anchor)`; gestures left-compose world-space factors
onto `linear`, always recomputed from the drag's start (nothing accumulates
per-move). Untouched factors are simply absent: a pure move keeps `linear`
bit-exactly the identity — §16.4's pure-translation exactness with no snapping
heuristics — and a sub-2-screen-px jiggle in any gesture snaps back to its start,
so touching the widget never resamples by accident. The bar carries **Flip ↔ /
Flip ↕ / Done** (flips are world-axis mirrors folded into `linear`; four mirrors
cancel bit-exactly).

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

Known rough edges: no rotation/angle snapping (shift-for-15° is the convention),
no cancel affordance other than Done at identity, and the keyboard selection
shortcuts (Ctrl+D / invert) still fire mid-mode and change what Done would cut.

### 16.7 Deliberately deferred

- **Cut / copy / paste** across layers and documents — the parcel machinery is
  the ingredient, the clipboard policy is not designed here.
- **Better minification** (EWA / mips) — a parcel-shader-local upgrade.
- **Snapping** (angles, integer translations) — a gesture-layer refinement.

### 16.8 Perspective — moving paint under a homography

The second transform family (numbered after §16.7 to keep citations stable;
thematically it follows §16.6). Where the affine acts on the whole plane,
a perspective is **rect-scoped**:

```rust
ActionKind::TransformPerspective { layer, map: PerspectiveMap { min, max, corners } }
```

The wire form is the four **corners the hand placed** — the images of the
rect's corners, twelve floats — and every peer re-derives the same homography
from them (identity → the literal identity matrix; a parallelogram target →
built and inverted through `Affine2` and embedded with a `(0,0,1)` bottom row,
so the projective divide is by exactly 1 and §16.4's tap exactness carries
over; a general quad → derived in f64, square-to-quad composed with the rect
normalization, inverse by adjugate, rounded to f32 once).

**The rect gate.** The map cuts and carries only the paint under
`mask · box(rect)`, where `box` is the rect's coverage with a half-pixel
antialiasing ramp — a pure function of canvas position, computed identically
on the cut side (the combine's new gate uniform) and the paste side (the
parcel fragment), so an identity map recombines exactly. Paint and mask
outside the rect are untouched *by construction*: tiles the rect never
overlaps are not even planned. This is also what tames the horizon — validity
requires a **strictly convex, positively oriented** target quad, which is
exactly the condition under which `w > 0` on the whole source rect, so
nothing inside the gate can be flung through infinity, and nothing outside it
ever meets the map. Concave, crossed, or mirrored quads are rejected
deterministically, like a degenerate affine (§16.1).

**Rendering** rides §16.5's machinery with one generalization: piece corners
(`tile ∩ rect` under the map) are computed on the CPU from shared values by
shared formulas and handed to the vertex stage precomputed, so adjacent
pieces stay bitwise-watertight even though the map is not affine; the
fragment maps its canvas centre back through the inverse homography. The
combine pass is the same shader with the gate uniform; the parcel law, the
lift law and the CoW discipline are untouched.

**The mask** cannot be pure Replace under a gate: coverage inside the rect
travels, coverage outside stays. Per destination tile the GPU computes
`max(old · (1 − box), moved)` — the residue laid down fullscreen, the moved
coverage drawn over with **max blending** (the soft union, §6.8's algebra, and
safe under any draw order). Mask tiles the rect never touches keep their
handles. A universal selection stays universal — there is no outline to
carry, matching the affine's behaviour.

**The gesture.** The widget is the quad itself, with corner handles: the map
is *defined* as "the homography putting the corners where the hand put them",
so the widget cannot disagree with the paint — the grabbed corner follows the
pointer exactly, §16.6's exact-follow promise in its purest form. Edges shift
whole sides (the foreshortening gesture), inside translates, and the receding
grid drawn through the quad is the transformed space itself (lines stay
straight under a homography, so two endpoints each). A drag that would run
the quad concave **holds at the last valid shape** rather than tearing
through the horizon — the same stance as the rim gesture's determinant floor.

### 16.9 Warp — moving paint through a mesh

The third family, same action shape:

```rust
ActionKind::TransformWarp { layer, map: WarpMap { min, max, cols, rows, points } }
```

The wire form is a coarse **control grid** (the UI uses 4×4; the format
allows up to 8×8) — the images of the rect's uniform grid, a few dozen
floats. Everything else is derived deterministically: a Catmull-Rom tensor
surface through the control points, sampled onto a fine lattice of 8×8
bilinear sub-cells per control cell. The GPU consumes only the lattice, whose
cells are straight-edged quads (a bilinear map keeps axis-aligned lines
straight), so the parcel machinery carries over piecewise: each drawn piece
is `sub-cell ∩ tile`, and the fragment inverts the cell's bilinear map (the
stable quadratic, computed entirely in cell-sized differences so large canvas
coordinates never meet a cross product).

**Identity is bit-exact by construction.** The surface is evaluated in
*deviation form*: control points split into `base + delta`, only the deltas
go through the Hermite arithmetic (every term a multiple of a delta
difference), and the bases are added back untouched. An untouched mesh has
all-zero deltas, so every lattice point lands exactly on its base, every
sub-cell is detected as a parallelogram and takes an exact inverse-affine
path — and the whole action is a byte-for-byte no-op, extending §16.4's
identity invariant to the mesh. Cell-edge lattice geometry is computed by
shared formulas from shared values (`lerp` with an exact `t == 1` branch), so
adjacent pieces rasterize watertight.

**No folds.** The Jacobian of a bilinear cell is bilinear in its parameters,
so its extrema are at the corners: four cross products per sub-cell decide
the whole cell, and a mesh with any non-positive corner Jacobian is rejected
deterministically — resampling through a crease would be last-write-wins
noise, and the gesture never offers it (below). The rect gate, the mask
union, and the caps are exactly §16.8's.

**The gesture.** The mesh curves the overlay draws are sampled from the very
surface the paint resamples through — a straight grid says "untouched", and
every bend in the curves is a bend the paint has taken. Two ways to shape it,
both exact-follow:

- **drag a control point** — it follows the pointer exactly;
- **grab the surface anywhere** — the least-norm control move that keeps the
  grabbed *paint* under the pointer: `Δpᵢ = Bᵢ·Δ / ΣB²`, with `B` the
  surface's basis weights at the grab. The hand holds the painting, not a
  handle — §6.9's "the tool disappears" applied to deformation.

Outside the mesh, dragging translates it whole. A drag that would fold the
mesh holds at the last valid shape; a sub-2-screen-px jiggle snaps back
(§16.6's rule — touching the widget never resamples by accident).

### 16.10 One bar, three families

The transform bar grows a selector: **Free / Perspective / Warp**, plus the
affine-only flips and "Done". Switching families mid-gesture **carries** the
accumulated deformation when the new family contains the old one exactly — an
orientation-preserving affine is exactly a parallelogram perspective, and
exactly a mesh whose smooth surface reproduces it (cubic interpolation
reproduces affine functions) — and otherwise **commits** it first, one honest
undo step, reopening fresh around the moved paint. Never a silent
approximation: a lossy carry would change what "Done" produces without the
hand having done anything.

Every family previews through the same `ViewCommand::PreviewTransform`
bargain (§16.6), now carrying a `TransformMap` (affine / perspective / warp);
the engine routes each family to its own action kind, so the in-process
command enum never touches the wire format. `preview == committed` holds for
all three, pinned by test.

---


