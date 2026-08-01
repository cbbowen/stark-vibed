# The brush engine

Tiles and channels, the fitted-path swept-segment stroke renderer, the wet-mixing dynamics loop, and brush shape assets — §6.1, §6.2, §6.6.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 6.1 Tiles and channels

A tile is a fixed `TILE_SIZE` (256×256) square in canvas space, addressed by
integer `TileCoord(i32, i32)`. Sparsity gives the infinite canvas: only painted
tiles allocate. Each tile is **multi-channel**, which is what enables strokes
that affect more than colour:

```rust
pub struct GpuTile {
    pub color:  wgpu::Texture,   // Rgba16Float — working-space channels + premult alpha
    pub height: wgpu::Texture,   // R16Float — total paint height
}
```

The colour texture stores **Oklab** components (or Mixbox concentrations), not
sRGB. Linear 16-bit float comfortably holds Oklab's range and the negative `a`/`b`
chroma axes, and keeps blends perceptually uniform. Alpha is premultiplied.

> **The colour alpha channel is *only* the paint's per-unit-thickness opacity** —
> a material property (how opaque the pigment is per unit of thickness). It says
> **nothing** about how much paint is on the canvas, nor even whether any is
> present. **The amount (and presence) of paint is the `height` channel**
> (precisely, `height − surface_height`, the paint *thickness*). The two combine
> only at display time in the translucent-slab law
> `visible = 1 − exp(−K · opacity · thickness)`.
>
> Consequences the brush dynamics must respect:
> - To **conserve paint** (move it without creating or destroying), conserve
>   **height** — never the alpha. Alpha is per-unit and is carried as a
>   height-weighted blend of the picked-up paint's opacity; it is not consumed.
> - A thin layer of opaque paint (alpha ≈ 1, tiny thickness) is *barely visible*;
>   a thick layer of translucent paint can be very visible. Opacity alone is not
>   coverage.
> - Lifting paint reduces the canvas **height**, leaving the remaining paint's
>   per-unit alpha unchanged; the source lightens because thickness — not alpha —
>   drops.

Channels are referenced through a small `ChannelSet` descriptor so renderer,
compositor and tile pool agree on layout without hard-coding it everywhere — a
new channel is a descriptor entry plus shader usage, not a structural rewrite.

`TilePool` recycles GPU textures of each channel format to avoid per-stroke
allocation churn; `acquire_tex()` returns a cleared tile, and dropping the last
`Arc<GpuTile>` returns it to the free list. The pool's formats come from the
colour space in use, never hardcoded.

## 6.2 The brush engine — natural media

Stroke rasterization is **swept-segment along a fitted path**: pointer samples are
fitted to control points, expanded to a smooth polyline, and each short segment
swept as a single quad. Layered on top is a **brush-dynamics** model that carries
loaded paint and smears what is already on the canvas, so wet-on-wet mixing feels
physical. Everything is deterministic — the only randomness is the explicit
`seed` — so live paint, replay and goldens agree.

**Path representation & cubic interpolation.** `path.rs` keeps three
representations deliberately distinct: an **`InputSample`** is one raw pointer
report (transient, never stored), a **`ControlPoint`** is a fitted curve knot
(what `StrokeRecord::path` holds), and an **`IntermediateSample`** is a point *of
the curve* — position plus derivative, with pen attributes interpolated there —
produced at render time and consumed by the stamp generator.

A `PathFitter` streams samples into control points as a **least-squares clamped
cubic B-spline** (`spline.rs`), grown and refit as they arrive:

- **Grow.** The control polygon lengthens with the stroke — one point per
  `KNOT_SPACING` of arc length, plus more wherever taking one measurably reduces
  the error, by at least `KNOT_COST`. Fitting is what smooths, so there is no
  separate low-pass stage: a polygon far coarser than the jitter averages a
  pixel-quantized staircase away. The arc-length floor is not redundant with the
  sagitta test — it is what makes the polygon grow, and so freezing advance, on a
  stroke the fit is already perfect on.
- **Refit** every sample, but only over the *live* points and the *free* control
  points, so work per sample follows the tail rather than the stroke so far.
- **Freeze** all but the last few control points. Those are final — nothing drawn
  later can move them — which makes the fit append-only and lets a caller treat
  the settled prefix (`frozen_spans`) as already rendered.

Both growth thresholds are denominated in an **input tolerance** the frontend
supplies with `GestureCommand::Start`, in canvas px (the error one as its square,
since it is compared against a mean square). Canvas px are the wrong unit to fix
them in: the same hand movement covers 64× as many zoomed in as out, and a pen
digitizer, a touchscreen and a mouse each report at a different grain through the
same pointer API. Only the frontend knows either fact, so it states the grain and
the fit becomes invariant to zoom. This is a *fitting* knob and reaches nothing
else — flattening's budget is an error against the curve, in the canvas px it
will actually be drawn in.

Both **ends are pinned**: a least-squares fit does not hold them, because a
stretch of parameter with no sample assigned costs nothing, so the curve
otherwise starts before the stroke and stops short of the pointer. The start is
set and frozen at the first sample; the live end moves to the newest sample each
update (and freezes there on release), which also keeps the preview under the
cursor.

Pen attributes ride along as **passenger channels**: pressure, tilt and time are
solved against the geometry's own assignment rather than fitted jointly with it,
so a pressure ramp cannot stretch the parameterization the way a longer path
does, and no weighting is needed to reconcile pixels with whatever units they are
in.

Rendering expands those control points through the same B-spline — converted per
span to cubic Bézier form, so the derivative is closed-form — into a polyline,
and subdivides **adaptively**: a piece is split until the straight segment
standing in for it is within a bounded error in position, in *tangent direction*,
and in the pen attributes. Sampling follows the curve rather than arc length: a
long gentle stroke costs a handful of segments where uniform stepping cost
hundreds, and a corner still gets the density it needs. The tangent bound buys
both — it is the term that cannot be fooled by a symmetric wiggle, and the one
that spikes exactly at a corner. This solves several problems:

- **No stair-step aliasing** — jittery pixel-stepped input collapses to a clean
  curve instead of axis-aligned segments. This is the fit doing it, and it is why
  the price of a control point must sit *above* the input's own quantization
  (what the frontend's declared tolerance is for). Priced below it, a staircase
  reads as curvature and gets traced rather than smoothed.
- **Continuous-looking stamping** — stamps ride a smooth path with smooth
  tangents, so even hard-edged tips read as one stroke rather than a row of dabs.
- **Smaller files** — a handful of control points replace hundreds of raw samples
  in the action log (§8).

Adaptive sampling has one hard prerequisite, easy to violate silently: **the
deposit must not depend on how the path was cut into segments.** Anything a
segment applies *per segment* rather than per fragment also caps segment length,
which the renderer supplies as a bound rather than the fitter assuming one
(`gpu::stroke::flatten_tolerance`).

**Tapered ends.** `start_taper_length` / `end_taper_length` scale the tip down to
a point over a run at each end, which is what turns an even-width digital line
into an inker's stroke. Both are quoted in **brush radii**, not canvas px, so a
brush keeps its look as it is resized. The profile is `f(t) = t(3 − t²)/2`:
`f'(1) = 0`, so the taper meets the full-width body with no crease (the artifact
that gives a taper away), and `f'(0) = 3/2`, so it leaves the tip as a wedge
rather than a blunt cap or a whisker-with-a-bulge. It is a polynomial, not the
sine it approximates, because the taper decides stored pixels and replay,
goldens and peers must agree on it bit for bit.

Two places the obvious implementation is wrong:

- The taper varies radius *with distance travelled* while a segment sweeps at a
  constant one. Paid for by cutting segments finer, but **locally**: only edges
  actually inside a taper are subdivided, so a long stroke pays ~75 extra
  segments at each end instead of flattening its whole length at the taper's step.
- A taper is measured from the ends of the **whole** stroke, and while the
  pointer is down the far end has not happened yet. So freezing is held back: a
  span is settled only once it is a trailing taper's length clear of the live end
  *and* a leading taper's length past the start — which together also prove the
  stroke has outgrown the "scale both zones to fit" compression that keeps a
  short flick a small pointed mark rather than a sliver. Both tests use chords,
  which under-estimate arc length, so what they admit is genuinely final; and an
  admitted prefix stays admitted however the stroke continues
  (`gpu::stroke::taper_safe_frozen`).

**Incremental repaint.** Freezing is what keeps a long stroke responsive. Drawing
a live stroke costs (segments × tiles covered), both growing with length, so
re-rendering the whole thing per pointer move is quadratic. Instead the engine
keeps a `FrozenHead`: the settled spans, rendered once onto the committed
document and kept. Each move draws only the live tail over that — a few spans,
whatever the stroke's length — and the head advances as the fitter freezes more
(`StrokeRenderer::render_range`, `path::flatten_spans`; adjacent ranges share
exactly one flattened point, so their segments tile with no gap and no overlap).

This is the *same* partition-independence the constraint above demands, spent
deliberately: the swept deposit is a definite integral per segment that composes
by summing optical depth, so cutting the path at a span boundary and compositing
the pieces in order gives what one pass gives.

The stamp loop that dynamic brushes run has no such property — it is
*sequential*, each segment reading the canvas the previous one left and the tool
the previous one loaded. It is cuttable anyway, because that carried state is
small and entirely **brush-local**: the reservoir texture (what paint the tip
holds, and where on the tip), plus travel since the last pickup, which sets the
reload cadence. A `ToolState` remembers both at the freeze boundary and the tail
resumes from it. Being brush-local is what makes this work at all — the state
says nothing about *where* the stroke is, so the region rectangle may change
completely between the piece that saved it and the piece that resumes. The canvas
side needs no carrying: it is already in the head's tiles.

The renderer cuts the path for its own purposes too, on the same argument. A
region is a 1:1 copy of the canvas under the stroke, so a stroke crossing the
document would want a region the size of the document; instead it is drawn in as
many region-sized **pieces** as it takes, each compositing what the last wrote
back (`gpu::stroke::chunk_segments`). Length therefore costs a dynamics stroke
pieces, not correctness — where it used to degrade past `MAX_REGION_DIM` to the
plain swept deposit, which is not a coarser version of the same brush but a
different one: the swept path only ever *adds* paint, so a brush whose purpose
was to lift it stopped doing the one thing it was for, on exactly the long
strokes and fat tips that wanted it most.

One thing must be decided from the record rather than the piece in hand, because
a live tail and the commit that replaces it must draw the same pixels: whether
the stroke runs the stamp loop at all. It is decided from the **brush alone** —
the strongest form of that guarantee, since there is nothing about the piece, or
about how long the stroke has grown, for it to disagree over — and what it asks
is the floor no subdivision gets under: whether one segment's own footprint fits
a region, since the reservoir pickup reduces over the whole tip at once. See
`gpu::stroke::dynamics_setup`.

**Continuous stamping (swept segments).** Discrete dabs are visible with hard
tips. The fix: stamp each short *segment* of the flattened curve as one quad
whose coverage is the brush **swept** along it — the path integral of the
footprint, instead of point samples. The enabling identity: alpha-"over" is
multiplicative in `(1−α)`, hence additive in **optical depth** `τ = −ln(1−α)`. So:

- Precompute, per brush, the **prefix integral of `τ` along the travel axis**.
  A length-`d` segment's swept depth at a point is `prefix(u) − prefix(u−d)` for
  that row — an O(1) lookup.
- A segment quad outputs `α_seg = 1 − exp(−opacity · sweptDepth)`. Because the
  existing premultiplied-"over" blend across overlapping segment quads combines
  as `1 − ∏(1−α) = 1 − exp(−Σ τ)`, it sums the depths **exactly** — no
  double-counting at joints, no scratch buffer, no second pass. The whole
  stroke's coverage is the continuous path integral `1 − exp(−τ_total)`.
- **Every** channel a segment deposits must be a function of that segment's `τ`
  in one of exactly two shapes: *additive* in `τ` (an amount — the height the aux
  target sums), or `1 − exp(−k·τ)` (a rate — the opacity the colour target
  over-blends). Those are the two that survive re-cutting the path, because `τ`
  is what sums. Any other shape makes the stroke depend on the *number* of
  segments: a per-segment `√`, for instance, deposits `Σ√(τ/N) = √(N·τ)`, so the
  stroke silently gains weight with sampling density. Invisible while sampling is
  uniform and immediately visible once it adapts — which is why the two forms are
  a standing constraint on the stamp shaders, not a detail of one.

Segments need only be short enough that the line + constant-radius approximation
holds, so the sweep uses *fewer* primitives than the dab model. Caveats:
per-stamp angle jitter no longer applies (the brush follows the tangent
continuously); the round tip's prefix depends on `hardness`, so it is generated
per stroke (image brushes precompute theirs at import, §6.6); a click is a
degenerate segment given a minimal length.

**Live vs. replay unification:** live painting renders the in-flight fitted
stroke onto CoW preview tiles; commit/replay render the same `StrokeRecord`
through the same path → same stamps, same pixels.

### The sequential swept-exchange loop (wet mixing & brush dynamics)

To smear paint already on the canvas, the brush picks up wet pigment under it,
carries it, and lays down an evolving mix downstream. This is **sequential and
order-dependent** (what is under the brush includes what it deposited a moment
ago), which the parallel swept pass cannot express. The loop embraces the
sequence *without giving up definite-integral rendering*: the canvas-side
exchange is **swept per flattened segment through the same prefix-τ integral as
the plain deposit**, so a dynamics stroke has the identical continuous, dab-free
footprint. All on the GPU with no readback (`gpu/stroke/dynamics.rs`,
`dynamics.wesl`):

1. **Region composite.** The base tiles under the stroke (the affected set plus a
   one-tile ring) are composited once into a 1:1 canvas **region** texture
   (colour + the wide aux). This is the working canvas the stroke evolves.
   Bounded by `MAX_REGION_DIM`, which bounds transient memory rather than the
   stroke: a stroke too big for one region is cut into pieces that fit.
2. **The loop.** The stroke's flattened segments run *in order* inside a **single
   compute pass** — the implicit barriers between dispatches give the sequential
   semantics, and usage scopes are per-dispatch, so the region can be sampled by
   one dispatch and storage-written by the next with no copies and no pass churn.
   Per-dispatch parameters ride one dynamic-offset uniform buffer.
   - Per segment: **bake** (integrate the reservoir along the travel axis),
     **exchange** (the tool's half of the transfer, one thread per reservoir
     texel), **snapshot** (copy the segment quad's region texels into an `under`
     scratch, so the deposit can read-modify-write) and **deposit** — one thread
     per footprint texel. A texel's **exposure** to the segment is the prefix-τ
     difference `e(x) = prefix(u) − prefix(u−d)`, and exposures add across
     overlapping quads of consecutive segments, so what the loop applies must be
     built from `e` in a way that survives re-cutting the path.
   - **The exchange is one closed-form solution, evaluated by both halves.** Over
     a segment a canvas point and the tool above it are a pair of coupled boxes
     trading the one conserved quantity:

     ```
     dh/de = −k_lift·h + k_dep·R + A      (canvas)
     dR/de = +k_lift·h − k_dep·R          (tool)
     ```

     Their sum moves only by the source `A`, so with `add = 0` the pair conserves
     height *identically*. The pair has one non-zero eigenvalue — it relaxes
     towards the split `k_dep : k_lift` at rate `s = k_lift + k_dep` — so every
     coefficient is a function of the single window `w(e) = (1 − exp(−s·e))/s`:
     the canvas keeps `1 − k_lift·w` of its own height, takes `k_dep·w` of the
     tool's load, and the tool takes exactly the complement of each. Being linear
     and autonomous in `e`, running it over `e₁` then `e₂` *is* running it over
     `e₁+e₂`: the whole stroke applies the continuous path integral whatever the
     spacing, with no dabbing.
   - **Both halves read the same pre-state** — which is why `exchange` is
     dispatched *before* `deposit` rather than after. Solving only half the pair
     (the canvas relaxing towards a tool that never took anything back) and then
     lifting from the region as the deposit had already left it makes the two
     sides disagree about how much changed hands by `O(lift²)` per segment. At
     `lift = deposit = 0.95` and a segment of half a radius that is 39% of the
     total `canvas + tool` height destroyed at *every* segment boundary — which
     shows as arcs at exactly the segment spacing through thick paint, and as a
     tip-shaped patch of missing paint wherever a stroke stops.
   - The **reservoir** is a real 2-D texture in brush-local coordinates
     (`BRUSH_RES`², ping-ponged), so each part of the tip carries what *it* rolled
     through. `bake` integrates it along the travel axis into a swept prefix, so
     the deposit reads what the tip presented over a whole pass rather than one
     mid-pass sample — exact for any segment length. What still bounds a dynamics
     brush's segments (`RESERVOIR_EXCHANGE_STEP`) is the tool side's single
     mid-segment tap of the canvas, not the exchange law.
3. **Write-back.** Each affected tile's full `TILE_TEX` block is sliced out of
   the shared region into a fresh CoW tile (`slice.wesl`, narrowing the wide aux
   to the persistent `(height)`). Aprons are bit-identical to neighbour interiors
   **by construction** — both are cut from the same texture — and the ring in the
   composite gives rewritten tiles real neighbour content (§6.4; the
   `apron_makes_dynamics_writeback_seamless_under_zoom` regression guards it).

*Conservation (§6.1).* Paint moves by transferring **height** — the one conserved
quantity. Colour and per-unit opacity ride as optical-mass (opacity·height)
weighted blends, and a parcel's blend weight is its own *visible* alpha
(`1 − exp(−K·mass)`, the same translucent-slab law as the media pass), so thick
deposits cover while thin glazes tint. The lift never touches the source's colour
or alpha: the source fades because its **thickness** drops. Both sides of every
transfer take complementary shares of the one solution above, over the same
segment and from the same pre-state (the canvas side measuring its exposure
through the prefix-τ, the reservoir side as `τ(l) · Δs/r` — two quadratures of the
same bilinear form, which agree texel for paired texel), so with `add = 0` total
height (canvas + tool) is conserved up to resampling error, whatever the segment
length.

What is *not* conserved is the load the tool still holds when the pen comes up:
the reservoir is per-stroke, so it is dropped. Under the last footprint that
leaves the canvas short of the trail behind it by the deposit the tip would have
made had it kept travelling, and for a near-hard tip the onset of that deficit is
compressed into the few pixels of the coverage shoulder — a faint tip-shaped disc
at the end of a heavy smear. No constant-free pen-up rule tested removes it
without introducing a worse one: settling the pair to equilibrium scrapes an
eraser's last footprint into a hard disc, and giving the footprint the pass it was
owed (`prefix(l)`) hands the *leading* rim a whole pass in one go and steps there
instead. A trailing `taper` sidesteps it entirely by taking the tip to a point.

*Order-dependence is real.* Pickup reads the region as already modified by
earlier segments, so a stroke smears **its own trail** when it crosses it; drag
falls out naturally; and there is no band, column or stamp structure to alias.

*The axes* (`BrushDynamics` on `BrushParams` — a flat record in the action log):

- `add` — lay the brush's own paint; the only inexhaustible **source**, and the
  tool's single *amount* knob: paint height laid per unit swept optical depth. A
  pure-`add` brush takes the swept fast path, untouched by the loop.
- `lift` — vertical flux canvas → tool (an eraser when alone).
- `deposit` — vertical flux tool → canvas (`lift`+`deposit` with `add = 0` is a
  true mass-conserving smudge).
- `charge` — a finite glob pre-loaded onto the tool (the palette-knife scoop); it
  depletes as the tool deposits and refills as it lifts.

That is the whole set. `drag`, `bleed`, `ridge`, `load_pressure` and
`deposit_tilt` were listed as inert placeholders and were **removed** rather than
carried. Each remains a local change to reintroduce when built (the loop already
carries per-dispatch state): a forward deposit offset for the bow-wave drag, a
footprint-local blur for bleed, edge displacement for ridge, per-segment
pressure/tilt modulation of the rates. Likewise `BrushParams` no longer carries
`spacing`, `flow`, `height` or `wetness`: with swept rendering there are no dabs
for `spacing` to space, and `flow`/`height` were redundant multipliers on `add` —
`flow` doubly so, since it also carried the `drain` factor into `τ` and applied
the run-dry falloff *twice*. `wetness` was the only source of the **wet channel**,
which is why that channel is gone too: a per-texel `wet` nothing could write is a
stored zero every pass paid for. Gloss is now a **uniform property of the paint**
(§6.3). The persistent aux is one channel, `(height)`.

*Determinism* — a stroke is a pure function of `base` + the `StrokeRecord`, so
replay and `preview == committed` hold and the log stays compact: only path +
params are stored, never per-segment data. *Perf* — two footprint-sized
dispatches per segment plus a reservoir-sized one per pickup, inside one pass; a
live stroke re-renders only its tail, resuming the reservoir from the frozen
head. What remains is per-segment dispatch overhead: a few hundred segments each
costing four small serialized dispatches dominates a move. Batching the
independent ones is the next win. *Paint never dries* — every texel stays as
workable as the moment it was laid, which is what lets there be no wetness state
at all; to glaze over "dry" paint the user adds a **new document layer**, which
composites as if dry.

### Colour dynamics (colour jitter)

The applied colour can vary **across the brush and along the stroke**:
`BrushParams.color_dynamics` (historized — it changes stored pixels) holds a
noise kind plus two per-axis **frequency** and three per-channel **amplitude**
factors. A 3-channel, exactly **tileable 2-D noise tile** is baked **once on the
CPU with fixed constants** (`noise.rs`, `Rgba8Snorm` 64², or 256² for `Mosaic`;
only correctly-rounded ops, no transcendentals ⇒ bit-reproducible across
platforms) and sampled with a repeat sampler. The kinds:

- `White` — per-texel hash.
- `Simplex` — a periodic simplex lattice: gradients hashed from
  `q = 6·(i,j,k) − (i+j+k)·𝟙 mod 6·P`, invariant under input translation by the
  period `P` (a multiple of 3). The lattice stays 3-D and the bake takes its
  `z = 0` plane, because only `G3 = 1/6` makes the unskewed lattice positions
  integral — the 2-D skew constant is irrational, so a 2-D lattice can be
  periodic along its own skewed vectors but not along the axes a tileable texture
  needs.
- `Voronoi` — Worley F1 on a jittered grid of `P` cells per side, feature points
  hashed from the cell index `mod P`. The usual 3×3 cell search is *exact* here
  rather than approximate: every feature outside that ring is more than one cell
  away and the shaping flattens the field past 0.8 cells.
- `Mosaic` — the same cells read discretely, one flat value per cell shared by
  all three channels, so facets are whole polygons with hard edges. Its owner
  search widens to 5×5, since a flat field has no clamp behind which a mis-picked
  owner could hide, and its tile is 256² because its walls are steps and so are
  only as sharp as the tile is fine.

The lookup domain is **stroke-local**: `(lateral·f₀, arc·f₁)/NOISE_TILE_PX` plus
a per-stroke translation derived from the stroke `seed`, where `lateral` is the
signed offset from the centreline and `arc` the length along it, both in canvas
px (brush-local y is in radius units, so it is scaled by the radius — the pattern
keeps one scale whatever size the tip is). One axis varies colour across the
footprint, the other evolves it along the stroke. Anchoring to the stroke rather
than the canvas makes the variation belong to the *gesture*, and costs nothing in
determinism: both coordinates are still functions of the fragment's canvas
position and the segment, so the deposit stays a pure function of the two and
tile aprons stay bit-consistent (§6.4). Clamping arc to each segment's body makes
it *continuous across overlapping segment quads*.

The sampled signed offsets perturb the brush's **channel triple in the current
colour space** (Oklab `L,a,b`; Mixbox concentrations), applied per fragment in
the sweep stamp (`brush_color`, `stamp_common.wesl`) and per texel to the `add`
paint in the exchange loop's `deposit` (`dynamics.wesl`) — both paths share the
field and parameters, so a brush looks the same whichever path renders it.
Amplitude 0 (the default) binds a 1×1 zero tile and early-outs — bit-identical to
the constant-colour deposit.


## 6.6 Brush shapes & the asset store

The default brush is a procedural soft disc, but natural media needs *organic*
tips. A brush shape is a **coverage mask**: a greyscale image where white = full
deposit and black = none. The mask drives coverage and, scaled, the height
channel too — so a worn-bristle tip lays down *broken* impasto rather than a
uniform slab.

**Brush shapes are content-addressed assets.** An imported image is identified by
the hash of its bytes; `BrushParams` references that id, never the pixels:

```rust
pub struct AssetId([u8; 32]);   // BLAKE3 of the canonical image bytes

pub enum BrushShape {
    Round { hardness: f32 },   // procedural soft disc
    Stamp(AssetId),            // sampled coverage mask from an imported image
}
// BrushParams gains:  shape: BrushShape, orientation: OrientationSource
```

`orientation` (`FollowStroke` | `Pen`) sets how the swept footprint is angled:
`FollowStroke` keeps the shape's native axis on the stroke tangent (what makes a
bristle brush read as a real stroke rather than a rubber stamp), while `Pen` pins
it to the pen's tilt azimuth in canvas space, like a calligraphy nib. The swept
integral runs along the travel direction, so the shape is pre-rotated into a
per-orientation prefix-τ volume indexed by the relative angle.

Content-addressing is the load-bearing choice:

- **The action log stays tiny.** `StrokeRecord` carries a 32-byte `AssetId`, not
  a 100 KB image; a thousand strokes with one brush reference one blob.
- **Determinism & dedup for free.** Same bytes → same id → same texture, so
  replay, goldens and peers resolve identically. Unlike shader drift across
  builds (§8), the brush image is *data the file owns*.
- **Collaboration fits the iroh model.** Content-addressed blobs are exactly what
  iroh blobs sync (§12.4): a peer seeing a stroke referencing an unknown
  `AssetId` fetches that blob by hash before rendering it.

**Asset store.** `AssetStore` maps `AssetId →` a GPU coverage texture
(single-channel `R8`, mip-mapped for clean minification). On import the image is
decoded, normalized to coverage (alpha if present, else luminance),
box-downsampled to `assets::MAX_SHAPE_DIM` (1024) so an oversized upload cannot
exceed device texture limits, hashed, uploaded and cached
(`Engine::import_brush(bytes) -> AssetId`). The store is **document-adjacent
resources**, not the action log: populated on import and on load, bundled into
the save file (§8). Selecting a brush is session state, not a historized edit.

**Stamp rendering.** `stamp.wesl` carries a per-instance rotation (cos/sin) and
samples the bound mask at the footprint's uv, so the mask's coverage is what the
swept optical depth integrates and therefore modulates both opacity and the
height `add` lays. `Round` is realized as a built-in generated mask under a
reserved id, so the shader always samples a texture — one code path.

**Assets are fetched at runtime, never embedded.** The engine is *given* image
bytes; it embeds none. Built-in assets (brush shapes, surface bump maps, the HDR)
live as static files under `stark-ui/assets/`, bundled by `asset!` with
cache-busting URLs; the frontend fetches them on demand with
`dioxus::asset_resolver::read_asset_bytes` (HTTP on web, filesystem on native)
and hands the bytes to the engine. The built-in shapes are listed in one table
(`stark-ui/src/builtins.rs`) and fetched once at startup, which is what makes an
id available to name them by: imported bytes are keyed by the hash of their
decoded coverage, so every engine (main canvas, brush-editor preview, a peer's)
lands on the same `AssetId`, and a built-in is referenced downstream exactly like
a user's imported shape — a `BrushShape::Stamp`, with no notion of "built-in"
anywhere. Adding a shape is a PNG plus a row. Brush *presets*
(`stark-ui/src/presets.rs`) are the one thing that has to wait for the fetch,
since a preset stores a content id. The large surface maps are fetched lazily,
only when a surface is selected. This keeps multi-megabyte assets out of the wasm
binary and is the path that scales as the libraries grow. (Headless tests, having
no frontend, read the same files from disk and register them directly.)


## 6.9 Drag-and-hold drawing assist

Drag out a rough line or a rough ellipse, then **keep the pen down without moving
it**. After a beat the stroke in flight snaps to the ideal shape it resembles, and
the rest of the *same* drag steers that shape — the end of the line, the angle and
size of the ellipse — until the pen lifts, which commits it. The gesture is one
stroke and one undo step, start to finish.

This is the shape-assist half of §18.1.3, and it attaches exactly where that
section predicted: **a path transform between the fitter and the renderer**.
Nothing downstream of `StrokeRecord::path` learns that assist exists. A snapped
stroke is still a list of control points, so the renderer, the wire format, the
save file, replay and the goldens are untouched, and the stroke is undoable,
replayable and collaborative for free. That is why `stark-core/src/assist.rs`
answers in control points rather than carrying a shape into the action log.

### Three pieces, deliberately separable

**Recognition** — which shape, if any, the raw pointer trace is. It works on the
*accepted reports* (`PathFitter::trace`) rather than on the fitted control points,
because a B-spline is pulled towards its control points rather than through them,
so asking whether *those* lie on a circle is asking the wrong question.

- A **line** is total least squares — perpendicular distance, not vertical offset,
  since ordinary least squares would answer differently for the same stroke drawn
  at a different angle, which on a canvas that can itself be rotated (§18.1.2) is
  not a fit at all. But it is **anchored at the first sample** rather than free,
  because the two ends of a drag are not the same kind of thing: where a stroke
  starts is placed deliberately, with the pen at rest on the point the hand chose,
  while where it ends is wherever the hand had got to. So the start is taken as
  drawn and the fit spends all its freedom on the direction — which also makes the
  residual honest, being measured against the line that will actually be drawn.
- An **ellipse** is fitted by **reweighted moments**. Points spread uniformly in an
  ellipse's own parameter have covariance exactly `½·diag(a², b²)` in its own
  frame, so the moments give the shape in closed form — but only for that measure.
  Everything difficult about the fit is in earning that measure, and the passes
  that do it are the whole of it: estimate, read off the parameter every sample
  sits at, reweight, repeat until it settles. The true ellipse is a fixed point, so
  the iteration converges to it rather than the first estimate having to land on
  it, and the correspondence is *declared* from the current estimate and never
  searched — the same discipline §6.2 applies to the stroke fit.

**What the reweighting has to correct is worth spelling out, because each is a way
a real hand draws a loop and each broke recognition outright.**

- **Speed.** Pointer reports are spread by how fast the hand moved. Resampling them
  uniformly by *arc length* does not fix it either: arc length runs fastest at the
  ends of the minor axis, so an arc-length measure reads a 2:1 ellipse as roughly
  1.7:1, and the error grows with eccentricity until a long thin loop misses the
  bar for being what it is.
- **Overshoot.** Closing a loop means coming back *past* where you started.
  Weighting each sample by the gap to its neighbours counts that wedge twice, which
  on a 6% overshoot walked the estimated centre 78px off a 400px ellipse and took
  the worst residual from 4px to 112px. So the weight is **coverage, not travel**:
  the parameter circle is cut into equal wedges, and every *occupied* wedge is
  worth the same, shared out among whatever landed in it. Going over an arc twice
  says nothing extra about the shape.
- **Undershoot.** Stopping short leaves a wedge with no data in it, and a
  closed-form inversion that assumes a whole turn then describes an arc — an 8%
  short loop was enough to fail the bar at every eccentricity. Empty wedges are
  therefore **filled from the estimate itself**, one point per wedge, which is
  sound because the truth stays a fixed point and the gap is at most an eighth of
  the circle, so what was drawn always outvotes it.

**Everything is barred on the worst sample, not the RMS.** This is the whole
difference between a bar that discriminates and one that does not: a hand's wobble
along a straight drag is noise, while a curve somebody *meant* deviates
systematically, and averaging is exactly the operation that hides the second. A
300px stroke bowed 40px reads as 4% RMS — indistinguishable from a shaky straight
line — and as 9% at its worst, which is not close to anything. Every threshold is
denominated in the gesture's **input tolerance** (§6.2), because "close enough to a
line" fixed in canvas px would mean two different things at two zoom levels.

Declining is a normal outcome. Dwelling at the end of a stroke that is neither a
line nor an ellipse leaves it exactly as drawn and the gesture carries on through
the fitter — a closed trace is offered to the ellipse fit first and *falls through*
to the line fit if it misses, so a rough rectangle simply does not snap. The cost
of a false positive (a considered stroke silently replaced) is far higher than the
cost of a miss (hold it straighter and try again), and the bars are set from that
asymmetry rather than from a hit rate.

**Adjustment** — what the rest of the drag means. A line moves the end being held,
*by the pointer's delta*: snapping moved that end off the hand by up to the fit
residual, and driving it absolutely would jump it back on the first move. An
ellipse turns and scales about its centre so the held point follows the pointer —
turning is what the feature is for, and the scale rides along because a
one-pointer drag has two degrees of freedom and the radius is the only other thing
a hand at that position could mean. The eccentricity the drawn loop established is
preserved. Both are always applied to the shape **as recognized** plus the total
displacement since, never to the previous frame's shape, so a minute of adjustment
is identical to the same drag made at once — the bargain §16.6 makes for transform.

**Realization** — the shape as a fitted path, carrying the pen channels the stroke
was actually drawn with, sampled at the same fraction of the way along. This is
what keeps a snapped stroke *painted*: a line snapped out of a stroke that swelled
in the middle still swells in the middle. Without it the feature would produce
vector art with a brush texture on it.

A **line** is placed in closed form — any collinear control polygon draws exactly
that line, so there is nothing to solve. An **ellipse** is *fitted*, for a reason
worth stating because it is not obvious: a clamped cubic B-spline's **first span is
a straight chord**, since the clamp collapses three of its four Bézier points onto
the first control point. Control points placed analytically on the ellipse
therefore leave an `O(Δ²)` bulge exactly at the seam however many of them there
are, and a least-squares solve over a dense sampling of the true ellipse is what
places the end rows to cancel it. That solve is `CubicBSpline::fit_channels` — the
same one the stroke fitter drives, at the same parameters, through the same arc
profile.

The leg count follows from that same end effect rather than from the interior
ripple (`r·Δ⁴/384`, microscopic at any leg count worth using): the first leg's
chord bows `r·Δ²/8` off the arc it stands for, the solve spreads it leaving a
little under a quarter, so `r·Δ²/24 ≤ 0.4px` fixes the count — 12 legs for a
thumbnail, 71 for a 1200px circle. Fewer control points than the fitter itself
would spend on a stroke that long. The path also runs **two legs past** a full turn
at each end, which puts the flat sixth-of-a-leg at the clamped ends underneath the
far end's correctly-curved interior instead of beside it, and is what makes a
closed loop join without a notch.

### The hold itself is the frontend's

The engine owns what a hold **means** (`GestureCommand::Hold`); noticing that the
pointer has stopped is `stark-ui/src/input.rs`. The split is the one §18.1.2 draws
for the navigator's rotate drag: how long a pause has to be and how still a hand
has to hold is *gesture feel*, a property of the device and the hand — and the
engine has no clock to measure it with anyway (§7). The dwell is measured in
**screen** pixels, not canvas pixels: holding still is a fact about the hand, and
on the canvas the same tremor would count as a hold at one zoom level and as
movement at another.

`Hold` is idempotent and a no-op for a gesture that has already snapped, for a
selection drag, and for a stroke that resembles nothing, so the frontend may send
it whenever it thinks the pointer has stopped without first asking what state the
gesture is in. Pointer moves go on arriving as `GestureCommand::To` either way —
what the drag *means* changed, how it arrives did not — so nothing about the dwell
is mirrored in the pointer handling.

Two consequences inside the session are worth naming, because both would be silent
bugs:

- A snap **bumps the gesture ordinal**. It is a discontinuity in a stream that is
  otherwise append-only — the path is replaced, not extended — and that one
  increment is what invalidates the renderer's cached head (§6.2) and makes peers
  restart their assembly instead of splicing a delta onto a path that no longer
  exists (§17.5).
- A snapped stroke reports **zero frozen spans**. Steering a shape moves every
  control point at once, so there is no settled prefix to retire — the same answer
  a marquee gives, for the same reason.

Assist can be turned off (Settings → Drawing), because it changes what an ordinary
stroke does and somebody who wants their line left crooked has to be able to say
so.

**Not built, and each a local change here:** rectangles and polygons (another
recognizer arm), arcs (an open trace with consistent turning), constraining to a
circle or to 15° increments while adjusting (a modifier read at the same seam), and
carrying the recognized shape into the action log so a committed stroke stays
editable as a shape — which is a wire-format change and belongs with §18.2.1, not
before it.
