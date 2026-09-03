# The brush engine

Tiles and channels, the fitted-path swept-segment stroke renderer, the wet-mixing dynamics loop, brush shape assets, the drag-and-hold shape assist, stroke smoothing, and the eraser — §6.1, §6.2, §6.6, §6.9, §6.11, §6.12.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.
> One name per thing: [glossary.md](glossary.md).

## 6.1 Tiles and channels

A tile is a fixed `TILE_SIZE` (256×256) square in canvas space, addressed by
integer `TileCoord(i32, i32)`. Sparsity gives the infinite canvas: only painted
tiles allocate. Each tile is **multi-channel**, which is what enables strokes
that affect more than color:

```rust
pub struct GpuTile {
    pub color:  wgpu::Texture,   // Rgba16Float — working-space channels + premult alpha
    pub height: wgpu::Texture,   // R16Float — total paint height
}
```

The color texture stores **Oklab** components (or Mixbox concentrations), not
sRGB. Linear 16-bit float comfortably holds Oklab's range and the negative `a`/`b`
chroma axes, and keeps blends perceptually uniform. Alpha is premultiplied.

> **The color alpha channel is *only* the paint's per-unit-thickness opacity** —
> a material property (how opaque the pigment is per unit of thickness). It says
> **nothing** about how much paint is on the canvas, nor even whether any is
> present. **The amount (and presence) of paint is the `height` channel**
> (precisely, `height − substrate_height`, the paint *thickness*). The two combine
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
color space in use, never hardcoded.

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
digitizer, a touchscreen and a mouse each report at a different tolerance through the
same pointer API. Only the frontend knows either fact, so it states the tolerance and
the fit becomes invariant to zoom. This is a *fitting* knob and reaches nothing
else — flattening's budget is an error against the curve, in the canvas px it
will actually be drawn in.

Both **geometric ends are pinned**: a least-squares fit does not hold them,
because a stretch of parameter with no sample assigned costs nothing, so the
curve otherwise starts before the stroke and stops short of the pointer. The
start is set and frozen at the first sample; the live end moves to the newest
sample each update (and freezes there on release), which also keeps the preview
under the cursor.

A stroke that begins after a watched hover takes its window as a **run-up**
(`PathFitter::seed_runup`; §18.1.10 is where the window comes from). The entry
is otherwise the fit's blind spot: the clamped ends squash the first span into a
fraction of a pixel of arc, so its heading is decided by where one control point
lands against tolerance-quantized samples. The run-up's reports are adopted as real
leading samples — **the fitted curve extends back through them** — and the
record marks where on that curve the press really happened
(`StrokeRecord::start`, a span-unit parameter, `#[serde(default)]` so a file
from before the field means 0: the whole curve is the stroke). The deposit
begins at the marker and everything before it is evidence, never paint; the one
flattening funnel both render paths share honours it
(`path::flatten_spans_from`), with the arc accumulator reading 0 there, so the
tapers, the `drain` falloff and the dynamics loop all measure the stroke from
the press. The front pin moves with the curve — the run-up's first report is
the pinned head, the press an *interior* sample — so the entry is smoothed
through the press exactly as every later report is, which is the point: the
pinned start was the last unsmoothed place on the stroke. Measured on a
staircase drag, the entry at the marker reads 14.4° on a line drawn at 14.04°,
0.2 px off the press, where the cold entry reads 12.4°. Run-up pressure is
replaced with the press's own (pressure begins at contact; tilt and time are
continuous through it and kept). The marker refines while the entry spans are
still free and settles the moment the frozen prefix's arc covers it — which is
why a cached head can bake spans behind it, and why presence sends it per frame
rather than in the stroke's invariant head (§17.5). A stroke with no run-up
behind it carries marker 0 and is bit-identical to what it always was; a click
at the end of a watched approach has its marker at the very end of its curve,
deposits nothing, and commits nothing.

Every report is weighted by **the arc it stands for** (`arc_weights`), which is
what makes the solve a fit to the stroke rather than to the hand. A pointer
reports on a clock, not on a ruler, so the same stretch of curve carries as many
reports as the hand took time over it; summed unweighted, a stretch the hand
dawdled on outvotes the rest by count. The trapezoid weight turns
`Σ residual²` over reports into `∫ residual² ds` over the stroke, and the growth
rule is scored the same way so that the price of a control point is charged
against the error the solve is actually minimizing.

**The pen leaving the tablet is what forced it.** A tablet keeps sampling through
the release, so a stroke ends with a run of reports carrying the pressure to zero
across a fraction of a pixel of nib drift. §6.2 says a piece of path with no
length deposits nothing — a segment's contribution is a definite integral over
travel, and the flattener discards a degenerate edge before any shader sees it —
but unweighted, those reports outvoted the whole last span of real curve: the
fitted pressure came down over 88 px of a 563 px stroke, 134 px of an 838 px one,
reaching the tip at half the weight the hand actually drew. It printed as a
blunted, thinned end whose size tracked how fast the pen was released. *Rejecting*
such reports does not work at any threshold: a release drifts, so its reports
accumulate past any fixed bar and the one that gets through arrives part-decayed.

Pen attributes ride along as **passenger channels**: pressure, tilt and time are
solved against the geometry's own assignment rather than fitted jointly with it,
so a pressure ramp cannot stretch the parameterization the way a longer path
does, and no weighting is needed to reconcile pixels with whatever units they are
in. Their **end is held at its neighbour rather than pinned to the last report**,
which is the other half of the release fix and the one thing that closes it: the
last control point is supported on the final span alone, so whatever sits in the
last sliver of the domain decides it outright, however light its weight — and
that is the least trustworthy report on the stroke. Geometry has an independent
claim on its endpoint (the mark must end where the hand did, and the eye checks
it); an attribute has none, since nobody sees where a pressure *ends*, only the
width it produces. So the attribute curve leaves the stroke flat. The cost is the
last sliver of a deliberate ramp, which no longer completes; `time` is exempt,
being a stamp on the report rather than pen state, and keeps the clock the
release really happened at (§8).

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
(`gpu::stroke::budget::flatten_tolerance`).

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

- The taper varies radius *with distance travelled*. **A segment does not have a
  radius**: it has a reference radius (its midpoint) and a *ramp*, the tip's
  fractional growth across its travel, and the tip in force a fraction `u` in is
  `radius·(1 + ramp·(u − ½))`. Both quantities that scale the tip vary with
  travel — the taper and the size modulation — so this is not a taper feature;
  the pen drives the same mechanism.

  Carrying a ramp is what makes the outline **continuous by construction**.
  Adjacent segments compute the tip at the knot they share from the same pen and
  the same taper at the same arc length, so they agree on it to the bit and there
  is no C⁰ break to alias. A single radius per segment cannot do this at any
  subdivision — it can only make the step smaller, and a step in an edge is
  visible far below the pixel it is quantized to. That is what drew a radius-500
  taper as a comb of ~5 px sawteeth. `|ramp| < 2` structurally, so the tip is
  positive at both ends without a clamp: the ends are floored at half a px and
  `|r₁ − r₀| < r₁ + r₀`.

  **The deposit has to agree at that knot too, and for a while it did not.** A
  point's exposure to a segment is the extent integrated over the offsets it
  occupied relative to the tip — and an offset is measured in whatever tip was in
  force at the moment, `r_start` when the sweep opens and `r_end` when it closes.
  Denominating both ends in the reference radius instead is right only to first
  order, and the residual is *not* a smooth error: it is a step at every knot, so
  a taper's exposure rippled at the flattener's own cadence. On a round tip that
  hides under the falloff; on an image stamp, whose mask has real structure
  across the tip, each bristle printed as a stepped streak, worst near the point
  where the ramp is largest and no subdivision can shrink it. Carrying the two
  travel coordinates as a **span** with a denominator each
  (`stamp_common::Sweep::span`) makes one segment's trailing coordinate and the
  next one's leading coordinate the same expression, so the exposure telescopes:
  a point's total over a pass is the mask's row total whatever the cut.

  The **lateral** axis cannot be made cut-free the same way and is not. A prefix
  difference reads one row, and the row a point belongs to drifts while the tip
  crosses it, because the tip is growing. The shader freezes it at the moment the
  tip is closest to the point — which is exactly right at the outline, and where
  the outline's own continuity comes from — leaving a first-order term in the
  segment length everywhere else. Freezing it anywhere the sweep telescopes
  instead (either end, or the point's own extrapolated tip) buys cut-independence
  by dropping the drift altogether, and draws a tapered stamp with a hard edge
  and unsmeared bristles: measurably further from the converged answer than what
  is there now, both on a round tip and on a stamp.

  What the cut still buys is second order — the ramp is a *chord* across a cubic
  profile, so the outline bows off it by the sagitta `|r''|·h²/8`, held under the
  flattener's own sub-pixel budget or under a quarter of the tip's shoulder where
  the falloff is wider than that. Only edges actually inside a taper are
  subdivided, and a whole zone now costs a handful of segments where the two
  earlier rules — a step in the radius *factor*, then a step in px — cost 211 and
  121 on the reference stroke, and ~700 per zone on a hard 500 px tip.

  There is deliberately no cap on the ramp itself. Where it is largest no cut can
  reduce it — cut an edge reaching a taper's point into `n` uniform pieces and
  piece `k` has ramp `1/(k + ½)`, independent of `n` — so a cap would be a bound
  on nothing. What a large ramp used to cost was accuracy in the deposit; the
  span above is what removed that, and the one guarantee left, `|ramp| < 2`, is
  what keeps both span denominators `1 ∓ ramp/2` positive as well as the tip.
  Away from the point the sagitta bound has already made it small.

  One consequence worth knowing: a segment's sweep is rasterized over a strip
  built on its *widest* tip, which puts real fragments outside the extent's
  own `|y| ≤ 1` square. The prefix-τ lookup returns zero out there explicitly —
  it must not clamp to the mask's border row, which is small but not zero, or
  every ramping segment prints a faint hard-edged rectangle the size of its quad.
- A taper is measured from the ends of the **whole** stroke, and while the
  pointer is down the far end has not happened yet. So freezing is held back: a
  span is settled only once it is a trailing taper's length clear of the live end
  *and* a leading taper's length past the start — which together also prove the
  stroke has outgrown the "scale both zones to fit" compression that keeps a
  short flick a small pointed mark rather than a sliver. Both tests use chords,
  which under-estimate arc length, so what they admit is genuinely final; and an
  admitted prefix stays admitted however the stroke continues
  (`gpu::stroke::incremental::safe_frozen`).
- **The commit is the preview.** Releasing the pointer does not render the
  stroke again: the fold runs once more so the last few samples are drawn, and
  the `CommitStroke` that follows *takes* the tiles that fold produced
  (`document::PreparedStroke`). `preview == committed` is the claim that the
  head-plus-tail render and one whole-stroke render are the same picture, and
  with the claim held there is nothing for a second render to add — only the
  whole stroke's length to pay for in one hitch, at exactly the moment the
  incremental repaint exists to avoid paying it. The offer is sound on two
  checks split by who can make them: the fold verifies that the tiles are a
  render of *this* record over *this* layer's tiles (the persistent map by
  identity), and the engine offers them only while nothing has replaced the
  document since they were drawn — the same epoch a cached head is trusted by
  (§17.6). Everything else still renders at the fold: a replay, a redo, a peer's
  copy, a file — the log carries the stroke and never its pixels. What lands
  live therefore differs from those by whatever a cut costs, which is an f16
  store of the head (the corpus's `seam` column); `Engine::strokes_reused`
  counts the commits that took the offer, because pixels cannot say which
  path ran.

  That difference is the one thing worth a **setting** rather than a decision
  (`ViewCommand::SetFastCommit`, on by default — `DEFAULT_FAST_COMMIT`; a switch
  in the ⚙ dialog, §25). Off, nothing is offered and the commit renders like
  every other consumer of the log, so a client's canvas reproduces bit for bit
  and not merely within the seam. It is per-client and never logged, like the
  undo budget beside it: what it changes is how *this* machine spends the moment
  of a release, and a peer receives the stroke as an action and renders it whole
  whatever it says. The pause it buys back is the whole stroke's render, which
  is why it is not the default.

**The pixel's own footprint (antialiasing).** The sweep evaluates a continuous
field at texel centres, and a point sample is the wrong reading wherever the
field outruns the tile grid's Nyquist: a tip a couple of px wide drops out or
ropes with its sub-pixel position (its paint falls *between* centres — a
height-conservation failure in §6.1's own terms), and a falloff narrower than a
px shimmers. The fix stays inside `τ`, because `τ` is the quantity everything
composes linearly in: the prefix-τ volume carries a **second prefix, across the
lateral axis** (`build_prefix_tau`'s `g` channel), so a fragment can read not the
profile *at* its centre but its exact box average over the pixel it stands for —
one more prefix difference, the travel axis's own trick turned sideways
(`stamp_common::prefix_span_box`), at twice the loads. Filtering is linear, so it
commutes with the sum over overlapping segments — the box average of `Στ` is the
sum of box averages — and the deposit stays exactly as cut-independent filtered
as plain; adjacent windows telescope, so the profile's lateral integral is
conserved wherever the window sits, which is what keeps a hairline's paint on
the canvas. The window is half a canvas px each way in the frame of the tip in
force, a pure function of canvas position like every input to the deposit
(§6.4). It engages only where it spans at least a mask texel (`swept_depth`):
below that the differenced prefix degenerates into a texel-wide blur scaled by
the radius, and a tip that large resolves its own profile over many px anyway —
the hardness floor (§6.6) owns that regime. The strip and the per-segment reach
both grow by `AA_RIM_PX` for the half-px the filter reads past the extent
(`sweep_vertex`, `Sweep::reach`).

**The supersampled resolve (the visible edge).** The filter above cannot chase
the *visible* edge of heavy paint: the slab law re-sharpens whatever `τ` carries
— `1 − exp(−K·m)` hugs saturation until `K·m` falls to order 1, so the 10–90%
transition occupies `ln 9 / (K·m_interior)` of the τ ramp, and a stroke whose
interior sits at `K·m ≈ 10` renders a fifth of a px of visible edge from a full
px of coverage. No per-segment correction can widen it back — shaping `τ` at the
rim needs the whole stroke's total, and an analytic average correction composes
multiplicatively where it should combine on the total, both cut-dependent
(§6.2's own law refusing its own patch). The average the eye wants is of the
**finished parcel**, taken after the cross-segment composition, and the one
place that exists is the integrate. So a stroke whose predicted visible edge
falls under ¾ px (`budget::supersample_scale` — interior mass `flow ·
TAU_PER_PASS` against the shoulder's px, a pure function of the brush, so live,
commit and replay agree) rasterizes its parcel at **2 subsample texels per px**
and the integrate box-resolves each destination texel from the finished 2×2
block (`integrate.wesl`). The resolve is shaped by the tile's own
representation: every consumer re-derives coverage from `(height, op)` through
the slab law, so a plain mean of subsample *masses* would hand the Jensen gap
straight back. Height resolves as the plain mean — §6.1's conserved channel,
kept exact — and the mass as the **visible-equivalent** one, the log-mean-exp
`−ln(mean exp(−K·mᵢ))/K`: the unique mass whose slab coverage is the mean of
the subsamples', exact where they agree, computed shifted by the block's
smallest mass so nothing underflows. A rim texel then reads as *the same paint,
thinner-bodied* — the adjustment lands entirely in the derived per-unit
opacity. The premultiplied latent is the plain mean, unpremultiplied by the
mean alpha into the coverage-weighted color, and the opacity ceiling needs no
second spelling: inverting the resolved mass recovers exactly the averaged
coverage, so the 1× branch already scales what a supersampled render of the
scaled stroke would show. The fragment's own box window shrinks to its
subsample's share (`TileXform::params.w`), the rest of the pixel being the
resolve's. Two structural notes: a gated stroke takes the carried-parcel path
even at full opacity, because over-blending per-piece averages differs from
averaging the whole parcel by the covariance of subsample alphas where pieces
meet — landing the whole accumulated parcel from pristine paint is what keeps
the resolve exact whatever the pointer's cadence; and only the transient scratch
ever holds a subsample — no document tile pays the 4×.

One boundary still drawn on purpose: the wet loop point-samples its exposure,
which is divided into bake rows integrated in the same measure
(`dynamics.wesl::deposit`), so filtering one side alone would shear the
reservoir means it recovers. Its brushes are the soft ones; the erase pass,
which shares `swept_depth`, is covered by the τ filter and stays at 1× — its
transparency runs a different law, whose resolve would live in `erase.wesl` the
day an eraser gates.

**Running dry.** `drain` fades what the brush lays, linearly with distance
travelled, until the stroke is bone dry. It is quoted **per brush radius** for the
taper's reason, and it is the stronger case of the two: `radius` is meant to read
as a pure *scale* on the mark — turn it up and get the same gesture, bigger — and
a falloff denominated in canvas px is exactly what such a scale fails to carry.
One setting that fades a 16px tip gently over a dozen of its own widths leaves a
160px one dry barely past its first, so the size knob quietly doubles as a dryness
knob, which is not a thing anyone reached for it to do. `BrushParams::drain_px` divides
by the radius on the way out of the model, and nothing below it moved: both render
paths measure arc in canvas px, so the rate they read has to be per canvas px, and
`stroke_drain` is untouched in both shaders. A radius of zero reads as
*inexhaustible* rather than as the infinity the division suggests — a tip with no
width lays nothing, and what lays nothing cannot run out.

Reinterpreting a field is the case §8's wire rule names but the *shape* rule does
not cover: the encoding is identical and every stroke already in a log renders
differently, so it bumped the ALPN (to 8). A file may be read by a build that
disagrees with it — an alpha-era document simply comes back with tamer drains,
which is the trade §19 makes deliberately — but a live session may not.

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
back (`gpu::stroke::segments::chunk_segments`). Length therefore costs a dynamics stroke
pieces, not correctness — where it used to degrade past `MAX_REGION_DIM` to the
plain swept deposit, which is not a coarser version of the same brush but a
different one: the swept path only ever *adds* paint, so a brush whose purpose
was to lift it stopped doing the one thing it was for, on exactly the long
strokes and fat tips that wanted it most.

One thing must be decided from the record rather than the piece in hand, because
a live tail and the commit that replaces it must draw the same pixels: whether
the stroke runs the stamp loop at all. It is the **effect's own variant** —
`BrushEffect::Wet` loops, `Paint` and `Erase` sweep — the strongest form of that
guarantee, since a variant carries no number for a piece and its commit to read
differently. It was a rate predicate on the brush's fluxes once, sound only
because the loop at zero rates degenerated to the swept deposit exactly; the
pixel-footprint filter broke that degeneracy (the swept deposit is antialiased,
the loop point-samples), which turned "fast path" into a real feature difference
— and a feature difference is a tool identity, so it moved into the enum where
`Erase` already lives. What the loop still asks of a wet brush is
the floor no cutting gets under: the reservoir pickup reduces over the whole tip
at once, so a region can never be smaller than one tip's extent plus one
segment's travel. The travel is the one knob subdivision still has, so it is
spent before anything is given up — a tip so wide that a full-length segment
would overflow the region flattens to shorter segments instead
(`gpu::stroke::budget::fit_len`), which is never wrong, only more numerous: the
exchange step a segment's length sets is a first-order discretization that
tightens as segments shrink. The renderer warns once per stroke when the
shortening binds, because the loop pays per segment. Only a brush whose tip
*alone* overflows the region still degrades to the swept deposit. See
`gpu::stroke::dynamics::dynamics_setup`.

**That last limit is published rather than discovered.** The region is a single
texture bounded by `MAX_TEXTURE_DIM_2D`, which is *the* number: the engine asks the
device for exactly it and never allocates past it, so it is a statement about which
devices Stark runs on rather than a tuning knob. At **8192** it is where WebGPU
itself sits — `Limits::default()`'s cap and the floor a conformant implementation
must meet — so every device Stark can run on already qualifies. One tip has to fit
a region whole, so this puts a ceiling on the tip's **reach**, `size × elongation`,
of about 3936 canvas px (less for a bleeding brush, whose firings reach back past
the segment they follow). That is past `MAX_RADIUS × 7.8`, so the editor gives up
nothing until the very top of its size slider.

**The region's two bounds are different numbers, and the difference is the point.**
`REGION_BUDGET_DIM` (2048, ~67 MB a piece) is what the chunker *aims* at, and
`MAX_REGION_DIM` is the ceiling a piece may reach when a single segment demands it
— which cutting can never get under, since cutting is by segment. Raising the
budget to the ceiling instead would let an ordinary long stroke grow its pieces
there too: a 10 px tip crossing the canvas draws exactly as well in 67 MB pieces as
in 1 GB ones. So the big region is paid only by the stroke that asks — around
124 MB for the widest brush the editor offers drawn along its facing axis, ~500 MB
for that same brush drawn at 45°, where an axis-aligned box around a long diagonal
tip is at its worst — transient, and freed per piece. `budget::max_tip_reach` is that ceiling and
`budget::max_stretch` is it expressed in the units the editor's slider moves in;
the frontend clamps every brush against them (`state::hold_the_tip_drawable`), so
no brush the app can build degrades, and `TipTooLarge` is left for a record from a
peer or another build. Two properties keep that honest and are tested as such: the
published cap is *exactly* the frontier `fit_len` refuses at — the same arithmetic
inverted, not a second copy — and what the editor offers always draws.

The ceiling is why the stretch slider's top moves with size. Up to about a 492 px
radius nothing is given up, the knob saturating at an elongation of 8 before the
region does; only the last of the size slider trades, and it is the **stretch**
that yields rather than the size, because `size` has four writers (the panel
slider, the editor row, the tuning drag and the `[`/`]` keys) and is what the
artist reaches for. Lifting the ceiling further would mean giving the loop more
than one region — narrower than it looks, since the reservoir is a fixed 64×64
brush-local texture and the deposit is per-texel local, so the only pass needing
the whole tip resident is `exchange`'s gather over the canvas beneath it.

**Continuous stamping (swept segments).** Discrete dabs are visible with hard
tips. The fix: stamp each short *segment* of the flattened curve as one quad
whose coverage is the brush **swept** along it — the path integral of the
extent, instead of point samples. The enabling identity: alpha-"over" is
multiplicative in `(1−α)`, hence additive in **optical depth** `τ = −ln(1−α)`. So:

- Precompute, per brush, the **prefix integral of `τ` along the travel axis**.
  A length-`d` segment's swept depth at a point is `prefix(u) − prefix(u−d)` for
  that row — an O(1) lookup.
- A segment quad lays a **parcel of paint**, not a coverage: `add · τ` of height,
  per-unit **opaque** — faded only by the drain; how much of the finished stroke
  shows is the effect's own dial, applied where the whole parcel lands (the
  opacity ceiling, below) — the two meeting only in the slab law (§6.1). What
  the color target carries is therefore that parcel's *visible alpha*,
  `α_seg = 1 − exp(−K · add · τ)`. Because the existing
  premultiplied-"over" blend across overlapping segment quads combines as
  `1 − ∏(1−α) = 1 − exp(−K·Σ m)`, it sums the parcels **exactly** — no
  double-counting at joints, no second pass — and the latent it carries stays
  ordered, so a stroke crossing itself covers rather than averages. The scratch's
  aux sums the height and the mass unsaturated alongside it, and `integrate.wesl`
  stacks the one parcel that comes out on the base through the shared law in
  `paint_common.wesl` — the very one a fill lands through and the one the stamp
  loop's `deposit` uses, so the two paths cannot drift.

  The same argument is made once more on the host side. Which path a brush takes is
  decided from axes that have nothing to do with color or flow, so everything the
  two share — the brush's channels in the working space, the canvas → substrate scale,
  the color-dynamics lookup — is resolved once into a `StrokeConstants` and read by
  both, rather than derived twice from the same record. Two derivations agreeing is
  a coincidence to maintain; one derivation is a fact.

  Weighting by the brush's per-unit **opacity** instead — which the fast path did
  — is the same defect §6.3 names in the layer composite, one level down. `add`
  is the only thing that decides how much paint lands, so leaving it out of the
  color meant a 5%-flow glaze drew as nothing over bare canvas (the media pass
  covers for it, since visible alpha is `opacity × height` there and the height
  was right) and repainted at full strength over existing paint, where nothing
  does. `tests/dynamics.rs::a_glaze_lands_the_same_whether_or_not_the_stamp_loop_runs`
  pins the agreement, by drawing the same glaze with `deposit` at 0 and at 0.01 —
  which routes it through the whole sequential loop with an empty reservoir, so
  every texel is still the brush's own `add` paint.
- **Every** channel a segment deposits must be a function of that segment's `τ`
  in one of exactly two shapes: *additive* in `τ` (an amount — the height and the
  optical mass the aux target sums), or `1 − exp(−k·τ)` (a rate — the visible
  alpha the color target over-blends). Those are the two that survive re-cutting
  the path, because `τ` is what sums. Any other shape makes the stroke depend on
  the *number* of segments: a per-segment `√`, for instance, deposits
  `Σ√(τ/N) = √(N·τ)`, so the stroke silently gains weight with sampling density.
  Invisible while sampling is uniform and immediately visible once it adapts —
  which is why the two forms are a standing constraint on the stamp shaders, not
  a detail of one.

Segments need only be short enough that the line + constant-radius approximation
holds, so the sweep uses *fewer* primitives than the dab model. Caveats:
per-stamp angle jitter no longer applies (the brush follows the tangent
continuously); and the round tip's prefix depends on `hardness`, so it is
generated per stroke (image brushes precompute theirs at import, §6.6).

A stroke deposits **exactly its own travel** — a definite integral over no
travel is no paint, and the tool says so honestly. A press that has not moved
lays nothing and, having nothing to show for itself, commits nothing either
(`Session::end_stroke`, the answer a marquee click has always been given), so no
undo step is spent invisibly. There used to be a fabricated minimum here — the
`DAB_TRAVEL` dwell, 0.6 radii swept about the stroke's midpoint so that a click
dotted and a short drag was topped up — retired for two reasons that arrived
together (§18.1.10, §6.2's run-up): the brush cursor now previews exactly what a
press would lay *before* it is made, which answers the "why did my click do
nothing?" the dwell existed to forestall; and the dwell overrode precisely the
entry geometry the run-up now conditions honestly — a short flick's mark was its
midpoint-padded stand-in, length forced to 0.6 radii whatever the hand did, led
by an arbitrary `+x` where a click had no tangent to give.

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
extent. All on the GPU with no readback (`gpu/stroke/dynamics.rs`,
`dynamics.wesl`):

1. **Region composite.** The base tiles under the stroke (the affected set plus a
   one-tile ring) are composited once into a 1:1 canvas **region** texture
   (color + the wide aux). This is the working canvas the stroke evolves.
   Bounded by `MAX_REGION_DIM`, which bounds transient memory rather than the
   stroke: a stroke too big for one region is cut into pieces that fit.
2. **The loop.** The stroke's flattened segments run *in order* inside a **single
   compute pass** — the implicit barriers between dispatches give the sequential
   semantics, and usage scopes are per-dispatch, so the region can be sampled by
   one dispatch and storage-written by the next with no copies and no pass churn.
   Per-dispatch parameters ride one dynamic-offset uniform buffer — as do the
   write-back's per-tile offsets below, and the swept path's per-tile transforms.
   That is a constraint rather than tidiness: a live stroke re-renders on every
   pointer move, and on the web a buffer and a bind group *per tile per move* is a
   rate of small allocations that JS GC cannot keep up with — the same pressure that
   makes every transient here `destroy()`d on drop rather than left to collect.
   A plan slot is one of exactly **three** shapes, and only the first touches the
   tool: a painting **segment**, a **bleed** firing (see `bleed` below), and the
   single **settle** that ends a stroke. The reservoir ping-pong therefore advances
   on segments and on nothing else.
   - Per segment: **bake** (integrate the reservoir
     along the travel axis), **exchange** (the tool's half of the transfer, one
     thread per reservoir texel, which also carries the **snapshot** — the copy of
     the segment quad's region texels into an `under` scratch that lets the deposit
     read-modify-write — in the tail of its own grid) and **deposit**, one thread
     per extent texel. The snapshot rides along because it depends on nothing
     the exchange writes, so the barrier that used to separate them bought no
     ordering: three serialized dispatches per segment where there were four. The
     range that ends the stroke closes with a standalone `snapshot` + `bake` +
     **settle** over the final extent (see *The pen-up* below) — standalone
     because there the settle *reads* what the snapshot and the bake write. A texel's **exposure** to the
     segment is the prefix-τ difference `e(x) = prefix(u) − prefix(u−d)`, and
     exposures add across overlapping quads of consecutive segments, so what the
     loop applies must be built from `e` in a way that survives re-cutting the path.
   - **The two directions are solved as decoupled one-sided decays, and that is a
     statement about geometry rather than a linearisation.** The obvious model is a
     closed pair — a canvas point and the tool above it trading the one conserved
     quantity, `dh/de = −k_lift·h + k_dep·R + A` against
     `dR/de = +k_lift·h − k_dep·R` — and that is what the loop used to solve, in
     closed form, both halves reading one solution. But it describes a point held
     under *one* reservoir cell for the whole segment, and that is not what happens:
     the point slides through a stream of cells, each pairing lasting the instant
     the cell is overhead, far too briefly for the partner to relax. Taking that
     limit splits the system in two:

     ```
     h(e) = h₀·exp(−k_lift·e)        each side's own loss, closed form
     R(e) = R₀·exp(−k_dep·e)
     ```

     with each side's *gain* the integral of the other's loss over the track they
     shared. So the canvas keeps `exp(−k_lift·e)` of its height and takes
     `1 − exp(−k_dep·e)` of the tool's load, and the tool takes exactly the
     complement of each. Being exponential in `e`, running it over `e₁` then `e₂`
     *is* running it over `e₁+e₂`: the whole stroke applies `1 − (1−axis)^∫e`, the
     continuous path integral whatever the spacing, with no dabbing.

     **It conserves, and the proof is a change of variable rather than an appeal to
     a stochastic matrix.** Cell `u` loses `R₀(u)(1 − exp(−k_dep·τ(u)·lr))`; the
     canvas gains `∫k_dep·τ(x−p)·R₀(x−p)·exp(−k_dep·τ(x−p)·p)dp` along the pass, and
     integrating that over `x` under `u = x−p` returns the tool's loss to the last
     term. The lift direction telescopes the same way under `q = P(1)−P(u)`. Each
     direction balances on its own, so neither needs the other evaluated at the same
     instant for the books to close.
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
     mid-pass sample — exact for any segment length.
   - **Paint does not migrate within the tip: the `wick` pass (2026-07-31 →
     2026-08-10) is retired.** The disease it treated is real, and worth keeping on
     the record. A cell's exposure is keyed on its own optical depth, and τ is flat
     across the interior (the coverage clamp caps it) then falls away over a hard
     tip's shoulder — at `hardness = 0.95`, some twenty times slower at the last
     cells still in contact. Left as isolated cells, that disparity strands paint:
     once a stroke's own source runs dry (`drain`) and the tip is only smearing, the
     interior empties in step with the fading trail while the shoulder ring still
     holds what it lifted hundreds of pixels back. With the settle of that era, the
     ring printed — a rim of surplus paint with a scraped groove inside it, curving
     into a tip-shaped chevron where the stroke stops
     (`golden_lift_end_regression`) — and the wick relaxed the disparity away with a
     four-neighbour flux over the reservoir, on a travel cadence of its own.

     It was treating a symptom. The shoulder ring's slow payout is what the visible
     trail is *made of* (it serves last), and once the pen-up settle became the
     remaining pass's exact delivery integral (*The pen-up* below), that payout is
     delivered in the right order and nothing is left stranded for a lateral
     smoothing pass to rescue. Measured at the removal, on a fifteen-radius drained
     smear of the golden's own brush and on the golden's 5.4-radius stroke, wick on
     vs off: no lateral rise above one level in either arm, the fade past the
     stroke end monotone both ways, worst 4 levels anywhere in the frame.
     `a_drained_smear_leaves_no_ring_at_the_lift_end` pins the property the wick
     used to guard, and what the removal buys is two fewer serialized dispatches
     per half-radius of travel in a loop that is dispatch-bound at small radii.

     One numerical lesson from its fixes outlives it, and the bleed's ladder below
     cites it: a sparse stencil at integer distance `d` couples only cells of the
     same parity in `d`, so widening a flux's *reach* to carry more variance
     decouples the grid's sublattices — at even reach the checkerboard mode's
     eigenvalue hits 1 and never decays. A step must ride its cadence (more
     firings), or fill in its taps down to true neighbours; it may not ride a
     sparse reach.
   - **What is left is first order in the segment length, and it is the loop's last
     open defect.** Both halves still read the state the segment *entered* with — the
     canvas side integrates over which reservoir cell is above it (`bake`), the tool side
     reads the canvas with a single tap at the segment's midpoint — so each is exact
     about the geometry of the slide and stale about the partner's value. Halving
     `RESERVOIR_EXCHANGE_STEP` halves the error, cleanly, with no knee to sit on.

     It prints. A tip lifts at a point and lays back down swept, so the smear
     translates the canvas by exactly one segment length per segment — a delay line
     ringing at the segment cadence — and a stroke long enough to run dry comes out
     ruled with one tip-shaped arc per segment. Worse, the arcs move: the flattener
     bisects, so a span's segment length depends on the *whole path's* length, and the
     same visible stretch of stroke renders differently depending on where the pen went
     afterwards. `golden_drained_brush_length_independent` paints one visible stretch
     with five different tails and pins that.

     Why it went unnoticed at a step of 0.5 is worth keeping. Nearly every golden paints
     with the shared `brush()` helper, whose `drain` used to impose its own `0.02/drain`
     = 13.3 px cap on segment length — tighter than the step for any tip wider than
     that, so the goldens rendered at 13.3 px segments *whatever the step said*, and
     moving it moved nothing. Only once `drain` became a per-fragment falloff did the
     step become the binding constraint and start deciding pixels. A golden that does
     not move is evidence about the test, not about the change.

     **No reformulation of the pair kernel can help, and that is provable.** Write it as
     the transfer matrix `M(e) = [[keep, dep], [1−keep, 1−dep]]`, whose columns sum to 1
     — that column-stochasticity *is* the complementarity, and it is why the transfer
     conserves. Its eigenvalues are `1` and `exp(−s·e)` with `s = k_lift + k_deposit`,
     and its stationary split `k_deposit : k_lift` does not depend on `e`, so
     `M(e/K)^K = M(e)` **exactly**, for every K, exposure and rate pair. The kernel
     already composes perfectly under subdivision; a product of column-stochastic
     matrices is the matrix it started from. Subdividing changes something only if the
     *partner* is frozen across the sub-steps, and that is not a refinement but a
     one-parameter deformation away from the pair model.

     So the error is not in the kernel *while the kernel is a closed pair*. It is in
     the two mean-field approximations either side of it, and those are bounded by the
     segment length alone. What the theorem really argues is for **leaving** the closed
     pair — which is the decoupled sliding form above, whose `K → ∞` limit it is.

     That move was once recorded as blocked, on the grounds that the sliding form gives
     up the column-stochasticity and so needs the **flux** baked rather than the load
     before it can conserve. Too pessimistic on both counts. The shares still sum to one
     identically — `keep + (1−keep)` is a tautology whatever `keep` is — so the
     39%-of-height failure, which came of the two sides solving *different* equations,
     cannot recur; and each direction balances in the aggregate on its own, by the
     change of variable above, with no flux bookkeeping at all. What genuinely remains is
     that the two sides evaluate their exponentials at different *quadratures* of the
     same exposure, and a harder-saturating exponential turns a small quadrature
     disagreement into a larger height one. Measured, that is worth one level: the worst
     lightening of a smeared field goes 50 → 51 against a bound of 60, and a 240-sample
     zig-zag smear's ink growth 0.97940 → 0.97938 against a bound of 1.0.

     The gain — 2–4× at every step, converging to the same answer — is banked as
     accuracy rather than spent as step size.
     Sliding at 0.25 would halve the segment count and still beat the old kernel at
     0.125 on absolute error, but its length-dependence is the row already weighed and
     rejected once, and that is the column a user sees.
   - **What the step costs is scaled by the transfer rate**, not charged flat. The
     error is first order in the transfer a segment *completes*, `(k_lift + k_deposit)·τ·lr`,
     so holding that fixed is what makes one constant mean the same thing to every
     brush; `stroke::budget::exchange_travel` does it in closed form, since the rates enter as
     `λ = ln(1 − axis)/TAU_PER_PASS` and the `τ` cancels. Measured across
     `lift = deposit` from 0.4 to 0.95, the length-dependence stays in a 1.1–2.2 level
     band while the segment length varies 6×. It only ever relaxes — a brush trading
     faster than the calibration point stays at it — and `charge` is excluded, being a
     starting load rather than a rate: a brush that only charges never enters the
     exchange at all.
3. **Write-back.** Each affected tile's full `TILE_TEX` block is sliced out of
   the shared region into a fresh CoW tile (`slice.wesl`, narrowing the wide aux
   to the persistent `(height)`). Aprons are bit-identical to neighbour interiors
   **by construction** — both are cut from the same texture — and the ring in the
   composite gives rewritten tiles real neighbour content (§6.4; the
   `apron_makes_dynamics_writeback_seamless_under_zoom` regression guards it).

*Conservation (§6.1).* Paint moves by transferring **height** — the one conserved
quantity. Color and per-unit opacity ride as optical-mass (opacity·height)
weighted blends, and a parcel's blend weight is its own *visible* alpha
(`1 − exp(−K·mass)`, the same translucent-slab law as the media pass), so thick
deposits cover while thin glazes tint. The lift never touches the source's color
or alpha: the source fades because its **thickness** drops. Both sides of every
transfer take complementary shares of the same decay, over the same segment and
from the same pre-state (the canvas side measuring its exposure through the
prefix-τ, the reservoir side as `τ(l) · Δs/r` — two quadratures of the same
bilinear form, which agree texel for paired texel), so with `add = 0` total height
(canvas + tool) is conserved up to resampling error, whatever the segment length.

*The pen-up.* A stroke stops with the tip still in contact and the transfer still
in flight. Everywhere else on the trail a point sees the whole extent pass over
it and leave by the trailing rim, where `τ` has fallen back to 0; the **last**
extent never gets that, so what is in flight there is stranded — and since the
reservoir is per-stroke, stranded means gone. It shows both ways round: a carrying
stroke ends a tip-shaped disc short of its own trail, and an eraser leaves a
tip-shaped patch of the paint it was standing on.

So the pen-up settles the pair once, through the same `exchange_at` kernel, over
an exposure bounded by the pass on **both** sides:

```
e = (owed⁻ᵖ + received⁻ᵖ)⁻¹ᐟᵖ,   owed = prefix(l),  received = rowtotal − prefix(l)
```

Both bounds are load-bearing, and both are readings of the very prefix-τ volume
the swept deposit integrates against. `received` is what the tip has already given
this texel — a point it has barely reached has no film to break, and without that
bound the settle steps against untouched canvas a radius *ahead* of the pen-up
(40% of the paint's own range on a smear; a fully-scraped disc with a hard rim on
an eraser). `owed` is what it still had to give — a point it had all but finished
with has nothing left to hand over, and without that bound the settle steps against
the *trail*, which got no settle at all, right where the two meet. They vanish on
the extent's rim (`owed` at the trailing edge, `received` at the leading one,
the row total itself laterally), so the settle fades to nothing all the way round.

**They are combined with a soft minimum rather than a `min`, and that is not
cosmetic.** The two cross at the tip centre, where `owed = received`; a `min` is
continuous there and its *slope* is not, and everything the settle does is
exponential in this exposure, so the corner lands in the height field as a corner
— which is a **step in the relief normal**, and the media pass (§6.3) prints it
as a hard line straight across the middle of the last extent, perpendicular to
the travel. Measured on a smear into thick paint under a structured sky, the
specular stepped 107 levels across one texel at exactly `l = 0`, against 2
anywhere along the trail. The p-norm above is `≤ min` everywhere, so both bounds
keep their ceiling exactly, and it differs from `min` by `(min/max)ᵖ` — nothing
where one bound is clearly the binding one. `p = 4`
(`dynamics.wesl::SETTLE_BOUND_SHARPNESS`) is bracketed from both sides: higher and
the handover, though smooth, packs its curvature into less than a texel and the
render cannot tell it from the corner; lower and the settle discounts the
transfer a texel just behind the centre still has in flight, which is the load
this dispatch exists to deliver.

**The parcel is the remaining pass's delivery, not the cell overhead.** At the end
of a long smear the reservoir is radially skewed: interior cells trade fast, so
they sit near equilibrium with the thin mid-pass canvas under them — nearly empty —
while the trailing shoulder, whose `τ` is a hundredth of the interior's, still
carries paint lifted hundreds of px back and pays it out slowly. That slow payout
is what the visible trail is *made of*, so a settle that pairs each canvas texel
with the cell that happens to sit above it lays almost nothing exactly where the
continued pass would have slid the loaded trailing cells over every interior
point — the whole extent printed as a tip-shaped disc of missing paint, stepped
where the kernel saturates (`golden_lift_end_regression` pins all three lengths).
So the settle walks the delivery integral instead — per texel, along the row of a
pen-up `bake` whose free channel carries the bare exposure prefix:

```
delivered(l) = k_dep · ∫ R·dτ · exp(−k_dep·τ(u)·(l−u)) · exp(−k_lift·P(u))
```

The first exponential is the cell **spending itself en route** — what it lays on
the points between its pen-up position and this one comes out of the same load —
which confines a saturated interior cell to its own neighbourhood while letting a
shoulder cell carry its payout the whole way across the extent; summed over
every point a cell serves, it telescopes to at most the cell's own load, so the
settle cannot mint for any pair of rates. The second is the parcel's **survival**
under the lift of the cells that serve after it — the same change of variable the
sliding kernel's conservation argument runs on. The `min(owed, received)` bound
lands as the smooth ratio `dep(e)/dep(owed)` scaling the whole delivery — cutting
the pass where the truncation falls was tried and prints a fresh cliff mid-
extent, because the ring serves last and drops out all at once. The en-route
factor couples cell to served point, so no single baked prefix carries it; the
settle runs once per stroke, which is what makes O(`BAKE_RES`) taps per texel the
right trade where the per-segment deposit could never afford it.

**Why the prefix and not `τ`.** The instantaneous depth would put the fall-off
across the few pixels of the tip's coverage knee — `κ = −ln(1−coverage)` rises
steeply wherever coverage approaches 1 — and simply print the tip's edge. `prefix`
is that same `τ` integrated *along the travel*, so it ramps across the whole
radius. It is the same reason the brush's own `add` caps smoothly and the exchange
does not: `add` is linear in the prefix, the exchange saturates exponentially in
it. Tapering the tip out at the pen-up instead makes matters worse, not better —
measured, the trailing `taper` raises the total curvature along the mark's
centreline by 6× and staircases at the taper's own radius steps.

A range that does not reach the stroke's end never settles: it hands its reservoir
on to the range that resumes, so nothing is stranded, and a live tail computes the
same settle its commit will.

*Order-dependence is real.* Pickup reads the region as already modified by
earlier segments, so a stroke smears **its own trail** when it crosses it; drag
falls out naturally; and there is no band, column or stamp structure to alias.

*The run-dry falloff is bounded at a full load, not only at an empty one.*
`drain` is sampled at each texel's **own** arc length on both paths
(`stroke_drain`), which is exact for a linear falloff wherever a texel sees the
whole pass. The texels behind the stroke's *start* never do — the tip's trailing
half is already over them at pen-down — so their arc is negative, and unbounded
the falloff hands them more than a full brush. The swept path lays the excess;
the dynamics loop cannot, because a load over full puts mass above the height
carrying it and the region stores the quotient `m / h` as a per-unit opacity.
Clamping the load at 1 is what keeps the two paths drawing the same start cap.

*The axes* (`BrushDynamics` on the wet effect — a flat record in the action log),
and the **flow** that scales them (`WetEffect::flow`). The axes say what the tool
*is* — a smudge, a knife, a blender — each quoted per pass at the neutral flow of
1; the flow says how hard a pass works, and it scales *everything* the tool does,
which is what keeps "Flow" one knob with one meaning across every effect: on a
blend brush the slider blends more or less instead of laying paint, which is the
whole reason the source axis and the rate are two numbers. Mechanically the flow
is exposure scaling, applied on the CPU where the segment resolves its rates
(`generate_segments_in`, `dynamics_plan`): what is linear in exposure — the `add`
mint, the `bleed` diffusivity — takes the factor outright, and the vertical
fractions take it on their λ exponents, `flow · ln(1 − axis)`, so a pass at flow
½ trades exactly what half a pass at flow 1 would and the exchange still
composes exactly across overlapping segments. The shaders never learn it exists.

- `add` — lay the brush's own paint; the only inexhaustible **source**, in
  [0, 1]: the share of the brush's own paint in what the tool does. What lands
  per unit swept optical depth is `add · flow`, so at `add = 1` a wet brush lays
  exactly what `Paint` would at the same flow and the Flow slider cannot jump on
  the switch. A pure-`add` brush takes the swept fast path, untouched by the
  loop.
- `lift` — vertical flux canvas → tool (an eraser when alone).
- `deposit` — vertical flux tool → canvas (`lift`+`deposit` with `add = 0` is a
  true mass-conserving smudge).
- `charge` — a finite glob pre-loaded onto the tool (the palette-knife scoop); it
  depletes as the tool deposits and refills as it lifts.
- `bleed` — **lateral** flux within the canvas itself: the paint already under the
  tip relaxes towards a neighbourhood **a fraction of the tip wide** at
  `1 − exp(−k_bleed·e)`, the same saturating form as the vertical rates, keyed on
  the same swept exposure — so a texel the tip never covers never moves, and
  overlapping segments compose to first order. Alone it is a blur brush; under
  `add` it melts the height ridges of the strokes being painted over instead of
  embossing them through the new paint.
  **The axis is a diffusivity, not a rate**, and that is the one place it differs
  from its three neighbours. What the stencil realises per unit exposure is
  `D = k_bleed·Σ(share·d²)` — a rate times a second moment — and only the second
  factor has headroom in it: the blend `w = 1 − exp(−k·e)` clips at 1, so at a
  fixed stencil `D` has a hard ceiling of `Σ(share·d²)` per firing however hard
  the rate is driven. Measured on a 40 px tip, whose firing carries `e ≈ 1.7`, the
  whole of `bleed` from 0.95 to 1.0 took `w` from 0.52 to 0.99 — ×1.9 in `D`, ×1.4
  in distance, and nothing past it. That ceiling is the stencil's geometry, not a
  stability bound, and `Σ(share·d²)` is quadratic in the reach. So the host is
  handed `D` (in radius² per pass, the unit that makes the axis mean one look at
  every brush size) and **solves for the pair**: the reach that delivers this
  window's variance at a fixed well-conditioned blend, and the rate that lands the
  window's nominal exposure on that blend (`stroke::budget::bleed_stencil`). The
  knob is then linear in `D` across its whole travel, and `σ = sqrt(2·D·τ)` —
  scrubbing keeps buying distance, as a blender does. Holding the blend near ½
  rather than at saturation is the other half of it: the stencil's worst-case
  eigenvalue is `1 − w`, so a firing at `w → 1` annihilates its worst mode — a hard
  local average, not a Laplacian — and consecutive firings stop composing.
  `BLEED_DIFFUSIVITY` is *derived* from the two ceilings (`BLEED_REACH_MAX`,
  `BLEED_BLEND`) rather than chosen, so full crank sits on both at once and the
  three cannot drift apart.
  **The taps are a ladder up the reach, not a few isolated scales**, and that is
  what makes one firing read as a blur rather than as a copy. A tap at `±d` lays a
  *displaced ghost* of the source: against a hard edge — what this axis is most
  often pointed at — the response is that edge again, `d` away. Loading most of the
  shed onto a single tap out at the reach printed exactly that, and a texel sees
  only `2/BLEED_TRAVEL_QUANTUM` firings, so the sum kept the structure instead of
  filling it in: a bleed-only pass across a painted bar came out as stacked slabs a
  reach tall with steps as sharp as the bar's own edge. Rungs at `j·reach/T`
  sharing the shed equally turn that into a staircase of `T` steps of `reach/T`,
  which consecutive firings — landing at different reaches, since the reach follows
  the window — fill in. It costs second moment (`(T+1)(2T+1)/(6T²)`, a third in the
  limit), and since variance adds linearly in travel, buying that back is exactly a
  cadence finer by the same factor: the ladder and `BLEED_TRAVEL_QUANTUM` moved
  together, and `BLEED_DIFFUSIVITY`, derived through both, left the top of the knob
  where it was.
  It runs inside `deposit` in **flux form** (both threads of a neighbour pair
  compute one number from the same `under` snapshot and apply it with opposite
  signs, `min` of the pair's exposures as the mobility), so
  it is a pure internal redistribution: height is conserved, the tool's books are
  untouched, and no paint leaks to texels outside the sweep, whose threads never
  write. Ahead of the lift, since its stability bound is a share of the *entering*
  height; the tool's half sampled the un-bled canvas, a disagreement of
  `O(k_bleed·k_lift·e²)` — second order, the class the loop already carries. The
  pen-up settles none of it: the axis has no reservoir, so a break of contact
  strands nothing.
  **It fires on dedicated slots at a travel cadence of its own, not on the painting
  segments** (`BLEED_TRAVEL_QUANTUM`, `stroke::dynamics::bleed_fires`): one
  quad per crossing of a quarter radius of *absolute arc*, whose sweep is exactly one
  quantum of path, bent along the crossing segment's own arc, and whose vertical
  rates and source are all zero — so its exposure is an ordinary, well-conditioned
  prefix difference, and the painting segments carry `λ_bleed = 0` and take the
  no-bleed path bit-for-bit. **One quantum per firing, not one firing per
  segment**, and that is what keeps the axis a diffusivity: a window asks for
  variance in proportion to its own travel while a firing can only carry
  `2·Σ(share·d²)`, so a window merged across N quanta is clamped back to roughly
  `1/N` of the axis. A segment at the travel cap crosses this cadence
  four times, so that was an ordinary fast stroke diffusing a tenth short, not a corner
  case; variance adds linearly in travel across firings, so N of them deliver N
  quanta exactly — more steps, not bigger ones, as in any explicit diffusion
  solver. The count per segment is capped (`MAX_BLEED_FIRES_PER_SEGMENT`) because
  the flattener buys segment length off the brush's *nominal* radius while the
  cadence is the modulated one, so a pen thinning the tip would otherwise let a
  degenerate stroke choose the plan's size. Per-segment firing is
  broken twice over on real slow input, which the fitter keeps at a control point
  per pointer sample (a field repro: 177 knots over 68 px): the per-texel exposure
  of a 0.4 px segment is prefix-cancellation noise, and the per-segment flux sits
  under the f16 ULP of the heights it edits. That second failure exposed a defect
  older than the axis — a storage write re-encodes f32→f16, and a backend may
  truncate that conversion toward zero (D3D12 does), so *re-storing an
  algebraically identical texel* walks it down one ULP per rewrite — which is why
  the `deposit` (and the settle) end with a **rewrite guard**: when the lift
  kept everything, the parcel is empty and no flux moved, the texel is not
  re-stored at all. Measured on the repro, the guard alone took a 28-level
  directional ghost to bit-exact zero. **The guard is not the whole answer,
  though, because it only fires on exactly-zero rates.** A brush that is merely
  *nearly* inert stores every texel of every segment's extent, and `h·keep`
  with `keep` a whisker under 1 is below `h` by far less than an ULP, so
  truncation takes the whole ULP every single time — a drift with no random half
  to cancel it. What decides how far it goes is the number of segments whose sweep
  covers a texel, which is a property of the *path*: a straight 256 px drag lost
  0.04% of a field's height at rates of `1e−4`, the same span walked as a 20-cycle
  wiggle lost **3.65%**, and the premultiplied latent walked down with the height,
  so a stroke that by every term of the model did nothing left a mark that was both
  thinner and darker than the paint it crossed. So the loop's stores now go through
  `lib::store::f16_nearest`, which snaps a value onto the binary16 lattice before
  handing it over: whichever way the backend converts, it converts exactly, an
  identity pass is an identity store on every adapter, and a real change rounds to
  nearest instead of down. That is structural rather than a rule each of the nine
  store sites could forget, and it is what
  `a_smear_that_transfers_nothing_leaves_the_canvas_alone` states.
  The stencil's taps sit at three scales per direction — 1 px, a
  `BLEED_MID_DIVISOR`-th of the reach, the reach — with the reach solved per
  firing and arriving in the slot, so every thread of a flux pair derives one set
  of integers from one uniform. Its ceiling is the extent (`BLEED_REACH_MAX`):
  a tap leaving the sweep has `w_n = 0` and carries nothing, so past about half
  the radius the long tap is truncated over most of the tip and `D` falls short
  *unevenly across the extent*, which is worse than falling short at all. Past
  that the honest way to diffuse further is a finer cadence — more firings, not
  longer taps. The three shares are declared one scalar apiece and **generated**
  into the host (§6.10), because the host computes `Σ(share·d²)` to solve for the
  reach and a second copy of them is a way for the two sides to disagree about how
  much a firing diffuses with nothing failing. The 1 px taps are the floor
  rather than the rate — sparse ±d taps alone decouple the grid into d²
  sublattices (the retired wick's parity failure generalized) and would let sub-reach
  texture ride through a "blur" untouched; coupling every texel to its true
  neighbours makes every non-zero frequency strictly decay. Shares sum to 1/8, so
  the worst mode's eigenvalue is `1 − w`: at the aimed-for blend it is damped by
  half, at full saturation it would be annihilated rather than flipped, and no
  firing can overshoot at any rate.

That is the whole set. `drag`, `ridge`, `load_pressure` and
`deposit_tilt` were listed as inert placeholders and were **removed** rather than
carried. Each remains a local change to reintroduce when built (the loop already
carries per-dispatch state): a forward deposit offset for the bow-wave drag, edge
displacement for ridge. (`bleed` was on that list, and its reintroduction as the
extent-local blur above is the pattern working as intended; `load_pressure`
and `deposit_tilt` came back as the general mapping below, which is the same
pattern again — the two specific knobs turned out to be one general one.)
Likewise `BrushParams` no longer carries
`spacing`, `flow`, `height` or `wetness`: with swept rendering there are no dabs
for `spacing` to space, and `flow`/`height` were redundant multipliers on `add` —
`flow` doubly so, since it also carried the `drain` factor into `τ` and applied
the run-dry falloff *twice*. `wetness` was the only source of the **wet channel**,
which is why that channel is gone too: a per-texel `wet` nothing could write is a
stored zero every pass paid for. Gloss is now a **uniform property of the paint**
(§6.3). The persistent aux is one channel, `(height)`.

*Determinism* — a stroke is a pure function of `base` + the `StrokeRecord`, so
replay and `preview == committed` hold and the log stays compact: only path +
params are stored, never per-segment data. Note that the bleed's cadence is keyed
on **absolute arc length**, not accumulated across the loop, precisely so it
carries no state across a range boundary: a live tail re-rendered from a span
fires a stretch of stroke identically to the commit that replaces it.
*Perf* — one extent-sized dispatch per segment plus the exchange (which
carries the snapshot) and the bake, inside one pass; a live stroke re-renders only
its tail, resuming the reservoir from the frozen head. What remains is per-segment
dispatch overhead: a few hundred segments each costing three small serialized
dispatches dominates a move, and the reservoir-side passes are sized by
`BRUSH_RES` rather than by the tip, so a *small* tip pays them most often. Mapping
the canvas into lateral × longitudinal stroke space, where the lateral rows
decouple and a workgroup can march many steps in shared memory without a barrier,
is the next structural win. *Paint never dries* — every texel stays as
workable as the moment it was laid, which is what lets there be no wetness state
at all; to glaze over "dry" paint the user adds a **new document layer**, which
composites as if dry.

### The opacity ceiling

Both effects carry an **opacity** (`BrushEffect::opacity`): the ceiling on what
a saturated stroke does. For paint it is §6.12's law run in the laying
direction — the erase pass subtracts `opacity·w` of coverage, the paint
integrate lays it:

```text
w  = 1 − exp(−K·Σm)          the coverage the whole parcel would lay
m′ = −ln(1 − opacity·w)/K    the scaled coverage, inverted through the slab law
h′ = h·(m′/m)                less of the same parcel; per-unit opacity unchanged
```

applied once per tile in `integrate.wesl`, before the parcel stacks on the
base. `w` saturates at 1 however long the stroke works one spot, so what lands
saturates at the dial: scrubbing walks the soft edge toward the cap, a stroke
crossing itself never outruns it, and a saturated stroke at 0.5 over bare
canvas is *pixel-equal to the fill at 0.5* (`tests/opacity.rs`). Separate
strokes still compound as glazes — each stroke's scaled parcel stacks its mass
through the shared law, so two saturated halves are the three-quarter fill. At
1 the shader takes an exact branch and the path is bit-for-bit what it always
was.

**The brush color lost its alpha to this dial.** Minted paint is per-unit
opaque (`m = h`, faded only by the drain): a per-unit alpha thinned the
*material* — a translucency scrubbing would build straight through, at full
relief however faint — which answered a question no digital artist was asking,
and its one honest use (how much shows) is exactly what the ceiling states.
`color` is three sRGB channels of pigment; everything about amount is the
effect's.

**Below 1 the swept path stops being stateless.** A scaled coverage is neither
of the two piece-composable forms above, so the path takes the erase pass's
shape (§6.12): the parcel — color, wide aux, residual — keeps accumulating
across the pieces of a live stroke, and every piece's integrate re-derives its
tiles from **pristine** paint under the whole of it (`ParcelCarry`, beside the
reservoir in `ToolState`; `render_swept_scaled`). Same copy-on-resume contract,
same no-seam result — literally the same, since the two effects that obey this
law run one procedure (`accum::IncrementalTileAccumulator`) and differ only in
what they rasterize, how many lanes their parcel has, what their landing pass
computes, and what bare canvas means. What it costs is a persistent
scratch pair per touched tile for the stroke's lifetime and the scratch ring's
overlap — which is why the branch is on the brush, and the full-opacity path
never pays it.

**The stamp loop mints the prefix differences of the same law**, because the
fast path must be an optimization and never a semantics: nudge `deposit` off
zero and the same brush must lay the same paint, opacity included. The loop has
no finished parcel to scale — fresh paint lands per segment and is smeared into
the picture as it goes — so instead the region aux's spare `.yz` lanes carry
the running **raw totals** of the *attempted* mint — the exposure's worth of
source, which is what stays independent of how the path is cut where the net
share does not — and each segment mints `M(F+ΔF) − M(F)` of the capped law
above, scaled by its own in-flight survival against the lift
(`dynamics.wesl::lay_parcel`). The differences telescope, so the per-texel
total is exactly the swept integrate's scaled parcel — mass *and* height,
drain variation included — and the two renderers agree texel for texel
wherever both can run the brush (`tests/opacity.rs`,
`a_whisper_of_deposit_does_not_change_the_paint`). The totals ride the
stroke's carry per touched tile (`LoopCarry::fresh`, cut from and seeded into
the region by the write-back's own copies), so a live stroke's pieces spend
one budget; at opacity 1 the law is the identity to the bit, the lanes are
never read, and nothing is carried or copied.

What the ceiling deliberately does not touch is **moved paint**: the lift, the
tool's deposit of carried paint, the bleed and the settle conserve height
(§6.1) — a cap on them would destroy paint — and fresh paint smeared away does
not refund the local budget. The `charge` glob needs no budget lane: a finite
source scaled at the mint *is* its own ceiling, everything it can ever deliver
being the scaled glob (`run.rs`'s reservoir init).

**The selection mask is the ceiling's other factor** (§6.8). A stroke's
ceiling is the dial times the mask's overall opacity — folded in once, in
`stroke_constants`, so it reaches the integrate, the erase, the loop's mint and
the charge through the one number both renderers read — times the mask's
coverage at the texel, read off the mask each pass binds. Not a lerp of the
result: a half-selected texel shows half of what the saturated stroke would,
the same reading a fill makes of the same mask. Which is why the carried shape
above is taken under *any* mask and not only below full opacity: every rim is
at least a pixel soft, and a texel under it scales the parcel exactly as a dial
below 1 does. Moved paint takes the mask as a fraction instead, the one gate
that conserves it.

### The ceiling under the pen

`opacity` is a modulation target on all three effects that have a ceiling
(`PaintModulations`, `WetModulations`, `EraseModulations` — the pen mapping,
below), and it is the one target that cannot ride the machinery the others ride.
A rate is per segment by nature: `add` scales what *this* segment lays. A
ceiling is per stroke — it caps what the whole accumulated parcel shows — and
under the pen there is no longer one number to cap by, only the factor each
segment asked for.

**What a texel shows is the largest ceiling any of its paint asked for**,
averaged over the texel: a light pass and a heavy pass over one spot, in either
order, leave the heavy pass's mark, and a heavy segment's soft shoulder over a
light mark lifts it only by the coverage the shoulder brings. With `c(θ)` the
coverage laid by segments whose ceiling is at least `θ`, that average is

```text
V = ∫₀¹ c(θ) dθ,    c(θ) = 1 − exp(−K · M(θ)),    M(θ) = the mass laid at or above θ
```

and this is a *max* that fixed-function blending can reach: a blend cannot take
the maximum of a running quantity, but `M(θ)` is a plain sum of per-segment
masses gated on the segment's factor, and a sum is what a blend accumulates. The
sweep that draws a pen-driven stroke has a fourth target, the **ceiling lane**
(`stamp_common.wesl`'s `FsOut::ceiling`, built only in the `ceiling` variant of
the shader; `paint_common::level_sums`): the segment's mass above each of two
levels — thirds of the dial — and its moment above each, additive like the aux.
The total mass and total moment ride the aux's own lanes. Between two levels the
mass sits at its mean factor (`claimed_coverage`), so the law is **exact
wherever all the mass in a third of the dial is at one factor** — a pen that
holds still, a mouse, which reports 1 — and **the max across thirds**: a
crossing whose two ceilings fall in different thirds shows the higher one
exactly. Two ceilings in the same third average by mass, never more than a third
of the dial from either; two more levels would halve that, and the lane has the
channels for them only in a colorimetric space, which is why there are two.

**The factor itself is interpolated across a segment**, and that is not a
refinement — it is what makes the target usable. Read once per segment a ceiling
is piecewise constant, so a stroke drawn at the rate a hand actually reports came
out in bands, stepping at every cut. `Paint` carries the factor at the segment's
two **ends** — its mean, and the difference between them in `opacity_ramp` — and
every fragment reads it at its own travel (`paint_common::linear_across`), which
is `Sweep::radius_ramp`'s construction for the radius in the one form that
differs: absolute rather than relative, a ceiling being a fraction rather than a
scale, so the interpolant stays inside `[0, 1]` with nothing to defend. Adjacent
segments read the pen at the knot they share from the same sample, so the ceiling
is continuous *by construction* rather than finely enough stepped; past either
end the overhanging cap takes the end it reached, which is the clamp
`radius_ramp_scale` already makes. The stamp loop reads the same law off the same
frame (`Stamp::opacity_ramp`), so a texel's factor is one number on either path,
and a brush the pen does not drive this way carries a zero ramp and takes an
exact branch.

The ceiling was the first target carried that way and is no longer the only one:
every pen target the shaders read per texel is, and for the same reason. The rule
and the line it draws are under *Pen mapping* below.

Every sum is symmetric in the segments, so the picture is the same whichever
pass came first, and it composes under re-cutting for the reason the aux does.
It is not strictly monotone, and that is the law's one real cost. Paint at two
factors inside one band averages, so mass arriving at the lower of them lowers
what the band is placed at — the claim *falls*, by up to the spread within the
band. Across bands it only ever rises.

**Where the claim dips the two paths part, in a bounded and directed way.** The
swept integrate reads the finished sums once, so it reports the fallen value; the
stamp loop mints prefix differences and a mint cannot run backwards — fresh paint
already smeared into the picture is not the loop's to take back, and a negative
parcel is not expressible in a colour that is a mass-weighted mean of the tool's
and the brush's own (it blew out to white when it was tried; `tests/opacity.rs`,
`a_smear_under_a_pen_driven_ceiling_stays_in_gamut`). So the loop holds the
running maximum, which is the nearer of the two to the max the law approximates.
On the crossing the suite draws the gap is inside the f16 stores
(`the_loop_lays_the_pen_driven_ceiling_the_fast_path_lays`, which is why that test
still asserts an equality); what bounds it in general is the law's own within-band
error.

That the two can part at all is a property of *any* estimator built from additive
sums: exactness for a pen that holds still forces a mean, and a mean falls when
lighter mass joins it where the max it stands in for does not. Making them agree
identically therefore means giving up either the exactness or the additivity, and
both are load-bearing — the first for the mouse, which reports a constant 1, the
second for the blend that accumulates the lane at all.

**A band's edge is a window, not a line** (`paint_common::LEVEL_WINDOW`), and that
is not a refinement either. With a hard edge a band's coverage is built only by
the fragments whose factor landed above it, so it carries *their swept footprints*
rather than the mark — and since a pen held at one pressure wobbles by a
hundredth, a stroke sitting on a boundary printed those footprints one at a time,
as scallops across its crossing (2026-09-02). The claim there is
`mu1·(c0 − c1) + mu2·c1`, so wherever the two bands' coverages differ the band gap
is what the ripple is worth — about seven levels of grey on the stroke that
reported it. Over a window every fragment contributes to both bands in proportion,
both coverages saturate together across the whole mark, and the term that carried
the footprints goes to zero. A pen that holds still is exactly the dial either
way.

Two other laws were built first and taken out, and both are worth naming. The
plain **mass-weighted mean** — one additive lane of `Σ oᵢ·mᵢ` — is exact and
composable and not a max at all: a light crossing eats a notch out of the heavy
line it crosses, since any function of two additive sums that reduces to
`o·w(B)` at constant `o` falls as zero-ceiling mass is added. **First claim
wins** — `Σ oᵢ·wᵢ·Π_{j<i}(1 − wⱼ)`, the one monotone law a single blend can
express — never fades, but at a flow that saturates coverage in one pass the
first pass over a spot claims all of it and a heavier pass back over it adds
nothing, which is the other notch. The level law is what both were reaching
for.

The integrate reads the lane in place of the dial: what shows is
`dial × mask × V`, inverted through the slab law with the parcel's own mass as
the cap (`t > exp(−K·m)` is asked before the log is). One corner follows from
the cap: at a ceiling of exactly 1 a saturated claim rounds to f16's 1, the
transmittance it leaves is 0, and the whole parcel lands — mass laid later at a
lighter touch included, as impasto rather than as coverage — which is what the
unmodulated brush at dial 1 does too, having no ceiling at all. Below 1 the
inversion is exact. The lane resolves under supersampling as the mean of the
subsamples' own claims, a claim being a visible coverage already. The erase
sweep carries the same lane by the same rule, with the moment in a lane of its
own since its accumulator has one channel, and `erase.wesl` reads the claim in
place of the extent's coverage (§6.12).

**The stamp loop reads the same lane**, drawn over its region rather than
storage-written: after each painting segment's deposit the host ends the compute
pass, draws that one segment through `stamp.wesl::fs_levels` into a region-sized
lane with the sweep's own pipeline, and resumes — so a deposit reads the lane as
the segments before it left it, adds its own share, and mints the prefix
difference of the *claimed* law `−ln(1 − o·V)/K` (`dynamics.wesl::claimed_m`),
capped at the raw total so a saturated claim lands the whole attempt. The moment
of the raw mass rides the region aux's `.w` beside the raw totals in `.yz`. The
differences telescope as the dial's own do, a difference below zero — the
within-third average falling — *un-mints*, taking the brush's own paint back as
far as the texel still holds it, and the lane rides the per-tile carry beside
the mint budget (`LoopCarry::levels`, cut and seeded by
the same copies). The two renderers agree texel for texel under a pressure ramp
and across a crossing (`tests/opacity.rs`,
`the_loop_lays_the_pen_driven_ceiling_the_fast_path_lays`). The `charge` glob is
scaled by the dial alone: it is minted once, before the pen has moved, and
`charge` is an initial condition the pen cannot reach (below). Moved paint is
under no ceiling, as before.

A stroke whose ceiling the pen drives takes the carried-parcel path whatever its
dial says (`render_swept`'s branch reads `StrokeConstants::ceiling_lane`), since
a claim is something the pieces of a live stroke build up together. The lane is
one more `Rgba16Float` per touched tile for the stroke's lifetime, and on the
loop a region-sized one per piece plus a render pass per segment; a brush that
leaves the target unmapped never allocates any of it, binds a 1×1 zero in the
slot, and renders bit for bit what it did. Both paths ride the corpus
(`opacity_pen`, `wet_opacity_pen`).

### Pen mapping — what drives which parameter

A brush is not a set of fixed numbers plus one hard-wired rule. Pressure scaling
the radius is what an ordinary brush wants; a palette knife wants pressure on
`lift` and tilt on `deposit`; a pencil wants pressure on `flow` and tilt on
`size`; a marker wants nothing on anything. `BrushParams.modulation` is the
mapping, one optional entry per target:

```rust
struct Modulation { source: ModSource, floor: f32, curve: f32 }   // ModSource = Pressure | Tilt
struct Modulations { size, flow, opacity, add, lift, deposit, bleed: Option<Modulation> }
```

On a wet brush the `flow` target scales the whole loop — mapped to pressure, a
light touch lays less *and* smears less — where the `add` target reaches the
source share alone, for a brush that lays more under the pen without working
the canvas harder. Two mapped targets can multiply into one rate (`add`'s, say);
the flattener charges the steeper of the two rather than their sum, a deliberate
under-charge in a corner where the attribute budget is already generous.

**A modulation is a multiplier in [0, 1], never a remap.** The value the renderer
sees is `param · factor(input)`, `factor(0) = floor`, `factor(1) = 1`. That bound
is the design, not a simplification of it. Every guarantee the engine derives
from a brush is stated against the brush's own numbers — the frozen-span radius
bound (`safe_frozen`), the region fit, the swept-vs-loop choice
(`dynamics_setup`), the flattener's exchange step (`exchange_travel`) — and each
one stays sound with no part of it learning that modulation exists, because a
parameter can only ever be *smaller* than its slider says. A remap that could
also scale up would put a correction into all of them, and a missed one is a
stroke that renders differently live and committed (§1.3). It costs nothing in
expressiveness: a pencil that widens as the pen leans over is the widest radius
on the slider, mapped to tilt, with the floor at the narrow end. **The slider is
the maximum and the pen takes it away.**

Two consequences fall straight out and are worth stating, because they are what
would otherwise need checking at each call site. An axis the brush leaves at zero
is zero at every point of every stroke it could draw, so `dynamics_setup` and
`bleed_fires` can gate on the brush's own rate and be exactly right. And a
modulated rate is never above the brush's, so `exchange_travel` prices the worst
case and every segment comes in under it — the wet flow included: the plan
scales every λ by the segment's modulated flow, which is never above the
brush's own, so the budget charges `flow · rate` off the brush and stays a
bound.

**The response curve is rational**, `x / (m(1 − x) + 1)` with `m = 1/k − 2` and
`k = (curve + 1)/2` — Schlick's bias, monotone from (0,0) to (1,1). Not `xᵞ`:
this decides stored pixels, so replay, goldens and peers must agree on it to the
last bit (§12.1), and IEEE-754 pins `+ − × ÷` where `powf` is specified nowhere.
It is the same argument that makes `taper_profile` a polynomial. `curve = 0`
lands on `m = 0` through steps that are all exact in binary, and the linear case
returns `x` itself — so the default brush's radius is the exact product
`radius · pressure · taper` it always was, and every golden holds.

**A steep response is paid for in segments.** A segment sweeps at one value of
everything, so the flattener's attribute budget — 2% of a pen unit — has to be 2%
of the *parameter*, and a curve stretches the one into the other. So
`flatten_tolerance` divides the budget by `BrushParams::max_slope`, the largest
`|d factor / d input|` across the mapped targets, which is why the bias is
clamped (`MIN_BIAS = 0.1`, so the slope tops out at 9) rather than left open. The
unmodulated brush and every plain linear mapping come out at exactly 1 and
flatten on the budget they always had.

The one thing the pen cannot reach is the **substrate**: `tooth_give` is on the brush
and mappable like the rest, but the grain it bites into is the *canvas* (§6.4), so a
pencil and a loaded brush drawn the same way across one paper break up on the
same faces. Mapping the give to pressure is the charcoal behaviour — bear down and
the tip gains the give to press past the falls it was bridging, so the grain fills
in. **That mapping is why the knob is the give rather than the bite**: a modulation
only scales down, so the other parameterization would have made a light touch solid
and a hard press dry (§6.4). Its neighbour `tooth_softness` is deliberately *not* a
target: what the tip is made of does not change under the hand. Which faces they are is the stroke's own business, though: contact reads the
substrate's rise *along the travel*, so the same brush over the same paper the other
way catches the other sides (§6.4).

**Resolution happens in one place**: `generate_segments_in`, alongside the taper,
where the pen attributes are already interpolated per segment. Both render paths
flatten through it, so a live tail and the commit that replaces it cannot read
the pen differently. Downstream the resolved values ride the `Segment` — the
stamp loop already carried its λs per dispatch and needed no change at all, and
the swept path moved `add` off the per-tile uniform onto the segment instance,
leaving `drain` behind because `drain` is a function of arc length that every
fragment recovers for itself. The tooth is a per-*fragment* gate on `τ` in the
same slot `drain` occupies, for the same composition reason (§6.4).

**A target the shaders read per texel is carried as a pair and interpolated
across the segment; one the *exchange* solves with is not.** That line is the
whole of what a segment may hold constant, and it is drawn where it is for a
reason on each side.

On the interpolated side — the radius (`Sweep::radius_ramp`), the ceiling, the
`add` source rate, the tooth's give, the liquify follow — a value held constant
across a segment steps at every cut, and a stroke at the rate a hand actually
reports is a handful of segments wide. That is visible: it drew a comb of
sawteeth down a tapered tip, and bands across a mark whose opacity the pen was
driving. So each is carried as a mean and a difference between the segment's two
**ends**, and read at the fragment's own travel
(`paint_common::linear_across`). Adjacent segments read the pen at the knot they
share from the same sample, so the value is continuous *by construction* rather
than finely enough stepped — `every_per_texel_target_is_continuous_across_a_knot`
is that claim, checked on the segments themselves rather than looked for in
pixels. It costs the deposit nothing to be exact about: the deposit is
`∫ add dτ`, `add` is one function of arc length whichever side of a knot reads
it, and a definite integral cut in two is the sum of its pieces — so the result
is as independent of the flattening as the constant it replaces, and nearer the
integral than the midpoint sample was.

On the other side are `lift`, `deposit` and `bleed`. Those are not read per
texel; they are the rates of a **transfer**, and the exchange solves it once per
dispatch as two complementary halves — a canvas half indexed by canvas position
and a tool half indexed by *reservoir* texel, which has no canvas position to
vary along. A rate that varied across the segment would leave the two halves
solving different problems, and the transfer would stop balancing; the shader
makes the same argument about sub-stepping, and about why the tool books its
share of the source at its own arc rather than the segment's. What bounds their
step instead is the flattener, which already buys segments against
`BrushParams::max_slope` — so a steeply mapped flux is paid for in segments,
which is the mechanism that was there all along.

`hardness` and `charge` are deliberately not targets: hardness is baked into a
prefix-τ texture per value, and `charge` is an initial condition rather than a
rate, so neither has a per-segment form to modulate. Adding the field bumped the wire version to 3 (§8) — back when it had to:
the encoding wrote no field names, so an appended field was still a break and
`#[serde(default)]` had nothing to fill from. The same addition today is exactly that
attribute and no version at all.

In the UI (§11) the mapping lives **on the parameter's own row** — a chip naming
what drives it, opening source / floor / curve in place, one row at a time — so a
brush with no mapping looks exactly as it did, and reading a brush does not mean
holding a separate matrix against the sliders.

### Color dynamics (color jitter)

The applied color can vary **across the brush and along the stroke**:
`BrushParams.color_dynamics` (historized — it changes stored pixels) holds a
noise kind plus two per-axis **frequency** and three per-channel **amplitude**
factors. A 3-channel, exactly **tileable 2-D noise tile** is baked **on the CPU,
per stroke from the stroke's `seed`** (`noise.rs`, `Rgba8Snorm` 64², or 256² for
`Mosaic`; only correctly-rounded ops, no transcendentals ⇒ bit-reproducible
across platforms) and sampled with a repeat sampler. Per stroke because a tile
is *one* field — the same `P²` clouds or facets — and strokes sharing one would
lay the same polygons and drifts, shifted, however each translated its lookup;
the bake is paid at pen-down, and the cellular kinds tabulate their sites once
per bake to keep it there. The kinds:

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

The lookup domain is **stroke-local**: `(lateral·f₀, arc·f₁)/NOISE_TILE_PX`,
where `lateral` is the signed offset from the centreline and `arc` the length
along it, both in canvas px (brush-local y is in radius units, so it is scaled
by the radius — the pattern keeps one scale whatever size the tip is). One axis
varies color across the
extent, the other evolves it along the stroke. Anchoring to the stroke rather
than the canvas makes the variation belong to the *gesture*, and costs nothing in
determinism: both coordinates are still functions of the fragment's canvas
position and the segment, so the deposit stays a pure function of the two and
tile aprons stay bit-consistent (§6.4). Clamping arc to each segment's body makes
it *continuous across overlapping segment quads*.

The sampled signed offsets perturb the brush's **channel triple in the current
color space** (Oklab `L,a,b`; Mixbox concentrations), applied per fragment in
the sweep stamp (`brush_color`, `stamp_common.wesl`) and per texel to the `add`
paint in the exchange loop's `deposit` (`dynamics.wesl`) — both paths share the
field and parameters, so a brush looks the same whichever path renders it.
Amplitude 0 (the default) binds a 1×1 zero tile and early-outs — bit-identical to
the constant-color deposit.

### The deposit jitter (quantity dither)

Color dynamics' sibling for the *amount*: every texel of a stroke scales the
exposure it presents by a multiplicative gate, uniform in `(1 − ε, 1 + ε)` and
hashed from the **absolute canvas texel** and a per-stroke seed
(`lib/noise.wesl::jitter`). `ε` is `BrushParams::jitter`, beside the color
dynamics it is the amount's half of rather than among the flux axes, because it
gates what *every* paint-laying path lays and not only the exchange loop's —
historized, since it changes stored pixels, defaulting (in the brush and for a
document that predates the field alike) to `DEFAULT_JITTER` (1%, sized just
above the f16 tile quantum);
the field's addition bumped the ALPN (§8). Its job is the display dither's
(§6.5) one layer down: the lift/deposit loop accumulates iteratively, and
whatever its residues and the f16 stores would pile into spatially coherent
bands lands as per-texel dither instead, because neighbouring texels accumulate
at decorrelated phases.

Where it applies is exactly where the deposition tooth does, and it holds the
same three laws by the same arguments. It scales the **exposure** — beside
`substrate_tooth` in the swept stamp, the exchange loop's two deposit kernels
and the settle — so the exchange's four shares stay exact complements of one
(jittered) exposure and nothing is minted per texel. It is constant along the
stroke at a given texel, so it factors out of every per-segment sum and the
deposit stays *exactly* independent of how the path was cut (§6.2) — which
per-segment keying would have destroyed. And the tool's side books the gate's
mean, as it books `tooth_bearing` — the jitter's mean is exactly 1, so the
tool's dispatch is untouched and conservation holds in expectation over the
extent. The bleed pair is never gated, for the tooth's own antisymmetry reason.
White per-texel noise rather than the baked field's smooth structure, because
the job is the opposite of a pattern: decorrelating neighbours. Per-stroke
seeding (its own splitmix64 draw off the record's `seed`, beside the draw the
color-dynamics tile is baked from) keeps repeated glazes averaging out rather
than compounding one texture; `ε = 0` is exactly the gate 1.0, bit-identical to
the unjittered deposit.


## 6.6 Brush shapes & the asset store

The default brush is a procedural soft disc, but natural media needs *organic*
tips. A brush shape is a **coverage mask**: a greyscale image where white = full
deposit and black = none. The mask drives coverage and, scaled, the height
channel too — so a worn-bristle tip lays down *broken* impasto rather than a
uniform slab.

**The round tip is specified by the stroke it draws, not by its own silhouette.**
What `hardness` names is the profile *across the stroke* — a full pass lays
`1 − |y|^h` at `y` radii off the centreline, for `h = 1/(1 − hardness)` — and the
extent is then whatever produces it. The two are not the same shape, because
the deposit composes in optical depth: what the sweep integrates along the travel
axis is `κ = −ln(1 − coverage)`, not coverage, so a mask carrying the profile's
own falloff draws a very different one. Asking instead for the field whose row
integrals are `τ(y) = −h·ln|y|` is an Abel transform, and it inverts in closed
form to the radial `κ(r) = (h/π)·acos(r)/r` — so the tip is `1 − exp(−κ(r))` and
the profile is exact rather than approached. (It was not, before: normalizing a
`1 − r^h` disc by its chord half-length aimed at the same profile through the
linear integral, and drew a stroke up to 0.54 in coverage fuller than its
hardness named, with the falloff crushed into the outermost texels.)

The hardness a stroke actually bakes is **floored by its own size**: the tip's
shoulder — `3·(1−hardness)·radius` px, the one definition
(`budget::shoulder_per_radius`) — never falls under one canvas px
(`budget::effective_hardness`, applied where the tip is resolved,
`TipCache::resolve`). A falloff narrower than a px is content the tile grid
cannot carry — it can only alias — so a hard edge keeps the ~px antialiased rim
every selection mask's feather already has ("floored at one, which *is* the
antialiased hard edge", §6.8), and a tip too small to hold even that comes out
as soft as its own footprint. This is the pixel footprint's other half: the box
filter (§6.2) owns the small tips, where its window spans mask texels, and the
floor owns the large hard ones, where the window is sub-texel but the profile
itself must stay ≥ a px wide. The budgets built on the *nominal* hardness (the
taper's subdivision, the coarse-deposit cell) are left there deliberately — the
nominal shoulder is the narrower, so they only over-provide. A `Stamp` carries
no hardness to floor and takes the box filter alone.

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

`orientation` (`FollowStroke` | `Pen`) sets how the swept extent is angled:
`FollowStroke` keeps the shape's native axis on the stroke tangent (what makes a
bristle brush read as a real stroke rather than a rubber stamp), while `Pen` pins
it to the pen's tilt azimuth in canvas space, like a calligraphy nib.

**A brush's `size` names the disc the mark fits in — for every shape.** An
imported mask is **reach-normalized** at decode (`stark_assetid::coverage`, part
of the identity contract's canonical form, §19): its content is scaled about the
mask's own centre so the circle that circumscribes it is the disc inscribed in
the mask square — grown to just under the rim when it huddled short of it,
shrunk inside when it spilled past. A dead band under the rim keeps the
operation idempotent, so re-decoding a stored mask lands on the same texels and
the same id; the centre is the author's own, so a deliberate offset survives as
composition rather than being re-centred away. The round tip has the property by
construction — exactly zero outside its unit disc — so one sentence covers every
brush: **nothing a tip can paint lies further than `radius` from its centre, at
any orientation**. The box that decides which tiles a segment is drawn into, the
rect the dynamics loop dispatches, and the region budget's tip floor
(`budget::fit_len`) all use exactly that. A mask could once be opaque to the
corners of its own square, and every one of those bounds carried a `√2` for a
worst case almost no mask was — with the region floor calibrated for the round
tip, that `√2` was also what silently sent every large gentle stamp back to the
swept deposit.

**The two orientation sources are two bakes, not one bake read two ways.** The
swept integral runs along the travel direction, so orienting the extent means
turning the mask *inside the frame that integral is taken in* — and the two
sources ask opposite amounts of that:

- `FollowStroke` never turns it at all. The relative angle is 0 by definition, so
  the volume is **one layer, the mask as it stands**, integrated over its own
  width. That is what nearly every stroke reads, and it costs a single pass.
- `Pen` turns it through the whole circle, which is safe **inside the mask's own
  square**: canonical content lies within the inscribed disc, and a rotation
  maps the disc to itself. So the pen volume is the same grid **stacked**, one
  layer per relative angle, built on first use rather than at import (the store
  cannot see a brush setting, and eagerly baking it would charge every
  follow-stroke brush for a mode it never enters). While a mask could fill its
  corners this stack had to be padded by `√2` — a square does not fit inside
  itself turned — and every pen-oriented brush paid double the texels, or at the
  fixed memory budget half the orientation resolution, for corners the canonical
  form now guarantees empty.

With every volume on the mask's own grid there is one frame and one column
width, so both stopped being quantities at all: the frame *is* the tip
(`Sweep::radius`, and the shader's `Stamp::radius` lane), and
`build_prefix_tau` integrates its rows at `2/width`. Both were mode-dependent
while the pen stack was padded — the frame `√2` larger so the mask inside
landed at the radius the brush asked for, the `dx` a number the padded texture
could not supply — and each had to reach exactly one of the two sides of the
dynamics loop, carrying a `frame_scale` conversion whose correctness no pixel
visibly depended on. A representation that cannot express the distinction is
what retired it.

**A tip is drawn out along the axis it faces.** Tilting a real pencil does not press a
bigger circle onto the paper — it rolls the cone over, and the patch in contact
*elongates along the lean*, growing along one axis and not the other. `stretch` is that
axis: `BrushParams::stretch` in `[0, 1)` names the elongation `s = 1/(1 − stretch)`, the
extent is scaled by `s` along the brush's facing direction and left alone across it,
and `BrushModulations::stretch` lets the pen drive it. Pointed at `Tilt` with
`OrientationSource::Pen` that *is* the pencil — and the same knob held at a value with
no mapping is a chisel nib, off a plain round tip with no shape asset at all.

The axis is `orientation`'s, not a second direction to set, and that is what makes the
whole of it free. Stretching by `s` along a canvas axis `û` is the linear map
`A = R_û·diag(s, 1)·R_ûᵀ` on the extent, so the deposit is that map's image dragged
along the travel `t̂`. Substituting `q = A⁻¹p` turns the integral into one of the
**unstretched** extent:

```text
τ(p) = (1/m) · ∫ mask, along v̂ = normalize(A⁻¹t̂), over a travel of m·L
       where m = |A⁻¹t̂|
```

— a different direction, a different travel, and a constant. And the prefix-τ volume is
*indexed by* the direction it is integrated along, so a different direction is a
different **slice** rather than a different bake. Because `û` is the facing axis, the
slice always lands somewhere the brush already has:

- `FollowStroke` faces along the tangent, so `û = t̂` and hence `v̂ = t̂`: the relative
  angle stays 0 and the single unpadded identity layer still serves.
- A round tip is rotation-invariant — one slice answers every angle.
- `Pen` on a stamp already reads the padded stack of *every* angle, so a shifted index
  costs nothing.

An axis independent of the facing one breaks all three at once (a follow-stroke stamp
would need the padded rotatable bake it never builds), which is why there is no second
direction to set — the one-number parametrization is the reason the feature is free,
not a simplification of it.

`Stretch::solve` does the trigonometry once per segment, and what reaches the shaders is
the solved map. It is **upper triangular** by construction — `M = R(v̂ → x̂)·A⁻¹` sends
`x̂` to `(m, 0)` — so three floats state it: the travel scale `m`, a **shear** (an
obliquely stretched tip's leading edge leans rather than staying square to its travel),
and a lateral scale, whose product with `m` is `det A⁻¹ = 1/s` at every angle. The
identity `(1, 0, 1)` is exact in floats, so every brush that never heard of stretch
renders bit for bit what it did before.

It is applied **last and only into the lookup** (`stretch_look` in `stamp_common.wesl`,
shared by both paths). Everything else a fragment reads out of its frame — the arc it
sits at, the color-noise domain, the tip in force, the substrate it is gated by, the canvas
position a reservoir texel is dragged to — is a property of where the *tip* went, and
the stretch is a property of the extent the tip carries. Keeping the frame the tip's
own the whole way down is what leaves `stroke_arc` measuring canvas px rather than the
lookup frame's units, and the radius ramp applying to the tip that is actually there.

Three things do have to grow with it, and each is the same bug in a different place — a
extent drawn outside the geometry drawn for it, cut off along a straight line:

- **The sweep strip** and the dynamics loop's rim test, which work in the reference
  travel frame: `stretch_hull` reads the box `|y| ≤ 1/lateral`, `|x| ≤ (1 +
  |shear|/lateral)/m` straight off the map.
- **`Segment::reach`**, the canvas box the segment is rasterized and dispatched into.
  One factor and not two, because it grows an axis-aligned box in every direction at
  once, and `‖A‖ = s` bounds every angle at once.
- **The arc cap.** `MAX_TIP_TURN` exists to keep the swept sector a simple polygon and
  the reservoir's crescent seams away, both of which are about the extent rather than
  the number naming it — so a tip reaching `s` times as far may bend `s` times less.
  Charged against the *brush's* elongation, like every other bound here, since a
  modulation only ever scales the knob down.

On the tool side of the dynamics loop the gain runs the other way: a mask texel on a
drawn-out tip stands over more canvas, so the track it crosses is that many fewer of its
own widths and `exchange` books `travel_radii · m`. That is what keeps the books
balanced — a stretch multiplies the canvas's total exposure by `s` and the area each
reservoir texel answers for by the same `s`. The reservoir itself does not stretch, for
the reason it does not ramp: it is the tool's own grid, and `bake`'s rotation out of the
travel frame is already the rotation out of the *lookup* frame, since the slice index it
reads carries the difference.

`MAX_ELONGATION` (8) bounds the knob, and what it bounds is **area**: every tile the
drawn-out tip reaches is a tile the stroke is rasterized into and the dynamics loop
dispatches over, so `s` prices the stroke roughly linearly. Past eight the mark stops
reading as a wider stroke and starts reading as a smear the length of the tip.

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
samples the bound mask at the extent's uv, so the mask's coverage is what the
swept optical depth integrates and therefore modulates both opacity and the
height `add` lays. `Round` is realized as a built-in generated mask under a
reserved id, so the shader always samples a texture — one code path.

**Assets are fetched at runtime, never embedded.** The engine is *given* image
bytes; it embeds none. Built-in assets (brush shapes, substrate height maps, the HDR)
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
since a preset stores a content id. The large substrate maps are fetched lazily,
only when a substrate is selected. This keeps multi-megabyte assets out of the wasm
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
replayable and collaborative for free. That is why `stark-engine/src/assist.rs`
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
  sound because the truth stays a fixed point and the gap is at most a fifth of
  the circle (`CLOSE_GAP`), so what was drawn always outvotes it. That bar sits
  where the fill stops paying: a fifth-turn gap on a 400px loop still lands the
  centre and the major axis within 3%, a quarter-turn gap walks the centre 5% off
  — and a three-quarter arc is a shape somebody can mean, so it is refused.

**Everything is barred on the worst sample, not the RMS.** This is the whole
difference between a bar that discriminates and one that does not: a hand's wobble
along a straight drag is noise, while a curve somebody *meant* deviates
systematically, and averaging is exactly the operation that hides the second. A
300px stroke bowed 40px reads as 4% RMS — indistinguishable from a shaky straight
line — and as 9% at its worst, which is not close to anything. Every threshold is
denominated in the gesture's **input tolerance** (§6.2), because "close enough to a
line" fixed in canvas px would mean two different things at two zoom levels.

**The ellipse's bar is much the looser of the two** — 15% of the mean radius against
the line's 3.5% of the length (`ELLIPSE_RESIDUAL`, `LINE_RESIDUAL`). Two reasons that
compound: the same hand movement is spread over a radius rather than over a length, and
a loop is a *longer* gesture than a drag of the same size, since going round costs π
times the diameter where crossing costs one and the wrist reverses twice on the way. A
worst-sample bar on a signal that long is asking about the one moment the hand was
least steady. It costs less discrimination than the line's would, too: a shape that is
not a loop of *some* ellipse misses by tens of percent rather than by ones, where a
bowed stroke sits just the other side of the line's bar.

Declining is a normal outcome. Dwelling at the end of a stroke that is neither a
line nor an ellipse leaves it exactly as drawn and the gesture carries on through
the fitter — a closed trace is offered to the ellipse fit first and *falls through*
to the line fit if it misses, so a rough rectangle simply does not snap. The cost
of a false positive (a considered stroke silently replaced) is far higher than the
cost of a miss (hold it straighter and try again), and the bars are set from that
asymmetry rather than from a hit rate.

A recognized shape is then offered whatever **perspective guides** are on the
screen. A line takes the nearest axis it is aimed within a few degrees of — a
line drawn roughly toward a vanishing point comes back aimed exactly at it, from
where the hand started it (§20.6). A loop becomes the **circle on a plane** it
nearly lies on, which corrects the two things a hand cannot judge about a circle
in perspective — how open it is and which way it leans — while keeping the size
and place it was given (§20.7). Both questions are put strictly *after* the
stroke has been accepted as a line or an ellipse, and never instead of it: the
grid may only choose *which* line a line is, or which plane a loop is a circle
on. Neither pays for itself downstream — a guided line is still a segment and a
perspective circle is still an ellipse, so both are the same `Vec<ControlPoint>`
in the end.

**Adjustment** — what the rest of the drag means. A line moves the end being held,
*by the pointer's delta*: snapping moved that end off the hand by up to the fit
residual, and driving it absolutely would jump it back on the first move. A line
that took a guide axis keeps it and takes only the travel *along* it, so the end
runs out and back on the grid line while the hand wanders off it — an alignment
a single sideways nudge could break would not be one. A perspective circle is
*sized*, and only sized, in the plane it is a circle on: turning a circle does
nothing, so the degree of freedom the free arm spends on the turn is not there
to spend. An
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


## 6.11 Stroke smoothing — the towed tip

Every brush carries a **smoothing** amount. At zero the stroke is the hand,
verbatim — today's behaviour, bit for bit. Turned up, the mark is drawn by a
**tip towed behind the pointer on a string**: the inking pen turns a shaky drag
into one confident line, the lettering brush's sweeps come out swept — while the
sketching pencil beside them on the rack keeps every tremor, because tremor is
what a pencil is for. The amount is part of what a brush *is* — set in the brush
editor, saved with the preset — not a mode the app is in. That is the difference
between this and the app-wide stabilizers of the prior art (§18): nobody should
have to visit a settings page because they switched from inking to hatching.

### The model is a tow, not a signal filter

The tip trails the pointer on a string of length `L`. While the pointer wanders
within `L` of the tip, the string is slack and the tip is **parked**. The moment
it comes taut, the tip is dragged along, and a dragged tip traces the classical
pursuit curve — the **tractrix**, the path of a thing pulled by a string, which
is also what a pinstriper's sword brush or a long-handled rigger does behind a
moving hand. Smoothing is thus a physical model of a real painting tool rather
than a low-pass filter with a cutoff to tune, and three feel properties fall out
of the geometry that the filters it replaces have to approximate:

- **A dead zone.** Jitter, hesitation and the pixel staircase are smaller than
  the string, so they never move the tip at all. The hand can stop, breathe and
  pivot mid-stroke and the mark holds still — which also composes with §6.9's
  dwell: holding for a snap does not creep the trace being recognized, where an
  averaging filter keeps easing the tip in for as long as the hand hovers.
- **Bounded lag.** However fast the hand, the tip is never more than `L` behind.
  A moving average's window and a time-constant filter both let the gap grow
  with speed; the string cannot.
- **Corner rounding at the string's own radius.** A corner sharper than `L`
  comes out as a tractrix arc of about that size. That is the honest price of
  smoothing — at the scale of the string, a corner the hand meant cannot be told
  from a wobble it didn't — and it is paid visibly, at a radius the artist chose,
  per brush.

**The tow is integrated exactly, per pointer-polyline segment, not stepped per
report.** For a target running along a straight segment the taut tow has a closed
form — the angle `θ` between string and travel obeys `tan(θ/2) =
tan(θ₀/2)·exp(−s/L)` — and the slack→taut crossing within a segment is a
quadratic. Two things are bought by taking it rather than iterating the obvious
`tip += (pointer − tip)·k` update. The towed path becomes a function of the
pointer's *path* rather than of its report clock — a 240 Hz pen and a 60 Hz
mouse towing the same trace produce the same tip — which is §6.2's
partition-independence discipline applied to input: cutting a target segment in
two composes exactly, because the exponential is exponential in arc. And the
update never overshoots or oscillates at any cadence, because it is not an
integrator with a step size, it is the trajectory itself.

Transcendentals are fine here, and stating why draws a useful boundary: the tow
runs **once, on the originating client, upstream of the record**. What lands in
`StrokeRecord::path` is the towed, fitted control points; peers, replay and
goldens re-run none of it. It is the same class of computation as the fitter's
own least-squares solve — pre-record, single-machine — so §12.1's bit-agreement
rules (the reason `taper_profile` is a polynomial and the modulation curve is
rational) do not reach it.

### What one pointer report costs

The trajectory is exact; the *samples taken along it* are a separate choice, and
the one place the tow can be expensive. They land on a grid of `L/4` — a quarter
of the string — which is fine while the tip is bending and pure waste once it is
not. A grid stated as a fraction of the string is inverse in the string, so a
smoothing amount near zero asks for an unbounded number of samples per report,
each one a full least-squares update in the fitter downstream: measured, one 4 px
report at 8× zoom cost 320 fitter pushes at amount `0.05` and 2000 at amount
`0.02` — 15 ms and 445 ms of fitting for a single pointer move, natively. Below
that the accumulator `s += L/4` stopped advancing in `f32` at all and the loop did
not terminate. A near-zero smoothing setting therefore froze the tab, and the
frozen time was billed to `input.fit` (§7.1), which measures a pointer report
rather than a fit.

**So the grid is laid across the bend and no further.** The tip's offset from its
straight asymptote is `2L·t/√(1+t²) ≤ 2L·t`, and the half-angle decays as
`t = t₀·exp(−Δs/L)` with `t₀ ≤ 1`, so the bend is under the input's own tolerance —
and therefore invisible to the fit that tolerance prices — after `L·ln(2L/tolerance)`.
Past that the tip is trailing dead astern on a straight line that two samples
describe exactly, so the run's final emission is the only one left worth making.
The tow is handed the same `tolerance` the fitter is built with, held to the same
bounds by the same function, because this is one more thing only the caller knows.

Two properties follow, and both are why the bound is read off the tractrix rather
than picked as a cap:

- **A report costs `4·ln(2L/tolerance) + 1` emissions at worst** — under 30 across
  the whole reachable range of the knob, at *any* zoom, since the string and the
  tolerance are both carried through the view and only their ratio survives. Bounded
  in the string rather than inverse in it. Counted up front rather than
  accumulated, so the loop is an integer one and cannot fail to terminate however
  degenerate the arithmetic.
- **A string at or under half the tolerance reaches zero**, and the tow emits exactly
  one sample per report — the untowed rate. The bottom of the knob *degrades* to
  the untowed path rather than falling off a cliff into it, so there is no
  threshold to discover and no dead region on the slider. The tip is still towed;
  what goes quiet is the sampling.

Goldens cannot catch a regression here, and that is structural rather than an
oversight: every replay runs at `rope = 0` by construction (the point of the
paragraph above about nothing downstream learning that smoothing exists), so the
tow's own unit tests are the only thing standing under it. They pin the count,
not just the trajectory.

### Attributes ride the pen, not the arc

The towed sample carries the **current** report's pressure and tilt, not
attributes delayed to match the tip's arc-length lag. The artist steers by the
tip they are watching — pressure included: they press where they see the mark
forming, so delivering that pressure anywhere else breaks the loop they are
actually in. Delayed attributes read as *pressure lag*, and unlike position lag
there is nothing on screen that explains it. (`time` is a stamp on the report
either way, §6.2.) A parked tip emits nothing at all, attribute changes
included — which is not a loss the tow introduces but the fitter's own standing
answer to a stationary hand: a report that did not move carries no geometry,
and its attributes apply to a zero-length piece of path (§6.2).

### The pen-up parks the tip

At lift the tip is up to `L` short of where the hand stopped, and **that is
where the mark ends**. Lifting the pen stops pulling the string; it does not
reel the tip in.

The alternative was built first and is wrong in the hand. **Winching** — `L`
runs to zero and the tip travels straight up the string to the lift point — has
a tidy derivation behind it (it is the tow's own continuation for a stationary
target, so it exits along the direction the curve already held) and it satisfies
the letter of §6.2's "the mark ends where the hand did". It still reads as a
defect, for two reasons that outrank the derivation:

- **Nothing steered it.** The string is the one part of the gesture the artist
  is deliberately *not* aiming with — the whole point of the dead zone is that
  the hand's last rope of travel is noise. A winch promotes exactly that
  discarded rope to a drawn line, appearing at release, at a place nothing was
  drawn, in a direction nobody chose. On a heavy setting it is up to 160 screen
  px of stroke the artist never made.
- **Nothing previewed it.** While the pen is down the preview ends at the tip;
  the winched run cannot be shown, because until the pen comes up there is no
  winch. That makes the commit differ from the preview at the last possible
  instant, which is the one thing the renderer's `preview == committed`
  invariant exists to forbid.

Parking makes the release the cheapest event in the stroke — the last towed
emission is simply the last control point — and extends the dead zone's
guarantee over the end of the mark as well as its middle: the release drift of
§6.2 is sub-pixel nib wander deep inside the string, so it never steers the tip
at all.

The price is stated rather than compensated. A flick shorter than the string
never brings it taut, so it lays down only the dab it was showing while it was
drawn; a hatch tick wants a brush with little smoothing or none, which is the
same sentence this section opened with about the pencil. And the trailing taper
lands on the towed trail's own end rather than on a splice, so a lettering
brush's exit is the tractrix it was already tracing.

### The string is visible

While a smoothing brush tows, the overlay draws **the string itself**: a hairline
from tip to pointer that sags while slack and straightens as it pulls taut. The
one thing that makes deliberate lag feel like latency is being unexplained; the
string makes the mechanism legible — the sag *is* the dead-zone state, readable
at a glance — and it is what makes heavy settings feel like towing a real tool
rather than fighting a broken one. Frontend-only, in the same screen-space
overlay class as the presence cursors; the engine renders pixels, not rigging.

### Where it attaches, and where the knob lives

The tow is an input transform: it sits between the raw `InputSample` stream and
`PathFitter::push`, inside the session's stroke builder — one stage *upstream*
of the fitter-to-renderer seam that §6.9 attaches at, with the same consequence
stated one level earlier. Nothing downstream of the fitter learns that smoothing
exists: the record, the renderer, the save format, replay, goldens, undo and the
CRDT all see an ordinary stroke whose path happens to be calm.

That is also why the amount is **not** a `BrushParams` field. The stored path
already embodies the smoothing, so a field there would be one that replay reads
and ignores — inert in the document domain, which is the class this codebase
deletes. The format would absorb the field now (§8), which is the point: the reason not
to carry it is that replay would ignore it, and that reason never depended on the
encoding. Instead:

- **The engine is told the string length per gesture.** `GestureCommand::Start`
  gains a `rope: f32` alongside `tolerance` — in canvas px, `0` = no tow is
  constructed at all, today's path bit for bit. Like the tolerance, it is fixed
  for the gesture.
- **The frontend owns the feel mapping**, exactly as it owns the tolerance and
  §6.9's dwell: the per-brush amount `0..1` maps to a string length in **screen
  px** (quadratic, so the low end is fine-grained), converted through the view
  at gesture start. Screen px because wobble is a fact about the *hand* — the
  same tremor spans 64× more canvas zoomed out than in — with the elegant
  consequence that zooming in shrinks the dead zone in canvas terms: the escape
  hatch from heavy smoothing is the one artists already reach for to do fine
  work.
- **The amount is stored with the preset, UI-side** — a field of the preset
  library (`stark-ui`, localStorage; the self-describing format reconciles by
  name, absent = 0), and so of every quick slot bound to a preset, since a slot
  stores the preset's name and only its own size and flow (§18.1.8) — never of
  the action log.

The name collision is deliberate avoided: `path.rs` already has a private
`SMOOTHING` — the fit's curvature ridge, numerical conditioning, not a feel knob
— so in code this feature is the **tow** (`stark-engine::tow`, unit-testable
headless), and "smoothing" is its user-facing name.

### Interactions

- **The fitter.** The live end now pins to the towed tip rather than the pointer
  — the preview ends at the tip, and the string explains the gap to the cursor.
  A towed trace is smoother than the device tolerance, so the fit's error rule
  simply buys fewer control points; `tolerance` still declares the device's
  tolerance, unchanged, because it states what position *differences* mean, which
  the tow does not alter.
- **Shape assist (§6.9).** Recognition works on `PathFitter::trace`, which is
  now the towed trace — a rough loop drawn through a smoothing brush arrives
  *cleaner*, so an inking brush snaps more readily, which is coherent: the brush
  that promises confident lines is the one quickest to believe you meant one.
  The dwell watcher stays on the raw pointer (stillness is a fact about the
  hand), and post-snap steering consumes the raw pointer — the tow feeds **the
  fitter only**, so adjusting a snapped shape is never towed. One stitch joins
  the two: at the snap the raw pointer sits up to a rope beyond the towed trace
  the shape was recognized from, so the session records the string's standing
  offset and every steer applies the pointer *shifted by it* — the hand's
  deltas land 1:1 and the first move does not jump, the same bargain the grip
  strikes for the fit residual.
- **Selection gestures.** Untouched — smoothing is a brush property, and a
  marquee or lasso fits no curve. The eraser is a brush preset and gets its own
  amount like any other.
- **The brush editor's preview.** Its test stroke is a *recorded hand* — the
  user's own drag on the preview canvas — so the replay that re-renders it on
  every edit runs through the tow at the live amount, and the Smoothing slider
  shows its work on the stroke beside it. Drawing on the preview tows live for
  the same reason. The one replay that stays raw is the fixed red reference
  stroke, which is a generated straight line and not a hand at all.

### Testable properties

Each of these pins a claim above: cutting a target segment anywhere yields the
same towed path (partition independence, exact); a straight tow settles to a
trail of exactly `L`; `rope = 0` is the identity on every corpus stroke
(goldens untouched by construction); the pen-up leaves the tip exactly where the
rope towed it, a string short of the lift; a staircase inside the dead zone
parks the tip — and, being never towed at all, is the whole of a flick that
short; the fitted knot count on a towed jitter trace drops against the raw fit.

**Not built, and deliberately:** prediction (a negative-lag mode that
extrapolates the pointer — a latency lever, §13's ledger owns that trade);
a soft-rope *pursuit* variant that creeps inside the dead zone, kept in reserve
in case feel-testing wants continuous response at low amounts more than it wants
the parked-tip stability; a mid-stroke modifier to change the amount; smoothing
for the lasso. Each is local to the tow and its Start parameter.

## 6.12 The eraser — coverage subtracted through the slab law

What a brush **does** is a sum type: `BrushParams::effect` is a
`BrushEffect` — `Paint`, carrying its opacity, the dynamics, the color
dynamics and their pen mappings, or `Erase`, carrying its opacity, its own flow
and its own mappings.
Each variant holds exactly the knobs that exist under it, so an eraser has no
`lift` for a pass to ignore and no color dynamics for a panel to show; what is
left on `BrushParams` — the tip, the tooth, the jitter, the tapers, the drain —
shapes the swept extent whichever effect then uses it.

`BrushEffect::Erase` routes a stroke through the **erase pass**: the same swept
extent every brush rasterizes, turned on the layer's *visible* opacity instead
of laid as paint. It exists because the conserved-flux eraser — `lift` up,
`deposit` 0 — is a *scraper*: it removes a fraction of the paint's **height**
per pass, and the slab law `1 − exp(−K·op·h)` (§6.1) is nearly flat in `h`
wherever paint is thick, so the knob never meant anything in the units a
digital artist reads. The erase pass acts in those units directly, and its dial
is honest: `opacity = 0.5` under a saturated stroke leaves half the visible
opacity that was there.

### The law

Per texel, with `m₀ = op·h` the resident paint's optical mass and `m_e` the
stroke's accumulated **transparency mass** (below):

```text
w  = 1 − exp(−K·m_e)                    the coverage the stroke would have covered by
v₁ = v₀ · (1 − opacity·w)               what must remain visible
m₁ = −ln(1 − v₁)/K,   h₁ = h₀·(m₁/m₀)   the slab law inverted; op, latent, residual untouched
```

Three properties are the design:

- **The erased fraction is the coverage the same stroke would have painted.**
  `w` is the parcel alpha of §6.2's deposit with per-unit opacity 1 — the same
  prefix-τ sweep, the same drain, tooth and jitter gates — so an eraser's edge
  is exactly the soft edge its brush would have drawn, and its flow is a rate
  with the meaning Flow always has.
- **`opacity` is a ceiling, not a rate.** `w` saturates at 1 however long
  the stroke works one spot, so a stroke removes at most `opacity` of what it
  finds; scrubbing walks the edge toward the cap rather than eating past it. A
  second *stroke* takes `opacity` of the remainder, which is what every
  buildup eraser does. The rate is the effect's own `flow`
  (`EraseEffect::flow`) — its own field, not a reading of the paint effect's,
  so switching a brush's effect never re-interprets a number that meant
  something else.
- **Only the amount falls.** The per-unit opacity is what the pigment is; the
  latent and the residual are which pigment. Erased paint is *less of the same
  paint* (§6.1), so the pass rewrites the height lane alone and the color
  textures pass through bit-for-bit.

An erasing brush **carries no color at all** — the erase parcel is "fully
opaque transparency", mass = height, `fill.wesl`'s arithmetic: an eraser lays
nothing for a color to be a property of, so the pigment lives inside the paint
effect (`PaintEffect::color`) and a stored erase stroke simply has none. The
*hand* still has one while the eraser is held — the Color panel writes it
whatever brush is in force (§18.1.8) — but that is the frontend's fact:
`stark-ui`'s `BrushConfig` holds both effects' configurations with one in
force, so the color lands on the remembered paint side, and a fill drawn
mid-erase still lays it (the hand's color rides beside the params on
`ViewCommand::SetBrush`, into `Session::color`). That the dynamics axes do not
run is the enum, not a rule: an `Erase` brush has none. No region is ever
needed either, so no tip is too large for this path (`dynamics_setup` answers
`Erase` off the variant alone).

### Why the extent accumulates across pieces

`1 − erase·w` is neither additive in τ nor `1 − exp(−k·τ)`, so it is *not* one
of the two forms that compose under re-cutting (§6.2) — applied per piece it
would compound at every cut a live stroke makes, and the committed piecewise
render would band against its own replay. The pass therefore keeps the two
composable halves separate: the **sweep** (`stamp.wesl::fs_erase`) accumulates
the transparency mass additively into a one-channel accumulator per touched
tile — across segments *and across pieces*, so re-cutting changes nothing —
and the **integrate** (`erase.wesl`) applies the non-composable law exactly
once per tile per render, always from the **pristine** base: the tile as the
stroke first found it, never an earlier piece's output.

Both ride the stroke's carry (`ParcelCarry`, beside the stamp loop's reservoir
in `ToolState`): per touched tile, the pristine handle (an `Arc` bump, tiles
being copy-on-write) and the accumulator lease. That carry, and the whole of the
bookkeeping around it, is shared with the scaled swept deposit — the other
effect whose law is outside §6.2's two forms — in
`accum::IncrementalTileAccumulator`; this pass is the `Skip` half of its
bare-canvas decision, and `stamp.wesl::fs_erase` plus `erase.wesl` are what it
contributes. A resuming piece **copies** the
carried accumulator into a working texture and extends the copy — the carried
one is only ever read — which is what lets the live tail re-render from the
same frozen head every pointer move, and makes `preview == committed` hold with
no seam term at all: the pieces' render *is* the whole's sums.

A tile the layer does not have is nothing to erase: no output tile, no
accumulator, no entry in the carry — an eraser over bare canvas mints nothing.

### What it costs, and the one soft spot

Per piece and touched tile: one accumulator copy, one sweep draw, one
fullscreen integrate — the swept fast path's cost plus a copy, no dynamics
region, no sequential loop. The accumulators are one `R16Float` tile texture
per touched tile for the stroke's lifetime, pooled (`scratch`).

The soft spot is heavy impasto. Visible opacity stops encoding mass around
`OPAQUE_MASS` (§6.1's law is saturated there), so honouring "remove half of
what the eye sees" on paint far past it means removing nearly all of its mass —
a partial-strength eraser grazing thick impasto cuts its *relief* much faster
than its opacity, because that is the only mass the target opacity leaves room
for. This is the arithmetic of the promise, not a defect to tune away: the
eraser is the digital artist's tool, calibrated in the coverage domain; paint
as *material* answers to the knife (`lift`), which removes amount and conserves
it. Both exist because they are different questions.

**Under the pen** (`EraseModulations::opacity`): the erase sweep carries the
ceiling lane beside its accumulator, summed exactly as the deposit's is (§6.2,
*The ceiling under the pen*), and the integrate reads the claimed coverage —
the largest ceiling any pass asked for, averaged over the texel — in place of
`w`. A light touch thins where a heavy one clears, and a light pass back over a
cleared band takes nothing more from it (`tests/erase.rs`,
`an_eraser_under_the_pen_removes_what_the_pen_asks`).

## 6.13 The liquify brush — dragging the picture itself

The fourth effect (`BrushEffect::Liquify`): the stroke **warps** the canvas
under it. Every channel — color, per-unit opacity, height — follows the travel
together as a resample of the field, so structure *moves* where the wet loop's
smudge would mix it toward a mean: an edge dragged is that edge, displaced; a
texture rides along whole. It is the brushed cousin of the selection
transform's mesh warp (§16.9) — the same "moving paint through a map" idea,
keyed to a gesture instead of a lattice — and the tool Photoshop, Procreate
and Clip Studio each ship under this name.

Its own effect rather than a `Wet` at some rate, for the eraser's reason:
warping and smearing are different tools with different available features. A
smudge trades paint through a tool reservoir and conserves height; a warp has
no reservoir, no pigment, no color for a jitter to wander, and no opacity for a
ceiling to cap — the follow is a fraction of *travel*, so scrubbing keeps
carrying the way the bleed keeps buying distance, and there is no saturated
stroke for a ceiling to be a fraction of.

### The kernel: a backward-mapped gather on the region loop

Liquify runs the dynamics path's own machinery (§6.2's region composite, piece
chunking and CoW write-back — `StrokePath::Liquify`,
`gpu/stroke/dynamics/plan.rs::liquify_plan`) with one kernel in place of the
whole exchange: per flattened segment, a **snapshot** of the extent scratch and
a **warp** dispatch (`dynamics.wesl::warp`). Each covered texel pulls the
region from upstream along the local travel tangent, one bilinear tap of every
channel:

```
follow(x) = min(strength · e(x) / peak_τ, travel) · sel(x)      canvas px
out(x)    = under(x − follow(x) · t̂(x))
```

- **`e(x)` is the deposit's own exposure**: the prefix-τ difference over the
  segment's sweep, gated by the tooth (§6.4), the deposit jitter and the drain
  at the texel's own arc. Riding that integral is what makes the follow
  **additive in τ** — §6.2's composition rule — so overlapping quads of
  consecutive segments hand a texel exactly the follow the joined segment
  would, and every tip knob shapes the warp the way it shapes a deposit: a
  taper's point drags less, a dry tip's pull skips the substrate's valleys, a
  drained stroke lets go as it travels, a stamp's gaps slip.
- **`peak_τ` normalizes to the tip's densest texel** (`assets::peak_tau`, on
  the resolved tip): at strength 1 the paint under the tip's core keeps pace
  with the hand exactly, and the rest follows in proportion to the coverage
  genuinely over it — so the falloff is the *tip's*, and hardness is the
  shape knob rather than a second strength.
- **The selection scales the follow, not a blend of the result** (§6.8): a
  half-selected texel's paint goes half as far, so a mask rim tapers the warp
  instead of ghosting two pictures over each other.

The `min` against the travel is load-bearing: `e ≤ peak_τ · travel` per row
makes every pull land inside the segment's own sweep box, which is what the
snapshot holds — so a **warp slot's snapshot copies its whole square** where
every other slot's skips texels outside the sweep (`snapshot_at` reads the
slot's `drag` lane to know). The clamp of the taps into the scratch is the
no-flux wall for whatever rounding remains, exactly the bleed ladder's.

Sequenced in place, the gather reads the region as earlier segments left it,
so a stroke drags **its own trail** when it crosses it — order-dependence is
the loop's, inherited whole. And because the warp's entire state *is* the
region, the loop's incremental story carries over with nothing to carry: no
reservoir, no settle (each segment's follow is applied whole as the tip
passes, so a break of contact strands nothing — the bleed axis's argument),
and `ToolState` stays `None` across ranges. A cut — a frozen head's bake, a
region-sized piece — materializes the warp into tiles and the next stretch
gathers through them; what a cut costs is the f16 store every loop cut costs,
plus one extra resample where the tail re-crosses the boundary, and the
corpus's `liquify` case bounds it in the same `seam` column as the smear's.

### What it trades, honestly

**Composition is preserved pointwise; height is not conserved.** Every value
the stroke leaves is one the field held nearby — but a warp with falloff is
not divergence-free, so stretching duplicates paint and pinching discards it,
exactly as the mesh warp of §16.9 resamples a selection and as every liquify
tool behaves. That is the tool's identity, stated against §6.1's ledger: the
conserving movers are the wet loop's fluxes; liquify treats the canvas as a
*picture*. (Scaling the height by the warp's Jacobian — pinches piling paint
up like real impasto — is the recorded refinement if the physical reading is
ever wanted; it is one finite-difference of the follow field away.)

**Each resample softens by its tent filter** — a sub-texel blur per step,
bounded by holding `strength · travel` fixed per segment
(`budget::WARP_TRAVEL_STEP`, the exchange step's cousin: a gentle drag earns
longer segments because the error is first order in the follow a segment
completes, and halving the step halves it with no knee). A long scrub
therefore softens what it carries — far less than a smudge averages, and the
honest cost of warping in place. The alternative — accumulating a
displacement field for the whole stroke and resampling the pristine base
once — keeps edges bit-sharp *within* a piece and was weighed and set aside:
its field cannot cross the cuts the live path is made of (a frozen head bakes
tiles, a piece's region moves), so every cut becomes a visible double-resample
seam between preview and replay, which is the column §6.2 already rejected a
cheaper error over. If the warp ever wants it, the field, like the wick's
ladder lesson, is recorded here rather than relearned.

Strength 0 — and the zero-rate slot a modulation can produce — stores nothing
at all (`WARP_MIN_FOLLOW`, the deposit's rewrite guard in this kernel's
terms), so a near-inert drag cannot walk texels down the f16 lattice; the
stores that do land go through `lib::store::f16_nearest` like every store in
the loop. A drag along a field that does not vary in its own direction is the
identity to the byte (`tests/liquify.rs`), which is this effect's reading of
"conserves where there is nothing to move".

*Determinism* — the plan is CPU float math over the record and the piece's
geometry, the gather's weights are a pure function of its inputs
(`FLOAT32_FILTERABLE` is never needed), so replay and `preview == committed`
hold exactly as they do for the loop (§12.1). *Cost* — two small dispatches
per segment (against the wet loop's three-plus), no reservoir passes, no
settle; the snapshot covers the square rather than the rect, which is the
bleed-weight pass's cost class.
