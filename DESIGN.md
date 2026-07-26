# Stark — Design

This document describes the architecture for **Stark**, the GPU-accelerated 2D
painting application specified in [GOALS.md](GOALS.md). It is written to be
directly implementable: the major modules, types, data flows, and the GPU
strategy are concrete enough to start building, while the seams are drawn so the
ambitious parts (natural-media brushes, infinite canvas, timelapse) can grow
without rework.

## 1. Guiding principles

1. **The document is a list of actions, not a bag of pixels.** Pixels are a
   *derived, cached* view of a replayable action log. This single decision
   delivers three goals at once: the native save format (GOALS §Outputs), undo
   after load, and timelapse rendering — all fall out for free.
2. **Cheap state, expensive pixels — keep them separate.** The `history` crate
   wants `State` values it can clone and retain in O(log n) snapshots. We make
   `State` a *persistent, structurally-shared map of tile handles*, never raw
   pixels. Cloning a document state is a handful of `Arc` bumps; the heavy GPU
   memory is shared across versions and reclaimed automatically by reference
   counting.
3. **One rendering path, used three ways.** The same deterministic stroke
   renderer drives live painting, history replay (undo/redo, load), and golden
   tests. If those ever diverge, tests lie. So we make replay *the* path and
   live painting an incremental front-end to it.
4. **Frontend-agnostic core.** The engine knows nothing about Dioxus, windows,
   or event loops. It consumes `InputCommand`s and GPU handles, and exposes
   state + a render entry point. Dioxus is one consumer; headless golden tests
   are another.
5. **Data-driven where it counts.** Channels (color/depth/wetness/…), tools,
   actions, and blend modes are open sets behind small traits/enums so new
   capabilities are additive.
6. **Perceptual color is the working space.** All color channels store and blend
   in **Oklab** (GOALS §1), so brush mixing, layer compositing, and gradients are
   perceptually uniform; conversion to a display space happens only at the final
   present. Color math never touches gamma-encoded sRGB.
7. **Convergence-ready from day one.** Every action carries a globally-unique id
   and the document derives purely from a *deterministically ordered* replay of
   actions. Single-user that's a linear timeline; multi-user (GOALS §Frontend,
   peer-to-peer) it becomes a replicated log that all peers replay to the same
   pixels. The same determinism that makes golden tests work makes collaboration
   converge — see §12.

## 2. Crate / workspace layout

```
stark/
├── Cargo.toml                  # workspace
├── crates/
│   ├── stark-core/             # the engine — no UI, no windowing
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine.rs       # the actor: owns everything, runs the loop
│   │   │   ├── command.rs      # InputCommand (raw user intent)
│   │   │   ├── session.rs      # ephemeral state: tool, view, in-flight stroke
│   │   │   ├── document/       # versioned state (the history)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── action.rs    # Action + ActionId (replayable mutations)
│   │   │   │   ├── state.rs     # DocState: persistent layer/tile map
│   │   │   │   ├── timeline.rs  # Timeline trait; Linear + Replicated impls
│   │   │   │   └── layer.rs
│   │   │   ├── color.rs         # Oklab working space, conversions, mixing
│   │   │   ├── assets.rs        # content-addressed brush/image asset store (§6.6)
│   │   │   ├── gpu/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── context.rs   # device/queue wrapper, capabilities
│   │   │   │   ├── tile.rs      # TilePool, CoW tile handles, channel set
│   │   │   │   ├── stroke.rs    # the brush engine / stroke rasterizer
│   │   │   │   ├── composite.rs # layer compositing + media lighting → display
│   │   │   │   └── readback.rs  # GPU→CPU texture readback (export, goldens)
│   │   │   ├── geom.rs          # tile coords, view transform, AABB
│   │   │   ├── path.rs          # streaming B-spline stroke fit + adaptive flatten (§6.2)
│   │   │   ├── spline.rs        # clamped cardinal cubic B-spline + least-squares solve
│   │   │   └── io.rs            # save/load of the action log
│   │   └── tests/
│   │       └── golden/         # scripted command sequences + reference PNGs
│   ├── stark-shaders/          # WESL sources + build.rs (wesl link/compile)
│   │   ├── build.rs
│   │   └── src/shaders/*.wesl
│   ├── stark-net/              # iroh transport ↔ Replicated timeline (optional)
│   └── stark-ui/               # Dioxus 0.7 frontend
└── DESIGN.md
```

Rationale: `stark-core` is the testable, frontend-agnostic backend GOALS calls
for. It is also **network-agnostic**: it owns the *merge semantics* of the
action log (the `Timeline` trait) but not the wire transport. `stark-net` adapts
iroh to it (§12) and can be pulled in by the frontend or omitted entirely.
`stark-shaders` is split out so shader compilation (a build step) doesn't pollute
the engine crate and can be reused by tools. `stark-ui` depends on core, never
the reverse.

## 3. Layered architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ stark-ui (Dioxus)   DOM chrome + a GPU canvas surface           │
│   - sends InputCommand     - subscribes to ObservableState      │
│   - owns wgpu::Surface, calls engine.render(view, target)       │
└───────────────▲─────────────────────────────┬───────────────────┘
        InputCommand                  ObservableState (signal)
                │                             │
┌───────────────┴─────────────────────────────▼────────────────────┐
│ Engine (async actor)   owns GPU + Session + Document             │
│   command → Session interprets → maybe emits Action              │
└───────┬──────────────────────────┬───────────────────────────────┘
        │                          │
┌───────▼──────────┐     ┌─────────▼───────────────────────────────┐
│ Session          │     │ Document = History<Action>              │
│  - active tool   │     │  - DocState (persistent tile/layer map) │
│  - brush params  │     │  - version timeline (undo/redo)         │
│  - view xform    │     └─────────┬───────────────────────────────┘
│  - in-flight     │               │ Action::apply(state, ctx)
│    stroke buffer │     ┌─────────▼───────────────────────────────┐
└──────────────────┘     │ GPU subsystem                           │
                         │  TilePool · StrokeRenderer · Compositor │
                         │  ShaderModules (WESL)                   │
                         └─────────────────────────────────────────┘
```

The crucial split is **Session vs Document**:

- **Session state** is ephemeral and *not* in history: current tool, brush
  settings, the view transform (pan/zoom), and the in-progress stroke being
  dragged out. Panning the canvas or switching tools must never create an undo
  step.
- **Document state** is everything that defines the artwork and is versioned by
  the `history` crate. Only committed, replayable mutations live here.

## 4. Commands vs. Actions (the most important boundary)

Two distinct vocabularies, deliberately not merged:

`InputCommand` — *raw, high-frequency user intent*, including ephemeral input
that never lands in history:

```rust
pub enum InputCommand {
    // --- stroke lifecycle (high frequency) ---
    StartStroke { tool: ToolId, sample: InputSample },
    StrokeTo    { sample: InputSample },
    EndStroke,
    CancelStroke,

    // --- history navigation ---
    Undo,
    Redo,

    // --- session / view (never historized) ---
    SetTool(ToolId),
    SetBrush(BrushParams),
    Pan { delta: Vec2 },
    Zoom { center: Vec2, factor: f32 },

    // --- document edits that ARE historized ---
    AddLayer { above: Option<LayerId> },
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),

    // --- io ---
    Load(DocumentFile),
}

pub struct InputSample {       // one pen/mouse sample
    pub pos: Vec2,             // canvas-space position
    pub pressure: f32,
    pub tilt: Vec2,
    pub time: f64,             // for velocity & timelapse
}
```

`Action` — *committed, deterministic, serializable document mutations* — the unit
the timeline stores and replays, and the unit we serialize to disk. Every action
is **globally identified** so it can later live in a replicated, multi-peer log
(§12) without changing its meaning:

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct Action {
    pub id: ActionId,           // globally unique; also gives total order
    pub kind: ActionKind,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Ord, Serialize, Deserialize)]
pub struct ActionId {
    pub lamport: u64,           // logical clock → causal/total ordering
    pub actor: ActorId,         // who authored it (one local user, or a peer)
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ActionKind {
    CommitStroke(StrokeRecord),
    AddLayer { id: LayerId, above: Option<LayerId> },
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    Undo(ActionId),             // undo-as-an-action (see §5.4 / §12)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StrokeRecord {
    pub layer: LayerId,
    pub tool: ToolId,
    pub brush: BrushParams,       // color in Oklab (§6.5); shape by AssetId (§6.6)
    pub path: Vec<ControlPoint>,  // cubic B-spline control points, fitted (§6.2)
    pub seed: u64,                // makes any brush jitter reproducible
}
```

`ActorId` is a single fixed value in the single-user case (and maps to an iroh
`NodeId` when collaborating). Generating ids locally costs nothing now and is the
one piece of forward-compatibility that would be painful to retrofit later, so we
pay it from the first commit.

The mapping happens in `Session`:

```
StartStroke/StrokeTo  → accumulate into an in-flight StrokeRecord,
                        render incrementally onto CoW preview tiles
EndStroke             → finalize record, push Action::CommitStroke onto History
CancelStroke          → discard preview tiles, no Action
Pan/Zoom/SetTool      → mutate Session only
Undo/Redo             → History::pop / re-derive version
```

Because `StrokeRecord` carries the entire sampled path plus a brush seed, a
committed stroke replays bit-for-bit — the foundation of both undo and golden
tests.

## 5. The history model (and why it's cheap)

The `history` crate gives us `History<A: Action>` storing O(log n) full `State`
snapshots, O(1) amortized push, O(log n) pop, and `get_state(version)` in
O(k + log n) by replaying from the nearest snapshot. Its `Action` trait is
roughly:

```rust
trait Action {
    type State;
    type Context;
    type Error;
    fn apply(&self, state: Self::State, ctx: &mut Self::Context)
        -> Result<Self::State, Self::Error>;
}
```

We bind it as:

```rust
impl history::Action for Action {
    type State   = DocState;       // CHEAP to clone (see below)
    type Context = ApplyCtx<'_>;   // GPU device/queue + TilePool + renderers
    type Error   = EngineError;

    fn apply(&self, state: DocState, ctx: &mut ApplyCtx)
        -> Result<DocState, EngineError>
    {
        match &self.kind {
            ActionKind::CommitStroke(rec)  => ctx.stroke.render(state, rec),
            ActionKind::AddLayer { id, above } => Ok(state.with_layer(*id, *above)),
            ActionKind::Undo(target)       => ctx.replay_without(state, *target),
            // ...
        }
    }
}
```

The document does **not** call the `history` crate directly; it goes through a
`Timeline` trait so the storage strategy can change without touching `Session`,
`Engine`, or the GPU code:

```rust
pub trait Timeline {
    fn push(&mut self, action: Action, ctx: &mut ApplyCtx);
    fn current(&self) -> &DocState;
    fn undo(&mut self, ctx: &mut ApplyCtx) -> bool;   // navigation (solo)
    fn redo(&mut self, ctx: &mut ApplyCtx) -> bool;
    fn clone_actions(&self) -> Vec<Action>;           // the save payload (§8)
    // Shared-mode hooks, defaulted so LinearTimeline ignores them (§12):
    fn undo_as_action(&self) -> Option<ActionId> { None } // what Undo should target
    fn redo_as_action(&self) -> Option<ActionId> { None } // which Undo to un-undo
    fn merge(&mut self, action: Action, ctx: &mut ApplyCtx) -> bool { false }
}
```

- **`LinearTimeline`** — the single-user impl, a thin wrapper over
  `history::History<Action>`. This is what ships first.
- **`ReplicatedTimeline`** — the multi-peer impl (§12): a totally-ordered set of
  actions reusing the very same `history::History` as a *materialization cache*
  for the ordered prefix.

`Session`/`Engine` only ever see the trait, so collaboration is added by swapping
the impl, not by surgery on the engine.

### 5.1 `DocState` is a persistent tile map, not pixels

```rust
#[derive(Clone)]
pub struct DocState {
    pub layers: rpds::Vector<Layer>,   // persistent (structural sharing)
    pub bounds: CanvasBounds,          // union of populated tiles (infinite)
}

#[derive(Clone)]
pub struct Layer {
    pub id: LayerId,
    pub blend: BlendMode,
    // sparse map: only populated tiles exist (infinite canvas)
    pub tiles: rpds::HashTrieMap<TileCoord, TileHandle>,
}

#[derive(Clone)]
pub struct TileHandle(Arc<GpuTile>);   // Arc bump = the entire "clone" cost
```

Cloning `DocState` clones `rpds`'s persistent collections — internally just bumps a
few `Arc`s (GOALS §dependencies). This is what makes the `history` crate's
snapshot retention affordable: each retained version holds *references* to shared
GPU tiles, not copies. `rpds`'s structural sharing also gives us cheap *diffing*
between two `DocState`s, which the compositor uses for damage tracking (§6.3) and
the collaboration layer uses to merge concurrent edits tile-by-tile (§12).

### 5.2 Copy-on-write at tile granularity ties memory to history

A stroke touches a small set of tiles. `StrokeRenderer::render` produces a new
`DocState` where **only the dirtied tiles are replaced** by freshly allocated
GPU tiles; every untouched tile is shared with the previous version.

```
version N      version N+1 (one stroke over 3 tiles)
┌──┬──┬──┐      ┌──┬──┬──┐
│A │B │C │      │A │B'│C │     B' is new; A and C are the SAME Arc.
├──┼──┼──┤  →   ├──┼──┼──┤
│D │E │F │      │D'│E'│F │     D',E' new; F shared.
└──┴──┴──┘      └──┴──┴──┘
```

The consequence is elegant: a `GpuTile` is freed back to the `TilePool` exactly
when its `Arc` refcount hits zero — i.e. when no `history` snapshot references it
anymore. **History retention drives GPU memory reclamation for free.** No manual
GC, no leak.

### 5.3 Undo/redo cost

For versions the `history` crate retains as snapshots, undo is *instant*: we
already hold the tile map. For versions between snapshots, `get_state` replays
the few intervening `CommitStroke` actions via `apply` — re-rasterizing those
strokes on the GPU. Strokes are small and deterministic, so replay is fast, and
because snapshots are cheap we can afford a dense checkpoint policy to keep
replay depth tiny. Redo is symmetric.

### 5.4 Two flavors of undo

There are deliberately two undo mechanisms, and they don't conflict:

- **Local timeline undo** (`Timeline::undo`) — the fast single-user path above,
  pure `history` navigation, nothing written to the log.
- **`ActionKind::Undo(target)`** — undo *as a logged action*. This exists for
  collaboration (§12), where "undo" must be a fact other peers can see and order,
  and must mean "undo *my* action" not "undo whatever happened last." It is
  deliberately **not interpreted by `Action::apply`** (undo needs the whole log,
  not just the prior state): the timeline layer computes the log's **effective
  sequence** — every non-`Undo` action not suppressed by an effective `Undo`
  (`timeline::effective_actions`) — and only that is ever materialized. Redo is
  an `Undo` of an `Undo`. Single-user mode never emits these (solo undo is pure
  `history` navigation), and a solo *load* of a shared log simply replays the
  effective sequence, flattening the undos away.

## 6. Rendering the canvas (infinite, tiled, multi-channel)

### 6.1 Tiles and channels

A tile is a fixed `TILE_SIZE` (256×256) square in canvas space, addressed by
integer `TileCoord(i32, i32)`. Sparsity gives the infinite canvas: only painted
tiles allocate. Each tile is **multi-channel** — this is what enables strokes
that affect more than color (GOALS §1):

```rust
pub struct GpuTile {
    pub color:  wgpu::Texture,   // Rgba16Float — Oklab (L,a,b) + premult alpha
    pub height: wgpu::Texture,   // R16Float — total paint height, thickness computed by subtracting surface height
    pub wet:    wgpu::Texture,   // R16Float — wetness for wet-on-wet mixing
    // future channels (normal, granulation, …) are additive here
}
```

The color texture stores **Oklab** components, not sRGB/RGB. Linear 16-bit float
comfortably holds Oklab's range and the negative `a`/`b` chroma axes, and keeps
blends perceptually uniform (GOALS §1). Alpha is premultiplied against `L,a,b`.

> **The color alpha channel is *only* the paint's per-unit-thickness opacity** — a
> material property (how opaque the pigment is per unit of thickness). It says
> **nothing** about how much paint is on the canvas, nor even whether any paint is
> present. **The amount (and presence) of paint is the `height` channel** (precisely,
> `height − surface_height`, the paint *thickness*). The two combine only at display
> time in the translucent-slab law `visible = 1 − exp(−K · opacity · thickness)`
> (media pass, §6.3). Consequences that the brush dynamics must respect:
> - To **conserve paint** (move it without creating or destroying), conserve
>   **height** — never the alpha. Alpha is per-unit and is carried as a
>   height-weighted blend of the picked-up paint's opacity; it is not consumed.
> - A thin layer of opaque paint (alpha ≈ 1, tiny thickness) is *barely visible*; a
>   thick layer of translucent paint can be very visible. Opacity alone is not coverage.
> - Lifting paint reduces the canvas **height** (less paint), leaving the remaining
>   paint's per-unit alpha unchanged; the source lightens because thickness — not
>   alpha — drops.

Channels are referenced through a small `ChannelSet` descriptor so the renderer,
compositor, and tile pool agree on layout without hard-coding it everywhere — a
new channel is a descriptor entry plus shader usage, not a structural rewrite.

`TilePool` recycles GPU textures of each channel format to avoid per-stroke
allocation churn; `acquire()` returns a cleared tile, `Drop` of the last
`Arc<GpuTile>` returns it to the free list.

### 6.2 The brush engine — natural media

Stroke rasterization is **swept-segment along a fitted path** (detailed under
*Path representation* and *Continuous stamping* below): pointer samples are fitted
to control points, expanded to a smooth polyline, and each short segment is swept
as a single quad. Layered on top is a pluggable **brush-dynamics** model that can
carry *loaded paint* and smear what is already on the canvas, so wet-on-wet mixing
feels physical (see *Wet mixing & brush dynamics* below). Everything is
deterministic — the only randomness is the explicit `seed` — so live paint,
replay, and golden tests agree.

**Path representation & cubic interpolation.** `path.rs` keeps three
representations deliberately distinct: an **`InputSample`** is one raw pointer
report (transient, never stored), a **`ControlPoint`** is a fitted curve knot
(this is what `StrokeRecord::path` holds), and an **`IntermediateSample`** is a
point *of the curve* — position plus its derivative, with the pen attributes
interpolated there — produced at render time and consumed by the stamp generator.

A `PathFitter` streams samples into control points as a **least-squares clamped
cubic B-spline** (`spline.rs`), grown and refit as they arrive:

- **Grow.** The control polygon lengthens with the stroke — one point per
  `KNOT_SPACING` of arc length, plus more wherever the input's *sagitta* over one
  such interval exceeds `SAGITTA_TOLERANCE`. Fitting is what smooths, so there is
  no separate low-pass stage: a polygon far coarser than the jitter averages a
  pixel-quantized staircase away. The arc-length floor is not redundant with the
  sagitta test — it is what makes the polygon grow, and so freezing advance, on a
  stroke the fit is already perfect on.
- **Refit** every sample, but only over the *live* points and the *free* control
  points, so the work per sample follows the tail rather than the stroke so far.
- **Freeze** all but the last few control points. Those are final — nothing drawn
  later can move them — which is what makes the fit append-only and lets a caller
  treat the settled prefix (`frozen_spans`) as already rendered.

Both **ends are pinned**: a least-squares fit does not hold them, because a
stretch of parameter with no sample assigned to it costs nothing, so the curve
otherwise starts before the stroke and stops short of the pointer. The start is
set and frozen at the first sample; the live end is moved to the newest sample
each update (and frozen there on release), which is also what keeps the preview
under the cursor.

Pen attributes ride along as **passenger channels**: pressure, tilt and time are
solved against the geometry's own assignment rather than fitted jointly with it,
so a pressure ramp cannot stretch the parameterization the way a longer path does,
and no weighting is needed to reconcile pixels with whatever units they are in.

Rendering expands those control points through that same B-spline — converted per
span to cubic Bézier form, so the derivative is closed-form — into a polyline, and
subdivides **adaptively**: a piece is split until the straight segment standing in
for it is within a bounded error in position, in *tangent direction*, and in the
pen attributes. Sampling therefore follows the curve rather than arc length: a
long gentle stroke costs a handful of segments where uniform stepping cost
hundreds, and a corner still gets the density it needs. The tangent bound is what
buys both — it is the term that cannot be fooled by a symmetric wiggle, and the
one that spikes exactly at a corner. This solves several problems:

- **No stair-step aliasing** — jittery pixel-stepped input (a slow diagonal drawn
  as right/up steps snapped to the device grid) is smoothed and collapses to a
  clean curve instead of axis-aligned segments. This is the fit doing it, and it
  is why `SAGITTA_TOLERANCE` has to sit *above* the input's own quantization: set
  it below and a staircase reads as curvature and gets traced rather than smoothed.
- **Continuous-looking stamping** — stamps ride a smooth path with smooth
  tangents, so even hard-edged tips read as one stroke rather than a row of
  discrete dabs (an approximation of a path integral over the stroke).
- **Smaller files** — a handful of control points replace hundreds of raw
  samples in the action log (§8).

The per-stamp GPU instance is **unchanged**; only stroke→stamp generation
differs. Live preview and commit run the *same* fitter, driven the same way
sample by sample, so the preview at release equals the committed stroke,
preserving the live == committed invariant (§1.3). Fitting and curve evaluation
are fixed float math, so determinism (and golden/replay/save-load equivalence)
holds.

Adaptive sampling has one hard prerequisite, easy to violate silently: **the
deposit must not depend on how the path was cut into segments** (see the swept
deposit below). Anything a segment applies *per segment* rather than per fragment
— the `drain` falloff, the stamp loop's reservoir cadence — also caps segment
length, which the renderer supplies as a length bound rather than the fitter
assuming one (`gpu::stroke::flatten_tolerance`).

**Incremental repaint.** Freezing is what keeps a long stroke responsive. Drawing
a live stroke costs (segments × tiles covered), both of which grow with its
length, so re-rendering the whole thing on every pointer move gets quadratically
expensive. Instead the engine keeps a `FrozenHead`: the settled spans, rendered
once onto the committed document and kept. Each move draws only the live tail over
that — a few spans, whatever the stroke's length — and the head advances as the
fitter freezes more (`StrokeRenderer::render_range`, `path::flatten_spans`;
adjacent ranges share exactly one flattened point, so their segments tile with no
gap and no overlap).

This is the *same* partition-independence the constraint above demands, spent
deliberately: the swept deposit is a definite integral per segment that composes
by summing optical depth, so cutting the path at a span boundary and compositing
the pieces in order gives what one pass gives.

The stamp loop that dynamic brushes run has no such property — it is *sequential*,
each segment reading the canvas the previous one left and the tool the previous one
loaded. It is cuttable anyway, because that carried state is small and entirely
**brush-local**: the reservoir texture (what paint the tip holds, and where on the
tip), plus the travel since the last pickup, which sets the reload cadence. A
`ToolState` remembers both at the freeze boundary and the tail resumes from it. Being
brush-local is what makes this work at all — the state says nothing about *where* the
stroke is, so the region rectangle may change completely between the piece that
saved it and the piece that resumes. The canvas side needs no carrying: it is already
in the head's tiles.

Two things then have to be decided from the **whole** record rather than from the
piece in hand, because a live tail and the commit that replaces it must draw the same
pixels: whether the stroke runs the stamp loop at all (it degrades to the swept
deposit past `MAX_REGION_DIM`) and how finely it flattens (it coarsens past
`MAX_STAMPS`). A gate that looked at the piece would answer differently for a short
tail than for the stroke it belongs to, and the stroke would visibly redraw on
release. See `gpu::stroke::dynamics_setup`.

**Continuous stamping (swept segments).** Discrete dabs are still visible with
hard tips. The fix: stamp each short *segment* of the flattened curve as one
quad whose coverage is the brush **swept** along it — the path integral of the
footprint, instead of point samples. The enabling identity: alpha-"over" is
multiplicative in `(1−α)`, hence additive in **optical depth** `τ = −ln(1−α)`.
So:

- Precompute, per brush, the **prefix integral of `τ` along the travel axis**
  (the tangent the brush is rotated to). A length-`d` segment's swept depth at a
  point is then `prefix(u) − prefix(u−d)` for that row — an O(1) lookup.
- A segment quad outputs `α_seg = 1 − exp(−opacity · sweptDepth)`. Because the
  existing premultiplied-"over" blend across overlapping segment quads combines
  as `1 − ∏(1−α) = 1 − exp(−Σ τ)`, it sums the depths **exactly** — no
  double-counting at joints, no scratch buffer, no second pass. The whole
  stroke's coverage is the continuous path integral `1 − exp(−τ_total)`.
- **Every** channel a segment deposits must therefore be a function of that
  segment's `τ` in one of exactly two shapes: *additive* in `τ` (an amount — the
  height and wet the aux target sums), or `1 − exp(−k·τ)` (a rate — the opacity
  the colour target over-blends). Those are the two that survive re-cutting the
  path, because `τ` is what sums. Any other shape makes the stroke depend on the
  *number* of segments rather than on the path: a per-segment `√`, for instance,
  deposits `Σ√(τ/N) = √(N·τ)`, so the stroke silently gains weight with sampling
  density. That is invisible while sampling is uniform and immediately visible
  once it adapts, which is why the two forms are a standing constraint on the
  stamp shaders and not a detail of one.

This removes intra-stroke banding while keeping the single-pass over-blend
architecture (both color spaces share one premultiplied-"over" stamp shader,
§6.7). Segments need only be short enough that the line + constant-radius
approximation holds, so the sweep uses *fewer* primitives than the dab model.
Caveats: per-stamp angle jitter no longer applies (the brush follows the tangent
continuously); the round tip's prefix depends on `hardness`, so it is generated
per stroke (image brushes precompute theirs at import, §6.6); a click is a
degenerate segment given a minimal length.

**Live vs. replay unification:** live painting renders the in-flight (fitted)
stroke onto CoW preview tiles; commit/replay render the same `StrokeRecord`
through the same path → same stamps, same pixels.

**Wet mixing & brush dynamics — the sequential swept-exchange loop.** To smear
paint already on the canvas — the core of a natural-media feel — the brush picks
up wet pigment under it, carries it, and lays down an evolving mix downstream.
This is **sequential and order-dependent** (what's under the brush includes what
it deposited a moment ago), which the parallel swept pass cannot express. The
loop embraces the sequence *without giving up the definite-integral rendering*:
the canvas-side exchange is **swept per flattened segment through the same
prefix-τ integral as the plain deposit**, so a dynamics stroke has the identical
continuous, dab-free footprint. All on the GPU with no readback
(`gpu/stroke.rs::render_dynamic`, `dynamics.wesl`):

1. **Region composite.** The base tiles under the stroke (the affected set plus a
   one-tile ring) are composited once into a 1:1 canvas **region** texture
   (colour + the wide aux). This is the working canvas the stroke evolves.
   Bounded by `MAX_REGION_DIM`; an oversized stroke degrades to the plain swept
   deposit.
2. **The loop.** The stroke's flattened segments (the same ones the fast path
   sweeps; re-flattened coarser past `MAX_STAMPS`) run *in order* inside a
   **single compute pass** — the implicit barriers between dispatches give the
   sequential semantics, and usage scopes are per-dispatch, so the region can be
   sampled by one dispatch and storage-written by the next with no copies and no
   pass churn. Per-dispatch parameters ride one dynamic-offset uniform buffer.
   - Per segment, **snapshot** (copy the segment quad's region texels into an
     `under` scratch, so the exchange can read-modify-write the region) then
     **deposit** — one thread per footprint texel. A texel's **exposure** to the
     segment is the prefix-τ difference `e(x) = prefix(u) − prefix(u−d)` — the
     brush's coverage integrated along the travel — and exposures add across the
     overlapping quads of consecutive segments, so what the loop applies must be
     built from `e` in a way that survives re-cutting the path. Removal is
     *multiplicative*, `h · exp(λ·e)` with `λ = ln(1 − lift)`, which composes
     exactly: the whole stroke applies `(1−lift)^∫e`, the continuous path
     integral, independent of any spacing — no dabbing. What the dispatch *adds*
     (the tool's deposit, and the brush's own `add` paint) is a source term of
     the same ODE, so it rides the integrating factor `∫₀ᵉ exp(λ·(e−s)) ds` —
     the amount laid during the pass, discounted by the lift still acting on it
     for the rest of the pass. That is the ODE's own solution rather than a
     one-step Euler approximation of it, so `h·A₁+B₁` then `·A₂+B₂` equals the
     single step over `e₁+e₂` exactly. A saturating `1 − exp(λ·e)` in the *added*
     term instead dumps the tool's load into whichever quad reaches a spot first:
     the stroke runs dry early and scallops at the segment spacing — invisible
     under uniform 2px sampling, immediate under adaptive.
   - The loop reads the reservoir at the tip's **mid-pass** position over each
     texel — one sample for a segment during which the tip sweeps a whole range
     of reservoir texels across that spot. That approximation, not the exchange
     math, is what bounds segment length for a dynamics brush: about one
     reservoir texel of travel (`gpu::stroke::flatten_tolerance`). Integrating
     the reservoir along the pass would lift the cap.
   - At `RESERVOIR_CADENCE · radius` cadence, **pickup** — one thread per **tool
     reservoir** texel. The reservoir is a real 2-D texture in brush-local
     coordinates (`BRUSH_RES`², ping-ponged), so each part of the tip carries
     what *it* rolled through. Each texel samples the evolving region under its
     spot with exposure `cov · Δs/r` (its footprint weight × the travel since
     the last pickup — the same exponential law, so depletion matches what the
     interleaved segments lay), lifts canvas height onto the tool, and depletes
     the tool by the upcoming deposits. The reservoir colour thus advances at a
     coarser, cheap cadence while the canvas footprint stays continuous.
3. **Write-back.** Each affected tile's full `TILE_TEX` block is sliced out of
   the shared region into a fresh CoW tile (`slice.wesl`, narrowing the wide aux
   to the persistent `(height, wet)`). Aprons are bit-identical to neighbour
   interiors **by construction** — both are cut from the same texture — and the
   ring in the composite gives rewritten tiles real neighbour content (§6.4; the
   `apron_makes_dynamics_writeback_seamless_under_zoom` regression guards it).

*Conservation (§6.1).* Paint moves by transferring **height** — the one conserved
quantity. Colour and per-unit opacity ride as optical-mass (opacity·height)
weighted blends, and a parcel's blend weight is its own *visible* alpha
(`1 − exp(−K·mass)`, the same translucent-slab law as the media pass), so thick
deposits cover while thin glazes tint. The lift never touches the source's colour
or alpha: the source fades because its **thickness** drops. Both sides of every
transfer integrate the same exponential rate over the same footprint (the canvas
side through the prefix-τ, the reservoir side as `cov · Δs/r` — two quadratures
of the same bilinear form), so with `add = 0` total height (canvas + tool) is
conserved up to resampling error, independent of the pickup cadence.

*Order-dependence is real.* Pickup reads the region as already modified by
earlier segments, so a stroke smears **its own trail** when it crosses it; drag
falls out naturally (`lift` + `deposit` physically carries paint downstream);
and there is no band, column, or stamp structure to alias — the failure modes of
the earlier 1-D per-band reservoir (banded seams, base-only reads, copy-smear)
do not exist in this model.

*The axes* (`BrushDynamics` on `BrushParams` — a flat record in the action log):

- `add` — lay the brush's own paint; the only inexhaustible **source**, and the
  tool's single *amount* knob: the paint height laid per unit swept optical depth.
  A pure-`add` brush takes the swept fast path above, untouched by the loop.
- `lift` — vertical flux canvas → tool (an eraser when alone).
- `deposit` — vertical flux tool → canvas (`lift`+`deposit` with `add = 0` is a
  true mass-conserving smudge).
- `charge` — a finite glob pre-loaded onto the tool (the palette-knife scoop);
  it depletes as the tool deposits and refills as it lifts.

That is the whole set. Earlier drafts also listed `drag`, `bleed`, `ridge`,
`load_pressure` and `deposit_tilt` as inert placeholders awaiting reintroduction
as refinements *of the loop*; they were **removed** rather than carried, because a
serialized field and a UI slider that move but change nothing cost more in
confusion than they save in future typing. Each remains a local change to
reintroduce when it is actually built (the loop already carries per-dispatch
state): a forward deposit offset for the bow-wave drag, a footprint-local blur for
bleed, edge displacement for ridge, per-segment pressure/tilt modulation of the
rates. Likewise `BrushParams` no longer carries `spacing`, `flow`, `height` or
`wetness`: with swept-segment rendering there are no dabs for `spacing` to space
(the reservoir reload cadence is now the fixed `RESERVOIR_CADENCE`), and `flow`
and `height` were redundant multipliers on the one amount `add` already sets —
`flow` doubly so, since it also carried the `drain` factor into `τ` and so applied
the run-dry falloff *twice*. `wetness` was the only source of the gloss channel,
so paint is now uniformly matte; the `(height, wet)` aux layout and the media
pass's roughness term are unchanged and simply read a constant 0.

*Determinism* — a stroke is a pure function of `base` + the `StrokeRecord`
(fixed segment/pickup plan, fixed shader math), so replay and
`preview == committed` hold and the log stays compact: only path + params are
stored, never per-segment data. *Perf* — two footprint-sized dispatches per
segment plus a reservoir-sized one per pickup, inside one pass. A live stroke
re-renders only its tail, resuming the reservoir from the frozen head (see
*Incremental repaint* above), so per-move cost follows the tail rather than the
stroke. What remains is per-segment dispatch overhead: the tail is a few hundred
segments and each costs four small serialized dispatches, which dominates a move.
Batching the independent ones is the next win here. *Paint never dries* — wetness
persists, the whole
canvas stays workable; to glaze over "dry" paint the user adds a **new document
layer**, which composites as if dry, so no drying model is needed.

**Colour dynamics (colour jitter).** The applied colour can vary **across the
brush and along the stroke**: `BrushParams.color_dynamics` (historized — it
changes stored pixels) holds a noise kind plus three per-axis **frequency** and
three per-channel **amplitude** factors. A 3-channel, exactly **tileable 3-D
noise volume** — `White` (per-texel hash) or `Simplex` (a periodic simplex
lattice: gradients hashed from `q = 6·(i,j,k) − (i+j+k)·𝟙 mod 6·P`, which is
invariant under input translation by the period `P`, a multiple of 3) — is baked
**once on the CPU with fixed constants** (`noise.rs`, `Rgba8Snorm` 64³, no
transcendentals ⇒ bit-reproducible across platforms) and sampled with a repeat
sampler. The lookup domain is `(canvas.x·f₀, canvas.y·f₁, arc·f₂)/NOISE_TILE_PX`
plus a per-stroke translation derived from the stroke `seed`: the two canvas
axes vary the colour across the footprint (and, being canvas-anchored, keep the
deposit a pure function of canvas position + the stroke, so tile aprons stay
bit-consistent — §6.4); the **arc-length** axis evolves it along the stroke, and
clamping the arc to each segment's body makes it *continuous across overlapping
segment quads* (a trailing margin's arc equals the next segment's start — no
joint artifacts). The sampled signed noise offsets the brush's **channel triple
in the current colour space** (Oklab `L,a,b`; Mixbox concentrations), applied
per fragment in the sweep stamp (`brush_color`, stamp_common.wesl) and per texel
to the `add` paint in the exchange loop's `deposit` (dynamics.wesl) — both
paths share the field and parameters, so a brush looks the same whichever path
renders it. Amplitude 0 (the default) binds a 1×1×1 zero volume and early-outs —
bit-identical to the constant-colour deposit (all prior goldens unchanged).

`Compositor` blends layers bottom-to-top per dirty tile **in Oklab**, then a
**media pass** turns the height/wet channels into the painterly result: it derives surface
normals from `height`, applies directional lighting so impasto ridges catch the
light, and modulates pigment with thickness/wetness. This media model is a
single shader stage we can iterate on (Kubelka–Munk pigment mixing, granulation,
varnish gloss) without touching the document or tile machinery.

Only **dirty tiles** are recomposited; a per-version damage set (the tiles whose
`Arc` differs from the previously presented version) bounds the work.

### 6.4 Presentation (pan/zoom to a surface)

The engine does **not** own the window surface — the frontend does. The engine
exposes:

```rust
impl Engine {
    pub fn render(&mut self, target: &wgpu::TextureView, view: ViewTransform);
}

pub struct ViewTransform {  // session-owned; pan/zoom never historized
    pub center: Vec2,       // canvas-space point at viewport center
    pub zoom: f32,
    pub viewport: Extent2,  // target size in px
}
```

The `Compositor` (§6.3) walks the tiles intersecting the view AABB, composites
them into a viewport-sized offscreen under the transform, and the media pass blits
the result into `target` — converting **the working channels → the surface's
display space** (e.g. sRGB) in that final pass, the only place gamma-encoded color
exists. (An earlier standalone `Presenter` did a plain color blit; it was retired
once the compositor/media pipeline subsumed it.) For zoomed-out views, tile
**mip/LOD** sampling is a future optimization (v1 samples full-res). The frontend
owns the `wgpu::Surface`, acquires the frame texture, calls `render`, and presents.

**Tile aprons (seamless boundaries).** Tiles are *separate* GPU textures, so the
compositor samples each one independently. The moment sampling isn't pixel-exact
— any sub-pixel pan or non-1:1 zoom — a bilinear tap at a tile's edge clamps to
that tile's own edge texel instead of reaching into the neighbor, because the
neighbor lives in a different texture. That leaves a discontinuity at every tile
boundary, which the media pass (§6.3) then *amplifies*, since the surface normal
is the gradient of the height field and a step in height becomes a bright ridge.

The fix is an **apron**: each tile texture is `TILE_TEX = TILE_SIZE + 2·TILE_APRON`
px square, carrying an `TILE_APRON`-wide halo of the neighboring canvas content
around its interior. Bilinear taps at the interior edge then fall into the apron
(real neighbor data), not a clamp. Mechanics (`geom.rs`, `gpu/stroke.rs`,
`composite.wesl`):

- **The apron is rendered, not copied.** The stamp pass maps the *whole*
  `TILE_TEX` target to NDC (texture origin = interior origin − apron) and a tile
  is selected for (re)drawing whenever a stroke touches its apron-extended bounds
  (`affected_tiles` inflates by `radius + TILE_APRON`). Because stamping at a
  canvas position is a deterministic function of that position, a tile's apron is
  *bit-identical* to the neighbor's interior over their overlap — no copy pass,
  no sync bookkeeping, and it composes correctly through CoW history.
- **Only the interior is presented.** The compositor/present quads still cover
  exactly the interior (tiles tile the plane with no overlap); they sample the
  interior sub-rect via `uv = corner·(TILE_SIZE/TILE_TEX) + APRON/TILE_TEX`, with
  the filter free to read into the apron at the edges.
- **Configurable width.** `TILE_APRON` (1 px — all bilinear needs) is a single
  constant; widen it if a future media effect needs more neighbor context. Cost
  is tiny: at 256² interior, a 1-px apron is ~1.6% more texels.

Alternatives considered and rejected: *composite-then-scale* (composite at 1:1
into one contiguous target, then scale) makes zooming far out balloon that
buffer with the visible tile count; a *padded tile atlas* centralizes the same
idea but is heavier machinery than this problem warrants. The translation
invariance the apron restores is locked by a regression test (`tests/seam.rs`):
a stroke across the 4-tile corner must render identically to the same stroke
shifted half a tile into one tile's interior.

**The canvas surface (tooth & relief).** Paint sits on a physical surface — a
tileable height/bump map (`gpu/surface.rs`), an `R8Unorm` texture sampled in
*canvas* space (so the weave is fixed to the canvas and pans/zooms with it),
shared by the stamp and media passes. It drives two effects:

- **Deposition tooth (stamp pass).** The deposited coverage is gated by the
  surface height at each fragment's canvas position: `cov ·= 1 − tooth·(1−h)·(1−cov)`.
  Light/dry strokes catch on the weave's peaks and skip its valleys; the effect
  fades as coverage builds (valleys fill). `tooth` is a **`BrushParams` field** —
  it changes *stored* pixels, so it's historized for deterministic replay.
- **Surface relief (media pass).** The relief feeds the normal everywhere
  (`height_at` = impasto + `surface_strength·(h−½)`), so the weave catches light
  across the whole viewport — including the bare substrate, whose shading is
  *normalized* so a flat surface leaves it unchanged. `surface_strength` is a
  view setting (`MediaParams`), like the lighting — it doesn't touch stored pixels.

The surface is a **document property** (`SurfaceId { Flat, Linen }` in
`CanvasMeta`, default `Flat`), because deposition depends on it: replay must
reproduce it. `Flat` is a 1×1 *full-height* texel — `h=1` makes tooth a no-op and
a constant height has zero gradient (no relief), so the flat default is *exactly*
equivalent to having no surface. That orthogonality is deliberate: most goldens
use `Flat` to test other features in isolation, and a dedicated golden
(`linen_surface`) exercises the weave. The set is open for future
custom/uploaded surfaces. The engine **embeds no image bytes**: image-backed
surfaces are fetched at runtime and handed to the engine via `register_surface`
(§6.6), which builds the texture (downsampling by an integer factor to fit the
2048 limit, preserving tileability); one bump tile spans `SURFACE_TILE_PX` canvas
px. `Flat` needs no bytes, and a surface with unregistered bytes falls back to it.

### 6.5 Color management (Oklab)

Color flows through exactly three representations, and conversions live in one
module (`color.rs`, with matching WESL helpers):

```
input (sRGB picker / image) ──→ Oklab  (on ingest: BrushParams, imported tiles)
        Oklab  ←──────────────── all storage, mixing, compositing, history
Oklab ──→ display (sRGB/Rec.2020) (only in the media pass's final blit)
```

- **Why Oklab end-to-end:** pigment mixing, gradient interpolation, and wet
  blends are perceptually uniform — no muddy mid-tones from sRGB lerps, no hue
  shifts through gray. This is the math behind the "old masters" blending goal.
- **Determinism:** the sRGB↔Oklab matrices/transfer functions are fixed
  constants shared by Rust and WESL, so ingest and present are reproducible
  across runs and peers — required by golden tests (§9) and convergence (§12).
- **Extensibility:** `CanvasMeta.color_space` records the working space so a
  future wide-gamut or spectral pipeline is a new variant, not a rewrite; the
  display transform is chosen from the surface format at present time.

### 6.6 Brush shapes & the asset store

The default brush is a procedural soft disc, but natural media needs *organic*
tips — worn bristles, chalk, a palette-knife edge. A brush shape is just a
**coverage mask**: a grayscale image where white = full deposit and black = none
(e.g. `crates/stark-ui/assets/shape/WornBristles.png`). The mask drives coverage and, scaled,
the height channel too — so a worn-bristle tip lays down *broken* impasto rather
than a uniform slab.

**Brush shapes are content-addressed assets.** An imported image is identified by
the hash of its bytes; `BrushParams` references that id, never the pixels:

```rust
pub struct AssetId([u8; 32]);   // BLAKE3 of the canonical image bytes

pub enum BrushShape {
    Round,            // procedural soft disc; `hardness` applies
    Stamp(AssetId),   // sampled coverage mask from an imported image
}
// BrushParams gains:  shape: BrushShape, orientation: OrientationSource
```

`orientation` (`FollowStroke` | `Pen`) sets how the swept footprint is angled:
`FollowStroke` keeps the shape's native axis on the stroke tangent (what makes a
bristle brush read as a real stroke rather than a rubber stamp), while `Pen` pins
it to the pen's tilt azimuth in canvas space, like a calligraphy nib. The swept
integral runs along the travel direction, so the shape is pre-rotated into a
per-orientation prefix-τ volume (§6.2) indexed by the relative angle.
Content-addressing is the load-bearing choice, and it keeps every existing
invariant intact:

- **The action log stays tiny.** `StrokeRecord` carries a 32-byte `AssetId`, not
  a 100 KB image; a thousand strokes with one brush reference one blob.
- **Determinism & dedup for free.** Same bytes → same id → same texture, so
  replay, golden tests, and peers resolve identically. And unlike shader drift
  across builds (§8), the brush image is *data the file owns* — shape-driven
  pixels are reproducible across builds, not just within one.
- **Collaboration fits the iroh model.** Content-addressed blobs are exactly
  what iroh blobs sync (§12.4): a peer seeing a stroke that references an unknown
  `AssetId` fetches that blob by hash before rendering it.

**Asset store (`assets.rs` + GPU).** An `AssetStore` maps `AssetId →` a GPU
coverage texture (single-channel `R8`, mip-mapped for clean minification when a
stamp is smaller than the source). On import the image is decoded, normalized to
coverage (alpha if present, else luminance), hashed, uploaded, and cached
(`Engine::import_brush(bytes) -> AssetId`). The store is **document-adjacent
resources**, not the action log: populated on import and on load, bundled into
the save file (§8). Selecting a brush is session state, like color (`SetBrush`),
not a historized edit.

**Stamp rendering.** `stamp.wesl` gains a per-instance rotation (cos/sin) and
samples the bound mask at the footprint's uv, so the mask's coverage is what the
swept optical depth integrates and therefore modulates both opacity and the height
`add` lays. `Round` is realized as a built-in generated mask
under a reserved id, so the shader always samples a texture — one code path.
Determinism holds throughout: fixed sampler, seeded jitter, content-addressed
mask.

**Assets are fetched at runtime, never embedded.** The engine is *given* image
bytes (GOALS §Inputs); it embeds none. Built-in assets (brush shapes, surface
bump maps) live as static files under `stark-ui/assets/` and are bundled by
`asset!` with cache-busting URLs; the frontend fetches them on demand with
`dioxus::asset_resolver::read_asset_bytes` (HTTP on web, filesystem on native)
and hands the bytes to the engine (`import_brush`, `register_surface`). The
built-in bristle brush is fetched once at startup; the large surface maps are
fetched lazily, only when a surface is selected. This keeps multi-megabyte assets
out of the wasm binary — shrinking it and cutting bundle time — and is the path
that scales as the built-in brush/surface libraries grow. (Headless tests, having
no frontend, read the same files from disk and register them directly.)

### 6.7 Pluggable color spaces (Oklab & Mixbox pigment mixing)

The tile channels are **color-space-agnostic**: tools deposit values and only
assume they *blend linearly*, never what color they represent. The meaning —
and the translation to screen — lives behind a trait:

```rust
pub trait ColorSpace {
    fn id(&self) -> ColorSpaceId;            // serialized in CanvasMeta (§8)

    // Tile layout: each space picks its channel textures and how dabs combine.
    fn color_format(&self) -> wgpu::TextureFormat;
    fn aux_format(&self) -> wgpu::TextureFormat;
    fn color_blend(&self) -> wgpu::BlendState;
    fn aux_blend(&self) -> wgpu::BlendState;

    // Picker / export: straight display RGB ↔ the space's channels.
    fn rgb_to_channels(&self, rgb: [f32; 3]) -> [f32; 4];
    fn channels_to_rgb(&self, ch: [f32; 4]) -> [f32; 3];

    // GPU: how a dab writes its channels, and how channels become display color.
    fn stamp_shader(&self) -> &'static str;  // MRT deposit (§6.2)
    fn media_shader(&self) -> &'static str;  // media/lighting + present (§6.3)
}
```

A document has one color space (`CanvasMeta.color_space`), so the tile format,
blend state, and shaders are fixed per document and chosen at engine
construction. The compositing pass A (sample tile → offscreen) is generic; only
the **stamp** and **media** shaders, the formats, and the blends are
space-specific. The CPU `color.rs` Oklab helpers become `OkLabColorSpace`.

**`OkLabColorSpace`** — the current pipeline, unchanged: `color = Rgba16Float`
holding premultiplied `(L, a, b, coverage)`, `aux = Rg16Float (height, wet)`,
premultiplied-"over" color blend (coverage *is* the blend alpha), additive aux.

**`MixboxColorSpace`** — the experimental one: realistic pigment mixing via
**Mixbox** (Secret Weapons), where blue + yellow makes green like real paint
rather than the muddy gray of an RGB blend. Mixbox represents a color as a
*latent* of pigment concentrations `c0..c3` plus a small residual, and mixes by
**linear interpolation in latent space**, then maps latent → RGB through a trained
polynomial. The decisive fit with our architecture: *latents blend linearly*, so
the ordinary premultiplied-"over" deposit **already performs Mixbox mixing** — no
programmable blend, no extra pass. Concretely the tile layout is **identical to
Oklab**: `color = Rgba16Float` holding premultiplied `(c0, c1, c2, coverage)`,
`aux = Rg16Float (height, wet)`, over-blend color + additive aux. The stamp shader
is reused verbatim; only the **media shader differs** — it un-premultiplies the
concentrations and evaluates Mixbox's polynomial (`c3 = 1 − (c0+c1+c2)` derived)
to a base color before the shared impasto lighting.

We **drop Mixbox's latent residual**: a tile has room for three concentrations
plus coverage, and the residual would need a fourth over-blended channel (a third
tile texture). Dropping it keeps zero architecture change and full *mixing*
fidelity; the only cost is slightly approximate reproduction of very saturated
colors (the residual ≈ 0 for in-gamut colors). Recovering it is a future
third-texture option.

Mixbox is **vendored as a git submodule** (`vendor/mixbox`, Mixbox 2.0 ©2022
Secret Weapons, **CC BY-NC 4.0** — non-commercial; commercial use needs a license
from `mixbox@scrtwpns.com`). CPU `rgb_to_channels`/`channels_to_rgb` call the
vendored `mixbox` crate (`no_std` + `libm`, so it builds for wasm and embeds its
own LUT). The GPU polynomial in `media_mixbox.wesl` is **generated at build time**
from the vendored GLSL (`stark-shaders/build.rs` transpiles `mixbox_eval_polynomial`
into a WESL module), so the trained coefficients stay sourced from the licensed
submodule rather than copied into this repo.

### 6.8 Selections — a soft mask, not a shape

A selection restricts where tools may act. The obvious implementation — remember
the rectangle, clip to it — is the one that does not survive contact with the
rest of this design: it cannot express a lasso combined with a rectangle minus an
ellipse, it has no answer for a feathered edge, and it has nothing to say about
the "select by colour" and painted quick-mask producers that will follow. So a
selection here is **not a shape**. It is a *coverage field* — the same sparse tile
map the paint lives in, one `R8Unorm` channel, `TILE_TEX` per tile, aprons and all
(§6.4).

**Representation** (`document/selection.rs`). A `Selection` is a persistent map of
mask tiles plus a single flag: whether canvas *outside* those tiles is selected.
That flag is what makes the infinite canvas work. "No selection" is `outside =
true` with no tiles — free — and so is its inverse, which is why `Invert` is a
constant-cost operation on an unbounded canvas rather than an impossible one. Only
0 and 1 can ever reach it: every combine rule maps `{0,1}²` into `{0,1}`, and the
one shape with non-zero coverage at infinity is `All`.

**Producers and the algebra.** A `SelectionOp` is a shape (`All` / `Rect` /
`Ellipse` / `Lasso`), a mode, and a feather width. Modes are the soft-set
operations, so they degrade to ordinary booleans on hard edges and stay meaningful
on feathered ones:

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
with the edge list uploaded as an `N×1` texture (a decimated polyline — the shader
costs one segment test per texel per vertex).

**Where it applies to the brush.** At the *end* of each stroke path, never by
clipping the footprint:

- The swept fast path masks in the **integrate** pass: `out = mix(base, merged,
  m)` (`integrate.wesl`).
- The brush-dynamics stamp loop masks in **deposit**, lerping its whole
  read-modify-write back toward the pre-segment snapshot, and scales **pickup**'s
  lift by the same coverage (`dynamics.wesl`) — so paint outside the selection is
  neither taken nor laid, and the two sides of the transfer still balance
  (§6.1 conservation).

Masking the *result* rather than the stroke's coverage is the whole point. A
half-covered mask texel must read as half of the finished paint; scaling optical
depth by 0.5 instead would barely fade an opaque brush at all, and a feathered
selection would have a hard edge.

Consumers never branch on whether a mask exists: where the selection has no tile,
a **1×1 texture holding the constant** is bound instead and every read clamps to
the bound texture's own extent. An unmasked document therefore costs one extra
texture fetch and nothing else — which is why the goldens are unchanged.

**Why it lives in `DocState`, and what the log carries.** A stroke's pixels depend
on the mask in force when it was drawn, so replay must be able to reconstruct it:
the selection is document state and edits are logged actions (`Select`,
`InvertSelection`). What travels is the **op**, not the mask — a few floats or a
decimated polyline — and every peer rasterizes it identically from the same
shader. The log stays compact, §12's convergence argument is untouched, and undo
steps through selection changes like anything else. An op that would need more
than `MAX_SELECTION_TILES` masks is rejected (deterministically, so peers agree)
rather than clipped; `All` already expresses "everything" at zero cost.

**Feedback.** A third compositor pass outlines the selection over the lit image
(`overlay.wesl`), one instanced quad per mask tile. The contour is recovered from
the mask itself rather than from the shape that produced it — `(m − 0.5) / |∇m|`
with the gradient taken at one canvas pixel, converted to screen px by the zoom —
so it stays a constant on-screen width at any zoom, stays thin over a feathered
edge, and needs no bookkeeping to survive union/subtract/intersect.

## 7. The engine actor (async backend)

The engine is an actor owning all mutable state, fed by a command channel —
matching GOALS' "asynchronous backend that accepts input commands and exposes
the current state."

```rust
pub struct Engine {
    gpu: GpuContext,            // Device, Queue, capabilities (inputs per GOALS)
    session: Session,           // tool, brush, view, in-flight stroke
    timeline: Box<dyn Timeline>,// Linear (solo) or Replicated (collab) — §5, §12
    actor: ActorId,             // this engine's author id for new actions
    clock: u64,                 // local Lamport counter
    pool: TilePool,
    stroke: StrokeRenderer,
    compositor: Compositor,
    observable: watch::Sender<ObservableState>,  // reactive snapshot for UI
}

impl Engine {
    pub fn new(gpu: GpuContext) -> Self;                  // takes wgpu handles
    pub async fn run(self, rx: mpsc::Receiver<InputCommand>); // event loop
    pub fn render(&mut self, target: &wgpu::TextureView, view: ViewTransform);
    pub fn observe(&self) -> watch::Receiver<ObservableState>;
}
```

`ObservableState` is the cheap, UI-facing projection the frontend renders from —
`can_undo`, `can_redo`, `active_tool`, `brush`, `view`, `doc_bounds`,
`is_stroking`. Published over a `watch`/signal channel so Dioxus re-renders
reactively without polling pixels.

The engine is runtime-agnostic: it uses channels and `async fn run`, so it drops
into tokio (desktop) or wasm-bindgen-futures (web). GPU buffer readback (used by
tests and export) is the only inherently async GPU op and is `await`ed there.

## 8. Save format & timelapse

The native format is **the serialized action log** (GOALS §Outputs):

```rust
pub struct DocumentFile {
    pub format_version: u32,
    pub app_build: BuildId,        // shaders/algorithm version for fidelity notes
    pub canvas: CanvasMeta,        // tile size, channel set, color_space=Oklab
    pub actions: Vec<Action>,      // the full, replayable log (each id-tagged)
    pub assets: Vec<(AssetId, Bytes)>, // content-addressed brush images (§6.6)
    pub checkpoints: Vec<Checkpoint>,  // OPTIONAL cached rasters (see below)
}
```

`assets` bundles every brush image any stroke references (by hash), so the file
stays self-contained and replayable; loading populates the asset store before
replay. Shapes are deduplicated and far smaller than the painted pixels.

Because every `Action` already carries its `ActionId` (actor + lamport), a saved
file is also a valid collaboration log: opening it, painting, and later sharing
it with a peer all use the same records. A solo file simply has a single actor.

- **Load** = replay the actions through `apply` to rebuild `DocState`, then the
  whole undo timeline is immediately available — undo-after-load, for free.
- **Timelapse** = replay actions in order, presenting after each (or each Nth)
  `CommitStroke`. Sample timing comes from `InputSample.time`.
- **Compactness** = a path of samples is far smaller than the painted pixels.
- **Fidelity across builds:** replay determinism holds *within* a build. Because
  shader/algorithm changes could alter pixels across builds, the file records
  `app_build`, and may embed periodic rasterized `Checkpoint` tiles as both a
  fast-open cache and a visual fallback. Strokes remain the source of truth;
  checkpoints are advisory. (`checkpoints` may be empty.)

Serialization uses `serde`; the on-disk container is a versioned binary (e.g.
`postcard` or CBOR) with a magic header. `format_version` gates migrations.

## 9. Testing — golden images

Separating backend from frontend (GOALS §Testing) lets us drive the engine
headlessly:

```rust
// pseudo-test
let gpu = GpuContext::headless();              // offscreen, no surface
let mut engine = Engine::new(gpu);
play(&mut engine, script);                     // a Vec<InputCommand>
let png = engine.export_region(rect);          // readback to RGBA8
assert_golden!("oil_blend_01", png, tolerance);
```

- **Scripts** are command sequences (recorded or hand-written), exercising each
  tool, undo/redo, layer ops, load+replay.
- **Determinism** is engineered in (seeded jitter, fixed flattening tolerances,
  fixed adapter selection, explicit float formats). The comparator uses a small
  perceptual tolerance to absorb legitimate cross-GPU rounding; goldens may be
  keyed by adapter class if needed.
- **Replay equivalence test:** paint a stroke, snapshot; undo then redo;
  serialize → load → snapshot. All three must match — this guards the
  "one rendering path" invariant from §1.3.

## 10. Extensibility map

| Want to add… | Touch only… |
|---|---|
| A new tool / brush behavior | `ToolId` + a `Brush` impl in `gpu/stroke.rs`; serialized in `BrushParams` |
| Image/organic brush shapes | content-addressed `AssetId` in `BrushShape`; `AssetStore` mask textures; stamp shader samples + rotates (§6.6) |
| A new channel (e.g. normal, granulation) | `ChannelSet` descriptor + tile alloc + shader usage; `DocState` unchanged |
| A new document edit | new `ActionKind` variant + its `apply` arm + serde (auto) |
| A new blend mode | `BlendMode` enum + compositor shader branch |
| A new media/lighting model | the media pass shader in `gpu/composite.rs` |
| A different frontend (native, CLI exporter) | new consumer of `Engine`; core untouched |
| Another selection producer (by colour, painted quick-mask, imported alpha) | a `SelectionShape` variant + an arm in `selection.wesl`; the mask representation, ops, history and masking sites are unchanged (§6.8) |
| Transforms, text | new `ActionKind`s + optionally new channels; the action-log model already supports them |
| A wider-gamut / spectral color pipeline | `color.rs` + `CanvasMeta.color_space` variant; storage stays float, present picks the transform |
| Multi-user collaboration | swap `LinearTimeline` → `ReplicatedTimeline`; add `stark-net` (iroh) transport; engine/GPU untouched (§12) |

The action-log + persistent-state core was chosen precisely so these are
*additive*. Nothing above requires changing the history binding, the tile CoW
scheme, or the command/action split.

## 11. Frontend (Dioxus)

`stark-ui` is a Dioxus 0.7 **web** app: the backend runs in WASM and the painting
surface is a dedicated `wgpu::Surface` bound to the page `<canvas>` via **WebGPU**,
which the engine draws into directly. DOM chrome (color palette, brush size,
undo/redo, layer panel) surrounds it.

- UI components dispatch `InputCommand`s; pointer events on the canvas become
  `StartStroke`/`StrokeTo`/`EndStroke`, with element coordinates mapped to canvas
  space via `ViewTransform::screen_to_canvas`.
- Components render from `ObservableState` (held in a Dioxus signal) so toggles
  like undo-availability stay reactive — **no pixel data crosses this boundary.**
- The engine (and its `wgpu::Surface`, both `!Send`) live in a signal; after each
  command the engine renders **straight into the surface texture**
  (`get_current_texture` → `engine.render(view)` → `present`) — no readback, no
  encode. The frontend supplies the GPU handles via `GpuContext::from_parts`
  (GOALS §Inputs); core needs no change to compile to wasm.

Because the engine is frontend-agnostic, this layer stays thin. (An earlier
interim cut ran on Dioxus *desktop* and bridged the canvas by reading the frame
back to a PNG data URL — correct but laggy; the WebGPU surface replaced it,
touching only `stark-ui`.) Run with `dx serve --web -p stark-ui` in a WebGPU
browser. A native winit/desktop surface frontend could reuse the same engine.

## 12. Collaboration (peer-to-peer)

GOALS targets **multi-user editing in a peer-to-peer model** over `iroh` —
**implemented** (build-order step 12) exactly as the additive layer this
section always planned: `ReplicatedTimeline` in `stark-core` (the merge
semantics), `stark-net` (the wire), and a share/join dialog in `stark-ui`.
The engine and GPU code were untouched. Three properties already in place
made it tractable:

1. The document is a **log of id-tagged, deterministic actions** (§4), not mutable
   pixels.
2. Replay is **bit-for-bit deterministic** (seeded brushes, fixed Oklab
   constants) — §6.5, §9.
3. The timeline is behind a **trait** (§5), so a replicated impl drops in.

### 12.1 Convergence model — a CRDT over the action log

We treat the document as a grow-only set of actions with a **total order** given
by `ActionId = (lamport, actor)`. The canonical `DocState` is the deterministic
replay of all actions in that order. Two peers that have seen the same set of
actions compute identical pixels — **strong eventual consistency** — because
ordering is total and replay is deterministic. This is the well-trodden "op-based
CRDT / replicated log" pattern, and it fits Stark almost for free since replay is
already how we derive every pixel.

- **Lamport clocks** give causal-consistent ordering; ties break on `actor` id.
  Every merge advances the local clock past the remote action
  (`Engine::merge_remote`), so an action always orders after everything its
  author had seen — which also guarantees an `Undo` orders after its target.
- **Commutativity isn't required**, only a deterministic order — paint is not
  commutative (later strokes cover earlier ones), and a fixed order captures
  exactly the "whoever's stroke is ordered later wins the overlap" intuition.
- **`Undo` is resolved at the timeline layer, not in `apply`** (§5.4): one
  descending pass over the total order computes which actions are *undone*,
  and the **effective sequence** (non-`Undo`, non-undone, in order) is what the
  `history` cache materializes. Duplicates (gossip redelivery) are rejected by
  id — merging is idempotent.

### 12.2 Inserting a late action (the one real cost)

When a remote action arrives with an id *earlier* than actions already applied
locally (or an `Undo` changes effectiveness mid-log), correctness requires the
canvas reflect the reordered sequence. Because state derives from replay,
`ReplicatedTimeline` diffs the new effective sequence against the materialized
one, pops `history` back to the first divergence, and replays forward. The
untouched prefix keeps its snapshots (and its tiles' `Arc`s) as-is; `history`'s
dense snapshot retention (cheap, per §5) keeps the pops shallow. *Future
optimization:* restrict the replay to the union of the reordered actions'
tile footprints — v1 replays whole actions from the divergence point, which is
correct and scales with how concurrent the editing actually is.

### 12.3 Undo under collaboration

This is why `ActionKind::Undo(target)` exists (§5.4): in a shared log, undo must
be *my* action others can observe and order, and "undo my last stroke" must skip
peers' intervening strokes. The engine asks the timeline first
(`undo_as_action`/`redo_as_action`) and only falls back to navigation undo when
they return `None` (solo). The concrete rules:

- **Undo targets** my most recent *effective* ordinary (non-`Undo`) action.
- **Redo** emits an `Undo` of my most recent effective `Undo` whose target is an
  ordinary action still undone — but only if that `Undo` is newer than my newest
  effective ordinary action, so a fresh edit clears the redo stack, matching
  solo expectations. Chains (Z Z Y Y) walk correctly because each redo
  suppresses exactly one undo.
- A file saved mid-session carries the **full log**; a solo load replays the
  effective sequence (undone work flattens away), while a joining peer gets the
  full log so later redos still resolve.

### 12.4 Transport — `stark-net` over iroh

Core stays **network-agnostic**; `stark-net` adapts iroh (1.0) to the engine's
hooks (`start_collaboration` / `join_collaboration` / `merge_remote` /
`take_outbox`):

- **Identity:** an iroh `EndpointId` (public key) maps to the `ActorId` — its
  first 8 bytes (`actor_from_endpoint_id`; collision odds across a session's
  peers are negligible). No central server. At share time the host's solo
  (`ActorId::SOLO`) actions are rewritten to its real actor — before any peer
  has seen them — so pre-share strokes stay undoable.
- **Live edits:** `iroh-gossip` broadcasts each newly committed `Action`
  (postcard-encoded; small — a fitted path, not pixels) on the session's random
  `TopicId`; received actions are fed into `Engine::merge_remote`. The gossip
  message ceiling is raised (256 KiB) so long strokes fit.
- **Join / catch-up:** a joining peer connects over the `stark/collab/0` ALPN
  and requests a **snapshot** — the save-format container (§8), assets bundled —
  then rides the gossip tail. It joins the topic *before* fetching, so the
  snapshot/gossip overlap covers the seam (dedup by id). Every member serves
  snapshots from a session **mirror** (log + assets, CPU-side), so sessions
  survive the original sharer leaving, and any member can mint a **ticket**
  (`stark…` base32: an `EndpointAddr` + the topic).
- **Assets:** brush-shape images are content-addressed (§6.6); a stroke
  referencing an unknown `AssetId` fetches those bytes over the same ALPN from
  the peer that delivered the stroke (with retries; a miss degrades to the
  round tip rather than blocking the log). The action gossip stays tiny (ids
  only).
- **Browser:** iroh runs in wasm over its relay (WebSocket) transport, so the
  Dioxus UI uses the same code path the native loopback tests exercise. The UI
  glue is two pumps: `dispatch` drains the engine outbox into
  `CollabSession::broadcast`, and a spawned task feeds `RemoteEvent`s into
  `merge_remote`/`import_brush` and repaints. **The page URL is the
  invitation:** a live session's ticket rides the URL fragment
  (`…#stark…`, via `replaceState`; cleared on leave), and opening a link with
  one auto-joins on load — the fragment never leaves the browser, so no server
  sees the ticket.
- **Presence (cursors, selections, names):** still future — ephemeral, broadcast
  over gossip but **never historized** — it's session state, the same category
  as pan/zoom (§3). Other users' live, in-progress strokes would render onto
  preview tiles exactly like the local in-flight stroke, and only become
  `Action`s when their author commits. (Today a peer's stroke appears when
  committed.)

### 12.5 What we deliberately defer

Authentication/permissions (anyone with a ticket can write), presence (§12.4),
large-session scaling (gossip fan-out, log compaction/GC of fully-superseded
tiles), recovery from gossip loss (a lagged receiver warns; a re-join
resnapshots), and offline-merge UX are out of scope for this first cut. None of
them perturb the convergence model above; they layer on top of it.

## 13. Suggested build order

1. **GPU + tiles skeleton:** `GpuContext`, the recycling `TilePool`, and a tile
   blitted to a target under a `ViewTransform`. Proves infinite-canvas pan/zoom
   and the surface contract. (The original standalone `Presenter` was later
   retired once the compositor/media pass subsumed plain blitting.)
2. **Stroke MVP:** color-only stamping along a path; `Session` in-flight stroke;
   `CommitStroke` action; wire `History`. Proves the command/action split and
   CoW.
3. **History + golden harness:** headless context, readback, first golden tests
   incl. undo/redo and replay-equivalence.
4. **Multi-channel + media pass:** add height/wet channels, the one-way load
   (`drain`) reservoir, and normal-from-height lighting — the "old masters"
   payoff. (Bidirectional canvas pickup is its own step, 10.)
5. **Save/load + timelapse:** serialize the action log; load-then-undo; replay
   exporter.
6. **Layers, LOD, and the Dioxus UI** — three largely orthogonal efforts, split
   into substeps:
   - **6a. Layers:** active-layer selection (session state), per-layer opacity /
     visibility / blend (document actions), and per-layer-aware compositing.
     Fully headless-testable.
   - **6b. Dioxus UI:** the `stark-ui` frontend — a wgpu canvas surface, DOM
     chrome, pointer→`InputCommand`, and `ObservableState` on a signal (§11).
     Verification shifts to manual/browser rather than golden tests.
   - **6c. Navigation:** pan (middle-drag) and cursor-anchored zoom (wheel) via
     `ViewTransform::zoom_about` / `Pan`. Navigation feels smooth at current
     scales, so **LOD is descoped to a nice-to-have** (see below) rather than a
     build step.
   Then iterate on color-space fidelity.
7. **Brush shapes & assets (§6.6):** content-addressed `AssetStore`, image
   coverage masks normalized to `R8`, stamps rotated to the path tangent,
   `Engine::import_brush`. Bundle referenced assets in the save file. Golden test
   painting with `crates/stark-ui/assets/shape/WornBristles.png`.
8. **Cubic stroke interpolation (§6.2):** make `StrokeRecord.path` fitted control
   points (RDP at commit, in `path.rs`); stamp generation walks a centripetal
   Catmull–Rom spline. Kills diagonal stair-stepping, makes stamping read
   continuous, and shrinks the action log. Per-stamp GPU interface unchanged;
   preview fits incrementally to stay == committed. Re-bless goldens.
   *Since revised (2026-07):* the fit is **streaming and append-only**
   (`PathFitter`, forward-window RDP) and flattening is **adaptive** — bounded
   error in position, tangent, and pen attributes rather than a fixed arc-length
   step — so a long stroke costs segments proportional to how much it *bends*, not
   to how far it runs. `ControlPoint` / `IntermediateSample` split off from
   `InputSample`, and the swept deposit was made independent of segment count
   (above), without which adaptive sampling changes stroke weight.
9. **Pluggable color spaces (§6.7):** introduce `trait ColorSpace`, migrate the
   current pipeline to `OkLabColorSpace` (behavior-preserving refactor — goldens
   stay green), then add `MixboxColorSpace` — realistic pigment mixing via the
   vendored Mixbox (latent mixes linearly, so the over-blend deposit *is* the mix;
   the media pass evaluates Mixbox's polynomial, generated from the licensed
   submodule at build time). `CanvasMeta.color_space` selects it; golden per space.
10. **Wet mixing & brush dynamics (§6.2) — DONE (rewritten 2026-07):** the
   **sequential swept-exchange loop** — region composite → ordered per-segment
   compute dispatches exchanging height between the evolving region and a 2-D
   tool reservoir, the canvas side swept through the prefix-τ definite integral
   (rates exponential in exposure, so they compose exactly — dab-free) → whole-
   block region write-back. `add`/`lift`/`deposit`/`charge` are the whole axis set
   and all four are live; the never-implemented `drag`/`bleed`/`ridge`/
   `load_pressure`/`deposit_tilt` placeholders were removed (§6.2) and will be
   added back as they are built. Goldens `smudge_drag`/`self_smear` plus the
   conservation/eraser/charge/determinism suite (`tests/dynamics.rs`) and the
   write-back seam regression (`tests/seam.rs`).
11. **Brush file upload:** a `<input type="file">` in the brush panel that reads
   image bytes and calls `Engine::import_brush`, so users can bring arbitrary
   brush shapes — not just built-ins. Pure frontend; the engine/asset/save paths
   from step 7 already accept arbitrary bytes.
12. **Collaboration (§12) — DONE (2026-07-12):** `ReplicatedTimeline` behind
   the existing `Timeline` seam (total-ordered log + effective-sequence
   resolution of `ActionKind::Undo` + rewind/replay merge over the `history`
   cache), engine hooks (`start_collaboration`/`join_collaboration`/
   `merge_remote`/`take_outbox`, undo routed through `undo_as_action`), and
   `stark-net` over iroh 1.0 (gossip live actions; snapshot/asset ALPN;
   tickets). Convergence **is** a test, twice: headless cross-merged engines
   (`stark-core/tests/collab.rs`) and two engines over real loopback iroh
   endpoints (`stark-net/tests/sync.rs`) must render bit-identical canvases.
   UI: "Shared drawing" dialog (share / join-by-ticket / leave) with two pumps
   in `stark-ui/src/collab.rs`. Presence/permissions remain future (§12.5).
13. **Mutable medium — subtractive & wet diffusion (§6.2):** the read-modify-write
   *write-back* path (footprint→scratch → combine → CoW tile), validated by a
   medium-`Dry` equivalence test (Phase 0); then `BrushDynamics::Knife` —
   subtractive palette-knife scraping with conservative reservoir carry, edge
   ridges, and tooth-revealed canvas (Phase 1); then `BrushDynamics::Wet` —
   region-based wet-on-wet diffusion + an optional `Settle` action (Phase 2).
   Single-buffer, always-wet; glazing is left to document layers. Extend the seam
   test to the write-back path; golden per phase.
   *Since unified (§6.2):* the Dry/Knife/Wet enum variants collapsed into **one
   tool** (`add`/`lift`/`deposit`/`charge`), every axis a flux on the single conserved
   quantity (paint `height`) — the integrate is one unified branch. The horizontal-flux
   axes sketched here (drag as conservative finite-volume advection, ridge as a
   zero-mean doublet) are still the intended design when they are built; they are no
   longer carried as inert fields in the meantime (§6.2).

Each step is independently testable through `stark-core` before any UI exists,
which is exactly the leverage the frontend/backend split was meant to provide.
Note that the `Timeline` trait (step 7) should be introduced as the seam *before*
its second implementation exists — cheap now, expensive to retrofit — which is
why §5 already routes the engine through it.

### Nice-to-have (not scheduled)

- **Tile LOD / mipmaps** — sample minified tiles when zoomed far out, for
  responsiveness and to avoid aliasing on huge canvases. Pan/zoom feel smooth
  without it at current scales, so it stays unscheduled until profiling on a
  large document says otherwise.
- **HiDPI** — the web canvas currently uses a 1× drawing buffer (CSS pixels);
  multiply by `devicePixelRatio` for crisp rendering on retina displays.
- **Pen pressure/tilt** — `onpointermove`'s `pressure()` into `InputSample`
  (the brush already varies with it).
