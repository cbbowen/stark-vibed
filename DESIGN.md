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
│   │   │   ├── path.rs          # stroke fitting (RDP) + Catmull–Rom flatten (§6.2)
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
    pub path: Vec<InputSample>,   // cubic-spline control points, fitted (§6.2)
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
    fn push(&mut self, action: Action, ctx: &mut ApplyCtx) -> Result<()>;
    fn current(&self) -> &DocState;
    fn undo(&mut self, ctx: &mut ApplyCtx) -> Result<()>;
    fn redo(&mut self, ctx: &mut ApplyCtx) -> Result<()>;
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
  and must mean "undo *my* action" not "undo whatever happened last." Applying it
  re-derives the state as if `target` were absent (`replay_without`). Single-user
  mode never needs to emit this, but the variant exists from the start so the log
  format is collaboration-stable.

## 6. Rendering the canvas (infinite, tiled, multi-channel)

### 6.1 Tiles and channels

A tile is a fixed `TILE_SIZE` (256×256) square in canvas space, addressed by
integer `TileCoord(i32, i32)`. Sparsity gives the infinite canvas: only painted
tiles allocate. Each tile is **multi-channel** — this is what enables strokes
that affect more than color (GOALS §1):

```rust
pub struct GpuTile {
    pub color:  wgpu::Texture,   // Rgba16Float — Oklab (L,a,b) + premult alpha
    pub height: wgpu::Texture,   // R16Float — paint thickness / impasto
    pub wet:    wgpu::Texture,   // R16Float — wetness for wet-on-wet mixing
    // future channels (normal, granulation, …) are additive here
}
```

The color texture stores **Oklab** components, not sRGB/RGB. Linear 16-bit float
comfortably holds Oklab's range and the negative `a`/`b` chroma axes, and keeps
blends perceptually uniform (GOALS §1). Alpha is premultiplied against `L,a,b`.

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

**Path representation & cubic interpolation.** A `StrokeRecord`'s `path` is not
the raw pointer samples but a compact set of **control points** fitted from them
(Ramer–Douglas–Peucker simplification at commit, in `path.rs`). Rendering
expands those through a **centripetal Catmull–Rom spline** into a fine polyline,
then walks it at even arc length (above). This one change solves three problems:

- **No stair-step aliasing** — jittery pixel-stepped input (a diagonal drawn as
  1-px right / 1-px up steps) collapses to a smooth curve instead of axis-aligned
  segments.
- **Continuous-looking stamping** — stamps ride a smooth path with smooth
  tangents, so even hard-edged tips read as one stroke rather than a row of
  discrete dabs (an approximation of a path integral over the stroke).
- **Smaller files** — a handful of control points replace hundreds of raw
  samples in the action log (§8).

The per-stamp GPU instance is **unchanged**; only stroke→stamp generation
differs. Live preview fits incrementally (re-fitting the in-flight samples each
update), so the preview at release equals the committed stroke — preserving the
live == committed invariant (§1.3). Fitting and Catmull–Rom evaluation are fixed
float math, so determinism (and golden/replay/save-load equivalence) holds.

**Continuous stamping (swept segments).** Discrete dabs are still visible with
hard tips. The fix: stamp each short *segment* of the flattened curve as one
quad whose coverage is the brush **swept** along it — the path integral of the
footprint, instead of point samples. The enabling identity: alpha-"over" is
multiplicative in `(1−α)`, hence additive in **optical depth** `τ = −ln(1−α)`.
So:

- Precompute, per brush, the **prefix integral of `τ` along the travel axis**
  (the tangent the brush is rotated to). A length-`d` segment's swept depth at a
  point is then `prefix(u) − prefix(u−d)` for that row — an O(1) lookup.
- A segment quad outputs `α_seg = 1 − exp(−flow · sweptDepth)`. Because the
  existing premultiplied-"over" blend across overlapping segment quads combines
  as `1 − ∏(1−α) = 1 − exp(−Σ τ)`, it sums the depths **exactly** — no
  double-counting at joints, no scratch buffer, no second pass. The whole
  stroke's coverage is the continuous path integral `1 − exp(−τ_total)`.

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

**Wet mixing & brush dynamics.** To smear paint already on the canvas — the core
of a natural-media feel — the brush picks up wet pigment under it, carries it, and
lays down an evolving mix downstream. This is **sequential and order-dependent**
(what's under the brush includes what it deposited a moment ago), which is exactly
what the swept-segment model is *not*. The reconciliation rests on two facts:

1. `SegmentInstance` already carries **per-segment** `ch[4]` (color) and `aux[2]`
   (height, wet). So smearing is just *"compute a different color per segment"* —
   the sweep shader and the parallel render path are **untouched**.
2. The sequential part collapses to a tiny **1-D reservoir recurrence** over the
   flattened path (a few floats of state, run once per stroke):

   ```
   reservoir = { ch: [f32;4], load: f32 }            // ch = the color space's
                                                      //   channels (Oklab L,a,b
                                                      //   or pigment loads §6.7);
                                                      //   extensible: + wet, height
   for each step i at footprint p_i along the path:
       (c, a, w) = sample base canvas under p_i       // active layer, in ch space
       PICKUP : lift = pickup · w · a                 // only wet, present paint lifts
                ch   = mass_weighted_mix(ch, load, c, lift)
                load = min(capacity, load + lift)
       DEPOSIT: segment.ch  = ch                      // → fed to the sweep
                segment.cov = flow · f(pressure) · g(load)
                load -= deposited                      // refills toward brush color
                                                       //   for a mixer; pure smudge
                                                       //   injects none
   ```

   Mixing happens in the color space's channels, so it composes with both spaces:
   Oklab gives a perceptually-uniform blend, and Mixbox channels are pigment
   latent concentrations whose linear mix *is* Mixbox pigment mixing (§6.7).

*Pluggable axis.* Brush dynamics are a serde **enum** on `BrushParams` (it lives
in the action log, so not a trait object):

```rust
enum BrushDynamics {
    Dry,             // one-way `drain` load — the DEFAULT, so existing strokes,
                     //   goldens, and saved files are unchanged
    Mixer(MixerParams),
    // reserved, designed-for but not built:
    // Bleed(BleedParams),   // Mixer + short-range wet diffusion at edges
    // Fluid(FluidParams),   // Eulerian advect+inject micro-sim per stroke
}
```

The renderer dispatches on it; the enum is the seam where higher-fidelity tiers
plug in. Pure smudge vs. mixer is a parameter (own-color injection), not a code
path.

*Determinism via per-stroke canvas read.* PICKUP samples **`base`** — the
pre-stroke committed canvas that `Action::apply`/`render` already hold. Two
properties fall out of reading `base` (not the evolving canvas):

- **Self-pickup is implicit** — paint the stroke just laid is still in the
  reservoir, so the carry needs no extra read.
- **Compact, replay-deterministic log** — only path + brush params are stored;
  per-segment colors are *recomputed* at apply time from the replayed `base` (same
  machine → identical). They are **never** baked into the log.

The accepted approximation: a stroke crossing its **own earlier** trail won't
smear it (`base` doesn't contain it). That boundary is where the future
persistent-wet-field / advection tiers take over.

*Implementation — fully on the GPU, no readback.* A CPU read of `base` is
impossible on the web: WebGPU cannot block for a buffer map (`getMappedRange`
panics), and a synchronous readback on the interactive path is off the table. So
pickup runs entirely on the GPU, chained into the stroke's command encoder before
the deposit (`gpu/stroke.rs::encode_mixer`):
1. **Region composite** — re-composite the active layer's `base` tiles that
   overlap the stroke's bbox into one 1:1 `color`+`aux` region texture (the
   `composite` shader with a region transform). One bindable "canvas under the
   stroke"; bounded by `MAX_REGION_DIM` (oversized strokes skip pickup).
2. **Reservoir scan** — a single-workgroup serial compute pass (`mixer.wesl`)
   walks the segments in order, samples the region (`textureLoad`, nearest),
   decodes per color space, runs the recurrence, and writes each segment's color
   into the instance buffer (`STORAGE | VERTEX`), patching the `ch` vertex
   attribute in place — so the sweep shaders are untouched.
3. **Deposit** — the existing swept render reads the patched colors.

`SegmentInstance` is padded to 64 B so the same buffer is a valid `std430`
`array<Instance>` for the compute pass. The native golden exercises this exact
path, so it validates the GPU pipeline without a browser.

*Compositing note.* Varying per-segment color is safe for the sweep: alpha still
sums exactly (`1 − ∏(1−α)`), and color in the heavy overlap between consecutive
segments becomes a coverage-weighted blend biased toward the most-recent (topmost)
segment — the correct direction for a forward smear. A golden (smear color X over
a committed patch of Y → trail transitions Y→X) guards this.

*Paint never dries.* Wetness persists, so the whole canvas stays workable like a
continuous oil session; a drying model (time/age decay or an explicit dry action)
is a future option, not built.

*Known limitation — non-conservative lift (copy-smear).* v1 pickup is
**non-destructive**: it copies canvas color into the reservoir but does **not**
remove paint from the source, so the source never lightens and total pigment is
not conserved (a long smear can "duplicate" color rather than drag a finite amount
of it). True mass-conserving lift requires the stroke to also *reduce* `base`
coverage where it lifts — i.e. write the source tiles — which the single-pass
additive/over deposit cannot express. Deferred; acceptable because copy-smear
reads as smearing in the overwhelming majority of cases.

### 6.3 Compositing & the "old masters" look

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

The surface is a **document property** (`SurfaceId { Flat, Canvas }` in
`CanvasMeta`, default `Flat`), because deposition depends on it: replay must
reproduce it. `Flat` is a 1×1 *full-height* texel — `h=1` makes tooth a no-op and
a constant height has zero gradient (no relief), so the flat default is *exactly*
equivalent to having no surface. That orthogonality is deliberate: most goldens
use `Flat` to test other features in isolation, and a dedicated golden
(`canvas_surface`) exercises the linen weave. The set is open for future
custom/uploaded surfaces. (The built-in linen is downsampled by an integer factor
to fit the 2048 texture limit, which preserves tileability; one bump tile spans
`SURFACE_TILE_PX` canvas px.)

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
(e.g. `resources/shape/WornBristles.png`). The mask drives coverage and, scaled,
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
// BrushParams gains:  shape: BrushShape, follow_path: bool, angle_jitter: f32
```

`follow_path`/`angle_jitter` rotate each stamp to the stroke tangent (with
seeded jitter), which is what makes a bristle brush read as a real stroke rather
than a rubber stamp. Content-addressing is the load-bearing choice, and it keeps
every existing invariant intact:

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
samples the bound mask at the footprint's uv: `coverage = mask · flow`, with the
mask also modulating height. `Round` is realized as a built-in generated mask
under a reserved id, so the shader always samples a texture — one code path.
Determinism holds throughout: fixed sampler, seeded jitter, content-addressed
mask.

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
- **Determinism** is engineered in (seeded jitter, fixed resample step, fixed
  adapter selection, explicit float formats). The comparator uses a small
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
| Selections, masks, transforms, text | new `ActionKind`s + optionally new channels; the action-log model already supports them |
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

## 12. Collaboration (peer-to-peer, future)

GOALS targets long-term **multi-user editing in a peer-to-peer model** over
`iroh`. This is explicitly future work, but the core is shaped now so it arrives
as an additive layer, never a rewrite. Three properties already in place make it
tractable:

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
- **Commutativity isn't required**, only a deterministic order — paint is not
  commutative (later strokes cover earlier ones), and a fixed order captures
  exactly the "whoever's stroke is ordered later wins the overlap" intuition.
- **Layers and tiles localize conflicts.** Concurrent strokes on different layers
  or different tiles never interact; only same-tile overlaps depend on order, and
  there the resolution is simply the total order. `im`'s structural diffing
  (§5.1) lets a peer reapply only the tiles a late-arriving remote action
  actually touches.

### 12.2 Inserting a late action (the one real cost)

When a remote action arrives with a `lamport` *earlier* than actions already
applied locally, correctness requires the affected tiles reflect the reordered
sequence. Because state derives from replay, the `ReplicatedTimeline` rewinds to
the insertion point (a retained `history` snapshot at or before it) and replays
forward — but only for the **tiles in the union of footprints** from that point
on. Unaffected tiles keep their existing `Arc`s untouched. Cost scales with
concurrent-edit overlap, not document size. Frequent, dense `history` checkpoints
(cheap, per §5) keep the rewind shallow.

### 12.3 Undo under collaboration

This is why `ActionKind::Undo(target)` exists (§5.4): in a shared log, undo must
be *my* action others can observe and order, and "undo my last stroke" must skip
peers' intervening strokes. `Undo(target)` logs the intent; replay derives state
as if `target` (and any already-undone-by-it) were absent. Redo is an `Undo` of
an `Undo`. Local solo editing keeps using the fast `history`-navigation path and
emits none of these.

### 12.4 Transport — `stark-net` over iroh

Core stays **network-agnostic**; `stark-net` adapts iroh to the `Timeline`:

- **Identity:** an iroh `NodeId` *is* the `ActorId`. No central server.
- **Live edits:** `iroh-gossip` broadcasts each newly committed `Action` (small —
  a sampled path, not pixels) to the session's peers; received actions are fed
  into `ReplicatedTimeline::merge`.
- **Join / catch-up:** a joining peer pulls the action log (and optionally a
  checkpoint blob to skip cold replay) via **iroh blobs / docs**, then subscribes
  to gossip for the live tail. The save format (§8) *is* this payload.
- **Assets:** brush-shape images are content-addressed blobs (§6.6); a stroke
  referencing an unknown `AssetId` fetches that blob by hash before rendering —
  exactly what iroh blobs are for. The action gossip stays tiny (ids only).
- **Presence (cursors, selections, names):** ephemeral, broadcast over gossip but
  **never historized** — it's session state, the same category as pan/zoom (§3).
  Other users' live, in-progress strokes render onto preview tiles exactly like
  the local in-flight stroke, and only become `Action`s when their author commits.

### 12.5 What we deliberately defer

Authentication/permissions, large-session scaling (gossip fan-out, log
compaction/GC of fully-superseded tiles), and offline-merge UX are out of scope
for the first collaboration cut. None of them perturb the convergence model
above; they layer on top of it.

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
   painting with `resources/shape/WornBristles.png`.
8. **Cubic stroke interpolation (§6.2):** make `StrokeRecord.path` fitted control
   points (RDP at commit, in `path.rs`); stamp generation walks a centripetal
   Catmull–Rom spline. Kills diagonal stair-stepping, makes stamping read
   continuous, and shrinks the action log. Per-stamp GPU interface unchanged;
   preview fits incrementally to stay == committed. Re-bless goldens.
9. **Pluggable color spaces (§6.7):** introduce `trait ColorSpace`, migrate the
   current pipeline to `OkLabColorSpace` (behavior-preserving refactor — goldens
   stay green), then add `MixboxColorSpace` — realistic pigment mixing via the
   vendored Mixbox (latent mixes linearly, so the over-blend deposit *is* the mix;
   the media pass evaluates Mixbox's polynomial, generated from the licensed
   submodule at build time). `CanvasMeta.color_space` selects it; golden per space.
10. **Wet mixing & brush dynamics (§6.2):** the `BrushDynamics` serde enum
   (default `Dry`, then `Mixer`) on `BrushParams`; a per-stroke **reservoir
   recurrence** that reads the pre-stroke `base` canvas and emits **per-segment
   color** (the swept renderer is unchanged). Runs **entirely on the GPU** (region
   composite + serial compute scan, no readback — WebGPU can't block for one).
   Golden: smear a brush over a committed patch and
   verify the trail transitions. Leaves the enum + canvas-probe seams for the
   future `Bleed` (edge diffusion) and `Fluid` (advection) tiers. Copy-smear only —
   non-conservative lift is a documented deferral.
11. **Brush file upload:** a `<input type="file">` in the brush panel that reads
   image bytes and calls `Engine::import_brush`, so users can bring arbitrary
   brush shapes — not just built-ins. Pure frontend; the engine/asset/save paths
   from step 7 already accept arbitrary bytes.
12. **Collaboration (§12):** introduce the `Timeline` trait split (refactor only,
   no behavior change), then `ReplicatedTimeline`, then `stark-net` over iroh —
   testable headlessly by merging two engines' logs and asserting identical
   golden output (convergence as a test).

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
