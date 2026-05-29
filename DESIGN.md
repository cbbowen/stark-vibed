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
│   │   │   ├── gpu/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── context.rs   # device/queue wrapper, capabilities
│   │   │   │   ├── tile.rs      # TilePool, CoW tile handles, channel set
│   │   │   │   ├── stroke.rs    # the brush engine / stroke rasterizer
│   │   │   │   ├── composite.rs # layer compositing + media lighting
│   │   │   │   └── present.rs   # canvas → surface (Oklab→display) + pan/zoom
│   │   │   ├── geom.rs          # tile coords, view transform, AABB
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
                         │  Presenter · ShaderModules (WESL)       │
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
    pub brush: BrushParams,       // brush color is Oklab (§6.5)
    pub path: Vec<InputSample>,   // resampled, full fidelity
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

Stroke rasterization is **stamp-based along a resampled path**, with a brush
state machine that carries *loaded paint* so wet-on-wet mixing feels physical:

```
for each stamp s spaced by (brush.spacing × f(pressure)) along path:
    footprint = brush.shape scaled by pressure/tilt at s
    1. PICKUP:  read canvas color/wet under footprint → blend into brush
                reservoir in Oklab, weighted by canvas wetness (bidirectional)
    2. DEPOSIT: write reservoir color (Oklab lerp), add height (impasto),
                add wetness — all modulated by pressure & remaining paint
    seed-driven jitter (scatter, rotation, grain) uses StrokeRecord.seed
```

This is sequential along the path (each stamp depends on the canvas the previous
stamp left), but each stamp's footprint is small and fully GPU-parallel. The
reservoir state lives in a small uniform/storage buffer updated per stamp.
Determinism — identical output for live paint, replay, and tests — is guaranteed
because the only randomness is the explicit `seed`.

**Live vs. replay unification:** `StrokeRenderer` exposes `begin(rec) →
extend(new_samples) → finish()`. Live painting calls `extend` as `StrokeTo`
arrives, stamping only the newly added path segment onto CoW preview tiles.
Replay calls `begin` + one `extend` with the whole path + `finish`. Same code,
same stamps, same pixels.

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

`Presenter` walks the tiles intersecting the view AABB and blits the composited
result into `target` under the transform, converting **Oklab → the surface's
display space** (e.g. sRGB) in this final pass — the only place gamma-encoded
color exists. For zoomed-out views it samples tile **mip/LOD** levels (a future
optimization; v1 may sample full-res) so panning a huge canvas stays responsive.
The frontend owns the `wgpu::Surface`, acquires the frame texture, calls
`render`, and presents.

### 6.5 Color management (Oklab)

Color flows through exactly three representations, and conversions live in one
module (`color.rs`, with matching WESL helpers):

```
input (sRGB picker / image) ──→ Oklab  (on ingest: BrushParams, imported tiles)
        Oklab  ←──────────────── all storage, mixing, compositing, history
Oklab ──→ display (sRGB/Rec.2020) (only in Presenter's final blit)
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
    presenter: Presenter,
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
    pub checkpoints: Vec<Checkpoint>, // OPTIONAL cached rasters (see below)
}
```

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

1. **GPU + tiles skeleton:** `GpuContext`, `TilePool`, one solid-color tile
   rendered through `Presenter` with `ViewTransform`. Proves infinite-canvas
   pan/zoom and the surface contract.
2. **Stroke MVP:** color-only stamping along a path; `Session` in-flight stroke;
   `CommitStroke` action; wire `History`. Proves the command/action split and
   CoW.
3. **History + golden harness:** headless context, readback, first golden tests
   incl. undo/redo and replay-equivalence.
4. **Multi-channel + media pass:** add height/wet, pickup/deposit reservoir,
   normal-from-height lighting — the "old masters" payoff.
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
   - **6c. LOD:** mipmapped tiles for responsive zoomed-out panning. A pure
     optimization; deferred until perf warrants it.
   Then iterate on pigment fidelity.
7. **Collaboration (§12):** introduce the `Timeline` trait split (refactor only,
   no behavior change), then `ReplicatedTimeline`, then `stark-net` over iroh —
   testable headlessly by merging two engines' logs and asserting identical
   golden output (convergence as a test).

Each step is independently testable through `stark-core` before any UI exists,
which is exactly the leverage the frontend/backend split was meant to provide.
Note that the `Timeline` trait (step 7) should be introduced as the seam *before*
its second implementation exists — cheap now, expensive to retrofit — which is
why §5 already routes the engine through it.
