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
5. **Data-driven where it counts.** Channels (color/height/…), tools,
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
├── Cargo.toml                  # workspace (vendor/ excluded — see below)
├── crates/
│   ├── stark-core/             # the engine — no UI, no windowing
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine.rs       # owns everything; process(InputCommand) (§4, §7)
│   │   │   ├── command.rs      # Gesture/Doc/View commands (§4)
│   │   │   ├── session.rs      # view state: tool, brush, view, in-flight gesture
│   │   │   ├── peer.rs         # presence: the roster + wire frames (PEER_DESIGN §4)
│   │   │   ├── error.rs        # EngineError + Result
│   │   │   ├── document/       # versioned state (the history)
│   │   │   │   ├── mod.rs
│   │   │   │   ├── action.rs    # Action + ActionId (replayable mutations)
│   │   │   │   ├── state.rs     # DocState: layers, per-actor selections, surface
│   │   │   │   ├── timeline.rs  # Timeline trait; Linear + Replicated impls
│   │   │   │   ├── selection.rs # Selection soft mask + ops (§6.8)
│   │   │   │   └── layer.rs
│   │   │   ├── color.rs        # Oklab working space, conversions, mixing (§6.5)
│   │   │   ├── colorspace.rs   # ColorSpace trait; Oklab + Mixbox impls (§6.7)
│   │   │   ├── assets.rs       # content-addressed brush/image asset store (§6.6)
│   │   │   ├── noise.rs        # tileable 2-D noise tiles for colour dynamics (§6.2)
│   │   │   ├── image.rs        # RgbaImage (readback / export)
│   │   │   ├── gpu/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── context.rs   # device/queue wrapper, capabilities
│   │   │   │   ├── tile.rs      # TilePool, CoW tile handles, channel set (§6.1)
│   │   │   │   ├── registry.rs  # frontend-provided resources: bytes + live object
│   │   │   │   ├── stroke/      # the brush engine / stroke rasterizer (§6.2)
│   │   │   │   │   ├── mod.rs      # StrokeRenderer, entry points, tip caches
│   │   │   │   │   ├── segments.rs # path → swept segments; region measurements
│   │   │   │   │   ├── swept.rs    # the plain swept fast path
│   │   │   │   │   └── dynamics.rs # the sequential swept-exchange loop
│   │   │   │   ├── composite.rs # compositing + the media/lighting pass (§6.3)
│   │   │   │   ├── environment.rs # HDR environment maps for IBL (§6.3)
│   │   │   │   ├── surface.rs   # canvas surface: the weave's relief (§6.4)
│   │   │   │   ├── selection.rs # selection-mask rasterization (§6.8)
│   │   │   │   └── readback.rs  # GPU→CPU texture readback (export, goldens)
│   │   │   ├── geom.rs         # tile coords, view transform, AABB
│   │   │   ├── path.rs         # streaming B-spline stroke fit + adaptive flatten (§6.2)
│   │   │   ├── spline.rs       # clamped cardinal cubic B-spline + least-squares solve
│   │   │   └── io.rs           # save/load of the action log (§8)
│   │   └── tests/
│   │       └── golden/         # scripted command sequences + reference PNGs (§9)
│   ├── stark-shaders/          # WESL sources + build.rs (wesl link/compile)
│   │   ├── build.rs
│   │   └── src/shaders/*.wesl
│   ├── stark-testdata/         # recorded pen input + asset paths; dev-only (§9)
│   ├── stark-net/              # iroh transport ↔ Replicated timeline (§12)
│   │   └── src/
│   │       ├── session.rs      # CollabSession: the frontend-facing API
│   │       ├── mesh/           # the live broadcast wire
│   │       ├── transport/      # iroh, plus the WebRTC path bootstrap
│   │       ├── mirror.rs       # CPU copy of the log, to serve joiners
│   │       └── ticket.rs       # shareable session tickets
│   └── stark-ui/               # Dioxus 0.7 frontend (§11)
│       ├── assets/             # shipped images + stylesheet (fetched at runtime)
│       └── src/
│           ├── main.rs         # app root, canvas, command rail
│           ├── state.rs        # AppState + the dispatch seam
│           ├── render.rs       # WebGPU surface + Engine wrapper
│           ├── input.rs        # DOM events → InputCommand
│           ├── layout.rs       # floating panel chrome + drag
│           ├── panels/         # one module per tool panel
│           ├── widgets.rs      # shared small controls
│           ├── platform.rs     # the two browser-only helpers
│           ├── brush_editor.rs # the brush dialog + its preview engine
│           └── collab.rs       # session lifecycle glue
├── vendor/                     # third-party, excluded from the workspace
│   ├── mixbox/                 # pigment mixing (submodule, CC BY-NC)
│   ├── iroh/                   # iroh 1.0 + custom-path-opening patch (§12.4)
│   └── iroh-webrtc-transport/ # WebRTC as an iroh custom transport (§12.4)
└── DESIGN.md
```

Rationale: `stark-core` is the testable, frontend-agnostic backend GOALS calls
for. It is also **network-agnostic**: it owns the *merge semantics* of the
action log (the `Timeline` trait) but not the wire transport. `stark-net` adapts
iroh to it (§12) and can be pulled in by the frontend or omitted entirely.
`stark-shaders` is split out so shader compilation (a build step) doesn't pollute
the engine crate and can be reused by tools. `stark-ui` depends on core, never
the reverse.

One caveat, stated rather than hidden: the large image assets (the studio HDR,
the linen weave, the bristle brush — 11 MB together) live in
`crates/stark-ui/assets/`, because Dioxus's `asset!` macro rejects any path
outside its own crate. stark-core's *tests* want those same bytes, so they read
them from there. That is a path pointing the wrong way, and it is confined to one
module — `stark_testdata::assets` — which is the only thing that breaks if the
frontend reorganizes. No code or Cargo dependency crosses that way; a second copy
of 11 MB was the alternative.

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
that never lands in history.

There are **three classes of engine state**, and which one a command touches decides
almost everything about it — whether it is logged, whether peers see it, whether
undo reaches it. So the split is in the type, not in a comment:

- **Document state** — historized, replicated, reproduced by replay. Layers, the
  canvas surface, and of course strokes. The selection is document state too, but
  **owned**: one mask per actor, so a collaborator's lasso never clips your brush
  ([PEER_DESIGN.md](PEER_DESIGN.md) §3).
- **View state** — per-client, transient, never logged *and never sent*. Tool, brush,
  pan/zoom, viewport, lighting. Two people sharing a drawing pan independently.
- **Presence** — per-client and never logged, like view state, but **published**:
  every collaborator reads it and only its owner writes it. The selected layer, the
  cursor, the gesture in flight (PEER_DESIGN.md §4). What puts something here rather
  than in the document is the rule this section already runs on — *does replay need
  it to reproduce pixels?* — and for all of these the answer is no.

A gesture is none of them, and gets its own kind: it *builds* in per-client state and
commits a document action at the end — or nothing at all, if cancelled. In a shared
session the building is published too, so peers watch a stroke as it is drawn.

```rust
pub enum InputCommand {
    Gesture(GestureCommand),
    Doc(DocCommand),
    View(ViewCommand),
    Peer(PeerCommand),
}

pub enum GestureCommand {          // press-drag-release, for brush *and* selection
    Start { tool: Tool, sample: InputSample },
    To    { sample: InputSample },
    End,                           // the one edge that produces document state
    Cancel,                        // produces none
}

pub enum DocCommand {              // each becomes an Action
    Undo, Redo,
    AddLayer { above: Option<LayerId> },
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    MoveLayer { id: LayerId, above: Option<LayerId> },
    Select(SelectionOp),
    InvertSelection,
    SetSurface(SurfaceId),         // which canvas the piece is painted on (§6.4)
}

pub enum ViewCommand {             // never logged, never sent
    SetTool(Tool),
    SetBrush(BrushParams),
    Pan { delta: Vec2 },
    Zoom { anchor: Vec2, factor: f32 },
    Resize(Extent2),
    SetSelectionMode(SelectionMode),
    SetSelectionFeather(f32),
    SetMediaParams(MediaParams),
    SetEnvironment(EnvironmentId),
}

pub enum PeerCommand {             // never logged, but published (PEER_DESIGN §7)
    SetActiveLayer(LayerId),       // collaborators paint on their own, and can see it
    SetCursor(Option<Vec2>),
    SetName(String),
}

pub struct InputSample {       // one pen/mouse sample
    pub pos: Vec2,             // canvas-space position
    pub pressure: f32,
    pub tilt: Vec2,
    pub time: f64,             // for velocity & timelapse
}
```

`Engine::process` takes `impl Into<InputCommand>`, so call sites name the class and
nothing more: `engine.process(ViewCommand::Pan { delta })`.

### Commands vs. requests

Commands are **one-way**: intent goes in, nothing comes back. That is what lets
them become messages over a channel when the engine moves off the UI thread (§7),
and it is the property to protect — a command that returns a value cannot be sent.

Reads therefore go through `Engine::observe()`, which projects *both* classes of
state (see §7). It deliberately includes the view settings a frontend would
otherwise have to read back off the engine — media params, surface, environment,
colour space — because a frontend that cannot observe them keeps its own copy, and
a copy seeded from `Default` goes stale the moment anything else changes them.

What genuinely cannot be a command is a **request**: an operation that must answer.

```rust
// assets — the frontend fetches bytes the engine cannot reach for itself (§6.6)
fn import_brush(&self, png: &[u8]) -> Result<AssetId>;
fn register_surface(&mut self, id: SurfaceId, png: Vec<u8>);
fn register_environment(&mut self, id: EnvironmentId, hdr: Vec<u8>);
// persistence (§8)
fn save_bytes(&self) -> Result<Vec<u8>>;
fn load_bytes(&mut self, bytes: &[u8]) -> Result<()>;
// sampling — the eyedropper (MISSING_FEATURES §0.2). Returns a *future*, because
// readback is the one inherently asynchronous GPU operation (§7); the future owns
// what it needs so the engine borrow ends before the await, as `export` does.
fn pick_color(&mut self, at: Vec2, o: PickOptions) -> impl Future<Output = Option<[f32; 3]>>;
// collaboration transport (§12)
fn merge_remote(&mut self, action: Action) -> bool;
fn take_outbox(&mut self) -> Vec<Action>;
```

These stay direct methods on `Engine`. Under the actor they become
request/response pairs with a reply channel; until then, keeping them *named* as a
tier is what stops them drifting back into ad-hoc setters. **A new engine method
that mutates state and returns nothing is a bug — it should be a command.**

One thing is neither: the **colour space**. Channel layouts differ between spaces,
so changing it cannot preserve a document — every caller asking to "set" it was
really asking for a new document. It is therefore fixed at document creation
(`Engine::new_document(color_space, surface)`) and there is no setter (§6.7).

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
    SetSurface(SurfaceId),      // the canvas the paint went onto (§6.4)
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
Gesture::Start/To   → accumulate into an in-flight StrokeRecord (or SelectionOp),
                      render incrementally onto CoW preview tiles
Gesture::End        → finalize, push Action::CommitStroke (or ::Select) onto History
Gesture::Cancel     → discard preview tiles, no Action
ViewCommand::*      → mutate Session only; nothing logged, nothing sent
DocCommand::Undo    → History::pop / re-derive version (or a logged Undo, §5.4)
DocCommand::*       → commit the corresponding ActionKind
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
    pub opacity: f32,
    pub visible: bool,
    pub content: LayerContent,
}

pub enum LayerContent {
    // sparse map: only populated tiles exist (infinite canvas)
    Paint(rpds::HashTrieMap<TileCoord, TileHandle>),
    // a procedural region + a flat fill — the frame, and later comic gutters
    // and grounds. See FRAME_DESIGN.md.
    Matte { region: MatteRegion, color: [f32; 3] },
}

#[derive(Clone)]
pub struct TileHandle(Arc<GpuTile>);   // Arc bump = the entire "clone" cost
```

Cloning `DocState` clones `rpds`'s persistent collections — internally just bumps a
few `Arc`s (GOALS §dependencies). This is what makes the `history` crate's
snapshot retention affordable: each retained version holds *references* to shared
GPU tiles, not copies. `rpds`'s structural sharing also gives us cheap *diffing*
between two `DocState`s, which is what damage tracking would key off if it
existed (§6.3) and
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
> (per layer in compositing pass A, §6.3). Consequences that the brush dynamics must
> respect:
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
  `KNOT_SPACING` of arc length, plus more wherever taking one on measurably
  reduces the error, by at least `KNOT_COST`. Fitting is what smooths, so there is
  no separate low-pass stage: a polygon far coarser than the jitter averages a
  pixel-quantized staircase away. The arc-length floor is not redundant with the
  sagitta test — it is what makes the polygon grow, and so freezing advance, on a
  stroke the fit is already perfect on.
- **Refit** every sample, but only over the *live* points and the *free* control
  points, so the work per sample follows the tail rather than the stroke so far.
- **Freeze** all but the last few control points. Those are final — nothing drawn
  later can move them — which is what makes the fit append-only and lets a caller
  treat the settled prefix (`frozen_spans`) as already rendered.

Both growth thresholds are denominated in an **input tolerance** the frontend
supplies with `GestureCommand::Start`, in canvas px — the error one as its square,
since it is compared against a mean square. Canvas px are the wrong unit to fix
them in: the same hand movement covers 64× as many zoomed in as zoomed out, and a
pen digitizer, a touchscreen and a mouse each report at a different grain through
the same pointer API. Only the frontend knows either fact, so it states the grain
and the fit becomes invariant to zoom. This is a *fitting* knob and reaches nothing
else — flattening's budget (below) is an error against the curve, in the canvas px
it will actually be drawn in.

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
  is why the price of a control point has to sit *above* the input's own
  quantization — which is what the frontend's declared tolerance is for. Priced
  below it, a staircase reads as curvature and gets traced rather than smoothed.
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

The renderer cuts the path for its own purposes too, on the same argument. A region
is a 1:1 copy of the canvas under the stroke, so a stroke that crosses the document
would want a region the size of the document; instead it is drawn in as many
region-sized **pieces** as it takes, each compositing what the last wrote back
(`gpu::stroke::chunk_segments`). Length therefore costs a dynamics stroke pieces, not
correctness — where it used to degrade past `MAX_REGION_DIM` to the plain swept
deposit, which is not a coarser version of the same brush but a different one: the
swept path only ever *adds* paint, so a brush whose purpose was to lift it stopped
doing the one thing it was for, on exactly the long strokes and fat tips that wanted
it most.

One thing still has to be decided from the record rather than from the piece in hand,
because a live tail and the commit that replaces it must draw the same pixels:
whether the stroke runs the stamp loop at all. It is decided from the **brush alone**
— the strongest form of that guarantee, since there is nothing about the piece, or
about how long the stroke has grown, for it to disagree over — and what it asks is
the floor no subdivision gets under: whether one segment's own footprint fits a
region, since the reservoir pickup reduces over the whole tip at once. See
`gpu::stroke::dynamics_setup`.

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
  height the aux target sums), or `1 − exp(−k·τ)` (a rate — the opacity
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
   Bounded by `MAX_REGION_DIM`, which bounds the transient memory rather than the
   stroke: a stroke too big for one region is cut into pieces that fit, run back to
   back (see incremental repaint above).
2. **The loop.** The stroke's flattened segments (the same ones the fast path
   sweeps, at the same budget) run *in order* inside a
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
   to the persistent `(height)`). Aprons are bit-identical to neighbour
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
the run-dry falloff *twice*. `wetness` was the only source of the **wet channel**,
which is why that channel is gone too: a per-texel `wet` that nothing could ever
write is a stored zero, and every pass that carried it — the stamp, the integrate,
the stamp loop's reservoir and bake packing, the write-back slice — paid for it.
Gloss is now a **uniform property of the paint** instead (§6.3): the media pass
ramps its roughness by the paint's own visible alpha, so paint is glossy wherever
it is and the substrate behind it stays matte, with no channel in between. The
persistent aux is one channel, `(height)`.

*Determinism* — a stroke is a pure function of `base` + the `StrokeRecord`
(fixed segment/pickup plan, fixed shader math), so replay and
`preview == committed` hold and the log stays compact: only path + params are
stored, never per-segment data. *Perf* — two footprint-sized dispatches per
segment plus a reservoir-sized one per pickup, inside one pass. A live stroke
re-renders only its tail, resuming the reservoir from the frozen head (see
*Incremental repaint* above), so per-move cost follows the tail rather than the
stroke. What remains is per-segment dispatch overhead: the tail is a few hundred
segments and each costs four small serialized dispatches, which dominates a move.
Batching the independent ones is the next win here. *Paint never dries* — every
texel stays as workable as the moment it was laid, which is what lets there be no
wetness state at all; to glaze over "dry" paint the user adds a **new document
layer**, which composites as if dry, so no drying model is needed.

**Colour dynamics (colour jitter).** The applied colour can vary **across the
brush and along the stroke**: `BrushParams.color_dynamics` (historized — it
changes stored pixels) holds a noise kind plus two per-axis **frequency** and
three per-channel **amplitude** factors. A 3-channel, exactly **tileable 2-D
noise tile** — `White` (per-texel hash), `Simplex` (a periodic simplex
lattice: gradients hashed from `q = 6·(i,j,k) − (i+j+k)·𝟙 mod 6·P`, which is
invariant under input translation by the period `P`, a multiple of 3; the
lattice stays 3-D and the bake takes its `z = 0` plane, because only `G3 = 1/6`
makes the unskewed lattice positions integral — the 2-D skew constant is
irrational, so a 2-D lattice can be made periodic along its own skewed vectors
but not along the axes a tileable texture needs), or `Voronoi` (Worley F1 on a
jittered grid of `P` cells per side, feature points hashed from the cell index
`mod P`; the usual 3×3 cell search is *exact* here rather than approximate,
because every feature outside that ring is more than one cell away and the
shaping flattens the field past 0.8 cells, so no missed feature can ever show),
or `Mosaic` (the same cells read discretely — one flat value per cell, shared by
all three channels so the facets are whole polygons with hard edges; its owner
search widens to 5×5, since a flat field has no clamp behind which a mis-picked
owner could hide) — is baked **once on the CPU with fixed constants**
(`noise.rs`, `Rgba8Snorm` 64², or 256² for `Mosaic`, whose walls are steps and
so are only as sharp as the tile is fine; only correctly-rounded ops, no
transcendentals ⇒ bit-reproducible across platforms) and sampled with a repeat
sampler.

The lookup domain is **stroke-local**: `(lateral·f₀, arc·f₁)/NOISE_TILE_PX` plus
a per-stroke translation derived from the stroke `seed`, where `lateral` is the
signed offset from the stroke's centreline and `arc` the length along it, both in
canvas px (brush-local y is in radius units, so it is scaled by the radius — the
pattern keeps one scale whatever size the tip is). One axis varies the colour
across the footprint, the other evolves it along the stroke. Anchoring to the
stroke rather than the canvas is what makes the variation belong to the *gesture*
— the same stroke carries the same colour wander wherever it is drawn — and it
costs nothing in determinism: both coordinates are still functions of the
fragment's canvas position and the segment, so the deposit remains a pure
function of the two and tile aprons stay bit-consistent (§6.4). Clamping the arc
to each segment's body makes it *continuous across overlapping segment quads* (a
trailing margin's arc equals the next segment's start — no joint artifacts).

The sampled signed noise offsets the brush's **channel triple in the current
colour space** (Oklab `L,a,b`; Mixbox concentrations), applied per fragment in
the sweep stamp (`brush_color`, stamp_common.wesl) and per texel to the `add`
paint in the exchange loop's `deposit` (dynamics.wesl) — both paths share the
field and parameters, so a brush looks the same whichever path renders it.
Amplitude 0 (the default) binds a 1×1 zero tile and early-outs — bit-identical
to the constant-colour deposit (all prior goldens unchanged).

### 6.3 Compositing and the media pass

Three passes turn tiles into pixels. The first two are the substance; the third
is chrome.

**A — composite.** Every visible tile of every visible layer is drawn, bottom to
top, into two viewport-sized offscreen targets: colour (premultiplied "over", in
the working colour space) and the `(height)` aux (additive). Layer opacity
rides on the instance.

A layer's "over" weight is its **visible alpha** — per-unit opacity and amount
combined by the slab law `1 − exp(−K·opacity·height)`, the same law
`paint_common.wesl` uses to stack parcels *within* a layer — so a layer covers
the stack below exactly as much as it shows. (Weighting by opacity alone was the
old §6.3 defect: a film with opacity 1 and no thickness — every soft brush's
fringe — drew as nothing over bare canvas yet replaced the colour over another
layer's paint.) Because the slab is multiplicative in optical mass, "over" on
these weights accumulates the *stack's* coverage in the target's alpha, and the
media pass reads it there instead of re-deriving it from stack totals; for a
single layer the two are algebraically identical. `tests/composite.rs` guards
the claim.

One consumer must *not* see that weighting: the dynamics loop composites base
tiles into its working region with this same shader, and that region holds the
tile representation itself (per-unit opacity in alpha, §6.1) — the exchange
loop's pickup reads it and the slice writes it back to persistent tiles.
Running the slab law there stores coverage as opacity, corrupting smeared paint
differently on either side of a piece or freeze cut — which is precisely how an
earlier attempt at this fix made smear previews drift from their commits. The
screen path and the region path are therefore separate fragment entry points
(`fs_main` / `fs_raw`, composite.wesl).

A layer whose `BlendMode` is not `Normal` cannot go through that, because its mode
is defined against *what is underneath it*. So pass A is cut into **blend groups**
(`CompositeGroup`): a run of consecutive `Normal` layers is one group and draws
straight into the accumulator — a document that uses no blend modes is a single
group and costs exactly what the flat tile list always did — while every other layer
is a group of its own, composited alone into an isolation target and then merged by
a fullscreen blend pass. That pass reads the accumulator and writes the merge, so it
needs somewhere else to write; rather than copy back, the accumulator ping-pongs
between the caller's target pair and a scratch pair, and the *starting* side is
chosen by the parity of the blend count so the final result always lands where the
caller asked. The media pass therefore keeps one bind group and the eyedropper keeps
its own targets. The scratch pairs are allocated on first use, so an ordinary
painting never pays for them.

The modes themselves are the interesting part, and they are deliberately not
Photoshop's: each is ordinary **addition of light, conjugated by a tone curve** —
`f(a,b) = T(T⁻¹(a) + T⁻¹(b))` — evaluated in CIE XYZ normalized to the display
white, which is the only space in play that is linear in light, non-negative for
every real colour, and free of an opinion about the display's primaries. `Glow`
takes Reinhard's `x/(1+x)`, whose asymptote means no stack of layers can ever clip;
`Radiance` takes Drago's `k·log(1 + x/k)`, which has no asymptote and pushes past
white into pass B's highlight roll-off on purpose. `Multiply` takes `e^{-x}`, which
collapses the conjugation to plain `ab` and makes the added quantity **optical
density** — Beer-Lambert, what stacked glazes do — so the same construction covers
the subtractive side, with white as its neutral element instead of black. Being
conjugations of `+` makes all three commutative and associative, so reordering a
stack of them is not a colour decision. See `document::BlendMode` and
`blend_common.wesl`; each colour
space supplies only its channels ↔ light conversion, which for Mixbox is the pigment
polynomial and its inverse LUT (`mixbox_lut.wesl`), the one place the engine inverts
Mixbox on the GPU.

**B — media / lighting.** One fullscreen pass turns those two buffers into the
painterly result, and it is where the "old masters" look lives:

- **Normals from height.** The gradient of the height field — impasto thickness
  plus the canvas weave scaled by `surface_strength` — tilted by
  `height_strength`, so ridges catch the light.
- **Image-based lighting.** The scene is lit by an [`Environment`](§6.3): an HDR
  decoded to a linear-RGB equirectangular texture with a full mip chain. Diffuse
  irradiance samples a very blurred mip in the *normal* direction; the specular
  samples a gloss-selected mip in the *view-reflection* direction, so paint
  picks up the environment's highlights. Two environments ship: `Neutral`,
  generated procedurally (an achromatic dome under a soft overhead key — relief
  still reads, nothing is tinted), and `Ferndale`, the bundled studio HDR. They
  differ only in the equirect image handed to the same prefilter, so there is one
  lighting path, not two: a reference light you can switch to, and a room you
  paint in. **Exposure is a property of the environment**, not a knob beside it
  (`EnvironmentId::exposure`): `Neutral` is shown at 1.0 and `Ferndale` at 0.65, and
  switching lights carries the value along. See the shoulder discussion below for
  why no single number serves both.

**The reference invariant.** Under `Neutral` (exposure 1.0), with
`height_strength = 0`, the media pass is an identity — paint comes back out its own
colour, within about two bytes. That is what makes the neutral environment worth
having: it is the light you switch to in order to *judge* a colour rather than
enjoy it. Three things have to hold for it, and each is a constraint on the model
rather than a correction bolted onto the end:

- **Exposure is divided by the irradiance a flat canvas actually receives** — the
  diffuse mip sampled dead ahead, computed on the CPU from the same mip chain the
  shader reads. The whole-image mean luminance it replaced only approximated that:
  averaging equirect texels over-weights the poles and counts light no
  front-facing canvas ever sees, which left flat paint ~13% dark.
- **The diffuse keeps `1 - spec_energy`, not `1 - fresnel`.** The split-sum's
  `env_brdf` already integrates Fresnel, so subtracting a second Schlick term from
  the diffuse was double-counting it and losing ~2.4% of every colour.
- **The tonemap is a reference curve, not a look.** Khronos "PBR Neutral", with
  its black point set to the sheen this fragment's BRDF actually contributed
  instead of an assumed F0 = 0.04, and its highlight knee at 1.0 instead of 0.8 so
  nothing representable is reshaped on the way to the display. Only what genuinely
  overflows gets rolled off.

Exactness in `[0,1]` and a filmic shoulder are not both available: a curve that is
the identity up to 1 has nowhere to bend. The shoulder is what was given up, and
`exposure` is what buys the headroom back — which is why it belongs to the *light*.
Dividing by `flat_irradiance` already makes 1.0 mean the same thing everywhere, but
that is a statement about the diffuse response, not about the peaks: a room with
bright windows puts saturated paint over 1.0 and into the clip long before a smooth
grey dome does. So `Neutral` stays at 1.0, where it has to be to be a reference at
all, and `Ferndale` is authored at 0.65 — the value it was judged at.
`tests/reference.rs` pins the invariant.
- **Paint gloss.** `specular` sets how smooth the paint film is, driving a
  Cook–Torrance term. It is a **uniform property of paint**, not a stored channel:
  the roughness ramp is the paint's own *visible alpha* — `1 − exp(−K · opacity ·
  thickness)`, the same quantity the composite-over-substrate uses — so paint is
  equally glossy everywhere it is, a thin glaze reads nearly as matte as the ground
  it barely covers, and the bare canvas behind the paint stays rough, so matte.
  There was once a per-texel `wet` channel here instead; nothing could source it
  after `BrushParams::wetness` was removed (§6.2), so it was a stored zero that
  every pass carried, and it is gone.
- **Present.** The working channels are converted to the surface's display space
  (e.g. sRGB) and composited over the substrate colour. This is the *only* place
  gamma-encoded colour exists.

**C — selection outline.** One instanced quad per mask tile, drawn over the lit
result in the same canvas→NDC frame as pass A (§6.8).

`MediaParams` (`height_strength`, `specular`, `surface_strength`) is a **view
setting** — per-client, never historized, changed by `ViewCommand::SetMediaParams`
(§4). So is the choice of environment: switching it re-lights the canvas and touches
no stored pixel. Exposure is neither: it is not tunable at all, it is what the
chosen environment says it is. Nothing here is in the save file.

The whole media model is a single shader stage, which is the point: Kubelka–Munk
pigment mixing, granulation, varnish gloss can be iterated on without touching the
document or tile machinery.

> **Not yet: damage tracking.** Every populated tile of every visible layer is
> composited on every frame — there is no per-version damage set and no
> view-AABB cull, so off-screen tiles are drawn and clipped by the rasterizer
> rather than skipped. Fine at current canvas sizes; the obvious first
> optimization when it stops being.

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

The `Compositor` (§6.3) composites the visible tiles into a viewport-sized
offscreen under the transform — all of them, not just those intersecting the view
AABB (§6.3) — and the media pass blits
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

**The canvas surface.** Paint sits on a physical surface — a
tileable height/bump map (`gpu/surface.rs`), an `R8Unorm` texture sampled in
*canvas* space (so the weave is fixed to the canvas and pans/zooms with it),
shared by the stamp and media passes. It drives two effects:

- **Deposition tooth — removed, may return.** The idea was to gate deposited
  coverage by the surface height at each fragment, `cov ·= 1 − tooth·(1−h)·(1−cov)`,
  so light strokes catch on the weave's peaks and skip its valleys. It was never
  implemented: `surface_tooth` was a pass-through stub, no stamp shader ever read
  the surface, and the `BrushParams::tooth` field steering it reached a slider that
  moved and changed nothing. All of it — the field, the stub, the stamp-time
  surface bindings — has been deleted rather than left as scaffolding for an idea
  with no implementation behind it. Every golden was unchanged by the removal,
  which is the proof it was inert.

  If it returns it needs a design first (the formula above is a guess, not a
  model), and `BrushParams` would carry a strength again. The surface is already
  document state (§4), so that would be a rendering change, not a history one.

- **Surface relief (media pass).** The relief feeds the normal everywhere
  (`height_at` = impasto + `surface_strength·(h−½)`), so the weave catches light
  across the whole viewport — including the bare substrate, whose shading is
  *normalized* so a flat surface leaves it unchanged. `surface_strength` is a
  view setting (`MediaParams`), like the lighting — it doesn't touch stored pixels.

The surface is **document state** (`SurfaceId { Flat, Linen }`): which canvas a
piece was painted on is part of what the document *is*, it is saved, and reopening
on a different weave would be a different painting. A fresh document starts on
`DEFAULT_SURFACE` = `Linen` — the honest substrate, and the one the stroke pass has
relief to work against — while `SurfaceId::default()` stays `Flat`, since that is
the builtin the registry falls back to before the frontend's bytes arrive.
`CanvasMeta` records the surface the log *starts* from; a mid-document switch is a
logged `ActionKind::SetSurface`, so it undoes, replays and replicates like any
other edit (§4). Today only the media pass reads it, so a switch changes no stored
pixel — logging it anyway is what would let a future deposition gate read it
without that becoming a history change. `Flat` is a 1×1 *full-height* texel — a
constant height has zero gradient (no relief), so it is *exactly* equivalent to
having no surface. That orthogonality is deliberate: most goldens
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
holding premultiplied `(L, a, b, coverage)`, `aux = R16Float (height)`,
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
`aux = R16Float (height)`, over-blend color + additive aux. The stamp shader
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
`InvertSelection`). It is **owned** document state, though — `DocState.selections`
holds one mask per `ActorId`, and `Action::apply` reads the key off `self.id.actor`,
never off the payload. That is what stops one collaborator's lasso clipping another's
brush, and it makes "only its owner may change it" structural rather than a rule a
call site could forget: there is no way to address anyone else's mask
([PEER_DESIGN.md](PEER_DESIGN.md) §3). A document that was never shared has a single
entry under `ActorId::SOLO` and behaves exactly as before. What travels is the **op**,
not the mask — a few floats or a decimated polyline — and every peer rasterizes it
identically from the same shader. The log stays compact, §12's convergence argument is untouched, and undo
steps through selection changes like anything else. An op that would need more
than `MAX_SELECTION_TILES` masks is rejected (deterministically, so peers agree)
rather than clipped; `All` already expresses "everything" at zero cost.

**Feedback.** A third compositor pass outlines the selection over the lit image
(`overlay.wesl`), one instanced quad per mask tile. The contour is recovered from
the mask itself rather than from the shape that produced it — `(m − 0.5) / |∇m|`
with the gradient taken at one canvas pixel, converted to screen px by the zoom —
so it stays a constant on-screen width at any zoom, stays thin over a feathered
edge, and needs no bookkeeping to survive union/subtract/intersect.

**The selection tools are momentary.** `Session::end_selection` hands the canvas
back to `Tool::Brush` the moment a gesture actually encloses something. Selecting
is a step *towards* painting and is essentially never done twice in a row, so a
modal selection tool charges a deliberate switch-back on the overwhelmingly common
path — and when the user forgets, their next brush gesture silently redefines the
selection instead of painting. A gesture that enclosed nothing (a stray click) is
not a selection, so it leaves the tool armed rather than punishing a mis-click.

This is engine-side, not chrome: the session owns `tool`, so every frontend gets
the same behaviour and `observe().tool` reports it in the same update that
committed the op. The frontend then needs no "Paint" tool chip at all — *no chip
lit* is the brush, and clicking the lit chip disarms it, so the control that armed
a tool is the one that takes it back. The two commands that act on a whole
selection (deselect, invert) live in a small floating bar mounted only while a
selection is in force: they are meaningless without one, and a bar that is present
or absent indicates the canvas is masked more directly than permanently-visible
buttons that happen to be greyed out.

## 7. The engine actor (async backend)

> **Status: the target, not the present.** Today `Engine::process` is called
> synchronously from the frontend's event handler, and `observe()` is *pulled*
> after each command rather than pushed over a `watch`. Nothing below is wired up
> yet. It is kept as the design because it is what the command/request split in §4
> is being maintained for: one-way commands are exactly the things that can become
> channel messages, and requests are exactly the ones that will need a reply
> channel. If the actor is ever abandoned, §4's discipline loses its main
> justification and should be revisited rather than quietly kept.

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

The **peer roster** is deliberately *not* in it, even though it is UI-facing
([PEER_DESIGN.md](PEER_DESIGN.md) §4): `ObservableState` is refreshed after every
command and drives the whole component tree, while presence changes thirty times a
second whenever anybody moves. It is read through `Engine::peers()` into a signal of
its own, so a remote cursor moving re-renders a cursor and not an application.

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
- **A missing GPU is a failure, not a skip.** Every GPU test needs an adapter,
  and a skipped test still reports `ok` — so a machine without one would take
  the whole golden / seam / dynamics / selection suite green having rendered
  nothing. Skipping has to be asked for: `STARK_ALLOW_NO_GPU=1`.
- **Goldens are adapter-specific.** A committed PNG can only match the adapter it
  was blessed on, so CI (on software Vulkan) sets `STARK_SKIP_GOLDEN=1`: the
  strokes still render — shader compilation, wgpu validation and panics are all
  still caught — and only the pixel comparison is dropped. Deleting a golden
  re-blesses it on the next run.
- **Recorded input** lives in the dev-only `stark-testdata` crate: real pen
  reports captured from the app, because synthetic curves are smooth and evenly
  sampled in ways real input is not, and the fitter's behaviour turns on exactly
  those details.

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
| A richer frame / comic gutters / a solid ground | a `MatteRegion` variant + an arm in `matte.wesl`; `LayerContent::Matte` and its compositing are unchanged (FRAME_DESIGN.md) |
| Text | a new `ActionKind` + optionally new channels; the action-log model already supports it (transforms landed exactly this way — [TRANSFORM_DESIGN.md](TRANSFORM_DESIGN.md)) |
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

- UI components dispatch `InputCommand`s through one seam, `state::dispatch`,
  which applies, repaints, refreshes `ObservableState`, and broadcasts whatever
  was committed — so no call site has to remember that sequence. Pointer events
  on the canvas become `GestureCommand::Start`/`To`/`End`, with element
  coordinates mapped to canvas space via `ViewTransform::screen_to_canvas`.
  `Start` also carries the **input tolerance** (§6.2): `devicePixelRatio` and the
  event's `pointerType` give the device's grain in CSS px, and the same view
  transform carries it into the canvas px the fit prices against.
- Components render from `ObservableState` (held in a Dioxus signal) so toggles
  like undo-availability stay reactive — **no pixel data crosses this boundary.**
- The floating chrome (panel stack, command rail, selection bar) **fades out while
  the canvas is in hand** — a stroke, a selection drag, a pan, or a run of wheel
  zooming — and fades back the moment the gesture ends. One signal,
  `AppState::canvas_active`, toggles a `dimmed` class the stylesheet animates; the
  chrome keeps its box (nothing reflows) and stops taking clicks while faded, so a
  stroke straying under a panel keeps painting.
- The engine (and its `wgpu::Surface`, both `!Send`) live in a signal; after each
  command the engine renders **straight into the surface texture**
  (`get_current_texture` → `engine.render(view)` → `present`) — no readback, no
  encode. The frontend supplies the GPU handles via `GpuContext::from_parts`
  (GOALS §Inputs); core needs no change to compile to wasm.

The crate is laid out by concern rather than as one file: `state` (the shared
signals and the dispatch seam), `input` (DOM → commands), `layout` (the floating
panel chrome and its drag), `panels/` (one module per tool panel), `widgets`,
`platform` (the two browser-only helpers), plus `render`, `brush_editor` and
`collab`. See §2.

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
untouched prefix keeps its snapshots (and their tiles' `Arc`s) as-is; `history`'s
dense snapshot retention (cheap, per §5) keeps the pops shallow. For an undo
the rewind rarely happens at all: when the undone action *commutes* with what
sits above it, the history shifts it out instead of replaying — see §12.6.

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
- **Redo-at-top:** a revived action re-materializes at the *reviving redo's*
  slot — the top of the stack as of the redo — rather than its original
  position (`revival_keys` in `timeline.rs`: the effective sequence orders by
  id, except that a revived action takes its latest effective revival's id as
  its key). Deliberately "good, not perfect": a redone stroke lands *over*
  work that happened while it was undone rather than back underneath it — and
  in exchange redo is a plain append for every client that is caught up,
  instead of a mid-log insert replaying everything after the original slot.
  Peers converge because the key is a pure function of the shared log; solo
  sessions can't tell the difference (nothing sits above an undo when the redo
  happens, or the redo stack would already be cleared).
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
- **Presence (cursors, selected layer, names, live strokes):** **implemented** —
  see [PEER_DESIGN.md](PEER_DESIGN.md). Ephemeral, broadcast as `Wire::Presence`
  but **never historized, never mirrored and never snapshotted**: nothing in the
  action log refers to it, which is exactly what lets it be dropped, coalesced or
  delayed without touching convergence. Other users' in-progress strokes render
  through the same entry point as the local one and only become `Action`s when
  their author commits. The *selection* is the one piece of per-client state that
  did **not** go here — a stroke's pixels depend on the mask it was drawn through,
  so replay needs it, and it lives in `DocState` keyed by `ActorId` instead
  (PEER_DESIGN.md §3).

### 12.5 What we deliberately defer

Authentication/permissions (anyone with a ticket can write), large-session scaling
(gossip fan-out, log compaction/GC of fully-superseded tiles), recovery from gossip
loss (a lagged receiver warns; a re-join resnapshots), and offline-merge UX are out
of scope for this first cut. None of them perturb the convergence model above; they
layer on top of it. Presence used to be on this list and is now built
([PEER_DESIGN.md](PEER_DESIGN.md)); what that document defers in turn is its §13.

### 12.6 Commutation fast paths — undo and late merges without replay

§12.2's rewind is the honest fallback, but most concurrent edits don't actually
interleave: your undo usually sits under a pile of *other people's* strokes on
other layers or other parts of the canvas. When the changed action **commutes**
with everything materialized after it, the reordered replay would recompute
exactly the pixels already on screen — so the timeline doesn't run it.

The mechanism lives in the `history` crate's `Action` trait, which stark
implements in three pieces:

- **Footprints** (`document/footprint.rs`). Every `ActionKind` declares the
  resources its `apply` reads and writes: a layer's paint within a tile rect
  (a stroke's padded control-point bbox — the B-spline stays in its hull; a
  transform claims the whole layer), layer existence, one per-layer property,
  the stack order (one coarse resource — concurrent reorders genuinely don't
  commute), the author's selection, the surface, the background. Two actions
  commute when no write overlaps the other's read or write set. This encodes
  the intuitive cases structurally: strokes on different layers commute; same-
  layer strokes commute when they share no tile (tile granularity is honest,
  not lazy: removal swaps whole tile handles, so strokes sharing a tile
  genuinely conflict even if their texels don't touch); a rename commutes with
  everything but its own layer's rename/removal; a selection edit blocks only
  its *own author's* later strokes. `Footprint` is the action's
  **`Centralizer`**: the history builds it once per removal and asks it about
  each later action. False conflicts only cost the fast path; a missed one
  would silently diverge peers.
- **`Action::inverse`** (`document/patch.rs`). Removes an action's effect from
  a state by restoring what its footprint wrote from the state it was
  originally applied to — the replaced tile *handles* (`Arc` bumps; tiles are
  copy-on-write, so identity is change detection), the prior prop value, the
  prior selection. Nothing is stored ahead of time and nothing re-renders: the
  history hands `inverse` the true prior state when it needs one. The restore
  is bounded to the footprint, and must be — the two states are not adjacent,
  and the suffix's own work sits outside the footprint on the same layer.
- **`History::try_remove_action_with`** (upstream `history`). Servicing an
  undo, it shifts the target past the run of later actions it commutes with —
  O(log n) cached-state fixes via `inverse`, no re-render at all when the whole
  suffix commutes — and replays only what sits past the first conflict.
  Degradation is graceful: a partially-commuting suffix replays its tail, a
  fully-conflicting one replays like §12.2 always did.

Inserts stay simple on purpose: a fresh commit, a caught-up remote arrival,
and a redo (§12.3's redo-at-top) are all plain appends, and the rare
concurrent arrival that lands mid-sequence takes §12.2's rewind — shallow by
construction, because a concurrent action's Lamport slot is near the top of
the stack.

Convergence is untouched: disjoint footprints mean the shifted materialization
computes bit-identical pixels to the canonical replay — provided every `apply`
touches only what its footprint declares, which is now an invariant `action.rs`
changes must maintain. `TimelineStats` (surfaced as `Engine::timeline_stats`)
counts fast removes vs. rebuilds, because pixels *can't* show which path ran —
`tests/commute.rs` asserts both the stats and exact pixel equality against a
fresh peer's canonical materialization of the same log.

## 13. Suggested build order

Status lives here and nowhere else. It used to be duplicated as a checklist in
`stark-core/src/lib.rs`, at a different granularity, and the two had drifted.

| # | Step | Status |
|---|---|---|
| 1 | GPU + tiles skeleton | done |
| 2 | Stroke MVP (command/action split, CoW tiles) | done |
| 3 | History + golden harness | done |
| 4 | Multi-channel + media pass | done |
| 5 | Save/load + timelapse | done |
| 6a | Layers | done |
| 6b | Dioxus UI | done |
| 6c | Navigation (pan/zoom) | done — tile LOD descoped, see below |
| 7 | Brush shapes & assets (§6.6) | done |
| 8 | Cubic stroke interpolation (§6.2) | done — revised to a streaming, append-only fit with adaptive flattening |
| 8b | Continuous swept-segment stamping (§6.2) | done — one quad per segment, coverage integrated through a prefix-τ texture |
| 8c | Tile aprons (§6.4) | done — killed the lighting seams the media pass amplified |
| 9 | Pluggable colour spaces (§6.7) | done — Oklab + Mixbox |
| 10 | Wet mixing & brush dynamics (§6.2) | done — GPU swept-exchange loop, no CPU readback |
| — | Surface bump maps (§6.4) | done — relief only; the deposition tooth idea was removed unimplemented (§6.4) |
| 11 | Brush file upload | done — custom shape library (import/normalize UI, localStorage persistence, mid-session peer replication) |
| 12 | Collaboration (§12) | done |
| — | Selections (§6.8) | done |
| 13 | Per-client state: owned selections + presence ([PEER_DESIGN.md](PEER_DESIGN.md)) | done — its own build order is PEER_DESIGN §14 |
| — | Transform ([TRANSFORM_DESIGN.md](TRANSFORM_DESIGN.md)) | **done (first cut)** — `ActionKind::Transform`, the parcel/combine/mask GPU passes, exactness + replay tests, and the ellipse gesture UI (inside = move, rim = rotate/scale, outside = stretch/skew; lossless preview through the same renderer); snapping + clipboard remain |

1. **GPU + tiles skeleton:** `GpuContext`, the recycling `TilePool`, and a tile
   blitted to a target under a `ViewTransform`. Proves infinite-canvas pan/zoom
   and the surface contract. (The original standalone `Presenter` was later
   retired once the compositor/media pass subsumed plain blitting.)
2. **Stroke MVP:** color-only stamping along a path; `Session` in-flight stroke;
   `CommitStroke` action; wire `History`. Proves the command/action split and
   CoW.
3. **History + golden harness:** headless context, readback, first golden tests
   incl. undo/redo and replay-equivalence.
4. **Multi-channel + media pass:** add the height channel, the one-way load
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
11. **Brush file upload — DONE (2026-07-29):** users bring their own shape
   images. A shape **library** (`stark-ui/src/shapes.rs`): entries are the
   engine's canonical grayscale PNGs keyed by `AssetId`, persisted per-browser
   in `localStorage` (`identity`-style, graceful without storage), shown as
   thumbnail chips in the Brush panel and a card gallery in the brush editor's
   Tip section (file picker + drag-and-drop). Imports run through the
   *browser's* decoder (`platform::normalize_shape_image`): any displayable
   format, downscaled to the engine cap, and dark-on-light images auto-inverted
   (border-ring heuristic) so scanned ink imports as the ink, not the paper.
   Engine hardening: imports are box-downsampled to `assets::MAX_SHAPE_DIM`
   (1024) before hashing, so an oversized upload can't exceed device texture
   limits (`tests/assets.rs`). Replication hardening (§12.4): a mid-session
   import seeds the session mirror at import time, and a *presence* stroke head
   referencing an unknown shape triggers a detached fetch, so a peer's live
   preview upgrades from the round-tip fallback without waiting for the commit
   (`stark-net/tests/sync.rs::custom_shapes_replicate_mid_session`).
12. **Collaboration (§12) — DONE (2026-07-12):** `ReplicatedTimeline` behind
   the existing `Timeline` seam (total-ordered log + effective-sequence
   resolution of `ActionKind::Undo` + rewind/replay merge over the `history`
   cache), engine hooks (`start_collaboration`/`join_collaboration`/
   `merge_remote`/`take_outbox`, undo routed through `undo_as_action`), and
   `stark-net` over iroh 1.0 (gossip live actions; snapshot/asset ALPN;
   tickets). Convergence **is** a test, twice: headless cross-merged engines
   (`stark-core/tests/collab.rs`) and two engines over real loopback iroh
   endpoints (`stark-net/tests/sync.rs`) must render bit-identical canvases.
   UI: a "Share" dialog (one click starts the session and shows the link to copy;
   leaving stops it) with two pumps in `stark-ui/src/collab.rs`. Joining has no UI —
   the ticket rides the page URL's fragment and opening such a link joins on load.
   Permissions remain future (§12.5); presence became step 13.
13. **Per-client state ([PEER_DESIGN.md](PEER_DESIGN.md)):** the third class —
   owned by one client, read by all. Two mechanisms, split by whether replay needs
   the state: the **selection** moves into `DocState` keyed by `ActorId`, so a
   collaborator's mask stops clipping your brush and every peer still reproduces
   your strokes through *your* mask; the selected layer, cursor and in-flight
   gesture become **presence**, a roster outside the timeline fed by a lossy
   `Wire::Presence` channel that never touches the mirror, the snapshot or the
   file. Live strokes ride the fitter's frozen prefix as deltas with a ~1 Hz
   resync, and every client folds them over the committed document in `ActorId`
   order. Also fixes two latent defects: concurrent `AddLayer` minting the same
   `LayerId` (a real convergence failure), and a remote `RemoveLayer` stranding the
   active layer. Tests: `stark-core/tests/peer_state.rs`,
   `stark-net/tests/presence.rs`.
14. **Mutable medium — subtractive & wet diffusion (§6.2):** the read-modify-write
   *write-back* path (footprint→scratch → combine → CoW tile), validated by a
   medium-`Dry` equivalence test (Phase 0); then `BrushDynamics::Knife` —
   subtractive palette-knife scraping with conservative reservoir carry and edge
   ridges (Phase 1) — "tooth-revealed canvas" would first need the deposition gate,
   which was removed unimplemented (§6.4); then `BrushDynamics::Wet` —
   region-based wet-on-wet diffusion + an optional `Settle` action (Phase 2).
   Single-buffer, always-wet; glazing is left to document layers. Extend the seam
   test to the write-back path; golden per phase.
   *Since unified (§6.2):* the Dry/Knife/Wet enum variants collapsed into **one
   tool** (`add`/`lift`/`deposit`/`charge`), every axis a flux on the single conserved
   quantity (paint `height`) — the integrate is one unified branch. The horizontal-flux
   axes sketched here (drag as conservative finite-volume advection, ridge as a
   zero-mean doublet) are still the intended design when they are built; they are no
   longer carried as inert fields in the meantime (§6.2). Nor is the `wet` *channel*:
   a real diffusion model would reintroduce it as a second aux component, which is a
   format change (`R16Float` → `Rg16Float`) plus the passes that carry it — cheap to
   redo, and cheaper than storing a zero until then (§6.3).

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
