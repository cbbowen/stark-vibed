# Architecture

Principles, crate layout, the command/action boundary, and the history model — §1–§5.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 1. Guiding principles

1. **The document is a list of actions, not a bag of pixels.** Pixels are a
   *derived, cached* view of a replayable action log. This one decision delivers
   the native save format, undo after load, and timelapse at once.
2. **Cheap state, expensive pixels — keep them separate.** The `history` crate
   wants `State` values it can clone and retain in O(log n) snapshots, so `State`
   is a *persistent, structurally-shared map of tile handles*, never raw pixels.
   Cloning a document state is a handful of `Arc` bumps; GPU memory is shared
   across versions and reclaimed by refcount.
3. **One rendering path, used three ways.** The same deterministic stroke
   renderer drives live painting, history replay (undo/redo, load), and golden
   tests. If those diverge, tests lie — so replay is *the* path and live painting
   is an incremental front-end to it. **`preview == committed`** is the invariant
   that states it.
4. **Frontend-agnostic core.** The engine knows nothing about Dioxus, windows or
   event loops. It consumes `InputCommand`s and GPU handles and exposes state
   plus a render entry point. Dioxus is one consumer; headless golden tests are
   another.
5. **Data-driven where it counts.** Channels, tools, actions and blend modes are
   open sets behind small traits/enums, so new capabilities are additive.
6. **Perceptual colour is the working space.** Colour stores and blends in
   **Oklab** (or Mixbox pigment latents, §6.7), so mixing, compositing and
   gradients are perceptually uniform; conversion to a display space happens only
   at the final present. Colour math never touches gamma-encoded sRGB.
7. **Convergence-ready from day one.** Every action carries a globally-unique id
   and the document derives purely from a *deterministically ordered* replay.
   Solo that is a linear timeline; multi-user it is a replicated log all peers
   replay to the same pixels. The determinism that makes golden tests work makes
   collaboration converge (§12).

Two habits that recur throughout and are worth naming as principles:

- **Rule out a class rather than enumerate its instances.** Where a guarantee can
  be made structural — ownership derived from the action id (§17.3), a
  representation that cannot express the wrong thing (§14.2) — it is, instead of
  a check a call site could forget.
- **Nothing inert ships.** Fields, sliders and shader stubs that move but change
  nothing were deleted rather than carried as scaffolding (`tooth`, `drag`,
  `bleed`, `wetness`, the `wet` channel, `TernaryPad`). Each is a local change to
  reintroduce when it is actually built; a serialized field that does nothing
  costs more in confusion than it saves in future typing. Two of them have since
  come back that way — `bleed` as the lateral-diffusion axis (§6.2) and `tooth`
  as the deposition gate (§6.4), each with the model the placeholder never had,
  which is the argument working rather than an exception to it.

## 2. Crate / workspace layout

```
stark/
├── Cargo.toml                  # workspace (vendor/ excluded — see below)
├── crates/
│   ├── stark-core/             # the engine — no UI, no windowing
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine.rs       # owns everything; process(InputCommand) (§4, §7)
│   │   │   ├── command.rs      # Gesture/Doc/View/Peer commands (§4)
│   │   │   ├── session.rs      # view state: tool, brush, view, in-flight gesture
│   │   │   ├── peer.rs         # presence: the roster + wire frames (§17.4)
│   │   │   ├── presence.rs     # the publish latch (§17.5)
│   │   │   ├── error.rs        # EngineError + Result
│   │   │   ├── document/       # versioned state (the history)
│   │   │   │   ├── action.rs    # Action + ActionId (replayable mutations)
│   │   │   │   ├── state.rs     # DocState: layers, per-actor selections, surface
│   │   │   │   ├── timeline.rs  # Timeline trait; Linear + Replicated impls
│   │   │   │   ├── selection.rs # Selection soft mask + ops (§6.8)
│   │   │   │   ├── fill.rs      # FillOp + ShapeAction: what a shape does (§6.8)
│   │   │   │   ├── transform.rs # affine transform planning (§16)
│   │   │   │   ├── footprint.rs # what an action reads/writes (§12.6)
│   │   │   │   ├── patch.rs     # Action::inverse (§12.6)
│   │   │   │   └── layer.rs     # Layer, LayerContent, carries (§14)
│   │   │   ├── color.rs        # Oklab working space, conversions, mixing (§6.5)
│   │   │   ├── colorspace.rs   # ColorSpace trait; Oklab + Mixbox impls (§6.7)
│   │   │   ├── assets.rs       # content-addressed brush/image asset store (§6.6)
│   │   │   ├── noise.rs        # tileable 2-D noise tiles for colour dynamics (§6.2)
│   │   │   ├── image.rs        # RgbaImage (readback / export)
│   │   │   ├── gpu/
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
│   │   │   │   ├── fill.rs      # region fill: a paint parcel through a mask (§6.8)
│   │   │   │   ├── transform.rs # the parcel / combine / mask passes (§16.5)
│   │   │   │   ├── pigment.rs   # the Mixbox LUT (§6.7)
│   │   │   │   └── readback.rs  # GPU→CPU texture readback (export, goldens)
│   │   │   ├── geom.rs         # tile coords, view transform, AABB
│   │   │   ├── path.rs         # streaming B-spline stroke fit + adaptive flatten (§6.2)
│   │   │   ├── spline.rs       # clamped cardinal cubic B-spline + least-squares solve
│   │   │   └── io.rs           # save/load of the action log (§8)
│   │   └── tests/
│   │       └── golden/         # scripted command sequences + reference PNGs (§9)
│   ├── stark-shaders/          # WESL sources + build.rs (wesl link/compile)
│   ├── stark-testdata/         # recorded pen input + asset paths; dev-only (§9)
│   ├── stark-net/              # iroh transport ↔ Replicated timeline (§12)
│   │   └── src/
│   │       ├── session.rs      # CollabSession: the frontend-facing API
│   │       ├── transport/      # the WebRTC path bootstrap
│   │       ├── mirror.rs       # CPU copy of the log, to serve joiners
│   │       └── ticket.rs       # shareable session tickets
│   └── stark-ui/               # Dioxus 0.7 frontend (§11)
│       ├── assets/             # shipped images + stylesheet (fetched at runtime)
│       └── src/
│           ├── main.rs         # app root, canvas, command rail
│           ├── state.rs        # AppState + the dispatch seam
│           ├── render.rs       # WebGPU surface + Engine wrapper
│           ├── input.rs        # DOM events → InputCommand
│           ├── layout.rs       # floating panel chrome + drag/reorder
│           ├── panels/         # one module per tool panel
│           ├── settings.rs     # the unified settings dialog
│           ├── prefs.rs        # what that dialog sets (localStorage)
│           ├── widgets.rs      # shared small controls
│           ├── platform.rs     # the two browser-only helpers
│           ├── shapes.rs       # the per-browser brush shape library
│           ├── presets.rs      # named brush presets (localStorage)
│           ├── builtins.rs     # the built-in shape table
│           ├── brush_editor.rs # the brush dialog + its preview engine
│           └── collab.rs       # session lifecycle glue
└── vendor/                     # third-party, EXCLUDED from the workspace
    ├── mixbox/                 # pigment mixing (submodule, CC BY-NC)
    ├── iroh/                   # iroh 1.0 + custom-path-opening patch (§12.4)
    └── iroh-webrtc-transport/  # WebRTC as an iroh custom transport (§12.4)
```

`stark-core` is the testable, frontend-agnostic backend. It is also
**network-agnostic**: it owns the *merge semantics* of the action log (the
`Timeline` trait) but not the wire transport. `stark-net` adapts iroh to it (§12)
and can be pulled in by the frontend or omitted entirely. `stark-shaders` is split
out so shader compilation (a build step) does not pollute the engine crate.
**`stark-ui` depends on core, never the reverse.**

Two caveats, stated rather than hidden:

- The large image assets (studio HDR, linen weave, bristle brush — 11 MB
  together) live in `crates/stark-ui/assets/`, because Dioxus's `asset!` macro
  rejects any path outside its own crate. stark-core's *tests* want the same
  bytes, so they read them from there. That is a path pointing the wrong way; it
  is confined to one module, `stark_testdata::assets`, which is the only thing
  that breaks if the frontend reorganizes. No code or Cargo dependency crosses
  that way; a second 11 MB copy was the alternative.
- **`vendor/` is in `[workspace] exclude`.** Cargo otherwise promotes an
  unexcluded path dependency to a workspace member, which drags vendored code
  into `cargo fmt --all` and `clippy --workspace`. Their own test suites
  therefore do not run under `cargo test --workspace` — run them by hand when the
  vendored code changes (§20).

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
  settings, the view transform, and the in-progress gesture. Panning the canvas
  or switching tools must never create an undo step.
- **Document state** is everything that defines the artwork, versioned by the
  `history` crate. Only committed, replayable mutations live here.

## 4. Commands vs. Actions (the most important boundary)

Two vocabularies, deliberately not merged. `InputCommand` is *raw, high-frequency
user intent*, including ephemeral input that never lands in history.

There are **three classes of engine state**, and which one a command touches
decides almost everything about it — whether it is logged, whether peers see it,
whether undo reaches it. The split is in the type, not in a comment:

- **Document state** — historized, replicated, reproduced by replay. Layers,
  paint, the canvas surface, the background. The selection is document state too,
  but **owned**: one mask per actor, so a collaborator's lasso never clips your
  brush (§17.3).
- **View state** — per-client, transient, never logged *and never sent*. Tool,
  brush, pan/zoom/rotation, viewport, lighting, media params. Two people sharing
  a drawing pan independently.
- **Presence** — per-client and never logged, like view state, but **published**:
  every collaborator reads it and only its owner writes it. The selected layer,
  the cursor, the gesture in flight (§17.4).

The discriminator between the last two is one question, applied without
exception: **does replay need it to reproduce pixels?**

A gesture is none of the three and gets its own kind: it *builds* in per-client
state and commits a document action at the end — or nothing, if cancelled. In a
shared session the building is published too, so peers watch a stroke as it is
drawn.

```rust
pub enum InputCommand {
    Gesture(GestureCommand),
    Doc(DocCommand),
    View(ViewCommand),      // private: never logged, never sent
    Peer(PeerCommand),      // published: never logged, but broadcast
}

pub enum GestureCommand {          // press-drag-release, for brush *and* selection
    Start { tool: Tool, sample: InputSample },
    To    { sample: InputSample },
    End,                           // the one edge that produces document state
    Cancel,                        // produces none
}

pub enum DocCommand {              // each becomes an Action
    Undo, Redo,
    AddLayer { above: Option<LayerId>, carrier: Option<LayerId> },
    AddMatte { .. },               // §15
    RemoveLayer(LayerId),
    SetLayerBlend(LayerId, BlendMode),
    SetLayerClip(LayerId, bool),   // §14.4
    SetLayerOpacity(LayerId, f32),
    SetLayerVisible(LayerId, bool),
    MoveLayer { id, at, carrier },  // reorder, carry and release in one action (§14.8)
    SetMatteRect(..), SetMatteColor(..),
    Select(SelectionOp),
    InvertSelection,
    SetSurface(SurfaceId),         // which canvas the piece is painted on (§6.4)
    SetBackground([f32; 3]),       // the substrate (§15.5)
}

pub enum ViewCommand {             // never logged, never sent
    SetTool(Tool), SetBrush(BrushParams),
    Pan { delta: Vec2 }, Zoom { anchor: Vec2, factor: f32 },
    Pinch { anchor: Vec2, to: Vec2, scale: f32, turn: f32 },  // two fingers (§18.1.7)
    SetRotation(f32), MirrorH,     // §18.1.2
    Resize(Extent2),
    SetSelectionMode(SelectionMode), SetSelectionFeather(f32),
    SetFillOpacity(f32),           // §6.8 — every fill's one strength knob
    SetMediaParams(MediaParams), SetEnvironment(EnvironmentId),
    SetActiveLayer(LayerId),       // (see PeerCommand — published when sharing)
    PreviewMatteRect(..),          // §15.7
    PreviewMatteColor(..),         // §15.7
    PreviewTransform(..),          // §16.6
    PreviewFill(..),               // §22.4 — the gradient fill's composing drag
    PreviewBackground(..),         // §15.5
    PreviewLayerOpacity(..),       // §14.6  — the in-flight half of a slider drag
    SetShowPeerSelections(bool),   // §17.3
}

pub enum PeerCommand {             // never logged, but published (§17.7)
    SetActiveLayer(LayerId),
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

`Engine::process` takes `impl Into<InputCommand>`, so call sites name the class
and nothing more: `engine.process(ViewCommand::Pan { delta })`.

### Commands vs. requests

Commands are **one-way**: intent goes in, nothing comes back. That is what lets
them become messages over a channel when the engine moves off the UI thread (§7),
and it is the property to protect — a command that returns a value cannot be
sent.

Reads therefore go through `Engine::observe()`, which projects *both* classes of
state (§7). It deliberately includes the view settings a frontend would otherwise
read back off the engine — media params, surface, environment, colour space —
because a frontend that cannot observe them keeps its own copy, and a copy seeded
from `Default` goes stale the moment anything else changes them.

What genuinely cannot be a command is a **request**: an operation that must
answer.

```rust
// assets — the frontend fetches bytes the engine cannot reach for itself (§6.6)
fn import_brush(&self, png: &[u8]) -> Result<AssetId>;
// grounds are content-addressed, so the id comes *out of* the image (§6.4);
// `accept_surface` takes one that arrives already named — from a file's bundle
// or a peer — and refuses bytes that don't hash to the id that asked for them.
fn import_surface(&mut self, png: &[u8]) -> Result<SurfaceId>;
fn accept_surface(&mut self, id: SurfaceId, png: &[u8]) -> Result<SurfaceId>;
fn register_environment(&mut self, id: EnvironmentId, hdr: Vec<u8>);
// persistence (§8)
fn save_bytes(&self) -> Result<Vec<u8>>;
fn load_bytes(&mut self, bytes: &[u8]) -> Result<()>;
// sampling — the eyedropper (§18.0.2), and export (§15.6). Both return a *future*:
// readback is the one inherently asynchronous GPU operation (§7), and the future
// owns what it needs so the engine borrow ends before the await.
fn pick_color(&mut self, at: Vec2, o: PickOptions) -> impl Future<Output = Option<[f32; 3]>>;
// …and the same sampling stretched along a traced line — the gradient capture (§22.2)
fn pick_gradient(&mut self, path: &[Vec2], o: PickOptions) -> impl Future<Output = Option<Gradient>>;
fn export(&mut self, frame: Option<LayerId>, scale: f32, bg: Background) -> Result<impl Future<..>>;
// collaboration transport (§12)
fn merge_remote(&mut self, action: Action) -> bool;
fn take_outbox(&mut self) -> Vec<Action>;
fn take_presence(&mut self) -> Option<PeerFrame>;
fn merge_presence(&mut self, actor: ActorId, frame: PeerFrame) -> bool;
```

These stay direct methods on `Engine`. Under the actor they become
request/response pairs with a reply channel; until then, keeping them *named* as
a tier is what stops them drifting back into ad-hoc setters. **A new engine
method that mutates state and returns nothing is a bug — it should be a command.**

One thing is neither: the **colour space**. Channel layouts differ between
spaces, so changing it cannot preserve a document — every caller asking to "set"
it was really asking for a new document. It is fixed at document creation
(`Engine::new_document(color_space, surface)`) and there is no setter (§6.7).

### Actions

`Action` is a *committed, deterministic, serializable document mutation* — the
unit the timeline stores and replays, and the unit serialized to disk. Every
action is **globally identified** so it can live in a replicated, multi-peer log
(§12) without changing meaning:

```rust
pub struct Action { pub id: ActionId, pub kind: ActionKind }

pub struct ActionId {
    pub lamport: u64,           // logical clock → causal/total ordering
    pub actor: ActorId,         // who authored it
}

pub enum ActionKind {
    CommitStroke(StrokeRecord),
    Fill(FillOp),                             // §6.8
    Transform { layer: LayerId, affine: Affine2 },  // §16
    AddLayer { id, above, carrier }, RemoveLayer(LayerId), MoveLayer { .. },
    AddMatte { .. }, SetMatteRect(..), SetMatteColor(..),   // §15
    SetLayerBlend(LayerId, BlendMode), SetLayerClip(LayerId, bool), ...
    Select(SelectionOp), InvertSelection,
    SetSurface(SurfaceId), SetBackground([f32; 3]),
    Undo(ActionId),             // undo-as-an-action (§5.4 / §12.3)
}

pub struct StrokeRecord {
    pub layer: LayerId,
    pub tool: ToolId,
    pub brush: BrushParams,       // colour in the working space; shape by AssetId
    pub path: Vec<ControlPoint>,  // cubic B-spline control points, fitted (§6.2)
    pub seed: u64,                // makes any brush jitter reproducible
}
```

`ActorId` is a single fixed value solo (`ActorId::SOLO`) and maps to an iroh
`EndpointId` when collaborating. Generating ids locally costs nothing and is the
one piece of forward-compatibility that would be painful to retrofit.

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

Because `StrokeRecord` carries the whole fitted path plus a seed, a committed
stroke replays bit-for-bit — the foundation of undo, goldens and convergence.

## 5. The history model (and why it's cheap)

The `history` crate gives `History<A: Action>` with O(log n) full `State`
snapshots, O(1) amortized push, O(log n) pop, and `get_state(version)` in
O(k + log n) by replaying from the nearest snapshot. Stark binds it as:

```rust
impl history::Action for Action {
    type State   = DocState;       // CHEAP to clone (below)
    type Context = ApplyCtx;       // GPU device/queue + TilePool + renderers
    type Error   = EngineError;

    fn apply(&self, state: DocState, ctx: &mut ApplyCtx) -> Result<DocState, EngineError>;
}
```

The document does **not** call `history` directly; it goes through a `Timeline`
trait so the storage strategy can change without touching `Session`, `Engine` or
the GPU code:

```rust
pub trait Timeline {
    fn push(&mut self, action: Action, ctx: &mut ApplyCtx);
    fn current(&self) -> &DocState;
    fn undo(&mut self, ctx: &mut ApplyCtx) -> bool;   // navigation (solo)
    fn redo(&mut self, ctx: &mut ApplyCtx) -> bool;
    fn clone_actions(&self) -> Vec<Action>;           // the save payload (§8)
    fn seek(&mut self, n: usize, ctx: &mut ApplyCtx); // timelapse scrubbing (§18.2.4)
    // Shared-mode hooks, defaulted so LinearTimeline ignores them (§12):
    fn undo_as_action(&self) -> Option<ActionId> { None }
    fn redo_as_action(&self) -> Option<ActionId> { None }
    fn merge(&mut self, action: Action, ctx: &mut ApplyCtx) -> bool { false }
}
```

- **`LinearTimeline`** — the solo impl, a thin wrapper over `history::History`.
- **`ReplicatedTimeline`** — the multi-peer impl (§12): a totally-ordered set of
  actions reusing the same `history::History` as a *materialization cache* for
  the ordered prefix.

`Session`/`Engine` only ever see the trait, so collaboration was added by
swapping the impl, not by surgery on the engine. Introducing the seam *before*
its second implementation existed is why that held.

### 5.1 `DocState` is a persistent tile map, not pixels

```rust
#[derive(Clone)]
pub struct DocState {
    pub layers: rpds::Vector<Layer>,   // persistent (structural sharing)
    pub bounds: CanvasBounds,          // union of populated tiles (infinite)
    pub selections: rpds::HashTrieMap<ActorId, Selection>,  // owned, §17.3
    pub surface: SurfaceId,
    pub background: [f32; 3],
}

#[derive(Clone)]
pub struct Layer {
    pub id: LayerId,
    pub blend: BlendMode,
    pub clip: bool,                    // §14.4
    pub opacity: f32,
    pub visible: bool,
    pub name: Option<Arc<str>>,
    pub content: LayerContent,
    pub carries: rpds::Vector<Layer>,  // a group is a layer that carries (§14.2)
}

pub enum LayerContent {
    /// sparse map: only populated tiles exist (the infinite canvas)
    Paint(rpds::HashTrieMap<TileCoord, TileHandle>),
    /// a procedural region + a flat fill — the frame, grounds, later gutters (§15)
    Matte { region: MatteRegion, color: [f32; 3] },
}

#[derive(Clone)]
pub struct TileHandle(Arc<GpuTile>);   // Arc bump = the entire "clone" cost
```

Cloning `DocState` bumps a few `Arc`s. That is what makes `history`'s snapshot
retention affordable: each retained version holds *references* to shared GPU
tiles. `rpds`'s structural sharing also gives cheap *diffing* between two
`DocState`s, which is what damage tracking would key off if it existed (§6.3) and
what the collaboration layer uses to merge concurrent edits.

### 5.2 Copy-on-write at tile granularity ties memory to history

A stroke touches a small set of tiles. Rendering produces a new `DocState` where
**only the dirtied tiles are replaced**; every untouched tile is shared.

```
version N      version N+1 (one stroke over 3 tiles)
┌──┬──┬──┐      ┌──┬──┬──┐
│A │B │C │      │A │B'│C │     B' is new; A and C are the SAME Arc.
├──┼──┼──┤  →   ├──┼──┼──┤
│D │E │F │      │D'│E'│F │     D',E' new; F shared.
└──┴──┴──┘      └──┴──┴──┘
```

A `GpuTile` returns to the `TilePool` exactly when its `Arc` refcount hits zero —
i.e. when no `history` snapshot references it. **History retention drives GPU
memory reclamation for free.** No manual GC, no leak. It is also why tile
*identity* works as change detection in `Action::inverse` (§12.6).

That returns a tile to the *pool*; the pool then decides when to return it to the
*driver*. It measures its own peak concurrent demand over an epoch of acquires and
hands back half of anything it owns beyond that, so a burst — a transform across a
wide selection, a stroke over a large canvas — decays geometrically once ordinary
work resumes, instead of being resident for the rest of the session. The epoch is
counted in acquires rather than seconds because that is the pool's only honest
clock: the engine renders on demand, so a frame counter would tick fastest when
there is nothing to reclaim. The cost of that choice is stated where the constant
is: an idle pool does not shrink, it shrinks on the next spell of work.

### 5.3 Undo/redo cost

For retained snapshots, undo is instant — the tile map is already held. Between
snapshots, `get_state` replays the few intervening actions, re-rasterizing those
strokes on the GPU. Strokes are small and deterministic, and snapshots are cheap,
so a dense checkpoint policy keeps replay depth tiny. Redo is symmetric.
Backwards bulk moves (timeline scrubbing) use `pop_actions_with`, which rebuilds
the snapshot cache once instead of once per step.

### 5.4 Two flavours of undo

Deliberately two mechanisms, and they do not conflict:

- **Local timeline undo** (`Timeline::undo`) — the fast solo path, pure `history`
  navigation, nothing written to the log.
- **`ActionKind::Undo(target)`** — undo *as a logged action*, for collaboration
  (§12.3), where undo must be a fact peers can see and order, and must mean "undo
  *my* action" not "undo whatever happened last". It is deliberately **not
  interpreted by `Action::apply`** (undo needs the whole log, not just the prior
  state): the timeline computes the log's **effective sequence** — every
  non-`Undo` action not suppressed by an effective `Undo`
  (`timeline::effective_actions`) — and only that is materialized. Redo is an
  `Undo` of an `Undo`. Solo mode never emits these, and a solo *load* of a shared
  log replays the effective sequence, flattening the undos away.


