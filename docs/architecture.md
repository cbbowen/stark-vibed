# Architecture

Principles, crate layout, the command/action boundary, and the history model — §1–§5.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.
> One name per thing: [glossary.md](glossary.md).

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
6. **Perceptual color is the working space.** Color stores and blends in
   **Oklab** (or Mixbox pigment latents, §6.7), so mixing, compositing and
   gradients are perceptually uniform; conversion to a display space happens only
   at the final present. Color math never touches gamma-encoded sRGB.
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
│   ├── stark-model/            # the document — no wgpu, no shaders, no build step
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── document/       # the log and its vocabulary
│   │       │   ├── action.rs    # Action + ActionKind: what a mutation *says* (§4)
│   │       │   ├── fold.rs      # Materialize + Logged: that a log folds (§5)
│   │       │   ├── footprint.rs # what an action reads/writes (§12.6)
│   │       │   ├── brush/       # the pen mappings, the effects, and the tip (§6.2)
│   │       │   ├── layer.rs     # LayerId, BlendMode, Place, MattePaint (§14, §15)
│   │       │   ├── selection.rs # SelectionOp and its shapes (§6.8)
│   │       │   ├── fill.rs      # FillOp, and the box it writes (§18.0.4)
│   │       │   ├── filter.rs    # the filter parameters (§21)
│   │       │   ├── guide/       # the perspective guide, and all §20 derives (§20.5)
│   │       │   ├── transform.rs # the maps, and the homography solve (§16)
│   │       │   └── warp.rs      # the warp lattice (§16.9)
│   │       ├── io.rs           # the save format, which *is* the action log (§8)
│   │       ├── content.rs      # what a log names but does not carry (§8, §19)
│   │       ├── peer.rs         # the presence wire frames (§17.4)
│   │       ├── error.rs        # DocError: what can go wrong with a *document*
│   │       ├── geom.rs         # tile coords, tile cover, the view transform
│   │       ├── path.rs         # ControlPoint: the stored form of a stroke (§6.2)
│   │       ├── color.rs        # Oklab working space, conversions (§6.5)
│   │       ├── colorspace.rs   # ColorSpaceId — the id, not the space (§6.7)
│   │       ├── substrate.rs    # SubstrateId — the id, not the map (§6.4)
│   │       └── gradient.rs     # Gradient + the fit through a traced line (§22)
│   ├── stark-engine/           # the derived view — no UI, no windowing
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine/         # owns everything; process(InputCommand) (§4, §7)
│   │   │   ├── command.rs      # Gesture/Doc/View/Peer commands (§4)
│   │   │   ├── session.rs      # view state: tool, brush, view, in-flight gesture
│   │   │   ├── peer.rs         # presence: the roster built from the frames (§17.4)
│   │   │   ├── presence.rs     # the publish latch (§17.5)
│   │   │   ├── error.rs        # EngineError + Result; folds DocError in
│   │   │   ├── filters.rs      # the filter passes' host-side numbers (§6.10, §21)
│   │   │   ├── document/       # the state the log folds into
│   │   │   │   ├── apply.rs     # impl Materialize for DocState — the fold (§4)
│   │   │   │   ├── state.rs     # DocState: layers, selections, substrate, guides
│   │   │   │   ├── timeline.rs  # Timeline trait; Linear + Replicated impls
│   │   │   │   ├── layer.rs     # Layer, LayerContent, PaintTiles (§14)
│   │   │   │   ├── selection.rs # the mask the op produces (§6.8)
│   │   │   │   ├── fill.rs      # planning a fill against the mask (§6.8)
│   │   │   │   ├── transform.rs # the tile plans (§16)
│   │   │   │   ├── merge.rs     # merging a layer down (§14.11)
│   │   │   │   └── patch.rs     # Materialize::unfold (§12.6)
│   │   │   ├── colorspace.rs   # ColorSpace trait; Oklab + Mixbox impls (§6.7)
│   │   │   ├── assets.rs       # the content-addressed asset store (§6.6)
│   │   │   ├── noise.rs        # tileable 2-D noise tiles for color dynamics (§6.2)
│   │   │   ├── image.rs        # RgbaImage (readback / export)
│   │   │   ├── gpu/
│   │   │   │   ├── context.rs   # device/queue wrapper, capabilities
│   │   │   │   ├── tile.rs      # TilePool, CoW tile handles, channel set (§6.1)
│   │   │   │   ├── registry.rs  # frontend-provided resources: bytes + live object
│   │   │   │   ├── stroke/      # the brush engine / stroke rasterizer (§6.2)
│   │   │   │   ├── composite/   # compositing + the media/lighting pass (§6.3)
│   │   │   │   ├── environment/ # HDR environment maps for IBL (§6.3)
│   │   │   │   ├── substrate/   # canvas substrate: the grain's relief (§6.4)
│   │   │   │   ├── selection.rs # selection-mask rasterization (§6.8)
│   │   │   │   ├── fill.rs      # region fill: a paint parcel through a mask (§6.8)
│   │   │   │   ├── transform.rs # the parcel / combine / mask passes (§16.5)
│   │   │   │   ├── pigment.rs   # the Mixbox LUT (§6.7)
│   │   │   │   └── readback.rs  # GPU→CPU texture readback (export, goldens)
│   │   │   ├── path/           # fit.rs, flatten.rs, arc.rs — three subjects the one
│   │   │   │                   # file's own banners already named (§6.2). The root
│   │   │   │                   # keeps what belongs to none of them: the span
│   │   │   │                   # arithmetic both ends ask in terms of
│   │   │   ├── assist/         # recognize.rs, adjust.rs, realize.rs (§6.9), likewise
│   │   │   └── spline.rs       # clamped cardinal cubic B-spline + least-squares
│   │   └── tests/
│   │       └── golden/         # scripted command sequences + reference PNGs (§9)
│   ├── stark-assetid/          # what a content id *is* — usable from a build script
│   ├── stark-shaders/          # WESL sources + build.rs (wesl link/compile)
│   ├── stark-testdata/         # recorded pen input + asset paths; dev-only (§9)
│   ├── stark-net/              # iroh transport ↔ the replicated timeline (§12)
│   │   └── src/
│   │       ├── session.rs      # CollabSession: the frontend-facing API
│   │       ├── transport/      # the WebRTC path bootstrap
│   │       ├── mirror.rs       # CPU copy of the log, to serve joiners
│   │       └── ticket.rs       # shareable session tickets
│   └── stark-dioxus-frontend/  # Dioxus 0.7 frontend (§11)
│       ├── assets/             # shipped images + stylesheet (fetched at runtime)
│       └── src/
│           ├── main.rs         # app root, canvas, command rail
│           ├── state.rs        # AppState + the dispatch seam
│           ├── render.rs       # WebGPU surface + Engine wrapper
│           ├── input.rs        # DOM events → InputCommand
│           ├── layout.rs       # floating panel chrome + drag/reorder
│           ├── visibility.rs   # what is on screen, as the browser keeps it
│           ├── panels/         # one module per tool panel
│           ├── settings.rs     # the unified settings dialog
│           ├── prefs.rs        # what that dialog sets (localStorage)
│           ├── widgets.rs      # shared small controls
│           ├── platform.rs     # the two browser-only helpers
│           ├── shapes.rs       # the per-browser brush shape library
│           ├── presets.rs      # named brush presets (localStorage)
│           ├── builtins.rs     # the built-in shape table
│           ├── brush_editor.rs # the brush dialog + its preview engine
│           ├── thumbs.rs       # rendered preset thumbnails (a shared engine)
│           ├── layer_thumbs.rs # rendered layer thumbnails (the live engine)
│           └── collab.rs       # session lifecycle glue
└── vendor/                     # third-party, EXCLUDED from the workspace
    ├── mixbox/                 # pigment mixing (submodule, CC BY-NC)
    ├── iroh/                   # iroh 1.0 + custom-path-opening patch (§12.4)
    └── iroh-webrtc-transport/  # WebRTC as an iroh custom transport (§12.4)
```

**`stark-model` is the document, and nothing else**: what an `Action` is, what it
reads and writes (`Footprint`, §12.6), and how a log is written to a file (§8) or
handed to a peer (§12). It compiles without wgpu, without `stark-shaders` and
without a build script — about fifty crates against the engine's two hundred — and
it is where the file format lives, so the wire-compatibility rules (§8, §19) have
one place to be true.

**`stark-engine` is the derived view**: `DocState`, the tile pool, the renderers,
the compositor, and the controller that drives them (`Session`, the command tier,
§4). It depends on `stark-model`; the model depends on none of them. The split is
this document's founding sentence made structural — pixels are a cached function of
the log, so the log does not know what a pixel is.

**The boundary is visible in the type names, and was before the crates existed**: an
**id** is in the log and a **resource** is in the engine. `AssetId`/`AssetStore`,
`SubstrateId`/`SubstrateMap`, `ColorSpaceId`/`ColorSpace`, `LayerId`/`Layer`,
`SelectionOp`/`Selection`, `Action`/`DocState`. A new pair follows the same rule, and
the mechanical form of it is `#[derive(Serialize)]`: if a type is serializable it is
a fact about the document and belongs left of the line; if it holds a tile it is a
cache and belongs right of it. That is not a judgement call — it is the invariant §8
already enforces, which is why the boundary can be checked rather than remembered.

Four modules are cut down the middle by that line and keep the same file name on
both sides — `document/layer.rs`, `document/selection.rs`, `document/fill.rs`,
`document/transform.rs`. Reading an import tells you which half you are in.

`document/guide/` (§20.5) is the module the rule places *whole*, and it is worth
knowing why. Everything §20 derives from a perspective — its vanishing points, its
fans, the draw-ready `GuideScene`, the `Scaffold` a snapped stroke is held to — is
a pure function of the camera and touches no state, no device and no shader, which
puts all of it beside the fact, on the argument that already put `fill_bounds` and
the homography solve there. The engine keeps the two pieces that genuinely need
its side: packing a `GuideScene` into the guide pass's uniform, and the roster's
per-client half — which of the document's guides *this* client is looking at.

**An action folds over the state through `stark_model::document::Materialize`.** The
history crate asks for `history::Action`, whose `State` would be `DocState`; with the
two apart, that impl is a foreign trait for a foreign type in one crate and
unnameable in the other. `Logged<S>` — a local wrapper carrying a generic impl — is
the way through, and it states the division exactly: the model owns *that* a log
folds and which actions commute, the engine owns what it folds into.
`ColorSpaceId::make`, `ActionKind::is_noop_on` and `SelectionOp`'s uniform packing
became free functions for the same reason, and each of those moves is the boundary
telling the truth about where the work belonged.

**Anything that reads the generated shader mirror stays with the shaders** (§6.10).
That is what `stark-engine`'s `filters` collects — `CONTRAST_PIVOT` and the
dispersion spectrum — and why `SelectionOp`'s uniform packing sits in `gpu/selection`
rather than beside the op. The op is a document fact; how it is packed for
`selection.wesl` is not.

`stark-net` adapts iroh to the log (§12) and depends on **`stark-model` alone** — it
names no engine type at all, and takes `stark-engine` only as a dev-dependency,
because its tests paint. `stark-dioxus-frontend` depends on both, and on neither
through the other: a type has one public path, now enforced by the crate
boundary instead of by convention. **`stark-dioxus-frontend` depends on the
engine, never the reverse.** `stark-shaders` is split out so shader compilation
(a build step) does not pollute the engine crate; `stark-assetid` is split out
so a *build script* can compute a content id without a GPU, which is what lets
the frontend know a bundled asset's id before fetching it (§19).

Two caveats, stated rather than hidden:

- The large image assets (studio HDR, linen substrate, bristle brush — 11 MB
  together) live in `crates/stark-dioxus-frontend/assets/`, because Dioxus's
  `asset!` macro rejects any path outside its own crate. stark-engine's *tests*
  want the same bytes, so they read them from there. That is a path pointing the
  wrong way; it is confined to one module, `stark_testdata::assets`, which is
  the only thing that breaks if the frontend reorganizes. No code or Cargo
  dependency crosses that way; a second 11 MB copy was the alternative.
- **`vendor/` is in `[workspace] exclude`.** Cargo otherwise promotes an
  unexcluded path dependency to a workspace member, which drags vendored code
  into `cargo fmt --all` and `clippy --workspace`. Their own test suites
  therefore do not run under `cargo nextest run --workspace` — run them by hand
  when the vendored code changes (§20).

## 3. Layered architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ stark-dioxus-frontend   DOM chrome + a GPU canvas surface       │
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
  paint, the canvas substrate, the substrate color. The selection is document state too,
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
    SetSubstrate(SubstrateId),         // which canvas the piece is painted on (§6.4)
    SetSubstrateColor([f32; 3]),       // the substrate (§15.5)
    AddGuide { guide, after, name },  // a perspective to construct through (§20.5)
    RemoveGuide(GuideId),
    SetGuide(GuideId, PerspectiveGuide),   // the whole camera, per settled gesture
    SetGuideName(GuideId, Option<String>),
    MoveGuide { id, after },       // the roster's arrangement
}

pub enum ViewCommand {             // never logged, never sent
    SetTool(Tool), SetBrush(BrushParams),
    Pan { delta: Vec2 }, Zoom { anchor: Vec2, factor: f32 },
    Pinch { anchor: Vec2, to: Vec2, scale: f32, turn: f32 },  // two fingers (§18.1.7)
    SetRotation(f32), MirrorH,     // §18.1.2
    CenterOn(Vec2),                // absolute — the navigator's click
    ShowPiece(Option<LayerId>),    // §15.6 — frame the whole piece, e.g. on load
    Resize(Extent2),
    SetSelectionMode(SelectionMode), SetSelectionFeather(f32),
    SetShapeOpacity(f32),          // §6.8 — how strongly the next gesture lands
    SetMediaParams(MediaParams), SetEnvironment(EnvironmentId),
    SetActiveLayer(LayerId),       // (see PeerCommand — published when sharing)
    PreviewMatteRect(..),          // §15.7
    PreviewMatteColor(..),         // §15.7
    PreviewTransform(..),          // §16.6
    PreviewFill(..),               // §22.4 — the gradient fill's composing drag
    PreviewBackground(..),         // §15.5
    PreviewLayerOpacity(..),       // §14.6  — the in-flight half of a slider drag
    PreviewLayerBlend(..),         // §6.3   — the same, for a mode's own parameters
    SetShowPeerSelections(bool),   // §17.3
    PreviewGuide(..),              // §20.5 — the in-flight half of a guide drag
    SetGuideVisible(GuideId, bool),// §20.5 — a guide's eye is *yours*, not the doc's
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
read back off the engine — media params, substrate, environment, color space —
because a frontend that cannot observe them keeps its own copy, and a copy seeded
from `Default` goes stale the moment anything else changes them.

What genuinely cannot be a command is a **request**: an operation that must
answer.

```rust
// assets — the frontend fetches bytes the engine cannot reach for itself (§6.6)
fn import_brush(&self, png: &[u8]) -> Result<AssetId>;
// substrates are content-addressed, so the id comes *out of* the image (§6.4);
// `accept_substrate` takes one that arrives already named — from a file's bundle
// or a peer — and refuses bytes that don't hash to the id that asked for them.
fn import_substrate(&mut self, png: &[u8]) -> Result<SubstrateId>;
fn accept_substrate(&mut self, id: SubstrateId, png: &[u8]) -> Result<SubstrateId>;
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

One thing is neither: the **color space**. Channel layouts differ between
spaces, so changing it cannot preserve a document — every caller asking to "set"
it was really asking for a new document. It is fixed at document creation
(`Engine::new_document(color_space, substrate)`) and there is no setter (§6.7).

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
    SetSubstrate(SubstrateId), SetSubstrateColor([f32; 3]),
    AddGuide { .. }, RemoveGuide(GuideId), SetGuide(..), SetGuideName(..),
    MoveGuide { .. },           // the drawing guides (§20.5)
    Undo(ActionId),             // undo-as-an-action (§5.4 / §12.3)
}

pub struct StrokeRecord {
    pub layer: LayerId,
    pub tool: ToolId,
    pub brush: BrushParams,       // color in the working space; shape by AssetId
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

The document does **not** call `history` directly; it goes through a `Timeline` so
the storage strategy can change without touching `Session`, `Engine` or the GPU
code. An **enum**, not a trait, and that is load-bearing rather than incidental:

```rust
pub enum Timeline {
    Linear(LinearTimeline),
    Replicated(ReplicatedTimeline),
}
```

Every operation is a `match`, and every refusal is written out as an arm — a
solo timeline saying no to a merge, a replicated one saying no to a seek. A trait
would have spelled those as defaulted methods, and a default is how a new
operation comes to do nothing in one mode without anyone deciding it should. Two
variants is also the whole of the space: the split is solo versus shared, and a
third storage strategy would be a different question rather than a third impl.

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
    pub substrate: SubstrateId,
    pub substrate_color: [f32; 3],
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
    /// a procedural region + a flat fill — the frame, backings, later gutters (§15)
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


