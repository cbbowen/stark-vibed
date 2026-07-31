# The engine, persistence, testing, and the frontend

The actor target, the save format, golden tests, the extensibility map, and the Dioxus UI — §7–§11.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 7. The engine actor (async backend)

> **Status: the target, not the present.** Today `Engine::process` is called
> synchronously from the frontend's event handler, and `observe()` is *pulled*
> after each command rather than pushed over a `watch`. Nothing below is wired
> up. It is kept as the design because it is what §4's command/request split is
> maintained for: one-way commands are exactly the things that can become channel
> messages, and requests are exactly the ones that will need a reply channel. If
> the actor is ever abandoned, §4's discipline loses its main justification and
> should be revisited rather than quietly kept.

```rust
pub struct Engine {
    gpu: GpuContext,            // Device, Queue, capabilities
    session: Session,           // tool, brush, view, in-flight gesture
    timeline: Box<dyn Timeline>,// Linear (solo) or Replicated (collab) — §5, §12
    actor: ActorId,             // this engine's author id for new actions
    clock: u64,                 // local Lamport counter
    pool: TilePool,
    stroke: StrokeRenderer,
    compositor: Compositor,
    peers: BTreeMap<ActorId, Peer>,              // the roster, incl. local (§17.4)
    observable: watch::Sender<ObservableState>,  // reactive snapshot for UI
}

impl Engine {
    pub fn new_document(color_space: ColorSpaceId, surface: SurfaceId) -> Self;
    pub async fn run(self, rx: mpsc::Receiver<InputCommand>);
    pub fn render(&mut self, target: &wgpu::TextureView, view: ViewTransform);
    pub fn observe(&self) -> watch::Receiver<ObservableState>;
}
```

`ObservableState` is the cheap, UI-facing projection the frontend renders from —
`can_undo`, `can_redo`, `active_tool`, `brush`, `view`, `doc_bounds`,
`is_stroking`, `doc_revision`, the media/surface/environment/colour-space view
settings (§4), `selection_hull`, and the layer tree with *previewed* matte rects
(§15.7). Published over a `watch`/signal so Dioxus re-renders reactively without
polling pixels. **No pixel data crosses this boundary.**

The **peer roster** is deliberately *not* in it, even though it is UI-facing
(§17.4): `ObservableState` is refreshed after every command and drives the whole
component tree, while presence changes thirty times a second whenever anybody
moves. It is read through `Engine::peers()` into a signal of its own, so a remote
cursor moving re-renders a cursor and not an application.

Engine field count is kept down by grouping: `gpu::Registry<R: Resource>` absorbs
the surface and environment clusters (registered bytes + current id + live GPU
object, the same shape twice), and the four action subsystems live in a *stored*
`ApplyCtx` rather than being rebuilt and cloned on every undo/redo/commit.
(`history::Action::Context` is an owned associated type, so there is nothing to
hand a borrow of — the fix is to store the context, not to borrow it.)

The engine is runtime-agnostic: it uses channels and `async fn run`, so it drops
into tokio (desktop) or wasm-bindgen-futures (web). **GPU buffer readback is the
only inherently async GPU op** — see §15.6 for why that makes `export` return a
future rather than being an `async fn`.

## 8. Save format & timelapse

The native format is **the serialized action log**:

```rust
pub struct DocumentFile {
    pub format_version: u32,
    pub app_build: BuildId,            // shaders/algorithm version for fidelity notes
    pub canvas: CanvasMeta,            // tile size, channel set, colour space, surface
    pub actions: Vec<Action>,          // the full, replayable log (each id-tagged)
    pub assets: Vec<(AssetId, Bytes)>, // content-addressed brush images (§6.6)
    pub checkpoints: Vec<Checkpoint>,  // OPTIONAL cached rasters (see below)
}
```

`assets` bundles every brush image any stroke references, so the file is
self-contained and replayable; loading populates the asset store before replay.

Because every `Action` carries its `ActionId`, a saved file is also a valid
collaboration log: opening it, painting, and later sharing it with a peer all use
the same records. A solo file simply has a single actor.

- **Load** = replay the actions through `apply` to rebuild `DocState`, then the
  whole undo timeline is immediately available — undo-after-load, for free.
- **Timelapse** = replay in order, presenting after each (or each Nth) commit.
  Shipped as **Timeline mode** (§18.2.4), and *not* as a separate replay path:
  `Timeline::seek` moves the applied/withheld boundary the undo stack already
  has, so the timelapse is the document's own history being walked rather than a
  second machine that reproduces it.
- **Compactness** = a path of samples is far smaller than the painted pixels.
- **Fidelity across builds:** replay determinism holds *within* a build. Because
  shader/algorithm changes could alter pixels across builds, the file records
  `app_build`, and may embed periodic rasterized `Checkpoint` tiles as both a
  fast-open cache and a visual fallback. Strokes remain the source of truth;
  checkpoints are advisory and may be empty.

Serialization uses `serde` over `postcard` with a magic header;
`format_version` gates migrations. **Postcard writes fields in order with no
names and no length**, so a field added in the middle of an existing struct
variant is not something it can absorb — that is what forced `WIRE_VERSION` to 2
when `MoveLayer` gained its `carrier` (§14.8). Appending a new enum variant last
is always safe (enums encode by index).

## 9. Testing — golden images

Separating backend from frontend lets the engine be driven headlessly:

```rust
let gpu = GpuContext::headless();              // offscreen, no surface
let mut engine = Engine::new_document(..);
play(&mut engine, script);                     // a Vec<InputCommand>
let png = engine.export_region(rect);          // readback to RGBA8
assert_golden!("oil_blend_01", png, tolerance);
```

- **Scripts** are command sequences exercising each tool, undo/redo, layer ops,
  load+replay.
- **Determinism** is engineered in (seeded jitter, fixed flattening tolerances,
  fixed adapter selection, explicit float formats). The comparator uses a small
  perceptual tolerance to absorb legitimate cross-GPU rounding.
- **Replay equivalence:** paint a stroke, snapshot; undo then redo; serialize →
  load → snapshot. All three must match — this guards §1.3.
- **A missing GPU is a failure, not a skip.** Every GPU test needs an adapter,
  and a skipped test still reports `ok` — so a machine without one would take the
  whole golden / seam / dynamics / selection suite green having rendered nothing.
  Skipping has to be asked for: `STARK_ALLOW_NO_GPU=1`.
- **Goldens are adapter-specific.** A committed PNG can only match the adapter it
  was blessed on, so CI (on software Vulkan) sets `STARK_SKIP_GOLDEN=1`: the
  strokes still render — shader compilation, wgpu validation and panics are all
  caught — and only the pixel comparison is dropped. Deleting a golden re-blesses
  it on the next run.
- **Recorded input** lives in the dev-only `stark-testdata` crate: real pen
  reports captured from the app, because synthetic curves are smooth and evenly
  sampled in ways real input is not, and the fitter's behaviour turns on exactly
  those details.
- **When a model is wrong, fix the model and re-bless.** No compensating fudge
  constants to keep an old golden green.

The suite files, roughly by subject: `golden`, `seam`, `stroke`, `dynamics`,
`path`, `selection`, `fill`, `matte`, `groups`, `blend`, `composite`,
`reference`, `transform`, `export`, `pick`, `view`, `layers`, `save_load`,
`replay`, `collab`, `commute`, `peer_state`, `assets`, `color_dynamics`,
`tile_pool`; and in `stark-net`: `sync`, `presence`, `handoff`.

## 10. Extensibility map

| Want to add… | Touch only… |
|---|---|
| A new tool / brush behaviour | `ToolId` + a `Brush` impl in `gpu/stroke/`; serialized in `BrushParams` |
| Image/organic brush shapes | content-addressed `AssetId` in `BrushShape`; `AssetStore` mask textures; stamp shader samples + rotates (§6.6) |
| A new channel (normal, granulation) | `ChannelSet` descriptor + tile alloc + shader usage; `DocState` unchanged |
| A new document edit | new `ActionKind` variant + its `apply` arm + a `Footprint` arm + serde (auto) |
| A new blend mode | one more `T` — a `BlendMode` variant + a `blend_common.wesl` branch (§6.3) |
| A new media/lighting model | the media pass shader (§6.3) |
| A different frontend (native, CLI exporter) | a new consumer of `Engine`; core untouched |
| Another selection producer (by colour, quick-mask, imported alpha) | a `SelectionShape` variant + an arm in `selection.wesl`; representation, ops, history and masking sites unchanged (§6.8) — and it becomes a *fill* producer in the same move |
| A gradient (or any position-varying fill) | the parcel in `fill.wesl` reads its latent from position rather than a uniform; region, gate, stacking law, action and footprint unchanged |
| A richer frame / comic gutters / a solid ground | a `MatteRegion` variant + an arm in `matte.wesl`; `LayerContent::Matte` and its compositing unchanged (§15) |
| Text | a new `ActionKind` + optionally new channels; transforms landed exactly this way (§16) |
| A wider-gamut / spectral colour pipeline | `color.rs` + a `CanvasMeta.color_space` variant; storage stays float, present picks the transform |
| Multi-user collaboration | swap `LinearTimeline` → `ReplicatedTimeline`; add `stark-net`; engine/GPU untouched (§12) |

The action-log + persistent-state core was chosen precisely so these are
*additive*. Nothing above requires changing the history binding, the tile CoW
scheme, or the command/action split.

## 11. Frontend (Dioxus)

`stark-ui` is a Dioxus 0.7 **web** app: the backend runs in WASM and the painting
surface is a dedicated `wgpu::Surface` bound to the page `<canvas>` via WebGPU,
which the engine draws into directly. DOM chrome surrounds it.

- UI components dispatch `InputCommand`s through one seam, `state::dispatch`,
  which applies, repaints, refreshes `ObservableState` and broadcasts whatever
  was committed — so no call site has to remember that sequence. Pointer events
  become `GestureCommand::Start`/`To`/`End`, with element coordinates mapped via
  `ViewTransform::screen_to_canvas`. `Start` also carries the **input tolerance**
  (§6.2): `devicePixelRatio` and the event's `pointerType` give the device's
  grain in CSS px, and the same view transform carries it into canvas px.
- The engine (and its `wgpu::Surface`, both `!Send`) live in a signal; after each
  command the engine renders **straight into the surface texture**
  (`get_current_texture` → `render` → `present`) — no readback, no encode. The
  frontend supplies GPU handles via `GpuContext::from_parts`; core needs no
  change to compile to wasm.
- **The floating chrome fades while the canvas is in hand** — a stroke, a
  selection drag, a pan, or a run of wheel zooming — and fades back the moment
  the gesture ends. One signal, `AppState::canvas_active`, toggles a `dimmed`
  class the stylesheet animates; the chrome keeps its box (nothing reflows) and
  stops taking clicks while faded, so a stroke straying under a panel keeps
  painting.
- **Panels are first-class.** `PanelId` + a `PanelLayout` context (order, hidden
  set, drag state, mounted refs) make the stack data-driven: each panel has a
  header with a drag handle and a ✕, a "Panels" menubar menu reopens closed ones
  into their original slot, and dragging a title bar reorders with a FLIP
  animation (measure previous tops, apply an inverted transform with no
  transition, then play to zero). `key:` on each panel must be the stable
  `PanelId` so reordering *moves* existing nodes rather than recreating them —
  that is what preserves each panel's internal signal state and makes FLIP
  possible. Lives in `layout.rs` + `assets/stark.css`.
- **Z-order is declared, not inherited from DOM order.** The canvas overlay
  (frame handles, transform widget) sits at `z-index: 10`; every piece of
  floating chrome is 20+. `.panel-stack` must declare its `z-index` explicitly —
  with none it sits at auto level and any positioned sibling with a `z-index`
  beats it regardless of DOM order.
- **The navigator's miniature is a second surface, not an image the UI carries.**
  An overview is the one piece of chrome that cannot be derived from
  `ObservableState` — it is pixels — so `panels/navigator.rs` mounts its own
  `<canvas>`, the frontend binds a second `wgpu::Surface` on the app's existing
  device, and the engine renders into it (`Engine::render_into`). It frames
  itself against the rect a file export would use (`Engine::export_plan`, whose
  returned plan *is* the view it renders through), so the overview cannot come to
  disagree with the picture a file would hold.
  - It subscribes to `ObservableState::doc_revision`, **not** the canvas: a
    refresh composites every tile, affordable per *edit* and ruinous per pointer
    sample. That counter moves when the committed document does and deliberately
    not when an in-flight gesture or an unlogged drag preview changes what the
    canvas shows. `Rendered::Committed` is the matching half engine-side. A
    settle delay collapses bursts, and a refresh due mid-gesture waits for the
    hand to lift. The viewport rectangle is a positioned `<div>` read from the
    live view, so pan and zoom move it for free.
  - This began as an `export` — render offscreen, read pixels back, hand them to
    a 2D canvas via `ImageData` — and every part of it after "render" existed
    only because the miniature had nowhere of its own to draw. Giving it a
    surface deleted the GPU→CPU copy and its frame of latency, the pixel buffer
    in a signal, the `putImageData` helper, and the imperative repaint that had
    to re-run whenever the element remounted.
- **Compositing splits along "does it depend on the target?"**, so a second view
  costs only its own attachments. `CompositorPipeline` holds pipelines, layouts,
  the pigment LUT and the view settings the media pass reads; a `Compositor`
  holds one target's offscreen attachments, blend scratch and instance streams.
  The surface keeps one across frames; anything rendered beside it brings its own
  through an `Offscreen` slot, so an off-screen render never resizes the screen's
  attachments out from under it. Whether a slot outlives its call is the
  *caller's* to state, because only the caller knows whether the render repeats:
  the navigator holds one for the app's life, while a file export uses a local
  one so a 4× export of a large frame does not park its several-hundred-megabyte
  pair for the session. View settings stay single-owned behind a process-wide
  generation stamp, so a swapped weave or light — or a whole rebuilt pipeline, as
  a colour-space change makes — reaches every consumer by being *noticed* rather
  than by a notification a new consumer could be left out of.
- **Settings are one dialog, not a control tucked into whichever panel it came
  from.** Panels hold what you are painting *with* and change constantly
  mid-stroke; document dialogs hold what the drawing *is*. A standing per-client
  preference is neither — set once, never part of the artwork — so it lives in
  the ⚙ dialog off the command rail (`settings.rs`). Its rows apply on the click
  (Done, no Cancel: nothing is staged) and stay mounted even when inert, saying
  so in their own text — deliberately the opposite of the §6.8 rule for tool
  bars, because a settings dialog is read as the map of what is configurable.

Because the engine is frontend-agnostic, this layer stays thin. (An earlier
interim cut ran on Dioxus *desktop* and bridged the canvas by reading the frame
back to a PNG data URL — correct but laggy; the WebGPU surface replaced it,
touching only `stark-ui`.) Run with `dx serve --web -p stark-ui` in a WebGPU
browser. A native winit/desktop frontend could reuse the same engine.


