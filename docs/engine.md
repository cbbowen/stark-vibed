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
`is_stroking`, `doc_revision`, the media/surface/environment/color-space view
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
    pub canvas: CanvasMeta,            // tile size, channel set, color space, surface
    pub actions: Vec<Action>,          // the full, replayable log (each id-tagged)
    pub assets: Vec<(AssetId, Bytes)>, // content-addressed brush images (§6.6)
    pub surfaces: Vec<(SurfaceId, Bytes)>, // the canvas grounds it names (§6.4)
    pub checkpoints: Vec<Checkpoint>,  // OPTIONAL cached rasters (see below)
}
```

`assets` bundles every brush image any stroke references and `surfaces` every
canvas ground the log names, so the file is self-contained and replayable;
loading populates both stores before replaying a single action.

**Both are replay inputs, and the ground took a bug to see it.** A brush mask
obviously decides pixels. So does a height map, once the deposition tooth gates
how much paint lands on it (§6.4) — but the ground was a *label* (`Linen`,
`Gesso`) resolved against whatever the reader happened to hold, so a file
recorded the name and left the image to chance. Open it on a build whose
`Gesso.png` had been re-authored and the strokes came back different, silently;
hand the log to a peer who had never fetched that ground and it replayed on the
flat stand-in, diverging. Naming a ground by the hash of its image and shipping
the image with the log is what makes a file mean one thing. **Every** ground the
log names is bundled, not just the one it ends on: the tooth reads whichever was
in force when a stroke was made, so a document that switched part-way needs both
to come back the same.

**A bundle may be deliberately incomplete** (format version 6). The log is fitted
paths and the bundle is megabytes — the built-in grounds canonicalize to 2.0 and
2.8 MB — so a doodle painted on a ground the app ships with was almost entirely a
copy of a file the reader already had. `save_bytes_resolvable` leaves out content
the opening app can produce itself, and `DocumentFile::unbundled_content` is the
bill: **settle it before replaying**, since a `SetSurface` whose height map is not
registered when its strokes replay deposits them through the flat stand-in.

**And the replay refuses if it was not settled** —
`EngineError::MissingContent`, from `load_document`, `load_bytes` and the
timelapse alike, carrying the outstanding needs. That used to be a log line with
`Ok(())` behind it, and the difference is not diligence: a missing ground does not
degrade the *view*, it bakes a smooth deposit into stored tiles, so there is
nothing left afterwards to notice it by. A dev harness replayed a captured bug
report perfectly smooth on that path, and the smoothness was the bug being
hunted. The check runs before the document is adopted, so a refusal leaves
whatever was open untouched rather than half-replacing it. A collaboration
**join** is the one caller that legitimately starts short — its blobs arrive over
the same transport as its actions and the waitlist parks anything that depends on
one (§12.4) — and it does not come through here.

Which leaves the frontend as the only thing that could ever pay a lean file's
bill, since only its build script hashes the shipped PNGs into a table
(`builtin_ids`). `stark-testdata::assets::bundled` is the same table derived at
*runtime* from the same files, so a test or a repro harness can open a capture the
way the app opens it. That is a dev-only mirror of a frontend concern, kept in the
one crate that already reaches into `stark-ui/assets` (§2).

This walks back part of the paragraph above, so it is worth being exact about
what changed. That bug was a ground named by a *label*, resolved against whatever
table the reader held — re-author `Gesso.png` and the pixels changed with nothing
able to notice. A lean file still names the ground by the hash of its image;
content that does not hash to it is refused rather than substituted. So the
failure mode of a re-authored asset is a document that **will not open**, not one
that opens wrong, and the frontend's shipped catalog is append-only (a test) so
that it does not arise. What is genuinely given up is self-containment: a lean
file needs an app that still ships the content, which is why `save_bytes` still
writes everything and the lean path is a separate call.

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
is always safe (enums encode by index); *re*-shaping an existing one is not,
which is half of what took the version to 4 when `SurfaceId` went from three
named grounds to `Flat | Image(AssetId)` — a version-3 file's `Linen` would
decode as an `Image` whose hash is whatever bytes followed it. The `surfaces`
field beside it is the other half. Files are alpha (§19), so old ones are refused
rather than migrated.

Giving an existing *unit* variant a payload is the same break read from the other
end, and it took the version to 11 when `BlendMode::Drago` gained its bend (§6.3):
a version-10 file writes the bare index, so a reader expecting a float takes the
next action's bytes as the curve and the log is off from there. Worth naming
separately because it is the case that looks safest — nothing was reordered, and
nothing about the enum's shape says a variant used to be empty.

The rule cuts the other way too, and it is worth knowing which side a change is
on before paying for it: an `Option<T>` *is* an enum on the wire (`0` for `None`,
`1` for `Some`), so widening one into a named enum whose first two variants are
those two cases costs nothing at all. That is how `MoveLayer`'s anchor grew a
third state (§14.8) without a version bump — and why the variant order is under
test rather than under a comment, since getting it wrong reinterprets every
affected action in every saved file with nothing able to notice.

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
| Another selection producer (by color, quick-mask, imported alpha) | a `SelectionShape` variant + an arm in `selection.wesl`; representation, ops, history and masking sites unchanged (§6.8) — and it becomes a *fill* producer in the same move |
| Another position-varying fill (noise, pattern) | a `Parcel` variant + an arm in `fill.wesl`'s parcel branch; region, gate, stacking law, action and footprint unchanged — the gradient landed exactly this way (§22.4) |
| A richer frame / comic gutters | a `MatteRegion` variant + an arm in `matte.wesl`; `LayerContent::Matte` and its compositing unchanged (§15) — the solid ground landed exactly this way (`Everything`, §15.5) |
| Text | a new `ActionKind` + optionally new channels; transforms landed exactly this way (§16) |
| A wider-gamut / spectral color pipeline | `color.rs` + a `CanvasMeta.color_space` variant; storage stays float, present picks the transform |
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
  was committed — so no call site has to remember that sequence. **No call site
  *can* forget it**: the renderer and its projection are held in `state::ReadOnly`
  handles, which have `read` and `peek` and no `write`, so `&mut Renderer` is
  unreachable outside `state.rs`. The doors are `dispatch` and `with_engine`,
  which publish the projection on the way out, and `with_engine_quiet`, for the
  `&mut` that cannot change what `observe()` projects — rendering, the readbacks,
  the outbox and presence drains, installing asset bytes an action will later
  name. That is a type where a convention was: twice, a panel reached the engine
  through the signal, moved state the chrome reads back, and left the chrome
  asserting the old value until an unrelated command refreshed it — once for the
  canvas ground, once for the lighting environment. Neither spelling compiles.
  Pointer events
  become `GestureCommand::Start`/`To`/`End`, with element coordinates mapped via
  `ViewTransform::screen_to_canvas`. `Start` also carries the **input tolerance**
  (§6.2): `devicePixelRatio` and the event's `pointerType` give the device's
  grain in CSS px, and the same view transform carries it into canvas px. A
  fourth, `Hold`, is sent by the dwell watcher when the pointer stops moving
  mid-stroke — the drawing assist (§6.9). It is the frontend that has the clock,
  and the moves after it are still plain `To`s.
- **Navigation is one vocabulary, asked three times.** `input::Nav` owns every
  binding that moves the view — the two-finger gesture (§18.1.7), middle-drag and
  space-drag pan, wheel zoom — and every surface over the canvas makes its own
  (the canvas, the transform mode's catcher). Its three entry points are a
  lifecycle, `begin` / `advance` / `release`, and each answers the same question:
  *was this event mine?* So a surface routes its pointers by asking three times
  and never by inspecting buttons or pointer types itself, and what "the pan
  bindings" and "the zoom rate" mean cannot drift between surfaces. Policy stays
  at the call site — the canvas fades the chrome while it navigates and cancels
  the stroke a second finger interrupted, the transform overlay deliberately does
  neither.
- **A composing mode is a state, not a stacking order.** Four gestures take the
  canvas away from the brush for the length of a composition: the transform
  widget (§16.6), the perspective-guide edit (§20.5), the gradient trace (§22.2)
  and the gradient fill's axis (§22.4). Each mounts a full-viewport catcher, and
  a catcher is a claim about *hit testing* only — which is not the only way a
  pointer reaches the canvas. A gesture already in flight has **captured** its
  pointer, and a captured pointer's moves and its release are delivered to the
  element that took them whatever has been stacked over it since; a pen drawing
  while the other hand opens a transform kept feeding the fitter under the
  widget and committed on release. So `modes::composing` makes the question
  askable in Rust: the canvas cancels a gesture the moment a mode opens under it
  (cancel, not commit — the canvas stopped taking paint, so it must leave no
  mark), and chrome stacked above the catchers stands down rather than floating
  grips over them.
  - `modes::leave` is the other half. Every entry into a mode leaves whichever
    was already live, dropping its preview and committing nothing, so "two modes
    composing at once" is unreachable rather than arbitrated — the four catchers
    share one `z-index`, where the *last* sibling takes the pointer, so the
    priority the comments had claimed ran backwards. Entering Timeline mode and
    an undo or redo from the keyboard call it too: a preview is computed against
    the committed document, and moving the playhead out from under one leaves the
    widget pointing at paint that is no longer there.
  - **The keyboard asks what the canvas asks.** A press is refused while the
    playhead is moving, because a commit clears the withheld half of the timeline
    (§18.2.4) — but `Ctrl+A` and `Ctrl+Shift+I` went through and truncated the
    history from the keyboard, and edited the very selection a transform was
    composing against while the bar carrying those same commands had stood down.
    They now ask both questions. Undo and redo are not refused but *resolve*:
    nothing on screen says they are unavailable, so instead they stop playback,
    put down the composition, and act.
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
  costs only its own attachments — and again along **"does it ever change?"**, so
  a second *engine* costs only its own view settings. `CompositorPipeline` holds
  the view settings the media pass reads over an `Arc` of `CompositorPasses` —
  the pipelines, layouts and pigment LUT, immutable once built and therefore
  shareable across engines (`Engine::new_sharing`); a `Compositor` holds one
  target's offscreen attachments, blend scratch and instance streams.
  The surface keeps one across frames; anything rendered beside it brings its own
  through an `Offscreen` slot, so an off-screen render never resizes the screen's
  attachments out from under it. Whether a slot outlives its call is the
  *caller's* to state, because only the caller knows whether the render repeats:
  the navigator holds one for the app's life, while a file export uses a local
  one so a 4× export of a large frame does not park its several-hundred-megabyte
  pair for the session. View settings stay per-pipeline behind a process-wide
  generation stamp, so a swapped weave or light — or a whole rebuilt pipeline, as
  a color-space change makes — reaches every consumer by being *noticed* rather
  than by a notification a new consumer could be left out of.
- **A preview engine is a sibling, not a second boot.** The brush editor's test
  canvas and the preset thumbnails each want an isolated document that renders
  *exactly* as the main canvas would — which is an argument for sharing the
  machinery, not merely an economy. `Engine::new_sharing` builds one around a
  fresh document, sharing everything expensive and un-disagreeable: the compiled
  pipelines (immutable), the content-addressed brush assets and the ground /
  environment byte-and-build caches (a `Registry`'s store is `Arc`-shared while
  each sibling keeps its own *current* id), and the tile pool (an allocator).
  What an engine can set stays its own. So the editor's preview
  (`Renderer::shared`) opens on the canvas's ground under its lighting with
  nothing fetched and nothing decoded, and the thumbnails' engine
  (`Renderer::shared_engine`, `thumbs.rs`) deliberately pins the opposite look —
  flat ground, neutral light — so a thumbnail is the *brush's* identity card and
  its cache key is the brush snapshot alone. Each preset row's picture is two
  half-canvas fills (the ground is all paint, so smearing and lifting read), one
  replayed stroke and one small `Engine::export_view` readback on that one kept
  engine, generated in the background and cached per session.
- **Settings are one dialog, not a control tucked into whichever panel it came
  from.** Panels hold what you are painting *with* and change constantly
  mid-stroke; document dialogs hold what the drawing *is*. A standing per-client
  preference is neither — set once, never part of the artwork — so it lives in
  the ⚙ dialog off the command rail (`settings.rs`). Its rows apply on the click
  (Done, no Cancel: nothing is staged) and stay mounted even when inert, saying
  so in their own text — deliberately the opposite of the §6.8 rule for tool
  bars, because a settings dialog is read as the map of what is configurable.
  - **They are saved on the click too** — per browser, in `localStorage`, never
    into the document and never to peers (`prefs.rs`). "Set once and left alone"
    is a promise a reload would otherwise break, so the settings follow the
    browser the way the shape and preset libraries do, and degrade to a
    per-session choice where storage is unavailable. One serde struct holds the
    lot: `#[serde(default)]` per field is what lets a preference added later read
    as its default out of values stored before it existed, instead of a parse
    failure resetting everything the user had set. The dialog's rows do not opt
    in — the toggle component persists after calling its handler, so a new row is
    durable by construction and only its *value* has to be named. Loading
    happens in two passes for the one preference that is engine session state
    rather than a frontend signal (peer selection outlines, §17.3): the frontend
    half applies in the root's body so the first render is already in the right
    mode, and the engine half waits for the renderer, exactly as
    `presets::load`/`apply_first` split.

- **The app installs, and it starts offline.** `index.html` (the crate root's
  own, which replaces the one `dx` would generate) links a web app manifest and
  registers a service worker; both, with the launcher icons, live in
  `stark-ui/public/`, which the CLI copies to the **site root** unhashed —
  unlike `assets/`, whose every file is renamed by content hash. That
  distinction is the whole design of `public/sw.js`: the navigation response
  names one build's hashed wasm, so it is fetched **network-first** and only
  falls back to cache; everything else same-origin is content-addressed and so
  can never be stale, and is served **cache-first** while a background fetch
  refreshes it. Cross-origin, non-`GET` and range requests are passed straight
  through, so the collaboration transport (§12.4) and any partial fetch are
  untouched.
  - This matters more here than for a typical app because the heavy assets are
    deliberately *not* in the wasm binary: the brush stamps (§6.6), the ground
    height maps (§6.4) and the environment HDR (§6.3) are all fetched after boot
    by `builtins::import_all` / `grounds::open_default`. Without the worker that
    is a fresh multi-megabyte download every start, and offline it is a document
    that opens smooth and unlit.
  - The custom `index.html` costs one thing: `dx serve`'s "rebuilding" toast,
    which lives in the CLI's dev shell. Hot reload itself rides the wasm glue and
    is unaffected. Note also that the CLI resolves its placeholders by literal
    text match over the whole file, comments included.
  - **A `.stark` opens in it.** The manifest declares a `file_handlers` entry for
    the extension, and `files::bind_file_launch` takes the other end: the
    browser's `launchQueue`, reached by reflection because neither it nor the
    `FileSystemFileHandle` it yields is in stable `web-sys`. Setting the consumer
    is what *delivers* a queued launch, so it is bound at the end of the startup
    task — a document has nowhere to load until the renderer exists. The manifest
    asks for `focus-existing` rather than `navigate-existing`, so a second launch
    reaches the *running* app: a reload would throw the open painting away before
    the new file had even been read, whereas this path refuses a bad file with
    the current one still on screen. Only the first file of a launch is taken —
    opening a document replaces the canvas (§8), so a second would be a painting
    nobody ever sees, which is what `single-client` says too.
  - The icons are painted with Stark's own bristle stamp by
    `stark-ui/tools/make-icons.py`, run by hand; the PNGs are checked in. Not a
    build step — a logo is not a build product, and nothing keys off its bytes
    the way `stark-assetid` keys off an asset's (§19).

Because the engine is frontend-agnostic, this layer stays thin. (An earlier
interim cut ran on Dioxus *desktop* and bridged the canvas by reading the frame
back to a PNG data URL — correct but laggy; the WebGPU surface replaced it,
touching only `stark-ui`.) Run with `dx serve --web -p stark-ui` in a WebGPU
browser. A native winit/desktop frontend could reuse the same engine.


