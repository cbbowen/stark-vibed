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

### 7.1 Timing — where a frame's time goes

A phase of the pipeline is **a span, and the span's name is its row**. That is the
whole model (`stark_engine::timing`):

```rust
fn flush_live(&mut self) {
    if !self.preview.stale { return; }
    timing::span!("live.fold");        // measured to the end of the block
    // …
}
```

A `tracing_subscriber` layer stamps the clock at span creation and records
`close − creation` into an HDR histogram per name; `timing::snapshot()` reads them
out as counts, means, quantiles and totals. Two consumers render whatever the
histograms hold — the **Timing Stats** dialog off the ☰ menu, and
`examples/stroke_bench`'s phase table — so **adding a `span!` anywhere makes a row
appear in both** with no list to keep in step.

The layer is forty lines here rather than the `tracing-timing` crate, and not by
preference. That crate's **read** path cannot run in a browser: draining a
`SyncHistogram` means `refresh_timeout`, which calls `std::time::Instant::now()`
before it looks at anything, and on `wasm32-unknown-unknown` that is a panic — "time
not implemented on this platform". `force_synchronize` is the only door to the
recorders and it always passes a timeout, so there is no way round it from outside.
The channel, the phase counter and the per-thread recorder that need that clock all
exist to let many OS threads record without synchronizing, and the browser build has
one thread. So `hdrhistogram` stays (its quantiles are the point) and the
cross-thread machinery goes, taking `crossbeam-channel` out of the wasm binary and
the write-lock off span creation with it.

Consequences worth stating, because they are what the numbers mean:

- **Rows are read against the window, not against each other.** A snapshot carries
  the wall clock it covers, so `count / window` is a rate and `total / window` is a
  share. Nested spans double-count against each other; against the window they do
  not. The two end-to-end rates fall straight out: `frame`'s count is the frame
  rate achieved, `input.sample`'s is how many pointer reports a second actually
  reached the engine.
- **The names are a taxonomy, not a call tree.** `stroke.range` is entered from a
  commit *and* from the live fold, and its histogram aggregates both.
- **Every row is CPU time to prepare work.** WebGPU offers no timestamp query on
  the web, so nothing here says what the GPU then spent executing it. The signals
  for that are the frame-skip counter (`frame.skipped`, from `Renderer::gpu_behind`)
  and, in the benchmark, the instrumented drain (`bench.gpu_wait`). Dividing the
  *wait* among the dispatches inside it still needs bracketing — gate a dispatch
  kind out, re-run, take the difference.
- **The browser's clock is the design constraint.** `performance.now()` is
  quantized to 100 µs in Chromium and a full millisecond in a Firefox that is not
  cross-origin isolated. So the instrumentation wraps *phases* that are
  milliseconds when they matter and never single operations, and the measured
  resolution is reported alongside the numbers — a row of `0.0 ms` has to read as
  "under the clock" rather than as "free". Aggregates survive the quantization that
  individual samples do not, which is why these are histograms and not gauges.

The spans are `info_span!`, because the workspace pins `release_max_level_info` and
instrumentation a release build compiles away cannot answer a question about the
shipped app. They are separated from the log by a **target**, `stark::timing`, and
the separation runs both ways — `TimingFilter::<true>` and `TimingFilter::<false>`,
two settings of one type because they must stay exact complements. The timing layer
takes only that target, or every `info_span!` in
`iroh` would open a row; the console layer takes everything else, because
`tracing_wasm` calls `performance.mark`/`measure` on every span it is shown, which
at a dozen phases a frame is pure waste. `timing::span!` exists so no call site
writes the target by hand, and neither filter's call site restates what a timing
span is.

It is **on for everyone, always** — measured at 234 ns a span, against 1.4 ns for the
same call site with no subscriber installed — because a profile you have to rebuild
to collect is a profile of a build nobody is using. That includes `benches/stroke.rs`,
which prints the phase table under every criterion line it measures: the
instrumentation costs 0.07–0.2% there, two orders under the ~15% those numbers drift
between runs of identical code, so paying it buys "which phase moved" for free.

## 8. Save format & timelapse

The native format is **the serialized action log**:

```rust
pub struct DocumentFile {
    pub app_build: BuildId,            // shaders/algorithm version for fidelity notes
    pub canvas: CanvasMeta,            // color space, and the ground it starts on
    pub actions: Vec<Action>,          // the full, replayable log (each id-tagged)
    pub assets: Vec<(AssetId, Bytes)>, // content-addressed brush images (§6.6)
    pub surfaces: Vec<(SurfaceId, Bytes)>, // the canvas grounds it names (§6.4)
    pub pictures: Vec<(AssetId, Bytes)>,   // the images it places (§23)
    pub checkpoints: Vec<Checkpoint>,  // OPTIONAL cached rasters (see below)
}
```

There is no `format_version`, and there is no tile size. The first went with the
encoding that needed it (§8.1). The second was recorded on every save and then, once
something read it, used to *refuse* a file written against a different `TILE_SIZE` —
on the argument that every tile boundary moves with the stride. But nothing in a log
is expressed in tile units: `TileCoord`, `TileRect` and `Extent2` are not
`Serialize` at all, and every action states itself in canvas px. The stride reaches
only *derived* things — which tiles a footprint quantizes to (§12.6), where an apron
sits (§6.4), whether an action clears a tile cap — and a document whose pixels come
back a little differently is exactly what §19 permits. All the field bought was making
`TILE_SIZE` unchangeable for the life of the format, since the first change would
orphan every file ever saved. An implementation detail is not a fact about a painting.

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
  whole undo timeline is immediately available — undo-after-load, for free. What
  a file does *not* carry is where you were looking at it from: the view is
  per-client session state (§18.1.2), so the frontend frames the piece it just
  replayed, by the rule an export frames one with (§15.6).
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

Serialization uses `serde` over [**carbonite**][carbonite] behind an eight-byte
`STARKDOC` magic and a deflate wrapper — and it carries **no version number at all**,
which is the whole design. A carbonite frame puts the writer's *schema* at its head:
every struct, field name and variant the log used. Loading reconciles that schema
against this build's types **by name**, exactly as reading JSON would, so:

| Change | What an older file does |
|---|---|
| A new **variant**, anywhere in an enum | never contains one |
| A new **field**, with `#[serde(default)]` | filled from the default |
| A new field **without** a default | refused, naming the field it wanted |
| A **renamed** field or variant, with `#[serde(alias)]` | found through the alias |
| A **removed** field | skipped |
| A variant's **shape** changed (unit ↔ fields, payload taken away) | reconciled as a product of fields in order |
| An **integer widened** (`u32` → `u64`) | read and widened |
| A **removed variant** | **refuses the whole document** |

That list is the contract, and `a_file_written_against_an_older_shape_still_loads`
(`io.rs`) is what holds it.

**The last row is the one to know.** A variant is matched by name and an unknown one
has nothing to fall back on, so a file that used it is refused — and because a log is
one value, one retired action in ten thousand takes the document with it. §19 promises
old files keep opening while promising nothing about what they produce, so an action
is retired by **tombstoning** it: keep the variant, hollow the payload out to what is
still read, make it a no-op with an empty footprint. Keep whatever is load-bearing
outside the fold, though — an `Add…` variant's `id` is still owed to `minted_layers`,
or a reload mints that layer id a second time (§17.9). And a tombstone changes what a
log *means* with its shape untouched, so it bumps the ALPN: a file may be read by a
build that disagrees with it, a live session may not (§12.6).

What is left that no encoding can absorb is a rename without an alias, and a
**meaning** changed with the names untouched — reusing a variant for something else,
or narrowing what a field may hold. Nothing in a file can notice the second; it is not
a format problem and never was. One consequence reaches the invariant funnels: a gate
that *refuses* on the way in (`Gradient`, alone in this) cannot be tightened later
without unloading files that were valid when saved, so a new condition there has to
arrive as repair rather than refusal.

**Two doors, and only one is bounded.** `DocumentFile::from_bytes` opens a file the
user owns; `from_untrusted_bytes` opens a peer's snapshot (§12.4) and refuses a body
expanding past 256 MB, since deflate's ratio lets a few kilobytes name as many
gigabytes as they like. One bound served both until it was noticed that nothing caps
how many pictures a document places (§23) — so a dozen photographic placements crossed
it and Stark refused to open a file it had itself saved, which is the one failure a
save format may not have. There is no threat model in which the artist's own file is
the attacker.

One thing that follows is easy to miss: the rules that used to hang off nearly every
`ActionKind` variant — *appended last so postcard keeps decoding older files* — are gone,
along with the variant-order tests that pinned `Place` to `Option<LayerId>`'s
discriminants. A case now goes wherever it reads best, and the tests assert *that*
instead, by reading a `Place` and a `BlendMode` written in another order.

### Where a schema comes from

Every type in the log carries `#[derive(carbonite::Schema)]`, so the schema is assembled
at **compile time** and writing a file discovers nothing. That is a correctness
requirement rather than an optimization, and the reason is worth knowing before adding a
type here.

A schema can also be found by **tracing** — driving a type's `Deserialize` impl with
synthetic values and recording what it asks for. Tracing needs no derive and reads the
real impl, which makes it the obvious default; it is also unusable here. Three types in
the log gate their own invariants in `Deserialize` (`FillOp` and `SelectionOp` clamp
through a `serde(from)` mirror, `Gradient` *refuses* through a `try_from`), and a funnel
that turns away a one-element stop list cannot describe itself to a file. Worse, it
poisons everything containing it: no `Action`, no `DocumentFile`, no gossip payload could
be traced either.

So each of the three states its wire shape outright:

```rust
#[serde(try_from = "Vec<GradientStop>", into = "Vec<GradientStop>")]
#[carbonite(as = "Vec<GradientStop>")]
pub struct Gradient { stops: Vec<GradientStop> }
```

The schema, the columns and the bytes are then the stop list's, nothing drives the
conversion to find that out, and `Gradient::new` stays a refusal — which is the point.
`carbonite(as)` is one declaration for both directions, so a one-sided `serde(from)` is a
compile error naming the attribute to add; that is why the two mirror types are
bidirectional now.

The other half is foreign types, which the orphan rule puts out of reach: only carbonite
can implement its own trait for `glam::Vec2`, and it does, behind a cargo feature. What
is left is four fields whose types belong to iroh or std — an `EndpointId`, two blob
hashes, a `SocketAddr` — each marked `#[carbonite(serde)]`, which describes that one
field by a memoized trace of its own impl and leaves the rest of the type on the fast
path. It is the right tool exactly when the shape should stay whatever the other crate
already writes.

The one place it cannot help is a foreign type that is *itself* untraceable, and
`stark-net`'s session link is the live case: an `EndpointAddr` holds a `RelayUrl` that
parses its own string, so it refuses the empty one. The link spells its addresses in
primitives instead (`ticket.rs`), which is the better design anyway — a pasted link's
format should be Stark's, not a projection of another crate's internals.

Columnar layout is also why the file is small: one field's values sit back to back, so
deflate has runs to find. A thousand small actions come to ~3.4 KB, of which ~1.4 KB is
the compressed schema — a fixed cost a real document amortizes immediately.

The **wire** made the opposite choice, and `stark-net`'s `codec` is where the trade is
written down: a message encodes against a schema both ends already hold, and the ALPN
(`stark/collab/N`) makes a disagreement fail to *meet* rather than decode wrong. A file
is a message to the future and the future cannot be asked to agree; a live session is a
meeting of builds, and a meeting may require agreement. The saving is real — a blob's
header spends a varint per column and an `Action`'s schema is some four hundred columns
wide, so a self-describing presence frame would be four kilobytes instead of two
hundred bytes, on a channel that floods.

Files are alpha (§19). A document written before this change is **named, not migrated**
(`DocError::Legacy`): its body is postcard, which wrote no field names, so those bytes
mean nothing without the exact schema that produced them.

[carbonite]: https://github.com/cbbowen/carbonite

### 8.1 The version history — closed

Thirteen numbered schemas, and what each one cost. Kept because it is the argument for
the format above: every row is a change that would now be free, or a change that would
still cost something and says what.

The old encoding was **positional** — postcard writes fields in order with no names and
an enum by variant index — so three shapes of break recurred: **a field inserted into
an existing variant** (everything after it misreads), **a variant reshaped or removed**
(the same, read from the other end), and **a meaning changed with the layout untouched**
(nothing misdecodes; the file simply is not what an older reader thinks it is). Only the
last of the three would still be a break today, which is the whole point of the table.

| # | What changed | Shape |
|---|---|---|
| 2 | Layer groups (§14): `AddLayer`, `AddMatte`, `MoveLayer` each grew a `carrier` | field inserted |
| 3 | Brush modulation (§6.2) and the deposition tooth (§6.4) on `BrushParams` | appended, but a reader still runs off the end of a brush and into the path behind it |
| 4 | The ground became content-addressed (§6.4): `SurfaceId` went from `Flat \| Linen \| Gesso` to `Flat \| Image(AssetId)`, and `surfaces` joined `assets` | variant reshaped |
| 5 | `StrokeRecord` dropped its `tool` | field removed — and worst placed, sitting second, so every number after it slid along |
| 6 | The bundle may be incomplete (§8, §12.4) | **meaning only** |
| 7 | `FillOp::color` became `paint: Parcel` (§22.4) | variant reshaped |
| 8 | The matte took the same step: `MattePaint`, `MatteRegion::Everything`, and `AddMatte`'s anchor widened to `Place` (§15.4, §15.5) | reshaped + a free widening |
| 9 | A fill's strength became one **coverage**, `opacity` (§18.0.4) | three fields out, one in |
| 10 | `SelectionOp` gained an `opacity` (§6.8) | appended inside the struct |
| 11 | `BlendMode::Drago` gained its bend `k` (§6.3) | payload on a variant that had none |
| 12 | `BrushParams` gained `stretch`, `Modulations` a lane to drive it (§6.6) | appended |
| 13 | Placed images (§23): `PlaceImage` joined the log and `pictures` joined the bundle | appended variant + a third bag |

Five of them are worth more than a row.

**4 — why a ground is a hash and not a name.** The tooth reads the ground, so a
document's pixels depend on a height map the file did not carry and named only
by a label. Open it on a build whose `Gesso.png` had been re-authored and the
strokes came back different, silently, with nothing in the file able to notice. A
file that bundles the ground it was painted on is a file that means one thing.

**5 — why a removal is allowed to be this cheap.** The `tool` field could only
ever hold `Brush` (the selection tools produce a `SelectionOp`, never a stroke)
and nothing read it back, so it was a constant written into every stroke of every
document. §1's rule about inert scaffolding applies to a field that has *stopped*
meaning something just as much as to one that never did.

**6 — walking part of 4 back, and what did not change.** A lean file needs the
app that wrote it, where a version-5 file needed nothing; that is a real cost,
and the reason `save_bytes` still writes everything unless told what may be left
out. But version 4's problem was that a ground was a *label* resolved through a
table. A ground is a content id now, the id stays in the file, and content that
does not hash to it is refused rather than substituted — so the failure mode of a
re-authored asset is a document that will not open, not one that opens wrong.

**9 — why coverage replaced a height.** Coverage is `1 − exp(−K·opacity·height)`,
so the brush's entire flow at full alpha covers 95% and no setting says "and the
rest". Naming the *coverage* and inverting the law for the mass — `slab.wesl`'s
inversion, already there for blended merges — makes 1 mean opaque and ½ mean
half, and leaves a fill only one control to disagree with.

**13 — the last bump, and the one that would now be free.** Appending `PlaceImage` to
`ActionKind` cost nothing even then (an enum encoded by index), but appending
`pictures` to `DocumentFile` did: a version-12 file simply stopped before that field,
so a version-13 reader ran off the end of it. Today that field is a `#[serde(default)]`
away from costing nothing — an empty bag is exactly what a document with no placed
images means. The entry is worth keeping for what it says about §23: a placed image is
*content* named by the log and carried beside it, exactly as a brush shape and a ground
are, so the third kind cost a third bag and no new mechanism.

**11 — the most dangerous shape on the list, and the clearest argument for the
change.** A version-10 `SetLayerBlend(id, Drago)` encoded as a bare variant index, so a
version-11 reader took the four bytes *after* it — the next action's — as the bend. The
misread bend is a plausible float and the actions it ate are a plausible log, so nothing
downstream could notice; the version number was the only thing that refused it. Reading
by name, a bend `k` on a variant that had none is a `#[serde(default)]` and an older
file says so itself. The parameter is on the variant rather than in a settings struct
beside the mode because a `Multiply` layer has no `k` to store, and a mode cannot
disagree with its own settings about which mode it is.

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
- **The panel stack stays away until it is reached for.** Fading back on release
  put the panels over the painting at the one moment the artist was looking at
  what they had just drawn, so the stack alone keeps its fade after the gesture
  (`AppState::panels_asleep`, set by `input::end_interaction` where — and only
  where — the fade was actually in force) until the pointer enters the *column*
  it lives in. That column is a full-height slice of the window, which the stack
  itself cannot be: it is exactly as tall as its panels and has to stay that way
  for its own scroller, so `.panel-wake` is a separate box, mounted only while
  the panels are asleep and no gesture is in flight — the two conditions that
  keep an invisible box over the painting from swallowing a stroke's moves or the
  release that ends it. It answers what it does take: the first press or wheel in
  a sleeping column brings the panels back rather than being lost, which is also
  the only way a touch-only hand can ask for them. Opening a panel wakes the
  stack too (`layout::open_panel`, the one door), or the command would tick a
  menu entry and show nothing. Nothing else sleeps: the rail and a mode's bars
  and handles live elsewhere on screen, and latching those off until the pointer
  had visited the right edge would be controls you cannot reach by reaching for
  them.
- **Every panel starts closed, and what is open follows the browser.** The
  opening screen is the painting and nothing else; the panels that come back next
  visit are the ones the artist actually reached for (`layout::stored_hidden`,
  keyed `stark.panels.v1`). The two halves are one decision — a set of panels
  chosen for you is only tolerable because it resets every visit, and once the
  choice sticks the honest starting point is none. What keeps "none" from meaning
  "hidden" is the tour: the Color panel arrives on the second stroke and the wake
  gesture is explained on the third (§24.5). Durability is structural rather than
  remembered — `layout::set_open` is the only thing that writes the hidden set and
  it persists after every change, so a new way to close a panel is durable without
  its author thinking about storage, the same move `settings::SettingToggle` makes
  for the preferences. The **open** set is what is written, not the hidden one, so
  a panel added in a later release arrives closed like everything else instead of
  appearing unbidden in the stack of every existing user.
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
- **The layer panel's thumbnails go back the other way, and the reasons invert**
  (`layer_thumbs.rs`, §14.6). They are pictures of the live document like the
  navigator's, so it would seem to follow that they want surfaces too — but there is
  one navigator and there are as many of these as the document has layers, so a
  surface each is a WebGPU context and a swapchain per row; the rows are moved and
  re-keyed by the drag that reorders them, and a CSS `background-image` survives a
  node moving where a bound surface is per-node state to rebind; and the frame of
  latency a surface buys off is worth nothing on a 20 px picture refreshed once per
  commit. So these are `Engine::export_view` readbacks into `data:` URLs, exactly as
  the *brush* thumbnails are. Three consumers, three answers, from one question asked
  each time: does this render repeat, how many of it are there, and who is waiting.
  - **The engine change was one parameter.** `composite_groups` had answered "just
    this layer" since the eyedropper needed it (§18.0.2) and `DrawKey` already keyed
    on it; nothing had ever asked for it as a *picture*, so `render_view` passed
    `None`. Threading it through is the whole of the feature engine-side, which is
    what a seam in the right place looks like from outside.
  - **The cost that is real is the draw cache's single slot.** An isolated render
    evicts the screen's list, and rebuilding that clones a tile handle per visible
    tile per layer — so generation is paced one row at a time and stands down while
    `canvas_active`. The cache key is the layer's own tile revision rather than
    `doc_revision`, so a stroke re-renders one row instead of all of them.
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
  its cache key is the brush snapshot alone. Each row's picture is two
  half-canvas fills (the ground is all paint, so smearing and lifting read), one
  replayed stroke and one small `Engine::export_view` readback on that one kept
  engine, generated in the background and cached per session. The key being the
  brush is what lets the cache have two viewers for the price of one: the preset
  library's rows and the quick-brush rack the number keys draw (§18.1.8) show the
  same picture of the same brush, and a slot that came from a preset is never
  rendered twice. The generator therefore belongs to the app **root**, not to
  either viewer — the Brush panel closes, and the rack's overlay exists only
  while a key is held, which is far too late to start rendering what it is there
  to show.
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
- **Timing Stats is a dialog, not a panel** (`timings.rs`, §7.1). The same
  argument as settings run from the other end: a live frame-rate readout beside the
  canvas is a thing to watch *instead of* painting, and the histograms behind it
  keep accruing whether or not anyone is looking — so it belongs in the ☰ menu with
  the other things read when a question comes up. It renders whatever rows the
  engine's histograms hold rather than a list of phases, which is what keeps it from
  becoming a second copy of the instrumentation; it polls twice a second while open
  and stops when it closes (a `use_future` in its own scope — the opposite of what
  the collaboration pumps need, and for the opposite reason).

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


