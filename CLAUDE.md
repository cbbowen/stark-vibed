# Stark

A GPU-accelerated 2D painting application in Rust, focused on:

- **Beautiful, natural brush strokes** that carry channels beyond color — paint
  height, and the wet-mixing of what is already on the canvas. Color blends in a
  perceptual space (Oklab) or in real pigment (Mixbox), so work can look like the
  oil paintings of the old masters.
- **Performant painting and compositing** built on WGPU, tiled for an infinite
  canvas, responsive enough that the tool disappears.
- **Powerful digital tools** — infinite canvas, complete undo history, replayable
  documents, peer-to-peer collaboration.

## The one idea everything else follows from

**The document is a list of actions, not a bag of pixels.** Pixels are a derived,
cached view of a replayable, deterministically-ordered action log. The native
save format, undo-after-load, timelapse, golden tests and CRDT collaboration all
fall out of that single decision.

Two structural consequences worth knowing before reading any code:

- **`DocState` is cheap to clone** — persistent (`rpds`) maps of `Arc<GpuTile>`
  handles, never pixels. Tiles are copy-on-write, so history retention drives GPU
  memory reclamation for free.
- **One rendering path, used three ways** — live painting, replay and goldens all
  go through the same deterministic renderer. `preview == committed` is the
  invariant that states it.

## Design docs

Full design lives in [docs/](docs/). **Section numbers (`§n.m`) are stable and
are cited from ~1000 places in the source** — keep them resolving.

| Doc | Sections | What's in it |
|---|---|---|
| [architecture.md](docs/architecture.md) | §1–§5 | Principles, crate layout, the command/action boundary, the history & timeline model |
| [brush.md](docs/brush.md) | §6.1, §6.2, §6.6, §6.9, §6.11 | Tiles and channels, path fitting, swept-segment stamping, the wet-mixing dynamics loop, brush shape assets, drag-and-hold shape assist, stroke smoothing (the towed tip) |
| [rendering.md](docs/rendering.md) | §6.3–§6.5, §6.7, §6.10 | The three compositing passes, blend modes, the media/lighting pass, aprons and the canvas surface, Oklab and Mixbox, the generated uniform mirrors |
| [selection.md](docs/selection.md) | §6.8, §16 | The soft-mask coverage field every tool acts through, fill, and transform |
| [layers.md](docs/layers.md) | §14, §15 | Groups and clipping as one mechanism; merging a layer down without changing the picture; matte layers, framing and export |
| [filters.md](docs/filters.md) | §21 | Filter layers: adjustment as a layer, where it sits *is* its scope, the color filter, the spectral chromatic aberration, the gradient map |
| [gradients.md](docs/gradients.md) | §22 | Gradients: stops fitted from a line traced through the painting — the eyedropper generalized — the browser-local library of them, the gradient fill (a `FillOp` parcel that varies with position), and the shared gradient bar that also grades matte paint |
| [engine.md](docs/engine.md) | §7–§11 | The actor target, the timing histograms, the save format, golden tests, the extensibility map, the Dioxus frontend |
| [collaboration.md](docs/collaboration.md) | §12, §17 | The CRDT over the action log, iroh transport, owned selections, the presence roster |
| [roadmap.md](docs/roadmap.md) | §13, §18, §19 | Build order and status, the gap analysis against the prior art, file-format stability |
| [drawing-guides.md](docs/drawing-guides.md) | §20 | The perspective grid: one projective camera behind 1/2/3-point, the fan parametrization, the guide overlay pass, aligning strokes to an axis and to circles on a plane, the stereographic fisheye lens |

§6 — "rendering the canvas" — is the one chapter split across files: the stroke
path is in [brush.md](docs/brush.md), the compositing path in
[rendering.md](docs/rendering.md), and the mask every tool acts through in
[selection.md](docs/selection.md). A bare `§6` citation means the chapter.

These absorbed the former `DESIGN.md`, `GOALS.md`, `BLEND_MODES.md`,
`CLEANUP.md`, `FRAME_DESIGN.md`, `GROUP_DESIGN.md`, `MISSING_FEATURES.md`,
`PANEL_UI_DESIGN.md`, `PEER_DESIGN.md`, `STABILITY.md` and
`TRANSFORM_DESIGN.md`. §1–§13 keep `DESIGN.md`'s original numbering; the feature
docs became §14–§19.

## Where things live

```
crates/
  stark-assetid/   what a content id *is*: decode, cap, hash for brush shapes
                   and canvas grounds. No GPU, so a build script can compute one
                   — which is what lets the frontend know a bundled asset's id
                   before fetching it. The file format's identity contract (§19)
  stark-model/     the document: the action log, its vocabulary, the save format
                   and the presence wire. No wgpu, no shaders, no build script
    document/      actions, footprints, and the halves of layer/selection/fill/
                   transform that are facts rather than tiles
    io.rs          the save format, which *is* the action log (§8)
  stark-engine/    the derived view — no UI, no windowing; compiles to wasm
    document/      DocState, the timeline, and the fold that fills them
    gpu/           tile pool, stroke renderer, compositor, readback
    filters.rs     the host's reads of the generated shader mirror (§6.10)
  stark-shaders/   WESL sources, the build step that links them, and the host
                   mirrors generated from them (§6.10)
    shaders/lib/   binding-free leaves — a module here may NOT declare a binding
    build/         the generator: WESL declarations -> Rust structs/consts/attrs
  stark-testdata/  recorded pen input + asset paths; dev-only
  stark-net/       iroh transport ↔ the replicated timeline
  stark-ui/        Dioxus 0.7 frontend; owns the wgpu::Surface
    index.html     the page shell: links the manifest, registers the worker
    public/        manifest, service worker, launcher icons — an installable,
                   offline-capable app (§11). Copied to the SITE ROOT unhashed,
                   unlike `assets/`, renamed by content hash
vendor/            third-party, EXCLUDED from the workspace
```

**Dependencies point one way: ui → engine → model.** `stark-net` depends on the
model *only* — it moves logs and assets and never names an engine type, which is
what the split bought (§2). Which side a type belongs on has a mechanical answer:
if it is `Serialize` it is a fact about the document and lives in the model; if it
holds a tile it is a cache and lives in the engine. `AssetId`/`AssetStore`,
`SurfaceId`/`Surface`, `LayerId`/`Layer`, `Action`/`DocState` are all the same pair.

The one crack in that: large image assets live in `stark-ui/assets/` because
Dioxus's `asset!` rejects paths outside its own crate, and the engine's *tests* read
them from there through `stark_testdata::assets` — the single module that breaks if
the frontend reorganizes.

## Commands

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check -p stark-ui --target wasm32-unknown-unknown
# the second configuration (§6.7): no Mixbox, and so no CC BY-NC 4.0 code at all.
# Also the ONLY gate that catches a `#[cfg]` that drifted off the `use` it
# guarded — the default build is green either way.
cargo clippy --workspace --all-targets --no-default-features \
  --features stark-net/webrtc -- -D warnings
dx serve --web -p stark-ui                  # run it (needs a WebGPU browser)
cargo bench -p stark-engine --bench stroke    # criterion; the dynamics perf gate
# where a stroke's time goes, phase by phase (§7.1) — seconds, not minutes
cargo run --release -p stark-engine --example stroke_bench
```

Both perf tools print a **phase table** per configuration (§7.1), and the same rows
are live in the app behind ☰ → Timing stats. Which question each answers:
`--bench stroke` is the regression **gate** — totals, confidence intervals, a saved
baseline — with the split under each line; `--example stroke_bench` is the quick
look, seconds not minutes, and the only one that accounts for the GPU drain
(`bench.gpu_wait`). Read the shares, not the totals: this class of box drifts ~15%
between runs of identical code.

CI (`.github/workflows/ci.yml`) runs fmt, clippy `-D warnings`, the test suite
and the wasm build. Vendored crates are excluded from the workspace, so run their
suites by hand when the vendored code changes:

```sh
cargo test --manifest-path vendor/iroh-webrtc-transport/Cargo.toml
```

Test-suite switches: `STARK_ALLOW_NO_GPU=1` opts into skipping when there is no
adapter — a missing GPU is otherwise a **failure**, because a skipped test still
reports `ok` and would take the whole golden/seam/dynamics suite green having
rendered nothing. `STARK_SKIP_GOLDEN=1` renders without comparing pixels (what CI
uses, since goldens are adapter-specific). Deleting a golden re-blesses it.

The workspace is on **nightly** for exactly one reason: `history`'s
`associated_type_defaults`.

## Rules that are easy to break silently

- **Every `apply` must touch only what its `Footprint` declares** (§12.6). A
  false conflict costs the commutation fast path; a missed one silently diverges
  peers, and pixels cannot show which path ran.
- **Every pass that writes tiles must be a pure function of canvas position**
  (§6.4). That is what makes a tile's apron bit-identical to its neighbour's
  interior without a copy pass. `tests/seam.rs` guards it.
- **A stroke's per-segment deposit must be additive in `τ` or of the form
  `1 − exp(−k·τ)`** (§6.2). Any other shape makes stroke weight depend on the
  *number* of segments — invisible under uniform sampling, immediate under
  adaptive.
- **Conserve `height`, never alpha** (§6.1). Color alpha is per-unit opacity, a
  material property; the amount of paint is the height channel. The two meet only
  in the slab law `1 − exp(−K · opacity · thickness)`.
- **Postcard encodes struct fields in order and enums by index** (§8). Appending
  a variant is safe; inserting a field into an existing variant is a wire-format
  break.
- **An `impl` that crosses the crate boundary has to move or become a function**
  (§2). The orphan rule is not an obstacle here, it is the boundary reporting
  itself: `history::Action` became `Materialize`/`Logged` in the model,
  `ColorSpaceId::make` and `SelectionOp::shader_params` became free functions in
  the engine, and in each case the side that ended up owning the work is the side
  that always should have.
- **Anything reading the generated shader mirror belongs with the shaders**
  (§6.10). `stark-model` compiles without `stark-shaders` at all, so a constant it
  wants from a `.wesl` declaration is a signal the item is the engine's —
  `filters.rs` is where those collected.
- **Never transcribe onto the host what a `.wesl` file already states** (§6.10).
  Uniform lanes, constants, vertex formats, binding indices and binding *types*
  are all generated from the shader's own declaration —`MIRRORS`, `CONSTS`,
  `VERTEX` and `BINDINGS` in `stark-shaders/build.rs`. Every one of those lists
  exists because a hand-written second copy drifted, and the drift was invisible
  until it was a picture. What is genuinely the host's — a name, a step mode,
  whether *this* pass samples *that* texture — is worth writing by hand for the
  same reason: the shader does not say it.
- **A new engine method that mutates state and returns nothing is a bug** — it
  should be a command (§4). Operations that must *answer* are a named request
  tier, so they stay countable when the engine moves behind a channel.

## Conventions

- **The test suite is slow — run it once.** Redirect a single
  `cargo test --workspace` to a file and grep that file; do not re-run to get
  names after counts.
- **Cite sections, not line numbers**, when referring to the docs from code
  (`§6.4`).
- **When a model is wrong, fix the model and re-bless the goldens.** No
  compensating fudge constants.
- **Do not add inert scaffolding.** If a field, slider or shader hook cannot yet
  change a pixel, leave it out and add it with the implementation. `tooth`,
  `drag`, `wetness` and the `wet` channel were all deleted rather than
  carried; each is a local change to reintroduce when built — `bleed` came back
  that way as the real lateral-diffusion axis, and `tooth` as the deposition gate
  (§6.4), each with the model its placeholder never had.
- **Rule out a class rather than enumerate its instances.** Where a guarantee can
  be made structural — ownership derived from the action id, a representation
  that cannot express the wrong thing — it is, instead of a check a call site
  could forget.
- **Line endings are pinned** (`.gitattributes`, `text=auto eol=lf`); tooling has
  silently rewritten files to CRLF here before.
