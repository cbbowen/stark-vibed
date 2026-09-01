# Glossary

One name per thing, and the file that owns it.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> This is the one doc with no section numbers — nothing cites it as `§n.m`,
> because it settles vocabulary rather than design.

This exists because several of these had three or four names each. The rule is
the same one the rest of the docs follow: **a term means one thing, and the file
in the "Defined in" column is where its meaning is settled.** If a name here is
wrong for what it describes, change the name — do not add a synonym.

The "Not to be confused with" rows are the reason this file exists at all. Every
one of them is a word that was, at some point, used for two things at once.

## The canvas

| Term | What it is | Defined in |
|---|---|---|
| **substrate** | The physical canvas a document is painted on: the material paint sits on, with both a grain and a color. The word covers the whole thing; the two rows below are its parts. | [`stark-model/src/substrate.rs`](../crates/stark-model/src/substrate.rs) |
| **`SubstrateId`** | *Which* substrate — `Flat` (procedural, no bytes) or `Image(AssetId)`, a height map named by the hash of its canonical decoded form. Document state, saved and replicated. | [`stark-model/src/substrate.rs`](../crates/stark-model/src/substrate.rs) |
| **`SubstrateScale`** | How large that height map is laid, as a quantized percentage of its natural size. Document state, because it decides what the tooth bites as surely as which substrate does. | [`stark-model/src/substrate.rs`](../crates/stark-model/src/substrate.rs) |
| **`Substrate`** (engine) | The `(SubstrateId, SubstrateScale)` pair — the bake key, and what every caller downstream passes around. The bytes are keyed by the id alone; the *bake* is per scale. | [`stark-engine/src/gpu/substrate.rs`](../crates/stark-engine/src/gpu/substrate.rs) |
| **`SubstrateMap`** | The baked GPU resource: height in `R`, the rise ahead in `GB`, plus the bearing table. One `Substrate` bakes one of these. | [`stark-engine/src/gpu/substrate.rs`](../crates/stark-engine/src/gpu/substrate.rs) |
| **substrate color** | The flat color under everything, straight sRGB. Document state (§15.5) — the paper color of the painting, so it is logged and saved. Field `DocState::substrate_color`. | [`stark-engine/src/document/state.rs`](../crates/stark-engine/src/document/state.rs) |
| **tooth** | The deposition gate (§6.4): where paint lands and where it bridges. Two knobs on the brush, never one — **`tooth_give`** is how far the tip settles in — 1 is full give and no tooth at all, 0 the driest tip, quoted that way round because a modulation only scales *down* and the mapping this axis exists for is pressure — and **`tooth_softness`** is how wide the contact transition around that is (what the tip is *made of*: acrylic narrow, charcoal wide). A CPU mirror of `paint_common.wesl`. | [`stark-engine/src/gpu/substrate/tooth.rs`](../crates/stark-engine/src/gpu/substrate/tooth.rs) |
| **rise** | How much higher the substrate stands one `TOOTH_REACH` ahead along a canvas axis. The whole of the deposition model — not a level set of the height, but the slope along the tip's own travel. | [`stark-engine/src/gpu/substrate/tooth.rs`](../crates/stark-engine/src/gpu/substrate/tooth.rs) |
| **bearing** | The fraction of a substrate a tip at a given tooth actually bears on, averaged over the rise distribution — what lets the *tool* book its half of a smear. | [`stark-engine/src/gpu/substrate/tooth.rs`](../crates/stark-engine/src/gpu/substrate/tooth.rs) |
| **grain** | The substrate's *fine structure* — the threads and faces a tooth catches on, as in canvas grain or paper grain. The substrate is the thing; the grain is what it is made of. | [`stark-shaders/src/shaders/lib/paint_common.wesl`](../crates/stark-shaders/src/shaders/lib/paint_common.wesl) |
| **relief** | The height field the media pass shades: the substrate height *plus* the paint's thickness. `relief_normal` is its gradient. Not the substrate alone. | [`stark-shaders/src/shaders/media_common.wesl`](../crates/stark-shaders/src/shaders/media_common.wesl) |

Retired synonyms for **substrate**: *weave*, *ground*, *canvas surface*. None of
these should appear in code again.

### Not to be confused with

| Term | What it is instead | Defined in |
|---|---|---|
| **backdrop** | What a layer composites *against* — the accumulated destination below it. A blend mode and a clip are both statements about a backdrop, which is why they are vacuous where there is none. | [`stark-model/src/document/layer.rs`](../crates/stark-model/src/document/layer.rs) |
| **`Background`** | An *export* choice: fill with the substrate, or carry the paint's own alpha out (`Transparent`). A render option, never document state. | [`stark-engine/src/engine/render.rs`](../crates/stark-engine/src/engine/render.rs) |
| **backing** | The §15.5 underpainting: an `Everything` matte born at the bottom of the stack, under the painting. A *layer*, unlike the substrate color — paintable, movable, undoable. | [`stark-ui/src/panels/frame.rs`](../crates/stark-ui/src/panels/frame.rs) |
| **surface** | Reserved for four things that are not the canvas: wgpu's swapchain `Surface`, the warp mesh's mathematical surface (§16), a *UI surface* (a pointer-receiving region of the chrome), and a module's *public surface*. | [`stark-ui/src/render.rs`](../crates/stark-ui/src/render.rs) |
| **ground** | Reserved for the §20 perspective **ground plane** and for English idiom ("on the grounds that", "ground truth"). Never the canvas. | [`stark-model/src/document/guide/mod.rs`](../crates/stark-model/src/document/guide/mod.rs) |
| **ray** | Two things in §20, and always qualified. The **eye's ray** is a *direction* in camera space, what a canvas point means to the camera (`PerspectiveGuide::ray`) — the one place the lens enters. A **cursor ray** is a *curve on the canvas*: where the world line through the pointer, parallel to one axis, images (§20.9). The second is derived from the first, which is why they share a word at all. | [`stark-model/src/document/guide/camera.rs`](../crates/stark-model/src/document/guide/camera.rs) |

## Input and the fitted path

| Term | What it is | Defined in |
|---|---|---|
| **tolerance** | The input device's own positional resolution, in canvas px — how finely it can say *where*. One word in prose and in code, held to a usable range by the one door `clamp_tolerance`. It was also called "the grain"; that word now belongs to the substrate alone. `flatten_tolerance` is a *different* quantity — how finely a curve is flattened — and keeps its qualifier. | [`stark-engine/src/path/fit.rs`](../crates/stark-engine/src/path/fit.rs) |
| **rope** | How far the towed tip lags the pointer, in canvas px — the stroke-smoothing knob (§6.11). | [`stark-engine/src/tow.rs`](../crates/stark-engine/src/tow.rs) |
| **tow** | The smoothing itself: the tip dragged behind the pointer on a rope, emitting samples as it bends. | [`stark-engine/src/tow.rs`](../crates/stark-engine/src/tow.rs) |
| **report** | One raw pointer event as the frontend hands it over. Distinct from an *emission* (what the tow produces from it) and from a *sample* (what the fitter consumes). | [`stark-engine/src/session.rs`](../crates/stark-engine/src/session.rs) |

## Strokes and stamping

| Term | What it is | Defined in |
|---|---|---|
| **tip** | The brush at one instant: a position, a radius, an angle. `BrushShape` is the tip's shape. | [`stark-model/src/document/brush.rs`](../crates/stark-model/src/document/brush.rs) |
| **`BrushConfig`** | The brush as the *frontend* carries it: the shared tip knobs beside **every** effect's configuration with one in force, plus the smoothing feel (§6.11) — what the live brush signal and the preset library hold. `BrushParams` is its projection: the record's shape, the effect in force alone, the hand's pigment stamped into whichever laying effect goes down. | [`stark-ui/src/brush_config.rs`](../crates/stark-ui/src/brush_config.rs) |
| **transient** | The half of a brush that is the hand's own state rather than the tool's: the size, the flow — the overall rate of whichever effect is in force — and the painting color (`Transient`, a value of its own beside `BrushConfig`, which is the **durable** half: what the tool *is*). A preset stores both halves; a quick slot is a preset's name beside a transient half of its own (§18.1.8). The color has one rule the other two knobs do not: it never arrives with a tool (`presets::wear` keeps the hand's), and a slot decides "did anything change?" with it set aside. Not the transient *rack*, which is the one drawn only while a key is held. | [`stark-ui/src/brush_config.rs`](../crates/stark-ui/src/brush_config.rs) |
| **extent** | The texels a tip covers at one instant — the area the deposit may reach. `extent_cell` is the square the exchange is evaluated over. Formerly "footprint", which now means only the two things below. | [`stark-engine/src/gpu/stroke/budget.rs`](../crates/stark-engine/src/gpu/stroke/budget.rs) |
| **sweep** | One segment's swept capsule: the tip dragged from one sample to the next. | [`stark-engine/src/gpu/stroke/segments.rs`](../crates/stark-engine/src/gpu/stroke/segments.rs) |
| **piece** | A cut of one stroke, sized so the stamp loop fits its budget. One stroke becomes several pieces. | [`stark-engine/src/gpu/stroke/region.rs`](../crates/stark-engine/src/gpu/stroke/region.rs) |
| **region** | The rectangle a piece renders into — measured by the chunker and allocated by the render from one `Coverage`, so the two cannot disagree. | [`stark-engine/src/gpu/stroke/region.rs`](../crates/stark-engine/src/gpu/stroke/region.rs) |
| **stencil** | The bleed's finite-difference kernel: its taps, their shares, and the reach it is solved at. A numerical-methods stencil, not a mask. | [`stark-engine/src/gpu/stroke/dynamics/bleed.rs`](../crates/stark-engine/src/gpu/stroke/dynamics/bleed.rs) |
| **wet brush** | A brush whose effect is `Wet` (§6.2): it lays paint *and* works what is already there through the sequential lift/deposit loop. Its own effect rather than a `Paint` at other rates, because the two are different tools with different available features — a wet stroke mixes with the canvas and carries a reservoir, and its deposit is point-sampled where paint's is antialiased through the pixel-footprint filter. The variant *is* the render path: `Wet` loops, `Paint` and `Erase` sweep. | [`stark-model/src/document/brush.rs`](../crates/stark-model/src/document/brush.rs) |
| **eraser** | A brush whose effect is `Erase` (§6.12) — `BrushEffect`, the tool's identity as a sum type: the stroke removes *visible* opacity through the slab law, capped per stroke at the dial. Not the `lift` axis, which is a **scraper** — it takes the amount, conserving paint, and reads in height rather than in what the eye sees. | [`stark-model/src/document/brush.rs`](../crates/stark-model/src/document/brush.rs) |
| **liquify brush** | A brush whose effect is `Liquify` (§6.13): the stroke drags the picture itself — every channel resampled along the travel, structure carried rather than mixed. The domain's own word (Photoshop, Procreate, Clip Studio), kept because "warp" already belongs to the selection transform's mesh (§16.9); its per-texel motion is the **follow**, a fraction of the tip's travel. Not a smudge, which trades paint through a reservoir and conserves height. | [`stark-model/src/document/brush.rs`](../crates/stark-model/src/document/brush.rs) |
| **transparency mass** | What an erase stroke's sweep accumulates (§6.12): the optical mass of a parcel of "nothing" at per-unit opacity 1, so mass and height are one number. Its slab-law coverage is the fraction the stroke uncovers. | [`stark-shaders/src/shaders/stamp.wesl`](../crates/stark-shaders/src/shaders/stamp.wesl) |
| **resolve** | Box-averaging supersampled texels down to their pixel — the operation, wherever it runs. Two passes do it: the **presentation resolve** (`resolve.wesl`, §6.4) on a zoomed-out render, and the integrate's box resolve of a gated stroke's 2× parcel (`integrate.wesl`, §6.2), where height resolves as the mean and the mass as the visible-equivalent log-mean-exp. | [`stark-shaders/src/shaders/resolve.wesl`](../crates/stark-shaders/src/shaders/resolve.wesl) |

### Not to be confused with

| Term | What it is instead | Defined in |
|---|---|---|
| **`Footprint`** | What one action reads and writes, tile-quantized — the CRDT conflict set (§12.6). Nothing to do with a stamp's area. A footprint may claim too much; it may never claim too little. | [`stark-model/src/document/footprint.rs`](../crates/stark-model/src/document/footprint.rs) |
| **memory footprint** | Bytes held. Always qualified with "memory" so it cannot be read as either of the above. | [`stark-engine/src/gpu/composite/resolve.rs`](../crates/stark-engine/src/gpu/composite/resolve.rs) |
| **nib** | Only ever a simile — "a chisel nib", "like a calligraphy nib" — describing a real instrument whose shape a brush emulates. The modeled thing is a **tip**. | [`stark-model/src/document/brush.rs`](../crates/stark-model/src/document/brush.rs) |

## Paint and compositing

| Term | What it is | Defined in |
|---|---|---|
| **height** | The amount of paint at a texel. The conserved channel (§6.1) — a smear moves it, never creates it. | [`stark-model/src/document/brush.rs`](../crates/stark-model/src/document/brush.rs) |
| **opacity** | Two things, and the context says which. *Per-unit* opacity: a material property of resident paint (a tile's color alpha), meeting height only in the slab law `1 − exp(−K · opacity · thickness)`. *The effect's* opacity (`BrushEffect::opacity`, §6.2/§6.12): the per-stroke ceiling on what a saturated stroke lays or removes — the digital artist's dial, beside the flow that is the rate. The brush color itself carries no alpha. | [`stark-model/src/document/brush.rs`](../crates/stark-model/src/document/brush.rs) |
| **coverage** | A per-texel fraction in `[0,1]`. Used for a brush shape's mask, and for the selection field every tool acts through (§6.8). | [`stark-model/src/document/selection.rs`](../crates/stark-model/src/document/selection.rs) |
| **parcel** | *What* paint a fill lays — one color everywhere, or one that varies with canvas position (§22.4). Never *how much*: that is the fill's opacity. | [`stark-model/src/document/fill.rs`](../crates/stark-model/src/document/fill.rs) |
| **matte** | A layer that fills a region with flat paint: the **frame** (`OutsideRect`) or the **backing** (`Everything`). One mechanism, two uses (§15). | [`stark-model/src/document/layer.rs`](../crates/stark-model/src/document/layer.rs) |
| **layer frame** | The coordinate space a layer's tiles are keyed in, placed on the canvas by `Layer::translation` — whole pixels (§14.12). Always qualified in prose, because the bare **frame** is the matte's; in code the qualifier is the context (`Layer::translation`, an action's `frame`). | [`stark-engine/src/document/layer.rs`](../crates/stark-engine/src/document/layer.rs) |
| **float** | The author's selection cut into a child layer at the foot of its source's stack, so a drag moves a frame instead of resampling paint (§16.12). The noun; as a verb, *to float a selection*. Not `f32`, which is never called a float in prose here. | [`stark-model/src/document/action.rs`](../crates/stark-model/src/document/action.rs) |
| **residual** | The part of a color a three-channel latent cannot carry, kept alongside it so Mixbox round-trips exactly (§6.7). | [`stark-engine/src/gpu/composite.rs`](../crates/stark-engine/src/gpu/composite.rs) |

## The document and the log

| Term | What it is | Defined in |
|---|---|---|
| **action** | An entry in the log — the document itself (§4). Deterministically ordered, replayable, never deleted (retired ones are tombstoned). | [`stark-model/src/document/action.rs`](../crates/stark-model/src/document/action.rs) |
| **command** | What the frontend *asks for*. A command may produce an action, several, or none; the boundary between the two is §4. | [`stark-engine/src/command.rs`](../crates/stark-engine/src/command.rs) |
| **deed** | What the guided tour reads off the `dispatch` seam: not what a command *says*, but what it *changed* (§24). Only `tutor.rs` uses this word. | [`stark-ui/src/tutor.rs`](../crates/stark-ui/src/tutor.rs) |
| **timeline** | The history structure: the ordered actions plus what is currently materialized. `DocState` is its derived, cached view. | [`stark-engine/src/document/timeline.rs`](../crates/stark-engine/src/document/timeline.rs) |
| **roster** | A list the document or session owns in order — the presence roster (§17), the drawing-guide roster (§20.5). Always qualified. | [`stark-engine/src/peer.rs`](../crates/stark-engine/src/peer.rs) |
| **ledger** | Browser-local, per-viewer state that is *not* in the document — what the tour has already said. | [`stark-ui/src/tutor.rs`](../crates/stark-ui/src/tutor.rs) |

## Naming rules this file implies

- **A word means one thing across the workspace**, not one thing per module. Where
  two concepts genuinely deserve the same English word, one of them takes a
  qualifier permanently (*memory* footprint, *flatten* tolerance) rather than
  relying on context.
- **Prefer the word the domain already owns.** `backdrop` is the W3C compositing
  term; `stencil` is the finite-difference one; `grain` is what a canvas's own
  texture is called. Coining is a last resort.
- **A uniform lane is a byte layout, not a term.** `view_a`, `sub_b` and friends
  are named for where they sit; the comment beside each says what it carries. They
  are not vocabulary and are not listed here.
- **Renaming stops at anything already written down.** A `localStorage` key, a
  golden's file name and a log variant are all data somebody's machine is holding.
  A key is simply left alone (`Store::Substrates` still spells its key
  `stark.grounds`); a golden is renamed together with its file; a log variant is
  renamed with a `#[serde(alias)]` for the name it used to have, or every older
  file stops loading (§8).
- **User-facing strings are not covered by this file.** They follow the same
  vocabulary where it reads naturally, but a label may say something plainer.
