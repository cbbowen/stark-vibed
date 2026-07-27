# Framing, mattes, and export

The infinite canvas has never had to answer *what rectangle is the piece?* Export
forces the question, and the answer shapes composition as much as output. This
document specifies the answer: a **matte layer** — a layer whose content is a
region and a fill rather than a map of tiles.

It is written against the code as it stands
([layer.rs](crates/stark-core/src/document/layer.rs),
[state.rs](crates/stark-core/src/document/state.rs),
[composite.rs](crates/stark-core/src/gpu/composite.rs),
[media_common.wesl](crates/stark-shaders/src/shaders/media_common.wesl)), and
several of its decisions are forced by that code rather than chosen — those are
called out where they occur.

## 1. The stance: a frame is a suggestion, not a wall

A frame **clips nothing**. Paint runs past it, it slides around afterward, and
one painting may carry several. Photoshop's crop is destructive and Procreate's
canvas is fixed at creation; ours is a decision you get to defer, which is how
framing actually works at an easel. This is the whole reason the infinite canvas
earns its keep rather than merely being unusual.

Two consequences worth stating because they are load-bearing later:

- **Onboarding.** "New document → 1920×1080" seeds a *frame*, not a canvas. A
  Photoshop refugee gets the familiar bounded feeling; the boundary is soft.
- **Overpaint is a technique.** Painting past the edge and letting the matte
  cover it is exactly how comic gutters and traditional inking work. The frame
  hiding your overshoot is a feature, not a compromise.

## 2. The representation: a region with a value at infinity

A frame is not "a rect plus a scrim." It is a **region and a fill**, where the
region has a defined value at infinity. That type already exists here:
[`Selection`](crates/stark-core/src/document/selection.rs) is a coverage field
over the infinite plane with an `outside` flag, and §6.8 argues at length why
that flag is what makes an unbounded canvas tractable.

One field then gives three features:

| Region | Position in stack | What it is |
|---|---|---|
| everywhere **except** a rect | top | the frame / mat board |
| everywhere **except** N panels | top | comic gutters |
| everywhere | bottom | an opaque ground / underpainting |

No `invert` flag and no separate scrim concept: `Invert` is already a
constant-cost operation on this representation.

```rust
pub enum LayerContent {
    /// Painted tiles — only populated ones exist (the infinite canvas).
    Paint(HashTrieMap<TileCoord, TilePairHandle>),
    /// A procedural region filled with a flat colour (DESIGN.md §6.3).
    Matte { region: MatteRegion, color: [f32; 4] },
}

pub enum MatteRegion {
    /// Everything outside this canvas-space rect — the frame.
    OutsideRect { min: Vec2, max: Vec2 },
}
```

`color` is **straight sRGB**, like `BrushParams::color`, converted to
working-space channels at composite time — so the log says the same thing whether
the document is Oklab or Mixbox. A matte has no alpha of its own: its
transparency *is* its layer opacity, which is the whole point of it being a
layer.

**The region is stored as geometry, not as a rasterized mask.** §6.8's selection
shader already evaluates shapes analytically from a signed distance at canvas
position, so a matte gets exactness at any zoom, zero tile budget (a 4000² frame
would otherwise cost ~16 MB of mask tiles and could trip `MAX_SELECTION_TILES`),
and a log entry of four floats. Rasterizing to tiles stays available later as a
pure caching optimization if a comic page ever gets expensive per fragment.

`MatteRegion` has exactly one variant today because that is the only one built.
It is the seam where the `SelectionOp` algebra lands in P4 (§9), at which point
gutters, lasso mattes, `All`, and frame-from-selection all arrive together. Per
DESIGN's own precedent — `tooth`, `drag`, `bleed` were deleted rather than kept
as inert scaffolding — no variant appears here before it does something.

## 3. Why a layer, and what that buys

Because it is a layer, all of this is already built:

- **The scrim is layer opacity.** A 50% black matte on top is the classic crop
  scrim; drag to 100% for presentation. No `SetFrameScrim` command, no toggle.
- **Visibility, ordering, naming, delete** — the Layers panel already does it.
  "Which frame is active" is "which layer is selected"; no new concept.
- **Multiple frames** are multiple matte layers. Variant crops for free.
- **Undo, save, replay, collaboration** — a matte is document state reached by
  the existing `AddLayer` / `RemoveLayer` / `MoveLayer` / `SetLayerOpacity`
  actions, so §5 and §12 need no new argument.
- **Blend modes, when they land.** A Multiply matte is a vignette; a Screen
  matte is a light wash. Free expressiveness we do not have to design.

The alternative — a `frames: Vector<Frame>` field beside `layers` — needs its
own id space, its own actions, its own z-order rule, its own panel, and its own
active-item concept. Every one of those is already solved for layers.

## 4. Compositing a matte (forced by the media pass)

This is where the design meets the engine, and two things are **forced** rather
than chosen.

### 4.1 A matte must write the aux target

[`media_common.wesl`](crates/stark-shaders/src/shaders/media_common.wesl#L61)
derives a texel's visible alpha from the translucent-slab law:

```
vis = 1 − exp(−OPACITY_K · color.a · (aux.x − surface_height))
```

Visibility comes from **per-unit opacity × thickness**, not from composited
alpha. A matte that wrote only colour would be perfectly invisible. So a matte
writes `color.a = 1` and a thickness `MATTE_THICKNESS`, chosen so the slab reads
solid: with `OPACITY_K = 1.0`, a thickness of 8 gives `vis > 0.999` even after
the surface height (≤ ~0.6) is subtracted.

The physical reading is honest rather than a workaround: **a matte is a flat,
opaque coat of paint.** Its interior has constant height, so its gradient is
zero, so it lights flat and matte — no weave, no gloss. That is what a mat board
looks like. Its boundary is a height cliff and therefore catches light, the same
way every paint stroke's edge already does; at the frame border that reads as a
crisp bevel, which is wanted.

### 4.2 The matte's aux blend must be *over*, not additive

The colour space's `aux_blend()` is **additive**
([colorspace.rs:92](crates/stark-core/src/colorspace.rs#L92)) — correct for
paint, where thickness accumulates. If a matte blended additively, the height of
paint *underneath* it would survive, and `height_at` would emboss that paint's
impasto as ghost ridges through an opaque mat board.

So the matte pipeline declares its own blend state: premultiplied **over** on
both targets. The aux then composites as `aux' = aux·(1−a) + (H·a, 0)`, which is
right at both ends — an opaque matte erases the relief beneath it, a 30% scrim
keeps 70% of it.

(`OneMinusSrcAlpha` as a destination factor is valid on the alpha-less
`Rg16Float` aux target: the factor reads the *source* alpha from the fragment
shader's output vec4, which exists regardless of the format's channel count.)

### 4.3 Matte opacity is non-linear, and that is deliberate

Layer opacity `λ` scales *both* inputs to the slab law — the premultiplied colour
(so `color.a = λ`) and the aux thickness (so `aux.x = λ·H`) — giving

```
vis(λ) = 1 − exp(−K · λ² · H)
```

which is **quadratic in the exponent**. With `H = 8` that is pronounced: `λ = 0.5`
covers ~86%, not 50%. Measured on a black frame over a red stroke, the outside
band reads `[222,61,36]` hidden → `[81,10,2]` at half → `[20,8,2]` opaque.

This is kept, because it is **exactly** what paint-layer opacity already does:
pass A scales premultiplied colour and additive aux by `λ` too, so a paint layer
is `1 − exp(−K·λ²·op·t)` — the same form. Consistency with paint is the entire
premise of making a matte a layer, and a compensating curve here would make the
matte the one layer whose opacity slider means something different.

It would be easy to make a matte alone exactly linear (write `color.a = 1` and
`aux.x = −ln(1−λ)`), and that is the *right* model for "opacity means visible
coverage". But if the curve is wrong it is wrong for paint layers first, and the
fix belongs in the vis law, once, for both. Noted here so the choice stays
visible rather than looking like an oversight.

### 4.4 Interleaving with tiles

Pass A currently flattens every visible tile into one instanced draw
([engine.rs:467](crates/stark-core/src/engine.rs#L467)). A matte has to composite
*in stack order*, so the flat list becomes an ordered item list:

```rust
pub enum CompositeItem {
    Tile { coord: TileCoord, handle: TilePairHandle, opacity: f32 },
    Matte(MatteDraw),
}
```

The compositor walks it in order, switching pipelines where a matte sits between
runs of tiles. This costs nothing: a tile already needs its own draw because it
needs its own bind group, so pass A was never one batched draw to begin with —
interleaving mattes adds no per-tile overhead, and an all-paint document issues
exactly the draws it did before (every golden is unchanged, which is the proof).

The matte draw is a fullscreen quad; the vertex stage inverts the view uniform's
canvas→NDC transform to hand the fragment stage a canvas-space position, and
coverage comes from a signed distance to the rect, antialiased over one screen
pixel (`1/zoom` canvas px). Same technique as `selection.wesl`, same seam-free
property: coverage is a pure function of canvas position.

One implementation constraint worth recording: pass A's view bind group is
declared **vertex-only**, so the fragment stage cannot read the zoom from it. The
antialiasing width is therefore computed in the vertex stage and passed as a flat
varying — it is constant across the quad, so this is exact, and it avoids
coupling the matte to the overlay pass's separate `VERTEX_FRAGMENT` layout.

## 5. What a matte is *not*: the substrate

A matte is a slab of opaque paint. The **substrate** — the colour of the canvas
itself — is a different thing: it is *under* everything, it is lit, and the weave
shows through it. The media pass already handles it as `m.bg`.

Today that colour is **view state owned by the frontend**
([render.rs:149](crates/stark-ui/src/render.rs#L149)), so the ground you painted
on is not saved with your painting. That is a real gap and this design closes it,
but not by making it a layer: it becomes `DocState.background`, sitting beside
`DocState.surface`, on precisely the argument §6.4 already makes for the weave —
which canvas a piece was painted on is part of what the document *is*.

Both exist and both make sense: `background` is the gesso, an `All`-region matte
(when P4 lands) is an opaque underpainting brushed over it.

## 6. Export

Export needs a rectangle and a pixel size. **Export takes a layer id**: a matte
layer's region bounding box is the output rect, and no matte selected falls back
to `DocState::bounds`. The layer panel's selection is therefore already the frame
picker, and multiple frames need no new machinery.

This composes without a single special case. Render every visible layer into the
frame's rect and the right thing happens by construction:

- the frame matte covers only *outside* the rect, which is clipped away, so it
  contributes nothing to its own export;
- a ground matte is inside and contributes exactly what it should;
- a matte whose visibility is off still defines the rect, because geometry and
  presentation are separate properties of the same layer.

The plumbing is one real change: `Engine::render` currently reads
`self.session.view` ([engine.rs:457](crates/stark-core/src/engine.rs#L457))
rather than taking a `ViewTransform`, and `render_to_image` exports the
*viewport* — so today's "export" is a screenshot. Restoring DESIGN §6.4's
documented signature (`render(target, view)`) makes export "render at
`frame.rect × scale`, centred on the frame, at `zoom = scale`."

Three decisions that go with it:

- **Scale** is a property of the output, not the artwork. The frame stores a
  canvas-space rect only; the export offers 1× / 2× / explicit pixel dimensions.
- **Transparent background** skips the media pass's substrate composite — a real
  branch, not merely an alpha — and is offered alongside the substrate colour.
- **The overlay pass is suppressed.** Selection outlines and composition guides
  are chrome and never reach a file.

Export is safe at any scale because the relief is already zoom-normalized:
`strength = m.light.w / m.surf_a.z`
([media_common.wesl:86](crates/stark-shaders/src/shaders/media_common.wesl#L86))
divides the screen-space gradient by the canvas px it spans, so a 2× export has
the same slope, resolved finer.

## 7. Interaction

A frame has **no permanent panel**, and — the sharper form of the same rule —
**nothing that is an ordinary layer property gets a frame-specific control.**
Creating one is `+ Frame` in the Layers panel; opacity (which *is* the crop scrim)
and removal are the Layers panel's single set of controls for whatever is
selected, applying to a frame and a paint layer alike. Only the fill colour lives
in the frame bar, because it is the one thing that is about the frame rather than
about a layer.

That is what a matte being a real layer is worth: duplicating opacity and delete
into a frame-specific bar would have meant two controls for one property, which is
the same duplication that made frame *selection* confusing.

The rest lives in a bar mounted only while a frame is **selected**, alongside the
selection bar and on the same argument (DESIGN §6.8):
controls meaningless without a frame should be absent rather than greyed out, and
a bar that is simply present or absent says "you are composing" more directly than
a mode indicator would.

### There is exactly one selection

Selecting a frame is clicking its row in the Layers panel — the same click, and
the same `ViewCommand::SetActiveLayer`, that selects a paint layer. **The frame
bar and the on-canvas handles key off `active_layer` being a matte.** There is no
separate frame-selection state anywhere.

That means `active_layer` is **the selected layer**, not "the paint target". The
widening is deliberate and is what makes the interface simple: with one selection
concept, "exactly one row is highlighted" is a *consequence* rather than a rule
two pieces of state have to be kept agreeing on. It also removes any way for a
stroke to land on a layer that does not look selected.

An earlier cut had two: the engine's `active_layer` (which refused mattes) plus a
frontend `selected_frame` signal. Both could be set at once, which read as two
selected rows, and patching it needed mutual-exclusion rules in the row
highlighting plus an auto-deselect when a stroke began. All of that is deleted;
the duplication was the bug, not the symptom.

**A stroke aimed at a frame does nothing** — refused identically by `apply` and by
the preview path, so no frontend needs a rule. Rather than block the gesture, the
canvas says so first: the brush crosshair becomes `not-allowed` whenever the
selected layer takes no paint. Blocking in the frontend was considered and
rejected — it is a rule a second frontend would have to reimplement, and this
codebase has consistently put such rules in the engine (`Session::end_selection`
hands the tool back engine-side; `apply` refuses the stroke).

An `Option<LayerId>` active layer was also considered, so "nothing selected" could
be expressed. Skipped: `DocState` always has at least one layer, so `None` would
be representable but unreachable — its own kind of lie. It becomes worth adding
when there is a real `None` to model.

Three creation paths matter more than dragging, and the first is the one people
actually want:

- **Add frame** — sized to the painted content if there is any, otherwise to what
  the viewport shows. Both are "frame what I am looking at", the only sensible
  default on an unbounded canvas.
- **Fit to art** — snap to `DocState::bounds`. **Fit to view** — snap to the
  viewport.
- **Aspect** — a drop-down of 1:1, 4:5, 3:2, 16:9, reshaping about the centre and
  *preserving area*, so switching neither grows nor shrinks the piece. It reads the
  frame's current ratio back, showing `Custom` when a dragged handle has landed on
  something arbitrary — a state readout rather than a row of fire-and-forget
  buttons. `Custom` is offered only while it is what the frame *is*, since picking
  it could not mean anything.

Once it exists, it is adjusted by **handles drawn over the canvas**: eight
edge/corner grips plus a move pill. Two decisions there are forced by this frame
being non-clipping:

- **The interior is not interactive.** `pointer-events: none` on the frame box,
  `auto` only on the grips. The inside of the frame is exactly where you paint, so
  it must pass every pointer event through to the canvas.
- **Hence the move pill, outside the top edge.** Dragging the interior is how
  every other crop tool moves a frame, and it is the one gesture this frame cannot
  borrow. A small handle above the frame is the substitute.

A drag **previews live and logs once**: each pointer move sends
`ViewCommand::PreviewMatteRect` (view state, never logged), and release commits a
single `DocCommand::SetMatteRect`. So a drag costs one undo step rather than one
per move. `observe()` reports the *previewed* layer rect, which is what keeps the
handles under the pointer rather than a frame behind on the committed value —
carefully **only** the layers, since `has_selection` must stay committed-only or a
marquee drag flashes the selection bar in and out.

This is a view command rather than a `GestureCommand` because a frame drag is
handle-relative, not sample-driven: there is no `InputSample` to feed
`Start`/`To`/`End`, and which grip is held is the frontend's business. What it
keeps is the shape that matters — build in view state, commit once on release.

The overlay sits at `z-index: 10`, below every piece of floating chrome (rail,
panels, bars, all at 20+): the handles belong to the canvas, and a panel over them
must win. Worth recording because the first cut got this wrong in a way that looks
impossible — `.panel-stack` carried *no* `z-index` at all, so it sat at auto level
and a positioned sibling with any `z-index` beat it regardless of DOM order. The
stack now declares 20 explicitly rather than relying on document order.

Still to come: snapping while dragging (to content bounds, to other frames, to the
canvas origin) is most of what makes a crop tool feel good and is cheap; frame
from selection; ratio-locked dragging.

**Painting on a matte is refused, not swallowed.** `apply` refuses a
`CommitStroke` naming one and the preview path refuses it identically, so
`preview == committed` holds and a replayed or remote log agrees. No
auto-switching, no silent rasterization. In the Layers panel a matte reads as
"◱ Frame" behind a dashed border; it is selected like any other row, and while it
is, the canvas cursor says the brush has nowhere to go.

## 8. Composition aids stay view state

Thirds, golden section, diagonals, centre cross, and a custom grid are an *aid*,
not the artwork: per-client, never replicated, never exported. They read their
rect off the selected matte layer. The temptation to make them a layer should be
resisted for the same reason `MediaParams` is not one.

**Review mode** is one keystroke — fit to frame, matte to full opacity, chrome
hidden, selection outline suppressed. That is stepping back from the easel, and
the `canvas_active` chrome-fade machinery already does the hard part. Paired with
the view mirror from `MISSING_FEATURES.md` §1.2 it becomes the complete "how does
this actually read?" check in two keys.

## 9. Phasing

Each phase is independently useful and independently shippable.

**P1 — the matte layer. ✅ Done.** `LayerContent` enum; `MatteRegion::OutsideRect`;
`matte.wesl` and its pipeline; `CompositeItem` ordering in pass A; `AddMatte` /
`SetMatteRect` / `SetMatteColor` actions and commands; strokes refuse matte
layers; `bounds` ignores them. Delivers the frame, the scrim, and
reorderability — engine-side. *No frontend and no export yet:* a frame is
reachable only by `DocCommand::AddMatte`, so P3's tool and handles are what make
it usable by hand.

**P3 — the composition tool. ✅ Mostly done** (taken before P2, since export needs
something to frame *against* before it can be tested by hand). `+ Frame` in the
Layers panel, beside one Remove button and one opacity slider that act on whatever
is selected; matte rows that select a frame; the frame bar (size readout, aspect
drop-down, fit-to-art, fit-to-view, fill colour, done) in a shared bottom-bar
column with the selection bar; the on-canvas grips with live preview and
single-action commit. **Not yet:** snapping, composition guides, review mode,
fit-to-frame, frame-from-selection.

**P2 — export.** `Engine::render` takes an explicit `ViewTransform`;
`export(layer, scale)`; `DocState.background` + `SetBackground`; PNG download and
save/open wired in `stark-ui`.

**P4 — the general region.** `MatteRegion` becomes the `SelectionOp` algebra:
comic gutters, lasso mattes, `All` slabs, frame-from-selection.

## 10. Testing

`crates/stark-core/tests/matte.rs` covers P1:

- **`frame_covers_outside_and_spares_inside`** — the core claim, and the §4.1
  regression: a matte that failed to write the aux target would be perfectly
  invisible and this would catch it.
- **`opaque_matte_erases_relief_beneath`** — the §4.2 ghost-ridge regression,
  formulated as *an opaque matte over a heavy stroke must render identically to
  the same matte over bare canvas*. Compared on the lit image, so a surviving
  height field shows up as shading. This is the failure the design is most likely
  to get wrong and the one least likely to be noticed by eye.
- **`matte_honors_layer_opacity_and_visibility`** — monotonic between opaque and
  hidden, deliberately asserting no midpoint (§4.3), and on total brightness
  rather than per channel: against a red stroke the blue channel is floored at
  both ends and has no range to be "between" in.
- **`matte_below_paint_does_not_cover_it`** — guards the ordered walk against
  being flattened back into "all tiles, then all mattes".
- **`a_matte_can_be_selected_but_takes_no_paint`** — the one-selection model: a
  matte *is* selectable (so the frontend needs no second concept), reports
  `is_paintable() == false`, takes no paint, and selecting a paint layer again
  resumes painting so nothing can get stuck.
- **`matte_does_not_extend_canvas_bounds`**, **`matte_undoes`**,
  **`dragging_a_frame_previews_without_logging`**.

Two notes on what is *not* tested and why:

- The §6.4 seam invariant needs nothing new here. A matte samples no tile — it is
  a fullscreen quad evaluated analytically in canvas space — so it cannot
  introduce a seam, and `tests/seam.rs` is unaffected.
- Every pre-existing golden is unchanged by this work, which is the evidence that
  turning pass A's flat tile list into an ordered item list is behaviour-
  preserving for paint-only documents.

Replay equivalence (DESIGN §9) extends unchanged: a matte is an action like any
other, so paint → matte → undo → redo → save → load agrees. Worth an explicit
case in `save_load.rs` when P2 wires persistence.
