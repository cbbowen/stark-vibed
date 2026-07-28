# Missing Features — a gap analysis against the prior art

What Stark cannot do that Photoshop, Procreate, Clip Studio Paint, and Corel
Painter can — read as *creative workflows enabled*, not as interfaces to copy.
The ranking is by how much finished work each item unblocks, and every entry
notes what it costs **given this architecture**, because that is the only thing
that makes the ordering actionable. Several items are far cheaper here than they
would be in a pixel-buffer app; a couple are the reverse.

The last section is the part worth the most attention: the action log is a
capability none of the four have, and we currently spend it only on undo.

---

## Tier 0 — you cannot finish and keep a painting

Not features. The floor.

### 0.1 Save, open, export

`Engine::save_bytes` / `load_bytes` exist in core (DESIGN §8) and **nothing in
[stark-ui](crates/stark-ui/src) calls them**. There is no raster export at all.
Nothing ever leaves the app.

Export forces a decision the infinite canvas has so far deferred: *what
rectangle is the piece?* Photoshop answers with a document, Procreate with a
fixed canvas; we need an explicit, movable **frame** (see
[FRAME_DESIGN.md](FRAME_DESIGN.md)). Once a frame exists, export is nearly free —
`render` already takes an arbitrary `ViewTransform`, and
[readback.rs](crates/stark-core/src/gpu/readback.rs) already does GPU→CPU. That
is why framing is the right thing to build first.

### 0.2 Eyedropper — **built**

Sampling colour off the canvas is the most-used non-brush action in painting,
and it matters more here than anywhere else: the entire point of Mixbox pigment
mixing (DESIGN §6.7) is to pick the mix back up. Without it the mixing engine is
a rendering feature rather than a working one.

`Engine::pick_color` is a **request**, not a command (it must answer), so it sits
in §4's request tier next to `save_bytes`, and returns a future because the
readback is the one inherently asynchronous GPU operation — the same shape as
`export`. Alt+drag on the canvas is the binding, as in Clip Studio Paint and
Rebelle, so a colour is picked up without putting the brush down.

The decision it turns on: it samples the **raw layer channels**, not the
composited, lit result. It runs the compositor's pass A into a small target and
stops there, so what comes back is the paint's own channels — not a colour that
has been through image-based lighting, a tonemap and an sRGB encode, and in a
Mixbox document not a display colour in place of the pigment mixture. Sharing
pass A with rendering rather than reimplementing it is what keeps a sample and
the screen from drifting apart. Bare canvas answers *nothing*: the substrate is
the ground, not paint to pick up. Sample-layer(s) and sample-radius are options
(`PickOptions`), in a floating bar mounted only while Alt arms the tool — the same
present-or-absent argument the selection and frame bars make, and what makes a
modifier binding discoverable rather than secret.

### 0.3 Transform

Move / scale / rotate / flip / free-transform of a selection or a layer, plus
cut / copy / paste. Today a selection can only *mask*
([selection.rs](crates/stark-core/src/document/selection.rs)); not one painted
pixel can be moved. Every competitor treats transform as co-equal with the
brush.

This is the largest genuine engineering item on the list — resampling tiles
under an affine, a new `ActionKind`, and a live preview that matches the commit —
but §10 already lists it as additive and the log model supports it.

### 0.4 Fill, gradient, and blend modes

[`BlendMode`](crates/stark-core/src/document/layer.rs#L17) has exactly one
variant, `Normal`. Multiply / Screen / Overlay is *how* digital painters shade
and glaze; losing them costs more finished work than any brush feature gains.
DESIGN §6.3 already notes richer modes need per-layer isolation — the same
prerequisite as groups and adjustment layers, so it is one investment paying
three ways.

Fill and gradient are how anyone blocks in. Both hit an infinite-canvas wrinkle
worth deciding once: a flood fill of an unbounded region is undefined, so fill
must be bounded by the selection, the layer's populated bounds
(`DocState::bounds`), or the frame — another argument for §0.1's frame.

---

## Tier 1 — the workflow multipliers that define the competitors

### 1.1 Layer masks, clipping masks, alpha lock, groups

The non-destructive workflow, and we are closer than it looks: `Selection` is
*already* a sparse `R8Unorm` tile map with aprons and a soft-set algebra — which
is exactly a layer mask. Alpha lock and clip-to-below are per-layer flags read
by the compositor. Groups need the same per-layer isolation as blend modes.

Layers also lack names, thumbnails, duplicate, and merge/flatten. Merge is
load-bearing twice: it is a workflow staple *and* it is how an append-only
action log stops growing forever.

### 1.2 Mirror and rotate the canvas view

`ViewTransform` carries center and zoom but no rotation or flip
([geom.rs](crates/stark-core/src/geom.rs)). Flipping horizontally to catch
drawing errors is universal across all four apps and costs essentially nothing —
pure view state, never logged, never sent. Rotating the canvas to get a
comfortable stroke direction is the same argument and the same change.

### 1.3 Symmetry and drawing guides

Per unit of implementation effort, probably the highest-leverage illustration
features in existence (Procreate's Drawing Guides, CSP's perspective and
symmetrical rulers). Our model makes them unusually clean: a guide is a **path
transform applied between the fitter and the renderer**. Mirror symmetry is one
gesture emitting N `StrokeRecord`s — or one record carrying its mirror axes,
which keeps the log tighter and leaves §12's convergence argument untouched.
Perspective snapping and shape assist (Procreate's QuickShape) attach at the
same seam.

### 1.4 Brush parameter mapping — inputs → parameters

The structural gap in the brush engine.
[`BrushParams`](crates/stark-core/src/document/action.rs#L204) holds fixed
scalars, and DESIGN §6.2 records that per-segment pressure/tilt modulation of
the dynamics rates was *removed* as inert scaffolding. What CSP, Procreate, and
Painter all have and we do not is a **mapping matrix**: any input (pressure,
tilt, azimuth, velocity, stroke-relative position, random) driving any parameter
(radius, opacity, `add`, `lift`, angle, hue) through a user-editable curve.

Every other axis of our brush model is more sophisticated than theirs. This one
is what makes a brush feel *authored* rather than configured — and it is what
makes a brush **library** (named presets, previews, import/export) worth
shipping, which is the thing users actually shop for.

### 1.5 A mixing palette

We have Mixbox and a mass-conserving wet-paint loop. Nobody's mixing surface is
good — Painter's Mixer Pad is the state of the art and it is twenty years old. A
scratch surface you genuinely mix on and then pick up from, running the *same*
dynamics loop and the *same* eyedropper, is a novel and defensible feature that
falls directly out of what is already built.

### 1.6 Adjustments and a few filters

Levels/curves, hue-sat, blur, sharpen, liquify/warp. Photoshop's bloat is not
that adjustments exist; it is that there are ninety of them behind eight menus.
Ten, shipped as **adjustment layers** (non-destructive, re-orderable,
log-native), is strictly better than Photoshop's destructive model and cheaper
to build than it.

---

## Tier 2 — where we can beat the prior art

Photoshop's history is a bounded, linear, destructive stack that vanishes when
the file closes. Ours is a complete, deterministic, replayable log of id-tagged
actions that *is* the save format. Three things follow that no competitor can
ship without a rewrite:

### 2.1 Post-hoc stroke editing

Change a committed stroke's colour, brush, or dynamics and replay. "Every stroke
stays editable" is a vector-app promise delivered with natural media.

It fits the CRDT cleanly: an amend is a **new action referencing the target** —
exactly the `Undo(ActionId)` shape from §5.4 — not a mutation. Grow-only stays
grow-only, and the timeline's `effective_actions` resolution generalizes to
"latest amendment wins" without touching `Action::apply`.

### 2.2 Branching history

Try a variation, keep both, compare, pick. `Timeline` is already a trait; a
branch is a second effective-sequence over the same log.

### 2.3 Selective undo

`ActionKind::Undo(ActionId)` already means "derive the document as if `target`
were absent." The hard part is built and only "undo the last thing" is exposed.

### 2.4 Timelapse as a tool, not an export

DESIGN §8 designs timelapse replay. Exposed as a **scrubber the artist can drag
while working**, it is a real critique tool — seeing your own process is how you
find the moment a piece went wrong — rather than a novelty output.

---

## Deliberate non-goals

Naming these is part of not becoming Photoshop.

- **Animation.** CSP and Procreate both have it. It is a product fork, not a
  feature, and it would pull the action log toward a timeline model that serves
  the painting case worse.
- **Text and vector layers.** CSP's turf, orthogonal to the oil-painting
  positioning, and a large surface area (fonts, shaping, i18n) for work that
  is not painting.

---

## Sequencing

1. **Now** — framing/export, save/open, ~~eyedropper~~, blend modes, fill. The
   difference between a tech demo and something a finished piece comes out of.
2. **Next** — transform; layer masks / clipping / alpha lock / merge; view
   mirror and rotate.
3. **Then, in parallel** — brush parameter mapping and the brush library
   (deepens what is already the strongest asset); symmetry and guides (highest
   value per line of code on this list).
4. **The bet** — post-hoc stroke editing and branching history. Structurally
   impossible for Adobe to match, and already 80% paid for.
