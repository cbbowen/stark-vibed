# Status, roadmap, and stability

Build order, the gap analysis against the prior art, and the file-format stability policy — §13, §18, §19.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 13. Build order & status

Status lives here and nowhere else.

| # | Step | Status |
|---|---|---|
| 1 | GPU + tiles skeleton | done |
| 2 | Stroke MVP (command/action split, CoW tiles) | done |
| 3 | History + golden harness | done |
| 4 | Multi-channel + media pass | done |
| 5 | Save/load + timelapse | done |
| 6a | Layers | done |
| 6b | Dioxus UI | done |
| 6c | Navigation (pan/zoom) | done — tile LOD descoped |
| 7 | Brush shapes & assets (§6.6) | done |
| 8 | Cubic stroke interpolation (§6.2) | done — streaming append-only fit, adaptive flattening |
| 8b | Continuous swept-segment stamping (§6.2) | done — one quad per segment, prefix-τ coverage |
| 8c | Tile aprons (§6.4) | done — killed the lighting seams the media pass amplified |
| 9 | Pluggable color spaces (§6.7) | done — Oklab + Mixbox |
| 10 | Wet mixing & brush dynamics (§6.2) | done — GPU swept-exchange loop, no CPU readback |
| — | Surface bump maps (§6.4) | done — relief, and the deposition tooth that gates what the brush lays, on the ground a moving tip is about to meet |
| 11 | Brush file upload | done — custom shape library, localStorage, mid-session peer replication |
| 12 | Collaboration (§12) | done |
| — | Selections (§6.8) | done |
| — | Fill (§6.8) | done — the fifth `ShapeAction` |
| 13 | Per-client state: owned selections + presence (§17) | done |
| — | Transform (§16) | done — engine, exactness tests, and the ellipse gesture UI; snapping + clipboard remain |
| — | Frames, mattes, export (§15) | done — P1/P2/P3; the general region (P4) remains |
| — | Blend modes (§6.3) | done — Glow / Radiance / Multiply |
| — | Eyedropper (§18.0.2) | done |
| — | View mirror & rotate (§18.1.2) | done |
| — | Timeline mode / scrubber (§18.2.4) | done |
| — | Groups & clipping (§14) | done |
| — | Filter layers (§21) | done — the architecture and the color filter; the rest of the kinds (§21.9) remain |
| — | Drag-and-hold drawing assist (§6.9) | done — line + ellipse; the shape-assist half of §18.1.3 |
| — | Stroke smoothing — the towed tip (§6.11) | done — the tow, the per-brush amount, the string overlay; the pursuit-mode soft rope stays in reserve |
| — | Brush parameter mapping (§6.2, §18.1.4) | done — pressure/tilt → size/flow/stretch/lift/deposit/bleed; more sources and targets are variants away |
| — | Modifier drags — scrubby zoom, Size/Flow (§18.1.9) | done — with the size ring; a flow readout is not |
| 14 | Mutable medium — horizontal flux (§14 open / §6.2) | **not started** |

Step 14, restated against what actually shipped: the Dry/Knife/Wet enum variants
collapsed into **one tool** (`add`/`lift`/`deposit`/`charge`), every axis a flux
on the single conserved quantity. What remains are the *horizontal*-flux axes —
drag as conservative finite-volume advection, ridge as a zero-mean doublet, bleed
as a footprint-local blur. They are the intended design when built and are no
longer carried as inert fields. Nor is the `wet` *channel*: a real diffusion
model would reintroduce it as a second aux component, an `R16Float → Rg16Float`
format change plus the passes that carry it — cheap to redo, and cheaper than
storing a zero until then. ("Tooth-revealed canvas" is now the deposition gate,
§6.4 — built, though it does not yet *fill* as paint piles into the weave.)

Each step is independently testable through `stark-engine` before any UI exists,
which is exactly the leverage the frontend/backend split was meant to provide.

### Nice-to-have (not scheduled)

- **Tile LOD / mipmaps** — sample minified tiles when zoomed far out, for
  *responsiveness* on huge canvases. The **aliasing** half of this is answered
  and no longer wants mips: presentation supersamples and resolves (§6.4),
  which also covers the weave and the relief shading, neither of which a
  prefiltered tile could have. What is left is the cost — supersampling pays
  more work to draw less picture, where a mip chain would pay less — so this
  stays unscheduled until profiling on a large document says otherwise.
- **HiDPI** — the web canvas uses a 1× drawing buffer (CSS pixels); multiply by
  `devicePixelRatio` for crisp rendering on retina displays.
- **Damage tracking** (§6.3) — the **view-AABB cull half is done**: pass A's draw
  list is built only from tiles the render's view can reach, so its cost follows
  the viewport rather than the document. What is left is the *per-version damage
  set* — knowing which tiles a commit actually changed, so an unchanged frame
  need not recomposite what it already drew. That one needs the timeline to
  carry the answer, not just the renderer to ask a cheaper question.
- **Batching the dynamics loop's independent dispatches** (§6.2) — the current
  per-move bottleneck is dispatch latency, not shader work.

---


## 18. Roadmap — a gap analysis against the prior art

What Stark cannot do that Photoshop, Procreate, Clip Studio Paint and Corel
Painter can — read as *creative workflows enabled*, not as interfaces to copy.
Ranked by how much finished work each item unblocks, with what it costs **given
this architecture**, because that is the only thing that makes the ordering
actionable. Several items are far cheaper here than in a pixel-buffer app; a
couple are the reverse. §18.2 is the part worth the most attention: the action
log is a capability none of the four have.

### 18.0 Tier 0 — you cannot finish and keep a painting

Not features. The floor. **All of Tier 0 is now built.**

#### 18.0.1 Save, open, export — built

See §8 and §15.6. Export forced the decision the infinite canvas had deferred
(*what rectangle is the piece?*), which is why framing was the right thing to
build first.

#### 18.0.2 Eyedropper — built

Sampling color off the canvas is the most-used non-brush action in painting, and
it matters more here than anywhere else: the entire point of Mixbox pigment
mixing (§6.7) is to pick the mix back up. Without it the mixing engine is a
rendering feature rather than a working one.

`Engine::pick_color` is a **request**, not a command (§4), and returns a future
because readback is the one inherently asynchronous GPU operation — the same
shape as `export`. Alt+drag is the binding, as in CSP and Rebelle, so a color is
picked up without putting the brush down.

The decision it turns on: it samples the **raw layer channels**, not the
composited, lit result. It runs pass A into a small target and stops there, so
what comes back is the paint's own channels — not a color that has been through
image-based lighting, a tonemap and an sRGB encode, and in a Mixbox document not
a display color in place of the pigment mixture. Sharing pass A with rendering
rather than reimplementing it is what keeps a sample and the screen from drifting
apart. Bare canvas answers *nothing*: the substrate is the ground, not paint to
pick up. Sample-layer(s) and sample-radius are `PickOptions`, in a floating bar
mounted only while Alt arms the tool — the same present-or-absent argument the
selection and frame bars make, and what makes a modifier binding discoverable
rather than secret.

There are three sources, not two, and the third is the one exception to *bare
canvas answers nothing*: the composite **over the substrate** (§15.5), which fills
in the canvas color wherever the paint does not cover. The other two answer with
the paint that is stored at a point; this one answers with the color that is
*seen* there, which is the question being asked when the paint is a glaze — and
the only source whose answer can be a color no layer holds. It runs the same
`over` the media pass runs, in the same latent channels, so it agrees with the
screen rather than offering a second opinion about it. It is also the one place a
sample stops being an opacity-weighted mean: with the ground behind it every texel
is opaque, so a patch half-covered by a stroke reads as a mixture of paint and
canvas instead of as the stroke alone.

The **sample-one-layer** source drops that layer's composite params entirely —
blend, clip and opacity (§14.4.3). All three say how the layer meets what is beneath
it, which is exactly what this source is asked to ignore: the first two decide how
much of the paint survives its surroundings, and the slider decides how much of the
layer the *document* shows. None of them says what the paint **is**, so a layer
turned down reports the same color rather than a paler one — which is the property
that matters in use, since the reason to sample a faded layer is usually to go on
painting with what is already on it. Zero is the exception and a different statement
rather than a fainter one: a layer switched off contributes nothing, so it answers
`None` like bare canvas.

That the color was already right is worth saying, because it is why this went
unnoticed: the sample divides by the coverage it sums, so the opacity cancelled. What
it did not cancel out of was the floor beneath which a patch is called empty, so a
faded enough layer reported *nothing at all* where the same paint at full strength
reported its color.

#### 18.0.3 Transform — built

Engine *and* gesture UI; see §16. Remaining: snapping, and the cut/copy/paste
clipboard, which reuses the parcel machinery.

#### 18.0.4 Fill and blend modes — built

See §6.3 (the modes) and §6.8 (fill). Fill cost almost nothing because it turned
out not to be a tool: the Select panel's four "actions" were really four *combine
modes*, and reading the shapes as producers of **coverage** made `Fill` the fifth
answer to the same question, landing that coverage on the paint instead of the
mask.

**Blend modes now carry their own parameters**, and `Radiance`'s bend is the first
— see §6.3 for what it does and why it sits on the enum variant rather than beside
it. The seam this section predicted for a mapping UI turned out not to need one:
the parameter is part of the mode, so the log action, the footprint, the patch, the
merge rule and the shader uniform all took it without moving. What the panel grew
is one row that exists only while the mode does, previewed per sample and committed
once, through the same `settle` the opacity slider and the filter bar use.

**The gradient fill is built** (§22.4), and it attached at exactly the seam
this section predicted: not a new pipeline but a `FillOp` whose parcel varies
with position — `Solid` grew a sibling, the region/gate/stacking law/footprint
never moved, and the composing mode is the transform's preview-until-Done
aimed at a fill. The gradient itself came first (§22.1–§22.3): stops fitted
from a line traced through the painting, kept in a browser-local library, and
embedded by value in the op the way a stroke embeds its brush color.

### 18.1 Tier 1 — the workflow multipliers that define the competitors

#### 18.1.1 Layer masks and alpha lock — groups and clipping built

The non-destructive workflow, and we are closer than it looks: `Selection` is
*already* a sparse `R8Unorm` tile map with aprons and a soft-set algebra — which
is exactly a layer mask. Alpha lock is a per-layer flag read by the compositor.

**Groups and clipping shipped as one feature** (§14), reusing the per-layer
isolation blend modes already had, recursed. **Duplicate** shipped with them
(§14.8) and cost almost nothing, which is the copy-on-write tile map paying out:
the copy holds the source's own handles, so duplicating a layer allocates no GPU
memory at all until one of the two is painted on. **Still missing: layer masks
proper, alpha lock, thumbnails, and merge/flatten.** Merge is load-bearing twice:
it is a workflow staple *and* it is how an append-only action log stops growing
forever.

#### 18.1.2 Mirror and rotate the canvas view — built

`ViewTransform` carries `rotation` and `flip_h` beside centre and zoom, and they
are view state exactly as predicted (§6.4).

- **Mirror**: `H` (`ViewCommand::MirrorH`). A toggle, and **screen-relative**: it
  swaps the left of the screen with the right at any angle, so the check means
  the same thing however the easel is turned. Reflecting the result keeps the
  view a rotation-and-a-mirror rather than a free matrix, because a reflection
  pushes back through a rotation (`M·R(θ) = R(−θ)·M`) — so the whole operation is
  "negate the angle, toggle the mirror", and twice is exactly the identity.
- **Rotate**: right-drag in the Navigator — the direction you drag becomes up.
  The core answers only the question (`ViewTransform::rotation_for_up`) and the
  frontend sends an absolute `ViewCommand::SetRotation`, because everything
  between the two is gesture feel: the drag *eases* toward the angle it points at
  in proportion to how far it has been pulled (near the press, a two-pixel
  vector's direction is almost pure noise, and following it exactly makes the
  canvas snap to a wild angle the instant the button goes down), and the target
  snaps to a quarter turn within ~5° so upright is reachable by hand.

The chrome over the canvas turns with it: the frame's box and handles and the
transform widget compose the view's orientation into the CSS matrix they already
carried, and pointer deltas come back through the full inverse rather than over
the zoom.

**The light stays in the room.** Relief shading is computed from the height field
as it falls on the *screen*, so turning or mirroring the canvas changes how
impasto and the weave catch the light — a real ~130-level difference on a woven
canvas, and the same thing that happens when you turn a real canvas under a fixed
lamp. That is the behaviour painters use rotation for; the alternative (the light
turning with the canvas, so a mirrored view is a pure mirror image) is a one-line
change to the environment lookup if the mirror ever wants to be exact.

#### 18.1.3 Symmetry and drawing guides — shape assist built

Per unit of implementation effort, probably the highest-leverage illustration
features in existence (Procreate's Drawing Guides, CSP's perspective and
symmetrical rulers). Our model makes them unusually clean: a guide is a **path
transform applied between the fitter and the renderer**. Mirror symmetry is one
gesture emitting N `StrokeRecord`s — or one record carrying its mirror axes,
which keeps the log tighter and leaves §12's convergence argument untouched.
Perspective snapping attaches at the same seam.

**Shape assist (QuickShape) shipped there** — see §6.9. It is the evidence for the
claim above: drag out a rough line or ellipse, hold the pen still, and the stroke
snaps and becomes steerable, in one new module plus a dwell watcher in the
frontend — because a snapped stroke is still a `Vec<ControlPoint>` and nothing
downstream of the fitter had to learn what it was.

**Perspective shipped at the same seam** — the guide and its overlay in §20, and
the snapping in §20.6–§20.7, which cost the assist one bool and one `Option`: a
line that lands near a visible guide's axis is aimed along it, and a loop that
lands where a circle on one of its planes would be *becomes* that circle. Both
are still a `Vec<ControlPoint>`, the guides are read and never touched, and
nothing below `StrokeRecord::path` learned about either. The perspective-circle
arm is the best evidence yet for the seam: the hardest construction in the
drawing-office repertoire arrived as one chart, one conic congruence and a
residual measured where the artist can see it. **Still missing: symmetry**, the
remaining arm — one gesture emitting N records, or one record carrying its
mirror axes.

#### 18.1.4 Brush parameter mapping — inputs → parameters — built

Was the structural gap in the brush engine: `BrushParams` held fixed scalars with
pressure → radius wired into the segment generator, and §6.2 recorded that
per-segment pressure/tilt modulation of the dynamics rates had been *removed* as
inert scaffolding. `BrushParams.modulation` is now the mapping — see §6.2, "Pen
mapping" — with **pressure and tilt** driving **size, flow, lift, deposit and
bleed** through a bounded rational response curve, and the pressure → size rule
demoted to the default entry in it rather than a rule.

The decision that made it cheap: a modulation is a **multiplier in [0, 1]**, so
every bound the engine derives from a brush stays sound without learning that
modulation exists. That is what kept the change to one place in the renderer
(`generate_segments_in`) and none at all in the stamp loop, which already carried
its rates per dispatch.

What is deliberately not built: **azimuth, velocity, stroke-relative position and
random** as sources, and **opacity and angle** as targets. Each is a variant on
existing enums plus a line in the segment generator; velocity needs a
normalisation constant argued for rather than picked, and azimuth is signed and
so wants a different response shape than the two [0, 1] sources share. Multiple
mappings onto one target is likewise a `Vec` away, and would multiply.

It is what makes a brush feel *authored* rather than configured — and what makes
a brush **library** worth shipping, which is the thing users actually shop for.
(The library's skeleton exists: named per-user presets persist in `localStorage`
and apply from the Brush panel, `stark-ui/src/presets.rs`, where the shipped
Pencil now maps tilt → stretch and pressure → flow — leaning the pen draws the
tip out along the lean rather than scaling it up, §6.6; shape import/persistence is
done, §6.6; §18.1.8 puts ten of them under the number keys; and every preset
shows the stroke it makes, rendered offscreen by a shared engine, §11. Preset
import/export does not.)

#### 18.1.5 A mixing palette

We have Mixbox and a mass-conserving wet-paint loop. Nobody's mixing surface is
good — Painter's Mixer Pad is the state of the art and it is twenty years old. A
scratch surface you genuinely mix on and then pick up from, running the *same*
dynamics loop and the *same* eyedropper, is a novel and defensible feature that
falls directly out of what is already built.

#### 18.1.6 Adjustments and a few filters — started

Levels/curves, hue-sat, blur, sharpen, liquify/warp. Photoshop's bloat is not
that adjustments exist; it is that there are ninety of them behind eight menus.
Ten, shipped as **adjustment layers** (non-destructive, re-orderable, log-native),
is strictly better than Photoshop's destructive model and cheaper to build than
it.

Built as **filter layers** (§21), and the model went one step past what this
paragraph asked for: an adjustment layer's *scope* is where its row sits, so the
clipping toggle Photoshop needs beside every adjustment is not a control here at
all (§21.1). Three kinds ship — the color filter (exposure / contrast /
saturation / hue / tint, in Oklab, §21.5), the spectral chromatic aberration
(§21.10), and the gradient map over the captured ramps of §22 (§21.11) — and
the rest are a variant and an arm each (§21.9).

#### 18.1.7 Touch: the two-finger gesture — built

Middle-drag pans and the wheel zooms, and a tablet has neither. Everything
§18.1.2 built was therefore unreachable by hand: on a touchscreen the canvas
could be painted on and not moved.

**Two fingers pan, zoom and turn the canvas at once** — the gesture every
touch-first painting app shares, so it needs no discovery. One finger still
paints, which is what makes the split work: the tool is what a single contact
does, and navigation is what a *pair* does. A pen is deliberately not counted as
a finger even on the same glass, so the canvas can be moved without putting it
down.

- **It is one command, not three.** `ViewCommand::Pinch { anchor, to, scale,
  turn }` — the canvas point under `anchor` ends up under `to`, scaled and turned
  about it. Sent as a pan, a zoom and a turn instead, each would anchor against
  the view the one before it left, so the second and third would be measured
  against a canvas the hand never saw and the paint would slide out from under
  the fingers. Composed in `ViewTransform::pinch`, what the fingers hold they
  hold — stated as a property and tested as one, over every angle and both
  handednesses. `zoom_about` is now that same call with the fingers standing
  still, so the wheel and the pinch cannot come to mean different things.
- **The turn stays a rotation-and-a-mirror.** The gesture is stated in *screen*
  terms — clockwise on the glass is clockwise on the screen at any angle and
  either handedness, the sense `mirror_screen_h` is already defined in — and
  `R(δ)·R(θ)·M = R(θ+δ)`·M, so the twist adds straight onto the angle and the
  mirror is untouched.
- **Two feel constants, both the frontend's**, for the reason §18.1.2 gives: a
  **deadzone** (~6°) the twist must earn before the canvas turns at all, because
  fingers closing on a target roll about the hand rather than travelling along
  the line between them, and without a band to spend that in every zoom would
  leave the piece a couple of degrees off true; and the **snap to a quarter
  turn** the navigator already had, now shared, so "square enough" means one
  thing however the canvas is being turned.
- **The gesture outlives the second finger.** It is born when a second lands and
  buried when the *last* one lifts, so a pinch that ends with one finger still on
  the glass keeps panning instead of going dead under a hand that never left; and
  one finger of several lifting ends nothing, or the gesture would end on
  whichever finger the hand happened to raise first. A third finger is a
  bystander — the pair is the first two — so a hand resting on the glass mid-pinch
  does not fight it.
- **A second finger cancels the stroke the first was drawing** (`Cancel`, not
  `End`): reaching for the canvas must leave no mark. That is also now what a
  middle-drag begun mid-stroke does, since both are the same question — `Nav`
  answering "this press is navigation, not yours".
- Stale fingers are ruled out rather than swept up: a *primary* touch is by
  definition the first contact of its type, so anything still listed when one
  arrives is a release that never came, and the set is cleared on that fact
  instead of on a list of the ways a release can go missing. One stale entry
  would make the next lone finger a pinch and stop touch painting for the rest of
  the session.

**Still missing** is everything else a tablet wants: a two-finger tap for undo, a
long-press eyedropper, and hit targets sized for a thumb — the chrome is still
laid out for a mouse.

#### 18.1.8 Quick brushes — the number keys, and the pen's other end — built

Every competitor binds keys to **tools**: B for brush, E for eraser, R for blur.
Stark has no such list to bind. An eraser here is a brush whose `lift` is up and
whose `add` is zero; a blur is one with `bleed` up; a smudge is `lift`+`deposit`
(§6.2). They are points in one parameter space, not entries in a menu — which is
the engine's whole claim about what a tool is, and it leaves the conventional
binding with nothing to name. A key that selected a tool would have to select a
*brush*, and which brush is the artist's answer, not ours.

It also does not fit the hardware. The model those bindings come from is a mouse
and a full keyboard; the hand this application is for holds a pen, rests on a
tablet, and has one spare finger for the number row at most.

So **the numbers hold brushes**, and there is exactly one rule:

> A held number is a temporary swap of the live brush. Whatever you change while
> it is held stays with the number; the brush you were holding comes back when
> you let go.

Everything the feature does falls out of that instead of being wired up three
times — this is the whole of `stark-ui/src/slots.rs`, and the panel and the
engine learn nothing:

- **Hold and draw** and the stroke is the number's, because the slot's brush *is*
  the live brush for the length of the hold, and a stroke takes its copy of the
  brush at `Start`.
- **Hold and click a preset** and the preset lands on the live brush, so at
  release it is what the number keeps.
- **Hold and drag Size or Flow** and the panel's sliders write the live brush, as
  they always did. The Brush panel shows the live brush; while a number is held
  the live brush is that number's, so it shows and edits the slot without a line
  of code that knows about slots.
- **Flip the pen over** and the eraser end holds slot 0 for as long as its tail
  is on the glass — *whatever it is pressed against*. The same hold, made by
  hardware rather than by a key, and bound at the window rather than by any one
  surface (`input::bind_pen`), so it earns all three of the lines above rather
  than only the first: erasing with the tail is one gesture, dragging Size or
  Flow with it tunes the eraser, and eraser-clicking a preset assigns that
  preset to the tail. A key and a hand do the same thing. The eraser is
  therefore a brush like any other and can be replaced with any other.

  Bound in the **capture** phase, which is load-bearing on the press: the swap
  has to be in force before the surface's own handler runs, or the canvas would
  open its stroke on the brush the eraser displaced. Capture runs window-inward,
  so it is ahead of every handler in the tree, and it cannot be silenced by a
  `stopPropagation` downstream — which matters most on the release, where a
  listener that could be skipped would strand the brush swapped. The press is
  tested strictly (it must really be the eraser, or the tip would erase); the
  release is any **pen** leaving the glass, since a stylus has one contact and a
  driver that reports the release without the eraser bit still has to end the
  hold. A finger's release is left alone, so a palm settling mid-erase does not
  hand the brush back under a pen that never moved.

Three things the rule has to get right, each a place a looser design goes wrong:

- **An unused hold keeps nothing.** Compared against the brush the hold *entered*
  on, so holding 5 and drawing does not quietly make 5 whatever was in hand. A
  slot that filled itself on first press would be indistinguishable from one the
  user had set.
- **Color is not part of a slot**, in both directions — the same rule the preset
  library already states, now shared as `presets::wear` rather than restated.
  Swapping never changes the color you are painting with, a color picked
  mid-hold survives the release, and the "was anything changed?" test is
  `presets::matches`, which is exactly *the same brush, color aside*. The
  brush's own opacity (`color[3]`, a material property — §6.1) does travel, as it
  does in a preset.
- **A hold ends only for whoever made it.** The grip is carried through, so a
  keyup cannot end an eraser stroke and lifting the pen cannot release a key
  still under a finger; the slot is carried too, so a hand rolling from 3 to 4
  and off 4 first does not end the hold 3 still has. And a `blur` on the window
  releases whatever is held, because focus leaving is the one way a key ends
  without ever sending its keyup.

The rack is read off the **physical** key (`code`, not `key`): on a French layout
the digit row types `&é"'` unshifted, and a rack reachable only through Shift
would be no rack.

**Holding a number draws the rack** (`slots::SlotOverlay`): a column down the
left of the window of the brushes the digits carry, each shown as the rendered
test stroke the preset library shows it by (§11) and named where the slot still
*is* one of the presets. The held row is ringed; an empty digit appears when it
is the one being held, since holding an empty number is not a mistake but how it
gets its first brush.

It replaced a permanent row of ten chips at the head of the Brush panel, and the
trade is the point. The chips spent the scarcest space in the app — panel height
— every second of every session to say a digit and three lit states, and the
digit is the one thing about a quick brush nobody needs told: what a rack of ten
unlabelled numbers actually raises is *what is on 4*. Being momentary is what
pays for the answer. On screen only while a finger is on the key that summons it,
the overlay can afford the width of a real preview and costs the panel nothing —
and the previews were already there to be shown, one cache keyed by the brush
itself, so a slot that came from a preset is rendered once for both viewers.

**The Panels menu can keep it up** ("Quick brushes" — the one entry in that menu
that is not a panel in the stack), and that is what makes the rack *clickable*:
pinned, a row takes the pointer and clicking one applies that slot for good
(`slots::pick`). It is the mouse-only way to a slot, and the reason the pin
exists — the rule this whole feature is built on needs a keyboard, and a hand
with a pen in it and a tablet under it has no spare finger for the number row.
Tapping the key still does not apply a slot, for the reason below.

Transient, it takes no pointer at all: the gesture it belongs to then is
hold-*and-draw*, and the hand is very often painting directly under it. Pinning
is the user asking for that strip to be a control and paying for it in canvas,
exactly as opening a panel is. The container declines the pointer either way, so
the gaps between rows and the column below the last one are never anything but
painting.

**Two marks, two questions.** The held row is ringed; the row the live brush
still *is* (color aside) is lit in the preset rows' own blue, on the same test —
so a slot and a preset holding one brush light together and say they are two ways
to it. They only look like one question while a key is down. Pinned and idle, the
lit row is the only thing saying which of the ten is in hand, which is what the
chips' `active` used to be for. Held wins where both apply: it is what the user is
doing, not a state they are in.

Reading the live brush costs nothing per stroke, and that is not an accident of
this component: a sample dispatches *quietly* and never refreshes the observable,
so chrome that watches the brush re-renders when the brush changes rather than
while one is being painted.

It does fade with the rest of the floating chrome, which goes to nothing while a
canvas gesture is in flight — it stands over the painting like the panels and the
bars, and while a stroke is being laid the screen goes back to being the
painting. What makes that safe on a thing this momentary is that **the hold
outlives the stroke**: the key is still down when the pen lifts, so the rack
returns and the answer is there whenever the hand wants it, not only before the
first mark. Appearing, though, is instant — the transition fires on the dim and
the return, not on the insert, which is the right way round for something
summoned by a finger that is already down.

Tapping the key deliberately does not apply a slot, because a tap and a hold are
one keystroke told apart by how long it lasted, and binding them to different
outcomes would make every hold a race against the user's own reflexes. A click is
its own gesture and says what it means — which is why the mouse's way in is a
click on a pinned row and not a shorter press of the same key.

Clicking a row *during* a hold is left meaning what the one rule already makes it
mean: the click puts that slot's brush on, and the hold keeps whatever is live
when it ends, so holding 3 and clicking 5 copies 5 onto 3 exactly as holding 3
and clicking a preset assigns that preset. One rule, no special case.

The hold that summons the overlay is a **key** hold alone, which is the one place
the two grips are told apart rather than being the same hold. The pen's tail
holds slot 0 for as long as it is on the glass — that is every erase stroke — and
a rack flying in and out of the corner of the eye on each one is noise answering
a question nobody asked. Holding `0` shows the same row. (A rack that is *pinned*
shows the tail's slot anyway, lit rather than ringed, because that slot's brush
is the live one for the length of the contact.)

A browser that has never set a slot is seeded from the preset library: each
shipped preset declares the digit it **ships on** (`PresetEntry::slot`), and the
rack is filled from those. So a tool reaches the keyboard under the same name and
with the same parameters the panel lists it by, adding one is a field rather than
a second table, and the two orders stay free of each other — the eraser is last
in the list and first on the keyboard, because a list is read top-down and a rack
is reached by the digit under the finger. It is also why `slots.rs` defines no
brush of its own: what a slot starts as is a question about the app's tools, and
those live in one place.

The seed is in memory and unpersisted, so storage is written only by the user's
act, an improved default reaches the rack on the next start exactly as it reaches
the list, and a start whose bundled shapes failed to fetch cannot freeze a
degraded preset into a slot.

**The app's own presets are code, not data** — built fresh every start and never
written to storage (`presets::default_presets`). That is what makes them
*updatable*: while the engine's own parameters are still moving under them, a
default that cannot be improved for a browser that already ran Stark is a fossil
of whichever version got there first. A stored preset whose name collides with a
shipped one is dropped on the merge, which is where the old scheme's persisted
copies go.

It also settles what the rows show. The user's carry a trash; the app's carry a
lock — not a rule the panel enforces but a fact it reports, since there is
nothing stored behind a built-in to remove and the next start would rebuild it
regardless. Saving under a shipped name is refused for the same reason rather
than offered as a replace: the work would not survive the next start, and a
second row of that name would make "the preset called Pen" two brushes to every
lookup by name.

**Not built**: a second row (Shift+digit) for twenty, any way to *assign* a slot
without the keyboard (dragging a preset onto a pinned row, say — applying one
needs no keyboard, filling one still does), and clearing a slot back to empty — a
slot is overwritten today, which is the only operation the rack has needed.

#### 18.1.9 Modifier drags — zoom, size and flow under the hand — built

The gestures every raster editor binds to a modifier plus a drag, because the
alternative is crossing the screen to a slider between strokes. Both are
*bindings*, not features: each is a few lines over machinery that already
existed, and neither the engine nor the panels learn anything.

- **Space + accelerator + drag zooms**, where space alone pans — the scrubby
  zoom (`input::Nav`, `Mode::Zoom`). **Right and up both zoom in**, Rebelle's two
  directions taken together rather than one or the other, so the hand does not
  have to know which axis this app picked; summed rather than projected onto the
  diagonal, so a drag along either axis alone runs at exactly the stated rate.
  **Exponential** in that distance (~180 page px per doubling), which is what
  makes the gesture feel the same at every scale — a fixed step per pixel added to
  a multiplicative quantity crawls when zoomed out and leaps when zoomed in. The
  rate is set from the range it has to cover rather than by taste: the view's
  whole zoom range is about ten doublings, so one screen width of drag takes the
  canvas from one end of it to the other. Being linear in the pointer's position,
  the zoom is a function of where the pointer *is*, so a drag that wanders out and
  back leaves the canvas where it found it. The anchor is the **press** position
  and stays there for the whole gesture — a zoom is a scale about a point, and
  re-anchoring it each move would slide the canvas out from under a hand that is
  still scaling.
- **Accelerator + drag tunes the brush**: sideways is **Size**, up and down is
  **Flow** (`input::Tune`) — the Brush panel's own two knobs, which is what earns
  them the binding.
- **The size is stated, not nudged.** The radius is a **quarter of how far the
  drag has reached from the press**, in canvas px — so the gesture does not adjust
  the size, it *describes* it, and the ring drawn at the press point is the picture
  of what it says. Left and right are the same gesture, because the hand is
  describing a circle and a circle has no side. A quarter, so the diameter is half
  the drag: the cursor stays outside the circle it is drawing instead of sitting
  in the middle of it. Since the canvas radius is the screen travel over the zoom,
  the ring is a quarter of the drag *on screen at any zoom* — the gesture measures
  the same in the hand, and what the zoom changes is the size in canvas px, which
  is the thing being set. Being a function of where the pointer *is*, it cannot
  drift over a long gesture, and it needs no read-modify-write of the brush.
  The zoom is latched at the press, so a wheel notch mid-drag (the pointer is
  captured; the wheel is not) cannot move the scale under a hand that is holding
  still.
- **Flow stays a rate** — linear, the whole 0..3 range over ~800 px — because
  there is nothing for it to be a picture *of*: a size drag can be shown as the
  circle it asks for, while flow has no length on screen to be measured against,
  so the honest mapping is every slider's. Its zero is also a value it must be
  able to reach, and no number of halvings gets there.
- **One knob per gesture.** The drag commits to an axis once it has travelled
  8 px — far enough for its direction to mean something — and keeps it. Both at
  once reads better on paper and is worse in the hand: flow's useful band is
  narrow enough that the incidental drift of a long sideways drag would empty or
  bury the brush, and there would be no way to ask for size *alone*.
- **The size drag draws itself** (`state::BrushRing`, `BrushSizeRing`): a ring at
  the radius being asked for, with the radius it started from dashed behind it, at
  the press point. Not decoration — the size is *defined* as a distance from that
  point, so the circle is the readout, and the old size beside it is what turns "it
  will be this big" into "it will be bigger than the last stroke". It comes up on
  the press, before the drag has said which knob it is about, which is also the one
  thing that makes the binding discoverable: press with the accelerator held and
  the brush draws itself. It goes again the moment the gesture turns out to be
  about flow, rather than sitting there advertising a number that is not moving.
  A `<div>` like the peer cursors and for their reason — it is chrome, and must
  never reach an export — which is affordable because a circle in canvas space is
  still a circle on screen at any angle or handedness, so a radius through the zoom
  is the whole of the transform. A **circle**, for now, though the brush may be any
  shape (§6.6): what the drag sets is one number, and a ring is the honest picture
  of one number, where an outline of the real tip would be a picture of the *shape*
  — which this gesture cannot change — and would claim a soft brush's mark is that
  crisp.
- **The rack and the pen's tail come for free**, and that is the test of where
  this was put: it writes through the same `update_brush` the sliders do, so
  while a number is held the live brush is that slot's and the drag tunes the
  slot (§18.1.8), and the same drag made with the eraser end tunes the eraser.
  It clamps to the sliders' own bounds (`panels::brush::MIN_RADIUS`,
  `MAX_RADIUS`, `MAX_FLOW` — now named once for both readers), so a drag cannot
  put the brush somewhere the panel is unable to show or take back.
- **The chrome does *not* fade** while tuning, unlike every other canvas
  gesture, for the eyedropper's reason (§18.0.2): the Brush panel is where this
  gesture's answer is read, and fading it out would hide the one thing the drag
  is for. The sliders move under the hand, which is the whole readout.
- Both accept **Ctrl or Command**, everywhere, rather than asking which platform
  this is (`input::accel`, already what the keyboard shortcuts did) — a binding
  that insisted on Ctrl would be unreachable on the one platform where Ctrl+drag
  is how the browser reports a secondary click.

**Still missing**: the number beside the ring, an indicator for flow (which is why
its drag is the one whose only readout is the panel), and a ring under the resting
cursor rather than only during a drag. Deliberately *no* cursor change while the
accelerator is merely held, unlike Alt and the eyedropper: Ctrl is also the front
half of Ctrl+Z, and flashing a resize cursor over the artwork on every undo would
cost more than the hint is worth — the press is early enough to say it.

### 18.2 Tier 2 — where we can beat the prior art

Photoshop's history is a bounded, linear, destructive stack that vanishes when
the file closes. Ours is a complete, deterministic, replayable log of id-tagged
actions that *is* the save format. Three things follow that no competitor can
ship without a rewrite.

#### 18.2.1 Post-hoc stroke editing

Change a committed stroke's color, brush or dynamics and replay. "Every stroke
stays editable" is a vector-app promise delivered with natural media. It fits the
CRDT cleanly: an amend is a **new action referencing the target** — exactly the
`Undo(ActionId)` shape from §5.4 — not a mutation. Grow-only stays grow-only, and
`effective_actions` generalizes to "latest amendment wins" without touching
`Action::apply`.

#### 18.2.2 Branching history

Try a variation, keep both, compare, pick. `Timeline` is already a trait; a
branch is a second effective-sequence over the same log.

#### 18.2.3 Selective undo

`ActionKind::Undo(ActionId)` already means "derive the document as if `target`
were absent". The hard part is built and only "undo the last thing" is exposed.

#### 18.2.4 Timelapse as a tool, not an export — shipped

Exposed as a **scrubber the artist can drag while working**, it is a real
critique tool — seeing your own process is how you find the moment a piece went
wrong — rather than a novelty output. Shipped as **Timeline mode** (☰ → Timeline;
`stark-ui/src/panels/timeline.rs`): a bar carrying a transport, a per-action
scrubber and a speed control, over `Timeline::seek`.

The whole feature is one observation: `LinearTimeline` already holds an *applied
prefix* and a *withheld suffix*, and undo and redo already move the boundary
between them one step at a time. A scrubber is that boundary with a handle on it,
so the mode stores no playhead of its own — and the two behaviours that would
otherwise need designing fall out instead. Leaving the mode scrubbed back **is**
being undone by that many steps (Redo is the way forward); painting there
truncates the future exactly as painting after an undo does. Playback is the only
thing that has to be guarded, and only against commits: a stroke laid under a
moving playhead would clear the withheld half and take the rest of the piece with
it, so the canvas refuses paint while the transport runs (panning still works).

Two limits worth stating:

- **Solo only.** A shared document's state is a function of a log peers are still
  appending to, so a scrub would be silently undone by the next arrival;
  `Timeline::scrub_range` answers `None` there and the bar says why.
- **Steps, not seconds.** Playback is paced in actions per second, not by the
  `InputSample.time` stamps the log carries. Wall-clock pacing wants an idle-gap
  policy — an hour away from the easel is one action apart — and that is a
  decision to make deliberately rather than to fall into.

### 18.3 Deliberate non-goals

Naming these is part of not becoming Photoshop.

- **Animation.** CSP and Procreate both have it. It is a product fork, not a
  feature, and it would pull the action log toward a timeline model that serves
  the painting case worse.
- **Text and vector layers.** CSP's turf, orthogonal to the oil-painting
  positioning, and a large surface area (fonts, shaping, i18n) for work that is
  not painting.

### 18.4 Sequencing

1. **Now** — layer masks / alpha lock / merge (§18.1.1);
   frame snapping, guides and review mode (§15.7–§15.8); the transform clipboard
   (§16.7).
2. **Then, in parallel** — brush parameter mapping and the brush library
   (§18.1.4; deepens what is already the strongest asset); symmetry and
   perspective guides (§18.1.3; highest value per line of code on this list, and
   shape assist has now paid for the seam they attach at).
3. **The bet** — post-hoc stroke editing and branching history (§18.2.1–2).
   Structurally impossible for Adobe to match, and already 80% paid for.

---

## 19. Stability policy — drawing files

- **Alpha** (current): no guarantees whatsoever. Files may not load at all in
  future versions. If you care about a piece, export it as an image.
- **Beta**: files will continue to load, but portions may be lost or changed.
- **Release**: files will continue to load and be perceptually similar. This does
  not guarantee pixel-perfect reproduction, even with the same version — see §8
  on `app_build` and cross-build fidelity.

---


