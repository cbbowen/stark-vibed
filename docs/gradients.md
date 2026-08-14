# Gradients

The gradient model, the trace capture that generalizes the eyedropper, and the
browser-local library — §22.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 22. Gradients

Every prior-art gradient editor asks the artist to place control points on a
strip and color-pick each one — a dialog's answer to a painter's question. The
observation this chapter is built on is that the colors an artist wants in a
ramp are usually **already on the canvas**, mixed by hand in the painting or on
a scrap corner of it. So Stark's gradients are *captured*, not authored: the
artist traces a line through their painting, and the machinery of control
points — where the stops go, what colors they carry, how many are needed — is
the engine's problem. The trace is the eyedropper generalized from a point to a
line (§18.0.2), and everything below follows from taking that sentence
literally.

The chapter runs from the gradient itself (§22.1) through the capture that
makes one (§22.2) and the library that keeps them (§22.3) to its consumers:
the **gradient fill** (§22.4), the position-varying `FillOp` parcel §18.0.4
and §10 always named, with the matte's graded paint beside it — and the
**gradient map** (§22.5), the filter layer that reads a ramp as a transfer
function rather than laying it as paint.

### 22.1 The model: stops in sRGB, a ramp in Oklab

`stark_core::gradient::Gradient` is a list of color stops — a position `t` in
`[0,1]` and a color — with three invariants held **by construction** rather
than checked by consumers: at least two stops, positions ascending, endpoints
at 0 and 1 (`Gradient::new` normalizes and refuses; deserialization funnels
through the same gate, so a stored or received stop list cannot smuggle in an
unsampleable ramp). A `Gradient` in hand is always sampleable, which is the
§1 habit of ruling out a class rather than enumerating its instances.

Stops store **straight sRGB**, because that is the convention on every CPU
color boundary — `BrushParams::color`, the matte and substrate colors, the
eyedropper's answer (§6.5) — and a gradient's stops are exactly that kind of
value: colors the picker could show and the brush could wear. Interpolation
between stops happens **in Oklab** (`Gradient::sample`), the same argument as
§1.6: a perceptually uniform ramp passes through the colors an artist would
mix on the way, where an sRGB lerp detours through grey. CSS interpolates
`linear-gradient(in oklab, …)` identically, which is what makes the panel's
preview strip *be* the gradient rather than a picture of it (§22.3).

Two deliberate absences:

- **No alpha on a stop.** A captured ramp is made of paint the canvas actually
  holds; what a fill does with coverage is the fill's parameter (§6.8's
  `FillOp` already carries one), not the gradient's. If a transparency ramp is
  ever worth having it is a wire-format decision to take then — inserting a
  field into a stop postcard-embedded in a `FillOp` is a version bump (§8) —
  not a `1.0` to carry meanwhile.
- **No per-stop interpolation modes.** One space, chosen once. A gradient that
  interpolates differently per segment is an authoring tool's feature; a
  captured gradient reproduces paint, and the fitter (§22.2) places extra stops
  wherever one interpolation law is not enough.

The stop count is bounded (`MAX_STOPS` = 16): past that the fitting tolerance
is allowed to give rather than the list allowed to grow, because sixteen stops
is already far more structure than a hand ever places and an unbounded list is
an unbounded uniform buffer when the fill lands.

### 22.2 The capture: trace → samples → stops

The pipeline is one request, `Engine::pick_gradient(path, options)` — a sibling
of `pick_color` in the request tier (§4): it must answer, so it cannot be a
command, and it renders synchronously returning a future for the readback, for
`export`'s reason.

**The trace never touches the document.** No action, no footprint, nothing a
peer sees, nothing replay needs — the whole interaction is view-side, ending in
a request. That is why there is no gradient *tool* in `Tool` (which would put
it in session state the engine forks gestures on) and no `GestureCommand`
involvement: the frontend collects the polyline itself under a full-viewport
catcher, the transform mode's pattern (§16.6), with `input::Nav` live on it so
the view stays reachable mid-trace. Points are kept in **canvas space**, so
panning or zooming mid-trace cannot corrupt a trace that is sound where it
matters. Release always ends the mode — a good trace captures, a too-short one
(a click that wandered) cancels, and either way the canvas is handed back.

Inside the request:

1. **Resample by arc length** (`gradient::resample`): evenly spaced samples,
   `t` = arc-length fraction. Spacing aims at 4 canvas px — comparable to the
   5×5 patch each sample averages, so consecutive patches overlap and the ramp
   cannot skip a color narrower than a patch — capped at `MAX_SAMPLES` = 128,
   which bounds what one trace may cost the way `MAX_PICK_RADIUS` bounds one
   pick.
2. **Pick every sample exactly as the eyedropper picks.** The same
   `PickOptions`, the same patch mean, the same sources, and above all the same
   raw-channels-not-lit rule: in a Mixbox document the samples are pigment
   mixtures with their residual, so the captured ramp is of paint, not of lit
   pixels (§18.0.2, §6.7). This is enforced structurally — `pick_color` and
   `pick_gradient` share one implementation (`pick_colors`), because "every
   sample is an eyedropper pick" is a promise two copies of the logic would
   quietly break. The batch renders one small patch per sample and reads the
   lot back through **one** buffer map (`read_many_rgba16f`), since a map per
   patch would be a frame of latency per sample on the web.
3. **Samples over bare canvas answer nothing and are skipped.** The
   `pick_color` rule stretched along the line: a stroke gap crossed mid-trace
   does not inject the paper into the ramp — the ramp runs from paint to paint.
   Fewer than two samples finding paint is no gradient at all, and the panel
   says so.
4. **Fit** (`gradient::fit`): three stages, each with one job.
   - *Median-of-3* per Oklab channel — a single outlier sample (the trace
     nicking a dark line) becomes a stop under any least-error criterion, and
     no artist means one sample of it. Endpoints are kept verbatim: they are
     the colors the artist deliberately started and finished on.
   - *Box-3 smoothing* — the patch average already handles texel noise; this
     handles sample-to-sample paint grain, so the fitter chases the ramp and
     not the tooth.
   - *Greedy stop insertion* in Oklab: start with the endpoints, repeatedly add
     the sample farthest from the current piecewise-linear ramp, stop when the
     worst error drops under `FIT_TOLERANCE` (0.01 — about a just-noticeable
     difference, Oklab L spanning `[0,1]`) or `MAX_STOPS` is reached.
     Farthest-point insertion rather than a corner detector because the
     criterion *is* the promise: nowhere along the trace does the fitted ramp
     drift a visible distance from the paint. A clean two-color blend fits to
     exactly two stops; a palette with a hard turn earns a stop at the turn.

`tests/gradient.rs` holds the capture to the promises: ends match an eyedropper
pick at the same points in both color spaces, bare canvas refuses, a mid-trace
gap stays a red-to-blue mixture with nothing foreign joining the ramp. The
fitter's own behaviour (two stops for a clean ramp, a stop at a turn, an
outlier ignored, the normalization refusals) is pinned CPU-side in
`gradient.rs`'s unit tests, GPU-free.

### 22.3 The library: the browser's, like the presets

A gradient is something the artist paints **with**, not part of what they have
painted — the same classification call as brush presets and the shape library
(§11), and the same consequences: entries live in the frontend
(`stark-ui/src/gradients.rs`), follow this browser across documents via
`localStorage` (`stark.gradients.v1`, one line per entry,
`b64(name)|b64(json(stops))` — line-oriented so one damaged entry is skipped,
JSON so the format outlives postcard's positional encoding), never enter the
document, and never reach a peer. When the gradient fill lands, the chosen ramp
is **embedded in the `FillOp` it commits** — the way a stroke embeds its brush
color — so documents stay self-contained and replayable with no reference
into anyone's library.

Unlike the presets there are **no built-in entries**: a gradient's whole story
is that it came off *your* canvas, and a panel opening on a stranger's sunset
would tell the opposite one. The empty state teaches the gesture instead.
Captures are named by the machinery ("Gradient N", first free) so the artist
can trace twice without a dialog between; rows wear the same trash every other
roster wears, and clicking one takes the ramp in hand for the fill (§22.4).
That click was deliberately withheld until the fill existed: while the library
was the whole feature, a selectable row would have promised an application
nothing could honour.

The panel (`PanelId::Gradients`, closed by default like the other
between-passages panels) shows each entry as a strip drawn by
`linear-gradient(in oklab, …)` from the fitted stops — the browser performing
the same interpolation `Gradient::sample` defines, so the preview cannot
disagree with what the engine will later fill with. Its Trace button arms the
capture mode and stays lit for the mode's life: the catcher it arms is
invisible, so the button is the mode's indicator as well as its switch.

### 22.4 The gradient fill: a parcel that varies with position

The fill §18.0.4 promised, at exactly the seam it named: `FillOp`'s paint
became a **`Parcel`** — `Solid` (a color) or `Gradient` (a §22.1 `Gradient`
embedded **by value** and a `GradientAxis`). Region, gate, stacking law, action,
footprint and inverse are all untouched: a gradient fill is gated by the
selection exactly as a brush is, deposits paint with real height, and stacks by
the shared parcel law. The ramp varies the parcel's *latent* only — one
`FillOp::opacity` for the whole fill, because a transition in thickness would
read as a lighting feature, not a color one. Beyond either end of the axis the
ramp holds its end stop: the fill covers its whole region, the axis only places
the transition.

A parcel carries **no opacity of its own**, in either variant. It says what
paint; how far the fill covers is `FillOp::opacity` — the Select panel's slider
(§6.8) — and asking the question once rather than per-parcel is what keeps a
gradient fill and a solid one answerable by the same control.

Two axis kinds, and both are **the drag, read two ways**: `Linear { from, to }`
is press-to-release along the ramp; `Radial { center, radius }` reads the same
drag as centre and reach. The composing UI keeps the two raw points and derives
the axis per kind, so switching Linear ↔ Radial on the bar reinterprets the
drag the hand already made instead of throwing it away.

**Interpolation happens in the working space, per fragment.** Every stop
converts on the CPU once per fill — `rgb_to_channels` *and* `rgb_to_resid`,
because in Mixbox the residual is half the color (§6.7) — and the shader
lerps between adjacent stops in those channels. In an Oklab document that is
exactly `Gradient::sample`, so the painted ramp is the panel's strip; in a
Mixbox document it is a **pigment ramp** — yellow-to-blue passes through green
the way paint does, which the sRGB strip cannot preview and which is the point
of painting in pigments. The fragment reads its canvas position through the
same per-tile origin discipline as the selection rasterizer, so the ramp is a
pure function of canvas position and a tile's apron cannot disagree with its
neighbour's interior (§6.4). The solid path kept its exact arithmetic and
lanes — branching on stop count, not restructuring — so the fill golden did
not move.

**The composing mode is the transform's, aimed at a fill** (§16.6). The
Selection bar — the fill needs a mask to be bounded by, and the bar exists
exactly when there is one — gains a **Gradient** button (disabled with an
explanatory title while the library is empty). It swaps in a bar of its own
(the ramp in hand as a strip, Linear/Radial, Done) and a full-viewport catcher
where the drag composes the axis. Every mutation funnels through one update
that issues `ViewCommand::PreviewFill` — the same `FillRenderer::apply` the
commit runs over the committed tiles, so **preview == commit bit-exactly**
(pinned in `tests/fill.rs`) and re-dragging previews one fill, never a stack of
glazes. "Done" commits a single `DocCommand::Fill`; entering Timeline mode
abandons the composition the way it abandons a transform. Nothing about strength
is captured at entry, unlike the brush opacity and `add` this mode used to take:
a gradient fill lays opaque paint through the selection, so how strongly it lands
is the selection's own coverage to say (§6.8).

This is what makes the library's rows *selectable*: "the gradient in hand" now
means something, so clicking a row takes it (the highlight always resolves —
an unset or orphaned choice falls back to the first entry), and picking a
different ramp mid-mode re-previews immediately. The choice is per-session
working state like the brush color, not a stored fact about the library.

The wire cost was taken openly: replacing `color` with the `Parcel` enum
reshapes `FillOp`, so `WIRE_VERSION` moved to 7 — and to 9 when the parcel's
own opacity went away in favour of `FillOp::opacity` (§6.8) — with old files
refusing rather than misreading, the §19 alpha policy, and the collab ALPN to
`stark/collab/1` (the
gossip path carries actions with no version of its own, so incompatible builds
must fail to meet rather than decode each other wrong).

**The matte is the second consumer** (§15.4, §15.5): `MattePaint::Gradient`
carries the same `Gradient` + `GradientAxis` pair the fill's parcel does, laid
by the same `ramp_position` leaf the two shaders share — so where `t = 0.5`
falls cannot drift between a filled region and a graded ground. It is composed
through the **same gradient bar**: the frame bar's Gradient chip enters the
mode with a matte target instead of a fill target, the drag is still the axis,
and Done commits one `SetMattePaint`. One interface for laying a ramp, wherever
the ramp lands. The one deliberate asymmetry: a fill reads its ramp live off
the library, while a matte target *carries* its ramp — re-composing an old
gradient's axis must not silently swap its colors for whatever the library
happens to have selected; a library click mid-mode still replaces it, because a
click is a choice.

### 22.5 The gradient map: the ramp as a transfer function

The third consumer, and the first that does not lay the ramp as paint: a
**filter layer** (§21) holding a `Gradient`, indexing it by the Oklab lightness
of whatever is composited beneath — dark paint takes the ramp's start, light
paint its end. The full design lives with the other filters, §21.11; what
belongs to this chapter is the seam. The ramp is the same embedded-by-value
`Gradient` the fill and the matte carry, chosen the same way — while a gradient
map is selected, clicking a library row hands it the ramp, the click-is-a-choice
rule of §22.4 — and `Gradient::reversed` exists because a *map* is the first
consumer for which the trace's direction has a meaning to be wrong about.

One deliberate asymmetry against §22.4: the fill and the matte interpolate
their stops in the **working space**, so a pigment document lays a pigment
ramp; the map interpolates in **Oklab**, `Gradient::sample`'s own space, in
every document. Those lay paint, and paint should mix like paint — a map is a
color adjustment, and adjustments here are defined in Oklab (§21.5). The
library strip previews the map exactly for the same reason it previews the
Oklab fill exactly (§22.3).
