# Gradients

The gradient model, the trace capture that generalizes the eyedropper, and the
browser-local library — §22.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 22. Gradients

Every prior-art gradient editor asks the artist to place control points on a
strip and colour-pick each one — a dialog's answer to a painter's question. The
observation this chapter is built on is that the colours an artist wants in a
ramp are usually **already on the canvas**, mixed by hand in the painting or on
a scrap corner of it. So Stark's gradients are *captured*, not authored: the
artist traces a line through their painting, and the machinery of control
points — where the stops go, what colours they carry, how many are needed — is
the engine's problem. The trace is the eyedropper generalized from a point to a
line (§18.0.2), and everything below follows from taking that sentence
literally.

What ships in this chapter is the gradient itself and the library of them; what
*consumes* a gradient ships at seams other chapters already name — the
position-varying `FillOp` parcel (§18.0.4, §10) first, a gradient-map filter
layer (§21) behind it. Nothing here anticipates them beyond being the value
they will embed: no fill mode, no map channel, no inert hook (§1's "nothing
inert ships").

### 22.1 The model: stops in sRGB, a ramp in Oklab

`stark_core::gradient::Gradient` is a list of colour stops — a position `t` in
`[0,1]` and a colour — with three invariants held **by construction** rather
than checked by consumers: at least two stops, positions ascending, endpoints
at 0 and 1 (`Gradient::new` normalizes and refuses; deserialization funnels
through the same gate, so a stored or received stop list cannot smuggle in an
unsampleable ramp). A `Gradient` in hand is always sampleable, which is the
§1 habit of ruling out a class rather than enumerating its instances.

Stops store **straight sRGB**, because that is the convention on every CPU
colour boundary — `BrushParams::color`, the matte and substrate colours, the
eyedropper's answer (§6.5) — and a gradient's stops are exactly that kind of
value: colours the picker could show and the brush could wear. Interpolation
between stops happens **in Oklab** (`Gradient::sample`), the same argument as
§1.6: a perceptually uniform ramp passes through the colours an artist would
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
   cannot skip a colour narrower than a patch — capped at `MAX_SAMPLES` = 128,
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
     the colours the artist deliberately started and finished on.
   - *Box-3 smoothing* — the patch average already handles texel noise; this
     handles sample-to-sample paint grain, so the fitter chases the ramp and
     not the tooth.
   - *Greedy stop insertion* in Oklab: start with the endpoints, repeatedly add
     the sample farthest from the current piecewise-linear ramp, stop when the
     worst error drops under `FIT_TOLERANCE` (0.01 — about a just-noticeable
     difference, Oklab L spanning `[0,1]`) or `MAX_STOPS` is reached.
     Farthest-point insertion rather than a corner detector because the
     criterion *is* the promise: nowhere along the trace does the fitted ramp
     drift a visible distance from the paint. A clean two-colour blend fits to
     exactly two stops; a palette with a hard turn earns a stop at the turn.

`tests/gradient.rs` holds the capture to the promises: ends match an eyedropper
pick at the same points in both colour spaces, bare canvas refuses, a mid-trace
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
colour — so documents stay self-contained and replayable with no reference
into anyone's library.

Unlike the presets there are **no built-in entries**: a gradient's whole story
is that it came off *your* canvas, and a panel opening on a stranger's sunset
would tell the opposite one. The empty state teaches the gesture instead.
Captures are named by the machinery ("Gradient N", first free) so the artist
can trace twice without a dialog between; rows offer the same trash every
other roster offers, and nothing else — a row is not clickable, because until
the fill lands there is nothing applying a gradient could mean, and a hover
highlight would promise one.

The panel (`PanelId::Gradients`, closed by default like the other
between-passages panels) shows each entry as a strip drawn by
`linear-gradient(in oklab, …)` from the fitted stops — the browser performing
the same interpolation `Gradient::sample` defines, so the preview cannot
disagree with what the engine will later fill with. Its Trace button arms the
capture mode and stays lit for the mode's life: the catcher it arms is
invisible, so the button is the mode's indicator as well as its switch.

### 22.4 What attaches here next

- **Gradient fill** (§18.0.4) — the `FillOp` parcel reads its latent from
  position rather than a uniform (§10's row); the op gains the fitted stops and
  a geometry (two points, linear first). Region, gate, stacking law, action and
  footprint unchanged — which is why the fill is *not* this chapter.
- **Gradient map** — a filter layer (§21) whose transfer function is a
  `Gradient`, indexing the ramp by the luminance beneath. The same embedded
  value, a different consumer.

Both consume a `Gradient` by value. Nothing about the library, the capture or
the model waits on either.
