# Filter layers

The third kind of layer: one that is a *function of what is beneath it* rather than content of its own — §21.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.
> One name per thing: [glossary.md](glossary.md).

## 21. Filters

A paint layer adds light to the stack. A matte covers it (§15). A **filter layer**
rewrites it. That is the whole feature, and everything below is a consequence of
saying it that way rather than any of the ways it is usually said.

### 21.1 The stance: scope is position, not a mode

Every other application answers *what does this adjustment apply to?* with machinery
bolted on beside the adjustment: Photoshop clips the adjustment layer to the one
below, or masks it, or asks you to drag it into a group; Procreate applies the
adjustment destructively to the active layer and calls the question closed. All three
are answers to a question the layer stack already answers.

> A filter layer adjusts **everything composited beneath it in its own stack**.

That is the same set a `clip` reads (§14.4), and it is not a coincidence — it is the
same sentence. So the four things a scope control is usually for are all one gesture
in the Layers panel:

| To filter… | …put the filter |
|---|---|
| the whole painting | at the top of the document |
| one group | at the top of that group |
| exactly one layer | carried **onto** that layer (§14.2) |
| everything up to a point | at that point in the stack |

There is no clipping toggle to teach, no mask to author, no "adjustment layer applies
to layers below" footnote. The one control is where the row sits, and the panel is
already a picture of where every row sits.

Two consequences worth stating because they are load-bearing later:

- **A filter is not destructive and never was.** It touches no tile. Its whole effect
  is one fullscreen pass at composite time, so it costs no GPU memory, is free to
  re-tune at any point in the session, and is removed by dropping one action.
- **The rest of a filter's behaviour is a layer's.** Visibility, opacity, naming,
  reordering, removal, duplication, undo, save, replay, collaboration — §5, §8, §12
  and §14 need no new argument, exactly as §15.3 argues for the matte.

### 21.2 Reach, where it is empty, and the arrangement that cannot exist

"Beneath it in its own stack" is the accumulator that stack has built so far, and the
compositor gives that meaning for free: a filter inside a group reads the group's
isolated accumulator, and a filter at the root reads the document's.

It reaches nothing at **the foot of a stack** — nothing has been composited yet — and
above nothing but layers the draw list culls: hidden, fully transparent, never
painted. In both, the filter is the identity, and the draw list leaves it out rather
than encoding a pass that provably cannot change a texel (§21.3). The panel says so:
`LayerInfo::has_underlay` is the projected predicate, and it is deliberately **not**
`has_backdrop`. It counts the *carrier's own content* — a base composites at the
bottom of its group (§14.1), so a filter carried onto a painted layer has that paint
beneath it even as the first carried row — and it counts only what would actually
draw, because it exists to answer for the renderer, not for the row order.

One arrangement is ruled out rather than special-cased: **a filter never carries.**
A group's members composite *over* their base (§14.1), so a filter that had layers
dropped onto it could reach none of them — an arrangement that means nothing, and
whose every consumer would have needed a rule for it. `DocState` refuses to build it
(every carrier-attachment path declines a filter carrier, deterministically), so no
local gesture, no replayed file and no peer can create it; the panel's drag never
offers the inside of a filter row as a drop target, on its own rule that it draws no
place it cannot drop into.

### 21.3 Compositing: the blend pass with the source removed

A filter is a pass that reads the accumulator and writes it back. That is exactly what
a blend pass does with the isolated source deleted, so it takes the same shape and the
same budget:

```rust
pub enum GroupContent {
    Run(Vec<CompositeItem>),
    Stack(Vec<CompositeGroup>),
    Filter(FilterDraw),          // <- no content: a function of what is already there
}
```

- **The same ping-pong.** A texture cannot be read and written in one pass, so the
  accumulator alternates between the caller's targets and the level's `swap`, and the
  parity trick that lands the final result in the caller's own targets (§6.3) counts a
  filter as one flip like any merge. `scratch_levels` already counts it, because
  `as_direct_run()` does not claim it.
- **The same scratch.** A filter needs a level's `swap` and does not use its `iso` —
  and a level allocates only the half its stack actually uses (`Plan::scratch`), so a
  document whose *only* non-`Normal` thing is a filter pays for the ping-pong pair
  alone.
- **No effect on a document without one.** Every existing golden is unchanged, which
  is the evidence: adding a variant to `GroupContent` changed no draw a paint-only
  document issues.

`FilterDraw` is deliberately not a `Filter`: which `u32` a filter kind is numbered is
a fact about `filter_common.wesl`, the split `blend_code` already makes for a blend
mode — and the `FILTER_*` codes are **mirrored** into the host by the generator
(§6.10), so `FilterDraw::new` names the shader's own constant and the two sides
cannot drift. Flattening the layer's opacity into it as a *strength* at the same time
is what leaves the encoder one thing to write rather than two to remember to combine.

Two things are dropped from the draw list rather than drawn, and both are exactness
requirements rather than optimizations:

- **A filter with nothing beneath it** (§21.2) — it would read an empty accumulator
  and write it back, which is the identity in principle and a rounding trip in
  practice.
- **A filter at its neutral setting**, which is what a freshly added one holds. That
  is what makes *adding* a filter a step you take before deciding what it does, rather
  than an edit in itself, and `a_neutral_filter_changes_no_pixel` pins it to the byte.

#### 21.3.1 What the pass does *not* touch

Pass A's color target holds the stack premultiplied by coverage; its aux holds the
amount of paint (§6.1). A **point** filter — one whose output texel is a function of
its input texel alone — has an opinion about neither:

- **Coverage comes out as it went in.** The adjustment runs on the *un*-premultiplied
  channels, so a half-covered texel is adjusted like a full one rather than like a
  darker one, and the alpha is written back unchanged. A filter says what color the
  paint already there should be, not how much of it there is.
- **Height is copied across verbatim.** Relief is a property of the medium, not of the
  color. It is a real copy rather than a skipped attachment: the ping-pong means the
  pass's output targets are not the ones it read, so an aux left unwritten would hold
  whatever the previous bounce left there.

A **resampling** filter — one that says what is *where*, not what color it is — is
the other case, and it transports all three lanes together. The reason is the media
pass's own visibility law: paint shows only where coverage *and* height both exist
(§6.3), so a chromatic fringe displaced past a stroke's edge with its color alone
would land on zero height and be a fringe that cannot be seen. §21.10 says what
"together" means precisely; what survives from the two rules above is the shape of
the promise — a filter changes only what its kind is *about*, and both kinds leave a
strength-0 pass bit-identical to the backdrop. A **clip** is the layer-side control
over exactly that difference: it holds a resampling filter to the point-filter
promise, coverage and height included (§21.4.1).

### 21.4 What a filter borrows from a layer, and what it cannot

| | on a filter | why |
|---|---|---|
| `opacity` | the filter's **strength** | a mix from the untouched backdrop to the filtered result — which is what fading a layer already means |
| `visible` | on/off | as everywhere |
| `name`, position, removal, duplication | as everywhere | it is a layer |
| merging down | **offered**, for a point filter whose backdrop is the layer beneath it | baking the adjustment into that paint is the same picture, so it is a merge like any other (§14.11.7). A filter that *resamples* is refused: a tile pass has to be a pure function of canvas position (§6.4) and a gather is not |
| `blend` | **refused** | a mode describes how a *source* meets a backdrop; a filter has no source, it *is* the backdrop — and it can never be a group's base (§21.2), so no outward-pointing merge exists either. State declines to store one, like paint on a matte, rather than holding a value nothing can ever read |
| `clip` | the bound on **where its result may land** | §14.4's sentence read for a layer with no source of its own: the filter's result exists only where the backdrop it read had coverage. So a clipped filter hands coverage and height back exactly as it found them — see below, and §21.4.1 for why that is inert on a point filter and live on a gather |
| carried layers | **refused** | a filter never carries — see §21.2; the state declines the attachment itself |
| paint | **refused** | no tile map — the same refusal a matte gives (§15.7), in `apply` and in the preview path alike, so replay and peers agree |

Strength is mixed in the **working space** rather than in light, which is what makes
strength 0 the *exact* identity rather than the identity plus a round trip's rounding.

The panel shows blend disabled on a filter row for the reason §14.4.3 shows it inert
on the bottom row: a control that cannot express anything here should say so rather
than accept a value nothing reads. The clip chip beside it stays **live** — the row
is where the two halves of that shared question come apart, and §21.4.1 is why. The
opacity slider is relabelled **Strength** on a filter row — "50% opacity" on a color
adjustment invites the reading that the filter is half transparent, when what it is
is half applied.

#### 21.4.1 What clipping a filter means

> A clipped filter may say what color the paint already there should be. It may not
> say where there is paint.

That is §14.4's *this layer exists only where there is paint beneath it in its group*,
with the one substitution a filter forces: it has no content of its own to exist, so
what is confined is its **result**. The confinement is stated on the two lanes that
carry "how much paint is here" — a clipped filter writes back the coverage and the
height it read, unchanged.

Two consequences, and the second is the whole reason the flag exists:

- **On a point filter it is a bit-exact no-op**, and structurally so rather than by
  convention. §21.3.1's first rule already says a point filter writes back the
  coverage it read and copies the height across; there is nothing left for a clip to
  delete. In
  the shader that is not a branch that happens to compute the same thing — it is the
  same function: `filter_common.wesl`'s `resolve`, which no clipped-point path is
  written against because the unclipped one is already it.
- **On a gather it is the live case.** The chromatic filter (§21.10) transports
  coverage and height with the light it displaces, which is what lets a fringe be seen
  past a stroke's edge — and is exactly what a clip refuses. Clipped, the gather's
  color still lands, over the coverage that was already there: the fringe colors the
  silhouette from inside instead of spilling out of it. The clipped gather tails
  through `resolve` too, and that it is the *same* tail is the statement — a clipped
  filter is one with no opinion about coverage, which is what a point filter is.

Inert with nothing beneath, exactly as it is for paint (§14.4.3) — and the engine
agrees for free, because a filter reaching nothing is dropped from the draw list
altogether (§21.2). What the panel greys the chip on is `has_underlay` rather than
`has_backdrop`, for the reason §21.2 gives that predicate at all: a filter carried
onto a painted layer *is* reaching that layer's paint even as the first carried row,
and that arrangement is the "filter just this layer" gesture. Reading the positional
predicate there would leave the chip dead in the one place it is reached for most.

Nothing about **scope** changes. Where the row sits is still the whole of what a
filter acts on (§21.1); the clip does not narrow that set, it bounds what the pass may
write within it. A clipped filter at the top of the document still grades the whole
painting — it just cannot grow it.

### 21.5 The color filter

The first filter, and the one the architecture was built against. Five numbers, all
applied in **Oklab**:

```rust
pub struct ColorAdjust {
    pub exposure: f32,    // stops; light is scaled by 2^exposure
    pub contrast: f32,    // gain on Oklab L about mid-grey; 1 is the identity
    pub saturation: f32,  // gain on Oklab chroma; 1 the identity, 0 a true greyscale
    pub hue: f32,         // rotation of the Oklab (a, b) plane, radians
    pub tint: [f32; 2],   // offset added to Oklab (a, b) last; [0, 0] the identity
}
```

**Why Oklab, and why it is five and not seven.** In a perceptual space lightness,
chroma and hue are separable, and that separability is the whole promise a color
slider makes. Saturation in sRGB shifts hue. Contrast in sRGB shifts saturation. A
"greyscale" that is a luminance-weighted RGB average moves lightness around — visibly,
on a saturated red, by about 0.1 in `L`. Here a chroma gain is a chroma gain, so these
controls cover what a levels dialog needs seven to approximate.

**Three of the five are one gesture.** `hue`, `saturation` and `tint` are a rotation,
a scale and a translation of the same `(a, b)` plane — between them the affine map

```
ab' = tint + saturation · R(hue) · ab
```

— and the order is fixed: rotate, scale, then translate. Rotation and scale commute,
so between *those* two the order is free and rotating first is chosen only because it
keeps "how far from grey" reading as a distance rather than as a distance turned. The
translation commutes with neither, and it goes last for two reasons that turn out to
be the same one. It is what makes the tint **the color a grey becomes** — an
achromatic texel arrives at the origin and is fixed by both the rotation and the gain,
so it comes out holding the tint itself. And it is what lets §21.6 draw the three as
one circle: the rotation and the gain carry the circle's rim, the tint carries its
centre, and translating first would make the centre the image of the tint rather than
the tint, so the number under the pointer would stop being the number in the log.

**Why a tint at all**, when three knobs already covered the plane: because a gain and
a rotation both fix the achromatic axis, so no setting of them can put color into a
grey. Every cast, every duotone and every "warm the shadows" is a translation, and
until there was one the filter could take color out of a picture but never give a
picture a color of its own. `saturation: 0` with a tint is a toned monochrome — which
is the same operation, not a second one.

**Its bound is the gamut's.** `TINT` is `±0.16` per axis, which is about as far from
the achromatic axis as sRGB itself reaches at mid-grey: past it every texel in the
picture is out of gamut on the same side and the pass returns a flat wash whatever was
underneath. A square bound rather than a disc, because the pair is Cartesian for the
reason the shader adds it that way, and the corners a square admits are reachable
settings rather than a region needing a rule of its own.

**Exposure is the exception, and it is one on purpose.** It is applied to *light* —
before the trip into Oklab. Doubling light is what an exposure *is*; `L` is roughly
the cube root of that, so scaling `L` by `2^n` would be a number with no referent.
(The shader computes the gain on linear sRGB, which is the same operation: the two
encodings differ by a fixed linear matrix, and a scalar commutes with it.) The pass
is bracketed per color space exactly as the blend pass is — `filter_oklab.wesl` and
`filter_mixbox.wesl` supply only channels ↔ **Oklab**, and `filter_common.wesl` holds
the adjustment; Oklab rather than light as the interface because it is where the
adjustment happens anyway, so an Oklab document passes its channels straight in
instead of paying a conversion the first thing inside would exactly undo.

**Contrast pivots on mid-grey, not on the picture's own mean.** A pivot that depends on
what is underneath would make the slider do something different every time a layer
below it changed, which is not what a contrast control is. The constant is declared
once, in `filter_common.wesl`, mirrored into the host as `document::CONTRAST_PIVOT`
by the generator (§6.10), and derived rather than trusted by a unit test.

**One honest consequence in a pigment document.** Pigment cannot be brighter than the
light falling on it and Mixbox's inverse LUT is defined on `[0,1]` sRGB, so a positive
exposure saturates at white there instead of pushing past it into the media pass's
highlight roll-off the way it does in an Oklab document. That is the same thing
`blend_mixbox.wesl` says about `Radiance`, and for the same reason: paint does not glow.

**Every parameter is bounded and sanitized on the way in — twice.** A fullscreen pass
has no coverage to hide behind — a `NaN` saturation from a file or a peer reaches every
texel of the frame, and nothing downstream can notice. `Filter::sanitized` clamps to
the documented ranges and replaces a non-finite value with the *neutral* setting for
that knob, because `NaN` says nothing about which end was meant and the identity is the
one answer that cannot make a picture worse. It runs where the action is minted, so
replay puts back what was applied rather than re-deriving it (the funnel `SetLayerName`
already goes through) — and again where a filter **enters state**
(`DocState::set_filter` / `insert_filter`), because a loaded file's replay and a
remote peer's action never pass through the mint. Idempotent on any log this engine
wrote; the only line of defence against one it did not.

### 21.6 Interaction

A filter has **no permanent panel**, on the frame's argument (§15.7): creating one is
`+ Filter` in the Layers panel — a filter *is* a layer — and everything that is an
ordinary layer property stays the Layers panel's single set of controls for whatever is
selected. What is left is the filter's own numbers, in a bar mounted only while a
filter layer is **selected**.

- **There is still exactly one selection.** Selecting a filter is clicking its row, the
  same `PeerCommand::SetActiveLayer` that selects a paint layer, and the bar keys off
  `active_layer` being a filter. No second selection concept, so a filter that is
  removed, undone or replaced by a document load stops being tuned with nothing to
  invalidate.
- **`+ Filter` lands above the selected layer, in that layer's own stack** — the same
  placement `+ Layer` uses, and the decision that matters: adding a filter while
  working inside a group grades that group.
- **A row reads as what it is.** A solid rule and the funnel glyph, against the frame's
  dashed border: a frame bounds the piece, a filter runs across everything under it.
  An unnamed filter row shows the *filter's* name ("Color") rather than the word
  "Filter", because a stack of three rows all reading "Filter" would say nothing.
- **A filter that reaches nothing says so** (§21.2), once, in the bar — rather than
  greying out every control, which would each have to explain the same thing.

**The color filter's plane is a picture, not three tracks.** `hue`, `saturation` and
`tint` are one affine map of the Oklab `(a, b)` plane (§21.5), and the honest picture
of an affine map of a plane is *the image of a circle*. So the bar draws one, over a
flat Oklab `(a, b)` slice, and every part of it is a fact rather than a
decoration:

| what is drawn | what it is |
|---|---|
| the dashed circle | where the reference colors are — one moderate chroma, every hue — untouched |
| the solid circle | where the filter sends them |
| the arm and its dot | where it sends **red**, which is what makes the circle *directed* and a rotation visible at all |
| the centre | where it sends **grey**, which is the tint |

Which makes the three drags the three parts of the map: **expand** the circle and that
is saturation, **turn** it and that is hue, **move** it and that is the tint. A circle
without the arm would be the wrong picture — a rotation of a circle is invisible — and
that is the whole reason the reference locus is drawn *directed*.

The plane is one fixed slice, at the mid-grey the contrast knob already pivots about,
and there is no slice control: the map is a map of `(a, b)` alone and is identical at
every `L`, so a second slice would show the same circle over a different backdrop and a
control for it would be a control that cannot change a pixel (§1). It is also the one
place the flat `(a, b)` picture survives: the Color panel's own wheel is fitted to the
sRGB gamut and so is *not* flat (§11), which is right for choosing a color and wrong
under a handle whose whole claim is that it draws an affine map undistorted. The two
still agree about orientation — `+a` right, `+b` up, warm at the top — and one function
draws both backdrops (`panels::color::ab_field_data_url`), which is what keeps that
much shared. By the same
division, `exposure` and `contrast` keep their tracks — they move along the one axis a
slice of constant lightness has nothing to say about. They stack in a column beside the
dial rather than trailing off it in a row: the dial has already bought that height, two
tracks laid end to end would make the bar half again as wide as it is tall for nothing,
and they are a pair — the two ways to move `L` — so a column reads as one group. Each
wears a mark, a sun and a half-lit disc, which is what lets their words go in minimal
mode; the dial's three numbers wear none and want none, since the dial is a picture of
all three at once and a glyph beside it would be a second, worse one.

Two geometric details are load-bearing rather than cosmetic. The plane is drawn as far
as *the farthest the centre can travel plus the widest the rim can get*, derived from
the core's own bounds rather than picked, so no combination of a strong cast and a
strong saturation can carry a handle outside the element that receives the pointer.
And which handle a drag has hold of is decided once, on pointer-down, and held for the
gesture — a live hit test would turn a rotation swung in past the centre into a
translation halfway through.

**The chromatic filter's two numbers are one vector, and the pad draws the fringe
itself.** `spread` is a length and `angle` is a direction (§21.10), so what they are
between them is the displacement from where the red end of the spectrum lands to where
the blue end does — and two tracks cannot show that an arrow is one thing. So the bar
draws the arrow, as the thing it describes:

| what is drawn | what it is |
|---|---|
| the bar | the dispersion spectrum, at the pass's own wavelengths and weighed by the pass's own `dispersion_weight` — red end to blue end |
| its length | the spread: the full width of the fringe every edge grows |
| the handle | where the **blue** end lands; the far tip is the red |
| the centre | the picture, which does not move — the two ends part around it |

Which makes the drag the effect: pull the rainbow out of the middle and turn it, and
the painting does what the pad just did. The spectrum is the pass's own color science
rather than a rainbow someone drew — `color::dispersion_weight`, the host copy of
`filter_common.wesl`'s `ca_weight`, over wavelengths from `ca_lambda`'s own Cauchy
inversion, with the two endpoints and the Cauchy span coming through the build-time
mirror (§6.10) so the range drawn and the range integrated cannot drift. Two honest
adjustments separate drawing a response from integrating one: the run is normalized so
its strongest channel is full intensity (the weights are a response, whose absolute
scale means nothing — the pass divides it out too), and it is encoded to sRGB, because
that is what a screen takes. Relative brightness survives both, so the deep ends read
dark exactly as the eye finds them.

One handle, not two, and the whole field is its target: the vector is symmetric, so the
red end is wherever the blue end is not, and a single click anywhere is a complete edit
— the dial's centre, on the dial's argument. The one place the pad is a *scale* rather
than a picture is its radius, which is a **square root** of the spread: the interesting
fringes are two or three canvas px and `SPREAD`'s ceiling is 128, so a linear pad would
spend nine tenths of its radius on settings nobody dials and leave "2" and "3" a pixel
apart. Equal area per unit of spread costs nothing in honesty because the law states
itself — the graduation rings are a quarter of each other and so sit at half each
other's radius, rings crowding outward *are* the compression, and the number itself is
in the readout. The rim is the core's own `SPREAD.1`, and the drawn radius stops a
handle's width inside the field, so no reachable setting puts the handle outside the
element that receives the pointer — `DIAL_AB`'s guarantee, made the other way round.

**The controls preview live and log once.** Each pointer move sends
`ViewCommand::PreviewFilter` (view state, never logged) and the settled drag commits a
single `DocCommand::SetFilter`. This is the bargain the frame drag (§15.7), the canvas
color (§15.5) and the opacity slider (§14.6) already make, and it is the one this
feature could least do without: a color adjustment is judged *by looking* — how much
saturation is too much is a question about the painting, not about the number — so
every value the pointer crosses has to reach the canvas and only the answer belongs in
the log. It is affordable at pointer rate because a filter is presentation: a preview
costs one fullscreen pass and resamples nothing, unlike `PreviewTransform`.

The same two details the opacity slider needs apply unchanged: a settled drag always
commits (so a preview is never left standing), and a commit to the filter the layer
already holds is refused engine-side (so the out-and-back drag that forces costs no
undo step).

**The whole filter travels on every edit**, rather than one command per knob. The bar
reads the current settings off the projection, replaces one number, and sends the
result back — the read-modify-commit shape `DocCommand::SetGuide` uses for a guide's
whole camera (§20.5). So a filter that grows a parameter, and the next *kind* of
filter, need no new command, no new action, and no wire-format break.

### 21.7 Plumbing

Less new machinery than the feature suggests, because the layer model is already the
right shape.

- **Content.** A third `LayerContent` variant, `Filter(Filter)`. It holds no tiles, so
  `tiles()` is `None`, `bounds()` is empty, `with_tiles` is a no-op and `is_paintable`
  is false — the matte's answers, for the matte's reasons.
- **Actions.** `AddFilter { id, carrier, above, filter }` and `SetFilter(LayerId,
  Filter)`. Both **appended**, which is the one shape of change §8 allows without a
  format break, so `WIRE_VERSION` is unmoved. `AddFilter` shares `AddLayer`'s
  footprint arm — what a new layer *is* differs, where it lands does not.
- **Footprints and patches.** One new `Prop::Filter` and one `PatchOp::Filter`, at the
  granularity `SetFilter` writes: the whole filter, because that is what the action
  carries. A filter edit therefore commutes with a stroke, a rename and an opacity on
  the same layer, and not with another filter edit.
- **Peers.** Nothing new. A filter is document state reached by ordinary layer actions,
  so §12 needs no argument it does not already make; two peers tuning one filter
  conflict through `Prop::Filter` and the total order serializes them.
- **Shaders.** `filter_common.wesl` plus a variant per color space, mirroring
  `blend_common` / `blend_oklab` / `blend_mixbox` exactly. `linear_to_light` and
  `light_to_linear` moved out of `blend_common.wesl` into `lib/color.wesl` on the way:
  two passes work in light now, and importing the pair out of `blend_common` would have
  dragged that file's bindings along with it.
- **The uniform.** `Filter` in `filter_common.wesl`, mirrored into Rust by the
  generator (§6.10). Its parameters are two `vec4` lanes read according to `kind`, so
  the next filter reads those eight floats as its own and neither the host struct nor
  the bind group layout learns about it. (One lane until the color filter grew its
  tint; the second costs nothing, since the struct rounds up to the same 48 bytes
  either way.) The gradient map added a third kind of lane on the same terms: a
  sixteen-stop table for its ramp (§21.11), zeroed for every other kind, and read as
  its own by whatever kind next needs a table.
- **Bind group.** The blend pass's numbering with the source's two slots simply not
  declared: `filter_common` owns 0–2 where `blend_common` owns 0–4, and the pigment LUT
  keeps 5–6 because `mixbox_lut.wesl` hard-codes them for whoever imports it. The LUT
  itself is the blend pass's, decoded once — both passes ask it the same question.

### 21.8 Invariants worth a test (`tests/filter.rs`)

Three of these are about a filter doing **nothing**, and that is deliberate: a pass
that runs over every texel has no coverage to hide behind, so the cases where it must
be the exact identity are the ones where a mistake is a whole-picture change with
nothing on screen to say where it came from.

1. A filter recolors what is composited beneath it.
2. **Desaturating keeps the lightness it found** — measured as Oklab `L`, which is the
   test rather than a detail of it: a saturated red and its correct grey have very
   different *luminance*, so an assertion on luminance would fail on the right answer
   and pass on `dot(rgb, luma)`.
3. **The tint is the color a grey becomes** — a toned greyscale points at the tint's
   own direction in Oklab, at roughly its strength. That sentence is the definition
   (§21.5), and what it actually tests is *where in the adjustment the offset lands*:
   the ordering exists only in the shader, and the two knobs that would break it are
   exactly the two the panel draws around it — a rotation or a gain applied after the
   offset would turn the color under the pointer into some other color. A direction
   and not an RGB triple, because the media pass's tonemap moves the magnitude and
   must be allowed to.
4. A **neutral** filter changes no pixel, to the byte.
5. A **hidden** filter, and one at zero **strength**, each change no pixel.
6. A **carried** filter reaches only its own group — the layer above the group is
   untouched. This is §21.1's whole claim, drawn.
7. A filter with **nothing beneath it in its own stack** changes no pixel, in both
   forms: at the foot of the document, and above nothing but a hidden layer.
8. A filter **refuses carried layers** — the drop and the add alike — and the
   refusal is the state's, so replay and peers agree (§21.2). And the panel's
   "nothing below it" note agrees with the renderer: a filter carried onto painted
   content reaches it; one above only a hidden layer does not.
9. A filter can be **selected** but takes no paint, and the refusal is the engine's.
10. A filter **undoes** — the adjustment, and the add behind it.
11. A drag — a track or the dial, they go through the same funnel — **previews
    without logging**, and the commit renders what the preview showed.
12. A filter **survives save and load** — pixel-identical and setting-identical.
    `AddFilter` and `SetFilter` are the first actions to carry a `Filter`, and what a
    filter is made of — a nest of same-typed floats, an enum of three kinds, an optional
    stop list — is where a serialization mistake is *quiet*: it comes back as a
    different adjustment rather than as an error (§8). So every field is distinct on the
    way in, and the tint's two distinct from each other.
13. A filter works in a **pigment** document — the road out through Mixbox's
    polynomial and back through its inverse LUT, with the latent residual carried on
    both legs (§6.7). Nothing in an Oklab test touches that half.
14. Chromatic aberration **parts the spectrum, both ways** (§21.10): across a
    stroke, the red-versus-blue separation the filter adds has opposite signs on
    the two sides of the dispersion axis — the claim that distinguishes a spectrum
    pulled apart from a picture merely smeared, checked without pinning any pixel's
    exact hue.
15. **Deep inside flat paint the gather is the identity** — the partition of unity,
    §21.10's load-bearing normalization, measured where every tap lands on the same
    paint. A tolerance rather than bytes: the identity is exact in the linear-light
    algebra, and the trip out and back is not.
16. The chromatic filter **works in a pigment document** — per-tap decodes through
    the polynomial, one re-entry through the inverse LUT, the residual recomputed
    for the arrived-at color (§6.7).
17. **The black→white gradient map is the saturation-0 greyscale** (§21.11) — two
    kinds, two code paths, one picture, which pins the map's index (Oklab `L`),
    its interpolation space and its stop packing in a single render.
18. A gradient map **repaints the paint and only the paint** — an all-red ramp
    turns a blue stroke red while the bare canvas beside it comes through byte
    for byte, §21.3.1's coverage rule drawn.
19. A **rampless** gradient map changes no pixel — the kind's neutral, held to
    the byte like every other — and one **works in a pigment document**, where a
    saturated mapped color needs the residual leg to survive re-entry (§6.7).
20. **Clipping a point filter changes nothing, to the byte** (§21.4.1). The claim
    is that a clip is inert wherever the filter has no opinion about coverage, and
    a tolerance would not have made it: what is asserted is that the two renders
    are the same bytes, which is what "the same tail" means.
21. **A clipped gather stays inside the paint it filters.** The chromatic filter
    unclipped changes bare canvas beside a stroke; clipped, every texel the paint
    does not cover comes through byte for byte, while the stroke itself is still
    filtered. Both halves, because either alone is a different bug — no spill and
    no effect is a filter that did not run.

### 21.9 Open

- **The rest of the filters.** Motion blur, outline, blur, glow. A point filter is a
  `Filter` variant and an arm in `filtered()`; one that reads *neighbouring* texels
  follows the chromatic filter instead (§21.10), which already answered the two
  questions this bullet used to hold open — a kernel stated in canvas px reaches the
  supersampled accumulator through the view's own linear map, per frame, and a
  neighbourhood pass branches in each space's `fs_main` where the space's decode is
  in reach per tap.
- **Radial dispersion.** The chromatic filter's axis is uniform (§21.10) because an
  infinite canvas has no centre for a lens's field to grow from. The frame (§15) is
  the obvious candidate to donate one — dispersion growing with distance from the
  framed centre is the full transverse-aberration look — and it is a third knob on
  the same integral, not a new filter.
- **Per-filter masking.** Not needed for scope — position is scope (§21.1) — but a
  *soft* boundary (grade the sky and not the ground) has no expression yet. The selection
  is the obvious source, and §15.9's P4 region algebra is the obvious representation.
- **Filters on export.** They composite in pass A, so an export already carries them;
  nothing to do, recorded because it is the question everyone asks.
- **Merging a filter that resamples.** Refused today, and by §6.4 rather than by
  taste (§14.11.7): a gather is not a function of canvas position, so it cannot be a
  tile pass, and a tile's apron is one texel against a reach the document sets. The
  shape an answer would take is the *transform*'s rather than the merge's — composite
  the destination into a region-wide scratch, gather there, split back into tiles,
  bounded by a tile cap and declined deterministically past it (§16). It would also
  commit the fringe to canvas resolution, where the screen computes it in the
  supersampled accumulator, so the picture would soften when merged from a zoomed-in
  view; whether that is a merge or a different edit is the open question, not the
  machinery.

### 21.10 The chromatic aberration filter

The second filter, and the first that reads neighbouring texels. Most applications
draw this effect as three copies of the picture — red, green, blue — shifted apart,
which is not what a lens does and looks like what it is: three ghosts. A lens
disperses *every wavelength* by its own amount, and a channel then holds that
continuum weighed by the eye's response. So this filter is the integral, not three
samples of it:

```
out(x) = ∫ w(λ) · image(x − d(λ)) dλ
```

Two numbers describe the whole effect because two numbers describe the lens:

```rust
pub struct ChromaticAberration {
    pub spread: f32,   // red end → blue end, canvas px; 0 is the identity
    pub angle: f32,    // the axis, radians, canvas space — where blue is carried
}
```

Everything else — the rainbow ordering of the fringe, the blue end spreading farther
than the red, a flat field coming through untouched — is the physics, computed in
the pass rather than parameterized. Four decisions carry it:

**The taps are uniform in displacement, and Cauchy's law names their wavelengths.**
Dispersion in glass is `n(λ) ≈ A + B/λ²`, so a wavelength's displacement is linear
in `1/λ²` — the blue end of the spectrum spreads farther than an equal run of the
red end, which is half of why a real fringe looks the way it does. Sampling evenly
*along the fringe* and inverting Cauchy to ask which wavelength landed at each tap
puts the samples where the picture is (no tap is wasted where the fringe is thin)
and gets that asymmetry exactly, rather than as a tuned constant. The weights are
the CIE 1931 color-matching functions — the Wyman–Sloan–Shirley analytic fit, so
there is no table to bind — taken to linear sRGB and clamped at zero, since a
monochromatic color sits outside the gamut and a negative weight would ring where
a fringe should glow.

**The weights form an exact partition of unity, by construction.** Each channel of
the gather divides by that channel's own summed weight, so wherever the image is
locally flat the pass returns it *exactly* — for any tap count, at any spread, with
no normalization constant to tune or to get wrong. This is the property that makes
the filter honest at its edges: the fringe lives only where the picture changes,
because everywhere else the integral provably cancels. (The identity is exact in
the linear-light algebra; the trip out to light and back costs the usual rounding,
which is why *neutral* — spread 0 — is still dropped from the draw list rather than
trusted to a round trip, §21.3.)

**The integral runs in linear light, bracketed by each space's own decode — per
tap.** A sum over wavelengths is a sum of light; summing Oklab or pigment
concentrations would mean nothing. So this is the one filter whose body lives in
the space files rather than behind `filtered()`: an Oklab tap pays `Oklab → linear`
and a Mixbox tap pays `poly(c) + r → linear` (§6.7), the accumulated light re-enters
the space once at the end (for pigment, through the inverse LUT with the residual
recomputed), and `filter_common.wesl` supplies the machinery every space shares —
the tap count, the wavelength map, the weights, the sample positions.

**Color, coverage and height travel together.** §21.3.1's point-filter rules bend
here, and the media pass is why: paint is visible only where coverage *and* height
both exist, so a fringe that escaped a stroke's edge carrying color alone would be
invisible. The spectral weights carry the premultiplied light; their luminance
carries the coverage and the height, so the amount of paint travels with the light
that shows it. Transport conserves what §6.1 demands conserved: a shift-and-average
of the height field moves paint without minting any. The strength mix then runs on
the raw premultiplied lanes — all of them, together — so strength 0 is the backdrop
bit for bit, the same exactness `weigh` gives the point filters (§21.4).

**And a clip is what takes that back** (§21.4.1). This is the one filter with an
opinion about coverage, so it is the one filter a clip has anything to say to:
clipped, the gather's color still lands but its displaced coverage and height do
not, and the pass tails through `filter_common.wesl`'s `resolve` — the point
filters' own ending. The fringe then colors the silhouette from inside rather than
spilling out of it, which is what a painter wants when the aberration is a grade on
a shape rather than a lens over the whole picture. It is also the shape of the
result that a merge can bake into the destination's own tiles, since the tile set
does not grow.

**The knobs are canvas facts; the view arrives per frame.** The accumulator is the
supersampled render, and a distance in screen texels is not a distance in canvas px
(§6.4) — the question §21.9 held open. The answer is the media pass's own: the
document states `spread` and `angle` on the canvas, and the encoder carries them
through the view's full canvas→screen linear map — zoom, supersample, rotation and
mirror alike — into a dispersion vector written into the uniform each frame
(`chromatic_disp`, beside the slot writes). The fringes therefore stay attached to
the artwork exactly as the canvas substrate does, and an export disperses the same
canvas distance the screen showed. The pass buys taps in proportion to that
on-screen dispersion — enough that bilinear filtering closes the gap between
neighbours — floored so a sub-texel spread still integrates over a few spectral
bands, and capped (64) so an extreme spread at an extreme zoom stays one affordable
pass; `SPREAD`'s bound is what keeps the cap out of reach at working zooms.

Two honest edges, stated rather than hidden. The gather clamps at the viewport rim
— a tap displaced past the edge reads the rim rather than wrapping the far side of
the picture into a fringe — so within one spread-width of the frame's edge the
fringe leans on repeated rim texels; an export shows the same, at its own border.
And past the tap cap the spectrum quantizes gently rather than banding hard,
because the taps stay bilinear and the partition of unity still normalizes whatever
count ran.

In the bar (§21.6) the filter is one picture rather than two sliders: a pad holding
the dispersion **vector**, drawn as a bar of the real spectrum growing out of the
centre, through the same preview-per-sample / commit-once funnel the color filter's
dial and tracks use; `+ Filter` grew the picker `Filter::ALL` always promised, each
kind landing neutral. Zero spread is neutral **at any angle** — an angle dialled
before its spread is not yet an edit, so the draw list stays free to drop the pass
and `a_neutral_filter_changes_no_pixel` keeps its byte-level meaning. The pad says
the same thing by drawing nothing but its graduation at spread 0: a gradient over a
zero-length line paints nothing, which is the correct picture of *no wavelength
moves*.

### 21.11 The gradient map filter

The third filter, and the first whose payload is a **value** rather than knobs: a
captured `Gradient` (§22), read as a transfer function. The backdrop's **Oklab
`L`** indexes the ramp and the ramp's color is what the paint becomes — chroma
and hue beneath are spent, deliberately, because that is what the operation *is*:
a duotone, a false-color relief, a palette unification over a value study are all
this one map with different ramps. It is a pure point filter, so §21.3.1 applies
unbent — coverage comes out as it went in, height is copied across — and paint
that is not there cannot be graded into being.

**`L`, not luminance**, and the choice buys an exact sentence. `L` is the
lightness the rest of the chapter already answers for — "desaturating keeps the
lightness it found" is *measured* in it — and with `L` as the index the
black→white ramp samples to `(L, 0, 0)`, which is precisely the color filter's
saturation-0 greyscale. Two kinds, two code paths, one picture;
`tests/filter.rs` pins the agreement, and with it the index, the interpolation
space and the stop packing in a single render.

**The lerp is `Gradient::sample`'s, in Oklab, in every document.** The library
strip, the CSS preview and the pass are one interpolation (§22.3's invariant,
doing its job a third time). This is a deliberate contrast with the fill and the
matte, which interpolate their stops in the **working space** so a pigment
document lays a pigment ramp (§22.4): those *lay paint*, and a ramp of paint
should mix like paint — a map is a color **adjustment**, and adjustments are
defined in Oklab here (§21.5's argument). In a pigment document the mapped color
still re-enters through the inverse LUT with the residual recomputed, exactly as
every point filter's answer does (§6.7).

**Its neutral is no ramp at all.** There is no identity ramp — even black→white
repaints every color with its greyscale — so the kind's payload is an `Option`,
`None` neutral, and a freshly added map changes nothing until a ramp is chosen:
the same shape as an angle dialled before its spread (§21.10), keeping §21.3's
byte-level neutral rule and costing `Filter` its old `Copy` (the day its own doc
comment named). Sanitizing holds the one thing `Gradient::new`'s gate does not —
stop colors clamped to the sRGB cube — since a finite `1e30` reaches every texel
as surely as a `NaN` gain.

**The ramp rides the filter uniform**: a lane of sixteen `(L, a, b, t)` stops —
converted from the stops' sRGB once, host-side, in `FilterDraw::new` — with the
live count in `params.x`, zeroed for every other kind as `disp` already is. The
literal sixteen is asserted against `gradient::MAX_STOPS` where the mirror lands
(§6.10), so the fitter's bound and the shader's array cannot drift; like the two
scalar lanes, a future kind that needs a table of vec4s reads this one as its own.

**In the bar, the ramp is chosen where ramps live.** The gradient map's bar is
deliberately thin: the library's well wearing the ramp (§22.3 — clicking the
strip flies the library's pop-out out of it) and a **Reverse** chip — because
the choosing already has a home, and that home is on the strip itself. While a
gradient map is selected, clicking a row in the pop-out hands that ramp to the
filter, the same click-is-a-choice the composing fill and the matte honour
(§22.4); with no ramp yet, the bar shows an empty well that opens the same
pop-out, where one is picked or traced. A click is a discrete edit, so it commits one
`SetFilter` with no preview funnel, and re-clicking the ramp already worn is
refused engine-side (§21.6's out-and-back rule). Reverse is the one edit that is
about the *mapping* rather than the ramp: a trace runs in whatever direction the
hand drew it, the map reads dark at 0, and `Gradient::reversed` is the one-click
answer to a trace made the other way. Strength stays the layer's opacity, as
everywhere (§21.4).

---
