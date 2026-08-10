# Filter layers

The third kind of layer: one that is a *function of what is beneath it* rather than content of its own — §21.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

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

### 21.2 Reach, and the two places it is empty

"Beneath it in its own stack" is the accumulator that stack has built so far, and the
compositor gives that meaning for free: a filter inside a group reads the group's
isolated accumulator, and a filter at the root reads the document's.

It reaches nothing in exactly two situations, and both are the *same* situation seen
twice:

- **The foot of a stack.** Nothing has been composited yet.
- **The base of a group.** A group's members composite *over* its base (§14.1), so a
  filter that has had a layer dropped onto it has nothing under it inside its own
  group. The layers it carries are above it, not below.

In both, the filter is the identity, and the draw list leaves it out rather than
encoding a pass that provably cannot change a texel (§21.3). The panel says so:
`LayerInfo::has_lower_sibling` is the projected predicate, and it is deliberately
**not** `has_backdrop`. The two differ on precisely the group-base row — which has a
backdrop (what lies under the group) but no lower sibling — and that is the one row
where a blend mode and a clip are live while a filter is not, because those point
outward (§14.4.3) and a filter points at its own stack.

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
- **The same scratch.** A filter needs a level's `swap` and does not use its `iso`.
  That is one viewport pair of waste in a document whose *only* non-`Normal` thing is
  a filter, and it buys one allocation path rather than two.
- **No effect on a document without one.** Every existing golden is unchanged, which
  is the evidence: adding a variant to `GroupContent` changed no draw a paint-only
  document issues.

`FilterDraw` is deliberately not a `Filter`: which `u32` a filter kind is numbered is
a fact about `filter_common.wesl`, the split `blend_code` already makes for a blend
mode. Flattening the layer's opacity into it as a *strength* at the same time is what
leaves the encoder one thing to write rather than two to remember to combine.

Two things are dropped from the draw list rather than drawn, and both are exactness
requirements rather than optimizations:

- **A filter with nothing beneath it** (§21.2) — it would read an empty accumulator
  and write it back, which is the identity in principle and a rounding trip in
  practice.
- **A filter at its neutral setting**, which is what a freshly added one holds. That
  is what makes *adding* a filter a step you take before deciding what it does, rather
  than an edit in itself, and `a_neutral_filter_changes_no_pixel` pins it to the byte.

#### 21.3.1 What the pass does *not* touch

Pass A's colour target holds the stack premultiplied by coverage; its aux holds the
amount of paint (§6.1). A filter has an opinion about neither:

- **Coverage comes out as it went in.** The adjustment runs on the *un*-premultiplied
  channels, so a half-covered texel is adjusted like a full one rather than like a
  darker one, and the alpha is written back unchanged. A filter says what colour the
  paint already there should be, not how much of it there is.
- **Height is copied across verbatim.** Relief is a property of the medium, not of the
  colour. It is a real copy rather than a skipped attachment: the ping-pong means the
  pass's output targets are not the ones it read, so an aux left unwritten would hold
  whatever the previous bounce left there.

### 21.4 What a filter borrows from a layer, and what it cannot

| | on a filter | why |
|---|---|---|
| `opacity` | the filter's **strength** | a mix from the untouched backdrop to the filtered result — which is what fading a layer already means |
| `visible` | on/off | as everywhere |
| `name`, position, removal, duplication | as everywhere | it is a layer |
| `blend` | **inert** | a mode describes how a *source* meets a backdrop; a filter has no source, it *is* the backdrop |
| `clip` | **inert** | same reason, and there is nothing to clip: a filter already writes only where the backdrop is |
| paint | **refused** | no tile map — the same refusal a matte gives (§15.7), in `apply` and in the preview path alike, so replay and peers agree |

Strength is mixed in the **working space** rather than in light, which is what makes
strength 0 the *exact* identity rather than the identity plus a round trip's rounding.

The panel shows blend and clip inert on a filter row for the reason §14.4.3 shows them
inert on the bottom row: a control that cannot express anything here should say so
rather than accept a value nothing reads. The opacity slider is relabelled
**Strength** on a filter row — "50% opacity" on a colour adjustment invites the reading
that the filter is half transparent, when what it is is half applied.

### 21.5 The colour filter

The first filter, and the one the architecture was built against. Four numbers, all
applied in **Oklab**:

```rust
pub struct ColorAdjust {
    pub exposure: f32,    // stops; light is scaled by 2^exposure
    pub contrast: f32,    // gain on Oklab L about mid-grey; 1 is the identity
    pub saturation: f32,  // gain on Oklab chroma; 1 the identity, 0 a true greyscale
    pub hue: f32,         // rotation of the Oklab (a, b) plane, radians
}
```

**Why Oklab, and why it is four and not seven.** In a perceptual space lightness,
chroma and hue are separable, and that separability is the whole promise a colour
slider makes. Saturation in sRGB shifts hue. Contrast in sRGB shifts saturation. A
"greyscale" that is a luminance-weighted RGB average moves lightness around — visibly,
on a saturated red, by about 0.1 in `L`. Here a chroma gain is a chroma gain, so four
controls cover what a levels dialog needs seven to approximate.

**Exposure is the exception, and it is one on purpose.** It is applied to *light* — the
same normalized XYZ the blend modes combine in (§18.0.4) — before the trip into Oklab.
Doubling light is what an exposure *is*; `L` is roughly the cube root of that, so
scaling `L` by `2^n` would be a number with no referent. It is also the reason the
pass is bracketed per colour space exactly as the blend pass is: `filter_oklab.wesl`
and `filter_mixbox.wesl` supply only channels ↔ light, and `filter_common.wesl` holds
the adjustment.

**Contrast pivots on mid-grey, not on the picture's own mean.** A pivot that depends on
what is underneath would make the slider do something different every time a layer
below it changed, which is not what a contrast control is. The constant is
`document::CONTRAST_PIVOT`, mirrored in the shader and derived rather than trusted by
a unit test.

**One honest consequence in a pigment document.** Pigment cannot be brighter than the
light falling on it and Mixbox's inverse LUT is defined on `[0,1]` sRGB, so a positive
exposure saturates at white there instead of pushing past it into the media pass's
highlight roll-off the way it does in an Oklab document. That is the same thing
`blend_mixbox.wesl` says about `Radiance`, and for the same reason: paint does not glow.

**Every parameter is bounded and sanitized on the way into the log.** A fullscreen pass
has no coverage to hide behind — a `NaN` saturation from a file or a peer reaches every
texel of the frame, and nothing downstream can notice. `Filter::sanitized` clamps to
the documented ranges and replaces a non-finite value with the *neutral* setting for
that knob, because `NaN` says nothing about which end was meant and the identity is the
one answer that cannot make a picture worse. It runs where the action is minted, so
replay puts back what was applied rather than re-deriving it (the funnel `SetLayerName`
already goes through).

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
  An unnamed filter row shows the *filter's* name ("Colour") rather than the word
  "Filter", because a stack of three rows all reading "Filter" would say nothing.
- **A filter that reaches nothing says so** (§21.2), once, in the bar — rather than
  greying out four sliders that would each have to explain the same thing.

**The sliders preview live and log once.** Each pointer move sends
`ViewCommand::PreviewFilter` (view state, never logged) and the settled drag commits a
single `DocCommand::SetFilter`. This is the bargain the frame drag (§15.7), the canvas
colour (§15.5) and the opacity slider (§14.6) already make, and it is the one this
feature could least do without: a colour adjustment is judged *by looking* — how much
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
result back — the read-modify-commit shape `ViewCommand::SetGuides` uses (§20.5). So a
filter that grows a parameter, and the next *kind* of filter, need no new command, no
new action, and no wire-format break.

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
- **Shaders.** `filter_common.wesl` plus a variant per colour space, mirroring
  `blend_common` / `blend_oklab` / `blend_mixbox` exactly. `linear_to_light` and
  `light_to_linear` moved out of `blend_common.wesl` into `lib/color.wesl` on the way:
  two passes work in light now, and importing the pair out of `blend_common` would have
  dragged that file's bindings along with it.
- **The uniform.** `Filter` in `filter_common.wesl`, mirrored into Rust by the
  generator (§6.10). Its parameters are one `vec4` lane read according to `kind`, so
  the next filter reads those four floats as its own and neither the host struct nor
  the bind group layout learns about it.
- **Bind group.** The blend pass's numbering with the source's two slots simply not
  declared: `filter_common` owns 0–2 where `blend_common` owns 0–4, and the pigment LUT
  keeps 5–6 because `mixbox_lut.wesl` hard-codes them for whoever imports it. The LUT
  itself is the blend pass's, decoded once — both passes ask it the same question.

### 21.8 Invariants worth a test (`tests/filter.rs`)

Three of the eight are about a filter doing **nothing**, and that is deliberate: a pass
that runs over every texel has no coverage to hide behind, so the cases where it must
be the exact identity are the ones where a mistake is a whole-picture change with
nothing on screen to say where it came from.

1. A filter recolours what is composited beneath it.
2. **Desaturating keeps the lightness it found** — measured as Oklab `L`, which is the
   test rather than a detail of it: a saturated red and its correct grey have very
   different *luminance*, so an assertion on luminance would fail on the right answer
   and pass on `dot(rgb, luma)`.
3. A **neutral** filter changes no pixel, to the byte.
4. A **hidden** filter, and one at zero **strength**, each change no pixel.
5. A **carried** filter reaches only its own group — the layer above the group is
   untouched. This is §21.1's whole claim, drawn.
6. A filter with **nothing beneath it in its own stack** changes no pixel, in both
   forms: at the foot of the document, and as a group base.
7. A filter can be **selected** but takes no paint, and the refusal is the engine's.
8. A filter **undoes** — the adjustment, and the add behind it.
9. A slider drag **previews without logging**, and the commit renders what the preview
   showed.
10. A filter **survives save and load** — pixel-identical and setting-identical.
    `AddFilter` and `SetFilter` are the first actions to carry a `Filter`, and
    postcard writes no field names and no lengths, so a layout mistake decodes into a
    *different adjustment* rather than into an error (§8).
11. A filter works in a **pigment** document — the road out through Mixbox's
    polynomial and back through its inverse LUT, with the latent residual carried on
    both legs (§6.7). Nothing in an Oklab test touches that half.

### 21.9 Open

- **The rest of the filters.** Motion blur, chromatic aberration, outline, blur, glow.
  Each is a `Filter` variant and an arm in `filtered()`; the ones that read *neighbouring*
  texels are the first to need something this pass does not have, since the accumulator
  it samples is the supersampled render and a kernel in screen px is not a kernel in
  canvas px (§6.4). That is the same question the media pass's relief already answers by
  zoom-normalizing, and it is where the next filter's design starts.
- **Per-filter masking.** Not needed for scope — position is scope (§21.1) — but a
  *soft* boundary (grade the sky and not the ground) has no expression yet. The selection
  is the obvious source, and §15.9's P4 region algebra is the obvious representation.
- **Filters on export.** They composite in pass A, so an export already carries them;
  nothing to do, recorded because it is the question everyone asks.

---
