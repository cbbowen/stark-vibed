# Layers, groups, frames, and export

One mechanism for groups and clipping; the matte layer that answers "what rectangle is the piece?" — §14, §15.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

The third kind of layer content — the **filter**, which is a function of what is
composited beneath it — is [filters.md](filters.md), §21. It leans on §14 rather than
extending it: where a filter sits in the tree *is* what it acts on, so the group
mechanism below is the whole of its scope control.

## 14. Groups and clipping

Two features every drawing app has, and neither is well modelled anywhere:

- **Layer groups.** Rebelle's are purely organizational — a folder that cannot
  change the picture. Photoshop's are functional, so they need a blend mode; but
  then a group that merely tidies the stack would change the render, so it also
  needs a fake **pass-through** mode; and once a group has a blend mode of its
  own, that mode and the bottom member's are two controls answering one question.
- **Clipping masks.** A toggle making a layer transparent where the *next
  unclipped layer below* is transparent. New users do not guess this, the arrow
  in the panel points at one layer while the behaviour involves a run of them,
  and in Rebelle and CSP a clipping chain quietly becomes a group — so there are
  two grouping mechanisms that look nothing alike.

### 14.1 The stance: one sentence

> A layer may **carry** other layers. A layer's blend mode, clipping and opacity
> describe how that layer *together with everything it carries* meets what lies
> beneath it.

Compositing is the same sentence read as an algorithm:

```
composite(layer, backdrop):
    stack ← layer's own content, alone on nothing
    for each carried layer c, bottom to top:
        stack ← composite(c, stack)         # c meets what is beneath it here
    stack ← stack scaled by layer.opacity
    return merge(backdrop, stack, layer.blend, layer.clip)
```

That is groups, clipping masks, blend modes and layer opacity — all four — with
no group object, no pass-through, no clipping chain, and no rule that applies at
only one level.

### 14.2 The representation: a layer carries layers

There is no `Group` type. A group **is** the layer at its base — see `Layer` in
§5.1, whose `carries: Vector<Layer>` is the whole mechanism. Four properties fall
out rather than being designed:

- **No group is empty**, structurally. There is nothing to be empty.
- **A group of one is the layer itself** — not "equivalent to", *is*. Wrapping a
  single layer is the identity, exactly as parenthesising a single term is.
- **The base's blend mode and the group's are one field.** Not an alias with a
  synchronisation rule; one field, one meaning.
- **The base cannot be swapped by a drag.** In a container model, dragging a
  layer to the bottom of a folder silently rewrites the group's outward blend
  mode. Here "which layer is the base" is *which layer carries the others*, so
  changing it is an explicit restructure, not a side effect of a reorder.

The document itself is a stack of top-level layers, unchanged: `DocState.layers`
stays a `Vector<Layer>`, and the tree lives inside it.

### 14.3 Why the base's blend mode is free

The objection to Photoshop's design is duplication. Taking the base's mode *as*
the group's is only an improvement if the base's mode was doing nothing — and in
this engine it provably is not, with the proof already in the tree. `merge()`
against an empty backdrop reduces with `cb = 0` to

```
out.rgb = mix(cs.rgb, blended · cs.a, 0.0) + 0.0 = cs.rgb
out.a   = cs.a + 0.0 · (1 − cs.a)              = cs.a
```

— bit-for-bit the `Normal` result, which is deliberate and which `tests/blend.rs`
already asserts to the byte. The substrate does not rescue it either: the ground
is composited in pass B, after all blending, so the bottom of a stack genuinely
has nothing underneath. So the bottom layer of any stack carries a blend-mode
slot that **cannot express anything**. We are not overloading a control; we are
filling a hole. The same argument runs for clipping, and §14.4.3 spends that slot
the same way.

That is also the rule for *which* properties the group takes from its base:

| | belongs to | why |
|---|---|---|
| `blend`, `clip`, `opacity` | **`CompositeParams`** — the group's, taken from the base | all three are stated against the backdrop, so all three are vacuous at the base and free to describe the group's own merge outward |
| `visible`, `name` | **intrinsic** — the group's own, as the base's own | they describe the layer itself and mean the same thing whatever is under it |

Opacity sits in the first row, and it took a bug to establish that. It reads as
intrinsic — *how faded is this layer* — but it is applied at the same step as the
other two and to the same thing: the group's composited whole (§14.7). Carried
separately, it was applied to the base's own content as well; grouped, the base
composites with `CompositeParams::IDENTITY` and there is one place to get it
right.

Group opacity is therefore *not* the duplication Photoshop's group blend mode
was: fading a group fades the base and everything on it as one unit. The one
thing this model cannot express is fading or hiding the base *alone* while what
it carries stays at full strength. That is a real loss, the price of having no
container object, and an operation with no use anyone has named.

### 14.4 Clipping (the clipping mask, restated)

`clip` is a per-layer boolean applied at the same step as the blend mode. It
means: **this layer exists only where there is paint beneath it in its group.**

Two ways it differs from every clipping mask in the field, both simplifications:

- It inherits the alpha of the **whole composited stack below it within its
  group**, not of the nearest unclipped layer. There is no chain, nothing to
  trace up the panel.
- **The group is what bounds "below".** Clipping to exactly one layer is not a
  special mode; it is that layer carrying the clipped one. One mechanism does
  both jobs, and what users actually mean by "clip to the layer below" is a
  single drag.

It keeps the field's name anyway. This is not the clipping mask other apps ship,
but it is the nearest analog by a wide margin, and a painter arriving with the
concept reaches for the right control on the first try — worth more than a name
accurate only to a reader who does not have the concept yet.

A **filter layer** takes the same flag, and the sentence above needs exactly one
substitution to fit it: a filter has no content of its own to *exist*, so what is
bounded is its **result**. Clipped, it may say what color the paint already there
should be and never where there is paint — which is the identity for a filter that
was already a function of one texel, and the live case for one that displaces
(§21.4.1). Its blend mode, by contrast, stays refused: that one describes a source
meeting a backdrop, and a filter has no source at all.

#### 14.4.1 The formula, and why the obvious one is wrong

The natural phrasing — *multiply by the opacity of what it is compositing onto* —
is wrong if implemented as `αs ← αs · αb`. With `αb = 0.5, αs = 1` that yields
output alpha `0.5 + 0.5·(1−0.5) = 0.75`: the clipped layer **invented coverage**
the backdrop did not have, and the backdrop shows through paint that should be
opaque.

The correct operation is not a scale, it is a **deletion**: drop the term for
source that lands where there is no backdrop. Writing the merge with that term
visible (`mix(x, y, αb) = (1−αb)·x + αb·y`):

```
unclipped:  rgb = (1−αb)·αs·Cs  +  αb·αs·B(Cb,Cs)  +  cb.rgb·(1−αs)
            a   =    αs·(1−αb)  +  αb·αs           +  αb·(1−αs)   =  αs + αb(1−αs)

clipped:    rgb =        0      +  αb·αs·B(Cb,Cs)  +  cb.rgb·(1−αs)
            a   =        0      +  αb·αs           +  αb·(1−αs)   =  αb
```

Output alpha collapses to exactly `αb` — the group's coverage is untouched by
anything clipped to it, the property that makes clipping composable and which the
scaled-alpha version does not have. The tail keeps the **unmodified** `αs`:
inside the backdrop's region the source still covers `αs` of it.

In the shader this is one factor. With `m = clip ? 0.0 : 1.0`:

```wgsl
out.color = vec4(mix(cs.rgb * m, blended * cs.a, cb.a) + cb.rgb * (1.0 - cs.a),
                 cs.a * (cb.a + m * (1.0 - cb.a)) + cb.a * (1.0 - cs.a));
```

which is `merge()`'s line with `* m` and one factor added, and which is
bit-identical to the unclipped output at `m = 1`.

#### 14.4.2 Clipping must scale the aux, or you get ghost impasto

**Forced by the media pass.** `merge()` sums the height field unconditionally, on
the grounds that height is *amount of paint* and paint stacks whatever its color
does. Clipping breaks the grounds: a clipped layer's color is suppressed outside
the backdrop, and if its height is not, the media pass lights relief where there
is no paint. So:

```
out.aux = hb + hs · (clip ? αb : 1.0)
```

Every ridge a clipped stroke lays outside its group's paint has to go with it.

#### 14.4.3 Clipping the base clips the group

The base's `clip` points **outward**, exactly as its blend mode does: it clips
the composited group to what lies beneath the *group*. That is not a second rule
— it is §14.1's sentence unchanged, and it is why clipping a whole group needs no
mechanism of its own. In the compositor it needs no branch either: the recursion
merges each subtree into its parent's backdrop through that subtree's own blend
and clip, and the base's fields *are* the subtree's.

So `clip` is live wherever there is a backdrop to clip to, at any depth:

```
has_backdrop(L) = L has a sibling below it in its stack
               ∨ (L is carried by C ∧ has_backdrop(C))
```

This is the same predicate that decides whether a **blend mode** does anything,
which is the point: the two relational properties go live and inert together, so
there is one fact to teach rather than two.

They part on one row, and it is the row with no source: on a **filter** the mode is
refused outright while the clip is live, and the clip is inert there by the
renderer's predicate (`has_underlay`, §21.2) rather than by this positional one —
a filter carried onto a painted layer is reaching that paint even with no sibling
below it. Same fact, asked of the thing each control is actually defined against.

It fails in exactly one place — the bottom-most layer of the root stack, which
has nothing under it anywhere (§14.3). There the two properties degrade
differently, and that asymmetry is the only reason the UI has to care: a blend
mode over an empty backdrop is the **identity**, so leaving it set is harmless,
while a clip over an empty backdrop is **annihilation** — the layer disappears.
Both controls are therefore shown inert on that one row.

### 14.5 What grouping costs, and why it is safe here

Groups are always isolated. A `Multiply` layer *inside* a group multiplies
against the group, not against what lies under it — so wrapping layers can change
the render. That is exactly what pass-through was invented to prevent, and
declining to invent it is the one place this model is worse than Photoshop's. It
is worth it because of what it is bought with, and because here the cost is much
smaller than it would be elsewhere.

**Pure organization is free, structurally.** A group whose base is `Normal`,
unclipped and fully opaque, *and every member of which is `Normal` and
unclipped*, has nothing to isolate: §14.7 collapses it into the surrounding run
at build time, so it produces the identical draw list and identical pixels. Only
groups that change a blending scope cost anything.

**Where a scope does change, this blend family absorbs it.** Every mode past
`Normal` is addition conjugated by a tone curve (§6.3), hence commutative,
associative, with an identity:

- **Opaque layers under one mode:** grouping is *exactly* invariant, by
  associativity. Grouping three glow layers changes nothing at all.
- **`Multiply` at any coverage:** exactly invariant. The `mix(…, αb)` tail and
  multiply's white identity cancel, and both routes give `A·(1−s+sB)·(1−t+tC)`
  for a backdrop `A` under layers `B, C` at coverage `s, t`.
- **The emissive modes at partial coverage:** a small drift. For `A=0.4, B=0.6,
  C=0.8` at `s=t=0.5` under Glow: `0.6902` ungrouped, `0.6886` grouped.

In Photoshop, where the modes are ad-hoc formulae with no algebraic relationship,
grouping changes the picture arbitrarily — which is *why* it needs pass-through.
Here the modes were chosen so that regrouping is a non-event, and the feature
that would paper over the difference is not needed because the difference is not
there.

### 14.6 What the panel shows

The tree renders the way clipping masks already render everywhere: **the base at
the bottom, what it carries indented above it.** Photoshop already draws that
picture for a clipping group; this model says that picture is simply the truth,
and that it is a group.

```
  ▾ Skin                    Normal
        Blush               ⌐ Glow
        Shading             ⌐ Multiply
      Freckles              Normal
    ── Skin's own paint is this row ──
    Background              Normal
```

- The disclosure triangle sits on any layer that carries; expanding shows what it
  carries, indented, **above** it — which is where those layers render.
- The base row shows the blend chip and the clip toggle because both are *the
  group's*, drawn at the bottom edge of the group's bracket to say so. Clipping
  there clips the whole group (§14.4.3), so the rail it draws runs down past the
  group's own bracket rather than inside it.
- `⌐` is the clipping rail: a left-edge rule from the clipped row down to the
  bottom of the stack it inherits from — the full run, not one layer.
  Photoshop's arrow points at one layer and is lying; this points at everything
  it actually reads.
- On the bottom-most row of the document both controls are inert (§14.4.3), the
  one place the panel has to say "this does nothing here".
- **Indent means membership. Rail means clipping.** Different marks because they
  are different facts, and a user arriving from Photoshop — where indent means
  clipping — must see that at a glance. `Freckles` above is in the group and not
  clipped; that is a state Photoshop's panel cannot draw.

Two commands cover the whole feature: **Carry** (put the selected layers on the
one below) and **Release** (promote what a layer carries into its parent stack).
"Clip to the layer below" is Carry plus the clip toggle, and can be one menu item
that does both.

Everything a row can do to its own layer is drawn *in* the row — Carry at the head
of the line, Release out in the indent, then **Duplicate**, **Remove** and the eye
against the right edge — rather than in header buttons acting on "the selected
layer". A control drawn in the row has already named what it acts on, so it needs
no inapplicable state: Release is simply absent on a layer that is in no group,
Remove on the row whose removal would empty the document, and Duplicate on neither,
because every layer can be copied.

#### Moving a layer by dragging it

The panel draws where every layer sits, so the way to put one somewhere else is to
drag it there. **The whole row is the grip** (its name — the thing you would reach
for), and one gesture spells all three of the moves above, because in the model
they are one move: a drop lands in *some stack*, at *some place in it*, which is
exactly `MoveLayer`'s two anchors (§14.8). Reordering within a stack, which Carry
and Release could not express at all, falls out of the same gesture rather than
needing a fourth control.

Two things make the gesture say what it will do before it does it:

- **The drop target is a seam between rows, not a row.** The rows themselves open
  the slot the layer is going into and the layer floats over that slot at the
  indent it will land at, so the preview *is* the answer rather than a symbol for
  one. A group travels as one block, which is the same fact §14.2 states about
  removal and duplication, drawn.
- **Sideways is what nests.** One seam can be the end of several different stacks
  — all of which draw at that one place — and how far right you are holding the
  layer is which of them you mean. Where a seam can only mean one thing, which is
  most of them, horizontal travel does nothing at all. The layer that would carry
  the drop is ringed while you hold it there, because that is the one part of the
  landing an indent alone leaves to be inferred.

Two invariants fall out of stating a drop against the rows that *stay put*, rather
than being checked at the drop:

- **A layer can never be dropped inside itself.** Every place a drop can name is
  named after a row that is not travelling. The engine declines a cycle anyway
  (§14.8), but a panel that offers a move the engine will silently refuse is a
  panel that lies.
- **Every place the panel can draw, it can drop into.** Each depth a seam admits
  names exactly one real position, because the ancestors of the row below it cover
  every depth beneath it without a gap. The one place this did *not* hold was the
  foot of a stack, which no sibling can be named against — hence `Place::Bottom`
  (§14.8).

**A gesture lasts exactly as long as the press it is made of.** The grip is also a
thing the pointer merely passes over, so the handler that steers a drag hears every
hover too — which makes "the press is still down" a fact the gesture has to carry
rather than one the panel can assume. It is carried two ways, both in `Grab`: a move
with no button held ends the drag instead of steering it, and a grab that has landed
is *terminal* — it lingers a moment for the click behind the release
(`reorder::claimed`) and cannot be woken by anything in that moment. Neither is
belt-and-braces: a release the panel never hears about, or a click that never arrives
because the drop's own reorder replaced the element it was aimed at, would otherwise
leave a row following the pointer around a panel with no button down, over a list it
had already been moved in.

The base of a group is still not swappable by a drag (§14.2): dropping a layer at
the foot of a group's carried stack puts it under everything the group carries,
above the base. Becoming the base is a different move — it is that layer carrying
the others — and it stays an explicit restructure rather than a side effect of a
reorder.

#### The opacity slider previews live and logs once

Most controls in the panel report a value the hand *chose* — a blend mode, a clip,
a name — and one of those per interaction. The opacity slider reports one
per pointer **move**, and so it makes the same bargain the frame drag (§15.7) and
the canvas color (§15.5) make, on the same slot: each sample sends
`ViewCommand::PreviewLayerOpacity` (view state, never logged), and the settled
drag commits a single `DocCommand::SetLayerOpacity`. A drag costs one undo step
rather than a hundred, and in a shared session one replicated action rather than
a hundred. `observe()` reports the *previewed* opacity, which keeps the track and
the canvas agreeing with each other under the pointer.

Two details are the difference between that working and nearly working:

- **The commit supersedes the preview, and a settled drag always commits.** A
  preview left standing pins the document to the last dragged value and shadows
  every later edit, so the release has to reach the engine even when the value
  did not change. The frontend cannot lean on the browser's `change` event alone
  for that, because a drag ending on the value it started from does not send one.
- **A commit to the value the layer already holds is refused** (`Engine::process`,
  as for `SetLayerName`) — that is what stops the out-and-back drag the previous
  point forces into the engine from spending a step that appears to do nothing
  when undo reaches it.

Both are asserted by `dragging_layer_opacity_previews_without_logging` and
`an_opacity_drag_that_ends_where_it_started_logs_nothing` (`tests/layers.rs`).

The **Bend** slider beside the blend picker is the second control on this bargain,
and it is here rather than in a section of its own because nothing about it is new:
`ViewCommand::PreviewLayerBlend` per sample, one `DocCommand::SetLayerBlend` on
release, both details above holding for the same reasons. What it does differ in is
that it carries the whole `BlendMode` rather than the number — the parameter belongs
to the mode (§6.3), so the mode is what both halves of the bargain already take, and
a mode that grows a second knob needs no third command. The row exists only while
the layer's mode has a parameter, which is the panel reading the model rather than a
rule of its own.

### 14.7 Compositing

`CompositeGroup` is a tree, and the existing fast path is an invariant of its
shape rather than a special case inside a loop:

```rust
/// How something meets what lies beneath it. One value, in the document and in
/// the draw list alike, because every rule about them is a rule about all three.
pub struct CompositeParams {
    pub blend: BlendMode,
    pub clip: bool,
    /// Applied to the composited whole at merge — members overlap, so it cannot be
    /// folded into per-tile opacity the way a leaf layer's can.
    pub opacity: f32,
}

pub struct CompositeGroup {
    pub params: CompositeParams,
    pub content: GroupContent,
}

pub enum GroupContent {
    /// A run that composites under plain premultiplied "over" — no isolation.
    Run(Vec<CompositeItem>),
    /// Isolated members, bottom-to-top, each merging into the one below.
    Stack(Vec<CompositeGroup>),
}
```

The three are **one value** rather than three fields, and that is load-bearing
rather than tidy. They are stated against a backdrop, so they are vacuous
together where there is none; they decide the fast path together, since a layer
needs isolating if *any* of them does something; and — the reason they were
grouped — they belong to the group **as a whole and never to its base**. A group's
members composite over its base (§14.1), so the base's own content is a member: it
composites with `CompositeParams::IDENTITY` and the layer's params are applied
once, to the result. Held apart, that was three chances to get one rule wrong, and
one of them was taken (below).

Build-time rules, in `Engine::composite_groups`:

1. A `Stack` all of whose members need no isolation, and which is itself
   `Normal`, unclipped and opaque, **collapses into a `Run`**. This is §14.5's
   "organization is free", enforced structurally rather than promised.
2. Anything that needs no isolation joins the enclosing `Run` — which, because
   rule 1 runs first and bottom-up, includes a *group* that collapsed. Tidying
   layers into a folder therefore costs nothing at all, not even a group boundary
   the encoder has to step over.
3. A document with no groups, no modes and no clipping produces exactly one
   `Run` — today's draw list, unchanged, at today's cost.

Encoding recurses. Each nesting level in use needs its own ping-pong pair plus
the pair its members isolate into — about 40 MB at 1080p per level, on the order
of what blend modes already allocate, and allocated lazily to the deepest level
the document actually reaches. The parity trick that lands the final result in
the caller's own targets (§6.3) still applies, per level.

`BlendUniform` gains two fields, `clip` and `opacity`, filling the 16 bytes it
was already padded to. `Normal` now reaches the pass — a clipped normal layer, or
a group faded as a unit — where before it never did, so `merge` grew a `Normal`
branch taking the premultiplied source verbatim rather than the mode's opinion.
That branch is not a shortcut but an exactness requirement: a clipped normal
layer over solid paint has to match the fixed-function `over` an unclipped one
gets, and dividing by `αs` to feed the blend function and multiplying it back
would not. **An unclipped `Normal` layer at full opacity is still the absence of
a pass.**

Two consequences to know about:

- **A leaf layer's opacity stays per-tile.** Tiles within a layer do not overlap,
  so scaling each tile is identical to scaling the composited layer — but members
  of a group *do* overlap, so a group's opacity must be applied at the merge.
  That asymmetry is not a wart; it is the same fact at two granularities.

  **A group's base is a member**, and the two granularities are therefore
  exclusive rather than cumulative: a base whose items carry the slider *and*
  whose group applies it at the merge is a base faded twice. That is exactly what
  happened for as long as this feature existed — a group base at 0.5 drew its own
  paint at 0.25 while everything it carried drew at 0.5 — because the item builder
  tagged with the layer's opacity, which is right for a leaf and wrong the moment
  the same items go into a `Stack`. The slider still faded, the two granularities
  still differed, and the only visible symptom was a base fading faster than the
  layers standing on it.

  The fix was one line; the *repair* was making the shape unable to say it.
  `CompositeGroup::leaf` is now the only constructor that folds an opacity into an
  item, and it takes that opacity off the group in the same expression, so folded
  and applied-at-a-merge are exclusive by construction. The item builder no longer
  knows what a layer's opacity is. And `composite_stack` states the whole rule
  twice, in two lines that read as the sentence:

  ```rust
  let group = if carried.is_empty() {
      CompositeGroup::leaf(layer.composite, own)                      // a leaf is the whole of it
  } else {
      members.push(CompositeGroup::leaf(CompositeParams::IDENTITY, own)); // the base is a member
      CompositeGroup::stack(layer.composite, members)                 // …the params are the group's
  };
  ```

  Two tests hold it. `tests/groups.rs::a_groups_opacity_fades_its_base_exactly_once`
  pins the pixels by the one case where the two granularities **coincide**: with
  non-overlapping members, a group at opacity `a` must render exactly as the same
  layers ungrouped at `a`, which a base at `a²` breaks while everything it carries
  stays right. `CompositeGroup`'s own unit tests pin the constructor contract with
  no GPU at all — including that the collapse in `stack` uses `run` and not `leaf`,
  since re-folding there would flatten two tidied-away faded layers back to full
  strength.
- **An opaque group does not erase the relief beneath it.** `merge()` sums the
  aux, so impasto under an opaque group embosses through it, exactly as it
  already does under an opaque non-`Normal` layer and unlike an opaque matte
  (§15.4.2). Groups make the existing wart easier to hit. The fix — `hb·(1−αs) + hs`
  for `Normal`, so paint hides the relief it covers — is a change to how *today's*
  blend layers render and is deliberately not bundled in.

### 14.8 Plumbing

Less new machinery than the feature suggests, because the existing structural
resource is already coarse.

- **Actions.** `AddLayer`, `AddMatte` and `MoveLayer` each grow a
  `carrier: Option<LayerId>` — "whose stack", beside the existing "above which
  sibling". One new kind, `SetLayerClip(LayerId, bool)`. Carry and Release get
  **no actions of their own**: carrying *is* a move to a position inside another
  layer, so `MoveLayer` covers reorder, carry and release by which of its two
  anchors changes. One structural action, one inverse.
- **The second anchor is three-state**, not two. A stack of `n` layers has `n + 1`
  places to land in and only `n` siblings to name them after, so "above this
  sibling, or on top" leaves exactly one place unsayable: the foot of the stack.
  `MoveLayer`'s `above: Option<LayerId>` is therefore a `Place { Top,
  Above(LayerId), Bottom }`. That mattered the moment the panel could drop a layer
  anywhere it could draw one (§14.6) — "put this behind everything" is where a
  background goes, not an exotic move. It cost no format break: postcard writes an
  `Option` as a `0`/`1` discriminant and an enum as its variant index, so `Top` and
  `Above` keep the encoding of the `None` and `Some` they replaced and `Bottom` is
  an appended third variant — the one shape of change §8 allows. A unit test in
  `document/layer.rs` asserts the three byte strings rather than leaving that to
  reasoning, because reordering the variants would silently reinterpret every
  `MoveLayer` in every saved document.
- **Duplicate.** `DuplicateLayer { ids: Vec<(LayerId, LayerId)> }`: the copy lands
  in the source's own stack directly above it, and the subtree travels as one, for
  the reason removing it does. The action pairs every layer of that subtree with
  the id its copy takes — the author mints them, as for `AddLayer`, so a replay
  mints what the recording run minted. Naming the *sources* as well is what lets
  the footprint be honest: a copy is a function of every tile and every property of
  every layer it copies, so it does not commute with a stroke or a rename inside
  the group, and an action naming only the root could not say so. Tiles come along
  as the shared handles they already are, so a duplicate costs no GPU memory until
  one of the two is painted on. The copy keeps the source's **name** verbatim: a
  name in the document is the author's own word, and "Sky copy" would be the engine
  writing one nobody typed.
- **The file format.** A field in the middle of an existing struct variant is not
  something postcard can absorb (§8), so this was the first change here that
  could not be *appended*, and `WIRE_VERSION` went to 2. The alternative — a
  second `MoveLayer` variant preserving the old layout — would have put the
  duplication this design exists to remove straight back into the log.
- **Footprints and patches** — see §12.6; `Resource::StackOrder` and
  `PatchOp::Structure` were shaped for exactly this.
- **Peers.** Concurrent moves that would make a layer carry its own ancestor are
  the only new failure mode. Because the log is totally ordered by
  `(lamport, actor)` and applied sequentially, the check is local and
  deterministic: a move whose target is a descendant of the moved layer applies
  as a no-op. No tree-CRDT cycle machinery is needed — the total order already
  supplies what one would.
- **Eyedropper.** `composite_groups(doc, Some(id))` means *that layer's own
  content*: no carried layers, unclipped, unblended. Sampling a clipped layer
  should show the paint that is there, not the paint that survives.

### 14.9 Invariants worth a golden test (`tests/groups.rs`)

1. A `Normal`, unclipped, opaque group of `Normal` unclipped layers is
   **bit-identical** to the same layers ungrouped.
2. Carrying a layer and releasing it again is bit-identical to never having done
   it — the round trip through the tree surgery, the bounds recomputation, and
   the collapse firing a second time on the way back.
3. `clip` over a solid backdrop leaves the layer alone. Within a couple of
   least-significant bits, not to the byte, and the gap is worth naming: paint
   coverage is `1 − exp(−K·α·h)`, which never actually reaches 1, so a clip over
   even the heaviest passage correctly removes the last fraction of a percent —
   and the clipped layer takes the blend pass where the unclipped one takes
   fixed-function `over`, which round differently at half precision. `merge` is
   still written to make the two coincide exactly where `αb` *is* 1, and the test
   bound is ~10× below what a wrong clip would produce.
4. `clip` over an empty backdrop renders nothing — and contributes no height.
5. A clip inherits the **whole stack below it in its group**: a clipped layer
   still shows over paint that only the group's *base* has, two layers down.
6. Clipping a group's **base** clips the whole group.
7. Grouping **does** rescope an interior blend mode. Asserted, not merely
   accepted: it is the one place this model is worse than pass-through, and the
   test is where anyone who decides to "fix" it has to come and read why.
8. A blend mode on a bottom-most layer with nothing beneath it is still the
   `Normal` render — the existing `tests/blend.rs` assertion, which this design
   depends on and must not break.
9. Duplicating a group copies the **whole subtree**, nested the same way, beside
   the group it copied — and hiding the copy is bit-identical to never having made
   it. Copy-on-write is asserted from the other side in `tests/layers.rs`: paint
   one of the two and the other must not change.

### 14.10 Open

- **Nesting depth.** Two target pairs per level is affordable at three or four
  and not at twenty. Either cap the depth or spill deep levels; nothing in the
  model needs a limit, so this is purely a budget decision.
- **Aux under an opaque group** (§14.7) — a pre-existing question this feature
  makes more visible, not one it creates.

### 14.11 Merging a layer down

Numbered after §14.10 rather than beside §14.4, where it belongs by subject,
because section numbers are cited from the source and are not renumbered.

#### The law

> **A merge must not change what the document looks like.**

That is the whole specification, and it is what separates a merge from any other
destructive edit. A painter merges to stop spending a layer on something that is
finished — not to accept a new picture in exchange. A merge that shifts a pixel
is a bug with no way to notice it: by the time the file is saved, the layers that
would have shown the difference are gone.

The law is not free. **A merge is not always possible**, so the operation returns
an `Option` and the panel offers the control only where there is one — absent
rather than greyed out, because a pair that does not composite as one layer is not
a weaker merge but a different edit (§14.11.4).

#### 14.11.1 Why it is exact, and not merely close

Write `h` for a texel's height, `op` for its per-unit opacity, and `m = op·h` for
its **optical mass**. Pass A gives a layer the weight

```text
    w = 1 − exp(−K·m)
```

— its visible alpha, the translucent-slab law (§6.1) — stacks the weighted colors
with premultiplied "over", and sums the heights. Stack two layers and

```text
    1 − w  =  (1 − w₀)(1 − w₁)  =  exp(−K·(m₀ + m₁))
```

so **masses add exactly as heights do**. The merged texel therefore has

```text
    H = h₀ + h₁          M = m₀ + m₁          opacity = M / H
```

and its opacity is the *height-weighted mean* of the two. Nothing is fitted and no
constant is tuned: a tile stores opacity and height as two numbers (§6.1), which is
exactly the freedom needed to name any (coverage, height) a stack can reach while
conserving height. The color follows through `blend_latent` — and that function is
not written for this, it is `paint_common.wesl`'s existing parcel-stacking law, the
one the brush deposits through and `fill.wesl` lays a fill with. A merge stacks two
layers the way a stroke stacks paint on paint, because those are the same act.

The **opacity slider** is the one thing that does not simply add. Pass A scales the
finished weight by it and the height by it (`w·opl`, `h·opl`), which is not the
weight of a slab of any opacity — so the merge inverts the law,
`M = −ln(1 − opl·(1 − e^{−K·m}))/K`, to recover the mass that does produce that
weight. It lands in range (`M ≤ h·opl` for every `opacity ≤ 1`), so the mean above
is still a per-unit opacity, and both sliders end up **inside the merged tiles**:
merging two half-faded layers gives one layer at full strength that looks the same.

#### 14.11.2 What has to hold for a pair

Write `B` for everything composited beneath the pair, `D` for the lower layer (the
**destination**, which survives, keeping its name and its place) and `S` for the upper
(the **source**, which is consumed). The document shows `merge_S(merge_D(B, D), S)` and
must go on showing it as `merge_D(B, D ⊕ S)`, for every `B`. Two independent questions:

- **Is `merge_S` associative?** Only then does `merge_S(merge_S(B,D),S)` equal
  `merge_S(B, merge_S(D,S))`, which is what lets `⊕` be `merge_S` itself. Every
  combining mode now is, at any coverage, because each weighs coverage in the space
  where its own blend function is affine (§18.0.4). While they weighed it in the
  working space, none of them was, and this whole family of merges was refused.
- **Is the backdrop `S` is stated against exactly `D`?** A clip reads the backdrop's
  coverage, so this decides whether "clipped to `D`" is even what `S` means — and it is
  what lets `S` carry a mode of its own into a carrier. It is `D` alone in exactly two
  places: `S` is the bottom of the stack its **carrier** `D` opens (a group's members
  composite over its base, §14.1), or `S` sits second from the bottom of the **root**
  stack, whose accumulator starts cleared.

Which gives the offered set — for a source made of **paint**; a filter source answers
only the second question and is §14.11.7. Both sides must be paint that carries nothing,
both must be equally visible (hiding a layer hides what it carries, §14.3, so a merge across a
difference would reveal or conceal paint), and:

| Where `S` sits | `S` may be | `D` may be |
|---|---|---|
| Bottom of its carrier's stack | **anything** — any mode, clipped or not | anything, opacity included |
| Second from the foot of the root stack | anything | any blend (inert with no backdrop, §14.4.3), unclipped, any opacity |
| Anywhere else, above a sibling | any mode, unclipped | the **same** mode, unclipped, any opacity |

Three rules stand behind that table, and each is one sentence:

- **Into a carrier, everything about `S` is absorbed.** The group's isolated content is
  `merge_S(base, S)` before the merge and the merged tile afterwards, so what the group
  merges outward is unchanged whatever `S` was — mode, clip and all.
- **The carrier keeps all three of its own params.** Its blend and clip point outward
  (§14.4.3) and its opacity applies to the group's composited whole (§14.7) — none of
  which the inside of that whole has anything to do with. So the base expands at
  **full strength** and the slider stays on the layer. Folding it into the tiles like a
  sibling's is what used to make a faded carrier unmergeable, and it would have faded
  the merged paint twice.
- **Between siblings the two must agree**, because afterwards one set of params speaks
  for both. Same mode, neither clipped. At the foot of the root stack they need not
  agree at all: nothing is stated against anything there.

  **Same mode means the same curve**, parameters included. Now that a mode can carry
  its own (§6.3), `Radiance` at one bend and `Radiance` at another are two functions
  and merge no better than `Glow` and `Multiply` do — the associativity this rests on
  belongs to a curve, not to a name. The check is `!=` on the whole `BlendMode`, so
  the model says it without a rule of its own; the frontend's `same_mode` is the
  deliberately weaker question, and only a picker asks it.

#### 14.11.3 Two laws, and which pairs take which

A `Normal` merge is settled **in tile space directly** (`merge.wesl`), because that is
the ordinary one and it needs no color conversion at all:

- **Unclipped** — the source stacks: heights add, masses add, opacity is their
  quotient (§14.11.1).
- **Clipped** — the source is *deleted outside its backdrop* (§14.4). Reading
  `blend_common.wesl::merge` at `MODE_NORMAL` with `clip = 1` against a backdrop that
  is exactly this tile gives

  ```text
      αo = αb                              coverage untouched, so M = m₀
      Co = αb·(αs·Cs + (1−αs)·Cb)          that coverage over a lerp
      ho = hb + hs·αb                      height suppressed with the color
  ```

  The height term is the half that matters: leaving it behind would light relief over
  paint that is not there (§14.4.2).

Everything else takes the general law, and the general law **borrows the compositor's
own blend pass**: expand both layers into what they composite to (`slab.wesl`), run
`blend_oklab`/`blend_mixbox` between them on tile-sized targets, store the result back
as a tile. Four passes and three scratch trios per tile where the direct path takes one
and none — the right trade for an action rather than a frame, and what it buys is that
**a merged tile is produced by the very shader the screen would have run**. No second
implementation of the blend algebra exists to drift from the first, which is the same
argument the eyedropper makes for sampling through the compositor rather than beside it
(§18.0.2).

`slab.wesl` is that conversion, both ways: `fs_expand` is `composite.wesl`'s fragment
stage with the view taken out, and `fs_store` inverts it — coverage back to an optical
mass, mass over height back to an opacity. The inverse is exact and lands in range by
construction; see §14.11.1.

#### 14.11.4 What is deliberately refused

**Modes that disagree, between siblings.** After the merge one set of params speaks for
both, and there is no third mode that means "glow here and multiply there". This is the
one refusal that is permanent rather than pending.

**Mattes, groups, and a filter as the destination.** A matte has no tile map, so it is
neither merged nor merged into — the same refusal a stroke aimed at one gets (§15.7) —
and a filter is never merged *into*, for that reason read the other way: there are no
channels there to rewrite. A group as the source would have to flatten a subtree; a
group as the *destination* is not what sits beneath the source, its whole group is. (A
**carrier** as the destination is not that case and is offered: the source is then the
bottom member of the stack that carrier opens, so what sits beneath it is the carrier's
own content.)

A filter as the *source* used to be refused here too, and is not: it is the second kind
of merge, §14.11.7.

Everything else that was once refused now merges. The history is worth keeping because
it was a wrong *reason* rather than a missing feature: same-mode siblings were called
unsound, and were, until the modes stopped weighing coverage in the working space
(§18.0.4); a source with a mode of its own into its carrier was always sound and only
ever blocked on the tile-space plumbing above; and a faded carrier was blocked by the
merge treating its slider as a sibling's.

#### 14.11.5 Plumbing

- **One action**, `MergeLayerDown { source, dest }`, appended last so postcard keeps
  decoding older files (§8). `dest` is derived rather than chosen — "down" names one
  layer — and travels anyway for the reason `DuplicateLayer`'s ids do: a `Footprint`
  is built from the action alone and cannot search the tree (§12.6).
- **The rule is a pure function of the state**, so the log carries no reasoning. The
  applying side re-derives the plan and **declines deterministically** if it now
  names a different destination, which is what a concurrent reorder or a blend mode
  set since looks like from here — so every peer and every replay declines together.
- **The footprint claims both layers whole**: their tiles, and the blend, clip,
  opacity, visibility and **filter** the plan reads. A merge that commuted with a
  blend-mode change on either layer would silently change its own answer — and one
  that commuted with a `SetFilter` would bake a grade the log says was replaced, with
  nothing in the pixels able to say which had run.
- **Cost is the overlap, not the document.** A tile only one side has passes through
  by handle, and within a shared tile the shader has an exact passthrough branch on
  each side — so merging a small stroke layer onto a canvas-spanning background
  rewrites the tiles the stroke covers and leaves the rest bit-identical, which
  `tests/merge.rs` asserts to the byte rather than to a tolerance.

#### 14.11.7 Merging a filter layer

Numbered after §14.11.6 for §14.11's own reason — the sections above are cited from the
source and are not renumbered — though it belongs by subject beside §14.11.2.

> A filter merge **rewrites** the destination's channels. It does not stack anything
> into them.

That is the whole difference, and everything below follows from it. A layer of paint
arrives with a coverage, a height and params that have to be reconciled with the
destination's; a filter arrives with none of those, because it never had any content —
it is a function of what it sits on (§21). So the merged tile is

```text
    channels ← filtered(channels)          opacity, height: untouched
```

and the destination keeps **every** one of its own params: its blend, its clip, its
opacity, its place. Nothing is folded into anything, because nothing arrived.

**Why the channels alone.** A tile stores its channels premultiplied by a *per-unit
opacity* while the accumulator carries them premultiplied by *coverage* (§6.1), and a
filter is defined on the un-premultiplied channels. For a layer composited alone those
are the same three numbers: pass A's slab law scales coverage and height and leaves the
color where it found it. So the filter's own numbers do not depend on the destination's
opacity slider at all, which is why there is no `dest_opacity` here and no slab
conversion — the merge runs the compositor's own filter shader on a second entry point
that reads a tile instead of the accumulator (`filter_oklab.wesl`).

**What has to hold** is §14.11.2's second question, and only that one. A filter rewrites
the accumulator beneath it, so baking it into `D` is the same picture exactly when that
accumulator *is* `D` — the same two positions any source has to sit in:

| Where the filter sits | Merges into |
|---|---|
| Bottom of its carrier's stack | the carrier's base — the "filter just this layer" gesture (§21.1), collapsed |
| Second from the foot of the root stack | the foot |

The first question — does the source reach the accumulator by plain "over"? — is
vacuous: nothing arrives, so there is no associativity to lean on and no mode to agree
about. The filter's own blend is refused by state (§21.4) and its clip is absorbed,
being the identity for every filter that may come this way (§21.4.1).

**One refusal of its own, and it is a law rather than a preference.** Every pass that
writes tiles must be a pure function of canvas position (§6.4) — that is what makes a
tile's apron bit-identical to its neighbour's interior without a copy pass, and
`tests/seam.rs` guards it. A point filter is such a function: the texel's own channels
in, its own channels out, so the apron texel and the neighbour's interior are the same
input and come out the same number. A **gather** (§21.10) is not, and no width of apron
would make it one — its reach is the document's to set, while the apron is one texel.
So `Filter::resamples` is asked, and a chromatic aberration is declined. It is refused
at the rule rather than approximated at the pass, which is the difference between a
merge that is not offered and a merge that leaves a seam at every tile boundary.

A **neutral** filter merges by doing nothing at all: the draw list already leaves one
out (§21.3), so it contributes nothing to the picture, and the tiles are handed across
by handle rather than run through a pass that would land the identity plus one round
trip's rounding.

#### 14.11.6 Invariants worth a test (`tests/merge.rs`)

Every test in that file has the same shape — render, merge, render, compare —
because a merge has no property of its own to assert: its whole content is
agreement with the compositor, and the compositor is what a render runs.

1. Two plain layers, two faded layers, a translucent glaze over opaque paint, a
   clipped layer, and a group's bottom member folded into its base all leave the
   composite unchanged within a least-significant bit — the tile arithmetic runs in
   f32 and lands in f16 storage, and no more than that is allowed. The pairs that
   take the general law — siblings sharing each mode, every mode into a carrier, a
   clipped member under a faded base — are held to three, the extra being the two
   further f16 round trips `slab.wesl` adds either side of the blend.
2. A merge whose layers do not overlap is exact **to the byte**, which is the
   structural half of the claim rather than a looser version of (1).
3. The upper layer is still on top afterwards — the one thing pixel equality alone
   would not have told you *which* bug had been avoided.
4. A merge the rule does not offer is a silent no-op that logs nothing, so the
   panel's rule and the engine's are one rule.
5. A **filter** merged into its carrier, and one merged into the foot of the root,
   leave the composite unchanged within a bit (§14.11.7) — while a **neutral** one is
   exact to the byte, which is the structural half: the tiles are handed across, so a
   merge that ran the identity through the pass anyway would fail it. A half-applied
   filter under a faded, `Multiply` carrier covers the rest of the sentence: the
   strength lands inside the tiles and the destination's own params do not move.
6. A filter that **resamples** is never offered one, and the same row with a point
   filter on it is — so the refusal is pinned to `Filter::resamples` rather than to
   anything about the arrangement.
5. Undo restores both layers — record, place, and the destination's own opacity,
   which the fold had set to 1 — by handle, so it is exact; redo reproduces the
   merge exactly.

---

## 15. Framing, mattes, and export

The infinite canvas never had to answer *what rectangle is the piece?* Export
forces the question, and the answer shapes composition as much as output: a
**matte layer** — a layer whose content is a region and a fill rather than a map
of tiles.

### 15.1 The stance: a frame is a suggestion, not a wall

A frame **clips nothing**. Paint runs past it, it slides around afterward, and
one painting may carry several. Photoshop's crop is destructive and Procreate's
canvas is fixed at creation; ours is a decision you get to defer, which is how
framing actually works at an easel. This is the whole reason the infinite canvas
earns its keep rather than merely being unusual.

Two load-bearing consequences:

- **Onboarding.** "New document → 1920×1080" seeds a *frame*, not a canvas. A
  Photoshop refugee gets the familiar bounded feeling; the boundary is soft.
- **Overpaint is a technique.** Painting past the edge and letting the matte
  cover it is how comic gutters and traditional inking work. The frame hiding
  your overshoot is a feature, not a compromise.

### 15.2 The representation: a region with a value at infinity

A frame is not "a rect plus a scrim". It is a **region and a fill**, where the
region has a defined value at infinity. That type already exists: `Selection` is
a coverage field over the infinite plane with an `outside` flag (§6.8). One field
then gives three features:

| Region | Position in stack | What it is |
|---|---|---|
| everywhere **except** a rect | top | the frame / mat board — **built** |
| everywhere **except** N panels | top | comic gutters |
| everywhere | bottom | an opaque ground / underpainting — **built** (`Everything`, §15.5) |

No `invert` flag and no separate scrim concept: `Invert` is already a
constant-cost operation on this representation.

```rust
pub enum MatteRegion {
    /// Everything outside this canvas-space rect — the frame.
    OutsideRect { min: Vec2, max: Vec2 },
    /// The whole plane — the ground (§15.5). No rect: it frames nothing,
    /// so export, the aspect readout and the handles all stand down
    /// (`MatteRegion::rect()` is an `Option`, and every consumer answers None).
    Everything,
}
```

`LayerContent` has a third variant beside `Paint` and `Matte` — `Filter`, §21 — and it
is worth reading the two together, because the matte is the argument this model is
built on: making a *thing that is not paint* a layer is what buys visibility,
opacity, ordering, naming, undo, save and collaboration for nothing (§15.3). A filter
makes the same trade for a different kind of not-paint.

The fill (in `LayerContent::Matte`, §5.1) is a **`MattePaint`**: one flat
straight-sRGB color like `BrushParams::color`, or a gradient — a §22.1
`Gradient` along a `GradientAxis`, the very pair the gradient fill lays
(§22.4), embedded by value the same way and edited through the same gradient
bar. Either converts to working-space channels at composite time, so the log
says the same thing whether the document is Oklab or Mixbox; a gradient's stops
convert once per frame build and interpolate per fragment in the working space,
so a Mixbox ground is a *pigment* wash. A matte has no alpha of its own in
either variant — its transparency *is* its layer opacity, which is the whole
point of it being a layer — and a gradient varies only the color, never the
slab's thickness: a graded wash is a transition in color, not in relief.

**The region is stored as geometry, not as a rasterized mask.** §6.8's selection
shader already evaluates shapes analytically from a signed distance at canvas
position, so a matte gets exactness at any zoom, zero tile budget (a 4000² frame
would otherwise cost ~16 MB of mask tiles and could trip `MAX_SELECTION_TILES`),
and a log entry of four floats. Rasterizing to tiles stays available later as a
pure caching optimization.

`MatteRegion` has exactly two variants today because those are the two built —
the table's first and third rows. It is still the seam where the `SelectionOp`
algebra lands (§15.9 P4), at which point gutters, lasso mattes and
frame-from-selection arrive together. Per this codebase's own precedent, no
variant appears here before it does something — `Everything` arrived with the
ground it draws, not ahead of it.

### 15.3 Why a layer, and what that buys

Because it is a layer, all of this is already built:

- **The scrim is layer opacity.** A 50% black matte on top is the classic crop
  scrim; drag to 100% for presentation. No `SetFrameScrim` command, no toggle.
- **Visibility, ordering, naming, delete** — the Layers panel already does it.
  "Which frame is active" is "which layer is selected"; no new concept.
- **Multiple frames** are multiple matte layers. Variant crops for free.
- **Undo, save, replay, collaboration** — a matte is document state reached by
  the existing layer actions, so §5 and §12 need no new argument.
- **Blend modes.** A Multiply matte is a vignette; a Glow matte is a light wash.
  Free expressiveness we did not have to design.

The alternative — a `frames: Vector<Frame>` field beside `layers` — needs its own
id space, actions, z-order rule, panel and active-item concept. Every one of
those is already solved for layers.

### 15.4 Compositing a matte (forced by the media pass)

#### 15.4.1 A matte must write the aux target

`media_common.wesl` derives a texel's visible alpha from the translucent-slab
law, `vis = 1 − exp(−OPACITY_K · color.a · (aux.x − surface_height))`. Visibility
comes from **per-unit opacity × thickness**, not from composited alpha. A matte
writing only color would be perfectly invisible. So a matte writes `color.a = 1`
and a thickness `MATTE_THICKNESS`, chosen so the slab reads solid: with
`OPACITY_K = 1.0`, a thickness of 8 gives `vis > 0.999` even after the surface
height (≤ ~0.6) is subtracted.

The physical reading is honest rather than a workaround: **a matte is a flat,
opaque coat of paint.** Its interior has constant height, so its gradient is
zero, so it lights flat — no weave, and no *varying* gloss: being opaque paint it
carries the film's uniform sheen (§6.3), but with a flat normal that sheen is an
even wash rather than a glint. That is what a mat board looks like. Its boundary
is a height cliff and therefore catches light, the same way every paint stroke's
edge does; at the frame border that reads as a crisp bevel, which is wanted.

#### 15.4.2 The matte's aux blend must be *over*, not additive

The color space's `aux_blend()` is **additive** — correct for paint, where
thickness accumulates. If a matte blended additively, the height of paint
*underneath* it would survive, and `height_at` would emboss that paint's impasto
as ghost ridges through an opaque mat board.

So the matte pipeline declares its own blend state: premultiplied **over** on
both targets. The aux then composites as `aux' = aux·(1−a) + H·a`, right at both
ends — an opaque matte erases the relief beneath it, a 30% scrim keeps 70% of it.
(`OneMinusSrcAlpha` as a destination factor is valid on the alpha-less `R16Float`
aux target: the factor reads the *source* alpha from the fragment shader's output
vec4, which exists regardless of the format's channel count.)

#### 15.4.3 Matte opacity is non-linear, and that is deliberate

Layer opacity `λ` scales *both* inputs to the slab law — premultiplied color
(so `color.a = λ`) and aux thickness (so `aux.x = λ·H`) — giving
`vis(λ) = 1 − exp(−K · λ² · H)`, **quadratic in the exponent**. With `H = 8` that
is pronounced: `λ = 0.5` covers ~86%, not 50%. Measured on a black frame over a
red stroke, the outside band reads `[222,61,36]` hidden → `[81,10,2]` at half →
`[20,8,2]` opaque.

This is kept, because it is **exactly** what paint-layer opacity already does:
pass A scales premultiplied color and additive aux by `λ` too, so a paint layer
is `1 − exp(−K·λ²·op·t)` — the same form. Consistency with paint is the entire
premise of making a matte a layer, and a compensating curve here would make the
matte the one layer whose opacity slider means something different.

It would be easy to make a matte alone exactly linear (write `color.a = 1` and
`aux.x = −ln(1−λ)`), and that is the *right* model for "opacity means visible
coverage". But if the curve is wrong it is wrong for paint layers first, and the
fix belongs in the vis law, once, for both. Noted so the choice stays visible.

#### 15.4.4 Interleaving with tiles

Pass A's flat tile list is an ordered item list, because a matte has to composite
*in stack order*:

```rust
pub enum CompositeItem {
    Tile { coord: TileCoord, handle: TilePairHandle, opacity: f32 },
    Matte(MatteDraw),
}
```

The compositor walks it in order, switching pipelines where a matte sits between
runs of tiles. This costs nothing: a tile already needs its own draw because it
needs its own bind group, so pass A was never one batched draw — interleaving
mattes adds no per-tile overhead, and an all-paint document issues exactly the
draws it did before (every golden unchanged, which is the proof).

The matte draw is a fullscreen quad; the vertex stage inverts the view uniform's
canvas→NDC transform to hand the fragment stage a canvas-space position, and
coverage comes from a signed distance to the rect, antialiased over one screen
pixel (`1/zoom` canvas px). Same technique as `selection.wesl`, same seam-free
property. One constraint worth recording: pass A's view bind group is declared
**vertex-only**, so the fragment stage cannot read the zoom from it. The
antialiasing width is computed in the vertex stage and passed as a flat varying —
constant across the quad, so this is exact, and it avoids coupling the matte to
the overlay pass's separate `VERTEX_FRAGMENT` layout.

### 15.5 What a matte is *not*: the substrate

A matte is a slab of opaque paint. The **substrate** — the color of the canvas
itself — is a different thing: it is *under* everything, it is lit, and the weave
shows through it. The media pass handles it as `m.bg`.

It is `DocState.background`, sitting beside `DocState.surface`, on precisely the
argument §6.4 makes for the weave — which canvas a piece was painted on is part
of what the document *is*. (It used to be view state owned by the frontend, so
the ground you painted on was not saved with your painting.) Both exist and both
make sense: `background` is the gesso; a `MatteRegion::Everything` matte — now
built, the frame bar's "Add background" — is an opaque underpainting brushed
over it. Born at the **bottom** of the stack (`AddMatte` takes the full `Place`
anchor for exactly this), it wears either `MattePaint`: a flat paper tone, or a
graded wash (§22.4). It defines no export rect and mounts no handles — its bar
is its paint and its Done — and, having no rect, it can never masquerade as the
frame the export dialog or the navigator picks.

Making it is **not** a button in the Layers panel's header, though the ground is
a layer like the frame beside it. A header button is a standing cost: it is on
screen for the whole session, and it earns that only if it is reached for
repeatedly. A ground is made once per painting at most — the second would have
nothing to be, since "the whole plane" admits no second — so it is a chip in the
frame bar instead, mounted exactly while there is no ground and gone once there
is one. That is the same argument the bar itself is built on (§15.7): a control
meaningless in the current state is absent rather than greyed out. The cost is
that the ground is reached through a frame, since the bar mounts only while a
matte is selected; the answer is that a piece wanting a ground almost always
wants a frame, and once the ground exists it is an ordinary row in the Layers
panel like everything else.

### 15.6 Export

**Save is not export.** Two files leave this app and they are not the same
object. **Save** writes the action log: replayable, still editable, undo history
intact on reopening — the thing that must never be lossy. **Export** writes a
picture: one frame, flattened and lit, from which nothing can be recovered. They
are named apart in the menu because an artist who "exports" thinking they saved
has lost the painting.

**Export takes a layer id**: a matte layer's region bounding box is the output
rect, and no matte selected falls back to `DocState::bounds`. The layer panel's
selection is therefore already the frame picker, and multiple frames need no new
machinery. This composes without a single special case — render every visible
layer into the frame's rect and the right thing happens by construction:

- the frame matte covers only *outside* the rect, which is clipped away, so it
  contributes nothing to its own export;
- a ground matte is inside and contributes exactly what it should;
- a matte whose visibility is off still defines the rect, because geometry and
  presentation are separate properties of the same layer.

**The rect an export writes is also the rect the view is framed to.** Opening a
document puts the whole piece on screen — `ViewCommand::ShowPiece`, dispatched by
`files::open_bytes` between the replay and the first paint — and it reaches that
rect through the same code, `Engine::piece_rect`: the named frame, else the painted
bounds. A view is per-client session state and so is *not* in the file (§18.1.2), so
without this a painting opens at whatever pan and zoom the last one was left at,
which on an unbounded canvas is routinely an empty stretch nowhere near the paint
that just arrived. The navigator's miniature (§11) is the third asker of the same
question, which is what makes "what a file would hold", "what the overview shows"
and "where you are when it opens" one answer rather than three that agree until one
of them is edited. Which frame *is* the piece stays the frontend's (`piece_frame`:
the topmost matte with a rect), exactly as it is for the export dialog, whose only
difference is that it prefers a selected frame — the one being composed.

Two deliberate differences, both about where the **last resort** is:

- **A document with neither paint nor frame does not move the view.** An export
  must write something and so falls back to the viewport; framing the view that
  way would zoom the window onto itself. The shared rule therefore answers
  `Option`, and each caller supplies its own fallback rather than inheriting one
  that only suits the other.
- **The view keeps a margin; the file does not.** A file is the piece and nothing
  else, while a view of it is a thing on an easel — a piece flush with all four
  window edges reads as one that carries on past them.

The easel is also straightened: `ViewTransform::show_rect` fits upright and
unmirrored, on the same reading that has `ExportPlan::view` write a file upright at
whatever angle the canvas is being worked at (§18.1.2). It is a different question
at an angle in any case, since the rect's screen footprint is then a larger, turned
box.

Export renders through an **explicit view**: centred on the frame, at
`zoom = scale`, into a target sized `frame.rect × scale`. That view-taking entry
point is **private** — the screen has no reason to render through anything but
the session's own view, and `export` is the only consumer. The public surface is
`render`, `render_to_image`, `export` and `export_plan`.

`ExportScale` is the vocabulary for "how large", and it has three words because
there are three questions a caller actually has: a `Factor` of the canvas size, an
exact `Width`, and `Fit`ting a box on both axes. The third is the navigator's
(§11), and it is in the engine rather than in the panel for a reason worth
recording. The panel used to ask for a **1× plan purely to learn the rect's size**
and work the fitting scale out itself — which put a question with a *stricter*
precondition in front of the one it wanted, since a 1× plan of a piece past the
device's `max_texture_dimension_2d` is refused as a texture it could not
allocate. Past that much painting or frame the query failed (on the app's device
around 8k), `draw_overview` returned `None`, and the
miniature silently went on showing a stale picture at exactly the size where an
overview earns its place. A preview must be answerable for a piece of *any* size,
so the question it asks may not be routed through one that isn't.

**What "too large" means is asked of the device, not written down.** `export_plan`
refuses a size past `device.limits().max_texture_dimension_2d` so a huge scale
comes back as an error rather than a wgpu validation panic. That used to be the
literal 8192, and the literal was wrong in the *permissive* direction:
`wgpu::Limits::default()` caps 2D textures there and the frontend requests it, so
on the app's device the two agreed by coincidence — but the headless device
(`GpuContext::headless`) asks for `downlevel_defaults`, the 2048-px web/WebGL2
floor, and every size from 2049 to 8192 passed a check written against a limit
that device did not have. A guard kept in step with a limit it does not read is a
guard already out of step somewhere. Reading it also lets the ceiling *rise*: the
adapters this runs on report far more (32768 is common), so a frontend that
requests more gets more, with nothing here to update.

**Export is `async`, and had to be.** Reading pixels back off the GPU is the one
inherently asynchronous GPU operation (§7), and it is the one place native and
web genuinely diverge:

- Natively `Device::poll(Wait)` blocks until the queue drains, so the map
  callback has already fired when it returns.
- On WebGPU there is **no blocking poll**. `mapAsync` returns a promise settling
  only when the browser's event loop runs, so `poll` is a no-op and
  `getMappedRange` on the next line fails with `OperationError`.

The first cut shipped the blocking shape and export died in the browser exactly
there. The *second* cut made `export` an `async fn`, which died differently and
more interestingly: an `async fn` holds `&mut self` for the whole readback, and a
frontend takes that borrow from a shared cell — so the engine stayed locked
across an await during which the UI re-rendered, read the renderer, and panicked
with `AlreadyBorrowedMut`.

So `export` is **not** an `async fn`. It renders synchronously and returns a
*future for the readback alone*, owning a cloned `GpuContext` (cheap — wgpu
handles are reference-counted) and the target texture, borrowing the engine not
at all. The caller drops its guard before awaiting:

```rust
let readback = { engine.write().export(frame, scale, bg)? }; // borrow ends here
let image = readback.await;
```

That is the shape any `!Send` engine behind a shared cell wants for an async GPU
operation, and it makes the discipline structural rather than a comment asking to
be believed. `gpu::readback` is async, with the *only* target-specific line being
how the callback gets driven: native blocks on `Wait` before awaiting (a
non-blocking `Poll` deadlocks — the executor parks the sole thread and nothing
polls again), while web relies on the await itself yielding. The blocking
`read_rgba8_blocking` remains for the golden tests, which are native by
construction, and is **`cfg`-compiled out on wasm** so the same mistake cannot be
reintroduced from the frontend. `render_to_image` and `replay_timelapse` are
native-only for the same reason.

**A readback returns the target's bytes in the *target's own* channel order**,
and `RgbaImage` is RGBA by definition, so a BGRA target has to be swizzled. Every
test here renders to `Rgba8Unorm`; a browser surface is typically `Bgra8Unorm`.
The first working export therefore came out with red and blue swapped — salmon
paper as pale blue, orange paint as blue — while green, black and white were
untouched, because all three are fixed points of an R↔B swap. That is what made a
byte-order bug read as a color-space bug. `RgbaImage::from_target_bytes`
normalizes it, and `export_is_rgba_whatever_the_target_format_is` paints the same
thing on an RGBA and a BGRA engine and demands identical bytes — the check no
single-format test could make, which is precisely why the class of bug survived
nine passing export tests.

Three decisions that go with it:

- **Scale** is a property of the output, not the artwork. The frame stores a
  canvas-space rect only; the export offers 1× / 2× / explicit pixel dimensions.
- **Transparent background** skips the media pass's substrate composite — a real
  branch, not merely an alpha. Compositing over the substrate and *then* zeroing
  alpha would leave every bare-canvas texel carrying substrate color at zero
  alpha, which fringes the moment the PNG is composited over anything else.
- **The overlay pass is suppressed.** Selection outlines and composition guides
  are chrome and never reach a file. Keyed on *exporting*, deliberately not on
  the background: the first cut tied chrome suppression to `Transparent`, which
  silently leaked the selection outline into every opaque PNG.
  `export_omits_the_selection_outline` is the regression.

Export is safe at any scale because the relief is already zoom-normalized:
`strength = m.light.w / m.surf_a.z` divides the screen-space gradient by the
canvas px it spans, so a 2× export has the same slope, resolved finer.

### 15.7 Interaction

A frame has **no permanent panel**, and — the sharper form of the same rule —
**nothing that is an ordinary layer property gets a frame-specific control.**
Creating one is `+ Frame` in the Layers panel; opacity (which *is* the crop
scrim) and removal are the Layers panel's single set of controls for whatever is
selected, applying to a frame and a paint layer alike. Only the fill color lives
in the frame bar, because it is the one thing about the frame rather than about a
layer. Duplicating opacity and delete into a frame-specific bar would have meant
two controls for one property. The rest lives in a bar mounted only while a frame
is **selected**, alongside the selection bar and on the same argument (§6.8).

**There is exactly one selection.** Selecting a frame is clicking its row in the
Layers panel — the same click, and the same `ViewCommand::SetActiveLayer`, that
selects a paint layer. **The frame bar and the on-canvas handles key off
`active_layer` being a matte.** There is no separate frame-selection state
anywhere.

That means `active_layer` is **the selected layer**, not "the paint target". The
widening is deliberate and is what makes the interface simple: with one selection
concept, "exactly one row is highlighted" is a *consequence* rather than a rule
two pieces of state have to be kept agreeing on. It also removes any way for a
stroke to land on a layer that does not look selected. (An earlier cut had two —
the engine's `active_layer`, which refused mattes, plus a frontend
`selected_frame` signal. Both could be set at once, which read as two selected
rows, and patching it needed mutual-exclusion rules plus an auto-deselect when a
stroke began. The duplication was the bug, not the symptom.)

**A stroke aimed at a frame does nothing** — refused identically by `apply` and
by the preview path, so `preview == committed` holds and a replayed or remote log
agrees, and no frontend needs a rule. Rather than block the gesture, the canvas
says so first: the brush crosshair becomes `not-allowed` whenever the selected
layer takes no paint. Blocking in the frontend was rejected — it is a rule a
second frontend would have to reimplement, and this codebase consistently puts
such rules in the engine. In the Layers panel a matte reads as "◱ Frame" behind a
dashed border. An `Option<LayerId>` active layer was also considered, so "nothing
selected" could be expressed; skipped, because `DocState` always has at least one
layer, so `None` would be representable but unreachable — its own kind of lie.

Three creation paths matter more than dragging:

- **Add frame** — sized to the painted content if there is any, otherwise to what
  the viewport shows. Both are "frame what I am looking at", the only sensible
  default on an unbounded canvas.
- **Fit to art** — snap to `DocState::bounds`. **Fit to view** — snap to the
  viewport.
- **Aspect** — a drop-down of 1:1, 4:5, 3:2, 16:9, reshaping about the centre and
  *preserving area*, so switching neither grows nor shrinks the piece. It reads
  the frame's current ratio back, showing `Custom` when a dragged handle has
  landed on something arbitrary — a state readout rather than a row of
  fire-and-forget buttons. `Custom` is offered only while it is what the frame
  *is*.

Once it exists, it is adjusted by **handles drawn over the canvas**: eight
edge/corner grips plus a move pill. Two decisions there are forced by this frame
being non-clipping:

- **The interior is not interactive.** `pointer-events: none` on the frame box,
  `auto` only on the grips. The inside of the frame is exactly where you paint,
  so it must pass every pointer event through to the canvas.
- **Hence the move pill, outside the top edge.** Dragging the interior is how
  every other crop tool moves a frame, and it is the one gesture this frame
  cannot borrow.

A drag **previews live and logs once**: each pointer move sends
`ViewCommand::PreviewMatteRect` (view state, never logged), and release commits a
single `DocCommand::SetMatteRect`. So a drag costs one undo step rather than one
per move. `observe()` reports the *previewed* layer rect, which keeps the handles
under the pointer rather than a frame behind — carefully **only** the layers,
since `has_selection` must stay committed-only or a marquee drag flashes the
selection bar in and out. This is a view command rather than a `GestureCommand`
because a frame drag is handle-relative, not sample-driven: there is no
`InputSample` to feed `Start`/`To`/`End`, and which grip is held is the
frontend's business. What it keeps is the shape that matters — build in view
state, commit once on release.

**The frame's color makes the same bargain**, through
`ViewCommand::PreviewMatteColor` and one `DocCommand::SetMatteColor` on release.
It has to: what is being chosen is how the mat board reads against the piece
inside it, which is a judgement made *by looking*, so every color the pointer
crosses has to reach the canvas and only the answer belongs in the log.

It is picked with the app's **Oklab picker**, the same control the substrate
color uses (§15.5) — one control asking the same question about a different
flat expanse, rather than two that resemble each other. That is not only
consistency:

- **A mat board is chosen by lightness against the piece.** Too close and the
  frame stops reading as a frame; too far and it shouts over what it surrounds.
  Oklab puts that search on an axis you can drag along — `L` moves lightness with
  hue and chroma held — where the sRGB triple a native color input offers moves
  all three at once.
- **The edges are the app's own.** The preview/commit split rests on
  `pointerup` / `pointercancel` over the picker's tracks, not on what a browser
  chooses to send when its color dialog closes. A cancelled pick still commits,
  for the reason `panels::color::end_pick` gives: every instant of it is a color
  the user chose and is already looking at, and discarding it would strand the
  preview with no commit to supersede it.

A commit to the color the matte already holds is refused engine-side (§14.6),
so a pick that lands back where it started logs nothing while still superseding
what it was showing. What was given up with the native input is typing a hex code
and the OS eyedropper; the pop-out is the trade, and the picker is where either
would be added.

Still to come: snapping while dragging (to content bounds, to other frames, to
the canvas origin) is most of what makes a crop tool feel good and is cheap;
frame from selection; ratio-locked dragging.

### 15.8 Composition aids stay view state

Thirds, golden section, diagonals, centre cross and a custom grid are an *aid*,
not the artwork: per-client, never replicated, never exported. They read their
rect off the selected matte layer. The temptation to make them a layer should be
resisted for the same reason `MediaParams` is not one.

**Review mode** is one keystroke — fit to frame, matte to full opacity, chrome
hidden, selection outline suppressed. That is stepping back from the easel, and
the `canvas_active` chrome-fade machinery already does the hard part. Paired with
the view mirror (§18.1.2) it becomes the complete "how does this actually read?"
check in two keys.

### 15.9 Phasing

- **P1 — the matte layer. Done.** `LayerContent` enum; `MatteRegion::OutsideRect`;
  `matte.wesl` and its pipeline; `CompositeItem` ordering in pass A; `AddMatte` /
  `SetMatteRect` / `SetMattePaint`; strokes refuse matte layers; `bounds` ignores
  them. Since extended with `MatteRegion::Everything` (the ground, §15.5) and
  the gradient `MattePaint` (§22.4) — each with the UI that makes it real.
- **P2 — export. Done.** `export(frame, scale, background)` and `export_plan`;
  `RgbaImage::to_png`; `DocState.background` + `SetBackground`;
  `Background::Transparent` as a real branch; the Open / Save / Export menu
  items, the export dialog, and the browser file plumbing.
- **P3 — the composition tool. Mostly done** (taken before P2, since export needs
  something to frame *against* before it can be tested by hand). `+ Frame`; matte
  rows; the frame bar; the on-canvas grips with live preview and single-action
  commit. **Not yet:** snapping, composition guides, review mode, fit-to-frame,
  frame-from-selection.
- **P4 — the general region.** `MatteRegion` becomes the `SelectionOp` algebra:
  comic gutters, lasso mattes, frame-from-selection. (The `All` slab arrived
  early as `Everything` — it needed no algebra, only a variant and a button.)

### 15.10 Testing (`tests/matte.rs`)

- **`frame_covers_outside_and_spares_inside`** — the core claim, and the §15.4.1
  regression: a matte that failed to write the aux target would be perfectly
  invisible and this catches it.
- **`opaque_matte_erases_relief_beneath`** — the §15.4.2 ghost-ridge regression,
  formulated as *an opaque matte over a heavy stroke must render identically to
  the same matte over bare canvas*. Compared on the lit image, so a surviving
  height field shows up as shading. The failure the design is most likely to get
  wrong and least likely to be noticed by eye.
- **`matte_honors_layer_opacity_and_visibility`** — monotonic between opaque and
  hidden, deliberately asserting no midpoint (§15.4.3), and on total brightness
  rather than per channel: against a red stroke the blue channel is floored at
  both ends and has no range to be "between" in.
- **`matte_below_paint_does_not_cover_it`** — guards the ordered walk against
  being flattened back into "all tiles, then all mattes".
- **`a_matte_can_be_selected_but_takes_no_paint`** — the one-selection model.
- **`matte_does_not_extend_canvas_bounds`**, **`matte_undoes`**,
  **`dragging_a_frame_previews_without_logging`**.

Two notes on what is *not* tested and why: the §6.4 seam invariant needs nothing
new — a matte samples no tile, so it cannot introduce a seam; and every
pre-existing golden is unchanged by this work, which is the evidence that turning
pass A's flat tile list into an ordered item list is behaviour-preserving for
paint-only documents.

---


