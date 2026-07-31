# Groups and clipping

Two features every drawing app has, and neither is well modelled anywhere:

- **Layer groups.** Rebelle's are purely organizational — a folder that cannot
  change the picture. Photoshop's are functional, so they need a blend mode; but
  then a group that merely tidies the stack would change the render, so it also
  needs a fake **pass-through** mode; and once a group has a blend mode of its
  own, that mode and the bottom member's are two controls answering one question.
- **Clipping masks.** A toggle that makes a layer transparent where the *next
  unclipped layer below* is transparent. New users do not guess this, the arrow
  in the panel points at one layer while the behaviour involves a run of them,
  and in Rebelle and CSP a clipping chain quietly becomes a group — so there are
  now two grouping mechanisms that look nothing alike.

This document specifies one mechanism that is both. It is written against the
code as it stands ([layer.rs](crates/stark-core/src/document/layer.rs),
[state.rs](crates/stark-core/src/document/state.rs),
[composite.rs](crates/stark-core/src/gpu/composite.rs),
[blend_common.wesl](crates/stark-shaders/src/shaders/blend_common.wesl),
[patch.rs](crates/stark-core/src/document/patch.rs),
[footprint.rs](crates/stark-core/src/document/footprint.rs)), and several of its
decisions are *forced* by that code rather than chosen — those are called out
where they occur.

## 1. The stance: one sentence

> A layer may **carry** other layers. A layer's blend mode, clipping and
> opacity describe how that layer *together with everything it carries* meets
> what lies beneath it.

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
no group object, no pass-through, no clipping chain, and no rule that only
applies at one level.

## 2. The representation: a layer carries layers

There is no `Group` type. A group **is** the layer at its base:

```rust
pub struct Layer {
    pub id: LayerId,
    pub blend: BlendMode,
    /// Clip to the paint beneath — §4.
    pub clip: bool,
    pub opacity: f32,
    pub visible: bool,
    pub name: Option<Arc<str>>,
    pub content: LayerContent,
    /// Layers carried on this one, bottom-to-top. A group is a layer with a
    /// non-empty `carries`; there is no other kind.
    pub carries: Vector<Layer>,
}
```

Four properties fall out of this rather than being designed:

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

## 3. Why the base's blend mode is free

The objection to Photoshop's design is duplication: the group's blend mode and
its bottom member's answer the same question. Taking the base's mode *as* the
group's is only an improvement if the base's mode was doing nothing. In this
engine it provably is not doing anything, and the proof is already in the tree.

`merge()` against an empty backdrop
([blend_common.wesl:237-248](crates/stark-shaders/src/shaders/blend_common.wesl#L237-L248))
reduces with `cb = 0` to

```
out.rgb = mix(cs.rgb, blended · cs.a, 0.0) + 0.0 = cs.rgb
out.a   = cs.a + 0.0 · (1 − cs.a)          = cs.a
```

— bit-for-bit the `Normal` result, which is deliberate and which `tests/blend.rs`
already asserts to the byte
([blend_common.wesl:226-228](crates/stark-shaders/src/shaders/blend_common.wesl#L226-L228)).
The substrate does not rescue it either: the ground is composited in pass B,
after all blending, so the bottom of a stack genuinely has nothing underneath
([layer.rs:174-182](crates/stark-core/src/document/layer.rs#L174-L182)).

So the bottom layer of any stack carries a blend-mode slot that **cannot express
anything**. We are not overloading a control; we are filling a hole. The same
argument runs for clipping, and §4.3 spends the slot the same way.

That is also the rule for *which* properties the group takes from its base:

| | belongs to | why |
|---|---|---|
| `blend`, `clip` | **relational** — the group's, taken from the base | vacuous at the base, so free |
| `opacity`, `visible`, `name` | **intrinsic** — the group's own, as the base's own | the base's opacity does real work; it cannot be borrowed |

Group opacity is therefore *not* the duplication Photoshop's group blend mode
was: fading a group fades the base and everything on it as one unit, which is
what it should do. The one thing this model cannot express is fading or hiding
the base *alone* while what it carries stays at full strength. That is a real
loss, and it is the price of having no container object; it is also an operation
with no use anyone has named.

## 4. Clipping (the clipping mask, restated)

`clip` is a per-layer boolean applied at the same step as the blend mode. It
means: **this layer exists only where there is paint beneath it in its group.**

Two ways it differs from every clipping mask in the field, both simplifications:

- It inherits the alpha of the **whole composited stack below it within its
  group**, not of the nearest unclipped layer. There is no chain, no "next
  layer that is not itself clipped", nothing to trace up the panel.
- **The group is what bounds "below".** Clipping to exactly one layer is not a
  special mode; it is that layer carrying the clipped one. One mechanism does
  both jobs, and the thing users actually mean by "clip to the layer below" is a
  single drag.

It keeps the field's name anyway. This is not the clipping mask the other apps
ship, but it is the nearest analog by a wide margin, and a painter arriving with
the concept will reach for the right control on the first try — which is worth
more than a name that would be accurate to a reader who does not have the
concept yet. §6 is where the two differences have to be made visible.

### 4.1 The formula, and why the obvious one is wrong

The natural phrasing — *multiply by the opacity of what it is compositing onto*
— is wrong if implemented as `αs ← αs · αb`. With `αb = 0.5, αs = 1` that yields
an output alpha of `0.5 + 0.5·(1−0.5) = 0.75`: the clipped layer **invented
coverage** the backdrop did not have, and the backdrop shows through paint that
should be opaque.

The correct operation is not a scale, it is a **deletion**: drop the term for
source that lands where there is no backdrop. Writing the existing merge with
that term visible (`mix(x, y, αb) = (1−αb)·x + αb·y`):

```
unclipped:  rgb = (1−αb)·αs·Cs  +  αb·αs·B(Cb,Cs)  +  cb.rgb·(1−αs)
            a   =    αs·(1−αb)  +  αb·αs           +  αb·(1−αs)      =  αs + αb(1−αs)

clipped:    rgb =        0      +  αb·αs·B(Cb,Cs)  +  cb.rgb·(1−αs)
            a   =        0      +  αb·αs           +  αb·(1−αs)      =  αb
```

The output alpha collapses to exactly `αb` — the group's coverage is untouched
by anything clipped to it, which is the property that makes clipping
composable and which the scaled-alpha version does not have. Note the tail keeps
the **unmodified** `αs`: inside the backdrop's region the source still covers
`αs` of it.

In the shader this is one factor. With `m = clip ? 0.0 : 1.0`:

```wgsl
out.color = vec4(mix(cs.rgb * m, blended * cs.a, cb.a) + cb.rgb * (1.0 - cs.a),
                 cs.a * (cb.a + m * (1.0 - cb.a)) + cb.a * (1.0 - cs.a));
```

which is `merge()`'s current line with `* m` and one factor added, and which is
bit-identical to today's output at `m = 1`.

### 4.2 Clipping must scale the aux, or you get ghost impasto

**Forced by the media pass.** `merge()` sums the height field unconditionally
([blend_common.wesl:243-247](crates/stark-shaders/src/shaders/blend_common.wesl#L243-L247))
on the grounds that height is *amount of paint* and paint stacks whatever its
colour does. Clipping is the case that breaks the grounds: a clipped layer's
colour is suppressed outside the backdrop, and if its height is not, the media
pass lights relief where there is no paint. So:

```
out.aux = hb + hs · (clip ? αb : 1.0)
```

Every ridge a clipped stroke lays outside its group's paint has to go with it.

### 4.3 Clipping the base clips the group

The base's `clip` points **outward**, exactly as its blend mode does: it clips
the composited group to what lies beneath the *group*. That is not a second rule
— it is §1's sentence unchanged, and it is why clipping a whole group needs no
mechanism of its own. In the compositor it needs no branch either: the recursion
merges each subtree into its parent's backdrop through that subtree's own blend
and clip, and the base's fields *are* the subtree's.

So `clip` is live wherever there is a backdrop to clip to, at any depth. Define
that once:

```
has_backdrop(L) = L has a sibling below it in its stack
               ∨ (L is carried by C ∧ has_backdrop(C))
```

This is the same predicate that decides whether a **blend mode** does anything,
which is the point: the two relational properties go live and inert together,
and there is one fact to teach rather than two.

It fails in exactly one place — the bottom-most layer of the root stack, which
has nothing under it anywhere (§3). There the two properties degrade
differently, and that asymmetry is the only reason the UI has to care: a blend
mode over an empty backdrop is the **identity**, so leaving it set is harmless,
while a clip over an empty backdrop is **annihilation** — the layer disappears.
Both controls are therefore shown inert on that one row rather than allowed to
do nothing and erase respectively.

## 5. What grouping costs, and why it is safe here

Groups are always isolated. A `Multiply` layer *inside* a group multiplies
against the group, not against what lies under the group — so wrapping layers
can change the render. That is exactly what pass-through was invented to prevent,
and declining to invent it is the one place this model is worse than Photoshop's.
It is worth it because of what it is bought with, and because in this app the
cost is much smaller than it would be elsewhere.

**Pure organization is free, structurally.** A group whose base is `Normal`,
unclipped and fully opaque, *and every member of which is `Normal` and
unclipped*, has nothing to isolate: §7 collapses it into the surrounding run at
build time, so it produces the identical draw list and the identical pixels. The
only groups that cost anything are the ones that change a blending scope.

**Where a scope does change, this blend family absorbs it.** Every mode past
`Normal` is addition conjugated by a tone curve, hence commutative, associative,
with an identity — that is the whole argument of
[layer.rs:79-101](crates/stark-core/src/document/layer.rs#L79-L101), and it pays
for itself here:

- **Opaque layers under one mode:** grouping is *exactly* invariant, by
  associativity. Grouping three glow layers changes nothing at all.
- **`Multiply` at any coverage:** exactly invariant. Working it through, the
  `mix(…, αb)` tail and multiply's white identity cancel, and both routes give
  `A·(1−s+sB)·(1−t+tC)` for a backdrop `A` under layers `B, C` at coverage `s, t`.
- **The emissive modes at partial coverage:** a small drift. For `A=0.4, B=0.6,
  C=0.8` at `s=t=0.5` under Glow: `0.6902` ungrouped, `0.6886` grouped.

In Photoshop, where the modes are ad-hoc formulae with no algebraic relationship,
grouping changes the picture arbitrarily — which is *why* it needs pass-through.
Here the modes were chosen so that regrouping is a non-event, and the feature
that would paper over the difference is not needed because the difference is not
there.

## 6. What the panel shows

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
  group's*, and they are drawn at the bottom edge of the group's bracket to say
  so. Clipping there clips the whole group to what is under it (§4.3), so the
  rail it draws runs down past the group's own bracket rather than inside it.
- `⌐` is the clipping rail: a left-edge rule running from the clipped row
  down to the bottom of the stack it inherits from — the full run, not one
  layer. Photoshop's arrow points at one layer and is lying; this points at
  everything it actually reads.
- On the bottom-most row of the document both controls are inert (§4.3), which
  is the one place the panel has to say "this does nothing here".
- Indent means membership. Rail means clipping. They are different marks
  because they are different facts, and a user arriving from Photoshop — where
  indent means clipping — must be able to see that at a glance. `Freckles` above
  is in the group and not clipped; that is a state Photoshop's panel cannot draw.

Two commands cover the whole feature: **Carry** (put the selected layers on the
one below) and **Release** (promote what a layer carries into its parent stack).
"Clip to the layer below" is Carry plus the clip toggle, and can be one item
in a menu that does both.

## 7. Compositing

`CompositeGroup` becomes a tree, and the existing fast path becomes an invariant
of its shape rather than a special case inside the loop:

```rust
pub struct CompositeGroup {
    pub blend: BlendMode,
    pub clip: bool,
    /// Applied to this whole subtree at merge — members overlap, so it cannot be
    /// folded into per-tile opacity the way a leaf layer's can.
    pub opacity: f32,
    pub content: GroupContent,
}

pub enum GroupContent {
    /// A run that composites under plain premultiplied "over" — no isolation.
    Run(Vec<CompositeItem>),
    /// Isolated members, bottom-to-top, each merging into the one below.
    Stack(Vec<CompositeGroup>),
}
```

Build-time rules, in `Engine::composite_groups`:

1. A `Stack` all of whose members need no isolation, and which is itself
   `Normal`, unclipped and opaque, **collapses into a `Run`**. This is §5's
   "organization is free", enforced structurally rather than promised.
2. Anything that needs no isolation joins the enclosing `Run` — which, because
   rule 1 runs first and bottom-up, includes a *group* that collapsed. Tidying
   layers into a folder therefore costs nothing at all, not even a group
   boundary the encoder has to step over.
3. A document with no groups, no modes and no clipping therefore produces
   exactly one `Run` — today's draw list, unchanged, at today's cost.

Encoding recurses. Each nesting level in use needs its own ping-pong pair plus
the pair its members isolate into — about 40 MB at 1080p per level, on the order
of what blend modes already allocate
([composite.rs:180-194](crates/stark-core/src/gpu/composite.rs#L180-L194)), and
allocated lazily to the deepest level the document actually reaches. The parity
trick that lands the final result in the caller's own targets
([composite.rs:1127-1133](crates/stark-core/src/gpu/composite.rs#L1127-L1133))
still applies, per level.

`BlendUniform` gains two fields, `clip` and `opacity`, filling the 16 bytes it
was already padded to. `Normal` now reaches the pass — a clipped normal layer,
or a group faded as a unit — where before it never did, so `merge` grew a
`Normal` branch that takes the premultiplied source verbatim rather than the
mode's opinion. That branch is not a shortcut but an exactness requirement: a
clipped normal layer over solid paint has to match the fixed-function `over` an
unclipped one gets, and dividing by `αs` to feed the blend function and
multiplying it back would not.

An unclipped `Normal` layer at full opacity is still the absence of a pass.

Two consequences to know about:

- **A leaf layer's opacity stays per-tile** ([composite.wesl:84-89](crates/stark-shaders/src/shaders/composite.wesl#L84-L89)).
  Tiles within a layer do not overlap, so scaling each tile is identical to
  scaling the composited layer — but members of a group *do* overlap, so a
  group's opacity must be applied at the merge. That asymmetry is not a wart; it
  is the same fact stated at two granularities.
- **An opaque group does not erase the relief beneath it.** `merge()` sums the
  aux, so impasto under an opaque group embosses through it, exactly as it
  already does under an opaque non-`Normal` layer and unlike an opaque matte
  ([blend_common.wesl:230-236](crates/stark-shaders/src/shaders/blend_common.wesl#L230-L236)).
  Groups make the existing wart easier to hit. The fix — `hb·(1−αs) + hs` for
  `Normal`, so paint hides the relief it covers — is a change to how *today's*
  blend layers render and is deliberately not bundled in here.

## 8. Plumbing

Less new machinery than the feature suggests, because the existing structural
resource is already coarse.

- **Actions.** `AddLayer`, `AddMatte` and `MoveLayer` each grow a
  `carrier: Option<LayerId>` — "whose stack", beside the existing "above which
  sibling". One new kind, `SetLayerClip(LayerId, bool)`. Carry and Release get
  **no actions of their own**: carrying *is* a move to a position inside another
  layer, so `MoveLayer` covers reorder, carry and release by which of its two
  anchors changes. One structural action, one inverse.
- **The file format.** A field in the middle of an existing struct variant is not
  something postcard can absorb — it writes fields in order with no names and no
  length — so this is the first change here that could not be *appended*, and
  `WIRE_VERSION` goes to 2. The alternative was a second `MoveLayer` variant
  preserving the old layout, which would have put the duplication this design
  exists to remove straight back into the log.
- **Footprints.** `Resource::StackOrder` is already "the relative z-order of the
  whole stack", one coarse resource, on the argument that concurrent reorders
  genuinely do not commute
  ([footprint.rs](crates/stark-core/src/document/footprint.rs)). Nesting rides on
  it unchanged. `Prop(LayerId, Prop::Clip)` is one new variant, beside `Blend`
  rather than folded into it: the two are applied at the same step but written by
  different actions, so a clip toggle has to commute with a blend change on the
  same layer.
- **Patches.** `PatchOp::Order(Vec<LayerId>)` becomes
  `Structure(Vec<(LayerId, Option<LayerId>)>)` — the flattened order *plus* each
  layer's carrier. Still one op, still restoring the whole shape, since it
  already restored the whole order; and still restoring only the *shape*, taking
  each layer's current record from the state it is rebuilding, so a commuting
  action that painted on a layer in the gap keeps its work.
  `PatchOp::Present { index, layer }` records a `LayerSite` — a carrier id and a
  position in that stack — rather than a flat index or an index path, because ids
  are stable under everything below them moving. `Layer` owns its subtree, so a
  removed group restores with what it carried.
- **Peers.** Concurrent moves that would make a layer carry its own ancestor are
  the only new failure mode. Because the log is totally ordered by
  `(lamport, actor)` and applied sequentially, the check is local and
  deterministic: a move whose target is a descendant of the moved layer applies
  as a no-op. No tree-CRDT cycle machinery is needed — the total order already
  supplies what one would.
- **Eyedropper.** `composite_groups(doc, Some(id))` means *that layer's own
  content*: no carried layers, unclipped, unblended. Sampling a clipped layer
  should show the paint that is there, not the paint that survives.

## 9. Invariants worth a golden test

1. A `Normal`, unclipped, opaque group of `Normal` unclipped layers is
   **bit-identical** to the same layers ungrouped. (§5, §7 rule 2.)
2. Carrying a layer and releasing it again is bit-identical to never having done
   it — the round trip through the tree surgery, the bounds recomputation, and
   the collapse firing a second time on the way back.
3. `clip` over a solid backdrop leaves the layer alone. Within a couple of
   least-significant bits, not to the byte, and the gap is worth naming: paint
   coverage is `1 − exp(−K·α·h)`, which never actually reaches 1, so a clip over
   even the heaviest passage correctly removes the last fraction of a percent —
   and the clipped layer takes the blend pass where the unclipped one takes
   fixed-function `over`, which round differently at half precision. `merge` is
   still written to make the two coincide exactly where `αb` *is* 1 (§4.1), and
   the test bound is ~10× below what a wrong clip would produce.
4. `clip` over an empty backdrop renders nothing — and contributes no height.
5. A clip inherits the **whole stack below it in its group**: a clipped layer
   still shows over paint that only the group's *base* has, two layers down,
   where a nearest-neighbour clip would cut it.
6. Clipping a group's **base** clips the whole group — the carried layers go with
   it (§4.3).
7. Grouping **does** rescope an interior blend mode (§5). Asserted, not merely
   accepted: it is the one place this model is worse than pass-through, and the
   test is where anyone who decides to "fix" it has to come and read why.
8. A blend mode on a bottom-most layer with nothing beneath it is still the
   `Normal` render — the existing `tests/blend.rs` assertion, which this design
   depends on and must not break.

## 10. Open

- **Nesting depth.** Two target pairs per level is affordable at three or four
  and not at twenty. Either cap the depth or spill deep levels; nothing in the
  model needs a limit, so this is purely a budget decision.
- **Aux under an opaque group** (§7) — a pre-existing question this feature
  makes more visible, not one it creates.
