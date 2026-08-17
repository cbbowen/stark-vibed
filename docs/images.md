# Placed images

Bringing a picture in from outside the document — an image file, or the system
clipboard — §23.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 23. Placing an image

Two gestures — **Place image…** in the menu, and **Ctrl+V** — that are the same
thing from the second step on: bytes arrive, the browser decodes them, and one
action lands them on the canvas as paint. Everything below is a consequence of
that last word. An imported image is not an attachment the document keeps a
reference to, nor a layer of a second kind that composites differently; it is
paint, laid by the same law a fill lays a parcel by (§18.0.4), and from the
moment it is placed the whole application applies to it. Stroke over it. Glaze
it. Lift it with a wet brush. Transform it (§16). Merge it down (§14.11). None
of that needed a line of code, and that is the argument for spending the
paragraphs below on the representation rather than on the feature.

The chapter runs from what the action names (§23.1) through where the pixels
land (§23.2), how they become paint (§23.3), and what the frontend contributes
(§23.4), ending with what is deliberately absent (§23.5).

### 23.1 A picture is content, named by the log

```rust
ActionKind::PlaceImage {
    id: LayerId,
    carrier: Option<LayerId>,
    above: Option<LayerId>,
    at: IVec2,
    name: Option<String>,
    image: AssetId,
}
```

**The same shape a custom brush shape has, and that is the whole design decision.**
A stroke names the shape it stamps with by the BLAKE3 hash of its decoded canonical
form, and a `SetSurface` names its ground the same way (§6.6, §6.4, §19); the log
carries the *name*, the bytes ride beside it in the bundle and over the wire as a
blob, and `content.rs` is the one place that answers "what does this document need,
and have I got it?". A placed picture is a third kind in exactly that mechanism —
`AssetNeed::Picture`, a third bag in `DocumentFile`, a third arm in four matches —
and no new machinery at all.

It was briefly built the other way, with the pixels in the action behind an `Arc` and
PNG-encoded on serialize, and the argument against content-addressing was that a
picture is neither named nor reused, so the indirection would buy nothing. That
argument was wrong in three places at once, and each is worth recording because each
is a property the mechanism *already had*:

- **It does not fit in a gossip message.** `MAX_MESSAGE_SIZE` is a megabyte, sized for
  the longest plausible stroke, and raising it to suit the largest picture anyone might
  paste would size every peer's gossip buffer for it too. Named, the action is forty
  bytes and floods like any other, while the pixels take the blob ALPN — BLAKE3-verified
  streaming, chunked and resumable by the protocol, with every member a provider (§12.4).
  Carried, it missed the flood and waited on a reconciliation sweep.
- **It deduplicates.** The same reference photograph placed on two layers is two
  actions naming one id: one entry in the bundle, one blob, and nothing at all to
  transfer for a peer that already holds it. Carried by value it was two copies in the
  log, two in the file, and two on the wire.
- **A joining peer can skip it.** `Request::SnapshotWithout` already lets a joiner say
  what it can resolve itself, and the join ceiling (`MAX_RESPONSE`, 64 MiB) is what
  stops new members joining a session that outgrows it. Pictures in the log push
  straight at that ceiling; pictures as content are the thing the mechanism was built
  to leave out.

What the `Arc` was solving disappears with it: an action is cloned constantly — a
commit clones one for the outbox, the history clones them while splicing an undo past
what it commutes with (§12.6), a peer's arrival clones one into the log — and now
every one of those copies thirty-two bytes. The `Arc` moved to
`PictureStore`, where it belongs: one decoded picture, shared by every replay that
crosses the placement.

**A layer arrives with the picture in it**, rather than the placement being spelled as
an add, a rename and a fill. A paste lands on its own layer in every tool that has
one, so three actions would put the familiar gesture three undo steps deep and leave
two of those steps meaning nothing on their own. `AddMatte` already carries this shape
(§15.2): a layer arriving with content is one fact, not a layer and then its content.
It mints its id like every other `Add…` (§17.9) and becomes the active layer, because
it is paint and an artist who has just placed a reference photograph is looking at the
layer they will work over.

**A picture is never given up on.** `content.rs`'s retry policy already had the
distinction this needs: a brush gives up after five rounds and the stroke draws with
the round tip, while a ground never gives up because applying the action against the
flat stand-in bakes a wrong deposit into tiles no later arrival un-bakes (§6.4). A
picture takes the ground's side, for a sharper reason than either — it has no degraded
form *at all*. A placement without its pixels is not a worse placement, it is an empty
layer, so releasing it would be releasing an action that adds nothing and reports
success.

**The footprint claims the layer whole** — `Existence`, `StackOrder`, `Paint(id, ALL)`
and the name. That is not a conservative shrug. Every other action that writes tiles
derives its box twice, once in the footprint and once where the tiles are planned, and
keeping the two in step is exactly the §12.6 hazard `fill_bounds` exists to close (a
fill wrote a tile its action never declared for as long as the two derivations were
separate). Here there is nothing to keep in step: the layer did not exist before this
action, so *all* of its paint is this action's whatever box the picture happens to
cover, and `image_tiles` is then the only quantization of that box in the tree.

### 23.2 Whole canvas pixels, which is a promise about resampling

`at` is an `IVec2`: the canvas position of the image's top-left texel, in whole
pixels. So the image's texels land on canvas pixels one for one, and nothing
between the file and the tiles is filtered.

That is a deliberate division of labour rather than a limitation. Scaling and
turning a placed image is `Transform` (§16), which is where resampling belongs
and where its exactness is already pinned to the byte (§16.4) — and where an
integer translation is *provably* lossless. Expressing the placement as a float
and then scaling it would spend one generation of bilinear blur on every import
to reach the same picture. An integer vector is how that is said in a form the
payload cannot express wrongly (§1's habit of ruling out a class rather than
checking for its instances).

Which tiles get written is `image_tiles`: the image's own extent, and every tile
whose **texture** — interior plus apron — holds any of it. Filtered rather than
merely quantized, because `TileRect::covering` floors both bounds, so an image
ending exactly on a tile boundary would otherwise name the tile past it; an
all-zero tile is worse than no tile, since it pollutes `bounds` and holds pool
memory for a texel of nothing (the same argument §16.5 makes for a transform's
dropped tiles).

The size cap is `MAX_PICTURE_DIM` (4096 px on the long edge), and it lives with the
other two caps in `stark-assetid` because it is the same kind of thing they are: part
of what the id *names*, so a source past it is box-downsampled before it is hashed
and the stored form reloads to the same id (§19). It is also a bound on what a
*stranger* can make this process allocate — content arrives from a file or a peer as
readily as from a clipboard, and a PNG says nothing about how much memory it decodes
to — so the decode reads the header's dimensions and refuses past a further ceiling
before expanding anything, which is §8's rule about a body that expands past the cap.
`MAX_IMAGE_TILES` is then **derived** from the dimension cap rather than chosen: it is
the only tile bound in the document that cannot disagree with the thing it bounds.

### 23.3 The tiles are built on the CPU, and that is the interesting part

`gpu::place` is the only tile writer in the engine with no shader.

Every other one computes a texel from texels — a stroke's sweep over what is
resident, a fill's parcel over a base, a transform's resample of a source quad —
so it belongs on the GPU, where the inputs already are. A placed image has no
such input: it lands on a layer that did not exist a moment ago, so there is
nothing beneath it to stack onto, and its texels are simply the file's texels
read through the paint representation. Uploading the image to a texture in order
to have a fragment shader copy it into another texture would be a round trip to
say nothing.

Three things fall out, and each is worth more than the pass it replaces:

- **Bit-exact everywhere.** These tiles are pure CPU `f32` arithmetic, so two
  peers and two replays on different adapters produce the same bytes — which is
  true of no render pass in this crate, and is why the goldens are
  adapter-specific in the first place (§9).
- **No dimension cap from the hardware.** Nothing binds the image as a texture,
  so the only bound on its size is the document's own rather than whatever
  `max_texture_dimension_2d` this device reports.
- **The apron is free, and provably right.** Each texel is computed from its own
  canvas position, so a tile's apron is bit-identical to its neighbour's interior
  by construction — §6.4's rule met by the strongest available form of the
  argument rather than by a pass being careful. `tests/seam.rs` holds this path
  to a far tighter tolerance than the two beside it for exactly that reason.

**What a source pixel becomes.** The source's alpha is a *coverage* — the
quantity the eye reads — and the paint that produces it is fully opaque paint of
whatever mass the slab law needs:

```
want = min(alpha, 1 − exp(−K · OPAQUE_MASS))
mass = −ln(1 − want) / K          // the inversion `fill.wesl` runs
color = latent · 1,  opacity = 1,  height = mass
```

This is the identical law and the identical constants a fill lands its parcel by
(§18.0.4) — `OPACITY_K` and `OPAQUE_MASS` are imported from
`lib/paint_common.wesl` by both, so the shader and the host cannot come to
different opinions about where "opaque" is. Two consequences are the whole point
of doing it this way rather than storing the image's alpha as coverage: an
opaque photograph lands opaque paint that takes the light, can be glazed over
and can be scraped back, and a soft-edged cut-out **thins** to nothing at its
edge rather than fading in color, which is what §6.1's "conserve height, never
alpha" means applied to imported pixels.

A fully transparent source pixel is an exact branch to nothing at all, for
`fill.wesl`'s reason: an all-but-zero height still reads as painted to `bounds`
and to the compositor, so a photograph with a transparent margin would otherwise
place a layer whose extent is the rectangle rather than the picture.

The color conversion is the host's `rgb_to_channels` (§6.7) — the same function
a fill converts its parcel with — so an image and a fill of the same color land
the same paint, in an Oklab document and a pigment one alike. It runs per texel
here rather than once per fill, which is the cost of an image being a picture,
and it is paid on import and on replay. The store is `f32 → f16` into the tile's
own channels, which is why `gpu::half`'s encoder is **signed**: an Oklab latent's
`a` and `b` axes run either side of zero, and the non-negative encoder it grew
from would have folded every negative one onto `+0` — no error, no NaN, just
every imported photograph pulled towards green.

### 23.4 What the frontend contributes: the decoder, and where to put it

**The browser is the decoder.** Every format it can display can be placed — JPEG,
PNG, WebP, AVIF, GIF — through the same `ImageBitmap` → canvas → `getImageData`
route custom brush shapes already take (§6.6). Shipping a decoder per format
would be a second, smaller answer to a question the platform answers completely,
and it would make "which formats can be imported" a fact about this build rather
than about the machine it is running on. What crosses the engine boundary is
pixels.

`getImageData` is specified as un-premultiplied sRGB, which is exactly the form
`stark_assetid::Picture` is defined in, so nothing has to be undone on either
side. The downscale to `MAX_PICTURE_DIM` happens here too, and it is an
optimization rather than the rule — `stark_assetid::picture` caps whatever it is
handed, so the id is the same either way. It is worth doing because the browser is
the only thing in the chain that can resample without first materializing the
full-size buffer, and a 48-megapixel phone photograph is 190 MB of RGBA before
anything has looked at it.

The frontend then takes the same two steps a custom brush shape does: import for
the id (`Engine::import_picture`), register the bytes with the session so peers can
fetch them (`add_content`) — **before** the commit that names them, which is the
ordering that leaves receivers able to fetch what an action needs — and only then
dispatch the placement.

**Pasting is the `paste` event, not `navigator.clipboard.read()`.** The event is
delivered inside the user's own gesture and needs no permission, where the async
read prompts in Chrome and does not carry images in every engine — and a paste
the page never sees is a feature that works for some people. It stands aside for
a paste into a **text field** (a layer being renamed, a session name), which is
the same question the keyboard shortcuts ask of an event's target, and it calls
`preventDefault` only once it has found an image to place, so an ordinary text
paste still reaches whatever would have handled it. Only the first image on the
clipboard is taken: a clipboard carrying a picture usually carries it several
times over, and taking every entry would place the same photograph twice.

**Where it lands is the frontend's arithmetic**, because the view is the
frontend's (§18.1.2): centred on what is being looked at, which is the only
placement that needs no explanation — an image arrives where the eye already is
— rounded to whole canvas pixels, and above the active layer in that layer's own
stack, so placing one while inside a group keeps it in the group (§14.8). The
engine is told a position, not a policy.

### 23.5 What is deliberately not here

- **No placement gesture.** An image does not arrive under a transform box with
  handles. It arrives at 1:1 where you were looking, and the transform tool
  (§16.6) is right there with an ellipse that already does move, scale, rotate
  and skew — on a whole layer when nothing is selected. A second, weaker
  transform widget that exists only during an import would be a worse version of
  a tool the application already has.
- **No import into the selection.** A fill and a stroke are gated by the author's
  mask (§6.8); a placement is not, because it mints its own layer and there is
  nothing beneath it for a mask to protect. "Paste into" is a different feature —
  a placement onto an *existing* layer — and it would want the gate, a base to
  stack onto and therefore a real GPU pass. That is a coherent thing to build and
  it is not this one.
- **No linked or re-linkable images.** The log names a picture by the hash of its
  *content*, and the bundle carries it; a document that referenced a file on disk
  would be one that stops being replayable the moment that file moves, which is the
  property §8 is entirely about. The indirection here is the opposite kind — the
  bytes are always somewhere the document can be made whole from, and
  `unbundled_content` is the bill that says so.
- **No drag-and-drop onto the canvas.** The two doors here cover the two ways a
  picture is usually in hand. A drop is a third that reuses `place_bytes`
  unchanged; it is a frontend addition when someone wants it, not a design
  question.
