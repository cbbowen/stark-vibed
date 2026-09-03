# Compositing, media, and color

The three passes, blend modes, presentation and the canvas substrate, Oklab, pluggable color spaces, and the generated CPU↔shader mirrors — §6.3, §6.4, §6.5, §6.7, §6.10.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.
> One name per thing: [glossary.md](glossary.md).

## 6.3 Compositing and the media pass

Three passes turn tiles into pixels. The first two are the substance; the third
is chrome.

**A — composite.** Every visible tile of every visible layer is drawn, bottom to
top, into two viewport-sized offscreen targets: color (premultiplied "over", in
the working color space) and the `(height)` aux (additive). Layer opacity rides
on the instance.

A layer's "over" weight is its **visible alpha** — per-unit opacity and amount
combined by the slab law `1 − exp(−K·opacity·height)`, the same law
`paint_common.wesl` uses to stack parcels *within* a layer — so a layer covers
the stack below exactly as much as it shows. (Weighting by opacity alone was the
old defect: a film with opacity 1 and no thickness — every soft brush's fringe —
drew as nothing over bare canvas yet replaced the color over another layer's
paint.) Because the slab is multiplicative in optical mass, "over" on these
weights accumulates the *stack's* coverage in the target's alpha, and the media
pass reads it there instead of re-deriving it. `tests/composite.rs` guards it.

One consumer must *not* see that weighting: the dynamics loop composites base
tiles into its working region with this same shader, and that region holds the
tile representation itself (per-unit opacity in alpha, §6.1) — the exchange
loop's pickup reads it and the slice writes it back to persistent tiles. Running
the slab law there stores coverage as opacity, corrupting smeared paint
differently on either side of a piece or freeze cut — which is precisely how an
earlier attempt at this fix made smear previews drift from their commits. Screen
path and region path are therefore separate fragment entry points (`fs_main` /
`fs_raw`, `composite.wesl`).

Layers that are not plain `Normal` cannot go through that, because their mode is
defined against *what is underneath*. So pass A is a tree of **blend groups**
(`CompositeGroup`, §14.7): a run of consecutive `Normal`, unclipped, opaque
layers is one `Run` drawing straight into the accumulator — a document that uses
no modes, no clipping and no groups is a single `Run` and costs exactly what the
flat tile list always did — while anything needing isolation composites alone
into an isolation target and is merged by a fullscreen blend pass. That pass
reads the accumulator and writes the merge, so it needs somewhere else to write;
rather than copy back, the accumulator ping-pongs between the caller's target
pair and a scratch pair, and the *starting* side is chosen by the parity of the
blend count so the final result always lands where the caller asked. The media
pass therefore keeps one bind group and the eyedropper keeps its own targets.
Scratch pairs are allocated on first use, so an ordinary painting never pays.

**B — media / lighting.** One fullscreen pass turns those two buffers into the
painterly result, and it is where the "old masters" look lives:

- **Normals from height.** The gradient of the height field — impasto thickness
  plus the canvas substrate scaled by `substrate_strength` — tilted by
  `height_strength`, so ridges catch the light.
- **Image-based lighting.** The scene is lit by an `Environment`: an HDR decoded
  to a linear-RGB equirectangular texture with a full mip chain. Diffuse
  irradiance samples a very blurred mip in the *normal* direction; the specular
  samples a gloss-selected mip in the *view-reflection* direction, so paint picks
  up the environment's highlights. Two environments ship: `Neutral`, generated
  procedurally (an achromatic dome under a soft overhead key — relief still
  reads, nothing is tinted), and `Ferndale`, the bundled studio HDR. They differ
  only in the equirect image handed to the same prefilter, so there is one
  lighting path, not two: a reference light you switch to, and a room you paint
  in. **Exposure is a property of the environment**, not a knob beside it
  (`EnvironmentId::exposure`): `Neutral` is 1.0 and `Ferndale` 0.65, and
  switching lights carries the value along.
- **Paint gloss.** `specular` sets how smooth the paint film is, driving a
  Cook–Torrance term. It is a **uniform property of paint**, not a stored
  channel: the roughness ramp is the paint's own *visible alpha*, so paint is
  equally glossy everywhere it is, a thin glaze reads nearly as matte as the
  substrate it barely covers, and bare canvas stays rough, so matte. (There was once
  a per-texel `wet` channel here; nothing could source it after
  `BrushParams::wetness` was removed, so it was a stored zero every pass carried.)
- **Present.** The working channels are converted to the surface's display space
  and composited over the substrate color. This is the *only* place
  gamma-encoded color exists.

**The reference invariant.** Under `Neutral` (exposure 1.0), with
`height_strength = 0`, the media pass is an identity — paint comes back out its
own color, within about two bytes. That is what makes the neutral environment
worth having: it is the light you switch to in order to *judge* a color rather
than enjoy it. Three things have to hold, each a constraint on the model rather
than a correction bolted on:

- **Exposure is divided by the irradiance a flat canvas actually receives** — the
  diffuse mip sampled dead ahead, computed on the CPU from the same mip chain the
  shader reads. The whole-image mean luminance it replaced only approximated
  that: averaging equirect texels over-weights the poles and counts light no
  front-facing canvas ever sees, leaving flat paint ~13% dark.
- **The diffuse keeps `1 - spec_energy`, not `1 - fresnel`.** The split-sum's
  `env_brdf` already integrates Fresnel, so subtracting a second Schlick term was
  double-counting it and losing ~2.4% of every color.
- **The tonemap is a reference curve, not a look.** Khronos "PBR Neutral", with
  its black point set to the sheen this fragment's BRDF actually contributed
  instead of an assumed F0 = 0.04, and its highlight knee at 1.0 instead of 0.8 so
  nothing representable is reshaped on the way to the display.

Exactness in `[0,1]` and a filmic shoulder are not both available: a curve that
is the identity up to 1 has nowhere to bend. The shoulder is what was given up,
and `exposure` buys the headroom back — which is why it belongs to the *light*.
Dividing by `flat_irradiance` already makes 1.0 mean the same thing everywhere,
but that is a statement about the diffuse response, not the peaks: a room with
bright windows puts saturated paint over 1.0 long before a smooth grey dome does.
So `Neutral` stays at 1.0. `tests/reference.rs` pins the invariant.

**The blend modes** are deliberately not Photoshop's: each is ordinary **addition
of light, conjugated by a tone curve** — `f(a,b) = T(T⁻¹(a) + T⁻¹(b))` —
evaluated in **CIE XYZ normalized to the display white**, the only space in play
that is linear in light, non-negative for every real color, and free of an
opinion about the display's primaries.

| Mode | `T(x)` | `f(a,b)` | Identity | Character |
|---|---|---|---|---|
| `Glow` | Reinhard `x/(1+x)` | `(a + b − 2ab)/(1 − ab)` | black | asymptotic — **cannot** blow out; stack a hundred and it approaches white without clipping. Glazes, mist, rim light. |
| `Radiance` | Drago `k·log(1 + x/k)` | `k·log(e^{a/k} + e^{b/k} − 1)` | black | no asymptote, so it *does* push past display white — deliberately, into pass B's highlight roll-off as a bloom with a filmic shoulder instead of a clip. |
| `Multiply` | `e^{-x}` | `ab` | **white** (black annihilates) | the conjugation collapses to plain `ab` and the added quantity is **optical density** — Beer-Lambert, what stacked glazes do. |

**`Radiance`'s `k` is the layer's own** — the first per-mode parameter, and the
reason `BlendMode` is a data-carrying enum rather than a bare one. It lives on the
variant, not in a settings struct beside the mode, so there is no `k` on a
`Multiply` layer to be edited, saved, replicated and ignored, and no way for a mode
and its settings to disagree about which mode they are. What the knob does is pick
a curve out of a family: large `k` tends to plain addition, so coincident lights
reach the roll-off sooner; small `k` tends to `max`, so the brighter one simply
wins. `DRAGO_K = 0.6` is where a fresh layer starts (two half-lit layers give 0.769
there, against Reinhard's 0.667 and Screen's 0.75) and `DRAGO_K_RANGE` is
`0.125..=4.0`, whose ends are where the mode stops changing — 0.586 and 0.944 on
the same pair, against `max`'s 0.5 and addition's 1.0.

Every setting is still a conjugation of addition, so the guarantees below are not
something a painter can dial away. What *does* follow is that two bends are two
functions: a merge-down of same-mode siblings (§14.11) requires the same curve, not
merely the same name, and `!=` on `BlendMode` already says so. The picker asks
`same_mode` instead, which is the one place the distinction is wanted the other way
round.

What deliberately did **not** ship is Screen. `a + b − ab` is what falls out of
inverting a multiply; it describes no optical process, and it flattens the top of
the range into the chalky white that reads as "digital glow" at a glance. Screen
is multiply conjugated again by `x ↦ 1−x`, and that second step is the one with
no physical referent — hence multiply, not Screen. Multiply in normalized XYZ
rather than RGB also means two saturated glazes cross without the dead channel an
RGB multiply produces when one primary sits near zero.

Being conjugations of `+` makes all three commutative and associative with a
neutral element, so reordering a stack of them is not a color decision. **That
holds at any coverage, and for a long time it did not.** Porter-Duff weighs a
layer by its coverage in the *working space* — the `(1 − αb)·Cs` term — which
applies the blend function to the accumulator's coverage-*averaged* color, and
averaging does not commute with a curve. Reordering three glow layers moved the
canvas by up to **20 levels** at every partly-covered texel, which is most of a
stroke: the slab law leaves an ordinary interior around 45% covered, so this was
not an edge-only effect.

So **every combining mode weighs coverage in the space where its own blend
function is affine** (`blend_common::combined_light`), which is the one sentence
this reduces to. It is the same three-term Porter-Duff expansion in every case;
only the space changes:

- The emissive modes are addition conjugated by a curve, so they are affine in
  **emission** (`T⁻¹` of the light), where the expansion collapses outright —
  `(1−αb)·αs·Ls + αb·αs·(Lb+Ls) + αb·(1−αs)·Lb = αs·Ls + αb·Lb`. Premultiplied
  emission simply adds, so a stack of any depth is one sum.
- `Multiply` is Beer-Lambert, so it is bilinear in **light**, light being a
  transmittance there. The expansion does not collapse and does not need to:
  bilinear is affine in each argument, which is all a weighted average asks for.

Multiply was **not** exempt, and the first attempt at this said it was. Its blend
function is bilinear in *light*, and the working space is Oklab — so mixing
coverage there was as wrong for it as for the others, and by a similar amount:
reordering three multiply layers moved the canvas by 13 levels where the emissive
modes moved 20. `Normal` is genuinely exempt, and for a reason no amount of
algebra changes: it covers rather than combines, so it has no blend function to be
affine anywhere.

Two properties follow, and the second is the stronger. Each mode is associative and
commutative at any coverage. And the result no longer depends on the
coverage-correlation guess, since `E[Σ] = ΣE[]` holds whatever the joint
distribution where `E[B(…)]` for a nonlinear `B` does not. It is also the
physically right rule rather than the convenient one: averaging over a
partly-covered pixel is linear in the quantity that is *there* — radiance for a
lamp, transmitted light for a glaze — and both are the mode's own space.

**What it costs, stated plainly.** The combining modes now disagree with `Normal`
about what partial coverage means, by single-digit levels. `Normal` is occlusion
and this engine mixes occluded paint in the perceptual working space on purpose
(§6.1), so both weightings are right for their own question — but it means "black
is the identity" is no longer *pixel* identity with a `Normal` render at partial
coverage. It remains the identity of the combination, which is what the law says.
`tests/blend.rs` pins all three facts separately: the identity at full coverage
where the conversions are the only thing running, the order-independence over the
whole canvas, and the coverage-law gap itself — asserted rather than accepted,
because it is the one place these modes part company with the rest of the
compositor.

The neutral element is where the family splits and where the tests split with it.
Each color space supplies only its channels ↔ light conversion, which for Mixbox
is the pigment polynomial and its inverse LUT (`mixbox_lut.wesl`) — the one place
the engine inverts Mixbox on the GPU. In a pigment document the emissive path is
also *simpler* than it was: the whole color falls out of one sum, so the latent's
residual is decoded from that answer rather than merged separately. The rest of
the darkening family attaches at the same seam: one more `T`, one more variant, no
new machinery.

One thing is *not* free: a glaze sees the layer stack but not the substrate,
which pass B composites underneath afterwards, so multiply over bare paper leaves
the paper alone. Correct on white paper (its identity), wrong on a toned substrate;
the fix is to make the substrate the bottom of the stack.

**C — selection outline.** One instanced quad per mask tile, drawn over the lit
result in the same canvas→NDC frame as pass A (§6.8). Suppressed on export (§15.6).

`MediaParams` (`height_strength`, `specular`, `substrate_strength`) is a **view
setting** — per-client, never historized, changed by
`ViewCommand::SetMediaParams`. So is the choice of environment: switching it
re-lights the canvas and touches no stored pixel. Exposure is neither: it is not
tunable, it is what the chosen environment says it is. Nothing here is in the
save file.

The whole media model is a single shader stage, which is the point: Kubelka–Munk
pigment mixing, granulation, varnish gloss can be iterated on without touching
the document or tile machinery.

**The draw list is culled to the view.** Pass A is built only from tiles the
render's view can reach — `Engine::visible_tiles`, a `TileRect` over
`ViewTransform::visible_bounds` — so its cost follows the viewport rather than the
document. Without it every populated tile of every visible layer was composited
every frame, each one an `Arc` clone, an instance, a bind group and a draw, and
the off-screen ones were merely clipped by the rasterizer after all of that.

The bound is conservative twice: `visible_bounds` is the AABB of the *rotated*
viewport, so it covers more canvas than is really on screen, and the quantizer
then floors to whole tiles. That direction is the one that cannot crop a picture,
which matters because the same path renders an export. A view it cannot measure —
non-finite, or so far out that tiles leave the `i32` grid — culls nothing.

Skipping a tile is a pure subtraction because a tile outside the view covers no
pixel of a viewport-sized target. The one thing it changes is that a layer can now
be *empty* because its paint is off-screen, and an empty layer is dropped rather
than given a group. That is sound on the blend algebra rather than on coincidence:
`merge` with a transparent source has `cs.a = 0`, so both source terms vanish and
the aux sum adds nothing — the result is the backdrop exactly, for every mode and
both clip states.

> **Not yet: damage tracking.** There is still no per-version damage set: an
> unchanged frame recomposites everything the view can see, rather than only what
> a commit actually touched. Fine at current canvas sizes; the next optimization
> when it stops being.

## 6.4 Presentation (pan/zoom/rotate to a surface)

The engine does **not** own the window surface — the frontend does. The engine
exposes:

```rust
impl Engine {
    pub fn render(&mut self, target: &wgpu::TextureView, view: ViewTransform);
}

pub struct ViewTransform {  // session-owned; never historized, never sent
    pub center: Vec2,       // canvas-space point at viewport center
    pub zoom: f32,
    pub rotation: f32,      // clockwise, radians
    pub flip_h: bool,       // mirrored left-right
    pub viewport: Extent2,  // target size in px
}
```

The canvas→screen linear map is therefore a 2×2 rather than a scale pair
(`ViewTransform::orientation`), stored as an *angle plus a mirror flag* rather
than a free matrix: that keeps it a rigid motion by construction, makes its
transpose its inverse, and leaves nothing that can drift into a skew. Four
shaders consume it — the composite's canvas→NDC, the matte's inverse of the same,
the media pass's screen→canvas for the substrate, and the overlay — and upright and
unmirrored it is diagonal and computes bit-for-bit what the scale pair did, which
keeps the goldens blessed against it valid. Rotation and mirroring are view state
exactly as predicted: never logged, never sent, invisible to replay, absent from
what a file or the navigator's overview shows — those frame the *piece*, not the
easel (§18.1.2).

Moving the view is `Pan`, `Zoom` and `SetRotation` separately, and
`ViewTransform::pinch` — pan, scale and turn about one screen point, in one act —
when they are not separable. Two fingers are the case that needs it: each of the
three anchors against the view it is applied to, so sent in sequence the last two
would be measured against a canvas the hand never saw and the paint would slide
out from under the fingers. `zoom_about` is that same call with the fingers
standing still (§18.1.7).

The frontend owns the `wgpu::Surface`, acquires the frame texture, calls
`render`, and presents.

**Minification: the whole pipeline is supersampled, not the tiles mip-filtered.**
Presentation undersamples in three places, and only the first of them is a
texture-filtering problem:

- **The paint.** `composite.wesl` takes one bilinear tap per fragment from a tile
  texture with no mip chain, so at zoom *z* it reads one texel in every 1/*z*².
- **The substrate.** `media_common.wesl::substrate_height_at` samples a periodic height field
  in canvas space at LOD 0, which moirés as soon as a screen pixel spans more
  than one period.
- **The relief shading.** The normal is a finite difference of the height field
  over *screen* pixels, fed through a specular lobe and a tonemap. Shading is a
  nonlinear function of height, so the correct minified pixel is the average of
  the shading, not the shading of the average — an impasto ridge thinner than a
  screen pixel sparkles under a pan however well the height that made it was
  filtered. **No prefiltered texture can fix this one.**

So a zoomed-out render runs passes A–D at `ss` samples per axis and a box filter
resolves it into the target (`resolve.wesl`, pass E). `ss` is the minification
ratio `⌈1/zoom⌉`, capped by `MAX_SUPERSAMPLE` (4), by a memory budget
(`MAX_SUPERSAMPLED_BYTES`, since *every* offscreen attachment scales with it), and
by the device's texture limit. Bytes rather than pixels deliberately: what an
attachment set costs depends on how many channels the colour space carries and
whether a residual rides along, so at 16 Mpx the real figures spread by a factor of
seven — a spread a megapixel ceiling cannot see. It is **1 at 1:1 and closer**, so magnifying costs
nothing and every golden — all blessed at zoom 1 — is bit-identical.

`ViewTransform::supersampled(n)` scales the viewport and the zoom *together*,
which is what makes this one substitution at the top of `Compositor::render` and
one pass at the bottom rather than a parameter every pass carries: the
canvas→NDC map is unchanged, and anything measured in target px (an outline's
width, a matte's edge fade, a guide's line) comes out `n` times wider in a picture
about to be `n` times smaller — the same width, drawn with `n²` samples of
coverage. Chrome therefore antialiases for free. The average is taken **in light**:
the target carries display-encoded sRGB in a non-sRGB format (§6.5), so the
resolve decodes, averages alpha-weighted, and re-encodes.

Mip-chaining the tiles was the alternative, and it is the more invasive of the two
for less of the problem. A `TILE_TEX`-square texture with a 1-px apron around a
254-px interior has no mip level at which the interior sub-rect still lands on
texel centres, so every level past 0 needs its own apron refresh or the seam
invariant below breaks; the chain has to be regenerated on every tile write; and
it would still leave the relief sparkling. `tests/minify.rs` pins what was bought
instead: a 1:4 render is the 1:1 render boxed down by four, and is *not* one texel
in sixteen of it.

**Tile aprons (seamless boundaries).** Tiles are *separate* GPU textures, so the
compositor samples each independently. The moment sampling is not pixel-exact —
any sub-pixel pan or non-1:1 zoom — a bilinear tap at a tile's edge clamps to
that tile's own edge texel instead of reaching into the neighbour, because the
neighbour lives in a different texture. That leaves a discontinuity at every tile
boundary, which the media pass then *amplifies*, since the normal is the gradient
of the height field and a step in height becomes a bright ridge.

The fix is an **apron**: each tile texture is `TILE_TEX = TILE_SIZE + 2·TILE_APRON`
px square, carrying a halo of neighbouring canvas content around its interior.

- **The apron is rendered, not copied.** The stamp pass maps the *whole*
  `TILE_TEX` target to NDC (texture origin = interior origin − apron) and a tile
  is selected for (re)drawing whenever a stroke touches its apron-extended bounds
  (`affected_tiles` inflates by `radius + TILE_APRON`). Because stamping at a
  canvas position is a deterministic function of that position, a tile's apron is
  *bit-identical* to the neighbour's interior over their overlap — no copy pass,
  no sync bookkeeping, and it composes correctly through CoW history. Every pass
  that writes tiles (stroke, dynamics write-back, fill, transform) is a pure
  function of canvas position for exactly this reason.
- **Only the interior is presented.** The compositor quads cover exactly the
  interior (tiles tile the plane with no overlap); they sample the interior
  sub-rect via `uv = corner·(TILE_SIZE/TILE_TEX) + APRON/TILE_TEX`, with the
  filter free to read into the apron at the edges.
- **Configurable width.** `TILE_APRON` (1 px — all bilinear needs) is a single
  constant. At 256² interior a 1-px apron is ~1.6% more texels.

Alternatives rejected: *composite-then-scale* makes zooming far out balloon that
buffer with the visible tile count; a *padded tile atlas* is heavier machinery
than the problem warrants. The translation invariance the apron restores is
locked by `tests/seam.rs`: a stroke across the 4-tile corner must render
identically to the same stroke shifted half a tile into one tile's interior.

**The canvas substrate.** Paint sits on a physical substrate — a tileable height map
(`gpu/substrate/`), an `Rgba8Unorm` texture sampled in *canvas* space (so the
grain is fixed to the canvas and pans/zooms with it). It is read twice: the
deposition tooth gates what a brush lays through it (below), and it feeds the
normal everywhere (`height_at` = impasto + `substrate_strength·(h−½)`), so the grain
catches light across the whole viewport — including bare canvas, whose
shading is *normalized* so a flat substrate leaves it unchanged. `substrate_strength`
is a view setting; it does not touch stored pixels.

The substrate is **document state**: which canvas a piece was painted on is part of
what the document *is*, it is saved, and reopening on a different substrate would be
a different painting. `CanvasMeta` records the substrate the log *starts* from; a
mid-document switch is a logged `ActionKind::SetSubstrate`, so it undoes, replays
and replicates like any other edit. The deposition tooth reads it, so a switch
changes the strokes made after it — logging it is what made that a rendering
change rather than a history one.

**A substrate is named by its image, not by a label.** `SubstrateId` is
`Flat | Image(AssetId)`: one procedural substrate that needs no bytes, and
everything else identified by the BLAKE3 hash of its canonical decoded height
field — the same bargain brush shapes make (§6.6), and for a sharper reason. A
label is only as good as the table the reader holds. When substrates were
`{ Flat, Linen, Rough }`, a peer who received `SetSubstrate(Rough)` without ever
having fetched it fell back to the flat stand-in *silently*, and from then on
deposited every stroke with no tooth at all; the two canvases diverged and
neither screen could say why. Unlike the media pass — which re-reads the substrate
each frame and rights itself the moment an image lands — a deposit is **stored**,
so nothing un-bakes it. A content id removes the failure rather than reporting
it: the holder either has those exact bytes or knows precisely what to ask a peer
for, and what comes back is verified against the id that asked (`accept_substrate`
refuses a mismatch). The same mechanism carries a substrate a *user* brings, which a
closed enum could never have named.

Three consequences follow. A save file **bundles every substrate its log names**,
not just the one it ends on (§8) — a height map is a replay input exactly as a
coverage mask is, and a document that switched part-way needs both. The engine
**embeds no image bytes**: substrates are fetched at runtime and handed to
`import_substrate`, which decodes, downsamples by an integer factor to fit the 2048
limit (preserving tileability), hashes, and returns the id — so an id is only
knowable once the bytes are in hand. And `DEFAULT_SUBSTRATE` is `Flat`, because
core naming linen would be core naming an image it cannot produce; the frontend
holds a catalog (`stark-dioxus-frontend/src/substrates.rs`, the analogue of
`builtins.rs` for shapes) and opens a fresh document on a substrate once its map
has landed.

**And a user brings one the same way.** This is the payoff of the paragraph above
rather than a feature beside it: because a substrate *is* its bytes, "built-in" is a
fact about the frontend's asset list and about nothing downstream, so an imported
substrate is saved, bundled, replayed and fetched by a peer through the machinery
`Linen` already uses. There is no second path — only a second place the bytes come
from. The frontend keeps the library the way it keeps imported brush stamps
(§25.6): a name and an id in `localStorage`, the height map in the blob store
beside it, and the parts both libraries share in
`stark-dioxus-frontend/src/library.rs`. The one real difference between
importing a stamp and importing a substrate is polarity. A stamp's is a
*spelling* — white paints, and a scan of ink on paper means the opposite — so
the import inverts a dark-on-light image; a substrate's polarity **is the
substrate**, so nothing inverts it, and the import's whole job is to make the
image grey (`normalize_substrate_image`).

**How large the substrate is laid** is document state too:
`SubstrateScale`, a percentage of the map's natural size, moved by a logged
`ActionKind::SetSubstrateScale`. It is not a view setting for the reason the substrate
is not — the tooth's reach is a distance in *canvas px* (below), so laying the
substrate larger means a tip crosses fewer threads per px, bridges further and rides
fewer faces. It decides what a stroke deposits, and a deposit is stored.

Two consequences worth naming. First, the renderer's substrate is built from the
**pair** — `gpu::substrate::Substrate { id, scale }` is the registry's key, while the
*bytes* stay keyed by the `SubstrateId` alone, so one height map bakes a map per
scale it is laid at. Scaling one bake's lookup instead would report the rise over
the old span under a new name, which is the compensating fudge §1 rules out.
Second, the scale is a **quantized integer** (5% steps, 25–400%) rather than an
`f32`: it has to be `Eq`/`Hash` to be that key, it has to be exact for two peers
to bake the same substrate, and each distinct value a document names is a texture
held for as long as the log can be replayed across it. The ladder is what keeps a
slider dragged end to end from naming three hundred of them.

That last cost is also why the slider *previews*. A scale is shown per pointer
sample and committed once on release (§25.4's bargain,
`stark-dioxus-frontend/src/preview.rs`) — and here the preview is genuinely
free, because the map's **height** channel is the same field however large it is
laid. All of a bake's scale-dependence is in its rise channels, which only the
deposit reads. So the light follows the slider instantly while the tooth waits
for the commit, and nothing filters a whole image under the hand.

The bundled substrates: linen, a regular woven grid; and rough, a brushed acrylic
substrate, irregular, whose height histogram is a broad spread rather than a
periodic peak — which is what makes it the interesting one for the tooth below,
since a periodic substrate prints a periodic mark and reads as a screen. `Flat` is a
1×1 *zero-height* texel — a constant height has zero gradient, so it is *exactly*
equivalent to having no substrate. That orthogonality is deliberate: most goldens
run on `Flat` to test other features in isolation, and a dedicated golden
(`linen_substrate`) exercises the substrate. One bump tile spans `SUBSTRATE_TILE_PX`
canvas px at natural scale, and `SubstrateScale` multiplies it.

**The deposition tooth.** Paint lands where the tip touches the substrate — and
what a *dragged* tip touches is not a level set of the substrate's height. A stamp
pressed straight down contacts the summits; a stroke is dragged, and a dragged
tip has **give**: it sinks after substrate that falls away beneath it and is
pressed up by substrate that rises to meet it, so it bears on the near face of
every bump and bridges the lee side behind it. That is why a dry brush prints
the leading edges of the grain rather than a speckle of its high points, and it
is the difference between a mark that looks brushed and one that looks screened
— a height threshold reads the same mark whichever way the stroke runs, which no
brushed mark does. So the field the gate thresholds is the **rise ahead**: the
height's derivative along the tip's own travel, taken across the contact's reach,

```
d(x, d̂) = ahead(x)·d̂
```

where `d̂` is the tip's travel *at that texel* and `ahead` is the rise the substrate
makes over one `TOOTH_REACH` (3 canvas px) along each axis.

**The tooth is two knobs, and they are different questions.** Where the
threshold sits, and how wide the band around it is.

`ToothParams::give` is the give itself — the gate thresholds the rise
against the steepest fall the tip can still follow, and the knob walks that
limit through three stations (`tooth_level`, a `2 − 1/(1 − give)` map): at 1 the
give is infinite, the tip tracks any fall and the substrate is ignored, exactly
— the solid default; at ½ there is no give left going down, so the tip holds its
level — it touches whatever is flat or rising and bridges every fall, which on
the bundled substrates is almost exactly half the substrate (`SubstrateMap::bearing`
measures 0.50 on rough, 0.51 on linen); at 0 the tip *demands* substrate rising
at the contact scale (`TOOTH_RISE`, ~the substrates' own mean |rise|) before it
presses, and only the leading faces print — 13% of rough, 25% of linen: the dry
mark, still a mark.

**The knob is the give and not the bite, and that is a fact about the pen rather
than about the substrate.** A `Modulation` only ever scales a parameter *down*
(§6.2), so which way this knob runs decides which way pressure reads. Quoted as
the depth of the bite, a pressure mapping made a light touch solid and a hard
press dry — backwards for the one mapping the axis exists for. Quoted as the
give, `BrushModulations::tooth_give` on pressure *is* the charcoal: barely touching
the paper the tip has no give and prints the peaks alone, borne down it presses
past the falls it was bridging and the grain fills in. The cost is that the
slider's interesting end is its left one, and the mapping is worth more than the
habit.

`ToothParams::softness` is the width of the transition around that level,
in the rise's own units, and it is what the tip is *made of* rather than how hard
it is pressed. A hard threshold is a binary indicator per texel: correct in the
mean, and at canvas resolution it aliases into speckle that reads as dither — so
the default (0.06, the bundled substrates' interquartile rise) spans the grain's
natural variation and the mark reads as a level set of it, which is paint sitting
on the substrate. A charcoal or a soft graphite does not sit on it: the stick
crumbles into the valleys instead of spanning them, its contact comes on over
several times that width, and the grain reads as a tone across the mark rather
than a pattern in it. Widening the band moves the bearing towards a half from
whichever side it started — it takes from the faces that bore fully and gives to
the ones that were bridged entirely — which
`a_softer_contact_bears_on_the_substrate_more_evenly` measures. Past about `2·RISE_LIMIT` every rise the map can encode
is inside the band and the gate is a flat scale on the deposit, which is where
the frontend's slider stops. A cubic smoothstep, for the reason `taper_profile`
is a polynomial.

The *give* is a modulation target and the *softness* is not, and that split is
the model: pressure changes how hard a tip is pressed, but no amount of it makes
a charcoal stick out of graphite. So the give rides the segment instance
(`SegmentInstance::tooth_give`, `Stamp::tooth_give`) and the softness rides the
stroke (`TileXform::paint.z`, and `Stamp::tooth_softness` because that loop's
uniform is per slot). The softness default lives on the brush and nowhere else —
the shader declares no constant for it, only the inert floor that keeps a
softness of zero from being a division by zero. Both lanes are **neutral at 1**
rather than at zero: `tooth_give = 1` is the short-circuit every bleed slot and
every untoothed brush takes, and a zeroed lane would be the driest tip there is.

Three things follow from writing `ahead` as a difference across a distance
rather than as a gain on a pointwise slope:

- **`ahead` is a difference across the reach, not `reach·∇s`.** A gradient at a
  point knows nothing about the distance it is being multiplied out to: it grows
  without bound in the reach and reports whatever the map's finest scale is
  doing, which on a nearest-sampled, ~2:1-minified height map is largely Nyquist
  noise — a dither that flips with the stroke instead of a face to catch on. A
  difference over a span is self-limiting (it saturates once the reach clears a
  feature's width — measured, 0.038 → 0.056 → 0.069 → 0.078 on rough at 1.5, 2,
  3, 4 px) and inherently blind to anything repeating faster than the span. The
  reach is set on the shoulder of that curve — past a feature's own width there
  is no more face to climb, and a longer reach only translates the mark. Only the
  sampling grid is left to answer for, and a half-px blur does that.
- **The reach is a distance in canvas px**, so the same substrate reads
  identically however finely it was stored — the span in texels follows the map's
  resolution, which is what keeps the integer downsample invisible to the mark.
- **It is baked into the substrate texture**, which is `Rgba8Unorm`: height in `R`
  (the media pass's relief), the two rise components in `GB`, each byte spanning
  ±`RISE_LIMIT` — a quarter of the height range, because a filtered difference
  across a few px *is* small (the substrates' 99th percentile is under 0.26) and
  spending the byte where the rises live is what keeps the gate's transition
  tonal rather than stepped. The deposit costs *one* texture tap, so the whole
  axis adds a dot product and nothing else to either render path — and, more to
  the point, the byte a texel is gated by is the byte the CPU tabulated, which
  is what lets the tool book against the map's exact distribution.

`d̂` is the tip's tangent carried round its own arc (`sweep_at`), not the
segment's start tangent, so a curve's tooth does not depend on where the
flattener cut it. `tooth_give = 1` still gates at `1.0` to the bit however steep the
substrate — which is what every golden in the suite paints at — twice over: the
shaders guard it before the map is read, and the follow limit dives past any
encodable fall well before the knob reaches 1, so a pen mapping sweeping
through it meets the guard continuously.

Three decisions do most of the work, and each is about *where* it is applied
rather than what it computes:

- **The grain is the canvas's, not the brush's.** A pencil and a loaded brush on
  one substrate see one grain; the brush says only how much give it meets it with.
  That is why `SubstrateId` is where the texture lives and the tooth's two knobs
  are the only thing on `BrushParams`. Painter and Procreate put the grain on the brush, which
  is why switching brushes there changes the paper under a half-finished
  painting.
- **It scales the exposure, not the transfer**, and that one choice is what makes
  it both compose and conserve. Every rate here is a function of swept optical
  depth τ — additive in it, or `1 − exp(−k·τ)` — so multiplying τ itself by a
  per-texel `g` leaves every one of them in its form: `exp(−k·g·τ₁)·exp(−k·g·τ₂) =
  exp(−k·g·(τ₁+τ₂))`, and `Σᵢ g·τᵢ = g·Στᵢ`, because `g` belongs to the canvas and
  not to the segment. Applied to the finished shares instead, a toothed lift would
  fade at a rate that depended on where the flattener cut. It is read at the
  fragment's own canvas position — the travel it is read *along* is recovered from
  the segment's frame at that same position — so tile aprons stay bit-consistent
  with no copy pass, the gate still factors out of the sum over a segment's
  overlapping quads, and — the reason the mark reads as paper rather than as noise
  — successive strokes the same way over the same substrate catch on the same faces
  and register with each other. A stroke run the *other* way deliberately does not:
  that is the physics, and `tests/tooth.rs` measures it — the two runs lay their
  ink on opposite sides of the grain.
- **It gates height, never the per-unit opacity** (§6.1). The tooth decides how
  much paint arrives, not what the pigment is. Both render paths call the same
  `tooth_gate` in `paint_common.wesl`, so nudging `lift` off zero cannot change
  what the substrate does.
- **The tool books against the substrate's mean, and that is what conserves paint.**
  A transfer has two halves in two dispatches — the canvas gives up `1 − keep` and
  takes `dep` of the tool's load, the tool takes and gives the exact complements
  (`dynamics.wesl::Exchange`). Gate one half and not the other and the books stop
  closing. Scaling the *exposure* keeps them complementary, because `exchange_at`
  is solved once at `g·e` and its four shares still sum to what went in. But a
  reservoir cell has no substrate of its own — it is dragged over fresh canvas at
  every sub-step — so where a canvas texel scales by the substrate beneath it, the
  cell scales by the **bearing fraction**: the mean of the gate over the whole
  map, which `SubstrateMap::bearing` computes from the map's own distribution. The
  two agree in expectation over any extent spanning many grain features, which
  is every usable tip, and the residual is the same order as the mean-field freeze
  either side of the kernel already carries.
- **…and the mean is a curve in two variables.** The rise is directional, so the
  distribution the tool books against depends on which way it is going: a tip
  crossing a substrate along the warp meets a different population of faces than one
  crossing it diagonally, and one running the stroke backwards meets the mirrored
  field outright. So `bearing_hist` is a table — one 256-bin row per each of 16
  directions, built by one pass over the map apiece and read with linear
  interpolation between neighbouring rows, so the tool's booking does not step as
  the pen turns. A segment books at its **midpoint** tangent, the same
  second-order choice its lift already samples the canvas at. Booking every
  direction against a single mean would leak paint at exactly the rate the
  direction matters.

That last point is why the substrate is read with **nearest** rather than bilinear.
Filtering would average away the faces the tooth exists to catch on, and — worse —
it would draw from a narrower distribution than the one the CPU integrates, so the
two halves of the transfer would disagree systematically. It also sidesteps the
reduced-precision filter weights `prefix_slice` documents, so the tap is
bit-reproducible. The histogram's bins are the byte lattice of the encoding
itself, so a projection that hovers a rounding error either side of flat — every
texel of a substrate crossed at right angles — bins identically from both directions
instead of straddling an edge. What is *not* exact is the direction: the row grid
quantizes it and the bins quantize the diagonal projections, both far under the
mean-field freeze the loop already carries.

The gate stops at deposition, which is where `add` and `deposit` put paint on the
canvas. **`bleed` is never gated by the substrate**, and that is not an omission:
bleed is wet paint spreading sideways *on* the canvas rather than a tip dragged
over it, so there is no travel to read the substrate's rise along. Structurally,
bleed slots carry `tooth_give = 1` (`bleed_fires`), which short-circuits the gate to
exactly `1.0` before any of this is consulted — which also keeps the lateral flux
antisymmetric, since the two threads of a pair stand over different substrate and
would otherwise disagree about their shared edge.

Orthogonality is structural rather than checked. `SubstrateMap::relief` is 0 on `Flat`
and on any substrate whose bytes have not arrived, which zeroes the uv scale the
shaders gate on before the brush's number is consulted — so every golden that
paints on `Flat` is untouched by the axis existing, exactly as it is by the media
pass's substrate.

Because deposition reads the substrate, **which** substrate is a question about the
action being applied rather than about what the compositor is showing: the
registry lives on `ApplyCtx`, and `CommitStroke` asks it for `DocState::substrate`
*as the log stood at that action*. A switch part-way through a document changes
the strokes after it and none before, on replay and on a peer alike. That is what
logging `SetSubstrate` bought, long before there was anything to spend it on — the
note this paragraph replaced said as much. It also means the substrate a document
names has to be registered before its log is replayed, or its strokes bake with
no tooth and stay that way; the frontend fetches and re-replays when it finds it
has opened a file whose substrate had not arrived.

> **Still open: the tooth does not fill.** The gate reads the substrate's rise
> alone, so overpainting never uses the substrate up — the same grain prints through
> every layer, which is the tell that gives a static grain multiply away. The fix
> is to read the rise of the *effective* relief, `max(R·s(x), paint_height(x))`
> — the media pass's own `height_at` — so a face buried in paint stops presenting
> a face. It needs two things this version does not have: a relief scale `R` in
> paint-height units (today `substrate_strength` is a *view* setting, and
> deposition cannot read one without making stored pixels depend on the
> viewport), and an answer to associativity — the commit renders a whole stroke
> in one range against the committed base while the live path renders
> head-then-tail, so a gate reading the evolving base breaks `preview ==
> committed` wherever a stroke crosses itself. The way out is to make the deposit
> a **flow**, `dh/dτ = add·g(h, x)`, which composes exactly under any subdivision
> of τ by construction; the stamp loop can integrate that (it is sequential and
> already solves this shape for lift/deposit) and the order-independent swept path
> cannot. Meanwhile `bleed` is the wet-paint valley-filler that already exists:
> charcoal is tooth with no bleed and stays broken, oil is tooth with bleed and
> levels out.

## 6.5 Color management (Oklab)

Color flows through exactly three representations, and conversions live in one
module (`color.rs`, with matching WESL helpers):

```
input (sRGB picker / image) ──→ Oklab  (on ingest: BrushParams, imported tiles)
        Oklab  ←──────────────── all storage, mixing, compositing, history
Oklab ──→ display (sRGB/Rec.2020) (only in the media pass's final blit)
```

- **Why Oklab end-to-end:** pigment mixing, gradient interpolation and wet blends
  are perceptually uniform — no muddy mid-tones from sRGB lerps, no hue shifts
  through grey. This is the math behind the "old masters" blending goal.
- **Determinism:** the sRGB↔Oklab matrices and transfer functions are fixed
  constants shared by Rust and WESL, so ingest and present are reproducible
  across runs and peers — required by goldens (§9) and convergence (§12).
- **Extensibility:** `CanvasMeta.color_space` records the working space so a
  future wide-gamut or spectral pipeline is a new variant, not a rewrite; the
  display transform is chosen from the surface format at present time.
- **Dither at the encode, and only there.** The walk from f32 down to the
  target's 8-bit codes — the media pass's store, and the resolve's re-encode
  when the view is supersampled (§6.4) — is the one place banding is born, so
  each of those two stores adds ±half a code of uniform noise before it rounds
  (`lib/noise.wesl::dither2`, hashed from the integer target pixel and a fixed
  per-pass seed). A value between two codes then renders as their mixture in
  exact proportion instead of banding, while a value the encode lands exactly
  on a code — black, white, anything the tonemap clamped — renders that code at
  every pixel: that is why the swing is half a code (uniform) rather than the
  audio-style full-code triangle, and why the helper keeps it strictly inside
  the half, out of reach of the rounding tie. One lane is shared across RGB so
  the noise is luma-only; a second covers alpha for cut-out exports (§15.6).
  The amplitude comes from the target format (`media::dither_step` — 0 for a
  target deep enough not to band), gated by `MediaParams::dither`: a consumer
  measuring *paint* through the render — the suite's reference-identity
  configurations — turns it off by the same argument that flattens the light
  there. The noise is a pure function of target pixel, so renders stay
  reproducible (§9), any two renders at the same
  viewport cancel exactly in a comparison, and nothing upstream ever sees it:
  tiles, the accumulators, and everything the CRDT (§12) or the eyedropper
  (§18.0.2) reads stay noise-free. Paint math is never dithered — the dynamics
  loop re-reads its own stores, and noise baked into a tile would compound
  there and become part of the document.


## 6.7 Pluggable color spaces (Oklab & Mixbox pigment mixing)

Tile channels are **color-space-agnostic**: tools deposit values and only assume
they *blend linearly*, never what color they represent. The meaning — and the
translation to screen — lives behind a trait:

```rust
pub trait ColorSpace {
    fn id(&self) -> ColorSpaceId;            // serialized in CanvasMeta (§8)
    // Tile layout: each space picks its channel textures and how dabs combine.
    fn color_format(&self) -> wgpu::TextureFormat;
    fn aux_format(&self) -> wgpu::TextureFormat;
    fn color_blend(&self) -> wgpu::BlendState;
    fn aux_blend(&self) -> wgpu::BlendState;
    // Picker / export: straight display RGB ↔ the space's channels.
    fn rgb_to_channels(&self, rgb: [f32; 3]) -> [f32; 4];
    fn channels_to_rgb(&self, ch: [f32; 4]) -> [f32; 3];
    // GPU: how a dab writes its channels, and how channels become display color.
    fn stamp_shader(&self) -> &'static str;  // MRT deposit (§6.2)
    fn media_shader(&self) -> &'static str;  // media/lighting + present (§6.3)
}
```

A document has one color space (`CanvasMeta.color_space`), so tile format, blend
state and shaders are fixed per document and chosen at engine construction. Pass
A is generic; only the **stamp** and **media** shaders, the formats and the
blends are space-specific.

**`OkLabColorSpace`** — `color = Rgba16Float` holding premultiplied
`(L, a, b, coverage)`, `aux = R16Float (height)`, premultiplied-"over" color
blend (coverage *is* the blend alpha), additive aux.

**`MixboxColorSpace`** — realistic pigment mixing via **Mixbox** (Secret
Weapons), where blue + yellow makes green like real paint rather than the muddy
grey of an RGB blend. Mixbox represents a color as a *latent* of pigment
concentrations `c0..c3` plus a small residual, and mixes by **linear interpolation
in latent space**, then maps latent → RGB through a trained polynomial. The
decisive fit: *latents blend linearly*, so the ordinary premultiplied-"over"
deposit **already performs Mixbox mixing** — no programmable blend, no extra
pass. The tile layout is **identical to Oklab**: `color = Rgba16Float` holding
premultiplied `(c0, c1, c2, coverage)` and `aux = R16Float (height)`, **plus a third
`resid` texture** (below). The stamp law is reused; the **media shader differs** — it
un-premultiplies the concentrations, evaluates Mixbox's polynomial (`c3 = 1 −
(c0+c1+c2)` derived), adds the residual back, and hands the base color to the shared
impasto lighting.

Mixbox's latent **residual is carried in a third tile texture** —
`resid = Rgba16Float` holding premultiplied `(r0, r1, r2, opacity)`, over-blended by
the color's own rule. It is not optional and never was:

- Four trained pigments do not span sRGB, so the polynomial alone reaches neither
  black — whose concentrations render `#383838`, a mid-grey — nor the saturated
  corners. Over the whole 64³ LUT the residual's mean magnitude is **0.049** and its
  max **0.389**; it is ≈0 only near white. *(This paragraph previously claimed "the
  residual ≈ 0 in gamut", which is what made dropping it look free.)*
- It is **not recoverable from the concentrations**. `rgb → c` is many-to-one: the
  262144 cube samples collapse to 154391 distinct quantized triples, up to 70 colors
  to a bucket, spanning 0.38 of the cube. A linear-in-`c` correction — the only form
  that commutes with mixing — leaves the mean at 0.048; a degree-8 polynomial in `c`
  leaves 0.027 with black at `#2A281A`. No function of the stored channels could have
  stood in for the stored value.

A residual is not a second quantity to conserve but **the rest of the same color**,
so every pass does to it exactly what it already does to the latent: the same
premultiply, the same "over", the same parcel law on the same masses, the same
bilinear taps. The blend pass gains from it too — its trip back from light now stores
`srgb − poly(blended)` and is exact, where it used to pull every blend result toward
the polynomial's reachable set (`tests/blend.rs` carried an 8/255 and a 40/255
tolerance for that; both are 2/255 now, the same as Oklab's).

**Oklab pays nothing.** Its three channels reproduce every sRGB color, so
`resid_format()` is `None`, no third texture is allocated, and the eight passes that
carry a tile's color are built in a second variant under WESL's `@if(resid)`
conditional compilation (`RESID_ENTRY_POINTS`). `media_mixbox.wesl` and
`blend_mixbox.wesl` declare their own residual bindings — past where the shared
`media_common`/`blend_common` stop — so a colorimetric document gets a shorter bind
group layout rather than a placeholder to bind. What does reach it is one uniform
lane per **mirrored** struct (`Media.bg_resid`, `TileXform.resid`, `Stamp.j`,
`Fill.r`, matte's instance attribute): those cannot be conditional, because the
mirror generator reads *unlinked* sources (§6.10) and has no feature set to evaluate,
so a member that came and went with the variant would give one host struct two
layouts. They are genuinely zero there — `rgb_to_resid` returns `[0.0; 3]` for a
space whose channels are already the whole color.

One cost the channel does carry, stated rather than hidden:

- **The device limit.** The stamp loop's `exchange` wrote four storage textures — the
  extent snapshot's color and aux, which ride in the tail of that dispatch (§6.2),
  and the reservoir's color and aux — which is exactly WebGPU's guaranteed
  `maxStorageTexturesPerShaderStage`. The residual's two put it at six, so
  `minimum_required_limits` now asks for six. See `gpu/context.rs` for what that costs
  and for the packing that would get back to four if a device ever reports it.

The `bake`'s workgroup scan pays nothing worth quoting: one more `vec4` array a
column takes it from 9 floats a cell to 13, which is 6.5 KB of the 16 KB downlevel
budget, so its width is unchanged and no tip size resolves any coarser for it.

Mixbox is **vendored as a git submodule** (`vendor/mixbox`, Mixbox 2.0 ©2022
Secret Weapons, **CC BY-NC 4.0** — non-commercial; commercial use needs a licence
from `mixbox@scrtwpns.com`). CPU `rgb_to_channels`/`channels_to_rgb` call the
vendored crate (`no_std` + `libm`, so it builds for wasm and embeds its own LUT).
The GPU polynomial in `media_mixbox.wesl` is **generated at build time** from the
vendored GLSL (`stark-shaders/build.rs` transpiles `mixbox_eval_polynomial` into
a WESL module), so the trained coefficients stay sourced from the licensed
submodule rather than copied into this repo.

### The `mixbox` cargo feature

Because that licence is non-commercial, the whole space is behind a **default-on
`mixbox` feature**, and turning it off means *not built* rather than merely
*unreached*: the vendored crate leaves the dependency graph, `build.rs` does not
transpile the polynomial, `blend_mixbox`/`media_mixbox` leave `ENTRY_POINTS`, and
`pigment.rs` does not `include_bytes!` the LUT. Nothing under `vendor/mixbox`
reaches the binary.

**The submodule still has to be checked out, and that is cargo rather than a
leak.** A `path` dependency's manifest is read while the dependency *graph* is
resolved, and that happens before features select anything — so `vendor/mixbox/rust`
must exist for cargo to get as far as deciding not to compile it. `optional = true`
governs compilation, not resolution. CI's `no-mixbox` job therefore checks the claim
where it is actually observable, against the build graph
(`cargo tree --no-default-features | grep mixbox` must find nothing), rather than by
deleting the directory. Making the checkout itself optional would mean severing the
path dependency — a shim crate that `#[path]`-includes the vendored source, resolved
at compile time rather than at resolve time — which is a larger change than the
licence question needs.

Three consequences worth stating, because each is a place the obvious move is wrong:

- **`ColorSpaceId::Mixbox` is still a variant.** Postcard encodes enums by index (§8),
  so a build that dropped it would renumber every variant appended after it, and two
  builds of the same version would disagree about the save format — §19's whole
  subject. The id stays nameable and serializable; `ColorSpaceId::make` returns `None`
  instead, and `DocumentFile::from_bytes` turns that into
  `EngineError::UnsupportedColorSpace` at the one boundary an untrusted space id
  crosses. A frontend builds its picker from `ColorSpaceId::all_available`.
- **Falling back to Oklab would be worse than failing.** The two spaces read the same
  tile bytes as different colors, so opening a pigment document through a
  colorimetric one renders every pixel wrong while looking like it worked.
- **The residual goes with it.** A residual is what a *pigment* space needs and this
  is the only one, so without the feature no space declares a `resid_format`: the
  eight `_resid` variants are not built, and the loop's storage-texture requirement
  drops from six back to WebGPU's guaranteed four. What stays unconditional is the
  handful of mirrored uniform lanes — one host struct layout in both configurations,
  16 bytes each, which is the same reason they are unconditional across the *WESL*
  feature (§6.10).

## 6.10 The CPU↔shader boundary: generated mirrors

Every uniform is one half of a pair the compiler cannot see across. The shader
decides how the lanes are *read*; nothing on the host knows what it decided. Both
halves used to be written by hand — nine `vec4` lanes in `dynamics.wesl` against
nine `[f32; 4]` fields in `dynamics.rs`, each carrying its own copy of the lane
map — and what two hand-written copies of one fact do is drift. All three of these
had, silently: `ViewUniform`'s doc said 32 bytes and it was 48, `MediaUniform`'s
said 80 and it was 96 (`surf_m`, §18.1.2), `GuideUniform`'s said 240 and it was
304 (§20.8), and `Stamp.e.zw` was still documented on the host as the midpoint
`exchange` samples the canvas at, long after the shader stopped reading the lane.

**So the shader's declaration is now the only one.** `stark-shaders/build.rs`
already holds a parsed WESL tree; `build/mirror.rs` walks it and emits the Rust
struct into `stark_shaders::mirror::<wesl module>::<Name>` — fields, padding, and
the lane documentation, which lives exactly once and is read off the WESL comment
that abuts each member.

### Adding a mirror

1. Add `(&["<wesl module>"], "<Struct>")` to `MIRRORS` in
   `stark-shaders/build.rs`. Where several shaders declare the same struct against
   one host type, list them all: the first is generated from and the rest are
   **checked to agree**, member for member and offset for offset. `View` is why —
   `composite.wesl`, `matte.wesl` and `overlay.wesl` each write it out separately.
2. Delete the hand-written struct and import the generated one in its place,
   aliasing it to the host's name (`use stark_shaders::mirror::fill::Fill as
   FillUniform;`). Namespacing by WESL module is not cosmetic: `selection.wesl` and
   `slice.wesl` both call theirs `Params`, with different members.
3. A constructor has to become a **free function** — the type lives in another
   crate now, and Rust allows no inherent impl on it. `ViewUniform::new` →
   `view_uniform`, `GuideUniform::pack` → `pack_guides`.

### Adding a constant

Add `("<wesl module>", "<NAME>")` to `CONSTS` and delete the host's copy. The path
may reach into `lib/` (`("lib/paint_common", "TOOTH_RISE")`); the generated Rust
module is named for the file, since `lib` is a placement rule — binding-free leaves
— rather than a namespace.

A constant that disagrees is worse-behaved than a struct that does. A struct
usually surfaces as a wgpu validation error; a constant leaves both sides rendering
perfectly plausible pixels that no longer add up. The tooth's two are the sharpest
case — the CPU averages the gate over the substrate's rise distribution for the
*tool's* half of a transfer while the shader evaluates it per texel for the
*canvas* half, so drift is a conservation leak proportional to how far they moved
(§6.4).

Values are **evaluated, not scraped**. The generator runs `wesl`'s const evaluator
over the initializer and converts the result to the declared type, so a derived
constant is the number the shader will actually compute with, and an abstract
literal (`2`, `4.0`) becomes the concrete type the shader gives it rather than
whichever one it looked like. An explicit type is required — WGSL's abstract
numerics have no single honest Rust counterpart, and guessing is how a host
constant ends up a different type from the shader's.

Where two generated constants stand in a **relation** rather than mirroring each
other, assert it where it constrains something:

```rust
const _: () = assert!(SWEEP_VERTS == 2 * (SWEEP_SLICES + 1), "…");
```

That was a runtime test reading the *linked* shader, and it was limited by it: it
had to check through `SWEEP_SLICES` because the shader states `SWEEP_VERTS` for
the host's benefit and never computes with it, so the linker stripped it. (The
retired wick's cadence relation was the other case, and the sharper one — its
`WICK_RATE` survived the linker only as prose, because the shader computes with a
baked kernel.) Reading unlinked sources retires that whole class, and the checks
moved from `cargo test` to `cargo build`.

### Why it is trustworthy

**The layout is the point, and it is not the layout `#[repr(C)]` would give.** WGSL
aligns a `vec3<f32>` to 16 and sizes it 12, rounds a struct up to its own
alignment, and pads array elements and matrix columns out to a stride. A Rust
struct of the obvious field types agrees with none of that in general — it only
happened to agree here because every member was a `vec4`. A `vec3` anywhere but
last would have put every later lane four bytes early, with nothing failing to say
so.

None of those rules are implemented in the generator. `wesl::eval::ty_eval_ty`
resolves a member's type and `wgsl-types` gives it the spec's own `size_of` /
`align_of`, including the `@size`/`@align` attributes, nested structs and `f16`.
What is left is where the *host* has a choice: which Rust spelling occupies a given
stride, and the explicit padding fields that put the real members on their offsets
(explicit, so the struct has no *implicit* padding and stays `Pod`; a `Default` of
zeroes is generated so callers can write `..Default::default()` rather than name
them). Each struct then carries `size_of`, `align_of` and per-field `offset_of`
assertions, so an error in that last part is a build failure at the struct it got
wrong rather than a lane misread at run time.

It reads the **unlinked** sources, never the artifacts. The linker mangles `Stamp`
to `package__1dynamics_Stamp`, emits it once per artifact that reaches it, strips
whatever no entry point uses, and drops the comments that are half of what is being
generated. Parsing the linked WGSL with `naga` would buy the offsets and cost all
four.

### Adding a vertex instance record

Add `("<wesl module>", "<vs entry>", "<RustName>")` to `VERTEX`. The name is the
one thing the shader cannot supply — a parameter list has no name of its own — so
it is written once, beside the declaration it names. Then delete the host struct
*and* its `vertex_attr_array!`, and build the buffer from the generated layout:

```rust
buffers: &[Some(stark_shaders::mirror::stamp::segment_instance_layout(
    wgpu::VertexStepMode::Instance,
))],
```

**Three transcriptions collapse here, not two.** A vertex input was written as the
shader's parameter list, as a host `#[repr(C)]` struct, and *again* as
`vertex_attr_array![0 => Float32x2, 1 => Float32]` — where the formats restate the
types and the offsets are implied by the order. The third is the one with no
redundancy to catch it: swapping two same-sized attributes made every instance read
its neighbour's lane, silently.

**Its layout rule is not the uniforms'.** A vertex attribute's offset is the
*host's* to choose — WGSL's alignment tables do not reach a vertex buffer at all —
and what `vertex_attr_array!` chooses, so what the shaders were built against, is
**tight packing**. Members follow one another with no padding, which for these
types is also exactly what `#[repr(C)]` does (each is 4-byte aligned and a multiple
of 4 in size). The emitted `offset_of` assertions are what say those two rules
still agree. `MatteInstance` had carried a `_pad: [f32; 3]` for no stated reason,
making its stride 48 where its attributes span 36; generating it tightly dropped
the pad, and the goldens confirmed the stride change is pixel-identical.

`step_mode` stays the caller's, because nothing in the shader says whether the host
means to advance the buffer per vertex or per instance. Everything else — stride,
formats, offsets — comes from the declaration.

This is also where the generator stops *catching* mismatches and starts making them
unrepresentable. Renumbering a `@location` or reordering the parameters moves the
struct, the formats and the offsets together, so there is nothing left to disagree;
the only thing that can still fail is the host assigning a wrong *value*, and a
type change fails at the assignment.

It is why `stark-shaders` depends on `wgpu`: a vertex buffer layout is derived from
a shader, so it belongs to the crate that owns the shaders, and the types it needs
(`VertexAttribute`, `VertexFormat`, `VertexBufferLayout`) are plain data — no
device, no runtime.

### Adding a binding table

Add the WESL module to `BINDINGS` in `stark-shaders/build.rs`. Two things are
generated from every `@binding` declaration in it:

- `mirror::<module>::binding::<NAME>` — the index, uppercased from the WESL
  variable. This is what a host layout and a host bind group name a slot by.
- `mirror::<module>::BINDINGS` — a `&[Binding]` carrying, per slot, its **kind**
  (uniform and how wide, sampler, texture, storage texture *and its format*) and
  whether it is `@if(resid)`-gated.

The host then writes **one list per entry point** naming the slots it reads, and
builds both sides from it:

```rust
const DEPOSIT: &[Slot] = &[Slot::at(b::ST), Slot::sampled(b::DYN_NOISE_TEX), …];

let bgl   = desc::layout_for(device, label, DEPOSIT, vis, BINDINGS, resid);
let group = desc::bind_group_for(device, label, &bgl, DEPOSIT, BINDINGS, resid,
                                 |slot| /* the view or buffer for that slot */);
```

`crates/stark-engine/src/gpu/stroke/dynamics/slots.rs` is the worked example.

**A list is a membership statement and nothing more.** What kind of thing occupies
a slot, what format a storage texture is, how wide a uniform is, and whether the
build has it at all are read off the declaration — so the host cannot choose
`stor` where the shader said `rgba32float`, and cannot miscount a residual tail.
Because the layout and the group come from the same list, they cannot disagree
about which slots are present or in what order.

The **one** thing a list still says for itself is filterability: `Slot::at` for a
slot this entry point reads with `textureLoad`, `Slot::sampled` for one it reads
through a sampler. That is not an omission in the generator — it is genuinely a
property of the *(entry point, binding)* pair. `dynamics.wesl`'s `region_color` is
loaded by `snapshot` and sampled by `exchange`, so the same slot is non-filterable
in one layout and filterable in the next.

Two unit tests stand behind the hand-written half (`slots::tests`): every list
names bindings the shader declares, and names none twice; and a list takes a
residual slot exactly when it takes the plain slot that rides with it — checked
against the generated `resid` flags, so the shader gaining an `@if(resid)`
declaration that nothing pairs is a failure rather than a gap.

### What this does and does not cover

Generation was first aimed at the **silent** half of the boundary — uniforms,
constants and vertex instance records — on the argument that a bind group layout
disagreeing with its shader is a *loud* failure: wgpu reflects the module at
`create_*_pipeline` and names the offending binding.

That argument was right about the failure and wrong about the cost. Loud is not
the same as cheap: a validation error arrives at run time, on a machine with an
adapter, which is the half of the suite CI does not exercise against real pixels —
and the seven layouts of `dynamics.wesl` were seven pairs of hand-kept arrays
joined by magic element counts (`[..12 + 4 * usize::from(resid)]`), recounted by
hand on every edit. So the slots are generated now too, and what is left on the
host is the shortest thing that is genuinely the host's: which bindings an entry
point reads, and how it reads them.

Entry-point names stay hand-written. So do the layouts outside the stamp loop,
which are short enough that a list would be longer than the entries.

`gpu/wesl.rs` is gone entirely, both of its halves with it. `mirrors_wesl!` pinned
a hand-written struct's size against a number written beside it; `wesl_const`
scraped a decimal literal out of the linked artifact, carrying four documented
limits — stripping, reachability, `f64` widening, and name mangling. Neither was
improved, because there is no second declaration left for either to check.

### The rule this generalizes

Every one of these started the same way: a fact the shader states, transcribed on
the host so the two could be kept in step by attention. The lane map, the constant,
the vertex format, the binding index, the binding's type. In each case the
transcription drifted, and in each case the drift was invisible until it was a
picture.

So before adding a host-side declaration of anything a `.wesl` file already says,
check whether it belongs in `MIRRORS`, `CONSTS`, `VERTEX` or `BINDINGS`. The
generator has an AST of every shader in the tree; if the shader states it, the host
should be reading it rather than repeating it. What is left over — a name, a step
mode, whether *this* pass samples *that* texture — is worth writing by hand
precisely because the shader does not say it.


