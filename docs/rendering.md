# Compositing, media, and colour

The three passes, blend modes, presentation and the canvas surface, Oklab, and pluggable colour spaces — §6.3, §6.4, §6.5, §6.7.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 6.3 Compositing and the media pass

Three passes turn tiles into pixels. The first two are the substance; the third
is chrome.

**A — composite.** Every visible tile of every visible layer is drawn, bottom to
top, into two viewport-sized offscreen targets: colour (premultiplied "over", in
the working colour space) and the `(height)` aux (additive). Layer opacity rides
on the instance.

A layer's "over" weight is its **visible alpha** — per-unit opacity and amount
combined by the slab law `1 − exp(−K·opacity·height)`, the same law
`paint_common.wesl` uses to stack parcels *within* a layer — so a layer covers
the stack below exactly as much as it shows. (Weighting by opacity alone was the
old defect: a film with opacity 1 and no thickness — every soft brush's fringe —
drew as nothing over bare canvas yet replaced the colour over another layer's
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
  plus the canvas weave scaled by `surface_strength` — tilted by
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
  ground it barely covers, and bare canvas stays rough, so matte. (There was once
  a per-texel `wet` channel here; nothing could source it after
  `BrushParams::wetness` was removed, so it was a stored zero every pass carried.)
- **Present.** The working channels are converted to the surface's display space
  and composited over the substrate colour. This is the *only* place
  gamma-encoded colour exists.

**The reference invariant.** Under `Neutral` (exposure 1.0), with
`height_strength = 0`, the media pass is an identity — paint comes back out its
own colour, within about two bytes. That is what makes the neutral environment
worth having: it is the light you switch to in order to *judge* a colour rather
than enjoy it. Three things have to hold, each a constraint on the model rather
than a correction bolted on:

- **Exposure is divided by the irradiance a flat canvas actually receives** — the
  diffuse mip sampled dead ahead, computed on the CPU from the same mip chain the
  shader reads. The whole-image mean luminance it replaced only approximated
  that: averaging equirect texels over-weights the poles and counts light no
  front-facing canvas ever sees, leaving flat paint ~13% dark.
- **The diffuse keeps `1 - spec_energy`, not `1 - fresnel`.** The split-sum's
  `env_brdf` already integrates Fresnel, so subtracting a second Schlick term was
  double-counting it and losing ~2.4% of every colour.
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
So `Neutral` stays at 1.0, and `Ferndale` is authored at 0.65 — the value it was
judged at. `tests/reference.rs` pins the invariant.

**The blend modes** are deliberately not Photoshop's: each is ordinary **addition
of light, conjugated by a tone curve** — `f(a,b) = T(T⁻¹(a) + T⁻¹(b))` —
evaluated in **CIE XYZ normalized to the display white**, the only space in play
that is linear in light, non-negative for every real colour, and free of an
opinion about the display's primaries.

| Mode | `T(x)` | `f(a,b)` | Identity | Character |
|---|---|---|---|---|
| `Glow` | Reinhard `x/(1+x)` | `(a + b − 2ab)/(1 − ab)` | black | asymptotic — **cannot** blow out; stack a hundred and it approaches white without clipping. Glazes, mist, rim light. |
| `Radiance` | Drago `k·log(1 + x/k)` | `k·log(e^{a/k} + e^{b/k} − 1)` | black | no asymptote, so it *does* push past display white — deliberately, into pass B's highlight roll-off as a bloom with a filmic shoulder instead of a clip. |
| `Multiply` | `e^{-x}` | `ab` | **white** (black annihilates) | the conjugation collapses to plain `ab` and the added quantity is **optical density** — Beer-Lambert, what stacked glazes do. |

What deliberately did **not** ship is Screen. `a + b − ab` is what falls out of
inverting a multiply; it describes no optical process, and it flattens the top of
the range into the chalky white that reads as "digital glow" at a glance. Screen
is multiply conjugated again by `x ↦ 1−x`, and that second step is the one with
no physical referent — hence multiply, not Screen. Multiply in normalized XYZ
rather than RGB also means two saturated glazes cross without the dead channel an
RGB multiply produces when one primary sits near zero.

Being conjugations of `+` makes all three commutative and associative with a
neutral element, so reordering a stack of them is not a colour decision —
`tests/blend.rs` pins that, along with the identity (a mode over an empty stack
is bit-for-bit the `Normal` render). The neutral element is where the family
splits and where the tests split with it. Each colour space supplies only its
channels ↔ light conversion, which for Mixbox is the pigment polynomial and its
inverse LUT (`mixbox_lut.wesl`) — the one place the engine inverts Mixbox on the
GPU. The rest of the darkening family attaches at the same seam: one more `T`,
one more variant, no new machinery.

One thing is *not* free: a glaze sees the layer stack but not the substrate,
which pass B composites underneath afterwards, so multiply over bare paper leaves
the paper alone. Correct on white paper (its identity), wrong on a toned ground;
the fix is to make the substrate the bottom of the stack.

**C — selection outline.** One instanced quad per mask tile, drawn over the lit
result in the same canvas→NDC frame as pass A (§6.8). Suppressed on export (§15.6).

`MediaParams` (`height_strength`, `specular`, `surface_strength`) is a **view
setting** — per-client, never historized, changed by
`ViewCommand::SetMediaParams`. So is the choice of environment: switching it
re-lights the canvas and touches no stored pixel. Exposure is neither: it is not
tunable, it is what the chosen environment says it is. Nothing here is in the
save file.

The whole media model is a single shader stage, which is the point: Kubelka–Munk
pigment mixing, granulation, varnish gloss can be iterated on without touching
the document or tile machinery.

> **Not yet: damage tracking.** Every populated tile of every visible layer is
> composited every frame — there is no per-version damage set and no view-AABB
> cull, so off-screen tiles are drawn and clipped by the rasterizer rather than
> skipped. Fine at current canvas sizes; the obvious first optimization when it
> stops being.

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
the media pass's screen→canvas for the weave, and the overlay — and upright and
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

For zoomed-out views, tile **mip/LOD** sampling is a future optimization (§13).
The frontend owns the `wgpu::Surface`, acquires the frame texture, calls
`render`, and presents.

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

**The canvas surface.** Paint sits on a physical surface — a tileable height/bump
map (`gpu/surface.rs`), an `R8Unorm` texture sampled in *canvas* space (so the
weave is fixed to the canvas and pans/zooms with it). It is read twice: the
deposition tooth gates what a brush lays through it (below), and it feeds the
normal everywhere (`height_at` = impasto + `surface_strength·(h−½)`), so the weave
catches light across the whole viewport — including the bare substrate, whose
shading is *normalized* so a flat surface leaves it unchanged. `surface_strength`
is a view setting; it does not touch stored pixels.

The surface is **document state**: which canvas a piece was painted on is part of
what the document *is*, it is saved, and reopening on a different weave would be
a different painting. `CanvasMeta` records the surface the log *starts* from; a
mid-document switch is a logged `ActionKind::SetSurface`, so it undoes, replays
and replicates like any other edit. The deposition tooth reads it, so a switch
changes the strokes made after it — logging it is what made that a rendering
change rather than a history one.

**A ground is named by its image, not by a label.** `SurfaceId` is
`Flat | Image(AssetId)`: one procedural ground that needs no bytes, and
everything else identified by the BLAKE3 hash of its canonical decoded height
field — the same bargain brush shapes make (§6.6), and for a sharper reason. A
label is only as good as the table the reader holds. When grounds were
`{ Flat, Linen, Gesso }`, a peer who received `SetSurface(Gesso)` without ever
having fetched gesso fell back to the flat stand-in *silently*, and from then on
deposited every stroke with no tooth at all; the two canvases diverged and
neither screen could say why. Unlike the media pass — which re-reads the ground
each frame and rights itself the moment an image lands — a deposit is **stored**,
so nothing un-bakes it. A content id removes the failure rather than reporting
it: the holder either has those exact bytes or knows precisely what to ask a peer
for, and what comes back is verified against the id that asked (`accept_surface`
refuses a mismatch). The same mechanism carries a ground a *user* brings, which a
closed enum could never have named.

Three consequences follow. A save file **bundles every ground its log names**,
not just the one it ends on (§8) — a height map is a replay input exactly as a
coverage mask is, and a document that switched part-way needs both. The engine
**embeds no image bytes**: grounds are fetched at runtime and handed to
`import_surface`, which decodes, downsamples by an integer factor to fit the 2048
limit (preserving tileability), hashes, and returns the id — so an id is only
knowable once the bytes are in hand. And `DEFAULT_SURFACE` is `Flat`, because
core naming linen would be core naming an image it cannot produce; the frontend
holds a catalog (`stark-ui/src/grounds.rs`, the analogue of `builtins.rs` for
shapes) and opens a fresh document on linen once its map has landed.

The bundled grounds: linen, a regular woven grid; and gesso, a brushed acrylic
ground, irregular, whose height histogram is a broad spread rather than a
periodic peak — which is what makes it the interesting one for the tooth below,
since a periodic weave prints a periodic mark and reads as a screen. `Flat` is a
1×1 *full-height* texel — a constant height has zero gradient, so it is *exactly*
equivalent to having no surface. That orthogonality is deliberate: most goldens
run on `Flat` to test other features in isolation, and a dedicated golden
(`linen_surface`) exercises the weave. One bump tile spans `SURFACE_TILE_PX`
canvas px.

**The deposition tooth.** Paint lands where the tip touches the ground, and on a
rough ground the tip touches the peaks before the valleys. `BrushParams::tooth`
is how deep a tool reaches — 0 = everywhere (the mark is solid, and the default),
1 = the very tops only, which is what a dry brush leaves.

The model is the substrate's **bearing-area curve** (Abbott–Firestone): the
fraction of a rough surface standing above a given level. The tip presses to a
level set by the knob, and a texel takes paint where the ground clears it. Per
texel that is one *sample* of the curve, so the mean over a footprint is the true
contact fraction — a prediction that can be checked against the height map's own
histogram rather than a curve tuned until it looked right. The transition is
softened over a band (`TOOTH_SOFTNESS`) because a hard threshold is a binary
indicator per texel: correct in the mean, and at canvas resolution it aliases
into speckle that reads as dither. A cubic smoothstep, for the reason
`taper_profile` is a polynomial; both ends of the knob map to exact limits, so
`tooth = 0` puts the whole band below the map's range and the gate is `1.0` to
the bit.

Three decisions do most of the work, and each is about *where* it is applied
rather than what it computes:

- **The grain is the canvas's, not the brush's.** A pencil and a loaded brush on
  one ground see one tooth; the brush says only how far into it it reaches. That
  is why `SurfaceId` is where the texture lives and `tooth` is the only thing on
  `BrushParams`. Painter and Procreate put the grain on the brush, which is why
  switching brushes there changes the paper under a half-finished painting.
- **It scales the exposure, not the transfer**, and that one choice is what makes
  it both compose and conserve. Every rate here is a function of swept optical
  depth τ — additive in it, or `1 − exp(−k·τ)` — so multiplying τ itself by a
  per-texel `g` leaves every one of them in its form: `exp(−k·g·τ₁)·exp(−k·g·τ₂) =
  exp(−k·g·(τ₁+τ₂))`, and `Σᵢ g·τᵢ = g·Στᵢ`, because `g` belongs to the canvas and
  not to the segment. Applied to the finished shares instead, a toothed lift would
  fade at a rate that depended on where the flattener cut. It is read at the
  fragment's own canvas position, so tile aprons stay bit-consistent with no copy
  pass, and — the reason the mark reads as paper rather than as noise —
  successive strokes catch on the same peaks and register with each other.
- **It gates height, never the per-unit opacity** (§6.1). The tooth decides how
  much paint arrives, not what the pigment is. Both render paths call the same
  `tooth_gate` in `paint_common.wesl`, so nudging `lift` off zero cannot change
  what the ground does.
- **The tool books against the ground's mean, and that is what conserves paint.**
  A transfer has two halves in two dispatches — the canvas gives up `1 − keep` and
  takes `dep` of the tool's load, the tool takes and gives the exact complements
  (`dynamics.wesl::Exchange`). Gate one half and not the other and the books stop
  closing. Scaling the *exposure* keeps them complementary, because `exchange_at`
  is solved once at `g·e` and its four shares still sum to what went in. But a
  reservoir cell has no ground of its own — it is dragged over fresh canvas at
  every sub-step — so where a canvas texel scales by the ground beneath it, the
  cell scales by the **bearing fraction**: the mean of the gate over the whole
  height map, which `Surface::bearing` computes from the map's own histogram. The
  two agree in expectation over any footprint spanning many grain features, which
  is every usable tip, and the residual is the same order as the mean-field freeze
  either side of the kernel already carries.

That last point is why the height map is read with **nearest** rather than
bilinear. Filtering would average away the peaks the tooth exists to catch on, and
— worse — it would draw from a narrower distribution than the histogram the CPU
integrates, so the two halves of the transfer would disagree systematically. It
also sidesteps the reduced-precision filter weights `prefix_slice` documents, so
the tap is bit-reproducible.

Orthogonality is structural rather than checked. `Surface::relief` is 0 on `Flat`
and on any surface whose bytes have not arrived, which zeroes the uv scale the
shaders gate on before the brush's number is consulted — so every golden that
paints on `Flat` is untouched by the axis existing, exactly as it is by the media
pass's weave.

Because deposition reads the surface, **which** surface is a question about the
action being applied rather than about what the compositor is showing: the
registry lives on `ApplyCtx`, and `CommitStroke` asks it for `DocState::surface`
*as the log stood at that action*. A switch part-way through a document changes
the strokes after it and none before, on replay and on a peer alike. That is what
logging `SetSurface` bought, long before there was anything to spend it on — the
note this paragraph replaced said as much. It also means the ground a document
names has to be registered before its log is replayed, or its strokes bake with
no tooth and stay that way; the frontend fetches and re-replays when it finds it
has opened a file whose ground had not arrived.

> **Still open: the tooth does not fill.** `g` reads the substrate alone, so
> overpainting never uses the weave up — the same grain prints through every
> layer, which is the tell that gives a static grain multiply away. The fix is to
> read `max(R·s(x), paint_height(x))`, the media pass's own `height_at`, so a
> valley full of paint stops being a valley. It needs two things this version does
> not have: a relief scale `R` in paint-height units (today `surface_strength` is
> a *view* setting, and deposition cannot read one without making stored pixels
> depend on the viewport), and an answer to associativity — the commit renders a
> whole stroke in one range against the committed base while the live path renders
> head-then-tail, so a gate reading the evolving base breaks `preview ==
> committed` wherever a stroke crosses itself. The way out is to make the deposit
> a **flow**, `dh/dτ = add·g(h, x)`, which composes exactly under any subdivision
> of τ by construction; the stamp loop can integrate that (it is sequential and
> already solves this shape for lift/deposit) and the order-independent swept path
> cannot. Meanwhile `bleed` is the wet-paint valley-filler that already exists:
> charcoal is tooth with no bleed and stays broken, oil is tooth with bleed and
> levels out.

## 6.5 Colour management (Oklab)

Colour flows through exactly three representations, and conversions live in one
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


## 6.7 Pluggable colour spaces (Oklab & Mixbox pigment mixing)

Tile channels are **colour-space-agnostic**: tools deposit values and only assume
they *blend linearly*, never what colour they represent. The meaning — and the
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
    // GPU: how a dab writes its channels, and how channels become display colour.
    fn stamp_shader(&self) -> &'static str;  // MRT deposit (§6.2)
    fn media_shader(&self) -> &'static str;  // media/lighting + present (§6.3)
}
```

A document has one colour space (`CanvasMeta.color_space`), so tile format, blend
state and shaders are fixed per document and chosen at engine construction. Pass
A is generic; only the **stamp** and **media** shaders, the formats and the
blends are space-specific.

**`OkLabColorSpace`** — `color = Rgba16Float` holding premultiplied
`(L, a, b, coverage)`, `aux = R16Float (height)`, premultiplied-"over" colour
blend (coverage *is* the blend alpha), additive aux.

**`MixboxColorSpace`** — realistic pigment mixing via **Mixbox** (Secret
Weapons), where blue + yellow makes green like real paint rather than the muddy
grey of an RGB blend. Mixbox represents a colour as a *latent* of pigment
concentrations `c0..c3` plus a small residual, and mixes by **linear interpolation
in latent space**, then maps latent → RGB through a trained polynomial. The
decisive fit: *latents blend linearly*, so the ordinary premultiplied-"over"
deposit **already performs Mixbox mixing** — no programmable blend, no extra
pass. The tile layout is **identical to Oklab**: `color = Rgba16Float` holding
premultiplied `(c0, c1, c2, coverage)`, `aux = R16Float (height)`. The stamp
shader is reused verbatim; only the **media shader differs** — it un-premultiplies
the concentrations and evaluates Mixbox's polynomial (`c3 = 1 − (c0+c1+c2)`
derived) to a base colour before the shared impasto lighting.

Mixbox's latent **residual is dropped**: a tile has room for three concentrations
plus coverage, and the residual would need a fourth over-blended channel (a third
tile texture). Dropping it keeps zero architecture change and full *mixing*
fidelity; the only cost is slightly approximate reproduction of very saturated
colours (the residual ≈ 0 in gamut). Recovering it is a future third-texture
option.

Mixbox is **vendored as a git submodule** (`vendor/mixbox`, Mixbox 2.0 ©2022
Secret Weapons, **CC BY-NC 4.0** — non-commercial; commercial use needs a licence
from `mixbox@scrtwpns.com`). CPU `rgb_to_channels`/`channels_to_rgb` call the
vendored crate (`no_std` + `libm`, so it builds for wasm and embeds its own LUT).
The GPU polynomial in `media_mixbox.wesl` is **generated at build time** from the
vendored GLSL (`stark-shaders/build.rs` transpiles `mixbox_eval_polynomial` into
a WESL module), so the trained coefficients stay sourced from the licensed
submodule rather than copied into this repo.


