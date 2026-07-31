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
weave is fixed to the canvas and pans/zooms with it). It feeds the normal
everywhere (`height_at` = impasto + `surface_strength·(h−½)`), so the weave
catches light across the whole viewport — including the bare substrate, whose
shading is *normalized* so a flat surface leaves it unchanged. `surface_strength`
is a view setting; it does not touch stored pixels.

The surface is **document state** (`SurfaceId { Flat, Linen }`): which canvas a
piece was painted on is part of what the document *is*, it is saved, and
reopening on a different weave would be a different painting. A fresh document
starts on `DEFAULT_SURFACE` = `Linen` — the honest substrate — while
`SurfaceId::default()` stays `Flat`, the builtin the registry falls back to before
the frontend's bytes arrive. `CanvasMeta` records the surface the log *starts*
from; a mid-document switch is a logged `ActionKind::SetSurface`, so it undoes,
replays and replicates like any other edit. Today only the media pass reads it,
so a switch changes no stored pixel — logging it anyway is what would let a
future deposition gate read it without that becoming a history change. `Flat` is
a 1×1 *full-height* texel — a constant height has zero gradient, so it is
*exactly* equivalent to having no surface. That orthogonality is deliberate: most
goldens use `Flat` to test other features in isolation, and a dedicated golden
(`linen_surface`) exercises the weave. The engine **embeds no image bytes**:
image-backed surfaces are fetched at runtime and handed over via
`register_surface` (§6.6), which builds the texture (downsampling by an integer
factor to fit the 2048 limit, preserving tileability); one bump tile spans
`SURFACE_TILE_PX` canvas px. A surface with unregistered bytes falls back to
`Flat`.

> **Deposition tooth — removed, may return.** The idea was to gate deposited
> coverage by the surface height at each fragment,
> `cov ·= 1 − tooth·(1−h)·(1−cov)`, so light strokes catch on the weave's peaks
> and skip its valleys. It was never implemented: `surface_tooth` was a
> pass-through stub, no stamp shader ever read the surface, and
> `BrushParams::tooth` reached a slider that moved and changed nothing. All of it
> — the field, the stub, the `TileXform::surf` uniform nothing read, the
> `group(2)` surface bindings and `StrokeRenderer::set_surface` that existed only
> to keep a texture bound for a function that ignored it — was deleted. Every
> golden was unchanged, which is the proof it was inert. If it returns it needs a
> design first (the formula above is a guess, not a model), and `BrushParams`
> would carry a strength again. The surface is already document state, so that
> would be a rendering change, not a history one.

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


