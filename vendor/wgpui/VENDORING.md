# Vendoring notes: wgpui (patched)

wgpui `0.3.5` from the crates.io source (`registry/src/.../wgpui-0.3.5`,
upstream commit `cc6706e62b` in `.cargo_vcs_info.json`), **less its `examples/`
tree**, plus four local patches. Substituted for the crates.io crate via
`[patch.crates-io]` in the root workspace manifest, which also redirects what
`wgpui-component` (§11.1) asks for, so the widget layer draws with the device the
first patch describes. License: Apache-2.0 (`LICENSE.md`, kept).

Consumed by `crates/stark-wgpui-frontend` (§11).

Two patches carried against 0.3.4 landed upstream in 0.3.5 in their own form and
are gone from here: the `flume` declaration without which the crate did not
compile on Windows at all (0.3.5 declares it on every platform), and the
compositor's surface bind groups going stale across a resize (0.3.5 keys the
cache by a `revision` the registry bumps, where the patch had keyed it by a
`generation` — same fix, its name). They are kept below the line for the
record, since the second is the shape of bug this fork produces.

## Patch 1 — the caller states the `DeviceDescriptor`

`Application::new` takes a `&wgpu::DeviceDescriptor` and threads it through
`platform::current_platform` → `CrossPlatform::new` → `WgpuContext::new`, which
hands it to `Adapter::request_device` in place of the literal it used to build.

Everything the literal said that is now a default was checked to be the same
value, so the patch changes only what the caller overrides:
`InstanceDescriptor::new_without_display_handle()` already defaults `backends`
to `Backends::all()`, and `RequestAdapterOptions::default()` already gives
`compatible_surface: None`, `force_fallback_adapter: false` and
`apply_limit_buckets: false`. `PowerPreference::HighPerformance` is still
written out, because its default is *not* that.

Two public functions went with the `headless` flag the signature displaced:
`Application::headless()` and `platform::background_executor()`. Neither is
reachable from this workspace, and upstream's own comment above
`current_platform` reads `TODO(mdeand): Support headless` — the flag was
threaded but never honoured, so what was removed was a name rather than a
behaviour. **This is the part of the patch that is not upstreamable as-is**: an
upstream version would keep both, either by giving `Application::new` a
descriptor-less form or by making the headless constructor take one too.

### Why

The engine's `GpuContext::minimum_required_limits` is the requirement, and it is
`#[cfg]`-dependent: the Mixbox stamp loop (§6.7) writes **six** storage textures
per shader stage where the Oklab one writes four. Upstream asks for
`wgpu::Limits::default()`, which guarantees four — so with wgpui owning the
device, a whole colour space was unreachable from this frontend and no amount of
cargo features could reach it, because limits are settled when the device is
created and the feature is compiled in long before.

The device is shared: wgpui's own renderer draws with it too. So the descriptor
cannot be either consumer's alone, and the frontend — the one thing that knows
about both — is where it belongs. `main::device_descriptor` starts from
`Limits::default()` (what wgpui's renderer was written against) and raises only
the fields the engine needs, with `or_better_values_from`.

## Patch 2 — `WindowBounds` is honoured whole

`WindowParams` carried `bounds: Bounds<Pixels>` — `WindowBounds::get_bounds()`, the
rect with the variant thrown away — so the platform never learned whether a window
was meant to open maximized. It carries `window_bounds: WindowBounds` now, and its
two readers (`platform::open_window`, the test window) call `get_bounds()`
themselves.

`CrossPlatform::open_window` then builds the winit attributes with `with_position`,
`with_maximized` and `with_fullscreen` beside the `with_inner_size` it already had.
And the `match window_bounds` in `Window::new` that used to apply the state
afterwards is gone.

### Why

Two independent faults, and the second is why the first was not obvious.

**The position was dropped.** The attributes were built from the size alone, so an
app restoring a window to where the user left it got the size back and the OS's
cascade for a position.

**The state was applied with a toggle.** `Window::new` did
`WindowBounds::Maximized(_) => platform_window.zoom()`, and `CrossWindow::zoom` is
`set_maximized(!is_maximized())`. Asking a window to *be* maximized by toggling it
is only right when it is not already — and on a freshly created window that has not
been pumped yet, the call does not take on Windows regardless, so the state was lost
either way.

The two interact: fixing only the first, by adding `with_maximized` to the
attributes, makes the toggle *correct in reverse* and the window opens restored.
They have to move together, which is why they are one patch.

Applying the state at creation is also what preserves the restore rect: winit's
Windows backend deliberately maximizes after `CreateWindowEx` "because if the size
is changed in `WM_CREATE`, the restored size will be stored in that size".

Upstreamable, and worth it — `zoom()` being a toggle used as a setter is a bug
independent of anything Stark wants.

## Patch 3 — `RenderImage` is RGBA, like the atlas it goes into

Every producer of a `RenderImage` swapped red and blue on the way in — the two image
decoders in `elements/img.rs`, the two in `platform.rs` (the clipboard's), the SVG
rasterizer, and colour emoji in `platform/text_system.rs`. The swaps are gone, the
type's doc says RGBA, and `swap_rgba_pa_to_bgra` is `unmultiply_alpha`: it did two
things and only one of them was wanted.

### Why

**It drew every loaded image with red and blue exchanged.** The chain is three files
and has no swizzle anywhere in it:

- `platform/atlas.rs` allocates the polychrome atlas as `wgpu::TextureFormat::Rgba8Unorm`.
- `window.rs::paint_image` hands the atlas `Cow::Borrowed(data.as_bytes(..))` — a
  straight copy, no reordering.
- `shaders/poly_sprites.wgsl` does `textureSample(..)` and uses `color.rgb` and
  `sample.a` as they come.

So the bytes reach the shader as RGBA, and every producer was writing BGRA.

This is GPUI heritage rather than an oversight: upstream renders through Metal, whose
natural surface format is `bgra8Unorm`, and the swaps are correct there. The fork
moved the atlas to wgpu and `Rgba8Unorm` and left the loaders behind.

**How it surfaced.** Stark's own consumers build a `RenderImage` by hand, and took the
doc at its word — so the asset galleries (§11.2 N7) swapped too, and *nothing could
show it*: an asset card is a brush shape's coverage or a substrate's height, both
grey, and grey survives exchanging red and blue exactly. The Oklab colour wheel (N8)
was the first coloured picture either frontend put through this path, and it was wrong
on its first frame — the marker sat on a blue the readout beside it called `#9c0a05`.

**The un-premultiply is kept.** `swap_rgba_pa_to_bgra` also divided out premultiplied
alpha, which the SVG rasterizer's output genuinely needs: `poly_sprites.wgsl` does its
own multiply when the global says to, so a buffer arriving already multiplied would be
darkened twice. Only the channel swap is dropped, and the name now says what is left.
Its `to_brga` parameter is `unmultiply`, and its two call sites still pass what they
passed — one `true`, one `false`. That asymmetry looks wrong too and is left alone:
it is a second question, and this patch is about one.

Upstreamable, and worth it — an image element that exchanges two channels is a bug
independent of anything Stark wants. Colour emoji are still garbled here for a
separate reason (they draw as stripes, which is a stride fault rather than a channel
one); that is untouched and unfixed.

## Patch 4 — an HDR swapchain, and the shaders decode for it (sites marked `STARK PATCH`)

`src/platform/renderer.rs`: where the window's surface is configured, prefer
`Rgba16Float` in `SurfaceColorSpace::ExtendedSrgbLinear` (scRGB) when the surface
advertises it — an HDR display on Windows (DXGI) or macOS (EDR) — else the 8-bit
non-sRGB format as before. `WGPUI_HDR=0` in the environment keeps the 8-bit path.
The `GlobalParams` padding lane becomes `linear_output`, set to 1 on the scRGB
swapchain, and a `sdr_white_scale` lane joins it — the display's SDR white in scRGB
units, read through `wgpu::DisplayLuminance::sdr_white_nits` and re-read on a 500 ms
throttle, since it moves with the brightness slider and with the display the window
was dragged onto. `src/platform/render_context.rs`: the globals buffer is
`size_of::<GlobalParams>()` where it was the literal `16` the struct used to be — a
uniform in WGSL rounds up to a multiple of 16, so the fifth lane took it to 32.
`WgpuRenderer::surface_color_space` and `display_headroom` report the choice and the
display's headroom. `src/platform.rs`, `src/platform/window.rs`, `src/window.rs`:
`surface_color_space()` and `display_headroom()` on `PlatformWindow` (defaulting to
`None`) and on the public `Window`.

`src/shaders/*.wgsl`: the `Globals` struct's `pad` lane is `linear_output`, and
every fragment shader that writes the swapchain decodes its sRGB-encoded output to
linear when it is set and scales it by `sdr_white_scale` — `blend_color` in each of
`mono_sprites`, `path_common`, `poly_sprites`, `quads`, `shadows` and `underlines`
(a `stark_to_linear` helper beside each), and `fs_path` in `paths.wgsl`, whose
intermediate is premultiplied and so is un-premultiplied, decoded and
re-premultiplied. `surfaces.wgsl` takes the scale and nothing else: an embedder's
texels are linear already, so there is no decode to do — but they are linear with
`1.0` at SDR white, and putting the canvas on the same reference white as the chrome
around it is the whole point of the lane being in `Globals` rather than in the
chrome's own helper.

### Why

wgpui's shaders work and blend in sRGB-encoded values written to a non-sRGB 8-bit
swapchain; on a linear one the same values read gamma-lifted. Stark renders its
canvas into a `WgpuSurface` and wants the swapchain to carry light above SDR white
(§6.5), which only a float, linear (or PQ) swapchain can. So the swapchain goes
linear where the display allows, and the chrome's output is decoded at the last
step. Blending then happens in linear light, which reads a shade thinner at
antialiased text edges — the physically right answer, and a small difference. An
`*Srgb` swapchain was not an option: its hardware encode would fight the shaders'
own.

### The reference white

**An scRGB swapchain is not dark by 80 nits' worth on its own — it is dark because
nothing else on the display is.** scRGB fixes `1.0` at 80 nits: that is what
`DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709` means, and wgpu passes the color space
down to `SetColorSpace1` without a scale of its own. Windows meanwhile composites
every *SDR* window at the brightness slider's white level — 200 nits or more out of
the box — so the first version of this patch drew a correct picture at 40% of the
luminance of every window beside it. The report was that the whole app, chrome and
canvas alike, went dark and flat the moment the display turned HDR on, **and that a
screenshot of it looked right**: the capture path reads the buffer and calls `1.0`
white, which is exactly the assumption that fails on the glass.

So the scale is the swapchain's, and lives with the swapchain. wgpu documents
`ExtendedSrgbLinear` as "`1.0` is SDR reference white", which is the contract
`surface_color_space()` hands an embedder and the one `stark-engine`'s
`Transfer::Linear` is written to (§6.5) — the engine says "SDR white" and this says
how many scRGB units that is here. Doing it the other way, by folding the ratio into
what the frontend hands the engine, would put a Windows display convention inside a
model that is meant to name light and not nits.

`1.0` where the level is unknown, which is every platform but Windows: macOS's
`extendedLinearSRGB` already puts SDR white at `1.0`, and Apple reports no absolute
nits to scale by.

Not upstreamable as is — upstream would want the color space to be a
`WindowOptions` choice rather than "HDR when the display has it" — but the shader
half is what any such option would need.

## The deletion

`examples/` is gone, and with it the thirty `[[example]]` blocks that named its
files in both manifests — cargo refuses a declared target whose path is missing,
so the two go together.

It was 4.9 MB of the 8.8 MB this directory weighed, and 4.5 MB of *that* was one
demo GIF. Nothing builds it: `vendor/wgpui` is excluded from the workspace, so
`--all-targets` does not reach these, and no example here is a reference the
frontend is written against — the crate's rustdoc is, and `docs/` is kept. A
blob that size is in git history for good, which is what tips a "copied
verbatim" that would otherwise be worth keeping for the clean update diff.

## Landed upstream in 0.3.5

**`flume` on Windows.** `Executor::spawn_realtime` called `flume::bounded`
unconditionally while the manifest declared `flume` only for macOS, Linux and
FreeBSD, so 0.3.4 did not compile on Windows at all. 0.3.5 declares it on every
platform, which was the whole of the patch.

**A resized surface gets new bind groups.** `WgpuRenderer::surface_bind_groups`
caches, per surface, the two bind groups that name the surface's two textures;
`SurfaceRegistry::resize` replaces both textures; and the cache was keyed on the
id alone. So after any resize the compositor went on sampling the pair from before
it, and because `swap_buffers` kept alternating, the window flickered between the
last two frames drawn before the resize while every stroke, pan and zoom since
landed in textures nobody read. Logs said it worked; it took screenshots. 0.3.5
keys the cache by `(SurfaceId, revision)` and drops entries not seen in a frame,
where the patch had carried a `generation` on the pair — the same fix.

## Notes

- `.cargo/config.toml` came with the published crate and is inert here: cargo
  discovers config by walking up from the **working directory**, which for
  every build in this workspace is the repo root. It applies only to somebody
  who `cd`s into this directory, and nothing does.
- Excluded from the workspace (root `Cargo.toml`), so `cargo fmt --all` and
  `cargo clippy --workspace` do not reach it — see the comment there. Its
  `[[example]]` targets are not built by `--all-targets` for the same reason.
- **`PlatformWindow::window_bounds` reports the window's *current* frame**, wrapped
  in whichever variant the state says — so a maximized window answers
  `Maximized(the screen it fills)` where the variant documents its bounds as the
  size to restore *to*. Not patched, because the answer is not winit's to give
  (there is no `restore_size`; Windows has `GetWindowPlacement` and the other
  platforms differ). `stark-wgpui-frontend`'s `window::remember` handles it where
  it costs nothing: a maximized window keeps the rect already on file, which is by
  construction the last size it was not maximized at.
- **`wgpui_derive` is not vendored.** This manifest asks the registry for
  `wgpui_derive = "0.3.5"`, and the derive's output has to match the structs here
  (0.3.5's derive emits a `BoxShadow.inset` that 0.3.4 lacked, which is how a
  `cargo update` once broke this tree). The lockfile is the guard; a bump of one
  is a bump of both.

## Updating

Unpack the new version over this directory, then re-apply the five changes: the
`DeviceDescriptor` threading, the `WindowBounds` plumbing, the RGBA
`RenderImage`, the HDR swapchain with its shader decode, and the deletion —
`rm -rf examples/` and strip the `[[example]]` blocks the new manifests bring
back. The 0.3.4 → 0.3.5 move was done as a `diff -ruN` of this tree against the
pristine registry copy, applied with `patch -p1` onto the new tarball; 22 files,
two rejected, both the resize patch upstream had already made.

Dropping this directory takes patch 1 landing upstream in some form, or the
frontend goes back to a device it cannot ask anything of.
