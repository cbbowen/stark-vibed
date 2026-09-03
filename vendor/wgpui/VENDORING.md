# Vendoring notes: wgpui (patched)

wgpui `0.3.4` from the crates.io source (`registry/src/.../wgpui-0.3.4`,
upstream commit `5e94b544ff` in `.cargo_vcs_info.json`), **less its `examples/`
tree**, plus three local patches. Substituted for the crates.io crate via
`[patch.crates-io]` in the root workspace manifest. License: Apache-2.0
(`LICENSE.md`, kept).

Consumed by `crates/stark-wgpui-frontend` (§11).

## Patch 1 — `flume` on Windows (sites marked `STARK PATCH`)

`Cargo.toml`: declare `flume = "0.12"` for `cfg(target_os = "windows")`.
`Cargo.toml.orig` gets the same line — cargo does not read that file, but a
packaged copy contradicting the manifest beside it is exactly the drift this
tree does not keep.

### Why

`Executor::spawn_realtime` (`src/scheduler/executor.rs`) calls
`flume::bounded` unconditionally, while the manifest declares `flume` only
under `cfg(target_os = "macos")` and `cfg(any(target_os = "linux", target_os
= "freebsd"))`. So **wgpui 0.3.4 does not compile on Windows at all**:

```
error[E0433]: cannot find module or crate `flume` in this scope
   --> src/scheduler/executor.rs:173:24
```

Upstream `root` HEAD has the same omission. Nothing else about the crate is
Windows-specific — the platform layer is winit throughout, with only a handful
of macOS files beside it — so this one line is the whole of the fix, and with
it the frontend builds, opens a window and paints.

Straightforwardly upstreamable: the fix is the missing declaration, not a
change of behaviour.

## Patch 2 — the caller states the `DeviceDescriptor`

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

## Patch 3 — `WindowBounds` is honoured whole

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

## Updating

Unpack the new version over this directory, then re-apply all four changes: the
`flume` line in each manifest, the `DeviceDescriptor` threading, the
`WindowBounds` plumbing, and the deletion — `rm -rf examples/` and strip the
`[[example]]` blocks the new manifests bring back.

Dropping this directory takes more than the `flume` fix landing upstream now:
patch 2 has to land too, in some form, or the frontend goes back to a device it
cannot ask anything of.
