# Vendoring notes: wgpui (patched)

wgpui `0.3.4` from the crates.io source (`registry/src/.../wgpui-0.3.4`,
upstream commit `5e94b544ff` in `.cargo_vcs_info.json`), **less its `examples/`
tree**, plus one local patch. Substituted for the crates.io crate via
`[patch.crates-io]` in the root workspace manifest. License: Apache-2.0
(`LICENSE.md`, kept).

Consumed by `crates/stark-wgpui-frontend` (§11).

## The patch (every site marked `STARK PATCH`)

`Cargo.toml`: declare `flume = "0.12"` for `cfg(target_os = "windows")`.
`Cargo.toml.orig` gets the same line — cargo does not read that file, but a
packaged copy contradicting the manifest beside it is exactly the drift this
tree does not keep.

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

## Why the patch

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

## Notes

- `.cargo/config.toml` came with the published crate and is inert here: cargo
  discovers config by walking up from the **working directory**, which for
  every build in this workspace is the repo root. It applies only to somebody
  who `cd`s into this directory, and nothing does.
- Excluded from the workspace (root `Cargo.toml`), so `cargo fmt --all` and
  `cargo clippy --workspace` do not reach it — see the comment there. Its
  `[[example]]` targets are not built by `--all-targets` for the same reason.

## Updating

Unpack the new version over this directory, then re-apply both patches: the
`flume` line in each manifest, and the deletion — `rm -rf examples/` and strip
the `[[example]]` blocks the new manifests bring back. Check whether upstream
has taken the `flume` fix; if it has, drop this directory and the
`[patch.crates-io]` entry with it.
