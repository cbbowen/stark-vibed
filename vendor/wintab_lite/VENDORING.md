# Vendoring notes: wintab_lite (patched)

`wintab_lite` `1.0.1` from the crates.io source (upstream commit `541d972e56` in
`.cargo_vcs_info.json`), **less its `examples/` tree and the dev-dependencies
that only the examples needed**, plus one local patch. Substituted for the
crates.io crate via `[patch.crates-io]` in the root workspace manifest. License:
MIT (no `LICENSE` file is shipped in the crate; the manifest states it).

Consumed by `crates/stark-pen` (§11.3), which loads `Wintab32.dll` at runtime and
needs this crate for the half that is genuinely hard: the `LOGCONTEXT` layout, the
`WTPKT` mask and the packed packet it describes, which have to match the DLL's
idea of them byte for byte or every reading is garbage.

## Patch 1 — the `windows` dependency, which was never needed (sites marked `STARK PATCH`)

`src/extern_function_types.rs`: `use crate::c_type_aliases::HWND` in place of `use
windows::Win32::Foundation::HWND`. `src/c_type_aliases.rs`: add that alias, as
`isize`. `Cargo.toml` and `Cargo.toml.orig`: drop the `windows` dependency — cargo
does not read the latter, but a packaged copy contradicting the manifest beside it
is exactly the drift this tree does not keep.

### Why

**Upstream 1.0.1 does not compile as a dependency at all.** It declares
`windows = "0.56.0"` with no features and then imports
`windows::Win32::Foundation::HWND`, which lives behind the `Win32` feature:

```
error[E0433]: could not find `Win32` in `windows`
  --> wintab_lite-1.0.1/src/extern_function_types.rs:11:14
   |
11 | use windows::Win32::Foundation::HWND;
```

It builds in its own repository because a **dev**-dependency there enables those
features, and cargo unifies features within one build — so the published crate has
never been compiled the way a consumer compiles it.

The whole dependency existed for that one import, used in one parameter. The
crate's own `raw-dylib` declaration in `extern_functions.rs` already spells the
same parameter `isize`, and `windows`'s `HWND` *is* `isize` on this platform, so
the alias agrees with both about the ABI while pulling in nothing.

Removing it is worth more than fixing the features would be: `windows` 0.62 is
already in this tree, and enabling `Win32_Foundation` on a second major version to
reach one type alias would have put a whole duplicate `windows` family in the
build. With the patch, the only crate `stark-pen` adds to the graph for Wintab is
`libloading`.

Straightforwardly upstreamable, and worth upstreaming: it is a fix for a crate
that is currently unusable outside its own repository, and it removes a
dependency rather than adding one.

## Patch 2 — export the types a packet's own fields are typed as (sites marked `STARK PATCH`)

`src/lib.rs`: add `TPS`, `Orientation` and `Rotation` to the `pub use packet::{..}`
list.

### Why

`Packet::pkStatus` is typed as `TPS` and `pkOrientation` as `Orientation`, and
neither name is exported — so a consumer holding a `Packet` cannot name the type
of a field it is being handed. Reading the eraser bit meant either transcribing
`0b10000` at the call site, which is a second copy of a declaration this crate
already makes, or this line.

Upstreamable, and an oversight rather than a decision: `ButtonChange` beside them
is exported and is the one type in that module the docs describe as unused.

## What is *not* patched

The lints. This crate raises 29 warnings under our lint set — non-snake-case
parameter names transcribed from `WINTAB.H`, which is precisely what a faithful
binding should do — and it is excluded from the workspace, so nothing asks it to
change. Same stance as `vendor/mixbox`.

## Running its tests

Excluded from the workspace, so run them by hand when the vendored code changes:

```sh
cargo test --manifest-path vendor/wintab_lite/Cargo.toml --features libloading
```
