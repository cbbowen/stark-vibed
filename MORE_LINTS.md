# More lints

A survey of which additional clippy/rustc lints Stark would benefit from, and —
more usefully — which it would not. Measured, not guessed: one
`cargo clippy --workspace --all-targets` run with `clippy::pedantic`,
`clippy::nursery` and ~20 individually-named restriction and rustc lints, tallied
by lint name from the JSON diagnostics.

Surveyed 2026-08-22 against `clippy 0.1.98 (df6ee909ef 2026-06-28)`.

## Where the tree starts

There is no `clippy.toml` and no `[workspace.lints]`. The whole configuration is
`-D warnings` over the default lint set, in CI and in CLAUDE.md's command list.
That baseline is **green** — as is the second configuration
(`--no-default-features --features stark-net/webrtc`), which any lint added here
has to stay green under too.

The survey raises **6,623 warnings**, 6,264 of them inside `crates/`. So the
groups are out wholesale, and the question is which tail is worth taking.

## Tier 1 — zero warnings today

These cost nothing to adopt and are pure ratchets: they cannot break a build
today, only a future one. Each was confirmed at 0 across `--all-targets`.

| Lint | Why |
|---|---|
| `unsafe_code` | **The workspace contains no hand-written unsafe at all** — zero hits for `unsafe {`, `unsafe fn`, `unsafe impl`, `unsafe extern`. `-F unsafe_code` over `--workspace --all-targets` exits 0. |
| `clippy::dbg_macro`, `clippy::todo` | Debug leftovers, in a crate tree whose output is a browser tab |
| `clippy::mem_forget`, `clippy::rc_buffer`, `clippy::float_cmp_const` | 0 |
| `meta_variable_misuse`, `trivial_numeric_casts` | 0 |

`unsafe_code = "forbid"` is the one worth taking first. A GPU-facing,
wasm-targeting, 130k-line tree that has never needed `unsafe` should say so where
the compiler can hold it, rather than leave it as a fact nobody has written down
— which is the same argument as CLAUDE.md's "rule out a class rather than
enumerate its instances".

It survives `forbid` even though `stark-shaders/build/mirror.rs` generates
`#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]`, and bytemuck's derive
expands to an `unsafe impl`: the lint does not see through an external proc
macro's expansion. So the claim `forbid` makes is "no unsafe *we* wrote", not "no
unsafe in the build". That is the claim worth making, but it is worth knowing it
is that one and not the stronger one.

## Tier 2 — a `clippy.toml` disallowed list

The highest-value item here, because it converts rules that currently live only
in reviewers' heads into build errors.

```toml
disallowed-types = [
  { path = "std::time::Instant", reason = "panics in a browser; use quanta::Clock (§7.1, stark_engine::timing)" },
]
```

Prototyped and confirmed firing — on `stark-net/tests/sync.rs:122`, `:125`,
`:319`, `:322` among others.

`std::time::Instant` is the sharpest case in the tree: it *compiles* for
wasm32 and then panics in the browser, so nothing before a hand-run of the app
catches it. The `quanta` dependency exists for exactly this reason
(`stark-engine/Cargo.toml`), and `timing.rs` says so in prose — but prose does not
fail a build.

All 16 present uses are legitimate: benches, examples and native-only tests. So
adoption costs roughly five `#[expect(clippy::disallowed_types, reason = "…")]`
annotations, and those annotations become the record of *why* it is allowed at
each site, which is better than the silence there is now.

The mechanism generalizes. `disallowed-methods` takes the same shape, and any
other API that is a hazard on one target and fine on another belongs in this list
rather than in a memory.

## Tier 3 — small, real cleanups

Single-digit-to-low counts, each a genuine improvement rather than a style
preference:

| Lint | In `crates/` | Note |
|---|---|---|
| `clippy::allow_attributes` | 8 | Converts `#[allow]` to `#[expect]`, so a suppression that stops being needed **warns**. There are 0 `#[expect]` in the tree today and 8 `#[allow(clippy::…)]`. |
| `clippy::allow_attributes_without_reason` | 10 | Forces the `reason = "…"`, which matches how everything else here is commented |
| `clippy::missing_assert_message` | 9 | |
| `clippy::redundant_clone` | 7 | |
| `elided_lifetimes_in_paths` | 5 | |
| `trivial_casts` | 3 | |
| `clippy::unnecessary_wraps` | 3 | |
| `clippy::needless_collect` | 3 | |
| `clippy::while_float` | 2 | Float loop counters — `document/fill.rs:138`, `tests/minify.rs:124` |
| `clippy::useless_let_if_seq` | 2 | |
| `clippy::iter_with_drain` | 2 | |
| `clippy::maybe_infinite_iter` | 2 | |

`clippy::allow_attributes` is the pick of these. An `#[allow]` is permanent by
construction: it goes silent when the thing it suppressed disappears, and nothing
ever removes it. `#[expect]` is the same suppression that reports itself once it
is unnecessary — the same property CLAUDE.md wants from "do not add inert
scaffolding", applied to the suppressions themselves.

## What the survey argues *against*

The more useful half. Each of these looks plausible for a graphics codebase and
is not.

**The cast family — 949 warnings.** `cast_precision_loss` 562,
`cast_possible_truncation` 281, `cast_sign_loss` 106. There are 972 `as` casts in
the tree and they are what pixel-and-index code is made of. No.

**`clippy::suboptimal_flops` — 275, and actively harmful.** It pushes `mul_add`.
That *changes the rounding*, so taking its advice re-blesses goldens in exchange
for nothing, and `f32::mul_add` falls back to a slow software path on hardware
without FMA. This lint is pointed the wrong way for a renderer whose output is
compared bit-for-bit.

**`clippy::float_cmp` — 14 in-tree, all false.** Every site was read. They are
deliberate exactness guards: the lerp endpoint at `document/warp.rs:66` and `:71`,
the `max == r` branch of an HSV conversion at `peer.rs:1015`, an opacity identity
check at `document/apply.rs:509`. Adopting it buys 14 suppressions and no
findings. (`clippy::float_cmp_const` is a different lint and *is* at 0 — Tier 1.)

**`clippy::suspicious_operation_groupings` — 2, both false.** Worth recording
because it is the one lint in the survey that claims to find *bugs* and here finds
none. `document/guide.rs:935` is `axx * ayy - axy * axy`, the determinant of a
symmetric 2×2, which is correct; `geom.rs:180` is a textbook AABB overlap test.

**`missing_docs` (345 + 93) and `unreachable_pub` (782).** Both defensible in
principle and both too large to land as one change. Note that **359 of the
survey's warnings fall outside `crates/`** — generated mirror code and macro
expansion — and these two lints account for most of that, so adopting either means
teaching `stark-shaders/build.rs` to emit the suppressions as well.

**`clippy::mod_module_files` (16) / `clippy::self_named_module_files` (~16).**
The tree is genuinely mixed, so neither direction is free; this is a style choice
costing ~16 file moves either way, not a lint finding.

## Mechanics

- `[workspace.lints]` in the root `Cargo.toml`, plus `[lints] workspace = true`
  in each member — including `stark-ui`, which is outside `default-members`.
- `clippy.toml` goes at the workspace root and configures the *contents* of
  `disallowed-types` / `disallowed-methods`; the lints themselves are still
  enabled through `[workspace.lints]`.
- Anything added here has to be green under the `--no-default-features
  --features stark-net/webrtc` job too, not just the default build.
- `clippy::string_to_string` has been **removed** upstream ("`clippy::implicit_clone`
  covers those cases") — it warns as removed on this toolchain. Do not reach for it.

## Reproducing this

```sh
cargo clippy --workspace --all-targets --message-format=json -- \
  -W clippy::pedantic -W clippy::nursery \
  -W clippy::dbg_macro -W clippy::todo -W clippy::print_stdout -W clippy::print_stderr \
  -W clippy::undocumented_unsafe_blocks -W clippy::unwrap_used -W clippy::expect_used \
  -W clippy::missing_assert_message -W clippy::float_cmp_const -W clippy::mod_module_files \
  -W clippy::allow_attributes -W clippy::allow_attributes_without_reason \
  -W clippy::rc_buffer -W clippy::str_to_string \
  -W missing_docs -W unreachable_pub -W unused_qualifications \
  -W missing_debug_implementations -W trivial_casts -W trivial_numeric_casts \
  -W elided_lifetimes_in_paths -W meta_variable_misuse \
  > clippy.json
```

then tally `message.code.code` over the `compiler-message` records, deduplicating
on (lint, file, line, column) — cargo replays diagnostics per target, so the raw
count double-reports anything reachable from two targets.

## Status

Nothing here is implemented. The proposed first change is Tier 1 + Tier 2 + the
`#[allow]` → `#[expect]` conversion, which is around 20 touched lines plus the
manifest wiring.
