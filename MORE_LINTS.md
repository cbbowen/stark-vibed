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

## A second round: four more, none of them free

Asked for specifically, on the grounds that a lint can be worth paying for if it
is catching something today. Each was read site by site. **None is adopted**, and
one of the four found a real thing anyway, which is recorded at the end.

### `clippy::manual_midpoint` — 20 sites, safe and empty

The interesting question is whether `a.midpoint(b)` changes any pixels, since
several sites are on the stroke path. Measured rather than assumed: over 800,000
random pairs, `(a + b) * 0.5` is **bit-identical** to a correctly-rounded midpoint
across both the coordinate range (±4096) and the unit range that `u`, pressure and
`curve` live in. Halving is exact in binary, so the double rounding people worry
about does not happen. The two diverge only when `a + b` overflows to infinity —
`midpoint(2e38, 2e38)` is finite where the naive form is not — and a canvas
coordinate cannot get there.

So: no golden re-bless, and no finding either. What is left is a rename across 20
sites, and at several of them it reads *worse*, because the arithmetic is not a
midpoint. `geom.rs:51` is half the trace of a symmetric 2×2 on the way to its
eigenvalues; `geom.rs:97` and `assist.rs:312` are the mean of two radii. Those are
averages, and `sxx.midpoint(syy)` says something the code does not mean.

### `clippy::imprecise_flops` — 11 sites, all one shape, wrong trade

Every one of the eleven is `(a * a + b * b).sqrt()` asking to become `a.hypot(b)`,
and four are on the stroke hot path (`segments.rs` ×3, `path.rs:1206`). What
`hypot` buys is the internal rescaling that keeps `a * a + b * b` from overflowing
or flushing to zero before the root — a real property, and one that bounded canvas
coordinates cannot exercise. Paying for it per segment is the wrong way round.

The part worth writing down is where this lint *would* have earned its place. The
deposit law is `1 − exp(−k·τ)` (§6.2, and CLAUDE.md's list of things easy to break
silently), and `exp_m1` is meaningfully more accurate than `exp() - 1` exactly
where τ is small — which is every lightly-loaded stamp. But that law lives in
WESL, and clippy does not read shaders. The one place the lint's advice is worth
taking is the one place it cannot look.

### `clippy::significant_drop_tightening` — 14 sites, and the one real finding

Half of them (7) are in `stark-shaders/build/mirror.rs`, which is single-threaded
code generation in a build script — a guard held a few lines long costs nothing
there. And it flags `gpu/stroke/scratch.rs:144`, where the author has *already*
tightened by hand, wrapping the locked section in an explicit block so
`create_texture` happens outside it. Being flagged anyway is the lint failing at
the one case that proves it works.

But it found one thing. **`AssetStore::load` holds `self.inner.lock()` across a
PNG encode and two GPU texture builds** — `build_prefix_tau` and
`build_coverage_r8` (`assets.rs:101`). The lock is taken before
`inner.masks.entry(id)` and the whole bake happens inside the `Vacant` arm,
because the entry API holds the borrow.

Not a bug today: the callers are the engine's own file path (`engine/file.rs`),
which is the actor, so nothing contends for it. Worth knowing all the same, since
"the asset store is locked for the duration of a GPU bake" is the kind of fact
that stops being free the moment a second thread imports something. Tightening it
means a check-compute-recheck around the cache insert, which is a real change to
correctness-sensitive code and not something a lint should be trusted to motivate
on its own — so it is recorded here rather than done.

### `clippy::manual_let_else` — 3 sites, the cheapest and the emptiest

Pure style, no behaviour, three sites, one of them in a build script.
`panels/transform.rs:229` genuinely does read better as
`let TransformUi::Affine { rect, .. } = ui else { unreachable!(..) }`. This is the
only one of the four with no argument against it — and none for it either, beyond
taste. Left out on the same standard as the rest of this document: a lint earns a
row by catching something.

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

**Implemented.** All three tiers landed on `more-lints`, in seven commits. The
lint set lives in `[workspace.lints]` in the root `Cargo.toml`, inherited by all
seven members; the disallowed list is `clippy.toml` at the workspace root.

Adopted, twenty lints:

| Tier | Lints |
|---|---|
| 1 | `unsafe_code` (forbid), `meta_variable_misuse`, `trivial_numeric_casts`, `clippy::dbg_macro`, `clippy::todo`, `clippy::rc_buffer`, `clippy::mem_forget`, `clippy::float_cmp_const` |
| 2 | `clippy::disallowed_types` / `disallowed_methods`, configured with `std::time::Instant`, `std::time::SystemTime`, `std::thread::sleep` |
| 3 | `clippy::allow_attributes`, `allow_attributes_without_reason`, `while_float`, `useless_let_if_seq`, `needless_collect`, `missing_assert_message`, `redundant_clone`, `trivial_casts`, `elided_lifetimes_in_paths` |

### What adopting them actually found

The survey predicted these would be cheap. It did not predict that three of them
would find something:

- **Three of the ten `#[allow]`s were already dead.** `clamp01`'s `manual_clamp`
  waiver never fired (clippy skips the lint in a const context, and `f32::clamp`
  is not const); `session::finish`'s `too_many_arguments` covers seven arguments
  against a threshold of eight; and `base64_decode`'s `cfg_attr` claimed the
  function is dead off wasm32 when it is dead off wasm32 *and* outside
  `cfg(test)`. An `#[allow]` cannot report any of that, which is the argument for
  `#[expect]` in one paragraph.
- **`redundant_clone` was right seven times out of seven**, three of them on a
  path that runs every render. Removing a clone either compiles or it does not,
  so the borrow checker did the adjudicating.
- **`missing_assert_message`'s nine sites each already had a comment** above them
  naming the invariant. None of it reached the failure output.

### Three Tier 3 entries did not survive inspection

Listed above on their counts; withdrawn once read. Recorded here because the
count was never the evidence:

- **`clippy::iter_with_drain`** — its two hits are `drain(..)` on a `Vec` that is
  refilled on the next pass, which is what `drain` is *for*. The suggested
  `into_iter()` fails to compile (E0382, borrow of moved value).
- **`clippy::maybe_infinite_iter`** — both hits are the
  `(1..).find(|n| the name is free)` behind "Gradient N" and "Preset N", which
  terminates after N+1 entries. Two `#[expect]`s, no finding.
- **`clippy::unnecessary_wraps`** — two of its three hits are signatures owed to
  something outside the function. `platform::stub::download_bytes` mirrors a wasm
  sibling that genuinely returns `Err`, and `desc::blended_target`'s `Option` *is*
  wgpu's `targets: &[Option<ColorTargetState>]` slot type, passed straight through
  by 16 call sites. One real fix against two permanent `#[expect]`s.

So the fuller lesson is the one the "argues against" section was already making:
a lint's count says how much work it is, never whether the work is worth doing.
Of the twelve Tier 3 lints proposed on counts alone, nine held up and three did
not — and the three were only distinguishable by reading every site.

### Verification

Every commit is green under both clippy configurations (default, and
`--no-default-features --features stark-net/webrtc`), the wasm32 check, and
`cargo fmt --all --check`. The two rewritten float loops are covered by the fill
test's own `steps > 3000` assertion and by the two minify goldens, which compare
pixels and so would have caught a moved stroke. `cargo nextest run --workspace`
was run once over the finished branch.

Two guards were checked in the failing direction rather than assumed: a
`std::time::Instant` added to `stark-model` fails the build (exit 101), and
`cargo clippy -v` shows all the flags reaching rustc rather than sitting in a
manifest nothing reads.
