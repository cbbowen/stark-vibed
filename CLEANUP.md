# Cleanup & Refactor Plan

A broad cleanup pass over the workspace, ordered by risk and leverage. Phases 0
and 1 are mechanical and safe; 2 is the biggest single win; 3 is module surgery;
4 needs design decisions; 5 is documentation rot.

## Baseline

Measured against `daf29d2` (2026-07-26):

| Check | Result |
|---|---|
| `cargo test --workspace` | **256 tests, 0 failures** |
| `cargo clippy --workspace --all-targets` | 18 warnings, **all in stark-ui** (12x `useless_format`, 1 `needless_borrow`, 1 `diverging_sub_expression`) |
| `cargo fmt --all --check` | **309 hunks / ~4,250 diff lines across ~45 files** |
| Size | 26,951 lines Rust in 5 crates, + 17.6k vendored |
| CI | `pages.yml` only — **nothing runs test, clippy, or fmt** |

Tests are green, so refactoring has a real safety net. The two things that
undercut it: nothing enforces that net, and GPU tests self-skip.

---

## Phase 0 — Lock the baseline — **DONE**

**0.1 Add `ci.yml`.** fmt-check, `clippy -D warnings`, `cargo test --workspace`,
and a `wasm32-unknown-unknown` build of stark-ui. This has to land *before* the
mechanical passes, or the reformat commit can hide a regression.

**0.2 Fix the silent-skip hole.** [common/mod.rs:29-38](crates/stark-core/tests/common/mod.rs#L29-L38)
— `engine_or_skip` prints and returns `None` when there's no adapter, so the test
**passes vacuously**. On a stock GitHub runner every golden, seam, dynamics and
selection test would go green having rendered nothing. Either install lavapipe in
CI, or make skipping opt-in (`STARK_ALLOW_NO_GPU=1`) so a missing adapter fails by
default.

## Phase 1 — Mechanical, zero behavior change — **DONE**

1. `cargo fmt --all` as one isolated commit + a `.git-blame-ignore-revs` entry.
2. `cargo clippy --fix` handles 13 of the 18; hand-fix
   [main.rs:1618](crates/stark-ui/src/main.rs#L1618) and
   [render.rs:290](crates/stark-ui/src/render.rs#L290).
3. Delete confirmed-dead code: `GpuContext::require`
   ([context.rs:76](crates/stark-core/src/gpu/context.rs#L76), commented "reserved
   for future"), and the `approx` dependency in spline-fit (**zero references** in
   src or benches).
4. Manifest hygiene: spline-fit is the odd one out — no
   `edition.workspace`/`license.workspace`, duplicates `nalgebra = "0.35"` instead
   of `workspace = true`, no `publish = false`, and it's the only underscore
   package name (`spline_fit`) in a hyphen workspace.

### Found while doing Phases 0–1

Three things the initial survey missed, all now handled or flagged:

- **The vendored trees were workspace members.** Cargo promotes an unexcluded
  path dependency to a member, so `cargo fmt --all` reformatted `vendor/mixbox`
  and `vendor/iroh-webrtc-transport`, and `clippy --workspace` raised 50 warnings
  in code that is not ours to fix. Both are now in `[workspace] exclude`. They
  still build; their own test suites no longer run under `cargo test
  --workspace` (which is why the workspace total drops from 200 to 148 across
  Phases 1–2: −52 vendored, −62 spline-fit, +10 ported spline tests). Run them with
  `cargo test --manifest-path vendor/iroh-webrtc-transport/Cargo.toml` if that
  coverage is wanted back — **open question**, since the iroh 1.0 port is
  precisely what those tests guard.
- **The baseline clippy count of 18 was low.** Clippy only reports for crates it
  recompiles, and the first run was against a warm cache. A cold workspace build
  raised 31 in `crates/` (plus 50 in vendor). All now cleared.
- **`TernaryPad` is orphaned.** The component, its three `TRI_*` constants,
  `ternary_weights`, and ~50 lines of `.ternary*` CSS in stark.css are complete —
  and nothing renders it. Marked `dead_code` with a pointer here rather than
  deleted. **Open question:** wire it up or remove it (including the CSS)?

## Phase 2 — spline-fit was ~55% unreachable — **DONE**

This is the highest-leverage finding. `path.rs` is spline-fit's **only** consumer,
and it calls exactly four things: `from_control_points`, `num_spans`, `evaluate`,
`fit_channels_smoothed`. And
[`fit_channels_smoothed` delegates straight to `m_step`](crates/spline-fit/src/lib.rs#L1628-L1650),
never touching the E-step. So unreachable from the sole consumer:

- `IncrementalFit` — [incremental.rs](crates/spline-fit/src/incremental.rs), 520
  lines, publicly exported, **never used**
- [poly.rs](crates/spline-fit/src/poly.rs) — 561 lines, only reached via the
  closest-point search
- `span_polys` / `locally_closest_*` / `all_critical_points` /
  `best_ordered_assignment*` / `candidates_in_ranges` /
  `assignment_from_candidates` — ~900 lines
- `fit_monotonic`, `refit_monotonic`, `fit_monotonic_adaptive`,
  `extend_control_points`, `polyline_init`, `e_step`, `em_fit_in_place`,
  `Settled`, `VariableTolerance` — ~350 lines

**~2,300 of 4,150 lines serve no caller.** Meanwhile
[`PathFitter`](crates/stark-core/src/path.rs#L249-L585) hand-rolls its own
grow/solve/freeze/adopt loop — the exact job `IncrementalFit` exists to do.

**Resolution: pruned, and then some.** The ~600 lines actually used are now
[spline.rs](crates/stark-core/src/spline.rs) in stark-core, monomorphized to `f32`
and to cubic, and the crate is deleted. **4,150 lines became 430.**
[path.rs:492-500](crates/stark-core/src/path.rs#L492-L500) documents a deliberate
move *away* from searched correspondence — "the reordering that makes a searched
correspondence dangerous is ruled out by construction" — so `IncrementalFit` and
the assignment search were an approach the fitter had already abandoned, not a
target to adopt.

Numerically identical by construction: path.rs already instantiated the fit at
`f32`, and the two places the old code narrowed from `f64` (`1/q` in the basis
recurrence, and the smoothing weight) still do. Goldens unaffected.

62 crate tests became 10 focused ones; the ~50 dropped covered deleted machinery.
One ported test needed its bound loosened from `1e-6` to 1%: it used to run in
`f64`, and the proximal ridge `n · √ε` is ~2300x stronger in `f32`. That was
always true of the engine's own solve — only the test was ever in `f64`.

**2.1 Nightly — half resolved.** `generic_const_exprs` went with the crate (the
`{P + 1}` basis arithmetic was its only use), so `history`'s
`associated_type_defaults` is now the **sole** reason the workspace is on nightly.
Still outstanding: `rust-version = "1.85"` in the workspace manifest is false.

## Phase 3 — Module surgery (behavior-preserving) — **DONE**

**3.1 `testdata` is shipped in the library.**
[testdata/mod.rs](crates/stark-core/src/testdata/mod.rs) is 2,384 lines / 57 KB of
recorded stroke literals, declared `pub mod` in
[lib.rs:102](crates/stark-core/src/lib.rs#L102) — public API, compiled into every
build including the wasm the browser downloads. Consumers are `tests/stroke.rs`
and one unit test in path.rs. Move it to a `stark-testdata` dev-only crate.

**Done.** Now unreachable from a non-test build by the dependency graph rather
than by hoping the linker strips it.

**3.2 `stark-ui/src/main.rs` (1,645 lines)** holds eight unrelated concerns. The
seams are already there: `panels/` (color + Oklab picker + BMP/base64 encoder,
brush, select, layer, lighting), `layout.rs` (`PanelLayout`, `DragState`,
`drag_*`, `Panel`, `PanelStack`), `widgets.rs` (`Slider`, `TernaryPad`),
`input.rs` (`handle_keydown/keyup`, `sample`, `dispatch`, `capture_pointer`).
Also collapse the four loose `collab_*` signals in `AppState` into one
`CollabState`.

**Done.** 1,724 lines -> 499, split exactly as above; `CollabState` grouped.

**3.3 `gpu/tile.rs` — cash the author's own TODOs.** Four of them
([lines 89, 96, 174, 181, 195](crates/stark-core/src/gpu/tile.rs#L89)) all say the
same thing: retire `TilePair`/`TilePairHandle`/`acquire`/`acquire_scratch` in
favor of `acquire_tex` handles. This ripples into `DocState` and `stroke.rs`, so
it's the biggest of the three — but it's a design the author already committed to
in writing.

**Done**, and it turned up a latent bug: `acquire` hardcoded `Rg16Float` for the
aux channel while the pool was built from `cs.aux_format()`. Harmless today (every
colour space returns that), a panic the moment one didn't. The pairing now reads
the formats off the colour space in use.

**3.4 `gpu/stroke.rs` (2,493 lines).** `render_dynamic` is a single ~600-line
function, and it re-does `render_swept`'s prefix-tau bind group, noise bind group,
and segment-instance construction. Extract those three, then split
`render_dynamic` into setup / segment loop / writeback. A `stroke/` directory
(`segments.rs`, `dynamics.rs`, `swept.rs`) falls out naturally.

**Done** — the directory split landed as described. The shared setup was smaller
than expected: only the prefix-tau *resolution* is common (now
`StrokeRenderer::prefix_view`). The bind groups around it are genuinely
different, because the two paths hang the texture off different layouts.

**3.5** Five `#[allow(clippy::too_many_arguments)]` sites (engine.rs:1054,
composite.rs:781, selection.rs:406, stroke.rs:607 and :959) → parameter structs.

**Done** — `StrokeScene`, `RasterShape`, `GpuBuild`, `OffscreenDesc`. Net fewer
lines: `StrokeScene` removed four parameters from four signatures, and
`OffscreenDesc` came with a `rebuild_offscreen` method that collapsed three
eight-argument call sites.

## Phase 4 — Architecture consistency (needs decisions)

**4.1 is DONE** — see the outcome note below it.

**4.1 Two parallel input APIs.** DESIGN §4 calls `InputCommand` "the most
important boundary," but ~18 engine mutations bypass it entirely as direct
`pub fn`s — `set_color_space`, `set_surface`, `set_environment`,
`register_surface`, `set_media_params`, `import_brush`, the four collaboration
hooks, `resize`, the replay family. `set_surface` is the sharp case: it mutates
`surface_id`, which is **saved in `CanvasMeta`**, from outside the logged path.
Either fold these into `InputCommand` or document a deliberate second
"configuration" tier in DESIGN §4.

**Done, and the framing was wrong.** The bypass was a symptom of two things:
`ObservableState` did not project the view settings kept as bare Engine fields
(so lighting.rs shadowed `MediaParams` from `Default`), and `InputCommand` mixed
document and view state so the distinction lived in comments. Now:
`DocCommand` / `ViewCommand` / `GestureCommand` put the state class in the type;
`observe()` projects media / surface / environment / colour space; `resize`,
`set_media_params` and `set_environment` became `ViewCommand`s. The surface moved
to document state (`ActionKind::SetSurface`) and the colour space became
`Engine::new_document(..)`. What is left outside the command path is a *named*
request tier — the operations that must answer — with the rule written down in
DESIGN §4. §7 now admits the actor is a target, not the present.

**4.2 Reverse dependency.**
[common/mod.rs:56](crates/stark-core/tests/common/mod.rs#L56) reads
`../stark-ui/assets/environment/ferndale_studio_11_1k.hdr`. DESIGN §2:
*"stark-ui depends on core, never the reverse."* Move the HDR to a neutral home.

**4.3 `Engine` is a 30-field god object** with ~50 public methods. Two clusters
are already visually obvious in the struct — `(surface, surface_id,
surface_assets)` and `(environment, environment_id, environment_assets)` are the
same shape twice: registered bytes + current id + live GPU object. One small
`Registry<Id, Gpu>` absorbs both. `(actor, clock, outbox, outbox_enabled)` is a
third.

**4.4** TODO.md's *"`Engine::apply_ctx` does a lot of cloning"* — it clones four
subsystems on every undo/redo/commit. Worth a borrow-based `ApplyCtx<'_>`.

## Phase 5 — Documentation rot

**5.1 DESIGN.md §6.3 does not exist.** It's cited **5x inside DESIGN.md itself**
and repeatedly across engine.rs, main.rs and the test harness as the
media/lighting pass — but the heading jumps §6.2 → §6.4. Restore it or repoint
every reference.

**5.2** DESIGN §2's file tree is missing ~9 modules that now exist:
`colorspace.rs`, `noise.rs`, `image.rs`, `error.rs`, `testdata/`,
`document/selection.rs`, `gpu/selection.rs`, `gpu/surface.rs`,
`gpu/environment.rs`.

**5.3** [lib.rs:1-86](crates/stark-core/src/lib.rs#L1-L86) is an 85-line
build-order changelog duplicating DESIGN §13 — two places to update, already
drifting. Keep the checklist in DESIGN.md; lib.rs gets a short orientation
paragraph.

**5.4** `surface_tooth` in `stamp_common.wesl` is a pass-through stub, so
`BrushParams::tooth` is historized but **inert**.
[stroke.rs:229](crates/stark-core/src/gpu/stroke.rs#L229) and `set_surface`'s doc
both know this. Either finish it or state plainly in DESIGN §6.4 that tooth is not
yet wired.

---

### Found while doing Phase 3

- **Tooling rewrote four files from LF to CRLF**, turning small changes into
  whole-file diffs. Caught before it reached more than two commits; those were
  rebuilt clean and `.gitattributes` now pins `text=auto eol=lf`. The pre-existing
  CRLF in the four root `.md` files is left alone.
- **`tests/tile_pool.rs` had the same silent-skip hole** Phase 0 fixed in the engine
  harness. Now closed the same way.

## Sequencing

Phases 0–3 are done. Phase 4 needs decisions before code. Phase 5 can happen any
time.

Open questions: **Phase 4.1** — is the direct-method tier intentional, or should it
collapse into `InputCommand`? — plus the two raised by Phases 0–1: whether to
restore the vendored crates' test coverage, and whether `TernaryPad` lives or dies.

## State after Phases 0–3

| Check | Before | After |
|---|---|---|
| `cargo fmt --all --check` | ~309 hunks | clean |
| `cargo clippy --workspace --all-targets -D warnings` | 31 warnings in `crates/` | clean |
| `cargo test --workspace` | 200 pass | 149 pass, 0 fail |
| `cargo check -p stark-ui --target wasm32-unknown-unknown` | (unchecked) | clean |
| Largest source file | 2,933 lines | 1,376 (`stroke/dynamics.rs`) |
| `#[allow]` in `crates/` | 10 | 5, each with a stated reason |
| Workspace members | 5 + 2 vendored | 5 |
| Nightly features needed | 2 | 1 (`history` only) |
| CI | pages deploy only | + fmt / clippy / test / wasm |

