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

## Phase 0 — Lock the baseline

**0.1 Add `ci.yml`.** fmt-check, `clippy -D warnings`, `cargo test --workspace`,
and a `wasm32-unknown-unknown` build of stark-ui. This has to land *before* the
mechanical passes, or the reformat commit can hide a regression.

**0.2 Fix the silent-skip hole.** [common/mod.rs:29-38](crates/stark-core/tests/common/mod.rs#L29-L38)
— `engine_or_skip` prints and returns `None` when there's no adapter, so the test
**passes vacuously**. On a stock GitHub runner every golden, seam, dynamics and
selection test would go green having rendered nothing. Either install lavapipe in
CI, or make skipping opt-in (`STARK_ALLOW_NO_GPU=1`) so a missing adapter fails by
default.

## Phase 1 — Mechanical, zero behavior change

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

## Phase 2 — The big one: spline-fit is ~55% unreachable

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

Two ways out, and it's a genuine fork:

- **(a) Adopt** — rebuild `PathFitter` on `IncrementalFit`. Keeps the library
  whole, deletes the duplicated loop. But the fit is *heavily* tuned (the
  `KNOT_COST` measurement table, the `arc_profile` reparameterization, the
  solve/score consistency argument in `mean_error`) — behavior moves, goldens
  re-bless.
- **(b) Prune** — delete the unreachable half, shrink spline-fit to the primitive
  actually used. Zero behavior change.

**Recommendation: (b).**
[path.rs:492-500](crates/stark-core/src/path.rs#L492-L500) documents a deliberate
move *away* from searched correspondence — "the reordering that makes a searched
correspondence dangerous is ruled out by construction." `IncrementalFit` and the
assignment search are an abandoned approach, not a target. Git keeps them if
wanted.

**2.1 Nightly.** [spline-fit/src/lib.rs:1](crates/spline-fit/src/lib.rs#L1) is an
*unconditional* `#![feature(generic_const_exprs)]` (an incomplete feature), which
— along with `history`'s `associated_type_defaults` — is why the whole workspace
is pinned to nightly. It's needed only for `{P + 1}` arithmetic, and `Const<3>` is
the sole instantiation. Monomorphizing to cubic removes one of the two nightly
blockers. Also: `rust-version = "1.85"` in the workspace manifest is simply false
today.

## Phase 3 — Module surgery (behavior-preserving)

**3.1 `testdata` is shipped in the library.**
[testdata/mod.rs](crates/stark-core/src/testdata/mod.rs) is 2,384 lines / 57 KB of
recorded stroke literals, declared `pub mod` in
[lib.rs:102](crates/stark-core/src/lib.rs#L102) — public API, compiled into every
build including the wasm the browser downloads. Consumers are `tests/stroke.rs`
and one unit test in path.rs. Move it to a `stark-testdata` dev-only crate.

**3.2 `stark-ui/src/main.rs` (1,645 lines)** holds eight unrelated concerns. The
seams are already there: `panels/` (color + Oklab picker + BMP/base64 encoder,
brush, select, layer, lighting), `layout.rs` (`PanelLayout`, `DragState`,
`drag_*`, `Panel`, `PanelStack`), `widgets.rs` (`Slider`, `TernaryPad`),
`input.rs` (`handle_keydown/keyup`, `sample`, `dispatch`, `capture_pointer`).
Also collapse the four loose `collab_*` signals in `AppState` into one
`CollabState`.

**3.3 `gpu/tile.rs` — cash the author's own TODOs.** Four of them
([lines 89, 96, 174, 181, 195](crates/stark-core/src/gpu/tile.rs#L89)) all say the
same thing: retire `TilePair`/`TilePairHandle`/`acquire`/`acquire_scratch` in
favor of `acquire_tex` handles. This ripples into `DocState` and `stroke.rs`, so
it's the biggest of the three — but it's a design the author already committed to
in writing.

**3.4 `gpu/stroke.rs` (2,493 lines).** `render_dynamic` is a single ~600-line
function, and it re-does `render_swept`'s prefix-tau bind group, noise bind group,
and segment-instance construction. Extract those three, then split
`render_dynamic` into setup / segment loop / writeback. A `stroke/` directory
(`segments.rs`, `dynamics.rs`, `swept.rs`) falls out naturally.

**3.5** Five `#[allow(clippy::too_many_arguments)]` sites (engine.rs:1054,
composite.rs:781, selection.rs:406, stroke.rs:607 and :959) → parameter structs.

## Phase 4 — Architecture consistency (needs decisions)

**4.1 Two parallel input APIs.** DESIGN §4 calls `InputCommand` "the most
important boundary," but ~18 engine mutations bypass it entirely as direct
`pub fn`s — `set_color_space`, `set_surface`, `set_environment`,
`register_surface`, `set_media_params`, `import_brush`, the four collaboration
hooks, `resize`, the replay family. `set_surface` is the sharp case: it mutates
`surface_id`, which is **saved in `CanvasMeta`**, from outside the logged path.
Either fold these into `InputCommand` or document a deliberate second
"configuration" tier in DESIGN §4.

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

## Sequencing

Phase 0 → 1 are low-risk and unblock everything. Phase 2 is the biggest single
win and is **independent** of 3 and 4, so it can run in parallel. Phase 3 items
are independent of each other; 3.3 is the riskiest and should go last within that
phase. Phase 4 needs decisions before code. Phase 5 can happen any time.

Open questions: **Phase 2, prune or adopt?** (recommendation: prune), and
**Phase 4.1** — is the direct-method tier intentional, or should it collapse into
`InputCommand`?
