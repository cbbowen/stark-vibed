# Core cleanup

A review of `crates/stark-core/src` **outside** `document/` and `gpu/`, which had
already been reviewed separately. Same shape as the retired `GPU_CLEANUP.md`: a
working checklist, to be deleted once it is empty, because the reasoning for each
change belongs in the commit that made it and in the doc comments around the code
it explains.

Ordered by consequence, not by effort.

---

## §1 — `refresh_live` retains frozen heads for actors who stopped gesturing

**Status: done.**

`Engine::refresh_live` takes the whole `heads` map, `remove`s and re-`insert`s
entries for the actors that *are* gesturing, and writes the map back. An actor
absent from `live_gestures()` is never touched, so its entry survives.
`self.heads.clear()` only runs when *nobody* is gesturing, so the retention is
reachable exactly when a shared session has one peer lift while another keeps
painting.

A `FrozenHead` owns a whole `DocState` — `Arc<GpuTile>` handles that would
otherwise return to the pool — so the cost is pinned GPU memory, held precisely
while two people are painting and the frame budget is already spoken for.

The fix is not a `retain`: build a *fresh* map and let anything not re-inserted
drop by construction, so "a head outlives its gesture" stops being a thing a
future edit can reintroduce (CLAUDE.md — rule out a class rather than enumerate
its instances).

## §2 — `engine.rs` is five modules in one file

**Status: done.** Six files: `mod`, `render`, `live`, `pick`, `collab`, `file`.
Verified as a pure move — the function inventory is byte-identical before and
after, and the only body differences are the `mod`/`pub use` lines, the import
reshuffle, and twelve items widened from private to `pub(super)` because they are
now called across the seam.

3193 lines, one `impl Engine` block, 57 public methods. The seams are already
visible in the file's own section comments:

| Concern | Members |
|---|---|
| Present & export | `Background`, `Chrome`, `Attachments`, `Rendered`, `ExportScale`, `ExportPlan`, `render*`, `export*`, `composite_groups`, `composite_stack`, `layer_items`, `visible_tiles` |
| Eyedropper | `PickSource`, `PickOptions`, `pick_color`, `pick_target`, `mean_channels`, `mean_over_substrate` |
| Live fold | `FrozenHead`, `refresh_live`, `live_gestures`, `render_live_stroke`, `advance_head`, `render_span_range`, `overlay_tiles`, `presented`, `set_doc_preview`, `preview_transform` |
| Collaboration & presence | `start`/`join`/`end_collaboration`, `merge_remote`, `take_outbox`, `PresenceTick`, `take`/`merge`/`leaving_presence` |
| File lifecycle | `document_file`, `referenced_surfaces`, `load_document`, `load_surfaces`, `replay_timelapse`, `reset_document`, `resync_counters`, `new_document` |

The property that makes this cheap: **it needs no visibility changes.** A child
module of `engine` can reach `Engine`'s private fields, so `engine/render.rs`,
`engine/live.rs`, `engine/collab.rs` and `engine/file.rs` each carrying an
`impl Engine` block is pure relocation — no field moves, no `pub(crate)`.

## §3 — Make the preview epoch structural

`live`, `doc_preview`, `heads` and `doc_epoch` are one subsystem with one
invariant — *a head stamped with an older epoch is stale* — but the invariant is
enforced in four places (`committed_changed` and `set_doc_preview` bump it;
`refresh_live` and `render_live_stroke` check it), and `refresh_live` is called
from some fifteen sites on a remember-to-call-it discipline.

`committed_changed`'s own doc comment already makes this argument for the two
*counters*. It applies one level up: a `Preview` type owning those four fields,
whose only mutators are `invalidate()` and `rebuild()`, would make §1
unrepresentable rather than fixed.

## §4 — Three near-copies of "adopt a file"

**Status: done.** The dedup turned up two further drifts beyond the one already
recorded below: the timelapse never matched the document's *colour space* either,
so a Mixbox document replayed through Oklab's shaders; and it bound the ground
once before the loop rather than per action, so a timelapse across a mid-document
`SetSurface` went on lighting every later frame through the weave the piece
started on.

`load_document`, `join_collaboration` and `replay_timelapse` each run the same
preamble: set `initial_surface`, `reset_document`, maybe `rebuild_gpu_for`,
install assets, `load_surfaces`, replay, `resync_counters`,
`apply_document_surface`.

They have already drifted once, and the comment in `replay_timelapse` records it:
it *missed* the `initial_surface` step, so every frame before the log's first
`SetSurface` was deposited against the wrong weave. That is the bug shape that
recurs while the sequence is written out three times. `replay_timelapse` still
differs a second way — it swallows asset errors with `let _` where the other two
log them.

## §5 — `GESTURE_RESYNC = None` is inert scaffolding, and its test is vacuous

**Status: done**, by separating the two questions. The *cadence* stays deferred —
it needs a loss-rate and latency measurement on a real transport, which this
crate cannot make — but the constant now says so, and the *mechanism* is tested
regardless of what the cadence is set to: `encode` already takes `resync` as a
parameter, so the repair is driven through `GestureTx`/`GestureRx` directly
instead of through `Session::publish`, which consults the constant. Turning it on
is a one-line change that cannot be the thing that first tells us whether the
repair works.

`peer::GESTURE_RESYNC` is `Option<f64> = None`, which makes `GestureTx::resync_due`
constantly false, `stamp_resync` unreachable, `encode`'s `resync` parameter
constantly false, and `GestureRx`'s whole resync branch — including the
frozen-watermark carry-over that a bug fix and a test were written for —
unreachable from an honest sender.

`presence::tests::a_resync_makes_the_receiver_exact` wraps its entire assertion
body in `if let Some(interval) = GESTURE_RESYNC`, so it passes having checked
nothing. That is the failure mode CLAUDE.md calls out for `STARK_ALLOW_NO_GPU`:
a test that reports `ok` having rendered nothing.

Either pick an interval (the deferral was pending a lag evaluation), or
`#[ignore]` the test with the reason — but it must not report green while
asserting nothing.

## §6 — Inert scaffolding, smaller

**Status: done.**

- `Engine::process_gesture`'s `Start` arm clears and pushes into `debug_samples`
  unconditionally, while `To` gates on `cfg!(feature = "debug-unfrozen")` and
  `log_debug_samples` early-returns without it. A shipping build allocates a
  one-element `Vec` per stroke and never reads it.
- `assets::Mask`'s two `#[allow(dead_code)]` texture fields exist to keep
  textures alive, but a wgpu view holds its own reference:
  `gpu/stroke/mod.rs` drops both with `let (_tex, …)` and the round tip renders.
  `build_prefix_tau` and `build_coverage_r8` should return the view alone,
  deleting both fields and both `_tex` bindings.
- `Engine::new_with_color_space` has `let _environment_id = EnvironmentId::default();`,
  which computes nothing.

## §7 — Dead public API

**Status: done.**

Zero callers anywhere in the workspace, tests included:

- `Peer::live_selection` and `Peer::gesture_id` — both superseded by `gesture_view`
- `Peers::len`
- `Session::cancel_selection` — `cancel_stroke` already clears both slots

(`Peer::gesture`, `live_stroke`, `live_frozen_spans` and `Peers::get`/`is_empty`
are test-only but reached from `tests/`, so they stay `pub`.)

## §8 — Small, cheap wins

**Status: done.**

- `PerspectiveGuide::name` is `Option<String>`, and `observe()` clones the whole
  guide list on every command — including `GestureCommand::To`, at pointer rate.
  `LayerInfo::name` is `Arc<str>` for exactly this reason and says so in its doc
  comment; the two should agree.
- `mean_channels` and `mean_over_substrate` open with the identical
  sum-over-chunks loop. One `sum_texels` helper, then the two divergent tails.
- `render_offscreen` and `pick_target` build the same `TextureDescriptor` with
  different labels and usages.
- The `Undo` and `Redo` arms of `process_doc_inner` are token-identical apart
  from which `*_as_action`/navigate pair they call.
- `image.rs` has two adjacent `impl RgbaImage` blocks with nothing between them.
- `(Vec2, Vec2, f32)` means "centre, semi-axes, angle" in `assist::moments`,
  `guides::ellipse_of` and `AxisPlane::circle_seen`. The two modules already
  share `principal_axis` via `geom`; a named `Ellipse` beside it would finish
  that move and rescue `assist::settled` from `a.0`/`a.1`/`b.2`.

---

## Deliberately not changed

`geom.rs`, `spline.rs`, `noise.rs`, `colorspace.rs`, `io.rs` and `presence.rs`
are cohesive and well-sized. `presence.rs` in particular earns its shape: both
protocol halves in one module, the pair invariants listed in the module doc, and
a round-trip test that drives one end into the other through a dropping,
duplicating, delaying channel.

`path.rs`, `assist.rs` and `guides.rs` are large but each is one subject, and
`Scaffold` is a clean one-way seam between the last two.
