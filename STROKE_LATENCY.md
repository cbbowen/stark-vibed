# Stroke latency — the end-to-end ledger

Investigated 2026-08-10 on post-revert master (`7b66760`); the stroke-space
march (branch `stroke-space-march`) is a recorded dead end and none of this
depends on it. This file records where the time between the pen and the screen
actually goes, and the ranked levers. Line references are of that date; the
mechanisms are the durable part.

## The chain as investigated

One `pointermove` during a stroke:

1. Dioxus delegated handler (`main.rs` `Canvas::onpointermove`) — synchronous.
2. `dispatch(GestureCommand::To)`: fit `push` + full live-tail GPU render +
   submit, inline in the handler (`state.rs` → `Engine::process` →
   `refresh_live`).
3. `obs.set(Some(r.observe()))` — walks the whole layer tree, marks every
   `obs` subscriber dirty (~39 sites, including the `Canvas` component).
4. `request_paint` — latches `paint_queued`, defers the paint to the next rAF.
5. Dioxus's scheduler polls tasks **only once no scope is dirty**, so the full
   VDOM diff + DOM flush from (3) runs before the rAF is even registered.
6. Next frame: rAF fires → *(before this branch)* two waker hops (JsFuture →
   scheduler channel → `queueMicrotask`), possibly behind another VDOM render,
   → `Renderer::paint`.
7. `paint` = `get_current_texture` → **full-viewport recomposite** (draw list
   rebuilt, a fresh bind group per visible tile per layer, passes A–E) →
   `present`, which is a **no-op on WebGPU** — the browser compositor takes the
   frame on its own schedule, typically 1–2 frames later, untunable
   (`PresentMode::Fifo` and `desired_maximum_frame_latency` are both dead
   values in wgpu's WebGPU backend).

Net: 2–4 display frames of scheduling latency on top of the GPU work, before
the browser's own compositor depth.

## The ledger

### Input side (`stark-ui`)

- **No `getCoalescedEvents()`** — Chromium delivers ~1 `pointermove` per
  frame; a 120–240 Hz pen's remaining samples are silently discarded before
  the fitter sees them. No `getPredictedEvents()`, no `pointerrawupdate`.
- **`InputSample.time` is always 0.0** — `input::sample` builds the sample
  with `..Default::default()` and `event.timeStamp` is never read. The
  fitter's time channel and anything velocity-derived operate on a constant.
- A mouse reports the spec's `pressure = 0.5` and is fed through as-is (the
  `InputSample` default would be 1.0) — a real mouse/pen behaviour difference.
- Per move, pre-fix: three separate `renderer.write()` borrows (dispatch,
  outbox flush, cursor publish), an unconditional `SetCursor` even solo, and
  the `observe()` walk + chrome diff of step (3) above.

### Engine per sample (`stark-core`)

- **Stroke render runs at input rate; presentation at frame rate.** The live
  tail render (deliberately per-sample for *integration* — samples must not
  drop) is also doing its GPU work per event for frames never shown.
- Dynamics live update: fresh region + reservoir ping-pong + snapshot
  textures and 7 bind groups allocated per update; the pen-up **settle chain
  runs every live update** (the price of `preview == committed`); reservoir
  passes sized by `BRUSH_RES = 64` regardless of tip, so small tips pay them
  most often.
- Small-radius strokes are **dispatch-bound**: segment length is capped by
  `RESERVOIR_EXCHANGE_STEP` (0.125 radii for a hard-trading brush, relaxed up to
  1 radius by `exchange_travel` for gentler ones), so r=8 runs hundreds of
  serialized dispatch chains — the 889 ms live gesture in the bench record.
  *(Correction 2026-08-10: this entry originally blamed `WICK_TRAVEL_QUANTUM` —
  "the wick may not be straddled" — but that coupling was march-only; master's
  wick fired on its own cadence inside a segment and never capped its length.
  What the wick did cost on master was its two serialized dispatches per
  half-radius of travel, since removed — see the record below.)*
- `overlay_tiles` re-inserts every tile the stroke has *ever* dirtied on each
  sample (grows with stroke length, not tail length). `path_as_finished` plus
  the whole control-point `Vec` and a full `StrokeRecord` clone per sample
  make per-stroke CPU O(n²).
- `Preview::rebuild` re-renders **every** in-flight gesture — every peer's
  tail, every live fill — on every local sample, with no per-gesture change
  gate.

### Presentation

- No damage tracking: an update that dirtied 2 tiles recomposites the whole
  viewport, rebuilding a bind group per visible tile per layer per frame
  (acknowledged in docs/rendering.md).
- At zoom < 1 an extra supersample target + resolve pass sits between the
  composite and the surface.

## The levers, ranked

### Tier 1 — recover whole frames (this branch)

- [x] **Paint inside the rAF callback.** `request_paint` now registers a
  one-shot `platform::on_animation_frame` closure that paints directly in the
  animation phase, ahead of the browser's rendering steps — replacing the
  dioxus task that resumed two waker hops after the rAF, potentially behind
  another VDOM render.
- [x] **Stop the chrome diff at pointer rate.** `GestureCommand::To` goes
  through `dispatch_sample`, which integrates and repaints but skips the
  `observe()` walk, the `obs.set` (nothing the chrome reads changes
  mid-gesture; the committed document stands until End) and the outbox flush
  (nothing commits mid-gesture; End still goes through the full `dispatch`).
- [x] **No presence writes when solo.** The per-move and on-leave `SetCursor`
  publishes are gated on `CollabState::active()`.

### Tier 2 — align engine work with frames; stop discarding input

- [x] Decouple ingestion from preview: every mutation that used to rebuild the
  fold (`Engine::refresh_live`) now marks it stale (`mark_live_stale`), and
  the read services it (`flush_live`) — once per frame painted, in
  `render_view`/`pick_color`/`live_head_count`. Every sample still reaches
  the fitter per event; only the fold is per-frame. Peer gesture frames off
  the collab pump coalesce the same way.
- [x] `getCoalescedEvents()` + real `event.timeStamp`: `input::samples` reads
  the coalesced list (client coords through the target rect, per-entry
  pressure/tilt/timestamp) and `input::sample` stamps `event.timeStamp` on
  every sample, so the Start sample seeds the fitter's `t0` with the same
  clock.
- [ ] Then `getPredictedEvents()` (preview tail only — prediction never enters
  the fitter, so `preview == committed` is untouched) to cover the browser
  compositor's untunable 1–2 frames.
- [x] Persist dynamics scratch across a stroke — the textures half is done (the
  `ScratchPool`, see the write-back/scratch implementation record below); the
  bind groups half is not, and the composite's per-tile groups cannot be, since
  the base tiles they reference are fresh CoW tiles every fold.
- [x] The wick-removal experiment **on master**: run 2026-08-10, and the removal
  shipped. The item as written promised a freed segment cap that turned out to be
  march-only (see the corrected ledger entry above); what the experiment actually
  gated was the removal itself, and it came back clean — master's
  delivery-integral settle (2026-08-02) cures the lift-end ring on its own,
  exactly as the march's settle-as-continuation did. Details in the
  implementation record below.
- [x] **Skip presentation frames when the GPU falls behind.** The WebGPU
  backend has no back-pressure of its own (`present` is a no-op, Fifo and
  `desired_maximum_frame_latency` are dead values, `get_current_texture` never
  blocks), so a frame whose fold+composite exceeded the frame budget used to
  deepen the GPU queue on every rAF, unboundedly, for as long as the stroke
  lasted. `Renderer::paint` now counts frames in flight via
  `Queue::on_submitted_work_done`, and the rAF callback (`schedule_paint`)
  skips the paint — holding the `paint_queued` latch and re-arming for the
  next frame — while ≥2 painted frames are still executing
  (`MAX_FRAMES_IN_FLIGHT`, the depth `desired_maximum_frame_latency` would
  have asked for). Safe by the ingestion/preview decoupling above: samples
  keep reaching the fitter per event, and one fold's cost is bounded by the
  unfrozen tail rather than by the samples accrued, so the first paint after
  the drain catches up in a single fold. One queue means a paint's completion
  also vouches for every submission before it, so commit renders and fills
  count against the depth too, one frame later. Invisible to the stroke bench
  (browser-side, like Tier 1); verify with the pointer-`timeStamp` → rAF probe.

### Tier 3 — scaling costs

- [ ] Damage-aware compositing; cache per-tile bind groups on the tile.
- [ ] Gate peer-tail and live-fill re-renders on a per-gesture change check.
- [ ] Delta-only `overlay_tiles`; fix the O(n²) per-sample record clone and
  control-point rebuild.

## Wide tips (r≈500): the measured split

Measured 2026-08-10 on `stroke-latency` (run-to-run noise that day <1%, so the
shares are firm). Baselines: `commit/dynamics/500` = 19.13 ms,
`live/dynamics/500` = 913.7 ms per 240-move gesture (~3.8 ms per fold) — the
wide tip's live gesture is as slow as r=8's was before the small-radius work.
The `live/{250,500}` bench lines exist as of this branch; the live bench
drives the lazy fold explicitly (`Engine::flush_live` per move).

Phase split, by env-gated dispatch skipping (the dynamics-perf-profile
bracketing method, re-measured for master's per-segment loop — the march-era
percentages died with the march):

| phase | share of commit | share of live |
|---|---|---|
| deposit | **53%** | **44%** |
| exchange | 23% | 20% |
| slice (write-back) | 6.5% | 15% |
| settle | 2.7% | 10.5% |
| wick+bake+snapshot | 7.8% | 7.4% |
| region composite | ~0 | 2.5% |

The opposite regime from small radii: r=8 is dispatch-bound; r=500 is bound by
per-texel work in the footprint-sized passes. Deposit runs at ~1 ns/texel —
an order of magnitude below streaming rates, so it is ALU/latency-bound on its
per-texel loads and exchange-law math, not bandwidth-bound.

**Done:** `prefix_at` round-tip fast path (a one-layer prefix volume was read
as two identical slices and mixed — eight loads where four suffice, and
`mix(a,a,t)` double-rounds): **−6.0% commit, −6.2% live** at r=500.

**Measured and rejected:** compiling the bleed ladder out of `deposit`
(`@if(bleed)` specialization for non-bleed slots). Deleting the block outright
moved the r=500 lines ≤0.9% beyond the fast path — the never-taken branch
costs no meaningful occupancy, so the specialization machinery isn't worth it.

**Remaining levers, in expected order:**

- **Shoulder-bounded coarse evaluation** — **done for `deposit`** (the 53%/44%
  pass); see the implementation record below. `exchange` (whose share at r=500
  is mostly the footprint `snapshot` riding its grid — a copy that cannot be
  coarsened) and `settle` were left exact.
- **Slice batching** — **done**, though not as the compute write-back this
  entry once sketched (the persistent tile aux is `R16Float`, which WebGPU
  cannot storage-write at all); see the implementation record below.
- **Scratch persistence across a stroke** (Tier 2 item) — **done** for the
  textures; the record below has what the 0.92 ms/move floor actually paid out.
- The settle share (10.5% live) is the preview==commit price and moves only
  with the model.

## Implemented: the shoulder-bounded coarse deposit (2026-08-10)

The brief below was executed the same day, for `deposit` alone as it
prescribes. What was built, and what the measurements said:

- **The cell law** lives in `gpu::stroke::budget::footprint_cell`: a pure
  function of the brush shape and the segment's radius,
  `cell = min(0.02·r, 0.25·shoulder)` with `shoulder = 3·(1−hardness)·r` for
  `Round`, 0 for `Stamp` (sharpest), engaging only above 2 texels and capped
  at 16. Hard tips and every radius ≤ 100 stay at cell 1 — and cell 1 is not
  a parameter value but **the exact kernel**: the host dispatches the
  untouched `deposit` pipeline, so bit-identity there is structural. Bleed
  and settle slots are always exact. CPU pins in `budget::tests` (hard tip
  never coarsened, softer never finer, the bench radii land where this
  section says).
- **Two new entry points** in `dynamics.wesl`: `cell_hoist` (one thread per
  canvas-anchored c×c cell) evaluates the exact kernel's front half — the two
  `prefix_at` differences, the six `bake_at` taps, the divides that recover
  the reservoir means — once at the cell's centre into an fp32 cell scratch;
  `deposit_coarse` runs the exact kernel's own texel grid reading those means
  back for two loads. Strictly per-texel, per model: `surface_tooth`, the
  selection, the snapshot loads, the arc/drain/jitter, the stores, and the
  f16 re-store guard (whose verdict stays exact — a zero-exposure cell takes
  the same "keeps its value" exit as `dpre <= 0`).
- **Canvas anchoring without canvas coordinates**: the cell index is
  `floor((region texel + anchor) / c)` with `anchor = region origin mod c`
  carried in the new `Stamp::k` lane — congruence arithmetic on small
  integers, so pieces and live folds with different region origins agree on
  every cell boundary and no f32 ever holds an absolute canvas position.
  Pinned by `cell_boundaries_are_canvas_anchored_whatever_the_region_origin`.
- **The wide case bound first**: `corpus_wide_smear` (r=250, hardness 0.9 →
  cell 5) exercises the coarse path through the whole battery — incremental
  vs fresh at five cut points, preview == commit, save/load, golden.
  Sabotaging the hoist (storing zero exposure) moves 42% of its pixels, so
  the green run is proof the path binds, not a floored cell. The blessed
  exact-path golden **still passes unchanged** (tol 6, <1% rule), so no
  re-bless was needed.
- **Ripple** (the march round's column-mean method, 21 px trend): exact
  0.377 → coarse 0.403 levels RMS — the added vertically-coherent structure
  is ~0.03 levels, well inside the 0.58 → 0.62 the march round accepted for
  its shoulder bound. Column-mean divergence 0.60 levels RMS; per-pixel mean
  0.51, worst 12 levels at the rim.
- **Bench** (criterion baseline `precell` saved minutes before the change,
  same session, observed same-day noise ~1%): `commit/dynamics/500`
  18.03 → 15.59 ms (**−13.5%**), `live/dynamics/500` 863 → 783 ms
  (**−9.3%**), `commit/250` −2.0%, `live/250` no change, every r ≤ 100 line
  within ±1.1% — the cell-1 floor held, dispatch for dispatch.
- **Why not the −35–45% the brief projected, measured**: a ceiling bracket
  (returning from `deposit_coarse` right after the cell load, so only the
  frame, the reject tests and one cell read remain) lands at −30.6% commit /
  −21.7% live. The gap between that ceiling and the realized win is the
  per-texel body — the selection + two snapshot loads, `exchange_at`, the
  parcel/blend algebra and the two f16 stores — i.e. the **read-modify-write
  itself**, which the projection had attributed to the hoistable taps. Any
  further coarse-deposit work is bounded by that bracket, and most of it
  (loads + stores) cannot move while the pass still writes pixels. The next
  levers on the wide-tip live number are therefore the ones already listed:
  slice batching, scratch persistence, and the settle's model.

## Implemented: the batched write-back and the scratch pool (2026-08-10)

The two levers the coarse-deposit round ranked next, built the same day in
order, each against a criterion baseline saved the same session
(`preslice` → `postslice` → the pool's compare run).

**The batched write-back.** The compute write-back the lever entry once
sketched is impossible as stated — the persistent tile aux is `R16Float`,
which WebGPU cannot storage-write at all. What shipped keeps every rounding
bit-identical instead: one region-sized render pass narrows the wide region
aux to the persistent height channel (`slice.wesl`, now uniform-free and
resid-free), and each tile then cuts its whole `TILE_TEX` block out of the
region by `copy_texture_to_texture` — colour and residual straight from the
region textures (the tile formats are the region's own), aux from the narrowed
texture. A copy is bit-exact, and the narrow pass render-writes a loaded f16
value back to its own lattice point, so every golden passes unchanged. The
tile pool's "a handle never hands out its texture" invariant survives:
`TexHandle::copy_into` encodes the copy inside `tile.rs` and hands nothing
out. Bench vs `preslice`: every commit line improved (commit/8 −6.2%,
commit/500 −2.8%), live/500 −2.9%; that run's live/100 and /250 lines drifted
high on wide CIs, and the pool run below repaid them with interest.

**The scratch pool** (`gpu::stroke::dynamics::scratch`). Every fold used to
create and destroy its region, narrow, snapshot, cell, reservoir and bake
textures — the per-fold fixed cost the 0.92 ms/move floor bounded at ≤24% of
live. `ScratchPool` keeps them on a free list keyed by the exact descriptor
(size, format, usage, label — exact size on purpose: the dynamics shaders read
`textureDimensions` of the snapshot and region, so an oversized stand-in would
change what they compute). It is shared across renderer clones, so live folds
and their commit draw from one free list, and a lease returns only *after* the
submit that recorded against it — the whole reuse argument on one queue. The
snapshot square rounds up to a 64-texel quantum at its allocation site
(`SNAPSHOT_QUANTUM`) so the measured maximum stops drifting a few texels per
fold; invisible to every pass, since stores and reads are gated by the slot
rects and the sweep test, and the `textureDimensions` bounds only widen onto
texels those gates reject. No consumer relies on the zero-init a fresh texture
gets — an audited property listed in the module doc (clear-loads, full
copies/stores before any read, the snapshot's shared `outside_sweep` gate,
hoist-before-read cells; the bake pair was already reused across a stroke's
segments on exactly this argument). Retention is budgeted: 256 MiB, LRU, with
explicit `destroy()` on eviction.

Proof of binding: a temporary probe through `corpus_wide_smear` counted 62
hits / 68 misses — the misses being each key's first take per battery cut,
since the battery builds a fresh renderer (fresh pool) per cut where the app
and the bench hold one per gesture. The battery's incremental-vs-fresh
equalities double as the stale-content detector: different cuts lay different
stale patterns, and the renders still agree.

Bench vs `postslice` (p < 0.05 on every live line): live/8 −3.9% (856 ms),
live/30 −10.2% (341 ms), live/100 −20.5% (253 ms), live/250 −17.0% (359 ms),
live/500 −9.4% (702 ms). Commit lines flat, as they must be — a commit
renders once, so its takes are the pool's misses.

**Net for the session** (`preslice` → after both): live/dynamics faster at
every radius — r=8 −4.2%, r=30 −9.2%, r=100 −13.1%, r=250 −14.2%, r=500
−12.0% — and commit/8 −5.8%, commit/500 −2.7%. Against the morning-of
baselines that opened this file's wide-tip section, live/500 is
913.7 → 702.5 ms (−23%). Still open on the live numbers: the settle's model
(10.5% of live at r=500), and the bind-group half of the per-fold fixed cost —
the loop's own groups could persist behind the pooled textures, but the
composite's per-tile groups cannot (they reference fresh CoW tiles every
fold), so that residual belongs to the damage-tracking/bind-group-cache round
(Tier 3).

## Implemented: the wick removal (2026-08-10)

The Tier 2 experiment, run and acted on the same day.

**What the item got wrong, first.** The promised prize — freeing a
`WICK_TRAVEL_QUANTUM` segment cap — did not exist on master: "a segment may not
straddle a firing" was the *march's* coupling, and master's binomial wick
(commit `5c624de`) already fired on its own cadence inside a segment.
`flatten_tolerance` caps a dynamics segment by `exchange_travel` alone
(0.125 radii for a hard-trading brush, up to 1 radius for gentle ones), and its
own comment says the wick's quantum is deliberately not in the sum. What the
wick actually cost on master was **two serialized dispatches per half-radius of
travel** in the dispatch-bound regime — on the bench's smear brush
(lift 0.6 / deposit 0.5, segments ≈ 0.47 radii) roughly two wick dispatches for
every three-dispatch segment chain.

**The experiment.** The `golden_lift_end_regression` brush (r=80,
hardness 0.95, drain 0.005, lift/deposit 0.95) on a fifteen-radius stroke *and*
on the golden's own 5.4-radius stroke, wick on vs off, against master's
delivery-integral settle (2026-08-02) — which postdates the `WICK_RATE` tuning,
so the tuning measurements were stale. Metrics: worst lateral rise above a
running minimum per column (a stroke's paint may only fall off walking away
from its axis; any rise is a rim outside a groove), and worst rise along the
axis past the stroke end (the trail may only fade). Result: **no ring in either
arm** — worst lateral rise 1 level (frame-edge noise, not at the stroke), fade
perfectly monotone, the arms within 4 levels anywhere. The march's conclusion
holds on master's own settle: the ring's slow payout is what the trail is made
of, the delivery-integral settle serves it in order, and the wick was treating
the symptom of the old settle's mispairing.

**What shipped.** The pass deleted end to end: `wick_x`/`wick_y` and helpers
plus the whole cadence/conditioning essay in `dynamics.wesl`, the two pipelines
(`kit.rs`), the `wick_steps` loop (`run.rs`), the plan's
`Segment { wick_steps }` payload and quantum arithmetic (`plan.rs`),
`WICK_TRAVEL_QUANTUM` and its compile-time stencil assertion (`budget.rs`), and
the `WICK_HALF`/`WICK_RATE` mirror constants (`build.rs`). A painting segment
now cycles the reservoir ping-pong once (exchange), not twice.
`a_drained_smear_leaves_no_ring_at_the_lift_end` (tests/dynamics.rs) pins both
metrics as behaviour; docs/brush.md §6.2 keeps the disease, the history and the
parity lesson (a sparse reach decouples sublattices — cited by the bleed's
ladder).

**Verification.** Full suite 691 tests / 45 binaries green; wasm and
no-default-features clippy clean. `golden_lift_end_regression` — the artifact's
own pin — passed **unchanged**. Exactly two goldens moved, both barely over the
1% rule (`straight_smear_into_paint` 1.03%, `wiggly_smear_into_paint` 1.51% of
pixels over tol 6): the carried patch keeps slightly crisper texture without
the lateral smoothing, the documented edge-softening cost of the wick running
in reverse. Re-blessed after visual inspection.

**Bench** (criterion `prewick` baseline saved minutes before the change, same
session; every line p < 0.05 except live/500):
`commit/dynamics/8` −23.7%, `/30` −18.5%, `/100` −6.9%, `/250` −2.9%,
`/500` −2.7%; `live/dynamics/8` **892 → 707 ms (−21.3%)**, `/30` −14.0%,
`/100` −5.8%, `/250` −3.0%, `/500` flat (p = 0.26). The exact dispatch-bound
profile: the win concentrates where the loop is serialized-dispatch-limited,
and vanishes where it is texel-bound.

## Investigation brief: shoulder-bounded coarse evaluation (as written before the work)

The lever sized to deposit+exchange's 64–75%. Everything below is what a fresh
session needs; read this section, then the referenced material, before code.

### The idea

`deposit`, `exchange` and `settle` evaluate the exchange laws at canvas
resolution across a footprint whose content cannot vary faster than the tip
resolves. A tip's finest feature is its **shoulder**, width
`3·(1−hardness)·r` for `Round` (treat `Stamp` as sharpest — its mask can be
arbitrarily hard). At r=500, hardness 0.5, the shoulder is ~750 px wide; the
per-texel τ taps, bake taps and exchange-law solves are computing a field that
is locally constant at that scale. The march learned this as its travel-cell
law — **resolve the shoulder, never the radius** — and its final bound was
`cell = max(fits, min(0.02·r, 0.25·shoulder), 1)`. The same scale law applies
here, in the footprint domain, with the floor at 1 keeping every tip below
the threshold (and every hard tip at any size) bit-identical by construction.

### The shape to try first

Hoist, don't resample. Deposit is ALU/latency-bound (~1 ns/texel), not
bandwidth-bound, so the win is computing the expensive per-texel quantities —
the two `prefix_at` differences, the six `bake_at` taps, the `exchange_at`
solve — once per c×c cell and applying them across the cell's texels, while
keeping strictly per-texel everything that is per-texel by *model*:

- `surface_tooth` — tooth is the canvas's resolution, never the cell's (the
  branch-only defect the tooth memory records; the deposition gate must stay
  a per-texel read).
- the selection mask, the snapshot loads, the store.
- the identity-store early-out (`ex.keep == 1 && parcel == 0 && !bled`) must
  stay **exact** — an interpolated `keep` of 0.9999 where the true value is 1
  re-stores untouched texels and re-opens the f16 truncation ratchet. Derive
  the cell's "identity" verdict from exact cell-level facts, not from
  interpolated factors.

Do deposit alone first (53% of commit); extend to exchange and settle only
after deposit pays.

### Hard constraints

- **§6.4 purity**: every pass that writes tiles must be a pure function of
  canvas position. The cell grid must be anchored to **canvas** coordinates,
  not to the region rect (region origins differ per piece and per move) —
  otherwise aprons stop matching neighbour interiors and `tests/seam.rs`
  fails. This is the easiest way to get it structurally wrong.
- **preview == commit**: the cell must be a pure function of the brush and
  segment (like the march's was), so live and commit pick the same cell.
- Bleed taps read neighbours at rung distances — check the interaction where
  taps cross cell boundaries before touching the Bleed slot at all (or leave
  Bleed slots at cell 1; they are rare and already small).

### Measurement kit

- Bench: `cargo bench -p stark-core --bench stroke -- "dynamics"` now has
  commit+live at {8,30,100,250,500}; live drives `Engine::flush_live` per
  move. Bracket master/change/master; a win must clear the whole master
  envelope (the noise floor is usually 15–20%, though 2026-08-10 measured
  <1% — check the day before trusting small deltas).
- Phase re-bracket: the `STARK_SKIP_DYN` env-gate instrumentation is not
  committed but is trivial to re-add (gate each dispatch kind in
  `DynamicsRun::record_loop`, the region composite pass, and the write-back
  loop; see the dynamics-perf-profile memory for the method).
- Ripple: the streaks a coarse cell prints are ~1 level RMS but vertically
  coherent — per-pixel max/mean metrics show nothing. Average each column
  down the trail, subtract a smooth trend, compare RMS. The march round's
  numbers: 0.58 (no coarsening) vs 0.62 (shoulder bound) vs 1.04 (radius
  bound) — the shoulder bound is what closed the hardness-1 regression.
- The independent reference: a wide glaze must land the same whether the
  stamp loop or the swept path's analytic integral renders it (the march
  round built `a_wide_glaze_lands_the_same_whether_or_not_the_stamp_loop_runs`
  at r=150 — it did **not** cross the revert; rebuild the method).
- Chris's captured repro (`painting.stark`, r=500 hardness 1 max flow) is the
  regression case for stroke-end spikes; `examples/repro_march` was the
  march-era harness — verify what still exists on master at session start.

### Traps on record (do not rediscover)

- **Every existing dynamics golden paints at r 30–110, where any shoulder
  bound floors to cell 1** — the whole suite would pass an arbitrarily wrong
  constant without exercising it once. Build the wide case first (the
  `golden_wide_smear_regression` r=250 curved smear pattern) and check what
  is *binding* before believing any green run.
- A hardness-1 tip must earn **zero** coarsening — the 2026-08-07 stroke-end
  spike regression was exactly a radius-scaled cell sampling a shoulderless
  tip at 10 px.
- Sub-sampling coverage *within* a step was measured on the march: cost
  15–17% and did not fix what the coarse cell broke. Bounding the cell did.
- The march memories describe stroke-space machinery — the scale law and the
  measurement methods transfer; the code does not. Read the
  stroke-space-march-reverted memory before trusting any march memory.
- Write CPU pins for the cell law itself (the march had
  `the_travel_cell_follows_the_tips_shoulder`-style tests: hard tip → no
  coarsening, soft tip → reaches the bound, soft never finer than hard,
  Stamp treated as sharpest, small tips march one texel). Those tests did not
  cross the revert; they need writing fresh for the footprint cell.

### What success looks like

Amdahl: deposit+exchange+settle ≈ 65–75% of live at r=500; a c=2 cell hoists
~75% of their ALU → roughly −35 to −45% on `live/dynamics/500` (913.7 ms →
~500–600 ms), more at c=4 for very soft tips. Nothing may move at r ≤ 100
(cell floors to 1 — verify with the bench sweep), and the hardness-1 capture
must be bit-identical.

## Measuring

- Tier 1 and the ingestion/preview decoupling are invisible to
  `cargo bench -p stark-core --bench stroke` — they need a browser-side probe
  (pointer `timeStamp` → rAF-presented delta) or a camera.
- The stroke bench's noise floor is ~15–20% on this box: bracket
  (master / change / master) and compare against the whole master envelope;
  never pair two runs.
- The per-live-update settle is what keeps `preview == committed` for
  dynamics; any throttling must move *when* it runs, never *whether*.
