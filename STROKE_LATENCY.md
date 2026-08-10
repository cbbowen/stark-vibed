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
  `WICK_TRAVEL_QUANTUM = 0.5` (the wick may not be straddled), so r=8 runs
  hundreds of serialized dispatch chains — the 889 ms live gesture in the
  bench record.
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

- [ ] Decouple ingestion from preview: keep pushing every sample into the
  fitter per event (µs), rebuild the live preview once per rAF just before
  paint. Halves live GPU work for input above display rate and makes
  coalesced events natural.
- [ ] `getCoalescedEvents()` + real `event.timeStamp` into `InputSample.time`.
- [ ] Then `getPredictedEvents()` (preview tail only — prediction never enters
  the fitter, so `preview == committed` is untouched) to cover the browser
  compositor's untunable 1–2 frames.
- [ ] Persist dynamics scratch (region/snapshot/reservoir textures, bind
  groups) across a stroke — the per-piece fixed cost the r≈100 bench residual
  pointed at.
- [ ] The wick-removal experiment **on master**: it frees the
  `WICK_TRAVEL_QUANTUM` segment cap (the small-radius dispatch lever), but the
  settle-as-continuation that made removal safe on the march branch did not
  cross the revert — re-run the 15-radius wick-on/off A/B against master's
  settle before believing it.

### Tier 3 — scaling costs

- [ ] Damage-aware compositing; cache per-tile bind groups on the tile.
- [ ] Gate peer-tail and live-fill re-renders on a per-gesture change check.
- [ ] Delta-only `overlay_tiles`; fix the O(n²) per-sample record clone and
  control-point rebuild.

## Measuring

- Tier 1 and the ingestion/preview decoupling are invisible to
  `cargo bench -p stark-core --bench stroke` — they need a browser-side probe
  (pointer `timeStamp` → rAF-presented delta) or a camera.
- The stroke bench's noise floor is ~15–20% on this box: bracket
  (master / change / master) and compare against the whole master envelope;
  never pair two runs.
- The per-live-update settle is what keeps `preview == committed` for
  dynamics; any throttling must move *when* it runs, never *whether*.
