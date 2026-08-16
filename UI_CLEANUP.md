# `stark-ui` cleanup

A review of [crates/stark-ui/](crates/stark-ui/) — all 44 modules, 21,675 lines —
with what to do about each finding and how you would know it worked. Where a claim
is about how dioxus behaves it was checked against `dioxus-signals-0.7.10` rather
than assumed.

> **The identifiers here are `U1`–`U9`, deliberately not `§n.m`.** Design-doc
> section numbers are stable and cited from ~1000 places in the source
> (CLAUDE.md); a work list is neither stable nor citable. Where a finding
> contradicts or extends a design section, it says so and cites it.

> **Every finding below describes the code as it stood when it was written**, line
> numbers and counts included; what happened to it is in the **Landed** note at the
> end of its section. Read the two together — a finding whose note says *done* is
> history, and the file it points at has moved on. This file goes when the last one
> is closed, for the reason `SHADERS_CLEANUP.md` went: a review that outlives its
> work becomes a second, staler account of the code.

The crate does the hard part well. The [`dispatch`](crates/stark-ui/src/state.rs)
seam is the best structural decision in it: `renderer` and `obs` are held in
`ReadOnly` handles whose inner `Signal` is private to the module, so `&mut Renderer`
is genuinely unreachable from a panel — "every mutation publishes" is a property of
the types rather than a rule a call site could forget (§4, §7).
[`modes.rs`](crates/stark-ui/src/modes.rs) makes "two composing modes at once" a
state the app cannot reach instead of one the DOM order has to arbitrate.
[`reorder.rs`](crates/stark-ui/src/panels/reorder.rs) is the model extraction —
pure, nine tests, shared by two panels — and [`platform.rs`](crates/stark-ui/src/platform.rs)
puts every browser call behind one door with the reasoning for each written down.

**Almost everything below is one shape: the reactive graph is a single node.**
Every panel subscribes to the whole projection, so every command re-renders the
whole chrome; and the one escape hatch built for it,
[`dispatch_sample`](crates/stark-ui/src/state.rs), has exactly **one call site** in
the crate while six pointer-rate gestures still take the loud door. U1, U2 and U7
are three views of that.

**U3 is the only live bug**, and it is the one that loses an artist's work: core
projects `gpu_failure` with an explicit contract for the frontend, and the frontend
reads it nowhere.

## Ranked

Seven of the nine are **done**; what is left is U8 and the halves of U1 and U6 that
are decisions rather than work. Each section below carries a **Landed** note saying
what was actually done — and where the finding turned out to be wrong the note says
so rather than quietly dropping it. Two were: U9's `update_brush` item, and U5's
proposal to move the base64 codec. U1 also turned up a latent bug that the
over-subscription had been hiding.

| | Finding | Kind | Size | Status |
|---|---|---|---|---|
| [U3](#u3-gpu_failure-is-projected-by-core-and-read-by-nothing) | `gpu_failure` is projected by core and read by nothing | correctness | medium | **done** |
| [U1](#u1-observablestate-is-one-signal-the-whole-chrome-subscribes-to) | `ObservableState` is one signal the whole chrome subscribes to | performance | large | **partly** — the seam and the panels; the split is open |
| [U2](#u2-the-root-effect-subscribes-to-the-renderer-so-every-engine-touch-runs-a-json-serializing-thumbnail-scan) | The root effect subscribes to the *renderer* | performance | small | **done** |
| [U4](#u4-layoutrs-is-a-third-untested-copy-of-the-reorder-gesture) | `layout.rs` is a third, untested copy of the reorder gesture | test health | medium | **done** |
| [U5](#u5-six-copies-of-the-localstorage-layer) | Six copies of the localStorage layer | structure | small | **done** |
| [U6](#u6-there-is-no-compile-time-boundary-between-browser-glue-gesture-math-and-chrome) | No compile-time boundary between browser glue, gesture math and chrome | structure | large | **partly** — `gesture.rs` is out; the boundary is a decision |
| [U7](#u7-canvas-gesture-state-has-no-owner) | Canvas gesture state has no owner | structure | medium | **done** |
| [U8](#u8-the-previewcommit-bargain-is-stated-11-times-and-implemented-5-ways) | The preview→commit bargain is stated 11 times and implemented 5 ways | structure | medium | open |
| [U9](#u9-smaller-items) | Smaller items | mixed | small | **done** |

## What is left, and why

- **U8** is untouched, and it is the one remaining finding that is plain work.
- **U1's structural half** — splitting the projection by cadence — is what stops
  `observe()` walking the layer tree for a pan, and half of it is in `stark-core`.
  The frontend half landed and takes most of the cost off; what remains is a core
  API decision.
- **U6's real half** — *decide what the host build is for* — is a decision, not a
  task. The cheap half, getting the testable code out of the modules that cannot be
  tested, landed.

---

## U1. `ObservableState` is one signal the whole chrome subscribes to

There are 29 `obs.read()` sites, nearly all in render bodies. A `Signal` write marks
**every** subscriber dirty with no equality check — `SignalSubscriberDrop::drop` →
`update_subscribers` in `dioxus-signals-0.7.10/src/signal.rs:255`, which the crate's
own comment in [`collab.rs`](crates/stark-ui/src/collab.rs) already states
("`Signal::write` marks its subscribers dirty whether or not the value changed").

So one `obs.set` is a full VDOM diff of the chrome. And `Engine::observe` is not
cheap to produce either: it walks the whole layer tree and allocates a fresh
`Vec<LayerInfo>` — with `String` names — plus a `Vec<PerspectiveGuide>`.

[`dispatch_sample`](crates/stark-ui/src/state.rs) exists precisely to dodge this,
and its doc comment is exactly right about why. It has **one call site**
([main.rs:656](crates/stark-ui/src/main.rs#L656)). Six other pointer-rate gestures
go through the full `dispatch`:

| Gesture | Site |
|---|---|
| Pan / pinch / scrubby zoom | [input.rs:261](crates/stark-ui/src/input.rs#L261), [input.rs:415](crates/stark-ui/src/input.rs#L415) |
| Brush size/flow tuning | [input.rs:678](crates/stark-ui/src/input.rs#L678) via `update_brush` |
| Eyedropper drag | [input.rs:1113](crates/stark-ui/src/input.rs#L1113) via `update_brush` |
| Transform / perspective / warp preview | [panels/transform.rs:123](crates/stark-ui/src/panels/transform.rs#L123) |
| Frame handles, gradient axis, filter dial and pad, layer opacity and bend | 20 `ViewCommand::Preview` dispatches across 6 files |

Each of those walks the layer tree and re-diffs the whole chrome, per pointer move,
to move a number the Layers panel does not read. A pan is the clearest case: the
view is the only thing that changed, and `Engine::view()` already answers it for
free.

**Fix, incremental.** The answer is already in the crate, twice:
[`navigator.rs:218`](crates/stark-ui/src/panels/navigator.rs#L218) and
[`timeline.rs:200`](crates/stark-ui/src/panels/timeline.rs#L200) read through
`use_memo`, so they wake only when their own slice changes — which is why the
navigator can afford to subscribe to `obs` at all. Apply the same to the panels that
read `obs` for commit-rate data. `LayerPanel` reading its layer list through a memo
stops it re-rendering during pans, with no core change and nothing to design.

**Fix, structural.** Split the projection by **cadence**, which is how it actually
changes:

- `view` — pointer rate
- `doc` — `layers`, `active_layer`, undo flags, `bounds`, `doc_revision`,
  `has_selection`, `selection_hull`: commit rate
- `tool` — `brush`, `tool`, `shape_action`, `selection_feather`, `shape_opacity`
- `session` — `history_budget`, `color_space`, `surface`, `background`, `media`,
  `environment`, `show_peer_selections`: near-never

`with_engine` publishes each only when it changed. This is also the shape §7 wants:
once the engine is behind a channel, shipping the whole layer list across it per pan
is exactly the cost an actor boundary should not pay. The core half — letting
`ViewCommand::Pan` answer without a layer walk — is the larger win and can land
independently.

**How you would know.** A stroke and a pan should dirty no panel scope that is not
showing the view. `cargo bench -p stark-core --bench stroke` will not see it (it is
frontend-side), so the check is a `tracing` counter on component renders per gesture,
or simply that the layer panel's row `key`s stop churning during a pan.

> **Landed (incremental half).** `state::use_obs` is the seam, and the canvas, the
> command rail and the Layers, Select, Lighting, Frame and Filter chrome read through
> it. `LayerRow` takes `active` as a prop rather than asking the projection per row.
> `frame::selected_frame` and `filter::selected_filter` split into a render-time memo
> and a handler-time peek over one shared rule, the way `modes::composing` and
> `modes::is_composing` already do.
>
> That split turned up **a latent bug the over-subscription was hiding**: the canvas
> took its tool from a `peek`, so its eyedropper cursor was only ever correct because
> something else happened to re-render the component. It reads it now.
>
> **Still open:** the structural half. `observe()` still walks the layer tree for a
> pan, and `update_brush` still pays for one per pointer move of a tuning drag (see
> U9). Nothing but splitting the projection in core removes that.

## U2. The root effect subscribes to the *renderer*, so every engine touch runs a JSON-serializing thumbnail scan

[main.rs:149-154](crates/stark-ui/src/main.rs#L149-L154):

```rust
use_effect(move || {
    let _ = state.presets.read().len();
    let _ = state.slots.brushes.read().len();
    let _ = state.renderer.read().is_some();   // <-- a reactive context
    thumbs::refresh(state);
});
```

`use_effect` **is** a reactive context, unlike an event handler — subscription is
gated on `ReactiveContext::current()` being `Some` (`signal.rs:411`), which is why
`read` in a handler is harmless and `read` here is not. And every door into the
engine — `with_engine`, `with_engine_quiet`, so also `dispatch_sample` and
`dispatch_quiet` — takes `renderer.write()`, which dirties subscribers
unconditionally.

The effect therefore re-runs per pointer sample. It calls
[`thumbs::refresh`](crates/stark-ui/src/thumbs.rs#L134) →
[`next_missing`](crates/stark-ui/src/thumbs.rs#L163), which computes
[`key(w)`](crates/stark-ui/src/thumbs.rs#L112) — **`serde_json::to_string(&BrushParams)`**
— for every preset *and* every rack slot, and since nothing is missing the scan runs
to the end every time. That is ~16–20 JSON serializations per frame during a stroke,
on the path the [stroke latency ledger](crates/stark-ui/src/input.rs) exists to
protect. It also re-renders the root `app()` per sample.

**Fix.** Two edits, neither of which touches a design:

- `thumbs::key` should hash the fields, not round-trip through `serde_json`. The
  stable-wire-form argument justifies the *storage* format
  (`presets::encode_wearable`), not a cache key, and the key is session-local
  anyway.
- The effect wants "a renderer **appeared**", not "a renderer was touched". Have
  [`publish_renderer`](crates/stark-ui/src/state.rs) set a `renderer_ready: Signal<bool>`
  once and depend on that. A `use_memo(|| state.renderer.read().is_some())` also
  works and is smaller, but the signal is the honest statement — the effect's
  dependency really is a one-time event.

**How you would know.** Put a `tracing::trace!` in `thumbs::refresh` and draw one
stroke: it should fire twice (library load, renderer publish), not sixty times a
second.

> **Landed.** `state::renderer_ready` is the boolean, set once by `publish_renderer`.
>
> The key is **gone** rather than made cheaper, which is not what this section
> proposed. Its one job is to say whether two brushes render the same picture, and
> `Wearable` already answers that exactly through a derived `PartialEq` the compiler
> extends when `BrushParams` gains a field. A hand-written hash would have been fast
> *and* would have silently ignored that field — a stale thumbnail for a changed
> brush, with nothing anywhere to say so. A linear scan of a few dozen `Copy` structs
> is cheaper than one of the serializations it replaces, so the key bought nothing.
>
> `PartialEq` makes non-termination expressible where a hash did not: a brush holding
> a NaN is never found after being cached, and the generator loop would hand back the
> same brush forever. Nothing can produce one, so the loop's progress check is a
> guarantee rather than a case.

## U3. `gpu_failure` is projected by core and read by nothing

`stark-core` projects `ObservableState::gpu_failure` and documents the contract at
length:

> **Projected because the document outlives the device.** […] What a frontend should
> do with it is stop dispatching and offer to save; what it must not do is keep
> painting, since nothing after this point reaches a pixel.

It is an `Arc` specifically so the projection stays cheap to clone at pointer rate —
the field was designed for this frontend to read.

`grep gpu_failure crates/stark-ui/src` returns **one comment** in
[thumbs.rs:255](crates/stark-ui/src/thumbs.rs#L255) and no code.

So after a device loss: [`Renderer::paint`](crates/stark-ui/src/render.rs#L653)
silently returns on a failed `get_current_texture`, dispatches keep landing in a dead
engine, and the artist sees a frozen canvas with no message and no prompt — while the
action log, which *is* the painting (§8), is intact in ordinary memory and
`save_bytes_resolvable` would still write it. This is the one place where the crate
does not implement its own stated design, and it is the failure mode that costs a
session's work.

**Fix.** A `GpuFailureModal` mounted from the root on `obs.gpu_failure.is_some()`,
offering Save and nothing else — a dismissible dialog would be wrong, because there
is nothing to go back to. `dispatch` and `request_paint` short-circuit once it is
set, so the app stops spending frames on a device that cannot answer. The save path
must not go through `with_engine` (which requests a paint); `save_document` already
only reads.

**How you would know.** `Engine` can be made to report a synthetic failure in a test;
failing that, the honest check is that the modal's mount condition is a plain read of
the projection, so it cannot be reached by a path that does not also set the field.

> **Landed.** `crate::failure::GpuFailureModal`, mounted at the root on its own read
> of the projection, offering Save and nothing else — no ✕, because there is nothing
> behind it to close onto.
>
> Both halves key off one field (`state::gpu_lost`), so the app cannot be stopped
> without saying so, or say so while still running. The guard is on the two doors
> rather than the 82 call sites; the command that *causes* the failure still runs and
> still publishes, which is what makes the report appear at all.
>
> One thing this section did not anticipate: a failure suffered **between** commands
> — a driver reset while the artist is looking at their work — is found by the paint
> loop, which is the one thing that goes on happening when nothing is dispatched.

## U4. `layout.rs` is a third, untested copy of the reorder gesture

[`layout::DragState`](crates/stark-ui/src/layout.rs#L193) and
[`reorder::Slide`](crates/stark-ui/src/panels/reorder.rs#L191) are the same
algorithm:

| `layout::DragState` | `panels::reorder` |
|---|---|
| `step()` = height + gap | `Slide::step` |
| `offset()` | `Slide::motion()` |
| `insert_index()` | `Slide.gap` |
| `start_drag` measures boxes by identity | `Grab::begin` |
| leading-edge rule ([layout.rs:210](crates/stark-ui/src/layout.rs#L210)) | the same rule, same words ([reorder.rs:208](crates/stark-ui/src/panels/reorder.rs#L208)) |

The panel-stack drag predates the extraction and was never migrated; the layer tree
and the guide list both went through it. `reorder.rs` has nine tests covering exactly
this arithmetic — including `dragging_up_leads_with_the_top_edge`, written because
"one rule stated once for a block that can be dragged either way is easy to write as
two rules that disagree near the ends." [`layout.rs`](crates/stark-ui/src/layout.rs)
has **zero tests**, and is the copy where that could already be true.

The same split runs through the every-declaration-every-render rule:
[`Motion::css`](crates/stark-ui/src/panels/reorder.rs#L313) has a test pinning it;
the identical rule in [`layout::Panel`](crates/stark-ui/src/layout.rs#L395) — carrying
the same `2026-08-03` scar — is prose only.

**Fix.** Migrate `PanelStack` onto `Grab` / `Slide` / `Motion`. Panels are a flat
list, so this is strictly simpler than the layer tree already using it: the block is
one row, and `Slide.gap` *is* the insertion index with no depth to resolve. It
deletes roughly 90 lines and inherits nine tests. `ResizeState` and the height map
stay where they are — the bottom-edge resize is a different gesture and shares
nothing with this.

**How you would know.** The existing `reorder` tests cover the arithmetic; the
panel-specific part left over is the `data-panel` round trip, which
`platform::panel_boxes` already keys on.

> **Landed.** `start_drag` is 8 lines where it was 48, and `Panel` takes a `Motion`
> where it took an offset and a flag — so the every-declaration-every-render rule is
> now stated by the function that has the test pinning it.
>
> Two behaviours changed, both by inheriting rules the other two rosters already had:
> a press must travel `GRAB_SLOP` before the panel moves, and a release this handler
> never hears about ends the gesture instead of leaving a panel following an
> unpressed pointer.
>
> `panel_key` is the one identity — the `data-panel` attribute, the grab's key and
> the landing's list were `{id:?}` written out three times. layout.rs has four tests
> where it had none.

## U5. Six copies of the localStorage layer

`fn storage() -> Option<web_sys::Storage>` appears **verbatim** in `gradients`,
`identity`, `prefs`, `presets`, `shapes` and `slots`. Four of those also carry
near-identical `persist` / `read_storage` / `parse_entry` triples with the same
line-oriented base64 codec, the same "skip a damaged line rather than poison the
library" invariant, and the same warn:

```rust
tracing::warn!("could not persist the X (storage full or unavailable)");
```

The invariant is stated in four module comments and tested in two of them.

**Fix.** A `storage.rs` with `get(key) -> Option<String>` and `set(key, &str)`
carrying the one warn, plus a line-table helper — `save_lines(key, iter)` /
`load_lines(key, parse)` — for the `b64|b64` record format the four libraries share.
That is roughly five lines per library instead of forty, and "one damaged line costs
one entry" gets stated and tested **once** instead of asserted four times.

Move `base64_encode` / `base64_decode` out of
[platform.rs](crates/stark-ui/src/platform.rs) alongside it. They are a codec, not
browser glue, and sitting in the browser-glue module is where they will be
overlooked — `platform.rs` is the file you read when something DOM-shaped is wrong,
not when a stored entry fails to parse.

> **Landed** — except the base64 move, which was **wrong**. `platform` is the bottom
> layer: it is what `storage` opens the store through, and it needs base64 itself, to
> read back the data URL the browser hands over for a re-encoded brush image. Owning
> the codec in `storage` would point a dependency up the stack. It stays where it is,
> with the reasoning recorded so it does not read as an oversight.
>
> `crate::storage` is the door and the format. `save_table`, `load_table`, `record`
> and `FIELD` replace four `persist`/`read_storage` triples and **seven**
> hand-written separators — one of them inside `presets::encode_wearable`, all of
> which had to agree for a stored brush to be readable. The skip-a-damaged-line rule
> is stated once and has the test. Each module keeps what is actually its own: why
> the gradient stops are JSON, why a slot record is keyed by digit rather than by
> position.

## U6. There is no compile-time boundary between browser glue, gesture math and chrome

`Cargo.toml` has this:

```toml
# Web platform glue (the crate targets wasm32 via `dx serve --platform web`).
# [target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
```

The target line **is commented out**. So `web-sys`, `wasm-bindgen` and `js-sys` are
unconditional dependencies, every module can reach the DOM, and "the crate still
compiles for the host" is achieved by `#[cfg]`-stubbing about fifteen function
*bodies* in `platform.rs` while `Renderer` holds a `web_sys::HtmlCanvasElement`
unconditionally.

The stated purpose of the host build ([render.rs](crates/stark-ui/src/render.rs#L711))
is that `cargo test --workspace` and clippy exercise it. But all 66 tests reach only
the pure logic, and that logic is diluted inside modules that cannot be tested:

- [state.rs](crates/stark-ui/src/state.rs) is 1,650 lines: about 350 of app state and
  the dispatch seam — the module's actual subject — wrapped around ~600 lines of
  transform / perspective / warp algebra carrying 18 of the crate's 66 tests. That
  algebra is pure `Vec2` / `Mat2` and has nothing to do with signals.
- [panels/layer.rs](crates/stark-ui/src/panels/layer.rs) is 1,510 lines:
  `Row`, `rows()`, [`landing()`](crates/stark-ui/src/panels/layer.rs#L145) and
  `subtree_len` are pure tree logic with 10 tests, under 700 lines of rsx.

**Fix, cheap half.** Pull the pure layers out — a `gesture/` module for the three
transform families, `layer_tree.rs` for the row and landing logic — on `reorder.rs`'s
own precedent. Not for tidiness: these are the only parts that *can* be tested
off-wasm, and today they are the parts hardest to find.

**Fix, real half.** Then decide what the host build is for. Either commit to the
boundary — a `Platform` trait, so the pure half is genuinely platform-free and the
browser half is one implementation — or cfg-gate the browser dependencies and stop
paying for web-sys in a build that cannot use it. The present middle has the
dependency cost of the browser everywhere and the testability of neither, and the
`#[cfg]` pairs grow by two every time a new browser API is reached for.

> **Landed (cheap half).** `crate::gesture` holds the transform, perspective and warp
> algebra — 863 lines and 18 of the crate's tests, none of which knows about signals,
> dioxus or the browser. `state.rs` went from 1650 lines to 938 and is now about what
> its module comment says it is about. One import line changed, because
> `panels::transform` was the only consumer — which is itself the evidence it was
> never state's business.
>
> **Still open:** the same move for `panels::layer.rs`'s tree logic (`Row`,
> `landing`, `rows`, 10 tests, still inside 1500 lines of rsx), and the real half —
> the decision about the host build.

## U7. Canvas gesture state has no owner

[`end_interaction`](crates/stark-ui/src/input.rs#L1394) takes five parameters and
[`abandon_gesture`](crates/stark-ui/src/input.rs#L1444) takes three. That is not an
API accident — it is because the canvas gesture's state lives in three places at
once:

- `drawing` and `action_restore`: local to the `Canvas` component
- `picking`, `canvas_active`, `brush_ring`, `tow`: in `AppState`
- `nav`, `tune`: hook-owned

The two teardown functions must agree about what "in flight" means, by hand, across
four call sites — two `abandon_gesture` and two `end_interaction` in
[main.rs](crates/stark-ui/src/main.rs). `Canvas::onpointerdown` is ~120 lines, most
of it that bookkeeping rather than the routing decision it is actually about.

**Fix.** A `CanvasGesture` hook in `input.rs` beside `Nav` and `Tune`, owning
`drawing` and `action_restore` and exposing `begin` / `advance` / `end` / `abandon`.
Same shape as the two hooks that already work, and it turns "these two teardowns must
agree" from a convention into one type. The signals that are shared for a *reason* —
`pick.dragging` because the options bar reads it, `brush_ring` and `tow` because they
are drawn by sibling overlays — stay in `AppState`; they are documented as shared and
the documentation is right.

This also serves U1: a gesture that owns its own state is a gesture that can choose
`dispatch_sample` for its samples without the choice being spread over four handlers.

> **Landed.** `input::Paint`, shaped like `Nav` and `Tune` beside it.
> `end_interaction` takes four `Copy` values and no `&mut`; `abandon_gesture` is
> gone, folded into `Paint::abandon`. Both teardowns share one private `close`, so
> what a finished gesture leaves behind is stated once instead of twice.
>
> The routing deliberately stayed in the component — which of navigation, tuning,
> sampling or paint a press turns out to be is the one thing only the canvas can
> decide, because it is the only place that sees all four bindings at once.
> `onpointerdown` is now that decision and nothing else.

## U8. The preview→commit bargain is stated 11 times and implemented 5 ways

[`widgets::settle`](crates/stark-ui/src/widgets.rs#L73) captures the subtle part
correctly — three end events, idempotent, no undo step for a drag that came back —
and it is used by `layer.rs` and `filter.rs` and nowhere else. `frame.rs`,
`gradient_bar.rs`, `lighting.rs` and `transform.rs` each roll their own, and each has
to independently remember: preview per sample, commit once, drop the preview on
abandon.

The tell is [`modes::leave`](crates/stark-ui/src/modes.rs#L96), which must `match` on
the mode to know *which* preview command drops it — and on `GradientTarget` again
inside that. The pairing of "the command that previews" with "the command that drops
it" is **data**, not control flow, and writing it as control flow is what makes a new
previewing mode a thing that can forget its own teardown.

**Fix.** A small `Preview<T>` carrying the two halves — the `ViewCommand` that shows
it and the `DocCommand` that commits it — so `settle` and `modes::leave` are each one
line, and a control that can preview cannot be written without saying how it is
dropped. This is the same move `reorder.rs` made for the drag: the subtleties are
real, so state them once.

## U9. Smaller items

- **`state::update_brush` takes the loud door.** It reads `obs`, mutates a copy and
  `dispatch`es — and it is called per pointer move by both the tuning drag and the
  eyedropper (U1). It should be `dispatch_sample`-shaped, or at least document why it
  is not. — **This was wrong.** A tuning drag's whole answer is read off the Brush
  panel's sliders and the eyedropper's off the Color panel's swatch, so the publish
  *is* the point of those gestures rather than overhead on them; `dispatch_sample`
  would break both readouts. What made it expensive was every *other* panel waking
  too, which U1 fixed. Documented rather than changed.
- **`NewDocumentModal` subscribes to the renderer.**
  [main.rs:1056](crates/stark-ui/src/main.rs#L1056) reads `state.renderer` in a render
  body, so the modal re-renders on every engine write while open. Harmless today (it
  is only open when idle), but it is U2's class, and
  [`PeerCursors`](crates/stark-ui/src/main.rs#L718) already peeks with a comment
  explaining exactly this hazard — so the convention exists and this is the deviation.
  **Done**, and both facts were in the projection anyway, so it reads them there and
  touches the renderer signal not at all.
- **`Renderer` is ~25 one-line forwarders.** The seam that matters is `process` and
  `&mut` access, and [render.rs](crates/stark-ui/src/render.rs#L143) argues for it
  well. The `&self` readers — `observe`, `view`, `color_space`, `surface`,
  `scrub_range`, `peers_revision`, … — are not a decision, and each one added to core
  has to be transcribed here. A `fn engine(&self) -> &Engine` is safe for precisely
  the reason the design is careful (it cannot hand out `&mut`) and removes ~150 lines
  that will otherwise drift. **Not done** — mechanical, ~60 call sites, and worth
  doing the next time this file is open rather than on its own.
- **`base64_decode` rebuilds a 256-byte lookup table per call**
  ([platform.rs:635](crates/stark-ui/src/platform.rs#L635)), once per stored entry at
  load. Make it a `const`. **Done** — built at compile time from the alphabet beside
  it.
- **Where a comment states an invariant, the invariant wants a test.** 36% of the
  crate's lines are comments and they are worth every one — they carry *why*, with
  dated scars, which is the thing that stops a fix being re-broken. The gap is that
  `reorder.rs` is the only module that also *pins* its rules
  (`a_resting_row_still_states_its_transform`). U4 and U6 are both instances: a rule
  stated in prose in two places, tested in one.

## What is working, and should not be disturbed

- **`ReadOnly` + `with_engine` / `with_engine_quiet` / `dispatch`.** Making the
  publish an attribute of the *door* rather than of the caller's memory is the right
  shape, and the naming of `with_engine_quiet` — named for what it declines to do —
  is what makes the quiet path reviewable. Nothing in U1 changes this; it changes what
  `publish` means.
- **`root_signal`.** Owning every `AppState` signal at `ScopeId::ROOT` because
  detached tasks read them is the fix the warning itself prescribes, and it is stated
  once in a constructor rather than remembered per field.
- **`request_paint` / `schedule_paint`.** Coalescing to one paint per animation frame,
  in the rAF callback rather than a task the rAF wakes, with `gpu_behind` as the
  back-pressure WebGPU does not give — this is the most carefully reasoned code in the
  crate and the reasoning is all written down.
- **`modes.rs`.** "Entering a mode leaves whichever was live" makes two-catchers-over-
  one-pointer unreachable rather than arbitrated, and the split between `composing`
  (`read`, render-time) and `is_composing` (`peek`, handler-time) is exactly right.
- **`reorder.rs`.** Grab / Slide / Motion, the terminal `spend`, and `claimed` for the
  click behind the release. U4 is a request for *more* of this, not less.
- **`platform.rs`'s one-door rule**, and the capture-phase argument for
  `on_window_pointer`: the pen's eraser end must be in force before any surface's own
  handler runs, and a listener the tree could silence is one that stops working the day
  something downstream calls `stopPropagation`.
- **The projection-not-a-shadow discipline.** `PickScope` holding the *choice* rather
  than a `LayerId`, the timeline keeping no playhead of its own, `prefs::capture`
  reading `history_budget` off `observe()` — each is a copy that was deliberately not
  made, and each is a class of staleness ruled out rather than an instance fixed.
