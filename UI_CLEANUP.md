# `stark-ui` cleanup

A review of `crates/stark-ui` as of 2026-08-21 (`f64f874`): five defects, six
structural changes and two sweeps, each with the file and line that shows it.

## Status

| | | |
|---|---|---|
| **D1** | peer cursor / view | **done** — `88ed008` |
| **D2** | no WebGPU | **done** — `56785a5` |
| **D3** | pop-out dismissal | **part** — Escape reaches them (`e9bb2cc`); light dismiss still owed, see the item |
| **D4** | `preview.rs` pair test | **done** — `88ed008` |
| **D5** | stale comments | **done** — `88ed008` |
| **A1** | `AppState` handle | **done** — `54af4e6` |
| **A2** | `state.rs` coupling | **done** — `5c09438` |
| **A3** | one mode signal | **done** — `1784c86` |
| **A4** | command registry table | **open** |
| **A5** | `platform.rs` cfg twins | **open** |
| **A6** | module grouping | **open** |
| **S1** | `use_obs` sweep | **done** — `86ee0fe` |
| **S2** | untested geometry | **done** — `f784fdb` |

The three left open are the three that are large and mechanical rather than
load-bearing, and each says at its own entry what it would cost. Nothing about
them changed in the doing of the rest; **A4** in particular is still exactly the
gap §25.2 step 5 admits to.

The line numbers below are from the review and are **not** updated as the fixes
land — they are what the finding was found at. Follow the named function, not
the number.

Nothing here is a redesign. The frontend's spine — one dispatch seam, `ReadOnly`
handles that make `&mut Renderer` unspellable outside `state.rs`, the
`Preview<T>` pair table, the command and drag registries — holds up, and every
item below is either a place the enforcement stops one step short of what the
design already claims, or a cost the design pays without meaning to. The list is
ordered by leverage inside each section, not by effort.

The three worth doing first, if only three are done: **D1** (one line, and the
picture is wrong today), **D2** (the app's worst first-run failure), and **A1**
(measure, then ~30 lines, for a 500× smaller handle on the hottest type in the
crate).

## Defects

### D1. Peer cursors do not follow pan or zoom while peers are idle

`main.rs:978` reads the view with `state.renderer.peek()`, and the component's
only subscription is `state.collab.peers`. The presence pump writes that signal
only when the roster's revision moves (`collab.rs:495-517`), so with a
collaborator holding still, panning the canvas leaves their cursor pinned to a
*screen* position. That contradicts the component's own doc:

> The positions are canvas-space, so they follow the painting under pan and zoom
> exactly as the paint does.

The `peek` is there for a good reason — subscribing to the renderer signal would
wake the overlay on every engine write — and `use_obs` is the seam that already
answers it. Replace the peek with `use_obs(state, |o| o.view)`: one memo, woken
by a view change and by nothing else.

Note that `PeerCursors`' doc comment is cited by three other overlays as the
precedent for peeking, so fixing it means re-reading those citations rather than
copying the fix: the others are pure layout and read no view at all
(`BrushSizeRing`, `TowStringOverlay`), which is a different and still-correct
argument.

### D2. A browser without WebGPU panics into a blank app

`render.rs:797` — `.expect("request adapter (WebGPU unavailable?)")` — inside the
startup task spawned at `main.rs:206`. The panic kills the task and nothing else:
the chrome renders (every panel falls through the `None` projection), the canvas
stays blank, and the only report is a console trace from
`console_error_panic_hook`.

§5 and `failure.rs` build a careful, undismissable notice for a device that
**dies**, and nothing at all for a device that never **arrives** — which is the
far more common case (Safari before 26, Firefox without the flag, blocklisted
GPUs, a headless CI browser). The asymmetry is the finding: the app already knows
how to say "there is no GPU and your document is still safe", and does not say it
on the one path where the artist has not yet drawn anything to lose.

`init` should return `Result<Renderer, _>` and route the failure into the same
modal. `request_device` at `render.rs:810` and `create_surface` at `render.rs:788`
want the same treatment; `caps.formats[0]` / `caps.alpha_modes[0]`
(`render.rs:917`, `render.rs:925`) are indexes into vectors that a successful
adapter makes non-empty, so they can stay.

### D3. Pop-outs have no dismissal discipline

Three surfaces of one kind behave three ways:

| Surface | Dismissal |
|---|---|
| `main::CommandSearch` | `onfocusout` + `platform::focus_stays_within`, rows on `pointerdown` |
| `panels::filter::AddFilterButton` (`filter.rs:294`) | bare `onfocusout` |
| the frame bar's color pop-out (`frame.rs:422`) | toggle only |
| `panels::gradients::GradientWell` (`gradients.rs:56`) | toggle only |

The last two stay open when the pointer goes elsewhere, and Esc cannot reach
them either: the ladder at `commands.rs:1501` knows root dialogs, composing
modes, composing layers and Timeline mode, and a pop-out's `use_signal(|| false)`
is none of those.

`widgets::Modal` exists so that "what a dialog owes" (§25.7) is written once and
a new dialog inherits it. There is no `Popout` counterpart, so a new pop-out
inherits nothing. The fix is that component — anchor, light dismiss, and a way
for the Esc ladder to see it — plus a line in §25.7 saying a pop-out owes the
same three things a dialog does.

**As built** (`e9bb2cc`), the Esc half only. `widgets::PopoutId` is one signal on
the app state for every pop-out — so two open at once is unexpressible, on
`Composing`'s argument — and Escape's first rung puts it down. It is deliberately
*not* a `Dialogs` flag: that list also stands `FinishMode` down, and the gradient
library opens from a bar while a fill composes, so a pop-out on it would take
Enter's "Done" away. Each well closes its pop-out on unmount, which the locals
got for free and app state does not.

**Light dismiss is still owed**, and the reason it is not a component: the
catcher has to be root-mounted the way `Modal`'s backdrop is, because
`.bottom-bars` carries a `transform` and every bar a `backdrop-filter`, each of
which makes a containing block a `position: fixed` catcher rendered inside the
bar cannot escape. Where it then sits among the z-indices decides which presses
it eats, and eating a canvas press would be worse than the bug — so it wants a
browser (`dx serve` + CDP) rather than an argument. `CommandSearch` and
`AddFilterButton` keep their own bespoke dismissal until then.

### D4. `preview.rs`'s central test covers 6 of 9 rows

`preview.rs:211` (`a_pair_shows_and_lays_the_same_value`) checks
`LAYER_OPACITY`, `LAYER_BLEND`, `MATTE_RECT`, `MATTE_PAINT`, `GUIDE` and
`BACKGROUND`. `FILTER`, `FILL` and `TRANSFORM` appear only in the *clearing*
test, which checks the `None` shape and not the payload.

The module's whole argument is that a mismatched pair is silent — "the canvas
shows the right thing under the hand, and the release lays down something else"
— and the three unchecked rows carry the two largest payloads in the table. Add
the three arms.

Worth considering while there: the test is written per-pair because "what has to
be compared is the payload inside two different enums, which only a `match` can
reach". True — but a `match` that is *not* exhaustive over the table is the same
by-hand list `ALL` is (see **A4**), and it has already fallen three rows behind.

**As built** (`88ed008`): all nine rows, each one call against a `check_pair!`
macro. Still by hand — nothing can make it not be, short of the table itself
being generated — but a missing row is now a hole in a column of nine rather
than a missing paragraph among six.

### D5. Two stale comments

- `main.rs:820` is a truncated duplicate of the comment at `main.rs:848`,
  spliced onto the head of the composing-mode block, so that block now opens by
  describing the layer carry.
- `main.rs:1715` is a `// --- reusable chrome ---` section marker with nothing
  under it. That content lives in `widgets.rs` now.

## Architecture

### A1. `AppState` is ~2 KB passed by value, 272 times

`state.rs:108` holds 86 signals transitively. On `wasm32` in release a
`Signal<T>` is `CopyValue<SignalData<T>>` = `{&'static Storage, NonZeroU64,
ScopeId}` = 24 bytes, so the handle is about 2 KB — and about 2.7 KB under
`dx serve`, where `GenerationalLocation` carries a `&'static Location` for
ownership debugging. It is a by-value parameter on **272** functions and is
re-copied at **57** `use_context::<AppState>()` sites per render.

The surgical fix keeps every call site:

```rust
#[derive(Clone, Copy)]
pub struct AppState(&'static Inner);

impl std::ops::Deref for AppState {
    type Target = Inner;
    fn deref(&self) -> &Inner { self.0 }
}
```

`AppState::new` leaks one `Inner` — which costs nothing that is not already
true, since the root component is never unmounted and `root_signal`'s doc says
so. Field access, the 272 signatures and the 57 `use_context` calls compile
untouched.

The one edit it forces: `Deref` and not `DerefMut`, so the six sites that call
`.set()` straight through `state` take the `let mut x = state.field;` form the
crate already prefers everywhere else — `input.rs:1826`, `input.rs:1832`,
`input.rs:1985`, `input.rs:2033`, `input.rs:2536`, `input.rs:2561`.

**Measure before acting.** LLVM passes aggregates this size indirectly and
elides much of the copy after inlining, so the win is in stack frames and
non-inlined edges rather than in a number a profile will hand over. But the
change is ~30 lines and the type is the hottest in the crate.

**As built** (`54af4e6`): measured first, and the estimate held — `Signals` is
**2752 bytes** on the host, `AppState` **8**. The six mutation sites were exactly
the six predicted. `state::tests::the_handle_is_one_pointer` pins it, and is
written so that a field added to `Signals` cannot fail it while a field added to
`AppState` — which is what reaching for one more flag would do — does.

### A2. `state.rs` names 22 other modules

More than any other file in the crate, `commands.rs` (18) included. The field
*types* are not what costs this — `ThumbState`, `GradientsState` and the rest are
declared in their own modules. It is that `AppState::new` (`state.rs:569`)
constructs every feature's internals field by field, so `state.rs` — the module
everything else depends on — has to know what each of them is made of, and every
new feature field is an edit here.

Give each feature struct its own `new()` calling `root_signal` internally.
`AppState::new` drops from ~110 lines to ~40, `state.rs` names each module once
for its type alone, and the invariant the doc worries about ("every signal here
goes through `root_signal` — the kind of thing a hand-written literal drifts out
of the moment a field is added") is kept next to the fields it is about.

### A3. Composing modes are four signals with a hand-held invariant

`modes.rs` reads `transform`, `guide_edit`, `gradients.armed` and `gradient_bar`
— plus `gradient_resume` as a fifth that must be kept in step — and says so
itself:

> At most one is live once every entry point goes through `leave`; the order
> here is only what an already-broken state would report.

Exclusivity is maintained by every entry point remembering to call `leave`.
Seven call sites do today (`commands.rs:1568`, `gradients.rs:156`,
`gradient_bar.rs:54`, `gradient_bar.rs:83`, `guides.rs:187`, `transform.rs:86`,
`timeline.rs:115`), and nothing catches the eighth that forgets.

One `Signal<Option<Composing>>` with each mode's payload inside its variant makes
two-modes-at-once unrepresentable, turns `leave` into a single `match` on the
taken value, and cuts `composing()` from four subscriptions to one. This is the
crate's own standing preference —

> Rule out a class rather than enumerate its instances.

— applied to the one place that enumerates.

`gradient_resume` stays a separate signal: it is deliberately *not* a live mode
(nothing previews off it, and `modes` never sees it), and that distinction
survives the change intact.

**As built** (`1784c86`), with three things the sketch did not have:
`modes::advance` — the writer a drag sample uses — refuses to change *which* mode
is live, since doing so silently is the very swap the enum removes;
`modes::leave_settled` for a "Done" that has already committed, because dropping
the preview after a commit shows the document without what was just laid;
and `gradients::armed` became a question put to `modes` rather than a second flag
recording that a trace is live. Two orderings turned out to be load-bearing and
are kept explicit rather than left to the `enter` inside `open` — `begin_matte`
still leaves before it reads the projection for its default axis, and `set_armed`
takes the parked bar out of `leave`'s reach before disarming.

### A4. The command registry pays for its metadata six times

34 variants × parallel `match self` in `name`, `word`, `aliases`, `icon`, `hint`
and `shortcut` — roughly 300 lines across `commands.rs:878-1200` — plus `ALL` at
`commands.rs:671`, which §25.2 step 5 concedes:

> **List it in `ALL`** — by hand, and nothing will remind you: a variant left out
> compiles clean and is simply unfindable in the palette.

A `commands! { ... }` macro emitting the enum, `ALL` and those six descriptive
matches from one row per act would make omission impossible and cut the file by
about a fifth. The two payload families (`TogglePanel(PanelId)`,
`SetPickScope(PickScope)`) fold in as rows that name their own `::ALL`, which is
what the existing tests `every_panel_has_a_toggle_row` and
`every_pick_scope_has_a_row` check by hand today.

`run`, `enabled`, `active`, `claims` and `rebindable` stay as matches. They are
code, not data, and §25.2's whole argument about where a gate lives is an
argument for keeping them visible as code.

### A5. `platform.rs` is ~60 hand-paired `#[cfg]` twins in one file

1722 lines in which every function is written twice, and only the `wasm32` half
carries the doc comment — so the host build, which is what `cargo doc`,
`cargo test` and clippy see, is a wall of undocumented stubs.

`platform/web.rs` + `platform/stub.rs` behind one module-level `#[cfg]`, with the
docs on a `platform/mod.rs` façade, gives one file per target and one place the
API surface is stated.

Worth stating plainly while this is open, because the module's own header does
not: the host build **links the stubs**, so the ~118 `target_arch` sites are all
untested logic, and a test that exercises anything through `platform` is testing
the unit-returning half. The module doc is right that gating buys a boundary that
cannot rot; it does not buy coverage, and the two are easy to conflate.

### A6. Module grouping

- `main.rs` (1715) is the entry point *plus* the canvas, five canvas overlays
  (`PeerCursors`, `BrushCursor`, `BrushSizeRing`, `PickLoupe`,
  `TowStringOverlay`), the command rail, the search palette, its bind chip, and
  the new-document dialog. → `canvas.rs`, `overlays.rs`, `rail.rs`.
- `input.rs` (2606) is six gesture types (`Nav`, `Tune`, `PickMove`, `Paint`,
  `Landing`, plus the assist watcher) *plus* window key binding, hover, the tow,
  and sampling. → `input/` with a file per gesture.
- `panels/` holds panels, bottom bars and full-viewport catchers under one name,
  though §11 treats them as three registers with different rules (a panel stacks
  and is remembered; a bar mounts with the thing it acts on; a catcher takes the
  pointer). `panels::gradient_bar::GradientBarOverlay` is neither a panel nor a
  bar.

All three are mechanical, and each makes ownership the docs already describe
visible in the tree.

## Sweeps

### S1. `use_obs` is not finished

It exists for a stated reason:

> a component that calls `state.obs.read()` in its body re-renders on **every**
> command: a pan, a brush-size drag, a transform preview, each sample of an
> eyedropper drag.

Nine sites use it; thirty call `state.obs.read()` directly. The clearest live
case is `navigator.rs:332`: the overlay carefully memoizes `subject` for exactly
this reason and then reads the whole projection fifty lines later for `o.view`
alone, so a brush-tuning drag or an eyedropper drag re-renders it at pointer
rate. Same shape, lower stakes, at `frame.rs:633`, `transform.rs:408`,
`gradient_bar.rs:403`, `guides.rs:983`, `gradients.rs:293` and `main.rs:1483`.

Three handler-side reads should be `peek` rather than `read`, on the crate's own
rule that a handler has nothing to re-run: `frame.rs:112`, `frame.rs:128`,
`gradient_bar.rs:116`.

**Structurally:** make `obs` private to `state.rs` and expose only `use_obs`
(renders) and a `peek_obs` (handlers). A raw render-body read then does not
compile. This is the same move `ReadOnly` already made for the renderer, and for
the same failure — a component asserting something the engine has moved past, or
waking for something it does not read.

**As built** (`86ee0fe`), the sweep but **not** that structural half, and the
reason is worth recording because it is not effort. A subscribing read is
legitimate in a plain helper called from a render body — `commands::armed`,
`panels::guides::guides_of` — and only a *hook* can narrow, which a helper called
per row inside a loop cannot be. `Command::active` is exactly such a helper. So
"no `read` outside `state.rs`" would have forced `Command::active` to stop being
a function, which is out of proportion to what it buys. Four subscribing reads
therefore remain on purpose; every other render-body read is a memo.

What did come out of it: `use_obs_opt` for readers with their own answer to
there being no engine yet, and two components split so that a memo could exist
without paying for itself when nobody is looking — `PaletteRow`, so a palette row
narrows the way the rail's menu rows already did, and `SlotRack`, because
`SlotOverlay` claims in a comment to read nothing at all while the rack is away
and a hook above that early return would have made it false.

### S2. Untested pure geometry, where the split pattern already exists

`panels/layer_tree.rs` and `panels/reorder.rs` were split out as "the half that
can be tested", and `gesture.rs` holds the transform math with tests. Three
places still hold pure arithmetic inside a panel that has none:

- `panels/filter.rs` (1194 lines, no tests): `dial_xy`/`dial_ab` and
  `pad_radius`/`pad_spread` are two **inverse pairs** whose round-trip is exactly
  the invariant that fails silently — as a dot drifting away from the pointer
  under the hand — plus `snapped`.
- `panels/frame.rs`: `to_aspect`, `matched_aspect`.
- `navigator.rs`: `viewport_style` and the press-to-canvas mapping.

Three small extractions on a pattern the crate already runs twice.

**As built** (`f784fdb`), the tests were added in place rather than by extracting
the functions, which turned out to be the cheaper half of the same benefit: what
was missing was the pinning, not the file boundary. Seventeen tests over the four
mappings, and two of them check a *property* rather than a number — the fringe
pad spends equal area per unit of spread, the dial is warm at the top — so the
constants may move and the claims the docs make cannot.

## What was checked and found sound

Recorded so the next reader does not re-derive it:

- **Borrow discipline.** Every `write()`/`peek()` guard that could outlive a
  dispatch is bound to a local first, and the reasons are written down where it
  matters (`modes::leave`, `state::with_engine`, `layout::set_open`,
  `Landing::advance`). Nothing found held across a re-entry.
- **`spawn` vs `spawn_forever`.** Every scope-tied `spawn` is deliberate and says
  so — the export dialog's is the clearest (`files.rs:283`), where cancellation
  on dismissal is the intended meaning.
- **The engine projection is not the pointer-rate cost it looks like.**
  `Engine::observe` caches the layer walk behind `projected_layers`, and
  `with_engine` publishes only when `ObservableState` actually moved. What is
  left to pay is the memo fan-out, which is **S1**'s business rather than a
  projection split.
- **`request_paint` back-pressure** (`MAX_FRAMES_IN_FLIGHT`, `frame.skipped`) is
  the right shape for a backend where `present` is a no-op.
- **The touch ladder** (`Landing`, `tap_of`, `TOUCH_SLOP`) is coherent: one
  constant behind the tap and the paint threshold, and a test that pins the two
  cannot drift apart.

## What the fixes were checked against

Every gate the project names, after the last commit on this branch: `cargo fmt
--all --check`; `cargo clippy --workspace --all-targets -D warnings` in **both**
configurations, the default and `--no-default-features --features
stark-net/webrtc`; `cargo nextest run --workspace` — 1105 tests, all passing;
`cargo test --workspace --doc`; and `cargo check -p stark-ui --target
wasm32-unknown-unknown`.

**What none of that covers is a browser.** Nothing in this crate's suite mounts a
component, so every change to what a surface *shows* — the memos in **S1**, the
new report arm in **D2**, the pop-out flag in **D3** — is checked by construction
and by the compiler, and not by having been looked at. Three of them are worth
looking at before this is relied on:

- the **no-WebGPU report** (`failure::NoGpu`), which by its nature only appears
  on a browser that cannot run the app at all — the way to see it is to make
  `render::init` return `Err(StartupFailure::Adapter)` unconditionally and load
  the page;
- the **mode refactor** (**A3**), where the thing to exercise is each way out of
  each mode: Done, Escape, and reaching for another mode mid-composition — and in
  particular arming a trace from the gradient bar and coming back to it, which is
  the one park-and-resume in the app;
- the **pop-out flag** (**D3**), where the thing to check is that a bar going
  away takes its pop-out with it.
