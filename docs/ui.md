# The frontend

The Dioxus app and the wgpu surface it wraps — §11 — and the chrome's registries:
commands, chords, drag bindings, the browser-local store and the shape of a
dialog, which a new UI feature joins and how — §25.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 11. Frontend (Dioxus)

`stark-ui` is a Dioxus 0.7 **web** app: the backend runs in WASM and the painting
surface is a dedicated `wgpu::Surface` bound to the page `<canvas>` via WebGPU,
which the engine draws into directly. DOM chrome surrounds it.

- UI components dispatch `InputCommand`s through one seam, `state::dispatch`,
  which applies, repaints, refreshes `ObservableState` and broadcasts whatever
  was committed — so no call site has to remember that sequence. **No call site
  *can* forget it**: the renderer and its projection are held in `state::ReadOnly`
  handles, which have `read` and `peek` and no `write`, so `&mut Renderer` is
  unreachable outside `state.rs`. The doors are `dispatch` and `with_engine`,
  which publish the projection on the way out, and `with_engine_quiet`, for the
  `&mut` that cannot change what `observe()` projects — rendering, the readbacks,
  the outbox and presence drains, installing asset bytes an action will later
  name. That is a type where a convention was: twice, a panel reached the engine
  through the signal, moved state the chrome reads back, and left the chrome
  asserting the old value until an unrelated command refreshed it — once for the
  canvas ground, once for the lighting environment. Neither spelling compiles.
  Pointer events
  become `GestureCommand::Start`/`To`/`End`, with element coordinates mapped via
  `ViewTransform::screen_to_canvas`. `Start` also carries the **input tolerance**
  (§6.2): `devicePixelRatio` and the event's `pointerType` give the device's
  grain in CSS px, and the same view transform carries it into canvas px. A
  fourth, `Hold`, is sent by the dwell watcher when the pointer stops moving
  mid-stroke — the drawing assist (§6.9). It is the frontend that has the clock,
  and the moves after it are still plain `To`s.
- **Navigation is one vocabulary, asked three times.** `input::Nav` owns every
  binding that moves the view — the two-finger gesture (§18.1.7), middle-drag and
  space-drag pan, wheel zoom — and every surface over the canvas makes its own
  (the canvas, the transform mode's catcher). Its three entry points are a
  lifecycle, `begin` / `advance` / `release`, and each answers the same question:
  *was this event mine?* So a surface routes its pointers by asking three times
  and never by inspecting buttons or pointer types itself, and what "the pan
  bindings" and "the zoom rate" mean cannot drift between surfaces. Policy stays
  at the call site — the canvas fades the chrome while it navigates and cancels
  the stroke a second finger interrupted, the transform overlay deliberately does
  neither.
- **A simple command is a row in a registry, not an arm in a match.** `commands`
  declares every simple act — one the chrome can ask for whole, with no argument
  at the call site — as a variant of `Command` carrying its entire description:
  display name, terse chip word, mark, tooltip, availability
  (`Command::enabled`, what a row greys on), the mode tick (`Command::checked`),
  the live mark (`Command::active` — Share while a session runs, worn as the
  icon taking the select blue rather than as a tick),
  the advertised shortcut (`Command::shortcut`), the gate its act must ask
  (`Command::run`), and the chord that reaches it from the keyboard
  (`commands::Bindings`). The chrome *renders* a command rather than restating it — a bar
  chip or panel-header button is `widgets::CommandButton`, a menu row is the
  rail's `CmdItem` — so what a menu claims, what a chip does and what the
  keyboard answers cannot drift. They had: the menu's Undo skipped the
  keyboard's stop-playback resolution, and its Deselect skipped the gate below
  outright. What is deliberately *not* a variant is anything aimed at the
  document's own rows — this layer's eye, that guide's trash name a target only
  the document knows, and a registry of every (act, target) pair would be a
  second copy of the panels. A payload from the chrome's own closed set is
  different: `Command::TogglePanel(PanelId)` makes each panel's toggle one
  nameable act — the Panels menu draws its rows from it, a search for "panel"
  lists the whole stack, and any of the six can be given a chord.
- **The rail's first entry is the registry, searchable.** `main::CommandSearch`
  is a field over `commands::search`: at rest it offers the file family
  (`commands::BASIC` — the acts with no muscle-memory home anywhere else), and a
  query narrows it over `commands::ALL`, prefix matches first and display names
  before aliases. An alias (`Command::aliases`) is what other software calls
  the same act — "Flip" for Mirror view, "Preferences" for Settings, "Crop" for
  Add frame — searched but never printed: the alias does the finding and the
  name does the teaching, so a hand trained elsewhere both reaches the act and
  learns our word for it.
  Arrows move the highlight, Enter runs it, a row acts on `pointerdown`, and
  every row is drawn from the registry like the menus' own. It replaced the
  catch-all ☰ menu — Undo now advertises its Ctrl+Z in the row a query turns
  up. Deliberately *not* a third `MenubarMenu`: the vendored trigger
  light-dismisses its menu when DOM focus leaves it for anything but a menu
  item, and this surface exists to hand focus to a text field — so it is the
  filter picker's own-dropdown arrangement, plus one question that pattern
  never had to ask: `platform::focus_stays_within`, so focus hopping from
  trigger to field on open does not read as dismissal.
- **A chord names its key the way the binding means it.** A chord names the
  accelerator tier (Ctrl or Command, `input::accel`), the Shift bit, and a key
  that is either the *character* it types — a mnemonic follows the layout,
  because Z undoes wherever the layout puts the Z — or the *position* it sits
  at — a spatial pair is about adjacency, and `[`/`]` step the brush precisely
  because they are side by side (`slots::of_code`'s argument, §18.1.8). Chords
  are exact: Ctrl+Shift+Z is its own row rather than Ctrl+Z plus a bystander,
  and Alt in any combination matches nothing, since AltGr arrives as Ctrl+Alt
  and a table that shrugged at Alt would fire its Ctrl rows under a layout's
  ordinary typing. The keydown handler asks the table once and claims a matched
  chord wholly (`prevent_default`) whether or not its act was accepted — a
  declined Ctrl+A must not answer with the browser highlighting the page.
  What is *not* a chord row is anything owning both edges of its key — a held
  digit (§18.1.8), space's pan, Alt's eyedropper stay in `input`, which owns
  keyup — and Ctrl+V, which is not a command but data arriving (§23).
- **The chord column is the user's** (`commands::Bindings`). The shipped rows
  are only defaults; this browser's rebindings lie over them as a signal on the
  app state, stored under their own key like the preset library, and `find` and
  `Command::shortcut` read the overlaid table — so a rebind moves what the
  keyboard answers and every advertisement of it in the same write. Only the
  *overrides* are stored, keyed by the variant's name: stealing Ctrl+Y leaves
  Redo advertising Ctrl+Shift+Z with nothing stored about Redo, and a default
  added in a later build reaches browsers that stored their table before it
  existed. Rebinding lives in the palette rather than a settings page: a row's
  shortcut is a chip (a `+` where there is none), click it and press the new
  chord — the field keeps the keyboard and `commands::capture` reads the one
  keydown, refusing what could never fire (Alt combinations, space and the bare
  digit holds, the paste's Ctrl+V), taking Escape as the way out and Backspace
  as the eraser: an unbind is the same gesture as clearing a field, at the cost
  that no shortcut can be Backspace itself. A capture
  names a character key by what it types and anything else by where it sits,
  which is `Chord`'s own mnemonic/spatial split applied at the moment of
  capture. The one advertisement that will not move is Import's Ctrl+V — the
  browser's paste is true whatever the table says
  (`Command::rebindable`).
- **A composing mode is a state, not a stacking order.** Four gestures take the
  canvas away from the brush for the length of a composition: the transform
  widget (§16.6), the perspective-guide edit (§20.5), the gradient trace (§22.2)
  and the gradient fill's axis (§22.4). Each mounts a full-viewport catcher, and
  a catcher is a claim about *hit testing* only — which is not the only way a
  pointer reaches the canvas. A gesture already in flight has **captured** its
  pointer, and a captured pointer's moves and its release are delivered to the
  element that took them whatever has been stacked over it since; a pen drawing
  while the other hand opens a transform kept feeding the fitter under the
  widget and committed on release. So `modes::composing` makes the question
  askable in Rust: the canvas cancels a gesture the moment a mode opens under it
  (cancel, not commit — the canvas stopped taking paint, so it must leave no
  mark), and chrome stacked above the catchers stands down rather than floating
  grips over them.
  - `modes::leave` is the other half. Every entry into a mode leaves whichever
    was already live, dropping its preview and committing nothing, so "two modes
    composing at once" is unreachable rather than arbitrated — the four catchers
    share one `z-index`, where the *last* sibling takes the pointer, so the
    priority the comments had claimed ran backwards. Entering Timeline mode and
    an undo or redo from the keyboard call it too: a preview is computed against
    the committed document, and moving the playhead out from under one leaves the
    widget pointing at paint that is no longer there.
  - **Every door asks what the canvas asks.** A press is refused while the
    playhead is moving, because a commit clears the withheld half of the timeline
    (§18.2.4) — but `Ctrl+A` and `Ctrl+Shift+I` went through and truncated the
    history from the keyboard, and edited the very selection a transform was
    composing against while the bar carrying those same commands had stood down.
    They now ask both questions — and because the gate lives on the act
    (`Command::run`) rather than at a call site, the menu rows and bar chips that
    carry the same acts ask them too, which the menu's Deselect once did not.
    Undo and redo are not refused but *resolve*: nothing on screen says they are
    unavailable, so instead they stop playback, put down the composition, and
    act.
- The engine (and its `wgpu::Surface`, both `!Send`) live in a signal; after each
  command the engine renders **straight into the surface texture**
  (`get_current_texture` → `render` → `present`) — no readback, no encode. The
  frontend supplies GPU handles via `GpuContext::from_parts`; core needs no
  change to compile to wasm.
- **The floating chrome fades while the canvas is in hand** — a stroke, a
  selection drag, a pan, or a run of wheel zooming — and fades back the moment
  the gesture ends. One signal, `AppState::canvas_active`, toggles a `dimmed`
  class the stylesheet animates; the chrome keeps its box (nothing reflows) and
  stops taking clicks while faded, so a stroke straying under a panel keeps
  painting.
- **The panel stack stays away until it is reached for.** Fading back on release
  put the panels over the painting at the one moment the artist was looking at
  what they had just drawn, so the stack alone keeps its fade after the gesture
  (`AppState::panels_asleep`, set by `input::end_interaction` where — and only
  where — the fade was actually in force) until the pointer enters the *column*
  it lives in. That column is a full-height slice of the window, which the stack
  itself cannot be: it is exactly as tall as its panels and has to stay that way
  for its own scroller, so `.panel-wake` is a separate box, mounted only while
  the panels are asleep and no gesture is in flight — the two conditions that
  keep an invisible box over the painting from swallowing a stroke's moves or the
  release that ends it. It answers what it does take: the first press or wheel in
  a sleeping column brings the panels back rather than being lost, which is also
  the only way a touch-only hand can ask for them. Opening a panel wakes the
  stack too (`layout::open_panel`, the one door), or the command would tick a
  menu entry and show nothing. Nothing else sleeps: the rail and a mode's bars
  and handles live elsewhere on screen, and latching those off until the pointer
  had visited the right edge would be controls you cannot reach by reaching for
  them.
- **How much of that happens is the artist's to choose** (`layout::ChromeHiding`,
  a ⚙ row): *Always show*, *Hide while painting*, *Hide after painting*. The two
  mechanisms above were one behavior with no switch, and which of them is wanted
  turns out to be a fact about the hardware rather than a matter of taste — a pen
  on a tablet crosses the panel column on the way to everything, so the wake
  gesture costs it nothing and it gets the whole window back, while a mouse
  reaching for a slider between strokes pays the reach every time. Three states
  and not two switches, because the fourth combination does not exist: a stack
  that stays down after a gesture it never faded for is a panel vanishing at the
  moment the artist stopped painting. `AppState::canvas_active` still says the
  canvas is in hand whatever this holds — half a dozen things ask that for their
  own reasons — so what the setting changes is who *looks* faded: one function
  decides the class (`layout::fading`) and one decides whether the stack sleeps
  (`layout::sleep_panels`, the one door, so the release that ends a stroke and the
  tour revealing its own lesson both obey it). The tour's card about the wake
  gesture steps aside entirely where the gesture is switched off — a lesson that
  can neither be shown nor dismissed would stall the three behind it
  (`tutor::Lesson::applies`, asked in `due`).
- **Every panel starts closed, and what is open follows the browser.** The
  opening screen is the painting and nothing else; the panels that come back next
  visit are the ones the artist actually reached for (`layout::stored_hidden`,
  keyed `stark.panels.v1`). The two halves are one decision — a set of panels
  chosen for you is only tolerable because it resets every visit, and once the
  choice sticks the honest starting point is none. What keeps "none" from meaning
  "hidden" is the tour: the Color panel arrives on the second stroke and the wake
  gesture is explained on the third (§24.5). Durability is structural rather than
  remembered — `layout::set_open` is the only thing that writes the hidden set and
  it persists after every change, so a new way to close a panel is durable without
  its author thinking about storage, the same move `settings::SettingToggle` makes
  for the preferences. The **open** set is what is written, not the hidden one, so
  a panel added in a later release arrives closed like everything else instead of
  appearing unbidden in the stack of every existing user.
- **Panels are first-class.** `PanelId` + a `PanelLayout` context (order, hidden
  set, folded set, drag state, mounted refs) make the stack data-driven: each
  panel has a header with a drag handle and a ✕, a "Panels" menubar menu reopens
  closed ones into their original slot, and dragging a title bar reorders with a
  FLIP animation (measure previous tops, apply an inverted transform with no
  transition, then play to zero). `key:` on each panel must be the stable
  `PanelId` so reordering *moves* existing nodes rather than recreating them —
  that is what preserves each panel's internal signal state and makes FLIP
  possible. Lives in `layout.rs` + `assets/stark.css`.
  - **A title bar is one grip with two gestures**: a press that travels reorders,
    and a press that does not **folds the panel to its bar**
    (`layout::release_title`, told apart by the drag's own `GRAB_SLOP` so there is
    no distance in between that does neither). A fold is deliberately not a close
    — the panel keeps its slot, its height and its subtree, which is why the
    content is hidden by the stylesheet rather than left out of the render: one of
    them owns a `wgpu::Surface`. It is hidden by one rule about the *wrapper*
    around that content (`.panel-body`, `display: contents` so it is a selector and
    not a box), never by rules about the children: minimal mode re-lays some of
    those children by name, out-specifies anything aimed at the same elements, and
    left a folded panel showing its sliders. A hidden ancestor cannot be undone by
    a rule about a descendant, whatever its specificity. It is remembered with which panels are open, as a
    second *field* on the same stored line, so the two states cross a version in
    either direction without a migration. Opening a panel unfolds it: a panel that
    came back as a bare title bar would be a menu entry that ticks itself and shows
    nothing.
  - **Nothing in the stack is selectable text** (`user-select: none` on
    `.panel-stack`, form fields excepted). Leaving it selectable cost the reorder
    drag: a title-bar drag that travels sweeps a selection down through whatever
    the pointer crosses, and the *next* press on that selection starts the
    browser's own text drag instead — a transparent ghost of the panel's words
    following the pointer while the panel stays put, with no way to clear it
    anybody would guess. `.panel-title`'s own `user-select` could not prevent it,
    because the selection it makes is in everything *below* the grip.
  - **The column has its own scrollbar** (`layout::PanelScrollbar`), in the strip
    of padding down its right edge, shown while the pointer is in the column and
    draggable. The wheel was the only way to reach the rest of a column taller than
    the window, which is fine for a mouse and no use to a hand holding a pen. It is
    the app's own rail rather than a styled `::-webkit-scrollbar` because a native
    one takes its width out of the *content* box: every panel would narrow by ten
    pixels the moment the column overflowed and widen again when one closed. It is
    a **sibling** of the stack, not a child — an absolutely positioned child of a
    scroll container scrolls away with the content it describes — so its box comes
    in inline from the measurement (`layout::Scroll`, re-read on the stack's scroll
    event, on the pointer arriving, and after any render that could change the
    column's height).
- **Z-order is declared, not inherited from DOM order.** The canvas overlay
  (frame handles, transform widget) sits at `z-index: 10`; every piece of
  floating chrome is 20+. `.panel-stack` must declare its `z-index` explicitly —
  with none it sits at auto level and any positioned sibling with a `z-index`
  beats it regardless of DOM order.
- **The navigator's miniature is a second surface, not an image the UI carries.**
  An overview is the one piece of chrome that cannot be derived from
  `ObservableState` — it is pixels — so `panels/navigator.rs` mounts its own
  `<canvas>`, the frontend binds a second `wgpu::Surface` on the app's existing
  device, and the engine renders into it (`Engine::render_into`). It frames
  itself against the rect a file export would use (`Engine::export_plan`, whose
  returned plan *is* the view it renders through), so the overview cannot come to
  disagree with the picture a file would hold.
  - It subscribes to `ObservableState::doc_revision`, **not** the canvas: a
    refresh composites every tile, affordable per *edit* and ruinous per pointer
    sample. That counter moves when the committed document does and deliberately
    not when an in-flight gesture or an unlogged drag preview changes what the
    canvas shows. `Rendered::Committed` is the matching half engine-side. A
    settle delay collapses bursts, and a refresh due mid-gesture waits for the
    hand to lift. The viewport rectangle is a positioned `<div>` read from the
    live view, so pan and zoom move it for free.
  - This began as an `export` — render offscreen, read pixels back, hand them to
    a 2D canvas via `ImageData` — and every part of it after "render" existed
    only because the miniature had nowhere of its own to draw. Giving it a
    surface deleted the GPU→CPU copy and its frame of latency, the pixel buffer
    in a signal, the `putImageData` helper, and the imperative repaint that had
    to re-run whenever the element remounted.
- **The layer panel's thumbnails go back the other way, and the reasons invert**
  (`layer_thumbs.rs`, §14.6). They are pictures of the live document like the
  navigator's, so it would seem to follow that they want surfaces too — but there is
  one navigator and there are as many of these as the document has layers, so a
  surface each is a WebGPU context and a swapchain per row; the rows are moved and
  re-keyed by the drag that reorders them, and a CSS `background-image` survives a
  node moving where a bound surface is per-node state to rebind; and the frame of
  latency a surface buys off is worth nothing on a 20 px picture refreshed once per
  commit. So these are `Engine::export_view` readbacks into `data:` URLs, exactly as
  the *brush* thumbnails are. Three consumers, three answers, from one question asked
  each time: does this render repeat, how many of it are there, and who is waiting.
  - **The engine change was one parameter.** `composite_groups` had answered "just
    this layer" since the eyedropper needed it (§18.0.2) and `DrawKey` already keyed
    on it; nothing had ever asked for it as a *picture*, so `render_view` passed
    `None`. Threading it through is the whole of the feature engine-side, which is
    what a seam in the right place looks like from outside.
  - **The cost that is real is the draw cache's single slot.** An isolated render
    evicts the screen's list, and rebuilding that clones a tile handle per visible
    tile per layer — so generation is paced one row at a time and stands down while
    `canvas_active`. The cache key is the layer's own tile revision rather than
    `doc_revision`, so a stroke re-renders one row instead of all of them.
- **Compositing splits along "does it depend on the target?"**, so a second view
  costs only its own attachments — and again along **"does it ever change?"**, so
  a second *engine* costs only its own view settings. `CompositorPipeline` holds
  the view settings the media pass reads over an `Arc` of `CompositorPasses` —
  the pipelines, layouts and pigment LUT, immutable once built and therefore
  shareable across engines (`Engine::new_sharing`); a `Compositor` holds one
  target's offscreen attachments, blend scratch and instance streams.
  The surface keeps one across frames; anything rendered beside it brings its own
  through an `Offscreen` slot, so an off-screen render never resizes the screen's
  attachments out from under it. Whether a slot outlives its call is the
  *caller's* to state, because only the caller knows whether the render repeats:
  the navigator holds one for the app's life, while a file export uses a local
  one so a 4× export of a large frame does not park its several-hundred-megabyte
  pair for the session. View settings stay per-pipeline behind a process-wide
  generation stamp, so a swapped weave or light — or a whole rebuilt pipeline, as
  a color-space change makes — reaches every consumer by being *noticed* rather
  than by a notification a new consumer could be left out of.
- **A preview engine is a sibling, not a second boot.** The brush editor's test
  canvas and the preset thumbnails each want an isolated document that renders
  *exactly* as the main canvas would — which is an argument for sharing the
  machinery, not merely an economy. `Engine::new_sharing` builds one around a
  fresh document, sharing everything expensive and un-disagreeable: the compiled
  pipelines (immutable), the content-addressed brush assets and the ground /
  environment byte-and-build caches (a `Registry`'s store is `Arc`-shared while
  each sibling keeps its own *current* id), and the tile pool (an allocator).
  What an engine can set stays its own. So the editor's preview
  (`Renderer::shared`) opens on the canvas's ground under its lighting with
  nothing fetched and nothing decoded, and the thumbnails' engine
  (`Renderer::shared_engine`, `thumbs.rs`) deliberately pins the opposite look —
  flat ground, neutral light — so a thumbnail is the *brush's* identity card and
  its cache key is the brush snapshot alone. The snapshot *as the picture paints
  it*, which is the same thing minus the painting color: the stroke is laid in
  the thumbnail's own red whatever RGB it is handed, so keying on the raw
  snapshot would file one picture under a fresh name for every color the artist
  happened to be holding — and a quick slot assigned under a hold stores the
  preset's brush wearing today's color (§18.1.8), so that is not a theoretical
  duplicate but the row of the slot you just filled, blank while it waits on a
  byte-for-byte copy of the thumbnail beside it. The brush's own opacity does
  stay in the key: the stroke really is laid with it (§6.1). Each row's picture is two
  half-canvas fills (the ground is all paint, so smearing and lifting read), one
  replayed stroke and one small `Engine::export_view` readback on that one kept
  engine, generated in the background and cached per session. The key being the
  brush is what lets the cache have two viewers for the price of one: the preset
  library's rows and the quick-brush rack the number keys draw (§18.1.8) show the
  same picture of the same brush, and a slot that came from a preset is never
  rendered twice. The generator therefore belongs to the app **root**, not to
  either viewer — the Brush panel closes, and the rack's overlay exists only
  while a key is held, which is far too late to start rendering what it is there
  to show.
- **The color picker shows the gamut, not the box the gamut fits in**
  (`panels::color`, one control in three homes — the Color panel, the frame bar's
  matte well, the Lighting panel's substrate). It used to draw a fixed square of
  the Oklab `(a, b)` plane, ±0.32 on each axis, because that is the box the *whole*
  sRGB gamut fits in. But a picker shows one lightness at a time, and one lightness
  is a thin slice of that box: at `L = 0.61` the slice is 28% of the square, at
  `L = 0.2` it is 3.8%. The rest was colors the display cannot show, drawn clamped —
  a flat wash answering every position with the same color, with everything the
  artist was choosing between crowded into what was left. That is the whole of
  *"the picker is too sensitive and I cannot see what I have"*: it was not the
  gain, it was that four fifths of the control did nothing.
  - **So the radius is chroma as a fraction of what this lightness and this hue can
    hold.** The rim *is* the sRGB boundary, found by bisecting this build's own
    conversion rather than by a fitted curve; every point inside is a distinct color
    the display has, and the panel gives the choice between 2.8× and 20× the room it
    did. Two things fall out. Dragging `L` now travels along a hue at constant
    relative chroma instead of walking a fixed `(a, b)` in and out of gamut — *the
    same color, lighter*, which is the move a painter makes constantly, and it is why
    the lightness track is drawn per row rather than handed to CSS as a
    `linear-gradient(in oklab …)`, which would clamp flat at both ends. And no state
    the picker can hold is out of gamut, so nothing it shows is a lie about what the
    brush will lay down.
  - **Which gamut: sRGB, because that is the one the document has.** Color enters
    through `srgb_to_oklab` and leaves through the media pass to an sRGB surface
    (§6.5); a picker fitted to Display P3 would offer chroma that the ingest clamps
    away — the same lie as the old square, moved from the corners to the rim. The
    day the pipeline goes wide, the fit is one predicate (`in_srgb`) and the docs
    are one paragraph.
  - **The costs are stated where they are paid.** A slice at constant lightness has
    corners, because the sRGB cube does, and per-hue normalization wears one as a
    crease near the blue primary — kept rather than smoothed, since shaving it puts
    `#0000ff` outside the wheel and rounding it outward puts clamped color back
    inside (`RIM_N`). And the slice is not quite star-shaped about the achromatic
    axis: the blue corner sits behind a hairline gap that a search from the centre
    stops at, which one thousandth of a linear channel bridges (`GAMUT_BRIDGE`).
  - **What is chosen is said, in a patch and in hex.** A 12px ring sitting on a
    color is not a sample of it, and the cursor was parked on the one pixel the
    control is about — so the pointer goes invisible for the duration of a drag, the
    lightness marker became two carets biting in from the track's edges instead of a
    line across the color it points at, and the readout under the picker carries a
    well and an editable `#rrggbb`. The hex field is also the answer no drag can
    give: a color named exactly. **Shift** on either track is the other half —
    the value moves *with* the pointer at a fifth of its travel, from where it
    already stood, so the press picks nothing and the hand spends the whole control
    on a fraction of its range.
- **Settings are one dialog, not a control tucked into whichever panel it came
  from.** Panels hold what you are painting *with* and change constantly
  mid-stroke; document dialogs hold what the drawing *is*. A standing per-client
  preference is neither — set once, never part of the artwork — so it lives in
  the ⚙ dialog off the command rail (`settings.rs`). Its rows apply on the click
  (Done, no Cancel: nothing is staged) and stay mounted even when inert, saying
  so in their own text — deliberately the opposite of the §6.8 rule for tool
  bars, because a settings dialog is read as the map of what is configurable.
  What every dialog owes its reader — the height it may take, what holds still
  while the rest scrolls — is §25.7 below, with the rest of the chrome's rules.
  - **They are saved on the click too** — per browser, in `localStorage`, never
    into the document and never to peers (`prefs.rs`). "Set once and left alone"
    is a promise a reload would otherwise break, so the settings follow the
    browser the way the shape and preset libraries do, and degrade to a
    per-session choice where storage is unavailable. One serde struct holds the
    lot: `#[serde(default)]` per field is what lets a preference added later read
    as its default out of values stored before it existed, instead of a parse
    failure resetting everything the user had set. The dialog's rows do not opt
    in — the toggle component persists after calling its handler, so a new row is
    durable by construction and only its *value* has to be named. Loading
    happens in two passes for the one preference that is engine session state
    rather than a frontend signal (peer selection outlines, §17.3): the frontend
    half applies in the root's body so the first render is already in the right
    mode, and the engine half waits for the renderer, exactly as
    `presets::load`/`apply_first` split.
- **Timing Stats is a dialog, not a panel** (`timings.rs`, §7.1). The same
  argument as settings run from the other end: a live frame-rate readout beside the
  canvas is a thing to watch *instead of* painting, and the histograms behind it
  keep accruing whether or not anyone is looking — so it belongs behind the command
  search with the other things read when a question comes up. It renders whatever rows the
  engine's histograms hold rather than a list of phases, which is what keeps it from
  becoming a second copy of the instrumentation; it polls twice a second while open
  and stops when it closes (a `use_future` in its own scope — the opposite of what
  the collaboration pumps need, and for the opposite reason).

- **The app installs, and it starts offline.** `index.html` (the crate root's
  own, which replaces the one `dx` would generate) links a web app manifest and
  registers a service worker; both, with the launcher icons, live in
  `stark-ui/public/`, which the CLI copies to the **site root** unhashed —
  unlike `assets/`, whose every file is renamed by content hash. That
  distinction is the whole design of `public/sw.js`: the navigation response
  names one build's hashed wasm, so it is fetched **network-first** and only
  falls back to cache; everything else same-origin is content-addressed and so
  can never be stale, and is served **cache-first** while a background fetch
  refreshes it. Cross-origin, non-`GET` and range requests are passed straight
  through, so the collaboration transport (§12.4) and any partial fetch are
  untouched.
  - This matters more here than for a typical app because the heavy assets are
    deliberately *not* in the wasm binary: the brush stamps (§6.6), the ground
    height maps (§6.4) and the environment HDR (§6.3) are all fetched after boot
    by `builtins::import_all` / `grounds::open_default`. Without the worker that
    is a fresh multi-megabyte download every start, and offline it is a document
    that opens smooth and unlit.
  - The custom `index.html` costs one thing: `dx serve`'s "rebuilding" toast,
    which lives in the CLI's dev shell. Hot reload itself rides the wasm glue and
    is unaffected. Note also that the CLI resolves its placeholders by literal
    text match over the whole file, comments included.
  - **A `.stark` opens in it.** The manifest declares a `file_handlers` entry for
    the extension, and `files::bind_file_launch` takes the other end: the
    browser's `launchQueue`, reached by reflection because neither it nor the
    `FileSystemFileHandle` it yields is in stable `web-sys`. Setting the consumer
    is what *delivers* a queued launch, so it is bound at the end of the startup
    task — a document has nowhere to load until the renderer exists. The manifest
    asks for `focus-existing` rather than `navigate-existing`, so a second launch
    reaches the *running* app: a reload would throw the open painting away before
    the new file had even been read, whereas this path refuses a bad file with
    the current one still on screen. Only the first file of a launch is taken —
    opening a document replaces the canvas (§8), so a second would be a painting
    nobody ever sees, which is what `single-client` says too.
  - The icons are painted with Stark's own bristle stamp by
    `stark-ui/tools/make-icons.py`, run by hand; the PNGs are checked in. Not a
    build step — a logo is not a build product, and nothing keys off its bytes
    the way `stark-assetid` keys off an asset's (§19).

Because the engine is frontend-agnostic, this layer stays thin. (An earlier
interim cut ran on Dioxus *desktop* and bridged the canvas by reading the frame
back to a PNG data URL — correct but laggy; the WebGPU surface replaced it,
touching only `stark-ui`.) Run with `dx serve --web -p stark-ui` in a WebGPU
browser. A native winit/desktop frontend could reuse the same engine.

## 25. Commands and drag bindings

The chrome reaches every act and every bound gesture through one of two tables.
The **command registry** (`stark-ui/src/commands.rs`) holds the simple acts —
undo, deselect, open the export dialog — each a variant of `Command` carrying
its whole description, with the keyboard as one rebindable column. The **drag
table** (`stark-ui/src/drags.rs`) holds the canvas presses that open something
other than painting — the brush-tuning drag (§18.1.9), the eyedropper (§18.0.2),
the layer carry (§16.11) — each a row binding an exact chord+button to a
`DragAction`. A third registry sits one layer down and answers a different
question — `Store` (§25.6), which enumerates every record the frontend keeps in
this *browser*: the libraries, the settings, the chord table above. §11 above tells the
first two designs' history; this chapter is the working guide, written for the
day a feature is added: which table the feature belongs to, if any, and the steps
that keep them true. §25.7 closes it with the one surface that joins no registry
and still has rules — the dialog.

One law covers all three, and it is the reason they exist: **one authority, and
surfaces render it rather than restating it.** Each table has one reader on its
dispatch path — `commands::find` on the window's keydown, `drags::find` on the
canvas's press — and every advertisement of a binding (a row's shortcut column,
a tooltip's parenthesis, the eyedropper cursor, the options bar) is printed
from the same table. So what a control claims, what a key answers and what a
press does cannot drift apart. Every rule below is that sentence applied
somewhere, usually somewhere it was once broken: the rail's menu dispatched
Deselect without the gate the keyboard asked, and the picking cursor once
promised a sample that a press with a second modifier held would not have
taken.

### 25.1 Which table, if either

The sorting question comes first, because the most common mistake is not a
missing method — the compiler catches those — but a feature wired as the wrong
kind of thing. Ask what the feature *is*:

- **A simple act** — it takes no argument at the call site, or its argument
  comes from a closed set the chrome itself enumerates (`PanelId`). Click it,
  press its chord, pick it from a menu, and it is the same act reached three
  ways. → A `Command` variant (§25.2).
- **A gesture** — a press opens it, moves feed it, a release or cancel ends
  it. → A gesture object of its own; and if a chord+button should open it on
  the canvas, a `DragAction` row names the binding (§25.3). The registry entry
  is the *routing*, never the gesture itself.
- **An act on one of the document's own rows** — this layer's eye, that
  guide's trash. → Neither table. The target is something only the document
  knows, and a registry of every (act, target) pair would be a second copy of
  the panels. It is the panel's own control; give it the same gates the act
  class demands (§25.2's `may_edit` discussion still applies to its handler).
- **A hold** — something that owns both edges of its key or button: space's
  pan, the digit rack (§18.1.8), the pen's eraser end. → `input`'s window
  bindings, which own keyup. Holds are deliberately not rows in either table
  (§25.3 says why for drags; the chord table's reason is in `commands.rs`).
- **Data arriving with the event** — a paste. → The browser's. Ctrl+V is not
  a binding of ours; a chord row would `prevent_default` the clipboard dead
  (§23). Advertise it by hand if the act deserves a shortcut column
  (`Command::shortcut` does this for Import), and refuse to let it rebind
  (`Command::rebindable`).

Do not confuse the chrome's `Command` with the engine's command tiers
(`DocCommand`, `ViewCommand`, `GestureCommand` — §4). A `Command::run` usually
*dispatches* one of those through `state::dispatch`; the registry variant is
the chrome-side name for the whole act, gate included. The §4 rule still binds
whatever you dispatch: an engine method that mutates and answers nothing
should be a command at that seam too.

### 25.2 Adding a command

The variant is the act's identity. Two chords may name it, three surfaces may
carry it, and a rebinding moves the chord while the variant holds still — the
stored table keys on the variant's *name* (`format!("{c:?}")`), so renaming a
variant silently drops any rebinding a user stored for it. Softer stakes than
§19's action variants — a dropped row is a binding for nothing, not an
unloadable file — but the same instinct: treat the name as published once any
build has shipped it.

The checklist, in the order the compiler and the tests will walk you through
it:

1. **Add the variant**, with a doc comment saying what the act means and any
   rule a call site could otherwise forget. Everything else hangs off it.
2. **`run` carries the act's gate.** This is the load-bearing rule of the
   whole registry: the gate is a fact about the *act*, not about whichever
   control happened to reach it. Three classes exist, and a new act should
   say (in a comment) which it is and why:
   - **Document edits ask `may_edit`** — refused while the playhead is moving
     (a commit truncates the withheld timeline, §18.2.4) and while a mode is
     composing (its preview is computed against the committed document,
     `crate::modes`).
   - **Undo and redo `edit_history`** — they *resolve* rather than refuse:
     stop playback, put the composition down, then act. Nothing on screen
     says they are unavailable, so a silent refusal would read as a broken
     keyboard.
   - **View, brush and chrome acts go ungated** — tuning the brush and
     toggling a panel commit nothing. Say so; an ungated arm with no comment
     reads as a missing gate.
3. **`enabled` is presentation only** — what a menu row greys on, read off the
   projection so it states a fact ("nothing to undo"). It is never the gate:
   a caller must not skip `run` because `enabled` said yes, and `run` must
   not assume `enabled` was consulted.
4. **`claims` only for a key with a double life.** The default is `true`:
   even a declined act claims its chord, because the browser's default is
   worse (a refused Ctrl+A must not highlight the page). Override it only
   when the *keystroke itself* belongs to someone else part-time — FinishMode
   claims bare Enter only while a mode is composing and no dialog stands over
   it, because Enter is otherwise the keyboard's activation of whatever has
   focus. If you find yourself writing a second exception, re-read that one
   first; the bar is that high.
5. **List it in `ALL`** — by hand, and nothing will remind you: a variant
   left out compiles clean and is simply unfindable in the palette. Add it to
   `BASIC` only if it is file-family — the resting offer exists for acts with
   no muscle-memory home, not for whatever is newest. Give it `aliases` for
   what other software calls the act (searched, never printed: the alias does
   the finding, the name does the teaching).
6. **A default chord is optional and rare.** Most commands have no row — a
   chord is one way to reach an act, not part of being one. If it earns one,
   add a row to `defaults()`: `Char` for a mnemonic (Z undoes wherever the
   layout puts the Z), `Code` for a spatial pair (`[`/`]` step the brush
   because they are adjacent). Chords are exact about modifiers, and Alt can
   never appear (AltGr arrives as Ctrl+Alt). `default_chords_are_disjoint`
   will fail a collision. Two traps with the bare keys: Escape can be
   rebound *off* but never back on (`capture` spends it on cancelling the
   capture), and anything claimed before the table — space, the digit rack —
   is not yours to row.
7. **Surfaces render the command.** A bar chip or panel-header button is
   `widgets::CommandButton`; a menu row is the rail's `CmdItem`; a control
   whose words are its own but whose key is the registry's (a mode bar's
   Done) appends the advertisement with `commands::advertised`. Never write a
   raw `button` that dispatches the same act — that is how the menu's
   Deselect skipped the gate.
8. **If the act moves state only the frontend sees**, tell the tour: the
   tutor's one reader hangs off `dispatch` (§24.2), so an act that reaches no
   engine — opening the brush editor — owes its deed by hand
   (`tutor::did`).

### 25.3 Adding a drag action

A drag action is routing, not behavior. The behavior is a **gesture object**,
and it comes first: a `Copy` hook shaped like `Nav`, `Tune` and `Paint` —
`begin` on press, `advance` on move, `stop`/`end` on release or cancel, each
answering *was this event mine?* — owning its own in-flight state. One thing
owns one gesture; the pre-`Paint` era, when a gesture's halves were spread
over a component, `AppState` and two free functions that had to agree, is the
counterexample. Share a signal on `AppState` only for what sibling chrome
genuinely reads (`PickState::dragging` exists because the options bar mounts
on *armed but not dragging*), and write down which reader earns it.

Then the table names the press that opens it:

1. **Add the `DragAction` variant and a `defaults()` row.** A `DragChord` is
   an exact `Mods` triple plus a button. **Exact** as the keyboard's chords
   are: Ctrl+Alt+drag is not the Ctrl row with a bystander, it is unbound,
   and an unbound press falls through to painting — the ground state, the
   drag table's equivalent of an unclaimed chord falling to the browser.
   Unlike the keyboard table, Alt is nameable here — a drag types nothing, so
   the AltGr trap has nothing to spring on. `Left` means a **contact**: the
   primary button *or the pen's eraser end* (§18.1.8), so every bound drag
   works whichever way up the stylus is. `Right` is free on the canvas — the
   context menu is refused everywhere but text fields.
2. **Put the gate on the action** — `DragAction::claims`, `Command::run`'s
   lesson restated for presses. PickColor declines over a selection tool
   (Alt is the subtract marquee there, §6.8) and during playback (a sample
   would read the replay, not the painting); PickAndTranslate declines in
   both places too, arriving from the other side — Shift is the *union*
   marquee, and its release commits, which the playhead forbids; TuneBrush
   declines nothing (the brush is view state, and the sliders it shadows are
   not refused mid-playback either). A declined press falls through to the paint path,
   which is usually exactly right — the modifiers then mean whatever the
   paint gesture says they mean. Do **not** encode a gate as position in the
   canvas ladder; the ladder orders *families* (§25.4), and `find` asks
   `claims` before the press ever reaches an arm.
3. **Add the arm** in the canvas's `onpointerdown` match: capture the
   pointer, open the gesture, abandon any paint stroke another pointer had in
   flight, and decide `canvas_active` deliberately — fade the chrome only if
   the gesture's answer is *not* read off a panel. TuneBrush and PickColor
   keep the chrome up, because the Brush and Color panels are where their
   answers land; PickAndTranslate fades it, because its answer is the
   painting itself moving. Then teach `end_interaction` to put the new
   gesture down: it is the one place that knows what a release ends.
4. **Advertise through the table.** Whatever the resting screen shows for
   the binding — a cursor class, a mounted bar, a stood-down overlay — must
   ask `drags::armed` over `AppState::held_mods`, never test a modifier
   directly. `held_mods` is the tracked triple (`input::track_mods`,
   self-correcting off every key event's modifier set), and `armed` answers
   with the same table `find` will consult, so the promise moves with the
   binding and is exactly as exact as the press. This is also the
   discoverability bill (§24's opening argument): a modifier binding is
   discoverable *only* through what appears while it is held — the
   eyedropper's cursor and options bar, the size drag's ring at the press,
   the layer carry's `move` cursor. A new action owes an announcement of the
   same kind, and it owes it through `armed`. The other half of the same
   bill is what the announcement must *displace*: chrome that promises paint
   — the brush circle, and the hover mark folded into the shown document —
   stands down for an act that shadows it (`DragAction::shadows_paint`),
   which is a property of the act rather than a list kept beside each
   cursor. The mark is the sharper case, because an act that reads the
   canvas back would read the hypothesis as paint: the wrong colour for the
   eyedropper, the wrong layer for the carry.

What is deliberately **not** a row, so that nobody re-litigates it per
feature:

- **Navigation.** Space-drag, middle-drag and the two-finger gestures are
  `input::Nav`'s, asked before the table and shared with surfaces that never
  consult it (the transform overlay, the gradient bar). The holds read their
  modifiers *inexactly* on purpose — space+Alt must stay a pan, where an
  exact table would drop it through to paint. `Pan` or `Zoom` could still
  become actions here one day, bound to ordinary chords (a bare right-drag
  pan is a plausible row); it is the holds themselves that are not rows.
- **The marquee's combine modifiers.** Shift and Alt over a selection tool
  modulate the paint gesture (`panels::select::modifier_mode`) rather than
  replacing it — which is precisely why PickColor's and PickAndTranslate's
  claims stand down there. Between them those two rows *are* the marquee's
  Alt and Shift, and that is not a collision to be resolved by finding
  quieter chords: each is the conventional binding for both acts, and which
  one a press means is a question the tool in hand already answers.
- **The plain press.** Painting is not a bound action; it is what an unbound
  press *is*, and the *tool* owns what it means (§6.8). A row for it would be
  a second authority over the question the tool already answers.

There is no user-facing rebinding for drags yet, on the inert-scaffolding
rule: a stored table nothing can edit is scaffolding. The day a surface
exists, copy `commands::Bindings` — overrides over defaults, keyed on the
variant's name — and the two readers (`find`, `armed`) are already the only
places that need to consult it.

### 25.4 The canvas press ladder

The canvas's `onpointerdown` is the one place that sees every binding at
once, and its order is meaning, not history. Each rung answers *was this
press mine* and returns; a press no rung takes is paint:

1. **`Nav::begin`** — a second finger, the middle button, space+contact
   (accelerator choosing zoom over pan). First because the holds outrank the
   chords: taking space here is what keeps space+accelerator a zoom and
   space+Alt a pan, before the drag table could read those modifiers as
   anything else.
2. **`drags::find`** — the drag table (§25.3), gates included. Above the
   playback guard *because* the gates are per-action: tuning survives
   playback, sampling does not, and the ladder must not decide that for
   them.
3. **The playback guard** — nothing commits while the playhead moves.
4. **Contact → paint** — `Paint::begin` with the current tool; the marquee
   modifiers apply inside the gesture, not before it.

The move and release handlers do not re-ask the table: **a drag is what it
was begun as** — each gesture's `advance` answers only if it has one in
flight, and `end_interaction` puts every family down in one place. When a new
gesture joins, it joins all three handlers, not just the press.

The move handler has an order of its own, and one rung of it is load-bearing:
the check for a **composing mode opening under a captured pointer** must sit
above every gesture that holds a document preview, so the mode's arrival
reaches that gesture's `abandon` before another move renews what it is
showing. `Nav` and `Tune` sit above it because neither edits a document;
`Paint` and `PickMove` sit below.

The keyboard's ladder is the same idea one seam over (`input`'s keydown):
the text-entry carve-out first (a field's keystrokes are the field's), then
the holds (space, the digit rack), then `commands::find` against the chord
table — which is why neither table can ever row a key the holds claim.

### 25.5 What holds still

A summary of the identities this chapter leans on, for the reviewer's
checklist:

- **The variant is the act's name.** Stored rebindings key on it (chords
  today, drags when rebinding arrives); rename one and stored rows for it
  become bindings for nothing, dropped on load.
- **The tables have two readers each.** Dispatch: `commands::find`,
  `drags::find`. Advertisement: `Command::shortcut`/`tooltip`/`advertised`;
  `drags::armed`. New code that compares a raw key, button or modifier
  against a hardcoded meaning — outside `input`'s holds and the gestures'
  own internals — is a drift bug that has not happened yet.
- **Gates live on acts.** `Command::run` and `DragAction::claims`, never the
  call site, never ladder position, never `enabled`.
- **Advertisements are as exact as their bindings.** `armed` and `find` read
  one table; a cursor that promises what a press will not do is the bug the
  pair exists to rule out.

### 25.6 The browser-local store

The third registry, and the one added last because it was learned the hard way.
`Store` (`stark-ui/src/storage.rs`) enumerates every record this browser keeps —
ten of them: the four libraries (shapes, presets, gradients, quick brushes), the
⚙ dialog's settings, the chord table, which panels are open, whether the
navigator is up, what the tour has seen, and this client's identity.

One law, the same one: **one authority, and callers hand it typed values rather
than spelling a format.** Here that is enforced by the type system — a type
declares which record it is, and the four functions ask the type:

```rust
impl storage::Record for Prefs      { const STORE: Store = Store::Prefs; }
impl storage::Entry  for ShapeEntry { const STORE: Store = Store::Shapes; }

storage::save(&prefs);                       // -> stark.prefs, and nowhere else
let prefs: Prefs = storage::load()?;         // whole record
let shapes: Vec<ShapeEntry> = storage::load_list()?;   // entry by entry
```

The `Store` was a *parameter* once, which meant the type and the key were two
separate choices at every call site and nothing checked that they agreed:
`load::<Prefs>(Store::Bindings)` compiled and returned garbage. Now there is one
choice. `get`/`set` are **private** for the same reason one layer down: there is
no untyped door, so there is nowhere for another format to come from.

The registry row still carries both facts about a record — its `localStorage`
key and the name a quota warning calls it by — and the impls name a *variant*
rather than restating those strings. Ten impls each spelling their own key would
scatter the answer to "what does this browser keep?" across ten modules, and
nothing would notice two of them colliding.

**Two traits, not one, because seven of the ten records are lists.** `Record` is
the whole of what a key holds (`load`/`save`); `Entry` is one item of a library
(`load_list`/`save_list`). Under a single trait, `load::<StoredPanel>()` would
compile and quietly answer `None` — an array is not an object — leaving a panel
stack that silently forgot itself. A type is one or the other, and the compiler
says which functions it is for.

The difference the traits encode is **what damage costs**. A list is read
element by element, so one entry today's build cannot make sense of is dropped
and the rest survive — which is what a library wants, and what every "a name
this build does not know" case leans on: a binding for a retired command, a
panel this build no longer has, a tally for a deed nothing counts. A whole
record is all-or-nothing, which is what `Prefs` wants: a half-read settings blob
is a worse answer than the defaults. `load_list` also tells `None` from
`Some(vec![])`, and two callers need it — an untouched quick-brush rack is
seeded from the preset library, an emptied one is left empty.

**Adding a record**, in four steps:

1. A `Store` variant, with its key and its human name on the one row.
2. A serde type in the owning module. Give every field `#[serde(default)]` or a
   `Default` on the struct — that, not a version suffix in the key, is how a
   record survives the app version that adds or drops a field. Bytes go through
   `storage::b64`, a content id or key through `storage::hex`. A record that is
   a bare primitive needs a newtype: `impl Record for bool` would make *every*
   boolean in the frontend that record (`navigator::Showing`).
3. The one-line `Record` or `Entry` impl pairing the two, and a line in
   `every_record_claims_one_store` — which fails on the count if you forget,
   and is what catches two types claiming one variant.
4. One writer, called by everything that changes the state — the move
   `layout::set_open`, `navigator::set_open` and `settings::SettingToggle` all
   make. Durability is then structural: a new way to change the thing is
   remembered without its author thinking about storage.

**What is *not* stored here**: anything the document owns. A gradient, a preset
and a brush shape follow this browser; the ramp a fill commits, the color a
stroke carries and the shape's bytes are embedded **by value** in the action
(§8, §22.4), so a document stays self-contained and the library stays personal.
If a record would change what a peer sees, it is not this registry's.

### 25.7 What a dialog owes

A dialog joins no table. It is here because it is the other thing a feature adds
to the chrome — §11 says which of them exist and why each is a dialog rather than
a panel — and because what holds for one holds for all of them by construction
rather than by each author's care, which is the law the registries above are made
of, one seam over.

**A dialog is capped and scrolls itself, whatever it holds.** Every
`.modal-dialog` stops at 80% of the window and scrolls inside that, with its
title and its buttons held to the top and the foot by `position: sticky`, so the
answer to "how do I leave this?" is never below the fold. The cap is on the
class, not in each dialog: the two that had already met the problem (Credits,
the brush editor) had each solved it privately, and the one that grew into it
next — Settings — had not. A dialog gains a row at a time, and the row that
overruns a short screen is never the one whose author was thinking about height.

Three consequences worth knowing before adding one:

- **An inner scroll region needs a reason**, and there is only one: something
  above the region has to hold still while it moves. The brush editor's sections
  scroll because the preview beside them is a fixed column; the timing table
  scrolls because its column headers are what tell Mean from p99 from Max. The
  credits list had one and lost it — it is read straight down, so the dialog's
  own bar serves it. Two bars for one list is the failure this rules out.
- **A dialog is opaque** (`--panel-solid`, the panels' own chrome with the tenth
  of the canvas that shows through it closed up). A panel floats over the
  painting and is worth seeing past; a title band with rows sliding under it has
  to cover them.
- **`.modal-title` and `.modal-actions` are held only as direct children** of the
  dialog. A dialog that wraps either in a header of its own — the brush editor
  does — lays that header out itself and is left alone.

The one deliberate exception is `.be-dialog`, which takes more than the 80%: it
is a working surface rather than something to read and dismiss, and every pixel
it takes goes to the preview or to a list that scrolls inside it.
