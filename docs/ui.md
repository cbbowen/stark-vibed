# The frontend

The Dioxus app and the wgpu surface it wraps, the native wgpui frontend beside
it, and the plan for the crate between them — §11 — and the chrome's registries:
commands, chords, drag bindings, the browser-local store and the shape of a
dialog, which a new UI feature joins and how — §25.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.
> One name per thing: [glossary.md](glossary.md).

## 11. Frontends

`stark-dioxus-frontend` is a Dioxus 0.7 **web** app: the backend runs in WASM
and the painting surface is a dedicated `wgpu::Surface` bound to the page
`<canvas>` via WebGPU, which the engine draws into directly. DOM chrome
surrounds it.

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
  canvas substrate, once for the lighting environment. Neither spelling compiles.
  Pointer events
  become `GestureCommand::Start`/`To`/`End`, with element coordinates mapped via
  `ViewTransform::screen_to_canvas`. `Start` also carries the **input tolerance**
  (§6.2): `devicePixelRatio` and the event's `pointerType` give the device's
  tolerance in CSS px, and the same view transform carries it into canvas px. A
  fourth, `Hold`, is sent by the dwell watcher when the pointer stops moving
  mid-stroke — the drawing assist (§6.9). It is the frontend that has the clock,
  and the moves after it are still plain `To`s.
- **The brush is the frontend's, and the engine holds a projection.** The model's
  `BrushParams` is shaped for what a stroke's record needs — the shared tip knobs
  and *the* effect in force (§6.2, §6.12, §6.13) — but a brush is *edited* across
  that line: every effect stays configured while one is in force, so toggling
  Paint ↔ Erase — or through Liquify — forgets nothing (the hand's color above
  all — an erasing or liquify brush carries no
  pigment of its own), and the stroke-smoothing feel (§6.11) travels with the
  brush though the record must not carry it. `brush_config::BrushConfig` is that
  editing shape — the **durable** half alone, what the tool *is* — held in
  `AppState::brush` beside `AppState::transient`, the **transient** half
  (`brush_config::Transient` — the hand's own state: the size, the flow and
  the painting color, adjusted without changing its mind about the tool;
  §18.1.8 has the split, and the color's own rule — it never arrives with a
  tool — lives at `presets::wear`). The preset
  library stores both halves; the quick-brush rack stores a preset's name
  beside a transient of its own; and `state::update_brush` is the one door down
  — it writes the two signals and dispatches `ViewCommand::SetBrush` with the
  projection (`BrushConfig::params`, which takes the transient) and the hand's
  color beside it, which is
  what a fill lays even mid-erase (`Session::color`). Nothing reads a brush back
  off `ObservableState`: what the engine cannot represent never has to
  round-trip through it. `input::Nav` owns every
  binding that moves the view — the two-finger gesture (§18.1.7), middle-drag and
  space-drag pan, wheel zoom — and every surface over the canvas makes its own
  (the canvas, the transform mode's catcher). Its three entry points are a
  lifecycle, `begin` / `advance` / `release`, and each answers the same question:
  *was this event mine?* So a surface routes its pointers by asking three times
  and never by inspecting buttons or pointer types itself, and what "the pan
  bindings" and "the zoom rate" mean cannot drift between surfaces. Policy stays
  at the call site — the canvas fades the chrome while it navigates and cancels
  the stroke a second finger interrupted, the transform overlay deliberately does
  neither. It reports a fourth thing nobody has to take: fingers that came and
  went without moving the view made a **tap**, and only the canvas spends it
  (§18.1.11).
- **A finger's press is held before it is believed.** `input::Landing` sits in
  front of `Paint` on the canvas and takes every press the paint gesture would
  have taken. A pen's or a mouse's is handed straight on; a finger's is *held*,
  because the same contact is the opening half of a pinch, the start of a stroke
  and the beginning of a hold, and which one it is only becomes known when a
  second finger lands, when this one travels, or when it does neither for long
  enough (§18.1.11). Reports arriving during the wait are kept and replayed into
  the stroke when it opens, so the wait costs latency and never shape.
- **A simple command is a row in a registry, not an arm in a match.** `commands`
  declares every simple act — one the chrome can ask for whole, with no argument
  at the call site — as a variant of `Command` carrying its entire description:
  display name, terse chip word (`Command::word`, the abbreviation a control in
  a narrow column wears — "Rect" for "Rectangle select"), mark, tooltip,
  availability (`Command::enabled`, what a row greys on), whether the act is
  live right now (`Command::active`, drawn two ways from one answer: the pale
  accent on the mark, which a menu row and a palette row draw alike, and a lit
  `CommandButton` — the armed shape tool, Share while a session runs),
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
  nameable act — the visibility menu draws its panel rows from it, a search for
  "panel" lists the whole stack, and any of the six can be given a chord.
- **The rail's first entry is the registry, searchable.** `rail::CommandSearch`
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
  up. It is the flyout that had to be **our own** first: the vendored
  `MenubarMenu`'s trigger light-dismisses when DOM focus leaves it for anything
  but a menu item, and this surface exists to hand focus to a text field — so it
  is the filter picker's own-dropdown arrangement, plus one question that pattern
  never had to ask: `platform::focus_stays_within`, so focus hopping from
  trigger to field on open does not read as dismissal. The menu beside it is that
  same arrangement now, and the vendored menubar left the app with it — which is
  what let the two stylesheets become one, a css_module having hashed the classes
  the palette could until then only restate by hand.
- **The rail's menu is a map of what is on screen, not a list of the panels.**
  It began as the panel stack's own — one row per `PanelId`, each wearing the
  mark its title bar wears, so the column is read at a glance rather than a word
  at a time — and then took in the chrome standing *outside* the stack: the
  navigator's miniature, the quick-brush rack, and Timeline mode. None of those
  has a title bar to close itself from, so for each of them the menu is the only
  way there and back; the timeline had no way at all before its row, being
  reachable by name in the palette and nowhere else. What the menu holds is one
  list — `commands::VisibilityToggle::ALL`, whose entries are `Panel(PanelId)`,
  `Navigator`, `QuickBrushes` and `Timeline` — and the menu is one loop over it.
  The three arrived one at a time before that, each as a row whose focus index was
  counted off `PanelId::ALL.len()` by hand: bookkeeping the loop beside it was
  already doing, restated where nothing would catch it going wrong. The enum is
  deliberately thin — an entry knows only *which* `Command` it is, and
  the word, mark, lit state and greyed state are the registry's — so it is a view of
  the registry rather than a second one, and nothing is reachable there that a
  search for its name would miss. And what the menu toggles, the browser keeps:
  one record over that same list, so every row of the map is remembered and a new
  row cannot be added without saying so (`crate::visibility`, §25.6).
  **A row leaves the menu standing.** Showing the Layers panel and hiding the
  navigator is one errand rather than two, and a map that closed on the first
  answer sent the artist back to the trigger to give the second — while the mark
  that has just changed under the pointer is the very thing they came to read.
  Nobody chose that either: it is what the vendored `MenubarItem` does on its way
  out of `on_select` and nothing outside that crate can decline, and it is the
  second of the two reasons the rail draws its own flyouts. Escape still puts the
  menu down, from the ladder rather than from the menu: the open flag is a
  `widgets::PopoutId` (§25.7), which is what keeps this menu's Escape from being a
  *second* actor on a keystroke the window is already hearing. The arrows and
  Enter stay the palette's, where a text field withholds that window binding —
  and every act in the map is in the palette by name.
- **A chord names its key the way the binding means it.** A chord names the
  accelerator tier (Ctrl or Command, `input::accel`), the Shift bit, and a key
  that is either the *character* it types — a mnemonic follows the layout,
  because Z undoes wherever the layout puts the Z — or the *position* it sits
  at — a spatial pair is about adjacency, and `[`/`]` step the brush precisely
  because they are side by side (`slots::of_code`'s argument, §18.1.8). Chords
  are exact: Ctrl+Shift+Z is its own row rather than Ctrl+Z plus a bystander,
  and Alt+H is not the bare `h` row with a passenger. **Ctrl+Alt is bindable
  and never shipped.** On Windows AltGr *is* Ctrl+Alt — the OS synthesizes the
  pair for the right-hand Alt and for a deliberate Ctrl+Alt alike — so a
  shipped row there would answer a German layout's `@` by claiming the
  keystroke and `prevent_default`ing the character dead, a bug invisible to
  everyone whose layout has no AltGr (the plain US one has none;
  US-International has a full set). A row the *user* captured is a keystroke
  they chose on their own keyboard, so the rule binds `defaults()` and not the
  type — which also leaves Command+Option reachable on a Mac, where it is
  idiomatic and no AltGr is involved. An Alt chord is always **spatial**,
  captured or shipped: Option+G types `©` on a Mac and AltGr+A types `ą` on a
  Polish layout, so a key held through Alt has no character of its own left to
  be named by. **Alt on its own is shippable**, and the table spends it in one
  place — the eyedropper's three reaches on Alt+Q / Alt+A / Alt+Z (§18.0.2),
  under the very modifier that raises the bar they change, so the chord is
  pressed by a hand already holding the tool. The keydown handler asks the
  table once and claims a matched chord wholly (`prevent_default`) whether or
  not its act was accepted — a declined Ctrl+A must not answer with the
  browser highlighting the page.
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
  keydown, refusing what could never fire (space and the bare digit holds, the
  paste's Ctrl+V), taking Escape as the way out and Backspace as the eraser:
  an unbind is the same gesture as clearing a field, at the cost
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
  stack too (`layout::open_panel`, the one door), or the command would light a
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
  visit are the ones the artist actually reached for
  (`visibility::stored_hidden` — one record with everything else the visibility
  menu toggles, §25.6). The two halves are one decision — a set of panels
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
    came back as a bare title bar would be a menu entry that lights itself and shows
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
  beats it regardless of DOM order. The whole ladder, since one rung is only ever
  right relative to the others:

  | Rung | What is on it |
  |---|---|
  | 20 | the chrome that stands in place: the panel stack, the bottom bars, the quick-brush rack and the navigator (`.left-chrome`) |
  | 25 | Timeline mode's bar, which spans the window rather than hugging its contents |
  | 26 | what flies out of that chrome and must cover it: a panel's pop-out, the tour's card |
  | 30 | an open menu — the command rail and its flyouts |
  | 100 | a dialog and its backdrop |
  | 101 | a tour card pointing *into* a dialog, the one card the backdrop must not cover |

  Two of those are worth the sentence they cost. **A menu covers everything it
  overlaps except a dialog**, because it is the surface the artist asked for a
  moment ago and closes the moment they are done with it — and the rung is on
  `.command-rail` rather than on the flyouts themselves, which cannot take it: a
  dropdown is absolutely positioned inside the rail, whose own `backdrop-filter`
  makes a stacking context, so its `z-index: 1000` is spent clearing the rail's
  buttons. Left level with the chrome, the rail lost to everything mounted after
  it in `main.rs` — the quick-brush rack covering the menu that puts the rack
  away. **Anything sharing a rung is ordered by the DOM**,
  so a tie there is a statement about the order in `main.rs` and is commented as
  one where it matters (the pop-out and the card at 26).
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
  The screen keeps one across frames; anything rendered beside it brings its own
  through an `Offscreen` slot, so an off-screen render never resizes the screen's
  attachments out from under it. Whether a slot outlives its call is the
  *caller's* to state, because only the caller knows whether the render repeats:
  the navigator holds one for the app's life, while a file export uses a local
  one so a 4× export of a large frame does not park its several-hundred-megabyte
  pair for the session. View settings stay per-pipeline behind a process-wide
  generation stamp, so a swapped substrate or light — or a whole rebuilt pipeline, as
  a color-space change makes — reaches every consumer by being *noticed* rather
  than by a notification a new consumer could be left out of.
- **A preview engine is a sibling, not a second boot.** The brush editor's test
  canvas and the preset thumbnails each want an isolated document that renders
  *exactly* as the main canvas would — which is an argument for sharing the
  machinery, not merely an economy. `Engine::new_sharing` builds one around a
  fresh document, sharing everything expensive and un-disagreeable: the compiled
  pipelines (immutable), the content-addressed brush assets and the substrate /
  environment byte-and-build caches (a `Registry`'s store is `Arc`-shared while
  each sibling keeps its own *current* id), and the tile pool (an allocator).
  What an engine can set stays its own. So the editor's preview
  (`Renderer::shared`) opens on the canvas's substrate under its lighting with
  nothing fetched and nothing decoded, and the thumbnails' engine
  (`Renderer::shared_engine`, `thumbs.rs`) deliberately pins the opposite look —
  flat substrate, neutral light — so a thumbnail is the *brush's* identity card and
  its cache key is the brush snapshot alone. The snapshot *as the picture paints
  it*, which is the same thing minus the painting color: the stroke is laid in
  the thumbnail's own gray whatever RGB it is handed, so keying on the raw
  snapshot would file one picture under a fresh name for every color a preset
  happened to be saved in — a preset carries the color the hand held when it
  was written, and `presets::wear` keeps the live one over it (§18.1.8), so one
  tool saved twice in two colors would be rendered twice for one row's worth of
  picture. The brush's own opacity does stay in the key: the stroke really is
  laid with it (§6.1) — and so do the size and flow, since a slot tuned off its
  preset's is a different stroke. Each row's picture is two half-canvas fills
  (the substrate is all paint, so smearing and lifting read), one replayed
  stroke and one small `Engine::export_view` readback on that one kept engine,
  generated in the background and cached per session. The key being the brush
  is what lets the cache have two viewers for the price of one: the preset
  library's rows and the quick-brush rack the number keys draw (§18.1.8) show
  the same picture of the same brush — a slot is its preset at a size and flow
  (`slots::resolve`), and one still at the preset's own is never rendered
  twice. The generator therefore belongs to the app **root**, not to
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
  One section of it is not preferences at all but a **table** — the canvas drags,
  with a segmented run of per-app presets over them (§25.8, §25.9) — because
  "where do I change that?" has to have one answer whether the thing being changed
  is a switch or a binding. It wears the same row shape as everything above it, so
  the dialog is still read down one column of labels.
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
  `stark-dioxus-frontend/public/`, which the CLI copies to the **site root**
  unhashed — unlike `assets/`, whose every file is renamed by content hash. That
  distinction is the whole design of `public/sw.js`: the navigation response
  names one build's hashed wasm, so it is fetched **network-first** and only
  falls back to cache; everything else same-origin is content-addressed and so
  can never be stale, and is served **cache-first** while a background fetch
  refreshes it. Cross-origin, non-`GET` and range requests are passed straight
  through, so the collaboration transport (§12.4) and any partial fetch are
  untouched.
  - This matters more here than for a typical app because the heavy assets are
    deliberately *not* in the wasm binary: the brush stamps (§6.6), the substrate
    height maps (§6.4) and the environment HDR (§6.3) are all fetched after boot
    by `builtins::import_all` / `substrates::open_default`. Without the worker that
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
    `stark-dioxus-frontend/tools/make-icons.py`, run by hand; the PNGs are
    checked in. Not a build step — a logo is not a build product, and nothing
    keys off its bytes the way `stark-assetid` keys off an asset's (§19).
- **The painting lives in the tab, so the tab asks before it goes.** There is no
  autosave and no server here: a reload, a closed tab, a followed link and the back
  button are all the end of anything neither Save nor Export has taken out, and
  `beforeunload` is the one place a page may object to that. `files::guard_unload`
  takes it. Bound in the root's body rather than at the end of the startup task,
  unlike the launch queue above — it is a predicate over the signals, so it wants no
  engine and cannot be left unbound by a start that fails before there is one. The
  prompt itself is entirely the browser's: its wording, its buttons, and whether it
  appears at all, which is why `platform::on_before_unload` takes a predicate and
  returns nothing.
  - **The question needs both sides of §2 to answer it, and neither alone.** The
    engine says whether the committed document has moved since it *arrived*
    (`ObservableState::edited`, against a baseline `Engine::doc_origin` takes in
    `reset_document` and takes again at the end of a load, after the replay that
    moved the counter). That is the half that would otherwise be a list of
    document-replacing call sites kept by hand in the frontend — a new document, an
    open, a launched file, a dropped one, a collaboration join — with the next one
    silently missing from it. The frontend says which revision it last *wrote to a
    file* (`AppState::written_revision`), which is not a thing a document has any way
    to notice happening to it. "Unsaved" is the conjunction, and `files::unsaved` is
    where the two meet.
  - **Committed only.** A stroke in flight and an unlogged drag preview both move
    what is on screen without moving `doc_revision`, which is the same narrowness the
    navigator's miniature is keyed on — and a gesture is over long before a hand
    reaches the tab strip.
  - **An Export counts, though an export is not a save.** A picture is not the
    document and nothing can be recovered from it (§15.6), so an artist who exports
    and closes still loses the editable painting; keeping those two words apart is
    the menu's job and stays the menu's job. What the guard asks is narrower — is
    any of this work *gone* — and that is the difference between an alarm worth
    reading and one raised over a copy the artist has just watched download.

Because the engine is frontend-agnostic, this layer stays thin. (An earlier
interim cut ran on Dioxus *desktop* and bridged the canvas by reading the frame
back to a PNG data URL — correct but laggy; the WebGPU surface replaced it,
touching only `stark-dioxus-frontend`.) Run with
`dx serve --web -p stark-dioxus-frontend` in a WebGPU browser.

### 11.1 The second frontend (wgpui, native)

`stark-wgpui-frontend` is a native window over the same engine: [wgpui][wgpui] —
a community fork of GPUI that renders through wgpu and winit — with the engine's
own texture in its element tree as a `WgpuSurface`. Run it with
`cargo run -p stark-wgpui-frontend`.

**It exists to be a second consumer.** "Because the engine is frontend-agnostic"
was a claim about a tree with one frontend in it, and a claim of that shape is
only tested by something that shares nothing with the first: no DOM, no browser,
no `<canvas>` — and no ownership of the device. What it carries is a canvas and
one brush (Hard Round, §6.2), deliberately: chrome is not what a second frontend
has to prove.

It found something on the first day. `GpuContext` carried a `wgpu::Instance` and
a `wgpu::Adapter` that no engine code ever read; wgpui's `WgpuSurfaceHandle`
hands out a device and a queue and nothing else. The two moved to the side that
uses them — the web frontend's own `Renderer`, which binds three `<canvas>`
elements over one device and needs the instance to make each surface and the
adapter to ask what it can do. `GpuContext::from_parts` takes a device and a
queue now, which is what an engine that is *given* its wgpu resources should
have taken all along. Holding a stale adapter — one that did not produce this
device — would have been worse than holding none.

Three differences from §11 above are the interesting ones:

- **The frontend does not own the device, and states what it must be anyway.**
  wgpui creates it and every element in the window draws with it, so the
  descriptor belongs to neither consumer alone — and upstream wrote it as
  `wgpu::Limits::default()`, four storage textures per shader stage where the
  Mixbox stamp loop writes six (§6.7). A whole colour space was therefore
  unreachable here, and no cargo feature could reach it: limits are settled when
  the device is created. The vendored patch threads a `DeviceDescriptor` down
  from `Application::new`, and `main::device_descriptor` starts from
  `Limits::default()` — what wgpui's own renderer was written against — raising
  only the fields `GpuContext::minimum_required_limits` asks for. That function is
  itself `#[cfg]`-dependent, so the ask tracks the build with no second copy of
  the number to drift. What that buys today is a device that *can*: this frontend
  opens the Oklab document `Engine::new` gives it and has no picker to ask for
  another, so the space is unblocked rather than reachable.
- **The paint happens inside `Render::render`**, not on a thread or a timer. The
  surface element resizes its textures during *prepaint* — after the view has
  rendered — so a resize is first visible one frame later, and a window resize
  schedules no further frame; the view therefore asks for an animation frame
  every time and lets a dirty flag decide whether the engine renders at all. The
  swap is `swap_buffers` rather than `present`: this is already inside the frame
  wgpui is building, so the swap is what that frame composites.
- **Screen space is device px.** The web canvas sizes its drawing buffer in CSS
  px, so its pointer coordinates need no conversion; here the surface is in
  device px and every `Pixels` the layout speaks in is logical, so the scale
  factor is the whole of the mapping — and the input tolerance and the smoothing
  rope, both screen-denominated (§6.2, §6.11), are quoted in device px too.

wgpui is **vendored** (`vendor/wgpui`) for three patches. The first is one line:
upstream 0.3.4 calls `flume::bounded` in `Executor::spawn_realtime` but declares
`flume` only for macOS, Linux and FreeBSD, so the published crate does not
compile on Windows at all. The second is the `DeviceDescriptor` above. The third
makes `WindowBounds` reach the platform whole, so a window can be reopened where
it was (§11.2, N1). See `vendor/wgpui/VENDORING.md`, which also records what the
second one *removed* — `Application::headless`, whose flag the new signature
displaced and which upstream had never honoured.

[wgpui]: https://github.com/muktidaya/wgpui

### 11.2 Parity, and the crate between the two frontends

§11.1 is a canvas and one brush. This is the plan from there to a native app an
artist could work in — and, because the two frontends will otherwise grow two
copies of every rule, the crate that stops them.

Status for each stage is in [§13](roadmap.md); the design is here.

#### What parity means, and four things it does not

Parity is of **acts**, not of appearance. The native app should be able to do what
the web one can do; it should not look like it.

- **Styling is not ported.** The web chrome's stylesheet is a web artefact. Native
  should read native, and the visual half of §25.7 and §25.9 — what a dialog and a
  run of buttons owe — carries over as *rules* (a dialog owes a way out; a run of
  buttons owes one visual weight per role) and not as CSS.
- **No minimal-UI mode.** The web app spends real design on fitting words into a
  narrow column — `Command::word`'s chip abbreviations, the panel fade during a
  stroke. Native gets icons plus hover tooltips, which is the same information in
  less space, so the abbreviation column is carried for the *other* frontend's
  benefit rather than used here.
- **No tour (§24).** Someone installing a native build has already met the web
  app. `tutor.rs` stays a web-only reader hung off that frontend's own `dispatch`,
  and the native seam does not grow a hook for it.
- **No browser-shaped affordances.** A session ticket rides the URL fragment
  because a link is how a browser is handed anything (§12); native exchanges the
  same string by paste. Likewise `on_before_unload`, the service worker and the
  installability half of §11 have no native counterpart and want none.

#### The seam, and the fact that it is already drawn

The rule that decides where a line of the web frontend goes is mechanical, and it
is the shape §2 already uses for model-versus-engine:

> If it names a `dioxus::` type or holds a `Signal`, it is **chrome** and stays in
> its frontend. If it is arithmetic over `ObservableState`, `BrushParams`,
> `ViewTransform` or a pointer report, it is the **frontend's model** and belongs
> below both of them.

The evidence that this is a real line and not a hopeful one is that the web
frontend has already drawn it twice, for its own reasons. `gesture.rs` opens with
"Nothing here knows about signals, dioxus, or the browser" and says it is a file
of its own because *it is the part that could be tested* — 18 of the crate's tests
are in it. `panels/layer_tree.rs` says the same thing in the same words for the
Layers panel's arithmetic, and holds ten more. Both splits are this crate
boundary, drawn one file early. `panels/reorder.rs` is a third, drawn between two
panels rather than between two frontends.

So the test for "does this belong below" has a cheap proxy: **code with tests has
already moved.**

#### The crate: `stark-chrome`

`frontend → chrome → engine → model`, and it must compile to wasm, because the web
app is one of its two consumers. Its invariant is `stark-net`'s, one level up:
**it never names a `dioxus::` or `wgpui::` type.** That is checkable by grep and
worth a test that greps.

Named for the docs' own word for the UI around the canvas (§11 passim, and the
glossary's `surface` row). `stark-ui` came free two commits ago and is
deliberately not reused: it meant *a* frontend until then, and a name that means
two things one week apart is exactly what the glossary rule exists to stop.

**Tier A — moves as it stands** (no `dioxus` in it today):

| module | lines | what it is |
|---|---|---|
| `brush_config.rs` | 594 | the durable/transient brush and `params()`, the one projection down to `BrushParams` |
| `gesture.rs` | 868 | the transform algebra (§16.6, §16.8, §16.9) + 18 tests |
| `panels/layer_tree.rs` | 549 | what the Layers panel draws and what a drop means (§14.6, §14.8) + 10 tests |
| `library.rs` | 180 | the thumbnail cache both asset libraries share (§6.4, §6.6) |
| `panels/reorder.rs` | 465 | moving a row of a list by dragging it. Tier B on paper; it turned out to be **one function** — `claimed`, six lines over a `Signal` — with 444 pure lines behind it |
| `identity.rs` | 95 | this client's durable actor key and boot counter (§17) — no toolkit in it, but it reads through `storage`, so it lands with **N1** rather than N0 |
| `panels.rs`, `layout::PanelId` | — | the register vocabulary only: the enums, not the frames |

**Tier B — moves after one decoupling each**, named:

| module | lines | what has to give |
|---|---|---|
| `storage.rs` | 633 | six `platform::` calls become a `Backend` trait (below) |
| `commands.rs` | 2462 | `Command`'s ~35 variants and every descriptive method (`name`, `word`, `aliases`, `icon`, `hint`, `tooltip`, `shortcut`, `rebindable`, `enabled`) plus `Chord`/`Bindings`/`search` go down; `active` and `run` take `AppState` and stay |
| `drags.rs` | 1343 | everything but `Mods::of(dioxus::html::Modifiers)`, which becomes a constructor each frontend feeds |
| `icons.rs` | 607 | the `include_str!` table goes down; the `Element`-returning helper stays |
| `presets.rs`, `slots.rs`, `prefs.rs`, `visibility.rs`, `modes.rs` | 3133 | each is a record plus a policy plus a signal-shaped shell; the shell stays |
| `input/` + `input.rs` | 3017 | the thresholds and the decisions go down — `TOUCH_SLOP`, `nav::MIN_SPAN`, the tune commit distance, the tolerance and rope maps, `is_contact`/`is_eraser`, the tap/pinch/hold discrimination. The five `Copy` hook carriers stay: they *are* signals |
| `panels/brush.rs`'s bounds | — | `MIN_RADIUS`, `MAX_RADIUS`, `MAX_FLOW` sit in a markup module and are read by `input::Tune` and `BrushConfig::max_flow`. They are brush vocabulary and belong beside `brush_config` (their doc comments still say `BrushParams::radius`, renamed to `size` — the move is when that gets fixed) |

**Tier C — stays in each frontend, twice**: everything under `panels/` that is
markup, `layout.rs`, `widgets.rs`, `rail.rs`, `overlays.rs`, `navigator.rs`,
`brush_editor.rs`, `settings.rs`, `canvas.rs`, `main.rs`, each `render.rs`, each
`platform.rs`, and `tutor.rs` by decision.

#### The two abstractions this needs, and no more

Both already have a shape in the tree, which is the argument that they are the
right two.

**`PointerReport`** — what `input::sample` reads off a `dioxus::Event<PointerData>`
and what `canvas::sample_at` reads off a `wgpui::MouseMoveEvent`: position,
pressure, tilt, time, pointer kind, buttons, modifiers, and the coalesced list
behind it. `platform::Coalesced` is already this struct less the modifiers.

One rule keeps it honest: **a report is in the units of its own surface's
viewport.** The web canvas sizes its drawing buffer in CSS px and the native one
in device px (§11.1), so the scale factor is applied at the edge, by the frontend,
and nothing below ever asks which frontend it is in.

**`storage::Backend`** — `get`/`set`/`remove` over text and `get_many`/`put`/
`delete` over blobs, the blob half async. Exactly the six `platform::` calls
`storage.rs` funnels to today, so this is a six-method trait with one impl per
frontend and no design left to do.

#### What a native platform layer owes

`platform.rs` is 1778 lines of web answers to a shorter list of questions. The
native ones:

| capability | web today | native |
|---|---|---|
| text store | `localStorage`, ~5 MB/origin | one JSON file in the config dir |
| blob store | IndexedDB by content id | files in a cache dir, named by id |
| open / save / export | `<input type=file>`, `<a download>` | wgpui's path dialogs |
| clipboard | `navigator.clipboard`, paste event | wgpui (arboard) |
| image decode + normalize | the browser decodes anything it can show | the `image` crate — narrower, and the difference should be *stated* rather than discovered on an unsupported file |
| scale factor | `devicePixelRatio` | `Window::scale_factor` |
| monotonic clock | `performance.now()` | `quanta` (already, §11.1) |
| coalesced pointer reports | `getCoalescedEvents` | winit delivers one per event; the list is length 1 and the fitter is unaffected |
| session ticket in | URL fragment | paste |
| file-launch / drop | `launchQueue`, paste event | winit file-drop, argv |

#### Stages

Each names what lands, what moves down with it, and one thing you can then *do* —
the exit criterion is an act, not a diff.

- **N0 — the crate, empty of opinion.** Create `stark-chrome`; move the five pure
  modules into it. **No `pub use` shims**: a shim would give every moved type two
  public paths, which is the one thing CLAUDE.md's module rule forbids, so the call
  sites move with the code. *Exit:* the web app is untouched and its tests run from
  the new crate.

  What it cost beyond the moves, which is the interesting half: `gesture` became
  `transform`, because `gesture` next to five *input* gestures that are not it
  would be read as those; `reorder`'s six-line `claimed` stayed behind and its 444
  pure lines came down; `pub(super)` widened to `pub` where `super` used to mean
  `panels`; and `MIN_RADIUS`/`MAX_RADIUS`/`MAX_FLOW` came down out of the Brush
  *panel* because `BrushConfig::max_flow` reads one of them — the move was forced by
  the compiler rather than argued for, which is the best kind. `Thumbs` grew a
  `Default`: `new_without_default` does not fire on a binary's `pub` items and does
  on a library's, so becoming a library is what asked.
- **N1 — persistence.** `storage::Backend`; the native impl over a config dir and
  a cache dir; `identity` and `prefs` move. *Exit:* the native app comes back with
  the window it was closed at.

  `visibility` did **not** come: its record is keyed by `VisibilityToggle`, which
  is the command registry's, and its rows hold `PanelId`s. A record cannot move
  before its key type has, so it goes with N3. The same rule brought
  `ChromeHiding` down early — `Prefs` has one as a *field*, so it had to.

  Three other things N1 turned up. `identity` could not name `SecretKey` without
  putting iroh under a crate whose other consumer has no collaboration, so the
  shared half keeps 32 bytes and takes the minting as a closure. `Prefs`'s
  `capture`/`apply_view`/`apply_engine` became free functions, because an `impl`
  on a type from another crate is exactly the orphan rule reporting the boundary
  (CLAUDE.md) — and each of the three reads or writes signals, so none was ever
  the record's business. And the registry's completeness test could no longer live
  beside the format: most record *types* are a frontend's, so
  `stark-dioxus-frontend`'s `records` owns it now, with the rows it does not keep
  listed by name.

  The window record was also the first thing to need a **third** vendored patch.
  wgpui 0.3.4 dropped the origin when it built the winit attributes, *and* applied
  the maximized state afterwards with `zoom()`, which is a toggle used as a
  setter. Either fault alone hides the other — add `with_maximized` and the toggle
  becomes correct in reverse — so the two moved together, and the placement is
  honoured whole now (`vendor/wgpui/VENDORING.md`, patch 3).
- **N2 — the brush in hand.** `brush_config` in use natively; a brush panel with
  size, flow, hardness, opacity and the four effects; the preset table and the
  library arithmetic move down. *Exit:* paint with any shipped preset, tuned,
  instead of one hard-coded `BrushParams`. **Done.**

  The headline was the duplication: `input` is now a shared module and the native
  frontend's copy of `ROPE_MAX_SCREEN_PX` and its quadratic are gone. The table
  moved with two things it names — `BuiltinShapes`, so a frontend can hand in the
  stamps it has resolved, and `slots::{COUNT, ERASER}`, because a preset declares
  the digit it ships on and the record cannot outrun its vocabulary.

  What the *panel* found is worth more than the panel. wgpui ships no widgets, so
  the first cut hand-derived every offset from the Tailwind-shaped spacing the tree
  asks for — and was wrong: a press where the arithmetic said "Airbrush" selected
  Hard Eraser, because the guessed row pitch was 26 px where Taffy had laid out 39.
  Two descriptions of one layout is the drift this whole stage exists to delete, one
  scale down. So the panel **measures**: each control carries a `canvas` element
  whose prepaint writes its laid-out bounds into a shared list, and the hit test
  reads that. No geometry constant survives outside the tests.

  A colour picker did not land. The transient's third knob is there and the engine
  takes it; what is missing is a control, and a colour well is its own design
  (§25.7's pop-out) rather than a fifth slider.
- **N3 — the two registries.** `Command`'s descriptive half, `Chord`, `Bindings`,
  `search`; `drags` and `Mods`. A native dispatcher over wgpui actions and a
  native `armed`. *Exit:* Ctrl+Z undoes, a chord rebound in one frontend is
  honoured by the other's table, and a modifier-drag tunes the brush.
- **N4 — layers.** `layer_tree` + `reorder` drive a native list. *Exit:* add,
  remove, reorder, group, clip, set opacity and blend, all from the native app.
- **N5 — documents on disk.** `files` splits; save, open and export through the
  native dialogs. *Exit:* a `.stark` file round-trips between the two frontends,
  history intact.
- **N6 — selection, fill, transform.** Tool arming, the marquee, the fill parcel,
  and `gesture` (already down since N0) behind a native transform overlay.
  *Exit:* select, fill, and commit a transform.
- **N7 — the asset libraries.** Shapes and substrates over `library` and the blob
  backend; native decode. *Exit:* import a brush shape and a canvas substrate.
- **N8 — the long tail.** Guides (§20), gradients (§22), filters (§21), frames and
  export (§15), the navigator, timeline mode. One panel at a time; each is a Tier
  B move plus native markup.
- **N9 — collaboration.** `collab`'s two pumps move down; the ticket is pasted
  rather than linked. *Exit:* the two frontends paint on one document.

N0–N3 are the ones with leverage: after them every later stage is markup over
rules that already exist and are already tested. N8 is the only stage that is
mostly typing.

#### What to expect this to find

§11.1's `GpuContext` was not a lucky catch — it is what a second consumer is
*for*, and the drift has already started. `stark-wgpui-frontend`'s `canvas.rs`
carries its own `ROPE_MAX_SCREEN_PX = 160.0` and its own copy of the quadratic
smoothing map, because `input::rope_in` was not reachable; the same brush at the
same smoothing is therefore towed by two constants that nothing holds together.
That is one file old and already the exact failure §25 was written to prevent —
"one authority, so what the keyboard answers and what a row claims cannot drift
apart" — and it is the first thing N2 should delete.

The other candidates already visible: the brush bounds sitting in a markup module
(above); `render::PeerInfo`, a chrome-facing projection of a peer that is
deliberately not the engine's `Peer` and that both frontends will want; and every
threshold in `input/` that two gestures read and one file owns. Each is the same
shape as the adapter — parked on the nearest wall rather than the right one, and
invisible while there was only one wall.

## 25. Commands and drag bindings

The chrome reaches every act and every bound gesture through one of two tables.
The **command registry** (`stark-dioxus-frontend/src/commands.rs`) holds the
simple acts — undo, deselect, open the export dialog — each a variant of
`Command` carrying its whole description, with the keyboard as one rebindable
column. The **drag table** (`stark-dioxus-frontend/src/drags.rs`) holds the
canvas presses that open something other than painting — the brush-tuning drag
(§18.1.9), the eyedropper (§18.0.2), the layer carry (§16.11) — each a row
binding an exact chord+button to a `DragAction`. A third registry sits one layer
down and answers a different question — `Store` (§25.6), which enumerates every
record the frontend keeps in this *browser*: the libraries, the settings, the
two tables above. §11 above tells the first two designs' history; this chapter
is the working guide, written for the day a feature is added: which table the
feature belongs to, if any, and the steps that keep them true. §25.7 and §25.9
cover the two things that join no registry and still have rules — the dialog,
and a run of buttons — and §25.8 the half of the drag table that belongs to the
user rather than to us.

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
- **A touch gesture** — a two-finger tap, a finger held still on the glass. →
  Neither table, for a reason neither table can fix: it has no chord and no
  button. What names it is a count of fingers and a length of time, and a row
  keyed on `(mods, button)` cannot hold either (§18.1.11). It lives in `input`
  with the rest of the pointer routing — but it *spends* itself through the
  registry (`Command::Undo`), so the gate, the tour's reading and the menu's
  state come free rather than being restated for the hand that has no keyboard.
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
5. **List it in `ALL`** — by hand, and the build will remind you:
   `tests::all_lists_every_command` counts the enum's variants, so a variant
   left out fails the suite rather than being quietly unfindable in the
   palette. (It used to be neither; that test is what changed it.) If the act
   shows or hides a piece of the window, give it a row in `VisibilityToggle`
   too: that enum is the whole content of the rail's visibility menu, and a
   toggle missing from it is an act with no home in the chrome at all. Add it to
   `BASIC` only if it is file-family — the resting offer exists for acts with
   no muscle-memory home, not for whatever is newest. Give it `aliases` for
   what other software calls the act (searched, never printed: the alias does
   the finding, the name does the teaching).
6. **A default chord is optional and rare.** Most commands have no row — a
   chord is one way to reach an act, not part of being one. If it earns one,
   add a row to `defaults()`: `Char` for a mnemonic (Z undoes wherever the
   layout puts the Z), `Code` for a spatial pair (`[`/`]` step the brush
   because they are adjacent). Chords are exact about all three modifiers.
   **Never ship a Ctrl+Alt row** — that pair is AltGr on any layout that has
   one, and `no_default_chord_is_ctrl_alt` fails the build if you do; a user
   may still bind it, because they chose it. **An Alt row is `Code` whatever
   it means**: under Alt a key does not type its own character, so `capture`
   names every Alt chord by position and a shipped one that named a character
   would disagree with a rebinding of the same key. Alt is worth spending
   where it already means something — the eyedropper's three reaches are Alt
   rows precisely because Alt is what raises the bar they change (§18.0.2).
   `default_chords_are_disjoint` will fail a collision. Two traps with the
   bare keys: Escape can be rebound *off* but never back on (`capture` spends it on cancelling the capture),
   and anything claimed before the table — space, the digit rack — is not
   yours to row.
7. **Surfaces render the command.** A bar chip or panel-header button is
   `widgets::CommandButton`; a menu row is the rail's `CmdItem`; a control
   whose words are its own but whose key is the registry's (a mode bar's
   Done) appends the advertisement with `commands::advertised`. Never write a
   raw `button` that dispatches the same act — that is how the menu's
   Deselect skipped the gate. The two things a surface is most tempted to
   keep for itself are both the registry's: an **abbreviation**, which is
   `word` (a 280px panel column cannot wear "Rectangle select", and the
   answer to that is a shorter spelling of the same name, never a different
   word — §25.9); and whether the button is **lit**, which is `active`, so a
   chip and the chord that lit it cannot disagree.
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
   `Mods` is the same modifier triple the keyboard's chords carry, and Alt is
   as nameable here as it is there — more so, in fact: a drag types nothing,
   so Ctrl+Alt+drag springs no AltGr trap and may even be a shipped row.
   `Left` means a **contact**: the
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

A new action is **rebindable the day it exists**, and costs nothing to make
so: it joins `DragAction::ALL`, owes a `name`, a `word` and a `hint` for the
settings row that appears for it automatically, and gets a row in whichever
presets have an answer for it (§25.8). The two readers already consult the
user's table rather than the shipped one, so there is no third place to
teach.

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

- **The variant is the act's name.** Stored rebindings key on it, chords and
  drags alike; rename one and stored rows for it become bindings for nothing,
  dropped on load.
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
`Store` (`stark-dioxus-frontend/src/storage.rs`) enumerates every record this
browser keeps — eleven of them: the five libraries (brush shapes, canvas
substrates, presets, gradients, quick brushes), the ⚙ dialog's settings, the
chord table, the drag table, what is on screen, what the tour has seen, and this
client's identity.

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
rather than restating those strings. Eleven impls each spelling their own key
would scatter the answer to "what does this browser keep?" across eleven modules,
and nothing would notice two of them colliding.

**Two traits, not one, because nine of the eleven records are lists.** `Record` is
the whole of what a key holds (`load`/`save`); `Entry` is one item of a library
(`load_list`/`save_list`). Under a single trait, `load::<StoredPanel>()` would
compile and quietly answer `None` — an array is not an object — leaving a panel
stack that silently forgot itself. A type is one or the other, and the compiler
says which functions it is for.

**One record per *question*, not per feature — `Store::Visible` is what that
costs when it is got wrong.** What is on screen used to be two records: the panel
stack kept `stark.panels`, the navigator kept `stark.navigator`, each with its
own type, its own reader and its own writer. Defensible while the navigator was
the only thing outside the stack. Then the quick-brush rack and Timeline mode
joined the visibility menu (§25.5) and neither joined a store — a row could be
added to the map of what is on screen with nothing anywhere asking where its bit
was kept, and two of the four were simply forgotten.

They are one record now, keyed by `VisibilityToggle`, and the fix that matters is
not the tidying: `visibility::persist` matches on `VisibilityToggle::ALL`
**exhaustively**, so a tenth entry in that menu does not compile until it says
whether it is showing. Durability stops being a line an author has to remember
and becomes a branch the compiler asks for — the same move `Store::named` makes
for keys one layer up, and the general shape of "rule out a class rather than
enumerate its instances".

```rust
impl storage::Entry for StoredVisible { const STORE: Store = Store::Visible; }
// [{"what":{"Panel":"Layers"},"collapsed":true},{"what":"Navigator"}]
```

A row is a thing that is *showing*, and absence answers for everything else —
which is what makes an entry added in a later release arrive put away rather than
appearing unbidden over the painting of every existing user. Folding rides the
panel's row rather than taking a record of its own: same fact, same panel, and a
panel is only ever folded while it is open. Reading happens where each signal is
built (`AppState::new`), so the first render is already the screen the artist
left.

A record may hold **more than one kind of row**, and two do: the tour's ledger
carries deed tallies beside the lessons already given, and the drag table carries
its rebindings beside the one bit saying this browser has been offered a preset
(§25.8). Untagged serde enums, and the alternative in both cases was a second key
in this registry to hold a handful of bytes belonging to one feature. The rule
for reaching for it is that the rows have to be *one feature's* state; two
features sharing a key would be the collision the registry exists to prevent,
arranged by hand.

The difference the traits encode is **what damage costs**. A list is read
element by element, so one entry today's build cannot make sense of is dropped
and the rest survive — which is what a library wants, and what every "a name
this build does not know" case leans on: a binding for a retired command, a
panel this build no longer has, a tally for a deed nothing counts. A whole
record is all-or-nothing, which is what `Prefs` wants: a half-read settings blob
is a worse answer than the defaults. `load_list` also tells `None` from
`Some(vec![])`, and two callers need it — an untouched quick-brush rack is
seeded from the preset library, an emptied one is left empty.

**Bytes are not kept in `localStorage` at all.** It is text, and ~5 MB of it per
origin *shared across all eleven records*. A brush shape's PNG lived there once,
base64'd inline in the shape library's rows: two of the app's own stamps are
408 KB and 226 KB on disk, half as much again as base64, and twice that against
the quota in an engine that counts a JS string's UTF-16. Five or ten imports
filled the origin — and a full origin does not break the shape library, it breaks
`set`, for the settings and the chord table and the tour's ledger and this
client's identity alike. Every standing choice this browser had made stopped
persisting, silently, because somebody imported a brush.

So a record may have a second half in **IndexedDB, keyed by the content id that
names the bytes** — a third trait, `Blob`, implemented *alongside* `Entry` on the
type that holds them:

```rust
impl storage::Entry for StoredShape { const STORE: Store = Store::Shapes; }  // name + id
impl storage::Blob  for ShapeEntry  { const STORE: Store = Store::Shapes; }  // the PNG

storage::blob_save::<ShapeEntry>(id, &png).await;      // -> stark.shapes/<hex>
let png = storage::blob_load_all::<ShapeEntry>(&ids).await;  // one exchange, in order
```

One `Store` row still answers where the record lives: a blob's key is the record's
key and then the id, so the two halves sort together, no record can reach into
another's bytes, and a *second* blob record is a second prefix rather than a
schema change. That last one is not tidiness — an object store can only be created
inside an IndexedDB `upgradeneeded`, so a store per record would put a version
bump behind every feature that ever wants to keep bytes, and a version bump is a
migration every other open tab has to be talked through.

Content-addressing is what keeps that door small. An id *names* its bytes (§19),
so a write is idempotent, a re-import is free, there is nothing to invalidate and
no schema to reconcile — which is exactly why the argument for JSON above does not
reach it. There is nothing in a blob store to reconcile *by* name.

The two halves are held together by the **write order**, since no transaction
spans two stores: *blob first, then the row; row first, then the blob.* A crash in
the middle strands some bytes, which costs space. The other order strands a row
whose shape has no image — a card that draws nothing and reports "failed to load"
every time it is clicked.

Two consequences worth stating, because both are new kinds of thing for this
registry to have. The blob store is **evictable** under storage pressure, so "the
row is here and the bytes are gone" is a state to expect rather than one that only
follows a crash; `shapes::load` drops such a row and writes the library back
without it, which is the list format's damage rule one store further down. And it
is **asynchronous**, which is the other half of what was wrong with the old
arrangement: `save_list` re-encodes a whole library per change, and it was doing
that on the thread the canvas paints on. Reading the shape library is a fetch now,
awaited once at start — ahead of `presets::apply_first`, the first thing that turns
a stamp id back into bytes.

**Adding a record**, in four steps:

1. A `Store` variant, with its key and its human name on the one row.
2. A serde type in the owning module. Give every field `#[serde(default)]` or a
   `Default` on the struct — that, not a version suffix in the key, is how a
   record survives the app version that adds or drops a field. A content id or
   key goes through `storage::hex`. A record that is a bare primitive needs a
   newtype: `impl Record for bool` would make *every* boolean in the frontend
   that record. **Bytes do not go in this type at all** —
   they go in the blob store above, named by the content id the row carries.
3. The one-line `Record` or `Entry` impl pairing the two, and a line in
   `every_record_claims_one_store` — which fails on the count if you forget,
   and is what catches two types claiming one variant. A `Blob` impl goes in the
   second list there, which checks that bytes never invent a record of their own.
4. One writer, called by everything that changes the state — the move
   `layout::set_open`, `navigator::set_open` and `settings::SettingToggle` all
   make. Durability is then structural: a new way to change the thing is
   remembered without its author thinking about storage.

**What is *not* stored here**: anything the document owns. A gradient, a preset
and a brush shape follow this browser; the ramp a fill commits, the color a
stroke carries and the shape's bytes are embedded **by value** in the action
(§8, §22.4), so a document stays self-contained and the library stays personal.
If a record would change what a peer sees, it is not this registry's.

### 25.7 What a dialog owes, and what a pop-out owes

A dialog joins no table. It is here because it is the other thing a feature adds
to the chrome — §11 says which of them exist and why each is a dialog rather than
a panel — and because what holds for one holds for all of them by construction
rather than by each author's care, which is the law the registries above are made
of, one seam over.

**A dialog is `widgets::Modal`.** The backdrop, the box on it, and the way out of
it are the component's, not the call site's; a dialog writes what is *inside* the
box and what its `on_close` does, and that is all. It went that way when the two
paragraphs below turned out to be one rule with two halves, and the second half
was a bug nobody could have found by reading the code.

**A dialog dismisses on the press it heard, not on any click.** The obvious
spelling — an `onclick` on the backdrop — is wrong, and wrong only under a pen,
which is why it survived in nine dialogs for as long as it did. A menu row acts
on `pointerdown`, deliberately: §11's dropdowns light-dismiss on blur, and acting
on the press is how a row wins that race — so the dialog is mounted while the
pointer that opened it is still down. A pen, like a touch, is a
*direct-manipulation* device: the browser withholds the whole
compatibility mouse sequence for the gesture and hit-tests it fresh **at the
release point**. So `mousedown`, `mouseup` and `click` are all delivered to the
backdrop the press itself created, and a dialog opened with a pen shut again
before the hand was off the tablet. Under a mouse the sequence is dispatched as
it goes and no click is generated at all once the press target has gone — the
whole class is invisible to the device the chrome was built with.

So the backdrop arms on its own `pointerdown` and a click dismisses only if it
finds itself armed. That is not a special case for menus: it is the general rule
that a click belongs to a press, and the press that opened the dialog belongs to
a menu row that no longer exists. The box stops both events on the way up, which
is what makes *armed* mean the press landed on the backdrop itself — a slider
dragged out of a dialog and let go over the dim stops reading as dismissal too.
Stopping them costs nothing above: the one listener that must hear every press
whatever it lands on binds in the capture phase for exactly that reason
(§11, `platform::on_window_pointer`).

What the rule covers is the *dismissal*, and one step of the same class is left
open: the retargeted click lands wherever the release was, so a control the
dialog itself puts under that point would be pressed by it. Every dialog we
raise is centred and every press that raises one comes from the rail or the
palette above it, so the release lands on the dim — but that is geometry
holding, not the rule. Closing it properly means the box refusing the click too,
and the box cannot do it in the bubble phase, where a button inside it has
already acted; it would take a capture-phase listener bound to the box's own
element.

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

**And what a pop-out owes.** A pop-out is the third surface — neither a panel nor
a dialog — and it is mostly what a *well* opens: a choice made by looking rather
than by reading, offered beside the control that holds it. There are five, and
they are one list, `widgets::PopoutId`, because at most one may be open at a time
and `modes::Composing`'s argument applies unchanged — two open at once is a state
nothing wants and nothing should have to prevent. They were `use_signal(|| false)`
locals of the surfaces that drew them, which made them invisible to the app and in
particular to Escape, whose ladder knew the dialogs, the composing modes, the
composing layers and Timeline mode and could not see a pop-out standing over all
of them. That is also why the rail's visibility menu is on the list though no well
opens it (§11): reaching it from the ladder is the only way its Escape is not a
second handler for a keystroke the window already hears.

Three things follow from being on that list rather than in a local:

- **Escape puts one down**, on its own rung above the dialogs — deliberately not a
  `Dialogs` flag, since that list is also what stands `FinishMode` down and the
  gradient library is opened *from* a bar while a fill is composing.
- **Whoever owns the well takes the pop-out with it.** A bar that unmounts, a panel
  that is closed: each clears the flag on the way out, or the next time that
  surface came up the pop-out would be standing open on it.
- **A canvas gesture closes the ones that float over the canvas.** The press that
  says the artist has gone back to painting is a stroke, not a dismissal, so the
  pop-out gets out of the way rather than catching it — a catcher over the canvas
  would eat the first stroke, which is worse than the bug it fixes. (General light
  dismiss — a press anywhere outside — is still owed for the four a well opens.
  The rail's menu has it already, and cheaply, because it holds the keyboard: a
  `focusout` that landed outside the flyout answers the question with no catcher
  at all.)

**Where a pop-out is drawn splits on one fact: whether the thing it flew out of is
clipped.** A bar's is drawn in the bar. Nothing clips `.bottom-bars`, so the frame
bar's colour picker hangs off the well in the markup, hangs *upward* because the
bar is on the floor of the window, and needs no coordinates at all.

A panel's cannot be. The stack is a scroll container that clips (`overflow-y:
auto`, `overflow-x: clip`) and every panel in it carries a `backdrop-filter`,
which makes a containing block — so a surface flown out of a panel row is cut off
at the column's edge whether it is `absolute` or `fixed`. There is no arrangement
of the markup that gets it out. It is mounted at the app root instead
(`panels::popout::StackPopouts`) and *placed*, against the row's own measured box —
the machinery the guided tour's card is placed with, which is why both now read
from one module (`stark-dioxus-frontend/src/anchor.rs`, §24.3).

Placing it is three answers, and only the middle one was a surprise:

- **Which row.** `PopoutId::in_stack` carries a selector per pop-out, and it names
  the *row* rather than the well inside it — a row spans the panel's content width,
  so its left edge is the panel's own and the pop-out's distance from the column is
  a fact about the column rather than about which control was pressed. Horizontally
  nothing is measured at all: the column is a fixed width at a fixed inset, so
  `.stack-popout` states its own right edge in `calc` and only `top` comes from
  Rust.
- **How it stays there.** A row moves without anything happening to the pop-out —
  the column scrolls, the window resizes, a panel above it opens or folds or is
  dragged. Those causes do not share an event shape (a scroll does not bubble; a
  fold is not an event at all), so a listener per cause is a list that is wrong
  again the next time somebody adds a way to move a panel. `anchor::follow` asks
  every animation frame instead and writes only when the answer changed: one
  `getBoundingClientRect` per frame for the seconds a pop-out is open, and the class
  ruled out rather than enumerated. It also answers **`None` while the row is
  scrolled out of the column**, so a pop-out is on screen exactly while the row it
  belongs to is, and comes back with it.
- **Which way it grows.** Down from the row's top or up from its bottom, whichever
  the window has more of, capped to what is left and scrolling inside the cap. Not
  centred on the row, which is the obvious third answer and the wrong one: centred,
  a surface may only be twice the room on its narrower side, so a pop-out beside the
  second row of the column would be capped at a couple of hundred pixels in exactly
  the case where the whole screen below it was free.

The Lighting panel is why this exists. Its canvas colour and its surface gallery
both stood *open in the column* — a 220px wheel and a grid of cards, for the two
choices in that panel made least often — and between them they pushed the light and
the substrate scale below the fold. What stayed behind is a swatch and a well, one
row each, saying which colour and which surface are in force; the wheel and the grid
are a press away and cost the column nothing.

### 25.8 Rebinding a drag, and the one time we ask

The drag table's rows are the user's. `DragBindings`
(`stark-dioxus-frontend/src/drags.rs`) is `defaults()` with this browser's own
rows laid over it, and it is what both readers ask — `find` on the press,
`armed` on the advertisement — so a rebind moves the cursor's promise in the
same frame it moves the press. The shape is `commands::Bindings`', copied
deliberately: the two tables share nothing but that shape, and a trait generic
over "a chord" would be two associated types and a blanket impl to save forty
lines of the plainest code in either module.

Four rules carry over from the chord table, each for its own reason:

- **An override is the action's whole binding.** Not "one more chord for it" —
  the row the user set is the row, and any shipped row for that action dies with
  it. Otherwise a rebind is an *addition*, and there is no gesture that removes
  the thing you were trying to move.
- **A rebind steals.** The action that held the chord keeps an override saying it
  now holds nothing, rather than falling back to a default: the default is a
  chord the user has just given away, and resurrecting it would undo the rebind
  they asked for.
- **Unbinding is an override to nothing**, for the same reason — an erased row
  must not come back on the next load.
- **The shipped preset is stored as no overrides at all.** Taking "Stark" clears
  the table rather than writing its three rows out; three stored rows saying what
  Stark already does is a browser no later build can ever move. Every other
  preset is stored in full, which is the mirror of the same argument — it is a
  claim about *another* app's table, and our defaults moving must not move it.

#### The presets

A modifier drag is discoverable only through what appears while it is held
(§25.3), which makes three of them three secrets — and every app an artist
arrives from keeps those secrets somewhere else. `DragPreset` is a named table
per app: Stark, Photoshop, Clip Studio Paint, Corel Painter, Rebelle, Krita.

Two things about that list are deliberate. It is indexed by **the app somebody is
arriving from**, not by distinct tables — Clip Studio Paint and Corel Painter
agree on all three and both have a row, because a preset is picked by
recognising a name and a merged row would offer neither. And a preset may leave
an action **unbound**: Krita reaches the layer carry through a tool rather than
through a modifier, and inventing one for it would be putting words in its
mouth. `matches` is asked per chip rather than answered once, so two presets that
agree both light up; lighting only the first would make clicking the second look
like it had done nothing.

The tables are transcribed from other software's documentation, which is a thing
that can be wrong and can go out of date. The design answer is not to be careful
about it, it is to make every row separately rebindable and say so on the
surface: a preset is a starting point, and the row under it is the fix.

#### The offer, made once

`Offer` is three states — `Unoffered`, `Due`, `Offered` — and the middle one is
what makes the feature work rather than annoy.

The trigger is `find` itself: a press whose chord this table has **nothing bound
to**, with at least one modifier held. That is somebody reaching for a binding
they already have in their hands, and it is the only evidence Stark will ever get
that the offer is worth making. Three exclusions fall out of stating it that way:

- **Asked of `lookup`, not of `find`'s answer.** A bound chord that *declines* —
  Shift over a selection tool, where it is the union marquee — is not an unbound
  one, and offering a preset table there answers a question nobody asked.
- **Modified presses only.** A bare contact is painting, and a bare right press
  is a chord nobody arrives holding.
- **Once ever, per browser.** The mark is written when the dialog is *shown*, not
  when it is answered: dismissing it is an answer, and a dialog that came back
  until it got the one it wanted would be a dialog nobody forgives.

Due and shown are two steps for the tour's reason (§24): the press that finds
nothing bound goes on to paint a stroke, so the dialog waits in `Offer::Due`
until `end_interaction` — the one place every canvas gesture is put down — takes
the hand off the canvas. A modal over a live stroke takes the canvas away
mid-mark.

The dialog is the one root dialog **no command opens**, which is why it has no
rail row and no chord. It is still in `AppState::root_dialogs`, so Esc lowers it
like any other, and it still owes what §25.7 says every dialog owes. The one
thing it owes on top is the way back: it says, above its buttons, that ⚙ Settings
lists these three drags and can change any of them at any time — because an offer
made once is a door that has to be visibly left open.

### 25.9 What a run of buttons owes

The other thing a feature adds to the chrome, here for §25.7's reason: what holds
for one run of buttons has to hold for all of them by construction rather than by
each author's care.

**Buttons that are alternatives to each other are one control, so they wear
`.segmented`.** That covers a mutually exclusive toggle group — the three
chrome-hiding states in ⚙ (`settings::SettingChoice`), the selection panel's tool
and combine rows, the eyedropper's scope, the timeline's speeds, the focal blur's
aperture (§21.12) — and equally any
run of buttons offering alternative answers to one question, whether or not one of
them stays lit: the drag presets (§25.8) are six tables to *start from*, and they
are the same control. The run's closed seams, single outer radius and one hairline
per seam say that picking one un-picks the rest. Six chips standing apart with gaps
between them say the opposite — six switches that could all be held down at once —
which is a promise the code then has to spend a comment apologising for.

**If they will not all fit on one line, it is a drop-down instead** (`.select`,
which several panels already use). Not a wrapped run: a segmented group that breaks
across two lines splits its seams mid-control and reads as two rows of buttons, so
the shape stops carrying "one of these" at exactly the point where there are enough
options for that to matter. Nor a run that shrinks to fit — a chip that quietly
narrows hides the day the option after next stopped fitting. So the ladder is
segmented while it fits on one line, `.select` when it does not, and never a
wrapped or squeezed run in between.

Two consequences worth stating, because they are what make "fits on one line" a
question with an answer:

- **Measure it, at the width the container actually has** — and measure the
  container too, against the row rather than against itself. A dialog is a fixed
  width (`.modal-wide` is 562px) and a panel column is 280px, so this is checkable
  rather than a matter of taste; it is worth checking in a real engine, since chips
  carry no `font-family` of their own and so render in the platform's UA button
  font. The drag presets take 85% of their column in Segoe UI and 92% in the widest
  UI fonts, which is the margin that keeps six of them chips. The trap is a check
  that cannot fail: `.setting-text` had no `flex-grow`, so every settings row was
  as wide as its own longest sentence — and asking whether a `margin-left: auto`
  child sat flush against *its parent's* right edge answered yes on all of them,
  while three controls that each wanted the row's edge each got a different one.
  Measure against the box you believe the width of, not the one you are inside.
- **Size the chips to whichever the labels are.** `.setting-choice` gives its three
  options an even split so none reads as the recommended one; the drag presets take
  their natural widths, because their labels are proper nouns of very different
  lengths and an even split could only fit "Clip Studio Paint" by clipping it. The
  even split is the default and the exception needs the sentence, which is this one.
