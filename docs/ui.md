# The chrome's registries

Commands, chords and drag bindings — what a new UI feature joins, and how — §25.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 25. Commands and drag bindings

The chrome reaches every act and every bound gesture through one of two tables.
The **command registry** (`stark-ui/src/commands.rs`) holds the simple acts —
undo, deselect, open the export dialog — each a variant of `Command` carrying
its whole description, with the keyboard as one rebindable column. The **drag
table** (`stark-ui/src/drags.rs`) holds the canvas presses that open something
other than painting — the brush-tuning drag (§18.1.9), the eyedropper (§18.0.2),
the layer carry (§16.11) — each a row binding an exact chord+button to a
`DragAction`. §11 tells both
designs' history; this chapter is the working guide, written for the day a
feature is added: which table the feature belongs to, if either, and the steps
that keep the tables true.

One law covers both, and it is the reason they exist: **one authority, and
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
