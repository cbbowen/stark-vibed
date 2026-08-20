# Recognizing and escaping modes

Recognizing and escaping editing modes is not intuitive today. The bottom bars
signify modes, but the signal is not visually evident, and there is no
consistent way out. This document records what the code actually does, the
decisions taken, and the build steps — in that order, because the fixes fall
out of an inventory that turned out smaller and sharper than "modal UI
overhaul."

## What the chrome does today

The bottom-bars column (`main.rs`, `.bottom-bars`) holds seven bars, but they
are two families with different contracts, currently drawn identically.

### Composing modes

Transform (§16.6), Perspective Guide (§20.5), Gradient Fill (§22.4) and
Gradient Trace (§22.2). Each mounts a deliberately invisible full-viewport
catcher that owns every pointer event — the pointer composes, it does not
paint — and holds an uncommitted preview computed against the committed
document. They are mutually exclusive by construction: every entry point calls
`modes::leave`, so two modes composing at once is a state the app cannot reach
(`crate::modes`). While one is live, every status bar stands down and every
edit command silently refuses (`commands::may_edit`).

Their exit contract is the problem:

- **"Done" is the only control.** Each mode bar offers commit and nothing
  else. The abandoning path — `modes::leave`, which drops the preview and
  commits nothing — is reachable only as a side effect: entering another mode,
  entering Timeline, or an undo/redo from the keyboard. **Cancel does not
  exist as an act the user can ask for.**
- **The trace has no bar at all.** Its indicators are a lit chip inside a
  pop-out that closes when the mode starts, plus a floating hint pill
  (`.gradient-trace-hint`). It is the least escapable of the four.

### Standing-state bars

Selection, Frame (§15.7), Filter (§21.6) and Pick (§18.0.2). These reflect a
fact — a selection exists, a frame or filter layer is selected, Alt is held —
and they end when the fact ends (deselect, select another layer, release Alt),
never by "Done". No catcher, no preview, no mutual exclusion beyond what layer
selection already provides.

What the pointer means under each varies, and the distinction between the
families is **not** "the canvas still paints":

- **Selection**: the canvas paints, through the mask. The one standing state
  under which painting continues.
- **Frame**: the pointer resizes and moves the frame by its edges and handles;
  a frame layer takes no paint.
- **Filter**: the bar carries the selected filter's own numbers; a filter
  layer takes no paint.
- **Pick**: a click samples color off the canvas rather than laying any.

So the honest split is the **exit contract**, not paintability: composing
modes hold something uncommitted and must be ended deliberately; standing
states describe what is selected or held and dissolve with it.

## Why it reads as confusing

The dangerous case is the composing family: the catcher is invisible on
purpose ("it is a hit area, and the thing it makes visible is elsewhere"), the
mode's bar looks exactly like a status bar, and when a stroke does nothing —
or an edit command silently refuses through `may_edit` — nothing on screen
says why. The one distinction that matters, *"why won't it paint?"*, is the
one the chrome does not draw.

## Decisions

Taken 2026-08-20; refinements from the build are marked **(as built)**:

1. **Esc leaves the composing mode**, committing nothing — `modes::cancel`
   under a key, plus a visible ✕ Cancel chip on the mode bars. Esc becomes
   the app-wide "put it down" key.
   - **(as built)** One exception, the trace: Esc pops back to the gradient
     bar the trace parked (`gradient_resume`) rather than dropping the whole
     composition. A stray click already ends the mode exactly that way — Esc
     is the same "never mind", and it must not take more with it than the
     click does. `modes::leave` (the entered-another-mode path) still drops
     everything, unchanged.
2. **Enter is Done** — commit and leave, exactly what the bar's Done chip
   does (`modes::finish`, dispatching to the same function each chip calls).
3. **Esc exits Timeline mode too.** It is called a mode, it wears the root
   class, and it should answer the same key. Ladder order: dialogs, then the
   composing mode, then the frame or filter selected for composing, then
   Timeline — one rung per press.
   - **(as built, second round)** The frame/filter rung was added on user
     report: selecting a frame reads as entering a mode, and Esc doing nothing
     there read as broken — the only visible effect was the clicked layer
     row's focus ring, promoted by the keydown, which looked like the name
     being "selected". Esc runs their bars' own Done (`frame::done_composing`
     — select the topmost paint layer, the only way these kinds are ever
     deselected; both Done chips now advertise Esc). Enter deliberately does
     **not** reach this rung: these are standing states, and an Enter claimed
     through one would eat every focused button's Enter for as long as the
     layer stayed selected.
   - **(as built)** The dialog rung **closes** the open dialog rather than
     declining in deference to it. The plan's premise was wrong: outside text
     fields, no dialog handles Esc at all — every element-level Escape in the
     app lives on an input, where the window binding is already withheld
     (`on_text_entry`). So the command is the only actor on the keystroke,
     which also retires the dioxus-vs-window handler-ordering hazard the plan
     worried about, and every dialog gains Esc-to-close for free. The flag
     list lives in one place, `AppState::root_dialogs`, beside the `Dialogs`
     struct it enumerates.
4. **A second Esc does not deselect.** A selection is committed, undoable
   document state, not a preview; Ctrl+D already names that act, and Esc doing
   double duty is how apps teach people to fear the key. (Revisitable — this
   was the close call.)
5. **No indicator that touches the artwork.** No color wash, no dimming, no
   hued border around the canvas: this is a painting app, and a cast biases
   exactly the judgment (color, alignment) the mode exists for. The mode
   register lives in the chrome — the bar, the cursor, the hint.

## Build steps

### 1. Esc and Enter as commands

Two new `Command` variants in the registry (`crate::commands`), so the acts
get names, tooltips, palette rows and rebindable chords like everything else:

- **CancelMode** — default chord `Code("Escape")`. Runs the ladder: close the
  open dialog(s); else `modes::cancel`; else leave Timeline. **(as built)**
  see decision 3 for why the dialog rung closes rather than declines.
- **FinishMode** — default chord `Code("Enter")`. Needs a `modes::finish`
  sibling to `leave` that dispatches per `Composing` variant. The commit logic
  already exists as `gradient_bar::finish` and `guides::end_guide_edit`; the
  transform's Done is inline in its bar's onclick and gets extracted to a
  `panels::transform::finish`. A trace has no commit — finish is the disarm,
  which hands back whatever the trace parked.

Implementation cautions, all learned from the code:

- **Dioxus `stop_propagation` never reaches the real DOM** (`input.rs`, the
  window-key binding's doc comment), so the command must be the keystroke's
  only actor. It is: outside text fields no dialog handles Esc itself, and
  text fields are exempt from the window binding on `on_text_entry` — the
  rename/draft fields and the palette keep their own Esc.
- **The dialog gate should not be a list somebody keeps** — or failing that,
  a list kept in exactly one place. **(as built)** `AppState::root_dialogs`,
  beside the `Dialogs` struct it enumerates, so a new modal's flag joins the
  list in the same edit that adds its field. The GPU-failure modal is
  deliberately absent: no flag, because it may not be dismissed (§5).
- **The Enter row must match only while a mode is composing.** Chips are
  `<button>`s, and a matched chord calls `prevent_default` unconditionally
  (`input::handle_keydown`), which would otherwise eat Enter-activation of any
  focused button the rest of the time. **(as built)** `Command::claims`,
  asked by `commands::find` before the caller's `prevent_default`: FinishMode
  claims Enter only while a mode is composing and no dialog is over it. Esc
  has no such double life and claims unconditionally.
- **Escape cannot be recaptured** — the palette's chord capture spends it on
  calling the capture off — so CancelMode's row is one-way for a user who
  rebinds it: movable off Escape, never back on. Backspace's own bargain,
  accepted for the same reason.

### 2. Make mode bars look like modes

- **A shared "composing" visual register** for the mode bars alone: accent
  edge or glow — the armed-trace blue (`#3a6ea5`) is already the app's "mode
  armed" color — and slightly stronger presence. Status bars stay neutral.
  The bar then reads as "you are in a mode," not "some controls appeared."
  **(as built)** `.mode-bar`: the accent ringing the surface, the label's ink
  tinted, and a 160ms rise on mount — the entering card the recess below is
  the other half of.
- **A ✕ Cancel chip on every mode bar**, wearing the CancelMode command whole
  so its tooltip advertises Esc through the registry. This is the
  discoverability fix for pen users, who never see a hotkey.
  **(as built)** Every mode bar but the guide's, where a Cancel would be a
  lie: a guide is shaped live (§20.5), nothing is uncommitted, so Esc and Done
  are one act and the bar offers it once. The Done chips advertise Enter
  through the same registry (`commands::advertised`) while keeping their
  per-mode tooltips.
- **Mode-specific cursors on the catchers.** They already flip to `grab` for
  space-pan; the gradient catcher already wore `crosshair`, and the transform
  and guide catchers now wear `move` — the first hover says "this pointer
  composes."
- **Fold the trace in.** **(as built)** The trace gets a real bar (`TraceBar`:
  the mode's name and its Cancel — no Done, because the release *is* the
  capture), and its hint-pill pattern is borrowed by the one other mode whose
  gesture is invisible until made: the gradient fill's axis, hinted per kind
  until the first drag composes one. The transform and guide modes stay
  pill-less deliberately — their widgets are on screen from the first frame.

### 3. The recessed stack

The nesting that actually exists is bounded to depth two by construction: a
mode bar stands in for the selection/frame bars (they return on Done), and the
one genuine park-and-resume — a trace suspends the gradient bar into
`gradient_resume` and hands it back. So no generalized stack. Instead: while a
mode composes, the standing bars stay **mounted but recessed** — dimmed,
scaled ~0.96, tucked behind the mode bar, `pointer-events: none` — with CSS
transitions. Entering reads as a card laid over; Done/Esc reads as the card
lifted off, and the return destination was visible the whole time.

**(as built, second round)** The column stacks **deepest on top**, on user
report (the trace's bar had landed below the gradient bar it parked): the
column is bottom-anchored, so an earlier child stands higher, and the order is
now the stack's — trace, then the mode bars, then the standing bars — so each
bar lands *above* the one it covers, the way a card lands on a pile. Recessed
bars therefore sit below the live bar and tuck upward toward it
(`translateY(-6px)`). The pick bar keeps the foot of the column: it coexists
with painting rather than covering anything.

Implementation note: SelectionBar/FrameBar currently return empty rsx while
composing, and a Dioxus unmount cannot be animated — recessing means keeping
them rendered with a class instead, a small local change to each.

**(as built)** SelectionBar and FrameBar recess; the parked gradient bar
renders recessed off `gradient_resume` (with the mode register taken off — a
shelved gesture is not the thing the canvas answers to). FilterBar turned out
never to have stood down at all — it mounts on its layer staying selected,
which no mode changes — so it keeps that behavior untouched. Recessed bars are
inert twice over: `pointer-events: none`, and every chip they carry runs an
act gated by `commands::may_edit`, which refuses while a mode is composing —
so even keyboard focus reaching one presses a button that declines. The
recess uses `filter: opacity()` rather than the `opacity` property, which the
chrome fade owns (`.chrome.dimmed`); the computed-style outcome (the
transition tie-break included) was verified in headless Blink against the real
stylesheet.

## Order and why

Step 1 first: it is small, structural, and adds the missing *act*, not just a
key. Step 2 is the recognition fix and rides on step 1's command variants.
Step 3 is pure polish once the semantics are visible, and does nothing for a
user who has not yet learned what the bars mean — which is why it goes last.
