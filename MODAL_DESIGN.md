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

Taken 2026-08-20:

1. **Esc leaves the composing mode**, committing nothing — `modes::leave`
   under a key, plus a visible ✕ Cancel chip on every mode bar. Esc becomes
   the app-wide "put it down" key.
2. **Enter is Done** — commit and leave, exactly what the bar's Done chip
   does.
3. **Esc exits Timeline mode too.** It is called a mode, it wears the root
   class, and it should answer the same key. Order in the ladder: an open
   dialog wins (handled by the dialog itself; the command declines), then a
   composing mode, then Timeline.
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

- **CancelMode** — default chord `Code("Escape")`. Runs the ladder: decline if
  a root dialog is open; else `modes::leave`; else leave Timeline.
- **FinishMode** — default chord `Code("Enter")`. Needs a `modes::finish`
  sibling to `leave` that dispatches per `Composing` variant. The commit logic
  already exists as `gradient_bar::finish` and `guides::end_guide_edit`; the
  transform's Done is inline in its bar's onclick and gets extracted to a
  `panels::transform::finish`. A trace has no commit — finish just leaves.

Implementation cautions, all learned from the code:

- **Dioxus `stop_propagation` never reaches the real DOM** (`input.rs`, the
  window-key binding's doc comment): with the brush editor open and focus not
  in a text field, the modal's element-level Esc *and* the window binding both
  fire. Without the dialog gate, one press would close the dialog and silently
  abandon the composition behind it. Text fields are already exempt — the
  keydown side of the window binding is withheld on `on_text_entry`, so the
  rename/draft fields and the palette keep their own Esc.
- **The dialog gate should not be a list somebody keeps.** The root dialogs
  are each a `Signal<bool>` on `AppState` today (brush editor, preset save,
  new document, export, settings, share, timing stats, credits). A
  `dialog_open` helper that enumerates them is the checkable minimum, but the
  standing preference is to rule the class out structurally — one signal that
  names the open dialog, which `main` already effectively switches over —
  rather than an enumeration that a ninth dialog forgets to join.
- **The Enter row must match only while a mode is composing.** Chips are
  `<button>`s, and a matched chord calls `prevent_default` unconditionally
  (`input::handle_keydown`), which would otherwise eat Enter-activation of any
  focused button the rest of the time. Esc has no such conflict worth
  guarding.

### 2. Make mode bars look like modes

- **A shared "composing" visual register** for the mode bars alone: accent
  edge or glow — the armed-trace blue (`#3a6ea5`) is already the app's "mode
  armed" color — and slightly stronger presence. Status bars stay neutral.
  The bar then reads as "you are in a mode," not "some controls appeared."
- **A ✕ Cancel chip on every mode bar**, wearing the CancelMode command whole
  so its tooltip advertises Esc through the registry. This is the
  discoverability fix for pen users, who never see a hotkey.
- **Mode-specific cursors on the catchers.** They already flip to `grab` for
  space-pan; a crosshair for the gradient axis and a move cursor for the
  transform make the first hover say "this pointer composes."
- **Fold the trace in.** It gets a real bar (or at least the register and the
  ✕), and its hint pill — a non-interactive "what to do" instruction riding
  the mode — is the pattern the other modes should borrow, not the exception.

### 3. The recessed stack

The nesting that actually exists is bounded to depth two by construction: a
mode bar stands in for the selection/frame bars (they return on Done), and the
one genuine park-and-resume — a trace suspends the gradient bar into
`gradient_resume` and hands it back. So no generalized stack. Instead: while a
mode composes, the standing bars stay **mounted but recessed** — dimmed,
scaled ~0.96, slid up behind the mode bar, `pointer-events: none` — with CSS
transitions. Entering reads as a card laid over; Done/Esc reads as the card
lifted off, and the return destination was visible the whole time.

Implementation note: SelectionBar/FrameBar/FilterBar currently return empty
rsx while composing, and a Dioxus unmount cannot be animated — recessing means
keeping them rendered with a class instead, a small local change to each.

## Order and why

Step 1 first: it is small, structural, and adds the missing *act*, not just a
key. Step 2 is the recognition fix and rides on step 1's command variants.
Step 3 is pure polish once the semantics are visible, and does nothing for a
user who has not yet learned what the bars mean — which is why it goes last.
