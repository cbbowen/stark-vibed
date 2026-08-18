# The guided tour

Lessons that arrive once the artist has earned them — §24.

> Part of the Stark design docs. Index and conventions: [CLAUDE.md](../CLAUDE.md).
> Section numbers are stable — code cites them as `§n.m`.

## 24. The guided tour

Stark's chrome differs from the apps people arrive from, and it differs mostly by
subtraction: no toolbar of forty tools, no modal dialog per operation, several of
the best bindings on a modifier rather than on a button. That is the design
working. It is also the design's standing bill, and the bill has a name — **a
control that is not on screen is a control nobody finds**. The eyedropper already
pays it for one binding by bringing its options bar up on Alt (§18.0.2): press the
modifier and the thing it does announces itself. This chapter is the same answer
generalized to everything no modifier announces.

The whole feature is `stark-ui/src/tutor.rs`, one line in `dispatch`, and two
gestures that say when they are running (§24.2). It turns on three decisions: what
brings a lesson (§24.1), where the counting comes from (§24.2), and what a card is
allowed to do to the screen (§24.3). §24.4 is the ledger, §24.5 the table of
lessons as it stands, and §24.6 what is deliberately absent.

### 24.1 A lesson is owed, not scheduled

Nothing runs at first launch. There is no tour to start, no Skip button, and no
step counter. A lesson is attached to a **deed** — something the user has done —
and a **count**:

> Paint three strokes and the Brush panel explains itself.
> Reach for the size slider ten times and the tour mentions the drag.

So a tip only ever reaches somebody who has already demonstrated that they care
about the thing it is about. The artist who never pans has never been shown the
Navigator; the one who has panned across two screens has, and it arrived as an
answer to a question they were visibly asking. That is the difference between
guidance and a manual, and it is worth stating as an inequality: **a lesson given
before the deed is noise, and its cost is paid by everyone.** A lesson given after
is at worst redundant, and its cost is paid by the one person it was aimed at.

Two consequences follow and are not negotiable:

- **No lesson fires on a first try.** `no_lesson_fires_on_a_first_try` refuses a
  threshold below two. The first minute in a new app is the minute with the least
  attention to spare.
- **The tally spans sessions.** "The third stroke" is not a claim about one visit,
  so the ledger follows the browser the way the shape and preset libraries do
  (§24.4).

Each lesson is given **once, ever**. There is no repetition, no "show again", and
no schedule that could bring one back.

### 24.2 Deeds are read off the command stream

`tutor::observe` is called from `state::dispatch`, which is the single seam every
mutation *this user* makes goes through (§4). Reading deeds there rather than
tapping the handlers that produce them buys three things, and the third is the one
that made the design work:

**A new way to do a thing is counted for free.** The brush's size is reachable
from the Brush panel's slider, from the accelerator drag over the canvas
(§18.1.9), from a quick-brush slot (§18.1.8) and from a preset. None of those
knows this module exists. A fifth way, added next year, will not have to either.
That is the standing preference for ruling out a class rather than enumerating its
instances, applied to instrumentation.

**A collaborator's work is not counted as yours.** Remote actions reach the engine
through `with_engine`, never through `dispatch` (§12.4) — a split that already
existed, for broadcasting, and turns out to be exactly the split this needs.
Nothing had to be added to get it.

**What a command means is decided by what moved.** `ViewCommand::SetBrush` carries
a whole `BrushParams`, and the same variant is emitted by the size slider, the
color picker, the eyedropper, a preset click and a slot release. Naming the
*sender* would need five call sites to agree; naming the *difference* needs none:

```rust
fn brush_deed(was: &BrushParams, now: &BrushParams) -> Option<Deed> {
    let mut tuned = *was;
    tuned.radius = now.radius;
    tuned.dynamics.add = now.dynamics.add;
    if tuned == *now && (was.radius != now.radius || was.dynamics.add != now.dynamics.add) {
        return Some(Deed::TunedBrush);
    }
    // …the same shape again for color…
}
```

The test is **confinement, not difference**: this counts as a size change when the
brush is otherwise untouched. A preset click moves a dozen fields at once and is
therefore an adjustment of nothing, which is the right answer and is not a special
case anywhere. And because "everything else is equal" is one `==` against the real
type rather than a list of comparisons, a brush parameter added later cannot
silently fall out of the rule.

This is only sound because `observe` runs **before** the command reaches the
engine. The brush it compares against is the one the engine is still holding; a
line moved below `with_engine` would compare the new brush with itself and the
tour would go quiet with no error anywhere.

#### The two things the stream cannot say

Both are the frontend's in the sense the dwell behind `GestureCommand::Hold` is
(§6.9) — facts about a hand, which the engine has neither the clock nor the pointer
to know.

**When a run of commands is one act.** A slider drag is one intention and sixty
`SetBrush`es. So this module has a clock, and `COALESCE` — half a second — is the
whole of it. Reports of the same deed closer together than that are one deed,
measured from the **last** report of a run rather than its first, so a slow drag
stays one deed however long the hand takes.

**Which gesture wrote the brush.** This one is not a detail, because two of the
lessons teach gestures that produce the very deed they are counted by: an Alt-drag
off the painting *is* a color change, and an accelerator drag *is* a size change.
Counted naively, the tour would wait for somebody to use the eyedropper five times
and then offer to explain the eyedropper.

So the two gestures say so while they run — `tutor::via_shortcut`, called in pairs
around the whole gesture rather than at each write — and a brush write made under
that flag is not counted. That turns the flaw into the property the tour most
wanted:

> **It never teaches a gesture you already use.**

Somebody who only ever eyedroppers never accumulates the deed and is never told
about it; somebody who has clicked the Oklab field five times is told there is a
faster way, which is the whole point. A flag left set by a gesture that ended
without saying so costs *counting*, never a wrong card — which is the failure
direction to have, since the other one is a card nobody asked for.

`via_shortcut` is the one thing the tour asks of the rest of the app, and it has
exactly two callers. It earns the exception on the same grounds the hold does:
what a gesture *is* cannot be derived from the commands it emits, so the side that
knows is the side that says.

Panning is the one deed that is a *measurement* rather than a report: no single
`Pan` is long. Its run accumulates across `Pan` **and** `Pinch` — a two-finger pan
on a tablet is a pan — and scores at the moment it crosses `LONG_PAN`, so nothing
has to detect a run's end. A run that stops short simply never scores.

Two commands are read for what they are not. `GestureCommand::End` is a stroke and
`Cancel` is not, because a stroke abandoned by a second finger (§18.1.7) left no
paint; and a `DocCommand::Redo` counts only while the timeline transport is not
playing, since playback drives the playhead with that very command (§18.2.4) and
would otherwise score eight redos a second. The user's own redo stops playback
before it dispatches (`input::edit_history`), so the guard reads false exactly when
it should.

### 24.3 The card, and what it may take

One card at a time, floating beside the thing it points at, with an arrow across
the gap. It is **ordinary floating chrome**: it wears `chrome_class`, so it fades
out mid-stroke and back when the hand lifts, along with the panels and the bars
(§11). It is not a modal, it never covers the middle of the window, and there is
nothing to dismiss before painting can continue.

Where it goes is measured, not guessed. `platform::anchor_box` reads the anchor's
box off the DOM and the card is placed against that box's own edges — with a
`translate` doing the work that knowing the card's own width would otherwise
require, so nothing has to be measured twice. Which *side* it sits on is the
lesson's to declare: which side has room is a fact about where that chrome lives in
the window, and the DOM does not say it. The panels are a column down the right, so
their cards go left; the timeline bar is across the bottom, so its card goes above.

Panels are found by `layout::panel_key` — the same function that writes the
`data-panel` attribute — rather than by a selector spelled out in the lesson table.
A box matched to the wrong panel is measured in silence (§11), and here it would
show as a card in the corner of the window with nothing beside it to explain.

#### Coming due and being shown are two steps

A lesson **opens** the panel it is about, and this is what forces the split. A
panel opened mid-stroke is put straight back to sleep by the release
(`input::end_interaction`, §11), so the card would come up beside a panel that had
just stood down. A lesson that comes due with the canvas in hand therefore waits in
`TutorState::due` until the screen is the user's again — which also covers the two
other states where a card would be *wrong* rather than merely unwelcome: a
composing mode owning the whole window (§16.6, §20.5, §22), and a dialog over the
top of everything a card could point at.

What a lesson opens is derived from what it points at, not named beside it:

```rust
fn reveal(self, state: AppState, layout: PanelLayout) {
    match self {
        Anchor::Panel(id) => open_panel(state, layout, id),
        Anchor::TimelineBar => panels::timeline::set_open(state, true),
    }
}
```

Two fields could state those differently, and a lesson that opened the Color panel
while pointing at the Brush one is the kind of thing that survives review.

**One card at a time, and a passed-over lesson is not lost.** A threshold is a
floor rather than an equality, so a lesson skipped because another card was up
comes due again on the very next deed of its kind. The same property is what makes
a reload safe: a lesson is written into the ledger when it is **dismissed**, so a
visit that ends with a card still open still owes it.

**Closing the panel a card is about dismisses the card.** It is an answer, and
taking it as one is also what keeps the state machine from latching: a lesson left
showing beside a panel that no longer exists would block every lesson after it, and
the tour would end silently at whichever tip the artist happened to close a panel
under. The test is the app's own state (`Anchor::on_screen`) and not whether the
anchor measured — a measurement comes back empty for a frame while the browser lays
out a panel that has only just opened, and dismissing on *that* would dismiss every
lesson the instant it appeared.

### 24.4 The ledger

One `localStorage` key, `stark.tutor.v1`, in `crate::storage`'s line table — the
format whose whole point is that a line nobody can read costs that line and not the
library. Two kinds of record:

```
deed|stroke|7
given|brush-panel
```

The two halves forget differently, on purpose:

- **A tally under a name this build does not know is dropped.** A deed no longer
  counted has no lesson to feed, so keeping the number would be keeping it for
  nobody.
- **A lesson name is kept whatever the table now says.** Somebody who has seen a
  tip must not be shown it again because a release renamed its neighbour. A key is
  stable across edits of the table for exactly this reason — an index would move
  the moment a lesson was inserted above it.

The tally is written after every deed, which is at most one write per `COALESCE`
and only while the app is being used. A deed nobody has done is the *absence* of a
line rather than a line saying zero, so the table only ever holds what happened.

The switch that turns the tour off is not here — it is a preference
(`Prefs::tips`, ⚙ → Guidance), because it is a choice somebody made rather than a
record of what happened. Deeds are tallied whether or not tips are shown, so
turning them back on resumes rather than restarts, and turning them off is not a
way to lose your place.

### 24.5 The lessons

The table is the whole feature: a lesson is a row, and adding one costs no code
anywhere else unless it counts a deed nothing counts yet.

| After | Deed | Points at | What it says |
|---|---|---|---|
| 3 | a brush stroke | Brush panel | Size and Flow, the brush editor's live test stroke, and that the list below is a library |
| 10 | size or flow moved *by a control* | Brush panel | Ctrl (⌘) + drag on the canvas — sideways for size, up and down for flow (§18.1.9) |
| 2 | a long pan | Navigator panel | Drag inside the miniature to travel; right-drag to turn the canvas |
| 5 | the color moved *by a control* | Color panel | Alt + drag samples off the painting, and the bar that comes up says what the sample sees (§18.0.2) |
| 2 | a redo | Timeline bar | The history is a place you can stand in, not a stack you pop (§18.2.4) |

The counts are set from what each deed *costs* to keep doing the hard way. Three
strokes is barely a commitment and the Brush panel is where everything about a
brush is. Ten trips to the size slider is somebody who has decided that this is how
they work, which is exactly the moment the drag is worth knowing and well past the
moment it would have been an interruption.

"By a control" in two of those rows is `via_shortcut` doing its work: the deed is
*reaching for the slider*, not *the size changing*, so the artist who already drags
never accrues it.

A card says what to **do** and then the thing about Stark that makes it worth
doing. A tip that only names a shortcut is a keyboard reference, and the menus
already carry one.

### 24.6 What is deliberately not here

- **No first-run tour.** Covered at length in §24.1: the minute with the least
  attention to spare is the first one.
- **No progress, no completion, no badge.** The tour has nothing to finish. A
  count of lessons seen would make the feature into a thing to get through, which
  is the opposite of what it is for.
- **No "show these again".** Turning the switch off and on keeps your place rather
  than resetting it, and there is no control that clears the ledger. Re-teaching
  somebody who has been taught is the failure mode this whole design is built to
  avoid, so the app declines to offer it; a browser's own site-data controls remain
  the honest way to become new again.
- **No timing out.** A card stays until it is acknowledged. It costs nothing to
  leave up — it fades with the rest of the chrome for every gesture — and a
  paragraph that vanished while it was being read would be worse than one that
  waited.
- **No lesson without an anchor.** Every lesson points at chrome that exists. A
  card floating in the middle of the window with nothing to explain would be an
  announcement, and this feature is not an announcement channel.
