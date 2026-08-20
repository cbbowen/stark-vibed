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

The whole feature is `stark-ui/src/tutor.rs`, one line in `dispatch`, and six call
sites that say something the command stream cannot — three brackets and three
reports (§24.2). It turns on three
decisions: what brings a lesson (§24.1), where the counting comes from (§24.2), and
what a card is allowed to do to the screen (§24.3). §24.4 is the ledger, §24.5 the
table of lessons as it stands, and §24.6 what is deliberately absent.

Since every panel now starts closed (§11), the tour also carries the opening
screen: the first three lessons are how the stack gets assembled for somebody who
has not found the Panels menu.

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

- **Almost no lesson fires on a first try.** `no_lesson_fires_on_a_first_try`
  refuses a threshold below two, and the first minute in a new app is why: it is the
  minute with the least attention to spare. The exceptions are named as **deeds**
  rather than as lessons, because the question is never "is this tip important" —
  every tip thinks it is — but *could somebody have done this without meaning to*.
  Three could not: closing a panel **raises** the question its lesson answers
  (answering on the second close would be answering late, with the gap spent
  believing the panel was gone); opening the brush editor **is** the request its
  series answers; and a guided line is not reachable by accident at all — a guide
  made, left visible, a stroke drawn *and* held still. An *exception* list, so a
  deed added later is held to the strict rule by default.
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

**What wrote the brush.** This one is not a detail, because three of the lessons
teach ways of changing the brush that produce the very deed they are counted by: an
Alt-drag off the painting *is* a color change, an accelerator drag *is* a size
change, and a quick slot is a whole tool arriving in exactly the shape a preset
click arrives in. Counted naively, the tour would wait for somebody to use the
eyedropper five times and then offer to explain the eyedropper.

So the code that makes those writes brackets them — `tutor::not_reaching`, opened
and closed around the stretch rather than called at each write — and a brush write
made inside a bracket is not counted. That turns the flaw into the property the
tour most wanted:

> **It never teaches a gesture you already use.**

Somebody who only ever eyedroppers never accumulates the deed and is never told
about it; somebody who has clicked the Oklab field five times is told there is a
faster way, which is the whole point.

Three callers, and they are three different reasons the write is not somebody
reaching for a slider:

| Bracket | Why |
|---|---|
| `input::Tune` (§18.1.9) | the gesture the size/flow lesson teaches |
| `input::pick_color` (§18.0.2) | the gesture the eyedropper lesson teaches |
| `presets::wear` | a whole tool arriving — a preset click, or a quick slot in either direction (§18.1.8). Not an adjustment of the brush you had, even in the rare case where it differs in nothing but its size |

It is a **depth count rather than a flag**, because the brackets nest: a number key
held mid-tuning-drag swaps the tool inside the drag's own bracket, and a flag would
let that swap's close cancel the drag's. A bracket left open by a caller that failed
to close it costs *counting*, never a wrong card — which is the failure direction to
have, since the other one is a card nobody asked for.

#### The deeds that are not in the stream at all

Three are reported outright (`tutor::did`):

- **A preset put on from the library.** The command it leads to says a brush
  changed and cannot say a row was clicked — and the quick slots, which that lesson
  goes on to teach, emit a command of exactly the same shape.
- **A panel closed.** Which panels are open is the frontend's alone and reaches no
  engine at all, so there is no command to read. `layout::close_panel` reports it,
  and only where a panel actually went away.
- **The brush editor opened.** A dialog is frontend state too, and this one is the
  request its whole series answers.

`not_reaching` and `did` are the only two things the tour asks of the rest of the
app, and they are opposites: one says a command should not be read, the other says
something happened that no command describes. Both earn the exception on the grounds
the hold does (§6.9): what a gesture *is* cannot be derived from the commands it
emits, so the side that knows is the side that says.

#### And one the engine has to be asked for

Whether a held stroke **snapped** is the mirror image of the hold itself. §6.9
splits that gesture down the middle: the frontend owns the dwell, because how long a
pause has to be is a fact about a hand and the engine has no clock; the engine owns
what a hold *means*. So whether a hold found anything is knowable only on the engine
side, and `Engine::assisted` is the named read that answers — a request in §4's
sense, alongside `view` and `tow_string`.

It answers **only before the gesture's `End`**, which is exactly when `observe`
runs. What a stroke commits is the path the shape produced, not the shape, so the
assist goes with the gesture; asked a moment later there would be nothing to see.
That the tour reads the engine before the command reaches it — already load-bearing
for `brush_deed` — is what makes this possible at all.

The read answers a two-variant `Assisted { Line, Ellipse }` rather than the assist's
own `AssistShape`. What a caller out there wants is which kind; what the shape
carries is geometry — two points, a frame, a winding, the plane it is a circle on —
and publishing those would fix the assist's internals as an interface for the sake
of a question answered by one bit.

**One command, three deeds.** A stroke that snapped along a vanishing line is a
stroke, an assisted stroke and a guided line all at once, and each feeds a different
lesson, so `read` answers with a list rather than an `Option`. Counting only the
most specific would stall the two behind it for somebody who works entirely on a
grid — and the list costs nothing where it matters, since `Vec::new()` does not
allocate and that is what every command at pointer rate gets.

Panning is the one deed that is a *measurement* rather than a report: no single
`Pan` is long. Its run accumulates across `Pan` **and** `Pinch` — a two-finger pan
on a tablet is a pan — and scores at the moment it crosses `LONG_PAN`, so nothing
has to detect a run's end. A run that stops short simply never scores.

Two commands are read for what they are not. `GestureCommand::End` is a stroke and
`Cancel` is not, because a stroke abandoned by a second finger (§18.1.7) left no
paint; and a `DocCommand::Redo` counts only while the timeline transport is not
playing, since playback drives the playhead with that very command (§18.2.4) and
would otherwise score eight redos a second. The user's own redo stops playback
before it dispatches (`commands::edit_history`), so the guard reads false exactly when
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
require, so nothing has to be measured twice.

**The measurement retries, and that is the ordinary path rather than a defence.** A
card is placed by the very effect that *revealed* what it points at — the panel it
opened, the rack it pinned — and a reveal is a signal write whose render has not
happened yet, let alone been laid out. So the first look routinely finds nothing,
and one animation frame is a race against Dioxus's own patch rather than a
guarantee. `measure` asks again for up to eight frames.

Losing that race was silent and strange, which is worth recording: the card was
armed and correct and simply never drew, until something *else* the measure effect
follows moved and measured it again. `canvas_active` is one of those, so the symptom
was a tip that appeared the next time the artist painted a stroke — nowhere near the
click that had earned it.

*Where* it sits is the lesson's to declare, because neither half is in the DOM:
which side has room is a fact about where that chrome lives in the window, and
whether the anchor has a meaningful top edge to line up with is a fact about what
kind of thing it is. The placements are named for the picture rather than composed
out of a side and an alignment — two enums would also spell several combinations
that mean nothing:

| `Side` | Used by | Why |
|---|---|---|
| `LeftAtTop` | the panels | a box hanging from the top of its column, so the top edge is the one always on screen |
| `LeftAtMiddle` | the panel column | a whole edge of the window, with no meaningful top to line up with |
| `RightAtTop` | the command rail | a box down the left that hugs its contents |
| `RightAtMiddle` | the quick-brush rack | the chrome down the left, and a box that centres its rows in a column running to the foot of the window — so its top edge is a long way above anything drawn |
| `RightAtBottom` | the navigator's miniature | a box that *sits on* the foot of the window, so the bottom edge is the one always on screen — and whose height is the artwork's aspect, so its arrow is placed from a measurement rather than from a constant |
| `Above` | the timeline bar | across the foot of the window |
| `Inside` | the canvas | not a control at all but a *place* — see below |

Panels are found by `layout::panel_key` — the same function that writes the
`data-panel` attribute — rather than by a selector spelled out in the lesson table.
A box matched to the wrong panel is measured in silence (§11), and here it would
show as a card in the corner of the window with nothing beside it to explain.

**Cards narrow rather than move when they run out of room.** Each placement adds a
`max-width` computed from the anchor's own edge — `calc(100vw - …)` where the
viewport is what constrains it, so nothing has to be measured and nothing has to
know the card's width. A card that *shifted* to stay on screen would leave its arrow
pointing at nothing, and the arrow is the half that says which thing is being talked
about.

**Vertically there is no such move, so the placement has to be right.** A card
narrowed to fit its width is a card that has grown *taller*, which is why the trick
above has no mirror image: nothing at runtime can rescue a card hung into an edge.
What decides it is the choice of edge to hang from — always the one the window
cannot push off screen, which is the top for a box in the panel stack and the bottom
for one standing in the corner. That is a claim about where each piece of chrome
lives, and `tutor`'s own test asserts it lesson by lesson: an anchor against an edge
of the window may not carry a placement that reaches toward it. The regression it
was written for is real — the navigator's lesson kept its centred placement when the
miniature moved out of the panel stack into the bottom-left corner (§11), and a card
centred on a box that stands 14px off the foot of the window hangs its lower half
over the edge.

**One anchor is the painting, and one placement goes over it.** A lesson about a
gesture made *on the canvas* has nothing to stand beside, and standing it beside a
panel would say the panel had something to do with it. So `Side::Inside` puts the
card in the middle of the picture a quarter of the way down, pointing down into it —
the only arrangement that says *the thing I am describing happens there*.

That is the one card that sits over the middle of the work, which is why **every**
card declines the pointer and lets its two buttons take it back — the
`.slot-overlay` bargain (§18.1.8). A press on a card's own background is far more
likely to have been the start of a stroke than an attempt to click a paragraph, so
it paints. The `.chrome` fade takes the buttons' events away as well for the length
of every gesture.

**One anchor is inside a dialog, and it inverts two rules.** Every other card
stands down while a modal is up, because a modal is over everything a card could
point at — but the brush editor's series points *at* the modal, so
`Anchor::inside_dialog` is what the promotion rule asks instead of testing the
signal directly. Those cards also sit above the backdrop (`z-index: 101`, and only
they do) and take the pointer whole, where a card over the painting declines it:
inside a dialog there is no stroke a press could have been the start of, and what is
underneath is somebody else's control — here, a test canvas that would read a click
on a paragraph as a mark to draw.

**One anchor is invisible, and that is also the point.** The panel column's wake slice
(`.panel-wake`, §11) is a box with nothing drawn in it, so the card is the only
thing that can say where it is — which is exactly the lesson. It also gets the
best dismissal in the set: reaching into the column wakes the panels, the slice
leaves the DOM, and `Anchor::on_screen` reads that as the lesson done. You are not
told you have learned it; you are shown, and the card goes.

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

**Dismissing a card offers the next one its deed still owes.** That is what makes a
*series* possible — the brush editor's five cards all wait on one deed, and being
opened once earns every one of them, so acknowledging one has to bring the next
rather than waiting for the dialog to be opened again. The button says which it is:
"Next" while another is owed, "Got it" on the last, because "Got it" on the first of
five is a lie about how many are coming.

It costs nothing elsewhere, because the ordinary case is a deed that owes one lesson
at a time; where it owes more, that is a backlog built up while cards were passed
over, and draining it in order beats making the artist earn each one twice. And it
cannot run away: every turn marks one lesson given and `due` skips those.

The chain runs on dismissal *by the button* and deliberately not when the anchor
disappears (`abandon`). Closing the brush editor takes the anchor out from under
every card in the series at once, and a chain there would retire the lot in a single
flush — the artist would have been "taught" four things they never saw. What happens
instead is that the card on screen is answered and the rest stay owed for next time.

**One card at a time, and a passed-over lesson is not lost.** A threshold is a
floor rather than an equality, so a lesson skipped because another card was up
comes due again on the very next deed of its kind. The same property is what makes
a reload safe: a lesson is written into the ledger when it is **dismissed**, so a
visit that ends with a card still open still owes it.

**A card pointing into the panel stack holds it up.** The stack is the one piece of
chrome with a second way to be out of the way: it stays down after a gesture until
the pointer reaches for it (§11). Revealing a panel wakes it, but the next stroke
would put it back to sleep underneath the card — which fades for the gesture like
all the chrome and comes back, leaving an arrow aimed at a panel that did not. So
`layout::standing_down` — the question the fade *and* the wake slice are both
decided by — asks the tour first, and a lesson anchored at a panel answers. The
lesson about the panel *column* deliberately does not: a strip you reach into to
bring the panels back is unteachable with the panels already up.

Letting go is a wake rather than a release, for the same reason. If the card merely
stopped holding, the stack would fade the instant "Got it" was pressed, which reads
as the acknowledgement having closed the thing it was about; waking it properly
leaves it where every other route to a panel leaves it — up until the next gesture.

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

Turning it off also **takes down the card that is already up** — it was promoted
under the old answer, and a preference that leaves the very thing it governs
standing on screen is one nobody would believe. The card is not marked as given on
the way out, for the same reason the rest of the ledger survives the switch: nobody
was taught a lesson they switched off mid-sentence. Both controls go through one
function (`tutor::set_enabled`) rather than writing the signal themselves — the
dialog's row and the card's own "Stop tips" are two controls for one switch, and a
rule kept in one of them is a rule the other disagrees with.

### 24.5 The lessons

The table is the whole feature: a lesson is a row, and adding one costs no code
anywhere else unless it counts a deed nothing counts yet.

| After | Deed | Points at | What it says |
|---|---|---|---|
| 2 | a brush stroke | Color panel | Here is the first panel, and the picker is Oklab rather than a hue wheel — the slider is lightness (§6.5) |
| 3 | a brush stroke | the panel column | The panels stand down while you paint; reach into the right-hand edge and they come back (§11) |
| 5 | a brush stroke | Brush panel | Size and Flow, the brush editor's live test stroke, and that the list below is a library |
| 10 | size or flow moved *by a control* | Brush panel | Ctrl (⌘) + drag on the canvas — sideways for size, up and down for flow (§18.1.9) |
| 5 | a preset put on from the library | the quick-brush rack | A held number is a brush you *borrow*; tuning under the hold keeps the change (§18.1.8) |
| 2 | a long pan | the navigator's miniature | Drag inside the miniature to travel; right-drag to turn the canvas |
| 5 | the color moved *by a control* | Color panel | Alt + drag samples off the painting, and the bar that comes up says what the sample sees (§18.0.2) |
| 2 | a redo | Timeline bar | The history is a place you can stand in, not a stack you pop (§18.2.4) |
| 1 | a panel closed | the command rail | Nothing is lost: the Panels menu lists all eight, and what you leave open is remembered (§11) |
| 10 | an undo | the canvas | Draw a rough line or ellipse and *hold* — it snaps to what you meant, and the drag steers it (§6.9) |
| 3 | a shape-assisted stroke | Drawing Guides panel | Straight is one thing; straight *to somewhere* is another — add a perspective guide (§20) |
| 1 | an assisted **line** with a guide visible | Drawing Guides panel | The grid aims held lines down its own axes, and turns a held circle into one in perspective (§20.6, §20.7) |
| 20 | a brush stroke | Select panel | Drag a marquee and every tool acts only inside it; the chips combine one selection with the last (§6.8) |
| 3 | a selection | Layers panel | A selection says *where*, the stack says *what* — and a group is also a clipping mask (§14) |
| 1 | the brush editor opened | the editor itself | **A series of five**, walked through with Next: the test stroke, then Tip, Paint, Color dynamics, Pickup |

**Order decides ties**, and the first three all wait on a stroke. Listed in that
order, so a stroke satisfying more than one gives the earliest still owed — which
is also what brings a card passed over while another was up back before the ones
behind it, rather than letting a busy stretch reorder the tour into whatever the
artist happened to do next.

The first three are also where the tour carries weight it did not used to: every
panel now starts closed (§11), so the opening screen is the painting alone, and the
sequence *color → where the panels went → the brush* is how the stack gets
assembled for somebody who has not found the Panels menu. That is the trade the
empty start buys — the panels arrive one at a time, each with a reason, instead of
five at once with none.

The counts are set from what each deed *costs* to keep doing the hard way. Two
strokes is no commitment at all, but a painter with no color picker has already
wanted one. Ten trips to the size slider is somebody who has decided that this is
how they work, which is exactly the moment the drag is worth knowing and well past
the moment it would have been an interruption. Ten undos is the same argument in a
different key: it is not a request for anything, it is somebody visibly not getting
the line they wanted, which is the one moment "draw it roughly and hold" reads as
help rather than as trivia.

The assist row is the one card shown **on the painting** rather than beside a
control, and the placement is the message: what it describes is a thing you do with
the pen, in the middle of the canvas, and there is no control anywhere that it could
have pointed at instead.

The last four rows are a **chain**, and it is the part of the table worth reading as
a sequence. Undos bring the shape assist; assisted strokes bring the perspective
guides, because a held line is exactly the stroke a grid has something to say about;
and a held line drawn with a guide on screen brings the fact that the two have
already been wired together (§20.6). Each lesson is the reason the next one's deed
starts happening, so the tour walks somebody from *my lines are wobbly* to *my lines
are on the vanishing point* without ever telling them anything they had not just
asked for.

#### The series

The brush editor is the one lesson that is not a lesson but five, and it is the only
place the tour ever says more than a sentence' worth in a row. The dialog earns it:
it is a wall of parameters, it is entered deliberately, and nothing else in Stark
has as much to say about *why* its knobs are grouped the way they are. Walking it —
the test stroke first, because tuning by looking is the whole point, then Tip,
Paint, Color dynamics, Pickup in the order they are laid out — turns that wall into
a reading order.

Each card anchors to its own section (`brush_editor::BrushPart`, written into the
markup as `data-be` and read back through the same function, so a section renamed on
screen keeps its anchor and one deleted stops compiling on both sides at once). The
section cards stand to the right, over the preview column; the preview's own card
stands to the left, over the sections. Each covers the half it is not talking about.

Collapsed sections still anchor, because a `Section` renders its header whether or
not it is open — so Color dynamics and Pickup, which start folded, are pointed at
where they sit rather than where they would be if expanded.

#### One honest limit

The undo count is **undos**, not stroke-undos. Nothing in the
projection says what the next undo would remove, and asking would mean a new method
on `Timeline` with two implementors — a lot of engine surface for a threshold. In a
painting app almost every undo is a stroke's, and the ones that are not point in the
same direction anyway.

"By a control" in two of those rows is `not_reaching` doing its work: the deed is
*reaching for the slider*, not *the size changing*, so the artist who already drags
never accrues it. The preset row is the same idea from the other side — the deed is
the row *click*, which the quick slots never produce, so somebody already fluent
with the number keys is never offered the lesson about them.

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
- **No lesson without an anchor.** Every lesson points at a box that exists — even
  the invisible one, which is a real box with a real gesture attached (§24.3). A
  card floating in the middle of the window with nothing to explain would be an
  announcement, and this feature is not an announcement channel.
