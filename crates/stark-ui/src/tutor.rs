//! The guided tour: short lessons that arrive once the artist has earned them
//! (§24).
//!
//! Stark's chrome differs from the apps people come to it from, and the
//! difference is mostly in what it *does not* have — no toolbar of forty tools, no
//! modal dialog per operation, several of the best bindings living on a modifier
//! rather than on a button. That is the design working, and it is also the
//! standing cost of the design: a control that is not on screen is a control
//! nobody finds. The eyedropper's options bar already answers this for one binding
//! by coming up on Alt (`crate::panels::pick`); this module is the same answer
//! generalized to the things no modifier announces.
//!
//! # A lesson is owed, not scheduled
//!
//! Nothing here runs at first launch and nothing takes the canvas away. A lesson
//! is attached to a **deed** — a thing the user has done — and a **count**: paint
//! three strokes and the Brush panel explains itself; reach for the size slider ten
//! times and the tour points out that the same knob is under the hand on the
//! canvas. So a tip only ever arrives to somebody who has already demonstrated
//! that they care about the thing it is about, which is the whole of why this is
//! not a tutorial mode with a Skip button.
//!
//! Each lesson is given **once, ever** — the ledger follows this browser the way
//! the shape and preset libraries do (`crate::storage`) — and the tally keeps
//! running across visits, because "the third stroke" is not a claim about one
//! session.
//!
//! # And answered by doing the thing
//!
//! A card is a claim that somebody does not know something, so **doing the thing is
//! the answer** — a better one than the button, because it is the skill rather than a
//! statement about it. Every lesson names the deeds that answer it ([`Answer`]), and
//! they are rarely the deed that earned it: the color panel is earned by *painting*
//! and answered by *picking a color*.
//!
//! It reaches back as well as forward, for some. Picking a color is deliberate, so
//! somebody who has already done it is never shown the card at all
//! ([`Answer::Known`]); waking the panels is not — the slice is the whole right-hand
//! edge of the window and a pointer on its way anywhere crosses it — so that one
//! counts only from the moment its card is up ([`Answer::Doing`]). The test is not
//! whether the deed is related but whether **having done it proves they found the
//! thing the card points at**.
//!
//! And a good many lessons have no answer but the button, which is the same design
//! working from the other end: the gestures they teach are bracketed by
//! [`not_reaching`] and produce no deed at all, so somebody already fluent with one
//! never earns its card in the first place. There is nothing left for an answer to
//! catch.
//!
//! # Deeds are read off the command stream, not off the call sites
//!
//! [`observe`] is called from [`dispatch`](crate::state::dispatch), which is the
//! single seam every mutation this user makes goes through (§4). Three things fall
//! out of reading deeds *there* rather than tapping the handlers that produce them:
//!
//! - **A new way to do a thing is counted for free.** Size is reachable from the
//!   Brush panel's slider, from the tuning drag, from the quick-brush rack and from
//!   a preset; none of them knows this module exists, and a fifth way added later
//!   will not have to.
//! - **A collaborator's work is not counted as yours.** Remote actions reach the
//!   engine through [`with_engine`](crate::state::with_engine), never through
//!   `dispatch` (`crate::collab`), so the split that already existed for
//!   broadcasting is the split this needs.
//! - **What a command *means* is decided by what moved**, not by who sent it. A
//!   `SetBrush` is a size change, a color change or neither depending on which
//!   fields differ from the brush the engine is still holding — see [`brush_deed`],
//!   which is what keeps a preset click (every field at once) from being read as an
//!   adjustment of anything.
//!
//! The command stream cannot say two things, and both are the frontend's in the
//! sense [`GestureCommand::Hold`] is (§6.9) — facts about a hand, which the engine
//! has neither the clock nor the pointer to know.
//!
//! The first is **when a run of commands is one act**. A slider drag is one
//! intention and sixty `SetBrush`es; [`COALESCE`] is the whole of the answer.
//!
//! The second is **what wrote the brush**, and it matters because three of the
//! lessons teach ways of changing it that produce the very deed they are counted
//! by: an Alt-drag off the painting is a color change, an accelerator drag is a size
//! change, and a quick slot is a whole tool arriving exactly as a preset click is.
//! Counted naively, the tour would wait for somebody to use the eyedropper five
//! times and then offer to explain the eyedropper. So those writes are bracketed by
//! the code that makes them ([`not_reaching`]) and are not counted — which turns the
//! flaw into the feature the tour most wanted: **it never teaches a gesture you
//! already use.** Somebody who only ever eyedroppers never accumulates the deed and
//! is never told about it.
//!
//! One deed is not in the stream at all and is reported outright ([`did`]): clicking
//! a row in the preset library. The command it leads to says a brush changed and
//! cannot say that a row was clicked — and the quick slots, which that lesson goes
//! on to teach, emit a command of exactly the same shape.
//!
//! # What the card is, and where it is allowed to appear
//!
//! One card at a time, anchored beside the thing it describes and pointing at it.
//! It is ordinary floating chrome: it wears
//! [`chrome_class`](crate::layout::chrome_class), so it fades out mid-stroke and
//! back when the hand lifts like every other box over the painting, and it never
//! takes a press meant for the canvas.
//!
//! Coming *due* and being *shown* are deliberately two steps. A lesson opens the
//! panel it is about, and a panel opened mid-stroke is put straight back to sleep
//! by the release ([`end_interaction`](crate::input::end_interaction)) — so a
//! lesson that comes due with the canvas in hand waits in [`TutorState::due`]
//! until it is not.

use std::collections::HashSet;

use dioxus::prelude::*;

use crate::brush_editor::BrushPart;
use crate::icons::{self, icon};
use crate::layout::{ChromeHiding, PanelId, PanelLayout, chrome_class, open_panel, panel_key};
use crate::platform::{self, ElementBox};
use crate::state::AppState;
use crate::storage::Store;
use stark_engine::InputCommand;
use stark_engine::command::{DocCommand, GestureCommand, ViewCommand};
use stark_model::document::BrushParams;

/// How long a gap between two reports of the same deed makes them two deeds, in
/// seconds.
///
/// The number the whole reader turns on: a slider drag arrives as sixty
/// `SetBrush`es a second and is one act, while two deliberate adjustments a beat
/// apart are two. Half a second is comfortably longer than any gap *inside* a drag
/// (a 60 Hz pointer reports every 16 ms) and comfortably shorter than the pause
/// between deciding to change something twice.
///
/// Measured from the **last** report of a run rather than from its first, so a slow
/// drag stays one deed however long the hand takes over it.
const COALESCE: f64 = 0.5;

/// How far one run of panning has to travel to count as a long one, in page px.
///
/// About a screen's width, which is the point at which reaching for the Navigator
/// would have been quicker than the drag that was made instead — and that is the
/// claim the lesson it feeds makes, so it is the figure to state it in. Runs, not
/// gestures: a hand that pans, pauses and pans again inside [`COALESCE`] is
/// travelling once, and a two-finger pan on a tablet counts the same as a
/// space-drag because both arrive here as view commands rather than as gestures.
const LONG_PAN: f32 = 1200.0;

/// How far the card sits from the thing it points at, in CSS px — the arrow lives
/// in this gap.
const GAP: f32 = 14.0;

/// How close to the window's edge a card may come, in CSS px. The panel column's own
/// inset, so a card beside the stack stops where the stack does.
const EDGE: f32 = 14.0;

/// A thing the user has done, at the grain a person would describe it in.
///
/// Deliberately not one variant per command: several commands are one deed (`Pan`,
/// `Pinch`), and one command is two deeds or none depending on what moved
/// ([`brush_deed`]). The vocabulary here is the *lessons'* — a deed exists because
/// a lesson **counts it or answers to it**, and one that neither would be a tally
/// kept for nobody (which is what `the_deeds_and_the_lessons_account_for_each_other`
/// refuses).
///
/// Each variant carries the name it is **stored** under as a `rename`, spelled out
/// rather than derived from the identifier on purpose: the name is what a browser
/// wrote last week, so a deed may be added, removed or reordered freely, and only
/// editing the string forgets a tally. A name this build no longer knows costs its own
/// row (`storage::load_list`) — a deed nothing counts has no lesson to feed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Deed {
    /// A brush stroke laid on the canvas and committed. A selection drag is not a
    /// stroke: it shares the gesture and produces a mask rather than paint (§6.8).
    #[serde(rename = "stroke")]
    Stroke,
    /// The brush's size or flow moved, however it was moved.
    #[serde(rename = "tune")]
    TunedBrush,
    /// One run of panning crossed [`LONG_PAN`].
    #[serde(rename = "long-pan")]
    LongPan,
    /// The brush's color moved, however it was moved.
    #[serde(rename = "color")]
    ChangedColor,
    /// A redo — the user's own, not the timeline transport's.
    #[serde(rename = "redo")]
    Redo,
    /// A preset put on from the Brush panel's library. Not on the command stream —
    /// reported by the row that was clicked ([`did`]).
    #[serde(rename = "preset")]
    AppliedPreset,
    /// An undo — the user's own, not the timeline transport's.
    #[serde(rename = "undo")]
    Undo,
    /// A panel closed. Not on the command stream either: which panels are open is
    /// the frontend's alone, so `layout::close_panel` reports it ([`did`]).
    #[serde(rename = "closed-panel")]
    ClosedPanel,
    /// A stroke that snapped to a line or an ellipse (§6.9).
    ///
    /// A [`Stroke`](Self::Stroke) as well, always — the two are counted together and
    /// neither is the other's expense. It is only knowable by asking the engine what
    /// the hold found ([`Engine::assisted`](stark_engine::Engine::assisted)), and
    /// only before the gesture ends.
    #[serde(rename = "assisted")]
    AssistedStroke,
    /// A stroke that snapped to a **line** while a perspective guide was on screen
    /// (§20.6) — the state in which the grid is about to start aiming strokes,
    /// whether or not this one landed near enough for it to.
    #[serde(rename = "guided-line")]
    GuidedLine,
    /// A marquee or lasso gesture that committed a **selection** (§6.8).
    ///
    /// The same gesture as a stroke, told apart by the tool and by what the panel's
    /// action says it builds: under Fill the drag lays paint instead, which is not
    /// a selection however much it looks like one from here (§18.0.4).
    #[serde(rename = "selection")]
    Selection,
    /// The brush editor opened. Not on the command stream: the dialog is frontend
    /// state and reaches no engine, so the button reports it ([`did`]).
    #[serde(rename = "brush-editor")]
    OpenedBrushEditor,
    /// The panel column reached into, bringing the sleeping stack back (§11).
    ///
    /// Not on the command stream, for [`ClosedPanel`](Self::ClosedPanel)'s reason, and
    /// reported by the slice the pointer lands in (`layout::PanelStack`) rather than by
    /// `layout::wake_panels` — which `open_panel` and the tour's own `release_panels`
    /// also go through. Neither of those is the artist reaching for anything, and the
    /// second is a card waking the stack as it comes down: a lesson answered by its own
    /// dismissal would be no lesson at all.
    #[serde(rename = "woke-panels")]
    WokePanels,
    /// A color sampled off the painting with the eyedropper (§18.0.2).
    ///
    /// Reported outright (`input::pick_color`) and **not** read as the
    /// [`ChangedColor`](Self::ChangedColor) it also is: the write is inside a
    /// [`not_reaching`] bracket, which is the whole of how the tour avoids waiting for
    /// somebody to eyedropper ten times and then offering to explain the eyedropper.
    /// This is the same act counted from the other side — as the gesture rather than as
    /// its effect on the brush.
    ///
    /// Only where a sample actually landed. A pick over bare canvas answers nothing and
    /// leaves the brush as it was, and counting it would be counting a colour nobody
    /// got. A drag is one deed, as every drag here is ([`COALESCE`]).
    #[serde(rename = "picked-color")]
    PickedColor,
    /// A layer added to the stack (§14).
    ///
    /// On the stream, and the one deed there that is a *document* command rather than
    /// a view or a gesture: [`DocCommand::AddLayer`] is minted by the Layers panel's
    /// button and by the command that shares it, and by nothing else — an image
    /// arriving carries its own command (§23) and a matte carries another (§15), so
    /// neither is read as somebody having discovered the stack.
    #[serde(rename = "added-layer")]
    AddedLayer,
    /// The navigator's miniature clicked or dragged (§11).
    ///
    /// On the stream after all, which is worth saying because the deed is a *gesture on
    /// a piece of chrome*: [`ViewCommand::CenterOn`] is absolute where every other way
    /// of travelling is a delta, so the miniature is the only thing in the app that
    /// emits one. That is also why a navigator drag is not counted as travel, which it
    /// must not be — the lesson this answers is the one about panning too far.
    #[serde(rename = "navigator")]
    UsedNavigator,
}

impl Deed {
    /// Every deed. The order **is** the tally's slot order, so this is
    /// [`Deed::slot`]'s only authority.
    const ALL: [Deed; 16] = [
        Deed::Stroke,
        Deed::TunedBrush,
        Deed::LongPan,
        Deed::ChangedColor,
        Deed::Redo,
        Deed::AppliedPreset,
        Deed::Undo,
        Deed::ClosedPanel,
        Deed::AssistedStroke,
        Deed::GuidedLine,
        Deed::Selection,
        Deed::OpenedBrushEditor,
        Deed::WokePanels,
        Deed::PickedColor,
        Deed::AddedLayer,
        Deed::UsedNavigator,
    ];

    /// How many there are, so the tally can be an array rather than a map.
    const COUNT: usize = Deed::ALL.len();

    /// This deed's slot in the tally.
    ///
    /// A linear scan of a dozen, at deed rate. It reads [`Deed::ALL`] rather than
    /// restating the order as a `match` that could come to disagree with it.
    fn slot(self) -> usize {
        Deed::ALL
            .iter()
            .position(|d| *d == self)
            .expect("every Deed is in Deed::ALL")
    }
}

/// What the tour has to remember between visits: how often each deed has been
/// done, and which lessons have been given.
///
/// The two halves forget differently, on purpose. A tally under a name this build
/// does not know is **dropped**, because a deed no longer counted has no lesson to
/// feed; a *lesson* name is **kept** whatever the table now says, because somebody
/// who has seen a tip should not be shown it again by a release that happened to
/// rename its neighbour.
#[derive(Clone, Default, PartialEq)]
pub struct Ledger {
    /// How many times each deed has been done, by [`Deed::slot`].
    tally: [u32; Deed::COUNT],
    /// The keys of the lessons already given.
    given: HashSet<String>,
}

/// The reader's working memory: what the last half-second was, which is what turns
/// a stream of commands into a count of acts.
///
/// Not persisted, and it would be wrong to persist it — every field here is a
/// claim about a gesture that is over by the time the page closes.
#[derive(Clone, Copy, PartialEq)]
pub struct Recent {
    /// When each deed was last counted, on [`platform::now_seconds`]'s clock.
    /// Negative infinity for one never counted, so the first report of any deed is
    /// always far enough from the last.
    at: [f64; Deed::COUNT],
    /// How far the run of panning in flight has travelled, page px.
    pan: f32,
    /// Whether that run has already been counted as a long one — so a pan that
    /// keeps going scores once rather than once per sample past the line.
    long: bool,
}

impl Default for Recent {
    fn default() -> Self {
        Self {
            at: [f64::NEG_INFINITY; Deed::COUNT],
            pan: 0.0,
            long: false,
        }
    }
}

/// Everything the tour holds, root-owned like the rest of
/// [`AppState`](crate::state::AppState).
#[derive(Clone, Copy)]
pub struct TutorState {
    /// The durable half: the tally and the lessons already given. Loaded by
    /// [`begin`].
    pub ledger: Signal<Ledger>,
    /// The half that is only about the last half-second — see [`Recent`].
    pub recent: Signal<Recent>,
    /// A lesson that has come due and is waiting for the canvas to be free, by its
    /// index in [`LESSONS`].
    ///
    /// Separate from [`showing`](Self::showing) because *revealing* a lesson opens
    /// a panel, and a panel opened mid-stroke is put straight back to sleep by the
    /// release — see the module comment.
    pub due: Signal<Option<usize>>,
    /// The lesson on screen, by its index in [`LESSONS`].
    pub showing: Signal<Option<usize>>,
    /// Whether the tour is listening at all. False until [`begin`], so the commands
    /// app startup makes on the user's behalf — the opening brush, the opening
    /// color — are not read as things the user did.
    pub armed: Signal<bool>,
    /// Whether tips are wanted (⚙ → Tips). Deeds are tallied either way, so turning
    /// them back on resumes rather than restarts.
    pub enabled: Signal<bool>,
    /// Bumped to make the card measure its anchor again — a window resize.
    pub epoch: Signal<u64>,
    /// How many open brackets say the brush is being written by something other
    /// than the artist reaching for a control — see [`not_reaching`], which is the
    /// only thing that writes it.
    pub not_reaching: Signal<u32>,
}

/// What a lesson points at. Each variant arrives with the lesson that needs it
/// rather than in advance of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Anchor {
    /// A floating tool panel, by the identity it wears in the DOM ([`panel_key`]).
    Panel(PanelId),
    /// The full-height slice down the right that the panels are woken by reaching
    /// into (§11) — the one anchor that is **invisible**, and the only one where
    /// that is the point: the lesson is about a piece of the window with nothing
    /// drawn in it, so the card is the only thing that can say where it is.
    PanelColumn,
    /// The quick-brush rack down the left (§18.1.8).
    QuickSlots,
    /// The navigator's miniature, in the bottom-left corner (§11) — the
    /// other half of that column, and the other thing the Panels menu shows
    /// and hides that is not a panel.
    Navigator,
    /// The command rail in the top-left corner (§11) — the command search, the
    /// Panels menu and the ⚙. Always on screen, so nothing has to reveal it and
    /// nothing can close it.
    CommandRail,
    /// A part of the brush editor's dialog ([`BrushPart`]).
    ///
    /// The only anchor **inside a dialog**, which is the one thing that makes it
    /// different from all the others: every other card stands down while a modal is
    /// up, because a modal is over everything a card could point at. These are the
    /// exception by construction — the dialog *is* what they point at — so they show
    /// while it is open and only while it is open.
    BrushEditor(BrushPart),
    /// The painting itself.
    ///
    /// Not a control, and that is what it is for: a lesson about a gesture made
    /// **on the canvas** has nothing to stand beside, and standing it beside a
    /// panel would say the panel had something to do with it. Pointed at from
    /// [`Side::Inside`], which is the only placement that goes *over* its anchor
    /// rather than next to it.
    Canvas,
    /// Timeline mode's bar across the foot of the window (§18.2.4).
    TimelineBar,
}

impl Anchor {
    /// The selector that finds this anchor's element.
    ///
    /// Built from [`panel_key`] rather than written out, so a panel and the card
    /// pointing at it cannot come to disagree about what a panel is called in the
    /// DOM — a disagreement that shows as a card in the corner of the window with
    /// nothing beside it to explain.
    fn selector(self) -> String {
        match self {
            Anchor::Panel(id) => {
                format!(".panel-stack > .panel[data-panel=\"{}\"]", panel_key(id))
            }
            Anchor::PanelColumn => ".panel-wake".to_string(),
            Anchor::QuickSlots => ".slot-overlay".to_string(),
            Anchor::Navigator => ".navigator-overlay".to_string(),
            Anchor::CommandRail => ".command-rail".to_string(),
            // By the id the app already gives it, rather than by its class: the
            // canvas is named once (`render::CANVAS_ID`) and this is that name,
            // so there is no second spelling to fall out of step.
            Anchor::Canvas => format!("#{}", crate::render::CANVAS_ID),
            // From the editor's own naming, so a section renamed on screen keeps
            // its anchor and a section deleted stops compiling on both sides at
            // once (`brush_editor::BrushPart`).
            Anchor::BrushEditor(part) => format!("[data-be=\"{}\"]", part.key()),
            Anchor::TimelineBar => ".timeline-bar".to_string(),
        }
    }

    /// Whether this anchor is still *meant* to be on screen.
    ///
    /// Deliberately not the same question as "did it measure". A measurement comes
    /// back `None` for a frame while the browser lays out a panel that has only
    /// just opened, and dismissing a lesson on that would be dismissing every
    /// lesson the moment it appeared. This asks the app's own state instead, so the
    /// one thing it answers `false` to is the user having **closed** the thing the
    /// card is about — which is an answer to the card, and is taken as one.
    fn on_screen(self, state: AppState, layout: PanelLayout) -> bool {
        match self {
            Anchor::Panel(id) => !layout.hidden.read().contains(&id),
            // Asleep, with something to wake. **The wake is the dismissal** — reach
            // into the column and the panels come back, which is the lesson done and
            // is a better acknowledgement than the button. Deliberately *not* also
            // testing `canvas_active`, which is the third thing the slice itself
            // wants: painting hides the card for the length of the stroke (the
            // measurement finds nothing) and must not end the lesson, since starting
            // a stroke is not an answer to it.
            Anchor::PanelColumn => {
                let asleep = (state.panels_asleep)();
                let hidden = layout.hidden.read();
                asleep && PanelId::ALL.iter().any(|id| !hidden.contains(id))
            }
            Anchor::QuickSlots => (state.slots.pinned)(),
            Anchor::Navigator => (state.navigator)(),
            // Always. Both are mounted for the life of the page and neither has a
            // control that puts it away, so these lessons are dismissed the ordinary
            // way and by nothing else.
            Anchor::CommandRail | Anchor::Canvas => true,
            // Exactly as long as the dialog is up. Closing it mid-series is an
            // answer to the card on screen and leaves the rest of the series owed
            // for the next time it is opened (`abandon`).
            Anchor::BrushEditor(_) => (state.brush_editor_open)(),
            Anchor::TimelineBar => (state.timeline.open)(),
        }
    }

    /// Whether this anchor lives inside a dialog.
    ///
    /// The one thing the promotion rule has to ask about an anchor, and the reason
    /// it is a method here rather than a condition there: "a dialog is over
    /// everything a card could point at" stops being true exactly when the card is
    /// pointing at the dialog.
    fn inside_dialog(self) -> bool {
        matches!(self, Anchor::BrushEditor(_))
    }

    /// The window edge this anchor's box is up against — the one there is no room
    /// beside, so a card that grew that way would grow off the screen. `None` for the
    /// anchors clear of both: the panel column and the canvas run the height of the
    /// window, the rack takes only what the miniature under it leaves, and a dialog
    /// is centred in what is left.
    ///
    /// Asked by [`tests::a_card_is_never_hung_into_the_edge_it_stands_on`] and by
    /// nothing at runtime, which is why it is compiled only for the test: a placement
    /// is *stated* on the lesson ([`Side`]) rather than derived, so this exists to
    /// check that statement and not to make it.
    #[cfg(test)]
    fn against(self) -> Option<Edge> {
        match self {
            // The stack hangs from the top of its column, which is the same edge
            // `Side::LeftAtTop` lines a card up with.
            Anchor::Panel(_) | Anchor::CommandRail => Some(Edge::Top),
            // Both sit on the foot of the window (`.left-chrome`, `.timeline-bar`).
            Anchor::Navigator | Anchor::TimelineBar => Some(Edge::Bottom),
            // The column and the canvas are the full height of the window, the rack
            // is what the navigator under it leaves, and a dialog is centred.
            Anchor::PanelColumn | Anchor::QuickSlots | Anchor::Canvas => None,
            Anchor::BrushEditor(_) => None,
        }
    }

    /// Put this anchor on screen, so there is something for the card to point at.
    ///
    /// Derived from the anchor rather than named separately on the lesson: "what it
    /// points at" and "what it opens" are one fact, and a lesson able to state them
    /// differently is a lesson that could open the Color panel and point at the
    /// Brush one.
    fn reveal(self, state: AppState, layout: PanelLayout) {
        match self {
            Anchor::Panel(id) => open_panel(state, layout, id),
            // The slice exists only while the panels are standing down, so what
            // "reveal" means here is to put them there. A no-op in practice — a
            // lesson is promoted with the canvas out of hand, and the release that
            // freed it is what set this — but written out rather than relied upon,
            // so the card cannot be shown pointing at a box that is not in the DOM.
            Anchor::PanelColumn => crate::layout::sleep_panels(state),
            Anchor::QuickSlots => {
                let mut pinned = state.slots.pinned;
                pinned.set(true);
            }
            Anchor::Navigator => crate::navigator::set_open(state, true),
            // Nothing to do: both are always there. Arms that say so, rather than a
            // catch-all, so a variant added later has to decide.
            // Nor here, and this one is a decision rather than a fact: the deed
            // that brings these due is the dialog being opened, so it is already
            // up — and opening it *for* somebody would be the tour taking the
            // screen, which nothing else here does.
            Anchor::CommandRail | Anchor::Canvas | Anchor::BrushEditor(_) => {}
            Anchor::TimelineBar => crate::panels::timeline::set_open(state, true),
        }
    }
}

/// Where the card sits relative to its anchor.
///
/// The DOM does not say this and cannot be asked: which side has room is a fact
/// about where that chrome lives in the window, and so is whether the anchor has a
/// meaningful top edge to line up with. Both are the lesson author's to know, which
/// is why this is a field on the lesson and not something measured.
///
/// Named for the picture rather than composed out of a side and an alignment: each
/// variant is exactly one placement the lessons use, where two enums would also
/// spell several that mean nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    /// Left of the anchor, their top edges level. For a box that hangs from the top
    /// of the window — a panel in the stack. Level rather than centred so a tall
    /// panel cannot push the card off the top: the stack hangs from the top of its
    /// column, so that is the edge always on screen.
    LeftAtTop,
    /// Left of the anchor, centred on it. For an anchor that is a whole edge of the
    /// window and so has no meaningful top — the panel column.
    LeftAtMiddle,
    /// Right of the anchor, their top edges level. For a box down the left that
    /// hugs its contents — the command rail.
    RightAtTop,
    /// Right of the anchor, centred on it, for the chrome down the left.
    ///
    /// The mirror of [`LeftAtMiddle`](Self::LeftAtMiddle) rather than of
    /// [`LeftAtTop`](Self::LeftAtTop), and the quick-brush rack is why: its box runs
    /// from under the command rail to the foot of the window and centres its rows in
    /// that, so the *box's* top edge is a long way above anything drawn. Lining up
    /// with it would put the card level with nothing.
    RightAtMiddle,
    /// Right of the anchor, their **bottom** edges level. For a box that sits on the
    /// foot of the window — the navigator's miniature in the corner.
    ///
    /// [`LeftAtTop`](Self::LeftAtTop)'s argument read the other way up: a card hangs
    /// from whichever of the anchor's edges the window cannot push off screen, and
    /// for a box pinned to the bottom that is the bottom one. Centring on the middle
    /// instead — which is what the navigator's lesson did when the miniature left the
    /// panel stack — hangs half a card below the anchor's middle, and the corner has
    /// no room for it: the miniature stands [`EDGE`] off the foot of the window, so
    /// everything below its middle is over the edge. Nothing catches that later,
    /// either. Sideways a card narrows to fit ([`room_left`]); there is no vertical
    /// equivalent, because a card narrowed to fit its width is a card that has grown
    /// *taller*. Which is why the placement is the whole of the fix, and why
    /// [`tests::a_card_is_never_hung_into_the_edge_it_stands_on`] is the guard.
    ///
    /// The arrow cannot be a constant here the way it is for every other placement:
    /// the miniature's height is the artwork's own aspect fitted into a box
    /// (`navigator::MAX_HEIGHT`), so it is anything from a sliver to the full 200px.
    /// What goes to the stylesheet instead is how far above that shared bottom edge
    /// the anchor's middle lies (`--tutor-reach`), and the arrow rides there.
    RightAtBottom,
    /// Above the anchor, centred on it.
    Above,
    /// **Over** the anchor rather than beside it: centred across it,
    /// [`INSIDE_DEPTH`] of the way down from its top, pointing down into it.
    ///
    /// The odd one out, and it earns it. Every other placement puts the card next
    /// to a control and says *this thing here*; this one is for an anchor that is
    /// not a control but a **place** — the canvas, where a gesture happens and
    /// where standing beside it would mean standing beside the window. A card in
    /// the middle of the painting, pointing down at it, is the only arrangement
    /// that says "the thing I am describing happens *there*".
    Inside,
}

/// An edge of the window, for the one question asked about both of them: which one a
/// piece of chrome sits against, and which one a card grows toward.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Edge {
    Top,
    Bottom,
}

#[cfg(test)]
impl Side {
    /// Which edge of the window the card reaches toward from the point it is hung
    /// at. `None` where it reaches both ways at once, which is every placement
    /// centred on its anchor — and which is why a centred card needs an anchor clear
    /// of both edges.
    fn grows(self) -> Option<Edge> {
        match self {
            // Hung level with the anchor's top and hanging down from it — and
            // `Inside` starts lower still.
            Side::LeftAtTop | Side::RightAtTop | Side::Inside => Some(Edge::Bottom),
            // Hung level with the anchor's bottom, or clear above it, and rising.
            Side::RightAtBottom | Side::Above => Some(Edge::Top),
            Side::LeftAtMiddle | Side::RightAtMiddle => None,
        }
    }
}

/// How far down its anchor a [`Side::Inside`] card sits, as a fraction of the
/// anchor's height.
///
/// Far enough down to leave the arrow pointing into the middle of the picture
/// rather than at its top edge, and not so far that the card is over the middle of
/// the work. A quarter is also about where a hand starts a stroke it means to drag
/// downward, which is the gesture the one lesson placed this way is about.
const INSIDE_DEPTH: f32 = 0.25;

/// What answers a lesson besides its own button: the deeds that **are** the thing it
/// teaches (§24.3).
///
/// A card is a claim that the artist does not know something. Doing the thing is that
/// claim being answered, and answering it any other way — leaving the card up until it
/// is acknowledged — is the tour asking somebody to press a button about a skill they
/// have just demonstrated. That is the whole of what this is for.
///
/// **Stated on the lesson rather than derived from its deed**, because the two are
/// different questions and usually different deeds: what *earns* a lesson is evidence
/// the artist wants the thing, and what *answers* it is evidence they already have it.
/// The color panel is earned by painting and answered by picking a color.
///
/// The two live variants differ only in **how far back an answer counts**, and the
/// distinction is not a nicety — it is the difference between a tip that helps and one
/// nobody ever sees:
///
/// - Picking a color is deliberate. Somebody who has done it once has found the picker,
///   so the lesson about it is not owed at all — [`Known`](Self::Known).
/// - Waking the panels is not: the slice is the whole right-hand edge of the window, so
///   a pointer on its way anywhere crosses it and the stack comes back without anybody
///   deciding anything. Counted backwards it would foreclose its own lesson for
///   practically every user — [`Doing`](Self::Doing).
///
/// So the test for [`Known`] is not "is the deed related" but **would having done it
/// prove they found the thing the card points at**.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Answer {
    /// The button, and nothing else.
    ///
    /// Written out rather than left as an empty list, so a lesson added later has to
    /// decide — and so this one says *why* there is nothing here. For most of the rows
    /// that wear it the reason is the same and is a feature: the gesture the lesson
    /// teaches is bracketed by [`not_reaching`] and produces no deed at all, which is
    /// what keeps the tour from ever teaching a gesture somebody already uses. There
    /// is nothing to answer with because there is nothing to count.
    Button,
    /// These deeds, from the moment the card is on screen.
    ///
    /// Doing one takes it down and marks it given, exactly as "Got it" would. One done
    /// *before* the lesson came due does not count — see the type's comment.
    Doing(&'static [Deed]),
    /// These deeds, ever.
    ///
    /// [`Doing`](Self::Doing) plus the backward half: a count already standing when the
    /// lesson would come due means it is **never offered**. Skipped rather than written
    /// into the ledger, for [`Lesson::applies`]'s reason — it steps aside and lets the
    /// next lesson its deed owes come forward, and it is owed again if the tally it was
    /// foreclosed by ever turns out to have been somebody else's browser.
    Known(&'static [Deed]),
}

impl Answer {
    /// The deeds that take a card **on screen** down. Both live variants; the
    /// difference between them is entirely in [`forecloses`](Self::forecloses).
    fn dismisses(self) -> &'static [Deed] {
        match self {
            Answer::Button => &[],
            Answer::Doing(deeds) | Answer::Known(deeds) => deeds,
        }
    }

    /// The deeds that, already done, mean the lesson is never offered at all.
    fn forecloses(self) -> &'static [Deed] {
        match self {
            Answer::Button | Answer::Doing(_) => &[],
            Answer::Known(deeds) => deeds,
        }
    }
}

/// One lesson: the deed it waits for, how many it waits for, what answers it, and what
/// it says.
struct Lesson {
    /// What the ledger records this lesson as. **Stable across edits of
    /// [`LESSONS`]** — an index would move when a lesson is inserted above it, and
    /// somebody would be shown a tip they had already dismissed.
    key: &'static str,
    /// The deed that brings it due.
    deed: Deed,
    /// How many of them. Read as "the Nth time".
    after: u32,
    /// What the artist can do to say they already know this — see [`Answer`].
    answer: Answer,
    /// What it points at — and, through [`Anchor::reveal`], what it opens.
    anchor: Anchor,
    /// Which side of the anchor it sits on.
    side: Side,
    /// The one line that says what the lesson is about.
    title: &'static str,
    /// The lesson itself. Says what to *do*, and then the thing about Stark that
    /// makes it worth doing — a tip that only names a shortcut is a keyboard
    /// reference, and there is one of those in the menus already.
    body: &'static str,
}

impl Lesson {
    /// Whether this lesson's subject exists for this browser at all.
    ///
    /// Derived from the anchor, for [`Anchor::reveal`]'s reason: what a lesson points
    /// at and what has to be true for it to point are one fact. Only one anchor has
    /// ever been able to answer no — the panel column's wake slice is a *setting* now
    /// (`layout::ChromeHiding`), and a tip explaining a gesture somebody has switched
    /// off is worse than no tip at all.
    ///
    /// Asked in [`due`] rather than at the card, so an inapplicable lesson does not
    /// merely fail to draw: it steps aside and lets the next one its deed owes come
    /// forward. A lesson that could not be shown *and* could not be passed would stall
    /// its whole deed — the stroke owes four — and it would stall it silently.
    fn applies(&self, chrome: ChromeHiding) -> bool {
        match self.anchor {
            Anchor::PanelColumn => chrome.sleeps(),
            _ => true,
        }
    }

    /// Whether the artist has **already** shown they know what this says
    /// ([`Answer::Known`]) — in which case it is not owed at all.
    ///
    /// Asked in [`due`] beside [`applies`](Self::applies) and for the same reason: a
    /// lesson that cannot be shown has to step aside rather than sit at the head of its
    /// deed's queue, or it stalls every lesson behind it silently. The difference is
    /// what they are about — one is a lesson whose *subject* is switched off, this is
    /// one whose subject is already understood.
    ///
    /// A tally rather than the `given` set, because the question is what was done and
    /// not what was shown.
    fn foreclosed(&self, book: &Ledger) -> bool {
        self.answer
            .forecloses()
            .iter()
            .any(|deed| book.tally[deed.slot()] > 0)
    }
}

/// Every lesson, in the order they are offered when two come due at once.
///
/// The table is the whole feature: a lesson is a row here, and adding one costs no
/// code anywhere else unless it counts a deed nothing counts yet.
///
/// **Order decides ties**, and three lessons wait on the same deed: the second
/// stroke brings the color, the third says where the panels went, and the fifth
/// opens the brush. Listed in that order, so a stroke satisfying more than one gives
/// the earliest still owed — which is also what brings a card passed over while
/// another was up back before the ones behind it.
///
/// The counts are set from what each deed *costs* to keep doing the hard way. Two
/// strokes is no commitment at all, but a painter with no color picker has already
/// wanted one; ten trips to the size slider is somebody who has decided this is how
/// they work, which is exactly when the drag is worth knowing. Almost none fires on a
/// first try, deliberately: a tip in the first minute is noise, because nothing has
/// been wanted yet. The two that do are argued for where the rule is tested
/// (`tests::AT_ONCE`) — one answers a question its own deed raises, and the other
/// waits on a deed nobody reaches by accident.
static LESSONS: &[Lesson] = &[
    Lesson {
        key: "color-panel",
        deed: Deed::Stroke,
        after: 4,
        // Picking a color is the whole of what this card is for, and nobody picks one
        // by accident — so somebody who already has is not shown it at all.
        answer: Answer::Known(&[Deed::ChangedColor]),
        anchor: Anchor::Panel(PanelId::Color),
        side: Side::LeftAtTop,
        title: "Every color, at your brush tip",
        body: "This color picker uses a color space tuned to human perception, so your \
               hue doesn't drift as you change lightness, and every step feels uniform. \
               The wheel holds every color your screen can show at that lightness \u{2014} \
               its edge is where they run out.",
    },
    Lesson {
        key: "panel-column",
        deed: Deed::Stroke,
        after: 5,
        // Reaching in *is* the lesson, so the wake is the acknowledgement — a better
        // one than the button, since it is the gesture rather than a claim about it.
        //
        // The card's anchor already takes it down when the panels come back
        // (`Anchor::on_screen`), and this is deliberately not that: the anchor says
        // the card has nothing to point at, and the deed says the artist answered it.
        // Only the second is a reason to write the lesson off as given.
        //
        // `Doing` and not `Known`, and this is the row that argues for the
        // distinction: the slice is the whole right-hand edge of the window, so a
        // pointer on its way to anything at all wakes the panels. Counted backwards
        // this would foreclose itself for very nearly everybody.
        answer: Answer::Doing(&[Deed::WokePanels]),
        anchor: Anchor::PanelColumn,
        side: Side::LeftAtMiddle,
        title: "Your panels are still here",
        body: "They just get out of the way while you paint. Point to the right of the canvas, and they'll come right back. You can always disable this behavior in the Settings.",
    },
    Lesson {
        key: "brush-panel",
        deed: Deed::Stroke,
        after: 10,
        // The panel's controls, all of them: the two sliders, the Edit brush… button
        // and a row of the library below. Touch any one and the panel has been found,
        // which is the only thing the card is there to say — so the rest of what it
        // lists is a paragraph nobody needs read to them.
        //
        // The Save preset button is the one control missing from this list, and it is
        // missing because there is no deed for it. It is also the one control nobody
        // reaches before they have used the others.
        answer: Answer::Known(&[
            Deed::TunedBrush,
            Deed::AppliedPreset,
            Deed::OpenedBrushEditor,
        ]),
        anchor: Anchor::Panel(PanelId::Brush),
        side: Side::LeftAtTop,
        title: "Pick a brush, then make it yours",
        body: "Adjust the brush size and how much paint to put on it. \
               \u{201C}Edit brush\u{2026}\u{201D} opens the full brush editor, or you can select a saved preset.",
    },
    Lesson {
        key: "select-panel",
        deed: Deed::Stroke,
        after: 40,
        // A selection committed is the answer, and the deed the *next* lesson waits on
        // — so somebody who takes this one goes on to the Layers panel rather than
        // being told twice about the same drag.
        answer: Answer::Known(&[Deed::Selection]),
        anchor: Anchor::Panel(PanelId::Select),
        side: Side::LeftAtTop,
        title: "Paint inside a shape",
        body: "Select a region and every tool acts only inside it.",
    },
    Lesson {
        key: "layers-panel",
        deed: Deed::Selection,
        after: 3,
        // A layer added, by whichever route. "Paint in layers" is a thing to be told
        // once and only before the first one — after that it is the app describing
        // what is plainly on screen.
        answer: Answer::Known(&[Deed::AddedLayer]),
        anchor: Anchor::Panel(PanelId::Layers),
        side: Side::LeftAtTop,
        title: "Paint in layers",
        body: "Build your painting in layers, group them, and change them all you want.",
    },
    Lesson {
        key: "panels-menu",
        deed: Deed::ClosedPanel,
        after: 1,
        // Opening one from the menu would be the answer, and there is no honest deed
        // for it: the tour opens panels itself (`Anchor::reveal`), so a deed counted at
        // `layout::open_panel` would have half these cards answering each other as they
        // came up.
        answer: Answer::Button,
        anchor: Anchor::CommandRail,
        side: Side::RightAtTop,
        title: "Nothing is lost by closing it",
        body: "Panels can be opened and closed in this menu.",
    },
    Lesson {
        key: "tune-drag",
        deed: Deed::TunedBrush,
        after: 10,
        // The drag this teaches is bracketed by `not_reaching`, so it produces no
        // deed to answer with — and that is the feature rather than the gap. Somebody
        // who already drags never accrues `TunedBrush` at all and is never offered
        // this; there is nothing left for an answer to catch.
        answer: Answer::Button,
        anchor: Anchor::Panel(PanelId::Brush),
        side: Side::LeftAtTop,
        title: "Size and flow, without leaving the painting",
        body: "Hold Ctrl (\u{2318} on a Mac) and drag on the canvas: sideways sets Size, \
               up and down sets Flow.",
    },
    Lesson {
        key: "quick-slots",
        deed: Deed::AppliedPreset,
        after: 3,
        // `tune-drag`'s argument, from the other side: the deed is the library row
        // being *clicked*, which the number keys never produce, so somebody already
        // fluent with the rack never earns this card in the first place.
        answer: Answer::Button,
        anchor: Anchor::QuickSlots,
        side: Side::RightAtMiddle,
        title: "Brushes one key press away",
        body: "Hold a number key to paint with a brush from this rack. Release to return to your previous brush. Change a slot while you hold the key and the slot keeps the change.",
    },
    Lesson {
        key: "shape-assist",
        deed: Deed::Undo,
        after: 10,
        // A stroke that snapped. Draw-and-hold is not something a hand does by
        // accident, so one of them is the artist knowing — and it is the deed the
        // *next* lesson waits on, which is the tour moving on rather than repeating
        // itself.
        answer: Answer::Known(&[Deed::AssistedStroke]),
        // The painting, and pointing down into it. The assist has no chrome at all
        // — it is a thing you do with the pen, on the canvas — so every other
        // anchor would have put the card beside a control that has nothing to do
        // with it. This is the one lesson whose subject *is* the place it is shown.
        anchor: Anchor::Canvas,
        side: Side::Inside,
        title: "A straight edge whenever you need it",
        body: "Draw a rough line or ellipse and then hold. Turn it off in the settings if it gets in the way.",
    },
    Lesson {
        key: "guides-panel",
        deed: Deed::AssistedStroke,
        after: 5,
        // A guided line means a guide was made and left visible, which cannot have
        // happened without this panel. So the card would be pointing at a panel the
        // artist has already used, to tell them it is where guides are edited.
        answer: Answer::Known(&[Deed::GuidedLine]),
        anchor: Anchor::Panel(PanelId::Guides),
        side: Side::LeftAtTop,
        title: "Getting some perspective",
        body: "Add and edit perspective guides here.",
    },
    Lesson {
        key: "perspective-assist",
        deed: Deed::GuidedLine,
        after: 2,
        // The one lesson that could only be answered by its own deed, which is the one
        // deed that cannot answer it: this card is what explains what those two guided
        // lines were. Nothing else in the app is the thing it describes.
        answer: Answer::Button,
        anchor: Anchor::Panel(PanelId::Guides),
        side: Side::LeftAtTop,
        title: "Your held lines know about the grid",
        body: "With a guide on screen, a line you draw-and-hold near one of its axes is \
               aimed exactly down that axis. Hold a rough circle on a plane \
               and it comes out a circle in perspective.",
    },
    Lesson {
        key: "navigator",
        deed: Deed::LongPan,
        after: 4,
        // "You don't have to drag that far", told to somebody who was already
        // travelling by the miniature, is the tour at its worst — so one use of it
        // ends the matter.
        answer: Answer::Known(&[Deed::UsedNavigator]),
        anchor: Anchor::Navigator,
        side: Side::RightAtBottom,
        title: "You don't have to drag that far",
        body: "The navigator is the whole piece at a glance. Click or drag inside it to go somewhere. Drag with the right button to rotate the canvas.",
    },
    Lesson {
        key: "eyedropper",
        deed: Deed::ChangedColor,
        after: 10,
        // A pick, which is the gesture this teaches. The bracket around the
        // eyedropper's write already keeps somebody fluent with it from accruing
        // `ChangedColor` at all (`tune-drag` says the same of the tuning drag) — but
        // that is an argument about a deed *not* being counted, and it cannot answer
        // the card already up. This can: the artist finds the Alt-drag while the tip
        // about the Alt-drag is on screen, which is exactly the moment the tip has
        // stopped being worth reading.
        answer: Answer::Known(&[Deed::PickedColor]),
        anchor: Anchor::Panel(PanelId::Color),
        side: Side::LeftAtTop,
        title: "Your canvas is a palette",
        body: "Hold Alt over the canvas and drag.",
    },
    Lesson {
        key: "pick-options",
        deed: Deed::PickedColor,
        after: 5,
        // The bar's own chips would be the answer, and there is no deed for them —
        // nor could there usefully be one, since the bar is only on screen while the
        // modifier is held.
        answer: Answer::Button,
        // The Color panel, and not the bar the options actually live on. The bar
        // exists only while Alt is down and not dragging (`panels::pick`), so a card
        // anchored there could only be shown during a held modifier and would be
        // taken down as an answer the moment the key came up — spent, unread. The
        // panel is where color lives, it is where this lesson's own sibling points,
        // and it is on screen when there is time to read.
        anchor: Anchor::Panel(PanelId::Color),
        side: Side::LeftAtTop,
        title: "Choose what the sampler sees",
        body: "While Alt is held, the bar by the cursor sets how far a sample reaches \
               \u{2014} this layer, everything under it, or the whole canvas \u{2014} and how \
               wide a patch it averages. Point takes one texel; 5\u{00D7}5 reads a blended \
               edge the way the eye does.",
    },
    // The brush editor's series (§24.5). Five cards on one deed, walked through in
    // this order by the chain in `dismiss` — so this list is the tour of the dialog
    // and its order is the only thing deciding what is said when.
    //
    // All five answer to the button alone, and it is the same reason five times: each
    // describes what a *section of a dialog is for*, and there is no act that
    // demonstrates having understood one. Moving a knob inside the editor is not it —
    // that is the thing the card is explaining, and somebody who moves it to see what
    // happens is the reader the paragraph was written for. The chain is what keeps
    // this cheap anyway: Next walks the series, so nobody presses five buttons to be
    // rid of it.
    Lesson {
        key: "be-preview",
        deed: Deed::OpenedBrushEditor,
        after: 1,
        answer: Answer::Button,
        anchor: Anchor::BrushEditor(BrushPart::Preview),
        side: Side::LeftAtTop,
        title: "What you see is what you get",
        body: "This is a real stroke, laid by the brush as it stands. Every knob you \
               move below redraws it, so you can tune by looking rather than by \
               reading.",
    },
    Lesson {
        key: "be-tip",
        deed: Deed::OpenedBrushEditor,
        after: 1,
        answer: Answer::Button,
        anchor: Anchor::BrushEditor(BrushPart::Tip),
        side: Side::RightAtTop,
        title: "Tip \u{2014} the footprint",
        body: "The shape swept along the path. A round tip has hardness; any image \
               you drop in becomes a stamp, and the gallery keeps it. What aims it is \
               the row of chips: follow the stroke, or hold an angle. Stretch gives a \
               round tip an axis, which is what turns it into a flat brush.",
    },
    Lesson {
        key: "be-paint",
        deed: Deed::OpenedBrushEditor,
        after: 1,
        answer: Answer::Button,
        anchor: Anchor::BrushEditor(BrushPart::Paint),
        side: Side::RightAtTop,
        title: "Paint \u{2014} how much, and how long it lasts",
        body: "Flow is how much goes down per unit travelled. Drain is the brush \
               running out: at zero it never does, which is a marker or a pencil; \
               above it the stroke thins as it goes, which is a loaded brush on dry \
               canvas. Those two together are most of the difference between media.",
    },
    Lesson {
        key: "be-color",
        deed: Deed::OpenedBrushEditor,
        after: 1,
        answer: Answer::Button,
        anchor: Anchor::BrushEditor(BrushPart::Color),
        side: Side::RightAtTop,
        title: "Color dynamics \u{2014} the wobble that reads as pigment",
        body: "The color wanders across the width of the tip and along the stroke, \
               following a noise field rather than a random number per stamp. That is \
               why it reads as pigment sitting unevenly instead of as static: \
               neighbouring bristles stay neighbours. A little of this is what keeps a \
               flat fill from looking printed.",
    },
    Lesson {
        key: "be-pickup",
        deed: Deed::OpenedBrushEditor,
        after: 1,
        answer: Answer::Button,
        anchor: Anchor::BrushEditor(BrushPart::Pickup),
        side: Side::RightAtTop,
        title: "Pickup \u{2014} moving paint that is already there",
        body: "The rest of the brush lays paint; this moves it. Lift takes canvas \
               paint onto the tip so the next stretch of stroke carries it \u{2014} which \
               is a smudge, and with no paint of its own a palette knife. Bleed \
               spreads sideways into what it passes over. Together they are how wet \
               paint behaves when something drags through it.",
    },
    Lesson {
        key: "timeline",
        deed: Deed::Redo,
        after: 2,
        // Scrubbing the bar would be the answer, and the bar is what this card's own
        // anchor opens (`Anchor::reveal`) — so the deed would have to tell a scrub
        // apart from the tour having put the bar on screen, which is `panels-menu`'s
        // problem and has `panels-menu`'s answer.
        answer: Answer::Button,
        anchor: Anchor::TimelineBar,
        side: Side::Above,
        title: "Rewind, look, come back",
        body: "Every stroke lives in the history. \
               Drag the scrubber back to the moment it went wrong and simply carry on \
               painting from there, or press play to watch the piece arrive. \
               \u{2630} \u{2192} Timeline brings this back.",
    },
];

/// Start listening, and load what this browser has stored.
///
/// Called at the **end** of app startup rather than from the root's body, for the
/// reason `files::bind_file_launch` is: everything before it — the opening preset,
/// the opening color, the stored preferences — reaches the engine through
/// [`dispatch`](crate::state::dispatch) too, and none of it is something the user
/// did.
pub fn begin(state: AppState) {
    let mut ledger = state.tutor.ledger;
    ledger.set(stored());
    let mut armed = state.tutor.armed;
    armed.set(true);
}

/// Open (`true`) or close (`false`) a bracket saying the brush is being written by
/// something other than the artist **reaching for one of its controls**.
///
/// The one thing the tour asks of the rest of the app, and it earns the exception:
/// what a `SetBrush` came out of is not on the command and cannot be derived from
/// it, in the way the dwell behind [`GestureCommand::Hold`] is not on one either
/// (§6.9). Three callers, and each is a different reason the write is not somebody
/// reaching for a slider:
///
/// - the tuning drag (`input::Tune`, §18.1.9) and the eyedropper
///   (`input::pick_color`, §18.0.2) — the two gestures a lesson *teaches*, so
///   counting them would be waiting for fluency and then offering to teach it;
/// - `presets::wear`, the one door a whole tool arrives through — a preset click or
///   a quick slot in either direction (§18.1.8). A tool arriving is not an
///   adjustment of the one you had, even in the rare case where it differs in
///   nothing but its size.
///
/// **A depth count rather than a flag**, because the brackets nest: a number key
/// held mid-tuning-drag swaps the tool inside the drag's own bracket, and a flag
/// would have that swap's close cancel the drag's. Underflow saturates, so a stray
/// close costs nothing.
///
/// A bracket left open by a caller that failed to close it costs *counting*, never
/// a wrong card — the failure direction to have, since the other one is a card
/// nobody asked for.
pub fn not_reaching(state: AppState, open: bool) {
    let mut depth = state.tutor.not_reaching;
    let now = *depth.peek();
    depth.set(if open {
        now.saturating_add(1)
    } else {
        now.saturating_sub(1)
    });
}

/// Report a deed the command stream cannot name at all.
///
/// [`not_reaching`]'s sibling and its opposite: that one says a command should not
/// be read, this one says something happened that no command describes. One caller
/// — `presets::apply`, which is the Brush panel's row click and the only thing that
/// means *the artist chose a different tool from the library*. The `SetBrush` it
/// leads to says a brush changed and cannot say that a row was clicked, and the
/// quick slots produce a command of exactly the same shape (§18.1.8) — which is the
/// gesture that lesson goes on to teach.
pub fn did(state: AppState, deed: Deed) {
    if !*state.tutor.armed.peek() {
        return;
    }
    tally(state, deed);
}

/// Whether a card on screen is pointing **into the panel stack**.
///
/// Asked by `layout::standing_down`, which is what the stack's fade and its wake
/// slice are decided by. The stack has a second way to be out of the way that no
/// other chrome has — it stays down after a gesture until the pointer reaches for it
/// (§11) — and a card that pointed at a panel which had faded out from under it
/// would be an arrow aimed at nothing.
///
/// Only [`Anchor::Panel`]. The lesson about the panel *column* is the one that wants
/// the stack down, since a strip you reach into to bring the panels back is
/// unteachable with the panels already up (§24.5); it holds nothing.
///
/// A subscribing read, since the caller is a render.
pub fn holding_panels(state: AppState) -> bool {
    (state.tutor.showing)()
        .and_then(|i| LESSONS.get(i))
        .is_some_and(|l| matches!(l.anchor, Anchor::Panel(_)))
}

/// Read one of the user's commands as a deed, and bring a lesson due if that was
/// the one it was waiting for.
///
/// **Called before the command reaches the engine**, and that ordering is
/// load-bearing: [`brush_deed`] decides what a `SetBrush` *means* by comparing it
/// against the brush the engine is still holding, which is only there to be read
/// until the command lands.
pub fn observe(state: AppState, command: &InputCommand) {
    if !*state.tutor.armed.peek() {
        return;
    }
    for deed in read(state, command) {
        tally(state, deed);
    }
}

/// The deeds `command` reports — usually none, sometimes one, and for a stroke that
/// snapped along a guide three at once (see [`stroke`]).
///
/// Split from [`observe`] because this is the half with the judgement in it, and the
/// half that writes nothing but the pan run it is obliged to accumulate. A `Vec`
/// where most of these arms answer with one deed or none, and it costs nothing where
/// it matters: `Vec::new()` does not allocate, and that is what every command at
/// pointer rate gets.
fn read(state: AppState, command: &InputCommand) -> Vec<Deed> {
    match command {
        // A commit, not a start: a stroke abandoned by a second finger (§18.1.7)
        // arrives as `Cancel` and left no paint, and counting it would be counting
        // a mark that is not on the canvas.
        InputCommand::Gesture(GestureCommand::End) => stroke(state),
        // Neither of these is the transport's. The playback loop drives the playhead
        // with these very commands (§18.2.4) and would otherwise score eight a
        // second; the user's own stops playback *before* it dispatches
        // (`commands::edit_history`), so this reads false by the time it is asked.
        InputCommand::Doc(DocCommand::Redo) => one(Deed::Redo, !playing(state)),
        InputCommand::Doc(DocCommand::Undo) => one(Deed::Undo, !playing(state)),
        // Not one that anybody reached for a control to make — see [`not_reaching`].
        InputCommand::View(ViewCommand::SetBrush(_)) if *state.tutor.not_reaching.peek() > 0 => {
            Vec::new()
        }
        InputCommand::View(ViewCommand::SetBrush(next)) => {
            let Some(held) = state.obs.peek().as_ref().map(|o| o.brush) else {
                return Vec::new();
            };
            brush_deed(&held, next).into_iter().collect()
        }
        // Both shapes of pan the app can make: the one-pointer drags and the last
        // finger of a two-finger gesture arrive as `Pan`, and the pair itself as
        // `Pinch`, whose translation is the distance between its two anchors. A
        // `Zoom` is deliberately not travel — same gesture family, different act.
        InputCommand::View(ViewCommand::Pan { delta }) => {
            pan(state, delta.length()).into_iter().collect()
        }
        InputCommand::View(ViewCommand::Pinch { anchor, to, .. }) => {
            pan(state, (*to - *anchor).length()).into_iter().collect()
        }
        // Travelling by the miniature rather than by dragging the painting. Absolute
        // where every other way of getting somewhere is a delta, and emitted by nothing
        // else — see [`Deed::UsedNavigator`], which is what makes this arm a read of a
        // *gesture on a piece of chrome* off a stream of document facts.
        InputCommand::View(ViewCommand::CenterOn(_)) => vec![Deed::UsedNavigator],
        // The Layers panel's button and the command that shares it. An image or a
        // matte arriving carries a command of its own, so neither lands here.
        InputCommand::Doc(DocCommand::AddLayer { .. }) => vec![Deed::AddedLayer],
        _ => Vec::new(),
    }
}

/// `deed` if `counts`, as the list [`read`] answers with.
fn one(deed: Deed, counts: bool) -> Vec<Deed> {
    if counts { vec![deed] } else { Vec::new() }
}

/// Whether the timeline transport is moving the playhead right now (§18.2.4).
fn playing(state: AppState) -> bool {
    crate::panels::timeline::is_playing(state)
}

/// What a committed gesture was: nothing for a selection drag, and otherwise a
/// stroke plus whatever the shape assist made of it (§6.9).
///
/// **Up to three deeds for one command, and they are not alternatives.** A stroke
/// that snapped along a vanishing line is a stroke, an assisted stroke and a guided
/// line all at once, and each feeds a different lesson: counting only the most
/// specific would stall the two behind it for somebody who works entirely on a grid.
///
/// Every read here happens **before** the command reaches the engine, which is the
/// only moment two of them answer: the assist lives on the gesture and goes with it
/// (`Engine::assisted`), so a line asked for after the `End` is a line nobody can
/// see any more.
fn stroke(state: AppState) -> Vec<Deed> {
    // Which tool the gesture was is not on the command — but it is on the
    // projection, which is where the whole app asks (§6.8).
    let Some(obs) = state.obs.peek().clone() else {
        return Vec::new();
    };
    if obs.tool.is_selection() {
        // The same gesture builds a mask or lays a fill, and which was decided when
        // the drag started (§18.0.4). Only the first is a selection; the second is
        // paint that happens to have arrived through a marquee.
        return one(Deed::Selection, obs.shape_action.is_select());
    }
    let mut deeds = vec![Deed::Stroke];
    // The engine rather than the projection, because a gesture in flight is not
    // projected: `observe` is what refreshes the chrome, and it runs after this.
    let assisted = crate::state::with_engine_quiet(state, |r| r.assisted()).flatten();
    if let Some(shape) = assisted {
        deeds.push(Deed::AssistedStroke);
        // A guide on the screen is what makes the grid's aiming reachable (§20.6) —
        // whether *this* line landed near enough for it to take is not the question,
        // because the lesson is about the thing that is about to start happening.
        let guided = shape == stark_engine::Assisted::Line && obs.guides.iter().any(|g| g.visible);
        if guided {
            deeds.push(Deed::GuidedLine);
        }
    }
    deeds
}

/// What moved between two brushes, as a deed — or `None` where the answer is "too
/// much to be one".
///
/// The test is **confinement**, not difference: a change counts as a size change
/// when the brush is otherwise untouched, and as a color change when the same is
/// true of the color. That is what tells the size slider (radius alone) and the
/// eyedropper (color alone) apart from a preset click or a quick-brush slot, which
/// move a dozen fields at once and are an adjustment of nothing.
///
/// Written by copying the old brush and grafting the candidate fields onto it, so
/// "everything else is equal" is one `==` against the real type rather than a list
/// of comparisons that a brush parameter added later would silently fall out of.
fn brush_deed(was: &BrushParams, now: &BrushParams) -> Option<Deed> {
    let mut tuned = *was;
    tuned.radius = now.radius;
    tuned.dynamics.add = now.dynamics.add;
    if tuned == *now && (was.radius != now.radius || was.dynamics.add != now.dynamics.add) {
        return Some(Deed::TunedBrush);
    }

    let mut colored = *was;
    colored.color = now.color;
    if colored == *now && was.color != now.color {
        return Some(Deed::ChangedColor);
    }

    None
}

/// Feed `travel` page px into the run of panning in flight, and answer
/// [`Deed::LongPan`] on the sample that takes it past [`LONG_PAN`].
///
/// The one deed with state of its own, because it is the one that is not a report
/// of an act but a *measurement* of one: no single `Pan` is long, and the run has
/// to be accumulated to know. Counted at the crossing rather than at the run's end,
/// so nothing has to detect an end — a run that stops short simply never scores,
/// and the next `Pan` after [`COALESCE`] starts a new one.
fn pan(state: AppState, travel: f32) -> Option<Deed> {
    let mut recent = state.tutor.recent;
    // Out into a local before the write below: a `peek` left live across a `write`
    // of the same signal is a panic in a handler, and this runs at pointer rate.
    let mut run = *recent.peek();
    let now = platform::now_seconds();
    if now - run.at[Deed::LongPan.slot()] > COALESCE {
        run.pan = 0.0;
        run.long = false;
    }
    run.at[Deed::LongPan.slot()] = now;
    run.pan += travel;
    let crossed = !run.long && run.pan >= LONG_PAN;
    run.long |= crossed;
    recent.set(run);
    crossed.then_some(Deed::LongPan)
}

/// Count `deed`, unless it is the same act as the last one — and offer whatever
/// lesson that brings due.
fn tally(state: AppState, deed: Deed) {
    let mut recent = state.tutor.recent;
    let now = platform::now_seconds();
    // The pan run keeps its own clock inside `pan`, which has already decided that
    // a crossing is one act; every other deed is coalesced here.
    if deed != Deed::LongPan {
        let mut run = *recent.peek();
        let repeat = now - run.at[deed.slot()] <= COALESCE;
        // The clock moves either way, so the gap is always measured from the last
        // report of the run rather than from its first.
        run.at[deed.slot()] = now;
        recent.set(run);
        if repeat {
            return;
        }
    }

    let mut ledger = state.tutor.ledger;
    // Cloned out before the write, for the borrow reason stated in `pan` — and the
    // clone is an array of a dozen and a small set of short strings.
    let mut book = ledger.peek().clone();
    let slot = deed.slot();
    book.tally[slot] = book.tally[slot].saturating_add(1);
    let owed = due(&book, deed, *state.chrome_hiding.peek());
    ledger.set(book);
    // The tally is what has to survive a reload; which lesson it brought due is
    // re-derived on the next deed of the same kind if this visit ends before the
    // card is dismissed.
    save(state);

    // One card at a time. A lesson passed over here is not lost: its threshold has
    // been crossed for good, so `due` answers it again on the very next deed of its
    // kind.
    let taken = state.tutor.due.peek().is_some() || state.tutor.showing.peek().is_some();
    if let Some(i) = owed
        && !taken
        && *state.tutor.enabled.peek()
    {
        let mut slot = state.tutor.due;
        slot.set(Some(i));
    }

    // **Last**, so a deed that answers the card up cannot also replace it. Answering
    // one and opening another in the same breath would read as the tour arguing back,
    // and the lesson this deed owes is not lost by being passed over — `due` hands it
    // out again on the next deed of its kind, which is the same rule every other
    // passed-over card lives by.
    answered(state, deed);
}

/// Take down whatever card `deed` has just answered ([`Answer`]) — the one on screen,
/// or the one still waiting to become it.
///
/// **Both, because a card that has come due and not yet been shown is the case this
/// most has to catch.** A lesson waits in [`TutorState::due`] for as long as the canvas
/// is in hand or a dialog is up, and what an artist does in that gap is exactly what
/// they were about to be taught — so the card arriving afterwards would be the tour at
/// its slowest and most patronizing.
///
/// Marked **given** either way, which is the difference between this and the card
/// merely losing its anchor ([`abandon`]): the artist has done the thing, and doing it
/// is a better acknowledgement than the button. And the button's chain is deliberately
/// not run — the next lesson this deed owes arrives on the deed's own terms rather than
/// filling the hole the answer just made.
fn answered(state: AppState, deed: Deed) {
    let answers = |i: usize| {
        LESSONS
            .get(i)
            .is_some_and(|l| l.answer.dismisses().contains(&deed))
    };

    // The waiting one first. The two slots are never both set — `tally` promotes into
    // `due` only when neither is taken, and `retire` empties `showing` before the
    // `dismiss` chain refills `due` — so the order is a formality; what it is not is a
    // reason to let either go unasked.
    let mut waiting = state.tutor.due;
    // Out into a local before the write, for the borrow reason `pan` gives.
    let owed = *waiting.peek();
    if let Some(i) = owed.filter(|i| answers(*i)) {
        waiting.set(None);
        if let Some(lesson) = LESSONS.get(i) {
            give(state, lesson);
        }
        return;
    }

    let up = *state.tutor.showing.peek();
    if let Some(i) = up.filter(|i| answers(*i)) {
        retire(state, i);
    }
}

/// The first lesson `deed` has brought due against `book`, if any — skipping any
/// whose subject this browser has switched off ([`Lesson::applies`]) and any the
/// artist has already answered by doing it ([`Lesson::foreclosed`]).
///
/// The two skips are the same shape and it is worth saying why: neither writes
/// anything. A lesson passed over here is *stepped past*, so the next one its deed
/// owes comes forward in the same breath — where marking it given would spend a row of
/// the ledger on a judgement that could change (a setting turned back on, a tally that
/// turns out to have been somebody else's browser).
///
/// `chrome` is passed rather than read off the app, so the whole of "which lesson is
/// owed" stays a function of the ledger and a value: it is asked from three places
/// and tested from a fourth, and a signal read in here would be a subscription taken
/// in whichever scope happened to ask.
fn due(book: &Ledger, deed: Deed, chrome: ChromeHiding) -> Option<usize> {
    let count = book.tally[deed.slot()];
    LESSONS.iter().position(|l| {
        l.deed == deed
            && count >= l.after
            && !book.given.contains(l.key)
            && l.applies(chrome)
            && !l.foreclosed(book)
    })
}

/// Put a lesson away for good, and **offer the next one its deed still owes**.
///
/// The chain is what makes a *series* possible: the brush editor's cards all wait on
/// one deed, and being opened once is the whole of what earns every one of them, so
/// acknowledging a card has to bring the next rather than waiting for the dialog to
/// be opened again (§24.5). It costs nothing anywhere else, because the ordinary
/// case is a deed that owes one lesson at a time — and where it owes more, that is a
/// *backlog* built up while cards were passed over, and draining it in order beats
/// making the artist earn each one twice.
///
/// It cannot run away: every turn marks one lesson given and [`due`] skips those, so
/// the chain is exactly as long as the lessons that were already owed.
fn dismiss(state: AppState, i: usize) {
    let Some(deed) = retire(state, i) else {
        return;
    };
    let book = state.tutor.ledger.peek().clone();
    if let Some(next) = due(&book, deed, *state.chrome_hiding.peek()) {
        let mut slot = state.tutor.due;
        slot.set(Some(next));
    }
}

/// Put a lesson away because **the thing it points at has gone** — the panel closed,
/// the dialog dismissed.
///
/// [`dismiss`] without the chain, and the difference is load-bearing: closing the
/// brush editor takes the anchor out from under every card in its series at once, so
/// a chain here would retire the lot in one flush and the artist would have been
/// "taught" several things they never saw. What happens instead is that the card on
/// screen is answered and the rest stay owed for the next time the dialog opens.
fn abandon(state: AppState, i: usize) {
    retire(state, i);
}

/// Turn the tour on or off — ⚙ → Tips, and the card's own "Stop tips".
///
/// **Writing the switch is not the whole of turning the tour off.** [`tally`] reads
/// it on the way *in*, so it keeps new lessons from coming due and says nothing about
/// the card already on screen — which was promoted under the old answer and would sit
/// there until it was answered, a preference that leaves the very thing it governs
/// standing. So the two go together, and they go together *here*: the switch has two
/// controls, and a rule kept in one of them is a rule the other disagrees with.
///
/// Nothing is retired on the way out. Somebody who turned the tour off has not been
/// taught the lesson they were looking at, and turning it back on should still owe it
/// to them — which is what the switch's own note in the dialog promises.
/// [`TutorState::due`] is left alone for the same reason: what keeps it down meanwhile
/// is a guard in the card's promotion effect, so a lesson brought due by any route at
/// all — the [`dismiss`] chain included — stays off the screen while the tour is off,
/// rather than only the ones that came through here.
///
/// Persisting is the caller's, and both callers already do it: the dialog saves after
/// every row it changes (`settings::SettingToggle`), and the button saves for itself.
pub fn set_enabled(state: AppState, on: bool) {
    let mut enabled = state.tutor.enabled;
    enabled.set(on);
    if on {
        return;
    }
    let mut showing = state.tutor.showing;
    // Out into a local before the write, for the borrow reason `pan` gives.
    let up = *showing.peek();
    showing.set(None);
    if let Some(lesson) = up.and_then(|i| LESSONS.get(i)) {
        release_panels(state, lesson);
    }
}

/// The half both ways out share: take the card down and write the lesson into the
/// ledger. Answers the deed it was waiting on, or `None` for an index that names no
/// lesson.
fn retire(state: AppState, i: usize) -> Option<Deed> {
    let mut showing = state.tutor.showing;
    showing.set(None);
    let lesson = LESSONS.get(i)?;
    release_panels(state, lesson);
    give(state, lesson);
    Some(lesson.deed)
}

/// Write `lesson` into the ledger as given, and persist.
///
/// [`retire`]'s other half, split out because a lesson can now be spent without ever
/// having been on screen: one answered while it was still waiting in
/// [`TutorState::due`] has no card to take down and no panel stack to let go of
/// ([`answered`]).
fn give(state: AppState, lesson: &Lesson) {
    let mut ledger = state.tutor.ledger;
    // Cloned out before the write, for the borrow reason `pan` gives.
    let mut book = ledger.peek().clone();
    book.given.insert(lesson.key.to_string());
    ledger.set(book);
    save(state);
}

/// Let go of the panel stack that `lesson`'s card was holding up
/// ([`holding_panels`]).
///
/// Letting go of it is not the same as asking for the panels to go away: without this
/// the stack would fade the instant the card came down, which reads as whatever took
/// the card down having *closed* the thing it was about. Waking it properly leaves it
/// where every other route to a panel leaves it — up until the next gesture.
///
/// Shared by every way out of a card rather than written into the one that
/// acknowledges it, because the argument is about the *stack* and not about the
/// answer: "Got it", "Stop tips" and the switch in the dialog all read the same way to
/// a panel that vanished as they were pressed.
fn release_panels(state: AppState, lesson: &Lesson) {
    if matches!(lesson.anchor, Anchor::Panel(_)) {
        crate::layout::wake_panels(state);
    }
}

/// One row of the stored ledger: a deed and its count, or a lesson already given.
///
/// A **list** of rows rather than one record with two fields, and the reason is
/// [`Ledger`]'s: the two halves forget differently, and only a row can be dropped on
/// its own. A tally under a name this build does not know costs that row
/// (`storage::load_list`), while the `given` names beside it survive — which they must,
/// because somebody who has seen a tip should not be shown it again by a release that
/// renamed its neighbour.
///
/// Untagged, so a row is `{"deed":"stroke","count":7}` or `{"given":"brush-panel"}`
/// rather than either wrapped in a discriminant it already carries. The two shapes
/// share no field, so nothing here is ambiguous.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub(crate) enum Row {
    Deed { deed: Deed, count: u32 },
    Given { given: String },
}

impl crate::storage::Entry for Row {
    const STORE: Store = Store::Tutor;
}

/// What this browser has stored, or an empty ledger — a browser that has stored
/// nothing and one whose store is damaged want the same thing, which is to be
/// treated as new.
fn stored() -> Ledger {
    let Some(rows) = crate::storage::load_list::<Row>() else {
        return Ledger::default();
    };
    let mut book = Ledger::default();
    for row in rows {
        match row {
            Row::Deed { deed, count } => book.tally[deed.slot()] = count,
            // Not checked against `LESSONS` — see [`Ledger`].
            Row::Given { given } => {
                book.given.insert(given);
            }
        }
    }
    book
}

/// Persist the ledger. Called after every deed, which is at most one write per
/// [`COALESCE`] and only while the app is being used.
fn save(state: AppState) {
    let book = state.tutor.ledger.peek().clone();
    let deeds = Deed::ALL.into_iter().filter_map(|deed| {
        let count = book.tally[deed.slot()];
        // A deed nobody has done is the absence of a row rather than a row saying
        // zero, so the record only ever holds what actually happened.
        (count > 0).then_some(Row::Deed { deed, count })
    });
    let given = book.given.iter().map(|k| Row::Given {
        given: k.to_string(),
    });
    let rows: Vec<Row> = deeds.chain(given).collect();
    crate::storage::save_list(&rows);
}

/// How wide a card whose **right** edge is pinned at `x` may be, as a declaration.
///
/// The card is placed from the anchor's edge and given no width to work with, so on
/// a narrow window it would otherwise run off the side. This is the answer, and it
/// is a *narrowing* rather than a nudge: a card that shifted to stay on screen would
/// leave its arrow pointing at nothing, and the arrow is the half that says which
/// thing is being talked about.
///
/// Written as a declaration and not measured, because it does not have to be: the
/// stylesheet already gives the card a width, this is a `max-width` over the top of
/// it, and `calc` knows the viewport where Rust would have to ask for it.
fn room_left(x: f32) -> String {
    format!("max-width: {:.1}px;", (x - EDGE).max(0.0))
}

/// [`room_left`] for a card whose **left** edge is pinned at `x`.
fn room_right(x: f32) -> String {
    format!("max-width: calc(100vw - {x:.1}px - {EDGE}px);")
}

/// [`room_left`] for a card **centred** on `x`, which is constrained by whichever
/// side of it has less room — hence the `min`, and the doubling: the card grows both
/// ways from the middle, so it may only be twice the narrower half.
fn room_about(x: f32) -> String {
    format!(
        "max-width: calc(min({x:.1}px, 100vw - {x:.1}px) * 2 - {:.1}px);",
        2.0 * EDGE
    )
}

/// How many animation frames [`measure`] will wait for its anchor before giving up.
///
/// Eight is about an eighth of a second, which is far longer than a render and a
/// layout and far shorter than anybody notices a card arriving. It is a *bound*
/// rather than a duration: the ordinary case answers on the first or second frame,
/// and this only decides how long an anchor that is never coming keeps being asked
/// for.
const TRIES: u32 = 8;

/// Measure whatever `selector` finds and hand it to the card, asking again on the
/// next few frames if it finds nothing yet.
///
/// **The retry is not defensive, it is the ordinary path.** A card is placed by the
/// very effect that *revealed* what it points at — the panel it opened, the rack it
/// pinned — and a reveal is a signal write whose render has not happened, let alone
/// been laid out. So the first look routinely finds nothing, and a single animation
/// frame is a race against Dioxus's own patch rather than a guarantee.
///
/// Losing that race used to be silent and strange: the card was armed and correct
/// and simply never drew, until something *else* this effect follows moved and
/// measured it again. Since `canvas_active` is one of those, the symptom was a tip
/// that appeared when the artist next painted a stroke — nowhere near the click that
/// earned it.
///
/// `for_lesson` is the card the chain was started for. A chain still in flight when
/// the card changes is answering a question nobody asked any more, and must not write
/// the new card's box away.
fn measure(
    state: AppState,
    mut anchored: Signal<Option<ElementBox>>,
    selector: String,
    for_lesson: usize,
    tries: u32,
) {
    if *state.tutor.showing.peek() != Some(for_lesson) {
        return;
    }
    if let Some(found) = platform::anchor_box(&selector) {
        anchored.set(Some(found));
        return;
    }
    // Out of frames: the anchor is not there, and saying so is what takes a card down
    // that is pointing at something gone. `on_screen` is the half that decides whether
    // the *lesson* survives that (`abandon`); this only stops it being drawn.
    if tries == 0 {
        anchored.set(None);
        return;
    }
    platform::on_animation_frame(move || measure(state, anchored, selector, for_lesson, tries - 1));
}

/// The lesson card: one at a time, floating beside the thing it points at.
///
/// Mounted at the app root for the life of the page and empty whenever no lesson
/// is showing, so the two effects below — the one that promotes a due lesson and
/// the one that measures its anchor — are never unmounted out from under a lesson
/// in flight.
#[component]
pub fn TutorCard() -> Element {
    let state = use_context::<AppState>();
    let layout = state.panels;
    let mut anchored = use_signal(|| None::<ElementBox>);

    // A resize moves everything the card could be pointing at. Bound to the window
    // once for the life of the page, like the app's other window listeners — this
    // component never unmounts, so there is nothing to unbind.
    use_hook(|| {
        let mut epoch = state.tutor.epoch;
        platform::on_window_event("resize", move |_| {
            let n = *epoch.peek();
            epoch.set(n + 1);
        });
    });

    // Show a lesson that has come due, once the screen is the user's again.
    //
    // Every condition here is a claim that the card would be *wrong* now rather
    // than merely unwelcome: mid-gesture the chrome is faded and a panel opened now
    // is put back to sleep by the release; a composing mode owns the whole window
    // (`crate::modes`); and a dialog is over the top of everything a card could
    // point at.
    use_effect(move || {
        let Some(i) = (state.tutor.due)() else { return };
        let Some(lesson) = LESSONS.get(i) else { return };
        // The switch, read here as well as in `tally`. That one is the way *in* — it
        // keeps a deed from bringing anything due while the tour is off — and this is
        // the way out, which is a separate question: `due` is also written by the
        // `dismiss` chain and can be left set by a lesson that was still owed when the
        // tour was switched off, and neither of those comes back through `tally`. A
        // subscribing read, so switching the tour back on offers what it still owed
        // rather than waiting for the next deed.
        if !(state.tutor.enabled)() {
            return;
        }
        // A dialog is over everything a card could point at — except when the card
        // *is* pointing at the dialog, which is the whole of why this asks the anchor
        // (`Anchor::inside_dialog`). The preset-save dialog is over even those, being
        // the one thing the brush editor can open on top of itself.
        let dialog = (state.brush_editor_open)() && !lesson.anchor.inside_dialog();
        let busy = (state.canvas_active)()
            || dialog
            || (state.preset_save_open)()
            || crate::modes::composing(state).is_some();
        if busy {
            return;
        }
        lesson.anchor.reveal(state, layout);
        let mut showing = state.tutor.showing;
        showing.set(Some(i));
        let mut due = state.tutor.due;
        due.set(None);
    });

    // Measure whatever the lesson on screen points at.
    //
    // Twice, and the second reading is the one that usually lands: this runs in the
    // render that *opened* the panel, which the browser has not laid out yet, so the
    // immediate measurement can be of an element that is not there. An animation
    // frame is after layout by definition, which is what makes the second reading
    // the true one.
    //
    // It follows the panel order and the hidden set as well as the lesson, because
    // both move the column the card points into: dragging a panel over the one being
    // explained would otherwise leave the card behind. Those two reads are also what
    // the dismissal below is watching for, which is why one effect does both.
    use_effect(move || {
        let showing = (state.tutor.showing)();
        let _ = (layout.order)();
        let _ = (state.tutor.epoch)();
        // The two anchors that come and go on their own: the wake slice is in the
        // DOM only while the panels are asleep *and* the canvas is out of hand
        // (`layout::PanelStack`), and the rack only while it is pinned. Followed
        // here so the card is measured again when either arrives — `on_screen`
        // below reads neither `canvas_active` nor `pinned` in a way that would
        // bring this effect back on its own.
        let _ = (state.canvas_active)();
        let _ = (state.slots.pinned)();
        let _ = (state.brush_editor_open)();
        // Closing the panel a card is about is an answer to the card, so the lesson
        // goes away with it rather than latching `showing` forever — which would be
        // a tour that ended silently at whichever tip the artist happened to close a
        // panel under. Asked before the measurement, and of the app's own state
        // rather than of the DOM, for the reason `on_screen` gives. It reads
        // `layout.hidden` and `timeline.open`, which is also what keeps this effect
        // following the two things that move the anchor's box.
        if let Some(i) = showing
            && let Some(lesson) = LESSONS.get(i)
            && !lesson.anchor.on_screen(state, layout)
        {
            abandon(state, i);
            return;
        }
        let (Some(i), Some(selector)) = (
            showing,
            showing
                .and_then(|i| LESSONS.get(i))
                .map(|l| l.anchor.selector()),
        ) else {
            anchored.set(None);
            return;
        };
        measure(state, anchored, selector, i, TRIES);
    });

    let Some(i) = (state.tutor.showing)() else {
        return rsx! {};
    };
    let Some(lesson) = LESSONS.get(i) else {
        return rsx! {};
    };
    // Nothing to point at — the panel was closed under the card, or the DOM has not
    // caught up yet. Drawing nothing is the honest answer; the effect above is still
    // watching, so the card comes back if the anchor does.
    let Some(at) = anchored() else {
        return rsx! {};
    };
    // Whether acknowledging this one brings another (`dismiss`). The button says so,
    // because "Got it" on the first of five is a lie about how many are coming — and
    // a card that turns out to have four behind it reads as the tour having got
    // stuck. Computed against the ledger *as it will be*, this lesson included, which
    // is what `retire` is about to write.
    let more = {
        let mut book = state.tutor.ledger.peek().clone();
        book.given.insert(lesson.key.to_string());
        due(&book, lesson.deed, *state.chrome_hiding.peek()).is_some()
    };

    // Placed against the anchor's own edges, with no reading of the viewport and no
    // measurement of the card itself: the translate does the work that knowing the
    // card's width would otherwise be needed for.
    let (side, place) = match lesson.side {
        Side::LeftAtTop => (
            "side-left",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateX(-100%); {}",
                at.left - GAP,
                at.top,
                room_left(at.left - GAP),
            ),
        ),
        Side::LeftAtMiddle => (
            "side-left-middle",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translate(-100%, -50%); {}",
                at.left - GAP,
                at.mid_y(),
                room_left(at.left - GAP),
            ),
        ),
        Side::RightAtTop => (
            "side-right",
            format!(
                "left: {:.1}px; top: {:.1}px; {}",
                at.right() + GAP,
                at.top,
                room_right(at.right() + GAP),
            ),
        ),
        Side::RightAtMiddle => (
            "side-right-middle",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateY(-50%); {}",
                at.right() + GAP,
                at.mid_y(),
                room_right(at.right() + GAP),
            ),
        ),
        Side::Inside => (
            "side-inside",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateX(-50%); {}",
                at.mid_x(),
                at.top + at.height * INSIDE_DEPTH,
                room_about(at.mid_x()),
            ),
        ),
        // The one placement that hands the stylesheet a measurement as well as a
        // position: where its arrow goes depends on how tall the anchor is, and this
        // is the only side whose anchor is sized by the artwork (`RightAtBottom`).
        //
        // A custom property is safe to write from one arm only, where a plain
        // declaration would not be: an inline style is applied property by property
        // in Dioxus, so a name this arm sets is *stranded* on the element when
        // another arm renders next. Nothing reads it there — the rule that does is
        // turned on by the side's class, and the class is rewritten whole.
        Side::RightAtBottom => (
            "side-right-bottom",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateY(-100%); \
                 --tutor-reach: {:.1}px; {}",
                at.right() + GAP,
                at.bottom(),
                at.height * 0.5,
                room_right(at.right() + GAP),
            ),
        ),
        Side::Above => (
            "side-above",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translate(-50%, -100%); {}",
                at.mid_x(),
                at.top - GAP,
                room_about(at.mid_x()),
            ),
        ),
    };

    let mut base = format!("tutor-card {side}");
    if lesson.anchor.inside_dialog() {
        base.push_str(" over-dialog");
    }

    rsx! {
        div {
            class: chrome_class(state, &base),
            style: "{place}",
            div { class: "tutor-head",
                span { class: "tutor-mark", {icon(icons::TOUR)} }
                span { class: "tutor-title", "{lesson.title}" }
            }
            div { class: "tutor-body", "{lesson.body}" }
            div { class: "tutor-actions",
                button {
                    class: "tutor-quiet",
                    title: "Stop showing tips. Settings has the switch to turn them back on.",
                    onclick: move |_| {
                        // The same door the dialog's switch goes through, which is
                        // what takes this card down as well as ending the tour — and,
                        // deliberately, does not mark it given ([`set_enabled`]).
                        set_enabled(state, false);
                        crate::prefs::save(state);
                    },
                    "Stop tips"
                }
                button {
                    class: if more { "chip tutor-done tutor-next" } else { "chip tutor-done" },
                    onclick: move |_| dismiss(state, i),
                    {icon(if more { icons::NEXT } else { icons::DONE })}
                    if more { "Next" } else { "Got it" }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every deed either brings a lesson due or answers one.
    ///
    /// The direction worth asserting: a deed nothing reads is a tally kept for
    /// nobody, which is exactly what a lesson removed without its deed leaves
    /// behind. The other direction is the compiler's — a lesson names a `Deed`.
    ///
    /// **Two ways to be read**, since an answer is as good a reason for a deed to
    /// exist as a threshold is. Three deeds are counted for their answers alone —
    /// reaching into the panel column, adding a layer, travelling by the miniature —
    /// and each is a thing no lesson would ever wait *for*: they are what an artist
    /// does once they already know.
    #[test]
    fn the_deeds_and_the_lessons_account_for_each_other() {
        for deed in Deed::ALL {
            let earns = LESSONS.iter().any(|l| l.deed == deed);
            let answers = LESSONS.iter().any(|l| l.answer.dismisses().contains(&deed));
            assert!(
                earns || answers,
                "{deed:?} is counted and no lesson waits for it or answers to it",
            );
        }
    }

    /// A lesson is never answered by the deed that earns it.
    ///
    /// Which would be a card that took itself down: the deed brings it due and the
    /// same deed spends it, so it would flash past on whichever report crossed the
    /// threshold — or, where it is a [`Answer::Known`], never be offered at all, since
    /// the tally that earns it is the tally that forecloses it.
    ///
    /// Worth a test rather than a type because the two fields are *usually* different
    /// deeds for a good reason (what earns a lesson is wanting the thing; what answers
    /// it is having it), and the case where they coincide looks perfectly reasonable
    /// on the row.
    #[test]
    fn no_lesson_answers_to_its_own_deed() {
        for l in LESSONS {
            assert!(
                !l.answer.dismisses().contains(&l.deed),
                "{}: {:?} both earns it and spends it",
                l.key,
                l.deed,
            );
        }
    }

    /// A live [`Answer`] carries deeds. An empty list is [`Answer::Button`] spelled
    /// the long way, and spelling it that way hides the reason there is nothing here
    /// — which is the one thing every `Button` row on the table is obliged to say.
    #[test]
    fn an_answer_that_answers_to_nothing_says_so() {
        for l in LESSONS {
            assert_eq!(
                l.answer == Answer::Button,
                l.answer.dismisses().is_empty(),
                "{}: an empty {:?} is a Button that did not admit it",
                l.key,
                l.answer,
            );
        }
    }

    /// A card grows away from the window edge its anchor stands on.
    ///
    /// The only thing about a placement that can be checked without a browser, and it
    /// is worth checking because nothing at runtime does. A card too *wide* narrows to
    /// fit ([`room_left`]); a card too *tall* has no equivalent — narrowing one only
    /// makes it taller — so a placement that hangs a card into an edge is simply a
    /// card half off the screen.
    ///
    /// The regression it exists for happened: the navigator's lesson kept
    /// [`Side::RightAtMiddle`] when the miniature moved out of the panel stack into
    /// the bottom-left corner, and a card centred on a box that stands [`EDGE`] off
    /// the foot of the window hangs its lower half over the edge.
    #[test]
    fn a_card_is_never_hung_into_the_edge_it_stands_on() {
        for lesson in LESSONS {
            let Some(edge) = lesson.anchor.against() else {
                continue;
            };
            assert!(
                lesson.side.grows().is_some_and(|toward| toward != edge),
                "{}: {:?} is against the {edge:?} of the window, and a {:?} card \
                 reaches that way — it would be drawn off the screen",
                lesson.key,
                lesson.anchor,
                lesson.side,
            );
        }
    }

    /// Lesson keys are what the ledger stores, so two lessons sharing one would be
    /// one lesson dismissing the other.
    #[test]
    fn lesson_keys_are_unique() {
        let keys: HashSet<&str> = LESSONS.iter().map(|l| l.key).collect();
        assert_eq!(keys.len(), LESSONS.len(), "two lessons share a key");
    }

    /// The deeds whose lessons may fire on the **first** one, where everything else
    /// waits for a second.
    ///
    /// A property of the *deed* rather than of the lesson, which is what makes it
    /// worth writing down: the question is never "is this tip important" — every tip
    /// thinks it is — but "could somebody have done this without meaning to". Two
    /// deeds could not, each for its own reason:
    ///
    /// - [`Deed::ClosedPanel`] **raises** the question its lesson answers. Close a
    ///   panel and "where did that go?" is immediate; answering on the second close
    ///   would be answering late, with the gap spent believing it was gone.
    /// - [`Deed::OpenedBrushEditor`] **is** the request. Opening the dialog is
    ///   somebody asking what is in it, and the series is the answer (§24.5).
    ///
    /// [`Deed::GuidedLine`] was a third and is not any more. It is still the hardest
    /// deed to reach by accident — a guide made, left visible, a stroke drawn *and*
    /// held still — but the pace it fires at was slowed to two along with the rest of
    /// the table, and an exemption nobody spends is what
    /// [`the_exceptions_are_all_spent`] exists to refuse.
    ///
    /// An *exception* list, so a deed added later is held to the strict rule by
    /// default — the safe direction, since the cost of getting it wrong is a tip in
    /// somebody's first minute. And a list of deeds rather than of lesson keys, so a
    /// renamed or deleted entry is a compile error instead of an exemption that
    /// quietly stops applying.
    const AT_ONCE: [Deed; 2] = [Deed::ClosedPanel, Deed::OpenedBrushEditor];

    /// A lesson at one fires on the user's first try, which the design is against
    /// everywhere it has not said otherwise — see [`AT_ONCE`].
    #[test]
    fn no_lesson_fires_on_a_first_try() {
        for l in LESSONS {
            if AT_ONCE.contains(&l.deed) {
                continue;
            }
            assert!(l.after >= 2, "{} fires after {}", l.key, l.after);
        }
    }

    /// Every exempted deed has a lesson that actually uses the exemption. One that
    /// did not would be a rule relaxed for nobody, and the next reader would take it
    /// as evidence that the deed is somehow special.
    #[test]
    fn the_exceptions_are_all_spent() {
        for deed in AT_ONCE {
            assert!(
                LESSONS.iter().any(|l| l.deed == deed && l.after == 1),
                "{deed:?} is exempted and no lesson fires at once on it",
            );
        }
    }

    /// The brush editor's series: several lessons on one deed, which the chain in
    /// `dismiss` walks through in table order.
    ///
    /// What is asserted is that it *is* a series and that it is contiguous — every
    /// card owed at the same moment, so the walk cannot stall halfway and leave the
    /// dialog half explained.
    #[test]
    fn the_brush_editor_is_a_series() {
        let series: Vec<&str> = LESSONS
            .iter()
            .filter(|l| l.deed == Deed::OpenedBrushEditor)
            .map(|l| l.key)
            .collect();
        assert!(series.len() > 1, "a series of one is not a series");

        let mut book = Ledger::default();
        book.tally[Deed::OpenedBrushEditor.slot()] = 1;
        // One open, and the whole series comes due one card at a time, in order.
        for key in &series {
            let owed =
                due(&book, Deed::OpenedBrushEditor, ChromeHiding::default()).expect("still owed");
            assert_eq!(LESSONS[owed].key, *key);
            book.given.insert((*key).to_string());
        }
        assert_eq!(
            due(&book, Deed::OpenedBrushEditor, ChromeHiding::default()),
            None,
            "and then no more"
        );
    }

    /// [`Deed::slot`] and the stored name both have to be total and one-to-one — the
    /// first indexes the tally, the second is what a stored row is found by.
    ///
    /// The names are `rename` attributes, so serde round-trips them by construction;
    /// what it does *not* check is that no two variants were given the same string,
    /// which would silently merge two tallies.
    #[test]
    fn every_deed_has_its_own_slot_and_its_own_name() {
        let slots: HashSet<usize> = Deed::ALL.into_iter().map(Deed::slot).collect();
        assert_eq!(slots.len(), Deed::COUNT);
        let names: HashSet<String> = Deed::ALL
            .into_iter()
            .map(|d| serde_json::to_string(&d).unwrap())
            .collect();
        assert_eq!(
            names.len(),
            Deed::COUNT,
            "a name spelled twice merges tallies"
        );
        assert!(serde_json::from_str::<Deed>("\"a deed no build has\"").is_err());
    }

    /// The confinement rule, which is the whole of what keeps a preset click from
    /// being read as an adjustment (see [`brush_deed`]).
    #[test]
    fn a_brush_change_counts_only_where_it_is_confined() {
        let was = BrushParams::default();

        let mut bigger = was;
        bigger.radius = was.radius * 2.0;
        assert_eq!(brush_deed(&was, &bigger), Some(Deed::TunedBrush));

        let mut looser = was;
        looser.dynamics.add = was.dynamics.add + 0.5;
        assert_eq!(brush_deed(&was, &looser), Some(Deed::TunedBrush));

        let mut red = was;
        red.color = [1.0, 0.0, 0.0, was.color[3]];
        assert_eq!(brush_deed(&was, &red), Some(Deed::ChangedColor));

        // A preset: the size and the color both, plus whatever else it carries. Not
        // an adjustment of either, so it counts as neither.
        let mut preset = was;
        preset.radius = was.radius * 2.0;
        preset.color = [1.0, 0.0, 0.0, was.color[3]];
        assert_eq!(brush_deed(&was, &preset), None);

        // The command that changes nothing — a slider dragged back to where it
        // started — is not a deed either.
        assert_eq!(brush_deed(&was, &was), None);

        // And a field that is neither size, flow nor color is somebody else's
        // business entirely.
        let mut drained = was;
        drained.drain = was.drain + 1.0;
        assert_eq!(brush_deed(&was, &drained), None);
    }

    /// A ledger written by a build that counted something this one does not still
    /// loads, and the half it understands survives — the rule `storage::load_list`
    /// exists for, asked of these rows.
    #[test]
    fn an_unknown_row_costs_itself_and_not_the_ledger() {
        let json = r#"[
            {"deed":"stroke","count":7},
            {"deed":"flying","count":3},
            {"deed":"redo","count":"not a number"},
            "nonsense",
            {"given":"brush-panel"},
            {"given":"a-lesson-this-build-dropped"}
        ]"#;
        let mut book = Ledger::default();
        let rows = serde_json::from_str::<Vec<serde_json::Value>>(json)
            .unwrap()
            .into_iter()
            .filter_map(|v| serde_json::from_value::<Row>(v).ok());
        for row in rows {
            match row {
                Row::Deed { deed, count } => book.tally[deed.slot()] = count,
                Row::Given { given } => {
                    book.given.insert(given);
                }
            }
        }
        assert_eq!(book.tally[Deed::Stroke.slot()], 7);
        assert_eq!(
            book.tally[Deed::Redo.slot()],
            0,
            "a count nobody can read is no count",
        );
        assert!(book.given.contains("brush-panel"));
        assert!(
            book.given.contains("a-lesson-this-build-dropped"),
            "a lesson name outlives the lesson, so a rename cannot re-teach it",
        );
    }

    /// What [`save`] writes is what [`stored`] reads.
    ///
    /// Worth asserting for [`Row`] and not for the other stored types, because it is
    /// the one that is `untagged`: the writer emits a shape and the reader guesses
    /// which variant a shape is, so the two are only joined by these field names
    /// agreeing. A tagged enum could not come apart this way.
    #[test]
    fn the_rows_written_are_the_rows_read() {
        let rows = vec![
            Row::Deed {
                deed: Deed::Stroke,
                count: 7,
            },
            Row::Given {
                given: "brush-panel".to_string(),
            },
        ];
        let json = serde_json::to_string(&rows).unwrap();
        assert_eq!(
            json,
            r#"[{"deed":"stroke","count":7},{"given":"brush-panel"}]"#
        );
        let mut book = Ledger::default();
        for row in serde_json::from_str::<Vec<Row>>(&json).unwrap() {
            match row {
                Row::Deed { deed, count } => book.tally[deed.slot()] = count,
                Row::Given { given } => {
                    book.given.insert(given);
                }
            }
        }
        assert_eq!(book.tally[Deed::Stroke.slot()], 7);
        assert!(book.given.contains("brush-panel"));
    }

    /// The threshold is a floor, not an equality: a count that ran past it while
    /// another card was up still owes the lesson.
    #[test]
    fn a_threshold_already_passed_still_comes_due() {
        let mut book = Ledger::default();
        book.tally[Deed::Redo.slot()] = 50;
        assert!(due(&book, Deed::Redo, ChromeHiding::default()).is_some());
        book.given.insert("timeline".to_string());
        assert_eq!(
            due(&book, Deed::Redo, ChromeHiding::default()),
            None,
            "a lesson is given once"
        );
    }

    /// Three lessons wait on a stroke, and they come in table order — the earliest
    /// still owed, whatever the count has run to.
    ///
    /// The property the whole queue rests on: a card passed over because another was
    /// on screen has to come back *before* the lessons behind it, or a busy stretch
    /// would silently reorder the tour into whatever the artist happened to do next.
    #[test]
    fn strokes_bring_their_lessons_in_order() {
        let key = |book: &Ledger| {
            due(book, Deed::Stroke, ChromeHiding::default()).map(|i| LESSONS[i].key)
        };
        let mut book = Ledger::default();

        book.tally[Deed::Stroke.slot()] = 1;
        assert_eq!(key(&book), None, "the first stroke owes nothing");

        book.tally[Deed::Stroke.slot()] = 4;
        assert_eq!(key(&book), Some("color-panel"));

        // Never dismissed, so the fifth stroke still owes the *first* of them.
        book.tally[Deed::Stroke.slot()] = 5;
        assert_eq!(key(&book), Some("color-panel"));

        book.given.insert("color-panel".to_string());
        assert_eq!(key(&book), Some("panel-column"));

        book.given.insert("panel-column".to_string());
        assert_eq!(key(&book), None, "the brush waits for the tenth");

        book.tally[Deed::Stroke.slot()] = 10;
        assert_eq!(key(&book), Some("brush-panel"));

        book.given.insert("brush-panel".to_string());
        assert_eq!(
            key(&book),
            None,
            "and the selection lesson is a long way off yet"
        );
    }

    /// A lesson the artist has already answered is never offered, and the one behind
    /// it comes forward in its place.
    ///
    /// The colour picker is the case the whole of [`Answer::Known`] exists for:
    /// somebody who picked a colour in their first four strokes has found the panel,
    /// and a card explaining where it is would be the tour reading out what is already
    /// on screen. Stepping *aside* rather than stalling is the same property
    /// `a_lesson_whose_subject_is_switched_off_lets_the_next_one_through` asserts, and
    /// it is worth asserting twice because the cost of getting it wrong is silent —
    /// three lessons wait behind this one.
    #[test]
    fn a_lesson_already_answered_is_never_offered() {
        let mut book = Ledger::default();
        book.tally[Deed::Stroke.slot()] = 5;
        let key = |book: &Ledger| {
            due(book, Deed::Stroke, ChromeHiding::AfterPainting).map(|i| LESSONS[i].key)
        };
        assert_eq!(key(&book), Some("color-panel"));

        book.tally[Deed::ChangedColor.slot()] = 1;
        assert_eq!(key(&book), Some("panel-column"), "the next one, at once");

        // And nothing was spent to skip it: a tally is not a card, so the lesson is
        // still merely *unowed* rather than given.
        assert!(!book.given.contains("color-panel"));
    }

    /// The eyedropper's two lessons hand over: a pick answers the one that teaches the
    /// gesture and is the deed the one that teaches its options waits on.
    ///
    /// The pair is worth a test of its own because it is the shape the whole answer
    /// mechanism is for, and both halves are load-bearing. Without the answer, somebody
    /// who takes the hint and immediately Alt-drags is left reading a card telling them
    /// to do what they are doing. Without the second lesson the deed would be counted
    /// for a dismissal alone, and the options bar — which is only ever on screen under
    /// a held modifier — would have nothing that could point at it at all.
    #[test]
    fn a_pick_answers_the_eyedropper_and_earns_its_options() {
        let mut book = Ledger::default();
        book.tally[Deed::ChangedColor.slot()] = 10;
        let owed =
            |book: &Ledger, deed| due(book, deed, ChromeHiding::default()).map(|i| LESSONS[i].key);
        assert_eq!(owed(&book, Deed::ChangedColor), Some("eyedropper"));

        // One pick and the gesture lesson is moot, however many colors were picked
        // from the panel beforehand.
        book.tally[Deed::PickedColor.slot()] = 1;
        assert_eq!(owed(&book, Deed::ChangedColor), None);
        assert_eq!(owed(&book, Deed::PickedColor), None, "a few, not one");

        // A few more and the options are worth knowing.
        book.tally[Deed::PickedColor.slot()] = 5;
        assert_eq!(owed(&book, Deed::PickedColor), Some("pick-options"));
    }

    /// An answer from **before** the card counts only where the lesson says it does.
    ///
    /// The panel column is the row that argues for the distinction and so is the row
    /// that guards it: its slice is the whole right-hand edge of the window, so a
    /// pointer on its way to anything at all wakes the stack. Read backwards, that
    /// would foreclose the lesson for practically every user — which is a tip nobody
    /// is ever shown, and the failure would look exactly like the feature working.
    #[test]
    fn a_doing_answer_does_not_reach_backwards() {
        let mut book = Ledger::default();
        book.tally[Deed::Stroke.slot()] = 5;
        book.given.insert("color-panel".to_string());
        book.tally[Deed::WokePanels.slot()] = 200;
        assert_eq!(
            due(&book, Deed::Stroke, ChromeHiding::AfterPainting).map(|i| LESSONS[i].key),
            Some("panel-column"),
            "a wake before the card is not an answer to it",
        );
        // What it *is* is the card's dismissal, once the card is up.
        let column = LESSONS
            .iter()
            .find(|l| l.key == "panel-column")
            .expect("the lesson is on the table");
        assert!(column.answer.dismisses().contains(&Deed::WokePanels));
        assert!(!column.foreclosed(&book));
    }

    /// A lesson whose subject this browser has switched off **steps aside** rather
    /// than waiting: the wake gesture is a setting now (`layout::ChromeHiding`), and
    /// its card can neither be shown — the slice it points at is never in the DOM —
    /// nor dismissed. Left owed it would stall the three lessons behind it, silently,
    /// for as long as the setting stood.
    #[test]
    fn a_lesson_whose_subject_is_switched_off_lets_the_next_one_through() {
        let mut book = Ledger::default();
        book.tally[Deed::Stroke.slot()] = 10;
        book.given.insert("color-panel".to_string());
        let key = |chrome| due(&book, Deed::Stroke, chrome).map(|i| LESSONS[i].key);
        assert_eq!(key(ChromeHiding::AfterPainting), Some("panel-column"));
        // And it is *skipped*, not spent: turn the gesture back on and it is owed
        // again, because nothing was written to the ledger to say otherwise.
        for off in [ChromeHiding::Never, ChromeHiding::WhilePainting] {
            assert_eq!(key(off), Some("brush-panel"), "{off:?}");
        }
        assert_eq!(key(ChromeHiding::AfterPainting), Some("panel-column"));
    }
}
