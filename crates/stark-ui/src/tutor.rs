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

use crate::icons::{self, icon};
use crate::layout::{PanelId, PanelLayout, chrome_class, open_panel, panel_key};
use crate::platform::{self, ElementBox};
use crate::state::AppState;
use stark_engine::InputCommand;
use stark_engine::command::{DocCommand, GestureCommand, ViewCommand};
use stark_model::document::BrushParams;

/// One key, namespaced and versioned like the other browser-local tables
/// (`crate::storage`).
const KEY_LEDGER: &str = "stark.tutor.v1";

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

/// A thing the user has done, at the grain a person would describe it in.
///
/// Deliberately not one variant per command: several commands are one deed (`Pan`,
/// `Pinch`), and one command is two deeds or none depending on what moved
/// ([`brush_deed`]). The vocabulary here is the *lessons'* — a deed exists because
/// a lesson counts it, and one that nothing counts would be a tally kept for
/// nobody (which is what `the_deeds_and_the_lessons_account_for_each_other`
/// refuses).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Deed {
    /// A brush stroke laid on the canvas and committed. A selection drag is not a
    /// stroke: it shares the gesture and produces a mask rather than paint (§6.8).
    Stroke,
    /// The brush's size or flow moved, however it was moved.
    TunedBrush,
    /// One run of panning crossed [`LONG_PAN`].
    LongPan,
    /// The brush's color moved, however it was moved.
    ChangedColor,
    /// A redo — the user's own, not the timeline transport's.
    Redo,
    /// A preset put on from the Brush panel's library. Not on the command stream —
    /// reported by the row that was clicked ([`did`]).
    AppliedPreset,
    /// An undo — the user's own, not the timeline transport's.
    Undo,
    /// A panel closed. Not on the command stream either: which panels are open is
    /// the frontend's alone, so `layout::close_panel` reports it ([`did`]).
    ClosedPanel,
    /// A stroke that snapped to a line or an ellipse (§6.9).
    ///
    /// A [`Stroke`](Self::Stroke) as well, always — the two are counted together and
    /// neither is the other's expense. It is only knowable by asking the engine what
    /// the hold found ([`Engine::assisted`](stark_engine::Engine::assisted)), and
    /// only before the gesture ends.
    AssistedStroke,
    /// A stroke that snapped to a **line** while a perspective guide was on screen
    /// (§20.6) — the state in which the grid is about to start aiming strokes,
    /// whether or not this one landed near enough for it to.
    GuidedLine,
}

impl Deed {
    /// Every deed. The order **is** the tally's slot order, so this is
    /// [`Deed::slot`]'s only authority.
    const ALL: [Deed; 10] = [
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
    ];

    /// How many there are, so the tally can be an array rather than a map.
    const COUNT: usize = Deed::ALL.len();

    /// This deed's slot in the tally.
    ///
    /// A linear scan of five, at deed rate. It reads [`Deed::ALL`] rather than
    /// restating the order as a `match` that could come to disagree with it.
    fn slot(self) -> usize {
        Deed::ALL
            .iter()
            .position(|d| *d == self)
            .expect("every Deed is in Deed::ALL")
    }

    /// The name this deed is stored under.
    ///
    /// Stable across edits of the enum, because it is what a browser wrote last
    /// week: a deed may be added, removed or reordered freely, and only *renaming*
    /// one forgets a tally.
    fn key(self) -> &'static str {
        match self {
            Deed::Stroke => "stroke",
            Deed::TunedBrush => "tune",
            Deed::LongPan => "long-pan",
            Deed::ChangedColor => "color",
            Deed::Redo => "redo",
            Deed::AppliedPreset => "preset",
            Deed::Undo => "undo",
            Deed::ClosedPanel => "closed-panel",
            Deed::AssistedStroke => "assisted",
            Deed::GuidedLine => "guided-line",
        }
    }

    /// The deed `key` names, or `None` where this build has no such deed — a line
    /// written by a version that counted something this one does not.
    fn from_key(key: &str) -> Option<Deed> {
        Deed::ALL.into_iter().find(|d| d.key() == key)
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
    /// The command rail in the top-left corner (§11) — the ☰ and Panels menus and
    /// the ⚙. Always on screen, so nothing has to reveal it and nothing can close
    /// it.
    CommandRail,
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
            Anchor::CommandRail => ".command-rail".to_string(),
            // By the id the app already gives it, rather than by its class: the
            // canvas is named once (`render::CANVAS_ID`) and this is that name,
            // so there is no second spelling to fall out of step.
            Anchor::Canvas => format!("#{}", crate::render::CANVAS_ID),
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
            // Always. Both are mounted for the life of the page and neither has a
            // control that puts it away, so these lessons are dismissed the ordinary
            // way and by nothing else.
            Anchor::CommandRail | Anchor::Canvas => true,
            Anchor::TimelineBar => (state.timeline.open)(),
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
            // Nothing to do: both are always there. Arms that say so, rather than a
            // catch-all, so a variant added later has to decide.
            Anchor::CommandRail | Anchor::Canvas => {}
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
/// Named for the picture rather than composed out of a side and an alignment: four
/// variants say exactly the four placements the lessons use, where two enums would
/// also spell several that mean nothing.
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

/// How far down its anchor a [`Side::Inside`] card sits, as a fraction of the
/// anchor's height.
///
/// Far enough down to leave the arrow pointing into the middle of the picture
/// rather than at its top edge, and not so far that the card is over the middle of
/// the work. A quarter is also about where a hand starts a stroke it means to drag
/// downward, which is the gesture the one lesson placed this way is about.
const INSIDE_DEPTH: f32 = 0.25;

/// One lesson: the deed it waits for, how many it waits for, and what it says.
struct Lesson {
    /// What the ledger records this lesson as. **Stable across edits of
    /// [`LESSONS`]** — an index would move when a lesson is inserted above it, and
    /// somebody would be shown a tip they had already dismissed.
    key: &'static str,
    /// The deed that brings it due.
    deed: Deed,
    /// How many of them. Read as "the Nth time".
    after: u32,
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
        after: 2,
        anchor: Anchor::Panel(PanelId::Color),
        side: Side::LeftAtTop,
        title: "Here is the color",
        body: "Stark starts with an empty screen and hands you the panels as you want \
               them \u{2014} this is the first. It picks in Oklab rather than on a hue \
               wheel, so the slider is lightness and the square is every color at that \
               lightness: slide down and each hue darkens by the same amount, which is \
               what mixing a shadow actually wants.",
    },
    Lesson {
        key: "panel-column",
        deed: Deed::Stroke,
        after: 3,
        anchor: Anchor::PanelColumn,
        side: Side::LeftAtMiddle,
        title: "Your panels are still here",
        body: "They stand down while you paint, so the screen goes back to being the \
               painting. Move the pointer into this strip \u{2014} anywhere down the \
               right-hand edge \u{2014} and they come straight back. On a tablet, one \
               tap in the same place.",
    },
    Lesson {
        key: "brush-panel",
        deed: Deed::Stroke,
        after: 5,
        anchor: Anchor::Panel(PanelId::Brush),
        side: Side::LeftAtTop,
        title: "Everything about the brush is here",
        body: "Size and Flow are the two you'll reach for without looking. \
               \u{201C}Edit brush\u{2026}\u{201D} opens the full set beside a live test \
               stroke you can draw on, and any brush you like can be saved to the list \
               below \u{2014} click a row to put it back in your hand.",
    },
    Lesson {
        key: "panels-menu",
        deed: Deed::ClosedPanel,
        after: 1,
        anchor: Anchor::CommandRail,
        side: Side::RightAtTop,
        title: "Nothing is lost by closing it",
        body: "The Panels button here lists all eight with a tick beside the ones on \
               screen \u{2014} click any of them to bring it back, in the slot it had. \
               What you leave open is remembered for next time, so the stack ends up \
               being the one you actually use.",
    },
    Lesson {
        key: "tune-drag",
        deed: Deed::TunedBrush,
        after: 10,
        anchor: Anchor::Panel(PanelId::Brush),
        side: Side::LeftAtTop,
        title: "Size and flow, without leaving the painting",
        body: "Hold Ctrl (\u{2318} on a Mac) and drag on the canvas: sideways sets Size, \
               up and down sets Flow. A ring at the point you pressed shows the size \
               you're asking for with the size you had behind it, so you can judge it \
               against the paint instead of against a number.",
    },
    Lesson {
        key: "quick-slots",
        deed: Deed::AppliedPreset,
        after: 5,
        anchor: Anchor::QuickSlots,
        side: Side::RightAtMiddle,
        title: "Ten brushes under your hand",
        body: "Hold a number key and this rack comes up with that brush in your hand; \
               let go and the brush you had is back. A number is a tool you borrow for \
               a few strokes rather than one you switch to. Tune a slot while you hold \
               it and the slot keeps the change; the pen's other end holds 0.",
    },
    Lesson {
        key: "shape-assist",
        deed: Deed::Undo,
        after: 10,
        // The painting, and pointing down into it. The assist has no chrome at all
        // — it is a thing you do with the pen, on the canvas — so every other
        // anchor would have put the card beside a control that has nothing to do
        // with it. This is the one lesson whose subject *is* the place it is shown.
        anchor: Anchor::Canvas,
        side: Side::Inside,
        title: "Let go of drawing it straight",
        body: "Draw a rough line or ellipse and then hold the pen still without \
               lifting: the stroke snaps to the shape you meant, and the rest of the \
               drag steers it \u{2014} lengthen the line, turn it, resize the ellipse. \
               Lift to keep it. It is still your stroke, with your pressure and your \
               brush; only the path is tidied. Settings (\u{2699}, top left) turns \
               it off if you would rather your lines stayed crooked.",
    },
    Lesson {
        key: "guides-panel",
        deed: Deed::AssistedStroke,
        after: 3,
        anchor: Anchor::Panel(PanelId::Guides),
        side: Side::LeftAtTop,
        title: "Straight is one thing. Straight *to somewhere* is another",
        body: "Add a perspective guide here \u{2014} one, two or three point \u{2014} and \
               the grid draws itself over the canvas. It is one camera behind all three, \
               so you can drag the horizon and the vanishing points around and the \
               construction stays true.",
    },
    Lesson {
        key: "perspective-assist",
        deed: Deed::GuidedLine,
        after: 1,
        anchor: Anchor::Panel(PanelId::Guides),
        side: Side::LeftAtTop,
        title: "Your held lines know about the grid",
        body: "With a guide on screen, a line you draw-and-hold near one of its axes is \
               aimed exactly down that axis \u{2014} and the rest of the drag runs the \
               end out along it instead of steering it off. The line stays where you put \
               it; only its angle comes from the grid. Hold a rough circle on a plane \
               and it comes out a circle in perspective.",
    },
    Lesson {
        key: "navigator",
        deed: Deed::LongPan,
        after: 2,
        anchor: Anchor::Panel(PanelId::Navigator),
        side: Side::LeftAtTop,
        title: "You don't have to drag that far",
        body: "The Navigator is the whole piece at a glance, with your viewport marked \
               on it. Click or drag inside it to go somewhere; drag with the right \
               button to turn the canvas \u{2014} the direction you drag becomes up.",
    },
    Lesson {
        key: "eyedropper",
        deed: Deed::ChangedColor,
        after: 5,
        anchor: Anchor::Panel(PanelId::Color),
        side: Side::LeftAtTop,
        title: "Take the color off the painting",
        body: "Hold Alt over the canvas and drag: the brush picks up whatever is under \
               the cursor, and this picker follows it. While Alt is down a small bar \
               appears with what the sample sees \u{2014} one layer, every layer, or \
               every layer over the canvas itself.",
    },
    Lesson {
        key: "timeline",
        deed: Deed::Redo,
        after: 2,
        anchor: Anchor::TimelineBar,
        side: Side::Above,
        title: "Undo is a place, not a stack",
        body: "The drawing is its own history here, so you can stand anywhere in it. \
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
        // (`input::edit_history`), so this reads false by the time it is asked.
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
        return Vec::new();
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
    // clone is an array of five and a small set of short strings.
    let mut book = ledger.peek().clone();
    let slot = deed.slot();
    book.tally[slot] = book.tally[slot].saturating_add(1);
    let owed = due(&book, deed);
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
}

/// The first lesson `deed` has brought due against `book`, if any.
fn due(book: &Ledger, deed: Deed) -> Option<usize> {
    let count = book.tally[deed.slot()];
    LESSONS
        .iter()
        .position(|l| l.deed == deed && count >= l.after && !book.given.contains(l.key))
}

/// Put a lesson away for good.
fn dismiss(state: AppState, i: usize) {
    let mut showing = state.tutor.showing;
    showing.set(None);
    let Some(lesson) = LESSONS.get(i) else {
        return;
    };
    let mut ledger = state.tutor.ledger;
    let mut book = ledger.peek().clone();
    book.given.insert(lesson.key.to_string());
    ledger.set(book);
    save(state);
}

/// What this browser has stored, or an empty ledger — a browser that has stored
/// nothing and one whose store is damaged want the same thing, which is to be
/// treated as new.
///
/// A line naming a deed this build does not count is dropped and the rest of the
/// ledger still loads, which is the property `storage`'s table format exists for.
fn stored() -> Ledger {
    let Some(rows) = crate::storage::load_table(KEY_LEDGER, parse) else {
        return Ledger::default();
    };
    let mut book = Ledger::default();
    for row in rows {
        match row {
            Row::Deed(deed, n) => book.tally[deed.slot()] = n,
            Row::Given(key) => {
                book.given.insert(key);
            }
        }
    }
    book
}

/// One readable line of the stored ledger.
enum Row {
    Deed(Deed, u32),
    Given(String),
}

/// One line of the stored ledger, or `None` for one this build cannot use — a
/// damaged record, or a deed it no longer counts. Either way it costs that line and
/// not the ledger (`crate::storage::load_table`).
fn parse(line: &str) -> Option<Row> {
    let mut fields = line.split(crate::storage::FIELD);
    match fields.next()? {
        "deed" => {
            let deed = Deed::from_key(fields.next()?)?;
            Some(Row::Deed(deed, fields.next()?.parse().ok()?))
        }
        // Not checked against `LESSONS` — see [`Ledger`].
        "given" => Some(Row::Given(fields.next()?.to_string())),
        _ => None,
    }
}

/// Persist the ledger. Called after every deed, which is at most one write per
/// [`COALESCE`] and only while the app is being used.
fn save(state: AppState) {
    let book = state.tutor.ledger.peek().clone();
    let deeds = Deed::ALL.into_iter().filter_map(|d| {
        let n = book.tally[d.slot()];
        // A deed nobody has done is the absence of a line rather than a line saying
        // zero, so the table only ever holds what actually happened.
        (n > 0).then(|| crate::storage::record(["deed", d.key(), &n.to_string()]))
    });
    let given = book
        .given
        .iter()
        .map(|k| crate::storage::record(["given", k]));
    crate::storage::save_table(KEY_LEDGER, "the tips you've seen", deeds.chain(given));
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
    let layout = use_context::<PanelLayout>();
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
        let busy = (state.canvas_active)()
            || (state.brush_editor_open)()
            || (state.preset_save_open)()
            || crate::modes::composing(state).is_some();
        if busy {
            return;
        }
        let Some(lesson) = LESSONS.get(i) else { return };
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
            dismiss(state, i);
            return;
        }
        let Some(selector) = showing
            .and_then(|i| LESSONS.get(i))
            .map(|l| l.anchor.selector())
        else {
            anchored.set(None);
            return;
        };
        anchored.set(platform::anchor_box(&selector));
        platform::on_animation_frame(move || {
            anchored.set(platform::anchor_box(&selector));
        });
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

    // Placed against the anchor's own edges, with no reading of the viewport and no
    // measurement of the card itself: the translate does the work that knowing the
    // card's width would otherwise be needed for.
    let (side, place) = match lesson.side {
        Side::LeftAtTop => (
            "side-left",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateX(-100%);",
                at.left - GAP,
                at.top
            ),
        ),
        Side::LeftAtMiddle => (
            "side-left-middle",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translate(-100%, -50%);",
                at.left - GAP,
                at.mid_y()
            ),
        ),
        Side::RightAtTop => (
            "side-right",
            format!("left: {:.1}px; top: {:.1}px;", at.right() + GAP, at.top),
        ),
        Side::RightAtMiddle => (
            "side-right-middle",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateY(-50%);",
                at.right() + GAP,
                at.mid_y()
            ),
        ),
        Side::Inside => (
            "side-inside",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateX(-50%);",
                at.mid_x(),
                at.top + at.height * INSIDE_DEPTH
            ),
        ),
        Side::Above => (
            "side-above",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translate(-50%, -100%);",
                at.mid_x(),
                at.top - GAP
            ),
        ),
    };

    rsx! {
        div {
            class: chrome_class(state, &format!("tutor-card {side}")),
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
                        let mut enabled = state.tutor.enabled;
                        enabled.set(false);
                        let mut showing = state.tutor.showing;
                        showing.set(None);
                        // Deliberately not marked as given: somebody who turned the
                        // tour off has not been taught this one, and turning it back
                        // on should still owe it to them.
                        crate::prefs::save(state);
                    },
                    "Stop tips"
                }
                button {
                    class: "chip tutor-done",
                    onclick: move |_| dismiss(state, i),
                    {icon(icons::DONE)}
                    "Got it"
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every deed feeds a lesson.
    ///
    /// The direction worth asserting: a deed nothing counts is a tally kept for
    /// nobody, which is exactly what a lesson removed without its deed leaves
    /// behind. The other direction is the compiler's — a lesson names a `Deed`.
    #[test]
    fn the_deeds_and_the_lessons_account_for_each_other() {
        for deed in Deed::ALL {
            assert!(
                LESSONS.iter().any(|l| l.deed == deed),
                "{deed:?} is counted and no lesson waits for it",
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

    /// The lessons that may fire on a **first** deed, and everything else waits for
    /// a second.
    ///
    /// An exception list rather than a blanket rule, because there are two — and an
    /// *exception* list rather than a list of the ordinary ones, so a lesson added
    /// later is held to the strict rule by default. That is the safe direction: the
    /// cost of getting this wrong is a tip in somebody's first minute.
    ///
    /// - `panels-menu` answers a question its own deed **raises**. Close a panel and
    ///   "where did that go?" is immediate; answering on the second close would be
    ///   answering it late, and the artist would have spent the gap thinking the
    ///   panel was gone.
    /// - `perspective-assist` waits on a deed nobody performs by accident: you have
    ///   to have made a guide, left it visible, drawn a stroke *and* held it still.
    ///   Having done all four once is stronger evidence of intent than ten of
    ///   anything else on this list.
    const AT_ONCE: [&str; 2] = ["panels-menu", "perspective-assist"];

    /// A lesson at one fires on the user's first try, which the design is against
    /// everywhere it has not said otherwise — see [`AT_ONCE`].
    #[test]
    fn no_lesson_fires_on_a_first_try() {
        for l in LESSONS {
            if AT_ONCE.contains(&l.key) {
                assert_eq!(l.after, 1, "{} is listed as firing at once", l.key);
                continue;
            }
            assert!(l.after >= 2, "{} fires after {}", l.key, l.after);
        }
    }

    /// Every name in [`AT_ONCE`] is a lesson. A key left behind by a renamed or
    /// deleted lesson would quietly exempt nothing and be impossible to notice.
    #[test]
    fn the_exceptions_name_real_lessons() {
        for key in AT_ONCE {
            assert!(
                LESSONS.iter().any(|l| l.key == key),
                "{key} is exempted and does not exist",
            );
        }
    }

    /// [`Deed::slot`] and [`Deed::key`] both have to be total and one-to-one — the
    /// first indexes the tally, the second is what a stored line is found by.
    #[test]
    fn every_deed_has_its_own_slot_and_its_own_name() {
        let slots: HashSet<usize> = Deed::ALL.into_iter().map(Deed::slot).collect();
        assert_eq!(slots.len(), Deed::COUNT);
        let keys: HashSet<&str> = Deed::ALL.into_iter().map(Deed::key).collect();
        assert_eq!(keys.len(), Deed::COUNT);
        for d in Deed::ALL {
            assert_eq!(Deed::from_key(d.key()), Some(d));
        }
        assert_eq!(Deed::from_key("a deed no build has"), None);
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
    /// loads, and the half it understands survives — the rule
    /// `crate::storage`'s format exists for, asked of this parser.
    #[test]
    fn an_unknown_line_costs_itself_and_not_the_ledger() {
        let text = "deed|stroke|7\n\
                    deed|flying|3\n\
                    deed|redo|not-a-number\n\
                    nonsense\n\
                    given|brush-panel\n\
                    given|a-lesson-this-build-dropped";
        let mut book = Ledger::default();
        for row in text.lines().filter_map(parse) {
            match row {
                Row::Deed(d, n) => book.tally[d.slot()] = n,
                Row::Given(k) => {
                    book.given.insert(k);
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

    /// The threshold is a floor, not an equality: a count that ran past it while
    /// another card was up still owes the lesson.
    #[test]
    fn a_threshold_already_passed_still_comes_due() {
        let mut book = Ledger::default();
        book.tally[Deed::Redo.slot()] = 50;
        assert!(due(&book, Deed::Redo).is_some());
        book.given.insert("timeline".to_string());
        assert_eq!(due(&book, Deed::Redo), None, "a lesson is given once");
    }

    /// Three lessons wait on a stroke, and they come in table order — the earliest
    /// still owed, whatever the count has run to.
    ///
    /// The property the whole queue rests on: a card passed over because another was
    /// on screen has to come back *before* the lessons behind it, or a busy stretch
    /// would silently reorder the tour into whatever the artist happened to do next.
    #[test]
    fn strokes_bring_their_lessons_in_order() {
        let key = |book: &Ledger| due(book, Deed::Stroke).map(|i| LESSONS[i].key);
        let mut book = Ledger::default();

        book.tally[Deed::Stroke.slot()] = 1;
        assert_eq!(key(&book), None, "the first stroke owes nothing");

        book.tally[Deed::Stroke.slot()] = 2;
        assert_eq!(key(&book), Some("color-panel"));

        // Never dismissed, so the third stroke still owes the *first* of them.
        book.tally[Deed::Stroke.slot()] = 3;
        assert_eq!(key(&book), Some("color-panel"));

        book.given.insert("color-panel".to_string());
        assert_eq!(key(&book), Some("panel-column"));

        book.given.insert("panel-column".to_string());
        assert_eq!(key(&book), None, "the brush waits for the fifth");

        book.tally[Deed::Stroke.slot()] = 5;
        assert_eq!(key(&book), Some("brush-panel"));

        book.given.insert("brush-panel".to_string());
        assert_eq!(key(&book), None, "and then a stroke owes nothing at all");
    }
}
