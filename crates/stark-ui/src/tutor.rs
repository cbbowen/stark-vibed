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
//! The second is **which gesture wrote the brush**, and it matters because two of
//! the lessons teach gestures that produce the very deed they are counted by: an
//! Alt-drag off the painting is a color change, and an accelerator drag is a size
//! change. Counted naively, the tour would wait for somebody to use the eyedropper
//! five times and then offer to explain the eyedropper. So the two gestures say so
//! while they run ([`via_shortcut`]), and a brush write made under that flag is not
//! counted at all — which turns the flaw into the feature the tour most wanted:
//! **it never teaches a gesture you already use.** Somebody who only ever
//! eyedroppers never accumulates the deed and is never told about it.
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
}

impl Deed {
    /// Every deed. The order **is** the tally's slot order, so this is
    /// [`Deed::slot`]'s only authority.
    const ALL: [Deed; 5] = [
        Deed::Stroke,
        Deed::TunedBrush,
        Deed::LongPan,
        Deed::ChangedColor,
        Deed::Redo,
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
    /// Set while the brush is being written by one of the gestures a lesson
    /// *teaches* — see [`via_shortcut`], which is the only thing that writes it.
    pub via_shortcut: Signal<bool>,
}

/// What a lesson points at.
///
/// Two variants, because two is what the lessons below actually name; a third
/// arrives with the lesson that needs it rather than in advance of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Anchor {
    /// A floating tool panel, by the identity it wears in the DOM ([`panel_key`]).
    Panel(PanelId),
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
            Anchor::TimelineBar => crate::panels::timeline::set_open(state, true),
        }
    }
}

/// Which side of its anchor the card sits on.
///
/// The DOM does not say this and cannot be asked: which side has room is a fact
/// about where that chrome lives in the window, which is the lesson author's to
/// know. Two, because the panels are a column down the right and the timeline bar
/// is across the bottom.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    /// Left of the anchor, its top edge aligned with the anchor's.
    ///
    /// Aligned rather than centred so a tall panel cannot push the card off the top
    /// of the window: the stack hangs from the top of its column, so the top edge is
    /// the one always on screen.
    Left,
    /// Above the anchor, centred on it.
    Above,
}

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
/// The counts are set from what each deed *costs* to keep doing the hard way.
/// Three strokes is barely a commitment and the Brush panel is where everything
/// about a brush is; ten trips to the size slider is somebody who has decided this
/// is how they work, which is exactly when the drag is worth knowing. None of them
/// fires on a first try, deliberately: a tip in the first minute is noise, because
/// nothing has been wanted yet.
static LESSONS: &[Lesson] = &[
    Lesson {
        key: "brush-panel",
        deed: Deed::Stroke,
        after: 3,
        anchor: Anchor::Panel(PanelId::Brush),
        side: Side::Left,
        title: "Everything about the brush is here",
        body: "Size and Flow are the two you'll reach for without looking. \
               \u{201C}Edit brush\u{2026}\u{201D} opens the full set beside a live test \
               stroke you can draw on, and any brush you like can be saved to the list \
               below \u{2014} click a row to put it back in your hand.",
    },
    Lesson {
        key: "tune-drag",
        deed: Deed::TunedBrush,
        after: 10,
        anchor: Anchor::Panel(PanelId::Brush),
        side: Side::Left,
        title: "Size and flow, without leaving the painting",
        body: "Hold Ctrl (\u{2318} on a Mac) and drag on the canvas: sideways sets Size, \
               up and down sets Flow. A ring at the point you pressed shows the size \
               you're asking for with the size you had behind it, so you can judge it \
               against the paint instead of against a number.",
    },
    Lesson {
        key: "navigator",
        deed: Deed::LongPan,
        after: 2,
        anchor: Anchor::Panel(PanelId::Navigator),
        side: Side::Left,
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
        side: Side::Left,
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

/// Say whether one of the gestures a lesson teaches is writing the brush right now.
///
/// The one thing the tour asks of the rest of the app, and it earns the exception:
/// which gesture a `SetBrush` came out of is not on the command and cannot be
/// derived from it, in the way the dwell behind [`GestureCommand::Hold`] is not on
/// one either (§6.9). Two callers, and both are the *subject* of a lesson — the
/// tuning drag (`input::Tune`, §18.1.9) and the eyedropper (`input::pick_color`,
/// §18.0.2).
///
/// A brush write made while this is set is not counted, so fluency with a gesture
/// is never mistaken for needing to be taught it. Called in pairs around the whole
/// gesture rather than at each write, so there is one place to set it and one to
/// clear it however many commands run between.
///
/// A flag left set by a gesture that ended without saying so costs counting, never
/// a wrong card — the failure direction to have, since the other one is a card
/// nobody asked for.
pub fn via_shortcut(state: AppState, in_flight: bool) {
    let mut flag = state.tutor.via_shortcut;
    // Guarded rather than set flat: the tuning drag clears this on every release the
    // canvas sees, and a signal set to the value it holds still wakes its readers.
    if *flag.peek() != in_flight {
        flag.set(in_flight);
    }
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
    if let Some(deed) = read(state, command) {
        tally(state, deed);
    }
}

/// Which deed, if any, `command` is a report of.
///
/// Split from [`observe`] because this is the half with the judgement in it, and
/// the half that writes nothing but the pan run it is obliged to accumulate.
fn read(state: AppState, command: &InputCommand) -> Option<Deed> {
    match command {
        // A commit, not a start: a stroke abandoned by a second finger (§18.1.7)
        // arrives as `Cancel` and left no paint, and counting it would be counting
        // a mark that is not on the canvas.
        InputCommand::Gesture(GestureCommand::End) => {
            // Which tool the gesture was is not on the command — but it is on the
            // projection, which is where the whole app asks (§6.8).
            let tool = state.obs.peek().as_ref().map(|o| o.tool)?;
            (!tool.is_selection()).then_some(Deed::Stroke)
        }
        // Not the transport's. The playback loop drives the playhead with this very
        // command (§18.2.4) and would otherwise score eight redos a second; the
        // user's own redo stops playback *before* it dispatches
        // (`input::edit_history`), so this reads false by the time it is asked.
        InputCommand::Doc(DocCommand::Redo) => {
            (!crate::panels::timeline::is_playing(state)).then_some(Deed::Redo)
        }
        // Not one the gesture a lesson teaches is making — see [`via_shortcut`].
        InputCommand::View(ViewCommand::SetBrush(_)) if *state.tutor.via_shortcut.peek() => None,
        InputCommand::View(ViewCommand::SetBrush(next)) => {
            let held = state.obs.peek().as_ref().map(|o| o.brush)?;
            brush_deed(&held, next)
        }
        // Both shapes of pan the app can make: the one-pointer drags and the last
        // finger of a two-finger gesture arrive as `Pan`, and the pair itself as
        // `Pinch`, whose translation is the distance between its two anchors. A
        // `Zoom` is deliberately not travel — same gesture family, different act.
        InputCommand::View(ViewCommand::Pan { delta }) => pan(state, delta.length()),
        InputCommand::View(ViewCommand::Pinch { anchor, to, .. }) => {
            pan(state, (*to - *anchor).length())
        }
        _ => None,
    }
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
        Side::Left => (
            "side-left",
            format!(
                "left: {:.1}px; top: {:.1}px; transform: translateX(-100%);",
                at.left - GAP,
                at.top
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

    /// A lesson at one would fire on the user's first try, which is the one thing
    /// the whole design is against.
    #[test]
    fn no_lesson_fires_on_a_first_try() {
        for l in LESSONS {
            assert!(l.after >= 2, "{} fires after {}", l.key, l.after);
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
        book.tally[Deed::Stroke.slot()] = 50;
        assert!(due(&book, Deed::Stroke).is_some());
        book.given.insert("brush-panel".to_string());
        assert_eq!(due(&book, Deed::Stroke), None, "a lesson is given once");
    }
}
