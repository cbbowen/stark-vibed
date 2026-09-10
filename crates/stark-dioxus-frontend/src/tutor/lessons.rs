//! What a deed is, and which lesson it owes (§24.1, §24.5).
//!
//! The vocabulary of deeds, the table of lessons, and the ledger they are counted into,
//! with its stored rows. Nothing here reads a signal or the DOM, so which lesson a count
//! owes is decided — and tested — as a function of values.

use std::collections::HashSet;

use stark_model::document::BrushParams;
use stark_ui::brush_editor::Section;
use stark_ui::panels::PanelId;
use stark_ui::prefs::ChromeHiding;
use stark_ui::storage::Store;
use strum::{EnumCount, VariantArray};

use crate::brush_editor::BrushPart;

/// A thing the user has done, described the way a person would describe it.
///
/// Not one variant per command: `Pan` and `Pinch` are one deed, and a `SetBrush` is a size
/// change, a color change or neither depending on what moved ([`brush_deed`]). A deed
/// exists because a lesson counts it or answers to it
/// (`tests::the_deeds_and_the_lessons_account_for_each_other`).
///
/// Each variant's stored name is a `rename` spelled out rather than derived from the
/// identifier, so a deed may be added, removed or reordered freely and only editing the
/// string forgets a tally (§24.4). Declaration order is only the in-memory slot order
/// ([`Deed::slot`]).
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    strum::VariantArray,
    strum::EnumCount,
)]
pub enum Deed {
    /// A brush stroke committed. A selection drag is not one: it builds a mask rather than
    /// paint (§6.8).
    #[serde(rename = "stroke")]
    Stroke,
    /// The brush's size or flow moved, however it was moved.
    #[serde(rename = "tune")]
    TunedBrush,
    /// One run of panning crossed `reader::LONG_PAN`.
    #[serde(rename = "long-pan")]
    LongPan,
    /// The brush's color moved, however it was moved.
    #[serde(rename = "color")]
    ChangedColor,
    /// A redo — the user's own, not the timeline transport's.
    #[serde(rename = "redo")]
    Redo,
    /// A preset put on from the Brush panel's library. Reported by the row
    /// ([`did`](super::did)): the `SetBrush` it leads to cannot say a row was clicked, and
    /// the quick slots emit one of exactly the same shape.
    #[serde(rename = "preset")]
    AppliedPreset,
    /// An undo — the user's own, not the timeline transport's.
    #[serde(rename = "undo")]
    Undo,
    /// A panel closed. Which panels are open is the frontend's alone, so
    /// `layout::close_panel` reports it ([`did`](super::did)).
    #[serde(rename = "closed-panel")]
    ClosedPanel,
    /// A stroke that snapped to a line or an ellipse (§6.9), and always a
    /// [`Stroke`](Self::Stroke) as well. Knowable only by asking the engine
    /// ([`Engine::assisted`](stark_engine::Engine::assisted)) before the gesture ends.
    #[serde(rename = "assisted")]
    AssistedStroke,
    /// A stroke that snapped to a **line** while a perspective guide was visible (§20.6):
    /// the state in which the grid is about to aim strokes, whether or not this one landed
    /// near an axis.
    #[serde(rename = "guided-line")]
    GuidedLine,
    /// A marquee or lasso gesture that committed a selection (§6.8). Under Fill the same
    /// drag lays paint instead, which is not a selection (§18.0.4).
    #[serde(rename = "selection")]
    Selection,
    /// The brush editor opened. The dialog is frontend state, so its button reports it
    /// ([`did`](super::did)).
    #[serde(rename = "brush-editor")]
    OpenedBrushEditor,
    /// The panel column reached into, waking the sleeping stack (§11).
    ///
    /// Reported by the slice the pointer lands in (`layout::reach_for_panels`) rather than
    /// by `layout::wake_panels`, which `open_panel` and the tour's own release also go
    /// through — and neither of those is the artist reaching for anything.
    #[serde(rename = "woke-panels")]
    WokePanels,
    /// A color sampled off the painting (§18.0.2), reported by `input::pick_color` where a
    /// sample actually landed.
    ///
    /// Not read as the [`ChangedColor`](Self::ChangedColor) it also is: that write is
    /// inside a [`not_reaching`](super::not_reaching) bracket, so the eyedropper is counted
    /// as a gesture and never as an adjustment of the brush.
    #[serde(rename = "picked-color")]
    PickedColor,
    /// A layer added (§14).
    /// [`DocCommand::AddLayer`](stark_engine::command::DocCommand::AddLayer) is minted by
    /// the Layers panel's button and the command that shares it alone — an image (§23) and
    /// a matte (§15) carry commands of their own.
    #[serde(rename = "added-layer")]
    AddedLayer,
    /// The navigator's miniature clicked or dragged (§11), read off
    /// [`ViewCommand::CenterOn`](stark_engine::command::ViewCommand::CenterOn): absolute
    /// where every other way of travelling is a delta, so nothing else emits one — and a
    /// navigator drag is never counted as travel.
    #[serde(rename = "navigator")]
    UsedNavigator,
}

impl Deed {
    /// This deed's slot in a tally: its position in the derived variant list, so the order
    /// is not restated anywhere it could disagree.
    pub(super) fn slot(self) -> usize {
        Deed::VARIANTS
            .iter()
            .position(|d| *d == self)
            .expect("every Deed is a variant of Deed")
    }
}

/// What the tour remembers between visits: how often each deed has been done, and which
/// lessons have been given (§24.4).
///
/// The halves forget differently. A tally under a name this build does not know is
/// dropped — a deed nothing counts feeds no lesson — while a lesson name is kept whatever
/// [`LESSONS`] now says, so a release that renames a neighbour cannot re-teach a tip.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Ledger {
    /// How many times each deed has been done, by [`Deed::slot`].
    tally: [u32; Deed::COUNT],
    /// The keys of the lessons already given.
    given: HashSet<String>,
}

impl Ledger {
    /// How many times `deed` has been done.
    pub(super) fn count(&self, deed: Deed) -> u32 {
        self.tally[deed.slot()]
    }

    /// Count one more `deed`.
    pub(super) fn add(&mut self, deed: Deed) {
        let slot = deed.slot();
        self.tally[slot] = self.tally[slot].saturating_add(1);
    }

    /// Whether the lesson stored as `key` has been given.
    pub(super) fn is_given(&self, key: &str) -> bool {
        self.given.contains(key)
    }

    /// Record the lesson stored as `key` as given.
    pub(super) fn give(&mut self, key: &str) {
        self.given.insert(key.to_owned());
    }

    /// The ledger `rows` were stored from. A lesson name is not checked against
    /// [`LESSONS`] — see the type.
    pub(super) fn from_rows(rows: impl IntoIterator<Item = Row>) -> Self {
        let mut book = Self::default();
        for row in rows {
            match row {
                Row::Deed { deed, count } => book.tally[deed.slot()] = count,
                Row::Given { given } => {
                    book.given.insert(given);
                }
            }
        }
        book
    }

    /// The rows to store this ledger as. A deed nobody has done is the absence of a row
    /// rather than a row saying zero, so the record holds only what happened.
    pub(super) fn rows(&self) -> Vec<Row> {
        let deeds = Deed::VARIANTS.iter().filter_map(|&deed| {
            let count = self.count(deed);
            (count > 0).then_some(Row::Deed { deed, count })
        });
        let given = self
            .given
            .iter()
            .map(|key| Row::Given { given: key.clone() });
        deeds.chain(given).collect()
    }
}

/// One stored row of the [`Ledger`]: a deed and its count, or a lesson already given.
///
/// A list of rows rather than one record with two fields, because only a row can be dropped
/// on its own (`storage::load_list`): an unreadable tally must not take the `given` names
/// beside it down too. Untagged — the two shapes share no field, so neither is ambiguous.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub(crate) enum Row {
    Deed { deed: Deed, count: u32 },
    Given { given: String },
}

impl stark_ui::storage::Entry for Row {
    const STORE: Store = Store::Tutor;
}

/// What a lesson points at — and, through the card's `Anchor::reveal`, what it opens.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Anchor {
    /// A floating tool panel, by the identity it wears in the DOM (`layout::panel_key`).
    Panel(PanelId),
    /// The full-height wake slice down the right that brings the sleeping panels back
    /// (§11). **Invisible**, which is the point: the card is the only thing that can say
    /// where it is.
    PanelColumn,
    /// The quick-brush rack down the left (§18.1.8).
    QuickSlots,
    /// The navigator's miniature, in the bottom-left corner (§11).
    Navigator,
    /// The command rail in the top-left corner (§11) — always on screen, so nothing reveals
    /// it and nothing closes it.
    CommandRail,
    /// A part of the brush editor's dialog: the one anchor inside a dialog
    /// ([`inside_dialog`](Self::inside_dialog)).
    BrushEditor(BrushPart),
    /// The painting itself, for a lesson about a gesture made on the canvas — which has no
    /// control to stand beside. Pointed at from [`Side::Inside`].
    Canvas,
    /// Timeline mode's bar across the foot of the window (§18.2.4).
    TimelineBar,
}

impl Anchor {
    /// Whether this anchor lives inside a dialog.
    ///
    /// Every other card stands down while a modal is up, since a modal covers everything a
    /// card could point at — which stops being true exactly when the card points at the
    /// dialog (§24.3).
    pub(super) fn inside_dialog(self) -> bool {
        matches!(self, Anchor::BrushEditor(_))
    }

    /// Whether a card pointing here holds the panel stack up while it is on screen, and
    /// wakes it on the way down (§24.3). Only a panel does: the lesson about the panel
    /// *column* is unteachable with the stack already up.
    pub(super) fn holds_panels(self) -> bool {
        matches!(self, Anchor::Panel(_))
    }

    /// The window edge this anchor's box stands against — the one there is no room beside —
    /// or `None` for an anchor clear of both.
    ///
    /// Test-only: a placement is *stated* on the lesson ([`Side`]), and this exists to check
    /// the statement (`tests::a_card_is_never_hung_into_the_edge_it_stands_on`).
    #[cfg(test)]
    fn against(self) -> Option<Edge> {
        match self {
            // The stack hangs from the top of its column, as the rail does from the window's.
            Anchor::Panel(_) | Anchor::CommandRail => Some(Edge::Top),
            // Both sit on the foot of the window (`.left-chrome`, `.timeline-bar`).
            Anchor::Navigator | Anchor::TimelineBar => Some(Edge::Bottom),
            // The column and the canvas run the window's height, the rack takes what the
            // navigator under it leaves, and a dialog is centred.
            Anchor::PanelColumn | Anchor::QuickSlots | Anchor::Canvas => None,
            Anchor::BrushEditor(_) => None,
        }
    }
}

/// Where the card sits relative to its anchor (§24.3).
///
/// Stated on the lesson because the DOM cannot say it: which side has room, and whether the
/// anchor has a meaningful top edge, are facts about where that chrome lives. Named for the
/// picture rather than composed from a side and an alignment, which would also spell
/// placements that mean nothing.
///
/// A card hangs from whichever of its anchor's edges the window cannot push off screen.
/// Sideways a card too wide narrows to fit (`anchor::room_left`); vertically nothing rescues
/// it, since narrowing only makes it taller — so the placement is the whole of the fix, and
/// `tests::a_card_is_never_hung_into_the_edge_it_stands_on` is the guard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Side {
    /// Left of the anchor, top edges level: a panel, which hangs from the top of its column.
    LeftAtTop,
    /// Left of the anchor, centred: the panel column, a whole edge of the window with no
    /// meaningful top.
    LeftAtMiddle,
    /// Right of the anchor, top edges level: the command rail, a box that hugs its contents.
    RightAtTop,
    /// Right of the anchor, centred: the quick-brush rack, whose box runs to the foot of the
    /// window and centres its rows, so its top edge is level with nothing drawn.
    RightAtMiddle,
    /// Right of the anchor, **bottom** edges level: the navigator's miniature, which stands
    /// [`EDGE`](crate::anchor::EDGE) off the foot of the window.
    ///
    /// The one placement whose arrow is not a constant: the miniature's height is the
    /// artwork's aspect (`navigator::MAX_HEIGHT`), so the card hands the stylesheet how far
    /// above the shared bottom edge the anchor's middle lies (`--tutor-reach`).
    RightAtBottom,
    /// Above the anchor, centred on it.
    Above,
    /// **Over** the anchor, centred across it and `card::INSIDE_DEPTH` of the way down,
    /// pointing down into it: for an anchor that is a place rather than a control — the
    /// canvas, where the gesture being described happens.
    Inside,
}

/// An edge of the window, for the one question asked about both: which one a piece of
/// chrome stands against, and which one a card grows toward.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Edge {
    Top,
    Bottom,
}

#[cfg(test)]
impl Side {
    /// Which edge of the window the card reaches toward from where it is hung. `None` for a
    /// placement centred on its anchor, which reaches both ways and so needs an anchor clear
    /// of both edges.
    fn grows(self) -> Option<Edge> {
        match self {
            // Hung level with the anchor's top and hanging down — `Inside` lower still.
            Side::LeftAtTop | Side::RightAtTop | Side::Inside => Some(Edge::Bottom),
            // Hung level with the anchor's bottom, or clear above it, and rising.
            Side::RightAtBottom | Side::Above => Some(Edge::Top),
            Side::LeftAtMiddle | Side::RightAtMiddle => None,
        }
    }
}

/// What answers a lesson besides its own button: the deeds that **are** the thing it teaches
/// (§24.3).
///
/// Stated on the lesson rather than derived from its deed, because what earns a lesson is
/// evidence the artist wants the thing and what answers it is evidence they have it: the
/// color panel is earned by painting and answered by picking a color.
///
/// The live variants differ only in how far back an answer counts. The test for
/// [`Known`](Self::Known) is whether having done the deed **proves they found the thing the
/// card points at**: picking a color does; drifting across the invisible wake slice, which a
/// pointer on its way anywhere crosses, does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Answer {
    /// The button, and nothing else — written out rather than an empty list, so a lesson
    /// added later has to decide. Usually because the gesture it teaches is bracketed by
    /// [`not_reaching`](super::not_reaching) and produces no deed to answer with.
    Button,
    /// These deeds, from the moment the lesson comes due — whether its card is on screen
    /// yet or still waiting.
    Doing(&'static [Deed]),
    /// These deeds, ever: [`Doing`](Self::Doing), and a count already standing means the
    /// lesson is never offered. Skipped rather than given ([`due`]), so it is owed again if
    /// that count turns out to have been somebody else's browser.
    Known(&'static [Deed]),
}

impl Answer {
    /// The deeds that take the lesson's card down — both live variants.
    pub(super) fn dismisses(self) -> &'static [Deed] {
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

/// One lesson: the deed it waits for, how many, what answers it, and what it says.
pub(super) struct Lesson {
    /// What the ledger records it as. Stable across edits of [`LESSONS`], where an index
    /// would move when a lesson is inserted above it.
    pub(super) key: &'static str,
    /// The deed that brings it due.
    pub(super) deed: Deed,
    /// How many of them: "the Nth time".
    pub(super) after: u32,
    /// What the artist can do to say they already know this.
    pub(super) answer: Answer,
    /// What it points at, and so what it opens.
    pub(super) anchor: Anchor,
    /// Which side of the anchor the card sits on.
    pub(super) side: Side,
    /// The one line that says what the lesson is about.
    pub(super) title: &'static str,
    /// What to *do*, then the thing about Stark that makes it worth doing — a tip that only
    /// names a shortcut is a keyboard reference, and the menus already carry one.
    pub(super) body: &'static str,
}

impl Lesson {
    /// Whether this lesson's subject exists for this browser at all. Only the wake slice
    /// can be switched off (`ChromeHiding`), and a tip about a gesture somebody turned off
    /// is worse than none.
    fn applies(&self, chrome: ChromeHiding) -> bool {
        match self.anchor {
            Anchor::PanelColumn => chrome.sleeps(),
            _ => true,
        }
    }

    /// Whether the artist has already shown they know this ([`Answer::Known`]). A tally
    /// rather than the `given` set, since the question is what was done, not what was
    /// shown.
    fn foreclosed(&self, book: &Ledger) -> bool {
        self.answer
            .forecloses()
            .iter()
            .any(|&deed| book.count(deed) > 0)
    }
}

/// Every lesson, in the order they are offered when two come due at once (§24.5).
///
/// A lesson is a row here, and adding one costs no code elsewhere unless it counts a deed
/// nothing counts yet. **Order decides ties**: a deed owing several lessons gives the
/// earliest still owed, so a card passed over while another was up comes back before the
/// ones behind it.
///
/// The counts are set from what each deed costs to keep doing the hard way, and almost none
/// fires on a first try — a tip in the first minute is noise (`tests::AT_ONCE` names the
/// exceptions).
pub(super) static LESSONS: &[Lesson] = &[
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
        // lists is a paragraph nobody needs read to them. (Saving a preset is the
        // editor's act, reached through the same button.)
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
        body: "Hold Ctrl (\u{2318} on a Mac) and drag on the canvas: right and left change \
               Size, up and down change Flow.",
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
        body: "Hold a number key to paint with a brush from this rack. Release to return to your previous brush, or tap the number twice to stay on its brush. Click a preset or change Size or Flow while you hold the key, and the slot keeps it.",
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
        anchor: Anchor::BrushEditor(BrushPart::Group(Section::Tip)),
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
        anchor: Anchor::BrushEditor(BrushPart::Group(Section::Effect)),
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
        anchor: Anchor::BrushEditor(BrushPart::Group(Section::Color)),
        side: Side::RightAtTop,
        title: "Color dynamics \u{2014} the wobble that reads as pigment",
        body: "The color wanders across the width of the tip and along the stroke, \
               following a noise field rather than a random number per stamp. That is \
               why it reads as pigment sitting unevenly instead of as static: \
               neighbouring bristles stay neighbours. A little of this is what keeps a \
               flat fill from looking printed.",
    },
    // Anchored at the effect chips rather than at the Wet section itself,
    // because the section exists only while the brush *is* wet — a card pointing
    // at a box the default paint brush never renders would be a card in the
    // corner with nothing beside it. The chips are where the switch lives, which
    // is also what the lesson teaches.
    Lesson {
        key: "be-wet",
        deed: Deed::OpenedBrushEditor,
        after: 1,
        answer: Answer::Button,
        anchor: Anchor::BrushEditor(BrushPart::Group(Section::Effect)),
        side: Side::RightAtTop,
        title: "Wet \u{2014} moving paint that is already there",
        body: "A Paint brush lays paint; a Wet brush also moves what is on the \
               canvas. Switch the effect to Wet and its section appears: Lift takes \
               canvas paint onto the tip so the next stretch of stroke carries it \
               \u{2014} a smudge, and with no paint of its own a palette knife. Bleed \
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

/// What moved between two brushes, as a deed — or `None` where the answer is "too much to
/// be one" (§24.2).
///
/// The test is **confinement**, not difference: a size change counts when the brush is
/// otherwise untouched, a color change when the same is true of the color. That tells the
/// size slider and the color picker apart from a preset click or a quick slot, which move a
/// dozen fields at once. "Everything else is equal" is one `==` against a copy with the
/// candidate fields grafted on, so a brush parameter added later cannot fall out of the rule.
///
/// The hand's color rides beside the params on `SetBrush`, because an erasing brush carries
/// none of its own; a color picked with the eraser held moves only that side channel, and
/// still has to read as a color change.
pub(super) fn brush_deed(
    was: &BrushParams,
    was_color: [f32; 3],
    now: &BrushParams,
    now_color: [f32; 3],
) -> Option<Deed> {
    let mut tuned = *was;
    tuned.size = now.size;
    // The flow grafted onto the effect `was` has, so a change of *effect* is never a tune:
    // `set_flow` writes the old kind's rate, and the `==` below sees the kinds differ.
    tuned.effect.set_flow(now.effect.flow());
    if tuned == *now
        && was_color == now_color
        && (was.size != now.size || was.effect.flow() != now.effect.flow())
    {
        return Some(Deed::TunedBrush);
    }

    let mut colored = *was;
    if let Some(p) = colored.paint_mut() {
        p.color = now.paint().map_or(p.color, |n| n.color);
    }
    if let Some(w) = colored.wet_mut() {
        w.color = now.wet().map_or(w.color, |n| n.color);
    }
    if colored == *now && was_color != now_color {
        return Some(Deed::ChangedColor);
    }

    None
}

/// The first lesson `deed` has brought due against `book`, if any (§24.3).
///
/// Skips a lesson whose subject is switched off ([`Lesson::applies`]) or already answered
/// ([`Lesson::foreclosed`]), and neither skip writes anything: the next lesson the deed owes
/// comes forward in the same breath, and the skipped one is owed again if the judgement
/// changes. A lesson that could neither be shown nor passed would stall its whole deed,
/// silently.
///
/// `chrome` is passed rather than read off the app, so which lesson is owed stays a function
/// of values.
pub(super) fn due(book: &Ledger, deed: Deed, chrome: ChromeHiding) -> Option<usize> {
    let count = book.count(deed);
    LESSONS.iter().position(|l| {
        l.deed == deed
            && count >= l.after
            && !book.is_given(l.key)
            && l.applies(chrome)
            && !l.foreclosed(book)
    })
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
        for &deed in Deed::VARIANTS {
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
    /// fit (`anchor::room_left`); a card too *tall* has no equivalent — narrowing one only
    /// makes it taller — so a placement that hangs a card into an edge is simply a
    /// card half off the screen.
    ///
    /// The regression it exists for happened: the navigator's lesson kept
    /// [`Side::RightAtMiddle`] when the miniature moved out of the panel stack into
    /// the bottom-left corner, and a card centred on a box that stands `anchor::EDGE` off
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

    /// Every deed is stored under a name of its own — what a stored row is found by.
    ///
    /// The names are `rename` attributes, so serde round-trips them by construction;
    /// what it does *not* check is that no two variants were given the same string,
    /// which would silently merge two tallies.
    #[test]
    fn every_deed_has_its_own_stored_name() {
        let names: HashSet<String> = Deed::VARIANTS
            .iter()
            .map(|d| serde_json::to_string(d).unwrap())
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
        // The hand's color as `update_brush` sends it beside the params: the
        // paint side's own.
        let color = |b: &BrushParams| b.paint().expect("these brushes paint").color;
        // Recolor both places a color pick writes: the paint effect's pigment
        // and the side channel it is projected onto.
        let recolor = |b: &mut BrushParams| {
            b.paint_mut().expect("these brushes paint").color = [1.0, 0.0, 0.0];
        };

        let mut bigger = was;
        bigger.size = was.size * 2.0;
        assert_eq!(
            brush_deed(&was, color(&was), &bigger, color(&bigger)),
            Some(Deed::TunedBrush)
        );

        let mut looser = was;
        looser.effect.set_flow(was.effect.flow() + 0.5);
        assert_eq!(
            brush_deed(&was, color(&was), &looser, color(&looser)),
            Some(Deed::TunedBrush)
        );

        let mut red = was;
        recolor(&mut red);
        assert_eq!(
            brush_deed(&was, color(&was), &red, color(&red)),
            Some(Deed::ChangedColor)
        );

        // A color picked while the eraser is held: the params move not at all —
        // an erasing brush carries no pigment — and the side channel alone
        // still reads as the eyedropper.
        let erasing = BrushParams {
            effect: stark_model::document::BrushEffect::Erase(Default::default()),
            ..BrushParams::default()
        };
        assert_eq!(
            brush_deed(&erasing, [0.0; 3], &erasing, [1.0, 0.0, 0.0]),
            Some(Deed::ChangedColor)
        );

        // A preset: the size and the color both, plus whatever else it carries. Not
        // an adjustment of either, so it counts as neither.
        let mut preset = was;
        preset.size = was.size * 2.0;
        recolor(&mut preset);
        assert_eq!(brush_deed(&was, color(&was), &preset, color(&preset)), None);

        // The command that changes nothing — a slider dragged back to where it
        // started — is not a deed either.
        assert_eq!(brush_deed(&was, color(&was), &was, color(&was)), None);

        // And a field that is neither size, flow nor color is somebody else's
        // business entirely.
        let mut drained = was;
        drained.drain = was.drain + 1.0;
        assert_eq!(
            brush_deed(&was, color(&was), &drained, color(&drained)),
            None
        );
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
        let rows = serde_json::from_str::<Vec<serde_json::Value>>(json)
            .unwrap()
            .into_iter()
            .filter_map(|v| serde_json::from_value::<Row>(v).ok());
        let book = Ledger::from_rows(rows);
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

    /// What [`Ledger::rows`] writes is what [`Ledger::from_rows`] reads.
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
        let book = Ledger::from_rows(serde_json::from_str::<Vec<Row>>(&json).unwrap());
        assert_eq!(book.tally[Deed::Stroke.slot()], 7);
        assert!(book.given.contains("brush-panel"));
        // And a ledger's own rows read back as the same ledger.
        let json = serde_json::to_string(&book.rows()).unwrap();
        let read = Ledger::from_rows(serde_json::from_str::<Vec<Row>>(&json).unwrap());
        assert_eq!(read, book, "a ledger survives its own rows");
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
