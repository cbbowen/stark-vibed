//! The reader (§24.2, §24.3): what turns reports of deeds into counts, and counts into
//! the one card the tour shows.
//!
//! A plain value, so its rules — the coalescing window, the pan run, the bracket depth,
//! promotion into an empty card, an answer spending a card still waiting — are tested
//! without signals. `TutorState` keeps it off the reactive graph and publishes only its
//! [`CardState`]; whatever else a step needs of the host comes back as [`Effects`].

use stark_ui::prefs::ChromeHiding;
use strum::EnumCount;

use super::lessons::{Deed, LESSONS, Ledger, Lesson, due};

/// How long a gap between two reports of one deed makes them two deeds, in seconds.
///
/// A slider drag is sixty `SetBrush`es a second and one act; two deliberate adjustments a
/// beat apart are two. Half a second is well past any gap inside a drag (a 60 Hz pointer
/// reports every 16 ms) and well short of deciding to change something twice. Measured
/// from the **last** report of a run, so a slow drag stays one deed however long it takes.
const COALESCE: f64 = 0.5;

/// How far one run of panning travels before it counts as a long one, in page px.
///
/// About a screen's width: the point where reaching for the navigator would have been
/// quicker than the drag, which is the claim the lesson this feeds makes.
const LONG_PAN: f32 = 1200.0;

/// The reader's working memory — what the last half-second was.
///
/// Never persisted: every field is about a gesture that is over by the time the page
/// closes.
#[derive(Clone, Copy, Debug)]
struct Recent {
    /// When each deed was last reported, by [`Deed::slot`], on the host's clock. Negative
    /// infinity for one never reported, so a first report is always far enough from the
    /// last.
    at: [f64; Deed::COUNT],
    /// How far the run of panning in flight has travelled, page px.
    pan: f32,
    /// Whether that run has scored already, so it scores once however far it goes.
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

/// The tour's one card (§24.3), by index in [`LESSONS`].
///
/// Coming due and being shown are two steps. A lesson opens what it points at, and a panel
/// opened mid-stroke is put straight back to sleep by the release, so a lesson that comes
/// due with the canvas in hand waits until the card finds the screen free.
///
/// One value rather than a slot for each step, because a lesson only ever comes due into
/// an empty card: waiting and showing are never both true.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum CardState {
    /// No lesson waiting or on screen.
    #[default]
    Empty,
    /// A lesson come due, waiting for the screen to be free.
    Due(usize),
    /// A lesson on screen.
    Showing(usize),
}

impl CardState {
    /// The lesson waiting, if any.
    pub(super) fn due(self) -> Option<usize> {
        match self {
            CardState::Due(i) => Some(i),
            CardState::Empty | CardState::Showing(_) => None,
        }
    }

    /// The lesson on screen, if any.
    pub(super) fn showing(self) -> Option<usize> {
        match self {
            CardState::Showing(i) => Some(i),
            CardState::Empty | CardState::Due(_) => None,
        }
    }
}

/// What a step of the [`Tour`] needs of the host besides publishing its [`CardState`].
#[must_use = "a step's effects are the host's to carry out"]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Effects {
    /// The ledger changed: persist it.
    pub(super) save: bool,
    /// A card pointing into the panel stack came down: wake the stack. Otherwise it fades
    /// the instant the card goes, which reads as the answer having closed the panel
    /// (§24.3).
    pub(super) wake_panels: bool,
}

/// Everything the tour holds but the switch that turns it off, which is a preference
/// (§24.4) and so is handed to the step that needs it.
#[derive(Default)]
pub(super) struct Tour {
    /// False until [`begin`](Self::begin), so the commands startup makes on the user's
    /// behalf are not read as things the user did.
    armed: bool,
    /// The durable half: the tally and the lessons given.
    ledger: Ledger,
    /// The half that is only about the last half-second.
    recent: Recent,
    /// The one card.
    card: CardState,
    /// How many brackets are open around brush writes that are not the artist reaching
    /// for a control (`tutor::not_reaching`).
    not_reaching: u32,
}

impl Tour {
    /// Start listening, from the ledger this browser stored.
    pub(super) fn begin(&mut self, ledger: Ledger) {
        self.ledger = ledger;
        self.armed = true;
    }

    /// Whether the tour is listening yet. Asked by the host before it reads anything, so
    /// the steps below are never handed a deed from before [`begin`](Self::begin).
    pub(super) fn is_armed(&self) -> bool {
        self.armed
    }

    pub(super) fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    pub(super) fn card(&self) -> CardState {
        self.card
    }

    /// Open a bracket around brush writes that are not the artist reaching for a control.
    ///
    /// A depth rather than a flag, because brackets nest: a number key held mid-tuning-drag
    /// swaps the tool inside the drag's own bracket, and a flag would let the swap's close
    /// cancel the drag's (§24.2).
    pub(super) fn open_bracket(&mut self) {
        self.not_reaching = self.not_reaching.saturating_add(1);
    }

    /// Close a bracket. Saturates, so a stray close costs nothing.
    pub(super) fn close_bracket(&mut self) {
        self.not_reaching = self.not_reaching.saturating_sub(1);
    }

    /// Whether a brush write now would be the artist reaching for a control: outside every
    /// bracket.
    pub(super) fn is_reaching(&self) -> bool {
        self.not_reaching == 0
    }

    /// Feed `travel` page px into the run of panning in flight, answering
    /// [`Deed::LongPan`] on the sample that carries the run past [`LONG_PAN`].
    ///
    /// Scored at the crossing, so nothing has to detect a run's end: a run that stops short
    /// never scores, and a gap longer than [`COALESCE`] starts a new one. The run keeps its
    /// clock in the deed's own slot of [`Recent::at`], which is why [`tally`](Self::tally)
    /// does not coalesce a long pan a second time.
    pub(super) fn pan(&mut self, now: f64, travel: f32) -> Option<Deed> {
        let run = &mut self.recent;
        let slot = Deed::LongPan.slot();
        if now - run.at[slot] > COALESCE {
            run.pan = 0.0;
            run.long = false;
        }
        run.at[slot] = now;
        run.pan += travel;
        let crossed = !run.long && run.pan >= LONG_PAN;
        run.long |= crossed;
        crossed.then_some(Deed::LongPan)
    }

    /// Count `deeds` reported at `now`, and bring due the first lesson one of them owes.
    ///
    /// A report within [`COALESCE`] of the last of the same deed is the same act, and only
    /// moves the clock. A lesson comes due only into an empty card and only with `tips` on;
    /// one passed over is not lost, since its threshold stays crossed and the next deed of
    /// its kind offers it again ([`due`]). Deeds are tallied with tips off, so turning them
    /// back on resumes rather than restarts (§24.4).
    pub(super) fn tally(
        &mut self,
        now: f64,
        deeds: &[Deed],
        chrome: ChromeHiding,
        tips: bool,
    ) -> Effects {
        let mut effects = Effects::default();
        for &deed in deeds {
            // A long pan is already one act per run (`pan`, whose clock shares this slot).
            if deed != Deed::LongPan {
                let slot = deed.slot();
                let repeat = now - self.recent.at[slot] <= COALESCE;
                self.recent.at[slot] = now;
                if repeat {
                    continue;
                }
            }
            self.ledger.add(deed);
            effects.save = true;
            if tips
                && self.card == CardState::Empty
                && let Some(i) = due(&self.ledger, deed, chrome)
            {
                self.card = CardState::Due(i);
            }
            // Last, so a deed that answers the card up cannot also replace it in the same
            // breath — which would read as the tour arguing back.
            self.answered(deed, &mut effects);
        }
        effects
    }

    /// Put the lesson waiting on screen. When is the card's to decide; this only moves it.
    pub(super) fn show(&mut self) {
        if let CardState::Due(i) = self.card {
            self.card = CardState::Showing(i);
        }
    }

    /// "Got it": put lesson `i` away for good, and bring due the next lesson its deed still
    /// owes (§24.3).
    ///
    /// The chain is what makes a series possible — the brush editor's cards all wait on one
    /// deed — and it cannot run away: each turn gives one lesson, and [`due`] skips those.
    pub(super) fn dismiss(&mut self, i: usize, chrome: ChromeHiding) -> Effects {
        let mut effects = Effects::default();
        if let Some(deed) = self.retire(i, &mut effects)
            && let Some(next) = due(&self.ledger, deed, chrome)
        {
            self.card = CardState::Due(next);
        }
        effects
    }

    /// Put lesson `i` away because what it points at has gone.
    ///
    /// [`dismiss`](Self::dismiss) without the chain: closing the brush editor takes the
    /// anchor from under its whole series, and a chain would retire the lot in one flush —
    /// cards nobody saw, marked as taught. The rest stay owed for the next time.
    pub(super) fn abandon(&mut self, i: usize) -> Effects {
        let mut effects = Effects::default();
        self.retire(i, &mut effects);
        effects
    }

    /// Tips switched off: take down the card on screen **without** giving it, since nobody
    /// was taught a lesson they switched off mid-sentence (§24.4).
    ///
    /// A lesson still waiting stays owed. The card keeps it off screen while tips are off,
    /// which also covers one brought due by the [`dismiss`](Self::dismiss) chain.
    pub(super) fn switch_off(&mut self) -> Effects {
        let mut effects = Effects::default();
        if let CardState::Showing(i) = self.card {
            self.card = CardState::Empty;
            effects.wake_panels = LESSONS.get(i).is_some_and(|l| l.anchor.holds_panels());
        }
        effects
    }

    /// Whether dismissing lesson `i` would bring another — "Next" rather than "Got it" on
    /// its button, since "Got it" on the first of five is a lie about how many are coming.
    pub(super) fn brings_another(&self, i: usize, chrome: ChromeHiding) -> bool {
        LESSONS.get(i).is_some_and(|lesson| {
            let mut book = self.ledger.clone();
            book.give(lesson.key);
            due(&book, lesson.deed, chrome).is_some()
        })
    }

    /// Spend the card `deed` answers ([`Answer`](super::lessons::Answer)): the one on
    /// screen, or the one still waiting to become it.
    ///
    /// The waiting one is the case this most has to catch — what the artist does while a
    /// lesson waits for the screen is exactly what they were about to be told (§24.3).
    /// Given either way, as the button would; the chain is deliberately not run, so the next
    /// lesson arrives on its deed's own terms.
    fn answered(&mut self, deed: Deed, effects: &mut Effects) {
        let answers = |i: usize| {
            LESSONS
                .get(i)
                .is_some_and(|l| l.answer.dismisses().contains(&deed))
        };
        match self.card {
            CardState::Due(i) if answers(i) => {
                self.card = CardState::Empty;
                self.give(i, effects);
            }
            CardState::Showing(i) if answers(i) => {
                self.retire(i, effects);
            }
            CardState::Empty | CardState::Due(_) | CardState::Showing(_) => {}
        }
    }

    /// The half every way out of a card shares: take it down, give lesson `i`, and let go
    /// of the panel stack it held. Answers the lesson's deed, or `None` for an index that
    /// names no lesson.
    fn retire(&mut self, i: usize, effects: &mut Effects) -> Option<Deed> {
        if let CardState::Showing(_) = self.card {
            self.card = CardState::Empty;
        }
        let lesson = self.give(i, effects)?;
        effects.wake_panels |= lesson.anchor.holds_panels();
        Some(lesson.deed)
    }

    /// Write lesson `i` into the ledger as given.
    fn give(&mut self, i: usize, effects: &mut Effects) -> Option<&'static Lesson> {
        let lesson = LESSONS.get(i)?;
        self.ledger.give(lesson.key);
        effects.save = true;
        Some(lesson)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHROME: ChromeHiding = ChromeHiding::AfterPainting;

    fn begun() -> Tour {
        let mut tour = Tour::default();
        tour.begin(Ledger::default());
        tour
    }

    fn lesson(key: &str) -> usize {
        LESSONS
            .iter()
            .position(|l| l.key == key)
            .expect("the lesson is on the table")
    }

    /// `deed` reported at `now`, with tips on.
    fn tally(tour: &mut Tour, now: f64, deed: Deed) -> Effects {
        tour.tally(now, &[deed], CHROME, true)
    }

    /// Four strokes a second apart, which bring the color panel's lesson due.
    fn earn_the_color_panel(tour: &mut Tour) -> usize {
        for n in 0..4_u8 {
            let _ = tally(tour, f64::from(n), Deed::Stroke);
        }
        let color = lesson("color-panel");
        assert_eq!(
            tour.card(),
            CardState::Due(color),
            "the fourth stroke owes the color panel"
        );
        color
    }

    /// Reports of one deed within `COALESCE` of each other are one act, measured from the
    /// last report of the run — so a slow drag stays one deed however long it takes.
    #[test]
    fn reports_within_the_window_are_one_deed() {
        let mut tour = begun();
        assert!(
            tally(&mut tour, 0.0, Deed::Undo).save,
            "the first report counts, and is saved"
        );
        assert_eq!(
            tally(&mut tour, 0.4, Deed::Undo),
            Effects::default(),
            "a repeat changes nothing worth saving"
        );
        // 0.8 s after the first report, but 0.4 s after the last.
        let _ = tally(&mut tour, 0.8, Deed::Undo);
        assert_eq!(tour.ledger().count(Deed::Undo), 1, "the run is one act");

        let _ = tally(&mut tour, 0.8 + 2.0 * COALESCE, Deed::Undo);
        assert_eq!(
            tour.ledger().count(Deed::Undo),
            2,
            "a pause makes a second act"
        );

        let _ = tally(&mut tour, 0.8 + 2.0 * COALESCE, Deed::Redo);
        assert_eq!(
            tour.ledger().count(Deed::Redo),
            1,
            "and the window is each deed's own"
        );
    }

    /// A run of panning scores once, on the sample that carries it past `LONG_PAN`, and a
    /// pause longer than `COALESCE` starts a new run from nothing.
    #[test]
    fn a_pan_run_scores_once_when_it_crosses() {
        /// A pan sample as the host feeds one: the run first, then whatever it scored.
        fn pan(tour: &mut Tour, now: f64, travel: f32) {
            let long = tour.pan(now, travel);
            let _ = tour.tally(now, long.as_slice(), CHROME, true);
        }
        let scored = |tour: &Tour| tour.ledger().count(Deed::LongPan);
        let mut tour = begun();

        pan(&mut tour, 0.0, 0.6 * LONG_PAN);
        assert_eq!(scored(&tour), 0, "short of the line");
        pan(&mut tour, 0.1, 0.6 * LONG_PAN);
        assert_eq!(scored(&tour), 1, "the crossing scores");
        pan(&mut tour, 0.2, 10.0 * LONG_PAN);
        assert_eq!(scored(&tour), 1, "the same run does not score again");

        pan(&mut tour, 1.0, 0.6 * LONG_PAN);
        pan(&mut tour, 2.0, 0.6 * LONG_PAN);
        assert_eq!(scored(&tour), 1, "two short runs are not one long one");
        pan(&mut tour, 2.1, 0.6 * LONG_PAN);
        assert_eq!(scored(&tour), 2, "a new run scores at its own crossing");
    }

    /// The brackets nest (§24.2): a quick slot worn mid-tuning-drag opens a bracket inside
    /// the drag's, and closing it must not count the drag's own writes.
    #[test]
    fn a_nested_bracket_does_not_close_the_outer_one() {
        let mut tour = begun();
        tour.open_bracket();
        tour.open_bracket();
        tour.close_bracket();
        assert!(!tour.is_reaching(), "still inside the drag's bracket");
        tour.close_bracket();
        assert!(tour.is_reaching(), "outside every bracket");

        tour.close_bracket();
        tour.open_bracket();
        assert!(
            !tour.is_reaching(),
            "a stray close leaves no debt for the next open to pay"
        );
    }

    /// An answer spends a lesson still waiting for the screen (§24.3): given, as its button
    /// would, with no card up to take down and no panel stack to let go of.
    #[test]
    fn an_answer_spends_the_lesson_still_waiting() {
        let mut tour = begun();
        earn_the_color_panel(&mut tour);

        let effects = tally(&mut tour, 10.0, Deed::ChangedColor);
        assert_eq!(
            tour.card(),
            CardState::Empty,
            "the waiting lesson is answered"
        );
        assert!(
            tour.ledger().is_given("color-panel"),
            "and given, as its button would"
        );
        assert_eq!(
            effects,
            Effects {
                save: true,
                wake_panels: false
            },
            "no card was up to hold the panels"
        );
    }

    /// The same answer with the card on screen takes it down and wakes the panel stack the
    /// card was holding up.
    #[test]
    fn an_answer_takes_down_the_card_on_screen() {
        let mut tour = begun();
        let color = earn_the_color_panel(&mut tour);
        tour.show();
        assert_eq!(tour.card(), CardState::Showing(color), "promoted");

        let effects = tally(&mut tour, 10.0, Deed::ChangedColor);
        assert_eq!(tour.card(), CardState::Empty, "taken down");
        assert!(tour.ledger().is_given("color-panel"), "and given");
        assert_eq!(
            effects,
            Effects {
                save: true,
                wake_panels: true
            },
            "a card at a panel lets go of the stack"
        );
    }

    /// Turning tips off leaves a lesson still waiting owed, and takes the card on screen
    /// down without giving it (§24.4).
    #[test]
    fn switching_off_leaves_the_waiting_lesson_owed() {
        let mut tour = begun();
        let color = earn_the_color_panel(&mut tour);
        assert_eq!(
            tour.switch_off(),
            Effects::default(),
            "nothing on screen, nothing to do"
        );
        assert_eq!(
            tour.card(),
            CardState::Due(color),
            "the waiting lesson stays owed"
        );

        tour.show();
        assert_eq!(
            tour.switch_off(),
            Effects {
                save: false,
                wake_panels: true
            },
            "the card comes down and lets go of the stack"
        );
        assert_eq!(tour.card(), CardState::Empty, "nothing on screen");
        assert!(
            !tour.ledger().is_given("color-panel"),
            "a lesson switched off mid-sentence was not taught"
        );
    }

    /// Deeds are tallied with tips off and nothing comes due; with tips back on, the next
    /// deed of the kind offers what was passed over.
    #[test]
    fn with_tips_off_deeds_count_and_nothing_comes_due() {
        let mut tour = begun();
        for n in 0..4_u8 {
            let _ = tour.tally(f64::from(n), &[Deed::Stroke], CHROME, false);
        }
        assert_eq!(tour.ledger().count(Deed::Stroke), 4, "counted all the same");
        assert_eq!(tour.card(), CardState::Empty, "but nothing came due");

        let _ = tally(&mut tour, 10.0, Deed::Stroke);
        assert_eq!(
            tour.card(),
            CardState::Due(lesson("color-panel")),
            "a threshold is a floor"
        );
    }

    /// "Got it" brings the next lesson its deed owes; an anchor going away does not (§24.3),
    /// so closing the brush editor mid-series leaves the rest of the series owed.
    #[test]
    fn dismissing_chains_and_abandoning_does_not() {
        let mut tour = begun();
        let _ = tally(&mut tour, 0.0, Deed::OpenedBrushEditor);
        let preview = lesson("be-preview");
        assert_eq!(
            tour.card(),
            CardState::Due(preview),
            "opening the editor owes its series"
        );
        assert!(
            tour.brings_another(preview, CHROME),
            "the first card says Next"
        );

        tour.show();
        let _ = tour.dismiss(preview, CHROME);
        let tip = lesson("be-tip");
        assert_eq!(
            tour.card(),
            CardState::Due(tip),
            "Next brings the second card"
        );

        tour.show();
        let effects = tour.abandon(tip);
        assert_eq!(
            tour.card(),
            CardState::Empty,
            "closing the dialog ends the walk"
        );
        assert!(
            tour.ledger().is_given("be-tip"),
            "the card on screen was answered"
        );
        assert!(
            !tour.ledger().is_given("be-paint"),
            "the rest stay owed for next time"
        );
        assert!(
            !effects.wake_panels,
            "a card inside the dialog holds no panels"
        );
    }
}
