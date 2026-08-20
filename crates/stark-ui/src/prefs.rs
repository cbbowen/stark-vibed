//! This browser's standing preferences — what the ⚙ dialog sets — and where
//! they are kept between visits (§11).
//!
//! A setting is the one kind of state that is neither the artwork nor the tool
//! in your hand: a standing choice about how Stark behaves for **this client**,
//! set once and then left alone (`crate::settings`). "Set once" is a promise
//! that a reload breaks, so the settings follow this browser the way the shape
//! and preset libraries do — `localStorage`, per-origin, degrading to a
//! per-session choice where storage is unavailable (the `crate::identity`
//! bargain). Nothing here is written into the document or sent to peers.
//!
//! # The contract for adding a setting
//!
//! One serde struct holds every persisted preference, so adding one is three
//! lines rather than a new key and a new pair of read/write functions:
//!
//! 1. a field on [`Prefs`], with the default it should have in [`Prefs::default`];
//! 2. a line in [`Prefs::capture`], reading it back out of the live app;
//! 3. a line in [`Prefs::apply_view`] or [`Prefs::apply_engine`] — whichever
//!    owns it — pushing it back in at startup.
//!
//! A row of the dialog does not have to remember to save: [`save`] runs after
//! every change made through the dialog's own controls (`crate::settings`).
//!
//! # Why every field defaults
//!
//! `localStorage` outlives app versions, so the stored form has to survive a
//! [`Prefs`] that has gained or lost a field since it was written. `#[serde(default)]`
//! on the struct is what makes that work: a preference added later reads as its
//! default out of every value stored before it existed, rather than the whole record
//! failing to parse and silently resetting everything the user had set. The format's
//! side of that bargain — and the reason it is JSON — is `crate::storage`'s, and this
//! record is the one it was first argued for (§25.6).
//!
//! It matters more here than in the libraries, because this is the one record where
//! damage is **all-or-nothing**: a library is read entry by entry, so a bad entry costs
//! one preset, while a `Prefs` nobody can read costs every setting at once. Which is
//! still the right answer — a half-applied read is worse than the defaults — but it is
//! why an enum stored in this struct must be lenient about a name it does not know
//! (`crate::layout::ChromeHiding`) rather than refusing and taking its neighbours down.
//!
//! # Why loading happens twice
//!
//! Most preferences are frontend signals and can be applied the moment the app
//! starts. One is not: whether peers' selections are drawn is *engine* session
//! state, reached by a command, and there is no engine to take one until the
//! renderer's async init finishes. So [`load`] runs at app start and
//! [`load_engine`] once the renderer is up — the same split
//! `presets::load`/`presets::apply_first` makes, and for the same reason.
//! Applying the frontend half early is what keeps minimal mode from flashing
//! its full-width chrome for the length of a WebGPU init.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::state::{AppState, dispatch};
use crate::storage::{Record, Store};
use stark_engine::command::ViewCommand;

/// Every preference the ⚙ dialog sets, in the form they are stored in.
///
/// `#[serde(default)]` is what makes the struct extensible across versions — see
/// the module comment.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// Whether holding a stroke still snaps it to a line or an ellipse (§6.9).
    pub assist: bool,
    /// Whether the chrome over the canvas drops its words and keeps its marks.
    pub minimal: bool,
    /// Whether the chrome gets out of the way while you paint, and for how long
    /// (§11). Not a `bool`, and the one preference here that is not: the two
    /// mechanisms behind it are three states, not four (`layout::ChromeHiding`).
    pub chrome_hiding: crate::layout::ChromeHiding,
    /// Whether collaborators' selections are outlined alongside your own (§17.3).
    pub show_peer_selections: bool,
    /// Whether the guided tour offers its lessons (§24).
    ///
    /// The switch alone lives here. What the tour has *learned* about this browser —
    /// the tally of deeds, the lessons already given — is a table of its own
    /// (`crate::tutor`), because it is a record of what happened rather than a
    /// choice anybody made, and because turning the tour off and on again must not
    /// be a way to lose it.
    pub tips: bool,
    /// How much GPU memory undo history may hold before the oldest steps are given
    /// up, in bytes (§5) — the one preference that is about the *machine* rather
    /// than about how Stark behaves, which is why it is the one with a slider.
    pub history_budget: u64,
}

impl Record for Prefs {
    const STORE: Store = Store::Prefs;
}

impl Default for Prefs {
    /// The app's defaults, and the authority on them: every signal these seed is
    /// overwritten by [`load`] at startup, so a value written anywhere else would
    /// be the one that never applies.
    fn default() -> Self {
        Self {
            // On, because the assist is most of the value of a hold and somebody
            // who wants their line left crooked can find the switch.
            assist: true,
            // Off, because the words are how the chrome is *learned*; minimal mode
            // is what you turn on once you no longer need them.
            minimal: false,
            // What Stark did before there was a choice, so the default is not a new
            // opinion — it is the behavior every existing browser already has, and
            // the ones that stored their preferences before this field existed read
            // as it (`ChromeHiding::default`).
            chrome_hiding: crate::layout::ChromeHiding::default(),
            // Off, because a second contour over the artwork is paid for on every
            // frame you look at it (§17.3).
            show_peer_selections: false,
            // On, because the tour exists for the artist who has not found the
            // switch yet, and it is the one preference whose default decides whether
            // anybody it is for ever sees it. It costs a newcomer five cards across
            // their first few sessions and nothing after that (§24).
            tips: true,
            // The engine's own default, not a second opinion about it: a value
            // written here would be the one that actually applies, and then there
            // would be two answers to what Stark does out of the box. `load_engine`
            // pushes this back in at startup, so the engine's constant has to be
            // what a browser that has never stored anything gets.
            history_budget: stark_engine::DEFAULT_HISTORY_BUDGET,
        }
    }
}

impl Prefs {
    /// The live app's preferences, as they would be stored.
    ///
    /// `peek` throughout: this runs inside event handlers, and a preference read
    /// is never a reason for the reading scope to re-render.
    fn capture(state: AppState) -> Self {
        Self {
            assist: *state.assist.enabled.peek(),
            minimal: *state.minimal.peek(),
            chrome_hiding: *state.chrome_hiding.peek(),
            show_peer_selections: state
                .obs
                .peek()
                .as_ref()
                .is_some_and(|o| o.show_peer_selections),
            tips: *state.tutor.enabled.peek(),
            // Read off the engine's projection rather than off a signal of our own,
            // for the reason the projection exists: a second copy is one that can
            // disagree. Before the renderer is up there is nothing to read, and the
            // default is the honest answer — `save` only ever runs from the dialog,
            // which cannot be open without one.
            history_budget: state
                .obs
                .peek()
                .as_ref()
                .map_or(stark_engine::DEFAULT_HISTORY_BUDGET, |o| o.history_budget),
        }
    }

    /// Push the preferences the **frontend** owns into their signals.
    fn apply_view(self, state: AppState) {
        let mut assist = state.assist.enabled;
        let mut minimal = state.minimal;
        let mut tips = state.tutor.enabled;
        let mut hiding = state.chrome_hiding;
        assist.set(self.assist);
        minimal.set(self.minimal);
        tips.set(self.tips);
        hiding.set(self.chrome_hiding);
    }

    /// Push the preferences the **engine** owns, as commands. Needs a renderer;
    /// see the module comment.
    fn apply_engine(self, state: AppState) {
        dispatch(
            state,
            ViewCommand::SetShowPeerSelections(self.show_peer_selections),
        );
        dispatch(state, ViewCommand::SetHistoryBudget(self.history_budget));
    }
}

/// Apply this browser's stored preferences to the frontend. Called once at app
/// start, before the engine exists.
pub fn load(state: AppState) {
    stored().apply_view(state);
}

/// Apply the stored preferences the engine owns. Called once the renderer is up,
/// unlike [`load`].
pub fn load_engine(state: AppState) {
    stored().apply_engine(state);
}

/// Persist the app's current preferences. Called after every change made through
/// the settings dialog, so no row has to remember to.
pub fn save(state: AppState) {
    crate::storage::save(&Prefs::capture(state));
}

/// What this browser has stored, or the defaults — a browser that has never
/// stored anything and one whose stored value is damaged are the same case, and
/// both want the defaults rather than a half-applied read.
fn stored() -> Prefs {
    crate::storage::load().unwrap_or_default()
}
