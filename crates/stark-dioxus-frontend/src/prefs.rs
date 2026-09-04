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
//! 1. a field on [`Prefs`], with the default it should have — both in
//!    `stark_ui::prefs`, which is where the record is;
//! 2. a line in [`capture`], reading it back out of the live app;
//! 3. a line in [`apply_view`] or [`apply_engine`] — whichever owns it — pushing
//!    it back in at startup.
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
//! side of that bargain — and the reason it is JSON — is `stark_ui::storage`'s, and this
//! record is the one it was first argued for (§25.6).
//!
//! It matters more here than in the libraries, because this is the one record where
//! damage is **all-or-nothing**: a library is read entry by entry, so a bad entry costs
//! one preset, while a `Prefs` nobody can read costs every setting at once. Which is
//! still the right answer — a half-applied read is worse than the defaults — but it is
//! why an enum stored in this struct must be lenient about a name it does not know
//! (`stark_ui::prefs::ChromeHiding`) rather than refusing and taking its neighbours down.
//!
//! # Why loading happens twice
//!
//! Most preferences are frontend signals and can be applied the moment the app
//! starts. Some are not: a preference the **engine** owns — the peer outlines,
//! the undo budget, fast commit — is reached by a command, and there is no
//! engine to take one until the renderer's async init finishes. So [`load`] runs
//! at app start and [`load_engine`] once the renderer is up — the same split
//! `presets::load`/`presets::apply_first` makes, and for the same reason.
//! Applying the frontend half early is what keeps minimal mode from flashing
//! its full-width chrome for the length of a WebGPU init.

use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use stark_engine::command::ViewCommand;
use stark_ui::prefs::Prefs;
use stark_ui::storage;

// The three below were an `impl Prefs` until the record moved to `stark_ui`
// (§11.2, N1). The orphan rule then refused them, which is the boundary reporting
// itself (CLAUDE.md): each reads or writes *this frontend's signals*, so none of
// them was ever the record's business — `Prefs` is a serde struct and these are the
// chrome that fills it in.

/// The live app's preferences, as they would be stored.
///
/// `peek` throughout: this runs inside event handlers, and a preference read
/// is never a reason for the reading scope to re-render.
fn capture(state: AppState) -> Prefs {
    Prefs {
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
        fast_commit: state
            .obs
            .peek()
            .as_ref()
            .map_or(stark_engine::DEFAULT_FAST_COMMIT, |o| o.fast_commit),
        // Off the signal, not the projection: the projection holds what the surface
        // was told, and the choice has to outlive a screen that cannot show HDR.
        hdr: *state.hdr.peek(),
    }
}

/// Push the preferences the **frontend** owns into their signals.
fn apply_view(prefs: Prefs, state: AppState) {
    let mut assist = state.assist.enabled;
    let mut minimal = state.minimal;
    let mut tips = state.tutor.enabled;
    let mut hiding = state.chrome_hiding;
    let mut hdr = state.hdr;
    assist.set(prefs.assist);
    minimal.set(prefs.minimal);
    tips.set(prefs.tips);
    hiding.set(prefs.chrome_hiding);
    hdr.set(prefs.hdr);
}

/// Push the preferences the **engine** owns, as commands. Needs a renderer;
/// see the module comment.
fn apply_engine(prefs: Prefs, state: AppState) {
    dispatch(
        state,
        ViewCommand::SetShowPeerSelections(prefs.show_peer_selections),
    );
    dispatch(state, ViewCommand::SetHistoryBudget(prefs.history_budget));
    dispatch(state, ViewCommand::SetFastCommit(prefs.fast_commit));
    // The HDR choice met with the surface, which only exists now (§6.5).
    crate::panels::lighting::apply_output(state);
}

/// Apply this browser's stored preferences to the frontend. Called once at app
/// start, before the engine exists.
pub fn load(state: AppState) {
    apply_view(stored(), state);
}

/// Apply the stored preferences the engine owns. Called once the renderer is up,
/// unlike [`load`].
pub fn load_engine(state: AppState) {
    apply_engine(stored(), state);
}

/// Persist the app's current preferences. Called after every change made through
/// the settings dialog, so no row has to remember to.
pub fn save(state: AppState) {
    storage::save(&capture(state));
}

/// What this browser has stored, or the defaults — a browser that has never
/// stored anything and one whose stored value is damaged are the same case, and
/// both want the defaults rather than a half-applied read.
fn stored() -> Prefs {
    storage::load().unwrap_or_default()
}
