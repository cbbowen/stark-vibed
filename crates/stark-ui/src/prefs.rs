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
//! # Why JSON, and why every field defaults
//!
//! `localStorage` outlives app versions, so the stored form has to survive a
//! [`Prefs`] that has gained or lost a field since it was written. JSON is
//! self-describing, and `#[serde(default)]` on the struct means a preference
//! added later reads as its default out of every value stored before it existed
//! — rather than the whole blob failing to parse and silently resetting
//! everything the user had set. Same reasoning as the preset library's, which
//! keeps JSON for the same reason the save file does not (§8).
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
use stark_core::command::ViewCommand;

/// One key, namespaced like the shape and preset libraries'; versioned so a
/// future format change can migrate rather than mis-parse.
const KEY_PREFS: &str = "stark.prefs.v1";

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
    /// Whether collaborators' selections are outlined alongside your own (§17.3).
    pub show_peer_selections: bool,
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
            // Off, because a second contour over the artwork is paid for on every
            // frame you look at it (§17.3).
            show_peer_selections: false,
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
            show_peer_selections: state
                .obs
                .peek()
                .as_ref()
                .is_some_and(|o| o.show_peer_selections),
        }
    }

    /// Push the preferences the **frontend** owns into their signals.
    fn apply_view(self, state: AppState) {
        let mut assist = state.assist.enabled;
        let mut minimal = state.minimal;
        assist.set(self.assist);
        minimal.set(self.minimal);
    }

    /// Push the preferences the **engine** owns, as commands. Needs a renderer;
    /// see the module comment.
    fn apply_engine(self, state: AppState) {
        dispatch(
            state,
            ViewCommand::SetShowPeerSelections(self.show_peer_selections),
        );
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
    let Some(store) = storage() else { return };
    let Ok(json) = serde_json::to_string(&Prefs::capture(state)) else {
        return;
    };
    if store.set_item(KEY_PREFS, &json).is_err() {
        // Quota or a storage-less browser. The settings still hold for this
        // session; only their durability is lost.
        tracing::warn!("could not persist the settings (storage full or unavailable)");
    }
}

/// What this browser has stored, or the defaults — a browser that has never
/// stored anything and one whose stored value is damaged are the same case, and
/// both want the defaults rather than a half-applied read.
fn stored() -> Prefs {
    storage()
        .and_then(|store| store.get_item(KEY_PREFS).ok().flatten())
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}
