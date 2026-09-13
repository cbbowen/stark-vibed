//! This browser's standing preferences — what the ⚙ dialog sets — and where
//! they are kept between visits (§11, §25.6).
//!
//! Nothing here is written into the document or sent to peers. The record is
//! `stark_ui::prefs::Prefs`; what is here is the frontend's half: the one signal
//! that holds it ([`Signals::prefs`](crate::state::Signals::prefs)) and the one
//! writer that changes it ([`set`]).
//!
//! # Adding a setting
//!
//! A field on `Prefs` with its default, a row in the dialog that calls [`set`], and
//! a line in [`apply`] saying what moving it does — which the compiler asks for,
//! since that function takes the record apart field by field.
//!
//! # Why loading happens twice
//!
//! The signal is seeded from the stored record when the app state is built, so the
//! first render is already in the mode the user left. The three preferences the
//! **engine** owns — the peer outlines, the undo budget, fast commit — are commands,
//! and there is no engine to take one until the renderer's async init finishes:
//! [`load_engine`] pushes them in then.

use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use stark_engine::command::ViewCommand;
use stark_ui::prefs::Prefs;
use stark_ui::storage;

/// What this browser has stored, or the defaults — a browser that has never
/// stored anything and one whose stored value is damaged are the same case, and
/// both want the defaults rather than a half-applied read.
pub fn stored() -> Prefs {
    storage::load().unwrap_or_default()
}

/// Change this browser's preferences, carry the change out, and keep it.
///
/// A change that moves nothing writes nothing: not the signal, not the engine, not
/// storage.
pub fn set(state: AppState, change: impl FnOnce(&mut Prefs)) {
    if apply(state, change) {
        save(state);
    }
}

/// [`set`] without the save, for a control that moves at pointer rate and keeps
/// its value on release by calling [`save`].
pub fn set_unsaved(state: AppState, change: impl FnOnce(&mut Prefs)) {
    apply(state, change);
}

/// Persist the preferences as they stand, the engine's half read back off the
/// projection.
pub fn save(state: AppState) {
    storage::save(&current(state));
}

/// Push the stored preferences the engine owns into the engine. Called once the
/// renderer is up.
pub fn load_engine(state: AppState) {
    let prefs = *state.prefs.peek();
    dispatch(
        state,
        ViewCommand::SetShowPeerSelections(prefs.show_peer_selections),
    );
    dispatch(state, ViewCommand::SetHistoryBudget(prefs.history_budget));
    dispatch(state, ViewCommand::SetFastCommit(prefs.fast_commit));
    // The HDR choice met with the surface, which only exists now (§6.5).
    crate::panels::lighting::apply_output(state);
}

/// The preferences as they stand: the signal, with the engine's half as the engine
/// holds it — or as stored, before there is an engine to ask.
fn current(state: AppState) -> Prefs {
    let mut prefs = *state.prefs.peek();
    if let Some(o) = state.obs.peek().as_ref() {
        prefs.show_peer_selections = o.show_peer_selections;
        prefs.history_budget = o.history_budget;
        prefs.fast_commit = o.fast_commit;
    }
    prefs
}

/// Apply `change` and do what each moved field needs; `true` if anything moved.
fn apply(state: AppState, change: impl FnOnce(&mut Prefs)) -> bool {
    let was = current(state);
    let mut now = was;
    change(&mut now);
    if now == was {
        return false;
    }
    // Every field by name, so one added to `Prefs` does not compile until it says
    // what moving it does.
    let Prefs {
        assist: _,
        minimal: _,
        chrome_hiding,
        show_peer_selections,
        tips,
        history_budget,
        fast_commit,
        hdr,
    } = now;
    if show_peer_selections != was.show_peer_selections {
        dispatch(
            state,
            ViewCommand::SetShowPeerSelections(show_peer_selections),
        );
    }
    if history_budget != was.history_budget {
        dispatch(state, ViewCommand::SetHistoryBudget(history_budget));
    }
    if fast_commit != was.fast_commit {
        dispatch(state, ViewCommand::SetFastCommit(fast_commit));
    }
    write_if_moved(state.prefs, now);
    // After the write: each of these reads the signal.
    if tips != was.tips {
        crate::tutor::set_enabled(state, tips);
    }
    if hdr != was.hdr {
        crate::panels::lighting::apply_output(state);
    }
    if chrome_hiding != was.chrome_hiding {
        // A stack asleep when the choice moves off "Hide after painting" has
        // nothing left to wake it: the slice that hears the pointer is mounted on
        // the state being switched off.
        crate::layout::wake_panels(state);
    }
    true
}

/// Write `value` only if the signal holds something else: a `set` wakes every
/// reader whatever it is handed.
fn write_if_moved(mut prefs: Signal<Prefs>, value: Prefs) {
    let moved = *prefs.peek() != value;
    if moved {
        prefs.set(value);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dioxus::dioxus_core::{ReactiveContext, ScopeId, VirtualDom};

    use super::*;

    fn root() -> Element {
        let state = AppState::new();
        use_context_provider(|| state);
        rsx! {}
    }

    /// **A change that moves nothing wakes nothing**, and one that moves a field
    /// wakes the signal's readers once.
    #[test]
    fn set_writes_the_signal_only_when_a_preference_moves() {
        let mut dom = VirtualDom::new(root);
        dom.rebuild_in_place();
        dom.in_scope(ScopeId::APP, || {
            let state = consume_context::<AppState>();
            let writes = Arc::new(AtomicUsize::new(0));
            let reader = ReactiveContext::new_with_callback(
                {
                    let writes = writes.clone();
                    move || {
                        writes.fetch_add(1, Ordering::Relaxed);
                    }
                },
                ScopeId::APP,
                std::panic::Location::caller(),
            );
            reader.run_in(|| {
                let _ = state.prefs.read();
            });

            let was = state.prefs.peek().minimal;
            set(state, |p| p.minimal = was);
            set(state, |_| {});
            assert_eq!(writes.load(Ordering::Relaxed), 0, "nothing moved");

            set(state, |p| p.minimal = !was);
            assert_eq!(writes.load(Ordering::Relaxed), 1, "one field moved");
            assert_eq!(state.prefs.peek().minimal, !was);
        });
    }
}
