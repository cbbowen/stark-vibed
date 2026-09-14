//! This browser's preferences — what the ⚙ dialog sets — kept between visits but
//! never in the document or sent to peers (§11, §25.6).
//!
//! The record is `stark_ui::prefs::Prefs`; this module holds its signal
//! ([`Signals::prefs`](crate::state::Signals::prefs)) and its one writer ([`set`]).
//! A new setting also needs a line in [`apply`], which the compiler demands.
//!
//! Loading happens twice: the signal is seeded when the app state is built, but the
//! preferences the engine owns are commands, so [`load_engine`] pushes them once the
//! renderer is up.

use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use stark_engine::command::ViewCommand;
use stark_ui::prefs::Prefs;
use stark_ui::storage;

/// What this browser has stored, or the defaults when nothing or a damaged value is.
pub fn stored() -> Prefs {
    storage::load().unwrap_or_default()
}

/// Change this browser's preferences, carry the change out, and keep it. A change
/// that moves nothing writes nothing: not the signal, the engine, or storage.
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

/// Persist the preferences as they stand, engine-owned fields read back from the engine.
pub fn save(state: AppState) {
    storage::save(&current(state));
}

/// Push the preferences the engine owns into the engine, once the renderer is up.
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

/// The signal, with the engine-owned fields read back from the engine once there is one.
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
    // Every field by name, so a new `Prefs` field must say what moving it does.
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
    // With no engine, `load_engine` pushes these later. Skipped rather than dispatched,
    // since any dispatch takes the renderer for writing and wakes its readers.
    let engine_up = state.obs.peek().is_some();
    if engine_up {
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
    }
    write_if_moved(state.prefs, now);
    // After the write: each of these reads the signal.
    if was.tips && !tips {
        crate::tutor::switch_off(state);
    }
    if hdr != was.hdr {
        crate::panels::lighting::apply_output(state);
    }
    if chrome_hiding != was.chrome_hiding {
        // Switching hiding off unmounts what would wake a sleeping stack.
        crate::layout::wake_panels(state);
    }
    true
}

/// Write `value` only if it differs: a `set` wakes every reader regardless.
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
    use stark_ui::storage::{Backend, BlobRead, Store, Stored};

    use super::*;

    /// Writes of the preferences record. Process-wide; nextest gives each test a process.
    static SAVES: AtomicUsize = AtomicUsize::new(0);

    /// A store that holds nothing and counts writes of the preferences record.
    struct Counting;

    impl Backend for Counting {
        fn get(&self, _: &str) -> Option<String> {
            None
        }

        fn set(&self, key: &str, _: &str) -> bool {
            if key == Store::Prefs.named().0 {
                SAVES.fetch_add(1, Ordering::Relaxed);
            }
            true
        }

        fn remove(&self, _: &str) {}

        fn blob_get_many<'a>(
            &'a self,
            keys: &'a [String],
        ) -> Stored<'a, Result<Vec<BlobRead>, String>> {
            Box::pin(std::future::ready(Ok(vec![Ok(None); keys.len()])))
        }

        fn blob_put<'a>(&'a self, _: &'a str, _: &'a [u8]) -> Stored<'a, Result<(), String>> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn blob_delete<'a>(&'a self, _: &'a str) -> Stored<'a, ()> {
            Box::pin(std::future::ready(()))
        }
    }

    fn root() -> Element {
        let state = AppState::new();
        use_context_provider(|| state);
        rsx! {}
    }

    /// A count of the writes that wake whoever `read` subscribes to.
    fn wakes(read: impl FnOnce()) -> Arc<AtomicUsize> {
        let wakes = Arc::new(AtomicUsize::new(0));
        let reader = ReactiveContext::new_with_callback(
            {
                let wakes = wakes.clone();
                move || {
                    wakes.fetch_add(1, Ordering::Relaxed);
                }
            },
            ScopeId::APP,
            std::panic::Location::caller(),
        );
        reader.run_in(read);
        wakes
    }

    /// Run `f` against a freshly built app state, over the counting store.
    fn with_state(f: impl FnOnce(AppState)) {
        stark_ui::storage::install(Counting);
        let mut dom = VirtualDom::new(root);
        dom.rebuild_in_place();
        dom.in_scope(ScopeId::APP, || f(consume_context::<AppState>()));
    }

    #[test]
    fn an_unchanged_preference_writes_neither_the_signal_nor_the_store() {
        with_state(|state| {
            let signal = wakes(|| {
                let _ = state.prefs.read();
            });
            let saves = SAVES.load(Ordering::Relaxed);
            let was = *state.prefs.peek();
            set(state, |p| p.fast_commit = was.fast_commit);
            set(state, |_| {});
            assert_eq!(signal.load(Ordering::Relaxed), 0);
            assert_eq!(SAVES.load(Ordering::Relaxed), saves);
        });
    }

    /// Before the renderer, the signal must move and be saved, since `load_engine`
    /// pushes it later.
    #[test]
    fn an_engine_preference_moved_with_no_renderer_is_kept_and_dispatches_nothing() {
        with_state(|state| {
            let signal = wakes(|| {
                let _ = state.prefs.read();
            });
            // Every door takes the renderer for writing, with or without an engine in it.
            let doors = wakes(|| {
                let _ = state.renderer.read();
            });
            let saves = SAVES.load(Ordering::Relaxed);
            let was = state.prefs.peek().fast_commit;
            set(state, |p| p.fast_commit = !was);
            assert_eq!(signal.load(Ordering::Relaxed), 1, "the signal moved");
            assert_eq!(state.prefs.peek().fast_commit, !was);
            assert_eq!(SAVES.load(Ordering::Relaxed), saves + 1, "and was saved");
            assert_eq!(doors.load(Ordering::Relaxed), 0, "nothing was dispatched");
        });
    }
}
