//! The app's shared state handle and the command path everything funnels through.
//!
//! [`dispatch`] is the single seam between the UI and the engine: every mutation
//! goes through it, so repaint, observable refresh and collaboration broadcast
//! happen in one place rather than at each call site (DESIGN.md §4).

use dioxus::dioxus_core::Task;
use dioxus::prelude::*;

use crate::collab;
use crate::render::Renderer;
use stark_core::command::ViewCommand;
use stark_core::document::BrushParams;
use stark_core::{InputCommand, ObservableState};

/// Shared `Copy` handle to the app's signals. Provided once by `app` and read back
/// through `use_context` wherever a component needs to reach the engine.
#[derive(Clone, Copy)]
pub struct AppState {
    /// Surface + engine, built asynchronously once the canvas mounts. `None`
    /// until WebGPU init completes. Not `Send` — lives in unsync storage.
    pub renderer: Signal<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command.
    pub obs: Signal<Option<ObservableState>>,
    /// Whether the user is holding space.
    pub space_down: Signal<bool>,
    /// Whether the brush editor dialog is open (rendered at the app root so its
    /// backdrop escapes the panels' `backdrop-filter` containing blocks).
    pub brush_editor_open: Signal<bool>,
    /// Everything to do with a shared drawing (DESIGN.md §12).
    pub collab: CollabState,
}

/// The shared-session signals, grouped because they share one lifecycle: they are
/// set together when a session starts and cleared together when it ends, and
/// nothing outside [`crate::collab`] should be writing them piecemeal.
#[derive(Clone, Copy)]
pub struct CollabState {
    /// The live session, if any. `!Send` iroh handles live in unsync storage
    /// beside the renderer.
    pub session: Signal<Option<stark_net::CollabSession>>,
    /// The shareable ticket string, while hosting/joined.
    pub ticket: Signal<Option<String>>,
    /// Where the session lifecycle stands (drives the dialog + rail badge).
    pub phase: Signal<collab::CollabPhase>,
    /// The last share/join failure, surfaced in the dialog.
    pub error: Signal<Option<String>>,
    /// The incoming-event pump for `session`. Its lifetime is tied to the
    /// session's: `collab::install` replaces it, [`crate::collab::leave`]
    /// cancels it.
    pub pump: Signal<Option<Task>>,
}

/// Apply a command, repaint the surface, and refresh the observable snapshot.
/// In a shared session, whatever the command committed is then broadcast.
pub fn dispatch(state: AppState, command: impl Into<InputCommand>) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    {
        let mut guard = renderer.write();
        if let Some(r) = guard.as_mut() {
            r.process(command);
            r.paint();
            obs.set(Some(r.observe()));
        }
    }
    collab::flush_outbox(state);
}

/// Resize the surface/engine, then repaint.
pub fn resize(state: AppState, width: u32, height: u32) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.resize(width, height);
        r.paint();
        obs.set(Some(r.observe()));
    }
}

/// Read the current brush, mutate a copy, and commit it (releasing the `obs`
/// read guard before `dispatch` writes — avoids an AlreadyBorrowed panic).
pub fn update_brush(state: AppState, f: impl FnOnce(&mut BrushParams)) {
    let brush = state.obs.read().as_ref().map(|o| o.brush);
    if let Some(mut brush) = brush {
        f(&mut brush);
        dispatch(state, ViewCommand::SetBrush(brush));
    }
}
