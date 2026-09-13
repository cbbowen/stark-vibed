//! The dialogs mounted at the app root, held as one stack (§25.7).
//!
//! The stack is in the order the dialogs were opened, and [`DialogStack`] mounts them
//! in that order, so the last one opened is both the one drawn on top and the one
//! Escape closes. The GPU-failure notice is not on it: it has no way out (§5).

use dioxus::prelude::*;

use crate::brush_editor::BrushEditorModal;
use crate::credits::CreditsModal;
use crate::panels::brush::PresetSaveModal;
use crate::settings::SettingsModal;
use crate::state::AppState;
use crate::substrates::NewDocumentModal;
use crate::{collab, drags, files, timings};

/// Every dialog the root can mount.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DialogId {
    /// "New document…" (`substrates::NewDocumentModal`).
    NewDocument,
    /// Share and join (`collab::SessionModal`).
    Session,
    /// "Export image…" (`files::ExportModal`).
    Export,
    /// The ⚙ preferences (`settings::SettingsModal`).
    Settings,
    /// Timing stats (§7.1, `timings::TimingModal`).
    TimingStats,
    /// Credits (`credits::CreditsModal`).
    Credits,
    /// The drag-preset offer (§25.8), raised by a release rather than a command.
    DragPresets,
    /// The brush editor (`brush_editor::BrushEditorModal`).
    BrushEditor,
    /// The name for a new preset (`panels::brush::PresetSaveModal`), usually opened
    /// over the brush editor.
    PresetSave,
}

/// Open `id` over whatever is open. A dialog already open keeps its place.
pub fn open(state: AppState, id: DialogId) {
    edit(state, |stack| push(stack, id));
}

/// Close `id`, wherever it stands in the stack.
pub fn close(state: AppState, id: DialogId) {
    edit(state, |stack| remove(stack, id));
}

/// Close the dialog on top; `true` if there was one — Escape's rung.
pub fn close_top(state: AppState) -> bool {
    edit(state, pop)
}

/// Whether `id` is open. A read, so an effect or a render asking it is woken when the
/// stack moves; asked from a handler, it subscribes nothing.
pub fn is_open(state: AppState, id: DialogId) -> bool {
    state.dialogs.read().contains(&id)
}

/// Whether any dialog is open — what stands `FinishMode` down.
pub fn any_open(state: AppState) -> bool {
    !state.dialogs.read().is_empty()
}

/// Apply `change` to a copy of the stack and write it back only if it moved: a `set`
/// wakes every reader whatever it is handed.
fn edit(state: AppState, change: impl FnOnce(&mut Vec<DialogId>) -> bool) -> bool {
    let mut stack = state.dialogs;
    let mut next = stack.peek().clone();
    let moved = change(&mut next);
    if moved {
        stack.set(next);
    }
    moved
}

fn push(stack: &mut Vec<DialogId>, id: DialogId) -> bool {
    if stack.contains(&id) {
        return false;
    }
    stack.push(id);
    true
}

fn remove(stack: &mut Vec<DialogId>, id: DialogId) -> bool {
    let before = stack.len();
    stack.retain(|open| *open != id);
    stack.len() != before
}

fn pop(stack: &mut Vec<DialogId>) -> bool {
    stack.pop().is_some()
}

/// Every open dialog, in the order it was opened.
///
/// Keyed by id, so opening a second dialog neither re-renders nor remounts the first,
/// and a dialog closed and opened again mounts fresh.
#[component]
pub fn DialogStack() -> Element {
    let state = use_context::<AppState>();
    let stack = state.dialogs.read().clone();
    rsx! {
        for id in stack {
            Dialog { key: "{id:?}", id }
        }
    }
}

/// One dialog, closed through the stack.
#[component]
fn Dialog(id: DialogId) -> Element {
    let state = use_context::<AppState>();
    let on_close = use_callback(move |()| close(state, id));
    match id {
        DialogId::NewDocument => rsx! { NewDocumentModal { on_close } },
        DialogId::Session => rsx! { collab::SessionModal { on_close } },
        DialogId::Export => rsx! { files::ExportModal { on_close } },
        DialogId::Settings => rsx! { SettingsModal { on_close } },
        DialogId::TimingStats => rsx! { timings::TimingModal { on_close } },
        DialogId::Credits => rsx! { CreditsModal { on_close } },
        DialogId::DragPresets => rsx! { drags::DragPresetModal { on_close } },
        DialogId::BrushEditor => rsx! { BrushEditorModal { on_close } },
        DialogId::PresetSave => rsx! { PresetSaveModal { on_close } },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_twice_keeps_one_entry() {
        let mut stack = Vec::new();
        assert!(push(&mut stack, DialogId::Settings));
        assert!(!push(&mut stack, DialogId::Settings), "already open");
        assert_eq!(stack, [DialogId::Settings]);
    }

    #[test]
    fn a_dialog_opened_over_another_is_the_top() {
        let mut stack = Vec::new();
        push(&mut stack, DialogId::BrushEditor);
        push(&mut stack, DialogId::PresetSave);
        assert_eq!(stack.last(), Some(&DialogId::PresetSave));
        // Opening the one underneath again does not lift it over the other.
        push(&mut stack, DialogId::BrushEditor);
        assert_eq!(stack.last(), Some(&DialogId::PresetSave));
    }

    #[test]
    fn close_removes_only_that_dialog() {
        let mut stack = vec![DialogId::BrushEditor, DialogId::PresetSave];
        assert!(remove(&mut stack, DialogId::BrushEditor));
        assert_eq!(stack, [DialogId::PresetSave]);
        assert!(!remove(&mut stack, DialogId::Credits), "not open");
        assert_eq!(stack, [DialogId::PresetSave]);
    }

    #[test]
    fn close_top_pops_the_last_and_says_whether_it_did() {
        let mut stack = vec![DialogId::BrushEditor, DialogId::PresetSave];
        assert!(pop(&mut stack));
        assert_eq!(stack, [DialogId::BrushEditor]);
        assert!(pop(&mut stack));
        assert!(!pop(&mut stack), "nothing left to close");
    }
}
