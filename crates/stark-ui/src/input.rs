//! Pointer and keyboard input: turning DOM events into [`InputCommand`]s
//! (DESIGN.md §4).

use dioxus::html::geometry::ElementPoint;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use stark_core::document::SelectionMode;
use stark_core::document::SelectionOp;
use stark_core::geom::Vec2;
use stark_core::{InputCommand, InputSample};

pub fn handle_keydown(mut state: AppState, e: &Event<KeyboardData>) {
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(true);
            e.prevent_default();
        }
        _ => {}
    }

    let m = e.modifiers();
    if !(m.contains(Modifiers::CONTROL) || m.contains(Modifiers::META)) {
        return;
    }
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case("z") => {
            let cmd = if m.contains(Modifiers::SHIFT) {
                InputCommand::Redo
            } else {
                InputCommand::Undo
            };
            dispatch(state, cmd);
            e.prevent_default();
        }
        Key::Character(c) if c.eq_ignore_ascii_case("y") => dispatch(state, InputCommand::Redo),
        // Selection commands (DESIGN.md §6.8). "Select all" and "Deselect" are the
        // same edit here — a selection covering the whole canvas *is* no selection —
        // so both shortcuts land on the same op.
        Key::Character(c) if c.eq_ignore_ascii_case("a") || c.eq_ignore_ascii_case("d") => {
            dispatch(state, InputCommand::Select(SelectionOp::select_all()));
            e.prevent_default();
        }
        Key::Character(c) if c.eq_ignore_ascii_case("i") && m.contains(Modifiers::SHIFT) => {
            dispatch(state, InputCommand::InvertSelection);
            e.prevent_default();
        }
        _ => {}
    }
}

pub fn handle_keyup(mut state: AppState, e: &Event<KeyboardData>) {
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(false);
            e.prevent_default();
        }
        _ => {}
    }
}

/// Pointer position within an element, in CSS pixels.
pub fn elem_xy(e: &Event<PointerData>) -> Vec2 {
    let ElementPoint { x, y, .. } = e.element_coordinates();
    Vec2::new(x as f32, y as f32)
}

/// Map an element-relative pointer position to a canvas-space input sample.
pub fn sample(state: AppState, e: &Event<PointerData>) -> InputSample {
    let view = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.view())
        .expect("renderer ready during input");
    InputSample {
        pos: view.screen_to_canvas(elem_xy(e)),
        pressure: e.pressure(),
        // Pen tilt (degrees from vertical, ±90 per axis) → a canvas-space lean vector. The
        // palette knife's deposit reads its component along the stroke direction (DESIGN
        // §6.2); a mouse reports (0, 0), so the deposit falls back to its constant rate.
        tilt: Vec2::new(e.tilt_x() as f32, e.tilt_y() as f32) / 90.0,
        ..Default::default()
    }
}

/// End any in-progress stroke, selection gesture, or pan, and put back the selection
/// mode a modifier key overrode for the gesture (DESIGN.md §6.8).
pub fn end_interaction(
    state: AppState,
    drawing: &mut Signal<bool>,
    panning: &mut Signal<bool>,
    mode_restore: &mut Signal<Option<SelectionMode>>,
) {
    if drawing() {
        dispatch(state, InputCommand::EndStroke);
        drawing.set(false);
    }
    if let Some(base) = mode_restore.take() {
        dispatch(state, InputCommand::SetSelectionMode(base));
    }
    panning.set(false);
}
