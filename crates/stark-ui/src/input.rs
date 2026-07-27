//! Pointer and keyboard input: turning DOM events into
//! [`InputCommand`](stark_core::InputCommand)s
//! (DESIGN.md §4).

use dioxus::html::geometry::ElementPoint;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use stark_core::InputSample;
use stark_core::command::{DocCommand, GestureCommand, ViewCommand};
use stark_core::document::SelectionMode;
use stark_core::document::SelectionOp;
use stark_core::geom::{Vec2, ViewTransform};

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
                DocCommand::Redo
            } else {
                DocCommand::Undo
            };
            dispatch(state, cmd);
            e.prevent_default();
        }
        Key::Character(c) if c.eq_ignore_ascii_case("y") => dispatch(state, DocCommand::Redo),
        // Selection commands (DESIGN.md §6.8). "Select all" and "Deselect" are the
        // same edit here — a selection covering the whole canvas *is* no selection —
        // so both shortcuts land on the same op.
        Key::Character(c) if c.eq_ignore_ascii_case("a") || c.eq_ignore_ascii_case("d") => {
            dispatch(state, DocCommand::Select(SelectionOp::select_all()));
            e.prevent_default();
        }
        Key::Character(c) if c.eq_ignore_ascii_case("i") && m.contains(Modifiers::SHIFT) => {
            dispatch(state, DocCommand::InvertSelection);
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

fn view_of(state: AppState) -> ViewTransform {
    state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.view())
        .expect("renderer ready during input")
}

/// Pointer position within an element, in CSS pixels.
pub fn elem_xy(e: &Event<PointerData>) -> Vec2 {
    let ElementPoint { x, y, .. } = e.element_coordinates();
    Vec2::new(x as f32, y as f32)
}

/// How finely a pointer report resolves position, in **CSS px**.
///
/// Not a preference — an estimate of the device's grain, which is what the fitter
/// needs in order to tell jitter from detail (see
/// [`PathFitter::with_tolerance`](stark_core::path::PathFitter::with_tolerance)). A
/// mouse walks the screen in whole *physical* pixels, so `1 / devicePixelRatio` CSS
/// px is its floor. A pen or a finger comes off a digitizer that resolves well below
/// the screen it sits under, so what limits those is the hand rather than the API;
/// half a physical pixel is a deliberate under-estimate — too fine only costs a few
/// extra control points, while too coarse rounds off detail that was really there.
fn input_resolution(e: &Event<PointerData>) -> f32 {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio() as f32)
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or(1.0);
    let physical = match e.pointer_type().as_str() {
        "pen" | "touch" => 0.5,
        _ => 1.0,
    };
    physical / dpr
}

/// The fitting tolerance to declare for a gesture starting with `e`, in canvas px:
/// the device's own grain (above) carried through `view`, since canvas space is
/// where the fit measures error.
pub fn input_tolerance_in(view: ViewTransform, e: &Event<PointerData>) -> f32 {
    input_resolution(e) / view.zoom
}

/// [`input_tolerance_in`] against the main canvas's view.
pub fn input_tolerance(state: AppState, e: &Event<PointerData>) -> f32 {
    input_tolerance_in(view_of(state), e)
}

/// Map an element-relative pointer position to a canvas-space input sample.
pub fn sample(state: AppState, e: &Event<PointerData>) -> InputSample {
    let view = view_of(state);
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
        dispatch(state, GestureCommand::End);
        drawing.set(false);
    }
    if let Some(base) = mode_restore.take() {
        dispatch(state, ViewCommand::SetSelectionMode(base));
    }
    panning.set(false);
}
