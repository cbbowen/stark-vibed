//! Pointer and keyboard input: turning DOM events into
//! [`InputCommand`](stark_core::InputCommand)s
//! (DESIGN.md §4).

use dioxus::dioxus_core::spawn_forever;
use dioxus::html::geometry::ElementPoint;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use crate::state::{AppState, dispatch, update_brush};
use stark_core::InputSample;
use stark_core::command::{DocCommand, GestureCommand, ViewCommand};
use stark_core::document::SelectionMode;
use stark_core::document::SelectionOp;
use stark_core::geom::{Vec2, ViewTransform};
use stark_core::{PickOptions, PickSource};

pub fn handle_keydown(mut state: AppState, e: &Event<KeyboardData>) {
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(true);
            e.prevent_default();
        }
        // Alt on its own focuses the browser's menu bar on Windows and Linux, which
        // would take the keyboard away the moment the eyedropper is reached for.
        Key::Alt => e.prevent_default(),
        _ => {}
    }

    let m = e.modifiers();
    track_alt(state, m);
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
    track_alt(state, e.modifiers());
}

/// Record whether Alt is held, so the canvas can wear the eyedropper cursor while it
/// is (MISSING_FEATURES §0.2).
///
/// Read off the event's **modifier set** rather than off the Alt key itself: a
/// keystroke that arrives after Alt was pressed or released while the window was not
/// focused then corrects the flag, instead of leaving it stuck on a press whose
/// release never came. Written only on a change, since every write re-renders the
/// canvas component.
fn track_alt(state: AppState, m: Modifiers) {
    let alt = m.contains(Modifiers::ALT);
    let mut held = state.pick.alt_down;
    if *held.peek() != alt {
        held.set(alt);
    }
}

/// Sample the canvas colour under `pos` and load the brush with it — the eyedropper
/// (MISSING_FEATURES §0.2).
///
/// One sample at a time. A pick is a render plus an asynchronous readback, and
/// Alt+drag asks for one per pointer move, so a move arriving while one is still in
/// flight is **dropped rather than queued**: queueing would spend a GPU submit per
/// pointer move and let an older sample land after a newer one, and for a sampler
/// being dragged only the latest answer matters anyway.
pub fn pick_color(state: AppState, pos: Vec2) {
    let mut busy = state.pick.busy;
    if *busy.peek() {
        return;
    }
    // The *choice* is what the panel holds; which layer it means is resolved here,
    // against whichever layer is selected at the moment of the sample.
    let all_layers = *state.pick.all_layers.peek();
    let active = state.obs.peek().as_ref().map(|o| o.active_layer);
    let options = PickOptions {
        source: match active {
            Some(id) if !all_layers => PickSource::Layer(id),
            _ => PickSource::Composite,
        },
        radius: *state.pick.radius.peek(),
    };

    // Render now and **drop the guard before awaiting** — the readback future owns
    // everything it needs, so nothing holds the renderer while the browser's event
    // loop runs the copy, which it must be free to do since the UI re-renders during
    // it (the same bargain `files::export_png` makes).
    let readback = {
        let mut renderer = state.renderer;
        let mut guard = renderer.write();
        let Some(r) = guard.as_mut() else { return };
        r.pick_color(pos, options)
    };
    busy.set(true);
    // Detached: the sample outlives the pointer gesture that asked for it (a release
    // must not cancel the answer to the press), and every signal it writes is
    // root-owned — see `state::root_signal`.
    spawn_forever(async move {
        let picked = readback.await;
        busy.set(false);
        // Nothing under the sampler: leave the brush as it was. Bare canvas is the
        // ground, not paint to pick up.
        let Some(rgb) = picked else { return };
        update_brush(state, |br| br.color = [rgb[0], rgb[1], rgb[2], br.color[3]]);
        // Tell the Color panel the colour moved from outside its own picker, so its
        // markers follow (see `AppState::color_epoch`).
        let mut epoch = state.color_epoch;
        let next = *epoch.peek() + 1;
        epoch.set(next);
    });
}

/// The engine's current view, or `None` before WebGPU init has finished.
///
/// Fallible rather than `expect`ing, because the canvas element is in the DOM and
/// taking pointer events from the first frame while [`render::init`](crate::render)
/// is still awaiting its adapter — so "a pointer event with no engine behind it" is
/// an ordinary early state, not a bug. Everything that needs canvas coordinates
/// therefore returns `None` too, and its callers do nothing until there is an engine
/// to do it to.
fn view_of(state: AppState) -> Option<ViewTransform> {
    state.renderer.read().as_ref().map(|r| r.view())
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

/// [`input_tolerance_in`] against the main canvas's view; `None` before the engine
/// exists.
pub fn input_tolerance(state: AppState, e: &Event<PointerData>) -> Option<f32> {
    Some(input_tolerance_in(view_of(state)?, e))
}

/// Map an element-relative pointer position to a canvas-space input sample; `None`
/// before the engine exists, since there is no view to map through yet.
pub fn sample(state: AppState, e: &Event<PointerData>) -> Option<InputSample> {
    let view = view_of(state)?;
    Some(InputSample {
        pos: view.screen_to_canvas(elem_xy(e)),
        pressure: e.pressure(),
        // Pen tilt (degrees from vertical, ±90 per axis) → a canvas-space lean vector. The
        // palette knife's deposit reads its component along the stroke direction (DESIGN
        // §6.2); a mouse reports (0, 0), so the deposit falls back to its constant rate.
        tilt: Vec2::new(e.tilt_x() as f32, e.tilt_y() as f32) / 90.0,
        ..Default::default()
    })
}

/// End any in-progress stroke, selection gesture, pan, or eyedropper drag, and put
/// back the selection mode a modifier key overrode for the gesture (DESIGN.md §6.8).
/// The canvas is no longer in hand once this returns, so the floating chrome fades
/// back in.
pub fn end_interaction(
    mut state: AppState,
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
    // Not a parameter like the two above because the eyedropper's drag flag is shared
    // state, not the canvas's own — the options bar reads it (see `PickState`).
    // Nothing to undo, either: a sample already in flight is left to land, since it
    // is the answer to a press the user made.
    state.pick.dragging.set(false);
    state.canvas_active.set(false);
}
