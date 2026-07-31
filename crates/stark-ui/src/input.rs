//! Pointer and keyboard input: turning DOM events into
//! [`InputCommand`](stark_core::InputCommand)s
//! (§4).

use std::str::FromStr;

use dioxus::dioxus_core::spawn_forever;
use dioxus::html::geometry::ElementPoint;
use dioxus::html::input_data::MouseButton;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use crate::platform::{capture_pointer, on_window_key};
use crate::state::{AppState, dispatch, update_brush};
use stark_core::InputSample;
use stark_core::command::{DocCommand, GestureCommand, ViewCommand};
use stark_core::document::SelectionOp;
use stark_core::document::ShapeAction;
use stark_core::geom::{Vec2, ViewTransform};
use stark_core::{PickOptions, PickSource};

/// The view-navigation bindings — middle-drag / space-drag pan, cursor-anchored
/// wheel zoom — shared by every surface that sits over the canvas: the canvas
/// itself and the transform mode's catcher, box and handles. One implementation,
/// so what "the pan buttons" and "the zoom rate" mean cannot drift between
/// surfaces.
///
/// Each surface makes its own with [`Nav::use_nav`]; the pointer capture on the
/// pressed element keeps two instances from ever panning at once. Policy stays
/// at the call site — the canvas fades the chrome while it pans, the transform
/// overlay deliberately does not — only the mechanics live here.
#[derive(Clone, Copy)]
pub struct Nav {
    state: AppState,
    /// The pan in flight: the pointer's last position in **page px** (the one
    /// frame every surface reports in, whatever its own origin), or `None`.
    last: Signal<Option<Vec2>>,
}

impl Nav {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_nav(state: AppState) -> Self {
        Self {
            state,
            last: use_signal(|| None),
        }
    }

    /// Whether `e` presses the pan bindings — middle button anywhere, or space
    /// with the primary button — and if so, begin the pan: capture the pointer
    /// and swallow the event. `true` means "this press is navigation, not
    /// yours"; callers check it before starting their own gesture.
    pub fn start_pan(self, e: &Event<PointerData>) -> bool {
        let pan = match e.trigger_button() {
            Some(MouseButton::Auxiliary) => true,
            Some(MouseButton::Primary) => *self.state.space_down.peek(),
            _ => false,
        };
        if pan {
            e.prevent_default(); // suppress middle-click autoscroll
            e.stop_propagation();
            capture_pointer(e);
            let mut last = self.last;
            last.set(Some(page_xy(e)));
        }
        pan
    }

    /// Advance the pan in flight, if any. `Pan` is an incremental command, so
    /// the anchor is re-set each move. `true` means the move was navigation and
    /// the caller's own gesture logic should not see it; a no-op otherwise.
    pub fn pan_move(self, e: &Event<PointerData>) -> bool {
        let mut last = self.last;
        let Some(prev) = last() else { return false };
        let p = page_xy(e);
        dispatch(self.state, ViewCommand::Pan { delta: p - prev });
        last.set(Some(p));
        true
    }

    /// End the pan in flight (release / cancel). Harmless when there is none.
    pub fn stop(self) {
        let mut last = self.last;
        if last.peek().is_some() {
            last.set(None);
        }
    }

    /// Cursor-anchored wheel zoom. Anchored by page position: it equals the
    /// canvas's own coordinates for full-viewport surfaces, and it is the only
    /// frame an element like the transform box (whose local coordinates move
    /// with it) can meaningfully report.
    pub fn wheel(self, e: Event<WheelData>) {
        e.prevent_default();
        e.stop_propagation();
        let dy = e.delta().strip_units().y;
        if dy != 0.0 {
            let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
            let p = e.page_coordinates();
            dispatch(
                self.state,
                ViewCommand::Zoom {
                    anchor: Vec2::new(p.x as f32, p.y as f32),
                    factor,
                },
            );
        }
    }
}

/// Pointer position in page coordinates — the frame that stays still while
/// absolutely-positioned chrome (frame handles, the transform box) moves under
/// the pointer mid-drag.
pub fn page_xy(e: &Event<PointerData>) -> Vec2 {
    let p = e.page_coordinates();
    Vec2::new(p.x as f32, p.y as f32)
}

/// Bind the app's keyboard shortcuts, once, for the life of the page.
///
/// On the window rather than on the app's root element — see
/// [`platform::on_window_key`] for why an element cannot hold them.
///
/// Only the **keydown** side is withheld from a field being typed into. Keyup is
/// what disarms `space_down` and corrects `alt_down`, and focus can move between a
/// press and its release — a click into the rename field with space held — so a
/// guarded keyup would leave the pan armed with nothing to release it. Nothing is
/// given up by letting it through: on keyup there is no default action left to
/// cancel, since a character is inserted on the press.
pub fn bind_shortcuts(state: AppState) {
    on_window_key("keydown", move |e| {
        if !typing_into_a_field(&e) {
            handle_keydown(state, &e);
        }
    });
    on_window_key("keyup", move |e| handle_keyup(state, &e));
}

/// Whether `e` was typed into a control that owns its own keystrokes — a text
/// field, a `<select>`, a contenteditable region.
///
/// Read off the event's target, which for a key event *is* what has focus, rather
/// than off a flag the fields set on focus and clear on blur. A field that unmounts
/// while focused — commit-and-close on a rename — never fires its blur, and a flag
/// left stuck on would kill every shortcut for the rest of the session. The DOM is
/// asked at the moment of the keystroke, so it cannot fall out of step.
///
/// Declining here is also what hands the field the browser's own editing bindings:
/// Ctrl+Z undoes the *text* rather than the document, and Ctrl+A selects the text
/// rather than the canvas, purely because nothing calls `prevent_default` on them.
///
/// This is the *only* place a widget can opt out. `e.stop_propagation()` in an
/// element's own `onkeydown` will not do it: dioxus-web reads `prevent_default`
/// off a handled event but never calls `stopPropagation` on the underlying DOM
/// event, so propagation is halted inside the virtual tree only and the real event
/// reaches the window regardless.
fn typing_into_a_field(e: &web_sys::KeyboardEvent) -> bool {
    use wasm_bindgen::JsCast;

    let Some(el) = e
        .target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlElement>().ok())
    else {
        return false;
    };
    el.is_content_editable()
        || match el.tag_name().as_str() {
            "TEXTAREA" | "SELECT" => true,
            // Sliders, checkboxes and colour wells are not text entry. They want
            // arrows and space from the browser, but Ctrl+Z over one still means
            // the document — there is no text there for it to mean anything else.
            "INPUT" => !matches!(
                el.unchecked_ref::<web_sys::HtmlInputElement>()
                    .type_()
                    .as_str(),
                "button" | "checkbox" | "color" | "file" | "radio" | "range" | "reset" | "submit"
            ),
            _ => false,
        }
}

/// The pressed key, in the same typed vocabulary the rsx! handlers read.
fn key_of(e: &web_sys::KeyboardEvent) -> Key {
    Key::from_str(&e.key()).unwrap_or(Key::Unidentified)
}

/// The modifier set held during `e`.
fn modifiers_of(e: &web_sys::KeyboardEvent) -> Modifiers {
    let mut m = Modifiers::empty();
    if e.alt_key() {
        m.insert(Modifiers::ALT);
    }
    if e.ctrl_key() {
        m.insert(Modifiers::CONTROL);
    }
    if e.meta_key() {
        m.insert(Modifiers::META);
    }
    if e.shift_key() {
        m.insert(Modifiers::SHIFT);
    }
    m
}

fn handle_keydown(mut state: AppState, e: &web_sys::KeyboardEvent) {
    match key_of(e) {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(true);
            e.prevent_default();
        }
        // Alt on its own focuses the browser's menu bar on Windows and Linux, which
        // would take the keyboard away the moment the eyedropper is reached for.
        Key::Alt => e.prevent_default(),
        _ => {}
    }

    let m = modifiers_of(e);
    track_alt(state, m);
    if !(m.contains(Modifiers::CONTROL) || m.contains(Modifiers::META)) {
        // Unmodified letters: the view bindings. Checked here, after the modifier set
        // is known, so `Ctrl+H` stays the browser's and only a bare press is ours.
        if !m.contains(Modifiers::ALT)
            && let Key::Character(c) = key_of(e)
            && c.eq_ignore_ascii_case("h")
        {
            // Screen-relative, so it swaps the left of the screen with the right
            // whatever angle the canvas is at (`ViewTransform::mirror_screen_h`).
            dispatch(state, ViewCommand::MirrorH);
            e.prevent_default();
        }
        return;
    }
    match key_of(e) {
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
        // Selection commands (§6.8). "Select all" and "Deselect" are the
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

fn handle_keyup(mut state: AppState, e: &web_sys::KeyboardEvent) {
    match key_of(e) {
        Key::Character(c) if c.eq_ignore_ascii_case(" ") => {
            state.space_down.set(false);
            e.prevent_default();
        }
        _ => {}
    }
    track_alt(state, modifiers_of(e));
}

/// Record whether Alt is held, so the canvas can wear the eyedropper cursor while it
/// is (§18.0.2).
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
/// (§18.0.2).
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
        // palette knife's deposit reads its component along the stroke direction
        // (§6.2); a mouse reports (0, 0), so the deposit falls back to its constant rate.
        tilt: Vec2::new(e.tilt_x() as f32, e.tilt_y() as f32) / 90.0,
        ..Default::default()
    })
}

/// End any in-progress stroke, shape gesture, pan, or eyedropper drag, and put back
/// the shape action a modifier key overrode for the gesture (§6.8). The
/// canvas is no longer in hand once this returns, so the floating chrome fades back
/// in.
pub fn end_interaction(
    mut state: AppState,
    drawing: &mut Signal<bool>,
    nav: Nav,
    action_restore: &mut Signal<Option<ShapeAction>>,
) {
    if drawing() {
        dispatch(state, GestureCommand::End);
        drawing.set(false);
    }
    if let Some(base) = action_restore.take() {
        dispatch(state, ViewCommand::SetShapeAction(base));
    }
    nav.stop();
    // Not a parameter like the two above because the eyedropper's drag flag is shared
    // state, not the canvas's own — the options bar reads it (see `PickState`).
    // Nothing to undo, either: a sample already in flight is left to land, since it
    // is the answer to a press the user made.
    state.pick.dragging.set(false);
    state.canvas_active.set(false);
}
