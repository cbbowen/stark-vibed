//! Pointer and keyboard input: turning DOM events into
//! [`InputCommand`](stark_core::InputCommand)s
//! (§4).

use std::str::FromStr;

use dioxus::dioxus_core::spawn_forever;
use dioxus::html::geometry::ElementPoint;
use dioxus::html::input_data::MouseButton;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use crate::collab::now_seconds;
use crate::platform::{capture_pointer, on_window_key, sleep_ms};
use crate::state::{AppState, Dwell, dispatch, update_brush};
use stark_core::InputSample;
use stark_core::command::{DocCommand, GestureCommand, ViewCommand};
use stark_core::document::SelectionOp;
use stark_core::document::ShapeAction;
use stark_core::geom::{Vec2, ViewTransform};
use stark_core::{PickOptions, PickSource};

/// How close to a quarter turn a turn has to land to be pulled onto it, radians
/// (about 5°).
///
/// The frontend's to decide, like the fitting tolerance: it is a property of turning
/// something with a hand rather than of the view. Without it a canvas that has been
/// turned could only be *approximately* straightened, and a piece left a degree off
/// square reads as an accident rather than as a choice.
pub const TURN_SNAP: f32 = 0.09;

/// How far a two-finger gesture has to twist before it turns the canvas at all,
/// radians (about 6°).
///
/// Not a tolerance on the input — a touchscreen resolves an order finer than this.
/// It is the width of the band in which a pinch is *only* a pinch: two fingers
/// closing on a target do not travel along the line between them, they roll about
/// the hand, and without a band to spend that in every zoom would leave the canvas a
/// couple of degrees off true. Measured from the start of the gesture and subtracted
/// once it is crossed, so the turn picks up from where the hand is instead of
/// jumping by the width of the band.
const TWIST_DEADZONE: f32 = 0.10;

/// How far apart two fingers have to be for the pair to mean anything, in page px.
///
/// Below this the pair's *direction* is noise and its length is a divisor, so a
/// pinch reported through it would be an arbitrary rotation and an unbounded zoom.
/// Pinches that close this far are simply not reported; the fingers slip, which is
/// what two fingers on the same spot deserve.
const MIN_SPAN: f32 = 8.0;

/// The view-navigation bindings — two-finger pan/zoom/turn, middle-drag and
/// space-drag pan, cursor-anchored wheel zoom — shared by every surface that sits
/// over the canvas: the canvas itself and the transform mode's catcher, box and
/// handles. One implementation, so what "the pan bindings" and "the zoom rate" mean
/// cannot drift between surfaces.
///
/// Each surface makes its own with [`Nav::use_nav`]; the pointer capture on the
/// pressed element keeps two instances from ever navigating at once. Policy stays
/// at the call site — the canvas fades the chrome while it pans and cancels the
/// stroke a second finger interrupted, the transform overlay deliberately does
/// neither — only the mechanics live here.
///
/// The three entry points are a lifecycle and are meant to be called as one:
/// [`begin`](Self::begin) on press, [`advance`](Self::advance) on move,
/// [`release`](Self::release) on release or cancel. Each answers the same question —
/// *was this event mine?* — so a surface routes its pointers by asking three times
/// and never by inspecting buttons or pointer types itself.
#[derive(Clone, Copy)]
pub struct Nav {
    state: AppState,
    /// The button-drag pan in flight: the pointer's last position in **page px**
    /// (the one frame every surface reports in, whatever its own origin), or `None`.
    last: Signal<Option<Vec2>>,
    /// The fingers on this surface (§18.1.7). Separate from `last` because a
    /// finger is identified by its id rather than by being *the* pointer — that is
    /// the whole difference touch makes.
    touch: Signal<Touch>,
}

/// The fingers on one surface, and the two-finger gesture they are making
/// (§18.1.7).
#[derive(Clone, Default)]
struct Touch {
    /// Every finger down, in the order it landed: pointer id and last position in
    /// page px. The order is load-bearing — the gesture is made by the **first two**,
    /// so a third finger joining changes nothing and a lift re-forms the pair from
    /// whoever is left, both without a jump.
    down: Vec<(i32, Vec2)>,
    /// The gesture in flight. Born when a second finger lands and buried when the
    /// last one lifts — deliberately outliving the *second* finger, so a pinch that
    /// ends with one finger still on the glass keeps panning rather than going dead
    /// under a hand that never left.
    pinch: Option<Pinch>,
}

/// What a two-finger gesture accumulates: the part of it that a pair of finger
/// positions cannot say, because it is a fact about the gesture's whole history.
#[derive(Copy, Clone)]
struct Pinch {
    /// The view's angle when the gesture began — what the twist is measured from.
    from: f32,
    /// Raw twist since then, radians, **before** the deadzone. Raw so that twisting
    /// back out of the band un-turns by exactly as much as twisting into it did:
    /// accumulating the deadzoned angle instead would ratchet, and a gesture that
    /// wandered a degree either way would walk the canvas around.
    twist: f32,
    /// The angle last asked for. Each report is the step from here, so the snap's
    /// pull onto a quarter turn is spent once instead of re-applied every move.
    asked: f32,
}

impl Nav {
    /// A hook: call unconditionally, like any `use_*`.
    pub fn use_nav(state: AppState) -> Self {
        Self {
            state,
            last: use_signal(|| None),
            touch: use_signal(Touch::default),
        }
    }

    /// Whether `e` is a press this takes as navigation — a second finger on the
    /// glass, the middle button anywhere, or space with the primary button — and if
    /// so, begin: capture the pointer and swallow the event. `true` means "this
    /// press is navigation, not yours"; callers check it before starting their own
    /// gesture, and abandon any gesture already in flight.
    pub fn begin(self, e: &Event<PointerData>) -> bool {
        if is_finger(e) {
            return self.finger_down(e);
        }
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

    /// Advance the navigation in flight, if any. `true` means the move was
    /// navigation and the caller's own gesture logic should not see it.
    ///
    /// Not a no-op when it answers `false`: a lone finger's moves are recorded even
    /// while it paints, because a second finger landing has to pair with where the
    /// first one has got to rather than with where it pressed. So call it on **every**
    /// move, ahead of whatever the surface does with the ones it keeps — not only on
    /// the ones a gesture has left over.
    pub fn advance(self, e: &Event<PointerData>) -> bool {
        if is_finger(e) {
            return self.finger_move(e);
        }
        let mut last = self.last;
        let Some(prev) = last() else { return false };
        let p = page_xy(e);
        // `Pan` is incremental, so the anchor is re-set each move.
        dispatch(self.state, ViewCommand::Pan { delta: p - prev });
        last.set(Some(p));
        true
    }

    /// Report a release or a cancel. `true` means fingers are **still down** and the
    /// interaction is not over, so the caller should hold its own teardown: lifting
    /// one finger of a pinch ends nothing, and a surface that tore down there would
    /// end the gesture on whichever finger the hand happened to raise first.
    ///
    /// Always `false` for a mouse or a pen, which have nothing to be the rest of.
    pub fn release(self, e: &Event<PointerData>) -> bool {
        if !is_finger(e) {
            return false;
        }
        let mut touch = self.touch;
        let mut t = touch.write();
        t.down.retain(|(id, _)| *id != e.pointer_id());
        if t.down.is_empty() {
            t.pinch = None;
            return false;
        }
        true
    }

    /// End the navigation in flight, whatever it was. Harmless when there is none.
    pub fn stop(self) {
        let mut last = self.last;
        if last.peek().is_some() {
            last.set(None);
        }
        let mut touch = self.touch;
        if !touch.peek().down.is_empty() {
            touch.set(Touch::default());
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

    /// A finger landing. `true` once there are two of them, which is where the
    /// gesture becomes navigation.
    fn finger_down(self, e: &Event<PointerData>) -> bool {
        // Read before the fingers are locked, so nothing holds two signals at once.
        let angle = view_of(self.state).map_or(0.0, |v| v.rotation);
        let mut touch = self.touch;
        let mut t = touch.write();
        // A *primary* touch is the first contact of its type, so anything still
        // listed when one arrives is a finger whose release never came — a cancel the
        // browser swallowed, a tab switched away from mid-gesture. Cleared on the
        // fact that says so rather than by trying to catch every way a release can go
        // missing, since a single stale entry would make the next lone finger a pinch
        // and painting by touch would stop working for the rest of the session.
        if e.is_primary() {
            *t = Touch::default();
        }
        let id = e.pointer_id();
        if !t.down.iter().any(|(down, _)| *down == id) {
            t.down.push((id, page_xy(e)));
        }
        if t.down.len() < 2 {
            return false; // one finger is the caller's — it paints
        }
        e.prevent_default();
        e.stop_propagation();
        capture_pointer(e);
        if t.pinch.is_none() {
            t.pinch = Some(Pinch {
                from: angle,
                twist: 0.0,
                asked: angle,
            });
        }
        true
    }

    /// A finger moving. Drives the view once a gesture is in flight; before then it
    /// only records where the finger is, which is what lets a second finger landing
    /// mid-stroke pair with where the first one *is* rather than where it pressed.
    fn finger_move(self, e: &Event<PointerData>) -> bool {
        let mut touch = self.touch;
        let mut t = touch.write();
        let id = e.pointer_id();
        let Some(i) = t.down.iter().position(|(down, _)| *down == id) else {
            return false;
        };
        let now = page_xy(e);
        let was = std::mem::replace(&mut t.down[i].1, now);
        let Some(mut pinch) = t.pinch else {
            return false; // a lone finger with no gesture behind it: painting
        };

        let command = if t.down.len() < 2 {
            // Down to the gesture's last finger: a plain pan, so the canvas stays in
            // hand until the hand itself leaves.
            ViewCommand::Pan { delta: now - was }
        } else {
            // A third finger is a bystander — the gesture is the first two — and its
            // moves are swallowed rather than acted on, so a hand resting on the glass
            // mid-pinch does not fight the two fingers doing the work.
            if i > 1 {
                return true;
            }
            // The pair as it was and as it now is. One finger moved, so the other side
            // of the pair is the same in both.
            let (a, b) = (t.down[0].1, t.down[1].1);
            let (before, after) = if i == 0 {
                ((was, b), (a, b))
            } else {
                ((a, was), (a, b))
            };
            let (u, v) = (before.1 - before.0, after.1 - after.0);
            let (span, spans) = (u.length(), v.length());
            if span < MIN_SPAN || spans < MIN_SPAN {
                return true;
            }
            // The twist the hand has actually made, then what the canvas is asked to
            // do about it: nothing until the deadzone has been crossed, and pulled
            // onto a quarter turn whenever the result lands near one.
            pinch.twist += u.angle_to(v);
            let earned = (pinch.twist.abs() - TWIST_DEADZONE).max(0.0) * pinch.twist.signum();
            let asked = snap_quarter(pinch.from + earned);
            let command = ViewCommand::Pinch {
                anchor: 0.5 * (before.0 + before.1),
                to: 0.5 * (after.0 + after.1),
                scale: spans / span,
                turn: asked - pinch.asked,
            };
            pinch.asked = asked;
            t.pinch = Some(pinch);
            command
        };
        // Released before dispatching: the command re-enters the engine and rewrites
        // the frontend's observable, and nothing that runs there should be able to
        // find this surface's fingers half-updated.
        drop(t);
        dispatch(self.state, command);
        true
    }
}

/// Whether `e` came from a finger — the one pointer type that arrives in pairs.
///
/// A pen is deliberately not one, on the same screen and through the same API: it
/// reports a single contact, and the whole point of the two-finger gesture is to be
/// able to move the canvas *without* putting the pen down.
fn is_finger(e: &Event<PointerData>) -> bool {
    e.pointer_type() == "touch"
}

/// `to` pulled onto the nearest quarter turn if it is within [`TURN_SNAP`] of one.
pub fn snap_quarter(to: f32) -> f32 {
    let quarter = (to / std::f32::consts::FRAC_PI_2).round() * std::f32::consts::FRAC_PI_2;
    if (to - quarter).abs() <= TURN_SNAP {
        quarter
    } else {
        to
    }
}

/// The signed turn from `from` to `to`, the short way round — so easing between two
/// angles never takes the long way about, and 1° short of a full circle is 1°.
pub fn shortest_turn(from: f32, to: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (to - from + PI).rem_euclid(TAU) - PI
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

// ---------------------------------------------------------------------------
// The hold, for the drawing assist (§6.9)
// ---------------------------------------------------------------------------

/// How long the pointer has to hold still before the stroke in flight snaps.
///
/// The cost of the two mistakes is asymmetric, and that is what sets it. Too short and
/// the natural pause before lifting the pen turns considered strokes into shapes; too
/// long and the gesture just feels unresponsive and gets tried again. So it sits at the
/// long end of what still reads as immediate — the same figure Procreate settled on.
const DWELL: f64 = 0.45;

/// How far the pointer may drift and still count as held, in **CSS px**: a hand's own
/// tremor on a pen resting against the glass, plus the digitizer's noise.
const DWELL_SLOP: f32 = 4.0;

/// How often the watcher looks. Well under a tenth of [`DWELL`], so the snap lands
/// within a frame or two of the hold being earned, and far too rare to cost anything.
const DWELL_POLL_MS: i32 = 60;

/// Begin watching a stroke gesture for a hold (§6.9). A no-op when the assist is off.
///
/// `at` is the press position in element (CSS) px — the frame the dwell is measured in;
/// see [`Dwell::at`].
pub fn watch_for_hold(state: AppState, at: Vec2) {
    if !*state.assist.enabled.peek() {
        return;
    }
    let mut dwell = state.assist.dwell;
    dwell.set(Some(Dwell {
        at,
        since: now_seconds(),
        fired: false,
    }));
    // `spawn_forever` for the reason `request_paint` uses it: this is started from a
    // component's event handler and must not be tied to that scope's lifetime. Every
    // signal it touches is root-owned (see `state::root_signal`).
    let task = spawn_forever(async move {
        let mut dwell = state.assist.dwell;
        loop {
            sleep_ms(DWELL_POLL_MS).await;
            // Cleared means the gesture is over, and that is the only way out: the
            // watcher runs for as long as the pointer is down, because a hold that was
            // declined (or that snapped a shape now being steered) may be followed by
            // another worth reporting.
            let Some(held) = *dwell.peek() else { return };
            if held.fired || now_seconds() - held.since < DWELL {
                continue;
            }
            // Latch *before* dispatching, so a pointer that simply stays put does not
            // report the same hold thirty times a second.
            dwell.set(Some(Dwell {
                fired: true,
                ..held
            }));
            dispatch(state, GestureCommand::Hold);
        }
    });
    let mut watcher = state.assist.task;
    if let Some(old) = watcher.write().replace(task) {
        old.cancel();
    }
}

/// Report a pointer move against the hold being watched, in element (CSS) px.
///
/// Restarts the clock only when the pointer has actually gone somewhere: a pen resting
/// on glass reports continuously, so treating every report as movement would mean the
/// dwell never completed on the one device the feature exists for.
pub fn pointer_moved(state: AppState, at: Vec2) {
    let mut dwell = state.assist.dwell;
    let Some(held) = *dwell.peek() else { return };
    if held.at.distance(at) > DWELL_SLOP {
        dwell.set(Some(Dwell {
            at,
            since: now_seconds(),
            fired: false,
        }));
    }
}

/// Stop watching for a hold. Harmless when nothing is being watched.
pub fn stop_watching(state: AppState) {
    let mut dwell = state.assist.dwell;
    if dwell.peek().is_some() {
        dwell.set(None);
    }
    let mut task = state.assist.task;
    if let Some(task) = task.write().take() {
        task.cancel();
    }
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
    restore_action(state, action_restore);
    stop_watching(state);
    nav.stop();
    // Not a parameter like the two above because the eyedropper's drag flag is shared
    // state, not the canvas's own — the options bar reads it (see `PickState`).
    // Nothing to undo, either: a sample already in flight is left to land, since it
    // is the answer to a press the user made.
    state.pick.dragging.set(false);
    state.canvas_active.set(false);
}

/// Abandon the gesture in flight **without committing it** — what a press that turns
/// out to be navigation does to the stroke it interrupted (§18.1.7). Harmless when
/// there is none.
///
/// [`GestureCommand::Cancel`] rather than `End`, and that is the whole point: a
/// second finger landing means the first was never drawing, it was opening a pinch,
/// so navigating must leave no mark. The same applies to a middle-drag begun
/// mid-stroke, which is the one other way [`Nav::begin`] can answer `true` with a
/// gesture already running.
///
/// Not folded into [`end_interaction`]: that one is the *end* of the interaction and
/// hands the canvas back (the chrome fades in, the pan stops). This one is the
/// interaction changing its mind about what it is, mid-flight, and the canvas is
/// still very much in hand.
pub fn abandon_gesture(
    state: AppState,
    drawing: &mut Signal<bool>,
    action_restore: &mut Signal<Option<ShapeAction>>,
) {
    if drawing() {
        dispatch(state, GestureCommand::Cancel);
        drawing.set(false);
    }
    restore_action(state, action_restore);
    stop_watching(state);
}

/// Put back the shape action a gesture's modifier keys overrode (§6.8).
fn restore_action(state: AppState, action_restore: &mut Signal<Option<ShapeAction>>) {
    if let Some(base) = action_restore.take() {
        dispatch(state, ViewCommand::SetShapeAction(base));
    }
}
