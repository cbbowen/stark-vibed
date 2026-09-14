//! Pointer and keyboard input: turning DOM events into
//! [`InputCommand`](stark_engine::command::InputCommand)s
//! (§4).
//!
//! # A file per gesture, and this one for what they share
//!
//! Five gesture objects, each a `Copy` hook with the same shape — `begin` on a
//! press, `advance` on a move, `stop`/`end` on a release or cancel, each
//! answering *was this event mine?* — and each owning its own in-flight state
//! (§25.3). They are independent of one another, so they are a file each:
//! [`Nav`] the view, [`Tune`] the brush, [`PickMove`] the layer carry, and
//! [`Paint`] with the [`Landing`] that holds a finger's press in front of it.
//! [`Gestures`](gestures::Gestures) carries the canvas's four as one value — which of them holds the
//! pointer, the order a move is offered to them in, and the release that puts
//! them all down ([`end_interaction`]). [`keys`] is not a gesture: it is what the
//! *window* hears, which is holds rather than acts.
//!
//! What is left here is the **vocabulary they are written in**, and it is here
//! because more than one of them needs it: how a DOM event becomes an
//! `InputSample`, what a pointer type means ([`is_contact`], [`is_eraser`]), the
//! tolerances a fit is given, the hover mark and the eyedropper's sample.
//!
//! The thresholds and the decisions under those gestures — the touch slop, the hold,
//! the pinch and the tap, the carry — are `stark_ui`'s (§11.2), so the native frontend
//! reads the same ones; what is left here is the DOM's side of each.

use dioxus::dioxus_core::{Task, spawn_forever};
use dioxus::html::geometry::ElementPoint;
use dioxus::html::input_data::MouseButton;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;
use stark_engine::command::Tool;

use crate::commands;
use crate::overlays::TowUi;
use crate::panels::select::current_action;
use crate::platform::{
    self, RawPointer, capture_pointer, now_seconds, on_window_blur, on_window_event, on_window_key,
    on_window_pointer, sleep_ms,
};
use crate::slots;
use crate::state::{AppState, dispatch, update_brush};
use stark_engine::ViewTransform;
use stark_engine::command::InputSample;
use stark_engine::command::{GestureCommand, PeerCommand, ViewCommand};
use stark_model::document::{LayerId, ShapeAction};
use stark_model::geom::Vec2;
use stark_ui::drags::Hand;
use stark_ui::input::PointerKind;
use stark_ui::pick::Sampler;
use stark_ui::slots::Grip;

mod carry;
mod gestures;
mod keys;
mod nav;
mod paint;
mod tune;

pub use carry::PickMove;
pub(crate) use gestures::{end_interaction, use_gestures};
pub use keys::{bind_context_menu, bind_pen, bind_shortcuts};
pub use nav::Nav;
pub use paint::{Landing, Paint};
pub use tune::{BrushRing, FlowBar, Tune, TuneReadout};

/// Which kind of pointer `e` came from, in the shared rules' vocabulary.
pub(crate) fn pointer_kind(e: &Event<PointerData>) -> PointerKind {
    match e.pointer_type().as_str() {
        "pen" => PointerKind::Pen,
        "touch" => PointerKind::Touch,
        _ => PointerKind::Mouse,
    }
}

/// Whether `e` came from a finger — the one pointer type that arrives in pairs.
///
/// A pen is deliberately not one, on the same screen and through the same API: it
/// reports a single contact, and the whole point of the two-finger gesture is to be
/// able to move the canvas *without* putting the pen down.
fn is_finger(e: &Event<PointerData>) -> bool {
    pointer_kind(e) == PointerKind::Touch
}

/// Whether `e` puts the tool **on** the canvas: the primary button, or the pen's
/// other end against the glass.
///
/// One definition, because "a press that draws" is asked in three places — the
/// canvas's own press, the space-drag pan, and the eraser hold — and a press that
/// counted as a contact in one of them and not another would either paint without
/// its brush or arm a brush without painting.
pub fn is_contact(e: &Event<PointerData>) -> bool {
    e.trigger_button() == Some(MouseButton::Primary) || is_eraser(e)
}

/// Whether `m` holds the **accelerator** — Ctrl or Command, on every OS, which is
/// `stark_ui::keys::accel`'s policy. What is this frontend's is that the DOM calls
/// Command `META`.
pub(crate) fn accel(m: Modifiers) -> bool {
    stark_ui::keys::accel(m.contains(Modifiers::CONTROL), m.contains(Modifiers::META))
}

/// The three modifiers as a DOM event reports them — the one translation from
/// [`Modifiers`], shared by pointer presses and keystrokes (`commands`) so a drag and
/// a chord cannot read the same modifiers differently.
pub(crate) fn mods_of(m: Modifiers) -> stark_ui::keys::Mods {
    stark_ui::keys::Mods {
        ctrl: accel(m),
        shift: m.contains(Modifiers::SHIFT),
        alt: m.contains(Modifiers::ALT),
    }
}

/// Whether `e` is the pen's **eraser end** — the tail of the stylus, reported as
/// a pen contact carrying the eraser button (§18.1.8).
///
/// Read off the raw event rather than through [`MouseButton`], which stops at the
/// fifth button and folds every code past it into `Unknown` — so a pen's eraser
/// (`button` 5, `buttons` bit 32, per Pointer Events) and a mouse's seventh
/// thumb button would arrive here as the same value. The web event says which,
/// and it is already in the tree; off-wasm the downcast simply finds nothing,
/// which is the right answer on a platform with no pens.
///
pub fn is_eraser(e: &Event<PointerData>) -> bool {
    platform::raw_pointer(e).is_some_and(|raw| is_eraser_event(&raw))
}

/// [`is_eraser`] against a raw web event — what the window-level binding sees
/// ([`bind_pen`]), where there is no dioxus event to unwrap.
///
/// Both button fields, because the two halves of a press report differently: the
/// press and the release name the button that *changed* (`button`), while every
/// move — hovering or in contact — names only what is still down (`buttons`,
/// with `button` at −1). A test on either alone would arm on the press and then,
/// one move later, disagree with itself.
fn is_eraser_event(raw: &RawPointer) -> bool {
    /// `button` for the eraser end, per Pointer Events.
    const ERASER_BUTTON: i16 = 5;
    /// The same, as its bit in `buttons`.
    const ERASER_BUTTONS: u16 = 32;

    raw.pen && (raw.button == ERASER_BUTTON || raw.buttons & ERASER_BUTTONS != 0)
}

/// Which edge of the pointer protocol a report arrived on — the three groups
/// [`tail_says`] answers differently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PenReport {
    /// A press, a move or an enter: the pen saying where it is.
    Present,
    /// A release or a cancel: the contact ending.
    Lifted,
    /// An out: the pointer leaving *something*, only sometimes the digitizer.
    Out,
}

/// Which end of the stylus faces the glass, so far as one report can say.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tail {
    /// The eraser end: hold its slot.
    Facing,
    /// The tip, or a pen that has gone: end the hold, if one is in flight.
    Away,
}

/// What one report says about the pen's tail — the whole of the eraser hold's
/// policy (§18.1.8), pure so it can be held to it without a browser.
///
/// The hold follows the tail **facing** the glass rather than touching it. A pen
/// that comes into range already inverted sends no press to say so, and the mark
/// under the cursor is a promise about the brush a press *would* use
/// (§18.1.10) — held to contact alone it previewed paint and then erased.
///
/// Read off every report rather than the edges alone, which is what makes a
/// missed edge cost a frame instead of the session: each one re-states which end
/// faces the glass, so a hold nothing armed, or nothing ended, is corrected by
/// the next report rather than stranding the swapped brush.
///
/// The three are deliberately not one test:
///
/// - **Present** — the eraser bit *is* the tail, and its absence the tip: a
///   press must really be the eraser or every ordinary stroke would erase, and a
///   hovering move without it is the pen flipped back.
/// - **Lifted** — leaving the glass is not leaving the range, so the eraser's
///   own release goes on holding; handing the brush back between two erase
///   strokes would flicker the cursor across the very preview this is for.
///   Anything else ends the hold, a driver that reports the release without the
///   eraser bit included.
/// - **Out** — a pen out of range and a pointer crossing between elements fire
///   the same event, and only the first entered nothing.
fn tail_says(report: PenReport, raw: &RawPointer) -> Option<Tail> {
    if !raw.pen {
        return None;
    }
    match report {
        PenReport::Present => Some(if is_eraser_event(raw) {
            Tail::Facing
        } else {
            Tail::Away
        }),
        PenReport::Lifted => (!is_eraser_event(raw)).then_some(Tail::Away),
        PenReport::Out => raw.entered_nothing.then_some(Tail::Away),
    }
}

/// Pointer position in page coordinates — the frame that stays still while
/// absolutely-positioned chrome (frame handles, the transform box) moves under
/// the pointer mid-drag.
pub fn page_xy(e: &Event<PointerData>) -> Vec2 {
    let p = e.page_coordinates();
    Vec2::new(p.x as f32, p.y as f32)
}

/// Where `e` lands in canvas space, through `view` ([`page_to_canvas`]).
pub fn canvas_xy(view: ViewTransform, e: &Event<PointerData>) -> Vec2 {
    page_to_canvas(view, page_xy(e))
}

/// Where page position `at` lands in canvas space, through `view` — the one mapping
/// every pointer position on the canvas takes, so a stroke's press and its coalesced
/// moves cannot disagree.
///
/// Page px go straight through the view because the `<canvas>` sits at the page
/// origin, so the view's screen space *is* the page — which is also what lets an
/// overlay's handler, whose own element is elsewhere, map through it. Move the canvas
/// off the origin and every caller of this is wrong by the offset; the native
/// frontend, whose canvas is not at its window's origin, subtracts it (`screen_at`).
fn page_to_canvas(view: ViewTransform, at: Vec2) -> Vec2 {
    view.screen_to_canvas(at)
}

/// Sample the canvas color under `pos` and load the brush with it — the eyedropper
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
    // The *choice* is what the bar holds; which layer it means is resolved by
    // `stark_ui::pick`, against whichever layer is selected at the moment of the
    // sample. Three signals here and one field natively, because how a frontend
    // *stores* the options is its own; what they mean is not (§11.2).
    let sampler = Sampler {
        scope: *state.pick.scope.peek(),
        group_only: *state.pick.group_only.peek(),
        radius: *state.pick.radius.peek(),
    };
    let active = state.obs.peek().as_ref().map(|o| o.active_layer);
    let options = sampler.options(active);

    // Render now and **drop the guard before awaiting** — the readback future owns
    // everything it needs, so nothing holds the renderer while the browser's event
    // loop runs the copy, which it must be free to do since the UI re-renders during
    // it (the same bargain `files::export_png` makes).
    let Some(readback) = crate::state::with_engine_quiet(state, |r| r.pick_color(pos, options))
    else {
        return;
    };
    busy.set(true);
    // Detached: the sample outlives the pointer gesture that asked for it (a release
    // must not cancel the answer to the press), and every signal it writes is
    // root-owned — see `state::root_signal`.
    spawn_forever(async move {
        let picked = readback.await;
        busy.set(false);
        // Nothing under the sampler leaves the brush as it was: bare canvas is the
        // substrate, not paint to pick up.
        let Some(rgb) = picked else { return };
        // A sample landed, which the tour counts as the **gesture** it is (§24.2).
        // Reported here rather than left to the command below for the same reason that
        // command is bracketed: what the stream will say is that a color changed, and
        // the two lessons this feeds are about the eyedropper rather than about the
        // color. Only on a sample that answered — a pick over bare canvas returns above
        // and is not a color anybody got.
        crate::tutor::did(state, crate::tutor::Deed::PickedColor);
        // The color about to be written comes off the painting, which is the gesture
        // one of the tour's lessons exists to teach — so the write is marked as the
        // eyedropper's rather than as somebody reaching for the picker (§24.2). The
        // bracket is drawn tight around the one write, with no `await` inside it, so
        // it cannot still be open while something else moves the brush.
        crate::tutor::not_reaching(state, true);
        update_brush(state, |_, t| t.color = [rgb[0], rgb[1], rgb[2]]);
        crate::tutor::not_reaching(state, false);
        // Tell the Color panel the color moved from outside its own picker, so its
        // markers follow (see `AppState::color_epoch`).
        let mut epoch = state.color_epoch;
        let next = *epoch.peek() + 1;
        epoch.set(next);
    });
}

/// Move the held pick's loupe to `at`, page px — the finger it belongs to
/// has dragged on to sample somewhere else (§18.1.11).
///
/// **Silent when no loupe is up**, which is what keeps it to the one gesture that
/// needs it: a sampler dragged with a mouse or a pen has a cursor on the point and a
/// clear view of the Color panel, and a swatch following it would be a third thing
/// saying what two already say. A finger has neither — it is *on* the place it is
/// asking about — so the answer is drawn where it can be seen past the hand.
pub fn move_loupe(state: AppState, at: Vec2) {
    let mut loupe = state.pick.loupe;
    if loupe.peek().is_some() {
        loupe.set(Some(at));
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
///
/// Off the projection rather than the renderer: every door that moves the view
/// publishes, and a peek there neither subscribes nor borrows the engine.
fn view_of(state: AppState) -> Option<ViewTransform> {
    state.obs.peek().as_ref().map(|o| o.view)
}

/// Pointer position within an element, in CSS pixels.
///
/// Not for the canvas's handlers, which read page px ([`page_to_canvas`]): the
/// element offset can force a layout, and mixing frames is how a stroke's head jumps.
pub fn elem_xy(e: &Event<PointerData>) -> Vec2 {
    let ElementPoint { x, y, .. } = e.element_coordinates();
    Vec2::new(x as f32, y as f32)
}

/// How finely a pointer report resolves position, in **CSS px**.
///
/// The device's half of `stark_ui::input::tolerance` — what this browser knows
/// and the shared map cannot: which kind of pointer this is, and how many physical
/// pixels a CSS one is. A mouse walks the screen in whole physical pixels, so
/// `1 / devicePixelRatio` CSS px is its floor; a pen or a finger comes off a
/// digitizer that resolves well below the screen it sits under.
///
/// The ratio is read per call rather than held: the read is a property getter, and a
/// held one goes stale on a move to a monitor of another scale, which fires no resize.
fn input_resolution(e: &Event<PointerData>) -> f32 {
    stark_ui::input::resolution(pointer_kind(e)) / platform::device_pixel_ratio()
}

/// The fitting tolerance to declare for a gesture starting with `e`, in canvas px.
pub fn input_tolerance_in(view: ViewTransform, e: &Event<PointerData>) -> f32 {
    stark_ui::input::tolerance(view, input_resolution(e))
}

/// [`input_tolerance_in`] against the main canvas's view; `None` before the engine
/// exists.
pub fn input_tolerance(state: AppState, e: &Event<PointerData>) -> Option<f32> {
    Some(input_tolerance_in(view_of(state)?, e))
}

/// The §6.11 rope for the live brush against the main canvas's view, in canvas px.
/// Zero (no tow at all) when the amount is zero or there is no view yet.
///
/// The map itself is `stark_ui::input::rope`, shared with the native frontend;
/// what is here is only *which* brush and *which* view.
pub fn input_rope(state: AppState) -> f32 {
    match view_of(state) {
        Some(view) => stark_ui::input::rope(view, state.brush.peek().smoothing),
        None => 0.0,
    }
}

/// Refresh the on-screen tow string from the engine (§6.11), converting to the
/// canvas element's own px against the view the stroke holds — a pinch cancels
/// the stroke it interrupts, so a live string never straddles two views. Sets
/// `None` when there is nothing to show, and leaves the signal untouched when
/// nothing changed, so an idle call dirties no scope.
pub fn refresh_tow(state: AppState) {
    let mut tow = state.tow;
    let ui = state.renderer.peek().as_ref().and_then(|r| {
        let t = r.tow_string()?;
        let view = r.view();
        Some(TowUi {
            tip: view.canvas_to_screen(t.tip),
            target: view.canvas_to_screen(t.target),
            rope: t.rope * view.zoom,
        })
    });
    if ui != *tow.peek() {
        tow.set(ui);
    }
}

/// Report the pointer hovering over the canvas at `at`, page px — what
/// the brush cursor is drawn on (§18.1.10, `BrushCursor`). Unconditional, unlike
/// [`hover_gone`]: a move's position is news by definition, and only the overlay
/// subscribes.
pub fn hover_at(state: AppState, at: Vec2) {
    let mut hover = state.brush_cursor;
    hover.set(Some(at));
}

/// Report where the pointer is on the canvas, or that it has left it — the
/// cursor collaborators watch (§17.4) *and* the one this client's own guides
/// draw their rays through (§20.9).
///
/// One fact with two readers, so one call — but the two want different things
/// from this door, and the difference is a repaint:
///
/// - **A peer** reads it off the engine on the presence pump's own cadence
///   (§17.5), so sending is all this owes: repainting our own canvas to show
///   ourselves a cursor the browser is already drawing would be pure waste, and
///   at pointer rate it would be a lot of it.
/// - **A guide** draws chrome in the frame, and chrome only appears when a frame
///   is painted. So a client with a guide open pays a repaint per pointer move,
///   which is what following the hand costs and is the same bargain the hover
///   mark already strikes ([`hover_stroke`]).
///
/// Hence the gate: sent when somebody is reading it, and painted only for the
/// reader that is on this side of the glass. With no session and no guide up the
/// value has no reader at all, and the engine borrow is skipped entirely.
pub fn point_at(state: AppState, at: Option<Vec2>) {
    // `peek`, not `read`: this runs per pointer move and must widen nothing that
    // a render subscribes to (`state::gpu_lost` peeks for the same reason).
    let drawn = state
        .obs
        .peek()
        .as_ref()
        .is_some_and(|o| o.guides.iter().any(|g| g.visible));
    if !drawn && !state.collab.active() {
        return;
    }
    let command = PeerCommand::SetCursor(at);
    if drawn {
        crate::state::dispatch_hover(state, command);
    } else {
        crate::state::dispatch_quiet(state, command);
    }
}

/// What this app knows about the hand that the drag table does not ([`Hand`]) — one
/// reading, for the press path (`drags::find`), the canvas cursor, the eyedropper's
/// bar and the hover mark, so what a cursor promises and what the press does cannot be
/// read two ways.
///
/// Reads rather than peeks: a render body that asks subscribes to what it shows, and
/// in an event handler a read subscribes nothing. `tool` is an argument because the
/// canvas takes it from its own memo, where a read of the projection would re-render
/// it on every engine write.
pub fn hand(state: AppState, tool: Tool) -> Hand {
    Hand {
        panning: (state.space_down)(),
        selecting: tool.is_selection(),
        playing: crate::panels::timeline::is_playing(state),
        sampling: (state.pick.dragging)(),
        busy: (state.canvas_active)(),
    }
}

/// Feed the hover mark one report (§18.1.10): the engine appends `s` to its
/// trailing window and folds the probe — the stroke a drag begun this instant
/// would open, carrying the hover's heading forward from the cursor — the
/// painted half of the brush cursor, under the circle [`hover_at`] places.
///
/// What the report *is* — the reach, and the full pressure a hovering hand does
/// not report — is `stark_ui::input::Hovering`, shared with the native frontend
/// (§11.2), and so is the list of states that promise the press to something
/// other than paint. What is here is only this frontend's reading of each — the
/// signals it keeps them in, and the drag table asked of the modifiers held.
pub fn hover_stroke(state: AppState, s: InputSample, e: &Event<PointerData>) {
    let shadowed = stark_ui::drags::armed(&state.drags.peek(), *state.held_mods.peek())
        .is_some_and(stark_ui::drags::DragAction::shadows_paint);
    let hovering = hand(state, crate::panels::select::current_tool(state)).hovering(shadowed);
    let Some(tolerance) = input_tolerance(state, e) else {
        return;
    };
    let Some(report) = hovering.report(s, tolerance) else {
        return;
    };
    crate::state::dispatch_hover(state, ViewCommand::PreviewHover(Some(report)));
}

/// Take the engine's hover mark down, if one is up (§18.1.10) — the half of
/// [`hover_gone`] that owns pixels rather than a `<div>`: the circle hides
/// reactively, but the mark is paint in the frame, and only a command (and the
/// repaint it asks for) removes it. The peek is the other half of the bargain —
/// an idle call must not spend a command or schedule a frame, and this runs per
/// move of a pan and on auto-repeating keydowns.
pub fn clear_hover_mark(state: AppState) {
    let held = state
        .renderer
        .peek()
        .as_ref()
        .is_some_and(crate::render::Renderer::hover_held);
    if held {
        crate::state::dispatch_hover(state, ViewCommand::PreviewHover(None));
    }
}

/// The hover is over — the pointer left the canvas, or the gesture in hand
/// stopped being paint (a pinch, a pan, a tuning drag). Written only on a
/// change, since a pan calls this per move and an idle call must dirty no scope
/// — [`clear_hover_mark`] guards itself the same way.
pub fn hover_gone(state: AppState) {
    clear_hover_mark(state);
    let mut hover = state.brush_cursor;
    if hover.peek().is_some() {
        hover.set(None);
    }
}

/// Map a pointer event to a canvas-space input sample; `None` before the engine
/// exists, since there is no view to map through yet.
pub fn sample(state: AppState, e: &Event<PointerData>) -> Option<InputSample> {
    let view = view_of(state)?;
    Some(InputSample {
        pos: canvas_xy(view, e),
        pressure: e.pressure(),
        // Pen tilt (degrees from vertical, ±90 per axis) → a canvas-space lean vector. The
        // palette knife's deposit reads its component along the stroke direction
        // (§6.2); a mouse reports (0, 0), so the deposit falls back to its constant rate.
        tilt: Vec2::new(e.tilt_x() as f32, e.tilt_y() as f32) / 90.0,
        time: platform::event_time(e),
    })
}

/// Every canvas-space sample a `pointermove` carries, oldest first.
///
/// The browser delivers roughly one `pointermove` per animation frame and folds
/// the reports it withheld — most of what a 120–240 Hz pen produces — into the
/// delivered event's *coalesced* list. Reading that list is what gets the full
/// input rate to the fitter; reading only the event caps every stroke at display
/// rate, whatever the device resolved. Each entry carries its own position,
/// pressure, tilt and timestamp, so the samples land as the hand made them
/// rather than as delivery batched them.
///
/// The reports come back from [`platform::coalesced`] in page px, mapped as
/// [`sample`] maps the event ([`page_to_canvas`]). Falls back to the event itself
/// when there is no list (off-wasm, a synthetic event), so the list is never empty;
/// `None` before the engine exists, like [`sample`].
///
/// A list read per call, so the canvas asks this once per move and only for a move
/// a stroke will take.
pub fn samples(state: AppState, e: &Event<PointerData>) -> Option<Vec<InputSample>> {
    let view = view_of(state)?;
    let folded = platform::coalesced(e).map(|list| {
        list.into_iter()
            .map(|c| InputSample {
                pos: page_to_canvas(view, Vec2::new(c.x, c.y)),
                pressure: c.pressure,
                tilt: Vec2::new(c.tilt_x, c.tilt_y) / 90.0,
                time: c.time,
            })
            .collect::<Vec<_>>()
    });
    match folded {
        Some(list) if !list.is_empty() => Some(list),
        _ => sample(state, e).map(|s| vec![s]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pen(button: i16, buttons: u16) -> RawPointer {
        RawPointer {
            pen: true,
            button,
            buttons,
            entered_nothing: true,
        }
    }

    #[test]
    fn the_tail_holds_its_slot_from_hover_rather_than_from_contact() {
        // A pen in range has pressed nothing: the changed button is −1 and
        // nothing is down. Inverted, it carries the eraser bit all the same, and
        // that bit is the only thing a hover can be told apart by (§18.1.8).
        assert_eq!(
            tail_says(PenReport::Present, &pen(-1, 32)),
            Some(Tail::Facing)
        );
        assert_eq!(tail_says(PenReport::Present, &pen(-1, 0)), Some(Tail::Away));
        // And the press still has to really be the eraser, or every ordinary
        // stroke would erase.
        assert_eq!(tail_says(PenReport::Present, &pen(0, 1)), Some(Tail::Away));
        assert_eq!(
            tail_says(PenReport::Present, &pen(5, 32)),
            Some(Tail::Facing)
        );
    }

    #[test]
    fn the_tail_leaving_the_glass_is_still_facing_it() {
        // The lift that ends an erase stroke names button 5 with nothing left
        // down — and the tail is a millimetre above the same glass, so the hold
        // stands rather than flickering the brush back for the gap between two
        // strokes.
        assert_eq!(tail_says(PenReport::Lifted, &pen(5, 0)), None);
        // Any other pen leaving ends it, which is what covers a driver that
        // reports the release without the bit, and a cancel.
        assert_eq!(tail_says(PenReport::Lifted, &pen(0, 0)), Some(Tail::Away));
        assert_eq!(tail_says(PenReport::Lifted, &pen(-1, 0)), Some(Tail::Away));
    }

    #[test]
    fn only_an_out_that_entered_nothing_is_the_pen_gone() {
        assert_eq!(tail_says(PenReport::Out, &pen(-1, 32)), Some(Tail::Away));
        let crossing = RawPointer {
            entered_nothing: false,
            ..pen(-1, 32)
        };
        assert_eq!(
            tail_says(PenReport::Out, &crossing),
            None,
            "crossing between elements fires the same event as leaving the range"
        );
    }

    #[test]
    fn nothing_but_a_pen_says_anything_about_the_pen() {
        // A mouse's press carries button 0 and a finger's leave enters nothing;
        // neither is evidence about which end of a stylus faces the glass.
        let other = RawPointer {
            pen: false,
            ..pen(0, 1)
        };
        for report in [PenReport::Present, PenReport::Lifted, PenReport::Out] {
            assert_eq!(tail_says(report, &other), None);
        }
    }
}
