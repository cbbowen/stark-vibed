//! Pointer and keyboard input: turning DOM events into
//! [`InputCommand`](stark_engine::InputCommand)s
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
//! [`keys`] is the sixth file and is not a gesture: it is what the *window*
//! hears, which is holds rather than acts.
//!
//! What is left here is the **vocabulary they are written in**, and it is here
//! because more than one of them needs it: how a DOM event becomes an
//! `InputSample`, what a pointer type means ([`is_contact`], [`is_eraser`]), the
//! tolerances a fit is given, the hover mark, the eyedropper's sample, and the
//! release that ends whatever was in flight ([`end_interaction`], which is the
//! one place all five are named together).
//!
//! [`TOUCH_SLOP`] is the clearest case and says so at its own declaration: two
//! gestures read it, and the test that they agree cannot be written if there are
//! two of it.

use dioxus::dioxus_core::{Task, spawn_forever};
use dioxus::html::geometry::ElementPoint;
use dioxus::html::input_data::MouseButton;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;
use stark_engine::command::Tool;

use crate::collab::now_seconds;
use crate::commands;
use crate::drags;
use crate::panels::brush::{MAX_FLOW, MAX_RADIUS, MIN_RADIUS};
use crate::panels::select::{current_action, modifier_mode};
use crate::platform::{
    self, RawPointer, capture_pointer, on_window_blur, on_window_event, on_window_key,
    on_window_pointer, sleep_ms,
};
use crate::slots::{self, Grip};
use crate::state::{AppState, BrushRing, Dwell, PickScope, TowUi, dispatch, update_brush};
use stark_engine::InputSample;
use stark_engine::ViewTransform;
use stark_engine::command::{GestureCommand, HoverReport, PeerCommand, ViewCommand};
use stark_engine::{PickOptions, PickSource};
use stark_model::document::{LayerId, ShapeAction, TransformMap};
use stark_model::geom::{Affine2, Vec2};

mod carry;
mod keys;
mod nav;
mod paint;
mod tune;

pub use carry::PickMove;
pub use keys::{bind_context_menu, bind_pen, bind_shortcuts};
pub use nav::Nav;
pub use paint::{Landing, Paint};
pub use tune::Tune;

/// How far a touch may travel and still mean **nothing yet**, in page px
/// (§18.1.11).
///
/// One number for three questions, which is the point of there being one: how far
/// a lone finger must go before its press is a stroke rather than a held question
/// ([`Landing`]), how far it may stray and still count as *held* for the
/// eyedropper, and how far a pair may stray and still count as a tap rather than a
/// pinch. Those are one question asked from three sides — *has this touch meant
/// anything yet?* — so a single constant is what keeps the three answers from
/// overlapping. It also makes the two-finger tap safe to spend on undo by
/// construction rather than by care: a pair that stayed inside this never opened a
/// stroke, because the same threshold is what would have opened one.
///
/// [`carry_deadzone`]'s figure for a finger, for its reason: a fingertip rolls
/// several px on its way onto and off the glass, and of the two mistakes it is the
/// one that leaves a mark that has to be protected against.
pub(super) const TOUCH_SLOP: f32 = 10.0;
//
// **Here rather than with either gesture that reads it**, which is the whole
// point of the constant: `Nav` measures a tap against it and `Landing` opens a
// stroke on it, and `nav::tests::a_tap_can_never_have_painted` is the assertion
// that those are the same number. Two copies could not fail that test — they
// would simply both be right about different thresholds — so there is one, and
// it lives where neither gesture owns it.

/// How close to a quarter turn a turn has to land to be pulled onto it, radians
/// (about 5°).
///
/// The frontend's to decide, like the fitting tolerance: it is a property of turning
/// something with a hand rather than of the view. Without it a canvas that has been
/// turned could only be *approximately* straightened, and a piece left a degree off
/// square reads as an accident rather than as a choice.
pub const TURN_SNAP: f32 = 0.09;

/// Whether `e` came from a finger — the one pointer type that arrives in pairs.
///
/// A pen is deliberately not one, on the same screen and through the same API: it
/// reports a single contact, and the whole point of the two-finger gesture is to be
/// able to move the canvas *without* putting the pen down.
fn is_finger(e: &Event<PointerData>) -> bool {
    e.pointer_type() == "touch"
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

/// Whether `m` holds the platform's **accelerator** — Ctrl, or Command on a Mac.
///
/// Either, everywhere, rather than asking which platform this is: the keyboard
/// shortcuts have always accepted both, and a binding that insisted on Ctrl would be
/// unreachable on the one platform where Ctrl+drag is how the browser reports a
/// secondary click in the first place.
pub(crate) fn accel(m: Modifiers) -> bool {
    m.contains(Modifiers::CONTROL) || m.contains(Modifiers::META)
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
/// move between them names only what is still down (`buttons`, with `button` at
/// −1). A test on either alone would arm on the press and then, one move later,
/// disagree with itself.
fn is_eraser_event(raw: &RawPointer) -> bool {
    /// `button` for the eraser end, per Pointer Events.
    const ERASER_BUTTON: i16 = 5;
    /// The same, as its bit in `buttons`.
    const ERASER_BUTTONS: u16 = 32;

    raw.pen && (raw.button == ERASER_BUTTON || raw.buttons & ERASER_BUTTONS != 0)
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
    // The *choice* is what the bar holds; which layer it means is resolved here,
    // against whichever layer is selected at the moment of the sample — and a
    // document with no layer selected falls back to the whole document rather
    // than sampling nothing. The canvas color stands behind the sample exactly
    // when the group fence is down (`PickState::group_only`): a group is paint,
    // and the whole document is a picture on a canvas.
    let scope = *state.pick.scope.peek();
    let group_only = *state.pick.group_only.peek();
    let active = state.obs.peek().as_ref().map(|o| o.active_layer);
    let options = PickOptions {
        source: match (scope, active) {
            (PickScope::ThisLayer, Some(id)) => PickSource::Layer(id),
            (PickScope::AndBelow, Some(id)) if group_only => PickSource::Group {
                layer: id,
                below: true,
            },
            (PickScope::AndBelow, Some(id)) => PickSource::Below(id),
            (PickScope::AllLayers, Some(id)) if group_only => PickSource::Group {
                layer: id,
                below: false,
            },
            _ if group_only => PickSource::Composite,
            _ => PickSource::CompositeOverSubstrate,
        },
        radius: *state.pick.radius.peek(),
    };

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
        update_brush(state, |br| br.color = [rgb[0], rgb[1], rgb[2], br.color[3]]);
        crate::tutor::not_reaching(state, false);
        // Tell the Color panel the color moved from outside its own picker, so its
        // markers follow (see `AppState::color_epoch`).
        let mut epoch = state.color_epoch;
        let next = *epoch.peek() + 1;
        epoch.set(next);
    });
}

/// Move the held pick's loupe to `at`, element (CSS) px — the finger it belongs to
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

// ---------------------------------------------------------------------------
// The hold, for the drawing assist (§6.9)
// ---------------------------------------------------------------------------

/// How long a pointer has to hold still to have **held** — before the stroke in
/// flight snaps to the shape it resembles (§6.9), and before a finger's press that
/// never became a stroke becomes the eyedropper instead ([`Landing`], §18.1.11).
///
/// One figure for both, deliberately, and not because the two acts are related: a
/// hold is a hold, and a wait that meant one length of time before a stroke and
/// another during one would be a hand having to learn the app twice. What the two
/// share is a threshold, not a mechanism — the assist watches for a pointer that
/// *stopped*, this one for a press that never *started*.
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
/// Not a preference — an estimate of the device's tolerance, which is what the fitter
/// needs in order to tell jitter from detail (see
/// [`PathFitter::with_tolerance`](stark_engine::path::PathFitter::with_tolerance)). A
/// mouse walks the screen in whole *physical* pixels, so `1 / devicePixelRatio` CSS
/// px is its floor. A pen or a finger comes off a digitizer that resolves well below
/// the screen it sits under, so what limits those is the hand rather than the API;
/// half a physical pixel is a deliberate under-estimate — too fine only costs a few
/// extra control points, while too coarse rounds off detail that was really there.
fn input_resolution(e: &Event<PointerData>) -> f32 {
    let dpr = platform::device_pixel_ratio();
    let physical = match e.pointer_type().as_str() {
        "pen" | "touch" => 0.5,
        _ => 1.0,
    };
    physical / dpr
}

/// The fitting tolerance to declare for a gesture starting with `e`, in canvas px:
/// the device's own tolerance (above) carried through `view`, since canvas space is
/// where the fit measures error.
pub fn input_tolerance_in(view: ViewTransform, e: &Event<PointerData>) -> f32 {
    input_resolution(e) / view.zoom
}

/// [`input_tolerance_in`] against the main canvas's view; `None` before the engine
/// exists.
pub fn input_tolerance(state: AppState, e: &Event<PointerData>) -> Option<f32> {
    Some(input_tolerance_in(view_of(state)?, e))
}

/// The longest smoothing string a brush can ask for, in **screen px** — what
/// `smoothing = 1` means. Screen px because wobble is a fact about the hand:
/// the same tremor spans 64× more canvas zoomed out than in.
const ROPE_MAX_SCREEN_PX: f32 = 160.0;

/// The §6.11 rope a smoothing amount means against `view`, in canvas px: the
/// 0..=1 knob mapped **quadratically** to a screen-px string — so the low end
/// is fine-grained while the top is a real lettering tow — then carried
/// through the view like the tolerance above. Zooming in therefore shrinks the
/// dead zone in canvas terms: the escape hatch from heavy smoothing is the one
/// artists already reach for to do fine work.
///
/// Stated once against an explicit view because two canvases ask: the main
/// canvas below, and the brush editor's preview against its own.
pub fn rope_in(view: ViewTransform, amount: f32) -> f32 {
    let a = amount.clamp(0.0, 1.0);
    a * a * ROPE_MAX_SCREEN_PX / view.zoom
}

/// [`rope_in`] for the live brush against the main canvas's view. Zero (no tow
/// at all) when the amount is zero or there is no view yet.
pub fn input_rope(state: AppState) -> f32 {
    match view_of(state) {
        Some(view) => rope_in(view, *state.smoothing.peek()),
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

/// Report the pointer hovering over the canvas at `at`, element (CSS) px — what
/// the brush cursor is drawn on (§18.1.10, `BrushCursor`). Unconditional, unlike
/// [`hover_gone`]: a move's position is news by definition, and only the overlay
/// subscribes.
pub fn hover_at(state: AppState, at: Vec2) {
    let mut hover = state.brush_cursor;
    hover.set(Some(at));
}

/// How far ahead of the cursor the hover mark reaches, in **canvas px**
/// (§18.1.10).
///
/// Canvas rather than screen px by nature, not oversight: the mark is a
/// hypothesis about *paint*, and paint is denominated on the canvas — fixed on
/// the screen, the predicted stroke grew in canvas terms as the view zoomed
/// out, promising more painting the less closely you looked. The size circle
/// over it already scales with the zoom, so the two halves of the cursor now
/// shrink and grow together. The *smoothing* does not ride this number: the
/// heading's estimator window is tolerance-relative inside the engine, so its
/// steadiness survives every zoom.
const HOVER_REACH_CANVAS_PX: f32 = 8.0;

/// Feed the hover mark one report (§18.1.10): the engine appends `s` to its
/// trailing window and folds the probe — the stroke a drag begun this instant
/// would open, carrying the hover's heading forward from the cursor — the
/// painted half of the brush cursor, under the circle [`hover_at`] places.
///
/// Gated on the states that promise the press to something other than paint —
/// space's pan, a chord the drag table answers with an act that shadows the
/// brush (`DragAction::shadows_paint`: the eyedropper, the layer carry), the
/// eyedropper already dragging, and playback, where a stroke would be refused
/// (`panels::timeline::is_playing`). The engine gates the rest itself: a
/// selection tool folds no mark, an unpaintable layer refuses the render, and a
/// real gesture always outranks the hypothesis.
///
/// The report's pressure is replaced with **full pressure**: a hovering pen
/// (and a mouse) reports zero, which would honestly preview no mark at all.
/// Full rather than a middle weight so the mark fills the size circle drawn
/// over it — two overlays about one brush must not disagree about its reach.
/// Tilt is kept: a hovering pen reports it, and the mark should lean as the
/// stroke would.
pub fn hover_stroke(state: AppState, s: InputSample, e: &Event<PointerData>) {
    if *state.space_down.peek()
        || drags::armed(&state.drags.peek(), *state.held_mods.peek())
            .is_some_and(drags::DragAction::shadows_paint)
        || *state.pick.dragging.peek()
        || crate::panels::timeline::is_playing(state)
    {
        return;
    }
    let Some(tolerance) = input_tolerance(state, e) else {
        return;
    };
    let report = HoverReport {
        sample: InputSample { pressure: 1.0, ..s },
        tolerance,
        reach: HOVER_REACH_CANVAS_PX,
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
/// The reports come back from [`platform::coalesced`] in the element's own px;
/// mapping them through the view is this side's business, since canvas space is
/// what a sample is in. Falls back to the event itself when there is no list
/// (off-wasm, a synthetic event); `None` before the engine exists, like
/// [`sample`].
pub fn samples(state: AppState, e: &Event<PointerData>) -> Option<Vec<InputSample>> {
    let view = view_of(state)?;
    let folded = platform::coalesced(e).map(|list| {
        list.into_iter()
            .map(|c| InputSample {
                pos: view.screen_to_canvas(Vec2::new(c.x, c.y)),
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

/// End every gesture the canvas can have in hand at once — the paint gesture, the
/// navigation, the brush-tuning drag, the layer carry and the eyedropper — and hand
/// the canvas back, so the floating chrome fades in.
///
/// `Copy` values and no `&mut`: each of them owns its own state now, so this
/// is the *order* they are put down in and nothing else. It used to take the paint
/// gesture apart into two signals it borrowed from the component, which is what
/// made "what counts as in flight" a thing two functions had to agree about
/// (`Paint`).
pub fn end_interaction(state: AppState, landing: Landing, nav: Nav, tune: Tune, carry: PickMove) {
    landing.end();
    nav.stop();
    tune.stop();
    // The one gesture here whose ending is not finished by the release: the
    // layer it picked up may still be a readback away, and the commit waits for
    // it (`PickMove::settle`).
    carry.stop();
    // Not a parameter like the three above because the eyedropper's drag flag is
    // shared state, not a gesture object — the options bar reads it (see
    // `PickState`). Nothing to undo, either: a sample already in flight is left to
    // land, since it is the answer to a press the user made.
    let mut dragging = state.pick.dragging;
    dragging.set(false);
    // And the swatch a held pick was showing goes with the finger that asked for it
    // (§18.1.11). Guarded like every other idle write here: this runs on every
    // release the canvas sees, and almost none of them had a loupe up.
    let mut loupe = state.pick.loupe;
    if loupe.peek().is_some() {
        loupe.set(None);
    }
    // The panel stack does not come straight back: it stays out of the way until the
    // pointer reaches into its column (`AppState::panels_asleep`, §11). The chrome
    // going *out* mid-stroke was never the distracting half — coming back the instant
    // the pen lifts is, because it happens at exactly the moment the artist is looking
    // at what they just drew.
    //
    // Gated on the fade having actually been in force. `end_interaction` runs on every
    // release the canvas sees, including the ones that deliberately keep the chrome up
    // — an eyedropper sample reads its answer off the Color panel, brush tuning off the
    // Brush panel (see the two `canvas_active` comments in `main.rs`) — and putting the
    // stack to sleep on the way out of those would hide the panel the gesture was for.
    // Read before the clear, since the clear is what makes it false.
    //
    // Whether it sleeps at all is this browser's own choice, and that question is
    // asked inside `sleep_panels` rather than here — one door, so the setting reaches
    // every caller (`layout::ChromeHiding`, §11).
    let was_faded = *state.canvas_active.peek();
    let mut canvas_active = state.canvas_active;
    canvas_active.set(false);
    if was_faded {
        crate::layout::sleep_panels(state);
    }
    // And a drag-preset offer brought due by a press this release is the end of
    // (§25.8). Here rather than at the press for `tutor`'s reason: the press
    // that finds nothing bound goes on to paint, and a modal over a live stroke
    // would take the canvas away mid-mark. Last, because it is the one thing in
    // this function that puts something *up*.
    crate::drags::settle_offer(state);
}
