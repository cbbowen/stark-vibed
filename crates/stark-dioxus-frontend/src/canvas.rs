//! The painting surface itself: the `<canvas>` the engine draws into, and the
//! press-move-release ladder that decides what a pointer on it means (§11,
//! §25.4).
//!
//! One component, and it is long because the ladder is: a press over this
//! element may open a pan, a brush-tuning drag, an eyedropper sample, a layer
//! carry or a stroke, and *which* is a question about the chord, the tool, the
//! pointer type and what is already in flight. The order those are asked in is
//! the whole of §25.4, and it is written here rather than distributed among the
//! gestures because an ordering that lives in one place is an ordering that can
//! be read.
//!
//! What is **not** here is any gesture's own state: each of `input`'s objects
//! owns that (§25.3), and this component holds only the answer to "which of them
//! is this press".

use dioxus::prelude::*;

use crate::commands::Command;
use crate::drags::{self, DragAction};
use crate::input::{
    self, Landing, Nav, Paint, PickMove, Tune, elem_xy, end_interaction, hover_at, hover_gone,
    hover_stroke, move_loupe, pick_color, point_at, sample,
};
use crate::modes;
use crate::panels;
use crate::panels::select::current_tool;
use crate::platform::capture_pointer;
use crate::render::CANVAS_ID;
use crate::state::{AppState, resize, use_obs};

/// The full-window painting surface (a WebGPU canvas the engine draws into).
#[component]
pub fn Canvas() -> Element {
    let state = use_context::<AppState>();
    // The paint gesture itself — the stroke or marquee, from the press that opens
    // one to the release that commits it (`input::Paint`). It owns whether one is
    // in flight and the shape action a modifier overrode, so this component is
    // left with the part that is genuinely its own: deciding, on a press, which of
    // the four bindings it is.
    let paint = Paint::use_paint(state);
    // And, in front of it, the wait a *finger's* press is held in until it says
    // which of the three touch gestures it is (`input::Landing`, §18.1.11). Every
    // press below goes through this instead of straight to `paint`; a pen's or a
    // mouse's is handed on unchanged, which is why nothing about drawing with a
    // stylus changes.
    let landing = Landing::use_landing(state, paint);
    // The shared pan/zoom bindings (`input::Nav`) — the same instance the
    // transform overlay makes for itself, so navigation means one thing.
    let nav = Nav::use_nav(state);
    // Accelerator+drag tunes Size and Flow instead of painting (`input::Tune`,
    // §18.1.9). The canvas's own, unlike `nav`: it moves the brush, and the
    // overlays that navigate have no brush.
    let tune = Tune::use_tune(state);
    // Shift+drag picks the layer under the press and carries it (`input::PickMove`,
    // §16.11). The canvas's own like `tune`, and for the mirror of its reason: it
    // moves the *painting*, which only the surface the painting is on can be
    // pointing at.
    let carry = PickMove::use_pick_move(state);
    // Whether an Alt+drag is sampling color off the canvas rather than painting on
    // it (§18.0.2). Shared rather than local, unlike the two above,
    // because the options bar is mounted on *armed but not dragging*.
    let mut picking = state.pick.dragging;
    // Set for as long as the canvas is the thing being used, which fades the floating
    // chrome out of the way. Pointer gestures clear it on release (`end_interaction`).
    let mut canvas_active = state.canvas_active;

    // Everything this component reads off the projection, in **one** memo — so the
    // canvas is re-rendered when its cursor would change and not when the engine is
    // merely touched (`state::use_obs`). It is the component that can least afford
    // the difference: it is the surface a stroke is being made on, and every sample
    // of that stroke writes the engine.
    //
    // The two facts are:
    //
    // - **Whether the selected layer takes paint.** A frame does not (§15.7).
    //   Rather than block the gesture, say so in the cursor: the brush crosshair
    //   becomes "not-allowed", so the canvas explains itself before the user draws a
    //   stroke that would go nowhere. Panning still works, so the pan cursor wins
    //   while space is held.
    // - **The tool**, for the eyedropper cursor below. It has to be *read* here
    //   rather than peeked as the handlers do (`current_tool`): a peek would leave
    //   the canvas wearing the wrong cursor until some other change happened to
    //   re-render it, which is precisely what subscribing to the whole projection
    //   was accidentally covering up.
    let look = use_obs(state, |o| {
        let paintable = o
            .layers
            .iter()
            .any(|l| l.id == o.active_layer && l.is_paintable());
        (paintable, o.tool)
    });
    let (paintable, tool) = look().unwrap_or((false, stark_engine::command::Tool::Brush));
    // The pick chord (Alt by default) arms the eyedropper over the brush, and the
    // cursor says so before it is used — the only thing that makes a modifier
    // binding discoverable. Asked of the drag table (`drags::armed`), the same
    // table the press will ask, so the promise moves with the binding. Not over a
    // selection tool, where alt already means "subtract from the selection"
    // (§6.8), so the cursor promises the pick exactly where a press would
    // take one. It beats `no-paint`, because a layer that takes no paint can still
    // be sampled.
    //
    // The layer carry announces itself the same way and owes it for the same
    // reason (§16.11): Shift+drag is a secret without a cursor that says so
    // before it is used. It stands down over a selection tool for the reason the
    // pick does — Shift is the union marquee there (§6.8) — which is the gate the
    // action itself declares, restated here because `armed` answers the table
    // about a chord and this is a question about the tool in hand.
    let armed = drags::armed(&state.drags.read(), (state.held_mods)());
    let over_paint = !(state.space_down)() && !tool.is_selection();
    let sampling = armed == Some(DragAction::PickColor) && over_paint;
    let carrying = armed == Some(DragAction::PickAndTranslate) && over_paint;
    // Whether a tuning drag is in flight (§18.1.9) — the crosshair goes while it is,
    // because the crosshair is a promise of paint *at a point* and this gesture is
    // about a number: nothing will land where it is pointing, and a crosshair sitting
    // in the middle of the size ring reads as the brush being there when the ring is
    // the only thing on screen saying anything true.
    //
    // Through a memo rather than off the signal, and that is the whole of why the
    // readout is worth asking: the drag rewrites it per pointer report, and this is the
    // surface a stroke is made on. A bare read would re-render the canvas per move to
    // find the answer unchanged; the memo wakes it twice a gesture.
    let tuning = use_memo(move || (state.tune_readout)().is_some());
    let canvas_class = if tuning() {
        // First in the ladder, over every cursor a held chord asks for: the pointer is
        // captured, so no other binding can be what this press is about.
        "paint-canvas tuning"
    } else if sampling {
        "paint-canvas picking"
    } else if carrying {
        // Above `no-paint` for the pick's reason: a layer that takes no paint is
        // no obstacle to picking a *different* layer up and moving it.
        "paint-canvas carrying"
    } else if paintable || (state.space_down)() {
        "paint-canvas"
    } else {
        "paint-canvas no-paint"
    };

    rsx! {
        canvas {
            id: "{CANVAS_ID}",
            class: canvas_class,
            onresize: move |e| {
                if let Ok(size) = e.get_content_box_size() {
                    resize(state, size.width as u32, size.height as u32);
                }
            },
            // Strokes and pans capture the pointer (like the pads/pickers): leaving the
            // window mid-stroke keeps painting — the infinite canvas extends past the
            // viewport anyway — and the interaction ends on release/cancel, never by
            // crossing the canvas edge.
            onpointerdown: move |e| {
                // Navigation first: a second finger on the glass, middle-drag, or
                // space + the primary button (`input::Nav` — the one definition of
                // the navigation bindings, shared with the transform overlay).
                // Taking it here is also what keeps space+Alt panning rather than
                // sampling.
                if nav.begin(&e) {
                    // Whatever was being drawn was never meant to be paint — it was
                    // the opening half of a pinch (§18.1.7). Cancelled rather
                    // than committed, so reaching for the canvas leaves no mark.
                    //
                    // Usually there is nothing even to cancel: the first finger's
                    // press is *held* rather than believed, so a second one landing
                    // inside the wait drops a question instead of taking back an
                    // answer (§18.1.11). That is the fix for the pinch that used to
                    // paint — what remains here is the pen mid-stroke, and the
                    // finger that had already travelled far enough to mean it.
                    landing.abandon();
                    // And the press is navigation, so the hover's promise of paint
                    // is withdrawn with it (§18.1.10).
                    hover_gone(state);
                    canvas_active.set(true);
                    return;
                }
                // The drag table (`drags`): which bound gesture this press's
                // chord+button opens, if any. Below `nav` — which is what leaves
                // space+accelerator a zoom and space+Alt a pan — and above the
                // playback guard, because whether an action survives playback is
                // the action's own claim, asked inside `find`, not this ladder's
                // ordering. An unbound or declined chord falls through to the
                // paint path: over a selection tool Alt+drag is still the
                // subtract marquee (§6.8).
                match drags::find(state, &e) {
                    // The brush-tuning drag — Size sideways, Flow up and down
                    // (§18.1.9).
                    //
                    // Deliberately *not* `canvas_active`, for the eyedropper's
                    // reason below: the Brush panel is where this gesture's
                    // answer is read, so fading the chrome would hide the one
                    // thing it is for.
                    Some(DragAction::TuneBrush) => {
                        if tune.begin(&e) {
                            // A stroke was in flight only if some *other* pointer
                            // opened one; it can no longer be finished by this
                            // press, and a gesture the hand has walked away from
                            // must leave no mark.
                            landing.abandon();
                            // The ring at the press is the size's readout now
                            // (§18.1.9); a second circle under it would be two
                            // sizes for one brush. The class above takes the
                            // crosshair down for the whole drag on top of that.
                            hover_gone(state);
                            return;
                        }
                        // No engine yet, so nothing to tune: the press falls
                        // through to the paint path, which does nothing with it
                        // for the same reason.
                    }
                    // The press samples the canvas instead of painting on it,
                    // and the drag keeps sampling — the binding Clip Studio
                    // Paint and Rebelle both put on Alt, so a color is picked
                    // up without putting the brush down (§18.0.2).
                    Some(DragAction::PickColor) => {
                        capture_pointer(&e);
                        // Deliberately *not* `canvas_active`: the chrome fade
                        // exists to hand the screen back to the painting
                        // mid-stroke, but the Color panel is where a pick's
                        // answer shows up, so fading it out would hide the one
                        // thing this gesture is for.
                        picking.set(true);
                        if let Some(s) = sample(state, &e) {
                            pick_color(state, s.pos);
                        }
                        return;
                    }
                    // The press picks up whichever layer is showing paint under
                    // it and the drag carries it (§16.11) — the Move tool's
                    // auto-select, without the tool.
                    Some(DragAction::PickAndTranslate) => {
                        // `begin` declines before the engine exists, where there
                        // is no view to map the press through and nothing
                        // painted to pick up.
                        let picked_up = carry.begin(&e);
                        if picked_up {
                            // A stroke another pointer opened can no longer be
                            // finished by this press, and a gesture the hand has
                            // walked away from must leave no mark (the tuning
                            // arm's argument, unchanged).
                            landing.abandon();
                            // The press is not paint, so the circle promising it
                            // goes with it (§18.1.10).
                            hover_gone(state);
                            // **Faded**, unlike the two arms above: this
                            // gesture's answer is the painting itself moving, so
                            // there is no panel to keep legible and every reason
                            // to hand the screen back to the picture (§25.3).
                            canvas_active.set(true);
                            return;
                        }
                        // Declined, so the press falls through to the paint
                        // path — which does nothing with it for the same
                        // reason, exactly as the tuning arm's decline does.
                    }
                    None => {}
                }
                // Nothing may be *committed* while the playhead is moving: a
                // commit clears the withheld half of the timeline, so a stroke
                // laid under a running playback would delete the rest of the
                // piece (`panels::timeline`). Panning is taken above and stays
                // available — looking around during playback costs the document
                // nothing.
                if panels::timeline::is_playing(state) {
                    return;
                }
                // The pen's other end draws too — it is a contact like the tip,
                // differing only in the brush it arrives holding (§18.1.8).
                if input::is_contact(&e) {
                    capture_pointer(&e);
                    // Painting and selecting are the same gesture from here — the
                    // tool decides what the engine builds (§6.8).
                    let tool = current_tool(state);
                    canvas_active.set(true);
                    // From here the press is paint — or, for a finger, a question
                    // whose answer is *probably* paint. What it does with itself
                    // is the gesture's business rather than this handler's,
                    // including the case where there is no view to land in yet,
                    // which opens nothing and leaves the moves after it inert
                    // (`input::Landing`, `input::Paint`).
                    landing.begin(&e, tool);
                }
            },
            onpointermove: move |e| {
                // Navigation is asked first, and **unconditionally** — including
                // while a stroke is in flight. A lone finger's moves say nothing to
                // the view (`Nav::advance` answers false and the stroke below sees
                // them), but they still have to be *recorded*, because a second
                // finger landing pairs with where the first one has got to rather
                // than with where it pressed (§18.1.7).
                if nav.advance(&e) {
                    // The view moved, so nothing below applies: a sample taken here
                    // would be mapped through the view as it was *before* the move,
                    // and with two fingers down there is no single pointer to report
                    // as a cursor anyway.
                    hover_gone(state);
                    return;
                }
                // The brush moved rather than the pointer's meaning on the canvas
                // (§18.1.9): nothing below applies, since this press was never
                // painting and a peer has no use for a cursor being used as a knob.
                if tune.advance(&e) {
                    hover_gone(state);
                    return;
                }
                // A composing mode opened under the hand (`crate::modes`). Its
                // catcher covers the canvas, so no *new* press can reach here —
                // but this pointer was captured by the canvas before the catcher
                // existed, and a captured pointer's moves are delivered to the
                // element that took them whatever has been stacked over it since.
                // A pen drawing while the other hand reaches for Transform is
                // exactly that, and without this the stroke would go on feeding
                // the fitter underneath the widget.
                //
                // Cancelled rather than left to commit, for the same reason a
                // pinch cancels the stroke it interrupts: the canvas stopped
                // taking paint the moment the mode took it, so the gesture must
                // leave no mark.
                if modes::is_composing(state) {
                    landing.abandon();
                    // And a layer being carried is put back rather than left to
                    // commit, for the identical reason (`PickMove::abandon`).
                    carry.abandon();
                    // And the canvas is no longer what is in hand — the mode is.
                    // Unlike the pinch, which goes on using it, so `nav` sets
                    // this the other way. Left dimmed, the mode's own bar would
                    // be faded and taking no clicks (§11) until the pen lifted,
                    // which is the one control the artist now needs.
                    canvas_active.set(false);
                    hover_gone(state);
                    return;
                }
                // A layer is being carried under the pointer (§16.11): the paint
                // is moving rather than being laid down, so nothing below applies
                // and the brush circle stays off the picture being moved.
                //
                // **Below** the composing check rather than beside `tune` above,
                // and the order is load-bearing: this gesture holds a document
                // preview, so a mode opening under a captured pointer has to
                // reach `abandon` before another move renews it. Tuning can sit
                // above because it edits no document and has nothing to renew.
                if carry.advance(&e) {
                    hover_gone(state);
                    return;
                }
                // The hover, ahead of the mapping below on purpose: the brush
                // cursor rides the pointer in the element's own px and needs no
                // view, so it is honest from the first frame — while the engine
                // is still being built, its overlay simply has no size to give
                // the position (§18.1.10).
                hover_at(state, elem_xy(&e));
                // The canvas takes pointer events from the first frame, while the
                // engine is still being built asynchronously — so there may be no
                // view to map through yet, and a move with nowhere to land simply
                // does nothing.
                if let Some(s) = sample(state, &e) {
                    if picking() {
                        // Alt+drag keeps sampling; `pick_color` drops a move that
                        // arrives while the last sample is still settling.
                        pick_color(state, s.pos);
                        // And a held touch pick carries its swatch along with the
                        // finger (§18.1.11). Silent for the chord binding, which
                        // has a cursor and a panel and needs neither
                        // (`input::move_loupe`).
                        move_loupe(state, elem_xy(&e));
                    } else if !landing.advance(&e) {
                        // The paint gesture takes the move if it has one in
                        // flight, and says so. A move with no gesture behind it
                        // is a *hover*, and the mark preview rides it
                        // (§18.1.10): the engine adds this sample to its
                        // trailing window and folds the stroke a drag begun
                        // this instant would open, continuing the hover's
                        // heading from the cursor.
                        hover_stroke(state, s, &e);
                    }
                    // Where collaborators see this client's pointer (§17.4), and
                    // where a guide open here draws its rays (§20.9) — one fact,
                    // and `point_at` owns which of those two readers is asking
                    // and therefore whether it costs a repaint.
                    //
                    // Outside the paint branch above on purpose: the hand is
                    // somewhere whether or not it is painting, and the rays are
                    // most use *during* a stroke, showing the line the grid would
                    // have it take.
                    point_at(state, Some(s.pos));
                }
            },
            onpointerleave: move |_| {
                // The hover ends where the canvas does — for the brush cursor
                // (§18.1.10) exactly as for the cursor peers see and the guide
                // rays hang from (§20.9). A finger's lift arrives here too:
                // pointer types that cannot hover are owed a leave after every
                // up, so a touch never strands the circle.
                hover_gone(state);
                point_at(state, None);
            },
            // One finger of several lifting ends nothing — the rest are still
            // navigating, and tearing down here would end the gesture on whichever
            // finger the hand happened to raise first (§18.1.7).
            onpointerup: move |e| {
                if !nav.release(&e) {
                    // The hand has left the glass. If it came and went without ever
                    // moving the view or laying a mark, it made a **tap** — two
                    // fingers for undo and three for redo, the pairing every
                    // touch-first painting app ships and therefore the one nobody
                    // has to be taught (§18.1.11).
                    //
                    // Read *before* the teardown, which is what clears it, and
                    // spent *after*, because undo puts down whatever is in hand
                    // (`commands::edit_history`) and this handler is what is
                    // holding it.
                    let tap = nav.take_tap();
                    // A tap never had the canvas in hand, so the chrome it faded on
                    // the way in comes back rather than being put to sleep behind
                    // it: `end_interaction` reads this flag to decide, and a
                    // gesture that lasted a tenth of a second was not somebody
                    // asking for the panels to get out of the way.
                    if matches!(tap, Some(2 | 3)) {
                        canvas_active.set(false);
                    }
                    end_interaction(state, landing, nav, tune, carry);
                    match tap {
                        Some(2) => Command::Undo.run(state),
                        Some(3) => Command::Redo.run(state),
                        // One finger is a dot the brush already painted, and four
                        // is a hand put down on the glass. Neither is an act.
                        _ => {}
                    }
                }
            },
            onpointercancel: move |e| {
                if !nav.release(&e) {
                    // No tap is taken here, and `Nav::stop` drops the one the
                    // release recorded: a cancel is the browser saying the gesture
                    // never finished, and an undo is not something to do on a
                    // gesture that was interrupted.
                    end_interaction(state, landing, nav, tune, carry);
                }
            },
            onwheel: move |e| nav.wheel(e),
        }
    }
}
