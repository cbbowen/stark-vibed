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
//! owns that (§25.3), and `input::Gestures` owns which of them holds the pointer
//! and the order a move is offered in. This component holds only the answer to
//! "which of them is this press".

use dioxus::prelude::*;

use crate::commands;
use crate::drags;
use crate::input::{
    self, Holder, elem_xy, end_interaction, hover_at, hover_gone, hover_stroke, move_loupe,
    pick_color, point_at, sample, samples, use_gestures,
};
use crate::panels;
use crate::panels::select::current_tool;
use crate::platform::capture_pointer;
use crate::render::CANVAS_ID;
use crate::state::{AppState, resize, use_obs};
use stark_ui::commands::Command;
use stark_ui::drags::DragAction;

/// The full-window painting surface (a WebGPU canvas the engine draws into).
#[component]
pub fn Canvas() -> Element {
    let state = use_context::<AppState>();
    // Paint — behind the wait a finger's press is held in (§18.1.11) — navigation,
    // the brush-tuning drag (§18.1.9) and the layer carry (§16.11), as one value.
    let gestures = use_gestures(state);
    // Whether an Alt+drag is sampling color off the canvas rather than painting on
    // it (§18.0.2). Shared state rather than one of `gestures`, because the options
    // bar is mounted on *armed but not dragging*.
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
    let paintable = look().is_some_and(|(paintable, _)| paintable);
    // The pick chord (Alt by default) arms the eyedropper over the brush, and the
    // cursor says so before it is used — the only thing that makes a modifier
    // binding discoverable. Asked of the drag table (`stark_ui::drags::armed`), the same
    // table the press will ask, so the promise moves with the binding. Not over a
    // selection tool, where alt already means "subtract from the selection"
    // (§6.8), so the cursor promises the pick exactly where a press would
    // take one. It beats `no-paint`, because a layer that takes no paint can still
    // be sampled.
    //
    // The layer carry announces itself the same way and owes it for the same
    // reason (§16.11): Shift+drag is a secret without a cursor that says so
    // before it is used. It stands down over exactly what the pick does — Shift is
    // the union marquee there (§6.8) — which is why both read one `Hand::free`
    // rather than each spelling the same three tests out.
    //
    // Both read the one hand the press path and the eyedropper's bar read
    // (`input::hand`), through a memo — so the canvas re-renders when either promise
    // would change, not on every fact a hand is made of (a press flips two of them).
    // `is_playing` peeks, and that is right here rather than merely cheap: the
    // answer is recomputed when the held modifiers change, which is the moment a
    // cursor could start promising anything at all.
    let promised = use_memo(move || {
        let tool = look().map_or(stark_engine::command::Tool::Brush, |(_, tool)| tool);
        let hand = input::hand(state, tool);
        let table = state.drags.read();
        let held = (state.held_mods)();
        let carrying = stark_ui::drags::armed(&table, held)
            .is_some_and(|a| a == DragAction::PickAndTranslate && a.claims(hand));
        (hand.armed(&table, held), carrying)
    });
    let (sampling, carrying) = promised();
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
                // Browser zoom and a move between monitors both land here.
                input::refresh_pixel_ratio();
                if let Ok(size) = e.get_content_box_size() {
                    resize(state, size.width as u32, size.height as u32);
                }
            },
            // Strokes and pans capture the pointer (like the pads/pickers): leaving the
            // window mid-stroke keeps painting — the infinite canvas extends past the
            // viewport anyway — and the interaction ends on release/cancel, never by
            // crossing the canvas edge.
            onpointerdown: move |e| {
                input::refresh_pixel_ratio();
                // Navigation first: a second finger on the glass, middle-drag, or
                // space + the primary button (`input::Nav` — the one definition of
                // the navigation bindings, shared with the transform overlay).
                // Taking it here is also what keeps space+Alt panning rather than
                // sampling.
                if gestures.begin_nav(&e) {
                    // Faded, unlike tuning: the pinch goes on using the canvas.
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
                let taken = match drags::find(state, &e) {
                    // The brush-tuning drag — Size sideways, Flow up and down
                    // (§18.1.9).
                    //
                    // Deliberately *not* `canvas_active`, for the eyedropper's
                    // reason below: the Brush panel is where this gesture's
                    // answer is read, so fading the chrome would hide the one
                    // thing it is for.
                    //
                    // Declined — no engine yet, or another pointer's gesture holds
                    // the canvas — the press falls through to the paint path,
                    // which declines it for the same reason.
                    Some(DragAction::TuneBrush) => gestures.begin_tune(&e),
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
                        true
                    }
                    // The press picks up whichever layer is showing paint under
                    // it and the drag carries it (§16.11) — the Move tool's
                    // auto-select, without the tool. Declined as the tuning arm is.
                    Some(DragAction::PickAndTranslate) => {
                        let taken = gestures.begin_carry(&e);
                        if taken {
                            // **Faded**, unlike the two arms above: this
                            // gesture's answer is the painting itself moving, so
                            // there is no panel to keep legible and every reason
                            // to hand the screen back to the picture (§25.3).
                            canvas_active.set(true);
                        }
                        taken
                    }
                    None => false,
                };
                if taken {
                    return;
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
                // differing only in the brush it arrives holding (§18.1.8). Asked
                // whether paint may open before the capture and the fade, which a
                // press another pointer's gesture refuses must not cost.
                if input::is_contact(&e) && gestures.admits(Holder::Paint) {
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
                    gestures.begin_paint(&e, tool);
                }
            },
            onpointermove: move |e| {
                // Navigation, tuning, a mode opened under the hand, the layer carry
                // — in that order, and a move one of them takes is nothing else's
                // (`input::Gestures::advance`, §25.4).
                if gestures.advance(&e).is_some() {
                    return;
                }
                // The hover, ahead of the mapping below on purpose: the brush
                // cursor rides the pointer in the element's own px and needs no
                // view, so it is honest from the first frame — while the engine
                // is still being built, its overlay simply has no size to give
                // the position (§18.1.10).
                hover_at(state, elem_xy(&e));
                // Mapped once, and every branch below returns before the engine
                // exists: the canvas takes pointer events from the first frame, and
                // a move with nowhere to land does nothing.
                let at = if *picking.peek() {
                    let Some(s) = sample(state, &e) else { return };
                    // Alt+drag keeps sampling; `pick_color` drops a move that
                    // arrives while the last sample is still settling.
                    pick_color(state, s.pos);
                    // And a held touch pick carries its swatch along with the
                    // finger (§18.1.11). Silent for the chord binding, which
                    // has a cursor and a panel and needs neither
                    // (`input::move_loupe`).
                    move_loupe(state, elem_xy(&e));
                    s.pos
                } else if gestures.paints(&e) {
                    // Every report the browser coalesced into this event, read
                    // only for a move a stroke takes — a hover needs one.
                    let Some(reports) = samples(state, &e) else { return };
                    // Another pointer's move under a live stroke — a refused palm —
                    // lays nothing, and is not where the hand drawing is.
                    if !gestures.advance_paint(&e, &reports) {
                        return;
                    }
                    let Some(last) = reports.last() else { return };
                    last.pos
                } else {
                    let Some(s) = sample(state, &e) else { return };
                    // A move with no gesture behind it is a *hover*, and the mark
                    // preview rides it (§18.1.10): the engine adds this sample to
                    // its trailing window and folds the stroke a drag begun this
                    // instant would open, continuing the hover's heading from the
                    // cursor.
                    hover_stroke(state, s, &e);
                    s.pos
                };
                // Where collaborators see this client's pointer (§17.4), and
                // where a guide open here draws its rays (§20.9) — one fact,
                // and `point_at` owns which of those two readers is asking
                // and therefore whether it costs a repaint.
                //
                // Outside the paint branch above on purpose: the hand is
                // somewhere whether or not it is painting, and the rays are
                // most use *during* a stroke, showing the line the grid would
                // have it take.
                point_at(state, Some(at));
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
                let nav = gestures.nav();
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
                    end_interaction(gestures);
                    match tap {
                        Some(2) => commands::run(Command::Undo, state),
                        Some(3) => commands::run(Command::Redo, state),
                        // One finger is a dot the brush already painted, and four
                        // is a hand put down on the glass. Neither is an act.
                        _ => {}
                    }
                }
            },
            onpointercancel: move |e| {
                if !gestures.nav().release(&e) {
                    // No tap is taken here, and `Nav::stop` drops the one the
                    // release recorded: a cancel is the browser saying the gesture
                    // never finished, and an undo is not something to do on a
                    // gesture that was interrupted.
                    end_interaction(gestures);
                }
            },
            onwheel: move |e| gestures.nav().wheel(e),
        }
    }
}
