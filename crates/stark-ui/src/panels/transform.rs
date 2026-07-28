//! The transform gesture's chrome (TRANSFORM_DESIGN.md §6): the on-canvas box
//! with drag handles, and the bar carrying the flips and "Done".
//!
//! Entered from the selection bar's "Transform" button. Everything before "Done"
//! is a **lossless preview**: the drags accumulate into one
//! [`TransformState`](crate::state::TransformState) whose affine maps the
//! original box onto the current one, and every change previews through
//! `ViewCommand::PreviewTransform` — the same renderer the commit runs, over the
//! committed tiles, so what is on screen is exactly what "Done" will produce and
//! a long drag never compounds resampling loss. "Done" commits a single
//! `DocCommand::Transform`: one undo step per gesture, like the frame drag
//! (FRAME_DESIGN.md §7).
//!
//! While the mode is active a full-viewport catcher sits over the canvas, so a
//! stray drag cannot paint under the handles — unlike the frame, whose interior
//! must stay paintable, the whole point of this mode is that the pointer is
//! *composing*, not painting.

use dioxus::prelude::*;

use super::frame::{Grip, content_rect, page_xy, view_rect};
use crate::layout::chrome_class;
use crate::state::{AppState, TransformState, dispatch};
use stark_core::command::{DocCommand, ViewCommand};
use stark_core::geom::Vec2;

/// Enter transform mode around the current selection.
///
/// The handles mount on the selection's analytic hull; an unbounded selection
/// (select-all, an inversion) falls back to the painted content's bounds — the
/// whole layer is what an unbounded selection holds — and an empty canvas to the
/// view, so the box always exists. The target layer is the active one, or the
/// topmost paintable layer when a matte is selected (a matte refuses transforms
/// the same way it refuses strokes).
pub fn begin_transform(state: AppState) {
    let obs = state.obs.read();
    let Some(o) = obs.as_ref() else { return };
    let layer = o
        .layers
        .iter()
        .find(|l| l.id == o.active_layer && l.is_paintable())
        .or_else(|| o.layers.iter().rev().find(|l| l.is_paintable()))
        .map(|l| l.id);
    let Some(layer) = layer else { return };
    let hull = o
        .selection_hull
        .or_else(|| content_rect(o))
        .unwrap_or_else(|| view_rect(o));
    drop(obs);
    let mut mode = state.transform;
    mode.set(Some(TransformState::begin(layer, hull)));
}

/// Update the gesture and show its consequence — every mutation funnels through
/// here, so the preview can never lag the state.
fn update(state: AppState, ts: TransformState) {
    let mut mode = state.transform;
    mode.set(Some(ts));
    dispatch(
        state,
        ViewCommand::PreviewTransform(Some((ts.layer, ts.affine()))),
    );
}

/// The transform bar: the two flips and "Done". Mounted only while the gesture
/// is in flight, in the same bottom column as the selection and frame bars.
#[component]
pub fn TransformBar() -> Element {
    let state = use_context::<AppState>();
    let Some(ts) = *state.transform.read() else {
        return rsx! {};
    };

    rsx! {
        div { class: chrome_class(state, "transform-bar"),
            span { class: "bar-label", "Transform" }
            button {
                class: "chip",
                title: "Mirror left \u{2194} right",
                onclick: move |_| {
                    update(state, TransformState { flip: (!ts.flip.0, ts.flip.1), ..ts });
                },
                "Flip \u{2194}"
            }
            button {
                class: "chip",
                title: "Mirror top \u{2195} bottom",
                onclick: move |_| {
                    update(state, TransformState { flip: (ts.flip.0, !ts.flip.1), ..ts });
                },
                "Flip \u{2195}"
            }
            span { class: "bar-sep" }
            button {
                class: "chip",
                title: "Apply the transform (one undo step)",
                onclick: move |_| {
                    if ts.is_identity() {
                        // Nothing changed: just drop the preview rather than
                        // spending an undo step on a no-op.
                        dispatch(state, ViewCommand::PreviewTransform(None));
                    } else {
                        // The commit clears the preview itself, so there is no
                        // intermediate frame showing the untransformed document.
                        dispatch(state, DocCommand::Transform {
                            layer: ts.layer,
                            affine: ts.affine(),
                        });
                    }
                    let mut mode = state.transform;
                    mode.set(None);
                },
                "Done"
            }
        }
    }
}

/// An in-flight handle drag: which grip, where the pointer went down (page px),
/// and the rect as it was then. Deltas are taken from the start rect rather than
/// accumulated, so rounding cannot drift over a long drag — same discipline as
/// the frame's [`FrameDrag`](super::frame::FrameDrag).
#[derive(Clone, Copy)]
struct Drag {
    grip: Grip,
    origin: Vec2,
    start: (Vec2, Vec2),
}

/// The transform box and its handles, over the canvas (TRANSFORM_DESIGN.md §6).
///
/// Reuses the frame's grips — same geometry, same stylesheet — with one
/// deliberate difference: the box *interior* also drags (as a move), because in
/// this mode the pointer composes rather than paints, and a full-viewport
/// catcher underneath blocks everything else from reaching the canvas.
#[component]
pub fn TransformOverlay() -> Element {
    let state = use_context::<AppState>();
    let mut drag = use_signal(|| None::<Drag>);

    let Some(ts) = *state.transform.read() else {
        return rsx! {};
    };
    let view = match state.obs.read().as_ref() {
        Some(o) => o.view,
        None => return rsx! {},
    };

    let tl = view.canvas_to_screen(ts.rect.0);
    let br = view.canvas_to_screen(ts.rect.1);
    let box_style = format!(
        "left: {}px; top: {}px; width: {}px; height: {}px;",
        tl.x,
        tl.y,
        br.x - tl.x,
        br.y - tl.y
    );

    // A pointer delta in screen px is a canvas delta over the zoom.
    let to_canvas = move |screen: Vec2, origin: Vec2| (screen - origin) / view.zoom;
    let mut start = move |e: &Event<PointerData>, grip: Grip| {
        e.stop_propagation();
        crate::platform::capture_pointer(e);
        drag.set(Some(Drag {
            grip,
            origin: page_xy(e),
            start: ts.rect,
        }));
    };
    let follow = move |e: &Event<PointerData>| {
        let Some(d) = drag() else { return };
        let rect = d.grip.apply(d.start, to_canvas(page_xy(e), d.origin));
        update(state, TransformState { rect, ..ts });
    };

    rsx! {
        // The catcher: soaks up every pointer event the box does not, so a drag
        // beside the box cannot start a stroke while composing.
        div { class: "transform-catcher" }

        div {
            class: chrome_class(state, "transform-overlay"),
            style: "{box_style}",
            // The interior moves the box — the composing mode's one free gesture.
            onpointerdown: move |e| start(&e, Grip::Move),
            onpointermove: move |e| follow(&e),
            onpointerup: move |e| {
                follow(&e);
                drag.set(None);
            },
            onpointercancel: move |_| { drag.set(None); },

            for grip in Grip::ALL {
                {
                    let (suffix, cursor) = grip.spec();
                    rsx! {
                        div {
                            key: "{suffix}",
                            class: "frame-grip frame-grip-{suffix}",
                            style: "cursor: {cursor};",
                            onpointerdown: move |e| start(&e, grip),
                            onpointermove: move |e| follow(&e),
                            onpointerup: move |e| {
                                follow(&e);
                                drag.set(None);
                            },
                            onpointercancel: move |_| { drag.set(None); },
                        }
                    }
                }
            }
        }
    }
}
