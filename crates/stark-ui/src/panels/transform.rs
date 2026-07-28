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

use dioxus::html::input_data::MouseButton;
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

/// What an in-flight drag has hold of: a resize/move grip, or the rotate handle
/// (which tracks the pointer's bearing about the box centre rather than a
/// positional delta).
#[derive(Clone, Copy)]
enum DragKind {
    Grip(Grip),
    Rotate {
        /// The pointer's bearing about the box centre when it went down.
        bearing: f32,
        /// The gesture's angle at that moment.
        angle: f32,
    },
    /// A view pan — middle-drag or space-drag, exactly as on the canvas.
    /// Composing a transform must not cost navigation, so every transform
    /// element diverts to this when those bindings are held; `Drag::origin` is
    /// re-anchored each move, since `ViewCommand::Pan` takes incremental deltas.
    Pan,
}

/// An in-flight handle drag: what it holds, where the pointer went down (page
/// px), and the rect as it was then. Deltas are taken from the *start* state
/// rather than accumulated, so rounding cannot drift over a long drag — same
/// discipline as the frame's [`FrameDrag`](super::frame::FrameDrag).
#[derive(Clone, Copy)]
struct Drag {
    kind: DragKind,
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
    // The box is laid out from the unrotated rect and turned by CSS about its
    // centre — the same composition the affine applies to the paint, so the
    // chrome and the preview always agree.
    let box_style = format!(
        "left: {}px; top: {}px; width: {}px; height: {}px; transform: rotate({}rad);",
        tl.x,
        tl.y,
        br.x - tl.x,
        br.y - tl.y,
        ts.angle,
    );
    // The box centre on screen — the rotate handle's pivot.
    let centre = view.canvas_to_screen((ts.rect.0 + ts.rect.1) * 0.5);

    // A pointer delta in screen px is a canvas delta over the zoom.
    let to_canvas = move |screen: Vec2, origin: Vec2| (screen - origin) / view.zoom;
    let bearing_about_centre = move |p: Vec2| (p.y - centre.y).atan2(p.x - centre.x);
    // The navigation bindings, exactly as the canvas has them: middle-drag or
    // space-drag pans. Checked first by every transform element, so holding
    // space over the box pans instead of moving the selection. Returns whether
    // the press was taken.
    let mut start_pan = move |e: &Event<PointerData>| -> bool {
        let pan =
            e.trigger_button() == Some(MouseButton::Auxiliary) || *state.space_down.peek();
        if pan {
            e.prevent_default(); // suppress middle-click autoscroll
            e.stop_propagation();
            crate::platform::capture_pointer(e);
            drag.set(Some(Drag {
                kind: DragKind::Pan,
                origin: page_xy(e),
                start: ts.rect,
            }));
        }
        pan
    };
    let mut start = move |e: &Event<PointerData>, kind: DragKind| {
        if start_pan(e) {
            return;
        }
        if e.trigger_button() != Some(MouseButton::Primary) {
            return;
        }
        e.stop_propagation();
        crate::platform::capture_pointer(e);
        drag.set(Some(Drag {
            kind,
            origin: page_xy(e),
            start: ts.rect,
        }));
    };
    // Wheel zoom, exactly as on the canvas. Anchored by page position: the
    // catcher shares the canvas's origin and the box does not, and page
    // coordinates are the one frame both report in.
    let wheel = move |e: Event<WheelData>| {
        e.prevent_default();
        e.stop_propagation();
        let dy = e.delta().strip_units().y;
        if dy != 0.0 {
            let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
            let p = e.page_coordinates();
            dispatch(state, ViewCommand::Zoom {
                anchor: Vec2::new(p.x as f32, p.y as f32),
                factor,
            });
        }
    };
    let mut follow = move |e: &Event<PointerData>| {
        let Some(d) = drag() else { return };
        match d.kind {
            // Translation is rotation-invariant: the box follows the pointer.
            DragKind::Grip(Grip::Move) => {
                let rect = Grip::Move.apply(d.start, to_canvas(page_xy(e), d.origin));
                update(state, TransformState { rect, ..ts });
            }
            // A resize works in the box's own (unrotated) frame: rotate the
            // pointer delta back by the current angle before applying it...
            DragKind::Grip(grip) => {
                let dc = to_canvas(page_xy(e), d.origin);
                let (sin, cos) = ts.angle.sin_cos();
                let local = Vec2::new(cos * dc.x + sin * dc.y, cos * dc.y - sin * dc.x);
                let rect = grip.apply(d.start, local);
                // ...and re-pin the un-dragged side. The box rotates about its
                // centre, and the resize moved the centre, which would otherwise
                // swing the whole box on screen by `(I − R)·(c₀ − c₁)`.
                let v = (d.start.0 + d.start.1 - rect.0 - rect.1) * 0.5;
                let shift = v - Vec2::new(cos * v.x - sin * v.y, sin * v.x + cos * v.y);
                let rect = (rect.0 + shift, rect.1 + shift);
                update(state, TransformState { rect, ..ts });
            }
            DragKind::Rotate { bearing, angle } => {
                let turned = angle + bearing_about_centre(page_xy(e)) - bearing;
                update(state, TransformState { angle: turned, ..ts });
            }
            // Incremental, re-anchoring the origin each move: `Pan` is a delta
            // command, unlike the grips' from-the-start geometry.
            DragKind::Pan => {
                let p = page_xy(e);
                dispatch(state, ViewCommand::Pan { delta: p - d.origin });
                drag.set(Some(Drag { origin: p, ..d }));
            }
        }
    };

    // The catcher advertises the pan while space is held, as the canvas cursor
    // would if it could still be seen.
    let catcher_class = if (state.space_down)() {
        "transform-catcher pan"
    } else {
        "transform-catcher"
    };

    rsx! {
        // The catcher: soaks up every pointer event the box does not, so a drag
        // beside the box cannot start a stroke while composing — but navigation
        // still works: middle-drag and space-drag pan, the wheel zooms.
        div {
            class: "{catcher_class}",
            onpointerdown: move |e| { start_pan(&e); },
            onpointermove: move |e| follow(&e),
            onpointerup: move |e| {
                follow(&e);
                drag.set(None);
            },
            onpointercancel: move |_| { drag.set(None); },
            onwheel: wheel,
        }

        div {
            class: chrome_class(state, "transform-overlay"),
            style: "{box_style}",
            // Grips and the rotate knob bubble their wheel events here, so the
            // zoom works over the box too.
            onwheel: wheel,
            // The interior moves the box — dragging the region *is* the
            // translation, which is what freed the top handle for rotation.
            onpointerdown: move |e| start(&e, DragKind::Grip(Grip::Move)),
            onpointermove: move |e| follow(&e),
            onpointerup: move |e| {
                follow(&e);
                drag.set(None);
            },
            onpointercancel: move |_| { drag.set(None); },

            // The eight resize grips. `Move` is skipped: its pill has become the
            // rotate handle below, and translation lives on the interior.
            for grip in Grip::ALL.into_iter().filter(|g| *g != Grip::Move) {
                {
                    let (suffix, cursor) = grip.spec();
                    rsx! {
                        div {
                            key: "{suffix}",
                            class: "frame-grip frame-grip-{suffix}",
                            style: "cursor: {cursor};",
                            onpointerdown: move |e| start(&e, DragKind::Grip(grip)),
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

            // The rotate handle, where the frame keeps its move pill: it tracks
            // the pointer's bearing about the box centre, so the box turns with
            // the hand rather than by an abstract delta.
            div {
                class: "transform-rotate",
                onpointerdown: move |e| {
                    let bearing = bearing_about_centre(page_xy(&e));
                    start(&e, DragKind::Rotate { bearing, angle: ts.angle });
                },
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
