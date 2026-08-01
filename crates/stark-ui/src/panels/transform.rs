//! The transform gesture's chrome (§16.6): an **ellipse**, not
//! a box of handles — the image of a reference ellipse under the accumulated
//! transform, so the widget's own shape shows the deformation. One surface,
//! three gestures, chosen by where the drag starts:
//!
//! - **inside** — translate;
//! - **on the rim** — rotate and scale uniformly: the grabbed rim point follows
//!   the pointer under a similarity about the centre, so tangential motion is
//!   pure rotation, radial motion pure scale, and anything between blends the
//!   two with no mode to pick;
//! - **outside** — stretch and shear along the grab direction: the grabbed
//!   point follows the pointer under a rank-1 map that pins the perpendicular
//!   diameter, so radial pull is a directional scale and tangential drag a
//!   skew.
//!
//! That last region is what a rectangle-of-handles never offers: skew and
//! non-axis-aligned scaling fall out of the same gesture vocabulary instead of
//! being a bolted-on mode. The maths lives on
//! [`TransformState`](crate::state::TransformState); this module is only the
//! catcher that classifies pointer events and the two visual elements (the
//! ellipse and its north dot).
//!
//! Entered from the selection bar's "Transform" button. Everything before
//! "Done" is a **lossless preview**: gestures accumulate into one
//! `TransformState` and every change previews through
//! `ViewCommand::PreviewTransform` — the same renderer the commit runs, over
//! the committed tiles, so what is on screen is exactly what "Done" will
//! produce. "Done" commits a single `DocCommand::Transform`: one undo step per
//! gesture, like the frame drag (§15.7).
//!
//! While the mode is active the full-viewport catcher owns the pointer, so a
//! stray drag cannot paint — but navigation still works: middle-drag and
//! space-drag pan, the wheel zooms (see `input::Nav`). All gesture maths is in
//! canvas space, so panning or zooming mid-gesture cannot corrupt it.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use super::frame::{content_rect, view_rect};
use crate::icons::{self, icon};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_class;
use crate::state::{AppState, TransformRegion, TransformState, dispatch};
use stark_core::command::{DocCommand, ViewCommand};
use stark_core::geom::Vec2;

/// Half-width of the rim's grab band, screen px — converted to canvas px by the
/// zoom, so the rim is equally grabbable at any magnification.
const RIM_BAND_PX: f32 = 10.0;

/// Screen-px floor for the widget's radius at entry, so a hairline selection
/// still mounts a circle with an inside to translate by.
const MIN_RADIUS_PX: f32 = 28.0;

/// Pointer travel below which a gesture reads as a jiggle and snaps back to its
/// start (screen px): an accidental touch must never resample the paint.
const SNAP_PX: f32 = 2.0;

/// Enter transform mode around the current selection.
///
/// The widget mounts as the ellipse inscribed in the selection's analytic hull;
/// an unbounded selection (select-all, an inversion) falls back to the painted
/// content's bounds — the whole layer is what an unbounded selection holds —
/// and an empty canvas to the view, so the widget always exists. The target
/// layer is the active one, or the topmost paintable layer when a matte is
/// selected (a matte refuses transforms the same way it refuses strokes).
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
    let min_radius = MIN_RADIUS_PX / o.view.zoom;
    drop(obs);
    let mut mode = state.transform;
    mode.set(Some(TransformState::begin(layer, hull, min_radius)));
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
            // The axis was already the only thing distinguishing these two buttons, and
            // it was already carried by a glyph — the arrow that used to trail the word.
            // Leading it instead puts the pair in the same shape every other chip in the
            // application wears, glyph then noun, and the mark that says *which* flip is
            // now a picture of the mirroring rather than of its axis.
            button {
                class: "chip",
                title: "Mirror left \u{2194} right",
                onclick: move |_| update(state, ts.flipped_h()),
                {icon(icons::FLIP_H)}
                "Flip"
            }
            button {
                class: "chip",
                title: "Mirror top \u{2195} bottom",
                onclick: move |_| update(state, ts.flipped_v()),
                {icon(icons::FLIP_V)}
                "Flip"
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

/// An in-flight drag: which region it started in (the gesture is locked at the
/// press — crossing the rim mid-drag must not change what the hand is doing),
/// where it started in canvas px, and the state as it was then. Everything is
/// recomputed from the start rather than accumulated, so rounding cannot drift
/// over a long drag, and it lives in canvas space, so panning or zooming
/// mid-gesture (the wheel still works) cannot corrupt it.
#[derive(Clone, Copy)]
struct Drag {
    region: TransformRegion,
    from: Vec2,
    start: TransformState,
}

/// The transform widget: a full-viewport catcher that owns every pointer event
/// and classifies it against the ellipse (the maths — regions, similarity,
/// rank-1 stretch — is on [`TransformState`]), plus the ellipse itself and its
/// north dot, which are purely visual. No per-handle DOM: the whole viewport is
/// the control surface.
#[component]
pub fn TransformOverlay() -> Element {
    let state = use_context::<AppState>();
    let mut drag = use_signal(|| None::<Drag>);
    // What the resting pointer is over, for cursor feedback only.
    let mut hover = use_signal(|| None::<TransformRegion>);
    // The canvas's own navigation bindings, live on the catcher: composing a
    // transform must not cost the view (see `input::Nav`).
    let nav = Nav::use_nav(state);

    let Some(ts) = *state.transform.read() else {
        return rsx! {};
    };
    let view = match state.obs.read().as_ref() {
        Some(o) => o.view,
        None => return rsx! {},
    };

    // The widget: a circle of the reference radius, deformed by the linear map
    // via CSS about its centre — the same composition the affine applies to the
    // paint, so the widget and the preview cannot disagree. It stays a circle
    // exactly as long as the transform is a similarity; eccentricity *is* the
    // distortion.
    let cs = view.canvas_to_screen(ts.center);
    let r = ts.radius * view.zoom;
    // The gesture's linear map is canvas-space; what CSS draws is on screen, so the
    // view's own orientation composes onto it. The zoom is already in `r`, and the
    // orientation carries no scale, so this is the whole difference — a turned or
    // mirrored canvas turns and mirrors the widget with the paint it stands for.
    let l = view.orientation() * ts.linear;
    let ellipse_style = format!(
        "left: {}px; top: {}px; width: {}px; height: {}px; \
         transform: matrix({}, {}, {}, {}, 0, 0);",
        cs.x - r,
        cs.y - r,
        2.0 * r,
        2.0 * r,
        l.x_axis.x,
        l.x_axis.y,
        l.y_axis.x,
        l.y_axis.y,
    );

    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));
    let classify = move |pc: Vec2| ts.region(pc, RIM_BAND_PX / view.zoom);
    let snap = SNAP_PX / view.zoom;

    let mut follow = move |e: &Event<PointerData>| {
        if nav.advance(e) {
            return;
        }
        let pc = to_canvas(e);
        let Some(d) = drag() else {
            // Resting: report what a press here would do, for the cursor.
            hover.set(Some(classify(pc)));
            return;
        };
        match d.region {
            TransformRegion::Inside => update(state, d.start.translated(d.from, pc, snap)),
            TransformRegion::Rim => update(state, d.start.turned_scaled(d.from, pc, snap)),
            TransformRegion::Outside => update(state, d.start.stretched(d.from, pc, snap)),
        }
    };
    let mut finish = move |e: &Event<PointerData>| {
        follow(e);
        nav.stop();
        drag.set(None);
    };

    // The cursor announces the region the press would grab; the pan class wins
    // while space is held (an inline cursor would override it, so none is set).
    let panning = (state.space_down)();
    let catcher_class = if panning {
        "transform-catcher pan"
    } else {
        "transform-catcher"
    };
    let cursor = match (panning, drag(), hover()) {
        (true, ..) => "",
        (_, Some(d), _) => match d.region {
            TransformRegion::Inside => "cursor: move;",
            _ => "cursor: grabbing;",
        },
        (_, None, Some(TransformRegion::Inside)) => "cursor: move;",
        (_, None, Some(TransformRegion::Rim)) => "cursor: grab;",
        (_, None, Some(TransformRegion::Outside)) => "cursor: crosshair;",
        (_, None, None) => "",
    };

    rsx! {
        div {
            class: "{catcher_class}",
            style: "{cursor}",
            onpointerdown: move |e| {
                if nav.begin(&e) {
                    // A second finger turns the drag into navigation (§18.1.7).
                    // The preview it had built stands — a transform commits on
                    // "Done", not on release, so letting go to look around costs
                    // nothing.
                    drag.set(None);
                    return;
                }
                if e.trigger_button() != Some(MouseButton::Primary) {
                    return;
                }
                e.stop_propagation();
                crate::platform::capture_pointer(&e);
                let pc = to_canvas(&e);
                drag.set(Some(Drag {
                    region: classify(pc),
                    from: pc,
                    start: ts,
                }));
            },
            onpointermove: move |e| follow(&e),
            // Fingers still on the glass mean the gesture is not over — see the
            // canvas's own release handler.
            onpointerup: move |e| if !nav.release(&e) { finish(&e) },
            onpointercancel: move |e| if !nav.release(&e) { nav.stop(); drag.set(None); },
            onwheel: move |e| nav.wheel(e),
        }

        div {
            class: chrome_class(state, "transform-ellipse"),
            style: "{ellipse_style}",
            // The north dot rides the same CSS transform as its parent, so it
            // marks the reference ellipse's "up" wherever the deformation has
            // carried it — without it a rotated circle looks unrotated.
            div { class: "transform-north" }
        }
    }
}
