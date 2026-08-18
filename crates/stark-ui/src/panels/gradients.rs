//! The gradient library's pop-out (§22.3) and the trace mode it arms (§22.2).
//!
//! The library has no panel: it lives behind the ramp strip on the bars that
//! consume it — the shared gradient bar (§22.4) and the gradient-map filter bar
//! (§21.11). Clicking the strip flies the library out of it ([`GradientWell`]),
//! the frame bar's color pop-out pattern: the well anchors it, the list is the
//! rows, and the Trace button that makes a new entry rides its header. The
//! *making* of one is a mode over the canvas, like the transform: while armed,
//! a full-viewport catcher owns the pointer, the drag is collected in **canvas
//! space** (so panning or zooming mid-trace cannot corrupt it — `input::Nav`
//! stays live on the catcher), and release hands the polyline to
//! [`gradients::capture`]. A drag too short to mean anything ends the mode
//! without a capture, so a stray click is its own undo.
//!
//! There is no gradient *tool* in the engine's `Tool` enum, deliberately: the
//! trace never touches the document — no action, no footprint, nothing a peer
//! sees — so it is a frontend mode ending in a request, the eyedropper's
//! pattern stretched along a line (§4, §18.0.2).

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::gradients;
use crate::icons::{self, icon, label};
use crate::input::{Nav, page_xy};
use crate::platform::capture_pointer;
use crate::state::AppState;
use stark_model::geom::Vec2;

/// How far the pointer must move, in **screen** px, before the trace keeps
/// another point. Screen rather than canvas px because it is decimation of the
/// *hand*: the engine resamples by canvas arc length either way, and a zoomed-out
/// trace should not collect a point per texel of jitter.
const TRACE_MIN_STEP_PX: f32 = 2.0;

/// The ramp strip on a bar, and the library pop-out it opens.
///
/// `strip` is the CSS of the ramp the bar is showing ([`gradients::css_strip`]),
/// or `None` when it has none yet — the face is then a dashed placeholder
/// wearing the library's glyph, so an empty well still says what arrives in it.
/// Either face is a button and the click is the same: the library flies up.
#[component]
pub fn GradientWell(strip: Option<String>, title: &'static str) -> Element {
    let mut open = use_signal(|| false);
    rsx! {
        span { class: "gradient-well",
            if let Some(strip) = strip {
                button {
                    class: "bar-gradient-strip",
                    title: "{title}",
                    style: "background: {strip};",
                    onclick: move |_| open.set(!open()),
                }
            } else {
                button {
                    class: "bar-gradient-strip empty",
                    title: "{title}",
                    onclick: move |_| open.set(!open()),
                    {icon(icons::GRADIENT)}
                }
            }
            if open() {
                GradientPopout { open }
            }
        }
    }
}

/// The library itself, flown up out of the well: the rows, the notice, and the
/// Trace button that makes a new entry.
///
/// Mounted only while open, like the frame bar's color pop-out — so it re-reads
/// the library each time and holds no state worth losing. Arming the trace
/// closes it (the catcher wants the canvas the pop-out is floating over); under
/// the gradient bar the mode's exclusivity unmounts the whole bar anyway
/// (`crate::modes`), and the well comes back when the composition does.
#[component]
fn GradientPopout(open: Signal<bool>) -> Element {
    let mut open = open;
    let state = use_context::<AppState>();
    let entries = (state.gradients.entries)();
    let armed = (state.gradients.armed)();
    let busy = (state.gradients.busy)();
    let notice = (state.gradients.notice)();

    // Hoisted out of the rsx! attribute: an `if`/`else` inside one trips
    // clippy's suspicious-else-formatting on the macro's expansion.
    let trace_title = if armed {
        "Stop tracing"
    } else {
        "Trace a line through the painting to make a gradient of the colors it crosses"
    };

    rsx! {
        div { class: "gradient-popout",
            div { class: "gradient-header",
                span { class: "gradient-header-title", "Gradients" }
                button {
                    class: if armed { "chip gradient-trace active" } else { "chip gradient-trace" },
                    disabled: busy,
                    title: trace_title,
                    onclick: move |_| {
                        let arm = !armed;
                        if arm {
                            // The catcher takes the canvas this pop-out floats
                            // over; the strip reopens it when the trace is done.
                            open.set(false);
                        }
                        gradients::set_armed(state, arm);
                    },
                    {icon(icons::TRACE_GRADIENT)}
                    {label(if busy { "Sampling\u{2026}" } else if armed { "Cancel" } else { "Trace" })}
                }
            }
            if let Some(text) = notice {
                div { class: "gradient-notice", "{text}" }
            }
            div { class: "gradient-list",
                if entries.is_empty() {
                    div { class: "gradient-empty",
                        "No gradients yet. Trace a line through your painting and the colors it crosses become one."
                    }
                }
                {
                    // The row the next fill would use (§22.4): the selected
                    // entry, or the first standing in — highlighted as *the*
                    // answer, so what a fill will lay is never a surprise.
                    let current = gradients::current_name(state);
                    rsx! {
                        for entry in entries {
                            {
                                let select_name = entry.name.clone();
                                let remove_name = entry.name.clone();
                                let strip = gradients::css_strip(&entry.gradient);
                                let active = current.as_deref() == Some(entry.name.as_str());
                                rsx! {
                                    div {
                                        key: "{entry.name}",
                                        class: if active { "gradient-row active" } else { "gradient-row" },
                                        // Clicking takes the ramp in hand — and
                                        // re-previews a composing fill, so mid-mode
                                        // the canvas answers the click.
                                        onclick: move |_| gradients::select(state, &select_name),
                                        div { class: "gradient-strip", style: "background: {strip};" }
                                        div { class: "gradient-row-foot",
                                            span { class: "gradient-row-name", title: "{entry.name}", "{entry.name}" }
                                            // The same trash the preset, layer and guide rows
                                            // wear: a fourth roster, same act, same mark.
                                            button {
                                                class: "gradient-remove",
                                                title: "Remove gradient",
                                                onclick: move |e| {
                                                    e.stop_propagation();
                                                    gradients::remove(state, &remove_name);
                                                },
                                                {icon(icons::REMOVE)}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The trace mode's catcher and rubber line, mounted while the pop-out's Trace
/// is armed. The transform overlay's shape (§16.6): a full-viewport catcher
/// that owns the pointer, with `Nav` live on it so the view stays reachable,
/// and an SVG polyline of the trace so far. The hint floats here rather than in
/// the pop-out, because arming closes the pop-out that would have held it.
#[component]
pub fn GradientTraceOverlay() -> Element {
    let state = use_context::<AppState>();
    // The trace in flight, canvas space; `None` while armed but not yet pressed.
    // Local to the overlay: the mode outlives no unmount — disarming is what
    // unmounts it, and a fresh arm should start clean anyway.
    let mut trace = use_signal(|| None::<Vec<Vec2>>);
    let nav = Nav::use_nav(state);

    if !(state.gradients.armed)() {
        return rsx! {};
    }
    let view = match state.obs.read().as_ref() {
        Some(o) => o.view,
        None => return rsx! {},
    };
    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));
    // Decimation of the hand, converted to the space the points are kept in.
    let min_step = TRACE_MIN_STEP_PX / view.zoom;

    let mut finish = move |e: &Event<PointerData>| {
        nav.stop();
        let Some(mut points) = trace.write().take() else {
            return;
        };
        points.push(to_canvas(e));
        // Release always ends the mode: a good trace captures, a stray click
        // cancels — either way the canvas is handed back.
        gradients::set_armed(state, false);
        if gradients::trace_long_enough(&points) {
            gradients::capture(state, points);
        }
    };

    let panning = (state.space_down)();
    let catcher_class = if panning {
        "gradient-catcher pan"
    } else {
        "gradient-catcher"
    };

    rsx! {
        div {
            class: "{catcher_class}",
            onpointerdown: move |e| {
                if nav.begin(&e) {
                    // A second finger or the pan bindings take the press: the
                    // view moves, the armed mode stands, and any half-drawn
                    // trace is abandoned — its points would straddle two views
                    // of the hand even though they are sound in canvas space.
                    trace.set(None);
                    return;
                }
                if e.trigger_button() != Some(MouseButton::Primary) {
                    return;
                }
                e.stop_propagation();
                capture_pointer(&e);
                trace.set(Some(vec![to_canvas(&e)]));
            },
            onpointermove: move |e| {
                if nav.advance(&e) {
                    return;
                }
                let mut guard = trace.write();
                let Some(points) = guard.as_mut() else { return };
                let p = to_canvas(&e);
                if points.last().is_none_or(|q| q.distance(p) >= min_step) {
                    points.push(p);
                }
            },
            onpointerup: move |e| if !nav.release(&e) { finish(&e) },
            onpointercancel: move |e| if !nav.release(&e) { nav.stop(); trace.set(None); },
            onwheel: move |e| nav.wheel(e),
        }

        div { class: "gradient-trace-hint",
            "Drag a line across the painting. Release to keep the ramp it crosses."
        }

        if let Some(points) = trace.read().as_ref() {
            {trace_line(points, view)}
        }
    }
}

/// The rubber line: the trace so far, in screen space — a dark casing under a
/// dashed light core so it reads over any painting, with a dot on the anchor
/// end. A plain function chosen by an `if`, like the transform's visuals.
fn trace_line(points: &[Vec2], view: stark_engine::ViewTransform) -> Element {
    let mut d = String::new();
    for (i, p) in points.iter().enumerate() {
        let s = view.canvas_to_screen(*p);
        d.push(if i == 0 { 'M' } else { 'L' });
        d.push_str(&format!("{:.2} {:.2} ", s.x, s.y));
    }
    let start = view.canvas_to_screen(points[0]);
    rsx! {
        svg { class: "gradient-trace-svg",
            path { class: "gradient-trace-casing", d: "{d}" }
            path { class: "gradient-trace-core", d: "{d}" }
            circle { class: "gradient-trace-anchor", cx: "{start.x}", cy: "{start.y}", r: "4" }
        }
    }
}
