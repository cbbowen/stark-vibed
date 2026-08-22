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

use dioxus::html::Key;
use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::commands::Command;
use crate::gradients;
use crate::icons::{self, icon, label};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_class;
use crate::platform::{capture_pointer, select_all};
use crate::state::{AppState, use_obs};
use crate::widgets::CommandButton;
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
/// closes it (the catcher wants the canvas the pop-out is floating over), and
/// under the gradient bar the mode's exclusivity takes the whole bar down with
/// it — but that bar is *set aside*, not abandoned
/// (`gradient_bar::suspend`), and stands back up when the trace ends, wearing
/// the ramp the trace just captured.
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
                            // over. Closed, not remembered: the trace ends with
                            // its capture already in hand on the strip, so
                            // reopening the list over the fresh preview would
                            // cover the answer with the question.
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
                            GradientRow {
                                key: "{entry.name}",
                                active: current.as_deref() == Some(entry.name.as_str()),
                                entry,
                            }
                        }
                    }
                }
            }
        }
    }
}

/// One library row: the ramp filling it edge to edge — the preset rows' recipe
/// (`.preset-row`): the visual is the star, so it gets every pixel the row
/// has, and the name floats over it lifted by shadow, with the trash overlaid
/// on the far end, hover-revealed. Double-click opens the rename field, the
/// layer and guide rows' interaction language.
///
/// A component per row because the rename draft is row-local state: opening
/// one leaves every other row alone, and closing it needs nothing cleaned up.
/// The draft is held here rather than read back off the field on commit
/// because both commit paths — Enter and blur — need it, and one of them
/// fires while the field is on its way out (the layer row's argument, whole).
#[component]
fn GradientRow(entry: gradients::GradientEntry, active: bool) -> Element {
    let state = use_context::<AppState>();
    let mut draft = use_signal(|| None::<String>);
    let strip = gradients::css_strip(&entry.gradient);
    let name = entry.name.clone();
    let select_name = entry.name.clone();
    let remove_name = entry.name.clone();
    let seed = entry.name.clone();
    let rename_from = entry.name;
    // Commit whatever the field holds, and close it. `take` is what makes the
    // two commit paths safe to both fire: whichever runs second finds no draft.
    // A collision or an untouched field is the library's to refuse
    // (`gradients::rename`), so no name is lost to a stray blur.
    let mut commit = move || {
        let text = draft.write().take();
        if let Some(text) = text {
            gradients::rename(state, &rename_from, &text);
        }
    };
    // Cloned for the second handler: unlike the layer row's, this closure
    // holds a `String` (the name the library is asked to rename *from*), so it
    // is not `Copy` — the clones share the one draft signal, and `take` above
    // keeps the pair single-fire either way.
    let mut commit_on_blur = commit.clone();
    rsx! {
        div {
            class: if active { "gradient-row active" } else { "gradient-row" },
            style: "background-image: {strip};",
            // Clicking takes the ramp in hand — and re-previews a composing
            // fill, so mid-mode the canvas answers the click.
            onclick: move |_| gradients::select(state, &select_name),
            ondoubleclick: move |_| draft.set(Some(seed.clone())),
            if let Some(text) = draft() {
                input {
                    class: "gradient-row-name gradient-rename",
                    r#type: "text",
                    value: "{text}",
                    // The field is the point of the double-click, so it takes
                    // focus as it appears — and selected, since the usual
                    // reason to open it is to replace the machinery's
                    // "Gradient N" rather than add to it (the layer row's
                    // ordering argument for awaiting the focus first).
                    onmounted: move |e: Event<MountedData>| {
                        spawn(async move {
                            let _ = e.set_focus(true).await;
                            select_all(&e);
                        });
                    },
                    oninput: move |e| draft.set(Some(e.value())),
                    // A click placing the caret is the field's, not the row's:
                    // bubbled up it would re-select the ramp under the edit —
                    // harmless for a fill, one committed `SetFilter` for a map.
                    onclick: move |e| e.stop_propagation(),
                    ondoubleclick: move |e| e.stop_propagation(),
                    // Blur commits (clicking away is an ordinary way to be
                    // finished); Enter commits directly rather than by
                    // blurring, since a removed field does not reliably fire
                    // `blur`. Escape abandons, dropping the draft first so the
                    // blur that follows has nothing left to send.
                    onblur: move |_| commit_on_blur(),
                    onkeydown: move |e| match e.key() {
                        Key::Enter => commit(),
                        Key::Escape => draft.set(None),
                        _ => {}
                    },
                }
            } else {
                span { class: "gradient-row-name", title: "{name}", "{name}" }
            }
            // The same trash the preset, layer and guide rows wear: a fourth
            // roster, same act, same mark — overlaid on the ramp and lifted
            // off it by shadow, as the preset rows' is off their stroke.
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

/// The trace mode's own bar, in the shared bottom column (MODAL_DESIGN.md).
///
/// The armed chip in the pop-out used to be the mode's only standing indicator
/// — and it closes with the pop-out the moment the catcher takes the canvas.
/// This names the mode where every other mode is named, and carries the one
/// act a trace can offer: Cancel, which pops back to whatever the trace parked
/// (`crate::modes::cancel`). There is no Done, because the release *is* the
/// capture (§22.2) — a chip that committed nothing would wear commitment's
/// tick.
#[component]
pub fn TraceBar() -> Element {
    let state = use_context::<AppState>();
    rsx! {
        if (state.gradients.armed)() {
            div { class: chrome_class(state, "selection-bar trace-bar mode-bar"),
                span { class: "bar-label",
                    {icon(icons::GRADIENT)}
                    {label("Trace")}
                }
                span { class: "bar-sep" }
                CommandButton { command: Command::CancelMode }
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
    // The view through a memo, unconditionally and ahead of the early returns
    // below like any `use_*`. Not a straight read of the projection, which is
    // what this was: that woke the overlay on every engine write rather than on
    // the one field it draws with (`state::use_obs`).
    let live_view = use_obs(state, |o| o.view);

    if !(state.gradients.armed)() {
        return rsx! {};
    }
    let Some(view) = live_view() else {
        return rsx! {};
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
        // The capture goes first, and the order is load-bearing: it samples the
        // **composite** at the instant it is called (§22.2), while ending the
        // mode stands the suspended gradient bar back up *with its preview*
        // (`gradient_bar::resume`). The other way round, a fill preview would be
        // traced as if it were paint and the new ramp fitted through the one it
        // was drawn to replace.
        if gradients::trace_long_enough(&points) {
            gradients::capture(state, points);
        }
        // Release always ends the mode: a good trace captures, a stray click
        // cancels — either way the canvas is handed back, to the bar the trace
        // was armed from if there was one.
        gradients::set_armed(state, false);
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
