//! The gradient-fill mode (§22.4): the Selection bar's Gradient button.
//!
//! The transform mode's shape, aimed at a fill. Entering swaps the Selection
//! bar for [`GradientFillBar`] (Linear/Radial, the ramp in hand, Done) and
//! mounts a full-viewport catcher; **the drag is the axis** — press where the
//! ramp starts, release where it ends, radial reads the same drag as centre
//! and reach. Every mutation funnels through one [`update`], which previews the
//! real fill through `ViewCommand::PreviewFill` — the same `FillRenderer` the
//! commit runs, so what is shown is what "Done" produces, and re-dragging
//! replaces the preview rather than stacking glazes.
//!
//! The fill itself is `FillOp::gradient_of_selection`: bounded by the mask
//! alone, like the bar's solid Fill — the bar only exists while a selection is
//! in force, which is what keeps the unbounded refusal out of sight.

use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::gradients;
use crate::icons::{self, icon, label};
use crate::input::{Nav, page_xy};
use crate::layout::chrome_class;
use crate::platform::capture_pointer;
use crate::state::{AppState, GradientAxisKind, GradientFillUi, dispatch};
use stark_core::command::{DocCommand, ViewCommand};
use stark_core::document::{FillOp, GradientParcel};
use stark_core::geom::Vec2;

/// Enter the mode. The target layer is the transform's choice — the active
/// layer if paintable, else the topmost paintable — and the brush's opacity
/// and amount are captured now (§6.8's shape-drag bargain). Refuses quietly
/// when the library is empty: the bar's button is disabled then, and says why.
pub fn begin(state: AppState) {
    if gradients::current(state).is_none() {
        return;
    }
    let obs = state.obs.read();
    let Some(o) = obs.as_ref() else { return };
    let layer = o
        .layers
        .iter()
        .find(|l| l.id == o.active_layer && l.is_paintable())
        .or_else(|| o.layers.iter().rev().find(|l| l.is_paintable()))
        .map(|l| l.id);
    let Some(layer) = layer else { return };
    let (opacity, height) = (o.brush.color[3], o.brush.dynamics.add);
    drop(obs);
    let mut mode = state.gradient_fill;
    mode.set(Some(GradientFillUi {
        layer,
        kind: GradientAxisKind::Linear,
        drag: None,
        opacity,
        height,
    }));
}

/// Update the gesture and show its consequence — every mutation funnels through
/// here, so the preview can never lag the state (the transform's rule).
fn update(state: AppState, ui: GradientFillUi) {
    let mut mode = state.gradient_fill;
    mode.set(Some(ui));
    dispatch(
        state,
        ViewCommand::PreviewFill(op_of(state, &ui).map(|op| (ui.layer, op))),
    );
}

/// Re-preview the composing fill, if one is composing — how a gradient picked
/// in the panel mid-mode reaches the canvas.
pub fn refresh(state: AppState) {
    let ui = *state.gradient_fill.peek();
    if let Some(ui) = ui {
        update(state, ui);
    }
}

/// The `FillOp` the current gesture would commit: the selected ramp along the
/// dragged axis, at the entry-time opacity and amount. `None` before the first
/// drag or with the library emptied mid-mode.
fn op_of(state: AppState, ui: &GradientFillUi) -> Option<FillOp> {
    let axis = ui.axis()?;
    let gradient = gradients::current(state)?;
    Some(FillOp::gradient_of_selection(
        GradientParcel {
            gradient,
            axis,
            opacity: ui.opacity,
        },
        ui.height,
    ))
}

/// Leave the mode: commit the composed fill, or just drop the preview when
/// nothing was dragged. The commit clears the preview itself, so there is no
/// intermediate frame showing the unfilled document (the transform's Done).
fn finish(state: AppState) {
    let ui = *state.gradient_fill.peek();
    match ui.and_then(|ui| op_of(state, &ui).map(|op| (ui.layer, op))) {
        Some((layer, op)) => dispatch(state, DocCommand::Fill { layer, op }),
        None => dispatch(state, ViewCommand::PreviewFill(None)),
    }
    let mut mode = state.gradient_fill;
    mode.set(None);
}

/// The mode's bar, standing in for the Selection bar while composing.
#[component]
pub fn GradientFillBar() -> Element {
    let state = use_context::<AppState>();
    let Some(ui) = *state.gradient_fill.read() else {
        return rsx! {};
    };
    // The ramp in hand, read reactively so a pick in the Gradients panel
    // repaints the bar's strip with the preview.
    let strip = gradients::current(state).map(|g| gradients::css_strip(&g));
    let kind_chip = |kind: GradientAxisKind, glyph: &'static str, name: &'static str| {
        let active = ui.kind == kind;
        rsx! {
            button {
                class: if active { "chip active" } else { "chip" },
                title: match kind {
                    GradientAxisKind::Linear => "The drag is the ramp: press at its start, release at its end",
                    GradientAxisKind::Radial => "The drag is the reach: press at the centre, release at the rim",
                },
                onclick: move |_| {
                    // Reinterpret the drag already made rather than losing it —
                    // `GradientFillUi::axis` reads the same two points either way.
                    update(state, GradientFillUi { kind, ..ui });
                },
                {icon(glyph)}
                {label(name)}
            }
        }
    };

    rsx! {
        div { class: chrome_class(state, "selection-bar gradient-fill-bar"),
            // The Gradients panel's mark: the bar is that library's ramp being
            // put to work, so it wears the library's glyph.
            span { class: "bar-label",
                {icon(icons::GRADIENT)}
                {label("Gradient")}
            }
            if let Some(strip) = strip {
                span {
                    class: "bar-gradient-strip",
                    title: "The ramp being laid \u{2014} pick another in the Gradients panel",
                    style: "background: {strip};",
                }
            }
            span { class: "bar-sep" }
            {kind_chip(GradientAxisKind::Linear, icons::GRADIENT_LINEAR, "Linear")}
            {kind_chip(GradientAxisKind::Radial, icons::GRADIENT_RADIAL, "Radial")}
            span { class: "bar-sep" }
            {
                // Hoisted out of the rsx! attribute: an `if`/`else` inside one
                // trips clippy's suspicious-else-formatting on the expansion.
                let done_title = if ui.drag.is_some() {
                    "Lay the gradient (undo takes it back)"
                } else {
                    "Nothing dragged yet \u{2014} leave without filling"
                };
                rsx! {
                    button {
                        class: "chip",
                        title: done_title,
                        onclick: move |_| finish(state),
                        {icon(icons::DONE)}
                        {label("Done")}
                    }
                }
            }
        }
    }
}

/// The mode's catcher and axis chrome: while composing, the pointer draws the
/// axis, it does not paint — the transform catcher's bargain, with `Nav` live
/// so the view stays reachable mid-compose.
#[component]
pub fn GradientFillOverlay() -> Element {
    let state = use_context::<AppState>();
    let mut dragging = use_signal(|| false);
    let nav = Nav::use_nav(state);

    let Some(ui) = *state.gradient_fill.read() else {
        return rsx! {};
    };
    let view = match state.obs.read().as_ref() {
        Some(o) => o.view,
        None => return rsx! {},
    };
    let to_canvas = move |e: &Event<PointerData>| view.screen_to_canvas(page_xy(e));

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
                    // Navigation takes the press; the composed axis stands — it
                    // is canvas-space, so the view moving under it costs nothing.
                    dragging.set(false);
                    return;
                }
                if e.trigger_button() != Some(MouseButton::Primary) {
                    return;
                }
                e.stop_propagation();
                capture_pointer(&e);
                dragging.set(true);
                let p = to_canvas(&e);
                update(state, GradientFillUi { drag: Some((p, p)), ..ui });
            },
            onpointermove: move |e| {
                if nav.advance(&e) || !dragging() {
                    return;
                }
                if let Some((from, _)) = ui.drag {
                    update(state, GradientFillUi { drag: Some((from, to_canvas(&e))), ..ui });
                }
            },
            onpointerup: move |e| if !nav.release(&e) { nav.stop(); dragging.set(false); },
            onpointercancel: move |e| if !nav.release(&e) { nav.stop(); dragging.set(false); },
            onwheel: move |e| nav.wheel(e),
        }

        if let Some((from, to)) = ui.drag {
            {axis_chrome(ui.kind, from, to, view)}
        }
    }
}

/// The axis, drawn in the trace's own visual language (dark casing, dashed
/// light core, a dot on the anchor): linear is the line the ramp runs along,
/// radial is the reach circle with its centre marked. A plain function chosen
/// by an `if`, like the transform's visuals.
fn axis_chrome(
    kind: GradientAxisKind,
    from: Vec2,
    to: Vec2,
    view: stark_core::ViewTransform,
) -> Element {
    let a = view.canvas_to_screen(from);
    let b = view.canvas_to_screen(to);
    match kind {
        GradientAxisKind::Linear => {
            let d = format!("M{:.2} {:.2} L{:.2} {:.2}", a.x, a.y, b.x, b.y);
            rsx! {
                svg { class: "gradient-trace-svg",
                    path { class: "gradient-trace-casing", d: "{d}" }
                    path { class: "gradient-trace-core", d: "{d}" }
                    circle { class: "gradient-trace-anchor", cx: "{a.x}", cy: "{a.y}", r: "4" }
                    circle { class: "gradient-trace-anchor", cx: "{b.x}", cy: "{b.y}", r: "3" }
                }
            }
        }
        GradientAxisKind::Radial => {
            // The circle is drawn about the centre with the *screen* radius; a
            // rotated or mirrored view moves the rim point, never the circle.
            let r = from.distance(to) * view.zoom;
            let d = format!("M{:.2} {:.2} L{:.2} {:.2}", a.x, a.y, b.x, b.y);
            rsx! {
                svg { class: "gradient-trace-svg",
                    circle { class: "gradient-axis-circle", cx: "{a.x}", cy: "{a.y}", r: "{r}" }
                    path { class: "gradient-trace-core", d: "{d}" }
                    circle { class: "gradient-trace-anchor", cx: "{a.x}", cy: "{a.y}", r: "4" }
                }
            }
        }
    }
}
