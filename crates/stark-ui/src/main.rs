//! Stark's Dioxus **web** frontend (DESIGN.md §11).
//!
//! The backend runs in WASM and paints through a WebGPU surface bound to the
//! page's `<canvas>` — the engine renders straight into the surface texture
//! after each command, with no GPU→CPU readback. DOM chrome (toolbar + layer
//! panel) dispatches [`InputCommand`]s and renders reactively from the engine's
//! `ObservableState`.
//!
//! Run with `dx serve --platform web` in a WebGPU-capable browser.

mod render;

use dioxus::prelude::*;

use render::{Renderer, CANVAS_ID, H, W};
use stark_core::document::Tool;
use stark_core::geom::Vec2;
use stark_core::{InputCommand, InputSample, LayerInfo, ObservableState};

/// Preset brush colors (label, straight sRGB RGBA).
const COLORS: [(&str, [f32; 4]); 6] = [
    ("Ink", [0.05, 0.05, 0.08, 1.0]),
    ("Crimson", [0.78, 0.12, 0.15, 1.0]),
    ("Ochre", [0.80, 0.55, 0.15, 1.0]),
    ("Viridian", [0.10, 0.55, 0.38, 1.0]),
    ("Ultramarine", [0.18, 0.22, 0.62, 1.0]),
    ("Titanium", [0.95, 0.95, 0.95, 1.0]),
];

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    dioxus::launch(app);
}

/// Shared `Copy` handle to the app's signals.
#[derive(Clone, Copy)]
struct AppState {
    /// The surface + engine, built asynchronously once the canvas is mounted.
    /// `None` until WebGPU init completes. Not `Send` — lives in unsync storage.
    renderer: Signal<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command.
    obs: Signal<Option<ObservableState>>,
}

fn app() -> Element {
    let renderer = use_signal(|| None::<Renderer>);
    let obs = use_signal(|| None::<ObservableState>);
    let state = AppState { renderer, obs };
    use_context_provider(|| state);

    // After first render the canvas exists; build the WebGPU surface + engine.
    use_hook(|| {
        let mut renderer = renderer;
        let mut obs = obs;
        spawn(async move {
            let mut r = render::init(render::canvas_element()).await;
            r.paint();
            obs.set(Some(r.observe()));
            renderer.set(Some(r));
        });
    });

    rsx! {
        div {
            style: "display:flex; flex-direction:column; height:100vh; margin:0; font-family:system-ui,sans-serif; background:#1b1b1f; color:#e8e8ea;",
            Toolbar {}
            div { style: "display:flex; flex:1; min-height:0;",
                CanvasView {}
                LayerPanel {}
            }
        }
    }
}

#[component]
fn Toolbar() -> Element {
    let state = use_context::<AppState>();
    let obs = state.obs.read().clone();
    let (can_undo, can_redo, radius) = match &obs {
        Some(o) => (o.can_undo, o.can_redo, o.brush.radius),
        None => (false, false, 16.0),
    };

    rsx! {
        div {
            style: "display:flex; gap:8px; align-items:center; padding:8px 12px; background:#26262b; border-bottom:1px solid #000;",
            for (name, col) in COLORS {
                button {
                    style: "padding:4px 8px; border:1px solid #444; border-radius:4px; background:#33333a; color:#eee; cursor:pointer;",
                    onclick: move |_| set_color(state, col),
                    "{name}"
                }
            }
            span { style: "margin-left:8px;", "Size" }
            input {
                r#type: "range", min: "1", max: "80", value: "{radius}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        set_radius(state, v);
                    }
                },
            }
            div { style: "flex:1;" }
            button { disabled: !can_undo, onclick: move |_| dispatch(state, InputCommand::Undo), "Undo" }
            button { disabled: !can_redo, onclick: move |_| dispatch(state, InputCommand::Redo), "Redo" }
        }
    }
}

#[component]
fn CanvasView() -> Element {
    let state = use_context::<AppState>();
    let mut drawing = use_signal(|| false);

    rsx! {
        div {
            style: "flex:1; display:flex; align-items:center; justify-content:center; background:#0e0e10; overflow:auto;",
            canvas {
                id: "{CANVAS_ID}",
                width: "{W}",
                height: "{H}",
                style: "background:#fff; box-shadow:0 2px 16px #000; cursor:crosshair; touch-action:none;",
                onmousedown: move |e| {
                    let (x, y) = elem_xy(&e);
                    dispatch(state, InputCommand::StartStroke { tool: Tool::Brush, sample: sample(state, x, y) });
                    drawing.set(true);
                },
                onmousemove: move |e| {
                    if drawing() {
                        let (x, y) = elem_xy(&e);
                        dispatch(state, InputCommand::StrokeTo { sample: sample(state, x, y) });
                    }
                },
                onmouseup: move |_| if drawing() {
                    dispatch(state, InputCommand::EndStroke);
                    drawing.set(false);
                },
                onmouseleave: move |_| if drawing() {
                    dispatch(state, InputCommand::EndStroke);
                    drawing.set(false);
                },
            }
        }
    }
}

#[component]
fn LayerPanel() -> Element {
    let state = use_context::<AppState>();
    let layers = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.layers.clone())
        .unwrap_or_default();

    rsx! {
        div {
            style: "width:180px; flex:none; padding:10px; background:#26262b; border-left:1px solid #000; overflow:auto;",
            div { style: "display:flex; justify-content:space-between; align-items:center;",
                strong { "Layers" }
                button { onclick: move |_| dispatch(state, InputCommand::AddLayer { above: None }), "+" }
            }
            for info in layers.iter().rev().cloned() {
                LayerRow { info }
            }
        }
    }
}

#[component]
fn LayerRow(info: LayerInfo) -> Element {
    let state = use_context::<AppState>();
    let active = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.active_layer == info.id)
        .unwrap_or(false);
    let bg = if active { "#2f6f4f" } else { "#33333a" };

    rsx! {
        div {
            style: "margin:6px 0; padding:6px; border-radius:4px; background:{bg};",
            div { style: "display:flex; gap:6px; align-items:center;",
                input {
                    r#type: "checkbox",
                    checked: info.visible,
                    onchange: move |_| dispatch(state, InputCommand::SetLayerVisible(info.id, !info.visible)),
                }
                button {
                    style: "flex:1; text-align:left; background:none; border:none; color:#eee; cursor:pointer;",
                    onclick: move |_| dispatch(state, InputCommand::SetActiveLayer(info.id)),
                    "Layer {info.id.0}"
                }
            }
            input {
                style: "width:100%;",
                r#type: "range", min: "0", max: "100",
                value: "{(info.opacity * 100.0) as i32}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        dispatch(state, InputCommand::SetLayerOpacity(info.id, v / 100.0));
                    }
                },
            }
        }
    }
}

// --- command dispatch ---

/// Apply a command, repaint the surface, and refresh the observable snapshot.
fn dispatch(state: AppState, command: InputCommand) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.process(command);
        r.paint();
        obs.set(Some(r.observe()));
    }
}

fn set_color(state: AppState, color: [f32; 4]) {
    // Read into a local so the signal's read guard is released before `dispatch`
    // (which writes `obs`) — otherwise the live borrow panics (AlreadyBorrowed).
    let brush = state.obs.read().as_ref().map(|o| o.brush);
    if let Some(mut brush) = brush {
        brush.color = color;
        dispatch(state, InputCommand::SetBrush(brush));
    }
}

fn set_radius(state: AppState, radius: f32) {
    let brush = state.obs.read().as_ref().map(|o| o.brush);
    if let Some(mut brush) = brush {
        brush.radius = radius;
        dispatch(state, InputCommand::SetBrush(brush));
    }
}

/// Pointer position within the canvas element, in CSS/canvas pixels.
fn elem_xy(e: &Event<MouseData>) -> (f64, f64) {
    let c = e.element_coordinates();
    (c.x, c.y)
}

/// Map an element-relative pointer position to a canvas-space input sample.
fn sample(state: AppState, x: f64, y: f64) -> InputSample {
    let view = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.view())
        .expect("renderer ready during input");
    InputSample::at(view.screen_to_canvas(Vec2::new(x as f32, y as f32)))
}
