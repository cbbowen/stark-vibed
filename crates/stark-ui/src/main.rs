//! Stark's Dioxus **web** frontend (DESIGN.md §11).
//!
//! The backend runs in WASM and paints through a WebGPU surface bound to the
//! page's `<canvas>` — the engine renders straight into the surface texture
//! after each command, with no GPU→CPU readback. The canvas fills the window;
//! unobtrusive floating panels (color, brush, layers) sit on top.
//!
//! Run with `dx serve --web -p stark-ui` in a WebGPU-capable browser.

mod components;
mod render;

use dioxus::html::input_data::MouseButton;
use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use components::menubar::{Menubar, MenubarContent, MenubarItem, MenubarMenu, MenubarTrigger};
use render::{Renderer, CANVAS_ID};
use stark_core::document::{BrushDynamics, BrushParams, BrushShape, MixerParams, Tool};
use stark_core::geom::Vec2;
use stark_core::{AssetId, ColorSpaceId, InputCommand, InputSample, LayerInfo, ObservableState};

/// Built-in brush shape, embedded so it's always available (DESIGN.md §6.6).
const BRISTLES_PNG: &[u8] = include_bytes!("../../../resources/shapes/WornBristles.png");

/// Saturation/value picker square size, in pixels.
const SV_W: f32 = 256.0;
const SV_H: f32 = 150.0;

/// The UI's global stylesheet — panel chrome (shared CSS custom properties) plus
/// every component class referenced below. Linked once by [`app`] so the rsx!
/// blocks carry class names, not inline styles. Custom properties are global, so
/// the css_module menubar styles pick up `--panel-shadow` / `--panel-noise` too.
static STARK_CSS: Asset = asset!("/assets/stark.css");

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    dioxus::launch(app);
}

/// Shared `Copy` handle to the app's signals.
#[derive(Clone, Copy)]
struct AppState {
    /// Surface + engine, built asynchronously once the canvas mounts. `None`
    /// until WebGPU init completes. Not `Send` — lives in unsync storage.
    renderer: Signal<Option<Renderer>>,
    /// UI-facing engine projection, refreshed after each command.
    obs: Signal<Option<ObservableState>>,
}

fn app() -> Element {
    let renderer = use_signal(|| None::<Renderer>);
    let obs = use_signal(|| None::<ObservableState>);
    let state = AppState { renderer, obs };
    use_context_provider(|| state);

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
        document::Stylesheet { href: STARK_CSS }

        div {
            class: "app-root",
            tabindex: "0",
            autofocus: true,
            onkeydown: move |e| handle_keys(state, &e),

            Canvas {}

            // Left command rail: rarely-used document commands, tucked away.
            CommandRail {}

            // Floating tool panels, stacked top-right.
            div { class: "panel-stack",
                ColorPanel {}
                BrushPanel {}
                LayerPanel {}
            }

            // Minimal, unobtrusive history controls (keyboard: Ctrl+Z / Ctrl+Shift+Z).
            HistoryControls {}
        }
    }
}

/// The full-window painting surface (a WebGPU canvas the engine draws into).
#[component]
fn Canvas() -> Element {
    let state = use_context::<AppState>();
    let mut drawing = use_signal(|| false);
    let mut pan_from = use_signal(|| None::<(f64, f64)>);

    rsx! {
        canvas {
            id: "{CANVAS_ID}",
            class: "paint-canvas",
            onresize: move |e| {
                if let Ok(size) = e.get_content_box_size() {
                    resize(state, size.width as u32, size.height as u32);
                }
            },
            onmousedown: move |e| {
                let (x, y) = elem_xy(&e);
                match e.trigger_button() {
                    Some(MouseButton::Primary) => {
                        dispatch(state, InputCommand::StartStroke { tool: Tool::Brush, sample: sample(state, x, y) });
                        drawing.set(true);
                    }
                    Some(MouseButton::Auxiliary) => {
                        e.prevent_default(); // suppress middle-click autoscroll
                        pan_from.set(Some((x, y)));
                    }
                    _ => {}
                }
            },
            onmousemove: move |e| {
                let (x, y) = elem_xy(&e);
                if drawing() {
                    dispatch(state, InputCommand::StrokeTo { sample: sample(state, x, y) });
                } else if let Some((lx, ly)) = pan_from() {
                    dispatch(state, InputCommand::Pan { delta: Vec2::new((x - lx) as f32, (y - ly) as f32) });
                    pan_from.set(Some((x, y)));
                }
            },
            onmouseup: move |_| end_interaction(state, &mut drawing, &mut pan_from),
            onmouseleave: move |_| end_interaction(state, &mut drawing, &mut pan_from),
            onwheel: move |e| {
                e.prevent_default();
                let dy = e.delta().strip_units().y;
                if dy != 0.0 {
                    let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
                    let c = e.element_coordinates();
                    dispatch(state, InputCommand::Zoom { anchor: Vec2::new(c.x as f32, c.y as f32), factor });
                }
            },
        }
    }
}

#[component]
fn ColorPanel() -> Element {
    let state = use_context::<AppState>();
    // Signals are `Copy`, so they can be handed to several event closures and to
    // the free helpers below.
    let hue = use_signal(|| 8.0_f32);
    let sat = use_signal(|| 0.8_f32);
    let val = use_signal(|| 0.6_f32);
    let mut picking = use_signal(|| false);

    let marker_x = sat() * SV_W;
    let marker_y = (1.0 - val()) * SV_H;

    rsx! {
        Panel { title: "Color",
            div {
                class: "sv-square",
                // Dynamic: the value/saturation field tinted by the current hue.
                style: "background: linear-gradient(to top, #000, rgba(0,0,0,0)), linear-gradient(to right, #fff, hsl({hue} 100% 50%));",
                onmousedown: move |e| { picking.set(true); pick_sv(state, hue, sat, val, &e); },
                onmousemove: move |e| { if picking() { pick_sv(state, hue, sat, val, &e); } },
                onmouseup: move |_| picking.set(false),
                onmouseleave: move |_| picking.set(false),
                div { class: "sv-marker", style: "left:{marker_x}px; top:{marker_y}px;" }
            }
            input {
                class: "color-hue",
                r#type: "range", min: "0", max: "360", value: "{hue()}",
                oninput: move |e| {
                    let mut hue = hue;
                    if let Ok(h) = e.value().parse::<f32>() {
                        hue.set(h);
                        apply_color(state, hue, sat, val);
                    }
                },
            }
        }
    }
}

/// Push the current (h, s, v) into the brush color, preserving its alpha.
fn apply_color(state: AppState, hue: Signal<f32>, sat: Signal<f32>, val: Signal<f32>) {
    update_brush(state, move |b| {
        let [r, g, b_] = hsv_to_srgb(hue(), sat(), val());
        b.color = [r, g, b_, b.color[3]];
    });
}

/// Set saturation/value from a pointer position over the S/V square, then apply.
fn pick_sv(
    state: AppState,
    hue: Signal<f32>,
    mut sat: Signal<f32>,
    mut val: Signal<f32>,
    e: &Event<MouseData>,
) {
    let c = e.element_coordinates();
    sat.set((c.x as f32 / SV_W).clamp(0.0, 1.0));
    val.set((1.0 - c.y as f32 / SV_H).clamp(0.0, 1.0));
    apply_color(state, hue, sat, val);
}

#[component]
fn BrushPanel() -> Element {
    let state = use_context::<AppState>();
    // Imported once, then cached for the session.
    let bristle_id = use_signal(|| None::<AssetId>);
    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();
    let is_round = matches!(brush.shape, BrushShape::Round);
    // Current wet-mixing params (or the defaults to show when switching on).
    let mixer = match brush.dynamics {
        BrushDynamics::Mixer(mp) => Some(mp),
        BrushDynamics::Dry => None,
    };
    let is_mixer = mixer.is_some();
    let mp = mixer.unwrap_or_default();

    let chip = |active: bool| if active { "chip active" } else { "chip" };

    rsx! {
        Panel { title: "Brush",
            div { class: "brush-shapes",
                button {
                    class: chip(is_round),
                    onclick: move |_| set_shape(state, BrushShape::Round, 0.25),
                    "Round"
                }
                button {
                    class: chip(!is_round),
                    onclick: move |_| set_bristles(state, bristle_id),
                    "Bristles"
                }
            }
            // Brush dynamics: dry paint vs. a mixer that smears wet canvas paint.
            div { class: "brush-shapes",
                button {
                    class: chip(!is_mixer),
                    onclick: move |_| set_dynamics(state, BrushDynamics::Dry),
                    "Dry"
                }
                button {
                    class: chip(is_mixer),
                    onclick: move |_| set_dynamics(state, BrushDynamics::Mixer(MixerParams::default())),
                    "Mixer"
                }
            }
            Slider { label: "Size", min: 1.0, max: 120.0, value: brush.radius,
                oninput: move |v| update_brush(state, move |b| b.radius = v) }
            Slider { label: "Opacity", min: 0.0, max: 1.0, value: brush.color[3],
                oninput: move |v| update_brush(state, move |b| b.color[3] = v) }
            Slider { label: "Rate", min: 0.05, max: 1.0, value: brush.flow,
                oninput: move |v| update_brush(state, move |b| b.flow = v) }
            // Mixer-only controls: how much canvas paint is lifted, and how fast
            // the stroke pulls back toward its own color (DESIGN.md §6.2).
            if is_mixer {
                Slider { label: "Pickup", min: 0.0, max: 1.0, value: mp.pickup,
                    oninput: move |v| set_mixer(state, move |m| m.pickup = v) }
                Slider { label: "Mix", min: 0.0, max: 1.0, value: mp.color_inject,
                    oninput: move |v| set_mixer(state, move |m| m.color_inject = v) }
            }
        }
    }
}

/// Switch to a shape, also setting a sensible default spacing for it.
fn set_shape(state: AppState, shape: BrushShape, spacing: f32) {
    update_brush(state, move |b| {
        b.shape = shape;
        b.spacing = spacing;
    });
}

/// Set the brush's canvas-pickup behavior (DESIGN.md §6.2).
fn set_dynamics(state: AppState, dynamics: BrushDynamics) {
    update_brush(state, move |b| b.dynamics = dynamics);
}

/// Edit the mixer reservoir params in place (no-op if the brush isn't a Mixer).
fn set_mixer(state: AppState, f: impl FnOnce(&mut MixerParams)) {
    update_brush(state, move |b| {
        if let BrushDynamics::Mixer(mp) = &mut b.dynamics {
            f(mp);
        }
    });
}

/// Select the bristle brush, importing it on first use.
fn set_bristles(mut state: AppState, mut bristle_id: Signal<Option<AssetId>>) {
    let id = match bristle_id() {
        Some(id) => id,
        None => {
            let mut guard = state.renderer.write();
            let Some(r) = guard.as_mut() else { return };
            match r.import_brush(BRISTLES_PNG) {
                Ok(id) => {
                    bristle_id.set(Some(id));
                    id
                }
                Err(_) => return,
            }
        }
    };
    set_shape(state, BrushShape::Stamp(id), 0.08);
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
        Panel { title: "Layers",
            div { class: "layer-header",
                button {
                    class: "layer-add",
                    onclick: move |_| dispatch(state, InputCommand::AddLayer { above: None }),
                    "+ Add"
                }
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
    let row_class = if active { "layer-row active" } else { "layer-row" };

    rsx! {
        div {
            class: row_class,
            div { class: "row",
                input {
                    r#type: "checkbox",
                    checked: info.visible,
                    onchange: move |_| dispatch(state, InputCommand::SetLayerVisible(info.id, !info.visible)),
                }
                button {
                    class: "layer-name",
                    onclick: move |_| dispatch(state, InputCommand::SetActiveLayer(info.id)),
                    "Layer {info.id.0}"
                }
            }
            input {
                class: "slider",
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

#[component]
fn HistoryControls() -> Element {
    let state = use_context::<AppState>();
    let (can_undo, can_redo) = state
        .obs
        .read()
        .as_ref()
        .map(|o| (o.can_undo, o.can_redo))
        .unwrap_or((false, false));

    rsx! {
        div { class: "history-controls",
            button { class: "chrome-button", disabled: !can_undo, onclick: move |_| dispatch(state, InputCommand::Undo), "⟲ Undo" }
            button { class: "chrome-button", disabled: !can_redo, onclick: move |_| dispatch(state, InputCommand::Redo), "Redo ⟳" }
        }
    }
}

/// A vertical menu rail on the far left for uncommon, document-level commands
/// (DESIGN.md §11). Built on the vendored `menubar` component; the dropdown flies
/// out to the right. Currently just "New document…", which opens a settings modal.
#[component]
fn CommandRail() -> Element {
    let mut show_new_doc = use_signal(|| false);

    rsx! {
        div { class: "command-rail",
            Menubar {
                MenubarMenu { index: 0usize,
                    // ☰ — the catch-all menu for infrequent commands.
                    MenubarTrigger { "\u{2630}" }
                    MenubarContent {
                        MenubarItem {
                            index: 0usize,
                            value: "new-document".to_string(),
                            on_select: move |_| show_new_doc.set(true),
                            "New document…"
                        }
                    }
                }
            }
        }
        if show_new_doc() {
            NewDocumentModal { on_close: move |_| show_new_doc.set(false) }
        }
    }
}

/// Modal for starting a fresh document. Today it carries the color-space choice
/// (DESIGN.md §6.7); it's a dialog so more document settings can join it later.
#[component]
fn NewDocumentModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let current = state
        .renderer
        .read()
        .as_ref()
        .map(|r| r.color_space())
        .unwrap_or(ColorSpaceId::Oklab);
    let choice = use_signal(|| current);

    // One selectable color-space card; `selected` toggles the highlight.
    let card = |id: ColorSpaceId, title: &str, desc: &str| {
        let class = if choice() == id { "space-card selected" } else { "space-card" };
        rsx! {
            div {
                class,
                onclick: move |_| { let mut choice = choice; choice.set(id); },
                div { class: "space-card-title", "{title}" }
                div { class: "space-card-desc", "{desc}" }
            }
        }
    };

    rsx! {
        // Dimmed backdrop; click outside the dialog to dismiss.
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "New Document" }
                div { class: "modal-subtitle", "Starting a new document replaces the current canvas." }

                div { class: "modal-section-label", "COLOR SPACE" }
                {card(ColorSpaceId::Oklab, "Oklab", "Perceptual color with smooth, predictable blending. The standard choice for digital painting.")}
                {card(ColorSpaceId::Mixbox, "Mixbox", "Realistic pigment mixing (Mixbox): blue + yellow makes green, like real paint. For natural media.")}

                div { class: "modal-actions",
                    button {
                        class: "btn btn-secondary",
                        onclick: move |_| on_close.call(()),
                        "Cancel"
                    }
                    button {
                        class: "btn btn-primary",
                        onclick: move |_| { new_document(state, choice()); on_close.call(()); },
                        "Create"
                    }
                }
            }
        }
    }
}

/// Replace the document with a fresh one in `id`'s color space, then repaint.
fn new_document(state: AppState, id: ColorSpaceId) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.set_color_space(id);
        r.paint();
        obs.set(Some(r.observe()));
    }
}

// --- reusable chrome ---

#[component]
fn Panel(title: String, children: Element) -> Element {
    rsx! {
        div { class: "panel",
            div { class: "panel-title", "{title}" }
            {children}
        }
    }
}

#[component]
fn Slider(label: String, min: f32, max: f32, value: f32, oninput: EventHandler<f32>) -> Element {
    rsx! {
        div { class: "slider-row",
            div { class: "slider-label", "{label}" }
            input {
                class: "slider",
                r#type: "range", min: "{min}", max: "{max}", step: "any", value: "{value}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() { oninput.call(v); }
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

/// Resize the surface/engine, then repaint.
fn resize(state: AppState, width: u32, height: u32) {
    let mut renderer = state.renderer;
    let mut obs = state.obs;
    let mut guard = renderer.write();
    if let Some(r) = guard.as_mut() {
        r.resize(width, height);
        r.paint();
        obs.set(Some(r.observe()));
    }
}

/// Read the current brush, mutate a copy, and commit it (releasing the `obs`
/// read guard before `dispatch` writes — avoids an AlreadyBorrowed panic).
fn update_brush(state: AppState, f: impl FnOnce(&mut BrushParams)) {
    let brush = state.obs.read().as_ref().map(|o| o.brush);
    if let Some(mut brush) = brush {
        f(&mut brush);
        dispatch(state, InputCommand::SetBrush(brush));
    }
}

fn handle_keys(state: AppState, e: &Event<KeyboardData>) {
    let m = e.modifiers();
    if !(m.contains(Modifiers::CONTROL) || m.contains(Modifiers::META)) {
        return;
    }
    match e.key() {
        Key::Character(c) if c.eq_ignore_ascii_case("z") => {
            let cmd = if m.contains(Modifiers::SHIFT) {
                InputCommand::Redo
            } else {
                InputCommand::Undo
            };
            dispatch(state, cmd);
        }
        Key::Character(c) if c.eq_ignore_ascii_case("y") => dispatch(state, InputCommand::Redo),
        _ => {}
    }
}

/// Pointer position within an element, in CSS pixels.
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

/// End any in-progress stroke or pan.
fn end_interaction(
    state: AppState,
    drawing: &mut Signal<bool>,
    pan_from: &mut Signal<Option<(f64, f64)>>,
) {
    if drawing() {
        dispatch(state, InputCommand::EndStroke);
        drawing.set(false);
    }
    pan_from.set(None);
}

/// HSV (h in degrees, s/v in [0,1]) → straight sRGB RGB in [0,1].
fn hsv_to_srgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let h6 = (h / 60.0).rem_euclid(6.0);
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}
