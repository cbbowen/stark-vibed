//! The floating Brush panel: the everyday quick controls. The full parameter set
//! lives in the brush editor dialog.

use dioxus::prelude::*;

use crate::presets;
use crate::state::{AppState, update_brush};
use crate::widgets::Slider;
use stark_core::document::{BrushShape, OrientationSource};

/// Built-in assets, bundled as static files and **fetched at runtime** so they
/// stay out of the wasm binary (DESIGN.md §6.6). The engine is handed the bytes.
pub const BRISTLE_BRUSH: Asset = asset!("/assets/shape/WornBristles.png");

/// The maximum brush radius (`BrushParams::radius`).
pub const MAX_RADIUS: f32 = 500.0;

/// The longest taper the editor offers, in brush radii
/// (`BrushParams::start_taper_length`). Twenty radii is ten stroke widths of run-in
/// — a dramatic inker's entry, and well past where a longer one reads as different
/// rather than more.
pub const MAX_TAPER: f32 = 20.0;

/// The floating Brush panel: the everyday quick controls (size, amount).
/// Everything else — the full grouped parameter set with a live test
/// stroke — lives in the brush editor dialog ("Edit brush…").
#[component]
pub fn BrushPanel() -> Element {
    let state = use_context::<AppState>();
    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();

    rsx! {
        Slider { label: "Size", min: 1.0, max: MAX_RADIUS, value: brush.radius,
            oninput: move |v| update_brush(state, move |b| b.radius = v) }
        Slider { label: "Amount", min: 0.0, max: 1.5, value: brush.dynamics.add,
            oninput: move |v| update_brush(state, move |b| b.dynamics.add = v) }
        button {
            class: "be-open",
            onclick: move |_| {
                let mut open = state.brush_editor_open;
                open.set(true);
            },
            "Edit brush\u{2026}"
        }
        PresetSection {}
    }
}

/// The preset library (`crate::presets`), folded away at the panel's foot: one
/// row per preset — click applies it, hover reveals a remove ✕ — and a save row
/// that snapshots the live brush under a chosen name (or the next free
/// "Preset N" when left blank). The row whose snapshot the live brush still *is*
/// (colour aside) is highlighted; it goes out the moment any knob moves.
///
/// Safe as a child component: nothing here spawns, so there is no task to die
/// with a fold (unlike the editor's slider rows — see `brush_editor::edit`).
#[component]
fn PresetSection() -> Element {
    let state = use_context::<AppState>();
    let mut open = use_signal(|| true);
    let mut draft = use_signal(String::new);
    let entries = (state.presets)();
    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();

    let mut save = move || {
        let name = draft().trim().to_string();
        let name = if name.is_empty() {
            presets::next_name(&state.presets.read())
        } else {
            name
        };
        presets::save_current(state, name);
        draft.set(String::new());
    };

    rsx! {
        div { class: "preset-section",
            button {
                class: "be-section-header",
                onclick: move |_| { let v = open(); open.set(!v); },
                span { class: if open() { "be-chevron open" } else { "be-chevron" }, "\u{25B8}" }
                "Presets"
            }
            if open() {
                div { class: "preset-list",
                    for entry in entries {
                        {
                            let apply_name = entry.name.clone();
                            let remove_name = entry.name.clone();
                            let active = presets::matches(&brush, &entry.brush);
                            rsx! {
                                div {
                                    key: "{entry.name}",
                                    class: if active { "preset-row active" } else { "preset-row" },
                                    onclick: move |_| presets::apply(state, &apply_name),
                                    span { class: "preset-row-name", title: "{entry.name}", "{entry.name}" }
                                    button {
                                        class: "preset-remove",
                                        title: "Remove preset",
                                        onclick: move |e| {
                                            e.stop_propagation();
                                            presets::remove(state, &remove_name);
                                        },
                                        "\u{00D7}"
                                    }
                                }
                            }
                        }
                    }
                }
                div { class: "preset-save",
                    input {
                        class: "preset-name",
                        r#type: "text",
                        placeholder: "Save current brush as\u{2026}",
                        value: "{draft}",
                        oninput: move |e| draft.set(e.value()),
                        onkeydown: move |e| {
                            if e.key() == Key::Enter {
                                save();
                            }
                        },
                    }
                    button { class: "chip", onclick: move |_| save(), "Save" }
                }
            }
        }
    }
}

/// Switch to a shape, also setting a sensible default spacing for it.
pub fn set_shape(state: AppState, shape: BrushShape) {
    update_brush(state, move |b| b.shape = shape);
}

/// Set what orients the brush shape as it sweeps (DESIGN.md §6.6).
pub fn set_orientation(state: AppState, orientation: OrientationSource) {
    update_brush(state, move |b| b.orientation = orientation);
}

/// Select the built-in bristle brush. It's fetched + imported once at startup
/// (DESIGN.md §6.6), so this is a no-op until those bytes have loaded.
pub fn set_bristles(state: AppState) {
    let id = state.renderer.read().as_ref().and_then(|r| r.bristle());
    let Some(id) = id else { return };
    set_shape(state, BrushShape::Stamp(id));
}
