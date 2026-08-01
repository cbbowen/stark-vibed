//! The floating Brush panel: the everyday quick controls. The full parameter set
//! lives in the brush editor dialog.

use dioxus::prelude::*;

use crate::platform::select_all;
use crate::presets;
use crate::state::{AppState, update_brush};
use crate::widgets::Slider;
use stark_core::document::{BrushShape, OrientationSource};

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
        Slider { label: "Flow", min: 0.0, max: 3.0, value: brush.dynamics.add,
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

/// The preset library (`crate::presets`) at the panel's foot: a header carrying the
/// Save button, over one row per preset — click applies it, hover reveals a remove ✕.
/// The row whose snapshot the live brush still *is* (colour aside) is highlighted; it
/// goes out the moment any knob moves.
///
/// The section takes every pixel the panel has left and scrolls its own overflow, so
/// the library's size is the user's to choose (by the panel's resize grip) rather than
/// something that pushes the sliders about as it grows. Save is a header button and
/// the name is asked for in a dialog ([`PresetSaveModal`]): a field that is only used
/// at the moment of saving does not earn a permanent row, and the dialog can say what
/// the field could not — that a familiar name will replace what is already there.
///
/// Safe as a child component: nothing here spawns, so there is no task to die with a
/// re-render (unlike the editor's slider rows — see `brush_editor::edit`).
#[component]
fn PresetSection() -> Element {
    let state = use_context::<AppState>();
    let entries = (state.presets)();
    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();

    rsx! {
        // `panel-grow`: the part of the panel that takes its spare height (and gives
        // it back when the grip shortens the panel) — see `.panel.resizable` in the
        // stylesheet. The Brush panel's only growable part.
        div { class: "preset-section panel-grow",
            div { class: "preset-header",
                span { class: "preset-header-title", "Presets" }
                button {
                    class: "chip preset-save",
                    title: "Save the current brush as a preset",
                    onclick: move |_| {
                        let mut open = state.preset_save_open;
                        open.set(true);
                    },
                    "Save"
                }
            }
            div { class: "preset-list",
                if entries.is_empty() {
                    div { class: "preset-empty", "No presets yet. Save keeps the brush you have now." }
                }
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
        }
    }
}

/// The "Save preset" dialog, opened by the Brush panel's Save button and mounted at
/// the app root (`main.rs`) so its backdrop covers the window rather than the panel.
///
/// Opens on the next free "Preset N" — selected, so typing replaces it — which is the
/// same default the old inline field applied silently to an empty box, except here it
/// is visible and editable before it is committed. Typing a name the library already
/// has is not an error but a deliberate act, so the dialog names it: the button says
/// "Replace" and a line underneath says what will be replaced. Blank is the one thing
/// it will not take (there is no such preset to click), so Save goes dead rather than
/// inventing a name behind the user's back.
#[component]
pub fn PresetSaveModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    // `peek`, not `read`: the proposed name is a starting point captured at open, not
    // a view of the library that should be re-derived under the user's typing.
    let mut name = use_signal(|| presets::next_name(&state.presets.peek()));

    let trimmed = name().trim().to_string();
    let replaces = !trimmed.is_empty() && state.presets.read().iter().any(|e| e.name == trimmed);

    let save = move || {
        let name = name().trim().to_string();
        if name.is_empty() {
            return;
        }
        presets::save_current(state, name);
        on_close.call(());
    };

    rsx! {
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "Save Preset" }
                div { class: "modal-subtitle",
                    "Keeps the whole brush — size, shape, dynamics, taper — under a name. Everything but the colour."
                }

                input {
                    class: "modal-input",
                    r#type: "text",
                    placeholder: "Preset name",
                    value: "{name}",
                    // Focused and selected as it appears: the dialog exists to take one
                    // word, and the proposed name is there to be typed over. `onmounted`
                    // rather than `autofocus`, which the browser does not honour for an
                    // element inserted after load (see `layer::LayerRow`).
                    onmounted: move |e: Event<MountedData>| {
                        spawn(async move {
                            let _ = e.set_focus(true).await;
                            select_all(&e);
                        });
                    },
                    oninput: move |e| name.set(e.value()),
                    onkeydown: move |e| match e.key() {
                        Key::Enter => save(),
                        Key::Escape => on_close.call(()),
                        _ => {}
                    },
                }
                // Always mounted, so the dialog does not change height under the hand
                // the moment the typed name lands on one the library already has.
                div { class: "modal-hint",
                    if replaces {
                        "Replaces the preset already called \u{201C}{trimmed}\u{201D}."
                    }
                }

                div { class: "modal-actions",
                    button {
                        class: "btn btn-secondary",
                        onclick: move |_| on_close.call(()),
                        "Cancel"
                    }
                    button {
                        class: "btn btn-primary",
                        disabled: trimmed.is_empty(),
                        onclick: move |_| save(),
                        if replaces { "Replace" } else { "Save" }
                    }
                }
            }
        }
    }
}

/// Switch to a shape, also setting a sensible default spacing for it.
pub fn set_shape(state: AppState, shape: BrushShape) {
    update_brush(state, move |b| b.shape = shape);
}

/// Set what orients the brush shape as it sweeps (§6.6).
pub fn set_orientation(state: AppState, orientation: OrientationSource) {
    update_brush(state, move |b| b.orientation = orientation);
}
