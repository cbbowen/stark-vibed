//! The floating Brush panel: the everyday quick controls. The full parameter set
//! lives in the brush editor dialog.

use dioxus::prelude::*;

use crate::icons::{self, icon, label};
use crate::platform::select_all;
use crate::presets;
use crate::slots;
use crate::state::{AppState, update_brush};
use crate::widgets::Slider;
use stark_core::document::{BrushShape, OrientationSource};

/// The smallest brush radius (`BrushParams::radius`). A tip finer than a canvas
/// pixel has nothing left to narrow.
pub const MIN_RADIUS: f32 = 1.0;

/// The maximum brush radius (`BrushParams::radius`).
pub const MAX_RADIUS: f32 = 500.0;

/// The most flow the sliders offer (`BrushDynamics::add`). Three, not one, because
/// `add` is a rate rather than a fraction — the everyday brush sits near 1 and a
/// loaded one that buries what is under it wants more.
///
/// Named beside the radius bounds because the *drag* bindings clamp against the same
/// three figures (`input::Tune`, §18.1.9): a knob reachable two ways must have one
/// range, or the drag would quietly go somewhere the slider cannot show.
pub const MAX_FLOW: f32 = 3.0;

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
        // First, because it says *which* brush the two sliders under it are the
        // knobs of: while a number is held they are that number's.
        SlotRack {}

        // The panel's two sliders are the two knobs a hand reaches for without looking
        // away from the canvas, which is what earns them their marks (`icons::SIZE`).
        Slider { label: "Size", glyph: icons::SIZE, min: MIN_RADIUS, max: MAX_RADIUS, value: brush.radius,
            oninput: move |v| update_brush(state, move |b| b.radius = v) }
        Slider { label: "Flow", glyph: icons::FLOW, min: 0.0, max: MAX_FLOW, value: brush.dynamics.add,
            oninput: move |v| update_brush(state, move |b| b.dynamics.add = v) }
        // A wrench, not a brush: the panel this button sits in is already the brush,
        // and what the dialog opens is the place it gets adjusted (`icons::EDIT_BRUSH`).
        button {
            class: "be-open",
            onclick: move |_| {
                let mut open = state.brush_editor_open;
                open.set(true);
            },
            {icon(icons::EDIT_BRUSH)}
            {label("Edit brush\u{2026}")}
        }

        hr {}

        PresetSection {}
    }
}

/// The quick-brush rack (§18.1.8): the ten brushes the number keys hold, as a row
/// of chips at the head of the Brush panel.
///
/// It is a *picture of a binding*, and that is most of its job — the numbers work
/// whether or not this row is on screen, and nothing here is the state of
/// anything. What it shows is which of them have brushes in, which one the live
/// brush currently is, and (while a key is down) which one is being held, so the
/// swap the hand just made has somewhere to be seen.
///
/// The chips are clickable as well, applying the slot for good, because a rack
/// reachable only through the number row would be no rack at all for the hand
/// this feature is for: a pen in one hand and a tablet under it leaves no spare
/// finger for the keyboard. A click is deliberately not what a *tap* on the key
/// does — see [`slots::pick`].
///
/// Safe as a child component: nothing here spawns, so there is no task to die
/// with a re-render (the same note `PresetSection` carries).
#[component]
fn SlotRack() -> Element {
    let state = use_context::<AppState>();
    let rack = (state.slots.brushes)();
    let held = (state.slots.held)().map(|h| h.slot);
    let brush = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.brush)
        .unwrap_or_default();

    rsx! {
        div { class: "slot-rack",
            // The digits in the order they sit on the keyboard, with the eraser's
            // own slot last — where the `0` key is, and where a tenth of anything
            // goes (`slots::ERASER`).
            for slot in (1..slots::COUNT).chain(std::iter::once(slots::ERASER)) {
                {
                    let assigned = rack[slot];
                    // Lit like a preset row, and on the same test: this is the
                    // brush in hand, color aside, until any knob moves off it.
                    // Held wins — it is momentary, and it is the thing the user is
                    // doing right now rather than a state they are in.
                    let mut class = String::from("slot-chip");
                    if assigned.is_none() {
                        class.push_str(" empty");
                    } else if assigned.is_some_and(|b| presets::matches(&brush, &b)) {
                        class.push_str(" active");
                    }
                    if held == Some(slot) {
                        class.push_str(" held");
                    }
                    let title = match (slot, assigned.is_some()) {
                        // The eraser's chip says the binding a glyph cannot: that
                        // the pen already in the hand reaches it. It does not claim
                        // the slot *is* an eraser — that is only what it ships
                        // with, and it is overwritten like any other.
                        (slots::ERASER, true) => "Hold 0, or flip the pen over".to_string(),
                        (slots::ERASER, false) => "Empty. Hold 0 (or the pen's eraser end) and click a preset".to_string(),
                        (n, true) => format!("Hold {n} to paint with this brush"),
                        (n, false) => format!("Empty. Hold {n} and click a preset to fill it"),
                    };
                    rsx! {
                        button {
                            key: "{slot}",
                            class,
                            title,
                            onclick: move |_| slots::pick(state, slot),
                            if slot == slots::ERASER {
                                {icon(icons::ERASER)}
                            } else {
                                "{slot}"
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The preset library (`crate::presets`) at the panel's foot: a header carrying the
/// Save button, over one row per preset — click applies it, hover reveals a remove ✕.
/// The row whose snapshot the live brush still *is* (color aside) is highlighted; it
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
                    {icon(icons::SAVE)}
                    {label("Save")}
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
                                if entry.builtin {
                                    // The app's own, and the row says so where the
                                    // user's rows offer to remove: a lock instead
                                    // of a trash (`icons::BUILTIN`), in the same
                                    // column, so the two kinds are told apart by
                                    // the one thing that differs between them.
                                    //
                                    // Always on, unlike the hover-revealed trash —
                                    // it is a state rather than an act, and a mark
                                    // that only appeared under the pointer would
                                    // distinguish nothing at rest. The eye in the
                                    // layer rows is the same argument.
                                    span {
                                        class: "preset-lock",
                                        title: "Built in \u{2014} kept up to date with the app",
                                        {icon(icons::BUILTIN)}
                                    }
                                } else {
                                    // The same trash the Layers and Guides rows wear
                                    // (`icons::REMOVE`): a third roster, and removing a
                                    // row from it is the same control, so it is the same
                                    // mark. The × it replaces was a character standing in
                                    // for a glyph the set already had.
                                    button {
                                        class: "preset-remove",
                                        title: "Remove preset",
                                        onclick: move |e| {
                                            e.stop_propagation();
                                            presets::remove(state, &remove_name);
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
///
/// A name one of the **app's own** presets holds is the second thing it will not take,
/// and it is a different refusal: not "you have not said enough" but "that one is not
/// yours". Overwriting is not on offer — the next start rebuilds a built-in from its
/// definition, so the work would go — and a second row under the same name would make
/// "the preset called Pen" two brushes to every lookup in `crate::presets`. So the
/// hint says which it is and Save goes dead, in the same line that would otherwise
/// have offered to replace.
#[component]
pub fn PresetSaveModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    // `peek`, not `read`: the proposed name is a starting point captured at open, not
    // a view of the library that should be re-derived under the user's typing.
    let mut name = use_signal(|| presets::next_name(&state.presets.peek()));

    let trimmed = name().trim().to_string();
    let taken = !trimmed.is_empty() && state.presets.read().iter().any(|e| e.name == trimmed);
    let builtin = !trimmed.is_empty() && presets::is_builtin(&state.presets.read(), &trimmed);
    // Taken by one of the user's own, which is the case Save offers to replace.
    let replaces = taken && !builtin;

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
                    "Keeps the whole brush — size, shape, dynamics, taper — under a name. Everything but the color."
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
                div { class: if builtin { "modal-hint refused" } else { "modal-hint" },
                    if builtin {
                        "\u{201C}{trimmed}\u{201D} is one of the app's own presets, which it keeps up to date. Pick another name."
                    } else if replaces {
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
                        disabled: trimmed.is_empty() || builtin,
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
