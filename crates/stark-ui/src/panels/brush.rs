//! The floating Brush panel: the everyday quick controls. The full parameter set
//! lives in the brush editor dialog.

use dioxus::prelude::*;

use crate::commands::Command;
use crate::icons::{self, icon};
use crate::platform::select_all;
use crate::presets;
use crate::state::{AppState, update_brush, use_obs};
use crate::widgets::{CommandButton, Modal, Slider};
use stark_model::document::{BrushShape, OrientationSource};

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

/// The widest contact transition the editor offers, in the rise's own units
/// (`BrushParams::tooth_softness`, §6.4).
///
/// A slider's end rather than a bound on the quantity, which is why it is here and
/// not on the brush — but it is not arbitrary either. The rise a substrate map can
/// carry spans ±`RISE_LIMIT` = 0.25, so a band of 0.5 already covers the whole of it:
/// every texel is somewhere inside the transition, the gate is a flat scale on the
/// deposit, and the grain has stopped reading. Past that the knob would only be
/// walking towards a half.
pub const MAX_TOOTH_SOFTNESS: f32 = 1.0;

/// The floating Brush panel: the everyday quick controls (size, amount).
/// Everything else — the full grouped parameter set with a live test
/// stroke — lives in the brush editor dialog ("Edit brush…").
#[component]
pub fn BrushPanel() -> Element {
    let state = use_context::<AppState>();
    // The one field the panel is about, through a memo: reading the projection
    // straight woke these two sliders on every engine write — a layer opacity
    // drag, a selection command — to redraw the numbers they were already
    // showing (`state::use_obs`).
    let brush = use_obs(state, |o| o.brush)().unwrap_or_default();

    rsx! {
        // The panel's two sliders are the two knobs a hand reaches for without looking
        // away from the canvas, which is what earns them their marks (`icons::SIZE`).
        //
        // They are the *live* brush's, which while a number is held is that
        // number's (§18.1.8) — the panel needs no line of code that knows about
        // slots, and the rack draws itself over the canvas while the key is down
        // (`slots::SlotOverlay`) rather than keeping a row of chips here.
        Slider { label: "Size", glyph: icons::SIZE, min: MIN_RADIUS, max: MAX_RADIUS, value: brush.size,
            oninput: move |v| update_brush(state, move |b| b.size = v) }
        // "Flow" is the effect's own source rate — how much a paint brush lays,
        // or how fast an eraser's bite builds (§6.12) — so the slider tunes
        // the tool in hand whichever it is (`BrushEffect::flow`).
        Slider { label: "Flow", glyph: icons::FLOW, min: 0.0, max: MAX_FLOW, value: brush.effect.flow(),
            oninput: move |v| update_brush(state, move |b| b.effect.set_flow(v)) }
        // The panel's two doors, side by side: adjust the brush you have, or keep it.
        // One line rather than two because they are the same size of thing — a button
        // that opens a dialog — and the panel's scarcest dimension is height, which
        // every panel below it in the column pays for. They read left to right in the
        // order the work happens: tune it, then save it.
        div { class: "brush-actions",
            // A wrench, not a brush: the panel this button sits in is already the brush,
            // and what the dialog opens is the place it gets adjusted (`icons::EDIT_BRUSH`).
            CommandButton { command: Command::EditBrush, class: "be-open" }
            // Moved up here from the preset list's own header, where it cost that
            // header its whole reason to exist. It says "Save preset" rather than
            // "Save" now that it no longer sits on the list it saves into: a button
            // beside "Edit brush…" has to name its own subject.
            CommandButton { command: Command::SavePreset, class: "be-open" }
        }

        PresetSection {}
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
/// Safe as a child component: nothing here spawns at all. The thumbnails the
/// rows are drawn with are generated by a task the **root** starts over
/// root-owned signals (`crate::thumbs`, `main::app`), so there is none to die
/// with a re-render (unlike the editor's slider rows — see `brush_editor::edit`)
/// and none that stops when this panel is closed.
#[component]
fn PresetSection() -> Element {
    let state = use_context::<AppState>();
    let entries = (state.presets)();
    // The whole tool, feel included (§6.11), so a row goes out when the
    // smoothing moves off its snapshot like it does for any other knob.
    // Through a memo, like the panel above: the roster's lit row moves with the
    // brush and with nothing else the projection carries (`state::use_obs`).
    let brush = presets::Wearable {
        params: use_obs(state, |o| o.brush)().unwrap_or_default(),
        smoothing: (state.smoothing)(),
    };

    rsx! {
        // `panel-grow`: the part of the panel that takes its spare height (and gives
        // it back when the grip shortens the panel) — see `.panel.resizable` in the
        // stylesheet. The Brush panel's only growable part.
        div { class: "preset-section panel-grow",
            // The word alone now that Save sits with Edit brush above (`BrushPanel`) —
            // and so the first thing minimal mode drops here, since a heading naming
            // the only list in the panel is exactly the kind of word that mode is for.
            // Not wrapped in `label()`: this is prose about what follows rather than a
            // control's name, and what stands in for it is the list itself.
            div { class: "preset-header",
                span { class: "preset-header-title", "Presets" }
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
                        // The brush as a stroke (`crate::thumbs`), filling the whole
                        // row as its background: the preview is the star, and the
                        // name floats over it (shadowed in the stylesheet) to tell
                        // apart what the marks cannot. `none` is written out rather
                        // than the property omitted — an edited preset re-keys its
                        // thumbnail, and a stale declaration on this reused node
                        // would keep showing the old brush while the new one
                        // renders (inline style merges per property).
                        let bg = match crate::thumbs::url(state, &entry.brush) {
                            Some(url) if !url.is_empty() => {
                                format!("background-image: url({url});")
                            }
                            _ => "background-image: none;".to_string(),
                        };
                        rsx! {
                            div {
                                key: "{entry.name}",
                                class: if active { "preset-row active" } else { "preset-row" },
                                style: "{bg}",
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
/// Opens on the next free "Preset N" — selected, so typing replaces it. The default is
/// visible and editable before it is committed, rather than applied silently to an
/// empty box. Typing a name the library already
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
        Modal { on_close,
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

/// Switch to a shape, also setting a sensible default spacing for it.
pub fn set_shape(state: AppState, shape: BrushShape) {
    update_brush(state, move |b| b.shape = shape);
}

/// Set what orients the brush shape as it sweeps (§6.6).
pub fn set_orientation(state: AppState, orientation: OrientationSource) {
    update_brush(state, move |b| b.orientation = orientation);
}
