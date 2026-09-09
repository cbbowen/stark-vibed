//! Reading *this browser's* pointer events against the drag table (§25).
//!
//! The table itself is `stark_ui::drags` — the actions, the chords, the presets, the
//! stored rows, the gate an act puts on a press (`DragAction::claims`) and the offer's
//! own policy (`Offer`). What is here is the half that could not travel: which button
//! a DOM event presses, what this app knows about the hand that the table does not,
//! the doors that write signals, and the settings UI that draws the rows.

use dioxus::html::Modifiers;
use dioxus::html::input_data::MouseButton;
use dioxus::prelude::*;

use crate::icons::icon;
use crate::input::{accel, is_contact};
use crate::state::AppState;
use crate::widgets::Modal;
use stark_ui::drags::{
    DragAction, DragBindings, DragButton, DragCapture, DragChord, DragPreset, Hand, capture,
    chord_label, persist_drags, stored_drags,
};
use stark_ui::keys::Mods;
use strum::VariantArray;

/// The three modifiers as a DOM event reports them — the one translation from
/// [`Modifiers`], shared by the press path and the key tracker so the two cannot read
/// the same keystroke differently.
///
/// The frontend's half of `stark_ui::keys::Mods`, for the reason the chord
/// table's `stroke` is a frontend's: only this side knows that the accelerator is
/// Command on a Mac.
pub fn mods_of(m: Modifiers) -> Mods {
    Mods {
        ctrl: accel(m),
        shift: m.contains(Modifiers::SHIFT),
        alt: m.contains(Modifiers::ALT),
    }
}

fn button_of(e: &Event<PointerData>) -> Option<DragButton> {
    if is_contact(e) {
        Some(DragButton::Left)
    } else if e.trigger_button() == Some(MouseButton::Secondary) {
        Some(DragButton::Right)
    } else {
        None
    }
}

/// The action `e` asks for, if any — the one reader on the canvas's press path.
/// A row that matches may still decline the *press*
/// ([`DragAction::claims`]), and a declined press falls through to the
/// paint path exactly as an unbound chord does: over a selection tool, Alt+drag
/// *is* the subtract marquee.
///
/// It also **notices** a modified press the table has nothing for, which is what
/// brings the preset offer due ([`on_press`](stark_ui::drags::Offer::on_press)). Here rather
/// than in the
/// canvas's ladder because it is the table's own observation — "somebody reached for
/// a binding I do not have" — and because this is the one place that has already
/// asked the question.
pub fn find(state: AppState, e: &Event<PointerData>) -> Option<DragAction> {
    let button = button_of(e)?;
    let mods = mods_of(e.modifiers());
    let found = state.drags.peek().lookup(mods, button);
    // Asked of `lookup`'s answer and not of this function's, which is `Offer::on_press`'
    // first exclusion and the reason the two lines are apart.
    let mut offer = state.drag_offer;
    let held = *offer.peek();
    let next = held.on_press(mods, found);
    if next != held {
        offer.set(next);
    }
    found.filter(|a| a.claims(hand(state)))
}

/// What this app knows about the hand that the drag table does not
/// ([`Hand`]) — the press path's half, which is the three stand-downs and
/// nothing about a gesture already in flight: a press that is asking this
/// question is by definition the one arriving.
///
/// `peek` throughout: this runs inside a pointer handler, and nothing here is
/// mounted on the answer.
fn hand(state: AppState) -> Hand {
    Hand {
        panning: (state.space_down)(),
        selecting: crate::panels::select::current_tool(state).is_selection(),
        playing: crate::panels::timeline::is_playing(state),
        ..Default::default()
    }
}

/// Show an offer that has come due, now the canvas is out of the artist's hand —
/// called from [`end_interaction`](crate::input::end_interaction), which is
/// where every canvas gesture is put down.
///
/// Why it waits, and why the mark goes down here rather than on the answer, are
/// [`settle`](stark_ui::drags::Offer::settle)'s; what is left is raising the dialog and
/// writing the record.
pub fn settle_offer(state: AppState) {
    let mut offer = state.drag_offer;
    let (next, show) = (*offer.peek()).settle();
    if !show {
        return;
    }
    offer.set(next);
    save(state);
    let mut showing = state.dialogs.drag_presets;
    showing.set(true);
}

/// Put this browser's stored table and offer mark where the chrome reads them. The
/// reading and the row shapes are the registry's ([`stored_drags`]); what is left here
/// is the two signals.
pub fn load(state: AppState) {
    let Some((bindings, offer)) = stored_drags() else {
        return;
    };
    let mut table = state.drags;
    table.set(bindings);
    let mut mark = state.drag_offer;
    mark.set(offer);
}

/// Give `action` the captured chord, and persist the table — written through
/// [`DragBindings::rebind`] so a stolen chord and its victim's row change in the
/// same write.
pub fn rebind(state: AppState, action: DragAction, chord: DragChord) {
    edit(state, |b| b.rebind(action, chord));
}

/// Take `action`'s binding away and persist the table.
pub fn unbind(state: AppState, action: DragAction) {
    edit(state, |b| b.unbind(action));
}

/// Take `preset`'s whole table and persist it — one write, so its rows cannot be
/// seen half-applied.
pub fn set_preset(state: AppState, preset: DragPreset) {
    edit(state, |b| b.take(preset));
}

/// One change to the table, written to the signal and to storage as one act — so
/// what the rows show, what a press answers, and what the next visit loads
/// cannot be three states.
fn edit(state: AppState, change: impl FnOnce(&mut DragBindings)) {
    let mut bindings = state.drags;
    let mut next = bindings.peek().clone();
    change(&mut next);
    bindings.set(next);
    save(state);
}

/// Write the whole record — the override rows and the offer's mark, through the one
/// writer both kinds of row have ([`persist_drags`]).
fn save(state: AppState) {
    persist_drags(&state.drags.peek(), *state.drag_offer.peek());
}

/// The ⚙ dialog's drag section (§25.8): the presets as a run of chips, then a
/// row per action carrying its chord as the door to changing it.
///
/// Mounted by `settings::SettingsModal` rather than written there, because this
/// is one feature's surface over one feature's table and the settings module's
/// business is the *dialog* — the same split that leaves `collab::SessionModal`
/// and `files::ExportModal` with their own modules.
#[component]
pub fn DragBindingSection() -> Element {
    let state = use_context::<AppState>();
    // The action whose chord is being recaptured, if any — armed by its own
    // chip, spent by the next press that chip hears. One at a time, like the
    // palette's, and held here rather than per row so arming a second row
    // disarms the first.
    let capturing: Signal<Option<DragAction>> = use_signal(|| None);
    let table = state.drags.read().clone();
    let known = DragPreset::VARIANTS.iter().any(|p| p.matches(&table));

    rsx! {
        div { class: "setting-row",
            // The checkbox's column, kept empty so this row's text starts where
            // every other row's does (`settings::SettingChoice`).
            div { class: "setting-check-spacer" }
            div { class: "setting-text",
                div { class: "setting-label", "Start from another app" }
                div { class: "setting-desc",
                    "Take the three drags below from the app you already know. Each row can \
                     still be changed afterwards."
                }
                // `segmented`, because the chips answer one question and
                // picking one un-picks the rest (§25.9) — butted into a single
                // control the shape carries that, where chips standing
                // apart would promise switches that could be held down
                // together. They fit one line at the dialog's width, which is
                // the condition the rule turns on; a seventh preset is the cue
                // to re-measure and, failing that, to become a `.select`.
                div { class: "drag-presets segmented",
                    for preset in DragPreset::VARIANTS.iter().copied() {
                        button {
                            key: "{preset:?}",
                            class: if preset.matches(&table) { "chip active" } else { "chip" },
                            r#type: "button",
                            title: "{preset.blurb()}",
                            onclick: move |_| set_preset(state, preset),
                            "{preset.name()}"
                        }
                    }
                }
                if !known {
                    div { class: "setting-note",
                        "Your own table \u{2014} pick an app above to start over from its drags."
                    }
                }
            }
        }
        for action in DragAction::VARIANTS.iter().copied() {
            DragBindingRow { key: "{action:?}", action, capturing }
        }
    }
}

/// One action's row: what the drag does, and the chord that opens it as a chip
/// that captures a new one.
///
/// Click, then press the chord you want *with the button you want* — the same
/// gesture as using it, which is `main::BindChip`'s bargain restated for
/// presses. The press is read by [`capture`], so a plain click calls the capture
/// off and the ✕ beside the chip is what erases the binding.
#[component]
fn DragBindingRow(action: DragAction, capturing: Signal<Option<DragAction>>) -> Element {
    let state = use_context::<AppState>();
    let mut capturing = capturing;
    let listening = capturing() == Some(action);
    let bound = state.drags.read().of(action);

    let press = move |e: Event<PointerData>| {
        // The chip's press is the chip's alone, and never the browser's: a
        // right press here is a chord being captured rather than a context
        // menu, and a left one must not move focus off whatever has it.
        e.stop_propagation();
        e.prevent_default();
        if !listening {
            capturing.set(Some(action));
            return;
        }
        match capture(mods_of(e.modifiers()), button_of(&e)) {
            DragCapture::Chord(chord) => {
                rebind(state, action, chord);
                capturing.set(None);
            }
            DragCapture::Cancel => capturing.set(None),
            DragCapture::Pending => {}
        }
    };

    rsx! {
        div { class: "setting-row",
            div { class: "setting-check-spacer" }
            // Not a `<label>`, unlike the toggle rows: there is no one control
            // for the text to stand in front of, and a click on the sentence
            // must not arm the capture beside it.
            div { class: "setting-text",
                div { class: "setting-label",
                    "{action.name()}"
                    span { class: "drag-chord",
                        if listening {
                            span {
                                class: "menu-shortcut bind-chip capturing",
                                title: "Hold the keys you want and press here \u{2014} a plain click keeps what was there",
                                onpointerdown: press,
                                "hold keys, press\u{2026}"
                            }
                        } else if let Some(chord) = bound {
                            span {
                                class: "menu-shortcut bind-chip",
                                title: "Click, then hold the keys you want and press again",
                                onpointerdown: press,
                                {chord_label(chord)}
                            }
                            button {
                                class: "drag-clear",
                                r#type: "button",
                                title: "Leave this act with no drag at all",
                                onclick: move |_| unbind(state, action),
                                {icon(stark_ui::icons::CLOSE)}
                            }
                        } else {
                            span {
                                class: "menu-shortcut bind-chip bind-add",
                                title: "Bind a drag: click, then hold the keys you want and press again",
                                onpointerdown: press,
                                {icon(stark_ui::icons::ADD)}
                            }
                        }
                    }
                }
                div { class: "setting-desc", "{action.hint()}" }
            }
        }
    }
}

/// The offer, made once (§25.8): the presets as cards, each printing the three
/// drags it would set.
///
/// Raised by [`settle_offer`] off the release after a modified press this table
/// had nothing for — so it arrives to somebody who has just reached for a
/// binding they know from elsewhere, and to nobody else. Dismissing it is an
/// answer, which is why the mark is written when the dialog is *shown*.
#[component]
pub fn DragPresetModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();

    rsx! {
        // Wide for the settings dialog's reason: rows of three
        // drags each, and at the standard width every line of them wraps.
        Modal { class: "modal-wide", on_close,
            div { class: "modal-title", "Drags on the canvas" }
            div { class: "modal-subtitle",
                "You held a modifier and pressed, and Stark has nothing bound to that. If \
                 you have come from another app, start from the drags you already know."
            }

            div { class: "drag-preset-list",
                for preset in DragPreset::VARIANTS.iter().copied() {
                    button {
                        key: "{preset:?}",
                        class: "drag-preset",
                        r#type: "button",
                        onclick: move |_| {
                            set_preset(state, preset);
                            on_close.call(());
                        },
                        div { class: "drag-preset-head",
                            span { class: "drag-preset-name", "{preset.name()}" }
                            span { class: "drag-preset-blurb", "{preset.blurb()}" }
                        }
                        div { class: "drag-preset-rows",
                            for action in DragAction::VARIANTS.iter().copied() {
                                div { key: "{action:?}", class: "drag-preset-row",
                                    span { class: "drag-preset-act", "{action.word()}" }
                                    span { class: "menu-shortcut",
                                        match preset.chord(action) {
                                            Some(chord) => chord_label(chord),
                                            // An act this app reaches some
                                            // other way. Said with a dash
                                            // rather than left blank, so a
                                            // short card reads as a choice
                                            // and not as a card cut off.
                                            None => "\u{2014}".to_string(),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            div { class: "drag-offer-note",
                "Whatever you pick, \u{2699} Settings lists these three drags and can \
                 change any of them at any time."
            }

            div { class: "modal-actions",
                button {
                    class: "btn btn-primary",
                    onclick: move |_| on_close.call(()),
                    {icon(stark_ui::icons::DONE)}
                    "Not now"
                }
            }
        }
    }
}
