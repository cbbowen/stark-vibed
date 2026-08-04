//! The unified Settings dialog: the client's own preferences, in one place
//! (§11).
//!
//! What belongs here is the third kind of state the UI carries. The panels hold
//! what you are painting *with* — a colour, a brush, a selection — and change
//! constantly mid-painting. The document dialogs hold what the drawing *is*. A
//! setting is neither: it is a standing choice about how Stark behaves for **this
//! client**, set once and then left alone, and it is never part of the artwork.
//! Scattering those across the tool panels costs panel space on controls nobody
//! touches twice a session, and — worse — leaves no answer to "where do I change
//! that?" other than remembering which panel it landed in. One dialog off the
//! command rail is that answer.
//!
//! Consequences worth stating, because the rest of the file follows from them:
//!
//! - Settings **apply on the click**, so the dialog has a Done button and no
//!   Cancel — there is nothing staged to discard, and a preference you can see
//!   taking effect behind the dialog is one you can judge.
//! - Every row is **always mounted**, including ones that only bite in some
//!   contexts. A tool panel earns the opposite rule (a control that is present or
//!   absent says whether the thing it governs exists — §6.8), but a
//!   settings dialog is read as the *map* of what is configurable, and a map with
//!   roads that appear only once you are already on them is not one. Rows that are
//!   currently inert say so in their own text instead.

use dioxus::prelude::*;

use crate::collab::CollabPhase;
use crate::icons::{self, icon};
use crate::state::{AppState, dispatch};
use stark_core::command::ViewCommand;

/// The settings dialog, opened from the command rail's ⚙ button and dismissed by
/// Done or by clicking the backdrop (as the other dialogs are).
#[component]
pub fn SettingsModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    let show_peers = state
        .obs
        .read()
        .as_ref()
        .is_some_and(|o| o.show_peer_selections);
    let mut assist_enabled = state.assist.enabled;
    let assist = assist_enabled();
    let mut minimal_enabled = state.minimal;
    let minimal = minimal_enabled();
    // Keyed on the *session*, not on whether anyone is currently here, so the note
    // under the peer-outline row does not flicker as collaborators come and go.
    let shared = (state.collab.phase)() == CollabPhase::Shared;

    rsx! {
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                class: "modal-dialog",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "Settings" }
                div { class: "modal-subtitle",
                    "These apply to this browser, not to the drawing — nothing here is saved into the document or sent to anyone you share with."
                }

                div { class: "modal-section-label", "DRAWING" }
                SettingToggle {
                    id: "drawing-assist",
                    label: "Snap shapes when you hold",
                    // Says what the gesture *is*, because a hold is not a control
                    // anybody can see — the dialog is the only place it is written
                    // down (§6.9).
                    description: "Draw a rough line or ellipse and keep the pen down without moving: the stroke snaps to the perfect shape, and the rest of the drag steers it. Lift to finish.",
                    // The one thing somebody turning it off is likely to be reacting
                    // to, stated rather than left to be discovered.
                    note: Some("Strokes that aren't close to a line or an ellipse are left exactly as you drew them.".to_string()),
                    checked: assist,
                    onchange: move |v| assist_enabled.set(v),
                }

                div { class: "modal-section-label", "APPEARANCE" }
                SettingToggle {
                    id: "minimal-chrome",
                    label: "Minimal chrome",
                    // Says which text goes, because "minimal" on its own could mean
                    // anything from a smaller font to hiding the panels outright — and
                    // the one thing somebody needs to know before turning it on is that
                    // the controls all stay exactly where they were (§11).
                    description: "Drop the words from the panels and the bars over the canvas, keeping the icons. Nothing moves and nothing is removed — the same controls, in the same places, quieter.",
                    // The reassurance that makes it safe to try, and the answer to the
                    // question it raises: dialogs, menus, panel titles and anything you
                    // have named keep their text, so nothing becomes unreadable.
                    note: Some("Dialogs, menus, panel titles and your own layer and preset names keep their text. Hover any control for its name.".to_string()),
                    checked: minimal,
                    onchange: move |v| minimal_enabled.set(v),
                }

                div { class: "modal-section-label", "COLLABORATION" }
                SettingToggle {
                    id: "show-peer-selections",
                    label: "Show others' selections",
                    // Says what it draws *and* what it costs, which is why it is off
                    // by default (§17.3): a second contour over the
                    // artwork is paid for on every frame you look at it.
                    description: "Outline the regions your collaborators have selected, each in their own colour, alongside your own.",
                    // A row that is inert right now explains itself rather than
                    // vanishing — see the module comment.
                    note: if shared { None } else { Some("Takes effect while you're sharing a session.".to_string()) },
                    checked: show_peers,
                    onchange: move |v| dispatch(state, ViewCommand::SetShowPeerSelections(v)),
                }

                div { class: "modal-actions",
                    button {
                        class: "btn btn-primary",
                        onclick: move |_| on_close.call(()),
                        {icon(icons::DONE)}
                        "Done"
                    }
                }
            }
        }
    }
}

/// One on/off setting: a checkbox, its label, the sentence that says what turning
/// it on actually does, and an optional note about when it applies.
///
/// The description is not optional in practice and so it is not optional here: a
/// settings dialog is where a control meets someone who has never seen it, and a
/// bare label leaves them to guess. The whole row is the `<label>`, so the text is
/// as clickable as the box.
#[component]
fn SettingToggle(
    id: String,
    label: String,
    description: String,
    note: Option<String>,
    checked: bool,
    onchange: EventHandler<bool>,
) -> Element {
    rsx! {
        div { class: "setting-row",
            input {
                id: "{id}",
                class: "setting-check",
                r#type: "checkbox",
                checked,
                onchange: move |e| onchange.call(e.checked()),
            }
            label { r#for: "{id}", class: "setting-text",
                div { class: "setting-label", "{label}" }
                div { class: "setting-desc", "{description}" }
                if let Some(note) = note {
                    div { class: "setting-note", "{note}" }
                }
            }
        }
    }
}
