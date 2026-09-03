//! The unified Settings dialog: the client's own preferences, in one place
//! (§11).
//!
//! What belongs here is the third kind of state the UI carries. The panels hold
//! what you are painting *with* — a color, a brush, a selection — and change
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
//! - They are **saved on the click too**, per browser rather than into the
//!   document (`crate::prefs`). A row does not opt in: [`SettingToggle`] persists
//!   after calling its handler, so a row added here is durable by construction
//!   and only its *value* has to be named, in `Prefs`. A future row that is not a
//!   toggle is the one case that has to call [`prefs::save`] itself.
//! - A row is **a label and one sentence**. The sentence says what turning it on
//!   does; a note is added only for a caveat the label cannot carry — where the
//!   row applies, or how to get out of the state it puts you in. A map is read
//!   down its labels, and a paragraph under each is what stops it being one.
//! - Every row is **always mounted**, including ones that only bite in some
//!   contexts. A tool panel earns the opposite rule (a control that is present or
//!   absent says whether the thing it governs exists — §6.8), but a
//!   settings dialog is read as the *map* of what is configurable, and a map with
//!   roads that appear only once you are already on them is not one. Rows that are
//!   currently inert say so in their own text instead.

use dioxus::prelude::*;

use crate::collab::CollabPhase;
use crate::icons::icon;
use crate::prefs;
use crate::state::{AppState, dispatch, use_obs};
use crate::widgets::{Modal, slider_fill};
use stark_chrome::prefs::ChromeHiding;
use stark_engine::command::ViewCommand;

/// The settings dialog, opened from the command rail's ⚙ button and dismissed by
/// Done or by clicking the backdrop (as the other dialogs are).
#[component]
pub fn SettingsModal(on_close: EventHandler<()>) -> Element {
    let state = use_context::<AppState>();
    // Every engine-owned row of the dialog in **one** memo, which is what
    // `state::use_obs` asks for where the fields are read together: read straight,
    // as these were, the dialog re-rendered on every engine write for as long as
    // it stood open. The three are the peer-outline switch, the history budget and
    // the fast-commit switch below; each is the engine's own value rather than a
    // copy here, so none can disagree with what the engine believes (§4).
    let engine_owned = use_obs(state, |o| {
        (o.show_peer_selections, o.history_budget, o.fast_commit)
    });
    let show_peers = engine_owned().is_some_and(|(show, ..)| show);
    // The engine's default rather than a second opinion about it, as the budget
    // below is — and unreachable in practice for its reason: the dialog cannot be
    // open before the renderer is up.
    let fast_commit = engine_owned().map_or(stark_engine::DEFAULT_FAST_COMMIT, |(.., f)| f);
    let mut assist_enabled = state.assist.enabled;
    let assist = assist_enabled();
    let mut minimal_enabled = state.minimal;
    let minimal = minimal_enabled();
    let tips = (state.tutor.enabled)();
    let mut chrome_hiding = state.chrome_hiding;
    let hiding = chrome_hiding();
    // Read off the engine's projection, like the peer-outline row above: the engine
    // owns this and a copy here would be one that can disagree. Before the renderer
    // is up the dialog cannot be open, so the fallback is unreachable in practice
    // and is the engine's own default rather than a second opinion about it.
    let budget = engine_owned().map_or(stark_engine::DEFAULT_HISTORY_BUDGET, |(_, b, _)| b);
    // Keyed on the *session*, not on whether anyone is currently here, so the note
    // under the peer-outline row does not flicker as collaborators come and go.
    let shared = (state.collab.phase)() == CollabPhase::Shared;

    rsx! {
        // Wide, for the reason Credits and Timing Stats are: this is the
        // dialog with the most to say, and at the standard width every
        // sentence under a label ran to three lines. The same words in two
        // take a hundred pixels off the dialog, which is a hundred fewer to
        // scroll on the screens that made the cap necessary.
        Modal { class: "modal-wide", on_close,
            div { class: "modal-title", "Settings" }
            div { class: "modal-subtitle",
                "These apply to this browser, not to the drawing."
            }

            // Three sections, all full. The split between the first two is what
            // a row is *about*: the canvas and the work on it, or the chrome
            // around it. Sections that held one row each — one per feature that
            // happened to have a preference — sorted the code rather than the
            // reader. The third is not a set of rows at all but a *table*
            // (§25.8), and it comes last for that reason: a reader scanning
            // labels for a switch should not have to pass four bindings to
            // reach the two below.
            div { class: "modal-section-label", "PAINTING" }
            SettingToggle {
                id: "drawing-assist",
                label: "Snap shapes when you hold",
                // Says what the gesture *is*, because a hold is not a control
                // anybody can see — the dialog is the only place it is written
                // down (§6.9).
                description: "Hold the pen still mid-stroke and a rough line or ellipse snaps to the perfect shape; the rest of the drag steers it. Anything else is left as you drew it.",
                checked: assist,
                onchange: move |v| assist_enabled.set(v),
            }
            SettingToggle {
                id: "fast-commit",
                label: "Finish strokes instantly",
                // The label is what the artist feels; the sentence is the mechanism,
                // because the row cannot be judged without it — both settings paint
                // the same stroke, and what differs is when it is drawn (§6.2).
                description: "Keep the stroke exactly as you watched it appear when you lift the pen, instead of drawing it over again. Turning this off puts a pause at the end of every long stroke.",
                // The caveat the label cannot carry: what the slower road buys. Said
                // in the terms somebody would want it in — saving, sharing, undo —
                // rather than as the seam it is measured by.
                note: Some("Off, every stroke is drawn the one way a saved file, an undo and a collaborator all draw it, so your canvas reproduces exactly. Either way the two differ by at most a level or two of color.".to_string()),
                checked: fast_commit,
                onchange: move |v| dispatch(state, ViewCommand::SetFastCommit(v)),
            }
            SettingToggle {
                id: "show-peer-selections",
                label: "Show others' selections",
                description: "Outline the regions your collaborators have selected, each in their own color.",
                // A row that is inert right now explains itself rather than
                // vanishing — see the module comment.
                note: if shared { None } else { Some("Takes effect while you're sharing a session.".to_string()) },
                checked: show_peers,
                onchange: move |v| dispatch(state, ViewCommand::SetShowPeerSelections(v)),
            }
            SettingSlider {
                id: "history-budget",
                label: "Undo memory",
                // In the terms the artist has — how far back you can go, and what
                // it costs. Bytes are how it is measured and not what it is for,
                // so the number is on the readout and the sentence is the trade.
                description: "Graphics memory kept for undo. Past it the oldest steps are given up — saving and sharing always include the whole drawing.",
                steps: BUDGET_STEPS,
                value: budget,
                onchange: move |bytes| dispatch(state, ViewCommand::SetHistoryBudget(bytes)),
            }

            div { class: "modal-section-label", "INTERFACE" }
            SettingToggle {
                id: "minimal-chrome",
                label: "Minimal UI",
                // Says which text goes and that nothing goes with it, because
                // "minimal" alone could mean either (§11).
                description: "Drop the words from the panels and bars, keeping the icons — the same controls, in one column. Hover any of them for its name.",
                checked: minimal,
                onchange: move |v| minimal_enabled.set(v),
            }
            SettingChoice {
                label: "Panels and bars while you paint",
                // Says what the chrome does today, since two of the three options
                // are only meaningful against it (§11).
                description: "They float over the canvas, so they can step aside for the length of a stroke.",
                // The one thing the last option has to say: getting them back is a
                // gesture, and nothing on screen names it.
                note: Some("With \u{201C}Hide after painting\u{201D}, reach for the right edge of the window to bring the panels back.".to_string()),
                options: CHROME_CHOICES,
                value: hiding.key(),
                onchange: move |name: String| {
                    chrome_hiding.set(ChromeHiding::from(name));
                    // A stack already standing down when the choice moves off
                    // "Hide after painting" has nothing left to bring it back —
                    // the slice that hears the pointer is mounted on the very
                    // state being switched off. Waking is idempotent and free
                    // (`layout::wake_panels`), so it is done on every change
                    // rather than on the one that needs it.
                    crate::layout::wake_panels(state);
                },
            }
            SettingToggle {
                id: "tips",
                label: "Show tips as you work",
                // The thing somebody wants to know before leaving this on is
                // whether it will interrupt them (§24). "Once each" is the answer.
                description: "Point out where Stark differs from the apps you came from \u{2014} once each, never during a stroke, never over the canvas.",
                checked: tips,
                // Through the tour's own door rather than straight onto the
                // signal: turning tips off has to take down the card that is
                // already up, which is a fact about the tour and is kept there
                // (`tutor::set_enabled`).
                onchange: move |v| crate::tutor::set_enabled(state, v),
            }

            div { class: "modal-section-label", "DRAGS ON THE CANVAS" }
            // The drag table's own surface, mounted rather than written here
            // (`drags::DragBindingSection`): what a row of it *is* — a chord,
            // a capture, a preset — is the table's business, and this dialog's
            // is being the place they are found. The one section whose rows
            // are not preferences, and it wears the same row shape anyway, so
            // the dialog still reads down one column of labels.
            crate::drags::DragBindingSection {}

            div { class: "modal-actions",
                button {
                    class: "btn btn-primary",
                    onclick: move |_| on_close.call(()),
                    {icon(stark_chrome::icons::DONE)}
                    "Done"
                }
            }
        }
    }
}

/// What the chrome-hiding row offers, in the order the chrome gets quieter
/// ([`ChromeHiding`]): each option's stored name, its label, and the sentence that
/// says what picking it does.
///
/// The names are [`ChromeHiding::key`]'s own, which is what keeps the dialog and the
/// store speaking one vocabulary — a row here cannot come to offer a value nothing
/// reads back.
const CHROME_CHOICES: &[(&str, &str, &str)] = &[
    (
        "never",
        "Always show",
        "Everything stays where it is, whatever the hand is doing.",
    ),
    (
        "while-painting",
        "Hide while painting",
        "The chrome fades for the length of a stroke and is back the moment you lift.",
    ),
    (
        "after-painting",
        "Hide after painting",
        "The chrome fades for the stroke, and the panels stay away until you reach for them.",
    ),
];

/// The undo-memory ladder, smallest first: what the slider's notches mean.
///
/// **A ladder rather than a linear range over bytes**, because the quantity is
/// scale-free — the difference between 256 MiB and 512 MiB matters to a phone in the
/// way 4 GiB to 8 GiB matters to a workstation, and a linear slider spends nine
/// tenths of its travel in a region only one of them cares about. Doubling gives
/// every notch the same meaning.
///
/// The top notch is genuinely unbounded: retention never trims, which is a real
/// choice on a machine with memory to spare and one a ladder can offer honestly
/// where a number entry could not. It still floors at the engine's minimum undo
/// depth, because that floor is about trimming being *useless* below it rather than
/// about the budget.
const BUDGET_STEPS: &[(u64, &str)] = &[
    (256 << 20, "256 MB"),
    (512 << 20, "512 MB"),
    (1 << 30, "1 GB"),
    (2 << 30, "2 GB"),
    (4 << 30, "4 GB"),
    (8 << 30, "8 GB"),
    (u64::MAX, "Unlimited"),
];

/// The notch `bytes` sits at, or the nearest one below it.
///
/// Nearest-below rather than exact, because the stored value is a `u64` that a
/// future ladder may not name — a preference written by one version has to read as
/// *something* in the next, and the safe direction is the smaller budget, which
/// errs toward less memory rather than more.
fn budget_step(bytes: u64) -> usize {
    BUDGET_STEPS
        .iter()
        .rposition(|(v, _)| *v <= bytes)
        .unwrap_or(0)
}

/// One setting chosen from a ladder of values: the same row as [`SettingToggle`],
/// with a range input and the chosen value's name beside it.
///
/// Sliding a **notch index** rather than the value itself, so the control is even
/// where the quantity is not — see [`BUDGET_STEPS`]. The index never leaves this
/// component; what the handler is given is the value the notch stands for.
///
/// It saves for itself, unlike the toggle: the module comment anticipates exactly
/// this — "a future row that is not a toggle is the one case that has to call
/// `prefs::save` itself" — because persistence hangs off `SettingToggle`'s own
/// input handler and a second control cannot inherit it.
#[component]
fn SettingSlider(
    id: String,
    label: String,
    description: String,
    note: Option<String>,
    steps: &'static [(u64, &'static str)],
    value: u64,
    onchange: EventHandler<u64>,
) -> Element {
    let state = use_context::<AppState>();
    let at = budget_step(value);
    let max = steps.len().saturating_sub(1);

    rsx! {
        div { class: "setting-row",
            // Where the toggle puts its checkbox. Empty rather than absent so the
            // text column starts at the same place on every row — a settings dialog
            // is read down its labels, and a ragged left edge is what a shared
            // gutter exists to prevent.
            div { class: "setting-check-spacer" }
            label { r#for: "{id}", class: "setting-text",
                div { class: "setting-label",
                    "{label}"
                    span { class: "setting-value", "{steps[at].1}" }
                }
                div { class: "setting-desc", "{description}" }
                input {
                    id: "{id}",
                    class: "slider setting-slider",
                    style: slider_fill(0.0, max as f32, at as f32),
                    r#type: "range",
                    min: "0",
                    max: "{max}",
                    step: "1",
                    value: "{at}",
                    oninput: move |e| {
                        let Ok(i) = e.value().parse::<usize>() else { return };
                        let Some((bytes, _)) = steps.get(i) else { return };
                        onchange.call(*bytes);
                        prefs::save(state);
                    },
                }
                if let Some(note) = note {
                    div { class: "setting-note", "{note}" }
                }
            }
        }
    }
}

/// One setting chosen from a handful of named states: the same row as
/// [`SettingToggle`], with the options as a segmented run of chips under the
/// description and the chosen one lit.
///
/// Chips rather than a drop-down or a run of radio buttons, because all three answers
/// are worth reading side by side — the choice is between *behaviors*, and each needs
/// its own sentence. Each chip carries that sentence as its title, so the row explains
/// the option under the pointer without spending three lines of dialog on states
/// nobody picked.
///
/// Stringly typed on purpose. The value is the same name the preference is stored
/// under (`ChromeHiding::key`), so the dialog, the store and the enum share one
/// vocabulary and this component stays a *choice* rather than a second component per
/// enum — the same bargain [`SettingSlider`] makes by sliding an index.
///
/// It saves for itself, like the slider and unlike the toggle — persistence hangs off
/// `SettingToggle`'s own input handler, and a second control cannot inherit it.
#[component]
fn SettingChoice(
    label: String,
    description: String,
    note: Option<String>,
    options: &'static [(&'static str, &'static str, &'static str)],
    value: &'static str,
    onchange: EventHandler<String>,
) -> Element {
    let state = use_context::<AppState>();

    rsx! {
        div { class: "setting-row",
            // The checkbox's column, kept empty so this row's text starts where every
            // other row's does — see `SettingSlider`.
            div { class: "setting-check-spacer" }
            div { class: "setting-text",
                div { class: "setting-label", "{label}" }
                div { class: "setting-desc", "{description}" }
                // `segmented`, because the three chips answer one question and
                // picking one un-picks the rest: butted into a single control the
                // shape carries that, where a run of separate chips promises three
                // switches that could be held down together.
                div { class: "setting-choice segmented",
                    for (key, name, about) in options.iter().copied() {
                        button {
                            key: "{key}",
                            class: if key == value { "chip active" } else { "chip" },
                            title: "{about}",
                            onclick: move |_| {
                                onchange.call(key.to_string());
                                prefs::save(state);
                            },
                            "{name}"
                        }
                    }
                }
                if let Some(note) = note {
                    div { class: "setting-note", "{note}" }
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
///
/// Persisting is done **here** rather than in each row's handler, so that a row
/// added to the dialog is durable without its author thinking about storage — the
/// same "rule out the class" move the rest of the app makes. It runs after the
/// handler, so what it captures is the state the click produced.
#[component]
fn SettingToggle(
    id: String,
    label: String,
    description: String,
    note: Option<String>,
    checked: bool,
    onchange: EventHandler<bool>,
) -> Element {
    let state = use_context::<AppState>();

    rsx! {
        div { class: "setting-row",
            input {
                id: "{id}",
                class: "setting-check",
                r#type: "checkbox",
                checked,
                onchange: move |e| {
                    onchange.call(e.checked());
                    prefs::save(state);
                },
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

#[cfg(test)]
mod tests {
    use super::*;

    /// **The engine's default must land on a notch**, or the slider opens showing a
    /// value the app is not using and moving it one step is a jump rather than a
    /// nudge. The ladder and `stark-engine`'s constant are set independently, so this
    /// is the only thing holding them together.
    #[test]
    fn the_default_budget_is_a_notch_on_the_ladder() {
        let at = budget_step(stark_engine::DEFAULT_HISTORY_BUDGET);
        assert_eq!(
            BUDGET_STEPS[at].0,
            stark_engine::DEFAULT_HISTORY_BUDGET,
            "the engine default {} sits between notches, nearest below is {}",
            stark_engine::DEFAULT_HISTORY_BUDGET,
            BUDGET_STEPS[at].1,
        );
        assert_eq!(BUDGET_STEPS[at].1, "2 GB");
    }

    /// A stored value the ladder does not name reads as the notch **below** it.
    ///
    /// Preferences outlive the version that wrote them, so a ladder that gains or
    /// loses a rung has to read every previously stored `u64` as something. Below
    /// rather than nearest, because the two directions are not symmetric: erring
    /// down costs undo depth the user can see and slide back, erring up quietly
    /// hands out memory they had asked not to spend.
    #[test]
    fn an_unnamed_budget_reads_as_the_notch_below() {
        // Between 1 GB and 2 GB.
        assert_eq!(BUDGET_STEPS[budget_step((1 << 30) + 1)].1, "1 GB");
        // Exactly on a notch is that notch, not the one below.
        assert_eq!(BUDGET_STEPS[budget_step(1 << 30)].1, "1 GB");
        // Below every notch — a zero from a caller that meant "as little as
        // possible" — is the smallest, not a panic and not the largest.
        assert_eq!(budget_step(0), 0);
        // And the top is reachable, so "Unlimited" is not a rung nothing selects.
        assert_eq!(BUDGET_STEPS[budget_step(u64::MAX)].1, "Unlimited");
    }

    /// The ladder ascends, which `budget_step`'s `rposition` scan assumes and which
    /// a slider's notches have to do to mean anything.
    #[test]
    fn the_ladder_ascends() {
        assert!(
            BUDGET_STEPS.windows(2).all(|w| w[0].0 < w[1].0),
            "the undo-memory ladder is not in increasing order",
        );
    }
}
