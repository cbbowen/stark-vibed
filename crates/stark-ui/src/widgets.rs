//! Small reusable controls shared by the panels and the brush editor.

use dioxus::prelude::*;

use crate::commands::Command;
use crate::icons::{icon, label as label_span};
use crate::state::AppState;

/// A button that runs a [`Command`], wearing the command's own mark, word and
/// tooltip (`crate::commands`) — so a control and the act it reaches cannot
/// describe each other differently, and a chord the act gains is advertised
/// here without the button changing.
///
/// `class` stays the call site's because the registry deliberately says nothing
/// about *where* a command is drawn: the same act is a `chip` on a bar and a
/// `layer-add` in a panel header, and the stylesheet keys on the slot, not the
/// act. What a call site may **not** vary is what the button says or does — a
/// site needing that (the Fill chip's paint-tinted bucket) writes its own
/// `button` and still reads the words off the command.
#[component]
pub fn CommandButton(
    command: Command,
    #[props(default = String::from("chip"))] class: String,
) -> Element {
    let state = use_context::<AppState>();
    rsx! {
        button {
            class,
            title: command.tooltip(&state.bindings.read()),
            onclick: move |_| command.run(state),
            {icon(command.icon())}
            {label_span(command.word())}
        }
    }
}

/// A labelled range control.
///
/// `glyph` is an `Option` because the brush editor's dense parameter list has not been
/// marked yet, **not** because a slider is expected to go without one. A control's mark
/// is the half of it that survives its label, so anything reachable wants one; a `None`
/// here is a row that would be blank if the words were hidden, and is a to-do rather
/// than a decision (see [`crate::icons::SIZE`]).
///
/// Which is exactly why the word is wrapped as hideable ([`crate::icons::label`]) only
/// when there *is* a mark to fall back on. An unmarked slider keeps its word in minimal
/// mode — not as a special case, but because the two facts are one fact here, and a
/// component that reads them off each other cannot be given the wrong pair. The rows
/// still to be marked therefore stay legible in the meantime instead of turning into a
/// column of anonymous tracks.
///
/// `marked` on the row carries that same fact out to the stylesheet, which needs it for
/// a second reason: in minimal mode a marked row folds onto **one line**, its glyph to
/// the left of the track instead of over it, which is where the mode's vertical saving
/// in the panel stack actually comes from. A row that kept its word cannot fold — the
/// words differ in length, so the tracks would start at a ragged left edge — and it
/// does not have to, because the class it would need is the one it does not get.
#[component]
pub fn Slider(
    label: String,
    #[props(default)] glyph: Option<&'static str>,
    min: f32,
    max: f32,
    value: f32,
    oninput: EventHandler<f32>,
) -> Element {
    rsx! {
        div { class: if glyph.is_some() { "slider-row marked" } else { "slider-row" },
            div { class: "slider-label",
                match glyph {
                    Some(glyph) => rsx! { {icon(glyph)} {label_span(&label)} },
                    None => rsx! { "{label}" },
                }
            }
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
