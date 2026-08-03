//! Small reusable controls shared by the panels and the brush editor.

use dioxus::prelude::*;

use crate::icons::icon;

/// A labelled range control.
///
/// `glyph` is an `Option` because the brush editor's dense parameter list has not been
/// marked yet, **not** because a slider is expected to go without one. A control's mark
/// is the half of it that survives its label, so anything reachable wants one; a `None`
/// here is a row that would be blank if the words were hidden, and is a to-do rather
/// than a decision (see [`crate::icons::SIZE`]).
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
        div { class: "slider-row",
            div { class: "slider-label",
                if let Some(glyph) = glyph {
                    {icon(glyph)}
                }
                "{label}"
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

// --- command dispatch ---
