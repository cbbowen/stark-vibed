//! The floating Layers panel: the layer stack, with per-layer opacity, visibility
//! and blend mode (DESIGN.md §6, step 6a).

use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use stark_core::{InputCommand, LayerInfo};

#[component]
pub fn LayerPanel() -> Element {
    let state = use_context::<AppState>();
    let layers = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.layers.clone())
        .unwrap_or_default();

    rsx! {
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

#[component]
pub fn LayerRow(info: LayerInfo) -> Element {
    let state = use_context::<AppState>();
    let active = state
        .obs
        .read()
        .as_ref()
        .map(|o| o.active_layer == info.id)
        .unwrap_or(false);
    let row_class = if active {
        "layer-row active"
    } else {
        "layer-row"
    };

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
