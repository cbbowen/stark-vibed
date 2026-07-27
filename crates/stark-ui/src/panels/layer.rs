//! The floating Layers panel: the layer stack, with per-layer opacity, visibility
//! and blend mode (DESIGN.md §6, step 6a).

use dioxus::prelude::*;

use crate::panels::frame::AddFrameButton;
use crate::state::{AppState, dispatch};
use stark_core::LayerInfo;
use stark_core::command::{DocCommand, ViewCommand};

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
            // A frame is a layer, so making one belongs here rather than in a
            // panel of its own (FRAME_DESIGN.md §7).
            AddFrameButton {}
            button {
                class: "layer-add",
                onclick: move |_| dispatch(state, DocCommand::AddLayer { above: None }),
                "+ Layer"
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
    // One selection, one highlight. A matte is selected exactly the way a paint
    // layer is (FRAME_DESIGN.md §7) — selecting it raises the frame bar and its
    // on-canvas handles, and the brush simply has nowhere to go until a paint layer
    // is selected again. Because there is only one thing to highlight, "exactly one
    // row is highlighted" is a consequence rather than a rule to keep.
    let active = state
        .obs
        .read()
        .as_ref()
        .is_some_and(|o| o.active_layer == info.id);
    let matte = info.matte.is_some();
    let row_class = match (matte, active) {
        (true, true) => "layer-row matte active",
        (true, false) => "layer-row matte",
        (false, true) => "layer-row active",
        (false, false) => "layer-row",
    };

    rsx! {
        div {
            class: row_class,
            div { class: "row",
                input {
                    r#type: "checkbox",
                    checked: info.visible,
                    onchange: move |_| dispatch(state, DocCommand::SetLayerVisible(info.id, !info.visible)),
                }
                button {
                    class: if matte { "layer-name layer-name-matte" } else { "layer-name" },
                    title: if matte {
                        "Compose this frame — shows its handles and controls"
                    } else {
                        "Paint on this layer"
                    },
                    onclick: move |_| dispatch(state, ViewCommand::SetActiveLayer(info.id)),
                    if matte { "\u{25F1} Frame" } else { "Layer {info.id.0}" }
                }
            }
            input {
                class: "slider",
                r#type: "range", min: "0", max: "100",
                value: "{(info.opacity * 100.0) as i32}",
                oninput: move |e| {
                    if let Ok(v) = e.value().parse::<f32>() {
                        dispatch(state, DocCommand::SetLayerOpacity(info.id, v / 100.0));
                    }
                },
            }
        }
    }
}
