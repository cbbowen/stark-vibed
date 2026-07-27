//! The floating Layers panel: the layer stack, with per-layer opacity, visibility
//! and blend mode (DESIGN.md §6, step 6a).

use dioxus::prelude::*;

use crate::panels::frame::AddFrameButton;
use crate::render::PeerInfo;
use crate::state::{AppState, dispatch};
use stark_core::LayerInfo;
use stark_core::command::{DocCommand, PeerCommand};

#[component]
pub fn LayerPanel() -> Element {
    let state = use_context::<AppState>();
    let obs = state.obs.read();
    let layers = obs.as_ref().map(|o| o.layers.clone()).unwrap_or_default();
    // The properties that belong to *whichever* layer is selected live here, once,
    // rather than being repeated per row and again in the frame bar. A frame is a
    // layer, so it needs no copies of its own (FRAME_DESIGN.md §7).
    let selected = obs
        .as_ref()
        .and_then(|o| o.layers.iter().find(|l| l.id == o.active_layer).copied());
    drop(obs);
    // Removing the last layer would leave a document with nothing to paint on and
    // no way for the active layer to fall back, so the floor is one.
    let can_remove = layers.len() > 1;

    rsx! {
        div { class: "layer-header",
            // A frame is a layer, so making one belongs here rather than in a
            // panel of its own (FRAME_DESIGN.md §7).
            AddFrameButton {}
            button {
                class: "layer-add",
                title: "Add a paint layer",
                onclick: move |_| dispatch(state, DocCommand::AddLayer { above: None }),
                "+ Layer"
            }
            button {
                class: "layer-add layer-remove",
                title: if can_remove { "Remove the selected layer" }
                       else { "A document needs at least one layer" },
                disabled: !can_remove || selected.is_none(),
                onclick: move |_| {
                    if let Some(l) = selected {
                        dispatch(state, DocCommand::RemoveLayer(l.id));
                    }
                },
                "\u{2212} Remove"
            }
        }

        if let Some(l) = selected {
            div { class: "slider-row",
                div { class: "slider-label", "Opacity" }
                input {
                    class: "slider",
                    r#type: "range", min: "0", max: "100", step: "any",
                    value: "{(l.opacity * 100.0) as i32}",
                    title: if l.is_paintable() { "Opacity of the selected layer" }
                           else { "Frame opacity \u{2014} drag down to see through it while composing" },
                    oninput: move |e| {
                        if let Ok(v) = e.value().parse::<f32>() {
                            dispatch(state, DocCommand::SetLayerOpacity(l.id, v / 100.0));
                        }
                    },
                }
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

    // A row is now one line — visibility, then the name that selects it. The
    // per-layer opacity slider moved to the panel's single set of controls for
    // whatever is selected, so the inner flex wrapper it needed is gone too.
    rsx! {
        div {
            class: "{row_class} row",
            input {
                r#type: "checkbox",
                title: "Show this layer",
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
                onclick: move |_| dispatch(state, PeerCommand::SetActiveLayer(info.id)),
                if matte { "\u{25F1} Frame" } else { "Layer {info.id.ordinal()}" }
            }
            // Who else is working here (PEER_DESIGN.md §4). The selected layer is
            // per-client, so this is the only place that answers "am I about to
            // paint over what someone else is doing?" before it happens.
            for peer in peers_on(state, info.id) {
                div {
                    class: "peer-chip",
                    style: "background:{peer.css_color()}",
                    title: "{peer.name} is working on this layer",
                    "{peer.initials()}"
                }
            }
        }
    }
}

/// The collaborators whose selected layer is `id`.
fn peers_on(state: AppState, id: stark_core::LayerId) -> Vec<PeerInfo> {
    state
        .collab
        .peers
        .read()
        .iter()
        .filter(|p| p.active_layer == id)
        .cloned()
        .collect()
}
