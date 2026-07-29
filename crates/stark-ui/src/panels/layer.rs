//! The floating Layers panel: the layer stack, with per-layer opacity, visibility
//! and blend mode (DESIGN.md §6, step 6a).

use dioxus::html::Key;
use dioxus::prelude::*;

use crate::panels::frame::AddFrameButton;
use crate::platform::select_all;
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
        .and_then(|o| o.layers.iter().find(|l| l.id == o.active_layer).cloned());
    drop(obs);
    // `LayerInfo` carries the layer's name now, so it is `Clone` rather than `Copy`
    // and cannot be read again after a handler has moved it. The id is all any
    // handler here wants, and it still copies.
    let selected_id = selected.as_ref().map(|l| l.id);
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
                disabled: !can_remove || selected_id.is_none(),
                onclick: move |_| {
                    if let Some(id) = selected_id {
                        dispatch(state, DocCommand::RemoveLayer(id));
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

/// What to call a layer that has never been named: its place in the stack, or what
/// it *is* when that says more (FRAME_DESIGN.md §7 — there is only ever one frame,
/// so numbering it would be noise).
///
/// Kept here rather than in the core because it is a way of *presenting* a stack,
/// not a fact about the document — which is exactly why an unnamed layer stores no
/// name (see [`LayerInfo::name`]). A layer the author has named shows that name,
/// frame or not.
fn layer_label(info: &LayerInfo) -> String {
    match (&info.name, info.matte.is_some()) {
        (Some(name), _) => name.to_string(),
        (None, true) => "\u{25F1} Frame".to_string(),
        (None, false) => format!("Layer {}", info.id.ordinal()),
    }
}

#[component]
pub fn LayerRow(info: LayerInfo) -> Element {
    let state = use_context::<AppState>();
    // The rename in progress on *this* row, or `None` while the row is just a row.
    // Row-local, so opening one leaves every other row alone and closing it needs
    // nothing cleaned up. The draft is held here rather than read back off the
    // field on commit because both commit paths — Enter and blur — need it, and one
    // of them fires while the field is on its way out.
    let mut draft = use_signal(|| None::<String>);
    let id = info.id;
    // Commit whatever the field holds, and close it. `take` is what makes the two
    // commit paths safe to both fire: whichever runs second finds no draft. Leaving
    // an untouched field costs nothing either — the engine drops a rename to the
    // name the layer already has, so no undo step is spent on it.
    let mut commit = move || {
        let text = draft.write().take();
        if let Some(text) = text {
            dispatch(state, DocCommand::SetLayerName(id, Some(text)));
        }
    };
    // The row's own fields, read out before the handlers below capture them:
    // `LayerInfo` is `Clone` rather than `Copy` now that it carries the name, and
    // several handlers want a piece of it.
    let visible = info.visible;
    let matte = info.matte.is_some();
    let label = layer_label(&info);
    // What the field opens on: the layer's *name*, which for one that has never been
    // named is empty. Deliberately not the label — seeding with the generated
    // "Layer 3" would turn opening the field and pressing Enter into a rename to
    // "Layer 3", quietly making a description into a name. The placeholder carries
    // the label instead, so the row still says what it is called while empty.
    let seed = info.name.as_deref().unwrap_or_default().to_string();
    // One selection, one highlight. A matte is selected exactly the way a paint
    // layer is (FRAME_DESIGN.md §7) — selecting it raises the frame bar and its
    // on-canvas handles, and the brush simply has nowhere to go until a paint layer
    // is selected again. Because there is only one thing to highlight, "exactly one
    // row is highlighted" is a consequence rather than a rule to keep.
    let active = state
        .obs
        .read()
        .as_ref()
        .is_some_and(|o| o.active_layer == id);
    let row_class = match (matte, active) {
        (true, true) => "layer-row matte active",
        (true, false) => "layer-row matte",
        (false, true) => "layer-row active",
        (false, false) => "layer-row",
    };

    let title = if matte {
        "Compose this frame — double-click to rename"
    } else {
        "Paint on this layer — double-click to rename"
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
                checked: visible,
                onchange: move |_| dispatch(state, DocCommand::SetLayerVisible(id, !visible)),
            }
            if let Some(text) = draft() {
                input {
                    class: "layer-name",
                    class: "layer-rename",
                    r#type: "text",
                    value: "{text}",
                    placeholder: "{label}",
                    // The field is the point of the double-click, so it takes focus
                    // as it appears rather than asking for a second click. The DOM
                    // node exists by the time `onmounted` runs, which is what the
                    // `autofocus` attribute cannot promise for an element inserted
                    // after load.
                    onmounted: move |e: Event<MountedData>| {
                        spawn(async move {
                            let _ = e.set_focus(true).await;
                            // Selected, not merely focused: the field opens on the
                            // name the layer already has, and the usual reason to
                            // open it is to replace that name rather than add to it.
                            // Typing over is one keystroke; keeping a word of it is
                            // one click. Ordered after the focus rather than left to
                            // `select`'s own — awaiting it is what puts the two in a
                            // known order.
                            select_all(&e);
                        });
                    },
                    oninput: move |e| draft.set(Some(e.value())),
                    // Committing on blur is what makes this feel like a label rather
                    // than a form: clicking away is an ordinary way to be finished,
                    // and nothing is lost by it.
                    //
                    // Enter commits directly rather than by blurring — a focused
                    // element that is removed does not reliably fire `blur` (the very
                    // thing `platform::on_window_key` exists to work around), so the
                    // field closing itself cannot be the commit. The two paths cannot
                    // double up: `commit` *takes* the draft, so whichever runs second
                    // finds nothing to send.
                    onblur: move |_| commit(),
                    // Everything else typed here is left alone: the global shortcuts
                    // already stand aside for a text field (`input::bind_shortcuts`),
                    // which is what leaves the browser's own Ctrl+Z editing this text
                    // instead of the document.
                    onkeydown: move |e| match e.key() {
                        Key::Enter => commit(),
                        // Escape abandons the edit — dropping the draft first, so the
                        // blur that follows the field's removal has nothing left to
                        // commit.
                        Key::Escape => draft.set(None),
                        _ => {}
                    },
                }
            } else {
                button {
                    class: if matte { "layer-name layer-name-matte" } else { "layer-name" },
                    title,
                    onclick: move |_| dispatch(state, PeerCommand::SetActiveLayer(id)),
                    ondoubleclick: move |_| draft.set(Some(seed.clone())),
                    "{label}"
                }
            }
            // Who else is working here (PEER_DESIGN.md §4). The selected layer is
            // per-client, so this is the only place that answers "am I about to
            // paint over what someone else is doing?" before it happens.
            for peer in peers_on(state, id) {
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
