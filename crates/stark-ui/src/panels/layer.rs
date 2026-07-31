//! The floating Layers panel: the layer tree, with per-layer opacity, visibility,
//! blend mode and clipping (DESIGN.md §6 step 6a, GROUP_DESIGN.md §6).
//!
//! The tree is drawn the way clipping masks are drawn everywhere: **the base at the
//! bottom, what it carries indented above it**. That picture is already how a
//! painter reads a clipping group in Photoshop; here it is simply the truth, because
//! a group *is* the layer at its base (GROUP_DESIGN.md §2).
//!
//! The one thing a Photoshop refugee has to unlearn is that the indent means
//! clipping. Here indent means **membership** and the rail down the left of a row
//! means **clipping**, and they are drawn as different marks because they are
//! different facts — a layer can be in a group without being clipped to it, which is
//! a state Photoshop's panel cannot draw at all.

use std::collections::HashSet;

use dioxus::html::Key;
use dioxus::prelude::*;

use crate::icons::{self, icon};
use crate::panels::frame::AddFrameButton;
use crate::platform::select_all;
use crate::render::PeerInfo;
use crate::state::{AppState, dispatch};
use stark_core::command::{DocCommand, PeerCommand};
use stark_core::document::BlendMode;
use stark_core::{LayerId, LayerInfo};

/// A row as the panel draws it: the layer, plus what its neighbours in the flat
/// list say about it that the layer alone cannot.
#[derive(Clone, PartialEq)]
pub struct Row {
    pub info: LayerInfo,
    /// Collapsed away under a group whose triangle is shut.
    hidden: bool,
    /// Shut, for a group. Meaningless for a layer that carries nothing.
    collapsed: bool,
}

#[component]
pub fn LayerPanel() -> Element {
    let state = use_context::<AppState>();
    // Which groups are shut. Panel-local view state — which is the whole point of
    // it not being in the document: whether *you* have a group folded away is not
    // part of the painting, is not saved, and is not something a collaborator
    // should see happen to their panel.
    let mut collapsed = use_signal(HashSet::<LayerId>::new);

    let obs = state.obs.read();
    let layers = obs.as_ref().map(|o| o.layers.clone()).unwrap_or_default();
    // The properties that belong to *whichever* layer is selected live here, once,
    // rather than being repeated per row and again in the frame bar. A frame is a
    // layer, so it needs no copies of its own (FRAME_DESIGN.md §7).
    let selected = obs
        .as_ref()
        .and_then(|o| o.layers.iter().find(|l| l.id == o.active_layer).cloned());
    drop(obs);
    let shut = collapsed.read().clone();
    let rows = rows(&layers, &shut);

    // `LayerInfo` carries the layer's name now, so it is `Clone` rather than `Copy`
    // and cannot be read again after a handler has moved it. The id is all most
    // handlers here want, and it still copies.
    let selected_id = selected.as_ref().map(|l| l.id);
    // Where "Add layer" puts one: into the selected layer's own stack, above it.
    // Read out here rather than in the handler because the row block below consumes
    // `selected`, and this is the only part of it the handler wants.
    let add_at = (
        selected.as_ref().map(|l| l.carrier).unwrap_or(None),
        selected_id,
    );
    // Removing a group takes what it carries with it (GROUP_DESIGN.md §2), so the
    // floor is not "more than one row" but "something would be left".
    let can_remove = selected_id.is_some_and(|id| subtree_len(&layers, id) < layers.len());
    // Carry puts the selection onto the layer below it *in its own stack*, which is
    // the layer it would be clipped to. Nothing below it in its stack, nothing to
    // carry it.
    let carry_onto = selected.as_ref().and_then(|l| sibling_below(&layers, l));
    // Release lifts it out of its group, to sit directly above the group.
    let release_to = selected
        .as_ref()
        .and_then(|l| l.carrier)
        .map(|c| (c, carrier_of(&layers, c)));

    rsx! {
        div { class: "layer-header",
            // A frame is a layer, so making one belongs here rather than in a
            // panel of its own (FRAME_DESIGN.md §7).
            AddFrameButton {}
            button {
                class: "layer-add",
                title: "Add a paint layer above the selected one",
                // Into the selected layer's own stack, not always the document's:
                // adding a layer while working inside a group should land in that
                // group, which is where you are looking.
                onclick: move |_| {
                    let (carrier, above) = add_at;
                    dispatch(state, DocCommand::AddLayer { carrier, above });
                },
                {icon(icons::ADD_LAYER)}
                "Layer"
            }
            button {
                class: "layer-add layer-remove",
                title: if can_remove { "Remove the selected layer, and anything it carries" }
                       else { "A document needs at least one layer" },
                disabled: !can_remove,
                onclick: move |_| {
                    if let Some(id) = selected_id {
                        dispatch(state, DocCommand::RemoveLayer(id));
                    }
                },
                // The `−` this replaces was the mirror of "+ Layer" in text; the two
                // glyphs are the same mirror, drawn — one stack gaining a member, the
                // other losing one. Which is also why Remove wears a *stack* glyph
                // though a frame is removable too: it takes the selected layer away,
                // and a frame is one.
                {icon(icons::REMOVE_LAYER)}
                "Remove"
            }
        }

        // Grouping, in the two words it takes. There is no third command: "clip to
        // the layer below" is Carry followed by the Clip toggle, because clipping to
        // exactly one layer *is* that layer carrying this one (GROUP_DESIGN.md §4).
        div { class: "layer-header",
            button {
                class: "layer-add",
                title: match &carry_onto {
                    Some(_) => "Put this layer on the one below it \u{2014} they become a group",
                    None => "Nothing below this layer in its stack to put it on",
                },
                disabled: carry_onto.is_none(),
                onclick: move |_| {
                    if let (Some(id), Some(onto)) = (selected_id, carry_onto) {
                        dispatch(state, DocCommand::MoveLayer {
                            id,
                            carrier: Some(onto),
                            above: None,
                        });
                    }
                },
                "Carry"
            }
            button {
                class: "layer-add",
                title: match &release_to {
                    Some(_) => "Lift this layer out of its group",
                    None => "This layer is not in a group",
                },
                disabled: release_to.is_none(),
                onclick: move |_| {
                    if let (Some(id), Some((group, outer))) = (selected_id, release_to) {
                        dispatch(state, DocCommand::MoveLayer {
                            id,
                            carrier: outer,
                            above: Some(group),
                        });
                    }
                },
                "Release"
            }
        }

        if let Some(l) = selected {
            div { class: "slider-row",
                div { class: "slider-label",
                    if l.is_group { "Opacity \u{2014} of the group" } else { "Opacity" }
                }
                input {
                    class: "slider",
                    r#type: "range", min: "0", max: "100", step: "any",
                    value: "{(l.opacity * 100.0) as i32}",
                    title: "{opacity_hint(&l)}",
                    oninput: move |e| {
                        if let Ok(v) = e.value().parse::<f32>() {
                            dispatch(state, DocCommand::SetLayerOpacity(l.id, v / 100.0));
                        }
                    },
                }
            }
            div { class: "slider-row",
                div { class: "slider-label",
                    if l.is_group { "Blend \u{2014} of the group" } else { "Blend" }
                }
                select {
                    class: "select",
                    // The mode's own description, so the difference between the two
                    // light modes is readable without painting a test stroke.
                    title: "{blend_hint(l.blend, &l)}",
                    // Inert at the bottom of the document, where there is nothing to
                    // blend with and every mode is the identity (GROUP_DESIGN.md
                    // §4.3). Shown rather than hidden: the control belongs to the
                    // layer wherever it sits, and a row that loses a control when it
                    // is dragged to the bottom reads as a bug.
                    disabled: !l.has_backdrop,
                    onchange: move |e| {
                        if let Some(m) = BlendMode::ALL.iter().find(|m| m.label() == e.value()) {
                            dispatch(state, DocCommand::SetLayerBlend(l.id, *m));
                        }
                    },
                    for mode in BlendMode::ALL {
                        option {
                            value: "{mode.label()}",
                            selected: mode == l.blend,
                            "{mode.label()}"
                        }
                    }
                }
            }
            label {
                class: "row layer-clip-row",
                title: "{clip_hint(&l)}",
                input {
                    r#type: "checkbox",
                    checked: l.clip,
                    // Inert for the same reason the blend picker is — except that
                    // where a mode over nothing is harmlessly the identity, a clip
                    // over nothing would erase the layer, which is the whole reason
                    // this one has to be stopped rather than merely left to do
                    // nothing (GROUP_DESIGN.md §4.3).
                    disabled: !l.has_backdrop,
                    onchange: move |_| dispatch(state, DocCommand::SetLayerClip(l.id, !l.clip)),
                }
                span { class: "slider-label layer-clip-label",
                    if l.is_group { "Clip the group to what is under it" } else { "Clip to the paint below" }
                }
            }
        }

        // Top of the document first, which is what a stack looks like from in front
        // of it — and within a group, what it carries above its base.
        for row in rows.iter().rev().filter(|r| !r.hidden).cloned() {
            LayerRow {
                row: row.clone(),
                ontoggle: move |id| {
                    let mut shut = collapsed.write();
                    if !shut.remove(&id) {
                        shut.insert(id);
                    }
                },
            }
        }
    }
}

/// The flat list decorated with what the panel needs and the projection does not
/// carry: which rows are folded away under a shut group.
///
/// Walks bottom-to-top, the order `observe()` produces, keeping the depth at which
/// the enclosing group was shut. Everything deeper than that is hidden until the
/// walk comes back out — which is exactly "hidden iff some ancestor is collapsed",
/// computed in one pass without ever looking a parent up.
fn rows(layers: &[LayerInfo], collapsed: &HashSet<LayerId>) -> Vec<Row> {
    let mut out = Vec::with_capacity(layers.len());
    let mut shut_at: Option<usize> = None;
    for info in layers {
        if shut_at.is_some_and(|d| info.depth <= d) {
            shut_at = None;
        }
        let hidden = shut_at.is_some();
        let collapsed = collapsed.contains(&info.id);
        if !hidden && collapsed && info.is_group {
            shut_at = Some(info.depth);
        }
        out.push(Row {
            info: info.clone(),
            hidden,
            collapsed,
        });
    }
    out
}

/// The layer directly below `layer` in its own stack — the one Carry would put it
/// on, and the top of the stack a clip would inherit from.
fn sibling_below(layers: &[LayerInfo], layer: &LayerInfo) -> Option<LayerId> {
    // The last layer sharing its carrier before it appears — i.e. its nearest
    // sibling below, which is also the top of the stack a clip would inherit from.
    layers
        .iter()
        .take_while(|l| l.id != layer.id)
        .filter(|l| l.carrier == layer.carrier)
        .last()
        .map(|l| l.id)
}

/// What carries `id`, from the flat list.
fn carrier_of(layers: &[LayerInfo], id: LayerId) -> Option<LayerId> {
    layers.iter().find(|l| l.id == id).and_then(|l| l.carrier)
}

/// How many rows `id` takes with it if removed: itself, plus everything it carries
/// at any depth. Those are exactly the rows that follow it while deeper than it.
fn subtree_len(layers: &[LayerInfo], id: LayerId) -> usize {
    let Some(at) = layers.iter().position(|l| l.id == id) else {
        return 0;
    };
    let depth = layers[at].depth;
    1 + layers[at + 1..]
        .iter()
        .take_while(|l| l.depth > depth)
        .count()
}

/// What a blend mode does, in one line, for the picker's tooltip.
///
/// Here rather than beside [`BlendMode`] for the same reason [`layer_label`] is: the
/// mode's *name* is part of what it is and travels with the document, but how you
/// explain it to someone hovering a drop-down is a frontend's business. The core
/// says "Glow"; deciding that a painter wants to hear "cannot blow out" rather than
/// "conjugate of addition under `x/(1+x)`" is a presentation call.
fn blend_hint(mode: BlendMode, layer: &LayerInfo) -> &'static str {
    // The two cases where the control is not saying what it usually says come
    // first, because they are about *this row* rather than about the mode.
    if !layer.has_backdrop {
        return "Nothing composites under this layer, so every mode looks the same here.";
    }
    if layer.is_group {
        return match mode {
            BlendMode::Normal => "This group sits on top of what is below it.",
            _ => {
                "How this group \u{2014} everything it carries, composited \u{2014} \
                  meets what is below it."
            }
        };
    }
    match mode {
        BlendMode::Normal => "The layer sits on top of what is below it.",
        BlendMode::Reinhard => {
            "Combines light instead of covering it \u{2014} softer than Screen, and it \
             cannot blow out however deep you stack it. For glazes, mist and rim light."
        }
        BlendMode::Drago => {
            "Combines light on a log curve \u{2014} hotter, and where two lights coincide \
             it pushes past white into the highlight roll-off. For flame and speculars."
        }
        BlendMode::Multiply => {
            "Takes light away instead of adding it, the way stacked glazes do \u{2014} \
             white leaves the layer below alone, black hides it. For shadows and tinting."
        }
    }
}

/// What the opacity slider fades, in one line.
///
/// Three answers, and the first is the one worth having: on a group, opacity is the
/// property that could *not* be borrowed from the base the way blend and clip are
/// (GROUP_DESIGN.md §3), so it fades the base and everything it carries as one unit.
fn opacity_hint(layer: &LayerInfo) -> &'static str {
    if layer.is_group {
        "Fades this layer and everything it carries, as one"
    } else if layer.is_paintable() {
        "Opacity of the selected layer"
    } else {
        "Frame opacity \u{2014} drag down to see through it while composing"
    }
}

/// What clipping would do to *this* layer, in one line.
///
/// Three different sentences, because the control means three different things
/// depending on where the row sits — and the difference is the part users get wrong
/// everywhere else (GROUP_DESIGN.md §4).
fn clip_hint(layer: &LayerInfo) -> &'static str {
    if !layer.has_backdrop {
        return "Nothing composites under this layer, so clipping it would leave nothing \
                to show.";
    }
    if layer.is_group {
        return "Clip: this group shows only where there is paint under the group.";
    }
    match layer.carrier {
        // Inside a group the bound is the group, which is the whole reason groups
        // and clipping are one feature rather than two.
        Some(_) => {
            "Clip: show only where there is paint under this layer *within its group* \
             \u{2014} the whole stack below it, not just the one layer."
        }
        None => {
            "Clip: show only where there is paint under this layer. To clip to one \
             layer alone, Carry it onto that layer first."
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
pub fn LayerRow(row: Row, ontoggle: EventHandler<LayerId>) -> Element {
    let state = use_context::<AppState>();
    let info = row.info.clone();
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
    // Membership is an indent; clipping is a rail. Two marks, because they are two
    // facts (GROUP_DESIGN.md §6) — and a row can wear one without the other, which
    // is the state Photoshop's single arrow cannot express.
    let row_class = if info.clip {
        format!("{row_class} clipped")
    } else {
        row_class.to_string()
    };
    let indent = info.depth * 14;
    let is_group = info.is_group;
    let collapsed = row.collapsed;

    let title = if matte {
        "Compose this frame — double-click to rename"
    } else {
        "Paint on this layer — double-click to rename"
    };

    // A row is one line — the group's triangle if it has one, visibility, then the
    // name that selects it. The per-layer opacity slider lives in the panel's single
    // set of controls for whatever is selected.
    rsx! {
        div {
            class: "{row_class} row",
            style: "margin-left:{indent}px",
            // Only a group gets a triangle, and the space is held either way so the
            // checkboxes down the panel stay in one column.
            if is_group {
                button {
                    class: "layer-fold",
                    title: if collapsed { "Show what this layer carries" }
                           else { "Fold away what this layer carries" },
                    onclick: move |_| ontoggle.call(id),
                    if collapsed { "\u{25B8}" } else { "\u{25BE}" }
                }
            } else {
                span { class: "layer-fold" }
            }
            input {
                r#type: "checkbox",
                title: if is_group { "Show this layer and what it carries" }
                       else { "Show this layer" },
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
