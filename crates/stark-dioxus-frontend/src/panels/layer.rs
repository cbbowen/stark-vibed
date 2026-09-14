//! The floating Layers panel: the layer tree, with per-layer opacity, visibility,
//! blend mode and clipping (§6 step 6a, §14.6).
//!
//! The base sits at the bottom with what it carries indented above it, since a group
//! *is* the layer at its base (§14.2). Indent means **membership** and the rail down a
//! row's left means **clipping**: a layer can be in a group without being clipped to it.
//!
//! Moves live on the rows, so each names its own layer and is absent where it has nowhere
//! to go. Dragging a row is one move, since a drop lands in some stack at some place
//! (§14.8); see [`landing`] for what a drop means and [`Motion`] for what a row does.

use std::collections::HashSet;

use dioxus::prelude::*;

use crate::collab::css_color;
use crate::icons::{icon, label};
use crate::panels::filter::AddFilterButton;
use crate::panels::reorder::{Grip, RowKey};
use crate::preview;
use crate::state::{AppState, dispatch, use_obs};
use crate::widgets::{Chip, CommandButton, InlineRename, PreviewSlider, Select};
use stark_engine::LayerInfo;
use stark_engine::command::{DocCommand, PeerCommand};
use stark_model::document::{BlendMode, LayerId};
use stark_ui::collab::Peer;
use stark_ui::commands::Command;
use stark_ui::layer_tree::{
    self, BEND_HINT, INDENT, Row, bend_ends, blend_hint, clip_hint, landing, opacity_hint, rows,
};
use stark_ui::reorder::{Grab, Motion};

/// Add a paint layer above the selected one, in its own stack (`Command::AddLayer`), so
/// adding while working inside a group lands in that group.
pub fn add_layer(state: AppState) {
    let (carrier, above) = state
        .obs
        .peek()
        .as_ref()
        .map(|o| {
            let selected = o.layers.iter().find(|l| l.id == o.active_layer);
            (selected.and_then(|l| l.carrier), selected.map(|l| l.id))
        })
        .unwrap_or((None, None));
    dispatch(state, DocCommand::AddLayer { carrier, above });
}

/// Dispatch `add`, a command that adds one layer, and select the layer it added.
///
/// Found by diffing the roster (`stark_ui::mint`); `dispatch` refreshes the projection
/// before it returns. [`add_layer`] needs none of this: the engine selects a new paint
/// layer itself.
pub fn add_and_select(state: AppState, add: DocCommand) {
    let ids = |state: AppState| -> Vec<LayerId> {
        state
            .obs
            .peek()
            .as_ref()
            .map(|o| o.layers.iter().map(|l| l.id).collect())
            .unwrap_or_default()
    };
    let before = ids(state);
    dispatch(state, add);
    if let Some(id) = stark_ui::mint::minted(&before, ids(state)) {
        dispatch(state, PeerCommand::SetActiveLayer(id));
    }
}

#[component]
pub fn LayerPanel() -> Element {
    let state = use_context::<AppState>();
    // Which groups are shut: panel-local view state, neither saved nor shared.
    let mut collapsed = use_signal(HashSet::<LayerId>::new);
    // The in-flight row drag, delimited by the browser's own gesture so no timer can leave
    // it armed (§11).
    let drag = use_signal(|| None::<Grab>);

    // The tree and the selected row through one memo (`state::use_obs`): both move on a
    // commit, never on a pan or a stroke sample.
    let tree = use_obs(state, |o| {
        (
            o.layers.clone(),
            o.layers.iter().find(|l| l.id == o.active_layer).cloned(),
        )
    });
    let (layers, selected) = tree().unwrap_or_default();
    // Keep the rows' pictures up to date (§14.6; `crate::layer_thumbs`). Driven from this
    // panel, their only viewer, so a closed panel renders none. `refresh` is idempotent.
    use_effect(use_reactive!(|layers| {
        crate::layer_thumbs::prune(state, &layers);
        crate::layer_thumbs::refresh(state);
    }));
    let shut = collapsed.read().clone();
    let rows = rows(&layers, &shut);
    // The rows as shown (`layer_tree::display`): top first, folded rows left out. Drawing,
    // drag resolution and `landing` all use this one list, and `landing` needs it turned or
    // it resolves the mirror image of the right drop.
    let display = layer_tree::display(&rows);
    // Resolved once here, so rows that do not move do not re-render as the pointer travels.
    let land = drag
        .read()
        .as_ref()
        .filter(|d| d.live())
        .and_then(|d| landing(&display, d));

    // `LayerInfo` is not `Copy`; the id, which is all most handlers want, is.
    let selected_id = selected.as_ref().map(|l| l.id);

    // Whether blend and clip have anything to say about the selected layer
    // (`layer_tree::Row::blend_inert`). Off `rows`, not `display`, so a selection folded
    // away still answers.
    let picked = selected_id.and_then(|id| rows.iter().find(|r| r.info.id == id));
    let blend_inert = picked.is_none_or(Row::blend_inert);
    let clip_inert = picked.is_none_or(Row::clip_inert);
    rsx! {
        if let Some(layer) = selected {
            SelectedLayerControls { layer, blend_inert, clip_inert }
        }

        hr {}

        div { class: "layer-header",
            // A frame is a layer, so its add sits here beside the paint layer's (§15.7).
            CommandButton { command: Command::AddLayer, class: "layer-add" }
            // No "+ Background": the substrate is made at most once, so it is a frame-bar chip
            // (§15.5). A filter is the third kind of layer (§21); where it lands is what it
            // acts on.
            AddFilterButton {}
            CommandButton { command: Command::AddFrame, class: "layer-add" }
        }

        // Top of the document first. A well of its own (`.layer-tree`): the rows are the panel's
        // picture, not its controls.
        div { class: "layer-tree",
            for (i, row) in display.iter().copied().enumerate() {
                LayerRow {
                    // Keyed by the layer, so a reorder moves the row's element; positional keys
                    // would leave the click after a drop on whichever row took its place.
                    key: "{row.info.id}",
                    row: row.clone(),
                    motion: land.map_or_else(Motion::default, |l| l.motion(i)),
                    // The layer that would carry the drop: a fact about the landing's meaning,
                    // so not part of `motion`.
                    carrying: land.is_some_and(|l| l.carrier == Some(row.info.id)),
                    active: selected_id == Some(row.info.id),
                    drag,
                    ontoggle: move |id| {
                        let mut shut = collapsed.write();
                        if !shut.remove(&id) {
                            shut.insert(id);
                        }
                    },
                    onland: move |id: LayerId| {
                        // Dragging a layer selects it, drop or no drop; the gesture has taken
                        // the click.
                        dispatch(state, PeerCommand::SetActiveLayer(id));
                        let Some(l) = land.filter(|l| !l.inert) else {
                            return;
                        };
                        // Open a folded carrier, or the dropped layer would vanish into it.
                        if let Some(c) = l.carrier {
                            collapsed.write().remove(&c);
                        }
                        dispatch(state, l.move_layer(id));
                    },
                }
            }
        }
    }
}

/// The properties of whichever layer is selected, once: a frame is a layer, so it needs
/// no copies (§15.7). A component so its drags' pending values live as long as the
/// selection.
#[component]
fn SelectedLayerControls(layer: LayerInfo, blend_inert: bool, clip_inert: bool) -> Element {
    let state = use_context::<AppState>();
    // Pending drag values. The whole mode for Bend, since that is what `SetLayerBlend` takes.
    let fading = use_signal(|| None::<(LayerId, f32)>);
    let bending = use_signal(|| None::<(LayerId, BlendMode)>);
    let id = layer.id;
    let clip = layer.clip;
    let modes: Vec<&'static str> = BlendMode::ALL.iter().map(|m| m.label()).collect();
    // `same_mode`, not `==`: a Radiance layer with a dragged Bend is still on the Radiance
    // row, and re-picking it would reset the Bend.
    let mode = BlendMode::ALL.iter().position(|m| m.same_mode(layer.blend));
    rsx! {
        div { class: "slider-row marked",
            div { class: "slider-label",
                {icon(stark_ui::icons::BLEND)}
                {label(if layer.is_group { "Blend \u{2014} of the group" } else { "Blend" })}
            }
            // Blend and clip share a row: how this layer meets what is below it. On a filter the
            // mode goes inert, having no source, while clip stays live (§21.4).
            div { class: "row blend-row",
                // Titled with the mode's description. Inert rather than hidden at the bottom of the
                // document, where every mode is the identity (§14.4.3).
                Select {
                    title: blend_hint(layer.blend, &layer),
                    disabled: blend_inert,
                    options: modes,
                    selected: mode,
                    onchange: move |i: usize| {
                        dispatch(state, DocCommand::SetLayerBlend(id, BlendMode::ALL[i]));
                    },
                }
                // Inert only where there is nothing beneath, where a clip would erase the layer
                // (§14.4.3).
                Chip {
                    active: clip,
                    title: clip_hint(&layer),
                    disabled: clip_inert,
                    onclick: move |_| dispatch(state, DocCommand::SetLayerClip(id, !clip)),
                    {icon(stark_ui::icons::CLIP)}
                }
            }
        }
        // A filter's opacity is its **strength** (§21.4): half applied, not half transparent.
        // "— of the group" (§14.3) is inside the hideable word, so minimal mode leaves no half
        // sentence.
        PreviewSlider {
            label: match (layer.is_group, layer.filter.is_some()) {
                (true, _) => "Opacity \u{2014} of the group",
                (false, true) => "Strength",
                (false, false) => "Opacity",
            },
            glyph: stark_ui::icons::OPACITY,
            min: 0.0,
            max: 100.0,
            value: layer.opacity * 100.0,
            title: opacity_hint(&layer),
            preview: preview::LAYER_OPACITY,
            pending: fading,
            map: move |v: f32| Some((id, v / 100.0)),
        }
        // Radiance's own parameter (§18.0.4), shown only under that mode.
        if let BlendMode::Drago { k } = layer.blend {
            PreviewSlider {
                label: "Bend",
                glyph: stark_ui::icons::BEND,
                // In octaves of `k`: the bend is a scale, so a linear track would spend most of its
                // travel in the flat end.
                min: bend_ends().0,
                max: bend_ends().1,
                value: k.log2(),
                title: BEND_HINT,
                // Inert with its mode: a bend over nothing bends nothing.
                disabled: blend_inert,
                preview: preview::LAYER_BLEND,
                pending: bending,
                map: move |stops: f32| Some((id, BlendMode::Drago { k: stops.exp2() })),
            }
        }
    }
}

#[component]
pub fn LayerRow(
    row: Row,
    motion: Motion,
    carrying: bool,
    /// Whether this is the selected row. A prop, so rows do not each subscribe to the
    /// projection and re-render on every engine write.
    active: bool,
    drag: Signal<Option<Grab>>,
    ontoggle: EventHandler<LayerId>,
    onland: EventHandler<LayerId>,
) -> Element {
    let state = use_context::<AppState>();
    let info = row.info.clone();
    // Row-local, so opening one rename leaves every other row alone.
    let mut editing = use_signal(|| false);
    let id = info.id;
    // Read out before the handlers capture them: `LayerInfo` is not `Copy`.
    let visible = info.visible;
    let matte = info.matte.is_some();
    let filter = info.filter.is_some();
    let label = stark_ui::layer_tree::layer_label(&info);
    let seed = info.name.as_deref().unwrap_or_default().to_string();
    // This layer's paint in miniature (§14.6): no box for a layer with no tiles, an empty
    // one while the first render is pending so the row does not change shape under the
    // pointer. Subscribes, which is how a row learns its picture is ready.
    let thumb = info
        .content_revision
        .map(|_| crate::cards::thumb_style(crate::layer_thumbs::url(state, &info).as_deref()));
    let indent = info.depth * INDENT;
    let is_group = info.is_group;
    let collapsed = row.collapsed;
    // The row's own moves (§14.2), each the whole command as the tree spells it.
    let carry = row.carry();
    let release = row.release();
    let removable = row.removable;
    // The layer this one folds into, or `None` where no merge preserves the picture
    // (§14.11). Off the projection, so the button cannot offer a merge the engine declines.
    let merge_down = info.merge_down;
    // "Down" is not always the row below: a group's bottom member folds into its carrier
    // (§14.1). A filter bakes into the paint it adjusts instead (§14.11.7).
    let merge_title = if filter {
        "Bake this filter into the paint it is filtering \u{2014} the picture stays \
         the same, and the row goes"
    } else if merge_down.is_some() && merge_down == info.carrier {
        "Merge this layer into the one carrying it \u{2014} the picture stays the same"
    } else {
        "Merge this layer down into the one below \u{2014} the picture stays the same"
    };

    let title = if matte {
        "Compose this frame — double-click to rename"
    } else if filter {
        "Tune this filter — it adjusts everything below it in its own stack. \
         Double-click to rename"
    } else {
        "Paint on this layer — double-click to rename"
    };
    // A frame or a filter is a *what*, not a place to paint, so its name is dimmed.
    let name_class = if matte || filter {
        "layer-name layer-name-kind"
    } else {
        "layer-name"
    };

    // A row is one line of controls, with two marks outside it: the fold on its top edge and
    // Release in the indent. `Motion` writes every declaration on every render, including
    // the "off" ones (see `super::reorder::css`).
    let shift = super::reorder::css(motion);

    rsx! {
        // The indent is padding, since Release is drawn in it, and also `--indent` for the
        // stratum band that paints that gutter (`.layer-item::before`).
        div {
            class: "layer-item",
            class: if motion.lifted { "dragging" },
            style: "--indent:{indent}px; padding-left:{indent}px; {shift}",
            // For `platform::layer_boxes`: a drag matches DOM boxes to rows by id, not by order.
            "data-layer": "{id}",
            // Release, in the last step of the indent. `carrier` is `Some` exactly when `depth`
            // is at least one (both from one walk in `observe()`), so `indent - INDENT` is never
            // below zero.
            if let Some(release) = release {
                button {
                    class: "layer-release",
                    style: "left:{indent - INDENT}px",
                    title: "Lift this layer out of its group",
                    onclick: move |_| dispatch(state, release.clone()),
                    {icon(stark_ui::icons::RELEASE)}
                }
            }
            div {
                class: "layer-row row",
                // A frame is dashed (§15.7) and a filter ruled (§21.6): the brush has nowhere
                // to go.
                class: if matte { "matte" } else if filter { "filter" },
                // One selection, one highlight; a matte is selected like any layer (§15.7).
                class: if active { "active" },
                // Membership is an indent; clipping is a rail (§14.6).
                class: if info.clip { "clipped" },
                // The layer that would carry the drop: the one part of the landing the indent
                // leaves to inference.
                class: if carrying { "carrying" },
                // A group's fold, centred on the row's top edge and aimed at what it carries,
                // which is drawn above (§14.6); the two states differ by a lid (see
                // `stark_ui::icons::FOLD_OPEN`).
                if is_group {
                    button {
                        class: "layer-fold",
                        title: if collapsed { "Show what this layer carries" }
                               else { "Fold away what this layer carries" },
                        onclick: move |_| ontoggle.call(id),
                        {icon(if collapsed { stark_ui::icons::FOLD_SHUT } else { stark_ui::icons::FOLD_OPEN })}
                    }
                }
                // Carry: put this layer on the one below and they become a group; clipping to
                // one layer is Carry plus Clip (§14.4). The slot is held either way so names
                // align at each depth. Hidden until hover, with Release and the eye
                // (`.layer-item:hover` in `stark.css`).
                if let Some(carry) = carry {
                    button {
                        class: "layer-carry",
                        title: "Put this layer on the one below it \u{2014} they become a group",
                        onclick: move |_| dispatch(state, carry.clone()),
                        {icon(stark_ui::icons::CARRY)}
                    }
                } else {
                    span { class: "layer-carry" }
                }
                if editing() {
                    // The engine drops a rename to the name the layer already has, so
                    // leaving an untouched field spends no undo step.
                    InlineRename {
                        class: "layer-name layer-rename",
                        seed,
                        placeholder: label.to_string(),
                        oncommit: move |text: String| {
                            dispatch(state, DocCommand::SetLayerName(id, Some(text)));
                        },
                        onclose: move |_| editing.set(false),
                    }
                } else {
                    Grip {
                        class: name_class,
                        title,
                        row: RowKey::Layer(id),
                        drag,
                        onclick: move |_| dispatch(state, PeerCommand::SetActiveLayer(id)),
                        ondoubleclick: move |_| editing.set(true),
                        onland: move |_| onland.call(id),
                        "{label}"
                    }
                }
                // Who else has this layer selected (§17.4).
                for peer in peers_on(state, id) {
                    div {
                        class: "peer-chip",
                        style: "background:{css_color(&peer)}",
                        title: "{peer.name} is working on this layer",
                        "{peer.initials()}"
                    }
                }
                // Merge down (§14.11). Absent rather than inert where the pair cannot merge: a
                // merge that changed the picture would be a different edit, not a blocked one.
                // The engine answers the same question (`LayerInfo::merge_down`). The slot is
                // held so the eyes stay in one column.
                if merge_down.is_some() {
                    button {
                        class: "layer-merge",
                        title: "{merge_title}",
                        onclick: move |_| dispatch(state, DocCommand::MergeLayerDown(id)),
                        {icon(stark_ui::icons::MERGE_DOWN)}
                    }
                } else {
                    span { class: "layer-merge" }
                }
                // Duplicate (§14.8): a copy directly above, with its tiles, its name and what
                // it carries.
                button {
                    class: "layer-duplicate",
                    title: if is_group { "Duplicate this layer, and everything it carries" }
                           else { "Duplicate this layer" },
                    onclick: move |_| dispatch(state, DocCommand::DuplicateLayer(id)),
                    {icon(stark_ui::icons::DUPLICATE)}
                }
                // Remove: hidden until hover, and undoable (§5), which is its safety. Absent on
                // the row whose removal would empty the document, with its slot held.
                if removable {
                    button {
                        class: "layer-remove",
                        title: if is_group { "Remove this layer, and everything it carries" }
                               else { "Remove this layer" },
                        onclick: move |_| dispatch(state, DocCommand::RemoveLayer(id)),
                        {icon(stark_ui::icons::REMOVE)}
                    }
                } else {
                    span { class: "layer-remove" }
                }
                // The eye, last control on the line, so the eyes form one column at any depth.
                // It shows the state the layer is in (see `stark_ui::icons::VISIBLE`); an open
                // eye rests hidden until hover, so only struck eyes stand in the column.
                button {
                    class: if visible { "layer-eye" } else { "layer-eye hidden" },
                    title: match (is_group, visible) {
                        (true, true) => "Hide this layer and what it carries",
                        (true, false) => "Show this layer and what it carries",
                        (false, true) => "Hide this layer",
                        (false, false) => "Show this layer",
                    },
                    onclick: move |_| dispatch(state, DocCommand::SetLayerVisible(id, !visible)),
                    {icon(if visible { stark_ui::icons::VISIBLE } else { stark_ui::icons::HIDDEN })}
                }
                // What this layer is, in a slot flush with the row's right edge (§14.6): a
                // paint layer's own paint (`crate::layer_thumbs`), a frame's crop marks, a
                // filter's funnel (§21.3).
                //
                // A background `<div>` with `pointer-events: none`: the whole row is the grip,
                // and an element taking the press would leave a dead patch in it.
                if let Some(style) = thumb {
                    div { class: "layer-thumb", style: "{style}" }
                } else if matte {
                    div { class: "layer-thumb layer-thumb-kind", {icon(stark_ui::icons::FRAME)} }
                } else if filter {
                    div { class: "layer-thumb layer-thumb-kind", {icon(stark_ui::icons::FILTER)} }
                }
            }
        }
    }
}

/// The collaborators whose selected layer is `id`.
fn peers_on(state: AppState, id: stark_model::document::LayerId) -> Vec<Peer> {
    state
        .collab
        .peers
        .read()
        .iter()
        .filter(|p| p.active_layer == id)
        .cloned()
        .collect()
}
