//! The floating Select panel: selection tool, combine mode, and feather
//! (DESIGN.md §6.8).

use dioxus::html::Modifiers;
use dioxus::prelude::*;

use crate::layout::chrome_class;
use crate::state::{AppState, dispatch};
use crate::widgets::Slider;
use stark_core::command::{DocCommand, ViewCommand};
use stark_core::document::{SelectionMode, SelectionOp, Tool};

/// Selection tools (DESIGN.md §6.8): rect / ellipse / lasso, how the next gesture
/// combines with the current selection, and the feather applied to it.
///
/// The tool chips **arm** a tool rather than selecting a mode to stay in: one of
/// them is lit only while a selection gesture is pending, and drawing a selection
/// disarms it ([`Session::end_selection`](stark_core::session::Session::end_selection)).
/// Clicking the lit one disarms it too, so the escape hatch from an armed tool is the
/// same control that armed it. Painting is therefore the resting state and needs no
/// chip of its own — no chip lit *is* the brush.
///
/// The mode chips set a *default*; holding shift / alt while starting a drag
/// overrides it for that gesture (see [`modifier_mode`]), which is how this is
/// reached in practice once the tool is in hand.
#[component]
pub fn SelectPanel() -> Element {
    let state = use_context::<AppState>();
    let obs = state.obs.read();
    let (tool, mode, feather, show_peers) = obs
        .as_ref()
        .map(|o| {
            (
                o.tool,
                o.selection_mode,
                o.selection_feather,
                o.show_peer_selections,
            )
        })
        .unwrap_or((Tool::Brush, SelectionMode::Replace, 0.0, false));
    drop(obs);
    // The peer-outline toggle is mounted only in a shared session, on the same
    // argument the selection bar is mounted only while a selection is in force
    // (DESIGN.md §6.8): a control that is present or absent says whether the thing
    // it governs exists, which a permanently-visible one greyed out does not.
    // Keyed on the *session*, not on whether anyone is currently here, so it does
    // not flicker as collaborators come and go.
    let shared = (state.collab.phase)() == crate::collab::CollabPhase::Shared;

    let chip = |on: bool| if on { "chip active" } else { "chip" };
    const TOOLS: [(Tool, &str); 3] = [
        (Tool::SelectRect, "Rect"),
        (Tool::SelectEllipse, "Ellipse"),
        (Tool::SelectLasso, "Lasso"),
    ];
    const MODES: [(SelectionMode, &str); 4] = [
        (SelectionMode::Replace, "New"),
        (SelectionMode::Union, "Add"),
        (SelectionMode::Subtract, "Sub"),
        (SelectionMode::Intersect, "\u{2229}"),
    ];

    rsx! {
        div { class: "tool-row",
            for (t, label) in TOOLS {
                button {
                    class: chip(tool == t),
                    title: "Draw a selection with this tool, once",
                    // Clicking the armed tool disarms it: the control that armed it is
                    // the one you reach for to take it back.
                    onclick: move |_| {
                        let next = if tool == t { Tool::Brush } else { t };
                        dispatch(state, ViewCommand::SetTool(next));
                    },
                    "{label}"
                }
            }
        }
        div { class: "tool-row",
            for (m, label) in MODES {
                button {
                    class: chip(mode == m),
                    title: "Hold shift to add, alt to subtract, both to intersect",
                    onclick: move |_| dispatch(state, ViewCommand::SetSelectionMode(m)),
                    "{label}"
                }
            }
        }
        Slider { label: "Feather", min: 0.0, max: 64.0, value: feather,
            oninput: move |v| dispatch(state, ViewCommand::SetSelectionFeather(v)) }

        if shared {
            div { class: "row peer-toggle",
                input {
                    id: "show-peer-selections",
                    r#type: "checkbox",
                    checked: show_peers,
                    onchange: move |_| {
                        dispatch(state, ViewCommand::SetShowPeerSelections(!show_peers))
                    },
                }
                label {
                    r#for: "show-peer-selections",
                    title: "Outline the regions your collaborators have selected, \
                            each in their own colour",
                    "Show others' selections"
                }
            }
        }
    }
}

/// A small floating bar carrying the two commands that act on a **whole** selection
/// (DESIGN.md §6.8). Mounted only while one is in force, which is the point: those
/// commands are meaningless without a selection, and a bar that is simply present or
/// absent says the canvas is masked more directly than a pair of permanently-visible
/// buttons that happen to be greyed out — and without spending panel space the rest
/// of the time.
///
/// Positioned by the shared `.bottom-bars` column in `main`, which it shares with
/// the frame bar (built on the same argument) so the two stack rather than overlap.
#[component]
pub fn SelectionBar() -> Element {
    let state = use_context::<AppState>();
    // The committed selection, not the in-flight preview — so the bar does not flicker
    // in and out under a drag that has not been released yet.
    let active = state.obs.read().as_ref().is_some_and(|o| o.has_selection);
    // While a transform gesture is composing, its bar stands in for this one: the
    // whole-selection commands would fight the gesture (deselecting mid-transform
    // would move the wrong region on "Done").
    let transforming = state.transform.read().is_some();

    rsx! {
        if active && !transforming {
            div { class: chrome_class(state, "selection-bar"),
                span { class: "bar-label", "Selection" }
                button {
                    class: "chip",
                    title: "Move, scale or flip the selected paint (TRANSFORM_DESIGN.md)",
                    onclick: move |_| crate::panels::transform::begin_transform(state),
                    "Transform"
                }
                button {
                    class: "chip",
                    title: "Deselect (Ctrl+D)",
                    onclick: move |_| {
                        dispatch(state, DocCommand::Select(SelectionOp::select_all()))
                    },
                    "Deselect"
                }
                button {
                    class: "chip",
                    title: "Invert selection (Ctrl+Shift+I)",
                    onclick: move |_| dispatch(state, DocCommand::InvertSelection),
                    "Invert"
                }
            }
        }
    }
}

/// The selection mode a gesture's modifier keys ask for, or `None` to keep the
/// panel's default. Mirrors the conventional marquee modifiers.
pub fn modifier_mode(m: Modifiers) -> Option<SelectionMode> {
    match (m.contains(Modifiers::SHIFT), m.contains(Modifiers::ALT)) {
        (true, true) => Some(SelectionMode::Intersect),
        (true, false) => Some(SelectionMode::Union),
        (false, true) => Some(SelectionMode::Subtract),
        (false, false) => None,
    }
}

/// The tool the next canvas gesture will use.
pub fn current_tool(state: AppState) -> Tool {
    state.obs.peek().as_ref().map_or(Tool::Brush, |o| o.tool)
}

/// The selection mode the panel currently has set (the base a gesture's modifiers
/// override).
pub fn current_mode(state: AppState) -> SelectionMode {
    state
        .obs
        .peek()
        .as_ref()
        .map_or(SelectionMode::Replace, |o| o.selection_mode)
}
