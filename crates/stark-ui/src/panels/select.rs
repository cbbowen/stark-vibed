//! The floating Select panel: selection tool, combine mode, and feather
//! (DESIGN.md §6.8).

use dioxus::html::Modifiers;
use dioxus::prelude::*;

use crate::state::{AppState, dispatch};
use crate::widgets::Slider;
use stark_core::command::{DocCommand, ViewCommand};
use stark_core::document::{SelectionMode, SelectionOp, Tool};

/// Selection tools (DESIGN.md §6.8): rect / ellipse / lasso, how the next gesture
/// combines with the current selection, and the feather applied to it.
///
/// The mode chips set a *default*; holding shift / alt while starting a drag
/// overrides it for that gesture (see [`modifier_mode`]), which is how this is
/// reached in practice once the tool is in hand.
#[component]
pub fn SelectPanel() -> Element {
    let state = use_context::<AppState>();
    let obs = state.obs.read();
    let (tool, mode, feather, active) = obs
        .as_ref()
        .map(|o| {
            (
                o.tool,
                o.selection_mode,
                o.selection_feather,
                o.has_selection,
            )
        })
        .unwrap_or((Tool::Brush, SelectionMode::Replace, 0.0, false));
    drop(obs);

    let chip = |on: bool| if on { "chip active" } else { "chip" };
    const TOOLS: [(Tool, &str); 4] = [
        (Tool::Brush, "Paint"),
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
                    onclick: move |_| dispatch(state, ViewCommand::SetTool(t)),
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
        div { class: "tool-row",
            button {
                class: "chip",
                disabled: !active,
                onclick: move |_| dispatch(state, DocCommand::Select(SelectionOp::select_all())),
                "Deselect"
            }
            button {
                class: "chip",
                disabled: !active,
                onclick: move |_| dispatch(state, DocCommand::InvertSelection),
                "Invert"
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
