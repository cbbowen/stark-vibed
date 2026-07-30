//! The floating Select panel: shape tool, what the shape *does*, and feather
//! (DESIGN.md §6.8, MISSING_FEATURES §0.4).

use dioxus::html::Modifiers;
use dioxus::prelude::*;

use crate::layout::chrome_class;
use crate::state::{AppState, dispatch};
use crate::widgets::Slider;
use stark_core::command::{DocCommand, ViewCommand};
use stark_core::document::{FillOp, SelectionMode, SelectionOp, ShapeAction, Tool};

/// Shape tools (DESIGN.md §6.8): rect / ellipse / lasso, what the next gesture does
/// with the region they enclose, and the feather applied to its edge.
///
/// The tool chips **arm** a tool rather than selecting a mode to stay in: one of
/// them is lit only while a shape gesture is pending, and drawing a selection
/// disarms it ([`Session::end_shape`](stark_core::session::Session::end_shape)).
/// Clicking the lit one disarms it too, so the escape hatch from an armed tool is the
/// same control that armed it. Painting is therefore the resting state and needs no
/// chip of its own — no chip lit *is* the brush.
///
/// The **action** row is five answers to one question — *what does this shape do?* —
/// rather than four ways of combining plus an odd one out. Rect, ellipse and lasso
/// never produced selections; they produce **coverage**, and the four combine modes
/// are only the four ways that coverage can land on the mask. `Fill` lands it on the
/// paint instead (MISSING_FEATURES §0.4), and everything the row already had comes
/// with it: the same shapes, the same rasterizer, the same feather slider below.
///
/// Two consequences the row does not have to explain, because they follow:
///
/// - a fill is still **clipped by the selection**, since the mask gates every tool;
/// - and unlike the four selecting actions, Fill **stays armed** after a gesture.
///   The momentary rule is about a gesture that is a step *towards* painting; a fill
///   *is* painting, and blocking in is done many times in a row.
///
/// The four selecting actions also set a *default* that shift / alt override for one
/// gesture (see [`modifier_mode`]) — the modifiers are inert under Fill, which has
/// no combining to do.
#[component]
pub fn SelectPanel() -> Element {
    let state = use_context::<AppState>();
    let obs = state.obs.read();
    let (tool, action, feather, show_peers, brush_color) = obs
        .as_ref()
        .map(|o| {
            (
                o.tool,
                o.shape_action,
                o.selection_feather,
                o.show_peer_selections,
                o.brush.color,
            )
        })
        .unwrap_or((Tool::Brush, ShapeAction::default(), 0.0, false, [0.0; 4]));
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
    // Five words, no glyph: `∩` was the weak link in this row when it had four
    // entries, and it could not survive being one of five — a symbol reads as a
    // different *kind* of control from its neighbours, which is exactly the wrong
    // signal here, where the whole point is that all five answer one question.
    const ACTIONS: [(ShapeAction, &str, &str); 5] = [
        (
            ShapeAction::Select(SelectionMode::Replace),
            "New",
            "Select this region, replacing the current selection",
        ),
        (
            ShapeAction::Select(SelectionMode::Union),
            "Add",
            "Add this region to the selection (or hold shift)",
        ),
        (
            ShapeAction::Select(SelectionMode::Subtract),
            "Sub",
            "Cut this region out of the selection (or hold alt)",
        ),
        (
            ShapeAction::Select(SelectionMode::Intersect),
            "Isect",
            "Keep only the overlap with the selection (or hold shift+alt)",
        ),
        (
            ShapeAction::Fill,
            "Fill",
            "Fill this region with the brush's paint instead of selecting it. \
             Stays armed, so you can keep blocking in",
        ),
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
            for (a, label, hint) in ACTIONS {
                button {
                    // Fill carries a swatch of the colour it would lay, so the row
                    // itself says what the gesture will deposit — the one thing
                    // that distinguishes this action from its four neighbours and
                    // the one thing a word cannot carry.
                    class: chip(action == a),
                    title: "{hint}",
                    onclick: move |_| dispatch(state, ViewCommand::SetShapeAction(a)),
                    if a == ShapeAction::Fill {
                        span { class: "chip-swatch", style: swatch_style(brush_color) }
                    }
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
                // The other reach for the same word. With a selection in force the
                // region is already drawn, so a fill needs no gesture at all —
                // which is also the one case `FillOp::of_selection` is defined for:
                // the mask is what bounds it, and this bar exists only when there
                // is one.
                button {
                    class: "chip",
                    title: "Fill the selection with the brush's paint",
                    onclick: move |_| fill_selection(state),
                    "Fill"
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

/// Fill whatever is selected, on the active layer, with the brush's paint
/// (MISSING_FEATURES §0.4).
///
/// Reads the colour and the amount off the brush rather than off controls of its
/// own, which is the same choice [`ShapeAction::Fill`] makes: a fill lays the paint
/// you have in hand, so the Color and Brush panels are already its settings.
pub fn fill_selection(state: AppState) {
    let Some((layer, brush)) = state.obs.peek().as_ref().map(|o| (o.active_layer, o.brush)) else {
        return;
    };
    dispatch(
        state,
        DocCommand::Fill {
            layer,
            op: FillOp::of_selection(brush.color, brush.dynamics.add),
        },
    );
}

/// The inline swatch on the Fill chip: the colour the gesture would lay, at the
/// opacity it would lay it — a thin wash reads as a thin wash here too.
///
/// A `background-image` rather than a `background-color`, so it paints *over* the
/// white base the stylesheet gives the swatch and the alpha reads against white
/// instead of against the chip's dark ground.
fn swatch_style(color: [f32; 4]) -> String {
    let c = |i: usize| (color[i] * 255.0).round().clamp(0.0, 255.0) as u8;
    let paint = format!("rgba({}, {}, {}, {})", c(0), c(1), c(2), color[3]);
    format!("background-image: linear-gradient({paint}, {paint})")
}

/// The selection mode a gesture's modifier keys ask for, or `None` to keep the
/// panel's default. Mirrors the conventional marquee modifiers.
///
/// Consulted only when the panel's action is a *selecting* one
/// ([`ShapeAction::is_select`]): under Fill there is nothing to combine, so shift
/// and alt mean nothing rather than silently turning a fill back into a selection.
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

/// What the panel currently has the next shape gesture set to do (the base a
/// gesture's modifiers override).
pub fn current_action(state: AppState) -> ShapeAction {
    state
        .obs
        .peek()
        .as_ref()
        .map_or(ShapeAction::default(), |o| o.shape_action)
}
