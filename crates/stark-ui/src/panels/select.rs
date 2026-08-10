//! The floating Select panel: shape tool, what the shape *does*, and feather
//! (§6.8, §18.0.4).

use dioxus::html::Modifiers;
use dioxus::prelude::*;

use crate::icons::{self, icon, icon_tinted, label};
use crate::layout::chrome_class;
use crate::state::{AppState, dispatch};
use crate::widgets::Slider;
use stark_core::command::{DocCommand, ViewCommand};
use stark_core::document::{FillOp, SelectionMode, SelectionOp, ShapeAction, Tool};

/// Shape tools (§6.8): rect / ellipse / lasso, what the next gesture does
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
/// paint instead (§18.0.4), and everything the row already had comes
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
    let (tool, action, feather, brush_color) = obs
        .as_ref()
        .map(|o| (o.tool, o.shape_action, o.selection_feather, o.brush.color))
        .unwrap_or((Tool::Brush, ShapeAction::default(), 0.0, [0.0; 4]));
    drop(obs);

    let chip = |on: bool| if on { "chip active" } else { "chip" };
    const TOOLS: [(Tool, &str, &str); 3] = [
        (Tool::SelectRect, icons::RECTANGLE, "Rect"),
        (Tool::SelectEllipse, icons::CIRCLE, "Ellipse"),
        (Tool::SelectLasso, icons::LASSO, "Lasso"),
    ];
    // A glyph *and* its word on all five, never one without the other. `∩` was the
    // weak link in this row when it had four entries and could not have survived
    // being one of five: a lone symbol among words reads as a different *kind* of
    // control from its neighbours, which is exactly the wrong signal here, where the
    // whole point is that all five answer one question. The rule that follows from
    // that is about evenness rather than about glyphs — five icons over five words
    // keeps the row one row, and one entry left bare would break it the same way `∩`
    // did.
    const ACTIONS: [(ShapeAction, &str, &str, &str); 5] = [
        (
            ShapeAction::Select(SelectionMode::Replace),
            icons::SELECTION_NEW,
            "New",
            "Select this region, replacing the current selection",
        ),
        (
            ShapeAction::Select(SelectionMode::Union),
            icons::SELECTION_ADD,
            "Add",
            "Add this region to the selection (or hold shift)",
        ),
        (
            ShapeAction::Select(SelectionMode::Subtract),
            icons::SELECTION_SUB,
            "Sub",
            "Cut this region out of the selection (or hold alt)",
        ),
        (
            ShapeAction::Select(SelectionMode::Intersect),
            icons::SELECTION_ISECT,
            "Isect",
            "Keep only the overlap with the selection (or hold shift+alt)",
        ),
        (
            ShapeAction::Fill,
            icons::PAINT_BUCKET,
            "Fill",
            "Fill this region with the brush's paint instead of selecting it. \
             Stays armed, so you can keep blocking in",
        ),
    ];

    rsx! {
        // `stacked`: glyph over word, which is what buys the icons their room. Side by
        // side, five chips of icon-plus-word do not fit the panel's width and the words
        // would be the thing to give — and a word is the half of each chip that is
        // unambiguous, so it is not the half to drop.
        //
        // `segmented`: both rows are one question each — which tool, and what the shape
        // does — so both are drawn as one control with a lit region rather than as three
        // and five switches that happen to be adjacent. The rows differ in that at most
        // one tool chip is lit and exactly one action chip always is, but that difference
        // is about arming, not about combining; neither row can ever have two lit at
        // once, which is the thing the shape is claiming.
        div { class: "tool-row stacked segmented",
            for (t, glyph, word) in TOOLS {
                button {
                    class: chip(tool == t),
                    title: "Draw a selection with this tool, once",
                    // Clicking the armed tool disarms it: the control that armed it is
                    // the one you reach for to take it back.
                    onclick: move |_| {
                        let next = if tool == t { Tool::Brush } else { t };
                        dispatch(state, ViewCommand::SetTool(next));
                    },
                    {icon(glyph)}
                    {label(word)}
                }
            }
        }
        div { class: "tool-row stacked segmented",
            for (a, glyph, word, hint) in ACTIONS {
                button {
                    class: chip(action == a),
                    title: "{hint}",
                    onclick: move |_| dispatch(state, ViewCommand::SetShapeAction(a)),
                    // Fill's bucket is *full of* the colour it would lay, so the row
                    // says what the gesture will deposit — the one thing that
                    // distinguishes this action from its four neighbours, and the one
                    // thing a word cannot carry. It used to be a separate swatch beside
                    // the word; the bucket already draws a vessel with paint in it, so
                    // colouring that is one mark doing both jobs instead of two marks
                    // splitting them, and the row keeps five glyphs on one baseline.
                    if a == ShapeAction::Fill {
                        {icon_tinted(glyph, brush_color)}
                    } else {
                        {icon(glyph)}
                    }
                    {label(word)}
                }
            }
        }
        Slider { label: "Feather", glyph: icons::FEATHER, min: 0.0, max: 64.0, value: feather,
            oninput: move |v| dispatch(state, ViewCommand::SetSelectionFeather(v)) }
    }
}

/// A small floating bar carrying the two commands that act on a **whole** selection
/// (§6.8). Mounted only while one is in force, which is the point: those
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
    let obs = state.obs.read();
    let active = obs.as_ref().is_some_and(|o| o.has_selection);
    // The bar's Fill lays the same paint the panel's Fill chip does, so it carries the
    // same loaded bucket: the brush's colour is a property of the *act*, not of the
    // panel that happens to host the control.
    let brush_color = obs.as_ref().map_or([0.0; 4], |o| o.brush.color);
    drop(obs);
    // While a transform gesture is composing, its bar stands in for this one: the
    // whole-selection commands would fight the gesture (deselecting mid-transform
    // would move the wrong region on "Done").
    let transforming = state.transform.read().is_some();

    rsx! {
        if active && !transforming {
            div { class: chrome_class(state, "selection-bar"),
                // The Select panel's own mark, on the bar the panel's gestures raise —
                // the bar is *this panel's* state made visible, so it says so with the
                // panel's glyph rather than a second picture of a marquee.
                span { class: "bar-label",
                    {icon(icons::SELECTION)}
                    {label("Selection")}
                }

                span { class: "bar-sep" }

                // Arrows out of a centre, not a box of grips: the widget this opens is
                // an ellipse and has no handles to promise (see `icons::TRANSFORM`).
                button {
                    class: "chip",
                    title: "Move, scale, flip, perspective or warp the selected paint (§16)",
                    onclick: move |_| crate::panels::transform::begin_transform(state),
                    {icon(icons::TRANSFORM)}
                    {label("Transform")}
                }
                // The other reach for the same word. With a selection in force the
                // region is already drawn, so a fill needs no gesture at all —
                // which is also the one case `FillOp::of_selection` is defined for:
                // the mask is what bounds it, and this bar exists only when there
                // is one.
                // The same bucket the panel's Fill chip wears, because it is the same
                // command with its region already drawn — one glyph for one act, wherever
                // it is reached from.
                button {
                    class: "chip",
                    title: "Fill the selection with the brush's paint",
                    onclick: move |_| fill_selection(state),
                    {icon_tinted(icons::PAINT_BUCKET, brush_color)}
                    {label("Fill")}
                }
                button {
                    class: "chip",
                    title: "Invert selection (Ctrl+Shift+I)",
                    onclick: move |_| dispatch(state, DocCommand::InvertSelection),
                    {icon(icons::SELECTION_INVERT)}
                    {label("Invert")}
                }
                button {
                    class: "chip",
                    title: "Deselect (Ctrl+D)",
                    onclick: move |_| {
                        dispatch(state, DocCommand::Select(SelectionOp::select_all()))
                    },
                    {icon(icons::SELECTION_NONE)}
                    {label("Deselect")}
                }
            }
        }
    }
}

/// Fill whatever is selected, on the active layer, with the brush's paint
/// (§18.0.4).
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
