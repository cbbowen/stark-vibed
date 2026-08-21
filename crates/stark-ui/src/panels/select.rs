//! The floating Select panel: shape tool, what the shape *does*, and feather
//! (§6.8, §18.0.4).

use dioxus::html::Modifiers;
use dioxus::prelude::*;

use crate::commands::Command;
use crate::icons::{self, icon, icon_tinted, label};
use crate::layout::chrome_class;
use crate::state::{AppState, dispatch, use_obs};
use crate::widgets::{CommandButton, Slider};
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::{FillOp, SelectionMode, ShapeAction, Tool};

/// Shape tools (§6.8): rect / ellipse / lasso, what the next gesture does
/// with the region they enclose, and the feather applied to its edge.
///
/// The tool chips **arm** a tool rather than selecting a mode to stay in: one of
/// them is lit only while a shape gesture is pending, and drawing a selection
/// disarms it ([`Session::end_shape`](stark_engine::session::Session::end_shape)).
/// Clicking the lit one disarms it too, so the escape hatch from an armed tool is the
/// same control that armed it. Painting is therefore the resting state and needs no
/// chip of its own — no chip lit *is* the brush.
///
/// All three of those sentences are the **registry's** now, not this panel's
/// (`crate::commands`): each chip is a `Command` worn whole, so the same act is
/// reached from the chip, from the search palette and from a chord (R / E / L),
/// and the lit state a chip shows is the one [`Command::active`] answers. The
/// panel keeps what is genuinely its own — that these three sit in one
/// `.segmented` run, above the row saying what their region *does*.
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
    // One memo over everything the panel shows (`state::use_obs`). All five are
    // *tool* state — what the next gesture will do — so they move together and
    // never at pointer rate; read straight off `obs` the panel re-rendered on
    // every pan and every sample of the stroke it is describing.
    let arm = use_obs(state, |o| {
        (
            o.shape_action,
            o.selection_feather,
            o.shape_opacity,
            o.brush.color,
        )
    });
    let (action, feather, opacity, brush_color) =
        arm().unwrap_or((ShapeAction::default(), 0.0, 1.0, [0.0; 4]));

    let chip = |on: bool| if on { "chip active" } else { "chip" };
    // Which tool is armed is deliberately *not* in the memo above: the three
    // chips are `CommandButton`s now, and each carries its own answer
    // ([`Command::active`]) — so arming a tool re-renders three buttons rather
    // than the panel and its two sliders.
    const TOOLS: [Command; 3] = [
        Command::SelectRect,
        Command::SelectEllipse,
        Command::SelectLasso,
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
            // Each chip is its command worn whole (`crate::commands`), the way
            // the bar below wears its five: the mark, the terse word, the
            // tooltip with its chord, the lit state and what a second click
            // means are all the registry's. Which is what buys these three a
            // keyboard — R / E / L reach the same act the chip does, and the
            // chip advertises the key that reaches it.
            for command in TOOLS {
                CommandButton { key: "{command:?}", command }
            }
        }
        div { class: "tool-row stacked segmented",
            for (a, glyph, word, hint) in ACTIONS {
                button {
                    class: chip(action == a),
                    title: "{hint}",
                    onclick: move |_| dispatch(state, ViewCommand::SetShapeAction(a)),
                    // Fill's bucket is *full of* the color it would lay, so the row
                    // says what the gesture will deposit — the one thing that
                    // distinguishes this action from its four neighbours, and the one
                    // thing a word cannot carry. The bucket already draws a vessel with
                    // paint in it, so coloring that is one mark doing both jobs rather
                    // than a separate swatch beside the word splitting them — and the
                    // row keeps five glyphs on one baseline.
                    if a == ShapeAction::Fill {
                        {icon_tinted(glyph, brush_color)}
                    } else {
                        {icon(glyph)}
                    }
                    {label(word)}
                }
            }
        }
        // Above Feather, and the exact counterpart of it: *how strong*, then *how
        // soft at the edge*. Both apply to whichever of the five actions the row is
        // set to, because both describe the coverage the gesture produces and the
        // five actions differ only in where that coverage lands (§6.8).
        //
        // Selecting, it dims the mask — and since every tool acts through the mask
        // in proportion, a half-strength selection is a half-strength brush, fill
        // and transform inside it. That is also what the two whole-selection fills
        // on the bar read: they lay opaque paint *through* the mask, so this one
        // slider governs them without their having a knob of their own.
        Slider { label: "Opacity", glyph: icons::OPACITY, min: 0.0, max: 1.0, value: opacity,
            oninput: move |v| dispatch(state, ViewCommand::SetShapeOpacity(v)) }
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
    //
    // Through a memo (`state::use_obs`): whether there is a selection changes on a
    // commit, and the bar is mounted on it — so re-rendering per pointer sample of
    // the very marquee drag it is waiting on would be to re-diff a bar that is not
    // there yet.
    //
    // The bar's Fill lays the same paint the panel's Fill chip does, so it carries the
    // same loaded bucket: the brush's color is a property of the *act*, not of the
    // panel that happens to host the control.
    let shown = use_obs(state, |o| (o.has_selection, o.brush.color));
    let (active, brush_color) = shown().unwrap_or((false, [0.0; 4]));
    // While any mode is composing, its own bar stands in for this one: the
    // whole-selection commands would fight the gesture (deselecting mid-transform
    // would move the wrong region on "Done"). Every mode, not the two that hold a
    // selection preview — a trace or a guide edit owns the canvas just as
    // completely, and a bar offering to fill through a catcher promises something
    // the pointer cannot reach (`crate::modes`).
    //
    // Stood down by **receding**, not unmounting (MODAL_DESIGN.md): the bar
    // stays on screen dimmed and inert behind the mode's own, so the place its
    // Done and Esc return to is visible the whole time. Inert twice over —
    // `.recessed` takes the pointer events, and every chip here runs an act
    // that asks `commands::may_edit`, which refuses while a mode is composing.
    let composing = crate::modes::composing(state).is_some();

    rsx! {
        if active {
            div {
                class: chrome_class(
                    state,
                    if composing { "selection-bar recessed" } else { "selection-bar" },
                ),
                // The Select panel's own mark, on the bar the panel's gestures raise —
                // the bar is *this panel's* state made visible, so it says so with the
                // panel's glyph rather than a second picture of a marquee.
                span { class: "bar-label",
                    {icon(icons::SELECTION)}
                    {label("Selection")}
                }

                span { class: "bar-sep" }

                // Each chip is its command worn whole (`crate::commands`): the
                // word, the mark, the tooltip with its advertised chord, and the
                // gate the act asks are all the registry's, so the bar cannot
                // say one thing about an act the menu says another about.
                CommandButton { command: Command::Transform }
                // The other reach for Fill's word. With a selection in force the
                // region is already drawn, so a fill needs no gesture at all —
                // which is also the one case `FillOp::of_selection` is defined for:
                // the mask is what bounds it, and this bar exists only when there
                // is one.
                // Hand-written where its neighbours are `CommandButton`s, for the
                // tint alone: the bucket wears the brush's own paint
                // (`icons::icon_tinted`), which is the one thing here the
                // registry cannot say. The words still come off the command.
                button {
                    class: "chip",
                    title: Command::FillSelection.tooltip(&state.bindings.read()),
                    onclick: move |_| Command::FillSelection.run(state),
                    {icon_tinted(icons::PAINT_BUCKET, brush_color)}
                    {label(Command::FillSelection.word())}
                }
                // The same act as Fill with the parcel varying along a dragged
                // axis (§22.4) — a mode rather than a click, because
                // the axis is composed by hand and judged by eye. Never
                // disabled: the mode's own bar carries the library (§22.3), so
                // it is the way to a first ramp as well as the way to lay one.
                CommandButton { command: Command::GradientFill }
                CommandButton { command: Command::InvertSelection }
                CommandButton { command: Command::Deselect }
            }
        }
    }
}

/// Fill whatever is selected, on the active layer, with the brush's paint
/// (§18.0.4).
///
/// The **color** comes off the brush, which is the choice [`ShapeAction::Fill`]
/// makes too: a fill lays the paint you have in hand, so the Color panel is
/// already its setting. How far it covers is not a question this button asks at
/// all — it fills the selection, so the selection's own coverage answers it
/// ([`FillOp::of_selection`]).
pub fn fill_selection(state: AppState) {
    let Some((layer, [r, g, b, _])) = state
        .obs
        .peek()
        .as_ref()
        .map(|o| (o.active_layer, o.brush.color))
    else {
        return;
    };
    dispatch(
        state,
        DocCommand::Fill {
            layer,
            op: FillOp::of_selection([r, g, b]),
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
