//! The floating Select panel: shape tool and the feather it strikes, what the
//! shape *does*, and how strongly a fill lands (§6.8, §18.0.4) — and the
//! selection bar, which carries the whole mask's opacity and the acts on it.

use crate::commands;
use dioxus::prelude::*;
use stark_engine::command::Tool;
use stark_model::Srgb;
use stark_ui::icons::Icon;

use crate::icons::{icon_tinted, label};
use crate::preview;
use crate::state::{AppState, dispatch, use_obs};
use crate::widgets::{
    Bar, Chip, Choice, CommandButton, Face, PreviewSlider, Segmented, Slider, SliderShape,
};
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::{FillOp, ShapeAction};
use stark_ui::commands::Command;
use stark_ui::selection::Dial;

/// Shape tools (§6.8): rect / ellipse / lasso, what the next gesture does with the region
/// they enclose, and the feather applied to its edge.
///
/// The tool chips **arm** a tool for one gesture: drawing a selection disarms it
/// ([`Session::end_shape`](stark_engine::Session::end_shape)), and no chip lit is the
/// brush. Each chip is a registry command (`crate::commands`).
///
/// The action row is five answers to *what does this shape do?* The four combine modes
/// land coverage on the mask; `Fill` lands it on the paint (§18.0.4), still clipped by the
/// selection, and unlike the other four stays armed after a gesture. Shift / alt override
/// the default for one gesture ([`stark_ui::selection::modifier_mode`]) and are inert
/// under Fill. Add on the unrestricted selection resolves to New
/// (`Session::start_selection`, §6.8).
#[component]
pub fn SelectPanel() -> Element {
    let state = use_context::<AppState>();
    // One memo (`state::use_obs`): this is tool state, which never moves at pointer rate.
    let arm = use_obs(state, |o| {
        (
            o.shape_action,
            o.selection_feather,
            o.shape_opacity,
            o.tool.is_selection(),
        )
    });
    let (action, feather, fill_opacity, armed) =
        arm().unwrap_or((ShapeAction::default(), 0.0, 1.0, false));
    // Whether the Opacity row is mounted — see the row itself.
    let filling = action == ShapeAction::Fill;
    // The hand's color: a fill lays it whatever the brush's effect (`BrushConfig::color`).
    let brush_color = (state.transient)().color;

    // *Which* tool is armed is not in the memo: each chip answers for itself
    // (`commands::active`), so moving the light re-renders two buttons. *Whether* one is
    // armed is, since the Feather row mounts on it.
    const TOOLS: [Command; 3] = [
        Command::SelectRect,
        Command::SelectEllipse,
        Command::SelectLasso,
    ];
    // A mark and its word on every entry: one bare glyph would read as a different kind of
    // control.
    /// The mark and the prose for each of `stark_ui::selection::SHAPE_ACTIONS`, in order.
    const MARKS: [(Icon, &str); 5] = [
        (
            stark_ui::icons::SELECTION_NEW,
            "Select this region, replacing the current selection",
        ),
        (
            stark_ui::icons::SELECTION_ADD,
            "Add this region to the selection (or hold shift). With nothing \n             selected, this selects just the region",
        ),
        (
            stark_ui::icons::SELECTION_SUB,
            "Cut this region out of the selection (or hold alt)",
        ),
        (
            stark_ui::icons::SELECTION_ISECT,
            "Keep only the overlap with the selection (or hold shift+alt)",
        ),
        (
            stark_ui::icons::PAINT_BUCKET,
            "Fill this region with the brush's paint instead of selecting it. \
             Stays armed, so you can keep blocking in",
        ),
    ];
    let actions: Vec<_> = stark_ui::selection::SHAPE_ACTIONS
        .into_iter()
        .zip(MARKS)
        .map(|(a, (glyph, hint))| {
            let word = stark_ui::selection::action_word(a);
            // Fill's bucket is full of the color it would lay, at the fill's own opacity: the brush
            // color carries no amount (§6.2).
            let face = if a == ShapeAction::Fill {
                Face::Drawn(rsx! {
                    {icon_tinted(glyph, [brush_color[0], brush_color[1], brush_color[2], fill_opacity])}
                    {label(word)}
                })
            } else {
                Face::Marked(glyph, word)
            };
            Choice::new(a, face, hint)
        })
        .collect();

    rsx! {
        // `stacked`: glyph over word, so five chips fit the panel without dropping their words.
        // An exclusive run of commands rather than of values, so not a `Segmented`.
        div { class: "tool-row stacked segmented",
            // Each chip is its command worn whole (`crate::commands`), so R / E / L reach the
            // same act.
            for command in TOOLS {
                CommandButton { key: "{command:?}", command }
            }
        }
        // Mounted only while a shape tool is armed: the feather is chosen before the gesture.
        // Under Fill the tool stays armed (§18.0.4), so the feather stays with it.
        if armed {
            Slider { label: Dial::Feather.label(), glyph: Dial::Feather.glyph(),
                min: Dial::Feather.range().0, max: Dial::Feather.range().1, value: feather,
                oninput: move |v| dispatch(state, ViewCommand::SetSelectionFeather(v)) }
        }
        Segmented {
            class: "tool-row stacked",
            choices: actions,
            selected: action,
            onpick: move |a: ShapeAction| pick_action(state, a),
        }
        // The fill's own opacity (§18.0.4), chosen before the gesture because paint once laid is
        // paint. For the selecting actions the same question is the mask's opacity, set after the
        // region is drawn, so it lives on the selection bar (§6.8).
        if filling {
            Slider { label: "Opacity", glyph: stark_ui::icons::OPACITY, min: 0.0, max: 1.0, value: fill_opacity,
                oninput: move |v| dispatch(state, ViewCommand::SetShapeOpacity(v)) }
        }
    }
}

/// A floating bar for the **whole** selection (§6.8): its opacity, and the commands that
/// act on all of it.
///
/// Mounted while a selection is in force or a shape tool is armed; its commands grey until
/// there is a mask, while its opacity is already live. Nothing else mounts it, so a bar on
/// screen says the canvas is (or is about to be) masked. Stacks with the frame bar in the
/// shared `.bottom-bars` column (`crate::app`).
#[component]
pub fn SelectionBar() -> Element {
    let state = use_context::<AppState>();
    // The committed selection, not the in-flight preview, so the controls do not light and
    // grey under an unreleased drag. The opacity *does* preview (`PreviewSelectionOpacity`):
    // the engine reports the previewed value back, so the track follows the pointer.
    //
    // One memo (`state::use_obs`): none of these moves per pointer sample of a marquee drag.
    let shown = use_obs(state, |o| {
        (o.has_selection, o.tool.is_selection(), o.selection_opacity)
    });
    let (has_selection, armed, opacity) = shown().unwrap_or((false, false, 1.0));
    let active = has_selection || armed;
    // What a settled drag of the mask's opacity would lay (`preview::settle`).
    let dimming = use_signal(|| None::<f32>);
    let brush_color = (state.transient)().color;

    rsx! {
        if active {
            // The Select panel's mark: this bar is that panel's state made visible.
            Bar {
                class: "selection-bar",
                glyph: stark_ui::icons::SELECTION,
                word: "Selection",

                span { class: "bar-sep" }

                // The mask's opacity: the other factor of the opacity ceiling (§6.2, §6.8), so
                // a half-dimmed selection is a half-opacity brush, fill and eraser inside it,
                // including this bar's two fills.
                //
                // On the bar because it is set after the region is drawn, which makes it
                // document state (`DocCommand::SetSelectionOpacity`). Live with nothing
                // selected: it is then the strength the next region takes
                // (`Selection::opacity`). A deselect resets it to 1 (`Selection::plan`).
                PreviewSlider {
                    shape: SliderShape::Bar,
                    label: "Opacity",
                    glyph: stark_ui::icons::OPACITY,
                    min: 0.0,
                    max: 1.0,
                    value: opacity,
                    title: "How strongly the selection takes paint \u{2014} a half-dimmed \
                            selection is a half-opacity brush, fill and eraser inside it",
                    preview: preview::SELECTION_OPACITY,
                    pending: dimming,
                    map: Some::<f32>,
                }

                span { class: "bar-sep" }

                // Each chip is its command worn whole (`crate::commands`), so the bar and the
                // menu cannot disagree about an act.
                CommandButton { command: Command::Transform }
                // The float's click route (§16.12): the cut the pinned drag commits on its
                // first travel (`input::carry`).
                CommandButton { command: Command::FloatSelection }
                // With a selection in force the region is already drawn, so Fill needs no
                // gesture (`FillOp::of_selection`). Hand-written for the tint alone
                // (`icons::icon_tinted`); the words and the greyed state still come off the
                // command, the latter read off this bar's memo so the button does not re-render
                // at pointer rate.
                Chip {
                    disabled: !has_selection,
                    title: Command::FillSelection.tooltip(&state.bindings.read()),
                    onclick: move |_| commands::run(Command::FillSelection, state),
                    // Full strength: this fill's coverage is the mask's own
                    // (`FillOp::of_selection`).
                    {icon_tinted(stark_ui::icons::PAINT_BUCKET, [brush_color[0], brush_color[1], brush_color[2], 1.0])}
                    {label(Command::FillSelection.word())}
                }
                // Fill with the parcel varying along a dragged axis (§22.4). Never disabled:
                // the mode's own bar carries the library (§22.3), so it is also the way to a
                // first ramp.
                CommandButton { command: Command::GradientFill }
                CommandButton { command: Command::InvertSelection }
                CommandButton { command: Command::Deselect }
            }
        }
    }
}

/// Fill whatever is selected, on the active layer, with the brush's paint (§18.0.4). The
/// selection's own coverage bounds it ([`FillOp::of_selection`]).
pub fn fill_selection(state: AppState) {
    let Some(layer) = state.obs.peek().as_ref().map(|o| o.active_layer) else {
        return;
    };
    let [r, g, b] = state.transient.peek().color;
    dispatch(
        state,
        DocCommand::Fill {
            layer,
            op: FillOp::of_selection(Srgb::new([r, g, b])),
        },
    );
}

/// Cut what the selection holds on the active layer into a floating child layer
/// (§16.12). A cut that would be empty, or a layer that is not paint, declines with
/// nothing logged (`plan_float`).
pub fn float_selection(state: AppState) {
    let Some(layer) = state.obs.peek().as_ref().map(|o| o.active_layer) else {
        return;
    };
    dispatch(state, DocCommand::FloatSelection { layer });
}

/// Pick what the next shape gesture does, and arm the shape tool it needs
/// ([`commands::arm_shape_tool`]): every answer is about a region not yet drawn (§6.8).
///
/// The arm is the frontend's rather than part of [`ViewCommand::SetShapeAction`]: the
/// canvas sends that command twice per modifier-held gesture, and the restore lands after
/// the gesture's disarm (`crate::input::paint`), so an arm there would re-arm on every
/// shift-drag.
fn pick_action(state: AppState, action: ShapeAction) {
    dispatch(state, ViewCommand::SetShapeAction(action));
    crate::commands::arm_shape_tool(state);
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
