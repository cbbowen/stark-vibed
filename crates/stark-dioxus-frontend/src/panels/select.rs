//! The floating Select panel: shape tool and the feather it strikes, what the
//! shape *does*, and how strongly a fill lands (§6.8, §18.0.4) — and the
//! selection bar, which carries the whole mask's opacity and the acts on it.

use crate::commands;
use dioxus::prelude::*;
use stark_engine::command::Tool;
use stark_model::Srgb;
use stark_ui::icons::Icon;

use crate::icons::{icon, icon_tinted, label};
use crate::layout::chrome_dimmed;
use crate::preview;
use crate::state::{AppState, dispatch, use_obs};
use crate::widgets::{CommandButton, Slider, slider_fill};
use stark_engine::command::{DocCommand, ViewCommand};
use stark_model::document::{FillOp, ShapeAction};
use stark_ui::commands::Command;

/// Shape tools (§6.8): rect / ellipse / lasso, what the next gesture does
/// with the region they enclose, and the feather applied to its edge.
///
/// The tool chips **arm** a tool rather than selecting a mode to stay in: one of
/// them is lit only while a shape gesture is pending, and drawing a selection
/// disarms it ([`Session::end_shape`](stark_engine::Session::end_shape)).
/// Clicking the lit one disarms it too, so the escape hatch from an armed tool is the
/// same control that armed it. Painting is therefore the resting state and needs no
/// chip of its own — no chip lit *is* the brush.
///
/// All three of those sentences are the **registry's** now, not this panel's
/// (`crate::commands`): each chip is a `Command` worn whole, so the same act is
/// reached from the chip, from the search palette and from a chord (R / E / L),
/// and the lit state a chip shows is the one [`commands::active`](crate::commands::active) answers. The
/// panel keeps what is genuinely its own — that these three sit in one
/// `.segmented` run, with the feather they will strike beneath them, above the
/// row saying what their region *does*.
///
/// The **action** row is five answers to one question — *what does this shape do?* —
/// rather than four ways of combining plus an odd one out. Rect, ellipse and lasso
/// never produced selections; they produce **coverage**, and the four combine modes
/// are only the four ways that coverage can land on the mask. `Fill` lands it on the
/// paint instead (§18.0.4), and everything the row already had comes
/// with it: the same shapes, the same rasterizer, the same feather slider.
///
/// Picking any of the five also **hands back a shape tool** to draw the region
/// with ([`pick_action`]), since all five are answers about a gesture that has not
/// been made yet.
///
/// Two consequences the row does not have to explain, because they follow:
///
/// - a fill is still **clipped by the selection**, since the mask gates every tool;
/// - and unlike the four selecting actions, Fill **stays armed** after a gesture.
///   The momentary rule is about a gesture that is a step *towards* painting; a fill
///   *is* painting, and blocking in is done many times in a row.
///
/// The four selecting actions also set a *default* that shift / alt override for one
/// gesture (see [`stark_ui::selection::modifier_mode`]) — the modifiers are inert
/// under Fill, which has no combining to do.
///
/// Add is the one whose word the mask algebra would not have honoured: a union with
/// the unrestricted selection is the unrestricted selection, so on a fresh document
/// the chip did nothing at all. The gesture resolves it to New instead
/// (`Session::start_selection`, §6.8), which is why the chip's tooltip says so and
/// the row still reads as five answers to one question.
#[component]
pub fn SelectPanel() -> Element {
    let state = use_context::<AppState>();
    // One memo over everything the panel shows (`state::use_obs`). All four are
    // *tool* state — what the next gesture will do — so they move together and
    // never at pointer rate; read straight off `obs` the panel re-rendered on
    // every pan and every sample of the stroke it is describing.
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
    // The hand's color, off the frontend's own brush signal — a fill lays it
    // whatever effect the brush held has (`BrushConfig::color`).
    let brush_color = (state.transient)().color;

    let chip = |on: bool| if on { "chip active" } else { "chip" };
    // *Which* tool is armed is deliberately not in the memo above: the three
    // chips are `CommandButton`s, and each carries its own answer
    // ([`commands::active`](crate::commands::active)) — so moving the light from rect to ellipse
    // re-renders two buttons rather than the panel and its sliders. *Whether*
    // one is armed is in it, because the Feather row is mounted on that; it
    // flips on an arm and on the gesture's disarm, never between chips.
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
    /// The mark and the prose for each of `stark_ui::selection::SHAPE_ACTIONS`,
    /// in that order — which is where the row's *membership* and its words live now,
    /// since both frontends draw it. What stays here is what is this one's: an inline
    /// SVG, and a sentence with room to be a sentence.
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
        // Under the tool row, and mounted only while one of the three is in
        // hand: feather is the edge the rasterizer strikes, so it is chosen
        // *before* the gesture — a fact about the gesture the armed tool is
        // about to make — and it is shown as one, next to that tool, for
        // exactly as long as the gesture is pending. Under Fill the tool stays
        // armed after a gesture (§18.0.4), so the feather stays with it; under
        // the four selecting actions the gesture's disarm takes it away along
        // with the tool.
        if armed {
            Slider { label: "Feather", glyph: stark_ui::icons::FEATHER, min: 0.0, max: 64.0, value: feather,
                oninput: move |v| dispatch(state, ViewCommand::SetSelectionFeather(v)) }
        }
        div { class: "tool-row stacked segmented",
            for (a, (glyph, hint)) in stark_ui::selection::SHAPE_ACTIONS.into_iter().zip(MARKS) {
                button {
                    class: chip(action == a),
                    title: "{hint}",
                    onclick: move |_| pick_action(state, a),
                    // Fill's bucket is *full of* the color it would lay, so the row
                    // says what the gesture will deposit — the one thing that
                    // distinguishes this action from its four neighbours, and the one
                    // thing a word cannot carry. The bucket already draws a vessel with
                    // paint in it, so coloring that is one mark doing both jobs rather
                    // than a separate swatch beside the word splitting them — and the
                    // row keeps five glyphs on one baseline.
                    if a == ShapeAction::Fill {
                        // The wash's strength is the marquee fill's own opacity —
                        // the slider below — since the brush color is three channels
                        // of pigment and nothing about amount (§6.2).
                        {icon_tinted(glyph, [brush_color[0], brush_color[1], brush_color[2], fill_opacity])}
                    } else {
                        {icon(glyph)}
                    }
                    {label(stark_ui::selection::action_word(a))}
                }
            }
        }
        // The fill's own opacity (§18.0.4), and Feather's counterpart for that
        // one action: *how soft at the edge*, then *how strong* — both chosen up
        // front, with the gesture's other settings, because paint once laid is
        // paint. Mounted under Fill alone. Under the four selecting actions the
        // same question — how strongly does this coverage land — is asked of
        // the *mask*, and that answer is the selection bar's slider: the whole
        // mask's opacity, one number on top of the shape arithmetic, set after
        // the region is drawn and reaching a region already drawn (§6.8). Two
        // places for one question because the answers are given at different
        // times, and each sits where its time is: this one with the gesture's
        // settings, the mask's with the acts on the whole selection.
        if filling {
            Slider { label: "Opacity", glyph: stark_ui::icons::OPACITY, min: 0.0, max: 1.0, value: fill_opacity,
                oninput: move |v| dispatch(state, ViewCommand::SetShapeOpacity(v)) }
        }
    }
}

/// A small floating bar for the **whole** selection (§6.8): its opacity, and the
/// commands that act on all of it. Mounted while one is in force — and from the
/// moment a shape tool is armed, since that is a selection about to be made: the
/// bar stands ready, its commands greyed until there is a mask to act on and its
/// opacity already live so the strength can be set before the region is drawn,
/// and the marquee drawn under it lights the rest rather than raising a bar under
/// the hand. Nothing else mounts it, which keeps the point: with the brush in hand
/// and no mask there is no bar, so a bar being present says the canvas is (or is
/// about to be) masked more directly than a row of permanently-visible buttons
/// that happen to be greyed out — and without spending panel space the rest of
/// the time.
///
/// Positioned by the shared `.bottom-bars` column in `main`, which it shares with
/// the frame bar (built on the same argument) so the two stack rather than overlap.
#[component]
pub fn SelectionBar() -> Element {
    let state = use_context::<AppState>();
    // The committed selection, not the in-flight preview — so the bar's controls
    // do not light and grey under a drag that has not been released yet. Read
    // with it: whether a tool is armed, and the mask's opacity — the one number
    // here that *does* preview (`PreviewSelectionOpacity`; the engine reports
    // the previewed value back, so the track follows the pointer instead of
    // snapping to the committed number under it).
    //
    // Through a memo (`state::use_obs`): all three move on a commit, an arm or a
    // slider drag, never per pointer sample of the marquee drag the bar is
    // standing over — so re-rendering on each sample would be to re-diff a bar
    // whose answer has not changed.
    //
    // The bar's Fill lays the same paint the panel's Fill chip does, so it carries the
    // same loaded bucket: the brush's color is a property of the *act*, not of the
    // panel that happens to host the control.
    let shown = use_obs(state, |o| {
        (o.has_selection, o.tool.is_selection(), o.selection_opacity)
    });
    let (has_selection, armed, opacity) = shown().unwrap_or((false, false, 1.0));
    let active = has_selection || armed;
    // What a settled drag of the mask's opacity would lay (`preview::settle`).
    let dimming = use_signal(|| None::<f32>);
    let brush_color = (state.transient)().color;
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
                class: "selection-bar chrome",
                class: if chrome_dimmed(state) { "dimmed" },
                class: if composing { "recessed" },
                // The Select panel's own mark, on the bar the panel's gestures raise —
                // the bar is *this panel's* state made visible, so it says so with the
                // panel's glyph rather than a second picture of a marquee.
                span { class: "bar-label",
                    {icon(stark_ui::icons::SELECTION)}
                    {label("Selection")}
                }

                span { class: "bar-sep" }

                // The mask's opacity: how strongly the selection gates every tool
                // inside it, in the units the brush's own dial is quoted in — the
                // mask is the other factor of the opacity ceiling (§6.2, §6.8),
                // so a half-dimmed selection is a half-opacity brush, fill and
                // eraser inside it, the same picture the Brush panel's slider at
                // a half would make. That is also what the two whole-selection
                // fills on this bar read: they lay opaque paint *through* the
                // mask, so this one slider governs them without their having a
                // knob of their own.
                //
                // On the bar rather than in the Select panel because of *when*
                // it is set: after the region is drawn, on a region already
                // drawn — the one selection number that is, which is what makes
                // it document state (`DocCommand::SetSelectionOpacity`) where
                // the feather and a fill's opacity are the gesture's own. So it
                // lives with the acts on the whole selection — but unlike them
                // it is live with nothing selected: the number is then the
                // strength the next region will take, and the whole canvas's
                // until one is drawn (`Selection::opacity`), which is what lets
                // it be set up front while the tool is armed. A deselect lands
                // it back on 1 (`Selection::plan`), so a dimming never outlives
                // its selection by accident.
                //
                // Previewed while dragging and committed once on release
                // (`preview::SELECTION_OPACITY`): no pixel changes until
                // something paints through the mask, but it is document state,
                // so a drag that logged every value it crossed would spend an
                // undo step per pointer move on an adjustment the hand made once.
                span { class: "bar-sub",
                    {icon(stark_ui::icons::OPACITY)}
                    {label("Opacity")}
                }
                input {
                    class: "slider",
                    style: slider_fill(0.0, 1.0, opacity),
                    r#type: "range", min: "0", max: "1", step: "any",
                    value: "{opacity}",
                    title: "How strongly the selection takes paint \u{2014} a half-dimmed \
                            selection is a half-opacity brush, fill and eraser inside it",
                    oninput: move |e| {
                        if let Ok(v) = e.value().parse::<f32>() {
                            preview::SELECTION_OPACITY.during(state, dimming, v);
                        }
                    },
                    // All three, because none of them alone ends every drag
                    // (`Preview::settle`, which is idempotent for this reason).
                    onchange: move |_| preview::SELECTION_OPACITY.settle(state, dimming),
                    onpointerup: move |_| preview::SELECTION_OPACITY.settle(state, dimming),
                    onpointercancel: move |_| preview::SELECTION_OPACITY.settle(state, dimming),
                }

                span { class: "bar-sep" }

                // Each chip is its command worn whole (`crate::commands`): the
                // word, the mark, the tooltip with its advertised chord, the
                // gate the act asks and the greyed state while there is nothing
                // selected are all the registry's, so the bar cannot say one
                // thing about an act the menu says another about.
                CommandButton { command: Command::Transform }
                // The float's click route (§16.12): the same cut the pinned
                // drag commits on its first travel (`input::carry`), for a hand
                // that wants the layer without having to move it yet.
                CommandButton { command: Command::FloatSelection }
                // The other reach for Fill's word. With a selection in force the
                // region is already drawn, so a fill needs no gesture at all —
                // which is also the one case `FillOp::of_selection` is defined for:
                // the mask is what bounds it, and this bar exists only when there
                // is one.
                // Hand-written where its neighbours are `CommandButton`s, for the
                // tint alone: the bucket wears the brush's own paint
                // (`icons::icon_tinted`), which is the one thing here the
                // registry cannot say. The words still come off the command,
                // and so does the greyed state — `Command::FillSelection.enabled`
                // is `has_selection`, read here off the bar's own memo rather
                // than the projection so this button, like its neighbours, does
                // not re-render at pointer rate.
                button {
                    class: "chip",
                    disabled: !has_selection,
                    title: Command::FillSelection.tooltip(&state.bindings.read()),
                    onclick: move |_| commands::run(Command::FillSelection, state),
                    // At full strength: this fill's coverage is the mask's own
                    // (`FillOp::of_selection`), so there is no thinner wash to show.
                    {icon_tinted(stark_ui::icons::PAINT_BUCKET, [brush_color[0], brush_color[1], brush_color[2], 1.0])}
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

/// Cut what the selection holds on the active layer into a floating child
/// layer (§16.12). The engine plans before it spends an action, so a cut that
/// would be empty — or a layer that is not paint — declines with nothing
/// logged (`plan_float`).
pub fn float_selection(state: AppState) {
    let Some(layer) = state.obs.peek().as_ref().map(|o| o.active_layer) else {
        return;
    };
    dispatch(state, DocCommand::FloatSelection { layer });
}

/// Pick what the next shape gesture does — the action row's whole act, which is
/// more than the setter it sends (§6.8).
///
/// Every one of the five answers is about a region that has not been drawn yet, so
/// picking one and being left holding the brush answers nothing: the row would sit
/// there lit on "Add" while the next gesture painted. So the pick arms the tool the
/// gesture needs, taking which of the three from the chips above
/// ([`commands::arm_shape_tool`](crate::commands::arm_shape_tool)) — it is their
/// question, not this row's, and the momentary rule is what makes the brush the
/// common thing to be holding: every selecting gesture ends by handing the canvas
/// back (§6.8). Fill included, so the rule stays one sentence: Fill encloses its
/// region with the same three tools, and its staying armed afterwards is about what
/// a *gesture* leaves behind rather than about what a pick means.
///
/// **The chrome's half**, where the disarm that makes it necessary is the session's
/// — because the engine is sent [`ViewCommand::SetShapeAction`] and nothing else,
/// and the canvas sends that same command twice per modifier-held gesture: once to
/// override the action, once to put it back (`crate::input::paint`). That restore
/// lands *after* the gesture disarmed the tool, so an arm attached to the command
/// itself would re-arm on every shift-drag and quietly repeal the momentary rule.
/// What the frontend has that the command does not is the knowledge that a person
/// picked.
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
