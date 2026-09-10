//! The Select section: which shape draws the next region and what that region does —
//! and, on a bar of its own, the acts on the selection once there is one (§6.8,
//! §18.0.4, §11.2 N6).
//!
//! Two rows and up to three dials, and almost nothing here decides anything. Which
//! tools there are and what a press on a lit one means, which actions there are and
//! what each is called, what a held modifier does to one — all of it is
//! `stark_ui::selection`, and the five acts are the registry's. What is this
//! module's is where the rows sit and how they are measured, which is the same split
//! the brush panel already makes (`crate::panel`).
//!
//! The section is drawn into the brush panel's column rather than a floating panel of
//! its own: this frontend has one column of chrome, and a second floating surface is
//! a design (§25.7) rather than a stage.
//!
//! # Why the acts are not in the column
//!
//! [`selection_bar`] is a strip over the canvas, mounted only while there *is* a
//! selection — so the bar being on screen is what says the canvas is masked, where a
//! permanent row of five chips said it only by being read for which of them are grey.
//! The web app's `SelectionBar` makes the same trade, and the gate is the registry's
//! on both sides ([`Command::enabled`]), so the two cannot come to disagree about
//! when an act is available.

use stark_engine::ObservableState;
use stark_engine::command::Tool;
use stark_model::document::ShapeAction;
use stark_ui::commands::{Bindings, Command};
use stark_ui::icons::Icon;
use stark_ui::selection::{Dial, SHAPE_ACTIONS, SHAPE_TOOLS, action_word, dials};
use wgpui::{Bounds, IntoElement, Pixels, Point, SharedString, canvas, div, prelude::*, rgb, rgba};

use crate::controls::Controls;
use crate::style::{self, StyleExt};

/// The acts a selection can be put through, in the order the bar draws them.
///
/// Every one of them is gated on there *being* a selection ([`Command::enabled`]),
/// which is exactly what [`selection_bar`] is mounted on — so the bar carries no dim
/// chip at all: the whole of the gate is spent on whether there is a bar.
pub const SELECT_ACTS: [Command; 5] = [
    Command::Deselect,
    Command::InvertSelection,
    Command::FillSelection,
    Command::FloatSelection,
    Command::Transform,
];

/// Which control a measured rectangle belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// One of [`SHAPE_TOOLS`], by index.
    Tool(usize),
    /// One of [`SHAPE_ACTIONS`], by index.
    Action(usize),
    /// One of [`SELECT_ACTS`], by index — measured into [`selection_bar`]'s own list
    /// rather than the section's, since that is where the acts are drawn.
    Act(usize),
    /// The bar's strip, behind its chips. A press that missed a chip is still not a
    /// press on the picture, and over the canvas there is nothing below to say so.
    Bar,
}

/// Where each of this section's controls was laid out — `crate::panel`'s device, for
/// its reason: the panel reports its geometry rather than deriving it.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

/// Measure one control.
///
/// Four insets rather than a size (`crate::menu` probes the same way). The two
/// resolve against the same rectangle — the parent's box inside its border, padding
/// included — but a child given only a size is *placed* where the flow had reached,
/// so it lands right only by being its parent's first child (§11.2 N6). The insets
/// say where it goes as well as how big it is, which is one fewer thing for a control
/// that gains a sibling to break.
fn probe(regions: &Regions, region: Region) -> impl IntoElement {
    let regions = regions.clone();
    canvas(
        move |bounds, _, _| regions.borrow_mut().push((region, bounds)),
        |_, (), _, _| {},
    )
    .absolute()
    .top_0()
    .left_0()
    .right_0()
    .bottom_0()
}

/// Which control a press landed on.
///
/// Innermost first (`.rev()`, as `crate::menu` reads its own list): the section's
/// rectangles are disjoint and would answer the same either way, but the bar's strip
/// contains every chip on it and is probed first, being the strip's first child — so
/// reading forwards would hand back the ground under a chip that was pressed.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .rev()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// Build the section.
///
/// Takes the projection rather than reading one, for `crate::panel`'s reason: the
/// section has no state of its own, and everything it does is the view's.
pub fn select_body(
    o: Option<&ObservableState>,
    bindings: &Bindings,
    controls: &Controls,
    regions: &Regions,
) -> impl IntoElement {
    let tool = o.map_or(Tool::Brush, |o| o.tool);
    let action = o.map_or(ShapeAction::default(), |o| o.shape_action);
    let dials = dials(o);
    div()
        .flex()
        .flex_col()
        .gap_1()
        // No heading: the section's own title bar carries it (`crate::panel`), and
        // this drew a second one under it for exactly one build.
        .child(
            // At most one of the three is lit, and none lit *is* the brush: arming is
            // momentary, so a fourth chip for painting would be one that can never be
            // off (`stark_ui::selection`).
            div()
                .flex()
                .gap_1()
                .children(SHAPE_TOOLS.iter().enumerate().map(|(i, t)| {
                    let command = tool_command(*t);
                    marked(
                        format!("tool-{}", command.word()).into(),
                        probe(regions, Region::Tool(i)),
                        command.icon(),
                        *t == tool,
                        command.tooltip(bindings),
                    )
                })),
        )
        .child(
            // Exactly one of the five is always lit: the row is one question — what
            // does this shape do — rather than five switches.
            div()
                .flex()
                .gap_1()
                .children(SHAPE_ACTIONS.iter().enumerate().map(|(i, a)| {
                    marked(
                        format!("shape-{}", action_word(*a)).into(),
                        probe(regions, Region::Action(i)),
                        action_mark(*a),
                        *a == action,
                        action_tip(*a).to_string(),
                    )
                })),
        )
        .children(dials.into_iter().map(|dial| {
            let v = o.map_or(0.0, |o| dial.read(o));
            crate::panel::Slider::new(
                dial.glyph(),
                dial.tip(),
                match dial {
                    Dial::Feather => format!("{v:.0}"),
                    _ => format!("{v:.2}"),
                },
                controls.dial(dial),
            )
        }))
}

/// Whether the acts have anything to be about — and so whether [`selection_bar`] is
/// mounted at all.
///
/// A function rather than a line at the call site: the bar stands on exactly what its
/// five acts are gated on ([`Command::enabled`]), and saying that once is what keeps
/// the two from drifting apart.
pub fn bar_mounted(o: Option<&ObservableState>) -> bool {
    o.is_some_and(|o| o.has_selection)
}

/// The acts on the whole selection, as a bar along the top of the canvas.
///
/// The same edge and the same ground as the transform bar (`crate::transform`), for
/// the same reason — the one edge of this frontend's canvas with nothing above it —
/// and the two never share it: a mode draws its own bar and stands this one down
/// (`crate::canvas`), so a Deselect cannot reach under a transform that has not been
/// committed.
///
/// The chips wear their word beside their mark where the column's wear the mark alone
/// (`marked`): a bar is as wide as the canvas, so the room the column never had is
/// here. The hover still carries the chord, as everything in this chrome does.
pub fn selection_bar(bindings: &Bindings, regions: &Regions) -> impl IntoElement {
    div()
        .absolute()
        .top_0()
        .left_0()
        .right_0()
        .flex()
        .items_center()
        .gap_1()
        .p_2()
        .bg(rgba(style::PANEL_OVER_CANVAS))
        .border_b_1()
        .border_color(rgb(style::EDGE))
        .text_color(rgb(style::INK_LIT))
        .child(probe(regions, Region::Bar))
        // The Select section's own mark: this bar is that section's state made
        // visible, so it says so with the section's glyph rather than with a second
        // picture of a marquee.
        .child(
            div()
                .flex()
                .items_center()
                .gap_1()
                .py_1()
                .px_2()
                .caption()
                .child(crate::icons::icon(
                    stark_ui::icons::SELECTION,
                    style::INK_LABEL,
                ))
                .child("Selection"),
        )
        .children(SELECT_ACTS.iter().enumerate().map(|(i, command)| {
            let chip = div()
                .id(SharedString::from(command.word()))
                .chip()
                .flex()
                .items_center()
                .gap_1()
                .py_1()
                .px_2()
                .resting()
                .child(probe(regions, Region::Act(i)))
                .child(crate::icons::icon(command.icon(), style::INK_MARK))
                .child(command.word());
            style::tip(chip, command.tooltip(bindings))
        }))
}

/// What the hover says one of the five shape actions does.
///
/// The word alone was what the chip wore; a hover has room to say what the word never
/// could, which is the whole bargain a marked chip makes (`crate::panel`).
fn action_tip(action: ShapeAction) -> &'static str {
    use stark_model::document::SelectionMode;
    match action {
        ShapeAction::Select(SelectionMode::Replace) => {
            "New \u{2014} the shape becomes the selection"
        }
        ShapeAction::Select(SelectionMode::Union) => "Add \u{2014} take in what the shape covers",
        ShapeAction::Select(SelectionMode::Subtract) => {
            "Subtract \u{2014} cut what the shape covers out"
        }
        ShapeAction::Select(SelectionMode::Intersect) => {
            "Intersect \u{2014} keep only what both cover"
        }
        ShapeAction::Fill => "Fill \u{2014} lay the paint in hand inside the shape",
    }
}

/// The command that arms `tool` — the registry's row for it, so the chip wears the
/// same word the palette and the chord do.
pub fn tool_command(tool: Tool) -> Command {
    match tool {
        Tool::SelectEllipse => Command::SelectEllipse,
        Tool::SelectLasso => Command::SelectLasso,
        // The rectangle is the marquee anything else would mean, and the brush has no
        // chip here at all — see the tool row above.
        Tool::SelectRect | Tool::Brush => Command::SelectRect,
    }
}

/// The mark for one of the five actions.
///
/// The web app's row draws these five and this one draws the same five, from the
/// catalog both read (`stark_ui::icons`) — which matters more here than usual:
/// the row's whole claim is that Add, Sub and Isect are one question answered three
/// ways, and that claim is carried by the glyphs being a family.
fn action_mark(action: ShapeAction) -> Icon {
    use stark_model::document::SelectionMode;
    match action {
        ShapeAction::Select(SelectionMode::Replace) => stark_ui::icons::SELECTION_NEW,
        ShapeAction::Select(SelectionMode::Union) => stark_ui::icons::SELECTION_ADD,
        ShapeAction::Select(SelectionMode::Subtract) => stark_ui::icons::SELECTION_SUB,
        ShapeAction::Select(SelectionMode::Intersect) => stark_ui::icons::SELECTION_ISECT,
        ShapeAction::Fill => stark_ui::icons::PAINT_BUCKET,
    }
}

/// One chip in a segmented run: its mark, and its word in the hover.
///
/// The word was stacked under the glyph until the column stopped carrying words at
/// all (`crate::panel`). Dropping it is what lets five chips share one row rather
/// than wrapping onto two — and the hover says more than the word ever fit.
///
/// Every chip this draws is one a press can always take: arming a tool and picking
/// what a shape does are answers about a gesture that has not been made, so there is
/// nothing for either to be unavailable *for*. The acts, which do have such a gate,
/// are the bar's ([`selection_bar`]).
fn marked(
    id: SharedString,
    probe: impl IntoElement,
    mark: Icon,
    lit: bool,
    tip: String,
) -> impl IntoElement {
    let ink = if lit { style::INK_LIT } else { style::INK_MARK };
    let chip = div()
        .id(id)
        .chip()
        .flex_1()
        .flex()
        .items_center()
        .justify_center()
        .py_1p5()
        .lit(lit)
        .child(probe)
        .child(crate::icons::icon(mark, ink));
    style::tip(chip, tip)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Before there is a document there is nothing to be armed and nothing to act
    /// on, so the section is its two rows and no dials and the bar is not up — the
    /// frame between the window opening and the engine's first projection, which is
    /// a real frame.
    #[test]
    fn nothing_is_offered_before_there_is_a_document() {
        assert!(dials(None).is_empty());
        assert!(!bar_mounted(None));
        for command in SELECT_ACTS {
            assert!(!command.enabled(None), "{command:?} has nothing to act on");
        }
    }

    /// Each chip names its tool through the registry, so the word on it is the word
    /// the palette and the chord hint use.
    #[test]
    fn each_tool_chip_wears_its_own_command() {
        assert_eq!(tool_command(Tool::SelectRect), Command::SelectRect);
        assert_eq!(tool_command(Tool::SelectEllipse), Command::SelectEllipse);
        assert_eq!(tool_command(Tool::SelectLasso), Command::SelectLasso);
    }
}
