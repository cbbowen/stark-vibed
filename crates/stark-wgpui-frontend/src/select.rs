//! The Select section: which shape draws the next region, what that region does, and
//! the acts on the selection once there is one (§6.8, §18.0.4, §11.2 N6).
//!
//! Three rows and up to three dials, and almost nothing here decides anything. Which
//! tools there are and what a press on a lit one means, which actions there are and
//! what each is called, what a held modifier does to one — all of it is
//! `stark_ui::selection`, and the five acts are the registry's. What is this
//! module's is where the rows sit and how they are measured, which is the same split
//! the brush panel already makes (`crate::panel`).
//!
//! The section is drawn into the brush panel's column rather than a floating panel of
//! its own: this frontend has one column of chrome, and a second floating surface is
//! a design (§25.7) rather than a stage.

use stark_ui::commands::{Bindings, Command};
use stark_ui::icons::Icon;
use stark_ui::selection::{SHAPE_ACTIONS, SHAPE_TOOLS, action_word};
use stark_engine::ObservableState;
use stark_engine::command::Tool;
use stark_model::document::ShapeAction;
use wgpui::{Bounds, IntoElement, Pixels, Point, canvas, div, prelude::*, px, rgb};

/// The acts a selection can be put through, in the order the row draws them.
///
/// Every one of them is gated on there *being* a selection
/// ([`Command::enabled`]) — which is the registry's answer, so a dim button here and
/// a dim palette row in the web app cannot come to disagree about when an act is
/// available.
pub const SELECT_ACTS: [Command; 5] = [
    Command::Deselect,
    Command::InvertSelection,
    Command::FillSelection,
    Command::FloatSelection,
    Command::Transform,
];

/// The three dials this section can show, each mounted only while it means something.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dial {
    /// The edge the next shape gesture's rasterizer strikes, canvas px. Chosen
    /// *before* the gesture, so it is shown for exactly as long as one is pending.
    Feather,
    /// How strongly a **fill** gesture's paint lands — mounted under the Fill action,
    /// because that is the only action it is about.
    FillOpacity,
    /// How strongly the whole mask gates. Not a gesture's setting but the
    /// selection's, set after the fact, so it appears with the selection rather than
    /// with the tool.
    MaskOpacity,
}

impl Dial {
    fn label(self) -> &'static str {
        match self {
            Dial::Feather => "Feather",
            Dial::FillOpacity => "Fill opacity",
            Dial::MaskOpacity => "Selection",
        }
    }

    /// The dial's range. Feather is a canvas-px length; the other two are strengths.
    fn range(self) -> (f32, f32) {
        match self {
            Dial::Feather => (0.0, MAX_FEATHER),
            Dial::FillOpacity | Dial::MaskOpacity => (0.0, 1.0),
        }
    }

    fn read(self, o: &ObservableState) -> f32 {
        match self {
            Dial::Feather => o.selection_feather,
            Dial::FillOpacity => o.shape_opacity,
            Dial::MaskOpacity => o.selection_opacity,
        }
    }

    /// The value a fraction along this dial's track means.
    pub fn value_at(self, fraction: f32) -> f32 {
        let (lo, hi) = self.range();
        lo + fraction.clamp(0.0, 1.0) * (hi - lo)
    }
}

/// The widest edge the feather dial offers, canvas px — the same ceiling the web
/// panel's slider carries.
const MAX_FEATHER: f32 = 64.0;

/// Which dials to show, given what is armed and what there is.
///
/// A list rather than three flags because the section draws them in order and the
/// press has to find them by that order; and computed in one place because "is a
/// shape tool in hand" is asked by each of the three answers.
pub fn dials(o: Option<&ObservableState>) -> Vec<Dial> {
    let Some(o) = o else { return Vec::new() };
    let mut out = Vec::new();
    if o.tool.is_selection() {
        out.push(Dial::Feather);
        if o.shape_action == ShapeAction::Fill {
            out.push(Dial::FillOpacity);
        }
    }
    if o.has_selection {
        out.push(Dial::MaskOpacity);
    }
    out
}

/// Which control a measured rectangle belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// One of [`SHAPE_TOOLS`], by index.
    Tool(usize),
    /// One of [`SHAPE_ACTIONS`], by index.
    Action(usize),
    Dial(Dial),
    /// One of [`SELECT_ACTS`], by index.
    Act(usize),
}

/// Where each of this section's controls was laid out — `crate::panel`'s device, for
/// its reason: the panel reports its geometry rather than deriving it.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

fn probe(regions: &Regions, region: Region) -> impl IntoElement {
    let regions = regions.clone();
    canvas(
        move |bounds, _, _| regions.borrow_mut().push((region, bounds)),
        |_, (), _, _| {},
    )
    .absolute()
    .size_full()
}

/// Which control a press landed on.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// How far along a dial's own track a position is, `0..=1` — read from the x alone,
/// so a drag that wanders off the track keeps moving the dial it took hold of.
pub fn fraction_at(regions: &Regions, dial: Dial, at: Point<Pixels>) -> Option<f32> {
    let bounds = regions
        .borrow()
        .iter()
        .find(|(r, _)| *r == Region::Dial(dial))
        .map(|(_, b)| *b)?;
    let left = f32::from(bounds.origin.x);
    let width = f32::from(bounds.size.width);
    (width > 0.0).then(|| ((f32::from(at.x) - left) / width).clamp(0.0, 1.0))
}

/// Build the section.
///
/// Takes the projection rather than reading one, for `crate::panel`'s reason: the
/// section has no state of its own, and everything it does is the view's.
pub fn select_panel(
    o: Option<&ObservableState>,
    bindings: &Bindings,
    regions: &Regions,
) -> impl IntoElement {
    regions.borrow_mut().clear();
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
                        probe(regions, Region::Tool(i)),
                        command.icon(),
                        command.word(),
                        *t == tool,
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
                        probe(regions, Region::Action(i)),
                        action_mark(*a),
                        action_word(*a),
                        *a == action,
                    )
                })),
        )
        .children(dials.into_iter().map(|dial| {
            let (lo, hi) = dial.range();
            let v = o.map_or(0.0, |o| dial.read(o));
            track(regions, dial, (v - lo) / (hi - lo), v)
        }))
        .child(div().flex().flex_wrap().gap_1().pt_1().children(
            SELECT_ACTS.iter().enumerate().map(|(i, command)| {
                // Dim rather than absent when there is nothing to act on, so the
                // row keeps its shape and a person can see what the selection
                // would buy them.
                let live = command.enabled(o);
                div()
                    .relative()
                    .flex_1()
                    .py_1()
                    .rounded_sm()
                    .bg(rgb(0x2a2d31))
                    .text_xs()
                    .text_center()
                    .text_color(if live { rgb(0xb0b4b8) } else { rgb(0x5a5f64) })
                    .cursor_pointer()
                    .child(probe(regions, Region::Act(i)))
                    .child(shortened(*command, bindings))
            }),
        ))
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

/// What an act's button says. The registry's terse word, with the chord that also
/// reaches it when there is one — this frontend has no tooltips, so the key has
/// nowhere else to be advertised.
fn shortened(command: Command, bindings: &Bindings) -> String {
    match command.shortcut(bindings) {
        Some(chord) => format!("{}  {chord}", command.word()),
        None => command.word().to_string(),
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

/// One chip in a segmented run, wearing its mark over its word.
///
/// Stacked rather than side by side: five chips of glyph-plus-word do not fit the
/// panel's column, and the word is the half that is unambiguous — so it is not the
/// half to drop. The same arrangement the web panel's action row settled on.
fn marked(probe: impl IntoElement, mark: Icon, word: &'static str, lit: bool) -> impl IntoElement {
    div()
        .relative()
        .flex_1()
        .flex()
        .flex_col()
        .items_center()
        .gap_0p5()
        .py_1()
        .rounded_sm()
        .text_xs()
        .cursor_pointer()
        .when_else(
            lit,
            |el| el.bg(rgb(0x35496b)).text_color(rgb(0xe8eaed)),
            |el| el.bg(rgb(0x2a2d31)).text_color(rgb(0xb0b4b8)),
        )
        .child(probe)
        .child(crate::icons::icon(
            mark,
            if lit { 0xe8eaed } else { 0xb0b4b8 },
        ))
        .child(word)
}

/// One labelled dial. The brush panel's slider in miniature, and deliberately not the
/// same type: that one is keyed by [`crate::panel::Knob`], and a shared widget would
/// have to be keyed by neither.
fn track(regions: &Regions, dial: Dial, fraction: f32, value: f32) -> impl IntoElement {
    let fill = fraction.clamp(0.0, 1.0);
    div()
        .flex()
        .flex_col()
        .gap_1()
        .py_1()
        .child(
            div()
                .flex()
                .justify_between()
                .text_xs()
                .text_color(rgb(0x9aa0a6))
                .child(dial.label())
                .child(match dial {
                    Dial::Feather => format!("{value:.0}"),
                    _ => format!("{value:.2}"),
                }),
        )
        .child(
            div()
                .relative()
                .h(px(14.))
                .w_full()
                .rounded_sm()
                .bg(rgb(0x2a2d31))
                .child(probe(regions, Region::Dial(dial)))
                .child(
                    div()
                        .h_full()
                        .w(wgpui::relative(fill))
                        .rounded_sm()
                        .bg(rgb(0x40474e)),
                ),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Before there is a document there is nothing to be armed and nothing to act
    /// on, so the section is its three rows and no dials — the frame between the
    /// window opening and the engine's first projection, which is a real frame.
    #[test]
    fn nothing_is_offered_before_there_is_a_document() {
        assert!(dials(None).is_empty());
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
