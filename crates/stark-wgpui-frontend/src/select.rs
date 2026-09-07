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

use stark_engine::ObservableState;
use stark_engine::command::Tool;
use stark_model::document::ShapeAction;
use stark_ui::commands::{Bindings, Command};
use stark_ui::icons::Icon;
use stark_ui::selection::{SHAPE_ACTIONS, SHAPE_TOOLS, action_word};
use wgpui::{Bounds, IntoElement, Pixels, Point, SharedString, canvas, div, prelude::*};

use crate::controls::Controls;
use crate::style::{self, StyleExt};

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
    /// The mark the track wears (`stark_ui::icons`).
    ///
    /// The fill's dial takes the bucket its own action chip wears, and the adjacency
    /// is the point: the dial is mounted only while Fill is the armed action, so it
    /// appears directly under the lit chip whose strength it sets.
    fn glyph(self) -> Icon {
        match self {
            Dial::Feather => stark_ui::icons::FEATHER,
            Dial::FillOpacity => stark_ui::icons::PAINT_BUCKET,
            Dial::MaskOpacity => stark_ui::icons::OPACITY,
        }
    }

    /// What the hover says the mark means.
    fn tip(self) -> &'static str {
        match self {
            Dial::Feather => {
                "Feather \u{2014} how far the next shape's edge is softened, in canvas px"
            }
            Dial::FillOpacity => "Fill opacity \u{2014} how strongly a fill gesture's paint lands",
            Dial::MaskOpacity => {
                "Selection strength \u{2014} how hard the mask gates what every tool does"
            }
        }
    }

    /// The dial's range. Feather is a canvas-px length; the other two are strengths.
    pub fn range(self) -> (f32, f32) {
        match self {
            Dial::Feather => (0.0, MAX_FEATHER),
            Dial::FillOpacity | Dial::MaskOpacity => (0.0, 1.0),
        }
    }

    /// The step a track moves in: whole px for a length, a hundredth for a strength.
    pub fn step(self) -> f32 {
        match self {
            Dial::Feather => 1.0,
            Dial::FillOpacity | Dial::MaskOpacity => 0.01,
        }
    }

    pub fn read(self, o: &ObservableState) -> f32 {
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
                        true,
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
                        true,
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
        .child(
            div()
                .flex()
                .gap_1()
                .pt_1()
                .children(SELECT_ACTS.iter().enumerate().map(|(i, command)| {
                    // Dim rather than absent when there is nothing to act on, so the
                    // row keeps its shape and a person can see what the selection
                    // would buy them.
                    let live = command.enabled(o);
                    marked(
                        format!("act-{}", command.word()).into(),
                        probe(regions, Region::Act(i)),
                        command.icon(),
                        false,
                        live,
                        command.tooltip(bindings),
                    )
                })),
        )
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
/// `live` is whether the chip has anything to act on: dim rather than absent, so the
/// row keeps its shape and a person can see what a selection would buy them.
fn marked(
    id: SharedString,
    probe: impl IntoElement,
    mark: Icon,
    lit: bool,
    live: bool,
    tip: String,
) -> impl IntoElement {
    let ink = match (lit, live) {
        (true, _) => style::INK_LIT,
        (false, true) => style::INK_MARK,
        (false, false) => style::INK_DEAD,
    };
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
