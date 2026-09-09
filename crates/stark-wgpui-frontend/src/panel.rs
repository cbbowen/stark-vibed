//! The two docked columns, and the Brush shelf that was the first thing in one
//! (§11.2, N2).
//!
//! It reads native rather than like the web app's stylesheet — §11.2 says parity is
//! of *acts*, not of appearance — so a column is a plain dark strip with no rounded
//! cards and no fade: what a native tool column looks like rather than what a
//! floating web one does.
//!
//! **Marks lead; words are the half that goes.** Every control here wears a glyph
//! from the shared catalog (`stark_ui::icons`) and says what it is in a *tooltip*
//! rather than in a label beside it — which is the web app's minimal mode (§11)
//! arrived at from the other side: over there the words are hidden by a class, and
//! here a docked column never had room for them to begin with. Two kinds of text
//! survive, on the web app's own rule: a shelf's **title**, whose row is spent on the
//! fold control either way, and any **name the artist chose** — a preset, a layer, a
//! guide — which no mark can stand in for.
//!
//! The tracks are the widget layer's (§11.1, `wgpui_component::slider`): each holds
//! its drag and its thumb, and the view hears it through `crate::controls`.
//!
//! Where each control *is* is measured rather than derived — see [`Regions`], and the
//! bug that taught it.

use std::collections::HashSet;

use stark_ui::brush_config::{MAX_FLOW, MAX_RADIUS, MIN_RADIUS};
use stark_ui::commands::{Command, VisibilityToggle};
use stark_ui::icons::Icon;
use wgpui::{
    AnyElement, App, Bounds, Entity, IntoElement, Pixels, Point, RenderOnce, SharedString, Window,
    canvas, div, prelude::*, px, rgb,
};
use wgpui_component::slider::SliderState;

use crate::brush::Brush;
use crate::controls::Controls;
use crate::style::{self, StyleExt};

/// The panel's own padding (`p_3`), in logical px.
///
/// Nothing at run time asks it where a control *is* — that is measured ([`Regions`]),
/// and the tests below build their stand-in rectangles from it. What it is still for
/// is how wide a control may be: see [`RIGHT_CONTENT`].
pub const PADDING: f32 = 12.0;

/// The tool column's width in logical px. Wide enough for a mark, a track that still
/// resolves a hundred steps, and the figure beside it.
pub const LEFT_WIDTH: f32 = 216.0;

/// The reading column's. Wider, because a layer row carries a name, a depth indent
/// and four controls — and because the color wheel is a picture rather than a control
/// and wants the room.
pub const RIGHT_WIDTH: f32 = 268.0;

/// The room a right-hand shelf's body actually has: the column, less its padding on
/// both sides.
///
/// A picture in this column is drawn *this* wide rather than at a figure of its own —
/// the color wheel and the navigator's miniature both are. A control narrower than
/// this reads as a mistake beside the layer rows, which fill the column because a row
/// of text and buttons has nothing to do but fill it.
pub const RIGHT_CONTENT: f32 = RIGHT_WIDTH - 2.0 * PADDING;

/// Which side of the window a column is docked to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    /// The shelves this column stacks, and the width it takes when it has any.
    fn shelves(self) -> &'static [VisibilityToggle] {
        match self {
            Side::Left => &crate::visibility::LEFT,
            Side::Right => &crate::visibility::RIGHT,
        }
    }

    fn full_width(self) -> f32 {
        match self {
            Side::Left => LEFT_WIDTH,
            Side::Right => RIGHT_WIDTH,
        }
    }
}

/// The knobs the Brush shelf offers, in the order it draws them.
///
/// **Two, and they are the transient's** (§18.1.8). The shelf is where a brush is
/// *worked* — the size and the flow are what a hand moves all day without the tool
/// becoming a different tool — and everything that says what the tool *is* moved into
/// the brush editor, which has a live stroke to show its work on
/// (`crate::brush_editor`). Hardness and the opacity ceiling were here until it
/// existed, which is the same reason the web app's Brush panel carries the same two.
pub const KNOBS: [Knob; 2] = [Knob::Size, Knob::Flow];

/// One of the brush's two shelf dials — what a track stands for, and how its value is
/// read off the brush and written back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Knob {
    Size,
    Flow,
}

impl Knob {
    /// The mark the track wears, from the shared catalog — which is where the
    /// argument for each of them is (`stark_ui::icons::SIZE`).
    pub fn glyph(self) -> Icon {
        match self {
            Knob::Size => stark_ui::icons::SIZE,
            Knob::Flow => stark_ui::icons::FLOW,
        }
    }

    /// What the hover says the mark means — the sentence that used to be a label,
    /// with room now to say what the knob actually does.
    fn tip(self) -> &'static str {
        match self {
            Knob::Size => "Size \u{2014} how far the tip reaches, in canvas px",
            Knob::Flow => {
                "Flow \u{2014} how much a pass lays, and how hard a wet one works the canvas"
            }
        }
    }

    /// The knob's range **for this brush**.
    ///
    /// The size's is the app's own bound — the one a tuning drag clamps against too,
    /// which is why it lives beside the brush rather than on a panel
    /// (`stark_ui::brush_config`). The flow's is the *in-force effect's*
    /// (`BrushConfig::max_flow`): the liquify strength stops at its quoted,
    /// load-bearing 1 (§6.13) where every other rate stops at the slider's own top.
    ///
    /// Taking the brush is the whole of why the tracks are fractions. A
    /// `SliderState`'s bounds are fixed when it is built and the flow's top moves with
    /// the effect chips, so the trough is a fraction and the view maps it — the shape
    /// the editor's own rows already had (`crate::controls`). Held to `MAX_FLOW`
    /// instead, this slider ran to 3 on a liquify brush while the projection clamped
    /// at 1, and the top two-thirds of it did nothing.
    pub fn range(self, brush: &Brush) -> (f32, f32) {
        match self {
            Knob::Size => (MIN_RADIUS, MAX_RADIUS),
            Knob::Flow => (0.0, brush.config.max_flow()),
        }
    }

    /// The step a track moves in, **as a fraction of its own range** — the trough is
    /// one (see [`range`](Self::range)), so the quantum has to be too: a whole px of
    /// the size's span, a hundredth of the widest flow.
    pub fn step(self) -> f32 {
        match self {
            Knob::Size => 1.0 / (MAX_RADIUS - MIN_RADIUS),
            Knob::Flow => 0.01 / MAX_FLOW,
        }
    }

    /// Where the knob currently stands.
    pub fn read(self, brush: &Brush) -> f32 {
        match self {
            Knob::Size => brush.tune.size,
            Knob::Flow => brush.tune.flow,
        }
    }

    /// Move the knob.
    ///
    /// Neither of the two takes a preset's name off the brush, which is the whole of
    /// why they are the two that stayed: the durable/transient split says working a
    /// brush at another size is the same tool (§18.1.8). Everything that does take the
    /// name off is the editor's now (`crate::brush_editor`).
    fn write(self, brush: &mut Brush, v: f32) {
        match self {
            Knob::Size => brush.tune.size = v,
            Knob::Flow => brush.tune.flow = v,
        }
    }
}

/// One track: its mark, the track itself, and the figure it stands at.
///
/// One line rather than the two the labelled version took, which is where a marked
/// row's saving actually comes from — and the saving is what pays for the four
/// shelves this column now stacks.
#[derive(IntoElement)]
pub struct Slider {
    /// What the row is for, as the hover says it.
    tip: SharedString,
    glyph: Icon,
    /// The value as the panel prints it. A `SharedString` so the component can be
    /// built once per frame without a copy.
    readout: SharedString,
    /// The track's own state, the view's to keep (`crate::controls`).
    state: Entity<SliderState>,
}

impl Slider {
    /// A track for one of the brush's knobs.
    pub fn knob(knob: Knob, brush: &Brush, state: Entity<SliderState>) -> Self {
        Self {
            tip: knob.tip().into(),
            glyph: knob.glyph(),
            readout: readout(knob, knob.read(brush)).into(),
            state,
        }
    }

    /// A track for anything else: the mark, the sentence the hover says, and how the
    /// value is printed — all the caller's, since only it knows what the number is.
    pub fn new(
        glyph: Icon,
        tip: impl Into<SharedString>,
        readout: impl Into<SharedString>,
        state: &Entity<SliderState>,
    ) -> Self {
        Self {
            tip: tip.into(),
            glyph,
            readout: readout.into(),
            state: state.clone(),
        }
    }
}

impl RenderOnce for Slider {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let row = div()
            .id(self.tip.clone())
            .flex()
            .items_center()
            .gap_2()
            .py_0p5()
            .child(crate::icons::icon(self.glyph, style::INK_MARK))
            .child(
                div()
                    .flex_1()
                    .child(wgpui_component::slider::Slider::new(&self.state).w_full()),
            )
            // The figure keeps its own column so the tracks all end in the same place
            // however wide the number is.
            .child(
                div()
                    .w(px(READOUT_WIDTH))
                    .flex_none()
                    .text_right()
                    .caption()
                    .child(self.readout),
            );
        style::tip(row, self.tip)
    }
}

/// The room a track's figure keeps, logical px — four digits of `0.00` at `text_xs`.
const READOUT_WIDTH: f32 = 26.0;

/// One preset in the library.
///
/// The **name stays**: it is the artist's own word, and no mark stands in for one
/// (the same carve-out the web app's minimal mode makes).
#[derive(IntoElement)]
pub struct PresetRow {
    name: SharedString,
    worn: bool,
    regions: Regions,
    index: usize,
}

impl RenderOnce for PresetRow {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let row = div()
            .id(self.name.clone())
            .relative()
            .child(probe(&self.regions, Region::Preset(self.index)))
            .px_2()
            .py_1()
            .rounded_sm()
            .text_sm()
            .cursor_pointer()
            .lit_row(self.worn)
            .hover(|s| s.bg(rgb(style::HOVER)))
            .child(self.name.clone());
        style::tip(row, format!("Take up “{}”", self.name))
    }
}

/// How the panel prints a knob's value: enough digits to see a change, no more.
fn readout(knob: Knob, v: f32) -> String {
    match knob {
        Knob::Size => format!("{v:.0}"),
        _ => format!("{v:.2}"),
    }
}

/// The Brush shelf's body: the two dials a hand works, the way into the editor, and
/// the preset library.
///
/// **What is *not* here is the point.** The effect chips and the stamp gallery were
/// both on this shelf until the editor existed, and both say what the tool *is* — so
/// they went where the rest of that lives, beside a stroke that shows what they do
/// (`crate::brush_editor`). What is left is the two transient knobs (§18.1.8), one
/// button, and the artist's own names.
///
/// A free function taking the pieces rather than a `Render` impl, because the shelf
/// has no state of its own: everything it shows is the brush's and everything it does
/// is the canvas view's (`crate::canvas`), which owns the engine the changes go to.
pub fn brush_body(brush: &Brush, controls: &Controls, regions: &Regions) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .children(
            KNOBS
                .iter()
                .zip(&controls.knobs)
                .map(|(knob, state)| Slider::knob(*knob, brush, state.clone())),
        )
        // The way to everything the two tracks above no longer carry. A full-width
        // button rather than a chip: what it opens is a *surface*, which is the
        // distinction the menu bar already draws between a row that toggles and one
        // that raises a dialog.
        .child(style::tip(
            div()
                .id("edit-brush")
                .chip()
                .flex()
                .items_center()
                .justify_center()
                .gap_1()
                .py_1p5()
                .mt_1()
                .resting()
                .child(probe(regions, Region::Edit))
                .child(crate::icons::icon(
                    stark_ui::icons::EDIT_BRUSH,
                    style::INK_MARK,
                ))
                .child(Command::EditBrush.word()),
            Command::EditBrush.hint(),
        ))
        .child(div().pt_2().heading().child("Presets"))
        .children(brush.library.iter().enumerate().map(|(i, e)| PresetRow {
            name: e.name.clone().into(),
            worn: brush.from.as_deref() == Some(e.name.as_str()),
            regions: regions.clone(),
            index: i,
        }))
}

/// One column of shelves, built by the view.
///
/// The body is `None` for a shelf that is folded to its title bar, and a shelf that is
/// **hidden** is not in the list at all — which is the difference the Window menu
/// makes, and the reason a hidden one costs no work: the view never builds its body.
pub struct Shelf {
    pub what: VisibilityToggle,
    pub body: Option<AnyElement>,
}

/// Build one column.
pub fn column(
    side: Side,
    hidden: &HashSet<VisibilityToggle>,
    regions: &Regions,
    shelves: Vec<Shelf>,
) -> impl IntoElement {
    let id = match side {
        Side::Left => "column-left",
        Side::Right => "column-right",
    };
    div()
        .id(id)
        .panel_column(width(side, hidden))
        // **Scrolls.** A column of four shelves that ran off the bottom of the window
        // would be one whose last shelf does not exist. The web app floats its panels
        // so they can overlap; this one is a column, so the column is what gives.
        .overflow_y_scroll()
        .when(side == Side::Left, |el| el.border_r_1())
        .when(side == Side::Right, |el| el.border_l_1())
        .border_color(rgb(style::EDGE))
        .children(shelves.into_iter().map(|shelf| {
            div()
                .flex()
                .flex_col()
                .gap_1()
                .child(title(regions, shelf.what, shelf.body.is_some()))
                .children(shelf.body)
        }))
}

/// A column's width, given what is hidden: its own figure while it still holds a
/// shelf, and **nothing at all** once every shelf in it has been put away.
///
/// The canvas takes the room back, because a column is what the surface is laid out
/// *beside* rather than over — so this number is not only what a press is tested
/// against ([`within`]) but where canvas space begins (`Canvas::origin`). Which is the
/// whole reason hiding a shelf here is a different kind of act from hiding a panel in
/// the web app, where a floating panel costs the painting nothing to begin with.
pub fn width(side: Side, hidden: &HashSet<VisibilityToggle>) -> f32 {
    if side.shelves().iter().any(|what| !hidden.contains(what)) {
        side.full_width()
    } else {
        0.0
    }
}

/// A shelf's title bar: its mark, its name, and the triangle that folds it away.
///
/// Keyed by [`VisibilityToggle`], which is the vocabulary both frontends name a piece
/// of chrome in (§25) — so what this client left folded is stored under the same word
/// the web app stores its own under, and a variant renamed costs the row rather than
/// mis-matching it (`stark_ui::visibility`).
///
/// **Folded is not hidden, and the Window menu is the difference.** A folded shelf
/// leaves its title bar in the column, which is what lets it be opened again by the
/// thing that closed it; a hidden one is not built at all, so the only way back is the
/// map of what is on screen (`crate::menu`'s Window menu, §25.5).
///
/// The title is one of the two kinds of word a marked column keeps: its row is spent
/// on the fold control either way, so the text costs nothing that was not already
/// spent.
fn title(regions: &Regions, what: VisibilityToggle, open: bool) -> impl IntoElement {
    let name = what.name();
    let row = div()
        .id(SharedString::from(format!("fold-{what:?}")))
        .relative()
        .flex()
        .items_center()
        .gap_2()
        .pt_2()
        .cursor_pointer()
        .child(probe(regions, Region::Fold(what)))
        .child(crate::icons::icon(what.icon(), style::INK_LABEL))
        .child(div().flex_1().heading().child(name))
        .child(
            div()
                .text_xs()
                .text_color(rgb(style::INK_FOLD))
                // Down for open, right for folded — which way the content lies, not
                // which way pressing it would go.
                .child(if open { "\u{25be}" } else { "\u{25b8}" }),
        );
    style::tip(row, if open { "Fold away" } else { "Unfold" })
}

// --- what the layout actually is ------------------------------------------
//
// **The panel does not compute its own geometry, it reports it.** The first cut of
// this hand-derived every offset from the Tailwind-shaped spacing the tree asks for —
// and was wrong: a press where the arithmetic said "Airbrush" selected Hard Eraser,
// because the guessed row pitch was 26 px and Taffy had laid out 39. Two descriptions
// of one layout is exactly the drift `stark-ui` exists to stop, one scale down,
// and the answer is the same: have one of them ask the other.
//
// So each control carries a zero-cost `canvas` element whose *prepaint* writes its
// laid-out bounds into [`Regions`]. The view clears the list each frame before
// building the tree and reads it on the next press — layout is stable between frames,
// and a press before the first prepaint simply finds nothing, which is the same
// answer as a press on empty panel.

/// Which control a measured rectangle belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    Preset(usize),
    /// The button that opens the brush editor (`crate::brush_editor`).
    Edit,
    /// A shelf's title bar — pressing it folds the shelf away.
    Fold(VisibilityToggle),
}

/// Where each control was laid out, as of the last frame that painted.
///
/// `Rc<RefCell<..>>` because a prepaint closure is `'static` and the view outlives it:
/// wgpui hands the closure no context to write through, so what it writes into has to
/// be something it owns a handle to.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

/// An invisible element that records its parent's bounds.
///
/// `absolute().size_full()` so it takes the whole of the control it sits in and adds
/// nothing to the layout. Its paint is empty: what is wanted is the *measurement*,
/// and the control draws itself.
pub fn probe(regions: &Regions, region: Region) -> impl IntoElement {
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

/// Whether a position is over one of the columns at all — the canvas begins where
/// they end, so a press neither column wants is paint.
///
/// The one measurement that is *not* read back off the layout, because it is what the
/// layout is told: a column is [`width`] wide because this module says so. Taking the
/// two widths as arguments rather than reading them here is what keeps one press and
/// one stroke agreeing about where the canvas starts — the caller has the numbers
/// already, and a second read of them is a second chance to read a stale set.
pub fn within(at: Point<Pixels>, left: f32, right: f32, window: f32) -> bool {
    let x = f32::from(at.x);
    (left > 0.0 && x <= left) || (right > 0.0 && x >= window - right)
}

/// The value a fraction along `knob`'s track means, for the brush the track is
/// showing — see [`Knob::range`] for why that is not a constant.
pub fn value_at(knob: Knob, brush: &Brush, fraction: f32) -> f32 {
    let (lo, hi) = knob.range(brush);
    lo + fraction.clamp(0.0, 1.0) * (hi - lo)
}

/// Move `knob` to `fraction` of its range.
pub fn drag_knob(brush: &mut Brush, knob: Knob, fraction: f32) {
    let value = value_at(knob, brush, fraction);
    knob.write(brush, value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_ui::panels::PanelId;
    use wgpui::{point, px, size};

    fn at(x: f32, y: f32) -> Point<Pixels> {
        point(px(x), px(y))
    }

    /// A stand-in for a painted frame: the regions a layout would have reported.
    fn measured(rows: &[(Region, f32, f32)]) -> Regions {
        let regions = Regions::default();
        for (region, top, height) in rows {
            regions.borrow_mut().push((
                *region,
                Bounds {
                    origin: point(px(PADDING), px(*top)),
                    size: size(px(LEFT_WIDTH - 2.0 * PADDING), px(*height)),
                },
            ));
        }
        regions
    }

    /// A press between the two columns is the canvas's, whatever it is level with.
    #[test]
    fn a_press_between_the_columns_is_paint() {
        const W: f32 = 1200.0;
        assert!(within(
            at(LEFT_WIDTH - 1.0, 100.0),
            LEFT_WIDTH,
            RIGHT_WIDTH,
            W
        ));
        assert!(!within(
            at(LEFT_WIDTH + 1.0, 100.0),
            LEFT_WIDTH,
            RIGHT_WIDTH,
            W
        ));
        assert!(within(
            at(W - RIGHT_WIDTH + 1.0, 100.0),
            LEFT_WIDTH,
            RIGHT_WIDTH,
            W
        ));
        assert!(!within(
            at(W - RIGHT_WIDTH - 1.0, 100.0),
            LEFT_WIDTH,
            RIGHT_WIDTH,
            W
        ));
    }

    /// A column with nothing in it is not a column: the canvas starts at the window's
    /// own edge, and a press at x=0 is paint rather than a press on a strip of nothing.
    #[test]
    fn a_column_of_no_shelves_takes_no_room() {
        let mut hidden: HashSet<VisibilityToggle> = crate::visibility::LEFT.into_iter().collect();
        assert!(
            width(Side::Left, &hidden) <= 0.0,
            "a column with nothing in it takes no room"
        );
        assert!(
            !within(at(0.0, 100.0), width(Side::Left, &hidden), 0.0, 1200.0),
            "so a press at the window's own edge is paint"
        );
        // One shelf left is a whole column: the width is what the layout is told, not
        // a sum over what is in it.
        hidden.remove(&VisibilityToggle::Panel(PanelId::Select));
        assert!(
            within(
                at(LEFT_WIDTH - 1.0, 100.0),
                width(Side::Left, &hidden),
                0.0,
                1200.0
            ),
            "one shelf left is a whole column"
        );
    }

    /// Before the first frame has painted there is nothing measured, and a press
    /// finds no control rather than guessing at one.
    #[test]
    fn nothing_is_hit_before_a_frame_has_been_laid_out() {
        let regions = Regions::default();
        assert_eq!(hit(&regions, at(LEFT_WIDTH / 2.0, 100.0)), None);
    }

    /// A press finds the control whose measured rectangle it is inside, and nothing
    /// between two of them.
    #[test]
    fn a_press_finds_the_control_it_is_inside() {
        let regions = measured(&[
            (Region::Edit, 60.0, 18.0),
            (Region::Preset(0), 110.0, 18.0),
            (Region::Preset(3), 300.0, 26.0),
        ]);
        let x = LEFT_WIDTH / 2.0;
        assert_eq!(hit(&regions, at(x, 66.0)), Some(Region::Edit));
        assert_eq!(hit(&regions, at(x, 118.0)), Some(Region::Preset(0)));
        assert_eq!(hit(&regions, at(x, 310.0)), Some(Region::Preset(3)));
        assert_eq!(
            hit(&regions, at(x, 95.0)),
            None,
            "the gap belongs to nobody"
        );
    }

    /// Every knob the shelf offers is the *hand's* rather than the tool's (§18.1.8) —
    /// which is what makes the two of them the ones that did not move into the editor,
    /// and what lets a drag on either keep the preset's name on the brush.
    #[test]
    fn the_shelf_keeps_only_the_transient_knobs() {
        let mut brush = Brush::new(Default::default());
        let tool = brush.config;
        for knob in KNOBS {
            drag_knob(&mut brush, knob, 0.5);
        }
        assert_eq!(brush.config, tool, "a shelf dial changed what the tool is");
    }

    /// A knob reads back what a drag wrote, through the range it declares — the round
    /// trip the panel relies on to draw a fill where the hand left it.
    #[test]
    fn a_knob_reads_back_what_a_drag_wrote() {
        let mut brush = Brush::new(Default::default());
        for knob in KNOBS {
            drag_knob(&mut brush, knob, 0.25);
            let (lo, hi) = knob.range(&brush);
            let want = lo + 0.25 * (hi - lo);
            assert!(
                (knob.read(&brush) - want).abs() < 1e-3,
                "{knob:?} read back {} rather than {want}",
                knob.read(&brush),
            );
        }
    }

    /// **The Flow track ends where the effect does.** A liquify brush's strength is
    /// clamped to its quoted 1 at the projection (§6.13), so a track running to
    /// `MAX_FLOW` would have had two thirds of its travel do nothing — and the knob is
    /// reachable three ways (this dial, the editor's row, the tuning drag), which is
    /// what `BrushConfig::max_flow` exists to keep to one answer.
    #[test]
    fn a_liquify_brush_gets_the_flow_track_its_effect_has() {
        let mut brush = Brush::new(Default::default());
        assert_eq!(Knob::Flow.range(&brush), (0.0, MAX_FLOW));
        brush.config.effect = stark_ui::brush_config::BrushEffectType::Liquify;
        assert_eq!(Knob::Flow.range(&brush), (0.0, 1.0));
        // The full track is the full strength, rather than a third of it.
        drag_knob(&mut brush, Knob::Flow, 1.0);
        assert_eq!(brush.tune.flow, 1.0);
        assert_eq!(
            brush.config.params(brush.tune).effect.flow(),
            1.0,
            "and the projection has nothing left to clamp away",
        );
    }

    /// Every knob and every effect wears a mark, which is the whole of what a column
    /// with no labels stands on: a control whose glyph went missing would be a blank
    /// square with a tooltip.
    #[test]
    fn every_control_that_lost_its_word_has_a_mark() {
        for knob in KNOBS {
            assert!(knob.glyph().svg().is_some(), "{knob:?} has no glyph");
            assert!(!knob.tip().is_empty(), "{knob:?} says nothing on hover");
        }
    }
}
