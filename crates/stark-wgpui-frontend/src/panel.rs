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

use stark_model::document::BrushShape;
use stark_ui::brush_config::{BrushEffectType, MAX_FLOW, MAX_RADIUS, MIN_RADIUS};
use stark_ui::commands::VisibilityToggle;
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
/// Test-only, and that is the point: nothing at run time needs it any more, because
/// nothing at run time computes where a control is. It survives here to build the
/// stand-in rectangles a measured layout would have reported.
#[cfg(test)]
const PADDING: f32 = 12.0;

/// The tool column's width in logical px. Wide enough for a mark, a track that still
/// resolves a hundred steps, and the figure beside it.
pub const LEFT_WIDTH: f32 = 216.0;

/// The reading column's. Wider, because a layer row carries a name, a depth indent
/// and four controls — and because the color wheel is a picture rather than a control
/// and wants the room.
pub const RIGHT_WIDTH: f32 = 268.0;

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
pub const KNOBS: [Knob; 4] = [Knob::Size, Knob::Flow, Knob::Hardness, Knob::Opacity];

/// One of the brush's four dials — what a track stands for, and how its value is
/// read off the brush and written back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Knob {
    Size,
    Flow,
    Hardness,
    Opacity,
}

impl Knob {
    /// The mark the track wears, from the shared catalog — which is where the
    /// argument for each of them is (`stark_ui::icons::SIZE`).
    pub fn glyph(self) -> Icon {
        match self {
            Knob::Size => stark_ui::icons::SIZE,
            Knob::Flow => stark_ui::icons::FLOW,
            Knob::Hardness => stark_ui::icons::HARDNESS,
            Knob::Opacity => stark_ui::icons::OPACITY,
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
            Knob::Hardness => "Hardness \u{2014} how abruptly a round tip's edge falls away",
            Knob::Opacity => "Opacity \u{2014} how much of the paint this stroke may show",
        }
    }

    /// The knob's range. Size and flow are the app's own bounds — the ones a tuning
    /// drag clamps against too, which is why they live beside the brush rather than
    /// on a panel (`stark_ui::brush_config`).
    pub fn range(self) -> (f32, f32) {
        match self {
            Knob::Size => (MIN_RADIUS, MAX_RADIUS),
            Knob::Flow => (0.0, MAX_FLOW),
            Knob::Hardness | Knob::Opacity => (0.0, 1.0),
        }
    }

    /// The step a track moves in: whole px for a size, a hundredth for a strength.
    pub fn step(self) -> f32 {
        match self {
            Knob::Size => 1.0,
            Knob::Flow | Knob::Hardness | Knob::Opacity => 0.01,
        }
    }

    /// Where the knob currently stands.
    pub fn read(self, brush: &Brush) -> f32 {
        match self {
            Knob::Size => brush.tune.size,
            Knob::Flow => brush.tune.flow,
            Knob::Hardness => match brush.config.shape {
                BrushShape::Round { hardness } => hardness,
                // A stamp has no hardness of its own; the dial shows the fallback the
                // renderer would use if the asset failed to resolve (§6.6).
                BrushShape::Stamp(_) => BrushShape::DEFAULT_HARDNESS,
            },
            Knob::Opacity => brush.config.opacity(),
        }
    }

    /// Move the knob, and say whether the tool itself changed.
    ///
    /// The distinction is the durable/transient split: size and flow are the hand's,
    /// so working them keeps the preset's name on the brush, while hardness and
    /// opacity are what the tool *is* and take it off (§18.1.8).
    fn write(self, brush: &mut Brush, v: f32) -> bool {
        match self {
            Knob::Size => {
                brush.tune.size = v;
                false
            }
            Knob::Flow => {
                brush.tune.flow = v;
                false
            }
            Knob::Hardness => {
                brush.config.shape = BrushShape::Round { hardness: v };
                true
            }
            Knob::Opacity => {
                brush.config.set_opacity(v);
                true
            }
        }
    }
}

/// The mark for a brush effect.
///
/// Three of the four are marks the catalog already holds and the sharing is the claim
/// each time (`stark_ui::icons::LIQUIFY` carries the argument); only the smear had no
/// picture.
fn effect_mark(effect: BrushEffectType) -> Icon {
    match effect {
        BrushEffectType::Paint => stark_ui::icons::BRUSH,
        BrushEffectType::Wet => stark_ui::icons::WET,
        BrushEffectType::Erase => stark_ui::icons::ERASER,
        BrushEffectType::Liquify => stark_ui::icons::LIQUIFY,
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

/// The Brush shelf's body: the four dials, the effect the brush is in, the stamp
/// gallery and the preset library.
///
/// A free function taking the pieces rather than a `Render` impl, because the shelf
/// has no state of its own: everything it shows is the brush's and everything it does
/// is the canvas view's (`crate::canvas`), which owns the engine the changes go to.
pub fn brush_body(
    brush: &Brush,
    controls: &Controls,
    effects: &[(BrushEffectType, &'static str)],
    regions: &Regions,
    // An `Option` because the view builds it once and hands it to whichever shelf
    // wants it — `None` cannot happen here, and drawing nothing is the right answer if
    // it ever does.
    shapes: Option<AnyElement>,
) -> impl IntoElement {
    let effect = brush.config.effect;
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
        .child(
            div()
                .flex()
                .gap_1()
                .pt_1()
                .children(effects.iter().enumerate().map(|(i, (kind, label))| {
                    let chip = div()
                        .id(*label)
                        .chip()
                        .child(probe(regions, Region::Effect(i)))
                        .flex_1()
                        .flex()
                        .items_center()
                        .justify_center()
                        .py_1p5()
                        .lit(*kind == effect)
                        .child(crate::icons::icon(
                            effect_mark(*kind),
                            if *kind == effect {
                                style::INK_LIT
                            } else {
                                style::INK_MARK
                            },
                        ));
                    style::tip(chip, effect_tip(*kind))
                })),
        )
        // The stamp gallery sits with the brush rather than with the presets: what a
        // shape *is* is the tool, and a preset is a way of arriving at one.
        .children(shapes)
        .child(div().pt_2().heading().child("Presets"))
        .children(brush.library.iter().enumerate().map(|(i, e)| PresetRow {
            name: e.name.clone().into(),
            worn: brush.from.as_deref() == Some(e.name.as_str()),
            regions: regions.clone(),
            index: i,
        }))
}

/// What the hover says an effect chip does — the word it used to wear, and then what
/// the word never had room to say.
fn effect_tip(effect: BrushEffectType) -> &'static str {
    match effect {
        BrushEffectType::Paint => "Paint \u{2014} lay the colour in hand",
        BrushEffectType::Wet => "Wet \u{2014} move and mix the paint already on the canvas",
        BrushEffectType::Erase => "Erase \u{2014} take paint away where the tip passes",
        BrushEffectType::Liquify => "Liquify \u{2014} push the paint about without adding any",
    }
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
    let command = what.command();
    let row = div()
        .id(SharedString::from(format!("fold-{what:?}")))
        .relative()
        .flex()
        .items_center()
        .gap_2()
        .pt_2()
        .cursor_pointer()
        .child(probe(regions, Region::Fold(what)))
        .child(crate::icons::icon(command.icon(), style::INK_LABEL))
        .child(div().flex_1().heading().child(command.name()))
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
    Effect(usize),
    Preset(usize),
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

/// The value a fraction along `knob`'s track means.
pub fn value_at(knob: Knob, fraction: f32) -> f32 {
    let (lo, hi) = knob.range();
    lo + fraction.clamp(0.0, 1.0) * (hi - lo)
}

/// Move `knob` to `fraction` of its range. Answers whether the *tool* changed, so the
/// caller can take the preset's name off the brush.
pub fn drag_knob(brush: &mut Brush, knob: Knob, fraction: f32) -> bool {
    knob.write(brush, value_at(knob, fraction))
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
            (Region::Effect(0), 60.0, 18.0),
            (Region::Effect(1), 110.0, 18.0),
            (Region::Preset(3), 300.0, 26.0),
        ]);
        let x = LEFT_WIDTH / 2.0;
        assert_eq!(hit(&regions, at(x, 66.0)), Some(Region::Effect(0)));
        assert_eq!(hit(&regions, at(x, 118.0)), Some(Region::Effect(1)));
        assert_eq!(hit(&regions, at(x, 310.0)), Some(Region::Preset(3)));
        assert_eq!(
            hit(&regions, at(x, 95.0)),
            None,
            "the gap belongs to nobody"
        );
    }

    /// Size and flow are the hand's, hardness and opacity are the tool's — so only
    /// the second pair takes a preset's name off the brush (§18.1.8).
    #[test]
    fn only_the_durable_knobs_change_the_tool() {
        let mut brush = Brush::new(Default::default());
        assert!(!drag_knob(&mut brush, Knob::Size, 0.5));
        assert!(!drag_knob(&mut brush, Knob::Flow, 0.5));
        assert!(drag_knob(&mut brush, Knob::Hardness, 0.5));
        assert!(drag_knob(&mut brush, Knob::Opacity, 0.5));
    }

    /// A knob reads back what a drag wrote, through the range it declares — the round
    /// trip the panel relies on to draw a fill where the hand left it.
    #[test]
    fn a_knob_reads_back_what_a_drag_wrote() {
        let mut brush = Brush::new(Default::default());
        for knob in KNOBS {
            drag_knob(&mut brush, knob, 0.25);
            let (lo, hi) = knob.range();
            let want = lo + 0.25 * (hi - lo);
            assert!(
                (knob.read(&brush) - want).abs() < 1e-3,
                "{knob:?} read back {} rather than {want}",
                knob.read(&brush),
            );
        }
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
        for effect in [
            BrushEffectType::Paint,
            BrushEffectType::Wet,
            BrushEffectType::Erase,
            BrushEffectType::Liquify,
        ] {
            assert!(
                effect_mark(effect).svg().is_some(),
                "{effect:?} has no glyph"
            );
            assert!(!effect_tip(effect).is_empty());
        }
    }
}
