//! The brush panel: what the tool is, and what the hand is working it at (§11.2, N2).
//!
//! The first chrome this frontend has. It reads native rather than like the web app's
//! stylesheet — §11.2 says parity is of *acts*, not of appearance — so it is a plain
//! dark column with no rounded cards and no fade: what a native tool panel looks like
//! rather than what a floating web one does.
//!
//! wgpui ships no widgets, so [`Slider`] and [`PresetRow`] are built here. Both are
//! `RenderOnce` components rather than views: they hold no state between frames, the
//! value they show is the brush's, and the drag that moves one belongs to the view
//! (`crate::canvas`) because a slider that owned its own drag would lose it the
//! moment the pointer left its bounds.
//!
//! Where each control *is* is measured rather than derived — see [`Regions`], and the
//! bug that taught it.

use std::collections::HashSet;

use stark_chrome::brush_config::{BrushEffectType, MAX_FLOW, MAX_RADIUS, MIN_RADIUS};
use stark_chrome::panels::PanelId;
use stark_model::document::BrushShape;
use wgpui::{
    App, Bounds, IntoElement, Pixels, Point, RenderOnce, SharedString, Window, canvas, div,
    prelude::*, px, rgb,
};

use crate::brush::Brush;

/// The panel's own padding (`p_3`), in logical px.
///
/// Test-only, and that is the point: nothing at run time needs it any more, because
/// nothing at run time computes where a control is. It survives here to build the
/// stand-in rectangles a measured layout would have reported.
#[cfg(test)]
const PADDING: f32 = 12.0;

/// The panel's width in logical px. Wide enough for a label, a value and a track that
/// still resolves a hundred steps.
pub const WIDTH: f32 = 232.0;

/// The knobs the panel offers, in the order it draws them.
pub const KNOBS: [Knob; 4] = [Knob::Size, Knob::Flow, Knob::Hardness, Knob::Opacity];

/// Which knob a drag is moving. A panel-level fact, not a slider's: a drag that
/// leaves the track keeps moving the knob it started on, which is what makes a
/// slider usable at the ends of its range.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Knob {
    Size,
    Flow,
    Hardness,
    Opacity,
}

impl Knob {
    fn label(self) -> &'static str {
        match self {
            Knob::Size => "Size",
            Knob::Flow => "Flow",
            Knob::Hardness => "Hardness",
            Knob::Opacity => "Opacity",
        }
    }

    /// The knob's range. Size and flow are the app's own bounds — the ones a tuning
    /// drag clamps against too, which is why they live beside the brush rather than
    /// on a panel (`stark_chrome::brush_config`).
    fn range(self) -> (f32, f32) {
        match self {
            Knob::Size => (MIN_RADIUS, MAX_RADIUS),
            Knob::Flow => (0.0, MAX_FLOW),
            Knob::Hardness | Knob::Opacity => (0.0, 1.0),
        }
    }

    /// Where the knob currently stands.
    fn read(self, brush: &Brush) -> f32 {
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

/// One labelled track with a value beside it.
#[derive(IntoElement)]
pub struct Slider {
    knob: Knob,
    /// Where the fill stops, `0..=1` — the knob's position in its range rather than
    /// its value, because a track knows about neither.
    fraction: f32,
    /// The value as the panel prints it. A `SharedString` so the component can be
    /// built once per frame without a copy.
    readout: SharedString,
    active: bool,
    regions: Regions,
}

impl RenderOnce for Slider {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let fill = (self.fraction.clamp(0.0, 1.0) * 100.0).round();
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
                    .child(self.knob.label())
                    .child(self.readout),
            )
            .child(
                // The track carries the knob's identity: the panel reads it back off
                // the press to know which one a drag is moving.
                div()
                    .id(SharedString::from(self.knob.label()))
                    .relative()
                    .h(px(18.))
                    .w_full()
                    .rounded_sm()
                    .bg(rgb(0x2a2d31))
                    .child(probe(&self.regions, Region::Knob(self.knob)))
                    .child(
                        div()
                            .h_full()
                            .w(wgpui::relative(fill / 100.0))
                            .rounded_sm()
                            .bg(if self.active {
                                rgb(0x5b9dd9)
                            } else {
                                rgb(0x40474e)
                            }),
                    ),
            )
    }
}

/// One preset in the library.
#[derive(IntoElement)]
pub struct PresetRow {
    name: SharedString,
    worn: bool,
    regions: Regions,
    index: usize,
}

impl RenderOnce for PresetRow {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        div()
            .id(self.name.clone())
            .relative()
            .child(probe(&self.regions, Region::Preset(self.index)))
            .px_2()
            .py_1()
            .rounded_sm()
            .text_sm()
            .cursor_pointer()
            .when_else(
                self.worn,
                |el| el.bg(rgb(0x35496b)).text_color(rgb(0xe8eaed)),
                |el| el.text_color(rgb(0xb0b4b8)),
            )
            .hover(|s| s.bg(rgb(0x2f3337)))
            .child(self.name)
    }
}

/// How the panel prints a knob's value: enough digits to see a change, no more.
fn readout(knob: Knob, v: f32) -> String {
    match knob {
        Knob::Size => format!("{v:.0}"),
        _ => format!("{v:.2}"),
    }
}

/// Build the panel's element tree.
///
/// A free function taking the pieces rather than a `Render` impl, because the panel
/// has no state of its own: everything it shows is the brush's and everything it does
/// is the canvas view's (`crate::canvas`), which owns the engine the changes go to.
/// `select` is the Select section, built by its own module and passed in rather than
/// reached for: the two share this column but nothing else, and a panel that
/// constructed its neighbour would be the one place either could learn about the
/// other's state.
/// The sections the column is built out of, each by its own module.
///
/// A struct rather than four more arguments: they are all `impl IntoElement`, so a
/// caller that shuffled two of them would build a panel with the Select rows under
/// the "Color" title and the compiler would have nothing to say.
pub struct Sections<C, S, G, U> {
    pub color: C,
    pub select: S,
    pub shapes: G,
    pub substrates: U,
}

pub fn brush_panel(
    brush: &Brush,
    dragging: Option<Knob>,
    effects: &[(BrushEffectType, &'static str)],
    regions: &Regions,
    folded: &HashSet<PanelId>,
    sections: Sections<impl IntoElement, impl IntoElement, impl IntoElement, impl IntoElement>,
) -> impl IntoElement {
    let Sections {
        color,
        select,
        shapes,
        substrates,
    } = sections;
    let effect = brush.config.effect;
    // Cleared here rather than after the press: prepaint refills it every frame, and
    // clearing on read would leave the frame between a press and the next paint with
    // no layout to test against.
    regions.borrow_mut().clear();
    let brush_body = div()
        .flex()
        .flex_col()
        .gap_1()
        .children(KNOBS.map(|knob| {
            let (lo, hi) = knob.range();
            let v = knob.read(brush);
            Slider {
                knob,
                fraction: (v - lo) / (hi - lo),
                readout: readout(knob, v).into(),
                active: dragging == Some(knob),
                regions: regions.clone(),
            }
        }))
        .child(
            div()
                .flex()
                .gap_1()
                .pt_1()
                .children(effects.iter().enumerate().map(|(i, (kind, label))| {
                    div()
                        .id(*label)
                        .relative()
                        .child(probe(regions, Region::Effect(i)))
                        .flex_1()
                        .py_1()
                        .rounded_sm()
                        .text_xs()
                        .text_center()
                        .cursor_pointer()
                        .when_else(
                            *kind == effect,
                            |el| el.bg(rgb(0x35496b)).text_color(rgb(0xe8eaed)),
                            |el| el.bg(rgb(0x2a2d31)).text_color(rgb(0xb0b4b8)),
                        )
                        .child(*label)
                })),
        )
        // The stamp gallery sits with the brush rather than with the presets: what a
        // shape *is* is the tool, and a preset is a way of arriving at one.
        .child(shapes)
        .child(substrates)
        .child(
            div()
                .pt_2()
                .text_sm()
                .text_color(rgb(0x9aa0a6))
                .child("Presets"),
        )
        .children(brush.library.iter().enumerate().map(|(i, e)| PresetRow {
            name: e.name.clone().into(),
            worn: brush.from.as_deref() == Some(e.name.as_str()),
            regions: regions.clone(),
            index: i,
        }));

    div()
        .id("panels")
        .flex()
        .flex_col()
        .w(px(WIDTH))
        .h_full()
        .p_3()
        .gap_2()
        // **Scrolls.** This column was a fixed run of controls when it held a brush;
        // it holds three panels now and will hold more (§11.2 N8), and a stack that
        // ran off the bottom of the window would be one whose last panel does not
        // exist. The web app floats its panels so they can overlap; this one is a
        // column, so the column is what gives.
        .overflow_y_scroll()
        .bg(rgb(0x1e2124))
        .border_r_1()
        .border_color(rgb(0x35393d))
        .text_color(rgb(0xe8eaed))
        .child(section(regions, folded, PanelId::Color, color))
        .child(section(regions, folded, PanelId::Brush, brush_body))
        .child(section(regions, folded, PanelId::Select, select))
}

/// One panel in the stack: a title bar that folds it, and its body when it is not.
///
/// Keyed by [`PanelId`], which is the vocabulary both frontends name a panel in
/// (§11) — so what this client left folded is stored under the same word the web app
/// stores its own under, and a variant renamed costs the row rather than mis-matching
/// it (`stark_chrome::visibility`).
///
/// **Folded rather than hidden.** A hidden panel is one a person has to remember
/// exists; a folded one leaves its title behind, which is the whole difference in a
/// column that is read top to bottom.
fn section(
    regions: &Regions,
    folded: &HashSet<PanelId>,
    id: PanelId,
    body: impl IntoElement,
) -> impl IntoElement {
    let open = !folded.contains(&id);
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(title(regions, folded, id))
        .children(open.then_some(body))
}

/// A section's title bar.
fn title(regions: &Regions, folded: &HashSet<PanelId>, id: PanelId) -> impl IntoElement {
    let open = !folded.contains(&id);
    div()
        .relative()
        .flex()
        .justify_between()
        .items_center()
        .pt_2()
        .cursor_pointer()
        .child(probe(regions, Region::Fold(id)))
        .child(div().text_sm().text_color(rgb(0x9aa0a6)).child(id.title()))
        .child(
            div()
                .text_xs()
                .text_color(rgb(0x6c7378))
                // Down for open, right for folded — which way the content lies, not
                // which way pressing it would go.
                .child(if open { "\u{25be}" } else { "\u{25b8}" }),
        )
}

// --- what the layout actually is ------------------------------------------
//
// **The panel does not compute its own geometry, it reports it.** The first cut of
// this hand-derived every offset from the Tailwind-shaped spacing the tree asks for —
// and was wrong: a press where the arithmetic said "Airbrush" selected Hard Eraser,
// because the guessed row pitch was 26 px and Taffy had laid out 39. Two descriptions
// of one layout is exactly the drift `stark-chrome` exists to stop, one scale down,
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
    Knob(Knob),
    Effect(usize),
    Preset(usize),
    /// A section's title bar — pressing it folds the section away.
    Fold(PanelId),
}

/// Where each control was laid out, as of the last frame that painted.
///
/// `Rc<RefCell<..>>` because a prepaint closure is `'static` and the view outlives it:
/// wgpui hands the closure no context to write through, so what it writes into has to
/// be something it owns a handle to.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

/// An invisible element that records its parent's bounds.
///
/// `absolute().inset_0()` so it takes the whole of the control it sits in and adds
/// nothing to the layout. Its paint is empty: what is wanted is the *measurement*,
/// and the control draws itself.
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

/// How far along a knob's own track a position is, `0..=1`.
///
/// Read from the x alone and clamped, which is what lets a drag wander off the track
/// vertically and still move the knob it took hold of. `None` when that knob has not
/// been laid out — there is nothing to be a fraction *of*.
pub fn fraction_at(regions: &Regions, knob: Knob, at: Point<Pixels>) -> Option<f32> {
    let bounds = regions
        .borrow()
        .iter()
        .find(|(r, _)| *r == Region::Knob(knob))
        .map(|(_, b)| *b)?;
    let left = f32::from(bounds.origin.x);
    let width = f32::from(bounds.size.width);
    (width > 0.0).then(|| ((f32::from(at.x) - left) / width).clamp(0.0, 1.0))
}

/// Whether a position is over the panel's column at all — the canvas begins where
/// this ends, so a press the panel does not want is paint.
///
/// The one measurement that is *not* read back off the layout, because it is what the
/// layout is told: the column is `WIDTH` wide because this module says so.
pub fn within(at: Point<Pixels>) -> bool {
    (0.0..=WIDTH).contains(&f32::from(at.x))
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
    use wgpui::{point, size};

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
                    size: size(px(WIDTH - 2.0 * PADDING), px(*height)),
                },
            ));
        }
        regions
    }

    /// A press past the panel's column is the canvas's, whatever it is level with.
    #[test]
    fn a_press_past_the_panel_is_paint() {
        assert!(!within(at(WIDTH + 1.0, 100.0)));
        assert!(within(at(WIDTH - 1.0, 100.0)));
    }

    /// Before the first frame has painted there is nothing measured, and a press
    /// finds no control rather than guessing at one.
    #[test]
    fn nothing_is_hit_before_a_frame_has_been_laid_out() {
        let regions = Regions::default();
        assert_eq!(hit(&regions, at(WIDTH / 2.0, 100.0)), None);
        assert_eq!(
            fraction_at(&regions, Knob::Size, at(WIDTH / 2.0, 100.0)),
            None
        );
    }

    /// A press finds the control whose measured rectangle it is inside, and nothing
    /// between two of them.
    #[test]
    fn a_press_finds_the_control_it_is_inside() {
        let regions = measured(&[
            (Region::Knob(Knob::Size), 60.0, 18.0),
            (Region::Knob(Knob::Flow), 110.0, 18.0),
            (Region::Preset(3), 300.0, 26.0),
        ]);
        let x = WIDTH / 2.0;
        assert_eq!(hit(&regions, at(x, 66.0)), Some(Region::Knob(Knob::Size)));
        assert_eq!(hit(&regions, at(x, 118.0)), Some(Region::Knob(Knob::Flow)));
        assert_eq!(hit(&regions, at(x, 310.0)), Some(Region::Preset(3)));
        assert_eq!(
            hit(&regions, at(x, 95.0)),
            None,
            "the gap belongs to nobody"
        );
    }

    /// A track reads left to right across its own measured width, and clamps rather
    /// than extrapolating — which is what lets a drag leave the track and keep going.
    #[test]
    fn a_track_reads_left_to_right_and_clamps() {
        let regions = measured(&[(Region::Knob(Knob::Size), 60.0, 18.0)]);
        let f = |x: f32| fraction_at(&regions, Knob::Size, at(x, 66.0)).expect("measured");
        assert!(f(PADDING) < 0.01);
        assert!(f(WIDTH - PADDING) > 0.99);
        assert_eq!(f(-500.0), 0.0);
        assert_eq!(f(WIDTH + 500.0), 1.0);
        // Vertically anywhere: the drag has left the track and still moves the knob.
        assert_eq!(
            f(PADDING),
            fraction_at(&regions, Knob::Size, at(PADDING, 900.0)).unwrap()
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
}
