//! The window's menu bar (§11.2 N8, §25).
//!
//! **Drawn rather than installed.** wgpui carries a `Menu` type and a `set_menus`,
//! but its implementation installs anything only on macOS — everywhere else it stores
//! the menus and shows nothing. So this is an in-app bar, which is what Zed, Blender
//! and Krita all draw on Windows and Linux anyway, and it is the one surface this
//! frontend has that the web app does not want: a browser tab has a menu bar above it
//! already, and the web app answers the same need with a search palette (§25.4).
//!
//! # What is shared and what is not
//!
//! Everything *about* a row is the registry's — its name, its mark, the chord that
//! also reaches it, and whether the document makes it available
//! (`Command::enabled`). A menu row and a panel button are then the same control
//! reached two ways rather than two controls that resemble each other. The one thing
//! the registry cannot answer is whether this *window* is in the state a row names,
//! which is why the tick is asked for rather than looked up (`Canvas::active`).
//!
//! **Which** commands, and in which menu, is this frontend's. That is not a
//! reluctance to share: the menus differ between the two apps *because the apps
//! implement different subsets* — there is no `NewDocument` here and no `ImportImage`
//! — so a shared table would be a list of things one of them has to filter. The day
//! the web app grows a menu bar, what moves down is whatever the two then agree about.
//!
//! Share and Join are where that filtering runs the other way. Both apps answer both
//! acts, but only one of them can be reached from a menu here, because a browser tab
//! joins by *being opened* on a link and a window has no such door (§12.4,
//! `crate::collab`).

use stark_engine::ObservableState;
#[cfg(test)]
use stark_ui::commands::VisibilityToggle;
use stark_ui::commands::{Bindings, Command};
use stark_ui::panels::PanelId;
use wgpui::{
    AnyElement, Bounds, IntoElement, Pixels, Point, canvas, deferred, div, prelude::*, px, rgb,
};

use crate::style::{self, StyleExt};

/// The bar's height, logical px.
pub const HEIGHT: f32 = 26.0;

/// A separator between two runs of a menu, rather than a command.
///
/// A `None` in the row list, so a menu is one array read top to bottom — where a
/// second list of "rules after index 3" would be a thing to keep in step with the
/// first as rows are added.
type Row = Option<Command>;

/// One menu: its title, and the rows under it.
pub struct Menu {
    pub title: &'static str,
    pub rows: &'static [Row],
}

/// The bar, left to right.
///
/// Every row is a command this frontend actually answers (`Canvas::run`) — a menu
/// that offers a dead act is worse than one that is short, because the act looks
/// available and does nothing. The absences are therefore real: no New, no Import, and
/// a View menu of one row, because one is what this frontend answers.
pub const MENUS: &[Menu] = &[
    Menu {
        title: "File",
        rows: &[
            Some(Command::OpenDocument),
            Some(Command::SaveDocument),
            None,
            Some(Command::ExportImage),
            None,
            // The two ends of one invitation (§12.4): Share puts a link on the
            // clipboard, Join takes one off it. In the File menu because that is where
            // the acts that move the whole document in and out of this window live —
            // sharing hands the painting to someone else exactly as saving hands it to
            // a disk, and a person looking for either looks in the same place.
            Some(Command::Share),
            Some(Command::Join),
        ],
    },
    Menu {
        title: "Edit",
        rows: &[Some(Command::Undo), Some(Command::Redo)],
    },
    Menu {
        title: "View",
        // The HDR switch (§6.5); Mirror waits until this frontend answers it.
        rows: &[Some(Command::ToggleHdr)],
    },
    Menu {
        title: "Select",
        rows: &[
            // The three shape tools, then the acts on the region they draw — the
            // Select panel's own order, because it is the same argument: a tool is
            // for making a selection and the rest are for what you then do with one.
            Some(Command::SelectRect),
            Some(Command::SelectEllipse),
            Some(Command::SelectLasso),
            None,
            Some(Command::Deselect),
            Some(Command::InvertSelection),
            None,
            Some(Command::Transform),
            Some(Command::FloatSelection),
            Some(Command::FillSelection),
        ],
    },
    Menu {
        title: "Brush",
        rows: &[Some(Command::BrushSmaller), Some(Command::BrushLarger)],
    },
    Menu {
        title: "Window",
        // What is on screen (§25.5): the web app's visibility menu, as a menu bar
        // draws one. Its rows carry a state where every other row here is an act, so
        // they are the reason a row can wear a tick at all ([`item`]).
        //
        // The seven shelves this frontend has, and no more — a row offering to show a
        // quick-brush rack it does not draw would be exactly the dead act the rule
        // above forbids. The list is `crate::visibility`'s, so what the menu offers
        // and what the record can hold cannot come apart; the rule between the runs is
        // where the window's two columns divide.
        rows: &[
            Some(Command::TogglePanel(PanelId::Brush)),
            Some(Command::TogglePanel(PanelId::Select)),
            Some(Command::TogglePanel(PanelId::Lighting)),
            Some(Command::TogglePanel(PanelId::Guides)),
            None,
            Some(Command::ToggleNavigator),
            Some(Command::TogglePanel(PanelId::Color)),
            Some(Command::TogglePanel(PanelId::Layers)),
        ],
    },
];

/// What a press on the bar landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// The bar itself, so a drop-down can be placed relative to it.
    Bar,
    /// A menu's title, by its index in [`MENUS`].
    Title(usize),
    /// A row of the open menu, by menu index and row index.
    Row(usize, usize),
}

/// Where the bar's controls were laid out — `crate::panel`'s device, for its reason.
pub type Regions = std::rc::Rc<std::cell::RefCell<Vec<(Region, Bounds<Pixels>)>>>;

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

/// Which control a press landed on. `None` for a press anywhere else, which is what
/// closes an open menu — a menu that stayed up through a click on the canvas would be
/// a menu the artist has to dismiss before painting.
pub fn hit(regions: &Regions, at: Point<Pixels>) -> Option<Region> {
    regions
        .borrow()
        .iter()
        .rev()
        .find(|(_, bounds)| bounds.contains(&at))
        .map(|(region, _)| *region)
}

/// The command a row names, if that row is one.
pub fn command(menu: usize, row: usize) -> Option<Command> {
    *MENUS.get(menu)?.rows.get(row)?
}

/// Build the bar. `open` is the menu whose rows are showing, if any.
///
/// The drop-down is part of this element rather than a sibling of it, and the bar is
/// `relative` so the panel can hang off a title's own left edge.
///
/// **The panel is `deferred`**, which is the whole reason that works. The bar is the
/// window's *first* child, because it is the top row of a column — and a first child
/// is painted under everything after it, so the drop-down hung off it was drawn and
/// then covered by the canvas. `deferred` keeps the layout where it is and moves the
/// paint to after every ancestor, which is the one thing that wants two different
/// answers to "where in the tree is this".
///
/// `state` answers "am I in it?" for a row that has such a state and `None` for one
/// that does not — the web frontend's `commands::active`, asked of this window
/// (`Canvas::active`). A `dyn` rather than an `impl Fn` because there is one call
/// site and the return type deliberately captures nothing.
///
/// `search` is the command field at the bar's right end (`crate::palette`), built by
/// the caller because it is a view's: which answers it shows is the view's state.
pub fn bar(
    open: Option<usize>,
    obs: Option<&ObservableState>,
    bindings: &Bindings,
    state: &dyn Fn(Command) -> Option<bool>,
    regions: &Regions,
    search: AnyElement,
) -> impl IntoElement + use<> {
    div()
        .relative()
        .w_full()
        .h(px(HEIGHT))
        .flex()
        .items_center()
        .px_1()
        .bg(rgb(style::BAR))
        .border_b_1()
        .border_color(rgb(style::EDGE))
        .child(probe(regions, Region::Bar))
        .children(MENUS.iter().enumerate().map(|(i, menu)| {
            div()
                .chip()
                .px_2()
                .py_0p5()
                .lit_row(open == Some(i))
                .child(probe(regions, Region::Title(i)))
                .child(menu.title)
        }))
        .child(div().flex_1())
        .child(search)
        .children(
            open.and_then(|i| MENUS.get(i).map(|menu| (i, menu)))
                .map(|(i, menu)| {
                    // Whether *this* menu reserves the tick column, decided once for
                    // the whole drop-down rather than per row: a menu where three rows
                    // of ten are indented and the rest are not reads as a ragged list
                    // rather than as a set of switches. So a menu with any state in it
                    // gives every row the gutter, and a menu with none gives it to no
                    // row at all.
                    let gutter = menu.rows.iter().flatten().any(|c| state(*c).is_some());
                    // Hung from the bar's own left edge with the title's offset added,
                    // rather than from the title element: an absolutely-positioned child
                    // is placed where the flow had reached unless it is told otherwise,
                    // which cost a frame's debugging once already (§11.2 N6).
                    deferred(
                        div()
                            .absolute()
                            .top(px(HEIGHT))
                            .left(px(title_x(regions, i)))
                            .flex()
                            .flex_col()
                            .py_1()
                            .min_w(px(200.))
                            .bg(rgb(style::MENU))
                            .border_1()
                            .border_color(rgb(style::RULE))
                            .rounded_sm()
                            .children(menu.rows.iter().enumerate().map(|(j, row)| match row {
                                Some(command) => {
                                    let tick = gutter.then(|| state(*command) == Some(true));
                                    item(*command, i, j, obs, bindings, tick, regions)
                                        .into_any_element()
                                }
                                None => rule().into_any_element(),
                            })),
                    )
                }),
        )
}

/// Where a menu's title starts, as an offset into the bar.
///
/// **Measured, not derived**, which is this frontend's rule and the one a hand-derived
/// panel layout broke once (`crate::panel`). A title's width is a font's answer, and
/// a character count that is close enough today is a drop-down hanging beside its own
/// title the day a menu is renamed.
///
/// Read off the *previous* frame, like every other measurement here: prepaint fills
/// the list after the tree is built, so this frame sees the last one's. The bar does
/// not move between frames, so what comes back is right from the first one — and a
/// menu opened before anything has been measured falls back to the bar's own left
/// edge, which is a drop-down in the wrong place for exactly one frame rather than a
/// panic.
fn title_x(regions: &Regions, menu: usize) -> f32 {
    let held = regions.borrow();
    let left = |region: Region| {
        held.iter()
            .find(|(r, _)| *r == region)
            .map(|(_, b)| f32::from(b.origin.x))
    };
    match (left(Region::Title(menu)), left(Region::Bar)) {
        (Some(title), Some(bar)) => title - bar,
        _ => 0.0,
    }
}

/// One command's row.
///
/// `tick` carries both halves of the switch: `None` where this row's *menu* reserves
/// no tick column at all ([`bar`]), and otherwise whether this row's own command is a
/// state that is on.
fn item(
    command: Command,
    menu: usize,
    row: usize,
    obs: Option<&ObservableState>,
    bindings: &Bindings,
    tick: Option<bool>,
    regions: &Regions,
) -> impl IntoElement {
    // The registry's own gate, so a row greys out for the same reason a panel button
    // does — "nothing to undo", "nothing selected" — rather than for a second one.
    let live = command.enabled(obs);
    let ink = if live {
        style::INK_ROW
    } else {
        style::INK_DEAD
    };
    div()
        .relative()
        .flex()
        .items_center()
        .gap_2()
        .px_2()
        .py_1()
        .text_xs()
        .text_color(rgb(ink))
        .when(live, |el| {
            el.cursor_pointer().hover(|s| s.bg(rgb(style::LIT)))
        })
        .child(probe(regions, Region::Row(menu, row)))
        .children(tick.map(|on| switch(on, ink)))
        .child(crate::icons::icon(command.icon(), ink))
        // The full name here, not the terse word: a menu row stands alone, where a
        // chip sits under a header that has already named the subject (§25).
        .child(div().flex_1().child(command.name()))
        .children(command.shortcut(bindings).map(|chord| {
            div()
                .text_color(rgb(if live {
                    style::INK_CHORD
                } else {
                    style::INK_CHORD_DEAD
                }))
                .child(chord)
        }))
}

/// The tick a row with a state wears, and the space it holds when the state is off.
///
/// A **character**, not [`crate::icons`]' own tick: that one is the mark of *finishing*
/// — every Done chip in the application wears it — and the catalog is explicit that a
/// state and that verb must not be confused by sharing a glyph. What is drawn here is
/// the gutter a native menu bar puts beside a checkable row, which is the same kind of
/// mark as the fold triangle in the panel column and no more an icon than that is.
///
/// The width is the mark's, so a row's word starts in the same place whether its
/// state is on or off.
fn switch(on: bool, ink: u32) -> impl IntoElement {
    div()
        .w(px(crate::icons::SIZE))
        .flex_none()
        .text_color(rgb(ink))
        .children(on.then_some("\u{2713}"))
}

/// A rule between two runs of rows.
fn rule() -> impl IntoElement {
    div().h(px(1.)).my_1().mx_2().bg(rgb(style::RULE))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row offers a command this frontend answers. A menu that lists a dead act
    /// is worse than a short menu: the row looks available and does nothing, and
    /// nothing on screen says which.
    ///
    /// Checked against `Canvas::run`'s own arms by name, which is the closest a test
    /// can get to "this does something" without a window — and it is enough, since a
    /// command reaching `run` with no arm is exactly the failure.
    #[test]
    fn every_row_is_an_act_this_frontend_answers() {
        let answered = include_str!("canvas.rs");
        for menu in MENUS {
            for command in menu.rows.iter().flatten() {
                // The variant, not the value: an arm for a command that carries a
                // payload *binds* the payload rather than spelling it, so a menu of
                // four `TogglePanel` rows shares one arm and this is what they share.
                let debug = format!("{command:?}");
                let head = debug.split('(').next().unwrap_or(&debug);
                let arm = format!("Command::{head}");
                assert!(
                    answered.contains(&format!("{arm} =>"))
                        || answered.contains(&format!("{arm}(")),
                    "{} offers {command:?}, which `Canvas::run` has no arm for",
                    menu.title
                );
            }
        }
    }

    /// The Window menu is the shelves this frontend draws — all of them, once each.
    ///
    /// Both lists are written out by hand and each is edited for its own reason: a
    /// shelf arrives in `visibility::SHELVES` because the window now has one, and in
    /// the menu because somebody remembered. This is the remembering.
    #[test]
    fn the_window_menu_is_every_shelf_this_frontend_has() {
        let listed: Vec<VisibilityToggle> = MENUS
            .iter()
            .find(|m| m.title == "Window")
            .expect("there is a Window menu")
            .rows
            .iter()
            .flatten()
            .map(|command| match command {
                Command::TogglePanel(id) => VisibilityToggle::Panel(*id),
                Command::ToggleNavigator => VisibilityToggle::Navigator,
                other => panic!("the Window menu offers {other:?}, which shows nothing"),
            })
            .collect();
        assert_eq!(listed, crate::visibility::SHELVES.to_vec());
    }

    /// A row's name comes off the registry, so a menu entry and the palette row the
    /// web app shows for the same act cannot come to say different things.
    #[test]
    fn a_row_is_named_by_the_registry() {
        assert_eq!(Command::ExportImage.name(), "Export image\u{2026}");
        assert_eq!(command(0, 3), Some(Command::ExportImage));
        assert_eq!(command(0, 5), Some(Command::Share));
        assert_eq!(command(0, 6), Some(Command::Join));
        // A rule is a row with no command, and reading one is not an error.
        assert_eq!(command(0, 2), None);
        assert_eq!(command(99, 0), None);
    }

    /// A menu opened before the bar has ever been laid out hangs at the bar's own
    /// left edge rather than panicking — which is the frame between a window opening
    /// and its first prepaint, and is reachable by a fast hand.
    #[test]
    fn an_unmeasured_bar_still_places_its_menu() {
        assert_eq!(title_x(&Regions::default(), 0), 0.0);
    }
}
