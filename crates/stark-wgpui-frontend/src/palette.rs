//! The command search (§11.1): the web app's palette, as a menu bar draws one.
//!
//! A field at the bar's right end and, while it has focus and holds a query, the
//! registry's answers under it (`stark_ui::commands::search`) — each a row that runs
//! on a press, the first on Enter. It reaches every act the registry names, which is
//! more than the menus beside it list; what this frontend cannot yet do with one is
//! `Canvas::run`'s to decline, and a row for such an act is dimmed rather than gone.
//!
//! The rows act on the **press**, for §25.7's reason: the field loses focus on the
//! same press, and the drop-down goes with the focus. Acting on the press is how a
//! row wins that race. The press is then stopped, or the canvas under the drop-down
//! would hear it too and open a stroke.

use std::rc::Rc;

use stark_engine::ObservableState;
use stark_ui::commands::{Bindings, Command};
use wgpui::{App, Entity, IntoElement, MouseButton, Window, deferred, div, prelude::*, px, rgb};
use wgpui_component::Sizable;
use wgpui_component::input::{Input, InputState};

use crate::menu;
use crate::style::{self, StyleExt};

/// How many answers the drop-down shows. The registry ranks them, so the first few
/// are the ones a query was for.
pub const SHOWN: usize = 8;

/// The field's width in logical px: room for a query, not for a sentence.
const WIDTH: f32 = 200.0;

/// What runs a row. `Rc` so the one closure the view builds per frame reaches every
/// row the drop-down holds.
pub type Run = Rc<dyn Fn(Command, &mut Window, &mut App)>;

/// What the field is built from, this frame.
pub struct Search<'a> {
    pub field: &'a Entity<InputState>,
    /// Whether the field has focus — the drop-down shows for exactly as long.
    pub open: bool,
    /// The registry's answers to what the field holds, already cut to [`SHOWN`].
    pub results: &'a [Command],
    pub obs: Option<&'a ObservableState>,
    pub bindings: &'a Bindings,
    pub run: Run,
}

/// The field, and the drop-down hung from the bar beneath it when there is one.
pub fn field(search: Search<'_>) -> impl IntoElement {
    let Search {
        field,
        open,
        results,
        obs,
        bindings,
        run,
    } = search;
    div()
        .w(px(WIDTH))
        .child(Input::new(field).xsmall())
        .when(open && !results.is_empty(), |el| {
            // Hung from the bar rather than the field, like the menus' own
            // drop-down, and `deferred` for the same reason: the bar is painted
            // under everything after it (`crate::menu`).
            el.child(deferred(
                div()
                    .absolute()
                    .top(px(menu::HEIGHT))
                    .right_0()
                    .w(px(WIDTH + 120.0))
                    .flex()
                    .flex_col()
                    .py_1()
                    .bg(rgb(style::MENU))
                    .border_1()
                    .border_color(rgb(style::RULE))
                    .rounded_sm()
                    .children(
                        results
                            .iter()
                            .enumerate()
                            .map(|(i, command)| row(i, *command, obs, bindings, run.clone())),
                    ),
            ))
        })
}

/// One answer: the act's name, its chord beside it, dimmed when there is nothing
/// for it to act on — the menu's rows, minus the tick.
fn row(
    i: usize,
    command: Command,
    obs: Option<&ObservableState>,
    bindings: &Bindings,
    run: Run,
) -> impl IntoElement {
    let live = command.enabled(obs);
    div()
        .id(("search", i))
        .chip()
        .flex()
        .justify_between()
        .gap_3()
        .px_2()
        .py_0p5()
        .text_sm()
        .text_color(rgb(if live {
            style::INK_ROW
        } else {
            style::INK_DEAD
        }))
        .hover(|s| s.bg(rgb(style::HOVER)))
        .child(command.name())
        .when_some(command.shortcut(bindings), |el, chord| {
            el.child(
                div()
                    .text_color(rgb(if live {
                        style::INK_CHORD
                    } else {
                        style::INK_CHORD_DEAD
                    }))
                    .child(chord),
            )
        })
        .when(live, |el| {
            el.on_mouse_down(MouseButton::Left, move |_, window, cx| {
                cx.stop_propagation();
                run(command, window, cx);
            })
        })
}
