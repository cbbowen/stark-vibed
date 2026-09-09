//! The transform mode's native chrome (§16.6, §16.8, §16.9): a bar over the canvas,
//! and the widget drawn on it.
//!
//! **Almost nothing is decided here.** What a press takes hold of, what a drag makes
//! of it, what switching families costs, where the widget mounts and how big its grab
//! bands are — all of it is `stark_ui::transform`, and the whole of the drag
//! reaching this frontend is `Grab::take` on the press and `Grab::follow` on the move.
//! What is left is what a frontend alone can do: put a bar somewhere, draw three
//! shapes, and say which cursor its own toolkit spells a [`Hint`] with.
//!
//! # Drawing the widget
//!
//! wgpui has a path builder, so the shapes are paths — but *which* polylines is not
//! this file's answer any more. `stark_ui::transform::{outline, grid, handles}` gives
//! them in canvas px and what is left here is the stroking: a weight, a colour, and
//! the canvas → screen mapping through the live [`ViewTransform`], so a pan or a zoom
//! mid-gesture moves the widget with the paint rather than away from it.
//!
//! That is a correction, not a tidy-up. This file drew the mesh through the control
//! points — where the hand grabs — and §16.9 says the opposite: the curves are
//! sampled from the surface the paint resamples through, so a straight grid means
//! untouched, and a mesh bent between its points drew here as if nothing had
//! happened. The perspective grid had the milder version, a line count and a sampling
//! of its own for a map whose lines are straight by definition. Two drawings of one
//! geometry is not duplication; two *derivations* of it is how they came to disagree.

use stark_engine::ViewTransform;
use stark_model::geom::Vec2;
use stark_ui::commands::{Bindings, Command};
use stark_ui::transform::{Family, Hint, TransformUi};
use wgpui::{
    Bounds, HitboxBehavior, IntoElement, PathBuilder, Pixels, Point, SharedString, canvas, div,
    prelude::*, px, rgb, rgba,
};

use crate::style::{self, StyleExt};

/// The three families, in the order the bar draws them, with the word each wears.
///
/// The words are this frontend's rather than the registry's because a family is not a
/// command — the bar sets which one is composing, and nothing else reaches them.
pub const FAMILIES: [(Family, &str); 3] = [
    (Family::Free, "Free"),
    (Family::Perspective, "Perspective"),
    (Family::Warp, "Warp"),
];

/// The two mirrors, offered only under the affine family — the other two maps
/// preserve orientation, so there is nothing there for a mirror to be.
/// Named by axis rather than by arrows: the system font this frontend renders with
/// has the horizontal arrow and not the vertical one, so one of the pair drew as a
/// tofu box. A glyph a frontend cannot guarantee is worse than the word it replaced.
pub const FLIPS: [&str; 2] = ["Flip H", "Flip V"];

/// The way out and the way through, both worn off the registry so the words and the
/// chords they advertise are the ones every other surface uses.
pub const BAR_ACTS: [Command; 2] = [Command::CancelMode, Command::FinishMode];

/// Which of the bar's controls a press landed on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    /// One of [`FAMILIES`], by index.
    Family(usize),
    /// One of [`FLIPS`], by index — 0 horizontal, 1 vertical.
    Flip(usize),
    /// One of [`BAR_ACTS`], by index.
    Act(usize),
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

/// The bar that fronts the mode.
///
/// Along the top of the canvas rather than the bottom: this frontend's canvas has
/// chrome on both sides and nothing above it, so the top is the one edge where a bar
/// costs no painting room it was not already costing.
pub fn bar(ui: TransformUi, bindings: &Bindings, regions: &Regions) -> impl IntoElement {
    let family = ui.family();
    div()
        .absolute()
        .top_0()
        .left_0()
        .right_0()
        .flex()
        .gap_1()
        .p_2()
        .bg(rgba(style::PANEL_OVER_CANVAS))
        .border_b_1()
        .border_color(rgb(style::EDGE))
        .text_color(rgb(style::INK_LIT))
        .child(div().py_1().px_2().caption().child("Transform"))
        .children(
            FAMILIES
                .iter()
                .enumerate()
                .map(|(i, (f, word))| chip(probe(regions, Region::Family(i)), word, *f == family)),
        )
        // Under the affine only, and mounted rather than dimmed: a mirror is not an
        // act that is unavailable in the other two families, it is one that has no
        // meaning there.
        .children(
            FLIPS
                .iter()
                .enumerate()
                .filter(|_| family == Family::Free)
                .map(|(i, word)| chip(probe(regions, Region::Flip(i)), word, false)),
        )
        .children(BAR_ACTS.iter().enumerate().map(|(i, command)| {
            // The word alone; the chord is the hover's (`style::tip`).
            let chip = div()
                .id(SharedString::from(command.word()))
                .chip()
                .py_1()
                .px_2()
                .resting()
                .child(probe(regions, Region::Act(i)))
                .child(command.word());
            style::tip(chip, command.tooltip(bindings))
        }))
}

fn chip(probe: impl IntoElement, word: &str, lit: bool) -> impl IntoElement {
    div()
        .chip()
        .py_1()
        .px_2()
        .lit(lit)
        .child(probe)
        .child(word.to_string())
}

/// The widget itself, drawn over the canvas column.
///
/// `scale` converts the view's device px to the logical px a layout is denominated in
/// — the same conversion `crate::canvas::sample_at` makes in the other direction, and
/// the only place this frontend's two unit systems meet (§11.2).
pub fn overlay(ui: TransformUi, view: ViewTransform, scale: f32, hint: Hint) -> impl IntoElement {
    canvas(
        // A hitbox over the canvas column, so the cursor `hint` asks for applies here
        // and not over the two panels — which is the whole reason to take one rather
        // than set the window's.
        move |bounds, window, _| window.insert_hitbox(bounds, HitboxBehavior::Normal),
        move |bounds, hitbox, window, _| {
            window.set_cursor_style(cursor(hint), &hitbox);
            let at = |p: Vec2| {
                let s = view.canvas_to_screen(p) / scale;
                wgpui::point(bounds.origin.x + px(s.x), bounds.origin.y + px(s.y))
            };
            let run = |window: &mut wgpui::Window, points: &[Vec2], w: f32, color: u32| {
                let screen: Vec<_> = points.iter().copied().map(at).collect();
                stroke(window, &screen, w, color);
            };
            // The lines that say what the map does between the handles first, so the
            // boundary and the handles read over them.
            for line in stark_ui::transform::grid(&ui) {
                run(window, &line, HAIR, GRID);
            }
            for line in stark_ui::transform::outline(&ui) {
                run(window, &line, LINE, RIM);
            }
            for h in stark_ui::transform::handles(&ui) {
                handle(window, at(h));
            }
        },
    )
    // **Pinned on all four sides, not `size_full`.** An absolutely-positioned child
    // with only a size is laid out at wherever the flow had reached, which for a
    // sibling *after* the surface is one full viewport below the window — the widget
    // was drawn correctly and off-screen for exactly one build.
    .absolute()
    .top_0()
    .left_0()
    .right_0()
    .bottom_0()
}

const LINE: f32 = 1.5;
const HAIR: f32 = 1.0;
const RIM: u32 = 0xd8e2eccc;
/// Bright enough to read over the darkest paint. The interior lines are the only
/// thing saying what the map does *between* the handles, so they have to survive
/// being drawn over a black fill.
const GRID: u32 = 0xa8c4e0aa;
const HANDLE: u32 = 0xe8eaedee;

/// A handle's half-width, logical px. Smaller than the *grab* radius
/// (`stark_ui::transform::HANDLE_PX`) on purpose: a target should be easier to
/// hit than it looks, never harder — asserted below, where a later edit to either
/// figure fails the build rather than a test.
const HANDLE_HALF: f32 = 3.5;

const _: () = assert!(
    HANDLE_HALF < stark_ui::transform::HANDLE_PX,
    "a handle must be grabbable at least as far out as it is drawn"
);

/// Stroke a polyline. A run that closes says so by repeating its first point, which
/// is `stark_ui::transform`'s contract — so there is no flag here to get wrong.
fn stroke(window: &mut wgpui::Window, points: &[Point<Pixels>], w: f32, color: u32) {
    if points.len() < 2 {
        return;
    }
    let mut path = PathBuilder::stroke(px(w));
    path.add_polygon(points, false);
    // A path that will not build is a degenerate one — a widget collapsed to a
    // sliver, which the shaping clamps make transient. Dropping the frame's line is
    // the whole of what a frontend can do about it, and better than a panic.
    if let Ok(built) = path.build() {
        window.paint_path(built, rgba(color));
    }
}

/// A square handle centred on `at`.
fn handle(window: &mut wgpui::Window, at: Point<Pixels>) {
    let h = px(HANDLE_HALF);
    window.paint_quad(wgpui::fill(
        Bounds::new(
            wgpui::point(at.x - h, at.y - h),
            wgpui::size(h * 2., h * 2.),
        ),
        rgba(HANDLE),
    ));
}

/// How this toolkit spells a [`Hint`].
pub fn cursor(hint: Hint) -> wgpui::CursorStyle {
    match hint {
        Hint::Move => wgpui::CursorStyle::OpenHand,
        Hint::Hold => wgpui::CursorStyle::PointingHand,
        Hint::Shape => wgpui::CursorStyle::Crosshair,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bar names its two acts off the registry, so the word on a button here is
    /// the word the web app's chip wears and the chord hint is the shipped one.
    #[test]
    fn the_bar_wears_the_registrys_acts() {
        assert_eq!(BAR_ACTS[0].word(), "Cancel");
        assert_eq!(BAR_ACTS[1].word(), "Done");
    }

    /// Every family the crate has is on the bar. A chip missing here would be a
    /// family reachable by no gesture at all in this frontend.
    #[test]
    fn every_family_has_a_chip() {
        for f in [Family::Free, Family::Perspective, Family::Warp] {
            assert!(FAMILIES.iter().any(|(g, _)| *g == f), "{f:?} has no chip");
        }
    }
}
