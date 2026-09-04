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
//! wgpui has a path builder, so the shapes are paths: the affine's **ellipse** — the
//! image of the reference circle under the accumulated linear map, so its eccentricity
//! *is* the distortion — the perspective's **quad** with its receding grid, and the
//! warp's **mesh**. The same three the web frontend draws in SVG, from the same
//! numbers, which is the point: two drawings of one geometry is not duplication, two
//! *derivations* of it would be.
//!
//! Every point is mapped canvas → screen through the live [`ViewTransform`], so a pan
//! or a zoom mid-gesture moves the widget with the paint rather than away from it.

use stark_engine::ViewTransform;
use stark_model::geom::Vec2;
use stark_ui::commands::{Bindings, Command};
use stark_ui::transform::{Family, Hint, TransformUi, WARP_GRID};
use wgpui::{
    Bounds, HitboxBehavior, IntoElement, PathBuilder, Pixels, Point, canvas, div, prelude::*, px,
    rgb, rgba,
};

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
    regions.borrow_mut().clear();
    let family = ui.family();
    div()
        .absolute()
        .top_0()
        .left_0()
        .right_0()
        .flex()
        .gap_1()
        .p_2()
        .bg(rgba(0x1e2124e0))
        .border_b_1()
        .border_color(rgb(0x35393d))
        .text_color(rgb(0xe8eaed))
        .child(
            div()
                .py_1()
                .px_2()
                .text_xs()
                .text_color(rgb(0x9aa0a6))
                .child("Transform"),
        )
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
            let word = match command.shortcut(bindings) {
                Some(chord) => format!("{}  {chord}", command.word()),
                None => command.word().to_string(),
            };
            div()
                .relative()
                .py_1()
                .px_2()
                .rounded_sm()
                .bg(rgb(0x2a2d31))
                .text_xs()
                .text_color(rgb(0xb0b4b8))
                .cursor_pointer()
                .child(probe(regions, Region::Act(i)))
                .child(word)
        }))
}

fn chip(probe: impl IntoElement, word: &str, lit: bool) -> impl IntoElement {
    div()
        .relative()
        .py_1()
        .px_2()
        .rounded_sm()
        .text_xs()
        .cursor_pointer()
        .when_else(
            lit,
            |el| el.bg(rgb(0x35496b)).text_color(rgb(0xe8eaed)),
            |el| el.bg(rgb(0x2a2d31)).text_color(rgb(0xb0b4b8)),
        )
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
            match ui {
                TransformUi::Affine { ts, .. } => {
                    // The image of the reference circle. Sampled rather than fitted
                    // with arcs: under a shear it is an ellipse at an angle, which no
                    // axis-aligned arc primitive can state.
                    let ring: Vec<_> = (0..ELLIPSE_STEPS)
                        .map(|i| {
                            let t = i as f32 / ELLIPSE_STEPS as f32 * std::f32::consts::TAU;
                            at(ts.center + ts.linear * (ts.radius * Vec2::new(t.cos(), t.sin())))
                        })
                        .collect();
                    stroke(window, &ring, true, LINE, RIM);
                    // The centre, so a translate has something to aim at.
                    let c = at(ts.center);
                    handle(window, c);
                }
                TransformUi::Perspective(p) => {
                    let quad: Vec<_> = p.corners.iter().map(|c| at(*c)).collect();
                    stroke(window, &quad, true, LINE, RIM);
                    // The receding grid: the transformed space itself, so the lines
                    // say what the map is doing *between* the corners rather than
                    // only at them. Absent while the quad is unusable — a concave one
                    // has no homography to draw, which the corners already show.
                    if let Some(h) = p.map().forward() {
                        let span = p.rect.1 - p.rect.0;
                        for i in 1..GRID_LINES {
                            let f = i as f32 / GRID_LINES as f32;
                            for axis in 0..2 {
                                let run: Vec<_> = (0..=GRID_STEPS)
                                    .map(|j| {
                                        let g = j as f32 / GRID_STEPS as f32;
                                        let uv = if axis == 0 {
                                            Vec2::new(f, g)
                                        } else {
                                            Vec2::new(g, f)
                                        };
                                        at(h.apply(p.rect.0 + span * uv))
                                    })
                                    .collect();
                                stroke(window, &run, false, HAIR, GRID);
                            }
                        }
                    }
                    for c in quad {
                        handle(window, c);
                    }
                }
                TransformUi::Warp(w) => {
                    // The mesh's own rows and columns, through the control points —
                    // which is where the hand grabs, so it is what should be drawn.
                    for i in 0..WARP_GRID {
                        for axis in 0..2 {
                            let run: Vec<_> = (0..WARP_GRID)
                                .map(|j| {
                                    let (r, c) = if axis == 0 { (i, j) } else { (j, i) };
                                    at(w.points[r * WARP_GRID + c])
                                })
                                .collect();
                            stroke(window, &run, false, HAIR, GRID);
                        }
                    }
                    for p in w.points {
                        handle(window, at(p));
                    }
                }
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

/// How many points the affine's ellipse is sampled at. Enough that the polygon reads
/// as a curve at any zoom the widget is usable at.
const ELLIPSE_STEPS: usize = 96;

/// Interior lines drawn through the perspective quad, per axis.
const GRID_LINES: usize = 4;

/// How finely one of those is sampled — a homography is not linear in the plane, so a
/// straight segment between its ends would not lie on it.
const GRID_STEPS: usize = 16;

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

/// Stroke a polyline, closing it when `closed`.
fn stroke(window: &mut wgpui::Window, points: &[Point<Pixels>], closed: bool, w: f32, color: u32) {
    if points.len() < 2 {
        return;
    }
    let mut path = PathBuilder::stroke(px(w));
    path.add_polygon(points, closed);
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
