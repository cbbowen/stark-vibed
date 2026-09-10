//! The Navigator: a miniature of the whole piece, the viewport marked on it, and a
//! press to go there (§11).
//!
//! # A shelf here, a corner overlay there
//!
//! The web app's navigator is deliberately not a panel, on three arguments
//! (`navigator` over there): it needs no title, its aspect is the artwork's rather
//! than a column's, and it is read rather than operated. A **docked** chrome answers
//! the first two differently — nothing here floats, so a corner would be canvas taken
//! rather than borrowed, and the column shrink-wraps its shelves. The third survives,
//! and is why it leads the reading column rather than sitting in the queue of things
//! reached for between strokes.
//!
//! It is still not a `PanelId` but a [`VisibilityToggle`] (`crate::visibility`), which
//! is the vocabulary both apps already name it in.
//!
//! [`VisibilityToggle`]: stark_ui::commands::VisibilityToggle
//!
//! # What it is a picture of
//!
//! Exactly what an export would write (§15.6), by being the *same call*:
//! `Renderer::overview_plan` is `Engine::export_plan`, and the plan it returns is the
//! view the miniature renders through — so the overview cannot come to disagree with
//! the picture a file would hold.
//!
//! It is a second `WgpuSurface` rather than an image, so this module holds **no
//! pixels**: four numbers saying where the picture sits in canvas space, and a refresh
//! is one render and a pointer swap.
//!
//! # Why it does not follow the canvas
//!
//! One refresh composites every tile in the document — nothing on an edit, ruinous per
//! pointer sample. So it is a picture of the **committed** document, refreshed when
//! that moves, never while a gesture is in flight, and at most once every [`stark_ui::bounds::SETTLE`]
//! seconds so a held undo collapses into one render rather than thirty. The marker
//! over the top is painted from the live view instead, so panning costs nothing.

use stark_engine::{Extent2, ViewTransform};
use stark_ui::bounds::{Overview, marker};
use wgpui::{
    Bounds, IntoElement, PathBuilder, Pixels, Point, WgpuSurfaceHandle, canvas, div, point,
    prelude::*, px, rgb, wgpu_surface,
};

use crate::style;

/// The box the miniature is fitted into, in logical px — the largest it is ever
/// drawn, on whichever axis the piece runs out of first.
///
/// A box rather than a width, and both numbers are caps in their own right: the shelf
/// shrink-wraps whatever comes back, so a landscape piece spends the width, a portrait
/// one the height, and neither pays for the axis it does not use.
///
/// The width is the reading column's, less its padding. The height is a bargain with
/// the shelves under it: every pixel here is a layer row they do not get, and an
/// overview too small to find the marker in is not worth the ones it does spend.
pub const MAX_WIDTH: f32 = crate::panel::RIGHT_CONTENT;
pub const MAX_HEIGHT: f32 = 168.0;

/// The width of the viewport marker's outline, logical px.
const MARKER_WIDTH: f32 = 1.5;

/// The box a plan is asked to fit, in the **device** px a plan is denominated in.
///
/// This frontend's, not the overview's: [`MAX_WIDTH`] and [`MAX_HEIGHT`] are a docked
/// column's bargain with the shelves under it, and the web app's corner overlay makes a
/// different one (§11.2).
pub fn box_for(scale: f32) -> Extent2 {
    let scale = if scale > 0.0 { scale } else { 1.0 };
    Extent2::new(
        ((MAX_WIDTH * scale).round() as u32).max(1),
        ((MAX_HEIGHT * scale).round() as u32).max(1),
    )
}

/// Which control a press on the shelf landed on. One, and it is the picture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    Miniature,
}

/// Where it was laid out — `crate::panel`'s device, for its reason.
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

/// Where in the miniature a position sits, as fractions of it — clamped, so a drag
/// that has left the box keeps moving the view it took hold of.
pub fn fraction_at(regions: &Regions, at: Point<Pixels>) -> Option<(f32, f32)> {
    let bounds = regions
        .borrow()
        .iter()
        .find(|(r, _)| *r == Region::Miniature)
        .map(|(_, b)| *b)?;
    let (w, h) = (f32::from(bounds.size.width), f32::from(bounds.size.height));
    (w > 0.0 && h > 0.0).then(|| {
        (
            ((f32::from(at.x) - f32::from(bounds.origin.x)) / w).clamp(0.0, 1.0),
            ((f32::from(at.y) - f32::from(bounds.origin.y)) / h).clamp(0.0, 1.0),
        )
    })
}

/// Build the shelf's body.
///
/// `surface` is `None` before the first miniature has been drawn — the frame between
/// the window opening and the engine's first committed picture, which is a real frame.
/// The box is laid out all the same, so the shelf does not change shape under the
/// first render.
pub fn navigator_body(
    over: Option<Overview>,
    surface: Option<WgpuSurfaceHandle>,
    view: Option<ViewTransform>,
    regions: &Regions,
) -> impl IntoElement + use<> {
    let corners = over.zip(view).map(|(o, v)| marker(o, v));
    let (w, h) = over.map_or((MAX_WIDTH, MAX_HEIGHT * 0.5), |o| (o.width, o.height));
    style::tip(
        div().id("navigator").flex().justify_center().child(
            div()
                .relative()
                .w(px(w))
                .h(px(h))
                .rounded_sm()
                .overflow_hidden()
                .bg(rgb(style::WELL))
                .cursor_pointer()
                .child(probe(regions, Region::Miniature))
                .children(surface.map(|handle| wgpu_surface(handle).size_full()))
                .children(corners.map(outline)),
        ),
        "Navigator \u{2014} press or drag to move the view around the piece",
    )
}

/// The viewport marker, painted rather than positioned.
///
/// A path, because the rectangle is *turned*: an absolutely-placed box could say where
/// the view is and not which way up it is, and which way up is half of what an
/// overview answers once the easel can be turned (§18.1.2).
fn outline(corners: [(f32, f32); 4]) -> impl IntoElement {
    canvas(
        move |_, _, _| {},
        move |bounds: Bounds<Pixels>, (), window, _| {
            let at = |(fx, fy): (f32, f32)| {
                point(
                    bounds.origin.x + px(fx * f32::from(bounds.size.width)),
                    bounds.origin.y + px(fy * f32::from(bounds.size.height)),
                )
            };
            let mut path = PathBuilder::stroke(px(MARKER_WIDTH));
            path.move_to(at(corners[0]));
            for corner in &corners[1..] {
                path.line_to(at(*corner));
            }
            path.close();
            if let Ok(path) = path.build() {
                window.paint_path(path, rgb(style::ACCENT));
            }
        },
    )
    .absolute()
    .size_full()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 400×200 piece under a miniature, and a viewport looking at it.
    /// The box a plan is fitted into is the shelf's own, in the device px a plan is
    /// denominated in — and never zero, which is not a texture.
    #[test]
    fn the_fitting_box_is_the_shelfs_own() {
        let at_1x = box_for(1.0);
        assert_eq!(at_1x.width, MAX_WIDTH.round() as u32);
        assert_eq!(at_1x.height, MAX_HEIGHT.round() as u32);
        let at_2x = box_for(2.0);
        assert_eq!(at_2x.width, at_1x.width * 2);
        assert!(box_for(0.0).width >= 1);
    }
}
