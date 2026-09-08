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
//! that moves, never while a gesture is in flight, and at most once every [`SETTLE`]
//! seconds so a held undo collapses into one render rather than thirty. The marker
//! over the top is painted from the live view instead, so panning costs nothing.

use stark_engine::{Extent2, ViewTransform};
use stark_model::geom::Vec2;
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

/// How long a change has to stop arriving before the miniature is drawn again, in
/// seconds. Long enough to collapse a burst — a held undo, a peer's actions landing —
/// short enough that a stroke's overview appears while the artist is still looking at
/// where it landed.
pub const SETTLE: f64 = 0.18;

/// The width of the viewport marker's outline, logical px.
const MARKER_WIDTH: f32 = 1.5;

/// Where the miniature sits in canvas space, and how large it is drawn.
///
/// All the shelf keeps: the picture itself lives on the GPU, in the surface behind it.
/// `Copy` and four numbers wide, so the view that re-renders on every engine write can
/// read it freely — where a readback path would keep a pixel buffer here and have to
/// be careful never to clone it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Overview {
    /// The canvas-space rect the miniature covers.
    pub min: Vec2,
    pub max: Vec2,
    /// Its size in **logical** px, which is the box the shelf lays out.
    pub width: f32,
    pub height: f32,
}

impl Overview {
    /// The overview a plan describes, at this display's scale.
    pub fn of(plan: &stark_engine::ExportPlan, scale: f32) -> Self {
        let scale = if scale > 0.0 { scale } else { 1.0 };
        Self {
            min: plan.min,
            max: plan.max,
            width: plan.size.width as f32 / scale,
            height: plan.size.height as f32 / scale,
        }
    }

    /// The box a plan is asked to fit, in the **device** px a plan is denominated in.
    pub fn box_for(scale: f32) -> Extent2 {
        let scale = if scale > 0.0 { scale } else { 1.0 };
        Extent2::new(
            ((MAX_WIDTH * scale).round() as u32).max(1),
            ((MAX_HEIGHT * scale).round() as u32).max(1),
        )
    }

    /// Where a press at fraction `(fx, fy)` of the miniature points, in canvas space.
    pub fn target(self, fx: f32, fy: f32) -> Vec2 {
        self.min + (self.max - self.min) * Vec2::new(fx.clamp(0.0, 1.0), fy.clamp(0.0, 1.0))
    }
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

/// The viewport marker's four corners in the miniature's own box, as fractions of it.
///
/// The miniature is always **upright** — it is a picture of the *piece*, and an
/// overview that turned with the easel would answer "where am I?" with a moving frame
/// of reference. So the turn shows in the marker instead: the viewport is a
/// screen-aligned rectangle, which in canvas space is a rotated one, and the corners
/// are that rectangle mapped back into the picture.
///
/// Not clamped: the rect is placed where it truly falls and the box clips it, so
/// panning off the piece shows the marker sliding out of frame rather than sticking to
/// an edge and claiming you are still on the painting.
pub fn marker(over: Overview, view: ViewTransform) -> [(f32, f32); 4] {
    let span = (over.max - over.min).max(Vec2::splat(1e-3));
    let half = Vec2::new(view.viewport.width as f32, view.viewport.height as f32) * 0.5;
    // Through the view's own screen→canvas mapping (`ViewTransform::canvas_delta`)
    // rather than an inverse worked out here: the zoom, the turn and the mirror are
    // all in it, and a second spelling of that map is the one thing an overview must
    // not have — it would put the marker somewhere the pointer does not agree with.
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(sx, sy)| {
        let corner = view.center + view.canvas_delta(half * Vec2::new(sx, sy));
        let f = (corner - over.min) / span;
        (f.x, f.y)
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
    fn piece() -> Overview {
        Overview {
            min: Vec2::new(-200.0, -100.0),
            max: Vec2::new(200.0, 100.0),
            width: 200.0,
            height: 100.0,
        }
    }

    /// The marker's own centre, as a fraction of the miniature.
    fn centre(corners: [(f32, f32); 4]) -> (f32, f32) {
        let n = corners.len() as f32;
        (
            corners.iter().map(|c| c.0).sum::<f32>() / n,
            corners.iter().map(|c| c.1).sum::<f32>() / n,
        )
    }

    /// Its width, as a fraction — the distance between two corners that share an edge.
    fn width(corners: [(f32, f32); 4]) -> f32 {
        let (ax, ay) = corners[0];
        let (bx, by) = corners[1];
        ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
    }

    /// The marker sits where the view is centred — so a view looking at the middle of
    /// the piece puts it in the middle of the miniature.
    #[test]
    fn the_marker_sits_where_the_view_is_centred() {
        let view = ViewTransform::identity(Extent2::new(200, 100));
        let (cx, cy) = centre(marker(piece(), view));
        assert!((cx - 0.5).abs() < 1e-3, "{cx}");
        assert!((cy - 0.5).abs() < 1e-3, "{cy}");
        // 200 screen px at zoom 1 over a 400 px piece is half of it.
        assert!((width(marker(piece(), view)) - 0.5).abs() < 1e-3);
    }

    /// Zooming in shrinks the marker, because the marker is how much of the piece the
    /// window covers — the one thing an overview is for.
    #[test]
    fn zooming_in_shrinks_the_marker() {
        let mut view = ViewTransform::identity(Extent2::new(200, 100));
        let wide = width(marker(piece(), view));
        view.zoom_about(Vec2::ZERO, 2.0);
        let close = width(marker(piece(), view));
        assert!((close - wide * 0.5).abs() < 1e-3, "2x should halve it");
    }

    /// Panning off the piece takes the marker **out of the frame** rather than pinning
    /// it to an edge — pinned because clamping is the obvious-looking edit and it would
    /// have the overview claim you are still on the painting.
    #[test]
    fn panning_off_the_piece_takes_the_marker_with_it() {
        let mut view = ViewTransform::identity(Extent2::new(200, 100));
        view.center_on(Vec2::new(4_000.0, 0.0));
        assert!(centre(marker(piece(), view)).0 > 1.0);
    }

    /// A degenerate overview — a frame dragged to nothing — divides by a floor rather
    /// than by zero, so the marker is a rectangle and not four NaNs the painter drops.
    #[test]
    fn a_collapsed_piece_still_yields_numbers() {
        let flat = Overview {
            min: Vec2::ZERO,
            max: Vec2::ZERO,
            ..piece()
        };
        let view = ViewTransform::identity(Extent2::new(200, 100));
        for (x, y) in marker(flat, view) {
            assert!(x.is_finite() && y.is_finite(), "{x},{y}");
        }
    }

    /// A press points at the canvas position under it, and the two ends of the box are
    /// the two ends of the piece.
    #[test]
    fn a_press_points_where_it_landed() {
        let over = piece();
        assert_eq!(over.target(0.0, 0.0), over.min);
        assert_eq!(over.target(1.0, 1.0), over.max);
        assert_eq!(over.target(0.5, 0.5), Vec2::ZERO);
        // Past the edge is the edge: a drag that has left the box keeps moving the
        // view it took hold of rather than flinging it.
        assert_eq!(over.target(-3.0, 9.0), Vec2::new(over.min.x, over.max.y));
    }

    /// The box a plan is fitted into is the shelf's own, in the device px a plan is
    /// denominated in — and never zero, which is not a texture.
    #[test]
    fn the_fitting_box_is_the_shelfs_own() {
        let at_1x = Overview::box_for(1.0);
        assert_eq!(at_1x.width, MAX_WIDTH.round() as u32);
        assert_eq!(at_1x.height, MAX_HEIGHT.round() as u32);
        let at_2x = Overview::box_for(2.0);
        assert_eq!(at_2x.width, at_1x.width * 2);
        assert!(Overview::box_for(0.0).width >= 1);
    }
}
