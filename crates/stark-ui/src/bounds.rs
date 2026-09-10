//! The canvas-space rectangles a frontend asks the document for, and the one way it
//! grows them.
//!
//! What they have in common is that each answers *where on the canvas* with no
//! reference to any surface: they read `ObservableState` and return canvas px. A
//! frontend that wanted them in its own units would convert at the edge, which is
//! `PointerReport`'s rule read the other way (§11.2).
//!
//! Two are not rectangles and are here for the same reason all the same.
//! [`piece_frame`] is a *reading of the layer roster* rather than a rect, and
//! [`Overview`] carries the size its rect is drawn at — but both answer a question
//! about the document that no surface is party to, which is the line this module is
//! actually drawn on.
//!
//! They collected here because two different features want the same fallback ladder.
//! Framing a piece (§15.7) and mounting the transform widget (§16.6) both have to
//! answer "which rectangle, when the obvious one is missing" — the selection's hull,
//! or the paint's, or failing both what is on screen — and an answer given twice is
//! two answers one edit apart.
//!
//! The navigator's [`Overview`] joined them for the same reason one rung down. It is
//! a canvas rectangle plus the size it is drawn at, and [`marker`] is where the
//! viewport falls inside it — which both frontends worked out for themselves, by two
//! different routes, while the native one's own doc said that "a second spelling of
//! that map is the one thing an overview must not have".

use stark_engine::{ObservableState, ViewTransform};
use stark_model::geom::Vec2;

/// The painted content's canvas-space bounds, inset to the populated tiles.
///
/// `None` for a canvas nobody has painted on: there is no content to bound, which is
/// a different answer from an empty rectangle and the callers below treat it as one.
pub fn content(o: &ObservableState) -> Option<(Vec2, Vec2)> {
    let (min, max) = o.bounds.tile_range()?;
    let t = stark_model::geom::TILE_SIZE as f32;
    Some((
        Vec2::new(min.x as f32 * t, min.y as f32 * t),
        Vec2::new((max.x + 1) as f32 * t, (max.y + 1) as f32 * t),
    ))
}

/// What the viewport currently shows, in canvas px, inset a little.
///
/// The inset is so a rectangle *made* from what is on screen — a frame, a transform
/// widget — reads as a thing on the canvas rather than as flush with the window edge.
///
/// Under a turned canvas the bound covers a little more than the window really shows,
/// which is the right way round: "frame what I am looking at" should not clip the
/// corners off it.
pub fn view(o: &ObservableState) -> (Vec2, Vec2) {
    let (min, max) = o.view.visible_bounds();
    let inset = (max - min) * VIEW_INSET;
    (min + inset, max - inset)
}

/// How much of the visible bound [`view`] gives back, per side.
const VIEW_INSET: f32 = 0.06;

/// The axis-aligned bound of `points`, or `None` if there are none.
///
/// Here rather than folded where it is wanted: the fold is a `min` and a `max` that
/// have to agree, and a second copy of it is a second chance for one of them to be
/// the other. `transform::image_rect` had the copy.
pub fn aabb(points: impl IntoIterator<Item = Vec2>) -> Option<(Vec2, Vec2)> {
    let mut points = points.into_iter();
    let first = points.next()?;
    Some(points.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
}

/// `rect`, grown symmetrically wherever an axis is thinner than `min`.
///
/// A hairline rectangle has corners a hand cannot tell apart, so anything that mounts
/// grabbable handles on one has to widen it first — and how much is a *screen*-px
/// figure the caller divides by the zoom, which is why `min` arrives already in
/// canvas px rather than being read from the view here.
///
/// **A backwards axis is normalized, not refused**: it comes out exactly `min` wide
/// about its own midpoint, the same answer a zero-width one gets. `TransformState::begin`
/// used to hold a second and contradictory answer — it zeroed the extent, mounting the
/// smallest widget on a rectangle this one would have kept whole — and now holds none,
/// because the radius it takes is a length and a length has no direction to be wrong.
pub fn inflate(rect: (Vec2, Vec2), min: f32) -> (Vec2, Vec2) {
    let (mut lo, mut hi) = rect;
    for axis in 0..2 {
        let (a, b) = (lo[axis], hi[axis]);
        if b - a < min {
            let pad = (min - (b - a)) * 0.5;
            lo[axis] = a - pad;
            hi[axis] = b + pad;
        }
    }
    (lo, hi)
}

/// **The frame that says where the piece ends**: the topmost matte layer with a
/// rect, or `None` in a document with none (§15.6).
///
/// Only a matte *with a rect* frames anything — a backing (§15.5) is under the piece,
/// not a statement of where it stops — and the topmost of them wins, on the same
/// reading that puts the newest work on top.
///
/// One rule with several askers, because they are all asking the same question and an
/// answer that differed between them would put a file, its miniature and the view onto
/// three different rects: the export dialog when nothing framing is selected, either
/// navigator's overview, and the framing a document load does. Each still supplies its
/// own *policy* around it — the dialog prefers whatever frame the artist has selected,
/// since that is the one being composed.
///
/// Here rather than in a frontend because the second one needed it and neither owns
/// it: this is a reading of the layer roster, which is the engine's projection, and
/// what it answers is a fact about the document rather than about a screen.
pub fn piece_frame(o: &ObservableState) -> Option<stark_model::document::LayerId> {
    o.layers
        .iter()
        .rev()
        .find(|l| l.matte.as_ref().is_some_and(|m| m.rect.is_some()))
        .map(|l| l.id)
}

/// How long a change has to stop arriving before a miniature is drawn again, in
/// seconds.
///
/// Long enough to collapse a burst — a held undo, a peer's actions landing, the several
/// commits a fill-then-recolour makes — short enough that a stroke's overview appears
/// while the artist is still looking at where it landed.
pub const SETTLE: f64 = 0.18;

/// Where a miniature of the piece sits in canvas space, and how large it is drawn.
///
/// All a navigator keeps: the picture itself lives on the GPU, in the surface behind
/// it. `Copy` and four numbers wide, so a view that re-renders on every engine write
/// can read it freely — where a readback path would keep a pixel buffer here and have
/// to be careful never to clone it.
///
/// The size is in whatever px the frontend lays out in — logical for a docked column,
/// CSS for a surface presented 1:1 — because nothing here divides by it except
/// [`Self::target`], which takes a fraction.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Overview {
    /// The canvas-space rect the miniature covers.
    pub min: Vec2,
    /// The far corner of it.
    pub max: Vec2,
    /// Its drawn width.
    pub width: f32,
    /// Its drawn height.
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

    /// The canvas rect's extent, floored off zero.
    ///
    /// A frame dragged to nothing is a real state, and every reader here divides by
    /// this — so the floor is the type's rather than each caller's to remember.
    pub fn span(self) -> Vec2 {
        (self.max - self.min).max(Vec2::splat(1e-3))
    }

    /// Where a press at fraction `(fx, fy)` of the miniature points, in canvas space.
    pub fn target(self, fx: f32, fy: f32) -> Vec2 {
        self.min + (self.max - self.min) * Vec2::new(fx.clamp(0.0, 1.0), fy.clamp(0.0, 1.0))
    }
}

/// The viewport marker's four corners inside `over`, as fractions of it, in the order
/// top-left, top-right, bottom-right, bottom-left of the **screen**.
///
/// A miniature is always upright — it is a picture of the *piece*, and an overview that
/// turned with the easel would answer "where am I?" with a moving frame of reference.
/// So the turn shows in the marker instead: the viewport is a screen-aligned rectangle,
/// which in canvas space is a rotated one, and these are its corners mapped back into
/// the picture.
///
/// Taken through the view's own screen→canvas mapping ([`ViewTransform::canvas_delta`])
/// rather than an inverse worked out here: the zoom, the turn and the mirror are all in
/// it, and a second spelling of that map would put the marker somewhere the pointer does
/// not agree with. Both frontends had one until this was the only one.
///
/// **Not clamped**: the rect is placed where it truly falls and the box clips it, so
/// panning off the piece shows the marker sliding out of frame rather than sticking to
/// an edge and claiming you are still on the painting.
pub fn marker(over: Overview, view: ViewTransform) -> [(f32, f32); 4] {
    let span = over.span();
    let half = Vec2::new(view.viewport.width as f32, view.viewport.height as f32) * 0.5;
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(sx, sy)| {
        let corner = view.center + view.canvas_delta(half * Vec2::new(sx, sy));
        let f = (corner - over.min) / span;
        (f.x, f.y)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_engine::Extent2;

    /// A thin axis is grown about its own centre, so the rectangle does not walk
    /// while it is being made grabbable.
    #[test]
    fn inflating_a_thin_axis_keeps_its_centre() {
        let (lo, hi) = inflate((Vec2::new(10.0, 0.0), Vec2::new(10.0, 40.0)), 20.0);
        assert_eq!(lo, Vec2::new(0.0, 0.0));
        assert_eq!(hi, Vec2::new(20.0, 40.0));
    }

    /// An axis already wide enough is left exactly alone — inflating is a floor, not
    /// a resize, or every entry into a transform would nudge the paint's bounds.
    #[test]
    fn inflating_leaves_a_wide_axis_alone() {
        let rect = (Vec2::new(0.0, 0.0), Vec2::new(100.0, 80.0));
        assert_eq!(inflate(rect, 20.0), rect);
    }

    /// A rectangle given back to front comes out the right way round, `min` wide
    /// about the midpoint it named — the same answer a flat one gets, and the answer
    /// everything mounting handles on a rectangle is entitled to assume.
    #[test]
    fn inflating_normalizes_a_backwards_axis() {
        let (lo, hi) = inflate((Vec2::new(30.0, 0.0), Vec2::new(10.0, 40.0)), 20.0);
        assert_eq!(lo, Vec2::new(10.0, 0.0));
        assert_eq!(hi, Vec2::new(30.0, 40.0));
    }

    /// The bound is the points' own, and no points is not an empty rectangle at the
    /// origin — the callers treat the two differently.
    #[test]
    fn an_aabb_covers_its_points_and_nothing_covers_none() {
        assert_eq!(aabb(std::iter::empty()), None);
        let got = aabb([
            Vec2::new(3.0, -1.0),
            Vec2::new(-2.0, 5.0),
            Vec2::new(0.0, 0.0),
        ]);
        assert_eq!(got, Some((Vec2::new(-2.0, -1.0), Vec2::new(3.0, 5.0))));
    }

    fn piece() -> Overview {
        Overview {
            min: Vec2::splat(-200.0),
            max: Vec2::splat(200.0),
            width: 200.0,
            height: 200.0,
        }
    }

    /// A view looking at the middle of the piece puts the marker in the middle of the
    /// miniature, and one covering half the piece covers half the miniature.
    #[test]
    fn the_marker_sits_where_the_view_is_centred() {
        let view = ViewTransform::identity(Extent2::new(200, 200));
        let corners = marker(piece(), view);
        let mid = corners
            .iter()
            .fold((0.0, 0.0), |a, c| (a.0 + c.0, a.1 + c.1));
        assert!((mid.0 / 4.0 - 0.5).abs() < 1e-4, "{corners:?}");
        assert!((mid.1 / 4.0 - 0.5).abs() < 1e-4, "{corners:?}");
        // 200 screen px at zoom 1 over a 400 px piece is half of it.
        assert!(
            (corners[1].0 - corners[0].0 - 0.5).abs() < 1e-4,
            "{corners:?}"
        );
    }

    /// Zooming in shrinks the marker — the one thing an overview is for.
    #[test]
    fn zooming_in_shrinks_the_marker() {
        let mut view = ViewTransform::identity(Extent2::new(200, 200));
        let wide = marker(piece(), view);
        view.zoom_about(Vec2::ZERO, 2.0);
        let close = marker(piece(), view);
        let w = |c: [(f32, f32); 4]| c[1].0 - c[0].0;
        assert!(
            (w(close) - w(wide) * 0.5).abs() < 1e-4,
            "2x should halve it"
        );
    }

    /// Panning off the piece slides the marker out of frame rather than pinning it to
    /// an edge and claiming you are still on the painting.
    #[test]
    fn panning_off_the_piece_takes_the_marker_with_it() {
        let mut view = ViewTransform::identity(Extent2::new(200, 200));
        view.center_on(Vec2::new(4_000.0, 0.0));
        let corners = marker(piece(), view);
        assert!(corners.iter().all(|c| c.0 > 1.0), "{corners:?}");
    }

    /// A frame dragged to nothing divides by a floor rather than by zero, so what comes
    /// back is still numbers and not a run of NaNs a frontend silently drops.
    #[test]
    fn a_collapsed_piece_still_yields_numbers() {
        let flat = Overview {
            min: Vec2::ZERO,
            max: Vec2::ZERO,
            width: 200.0,
            height: 100.0,
        };
        let view = ViewTransform::identity(Extent2::new(200, 100));
        let corners = marker(flat, view);
        assert!(corners.iter().all(|c| c.0.is_finite() && c.1.is_finite()));
    }

    /// A press at a fraction of the miniature points at the matching fraction of the
    /// piece, and a press outside it is held to the edge — a click cannot ask for a
    /// centre off the picture.
    #[test]
    fn a_press_points_at_the_fraction_it_lands_on() {
        assert_eq!(piece().target(0.5, 0.5), Vec2::ZERO);
        assert_eq!(piece().target(0.0, 0.0), Vec2::splat(-200.0));
        assert_eq!(piece().target(2.0, -1.0), Vec2::new(200.0, -200.0));
    }
}
