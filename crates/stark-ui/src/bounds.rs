//! The canvas-space rectangles a frontend asks the document for, and the one way it
//! grows them.
//!
//! Three functions, and what they have in common is that each answers *where on the
//! canvas* with no reference to any surface: they read `ObservableState` and return
//! canvas px. A frontend that wanted them in its own units would convert at the edge,
//! which is `PointerReport`'s rule read the other way (§11.2).
//!
//! They collected here because two different features want the same fallback ladder.
//! Framing a piece (§15.7) and mounting the transform widget (§16.6) both have to
//! answer "which rectangle, when the obvious one is missing" — the selection's hull,
//! or the paint's, or failing both what is on screen — and an answer given twice is
//! two answers one edit apart.

use stark_engine::ObservableState;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
