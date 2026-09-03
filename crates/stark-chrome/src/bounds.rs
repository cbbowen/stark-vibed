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

/// `rect`, grown symmetrically wherever an axis is thinner than `min`.
///
/// A hairline rectangle has corners a hand cannot tell apart, so anything that mounts
/// grabbable handles on one has to widen it first — and how much is a *screen*-px
/// figure the caller divides by the zoom, which is why `min` arrives already in
/// canvas px rather than being read from the view here.
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
}
