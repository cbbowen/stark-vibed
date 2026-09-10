//! The frame bar's presets (§15.7): the aspects a frame is reshaped to, and the paint a
//! new matte arrives in.
//!
//! The bar, the on-canvas handles and the acts that add a matte are each frontend's.
//! What is here is what two bars would otherwise spell twice — and a preset list that
//! differed between them would read one frame as 16:9 in one app and "Custom" in the
//! other.

use stark_model::geom::Vec2;

/// A new frame's fill: a near-black mat board. Dark reads as "not the piece" against
/// almost any painting, which is what a crop scrim is for.
pub const DEFAULT_MATTE: [f32; 3] = [0.06, 0.06, 0.07];

/// A new backing's fill: a warm paper tone. A backing is *under* the painting, so it
/// defaults to something to paint over rather than a scrim (§15.5).
pub const DEFAULT_BACKING: [f32; 3] = [0.93, 0.91, 0.86];

/// Aspect presets, as width:height.
pub const ASPECTS: [(&str, f32); 4] = [
    ("1:1", 1.0),
    ("4:5", 0.8),
    ("3:2", 1.5),
    ("16:9", 16.0 / 9.0),
];

/// What an aspect control shows when the frame matches no preset — a real state, since
/// a dragged handle lands on an arbitrary ratio and the control should say so rather
/// than name the nearest preset.
pub const CUSTOM: &str = "Custom";

/// The preset a frame of `(width, height)` matches, or [`CUSTOM`].
///
/// The tolerance is relative, so it holds at any size, and loose enough that a handle
/// dragged to visually 16:9 reads as 16:9 rather than flicking to "Custom" on a
/// sub-pixel difference.
pub fn matched_aspect((w, h): (f32, f32)) -> &'static str {
    if h.abs() < 1e-3 {
        return CUSTOM;
    }
    let ratio = w / h;
    ASPECTS
        .iter()
        .find(|(_, a)| (ratio - a).abs() <= a * 0.005)
        .map_or(CUSTOM, |(label, _)| label)
}

/// The rect `min..max` reshaped to `aspect` about its centre, keeping its area — so
/// switching presets neither grows nor shrinks the piece.
pub fn to_aspect(min: Vec2, max: Vec2, aspect: f32) -> (Vec2, Vec2) {
    let center = (min + max) * 0.5;
    let area = ((max.x - min.x) * (max.y - min.y)).max(1.0);
    let h = (area / aspect).sqrt();
    let half = Vec2::new(aspect * h, h) * 0.5;
    (center - half, center + half)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reshaping to a preset **preserves area** (§15.7), which is what makes the
    /// drop-down a way to change the shape of a piece rather than its size — flick
    /// through 1:1, 4:5, 16:9 and back and the frame is the one you started with, not
    /// a sliver.
    #[test]
    fn reshaping_keeps_the_area() {
        let (min, max) = (Vec2::new(-160.0, -90.0), Vec2::new(160.0, 90.0));
        let area = |lo: Vec2, hi: Vec2| (hi.x - lo.x) * (hi.y - lo.y);
        let was = area(min, max);
        for (_, aspect) in ASPECTS {
            let (lo, hi) = to_aspect(min, max, aspect);
            assert!(
                (area(lo, hi) - was).abs() < was * 1e-4,
                "{aspect} changed the area from {was} to {}",
                area(lo, hi)
            );
        }
    }

    /// …and reshapes **about the centre**, so the piece does not walk across the
    /// canvas as the artist tries ratios.
    #[test]
    fn reshaping_holds_the_centre() {
        let (min, max) = (Vec2::new(40.0, -10.0), Vec2::new(200.0, 70.0));
        let center = (min + max) * 0.5;
        for (_, aspect) in ASPECTS {
            let (lo, hi) = to_aspect(min, max, aspect);
            let moved = (lo + hi) * 0.5;
            assert!(
                (moved - center).length() < 1e-3,
                "{aspect} moved the centre from {center:?} to {moved:?}"
            );
        }
    }

    /// And the shape it lands on is the one asked for — the round trip through
    /// [`matched_aspect`], which is what the drop-down reads back.
    #[test]
    fn a_reshaped_frame_reads_as_the_preset_it_was_given() {
        let (min, max) = (Vec2::new(-100.0, -100.0), Vec2::new(100.0, 100.0));
        for (label, aspect) in ASPECTS {
            let (lo, hi) = to_aspect(min, max, aspect);
            assert_eq!(matched_aspect((hi.x - lo.x, hi.y - lo.y)), label);
        }
    }

    /// A ratio that is nobody's preset says so rather than snapping to the nearest —
    /// the whole point of `Custom` being a real state (§15.7), since a dragged handle
    /// lands wherever the hand left it.
    #[test]
    fn an_arbitrary_ratio_is_custom() {
        assert_eq!(matched_aspect((100.0, 73.0)), CUSTOM);
        // Just outside the relative tolerance on either side of 1:1.
        assert_eq!(matched_aspect((1.0, 1.0)), "1:1");
        assert_eq!(matched_aspect((1.02, 1.0)), CUSTOM);
        // The tolerance is relative, so a preset holds at any size.
        assert_eq!(matched_aspect((16_000.0, 9_000.0)), "16:9");
        assert_eq!(matched_aspect((0.016, 0.009)), "16:9");
    }

    /// A degenerate frame divides by nothing: a zero-height rect is `Custom`, not a
    /// NaN ratio that matches whichever preset the comparison happens to answer for.
    #[test]
    fn a_flat_frame_has_no_preset() {
        assert_eq!(matched_aspect((100.0, 0.0)), CUSTOM);
        assert_eq!(matched_aspect((0.0, 0.0)), CUSTOM);
    }
}
