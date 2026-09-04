//! The two screen-denominated lengths a gesture has to declare, and the map from a
//! knob to each (§6.2, §6.11).
//!
//! `GestureCommand::Start` asks the frontend for a **tolerance** and a **rope**, and
//! is right to: both are canvas-space lengths derived from *screen* quantities, and
//! only a frontend holds the view that converts one to the other. What it does not
//! hold is a reason for the conversion to differ between frontends — so this is where
//! the conversion lives, and the frontend supplies only what it alone can know.
//!
//! **This module is the first thing `stark-ui` was built to prevent.** The native
//! frontend was one commit old and already carried its own `ROPE_MAX_SCREEN_PX = 160`
//! and its own copy of the quadratic map, because the web one's was unreachable — so
//! the same brush at the same smoothing was towed by two constants nothing held
//! together (§11.2).

use stark_engine::ViewTransform;

/// The longest smoothing string a brush can ask for, in **screen px** — what
/// `smoothing = 1` means.
///
/// Screen px because wobble is a fact about the hand: the same tremor spans 64× more
/// canvas zoomed out than in.
const ROPE_MAX_SCREEN_PX: f32 = 160.0;

/// The §6.11 rope a smoothing amount means against `view`, in canvas px: the `0..=1`
/// knob mapped **quadratically** to a screen-px string — so the low end is
/// fine-grained while the top is a real lettering tow — then carried through the view.
///
/// Zooming in therefore shrinks the dead zone in canvas terms: the escape hatch from
/// heavy smoothing is the one artists already reach for to do fine work.
///
/// Stated against an explicit view because more than two ask: each frontend's canvas,
/// and the web brush editor's preview against its own.
pub fn rope(view: ViewTransform, amount: f32) -> f32 {
    let a = amount.clamp(0.0, 1.0);
    a * a * ROPE_MAX_SCREEN_PX / view.zoom
}

/// The fitting tolerance to declare for a gesture, in canvas px: the device's own
/// resolution carried through `view`, since canvas space is where the fit measures
/// its error.
///
/// `resolution` is in the units the frontend's own surface is denominated in — CSS px
/// on the web canvas, device px on the native one (§11.1) — because that is the space
/// `ViewTransform` maps out of. Which is also why it is a parameter: what a *device*
/// resolves to is the one half of this only a frontend can answer, and the two
/// disagree about the unit before they disagree about the number.
pub fn tolerance(view: ViewTransform, resolution: f32) -> f32 {
    resolution / view.zoom
}

/// What a **mouse** resolves position to, in the screen units of whatever surface it
/// is over: it walks the screen in whole physical pixels, so one is its floor.
///
/// A pen or a finger comes off a digitizer that resolves well below the screen it
/// sits under, so what limits those is the hand rather than the API — see
/// [`PEN_RESOLUTION`]. Not a preference either way: an estimate of the device, which
/// is what the fitter needs in order to tell jitter from detail.
pub const MOUSE_RESOLUTION: f32 = 1.0;

/// What a **pen or finger** resolves to, in physical px — a deliberate
/// under-estimate. Too fine only costs a few extra control points, while too coarse
/// rounds off detail that was really there.
pub const PEN_RESOLUTION: f32 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;
    use stark_model::geom::Extent2;

    fn view(zoom: f32) -> ViewTransform {
        let mut v = ViewTransform::identity(Extent2::new(800, 600));
        v.zoom = zoom;
        v
    }

    /// Smoothing off is no tow at all — the raw samples reach the fitter exactly as
    /// they did before there was a knob (§6.11).
    #[test]
    fn no_smoothing_is_no_rope() {
        assert_eq!(rope(view(1.0), 0.0), 0.0);
    }

    /// The map is quadratic, so the bottom of the knob is fine-grained: half the
    /// slider is a quarter of the string, not half of it.
    #[test]
    fn the_knob_is_quadratic() {
        let full = rope(view(1.0), 1.0);
        assert!((rope(view(1.0), 0.5) - full / 4.0).abs() < 1e-5);
    }

    /// Both lengths are screen quantities divided by the zoom, which is what makes
    /// them mean the same thing to the hand at any magnification — and what makes
    /// zooming in the way out of heavy smoothing.
    #[test]
    fn both_shrink_in_canvas_terms_as_the_view_zooms_in() {
        assert!(rope(view(4.0), 1.0) < rope(view(1.0), 1.0));
        assert!(tolerance(view(4.0), MOUSE_RESOLUTION) < tolerance(view(1.0), MOUSE_RESOLUTION));
        // And exactly proportionally, which is the whole of the conversion.
        assert!((tolerance(view(4.0), 1.0) - 0.25).abs() < 1e-6);
    }

    /// A knob past its range is clamped rather than extrapolated: the map is only
    /// defined on `0..=1`, and a preset from a build that allowed more must not tow
    /// a stroke off the canvas.
    #[test]
    fn a_knob_out_of_range_is_clamped() {
        assert_eq!(rope(view(1.0), 2.0), rope(view(1.0), 1.0));
        assert_eq!(rope(view(1.0), -1.0), 0.0);
    }
}
