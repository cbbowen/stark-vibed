//! The two screen-denominated lengths a gesture has to declare, the map from a knob
//! to each (§6.2, §6.11), and the report a *hovering* move owes the engine
//! (§18.1.10).
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
use stark_engine::command::{HoverReport, InputSample};

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

/// How far ahead of the cursor the hover mark reaches, in **canvas px** (§18.1.10).
///
/// Canvas rather than screen px by nature: the mark is a hypothesis about *paint*,
/// and a screen-fixed length would promise more painting the further the view zoomed
/// out. The size circle over it already scales with the zoom, so the two halves of
/// the brush cursor shrink and grow together. The smoothing does not ride it — the
/// heading's estimator window is tolerance-relative inside the engine.
const HOVER_REACH_CANVAS_PX: f32 = 8.0;

/// What a frontend knows about a hovering move that the engine cannot: which presses
/// this one would *not* be paint (§18.1.10).
///
/// The engine gates the rest itself — a selection tool folds no mark, an unpaintable
/// layer refuses the render, and a real gesture outranks the hypothesis — so what is
/// left here is the chrome's own arming, which no engine state records. A struct
/// rather than a run of arguments because they are four spellings of one question,
/// and a caller transposing two bools would take the mark down for the wrong reason.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Hovering {
    /// Space is down, so the press pans (§18.1.7).
    pub panning: bool,
    /// A held chord arms an act that shadows the brush — the eyedropper, the layer
    /// carry ([`drags::armed`](crate::drags::armed) answered by
    /// [`DragAction::shadows_paint`](crate::drags::DragAction::shadows_paint)). Both
    /// read the *shown* canvas back, so the hypothesis must be off it before a press
    /// can read one: the wrong color for the sample, the wrong layer for the carry.
    pub shadowed: bool,
    /// The eyedropper is already down (§18.0.2).
    pub sampling: bool,
    /// The timeline is playing, where a stroke is refused outright.
    pub playing: bool,
}

impl Hovering {
    /// The report a hovering move owes `ViewCommand::PreviewHover`, or `None` where
    /// the press is promised elsewhere — which is the answer that takes a standing
    /// mark **down** rather than merely declining to renew it.
    ///
    /// The sample's pressure is replaced with **full pressure**: a hovering pen, and
    /// a mouse, reports zero, which would honestly preview no mark at all. Full
    /// rather than a middle weight so the mark fills the size circle drawn over it —
    /// two overlays about one brush must not disagree about its reach. Tilt is kept,
    /// since a hovering pen reports it and the mark should lean as the stroke would.
    ///
    /// `tolerance` is the frontend's statement of its input tolerance, exactly as
    /// [`tolerance`] gives it to a gesture — restated per report, because the zoom it
    /// derives from can change mid-hover.
    pub fn report(self, sample: InputSample, tolerance: f32) -> Option<HoverReport> {
        if self.panning || self.shadowed || self.sampling || self.playing {
            return None;
        }
        Some(HoverReport {
            sample: InputSample {
                pressure: 1.0,
                ..sample
            },
            tolerance,
            reach: HOVER_REACH_CANVAS_PX,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stark_engine::Extent2;
    use stark_model::geom::Vec2;

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

    /// A hovering report carries no weight at all — a pen out of contact reports zero
    /// and so does a mouse — so without the substitution the mark would honestly
    /// preview nothing (§18.1.10).
    #[test]
    fn the_mark_is_previewed_at_full_pressure() {
        let hand = Hovering::default();
        let mut sample = InputSample::at(Vec2::new(3.0, 4.0));
        sample.pressure = 0.0;
        let report = hand.report(sample, 1.0).expect("a free hand lays the mark");
        assert_eq!(report.sample.pressure, 1.0);
    }

    /// Everything else about the report is the hand's own: the mark leans with a
    /// hovering pen's tilt, and stands where the pointer is rather than a reach away
    /// — carrying the heading forward is the engine's half (`Session::hover_to`).
    #[test]
    fn the_hands_own_channels_reach_the_engine() {
        let mut sample = InputSample::at(Vec2::new(3.0, 4.0));
        sample.tilt = Vec2::new(0.25, -0.5);
        sample.time = 12.5;
        let report = Hovering::default()
            .report(sample, 0.25)
            .expect("a free hand lays the mark");
        assert_eq!(report.sample.pos, sample.pos);
        assert_eq!(report.sample.tilt, sample.tilt);
        assert_eq!(report.sample.time, sample.time);
        assert_eq!(report.tolerance, 0.25);
    }

    /// Each of the four states the chrome has promised the press to refuses the
    /// report, which is also what takes a standing mark down. Measured against the
    /// free hand above, so none of them can pass by the door being shut on
    /// everything.
    #[test]
    fn a_press_promised_elsewhere_lays_no_mark() {
        let sample = InputSample::at(Vec2::ZERO);
        let promised = [
            Hovering {
                panning: true,
                ..Default::default()
            },
            Hovering {
                shadowed: true,
                ..Default::default()
            },
            Hovering {
                sampling: true,
                ..Default::default()
            },
            Hovering {
                playing: true,
                ..Default::default()
            },
        ];
        for hand in promised {
            assert!(hand.report(sample, 1.0).is_none(), "{hand:?} laid a mark");
        }
    }
}
