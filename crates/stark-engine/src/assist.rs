//! Drag-and-hold drawing assist (§6.9).
//!
//! Drag out a rough line or a rough ellipse and then **hold the pen down without
//! moving it**: the stroke in flight snaps to the ideal shape it resembles, and the
//! rest of the same drag steers that shape rather than extending the path. Lift to
//! commit, exactly as for any other stroke.
//!
//! The feature attaches at the seam §18.1.3 names — *a path transform applied between
//! the fitter and the renderer* — so it answers in control points rather than carrying
//! a shape into the action log. Nothing downstream of
//! [`StrokeRecord::path`](stark_model::document::StrokeRecord) learns that assist
//! exists, and an assisted stroke is undoable, replayable and collaborative for free.
//!
//! Three pieces, one per file:
//!
//! - **Recognition** ([`recognize`](mod@recognize)) — which shape, if any, the raw
//!   pointer trace is. It declines freely, and deterministically: a false positive (a
//!   considered stroke silently replaced) costs far more than a miss. A line that lands
//!   near an axis of a perspective guide the artist has up is turned onto it (§20.6).
//! - **Adjustment** ([`AssistShape::adjust`]) — what the rest of the drag means. Derived
//!   from the shape *as recognized* plus the pointer's travel since, never from the
//!   previous frame's shape, so a long adjustment cannot accumulate drift (§16.6).
//! - **Realization** ([`AssistShape::to_path`]) — the shape as a fitted path, carrying
//!   the pen channels the stroke was actually drawn with. That a snapped line keeps its
//!   pressure swell is the whole difference between this and a shape tool.
//!
//! Every bar the recognizer measures against is denominated in the gesture's **input
//! tolerance** — the device's own tolerance in canvas px, which the frontend already
//! states for the fit (see [`PathFitter::with_tolerance`](crate::path::PathFitter::with_tolerance)).
//! A bar fixed in canvas px would mean two different things at two zoom levels.

mod adjust;
mod realize;
mod recognize;

pub use realize::PenProfile;
pub use recognize::recognize;

use stark_model::document::AxisPlane;
use stark_model::geom::Vec2;

/// The ideal shape a held gesture snapped to (§6.9).
///
/// Held in the session for the rest of the gesture and thrown away at the end: what is
/// committed is the path it produces, not the shape itself. Keeping it would start to
/// pay for post-hoc stroke editing (§18.2.1), at the price of a wire-format change.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AssistShape {
    /// A straight segment from `a` to `b`. `b` is the end the stroke finished at —
    /// the end under the pointer, and so the end a drag moves.
    Line {
        a: Vec2,
        b: Vec2,
        /// Whether the direction is a perspective guide's rather than the hand's
        /// (§20.6). A guided line is the same segment held to a direction it may not
        /// leave: the rest of the drag runs **along** it instead of steering it off.
        on_axis: bool,
    },
    /// An ellipse, drawn as one closed loop.
    Ellipse {
        center: Vec2,
        /// Semi-axes along the ellipse's own frame, major first.
        radii: Vec2,
        /// Rotation of that frame, radians clockwise in canvas space.
        angle: f32,
        /// The perspective plane this is a **circle** on, if it is one (§20.7).
        ///
        /// The ellipse above stays the truth — it is what is drawn, and every
        /// consumer below reads it without knowing a guide exists. This says only
        /// which family the shape came out of and may not leave, which is all
        /// [`adjust`](Self::adjust) needs to go on steering it a circle.
        plane: Option<AxisPlane>,
        /// Where on the ellipse the loop starts and ends, in the ellipse's **own**
        /// frame — so turning the shape carries the seam with it, and the join stays
        /// where the hand put it rather than sliding round under a rotation.
        phase: f32,
        /// Which way round the loop travels: `+1` or `-1` on [`phase`](Self::Ellipse::phase).
        /// Taken from the drawn trace so the pen channels ride the shape the way they
        /// were laid down.
        winding: f32,
    },
}

/// Which family a held gesture snapped to (§6.9) — as much of [`AssistShape`] as
/// anything outside the engine needs.
///
/// A second, smaller enum rather than exporting the shape itself: publishing the
/// geometry would fix the assist's internals as an interface for the sake of a question
/// answered by one bit. The caller outside the crate is the guided tour (§24.2), which
/// counts an assisted stroke and tells a line from a loop.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Assisted {
    /// A straight segment.
    Line,
    /// A closed loop.
    Ellipse,
}

impl AssistShape {
    /// A point of the shape's own parameterization, scaled about the centre by `bulge`
    /// (1.0 for the shape itself). For a line the parameter is the fraction from `a`
    /// to `b`; for an ellipse it is radians past the seam, signed by the winding.
    ///
    /// `bulge` is realization's alone — it undoes a B-spline's inset — and adjustment
    /// passes 1.0, which is the shape itself.
    fn at(&self, t: f32, bulge: f32) -> Vec2 {
        match *self {
            Self::Line { a, b, .. } => a.lerp(b, t),
            Self::Ellipse {
                center,
                radii,
                angle,
                phase,
                winding,
                ..
            } => {
                let u = phase + winding * t;
                let local = Vec2::new(radii.x * u.cos(), radii.y * u.sin()) * bulge;
                center + Vec2::from_angle(angle).rotate(local)
            }
        }
    }

    /// Which family this is.
    pub fn family(&self) -> Assisted {
        match self {
            AssistShape::Line { .. } => Assisted::Line,
            AssistShape::Ellipse { .. } => Assisted::Ellipse,
        }
    }
}
