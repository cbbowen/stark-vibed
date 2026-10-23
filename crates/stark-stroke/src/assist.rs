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
//! - **Recognition** (`recognize.rs`) — which shape, if any, the raw
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
use stark_model::geom::{Ellipse, Vec2};

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
        /// What is drawn, and all any consumer below reads.
        shape: Ellipse,
        /// The perspective plane `shape` is a **circle** on, if it is one (§20.7): the
        /// family the shape came out of and may not leave, which is all
        /// [`adjust`](Self::adjust) needs to go on steering it a circle.
        plane: Option<AxisPlane>,
        /// Where on the ellipse the loop starts and ends, in the ellipse's **own**
        /// frame — so turning the shape carries the seam with it, and the join stays
        /// where the hand put it rather than sliding round under a rotation.
        phase: f32,
        /// Which way round the loop travels, taken from the drawn trace so the pen
        /// channels ride the shape the way they were laid down.
        winding: Winding,
    },
}

/// Which way round a loop travels on its ellipse's own parameter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Winding {
    /// The parameter increases along the loop.
    Forward,
    /// The parameter decreases along the loop.
    Backward,
}

impl Winding {
    /// The ellipse parameter `t` radians along the loop from `phase`.
    fn advance(self, phase: f32, t: f32) -> f32 {
        match self {
            Self::Forward => phase + t,
            Self::Backward => phase - t,
        }
    }
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
    /// A point of the shape's own parameterization: for a line the fraction from `a` to
    /// `b`, for an ellipse radians past the seam along the winding.
    fn at(&self, t: f32) -> Vec2 {
        match *self {
            Self::Line { a, b, .. } => a.lerp(b, t),
            Self::Ellipse {
                shape,
                phase,
                winding,
                ..
            } => Frame::of(shape).point(winding.advance(phase, t)),
        }
    }

    /// Which family this is.
    pub fn family(&self) -> Assisted {
        match self {
            AssistShape::Line { .. } => Assisted::Line,
            AssistShape::Ellipse { .. } => Assisted::Ellipse,
        }
    }

    /// The shape sampled along its own parameterization, from the seam — how two shapes
    /// are compared, since a circle's angle and seam trade off against each other.
    #[cfg(test)]
    fn outline(&self) -> Vec<Vec2> {
        let reach = match self {
            Self::Line { .. } => 1.0,
            Self::Ellipse { .. } => std::f32::consts::TAU,
        };
        (0..=32).map(|i| self.at(reach * i as f32 / 32.0)).collect()
    }
}

/// An ellipse's own frame: the one place a point of an ellipse is computed, and what
/// turns a canvas point back into the parameter it sits at.
struct Frame {
    center: Vec2,
    radii: Vec2,
    /// The frame's rotation and its inverse, ready to apply.
    rotate: Vec2,
    unrotate: Vec2,
}

impl Frame {
    fn of(e: Ellipse) -> Self {
        Self {
            center: e.center,
            radii: e.radii,
            rotate: Vec2::from_angle(e.angle),
            unrotate: Vec2::from_angle(-e.angle),
        }
    }

    /// The point of this ellipse at parameter `t`, scaled about the centre by `bulge` —
    /// realization's, which undoes a B-spline's inset (`realize`).
    fn point_scaled(&self, t: f32, bulge: f32) -> Vec2 {
        let local = Vec2::new(self.radii.x * t.cos(), self.radii.y * t.sin()) * bulge;
        self.center + self.rotate.rotate(local)
    }

    /// The point of this ellipse at parameter `t`.
    fn point(&self, t: f32) -> Vec2 {
        self.point_scaled(t, 1.0)
    }

    /// `p` in the ellipse's own axes.
    fn local(&self, p: Vec2) -> Vec2 {
        self.unrotate.rotate(p - self.center)
    }

    /// The ellipse parameter `p` sits at, in `(-π, π]`.
    fn param(&self, p: Vec2) -> f32 {
        let q = self.local(p);
        (q.y / self.radii.y.max(1e-6)).atan2(q.x / self.radii.x.max(1e-6))
    }

    /// Distance from `p` to the ellipse.
    ///
    /// The algebraic residual divided by its own gradient — the first-order correction,
    /// exact in the limit of a point on the curve. Measuring *radially* over-reads by up
    /// to the axis ratio at the ends of the major axis, which would mean a 2:1 ellipse
    /// had to be drawn twice as accurately as a circle to snap at all.
    fn distance(&self, p: Vec2) -> f32 {
        let q = self.local(p);
        let (a2, b2) = (
            (self.radii.x * self.radii.x).max(1e-6),
            (self.radii.y * self.radii.y).max(1e-6),
        );
        let residual = q.x * q.x / a2 + q.y * q.y / b2 - 1.0;
        let grad = 2.0 * Vec2::new(q.x / a2, q.y / b2).length();
        if grad < 1e-9 {
            self.radii.min_element()
        } else {
            residual.abs() / grad
        }
    }

    /// The furthest any point of `trace` sits from the curve, by
    /// [`distance`](Self::distance) — what a fit is scored on.
    fn worst(&self, trace: &[Vec2]) -> f32 {
        trace.iter().map(|p| self.distance(*p)).fold(0.0, f32::max)
    }
}
